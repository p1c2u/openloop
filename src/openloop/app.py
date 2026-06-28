"""FastAPI application — the runtime's HTTP entrypoint.

Loads agents from config-as-code, wires the first agent that exposes a Slack
surface to the Slack events endpoint, sets up channel memory, and exposes a
health check. Run with:

    uvicorn openloop.app:app --reload
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from openloop.agents import load_agents
from openloop.agents.schema import Agent
from openloop.approvals import ApprovalStore, InMemoryApprovalStore
from openloop.approvals.postgres import PostgresApprovalStore
from openloop.checkpoints import CheckpointStore, InMemoryCheckpointStore
from openloop.checkpoints.postgres import PostgresCheckpointStore
from openloop.config import Settings, get_settings
from openloop.coordination import (
    DistributedLock,
    InProcessLock,
    PostgresLock,
    RedisLock,
    guard,
)
from openloop.memory import Embedder, InMemoryStore, LiteLLMEmbedder, MemoryStore
from openloop.memory.postgres import PostgresMemoryStore
from openloop.runtime import Runtime
from openloop.sessions import (
    InMemorySurfaceSessionStore,
    SurfaceSessionStore,
)
from openloop.sessions.postgres import PostgresSurfaceSessionStore
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

from openloop.surfaces.slack import build_slack_app
from openloop.tools import Invocation, ToolGateway
from openloop.tools.coding_worker import CodingWorkerConnector, GitCodingWorker
from openloop.tools.github import GitHubConnector, HttpGitHubClient
from openloop.tools.mcp import HttpMCPClient, MCPConnector
from openloop.usage import InMemoryUsageStore, UsageStore, budget_scope_key
from openloop.usage.postgres import PostgresUsageStore
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine, WorkflowStore
from openloop.workflows.coding_worker import build_coding_worker_workflow
from openloop.workflows.postgres import PostgresWorkflowStore

log = logging.getLogger("openloop")


def build_embedder(settings: Settings) -> Embedder | None:
    """Build an embedder only if enabled and its provider key is configured."""
    if not settings.embeddings_enabled:
        return None
    if settings.embedding_provider not in settings.configured_providers:
        log.warning(
            "embeddings disabled: no API key for provider %r — "
            "memory will use recency-only recall",
            settings.embedding_provider,
        )
        return None
    return LiteLLMEmbedder(settings.embedding_model)


def build_memory_store(settings: Settings) -> MemoryStore:
    """Pick a memory backend. Postgres setup happens at startup."""
    if settings.memory_backend == "postgres":
        return PostgresMemoryStore(
            settings.database_url, embedding_dim=settings.embedding_dim
        )
    return InMemoryStore()


def build_usage_store(settings: Settings) -> UsageStore:
    """Pick a usage/audit backend. Postgres setup happens at startup."""
    if settings.memory_backend == "postgres":
        return PostgresUsageStore(settings.database_url)
    return InMemoryUsageStore()


def build_approval_store(settings: Settings) -> ApprovalStore:
    """Pick an approval backend. Postgres setup happens at startup."""
    if settings.memory_backend == "postgres":
        return PostgresApprovalStore(settings.database_url)
    return InMemoryApprovalStore()


def build_checkpoint_store(settings: Settings) -> CheckpointStore:
    """Pick a worker-checkpoint backend. Postgres setup happens at startup."""
    if settings.memory_backend == "postgres":
        return PostgresCheckpointStore(settings.database_url)
    return InMemoryCheckpointStore()


def build_workflow_store(settings: Settings) -> WorkflowStore:
    """Pick a workflow-instance backend. Postgres setup happens at startup."""
    if settings.memory_backend == "postgres":
        return PostgresWorkflowStore(settings.database_url)
    return InMemoryWorkflowStore()


def build_surface_session_store(settings: Settings) -> SurfaceSessionStore:
    """Pick a surface-session backend (Phase D). Postgres setup at startup."""
    if settings.memory_backend == "postgres":
        return PostgresSurfaceSessionStore(settings.database_url)
    return InMemorySurfaceSessionStore()


def _resolve_lock_backend(settings: Settings) -> str:
    """Resolve ``lock_backend``, expanding ``auto`` to follow ``memory_backend``."""
    backend = settings.lock_backend
    if backend == "auto":
        # A Postgres deploy already has the shared dependency advisory locks need,
        # so a multi-replica Postgres deploy gets coordination without extra infra;
        # otherwise there's nothing to coordinate against → process-local.
        return "postgres" if settings.memory_backend == "postgres" else "memory"
    return backend


def build_lock(settings: Settings) -> DistributedLock:
    """Pick a coordination backend; its store/connection is set up at startup.

    ``auto`` (default) follows ``memory_backend``. ``postgres`` reuses the existing
    database; ``redis`` needs the optional ``redis`` extra (a missing package
    degrades to in-process here, a connectivity failure degrades at startup).
    """
    backend = _resolve_lock_backend(settings)
    if backend == "postgres":
        return PostgresLock(settings.database_url)
    if backend == "redis":
        try:
            return RedisLock.from_url(settings.redis_url)
        except Exception:
            log.exception(
                "redis lock unavailable (is the `redis` extra installed?) — "
                "falling back to in-process coordination"
            )
    return InProcessLock()


_COORD_LABEL = {RedisLock: "redis", PostgresLock: "postgres"}


async def _setup_coordination(
    coordinator: DistributedLock, settings: Settings
) -> DistributedLock:
    """Set up the coordination backend, degrading to in-process on failure.

    A backend the operator *explicitly* asked for (``lock_backend`` postgres/redis)
    is a deliberate request for cross-process coordination, so a setup failure is
    logged loudly — silently running process-local locks across replicas is the
    footgun this feature exists to remove. An ``auto``-selected backend degrades
    quietly (consistent with the other stores' "degrade, don't fail boot" posture).
    """
    setup = getattr(coordinator, "setup", None)
    if setup is None:  # InProcessLock — nothing to start
        log.info("coordination backend: in-process (single-replica)")
        return coordinator
    try:
        await setup()
        log.info(
            "coordination backend: %s",
            _COORD_LABEL.get(type(coordinator), type(coordinator).__name__),
        )
        return coordinator
    except Exception:
        if settings.lock_backend in ("postgres", "redis"):
            log.error(
                "CROSS-PROCESS COORDINATION DISABLED: LOCK_BACKEND=%s could not "
                "start; multiple replicas may run recovery concurrently. Falling "
                "back to a process-local lock.",
                settings.lock_backend, exc_info=True,
            )
        else:
            log.exception(
                "coordination backend setup failed — falling back to in-process"
            )
        await _safe_close(coordinator)
        return InProcessLock()


def build_tool_gateway(
    settings: Settings,
    agents: dict[str, Agent],
    approvals: ApprovalStore,
    checkpoints: CheckpointStore,
    engine: WorkflowEngine,
) -> ToolGateway:
    """Register native connectors plus an MCP connector per configured server.

    MCP connectors need an async setup() (tool discovery); the returned list is
    set up in the app lifespan.
    """
    gateway = ToolGateway(approvals=approvals, engine=engine)
    if settings.github_token:
        github_client = HttpGitHubClient(settings.github_token)
        gateway.register(GitHubConnector(github_client))
        log.info("registered native tool: github")
        # The coding worker runs model-generated edits, so it stays off unless
        # explicitly enabled (it needs a contents:write token + a sandbox).
        if settings.coding_worker_enabled:
            worker = GitCodingWorker(
                settings.github_token, model=settings.coding_worker_model
            )
            gateway.register(
                CodingWorkerConnector(worker, github_client, checkpoints=checkpoints)
            )
            # Register the worker as a durable workflow (approval = wait node).
            engine.register(build_coding_worker_workflow(worker, github_client))
            log.info(
                "registered native tool: coding_worker (model=%s)",
                settings.coding_worker_model,
            )
        else:
            log.info(
                "coding_worker tool not registered: set CODING_WORKER_ENABLED=1"
            )
    else:
        log.warning("github tool not registered: GITHUB_TOKEN unset")

    mcp_connectors: list[MCPConnector] = []
    seen: set[str] = set()
    for agent in agents.values():
        for tool in agent.spec.tools:
            if tool.type == "mcp" and tool.server and tool.name not in seen:
                connector = MCPConnector(tool.name, HttpMCPClient(tool.server))
                gateway.register(connector)
                mcp_connectors.append(connector)
                seen.add(tool.name)
                log.info("registered MCP tool %r -> %s", tool.name, tool.server)
    gateway.mcp_connectors = mcp_connectors  # type: ignore[attr-defined]
    return gateway


class InvokeBody(BaseModel):
    action: str
    args: dict = {}
    requested_by: str | None = None


class ResolveBody(BaseModel):
    approver: str
    approve: bool = True


def _invocation_json(inv: Invocation) -> dict:
    return {
        "status": inv.status,
        "message": inv.message,
        "result": dataclasses.asdict(inv.result) if inv.result else None,
        "approval_id": inv.approval.id if inv.approval else None,
    }


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    agents = load_agents(settings.agents_dir)
    log.info("loaded %d agent(s): %s", len(agents), ", ".join(agents) or "none")

    embedder = build_embedder(settings)
    store = build_memory_store(settings)
    usage = build_usage_store(settings)
    approvals = build_approval_store(settings)
    checkpoints = build_checkpoint_store(settings)
    workflows = build_workflow_store(settings)
    engine = WorkflowEngine(workflows)
    sessions = build_surface_session_store(settings)
    # Cross-process lock: lets one replica lead startup recovery. Rebound to an
    # in-process lock in the lifespan if a configured Redis can't be reached.
    coordinator = build_lock(settings)
    # The Slack SessionRunner captures the session store by reference; the lifespan
    # needs a handle to it to repoint after a Postgres fallback. Set in the Slack
    # block below (stays None when no Slack surface is bound).
    session_runner = None
    tools = build_tool_gateway(settings, agents, approvals, checkpoints, engine)
    # The agent that tool/approval endpoints act on (first configured).
    primary_agent: Agent | None = next(iter(agents.values()), None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal coordinator
        recovery_task: asyncio.Task | None = None
        if isinstance(store, PostgresMemoryStore):
            try:
                await store.setup()
                log.info("memory backend: postgres (pgvector)")
            except Exception:
                log.exception(
                    "postgres memory setup failed — falling back to in-memory"
                )
                app.state.memory = InMemoryStore()
                _rebind(app, "memory", app.state.memory)
        else:
            log.info("memory backend: in-memory (process-local)")

        if isinstance(usage, PostgresUsageStore):
            try:
                await usage.setup()
                log.info("usage backend: postgres")
            except Exception:
                log.exception(
                    "postgres usage setup failed — falling back to in-memory"
                )
                app.state.usage = InMemoryUsageStore()
                _rebind(app, "usage", app.state.usage)
        else:
            log.info("usage backend: in-memory (process-local)")

        if isinstance(approvals, PostgresApprovalStore):
            try:
                await approvals.setup()
                log.info("approval backend: postgres")
            except Exception:
                log.exception(
                    "postgres approval setup failed — falling back to in-memory"
                )
                tools.approvals = InMemoryApprovalStore()
        else:
            log.info("approval backend: in-memory (process-local)")

        if isinstance(checkpoints, PostgresCheckpointStore):
            try:
                await checkpoints.setup()
                log.info("checkpoint backend: postgres")
            except Exception:
                log.exception(
                    "postgres checkpoint setup failed — worker resume disabled"
                )
                _disable_checkpoints(tools)
        else:
            log.info("checkpoint backend: in-memory (process-local)")

        if isinstance(workflows, PostgresWorkflowStore):
            try:
                await workflows.setup()
                log.info("workflow backend: postgres")
            except Exception:
                log.exception(
                    "postgres workflow setup failed — falling back to in-memory"
                )
                # Swap the shared engine's store so BOTH the gateway and the
                # already-constructed runtime keep working (process-local, not
                # crash-durable) rather than hitting an un-setup pool.
                engine.store = InMemoryWorkflowStore()
        else:
            log.info("workflow backend: in-memory (process-local)")

        if isinstance(sessions, PostgresSurfaceSessionStore):
            try:
                await sessions.setup()
                log.info("surface-session backend: postgres")
            except Exception:
                log.exception(
                    "postgres surface-session setup failed — falling back "
                    "to in-memory"
                )
                # Repoint BOTH the app state and the already-built Slack runner
                # (which captured the Postgres store) at one shared in-memory
                # fallback, so mentions don't hit an un-setup pool in the
                # background — mirrors the workflow engine's store swap above.
                fallback_sessions = InMemorySurfaceSessionStore()
                app.state.sessions = fallback_sessions
                if session_runner is not None:
                    session_runner.sessions = fallback_sessions
        else:
            log.info("surface-session backend: in-memory (process-local)")

        coordinator = await _setup_coordination(coordinator, settings)
        app.state.coordinator = coordinator  # the resolved lock (post-fallback)

        # Recover work left incomplete by a crash, as the recovery *leader* so that
        # when several replicas boot together only one runs the (idempotent but
        # wasteful-to-duplicate) sweeps. Run once now (before serving), then keep
        # re-running on an interval: a leader that crashes mid-sweep — whose Redis
        # lock then lingers for its whole TTL, even past its own restart — is
        # covered when a surviving replica's next pass acquires the lapsed lock.
        # Without the periodic retry a one-shot skip would strand that work until
        # an unrelated restart.
        await run_recovery_pass(coordinator, tools, session_runner)
        interval = settings.recovery_interval_seconds
        if interval > 0:
            recovery_task = asyncio.create_task(
                _recovery_loop(coordinator, tools, session_runner, interval=interval)
            )

        for connector in getattr(tools, "mcp_connectors", []):
            try:
                await connector.setup()
            except Exception:
                log.exception(
                    "MCP connector %r setup failed — its tools stay unavailable",
                    connector.name,
                )

        yield

        if recovery_task is not None:
            recovery_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recovery_task
        if isinstance(store, PostgresMemoryStore):
            await store.close()
        if isinstance(usage, PostgresUsageStore):
            await usage.close()
        if isinstance(approvals, PostgresApprovalStore):
            await approvals.close()
        if isinstance(checkpoints, PostgresCheckpointStore):
            await checkpoints.close()
        if isinstance(workflows, PostgresWorkflowStore):
            await workflows.close()
        if isinstance(sessions, PostgresSurfaceSessionStore):
            await sessions.close()
        if hasattr(coordinator, "close"):  # RedisLock / PostgresLock (not in-process)
            await _safe_close(coordinator)

    app = FastAPI(title="OpenLoop", version="0.0.1", lifespan=lifespan)
    app.state.settings = settings
    app.state.agents = agents
    app.state.memory = store
    app.state.usage = usage
    app.state.tools = tools
    app.state.sessions = sessions
    # app.state.coordinator is set in the lifespan, after Redis connectivity is
    # checked, so it reflects the resolved lock (post any fallback).
    app.state.primary_agent = primary_agent

    # Bind the first Slack-enabled agent. The Bolt app is built when a bot
    # token exists (Socket Mode reuses it); the HTTP events route is added only
    # when a signing secret is present to verify requests.
    app.state.slack_app = None
    slack_agent = next(
        (a for a in agents.values() if a.has_slack_surface()), None
    )
    if slack_agent and settings.slack_bot_token:
        runtime = Runtime(
            slack_agent, memory=store, embedder=embedder, usage=usage,
            tools=tools, engine=engine,
        )
        app.state.runtime = runtime
        slack_app = build_slack_app(
            runtime,
            sessions,
            bot_token=settings.slack_bot_token,
            signing_secret=settings.slack_signing_secret or None,
        )
        app.state.slack_app = slack_app
        # Captured so the session-store fallback can repoint the runner (above).
        session_runner = getattr(slack_app, "_session_runner", None)
        app.state.session_runner = session_runner

        if settings.slack_signing_secret:
            slack_handler = AsyncSlackRequestHandler(slack_app)

            @app.post("/slack/events")
            async def slack_events(req: Request):  # type: ignore[no-untyped-def]
                return await slack_handler.handle(req)

            log.info("Slack HTTP events bound to agent %r", slack_agent.metadata.name)
        else:
            log.info(
                "Slack app built for %r (Socket Mode); HTTP events disabled "
                "without SLACK_SIGNING_SECRET",
                slack_agent.metadata.name,
            )
    else:
        log.warning(
            "Slack surface not bound: need a Slack-enabled agent and "
            "SLACK_BOT_TOKEN"
        )

    @app.post("/tools/invoke")
    async def invoke_tool(body: InvokeBody, request: Request):  # type: ignore[no-untyped-def]
        """Run a tool action through the gateway (allowlist + approval gate)."""
        inv = await request.app.state.tools.invoke(
            _require_primary(request),
            body.action,
            body.args,
            requested_by=body.requested_by,
        )
        return _invocation_json(inv)

    @app.get("/approvals")
    async def list_approvals(request: Request):  # type: ignore[no-untyped-def]
        agent = _require_primary(request)
        pending = await request.app.state.tools.approvals.pending(
            agent=agent.metadata.name
        )
        return [
            {"id": r.id, "action": r.action, "summary": r.summary,
             "approvers": r.approvers, "requested_by": r.requested_by}
            for r in pending
        ]

    @app.post("/approvals/{request_id}/resolve")
    async def resolve_approval(request_id: str, body: ResolveBody, request: Request):  # type: ignore[no-untyped-def]
        inv = await request.app.state.tools.resolve(
            request_id, body.approver, approve=body.approve
        )
        if inv.status == "forbidden":
            raise HTTPException(403, inv.message)
        return _invocation_json(inv)

    @app.get("/usage")
    async def usage_summary(request: Request):  # type: ignore[no-untyped-def]
        """Month-to-date spend vs. budget for the primary agent."""
        agent = _require_primary(request)
        store: UsageStore = request.app.state.usage
        spent = await store.monthly_total(budget_scope_key(agent))
        budget = agent.spec.budget
        return {
            "agent": agent.metadata.name,
            "workspace": agent.metadata.workspace,
            "month_to_date_usd": round(spent, 6),
            "monthly_budget_usd": budget.monthly_usd,
            "per_task_budget_usd": budget.per_task_usd,
            "on_exceeded": budget.on_exceeded,
        }

    @app.get("/audit")
    async def audit(request: Request, limit: int = 50):  # type: ignore[no-untyped-def]
        """Recent usage records — the audit trail."""
        store: UsageStore = request.app.state.usage
        records = await store.recent(limit=min(limit, 500))
        return [
            {
                "agent": r.agent,
                "channel": r.channel,
                "surface": r.surface,
                "user": r.user,
                "task_kind": r.task_kind,
                "model": r.model,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "cost_usd": r.cost_usd,
                "outcome": r.outcome,
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]

    @app.get("/healthz")
    async def healthz():  # type: ignore[no-untyped-def]
        actions = tools.available_actions(primary_agent) if primary_agent else []
        return {
            "status": "ok",
            "agents": list(agents),
            "providers": settings.configured_providers,
            "memory": type(app.state.memory).__name__,
            "usage": type(app.state.usage).__name__,
            "tools": actions,
        }

    return app


def _require_primary(request: Request) -> Agent:
    agent = getattr(request.app.state, "primary_agent", None)
    if agent is None:
        raise HTTPException(404, "no agents configured")
    return agent


def _rebind(app: FastAPI, attr: str, value) -> None:
    """Point the live runtime at a fallback store after a setup failure."""
    runtime = getattr(app.state, "runtime", None)
    if runtime is not None:
        setattr(runtime, attr, value)


def _disable_checkpoints(tools: ToolGateway) -> None:
    """Fall back to process-local checkpoints if the durable store can't start.

    The worker still runs and stays idempotent within a process, but jobs no
    longer resume across a restart.
    """
    worker = tools._tools.get("coding_worker")
    if worker is not None:
        worker.checkpoints = InMemoryCheckpointStore()


# Lease for the recovery lock. Short on purpose — a crashed leader's lock frees
# within this window — but renewed every _RECOVERY_LOCK_RENEW_SECONDS while a pass
# runs, so a live leader keeps the lock no matter how long the sweep takes (the
# worker resume can re-drive model generation + git push + PR open, well past the
# raw TTL). This decouples "self-heal latency" from "max sweep duration", so two
# replicas can't both acquire and re-drive the same jobs concurrently.
_RECOVERY_LOCK_TTL_SECONDS = 60.0
_RECOVERY_LOCK_RENEW_SECONDS = 20.0


async def run_recovery_pass(coordinator, tools: ToolGateway, session_runner) -> bool:
    """Guarded sweep of the crash-recovery reconcilers; returns whether we led it.

    Acquires the shared ``startup-recovery`` lock so that across replicas only one
    sweeps at a time; a contended replica returns ``False`` without doing the work
    (the leader's shared-store reconcile covers it). Idempotent and safe to repeat,
    which is what lets the periodic loop heal a leader that crashed mid-sweep once
    its lock TTL lapses. The lock is coordination, not correctness — a TTL expiry
    or in-process fallback at worst re-does idempotent work.
    """
    async with guard(
        coordinator,
        "startup-recovery",
        ttl_seconds=_RECOVERY_LOCK_TTL_SECONDS,
        renew_interval=_RECOVERY_LOCK_RENEW_SECONDS,
    ) as is_leader:
        if not is_leader:
            log.debug("another replica is leading recovery — skipping this pass")
            return False
        # The workflow engine re-drives any instance left "running"; the connector
        # reconciler covers the Phase B (no-engine) checkpoint path. resolve() won't
        # re-invoke an approved request, so these reconcilers trigger the resume.
        engine = getattr(tools, "engine", None)
        if engine is not None:
            try:
                resumed = await engine.resume_incomplete()
                if resumed:
                    log.info("resumed %d incomplete workflow(s)", len(resumed))
            except Exception:
                log.exception("workflow resume failed")
        await _resume_worker_jobs(tools)

        # Repair surface-session delivery left mid-flight by a crash — after the
        # engine resume above so each session's workflow is already terminal.
        if session_runner is not None:
            try:
                repaired = await session_runner.reconcile()
                if repaired:
                    log.info("reconciled %d surface session(s)", len(repaired))
            except Exception:
                log.exception("surface-session reconcile failed")
        return True


async def _recovery_loop(
    coordinator, tools: ToolGateway, session_runner, *, interval: float
) -> None:
    """Re-run :func:`run_recovery_pass` every ``interval`` seconds until cancelled.

    The self-healing backstop for a leader that died mid-sweep: a surviving
    replica's next pass acquires the lapsed lock and finishes the recovery.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            await run_recovery_pass(coordinator, tools, session_runner)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("periodic recovery pass failed")


async def _safe_close(closeable) -> None:
    """Best-effort close of a coordination client; never raise from teardown."""
    try:
        await closeable.close()
    except Exception:
        log.warning("failed to close coordination client", exc_info=True)


async def _resume_worker_jobs(tools: ToolGateway) -> None:
    """Drive the coding worker's startup reconciler, if it is registered."""
    worker = tools._tools.get("coding_worker")
    if worker is None or not hasattr(worker, "resume_incomplete"):
        return
    try:
        resumed = await worker.resume_incomplete()
        if resumed:
            log.info("resumed %d incomplete coding-worker job(s)", len(resumed))
    except Exception:
        log.exception("coding-worker job resume failed")


app = create_app()
