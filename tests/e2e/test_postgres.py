"""Happy-path end-to-end test against a real Postgres (pgvector).

Validates what the unit tests can't: the actual SQL, asyncpg type handling,
pgvector distance search, and approval persistence. The model, embedder, and
GitHub client are faked (no external credentials), but every store is real.

Runs only when a Postgres is reachable — set OPENLOOP_TEST_DATABASE_URL, or it
falls back to the docker-compose default. Skips cleanly otherwise so the normal
suite stays green without Docker.
"""

import os
import uuid

import pytest

from openloop.agents import load_agent
from openloop.approvals.postgres import PostgresApprovalStore
from openloop.memory.postgres import PostgresMemoryStore
from openloop.memory.store import MemoryRecord, scope_key_for
from openloop.runtime import Runtime, Task
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.usage import budget_scope_key
from openloop.usage.postgres import PostgresUsageStore
from openloop.testing import (
    EXAMPLE_AGENT,
    FakeEmbedder,
    FakeGitHub,
    ScriptedGateway,
    tool_call_response,
)

DSN = os.environ.get(
    "OPENLOOP_TEST_DATABASE_URL",
    "postgresql://openloop:change-me@localhost:5432/openloop_agents",
)

# 26-dim to match FakeEmbedder (the real default is 1536; dim is configurable).
EMBED_DIM = 26

pytestmark = [pytest.mark.e2e, pytest.mark.postgres]


async def _reachable() -> bool:
    try:
        import asyncpg

        conn = await asyncpg.connect(DSN, timeout=3)
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def stores():
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")
    # Unique table-free isolation isn't possible (shared tables), so scope keys
    # are made unique per run instead.
    memory = PostgresMemoryStore(DSN, embedding_dim=EMBED_DIM)
    usage = PostgresUsageStore(DSN)
    approvals = PostgresApprovalStore(DSN)
    await memory.setup()
    await usage.setup()
    await approvals.setup()
    try:
        yield memory, usage, approvals
    finally:
        await memory.close()
        await usage.close()
        await approvals.close()


async def test_happy_path_end_to_end(stores):
    memory, usage, approvals = stores
    agent = load_agent(EXAMPLE_AGENT)
    run_id = uuid.uuid4().hex[:8]
    channel = f"#e2e-{run_id}"  # unique scope so the run is isolated
    scope = scope_key_for(agent, channel)

    # Seed a prior decision into channel memory (real pgvector insert).
    embedder = FakeEmbedder()
    seed_vec = (await embedder.embed(["Use Redis Streams for ingestion v1."]))[0]
    await memory.remember(MemoryRecord(
        scope_key=scope, text="Use Redis Streams for ingestion v1.",
        embedding=seed_vec))

    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)], approvals=approvals)

    # The model recalls context, then asks to open a GitHub issue (write action).
    gateway = ScriptedGateway([
        tool_call_response("m", [("c1", "github_issues_write",
                                  {"repo": "acme/ingestion",
                                   "title": "Track: Redis Streams for v1"})]),
    ])
    runtime = Runtime(agent, gateway=gateway, memory=memory, embedder=embedder,
                      usage=usage, tools=tools)

    # --- the turn: write action is held for approval ---
    result = await runtime.handle(Task(
        text="open an issue to track the ingestion decision",
        surface="slack", channel=channel, user="U_requester"))

    assert result.model == "approval-gate"
    assert len(result.approval_ids) == 1
    approval_id = result.approval_ids[0]

    # Recall worked against pgvector: the seeded memory reached the model.
    system_text = " ".join(
        m["content"] for m in gateway.calls[0]["messages"]
        if m["role"] == "system")
    assert "Redis Streams" in system_text

    # The approval is persisted as pending in Postgres.
    pending = await approvals.pending(agent="dev-platform")
    assert any(p.id == approval_id for p in pending)
    assert github.created == []  # nothing executed yet

    # --- a human approves; the action executes and persists ---
    inv = await tools.resolve(approval_id, "@priya", approve=True)
    assert inv.status == "executed"
    assert github.created  # the issue was created on approval

    stored = await approvals.get(approval_id)
    assert stored.status == "approved"
    assert stored.decided_by == "@priya"

    # Usage was recorded to the real audit trail, and the turn was remembered.
    spent_records = await usage.recent(limit=200)
    assert any(r.channel == channel for r in spent_records)
    assert await usage.monthly_total(budget_scope_key(agent)) >= 0.0

    recalled = await memory.recall(scope, seed_vec, limit=5)
    texts = [r.text for r in recalled]
    assert "Use Redis Streams for ingestion v1." in texts
    # The requester's message was remembered this turn.
    assert any("open an issue to track" in t for t in texts)


async def test_worker_checkpoint_resume_across_real_postgres():
    """A worker job persisted to Postgres resumes on a fresh store instance."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.checkpoints.postgres import PostgresCheckpointStore
    from openloop.tools.coding_worker import (
        STEPS,
        CodingWorkerConnector,
        WorkerOutcome,
    )

    job_id = f"e2e-{uuid.uuid4().hex[:8]}"

    class _Worker:
        def __init__(self):
            self.runs = 0

        async def run(self, state, on_step=None):
            self.runs += 1
            for step in STEPS:
                state.completed_steps.append(step)
                if on_step is not None:
                    await on_step(state)
            state.title, state.body = "t", "b"
            return WorkerOutcome(branch=state.branch, title="t", body="b")

    class _FlakyGitHub(FakeGitHub):
        def __init__(self):
            super().__init__()
            self.fail_next = True

        async def create_pull(self, *a, **k):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("blip")
            return await super().create_pull(*a, **k)

    store = PostgresCheckpointStore(DSN)
    await store.setup()
    try:
        args = {"repo": "acme/x", "instruction": "do x", "job_id": job_id}

        # First store/worker: pushes, but the PR open fails — persisted to PG.
        worker1, github1 = _Worker(), _FlakyGitHub()
        conn1 = CodingWorkerConnector(worker1, github1, checkpoints=store)
        first = await conn1.execute("pr:write", args)
        assert not first.ok
        cp = await store.get(job_id)
        assert cp.status == "open_pr_failed" and "push" in cp.completed_steps

        # A *fresh* store + connector (simulating a restart) resumes from PG:
        # the worker is not re-run and exactly one PR is opened.
        store2 = PostgresCheckpointStore(DSN)
        await store2.setup()
        try:
            worker2, github2 = _Worker(), FakeGitHub()
            conn2 = CodingWorkerConnector(worker2, github2, checkpoints=store2)
            second = await conn2.execute("pr:write", args)
            assert second.ok
            assert worker2.runs == 0  # resumed past the push
            assert len(github2.pulls) == 1
            assert (await store2.get(job_id)).status == "opened"
        finally:
            await store2.close()
    finally:
        # Best-effort cleanup of this run's row.
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM worker_checkpoints WHERE job_id = $1", job_id
                )
        except Exception:
            pass
        await store.close()


async def test_workflow_resume_across_real_postgres():
    """A workflow parked at a wait node resumes from Postgres on a fresh engine."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.workflows import Step, Workflow, WorkflowEngine
    from openloop.workflows.postgres import PostgresWorkflowStore

    instance_id = f"wf-{uuid.uuid4().hex[:8]}"

    def _wf():
        async def finish(ctx):
            ctx.instance.result = {"ok": True}
            ctx.state["ran"] = True

        return Workflow("t", [Step("gate", wait=True), Step("finish", finish)])

    store = PostgresWorkflowStore(DSN)
    await store.setup()
    try:
        engine1 = WorkflowEngine(store, {"t": _wf()})
        parked = await engine1.start("t", instance_id, {"seed": 1})
        assert parked.status == "waiting" and parked.waiting_on == "gate"

        # Fresh store + engine (a restart) delivers the event and completes.
        store2 = PostgresWorkflowStore(DSN)
        await store2.setup()
        try:
            engine2 = WorkflowEngine(store2, {"t": _wf()})
            done = await engine2.send_event(instance_id, "gate", {"by": "x"})
            assert done.status == "completed"
            assert done.result == {"ok": True}
            assert done.state["ran"] is True
            assert done.state["seed"] == 1  # original state survived the restart
        finally:
            await store2.close()
    finally:
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM workflow_instances WHERE id = $1", instance_id
                )
        except Exception:
            pass
        await store.close()


async def test_surface_session_roundtrip_across_real_postgres():
    """Persist a surface session and look it up by event + approval id (Phase D)."""
    if not await _reachable():
        pytest.skip(f"no Postgres reachable at {DSN}")

    from openloop.sessions.postgres import PostgresSurfaceSessionStore
    from openloop.sessions.store import SurfaceSession, SurfaceTarget

    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    event_id = f"ev-{uuid.uuid4().hex[:8]}"
    approval_id = f"appr-{uuid.uuid4().hex[:8]}"

    store = PostgresSurfaceSessionStore(DSN)
    await store.setup()
    try:
        await store.upsert(SurfaceSession(
            id=session_id,
            target=SurfaceTarget(
                surface="slack", workspace="acme", agent="dev-platform",
                channel="C1", thread="100.1", event_id=event_id,
            ),
            status="waiting",
            workflow_instance_id=session_id,
            progress_message_id="ts-1",
            approval_ids=[approval_id],
        ))

        # A fresh store (a restart) reads it back by all three keys.
        store2 = PostgresSurfaceSessionStore(DSN)
        await store2.setup()
        try:
            by_id = await store2.get(session_id)
            assert by_id is not None and by_id.status == "waiting"
            assert by_id.target.thread == "100.1"
            assert by_id.approval_ids == [approval_id]
            assert (await store2.get_by_event(event_id)).id == session_id
            # The `@>` containment lookup (button → session) resolves the owner.
            assert (await store2.get_by_approval(approval_id)).id == session_id
            assert await store2.get_by_approval("nope") is None
        finally:
            await store2.close()
    finally:
        try:
            pool = store._require_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM surface_sessions WHERE id = $1", session_id
                )
        except Exception:
            pass
        await store.close()
