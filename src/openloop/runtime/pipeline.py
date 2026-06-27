"""The async task pipeline.

An inbound mention becomes a :class:`Task`. The runtime enforces budget,
recalls channel memory, resolves a model, then runs a tool-calling loop: the
model may call tools the agent is allowed; the gateway enforces the allowlist
and routes write actions through human approval; results feed back until the
model produces a final answer. Usage is recorded and the exchange remembered.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openloop.agents.schema import Agent
from openloop.memory import (
    Embedder,
    InMemoryStore,
    MemoryRecord,
    MemoryStore,
    scope_key_for,
)
from openloop.models.gateway import ModelGateway, ModelResponse
from openloop.tools import ToolGateway
from openloop.usage import (
    InMemoryUsageStore,
    UsageRecord,
    UsageStore,
    budget_scope_key,
    check_budget,
)

if TYPE_CHECKING:
    from openloop.workflows.engine import WorkflowContext, WorkflowEngine

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are {name}, a team AI agent operating in the {workspace} workspace. "
    "You are reachable across shared channels and act on behalf of the team. "
    "Be concise and helpful. When unsure, ask a clarifying question."
)

# How many memories to pull into context per task.
RECALL_LIMIT = 5
# Safety cap on model<->tool round-trips per task.
MAX_TOOL_ITERS = 4


def _tool_message(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


@dataclass(slots=True)
class Task:
    """A unit of work routed to an agent from some surface."""

    text: str
    surface: str
    channel: str | None = None
    user: str | None = None
    # Optional task class (e.g. "summarize", "code") used for model routing.
    kind: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)


class Runtime:
    """Routes a task to a model and produces a reply, with channel memory."""

    def __init__(
        self,
        agent: Agent,
        gateway: ModelGateway | None = None,
        memory: MemoryStore | None = None,
        embedder: Embedder | None = None,
        usage: UsageStore | None = None,
        tools: ToolGateway | None = None,
        engine: "WorkflowEngine | None" = None,
        *,
        remember: bool = True,
    ) -> None:
        self.agent = agent
        self.gateway = gateway or ModelGateway()
        self.memory = memory or InMemoryStore()
        self.embedder = embedder
        self.usage = usage or InMemoryUsageStore()
        self.tools = tools
        self.remember = remember
        # When an engine is wired, each task runs as a durable `agent_task`
        # workflow (consumer #2). Namespaced per agent so multiple agents on one
        # engine don't collide. Without an engine, handle() runs inline.
        self.engine = engine
        self.workflow_name = f"agent_task:{agent.metadata.name}"
        if engine is not None:
            engine.register(self._build_workflow())

    def _build_messages(
        self, task: Task, recalled: list[MemoryRecord]
    ) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT.format(
            name=self.agent.metadata.name,
            workspace=self.agent.metadata.workspace,
        )
        messages = [{"role": "system", "content": system}]
        if recalled:
            bullets = "\n".join(f"- {r.text}" for r in recalled)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant team memory for this channel "
                        "(most relevant first):\n" + bullets
                    ),
                }
            )
        messages.extend(task.history)
        messages.append({"role": "user", "content": task.text})
        return messages

    async def _embed(self, text: str) -> list[float] | None:
        if self.embedder is None:
            return None
        vectors = await self.embedder.embed([text])
        return vectors[0] if vectors else None

    async def handle(
        self, task: Task, *, instance_id: str | None = None
    ) -> ModelResponse:
        if self.engine is not None:
            return await self._handle_workflow(task, instance_id)
        return await self._handle_inline(task)

    async def _handle_inline(self, task: Task) -> ModelResponse:
        """Run the pipeline directly (no durable workflow)."""
        model, scope, messages, query_embedding, block_reason = await self._prepare(task)
        if block_reason is not None:
            await self._record_usage(
                task, model, ModelResponse(text="", model=model), outcome="blocked"
            )
            return _blocked_response(block_reason)

        final_text, accounted, approval_ids = await self._run_tool_loop(
            model, messages, task
        )
        if self.remember:
            await self._remember(task, scope, query_embedding)
        outcome = self._task_outcome(accounted)
        await self._record_usage(task, model, accounted, outcome=outcome)
        self._log_completion(task, accounted, outcome)
        return _final_response(final_text, accounted, approval_ids, model)

    # --- shared phases (used by both the inline and workflow paths) ---

    async def _prepare(
        self, task: Task
    ) -> tuple[str, str, list[dict] | None, list[float] | None, str | None]:
        """Resolve model, enforce budget, recall memory, build messages.

        Returns ``(model, scope, messages, query_embedding, block_reason)``;
        a non-None ``block_reason`` means the budget guard tripped (no model call).
        """
        model = self.agent.model_for(task.kind)
        scope = scope_key_for(self.agent, task.channel)
        logger.info("routing task on %s/%s -> %s", task.surface, task.channel, model)

        decision = await check_budget(self.agent, self.usage)
        if not decision.allowed:
            logger.warning("blocked task for %s: %s", scope, decision.reason)
            return model, scope, None, None, decision.reason

        # Embed the request once and reuse the vector when remembering.
        query_embedding = await self._embed(task.text)
        recalled = await self.memory.recall(scope, query_embedding, limit=RECALL_LIMIT)
        if recalled:
            logger.info("recalled %d memory item(s) for %s", len(recalled), scope)
        messages = self._build_messages(task, recalled)
        return model, scope, messages, query_embedding, None

    async def _remember(
        self, task: Task, scope: str, query_embedding: list[float] | None
    ) -> None:
        await self.memory.remember(
            MemoryRecord(
                scope_key=scope,
                text=task.text,
                kind="message",
                metadata={"user": task.user or "", "surface": task.surface},
                embedding=query_embedding,
            )
        )

    def _log_completion(
        self, task: Task, accounted: ModelResponse, outcome: str
    ) -> None:
        logger.info(
            "completed task on %s/%s with %s (%d+%d tok, $%.4f, %s)",
            task.surface, task.channel, accounted.model,
            accounted.prompt_tokens, accounted.completion_tokens,
            accounted.cost_usd, outcome,
        )

    # --- durable workflow path (consumer #2) ---

    def _build_workflow(self):
        # Imported here to avoid a cycle (engine has no runtime dependency).
        from openloop.workflows.engine import Step, Workflow

        return Workflow(
            self.workflow_name,
            [
                Step("prepare", self._wf_prepare),
                # Model calls aren't idempotent: a crash before `run` completes is
                # abandoned (not replayed); once `run` is done, the idempotent
                # `persist` tail can still resume.
                Step("run", self._wf_run, resumable=False),
                Step("persist", self._wf_persist),
            ],
        )

    async def _handle_workflow(
        self, task: Task, instance_id: str | None = None
    ) -> ModelResponse:
        # A caller (e.g. the Phase D session runner) can bind the workflow
        # instance to its own id so the two share one identity; otherwise mint one.
        instance = await self.engine.start(
            self.workflow_name,
            instance_id or uuid.uuid4().hex,
            {"task": _task_to_dict(task)},
        )
        return self._response_from(instance)

    async def _wf_prepare(self, ctx: "WorkflowContext") -> None:
        task = _task_from_dict(ctx.state["task"])
        model, scope, messages, query_embedding, block_reason = await self._prepare(task)
        ctx.state.update({"model": model, "scope": scope})
        if block_reason is not None:
            ctx.state.update({"blocked": True, "block_reason": block_reason})
            return
        # Persisted turn state: messages (system+history+user), recall vector.
        ctx.state.update({"messages": messages, "query_embedding": query_embedding})

    async def _wf_run(self, ctx: "WorkflowContext") -> None:
        s = ctx.state
        if s.get("blocked"):
            return
        task = _task_from_dict(s["task"])
        final_text, accounted, approval_ids = await self._run_tool_loop(
            s["model"], s["messages"], task
        )
        # Persist received model outputs + tool-loop state + approvals.
        s["messages"] = s["messages"]  # mutated in place by the loop
        s["final_text"] = final_text
        s["accounted"] = _resp_to_dict(accounted)
        s["approval_ids"] = approval_ids

    async def _wf_persist(self, ctx: "WorkflowContext") -> None:
        s = ctx.state
        task = _task_from_dict(s["task"])
        if s.get("blocked"):
            if not s.get("usage_recorded"):
                await self._record_usage(
                    task, s["model"], ModelResponse(text="", model=s["model"]),
                    outcome="blocked",
                )
                s["usage_recorded"] = True
            return

        accounted = _resp_from_dict(s["accounted"])
        # Idempotent writes: flags guard against a resumed persist double-writing.
        if self.remember and not s.get("remembered"):
            await self._remember(task, s["scope"], s.get("query_embedding"))
            s["remembered"] = True
            await self.engine.checkpoint(ctx.instance)
        if not s.get("usage_recorded"):
            outcome = self._task_outcome(accounted)
            await self._record_usage(task, s["model"], accounted, outcome=outcome)
            s["usage_recorded"] = True
            self._log_completion(task, accounted, outcome)

    async def recover_response(
        self, instance_id: str
    ) -> tuple[bool, "ModelResponse | None"]:
        """For the Phase D session reconciler: recover a crashed turn's response
        from its persisted workflow, **without** re-running it.

        Returns ``(found, response)``:

        - ``(False, None)`` — no engine, or the instance is gone: unrecoverable,
          so the reconciler should post an interrupted notice.
        - ``(True, None)`` — the instance exists but is **not terminal** (still
          running/waiting, e.g. the engine's own resume hasn't finished or failed):
          leave it for a later restart rather than delivering a half-finished turn.
        - ``(True, response)`` — terminal: the answer (``completed``) or an
          interrupted notice (``failed`` / ``cancelled`` / ``abandoned``).
        """
        from openloop.workflows.store import TERMINAL as WF_TERMINAL

        if self.engine is None:
            return False, None
        instance = await self.engine.store.get(instance_id)
        if instance is None:
            return False, None
        if instance.status not in WF_TERMINAL:
            return True, None
        if instance.status in ("failed", "cancelled", "abandoned"):
            return True, _interrupted_response()
        return True, self._response_from(instance)

    def _response_from(self, instance) -> ModelResponse:
        s = instance.state
        if s.get("blocked"):
            return _blocked_response(s["block_reason"])
        if instance.status in ("failed", "abandoned"):
            return _interrupted_response()
        accounted = _resp_from_dict(s.get("accounted", {}))
        return _final_response(
            s.get("final_text", ""), accounted, s.get("approval_ids", []),
            s.get("model", accounted.model),
        )

    async def _run_tool_loop(
        self, model: str, messages: list[dict], task: Task
    ) -> tuple[str, ModelResponse, list[str]]:
        """Drive model<->tool round-trips until a final answer or an approval.

        Returns the user-facing text, an accumulated ModelResponse for usage
        accounting (real model + summed tokens/cost), and the IDs of any write
        actions left awaiting human approval.
        """
        specs = self.tools.tool_specs(self.agent) if self.tools else None
        tool_defs = specs.definitions if specs and specs.definitions else None
        by_name = specs.by_name if specs else {}

        total_cost = 0.0
        total_pt = total_ct = 0
        final_model = model
        final_text = ""
        approval_messages: list[str] = []
        approval_ids: list[str] = []
        response = None

        for _ in range(MAX_TOOL_ITERS):
            response = await self.gateway.complete(model, messages, tools=tool_defs)
            total_cost += response.cost_usd
            total_pt += response.prompt_tokens
            total_ct += response.completion_tokens
            final_model = response.model or model

            if not response.tool_calls:
                final_text = response.text
                break

            messages.append(
                response.raw_message
                or {"role": "assistant", "content": response.text}
            )
            stop_for_approval = False
            for call in response.tool_calls:
                action = by_name.get(call.name)
                if action is None:
                    messages.append(
                        _tool_message(call.id, f"error: unknown tool {call.name}")
                    )
                    continue
                inv = await self.tools.invoke(
                    self.agent, action, call.arguments, requested_by=task.user
                )
                if inv.status == "executed":
                    summary = inv.result.summary if inv.result else "done"
                    messages.append(_tool_message(call.id, summary))
                elif inv.status == "pending_approval":
                    approval_messages.append(inv.message or "approval required")
                    if inv.approval is not None:
                        approval_ids.append(inv.approval.id)
                    messages.append(
                        _tool_message(call.id, f"held for human approval: {inv.message}")
                    )
                    stop_for_approval = True
                else:  # forbidden / denied
                    messages.append(
                        _tool_message(call.id, f"{inv.status}: {inv.message}")
                    )
            if stop_for_approval:
                break
        else:
            final_text = (
                (response.text if response else "")
                or "I couldn't finish that within the tool-call limit."
            )

        if approval_messages:
            final_text = "\n".join(approval_messages)

        accounted = ModelResponse(
            text=final_text,
            model=final_model,
            prompt_tokens=total_pt,
            completion_tokens=total_ct,
            cost_usd=total_cost,
        )
        return final_text, accounted, approval_ids

    def _task_outcome(self, response: ModelResponse) -> str:
        per_task = self.agent.spec.budget.per_task_usd
        if per_task is not None and response.cost_usd > per_task:
            logger.warning(
                "task cost $%.4f exceeded per-task budget $%.4f",
                response.cost_usd,
                per_task,
            )
            return "over_task_budget"
        return "ok"

    async def _record_usage(
        self, task: Task, model: str, response: ModelResponse, *, outcome: str
    ) -> None:
        await self.usage.record(
            UsageRecord(
                scope_key=budget_scope_key(self.agent),
                workspace=self.agent.metadata.workspace,
                agent=self.agent.metadata.name,
                model=response.model or model,
                channel=task.channel,
                surface=task.surface,
                user=task.user,
                task_kind=task.kind,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cost_usd=response.cost_usd,
                outcome=outcome,
            )
        )


def _blocked_response(reason: str) -> ModelResponse:
    return ModelResponse(
        text=f"💸 Budget guard: {reason}. Action blocked.", model="budget-guard"
    )


def _interrupted_response() -> ModelResponse:
    return ModelResponse(
        text="⚠️ This task was interrupted and could not be completed.",
        model="error",
    )


def _final_response(
    final_text: str,
    accounted: ModelResponse,
    approval_ids: list[str],
    model: str,
) -> ModelResponse:
    return ModelResponse(
        text=final_text,
        model="approval-gate" if approval_ids else (accounted.model or model),
        prompt_tokens=accounted.prompt_tokens,
        completion_tokens=accounted.completion_tokens,
        cost_usd=accounted.cost_usd,
        approval_ids=approval_ids,
    )


def _task_to_dict(task: Task) -> dict:
    return {
        "text": task.text,
        "surface": task.surface,
        "channel": task.channel,
        "user": task.user,
        "kind": task.kind,
        "history": task.history,
    }


def _task_from_dict(data: dict) -> Task:
    return Task(
        text=data["text"],
        surface=data["surface"],
        channel=data.get("channel"),
        user=data.get("user"),
        kind=data.get("kind"),
        history=data.get("history", []),
    )


def _resp_to_dict(response: ModelResponse) -> dict:
    return {
        "text": response.text,
        "model": response.model,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "cost_usd": response.cost_usd,
    }


def _resp_from_dict(data: dict) -> ModelResponse:
    return ModelResponse(
        text=data.get("text", ""),
        model=data.get("model", ""),
        prompt_tokens=data.get("prompt_tokens", 0),
        completion_tokens=data.get("completion_tokens", 0),
        cost_usd=data.get("cost_usd", 0.0),
    )
