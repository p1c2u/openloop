"""Integration: Runtime.handle running as a durable `agent_task` workflow.

Phase C consumer #2 — when an engine is wired, each task runs through the workflow
engine (prepare → run → persist), persisting turn state and writing usage/memory
idempotently, while never replaying non-idempotent model calls on crash.
"""

from openloop.agents import load_agent
from openloop.memory import InMemoryStore, scope_key_for
from openloop.models.gateway import ModelResponse
from openloop.runtime import Runtime, Task
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.usage import InMemoryUsageStore, UsageRecord, budget_scope_key
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine, WorkflowInstance
from openloop.testing import (
    EXAMPLE_AGENT,
    FakeGitHub,
    ScriptedGateway,
    tool_call_response,
)


class CountingGateway:
    """Model gateway that records how many times it was called."""

    def __init__(self):
        self.calls = 0

    async def complete(self, model, messages, **kwargs):
        self.calls += 1
        return ModelResponse(text="hi", model="m")


def _agent():
    return load_agent(EXAMPLE_AGENT)


def _task(text="hi"):
    return Task(text=text, surface="slack", channel="#dev-platform", user="U1")


def _runtime(model_gateway, *, tools=None, usage=None, memory=None):
    store = InMemoryWorkflowStore()
    engine = WorkflowEngine(store)
    rt = Runtime(
        _agent(),
        gateway=model_gateway,
        tools=tools,
        usage=usage or InMemoryUsageStore(),
        memory=memory or InMemoryStore(),
        engine=engine,
    )
    return rt, engine, store


async def test_plain_chat_runs_through_the_workflow():
    rt, engine, store = _runtime(ScriptedGateway([ModelResponse(text="hello", model="m")]))
    res = await rt.handle(_task())

    assert res.text == "hello"
    inst = (await store.recent())[0]
    assert inst.workflow == rt.workflow_name
    assert inst.status == "completed"
    assert inst.completed_steps == ["prepare", "run", "persist"]
    # Turn state persisted: messages + received model output.
    assert inst.state["final_text"] == "hello"
    assert inst.state["messages"][-1] == {"role": "user", "content": "hi"}


async def test_write_tool_call_held_for_approval_via_workflow():
    usage = InMemoryUsageStore()
    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)])
    rt, engine, store = _runtime(
        ScriptedGateway([
            tool_call_response("m", [("c1", "github_issues_write",
                                      {"repo": "acme/x", "title": "T"})]),
        ]),
        tools=tools, usage=usage,
    )

    res = await rt.handle(_task("open an issue"))

    assert res.model == "approval-gate"
    assert res.approval_ids
    assert github.created == []  # not executed yet
    inst = (await store.recent())[0]
    assert inst.status == "completed"
    # Approvals are part of the persisted turn state.
    assert inst.state["approval_ids"] == res.approval_ids
    assert len(usage.records) == 1  # usage written exactly once


async def test_budget_block_through_workflow():
    usage = InMemoryUsageStore()
    await usage.record(UsageRecord(
        scope_key=budget_scope_key(_agent()), workspace="acme", agent="dev-platform",
        model="m", cost_usd=1000.0, outcome="ok",
    ))
    rt, engine, store = _runtime(
        ScriptedGateway([ModelResponse(text="x", model="m")]), usage=usage
    )

    res = await rt.handle(_task())

    assert res.model == "budget-guard"
    inst = (await store.recent())[0]
    assert inst.state.get("blocked") is True
    assert any(r.outcome == "blocked" for r in usage.records)


async def test_turn_is_remembered_once():
    memory = InMemoryStore()
    rt, engine, store = _runtime(
        ScriptedGateway([ModelResponse(text="ok", model="m")]), memory=memory
    )
    await rt.handle(_task("remember this"))

    inst = (await store.recent())[0]
    assert inst.state.get("remembered") is True
    # The user's message was remembered for the channel scope.
    from openloop.memory import scope_key_for
    recalled = await memory.recall(scope_key_for(_agent(), "#dev-platform"), None, limit=5)
    assert any("remember this" in r.text for r in recalled)


async def test_crash_before_run_completes_is_abandoned_not_replayed():
    gateway = CountingGateway()
    rt, engine, store = _runtime(gateway)
    # Crashed mid-run: prepare done, run (non-resumable) not yet complete.
    await store.upsert(WorkflowInstance(
        id="crashed", workflow=rt.workflow_name, status="running",
        completed_steps=["prepare"],
        state={
            "task": {"text": "hi", "surface": "slack", "channel": "#dev-platform",
                     "user": "U1"},
            "model": "m", "messages": [],
        },
    ))

    resumed = await engine.resume_incomplete()

    assert "crashed" not in resumed  # not resumed
    assert (await store.get("crashed")).status == "abandoned"
    assert gateway.calls == 0  # the model was never re-called


async def test_crash_after_run_resumes_idempotent_persist_tail():
    # The recoverable case: run already completed (model output persisted), only
    # the idempotent persist tail remains — resume it instead of abandoning.
    usage = InMemoryUsageStore()
    gateway = CountingGateway()
    rt, engine, store = _runtime(gateway, usage=usage)
    await store.upsert(WorkflowInstance(
        id="midpersist", workflow=rt.workflow_name, status="running",
        completed_steps=["prepare", "run"],
        state={
            "task": {"text": "hi", "surface": "slack", "channel": "#dev-platform",
                     "user": "U1"},
            "model": "m", "scope": scope_key_for(_agent(), "#dev-platform"),
            "messages": [], "query_embedding": None, "final_text": "answer",
            "accounted": {"model": "m", "prompt_tokens": 3,
                          "completion_tokens": 2, "cost_usd": 0.01},
            "approval_ids": [],
        },
    ))

    resumed = await engine.resume_incomplete()

    assert "midpersist" in resumed
    assert (await store.get("midpersist")).status == "completed"
    assert len(usage.records) == 1  # the persist tail wrote usage on resume
    assert gateway.calls == 0  # no model replay
