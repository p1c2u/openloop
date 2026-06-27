"""Unit tests for Phase D — surface sessions, delivery, and the session runner.

Covers the session-store state transitions + event dedup, and the runner's
mention → progress → final / waiting / interrupted flows with idempotent delivery
(a duplicate event never starts a second turn or posts a second answer).
"""

import pytest

from openloop.agents import load_agent
from openloop.memory import InMemoryStore
from openloop.models.gateway import ModelResponse
from openloop.runtime import Runtime, Task
from openloop.sessions import (
    InMemorySurfaceSessionStore,
    SessionRunner,
    SurfaceSession,
    SurfaceTarget,
)
from openloop.tools import ToolGateway
from openloop.tools.github import GitHubConnector
from openloop.usage import InMemoryUsageStore
from openloop.workflows import InMemoryWorkflowStore, WorkflowEngine
from openloop.testing import (
    EXAMPLE_AGENT,
    FakeGitHub,
    FakeSurfaceDelivery,
    ScriptedGateway,
    tool_call_response,
)

pytestmark = pytest.mark.unit


def _target(event_id="ev1"):
    return SurfaceTarget(
        surface="slack",
        workspace="acme",
        agent="dev-platform",
        channel="C1",
        thread="100.1",
        event_id=event_id,
    )


def _task(text="hi"):
    return Task(text=text, surface="slack", channel="C1", user="U1")


def _runner(model_gateway, *, tools=None, delivery=None):
    sessions = InMemorySurfaceSessionStore()
    delivery = delivery or FakeSurfaceDelivery()
    engine = WorkflowEngine(InMemoryWorkflowStore())
    runtime = Runtime(
        load_agent(EXAMPLE_AGENT),
        gateway=model_gateway,
        tools=tools,
        usage=InMemoryUsageStore(),
        memory=InMemoryStore(),
        engine=engine,
    )
    return SessionRunner(runtime, sessions, delivery), sessions, delivery


# --- session store -------------------------------------------------------

async def test_store_upsert_get_and_event_lookup():
    store = InMemorySurfaceSessionStore()
    session = SurfaceSession(id="s1", target=_target("ev-abc"))
    await store.upsert(session)

    assert (await store.get("s1")).id == "s1"
    assert (await store.get_by_event("ev-abc")).id == "s1"
    assert await store.get_by_event("nope") is None
    assert await store.get_by_event("") is None


async def test_store_upsert_preserves_created_at_bumps_updated_at():
    store = InMemorySurfaceSessionStore()
    session = SurfaceSession(id="s1", target=_target())
    await store.upsert(session)
    created = session.created_at

    session.status = "completed"
    await store.upsert(session)

    stored = await store.get("s1")
    assert stored.status == "completed"
    assert stored.created_at == created
    assert stored.updated_at >= created


# --- runner: happy path --------------------------------------------------

async def test_mention_to_progress_then_final():
    runner, sessions, delivery = _runner(
        ScriptedGateway([ModelResponse(text="here you go", model="m")])
    )

    session = await runner.run(_task(), _target())

    assert session.status == "completed"
    assert session.result_summary == "here you go"
    # Progress posted first, then a single final answer.
    assert len(delivery.progress) == 1
    assert len(delivery.finals) == 1
    assert delivery.finals[0]["text"] == "here you go"
    # Both message ids persisted on the session, and the workflow shares its id.
    assert session.progress_message_id == delivery.progress[0]["id"]
    assert session.final_message_id == delivery.finals[0]["id"]
    assert session.workflow_instance_id == session.id


async def test_duplicate_event_is_deduped():
    gateway = ScriptedGateway([ModelResponse(text="once", model="m")])
    runner, sessions, delivery = _runner(gateway)

    first = await runner.run(_task(), _target("dupe"))
    second = await runner.run(_task(), _target("dupe"))

    assert first.id == second.id
    # No second turn, no second final answer.
    assert gateway._responses == []  # only one response consumed
    assert len(delivery.finals) == 1
    assert len(sessions._by_id) == 1


# --- runner: waiting for approval ---------------------------------------

async def test_pending_approval_parks_session_waiting():
    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)])
    runner, sessions, delivery = _runner(
        ScriptedGateway([
            tool_call_response(
                "m", [("c1", "github_issues_write", {"repo": "acme/x", "title": "T"})]
            ),
        ]),
        tools=tools,
    )

    session = await runner.run(_task("open an issue"), _target())

    assert session.status == "waiting"
    # The approval ids are persisted so Slice 4 can map a button back here.
    assert len(session.approval_ids) == 1
    assert (await sessions.get(session.id)).approval_ids == session.approval_ids
    # No final answer yet — the approval continuation (Slice 4) delivers it.
    assert delivery.finals == []
    # The progress message was turned into an approval card carrying the request.
    assert len(delivery.approvals) == 1
    assert [r.id for r in delivery.approvals[0]["requests"]] == session.approval_ids
    assert github.created == []  # write not executed


def _waiting_runner(*, delivery=None):
    """A runner whose session is parked on a github write approval."""
    github = FakeGitHub()
    tools = ToolGateway(tools=[GitHubConnector(github)])
    runner, sessions, delivery = _runner(
        ScriptedGateway([
            tool_call_response(
                "m", [("c1", "github_issues_write", {"repo": "acme/x", "title": "T"})]
            ),
        ]),
        tools=tools,
        delivery=delivery,
    )
    return runner, sessions, delivery, github


# --- runner: approval continuation (Slice 4) ----------------------------

async def test_approve_continues_session_and_posts_outcome_in_thread():
    runner, sessions, delivery, github = _waiting_runner()
    session = await runner.run(_task("open an issue"), _target())
    approval_id = session.approval_ids[0]

    message = await runner.resolve_approval(approval_id, "@priya", approve=True)

    assert message.startswith("✅ Approved by @priya")
    assert github.created  # the write executed on approval
    # The outcome is delivered as the final answer in the original thread.
    assert len(delivery.finals) == 1
    assert delivery.finals[0]["target"].thread == "100.1"
    # The session is now completed and the approval card was collapsed (no buttons).
    done = await sessions.get(session.id)
    assert done.status == "completed"
    assert done.final_message_id is not None
    assert delivery.approvals[-1]["requests"] == []


async def test_deny_continues_session_without_executing():
    runner, sessions, delivery, github = _waiting_runner()
    session = await runner.run(_task("open an issue"), _target())

    message = await runner.resolve_approval(
        session.approval_ids[0], "@priya", approve=False
    )

    assert message.startswith("🚫 Denied")
    assert github.created == []
    done = await sessions.get(session.id)
    assert done.status == "completed"
    assert "Denied" in delivery.finals[-1]["text"]


async def test_non_approver_leaves_session_waiting():
    runner, sessions, delivery, github = _waiting_runner()
    session = await runner.run(_task("open an issue"), _target())

    message = await runner.resolve_approval(
        session.approval_ids[0], "@random", approve=True
    )

    assert message.startswith("⛔")
    assert github.created == []
    # The session stays parked; no final answer posted.
    assert (await sessions.get(session.id)).status == "waiting"
    assert delivery.finals == []


async def test_failed_outcome_delivery_is_repaired_on_second_click():
    # The write succeeds but the Slack post_final fails the first time. The session
    # is left terminal-without-final; a second click must re-deliver the answer
    # (and never re-execute the write), rather than the user getting nothing.
    class FlakyDelivery(FakeSurfaceDelivery):
        def __init__(self):
            super().__init__()
            self.fail_finals = 1

        async def post_final(self, target, text, *, blocks=None):
            if self.fail_finals > 0:
                self.fail_finals -= 1
                raise RuntimeError("slack down")
            return await super().post_final(target, text, blocks=blocks)

    runner, sessions, delivery, github = _waiting_runner(delivery=FlakyDelivery())
    session = await runner.run(_task("open an issue"), _target())
    approval_id = session.approval_ids[0]

    # First click: write executes, but delivering the answer fails (swallowed).
    msg1 = await runner.resolve_approval(approval_id, "@priya", approve=True)
    assert msg1.startswith("✅ Approved by @priya")
    assert len(github.created) == 1
    stuck = await sessions.get(session.id)
    assert stuck.status == "completed" and stuck.final_message_id is None
    assert delivery.finals == []  # nothing delivered yet

    # Second click: no re-execution, and the persisted outcome is re-delivered.
    await runner.resolve_approval(approval_id, "@priya", approve=True)
    assert len(github.created) == 1  # write was not repeated
    repaired = await sessions.get(session.id)
    assert repaired.final_message_id is not None
    assert len(delivery.finals) == 1
    assert delivery.finals[0]["text"] == repaired.result_summary


# --- runner: crash-before-delivery repaired on retry ---------------------

async def test_retry_redelivers_terminal_session_without_final():
    # A session that reached `completed` but crashed before posting its final
    # answer (final_message_id is None). A retry of the same event re-delivers it
    # exactly once instead of returning a stuck, answerless session.
    runner, sessions, delivery = _runner(ScriptedGateway([]))
    await sessions.upsert(SurfaceSession(
        id="s-crash", target=_target("ev-crash"), status="completed",
        workflow_instance_id="s-crash", progress_message_id="progress-0",
        result_summary="the answer",
    ))

    session = await runner.run(_task(), _target("ev-crash"))

    assert session.id == "s-crash"
    assert len(delivery.finals) == 1
    assert delivery.finals[0]["text"] == "the answer"
    assert session.final_message_id == delivery.finals[0]["id"]

    # A further retry must not post a second final answer.
    again = await runner.run(_task(), _target("ev-crash"))
    assert again.final_message_id == session.final_message_id
    assert len(delivery.finals) == 1


# --- runner: interrupted / error ----------------------------------------

async def test_interrupted_turn_marks_abandoned_and_posts_error():
    # A model exception is caught by the workflow engine (step -> failed), so
    # handle() returns the interrupted `model="error"` response rather than
    # raising; the runner reflects that as an abandoned session + error notice.
    class BoomGateway:
        async def complete(self, model, messages, **kwargs):
            raise RuntimeError("model exploded")

    runner, sessions, delivery = _runner(BoomGateway())

    session = await runner.run(_task(), _target())

    assert session.status == "abandoned"
    assert session.error
    assert len(delivery.errors) == 1
    assert delivery.finals == []
