"""Session runner — binds one surface session to one ``agent_task`` workflow.

This is the delivery layer Phase D adds on top of Phase C's durable chat turn.
Given an inbound surface event the runner:

1. creates (or re-uses) a :class:`SurfaceSession`, idempotent on the event id;
2. posts a short progress message and marks the session ``running``;
3. drives the turn via :meth:`Runtime.handle`, binding the workflow instance id
   to the session id so the two share one identity;
4. records the result/error on the session and asks the
   :class:`~openloop.sessions.delivery.SurfaceDelivery` to post the final answer.

Progress is coarse for this first pass (``queued`` → ``running`` → ``waiting`` /
``completed`` / ``failed``). Every delivery is guarded by a persisted message id,
so a duplicate event never posts a second final answer, and a retry of a session
that crashed *after* reaching a terminal state but *before* posting re-delivers
it once. Two gaps remain by design: a session that crashed mid-turn is recovered
by the startup reconciler (Slice 6), not this inline path (it must not replay the
model call); and the narrow window between a successful provider post and
recording its message id is at-least-once — closing it needs a provider
idempotency key. The original request does not own the task's lifetime — the
runner does, and it can be awaited inline (tests) or scheduled in the background
(Slack).
"""

from __future__ import annotations

import logging
import uuid

from openloop.runtime import Runtime, Task
from openloop.sessions.delivery import SurfaceDelivery
from openloop.sessions.store import (
    TERMINAL,
    SurfaceSession,
    SurfaceSessionStore,
    SurfaceTarget,
)

logger = logging.getLogger(__name__)

PROGRESS_TEXT = "🤖 On it…"
WAITING_TEXT = "⏳ Waiting for approval…"
ERROR_TEXT = "⚠️ This task was interrupted and could not be completed."


class SessionRunner:
    """Runs a task as a background session and delivers the answer back."""

    def __init__(
        self,
        runtime: Runtime,
        sessions: SurfaceSessionStore,
        delivery: SurfaceDelivery,
    ) -> None:
        self.runtime = runtime
        self.sessions = sessions
        self.delivery = delivery

    async def run(self, task: Task, target: SurfaceTarget) -> SurfaceSession:
        """Create/resume a session for ``task`` and deliver its outcome.

        Idempotent on ``target.event_id``: a duplicate inbound event reuses the
        existing session rather than starting a second turn. If that session
        reached a terminal state but crashed before its answer was posted, the
        retry re-delivers it (guarded by the persisted message id, so never
        twice). A session still mid-turn is left for the startup reconciler
        (Slice 6) — this inline retry path does not replay the model call.
        """
        existing = await self.sessions.get_by_event(target.event_id)
        if existing is not None:
            return await self._ensure_delivered(existing)

        session = SurfaceSession(id=uuid.uuid4().hex, target=target, status="queued")
        # One session : one workflow instance — share the id so the approval
        # continuation / reconciler can map between them trivially.
        session.workflow_instance_id = session.id
        try:
            await self.sessions.upsert(session)
        except Exception:  # noqa: BLE001 — a concurrent duplicate won the race
            # The event_id unique index rejected this insert: another delivery of
            # the same event created the session first. Defer to the winner.
            racer = await self.sessions.get_by_event(target.event_id)
            if racer is not None:
                return await self._ensure_delivered(racer)
            raise

        await self._post_progress(session)
        session.status = "running"
        await self.sessions.upsert(session)

        try:
            response = await self.runtime.handle(
                task, instance_id=session.workflow_instance_id
            )
        except Exception as exc:  # noqa: BLE001 — record + deliver, don't crash caller
            logger.exception("session %s failed while handling the task", session.id)
            session.status = "failed"
            session.error = str(exc)
            await self.sessions.upsert(session)
            await self._post_error(session)
            return session

        return await self._deliver(session, response)

    async def _deliver(self, session: SurfaceSession, response) -> SurfaceSession:
        if response.model == "error":
            # The workflow was interrupted inside a non-resumable model step.
            session.status = "abandoned"
            session.error = response.text or ERROR_TEXT
            await self.sessions.upsert(session)
            await self._post_error(session)
            return session

        if response.approval_ids:
            # Parked on a human approval. Persist the approval ids so Slice 4 can
            # map a button click back to this session and post the eventual answer.
            session.status = "waiting"
            session.approval_ids = list(response.approval_ids)
            session.result_summary = response.text or WAITING_TEXT
            await self.sessions.upsert(session)
            # Turn the progress message into an approval card (buttons in-thread).
            requests = await self._approval_requests(session.approval_ids)
            await self._update_approval(
                session, response.text or WAITING_TEXT, requests
            )
            return session

        session.status = "completed"
        session.result_summary = response.text or "(no response)"
        await self.sessions.upsert(session)
        await self._post_final(session, session.result_summary)
        return session

    async def resolve_approval(
        self, approval_id: str, approver: str, *, approve: bool
    ) -> str:
        """Resolve an approval and continue the session that was waiting on it.

        Resolves the approval through the tool gateway (which executes the write /
        wakes its workflow), then — if a session is parked on this approval —
        posts the outcome as the final answer in the original thread and closes
        the session. Returns the status line for the button-click reply.

        Delivery failures never block the button reply and always leave the
        session in a repairable state: a session left ``waiting`` retries the
        whole continuation on the next click; one already flipped terminal but
        not yet delivered is repaired idempotently from its persisted outcome. So
        even if the tool side effect succeeds but a Slack post fails, a second
        click (or the startup reconciler) still delivers the answer.
        """
        from openloop.surfaces.approvals import resolution_message

        tools = getattr(self.runtime, "tools", None)
        if tools is None:
            return "⛔ approvals are not available"
        inv = await tools.resolve(approval_id, approver, approve=approve)
        message = resolution_message(inv, approver)

        session = await self.sessions.get_by_approval(approval_id)
        if session is not None:
            try:
                if session.status == "waiting":
                    await self._continue_session(session, inv, approver, message)
                elif session.status in TERMINAL and session.final_message_id is None:
                    # A prior continuation flipped the session terminal but a Slack
                    # post failed before the answer landed — re-deliver it from the
                    # persisted outcome (idempotent; reuses result_summary).
                    await self._ensure_delivered(session)
            except Exception:  # noqa: BLE001 — leave it repairable, still reply
                logger.exception(
                    "failed to deliver approval outcome for session %s", session.id
                )
        return message

    async def _continue_session(
        self, session: SurfaceSession, inv, approver: str, message: str
    ) -> None:
        """Post a resolved approval's outcome in-thread and close the session."""
        if inv.status == "executed":
            detail = inv.result.summary if inv.result else (inv.message or "done")
            final_text = detail
        elif inv.status == "denied":
            final_text = f"🚫 Denied by {approver}."
        else:  # forbidden / not-an-approver / already resolved — leave it parked
            return
        # Persist the outcome (so a failed post is repairable from result_summary),
        # then deliver the ANSWER first — the approval card collapse is cosmetic and
        # must never block or lose the final reply.
        session.status = "completed"
        session.result_summary = final_text
        await self.sessions.upsert(session)
        await self._post_final(session, final_text)
        try:
            await self._update_approval(session, message, [])
        except Exception:  # noqa: BLE001 — buttons going stale is cosmetic
            logger.exception(
                "failed to collapse approval card for session %s", session.id
            )

    async def _ensure_delivered(self, session: SurfaceSession) -> SurfaceSession:
        """Re-deliver an existing session's answer if it crashed before posting.

        Called for a duplicate event / retry. The ``_post_*`` helpers are guarded
        by ``final_message_id``, so a fully delivered session is returned
        untouched while a terminal-but-undelivered one finally gets its answer. A
        session still ``queued`` / ``running`` (a mid-turn crash) or ``waiting``
        is returned as-is — recovering those is the reconciler's job, not this
        synchronous retry path (which must not replay the model call).
        """
        if session.final_message_id is not None:
            return session
        if session.status == "completed":
            await self._post_final(session, session.result_summary or "(no response)")
        elif session.status in ("failed", "abandoned"):
            await self._post_error(session)
        return session

    # --- idempotent delivery helpers (guarded by persisted message ids) ---

    async def _post_progress(self, session: SurfaceSession) -> None:
        if session.progress_message_id is not None:
            return
        mid = await self.delivery.post_progress(session.target, PROGRESS_TEXT)
        session.progress_message_id = mid
        await self.sessions.upsert(session)

    async def _update_progress(self, session: SurfaceSession, text: str) -> None:
        if session.progress_message_id is None:
            return
        await self.delivery.update_progress(
            session.target, session.progress_message_id, text
        )

    async def _update_approval(self, session: SurfaceSession, text: str, requests) -> None:
        if session.progress_message_id is None:
            return
        await self.delivery.update_approval(
            session.target, session.progress_message_id, text, requests
        )

    async def _approval_requests(self, approval_ids: list[str]) -> list:
        """Fetch the pending ApprovalRequest objects so delivery can render them."""
        tools = getattr(self.runtime, "tools", None)
        if tools is None:
            return []
        out = []
        for rid in approval_ids:
            req = await tools.approvals.get(rid)
            if req is not None:
                out.append(req)
        return out

    async def _post_final(self, session: SurfaceSession, text: str) -> None:
        if session.final_message_id is not None:
            return  # already delivered — never post a second final answer
        mid = await self.delivery.post_final(session.target, text)
        session.final_message_id = mid
        await self.sessions.upsert(session)

    async def _post_error(self, session: SurfaceSession) -> None:
        if session.final_message_id is not None:
            return
        mid = await self.delivery.post_error(session.target, session.error or ERROR_TEXT)
        session.final_message_id = mid
        await self.sessions.upsert(session)
