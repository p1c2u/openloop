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
it once. The narrow window between a successful provider post and recording its
message id — where the persisted-id guard can't help — is covered by a
deterministic delivery key: every post is tagged with it and the recovery path
looks the message up by key instead of re-posting (best-effort; a surface whose
lookup can't run degrades back to at-least-once). One gap remains by design: a
session that crashed mid-turn is recovered by the startup reconciler (Slice 6),
not this inline path (it must not replay the model call). The original request
does not own the task's lifetime — the runner does, and it can be awaited inline
(tests) or scheduled in the background (Slack).
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

# How many prior thread turns to replay as conversation history. A safety bound
# on context size, not a correctness limit — older turns fall back to recall.
HISTORY_TURN_LIMIT = 20


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

        session = SurfaceSession(
            id=uuid.uuid4().hex,
            target=target,
            status="queued",
            # Persist the inbound text so a later turn in this thread can replay it
            # as conversation history (see _apply_thread_history).
            request_text=task.text,
        )
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

        # Replay earlier turns of this thread so the model has the conversation in
        # context, not just semantic recall. Done before handle() so the history
        # is baked into the workflow's persisted turn state (resume-safe).
        await self._apply_thread_history(task, session)

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

    async def reconcile(self) -> list[str]:
        """Repair delivery state for sessions left mid-flight by a crash.

        Call once at startup, **after** the workflow engine's own
        ``resume_incomplete`` has driven crashed turns to a terminal state. For
        each session:

        - ``waiting`` (parked on a human approval) or already-delivered → leave
          it alone;
        - terminal but with no final message (the turn finished but a Slack post
          failed, or it crashed between the status flip and the post) →
          re-deliver from the persisted outcome;
        - still ``queued`` / ``running`` (the turn crashed before it was
          delivered) → recover the answer from the now-terminal workflow instance
          and deliver it, or post an interrupted notice if it can't be recovered.

        Idempotent, so safe to run on every boot. Across replicas, the app lifespan
        runs it under a ``startup-recovery`` :class:`~openloop.coordination.\
        DistributedLock` so only the leader sweeps; delivery stays id-/key-guarded
        if two ever overlap.
        """
        repaired: list[str] = []
        for session in await self.sessions.recent(limit=1000):
            if session.status == "waiting" or session.final_message_id is not None:
                continue
            if session.status in TERMINAL:
                await self._ensure_delivered(session)
                repaired.append(session.id)
                continue
            # queued / running — recover from the workflow the session is bound to.
            found, response = await self._recover(session)
            if response is not None:
                await self._deliver(session, response)
            elif not found:
                # No recoverable workflow (missing instance / no engine) → notice.
                session.status = "abandoned"
                session.error = ERROR_TEXT
                await self.sessions.upsert(session)
                await self._post_error(session)
            else:
                # The workflow exists but isn't terminal yet — leave it for a later
                # restart rather than delivering a half-finished turn.
                continue
            repaired.append(session.id)
        return repaired

    async def _apply_thread_history(self, task: Task, session: SurfaceSession) -> None:
        """Populate ``task.history`` from earlier delivered turns in this thread.

        Rebuilds the conversation from the durable sessions — each prior delivered
        exchange contributes a ``user`` (its request) + ``assistant`` (its answer)
        pair, oldest-first — rather than re-fetching the surface's own transcript.
        That keeps it surface-agnostic and free of delivery scaffolding (progress
        notes, approval cards never appear). The store decides what's replayable
        (only completed, *delivered* exchanges — never an answer the user didn't
        see; see ``thread_history``), so this just maps them to messages. A caller
        that already supplied history is left untouched, and a session with no
        thread (or the thread's first turn) simply gets no history.
        """
        if task.history or session.target.thread is None:
            return
        prior = await self.sessions.thread_history(
            session.target, exclude_id=session.id, limit=HISTORY_TURN_LIMIT
        )
        turns: list[dict[str, str]] = []
        for s in prior:
            turns.append({"role": "user", "content": s.request_text})
            turns.append({"role": "assistant", "content": s.result_summary})
        if turns:
            task.history = turns

    async def _recover(self, session: SurfaceSession) -> tuple[bool, object]:
        """``(found, response)`` for a session's workflow — see
        :meth:`Runtime.recover_response`."""
        instance_id = session.workflow_instance_id
        recover = getattr(self.runtime, "recover_response", None)
        if instance_id is None or recover is None:
            return False, None
        return await recover(instance_id)

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
        # This is the retry path: the post may already have landed before its id
        # was persisted, so ask delivery to recover-or-post (recover=True) rather
        # than blindly re-posting and duplicating the answer.
        if session.status == "completed":
            await self._post_final(
                session, session.result_summary or "(no response)", recover=True
            )
        elif session.status in ("failed", "abandoned"):
            await self._post_error(session, recover=True)
        return session

    # --- idempotent delivery helpers (guarded by persisted message ids) ---

    @staticmethod
    def _delivery_key(session: SurfaceSession, role: str) -> str:
        """Deterministic dedup key for one of a session's posts.

        Stable across retries (keyed on the session id), so a recovery post can
        find the message a crashed first attempt already sent. One key per role so
        progress / final / error never collide.
        """
        return f"{session.id}:{role}"

    async def _post_progress(self, session: SurfaceSession) -> None:
        if session.progress_message_id is not None:
            return
        mid = await self.delivery.post_progress(
            session.target, PROGRESS_TEXT, key=self._delivery_key(session, "progress")
        )
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

    async def _post_final(
        self, session: SurfaceSession, text: str, *, recover: bool = False
    ) -> None:
        if session.final_message_id is not None:
            return  # already delivered — never post a second final answer
        mid = await self.delivery.post_final(
            session.target, text,
            key=self._delivery_key(session, "final"), recover=recover,
        )
        session.final_message_id = mid
        await self.sessions.upsert(session)

    async def _post_error(
        self, session: SurfaceSession, *, recover: bool = False
    ) -> None:
        if session.final_message_id is not None:
            return
        mid = await self.delivery.post_error(
            session.target, session.error or ERROR_TEXT,
            key=self._delivery_key(session, "error"), recover=recover,
        )
        session.final_message_id = mid
        await self.sessions.upsert(session)
