"""Surface sessions — the durable identity for a user-visible async task (Phase D).

Phase C made the chat pipeline durable *internally*, but Slack still replied
inside the original request lifecycle. Phase D turns that into Claude Tag-like
delivery: a surface event creates a persisted :class:`SurfaceSession`, the agent
works in the background, and progress / final answers are posted back to the
thread later. A ``session_id`` is the stable thread tying the surface event → the
``agent_task`` workflow instance → the progress message → the final answer.

The session persists its **surface target** (where to deliver) and its **delivery
state** (which messages were already posted) so the runner can keep delivery
idempotent: a duplicate Slack event or a retry reuses the recorded message ids
rather than posting a second final answer (full crash recovery is the Slice 6
reconciler's job). Like the other stores it is a Protocol with an in-memory
default and a Postgres implementation (``surface_sessions`` table).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

# Lifecycle of a session. Mirrors the workflow statuses where it makes sense but
# is its own thing — a session can be `waiting` (parked on an approval) while its
# first turn's workflow already `completed`.
#   queued     — created, not started
#   running    — the agent is working the turn
#   waiting    — parked (e.g. on a human approval) until an event wakes it
#   completed  — final answer delivered
#   failed     — the turn errored; an error notice is delivered
#   abandoned  — interrupted inside a non-resumable model step (never replayed)
TERMINAL = ("completed", "failed", "abandoned")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class SurfaceTarget:
    """Where a session's output is delivered — the surface-addressing tuple.

    Surface-agnostic on purpose: ``channel`` / ``thread`` / ``event_id`` are
    generic ids a concrete :class:`~openloop.sessions.delivery.SurfaceDelivery`
    interprets (for Slack: channel id, thread_ts, the triggering event ts).
    """

    surface: str
    workspace: str
    agent: str
    channel: str | None = None
    thread: str | None = None
    # The id of the surface event/message that initiated the session. Used to
    # dedupe duplicate deliveries of the same inbound event.
    event_id: str | None = None


@dataclass(slots=True)
class SurfaceSession:
    """A persisted user-visible task: its target, workflow, and delivery state."""

    id: str
    target: SurfaceTarget
    status: str = "queued"
    workflow_instance_id: str | None = None
    progress_message_id: str | None = None
    final_message_id: str | None = None
    # Approvals the turn is parked on (Slice 4 maps a button click back here).
    approval_ids: list[str] = field(default_factory=list)
    result_summary: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@runtime_checkable
class SurfaceSessionStore(Protocol):
    async def get(self, session_id: str) -> SurfaceSession | None: ...

    async def get_by_event(self, event_id: str) -> SurfaceSession | None: ...

    async def get_by_approval(self, approval_id: str) -> SurfaceSession | None: ...

    async def upsert(self, session: SurfaceSession) -> None: ...

    async def recent(self, limit: int = 100) -> list[SurfaceSession]: ...


class InMemorySurfaceSessionStore:
    """Process-local sessions — good for dev and tests (not crash-durable)."""

    def __init__(self) -> None:
        self._by_id: dict[str, SurfaceSession] = {}

    async def get(self, session_id: str) -> SurfaceSession | None:
        return self._by_id.get(session_id)

    async def get_by_event(self, event_id: str) -> SurfaceSession | None:
        if not event_id:
            return None
        # Most-recent-first so a re-created session for the same event wins.
        for session in sorted(
            self._by_id.values(), key=lambda s: s.created_at, reverse=True
        ):
            if session.target.event_id == event_id:
                return session
        return None

    async def get_by_approval(self, approval_id: str) -> SurfaceSession | None:
        if not approval_id:
            return None
        for session in sorted(
            self._by_id.values(), key=lambda s: s.updated_at, reverse=True
        ):
            if approval_id in session.approval_ids:
                return session
        return None

    async def upsert(self, session: SurfaceSession) -> None:
        existing = self._by_id.get(session.id)
        if existing is not None:
            session.created_at = existing.created_at
        session.updated_at = _now()
        self._by_id[session.id] = session

    async def recent(self, limit: int = 100) -> list[SurfaceSession]:
        ordered = sorted(
            self._by_id.values(), key=lambda s: s.updated_at, reverse=True
        )
        return ordered[:limit]
