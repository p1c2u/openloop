"""Durable workflow instances — the general form of Phase B's worker checkpoint.

A :class:`WorkflowInstance` is one running workflow: its position (which steps are
done, whether it is parked on a wait node), a JSON ``state`` snapshot, and a
terminal ``result`` / ``error``. The store persists it after every step so a
crash resumes from the last completed step, exactly like the worker checkpoint —
but for any workflow, not just the coding worker. Phase C generalizes the
checkpoint store into this; the worker becomes one workflow on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

# Terminal statuses never resume; "running" resumes on startup (crashed mid-step);
# "waiting" stays parked until an event wakes it. "abandoned" is a crashed run of
# a non-resumable workflow (e.g. a chat turn — we never replay paid model calls).
TERMINAL = ("completed", "failed", "cancelled", "abandoned")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class WorkflowInstance:
    """A persisted snapshot of one workflow run."""

    id: str
    workflow: str
    status: str = "running"  # running | waiting | completed | failed
    completed_steps: list[str] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    waiting_on: str | None = None  # name of the wait node it is parked at
    result: dict | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@runtime_checkable
class WorkflowStore(Protocol):
    async def get(self, instance_id: str) -> WorkflowInstance | None: ...

    async def upsert(self, instance: WorkflowInstance) -> None: ...

    async def recent(self, limit: int = 100) -> list[WorkflowInstance]: ...


class InMemoryWorkflowStore:
    """Process-local instances — good for dev and tests (not crash-durable)."""

    def __init__(self) -> None:
        self._by_id: dict[str, WorkflowInstance] = {}

    async def get(self, instance_id: str) -> WorkflowInstance | None:
        return self._by_id.get(instance_id)

    async def upsert(self, instance: WorkflowInstance) -> None:
        existing = self._by_id.get(instance.id)
        if existing is not None:
            instance.created_at = existing.created_at
        instance.updated_at = _now()
        self._by_id[instance.id] = instance

    async def recent(self, limit: int = 100) -> list[WorkflowInstance]:
        ordered = sorted(
            self._by_id.values(), key=lambda i: i.updated_at, reverse=True
        )
        return ordered[:limit]
