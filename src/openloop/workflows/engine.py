"""A minimal durable-workflow engine (the Architecture "Later" runtime item).

A :class:`Workflow` is an ordered list of named :class:`Step`s. The engine runs
them in order, persisting the :class:`WorkflowInstance` after each one, so a crash
resumes from the last completed step. A step marked ``wait`` parks the
instance: the engine persists ``status="waiting"`` and returns, and a later
:meth:`WorkflowEngine.send_event` delivers the awaited event and drives the rest.

This is how approval stops being a special case in ``ToolGateway.resolve``:
approval is just a wait node, and ``resolve`` becomes a thin adapter that emits
the approval event. Steps must be **idempotent** — a crash between a step's side
effect and its checkpoint write means the step re-runs on resume.

Wakeups are in-process today (``send_event`` drives synchronously, and a startup
reconciler re-drives anything left running). Redis pub/sub can later make wakeups
cross-process without changing the workflow contract.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from openloop.workflows.store import (
    TERMINAL,
    WorkflowInstance,
    WorkflowStore,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorkflowContext:
    """What a step sees: the live instance (mutate ``state`` / ``result``)."""

    instance: WorkflowInstance

    @property
    def state(self) -> dict:
        return self.instance.state


StepFn = Callable[[WorkflowContext], Awaitable[None]]


@dataclass(slots=True)
class Step:
    """One named step. ``wait`` nodes park the instance until an event arrives.

    ``resumable=False`` marks a step that must not be replayed (e.g. a chat turn's
    non-idempotent model call). A crash is recoverable only once every
    non-resumable step has completed; before that the instance is abandoned rather
    than re-driven into the non-resumable step. Idempotent steps stay resumable.
    """

    name: str
    run: StepFn | None = None
    wait: bool = False
    resumable: bool = True


@dataclass(slots=True)
class Workflow:
    name: str
    steps: list[Step]


class WorkflowEngine:
    """Runs workflows durably: checkpoint per step, park on wait, resume on event."""

    def __init__(
        self, store: WorkflowStore, workflows: dict[str, Workflow] | None = None
    ) -> None:
        self.store = store
        self.workflows = workflows or {}

    def register(self, workflow: Workflow) -> None:
        self.workflows[workflow.name] = workflow

    async def checkpoint(self, instance: WorkflowInstance) -> None:
        """Persist mid-step state (e.g. after an idempotent write inside a step)."""
        await self.store.upsert(instance)

    async def start(
        self, workflow: str, instance_id: str, initial_state: dict
    ) -> WorkflowInstance:
        """Create a new instance and drive it to its first park/terminal.

        Idempotent on the instance id: if one already exists it is returned
        as-is, never re-driven — that could replay a non-resumable step. Resuming
        is the job of :meth:`send_event` (waits) and :meth:`resume_incomplete`
        (crashes), which both apply the resumability rules.
        """
        existing = await self.store.get(instance_id)
        if existing is not None:
            return existing
        instance = WorkflowInstance(
            id=instance_id, workflow=workflow, state=dict(initial_state)
        )
        await self.store.upsert(instance)
        return await self._drive(instance)

    async def send_event(
        self, instance_id: str, event: str, payload: dict | None = None
    ) -> WorkflowInstance | None:
        """Deliver an awaited event, waking a parked instance and driving on."""
        instance = await self.store.get(instance_id)
        if instance is None:
            return None
        if instance.status != "waiting" or instance.waiting_on != event:
            # Idempotent: the event was already consumed, or the instance moved
            # past this wait (e.g. a double-approve). Nothing to do.
            return instance
        instance.state.setdefault("events", {})[event] = payload or {}
        instance.completed_steps.append(event)
        instance.status = "running"
        instance.waiting_on = None
        await self.store.upsert(instance)
        return await self._drive(instance)

    async def cancel(self, instance_id: str, reason: str = "") -> WorkflowInstance | None:
        """Cancel a parked/running instance (e.g. its approval was denied)."""
        instance = await self.store.get(instance_id)
        if instance is None or instance.status in TERMINAL:
            return instance
        instance.status = "cancelled"
        instance.waiting_on = None
        instance.error = reason or None
        await self.store.upsert(instance)
        return instance

    async def resume_incomplete(self) -> list[str]:
        """Re-drive instances left ``running`` by a crash. Call once at startup.

        ``waiting`` instances stay parked (their event hasn't arrived);
        ``completed`` / ``failed`` are terminal. Idempotent; across replicas the
        app lifespan runs it under a ``startup-recovery``
        :class:`~openloop.coordination.DistributedLock` so only the leader sweeps.
        """
        resumed: list[str] = []
        for instance in await self.store.recent(limit=1000):
            if instance.status != "running":
                continue
            workflow = self.workflows.get(instance.workflow)
            if workflow is None:
                # Its workflow isn't registered in this process; leave it be.
                continue
            if _has_pending_non_resumable_step(workflow, instance):
                # A non-resumable step (e.g. a chat turn's model call) hasn't
                # completed — resuming would replay it. Abandon instead.
                instance.status = "abandoned"
                instance.error = "interrupted before a non-resumable step completed"
                await self.store.upsert(instance)
                continue
            logger.info("resuming workflow %s (%s)", instance.id, instance.workflow)
            await self._drive(instance)
            resumed.append(instance.id)
        return resumed

    async def _drive(self, instance: WorkflowInstance) -> WorkflowInstance:
        """Run steps from where the instance left off, checkpointing each."""
        workflow = self.workflows.get(instance.workflow)
        if workflow is None:
            raise KeyError(f"unknown workflow {instance.workflow!r}")
        if instance.status in TERMINAL:
            return instance

        for step in workflow.steps:
            if step.name in instance.completed_steps:
                continue
            if step.wait:
                instance.status = "waiting"
                instance.waiting_on = step.name
                await self.store.upsert(instance)
                return instance
            ctx = WorkflowContext(instance)
            try:
                assert step.run is not None
                await step.run(ctx)
            except Exception as exc:  # noqa: BLE001 — record failure, don't crash caller
                instance.status = "failed"
                instance.error = str(exc)
                await self.store.upsert(instance)
                logger.exception("workflow %s failed at step %s", instance.id, step.name)
                return instance
            instance.completed_steps.append(step.name)
            await self.store.upsert(instance)  # checkpoint after each step

        instance.status = "completed"
        await self.store.upsert(instance)
        return instance


def _has_pending_non_resumable_step(
    workflow: Workflow, instance: WorkflowInstance
) -> bool:
    """True if a non-resumable step has not yet completed.

    Once every non-resumable step is done, only idempotent steps remain and the
    instance is safe to re-drive on resume.
    """
    done = set(instance.completed_steps)
    return any(
        not step.resumable and step.name not in done for step in workflow.steps
    )
