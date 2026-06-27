"""Postgres-backed surface sessions — async tasks survive a process restart.

Mirrors :class:`InMemorySurfaceSessionStore` against a ``surface_sessions``
table, following the approvals/usage/checkpoint/workflow store pattern. The
surface target is flattened into columns so the startup reconciler can query by
status without deserializing JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from openloop.sessions.store import SurfaceSession, SurfaceTarget


class PostgresSurfaceSessionStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool = None  # asyncpg.Pool, created in setup()

    async def setup(self) -> None:
        import asyncpg

        self._pool = await asyncpg.create_pool(self.dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS surface_sessions (
                    id                   TEXT PRIMARY KEY,
                    surface              TEXT NOT NULL,
                    workspace            TEXT NOT NULL,
                    agent                TEXT NOT NULL,
                    channel              TEXT,
                    thread               TEXT,
                    event_id             TEXT,
                    status               TEXT NOT NULL,
                    workflow_instance_id TEXT,
                    progress_message_id  TEXT,
                    final_message_id     TEXT,
                    approval_ids         JSONB NOT NULL DEFAULT '[]',
                    result_summary       TEXT,
                    error                TEXT,
                    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS surface_sessions_status_idx "
                "ON surface_sessions (status, updated_at DESC)"
            )
            # event_id is the idempotency key for an inbound surface event: a
            # partial unique index makes a second, concurrent delivery of the same
            # event fail the insert (the runner catches it and defers to the
            # winner) rather than silently creating a duplicate session.
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS surface_sessions_event_uniq "
                "ON surface_sessions (event_id) WHERE event_id IS NOT NULL"
            )
            # GIN index backs the `approval_ids ? $1` lookup (button → session).
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS surface_sessions_approval_idx "
                "ON surface_sessions USING GIN (approval_ids)"
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self):
        if self._pool is None:
            raise RuntimeError(
                "PostgresSurfaceSessionStore.setup() must be called first"
            )
        return self._pool

    async def get(self, session_id: str) -> SurfaceSession | None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM surface_sessions WHERE id = $1", session_id
            )
        return _row_to_session(row) if row else None

    async def get_by_event(self, event_id: str) -> SurfaceSession | None:
        if not event_id:
            return None
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM surface_sessions WHERE event_id = $1 "
                "ORDER BY created_at DESC LIMIT 1",
                event_id,
            )
        return _row_to_session(row) if row else None

    async def get_by_approval(self, approval_id: str) -> SurfaceSession | None:
        if not approval_id:
            return None
        pool = self._require_pool()
        async with pool.acquire() as conn:
            # `@>` (JSONB containment) tests that the array holds this id. Unlike
            # the `?` operator it takes a normal $1 param and is GIN-indexable.
            row = await conn.fetchrow(
                "SELECT * FROM surface_sessions WHERE approval_ids @> $1::jsonb "
                "ORDER BY updated_at DESC LIMIT 1",
                json.dumps([approval_id]),
            )
        return _row_to_session(row) if row else None

    async def upsert(self, session: SurfaceSession) -> None:
        pool = self._require_pool()
        t = session.target
        async with pool.acquire() as conn:
            # created_at is set once; updated_at always bumped to now().
            await conn.execute(
                """
                INSERT INTO surface_sessions (
                    id, surface, workspace, agent, channel, thread, event_id,
                    status, workflow_instance_id, progress_message_id,
                    final_message_id, approval_ids, result_summary, error,
                    updated_at
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14, now())
                ON CONFLICT (id) DO UPDATE SET
                    surface = EXCLUDED.surface,
                    workspace = EXCLUDED.workspace,
                    agent = EXCLUDED.agent,
                    channel = EXCLUDED.channel,
                    thread = EXCLUDED.thread,
                    event_id = EXCLUDED.event_id,
                    status = EXCLUDED.status,
                    workflow_instance_id = EXCLUDED.workflow_instance_id,
                    progress_message_id = EXCLUDED.progress_message_id,
                    final_message_id = EXCLUDED.final_message_id,
                    approval_ids = EXCLUDED.approval_ids,
                    result_summary = EXCLUDED.result_summary,
                    error = EXCLUDED.error,
                    updated_at = now()
                """,
                session.id,
                t.surface,
                t.workspace,
                t.agent,
                t.channel,
                t.thread,
                t.event_id,
                session.status,
                session.workflow_instance_id,
                session.progress_message_id,
                session.final_message_id,
                json.dumps(session.approval_ids),
                session.result_summary,
                session.error,
            )

    async def recent(self, limit: int = 100) -> list[SurfaceSession]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM surface_sessions ORDER BY updated_at DESC LIMIT $1",
                limit,
            )
        return [_row_to_session(r) for r in rows]


def _row_to_session(row) -> SurfaceSession:
    now = datetime.now(timezone.utc)
    return SurfaceSession(
        id=row["id"],
        target=SurfaceTarget(
            surface=row["surface"],
            workspace=row["workspace"],
            agent=row["agent"],
            channel=row["channel"],
            thread=row["thread"],
            event_id=row["event_id"],
        ),
        status=row["status"],
        workflow_instance_id=row["workflow_instance_id"],
        progress_message_id=row["progress_message_id"],
        final_message_id=row["final_message_id"],
        approval_ids=json.loads(row["approval_ids"]) if row["approval_ids"] else [],
        result_summary=row["result_summary"],
        error=row["error"],
        created_at=row["created_at"] or now,
        updated_at=row["updated_at"] or now,
    )
