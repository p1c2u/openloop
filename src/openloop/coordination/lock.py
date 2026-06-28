"""Cross-process coordination — a distributed lock for multi-replica deploys.

The runtime's recovery paths (the workflow engine's ``resume_incomplete``, the
coding worker's reconciler, and the surface-session reconciler) are *idempotent*
but were written for a single process: if two replicas boot together they each
sweep the same shared Postgres rows and redo the same work, racing in the narrow
delivery windows. Correctness still holds (delivery is id- and key-guarded), but
the duplicated work and the wider race are wasteful.

A :class:`DistributedLock` lets exactly one replica lead a piece of work. Two
implementations, chosen like the other backends:

* :class:`InProcessLock` — process-local, the default. Correct for a single
  replica (and dev/tests): there is no other process to contend with, so the
  leader is always *this* process. It does **not** coordinate across processes.
* :class:`RedisLock` — a real cross-process lock (Redis ``SET NX PX`` with a
  fenced, compare-and-delete release) for running more than one replica.

The lock is a *coordination* layer, not a correctness one: a TTL expiry or a
fallback to :class:`InProcessLock` at worst lets a second replica redo idempotent
work — it never corrupts state.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class DistributedLock(Protocol):
    async def acquire(self, key: str, *, ttl_seconds: float) -> str | None:
        """Try to take ``key``; return an owner token, or ``None`` if held.

        The token fences the release: only the holder that acquired it can free
        the lock, so a TTL-expired-then-reacquired lock isn't released by a stale
        owner.
        """
        ...

    async def release(self, key: str, token: str) -> bool:
        """Release ``key`` iff ``token`` still owns it. Returns whether it did."""
        ...

    async def renew(self, key: str, token: str, *, ttl_seconds: float) -> bool:
        """Extend ``key``'s TTL iff ``token`` still owns it. Returns whether it did.

        Lets a holder keep a short-TTL lock alive across long work by renewing
        periodically: a live holder retains it indefinitely, while a dead one's
        lock still expires within one TTL.
        """
        ...


async def _renew_loop(
    lock: DistributedLock, key: str, token: str, ttl_seconds: float, interval: float
) -> None:
    """Re-extend ``key``'s lease every ``interval`` seconds until cancelled.

    Stops if the lease is lost (a renew that returns ``False`` — the lock expired
    or was taken): there is nothing left to extend, and continuing to renew a key
    we no longer own could stamp on the new holder.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            if not await lock.renew(key, token, ttl_seconds=ttl_seconds):
                logger.warning(
                    "lost lease on lock %r mid-pass (it expired or was taken)", key
                )
                return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a transient renew failure isn't fatal
            logger.warning("failed to renew lock %r; it may expire", key, exc_info=True)


@asynccontextmanager
async def guard(
    lock: DistributedLock,
    key: str,
    *,
    ttl_seconds: float,
    renew_interval: float | None = None,
) -> AsyncIterator[bool]:
    """Run a block as the leader for ``key``; yields whether we acquired it.

    Always releases on exit (best-effort — a failed release just waits out the
    TTL). Pass ``renew_interval`` to hold a short-TTL lock across long work: while
    the body runs, a background task re-extends the lease every ``renew_interval``
    seconds, so a live leader keeps the lock no matter how long the work takes
    while a dead one's lock still expires within one ``ttl_seconds``. Callers gate
    their work on the yielded flag::

        async with guard(lock, "startup-recovery", ttl_seconds=60,
                         renew_interval=20) as leader:
            if leader:
                await do_recovery()  # may run far longer than the TTL
    """
    token = await lock.acquire(key, ttl_seconds=ttl_seconds)
    renewer: asyncio.Task | None = None
    try:
        if token is not None and renew_interval:
            renewer = asyncio.create_task(
                _renew_loop(lock, key, token, ttl_seconds, renew_interval)
            )
        yield token is not None
    finally:
        if renewer is not None:
            renewer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renewer
        if token is not None:
            try:
                await lock.release(key, token)
            except Exception:  # noqa: BLE001 — TTL is the backstop; never raise here
                logger.warning("failed to release lock %r; it will expire", key,
                               exc_info=True)


class InProcessLock:
    """Process-local lock — correct within one process, not across them.

    Models single-replica semantics: this process owns everything, so a first
    acquire of a free key wins and a concurrent re-acquire is refused (so even two
    coroutines racing startup recovery in one process serialize). TTL is ignored —
    if the process dies the lock dies with it. Use :class:`RedisLock` for >1
    replica.
    """

    def __init__(self) -> None:
        self._held: dict[str, str] = {}

    async def acquire(self, key: str, *, ttl_seconds: float) -> str | None:
        if key in self._held:
            return None
        token = uuid.uuid4().hex
        self._held[key] = token
        return token

    async def release(self, key: str, token: str) -> bool:
        if self._held.get(key) == token:
            del self._held[key]
            return True
        return False

    async def renew(self, key: str, token: str, *, ttl_seconds: float) -> bool:
        # No expiry to extend within one process; just confirm we still own it.
        return self._held.get(key) == token


class RedisLock:
    """Cross-process lock over Redis: ``SET key token NX PX ttl`` to acquire, a
    compare-and-delete Lua script to release only if we still own it.

    Holds a duck-typed async Redis client (``redis.asyncio``), so it unit-tests
    against a fake without the dependency installed. Keys are namespaced to keep
    the runtime's locks clear of other Redis users.
    """

    # Atomic fenced release: delete only when the stored token is still ours, so a
    # holder whose lock already expired (and was re-taken) can't free someone else.
    _RELEASE_SCRIPT = (
        "if redis.call('get', KEYS[1]) == ARGV[1] "
        "then return redis.call('del', KEYS[1]) else return 0 end"
    )
    # Atomic fenced renew: extend the TTL only while the token is still ours.
    _RENEW_SCRIPT = (
        "if redis.call('get', KEYS[1]) == ARGV[1] "
        "then return redis.call('pexpire', KEYS[1], ARGV[2]) else return 0 end"
    )

    def __init__(self, client, *, namespace: str = "openloop:lock:") -> None:
        self.client = client
        self.namespace = namespace

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "RedisLock":
        """Build from a Redis URL (imports ``redis`` lazily so it stays optional)."""
        import redis.asyncio as redis  # noqa: PLC0415 — optional dependency

        return cls(redis.from_url(url, decode_responses=True), **kwargs)

    async def setup(self) -> None:
        """Validate connectivity so the caller can fall back if Redis is down."""
        await self.client.ping()

    def _key(self, key: str) -> str:
        return f"{self.namespace}{key}"

    async def acquire(self, key: str, *, ttl_seconds: float) -> str | None:
        token = uuid.uuid4().hex
        ok = await self.client.set(
            self._key(key), token, nx=True, px=int(ttl_seconds * 1000)
        )
        return token if ok else None

    async def release(self, key: str, token: str) -> bool:
        result = await self.client.eval(self._RELEASE_SCRIPT, 1, self._key(key), token)
        return bool(result)

    async def renew(self, key: str, token: str, *, ttl_seconds: float) -> bool:
        result = await self.client.eval(
            self._RENEW_SCRIPT, 1, self._key(key), token, int(ttl_seconds * 1000)
        )
        return bool(result)

    async def close(self) -> None:
        # redis-py 5 uses aclose(); older builds use close(). Tolerate both/none.
        closer = getattr(self.client, "aclose", None) or getattr(self.client, "close", None)
        if closer is not None:
            await closer()
