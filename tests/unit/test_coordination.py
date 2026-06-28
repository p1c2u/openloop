"""Unit tests for cross-process coordination (open hardening item #2).

Covers the distributed-lock contract: fenced acquire/release, the ``guard``
context manager's leader/contended behaviour, and — via a shared fake Redis — the
cross-replica guarantee that only one of two processes leads a piece of work.
"""

import asyncio

import pytest

from openloop.coordination import InProcessLock, PostgresLock, RedisLock, guard

pytestmark = pytest.mark.unit


# --- InProcessLock --------------------------------------------------------

async def test_in_process_lock_is_exclusive_then_releasable():
    lock = InProcessLock()

    first = await lock.acquire("k", ttl_seconds=10)
    assert first is not None
    # A second acquire while held is refused (even within one process).
    assert await lock.acquire("k", ttl_seconds=10) is None

    assert await lock.release("k", first) is True
    # Released → acquirable again, with a fresh token.
    second = await lock.acquire("k", ttl_seconds=10)
    assert second is not None and second != first


async def test_in_process_release_is_fenced_to_the_owner():
    lock = InProcessLock()
    token = await lock.acquire("k", ttl_seconds=10)

    # A stale/foreign token must not free a lock it no longer owns.
    assert await lock.release("k", "not-the-token") is False
    assert await lock.acquire("k", ttl_seconds=10) is None  # still held
    assert await lock.release("k", token) is True


async def test_distinct_keys_do_not_contend():
    lock = InProcessLock()
    assert await lock.acquire("a", ttl_seconds=10) is not None
    assert await lock.acquire("b", ttl_seconds=10) is not None


# --- guard context manager ------------------------------------------------

async def test_guard_yields_leadership_and_releases_on_exit():
    lock = InProcessLock()

    async with guard(lock, "recovery", ttl_seconds=10) as leader:
        assert leader is True
        # A concurrent guard on the same key is not the leader.
        async with guard(lock, "recovery", ttl_seconds=10) as contender:
            assert contender is False

    # Released on exit → the key can be led again.
    async with guard(lock, "recovery", ttl_seconds=10) as leader_again:
        assert leader_again is True


async def test_guard_releases_even_when_body_raises():
    lock = InProcessLock()
    with pytest.raises(ValueError):
        async with guard(lock, "recovery", ttl_seconds=10) as leader:
            assert leader is True
            raise ValueError("boom")
    # The lock was freed despite the error.
    assert await lock.acquire("recovery", ttl_seconds=10) is not None


# --- RedisLock over a fake client ----------------------------------------

class FakeRedis:
    """Minimal async Redis stand-in: SET NX + EVAL compare-and-delete.

    Shared between two RedisLock instances it models two processes against one
    Redis. TTL/px is recorded but not expired (tests don't exercise expiry).
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.pinged = False
        self.closed = False

    async def set(self, name, value, *, nx=False, px=None, **_kw):
        if nx and name in self.store:
            return None
        self.store[name] = value
        return True

    async def eval(self, script, numkeys, *args):
        keys = args[:numkeys]
        argv = args[numkeys:]
        if self.store.get(keys[0]) != argv[0]:
            return 0  # not the owner → fenced operations are no-ops
        if "pexpire" in script:
            return 1  # renew: ownership confirmed (no real expiry modelled here)
        del self.store[keys[0]]  # release
        return 1

    async def ping(self):
        self.pinged = True
        return True

    async def aclose(self):
        self.closed = True


async def test_redis_lock_acquire_and_fenced_release():
    client = FakeRedis()
    lock = RedisLock(client, namespace="t:")

    token = await lock.acquire("k", ttl_seconds=1)
    assert token is not None
    # Stored under the namespaced key (NX) so a second acquire is refused.
    assert client.store["t:k"] == token
    assert await lock.acquire("k", ttl_seconds=1) is None

    # A foreign token doesn't delete (the Lua compare guards it); the owner does.
    assert await lock.release("k", "stale") is False
    assert await lock.release("k", token) is True
    assert "t:k" not in client.store


async def test_redis_lock_two_replicas_one_leader():
    # Two RedisLock instances over a SHARED Redis = two replicas. Exactly one wins.
    client = FakeRedis()
    replica_a = RedisLock(client)
    replica_b = RedisLock(client)

    a = await replica_a.acquire("startup-recovery", ttl_seconds=600)
    b = await replica_b.acquire("startup-recovery", ttl_seconds=600)
    assert (a is None) != (b is None)  # exactly one acquired

    # After the leader releases, the other replica can take it (e.g. next boot).
    winner, winner_token = (replica_a, a) if a is not None else (replica_b, b)
    await winner.release("startup-recovery", winner_token)
    assert await replica_b.acquire("startup-recovery", ttl_seconds=600) is not None


async def test_redis_lock_setup_pings_and_close_closes():
    client = FakeRedis()
    lock = RedisLock(client)
    await lock.setup()
    assert client.pinged is True
    await lock.close()
    assert client.closed is True


# --- lease renewal (Finding: TTL vs long sweeps) -------------------------

async def test_in_process_renew_is_owner_fenced():
    lock = InProcessLock()
    token = await lock.acquire("k", ttl_seconds=10)

    assert await lock.renew("k", token, ttl_seconds=10) is True
    assert await lock.renew("k", "stale", ttl_seconds=10) is False
    await lock.release("k", token)
    # Nothing to renew once released.
    assert await lock.renew("k", token, ttl_seconds=10) is False


async def test_redis_lock_renew_extends_only_for_owner():
    client = FakeRedis()
    lock = RedisLock(client, namespace="t:")
    token = await lock.acquire("k", ttl_seconds=1)

    assert await lock.renew("k", token, ttl_seconds=5) is True
    # A foreign token can't extend the lease.
    assert await lock.renew("k", "stale", ttl_seconds=5) is False
    # Still owned (the key survives a renew).
    assert await lock.acquire("k", ttl_seconds=1) is None


class _RenewCountingLock:
    """Lock double that records renew calls (acquire always wins)."""

    def __init__(self) -> None:
        self.renews = 0
        self.released = False

    async def acquire(self, key, *, ttl_seconds):
        return "tok"

    async def release(self, key, token):
        self.released = True
        return True

    async def renew(self, key, token, *, ttl_seconds):
        self.renews += 1
        return True


async def test_guard_renews_lease_while_held_then_stops():
    # A long-running guarded body keeps the lease alive (renews fire), and the
    # renewer is cancelled on exit so it stops once released.
    lock = _RenewCountingLock()

    async with guard(lock, "k", ttl_seconds=10, renew_interval=0.001) as leader:
        assert leader is True
        for _ in range(500):  # bounded wait for a few renewals
            if lock.renews >= 3:
                break
            await asyncio.sleep(0.001)
        assert lock.renews >= 3

    assert lock.released is True
    # Renewals stop after the guard exits.
    settled = lock.renews
    await asyncio.sleep(0.01)
    assert lock.renews == settled


async def test_guard_without_renew_interval_does_not_renew():
    lock = _RenewCountingLock()
    async with guard(lock, "k", ttl_seconds=10) as leader:
        assert leader is True
        await asyncio.sleep(0.01)
    assert lock.renews == 0


# --- PostgresLock key hashing (no DB) ------------------------------------

def test_postgres_lock_id_is_deterministic_and_in_bigint_range():
    a = PostgresLock._lock_id("startup-recovery")
    assert a == PostgresLock._lock_id("startup-recovery")  # stable across calls
    assert -(2**63) <= a < 2**63  # fits the signed bigint pg_advisory_lock takes
    assert PostgresLock._lock_id("other-key") != a
