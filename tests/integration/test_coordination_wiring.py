"""Integration: the app wires a coordination lock and degrades it gracefully."""

import asyncio
import contextlib
import logging

import pytest
from fastapi.testclient import TestClient

from openloop import app as appmod
from openloop.config import Settings
from openloop.coordination import InProcessLock, PostgresLock, RedisLock

pytestmark = pytest.mark.integration


class _SpyEngine:
    def __init__(self) -> None:
        self.calls = 0

    async def resume_incomplete(self):
        self.calls += 1
        return []


class _SpyTools:
    def __init__(self, engine) -> None:
        self.engine = engine
        self._tools: dict = {}  # _resume_worker_jobs looks up "coding_worker" here


class _SpyRunner:
    def __init__(self) -> None:
        self.calls = 0

    async def reconcile(self):
        self.calls += 1
        return []


def test_build_lock_auto_follows_memory_backend():
    # auto + in-memory stores → process-local; auto + postgres → advisory lock
    # (reuses the DB the deploy already runs, no extra service).
    auto_mem = appmod.build_lock(Settings(lock_backend="auto", memory_backend="memory"))
    auto_pg = appmod.build_lock(Settings(lock_backend="auto", memory_backend="postgres"))
    assert isinstance(auto_mem, InProcessLock)
    assert isinstance(auto_pg, PostgresLock)


def test_build_lock_explicit_overrides_memory_backend():
    forced_mem = appmod.build_lock(
        Settings(lock_backend="memory", memory_backend="postgres")
    )
    forced_pg = appmod.build_lock(
        Settings(lock_backend="postgres", memory_backend="memory")
    )
    assert isinstance(forced_mem, InProcessLock)
    assert isinstance(forced_pg, PostgresLock)


class _FailingSetupLock:
    async def setup(self):
        raise RuntimeError("coordination backend down")

    async def close(self):
        pass


async def test_explicit_backend_setup_failure_logs_loudly(caplog):
    # An operator who asked for cross-process coordination must SEE it was disabled.
    lock = _FailingSetupLock()
    with caplog.at_level(logging.ERROR):
        resolved = await appmod._setup_coordination(lock, Settings(lock_backend="redis"))
    assert isinstance(resolved, InProcessLock)
    assert "CROSS-PROCESS COORDINATION DISABLED" in caplog.text


async def test_auto_backend_setup_failure_degrades_quietly(caplog):
    # auto wasn't an explicit coordination request → quiet degrade, no loud banner.
    lock = _FailingSetupLock()
    with caplog.at_level(logging.ERROR):
        resolved = await appmod._setup_coordination(lock, Settings(lock_backend="auto"))
    assert isinstance(resolved, InProcessLock)
    assert "CROSS-PROCESS COORDINATION DISABLED" not in caplog.text


def test_build_lock_redis_missing_package_falls_back(monkeypatch):
    # lock_backend=redis but constructing the client raises (e.g. `redis` extra not
    # installed) → build_lock degrades to in-process instead of failing boot.
    def boom(url, **kw):
        raise ImportError("no redis")

    monkeypatch.setattr(RedisLock, "from_url", classmethod(lambda cls, url, **kw: boom(url)))
    lock = appmod.build_lock(Settings(lock_backend="redis", redis_url="redis://x"))
    assert isinstance(lock, InProcessLock)


class _PingFailRedis:
    async def ping(self):
        raise RuntimeError("redis down")

    async def aclose(self):
        pass


def test_lifespan_falls_back_when_redis_ping_fails(monkeypatch):
    # A configured-but-unreachable Redis must not break startup: the lifespan pings,
    # falls back to an in-process lock, and still runs recovery as the leader.
    monkeypatch.setattr(
        appmod, "build_lock", lambda settings: RedisLock(_PingFailRedis())
    )

    app = appmod.create_app()
    with TestClient(app):  # runs the lifespan → ping fails → fallback
        assert isinstance(app.state.coordinator, InProcessLock)


# --- recovery pass: leader runs, contended retries (Finding #1) ----------

async def test_recovery_pass_leads_and_runs_reconcilers():
    engine, runner = _SpyEngine(), _SpyRunner()
    tools = _SpyTools(engine)

    led = await appmod.run_recovery_pass(InProcessLock(), tools, runner)

    assert led is True
    assert engine.calls == 1 and runner.calls == 1


async def test_recovery_pass_skips_when_contended_but_retries_after_release():
    # A pass that can't get the lock must NOT run the reconcilers — and crucially,
    # a LATER pass leads once the lock frees (e.g. a crashed leader's TTL lapses),
    # so recovery is never permanently skipped (the Finding #1 fix).
    coordinator = InProcessLock()
    engine, runner = _SpyEngine(), _SpyRunner()
    tools = _SpyTools(engine)

    held = await coordinator.acquire("startup-recovery", ttl_seconds=120)
    assert held is not None

    skipped = await appmod.run_recovery_pass(coordinator, tools, runner)
    assert skipped is False
    assert engine.calls == 0 and runner.calls == 0

    # The holder goes away (crash + TTL expiry / release) → the next pass leads.
    await coordinator.release("startup-recovery", held)
    led = await appmod.run_recovery_pass(coordinator, tools, runner)
    assert led is True
    assert engine.calls == 1 and runner.calls == 1


async def test_recovery_loop_reruns_until_cancelled(monkeypatch):
    # The periodic backstop keeps re-running so a crashed leader is eventually
    # covered; it is cancellable for clean shutdown.
    calls = 0

    async def spy(coordinator, tools, session_runner):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(appmod, "run_recovery_pass", spy)

    task = asyncio.create_task(
        appmod._recovery_loop(InProcessLock(), object(), None, interval=0.001)
    )
    for _ in range(200):  # bounded wait for a few iterations
        if calls >= 3:
            break
        await asyncio.sleep(0.001)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert calls >= 3
