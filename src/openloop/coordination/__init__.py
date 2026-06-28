"""Cross-process coordination primitives (distributed locking)."""

from openloop.coordination.lock import (
    DistributedLock,
    InProcessLock,
    PostgresLock,
    RedisLock,
    guard,
)

__all__ = [
    "DistributedLock",
    "InProcessLock",
    "PostgresLock",
    "RedisLock",
    "guard",
]
