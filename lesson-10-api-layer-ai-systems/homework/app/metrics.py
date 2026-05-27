"""Global counters in a single Redis hash so they survive restarts and are
shared across instances. ``HINCRBY`` is atomic — no read-modify-write needed."""
from __future__ import annotations

import redis.asyncio as aioredis

METRICS_KEY = "metrics:global"


async def metric_incr(redis: aioredis.Redis, name: str, by: int = 1) -> None:
    await redis.hincrby(METRICS_KEY, name, by)


async def metrics_snapshot(redis: aioredis.Redis) -> dict[str, int]:
    """Return every counter as ``name -> int``. Missing fields are absent;
    callers should ``.get(name, 0)`` when reading specific fields."""
    raw = await redis.hgetall(METRICS_KEY)
    return {k: int(v) for k, v in raw.items()}
