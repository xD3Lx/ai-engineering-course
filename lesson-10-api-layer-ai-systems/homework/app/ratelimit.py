"""Per-API-key token bucket in Redis (reserve-and-settle).

A permissive pre-check (GET, then INCR after) loses under concurrency: every
parallel request sees ``used < limit`` at the same instant and passes. So we
reserve a worst-case estimate up front atomically and settle to real usage
after the model finishes::

    1) reserve:  INCRBY est  (+ EXPIRE on first write)
                  if new_val > limit -> DECRBY est, return 429 + Retry-After
    2) settle:   delta = actual - est
                  INCRBY delta   (or DECRBY -delta) once usage arrives
    3) abort:    DECRBY est       (full refund if the request didn't complete)

Only INCRBY / DECRBY / EXPIRE / GET / TTL — works on Upstash REST."""
from __future__ import annotations

import redis.asyncio as aioredis

WINDOW_SECONDS = 180  # bucket TTL — fully refills this long after the first write


def quota_key(api_key: str) -> str:
    return f"quota:{api_key}"


def estimate_tokens(messages: list[dict], token_limit: int) -> int:
    """Worst-case upfront reservation: ~4 chars/token input + max_tokens output."""
    prompt_chars = sum(len(m["content"]) for m in messages)
    input_estimate = max(1, prompt_chars // 4)
    output_estimate = min(1024, token_limit)
    return input_estimate + output_estimate


async def quota_reserve(
    redis: aioredis.Redis, api_key: str, tokens: int, limit: int
) -> int:
    """Atomically reserve ``tokens``. Return 0 if reserved, else Retry-After
    seconds. INCRBY serializes concurrent reservations so even parallel bursts
    get capped correctly."""
    new_val = await redis.incrby(quota_key(api_key), tokens)
    if new_val == tokens:
        # First write of a fresh window — start the refill timer.
        await redis.expire(quota_key(api_key), WINDOW_SECONDS)
    if new_val > limit:
        # Over budget — give the reservation back so we don't waste it.
        await redis.decrby(quota_key(api_key), tokens)
        ttl = await redis.ttl(quota_key(api_key))
        return ttl if ttl > 0 else WINDOW_SECONDS
    return 0


async def quota_settle(
    redis: aioredis.Redis, api_key: str, reservation: int, actual: int
) -> None:
    """Reconcile a reservation with the real token count once usage arrives."""
    delta = actual - reservation
    if delta > 0:
        await redis.incrby(quota_key(api_key), delta)
    elif delta < 0:
        await redis.decrby(quota_key(api_key), -delta)


async def quota_refund(redis: aioredis.Redis, api_key: str, tokens: int) -> None:
    """Hand a reservation back unspent (used when the request aborts)."""
    if tokens > 0:
        await redis.decrby(quota_key(api_key), tokens)
