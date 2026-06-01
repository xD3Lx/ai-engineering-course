"""Per-API-key token bucket in Redis (reserve-and-settle).

A permissive pre-check (GET, then INCR after) loses under concurrency: every
parallel request sees ``used < limit`` at the same instant and passes. So we
reserve an expected token cost up front atomically and settle to real usage
after the model finishes. The expected cost starts conservative, then uses the
key's observed average request size once the key has completed traffic::

    1) reserve:  INCRBY est  (+ EXPIRE on first write)
                  if new_val > limit -> DECRBY est, return 429 + Retry-After
    2) settle:   delta = actual - est
                  INCRBY delta   (or DECRBY -delta) once usage arrives
    3) abort:    DECRBY est       (full refund if the request didn't complete)

Only INCRBY / DECRBY / EXPIRE / MGET / PTTL — works on Upstash REST."""
from __future__ import annotations

import redis.asyncio as aioredis

WINDOW_SECONDS = 60  # bucket TTL — fully refills this long after the first write
AVERAGE_WINDOW_SECONDS = WINDOW_SECONDS


def quota_key(api_key: str) -> str:
    return f"quota:{api_key}"


def average_tokens_key(api_key: str) -> str:
    return f"quota_avg_tokens:{api_key}"


def average_requests_key(api_key: str) -> str:
    return f"quota_avg_requests:{api_key}"


def estimate_tokens(
    messages: list[dict],
    token_limit: int,
    observed_average_tokens: int | None = None,
) -> int:
    """Estimate the upfront reservation for a request.

    The first request for a key uses a conservative worst-case hold. Once the
    key has completed requests in the current window, reserve the observed
    average instead, still bounded by the prompt estimate and hard max output.
    """
    prompt_chars = sum(len(m["content"]) for m in messages)
    input_estimate = max(1, prompt_chars // 4)
    output_estimate = min(1024, token_limit)
    worst_case = input_estimate + output_estimate
    if observed_average_tokens is None:
        return worst_case
    expected = max(input_estimate, observed_average_tokens)
    return min(worst_case, expected)


async def observed_average_tokens(redis: aioredis.Redis, api_key: str) -> int | None:
    """Return average real tokens/request for this key's recent completions."""
    raw_tokens, raw_requests = await redis.mget(
        average_tokens_key(api_key),
        average_requests_key(api_key),
    )
    if raw_tokens is None or raw_requests is None:
        return None
    tokens = int(raw_tokens)
    requests = int(raw_requests)
    if tokens <= 0 or requests <= 0:
        return None
    return max(1, tokens // requests)


async def retry_after_seconds(redis: aioredis.Redis, api_key: str) -> int:
    """Return a client-safe Retry-After value for the current quota window."""
    ttl_ms = await redis.pttl(quota_key(api_key))
    if ttl_ms >= 0:
        return max(1, (ttl_ms + 999) // 1000)
    if ttl_ms == -2:
        return 1
    return WINDOW_SECONDS


async def ensure_expiring_key(
    redis: aioredis.Redis, key: str, seconds: int
) -> bool:
    """Return False when ``key`` is gone; add expiry to stale immortal keys."""
    ttl_ms = await redis.pttl(key)
    if ttl_ms == -2:
        return False
    if ttl_ms == -1:
        await redis.expire(key, seconds)
    return True


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
    else:
        await ensure_expiring_key(redis, quota_key(api_key), WINDOW_SECONDS)
    if new_val > limit:
        # Over budget — give the reservation back so we don't waste it.
        await redis.decrby(quota_key(api_key), tokens)
        return await retry_after_seconds(redis, api_key)
    return 0


async def quota_settle(
    redis: aioredis.Redis, api_key: str, reservation: int, actual: int
) -> None:
    """Reconcile a reservation with the real token count once usage arrives."""
    delta = actual - reservation
    quota_exists = await ensure_expiring_key(redis, quota_key(api_key), WINDOW_SECONDS)
    if quota_exists:
        if delta > 0:
            await redis.incrby(quota_key(api_key), delta)
        elif delta < 0:
            await redis.decrby(quota_key(api_key), -delta)
    if actual > 0:
        total = await redis.incrby(average_tokens_key(api_key), actual)
        requests = await redis.incrby(average_requests_key(api_key), 1)
        if total == actual:
            await redis.expire(average_tokens_key(api_key), AVERAGE_WINDOW_SECONDS)
        else:
            await ensure_expiring_key(
                redis, average_tokens_key(api_key), AVERAGE_WINDOW_SECONDS
            )
        if requests == 1:
            await redis.expire(average_requests_key(api_key), AVERAGE_WINDOW_SECONDS)
        else:
            await ensure_expiring_key(
                redis, average_requests_key(api_key), AVERAGE_WINDOW_SECONDS
            )


async def quota_refund(redis: aioredis.Redis, api_key: str, tokens: int) -> None:
    """Hand a reservation back unspent (used when the request aborts)."""
    if tokens > 0 and await ensure_expiring_key(
        redis, quota_key(api_key), WINDOW_SECONDS
    ):
        await redis.decrby(quota_key(api_key), tokens)
