"""Per-request usage log in Postgres. One row per /chat/stream call.

Source of truth for the /usage/* endpoints. Cost comes from
:func:`app.pricing.cost_usd` so the math stays consistent everywhere.
"""
from __future__ import annotations

import os

import psycopg

from .pricing import cost_usd

USAGE_LOG_DDL = """
CREATE TABLE IF NOT EXISTS usage_log (
    request_id    UUID         PRIMARY KEY,
    api_key       TEXT         NOT NULL,
    model         TEXT         NOT NULL,
    input_tokens  INT          NOT NULL,
    output_tokens INT          NOT NULL,
    cost_usd      NUMERIC(12, 6) NOT NULL,
    latency_ms    INT          NOT NULL,
    ttft_ms       INT,
    cache_hit     BOOLEAN      NOT NULL,
    fallback_used BOOLEAN      NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS usage_log_api_key_created_at
    ON usage_log (api_key, created_at DESC);
"""


async def ensure_usage_log() -> None:
    async with await psycopg.AsyncConnection.connect(os.environ["DB_URL"]) as conn:
        async with conn.cursor() as cur:
            await cur.execute(USAGE_LOG_DDL)
        await conn.commit()


async def log_usage(
    *,
    request_id: str,
    api_key: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    ttft_ms: int | None,
    cache_hit: bool,
    fallback_used: bool,
) -> None:
    """Insert a single usage row. Cost is computed from pricing.cost_usd."""
    cost = cost_usd(model, input_tokens, output_tokens)
    async with await psycopg.AsyncConnection.connect(os.environ["DB_URL"]) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO usage_log (
                    request_id, api_key, model, input_tokens, output_tokens,
                    cost_usd, latency_ms, ttft_ms, cache_hit, fallback_used
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    request_id, api_key, model, input_tokens, output_tokens,
                    cost, latency_ms, ttft_ms, cache_hit, fallback_used,
                ),
            )
        await conn.commit()


# ---------------------------------------------------------------------------
# Read paths used by /usage/today and /usage/breakdown.
# ---------------------------------------------------------------------------
async def fetch_today_totals(api_key: str) -> dict:
    """Headline numbers for one API key today: requests / tokens / cost."""
    async with await psycopg.AsyncConnection.connect(os.environ["DB_URL"]) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(SUM(input_tokens + output_tokens), 0),
                    COALESCE(SUM(cost_usd), 0)
                FROM usage_log
                WHERE api_key = %s AND created_at >= CURRENT_DATE
                """,
                (api_key,),
            )
            row = await cur.fetchone()
    requests, tokens, cost = row if row else (0, 0, 0)
    return {
        "requests": int(requests),
        "tokens": int(tokens),
        "cost_usd": round(float(cost), 6),
    }


async def fetch_today_breakdown(api_key: str) -> dict:
    """Per-model breakdown + headline cache / fallback / latency stats."""
    async with await psycopg.AsyncConnection.connect(os.environ["DB_URL"]) as conn:
        async with conn.cursor() as cur:
            # Per-model rollup.
            await cur.execute(
                """
                SELECT
                    model,
                    COUNT(*) AS requests,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(cost_usd), 0) AS cost_usd,
                    COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                    COALESCE(
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms),
                        0
                    ) AS p95_latency_ms
                FROM usage_log
                WHERE api_key = %s AND created_at >= CURRENT_DATE
                GROUP BY model
                ORDER BY requests DESC
                """,
                (api_key,),
            )
            by_model_rows = await cur.fetchall()
            # Headline aggregates (cache / fallback rate, overall latency).
            await cur.execute(
                """
                SELECT
                    COUNT(*) AS requests,
                    COALESCE(AVG(CASE WHEN cache_hit     THEN 1.0 ELSE 0.0 END), 0),
                    COALESCE(AVG(CASE WHEN fallback_used THEN 1.0 ELSE 0.0 END), 0),
                    COALESCE(AVG(latency_ms), 0),
                    COALESCE(
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms),
                        0
                    )
                FROM usage_log
                WHERE api_key = %s AND created_at >= CURRENT_DATE
                """,
                (api_key,),
            )
            agg = await cur.fetchone()

    by_model = [
        {
            "model": m,
            "requests": int(reqs),
            "input_tokens": int(inp),
            "output_tokens": int(out),
            "cost_usd": round(float(cost), 6),
            "avg_latency_ms": round(float(avg_lat), 1),
            "p95_latency_ms": round(float(p95_lat), 1),
        }
        for m, reqs, inp, out, cost, avg_lat, p95_lat in by_model_rows
    ]
    total_reqs, cache_rate, fb_rate, avg_lat, p95_lat = agg if agg else (0, 0, 0, 0, 0)
    return {
        "requests": int(total_reqs),
        "cache_hit_rate": round(float(cache_rate), 4),
        "fallback_rate": round(float(fb_rate), 4),
        "avg_latency_ms": round(float(avg_lat), 1),
        "p95_latency_ms": round(float(p95_lat), 1),
        "by_model": by_model,
    }
