"""Semantic response cache backed by Qdrant.

Global cache (one collection for all API keys) — this is a public Q&A bot,
not private data. RAG and cache share the same embedding model so vectors
are directly comparable.

Qdrant has no built-in TTL, so we stamp ``expire_at`` (unix seconds) into the
payload and filter on it at query time. Stale points still sit in the
collection until something overwrites them; a periodic cleanup job is fine
but not required for correctness.
"""
from __future__ import annotations

import json
import time
import uuid

from fastapi import Request
from qdrant_client import AsyncQdrantClient, models as qmodels

import redis.asyncio as aioredis

from .metrics import metric_incr
from .usage_log import log_usage

CACHE_COLLECTION = "cache_collection"
CACHE_THRESHOLD = 0.92  # cosine similarity required to serve from cache
CACHE_TTL_SECONDS = 3600  # 1 hour


def _sse(payload: dict) -> str:
    """Local SSE encoder (kept here so the cache module is self-contained)."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def ensure_collection(qdrant: AsyncQdrantClient, embed_dim: int) -> None:
    """Create ``cache_collection`` and the expire_at payload index. Safe to
    call repeatedly — silently no-ops if the collection already exists."""
    existing = {c.name for c in (await qdrant.get_collections()).collections}
    if CACHE_COLLECTION in existing:
        return
    await qdrant.create_collection(
        collection_name=CACHE_COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=embed_dim, distance=qmodels.Distance.COSINE
        ),
    )
    # Payload index so the expire_at filter stays fast as the collection grows.
    await qdrant.create_payload_index(
        collection_name=CACHE_COLLECTION,
        field_name="expire_at",
        field_schema=qmodels.PayloadSchemaType.INTEGER,
    )


async def cache_lookup(qdrant: AsyncQdrantClient, vec: list[float]) -> dict | None:
    """Return the most similar non-expired cached payload, or None on miss."""
    now = int(time.time())
    # ``query_points`` replaced the (removed) ``search`` method in qdrant-client
    # 1.10+. The response is a QueryResponse with .points instead of a bare list.
    result = await qdrant.query_points(
        collection_name=CACHE_COLLECTION,
        query=vec,
        limit=1,
        score_threshold=CACHE_THRESHOLD,
        query_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="expire_at",
                range=qmodels.Range(gte=now),
            )],
        ),
    )
    if not result.points:
        return None
    hit = result.points[0]
    payload = dict(hit.payload or {})
    payload["score"] = hit.score
    return payload


async def cache_store(
    qdrant: AsyncQdrantClient,
    vec: list[float],
    query: str,
    response: str,
    model: str,
    sources: list[str],
    input_tokens: int,
    output_tokens: int,
    fallback_used: bool,
) -> None:
    """Upsert a (query, response) pair into the semantic cache.

    Token counts and fallback flag are stored alongside so future cache hits
    can be logged into ``usage_log`` with the original model context.
    """
    now = int(time.time())
    await qdrant.upsert(
        collection_name=CACHE_COLLECTION,
        points=[qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "query": query,
                "response": response,
                "model": model,
                "sources": sources,
                "timestamp": now,
                "expire_at": now + CACHE_TTL_SECONDS,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "fallback_used": fallback_used,
            },
        )],
    )


async def replay_cached(
    payload: dict,
    tier: str,
    request_id: str,
    api_key: str,
    start_ts: float,
    redis: aioredis.Redis,
    request: Request,
):
    """Stream a cached response back token-by-token so the UX matches a fresh
    call (same SSE event shape, same word-aligned chunking). Logs the request
    into ``usage_log`` and bumps the global stream counters on the way out so
    cache hits show up in /health alongside LLM-served requests."""
    completed = False
    try:
        response = payload.get("response", "")
        ttft_ms: int | None = None
        # Word-by-word replay keeps each event aligned to a whitespace boundary,
        # matching the live-streaming buffer's behavior.
        for word in response.split(" "):
            if word == "":
                continue
            if await request.is_disconnected():
                # Client gave up mid-replay — finally-block records the abort.
                return
            if ttft_ms is None:
                ttft_ms = int((time.time() - start_ts) * 1000)
            yield _sse({"type": "token", "content": word + " "})
        yield _sse({
            "type": "done",
            "model": payload.get("model"),
            "tier": tier,
            "usage": {"input_tokens": 0, "output_tokens": 0, "estimated": False},
            "cost_usd": 0.0,
            "cache_hit": True,
            "cache_score": payload.get("score"),
            "sources": payload.get("sources", []),
            "request_id": request_id,
        })
        latency_ms = int((time.time() - start_ts) * 1000)
        await log_usage(
            request_id=request_id,
            api_key=api_key,
            model=payload.get("model", "unknown"),
            # Cache hits don't burn upstream tokens — store 0 so daily token
            # totals reflect actual LLM consumption, not the saved-by-cache amount.
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            cache_hit=True,
            fallback_used=bool(payload.get("fallback_used", False)),
        )
        completed = True
    finally:
        await metric_incr(
            redis, "completed_streams" if completed else "aborted_streams"
        )
