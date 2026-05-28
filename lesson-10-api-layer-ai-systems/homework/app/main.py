"""Q&A bot — pgvector retrieval + semantic cache + OpenRouter SSE streaming.

This module is intentionally thin: it owns the FastAPI app, the lifespan
context, and the request handlers. The actual mechanics live in:

    - app.auth        X-API-Key auth + Caller
    - app.tiers       tier config, fallback chain
    - app.ratelimit   token-bucket reserve/settle/refund
    - app.metrics     Redis-backed counters
    - app.rag         embedding model + pgvector retrieval
    - app.cache       Qdrant semantic cache + replay generator
    - app.llm         OpenRouter calls + usage extraction
    - app.usage_log   per-request log + /usage read paths
    - app.pricing     single source of truth for cost math
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

from .auth import Caller, require_api_key
from .cache import (
    cache_lookup,
    cache_store,
    ensure_collection as ensure_cache_collection,
    replay_cached,
)
from .llm import extract_usage, open_with_fallback
from .metrics import metric_incr, metrics_snapshot
from .pricing import cost_usd
from .rag import EMBED_DIM, embed, init_embedder, retrieve
from .ratelimit import (
    WINDOW_SECONDS,
    estimate_tokens,
    quota_refund,
    quota_reserve,
    quota_settle,
)
from .tiers import TIERS, fallback_chain, token_limit
from .usage_log import (
    ensure_usage_log,
    fetch_today_breakdown,
    fetch_today_totals,
    log_usage,
)


# ---------------------------------------------------------------------------
# Lifespan: load the embedder, open clients to OpenRouter / Redis / Qdrant,
# make sure the cache collection and usage_log table exist.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_embedder()
    app.state.llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    # Upstash exposes a regular RESP endpoint at rediss://... that works
    # transparently with redis-py — REST is only needed in serverless runtimes.
    #
    # Managed Redis (Upstash, ElastiCache, etc.) reaps idle TCP connections,
    # which surfaces here as "Connection reset by peer" on the next command.
    # Three guards together keep the client healthy:
    #   - health_check_interval: PING every 30s to detect dead conns early
    #   - socket_keepalive: OS-level TCP keepalive on the socket
    #   - retry: transparently reconnect + retry on connection errors
    app.state.redis = aioredis.from_url(
        os.environ["REDIS_URL"],
        decode_responses=True,
        health_check_interval=30,
        socket_keepalive=True,
        retry=Retry(ExponentialBackoff(cap=1, base=0.1), retries=3),
        retry_on_error=[RedisConnectionError, RedisTimeoutError, ConnectionResetError],
    )
    # ``:memory:`` runs the whole vector store inside this process — no server
    # required, but the cache is wiped on restart and not shared across instances.
    app.state.qdrant = AsyncQdrantClient(location=":memory:")
    await ensure_cache_collection(app.state.qdrant, EMBED_DIM)
    await ensure_usage_log()
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await app.state.qdrant.close()


app = FastAPI(lifespan=lifespan)


class ChatIn(BaseModel):
    message: str


def sse(payload: dict) -> str:
    """Encode a payload as a single SSE ``data:`` event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(
    body: ChatIn,
    request: Request,
    caller: Caller = Depends(require_api_key),
):
    redis: aioredis.Redis = request.app.state.redis
    qdrant: AsyncQdrantClient = request.app.state.qdrant
    llm: AsyncOpenAI = request.app.state.llm
    limit = token_limit(caller.tier)

    # Per-request observability handles. Carried through the whole flow so
    # both cache-hit and cache-miss paths log a consistent usage_log row.
    request_id = str(uuid.uuid4())
    start_ts = time.time()

    # Single embedding call, reused for cache lookup and RAG retrieval.
    vec = embed(body.message)

    # Semantic cache check. A HIT short-circuits the LLM call entirely — no
    # rate-limit charge, no upstream tokens, just replay the stored response.
    cached = await cache_lookup(qdrant, vec)
    if cached is not None:
        await metric_incr(redis, "cache_hits")
        return StreamingResponse(
            replay_cached(
                cached, caller.tier,
                request_id=request_id,
                api_key=caller.api_key,
                start_ts=start_ts,
                redis=redis,
                request=request,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    await metric_incr(redis, "cache_misses")

    # MISS — RAG using the same vector, then build the prompt as usual.
    rows = await retrieve(vec)
    sources = [f"chunk_{i}" for i, _ in rows]
    context = "\n\n".join(f"[chunk_{i}] {c}" for i, c in rows)
    messages = [
        {"role": "system", "content": "Answer using only the provided context. Be concise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {body.message}"},
    ]

    # Reserve worst-case budget atomically. This is what makes parallel bursts
    # actually get capped — INCRBY serializes the concurrent reservations.
    reservation = estimate_tokens(messages, limit)
    retry_after = await quota_reserve(redis, caller.api_key, reservation, limit)
    if retry_after:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded for tier '{caller.tier}' "
                f"({limit} tokens / {WINDOW_SECONDS}s)"
            ),
            headers={"Retry-After": str(retry_after)},
        )

    # Open the LLM stream, trying each model in the fallback chain.
    try:
        resp, model_name, fallback_used = await open_with_fallback(
            llm, fallback_chain(caller.tier), messages, limit
        )
    except Exception:
        # All models failed to open — refund the reservation we just made so
        # the caller isn't billed for a stream that never started.
        await quota_refund(redis, caller.api_key, reservation)
        raise

    async def gen():
        completed = False
        ttft_ms: int | None = None
        try:
            usage = None
            # Word-boundary buffer: upstream subword tokens (e.g. " twe",
            # "lve-facto") get coalesced so each SSE event ends on whitespace.
            buf = ""
            output_chars = 0  # Tracked for the no-usage fallback estimate.
            full_response = ""  # Accumulated text to store in the cache on success.
            async for chunk in resp:
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if not piece:
                    continue
                if await request.is_disconnected():
                    return  # finally-block bumps aborted_streams + closes resp
                if ttft_ms is None:
                    ttft_ms = int((time.time() - start_ts) * 1000)
                output_chars += len(piece)
                full_response += piece
                buf += piece
                cut = max(buf.rfind(" "), buf.rfind("\n"))
                if cut >= 0:
                    yield sse({"type": "token", "content": buf[: cut + 1]})
                    buf = buf[cut + 1 :]
            # Flush any trailing partial word once the upstream stream ends.
            if buf:
                yield sse({"type": "token", "content": buf})

            prompt_chars = sum(len(m["content"]) for m in messages)
            inp, out, estimated, prompt_cache_hit = extract_usage(
                usage, prompt_chars=prompt_chars, output_chars=output_chars
            )
            # Reconcile the reservation with what we actually used. This
            # converts the worst-case hold into the real charge.
            await quota_settle(redis, caller.api_key, reservation, inp + out)
            # Store the completed response in the semantic cache so the next
            # similar question can short-circuit the LLM. Token counts and the
            # fallback flag travel with the payload so future cache hits log
            # with the original request's context.
            if full_response.strip():
                await cache_store(
                    qdrant, vec, body.message, full_response,
                    model_name, sources,
                    input_tokens=inp,
                    output_tokens=out,
                    fallback_used=fallback_used,
                )
            yield sse({
                "type": "done",
                "model": model_name,
                "tier": caller.tier,
                "usage": {"input_tokens": inp, "output_tokens": out, "estimated": estimated},
                "cost_usd": cost_usd(model_name, inp, out),
                "cache_hit": False,
                "prompt_cache_hit": prompt_cache_hit,
                "fallback_used": fallback_used,
                "sources": sources,
                "request_id": request_id,
            })
            completed = True
            latency_ms = int((time.time() - start_ts) * 1000)
            await log_usage(
                request_id=request_id,
                api_key=caller.api_key,
                model=model_name,
                input_tokens=inp,
                output_tokens=out,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                cache_hit=False,
                fallback_used=fallback_used,
            )
        finally:
            await resp.close()  # cancels the upstream HTTP request → stops billing
            if not completed:
                # Stream aborted (client disconnect, exception, etc.) — refund
                # the full reservation since we never got a real usage figure.
                await quota_refund(redis, caller.api_key, reservation)
            await metric_incr(
                redis, "completed_streams" if completed else "aborted_streams"
            )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health(request: Request):
    snap = await metrics_snapshot(request.app.state.redis)
    return {
        "status": "ok",
        "tiers": {tier: fallback_chain(tier) for tier in TIERS},
        "completed_streams": snap.get("completed_streams", 0),
        "aborted_streams": snap.get("aborted_streams", 0),
        "cache_hits": snap.get("cache_hits", 0),
        "cache_misses": snap.get("cache_misses", 0),
    }


# ---------------------------------------------------------------------------
# Usage endpoints. Scoped to the caller's API key (auth via X-API-Key) and
# always covering "today" (Postgres CURRENT_DATE in the DB's timezone).
# ---------------------------------------------------------------------------
@app.get("/usage/today")
async def usage_today(caller: Caller = Depends(require_api_key)):
    """Headline numbers for the caller's traffic today: requests, total tokens,
    total cost. Cache hits count as requests but contribute 0 tokens / 0 cost."""
    return await fetch_today_totals(caller.api_key)


@app.get("/usage/breakdown")
async def usage_breakdown(caller: Caller = Depends(require_api_key)):
    """Per-model breakdown plus headline cache / fallback / latency stats for
    the caller's traffic today. p95 uses Postgres' PERCENTILE_CONT."""
    return await fetch_today_breakdown(caller.api_key)
