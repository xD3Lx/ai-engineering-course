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

import asyncio
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
from .guardrails import (
    MAX_INPUT_CHARS,
    SYSTEM_PROMPT,
    detect_injection,
    log_suspicious_request,
    log_suspicious_response,
    sanitize_user_input,
    scan_output,
)
from .llm import extract_usage, open_with_fallback
from .metrics import metric_incr, metrics_snapshot
from .pricing import cost_usd
from .rag import EMBED_DIM, embed, init_embedder, retrieve
from .ratelimit import (
    WINDOW_SECONDS,
    estimate_tokens,
    observed_average_tokens,
    quota_refund,
    quota_reserve,
    quota_settle,
)
from . import tracing
from .tiers import TIERS, fallback_chain, token_limit
from .usage_log import (
    ensure_usage_log,
    fetch_today_breakdown,
    fetch_today_totals,
    log_usage,
)


# Cap on concurrent in-flight LLM streams per process. Bounds load on
# OpenRouter (which has its own rate limits) and caps memory during traffic
# spikes — excess requests await a free slot instead of all hitting upstream
# at once. The semaphore lives on app.state so it's one shared instance.
MAX_CONCURRENT_LLM = 20


# ---------------------------------------------------------------------------
# Lifespan: load the embedder, open clients to OpenRouter / Redis / Qdrant,
# make sure the cache collection and usage_log table exist.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_embedder()
    # Concurrency limiter for upstream LLM streams (see MAX_CONCURRENT_LLM).
    app.state.llm_sem = asyncio.Semaphore(MAX_CONCURRENT_LLM)
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
    # Qdrant Cloud when QDRANT_URL is set — a persistent, managed vector store
    # shared across instances (the semantic cache survives restarts and is hit
    # by every worker). Falls back to an in-process ``:memory:`` store for local
    # dev, which needs no server but is wiped on restart and not shared.
    qdrant_url = os.environ.get("QDRANT_URL")
    if qdrant_url:
        app.state.qdrant = AsyncQdrantClient(
            url=qdrant_url,
            api_key=os.environ.get("QDRANT_API_KEY"),
            # Cloud round-trips are slower than in-process; give upserts/queries
            # headroom so a slow network blip doesn't surface as an error.
            timeout=30,
        )
    else:
        app.state.qdrant = AsyncQdrantClient(location=":memory:")
    await ensure_cache_collection(app.state.qdrant, EMBED_DIM)
    await ensure_usage_log()
    # Langfuse tracing — no-op if LANGFUSE_* env vars are unset.
    tracing.init_tracing()
    try:
        yield
    finally:
        tracing.shutdown()  # flush buffered spans before exit
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

    # --- Langfuse trace: one root span per request, spanning the whole
    # pipeline (auth → rate limit → embed → cache → vector search → LLM →
    # stream). Child spans are ended as each step finishes; the root is ended
    # in gen()'s cleanup (or on an early-return error path below).
    trace = tracing.start_root(
        "chat_stream",
        input={"message": body.message},
        metadata={"request_id": request_id, "tier": caller.tier, "api_key": caller.api_key},
    )
    tracing.set_trace(
        trace,
        name="chat_stream",
        user_id=caller.api_key,
        tags=[f"tier:{caller.tier}"],
        metadata={"request_id": request_id},
    )
    # Auth already ran in the require_api_key dependency — record it as a step.
    tracing.end(
        tracing.start_child(trace, "auth", input={"api_key": caller.api_key}),
        output={"tier": caller.tier},
    )

    # --- input guardrails ---------------------------------------------------
    guard_span = tracing.start_child(
        trace, "input_guardrails", input={"chars": len(body.message)}
    )
    # Length cap: reject oversized input before any embedding / LLM work.
    if len(body.message) > MAX_INPUT_CHARS:
        tracing.end(guard_span, output="rejected: too_long", level="WARNING")
        tracing.end(trace, output={"status": "rejected", "reason": "input_too_long"})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input too long: {len(body.message)} chars (max {MAX_INPUT_CHARS}).",
        )
    # Prompt-injection detection: log the hit, then reject.
    matched_pattern = detect_injection(body.message)
    if matched_pattern is not None:
        log_suspicious_request(
            api_key=caller.api_key,
            request_id=request_id,
            pattern=matched_pattern,
            message=body.message,
        )
        tracing.end(
            guard_span, output="rejected: injection",
            metadata={"pattern": matched_pattern}, level="WARNING",
        )
        tracing.end(trace, output={"status": "rejected", "reason": "prompt_injection"})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input rejected: prompt-injection pattern detected.",
        )
    # Neutralize any forged XML delimiters before the input enters the prompt.
    safe_message = sanitize_user_input(body.message)
    tracing.end(guard_span, output="passed")

    # Single embedding call, reused for cache lookup and RAG retrieval.
    embed_span = tracing.start_child(trace, "embed_query", input={"message": body.message})
    vec = embed(body.message)
    tracing.end(embed_span, output={"embedding_dim": len(vec)})

    # Semantic cache check. A HIT short-circuits the LLM call entirely — no
    # rate-limit charge, no upstream tokens, just replay the stored response.
    cache_span = tracing.start_child(trace, "cache_check")
    cached = await cache_lookup(qdrant, vec)
    if cached is not None:
        await metric_incr(redis, "cache_hits")
        tracing.end(cache_span, output={"cache_hit": True, "score": cached.get("score")})
        # Record the cached answer as a generation so cache hits show the same
        # prompt/completion shape as live calls. We have the full payload here,
        # so the trace is complete without threading into the replay generator.
        cached_model = cached.get("model")
        cache_gen = tracing.start_child(
            trace, "llm_call", as_type="generation",
            model=cached_model,
            input={"cached_query": cached.get("query")},
            metadata={
                "model": cached_model, "api_key": caller.api_key, "tier": caller.tier,
                "cache_hit": True, "fallback_used": bool(cached.get("fallback_used")),
            },
        )
        tracing.end(
            cache_gen, output=cached.get("response", ""),
            usage_details={"input": 0, "output": 0, "total": 0},
        )
        tracing.set_trace(trace, tags=[
            f"tier:{caller.tier}", f"model:{cached_model}",
            "cache_hit:true", f"fallback:{bool(cached.get('fallback_used'))}",
        ])
        tracing.end(trace, output={"cache_hit": True})
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
    tracing.end(cache_span, output={"cache_hit": False})

    # MISS — RAG using the same vector, then build the prompt as usual.
    search_span = tracing.start_child(trace, "vector_search", input={"message": body.message})
    rows = await retrieve(vec)
    sources = [f"chunk_{i}" for i, _ in rows]
    context = "\n\n".join(f"[chunk_{i}] {c}" for i, c in rows)
    tracing.end(search_span, output={"sources": sources, "num_chunks": len(rows)})
    # Role separation + XML envelopes around untrusted data. The system prompt
    # explicitly tells the model to treat tag contents as data, not commands,
    # so user input can't override the instructions.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"<context>\n{context}\n</context>\n\n"
            f"<user_query>\n{safe_message}\n</user_query>"
        )},
    ]

    # Reserve an expected budget atomically. This is what makes parallel bursts
    # actually get capped — INCRBY serializes the concurrent reservations.
    average_tokens = await observed_average_tokens(redis, caller.api_key)
    reservation = estimate_tokens(messages, limit, average_tokens)
    rl_span = tracing.start_child(
        trace, "rate_limit", input={"reservation": reservation, "limit": limit}
    )
    retry_after = await quota_reserve(redis, caller.api_key, reservation, limit)
    if retry_after:
        # Rate-limited before the stream ever opened — count it as an aborted
        # stream so /health reflects requests that never produced tokens.
        await metric_incr(redis, "aborted_streams")
        tracing.end(
            rl_span, output={"allowed": False, "retry_after": retry_after}, level="WARNING"
        )
        tracing.end(trace, output={"status": "rate_limited"})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded for tier '{caller.tier}' "
                f"({limit} tokens / {WINDOW_SECONDS}s)"
            ),
            headers={"Retry-After": str(retry_after)},
        )
    tracing.end(rl_span, output={"allowed": True})

    # Concurrency gate: acquire a slot before touching OpenRouter. During a
    # spike this awaits here rather than opening more upstream streams than
    # MAX_CONCURRENT_LLM. The slot is released in gen()'s finally (or below if
    # the open fails). acquire() is cancellation-safe (frees itself on cancel).
    llm_sem: asyncio.Semaphore = request.app.state.llm_sem

    # LLM generation span — input is the full prompt (system + retrieved chunks
    # + user query) for debugging hallucinations / RAG retrieval. Output, usage
    # and cost are filled in once streaming completes; the span is ended in
    # gen()'s cleanup so it stays open for the whole stream.
    llm_gen = tracing.start_child(
        trace, "llm_call", as_type="generation",
        input={"messages": messages},
        metadata={"api_key": caller.api_key, "tier": caller.tier},
    )

    await llm_sem.acquire()

    # Open the LLM stream, trying each model in the fallback chain.
    try:
        resp, model_name, fallback_used = await open_with_fallback(
            llm, fallback_chain(caller.tier), messages, limit
        )
    except BaseException:
        # Open failed or was cancelled — free the slot and refund the
        # reservation so the caller isn't billed for a stream that never began.
        llm_sem.release()
        await quota_refund(redis, caller.api_key, reservation)
        tracing.end(llm_gen, output="open_failed", level="ERROR")
        tracing.end(trace, output={"status": "error", "reason": "all_models_failed"})
        raise

    # Stream is open and about to consume upstream tokens — it's now active.
    await metric_incr(redis, "active_streams", 1)
    # Record the actually-used model + tags now that the stream is open, so the
    # trace is tagged even if the client disconnects mid-stream.
    tracing.update(llm_gen, model=model_name, metadata={
        "model": model_name, "fallback_used": fallback_used,
        "api_key": caller.api_key, "tier": caller.tier, "cache_hit": False,
    })
    tracing.set_trace(trace, tags=[
        f"tier:{caller.tier}", f"model:{model_name}",
        "cache_hit:false", f"fallback:{fallback_used}",
    ])

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
                    # Early-out on client disconnect. (A disconnect also raises
                    # CancelledError through `async for chunk in resp` into the
                    # SDK, aborting the upstream request — this is the graceful
                    # path that avoids waiting for the next chunk.) The finally
                    # block frees the slot, refunds, and bumps aborted_streams.
                    return
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
            # Output filtering (post-stream): scan the accumulated answer for
            # leaked system-prompt fragments. Live-blocking mid-stream isn't
            # required — we flag the record and log it after the fact.
            leaked_fragment = scan_output(full_response)
            output_filtered = leaked_fragment is not None
            if output_filtered:
                log_suspicious_response(
                    api_key=caller.api_key,
                    request_id=request_id,
                    fragment=leaked_fragment,
                    response=full_response,
                )
            # Store the completed response in the semantic cache so the next
            # similar question can short-circuit the LLM. Token counts and the
            # fallback flag travel with the payload so future cache hits log
            # with the original request's context. A flagged response is NOT
            # cached, so we don't replay leaked content to future callers.
            if full_response.strip() and not output_filtered:
                await cache_store(
                    qdrant, vec, body.message, full_response,
                    model_name, sources,
                    input_tokens=inp,
                    output_tokens=out,
                    fallback_used=fallback_used,
                )
            # Fill in the LLM generation span: the prompt went in as input;
            # here we attach the completion, token usage, cost, and run flags
            # (for debugging hallucinations and RAG retrieval issues).
            tracing.update(
                llm_gen,
                output=full_response,
                usage_details={"input": inp, "output": out, "total": inp + out},
                cost_details={"total": cost_usd(model_name, inp, out)},
                metadata={
                    "model": model_name, "api_key": caller.api_key, "tier": caller.tier,
                    "cache_hit": False, "fallback_used": fallback_used,
                    "output_filtered": output_filtered, "estimated": estimated,
                    "prompt_cache_hit": prompt_cache_hit,
                },
            )
            latency_ms = int((time.time() - start_ts) * 1000)
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
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
                "output_filtered": output_filtered,
            })
            completed = True
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
                output_filtered=output_filtered,
            )
        finally:
            # release() is synchronous, so the slot is freed immediately even
            # if the cleanup awaits below are interrupted by cancellation.
            llm_sem.release()

            async def _cleanup() -> None:
                await resp.close()  # cancels the upstream HTTP request → stops billing
                await metric_incr(redis, "active_streams", -1)
                if not completed:
                    # Aborted (client disconnect, exception, etc.) — refund the
                    # full reservation; no settle, so tokens never hit the rate
                    # limiter and nothing is written to the cost tracker.
                    await quota_refund(redis, caller.api_key, reservation)
                await metric_incr(
                    redis, "completed_streams" if completed else "aborted_streams"
                )

            # Shield so a client-disconnect cancellation still runs cleanup to
            # completion: upstream closed, reservation refunded, gauges fixed.
            await asyncio.shield(_cleanup())

            # Close out the Langfuse generation + trace. Sync + buffered, so
            # this is cheap and safe even on the abort path (output set above
            # only on the success branch; an aborted stream ends with none).
            tracing.end(llm_gen, metadata={"completed": completed})
            tracing.end(trace, output={"completed": completed, "request_id": request_id})

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
        # Gauge: streams consuming an LLM slot right now. Clamped at 0 in case
        # a hard cancellation ever skips a decrement.
        "active_streams": max(0, snap.get("active_streams", 0)),
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
