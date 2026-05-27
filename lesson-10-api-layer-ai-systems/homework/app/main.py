"""Q&A bot — pgvector retrieval + OpenRouter SSE streaming."""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass

import psycopg
import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from openai import AsyncOpenAI
from pgvector.psycopg import register_vector_async
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models as qmodels
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Tiers: each entry is a list of (model_name, price_in_per_mtok, price_out_per_mtok).
# Position 0 is the primary model; position 1 is the secondary (fallback).
# Prices are USD per 1,000,000 tokens.
# ---------------------------------------------------------------------------
TIERS = {
    "demo-free": {
        "token_limit": 5_000,
        "models": [
            ("google/gemma-4-31b-it:free", 0.0, 0.0),
            ("meta-llama/llama-3.3-70b-instruct:free", 0.0, 0.0),
            ("meta-llama/llama-3.2-3b-instruct:free", 0.0, 0.0),
        ],
    },
    "demo-pro": {
        "token_limit": 20_000,
        "models": [
            ("google/gemma-4-31b-it", 0.12, 0.37),
            ("openai/gpt-5.4-nano", 0.18, 1.25),
            ("anthropic/claude-3-haiku", 0.25, 1.25),
        ],
    },
    "demo-enterprise": {
        "token_limit": 100_000,
        "models": [
            ("openai/gpt-5.4-mini", 0.75, 4.5),
            ("google/gemini-3.5-flash", 1.5, 9),
            ("anthropic/claude-haiku-4.5", 0.8, 4),
        ]
    }
}

API_KEYS = {
    "demo-free-key": "demo-free",
    "demo-pro-key": "demo-pro",
    "demo-enterprise-key": "demo-enterprise",
}


def model_for(tier: str, slot: int) -> tuple[str, float, float] | None:
    """Return ``(name, price_in_per_mtok, price_out_per_mtok)`` for the tier slot,
    or None if the tier has no model at that position."""
    models = TIERS[tier]["models"]
    return models[slot] if slot < len(models) else None


def fallback_chain(tier: str) -> list[tuple[str, float, float]]:
    """Ordered model chain for ``tier`` (primary -> secondary -> tertiary)."""
    return [m for m in (model_for(tier, i) for i in range(3)) if m is not None]


# ---------------------------------------------------------------------------
# Auth: X-API-Key is mandatory. Missing -> 401, unknown -> 403.
# auto_error=False so we control the 401 body / WWW-Authenticate header.
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class Caller:
    api_key: str
    tier: str


def require_api_key(api_key: str | None = Security(api_key_header)) -> Caller:
    """Validate ``X-API-Key`` and return the caller (key + tier)."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    tier = API_KEYS.get(api_key)
    if tier is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return Caller(api_key=api_key, tier=tier)


# ---------------------------------------------------------------------------
# Rate limit: per-API-key token bucket in Redis (reserve-and-settle).
#
# A permissive pre-check (GET, then INCR after) loses under concurrency: every
# parallel request sees `used < limit` at the same instant and passes. So we
# reserve a worst-case estimate up front *atomically* and settle to real usage
# after the model finishes:
#
#   1) reserve:  INCRBY est  (+ EXPIRE 60 on first write)
#                 if new_val > limit -> DECRBY est, return 429 + Retry-After
#   2) settle:   delta = actual - est
#                 INCRBY delta   (or DECRBY -delta) once usage arrives
#   3) abort:    DECRBY est       (full refund if the request didn't complete)
#
# Only INCRBY / DECRBY / EXPIRE / GET / TTL — works on Upstash REST.
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 180


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
    """Atomically reserve ``tokens``. Return 0 if reserved, else Retry-After seconds.

    Because INCRBY is atomic, concurrent reservations are serialized on the
    Redis side: the (N+1)-th caller sees a value that already includes the
    previous N reservations and gets rejected if it crosses ``limit``.
    """
    new_val = await redis.incrby(quota_key(api_key), tokens)
    if new_val == tokens:
        # First write of a fresh window — start the 60s refill timer.
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


# ---------------------------------------------------------------------------
# Semantic cache: Qdrant collection of (question embedding -> response).
#
# Global cache (one collection for all API keys) — this is a public Q&A bot,
# not private data. RAG and cache share the same embedding model so vectors
# are directly comparable.
#
# Qdrant has no built-in TTL, so we stamp `expire_at` (unix seconds) into the
# payload and filter on it at query time. Stale points still sit in the
# collection until something overwrites them; a periodic cleanup job is fine
# but not required for correctness.
# ---------------------------------------------------------------------------
CACHE_COLLECTION = "cache_collection"
EMBED_DIM = 384  # all-MiniLM-L6-v2
CACHE_THRESHOLD = 0.92  # cosine similarity required to serve from cache
CACHE_TTL_SECONDS = 3600  # 1 hour


def embed(question: str) -> list[float]:
    """Encode once per request — reused for cache lookup and RAG retrieval."""
    assert embedder is not None
    return embedder.encode([question], normalize_embeddings=True)[0].tolist()


async def cache_lookup(qdrant: AsyncQdrantClient, vec: list[float]) -> dict | None:
    """Return the most similar non-expired cached payload, or None on miss."""
    now = int(time.time())
    hits = await qdrant.search(
        collection_name=CACHE_COLLECTION,
        query_vector=vec,
        limit=1,
        score_threshold=CACHE_THRESHOLD,
        query_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="expire_at",
                range=qmodels.Range(gte=now),
            )],
        ),
    )
    if not hits:
        return None
    payload = dict(hits[0].payload or {})
    payload["score"] = hits[0].score
    return payload


async def cache_store(
    qdrant: AsyncQdrantClient,
    vec: list[float],
    query: str,
    response: str,
    model: str,
    sources: list[str],
) -> None:
    """Upsert a (query, response) pair into the semantic cache."""
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
            },
        )],
    )


metrics: Counter = Counter()
embedder: SentenceTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    app.state.llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    # Upstash exposes a regular RESP endpoint at rediss://... that works
    # transparently with redis-py — REST is only needed in serverless runtimes.
    app.state.redis = aioredis.from_url(
        os.environ["REDIS_URL"], decode_responses=True
    )
    # Qdrant for the semantic response cache. ``:memory:`` runs the whole
    # vector store inside this process — no server, no env vars, but the
    # cache is wiped on every restart and not shared across instances.
    app.state.qdrant = AsyncQdrantClient(location=":memory:")
    await app.state.qdrant.create_collection(
        collection_name=CACHE_COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=EMBED_DIM, distance=qmodels.Distance.COSINE
        ),
    )
    # Payload index so the expire_at filter stays fast as the collection grows.
    await app.state.qdrant.create_payload_index(
        collection_name=CACHE_COLLECTION,
        field_name="expire_at",
        field_schema=qmodels.PayloadSchemaType.INTEGER,
    )
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await app.state.qdrant.close()


app = FastAPI(lifespan=lifespan)


class ChatIn(BaseModel):
    message: str


async def retrieve(vec: list[float], k: int = 3) -> list[tuple[int, str]]:
    """Return [(chunk_index, content), ...] for the top-K nearest chunks.

    Takes a pre-computed vector so the embedding step is only paid once per
    request — the same vector is used for the semantic cache lookup.
    """
    async with await psycopg.AsyncConnection.connect(os.environ["DB_URL"]) as conn:
        await register_vector_async(conn)
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT chunk_index, content FROM documents "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (vec, k),
            )
            return await cur.fetchall()


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def open_completion(
    client: AsyncOpenAI,
    model_name: str,
    messages: list[dict],
    token_limit: int,
):
    return await client.chat.completions.create(
        model=model_name,
        stream=True,
        # Two redundant usage-include flags: ``stream_options`` is the OpenAI
        # standard, while OpenRouter wants ``usage: {include: true}`` in the
        # body (free models in particular skip the OpenAI-style flag).
        stream_options={"include_usage": True},
        extra_body={"usage": {"include": True}},
        # Cap to the tier budget so a runaway response can't blow it.
        max_tokens=min(1024, token_limit),
        messages=messages,
    )


async def replay_cached(payload: dict, tier: str):
    """Stream a cached response back token-by-token so the UX matches a fresh
    call (same SSE event shape, same word-aligned chunking)."""
    response = payload.get("response", "")
    # Word-by-word replay keeps each event aligned to a whitespace boundary,
    # matching the live-streaming buffer's behavior.
    for word in response.split(" "):
        if word == "":
            continue
        yield sse({"type": "token", "content": word + " "})
    yield sse({
        "type": "done",
        "model": payload.get("model"),
        "tier": tier,
        "usage": {"input_tokens": 0, "output_tokens": 0, "estimated": False},
        "cost_usd": 0.0,
        "cache_hit": True,
        "cache_score": payload.get("score"),
        "sources": payload.get("sources", []),
    })


@app.post("/chat/stream")
async def chat_stream(
    body: ChatIn,
    request: Request,
    caller: Caller = Depends(require_api_key),
):
    redis: aioredis.Redis = request.app.state.redis
    qdrant: AsyncQdrantClient = request.app.state.qdrant
    token_limit = TIERS[caller.tier]["token_limit"]

    # Single embedding call, reused for cache lookup and RAG retrieval below.
    vec = embed(body.message)

    # Semantic cache check. A HIT short-circuits the LLM call entirely — no
    # rate-limit charge, no upstream tokens, just replay the stored response.
    cached = await cache_lookup(qdrant, vec)
    if cached is not None:
        metrics["cache_hits"] += 1
        return StreamingResponse(
            replay_cached(cached, caller.tier),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    metrics["cache_misses"] += 1

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
    reservation = estimate_tokens(messages, token_limit)
    retry_after = await quota_reserve(
        redis, caller.api_key, reservation, token_limit
    )
    if retry_after:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded for tier '{caller.tier}' "
                f"({token_limit} tokens / {WINDOW_SECONDS}s)"
            ),
            headers={"Retry-After": str(retry_after)},
        )

    chain = fallback_chain(caller.tier)

    # Try each model in order (primary -> secondary -> tertiary). Fallback only
    # happens on failure to *open* the stream; we don't retry mid-stream — that
    # would send the client a second answer after they've already seen tokens.
    llm: AsyncOpenAI = request.app.state.llm
    chosen: tuple[str, float, float] | None = None
    resp = None
    last_error: Exception | None = None
    for candidate in chain:
        try:
            resp = await open_completion(llm, candidate[0], messages, token_limit)
            chosen = candidate
            break
        except Exception as exc:  # noqa: BLE001 — bubble up only if all fail
            last_error = exc
    if chosen is None or resp is None:
        # All models failed to open — refund the reservation we just made so
        # the caller isn't billed for a stream that never started.
        await quota_refund(redis, caller.api_key, reservation)
        raise last_error if last_error else RuntimeError("No models configured")

    model_name, price_in_mtok, price_out_mtok = chosen
    price_in = price_in_mtok / 1_000_000
    price_out = price_out_mtok / 1_000_000

    async def gen():
        completed = False
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

            # Prefer real usage; otherwise estimate from character counts so the
            # bucket is always charged something (free OpenRouter models often
            # omit the usage chunk entirely). ~4 chars per token is a coarse but
            # widely-used heuristic.
            if usage:
                inp = usage.prompt_tokens or 0
                out = usage.completion_tokens or 0
                estimated = False
            else:
                prompt_chars = sum(len(m["content"]) for m in messages)
                inp = max(1, prompt_chars // 4)
                out = max(1, output_chars // 4)
                estimated = True
            cached = 0
            details = getattr(usage, "prompt_tokens_details", None) if usage else None
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            # Reconcile the reservation with what we actually used. This
            # converts the worst-case hold into the real charge.
            await quota_settle(redis, caller.api_key, reservation, inp + out)
            # Store the completed response in the semantic cache so the next
            # similar question can short-circuit the LLM.
            if full_response.strip():
                await cache_store(
                    qdrant, vec, body.message, full_response,
                    model_name, sources,
                )
            yield sse({
                "type": "done",
                "model": model_name,
                "tier": caller.tier,
                "usage": {"input_tokens": inp, "output_tokens": out, "estimated": estimated},
                "cost_usd": round(inp * price_in + out * price_out, 6),
                "cache_hit": False,
                "prompt_cache_hit": cached > 0,
                "sources": sources,
            })
            completed = True
        finally:
            await resp.close()  # cancels the upstream HTTP request → stops billing
            if not completed:
                # Stream aborted (client disconnect, exception, etc.) — refund
                # the full reservation since we never got a real usage figure.
                await quota_refund(redis, caller.api_key, reservation)
            metrics["completed_streams" if completed else "aborted_streams"] += 1

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "tiers": {
            tier: [m[0] for m in fallback_chain(tier)] for tier in TIERS
        },
        "completed_streams": metrics["completed_streams"],
        "aborted_streams": metrics["aborted_streams"],
        "cache_hits": metrics["cache_hits"],
        "cache_misses": metrics["cache_misses"],
    }
