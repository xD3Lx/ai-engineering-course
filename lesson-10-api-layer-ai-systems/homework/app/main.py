"""Q&A bot — pgvector retrieval + OpenRouter SSE streaming."""
from __future__ import annotations

import json
import os
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
# Rate limit: per-API-key token bucket in Redis.
#
# Bucket = TIER token_limit, refills fully every WINDOW_SECONDS. We store the
# *consumed* count in `quota:{api_key}` and rely on key TTL to "refill":
#
#   - First charge:   INCRBY n  +  EXPIRE 60   -> starts the 60s window
#   - Later charges:  INCRBY n                  -> TTL keeps ticking down
#   - Window ends:    key expires               -> bucket is full again
#
# Pre-check reads the current count; if it's already >= limit we reject with
# 429 and use the remaining TTL as Retry-After. This is the standard
# INCR + EXPIRE pattern that works against Upstash REST (no Lua needed).
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 60


def quota_key(api_key: str) -> str:
    return f"quota:{api_key}"


async def quota_retry_after(redis: aioredis.Redis, api_key: str, limit: int) -> int:
    """Return seconds-to-wait if the bucket is empty, else 0 (request allowed).

    Permissive pre-check: we don't know the cost upfront, so we let any request
    through whenever *some* budget remains. The real charge happens after the
    upstream finishes and we have a usage chunk.
    """
    raw = await redis.get(quota_key(api_key))
    if raw is None or int(raw) < limit:
        return 0
    ttl = await redis.ttl(quota_key(api_key))
    # ttl may be -1 (no expiry) or -2 (missing) due to a race — fall back to
    # the full window so the client doesn't hammer us in a tight loop.
    return ttl if ttl > 0 else WINDOW_SECONDS


async def quota_consume(redis: aioredis.Redis, api_key: str, tokens: int) -> None:
    """Charge ``tokens`` against the bucket. Set the TTL on the first write."""
    if tokens <= 0:
        return
    new_val = await redis.incrby(quota_key(api_key), tokens)
    if new_val == tokens:
        # First write of a fresh window — start the 60s refill timer.
        await redis.expire(quota_key(api_key), WINDOW_SECONDS)

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
    try:
        yield
    finally:
        await app.state.redis.aclose()


app = FastAPI(lifespan=lifespan)


class ChatIn(BaseModel):
    message: str


async def retrieve(question: str, k: int = 3) -> list[tuple[int, str]]:
    """Return [(chunk_index, content), ...] for the top-K nearest chunks."""
    vec = embedder.encode([question], normalize_embeddings=True)[0].tolist()
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
        stream_options={"include_usage": True},
        # Cap to the tier budget so a runaway response can't blow it.
        max_tokens=min(1024, token_limit),
        messages=messages,
    )


@app.post("/chat/stream")
async def chat_stream(
    body: ChatIn,
    request: Request,
    caller: Caller = Depends(require_api_key),
):
    redis: aioredis.Redis = request.app.state.redis
    token_limit = TIERS[caller.tier]["token_limit"]

    # Token-bucket pre-check. We don't know the cost upfront, so this only
    # rejects when the bucket is already empty for this window.
    retry_after = await quota_retry_after(redis, caller.api_key, token_limit)
    if retry_after:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded for tier '{caller.tier}' "
                f"({token_limit} tokens / {WINDOW_SECONDS}s)"
            ),
            headers={"Retry-After": str(retry_after)},
        )

    rows = await retrieve(body.message)
    sources = [f"chunk_{i}" for i, _ in rows]
    context = "\n\n".join(f"[chunk_{i}] {c}" for i, c in rows)
    messages = [
        {"role": "system", "content": "Answer using only the provided context. Be concise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {body.message}"},
    ]

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
                buf += piece
                cut = max(buf.rfind(" "), buf.rfind("\n"))
                if cut >= 0:
                    yield sse({"type": "token", "content": buf[: cut + 1]})
                    buf = buf[cut + 1 :]
            # Flush any trailing partial word once the upstream stream ends.
            if buf:
                yield sse({"type": "token", "content": buf})

            inp = (usage.prompt_tokens if usage else 0) or 0
            out = (usage.completion_tokens if usage else 0) or 0
            cached = 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            # Charge the bucket with actual usage. Early disconnects skip this
            # because the `return` above exits before we reach here.
            await quota_consume(redis, caller.api_key, inp + out)
            yield sse({
                "type": "done",
                "model": model_name,
                "tier": caller.tier,
                "usage": {"input_tokens": inp, "output_tokens": out},
                "cost_usd": round(inp * price_in + out * price_out, 6),
                "cache_hit": cached > 0,
                "sources": sources,
            })
            completed = True
        finally:
            await resp.close()  # cancels the upstream HTTP request → stops billing
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
    }
