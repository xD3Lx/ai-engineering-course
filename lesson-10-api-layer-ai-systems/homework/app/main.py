"""Q&A bot — pgvector retrieval + OpenRouter SSE streaming."""
from __future__ import annotations

import json
import os
from collections import Counter
from contextlib import asynccontextmanager

import psycopg
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


def require_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """Validate ``X-API-Key`` and return the caller's tier."""
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
    return tier

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
    yield


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
    tier: str = Depends(require_api_key),
):
    rows = await retrieve(body.message)
    sources = [f"chunk_{i}" for i, _ in rows]
    context = "\n\n".join(f"[chunk_{i}] {c}" for i, c in rows)
    messages = [
        {"role": "system", "content": "Answer using only the provided context. Be concise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {body.message}"},
    ]

    chain = fallback_chain(tier)
    token_limit = TIERS[tier]["token_limit"]

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
            yield sse({
                "type": "done",
                "model": model_name,
                "tier": tier,
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
