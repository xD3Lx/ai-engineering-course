"""Q&A bot — pgvector retrieval + OpenRouter SSE streaming."""
from __future__ import annotations

import json
import os
from collections import Counter
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
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

ACTIVE_TIER = os.environ.get("TIER", "demo-free")


def model_for(tier: str, slot: int) -> tuple[str, float, float] | None:
    """Return ``(name, price_in_per_mtok, price_out_per_mtok)`` for the tier slot,
    or None if the tier has no model at that position."""
    models = TIERS[tier]["models"]
    return models[slot] if slot < len(models) else None


PRIMARY = model_for(ACTIVE_TIER, 0)
SECONDARY = model_for(ACTIVE_TIER, 1)
TERTIARY = model_for(ACTIVE_TIER, 2)
# Ordered chain — try primary first, fall through to the next on open-time errors.
FALLBACK_CHAIN: list[tuple[str, float, float]] = [
    m for m in (PRIMARY, SECONDARY, TERTIARY) if m is not None
]
TOKEN_LIMIT = TIERS[ACTIVE_TIER]["token_limit"]

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


async def open_completion(client: AsyncOpenAI, model_name: str, messages: list[dict]):
    return await client.chat.completions.create(
        model=model_name,
        stream=True,
        stream_options={"include_usage": True},
        # Cap to the tier budget so a runaway response can't blow it.
        max_tokens=min(1024, TOKEN_LIMIT),
        messages=messages,
    )


@app.post("/chat/stream")
async def chat_stream(body: ChatIn, request: Request):
    rows = await retrieve(body.message)
    sources = [f"chunk_{i}" for i, _ in rows]
    context = "\n\n".join(f"[chunk_{i}] {c}" for i, c in rows)
    messages = [
        {"role": "system", "content": "Answer using only the provided context. Be concise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {body.message}"},
    ]

    # Try each model in order (primary -> secondary -> tertiary). Fallback only
    # happens on failure to *open* the stream; we don't retry mid-stream — that
    # would send the client a second answer after they've already seen tokens.
    llm: AsyncOpenAI = request.app.state.llm
    chosen: tuple[str, float, float] | None = None
    resp = None
    last_error: Exception | None = None
    for candidate in FALLBACK_CHAIN:
        try:
            resp = await open_completion(llm, candidate[0], messages)
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
            async for chunk in resp:
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content
                if not token:
                    continue
                if await request.is_disconnected():
                    return  # finally-block bumps aborted_streams + closes resp
                yield sse({"type": "token", "content": token})

            inp = (usage.prompt_tokens if usage else 0) or 0
            out = (usage.completion_tokens if usage else 0) or 0
            cached = 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            yield sse({
                "type": "done",
                "model": model_name,
                "tier": ACTIVE_TIER,
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
        "tier": ACTIVE_TIER,
        "primary_model": PRIMARY[0] if PRIMARY else None,
        "secondary_model": SECONDARY[0] if SECONDARY else None,
        "tertiary_model": TERTIARY[0] if TERTIARY else None,
        "completed_streams": metrics["completed_streams"],
        "aborted_streams": metrics["aborted_streams"],
    }
