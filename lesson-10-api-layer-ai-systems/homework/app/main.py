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

TIERS = {
    "demo-free": {
        "token_limit": 5000,
        "models": [
            ("google/gemma-4-31b-it", 0.12, 0.37),
            ("meta-llama/llama-3.1-8b-instruct:free", 0.0, 0.0),
            ("meta-llama/llama-3.2-3b-instruct:free", 0.0, 0.0)
        ],
    }
}

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


@app.post("/chat/stream")
async def chat_stream(body: ChatIn, request: Request):
    rows = await retrieve(body.message)
    sources = [f"chunk_{i}" for i, _ in rows]
    context = "\n\n".join(f"[chunk_{i}] {c}" for i, c in rows)

    async def gen():
        completed = False
        resp = await request.app.state.llm.chat.completions.create(
            model=MODEL,
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=1024,
            messages=[
                {"role": "system", "content": "Answer using only the provided context. Be concise."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {body.message}"},
            ],
        )
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
                "usage": {"input_tokens": inp, "output_tokens": out},
                "cost_usd": round(inp * PRICE_IN + out * PRICE_OUT, 6),
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
        "completed_streams": metrics["completed_streams"],
        "aborted_streams": metrics["aborted_streams"],
    }
