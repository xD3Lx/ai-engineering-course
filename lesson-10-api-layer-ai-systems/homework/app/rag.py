"""Embedding model + pgvector retrieval.

Single embedding model instance, reused for both query embeddings (cache
lookup AND RAG retrieval) so we pay the sentence-transformers cost once per
request.
"""
from __future__ import annotations

import os

import psycopg
from pgvector.psycopg import register_vector_async
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimensionality

_embedder: SentenceTransformer | None = None


def init_embedder() -> SentenceTransformer:
    """Load the model into memory (called once from the FastAPI lifespan)."""
    global _embedder
    _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def embed(question: str) -> list[float]:
    """Encode once per request — reused for cache lookup and RAG retrieval."""
    assert _embedder is not None, "init_embedder() must be called at startup"
    return _embedder.encode([question], normalize_embeddings=True)[0].tolist()


async def retrieve(vec: list[float], k: int = 3) -> list[tuple[int, str]]:
    """Return ``[(chunk_index, content), ...]`` for the top-K nearest chunks.

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
