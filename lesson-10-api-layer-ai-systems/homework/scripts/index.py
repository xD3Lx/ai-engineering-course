"""Index ``data/twelve.md`` into pgvector on Supabase.

Pipeline:
    1. Read ``data/twelve.md``.
    2. Split into ~256-token chunks with 50-token overlap, using the
       tokenizer that belongs to the embedding model so the chunk
       boundaries line up with the model's vocabulary.
    3. Embed each chunk locally with ``all-MiniLM-L6-v2`` (384 dims).
    4. Upsert the chunks into a ``documents`` table on a Supabase
       Postgres instance with the ``pgvector`` extension enabled.

Run:
    uv run python scripts/index.py

    ``DB_URL`` can be exported in the shell or placed in a project-root
    ``.env`` file as ``DB_URL=postgresql://...``.

Pre-requisites on Supabase:
    Database -> Extensions -> enable ``vector``.

Dependencies (add once):
    uv add "psycopg[binary]" pgvector sentence-transformers langchain-text-splitters
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimensionality
CHUNK_TOKENS = 256
CHUNK_OVERLAP = 50

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = PROJECT_ROOT / "data" / "twelve.md"
SOURCE_NAME = SOURCE_PATH.name  # stored alongside each chunk for traceability
ENV_PATH = PROJECT_ROOT / ".env"

TABLE_NAME = "documents"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def read_document(path: Path) -> str:
    if not path.exists():
        sys.exit(f"Source file not found: {path}")
    return path.read_text(encoding="utf-8")


def read_env_file_value(path: Path, key: str) -> str | None:
    """Read a single key from a simple .env file without an extra dependency."""
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()

        name, separator, value = line.partition("=")
        if separator and name.strip() == key:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            return value

    return None


def get_db_url() -> str:
    db_url = os.environ.get("DB_URL") or read_env_file_value(ENV_PATH, "DB_URL")
    if not db_url:
        sys.exit(
            "DB_URL is not set. Export it in the shell or add "
            f"DB_URL=postgresql://... to {ENV_PATH}."
        )
    return db_url


def split_into_chunks(text: str) -> list[str]:
    """Token-based split using the embedding model's tokenizer.

    Using ``from_huggingface_tokenizer`` makes ``chunk_size`` count actual
    model tokens (not characters), so each chunk fits a predictable budget.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Encode chunks locally with sentence-transformers."""
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # makes cosine == dot product, cheaper queries
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create the extension, table, and an ANN index if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id           BIGSERIAL PRIMARY KEY,
                source       TEXT      NOT NULL,
                chunk_index  INT       NOT NULL,
                content      TEXT      NOT NULL,
                embedding    VECTOR({EMBED_DIM}) NOT NULL,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (source, chunk_index)
            )
            """
        )
        # HNSW + cosine; works well for normalised embeddings like MiniLM's.
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_hnsw
            ON {TABLE_NAME}
            USING hnsw (embedding vector_cosine_ops)
            """
        )
    conn.commit()


def upsert_chunks(
    conn: psycopg.Connection,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    """Replace every chunk for SOURCE_NAME in one transaction (idempotent re-runs)."""
    assert len(chunks) == len(embeddings), "chunks and embeddings length mismatch"

    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {TABLE_NAME} WHERE source = %s", (SOURCE_NAME,))
        cur.executemany(
            f"""
            INSERT INTO {TABLE_NAME} (source, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            [
                (SOURCE_NAME, i, chunk, embedding)
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ],
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    db_url = get_db_url()

    print(f"Reading {SOURCE_PATH}")
    text = read_document(SOURCE_PATH)

    print(f"Splitting into ~{CHUNK_TOKENS}-token chunks (overlap {CHUNK_OVERLAP})")
    chunks = split_into_chunks(text)
    print(f"  -> {len(chunks)} chunks")

    print(f"Embedding with {MODEL_NAME}")
    embeddings = embed_chunks(chunks)

    print("Connecting to Supabase / pgvector")
    with psycopg.connect(db_url) as conn:
        register_vector(conn)
        ensure_schema(conn)
        print(f"Upserting {len(chunks)} chunks into '{TABLE_NAME}'")
        upsert_chunks(conn, chunks, embeddings)

    print("Done.")


if __name__ == "__main__":
    main()
