# Production-Ready RAG API — Q&A bot over the Twelve-Factor App

A minimal, from-scratch RAG (Retrieval-Augmented Generation) service wrapped in a production API with every layer from the assignment: SSE streaming, semantic cache, token-based rate limiting, cost tracking, multi-provider fallback, prompt-injection defense, concurrency control, observability, and a public deploy.

The indexed document is the [Twelve-Factor App](https://12factor.net/) methodology (`data/twelve.md`). Ask it a question and it retrieves the relevant chunks, feeds them to an LLM as context, and streams back a grounded answer.

> **No high-level RAG abstractions.** No LangChain `RetrievalQA`, no LlamaIndex `QueryEngine`. The retrieval, prompting, cache, and fallback logic are hand-written so the mechanics are visible. Only low-level building blocks are used (OpenAI SDK, vector-DB clients, a text splitter, an embedding model).

---

## Request pipeline

```
POST /chat/stream
  auth (X-API-Key → tier)
   → input guardrails (length + injection)
   → embed query (one embedding, reused)
   → semantic cache check ──HIT──► replay cached answer token-by-token
   → vector search (pgvector top-k=3)
   → rate-limit reserve (token bucket)
   → concurrency gate (semaphore)
   → LLM call (OpenRouter, fallback chain + circuit breaker)
   → stream tokens (SSE)
   → output filter (system-prompt leak scan)
   → settle tokens + log cost + close Langfuse trace
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.14 · FastAPI · Uvicorn · Pydantic |
| LLM gateway | OpenRouter (OpenAI-compatible) via the `openai` async SDK |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim, local CPU) |
| RAG vector store | **pgvector** in Postgres (Supabase), HNSW + cosine |
| Semantic cache | **Qdrant Cloud** (separate `cache_collection`) |
| Rate limit + metrics | **Redis** (Upstash) — token bucket + global counters |
| Cost tracking | **Postgres** (Supabase) — `usage_log` table |
| Observability | **Langfuse** Cloud (OpenTelemetry-based v3 SDK) |
| Packaging | pip + `requirements.txt`, multi-stage Docker (CPU-only torch) |
| Deploy | **Fly.io** (`auto_stop_machines = off`) |

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat/stream` | Main RAG chat endpoint — SSE streaming. Body: `{"message": str}` |
| `GET` | `/usage/today` | Today's totals for the caller: requests / tokens / cost |
| `GET` | `/usage/breakdown` | Per-model breakdown + cache-hit rate, fallback rate, avg/p95 latency |
| `GET` | `/health` | Liveness + live counters (`active_streams`, `aborted_streams`, cache hits/misses) |

Re-indexing the document is done with the `scripts/index.py` CLI rather than an admin HTTP endpoint (see [Scripts](#scripts)).

### `done` SSE event shape

```json
{"type": "done", "model": "google/gemma-4-31b-it", "tier": "demo-pro", "usage": {"input_tokens": 0, "output_tokens": 0, "estimated": false}, 
"cost_usd": 0.0, "cache_hit": true, "cache_score": 0.9692786,
"sources": ["chunk_0", "chunk_15", "chunk_19"], "request_id": "0ce8a885-7358-484e-bd85-8e50bf75798c", "latency_ms": 257, "ttft_ms": 239}
```

---

## Project structure

```
app/
  main.py        FastAPI app, lifespan, the /chat/stream handler + SSE generator, /health, /usage/*
  auth.py        X-API-Key auth → Caller(api_key, tier)
  tiers.py       Per-tier token budgets + ordered model fallback chains
  ratelimit.py   Redis token bucket: reserve / settle / refund
  metrics.py     Redis-backed global counters
  rag.py         Embedding model + pgvector retrieval
  cache.py       Qdrant semantic cache (lookup / store / token-by-token replay)
  llm.py         OpenRouter calls, fallback chain, timeout, circuit breaker, usage extraction
  pricing.py     Single source of truth for model pricing → cost_usd()
  usage_log.py   Per-request Postgres log + /usage read queries
  guardrails.py  Input length/injection checks, output leak scan, suspicious logs, system prompt
  tracing.py     Optional Langfuse tracing wrapper (no-op when unconfigured)
scripts/
  index.py          Build the pgvector index from data/twelve.md
  ask_samples.sh    Fire 20 sample questions at the API (smoke test / cache warming)
  burst_twelve.sh   One heavy request per factor (rate-limit stress test)
data/twelve.md      The indexed document (Twelve-Factor App)
Dockerfile          Multi-stage, pip-based, CPU-only torch (~2 GB)
requirements.txt    Direct dependencies only (pip resolves the rest)
fly.toml            Fly.io deploy config
```

---

## Implementation steps (what was built)

### 1 · RAG base layer
`scripts/index.py` reads `data/twelve.md`, splits it into ~256-token chunks with 50-token overlap using `langchain-text-splitters` driven by the **embedding model's own tokenizer** (so chunk boundaries align with the model vocab), embeds each chunk locally with `all-MiniLM-L6-v2` (384 dims, normalized), and upserts into a `documents` table in **Supabase Postgres** with a `pgvector` HNSW cosine index. At query time `app/rag.py` embeds the query once and returns top-k=3 chunks. The `done` event carries `sources: [chunk_…]`. *Tech: sentence-transformers, langchain-text-splitters, psycopg, pgvector.*

### 2 · FastAPI + SSE streaming
`POST /chat/stream` returns a `StreamingResponse(media_type="text/event-stream")`. An async generator yields `{"type":"token"}` events with a word-boundary buffer (so each event ends on whitespace) and a final `{"type":"done"}` with usage, cost, `cache_hit`, `sources`, `latency_ms`, and `ttft_ms`. Client disconnects are detected via `await request.is_disconnected()`, which stops the stream, cancels upstream, and bumps `aborted_streams`. *Tech: FastAPI, Starlette StreamingResponse.*

### 3 · Auth (API keys)
`X-API-Key` header is required (missing → `401`, unknown → `403`). Three demo keys (`demo-free-key`, `demo-pro-key`, `demo-enterprise-key`) map to tiers in `app/auth.py`; each tier's token budget and ordered model chain live in `app/tiers.py`. *Tech: FastAPI `APIKeyHeader` dependency.*

### 4 · Token-based rate limiting
A per-API-key **token bucket in Redis** charges *actual* input+output tokens, not request count. The flow reserves a worst-case estimate atomically (`INCRBY`), then settles to the real usage after the stream (or refunds on abort). The window refills over 60s; exceeding it returns `429` with a `Retry-After` header. Uses plain Redis commands (Upstash-compatible, no Lua). *Tech: Redis (Upstash), `app/ratelimit.py`.*

### 5 · Semantic cache
The **same embedding** computed for retrieval is reused for the cache lookup (one embed per request). Cache vectors live in a separate **Qdrant Cloud** `cache_collection`; a hit (cosine > 0.92) short-circuits the LLM entirely and the stored answer is **replayed token-by-token** for UX parity. Entries carry a 1-hour TTL via an `expire_at` payload field (filtered at query time, since Qdrant has no built-in TTL). The cache is global to the document. *Tech: Qdrant Cloud, `app/cache.py`.*

### 6 · Cost tracking
Every served request writes one row to the Postgres `usage_log` table: `request_id, api_key, model, input/output_tokens, cost_usd, latency_ms, ttft_ms, cache_hit, fallback_used, output_filtered`. `app/pricing.py` is the single source for prices (USD per 1M tokens). `GET /usage/today` returns totals; `GET /usage/breakdown` returns per-model rollups plus `cache_hit_rate`, `fallback_rate`, and avg/p95 latency (Postgres `PERCENTILE_CONT`). *Tech: Postgres (Supabase), psycopg.*

### 7 · Multi-provider fallback
All calls go through **OpenRouter** (`https://openrouter.ai/api/v1`) via the OpenAI SDK. `app/llm.py` walks the tier's model chain (primary → fallback 1 → fallback 2). Each open attempt is bounded by a **15s `asyncio.wait_for`**. Retryable failures (`429`, `5xx`, timeouts, network errors, and a model-not-found `400`) advance to the next model; terminal failures (`400/401/403/422`, content filter) are surfaced to the client. An in-process **circuit breaker** trips after 5 errors in 60s and skips a flapping model for 60s. The cost record's `fallback_used` flag and `model` field reflect the model that actually served. *Tech: OpenRouter, `openai` async SDK.*

### 8 · Security — prompt-injection defense
`app/guardrails.py` enforces a 4,000-char input cap (`400` on overflow) and scans input against **12 case-insensitive injection patterns** (`ignore previous instructions`, `system:`, `<|im_start|>`, `</s>`, jailbreak/role-switch markers…); a match logs to `logs/suspicious_requests.log` and returns `400`. After streaming, the accumulated answer is scanned for leaked **system-prompt fragments** — on a hit the cost record gets `output_filtered=true`, the response is logged to `logs/suspicious_responses.log`, and it is not cached. The system prompt is hardened with **role separation + XML envelopes** (`<context>…</context>`, `<user_query>…</user_query>`) and forged delimiters are stripped from user input. *Tech: regex, structured logging.*

### 9 · Async / concurrency control
An `asyncio.Semaphore(20)` caps concurrent in-flight LLM streams (protecting OpenRouter limits and memory during spikes); excess requests wait for a slot. A client disconnect propagates `CancelledError` into the SDK, cancelling the OpenRouter request — those tokens are **not** charged to the rate limiter and **not** logged. Cleanup (slot release, refund, counter updates) runs under `asyncio.shield` so it completes even mid-cancel. `GET /health` exposes the `active_streams` gauge and `aborted_streams` counter. *Tech: asyncio.*

### 10 · Observability — Langfuse
`app/tracing.py` wraps the **Langfuse v3 (OpenTelemetry) SDK** and traces the full pipeline as one trace per request with child spans: `auth → input_guardrails → embed_query → cache_check → vector_search → rate_limit → llm_call`. Spans carry `model`, `api_key`, `cache_hit`, `fallback_used`, `tier`; the LLM generation logs the full **prompt** (system + retrieved chunks + user query) and **completion** with token usage and cost — for debugging hallucinations and RAG retrieval. Tracing is optional: it no-ops cleanly when `LANGFUSE_*` env vars are unset. *Tech: Langfuse Cloud, OpenTelemetry.*

### 11 · Deployment & image optimization
Deployed on **Fly.io** (`fly.toml`, `auto_stop_machines = off` so streaming survives cold starts). The container is a **multi-stage, pip-based Dockerfile** built from `python:3.14-slim-trixie` (no ghcr dependency). The image was cut from **~8 GB to ~2 GB** by installing **CPU-only torch** from the PyTorch wheel index (dropping ~5–6 GB of unused CUDA libraries) and copying only the venv + source into a clean runtime stage. *Tech: Docker (BuildKit), pip, Fly.io.*

---

## Running locally

Requirements: Python 3.14, a Redis URL, a Postgres URL, an OpenRouter key. Qdrant and Langfuse are optional (the cache falls back to in-memory; tracing no-ops).

```bash
# 1. install deps
pip install -r requirements.txt

# 2. configure environment (see below), then build the index
python scripts/index.py

# 3. run the API
uvicorn app.main:app --env-file .env --host 0.0.0.0 --port 8080
```

Smoke-test it:

```bash
# streaming (tokens arrive one by one)
curl -N -X POST http://localhost:8080/chat/stream \
  -H "X-API-Key: demo-free-key" -H "Content-Type: application/json" \
  -d '{"message":"What does the Config factor recommend?"}'

# 20 sample questions + summary (run twice to see the cache kick in)
./scripts/ask_samples.sh
```

### With Docker

```bash
docker build -t homework10-api .
docker run --rm -p 8080:8080 --env-file .env homework10-api
```

---

## Environment variables

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key (LLM calls) |
| `REDIS_URL` | Redis/Upstash connection (rate limit + metrics) |
| `DB_URL` | Postgres connection (pgvector index + `usage_log`) |
| `QDRANT_URL` | Qdrant Cloud endpoint (`https://…:6333`). Unset → in-memory cache |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | Langfuse tracing. Unset → tracing no-ops |
| `GUARDRAILS_LOG_DIR` | Optional override for the suspicious-request/response log directory |

A `.env.example` with placeholder keys is included; fill it with your own credentials.

---

## Scripts

| Script | What it does |
|---|---|
| `scripts/index.py` | Chunk + embed `data/twelve.md` and upsert into pgvector (idempotent re-runs). Run this to (re)index. |
| `scripts/ask_samples.sh` | Send 20 natural Twelve-Factor questions to `/chat/stream`; prints per-request model/tokens/cost/latency/cache and a summary. Good for warming the cache and eyeballing latency. |
| `scripts/burst_twelve.sh` | One heavy prompt per factor — designed to trip the rate limiter (`429` + `Retry-After`). |

---

## Configuration notes

- **Models** are configured per tier in `app/tiers.py` and priced in `app/pricing.py`. Swap in valid OpenRouter model IDs for your account/budget — use `:free` variants for development. Pricing for an unknown model defaults to `$0`.
- **CPU-only torch** is pinned via `--extra-index-url https://download.pytorch.org/whl/cpu` in `requirements.txt`; pip prefers the `+cpu` build over the CUDA one. Keep that line if you regenerate the file.
- `requirements.txt` lists **direct dependencies only**; the transitive tree is resolved at install time. `pyproject.toml` / `uv.lock` remain in the repo for exact-pin regeneration if needed.

---

## Deployment

Deployed to Fly.io (app `homework-10-final`, region `fra`):

```bash
fly secrets set OPENROUTER_API_KEY=... REDIS_URL=... DB_URL=... \
                QDRANT_URL=... QDRANT_API_KEY=... \
                LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... LANGFUSE_BASE_URL=...
fly deploy
```

Public URL: `https://homework-10-final.fly.dev`
