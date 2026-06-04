# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — builder: resolve + install dependencies into /app/.venv with uv.
# The uv binary, build caches, and toolchain stay in this stage and never
# reach the final image.
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS builder

WORKDIR /app

# uv build settings:
#   - compile bytecode for faster cold starts
#   - copy (not hardlink) so the venv is self-contained and copyable to stage 2
#   - skip dev dependencies
#   - use the image's Python (don't download a standalone interpreter), so the
#     venv references /usr/local/bin/python — which also exists in the runtime
#     image, keeping the copied venv valid.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0

# Install dependencies first, without the project, for optimal layer caching.
# The uv cache is a BuildKit cache mount, so it never lands in any image layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Then add the source and install the project itself.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ---------------------------------------------------------------------------
# Stage 2 — runtime: a clean Python image with only the venv + app source.
# No uv, no build caches, no compiler toolchain.
# ---------------------------------------------------------------------------
FROM python:3.14-slim-trixie AS runtime

# Non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

WORKDIR /app

# Copy the fully-built application (source + /app/.venv) from the builder.
COPY --from=builder --chown=nonroot:nonroot /app /app

# Put the venv on PATH; sensible Python runtime defaults.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER nonroot

EXPOSE 8080

# Run the FastAPI app. `--host 0.0.0.0` to allow access from outside the container.
CMD ["uvicorn", "app.main:app", "--env-file", ".env", "--host", "0.0.0.0", "--port", "8080"]
