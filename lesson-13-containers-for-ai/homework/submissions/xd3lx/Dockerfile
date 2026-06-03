FROM python:3.11-slim AS builder

WORKDIR /build
COPY app/requirements.txt .
RUN pip install --no-cache-dir --target=/deps -r requirements.txt


FROM python:3.11-slim AS runtime

RUN groupadd --gid 1000 app \
  && useradd --create-home --uid 1000 --gid app app
WORKDIR /home/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/deps PATH=/deps/bin:$PATH

COPY --chown=app:app --from=builder /deps /deps
COPY --chown=app:app app/ ./app/
COPY --chown=app:app data/ ./data/

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import httpx, sys; r = httpx.get('http://localhost:8000/health', timeout=3); sys.exit(0 if r.json().get('status') == 'ok' else 1)"

USER app

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
