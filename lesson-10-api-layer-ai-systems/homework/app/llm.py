"""OpenRouter (OpenAI-compatible) calls + small streaming helpers."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)

# --- fallback policy ---------------------------------------------------------

# Per-model open timeout. A slow upstream shouldn't pin the request — if the
# stream doesn't open within this budget we treat it as a retryable failure
# and move on to the next model in the chain.
CALL_TIMEOUT_SECONDS = 15.0

# HTTP status codes worth retrying on another model (transient upstream issues).
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
# Status codes that another model won't fix — the caller's input or auth is the
# problem (400 also covers content-filter rejections). Surfaced immediately.
NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 422})

# --- circuit breaker (in-process, per model) --------------------------------

CB_ERROR_THRESHOLD = 5      # retryable errors within the window to trip
CB_WINDOW_SECONDS = 60.0    # rolling window over which errors are counted
CB_OPEN_SECONDS = 60.0      # how long the breaker stays open once tripped

# Keyed by model name. Each entry: {"errors": [monotonic ts...], "open_until": ts}.
# In-process only — every worker keeps its own tally (no cross-instance coord).
_breaker: dict[str, dict] = {}


def _breaker_state(model: str) -> dict:
    state = _breaker.get(model)
    if state is None:
        state = {"errors": [], "open_until": 0.0}
        _breaker[model] = state
    return state


def _breaker_is_open(model: str, now: float) -> bool:
    """True while the breaker is open — callers should skip this model."""
    return now < _breaker_state(model)["open_until"]


def _record_error(model: str, now: float) -> None:
    """Log a retryable error; trip the breaker once threshold is hit in-window."""
    state = _breaker_state(model)
    cutoff = now - CB_WINDOW_SECONDS
    state["errors"] = [t for t in state["errors"] if t >= cutoff]
    state["errors"].append(now)
    if len(state["errors"]) >= CB_ERROR_THRESHOLD:
        state["open_until"] = now + CB_OPEN_SECONDS
        state["errors"].clear()


def _record_success(model: str) -> None:
    """A clean open resets the tally and closes the breaker."""
    state = _breaker_state(model)
    state["errors"].clear()
    state["open_until"] = 0.0


def _is_retryable(exc: Exception) -> bool:
    """Whether ``exc`` should trigger a fallback to the next model.

    Retryable: timeouts (our wait_for or the SDK's), network errors, and
    429/5xx from upstream. Non-retryable: 400/401/403/422 and content-filter
    rejections (surfaced as 400) — these won't succeed on another model.
    """
    if isinstance(exc, (asyncio.TimeoutError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        code = exc.status_code
        if code in NON_RETRYABLE_STATUS:
            return False
        if code in RETRYABLE_STATUS:
            return True
        # Any other 4xx is treated as terminal; unknown 5xx as retryable.
        return not (400 <= code < 500)
    # Unexpected error type — be conservative and try the next model.
    return True


async def open_completion(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    token_limit: int,
):
    """Open a streaming chat completion. Two redundant usage-include flags:
    ``stream_options`` is the OpenAI standard; OpenRouter additionally needs
    ``usage: {include: true}`` in the body (free models in particular skip
    the OpenAI-style flag)."""
    return await client.chat.completions.create(
        model=model,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"usage": {"include": True}},
        # Cap to the tier budget so a runaway response can't blow it.
        max_tokens=min(1024, token_limit),
        messages=messages,
    )


async def open_with_fallback(
    client: AsyncOpenAI,
    chain: list[str],
    messages: list[dict],
    token_limit: int,
) -> tuple[Any, str, bool]:
    """Try each model in ``chain`` (primary -> secondary -> ...) under the
    fallback policy. Returns ``(resp, model_name, fallback_used)`` where
    ``fallback_used`` is True iff a non-primary model served the request, and
    ``model_name`` is the model that actually opened the stream.

    Policy:
      * Each open attempt is bounded by ``CALL_TIMEOUT_SECONDS`` via
        ``asyncio.wait_for``; a timeout is a retryable failure -> next model.
      * Retryable failures (timeout, network error, 429/5xx) advance to the
        next model. Non-retryable failures (400/401/403/422, content filter)
        are raised immediately — another model won't help.
      * An in-process circuit breaker skips any model that produced
        ``CB_ERROR_THRESHOLD`` retryable errors within ``CB_WINDOW_SECONDS``,
        staying open ``CB_OPEN_SECONDS`` so a flapping primary isn't hammered
        and traffic goes straight to the fallback.

    Only retries on failure to *open* the stream; mid-stream errors are not
    re-tried because the client may have already seen partial tokens.
    """
    last_error: Exception | None = None
    for slot, model in enumerate(chain):
        if _breaker_is_open(model, time.monotonic()):
            # Breaker open for this model — skip straight to the next one.
            continue
        try:
            resp = await asyncio.wait_for(
                open_completion(client, model, messages, token_limit),
                timeout=CALL_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001
            if not _is_retryable(exc):
                # Terminal error (bad request, auth, content filter) — surface
                # to the client without trying any further model.
                raise
            last_error = exc
            _record_error(model, time.monotonic())
            continue
        _record_success(model)
        return resp, model, slot > 0
    # Every model either failed retryably or was skipped by an open breaker.
    raise last_error if last_error else RuntimeError(
        "No model available (all circuit breakers open)"
    )


def extract_usage(
    usage_obj: Any,
    *,
    prompt_chars: int,
    output_chars: int,
) -> tuple[int, int, bool, bool]:
    """Pull ``(input_tokens, output_tokens, estimated, prompt_cache_hit)`` out
    of the final usage chunk, falling back to a char/4 estimate when missing
    (common with free OpenRouter models that omit the usage chunk)."""
    if usage_obj:
        inp = (usage_obj.prompt_tokens or 0) or 0
        out = (usage_obj.completion_tokens or 0) or 0
        estimated = False
    else:
        inp = max(1, prompt_chars // 4)
        out = max(1, output_chars // 4)
        estimated = True
    prompt_cached = 0
    details = getattr(usage_obj, "prompt_tokens_details", None) if usage_obj else None
    if details is not None:
        prompt_cached = getattr(details, "cached_tokens", 0) or 0
    return inp, out, estimated, prompt_cached > 0
