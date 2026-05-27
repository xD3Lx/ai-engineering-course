"""OpenRouter (OpenAI-compatible) calls + small streaming helpers."""
from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI


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
    """Try each model in ``chain`` (primary -> secondary -> ...). Returns
    ``(resp, model_name, fallback_used)`` or re-raises the last error if all
    fail. ``fallback_used`` is True iff the primary slot was bypassed.

    Only retries on failure to *open* the stream; mid-stream errors are not
    re-tried because the client may have already seen partial tokens.
    """
    last_error: Exception | None = None
    for slot, model in enumerate(chain):
        try:
            resp = await open_completion(client, model, messages, token_limit)
            return resp, model, slot > 0
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise last_error if last_error else RuntimeError("No models configured")


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
