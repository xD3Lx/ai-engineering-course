"""Single source of truth for model pricing.

Prices are USD per 1,000,000 tokens. Anything that needs to compute a cost —
the SSE `done` event, the `usage_log` table, the `/usage` endpoints — should
call :func:`cost_usd` so the math stays consistent everywhere.

A model that doesn't appear in this map is treated as free (cost 0).
"""
from __future__ import annotations

PRICING: dict[str, dict[str, float]] = {
    # demo-free — OpenRouter ":free" variants
    "google/gemma-4-31b-it:free": {"input": 0.0, "output": 0.0},
    "meta-llama/llama-3.3-70b-instruct:free": {"input": 0.0, "output": 0.0},
    "meta-llama/llama-3.2-3b-instruct:free": {"input": 0.0, "output": 0.0},
    # demo-pro
    "google/gemma-4-31b-it": {"input": 0.12, "output": 0.37},
    "openai/gpt-5.4-nano": {"input": 0.18, "output": 1.25},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    # demo-enterprise
    "openai/gpt-5.4-mini": {"input": 0.75, "output": 4.5},
    "google/gemini-3.5-flash": {"input": 1.5, "output": 9.0},
    "anthropic/claude-haiku-4.5": {"input": 0.8, "output": 4.0},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of a (model, input_tokens, output_tokens) triple.

    Rounded to 6 decimal places to keep the numbers stable across DB roundtrips.
    Unknown models cost 0 — preferable to crashing, since pricing data may lag
    when a new model is added to a tier.
    """
    p = PRICING.get(model)
    if not p:
        return 0.0
    cost = input_tokens * p["input"] / 1_000_000 + output_tokens * p["output"] / 1_000_000
    return round(cost, 6)
