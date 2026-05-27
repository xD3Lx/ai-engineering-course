"""Per-tier token budget and ordered model fallback chain.

Pricing lives in :mod:`app.pricing`; this module is just the routing config.
"""
from __future__ import annotations

TIERS: dict[str, dict] = {
    "demo-free": {
        "token_limit": 5_000,
        "models": [
            "google/gemma-4-31b-it:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
        ],
    },
    "demo-pro": {
        "token_limit": 20_000,
        "models": [
            "google/gemma-4-31b-it",
            "openai/gpt-5.4-nano",
            "anthropic/claude-3-haiku",
        ],
    },
    "demo-enterprise": {
        "token_limit": 100_000,
        "models": [
            "openai/gpt-5.4-mini",
            "google/gemini-3.5-flash",
            "anthropic/claude-haiku-4.5",
        ],
    },
}


def model_for(tier: str, slot: int) -> str | None:
    """Return the model name for ``tier`` at ``slot``, or None if absent."""
    models = TIERS[tier]["models"]
    return models[slot] if slot < len(models) else None


def fallback_chain(tier: str) -> list[str]:
    """Ordered model chain for ``tier`` (primary -> secondary -> tertiary)."""
    return list(TIERS[tier]["models"])


def token_limit(tier: str) -> int:
    return int(TIERS[tier]["token_limit"])
