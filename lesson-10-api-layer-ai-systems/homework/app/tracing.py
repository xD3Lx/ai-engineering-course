"""Langfuse tracing — optional, never breaks the request.

Thin wrapper over the Langfuse Python SDK (v3+, OpenTelemetry-based). We use
*manual* observations (``start_observation`` / ``parent.start_observation``)
rather than the ``with`` context managers because the request's LLM span must
stay open across the SSE generator, which outlives the handler function — a
``with`` block can't span that boundary, but a manually-ended observation can.

Design goals:
    - Zero hard dependency at runtime: if ``langfuse`` isn't installed or the
      LANGFUSE_* env vars are unset, every helper degrades to a no-op.
    - Never raise into the request path: all SDK calls are guarded.

Configure via env (Langfuse Cloud free tier):
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or us./jp. region
"""
from __future__ import annotations

import os
from typing import Any

_client: Any = None
_enabled: bool = False


class _NoopObservation:
    """Stand-in returned when tracing is disabled. Every method is a no-op and
    returns ``self`` so chained calls and nested ``start_*`` keep working."""

    trace_id = None
    id = None

    def update(self, *args: Any, **kwargs: Any) -> "_NoopObservation":
        return self

    def update_trace(self, *args: Any, **kwargs: Any) -> "_NoopObservation":
        return self

    def end(self, *args: Any, **kwargs: Any) -> "_NoopObservation":
        return self

    def start_observation(self, *args: Any, **kwargs: Any) -> "_NoopObservation":
        return self


_NOOP = _NoopObservation()


def init_tracing() -> bool:
    """Initialise the Langfuse client once at startup. Returns whether tracing
    is active. Safe to call when unconfigured — flips tracing to no-op mode."""
    global _client, _enabled
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        _client, _enabled = None, False
        return False
    try:
        from langfuse import get_client

        _client = get_client()
        # auth_check pings Langfuse; if creds are wrong we stay in no-op mode
        # rather than emitting spans that will be rejected.
        _enabled = bool(_client.auth_check())
    except Exception:
        _client, _enabled = None, False
    return _enabled


def enabled() -> bool:
    return _enabled and _client is not None


def start_root(name: str, **kwargs: Any) -> Any:
    """Open a root span (defines the trace). Returns a no-op when disabled."""
    if not enabled():
        return _NOOP
    try:
        return _client.start_observation(as_type="span", name=name, **kwargs)
    except Exception:
        return _NOOP


def start_child(parent: Any, name: str, as_type: str = "span", **kwargs: Any) -> Any:
    """Open a child observation under ``parent`` (a span/generation object)."""
    if parent is None or parent is _NOOP or not enabled():
        return _NOOP
    try:
        return parent.start_observation(name=name, as_type=as_type, **kwargs)
    except Exception:
        return _NOOP


def update(obs: Any, **kwargs: Any) -> None:
    """Update an observation's fields (output, metadata, usage_details, ...)."""
    if obs is None or obs is _NOOP:
        return
    try:
        obs.update(**kwargs)
    except Exception:
        pass


def set_trace(obs: Any, **kwargs: Any) -> None:
    """Set trace-level attributes (tags, user_id, session_id, metadata, name).

    Tags are how the per-request labels (model, api_key, cache_hit,
    fallback_used, tier) become filterable in the Langfuse UI.
    """
    if obs is None or obs is _NOOP:
        return
    try:
        obs.update_trace(**kwargs)
    except Exception:
        pass


def end(obs: Any, **kwargs: Any) -> None:
    """End an observation. Optionally pass final fields via kwargs first. The
    span is ended even if applying the fields fails, so spans never leak."""
    if obs is None or obs is _NOOP:
        return
    try:
        if kwargs:
            obs.update(**kwargs)
    except Exception:
        pass
    try:
        obs.end()
    except Exception:
        pass


def flush() -> None:
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            pass


def shutdown() -> None:
    """Flush buffered spans and stop background threads (call on app shutdown)."""
    if _client is not None:
        try:
            _client.shutdown()
        except Exception:
            pass
