"""API-key auth: ``X-API-Key`` is required; missing -> 401, unknown -> 403."""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

# Demo keys -> tier. In production these would live in a secret store /
# database; the env-var loader from earlier iterations is easy to layer on top.
API_KEYS: dict[str, str] = {
    "demo-free-key": "demo-free",
    "demo-pro-key": "demo-pro",
    "demo-enterprise-key": "demo-enterprise",
}


@dataclass(frozen=True)
class Caller:
    api_key: str
    tier: str


# auto_error=False so we control the 401 body / WWW-Authenticate header.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)) -> Caller:
    """Validate ``X-API-Key`` and return the caller (key + tier)."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    tier = API_KEYS.get(api_key)
    if tier is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return Caller(api_key=api_key, tier=tier)
