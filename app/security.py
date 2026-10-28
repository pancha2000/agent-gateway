"""API-key authentication. Your local PC sends the shared secret in `X-API-Key`."""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from .config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    # compare_digest -> constant-time comparison (no timing side channel)
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")
