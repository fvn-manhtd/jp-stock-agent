"""ASGI middleware for auth & rate limiting on HTTP/SSE transports.

For ``stdio`` transport (Claude Desktop, Cursor), auth is skipped because
the server runs as a local subprocess – the user *is* the owner.

For ``sse`` and ``http`` transports (MCP marketplace, web clients),
the middleware intercepts every request and checks:
1. API key in ``Authorization: Bearer <key>`` header or ``X-API-Key`` header
2. Tier-based tool access
3. Rate limits

Usage
-----
Wrap the FastMCP ASGI app::

    from .middleware import AuthMiddleware
    asgi_app = AuthMiddleware(mcp_app.asgi_app)
"""

from __future__ import annotations

import json
import time
from typing import Any

from .auth import (
    AuthResult,
    get_daily_limit,
    get_key_store,
)
from .config import get_settings
from .ratelimit import get_rate_limiter
from .usage import get_usage_tracker


def _extract_api_key(headers: list[tuple[bytes, bytes]]) -> str:
    """Extract API key from request headers.

    Checks (in order):
    1. ``Authorization: Bearer <key>``
    2. ``X-API-Key: <key>``
    """
    for name, value in headers:
        name_lower = name.lower()
        if name_lower == b"authorization":
            val = value.decode("utf-8", errors="replace")
            if val.lower().startswith("bearer "):
                return val[7:].strip()
        elif name_lower == b"x-api-key":
            return value.decode("utf-8", errors="replace").strip()
    return ""


def _json_response(status: int, body: dict[str, Any]) -> dict:
    """Build a minimal ASGI JSON response spec."""
    return {
        "status": status,
        "headers": [(b"content-type", b"application/json")],
        "body": json.dumps(body).encode("utf-8"),
    }


class AuthMiddleware:
    """ASGI middleware that enforces auth + rate limiting.

    Parameters
    ----------
    app : ASGI application
        The underlying FastMCP ASGI application.
    """

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            # Pass through non-HTTP (e.g. lifespan, websocket)
            await self.app(scope, receive, send)
            return

        settings = get_settings()

        # Skip auth entirely if disabled
        if not settings.jpstock_auth_enabled:
            # Still apply anonymous rate limiting if enabled
            if settings.jpstock_rate_limit_enabled:
                limiter = get_rate_limiter(
                    default_daily_limit=get_daily_limit("pro"),
                    burst_per_minute=settings.jpstock_burst_per_minute,
                )
                rl = limiter.check("_anonymous_")
                if not rl.allowed:
                    resp = _json_response(429, {
                        "error": rl.error,
                        "retry_after_seconds": round(rl.retry_after, 1),
                    })
                    await _send_response(send, resp)
                    return
            await self.app(scope, receive, send)
            return

        # -- Extract key --
        headers = scope.get("headers", [])
        api_key = _extract_api_key(headers)

        # Master key bypass
        if settings.jpstock_master_key and api_key == settings.jpstock_master_key:
            await self.app(scope, receive, send)
            return

        if not api_key:
            resp = _json_response(401, {
                "error": "Authentication required.",
                "hint": "Set Authorization: Bearer <api_key> header.",
            })
            await _send_response(send, resp)
            return

        # -- Validate key --
        store = get_key_store(settings.jpstock_auth_key_file or None)
        result: AuthResult = store.validate(api_key)

        if not result.authenticated:
            resp = _json_response(401, {"error": result.error})
            await _send_response(send, resp)
            return

        # -- Rate limit --
        if settings.jpstock_rate_limit_enabled:
            limiter = get_rate_limiter(
                default_daily_limit=get_daily_limit(result.tier),
                burst_per_minute=settings.jpstock_burst_per_minute,
            )
            limiter.set_limit(result.key_hash, get_daily_limit(result.tier))
            rl = limiter.check(result.key_hash)
            if not rl.allowed:
                resp = _json_response(429, {
                    "error": rl.error,
                    "retry_after_seconds": round(rl.retry_after, 1),
                    "daily_limit": rl.limit,
                })
                await _send_response(send, resp)
                return

        # -- Inject auth context into scope for downstream use --
        scope["auth"] = {
            "tier": result.tier,
            "owner": result.owner,
            "key_hash": result.key_hash,
            "authenticated_at": time.time(),
        }

        # -- Record usage (best-effort, don't block on failure) --
        try:
            path = scope.get("path", "")
            tool_name = path.rsplit("/", 1)[-1] if path else "unknown"
            tracker = get_usage_tracker()
            tracker.record(
                key_hash=result.key_hash,
                tier=result.tier,
                tool=tool_name,
            )
        except Exception:
            pass  # never let tracking break a real request

        await self.app(scope, receive, send)


async def _send_response(send: Any, resp: dict) -> None:
    """Send a complete HTTP response via ASGI."""
    await send({
        "type": "http.response.start",
        "status": resp["status"],
        "headers": resp["headers"],
    })
    await send({
        "type": "http.response.body",
        "body": resp["body"],
    })
