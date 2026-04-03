"""Tests for middleware module – ASGI auth & rate limiting."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from jpstock_agent.middleware import AuthMiddleware, _extract_api_key


# ---------------------------------------------------------------------------
# _extract_api_key
# ---------------------------------------------------------------------------


class TestExtractApiKey:
    def test_bearer_token(self):
        headers = [(b"authorization", b"Bearer jpsk_pro_abc123")]
        assert _extract_api_key(headers) == "jpsk_pro_abc123"

    def test_bearer_case_insensitive(self):
        headers = [(b"Authorization", b"bearer jpsk_pro_abc123")]
        assert _extract_api_key(headers) == "jpsk_pro_abc123"

    def test_x_api_key(self):
        headers = [(b"x-api-key", b"jpsk_free_xyz")]
        assert _extract_api_key(headers) == "jpsk_free_xyz"

    def test_no_key(self):
        headers = [(b"content-type", b"application/json")]
        assert _extract_api_key(headers) == ""

    def test_empty_headers(self):
        assert _extract_api_key([]) == ""

    def test_bearer_preferred_over_x_api_key(self):
        headers = [
            (b"authorization", b"Bearer from_bearer"),
            (b"x-api-key", b"from_header"),
        ]
        # First match wins
        assert _extract_api_key(headers) == "from_bearer"


# ---------------------------------------------------------------------------
# AuthMiddleware
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    """Integration tests using the middleware with mock ASGI app."""

    @pytest.fixture
    def mock_app(self):
        async def inner(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"ok": true}',
            })
        return inner

    @pytest.fixture
    def middleware(self, mock_app):
        return AuthMiddleware(mock_app)

    def _make_scope(self, headers=None):
        return {
            "type": "http",
            "headers": headers or [],
            "path": "/mcp/v1/tools/call",
            "method": "POST",
        }

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_passes_through_non_http(self, middleware):
        """Non-HTTP scopes (lifespan, etc.) pass through."""
        scope = {"type": "lifespan"}
        calls = []

        async def mock_send(msg):
            calls.append(msg)

        async def mock_receive():
            return {}

        self._run(middleware(scope, mock_receive, mock_send))
        # The mock app sends 2 messages (start + body) for HTTP,
        # but lifespan scope also goes through to app

    @patch("jpstock_agent.middleware.get_settings")
    def test_auth_disabled_passes(self, mock_settings, middleware):
        """When auth is disabled, requests pass through."""
        settings = MagicMock()
        settings.jpstock_auth_enabled = False
        settings.jpstock_rate_limit_enabled = False
        mock_settings.return_value = settings

        responses = []

        async def capture_send(msg):
            responses.append(msg)

        scope = self._make_scope()
        self._run(middleware(scope, MagicMock(), capture_send))
        # Should get 200 from mock app
        assert any(r.get("status") == 200 for r in responses)

    @patch("jpstock_agent.middleware.get_settings")
    def test_auth_enabled_no_key_401(self, mock_settings, middleware):
        """Missing API key returns 401."""
        settings = MagicMock()
        settings.jpstock_auth_enabled = True
        settings.jpstock_master_key = ""
        mock_settings.return_value = settings

        responses = []

        async def capture_send(msg):
            responses.append(msg)

        scope = self._make_scope(headers=[])
        self._run(middleware(scope, MagicMock(), capture_send))
        assert any(r.get("status") == 401 for r in responses)

    @patch("jpstock_agent.middleware.get_settings")
    def test_master_key_bypasses(self, mock_settings, middleware):
        """Master key bypasses all auth checks."""
        settings = MagicMock()
        settings.jpstock_auth_enabled = True
        settings.jpstock_master_key = "master_secret"
        mock_settings.return_value = settings

        responses = []

        async def capture_send(msg):
            responses.append(msg)

        headers = [(b"authorization", b"Bearer master_secret")]
        scope = self._make_scope(headers=headers)
        self._run(middleware(scope, MagicMock(), capture_send))
        assert any(r.get("status") == 200 for r in responses)

    @patch("jpstock_agent.middleware.get_rate_limiter")
    @patch("jpstock_agent.middleware.get_settings")
    def test_rate_limit_anonymous(self, mock_settings, mock_limiter, middleware):
        """Anonymous rate limiting when auth is disabled."""
        settings = MagicMock()
        settings.jpstock_auth_enabled = False
        settings.jpstock_rate_limit_enabled = True
        settings.jpstock_burst_per_minute = 30
        mock_settings.return_value = settings

        rl_result = MagicMock()
        rl_result.allowed = False
        rl_result.error = "Daily limit exceeded"
        rl_result.retry_after = 3600.0
        limiter_inst = MagicMock()
        limiter_inst.check.return_value = rl_result
        mock_limiter.return_value = limiter_inst

        responses = []

        async def capture_send(msg):
            responses.append(msg)

        scope = self._make_scope()
        self._run(middleware(scope, MagicMock(), capture_send))
        assert any(r.get("status") == 429 for r in responses)

    @patch("jpstock_agent.middleware.get_key_store")
    @patch("jpstock_agent.middleware.get_settings")
    def test_invalid_key_401(self, mock_settings, mock_store, middleware):
        """Invalid API key returns 401."""
        settings = MagicMock()
        settings.jpstock_auth_enabled = True
        settings.jpstock_master_key = ""
        settings.jpstock_auth_key_file = ""
        mock_settings.return_value = settings

        from jpstock_agent.auth import AuthResult
        store_inst = MagicMock()
        store_inst.validate.return_value = AuthResult(
            authenticated=False, error="Unknown API key"
        )
        mock_store.return_value = store_inst

        responses = []

        async def capture_send(msg):
            responses.append(msg)

        headers = [(b"authorization", b"Bearer jpsk_pro_bad")]
        scope = self._make_scope(headers=headers)
        self._run(middleware(scope, MagicMock(), capture_send))
        assert any(r.get("status") == 401 for r in responses)
