"""Tests for auth module – API key management, tiers, access control."""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from jpstock_agent.auth import (
    BASIC_TOOLS,
    TIERS,
    APIKey,
    AuthResult,
    KeyStore,
    _hash_key,
    _tier_from_key,
    check_tool_access,
    get_daily_limit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path):
    """KeyStore backed by a temporary file."""
    path = str(tmp_path / "keys.json")
    return KeyStore(path)


# ---------------------------------------------------------------------------
# _hash_key / _tier_from_key
# ---------------------------------------------------------------------------


class TestHashKey:
    def test_deterministic(self):
        assert _hash_key("foo") == _hash_key("foo")

    def test_different_inputs(self):
        assert _hash_key("a") != _hash_key("b")

    def test_returns_hex(self):
        h = _hash_key("test")
        assert len(h) == 64  # SHA-256 hex


class TestTierFromKey:
    def test_free(self):
        assert _tier_from_key("jpsk_free_abc123") == "free"

    def test_pro(self):
        assert _tier_from_key("jpsk_pro_abc123") == "pro"

    def test_enterprise(self):
        assert _tier_from_key("jpsk_ent_abc123") == "enterprise"

    def test_invalid(self):
        assert _tier_from_key("invalid_key") is None

    def test_empty(self):
        assert _tier_from_key("") is None


# ---------------------------------------------------------------------------
# APIKey dataclass
# ---------------------------------------------------------------------------


class TestAPIKey:
    def test_to_dict_roundtrip(self):
        ak = APIKey(
            key_hash="abc",
            tier="pro",
            owner="user@test.com",
            created_at=1000.0,
        )
        d = ak.to_dict()
        ak2 = APIKey.from_dict(d)
        assert ak2.key_hash == "abc"
        assert ak2.tier == "pro"
        assert ak2.owner == "user@test.com"
        assert ak2.active is True

    def test_inactive_key(self):
        ak = APIKey(
            key_hash="xyz",
            tier="free",
            owner="test",
            created_at=1000.0,
            active=False,
        )
        d = ak.to_dict()
        assert d["active"] is False
        ak2 = APIKey.from_dict(d)
        assert ak2.active is False


# ---------------------------------------------------------------------------
# KeyStore
# ---------------------------------------------------------------------------


class TestKeyStore:
    def test_generate_and_validate_free(self, tmp_store):
        raw = tmp_store.generate_key("free", "user@test.com")
        assert raw.startswith("jpsk_free_")
        result = tmp_store.validate(raw)
        assert result.authenticated is True
        assert result.tier == "free"
        assert result.owner == "user@test.com"

    def test_generate_and_validate_pro(self, tmp_store):
        raw = tmp_store.generate_key("pro", "pro@test.com")
        assert raw.startswith("jpsk_pro_")
        result = tmp_store.validate(raw)
        assert result.authenticated is True
        assert result.tier == "pro"

    def test_generate_and_validate_enterprise(self, tmp_store):
        raw = tmp_store.generate_key("enterprise", "ent@test.com")
        assert raw.startswith("jpsk_ent_")
        result = tmp_store.validate(raw)
        assert result.authenticated is True
        assert result.tier == "enterprise"

    def test_invalid_tier_raises(self, tmp_store):
        with pytest.raises(ValueError, match="Unknown tier"):
            tmp_store.generate_key("platinum", "user@test.com")

    def test_validate_empty(self, tmp_store):
        result = tmp_store.validate("")
        assert result.authenticated is False
        assert "No API key" in result.error

    def test_validate_bad_format(self, tmp_store):
        result = tmp_store.validate("not_a_valid_key")
        assert result.authenticated is False
        assert "Invalid" in result.error

    def test_validate_unknown_key(self, tmp_store):
        result = tmp_store.validate("jpsk_pro_0000000000000000")
        assert result.authenticated is False
        assert "Unknown" in result.error

    def test_revoke(self, tmp_store):
        raw = tmp_store.generate_key("pro", "user@test.com")
        assert tmp_store.revoke(raw) is True
        result = tmp_store.validate(raw)
        assert result.authenticated is False
        assert "deactivated" in result.error

    def test_revoke_unknown(self, tmp_store):
        assert tmp_store.revoke("jpsk_pro_nonexistent") is False

    def test_list_keys(self, tmp_store):
        tmp_store.generate_key("free", "a@test.com")
        tmp_store.generate_key("pro", "b@test.com")
        keys = tmp_store.list_keys()
        assert len(keys) == 2
        # Keys never contain raw key
        for k in keys:
            assert "key_hash_short" in k
            assert k["key_hash_short"].endswith("...")

    def test_list_keys_filter_owner(self, tmp_store):
        tmp_store.generate_key("free", "a@test.com")
        tmp_store.generate_key("pro", "b@test.com")
        keys = tmp_store.list_keys(owner="a@test.com")
        assert len(keys) == 1
        assert keys[0]["owner"] == "a@test.com"

    def test_get_tier(self, tmp_store):
        raw = tmp_store.generate_key("pro", "user@test.com")
        assert tmp_store.get_tier(raw) == "pro"

    def test_get_tier_invalid(self, tmp_store):
        assert tmp_store.get_tier("jpsk_pro_bad") is None

    def test_count(self, tmp_store):
        assert tmp_store.count == 0
        tmp_store.generate_key("free", "a@test.com")
        assert tmp_store.count == 1
        tmp_store.generate_key("pro", "b@test.com")
        assert tmp_store.count == 2

    def test_persistence(self, tmp_path):
        """Keys survive across store instances."""
        path = str(tmp_path / "keys.json")
        store1 = KeyStore(path)
        raw = store1.generate_key("pro", "persist@test.com")

        # New store instance reading same file
        store2 = KeyStore(path)
        result = store2.validate(raw)
        assert result.authenticated is True
        assert result.tier == "pro"

    def test_corrupt_file(self, tmp_path):
        """Gracefully handle corrupt key file."""
        path = str(tmp_path / "keys.json")
        with open(path, "w") as f:
            f.write("not valid json")
        store = KeyStore(path)
        assert store.count == 0


# ---------------------------------------------------------------------------
# check_tool_access / get_daily_limit
# ---------------------------------------------------------------------------


class TestToolAccess:
    def test_pro_all_access(self):
        assert check_tool_access("pro", "stock_history") is True
        assert check_tool_access("pro", "backtest_strategy") is True
        assert check_tool_access("pro", "ml_predict") is True

    def test_enterprise_all_access(self):
        assert check_tool_access("enterprise", "ml_predict") is True

    def test_free_basic_tools(self):
        for tool in ["stock_history", "company_overview", "fx_history"]:
            assert check_tool_access("free", tool) is True

    def test_free_blocked_tools(self):
        for tool in ["backtest_strategy", "ml_predict", "options_chain"]:
            assert check_tool_access("free", tool) is False

    def test_unknown_tier(self):
        assert check_tool_access("platinum", "stock_history") is False


class TestDailyLimit:
    def test_tiers(self):
        assert get_daily_limit("free") == 50
        assert get_daily_limit("pro") == 1000
        assert get_daily_limit("enterprise") == 5000

    def test_unknown_falls_to_free(self):
        assert get_daily_limit("unknown") == 50


# ---------------------------------------------------------------------------
# TIERS / BASIC_TOOLS consistency
# ---------------------------------------------------------------------------


class TestConstants:
    def test_all_tiers_have_required_fields(self):
        for name, info in TIERS.items():
            assert "daily_limit" in info
            assert "tools" in info
            assert "display_name" in info
            assert "description" in info

    def test_basic_tools_not_empty(self):
        assert len(BASIC_TOOLS) > 0
        assert "stock_history" in BASIC_TOOLS
