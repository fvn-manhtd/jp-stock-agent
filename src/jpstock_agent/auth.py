"""Authentication and API key management for jpstock-agent.

Tier system
-----------
- **free**       : 50  calls/day, basic tools only (quote, history, overview)
- **pro**        : 1000 calls/day, all tools
- **enterprise** : 5000 calls/day, all tools + priority support

API key format
--------------
``jpsk_{tier}_{random_hex}``  e.g. ``jpsk_pro_a1b2c3d4e5f6``

Storage
-------
Keys are stored in a JSON file (default ``~/.jpstock/keys.json``).
For production MCP marketplace deployment, replace with a database backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

TIERS: dict[str, dict[str, Any]] = {
    "free": {
        "display_name": "Free",
        "daily_limit": 50,
        "tools": "basic",  # basic = quote/history/overview only
        "description": "Basic access – stock quotes, history, company overview",
    },
    "pro": {
        "display_name": "Pro",
        "daily_limit": 1000,
        "tools": "all",
        "description": "Full access – all 100+ tools including TA, backtest, ML, options",
    },
    "enterprise": {
        "display_name": "Enterprise",
        "daily_limit": 5000,
        "tools": "all",
        "description": "Full access + priority support, higher rate limits",
    },
}

# Tools accessible at the "basic" (free) tier
BASIC_TOOLS: frozenset[str] = frozenset({
    "stock_history",
    "stock_history_batch",
    "stock_intraday",
    "company_overview",
    "company_news",
    "listing_all_symbols",
    "listing_sectors",
    "listing_symbols_by_market",
    "listing_symbols_by_sector",
    "fx_history",
    "crypto_history",
    "world_index_history",
    "financial_ratio",
    "financial_balance_sheet",
    "financial_income_statement",
    "financial_cash_flow",
    "stock_report_quick",
    "alert_list_conditions",
    "strategy_list_conditions",
})

# Key prefix → tier mapping
_PREFIX_MAP = {
    "jpsk_free": "free",
    "jpsk_pro": "pro",
    "jpsk_ent": "enterprise",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class APIKey:
    """Represents a stored API key."""

    key_hash: str  # SHA-256 hash of the full key
    tier: str
    owner: str  # email or identifier
    created_at: float  # Unix timestamp
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_hash": self.key_hash,
            "tier": self.tier,
            "owner": self.owner,
            "created_at": self.created_at,
            "active": self.active,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> APIKey:
        return cls(
            key_hash=d["key_hash"],
            tier=d["tier"],
            owner=d["owner"],
            created_at=d["created_at"],
            active=d.get("active", True),
            metadata=d.get("metadata", {}),
        )


@dataclass
class AuthResult:
    """Result of an authentication check."""

    authenticated: bool
    tier: str = "free"
    owner: str = ""
    error: str = ""
    key_hash: str = ""


# ---------------------------------------------------------------------------
# Key store
# ---------------------------------------------------------------------------

_DEFAULT_KEY_DIR = os.path.expanduser("~/.jpstock")
_DEFAULT_KEY_FILE = os.path.join(_DEFAULT_KEY_DIR, "keys.json")


class KeyStore:
    """File-based API key store.

    Thread-safe for reads; writes acquire no lock (acceptable for
    low-volume key management operations).
    """

    def __init__(self, path: str | None = None):
        self._path = path or _DEFAULT_KEY_FILE
        self._keys: dict[str, APIKey] = {}  # key_hash → APIKey
        self._load()

    # -- persistence --

    def _load(self) -> None:
        p = Path(self._path)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for item in data.get("keys", []):
                    ak = APIKey.from_dict(item)
                    self._keys[ak.key_hash] = ak
            except (json.JSONDecodeError, KeyError):
                self._keys = {}

    def _save(self) -> None:
        p = Path(self._path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"keys": [k.to_dict() for k in self._keys.values()]}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -- public API --

    def generate_key(self, tier: str, owner: str, metadata: dict | None = None) -> str:
        """Generate a new API key, store it, and return the raw key.

        The raw key is shown only once – only the hash is persisted.
        """
        if tier not in TIERS:
            raise ValueError(f"Unknown tier: {tier!r}. Valid: {list(TIERS)}")

        prefix = {"free": "jpsk_free", "pro": "jpsk_pro", "enterprise": "jpsk_ent"}[tier]
        raw = f"{prefix}_{secrets.token_hex(16)}"
        key_hash = _hash_key(raw)

        ak = APIKey(
            key_hash=key_hash,
            tier=tier,
            owner=owner,
            created_at=time.time(),
            active=True,
            metadata=metadata or {},
        )
        self._keys[key_hash] = ak
        self._save()
        return raw

    def validate(self, raw_key: str) -> AuthResult:
        """Validate a raw API key and return an :class:`AuthResult`."""
        if not raw_key:
            return AuthResult(authenticated=False, error="No API key provided")

        # Quick tier inference from prefix
        tier_from_prefix = _tier_from_key(raw_key)
        if tier_from_prefix is None:
            return AuthResult(authenticated=False, error="Invalid API key format")

        key_hash = _hash_key(raw_key)
        ak = self._keys.get(key_hash)

        if ak is None:
            return AuthResult(authenticated=False, error="Unknown API key")
        if not ak.active:
            return AuthResult(authenticated=False, error="API key is deactivated")

        return AuthResult(
            authenticated=True,
            tier=ak.tier,
            owner=ak.owner,
            key_hash=key_hash,
        )

    def revoke(self, raw_key: str) -> bool:
        """Revoke (deactivate) an API key. Returns True if found."""
        key_hash = _hash_key(raw_key)
        ak = self._keys.get(key_hash)
        if ak is None:
            return False
        ak.active = False
        self._save()
        return True

    def list_keys(self, owner: str | None = None) -> list[dict[str, Any]]:
        """List stored keys (hashes only, never raw keys)."""
        keys = self._keys.values()
        if owner:
            keys = [k for k in keys if k.owner == owner]
        return [
            {
                "key_hash_short": k.key_hash[:12] + "...",
                "tier": k.tier,
                "owner": k.owner,
                "active": k.active,
                "created_at": k.created_at,
            }
            for k in keys
        ]

    def get_tier(self, raw_key: str) -> str | None:
        """Return the tier for a raw key, or None if invalid."""
        result = self.validate(raw_key)
        return result.tier if result.authenticated else None

    @property
    def count(self) -> int:
        return len(self._keys)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _hash_key(raw_key: str) -> str:
    """SHA-256 hash of a raw API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _tier_from_key(raw_key: str) -> str | None:
    """Infer tier from the key prefix."""
    for prefix, tier in _PREFIX_MAP.items():
        if raw_key.startswith(prefix + "_"):
            return tier
    return None


# ---------------------------------------------------------------------------
# Authorization check
# ---------------------------------------------------------------------------


def check_tool_access(tier: str, tool_name: str) -> bool:
    """Return True if the given tier may call *tool_name*."""
    tier_info = TIERS.get(tier)
    if tier_info is None:
        return False
    if tier_info["tools"] == "all":
        return True
    # "basic" tier – restricted set
    return tool_name in BASIC_TOOLS


def get_daily_limit(tier: str) -> int:
    """Return the daily call limit for a tier."""
    return TIERS.get(tier, TIERS["free"])["daily_limit"]


# ---------------------------------------------------------------------------
# Singleton convenience
# ---------------------------------------------------------------------------

_store: KeyStore | None = None


def get_key_store(path: str | None = None) -> KeyStore:
    """Return the module-level KeyStore singleton."""
    global _store
    if _store is None:
        _store = KeyStore(path)
    return _store
