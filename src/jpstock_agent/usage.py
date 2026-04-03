"""Persistent usage tracking and analytics for jpstock-agent.

Records every tool call (key, tool, timestamp, latency) to a SQLite
database for analytics, billing reconciliation, and abuse detection.

Storage
-------
Default location: ``~/.jpstock/usage.db``
Override via ``JPSTOCK_USAGE_DB`` env var or constructor argument.

The schema is intentionally minimal – one ``calls`` table:

.. code-block:: sql

    CREATE TABLE calls (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        key_hash  TEXT    NOT NULL,
        tier      TEXT    NOT NULL,
        tool      TEXT    NOT NULL,
        ts        REAL    NOT NULL,   -- Unix timestamp
        latency   REAL    DEFAULT 0,  -- seconds
        status    TEXT    DEFAULT 'ok',  -- 'ok' or 'error'
        date_str  TEXT    NOT NULL    -- YYYY-MM-DD (for fast daily GROUP BY)
    );
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DIR = os.path.expanduser("~/.jpstock")
_DEFAULT_DB = os.path.join(_DEFAULT_DIR, "usage.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS calls (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash  TEXT    NOT NULL,
    tier      TEXT    NOT NULL,
    tool      TEXT    NOT NULL,
    ts        REAL    NOT NULL,
    latency   REAL    DEFAULT 0,
    status    TEXT    DEFAULT 'ok',
    date_str  TEXT    NOT NULL
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_calls_key_date ON calls (key_hash, date_str);",
    "CREATE INDEX IF NOT EXISTS idx_calls_tool_date ON calls (tool, date_str);",
    "CREATE INDEX IF NOT EXISTS idx_calls_date ON calls (date_str);",
]


# ---------------------------------------------------------------------------
# UsageTracker
# ---------------------------------------------------------------------------


class UsageTracker:
    """SQLite-backed usage recorder and analytics engine.

    Thread-safe: each thread gets its own connection via thread-local
    storage, and writes are serialised by SQLite's internal locking.
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or _DEFAULT_DB
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Init schema on the calling thread
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.execute(_CREATE_TABLE)
        for idx in _CREATE_INDEXES:
            conn.execute(idx)
        conn.commit()

    # -- recording --

    def record(
        self,
        key_hash: str,
        tier: str,
        tool: str,
        latency: float = 0.0,
        status: str = "ok",
    ) -> None:
        """Record a single tool call."""
        now = time.time()
        date_str = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO calls (key_hash, tier, tool, ts, latency, status, date_str) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key_hash, tier, tool, now, latency, status, date_str),
        )
        conn.commit()

    # -- analytics --

    def daily_summary(self, date_str: str | None = None) -> dict[str, Any]:
        """Aggregate stats for a single day.

        Returns total calls, unique keys, calls by tier, top tools.
        """
        if date_str is None:
            date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        conn = self._get_conn()

        total = conn.execute(
            "SELECT COUNT(*) FROM calls WHERE date_str = ?", (date_str,)
        ).fetchone()[0]

        unique_keys = conn.execute(
            "SELECT COUNT(DISTINCT key_hash) FROM calls WHERE date_str = ?",
            (date_str,),
        ).fetchone()[0]

        by_tier = dict(
            conn.execute(
                "SELECT tier, COUNT(*) FROM calls WHERE date_str = ? GROUP BY tier",
                (date_str,),
            ).fetchall()
        )

        top_tools = conn.execute(
            "SELECT tool, COUNT(*) as cnt FROM calls WHERE date_str = ? "
            "GROUP BY tool ORDER BY cnt DESC LIMIT 10",
            (date_str,),
        ).fetchall()

        errors = conn.execute(
            "SELECT COUNT(*) FROM calls WHERE date_str = ? AND status = 'error'",
            (date_str,),
        ).fetchone()[0]

        avg_latency = conn.execute(
            "SELECT AVG(latency) FROM calls WHERE date_str = ? AND latency > 0",
            (date_str,),
        ).fetchone()[0]

        return {
            "date": date_str,
            "total_calls": total,
            "unique_keys": unique_keys,
            "by_tier": by_tier,
            "top_tools": [{"tool": t, "count": c} for t, c in top_tools],
            "errors": errors,
            "error_rate_pct": round(errors / total * 100, 1) if total else 0,
            "avg_latency_ms": round((avg_latency or 0) * 1000, 1),
        }

    def key_usage(self, key_hash: str, days: int = 7) -> dict[str, Any]:
        """Usage breakdown for a specific API key over *days* days."""
        conn = self._get_conn()
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%d"
        )

        daily = conn.execute(
            "SELECT date_str, COUNT(*) as cnt FROM calls "
            "WHERE key_hash = ? AND date_str >= ? GROUP BY date_str ORDER BY date_str",
            (key_hash, cutoff),
        ).fetchall()

        total = sum(row[1] for row in daily)

        top_tools = conn.execute(
            "SELECT tool, COUNT(*) as cnt FROM calls "
            "WHERE key_hash = ? AND date_str >= ? GROUP BY tool ORDER BY cnt DESC LIMIT 10",
            (key_hash, cutoff),
        ).fetchall()

        errors = conn.execute(
            "SELECT COUNT(*) FROM calls "
            "WHERE key_hash = ? AND date_str >= ? AND status = 'error'",
            (key_hash, cutoff),
        ).fetchone()[0]

        return {
            "key_hash_short": key_hash[:12] + "..." if len(key_hash) > 12 else key_hash,
            "period_days": days,
            "total_calls": total,
            "daily": [{"date": d, "calls": c} for d, c in daily],
            "top_tools": [{"tool": t, "count": c} for t, c in top_tools],
            "errors": errors,
        }

    def tool_stats(self, days: int = 7) -> list[dict[str, Any]]:
        """Per-tool usage stats over *days* days."""
        conn = self._get_conn()
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%d"
        )

        rows = conn.execute(
            "SELECT tool, COUNT(*) as cnt, "
            "       AVG(latency) as avg_lat, "
            "       SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errs "
            "FROM calls WHERE date_str >= ? GROUP BY tool ORDER BY cnt DESC",
            (cutoff,),
        ).fetchall()

        return [
            {
                "tool": r[0],
                "calls": r[1],
                "avg_latency_ms": round((r[2] or 0) * 1000, 1),
                "errors": r[3],
                "error_rate_pct": round(r[3] / r[1] * 100, 1) if r[1] else 0,
            }
            for r in rows
        ]

    def revenue_estimate(self, days: int = 30) -> dict[str, Any]:
        """Estimate revenue based on tier usage over *days* days.

        Uses tier pricing: free=$0, pro=$9/mo, enterprise=$19/mo.
        Counts unique active keys per tier.
        """
        conn = self._get_conn()
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%d"
        )
        prices = {"free": 0, "pro": 9, "enterprise": 19}

        rows = conn.execute(
            "SELECT tier, COUNT(DISTINCT key_hash) as keys "
            "FROM calls WHERE date_str >= ? GROUP BY tier",
            (cutoff,),
        ).fetchall()

        breakdown = []
        total_mrr = 0.0
        for tier, count in rows:
            price = prices.get(tier, 0)
            mrr = price * count
            total_mrr += mrr
            breakdown.append({
                "tier": tier,
                "active_keys": count,
                "price_per_key": price,
                "mrr": mrr,
            })

        return {
            "period_days": days,
            "breakdown": breakdown,
            "total_mrr": total_mrr,
            "estimated_arr": round(total_mrr * 12, 2),
        }

    def cleanup(self, keep_days: int = 90) -> int:
        """Delete records older than *keep_days*. Returns rows deleted."""
        conn = self._get_conn()
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=keep_days)).strftime(
            "%Y-%m-%d"
        )
        cur = conn.execute(
            "DELETE FROM calls WHERE date_str < ?", (cutoff,)
        )
        conn.commit()
        return cur.rowcount

    def close(self) -> None:
        """Close the thread-local connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_tracker: UsageTracker | None = None


def get_usage_tracker(db_path: str | None = None) -> UsageTracker:
    """Return the module-level UsageTracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker(db_path)
    return _tracker
