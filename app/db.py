"""SQLite-backed organization registry and query audit log.

SQLite was chosen over a heavier DB because this is a prototype — zero
infrastructure, file-based, and the write volume (one row per query) is
trivially low.  The schema is intentionally minimal but supports the two
things we actually need at runtime: "does this org exist?" and "what
queries were made?".
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

from app.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS organizations (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    display_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS query_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    org_id    TEXT    NOT NULL REFERENCES organizations(id),
    question  TEXT    NOT NULL,
    sources   TEXT    NOT NULL  -- JSON array of filenames
);
"""

# Pre-seeded demo orgs — matched to the sample data directories.
_SEED_ORGS = [
    ("org_001", "aurora_textiles", "Aurora Textiles Pvt Ltd"),
    ("org_002", "meridian_logistics", "Meridian Logistics Ltd"),
]


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    """Short-lived connection context — keeps the DB unlocked between requests."""
    conn = sqlite3.connect(str(settings.SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if missing and seed demo organizations."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        for org_id, name, display_name in _SEED_ORGS:
            conn.execute(
                "INSERT OR IGNORE INTO organizations (id, name, display_name) VALUES (?, ?, ?)",
                (org_id, name, display_name),
            )


def org_exists(org_id: str) -> bool:
    """Fast existence check — used by every query to fail early on unknown orgs."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM organizations WHERE id = ?", (org_id,)
        ).fetchone()
        return row is not None


def get_org(org_id: str) -> Optional[dict]:
    """Return org dict or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, display_name FROM organizations WHERE id = ?",
            (org_id,),
        ).fetchone()
        return dict(row) if row else None


def list_orgs() -> list[dict]:
    """Return all registered organizations."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, display_name FROM organizations ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]


def log_query(org_id: str, question: str, sources: list[str]) -> None:
    """Append a query to the audit log — fire-and-forget, never blocks the response."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO query_log (timestamp, org_id, question, sources) VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                org_id,
                question,
                json.dumps(sources),
            ),
        )
