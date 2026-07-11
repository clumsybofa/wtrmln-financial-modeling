"""SQLite persistence for connections and session event logs."""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "wtrmln.db"

_lock = threading.Lock()
_conn = None


def _db():
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS connections (
                id TEXT PRIMARY KEY,
                connector TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT,
                sync_config TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                type TEXT NOT NULL,
                data TEXT NOT NULL,
                ts REAL NOT NULL,
                PRIMARY KEY (session_id, seq)
            );
            """
        )
        _conn.commit()
    return _conn


def create_connection(connector: str) -> str:
    cid = uuid.uuid4().hex[:12]
    now = time.time()
    with _lock:
        _db().execute(
            "INSERT INTO connections (id, connector, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (cid, connector, "starting", now, now),
        )
        _db().commit()
    return cid


def update_connection(cid: str, status: str, summary: str | None = None, sync_config: dict | None = None):
    with _lock:
        _db().execute(
            "UPDATE connections SET status = ?, summary = COALESCE(?, summary), "
            "sync_config = COALESCE(?, sync_config), updated_at = ? WHERE id = ?",
            (status, summary, json.dumps(sync_config) if sync_config else None, time.time(), cid),
        )
        _db().commit()


def list_connections() -> list[dict]:
    with _lock:
        rows = _db().execute("SELECT * FROM connections ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def log_event(session_id: str, seq: int, type_: str, data: dict):
    with _lock:
        _db().execute(
            "INSERT OR REPLACE INTO events (session_id, seq, type, data, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, seq, type_, json.dumps(data), time.time()),
        )
        _db().commit()
