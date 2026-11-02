# -*- coding: utf-8 -*-
"""cqfm1._cache —— 本地 SQLite 缓存层。

三组表各司其职
--------------
* api_cache  : 接口原始响应（键 = 接口名 + 参数哈希），避免重复计费/触发限速
* api_state  : 每个接口的"上次调用时间"和"是否有权限"，实现跨进程限速
* instruments / prices / fetch_log : 物化的行情库，见 cqfm1.db

为什么用 SQLite 而不是 CSV / parquet：
* 零依赖、单文件、支持事务与 UPSERT（INSERT ... ON CONFLICT DO UPDATE）
* 主键天然去重，重复回填不会产生重复行（这正是 Lab 2 的核心思想）
* sqlite3 是 C 扩展，读写比逐条 pickle 快得多；WAL 模式下可并发读
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

from .config import logger, settings

__all__ = ["connect", "init_db", "cache_get", "cache_put", "state_get", "state_set",
           "make_key", "transaction", "purge"]

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS api_cache (
    key        TEXT PRIMARY KEY,
    payload    BLOB NOT NULL,
    n_rows     INTEGER,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS api_state (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS instruments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT UNIQUE NOT NULL,
    name            TEXT,
    exchange        TEXT,
    asset_class     TEXT,
    instrument_type TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    instrument_id INTEGER NOT NULL,
    date          TEXT NOT NULL,
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL,
    volume        REAL,
    amount        REAL,
    PRIMARY KEY (instrument_id, date),
    FOREIGN KEY (instrument_id) REFERENCES instruments(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

CREATE TABLE IF NOT EXISTS series (
    key        TEXT PRIMARY KEY,
    ts_code    TEXT,
    kind       TEXT,
    adjust     TEXT,
    start_date TEXT,
    end_date   TEXT,
    payload    BLOB,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS fetch_log (
    instrument_id INTEGER NOT NULL,
    start_date    TEXT NOT NULL,
    end_date      TEXT NOT NULL,
    fetched_at    REAL NOT NULL,
    n_rows        INTEGER,
    PRIMARY KEY (instrument_id, start_date, end_date)
);
"""

_conn: Optional[sqlite3.Connection] = None
_conn_path: Optional[str] = None
_LOCK = threading.RLock()      # 多线程回填时串行化写操作，避免 database is locked


def connect(path=None) -> sqlite3.Connection:
    """返回（并复用）连接；首次调用时建表。"""
    global _conn, _conn_path
    target = str(path or settings.db_path)
    with _LOCK:
        if _conn is not None and _conn_path == target:
            return _conn
        if _conn is not None:
            _conn.close()
        _conn = sqlite3.connect(target, timeout=30.0, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn_path = target
        _conn.executescript(_SCHEMA)
        _conn.commit()
    return _conn


def init_db(path=None) -> Path:
    """建库并返回路径（幂等）。"""
    connect(path)
    return Path(path or settings.db_path)


def make_key(endpoint: str, **kwargs: Any) -> str:
    """把接口名 + 参数压成稳定字符串键（参数顺序不影响结果）。"""
    payload = json.dumps(kwargs, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return endpoint + ":" + digest


def cache_get(key: str) -> Optional[pd.DataFrame]:
    """读缓存；命中返回 DataFrame，未命中返回 None。"""
    conn = connect()
    with _LOCK:
        row = conn.execute("SELECT payload FROM api_cache WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    try:
        return pickle.loads(row["payload"])
    except Exception:
        conn.execute("DELETE FROM api_cache WHERE key = ?", (key,))
        conn.commit()
        return None


def cache_put(key: str, df: pd.DataFrame) -> None:
    """写入/覆盖缓存（pickle 仅用于本机文件，不跨网络传输）。"""
    conn = connect()
    blob = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
    with _LOCK:
        conn.execute(
        "INSERT INTO api_cache(key, payload, n_rows, created_at) VALUES (?,?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload, "
        "n_rows=excluded.n_rows, created_at=excluded.created_at",
            (key, sqlite3.Binary(blob), int(len(df)), time.time()),
        )
        conn.commit()


def state_get(key: str, default: Any = None) -> Any:
    conn = connect()
    with _LOCK:
        row = conn.execute("SELECT value FROM api_state WHERE key = ?", (key,)).fetchone()
    return default if row is None else row["value"]


def state_set(key: str, value: Any) -> None:
    conn = connect()
    with _LOCK:
        conn.execute(
            "INSERT INTO api_state(key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, str(value), time.time()),
        )
        conn.commit()


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """事务上下文：批量写入比逐条 commit 快一到两个数量级。"""
    conn = connect()
    with _LOCK:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def series_get(key: str):
    """读取一条物化序列 -> (DataFrame, start, end)；不存在返回 None。"""
    conn = connect()
    with _LOCK:
        row = conn.execute(
            "SELECT payload, start_date, end_date FROM series WHERE key = ?", (key,)
        ).fetchone()
    if row is None or row["payload"] is None:
        return None
    try:
        return pickle.loads(row["payload"]), row["start_date"], row["end_date"]
    except Exception:
        return None


def series_put(key: str, df: pd.DataFrame, start: str, end: str,
               ts_code: str = "", kind: str = "", adjust: str = "") -> None:
    """写入/覆盖一条物化序列。"""
    conn = connect()
    blob = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
    with _LOCK:
        conn.execute(
            "INSERT INTO series(key, ts_code, kind, adjust, start_date, end_date, payload, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
            "start_date=excluded.start_date, end_date=excluded.end_date, "
            "payload=excluded.payload, updated_at=excluded.updated_at",
            (key, ts_code, kind, adjust, start, end, sqlite3.Binary(blob), time.time()),
        )
        conn.commit()


def purge() -> None:
    """清空缓存（保留表结构）。"""
    conn = connect()
    for table in ("api_cache", "api_state", "prices", "fetch_log", "instruments", "series"):
        conn.execute("DELETE FROM " + table)
    conn.commit()
    logger.info("缓存已清空")


def size_mb() -> float:
    p = settings.db_path
    return round(p.stat().st_size / 1e6, 2) if p.exists() else 0.0
