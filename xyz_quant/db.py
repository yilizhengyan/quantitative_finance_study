# -*- coding: utf-8 -*-
"""cqfm1.db —— 本地版 QuantmodDB（SQLite 取代 Supabase / PostgreSQL）。

原课程用 Supabase 托管 Postgres，需要联网、注册、拿 API Key。
这里换成同一个设计思想的本地实现：

============================  ==========================================
原方案                        本实现
============================  ==========================================
Postgres 表 instruments       SQLite 表 instruments
Postgres 表 prices            SQLite 表 prices
PRIMARY KEY (instrument_id,date) 完全一致：主键即去重
INSERT ... ON CONFLICT        SQLite 3.24+ 原生 UPSERT，语法几乎一样
Supabase URL + Key            本地 .db 文件路径
============================  ==========================================

方法名保持一致，所以课程代码只需要把 import 换成本库。
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd

from . import _cache, marketdata
from .config import logger, settings

__all__ = ["QuantmodDB"]

_PRICE_COLS = ("open", "high", "low", "close", "volume", "amount")


class QuantmodDB:
    """一个极简的行情数据仓库。

    Examples
    --------
    >>> qm = QuantmodDB()
    >>> qm.register([{"symbol": "600519.SH", "name": "贵州茅台", "asset_class": "EQUITY"}])
    >>> qm.load_history(["600519.SH"], "2022-01-01", "2025-12-31")
    >>> qm.get_asset_prices().tail()
    """

    def __init__(self, path: Optional[Union[str, Path]] = None,
                 supabase_url: Optional[str] = None,
                 supabase_key: Optional[str] = None) -> None:
        # 保留原签名，方便直接替换；传了就提示一下
        if supabase_url or supabase_key:
            logger.warning("已忽略 supabase_url / supabase_key：本实现使用本地 SQLite 文件")
        self.path = Path(path) if path else settings.db_path
        _cache.init_db(self.path)
        self.conn = _cache.connect(self.path)

    def __repr__(self) -> str:
        return "QuantmodDB(path='" + str(self.path) + "')"

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------
    def register(self, instruments: Sequence[Dict]) -> int:
        """注册（或更新）标的元数据，等价于原课程的 qm.register(list_of_dict)。

        Parameters
        ----------
        instruments : list[dict]
            每个字典至少要有 symbol；可选 name / exchange / asset_class / instrument_type。

        Returns
        -------
        int
            受影响的标的数量。
        """
        rows = []
        now = dt.datetime.now().isoformat(timespec="seconds")
        for item in instruments:
            if isinstance(item, str):
                item = {"symbol": item}
            sym = marketdata.normalize_symbol(item["symbol"])
            rows.append((sym, item.get("name"), item.get("exchange"),
                         item.get("asset_class"), item.get("instrument_type"), now))
        with _cache.transaction() as conn:
            conn.executemany(
                "INSERT INTO instruments(symbol, name, exchange, asset_class, instrument_type, updated_at) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET "
                "name=excluded.name, exchange=excluded.exchange, "
                "asset_class=excluded.asset_class, instrument_type=excluded.instrument_type, "
                "updated_at=excluded.updated_at",
                rows,
            )
        logger.info("注册 %d 个标的", len(rows))
        return len(rows)

    def get_instrument_id(self, symbol: str) -> Dict:
        """按代码查标的（含内部 id）。"""
        sym = marketdata.normalize_symbol(symbol)
        row = self.conn.execute("SELECT * FROM instruments WHERE symbol = ?", (sym,)).fetchone()
        if row is None:
            return {}
        return {k: row[k] for k in row.keys()}

    def list_instruments(self, symbols: Optional[Iterable[str]] = None,
                         exchange: Optional[str] = None,
                         asset_class: Optional[str] = None) -> pd.DataFrame:
        """列出已注册标的，可按代码/交易所/资产类别过滤。"""
        sql = "SELECT id, symbol, name, exchange, asset_class, instrument_type FROM instruments WHERE 1=1"
        params: List = []
        if symbols:
            syms = [marketdata.normalize_symbol(s) for s in symbols]
            sql += " AND symbol IN (" + ",".join("?" * len(syms)) + ")"
            params.extend(syms)
        if exchange:
            sql += " AND exchange = ?"
            params.append(exchange)
        if asset_class:
            sql += " AND asset_class = ?"
            params.append(asset_class)
        return pd.read_sql_query(sql + " ORDER BY id", self.conn, params=params)

    def _id_map(self) -> Dict[str, int]:
        df = self.list_instruments()
        return dict(zip(df["symbol"], df["id"]))

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def load_history(self, symbols: Union[str, Sequence[str]],
                     start_date: str, end_date: str,
                     adjust: Optional[str] = "qfq",
                     refresh: bool = False) -> Dict[str, int]:
        """下载并入库历史行情（幂等：重复执行不会产生重复行）。

        Returns
        -------
        dict
            {代码: 写入行数}
        """
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        syms = [marketdata.normalize_symbol(s) for s in syms]
        self.register([{"symbol": s, "asset_class": "EQUITY", "instrument_type": "STOCK"} for s in syms])
        ids = self._id_map()
        out: Dict[str, int] = {}
        for sym in syms:
            df = marketdata.getData(sym, start_date=start_date, end_date=end_date,
                                    adjust=adjust, refresh=refresh)
            if df is None or df.empty:
                out[sym] = 0
                continue
            records = []
            for ts, row in df.iterrows():
                records.append((
                    ids[sym], ts.strftime("%Y-%m-%d"),
                    _f(row.get("Open")), _f(row.get("High")), _f(row.get("Low")),
                    _f(row.get("Close")), _f(row.get("Volume")), _f(row.get("Amount")),
                ))
            with _cache.transaction() as conn:
                conn.executemany(
                    "INSERT INTO prices(instrument_id, date, open, high, low, close, volume, amount) "
                    "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(instrument_id, date) DO UPDATE SET "
                    "open=excluded.open, high=excluded.high, low=excluded.low, "
                    "close=excluded.close, volume=excluded.volume, amount=excluded.amount",
                    records,
                )
                conn.execute(
                    "INSERT INTO fetch_log(instrument_id, start_date, end_date, fetched_at, n_rows) "
                    "VALUES (?,?,?,?,?) ON CONFLICT(instrument_id, start_date, end_date) DO UPDATE SET "
                    "fetched_at=excluded.fetched_at, n_rows=excluded.n_rows",
                    (ids[sym], start_date, end_date, dt.datetime.now().timestamp(), len(records)),
                )
            out[sym] = len(records)
            logger.info("入库 %s：%d 行", sym, len(records))
        return out

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def _query_prices(self, symbols=None, start_date=None, end_date=None, field=None) -> pd.DataFrame:
        sql = ("SELECT i.symbol AS symbol, p.date AS date, p.open, p.high, p.low, p.close, p.volume, p.amount "
               "FROM prices p JOIN instruments i ON i.id = p.instrument_id WHERE 1=1")
        params: List = []
        if symbols:
            syms = [marketdata.normalize_symbol(s)] if isinstance(symbols, str) else \
                   [marketdata.normalize_symbol(s) for s in symbols]
            sql += " AND i.symbol IN (" + ",".join("?" * len(syms)) + ")"
            params.extend(syms)
        if start_date:
            sql += " AND p.date >= ?"
            params.append(marketdata._to_iso(start_date))
        if end_date:
            sql += " AND p.date <= ?"
            params.append(marketdata._to_iso(end_date))
        sql += " ORDER BY i.symbol, p.date"
        df = pd.read_sql_query(sql, self.conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        if field:
            keep = ["symbol", "date", field.lower()]
            df = df[keep]
        return df

    def get_prices(self, symbols=None, start_date=None, end_date=None) -> pd.DataFrame:
        """长表格式的 OHLCV：每行一个 (标的, 交易日)。"""
        return self._query_prices(symbols, start_date, end_date)

    def get_asset_prices(self, field: str = "close", symbols=None,
                         start_date=None, end_date=None) -> pd.DataFrame:
        """宽表格式（透视）：行=日期，列=标的，值=指定字段。"""
        long = self._query_prices(symbols, start_date, end_date)
        if long.empty:
            return pd.DataFrame()
        col = field.lower()
        if col not in long.columns:
            raise ValueError("字段 " + field + " 不存在；可选 open/high/low/close/volume/amount")
        out = long.pivot(index="date", columns="symbol", values=col).sort_index()
        out.columns.name = None
        return out

    def get_latest_prices(self) -> pd.DataFrame:
        """每个标的的最新一条行情。"""
        sql = ("SELECT i.symbol, p.date, p.open, p.high, p.low, p.close, p.volume, p.amount "
               "FROM prices p JOIN instruments i ON i.id = p.instrument_id "
               "JOIN (SELECT instrument_id, MAX(date) AS d FROM prices GROUP BY instrument_id) m "
               "ON m.instrument_id = p.instrument_id AND m.d = p.date "
               "ORDER BY i.symbol")
        df = pd.read_sql_query(sql, self.conn)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------
    # 运维
    # ------------------------------------------------------------------
    def stats(self) -> pd.DataFrame:
        """各表的规模，用来讲数据治理很直观。"""
        rows = []
        for table in ("instruments", "prices", "series", "api_cache"):
            try:
                n = self.conn.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
            except sqlite3.Error:
                n = 0
            rows.append({"table": table, "rows": n})
        df = pd.DataFrame(rows)
        df.loc[len(df)] = {"table": "file_mb", "rows": _cache.size_mb()}
        return df

    def close(self) -> None:
        self.conn.close()


def _f(v) -> Optional[float]:
    """NaN -> None，避免 SQLite 里存的全是 nan。"""
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None
