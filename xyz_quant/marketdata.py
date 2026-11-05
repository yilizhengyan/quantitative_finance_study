# -*- coding: utf-8 -*-
"""cqfm1.marketdata —— tushare 版的 getData / getTicker。

对应关系（Bloomberg -> 原课程库 -> 本模块）
------------------------------------------------
PX_LAST  <- getData('SPY')            -> getData('510300.SH')
DES      <- getTicker('TSLA').info    -> getTicker('600519.SH').info
FA       <- getTicker(...).financials -> get_fundamentals('600519.SH')
BETA     <- cov/var                   -> timeseries.beta()
YC       <- getData('^TNX')           -> get_yield_curve()
N        <- feedparser                -> get_news()
OV/IV    <- option_chain              -> get_option_chain('510050.SH')

设计要点
--------
* 行情走物化序列缓存（series 表），同一标的理论上只联网一次，之后任意区间都本地切片
* 列名沿用 yfinance 风格（Open/High/Low/Close/Volume），这样原课程代码几乎不用改
* 多标的返回 MultiIndex 列 (字段, 代码)，单标的返回普通列
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import _cache, datasource
from .config import logger, settings
from .datasource import TushareError, TusharePermissionError, TushareQuotaError

__all__ = [
    "normalize_symbol", "parse_period", "getData", "getTicker", "Ticker",
    "get_index", "get_yield_curve", "get_news", "get_option_chain",
    "get_fundamentals", "search_instruments", "trade_calendar", "FIELDS",
]

# tushare 列 -> 对外列名（yfinance 风格）
FIELDS = {
    "open": "Open", "high": "High", "low": "Low", "close": "Close",
    "vol": "Volume", "volume": "Volume", "amount": "Amount",
}

_SH_PREFIX = ("60", "68", "90", "50", "51", "58", "11", "13")
_SZ_PREFIX = ("00", "30", "20", "15", "16", "12", "39")
_BJ_PREFIX = ("43", "83", "87", "88", "92")


# ---------------------------------------------------------------------------
# 代码与时间窗口
# ---------------------------------------------------------------------------
def normalize_symbol(symbol: str) -> str:
    """把 600519 / sh600519 / 600519.sh 统一成 600519.SH。"""
    s = str(symbol).strip().upper().replace(" ", "")
    m = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ)", s)
    if m:
        return s
    m = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", s)
    if m:
        return m.group(2) + "." + m.group(1)
    m = re.fullmatch(r"(\d{6})", s)
    if not m:
        raise ValueError("无法识别的证券代码：" + str(symbol) + "（示例：600519、600519.SH）")
    code = s
    if code.startswith(_SH_PREFIX):
        return code + ".SH"
    if code.startswith(_BJ_PREFIX):
        return code + ".BJ"
    if code.startswith(_SZ_PREFIX):
        return code + ".SZ"
    return code + ".SZ"


def _to_compact(d: Union[str, dt.date, dt.datetime]) -> str:
    """YYYY-MM-DD -> YYYYMMDD（tushare 的日期格式）。"""
    if isinstance(d, (dt.date, dt.datetime)):
        return d.strftime("%Y%m%d")
    return str(d).replace("-", "").replace("/", "")[:8]


def _to_iso(d: Union[str, dt.date, dt.datetime]) -> str:
    s = str(d).replace("-", "")[:8]
    return s[:4] + "-" + s[4:6] + "-" + s[6:8]


def parse_period(period: str, end: Optional[dt.date] = None) -> tuple:
    """把 '2y' / '10d' / 'ytd' / 'max' 解释成 (start, end) 两个 date。

    支持：Nd / Nw / Nm / Ny / ytd / max / all；默认 1y。
    """
    today = end or dt.date.today()
    p = str(period or "1y").strip().lower()
    if p in ("max", "all"):
        return dt.date(2005, 1, 1), today
    if p == "ytd":
        return dt.date(today.year, 1, 1), today
    m = re.fullmatch(r"(\d+)([dwmy])", p)
    if not m:
        raise ValueError("无法解析 period=" + p + "（示例：10d / 6m / 2y / ytd / max）")
    n, unit = int(m.group(1)), m.group(2)
    days = {"d": 1, "w": 7, "m": 30.44, "y": 365.25}[unit] * n
    return today - dt.timedelta(days=int(days)), today


# ---------------------------------------------------------------------------
# 物化序列：一次联网，长期切片
# ---------------------------------------------------------------------------
def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)


def _fetch_raw(code: str, start: str, end: str, kind: str) -> pd.DataFrame:
    """kind='stock' 用 daily，kind='index' 用 index_daily。"""
    endpoint = "daily" if kind == "stock" else "index_daily"
    raw = None
    try:
        raw = datasource.query(endpoint, ts_code=code,
                               start_date=_to_compact(start), end_date=_to_compact(end))
    except Exception as exc:                      # 权限/限流/网络问题都降级
        logger.info("tushare %s 不可用（%s），改用 akshare", endpoint, str(exc)[:40])
    if raw is None or raw.empty:
        if not settings.fallback_akshare:
            raise TushareError("取不到 " + code + " 的数据")
        raw = _fetch_akshare(code, start, end, kind)
    if (raw is None or raw.empty) and settings.fallback_akshare and kind == "stock":
        raw = _fetch_etf(code, start, end)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume", "Amount"])
    df = raw.rename(columns=FIELDS).copy()
    df["Date"] = pd.to_datetime(df["trade_date"])
    keep = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume", "Amount") if c in df.columns]
    df = df[keep].sort_values("Date").drop_duplicates("Date")
    return df.reset_index(drop=True)


def _fetch_akshare_adjusted(code: str, start: str, end: str, adjust: str):
    """直接从新浪取复权价，替代 adj_factor 的计算。"""
    try:
        raw = datasource.ak_daily_sina(code, start, end, adjust=adjust)
    except Exception as exc:
        logger.warning("新浪复权价也不可用：%s", str(exc)[:60])
        return None
    if raw is None or raw.empty:
        return None
    out = raw.rename(columns=FIELDS).copy()
    out["Date"] = pd.to_datetime(out["trade_date"])
    keep = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume", "Amount") if c in out.columns]
    return out[keep].sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def _fetch_etf(code: str, start: str, end: str) -> pd.DataFrame:
    """tushare 的 daily 不覆盖 ETF，这里用新浪 ETF 源补上（只在 daily 返回空时触发）。"""
    if not code.endswith((".SH", ".SZ")):
        return pd.DataFrame()
    try:
        return datasource.ak_etf_daily(code, start, end)
    except Exception as exc:
        logger.info("ETF 兜底源不可用：" + str(exc)[:80])
        return pd.DataFrame()


def _fetch_akshare(code: str, start: str, end: str, kind: str) -> pd.DataFrame:
    """akshare 兜底：先试东财，再试新浪，都失败就返回空表。"""
    if kind == "index":
        chain = [lambda: datasource.ak_index_daily(code, start, end),
                 lambda: datasource.ak_index_daily_sina(code, start, end)]
    else:
        chain = [lambda: datasource.ak_daily(code, start, end),
                 lambda: datasource.ak_daily_sina(code, start, end)]
    for fn in chain:
        try:
            df = fn()
        except Exception as exc:
            logger.info("akshare 源失败：%s", str(exc)[:60])
            continue
        if df is not None and not df.empty:
            return df
    return pd.DataFrame()


def _series(code: str, start: str, end: str, kind: str = "stock",
            adjust: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
    """取一条已物化的行情序列；窗口不够时只补差集部分。"""
    key = kind + ":" + code          # series 里永远存原始价，复权在读取时再套
    cached = None if refresh else _cache.series_get(key)
    if cached is not None:
        df, cs, ce = cached
        if start >= cs and end <= ce:
            return _slice(df, start, end)
        gap = (dt.date.fromisoformat(end) - dt.date.fromisoformat(ce)).days
        if start >= cs and 0 < gap <= settings.stale_days:
            return _slice(df, start, ce)          # 尾部差几天，先用手上的
    else:
        df, cs, ce = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume", "Amount"]), start, start

    new_start = min(start, cs) if len(df) else start
    new_end = max(end, ce) if len(df) else end
    fresh = _fetch_raw(code, new_start, new_end, kind)
    merged = pd.concat([df, fresh], ignore_index=True).drop_duplicates("Date").sort_values("Date")
    merged = merged.reset_index(drop=True)
    _cache.series_put(key, merged, new_start, new_end, ts_code=code, kind=kind, adjust="raw")
    return _slice(merged, start, end)


def _apply_adjust(code: str, df: pd.DataFrame, adjust: Optional[str], start: str, end: str) -> pd.DataFrame:
    """用 adj_factor 做前/后复权；免费额度下每个标的只请求一次。"""
    if not adjust or df.empty:
        return df
    if adjust not in ("qfq", "hfq"):
        raise ValueError("adjust 只能是 None / 'qfq' / 'hfq'")
    key = "adj:" + code
    cached = _cache.series_get(key)
    if cached is None:
        try:
            raw = datasource.query("adj_factor", ts_code=code,
                                   start_date=_to_compact("2005-01-01"), end_date=_to_compact(dt.date.today()))
            adj = raw.rename(columns={"trade_date": "Date"})
            adj["Date"] = pd.to_datetime(adj["Date"])
            adj = adj[["Date", "adj_factor"]].sort_values("Date").reset_index(drop=True)
        except Exception:
            logger.info("adj_factor 不可用，改用新浪复权价：" + code)
            ak = _fetch_akshare_adjusted(code, start, end, adjust)
            return ak if ak is not None and not ak.empty else df
        _cache.series_put(key, adj, "2005-01-01", str(dt.date.today()), ts_code=code, kind="adj")
    else:
        adj = cached[0]

    if adj is None or adj.empty:
        logger.warning("adj_factor 为空，按未复权处理：" + code)
        return df
    merged = df.merge(adj, on="Date", how="left")
    merged["adj_factor"] = merged["adj_factor"].ffill().bfill()
    if merged["adj_factor"].isna().all():
        logger.warning("adj_factor 全为 NaN，按未复权处理：" + code)
        return df
    factor = merged["adj_factor"].astype(float)
    base = factor.iloc[-1] if adjust == "qfq" else factor.iloc[0]
    ratio = factor / base
    out = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out[col] = out[col] * ratio.to_numpy()
    return out


# ---------------------------------------------------------------------------
# getData：主入口
# ---------------------------------------------------------------------------
def _one(code: str, start: str, end: str, kind: str, adjust: Optional[str], refresh: bool) -> pd.DataFrame:
    df = _series(code, start, end, kind=kind, adjust=adjust, refresh=refresh)
    if adjust and kind == "stock":
        df = _apply_adjust(code, df, adjust, start, end)
    if df.empty:
        return df
    if "Date" in df.columns:
        return df.set_index("Date").rename_axis("Date")
    return df.rename_axis("Date")            # 已经是日期索引了


def getData(symbols: Union[str, Sequence[str]],
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            period: Optional[str] = None,
            interval: str = "1d",
            fields: Optional[Iterable[str]] = None,
            adjust: Optional[str] = None,
            kind: str = "stock",
            refresh: bool = False) -> pd.DataFrame:
    """获取行情数据（替代原课程的 quantmod.markets.getData / yfinance.download）。

    Parameters
    ----------
    symbols : str 或 list[str]
        证券代码，600519 / 600519.SH / sh600519 都可以。
    start_date, end_date : str, optional
        'YYYY-MM-DD'；与 period 二选一。
    period : str, optional
        '10d' / '6m' / '2y' / 'ytd' / 'max'，默认根据 start_date 推断。
    interval : str
        '1d' 日线；'1m'/'5m'/'15m'/'30m'/'60m' 走 akshare 分钟线。
    fields : list, optional
        只要部分字段，如 ['Close']。
    adjust : {'qfq','hfq',None}
        复权方式。做收益率、波动率分析时建议 'qfq'。
    kind : {'stock','index'}
        指数请用 'index'（或直接调用 get_index）。

    Returns
    -------
    DataFrame
        单标的：行索引为日期，列 Open/High/Low/Close/Volume/Amount。
        多标的：列为 MultiIndex(字段, 代码)，df['Close'] 即可取出收盘价矩阵。
    """
    single = isinstance(symbols, str)
    codes = [normalize_symbol(s) for s in ([symbols] if single else list(symbols))]
    if not codes:
        raise ValueError("symbols 不能为空")

    if start_date and end_date:
        start, end = _to_iso(start_date), _to_iso(end_date)
    else:
        start, end = (s.isoformat() for s in parse_period(period or "1y"))
        if start_date:
            start = _to_iso(start_date)
        if end_date:
            end = _to_iso(end_date)

    if interval and interval != "1d":
        return _intraday(codes, period or "5d", interval, single)

    frames = {}
    for code in codes:
        frames[code] = _one(code, start, end, kind, adjust, refresh)

    if single:
        out = frames[codes[0]]
    else:
        # 用 (字段, 代码) 两层列，保持字段顺序与传入代码顺序
        cols = ["Open", "High", "Low", "Close", "Volume", "Amount"]
        cols = [c for c in cols if any(c in f.columns for f in frames.values())]
        out = pd.concat(
            {field: pd.DataFrame({code: frames[code][field] for code in codes})
             for field in cols},
            axis=1,
        )
        out.columns.names = ["Field", "Symbol"]
    if fields:
        wanted = [str(f).capitalize() for f in fields]
        out = out[wanted] if single else out.loc[:, (wanted, slice(None))]
    return out


def _intraday(codes: List[str], period: str, interval: str, single: bool) -> pd.DataFrame:
    """分钟线：tushare 需更高积分，这里用 akshare 免费源。"""
    ak = datasource._import_akshare()
    per = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"}.get(interval)
    if per is None:
        raise ValueError("interval 只支持 1d / 1m / 5m / 15m / 30m / 60m")
    days = 5
    m = re.fullmatch(r"(\d+)d", str(period))
    if m:
        days = int(m.group(1))
    frames = {}
    for code in codes:
        df = ak.stock_zh_a_hist_min_em(symbol=code.split(".")[0], period=per, adjust="qfq")
        df = df.rename(columns={"时间": "Date", "开盘": "Open", "收盘": "Close",
                                "最高": "High", "最低": "Low", "成交量": "Volume", "成交额": "Amount"})
        df["Date"] = pd.to_datetime(df["Date"])
        frames[code] = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume", "Amount"]].tail(days * 48)
    if single:
        return frames[codes[0]]
    out = pd.concat(frames, axis=1)
    out.columns.names = ["Field", "Symbol"]
    return out


def get_index(symbol: str = "000300.SH", **kwargs) -> pd.DataFrame:
    """指数行情：get_index('000300.SH', period='2y')。"""
    kwargs["kind"] = "index"
    return getData(symbol, **kwargs)


# ---------------------------------------------------------------------------
# getTicker：DES / FA / 期权链
# ---------------------------------------------------------------------------
class OptionChain:
    """对齐 yfinance 的 option_chain 返回对象（.calls / .puts）。"""

    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame, expiry: str = "") -> None:
        self.calls = calls
        self.puts = puts
        self.expiry = expiry

    def __repr__(self) -> str:
        return "OptionChain(expiry=" + self.expiry + ", calls=" + str(len(self.calls)) + ", puts=" + str(len(self.puts)) + ")"


class Ticker:
    """对齐 yfinance.Ticker 的最小实现，属性都是惰性的。"""

    def __init__(self, symbol: str) -> None:
        self.symbol = normalize_symbol(symbol)
        self._cache: dict = {}

    def __repr__(self) -> str:
        return "Ticker(" + self.symbol + ")"

    # -- DES --
    @property
    def info(self) -> dict:
        if "info" in self._cache:
            return self._cache["info"]
        basic = _stock_basic(self.symbol)
        val = _valuation(self.symbol)
        row = {}
        if basic is not None and len(basic):
            row = basic.iloc[0].to_dict()
        info = {
            "symbol": self.symbol,
            "longName": row.get("name"),
            "sector": row.get("industry"),
            "exchange": row.get("exchange"),
            "list_date": row.get("list_date"),
            "marketCap": None if val.get("total_mv") is None else float(val["total_mv"]) * 1e4,
            "trailingPE": val.get("pe_ttm"),
            "priceToBook": val.get("pb"),
            "dividendYield": val.get("dv_ratio"),
            "turnoverRate": val.get("turnover_rate"),
            "returnOnEquity": None,   # 需 fina_indicator 权限；见 get_fundamentals 的降级方案
            "currency": "CNY",
        }
        self._cache["info"] = info
        return info

    def history(self, period: str = "1y", **kwargs) -> pd.DataFrame:
        return getData(self.symbol, period=period, **kwargs)

    # -- 公司行为 --
    @property
    def dividends(self) -> pd.Series:
        df = self._dividend_raw()
        if df is None or df.empty:
            return pd.Series(dtype="float64")
        d = df.dropna(subset=["ex_date"]).copy()
        d["ex_date"] = pd.to_datetime(d["ex_date"], format="%Y%m%d", errors="coerce")
        d = d.dropna(subset=["ex_date"]).sort_values("ex_date")
        s = pd.Series(d["cash_div"].astype(float).to_numpy(), index=pd.DatetimeIndex(d["ex_date"]), name="dividends")
        return s[~s.index.duplicated(keep="last")]

    @property
    def splits(self) -> pd.Series:
        df = self._dividend_raw()
        if df is None or df.empty:
            return pd.Series(dtype="float64")
        d = df.dropna(subset=["ex_date"]).copy()
        d["ex_date"] = pd.to_datetime(d["ex_date"], format="%Y%m%d", errors="coerce")
        d = d.dropna(subset=["ex_date"]).sort_values("ex_date")
        ratio = 1.0 + d["stk_div"].fillna(0).astype(float) / 10.0
        s = pd.Series(ratio.to_numpy(), index=pd.DatetimeIndex(d["ex_date"]), name="splits")
        return s[~s.index.duplicated(keep="last")]

    def _dividend_raw(self) -> Optional[pd.DataFrame]:
        try:
            df = datasource.query("dividend", ts_code=self.symbol)
            if df is not None and not df.empty and "ex_date" in df.columns:
                return df
        except Exception as exc:
            logger.info("tushare dividend 不可用（%s），改用 akshare", str(exc)[:40])
        try:
            df = datasource.ak_dividend(self.symbol)
            return df if df is not None and not df.empty else None
        except Exception as exc:
            logger.warning("分红数据取不到：%s", str(exc)[:60])
            return None

    # -- FA --
    @property
    def balancesheet(self) -> pd.DataFrame:
        return get_fundamentals(self.symbol)["balancesheet"]

    @property
    def financials(self) -> pd.DataFrame:
        return get_fundamentals(self.symbol)["income"]

    @property
    def cashflow(self) -> pd.DataFrame:
        return get_fundamentals(self.symbol)["cashflow"]

    @property
    def balance_sheet(self) -> pd.DataFrame:
        return self.balancesheet

    # -- 期权 --
    @property
    def options(self) -> List[str]:
        return _option_expiries(self.symbol)

    def option_chain(self, expiry: Optional[str] = None) -> OptionChain:
        return get_option_chain(self.symbol, expiry)

    # -- 便捷方法 --
    def beta(self, benchmark: str = "000300.SH", period: str = "2y") -> float:
        from . import timeseries
        return timeseries.beta(getData(self.symbol, period=period, adjust="qfq")["Close"],
                               getData(benchmark, period=period, kind="index")["Close"])


_STOCK_BASIC_COLS = "ts_code,name,industry,area,market,exchange,list_status,list_date,is_hs"


def _stock_basic(code: str) -> Optional[pd.DataFrame]:
    """stock_basic 按交易所批量拉一次并缓存（按代码单查会给限速添麻烦）。"""
    try:
        df = datasource.query("stock_basic", exchange="", list_status="L", fields=_STOCK_BASIC_COLS)
        if df is not None and not df.empty:
            hit = df[df["ts_code"] == code]
            if not hit.empty:
                return hit
    except Exception as exc:
        logger.info("stock_basic 不可用（%s），改用新浪快照", str(exc)[:40])
    try:
        names = datasource.ak_names()
        hit = names[names["ts_code"] == code]
        return hit if not hit.empty else None
    except Exception:
        return None


def _valuation(code: str) -> dict:
    """最新一个交易日的估值指标（daily_basic 按 trade_date 一次拿全市场，只调用一次）。"""
    last = _latest_trade_date()
    try:
        df = datasource.query("daily_basic", trade_date=last)
    except Exception as exc:
        logger.info("daily_basic 不可用（%s），估值字段留空", str(exc)[:50])
        return {}
    if df is None or df.empty:
        return {}
    row = df[df["ts_code"] == code]
    if row.empty:
        return {}
    r = row.iloc[0]
    out = {}
    for k in ("total_mv", "pe_ttm", "pb", "dv_ratio", "turnover_rate"):
        if k in row.columns:
            out[k] = None if pd.isna(r[k]) else float(r[k])
    return out


def _option_expiries(underlying: str) -> List[str]:
    try:
        df = datasource.query("opt_basic", exchange="SSE")
    except TushareError as exc:
        logger.warning("opt_basic 不可用：" + str(exc))
        return []
    if df is None or df.empty:
        return []
    col = "underlying_code" if "underlying_code" in df.columns else None
    d = df[df[col] == underlying] if col else df[df["name"].astype(str).str.contains(underlying.split(".")[0][2:4])]
    if d.empty:
        return []
    return sorted(d["maturity_date"].dropna().astype(str).unique().tolist())


def getTicker(symbol: str) -> "Ticker":
    """返回一个 Ticker 对象（对齐 yfinance.Ticker 的用法）。

    Examples
    --------
    >>> t = getTicker("600519.SH")
    >>> t.info["longName"], t.dividends.tail()
    """
    return Ticker(symbol)


def get_option_chain(underlying: str = "510050.SH", expiry: Optional[str] = None) -> OptionChain:
    """期权链：opt_basic 取合约要素，opt_daily 取最新行情，隐含波动率由本库自行反解。"""
    code = normalize_symbol(underlying) if not underlying.endswith((".SH", ".SZ")) else underlying
    try:
        basic = datasource.query("opt_basic", exchange="SSE")
    except TushareError as exc:
        raise TushareError("opt_basic 不可用：" + str(exc)) from exc
    if basic is None or basic.empty:
        return OptionChain(pd.DataFrame(), pd.DataFrame(), "")
    col = "underlying_code" if "underlying_code" in basic.columns else None
    if col:
        base = basic[basic[col] == code]
    else:
        base = basic[basic["name"].astype(str).str.startswith(_underlying_prefix(code))]
    if base.empty:
        raise ValueError("没有找到 " + code + " 的期权合约（目前只有 510050.SH / 510300.SH 等 ETF 期权）")

    expiries = sorted(base["maturity_date"].dropna().astype(str).unique().tolist())
    chosen = expiry or expiries[0]
    contracts = base[base["maturity_date"].astype(str) == str(chosen)]
    trade_date = _latest_trade_date()
    try:
        px = datasource.query("opt_daily", trade_date=trade_date)
    except TushareError as exc:
        logger.warning("opt_daily 不可用：" + str(exc))
        px = pd.DataFrame()
    df = contracts.merge(px, on="ts_code", how="left") if not px.empty else contracts
    calls = df[df["call_put"] == "C"].copy() if "call_put" in df.columns else df
    puts = df[df["call_put"] == "P"].copy() if "call_put" in df.columns else df.iloc[0:0]
    keep = [c for c in ("ts_code", "name", "exercise_price", "maturity_date", "close", "settle",
                        "vol", "amount", "oi") if c in df.columns]
    return OptionChain(calls[keep].reset_index(drop=True), puts[keep].reset_index(drop=True), str(chosen))


def _underlying_prefix(code: str) -> str:
    return {"510050.SH": "50ETF", "510300.SH": "300ETF", "588000.SH": "500ETF"}.get(code, "")


def trade_calendar(start: str = "2015-01-01", end: Optional[str] = None,
                   refresh: bool = False) -> pd.DataFrame:
    """交易日历（只联网一次，之后本地切片，避免天天打接口）。"""
    end = end or dt.date.today().isoformat()
    key = "cal:SSE"
    cached = None if refresh else _cache.series_get(key)
    if cached is not None:
        df, cs, ce = cached
        if end <= ce:
            return _slice(df, start, end).rename(columns={"Date": "cal_date"})
    try:
        df = datasource.query("trade_cal", exchange="SSE",
                              start_date=_to_compact(start), end_date=_to_compact(end))
    except Exception:
        df = datasource.ak_trade_calendar()
    df = df.copy()
    df["cal_date"] = pd.to_datetime(df["cal_date"])
    df = df[["cal_date", "is_open", "pretrade_date"]].sort_values("cal_date").reset_index(drop=True)
    _cache.series_put(key, df.rename(columns={"cal_date": "Date"}), start, end,
                      ts_code="SSE", kind="cal")
    return df


def _latest_trade_date() -> str:
    """最近一个交易日（YYYYMMDD）。"""
    try:
        cal = trade_calendar()
        return cal[cal["is_open"] == 1]["cal_date"].max().strftime("%Y%m%d")
    except Exception:
        return dt.date.today().strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# 宏观 / 利率 / 新闻 / 财报
# ---------------------------------------------------------------------------
def get_yield_curve(start_date: str = "2023-01-01", end_date: Optional[str] = None) -> pd.DataFrame:
    """收益率曲线。优先中债 yc_cb（需权限），否则用美国国债 us_tycr（对应 Bloomberg YC / ^TNX）。"""
    end_date = end_date or dt.date.today().isoformat()
    try:
        df = datasource.query("yc_cb", ts_code="1001.CB", curve_type="0", curve_term=10,
                              start_date=_to_compact(start_date), end_date=_to_compact(end_date))
        out = df[["trade_date", "yield"]].rename(columns={"trade_date": "Date", "yield": "CN10Y"})
        out["Date"] = pd.to_datetime(out["Date"])
        return out.set_index("Date").sort_index()
    except TushareError as exc:
        logger.info("yc_cb 不可用（" + str(exc)[:40] + "...），改用 us_tycr")
    try:
        df = datasource.query("us_tycr", start_date=_to_compact(start_date), end_date=_to_compact(end_date))
        out = df.rename(columns={"date": "Date"})
        out["Date"] = pd.to_datetime(out["Date"])
        return out.set_index("Date").sort_index()
    except TushareError as exc:
        logger.info("us_tycr 也不可用（%s），改用 akshare 中美国债", str(exc)[:40])
    ak = datasource.ak_bond_rate(start_date)
    ak = ak.rename(columns={"日期": "Date"})
    ak["Date"] = pd.to_datetime(ak["Date"])
    return ak.set_index("Date").sort_index()


def get_news(symbol: str = "600519.SH", limit: int = 20) -> pd.DataFrame:
    """新闻流（Bloomberg N）。tushare news 需权限，默认走东财个股新闻。"""
    try:
        df = datasource.query("news", src="sina", start_date=(dt.date.today() - dt.timedelta(days=3)).isoformat() + " 00:00:00",
                              end_date=dt.date.today().isoformat() + " 23:59:59")
        out = df.rename(columns={"datetime": "Date", "content": "title"})[["Date", "title"]]
        return out.head(limit)
    except TushareError:
        pass
    df = datasource.ak_news(symbol)
    rename = {"发布时间": "Date", "datetime": "Date", "新闻标题": "title", "标题": "title",
              "新闻内容": "content", "内容": "content", "文章来源": "source", "新闻链接": "url"}
    out = df.rename(columns=rename)
    keep = [c for c in ("Date", "title", "source", "url") if c in out.columns]
    out = out[keep].head(limit)
    if "content" in df.rename(columns=rename).columns:
        out["content"] = out.get("content", df.rename(columns=rename)["content"].head(limit))
    return out.reset_index(drop=True)


def get_fundamentals(symbol: str) -> dict:
    """三大报表。tushare 需 2000 积分；无权限时自动降级到 akshare 的财报摘要。"""
    code = normalize_symbol(symbol)
    out = {"balancesheet": pd.DataFrame(), "income": pd.DataFrame(), "cashflow": pd.DataFrame(),
           "source": "tushare"}
    try:
        for name in ("balancesheet", "income", "cashflow"):
            df = datasource.query(name, ts_code=code, start_date="20180101",
                                  end_date=dt.date.today().strftime("%Y%m%d"))
            out[name] = df
    except TushareError as exc:
        logger.info("三大报表无权限（" + str(exc)[:42] + "...），降级到 akshare")
        out["source"] = "akshare"
        try:
            ak = datasource._import_akshare()
            df = ak.stock_financial_abstract(symbol=code.split(".")[0])
            out["income"] = df
            out["balancesheet"] = df
            out["cashflow"] = df
            out["note"] = "akshare.stock_financial_abstract 返回的是三大报表摘要合并表"
        except Exception as exc2:
            out["note"] = "akshare 也不可用：" + str(exc2)[:120]
    return out


def get_macro(name: str = "gdp") -> pd.DataFrame:
    """宏观数据：'gdp' 或 'cpi'（tushare 优先，akshare 兜底）。"""
    try:
        df = datasource.query("cn_gdp" if name == "gdp" else "cn_cpi")
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return datasource.ak_macro(name)


def search_instruments(keyword: str, limit: int = 20) -> pd.DataFrame:
    """按名称/代码模糊查股票（替代 Bloomberg 的自动补全）。"""
    try:
        df = datasource.query("stock_basic", exchange="", list_status="L", fields=_STOCK_BASIC_COLS)
    except TushareError:
        return pd.DataFrame()
    k = str(keyword)
    hit = df[df["name"].astype(str).str.contains(k) | df["ts_code"].str.contains(k)]
    return hit.head(limit)
