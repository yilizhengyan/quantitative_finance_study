# -*- coding: utf-8 -*-
"""cqfm1.datasource —— 统一的数据获取层（tushare 为主，akshare 兜底）。

对外只有一个入口 query()，它把四件麻烦事一次解决：

1. 缓存命中    —— 同一个请求只真正联网一次
2. 跨进程限速  —— 免费 token 每个接口 1 次/分钟，本地记时间戳提前 sleep
3. 权限探测    —— 记录哪些接口没权限，避免每次重试都报同样的错
4. 错误翻译    —— 把 tushare 的中文提示转成可编程捕获的异常类型
"""

from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd

from . import _cache
from .config import logger, settings

__all__ = ["TushareError", "TusharePermissionError", "TushareQuotaError", "NoTokenError",
           "query", "pro", "import_tushare", "ak_daily", "ak_daily_sina", "ak_index_daily",
           "ak_index_daily_sina", "ak_news", "ak_etf_daily", "ak_names", "ak_trade_calendar",
           "ak_bond_rate", "ak_macro"]


class TushareError(RuntimeError):
    """数据源相关错误的基类。"""


class NoTokenError(TushareError):
    """没有配置 token。"""


class TusharePermissionError(TushareError):
    """当前账号积分不够，接口未开通。"""


class TushareQuotaError(TushareError):
    """调用频次超限。"""


def import_tushare():
    try:
        import tushare as ts
    except ImportError as exc:  # pragma: no cover
        raise TushareError("未安装 tushare，请先 pip install tushare") from exc
    return ts


_pro_client: Any = None


def pro():
    """惰性创建并复用 ProApi 客户端（不要每次调用都新建）。"""
    global _pro_client
    if _pro_client is not None:
        return _pro_client
    ts = import_tushare()
    if not settings.token:
        raise NoTokenError(
            "没有找到 tushare token。三种设置方式任选其一：\n"
            "  1) from cqfm1 import config; config.set_token('你的token', persist=True)\n"
            "  2) 设置环境变量 TUSHARE_TOKEN\n"
            "  3) 直接读本地缓存（settings.offline = True）"
        )
    _pro_client = ts.pro_api(settings.token)
    return _pro_client


def _quota_unit(msg: str) -> str:
    """从提示里抠出限流单位：'分钟' / '小时' / '天'。"""
    for unit in ("小时", "分钟", "天"):
        if unit in msg:
            return unit
    return "分钟"


def _classify(exc: Exception, endpoint: str) -> Exception:
    """把 tushare 的中文错误串翻译成有类型、可捕获的异常。"""
    msg = str(exc)
    if "频率超限" in msg:
        return TushareQuotaError("接口 " + endpoint + " 调用超限（" + _quota_unit(msg) + "）：" + msg)
    if "权限" in msg or "积分" in msg:
        return TusharePermissionError("接口 " + endpoint + " 无访问权限（需更高积分）：" + msg)
    return TushareError("接口 " + endpoint + " 调用失败：" + msg)


_TTL = {"分钟": 65.0, "小时": 3660.0, "天": 86460.0}


def blocked_state(endpoint: str):
    """返回该接口还剩多久解封（秒）；没有限流记录返回 None。"""
    raw = _cache.state_get("blocked:" + endpoint)
    if not raw:
        return None
    try:
        unit, ts = str(raw).split(":")
        ttl = _TTL.get(unit, 65.0)
        left = ttl - (time.time() - float(ts))
        return left if left > 0 else None
    except (ValueError, TypeError):
        return None


def _wait_rate(endpoint: str) -> float:
    """按接口维度的最小间隔等待，返回实际等待秒数。"""
    if settings.rate_limit <= 0:
        return 0.0
    last = _cache.state_get("last:" + endpoint)
    if last is None:
        return 0.0
    gap = time.time() - float(last)
    if gap >= settings.rate_limit:
        return 0.0
    wait = settings.rate_limit - gap
    if settings.verbose:
        print("[cqfm1] " + endpoint + " 限速等待 %.0fs ..." % wait)
    time.sleep(wait)
    return wait


def query(endpoint: str, refresh: bool = False, **kwargs: Any) -> pd.DataFrame:
    """调用 tushare 接口，带缓存与限速。

    Parameters
    ----------
    endpoint : str
        ProApi 上的方法名，如 daily、stock_basic、opt_daily。
    refresh : bool
        为 True 时忽略缓存强制联网。
    **kwargs
        透传给 tushare 的参数；值为 None 的会被丢弃。

    Returns
    -------
    pandas.DataFrame
        接口原始结果（列名清洗在 marketdata 里做）。
    """
    params = {k: v for k, v in kwargs.items() if v is not None}
    key = _cache.make_key(endpoint, **params)
    tag = endpoint + "(" + ",".join(k + "=" + str(v) for k, v in params.items()) + ")"

    if not refresh:
        hit = _cache.cache_get(key)
        if hit is not None:
            return hit.copy()

    if settings.offline:
        raise TushareError("offline 模式下本地没有 " + tag + " 的缓存。请关闭 settings.offline 或先联网回填。")

    if _cache.state_get("perm:" + endpoint) == "0":
        raise TusharePermissionError("接口 " + endpoint + " 已记录为无权限（需要更高积分）")
    left = blocked_state(endpoint)
    if left is not None:
        raise TushareQuotaError("接口 " + endpoint + " 限流中，%d 秒后解封，直接走降级" % left)

    client = pro()
    last_exc: Optional[Exception] = None
    for attempt in range(settings.max_retry + 1):
        _wait_rate(endpoint)
        try:
            df = getattr(client, endpoint)(**params)
        except Exception as exc:  # tushare 抛的是裸 Exception
            err = _classify(exc, endpoint)
            last_exc = err
            if isinstance(err, TusharePermissionError):
                _cache.state_set("perm:" + endpoint, "0")
                raise err
            if isinstance(err, TushareQuotaError):
                _cache.state_set("last:" + endpoint, time.time())
                unit = _quota_unit(str(err))
                _cache.state_set("blocked:" + endpoint, unit + ":" + str(time.time()))
                # 按小时/按天限流的不值得等，直接交给上层降级；按分钟的等一轮再试
                if unit == "分钟" and attempt < settings.max_retry:
                    if settings.verbose:
                        print("[cqfm1] " + endpoint + " 超限，%.0fs 后重试" % settings.rate_limit)
                    continue
                raise err
            if attempt < settings.max_retry:
                time.sleep(2 ** attempt)
                continue
            raise err
        else:
            _cache.state_set("last:" + endpoint, time.time())
            if df is None:
                df = pd.DataFrame()
            _cache.cache_put(key, df)
            return df.copy()
    raise last_exc or TushareError("未知错误")


# --------------------------------------------------------------------------
# akshare 兜底（免费、免 token，只有行情类接口）
# --------------------------------------------------------------------------
_AK_COLUMNS = {
    "日期": "trade_date", "开盘": "open", "收盘": "close", "最高": "high",
    "最低": "low", "成交量": "vol", "成交额": "amount", "换手率": "turnover_rate",
    "涨跌幅": "pct_chg",
}


def _import_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise TushareError("未安装 akshare，无法降级；请 pip install akshare") from exc
    return ak


def _ak_fetch(cache_key: str, fetcher) -> pd.DataFrame:
    """给 akshare 调用套上同一套缓存。"""
    hit = _cache.cache_get(cache_key)
    if hit is not None:
        return hit.copy()
    if settings.offline:
        raise TushareError("offline 模式下本地没有 akshare 缓存")
    df = fetcher()
    _cache.cache_put(cache_key, df)
    return df.copy()


def ak_daily(symbol6: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    """A 股日线（默认前复权），列名与 tushare.daily 对齐。"""
    ak = _import_akshare()
    s = start_date.replace("-", "")
    e = end_date.replace("-", "")
    code = symbol6.split(".")[0]

    def _fetch() -> pd.DataFrame:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s, end_date=e, adjust=adjust)
        df = df.rename(columns=_AK_COLUMNS)
        keep = [c for c in ("trade_date", "open", "high", "low", "close", "vol", "amount", "pct_chg") if c in df.columns]
        out = df[keep].copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        out.insert(0, "ts_code", symbol6)
        return out

    key = _cache.make_key("ak_daily", code=code, start=s, end=e, adjust=adjust)
    return _ak_fetch(key, _fetch)


def ak_index_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """指数日线，如 000300。"""
    ak = _import_akshare()
    s = start_date.replace("-", "")
    e = end_date.replace("-", "")
    code = symbol.split(".")[0]

    def _fetch() -> pd.DataFrame:
        df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=s, end_date=e)
        df = df.rename(columns=_AK_COLUMNS)
        out = df[["trade_date", "open", "high", "low", "close", "vol", "amount"]].copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        out.insert(0, "ts_code", symbol)
        return out

    key = _cache.make_key("ak_index_daily", code=code, start=s, end=e)
    return _ak_fetch(key, _fetch)


def ak_news(symbol6: str) -> pd.DataFrame:
    """个股新闻（东方财富）。"""
    ak = _import_akshare()
    code = symbol6.split(".")[0]

    def _fetch() -> pd.DataFrame:
        df = ak.stock_news_em(symbol=code)
        df.columns = [str(c).lower() for c in df.columns]
        return df

    key = _cache.make_key("ak_news", code=code)
    return _ak_fetch(key, _fetch)


def ak_etf_daily(symbol6: str, start_date: str = "2005-01-01", end_date: str = "2100-01-01") -> pd.DataFrame:
    """ETF 日线（新浪源，免费）。tushare 的 fund_daily 需要高积分，这里补位。"""
    ak = _import_akshare()
    prefix = "sh" if symbol6.endswith(".SH") else "sz"
    code = prefix + symbol6.split(".")[0]

    def _fetch() -> pd.DataFrame:
        df = ak.fund_etf_hist_sina(symbol=code)
        out = pd.DataFrame({
            "ts_code": symbol6,
            "trade_date": pd.to_datetime(df["date"]),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "vol": df["volume"].astype(float) / 100.0,     # 股 -> 手，对齐 tushare
            "amount": df["amount"].astype(float) / 1000.0,  # 元 -> 千元
        })
        return out

    key = _cache.make_key("ak_etf_daily", code=code)
    df = _ak_fetch(key, _fetch)
    mask = (df["trade_date"] >= pd.Timestamp(start_date)) & (df["trade_date"] <= pd.Timestamp(end_date))
    return df.loc[mask].reset_index(drop=True)


def _sina_symbol(symbol6: str) -> str:
    """600519.SH -> sh600519；000858.SZ -> sz000858。"""
    return ("sh" if symbol6.endswith(".SH") else "sz") + symbol6.split(".")[0]


def ak_daily_sina(symbol6: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    """新浪日线（含复权），作为东财接口和 tushare 的第二重兜底。"""
    ak = _import_akshare()
    code = _sina_symbol(symbol6)
    s, e = start_date.replace("-", ""), end_date.replace("-", "")

    def _fetch() -> pd.DataFrame:
        df = ak.stock_zh_a_daily(symbol=code, start_date=s, end_date=e, adjust=adjust)
        out = pd.DataFrame({
            "ts_code": symbol6,
            "trade_date": pd.to_datetime(df["date"]),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "vol": df["volume"].astype(float) / 100.0,
            "amount": df["amount"].astype(float) / 1000.0,
        })
        return out

    key = _cache.make_key("ak_daily_sina", code=code, start=s, end=e, adjust=adjust)
    return _ak_fetch(key, _fetch)


def ak_index_daily_sina(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """新浪指数日线，如 000300.SH。"""
    ak = _import_akshare()
    code = _sina_symbol(symbol)

    def _fetch() -> pd.DataFrame:
        df = ak.stock_zh_index_daily(symbol=code)
        out = pd.DataFrame({
            "ts_code": symbol,
            "trade_date": pd.to_datetime(df["date"]),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "vol": pd.to_numeric(df["volume"], errors="coerce").astype(float) / 100.0,
            "amount": float("nan"),
        })
        return out

    key = _cache.make_key("ak_index_daily_sina", code=code)
    df = _ak_fetch(key, _fetch)
    mask = (df["trade_date"] >= pd.Timestamp(start_date)) & (df["trade_date"] <= pd.Timestamp(end_date))
    return df.loc[mask].reset_index(drop=True)


def ak_names() -> pd.DataFrame:
    """全 A 股代码与简称（新浪快照，慢但稳，缓存一次即可）。"""
    ak = _import_akshare()

    def _fetch() -> pd.DataFrame:
        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()):   # 屏蔽 tqdm 进度条
            df = ak.stock_zh_a_spot()
        code = df["代码"].astype(str)
        pref = code.str[:2].str.upper()          # sh/sz/bj -> SH/SZ/BJ
        num = code.str[2:]                        # 600519
        names = df["名称"].astype(str).str.replace(" ", "", regex=False)   # 新浪返回"五 粮 液"
        return pd.DataFrame({"ts_code": num + "." + pref, "name": names})

    key = _cache.make_key("ak_names")
    return _ak_fetch(key, _fetch)


def ak_trade_calendar() -> pd.DataFrame:
    """交易日历（新浪），返回 cal_date / is_open。"""
    ak = _import_akshare()

    def _fetch() -> pd.DataFrame:
        df = ak.tool_trade_date_hist_sina()
        d = pd.to_datetime(df["trade_date"])
        return pd.DataFrame({"cal_date": d, "is_open": 1, "pretrade_date": pd.NaT})

    key = _cache.make_key("ak_trade_cal")
    return _ak_fetch(key, _fetch)


def ak_bond_rate(start_date: str = "2023-01-01") -> pd.DataFrame:
    """中美国债收益率曲线（乐咕），一次拿到 2/5/10/30 年期。"""
    ak = _import_akshare()

    def _fetch() -> pd.DataFrame:
        return ak.bond_zh_us_rate(start_date=start_date.replace("-", ""))

    key = _cache.make_key("ak_bond_rate", start=start_date)
    return _ak_fetch(key, _fetch)


def ak_macro(name: str) -> pd.DataFrame:
    """宏观：'gdp' 或 'cpi'（金十/新浪源，免 token）。"""
    ak = _import_akshare()
    fn = {"gdp": ak.macro_china_gdp_yearly, "cpi": ak.macro_china_cpi_yearly}[name]

    def _fetch() -> pd.DataFrame:
        d = fn()
        d["日期"] = pd.to_datetime(d["日期"])
        return d.sort_values("日期").reset_index(drop=True)

    key = _cache.make_key("ak_macro", name=name)
    return _ak_fetch(key, _fetch)


def ak_dividend(symbol6: str) -> pd.DataFrame:
    """分红送转（新浪），列名对齐 tushare.dividend。"""
    ak = _import_akshare()
    code = symbol6.split(".")[0]

    def _fetch() -> pd.DataFrame:
        df = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
        out = pd.DataFrame({
            "ex_date": pd.to_datetime(df["除权除息日"]).dt.strftime("%Y%m%d"),
            "cash_div": pd.to_numeric(df["派息"], errors="coerce").astype(float) / 10.0,
            "stk_div": (pd.to_numeric(df["送股"], errors="coerce").fillna(0)
                        + pd.to_numeric(df["转增"], errors="coerce").fillna(0)).astype(float) / 10.0,
            "div_proc": df["进度"],
        })
        return out.dropna(subset=["ex_date"]).sort_values("ex_date").reset_index(drop=True)

    key = _cache.make_key("ak_dividend", code=code)
    return _ak_fetch(key, _fetch)
