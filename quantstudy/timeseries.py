# -*- coding: utf-8 -*-
"""cqfm1.timeseries —— 收益率与风险指标（替代 quantmod.timeseries）。

全部基于 pandas / numpy 的向量化实现：
* 收益率用 pct_change，分组用 resample，没有一行 Python 循环
* 年化因子按交易日 252 计算，可传参覆盖
* 所有函数都接受 Series 或 DataFrame，返回同样形状，方便直接画图
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "periodReturn", "dailyReturn", "weeklyReturn", "monthlyReturn",
    "quarterlyReturn", "annualReturn", "allReturn", "volatility",
    "cumReturn", "toPriceIndex", "drawdown", "maxDrawdown", "sharpe",
    "sortino", "beta", "betaPrices", "zscore", "var", "cvar", "downsideRisk",
]

SeriesLike = Union[pd.Series, pd.DataFrame]

# 频率别名 -> pandas offset 别名（pandas 3 用 ME/QE/YE）
_FREQ = {"D": None, "W": "W-FRI", "M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}
_SPAN = {"D": 252.0, "W": 52.0, "M": 12.0, "Q": 4.0, "A": 1.0, "Y": 1.0}


def _pct_change(px: SeriesLike) -> SeriesLike:
    """pct_change 的薄封装：显式传 fill_method=None，行为跨版本一致。"""
    return px.pct_change(fill_method=None)


def _last_per_period(price: SeriesLike, rule: Optional[str]) -> SeriesLike:
    """按周期取每期最后一个价格；rule=None 表示不重采样（日线）。"""
    if rule is None:
        return price
    if not isinstance(price.index, pd.DatetimeIndex):
        raise TypeError("价格序列的索引必须是 DatetimeIndex")
    return price.resample(rule).last()


def periodReturn(price: SeriesLike, period: str = "D") -> SeriesLike:
    """按周期计算收益率。

    Parameters
    ----------
    price : Series 或 DataFrame
        价格序列（收盘价即可）。
    period : {'D','W','M','Q','A'} 或 'all'
        D=日（默认），W=周，M=月，Q=季，A=年；'all' 返回五档汇总表。

    Examples
    --------
    >>> periodReturn(df["Close"], period="Q").tail()
    """
    key = str(period).upper()
    if key == "ALL":
        return allReturn(price)
    if key not in _FREQ:
        raise ValueError("period 只支持 D/W/M/Q/A/all，收到 " + str(period))
    return _pct_change(_last_per_period(price, _FREQ[key])).dropna(how="all")


def dailyReturn(price: SeriesLike) -> SeriesLike:
    """简单日收益率 r_t = P_t / P_(t-1) - 1。"""
    return periodReturn(price, "D")


def weeklyReturn(price: SeriesLike) -> SeriesLike:
    """周收益率（以每周最后一个交易日收盘价为基准）。"""
    return periodReturn(price, "W")


def monthlyReturn(price: SeriesLike) -> SeriesLike:
    """月收益率。"""
    return periodReturn(price, "M")


def quarterlyReturn(price: SeriesLike) -> SeriesLike:
    """季度收益率。"""
    return periodReturn(price, "Q")


def annualReturn(price: SeriesLike) -> SeriesLike:
    """年度收益率。"""
    return periodReturn(price, "A")


def allReturn(price: SeriesLike) -> pd.DataFrame:
    """日/周/月/季/年五种频率的统计摘要。

    Returns
    -------
    DataFrame
        索引 (Frequency, Symbol)，列 [mean, std, ann_ret, ann_vol]。
    """
    frames = []
    labels = [("Daily", "D"), ("Weekly", "W"), ("Monthly", "M"),
              ("Quarterly", "Q"), ("Annual", "A")]
    for label, freq in labels:
        r = periodReturn(price, freq)
        r = r.dropna(how="all") if isinstance(r, pd.DataFrame) else r.dropna()
        if len(r) == 0:
            continue
        span = _SPAN[freq]
        vals = {
            "mean": r.mean(),
            "std": r.std(ddof=1),
            "ann_ret": r.mean() * span,
            "ann_vol": r.std(ddof=1) * np.sqrt(span),
        }
        if isinstance(vals["mean"], pd.Series):
            stats = pd.DataFrame(vals)
            stats.index = pd.Index([str(i) for i in stats.index], name="Symbol")
        else:   # 单标的：所有值是标量，必须显式给索引
            name = getattr(r, "name", None) or "value"
            stats = pd.DataFrame({k: [float(v)] for k, v in vals.items()},
                                 index=pd.Index([str(name)], name="Symbol"))
        stats["Frequency"] = label
        frames.append(stats.set_index("Frequency", append=True))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).reorder_levels(["Frequency", "Symbol"])
    return out.sort_index()


def volatility(ret: SeriesLike, annualize: bool = True, periods: float = 252.0) -> SeriesLike:
    """已实现波动率 RV：收益率样本标准差，默认年化。"""
    vol = ret.std(ddof=1)
    return vol * np.sqrt(periods) if annualize else vol


def cumReturn(ret: SeriesLike, start: float = 1.0) -> SeriesLike:
    """把收益率序列累乘成净值曲线。"""
    return start * (1.0 + ret.fillna(0)).cumprod()


def toPriceIndex(price: SeriesLike, base: float = 100.0) -> SeriesLike:
    """价格归一化到 base（等价于图表里的 normalized）。"""
    first = price.dropna().iloc[0]
    return price.div(first).mul(base)


def drawdown(nav: SeriesLike) -> SeriesLike:
    """回撤序列（<=0）；nav 可以是净值曲线或价格。"""
    return nav / nav.cummax() - 1.0


def maxDrawdown(nav: SeriesLike) -> SeriesLike:
    """最大回撤（负数）。"""
    return drawdown(nav).min()


def sharpe(ret: SeriesLike, rf: float = 0.0, periods: float = 252.0) -> SeriesLike:
    """夏普比率；rf 为年化无风险利率。"""
    excess = ret - rf / periods
    return excess.mean() / excess.std(ddof=1) * np.sqrt(periods)


def sortino(ret: SeriesLike, rf: float = 0.0, periods: float = 252.0) -> SeriesLike:
    """索提诺比率：只惩罚下行波动。"""
    excess = ret - rf / periods
    downside = excess.where(excess < 0, 0.0)
    dd = np.sqrt((downside ** 2).mean())
    return excess.mean() / dd * np.sqrt(periods)


def beta(asset_returns: SeriesLike, market_returns: SeriesLike) -> float:
    """CAPM 贝塔 = cov(r_a, r_m) / var(r_m)。

    Parameters
    ----------
    asset_returns, market_returns : Series
        必须是**收益率**（不是价格）。价格请用 betaPrices()。
    """
    a = pd.Series(np.asarray(asset_returns, dtype=float)).reset_index(drop=True)
    b = pd.Series(np.asarray(market_returns, dtype=float)).reset_index(drop=True)
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / df.iloc[:, 1].var())


def betaPrices(asset_price: SeriesLike, market_price: SeriesLike) -> float:
    """价格版本的贝塔：先转日收益率再算，避免把趋势当成风险敞口。"""
    return beta(dailyReturn(asset_price), dailyReturn(market_price))


def zscore(x: SeriesLike, window: Optional[int] = None) -> SeriesLike:
    """Z 分数；给定 window 时为滚动 Z 分数。"""
    if window is None:
        return (x - x.mean()) / x.std(ddof=1)
    roll = x.rolling(window)
    return (x - roll.mean()) / roll.std(ddof=1)


def var(ret: SeriesLike, level: float = 0.05) -> float:
    """历史模拟法 VaR（返回正数表示潜在损失比例）。"""
    return float(-np.nanquantile(np.asarray(ret, dtype=float), level))


def cvar(ret: SeriesLike, level: float = 0.05) -> float:
    """条件 VaR / 期望损失 ES。"""
    arr = np.asarray(ret, dtype=float)
    arr = arr[~np.isnan(arr)]
    q = np.quantile(arr, level)
    return float(-arr[arr <= q].mean())


def downsideRisk(ret: SeriesLike, mar: float = 0.0) -> float:
    """下行标准差（MAR 默认 0）。"""
    arr = np.asarray(ret, dtype=float)
    arr = arr[~np.isnan(arr)]
    neg = np.minimum(arr - mar, 0.0)
    return float(np.sqrt((neg ** 2).mean()))
