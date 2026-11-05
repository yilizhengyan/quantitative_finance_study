# -*- coding: utf-8 -*-
"""cqfm1.charts —— 给 pandas 对象加上 .iplot()（替代 quantmod.charts）。

原理：pandas 允许第三方库注册 accessor，注册之后任何 DataFrame / Series
都可以写成 df.iplot(kind="candlestick")，用法和自己写的方法一模一样。

支持的 kind
-----------
line / scatter / ohlc / candlestick / overlay / subplots / normalized /
histogram / box / bar / pie / heatmap / treemap / surface

实现要点
--------
* 全部构建 plotly 的 Figure 对象并返回，Jupyter 里当表达式直接显示
* 不修改传进来的 DataFrame（无副作用），所以可以反复画
* 每个 kind 只在需要时才导入 plotly，冷启动更快
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["plot", "register"]

_TEMPLATE = "plotly_white"


def _lookup(df: pd.DataFrame, name: str) -> str:
    """列名大小写不敏感查找，返回真实列名。"""
    for c in df.columns:
        if str(c).lower() == str(name).lower():
            return c
    raise KeyError("找不到列 " + str(name) + "；现有列：" + ", ".join(map(str, df.columns)))


def _ohlc_frame(df: pd.DataFrame) -> pd.DataFrame:
    """把 DataFrame 归一成 Open/High/Low/Close 四列。"""
    need = ["open", "high", "low", "close"]
    cols = []
    for n in need:
        cols.append(_lookup(df, n))
    out = df[cols].copy()
    out.columns = ["Open", "High", "Low", "Close"]
    return out


def _as_x(obj) -> Optional[pd.Index]:
    return obj.index if isinstance(obj, pd.DataFrame) else obj.index


def _series_list(obj, showlegend: bool = True):
    """DataFrame -> 每列一条；Series -> 一条。"""
    if isinstance(obj, pd.Series):
        return [obj]
    return [obj[c] for c in obj.columns]


def _layout(fig, title=None, height=420, **kw):
    fig.update_layout(template=_TEMPLATE, height=kw.pop("height", height),
                      title=title, margin=dict(l=40, r=20, t=50, b=40),
                      **{k: v for k, v in kw.items() if k in ("showlegend",)})
    return fig


# ---------------------------------------------------------------------------
# 各个图表类型
# ---------------------------------------------------------------------------
def _line(obj, title=None, height=420, showlegend=True, y=None, **kw):
    import plotly.graph_objects as go
    if y is not None:
        if isinstance(obj, pd.Series):
            raise TypeError("y= 只在 DataFrame 上有意义")
        obj = obj[[_lookup(obj, y)]]
    fig = go.Figure()
    for s in _series_list(obj):
        fig.add_trace(go.Scatter(x=s.index, y=s.to_numpy(), name=str(s.name), mode="lines"))
    _layout(fig, title, height, showlegend=showlegend)
    return fig


def _scatter(df, x, y, title=None, height=420, showlegend=True, **kw):
    import plotly.graph_objects as go
    cx, cy = _lookup(df, x), _lookup(df, y)
    fig = go.Figure(go.Scatter(x=df[cx].to_numpy(), y=df[cy].to_numpy(), mode="markers",
                               name=str(cx) + " vs " + str(cy),
                               marker=dict(size=5, opacity=0.7)))
    _layout(fig, title, height, showlegend=showlegend)
    fig.update_xaxes(title_text=str(x))
    fig.update_yaxes(title_text=str(y))
    return fig


def _candle(obj, kind="candlestick", title=None, height=520, showlegend=True, **kw):
    import plotly.graph_objects as go
    ohlc = _ohlc_frame(obj)
    if kind == "ohlc":
        trace = go.Ohlc(x=ohlc.index, open=ohlc["Open"], high=ohlc["High"],
                        low=ohlc["Low"], close=ohlc["Close"], name="OHLC")
    else:
        trace = go.Candlestick(x=ohlc.index, open=ohlc["Open"], high=ohlc["High"],
                               low=ohlc["Low"], close=ohlc["Close"], name="Candle")
    fig = go.Figure(trace)
    _layout(fig, title, height, showlegend=showlegend)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


def _overlay(df, secondary_y=None, title=None, height=420, showlegend=True, **kw):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if df.shape[1] < 2:
        raise ValueError("overlay 至少需要两列")
    left = [c for c in df.columns if c != secondary_y][:1]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for c in left:
        fig.add_trace(go.Scatter(x=df.index, y=df[c].to_numpy(), name=str(c), mode="lines"), secondary_y=False)
    if secondary_y is not None:
        cy = _lookup(df, secondary_y)
        fig.add_trace(go.Scatter(x=df.index, y=df[cy].to_numpy(), name=str(cy), mode="lines"),
                      secondary_y=True)
    _layout(fig, title, height, showlegend=showlegend)
    return fig


def _subplots(df, title=None, height=180, showlegend=True, **kw):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    if df.shape[1] == 0:
        raise ValueError("没有可画的列")
    n = df.shape[1]
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[str(c) for c in df.columns],
                        vertical_spacing=min(0.25, 0.06 / max(n, 1) + 0.01))
    for i, c in enumerate(df.columns, start=1):
        fig.add_trace(go.Scatter(x=df.index, y=df[c].to_numpy(), name=str(c), mode="lines"),
                      row=i, col=1)
    _layout(fig, title, max(height * n, 300), showlegend=showlegend)
    return fig


def _normalized(df, title=None, height=420, showlegend=True, **kw):
    import plotly.graph_objects as go
    if isinstance(df, pd.Series):
        norm = df / df.dropna().iloc[0] * 100.0
        return _line(norm.rename(df.name), title, height, showlegend)
    norm = df / df.apply(lambda s: s.dropna().iloc[0]) * 100.0
    fig = go.Figure()
    for c in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[c].to_numpy(), name=str(c), mode="lines"))
    _layout(fig, title, height, showlegend=showlegend)
    fig.update_yaxes(title_text="指数化（起点=100）")
    return fig


def _histogram(df, overlap=True, title=None, height=420, nbins=50, showlegend=True, **kw):
    import plotly.graph_objects as go
    fig = go.Figure()
    for s in _series_list(df):
        fig.add_trace(go.Histogram(x=s.dropna().to_numpy(), name=str(s.name), nbinsx=nbins,
                                   opacity=0.6 if overlap else 1.0))
    _layout(fig, title, height, showlegend=showlegend)
    fig.update_layout(barmode="overlay" if overlap else "group")
    return fig


def _box(df, title=None, height=420, showlegend=True, **kw):
    import plotly.graph_objects as go
    fig = go.Figure()
    for s in _series_list(df):
        fig.add_trace(go.Box(y=s.dropna().to_numpy(), name=str(s.name), boxpoints="outliers"))
    _layout(fig, title, height, showlegend=showlegend)
    return fig


def _bar(df, title=None, height=420, showlegend=True, **kw):
    import plotly.graph_objects as go
    fig = go.Figure()
    x = df.index.astype(str) if isinstance(df.index, pd.DatetimeIndex) else df.index
    for s in _series_list(df):
        fig.add_trace(go.Bar(x=list(x), y=s.to_numpy(), name=str(s.name)))
    _layout(fig, title, height, showlegend=showlegend)
    return fig


def _pie(df, title=None, height=420, top_n=None, threshold=None, showlegend=True, **kw):
    import plotly.graph_objects as go
    if isinstance(df, pd.Series):
        s = df.dropna()
        labels, values = s.index.astype(str), s.to_numpy()
    else:
        col = df.columns[0]
        s = df[col].dropna()
        labels, values = s.index.astype(str), s.to_numpy()
    labels, values = _positive_only(labels, values)
    if top_n:
        order = np.argsort(np.abs(values))[::-1][:int(top_n)]
        labels, values = np.asarray(labels)[order], values[order]
    if threshold:
        keep = np.abs(values) >= float(threshold) * np.abs(values).sum()
        labels, values = np.asarray(labels)[keep], values[keep]
    fig = go.Figure(go.Pie(labels=list(labels), values=values, hole=0.35))
    _layout(fig, title, height, showlegend=showlegend)
    return fig


def _heatmap(df, title=None, height=520, **kw):
    import plotly.graph_objects as go
    z = df.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in df.columns], y=[str(i) for i in df.index],
                               colorscale="RdBu", zmid=0.0,
                               text=np.round(z, 3), texttemplate="%{text}"))
    _layout(fig, title, height)
    return fig


def _treemap(df, labels=None, parents=None, values=None, title=None, height=520, **kw):
    import plotly.graph_objects as go
    if isinstance(df, pd.Series):
        s = df.dropna()
        labs = list(labels) if labels is not None else [str(i) for i in s.index]
        vals = s.to_numpy()
    else:
        col = values if values is not None else df.columns[0]
        s = df[col].dropna()
        labs = list(labels) if labels is not None else [str(i) for i in s.index]
        vals = s.to_numpy()
    labs, vals = _positive_only(np.asarray(labs, dtype=object), np.asarray(vals, dtype=float))
    labs = [str(x) for x in labs]
    pars = list(parents) if parents is not None and len(parents) == len(labs) else [""] * len(labs)
    fig = go.Figure(go.Treemap(labels=labs, parents=pars, values=vals))
    _layout(fig, title, height)
    return fig


def _surface(df, title=None, height=620, xaxis_title=None, yaxis_title=None, zaxis_title=None, **kw):
    import plotly.graph_objects as go
    z = df.to_numpy(dtype=float)
    x = [float(c) if _is_num(c) else str(c) for c in df.columns]
    y = [float(i) if _is_num(i) else str(i) for i in df.index]
    fig = go.Figure(go.Surface(z=z, x=x, y=y, colorscale="Viridis"))
    _layout(fig, title, height)
    fig.update_layout(scene=dict(xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                                 zaxis_title=zaxis_title))
    return fig


def _positive_only(labels, values):
    """饼图/树图不能画负数：只保留正值；全负时退回绝对值。"""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=object)
    mask = np.isfinite(values) & (values > 0)
    if mask.any():
        return labels[mask], values[mask]
    return labels, np.abs(np.nan_to_num(values))


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


_DISPATCH = {
    "line": _line, "scatter": _scatter, "ohlc": _candle, "candlestick": _candle,
    "overlay": _overlay, "subplots": _subplots, "normalized": _normalized,
    "histogram": _histogram, "hist": _histogram, "box": _box, "bar": _bar,
    "pie": _pie, "heatmap": _heatmap, "treemap": _treemap, "surface": _surface,
}


def plot(obj, kind: str = "line", **kwargs):
    """统一的绘图入口：plot(df, kind='candlestick')。"""
    fn = _DISPATCH.get(str(kind).lower())
    if fn is None:
        raise ValueError("不支持的 kind=" + str(kind) + "；可选：" + ", ".join(sorted(set(_DISPATCH))))
    if fn is _candle and not isinstance(obj, pd.DataFrame):
        raise TypeError("ohlc / candlestick 需要 DataFrame（含 Open/High/Low/Close）")
    if isinstance(obj, pd.Series) and fn in (_scatter, _overlay, _heatmap):
        raise TypeError(str(kind) + " 需要 DataFrame")
    return fn(obj, kind=kind, **kwargs) if fn is _candle else fn(obj, **kwargs)


# ---------------------------------------------------------------------------
# 注册 accessor：注册后 df.iplot(...) 即可用
# ---------------------------------------------------------------------------
@pd.api.extensions.register_dataframe_accessor("iplot")
class DataframeIplot:
    """>>> df.iplot(kind="candlestick", title="贵州茅台")"""

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._obj = pandas_obj

    def __call__(self, kind: str = "line", **kwargs):
        return plot(self._obj, kind=kind, **kwargs)

    def __repr__(self) -> str:
        return "iplot(kinds=" + ", ".join(sorted(set(_DISPATCH))) + ")"


@pd.api.extensions.register_series_accessor("iplot")
class SeriesIplot:
    """>>> s.iplot(kind="line")"""

    def __init__(self, pandas_obj: pd.Series) -> None:
        self._obj = pandas_obj

    def __call__(self, kind: str = "line", **kwargs):
        return plot(self._obj, kind=kind, **kwargs)


def register() -> None:
    """显式调用（import cqfm1.charts 时已自动注册）。"""
    return None
