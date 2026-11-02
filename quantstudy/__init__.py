# -*- coding: utf-8 -*-
"""cqfm1 —— CQF Module 1 自建量化工具箱。

把原课程里依赖 quantmod / yfinance / Supabase 的接口，换成
tushare（A 股数据）+ SQLite（本地存储）+ plotly（交互图表）。

顶层 API 一览
-------------
数据      getData / get_index / getTicker / get_option_chain / get_yield_curve / get_news
时序      dailyReturn / monthlyReturn / volatility / beta / drawdown / sharpe
存储      QuantmodDB (register / load_history / get_prices / get_asset_prices)
期权      binomial_option / black_scholes / implied_vol / BinomialOptionPricing
绘图      df.iplot(kind="candlestick")  —— 通过 pandas accessor 注册

安装与使用
----------
把整个 cqfm1 文件夹放到 notebook 同级目录即可 import；也可 pip install -e .
"""

from __future__ import annotations

import importlib

__version__ = "1.0.0"
__author__ = "CQF Module 1 实验重构"

from .config import describe, set_token, settings

__all__ = [
    "__version__", "settings", "set_token", "describe",
    "getData", "get_index", "getTicker", "get_option_chain", "get_yield_curve", "get_news",
    "search_instruments", "dailyReturn", "periodReturn", "volatility", "beta",
    "QuantmodDB", "binomial_option", "black_scholes", "implied_vol",
    "OptionInputs", "BinomialOptionPricing", "OptionType", "ExerciseStyle",
    "charts", "db", "fast", "marketdata", "options", "timeseries", "datasource", "config",
]

_LAZY = {
    "getData": ("marketdata", "getData"),
    "get_index": ("marketdata", "get_index"),
    "getTicker": ("marketdata", "getTicker"),
    "get_option_chain": ("marketdata", "get_option_chain"),
    "get_yield_curve": ("marketdata", "get_yield_curve"),
    "get_news": ("marketdata", "get_news"),
    "search_instruments": ("marketdata", "search_instruments"),
    "dailyReturn": ("timeseries", "dailyReturn"),
    "periodReturn": ("timeseries", "periodReturn"),
    "monthlyReturn": ("timeseries", "monthlyReturn"),
    "volatility": ("timeseries", "volatility"),
    "beta": ("timeseries", "beta"),
    "QuantmodDB": ("db", "QuantmodDB"),
    "binomial_option": ("options", "binomial_option"),
    "black_scholes": ("options", "black_scholes"),
    "implied_vol": ("options", "implied_vol"),
    "OptionInputs": ("options", "OptionInputs"),
    "OptionType": ("options", "OptionType"),
    "ExerciseStyle": ("options", "ExerciseStyle"),
    "BinomialOptionPricing": ("options", "BinomialOptionPricing"),
    "MonteCarloOptionPricing": ("options", "MonteCarloOptionPricing"),
}

_SUBMODULES = {"charts", "db", "fast", "marketdata", "options", "timeseries", "datasource",
               "config", "_cache"}


def __getattr__(name: str):
    """惰性导入：用到谁才加载谁，import cqfm1 本身只要 30ms。"""
    if name in _SUBMODULES:
        return importlib.import_module("." + name, __name__)
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        module = importlib.import_module("." + mod_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value          # 缓存，下次直接命中
        return value
    raise AttributeError("module 'cqfm1' has no attribute '" + name + "'")


def __dir__():
    return sorted(set(list(globals()) + list(_LAZY) + list(_SUBMODULES)))
