# -*- coding: utf-8 -*-
"""cqfm1 冒烟测试：不联网，只验证纯计算部分。

    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # 包的上一级目录

from cqfm1 import options, timeseries as ts


def test_symbol_normalize():
    from cqfm1.marketdata import normalize_symbol
    assert normalize_symbol("600519") == "600519.SH"
    assert normalize_symbol("000858") == "000858.SZ"
    assert normalize_symbol("sh600519") == "600519.SH"
    assert normalize_symbol("300750.sz") == "300750.SZ"


def test_period_parse():
    from cqfm1.marketdata import parse_period
    start, end = parse_period("2y")
    assert (end - start).days >= 729


def test_binomial_matches_bs():
    from cqfm1 import fast
    bs = options.black_scholes(100, 100, 1, 0.05, 0.2)
    tree = fast.binomial_numpy(100, 100, 0.05, 0.2, 1, 2000)
    assert abs(tree - bs) < 0.02


def test_binomial_implementations_agree():
    from cqfm1 import fast
    args = (100, 100, 0.05, 0.2, 1, 200)
    a = fast.binomial_py(*args)
    b = fast.binomial_numpy(*args)
    c = fast.binomial_european_cf(*args)
    assert abs(a - b) < 1e-6
    assert abs(a - c) < 1e-6


def test_implied_vol_roundtrip():
    price = options.black_scholes(100, 100, 1, 0.05, 0.25)
    iv = options.implied_vol(price, 100, 100, 1, 0.05)
    assert abs(iv - 0.25) < 1e-6


def test_american_put_ge_european():
    eu = options.binomial_option(100, 100, 0.05, 0.2, 1, 100, output=2,
                                 option_type="put", style="european")[0, 0]
    am = options.binomial_option(100, 100, 0.05, 0.2, 1, 100, output=2,
                                 option_type="put", style="american")[0, 0]
    assert am >= eu - 1e-9


def test_returns_and_vol():
    idx = pd.bdate_range("2024-01-01", periods=260)
    px = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 260))), index=idx)
    r = ts.dailyReturn(px)
    assert len(r) == 259          # 首日没有前值，第一行收益率为 NaN 被丢掉
    assert ts.volatility(r) > 0
    assert ts.maxDrawdown(px) <= 0


def test_all_return_shape():
    idx = pd.bdate_range("2022-01-01", periods=700)
    px = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 700))), index=idx)
    df = ts.allReturn(px)
    assert isinstance(df, pd.DataFrame)
    assert "Daily" in df.index.get_level_values(0)


def test_iplot_registered():
    import cqfm1.charts  # noqa: F401
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    fig = df.iplot(kind="line")
    assert fig is not None


def test_option_inputs_validation():
    with pytest.raises(ValueError):
        options.OptionInputs(spot=-1, strike=100, ttm=1, rate=0.05, volatility=0.2)
