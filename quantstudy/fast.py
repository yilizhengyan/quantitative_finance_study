# -*- coding: utf-8 -*-
"""cqfm1.fast —— 同一个算法，四种写法，四档速度。

这一模块专门用来回答"怎么写得更快、更底层"：

1. binomial_py     纯 Python，逐节点读写 numpy 数组（讲义原版写法）——最慢
2. binomial_numpy  把内层循环向量化，用 O(n) 滚动缓冲代替 O(n^2) 矩阵——快 50~200 倍
3. binomial_numba  Numba @njit 把整段编译成机器码，双循环也不怕——再快 5~20 倍
4. binomial_european_cf  直接套二项分布的闭式解，O(n) 一次算完——欧式期权最快

结论：**先向量化，再考虑 JIT/编译**。绝大多数金融计算（定价、回测、风控）
瓶颈都在"用 Python 循环遍历标量"，而不是在缺一个 C 扩展。
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Callable, Optional

import numpy as np

__all__ = [
    "HAS_NUMBA", "binomial_py", "binomial_numpy", "binomial_numba",
    "binomial_european_cf", "benchmark", "timer", "sweep_steps",
]

try:  # Numba 是可选依赖：装了就自动加速，没装就退回 NumPy 版
    from numba import njit, prange
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    njit = None
    prange = range
    HAS_NUMBA = False


@contextmanager
def timer(label: str = ""):
    """简易计时器：with timer('循环版'): ..."""
    t0 = time.perf_counter()
    box = {}
    yield box
    box["seconds"] = time.perf_counter() - t0
    if label:
        print(label + ": " + format(box["seconds"] * 1000, ".1f") + " ms")


def _params(rate, sigma, ttm, n):
    dt = ttm / n
    u = 1.0 + sigma * math.sqrt(dt)
    d = 1.0 - sigma * math.sqrt(dt)
    p = 0.5 + rate * math.sqrt(dt) / (2.0 * sigma)
    df = 1.0 / (1.0 + rate * dt)
    return u, d, p, df


# ---------------------------------------------------------------------------
# 写法 1：纯 Python（可读性最好，速度最差）
# ---------------------------------------------------------------------------
def binomial_py(spot, strike, rate, sigma, ttm, n, option_type="call",
                style="european") -> float:
    """用 Python 列表做滚动缓冲，内存 O(n)。

    即使不向量化，把 O(n^2) 的矩阵换成两个长度 n+1 的列表，
    内存占用和缓存命中率都会有数量级的改善。
    """
    u, d, p, df = _params(rate, sigma, ttm, n)
    is_call = str(option_type).lower() == "call"
    vals = [0.0] * (n + 1)
    prices = [0.0] * (n + 1)
    for i in range(n + 1):
        s = spot * u ** (n - i) * d ** i
        prices[i] = s
        vals[i] = max(s - strike, 0.0) if is_call else max(strike - s, 0.0)
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            cont = df * (p * vals[i] + (1.0 - p) * vals[i + 1])
            vals[i] = cont
            if style == "american":
                s = spot * u ** (j - i) * d ** i
                ex = max(s - strike, 0.0) if is_call else max(strike - s, 0.0)
                if ex > cont:
                    vals[i] = ex
    return vals[0]


# ---------------------------------------------------------------------------
# 写法 2：NumPy 向量化（内层循环消失）
# ---------------------------------------------------------------------------
def binomial_numpy(spot, strike, rate, sigma, ttm, n, option_type="call",
                   style="european") -> float:
    """后向归纳只在"步"这一维循环，节点维一次性算完。"""
    u, d, p, df = _params(rate, sigma, ttm, n)
    is_call = str(option_type).lower() == "call"

    i = np.arange(n + 1, dtype=np.float64)
    terminal = spot * u ** (n - i) * d ** i
    vals = np.maximum(terminal - strike, 0.0) if is_call else np.maximum(strike - terminal, 0.0)

    if style != "american":
        for j in range(n - 1, -1, -1):
            vals = df * (p * vals[:-1] + (1.0 - p) * vals[1:])
        return float(vals[0])

    for j in range(n - 1, -1, -1):
        vals = df * (p * vals[:-1] + (1.0 - p) * vals[1:])
        k = np.arange(j + 1, dtype=np.float64)
        s = spot * u ** (j - k) * d ** k
        ex = np.maximum(s - strike, 0.0) if is_call else np.maximum(strike - s, 0.0)
        np.maximum(vals, ex, out=vals)   # 复用缓冲区，不产生临时数组
    return float(vals[0])


# ---------------------------------------------------------------------------
# 写法 3：Numba JIT（编译成本地机器码）
# ---------------------------------------------------------------------------
if HAS_NUMBA:

    @njit(cache=True, fastmath=True, nogil=True)
    def _binomial_kernel(spot, strike, rate, sigma, ttm, n, is_call, is_american):
        dt = ttm / n
        u = 1.0 + sigma * math.sqrt(dt)
        d = 1.0 - sigma * math.sqrt(dt)
        p = 0.5 + rate * math.sqrt(dt) / (2.0 * sigma)
        df = 1.0 / (1.0 + rate * dt)
        vals = np.empty(n + 1, dtype=np.float64)
        for i in range(n + 1):
            s = spot * u ** (n - i) * d ** i
            vals[i] = max(s - strike, 0.0) if is_call else max(strike - s, 0.0)
        for j in range(n - 1, -1, -1):
            for i in range(j + 1):
                cont = df * (p * vals[i] + (1.0 - p) * vals[i + 1])
                if is_american:
                    s = spot * u ** (j - i) * d ** i
                    ex = max(s - strike, 0.0) if is_call else max(strike - s, 0.0)
                    cont = max(cont, ex)
                vals[i] = cont
        return vals[0]

    def binomial_numba(spot, strike, rate, sigma, ttm, n, option_type="call",
                       style="european") -> float:
        """Numba 版：第一次调用要编译（约 0.3s），之后每次都是机器码速度。"""
        return float(_binomial_kernel(float(spot), float(strike), float(rate),
                                      float(sigma), float(ttm), int(n),
                                      str(option_type).lower() == "call",
                                      str(style).lower() == "american"))
else:  # pragma: no cover
    def binomial_numba(*args, **kwargs) -> float:
        raise RuntimeError("未安装 numba：pip install numba")


# ---------------------------------------------------------------------------
# 写法 4：闭式解（欧式期权一次算完，连"步"的循环都没有）
# ---------------------------------------------------------------------------
def binomial_european_cf(spot, strike, rate, sigma, ttm, n, option_type="call") -> float:
    """CRR 的解析形式：

        V = df^n * sum_k C(n,k) p^k (1-p)^(n-k) * payoff(S u^(n-k) d^k)

    整条式子只用一次数组运算，复杂度 O(n)，是欧式期权最快的离散解法。
    """
    from scipy.special import gammaln
    u, d, p, df = _params(rate, sigma, ttm, n)
    k = np.arange(n + 1, dtype=np.float64)          # k = 下跌次数
    # 直接在概率空间算 comb(n,k)*(1-p)^k*p^(n-k) 会在 n>1000 时上溢成 inf*0=nan，
    # 所以先取对数再 exp 回来（这正是"数值稳定"要处理的事）
    log_w = (gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
             + k * np.log1p(-p) + (n - k) * np.log(p))
    w = np.exp(log_w)
    s = spot * u ** (n - k) * d ** k
    payoff = np.maximum(s - strike, 0.0) if str(option_type).lower() == "call" \
        else np.maximum(strike - s, 0.0)
    return float(df ** n * np.dot(w, payoff))


# ---------------------------------------------------------------------------
# 对比与扫参
# ---------------------------------------------------------------------------
def benchmark(spot=100.0, strike=100.0, rate=0.05, sigma=0.2, ttm=1.0,
              sizes=(50, 200, 1000, 3000), repeat=5) -> "object":
    """跑一遍四种写法，返回 pandas.DataFrame（毫秒）。"""
    import pandas as pd

    impls = [
        ("纯 Python", binomial_py),
        ("NumPy 向量化", binomial_numpy),
        ("闭式解(欧式)", binomial_european_cf),
    ]
    if HAS_NUMBA:
        impls.append(("Numba JIT", binomial_numba))

    rows = []
    for n in sizes:
        row = {"steps": n}
        for name, fn in impls:
            try:
                fn(spot, strike, rate, sigma, ttm, 3)          # 预热（Numba 编译）
                t0 = time.perf_counter()
                for _ in range(repeat):
                    fn(spot, strike, rate, sigma, ttm, n)
                row[name] = (time.perf_counter() - t0) / repeat * 1000
            except Exception as exc:
                row[name] = float("nan")
        rows.append(row)
    df = pd.DataFrame(rows).set_index("steps")
    if "纯 Python" in df.columns:
        for c in df.columns:
            if c != "纯 Python":
                df[c + " 加速"] = df["纯 Python"] / df[c]
    return df.round(3)


def sweep_steps(spot=100.0, strike=100.0, rate=0.05, sigma=0.2, ttm=1.0,
                sizes=(5, 10, 25, 50, 100, 200, 500, 1000, 2000)) -> "object":
    """步数 -> 价格 的收敛曲线数据（用来讲"锯齿状收敛"很直观）。"""
    import pandas as pd
    from .options import black_scholes
    rows = []
    for n in sizes:
        rows.append({
            "steps": n,
            "binomial": binomial_numpy(spot, strike, rate, sigma, ttm, n),
            "black_scholes": black_scholes(spot, strike, ttm, rate, sigma),
        })
    df = pd.DataFrame(rows).set_index("steps")
    df["abs_error"] = (df["binomial"] - df["black_scholes"]).abs()
    return df
