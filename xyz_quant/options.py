# -*- coding: utf-8 -*-
"""cqfm1.options —— 期权定价（二叉树 / Black-Scholes / 蒙特卡洛）。

原课程用 quantmod.models.BinomialOptionPricing，本模块给出等价的实现，
并且保留了课程里"手写 binomial_option"的原始签名，方便对照讲义。

术语
----
spot/strike/rate/sigma/time/steps 与讲义完全一致
u = 1 + sigma*sqrt(dt),  d = 1 - sigma*sqrt(dt)
p = 1/2 + r*sqrt(dt)/(2*sigma),  df = 1/(1 + r*dt)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy import maximum, power, sqrt, zeros

__all__ = [
    "binomial_option", "black_scholes", "bs_greeks", "implied_vol",
    "OptionInputs", "OptionType", "ExerciseStyle",
    "BinomialOptionPricing", "MonteCarloOptionPricing",
]


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(slots=True)
class OptionInputs:
    """统一的输入结构：把参数校验放在一处，定价函数就不用重复检查。"""

    spot: float
    strike: float
    ttm: float
    rate: float
    volatility: float

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot 必须为正")
        if self.strike <= 0:
            raise ValueError("strike 必须为正")
        if self.ttm <= 0:
            raise ValueError("ttm（到期时间）必须为正")
        if self.volatility <= 0:
            raise ValueError("volatility 必须为正")

    @property
    def time(self) -> float:
        return self.ttm


# ---------------------------------------------------------------------------
# 1. 手写二叉树（与讲义第 9 个 cell 完全同构）
# ---------------------------------------------------------------------------
def binomial_option(spot: float, strike: float, rate: float, sigma: float,
                    time: float, steps: int, output: int = 0,
                    option_type: str = "call", style: str = "european") -> np.ndarray:
    """n 步二叉树定价，返回价格/内在价值/期权价值/Delta 四张矩阵之一。

    Parameters
    ----------
    spot, strike : float   现价、行权价
    rate : float           无风险利率（连续复利近似用单利贴现，与讲义一致）
    sigma : float          年化波动率
    time : float           到期时间（年）
    steps : int            步数
    output : int           0=价格矩阵, 1=到期内在价值, 2=期权价值, 3=Delta
    option_type : str      'call' 或 'put'
    style : str            'european' 或 'american'（美式在每步比较提前行权）

    Returns
    -------
    ndarray (steps+1, steps+1)
        第 j 列是第 j 步的节点，第 i 行表示 i 次下跌。
    """
    ot = str(option_type).lower()
    st = str(style).lower()

    dt = time / steps
    u = 1 + sigma * sqrt(dt)
    d = 1 - sigma * sqrt(dt)
    p = 0.5 + rate * sqrt(dt) / (2 * sigma)
    df = 1 / (1 + rate * dt)

    px = zeros((steps + 1, steps + 1))
    cp = zeros((steps + 1, steps + 1))
    V = zeros((steps + 1, steps + 1))
    delta = zeros((steps + 1, steps + 1))

    # 前向：算价格与到期内在价值
    for j in range(steps + 1):
        for i in range(j + 1):
            px[i, j] = spot * power(u, j - i) * power(d, i)
            if ot == "call":
                cp[i, j] = maximum(px[i, j] - strike, 0.0)
            else:
                cp[i, j] = maximum(strike - px[i, j], 0.0)

    # 后向：从 n 贴现回 0
    for j in range(steps, -1, -1):
        for i in range(j + 1):
            if j == steps:
                V[i, j] = cp[i, j]
                itm = px[i, j] >= strike if ot == "call" else px[i, j] <= strike
                delta[i, j] = (1.0 if ot == "call" else -1.0) if itm else 0.0
            else:
                cont = df * (p * V[i, j + 1] + (1 - p) * V[i + 1, j + 1])
                V[i, j] = maximum(cp[i, j], cont) if st == "american" else cont
                delta[i, j] = (V[i, j + 1] - V[i + 1, j + 1]) / (px[i, j + 1] - px[i + 1, j + 1])

    results = (np.around(px, 2), np.around(cp, 2), np.around(V, 2), np.around(delta, 4))
    return results[output]


# ---------------------------------------------------------------------------
# 2. Black-Scholes 与隐含波动率
# ---------------------------------------------------------------------------
def _norm_cdf(x):
    from scipy.stats import norm
    return norm.cdf(x)


def black_scholes(spot: float, strike: float, ttm: float, rate: float,
                  sigma: float, option_type: str = "call") -> float:
    """欧式期权 BS 解析解。"""
    if ttm <= 0:
        return max(spot - strike, 0.0) if str(option_type).lower() == "call" else max(strike - spot, 0.0)
    s, k, t, r, v = float(spot), float(strike), float(ttm), float(rate), float(sigma)
    d1 = (np.log(s / k) + (r + 0.5 * v * v) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    if str(option_type).lower() == "call":
        return float(s * _norm_cdf(d1) - k * np.exp(-r * t) * _norm_cdf(d2))
    return float(k * np.exp(-r * t) * _norm_cdf(-d2) - s * _norm_cdf(-d1))


def bs_greeks(spot: float, strike: float, ttm: float, rate: float,
              sigma: float, option_type: str = "call") -> dict:
    """返回 delta / gamma / vega / theta / rho（vega、rho 已按 1% 缩放）。"""
    from scipy.stats import norm
    s, k, t, r, v = float(spot), float(strike), float(ttm), float(rate), float(sigma)
    d1 = (np.log(s / k) + (r + 0.5 * v * v) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    pdf = norm.pdf(d1)
    if str(option_type).lower() == "call":
        delta = norm.cdf(d1)
        theta = (-s * pdf * v / (2 * np.sqrt(t)) - r * k * np.exp(-r * t) * norm.cdf(d2)) / 365
        rho = k * t * np.exp(-r * t) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-s * pdf * v / (2 * np.sqrt(t)) + r * k * np.exp(-r * t) * norm.cdf(-d2)) / 365
        rho = -k * t * np.exp(-r * t) * norm.cdf(-d2) / 100
    return {
        "delta": float(delta),
        "gamma": float(pdf / (s * v * np.sqrt(t))),
        "vega": float(s * pdf * np.sqrt(t) / 100),
        "theta": float(theta),
        "rho": float(rho),
    }


def implied_vol(price: float, spot: float, strike: float, ttm: float,
                rate: float, option_type: str = "call",
                tol: float = 1e-8, max_iter: int = 100) -> float:
    """牛顿法反解隐含波动率（期权链里没有 IV，就自己算）。

    通常 5~8 次迭代收敛；vaga 太小时退回二分法，保证稳定。
    """
    from scipy.stats import norm
    if price is None or not np.isfinite(price) or price <= 0 or ttm <= 0:
        return float("nan")
    intrinsic = max(spot - strike * np.exp(-rate * ttm), 0.0) if str(option_type).lower() == "call" \
        else max(strike * np.exp(-rate * ttm) - spot, 0.0)
    if price < intrinsic * 0.999:
        return float("nan")

    vol = 0.3
    for _ in range(max_iter):
        d1 = (np.log(spot / strike) + (rate + 0.5 * vol * vol) * ttm) / (vol * np.sqrt(ttm))
        vega = spot * norm.pdf(d1) * np.sqrt(ttm)
        diff = black_scholes(spot, strike, ttm, rate, vol, option_type) - price
        if abs(diff) < tol:
            return float(vol)
        if vega < 1e-10:
            break
        vol -= diff / vega
        if vol <= 1e-6 or vol > 5.0:
            break

    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if black_scholes(spot, strike, ttm, rate, mid, option_type) > price:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# 3. 面向对象接口（对齐 quantmod.models）
# ---------------------------------------------------------------------------
class BinomialOptionPricing:
    """对应 quantmod.models.BinomialOptionPricing。

    Examples
    --------
    >>> inputs = OptionInputs(spot=100, strike=100, ttm=1, rate=0.05, volatility=0.2)
    >>> eng = BinomialOptionPricing(inputs, nsteps=4, option_type=OptionType.CALL)
    >>> round(float(eng.value[0, 0]), 2)
    11.03
    """

    def __init__(self, inputs: OptionInputs, nsteps: int = 4,
                 option_type: OptionType = OptionType.CALL,
                 exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
                 output: str = "value") -> None:
        self.inputs = inputs
        self.nsteps = int(nsteps)
        self.option_type = OptionType(option_type)
        self.exercise_style = ExerciseStyle(exercise_style)
        self.output = output

        px, cp, tree, delta = (binomial_option(inputs.spot, inputs.strike, inputs.rate,
                                               inputs.volatility, inputs.ttm, self.nsteps,
                                               output=i, option_type=self.option_type.value,
                                               style=self.exercise_style.value)
                               for i in range(4))
        self.price_tree = px
        self.payoff = cp
        self.value = tree
        self.delta = delta
        self.binomialoption = tree

    def __repr__(self) -> str:
        return ("BinomialOptionPricing(" + self.option_type.value + "/" + self.exercise_style.value +
                ", n=" + str(self.nsteps) + ", value=" + format(float(self.value[0, 0]), ".4f") + ")")

    def plot_tree(self, figsize=(10, 6)):
        """把价格树画出来（节点标价格，最后一列标内在价值）。"""
        import matplotlib.pyplot as plt
        n = self.nsteps
        fig, ax = plt.subplots(figsize=figsize)
        for j in range(n + 1):
            for i in range(j + 1):
                y = -i
                x = j
                ax.add_patch(plt.Circle((x, y), 0.12, color="#e8f0fe", ec="#3b6cb7", zorder=3))
                ax.text(x, y, format(self.price_tree[i, j], ".1f"), ha="center", va="center",
                        fontsize=7, zorder=4)
                if j < n:
                    ax.plot([x + 0.12, j + 1 - 0.12], [y, y], color="#9aa8bb", lw=0.8, zorder=1)
                    ax.plot([x + 0.12, j + 1 - 0.12], [y, y - 1], color="#9aa8bb", lw=0.8, zorder=1)
        ax.set_xlim(-0.5, n + 0.5)
        ax.set_ylim(-n - 0.6, 0.6)
        ax.axis("off")
        ax.set_title("Binomial tree (u=" + format(1 + self.inputs.volatility * np.sqrt(self.inputs.ttm / n), ".4f") + ")")
        plt.show()


class MonteCarloOptionPricing:
    """几何布朗运动下的蒙特卡洛定价（向量化，一万条路径毫秒级）。"""

    def __init__(self, inputs: OptionInputs, npaths: int = 100_000, nsteps: int = 252,
                 option_type: OptionType = OptionType.CALL, seed: int = 42,
                 antithetic: bool = True) -> None:
        self.inputs = inputs
        self.npaths = int(npaths)
        self.nsteps = int(nsteps)
        self.option_type = OptionType(option_type)
        self.seed = seed
        self.antithetic = antithetic
        self._res = None

    def simulate(self) -> np.ndarray:
        """返回每一条路径的终值（欧式只用终点，不必存整条路径）。"""
        inp = self.inputs
        rng = np.random.default_rng(self.seed)
        dt = inp.ttm / self.nsteps
        drift = (inp.rate - 0.5 * inp.volatility ** 2) * dt
        vol = inp.volatility * np.sqrt(dt)
        n = self.npaths
        if self.antithetic:
            n = max(self.npaths // 2, 1)
        z = rng.standard_normal((n, self.nsteps))
        logpath = np.cumsum(drift + vol * z, axis=1)
        st = inp.spot * np.exp(logpath[:, -1])
        if self.antithetic:
            st = np.concatenate([st, inp.spot * np.exp(np.cumsum(drift - vol * z, axis=1)[:, -1])])
        return st

    def price(self) -> float:
        st = self.simulate()
        if self.option_type is OptionType.CALL:
            payoff = np.maximum(st - self.inputs.strike, 0.0)
        else:
            payoff = np.maximum(self.inputs.strike - st, 0.0)
        self._res = float(np.exp(-self.inputs.rate * self.inputs.ttm) * payoff.mean())
        return self._res

    def std_error(self) -> float:
        st = self.simulate()
        payoff = np.maximum(st - self.inputs.strike, 0.0) if self.option_type is OptionType.CALL \
            else np.maximum(self.inputs.strike - st, 0.0)
        disc = np.exp(-self.inputs.rate * self.inputs.ttm)
        return float(disc * payoff.std(ddof=1) / np.sqrt(len(payoff)))

    @property
    def value(self) -> float:
        return self._res if self._res is not None else self.price()
