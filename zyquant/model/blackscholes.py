# -*- coding: utf-8 -*-
"""
zyquant.model.blackscholes — Vectorized Black-Scholes

Provides 5 functional APIs and 1 object-oriented interface:
    - black_scholes         European option price (scalar / array)
    - black_76              Futures / forward option price (Black-76), q = r
    - garman_kohlhagen      FX option price (Garman-Kohlhagen), r = r_d, q = r_f
    - bs_greeks             Price plus five Greeks (scalar / array)
    - implied_vol           Implied volatility from a market price (scalar / array)
    - BlackScholes          OOP wrapper (compute once, print a report)

`BlackScholes` highlights:
    - Attributes: `model.price`, `model.delta`, `model.gamma`, `model.vega`,
      `model.theta`, `model.rho`
    - Pretty Print: `print(model)` (aligned six-decimal report)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from scipy.stats import norm
from .inputs import Kind, Number, OptionInputs, OptionType, check_inputs

__all__ = ["black_scholes", "black_76", "garman_kohlhagen", "bs_greeks", "implied_vol", "BlackScholes"]

_DAYS_PER_YEAR = 365.0
_PERCENT = 100.0
_VOL_START = 0.3
_VOL_FLOOR = 1e-8
_VOL_CAP = 10.0
_BISECTION_ROUNDS = 60


def _intrinsic(S, K, kind: OptionType):
    '''Undiscounted intrinsic value.'''
    if kind is OptionType.CALL:
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


def _d1_d2(S, K, r, q, sigma, T):
    '''Return the d1 and d2 terms.'''
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return d1, d1 - sigma * sqrt_T


def black_scholes(S: Number, K: Number, r: float, sigma: Number, T: Number, *,
                  kind: Kind = "call", q: float = 0.0) -> Number:
    """
    Price a European option with the Black-Scholes formula.

    Parameters
    ----------
    S, K : float or np.ndarray
        Spot price and strike price.
    r : float
        Risk-free interest rate (decimal, e.g. 0.05 for 5%).
    sigma : float or np.ndarray
        Annualized volatility (decimal).
    T : float or np.ndarray
        Time to maturity (in years).
    kind : str
        Option type: "call" or "put".
    q : float, optional
        Continuous dividend yield, default is 0.0. A positive q gives the
        Merton model.

    Returns
    -------
    np.float64 or np.ndarray
        Option price; arrays are priced element-wise under numpy broadcasting.
    """
    kind = OptionType.parse(kind)
    S, K, sigma, T = (np.asarray(value, dtype=float) for value in (S, K, sigma, T))

    if np.all(T <= 0):
        return _intrinsic(S, K, kind)
    if np.any(T <= 0):
        raise ValueError(f"T must be either all positive or all non-positive, got {T!r}")

    if np.all(sigma <= 0):
        pv_spot = S * np.exp(-q * T)
        pv_strike = K * np.exp(-r * T)
        if kind is OptionType.CALL:
            return np.maximum(pv_spot - pv_strike, 0.0)
        return np.maximum(pv_strike - pv_spot, 0.0)
    if np.any(sigma <= 0):
        raise ValueError(f"sigma must be either all positive or all zero, got {sigma!r}")

    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if kind is OptionType.CALL:
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    return price


def black_76(F: Number, K: Number, r: float, sigma: Number, T: Number, *, kind: Kind = "call") -> Number:
    """
    Price a futures / forward option with the Black-76 model.

    Parameters
    ----------
    F : float or np.ndarray
        Futures or forward price.
    K : float or np.ndarray
        Strike price.
    r, sigma, T : float or np.ndarray
        Risk-free rate, annualized volatility and time to maturity (in years).
    kind : str
        Option type: "call" or "put".

    Returns
    -------
    np.float64 or np.ndarray
        Option price.
    """
    return black_scholes(F, K, r, sigma, T, kind=kind, q=r)


def garman_kohlhagen(S: Number, K: Number, r_d: float, r_f: float,
                     sigma: Number, T: Number, *, kind: Kind = "call") -> Number:
    """
    Price an FX option with the Garman-Kohlhagen model.

    Parameters
    ----------
    S : float or np.ndarray
        Spot exchange rate.
    K : float or np.ndarray
        Strike price.
    r_d, r_f : float
        Domestic and foreign risk-free rates (decimal).
    sigma, T : float or np.ndarray
        Annualized volatility and time to maturity (in years).
    kind : str
        Option type: "call" or "put".

    Returns
    -------
    np.float64 or np.ndarray
        Option price.
    """
    return black_scholes(S, K, r_d, sigma, T, kind=kind, q=r_f)


def bs_greeks(S: Number, K: Number, r: float, sigma: Number, T: Number, *,
              kind: Kind = "call", q: float = 0.0) -> Dict[str, Number]:
    """
    Return the price together with five Greeks in a single pass.

    Parameters
    ----------
    S, K : float or np.ndarray
        Spot price and strike price.
    r : float
        Risk-free interest rate (decimal).
    sigma : float or np.ndarray
        Annualized volatility (decimal).
    T : float or np.ndarray
        Time to maturity (in years).
    kind : str
        Option type: "call" or "put".
    q : float, optional
        Continuous dividend yield, default is 0.0.

    Returns
    -------
    dict
        Keys: price / delta / gamma / vega / theta / rho. Array inputs produce
        arrays of the same shape.

    Notes
    -----
    Scalar inputs go through check_inputs first; array inputs skip the check
    because validating every element is too expensive, leaving the caller
    responsible for sensible parameters.
    """
    kind = OptionType.parse(kind)
    try:
        check_inputs(float(S), float(K), float(r), float(sigma), float(T), q)
    except TypeError:
        pass

    S, K, sigma, T = (np.asarray(value, dtype=float) for value in (S, K, sigma, T))
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)

    sqrt_T = np.sqrt(T)

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_minus_d1 = norm.cdf(-d1)
    N_minus_d2 = norm.cdf(-d2)
    phi_d1 = norm.pdf(d1)

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if kind is OptionType.CALL:
        price = S * disc_q * N_d1 - K * disc_r * N_d2
        delta = disc_q * N_d1
        theta = (-S * disc_q * phi_d1 * sigma / (2.0 * sqrt_T)
                 + q * S * disc_q * N_d1
                 - r * K * disc_r * N_d2) / _DAYS_PER_YEAR
        rho = K * T * disc_r * N_d2 / _PERCENT
    else:
        price = K * disc_r * N_minus_d2 - S * disc_q * N_minus_d1
        delta = -disc_q * N_minus_d1
        theta = (-S * disc_q * phi_d1 * sigma / (2.0 * sqrt_T)
                 - q * S * disc_q * N_minus_d1
                 + r * K * disc_r * N_minus_d2) / _DAYS_PER_YEAR
        rho = -K * T * disc_r * N_minus_d2 / _PERCENT

    return {
        "price": price,
        "delta": delta,
        "gamma": disc_q * phi_d1 / (S * sigma * sqrt_T),
        "vega": S * disc_q * phi_d1 * sqrt_T / _PERCENT,
        "theta": theta,
        "rho": rho}


def implied_vol(price: Number, S: Number, K: Number, r: float, T: Number, *,
                kind: Kind = "call", q: float = 0.0,
                tol: float = 1e-10, max_iter: int = 100) -> Number:
    """
    Solve for the implied volatility of a European option.

    Parameters
    ----------
    price : float or np.ndarray
        Observed market price.
    S, K : float or np.ndarray
        Spot price and strike price.
    r : float
        Risk-free interest rate (decimal).
    T : float or np.ndarray
        Time to maturity (in years).
    kind : str
        Option type: "call" or "put".
    q : float, optional
        Continuous dividend yield, default is 0.0.
    tol : float, optional
        Price convergence tolerance, default is 1e-10.
    max_iter : int, optional
        Maximum Newton iterations before falling back, default is 100.

    Returns
    -------
    np.float64 or np.ndarray
        Implied volatility; NaN where the price is not attainable
        (non-positive, or below intrinsic value).
    """
    kind = OptionType.parse(kind)
    price = np.asarray(price, dtype=float)
    S, K, T = (np.asarray(value, dtype=float) for value in (S, K, T))

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    pv_spot = S * disc_q
    pv_strike = K * disc_r

    if kind is OptionType.CALL:
        intrinsic = np.maximum(pv_spot - pv_strike, 0.0)
    else:
        intrinsic = np.maximum(pv_strike - pv_spot, 0.0)

    attainable = np.isfinite(price) & (price > 0) & (price >= intrinsic - 1e-12)
    vol = np.full(np.broadcast(price, S, K, T).shape, _VOL_START)
    pending = attainable.copy()

    for _ in range(max_iter):
        if not pending.any():
            break
        model_price = black_scholes(S, K, r, vol, T, kind=kind, q=q)
        gap = np.asarray(model_price) - price
        pending &= np.abs(gap) >= tol
        if not pending.any():
            break
        d1, _ = _d1_d2(S, K, r, q, vol, T)

        vega = S * disc_q * norm.pdf(d1) * np.sqrt(T)

        usable = pending & (vega > 1e-12)
        step = np.where(usable, gap / np.where(usable, vega, 1.0), 0.0)
        vol = np.where(pending, vol - step, vol)
        escaped = pending & ((vol <= _VOL_FLOOR) | (vol > _VOL_CAP))
        pending &= ~escaped

    final_gap = np.abs(np.asarray(black_scholes(S, K, r, vol, T, kind=kind, q=q)) - price)
    stuck = np.flatnonzero((attainable & (final_gap >= tol)).reshape(-1))
    if stuck.size:
        low = np.full(stuck.size, _VOL_FLOOR)
        high = np.full(stuck.size, _VOL_CAP)
        flat = vol.reshape(-1)
        for _ in range(_BISECTION_ROUNDS):
            middle = 0.5 * (low + high)
            flat[stuck] = middle
            model_price = np.asarray(black_scholes(S, K, r, vol, T, kind=kind, q=q)).reshape(-1)
            above = model_price[stuck] > price.reshape(-1)[stuck]
            high = np.where(above, middle, high)
            low = np.where(above, low, middle)
        flat[stuck] = 0.5 * (low + high)
        vol = flat.reshape(vol.shape)

    vol = np.where(attainable, vol, np.nan)
    return vol


@dataclass
class BlackScholes:
    """
    A single option: feed it parameters, read price and Greeks off the object,
    and print(model) gives an aligned report.
    """
    inputs: OptionInputs
    option_type: OptionType = OptionType.CALL

    def __post_init__(self) -> None:
        self.option_type = OptionType.parse(self.option_type)
        a = self.inputs
        check_inputs(a.S, a.K, a.r, a.sigma, a.T, a.q)
        self._values = bs_greeks(a.S, a.K, a.r, a.sigma, a.T, kind=self.option_type, q=a.q)

    @property
    def price(self) -> float:
        return float(self._values["price"])

    @property
    def delta(self) -> float:
        return float(self._values["delta"])

    @property
    def gamma(self) -> float:
        return float(self._values["gamma"])

    @property
    def vega(self) -> float:
        return float(self._values["vega"])

    @property
    def theta(self) -> float:
        return float(self._values["theta"])

    @property
    def rho(self) -> float:
        return float(self._values["rho"])

    def __str__(self) -> str:
        a = self.inputs
        lines = [
            f"{'─' * 54}",
            f"  Black-Scholes ({self.option_type.value.title()})",
            f"{'─' * 54}",
            f"  Spot = {a.S:<10.4f} Strike = {a.K:<10.4f}",
            f"  Rate = {a.r:<10.4f} Vol    = {a.sigma:<10.4f}",
            f"  Time = {a.T:<10.4f} Carry  = {a.q:<10.4f}",
            f"{'─' * 54}",
            f"  {'Price':22s}  {self.price:>10.6f}",
            f"  {'Delta':22s}  {self.delta:>10.6f}",
            f"  {'Gamma':22s}  {self.gamma:>10.6f}",
            f"  {'Vega':22s}  {self.vega:>10.6f}",
            f"  {'Theta':22s}  {self.theta:>10.6f}",
            f"  {'Rho':22s}  {self.rho:>10.6f}",
            f"{'─' * 54}"]
        return "\n".join(line.rstrip() for line in lines)

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=110)

    print("--- 1. Three models, same inputs ---")
    print(f"  Black-Scholes    : {float(black_scholes(100, 100, 0.05, 0.2, 1)):.6f}")
    print(f"  Black-76         : {float(black_76(100, 100, 0.05, 0.2, 1)):.6f}")
    print(f"  Garman-Kohlhagen : {float(garman_kohlhagen(100, 100, 0.05, 0.03, 0.2, 1)):.6f}")

    print("\n--- 2. Price and five Greeks ---")
    for kind in ("call", "put"):
        greeks = bs_greeks(100, 100, 0.05, 0.2, 1, kind=kind)
        pairs = "  ".join(f"{name}={float(value):>9.6f}" for name, value in greeks.items())
        print(f"  {kind:<5} {pairs}")

    print("\n--- 3. Batch: strikes 90 / 100 / 110 ---")
    strikes = np.array([90.0, 100.0, 110.0])
    prices = black_scholes(100.0, strikes, 0.05, 0.25, 0.5, q=0.02)
    deltas = bs_greeks(100.0, strikes, 0.05, 0.25, 0.5, q=0.02)["delta"]
    vols = implied_vol(prices, 100.0, strikes, 0.05, 0.5, q=0.02)
    print(f"  price {prices}")
    print(f"  delta {deltas}")
    print(f"  iv    {vols}  (target 0.25)")

    print("\n--- 4. Object wrapper ---")
    option = OptionInputs(S=100, K=105, T=0.5, r=0.05, sigma=0.25, q=0.02)
    for kind in ("call", "put"):
        print(BlackScholes(inputs=option, option_type=kind))
        print("\n")
