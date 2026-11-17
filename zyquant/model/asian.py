# -*- coding: utf-8 -*-
"""
zyquant.model.asian — Geometric Asian Option Pricing (Kemna-Vorst)

Provides 2 functional APIs and 1 object-oriented interface:
    - geometric_asian_option    Closed-form price of a geometric-average option
    - geometric_asian_forward   Risk-neutral expectation E[G] of the average price
    - AsianModel                OOP wrapper (compute once, print a report)

Only the geometric average has a closed form. Taking logs turns the average into
a linear combination of Gaussians, so G stays log-normal and the Black-Scholes
machinery applies unchanged. The arithmetic average has no such property and
needs an approximation (Turnbull-Wakeman, Levy) or Monte Carlo.

`AsianModel` highlights:
    - Attributes: `model.price`, `model.forward`, `model.strike` (t=0 states)
    - Pretty Print: `print(model)` (aligned six-decimal report)
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from scipy.stats import norm
from .inputs import (Kind, OptionInputs, OptionType, check_inputs, check_steps)

__all__ = ["geometric_asian_option", "geometric_asian_forward", "AsianModel"]

_DEFAULT_STEPS = 252


def _moments(S: float, r: float, sigma: float, T: float, q: float, n: int):
    '''Mean and variance of ln G under the risk-neutral measure.'''
    dt = T / n
    mean_ln = math.log(S) + (r - q - 0.5 * sigma ** 2) * dt * (n + 1) / 2.0
    var_ln = sigma ** 2 * T * (n + 1) * (2 * n + 1) / (6.0 * n ** 2)
    return mean_ln, var_ln


def _d1_d2(forward: float, strike: float, var_ln: float):
    '''d1 and d2 for an option on the geometric average.'''
    sd = math.sqrt(var_ln)
    d1 = (math.log(forward / strike) + 0.5 * var_ln) / sd
    return d1, d1 - sd


def geometric_asian_forward(S: float, r: float, sigma: float, T: float, *,
                            q: float = 0.0, n_steps: int = _DEFAULT_STEPS) -> float:
    """
    Return the risk-neutral expectation E[G] of the geometric average price.

    Parameters
    ----------
    S : float
        Spot price.
    r : float
        Risk-free interest rate (decimal).
    sigma : float
        Annualized volatility (decimal).
    T : float
        Time to maturity (in years).
    q : float, optional
        Continuous dividend yield, default is 0.0.
    n_steps : int, optional
        Number of averaging dates, default is 252.

    Returns
    -------
    float
        Expected geometric average, exp(E[ln G] + Var[ln G] / 2).

    Notes
    -----
    With averaging dates t_i = i * dt (i = 1..n):

        E[ln G]   = ln S + (r - q - sigma^2 / 2) * dt * (n + 1) / 2
        Var[ln G] = sigma^2 * T * (n + 1)(2n + 1) / (6 n^2)

    As n grows the variance factor tends to 1/3, giving the familiar sigma / sqrt(3).
    """
    n = check_steps(n_steps)
    mean_ln, var_ln = _moments(float(S), float(r), float(sigma), float(T), float(q), n)
    return math.exp(mean_ln + 0.5 * var_ln)


def geometric_asian_option(S: float, K: float, r: float, sigma: float, T: float, *,
                           kind: Kind = "call", q: float = 0.0,
                           n_steps: int = _DEFAULT_STEPS) -> float:
    """
    Price a geometric-average Asian option in closed form.

    Parameters
    ----------
    S, K : float
        Spot price and strike price.
    r : float
        Risk-free interest rate (decimal).
    sigma : float
        Annualized volatility (decimal).
    T : float
        Time to maturity (in years).
    kind : str
        Option type: "call" or "put".
    q : float, optional
        Continuous dividend yield, default is 0.0.
    n_steps : int, optional
        Number of averaging dates, default is 252. Match this to the Monte Carlo
        path resolution if the two are being compared.

    Returns
    -------
    float
        Option price.

    Notes
    -----
    G is treated as a new underlying that is log-normal with mean E[G] and
    volatility sigma_A, so the price has the Black-Scholes shape:

        d1 = [ln(E[G] / K) + sigma_A^2 T / 2] / (sigma_A sqrt(T))
        d2 = d1 - sigma_A sqrt(T)
        C  = exp(-rT) * [E[G] * N(d1) - K * N(d2)]
        P  = exp(-rT) * [K * N(-d2) - E[G] * N(-d1)]
    """
    kind = OptionType.parse(kind)
    check_inputs(float(S), float(K), float(r), float(sigma), float(T), q)
    n = check_steps(n_steps)

    S, K, T = float(S), float(K), float(T)
    mean_ln, var_ln = _moments(S, float(r), float(sigma), T, float(q), n)
    forward = math.exp(mean_ln + 0.5 * var_ln)
    d1, d2 = _d1_d2(forward, K, var_ln)
    discount = math.exp(-float(r) * T)

    if kind is OptionType.CALL:
        return discount * (forward * norm.cdf(d1) - K * norm.cdf(d2))
    return discount * (K * norm.cdf(-d2) - forward * norm.cdf(-d1))


@dataclass
class AsianModel:
    """
    A single geometric-average Asian option: compute once, read the fields off
    the object, and print(model) gives an aligned report.
    """

    inputs: OptionInputs
    option_type: OptionType = OptionType.CALL
    n_steps: int = _DEFAULT_STEPS

    def __post_init__(self) -> None:
        self.option_type = OptionType.parse(self.option_type)
        self.n_steps = check_steps(self.n_steps)
        a = self.inputs
        check_inputs(a.S, a.K, a.r, a.sigma, a.T, a.q)
        self.forward = geometric_asian_forward(a.S, a.r, a.sigma, a.T, q=a.q, n_steps=self.n_steps)
        self._price = geometric_asian_option(
            a.S, a.K, a.r, a.sigma, a.T,
            kind=self.option_type, q=a.q, n_steps=self.n_steps)

    @property
    def strike(self) -> float:
        return float(self.inputs.K)

    @property
    def price(self) -> float:
        return float(self._price)

    def __str__(self) -> str:
        a = self.inputs
        lines = [
            f"{'─' * 54}",
            f"  Geometric Asian ({self.option_type.value.title()})",
            f"{'─' * 54}",
            f"  Spot = {a.S:<10.4f} Strike = {a.K:<10.4f}",
            f"  Rate = {a.r:<10.4f} Vol    = {a.sigma:<10.4f}",
            f"  Time = {a.T:<10.4f} Carry  = {a.q:<10.4f}",
            f"  Steps = {self.n_steps}",
            f"{'─' * 54}",
            f"  {'Forward (E[G])':22s}  {self.forward:>10.6f}",
            f"  {'Price':22s}  {self.price:>10.6f}",
            f"{'─' * 54}"]
        return "\n".join(line.rstrip() for line in lines)


if __name__ == "__main__":
    option = OptionInputs(S=100, K=105, T=0.5, r=0.05, sigma=0.25, q=0.02)

    print("--- 1. Closed form, both directions ---")
    for kind in ("call", "put"):
        price = geometric_asian_option(100, 105, 0.05, 0.25, 0.5, kind=kind, q=0.02)
        print(f"  {kind:<5} {price:>10.6f}")

    print("\n--- 2. Convergence in averaging dates ---")
    for steps in (12, 52, 252, 1000, 10000):
        price = geometric_asian_option(100, 100, 0.05, 0.2, 1.0, n_steps=steps)
        print(f"  steps={steps:<6d} price={price:>10.6f}")

    print("\n--- 3. Put-call parity ---")
    forward = geometric_asian_forward(100, 0.05, 0.2, 1.0, n_steps=252)
    call = geometric_asian_option(100, 100, 0.05, 0.2, 1.0, kind="call")
    put = geometric_asian_option(100, 100, 0.05, 0.2, 1.0, kind="put")
    gap = (call - put) - math.exp(-0.05) * (forward - 100)
    print(f"  C - P            = {call - put:>12.10f}")
    print(f"  e^-rT (E[G] - K) = {math.exp(-0.05) * (forward - 100):>12.10f}")
    print(f"  residual         = {gap:.2e}")

    print("\n--- 4. Object wrapper ---")
    for kind in ("call", "put"):
        print(AsianModel(inputs=option, option_type=kind, n_steps=252))
        print()
