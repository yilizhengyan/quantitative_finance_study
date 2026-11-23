# -*- coding: utf-8 -*-
"""
zyquant.model.binomial — Binomial Tree Option Pricing (CRR Model)

Provides 3 functional APIs and 1 object-oriented interface:
    - binomial_option               Core pricing engine (returns 1 of 4 matrices)
    - binomial_option_closed_form   European closed-form solution (fast summation)
    - plot_binomial_tree            Plot the binomial tree nodes
    - BinomialModel                 OOP wrapper (compute once, access anytime)

The core engine (`binomial_option`) supports two methods with identical results:
    - "loop"                        Node-by-node double loop (default, intuitive)
    - "vectorized"                  Node-dimension vectorized (extremely fast for large steps)

`BinomialModel` highlights:
    - Attributes: `model.price`, `model.payoff`, `model.value`, `model.delta` (t=0 states)
    - Pretty Print: `print(model)` (outputs an aligned 6-decimal terminal report)
    - Plotting: `model.plot(output="value")` (reuses parameters, no need to re-pass)
"""

from __future__ import annotations
from dataclasses import dataclass
from math import lgamma
import numpy as np
from .inputs import (ExerciseStyle, OptionInputs, OptionType, Kind, Style, check_inputs, check_steps)

__all__ = ["binomial_option", "binomial_option_closed_form", "BinomialModel", "plot_binomial_tree"]

_OUTPUT = {"price": 0, "payoff": 1, "value": 2, "delta": 3}
_METHOD = ("loop", "vectorized")


def _crr_params(r: float, sigma: float, dt: float, q: float = 0.0):
    '''Calculate CRR model parameters.'''
    u = float(np.exp(sigma * np.sqrt(dt)))
    v = float(1.0 / u)
    p = float((np.exp((r - q) * dt) - v) / (u - v))
    df = float(np.exp(-r * dt))
    return u, v, p, df


def _payoff(price, K, kind: OptionType):
    if kind is OptionType.CALL:
        return np.maximum(price - K, 0.0)
    return np.maximum(K - price, 0.0)


def _check_method(method: str) -> str:
    '''Validate the calculation method.'''
    method = str(method).strip().lower()
    if method not in _METHOD:
        raise ValueError(f"method must be one of {_METHOD}, got {method!r}")
    return method


def _lattice_loop(S, K, r, sigma, T, n, kind: OptionType, style: ExerciseStyle, q: float):
    """Compute the lattice using a node-by-node double loop."""
    dt = T / n
    u, v, p, df = _crr_params(r, sigma, dt, q)
    if not (0.0 < p < 1.0):
        raise ValueError(f"Risk-neutral probability p={p:.6f} is outside (0,1), please check the parameters")

    price = np.zeros((n + 1, n + 1))
    payoff = np.zeros((n + 1, n + 1))
    value = np.zeros((n + 1, n + 1))
    delta = np.zeros((n + 1, n + 1))
    american = style is ExerciseStyle.AMERICAN

    for j in range(n + 1):
        for i in range(j + 1):
            price[i, j] = S * (u ** (j - i)) * (v ** i)
            payoff[i, j] = _payoff(price[i, j], K, kind)

    for j in range(n, -1, -1):
        for i in range(j + 1):
            if j == n:
                value[i, j] = payoff[i, j]
                itm = price[i, j] >= K if kind is OptionType.CALL else price[i, j] <= K
                delta[i, j] = (1.0 if kind is OptionType.CALL else -1.0) if itm else 0.0
            else:
                cont = df * (p * value[i, j + 1] + (1.0 - p) * value[i + 1, j + 1])
                value[i, j] = np.maximum(payoff[i, j], cont) if american else cont
                delta[i, j] = ((value[i, j + 1] - value[i + 1, j + 1]) / (price[i, j + 1] - price[i + 1, j + 1]))

    return price, payoff, value, delta


def _lattice_vector(S, K, r, sigma, T, n, kind: OptionType, style: ExerciseStyle, q: float):
    """Compute the lattice using node-dimension vectorization."""
    dt = T / n
    u, v, p, df = _crr_params(r, sigma, dt, q)
    if not (0.0 < p < 1.0):
        raise ValueError(f"Risk-neutral probability p={p:.6f} is outside (0,1), please check the parameters")

    i = np.arange(n + 1)[:, None]
    j = np.arange(n + 1)[None, :]
    valid = i <= j

    price = np.where(valid, S * (u ** (j - i)) * (v ** i), 0.0)
    payoff = np.where(valid, _payoff(price, K, kind), 0.0)
    value = np.zeros_like(price)
    delta = np.zeros_like(price)

    value[:, n] = payoff[:, n]
    itm = (price[:, n] >= K) if kind is OptionType.CALL else (price[:, n] <= K)
    delta[:, n] = np.where(itm, 1.0 if kind is OptionType.CALL else -1.0, 0.0)

    american = style is ExerciseStyle.AMERICAN
    for col in range(n - 1, -1, -1):
        v_up, v_down = value[:col + 1, col + 1], value[1:col + 2, col + 1]
        cont = df * (p * v_up + (1.0 - p) * v_down)
        if american:
            value[:col + 1, col] = np.maximum(payoff[:col + 1, col], cont)
        else:
            value[:col + 1, col] = cont
        s_up, s_down = price[:col + 1, col + 1], price[1:col + 2, col + 1]
        delta[:col + 1, col] = (v_up - v_down) / (s_up - s_down)

    return price, payoff, value, delta


def _lattice(S, K, r, sigma, T, n, kind: OptionType, style: ExerciseStyle, method: str, q: float = 0.0):
    check_inputs(S, K, r, sigma, T, q)
    n = check_steps(n)
    method = _check_method(method)
    if style not in (ExerciseStyle.EUROPEAN, ExerciseStyle.AMERICAN):
        raise ValueError(f"Binomial tree only supports european / american, got {style.value}")
    if method == "vectorized":
        return _lattice_vector(S, K, r, sigma, T, n, kind, style, q)
    return _lattice_loop(S, K, r, sigma, T, n, kind, style, q)


def binomial_option(S: float, K: float, r: float, sigma: float,
                    T: float, n: int, *, kind: Kind = "call",
                    style: Style = "european",
                    method: str = "loop", output: str = "value",
                    q: float = 0.0) -> np.ndarray:
    """
    Price an option using an n-step binomial tree.

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
    n : int
        Number of steps (positive integer).
    kind : str
        Option type: "call" or "put".
    style : str
        Exercise style: "european" or "american".
    method : str
        Calculation method: "loop" or "vectorized".
    output : str
        Return matrix type: "price", "payoff", "value", or "delta".
        Default is "value", i.e. the option value at each node.
    q : float, optional
        Continuous dividend yield, default is 0.0.

    Returns
    -------
    np.ndarray
        A matrix of shape (n+1, n+1). Column j represents the nodes at step j,
        and row i represents "i" down moves.
    """
    kind = OptionType.parse(kind)
    style = ExerciseStyle.parse(style)
    output = str(output).strip().lower()
    if output not in _OUTPUT:
         raise ValueError(f"output must be one of {list(_OUTPUT)}, got {output!r}")
    return _lattice(S, K, r, sigma, T, n, kind, style, method, q)[_OUTPUT[output]]


def binomial_option_closed_form(S: float, K: float, r: float, sigma: float,
                                T: float, n: int, *, kind: Kind = "call",
                                q: float = 0.0) -> float:
    """
    European closed-form solution for the CRR binomial tree.
    
    Returns
    -------
    float
        Current option price.
    """
    kind = OptionType.parse(kind)
    n = check_steps(n)
    check_inputs(S, K, r, sigma, T, q)
    dt = T / n
    u, v, p, df = _crr_params(r, sigma, dt, q)
    if not (0.0 < p < 1.0):
        raise ValueError(f"Risk-neutral probability p={p:.6f} is outside (0,1), please check the parameters")

    k = np.arange(n + 1, dtype=float)
    log_comb = np.array([(lgamma(n + 1.0) - lgamma(j + 1.0) - lgamma(n - j + 1.0)) for j in range(n + 1)])
    p_c = min(max(p, 1e-300), 1.0 - 1e-16)
    weights = np.exp(log_comb + (n - k) * np.log(p_c) + k * np.log1p(-p_c))

    terminal = S * (u ** (n - k)) * (v ** k)
    return float((df ** n) * np.dot(weights, _payoff(terminal, K, kind)))


def plot_binomial_tree(S: float, K: float, r: float, sigma: float,
                       T: float, n: int, *, kind: Kind = "call",
                       style: Style = "european",
                       output: str = "value", max_steps_plot: int = 5,
                       figsize=(10, 6), q: float = 0.0):
    """Plot the binomial tree nodes. `max_steps_plot` limits the number of steps to draw."""
    import matplotlib.pyplot as plt

    kind = OptionType.parse(kind)
    style = ExerciseStyle.parse(style)
    if output not in _OUTPUT:
        raise ValueError(f"output must be one of {list(_OUTPUT)}, got {output!r}")

    price_all, payoff_all, value_all, delta_all = _lattice(S, K, r, sigma, T, n, kind, style, "loop", q)
    matrix = (price_all, payoff_all, value_all, delta_all)[_OUTPUT[output]]

    name = output.capitalize()
    plot_steps = min(n, max_steps_plot)

    fig, ax = plt.subplots(figsize=figsize)
    title_style = "American" if style is ExerciseStyle.AMERICAN else "European"
    ax.set_title(f"Binomial Tree ({kind.value.title()} - {title_style}) - {name}")

    for j in range(plot_steps + 1):
        for i in range(j + 1):
            y = (j - 2 * i) / 2
            ax.text(j, y, f"S: {price_all[i, j]:.2f}\n{name}: {matrix[i, j]:.4f}",
                    ha="center", va="center", fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))
            if j < plot_steps:
                ax.plot([j, j + 1], [y, y + 0.5], "k-", lw=1)
                ax.plot([j, j + 1], [y, y - 0.5], "k-", lw=1)

    ax.set_xlim(-0.5, plot_steps + 0.5)
    ax.set_ylim(-plot_steps / 2 - 0.6, plot_steps / 2 + 0.6)
    ax.axis("off")
    plt.show()
    return fig


@dataclass
class BinomialModel:

    inputs: OptionInputs
    n: int = 4
    option_type: OptionType = OptionType.CALL
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    method: str = "loop"

    def __post_init__(self) -> None:
        self.option_type = OptionType.parse(self.option_type)
        self.exercise_style = ExerciseStyle.parse(self.exercise_style)
        self.method = _check_method(self.method)
        a = self.inputs
        self.price_tree, self.payoff_tree, self.value_tree, self.delta_tree = _lattice(
            a.S, a.K, a.r, a.sigma, a.T, self.n,
            self.option_type, self.exercise_style, self.method, a.q)

    @property
    def price(self) -> float:
        return float(self.price_tree[0, 0])

    @property
    def payoff(self) -> float:
        return float(self.payoff_tree[0, 0])

    @property
    def value(self) -> float:
        return float(self.value_tree[0, 0])

    @property
    def delta(self) -> float:
        return float(self.delta_tree[0, 0])

    def plot(self, output: str = "value", max_steps_plot: int = 5):
        return plot_binomial_tree(
            self.inputs.S, self.inputs.K, self.inputs.r,
            self.inputs.sigma, self.inputs.T, self.n,
            kind=self.option_type, style=self.exercise_style,
            output=output, max_steps_plot=max_steps_plot, q=self.inputs.q)

    def __str__(self) -> str:
        lines = [
            f"{'─'*54}",
            f"  Binomial Tree ({self.option_type.value.title()} - {self.exercise_style.value.title()})",
            f"{'─'*54}",
            f"  Steps = {self.n:<10d} Method = {self.method}",
            f"{'─'*54}",
            f"  {'Price':22s}  {self.price:>10.6f}",
            f"  {'Payoff':22s}  {self.payoff:>10.6f}",
            f"  {'Value':22s}  {self.value:>10.6f}",
            f"  {'Delta':22s}  {self.delta:>10.6f}",
            f"{'─'*54}"]
        return "\n".join(lines)

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=110)
    
    for name in _OUTPUT:
        print(f"\n--- {name} (Matrix) ---")
        print(binomial_option(100, 100, 0.05, 0.2, 1, 4, output=name))
        plot_binomial_tree(100, 100, 0.05, 0.2, 1, 4, output=name)
    
    my_inputs = OptionInputs(S=100, K=100, T=1, r=0.05, sigma=0.2)
    model = BinomialModel(inputs=my_inputs, n=4)
    print(model)
