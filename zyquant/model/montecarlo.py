# -*- coding: utf-8 -*-
"""
zyquant.model.montecarlo — Monte Carlo Option Pricing

Provides 2 functional APIs and 1 object-oriented interface:
    - montecarlo_option       Core pricing engine (returns price, error and CI)
    - plot_montecarlo_paths   Plot simulated paths and the terminal distribution
    - MonteCarloModel         OOP wrapper (compute once, access anytime)

Exercise styles supported by the core engine:
    - "european"            Payoff settled at maturity
    - "asian"               Payoff settled on the average price
    - "barrier"             Knocked in or out when a barrier is touched
    - "american"            Longstaff-Schwartz least-squares early exercise

Samplers:
    - "pseudo"              Mersenne Twister with antithetic variates
    - "sobol"               Scrambled Sobol; n_paths is rounded up to a power of 2
                            and antithetic variates are switched off

MonteCarloModel highlights:
    - Attributes: model.price, model.std_error, model.ci_low, model.ci_high
    - Pretty Print: print(model) (aligned six-decimal report)
    - Plotting: model.plot() (reuses parameters, no need to re-pass)
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional
import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from .inputs import (BarrierType, ExerciseStyle, Kind, OptionInputs, OptionType,
                     Style, check_inputs)

__all__ = ["MonteCarloModel", "montecarlo_option", "plot_montecarlo_paths"]

_SAMPLERS = ("pseudo", "sobol")
_PAYOFFS = ("vanilla", "asian", "barrier")
_AVERAGES = ("arithmetic", "geometric")
_Z95 = 1.959963984540054
_BETA = 0.5826                      # Broadie-Glasserman-Kou continuity constant


def _next_power_of_two(n: int) -> int:
    '''Smallest power of two that is >= n.'''
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))


def _intrinsic(prices, K: float, kind: OptionType):
    '''Undiscounted intrinsic value.'''
    if kind is OptionType.CALL:
        return np.maximum(prices - K, 0.0)
    return np.maximum(K - prices, 0.0)


def _vanilla(paths, K: float, kind: OptionType):
    '''Payoff from the terminal price.'''
    return _intrinsic(paths[:, -1], K, kind)


def _asian(paths, K: float, kind: OptionType, average: str):
    '''Payoff from the average price, skipping the t=0 column.'''
    if average == "geometric":
        return _intrinsic(np.exp(np.log(paths[:, 1:]).mean(axis=1)), K, kind)
    return _intrinsic(paths[:, 1:].mean(axis=1), K, kind)


def _barrier(paths, K: float, kind: OptionType, barrier: float,
             barrier_type: BarrierType, rebate: float, sigma: float, dt: float):
    '''
    Barrier payoff with the Broadie-Glasserman-Kou continuity correction.

    Discrete monitoring understates the touch probability, so BGK shifts the
    effective barrier:
        up barrier    B_eff = B * exp(+beta * sigma * sqrt(dt))
        down barrier  B_eff = B * exp(-beta * sigma * sqrt(dt))

    Out options pay the rebate once touched, in options pay the vanilla payoff.
    '''
    shift = _BETA * sigma * np.sqrt(dt)
    vanilla = _intrinsic(paths[:, -1], K, kind)

    if barrier_type is BarrierType.UP_AND_OUT:
        crossed = np.max(paths, axis=1) >= barrier * np.exp(+shift)
        return np.where(crossed, rebate, vanilla)
    if barrier_type is BarrierType.UP_AND_IN:
        crossed = np.max(paths, axis=1) >= barrier * np.exp(+shift)
        return np.where(crossed, vanilla, rebate)
    if barrier_type is BarrierType.DOWN_AND_OUT:
        crossed = np.min(paths, axis=1) <= barrier * np.exp(-shift)
        return np.where(crossed, rebate, vanilla)
    crossed = np.min(paths, axis=1) <= barrier * np.exp(-shift)
    return np.where(crossed, vanilla, rebate)


def _lsm(paths, K: float, kind: OptionType, r: float, dt: float):
    '''
    Longstaff-Schwartz least-squares Monte Carlo for American options.

    Walks backwards from maturity, regressing the discounted continuation value
    on {1, x, x^2} with x = S/K, and only over in-the-money paths. Returns the
    value already discounted to t=0, so the caller must not discount again.
    '''
    disc = np.exp(-r * dt)
    cash = _intrinsic(paths[:, -1], K, kind)

    for step in range(paths.shape[1] - 2, 0, -1):
        cash *= disc
        spot = paths[:, step]
        intrinsic = _intrinsic(spot, K, kind)
        in_the_money = intrinsic > 0
        if not in_the_money.any():
            continue
        x = spot[in_the_money] / K
        basis = np.column_stack([np.ones_like(x), x, x * x, x ** 3])
        coef, *_ = np.linalg.lstsq(basis, cash[in_the_money], rcond=None)
        exercise = intrinsic[in_the_money] > basis @ coef
        index = np.flatnonzero(in_the_money)[exercise]
        cash[index] = intrinsic[index]
    return cash * disc


def _normals(n_paths: int, n_steps: int, sampler: str, seed: Optional[int],
             antithetic: bool):
    '''
    Standard normal matrix of shape (n_paths, n_steps).

    Sobol draws a scrambled sequence in n_steps dimensions and maps it through
    norm.ppf; the uniform draws are clipped to the open unit interval so ppf
    never sees 0 or 1.
    '''
    if sampler == "sobol":
        engine = Sobol(d=n_steps, scramble=True, seed=seed)
        uniform = np.clip(engine.random(n_paths), 1e-10, 1.0 - 1e-10)
        return norm.ppf(uniform)

    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2 if antithetic else n_paths
    z = rng.standard_normal((half, n_steps))
    if not antithetic:
        return z
    return np.vstack([z, -z])[: n_paths]


def _simulate(inputs: OptionInputs, n_paths: int, n_steps: int, sampler: str,
              seed: Optional[int], antithetic: bool):
    '''Simulate (n_paths, n_steps+1) GBM paths; column 0 is the spot price.'''
    check_inputs(inputs.S, inputs.K, inputs.r, inputs.sigma, inputs.T, inputs.q)

    dt = inputs.T / n_steps
    drift = (inputs.r - inputs.q - 0.5 * inputs.sigma ** 2) * dt
    vol = inputs.sigma * np.sqrt(dt)

    z = _normals(n_paths, n_steps, sampler, seed, antithetic)
    log_path = np.cumsum(drift + vol * z, axis=1)
    zeros = np.zeros((n_paths, 1))
    return inputs.S * np.exp(np.hstack([zeros, log_path]))


def _discounted(paths, inputs: OptionInputs, n_steps: int, kind: OptionType,
                style: ExerciseStyle, payoff: str, average: str):
    '''Per-path payoff, discounted to t=0.'''
    a = inputs
    dt = a.T / n_steps

    if payoff == "asian":
        value = _asian(paths, a.K, kind, average)
    elif payoff == "barrier":
        value = _barrier(paths, a.K, kind, a.barrier, a.barrier_type,
                         float(a.rebate or 0.0), a.sigma, dt)
    elif style is ExerciseStyle.AMERICAN:
        value = _lsm(paths, a.K, kind, a.r, dt)
    else:
        value = _vanilla(paths, a.K, kind)

    # American values are already discounted to t=0 inside _lsm
    if style is ExerciseStyle.AMERICAN:
        return value
    return value * np.exp(-a.r * a.T)


def _summary(values, n_paths: int):
    '''Collapse per-path values into (price, std_error, ci_low, ci_high).'''
    price = float(values.mean())
    error = float(values.std(ddof=1) / np.sqrt(n_paths))
    return price, error, price - _Z95 * error, price + _Z95 * error


def _normalize(inputs: OptionInputs, n_paths: int, n_steps: int,
               option_type, exercise_style, payoff: str, average: str,
               sampler: str, antithetic: bool):
    '''Validate and normalise every setting; returns them in canonical order.'''
    if n_paths < 2:
        raise ValueError(f"n_paths must be at least 2, got {n_paths!r}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps!r}")

    sampler = str(sampler).strip().lower()
    if sampler not in _SAMPLERS:
        raise ValueError(f"sampler must be one of {list(_SAMPLERS)}, got {sampler!r}")
    average = str(average).strip().lower()
    if average not in _AVERAGES:
        raise ValueError(f"average must be one of {list(_AVERAGES)}, got {average!r}")

    kind = OptionType.parse(option_type)
    style = ExerciseStyle.parse(exercise_style)
    payoff = _resolve_payoff(style, payoff, inputs)

    if sampler == "sobol":
        n_paths = _next_power_of_two(n_paths)
    antithetic = bool(antithetic) and (sampler == "pseudo")
    return kind, style, payoff, average, sampler, n_paths, int(n_steps), antithetic


def _resolve_payoff(style: ExerciseStyle, payoff: str, inputs: OptionInputs) -> str:
    '''Exercise style decides the payoff; reject combinations we cannot price.'''
    if style is ExerciseStyle.ASIAN:
        payoff = "asian"
    elif style is ExerciseStyle.BARRIER:
        payoff = "barrier"
    payoff = str(payoff).strip().lower()
    if payoff not in _PAYOFFS:
        raise ValueError(f"payoff must be one of {list(_PAYOFFS)}, got {payoff!r}")
    if payoff == "asian" and style is ExerciseStyle.AMERICAN:
        raise ValueError(
            "Asian options are European only; an American Asian needs richer LSM basis")
    if payoff == "barrier":
        if style is ExerciseStyle.AMERICAN:
            raise ValueError(
                "Barrier options are European only; "
                "an American barrier needs node-level exercise")
        if inputs.barrier is None or inputs.barrier_type is None:
            raise ValueError("Barrier options need both barrier (level) and barrier_type")
    return payoff


def _settle(inputs: OptionInputs, n_paths: int, n_steps: int, kind: OptionType,
            style: ExerciseStyle, payoff: str, average: str, sampler: str,
            seed: Optional[int], antithetic: bool):
    '''Simulate, price and summarise in one go: (price, se, ci_low, ci_high).'''
    paths = _simulate(inputs, n_paths, n_steps, sampler, seed, antithetic)
    values = _discounted(paths, inputs, n_steps, kind, style, payoff, average)
    return _summary(values, n_paths)


def montecarlo_option(S: float, K: float, r: float, sigma: float, T: float, *,
                      kind: Kind = "call",
                      style: Style = "european",
                      payoff: str = "vanilla",
                      n_paths: int = 100_000, n_steps: int = 252,
                      seed: Optional[int] = 42, q: float = 0.0,
                      barrier: float | None = None, barrier_type=None,
                      rebate: float = 0.0,
                      average: str = "arithmetic",
                      sampler: str = "pseudo") -> tuple:
    """
    Price an option by Monte Carlo.

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
    style : str
        Exercise style: "european", "asian", "barrier" or "american".
    payoff : str, optional
        Settlement convention; normally inferred from style.
    n_paths, n_steps : int, optional
        Number of paths and time steps, default 100_000 and 252.
    seed : int, optional
        Random seed, default 42 (used by both samplers).
    q : float, optional
        Continuous dividend yield, default 0.0.
    barrier, barrier_type, rebate
        Barrier level, barrier type and cash rebate (barrier options only).
    average : str, optional
        Asian averaging convention: "arithmetic" or "geometric".
    sampler : str, optional
        Sampling method: "pseudo" or "sobol".

    Returns
    -------
    tuple of float
        (price, std_error, ci_low, ci_high), in that order.
    """
    inputs = OptionInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q,
                          barrier=barrier, barrier_type=barrier_type, rebate=rebate)
    settings = _normalize(inputs, n_paths, n_steps, kind, style, payoff, average,
                          sampler, True)
    kind, style, payoff, average, sampler, n_paths, n_steps, antithetic = settings
    return _settle(inputs, n_paths, n_steps, kind, style, payoff, average,
                   sampler, seed, antithetic)


def plot_montecarlo_paths(S: float, K: float, r: float, sigma: float, T: float, *,
                          kind: Kind = "call", style: Style = "european",
                          payoff: str = "vanilla", q: float = 0.0,
                          barrier: float | None = None, barrier_type=None,
                          rebate: float = 0.0, average: str = "arithmetic",
                          n_paths: int = 200, n_steps: int = 252,
                          seed: Optional[int] = 42, sampler: str = "pseudo",
                          max_paths_plot: int = 60, show_band: bool = True,
                          show_mean: bool = True, show_terminal: bool = True,
                          figsize=(11, 5.2)):
    """
    Plot simulated paths on the left and the terminal distribution on the right.

    Statistics use all n_paths draws; only max_paths_plot of them are drawn,
    otherwise a thousand lines smear into a single blob.

    Parameters
    ----------
    n_paths : int, optional
        Paths simulated for the statistics.
    max_paths_plot : int, optional
        Paths actually drawn, default 60.
    show_band : bool, optional
        Draw the 5%-95% quantile band.
    show_mean : bool, optional
        Draw the mean path.
    show_terminal : bool, optional
        Add the terminal price histogram on the right.
    sampler : str, optional
        Sampling method: "pseudo" or "sobol".

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    inputs = OptionInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q,
                          barrier=barrier, barrier_type=barrier_type, rebate=rebate)
    settings = _normalize(inputs, n_paths, n_steps, kind, style, payoff, average,
                          sampler, True)
    kind, style, payoff, average, sampler, n_paths, n_steps, antithetic = settings

    paths = _simulate(inputs, n_paths, n_steps, sampler, seed, antithetic)
    values = _discounted(paths, inputs, n_steps, kind, style, payoff, average)
    price, error, ci_low, ci_high = _summary(values, n_paths)

    t = np.linspace(0.0, T, n_steps + 1)
    n_draw = min(int(max_paths_plot), n_paths)
    show = paths[:n_draw]

    if show_terminal:
        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                 gridspec_kw={"width_ratios": [2.2, 1]})
        ax_path, ax_hist = axes[0], axes[1]
    else:
        fig, ax_path = plt.subplots(1, 1, figsize=figsize)
        ax_hist = None

    term = paths[:, -1]
    scale = plt.Normalize(term.min(), term.max())
    cmap = plt.get_cmap("coolwarm")
    segs = np.stack([np.column_stack([t, row]) for row in show])
    ax_path.add_collection(LineCollection(segs, colors=cmap(scale(term[:n_draw])),
                                          linewidths=0.8, alpha=0.55))

    if show_band:
        lo, hi = np.percentile(paths, 5, axis=0), np.percentile(paths, 95, axis=0)
        ax_path.fill_between(t, lo, hi, color="steelblue", alpha=0.12,
                             label="5%-95% band")
    if show_mean:
        ax_path.plot(t, paths.mean(axis=0), color="black", lw=1.6, label="mean path")
    ax_path.axhline(K, color="crimson", ls="--", lw=1.2, label=f"strike K={K:g}")
    if barrier is not None:
        ax_path.axhline(float(barrier), color="purple", ls="-.", lw=1.4,
                        label=f"barrier B={float(barrier):g}")

    ax_path.set_xlim(0, T)
    ax_path.set_ylim(max(0.0, paths.min() * 0.92), paths.max() * 1.05)
    ax_path.set_xlabel("Time (years)")
    ax_path.set_ylabel("Underlying price")
    ax_path.set_title(f"Monte Carlo paths: {n_draw}/{n_paths} paths x {n_steps} steps")
    ax_path.legend(loc="upper left", fontsize=9, framealpha=0.9)

    if show_terminal:
        ax_hist.hist(term, bins=30, color="steelblue", alpha=0.75, edgecolor="white")
        ax_hist.axvline(K, color="crimson", ls="--", lw=1.2)
        ax_hist.axvline(term.mean(), color="black", lw=1.2)
        ax_hist.set_xlabel("Terminal price S_T")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"Terminal distribution (mean {term.mean():.2f})")

    caption = f"{payoff}  price={price:.6f}  se={error:.6f}  ({n_paths:,} paths)"
    fig.suptitle(caption, fontsize=8.5, color="dimgray", y=0.01, va="bottom")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    plt.show()
    return fig


@dataclass
class MonteCarloModel:
    '''Monte Carlo engine: parameters in, price and statistics on the object.'''

    inputs: OptionInputs
    n_paths: int = 100_000
    n_steps: int = 252
    option_type: OptionType = OptionType.CALL
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    payoff: str = "vanilla"
    average: str = "arithmetic"
    sampler: str = "pseudo"
    seed: Optional[int] = 42
    antithetic: bool = True

    def __post_init__(self) -> None:
        (self.option_type, self.exercise_style, self.payoff, self.average,
         self.sampler, self.n_paths, self.n_steps, self.antithetic) = _normalize(
            self.inputs, self.n_paths, self.n_steps, self.option_type,
            self.exercise_style, self.payoff, self.average, self.sampler,
            self.antithetic)

        self.price, self.std_error, self.ci_low, self.ci_high = _settle(
            self.inputs, self.n_paths, self.n_steps, self.option_type,
            self.exercise_style, self.payoff, self.average, self.sampler,
            self.seed, self.antithetic)

    def simulate(self, n_paths: int = 200):
        '''Simulate a fresh (n_paths, n_steps+1) path matrix for inspection.'''
        return _simulate(self.inputs, n_paths, self.n_steps, self.sampler,
                         self.seed, self.antithetic)

    def plot(self, n_paths: int = 200, max_paths_plot: int = 60, **kwargs):
        '''Plot fresh paths with the same parameters, no need to re-pass them.'''
        a = self.inputs
        return plot_montecarlo_paths(
            a.S, a.K, a.r, a.sigma, a.T,
            kind=self.option_type, style=self.exercise_style,
            q=a.q, average=self.average, sampler=self.sampler, seed=self.seed,
            n_paths=n_paths, n_steps=self.n_steps, max_paths_plot=max_paths_plot,
            barrier=a.barrier, barrier_type=a.barrier_type, rebate=a.rebate,
            **kwargs)

    def __str__(self) -> str:
        label = self.payoff if self.payoff != "vanilla" else self.exercise_style.value
        lines = [
            f"{'─' * 54}",
            f"  Monte Carlo ({self.option_type.value.title()} - {label.title()})",
            f"{'─' * 54}",
            f"  Paths = {self.n_paths:<10,d}  Steps = {self.n_steps}",
            f"  Sampler = {self.sampler}",
            f"{'─' * 54}",
            f"  {'Price':22s}  {self.price:>10.6f}",
            f"  {'Std Error':22s}  {self.std_error:>10.6f}",
            f"  {'95% CI Low':22s}  {self.ci_low:>10.6f}",
            f"  {'95% CI High':22s}  {self.ci_high:>10.6f}",
            f"{'─' * 54}"]
        return "\n".join(lines)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=110)

    print("--- 1. Function API: three styles ---")
    for style, extra in (("european", {}), ("asian", {}), ("american", {"kind": "put"})):
        price, error, ci_low, ci_high = montecarlo_option(
            100, 100, 0.05, 0.2, 1.0, style=style, n_paths=50_000, seed=7, **extra)
        print(f"  {style:<10} price={price:>9.6f}  se={error:.6f}  "
              f"CI=[{ci_low:.6f}, {ci_high:.6f}]")

    print("\n--- 2. Sampler comparison (European call, theoretical 10.450584) ---")
    for sampler in _SAMPLERS:
        price, error, _, _ = montecarlo_option(100, 100, 0.05, 0.2, 1.0,
                                               n_paths=65_536, sampler=sampler, seed=7)
        print(f"  {sampler:<7} {price:>10.6f}  error {abs(price - 10.450584):>9.6f}")

    print("\n--- 3. Object wrapper ---")
    option = OptionInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    model = MonteCarloModel(option, n_paths=50_000, seed=7)
    print(model)

    print("\n--- 4. Attributes ---")
    for name in ("price", "std_error", "ci_low", "ci_high"):
        print(f"  model.{name:<10} = {getattr(model, name):.6f}")
