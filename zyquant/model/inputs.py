# -*- coding: utf-8 -*-
"""
zyquant.model.inputs — Shared input structures and validation for all models

Tolerant boundaries.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Union
import numpy as np

__all__ = [
    "Kind", "Style", "Number",
    "OptionType", "ExerciseStyle", "BarrierType", "OptionInputs",
    "check_inputs", "check_steps"]

Kind = Union[str, "OptionType"]
Style = Union[str, "ExerciseStyle"]
Number = Union[float, int, np.ndarray]


class OptionType(str, Enum):
    """Call / Put option type."""

    CALL = "call"
    PUT = "put"

    @classmethod
    def parse(cls, value: Kind) -> "OptionType":
        if isinstance(value, OptionType):
            return value
        key = str(value).strip().lower()
        try:
            return _OPTION_ALIASES[key]
        except KeyError:
            raise ValueError(f"Invalid option type: {value!r}, not in {sorted(_OPTION_ALIASES)}") from None


class ExerciseStyle(str, Enum):
    """
    Exercise style.
    
    EUROPEAN  European: Exercise only at maturity.
    AMERICAN  American: Exercise at any time before maturity.
    ASIAN     Asian: Payoff based on average price, not terminal price.
    BARRIER   Barrier: Contract activates or expires upon hitting a barrier.
    """

    EUROPEAN = "european"
    AMERICAN = "american"
    ASIAN = "asian"
    BARRIER = "barrier"

    @classmethod
    def parse(cls, value: Style) -> "ExerciseStyle":
        if isinstance(value, ExerciseStyle):
            return value
        key = str(value).strip().lower()
        try:
            return _STYLE_ALIASES[key]
        except KeyError:
            raise ValueError(f"Invalid Exercise Style: {value!r}, not in {sorted(_STYLE_ALIASES)}") from None


class BarrierType(str, Enum):
    """
    Barrier types:

    UP_AND_OUT    Up-and-Out: Knocked out if price breaches barrier from below.
    UP_AND_IN     Up-and-In: Activated if price breaches barrier from below.
    DOWN_AND_OUT  Down-and-Out: Knocked out if price breaches barrier from above.
    DOWN_AND_IN   Down-and-In: Activated if price breaches barrier from above.
    """

    UP_AND_OUT = "up_and_out"
    UP_AND_IN = "up_and_in"
    DOWN_AND_OUT = "down_and_out"
    DOWN_AND_IN = "down_and_in"

    @classmethod
    def parse(cls, value) -> "BarrierType":
        if isinstance(value, BarrierType):
            return value
        key = str(value).strip().lower()
        try:
            return _BARRIER_ALIASES[key]
        except KeyError:
            raise ValueError(f"Invalid Barrier Type: {value!r}, not in {sorted(_BARRIER_ALIASES)}") from None


_OPTION_ALIASES = {
    "call": OptionType.CALL, "c": OptionType.CALL, "ce": OptionType.CALL,
    "涨": OptionType.CALL, "看涨": OptionType.CALL,
    "put": OptionType.PUT, "p": OptionType.PUT, "pe": OptionType.PUT,
    "跌": OptionType.PUT, "看跌": OptionType.PUT}

_STYLE_ALIASES = {
    "asian": ExerciseStyle.ASIAN, "average": ExerciseStyle.ASIAN,
    "亚式": ExerciseStyle.ASIAN, "亚": ExerciseStyle.ASIAN,
    "barrier": ExerciseStyle.BARRIER, "障碍": ExerciseStyle.BARRIER,
    "敲入敲出": ExerciseStyle.BARRIER,
    "european": ExerciseStyle.EUROPEAN, "eu": ExerciseStyle.EUROPEAN,
    "欧式": ExerciseStyle.EUROPEAN, "欧": ExerciseStyle.EUROPEAN,
    "american": ExerciseStyle.AMERICAN, "am": ExerciseStyle.AMERICAN,
    "美式": ExerciseStyle.AMERICAN, "美": ExerciseStyle.AMERICAN}

_BARRIER_ALIASES = {
    "up_and_out": BarrierType.UP_AND_OUT, "uo": BarrierType.UP_AND_OUT,
    "向上敲出": BarrierType.UP_AND_OUT,
    "up_and_in": BarrierType.UP_AND_IN, "ui": BarrierType.UP_AND_IN,
    "向上敲入": BarrierType.UP_AND_IN,
    "down_and_out": BarrierType.DOWN_AND_OUT, "do": BarrierType.DOWN_AND_OUT,
    "向下敲出": BarrierType.DOWN_AND_OUT,
    "down_and_in": BarrierType.DOWN_AND_IN, "di": BarrierType.DOWN_AND_IN,
    "向下敲入": BarrierType.DOWN_AND_IN}


def check_inputs(S, K, r, sigma, T, q=0.0) -> None:
    """Common validation, read-only, no return value."""
    try:
        if not (S > 0):
            raise ValueError(f"S must be strictly positive, got {S!r}")
        if not (K > 0):
            raise ValueError(f"K must be strictly positive, got {K!r}")
        if not (sigma > 0):
            raise ValueError(f"sigma must be strictly positive, got {sigma!r}")
        if not (T > 0):
            raise ValueError(f"T must be strictly positive, got {T!r}")
        if r < 0:
            raise ValueError(f"r must be non-negative, got {r!r}")
        if q < 0:
            raise ValueError(f"q must be non-negative, got {q!r}")
    except TypeError:
        raise TypeError("All parameters (S, K, r, sigma, T, q) must be numbers.")


def check_steps(n) -> int:
    """Number of steps must be a positive integer."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"n must be an int, got {type(n).__name__} ({n!r})")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    return int(n)


@dataclass(slots=True)
class OptionInputs:
    """Option parameters."""

    S: float
    K: float
    T: float                                        # Time to maturity in years (e.g., 0.25 for 3 months)
    r: float                                        # Risk-free rate in decimal (e.g., 0.05 for 5%)
    sigma: float                                    # Implied volatility in decimal (e.g., 0.20 for 20%)
    q: float = 0.0
    barrier: float | None = None                    # Barrier price; only for barrier options
    barrier_type: BarrierType | str | None = None   # Accepts string values
    rebate: float = 0.0                             # Rebate received when knocked out

    def __post_init__(self) -> None:
        check_inputs(self.S, self.K, self.r, self.sigma, self.T, self.q)
        
        if (self.barrier is None) != (self.barrier_type is None):
            raise ValueError("Both 'barrier' and 'barrier_type' must be provided together, or neither.")
            
        if self.barrier is not None:
            try:
                self.barrier = float(self.barrier)
            except (TypeError, ValueError):
                raise TypeError(f"barrier must be a number, got {type(self.barrier).__name__} ({self.barrier!r})") from None
                
            if self.barrier <= 0:
                raise ValueError(f"barrier must be strictly positive, got {self.barrier!r}")
                
            self.barrier_type = BarrierType.parse(self.barrier_type)
            
        try:
            self.rebate = float(self.rebate)
        except (TypeError, ValueError):
            raise TypeError(f"rebate must be a number, got {type(self.rebate).__name__} ({self.rebate!r})") from None
            
        if self.rebate < 0:
            raise ValueError(f"rebate must be non-negative, got {self.rebate!r}")