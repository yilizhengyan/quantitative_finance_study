# -*- coding: utf-8 -*-
"""
zyquant.model — Option pricing models

One module per model family:
    inputs.py         Shared parameters, enums and validation (all others depend on it)
    binomial.py       CRR tree: pricing, closed form, BinomialModel, tree plot
    blackscholes.py   Black-Scholes / Black-76 / Garman-Kohlhagen, Greeks, implied vol
    asian.py          Geometric-average Asian closed form (Kemna-Vorst), AsianModel
    montecarlo.py     Monte Carlo: European / Asian / Barrier / American, pseudo or Sobol

Every model exposes the same three shapes:
    - a function that returns the raw result (binomial_option, black_scholes, ...)
    - a plotting function where a picture helps (plot_binomial_tree, ...)
    - an OOP wrapper that computes once and keeps the numbers on the object
"""

from .inputs import (BarrierType, ExerciseStyle, Kind, Number, OptionInputs,
                     OptionType, Style, check_inputs, check_steps)
from .binomial import (BinomialModel, binomial_option,
                       binomial_option_closed_form, plot_binomial_tree)
from .blackscholes import (BlackScholes, black_76, black_scholes, bs_greeks,
                           garman_kohlhagen, implied_vol)
from .asian import AsianModel, geometric_asian_forward, geometric_asian_option
from .montecarlo import MonteCarloModel, montecarlo_option, plot_montecarlo_paths

__all__ = [
    # inputs
    "Kind", "Style", "Number",
    "OptionInputs", "OptionType", "ExerciseStyle", "BarrierType",
    "check_inputs", "check_steps",
    # binomial tree
    "binomial_option", "binomial_option_closed_form", "BinomialModel", "plot_binomial_tree",
    # closed form
    "black_scholes", "black_76", "garman_kohlhagen", "bs_greeks", "implied_vol",
    "BlackScholes", "geometric_asian_option", "geometric_asian_forward", "AsianModel",
    # monte carlo
    "MonteCarloModel", "montecarlo_option", "plot_montecarlo_paths"]
