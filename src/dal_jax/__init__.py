"""Monte Carlo derivatives pricing with JAX, re-implementing DAL's simulation engine.

Importing ``dal_jax`` enables ``jax_enable_x64``: DAL computes in double
precision and parity with it needs float64 available.
"""

from dal_jax import config

config.enable_x64()

from dal_jax.errors import DalError  # noqa: E402
from dal_jax.mc import EvalContext, MonteCarloEngine, MonteCarloSettings, PathProduct  # noqa: E402
from dal_jax.models import BlackScholes, CorrelatedBlackScholes, LocalVol, LocalVolSurface, SampleDef, Scenario  # noqa: E402
from dal_jax.script.fixings import FixingSnapshot, TodayFixingPolicy, ValuationSettings, set_global_fixings  # noqa: E402
from dal_jax.script.preparation import PreparedProduct, prepare  # noqa: E402

__version__ = "0.0.1"

__all__ = [
    "BlackScholes",
    "CorrelatedBlackScholes",
    "LocalVol",
    "LocalVolSurface",
    "DalError",
    "EvalContext",
    "FixingSnapshot",
    "TodayFixingPolicy",
    "ValuationSettings",
    "set_global_fixings",
    "MonteCarloEngine",
    "MonteCarloSettings",
    "PathProduct",
    "PreparedProduct",
    "SampleDef",
    "Scenario",
    "config",
    "prepare",
]
