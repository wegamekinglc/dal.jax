"""Hand-written path payoffs shared by the tests and the benchmarks' reference setup."""

import math

import jax.numpy as jnp
import numpy as np

from dal_jax import BlackScholes, PathProduct
from dal_jax.script.lower import cspr

SPOT, VOL, RATE, DIV = 100.0, 0.15, 0.05, 0.03
STRIKE, BARRIER, MATURITY = 120.0, 150.0, 3.0
BARRIER_EPS = 0.1


def bs_model(**overrides) -> BlackScholes:
    return BlackScholes(**({"spot": SPOT, "vol": VOL, "rate": RATE, "div": DIV} | overrides))


def call_payoff(params, scenario, ctx):
    return jnp.maximum(scenario.spot[-1] - params["script"]["STRIKE"], 0.0) / scenario.numeraire[-1]


def european_call(maturity: float = MATURITY, strike: float = STRIKE) -> PathProduct:
    """DAL: ``["STRIKE", maturity] / [strike, "call pays MAX(spot() - STRIKE, 0.0)"]``."""
    return PathProduct(timeline=(maturity,), payoff=call_payoff, payoff_names=("call",), script_params={"STRIKE": strike})


def up_and_out_call(timeline: tuple[float, ...], barrier_dates: tuple[int, ...], *, vectorized: bool = False) -> PathProduct:
    """``alive = 1`` today; ``if spot() >= BARRIER:0.1 then alive = 0 end`` on each of
    ``barrier_dates`` (sample indices, repeats allowed); ``call pays alive * MAX(spot() - STRIKE, 0)``
    on the last sample.  Fuzzy mode blends with DAL's call-spread kernel.

    The default evaluates date by date, like DAL's event loop.  ``vectorized``
    computes the same survival product over all dates at once, which keeps the
    traced program small for long schedules."""

    def payoff(params, scenario, ctx):
        barrier, strike = params["script"]["BARRIER"], params["script"]["STRIKE"]
        samples = scenario.samples()
        alive = jnp.ones((), dtype=scenario.spot.dtype)
        for i in barrier_dates:
            x = samples[i].spot - barrier
            hit = cspr(x, BARRIER_EPS) if ctx.fuzzy else (x >= 0.0).astype(x.dtype)
            alive = alive * (1.0 - hit)
        last = samples[-1]
        return alive * jnp.maximum(last.spot - strike, 0.0) / last.numeraire

    multiplicity = np.bincount(np.asarray(barrier_dates, dtype=int), minlength=len(timeline))

    def payoff_vectorized(params, scenario, ctx):
        barrier, strike = params["script"]["BARRIER"], params["script"]["STRIKE"]
        x = scenario.spot - barrier
        survive = 1.0 - (cspr(x, BARRIER_EPS) if ctx.fuzzy else (x >= 0.0).astype(x.dtype))
        factors = jnp.ones_like(survive)
        for k in range(int(multiplicity.max())):
            factors = factors * jnp.where(multiplicity > k, survive, 1.0)
        return jnp.prod(factors) * jnp.maximum(scenario.spot[-1] - strike, 0.0) / scenario.numeraire[-1]

    return PathProduct(
        timeline=timeline,
        payoff=payoff_vectorized if vectorized else payoff,
        payoff_names=("call",),
        script_params={"STRIKE": STRIKE, "BARRIER": BARRIER},
    )


def monthly_barrier_timeline(years: int = 3) -> tuple[float, ...]:
    """Today plus monthly dates from 2022-09-15, as DAL's ``START/END/FREQ: 1M`` schedule
    (ACT/365F year fractions; the last date is today + 365 * years days)."""
    import datetime as dt

    today = dt.date(2022, 9, 15)
    end = today + dt.timedelta(days=365 * years)
    dates = []
    for k in range(1, 12 * years):
        month = today.month - 1 + k
        dates.append(dt.date(today.year + month // 12, month % 12 + 1, today.day))
    dates.append(end)
    return (0.0,) + tuple((d - today).days / 365.0 for d in dates)


def bs_call_price(spot, strike, vol, rate, div, t) -> float:
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    n = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    return spot * math.exp(-div * t) * n(d1) - strike * math.exp(-rate * t) * n(d2)
