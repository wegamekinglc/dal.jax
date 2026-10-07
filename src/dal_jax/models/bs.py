"""Black-Scholes model, a port of DAL's ``BlackScholes_``.

Log-spot is accumulated from per-step drifts ``(r - q - vol^2 / 2) dt`` and
standard deviations ``vol sqrt(dt)`` precomputed in ``init``; the numeraire is
``exp(r t)`` and discount factors are ``exp(-r (T - t))``.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from dal_jax.errors import InvalidModelParameter
from dal_jax.models.base import ModelParams, SampleDef, Scenario, validate_timeline


def _steps_from_today(times: tuple[float, ...]) -> tuple[float, ...]:
    """Step lengths of DAL's model grid: today, then every positive product time."""
    grid = [0.0] + [t for t in times if t > 0.0]
    return tuple(right - left for left, right in zip(grid, grid[1:]))


@dataclass(frozen=True, slots=True)
class BSPlan:
    times: tuple[float, ...]  # product timeline
    today_on_timeline: bool
    dts: tuple[float, ...]  # model steps between successive positive times, from 0
    numeraire: tuple[bool, ...]
    discount_mats: tuple[tuple[float, ...], ...]
    max_observations: int
    max_discounts: int


class BSState(NamedTuple):
    spot: Array
    log_spot: Array
    drifts: Array  # [n_steps]
    stds: Array  # [n_steps]
    numeraires: Array  # [n_samples]
    discounts: Array  # [n_samples, max_discounts]


@dataclass(frozen=True, slots=True, kw_only=True)
class BlackScholes:
    spot: float
    vol: float
    rate: float = 0.0
    div: float = 0.0

    param_labels: ClassVar[tuple[str, ...]] = ("spot", "vol", "rate", "div")
    n_factors: ClassVar[int] = 1
    numeraire_is_deterministic: ClassVar[bool] = True
    supports_discount_factors: ClassVar[bool] = True
    max_observed_indices: ClassVar[int] = 1
    max_output_slots_per_sample: ClassVar[int] = 1

    def __post_init__(self) -> None:
        self.validate_params(self.default_params())

    @property
    def supports_bb(self) -> bool:
        return self.n_factors == 1

    def default_params(self) -> dict[str, Array]:
        return {label: jnp.asarray(getattr(self, label), dtype=jnp.float64) for label in self.param_labels}

    def validate_params(self, params: ModelParams) -> None:
        spot, vol, rate, div = (float(params[label]) for label in self.param_labels)
        if not (math.isfinite(spot) and spot > 0.0):
            raise InvalidModelParameter("spot must be finite and positive")
        if not (math.isfinite(vol) and vol >= 0.0):
            raise InvalidModelParameter("vol must be finite and nonnegative")
        if not math.isfinite(rate):
            raise InvalidModelParameter("rate must be finite")
        if not math.isfinite(div):
            raise InvalidModelParameter("div must be finite")

    def allocate(self, timeline: Sequence[float], sample_defs: Sequence[SampleDef]) -> BSPlan:
        times = tuple(float(t) for t in timeline)
        validate_timeline(self, times, sample_defs)
        return BSPlan(
            times=times,
            today_on_timeline=times[0] == 0.0,
            dts=_steps_from_today(times),
            numeraire=tuple(d.numeraire for d in sample_defs),
            discount_mats=tuple(tuple(float(m) for m in d.discount_mats) for d in sample_defs),
            max_observations=max(len(d.index_names) for d in sample_defs),
            max_discounts=max(len(d.discount_mats) for d in sample_defs),
        )

    def sim_dim(self, plan: BSPlan) -> int:
        return len(plan.dts)

    def init(self, params: Mapping[str, Array], plan: BSPlan) -> BSState:
        spot, vol, rate, div = (jnp.asarray(params[label]) for label in self.param_labels)
        mu = rate - div
        dts = jnp.asarray(plan.dts, dtype=jnp.float64)
        times = jnp.asarray(plan.times, dtype=jnp.float64)
        numeraires = jnp.where(jnp.asarray(plan.numeraire), jnp.exp(rate * times), 1.0)
        mats = np.ones((len(plan.times), plan.max_discounts))
        has_mat = np.zeros_like(mats, dtype=bool)
        for i, row in enumerate(plan.discount_mats):
            mats[i, : len(row)] = row
            has_mat[i, : len(row)] = True
        discounts = jnp.where(has_mat, jnp.exp(-rate * (jnp.asarray(mats) - times[:, None])), 1.0)
        return BSState(
            spot=spot,
            log_spot=jnp.log(spot),
            drifts=(mu - 0.5 * vol * vol) * dts,
            stds=vol * jnp.sqrt(dts),
            numeraires=numeraires,
            discounts=discounts,
        )

    def generate(self, state: BSState, plan: BSPlan, normals: Array) -> Scenario:
        if not plan.dts:
            spots = state.spot[None]
        else:
            increments = state.drifts + state.stds * normals
            log_spots = jnp.cumsum(jnp.concatenate([state.log_spot[None], increments]))[1:]
            spots = jnp.exp(log_spots)
            if plan.today_on_timeline:
                spots = jnp.concatenate([state.spot[None], spots])
        observations = jnp.broadcast_to(spots[:, None], (len(plan.times), plan.max_observations))
        return Scenario(spot=spots, numeraire=state.numeraires, observations=observations, discounts=state.discounts)
