"""DAL's correlated multi-equity Black-Scholes with step-major normal factors."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from dal_jax.errors import (
    DuplicateModelIndex,
    InvalidCorrelation,
    InvalidModelIndex,
    InvalidModelParameter,
    UnsupportedModelObservation,
)
from dal_jax.index import parse_index
from dal_jax.models.base import SampleDef, Scenario
from dal_jax.models.bs import BlackScholes, BSPlan, deterministic_outputs
from dal_jax.strings import ci_eq, ci_key


def plain_equity(name):
    index = parse_index(name)
    if index.kind != "EQ" or not index.name.endswith("]"):
        raise InvalidModelIndex(f"expected an ordinary EQ index: {name}")
    return index.name


def _validate_correlations(values, size):
    if values.shape != (size, size) or not np.isfinite(values).all():
        raise InvalidCorrelation("dimensions must match the assets and entries must be finite")
    symmetric = np.allclose(values, values.T, rtol=0.0, atol=1e-12)
    unit_diagonal = np.allclose(np.diag(values), 1.0, rtol=0.0, atol=1e-12)
    if not symmetric or not unit_diagonal:
        raise InvalidCorrelation("matrix must be symmetric with unit diagonal")


def _correlation_factor(matrix, size):
    values = np.asarray(matrix, dtype=np.float64)
    _validate_correlations(values, size)
    try:
        lower = np.linalg.cholesky(values)
    except np.linalg.LinAlgError as error:
        raise InvalidCorrelation("matrix must be positive definite") from error
    if np.min(np.diag(lower) ** 2) <= 1e-14:
        raise InvalidCorrelation("matrix must be positive definite")
    return tuple(tuple(float(x) for x in row) for row in lower)


def _asset_names(indices):
    names = tuple(plain_equity(name) for name in indices)
    if not names:
        raise InvalidModelParameter("correlated BS requires at least one asset")
    if len({ci_key(name) for name in names}) != len(names):
        raise DuplicateModelIndex("asset indices must be unique")
    return names


@dataclass(frozen=True, slots=True)
class CorrelatedBSPlan:
    base: BSPlan
    observation_slots: tuple[tuple[int, ...], ...]
    lower: tuple[tuple[float, ...], ...]


class CorrelatedBSState(NamedTuple):
    spots: Array
    log_spots: Array
    drifts: Array
    stds: Array
    numeraires: Array
    discounts: Array


@dataclass(frozen=True, slots=True, kw_only=True)
class CorrelatedBlackScholes:
    indices: tuple[str, ...]
    spots: tuple[float, ...]
    vols: tuple[float, ...]
    divs: tuple[float, ...]
    rate: float
    correlations: tuple[tuple[float, ...], ...]

    supports_bb = True
    numeraire_is_deterministic = True
    supports_discount_factors = True
    max_output_slots_per_sample = 2**31 - 1

    def __post_init__(self):
        names = _asset_names(self.indices)
        object.__setattr__(self, "indices", names)
        for field in ("spots", "vols", "divs"):
            values = tuple(float(value) for value in getattr(self, field))
            if len(values) != len(names):
                raise InvalidModelParameter("correlated BS asset parameter sizes differ")
            object.__setattr__(self, field, values)
        _correlation_factor(self.correlations, len(names))
        object.__setattr__(
            self, "correlations", tuple(tuple(float(x) for x in row) for row in self.correlations)
        )
        self.validate_params(self.default_params())

    @property
    def num_assets(self):
        return len(self.indices)

    @property
    def n_factors(self):
        return self.num_assets

    @property
    def max_observed_indices(self):
        return self.num_assets

    @property
    def param_labels(self):
        return tuple(
            f"{kind}:{name}" for name in self.indices for kind in ("spot", "vol", "div")
        ) + ("rate",)

    def default_params(self):
        values = {
            f"{kind}:{name}": jnp.asarray(value, dtype=jnp.float64)
            for i, name in enumerate(self.indices)
            for kind, value in (
                ("spot", self.spots[i]),
                ("vol", self.vols[i]),
                ("div", self.divs[i]),
            )
        }
        return values | {"rate": jnp.asarray(self.rate, dtype=jnp.float64)}

    def validate_params(self, params):
        if not math.isfinite(float(params["rate"])):
            raise InvalidModelParameter("rate must be finite")
        for name in self.indices:
            BlackScholes(
                spot=float(params[f"spot:{name}"]),
                vol=float(params[f"vol:{name}"]),
                rate=float(params["rate"]),
                div=float(params[f"div:{name}"]),
            )

    def supports_index(self, index):
        return index.kind == "EQ" and any(ci_eq(index.name, name) for name in self.indices)

    def _asset_slot(self, name):
        canonical = parse_index(name).name
        for i, index in enumerate(self.indices):
            if ci_eq(canonical, index):
                return i
        raise UnsupportedModelObservation(name)

    def allocate(self, timeline: Sequence[float], sample_defs: Sequence[SampleDef]):
        from dal_jax.models.base import validate_timeline

        validate_timeline(self, timeline, sample_defs)
        # Reuse deterministic BS grid and discount construction; all asset outputs
        # are mapped separately and the single-asset observation budget is not used.
        dummy = tuple(
            SampleDef(numeraire=d.numeraire, discount_mats=d.discount_mats) for d in sample_defs
        )
        base = BlackScholes(
            spot=self.spots[0], vol=self.vols[0], rate=self.rate, div=self.divs[0]
        ).allocate(timeline, dummy)
        width = max(len(d.index_names) for d in sample_defs)
        slots = tuple(
            tuple(self._asset_slot(name) for name in d.index_names)
            + (0,) * (width - len(d.index_names))
            for d in sample_defs
        )
        return CorrelatedBSPlan(
            base, slots, _correlation_factor(self.correlations, self.num_assets)
        )

    def sim_dim(self, plan):
        return len(plan.base.dts) * self.n_factors

    def init(self, params, plan):
        spots, vols, divs = (
            jnp.stack([params[f"{kind}:{name}"] for name in self.indices])
            for kind in ("spot", "vol", "div")
        )
        rate = params["rate"]
        dts = jnp.asarray(plan.base.dts)
        numeraires, discounts = deterministic_outputs(rate, plan.base)
        return CorrelatedBSState(
            spots,
            jnp.log(spots),
            dts[:, None] * (rate - divs - 0.5 * vols * vols),
            jnp.sqrt(dts)[:, None] * vols,
            numeraires,
            discounts,
        )

    def generate(self, state, plan, normals):
        base = plan.base
        if not base.dts:
            spots = state.spots[None, :]
        else:
            correlated = (
                normals.reshape(-1, self.n_factors) @ jnp.asarray(plan.lower, dtype=normals.dtype).T
            )
            increments = state.drifts + state.stds * correlated
            logs = jnp.cumsum(jnp.concatenate((state.log_spots[None, :], increments)), axis=0)[1:]
            spots = jnp.exp(logs)
            if base.today_on_timeline:
                spots = jnp.concatenate((state.spots[None, :], spots))
        slots = jnp.asarray(plan.observation_slots, dtype=jnp.int32)
        observations = jnp.take_along_axis(spots, slots, axis=1)
        return Scenario(spots[:, 0], state.numeraires, observations, state.discounts)
