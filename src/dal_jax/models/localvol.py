"""Local volatility: log-spot/time interpolation and checkpointed log-Euler steps.

This is DAL's flat-rate HybridLocalVolEquity component. The BS input's scalar
volatility is replaced by the surface; every grid volatility is a parameter.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dal_jax.errors import InvalidLocalVolSurface, InvalidModelParameter, InvalidModelTimeline, UnsupportedModelObservation
from dal_jax.index import CURRENCIES
from dal_jax.models.base import SampleDef, Scenario, validate_timeline
from dal_jax.models.bs import BlackScholes, BSPlan, deterministic_outputs
from dal_jax.models.correlated_bs import plain_equity
from dal_jax.strings import ci_eq


def _axis(values, name, lower, strict):
    axis = tuple(float(value) for value in values)
    array = np.asarray(axis)
    valid_lower = np.all(array > lower) if strict else np.all(array >= lower)
    if not axis or not np.isfinite(array).all() or not valid_lower or not np.all(np.diff(array) > 0):
        raise InvalidLocalVolSurface(f"{name} must be finite, increasing and above {lower}")
    return axis


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalVolSurface:
    spots: tuple[float, ...]
    times: tuple[float, ...]
    vols: tuple[tuple[float, ...], ...]
    name: str = "local_vol"

    def __post_init__(self):
        object.__setattr__(self, "spots", _axis(self.spots, "spots", 0., True))
        object.__setattr__(self, "times", _axis(self.times, "times", 0., False))
        values = np.asarray(self.vols, dtype=np.float64)
        if values.shape != (len(self.spots), len(self.times)) or not np.isfinite(values).all() or np.any(values < 0):
            raise InvalidLocalVolSurface("grid shape must match axes and volatilities must be finite and nonnegative")
        object.__setattr__(self, "vols", tuple(tuple(float(value) for value in row) for row in values))

    def volatility(self, time, spot, *, vols=None):
        """Bilinear interpolation in time and log(spot), flat outside both axes."""
        grid = jnp.asarray(self.vols if vols is None else vols)
        row_values = _interpolate_time(self.times, grid, time)
        return _interpolate_spot(tuple(math.log(s) for s in self.spots), row_values, jnp.log(spot))


def _interpolate_time(times, grid, time):
    if len(times) == 1:
        return grid[:, 0]
    axis = jnp.asarray(times, dtype=grid.dtype)
    upper = jnp.clip(jnp.searchsorted(axis, time, side="right", method="compare_all"), 1, len(times)-1)
    weight = (time-axis[upper-1])/(axis[upper]-axis[upper-1])
    value = (1.-weight)*grid[:, upper-1]+weight*grid[:, upper]
    return jnp.where(time <= axis[0], grid[:, 0], jnp.where(time >= axis[-1], grid[:, -1], value))


def _interpolate_spot(log_spots, row_values, log_spot):
    if len(log_spots) == 1:
        return row_values[0]
    axis = jnp.asarray(log_spots, dtype=log_spot.dtype)
    # Small surface axes suit compare_all. Binary-search scans nested inside
    # rematerialized Euler/AAD scans fail GPU HLO evaluation on JAX 0.11.2.
    upper = jnp.clip(jnp.searchsorted(axis, log_spot, side="right", method="compare_all"), 1, len(log_spots)-1)
    weight = (log_spot-axis[upper-1])/(axis[upper]-axis[upper-1])
    value = (1.-weight)*row_values[upper-1]+weight*row_values[upper]
    return jnp.where(log_spot <= axis[0], row_values[0], jnp.where(log_spot >= axis[-1], row_values[-1], value))


def _grid(times, max_step):
    grid = [0.]
    sample_indices = []
    for time in times:
        if time > grid[-1]:
            count = math.ceil((time-grid[-1])/max_step)
            if count > 1_000_000:
                raise InvalidModelTimeline("local-vol step limit exceeded")
            start = grid[-1]
            grid.extend(start+(time-start)*i/count for i in range(1, count))
            grid.append(time)
        sample_indices.append(len(grid)-1)
    return tuple(grid), tuple(sample_indices)


@dataclass(frozen=True, slots=True)
class LocalVolPlan:
    base: BSPlan
    grid: tuple[float, ...]
    sample_indices: tuple[int, ...]
    max_observations: int


class LocalVolState(NamedTuple):
    spot: object
    log_spot: object
    grid: object
    dts: object
    carry: object
    div: object
    vols: object
    numeraires: object
    discounts: object


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalVol:
    spot: float
    surface: LocalVolSurface
    index: str = "EQ[spot]"
    rate: float = 0.
    div: float = 0.
    currency: str = "USD"
    max_step: float = 1./12.
    name: str = "local_vol"
    factor: str = "W_EQ"

    n_factors = 1
    num_assets = 1
    supports_bb = True
    supports_discount_factors = True
    numeraire_is_deterministic = True
    max_observed_indices = 1
    max_output_slots_per_sample = 2**31-1

    def __post_init__(self):
        object.__setattr__(self, "index", plain_equity(self.index))
        if self.currency not in CURRENCIES or not math.isfinite(self.max_step) or self.max_step <= 0:
            raise InvalidModelParameter("currency must be supported and maximum step finite and positive")
        self.validate_params(self.default_params())

    @property
    def param_labels(self):
        grid = tuple(f"lvol:{self.index}:{i}:{j}" for i in range(len(self.surface.spots)) for j in range(len(self.surface.times)))
        return (f"spot:{self.index}", f"div:{self.index}")+grid+(f"rate:{self.currency}",)

    def default_params(self):
        grid = {f"lvol:{self.index}:{i}:{j}": jnp.asarray(v, dtype=jnp.float64)
                for i, row in enumerate(self.surface.vols) for j, v in enumerate(row)}
        values = {f"spot:{self.index}": self.spot, f"div:{self.index}": self.div, f"rate:{self.currency}": self.rate}
        return grid | {name: jnp.asarray(value, dtype=jnp.float64) for name, value in values.items()}

    def vol_grid(self, params):
        return jnp.stack([jnp.stack([params[f"lvol:{self.index}:{i}:{j}"] for j in range(len(self.surface.times))])
                          for i in range(len(self.surface.spots))])

    def validate_params(self, params):
        spot = float(params[f"spot:{self.index}"])
        if not math.isfinite(spot) or spot <= 0:
            raise InvalidModelParameter("spot must be finite and positive")
        for name in self.param_labels:
            value = float(params[name])
            if not math.isfinite(value) or (name.startswith("lvol:") and value < 0):
                raise InvalidModelParameter(f"invalid local-vol parameter {name}")

    def supports_index(self, index):
        return index.kind == "EQ" and ci_eq(index.name, self.index)

    def allocate(self, timeline, sample_defs):
        validate_timeline(self, timeline, sample_defs)
        for definition in sample_defs:
            for name in definition.index_names:
                from dal_jax.index import parse_index
                if not self.supports_index(parse_index(name)):
                    raise UnsupportedModelObservation(name)
        dummy = tuple(SampleDef(numeraire=d.numeraire, discount_mats=d.discount_mats) for d in sample_defs)
        base = BlackScholes(spot=self.spot, vol=0., rate=self.rate, div=self.div).allocate(timeline, dummy)
        grid, indices = _grid(base.times, self.max_step)
        return LocalVolPlan(base, grid, indices, max(len(d.index_names) for d in sample_defs))

    def sim_dim(self, plan):
        return len(plan.grid)-1

    def init(self, params, plan):
        spot, div, rate = (params[name] for name in (f"spot:{self.index}", f"div:{self.index}", f"rate:{self.currency}"))
        grid = jnp.asarray(plan.grid)
        log_discounts = -rate*grid
        numeraires, discounts = deterministic_outputs(rate, plan.base)
        return LocalVolState(spot, jnp.log(spot), grid, jnp.diff(grid), log_discounts[:-1]-log_discounts[1:],
                             div, self.vol_grid(params), numeraires, discounts)

    def generate(self, state, plan, normals):
        if not self.sim_dim(plan):
            spots = state.spot[None]
        else:
            def step(log_spot, data):
                time, dt, carry, normal = data
                vol = self.surface.volatility(time, jnp.exp(log_spot), vols=state.vols)
                std = vol*jnp.sqrt(dt)
                increment = carry-state.div*dt-.5*std*std+std*normal
                log_spot = log_spot+increment
                return log_spot, jnp.exp(log_spot)
            axes = jax.typeof(normals).manual_axis_type.varying-jax.typeof(state.log_spot).manual_axis_type.varying
            initial = jax.lax.pcast(state.log_spot, tuple(axes), to="varying") if axes else state.log_spot
            _, future = jax.lax.scan(jax.checkpoint(step, prevent_cse=False), initial,
                                     (state.grid[:-1], state.dts, state.carry, normals))
            all_spots = jnp.concatenate((state.spot[None], future))
            spots = jnp.take(all_spots, jnp.asarray(plan.sample_indices))
        observations = jnp.broadcast_to(spots[:, None], (len(plan.base.times), plan.max_observations))
        return Scenario(spots, state.numeraires, observations, state.discounts)
