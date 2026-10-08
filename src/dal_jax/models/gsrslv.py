"""GSR stochastic local volatility with full-truncation variance and HJM bonds."""

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dal_jax.errors import InvalidModelParameter, script_error
from dal_jax.models.gsr import (
    GSR,
    MultiFactorGSRVol,
    covariance_factor,
    validate_correlation,
    varying_initial,
)
from dal_jax.models.localvol import _interpolate_spot, _interpolate_time


@dataclass(frozen=True, slots=True, kw_only=True)
class GSRLeverage:
    rate_shifts: tuple
    times: tuple
    values: tuple
    name: str = "leverage"

    def __post_init__(self):
        for label in ("rate_shifts", "times"):
            object.__setattr__(self, label, _leverage_axis(label, getattr(self, label)))
        matrix = np.asarray(self.values, dtype=float)
        if (
            matrix.shape != (len(self.rate_shifts), len(self.times))
            or not np.isfinite(matrix).all()
            or np.any(matrix <= 0.0)
        ):
            raise script_error(
                "InvalidGSRLeverage: grid must match axes and values must be finite and strictly positive"
            )
        object.__setattr__(self, "values", tuple(map(tuple, matrix)))

    def at(self, time, shift, values):
        return _interpolate_spot(
            self.rate_shifts, _interpolate_time(self.times, values, time), shift
        )


def _leverage_axis(label, values):
    values = tuple(map(float, values))
    if (
        not values
        or not np.isfinite(values).all()
        or np.any(np.diff(values) <= 0.0)
        or (label == "times" and values[0] < 0.0)
    ):
        raise script_error(
            "InvalidGSRLeverage: axes must be finite and strictly increasing; times must be nonnegative"
        )
    return values


@dataclass(frozen=True, slots=True, kw_only=True)
class GSRSLVSettings:
    kappa: float = 1.0
    vol_of_vol: float = 0.5
    variance_correlations: tuple[float, ...] = ()
    max_step: float = 1.0 / 52.0

    def __post_init__(self):
        object.__setattr__(
            self,
            "variance_correlations",
            tuple(float(value) for value in self.variance_correlations),
        )


def integration_grid(anchors, max_step, *, dated=False):
    grid = [0.0]
    for time in sorted(set(anchors)):
        if time <= grid[-1]:
            continue
        start = grid[-1]
        grid.extend(
            _dated_steps(start, time, max_step) if dated else _uniform_steps(start, time, max_step)
        )
        grid.append(time)
        if len(grid) > 1_000_001:
            raise script_error("InvalidGSRSLVStep: integration grid exceeds one million steps")
    return tuple(grid)


def _dated_steps(start, time, max_step):
    if max_step < 1.0 / 365.0 or abs(time * 365.0 - round(time * 365.0)) > 1e-7:
        raise script_error(
            "InvalidHybridTimeline: dated rate models require whole calendar days and maximum step of at least one day"
        )
    days = (
        max(1, math.floor(min(max_step, time - start) * 365.0))
        if math.isfinite(max_step)
        else round((time - start) * 365.0)
    )
    return (day / 365.0 for day in range(round(start * 365.0) + days, round(time * 365.0), days))


def _uniform_steps(start, time, max_step):
    count = max(1, math.ceil((time - start) / max_step)) if math.isfinite(max_step) else 1
    if count > 1_000_000:
        raise script_error("InvalidGSRSLVStep: integration grid exceeds one million steps")
    return (start + (time - start) * i / count for i in range(1, count))


@dataclass(frozen=True, slots=True)
class SLVPlan:
    rates: object
    grid: tuple
    sample_indices: tuple


class SLVState(NamedTuple):
    rates: object
    times: object
    widths: object
    g: object
    h: object
    covariance: object
    covariance_h: object
    bank_base: object
    bridge_variance: object
    bridge_std: object
    leverage: object
    kappa: object
    vol_of_vol: object
    lower: object


@dataclass(frozen=True, slots=True, kw_only=True)
class GSRSLV:
    gaussian: GSR
    leverage: GSRLeverage
    settings: GSRSLVSettings = GSRSLVSettings()
    name: str = "smile"

    num_assets = 0
    supports_bb = True
    supports_discount_factors = False
    max_observed_indices = 2**31 - 1
    max_output_slots_per_sample = 2**31 - 1

    def __post_init__(self):
        if not isinstance(self.gaussian, GSR) or not isinstance(
            self.gaussian.vol, MultiFactorGSRVol
        ):
            raise script_error("InvalidGSRSLVModel: Gaussian model must use MultiFactorGSRVol")
        if not math.isfinite(self.settings.max_step) or self.settings.max_step <= 0.0:
            raise script_error("InvalidGSRSLVStep: maximum step must be finite and positive")
        self.validate_params(self.default_params())
        self.driver_correlations()

    @property
    def n_factors(self):
        return self.gaussian.n_factors + 2

    @property
    def evaluation_date(self):
        return self.gaussian.evaluation_date

    @property
    def numeraire_is_deterministic(self):
        return self.gaussian.numeraire_is_deterministic

    @property
    def param_labels(self):
        return (
            self.gaussian.param_labels
            + ("kappa", "volOfVol")
            + tuple(
                f"leverage:{i}:{j}"
                for i in range(len(self.leverage.rate_shifts))
                for j in range(len(self.leverage.times))
            )
        )

    def default_params(self):
        values = {"kappa": self.settings.kappa, "volOfVol": self.settings.vol_of_vol}
        values.update(
            {
                f"leverage:{i}:{j}": value
                for i, row in enumerate(self.leverage.values)
                for j, value in enumerate(row)
            }
        )
        return self.gaussian.default_params() | {
            name: jnp.asarray(value) for name, value in values.items()
        }

    def validate_params(self, params):
        self.gaussian.validate_params(params)
        for name in self.param_labels[len(self.gaussian.param_labels) :]:
            value = float(params[name])
            if not math.isfinite(value) or (
                value <= 0.0 if name.startswith("leverage:") else value < 0.0
            ):
                raise InvalidModelParameter(f"invalid SLV parameter {name}")

    def driver_correlations(self):
        n = self.gaussian.n_factors
        correlations = self.settings.variance_correlations or (0.0,) * n
        if len(correlations) != n:
            raise script_error(
                "InvalidGSRSLVCorrelation: one variance correlation per rate factor is required"
            )
        matrix = np.eye(n + 1)
        matrix[:n, :n] = self.gaussian.correlations
        matrix[n, :n] = matrix[:n, n] = correlations
        return validate_correlation(matrix, n + 1, "InvalidGSRSLVCorrelation")

    def supports_index(self, index):
        return self.gaussian.supports_index(index)

    def breakpoints(self):
        return (
            tuple(
                self.gaussian.time(date)
                for dates in (self.gaussian.vol.g_knot_dates, self.gaussian.vol.h_knot_dates)
                for date in dates
            )
            + self.leverage.times
        )

    def allocate(self, timeline, sample_defs):
        if any(definition.discount_mats for definition in sample_defs):
            raise script_error(
                "UnsupportedDelayedPayment: GSRSLV does not provide delayed-payment discount factors"
            )
        rates = self.gaussian.allocate(timeline, sample_defs)
        anchors = rates.times + tuple(
            time for time in self.breakpoints() if 0.0 < time < rates.times[-1]
        )
        grid = integration_grid(anchors, self.settings.max_step)
        return SLVPlan(rates, grid, tuple(grid.index(time) for time in rates.times))

    def sim_dim(self, plan):
        return (len(plan.grid) - 1) * self.n_factors

    def prepare_steps(self, params, grid, rate_state, correlations=None):
        correlation = jnp.asarray(
            self.gaussian.correlations if correlations is None else correlations
        )
        gs, hs, covs, covhs, banks, variances, stds = [], [], [], [], [], [], []
        for start, end in zip(grid, grid[1:]):
            g, h = self.gaussian.pieces(params, start)
            covariance = g[:, None] * g[None, :] * correlation
            variance = jnp.einsum("i,ij,j->", h, covariance, h)
            positive = variance > 0.0
            std = jnp.where(
                positive,
                jnp.sqrt(jnp.where(positive, variance * (end - start) ** 3 / 12.0, 1.0)),
                0.0,
            )
            gs.append(g)
            hs.append(h)
            covs.append(covariance)
            covhs.append(covariance @ h)
            banks.append(
                self.gaussian.curve.log_df(params, start) - self.gaussian.curve.log_df(params, end)
            )
            variances.append(variance)
            stds.append(std)
        n = self.gaussian.n_factors

        def stack(values, shape):
            return jnp.stack(values) if values else jnp.empty((0,) + shape)

        leverage = jnp.stack(
            [
                jnp.stack([params[f"leverage:{i}:{j}"] for j in range(len(self.leverage.times))])
                for i in range(len(self.leverage.rate_shifts))
            ]
        )
        return SLVState(
            rate_state,
            jnp.asarray(grid[:-1]),
            jnp.diff(jnp.asarray(grid)),
            stack(gs, (n,)),
            stack(hs, (n,)),
            stack(covs, (n, n)),
            stack(covhs, (n,)),
            stack(banks, ()),
            stack(variances, ()),
            stack(stds, ()),
            leverage,
            params["kappa"],
            params["volOfVol"],
            covariance_factor(jnp.asarray(self.driver_correlations())),
        )

    def init(self, params, plan):
        return self.prepare_steps(
            params, plan.grid, self.gaussian.init(params, plan.rates, hjm=True)
        )

    def advance(self, state, carry, data, drivers, bridge):
        x, y, variance, latent, log_n = carry
        time, dt, g, h, cov, covh, bank, bridge_var, bridge_std = data
        shift = jnp.dot(h, x)
        leverage = self.leverage.at(time, shift, state.leverage)
        scale = leverage * leverage * variance
        positive = variance > 0.0
        sqrt_variance = jnp.where(positive, jnp.sqrt(jnp.where(positive, variance, 1.0)), 0.0)
        diffusion = leverage * sqrt_variance
        log_n = (
            log_n
            + bank
            + dt * shift
            + 0.5 * dt * dt * jnp.einsum("i,ij,j->", h, y, h)
            + scale * bridge_var * (dt**3 / 6.0)
            + diffusion
            * (0.5 * dt * jnp.sqrt(dt) * jnp.dot(h * g, drivers[:-1]) + bridge_std * bridge)
        )
        x = (
            x
            + dt * (y @ h)
            + 0.5 * dt * dt * scale * covh
            + diffusion * jnp.sqrt(dt) * g * drivers[:-1]
        )
        y = y + dt * scale * cov
        latent = (
            latent
            + dt * state.kappa * (1.0 - variance)
            + state.vol_of_vol * jnp.sqrt(dt) * sqrt_variance * drivers[-1]
        )
        variance = jnp.where(latent > 0.0, latent, 0.0)
        return x, y, variance, latent, log_n

    def generate(self, state, plan, normals):
        n = self.gaussian.n_factors

        def step(carry, data):
            *coefficients, z = data
            drivers = state.lower @ z[:-1]
            carry = self.advance(state, carry, coefficients, drivers, z[-1])
            return carry, (carry[0], carry[1], carry[-1])

        initial = tuple(
            varying_initial(value, normals)
            for value in (
                jnp.zeros(n, dtype=state.g.dtype),
                jnp.zeros((n, n), dtype=state.g.dtype),
                jnp.asarray(1.0, dtype=state.g.dtype),
                jnp.asarray(1.0, dtype=state.g.dtype),
                jnp.asarray(0.0, dtype=state.g.dtype),
            )
        )
        _, (xs, ys, log_n) = jax.lax.scan(
            jax.checkpoint(step, prevent_cse=False),
            initial,
            tuple(state[1:10]) + (normals.reshape((-1, self.n_factors)),),
        )
        xs = jnp.concatenate((jnp.zeros((1, n), dtype=xs.dtype), xs))
        ys = jnp.concatenate((jnp.zeros((1, n, n), dtype=ys.dtype), ys))
        log_n = jnp.concatenate((jnp.zeros(1, dtype=log_n.dtype), log_n))
        ids = jnp.asarray(plan.sample_indices)
        return self.gaussian.observe(state.rates, plan.rates, xs[ids], log_n[ids], ys[ids])
