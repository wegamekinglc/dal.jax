"""Dated Gaussian short-rate kernels with log-linear OIS/projection curves."""

import bisect
import datetime
import math
import re
from dataclasses import dataclass
from itertools import pairwise
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from dal_jax.dates import Date
from dal_jax.dates.daybasis import DayBasis
from dal_jax.dates.holidays import NO_HOLIDAYS, Holidays, adjust
from dal_jax.dates.increment import parse_increment
from dal_jax.dates.schedule import date_generate
from dal_jax.errors import InvalidModelParameter, script_error
from dal_jax.index import CURRENCIES, parse_index
from dal_jax.models.base import Scenario, validate_timeline
from dal_jax.strings import ci_key


def as_date(value):
    if isinstance(value, Date):
        return value
    if isinstance(value, datetime.date):
        return Date.from_python(value)
    raise TypeError("expected Date or datetime.date")


def _dates(values, code):
    dates = tuple(as_date(value) for value in values)
    if not dates or any(a >= b for a, b in zip(dates, dates[1:])):
        raise script_error(f"{code}: knot dates must be nonempty and strictly increasing")
    return dates


def tenor_months(text):
    match = re.fullmatch(r"([1-9][0-9]*)([MY])", text.upper())
    if not match:
        raise script_error("InvalidGSRCurve: projection tenor must have positive months")
    return int(match[1]) * (12 if match[2] == "Y" else 1)


def covariance_factor(matrix):
    """DAL's PSD Cholesky, including zero pivots and their zero derivatives."""
    n = matrix.shape[0]
    lower = jnp.zeros_like(matrix)
    for i in range(n):
        # Empty prefix reductions can crash XLA's remat transpose under shard_map.
        residual = matrix[i, i] - (jnp.sum(lower[i, :i] ** 2) if i else 0.0)
        positive = residual > 0.0
        diagonal = jnp.where(positive, jnp.sqrt(jnp.where(positive, residual, 1.0)), 0.0)
        lower = lower.at[i, i].set(diagonal)
        for j in range(i + 1, n):
            value = matrix[j, i] - (jnp.sum(lower[j, :i] * lower[i, :i]) if i else 0.0)
            lower = lower.at[j, i].set(
                jnp.where(positive, value / jnp.where(positive, diagonal, 1.0), 0.0)
            )
    return lower


def validate_correlation(values, n, code="InvalidCorrelation"):
    matrix = np.asarray(values, dtype=np.float64)
    tolerance = 64 * n * np.finfo(float).eps  # pylint: disable=no-member
    if (
        matrix.shape != (n, n)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0)
        or not np.allclose(np.diag(matrix), 1.0, atol=tolerance, rtol=0.0)
        or np.linalg.eigvalsh(matrix).min() < -tolerance
    ):
        raise script_error(
            f"{code}: factor correlations must be symmetric, unit-diagonal and positive semidefinite"
        )
    return tuple(tuple(float(x) for x in row) for row in matrix)


def _curve_interval(knots, time):
    if not math.isfinite(time) or time < 0.0 or time > knots[-1]:
        raise script_error("InvalidGSRDate: date lies outside the curve domain")
    upper = min(max(bisect.bisect_right(knots, time), 1), len(knots) - 1)
    return upper, (time - knots[upper - 1]) / (knots[upper] - knots[upper - 1])


@dataclass(frozen=True, slots=True)
class CurveInterpolationPlan:
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class GSRCurve:
    evaluation_date: Date
    currency: str
    node_dates: tuple
    discount_log_df: tuple
    projection_tenors: tuple = ()
    projection_log_df: tuple = ()
    name: str = "curve"

    def __post_init__(self):
        today = as_date(self.evaluation_date)
        dates = _dates(self.node_dates, "InvalidGSRCurve")
        values = tuple(float(x) for x in self.discount_log_df)
        tenors = tuple(self.projection_tenors)
        rows = tuple(tuple(float(x) for x in row) for row in self.projection_log_df)
        _validate_discount_nodes(self.currency, today, dates, values)
        _validate_projection_nodes(dates, tenors, rows)
        for key, value in (
            ("evaluation_date", today),
            ("node_dates", dates),
            ("discount_log_df", values),
            ("projection_tenors", tenors),
            ("projection_log_df", rows),
        ):
            object.__setattr__(self, key, value)

    @property
    def times(self):
        return tuple((date - self.evaluation_date) / 365.0 for date in self.node_dates)

    @property
    def param_labels(self):
        return tuple(
            f"logdf:{tenor}:{date}"
            for tenor in ("OIS",) + self.projection_tenors
            for date in self.node_dates[1:]
        )

    def default_params(self):
        values = tuple(
            x for row in (self.discount_log_df,) + self.projection_log_df for x in row[1:]
        )
        return dict(zip(self.param_labels, map(jnp.asarray, values)))

    def interpolation_plan(self, times, rows=None) -> CurveInterpolationPlan:
        times = tuple(times)
        rows = (-1,) * len(times) if rows is None else tuple(rows)
        knots = self.times
        intervals = tuple(_curve_interval(knots, time) for time in times)
        upper = tuple(index for index, _ in intervals)
        weights = tuple(weight for _, weight in intervals)
        slots = tuple(
            (row + 1) * len(knots) + index for row, index in zip(rows, upper, strict=True)
        )
        return CurveInterpolationPlan(tuple(index - 1 for index in slots), slots, weights)

    def interpolate(self, params, plan: CurveInterpolationPlan):
        rows = tuple(
            jnp.concatenate(
                (
                    jnp.zeros(1),
                    jnp.stack(
                        tuple(params[f"logdf:{tenor}:{date}"] for date in self.node_dates[1:])
                    ),
                )
            )
            for tenor in ("OIS",) + self.projection_tenors
        )
        values = jnp.concatenate(rows)
        weights = jnp.asarray(plan.weights)
        return (1.0 - weights) * values[
            jnp.asarray(plan.lower, dtype=jnp.int32)
        ] + weights * values[jnp.asarray(plan.upper, dtype=jnp.int32)]

    def log_dfs(self, params, times):
        return self.interpolate(params, self.interpolation_plan(times))

    def log_df(self, params, time, row=-1):
        return self.interpolate(params, self.interpolation_plan((time,), (row,)))[0]


def _validate_discount_nodes(currency, today, dates, values):
    if (
        currency not in CURRENCIES
        or len(dates) < 2
        or dates[0] != today
        or len(values) != len(dates)
        or values[0] != 0.0
        or not np.isfinite(values).all()
    ):
        raise script_error(
            "InvalidGSRCurve: dated nodes and finite logDF values must start with the evaluation-date zero anchor"
        )


def _validate_projection_nodes(dates, tenors, rows):
    months = [tenor_months(tenor) for tenor in tenors]
    if (
        len(set(months)) != len(months)
        or len(rows) != len(tenors)
        or any(
            len(row) != len(dates) or row[0] != 0.0 or not np.isfinite(row).all() for row in rows
        )
    ):
        raise script_error(
            "InvalidGSRCurve: projection rows must match unique tenors and dated zero anchors"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GSRVol:
    g_knot_dates: tuple
    g_values: tuple
    h_knot_dates: tuple
    h_values: tuple
    name: str = "vol"

    def __post_init__(self):
        for prefix, positive in (("g", False), ("h", True)):
            dates = _dates(getattr(self, f"{prefix}_knot_dates"), "InvalidGSRVol")
            values = tuple(float(x) for x in getattr(self, f"{prefix}_values"))
            if (
                len(dates) != len(values)
                or not np.isfinite(values).all()
                or any(x <= 0.0 if positive else x < 0.0 for x in values)
            ):
                raise script_error(
                    "InvalidGSRVol: g must be nonnegative and H positive, with one finite value per knot"
                )
            object.__setattr__(self, f"{prefix}_knot_dates", dates)
            object.__setattr__(self, f"{prefix}_values", values)


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiFactorGSRVol:
    factor_names: tuple
    g_knot_dates: tuple
    g_values: tuple
    h_knot_dates: tuple
    h_values: tuple
    correlations: tuple
    name: str = "vol"

    def __post_init__(self):
        names = tuple(self.factor_names)
        if not names or any(not x for x in names) or len(set(map(ci_key, names))) != len(names):
            raise script_error("InvalidMultiFactorGSRVol: factor names must be nonempty and unique")
        object.__setattr__(self, "factor_names", names)
        for prefix in ("g", "h"):
            self._freeze_knots(prefix, len(names))
        object.__setattr__(
            self,
            "correlations",
            validate_correlation(self.correlations, len(names), "InvalidMultiFactorGSRVol"),
        )

    def _freeze_knots(self, prefix, count):
        dates = _dates(getattr(self, f"{prefix}_knot_dates"), "InvalidMultiFactorGSRVol")
        matrix = np.asarray(getattr(self, f"{prefix}_values"), dtype=float)
        if (
            matrix.shape != (count, len(dates))
            or not np.isfinite(matrix).all()
            or (prefix == "g" and np.any(matrix < 0.0))
        ):
            raise script_error(
                "InvalidMultiFactorGSRVol: finite g/H matrices must match factors and knots; g must be nonnegative"
            )
        object.__setattr__(self, f"{prefix}_knot_dates", dates)
        object.__setattr__(self, f"{prefix}_values", tuple(map(tuple, matrix)))


@dataclass(frozen=True, slots=True)
class IntegralPlan:
    widths: tuple[tuple[float, ...], ...]
    g_slots: tuple[tuple[int, ...], ...]
    h_slots: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class GSRPlan:
    times: tuple
    definitions: tuple
    bonds: tuple
    observations: tuple
    projections: tuple
    discount_ids: tuple
    max_bonds: int
    max_observations: int
    max_discounts: int
    step_integrals: IntegralPlan
    cumulative_integrals: IntegralPlan
    bond_integrals: IntegralPlan
    date_curve: CurveInterpolationPlan
    bond_curve: CurveInterpolationPlan
    projection_curve: CurveInterpolationPlan


class GSRState(NamedTuple):
    loading: Array
    lower: Array
    discount_normals: Array
    drift: Array
    bond_loading: Array
    bond_intercept: Array
    scales: Array
    prefix_variance: Array
    prefix_covariance: Array


def _discount_noise(lower, covariance):
    noise = jnp.zeros_like(covariance)
    for i in range(covariance.shape[0]):
        residual = covariance[i] - (jnp.sum(lower[i, :i] * noise[:i]) if i else 0.0)
        positive = lower[i, i] > 0.0
        noise = noise.at[i].set(
            jnp.where(positive, residual / jnp.where(positive, lower[i, i], 1.0), 0.0)
        )
    return noise


def varying_initial(value, normals):
    axes = jax.typeof(normals).manual_axis_type.varying - jax.typeof(value).manual_axis_type.varying
    return jax.lax.pcast(value, tuple(axes), to="varying") if axes else value


@dataclass(frozen=True, slots=True, kw_only=True)
class GSR:
    curve: GSRCurve
    vol: GSRVol | MultiFactorGSRVol
    name: str = "rates"

    num_assets = 0
    supports_bb = True
    supports_discount_factors = True
    max_observed_indices = 2**31 - 1
    max_output_slots_per_sample = 2**31 - 1

    def __post_init__(self):
        if (
            self.vol.g_knot_dates[0] != self.evaluation_date
            or self.vol.h_knot_dates[0] != self.evaluation_date
        ):
            raise script_error("InvalidGSRModel: g and H must start at the curve evaluation date")

    @property
    def evaluation_date(self):
        return self.curve.evaluation_date

    @property
    def n_factors(self):
        return len(self.vol.factor_names) if isinstance(self.vol, MultiFactorGSRVol) else 1

    @property
    def correlations(self):
        return self.vol.correlations if isinstance(self.vol, MultiFactorGSRVol) else ((1.0,),)

    @property
    def numeraire_is_deterministic(self):
        # g is active: live parameters can make a zero-volatility model stochastic.
        return False

    def _rows(self, prefix):
        values = getattr(self.vol, f"{prefix}_values")
        return values if isinstance(self.vol, MultiFactorGSRVol) else (values,)

    def _labels(self, prefix):
        names = self.vol.factor_names if isinstance(self.vol, MultiFactorGSRVol) else ("",)
        dates = self.vol.g_knot_dates if prefix == "g" else self.vol.h_knot_dates
        return tuple(
            f"{prefix}:{name + ':' if name else ''}{date}" for name in names for date in dates
        )

    @property
    def param_labels(self):
        return self.curve.param_labels + self._labels("g") + self._labels("H")

    def default_params(self):
        result = self.curve.default_params()
        for prefix in ("g", "H"):
            result.update(
                zip(
                    self._labels(prefix),
                    map(jnp.asarray, (x for row in self._rows(prefix.lower()) for x in row)),
                )
            )
        return result

    def validate_params(self, params):
        legacy = isinstance(self.vol, GSRVol)
        for label in self.param_labels:
            value = float(params[label])
            if (
                not math.isfinite(value)
                or (label.startswith("g:") and value < 0.0)
                or (legacy and label.startswith("H:") and value <= 0.0)
            ):
                raise InvalidModelParameter(f"invalid rate parameter {label}")

    def date_at(self, time):
        days = round(time * 365.0)
        if (
            not math.isfinite(time)
            or time < 0.0
            or abs(time * 365.0 - days) > 1e-7
            or time > self.curve.times[-1]
        ):
            raise script_error(
                "InvalidGSRDate: model times must represent whole calendar days inside the curve domain"
            )
        return self.evaluation_date.add_days(days)

    def supports_index(self, index):
        if index.kind != "IR":
            return False
        name = index.name
        ccy = name[7:].split(",")[0] if name.startswith("IR[DF]:") else name[3:].split(",")[0]
        return ccy == self.curve.currency

    def _knots(self, start, end):
        return sorted(
            {start, end}
            | {
                self.time(date)
                for dates in (self.vol.g_knot_dates, self.vol.h_knot_dates)
                for date in dates
                if start < self.time(date) < end
            }
        )

    def time(self, date):
        return (date - self.evaluation_date) / 365.0

    def pieces(self, params, time):
        values = []
        for prefix in ("g", "H"):
            dates = self.vol.g_knot_dates if prefix == "g" else self.vol.h_knot_dates
            col = bisect.bisect_right(tuple(self.time(date) for date in dates), time) - 1
            labels = self._labels(prefix)
            values.append(
                jnp.stack(
                    [params[labels[factor * len(dates) + col]] for factor in range(self.n_factors)]
                )
            )
        return tuple(values)

    def _piece_slots(self, dates, times):
        knots = tuple(self.time(date) for date in dates)
        return tuple(bisect.bisect_right(knots, time) - 1 for time in times)

    def _integral_row(self, start, end):
        segments = tuple(pairwise(self._knots(start, end)))
        times = tuple(left for left, _ in segments)
        widths = tuple(right - left for left, right in segments)
        return (
            widths,
            self._piece_slots(self.vol.g_knot_dates, times),
            self._piece_slots(self.vol.h_knot_dates, times),
        )

    def integral_plan(self, intervals) -> IntegralPlan:
        rows = tuple(self._integral_row(start, end) for start, end in intervals)
        width = max(1, max((len(row[0]) for row in rows), default=0))
        padding = (0.0, 0, 0)
        fields = tuple(
            tuple(row[index] + (padding[index],) * (width - len(row[index])) for row in rows)
            for index in range(3)
        )
        return IntegralPlan(*fields)

    def batch_pieces(self, params, times):
        values = []
        for prefix, dates in (("g", self.vol.g_knot_dates), ("H", self.vol.h_knot_dates)):
            knots = tuple(self.time(date) for date in dates)
            slots = jnp.asarray(
                tuple(bisect.bisect_right(knots, time) - 1 for time in times), dtype=jnp.int32
            )
            table = jnp.stack(tuple(params[label] for label in self._labels(prefix))).reshape(
                (self.n_factors, len(dates))
            )
            values.append(table[:, slots].T)
        return tuple(values)

    def integrate(self, params, plan: IntegralPlan):
        g = jnp.stack(tuple(params[label] for label in self._labels("g"))).reshape(
            (self.n_factors, -1)
        )
        h = jnp.stack(tuple(params[label] for label in self._labels("H"))).reshape(
            (self.n_factors, -1)
        )
        gs = g[:, jnp.asarray(plan.g_slots, dtype=jnp.int32)].transpose((1, 2, 0))
        hs = h[:, jnp.asarray(plan.h_slots, dtype=jnp.int32)].transpose((1, 2, 0))
        correlation = jnp.asarray(self.correlations)

        def one(widths, gs, hs):
            vector = varying_initial(jnp.zeros(self.n_factors, dtype=gs.dtype), gs + hs)
            matrix = varying_initial(
                jnp.zeros((self.n_factors, self.n_factors), dtype=gs.dtype), gs + hs
            )

            def forward(carry, data):
                variance, loading = carry
                width, g, h = data
                return (
                    variance + g[:, None] * g[None, :] * correlation * width,
                    loading + h * width,
                ), None

            def backward(carry, data):
                covariance, remaining = carry
                width, g, h = data
                increment = jnp.sum(
                    g[:, None]
                    * g[None, :]
                    * correlation
                    * (width * remaining + 0.5 * width * width * h)[None, :],
                    axis=1,
                )
                return (covariance + increment, remaining + width * h), None

            variance, loading = jax.lax.scan(
                forward,
                (matrix, vector),
                (widths, gs, hs),
            )[0]
            covariance, _ = jax.lax.scan(
                backward,
                (vector, vector),
                (widths, gs, hs),
                reverse=True,
            )[0]
            return variance, loading, covariance

        return jax.vmap(one)(jnp.asarray(plan.widths), gs, hs)

    def integrals(self, params, start, end):
        n = self.n_factors
        variance, loading, covariance = jnp.zeros((n, n)), jnp.zeros(n), jnp.zeros(n)
        knots = self._knots(start, end)
        for left, right in zip(knots, knots[1:]):
            g, h = self.pieces(params, left)
            variance = variance + g[:, None] * g[None, :] * jnp.asarray(self.correlations) * (
                right - left
            )
            loading = loading + h * (right - left)
        remaining = jnp.zeros(n)
        for left, right in reversed(tuple(zip(knots, knots[1:]))):
            g, h = self.pieces(params, left)
            width = right - left
            covariance = covariance + jnp.sum(
                g[:, None]
                * g[None, :]
                * jnp.asarray(self.correlations)
                * (width * remaining + 0.5 * width * width * h)[None, :],
                axis=1,
            )
            remaining = remaining + width * h
        return variance, loading, covariance

    def allocate(self, timeline, sample_defs):
        times = tuple(map(float, timeline))
        validate_timeline(self, times, sample_defs)
        dates = tuple(self.date_at(time) for time in times)
        planner = _ObservationPlanner(self)
        bonds, observations, discounts = [], [], []
        for time, date, definition in zip(times, dates, sample_defs):
            planner.bonds = []
            observations.append(
                tuple(planner.observation(name, date) for name in definition.index_names)
            )
            discounts.append(
                tuple(planner.bond(time, maturity) for maturity in definition.discount_mats)
            )
            bonds.append(tuple(planner.bonds))
        width = max(1, max(map(len, bonds)))
        return GSRPlan(
            times,
            tuple(sample_defs),
            tuple(bonds),
            tuple(observations),
            tuple(planner.projections),
            tuple(discounts),
            width,
            max(map(len, observations)),
            max(map(len, discounts)),
            *self._coefficient_plans(times, bonds, planner.projections, width),
        )

    def _coefficient_plans(self, times, bonds, projections, width):
        intervals = tuple(
            (time, maturity)
            for time, row in zip(times, bonds, strict=True)
            for maturity in row + (time,) * (width - len(row))
        )
        return (
            self.integral_plan(zip((0.0,) + times[:-1], times, strict=True)),
            self.integral_plan((0.0, time) for time in (0.0,) + times),
            self.integral_plan(intervals),
            self.curve.interpolation_plan((0.0,) + times),
            self.curve.interpolation_plan(tuple(end for _, end in intervals)),
            self._projection_plan(projections),
        )

    def _projection_plan(self, projections):
        projection_times = tuple(
            time for start, end, _ in projections for time in (start, end, start, end)
        )
        projection_rows = tuple(value for _, _, row in projections for value in (row, row, -1, -1))
        return self.curve.interpolation_plan(projection_times, projection_rows)

    def sim_dim(self, plan):
        return self.n_factors * sum(time > 0.0 for time in plan.times)

    def init(self, params, plan, *, hjm=False):
        variance, loading, covariance = self.integrate(params, plan.step_integrals)
        prefix_v, _, prefix_c = self.integrate(params, plan.cumulative_integrals)
        lower = jax.vmap(covariance_factor)(variance)
        noise = jax.vmap(_discount_noise)(lower, covariance)
        log_df = self.curve.interpolate(params, plan.date_curve)
        drift = (
            log_df[1:]
            - log_df[:-1]
            - jnp.einsum("si,si->s", loading, prefix_c[:-1])
            - 0.5 * jnp.einsum("si,sij,sj->s", loading, prefix_v[:-1], loading)
            - 0.5 * jnp.sum(noise * noise, axis=1)
        )
        _, bond_b, _ = self.integrate(params, plan.bond_integrals)
        bond_b = bond_b.reshape((len(plan.times), plan.max_bonds, self.n_factors))
        bond_a = (
            self.curve.interpolate(params, plan.bond_curve).reshape(
                (len(plan.times), plan.max_bonds)
            )
            - log_df[1:, None]
        )
        if not hjm:
            bond_a = (
                bond_a
                - jnp.einsum("sbi,si->sb", bond_b, prefix_c[1:])
                - 0.5 * jnp.einsum("sbi,sij,sbj->sb", bond_b, prefix_v[1:], bond_b)
            )
        projections = self.curve.interpolate(params, plan.projection_curve).reshape((-1, 4))
        scales = (
            jnp.exp(projections[:, 0] - projections[:, 1] - projections[:, 2] + projections[:, 3])
            if plan.projections
            else jnp.ones(1)
        )
        return GSRState(
            loading, lower, noise, drift, bond_b, bond_a, scales, prefix_v[1:], prefix_c[1:]
        )

    def observe(self, state, plan, xs, log_n, covariance=None):
        exponent = state.bond_intercept - jnp.einsum("sbf,sf->sb", state.bond_loading, xs)
        if covariance is not None:
            exponent = exponent - 0.5 * jnp.einsum(
                "sbi,sij,sbj->sb", state.bond_loading, covariance, state.bond_loading
            )
        prices = jnp.exp(exponent)
        values, discounts = [], []
        for sample, requests in enumerate(plan.observations):
            row = [
                _observation_value(request, prices[sample], state.scales) for request in requests
            ]
            row += [jnp.asarray(0.0, dtype=xs.dtype)] * (plan.max_observations - len(row))
            values.append(jnp.stack(row) if row else jnp.empty(0, dtype=xs.dtype))
            discount = [prices[sample, index] for index in plan.discount_ids[sample]]
            discount += [jnp.asarray(1.0, dtype=xs.dtype)] * (plan.max_discounts - len(discount))
            discounts.append(jnp.stack(discount) if discount else jnp.empty(0, dtype=xs.dtype))
        bank = jnp.where(
            jnp.asarray([definition.numeraire for definition in plan.definitions]),
            jnp.exp(log_n),
            1.0,
        )
        return Scenario(
            jnp.zeros(len(plan.times), dtype=xs.dtype),
            bank,
            jnp.stack(values),
            jnp.stack(discounts),
        )

    def generate(self, state, plan, normals):
        count, n = len(plan.times), self.n_factors
        zs = normals.reshape((-1, n))
        if plan.times[0] == 0.0:
            zs = jnp.concatenate((jnp.zeros((1, n), dtype=normals.dtype), zs))

        def step(carry, data):
            x, log_n = carry
            b, lower, q, drift, z = data
            log_n = log_n - drift + jnp.dot(b, x) + jnp.dot(q, z)
            x = x + lower @ z
            return (x, log_n), (x, log_n)

        initial = (
            varying_initial(jnp.zeros(n, dtype=state.drift.dtype), normals),
            varying_initial(jnp.asarray(0.0, dtype=state.drift.dtype), normals),
        )
        _, (xs, log_n) = jax.lax.scan(
            jax.checkpoint(step, prevent_cse=False),
            initial,
            (state.loading, state.lower, state.discount_normals, state.drift, zs),
        )
        return self.observe(state, plan, xs.reshape((count, n)), log_n)


def _observation_value(request, prices, scales):
    def libor(recipe):
        start, end, scale, accrual = recipe
        return (scales[scale] * prices[start] / prices[end] - 1.0) / accrual

    if request[0] == "df":
        return prices[request[2]] / prices[request[1]]
    if request[0] == "libor":
        return libor(request[1])
    annuity = sum(accrual * prices[bond] for bond, accrual in request[1])
    floating = sum(accrual * libor(recipe) * prices[bond] for recipe, bond, accrual in request[2])
    return floating / annuity


def _forward_request_fields(name):
    fields = name[3:].rstrip("]").split(",")
    explicit_swap = name.startswith("IR[")
    tenor = fields[2] if explicit_swap else fields[1]
    suffix = (
        fields[3]
        if explicit_swap and len(fields) == 4
        else (fields[2] if not explicit_swap and len(fields) == 3 else "")
    )
    return explicit_swap, tenor, suffix


class _ObservationPlanner:
    def __init__(self, model):
        self.model, self.bonds, self.projections = model, [], []

    def bond(self, time, maturity):
        self.model.curve.log_df(self.model.default_params(), maturity)
        if maturity < time:
            raise script_error("InvalidGSRObservation: bond maturity precedes observation")
        if maturity not in self.bonds:
            self.bonds.append(maturity)
        return self.bonds.index(maturity)

    def date(self, text, sample):
        return (
            Date.from_string(text)
            if len(text) == 10 and text[4] == "-"
            else parse_increment(text).fwd_from(sample)
        )

    def start(self, text, sample):
        if text and len(text) == 10 and text[4] == "-":
            return Date.from_string(text)
        start = self.date(text, sample) if text else sample
        cny = self.model.curve.currency == "CNY"
        holidays = Holidays("CN.IB") if cny else NO_HOLIDAYS
        for _ in range(1 if cny else 2):
            start = holidays.next_bus(start.add_days(1))
        return start

    def libor(self, time, start, end, months):
        if end <= start:
            raise script_error("InvalidGSRObservation: invalid Libor accrual")
        curve = self.model.curve
        tenors = tuple(map(tenor_months, curve.projection_tenors))
        projection = tenors.index(months) if months in tenors else -1
        start_t, end_t = self.model.time(start), self.model.time(end)
        scale_id = len(self.projections)
        self.projections.append((start_t, end_t, projection))
        return (self.bond(time, start_t), self.bond(time, end_t), scale_id, (end - start) / 360.0)

    def observation(self, name, sample):
        index = parse_index(name)
        if not self.model.supports_index(index):
            raise script_error(f"UnsupportedGSRObservation: {name}")
        time, name = self.model.time(sample), index.name
        if name.startswith("IR[DF]:"):
            return self.discount_request(name, sample, time)
        explicit_swap, tenor, suffix = _forward_request_fields(name)
        start = self.start(suffix, sample)
        if not explicit_swap and tenor.startswith("LIBOR_"):
            months = 6 if "6M" in tenor else 3
            return ("libor", self.libor(time, start, start.add_months(months), months))
        return self.swap_request(time, start, tenor)

    def discount_request(self, name, sample, time):
        fields = name[7:].split(",")
        start = self.date(fields[1], sample) if len(fields) == 3 else sample
        end = self.date(fields[-1], sample)
        if start < sample or end <= start:
            raise script_error("InvalidGSRObservation: invalid discount interval")
        return (
            "df",
            self.bond(time, self.model.time(start)),
            self.bond(time, self.model.time(end)),
        )

    def swap_request(self, time, start, tenor):
        end = parse_increment(tenor).fwd_from(start)
        if end <= start:
            raise script_error("InvalidGSRObservation: swap tenor must be positive")
        fixed, floating = [], []
        for months, output in ((6, fixed), (3, floating)):
            dates = date_generate(start, end, parse_increment(f"{months}M"))
            for left, right in zip(dates, dates[1:]):
                left, right = (
                    adjust(NO_HOLIDAYS, date, "ModifiedFollowing") for date in (left, right)
                )
                bond = self.bond(time, self.model.time(right))
                accrual = DayBasis.parse("30_360" if months == 6 else "ACT_360").year_fraction(
                    left, right
                )
                output.append(
                    (bond, accrual)
                    if months == 6
                    else (self.libor(time, left, right, 3), bond, accrual)
                )
        return ("swap", tuple(fixed), tuple(floating))
