"""Dupire local-volatility calibration, mirroring dal-python 2026.11.1.

The densification, stable band, central-difference stencil and validation
order mirror the native ``DupireCentral1e-4FixedBaseBand/v1`` implementation
(dal-cpp ``dal/model/dupirerisk.cpp``).  The pricing helpers below are exact
float mirrors of the native Black/Merton/Brent stack — they only use libm
functions through :mod:`math`, which the native wheel shares — so calibrated
snapshot values match native bit-for-bit.

Base implied-volatility surfaces come in three kinds, as natively: a
Black-Scholes model (constant volatility), a Merton jump-diffusion smile, or
a callable.  Native's callable base subclasses a C++ trampoline; ours accepts
the callable directly through ``IVS(implied_vol=...)`` — the documented
ergonomic divergence — while subclassing ``IVS`` and overriding
``implied_vol`` keeps working.
"""

import enum
import math
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import ClassVar

from dal_jax.errors import (
    DupireSnapshotMismatch,
    InvalidDupireCalibration,
    InvalidDupireInput,
    InvalidDupirePullback,
    InvalidDupireQuote,
    InvalidModelIndex,
    script_error,
)
from dal_jax.models.base import Model
from dal_jax.models.bs import BlackScholes
from dal_jax.models.correlated_bs import plain_equity
from dal_jax.models.hybrid import (
    Hybrid,
    HybridCorrelation,
    HybridDeterministicRate,
    HybridLocalVolEquity,
)
from dal_jax.models.localvol import LocalVolSurface
from dal_jax.strings import ci_eq

#  Native constants (dal-cpp dal/platform/consts.hpp and the Dupire sources).
_DAL_EPSILON = 2e-14
_DBL_EPSILON = sys.float_info.epsilon
_SQRT_2 = 1.4142135623730951
_SQRT_2_PI = 2.5066282746310002
_INT_MAX = 2147483647
_ONE_HOUR_YF = 0.000114469  # the one-hour year fraction flooring the time grid
_SPOT_FILL_MARGIN = 0.01  # native FillData minDx for the spot axis
_ALGORITHM = "DupireCentral1e-4FixedBaseBand/v1"
_NUMERIC_EXPECTED = "finite int or float, excluding bool and enums"


def _input_context(value, field, expected):
    """Native InputContext detail: type (and value for scalars) plus the expectation.

    The raising error class supplies the ``<code>: `` prefix, as :class:`DalError`
    renders ``str(error)``.
    """
    detail = f"type={type(value).__name__}"
    if value is None or isinstance(value, (str, int, float, bytes)):
        detail += f", value={value!r}"
    return f"{field}; {detail}; expected {expected}"


def _numeric_input(value, field):
    """Native NumericInput: a finite non-bool, non-enum number."""
    context = _input_context(value, field, _NUMERIC_EXPECTED)
    numeric = not isinstance(value, bool) and isinstance(value, (int, float))
    if isinstance(value, enum.Enum) or not numeric:
        raise InvalidDupireCalibration(context)
    try:
        result = float(value)
    except OverflowError as error:
        raise InvalidDupireCalibration(context) from error
    if not math.isfinite(result):
        raise InvalidDupireCalibration(context)
    return result


#  ---------------------------------------------------------------------------
#  Black/Merton pricing and the Brent implied-volatility solver, mirroring
#  dal-cpp dal/math/distribution/black.{hpp,cpp}, dal/math/analytics/vanilla
#  and dal/math/rootfind to the same operation order.


def _ncdf(z):
    return 0.5 * math.erfc(-z / _SQRT_2)


def _black_call(forward, std, strike):
    """Native Distribution::BlackOpt for a call."""
    if -_DAL_EPSILON < std < _DAL_EPSILON or forward * strike < _DAL_EPSILON:
        return forward - strike if forward > strike else 0.0
    d_minus = math.log(forward / strike) / std - 0.5 * std
    d_plus = d_minus + std
    return forward * _ncdf(d_plus) - strike * _ncdf(d_minus)


def _merton_call(spot, strike, vol, maturity, intensity, mean_jump, jump_std):
    """Native AAD::Merton: ten jump-size terms over shifted Black calls."""
    var_jump = jump_std * jump_std
    mv2 = mean_jump + 0.5 * var_jump
    comp = intensity * (math.exp(mv2) - 1.0)
    var = vol * vol
    intensity_t = intensity * maturity
    fact = 1
    weighted = 1.0
    result = 0.0
    for n in range(10):
        shifted = spot * math.exp(n * mv2 - comp * maturity)
        total_std = math.sqrt(var + n * var_jump / maturity)
        prob = math.exp(-intensity_t) * weighted / fact
        result += prob * _black_call(shifted, total_std * math.sqrt(maturity), strike)
        fact *= n + 1
        weighted *= intensity_t
    return result


class _BracketedBrent:
    """Native BracketedBrent_: Brent's method on an enforced bracket."""

    def __init__(self, tol):
        self.tol = tol
        self.a = self.b = self.c = (0.0, 0.0)
        self.bisect = True
        self.d = 0.0

    def initialize(self, low, high):
        self.a, self.b = low, high
        if self.a[1] * self.b[1] > 0.0:
            raise RuntimeError("Interval is not bracketed")
        if abs(self.a[1]) < abs(self.b[1]):
            self.a, self.b = self.b, self.a
        self.c = self.a
        self.bisect = True

    def next_x(self):
        if self.c[1] != self.a[1] and self.c[1] != self.b[1]:
            s = (
                self.a[0]
                * self.b[1]
                * self.c[1]
                / ((self.a[1] - self.b[1]) * (self.a[1] - self.c[1]))
                + self.b[0]
                * self.a[1]
                * self.c[1]
                / ((self.b[1] - self.a[1]) * (self.b[1] - self.c[1]))
                + self.c[0]
                * self.a[1]
                * self.b[1]
                / ((self.c[1] - self.a[1]) * (self.c[1] - self.b[1]))
            )
        else:
            s = (self.a[0] * self.b[1] - self.b[0] * self.a[1]) / (self.b[1] - self.a[1])
        c_dist = abs(self.c[0] - (self.b[0] if self.bisect else self.d))
        self.bisect = (
            (s - self.b[0]) * (s - 0.75 * self.a[0] - 0.25 * self.b[0]) >= 0.0
            or abs(s - self.b[0]) > 0.5 * c_dist
            or c_dist < self.tol
        )
        if self.bisect:
            s = 0.5 * (self.a[0] + self.b[0])
        self.d = s
        return self.d

    def put_y(self, f_s):
        s = self.d
        self.d = self.c[0]
        self.c = self.b
        if f_s * self.a[1] > 0:
            self.a = (s, f_s)
        else:
            self.b = (s, f_s)
        if abs(self.a[1]) < abs(self.b[1]):
            self.a, self.b = self.b, self.a

    def bracket_width(self):
        return abs(self.a[0] - self.b[0])


class _Brent:
    """Native Brent_: a hunt phase growing the step, then bracketed Brent."""

    _INF = 1e29

    def __init__(self, guess, tol=_DAL_EPSILON, step_size=0.0):
        self.phase = "initialize"
        self.increasing = True
        self.step_size = step_size if step_size > 0.0 else 0.1 * max(0.01, abs(guess))
        self.trial_x = guess
        self.known = (self._INF, self._INF)
        self.engine = _BracketedBrent(tol)

    def next_x(self):
        return self.engine.next_x() if self.phase == "bracketed" else self.trial_x

    def put_y(self, y):
        if self.phase == "initialize":
            self.known = (self.trial_x, y)
            self.trial_x += (
                self.step_size * (1 if self.increasing else -1) * (-1 if y >= 0.0 else 1)
            )
            self.phase = "hunt"
        elif self.phase == "bracketed":
            self.engine.put_y(y)
        elif y * self.known[1] <= 0.0:
            self.engine.initialize(self.known, (self.trial_x, y))
            self.phase = "bracketed"
        else:
            self._extend_hunt(y)

    def _extend_hunt(self, y):
        if abs(y) > abs(self.known[1]):
            self.increasing = not self.increasing
        else:
            self.known = (self.trial_x, y)
        self.step_size *= math.exp(1.0)
        direction = 1 if self.increasing else -1
        sign = -1 if self.known[1] >= 0.0 else 1
        self.trial_x = self.known[0] + direction * sign * self.step_size

    def bracket_width(self):
        return self.engine.bracket_width() if self.phase == "bracketed" else self._INF


def _black_iv(forward, strike, price):
    """Native Distribution::BlackIV: solve the total standard deviation."""
    if price < (forward - strike if forward > strike else 0.0):
        raise RuntimeError("value below intrinsic value in BlackIV")
    task = _Brent(-1.5)
    x_tol = 1e-10 * max(1.0, abs(forward))
    f_tol = 1e-10 * max(1.0, price)
    for _ in range(30):
        vol = math.exp(task.next_x())
        residual = _black_call(forward, vol, strike) - price
        task.put_y(residual)
        if abs(residual) < f_tol or task.bracket_width() < x_tol:
            return vol
    raise RuntimeError("exhausted iterations in BlackIV")


def _black_scholes_ivol(spot, strike, price, maturity):
    """Native BlackScholesIVol: the annualised implied volatility."""
    if maturity <= 0.0:
        raise RuntimeError("BlackScholesIVol: maturity must be positive")
    return _black_iv(spot, strike, price) / math.sqrt(maturity)


#  ---------------------------------------------------------------------------
#  Base implied-volatility surfaces.


class IVS:
    """Base implied-volatility surface, mirroring native ``IVS_``.

    Pass ``implied_vol`` as a ``strike, maturity`` callable (the ergonomic
    divergence from native's trampoline subclass), or subclass and override
    ``implied_vol`` exactly as natively.
    """

    def __init__(self, *, spot, rate=0.0, dividend_yield=0.0, implied_vol=None):
        spot = _numeric_input(spot, "IVS_; spot")
        if spot <= 0.0:
            raise InvalidDupireCalibration("IVS_; spot must be positive")
        self.spot = spot
        self.rate = _numeric_input(rate, "IVS_; rate")
        self.dividend_yield = _numeric_input(dividend_yield, "IVS_; dividend_yield")
        if implied_vol is not None and not callable(implied_vol):
            raise InvalidDupireCalibration("IVS_; implied_vol override is required")
        self._callable = implied_vol

    def implied_vol(self, strike, maturity):
        if self._callable is None:
            raise InvalidDupireCalibration("IVS_; implied_vol override is required")
        return _numeric_input(self._callable(strike, maturity), "IVS_; implied_vol")


class MertonIVS(IVS):
    """Merton jump-diffusion smile, mirroring native ``MertonIVS_``."""

    def __init__(self, *, spot, vol, intensity, average_jump, jump_std):
        spot = _numeric_input(spot, "MertonIVS_; spot")
        vol = _numeric_input(vol, "MertonIVS_; vol")
        intensity = _numeric_input(intensity, "MertonIVS_; intensity")
        average_jump = _numeric_input(average_jump, "MertonIVS_; average_jump")
        jump_std = _numeric_input(jump_std, "MertonIVS_; jump_std")
        if spot <= 0.0:
            raise InvalidDupireCalibration("MertonIVS_; spot must be finite and positive")
        if vol < 0.0:
            raise InvalidDupireCalibration("MertonIVS_; vol must be finite and nonnegative")
        if intensity < 0.0:
            raise InvalidDupireCalibration("MertonIVS_; intensity must be finite and nonnegative")
        if jump_std < 0.0:
            raise InvalidDupireCalibration("MertonIVS_; jump_std must be finite and nonnegative")
        super().__init__(spot=spot)
        self.vol = vol
        self.intensity = intensity
        self.average_jump = average_jump
        self.jump_std = jump_std

    def implied_vol(self, strike, maturity):
        price = _merton_call(
            self.spot, strike, self.vol, maturity, self.intensity, self.average_jump, self.jump_std
        )
        return _black_scholes_ivol(self.spot, strike, price, maturity)


class _ConstantVolIVS(IVS):
    """The flat smile a Black-Scholes base contributes (native ConstantVolIVS_)."""

    def __init__(self, spot, rate, dividend_yield, vol):
        super().__init__(spot=spot, rate=rate, dividend_yield=dividend_yield)
        self.vol = float(vol)

    def implied_vol(self, strike, maturity):
        return self.vol


#  ---------------------------------------------------------------------------
#  Risk inputs.


def _axis_input(value, field):
    if not isinstance(value, (list, tuple)):
        raise InvalidDupireQuote(
            _input_context(value, f"DupireRiskInputs_; {field}", "list, tuple or DoubleVector")
        )
    return tuple(_numeric_input(item, f"DupireRiskInputs_; {field}") for item in value)


def _spreads_input(value):
    context = _input_context(value, "DupireRiskInputs_; quote_spreads", "DoubleMatrix_")
    if not isinstance(value, (list, tuple)):
        raise InvalidDupirePullback(context)
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            raise InvalidDupirePullback(
                _input_context(row, "DupireRiskInputs_; quote_spreads", "DoubleMatrix_")
            )
        converted = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise InvalidDupirePullback(
                    _input_context(item, "DupireRiskInputs_; quote_spreads", "DoubleMatrix_")
                )
            converted.append(float(item))
        rows.append(tuple(converted))
    return tuple(rows)


@dataclass(frozen=True, slots=True, kw_only=True)
class DupireRiskInputs:
    """Calibration inputs and the quote grid, mirroring native ``DupireRiskInputs_``."""

    quote_strikes: tuple
    quote_maturities: tuple
    quote_spreads: tuple
    inclusion_spots: tuple
    max_spot_spacing: float
    inclusion_times: tuple
    max_time_spacing: float

    def __post_init__(self):
        object.__setattr__(self, "quote_strikes", _axis_input(self.quote_strikes, "quote_strikes"))
        object.__setattr__(
            self, "quote_maturities", _axis_input(self.quote_maturities, "quote_maturities")
        )
        object.__setattr__(self, "quote_spreads", _spreads_input(self.quote_spreads))
        object.__setattr__(
            self, "inclusion_spots", _axis_input(self.inclusion_spots, "inclusion_spots")
        )
        object.__setattr__(
            self,
            "max_spot_spacing",
            _numeric_input(self.max_spot_spacing, "DupireRiskInputs_; max_spot_spacing"),
        )
        object.__setattr__(
            self, "inclusion_times", _axis_input(self.inclusion_times, "inclusion_times")
        )
        object.__setattr__(
            self,
            "max_time_spacing",
            _numeric_input(self.max_time_spacing, "DupireRiskInputs_; max_time_spacing"),
        )


#  ---------------------------------------------------------------------------
#  Input validation, mirroring native ValidateInputs/ValidateAxis/ValidateSpacing.


def _validate_axis(axis, field):
    if not axis or len(axis) > _INT_MAX:
        raise InvalidDupireQuote(f"empty or oversized axis; field={field}")
    for ordinal, value in enumerate(axis):
        if not math.isfinite(value) or value <= 0.0 or (ordinal > 0 and value <= axis[ordinal - 1]):
            raise InvalidDupireQuote(
                f"axis must be finite, positive and strictly increasing; field={field}"
            )


def _validate_spacing(axis, spacing, add_spacing):
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise InvalidDupireCalibration("grid spacing must be finite and positive")
    low = min(axis[0], spacing) if add_spacing else axis[0]
    high = max(axis[-1], spacing) if add_spacing else axis[-1]
    if high != low and spacing < high - math.nextafter(high, 0.0):
        raise InvalidDupireCalibration("grid spacing is below coordinate precision")
    if (high - low) / spacing >= _INT_MAX - 2.0:
        raise InvalidDupireCalibration("grid spacing exceeds the supported fill extent")


def _validate_inputs(inputs):
    _validate_axis(inputs.quote_strikes, "quoteStrikes")
    _validate_axis(inputs.quote_maturities, "quoteMaturities")
    _validate_axis(inputs.inclusion_spots, "inclusionSpots")
    _validate_axis(inputs.inclusion_times, "inclusionTimes")
    if len(inputs.quote_spreads) != len(inputs.quote_strikes) or any(
        len(row) != len(inputs.quote_maturities) for row in inputs.quote_spreads
    ):
        raise InvalidDupireQuote("spread matrix must match strike/maturity axes")
    for row in inputs.quote_spreads:
        for spread in row:
            if not math.isfinite(spread):
                raise InvalidDupireQuote("non-finite spread")
    _validate_spacing(inputs.inclusion_spots, inputs.max_spot_spacing, False)
    _validate_spacing(inputs.inclusion_times, inputs.max_time_spacing, True)


#  ---------------------------------------------------------------------------
#  Grid densification, mirroring native AAD::FillData.


def _set_union_first_wins(axis, extra, tolerance):
    """std::set_union of the axis with one extra point; near-equal keeps the axis copy."""
    merged = []
    index = 0
    taken = False
    while index < len(axis) and not taken:
        if axis[index] < extra - tolerance:
            merged.append(axis[index])
            index += 1
        elif extra < axis[index] - tolerance:
            merged.append(extra)
            taken = True
        else:
            merged.append(axis[index])
            index += 1
            taken = True
    if not taken:
        merged.append(extra)
    merged.extend(axis[index:])
    return merged


def _fill_data(sequence, max_dx, min_dx):
    """Native AAD::FillData: supplement points so gaps stay within max_dx."""
    filled = [sequence[0]]
    for nxt in sequence[1:]:
        current = filled[-1]
        if nxt - current > max_dx:
            add_points = int((nxt - current) / max_dx - _DAL_EPSILON) + 1
            spacing = (nxt - current) / add_points
            point = current + spacing
            while point < nxt - min_dx:
                filled.append(point)
                point += spacing
        filled.append(nxt)
    return filled


#  ---------------------------------------------------------------------------
#  Quote-spread interpolation, mirroring native RiskView_::Spread
#  (Interp2DLinearImplX over InterpLinearImplX, flat outside both axes).


def _is_zero(x):
    return -_DAL_EPSILON < x < _DAL_EPSILON


def _interp_linear(xs, ys, x0):
    index = bisect_left(xs, x0)
    if index == len(xs):
        return ys[-1]
    if index == 0 or _is_zero(x0 - xs[index]):
        return ys[index]
    frac = (x0 - xs[index - 1]) / (xs[index] - xs[index - 1])
    return ys[index - 1] + frac * (ys[index] - ys[index - 1])


def _interp2_linear(xs, ys, rows, x0, y0):
    upper = bisect_right(xs, x0)
    if upper == len(xs):
        return _interp_linear(ys, rows[upper - 1], y0)
    if upper == 0:
        return _interp_linear(ys, rows[0], y0)
    lower = upper - 1
    z_low = _interp_linear(ys, rows[lower], y0)
    z_high = _interp_linear(ys, rows[upper], y0)
    weight = (x0 - xs[lower]) / (xs[upper] - xs[lower])
    return z_low + (z_high - z_low) * weight


def _spread_view(inputs):
    def spread(strike, maturity):
        return _interp2_linear(
            inputs.quote_strikes, inputs.quote_maturities, inputs.quote_spreads, strike, maturity
        )

    return spread


#  ---------------------------------------------------------------------------
#  Base freezing, the stable band and the calibration stencil.


def _base_call(carry, strike, maturity, vol):
    """Native AAD::IVS_::Call for one already-sampled implied volatility."""
    spot, rate, dividend = carry
    forward = spot * math.exp((rate - dividend) * maturity)
    discount = math.exp(-rate * maturity)
    return discount * _black_call(forward, vol * math.sqrt(maturity), strike)


def _stencil_points(strike, maturity):
    dt = 1e-4 * maturity
    ds = 1e-4 * strike
    return (
        (strike, maturity),
        (strike, maturity - dt),
        (strike, maturity + dt),
        (strike - ds, maturity),
        (strike + ds, maturity),
    )


def _sample_base(base, point, samples):
    if point in samples:
        return
    vol = base.implied_vol(*point)
    if not math.isfinite(vol) or vol < 0.0:
        raise InvalidDupireCalibration("base implied volatility must be finite and nonnegative")
    samples[point] = float(vol)


def _stable_band(samples, carry, spots, maturity):
    atm_call = _base_call(carry, carry[0], maturity, samples[(carry[0], maturity)])
    width = 2.5 * atm_call * _SQRT_2_PI
    if not math.isfinite(width) or width < 0.0:
        raise InvalidDupireCalibration("non-finite or negative base ATM call band")
    low = bisect_left(spots, carry[0] - width)
    high = bisect_right(spots, carry[0] + width)
    if low >= high:
        raise InvalidDupireCalibration("no spot nodes inside stable calibration band")
    return low, high


def _freeze_base(base, spots, times):
    """Sample the base smile at the ATM and band-row stencil points, per maturity."""
    spot, rate, dividend = base.spot, base.rate, base.dividend_yield
    if not (math.isfinite(spot) and spot > 0.0 and math.isfinite(rate) and math.isfinite(dividend)):
        raise InvalidDupireCalibration(
            "spot and deterministic carry must be finite with positive spot"
        )
    carry = (spot, rate, dividend)
    samples = {}
    bands = []
    for maturity in times:
        _sample_base(base, (spot, maturity), samples)
        band = _stable_band(samples, carry, spots, maturity)
        bands.append(band)
        for row in range(band[0], band[1]):
            for point in _stencil_points(spots[row], maturity):
                _sample_base(base, point, samples)
    return samples, bands


def _checked_call(samples, carry, point, spread):
    """Native CheckedCall: the bumped stencil call, validated as native orders it."""
    implied = samples[point] + spread(*point)
    if not math.isfinite(implied) or implied <= 0.0:
        raise InvalidDupireQuote("bumped implied volatility must be finite and positive")
    call = _base_call(carry, point[0], point[1], implied)
    if not math.isfinite(call):
        raise InvalidDupireCalibration("non-finite stencil call")
    return call


def _stencil_local_vol(samples, carry, strike, maturity, spread):
    """Validate one native stencil, then its Dupire local volatility."""
    calls = [
        _checked_call(samples, carry, point, spread) for point in _stencil_points(strike, maturity)
    ]
    ds = 1e-4 * strike
    dt = 1e-4 * maturity
    centered = calls[3] + calls[4] - 2.0 * calls[0]
    scale = abs(calls[3]) + abs(calls[4]) + 2.0 * abs(calls[0])
    if not centered > 8.0 * _DBL_EPSILON * scale:
        raise InvalidDupireCalibration(
            "discrete call curvature is nonpositive or numerically unresolved"
        )
    curvature = centered / ds / ds
    ct = (calls[2] - calls[1]) * 0.5 / dt
    ck = (calls[4] - calls[3]) * 0.5 / ds
    numerator = ct + carry[2] * calls[0] + (carry[1] - carry[2]) * strike * ck
    variance = 2.0 * numerator / curvature
    if not (
        math.isfinite(curvature)
        and math.isfinite(ct)
        and math.isfinite(ck)
        and math.isfinite(variance)
        and variance > 0.0
    ):
        raise InvalidDupireCalibration("local variance is nonpositive or non-finite")
    return math.sqrt(variance) / strike


def _calibrate_grid(samples, carry, spots, times, bands, spread):
    """Per-maturity band calibration with flat fills outside, as native DupireCalib."""
    rows = [[0.0] * len(times) for _ in spots]
    for column, maturity in enumerate(times):
        low, high = bands[column]
        for row in range(low, high):
            rows[row][column] = _stencil_local_vol(samples, carry, spots[row], maturity, spread)
        for row in range(low):
            rows[row][column] = rows[low][column]
        for row in range(high, len(spots)):
            rows[row][column] = rows[high - 1][column]
    return tuple(tuple(row) for row in rows)


#  ---------------------------------------------------------------------------
#  The calibration entry point and its snapshot.


def _setting_string(value, field, error=InvalidDupireCalibration):
    context = _input_context(value, field, "str or DAL String_ without NUL")
    if not isinstance(value, str) or "\0" in value:
        raise error(context)
    return value


def _equity_index(index):
    """The canonical EQ index name, gated as native's ordinary-EQ check.

    Native requires the name to CI-start with ``EQ[`` before parsing (DAL's
    ``String_`` equality is case-insensitive); non-EQ names fail with
    ``InvalidModelIndex`` while EQ-prefixed-but-malformed names surface the
    equity parser's own ``InvalidIndex``.
    """
    if not ci_eq(index[:3], "EQ["):
        raise InvalidModelIndex(f"expected an ordinary EQ index: {index}")
    return plain_equity(index)


def dupire_model_data(calibration, index, currency, factor, name="", max_step=1.0 / 12.0):
    """Mirror native NewDupireModelData: local-vol equity plus deterministic rate.

    Native assembles a hybrid whose components are named ``equity`` and ``rate``
    with an identity correlation, so the risk axis is ``spot``/``div``, the
    row-major surface nodes, then ``rate`` — exactly as ``ValueWithRisk``
    reports it.
    """
    if not isinstance(calibration, DupireCalibrationSnapshot):
        raise InvalidDupirePullback(
            _input_context(
                calibration, "DupireModelData_New; calibration", "DupireCalibrationSnapshot_"
            )
        )
    label = _setting_string(name, "DupireModelData_New; name", InvalidDupireInput)
    _setting_string(index, "DupireModelData_New; index", InvalidDupireInput)
    _setting_string(currency, "DupireModelData_New; currency", InvalidDupireInput)
    _setting_string(factor, "DupireModelData_New; factor", InvalidDupireInput)
    step = _numeric_input(max_step, "DupireModelData_New; max_step")
    if step <= 0.0:
        raise InvalidDupireInput("DupireModelData_New; max_step must be positive")
    equity_index = _equity_index(index)
    if not currency or not factor:
        raise script_error(
            "InvalidHybridComponent: local-vol equity name, currency, factor, and surface are required"
        )
    equity = HybridLocalVolEquity(
        name="equity",
        index=equity_index,
        currency=currency,
        factor=factor,
        spot=calibration.spot,
        div=calibration.dividend_yield,
        surface=calibration.surface,
        max_step=step,
    )
    rate = HybridDeterministicRate(name="rate", currency=currency, rate=calibration.rate)
    correlation = HybridCorrelation(
        name="correlation", factor_names=(factor,), correlations=((1.0,),)
    )
    return Hybrid(
        name=label, domestic_currency=currency, components=(equity, rate), correlation=correlation
    )


def _base_ivs(base):
    if isinstance(base, IVS):
        return base
    if isinstance(base, BlackScholes):
        return _ConstantVolIVS(base.spot, base.rate, base.div, base.vol)
    if isinstance(base, Model):
        raise InvalidDupireCalibration("DupireCalibration_New; base must be BSModelData_ or IVS_")
    raise InvalidDupireCalibration(
        _input_context(base, "DupireCalibration_New; base", "BSModelData_ or IVS_")
    )


@dataclass(frozen=True, slots=True)
class DupireCalibrationSnapshot:
    """A frozen calibration, mirroring native ``DupireCalibrationSnapshot_``.

    ``base_samples`` and ``bands`` retain the frozen smile and per-maturity
    stable bands the pullback replays; ``matches`` compares everything native
    compares (never the name).
    """

    inputs: DupireRiskInputs
    spot: float
    rate: float
    dividend_yield: float
    surface: LocalVolSurface
    base_samples: tuple
    bands: tuple

    algorithm: ClassVar[str] = _ALGORITHM

    @property
    def spots(self):
        return self.surface.spots

    @property
    def times(self):
        return self.surface.times

    @property
    def vols(self):
        return self.surface.vols

    def matches(self, other):
        """Native Matches: algorithm, inputs, frozen base and surface; never the name."""
        if self is other:
            return True
        return (
            self.algorithm == other.algorithm
            and self.inputs == other.inputs
            and self._same_base(other)
            and self._same_surface(other)
        )

    def _same_base(self, other):
        return (
            (self.spot, self.rate, self.dividend_yield)
            == (other.spot, other.rate, other.dividend_yield)
            and self.base_samples == other.base_samples
            and self.bands == other.bands
        )

    def _same_surface(self, other):
        return (
            self.surface.spots == other.surface.spots
            and self.surface.times == other.surface.times
            and self.surface.vols == other.surface.vols
        )


def calibrate_dupire(base, inputs, name=""):
    """Mirror native CalibrateDupireWithRisk onto the densified inclusion grid."""
    if not isinstance(inputs, DupireRiskInputs):
        raise InvalidDupirePullback(
            _input_context(inputs, "DupireCalibration_New; inputs", "DupireRiskInputs_")
        )
    label = _setting_string(name, "DupireCalibration_New; name")
    ivs = _base_ivs(base)
    _validate_inputs(inputs)
    spots = tuple(_fill_data(inputs.inclusion_spots, inputs.max_spot_spacing, _SPOT_FILL_MARGIN))
    merged_times = _set_union_first_wins(
        inputs.inclusion_times, inputs.max_time_spacing, _ONE_HOUR_YF
    )
    times = tuple(_fill_data(merged_times, inputs.max_time_spacing, _ONE_HOUR_YF))
    _validate_axis(spots, "completedSpots")
    _validate_axis(times, "completedTimes")
    samples, bands = _freeze_base(ivs, spots, times)
    vols = _calibrate_grid(
        samples, (ivs.spot, ivs.rate, ivs.dividend_yield), spots, times, bands, _spread_view(inputs)
    )
    return DupireCalibrationSnapshot(
        inputs=inputs,
        spot=ivs.spot,
        rate=ivs.rate,
        dividend_yield=ivs.dividend_yield,
        surface=LocalVolSurface(name=label, spots=spots, times=times, vols=vols),
        base_samples=tuple(sorted(samples.items())),
        bands=tuple(bands),
    )


#  ---------------------------------------------------------------------------
#  The quote pullback: the analytic VJP of the calibration through the frozen
#  base, mirroring native PullbackDupireCalibration's AAD replay.  The frozen
#  implied volatilities are constants; quote spreads enter only through the
#  bilinear RiskView interpolation, so the adjoint chain is the call vega, the
#  Dupire stencil partials and the interpolation weights.


@dataclass(frozen=True, slots=True, kw_only=True)
class DupireParameterAdjoints:
    """Surface-node seeds, mirroring native ``DupireParameterAdjoints_``."""

    calibration: DupireCalibrationSnapshot
    adjoints: tuple

    def __post_init__(self):
        if not isinstance(self.calibration, DupireCalibrationSnapshot):
            raise InvalidDupirePullback(
                _input_context(
                    self.calibration, "DupireAdjoints; calibration", "DupireCalibrationSnapshot_"
                )
            )
        object.__setattr__(self, "adjoints", _adjoints_matrix(self.adjoints))


@dataclass(frozen=True, slots=True, kw_only=True)
class DupireDirectQuoteAdjoints:
    """Additive quote-space seeds, mirroring native ``DupireDirectQuoteAdjoints_``."""

    calibration: DupireCalibrationSnapshot
    adjoints: tuple

    def __post_init__(self):
        if not isinstance(self.calibration, DupireCalibrationSnapshot):
            raise InvalidDupirePullback(
                _input_context(
                    self.calibration, "DupireAdjoints; calibration", "DupireCalibrationSnapshot_"
                )
            )
        object.__setattr__(self, "adjoints", _adjoints_matrix(self.adjoints))


@dataclass(frozen=True, slots=True)
class DupireQuoteRisk:
    """Quote-space adjoints, mirroring native ``DupireQuoteRisk_``."""

    calibration: DupireCalibrationSnapshot
    calibration_adjoints: tuple
    direct_adjoints: tuple
    total_adjoints: tuple

    method = "NativeAADCalibrationVJP"
    unit = "decimal-vol"
    boundary = "FixedBaseIVSDeterministicCarryFixedGrids"


def _adjoints_matrix(value):
    """Native RequiredInput<DoubleMatrix_>: nested numeric rows; shape checked later."""
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(row, (list, tuple)) for row in value
    ):
        raise InvalidDupirePullback(
            _input_context(value, "DupireAdjoints; adjoints", "DoubleMatrix_")
        )
    rows = []
    for row in value:
        converted = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise InvalidDupirePullback(
                    _input_context(item, "DupireAdjoints; adjoints", "DoubleMatrix_")
                )
            converted.append(float(item))
        rows.append(tuple(converted))
    return tuple(rows)


def _npdf(z):
    return 0.0 if z < -10.0 or 10.0 < z else math.exp(-0.5 * z * z) / _SQRT_2_PI


def _call_vega(carry, strike, maturity, vol):
    """The native call's derivative with respect to its bumped implied volatility."""
    spot, rate, dividend = carry
    forward = spot * math.exp((rate - dividend) * maturity)
    discount = math.exp(-rate * maturity)
    std = vol * math.sqrt(maturity)
    if -_DAL_EPSILON < std < _DAL_EPSILON or forward * strike < _DAL_EPSILON:
        return 0.0
    d_minus = math.log(forward / strike) / std - 0.5 * std
    d_plus = d_minus + std
    dd_minus = -math.log(forward / strike) / (std * std) - 0.5
    dd_plus = dd_minus + 1.0
    d_price = forward * _npdf(d_plus) * dd_plus - strike * _npdf(d_minus) * dd_minus
    return discount * d_price * math.sqrt(maturity)


def _interp1_weights(xs, x0):
    index = bisect_left(xs, x0)
    if index == len(xs):
        return [(len(xs) - 1, 1.0)]
    if index == 0 or _is_zero(x0 - xs[index]):
        return [(index, 1.0)]
    frac = (x0 - xs[index - 1]) / (xs[index] - xs[index - 1])
    return [(index - 1, 1.0 - frac), (index, frac)]


def _interp2_weights(strikes, maturities, strike, maturity):
    """The bilinear quote weights, branch-for-branch as the value interpolation."""
    upper = bisect_right(strikes, strike)
    if upper == len(strikes):
        strike_weights = [(upper - 1, 1.0)]
    elif upper == 0:
        strike_weights = [(0, 1.0)]
    else:
        lower = upper - 1
        weight = (strike - strikes[lower]) / (strikes[upper] - strikes[lower])
        strike_weights = [(lower, 1.0 - weight), (upper, weight)]
    maturity_weights = _interp1_weights(maturities, maturity)
    return [
        ((i, j), w_strike * w_maturity)
        for i, w_strike in strike_weights
        for j, w_maturity in maturity_weights
    ]


def _stencil_gradient(samples, carry, inputs, strike, maturity, spread):
    """d(local vol)/d(quote spread cell) for one in-band node, as native's replay."""
    points = _stencil_points(strike, maturity)
    calls = [_checked_call(samples, carry, point, spread) for point in points]
    ds = 1e-4 * strike
    dt = 1e-4 * maturity
    rate, dividend = carry[1], carry[2]
    ct = (calls[2] - calls[1]) * 0.5 / dt
    ck = (calls[4] - calls[3]) * 0.5 / ds
    numerator = ct + dividend * calls[0] + (rate - dividend) * strike * ck
    curvature = (calls[3] + calls[4] - 2.0 * calls[0]) / ds / ds
    local = math.sqrt(2.0 * numerator / curvature) / strike
    d_numerator = (
        dividend,
        -0.5 / dt,
        0.5 / dt,
        -(rate - dividend) * strike * 0.5 / ds,
        (rate - dividend) * strike * 0.5 / ds,
    )
    d_curvature = (-2.0 / ds / ds, 0.0, 0.0, 1.0 / ds / ds, 1.0 / ds / ds)
    gradient = {}
    for index, point in enumerate(points):
        d_local = local * 0.5 * (d_numerator[index] / numerator - d_curvature[index] / curvature)
        chain = d_local * _call_vega(carry, point[0], point[1], samples[point] + spread(*point))
        for cell, weight in _interp2_weights(
            inputs.quote_strikes, inputs.quote_maturities, point[0], point[1]
        ):
            gradient[cell] = gradient.get(cell, 0.0) + chain * weight
    return gradient


def pullback_dupire(snapshot, seeds, direct=None):
    """Mirror native PullbackDupireCalibration: surface seeds to quote adjoints."""
    calibrated = _validated_seeds(snapshot, seeds, "parameters")
    contribution = _validated_direct(snapshot, direct)
    adjoints = _quote_adjoint_matrix(snapshot, calibrated)
    rows = len(snapshot.inputs.quote_spreads)
    columns = len(snapshot.inputs.quote_maturities)
    total = [
        [adjoints[row][column] + contribution[row][column] for column in range(columns)]
        for row in range(rows)
    ]
    _require_finite(adjoints)
    _require_finite(total)
    return DupireQuoteRisk(
        calibration=snapshot,
        calibration_adjoints=tuple(tuple(row) for row in adjoints),
        direct_adjoints=tuple(tuple(row) for row in contribution),
        total_adjoints=tuple(tuple(row) for row in total),
    )


def _quote_adjoint_matrix(snapshot, seeds):
    """The VJP itself: seed-weighted stencil gradients, band edges flat-filled."""
    carry = (snapshot.spot, snapshot.rate, snapshot.dividend_yield)
    samples = dict(snapshot.base_samples)
    spread = _spread_view(snapshot.inputs)
    rows = len(snapshot.inputs.quote_spreads)
    columns = len(snapshot.inputs.quote_maturities)
    adjoints = [[0.0] * columns for _ in range(rows)]
    for column, maturity in enumerate(snapshot.times):
        low, high = snapshot.bands[column]
        gradients = {
            row: _stencil_gradient(
                samples, carry, snapshot.inputs, snapshot.spots[row], maturity, spread
            )
            for row in range(low, high)
        }
        for row in range(len(snapshot.spots)):
            seed = seeds[row][column]
            source = gradients[min(max(row, low), high - 1)]
            for cell, weight in source.items():
                adjoints[cell[0]][cell[1]] += seed * weight
    return adjoints


def _require_finite(matrix):
    if any(not math.isfinite(cell) for row in matrix for cell in row):
        raise InvalidDupirePullback("non-finite total quote adjoint")


def _validated_seeds(snapshot, seeds, field):
    """Native ValidatedDirectSeeds: calibration match, shape and finiteness."""
    if not isinstance(seeds, DupireParameterAdjoints):
        raise InvalidDupirePullback(
            _input_context(seeds, f"DupireQuoteRisk; {field}", "DupireParameterAdjoints_")
        )
    if not snapshot.matches(seeds.calibration):
        raise DupireSnapshotMismatch("parameter seed calibration does not match")
    return _checked_matrix(
        seeds.adjoints, snapshot.surface.vols, f"seed dimensions disagree; field={field}", field
    )


def _validated_direct(snapshot, direct):
    if direct is None:
        return _zero_quote_matrix(snapshot.inputs)
    if not isinstance(direct, DupireDirectQuoteAdjoints):
        raise InvalidDupirePullback(
            _input_context(direct, "DupireQuoteRisk; direct", "DupireDirectQuoteAdjoints_ or None")
        )
    if not _same_quotes(snapshot.inputs, direct.calibration.inputs):
        raise DupireSnapshotMismatch("direct quote definition does not match")
    return _checked_matrix(
        direct.adjoints,
        snapshot.inputs.quote_spreads,
        "seed dimensions disagree; field=directQuotes",
        "directQuotes",
    )


def _checked_matrix(matrix, expected, dimension_detail, field):
    if len(matrix) != len(expected) or any(len(row) != len(expected[0]) for row in matrix):
        raise InvalidDupirePullback(dimension_detail)
    for row in matrix:
        for seed in row:
            if not math.isfinite(seed):
                raise InvalidDupirePullback(f"non-finite seed; field={field}")
    return [list(row) for row in matrix]


def _zero_quote_matrix(inputs):
    return [[0.0] * len(inputs.quote_maturities) for _ in inputs.quote_spreads]


def _same_quotes(lhs, rhs):
    return (
        lhs.quote_strikes == rhs.quote_strikes
        and lhs.quote_maturities == rhs.quote_maturities
        and lhs.quote_spreads == rhs.quote_spreads
    )


#  ---------------------------------------------------------------------------
#  Recalibration on bumped quotes, mirroring native RecalibrateDupireWithRisk:
#  the frozen base and bands are kept, only the spread view changes.


def recalibrate_dupire(snapshot, quote_spreads):
    """Recalibrate on bumped spreads over the frozen base, as native."""
    replaced, carry, samples, spread = _check_recalibration(snapshot, quote_spreads)
    vols = _calibrate_grid(samples, carry, snapshot.spots, snapshot.times, snapshot.bands, spread)
    return DupireCalibrationSnapshot(
        inputs=replaced,
        spot=snapshot.spot,
        rate=snapshot.rate,
        dividend_yield=snapshot.dividend_yield,
        surface=LocalVolSurface(
            name=snapshot.surface.name, spots=snapshot.spots, times=snapshot.times, vols=vols
        ),
        base_samples=snapshot.base_samples,
        bands=snapshot.bands,
    )


def _check_recalibration(snapshot, quote_spreads):
    """Native ValidateDupireQuoteRecalibration: shape, finiteness, then domains."""
    replaced = DupireRiskInputs(
        quote_strikes=snapshot.inputs.quote_strikes,
        quote_maturities=snapshot.inputs.quote_maturities,
        quote_spreads=[[float(value) for value in row] for row in _spreads_input(quote_spreads)],
        inclusion_spots=snapshot.inputs.inclusion_spots,
        max_spot_spacing=snapshot.inputs.max_spot_spacing,
        inclusion_times=snapshot.inputs.inclusion_times,
        max_time_spacing=snapshot.inputs.max_time_spacing,
    )
    _validate_inputs(replaced)
    carry = (snapshot.spot, snapshot.rate, snapshot.dividend_yield)
    samples = dict(snapshot.base_samples)
    spread = _spread_view(replaced)
    _calibrate_grid(samples, carry, snapshot.spots, snapshot.times, snapshot.bands, spread)
    return replaced, carry, samples, spread
