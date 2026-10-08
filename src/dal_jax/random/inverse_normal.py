"""Port of DAL's ``InverseNCDF`` / ``NCDF`` (``dal/math/specialfunctions.cpp``).

The default (``polish=False``) is Acklam's rational approximation only, which
is what DAL's Sobol generator uses in Monte Carlo.  ``polish=True`` adds one
Newton step against ``NCDF``: the cubic-spline approximation when
``precise=False``, ``erfc`` when ``precise=True``.  The operation order follows
the C++ code so Sobol normals agree with DAL to the last few ulps.
"""

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erfc, ndtri

M_SQRT_2 = 1.4142135623730951
M_SQRT_2_PI = 2.5066282746310002

_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_X_LOW = 0.02425
_X_HIGH = 1.0 - _X_LOW


def _tail_rational(p: Array) -> Array:
    c1, c2, c3, c4, c5, c6 = _C
    d1, d2, d3, d4 = _D
    r = jnp.sqrt(-2.0 * jnp.log(p))
    return (((((c1 * r + c2) * r + c3) * r + c4) * r + c5) * r + c6) / (
        (((d1 * r + d2) * r + d3) * r + d4) * r + 1.0
    )


def _acklam(x: Array) -> Array:
    a1, a2, a3, a4, a5, a6 = _A
    b1, b2, b3, b4, b5 = _B
    low = x < _X_LOW
    high = x > _X_HIGH
    tail = jnp.where(
        low, _tail_rational(jnp.where(low, x, 0.5)), -_tail_rational(jnp.where(high, 1.0 - x, 0.5))
    )
    z = x - 0.5
    r = z * z
    central = (
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        * z
        / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    )
    return jnp.where(low | high, tail, central)


# --- DAL's NcdfBySpline: clamped cubic spline on [MIN_SPLINE_X, 0] -----------------

_MIN_SPLINE_X = -3.734582185
_MIN_SPLINE_F = 9.47235e-05
_SPLINE_X = (
    _MIN_SPLINE_X,
    -3.347382781,
    -3.030883722,
    -2.75090681,
    -2.492289824,
    -2.243141537,
    -1.992179668,
    -1.494029881,
    -1.290815576,
    -1.120050999,
    -0.954303629,
    -0.792072249,
    -0.629093487,
    -0.460389924,
    -0.276889742,
    0.0,
)
_SPLINE_F = (
    _MIN_SPLINE_F,
    0.000408582,
    0.001219907,
    0.002972237,
    0.00634685,
    0.012444548,
    0.023176395,
    0.067583453,
    0.098383227,
    0.131345731,
    0.16996458,
    0.214158839,
    0.264643073,
    0.322617682,
    0.39093184,
    0.5,
)
_SPLINE_LHS_SLOPE = 0.000373538
_SPLINE_RHS_SLOPE = 0.39898679


def _clamped_spline_fpp(
    x: tuple[float, ...], f: tuple[float, ...], lhs: float, rhs: float
) -> tuple[float, ...]:
    """Second derivatives exactly as ``Cubic1_`` builds them with first-order boundaries."""
    n = len(x)
    fpp = [0.0] * n
    u = [0.0] * (n - 1)
    dx = x[1] - x[0]
    fpp[0] = ((f[1] - f[0]) / dx - lhs) * (3.0 / dx)
    u[0] = -0.5
    for i in range(1, n - 1):
        dx = x[i] - x[i - 1]
        d2 = x[i + 1] - x[i - 1]
        sig = dx / d2
        p = sig * u[i - 1] + 2.0
        u[i] = (sig - 1.0) / p
        temp = (f[i + 1] - f[i]) / (x[i + 1] - x[i]) - (f[i] - f[i - 1]) / dx
        fpp[i] = (6.0 * temp - dx * fpp[i - 1]) / (p * d2)
    dx = x[n - 1] - x[n - 2]
    un = (3.0 / dx) * (rhs - (f[n - 1] - f[n - 2]) / dx)
    fpp[n - 1] = (2.0 * un - fpp[n - 2]) / (2.0 + u[n - 2])
    for k in range(n - 2, -1, -1):
        fpp[k] += u[k] * fpp[k + 1]
    return tuple(fpp)


_SPLINE_FPP = _clamped_spline_fpp(_SPLINE_X, _SPLINE_F, _SPLINE_LHS_SLOPE, _SPLINE_RHS_SLOPE)


def _spline(z: Array) -> Array:
    """Numerical Recipes ``splint`` as in ``Cubic1_::operator()``; ``z`` in [MIN_SPLINE_X, 0]."""
    xs = jnp.asarray(_SPLINE_X)
    fs = jnp.asarray(_SPLINE_F)
    fpp = jnp.asarray(_SPLINE_FPP)
    n = len(_SPLINE_X)
    ge = jnp.searchsorted(xs, z, side="left")
    on_knot = (ge < n) & (xs[jnp.minimum(ge, n - 1)] == z)
    i_ge = jnp.clip(ge, 1, n - 1)
    i_lt = i_ge - 1
    h = xs[i_ge] - xs[i_lt]
    b = (z - xs[i_lt]) / h
    a = 1.0 - b
    value = (
        a * fs[i_lt]
        + b * fs[i_ge]
        - a * b * ((1.0 + a) * fpp[i_lt] + (1.0 + b) * fpp[i_ge]) * (h * h) / 6.0
    )
    return jnp.where(on_knot, fs[jnp.minimum(ge, n - 1)], value)


def _ncdf_by_spline(z: Array) -> Array:
    neg = jnp.where(z > 0.0, -z, z)
    tail = _MIN_SPLINE_F * jnp.exp(-1.1180061 * (neg * neg - _MIN_SPLINE_X * _MIN_SPLINE_X))
    lower = jnp.where(neg < _MIN_SPLINE_X, tail, _spline(jnp.maximum(neg, _MIN_SPLINE_X)))
    return jnp.where(z > 0.0, 1.0 - lower, lower)


def ncdf(z: Array, precise: bool = True) -> Array:
    """DAL's ``NCDF``: ``0.5 * erfc(-z / sqrt(2))`` or the spline approximation."""
    z = jnp.asarray(z)
    return 0.5 * erfc(-z / M_SQRT_2) if precise else _ncdf_by_spline(z)


def inverse_ncdf(x: Array, precise: bool = False, polish: bool = False) -> Array:
    """DAL's ``InverseNCDF`` for ``x`` in (0, 1).  Defaults match DAL's Sobol generator."""
    x = jnp.asarray(x)
    z = _acklam(x)
    if polish:
        err = ncdf(z, precise) - x
        z = z - err * M_SQRT_2_PI * jnp.exp(jnp.minimum(8.0, 0.5 * (z * z)))
    return z


def inverse_ncdf_ndtri(x: Array) -> Array:
    """Full-precision inverse via ``jax.scipy.special.ndtri`` (not DAL path-compatible)."""
    return ndtri(jnp.asarray(x))


__all__ = ["inverse_ncdf", "inverse_ncdf_ndtri", "ncdf", "M_SQRT_2", "M_SQRT_2_PI"]
