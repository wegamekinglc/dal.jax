"""Exact L'Ecuyer MRG32k3a stream matching DAL's ``PseudoRSG`` (MRG32)."""

import numbers
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from dal_jax.errors import InvalidSetting
from dal_jax.random.inverse_normal import inverse_ncdf

M1 = 4294967087
M2 = 4294944443
A12, A13 = 1403580, 810728
A21, A23 = 527612, 1370589


class Mrg32State(NamedTuple):
    """``x`` and ``y`` hold (current, lag1, lag2), matching DAL's MRG32k32a_."""

    x: Array  # int64[3]
    y: Array  # int64[3]


def mrg32_state(seed: int) -> Mrg32State:
    seed = int(seed)
    return Mrg32State(
        jnp.array([seed] * 3, dtype=jnp.int64),
        jnp.array([seed + 1] * 3, dtype=jnp.int64),
    )


def _mrg32_ks(state: Mrg32State, n: int) -> tuple[Array, Mrg32State]:
    """``n`` raw draws ``k`` in ``[1, M1]``, plus the advanced state."""
    x0, y0 = state.x, state.y

    def step(carry, _):
        x, y = carry
        x_new = (A12 * x[1] - A13 * x[2]) % M1
        y_new = (A21 * y[0] - A23 * y[2]) % M2
        diff = x_new - y_new
        k = jnp.where(diff > 0, diff, diff + M1)
        return (jnp.stack([x_new, x[0], x[1]]), jnp.stack([y_new, y[0], y[1]])), k

    (x, y), ks = jax.lax.scan(step, (x0, y0), xs=None, length=int(n))
    return ks, Mrg32State(x, y)


def mrg32_uniforms(state: Mrg32State, n: int) -> tuple[Array, Mrg32State]:
    """``n`` uniforms in DAL's order, plus the advanced state."""
    ks, new_state = _mrg32_ks(state, n)
    return ks.astype(jnp.float64) / jnp.float64(M1 + 1), new_state


class PseudoRSG:
    """DAL's ``PseudoRSG_``: streaming MRG32 uniforms with antithetic rows.

    Uniform rows alternate fresh ``u = k / NORM`` and ``1.0 - u`` (native
    ``FillUniformWith``); normals consume fresh uniforms per element with the
    precise inverse normal.  The division runs through numpy: XLA's vectorized
    float64 divide lands an ULP above the correctly rounded result, which the
    antithetic complement amplifies past parity tolerance.
    """

    NORM = M1 + 1

    def __init__(self, seed: int, ndim: int = 1, precise: bool = True) -> None:
        if not isinstance(seed, numbers.Integral) or isinstance(seed, bool):
            raise InvalidSetting(f"seed={seed!r}; expected an integer")
        if seed < 0:
            raise InvalidSetting(f"seed={seed}; expected a nonnegative MRG32 seed")
        if not isinstance(ndim, numbers.Integral) or isinstance(ndim, bool) or ndim < 1:
            raise InvalidSetting(f"ndim={ndim!r}; expected a positive integer")
        self.seed, self.ndim, self.precise = int(seed), int(ndim), bool(precise)
        self._state = mrg32_state(self.seed)
        self._anti = False
        self._fresh: np.ndarray | None = None

    def _draw(self, n: int) -> np.ndarray:
        ks, self._state = _mrg32_ks(self._state, n)
        return np.asarray(ks) / self.NORM

    def fill_uniform(self, n_rows: int) -> np.ndarray:
        rows = []
        for _ in range(int(n_rows)):
            if self._anti:
                rows.append(1.0 - self._fresh)
            else:
                self._fresh = self._draw(self.ndim)
                rows.append(self._fresh.copy())
            self._anti = not self._anti
        return np.stack(rows)

    def fill_normal(self, n_rows: int) -> np.ndarray:
        uniforms = self._draw(int(n_rows) * self.ndim)
        normals = np.asarray(
            inverse_ncdf(jnp.asarray(uniforms), precise=self.precise, polish=self.precise)
        )
        return normals.reshape(int(n_rows), self.ndim)
