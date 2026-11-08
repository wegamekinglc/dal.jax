"""Exact L'Ecuyer MRG32k3a stream matching DAL's ``PseudoRSG`` (MRG32)."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

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


def mrg32_uniforms(state: Mrg32State, n: int) -> tuple[Array, Mrg32State]:
    """``n`` uniforms in DAL's order, plus the advanced state."""
    ones = jnp.ones(3, dtype=jnp.int64)
    x0, y0 = state.x * ones, state.y * ones

    def step(carry, _):
        x, y = carry
        xn, xn1, xn2 = x[0], x[1], x[2]
        yn, yn1, yn2 = y[0], y[1], y[2]
        x_new = (A12 * xn1 - A13 * xn2) % M1
        y_new = (A21 * yn - A23 * yn2) % M2
        diff = x_new - y_new
        u = jnp.where(diff > 0, diff, diff + M1).astype(jnp.float64) / jnp.float64(M1 + 1)
        return (jnp.stack([x_new, xn, xn1]), jnp.stack([y_new, yn, yn1])), u

    (x, y), uniforms = jax.lax.scan(step, (x0, y0), xs=None, length=int(n))
    return uniforms, Mrg32State(x, y)
