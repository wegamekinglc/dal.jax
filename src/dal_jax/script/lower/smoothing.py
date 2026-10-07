"""Smoothed condition kernels, a port of DAL's ``script/visitor/smoothing.hpp``.

Fuzzy evaluation replaces a hard ``x > 0`` / ``x >= 0`` by the call-spread
``CSpr`` and ``x = 0`` by the butterfly ``BFly``; that keeps the pathwise
derivative of a discontinuous payoff (e.g. with respect to a barrier level).
The two-argument forms take explicit ``lb`` / ``rb`` bounds.
"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def cspr(x: ArrayLike, eps: float) -> Array:
    """0 below ``-eps/2``, 1 above ``eps/2``, linear in between."""
    x = jnp.asarray(x)
    half = 0.5 * eps
    return jnp.where(x < -half, 0.0, jnp.where(x > half, 1.0, (x + half) / eps))


def cspr_bounds(x: ArrayLike, lb: float, rb: float) -> Array:
    x = jnp.asarray(x)
    return jnp.where(x < lb, 0.0, jnp.where(x > rb, 1.0, (x - lb) / (rb - lb)))


def bfly(x: ArrayLike, eps: float) -> Array:
    """Triangle of height 1 at 0 and half-width ``eps/2``."""
    x = jnp.asarray(x)
    half = 0.5 * eps
    return jnp.where((x < -half) | (x > half), 0.0, (half - jnp.abs(x)) / half)


def bfly_bounds(x: ArrayLike, lb: float, rb: float) -> Array:
    x = jnp.asarray(x)
    inside = jnp.where(x < 0.0, 1.0 - x / lb, 1.0 - x / rb)
    return jnp.where((x < lb) | (x > rb), 0.0, inside)
