"""Smoothed condition kernels, a port of DAL's ``script/visitor/smoothing.hpp``."""

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


def smoothstep_cspr_bounds(x: ArrayLike, lb: float, rb: float) -> Array:
    """Bounded C1 call spread; zero slope at each end of the transition."""
    t = jnp.clip((jnp.asarray(x) - lb) / (rb - lb), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smoothstep_cspr(x: ArrayLike, eps: float) -> Array:
    return smoothstep_cspr_bounds(x, -0.5 * eps, 0.5 * eps)


def smoothstep_bfly_bounds(x: ArrayLike, lb: float, rb: float) -> Array:
    """C1 butterfly, including at its peak; asymmetric bounds are supported."""
    x = jnp.asarray(x)
    left = smoothstep_cspr_bounds(x, lb, 0.0)
    right = 1.0 - smoothstep_cspr_bounds(x, 0.0, rb)
    return jnp.where(x < 0.0, left, right)


def smoothstep_bfly(x: ArrayLike, eps: float) -> Array:
    return smoothstep_bfly_bounds(x, -0.5 * eps, 0.5 * eps)
