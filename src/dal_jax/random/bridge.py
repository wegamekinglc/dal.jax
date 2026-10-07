"""Port of DAL's ``BrownianBridgeTransform_`` and ``FactorBrownianBridge_``.

DAL builds the bridge on unit steps ``t_i = i + 1`` (not the event times) and
returns normalised increments, so the transform maps N(0, I) to N(0, I) while
moving the first Sobol coordinates onto the coarsest path structure.  The
construction tables are computed on the host; ``apply`` is a single-path
function.
"""

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from dal_jax.errors import InvalidBrownianBridge

#  Up to this many steps the recursion is unrolled with static indices; longer
#  bridges use a scan so the traced program size stays constant.
UNROLL_MAX_STEPS = 64


@dataclass(frozen=True, slots=True)
class BridgePlan:
    n: int
    bridge_index: tuple[int, ...]
    left_index: tuple[int, ...]
    right_index: tuple[int, ...]
    left_weight: tuple[float, ...]
    right_weight: tuple[float, ...]
    std_dev: tuple[float, ...]
    sqrt_dt: tuple[float, ...]


def _interpolation(t: list[float], j: int, k: int, l: int) -> tuple[float, float, float]:
    """Weights and conditional std of W(t_l) given W(t_{j-1}) (W(0) = 0 when j == 0) and W(t_k)."""
    if j != 0:
        return (
            (t[k] - t[l]) / (t[k] - t[j - 1]),
            (t[l] - t[j - 1]) / (t[k] - t[j - 1]),
            math.sqrt(((t[l] - t[j - 1]) * (t[k] - t[l])) / (t[k] - t[j - 1])),
        )
    return (t[k] - t[l]) / t[k], t[l] / t[k], math.sqrt(t[l] * (t[k] - t[l]) / t[k])


@functools.cache
def bridge_plan(n: int) -> BridgePlan:
    """``BrownianBridgeTransform_::Initialize`` for ``n`` unit steps."""
    if n <= 0:
        raise InvalidBrownianBridge("dimension must be positive")
    t = [float(i + 1) for i in range(n)]
    sqrt_dt = [math.sqrt(t[0])] + [math.sqrt(t[i] - t[i - 1]) for i in range(1, n)]
    bridge_index = [0] * n
    left_index = [0] * n
    right_index = [0] * n
    left_weight = [0.0] * n
    right_weight = [0.0] * n
    std_dev = [0.0] * n

    used = [0] * n
    used[n - 1] = 1
    bridge_index[0] = n - 1
    std_dev[0] = math.sqrt(t[n - 1])
    j = 0
    for i in range(1, n):
        while used[j] != 0:
            j += 1
        k = j
        while used[k] == 0:
            k += 1
        l = j + ((k - 1 - j) >> 1)
        used[l] = i
        bridge_index[i] = l
        left_index[i] = j
        right_index[i] = k
        left_weight[i], right_weight[i], std_dev[i] = _interpolation(t, j, k, l)
        j = k + 1
        if j >= n:
            j = 0
    return BridgePlan(n, tuple(bridge_index), tuple(left_index), tuple(right_index),
                      tuple(left_weight), tuple(right_weight), tuple(std_dev), tuple(sqrt_dt))


def _path_unrolled(plan: BridgePlan, z: Array) -> Array:
    w: list[Array | None] = [None] * plan.n
    w[plan.n - 1] = plan.std_dev[0] * z[0]
    for i in range(1, plan.n):
        j, k, l = plan.left_index[i], plan.right_index[i], plan.bridge_index[i]
        if j != 0:
            w[l] = plan.left_weight[i] * w[j - 1] + plan.right_weight[i] * w[k] + plan.std_dev[i] * z[i]
        else:
            w[l] = plan.right_weight[i] * w[k] + plan.std_dev[i] * z[i]
    return jnp.stack(w)


def _path_scan(plan: BridgePlan, z: Array) -> Array:
    w0 = jnp.zeros(plan.n, dtype=z.dtype).at[plan.n - 1].set(plan.std_dev[0] * z[0])
    left = np.asarray(plan.left_index[1:])
    xs = (
        np.asarray(plan.bridge_index[1:]),
        np.maximum(left - 1, 0),
        left != 0,
        np.asarray(plan.right_index[1:]),
        np.asarray(plan.left_weight[1:]),
        np.asarray(plan.right_weight[1:]),
        np.asarray(plan.std_dev[1:]),
        z[1:],
    )

    def step(w, x):
        l, jm1, has_left, k, lw, rw, sd, zi = x
        value = jnp.where(has_left, lw * w[jm1], 0.0) + rw * w[k] + sd * zi
        return w.at[l].set(value), None

    w, _ = jax.lax.scan(step, w0, xs)
    return w


def apply(plan: BridgePlan, z: Array) -> Array:
    """Map ``n`` independent normals to bridged, normalised increments (one path)."""
    z = jnp.asarray(z)
    if z.shape != (plan.n,):
        raise InvalidBrownianBridge(f"input dimension mismatch: expected ({plan.n},), got {z.shape}")
    w = _path_unrolled(plan, z) if plan.n <= UNROLL_MAX_STEPS else _path_scan(plan, z)
    previous = jnp.concatenate([jnp.zeros(1, dtype=w.dtype), w[:-1]])
    return (w - previous) / jnp.asarray(plan.sqrt_dt, dtype=w.dtype)


def apply_factors(plan: BridgePlan, z: Array, n_factors: int) -> Array:
    """``FactorBrownianBridge_``: input ``factor * N + c`` maps to output ``step * F + factor``."""
    z = jnp.asarray(z)
    if n_factors <= 0 or z.shape != (plan.n * n_factors,):
        raise InvalidBrownianBridge("dimension must be a positive multiple of the factor count")
    if n_factors == 1:
        return apply(plan, z)
    per_factor = jax.vmap(lambda zf: apply(plan, zf))(z.reshape(n_factors, plan.n))
    return per_factor.T.reshape(-1)
