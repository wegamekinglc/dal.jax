import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dal_jax.errors import InvalidBrownianBridge
from dal_jax.random import bridge


def transform_matrix(n: int) -> np.ndarray:
    plan = bridge.bridge_plan(n)
    return np.asarray(jax.vmap(lambda e: bridge.apply(plan, e))(jnp.eye(n))).T


@pytest.mark.parametrize("n", [1, 2, 3, 8, 13, 36, 64, 65, 150])
def test_bridge_is_orthogonal(n):
    m = transform_matrix(n)
    np.testing.assert_allclose(m @ m.T, np.eye(n), atol=1e-12)


@pytest.mark.parametrize("n", [1, 5, 36, 100])
def test_first_coordinate_sets_terminal_value(n):
    plan = bridge.bridge_plan(n)
    z = jax.random.normal(jax.random.key(3), (n,), dtype=jnp.float64)
    out = bridge.apply(plan, z)
    np.testing.assert_allclose(float(jnp.sum(out)), math.sqrt(n) * float(z[0]), rtol=1e-12)


def test_plan_matches_dal_construction_for_eight_steps():
    plan = bridge.bridge_plan(8)
    assert plan.bridge_index == (7, 3, 1, 5, 0, 2, 4, 6)
    assert plan.left_index == (0, 0, 0, 4, 0, 2, 4, 6)
    assert plan.right_index == (0, 7, 3, 7, 1, 3, 5, 7)
    np.testing.assert_allclose(plan.std_dev[:2], [math.sqrt(8.0), math.sqrt(4.0 * 4.0 / 8.0)])


@pytest.mark.parametrize("n", [3, 20, 64, 90])
def test_scan_and_unrolled_recursions_agree(n):
    #  Same arithmetic; XLA may contract multiply-adds differently, so allow a few ulps.
    plan = bridge.bridge_plan(n)
    z = jax.random.normal(jax.random.key(n), (n,), dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(bridge._path_unrolled(plan, z)),
        np.asarray(bridge._path_scan(plan, z)),
        rtol=1e-14,
        atol=1e-15,
    )


def test_factor_bridge_layout():
    n, f = 6, 3
    plan = bridge.bridge_plan(n)
    z = jax.random.normal(jax.random.key(0), (n * f,), dtype=jnp.float64)
    out = np.asarray(bridge.apply_factors(plan, z, f))
    for factor in range(f):
        expected = np.asarray(bridge.apply(plan, z[factor * n : (factor + 1) * n]))
        np.testing.assert_array_equal(out[factor::f], expected)
    np.testing.assert_array_equal(
        np.asarray(bridge.apply_factors(plan, z[:n], 1)), np.asarray(bridge.apply(plan, z[:n]))
    )


def test_invalid_dimensions():
    with pytest.raises(InvalidBrownianBridge):
        bridge.bridge_plan(0)
    with pytest.raises(InvalidBrownianBridge):
        bridge.apply(bridge.bridge_plan(4), jnp.zeros(5))
    with pytest.raises(InvalidBrownianBridge):
        bridge.apply_factors(bridge.bridge_plan(4), jnp.zeros(10), 2)
