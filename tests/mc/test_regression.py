"""DAL regression gold cases and independent least-squares references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dal_jax.errors import ScriptError
from dal_jax.mc.regression import Regression, select_regression, solve_regression
from dal_jax.mc.regression_device import select_device, solve_device


@pytest.mark.parametrize("degree", range(1, 9))
def test_polynomial_matches_independent_householder(degree):
    x = np.linspace(-2.0, 3.0, 400)
    y = np.exp(0.1 * x) + 0.02 * np.cos(3 * x)
    mask = np.arange(x.size) % 5 != 0
    fit = solve_regression(x, y, mask, degree)
    z = (x[mask] - fit.means[0]) / fit.sigmas[0]
    design = np.vander(z, degree + 1, increasing=True)
    ridge = np.diag(np.sqrt(np.sum(design**2, axis=0) * 1e-12))
    reference = np.linalg.lstsq(
        np.concatenate((design, ridge)), np.concatenate((y[mask], np.zeros(degree + 1))), rcond=None
    )[0]
    np.testing.assert_allclose(fit.coefficients, reference, rtol=1e-8, atol=2e-9)
    assert fit.solver == "MomentsCholesky"


@pytest.mark.parametrize(
    "count,reason",
    [(0, "ConditionPathsBelowMin"), (10, "ConditionPathsBelowMin"), (100, "SigmaFloor")],
)
def test_degenerate_regression_remains_finite(count, reason):
    x = np.ones(100) if count == 100 else np.arange(100.0)
    mask = np.arange(100) < count
    fit = solve_regression(x, np.full(100, 7.0), mask, 3)
    assert fit.reason == reason
    assert fit.coefficients == ((7.0,) if count else (0.0,))
    assert np.isfinite(fit.predict(1e300))


@pytest.mark.parametrize("states,degree", [(2, 3), (3, 6)])
def test_rank_loss_reduces_polynomial_degree(states, degree):
    x = np.tile(np.arange(states, dtype=float), 200)
    y = 1.0 + 2 * x + 3 * x * x
    fit = solve_regression(x, y, np.ones(x.size, bool), degree)
    assert fit.solver == "PivotedQR"
    assert fit.degree == states - 1
    assert fit.fallback_reason == "RankDeficient"
    np.testing.assert_allclose(fit.predict(x), y, atol=3e-13)


@pytest.mark.parametrize("features", [2, 3])
def test_cross_terms_recover_multivariate_polynomial(features):
    rng = np.random.default_rng(1024)
    x = rng.normal(size=(500, features))
    y = 2.0 + 3 * x[:, 0] - x[:, 1] + 4 * x[:, 0] * x[:, 1]
    if features == 3:
        y += 0.7 * x[:, 0] * x[:, 1] * x[:, 2]
    fit = solve_regression(x, y, np.ones(500, bool), 3)
    np.testing.assert_allclose(jax.jit(fit.predict)(x), y, atol=1e-12)
    assert fit.rank == len(fit.coefficients)


def test_multivariate_rank_loss_keeps_independent_columns():
    x = np.linspace(-2.0, 2.0, 500)
    fit = solve_regression(np.stack((x, 2 * x), axis=1), 3.0 + x * x, np.ones(500, bool), 2)
    assert fit.solver == "PivotedQR" and fit.rank == 3
    assert fit.fallback_reason == "RankDeficient"
    np.testing.assert_allclose(fit.predict(np.stack((x, 2 * x), axis=1)), 3.0 + x * x, atol=2e-12)


def test_mask_excludes_nan_without_poisoning_moments():
    x = np.arange(200.0)
    y = x * 2.0 + 3.0
    mask = x < 180
    x[~mask], y[~mask] = np.nan, np.inf
    fit = solve_regression(x, y, mask, 3)
    np.testing.assert_allclose(
        fit.predict(np.arange(180.0)), np.arange(180.0) * 2.0 + 3.0, atol=2e-9
    )
    with pytest.raises(ScriptError, match="InvalidRegressionInput"):
        solve_regression(x, y, np.ones(200, bool))


def test_validation_prefers_smallest_basis_within_one_standard_error():
    x = np.linspace(-2.0, 2.0, 500)
    target = 2.0 + x + 0.1 * np.sin(10 * x)
    vx = np.linspace(-1.99, 1.99, 400)
    fit = select_regression(
        x, target, np.ones(500, bool), 5, (vx, 2.0 + vx + 0.1 * np.sin(10 * vx), np.ones(400, bool))
    )
    assert fit.degree == 1
    assert fit.validation_mse is not None
    missing = select_regression(x, target, np.ones(500, bool), 5, (vx, vx, np.zeros(400, bool)))
    assert missing.degree == 5 and missing.validation_mse is None


def test_prediction_is_a_pure_jax_function_with_live_regressors():
    fit = Regression(coefficients=(1.0, 2.0, 3.0), means=(4.0,), sigmas=(2.0,), degree=2)
    assert jax.grad(fit.predict)(6.0) == 4.0
    assert jax.hessian(fit.predict)(6.0) == 1.5
    assert Regression().predict(jnp.asarray(5.0)) == 0.0


@pytest.mark.parametrize(
    "features,states,degree", [(1, None, 3), (1, 2, 3), (1, 3, 6), (2, None, 3), (3, None, 3)]
)
def test_device_fit_and_vmapped_bumps_match_host_solver(features, states, degree):
    rng = np.random.default_rng(1042)
    x = (
        rng.normal(size=(400, features))
        if states is None
        else (np.arange(400) % states)[:, None].astype(float)
    )
    y = 2.0 + x[:, 0] + 0.7 * x[:, 0] ** 2
    mask = np.arange(400) % 5 != 0
    host = solve_regression(x[:, 0] if features == 1 else x, y, mask, degree)
    device = jax.jit(lambda x, y: solve_device(x, y, jnp.asarray(mask), degree))(x, y)
    np.testing.assert_allclose(
        device.coefficients[: len(host.coefficients)], host.coefficients, atol=2e-11, rtol=1e-8
    )
    assert int(device.degree) == host.degree
    bumps = jnp.asarray([-0.05, 0.0, 0.05])
    batch = jax.jit(
        jax.vmap(
            lambda bump: solve_device(
                jnp.asarray(x), jnp.asarray(y) + bump, jnp.asarray(mask), degree
            )
        )
    )(bumps)
    np.testing.assert_allclose(batch.coefficients[:, 0] - device.coefficients[0], bumps, atol=2e-11)


def test_device_validation_matches_host_selection():
    x = jnp.linspace(-2.0, 2.0, 500)[:, None]
    y = 2.0 + x[:, 0] + 0.1 * jnp.sin(10 * x[:, 0])
    vx = jnp.linspace(-1.99, 1.99, 400)[:, None]
    vy = 2.0 + vx[:, 0] + 0.1 * jnp.sin(10 * vx[:, 0])
    fit = jax.jit(
        lambda x, y: select_device(x, y, jnp.ones(500, bool), 5, (vx, vy, jnp.ones(400, bool)))
    )(x, y)
    assert int(fit.degree) == 1
