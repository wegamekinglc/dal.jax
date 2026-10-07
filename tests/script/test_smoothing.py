"""C1 kernels preserve bounded degrees and remove slopes at joins and peaks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dal_jax import EvalContext
from dal_jax.script.lower import smoothing as S
from test_fuzzy import path, payoff


@pytest.mark.parametrize("kernel,joins,values", [(S.smoothstep_cspr, [-1., 1.], [0., 1.]), (S.smoothstep_bfly, [-1., 0., 1.], [0., 1., 0.])])
def test_c1_values_and_slopes_at_joins(kernel, joins, values):
    evaluate = lambda x: kernel(x, 2.0)
    np.testing.assert_allclose(jax.vmap(evaluate)(jnp.asarray(joins)), values, atol=1e-15)
    derivative = jax.vmap(jax.grad(evaluate))
    np.testing.assert_allclose(derivative(jnp.asarray(joins)), 0, atol=1e-15)
    np.testing.assert_allclose(derivative(jnp.asarray(joins) - 1e-9), derivative(jnp.asarray(joins) + 1e-9), atol=1e-7)
    degrees = jax.vmap(evaluate)(jnp.linspace(-2., 2., 101))
    assert np.all((degrees >= 0) & (degrees <= 1))


@pytest.mark.parametrize("kernel", [S.smoothstep_cspr_bounds, S.smoothstep_bfly_bounds])
def test_asymmetric_c1_bounds(kernel):
    evaluate = lambda x: kernel(x, -2.0, 1.0)
    np.testing.assert_allclose(jax.vmap(jax.grad(evaluate))(jnp.asarray([-2., 1.])), 0, atol=1e-15)
    assert float(evaluate(-3.0)) == 0 and float(evaluate(2.0)) == (1 if kernel is S.smoothstep_cspr_bounds else 0)


def test_script_c1_kernel_has_the_expected_first_and_second_derivative():
    f = payoff("IF SPOT()>LEVEL:2 THEN pay PAYS 1 ELSE pay PAYS 0 END", [("LEVEL", 100)])
    evaluate = lambda level: f({"script": {"LEVEL": level}}, path(100.25), EvalContext(fuzzy=True, smoothing_kernel="smoothstep"))
    assert float(jax.jit(evaluate)(100.0)) == pytest.approx(.68359375, abs=1e-14)
    assert float(jax.jit(jax.grad(evaluate))(100.0)) == pytest.approx(-.703125, abs=1e-14)
    assert float(jax.jit(jax.hessian(evaluate))(100.0)) == pytest.approx(-.375, abs=1e-14)
