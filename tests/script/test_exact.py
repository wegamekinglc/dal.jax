"""Exact scalar arithmetic, branch state and inactive-input gradient safety."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from script_cases import EVALUATION, MATURITY

from dal_jax import EvalContext
from dal_jax.api import Product_New
from dal_jax.models.base import Scenario
from dal_jax.script.preparation import prepare


def single_payoff(script, definitions=()):
    dates = [name for name, _ in definitions] + [MATURITY]
    events = [str(value) for _, value in definitions] + [script]
    return prepare(Product_New(dates, events), EVALUATION).path_product().payoff


def scenario(spot, numeraire=1.0):
    return Scenario(jnp.reshape(spot, (1,)), jnp.reshape(jnp.asarray(numeraire), (1,)), jnp.empty((1, 0)), jnp.empty((1, 0)))


@pytest.mark.parametrize("expression,expected", [
    ("SPOT()+2", 11.0), ("SPOT()-2", 7.0), ("SPOT()*2", 18.0), ("SPOT()/2", 4.5), ("SPOT()^2", 81.0),
    ("+SPOT()", 9.0), ("-SPOT()", -9.0), ("SQRT(SPOT())", 3.0), ("LOG(SPOT())", math.log(9)), ("EXP(SPOT())", math.exp(9)),
    ("MAX(1,2,SPOT(),10)", 10.0), ("MIN(12,11,SPOT(),8)", 8.0), ("MAX(1,2,3)", 3.0), ("MIN(4,3,2)", 2.0),
    ("2^3^2", 64.0), ("2^(3^2)", 512.0),
])
def test_arithmetic_and_functions(expression, expected):
    payoff = single_payoff(f"pay PAYS {expression}")
    actual = jax.jit(lambda spot: payoff({"script": {}}, scenario(spot, 2.0), EvalContext()))(jnp.asarray(9.0))
    assert float(actual) == pytest.approx(expected / 2.0, rel=1e-14)


@pytest.mark.parametrize("comparison,expected", [("SPOT()>100", [2, 2, 1]), ("SPOT()>=100", [2, 1, 1]),
                                                 ("SPOT()<100", [1, 2, 2]), ("SPOT()<=100", [1, 1, 2]),
                                                 ("SPOT()=100", [2, 1, 2]), ("SPOT()!=100", [1, 2, 1]),
                                                 ("SPOT()>95 AND SPOT()<105", [2, 1, 2]),
                                                 ("SPOT()<95 OR SPOT()>105", [1, 2, 1])])
def test_comparison_and_eager_logic(comparison, expected):
    payoff = single_payoff(f"IF {comparison} THEN pay PAYS 1 ELSE pay PAYS 2 END")
    values = jax.jit(jax.vmap(lambda spot: payoff({"script": {}}, scenario(spot), EvalContext())))(jnp.asarray([90.0, 100.0, 110.0]))
    np.testing.assert_array_equal(values, expected)


def test_branches_start_from_the_same_state_and_reset_between_paths():
    payoff = single_payoff("x = 2 y = 3 IF SPOT() > 100 THEN x = x + y y = x ELSE y = x END pay PAYS x + y")
    values = jax.jit(jax.vmap(lambda spot: payoff({"script": {}}, scenario(spot), EvalContext())))(jnp.asarray([110.0, 90.0, 110.0]))
    np.testing.assert_array_equal(values, [10.0, 4.0, 10.0])


def test_if_without_else_preserves_untouched_state():
    payoff = single_payoff("x = 2 IF SPOT() > 100 THEN x = x + 3 END pay PAYS x")
    values = jax.vmap(lambda spot: payoff({"script": {}}, scenario(spot), EvalContext()))(jnp.asarray([90.0, 110.0]))
    np.testing.assert_array_equal(values, [2.0, 5.0])


@pytest.mark.parametrize("expression,selected,derivative", [
    ("LOG(SPOT()-100)", math.log(10), 0.1), ("SQRT(SPOT()-100)", math.sqrt(10), 0.5 / math.sqrt(10)),
    ("1/(SPOT()-100)", 0.1, -0.01), ("(SPOT()-100)^0.5", math.sqrt(10), 0.5 / math.sqrt(10)),
])
def test_unused_dynamic_arithmetic_is_safe_for_values_and_gradients(expression, selected, derivative):
    payoff = single_payoff(f"IF SPOT() > 100 THEN pay PAYS SCALE*{expression} ELSE pay PAYS SCALE END", [("SCALE", 1)])

    def evaluate(spot, numeraire, scale):
        return payoff({"script": {"SCALE": scale}}, scenario(spot, numeraire), EvalContext())

    run = jax.jit(jax.vmap(jax.value_and_grad(evaluate, argnums=(0, 1, 2)), in_axes=(0, None, None)))
    values, (d_spot, d_numeraire, d_scale) = run(jnp.asarray([90.0, 100.0, 110.0]), 1.0, 1.0)
    np.testing.assert_allclose(values, [1.0, 1.0, selected], rtol=1e-14)
    np.testing.assert_allclose(d_spot, [0.0, 0.0, derivative], rtol=1e-14)
    np.testing.assert_allclose(d_numeraire, -values, rtol=1e-14)
    np.testing.assert_allclose(d_scale, values, rtol=1e-14)


@pytest.mark.parametrize("expression", ["LOG(-1)", "SQRT(-1)", "1/0", "(-1)^0.5", "EXP(1000)"])
def test_unused_literal_nan_or_infinity_does_not_contaminate_gradients(expression):
    payoff = single_payoff(f"IF SPOT() > 100 THEN pay PAYS SCALE * {expression} ELSE pay PAYS SCALE * SPOT() END", [("SCALE", 2)])
    evaluate = lambda spot, scale, numeraire: payoff({"script": {"SCALE": scale}}, scenario(spot, numeraire), EvalContext())
    value, gradients = jax.jit(jax.value_and_grad(evaluate, argnums=(0, 1, 2)))(90.0, 2.0, 1.0)
    assert float(value) == 180.0
    np.testing.assert_allclose(gradients, [2.0, 90.0, -180.0], rtol=1e-14)


def test_nested_inactive_branch_propagates_the_outer_mask():
    payoff = single_payoff("IF SPOT() > 100 THEN IF SPOT() > 50 THEN x = LOG(SPOT()-100) ELSE x = 1/(SPOT()-90) END "
                           "pay PAYS x ELSE pay PAYS SPOT() END")
    evaluate = lambda spot: payoff({"script": {}}, scenario(spot), EvalContext())
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(evaluate)))(jnp.asarray([90.0, 110.0]))
    np.testing.assert_allclose(values, [90.0, math.log(10)], rtol=1e-14)
    np.testing.assert_allclose(gradients, [1.0, 0.1], rtol=1e-14)


def test_named_parameters_update_without_repreparing_or_folding():
    payoff = single_payoff("IF SPOT()>Strike THEN pay PAYS SPOT()-STRIKE ELSE pay PAYS 0 END", [("Strike", 100)])
    evaluate = jax.jit(lambda strike: payoff({"script": {"Strike": strike}}, scenario(jnp.asarray(110.0)), EvalContext()))
    assert float(evaluate(100.0)) == 10.0
    assert float(evaluate(120.0)) == 0.0
