"""Fuzzy scalar arithmetic, nested branches, endpoint safety and history risks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from script_cases import EVALUATION as TODAY, MATURITY

from dal_jax import EvalContext, MonteCarloEngine, MonteCarloSettings, prepare
from dal_jax.api import BSModelData_New, Product_New
from dal_jax.models.base import Scenario
from dal_jax.script import ast as A
from dal_jax.script.lower import smoothing as S
from dal_jax.script.lower.fuzzy import lower_event


def path(spot, numeraire=1.0):
    return Scenario(jnp.reshape(jnp.asarray(spot), (1,)), jnp.reshape(jnp.asarray(numeraire), (1,)), jnp.empty((1, 0)), jnp.empty((1, 0)))


def payoff(script, definitions=()):
    dates = [name for name, _ in definitions] + [MATURITY]
    texts = [str(value) for _, value in definitions] + [script]
    return prepare(Product_New(dates, texts), TODAY).path_product().payoff


def test_exact_folding_does_not_remove_the_fuzzy_transition():
    prepared = prepare(Product_New([MATURITY], ["IF 1 > 1 THEN pay PAYS 10 ELSE pay PAYS 2 END"]), TODAY)
    assert isinstance(prepared.events[0][0], A.Collect)
    assert isinstance(prepared.fuzzy_events[0][0], A.If)
    f = prepared.path_product().payoff
    assert float(f({"script": {}}, path(100.0), EvalContext())) == 2.0
    assert float(f({"script": {}}, path(100.0), EvalContext(fuzzy=True))) == 6.0


def test_prepared_depth_includes_ifs_retained_only_in_fuzzy_mode():
    script = "IF 1>1 THEN IF 1>1 THEN pay PAYS 1 ELSE pay PAYS 2 END ELSE pay PAYS 3 END"
    prepared = prepare(Product_New([MATURITY], [script]), TODAY)
    assert prepared.max_nested_ifs == 2


@pytest.mark.parametrize("comparison,kernel", [("SPOT()>100", S.cspr), ("SPOT()>=100", S.cspr), ("SPOT()=100", S.bfly)])
def test_continuous_comparisons_use_default_width(comparison, kernel):
    f = payoff(f"IF {comparison} THEN pay PAYS 1 ELSE pay PAYS 0 END")
    xs = jnp.asarray([99.0, 99.9375, 100.0, 100.0625, 101.0])
    values = jax.jit(jax.vmap(lambda x: f({"script": {}}, path(x), EvalContext(fuzzy=True, smooth=.25))))(xs)
    np.testing.assert_allclose(values, kernel(xs - 100, .25), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("op", ["AND", "OR"])
def test_fuzzy_boolean_algebra(op):
    f = payoff(f"IF SPOT() > 100:2 {op} SPOT() < 101:2 THEN pay PAYS 1 ELSE pay PAYS 0 END")
    x = 100.25
    a, b = float(S.cspr(x - 100, 2)), float(S.cspr(101 - x, 2))
    expected = a * b if op == "AND" else a + b - a * b
    assert float(f({"script": {}}, path(x), EvalContext(fuzzy=True))) == pytest.approx(expected)


def test_discrete_bounds_override_the_continuous_width():
    node = A.Sup(args=(A.Var(name="x", index=0, is_const=False),), is_discrete=True, lb=-2, rb=1, eps=10)
    event = (A.If(args=(node, A.Assign(args=(A.Var(name="p", index=1), A.Const(const_val=1)))), affected_vars=(1,)),)
    f = lower_event(event, smooth=100)
    sample = path(100.0).samples()[0]
    values = jax.vmap(lambda x: f(jnp.asarray([x, 0.0]), sample, {})[1])(jnp.asarray([-2.0, -1.0, 0.0, 1.0]))
    np.testing.assert_allclose(values, [0, 1/3, 2/3, 1], rtol=1e-14)


@pytest.mark.parametrize("suffix,smooth", [("", .2), (":0.2", .01), (";0.2", 3)])
def test_nested_fractional_state_and_parameter_derivative(suffix, smooth):
    f = payoff(f"x = SCALE * .025 IF x > 0{suffix} THEN IF x > 0{suffix} THEN y = x ELSE y = 2*x END "
               "ELSE y = 3*x END pay PAYS y", [("SCALE", 2)])
    evaluate = lambda scale: f({"script": {"SCALE": scale}}, path(100.0), EvalContext(fuzzy=True, smooth=smooth))
    value, derivative = jax.jit(jax.value_and_grad(evaluate))(2.0)
    assert float(value) == pytest.approx(.084375, abs=1e-14)
    assert float(derivative) == pytest.approx(.0265625, abs=1e-14)


@pytest.mark.parametrize("denominator,history", [("0.00000000000001", 2.5e-16), ("(-0.00000000000001)", -2.5e-16)])
def test_small_signed_divisors_retain_fuzzy_arithmetic(denominator, history):
    past = TODAY.add_days(-1)
    data = Product_New(["SCALE", past, MATURITY], ["2", "h = SPOT()", f"x = SCALE*h/{denominator} IF x>0:0.2 THEN pay PAYS x ELSE pay PAYS 0 END"])
    prepared = prepare(data, TODAY, historical_spots={past: history})
    engine = MonteCarloEngine(prepared.path_product(), BSModelData_New(100, .2), MonteCarloSettings(enable_aad=True, parallel="none"))
    result = engine.value(257)
    assert result["PV"] == pytest.approx(.0375, abs=1e-14)
    assert result["d_SCALE"] == pytest.approx(.025, abs=1e-14)
    assert result["d_rate"] == pytest.approx(-.0375, abs=1e-14)


@pytest.mark.parametrize("expression", ["LOG(SPOT()-100)", "SQRT(SPOT()-100)", "1/(SPOT()-100)", "(SPOT()-100)^0.5",
                                        "LOG(-1)", "SQRT(-1)", "1/0", "(-1)^0.5", "EXP(1000)"])
def test_inactive_branch_values_and_gradients_are_finite(expression):
    f = payoff(f"IF SPOT()>100:0.2 THEN pay PAYS SCALE*{expression} ELSE pay PAYS SCALE*SPOT() END", [("SCALE", 2)])
    evaluate = lambda spot, scale, numeraire: f({"script": {"SCALE": scale}}, path(spot, numeraire), EvalContext(fuzzy=True))
    value, gradients = jax.jit(jax.value_and_grad(evaluate, argnums=(0, 1, 2)))(90.0, 2.0, 1.0)
    assert float(value) == 180.0
    np.testing.assert_allclose(gradients, [2, 90, -180], rtol=1e-14)


def test_outer_inactive_branch_masks_nested_partial_mixtures():
    f = payoff("IF SPOT()>100:0.2 THEN IF SPOT()>90:2 THEN x=LOG(-1) ELSE x=1/0 END pay PAYS x ELSE pay PAYS SPOT() END")
    value, gradient = jax.jit(jax.value_and_grad(lambda x: f({"script": {}}, path(x), EvalContext(fuzzy=True))))(90.0)
    assert float(value) == 90.0 and float(gradient) == 1.0
