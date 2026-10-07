"""Fixed-capacity vectors: DAL values, fuzzy lengths, history, AD and named errors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dal_jax import BlackScholes, MonteCarloEngine, MonteCarloSettings, prepare
from dal_jax.api import Product_New
from dal_jax.dates import Date
from dal_jax.errors import EmptyVectorReduction, VectorIndexOutOfRange
from dal_jax.mc.engine import EvalContext
from dal_jax.models.base import Scenario

TODAY = Date.ymd(2022, 9, 15)


def _product(body, *, history=None):
    dates, events = ([TODAY], [body]) if history is None else (
        ["SCALE", TODAY.add_days(-1), TODAY], ["2", history, body])
    return prepare(Product_New(dates, events), TODAY)


def _path(product, spot, *, fuzzy=False):
    payoff = product.path_product().payoff
    scenario = Scenario(jnp.asarray([spot]), jnp.ones(1), jnp.empty((1, 0)), jnp.empty((1, 0)))
    params = {"script": dict(product.script_params)}
    return payoff(params, scenario, EvalContext(fuzzy=fuzzy))


@pytest.mark.parametrize("kind,expected,delta", [("SUM", 108., 1.), ("AVERAGE", 27., .25),
                                                ("MIN", 0., 0.), ("MAX", 100., 1.)])
def test_indexed_write_extends_with_zero_padding(kind, expected, delta):
    product = _product(f"v[3]=SPOT() v[1]=8 pay PAYS {kind}(v)")
    value = lambda s: _path(product, s)[0]
    assert float(value(100.)) == expected
    assert float(jax.grad(value)(100.)) == delta


def test_for_append_and_reductions_have_live_gradients():
    product = _product("FOR(i,0,3) APPEND(v,SPOT()+i) END pay PAYS AVERAGE(v)+v[2]")
    value = lambda s: _path(product, s)[0]
    np.testing.assert_allclose(jax.jit(jax.value_and_grad(value))(100.), [203., 2.], atol=0)


@pytest.mark.parametrize("spot,expected,delta", [(98., 2., 0.), (99.5, 5.5, 7.), (100., 9., 7.),
                                               (100.5, 12.5, 7.), (102., 16., 0.)])
def test_fuzzy_vector_length_is_max_only_inside_transition(spot, expected, delta):
    product = _product("IF SPOT()>100:2 THEN APPEND(v,6) APPEND(v,10) ELSE APPEND(v,2) END pay PAYS SUM(v)")
    value = lambda s: _path(product, s, fuzzy=True)[0]
    np.testing.assert_allclose(jax.jit(jax.value_and_grad(value))(spot), [expected, delta], atol=1e-12)


def test_fuzzy_padding_is_readable_and_average_uses_blended_length():
    product = _product("IF SPOT()>100:2 THEN APPEND(v,6) APPEND(v,10) ELSE APPEND(v,2) END pay PAYS v[1]+AVERAGE(v)")
    np.testing.assert_allclose(_path(product, 100., fuzzy=True), [9.5, 0., 0.], atol=0)
    assert float(_path(product, 98., fuzzy=True)[1]) == 1.  # shorter endpoint has no entry 1


@pytest.mark.parametrize("fuzzy", [False, True])
def test_unused_branch_cannot_raise_vector_errors_or_pollute_gradients(fuzzy):
    product = _product("IF SPOT()>100:2 THEN pay PAYS SPOT() ELSE pay PAYS AVERAGE(v)+v[3] END")
    value = lambda s: _path(product, s, fuzzy=fuzzy)[0]
    np.testing.assert_allclose(jax.value_and_grad(value)(110.), [110., 1.], atol=0)
    assert np.count_nonzero(_path(product, 110., fuzzy=fuzzy)[1:]) == 0


def test_nested_fuzzy_vectors_start_each_branch_from_same_entry_state():
    product = _product("APPEND(v,1) IF SPOT()>100:2 THEN APPEND(v,4) "
                       "IF SPOT()>100:2 THEN v[0]=9 ELSE APPEND(v,8) END ELSE APPEND(v,2) END pay PAYS SUM(v)")
    # Inner true [9,4] / false [1,4,8] -> [5,4,4]; outer false [1,2].
    assert float(_path(product, 100., fuzzy=True)[0]) == 8.


def test_history_seeds_vector_values_and_named_parameter_adjoint():
    product = _product("APPEND(v,SPOT()) pay PAYS SUM(v)", history="APPEND(v,SCALE) APPEND(v,3) discarded PAYS 99")
    assert product.initial_lengths == (2,)
    engine = MonteCarloEngine(product.path_product(), BlackScholes(spot=100., vol=.15), MonteCarloSettings(enable_aad=True))
    result = engine.value(16)
    np.testing.assert_allclose([result["PV"], result["d_SCALE"], result["d_spot"]], [105., 1., 1.], atol=0)


def test_vmap_lengths_and_data_reset_between_paths():
    product = _product("IF SPOT()>100 THEN APPEND(v,SPOT()) APPEND(v,2) ELSE APPEND(v,1) END pay PAYS SUM(v)")
    values = jax.jit(jax.vmap(lambda s: _path(product, s)))(jnp.asarray([90., 110., 95., 120.]))
    np.testing.assert_array_equal(values[:, 0], [1., 112., 1., 122.])
    assert not np.any(np.asarray(values[:, 1:]))


@pytest.mark.parametrize("body,error", [("pay PAYS v[2]", VectorIndexOutOfRange),
                                         ("pay PAYS AVERAGE(v)", EmptyVectorReduction),
                                         ("pay PAYS MIN(v)", EmptyVectorReduction)])
@pytest.mark.parametrize("aad", [False, True])
def test_named_runtime_errors_cross_host_api(body, error, aad):
    engine = MonteCarloEngine(_product(body).path_product(), BlackScholes(spot=100., vol=.15),
                             MonteCarloSettings(enable_aad=aad))
    with pytest.raises(error, match="v"):
        engine.value(16)
    values, rates = jax.jit(engine.checked_pricer(16))(engine.default_params())
    assert np.any(np.asarray(rates) > 0)
    assert np.isnan(np.asarray(engine.pricer(16)(engine.default_params()))).all()
    assert values.shape == (1,)


def test_empty_sum_is_zero_and_empty_non_sum_in_history_raises():
    assert float(_path(_product("pay PAYS SUM(v)"), 100.)[0]) == 0.
    with pytest.raises(EmptyVectorReduction):
        _product("pay PAYS SPOT()", history="x=AVERAGE(v)")


def test_padding_path_errors_are_ignored():
    product = prepare(Product_New([TODAY.add_days(365)], ["IF SPOT()>105 THEN pay PAYS v[0] ELSE pay PAYS 1 END"]), TODAY)
    engine = MonteCarloEngine(product.path_product(), BlackScholes(spot=100., vol=.15),
                             MonteCarloSettings(block_size=4, block_bucketing=True))
    assert engine.value(1)["PV"] == 1.
    with pytest.raises(VectorIndexOutOfRange):
        engine.value(2)
