"""Script risks through history, JAX transforms and parallel event scans."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from script_cases import CASES, MATURITY
from script_cases import EVALUATION as TODAY

from dal_jax import MonteCarloEngine, MonteCarloSettings, prepare
from dal_jax.api import BSModelData_New, Product_New
from dal_jax.script import ast as A


def scanned_product():
    end = TODAY.add_days(90)
    dates = ["SCALE", TODAY, f"START: {TODAY} END: {end} FREQ: 1W", end]
    events = ["2", "x=0", "IF SPOT()>100:2 THEN x=x+SCALE ELSE x=x+1 END", "pay PAYS x"]
    return prepare(Product_New(dates, events), TODAY).path_product()


def engine(product, **settings):
    return MonteCarloEngine(
        product,
        BSModelData_New(100, 0.2, 0.05, 0.02),
        MonteCarloSettings(enable_aad=True, block_size=128, **settings),
    )


def test_autocall_preparation_retains_continuous_fuzzy_kernels():
    product = prepare(Product_New(*CASES["autocall"]), TODAY)
    comparisons = [
        node
        for event in product.fuzzy_events
        for statement in event
        for node in A.walk(statement)
        if isinstance(node, A.Comparison)
    ]
    assert comparisons and all(not node.is_discrete for node in comparisons)


@pytest.mark.parametrize("strategy", ["none", "shard_map"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_historical_parameter_gradient_uses_hard_conditions_and_discards_pays(
    strategy, deterministic
):
    past = TODAY.add_days(-1)
    data = Product_New(
        ["SCALE", past, MATURITY],
        ["0.025", "IF SCALE>0:0.2 THEN x=SCALE*SPOT() ELSE x=7 END pay PAYS 100", "pay PAYS x"],
    )
    prepared = prepare(data, TODAY, historical_spots={past: 101.0})
    eng = engine(prepared.path_product(), parallel=strategy, deterministic_reduction=deterministic)
    result = eng.value(257)
    expected = 101 * np.exp(-0.05)
    assert result["PV"] == pytest.approx(0.025 * expected, abs=1e-14)
    assert result["d_SCALE"] == pytest.approx(expected, abs=1e-13)
    f = jax.jit(
        lambda scale: eng.pricer(257)(
            {"model": eng.default_params()["model"], "script": {"SCALE": scale}}
        )[0]
    )
    h = 1e-6
    assert result["d_SCALE"] == pytest.approx(
        float((f(0.025 + h) - f(0.025 - h)) / (2 * h)), rel=1e-10
    )
    assert float(f(-0.025)) == pytest.approx(7 * np.exp(-0.05), abs=1e-14)


@pytest.mark.parametrize("strategy", ["none", "shard_map", "auto", "pmap"])
def test_scanned_script_greeks_agree_across_parallel_strategies(cpu_devices, strategy):
    product = scanned_product()
    reference = engine(product, parallel="none", devices=cpu_devices[:1]).value(513)
    result = engine(product, parallel=strategy, devices=cpu_devices).value(513)
    for name in reference:
        np.testing.assert_allclose(
            result[name], reference[name], rtol=1e-13, atol=1e-13, err_msg=name
        )


def test_scanned_script_deterministic_reduction_is_bitwise_invariant(cpu_devices):
    product = scanned_product()
    a = engine(product, devices=cpu_devices[:1], deterministic_reduction=True).value(513)
    b = engine(product, devices=cpu_devices, deterministic_reduction=True).value(513)
    assert a == b


def test_scanned_script_float32_values_and_greeks():
    product = scanned_product()
    a = engine(product, parallel="none").value(513)
    b = engine(product, parallel="none", dtype="float32").value(513)
    for name in a:
        np.testing.assert_allclose(b[name], a[name], rtol=1e-4, atol=1e-5, err_msg=name)


def test_scanned_script_composes_with_jax_transforms():
    eng = engine(scanned_product(), parallel="none", smoothing_kernel="smoothstep")
    params = eng.default_params()
    f = eng.pricer(129)
    reverse = jax.jit(jax.jacrev(f))(params)
    forward = jax.jit(jax.jacfwd(f))(params)
    for a, b in zip(jax.tree.leaves(reverse), jax.tree.leaves(forward)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    by_scale = lambda scale: f(params | {"script": {"SCALE": scale}})[0]
    _, tangent = jax.jvp(by_scale, (2.0,), (1.0,))
    np.testing.assert_allclose(tangent, reverse["script"]["SCALE"][0], rtol=1e-13)
    gamma = jax.jit(
        jax.hessian(lambda spot: f(params | {"model": params["model"] | {"spot": spot}})[0])
    )(100.0)
    assert np.isfinite(float(gamma))
    np.testing.assert_allclose(
        jax.jit(jax.vmap(by_scale))(jnp.asarray([1.0, 2.0, 3.0])),
        [float(by_scale(x)) for x in (1.0, 2.0, 3.0)],
        rtol=1e-13,
    )
