"""P5 scan/vector state and model adjoints across CPU strategies and transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from p5_cases import TODAY, case, compare

from dal_jax import BlackScholes, MonteCarloEngine, MonteCarloSettings, prepare
from dal_jax.api import Product_New


@pytest.mark.parametrize("name", ["vector_asian", "correlated_basket", "localvol_skew"])
@pytest.mark.parametrize("strategy", ["none", "shard_map", "auto", "pmap"])
def test_all_strategies_with_nondivisible_path_count(cpu_devices, name, strategy):
    rows, model = case(name)
    product = prepare(Product_New(*rows), TODAY, model=model).path_product()
    one = MonteCarloEngine(
        product, model, MonteCarloSettings(enable_aad=True, devices=cpu_devices[:1], block_size=128)
    )
    many = MonteCarloEngine(
        product,
        model,
        MonteCarloSettings(enable_aad=True, devices=cpu_devices, block_size=128, parallel=strategy),
    )
    compare(many.value(513), one.value(513))


def test_vector_scan_checkpoint_and_deterministic_reduction(cpu_devices):
    rows, model = case("vector_asian")
    product = prepare(Product_New(*rows), TODAY).path_product()
    values = []
    for devices, checkpoint, threshold in [
        (cpu_devices[:1], True, 4),
        (cpu_devices, True, 4),
        (cpu_devices, False, 0),
    ]:
        engine = MonteCarloEngine(
            product,
            model,
            MonteCarloSettings(
                enable_aad=True,
                devices=devices,
                block_size=128,
                deterministic_reduction=True,
                checkpoint=checkpoint,
                scan_group_threshold=threshold,
            ),
        )
        values.append(engine.value(513))
    assert values[0] == values[1]
    compare(values[2], values[0])


def test_vector_history_composes_with_forward_reverse_hessian_and_vmap(cpu_devices):
    rows = (
        ["SCALE", TODAY.add_days(-1), TODAY.add_days(365)],
        ["2", "APPEND(v,SCALE) APPEND(v,3)", "APPEND(v,SPOT()) pay PAYS SCALE*SUM(v)"],
    )
    model = BlackScholes(spot=100.0, vol=0.15, rate=0.05, div=0.03)
    product = prepare(Product_New(*rows), TODAY).path_product()
    engine = MonteCarloEngine(
        product, model, MonteCarloSettings(enable_aad=True, devices=cpu_devices, block_size=128)
    )
    base = engine.default_params()

    def price(x):
        p = base | {"script": {"SCALE": x[1]}, "model": base["model"] | {"spot": x[0]}}
        return engine.pricer(257)(p)[0]

    x = jnp.asarray([100.0, 2.0])
    a = jax.jit(jax.jacfwd(price))(x)
    b = jax.jit(jax.jacrev(price))(x)
    np.testing.assert_allclose(a, b, rtol=1e-13)
    h = np.asarray(jax.jit(jax.hessian(price))(x))
    np.testing.assert_allclose(h, h.T, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(h[1, 1], 2 * np.exp(-0.05), rtol=1e-13)
    ladder = jnp.asarray([[90.0, 2.0], [100.0, 2.0], [110.0, 2.0]])
    batched = jax.jit(jax.vmap(price))(ladder)
    np.testing.assert_allclose(batched, [price(p) for p in ladder], rtol=1e-13)
