"""RBG must retain each block's key under nested vmaps and automatic sharding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from support import bs_model, european_call

from dal_jax import MonteCarloEngine, MonteCarloSettings
from dal_jax.random.prng import block_normals, prng_key


@pytest.mark.parametrize("impl", ["rbg", "unsafe_rbg"])
def test_rbg_batching_uses_every_block_key(impl):
    key = prng_key("mrg32", 1024, impl)
    keys = jax.random.split(key, 2)
    ids = jnp.arange(6).reshape(2, 3)
    draw = lambda stream, block: block_normals(stream, block, 32, 3)
    actual = jax.jit(jax.vmap(jax.vmap(draw, in_axes=(None, 0))))(keys, ids)
    expected = jnp.stack([jnp.stack([draw(stream, block) for block in row]) for stream, row in zip(keys, ids)])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("impl", ["rbg", "unsafe_rbg"])
@pytest.mark.parametrize("strategy", ["shard_map", "auto", "pmap"])
def test_rbg_prices_and_greeks_do_not_change_with_parallel_strategy(cpu_devices, impl, strategy):
    def evaluate(parallel, devices):
        settings = MonteCarloSettings(enable_aad=True, rsg="mrg32", prng_impl=impl, block_size=128,
                                      parallel=parallel, devices=devices)
        return MonteCarloEngine(european_call(), bs_model(), settings).value(1024)

    reference = evaluate("none", cpu_devices[:1])
    result = evaluate(strategy, cpu_devices)
    for label in reference:
        np.testing.assert_allclose(result[label], reference[label], rtol=1e-13, atol=1e-13, err_msg=label)
