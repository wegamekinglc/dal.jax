"""Parallel consistency: device count, strategy and block size must not move the result."""

import jax
import numpy as np
import pytest
from support import bs_model, european_call, monthly_barrier_timeline, up_and_out_call

from dal_jax import MonteCarloEngine, MonteCarloSettings

N_PATHS = 2**13 + 1000  # deliberately not a multiple of the block size


@pytest.fixture(scope="module")
def barrier():
    return up_and_out_call(monthly_barrier_timeline(1), tuple(range(1, 13)) + (12,))


def run(product, devices, **settings):
    eng = MonteCarloEngine(
        product,
        bs_model(),
        MonteCarloSettings(devices=devices, block_size=1024, enable_aad=True, **settings),
    )
    return eng.value(N_PATHS)


def assert_close(a, b, rtol=1e-13):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=rtol, atol=1e-15, err_msg=key)


def test_one_device_against_four(barrier, cpu_devices):
    assert_close(run(barrier, cpu_devices[:1]), run(barrier, cpu_devices))


@pytest.mark.parametrize("strategy", ["none", "auto", "pmap"])
def test_strategies_agree_with_shard_map(barrier, cpu_devices, strategy):
    assert_close(
        run(barrier, cpu_devices, parallel=strategy),
        run(barrier, cpu_devices, parallel="shard_map"),
    )


@pytest.mark.parametrize("block_size", [512, 2048, 2**13 + 1000])
def test_block_size_only_reorders_the_sum(barrier, cpu_devices, block_size):
    eng = MonteCarloEngine(
        barrier,
        bs_model(),
        MonteCarloSettings(devices=cpu_devices, block_size=block_size, enable_aad=True),
    )
    assert_close(eng.value(N_PATHS), run(barrier, cpu_devices))


def test_prng_streams_do_not_depend_on_device_count(cpu_devices):
    product = european_call()
    assert_close(run(product, cpu_devices[:1], rsg="mrg32"), run(product, cpu_devices, rsg="mrg32"))


def test_deterministic_reduction_is_bitwise_invariant(barrier, cpu_devices):
    results = [
        run(barrier, cpu_devices[:1], deterministic_reduction=True, parallel="none"),
        run(barrier, cpu_devices[:1], deterministic_reduction=True),
        run(barrier, cpu_devices[:2], deterministic_reduction=True),
        run(barrier, cpu_devices[:3], deterministic_reduction=True),
        run(barrier, cpu_devices, deterministic_reduction=True),
    ]
    for other in results[1:]:
        assert other == results[0]
    assert_close(results[0], run(barrier, cpu_devices))


def test_deterministic_reduction_supports_reverse_mode_on_the_pure_pricer(barrier, cpu_devices):
    eng = MonteCarloEngine(
        barrier,
        bs_model(),
        MonteCarloSettings(devices=cpu_devices, block_size=1024, deterministic_reduction=True),
    )
    f = eng.pricer(N_PATHS, fuzzy=True)
    params = eng.default_params()
    grads = jax.jit(jax.grad(lambda p: f(p)[0]))(params)
    reference = run(barrier, cpu_devices)
    np.testing.assert_allclose(
        float(grads["script"]["BARRIER"]), reference["d_BARRIER"], rtol=1e-13
    )
