"""Actual GPU parity, precision, PRNG and explicit CPU/GPU placement checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from script_cases import CASES, EVALUATION, MATURITY, oracle_dates

from dal_jax import MonteCarloEngine, MonteCarloSettings, prepare
from dal_jax.api import BSModelData_New, Product_New

pytestmark = pytest.mark.gpu


def rows(case):
    if case == "autocall_long":
        end = EVALUATION.add_days(1095)
        dates = ["LEVEL", "COUPON", EVALUATION, f"START: {EVALUATION} END: {end} FREQ: 1M", end]
        texts = ["100", "5", "alive=1", "IF alive=1 THEN IF SPOT()>=LEVEL THEN pay PAYS 100+COUPON alive=0 "
                 "ELSE pay PAYS COUPON END END", "IF alive=1 THEN pay PAYS MIN(SPOT(),100) END"]
        return dates, texts
    if case.startswith("barrier"):
        end = EVALUATION.add_days(1095)
        freq = "1M" if case == "barrier_1m" else "1W"
        dates = ["STRIKE", "BARRIER", EVALUATION, f"START: {EVALUATION} END: {end} FREQ: {freq}", end]
        texts = ["120", "150", "alive=1", "IF SPOT()>=BARRIER:0.1 THEN alive=0 END",
                 "IF SPOT()>=BARRIER:0.1 THEN alive=0 END pay PAYS alive*MAX(SPOT()-STRIKE,0)"]
        return dates, texts
    return CASES[case]


def engine(case, platform="gpu", **options):
    prepared = prepare(Product_New(*rows(case)), EVALUATION)
    settings = {"platform": platform, "enable_aad": True, "block_size": 512} | options
    return MonteCarloEngine(prepared.path_product(), BSModelData_New(100., .2, .05, .02),
                            MonteCarloSettings(**settings))


def compare(a, b, *, pv_rtol=1e-10, greek_rtol=1e-8, atol=1e-10):
    assert a.keys() == b.keys()
    np.testing.assert_allclose(a["PV"], b["PV"], rtol=pv_rtol, atol=atol)
    for name in a.keys() - {"PV"}:
        np.testing.assert_allclose(a[name], b[name], rtol=greek_rtol, atol=atol, err_msg=name)


@pytest.mark.parametrize("case", ["european_call", "asian", "autocall", "barrier_1m", "barrier_1w"])
@pytest.mark.parametrize("use_bb", [False, True])
def test_gpu_float64_script_prices_and_greeks_match_dal(gpu_devices, dal, case, use_bb):
    dal.EvaluationDate_Set(dal.Date_(EVALUATION.year, EVALUATION.month, EVALUATION.day))
    dates, texts = rows(case)
    product = dal.Product_New(oracle_dates(dates, dal), texts)
    reference = dict(dal.MonteCarlo_Value(product, dal.BSModelData_New(100., .2, .05, .02), 4097, "sobol", use_bb, True))
    compare(engine(case, devices=gpu_devices, use_bb=use_bb).value(4097), reference)


@pytest.mark.parametrize("strategy", ["none", "shard_map", "auto", "pmap"])
def test_gpu_parallel_strategies_preserve_sobol_results(gpu_devices, strategy):
    reference = engine("barrier_1m", devices=gpu_devices, parallel="none").value(1025)
    compare(engine("barrier_1m", devices=gpu_devices, parallel=strategy).value(1025), reference, pv_rtol=1e-13, greek_rtol=1e-13)


def test_gpu_and_cpu_float64_paths_and_all_risks_agree(gpu_devices, cpu_devices):
    cpu = engine("barrier_1m", "cpu", devices=cpu_devices, parallel="shard_map")
    gpu = engine("barrier_1m", devices=gpu_devices, parallel="shard_map")
    compare(gpu.value(4097), cpu.value(4097))
    params = gpu.default_params()
    assert all(next(iter(leaf.devices())).platform == "gpu" for leaf in jax.tree.leaves(params))
    # Compatibility API transfers caller parameters to the explicitly selected platform.
    compare(cpu.value(4097, params), cpu.value(4097))
    assert all(next(iter(leaf.devices())).platform == "cpu" for leaf in jax.tree.leaves(cpu.default_params()))


def test_gpu_deterministic_reduction_and_block_size(gpu_devices):
    a = engine("barrier_1m", devices=gpu_devices, deterministic_reduction=True).value(1025)
    b = engine("barrier_1m", devices=gpu_devices, deterministic_reduction=True).value(1025)
    assert a == b
    compare(a, engine("barrier_1m", devices=gpu_devices, block_size=1024).value(1025))


@pytest.mark.parametrize("case", ["european_call", "asian", "barrier_1m", "autocall"])
def test_gpu_float32_short_script_cases_match_float64_within_tolerance(gpu_devices, case):
    a = engine(case, devices=gpu_devices).value(8193)
    b = engine(case, devices=gpu_devices, dtype="float32").value(8193)
    # A narrow 0.1 barrier kernel amplifies float32 spot rounding in its Greeks.
    compare(b, a, pv_rtol=2e-5, greek_rtol=5e-3, atol=2e-4)


def test_gpu_long_autocall_float64_risks_match_dal(gpu_devices, dal):
    dal.EvaluationDate_Set(dal.Date_(EVALUATION.year, EVALUATION.month, EVALUATION.day))
    dates, texts = rows("autocall_long")
    product = dal.Product_New(oracle_dates(dates, dal), texts)
    model = dal.BSModelData_New(100., .15, .05, .03)
    reference = dict(dal.MonteCarlo_Value(product, model, 2**20, "sobol", False, True))
    prepared = prepare(Product_New(dates, texts), EVALUATION)
    eng = MonteCarloEngine(prepared.path_product(), BSModelData_New(100., .15, .05, .03),
                          MonteCarloSettings(platform="gpu", devices=gpu_devices, enable_aad=True))
    compare(eng.value(2**20), reference)


def test_gpu_float32_autocall_with_wide_smoothing_matches_float64(gpu_devices):
    # Width 0.01 is ill-conditioned in float32 for this repeated equality gate;
    # the million-path error is documented, not covered by a blanket tolerance.
    prepared = prepare(Product_New(*rows("autocall_long")), EVALUATION)
    values = []
    for dtype in ("float64", "float32"):
        eng = MonteCarloEngine(prepared.path_product(), BSModelData_New(100., .15, .05, .03),
                              MonteCarloSettings(platform="gpu", devices=gpu_devices, dtype=dtype,
                                                 enable_aad=True, smooth=1.))
        values.append(eng.value(2**20))
    compare(values[1], values[0], pv_rtol=2e-5, greek_rtol=5e-3, atol=2e-4)


@pytest.mark.parametrize("impl", ["threefry2x32", "rbg"])
def test_gpu_prng_blocks_are_repeatable_and_statistically_correct(gpu_devices, impl):
    product = prepare(Product_New([MATURITY], ["pay PAYS SPOT()"]), EVALUATION)
    settings = MonteCarloSettings(platform="gpu", devices=gpu_devices, rsg="mrg32", prng_impl=impl, block_size=1024)
    eng = MonteCarloEngine(product.path_product(), BSModelData_New(100., .2, .05, .02), settings)
    params = eng.default_params()
    n = 2**14
    path = jax.jit(lambda block: eng.path_payoffs(params, block, n)[1][:, 0])
    first = np.asarray(path(jnp.asarray(0)))
    np.testing.assert_array_equal(first, path(jnp.asarray(0)))
    values = np.concatenate([np.asarray(path(jnp.asarray(block))) for block in range(eng.layout(n).n_blocks)])
    expected = 100*np.exp(-.02)
    assert abs(values.mean()-expected) < 3*values.std()/np.sqrt(n)


def test_gpu_auto_sizing_and_precision_are_resolved_on_the_host(gpu_devices):
    prepared = prepare(Product_New(*rows("barrier_1w")), EVALUATION)
    eng = MonteCarloEngine(prepared.path_product(), BSModelData_New(100., .2),
                          MonteCarloSettings(platform="gpu", devices=gpu_devices, dtype="auto", enable_aad=True))
    assert eng.dtype == jnp.float32
    assert 256 <= eng.block_size <= 32768 and eng.block_size & (eng.block_size-1) == 0
    assert np.isfinite(eng.value(513)["PV"])


def test_gpu_scan_and_unrolled_payoff_gradients_agree(gpu_devices):
    dates, texts = rows("barrier_1m")
    texts = [text.replace(":0.1", ":0.25") for text in texts]
    product = prepare(Product_New(dates, texts), EVALUATION)
    payoff = product.path_product().payoff
    n = len(product.events)
    spots = jnp.full(n, 100.).at[jnp.asarray([2, 3, 4])].set(149.9375).at[-1].set(140.)
    from dal_jax import EvalContext, Scenario
    scenario = jax.device_put(Scenario(spots, jnp.ones(n), jnp.empty((n, 0)), jnp.empty((n, 0))), gpu_devices[0])
    level = jax.device_put(jnp.asarray(150.), gpu_devices[0])

    def evaluate(barrier, threshold):
        return payoff({"script": {"BARRIER": barrier, "STRIKE": 120.}}, scenario,
                      EvalContext(fuzzy=True, scan_group_threshold=threshold))

    a = jax.jit(jax.value_and_grad(lambda barrier: evaluate(barrier, 0)))(level)
    b = jax.jit(jax.value_and_grad(lambda barrier: evaluate(barrier, 4)))(level)
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-14)
