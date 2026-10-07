"""Actual GPU validation of vector scans, dated observations and new models."""

import pytest
from p5_cases import TODAY, case, compare, native_dates, native_model

from dal_jax import MonteCarloEngine, MonteCarloSettings, prepare
from dal_jax.api import Product_New
from dal_jax.errors import VectorIndexOutOfRange

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("name", ["vector_asian", "vector_fuzzy", "dated_fix_payment", "correlated_basket", "localvol_skew"])
@pytest.mark.parametrize("bridge", [False, True])
def test_gpu_float64_values_and_all_greeks(gpu_devices, cpu_devices, dal, name, bridge):
    rows, model = case(name)
    product = prepare(Product_New(*rows), TODAY, model=model).path_product()
    settings = dict(enable_aad=True, use_bb=bridge, block_size=512)
    gpu = MonteCarloEngine(product, model, MonteCarloSettings(**settings, devices=gpu_devices, platform="gpu"))
    cpu = MonteCarloEngine(product, model, MonteCarloSettings(**settings, devices=cpu_devices, platform="cpu"))
    actual = gpu.value(1025)
    compare(actual, cpu.value(1025))
    if not hasattr(dal, "CorrelatedBSModelData_New"):
        pytest.skip("CPU/GPU passed; native P5 comparison needs the pinned source oracle")
    dal.EvaluationDate_Set(dal.Date_(TODAY.year, TODAY.month, TODAY.day))
    native = dal.MonteCarlo_Value(dal.Product_New(native_dates(dal, rows[0]), rows[1]), native_model(dal, model),
                                  1025, enable_aad=True, use_bb=bridge)
    compare(actual, dict(native))


def test_gpu_vector_error_flags_reach_host(gpu_devices):
    prepared = prepare(Product_New([TODAY], ["pay PAYS v[0]"]), TODAY)
    _, model = case("vector_asian")
    engine = MonteCarloEngine(prepared.path_product(), model, MonteCarloSettings(platform="gpu", devices=gpu_devices, enable_aad=True))
    with pytest.raises(VectorIndexOutOfRange):
        engine.value(16)
