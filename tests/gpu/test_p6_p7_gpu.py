"""Actual GPU exercise training/retraining and joint rate-model parity."""

from dataclasses import replace

import numpy as np
import pytest

import dal_jax.api as api
from dal_jax import MonteCarloSettings, prepare
from dal_jax.script.fixings import ValuationSettings
from oracle.test_dal_rates import TODAY, native_date, model_pair
from oracle.test_dal_hybrid import hybrid_pair, zero_rate_model, native_zero_slv_spot_delta

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("kind",["gsr","slv"])
@pytest.mark.parametrize("zero",[False,True])
def test_gpu_live_rate_volatility_from_zero_model(gpu_devices,cpu_devices,dal,kind,zero):
    positive,native_model = hybrid_pair(dal,kind,zero=zero)
    model = zero_rate_model(positive)
    dates = (TODAY.add_days(180),TODAY.add_days(365))
    events = ("pay PAYS MAX(FIX(EQ[A])-101,0) + 100*FIX(IR[USD,LIBOR_3M_CME])",)*2
    prepared = prepare(api.Product_New(dates,events),TODAY,model=model)
    common = dict(enable_aad=True,use_bb=True,block_size=128)
    gpu = prepared.engine(model,MonteCarloSettings(**common,platform="gpu",devices=gpu_devices))
    cpu = prepared.engine(model,MonteCarloSettings(**common,platform="cpu",devices=cpu_devices))
    params = gpu.default_params() | {"model":positive.default_params()}
    actual,expected = gpu.value(257,params),cpu.value(257,params)
    product = dal.Product_New([native_date(dal,d) for d in dates],events)
    valuation = dal.ScriptValuationSettings_(evaluation_date=native_date(dal,TODAY))
    native = dict(dal.MonteCarlo_ValueWithSettings(product,native_model,257,valuation=valuation,
                    simulation=dal.MonteCarloSettings_(enable_aad=True,use_bb=True)))
    if zero and kind == "slv":
        native["d_spot:EQ[A]"] = native_zero_slv_spot_delta(dal,product,valuation,257)
    for name in actual:
        for reference in (expected,native):
            np.testing.assert_allclose(actual[name],reference[name],rtol=1e-10 if name == "PV" else 1e-8,atol=1e-10,err_msg=name)


@pytest.mark.parametrize("kind",["gsr","multi","hybrid","local_hybrid","slv_hybrid"])
@pytest.mark.parametrize("bridge",[False,True])
def test_gpu_rate_values_and_all_greeks(gpu_devices,cpu_devices,dal,kind,bridge):
    if kind in ("gsr","multi"):
        model,nmodel = model_pair(dal,multi=kind == "multi",knots=True,projections=True)
        event = "pay PAYS MAX(FIX(IR[USD,DF,2028-10-01])-.96,0) + FIX(IR[USD,SWAP,5Y])"
    else:
        model,nmodel = hybrid_pair(dal,{"hybrid":"gsr","local_hybrid":"local_gsr","slv_hybrid":"slv"}[kind])
        event = "pay PAYS MAX(FIX(EQ[A])-101,0) + 100*FIX(IR[USD,LIBOR_3M_CME])"
    dates = (TODAY.add_days(180),TODAY.add_days(365))
    product = api.Product_New(dates,(event,)*2)
    prepared = prepare(product,TODAY,model=model)
    common = dict(enable_aad=True,use_bb=bridge,block_size=256)
    gpu = prepared.engine(model,MonteCarloSettings(**common,platform="gpu",devices=gpu_devices))
    cpu = prepared.engine(model,MonteCarloSettings(**common,platform="cpu",devices=cpu_devices))
    actual = gpu.value(513)
    expected = cpu.value(513)
    native = dal.MonteCarlo_ValueWithSettings(dal.Product_New([native_date(dal,d) for d in dates],(event,)*2),nmodel,513,
                valuation=dal.ScriptValuationSettings_(evaluation_date=native_date(dal,TODAY)),
                simulation=dal.MonteCarloSettings_(enable_aad=True,use_bb=bridge))
    for name in actual:
        for reference in (expected,native):
            np.testing.assert_allclose(actual[name],reference[name],rtol=1e-8 if name != "PV" else 1e-10,atol=1e-10,err_msg=name)


@pytest.mark.parametrize("mode",["Frozen","RetrainedBump"])
@pytest.mark.parametrize("rqmc",[False,True])
def test_gpu_lsmc_training_and_vmapped_policy_bumps(gpu_devices,cpu_devices,dal,mode,rqmc):
    model = api.BSModelData_New(100.,.2,.05,0.)
    dates = (TODAY.add_days(180),TODAY.add_days(365))
    event = "EXERCISE MAX(100-SPOT(),0)"
    product = api.Product_New(dates,(event,)*2)
    prepared = prepare(product,TODAY,model=model)
    common = dict(enable_aad=True,use_bb=True,block_size=128,lsmc_training_paths=256,lsmc_validation_paths=64,lsmc_policy_risk_mode=mode)
    if rqmc:
        common |= dict(lsmc_rqmc_replicates=2,lsmc_training_seed=17,lsmc_pricing_seed=29)
    native_options = common.copy()
    del native_options["block_size"]
    gpu = prepared.engine(model,MonteCarloSettings(**common,platform="gpu",devices=gpu_devices))
    cpu = prepared.engine(model,MonteCarloSettings(**common,platform="cpu",devices=cpu_devices))
    actual,expected = gpu.value(257),cpu.value(257)
    native = dal.MonteCarlo_ValueWithSettings(dal.Product_New([native_date(dal,d) for d in dates],(event,)*2),dal.BSModelData_New(100.,.2,.05,0.),257,
                valuation=dal.ScriptValuationSettings_(evaluation_date=native_date(dal,TODAY)),
                simulation=dal.MonteCarloSettings_(**native_options))
    for name in actual:
        for reference in (expected,native):
            np.testing.assert_allclose(actual[name],reference[name],rtol=1e-8 if name != "PV" else 1e-6,atol=1e-9,err_msg=name)
    for a,b in zip(gpu.regressions,cpu.regressions):
        np.testing.assert_allclose(a.coefficients,b.coefficients,rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize("kind",["gsr","local_gsr","slv"])
def test_gpu_float32_rate_scan_is_finite_and_matches_float64(gpu_devices,dal,kind):
    model,_ = hybrid_pair(dal,kind)
    date = TODAY.add_days(365)
    product = api.Product_New((date,),("pay PAYS FIX(EQ[A]) + FIX(IR[USD,LIBOR_3M_CME])",))
    prepared = prepare(product,TODAY,model=model)
    settings = MonteCarloSettings(enable_aad=True,platform="gpu",devices=gpu_devices,block_size=256,use_bb=True)
    reference = prepared.engine(model,settings).value(1025)
    actual = prepared.engine(model,replace(settings,dtype="float32")).value(1025)
    for name in reference:
        np.testing.assert_allclose(actual[name],reference[name],rtol=2e-5 if name == "PV" else 5e-3,atol=2e-4,err_msg=name)
