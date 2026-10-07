"""Rate-model domain checks, zero volatility, transforms and parallel scans."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dal_jax.api as api
from dal_jax import MonteCarloSettings, prepare
from dal_jax.dates import Date
from dal_jax.errors import DalError
from dal_jax.models.gsr import GSR, GSRCurve, MultiFactorGSRVol
from dal_jax.models.gsrslv import GSRSLV, GSRSLVSettings, GSRLeverage
from dal_jax.models.hybrid import Hybrid, HybridBSEquity, HybridGSRRate, HybridGSRSLVRate, HybridCorrelation, assemble_correlation
from dal_jax.script.product import ScriptProductSettings

TODAY = Date.ymd(2026,10,2)


def rates(zero=False,singular=False):
    curve = GSRCurve(evaluation_date=TODAY,currency="USD",node_dates=(TODAY,TODAY.add_days(365),TODAY.add_days(365*6)),
                     discount_log_df=(0.,-.03,-.18))
    vol = MultiFactorGSRVol(factor_names=("level","slope"),g_knot_dates=(TODAY,),g_values=((0. if zero else .02,),(0. if zero else .01,)),
                            h_knot_dates=(TODAY,),h_values=((1.,),(-.4,)),correlations=((1.,1. if singular else .3),(1. if singular else .3,1.)))
    return GSR(curve=curve,vol=vol)


def rate_model(kind,zero=False):
    gaussian = rates(zero)
    smile = GSRSLV(gaussian=gaussian,leverage=GSRLeverage(rate_shifts=(-.05,.05),times=(0.,1.),values=((.9,.95),(1.2,1.1))),
                  settings=GSRSLVSettings(max_step=.25,variance_correlations=(.2,-.1)))
    if kind == "gsr":
        return gaussian
    if kind == "slv":
        return smile
    equity = HybridBSEquity(name="A",index="EQ[A]",currency="USD",factor="FA",spot=100.,vol=.2,div=.01)
    rate = HybridGSRRate(name="R",model=gaussian,factors=("level","slope")) if kind == "hybrid" else HybridGSRSLVRate(
        name="R",model=smile,vol_factor="FV",bridge_factor="FB")
    return Hybrid(domestic_currency="USD",components=(rate,equity),correlation=assemble_correlation((rate,equity),(("FA","level",.2),)))


@pytest.mark.parametrize("kind",["gsr","slv","hybrid","hybrid_slv"])
def test_parallel_rate_scan_values_and_all_greeks(kind,cpu_devices):
    model = rate_model(kind)
    event = "pay PAYS FIX(IR[USD,LIBOR_3M_CME]) + FIX(IR[USD,DF,2028-10-01])"
    if kind.startswith("hybrid"):
        event += " + .01*FIX(EQ[A])"
    prepared = prepare(api.Product_New((TODAY.add_days(180),TODAY.add_days(365)),(event,)*2),TODAY,model=model)
    settings = MonteCarloSettings(enable_aad=True,use_bb=True,block_size=128,platform="cpu",devices=cpu_devices)
    reference = prepared.engine(model,replace(settings,parallel="none",devices=(cpu_devices[0],))).value(257)
    for strategy in ("shard_map","auto","pmap"):
        result = prepared.engine(model,replace(settings,parallel=strategy)).value(257)
        for name in reference:
            np.testing.assert_allclose(result[name],reference[name],rtol=1e-12,atol=1e-12,err_msg=name)


@pytest.mark.parametrize("kind",["gsr","slv","hybrid","hybrid_slv"])
def test_zero_rate_volatility_has_finite_risks_and_reprices_discount_curve(kind):
    model = rate_model(kind,True)
    event = "pay PAYS FIX(IR[USD,DF,2028-10-01])"
    product = prepare(api.Product_New((TODAY.add_days(365),),(event,)),TODAY,model=model)
    result = product.engine(model,MonteCarloSettings(enable_aad=True,parallel="none",block_size=32)).value(16)
    assert result["PV"] == pytest.approx(np.exp(-.06),abs=1e-12)
    assert np.isfinite(list(result.values())).all()
    if kind in ("slv","hybrid_slv"):
        assert result["d_volOfVol"] == 0.


def test_gaussian_rank_deficient_covariance_is_supported():
    model = rates(singular=True)
    product = prepare(api.Product_New((TODAY.add_days(365),),("pay PAYS FIX(IR[USD,DF,2028-10-01])",)),TODAY,model=model)
    result = product.engine(model,MonteCarloSettings(enable_aad=True,parallel="none",block_size=32)).value(32)
    assert np.isfinite(list(result.values())).all()


def test_gsr_jacfwd_reverse_and_common_path_differences_agree():
    model = rates()
    product = prepare(api.Product_New((TODAY.add_days(365),),("pay PAYS FIX(IR[USD,LIBOR_3M_CME])",)),TODAY,model=model)
    engine = product.engine(model,MonteCarloSettings(enable_aad=True,parallel="none",block_size=64))
    params = engine.default_params()
    price = jax.jit(lambda p:engine.pricer(129)(p)[0])
    reverse,forward = jax.jit(jax.grad(price))(params),jax.jit(jax.jacfwd(price))(params)
    for name in model.param_labels:
        np.testing.assert_allclose(reverse["model"][name],forward["model"][name],rtol=1e-10,atol=1e-12)
        step = 1e-7
        up = params|{"model":params["model"]|{name:params["model"][name]+step}}
        down = params|{"model":params["model"]|{name:params["model"][name]-step}}
        np.testing.assert_allclose(reverse["model"][name],(price(up)-price(down))/(2*step),rtol=2e-6,atol=1e-9)


@pytest.mark.parametrize("bad",["anchor","dates","projection","correlation","evaluation","fractional_time","maturity","leverage","variance","step","hybrid_currency","bridge"])
def test_invalid_rate_inputs_fail_before_tracing(bad):
    model = rates()
    with pytest.raises((DalError,ValueError)):
        if bad == "anchor":
            replace(model.curve,discount_log_df=(.01,-.03,-.18))
        elif bad == "dates":
            replace(model.curve,node_dates=(TODAY,TODAY,TODAY.add_days(365)))
        elif bad == "projection":
            replace(model.curve,projection_tenors=("3M","3M"),projection_log_df=((0.,-.04,-.2),)*2)
        elif bad == "correlation":
            replace(model.vol,correlations=((1.,1.01),(1.01,1.)))
        elif bad == "evaluation":
            prepare(api.Product_New((TODAY.add_days(365),),("pay PAYS 1",)),TODAY.add_days(1),model=model)
        elif bad == "fractional_time":
            from dal_jax.models.base import SampleDef
            model.allocate((.5,),(SampleDef(),))
        elif bad == "maturity":
            prepare(api.Product_New((TODAY.add_days(365),),("pay PAYS FIX(IR[USD,DF,2040-01-01])",)),TODAY,model=model).engine(model)
        elif bad == "leverage":
            GSRLeverage(rate_shifts=(0.,),times=(0.,),values=((0.,),))
        elif bad in ("variance","step"):
            slv = rate_model("slv")
            replace(slv,settings=replace(slv.settings,variance_correlations=(1.01,0.)) if bad == "variance" else replace(slv.settings,max_step=0.))
        elif bad == "hybrid_currency":
            replace(rate_model("hybrid"),domestic_currency="EUR")
        else:
            hybrid = rate_model("hybrid_slv")
            names = hybrid.factor_names
            matrix = hybrid.ordered_correlation()
            a,b = names.index("FB"),names.index("FA")
            matrix[a,b] = matrix[b,a] = .1
            replace(hybrid,correlation=HybridCorrelation(factor_names=names,correlations=matrix))


def test_rate_exercise_requires_a_feature_and_accepts_ir_regressors():
    model = rates()
    dates = (TODAY.add_days(180),TODAY.add_days(365))
    events = ("EXERCISE MAX(FIX(IR[USD,LIBOR_3M_CME])-.02,0)",)*2
    with pytest.raises(DalError,match="MissingLsmcRegressionFeature"):
        prepare(api.Product_New(dates,events),TODAY,model=model)
    data = api.Product_New(dates,events,settings=ScriptProductSettings(regression_features=("IR[USD,LIBOR_3M_CME]",)))
    prepared = prepare(data,TODAY,model=model)
    result = prepared.engine(model,MonteCarloSettings(enable_aad=True,lsmc_training_paths=128,parallel="none",block_size=64)).value(129)
    assert result["PV"] > 0.
    assert np.isfinite(list(result.values())).all()
