"""Three LSMC phases, policies and all Greeks against the native DAL oracle."""

from dataclasses import replace

import numpy as np
import pytest

from dal_jax import BlackScholes, CorrelatedBlackScholes, LocalVol, LocalVolSurface, MonteCarloSettings, prepare
from dal_jax.api import Product_New
from dal_jax.dates import Date
from dal_jax.script.product import ScriptProductSettings
from dal_jax.script.fixings import FixingSnapshot, ValuationSettings

pytestmark = pytest.mark.oracle


@pytest.fixture(autouse=True)
def require_native_lsmc(dal):
    if not hasattr(dal,"HybridModelData_New"):
        pytest.skip("validation/RQMC/retrained-policy parity requires the pinned source oracle")
TODAY = Date.ymd(2026,9,20)
MID = Date.ymd(2027,9,20)
END = Date.ymd(2028,3,20)


def _native_date(native, day):
    return native.Date_(day.year, day.month, day.day)


def _case(name):
    model = BlackScholes(spot=100., vol=.2, rate=.05)
    if name == "coupon":
        return ((TODAY.add_days(100), MID, END), ("pay PAYS 3", "pay PAYS 1 EXERCISE MAX(100-SPOT(),0)",
                                                 "pay PAYS 1 EXERCISE MAX(100-SPOT(),0)")), model
    if name == "conditional":
        script = "EXERCISE MAX(100-SPOT(),0) IF SPOT() > 80:2"
    elif name == "zero":
        script = "pay PAYS MAX(SPOT()-100,0) EXERCISE 0"
    elif name == "fuzzy_branch":
        model = BlackScholes(spot=1., vol=0.)
        script = "x = 4 IF SPOT() > 1 THEN x = 2 ELSE dead = LOG(SPOT()) END EXERCISE x"
    else:
        script = "EXERCISE MAX(100-SPOT(),0)"
    return ((MID, END), (script, script)), model


def _settings(native, ours):
    names = ("use_bb", "enable_aad", "smooth", "lsmc_basis_degree", "lsmc_training_paths", "lsmc_validation_paths",
             "lsmc_rqmc_replicates", "lsmc_training_seed", "lsmc_pricing_seed", "lsmc_policy_risk_mode", "lsmc_policy_bump_relative")
    return native.MonteCarloSettings_(**{name: getattr(ours, name) for name in names if getattr(ours,name) is not None})


@pytest.mark.parametrize("name", ["put", "coupon", "conditional", "zero", "fuzzy_branch"])
@pytest.mark.parametrize("aad", [False, True])
@pytest.mark.parametrize("bridge", [False, True])
def test_price_risks_and_training_coefficients(dal, name, aad, bridge):
    rows, model = _case(name)
    settings = MonteCarloSettings(enable_aad=aad, use_bb=bridge, block_size=256, parallel="none",
                                 lsmc_training_paths=1024, lsmc_validation_paths=128)
    engine = prepare(Product_New(*rows), TODAY, model=model).engine(model, settings)
    ours = engine.value(513)
    product = dal.Product_New([_native_date(dal, day) for day in rows[0]], rows[1])
    nmodel = dal.BSModelData_New(model.spot,model.vol,model.rate,model.div)
    valuation = dal.ScriptValuationSettings_(evaluation_date=_native_date(dal,TODAY))
    theirs = dal.MonteCarlo_ValueWithSettings(product,nmodel,513,valuation=valuation,simulation=_settings(dal,settings))
    assert ours.keys() == theirs.keys()
    for key in ours:
        np.testing.assert_allclose(ours[key], theirs[key], rtol=1e-8 if key != "PV" else 1e-6, atol=1e-10)
    diagnostics = dal.ScriptSimulation_Explain(product,nmodel,513,valuation=valuation,
                                                simulation=_settings(dal,replace(settings,enable_aad=False)))
    for fit, event in zip(engine.regressions, diagnostics["exercise_events"]):
        assert fit.degree == event["basis_degree"]
        assert fit.solver == event["solver"]
        assert fit.count == event["num_cond_true_paths"]
        np.testing.assert_allclose(fit.coefficients,event["coefficients"],rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(fit.means,event["normalization_means"],rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize("aad", [False, True])
def test_shifted_replicas_match_individual_native_prices(dal, aad):
    rows, model = _case("put")
    settings = MonteCarloSettings(enable_aad=aad,use_bb=True,parallel="none",lsmc_training_paths=512,
                                 lsmc_validation_paths=128,lsmc_rqmc_replicates=3,lsmc_training_seed=17,lsmc_pricing_seed=29)
    engine = prepare(Product_New(*rows),TODAY,model=model).engine(model,settings)
    ours = engine.value(256)
    product = dal.Product_New([_native_date(dal,day) for day in rows[0]],rows[1])
    nmodel = dal.BSModelData_New(model.spot,model.vol,model.rate,model.div)
    valuation = dal.ScriptValuationSettings_(evaluation_date=_native_date(dal,TODAY))
    theirs = dal.MonteCarlo_ValueWithSettings(product,nmodel,256,valuation=valuation,simulation=_settings(dal,settings))
    for key in ours:
        np.testing.assert_allclose(ours[key],theirs[key],rtol=1e-8,atol=1e-10)
    diagnostics = dal.ScriptSimulation_Explain(product,nmodel,256,valuation=valuation,
                                                simulation=_settings(dal,replace(settings,enable_aad=False)))
    if not aad:
        np.testing.assert_allclose(engine.replicate_means,diagnostics["uncertainty"]["replicate_means"],atol=1e-12)


def test_vmapped_retrained_policy_risks_match_native(dal):
    rows, model = _case("put")
    settings = MonteCarloSettings(enable_aad=True,parallel="none",lsmc_training_paths=256,
                                 lsmc_policy_risk_mode="RetrainedBump",block_size=256)
    ours = prepare(Product_New(*rows),TODAY,model=model).engine(model,settings).value(512)
    product = dal.Product_New([_native_date(dal,day) for day in rows[0]],rows[1])
    theirs = dal.MonteCarlo_ValueWithSettings(product,dal.BSModelData_New(model.spot,model.vol,model.rate,model.div),512,
                    valuation=dal.ScriptValuationSettings_(evaluation_date=_native_date(dal,TODAY)),simulation=_settings(dal,settings))
    for key in ours:
        np.testing.assert_allclose(ours[key],theirs[key],rtol=1e-8,atol=1e-10)


def test_vector_loop_and_live_script_parameter_exercise(dal):
    if not hasattr(dal,"CorrelatedBSModelData_New"):
        pytest.skip("vector exercise requires the pinned source oracle")
    rows = (("K",MID,END),("100","FOR(i,0,3) APPEND(v,SPOT()+i) END EXERCISE MAX(K-AVERAGE(v),0)",
                                    "APPEND(v,SPOT()) EXERCISE MAX(K-AVERAGE(v),0)"))
    model = BlackScholes(spot=100.,vol=.2,rate=.05)
    settings = MonteCarloSettings(enable_aad=True,parallel="none",lsmc_training_paths=1024)
    ours = prepare(Product_New(*rows),TODAY,model=model).engine(model,settings).value(512)
    product = dal.Product_New([day if isinstance(day,str) else _native_date(dal,day) for day in rows[0]],rows[1])
    theirs = dal.MonteCarlo_ValueWithSettings(product,dal.BSModelData_New(model.spot,model.vol,model.rate,model.div),512,
                valuation=dal.ScriptValuationSettings_(evaluation_date=_native_date(dal,TODAY)),simulation=_settings(dal,settings))
    for key in ours:
        np.testing.assert_allclose(ours[key],theirs[key],rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize("features",[("EQ[A]","EQ[B]"),("EQ[A]","EQ[B]","VAR[x]")])
def test_multivariate_training_coefficients_and_risks(dal,features):
    model = CorrelatedBlackScholes(indices=("EQ[A]","EQ[B]"),spots=(100.,95.),vols=(.2,.25),divs=(0.,0.),rate=.05,
                                  correlations=((1.,.4),(.4,1.)))
    events = ("x=FIX(EQ[A])*FIX(EQ[B])/100 EXERCISE MAX(150-MIN(FIX(EQ[A]),FIX(EQ[B])),0)",)*2
    rows = ((MID,END),events)
    product_settings = ScriptProductSettings(regression_features=features)
    data = Product_New(*rows,settings=product_settings)
    settings = MonteCarloSettings(enable_aad=True,parallel="none",use_bb=True,lsmc_training_paths=1024,lsmc_validation_paths=128)
    engine = prepare(data,TODAY,model=model).engine(model,settings)
    ours = engine.value(257)
    nproduct = dal.Product_New([_native_date(dal,d) for d in rows[0]],events,settings=dal.ScriptProductSettings_(regression_features=list(features)))
    nmodel = dal.CorrelatedBSModelData_New(list(model.indices),list(model.spots),list(model.vols),list(model.divs),model.rate,
                                         dal.DoubleMatrix_(list(map(list,model.correlations))))
    nv = dal.ScriptValuationSettings_(evaluation_date=_native_date(dal,TODAY))
    theirs = dal.MonteCarlo_ValueWithSettings(nproduct,nmodel,257,valuation=nv,simulation=_settings(dal,settings))
    for name in ours:
        np.testing.assert_allclose(ours[name],theirs[name],rtol=1e-8 if name != "PV" else 1e-6,atol=1e-10,err_msg=name)
    diagnostic = dal.ScriptSimulation_Explain(nproduct,nmodel,257,valuation=nv,simulation=_settings(dal,replace(settings,enable_aad=False)))
    for fit,event in zip(engine.regressions,diagnostic["exercise_events"]):
        np.testing.assert_allclose(fit.coefficients,event["coefficients"],rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize("degree",[6,8])
def test_high_degree_native_regression_and_rank_fallback(dal,degree):
    rows,model = _case("put")
    settings = MonteCarloSettings(parallel="none",lsmc_training_paths=4096,lsmc_basis_degree=degree)
    engine = prepare(Product_New(*rows),TODAY,model=model).engine(model,settings)
    ours = engine.value(257)
    nproduct = dal.Product_New([_native_date(dal,d) for d in rows[0]],rows[1])
    nmodel = dal.BSModelData_New(model.spot,model.vol,model.rate,model.div)
    nv = dal.ScriptValuationSettings_(evaluation_date=_native_date(dal,TODAY))
    diagnostic = dal.ScriptSimulation_Explain(nproduct,nmodel,257,valuation=nv,simulation=_settings(dal,settings))
    theirs = dal.MonteCarlo_ValueWithSettings(nproduct,nmodel,257,valuation=nv,simulation=_settings(dal,settings))
    np.testing.assert_allclose(ours["PV"],theirs["PV"],rtol=1e-6,atol=1e-10)
    for fit,event in zip(engine.regressions,diagnostic["exercise_events"]):
        assert fit.solver == event["solver"]
        np.testing.assert_allclose(fit.coefficients,event["coefficients"],rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize("history,local",[(False,False),(True,False),(False,True)])
def test_retrained_constants_history_and_zero_volatility_boundaries(dal,history,local):
    if local:
        surface = LocalVolSurface(spots=(80.,120.),times=(0.,1.),vols=((0.,.2),(.2,.2)))
        model = LocalVol(spot=100.,surface=surface,index="EQ[A]",rate=.05,max_step=.25)
        ns = dal.LocalVolSurfaceData_New("surface",[80.,120.],[0.,1.],dal.DoubleMatrix_([[0.,.2],[.2,.2]]))
        nmodel = dal.BSLocalVolModelData_New("local","EQ[A]","USD","W_EQ",dal.BSModelData_New(100.,.2,.05,0.),ns,.25)
    else:
        model = BlackScholes(spot=100.,vol=.2 if history else 0.,rate=.05)
        nmodel = dal.BSModelData_New(model.spot,model.vol,model.rate,model.div)
    dates,events = ("K",MID,END),("100","EXERCISE MAX(K-SPOT(),0)","EXERCISE MAX(K-SPOT(),0)")
    if history:
        past = TODAY.add_days(-1)
        dates,events = ("K",past,MID,END),("1.25","x=K*FIX(EQ[A])","EXERCISE MAX(x-FIX(EQ[A]),0)","EXERCISE MAX(x-FIX(EQ[A]),0)")
    default = "EQ[A]" if history or local else ""
    data = Product_New(dates,events,settings=ScriptProductSettings(default_index=default))
    nproduct = dal.Product_New([_native_date(dal,d) if not isinstance(d,str) else d for d in dates],events,
                              settings=dal.ScriptProductSettings_(default_index=default))
    valuation = ValuationSettings(evaluation_date=TODAY,fixings=FixingSnapshot({"EQ[A]":{past:80.}}) if history else None)
    nfix = dal.MarketFixingSnapshot_New({"EQ[A]":{dal.DateTime_(_native_date(dal,past),0):80.}}) if history else None
    nv = dal.ScriptValuationSettings_(evaluation_date=_native_date(dal,TODAY),fixings=nfix)
    settings = MonteCarloSettings(enable_aad=True,smooth=2.,parallel="none",lsmc_training_paths=256,lsmc_policy_risk_mode="RetrainedBump",block_size=128)
    ours = prepare(data,model=model,valuation=valuation).engine(model,settings).value(257)
    theirs = dal.MonteCarlo_ValueWithSettings(nproduct,nmodel,257,valuation=nv,simulation=_settings(dal,settings))
    for name in ours:
        np.testing.assert_allclose(ours[name],theirs[name],rtol=1e-8 if name != "PV" else 1e-6,atol=1e-9,err_msg=name)
