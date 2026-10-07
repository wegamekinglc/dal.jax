"""Pathwise GSR prices and all curve/volatility risks against the pinned DAL."""

import numpy as np
import pytest

from dal_jax import MonteCarloSettings, prepare
from dal_jax.api import Product_New
from dal_jax.dates import Date
from dal_jax.models.gsr import GSR, GSRCurve, GSRVol, MultiFactorGSRVol
from dal_jax.models.gsrslv import GSRSLV, GSRSLVSettings, GSRLeverage

TODAY = Date.ymd(2026, 10, 2)


def native_date(dal, date):
    return dal.Date_(date.year, date.month, date.day)


@pytest.mark.oracle
def test_model_and_valuation_date_mismatch_has_native_error_code(dal):
    if not hasattr(dal,"MultiFactorGSRModelData_New"):
        pytest.skip("rates validation requires pinned DAL source build")
    model,native_model = model_pair(dal)
    dates,events = [TODAY.add_days(365)],["pay PAYS 1"]
    from dal_jax.errors import ScriptError
    with pytest.raises(ScriptError,match="InvalidModelEvaluationDate"):
        prepare(Product_New(dates,events),TODAY.add_days(1),model=model)
    valuation = dal.ScriptValuationSettings_(evaluation_date=native_date(dal,TODAY.add_days(1)))
    with pytest.raises(RuntimeError,match="InvalidModelEvaluationDate"):
        dal.ScriptValuation_Explain(dal.Product_New([native_date(dal,d) for d in dates],events),native_model,valuation=valuation)


def model_pair(dal, *, multi=False, zero=False, projections=False, knots=False):
    dates = (TODAY, TODAY.add_days(365), TODAY.add_days(730), TODAY.add_days(365*8))
    curve = GSRCurve(evaluation_date=TODAY, currency="USD", node_dates=dates, discount_log_df=(0., -.03, -.06, -.25),
                     projection_tenors=("3M", "6M") if projections else (),
                     projection_log_df=((0., -.034, -.068, -.29), (0., -.036, -.072, -.31)) if projections else ())
    gdates = (TODAY, TODAY.add_days(100)) if knots else (TODAY,)
    hdates = (TODAY, TODAY.add_days(240)) if knots else (TODAY,)
    gs = (0., 0.) if zero else (.02, .013)
    if multi:
        vol = MultiFactorGSRVol(factor_names=("level", "slope"), g_knot_dates=gdates,
                               g_values=(gs[:len(gdates)], tuple(x*.7 for x in gs[:len(gdates)])),
                               h_knot_dates=hdates, h_values=((1., .7)[:len(hdates)], (-.4, .2)[:len(hdates)]),
                               correlations=((1., .3), (.3, 1.)))
    else:
        vol = GSRVol(g_knot_dates=gdates, g_values=gs[:len(gdates)], h_knot_dates=hdates, h_values=(1., .7)[:len(hdates)])
    nc = dal.GSRCurveData_New("curve", native_date(dal, TODAY), "USD", list(map(lambda d:native_date(dal,d), dates)),
                             list(curve.discount_log_df), list(curve.projection_tenors),
                             dal.DoubleMatrix_(list(map(list, curve.projection_log_df))) if projections else dal.DoubleMatrix_(0,0))
    if multi:
        nv = dal.MultiFactorGSRVolData_New("vol", list(vol.factor_names), [native_date(dal,d) for d in gdates],
                                          dal.DoubleMatrix_(list(map(list,vol.g_values))), [native_date(dal,d) for d in hdates],
                                          dal.DoubleMatrix_(list(map(list,vol.h_values))), dal.DoubleMatrix_(list(map(list,vol.correlations))))
        nm = dal.MultiFactorGSRModelData_New("rates", nc, nv)
    else:
        nv = dal.GSRVolData_New("vol", [native_date(dal,d) for d in gdates], list(vol.g_values),
                               [native_date(dal,d) for d in hdates], list(vol.h_values))
        nm = dal.GSRModelData_New("rates", nc, nv)
    return GSR(curve=curve, vol=vol), nm


@pytest.mark.oracle
@pytest.mark.parametrize("kind", ["bond", "libor", "swap", "option", "delayed"])
@pytest.mark.parametrize("multi,zero,knots,bridge", [(False,False,False,False), (True,False,True,True), (True,True,False,True)])
def test_gsr_price_and_every_parameter_matches_native(dal, kind, multi, zero, knots, bridge):
    if not hasattr(dal, "MultiFactorGSRModelData_New"):
        pytest.skip("rates parity requires pinned DAL source build")
    model, native_model = model_pair(dal, multi=multi, zero=zero, projections=True, knots=knots)
    end = TODAY.add_days(365)
    rows = {
        "bond": ((end,), ("pay PAYS FIX(IR[USD,DF,2028-10-01])",)),
        "libor": ((TODAY,end), ("pay PAYS FIX(IR[USD,LIBOR_6M_CME])", "pay PAYS FIX(IR[USD,LIBOR_3M_CME])")),
        "swap": ((end,), ("pay PAYS FIX(IR[USD,SWAP,5Y]) + FIX(IR[USD,SWAP,3M])",)),
        "option": ((TODAY.add_days(180),end), ("pay PAYS MAX(FIX(IR[USD,DF,2028-10-01]) - .96, 0)",)*2),
        "delayed": ((end,), ("pay PAYS FIX(IR[USD,LIBOR_3M_CME]) ON 2028-10-01",)),
    }[kind]
    settings = MonteCarloSettings(enable_aad=True, use_bb=bridge, parallel="none", block_size=256)
    ours = prepare(Product_New(*rows), TODAY, model=model).engine(model, settings).value(257)
    theirs = dal.MonteCarlo_ValueWithSettings(dal.Product_New([native_date(dal,d) for d in rows[0]], rows[1]), native_model, 257,
                                             valuation=dal.ScriptValuationSettings_(evaluation_date=native_date(dal,TODAY)),
                                             simulation=dal.MonteCarloSettings_(enable_aad=True,use_bb=bridge))
    assert ours.keys() == theirs.keys()
    for name in ours:
        np.testing.assert_allclose(ours[name], theirs[name], rtol=1e-8 if name != "PV" else 1e-10, atol=1e-10, err_msg=name)


@pytest.mark.oracle
@pytest.mark.parametrize("zero,bridge,vov", [(True,True,.5), (False,False,.5), (False,True,0.)])
def test_slv_stochastic_numeraire_and_all_risks_match_native(dal, zero, bridge, vov):
    if not hasattr(dal,"GSRSLVModelData_New"):
        pytest.skip("SLV parity requires pinned DAL source build")
    gaussian,native_gaussian = model_pair(dal,multi=True,zero=zero,projections=True,knots=True)
    leverage = GSRLeverage(rate_shifts=(-.05,0.,.05),times=(0.,1.),values=((.9,.85),(1.,1.1),(1.2,1.25)))
    settings = GSRSLVSettings(kappa=.8,vol_of_vol=vov,variance_correlations=(.2,-.1),max_step=.25)
    model = GSRSLV(gaussian=gaussian,leverage=leverage,settings=settings)
    ns = dal.GSRSLVSettings_()
    ns.kappa, ns.vol_of_vol, ns.variance_correlations, ns.max_step = settings.kappa, vov, [.2,-.1], .25
    nl = dal.GSRLeverageData_New("leverage",list(leverage.rate_shifts),list(leverage.times),dal.DoubleMatrix_(list(map(list,leverage.values))))
    native_model = dal.GSRSLVModelData_New("smile",native_gaussian,nl,ns)
    dates = (TODAY.add_days(180),TODAY.add_days(365))
    events = ("pay PAYS MAX(FIX(IR[USD,DF,2028-10-01]) - .96, 0) + FIX(IR[USD,LIBOR_3M_CME])",)*2
    ours = prepare(Product_New(dates,events),TODAY,model=model).engine(model,MonteCarloSettings(enable_aad=True,use_bb=bridge,parallel="none",block_size=128)).value(257)
    theirs = dal.MonteCarlo_ValueWithSettings(dal.Product_New([native_date(dal,d) for d in dates],events),native_model,257,
                                             valuation=dal.ScriptValuationSettings_(evaluation_date=native_date(dal,TODAY)),
                                             simulation=dal.MonteCarloSettings_(enable_aad=True,use_bb=bridge))
    assert ours.keys() == theirs.keys()
    for name in ours:
        np.testing.assert_allclose(ours[name],theirs[name],rtol=1e-8 if name != "PV" else 1e-10,atol=1e-10,err_msg=name)
