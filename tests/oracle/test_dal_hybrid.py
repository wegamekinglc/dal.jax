"""Joint equity/rate observations, bank accounts and risks against native DAL."""

import numpy as np
import pytest

import dal_jax.api as api
from dal_jax import MonteCarloSettings, prepare
from dal_jax.models.gsr import GSR
from dal_jax.models.gsrslv import GSRSLV, GSRSLVSettings, GSRLeverage
from dal_jax.models.hybrid import *
from oracle.test_dal_rates import TODAY, native_date, model_pair


def hybrid_pair(dal,kind):
    equity = HybridBSEquity(name="A",index="EQ[A]",currency="USD",factor="FA",spot=100.,vol=.2,div=.01)
    ne = dal.HybridBSEquityData_New("A","EQ[A]","USD","FA",100.,.2,.01)
    if kind.startswith("local"):
        surface = api.LocalVolSurfaceData_New("surface",[60.,100.,150.],[0.,1.],[[.24,.23],[.2,.21],[.18,.19]])
        equity = HybridLocalVolEquity(name="A",index="EQ[A]",currency="USD",factor="FA",spot=100.,surface=surface,div=.01,max_step=.25)
        ns = dal.LocalVolSurfaceData_New("surface",[60.,100.,150.],[0.,1.],dal.DoubleMatrix_([[.24,.23],[.2,.21],[.18,.19]]))
        ne = dal.HybridLocalVolEquityData_New("A","EQ[A]","USD","FA",100.,.01,ns,.25)
    if kind in ("flat","local_flat"):
        rate = HybridDeterministicRate(name="R",currency="USD",rate=.03)
        nr = dal.HybridDeterministicRateData_New("R","USD",.03)
        links = []
    elif kind.startswith("curve"):
        scheme = kind.split(":")[1]
        rate = HybridLogDfRate(name="R",currency="USD",times=(0.,.5,1.,2.,4.),log_df_values=(0.,-.012,-.03,-.068,-.15),scheme=scheme)
        nr = dal.HybridLogDfRateData_New("R","USD",list(rate.times),list(rate.log_df_values),scheme)
        links = []
    else:
        rates,_ = model_pair(dal,multi=True,projections=True,knots=True)
        c,v = rates.curve,rates.vol
        nc = dal.GSRCurveData_New("curve",native_date(dal,TODAY),"USD",[native_date(dal,d) for d in c.node_dates],list(c.discount_log_df),
                                 list(c.projection_tenors),dal.DoubleMatrix_(list(map(list,c.projection_log_df))))
        nv = dal.MultiFactorGSRVolData_New("vol",list(v.factor_names),[native_date(dal,d) for d in v.g_knot_dates],dal.DoubleMatrix_(list(map(list,v.g_values))),
                                          [native_date(dal,d) for d in v.h_knot_dates],dal.DoubleMatrix_(list(map(list,v.h_values))),dal.DoubleMatrix_(list(map(list,v.correlations))))
        if kind == "slv":
            leverage = GSRLeverage(rate_shifts=(-.05,.05),times=(0.,),values=((.9,),(1.2,)))
            smile = GSRSLV(gaussian=rates,leverage=leverage,settings=GSRSLVSettings(max_step=.25,variance_correlations=(.2,-.1)))
            rate = HybridGSRSLVRate(name="R",model=smile,vol_factor="FV",bridge_factor="FB")
            ns = dal.GSRSLVSettings_(); ns.max_step=.25; ns.variance_correlations=[.2,-.1]
            nl = dal.GSRLeverageData_New("leverage",[-.05,.05],[0.],dal.DoubleMatrix_([[.9],[1.2]]))
            nm = dal.GSRSLVModelData_New("smile",dal.MultiFactorGSRModelData_New("rates",nc,nv),nl,ns)
            nr = dal.HybridGSRSLVRateData_New("R","FV","FB",nm)
        else:
            rate = HybridGSRRate(name="R",model=rates,factors=v.factor_names)
            nr = dal.HybridGSRRateDataMulti_New("R",list(v.factor_names),nc,nv)
        links = [("FA","level",.2),("FA","slope",-.1)]
    correlation = assemble_correlation((rate,equity),links)
    native_correlation = dal.HybridCorrelation_Assemble("correlation",[nr,ne],[dal.HybridFactorLink_(*link) for link in links])
    return Hybrid(domestic_currency="USD",components=(rate,equity),correlation=correlation), dal.HybridModelData_New("hybrid","USD",[nr,ne],native_correlation)


@pytest.mark.oracle
@pytest.mark.parametrize("kind",["flat","local_flat","curve:LOG_LINEAR","curve:LOG_CUBIC_NATURAL","curve:MIXED","gsr","local_gsr","slv"])
@pytest.mark.parametrize("bridge",[False,True])
def test_hybrid_joint_price_and_all_greeks(dal,kind,bridge):
    if not hasattr(dal,"HybridModelData_New"):
        pytest.skip("hybrid parity requires pinned source oracle")
    model,native_model = hybrid_pair(dal,kind)
    dates = (TODAY,TODAY.add_days(180),TODAY.add_days(365))
    event = "pay PAYS MAX(FIX(EQ[A]) - 101, 0) + 5"
    if kind in ("gsr","local_gsr","slv"):
        event += " + 100 * FIX(IR[USD,LIBOR_3M_CME])"
    if kind != "slv":
        event += " ON 2028-10-01"
    events = ("pay PAYS FIX(EQ[A])*.01",event,event)
    ours = prepare(api.Product_New(dates,events),TODAY,model=model).engine(model,
            MonteCarloSettings(enable_aad=True,use_bb=bridge,parallel="none",block_size=128)).value(257)
    theirs = dal.MonteCarlo_ValueWithSettings(dal.Product_New([native_date(dal,d) for d in dates],events),native_model,257,
                            valuation=dal.ScriptValuationSettings_(evaluation_date=native_date(dal,TODAY)),
                            simulation=dal.MonteCarloSettings_(enable_aad=True,use_bb=bridge))
    assert ours.keys() == theirs.keys()
    for name in ours:
        np.testing.assert_allclose(ours[name],theirs[name],rtol=1e-8 if name != "PV" else 1e-10,atol=1e-10,err_msg=name)


@pytest.mark.oracle
def test_exact_at_the_money_tie_keeps_first_operand_risk(dal):
    date = TODAY
    product = api.Product_New((date,),("pay PAYS MAX(SPOT()-100,0) + MIN(SPOT()-100,0)",))
    model = api.BSModelData_New(100.,0.)
    ours = api.MonteCarlo_Value(product,model,16,evaluation_date=date,enable_aad=True,parallel="none")
    theirs = dal.MonteCarlo_ValueWithSettings(dal.Product_New([native_date(dal,date)],["pay PAYS MAX(SPOT()-100,0) + MIN(SPOT()-100,0)"]),
                                            dal.BSModelData_New(100.,0.,0.,0.),16,
                                            valuation=dal.ScriptValuationSettings_(evaluation_date=native_date(dal,date)),
                                            simulation=dal.MonteCarloSettings_(enable_aad=True))
    assert ours["d_spot"] == theirs["d_spot"] == 2.
