"""Exercise the installed alpha package without the native DAL oracle."""

import importlib.metadata
import importlib.util

import numpy as np

import dal_jax as dj
import dal_jax.api as api
from dal_jax.dates import Date


def main():
    if importlib.util.find_spec("dal") is not None:
        raise RuntimeError("Run this check in a clean wheel-only environment")
    today = Date.ymd(2026,10,8)
    date = today.add_days(365)
    dates = [today,today.add_days(365*8)]
    curve = api.GSRCurveData_New("curve",today,"USD",dates,[0.,-.24])
    vol = api.MultiFactorGSRVolData_New("vol",["level"],[today],[[.02]],[today],[[1.]],[[1.]])
    gsr = api.GSRModelData_New("rates",curve,vol)
    leverage = api.GSRLeverageData_New("leverage",[-.05,.05],[0.,1.],[[1.,1.],[1.,1.]])
    slv_settings = api.GSRSLVSettings_()
    slv_settings.max_step = .25
    slv = api.GSRSLVModelData_New("slv",gsr,leverage,slv_settings)
    equity = api.HybridBSEquityData_New("equity","EQ[A]","USD","FA",100.,.2,.01)
    rate = api.HybridGSRRateDataMulti_New("rate",["level"],curve,vol)
    correlation = api.HybridCorrelation_Assemble("correlation",[rate,equity],[])
    hybrid = api.HybridModelData_New("hybrid","USD",[rate,equity],correlation)
    cases = [
        ("BS",api.BSModelData_New(100.,.2,.03),[date],["pay PAYS MAX(SPOT()-100,0)"]),
        ("GSR",gsr,[date],["pay PAYS FIX(IR[USD,LIBOR_3M_CME])"]),
        ("GSRSLV",slv,[date],["pay PAYS FIX(IR[USD,LIBOR_3M_CME])"]),
        ("Hybrid",hybrid,[date],["pay PAYS MAX(FIX(EQ[A])-101,0)"]),
        ("LSMC",api.BSModelData_New(100.,.2,.03),[today.add_days(180),date],["EXERCISE MAX(100-SPOT(),0)"]*2),
    ]
    for label,model,event_dates,events in cases:
        result = api.MonteCarlo_ValueWithSettings(api.Product_New(event_dates,events),model,128,
            valuation=dj.ValuationSettings(evaluation_date=today),
            simulation=dj.MonteCarloSettings(parallel="none",block_size=128,enable_aad=True,lsmc_training_paths=128))
        if not np.isfinite(list(result.values())).all():
            raise RuntimeError(f"{label}: non-finite installed-wheel result")
        print(f"{label}: PV={result['PV']:.12g}, {len(result)-1} risks")
    print(f"dal-jax {importlib.metadata.version('dal-jax')}: {dj.__file__}")


if __name__ == "__main__":
    main()
