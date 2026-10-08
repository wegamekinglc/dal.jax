"""GSRSLV and equity/rate hybrids, including local-volatility equity buckets."""

from importlib import import_module

import dal
from _common import TODAY, VALUATION, arguments, compare, finish, require_p5_oracle, settings

import dal_jax as dj
import dal_jax.api as api

rates_example = import_module("15_gsr_rates")


def main():
    args = arguments(__doc__)
    require_p5_oracle()
    gaussian, native_gaussian = rates_example.model_pair(True)
    leverage = api.GSRLeverageData_New(
        "leverage", [-0.05, 0.0, 0.05], [0.0, 1.0], [[0.9, 0.85], [1.0, 1.1], [1.2, 1.25]]
    )
    native_leverage = dal.GSRLeverageData_New(
        "leverage",
        [-0.05, 0.0, 0.05],
        [0.0, 1.0],
        dal.DoubleMatrix_([[0.9, 0.85], [1.0, 1.1], [1.2, 1.25]]),
    )
    ls = api.GSRSLVSettings_()
    ls.max_step = 0.25
    ls.variance_correlations = [0.2, -0.1]
    ns = dal.GSRSLVSettings_()
    ns.max_step = 0.25
    ns.variance_correlations = [0.2, -0.1]
    smile = api.GSRSLVModelData_New("smile", gaussian, leverage, ls)
    native_smile = dal.GSRSLVModelData_New("smile", native_gaussian, native_leverage, ns)
    date = TODAY.add_days(365)
    rows = ([date], ["pay PAYS MAX(FIX(IR[USD,LIBOR_3M_CME])-.03,0)"])
    comparisons = [
        compare(
            "Standalone rate SLV",
            dj.prepare(api.Product_New(*rows), valuation=VALUATION, model=smile).engine(
                smile, settings(args, use_bb=True, block_size=1024)
            ),
            rows,
            args,
            bs=native_smile,
        )
    ]
    surface = api.LocalVolSurfaceData_New(
        "equity_vol", [60.0, 100.0, 150.0], [0.0, 1.0], [[0.24, 0.23], [0.2, 0.21], [0.18, 0.19]]
    )
    native_surface = dal.LocalVolSurfaceData_New(
        "equity_vol",
        [60.0, 100.0, 150.0],
        [0.0, 1.0],
        dal.DoubleMatrix_([[0.24, 0.23], [0.2, 0.21], [0.18, 0.19]]),
    )
    for local in (False, True):
        if local:
            equity = api.HybridLocalVolEquityData_New(
                "A", "EQ[A]", "USD", "FA", 100.0, 0.01, surface, 0.25
            )
            native_equity = dal.HybridLocalVolEquityData_New(
                "A", "EQ[A]", "USD", "FA", 100.0, 0.01, native_surface, 0.25
            )
        else:
            equity = api.HybridBSEquityData_New("A", "EQ[A]", "USD", "FA", 100.0, 0.2, 0.01)
            native_equity = dal.HybridBSEquityData_New("A", "EQ[A]", "USD", "FA", 100.0, 0.2, 0.01)
        rate = api.HybridGSRSLVRateData_New("R", "FV", "FB", smile)
        native_rate = dal.HybridGSRSLVRateData_New("R", "FV", "FB", native_smile)
        links = [("FA", "level", 0.2), ("FA", "slope", -0.1)]
        correlation = api.HybridCorrelation_Assemble("correlation", [rate, equity], links)
        native_correlation = dal.HybridCorrelation_Assemble(
            "correlation",
            [native_rate, native_equity],
            [dal.HybridFactorLink_(*link) for link in links],
        )
        hybrid = api.HybridModelData_New("hybrid", "USD", [rate, equity], correlation)
        native_hybrid = dal.HybridModelData_New(
            "hybrid", "USD", [native_rate, native_equity], native_correlation
        )
        rows = ([date], ["pay PAYS MAX(FIX(EQ[A])-101,0) + 100*FIX(IR[USD,LIBOR_3M_CME])"])
        engine = dj.prepare(api.Product_New(*rows), valuation=VALUATION, model=hybrid).engine(
            hybrid, settings(args, use_bb=True, block_size=1024)
        )
        comparisons.append(
            compare(
                "Local-vol equity + rate SLV" if local else "BS equity + rate SLV",
                engine,
                rows,
                args,
                bs=native_hybrid,
            )
        )
    finish(args, comparisons, hybrid_grid=engine.plan.grid, factor_names=hybrid.factor_names)


if __name__ == "__main__":
    main()
