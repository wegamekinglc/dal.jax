"""Joint equity/rate observations, bank accounts and risks against native DAL."""

from dataclasses import replace

import numpy as np
import pytest
from oracle.test_dal_rates import TODAY, model_pair, native_curve, native_date, native_vol

import dal_jax.api as api
from dal_jax import MonteCarloSettings, prepare
from dal_jax.models.gsrslv import GSRSLV, GSRLeverage, GSRSLVSettings
from dal_jax.models.hybrid import (
    Hybrid,
    HybridBSEquity,
    HybridDeterministicRate,
    HybridGSRRate,
    HybridGSRSLVRate,
    HybridLocalVolEquity,
    HybridLogDfRate,
    assemble_correlation,
)


def hybrid_pair(dal, kind, *, zero=False, spot=100.0):
    equity, ne = equity_pair(dal, kind, spot)
    rate, nr, links = hybrid_rate_pair(dal, kind, zero)
    correlation = assemble_correlation((rate, equity), links)
    native_correlation = dal.HybridCorrelation_Assemble(
        "correlation", [nr, ne], [dal.HybridFactorLink_(*link) for link in links]
    )
    return Hybrid(
        domestic_currency="USD", components=(rate, equity), correlation=correlation
    ), dal.HybridModelData_New("hybrid", "USD", [nr, ne], native_correlation)


def equity_pair(dal, kind, spot):
    equity = HybridBSEquity(
        name="A", index="EQ[A]", currency="USD", factor="FA", spot=spot, vol=0.2, div=0.01
    )
    ne = dal.HybridBSEquityData_New("A", "EQ[A]", "USD", "FA", spot, 0.2, 0.01)
    if kind.startswith("local"):
        surface = api.LocalVolSurfaceData_New(
            "surface", [60.0, 100.0, 150.0], [0.0, 1.0], [[0.24, 0.23], [0.2, 0.21], [0.18, 0.19]]
        )
        equity = HybridLocalVolEquity(
            name="A",
            index="EQ[A]",
            currency="USD",
            factor="FA",
            spot=spot,
            surface=surface,
            div=0.01,
            max_step=0.25,
        )
        ns = dal.LocalVolSurfaceData_New(
            "surface",
            [60.0, 100.0, 150.0],
            [0.0, 1.0],
            dal.DoubleMatrix_([[0.24, 0.23], [0.2, 0.21], [0.18, 0.19]]),
        )
        ne = dal.HybridLocalVolEquityData_New("A", "EQ[A]", "USD", "FA", spot, 0.01, ns, 0.25)
    return equity, ne


def hybrid_rate_pair(dal, kind, zero):
    if kind in ("flat", "local_flat"):
        rate = HybridDeterministicRate(name="R", currency="USD", rate=0.03)
        nr = dal.HybridDeterministicRateData_New("R", "USD", 0.03)
        links = []
    elif kind.startswith("curve"):
        scheme = kind.split(":")[1]
        rate = HybridLogDfRate(
            name="R",
            currency="USD",
            times=(0.0, 0.5, 1.0, 2.0, 4.0),
            log_df_values=(0.0, -0.012, -0.03, -0.068, -0.15),
            scheme=scheme,
        )
        nr = dal.HybridLogDfRateData_New(
            "R", "USD", list(rate.times), list(rate.log_df_values), scheme
        )
        links = []
    else:
        rates, _ = model_pair(dal, multi=True, projections=True, knots=True, zero=zero)
        c, v = rates.curve, rates.vol
        nc, nv = native_curve(dal, c), native_vol(dal, v)
        if kind == "slv":
            leverage = GSRLeverage(rate_shifts=(-0.05, 0.05), times=(0.0,), values=((0.9,), (1.2,)))
            smile = GSRSLV(
                gaussian=rates,
                leverage=leverage,
                settings=GSRSLVSettings(max_step=0.25, variance_correlations=(0.2, -0.1)),
            )
            rate = HybridGSRSLVRate(name="R", model=smile, vol_factor="FV", bridge_factor="FB")
            ns = dal.GSRSLVSettings_()
            ns.max_step = 0.25
            ns.variance_correlations = [0.2, -0.1]
            nl = dal.GSRLeverageData_New(
                "leverage", [-0.05, 0.05], [0.0], dal.DoubleMatrix_([[0.9], [1.2]])
            )
            nm = dal.GSRSLVModelData_New(
                "smile", dal.MultiFactorGSRModelData_New("rates", nc, nv), nl, ns
            )
            nr = dal.HybridGSRSLVRateData_New("R", "FV", "FB", nm)
        else:
            rate = HybridGSRRate(name="R", model=rates, factors=v.factor_names)
            nr = dal.HybridGSRRateDataMulti_New("R", list(v.factor_names), nc, nv)
        links = [("FA", "level", 0.2), ("FA", "slope", -0.1)]
    return rate, nr, links


@pytest.mark.oracle
@pytest.mark.parametrize(
    "kind",
    [
        "flat",
        "local_flat",
        "curve:LOG_LINEAR",
        "curve:LOG_CUBIC_NATURAL",
        "curve:MIXED",
        "gsr",
        "local_gsr",
        "slv",
    ],
)
@pytest.mark.parametrize("bridge", [False, True])
def test_hybrid_joint_price_and_all_greeks(dal, kind, bridge):
    if not hasattr(dal, "HybridModelData_New"):
        pytest.skip("hybrid parity requires pinned source oracle")
    model, native_model = hybrid_pair(dal, kind)
    dates = (TODAY, TODAY.add_days(180), TODAY.add_days(365))
    event = "pay PAYS MAX(FIX(EQ[A]) - 101, 0) + 5"
    if kind in ("gsr", "local_gsr", "slv"):
        event += " + 100 * FIX(IR[USD,LIBOR_3M_CME])"
    if kind != "slv":
        event += " ON 2028-10-01"
    events = ("pay PAYS FIX(EQ[A])*.01", event, event)
    ours = (
        prepare(api.Product_New(dates, events), TODAY, model=model)
        .engine(
            model,
            MonteCarloSettings(enable_aad=True, use_bb=bridge, parallel="none", block_size=128),
        )
        .value(257)
    )
    theirs = dal.MonteCarlo_ValueWithSettings(
        dal.Product_New([native_date(dal, d) for d in dates], events),
        native_model,
        257,
        valuation=dal.ScriptValuationSettings_(evaluation_date=native_date(dal, TODAY)),
        simulation=dal.MonteCarloSettings_(enable_aad=True, use_bb=bridge),
    )
    assert ours.keys() == theirs.keys()
    for name in ours:
        np.testing.assert_allclose(
            ours[name], theirs[name], rtol=1e-8 if name != "PV" else 1e-10, atol=1e-10, err_msg=name
        )


@pytest.mark.oracle
def test_exact_at_the_money_tie_keeps_first_operand_risk(dal):
    date = TODAY
    product = api.Product_New((date,), ("pay PAYS MAX(SPOT()-100,0) + MIN(SPOT()-100,0)",))
    model = api.BSModelData_New(100.0, 0.0)
    ours = api.MonteCarlo_Value(
        product, model, 16, evaluation_date=date, enable_aad=True, parallel="none"
    )
    theirs = dal.MonteCarlo_ValueWithSettings(
        dal.Product_New(
            [native_date(dal, date)], ["pay PAYS MAX(SPOT()-100,0) + MIN(SPOT()-100,0)"]
        ),
        dal.BSModelData_New(100.0, 0.0, 0.0, 0.0),
        16,
        valuation=dal.ScriptValuationSettings_(evaluation_date=native_date(dal, date)),
        simulation=dal.MonteCarloSettings_(enable_aad=True),
    )
    assert ours["d_spot"] == theirs["d_spot"] == 2.0


@pytest.mark.oracle
@pytest.mark.parametrize("kind", ["gsr", "slv"])
def test_hybrid_bumped_from_zero_rate_volatility_matches_native(dal, kind):
    if not hasattr(dal, "HybridModelData_New"):
        pytest.skip("hybrid live-parameter parity requires pinned source oracle")
    positive, native = hybrid_pair(dal, kind)
    model = zero_rate_model(positive)
    dates = (TODAY.add_days(180), TODAY.add_days(365))
    events = ("pay PAYS MAX(FIX(EQ[A])-101,0) + 100*FIX(IR[USD,LIBOR_3M_CME])",) * 2
    engine = prepare(api.Product_New(dates, events), TODAY, model=model).engine(
        model, MonteCarloSettings(enable_aad=True, use_bb=True, parallel="none", block_size=128)
    )
    params = engine.default_params() | {"model": positive.default_params()}
    actual = engine.value(257, params)
    expected = dal.MonteCarlo_ValueWithSettings(
        dal.Product_New([native_date(dal, d) for d in dates], events),
        native,
        257,
        valuation=dal.ScriptValuationSettings_(evaluation_date=native_date(dal, TODAY)),
        simulation=dal.MonteCarloSettings_(enable_aad=True, use_bb=True),
    )
    for name in expected:
        np.testing.assert_allclose(
            actual[name],
            expected[name],
            rtol=1e-10 if name == "PV" else 1e-8,
            atol=1e-10,
            err_msg=name,
        )


def zero_rate_model(positive):
    """Keep a hybrid's topology while zeroing its construction-time rate volatility."""
    kernel = positive.rate.model
    gaussian = kernel.gaussian if isinstance(kernel, GSRSLV) else kernel
    zero_vol = replace(gaussian.vol, g_values=np.zeros_like(gaussian.vol.g_values).tolist())
    zero_gaussian = replace(gaussian, vol=zero_vol)
    zero_kernel = (
        replace(kernel, gaussian=zero_gaussian) if isinstance(kernel, GSRSLV) else zero_gaussian
    )
    zero_rate = replace(positive.rate, model=zero_kernel)
    return replace(
        positive,
        components=tuple(
            zero_rate if c.name == positive.rate.name else c for c in positive.components
        ),
    )


def native_zero_slv_spot_delta(dal, product, valuation, n_paths):
    """Pinned native zero-rate SLV AAD has an incorrect spot risk; use its own FD."""
    values = []
    for spot in (100.001, 99.999):
        _, model = hybrid_pair(dal, "slv", zero=True, spot=spot)
        value = dal.MonteCarlo_ValueWithSettings(
            product,
            model,
            n_paths,
            valuation=valuation,
            simulation=dal.MonteCarloSettings_(use_bb=True),
        )
        values.append(value["PV"])
    return (values[0] - values[1]) / 0.002


@pytest.mark.oracle
@pytest.mark.parametrize("kind", ["gsr", "slv"])
def test_zero_rate_hybrid_prices_and_boundary_risks_match_native(dal, kind):
    if not hasattr(dal, "HybridModelData_New"):
        pytest.skip("zero-rate hybrid parity requires pinned source oracle")
    model, native = hybrid_pair(dal, kind, zero=True)
    dates = (TODAY.add_days(180), TODAY.add_days(365))
    events = ("pay PAYS MAX(FIX(EQ[A])-101,0) + 100*FIX(IR[USD,LIBOR_3M_CME])",) * 2
    actual = (
        prepare(api.Product_New(dates, events), TODAY, model=model)
        .engine(
            model, MonteCarloSettings(enable_aad=True, use_bb=True, parallel="none", block_size=128)
        )
        .value(257)
    )
    product = dal.Product_New([native_date(dal, d) for d in dates], events)
    valuation = dal.ScriptValuationSettings_(evaluation_date=native_date(dal, TODAY))
    expected = dict(
        dal.MonteCarlo_ValueWithSettings(
            product,
            native,
            257,
            valuation=valuation,
            simulation=dal.MonteCarloSettings_(enable_aad=True, use_bb=True),
        )
    )
    if kind == "slv":
        # Native spot AAD is 85.55759 here; its common-path FD is 1.06558.
        expected["d_spot:EQ[A]"] = native_zero_slv_spot_delta(dal, product, valuation, 257)
    for name in expected:
        np.testing.assert_allclose(
            actual[name],
            expected[name],
            rtol=1e-10 if name == "PV" else 1e-8,
            atol=1e-10,
            err_msg=name,
        )
