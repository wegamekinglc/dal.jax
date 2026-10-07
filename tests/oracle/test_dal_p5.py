"""P5 values and every adjoint against the pinned native DAL source oracle."""

import numpy as np
import pytest
from p5_cases import TODAY, case, compare, native_dates, native_model

from dal_jax import BlackScholes, FixingSnapshot, MonteCarloEngine, MonteCarloSettings, ValuationSettings, prepare
from dal_jax.api import Product_New
from dal_jax.errors import EmptyVectorReduction, VectorIndexOutOfRange

pytestmark = pytest.mark.oracle


@pytest.fixture(scope="module")
def native(dal):
    if not hasattr(dal, "CorrelatedBSModelData_New"):
        pytest.skip("P5 requires the pinned source oracle; see scripts/build_dal_oracle.sh")
    dal.EvaluationDate_Set(dal.Date_(TODAY.year, TODAY.month, TODAY.day))
    return dal


@pytest.mark.parametrize("name", ["vector_asian", "vector_fuzzy", "dated_fix_payment", "payment_schedule", "correlated_basket",
                                  "correlated_worst", "localvol_flat", "localvol_skew"])
@pytest.mark.parametrize("aad", [False, True])
@pytest.mark.parametrize("bridge", [False, True])
def test_price_and_all_greeks(native, name, aad, bridge):
    rows, model = case(name)
    prepared = prepare(Product_New(*rows), TODAY, model=model)
    settings = MonteCarloSettings(enable_aad=aad, use_bb=bridge, block_size=512)
    ours = MonteCarloEngine(prepared.path_product(), model, settings).value(1025)
    theirs = native.MonteCarlo_Value(native.Product_New(native_dates(native, rows[0]), rows[1]), native_model(native, model),
                                    1025, use_bb=bridge, enable_aad=aad)
    compare(ours, dict(theirs))


@pytest.mark.parametrize("policy", ["Model", "RequireHistorical"])
def test_historical_vector_and_today_snapshot(native, policy):
    past = TODAY.add_days(-1)
    rows = (["SCALE", past, TODAY], ["2", "APPEND(v,SCALE*FIX(EQ[A]))", "APPEND(v,FIX(EQ[A])) pay PAYS SUM(v)"])
    snapshot = FixingSnapshot({"EQ[A]": {past: 95., TODAY: 97.}})
    model = BlackScholes(spot=100., vol=.15, rate=.05, div=.03)
    valuation = ValuationSettings(evaluation_date=TODAY, fixings=snapshot, today_fixing_policy=policy)
    ours = MonteCarloEngine(prepare(Product_New(*rows), model=model, valuation=valuation).path_product(), model,
                            MonteCarloSettings(enable_aad=True)).value(16)
    ndates = native_dates(native, [past, TODAY])
    nfix = native.MarketFixingSnapshot_New({"EQ[A]": {native.DateTime_(ndates[0], 0): 95., native.DateTime_(ndates[1], 0): 97.}})
    nv = native.ScriptValuationSettings_(evaluation_date=ndates[1], fixings=nfix, today_fixing=policy)
    theirs = native.MonteCarlo_ValueWithSettings(native.Product_New(native_dates(native, rows[0]), rows[1]),
                                               native_model(native, model), 16, valuation=nv,
                                               simulation=native.MonteCarloSettings_(enable_aad=True))
    compare(ours, dict(theirs))
    np.testing.assert_allclose(ours["d_SCALE"], 95., atol=0)


@pytest.mark.parametrize("body,error,code", [("pay PAYS v[2]", VectorIndexOutOfRange, "VectorIndexOutOfRange"),
                                            ("pay PAYS AVERAGE(v)", EmptyVectorReduction, "EmptyVectorReduction")])
@pytest.mark.parametrize("aad", [False, True])
def test_named_runtime_errors_match_native(native, body, error, code, aad):
    rows = ([TODAY], [body])
    model = BlackScholes(spot=100., vol=.15)
    ours = MonteCarloEngine(prepare(Product_New(*rows), TODAY).path_product(), model, MonteCarloSettings(enable_aad=aad))
    with pytest.raises(error, match=code):
        ours.value(16)
    with pytest.raises(RuntimeError, match=code):
        native.MonteCarlo_Value(native.Product_New(native_dates(native, rows[0]), rows[1]), native_model(native, model), 16,
                               enable_aad=aad)
