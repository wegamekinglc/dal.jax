"""P3 fuzzy script prices and model/script Greeks against DAL on Sobol paths."""

import numpy as np
import pytest
from script_cases import CASES, EVALUATION, oracle_dates

import jax

from dal_jax import MonteCarloEngine, MonteCarloSettings, prepare
from dal_jax.api import BSModelData_New, MonteCarlo_Value, Product_New

pytestmark = pytest.mark.oracle


@pytest.fixture(scope="module")
def oracle(dal):
    previous = dal.EvaluationDate_Get()
    dal.EvaluationDate_Set(dal.Date_(EVALUATION.year, EVALUATION.month, EVALUATION.day))
    yield dal
    dal.EvaluationDate_Set(previous)


def compare(ours, theirs):
    assert set(ours) == set(theirs)
    np.testing.assert_allclose(ours["PV"], theirs["PV"], rtol=1e-10, atol=1e-12)
    for label in ours.keys() - {"PV"}:
        np.testing.assert_allclose(ours[label], theirs[label], rtol=1e-8, atol=1e-10, err_msg=label)


@pytest.mark.parametrize("case", ["european_call", "european_put", "asian", "autocall", "nested_if", "conditions", "dcf", "today", "expired"])
def test_scalar_fuzzy_script_matches_dal(oracle, case):
    dates, events = CASES[case]
    ours = MonteCarlo_Value(Product_New(dates, events), BSModelData_New(100, .2, .05, .02), 4097,
                           enable_aad=True, evaluation_date=EVALUATION)
    theirs = oracle.MonteCarlo_Value(oracle.Product_New(oracle_dates(dates, oracle), events), oracle.BSModelData_New(100, .2, .05, .02),
                                   4097, enable_aad=True)
    compare(ours, dict(theirs))


def test_script_barrier_greeks_match_the_million_path_reference():
    dates, events = barrier_rows()
    ours = MonteCarlo_Value(Product_New(dates, events), BSModelData_New(100, .15, .05, .03), 2**20,
                           enable_aad=True, evaluation_date=EVALUATION)
    assert ours["d_BARRIER"] == pytest.approx(0.08925165480978217, rel=1e-6)
    assert ours["d_vol"] == pytest.approx(-7.227015534889115, rel=1e-6)


def test_script_barrier_gradients_match_common_path_finite_differences():
    product = prepare(Product_New(*barrier_rows()), EVALUATION)
    engine = MonteCarloEngine(product.path_product(), BSModelData_New(100, .15, .05, .03),
                              MonteCarloSettings(enable_aad=True, parallel="none"))
    f = jax.jit(lambda params: engine.pricer(2**12)(params)[0])
    params = engine.default_params()
    gradients = jax.jit(jax.grad(f))(params)
    for group, label, h in [("model", "spot", 1e-6), ("model", "vol", 1e-8), ("model", "rate", 1e-8),
                            ("model", "div", 1e-8), ("script", "BARRIER", 1e-6), ("script", "STRIKE", 1e-6)]:
        up = params | {group: params[group] | {label: params[group][label] + h}}
        down = params | {group: params[group] | {label: params[group][label] - h}}
        difference = float((f(up)-f(down))/(2*h))
        np.testing.assert_allclose(gradients[group][label], difference, rtol=1e-6, atol=1e-8, err_msg=label)


def barrier_rows():
    end = EVALUATION.add_days(1095)
    dates = ["STRIKE", "BARRIER", EVALUATION, f"START: {EVALUATION} END: {end} FREQ: 1M", end]
    events = ["120", "150", "alive=1", "IF SPOT()>=BARRIER:0.1 THEN alive=0 END",
              "IF SPOT()>=BARRIER:0.1 THEN alive=0 END call PAYS alive*MAX(SPOT()-STRIKE,0)"]
    return dates, events


@pytest.mark.parametrize("use_bb", [False, True])
@pytest.mark.parametrize("threshold", [0, 4])
def test_monthly_barrier_price_and_all_greeks(oracle, use_bb, threshold):
    dates, events = barrier_rows()
    ours = MonteCarlo_Value(Product_New(dates, events), BSModelData_New(100, .15, .05, .03), 2**16, use_bb=use_bb,
                           enable_aad=True, scan_group_threshold=threshold, evaluation_date=EVALUATION)
    theirs = oracle.MonteCarlo_Value(oracle.Product_New(oracle_dates(dates, oracle), events), oracle.BSModelData_New(100, .15, .05, .03),
                                   2**16, use_bb=use_bb, enable_aad=True)
    compare(ours, dict(theirs))
