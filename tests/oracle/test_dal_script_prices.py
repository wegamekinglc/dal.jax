"""Exact scalar script PV parity on identical Sobol paths (issue #1 P2)."""

import pytest
from script_cases import CASES, EVALUATION, oracle_dates

from dal_jax.api import BSModelData_New, MonteCarlo_Value, Product_New

pytestmark = pytest.mark.oracle


@pytest.fixture(scope="module")
def script_oracle(dal):
    previous = dal.EvaluationDate_Get()
    dal.EvaluationDate_Set(dal.Date_(EVALUATION.year, EVALUATION.month, EVALUATION.day))
    yield dal
    dal.EvaluationDate_Set(previous)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("use_bb", [False, True])
def test_scalar_script_pv_matches_dal(script_oracle, cpu_devices, case, use_bb):
    dal = script_oracle
    dates, events = CASES[case]
    model = BSModelData_New(100, 0.2, 0.05, 0.02)
    product = Product_New(dates, events)
    n_paths = 8193  # cross DAL's batch boundary and exercise last-block masking
    ours = MonteCarlo_Value(product, model, n_paths, use_bb=use_bb, evaluation_date=EVALUATION, devices=cpu_devices)
    theirs = dal.MonteCarlo_Value(dal.Product_New(oracle_dates(dates, dal), events), dal.BSModelData_New(100, 0.2, 0.05, 0.02), n_paths,
                                 "sobol", use_bb, False)
    assert set(ours) == set(theirs) == {"PV"}
    assert ours["PV"] == pytest.approx(theirs["PV"], rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("bad_expression", ["LOG(-1)", "SQRT(-1)", "1/0", "(-1)^0.5"])
def test_unused_invalid_branch_matches_dal(script_oracle, bad_expression):
    dal = script_oracle
    script = f"IF SPOT() > 100 THEN pay PAYS {bad_expression} ELSE pay PAYS 1 END"
    ours = MonteCarlo_Value(Product_New([EVALUATION], [script]), BSModelData_New(90, 0), 1, evaluation_date=EVALUATION)
    theirs = dal.MonteCarlo_Value(dal.Product_New(oracle_dates([EVALUATION], dal), [script]), dal.BSModelData_New(90, 0, 0, 0), 1)
    assert ours == dict(theirs) == {"PV": 1.0}
