"""P2 compatibility API and simulation cases ported from DAL test_simulation.cpp."""

import math

import numpy as np
import pytest
from script_cases import CASES, EVALUATION, MATURITY
from support import bs_call_price

from dal_jax import BlackScholes
from dal_jax.api import BSModelData_New, EvaluationDate_Get, EvaluationDate_Set, MonteCarlo_Value, Product_New
from dal_jax.errors import InvalidPathCount, InvalidPayoff, InvalidSetting, InvalidSmoothing, UnsupportedExecutionMode


def call():
    dates, events = CASES["european_call"]
    return Product_New(dates, events)


@pytest.mark.parametrize("rsg", ["sobol", "mrg32", "irn"])
def test_fixed_seed_produces_repeatable_prices(rsg):
    model = BSModelData_New(100, 0.2, 0.05, 0.02)
    values = [MonteCarlo_Value(call(), model, 257, rsg, evaluation_date=EVALUATION, parallel="none")["PV"] for _ in range(2)]
    assert values[0] == values[1] and math.isfinite(values[0])


def test_ported_analytic_bs_case():
    model = BSModelData_New(10, 0.20, 0.034, 0.021)
    today, maturity = EVALUATION.ymd(2022, 6, 22), EVALUATION.ymd(2024, 6, 21)
    product = Product_New(["STRIKE", maturity], ["11", "call PAYS MAX(SPOT()-STRIKE,0)"])
    actual = MonteCarlo_Value(product, model, 65536, evaluation_date=today)
    reference = bs_call_price(10, 11, .20, .034, .021, (maturity-today) / 365.0)
    assert actual["PV"] == pytest.approx(reference, abs=1e-3)


def test_zero_payoff_and_single_path():
    assert MonteCarlo_Value(Product_New([MATURITY], ["pay PAYS 0"]), BSModelData_New(100, .2), 1024,
                           evaluation_date=EVALUATION) == {"PV": 0.0}
    result = MonteCarlo_Value(call(), BSModelData_New(100, .2), 1, evaluation_date=EVALUATION)
    assert math.isfinite(result["PV"])


@pytest.mark.parametrize("use_bb", [False, True])
@pytest.mark.parametrize("parallel", ["none", "shard_map", "auto", "pmap"])
def test_today_only_product_compiles_without_a_random_dimension(cpu_devices, parallel, use_bb):
    product = Product_New([EVALUATION], ["pay PAYS SPOT()"])
    assert MonteCarlo_Value(product, BSModelData_New(100, .2), 8193, evaluation_date=EVALUATION,
                           devices=cpu_devices, parallel=parallel, use_bb=use_bb) == {"PV": 100.0}


def test_expired_product_skips_model_allocation_and_generation():
    class NoSimulation(BlackScholes):
        def allocate(self, *args):
            pytest.fail("expired products must not allocate a model plan")

        def init(self, *args):
            pytest.fail("expired products must not initialize the model")

        def generate(self, *args):
            pytest.fail("expired products must not generate paths")

    product = Product_New([EVALUATION.add_days(-1)], ["pay PAYS 7"])
    assert MonteCarlo_Value(product, NoSimulation(spot=100, vol=.2), 1, evaluation_date=EVALUATION) == {"PV": 0.0}


@pytest.mark.parametrize("n_paths", [0, -1, True, False, 1.5, "4", np.nan])
def test_invalid_path_counts(n_paths):
    with pytest.raises(InvalidPathCount):
        MonteCarlo_Value(call(), BSModelData_New(100, .2), n_paths, evaluation_date=EVALUATION)


def test_numpy_integer_path_count_is_accepted():
    result = MonteCarlo_Value(call(), BSModelData_New(100, .2), np.int64(8), evaluation_date=EVALUATION)
    assert math.isfinite(result["PV"])


def test_invalid_rng_and_smoothing():
    with pytest.raises(InvalidSetting):
        MonteCarlo_Value(call(), BSModelData_New(100, .2), 8, "unknown", evaluation_date=EVALUATION)
    with pytest.raises(InvalidSmoothing):
        MonteCarlo_Value(call(), BSModelData_New(100, .2), 8, smooth=0, evaluation_date=EVALUATION)


def test_fuzzy_script_request_is_explicitly_deferred():
    with pytest.raises(UnsupportedExecutionMode, match="P3"):
        MonteCarlo_Value(call(), BSModelData_New(100, .2), 8, enable_aad=True, evaluation_date=EVALUATION)


@pytest.mark.parametrize("compiled", [True, False])
def test_compiled_option_is_accepted_with_a_warning(compiled):
    with pytest.warns(UserWarning, match="XLA"):
        actual = MonteCarlo_Value(call(), BSModelData_New(100, .2), 8, compiled=compiled, evaluation_date=EVALUATION)
    assert actual == MonteCarlo_Value(call(), BSModelData_New(100, .2), 8, evaluation_date=EVALUATION)


def test_method_alias_and_conflicting_rng_options():
    model = BSModelData_New(100, .2)
    assert MonteCarlo_Value(call(), model, 8, method="sobol", evaluation_date=EVALUATION) == MonteCarlo_Value(call(), model, 8, evaluation_date=EVALUATION)
    with pytest.raises(InvalidSetting, match="method and rsg"):
        MonteCarlo_Value(call(), model, 8, "mrg32", method="irn", evaluation_date=EVALUATION)


def test_python_dates_and_global_evaluation_date():
    previous = EvaluationDate_Get()
    try:
        EvaluationDate_Set(EVALUATION.to_python())
        product = Product_New([EVALUATION.to_python()], ["pay PAYS SPOT()"])
        assert MonteCarlo_Value(product, BSModelData_New(100, .2), 1) == {"PV": 100.0}
        past = EVALUATION.add_days(-1)
        data = Product_New([past.to_python(), MATURITY.to_python()], ["x = SPOT()", "pay PAYS x"])
        assert MonteCarlo_Value(data, BSModelData_New(100, .2), 1, historical_spots={past.to_python(): 95}) == {"PV": 95.0}
    finally:
        EvaluationDate_Set(previous)


@pytest.mark.parametrize("expression", ["LOG(-1)", "SQRT(-1)", "1/0", "(-1)^0.5"])
def test_selected_invalid_arithmetic_raises_invalid_payoff(expression):
    product = Product_New([MATURITY], [f"pay PAYS {expression}"])
    with pytest.raises(InvalidPayoff):
        MonteCarlo_Value(product, BSModelData_New(100, .2), 8, evaluation_date=EVALUATION)
