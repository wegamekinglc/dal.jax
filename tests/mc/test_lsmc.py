"""Exercise cash flows, frozen AD and independent path-budget checks."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dal_jax import BlackScholes, MonteCarloSettings, prepare
from dal_jax.api import MonteCarlo_Value, Product_New
from dal_jax.dates import Date
from dal_jax.errors import EmptyVectorReduction, ScriptError
from dal_jax.models import CorrelatedBlackScholes
from dal_jax.script.product import ScriptProductSettings

TODAY = Date.ymd(2026, 9, 20)
MID = Date.ymd(2027, 9, 20)
END = Date.ymd(2028, 3, 20)
MODEL = BlackScholes(spot=100.0, vol=0.2, rate=0.05)


def put_engine(settings=None, dates=(MID, END)):
    data = Product_New(dates, ["EXERCISE MAX(100-SPOT(),0)"] * len(dates))
    return prepare(data, TODAY, model=MODEL).engine(
        MODEL, settings or MonteCarloSettings(parallel="none", block_size=512)
    )


def test_european_limit_and_bermudan_premium():
    european = put_engine(dates=(END,)).value(4096)["PV"]
    bermudan = put_engine().value(4096)["PV"]
    assert 6.1 < european < 6.4
    assert european + 0.1 < bermudan < european + 0.75


def test_frozen_policy_gradient_and_hessian_use_common_paths():
    engine = put_engine(
        MonteCarloSettings(enable_aad=True, parallel="none", smooth=1.0, block_size=512)
    )
    policy = engine.train(1024)
    params = engine.default_params()

    def value(spot):
        changed = params | {"model": params["model"] | {"spot": spot}}
        return engine.pricer(1024, policy=policy)(changed)[0]

    spot, h = jnp.asarray(100.0), 0.001
    derivative = jax.jit(jax.grad(value))(spot)
    fd = (jax.jit(value)(spot + h) - jax.jit(value)(spot - h)) / (2 * h)
    np.testing.assert_allclose(derivative, fd, atol=2e-8)
    gamma = jax.jit(jax.hessian(value))(spot)
    derivative_fn = jax.jit(jax.grad(value))
    np.testing.assert_allclose(
        gamma, (derivative_fn(spot + h) - derivative_fn(spot - h)) / (2 * h), atol=2e-7
    )
    np.testing.assert_allclose(jax.jacfwd(value)(spot), derivative, atol=1e-13)


@pytest.mark.parametrize("strategy", ["none", "shard_map", "auto", "pmap"])
def test_parallel_frozen_prices_and_risks_match(strategy, cpu_devices):
    settings = MonteCarloSettings(
        enable_aad=True, block_size=256, devices=cpu_devices, parallel=strategy
    )
    reference = put_engine(replace(settings, parallel="none")).value(1025)
    actual = put_engine(settings).value(1025)
    for key in reference:
        np.testing.assert_allclose(actual[key], reference[key], atol=1e-12, rtol=1e-12)


def test_training_policy_does_not_change_with_pricing_budget():
    settings = MonteCarloSettings(
        parallel="none", lsmc_training_paths=1024, lsmc_validation_paths=256
    )
    small, large = put_engine(settings), put_engine(settings)
    small.train(128)
    large.train(4096)
    assert small.regressions == large.regressions


@pytest.mark.parametrize("aad", [False, True])
def test_exercise_preserves_prior_payment_and_replaces_same_day_payment(aad):
    model = BlackScholes(spot=100.0, vol=0.0, rate=0.05)
    rows = ("pay PAYS 3", "pay PAYS 100 EXERCISE 4", "pay PAYS 1000")
    dates = (TODAY.add_days(100), MID, END)
    engine = prepare(Product_New(dates, rows), TODAY, model=model).engine(
        model, MonteCarloSettings(enable_aad=aad, parallel="none")
    )
    # Future payments dominate continuation, so exercise is deliberately skipped.
    value = engine.value(64)["PV"]
    assert value == pytest.approx(
        sum(
            amount * np.exp(-0.05 * ((day - TODAY) / 365.0))
            for amount, day in zip((3.0, 100.0, 1000.0), dates)
        )
    )
    rows = ("pay PAYS 3", "pay PAYS 1 EXERCISE 4", "pay PAYS 1")
    engine = prepare(Product_New(dates, rows), TODAY, model=model).engine(
        model, MonteCarloSettings(enable_aad=aad, parallel="none")
    )
    assert engine.value(64)["PV"] == pytest.approx(
        3.0 * np.exp(-0.05 * 100 / 365.0) + 4.0 * np.exp(-0.05)
    )


def test_hard_exercise_masks_later_vector_error():
    model = BlackScholes(spot=100.0, vol=0.0)
    data = Product_New((MID, END), ("EXERCISE 4", "x = AVERAGE(v) EXERCISE x"))
    engine = prepare(data, TODAY, model=model).engine(
        model, MonteCarloSettings(parallel="none", lsmc_training_paths=64)
    )
    # Training records every date and must reject the invalid live continuation.
    with pytest.raises(EmptyVectorReduction):
        engine.train(64)


@pytest.mark.parametrize(
    "rows", [("pay PAYS 100 pay = 1 EXERCISE 2",), ("pay = 1 pay PAYS 1 EXERCISE 2",)]
)
def test_invalid_payoff_assignments_are_rejected(rows):
    with pytest.raises(ScriptError, match="UnsupportedExercisePayoff"):
        prepare(Product_New((MID,), rows), TODAY, model=MODEL)


@pytest.mark.parametrize("day", [TODAY.add_days(-1), TODAY])
def test_exercise_dates_must_be_strictly_future(day):
    with pytest.raises(ScriptError, match="UnsupportedExerciseDate"):
        prepare(Product_New((day,), ("EXERCISE 1",)), TODAY, model=MODEL)


def test_sobol_only_and_path_overflow_are_rejected_before_allocation():
    with pytest.raises(ScriptError, match="UnsupportedRsgForExercise"):
        put_engine(MonteCarloSettings(rsg="mrg32"))
    with pytest.raises(ScriptError, match="InvalidPathCount"):
        put_engine(MonteCarloSettings(lsmc_training_paths=2**31 - 1)).train(2**31 + 1)


def test_variable_features_and_multiple_equities_are_bound():
    settings = ScriptProductSettings(regression_features=("VAR[x]",))
    data = Product_New((MID, END), ("x = SPOT() EXERCISE MAX(100-x,0)",) * 2, settings=settings)
    engine = prepare(data, TODAY, model=MODEL).engine(MODEL, MonteCarloSettings(parallel="none"))
    assert engine.value(1024)["PV"] == pytest.approx(put_engine().value(1024)["PV"], abs=1e-12)
    model = CorrelatedBlackScholes(
        indices=("EQ[A]", "EQ[B]"),
        spots=(100.0, 95.0),
        vols=(0.2, 0.25),
        divs=(0.0, 0.0),
        rate=0.05,
        correlations=((1.0, 0.4), (0.4, 1.0)),
    )
    data = Product_New((MID, END), ("EXERCISE MAX(100-MIN(FIX(EQ[A]),FIX(EQ[B])),0)",) * 2)
    with pytest.raises(ScriptError, match="AmbiguousLsmcRegressor"):
        prepare(data, TODAY, model=model)
    data = replace(data, settings=ScriptProductSettings(regression_features=("EQ[A]", "EQ[B]")))
    engine = prepare(data, TODAY, model=model).engine(
        model, MonteCarloSettings(parallel="none", lsmc_training_paths=256)
    )
    assert engine.value(128)["PV"] > 10.0


def test_adaptive_degree_and_rqmc_replicas_are_repeatable():
    settings = MonteCarloSettings(
        enable_aad=True,
        parallel="none",
        lsmc_training_paths=512,
        lsmc_validation_paths=256,
        lsmc_rqmc_replicates=3,
        lsmc_training_seed=17,
        lsmc_pricing_seed=29,
    )
    first, second = put_engine(settings), put_engine(settings)
    assert first.value(256) == second.value(256)
    assert first.replicate_means == second.replicate_means
    assert len(set(first.replicate_means)) == 3
    assert all(fit.validation_mse is not None for fit in first.regressions)


def test_compatibility_api_routes_exercise_to_lsmc():
    data = Product_New((MID, END), ("EXERCISE MAX(100-SPOT(),0)",) * 2)
    result = MonteCarlo_Value(
        data, MODEL, 512, evaluation_date=TODAY, enable_aad=True, parallel="none"
    )
    assert set(result) == {"PV", "d_spot", "d_vol", "d_rate", "d_div"}
