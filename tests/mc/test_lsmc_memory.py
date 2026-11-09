"""``lsmc_training_memory_mb`` forces the checkpointed training scan."""

import pytest

from dal_jax import BlackScholes
from dal_jax.api import MonteCarloSettings, Product_New, ValuationSettings
from dal_jax.dates import Date
from dal_jax.errors import InvalidSetting
from dal_jax.script.preparation import prepare

TODAY = Date.ymd(2022, 9, 15)
#  Six equal-template events so the training collector runs scanned (not unrolled)
#  event groups, i.e. the branch where the checkpoint choice is exercised.
DATES = [Date.ymd(2026 + i, 9, 20) for i in range(6)]
SCRIPT = "EXERCISE MAX(100 - spot(), 0.0)\np pays MAX(100 - spot(), 0.0)"


def _engine(settings):
    model = BlackScholes(spot=100.0, vol=0.2)
    prepared = prepare(
        Product_New(DATES, [SCRIPT] * len(DATES)),
        TODAY,
        model=model,
        valuation=ValuationSettings(evaluation_date=TODAY),
    )
    return prepared.engine(model, settings)


def _train(settings):
    engine = _engine(settings)
    return engine, engine.train_result(257, engine.default_params())


def test_memory_budget_forces_checkpoint_and_preserves_numbers():
    plain, a = _train(
        MonteCarloSettings(
            parallel="none", lsmc_training_paths=128, lsmc_validation_paths=32, checkpoint=False
        )
    )
    budgeted, b = _train(
        MonteCarloSettings(
            parallel="none",
            lsmc_training_paths=128,
            lsmc_validation_paths=32,
            lsmc_training_memory_mb=512,
            lsmc_checkpoint_block_events=1,
            checkpoint=False,
        )
    )
    for x, y in zip(a.regressions, b.regressions):
        assert list(x.coefficients) == list(y.coefficients)
    #  A memory budget forces the checkpointed training scan despite checkpoint=False;
    #  the user-facing settings keep the generic flag untouched.
    assert plain.training_checkpoint is False
    assert budgeted.training_checkpoint is True
    assert budgeted._training.settings.checkpoint is True
    assert budgeted.settings.checkpoint is False


def test_default_settings_checkpoint_the_training_scan():
    assert _engine(MonteCarloSettings(parallel="none")).training_checkpoint is True


def test_checkpoint_block_events_requires_memory_mb():
    with pytest.raises(InvalidSetting, match="requires lsmc_training_memory_mb"):
        MonteCarloSettings(lsmc_checkpoint_block_events=4)
