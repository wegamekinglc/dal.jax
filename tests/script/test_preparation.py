"""Scalar preparation, historical replay and scope boundaries for issue #1 P2."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from script_cases import EVALUATION as TODAY, MATURITY

from dal_jax import EvalContext, MonteCarloEngine, MonteCarloSettings
from dal_jax.api import BSModelData_New, Product_New
from dal_jax.errors import InvalidScriptStructure, MissingFixing, PreparationRequired, UnboundHistoricalSpot, UnsupportedExecutionMode, UnsupportedDelayedPayment
from dal_jax.models.base import Scenario
from dal_jax.script import ast as A
from dal_jax.script.preparation import prepare


def test_prepared_plan_is_immutable_hashable_and_repeatable():
    data = Product_New([TODAY.add_days(-1), TODAY, MATURITY], ["x = 2", "x = x + SPOT()", "pay PAYS x + SPOT()"])
    prepared = prepare(data, TODAY)
    assert prepared == prepare(data, TODAY) and hash(prepared) == hash(prepare(data, TODAY))
    assert prepared.timeline == (0.0, 1.0)
    assert prepared.event_dates == (TODAY, MATURITY)
    assert prepared.past_event_dates == (TODAY.add_days(-1),)
    assert prepared.variables.var_names == ("x", "pay") and prepared.payoff_index == 1
    assert prepared.initial_values == (2.0, 0.0)
    assert all(definition.numeraire for definition in prepared.sample_defs)
    ids = [node.observation_id for event in prepared.events for statement in event for node in A.walk(statement) if isinstance(node, A.Spot)]
    assert ids == [0, 1]
    with pytest.raises(FrozenInstanceError):
        prepared.payoff_index = 0
    assert data.product().past_events == []


def test_spots_on_one_date_share_an_observation():
    data = Product_New([MATURITY], ["pay PAYS SPOT() + SPOT()"])
    prepared = prepare(data, TODAY)
    spots = [node for statement in prepared.events[0] for node in A.walk(statement) if isinstance(node, A.Spot)]
    assert [spot.observation_id for spot in spots] == [0, 0]
    assert len(prepared.observations) == 1


def test_passes_fold_static_branches_and_keep_script_parameters_live():
    data = Product_New(["K", MATURITY], ["2", "IF 1 > 0 THEN x = K ELSE x = 7 END IF x > 3 THEN pay PAYS x ELSE pay PAYS 1 END"])
    prepared = prepare(data, TODAY)
    assert isinstance(prepared.events[0][0], A.Collect)
    assert isinstance(prepared.events[0][1], A.If)
    assert prepared.script_params == (("K", 2.0),)
    assert prepared.max_nested_ifs == 1


def test_historical_spot_replay_uses_hard_branches_and_discards_payments():
    past = TODAY.add_days(-5)
    data = Product_New([past, MATURITY], ["IF SPOT() > 100:50 THEN x = 2 ELSE x = 3 END pay PAYS SPOT()", "pay PAYS x"])
    prepared = prepare(data, TODAY, historical_spots={past: 101.0})
    assert prepared.initial_values == (2.0, 0.0)
    assert prepared.observations[0].sample_id is None and prepared.observations[0].value == 101.0
    value = MonteCarloEngine(prepared.path_product(), BSModelData_New(100, 0), MonteCarloSettings(parallel="none")).value(1)
    assert value == {"PV": 2.0}


def test_historical_parameters_stay_live_in_the_native_pricer():
    data = Product_New(["K", TODAY.add_days(-1), MATURITY], ["2", "IF K > 1 THEN x = K ELSE x = 7 END", "pay PAYS x"])
    prepared = prepare(data, TODAY)
    engine = MonteCarloEngine(prepared.path_product(), BSModelData_New(100, 0), MonteCarloSettings(parallel="none"))
    price = jax.jit(engine.pricer(1))
    params = engine.default_params()
    assert float(price(params)[0]) == 2.0
    assert float(price(params | {"script": {"K": jnp.asarray(4.0)}})[0]) == 4.0
    assert float(price(params | {"script": {"K": jnp.asarray(0.5)}})[0]) == 7.0


def test_history_payments_do_not_affect_future_condition_folding():
    data = Product_New([TODAY.add_days(-1), MATURITY], ["pay PAYS 10", "IF pay = 0 THEN pay PAYS SPOT() ELSE pay PAYS 1 END"])
    prepared = prepare(data, TODAY)
    assert isinstance(prepared.events[0][0], A.Collect)
    scenario = Scenario(jnp.asarray([123.0]), jnp.ones(1), jnp.empty((1, 0)), jnp.empty((1, 0)))
    assert float(prepared.path_product().payoff({"script": {}}, scenario, EvalContext())) == 123.0


@pytest.mark.parametrize("missing", [None, {}])
def test_past_spot_requires_explicit_history(missing):
    data = Product_New([TODAY.add_days(-1), MATURITY], ["x = SPOT()", "pay PAYS x"])
    with pytest.raises(UnboundHistoricalSpot):
        prepare(data, TODAY, historical_spots=missing)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_history_is_rejected(value):
    past = TODAY.add_days(-1)
    data = Product_New([past, MATURITY], ["x = SPOT()", "pay PAYS x"])
    with pytest.raises(MissingFixing):
        prepare(data, TODAY, historical_spots={past: value})


@pytest.mark.parametrize("script,error", [("pay PAYS FIX(EQ[A])", PreparationRequired), ("EXERCISE SPOT()", UnsupportedExecutionMode), ("pay PAYS 1 ON 2024-01-01", UnsupportedDelayedPayment)])
def test_later_milestones_raise_explicit_errors(script, error):
    with pytest.raises(error):
        prepare(Product_New([MATURITY], [script]), TODAY)


def test_same_date_pays_on_normalizes_to_immediate_payment():
    prepared = prepare(Product_New([MATURITY], [f"pay PAYS 1 ON {MATURITY}"]), TODAY)
    assert prepared.events[0][0].payment_date is None


@pytest.mark.parametrize("dates,events", [([], []), (["K"], ["2"]), ([MATURITY], ["x = 1"])])
def test_valuation_requires_dated_events_and_a_payoff(dates, events):
    with pytest.raises(InvalidScriptStructure):
        prepare(Product_New(dates, events), TODAY)
