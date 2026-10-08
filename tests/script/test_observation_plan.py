"""FIX preparation rules from DAL test_preparation/test_fix_valuation/test_past_replay."""

import datetime as dt
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from dal_jax import (
    BlackScholes,
    FixingSnapshot,
    MonteCarloEngine,
    MonteCarloSettings,
    TodayFixingPolicy,
    ValuationSettings,
    prepare,
    set_global_fixings,
)
from dal_jax.api import (
    EvaluationDate_Get,
    EvaluationDate_Set,
    MonteCarlo_ValueWithSettings,
    Product_New,
)
from dal_jax.dates import Date
from dal_jax.errors import (
    InvalidFixingSnapshot,
    InvalidSetting,
    LookAheadObservation,
    MissingDefaultIndex,
    MissingFixing,
    MultipleModelIndices,
    UnsettledDelayedPayment,
    UnsupportedHistoricalIndex,
    UnsupportedModelObservation,
)
from dal_jax.script import ast as A
from dal_jax.script.product import ScriptProductSettings

D = Date.ymd(2026, 9, 12)
H = D.add_days(-1)
P = D.add_days(10)
INDEX = "EQ[A]"


def model(**kw):
    return BlackScholes(**({"spot": 100.0, "vol": 0.0, "rate": 0.0, "div": 0.0} | kw))


def value(rows, *, bs=None, aad=False, valuation=None):
    return MonteCarlo_ValueWithSettings(
        Product_New(*rows),
        model() if bs is None else bs,
        257,
        valuation=valuation or ValuationSettings(evaluation_date=D),
        simulation=MonteCarloSettings(enable_aad=aad),
    )


@pytest.mark.parametrize(
    "policy,expected,delta",
    [(TodayFixingPolicy.MODEL, 100.0, 1.0), (TodayFixingPolicy.REQUIREHISTORICAL, 80.0, 0.0)],
)
@pytest.mark.parametrize("aad", [False, True])
def test_today_policy_uses_model_or_exact_history(policy, expected, delta, aad):
    valuation = ValuationSettings(
        evaluation_date=D, today_fixing_policy=policy, fixings=FixingSnapshot({INDEX: {D: 80.0}})
    )
    result = value(([D], ["pay PAYS FIX(EQ[A])"]), aad=aad, valuation=valuation)
    assert result["PV"] == expected
    if aad:
        assert result["d_spot"] == delta


def test_future_fixing_adds_distinct_sample_and_ignores_snapshot_quote():
    date = D.add_days(3)
    product = Product_New(
        [P], [f"pay PAYS FIX(EQ[A], {date})-FIX(eq[a], {date})+FIX(EQ[A], {date})"]
    )
    prepared = prepare(
        product, D, model=model(rate=0.05, div=0.02), fixings=FixingSnapshot({INDEX: {date: 999.0}})
    )
    assert prepared.timeline == (3 / 365, 10 / 365)
    assert prepared.event_to_sample == (1,)
    assert len(prepared.observations) == 1 and prepared.observations[0].value is None
    assert prepared.observations[0].sample_id == 0 and prepared.observations[0].output_id == 0
    assert [definition.numeraire for definition in prepared.sample_defs] == [False, True]
    result = MonteCarloEngine(
        prepared.path_product(), model(rate=0.05, div=0.02), MonteCarloSettings(enable_aad=True)
    ).value(257)
    expected = 100 * math.exp(0.03 * 3 / 365 - 0.05 * 10 / 365)
    np.testing.assert_allclose(
        [result["PV"], result["d_spot"], result["d_rate"], result["d_div"]],
        [expected, expected / 100, expected * (3 - 10) / 365, -expected * 3 / 365],
        rtol=1e-14,
    )


def test_historical_spot_and_fix_share_default_index_request():
    data = Product_New(
        [H, P],
        ["x=SPOT()+FIX(eq[a])", "pay PAYS x"],
        settings=ScriptProductSettings(default_index=INDEX),
    )
    prepared = prepare(data, D, model=model(), fixings=FixingSnapshot({INDEX: {H: 80.0}}))
    assert len(prepared.observations) == 1
    assert prepared.initial_values == (160.0, 0.0)
    assert MonteCarloEngine(prepared.path_product(), model()).value(257)["PV"] == 160.0


def test_snapshot_copies_source_and_preparation_captures_global_snapshot():
    source = {INDEX: {H: 80.0}}
    snapshot = FixingSnapshot(source)
    source[INDEX][H] = 90.0
    assert snapshot.find("eq[a]", H) == 80.0
    assert hash(snapshot) == hash(FixingSnapshot({INDEX: {H: 80.0}}))
    with pytest.raises(FrozenInstanceError):
        snapshot.entries = ()
    set_global_fixings(snapshot)
    try:
        prepared = prepare(Product_New([P], [f"pay PAYS FIX(EQ[A], {H})"]), D, model=model())
        set_global_fixings(FixingSnapshot({INDEX: {H: 90.0}}))
        assert MonteCarloEngine(prepared.path_product(), model()).value(1)["PV"] == 80.0
    finally:
        set_global_fixings(FixingSnapshot())


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        FixingSnapshot(),
        FixingSnapshot({INDEX: {dt.datetime(2026, 9, 11, 11): 80.0}}),
        FixingSnapshot({INDEX: {D: 80.0}}),
    ],
)
def test_history_needs_exact_midnight_fixing_and_never_falls_back(snapshot):
    with pytest.raises(MissingFixing, match=f"{H} 00:00:00") as error:
        prepare(
            Product_New([H, P], ["x=FIX(EQ[A])", "pay PAYS x"]), D, model=model(), fixings=snapshot
        )
    assert "row=1" in str(error.value) and "no model fallback" in str(error.value)


@pytest.mark.parametrize(
    "quotes",
    [
        {INDEX: {H: np.nan}},
        {"FX[EUR/USD]": {H: 0.0}},
        {"FX[EUR/USD]": {H: 1.25}, "FX[USD/EUR]": {H: 0.9}},
    ],
)
def test_invalid_snapshot_quotes(quotes):
    with pytest.raises(InvalidFixingSnapshot):
        FixingSnapshot(quotes)


def test_historical_fx_reverse_and_equity_delivery_identity():
    snapshot = FixingSnapshot(
        {"FX[USD/EUR]": {H: 0.8}, "EQ[A]@2026-12-31": {H: 120.0}, INDEX: {H: 80.0}}
    )
    rows = ([P], [f"pay PAYS FIX(FX[EUR/USD], {H})+FIX(EQ[A]@2026-12-31,{H})+FIX(eq[a],{H})"])
    assert (
        value(rows, valuation=ValuationSettings(evaluation_date=D, fixings=snapshot))["PV"]
        == 201.25
    )
    assert snapshot.find("EQ[A]>3M", H) is None


def test_historical_libor_is_supported_and_other_ir_types_are_rejected():
    name = "IR:USD,LIBOR_3M_CME"
    valuation = ValuationSettings(evaluation_date=D, fixings=FixingSnapshot({name: {H: 0.025}}))
    assert value(([P], [f"pay PAYS FIX(IR[USD,LIBOR3MCME],{H})"]), valuation=valuation)[
        "PV"
    ] == pytest.approx(0.025, abs=1e-15)
    with pytest.raises(UnsupportedHistoricalIndex):
        prepare(Product_New([P], [f"pay PAYS FIX(IR[USD,DF,2027-09-12],{H})"]), D, model=model())


@pytest.mark.parametrize(
    "body",
    [
        "pay PAYS FIX(EQ[A],2026-09-23)",
        "IF 1=0 THEN pay PAYS FIX(EQ[A],2026-09-23) ELSE pay PAYS 0 END",
    ],
)
def test_lookahead_is_rejected_before_branch_folding(body):
    with pytest.raises(LookAheadObservation):
        prepare(Product_New([P], [body]), D, model=model())


def test_dead_branch_history_is_still_validated_but_fully_expired_fix_is_skipped():
    with pytest.raises(MissingFixing):
        value(([P], [f"IF 1=0 THEN pay PAYS FIX(EQ[A],{H}) ELSE pay PAYS 0 END"]))
    result = value((["SCALE", H], ["2", "pay PAYS SCALE*FIX(EQ[ABSENT])"]), aad=True)
    assert result and all(v == 0.0 for v in result.values())


@pytest.mark.parametrize(
    "body,error",
    [
        ("pay PAYS FIX(EQ[A])+FIX(EQ[B])", MultipleModelIndices),
        ("pay PAYS SPOT()+FIX(EQ[A])", MissingDefaultIndex),
        ("pay PAYS FIX(FX[EUR/USD])", UnsupportedModelObservation),
        ("pay PAYS FIX(EQ[A]>3M)", UnsupportedModelObservation),
    ],
)
def test_future_binding_errors(body, error):
    with pytest.raises(error):
        prepare(Product_New([P], [body]), D, model=model())


def test_hard_history_keeps_parameter_gradients_and_future_fuzzy_fix_is_live():
    rows = (
        ["SCALE", "K", H, P],
        [
            "2",
            "79.95",
            "x=SCALE*FIX(EQ[A]) discarded PAYS 99",
            f"IF FIX(EQ[A],{H})>K:0.2 THEN pay PAYS x ELSE pay PAYS 0 END",
        ],
    )
    valuation = ValuationSettings(evaluation_date=D, fixings=FixingSnapshot({INDEX: {H: 80.0}}))
    result = value(rows, aad=True, valuation=valuation)
    np.testing.assert_allclose(
        [result["PV"], result["d_SCALE"], result["d_K"]], [120.0, 60.0, -800.0], atol=1e-10
    )
    assert result["d_spot"] == result["d_vol"] == 0.0


def test_delayed_payments_share_discount_slot_and_correct_rate_risk():
    settlement = P.add_days(182)
    product = Product_New([P], [f"pay PAYS 10 ON {settlement} pay PAYS 5 ON {settlement}"])
    prepared = prepare(product, D, model=model(rate=0.05))
    assert prepared.sample_defs[0].discount_mats == ((settlement - D) / 365.0,)
    payments = [
        node
        for statement in prepared.events[0]
        for node in A.walk(statement)
        if isinstance(node, A.Pays)
    ]
    assert [payment.discount_id for payment in payments] == [0, 0]
    result = MonteCarloEngine(
        prepared.path_product(), model(rate=0.05), MonteCarloSettings(enable_aad=True)
    ).value(257)
    expected = 15 * math.exp(-0.05 * (settlement - D) / 365)
    np.testing.assert_allclose(
        [result["PV"], result["d_rate"]], [expected, -(settlement - D) / 365 * expected], rtol=1e-14
    )


@pytest.mark.parametrize("payment", [D, P])
def test_unsettled_past_event_payments_are_rejected(payment):
    with pytest.raises(UnsettledDelayedPayment):
        value(([H, P], [f"pay PAYS 10 ON {payment}", "pay PAYS 1"]))


def test_settled_delayed_history_is_evaluated_and_discarded():
    assert value(([H, P], [f"pay PAYS 10 ON {H}", "pay PAYS 1"]))["PV"] == 1.0


def test_valuation_date_is_local_and_conflicting_snapshot_is_rejected():
    old = EvaluationDate_Get()
    try:
        EvaluationDate_Set(D.add_days(30))
        assert value(([P], ["pay PAYS 1"]))["PV"] == 1.0
        assert EvaluationDate_Get() == D.add_days(30)
    finally:
        EvaluationDate_Set(old)
    valuation = ValuationSettings(evaluation_date=D, fixings=FixingSnapshot())
    with pytest.raises(InvalidSetting):
        prepare(
            Product_New([P], ["pay PAYS 1"]),
            D,
            model=model(),
            valuation=valuation,
            fixings=FixingSnapshot(),
        )
    with pytest.raises(InvalidSetting):
        ValuationSettings(today_fixing_policy="model")


def test_vector_history_can_contain_a_fixing_and_live_parameter():
    rows = (
        ["SCALE", H, P],
        ["2", "APPEND(v,SCALE*FIX(EQ[A]))", "APPEND(v,SCALE*10) pay PAYS AVERAGE(v)"],
    )
    result = value(
        rows,
        aad=True,
        valuation=ValuationSettings(evaluation_date=D, fixings=FixingSnapshot({INDEX: {H: 80.0}})),
    )
    np.testing.assert_allclose([result["PV"], result["d_SCALE"]], [90.0, 45.0], atol=1e-12)
