"""Request-driven risk API parity with dal-python 2026.11.1."""

import pytest

from dal_jax.api import Product_Get_RiskOutputs, Product_New
from dal_jax.dates import Date
from dal_jax.errors import InvalidWeightedRiskRequest
from dal_jax.mc.risk import payoff_outputs

pytestmark = pytest.mark.oracle


def _cells(dates):
    return [Date.ymd(*d) if isinstance(d, tuple) else d for d in dates]


def _native_product(dal, dates, events, settings=None):
    return dal.Product_New(
        [dal.Date_(*d) if isinstance(d, tuple) else d for d in dates],
        events,
        settings=settings if settings is not None else dal.ScriptProductSettings_(),
    )


#  The output axis is one coordinate per distinct scalar ``=``/``PAYS`` target, in
#  first-assignment order over date-sorted events; the last one is id='payoff'.
RISK_CASES = [
    pytest.param(
        [(2024, 1, 2)],
        ["p pays spot()"],
        id="single-payoff",
    ),
    pytest.param(
        [(2024, 1, 2)],
        ["a = 1\nb = 2\na pays spot()\nb pays spot() * 2"],
        id="two-pays-one-event",
    ),
    pytest.param(
        [(2024, 1, 2), (2024, 7, 2)],
        ["a pays spot()", "b pays spot() * 2"],
        id="two-events",
    ),
    pytest.param(
        [(2024, 1, 2), (2024, 7, 2), (2025, 1, 2)],
        ["a pays spot()", "b pays spot() * 2", "c pays spot() * 3"],
        id="three-events",
    ),
    pytest.param(
        [(2024, 7, 2), (2024, 1, 2)],
        ["a pays spot()", "b pays spot() * 2"],
        id="reversed-dates",
    ),
    pytest.param(
        [(2024, 1, 2)],
        ["x = 1\ny = x * 2"],
        id="no-pays",
    ),
    pytest.param(
        [(2024, 1, 2)],
        ["a = 1\na = 2\na pays spot()"],
        id="repeated-target",
    ),
    pytest.param(
        [(2024, 1, 2)],
        ["a pays spot()\nb = 2"],
        id="pays-then-assign",
    ),
    pytest.param(
        [(2024, 1, 2)],
        ["if spot() > 100 then\na pays spot()\nelse\nb pays spot() * 2\nend"],
        id="pays-in-if",
    ),
    pytest.param(
        [(2024, 1, 2)],
        ["A = 1\na PAYS SPOT()"],
        id="case-insensitive-dedupe",
    ),
    pytest.param(
        ["K", (2024, 1, 2), (2024, 7, 2)],
        ["100", "APPEND(v,SPOT())", "pay PAYS MAX(AVERAGE(v)-K,0)"],
        id="const-and-vector-excluded",
    ),
    pytest.param(
        [(2024, 7, 2)],
        ["FOR(i, 0, 2)\ns = i * spot()\nEND\ns PAYS s"],
        id="for-loop-index-excluded",
    ),
    pytest.param(
        [(2024, 1, 2), (2024, 1, 2)],
        ["b = 2", "a = 1"],
        id="same-date-rows",
    ),
    pytest.param(
        [(2022, 1, 15), (2024, 1, 2)],
        ["a = 1", "b = 2"],
        id="past-event-counts",
    ),
    pytest.param(
        [(2024, 2, 1), (2024, 3, 1), (2024, 4, 1)],
        ["pay PAYS spot() on 2024-02-15", "pay PAYS spot() * 2 on 2024-03-15", "pay PAYS 3.0"],
        id="schedule-same-variable",
    ),
    pytest.param(
        [(2024, 1, 15), (2024, 7, 2)],
        ["x = 1\nexercise x", "y PAYS SPOT()"],
        id="exercise-plus-pays",
    ),
    pytest.param(
        [(2024, 7, 2)],
        ["FOR(i, 0, 2)\ns = i * spot()\nEND\nq PAYS s"],
        id="for-body-only-target",
    ),
    pytest.param(
        [(2024, 7, 2)],
        ["APPEND(v,SPOT())\nv[0] = 2.0\nq PAYS SUM(v)"],
        id="vector-entry-assign-excluded",
    ),
]


@pytest.mark.parametrize(("dates", "events"), RISK_CASES)
def test_risk_outputs_match_native(dal, dates, events):
    ours = Product_Get_RiskOutputs(Product_New(_cells(dates), list(events)))
    theirs = dal.Product_Get_RiskOutputs(_native_product(dal, dates, events))
    assert [(c.id, c.label, c.slot) for c in ours] == [(c.id, c.label, c.slot) for c in theirs]


def test_risk_outputs_expected_coordinates():
    from dal_jax.mc.risk import RiskOutputCoordinate

    outputs = Product_Get_RiskOutputs(
        Product_New(_cells([(2024, 1, 2)]), ["a = 1\nb = 2\na pays spot()\nb pays spot() * 2"])
    )
    assert outputs == [
        RiskOutputCoordinate(id="output:0", label="a", slot=0),
        RiskOutputCoordinate(id="payoff", label="b", slot=1),
    ]


_RECEIVERLESS = (
    "InvalidWeightedRiskRequest: indexed scalar default receiver is required; output=payoff"
)


@pytest.mark.parametrize(
    ("dates", "events"),
    [
        pytest.param(
            [(2024, 1, 2), (2024, 7, 2)], ["s = spot()", "exercise s"], id="exercise-only"
        ),
        pytest.param([], [], id="empty"),
        pytest.param([(2024, 1, 2)], ["APPEND(v,SPOT())"], id="vector-only"),
    ],
)
def test_risk_outputs_require_scalar_receiver(dal, dates, events):
    with pytest.raises(Exception, match=_RECEIVERLESS):
        dal.Product_Get_RiskOutputs(_native_product(dal, dates, events))
    with pytest.raises(InvalidWeightedRiskRequest, match=_RECEIVERLESS) as error:
        Product_Get_RiskOutputs(Product_New(_cells(dates), events))
    assert str(error.value) == _RECEIVERLESS
    assert error.value.detail == "indexed scalar default receiver is required; output=payoff"


def test_payoff_outputs_accepts_parsed_product():
    data = Product_New(_cells([(2024, 1, 2)]), ["a pays spot()"])
    assert payoff_outputs(data) == payoff_outputs(data.product())
