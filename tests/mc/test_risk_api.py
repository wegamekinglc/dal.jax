"""Unit behaviour of the risk request machinery (no oracle)."""

import numpy as np
import pytest

from dal_jax.api import (
    BSModelData_New,
    JacobianRiskRequest_,
    MonteCarlo_ValueWithJacobianRisk,
    MonteCarlo_ValueWithRisk,
    MonteCarlo_ValueWithWeightedRisk,
    Product_New,
    RiskRequest_,
    ValuationSettings,
    WeightedRiskRequest_,
)
from dal_jax.dates import Date
from dal_jax.errors import (
    InvalidRiskRequest,
    InvalidSetting,
    InvalidWeightedRiskRequest,
    UnsupportedWeightedRisk,
)

_MODEL = (100.0, 0.2, 0.03, 0.01)
_VALUATION = ValuationSettings(evaluation_date=Date.ymd(2022, 9, 15))
#  Three scalar outputs (a, b, c with c the payoff) for multi-output requests.
_THREE_DATES, _THREE_EVENTS = (
    [(2024, 1, 2)],
    ["a = 1\nb = 2\nc = 3\na pays spot()\nb pays spot() * 2\nc pays spot() * 3"],
)


def _product(dates=((2024, 1, 2),), events=("p pays spot()",)):
    return Product_New([Date.ymd(*d) for d in dates], list(events))


def _risk(dates, events, n_paths=256, *, request=None):
    return MonteCarlo_ValueWithRisk(
        _product(dates, events),
        BSModelData_New(*_MODEL),
        n_paths,
        valuation=_VALUATION,
        request=request,
    )


def _weighted(dates, events, n_paths=256, *, request=None):
    return MonteCarlo_ValueWithWeightedRisk(
        _product(dates, events),
        BSModelData_New(*_MODEL),
        n_paths,
        valuation=_VALUATION,
        request=request,
    )


def _jacobian(dates, events, n_paths=256, *, request=None):
    return MonteCarlo_ValueWithJacobianRisk(
        _product(dates, events),
        BSModelData_New(*_MODEL),
        n_paths,
        valuation=_VALUATION,
        request=request,
    )


def test_unknown_input_label_is_rejected():
    message = "InvalidRiskRequest: unknown input ID; input=nonsense"
    with pytest.raises(InvalidRiskRequest, match=message) as error:
        _risk([(2024, 1, 2)], ["p pays spot()"], request=RiskRequest_(inputs=["nonsense"]))
    assert str(error.value) == message
    assert error.value.detail == "unknown input ID; input=nonsense"


def test_input_selector_type_is_rejected():
    message = (
        "InvalidRiskRequest: RiskRequest_; inputs; type=int, value=0; "
        "expected list or tuple, or None"
    )
    with pytest.raises(InvalidRiskRequest, match=message) as error:
        _risk([(2024, 1, 2)], ["p pays spot()"], request=RiskRequest_(inputs=0))
    assert str(error.value) == message


def test_weighted_unknown_input_keeps_the_plain_request_code():
    message = "InvalidRiskRequest: unknown input ID; input=nonsense"
    with pytest.raises(InvalidRiskRequest, match=message) as error:
        _weighted(
            [(2024, 1, 2)], ["p pays spot()"], request=WeightedRiskRequest_(inputs=["nonsense"])
        )
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("outputs", "weights", "detail"),
    [
        (["nonsense"], [1.0], "unknown scalar output ID; output=nonsense"),
        (["payoff", "PAYOFF"], [1.0, 2.0], "repeated output ID; output=PAYOFF"),
        ([], [], "output selection must be nonempty; field=outputs"),
    ],
)
def test_weighted_rejects_bad_output_selectors(outputs, weights, detail):
    message = f"InvalidWeightedRiskRequest: {detail}"
    with pytest.raises(InvalidWeightedRiskRequest, match=message) as error:
        _weighted(
            _THREE_DATES,
            _THREE_EVENTS,
            request=WeightedRiskRequest_(outputs=outputs, weights=weights),
        )
    assert str(error.value) == message


def test_weighted_weight_count_must_match_selected_outputs():
    message = "InvalidWeightedRiskRequest: weight count must match selected outputs; field=weights"
    with pytest.raises(InvalidWeightedRiskRequest, match=message) as error:
        _weighted(_THREE_DATES, _THREE_EVENTS, request=WeightedRiskRequest_(weights=[1.0, 2.0]))
    assert str(error.value) == message
    assert error.value.detail == "weight count must match selected outputs; field=weights"


@pytest.mark.parametrize(
    ("weights", "detail"),
    [
        (2.0, "weights; type=float, value=2.0; expected list or tuple, or None"),
        (["a"], "weights; type=str, value='a'; expected int or float, excluding bool"),
        ([True], "weights; type=bool, value=True; expected int or float, excluding bool"),
    ],
)
def test_weighted_rejects_bad_weight_types(weights, detail):
    message = f"InvalidWeightedRiskRequest: WeightedRiskRequest_; {detail}"
    with pytest.raises(InvalidWeightedRiskRequest, match=message) as error:
        _weighted([(2024, 1, 2)], ["p pays spot()"], request=WeightedRiskRequest_(weights=weights))
    assert str(error.value) == message


def test_weighted_defaults_to_unit_weights_per_selected_output():
    result = _weighted(
        _THREE_DATES, _THREE_EVENTS, request=WeightedRiskRequest_(outputs=["output:0", "payoff"])
    )
    assert result.weights == [1.0, 1.0]
    assert result.weighted_value == pytest.approx(sum(result.component_means), rel=1e-12)
    no_request = _weighted([(2024, 1, 2)], ["p pays spot()"])
    assert no_request.weights == [1.0]
    assert no_request.weighted_value == pytest.approx(no_request.component_means[0], rel=1e-12)


def test_weighted_aggregates_each_selected_slot():
    outputs, weights = ["output:0", "payoff"], [3.0, 5.0]
    per_slot = _jacobian(_THREE_DATES, _THREE_EVENTS, request=JacobianRiskRequest_(outputs=outputs))
    result = _weighted(
        _THREE_DATES,
        _THREE_EVENTS,
        request=WeightedRiskRequest_(outputs=outputs, weights=weights),
    )
    assert [c.id for c in result.output_axis] == outputs
    assert result.weighted_value == pytest.approx(
        weights[0] * per_slot.values[0] + weights[1] * per_slot.values[1], rel=1e-9
    )
    np.testing.assert_allclose(result.component_means, per_slot.values, rtol=1e-9)
    rows = np.asarray(per_slot.jacobian)
    np.testing.assert_allclose(
        np.asarray(result.jacobian),
        (weights[0] * rows[0] + weights[1] * rows[1])[None, :],
        rtol=1e-9,
        atol=1e-12,
    )
    assert np.asarray(result.jacobian).shape == (1, len(result.input_axis))


def test_weighted_rejects_exercise():
    message = "UnsupportedWeightedRisk: EXERCISE requires a separate weighted estimator"
    dates, events = [(2024, 1, 2), (2024, 7, 2)], ["x = 1\nexercise x", "y PAYS SPOT()"]
    with pytest.raises(UnsupportedWeightedRisk, match=message) as error:
        _weighted(dates, events)
    assert str(error.value) == message
    assert error.value.detail == "EXERCISE requires a separate weighted estimator"


def test_weighted_rejects_expired_products():
    message = "UnsupportedWeightedRisk: fully expired products are not supported"
    with pytest.raises(UnsupportedWeightedRisk, match=message) as error:
        _weighted([(2022, 1, 15)], ["p pays 1.0"])
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("bad_request", "name"),
    [(RiskRequest_(), "RiskRequest_"), (JacobianRiskRequest_(), "JacobianRiskRequest_")],
    ids=["risk-request", "jacobian-request"],
)
def test_weighted_enforces_request_type(bad_request, name):
    message = (
        f"InvalidSetting: MonteCarlo_ValueWithWeightedRisk; request; type={name}; "
        f"expected WeightedRiskRequest_ or None"
    )
    with pytest.raises(InvalidSetting, match=message) as error:
        _weighted([(2024, 1, 2)], ["p pays spot()"], request=bad_request)
    assert str(error.value) == message
