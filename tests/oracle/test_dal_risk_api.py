"""Request-driven risk API parity with dal-python 2026.11.1."""

import dataclasses

import numpy as np
import pytest

from dal_jax.api import (
    BSModelData_New,
    CorrelatedBSModelData_New,
    JacobianRiskRequest_,
    MonteCarlo_ValueWithJacobianRisk,
    MonteCarlo_ValueWithRisk,
    MonteCarlo_ValueWithWeightedRisk,
    Product_Get_RiskOutputs,
    Product_New,
    RiskRequest_,
    ValuationSettings,
    WeightedRiskRequest_,
)
from dal_jax.dates import Date
from dal_jax.errors import (
    InvalidJacobianRiskRequest,
    InvalidRiskRequest,
    InvalidSetting,
    InvalidWeightedRiskRequest,
    UnsupportedJacobianRisk,
    UnsupportedWeightedRisk,
)
from dal_jax.mc.risk import payoff_outputs
from dal_jax.script.product import ScriptProductSettings

pytestmark = pytest.mark.oracle


def _cells(dates):
    return [Date.ymd(*d) if isinstance(d, tuple) else d for d in dates]


def _native_product(dal, dates, events, settings=None):
    return dal.Product_New(
        [dal.Date_(*d) if isinstance(d, tuple) else d for d in dates],
        events,
        settings=settings if settings is not None else dal.ScriptProductSettings_(),
    )


#  The output axis is one coordinate per scalar variable of the var table, in
#  first-reference order over date-sorted events (reads count); the last one is
#  id='payoff' and slot is the var-table index.
RISK_CASES = [
    pytest.param(
        [(2024, 1, 2)],
        ["p pays spot()"],
        id="single-payoff",
    ),
    pytest.param(
        [(2024, 1, 2)],
        ["if w > 1 then\nq = 1\nelse\nq = 2\nend\nq PAYS q"],
        id="condition-only-var-included",
    ),
    pytest.param(
        [(2024, 1, 2)],
        ["z = x\nx = spot()\nz PAYS z"],
        id="read-before-assign",
    ),
    pytest.param(
        [(2024, 1, 2)],
        ["p PAYS MAX(SPOT()-W,0)"],
        id="rhs-only-var-included-target-first",
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


_MODEL = (100.0, 0.2, 0.03, 0.01)
_VALUATION = ValuationSettings(evaluation_date=Date.ymd(2022, 9, 15))


def _axis_records(result):
    return [
        (c.family, c.id, c.label, c.ordinal, c.native_unit, c.physical_unit, c.report_scale)
        for c in result.input_axis
    ]


def _native_valuation(dal):
    return dal.ScriptValuationSettings_(evaluation_date=dal.Date_(2022, 9, 15))


def _ours(dates, events, n_paths=1024, *, request=None, settings=None):
    return MonteCarlo_ValueWithRisk(
        Product_New(_cells(dates), list(events), settings=settings),
        BSModelData_New(*_MODEL),
        n_paths,
        valuation=_VALUATION,
        request=request,
    )


def _theirs(dal, dates, events, n_paths=1024, *, request=None, settings=None):
    return dal.MonteCarlo_ValueWithRisk(
        _native_product(dal, dates, events, settings=settings),
        dal.BSModelData_New(*_MODEL),
        n_paths,
        valuation=_native_valuation(dal),
        request=request,
    )


def _assert_risk_parity(ours, theirs, *, rtol=1e-9):
    np.testing.assert_allclose(ours.values, theirs.values, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(ours.jacobian), np.asarray(theirs.jacobian.to_rows()), rtol=rtol, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(ours.reported_jacobian),
        np.asarray(theirs.reported_jacobian.to_rows()),
        rtol=rtol,
        atol=1e-12,
    )
    assert _axis_records(ours) == _axis_records(theirs)
    assert ours.output_ids == list(theirs.output_ids)
    assert [c.id for c in ours.complete_input_axis] == [c.id for c in theirs.complete_input_axis]
    for key, value in theirs.legacy_values.items():
        assert ours.legacy_values[key] == pytest.approx(value, rel=rtol)
    assert ours.provenance.method == theirs.provenance.method
    assert ours.provenance.normalization == theirs.provenance.normalization
    assert ours.provenance.execution.paths_per_replicate == (
        theirs.provenance.execution.paths_per_replicate
    )


def test_value_with_risk_matches_native(dal):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    _assert_risk_parity(_ours(dates, events), _theirs(dal, dates, events))


def test_value_with_risk_selects_inputs_by_id(dal):
    dates, events = [(2024, 1, 2)], ["a = 1\nb = 2\na pays spot()\nb pays spot() * 2"]
    for inputs in (["model:2", "model:0"], ["model:3"], []):
        ours = _ours(dates, events, request=RiskRequest_(inputs=inputs))
        theirs = _theirs(dal, dates, events, request=dal.RiskRequest_(inputs=inputs))
        _assert_risk_parity(ours, theirs)


def test_value_with_risk_axis_includes_script_constants(dal):
    dates, events = ["K", (2024, 1, 2)], ["100", "p pays MAX(SPOT()-K,0)"]
    _assert_risk_parity(_ours(dates, events), _theirs(dal, dates, events))
    ours = _ours(dates, events)
    assert [c.id for c in ours.input_axis] == [
        "model:0",
        "model:1",
        "model:2",
        "model:3",
        "constant:0",
    ]
    assert ours.input_axis[-1].family == "constant"
    assert ours.input_axis[-1].native_unit == "script-number"
    assert ours.input_axis[-1].value == 100.0


def test_value_with_risk_output_must_select_the_single_payoff(dal):
    dates, events = [(2024, 1, 2)], ["a = 1\nb = 2\na pays spot()\nb pays spot() * 2"]
    message = "InvalidRiskRequest: outputs must select the single supported payoff"
    for outputs in (["a"], ["output:0"], ["nonsense"], ["payoff", "payoff"], []):
        with pytest.raises(RuntimeError, match="InvalidRiskRequest"):
            _theirs(dal, dates, events, request=dal.RiskRequest_(outputs=outputs))
        with pytest.raises(InvalidRiskRequest, match=message) as error:
            _ours(dates, events, request=RiskRequest_(outputs=outputs))
        assert str(error.value) == message
    selected = _ours(dates, events, request=RiskRequest_(outputs=["PAYOFF"]))
    assert selected.output_ids == ["payoff"]
    _assert_risk_parity(selected, _theirs(dal, dates, events))


@pytest.mark.parametrize(
    ("inputs", "detail"),
    [
        (["nonsense"], "unknown input ID; input=nonsense"),
        (["vol"], "unknown input ID; input=vol"),
        (["model:1", "model:1"], "repeated input ID; input=model:1"),
        (["model:1", "MODEL:1"], "repeated input ID; input=MODEL:1"),
    ],
)
def test_value_with_risk_rejects_bad_input_selectors(dal, inputs, detail):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    message = f"InvalidRiskRequest: {detail}"
    with pytest.raises(RuntimeError, match="InvalidRiskRequest"):
        _theirs(dal, dates, events, request=dal.RiskRequest_(inputs=inputs))
    with pytest.raises(InvalidRiskRequest, match=message) as error:
        _ours(dates, events, request=RiskRequest_(inputs=inputs))
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("selector", "detail"),
    [
        (0, "inputs; type=int, value=0; expected list or tuple, or None"),
        ("model:0", "inputs; type=str, value='model:0'; expected list or tuple, or None"),
        (["model:0", 1], "inputs; type=int, value=1; expected str or DAL String_ without NUL"),
        ([0], "inputs; type=int, value=0; expected str or DAL String_ without NUL"),
    ],
)
def test_value_with_risk_validates_input_selector_types(dal, selector, detail):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    message = f"InvalidRiskRequest: RiskRequest_; {detail}"
    with pytest.raises((RuntimeError, TypeError), match="InvalidRiskRequest"):
        _theirs(dal, dates, events, request=dal.RiskRequest_(inputs=selector))
    with pytest.raises(InvalidRiskRequest, match=message) as error:
        _ours(dates, events, request=RiskRequest_(inputs=selector))
    assert str(error.value) == message


def _assert_jacobian_parity(ours, theirs, *, rtol=1e-9):
    np.testing.assert_allclose(ours.values, theirs.values, rtol=rtol)
    np.testing.assert_allclose(
        np.asarray(ours.jacobian), np.asarray(theirs.jacobian.to_rows()), rtol=rtol, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(ours.reported_jacobian),
        np.asarray(theirs.reported_jacobian.to_rows()),
        rtol=rtol,
        atol=1e-12,
    )
    assert _axis_records(ours) == _axis_records(theirs)
    assert [c.id for c in ours.complete_input_axis] == [c.id for c in theirs.complete_input_axis]
    assert [(c.id, c.label, c.slot) for c in ours.output_axis] == [
        (c.id, c.label, c.slot) for c in theirs.output_axis
    ]
    assert [(c.id, c.label, c.slot) for c in ours.complete_output_axis] == [
        (c.id, c.label, c.slot) for c in theirs.complete_output_axis
    ]
    assert ours.provenance.method == theirs.provenance.method == "NativeAAD"
    assert ours.provenance.normalization == theirs.provenance.normalization
    assert ours.execution.replay_attempts == theirs.execution.replay_attempts
    assert ours.execution.executed_paths == theirs.execution.executed_paths
    assert list(ours.execution.actual_widths) == list(theirs.execution.actual_widths)


def _ours_jacobian(dates, events, n_paths=256, *, request=None):
    return MonteCarlo_ValueWithJacobianRisk(
        Product_New(_cells(dates), list(events)),
        BSModelData_New(*_MODEL),
        n_paths,
        valuation=_VALUATION,
        request=request,
    )


def _theirs_jacobian(dal, dates, events, n_paths=256, *, request=None):
    return dal.MonteCarlo_ValueWithJacobianRisk(
        _native_product(dal, dates, events),
        dal.BSModelData_New(*_MODEL),
        n_paths,
        valuation=_native_valuation(dal),
        request=request,
    )


def test_value_with_jacobian_risk_matches_native(dal):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    _assert_jacobian_parity(_ours_jacobian(dates, events), _theirs_jacobian(dal, dates, events))
    _assert_jacobian_parity(
        _ours_jacobian(dates, events, request=JacobianRiskRequest_(inputs=["model:1"])),
        _theirs_jacobian(dal, dates, events, request=dal.JacobianRiskRequest_(inputs=["model:1"])),
    )


def test_value_with_jacobian_risk_prices_each_selected_slot(dal):
    dates = [(2024, 1, 2)]
    events = ["a = 1\nb = 2\nc = 3\na pays spot()\nb pays spot() * 2\nc pays spot() * 3"]
    for outputs in (None, ["output:0", "payoff"], ["payoff", "output:0"], ["output:1"]):
        ours = _ours_jacobian(
            dates,
            events,
            request=None if outputs is None else JacobianRiskRequest_(outputs=outputs),
        )
        theirs = _theirs_jacobian(
            dal,
            dates,
            events,
            request=None if outputs is None else dal.JacobianRiskRequest_(outputs=outputs),
        )
        _assert_jacobian_parity(ours, theirs)
    all_slots = _ours_jacobian(
        dates, events, request=JacobianRiskRequest_(outputs=["output:0", "output:1", "payoff"])
    )
    assert [c.id for c in all_slots.output_axis] == ["output:0", "output:1", "payoff"]
    assert [c.id for c in all_slots.complete_output_axis] == ["output:0", "output:1", "payoff"]
    assert all_slots.execution.replay_attempts == 3
    np.testing.assert_allclose(all_slots.values[1], 2 * all_slots.values[0], rtol=1e-12)
    np.testing.assert_allclose(all_slots.values[2], 3 * all_slots.values[0], rtol=1e-12)


@pytest.mark.parametrize(
    ("outputs", "detail"),
    [
        (["a"], "unknown scalar output ID; output=a"),
        (["nonsense"], "unknown scalar output ID; output=nonsense"),
        (["payoff", "payoff"], "repeated output ID; output=payoff"),
        ([], "output selection must be nonempty; field=outputs"),
    ],
)
def test_value_with_jacobian_risk_rejects_bad_output_selectors(dal, outputs, detail):
    dates, events = [(2024, 1, 2)], ["a = 1\nb = 2\na pays spot()\nb pays spot() * 2"]
    message = f"InvalidJacobianRiskRequest: {detail}"
    with pytest.raises(RuntimeError, match="InvalidJacobianRiskRequest"):
        _theirs_jacobian(dal, dates, events, request=dal.JacobianRiskRequest_(outputs=outputs))
    with pytest.raises(InvalidJacobianRiskRequest, match=message) as error:
        _ours_jacobian(dates, events, request=JacobianRiskRequest_(outputs=outputs))
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("selector", "error_class", "message"),
    [
        (
            0,
            InvalidJacobianRiskRequest,
            "InvalidJacobianRiskRequest: JacobianRiskRequest_; inputs; type=int, value=0; "
            "expected list or tuple, or None",
        ),
        (
            ["model:0", 1],
            InvalidJacobianRiskRequest,
            "InvalidJacobianRiskRequest: JacobianRiskRequest_; inputs; type=int, value=1; "
            "expected str or DAL String_ without NUL",
        ),
        (
            ["nonsense"],
            InvalidRiskRequest,
            "InvalidRiskRequest: unknown input ID; input=nonsense",
        ),
    ],
)
def test_value_with_jacobian_risk_validates_input_selectors(dal, selector, error_class, message):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    with pytest.raises((RuntimeError, TypeError), match=message.split(":")[0]):
        _theirs_jacobian(dal, dates, events, request=dal.JacobianRiskRequest_(inputs=selector))
    with pytest.raises(error_class, match=message) as error:
        _ours_jacobian(dates, events, request=JacobianRiskRequest_(inputs=selector))
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("function", "make_request", "detail"),
    [
        (
            "MonteCarlo_ValueWithRisk",
            lambda: JacobianRiskRequest_(),
            "MonteCarlo_ValueWithRisk; request; type=JacobianRiskRequest_; "
            "expected RiskRequest_ or None",
        ),
        (
            "MonteCarlo_ValueWithRisk",
            WeightedRiskRequest_,
            "MonteCarlo_ValueWithRisk; request; type=WeightedRiskRequest_; "
            "expected RiskRequest_ or None",
        ),
        (
            "MonteCarlo_ValueWithJacobianRisk",
            lambda: RiskRequest_(),
            "MonteCarlo_ValueWithJacobianRisk; request; type=RiskRequest_; "
            "expected JacobianRiskRequest_ or None",
        ),
        (
            "MonteCarlo_ValueWithJacobianRisk",
            WeightedRiskRequest_,
            "MonteCarlo_ValueWithJacobianRisk; request; type=WeightedRiskRequest_; "
            "expected JacobianRiskRequest_ or None",
        ),
        (
            "MonteCarlo_ValueWithWeightedRisk",
            lambda: RiskRequest_(),
            "MonteCarlo_ValueWithWeightedRisk; request; type=RiskRequest_; "
            "expected WeightedRiskRequest_ or None",
        ),
        (
            "MonteCarlo_ValueWithWeightedRisk",
            lambda: JacobianRiskRequest_(),
            "MonteCarlo_ValueWithWeightedRisk; request; type=JacobianRiskRequest_; "
            "expected WeightedRiskRequest_ or None",
        ),
    ],
)
def test_risk_variants_enforce_request_types(dal, function, make_request, detail):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    entry = globals()[function]
    message = f"InvalidSetting: {detail}"
    native_entry = getattr(dal, function)
    with pytest.raises(TypeError, match="InvalidSetting"):
        native_entry(
            _native_product(dal, dates, events),
            dal.BSModelData_New(*_MODEL),
            64,
            valuation=_native_valuation(dal),
            request=make_request(),
        )
    with pytest.raises(InvalidSetting, match=message) as error:
        entry(
            Product_New(_cells(dates), list(events)),
            BSModelData_New(*_MODEL),
            64,
            valuation=_VALUATION,
            request=make_request(),
        )
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("events", "zero_slot"),
    [
        (["if w > 1 then\nq = 1\nelse\nq = 2\nend\nq PAYS q"], 0),
        (["z = x\nx = spot()\nz PAYS z"], 0),
        (["p PAYS MAX(SPOT()-W,0)"], 1),
    ],
    ids=["condition-only-var", "read-before-assign", "rhs-only-var"],
)
def test_value_with_jacobian_risk_prices_each_var_table_slot(dal, events, zero_slot):
    dates = [(2024, 1, 2)]
    outputs = ["output:0", "payoff"]
    ours = _ours_jacobian(dates, events, request=JacobianRiskRequest_(outputs=outputs))
    theirs = _theirs_jacobian(dal, dates, events, request=dal.JacobianRiskRequest_(outputs=outputs))
    _assert_jacobian_parity(ours, theirs)
    #  A variable with no accumulated payments prices to zero with an all-zero row.
    assert ours.values[zero_slot] == 0.0
    row = np.asarray(ours.jacobian)[zero_slot]
    assert row.tolist() == [0.0] * row.shape[0]


def test_value_with_risk_prices_the_payoff_variable_not_the_payoff_target(dal):
    dates, events = [(2024, 1, 2)], ["if w > 1 then\nq = 1\nelse\nq = 2\nend\nq PAYS q"]
    _assert_risk_parity(_ours(dates, events, 512), _theirs(dal, dates, events, 512))


_EXERCISE_MESSAGE = "UnsupportedJacobianRisk: EXERCISE requires a separate Jacobian estimator"


def test_value_with_jacobian_risk_rejects_exercise(dal):
    dates = [(2024, 1, 2), (2024, 7, 2)]
    events = ["x = 1\nexercise x", "y PAYS SPOT()"]
    for request, native_request in (
        (None, None),
        (JacobianRiskRequest_(inputs=["nonsense"]), dal.JacobianRiskRequest_(inputs=["nonsense"])),
    ):
        with pytest.raises(RuntimeError, match="UnsupportedJacobianRisk"):
            _theirs_jacobian(dal, dates, events, request=native_request)
        with pytest.raises(UnsupportedJacobianRisk, match=_EXERCISE_MESSAGE) as error:
            _ours_jacobian(dates, events, request=request)
        assert str(error.value) == _EXERCISE_MESSAGE
    #  The plain AAD variant still supports EXERCISE products.
    _assert_risk_parity(_ours(dates, events, 256), _theirs(dal, dates, events, 256))


@pytest.mark.parametrize(
    ("entry", "native_entry", "make_request", "native_request", "error_class", "message"),
    [
        (
            MonteCarlo_ValueWithRisk,
            "MonteCarlo_ValueWithRisk",
            RiskRequest_(inputs=["nonsense"], outputs=["a"]),
            lambda dal: dal.RiskRequest_(inputs=["nonsense"], outputs=["a"]),
            InvalidRiskRequest,
            "InvalidRiskRequest: unknown input ID; input=nonsense",
        ),
        (
            MonteCarlo_ValueWithJacobianRisk,
            "MonteCarlo_ValueWithJacobianRisk",
            JacobianRiskRequest_(inputs=["nonsense"], outputs=["a"]),
            lambda dal: dal.JacobianRiskRequest_(inputs=["nonsense"], outputs=["a"]),
            InvalidJacobianRiskRequest,
            "InvalidJacobianRiskRequest: unknown scalar output ID; output=a",
        ),
    ],
    ids=["aad-validates-inputs-first", "jacobian-validates-outputs-first"],
)
def test_risk_selector_validation_order(
    dal, entry, native_entry, make_request, native_request, error_class, message
):
    dates, events = [(2024, 1, 2)], ["a = 1\nb = 2\na pays spot()\nb pays spot() * 2"]
    with pytest.raises(RuntimeError, match=message.split(":")[0]):
        getattr(dal, native_entry)(
            _native_product(dal, dates, events),
            dal.BSModelData_New(*_MODEL),
            64,
            valuation=_native_valuation(dal),
            request=native_request(dal),
        )
    with pytest.raises(error_class, match=message) as error:
        entry(
            Product_New(_cells(dates), list(events)),
            BSModelData_New(*_MODEL),
            64,
            valuation=_VALUATION,
            request=make_request,
        )
    assert str(error.value) == message


def test_value_with_risk_correlated_axis_matches_native(dal):
    model = CorrelatedBSModelData_New(
        ["EQ[A]", "EQ[B]"], [100.0, 90.0], [0.2, 0.3], [0.01, 0.02], 0.03, [[1.0, 0.5], [0.5, 1.0]]
    )
    nmodel = dal.CorrelatedBSModelData_New(
        ["EQ[A]", "EQ[B]"],
        [100.0, 90.0],
        [0.2, 0.3],
        [0.01, 0.02],
        0.03,
        dal.DoubleMatrix_([[1.0, 0.5], [0.5, 1.0]]),
    )
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    settings = ScriptProductSettings(default_index="EQ[A]")
    ours = MonteCarlo_ValueWithRisk(
        Product_New(_cells(dates), list(events), settings=settings),
        model,
        1024,
        valuation=_VALUATION,
    )
    theirs = dal.MonteCarlo_ValueWithRisk(
        dal.Product_New(
            [dal.Date_(*dates[0])],
            list(events),
            settings=dal.ScriptProductSettings_(default_index="EQ[A]"),
        ),
        nmodel,
        1024,
        valuation=_native_valuation(dal),
    )
    assert _axis_records(ours) == _axis_records(theirs)
    np.testing.assert_allclose(
        np.asarray(ours.jacobian), np.asarray(theirs.jacobian.to_rows()), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(ours.values, theirs.values, rtol=1e-12)


#  Three scalar outputs (a, b, c with c the payoff) for multi-output weighted requests.
_THREE_DATES, _THREE_EVENTS = (
    [(2024, 1, 2)],
    ["a = 1\nb = 2\nc = 3\na pays spot()\nb pays spot() * 2\nc pays spot() * 3"],
)


def _ours_weighted(dates, events, n_paths=1024, *, request=None):
    return MonteCarlo_ValueWithWeightedRisk(
        Product_New(_cells(dates), list(events)),
        BSModelData_New(*_MODEL),
        n_paths,
        valuation=_VALUATION,
        request=request,
    )


def _theirs_weighted(dal, dates, events, n_paths=1024, *, request=None):
    return dal.MonteCarlo_ValueWithWeightedRisk(
        _native_product(dal, dates, events),
        dal.BSModelData_New(*_MODEL),
        n_paths,
        valuation=_native_valuation(dal),
        request=request,
    )


def _assert_weighted_parity(ours, theirs, *, rtol=1e-9):
    assert ours.weighted_value == pytest.approx(theirs.weighted_value, rel=rtol)
    np.testing.assert_allclose(ours.component_means, theirs.component_means, rtol=rtol)
    np.testing.assert_allclose(ours.weights, theirs.weights, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(ours.jacobian), np.asarray(theirs.jacobian.to_rows()), rtol=rtol, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(ours.reported_jacobian),
        np.asarray(theirs.reported_jacobian.to_rows()),
        rtol=rtol,
        atol=1e-12,
    )
    assert _axis_records(ours) == _axis_records(theirs)
    assert [c.id for c in ours.complete_input_axis] == [c.id for c in theirs.complete_input_axis]
    assert [(c.id, c.label, c.slot) for c in ours.output_axis] == [
        (c.id, c.label, c.slot) for c in theirs.output_axis
    ]
    assert ours.provenance.method == theirs.provenance.method == "NativeAAD"
    assert ours.provenance.normalization == theirs.provenance.normalization
    assert ours.weighted_value == pytest.approx(
        sum(w * m for w, m in zip(ours.weights, ours.component_means)), rel=1e-12
    )


def test_weighted_risk_matches_native(dal):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    ours = _ours_weighted(dates, events)
    theirs = _theirs_weighted(dal, dates, events)
    _assert_weighted_parity(ours, theirs)
    assert ours.weights == [1.0]
    #  Native's weighted result exposes exactly these attributes, ours field-for-field.
    assert sorted(f.name for f in dataclasses.fields(ours)) == sorted(
        attr for attr in dir(theirs) if not attr.startswith("_")
    )


def test_weighted_risk_scales_with_weights(dal):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    base = _ours_weighted(dates, events)
    for weights in ([2.0], [2], (2.0,)):
        ours = _ours_weighted(dates, events, request=WeightedRiskRequest_(weights=weights))
        theirs = _theirs_weighted(
            dal, dates, events, request=dal.WeightedRiskRequest_(weights=weights)
        )
        _assert_weighted_parity(ours, theirs)
        assert ours.weights == [2.0]
        assert ours.weighted_value == pytest.approx(2.0 * base.weighted_value, rel=1e-12)
    #  A missing weights list defaults to unit weights per selected output.
    for request in (WeightedRiskRequest_(weights=None), WeightedRiskRequest_()):
        ours = _ours_weighted(dates, events, request=request)
        _assert_weighted_parity(ours, _theirs_weighted(dal, dates, events))
        assert ours.weights == [1.0]


def test_weighted_risk_selects_multiple_outputs(dal):
    dates, events = _THREE_DATES, _THREE_EVENTS
    cases = [
        (["output:0", "payoff"], [1.0, 2.0]),
        (["payoff", "output:0"], [1.0, 2.0]),
        (["payoff"], [2.0]),
        (["output:0", "output:1", "payoff"], [1.0, 2.0, 3.0]),
        (None, [2.0]),
    ]
    for outputs, weights in cases:
        request = (
            WeightedRiskRequest_(weights=weights)
            if outputs is None
            else WeightedRiskRequest_(outputs=outputs, weights=weights)
        )
        native_request = (
            dal.WeightedRiskRequest_(weights=weights)
            if outputs is None
            else dal.WeightedRiskRequest_(outputs=outputs, weights=weights)
        )
        ours = _ours_weighted(dates, events, request=request)
        theirs = _theirs_weighted(dal, dates, events, request=native_request)
        _assert_weighted_parity(ours, theirs)
        assert [c.id for c in ours.output_axis] == (["payoff"] if outputs is None else outputs)
    #  weights=None with several selected outputs defaults to one unit weight each.
    both = _ours_weighted(
        dates, events, request=WeightedRiskRequest_(outputs=["output:0", "payoff"])
    )
    _assert_weighted_parity(
        both,
        _theirs_weighted(
            dal, dates, events, request=dal.WeightedRiskRequest_(outputs=["output:0", "payoff"])
        ),
    )
    assert both.weights == [1.0, 1.0]


def test_weighted_risk_jacobian_is_the_weighted_row_sum(dal):
    dates, events = _THREE_DATES, _THREE_EVENTS
    request = WeightedRiskRequest_(
        outputs=["output:0", "payoff"], weights=[3.0, 5.0], inputs=["model:1", "model:0"]
    )
    ours = _ours_weighted(dates, events, request=request)
    theirs = _theirs_weighted(
        dal,
        dates,
        events,
        request=dal.WeightedRiskRequest_(
            outputs=["output:0", "payoff"], weights=[3.0, 5.0], inputs=["model:1", "model:0"]
        ),
    )
    _assert_weighted_parity(ours, theirs)
    assert [c.id for c in ours.input_axis] == ["model:1", "model:0"]
    per_slot = _ours_jacobian(
        dates,
        events,
        n_paths=1024,
        request=JacobianRiskRequest_(outputs=["output:0", "payoff"], inputs=["model:1", "model:0"]),
    )
    rows = np.asarray(per_slot.jacobian)
    np.testing.assert_allclose(
        np.asarray(ours.jacobian)[0], 3.0 * rows[0] + 5.0 * rows[1], rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(ours.component_means, per_slot.values, rtol=1e-9)


_WEIGHT_COUNT = (
    "InvalidWeightedRiskRequest: weight count must match selected outputs; field=weights"
)


@pytest.mark.parametrize(
    ("outputs", "weights"),
    [
        (None, [1.0, 2.0]),
        (None, []),
        (["payoff"], [1.0, 2.0]),
        (["output:0", "payoff"], [1.0]),
    ],
)
def test_weighted_risk_validates_weight_count(dal, outputs, weights):
    dates, events = _THREE_DATES, _THREE_EVENTS
    request = (
        WeightedRiskRequest_(weights=weights)
        if outputs is None
        else WeightedRiskRequest_(outputs=outputs, weights=weights)
    )
    with pytest.raises(RuntimeError, match="weight count must match selected outputs"):
        _theirs_weighted(
            dal,
            dates,
            events,
            n_paths=64,
            request=(
                dal.WeightedRiskRequest_(weights=weights)
                if outputs is None
                else dal.WeightedRiskRequest_(outputs=outputs, weights=weights)
            ),
        )
    with pytest.raises(InvalidWeightedRiskRequest, match=_WEIGHT_COUNT) as error:
        _ours_weighted(dates, events, n_paths=64, request=request)
    assert str(error.value) == _WEIGHT_COUNT
    assert error.value.detail == "weight count must match selected outputs; field=weights"


@pytest.mark.parametrize(
    ("weights", "detail"),
    [
        (2.0, "weights; type=float, value=2.0; expected list or tuple, or None"),
        ("x", "weights; type=str, value='x'; expected list or tuple, or None"),
        (["a"], "weights; type=str, value='a'; expected int or float, excluding bool"),
        ([True], "weights; type=bool, value=True; expected int or float, excluding bool"),
        ([1.0, None], "weights; type=NoneType, value=None; expected int or float, excluding bool"),
    ],
)
def test_weighted_risk_validates_weight_types(dal, weights, detail):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    message = f"InvalidWeightedRiskRequest: WeightedRiskRequest_; {detail}"
    with pytest.raises(TypeError, match="InvalidWeightedRiskRequest"):
        _theirs_weighted(
            dal, dates, events, n_paths=64, request=dal.WeightedRiskRequest_(weights=weights)
        )
    with pytest.raises(InvalidWeightedRiskRequest, match=message) as error:
        _ours_weighted(dates, events, n_paths=64, request=WeightedRiskRequest_(weights=weights))
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("outputs", "weights", "detail"),
    [
        (["nonsense"], [1.0], "unknown scalar output ID; output=nonsense"),
        (["a"], [1.0], "unknown scalar output ID; output=a"),
        (["payoff", "PAYOFF"], [1.0, 2.0], "repeated output ID; output=PAYOFF"),
        ([], [], "output selection must be nonempty; field=outputs"),
    ],
)
def test_weighted_risk_rejects_bad_output_selectors(dal, outputs, weights, detail):
    dates, events = _THREE_DATES, _THREE_EVENTS
    message = f"InvalidWeightedRiskRequest: {detail}"
    with pytest.raises(RuntimeError, match="InvalidWeightedRiskRequest"):
        _theirs_weighted(
            dal,
            dates,
            events,
            n_paths=64,
            request=dal.WeightedRiskRequest_(outputs=outputs, weights=weights),
        )
    with pytest.raises(InvalidWeightedRiskRequest, match=message) as error:
        _ours_weighted(
            dates,
            events,
            n_paths=64,
            request=WeightedRiskRequest_(outputs=outputs, weights=weights),
        )
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("inputs", "error_class", "message"),
    [
        (
            ["nonsense"],
            InvalidRiskRequest,
            "InvalidRiskRequest: unknown input ID; input=nonsense",
        ),
        (
            ["model:0", "model:0"],
            InvalidRiskRequest,
            "InvalidRiskRequest: repeated input ID; input=model:0",
        ),
        (
            0,
            InvalidWeightedRiskRequest,
            "InvalidWeightedRiskRequest: WeightedRiskRequest_; inputs; type=int, value=0; "
            "expected list or tuple, or None",
        ),
    ],
)
def test_weighted_risk_rejects_bad_input_selectors(dal, inputs, error_class, message):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    with pytest.raises((RuntimeError, TypeError), match=message.split(":")[0]):
        _theirs_weighted(
            dal, dates, events, n_paths=64, request=dal.WeightedRiskRequest_(inputs=inputs)
        )
    with pytest.raises(error_class, match=message) as error:
        _ours_weighted(dates, events, n_paths=64, request=WeightedRiskRequest_(inputs=inputs))
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("inputs", "outputs", "weights", "message"),
    [
        (
            ["nonsense"],
            ["nonsense"],
            [1.0, 2.0],
            "InvalidWeightedRiskRequest: unknown scalar output ID; output=nonsense",
        ),
        (
            ["nonsense"],
            None,
            [1.0, 2.0],
            "InvalidWeightedRiskRequest: weight count must match selected outputs; field=weights",
        ),
        (
            ["nonsense"],
            ["nonsense"],
            [2.0],
            "InvalidWeightedRiskRequest: unknown scalar output ID; output=nonsense",
        ),
    ],
    ids=["outputs-before-weights", "weights-before-inputs", "outputs-before-inputs"],
)
def test_weighted_risk_validation_order(dal, inputs, outputs, weights, message):
    dates, events = _THREE_DATES, _THREE_EVENTS
    with pytest.raises(RuntimeError, match="InvalidWeightedRiskRequest"):
        _theirs_weighted(
            dal,
            dates,
            events,
            n_paths=64,
            request=(
                dal.WeightedRiskRequest_(inputs=inputs, weights=weights)
                if outputs is None
                else dal.WeightedRiskRequest_(inputs=inputs, outputs=outputs, weights=weights)
            ),
        )
    with pytest.raises(InvalidWeightedRiskRequest, match=message) as error:
        _ours_weighted(
            dates,
            events,
            n_paths=64,
            request=(
                WeightedRiskRequest_(inputs=inputs, weights=weights)
                if outputs is None
                else WeightedRiskRequest_(inputs=inputs, outputs=outputs, weights=weights)
            ),
        )
    assert str(error.value) == message


_EXERCISE_WEIGHTED = "UnsupportedWeightedRisk: EXERCISE requires a separate weighted estimator"


def test_weighted_risk_rejects_exercise(dal):
    dates = [(2024, 1, 2), (2024, 7, 2)]
    events = ["x = 1\nexercise x", "y PAYS SPOT()"]
    cases = [
        (None, None),
        (
            WeightedRiskRequest_(outputs=["nonsense"], weights=[1.0, 2.0]),
            dal.WeightedRiskRequest_(outputs=["nonsense"], weights=[1.0, 2.0]),
        ),
        (
            WeightedRiskRequest_(weights=[1.0, 2.0]),
            dal.WeightedRiskRequest_(weights=[1.0, 2.0]),
        ),
    ]
    for request, native_request in cases:
        with pytest.raises(RuntimeError, match="UnsupportedWeightedRisk"):
            _theirs_weighted(dal, dates, events, n_paths=64, request=native_request)
        with pytest.raises(UnsupportedWeightedRisk, match=_EXERCISE_WEIGHTED) as error:
            _ours_weighted(dates, events, n_paths=64, request=request)
        assert str(error.value) == _EXERCISE_WEIGHTED
        assert error.value.detail == "EXERCISE requires a separate weighted estimator"


_EXPIRED_WEIGHTED = "UnsupportedWeightedRisk: fully expired products are not supported"


def test_weighted_risk_rejects_expired_products(dal):
    dates, events = [(2022, 1, 15)], ["p pays 1.0"]
    with pytest.raises(RuntimeError, match="UnsupportedWeightedRisk"):
        _theirs_weighted(dal, dates, events, n_paths=64)
    with pytest.raises(UnsupportedWeightedRisk, match=_EXPIRED_WEIGHTED) as error:
        _ours_weighted(dates, events, n_paths=64)
    assert str(error.value) == _EXPIRED_WEIGHTED
