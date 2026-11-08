"""BlackScholesMonteCarlo plan API parity with dal-python 2026.11.1."""

import math
import re

import numpy as np
import pytest

from dal_jax.api import (
    BlackScholesMonteCarlo_Get_Curvature,
    BlackScholesMonteCarlo_Get_Risk,
    BlackScholesMonteCarloPlan_New,
    BumpOverAADRequest_,
    MonteCarloSettings_,
    Product_New,
    SegmentedMonteCarloSettings_,
)
from dal_jax.dates import Date
from dal_jax.errors import (
    InvalidModelParameter,
    InvalidPathCount,
    InvalidSetting,
    ScriptError,
    script_error,
)
from dal_jax.script.fixings import ValuationSettings
from dal_jax.script.product import ScriptProductSettings

pytestmark = pytest.mark.oracle

TODAY = Date.ymd(2022, 9, 15)
VALUATION = ValuationSettings(evaluation_date=TODAY)
SETTINGS_ECHO_FIELDS = (
    "first_path",
    "rsg",
    "scramble_key",
    "segment_steps",
    "use_bb",
    "normal_precision",
    "recording_capacity_budget_bytes",
    "checkpoint_capacity_budget_bytes",
)
#  The execution snapshot echoes the execution settings, minus the two budgets.
EXECUTION_ECHO_FIELDS = SETTINGS_ECHO_FIELDS[:6]


def _error_class(message):
    """The exception class dal_jax raises for a native error-code prefix."""
    return type(script_error(message))


def _raises_native(message):
    #  pybind raises TypeError for binding-level type errors, RuntimeError from the core.
    return pytest.raises((RuntimeError, TypeError), match=re.escape(message))


def _raises_ours(message):
    return pytest.raises(_error_class(message), match="^" + re.escape(message) + "$")


def _cells(dates):
    return [Date.ymd(*d) if isinstance(d, tuple) else d for d in dates]


def _ours(dates, events, default_index="", **kwargs):
    return BlackScholesMonteCarloPlan_New(
        Product_New(
            _cells(dates),
            list(events),
            settings=ScriptProductSettings(default_index=default_index),
        ),
        valuation=VALUATION,
        **kwargs,
    )


def _theirs(dal, dates, events, default_index="", **kwargs):
    return dal.BlackScholesMonteCarloPlan_New(
        dal.Product_New(
            [dal.Date_(*d) if isinstance(d, tuple) else d for d in dates],
            list(events),
            settings=dal.ScriptProductSettings_(default_index=default_index),
        ),
        valuation=dal.ScriptValuationSettings_(evaluation_date=dal.Date_(2022, 9, 15)),
        **kwargs,
    )


def _settings(dal, **kwargs):
    return dal.SegmentedMonteCarloSettings_(**kwargs)


def _observation_records(observations, native=False):
    key = (lambda o: str(o.fixing_time)[:10]) if native else (lambda o: str(o.fixing_time))
    return [(key(o), o.historical, o.index, o.value) for o in observations]


def _assert_settings_echo(ours, theirs):
    for field in SETTINGS_ECHO_FIELDS:
        assert getattr(ours, field) == getattr(theirs, field), field


PLAN_CASES = [
    pytest.param([(2024, 1, 2)], ["p pays MAX(spot()-100.0,0)"], "", id="call"),
    pytest.param(
        ["K", (2024, 1, 2), (2024, 7, 2)],
        ["100", "s = SPOT()", "p PAYS MAX(s - K, 0)"],
        "",
        id="constant-table",
    ),
    pytest.param(
        ["K", (2024, 1, 2), (2024, 7, 2)],
        ["100", "s = SPOT()", "p PAYS MAX(FIX(EQ[A], 2024-03-01) + s - K, 0)"],
        "EQ[A]",
        id="fix-observation",
    ),
]


@pytest.mark.parametrize(("dates", "events", "default_index"), PLAN_CASES)
def test_plan_attributes_match_native(dal, dates, events, default_index):
    ours = _ours(dates, events, default_index)
    theirs = _theirs(dal, dates, events, default_index)
    assert list(ours.parameter_labels) == list(theirs.parameter_labels)
    assert ours.parameter_labels[:4] == ("spot", "vol", "rate", "div")
    assert ours.smoothing == theirs.smoothing == 0.01
    assert list(ours.contract_events) == list(theirs.contract_events)
    assert len(ours.contract_dates) == len(theirs.contract_dates) == len(dates)
    assert list(ours.script_constants) == list(theirs.script_constants)
    assert _observation_records(ours.observations) == _observation_records(
        theirs.observations, native=True
    )
    assert str(ours.valuation.evaluation_date) == str(theirs.valuation.evaluation_date)


def test_plan_attribute_contents():
    from dal_jax.mc.plan import RiskObservationSnapshot

    call = _ours([(2024, 1, 2)], ["p pays MAX(spot()-100.0,0)"])
    assert call.parameter_labels == ("spot", "vol", "rate", "div")
    assert call.script_constants == ()
    assert call.observations == ()
    const = _ours(["K", (2024, 1, 2), (2024, 7, 2)], ["100", "s = SPOT()", "p PAYS MAX(s - K, 0)"])
    assert const.parameter_labels == ("spot", "vol", "rate", "div", "K")
    assert const.script_constants == (100.0,)
    fix = _ours(
        ["K", (2024, 1, 2), (2024, 7, 2)],
        ["100", "s = SPOT()", "p PAYS MAX(FIX(EQ[A], 2024-03-01) + s - K, 0)"],
        "EQ[A]",
    )
    assert fix.observations == (
        RiskObservationSnapshot(
            fixing_time=Date.ymd(2024, 1, 2), historical=False, index="EQ[A]", value=None
        ),
        RiskObservationSnapshot(
            fixing_time=Date.ymd(2024, 3, 1), historical=False, index="EQ[A]", value=None
        ),
    )
    assert fix.contract_settings.default_index == "EQ[A]"
    assert list(fix.contract_dates) == ["K", Date.ymd(2024, 1, 2), Date.ymd(2024, 7, 2)]


def test_plan_smoothing_is_carried(dal):
    dates = [(2024, 1, 2)]
    events = ["if spot() > 100 then\np PAYS 1\nelse\np PAYS 0\nend"]
    ours = _ours(dates, events, smoothing=0.05)
    assert ours.smoothing == _theirs(dal, dates, events, smoothing=0.05).smoothing == 0.05
    point = [100.0, 0.2, 0.03, 0.01]
    wide = BlackScholesMonteCarlo_Get_Risk(ours, point, 4096).value
    tight = BlackScholesMonteCarlo_Get_Risk(_ours(dates, events), point, 4096).value
    assert wide != tight
    native = dal.BlackScholesMonteCarlo_Get_Risk(
        _theirs(dal, dates, events, smoothing=0.05), point, 4096
    ).value
    assert wide == pytest.approx(native, rel=1e-11)


PLAN_CTOR_ERRORS = [
    pytest.param(
        {"smoothing": True},
        "InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarloPlan_New; smoothing; "
        "type=bool, value=True; expected int or float, excluding bool",
        id="smoothing-bool",
    ),
    pytest.param(
        {"smoothing": "x"},
        "InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarloPlan_New; smoothing; "
        "type=str, value='x'; expected int or float, excluding bool",
        id="smoothing-string",
    ),
    pytest.param(
        {"smoothing": 0.0},
        "InvalidSetting: InvalidSmoothing; simulation.smooth_=0.000000; "
        "expected a finite positive width",
        id="smoothing-zero",
    ),
    pytest.param(
        {"smoothing": -1.0},
        "InvalidSetting: InvalidSmoothing; simulation.smooth_=-1.000000; "
        "expected a finite positive width",
        id="smoothing-negative",
    ),
    pytest.param(
        {"valuation": "x"},
        "InvalidSetting: BlackScholesMonteCarloPlan_New; valuation; "
        "type=str, value='x'; expected ScriptValuationSettings_ or None",
        id="valuation-type",
    ),
]


@pytest.mark.parametrize(("kwargs", "message"), PLAN_CTOR_ERRORS)
def test_plan_rejects_bad_ctor_arguments(dal, kwargs, message):
    if "valuation" in kwargs:  # the helpers already pass a valuation; call the ctors directly
        with _raises_native(message):
            dal.BlackScholesMonteCarloPlan_New(_theirs_product(dal), **kwargs)
        with _raises_ours(message):
            BlackScholesMonteCarloPlan_New(
                Product_New([Date.ymd(2024, 1, 2)], ["p pays spot()"]), **kwargs
            )
        return
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    with _raises_native(message):
        _theirs(dal, dates, events, **kwargs)
    with _raises_ours(message):
        _ours(dates, events, **kwargs)


def _theirs_product(dal):
    return dal.Product_New(
        [dal.Date_(2024, 1, 2)], ["p pays spot()"], settings=dal.ScriptProductSettings_()
    )


def test_plan_rejects_non_product(dal):
    message = (
        "InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarloPlan_New; product; "
        "type=str, value='not a product'; expected ScriptProductData_"
    )
    with _raises_native(message):
        dal.BlackScholesMonteCarloPlan_New("not a product")
    with _raises_ours(message):
        BlackScholesMonteCarloPlan_New("not a product", valuation=VALUATION)


def test_plan_rejects_exercise(dal):
    message = "BlackScholesSegmentedPath: EXERCISE is unsupported"
    dates, events = [(2024, 1, 2)], ["exercise spot()\np pays spot()"]
    with _raises_native(message):
        _theirs(dal, dates, events)
    with _raises_ours(message):
        _ours(dates, events)


@pytest.mark.parametrize(
    "point",
    [
        pytest.param([100.0, 0.2, 0.03, 0.01], id="at-the-money"),
        pytest.param([110.0, 0.25, 0.02, 0.0], id="in-the-money"),
    ],
)
def test_get_risk_matches_native(dal, point):
    dates, events = [(2024, 1, 2)], ["p pays MAX(spot()-100.0,0)"]
    ours = BlackScholesMonteCarlo_Get_Risk(_ours(dates, events), point, 1024)
    theirs = dal.BlackScholesMonteCarlo_Get_Risk(_theirs(dal, dates, events), list(point), 1024)
    assert ours.value == pytest.approx(theirs.value, rel=1e-11)
    np.testing.assert_allclose(ours.gradient, theirs.gradient, rtol=1e-9, atol=1e-12)
    assert list(ours.parameter_labels) == list(theirs.parameter_labels)
    assert ours.point == list(point) == list(theirs.point)
    _assert_settings_echo(ours.settings, theirs.settings)
    assert ours.execution.path_count == theirs.execution.path_count == 1024
    for field in EXECUTION_ECHO_FIELDS:
        assert getattr(ours.execution, field) == getattr(theirs.execution, field), field


@pytest.mark.parametrize("strike", [100.0, 95.0])
def test_get_risk_covers_script_constants(dal, strike):
    dates = ["K", (2024, 1, 2), (2024, 7, 2)]
    events = ["100", "s = SPOT()", "p PAYS MAX(s - K, 0)"]
    point = [100.0, 0.2, 0.03, 0.01, strike]
    ours = BlackScholesMonteCarlo_Get_Risk(_ours(dates, events), point, 1024)
    theirs = dal.BlackScholesMonteCarlo_Get_Risk(_theirs(dal, dates, events), list(point), 1024)
    assert ours.value == pytest.approx(theirs.value, rel=1e-11)
    np.testing.assert_allclose(ours.gradient, theirs.gradient, rtol=1e-9, atol=1e-12)
    assert list(ours.parameter_labels) == ["spot", "vol", "rate", "div", "K"]


def test_get_risk_settings_none_equals_default_settings(dal):
    dates, events = [(2024, 1, 2)], ["p pays MAX(spot()-100.0,0)"]
    point = [100.0, 0.2, 0.03, 0.01]
    ours_default = BlackScholesMonteCarlo_Get_Risk(_ours(dates, events), point, 512)
    ours_none = BlackScholesMonteCarlo_Get_Risk(
        _ours(dates, events), point, 512, settings=SegmentedMonteCarloSettings_()
    )
    theirs = dal.BlackScholesMonteCarlo_Get_Risk(
        _theirs(dal, dates, events), point, 512, settings=_settings(dal)
    )
    assert ours_none.value == ours_default.value == pytest.approx(theirs.value, rel=1e-11)
    _assert_settings_echo(ours_none.settings, theirs.settings)


SETTINGS_EFFECTS = [
    pytest.param({"first_path": 100}, 1e-11, id="first-path"),
    pytest.param({"scramble_key": 7}, 1e-11, id="scramble-key"),
    pytest.param({"use_bb": True}, 1e-11, id="brownian-bridge"),
    pytest.param({"segment_steps": 16}, 1e-11, id="segment-steps"),
    pytest.param({"normal_precision": "Precise"}, 1e-9, id="precise-inversion"),
    pytest.param({"rsg": "SOBOL"}, 1e-11, id="rsg-case-insensitive"),
]


@pytest.mark.parametrize(("kwargs", "rtol"), SETTINGS_EFFECTS)
def test_segmented_settings_match_native(dal, kwargs, rtol):
    dates, events = [(2024, 1, 2), (2025, 1, 2)], ["a = spot()", "p PAYS MAX(a - 100.0, 0)"]
    point = [100.0, 0.2, 0.03, 0.01]
    ours = BlackScholesMonteCarlo_Get_Risk(
        _ours(dates, events), point, 1024, settings=SegmentedMonteCarloSettings_(**kwargs)
    )
    theirs = dal.BlackScholesMonteCarlo_Get_Risk(
        _theirs(dal, dates, events), point, 1024, settings=_settings(dal, **kwargs)
    )
    assert ours.value == pytest.approx(theirs.value, rel=rtol)
    np.testing.assert_allclose(ours.gradient, theirs.gradient, rtol=rtol, atol=1e-12)
    _assert_settings_echo(ours.settings, theirs.settings)
    for field in EXECUTION_ECHO_FIELDS:
        assert getattr(ours.execution, field) == getattr(theirs.execution, field), field


def test_segmented_settings_move_the_stream(dal):
    """first_path and scramble_key select a different Sobol window on both sides."""
    dates, events = [(2024, 1, 2)], ["p pays MAX(spot()-100.0,0)"]
    point = [100.0, 0.2, 0.03, 0.01]
    plan, native_plan = _ours(dates, events), _theirs(dal, dates, events)
    base = BlackScholesMonteCarlo_Get_Risk(plan, point, 1024).value
    native_base = dal.BlackScholesMonteCarlo_Get_Risk(native_plan, point, 1024).value
    for kwargs in ({"first_path": 100}, {"scramble_key": 7}):
        moved = BlackScholesMonteCarlo_Get_Risk(
            plan, point, 1024, settings=SegmentedMonteCarloSettings_(**kwargs)
        ).value
        native_moved = dal.BlackScholesMonteCarlo_Get_Risk(
            native_plan, point, 1024, settings=_settings(dal, **kwargs)
        ).value
        assert moved != base
        assert native_moved != native_base
        assert moved == pytest.approx(native_moved, rel=1e-11)


def test_segment_steps_leave_the_stream_unchanged(dal):
    """segment_steps re-blocks native's reduction only; the path stream is unchanged."""
    dates, events = [(2024, 1, 2)], ["p pays MAX(spot()-100.0,0)"]
    point = [100.0, 0.2, 0.03, 0.01]
    plan, native_plan = _ours(dates, events), _theirs(dal, dates, events)
    default = BlackScholesMonteCarlo_Get_Risk(plan, point, 512).value
    native_default = dal.BlackScholesMonteCarlo_Get_Risk(native_plan, point, 512).value
    for steps in (16, 256):
        ours = BlackScholesMonteCarlo_Get_Risk(
            plan, point, 512, settings=SegmentedMonteCarloSettings_(segment_steps=steps)
        ).value
        theirs = dal.BlackScholesMonteCarlo_Get_Risk(
            native_plan, point, 512, settings=_settings(dal, segment_steps=steps)
        ).value
        assert ours == default
        assert theirs == pytest.approx(native_default, rel=1e-11)
        assert ours == pytest.approx(theirs, rel=1e-11)


SETTINGS_VALIDATION = [
    pytest.param(
        {"first_path": -1},
        "InvalidSegmentedMonteCarloRequest: SegmentedMonteCarloSettings_; first_path; "
        "type=int, value=-1; expected nonnegative size_t integer or None, excluding bool",
        id="first-path-negative",
    ),
    pytest.param(
        {"first_path": 1.5},
        "InvalidSegmentedMonteCarloRequest: SegmentedMonteCarloSettings_; first_path; "
        "type=float, value=1.5; expected nonnegative size_t integer or None, excluding bool",
        id="first-path-float",
    ),
    pytest.param(
        {"first_path": True},
        "InvalidSegmentedMonteCarloRequest: SegmentedMonteCarloSettings_; first_path; "
        "type=bool, value=True; expected nonnegative size_t integer or None, excluding bool",
        id="first-path-bool",
    ),
    pytest.param(
        {"segment_steps": 2.5},
        "InvalidSegmentedMonteCarloRequest: SegmentedMonteCarloSettings_; segment_steps; "
        "type=float, value=2.5; expected nonnegative size_t integer or None, excluding bool",
        id="segment-steps-float",
    ),
    pytest.param(
        {"segment_steps": 0},
        "InvalidSegmentedMonteCarloRequest: segment_steps must be positive",
        id="segment-steps-zero",
    ),
    pytest.param(
        {"scramble_key": -1},
        "InvalidSegmentedMonteCarloRequest: SegmentedMonteCarloSettings_; scramble_key; "
        "type=int, value=-1; expected uint64 integer or None, excluding bool",
        id="scramble-key-negative",
    ),
    pytest.param(
        {"scramble_key": 1.5},
        "InvalidSegmentedMonteCarloRequest: SegmentedMonteCarloSettings_; scramble_key; "
        "type=float, value=1.5; expected uint64 integer or None, excluding bool",
        id="scramble-key-float",
    ),
    pytest.param(
        {"use_bb": 1},
        "InvalidSegmentedMonteCarloRequest: SegmentedMonteCarloSettings_; use_bb; "
        "type=int, value=1; expected bool",
        id="use-bb-int",
    ),
]


@pytest.mark.parametrize(("kwargs", "message"), SETTINGS_VALIDATION)
def test_segmented_settings_validation_matches_native(dal, kwargs, message):
    with _raises_native(message):
        _settings(dal, **kwargs)
    with _raises_ours(message):
        SegmentedMonteCarloSettings_(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"rsg": "bogus"},
            "InvalidSetting: simulation.rsg_=bogus; expected sobol, mrg32 or irn; "
            "rng method is not known",
        ),
        (
            {"normal_precision": "bogus"},
            "InvalidSetting: simulation.normalPrecision_=bogus; expected Default, Fast or Precise",
        ),
        (
            {"scramble_key": 7, "rsg": "mrg32"},
            "InvalidSetting: a Sobol digital shift requires simulation.rsg_=sobol",
        ),
    ],
)
def test_get_risk_rejects_unknown_execution_settings(dal, kwargs, message):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    point = [100.0, 0.2, 0.03, 0.01]
    with _raises_native(message):
        dal.BlackScholesMonteCarlo_Get_Risk(
            _theirs(dal, dates, events), point, 64, settings=_settings(dal, **kwargs)
        )
    with pytest.raises(InvalidSetting, match="^" + re.escape(message) + "$"):
        BlackScholesMonteCarlo_Get_Risk(
            _ours(dates, events), point, 64, settings=SegmentedMonteCarloSettings_(**kwargs)
        )


POINT_ERRORS = [
    pytest.param(
        [100.0, 0.2, 0.03],
        "BlackScholesSegmentedPath: parameter size must equal ParameterLabels",
        id="too-short",
    ),
    pytest.param(
        [100.0, 0.2, 0.03, 0.01, 1.0],
        "BlackScholesSegmentedPath: parameter size must equal ParameterLabels",
        id="too-long",
    ),
    pytest.param(
        {"spot": 100.0},
        "InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarlo_Get_Risk; point; "
        "type=dict; expected list or tuple, or None",
        id="dict",
    ),
    pytest.param(
        [100.0, "x", 0.03, 0.01],
        "InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarlo_Get_Risk; point; "
        "type=str, value='x'; expected int or float, excluding bool",
        id="string-entry",
    ),
    pytest.param(
        [100.0, None, 0.03, 0.01],
        "InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarlo_Get_Risk; point; "
        "type=NoneType, value=None; expected int or float, excluding bool",
        id="none-entry",
    ),
    pytest.param(
        [True, 0.2, 0.03, 0.01],
        "InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarlo_Get_Risk; point; "
        "type=bool, value=True; expected int or float, excluding bool",
        id="bool-entry",
    ),
]


@pytest.mark.parametrize(("point", "message"), POINT_ERRORS)
def test_get_risk_rejects_bad_points(dal, point, message):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    with _raises_native(message):
        dal.BlackScholesMonteCarlo_Get_Risk(_theirs(dal, dates, events), point, 64)
    with _raises_ours(message):
        BlackScholesMonteCarlo_Get_Risk(_ours(dates, events), point, 64)


@pytest.mark.parametrize(
    ("point", "detail"),
    [
        ([0.0, 0.2, 0.03, 0.01], "spot must be finite and positive"),
        ([100.0, -0.2, 0.03, 0.01], "vol must be finite and nonnegative"),
    ],
)
def test_get_risk_rejects_invalid_model_points(dal, point, detail):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    with _raises_native(f"InvalidModelParameter: {detail}"):
        dal.BlackScholesMonteCarlo_Get_Risk(_theirs(dal, dates, events), point, 64)
    with pytest.raises(InvalidModelParameter, match="^InvalidModelParameter: " + detail + "$"):
        BlackScholesMonteCarlo_Get_Risk(_ours(dates, events), point, 64)


NUM_PATH_ERRORS = [
    pytest.param(
        0,
        "InvalidPathCount: BlackScholesMonteCarlo_Get_Risk; num_path / numPath; "
        "type=int, value=0; expected a positive integer in 1..2147483647; "
        "number of Monte Carlo paths must be positive",
        id="zero",
    ),
    pytest.param(
        -1,
        "InvalidPathCount: BlackScholesMonteCarlo_Get_Risk; num_path / numPath; "
        "type=int, value=-1; expected a positive integer in 1..2147483647; "
        "number of Monte Carlo paths must be positive",
        id="negative",
    ),
    pytest.param(
        2**31,
        "InvalidPathCount: BlackScholesMonteCarlo_Get_Risk; num_path / numPath; "
        "type=int, value=2147483648; expected a positive integer in 1..2147483647; "
        "number of Monte Carlo paths must be positive",
        id="too-large",
    ),
    pytest.param(
        64.0,
        "InvalidPathCount: BlackScholesMonteCarlo_Get_Risk; num_path / numPath; "
        "type=float, value=64.0; expected a positive integer in 1..2147483647",
        id="float",
    ),
    pytest.param(
        True,
        "InvalidPathCount: BlackScholesMonteCarlo_Get_Risk; num_path / numPath; "
        "type=bool, value=True; expected a positive integer in 1..2147483647",
        id="bool",
    ),
    pytest.param(
        None,
        "InvalidPathCount: BlackScholesMonteCarlo_Get_Risk; num_path / numPath; "
        "type=NoneType, value=None; expected a positive integer in 1..2147483647",
        id="none",
    ),
]


@pytest.mark.parametrize(("num_path", "message"), NUM_PATH_ERRORS)
def test_get_risk_rejects_bad_path_counts(dal, num_path, message):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    point = [100.0, 0.2, 0.03, 0.01]
    with _raises_native(message):
        dal.BlackScholesMonteCarlo_Get_Risk(_theirs(dal, dates, events), point, num_path)
    with pytest.raises(InvalidPathCount, match="^" + re.escape(message) + "$"):
        BlackScholesMonteCarlo_Get_Risk(_ours(dates, events), point, num_path)


def test_get_risk_rejects_wrong_settings_type(dal):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    point = [100.0, 0.2, 0.03, 0.01]
    message = (
        "InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarlo_Get_Risk; settings; "
        "type=MonteCarloSettings"
    )
    with _raises_native(message):
        dal.BlackScholesMonteCarlo_Get_Risk(
            _theirs(dal, dates, events), point, 64, settings=dal.MonteCarloSettings_()
        )
    with pytest.raises(ScriptError, match="^" + re.escape(message)):
        BlackScholesMonteCarlo_Get_Risk(
            _ours(dates, events), point, 64, settings=MonteCarloSettings_()
        )


def _bumps(rows, steps):
    return BumpOverAADRequest_(directions=[list(row) for row in rows], steps=list(steps))


def _native_bumps(dal, rows, steps):
    return dal.BumpOverAADRequest_(directions=dal.DoubleMatrix_(rows), steps=steps)


def test_curvature_matches_native(dal):
    dates, events = [(2024, 1, 2)], ["p pays MAX(spot()-100.0,0)"]
    point = [100.0, 0.2, 0.03, 0.01]
    directions = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    steps = [1.0, 0.01]
    ours = BlackScholesMonteCarlo_Get_Curvature(
        _ours(dates, events), point, 1024, _bumps(directions, steps)
    )
    theirs = dal.BlackScholesMonteCarlo_Get_Curvature(
        _theirs(dal, dates, events), point, 1024, _native_bumps(dal, directions, steps)
    )
    np.testing.assert_allclose(
        ours.hessian_products, theirs.hessian_products.to_rows(), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(ours.directions, theirs.directions.to_rows(), rtol=1e-12)
    assert list(ours.steps) == steps
    assert ours.base.value == pytest.approx(theirs.base.value, rel=1e-11)
    np.testing.assert_allclose(ours.base.gradient, theirs.base.gradient, rtol=1e-9, atol=1e-12)
    assert list(ours.base.parameter_labels) == list(theirs.base.parameter_labels)
    assert ours.execution.method == theirs.execution.method
    assert ours.execution.gradient_evaluations == theirs.execution.gradient_evaluations == 5
    assert ours.point == list(point) == list(theirs.point)
    _assert_settings_echo(ours.settings, theirs.settings)


def test_curvature_covers_script_constants(dal):
    dates = ["K", (2024, 1, 2), (2024, 7, 2)]
    events = ["100", "s = SPOT()", "p PAYS MAX(s - K, 0)"]
    point = [100.0, 0.2, 0.03, 0.01, 100.0]
    directions = [[0.0, 0.0, 0.0, 0.0, 1.0]]
    steps = [1.0]
    ours = BlackScholesMonteCarlo_Get_Curvature(
        _ours(dates, events), point, 1024, _bumps(directions, steps)
    )
    theirs = dal.BlackScholesMonteCarlo_Get_Curvature(
        _theirs(dal, dates, events), point, 1024, _native_bumps(dal, directions, steps)
    )
    np.testing.assert_allclose(
        ours.hessian_products, theirs.hessian_products.to_rows(), rtol=1e-9, atol=1e-12
    )
    assert ours.hessian_products.shape == (1, 5)


def test_curvature_is_central_difference_of_the_gradient():
    dates, events = [(2024, 1, 2)], ["p pays MAX(spot()-100.0,0)"]
    plan = _ours(dates, events)
    point = [100.0, 0.2, 0.03, 0.01]
    step, direction = 1.0, [1.0, 0.0, 0.0, 0.0]
    up = [point[i] + step * direction[i] for i in range(4)]
    down = [point[i] - step * direction[i] for i in range(4)]
    curvature = BlackScholesMonteCarlo_Get_Curvature(plan, point, 1024, _bumps([direction], [step]))
    g_up = np.asarray(BlackScholesMonteCarlo_Get_Risk(plan, up, 1024).gradient)
    g_down = np.asarray(BlackScholesMonteCarlo_Get_Risk(plan, down, 1024).gradient)
    np.testing.assert_allclose(
        curvature.hessian_products[0], (g_up - g_down) / (2.0 * step), rtol=1e-12, atol=1e-15
    )
    risk = BlackScholesMonteCarlo_Get_Risk(plan, point, 1024)
    assert curvature.base.value == risk.value
    assert list(curvature.base.gradient) == list(risk.gradient)


BUMP_ERRORS = [
    pytest.param(
        [[1.0, 0.0, 0.0, 0.0]],
        [-0.01],
        "InvalidCurvatureRequest: BumpOverAADRequest_; steps must be finite and positive",
        id="steps-negative",
    ),
    pytest.param(
        [[1.0, 0.0, 0.0, 0.0]],
        [0.0],
        "InvalidCurvatureRequest: BumpOverAADRequest_; steps must be finite and positive",
        id="steps-zero",
    ),
    pytest.param(
        [[1.0, 0.0, 0.0, 0.0]],
        ["x"],
        "InvalidCurvatureRequest: BumpOverAADRequest_; steps; "
        "type=str, value='x'; expected int or float, excluding bool",
        id="steps-string",
    ),
    pytest.param(
        [[1.0, 0.0, 0.0, 0.0]],
        [0.01, 0.02],
        "InvalidCurvatureRequest: BumpOverAADRequest_; steps requires one value per direction row",
        id="steps-count",
    ),
    pytest.param(
        [[1.0, 0.0, 0.0]],
        [0.01],
        "BumpOverAAD: directions must have nonnegative rows and one column per input",
        id="direction-columns",
    ),
    pytest.param(
        [],
        [],
        "BumpOverAAD: directions must have nonnegative rows and one column per input",
        id="direction-empty",
    ),
]


@pytest.mark.parametrize(("rows", "steps", "message"), BUMP_ERRORS)
def test_curvature_rejects_bad_bumps(dal, rows, steps, message):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    point = [100.0, 0.2, 0.03, 0.01]
    with _raises_native(message):
        dal.BlackScholesMonteCarlo_Get_Curvature(
            _theirs(dal, dates, events), point, 64, _native_bumps(dal, rows, steps)
        )
    with _raises_ours(message):
        BlackScholesMonteCarlo_Get_Curvature(_ours(dates, events), point, 64, _bumps(rows, steps))


def test_curvature_rejects_a_non_request_bumps(dal):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    point = [100.0, 0.2, 0.03, 0.01]
    message = (
        "InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarlo_Get_Curvature; bumps; "
        "type=tuple; expected BumpOverAADRequest_"
    )
    with _raises_native(message):
        dal.BlackScholesMonteCarlo_Get_Curvature(_theirs(dal, dates, events), point, 64, (0.01,))
    with _raises_ours(message):
        BlackScholesMonteCarlo_Get_Curvature(_ours(dates, events), point, 64, (0.01,))


def test_curvature_validates_bumped_points(dal):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    point = [100.0, 0.2, 0.03, 0.01]
    message = "MonteCarloCurvature: direction=0; minus"
    with _raises_native(message):
        dal.BlackScholesMonteCarlo_Get_Curvature(
            _theirs(dal, dates, events),
            point,
            64,
            _native_bumps(dal, [[0.0, 1.0, 0.0, 0.0]], [0.5]),
        )
    with pytest.raises(ScriptError, match="^" + re.escape(message)):
        BlackScholesMonteCarlo_Get_Curvature(
            _ours(dates, events), point, 64, _bumps([[0.0, 1.0, 0.0, 0.0]], [0.5])
        )


def test_curvature_point_length_follows_bumps_validation(dal):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    message = "BumpOverAAD: directions must have nonnegative rows and one column per input"
    with _raises_native(message):
        dal.BlackScholesMonteCarlo_Get_Curvature(
            _theirs(dal, dates, events),
            [1.0, 2.0, 3.0],
            64,
            _native_bumps(dal, [[1.0, 0.0, 0.0, 0.0]], [0.01]),
        )
    with _raises_ours(message):
        BlackScholesMonteCarlo_Get_Curvature(
            _ours(dates, events), [1.0, 2.0, 3.0], 64, _bumps([[1.0, 0.0, 0.0, 0.0]], [0.01])
        )


def test_curvature_rejects_bad_path_count_and_settings(dal):
    dates, events = [(2024, 1, 2)], ["p pays spot()"]
    point = [100.0, 0.2, 0.03, 0.01]
    count_message = "InvalidPathCount: BlackScholesMonteCarlo_Get_Curvature"
    with _raises_native(count_message):
        dal.BlackScholesMonteCarlo_Get_Curvature(
            _theirs(dal, dates, events),
            point,
            0,
            _native_bumps(dal, [[1.0, 0.0, 0.0, 0.0]], [0.01]),
        )
    with pytest.raises(InvalidPathCount, match="^" + re.escape(count_message)):
        BlackScholesMonteCarlo_Get_Curvature(
            _ours(dates, events), point, 0, _bumps([[1.0, 0.0, 0.0, 0.0]], [0.01])
        )
    settings_message = (
        "InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarlo_Get_Curvature; settings; "
        "type=MonteCarloSettings"
    )
    with _raises_native(settings_message):
        dal.BlackScholesMonteCarlo_Get_Curvature(
            _theirs(dal, dates, events),
            point,
            64,
            _native_bumps(dal, [[1.0, 0.0, 0.0, 0.0]], [0.01]),
            settings=dal.MonteCarloSettings_(),
        )
    with pytest.raises(ScriptError, match="^" + re.escape(settings_message)):
        BlackScholesMonteCarlo_Get_Curvature(
            _ours(dates, events),
            point,
            64,
            _bumps([[1.0, 0.0, 0.0, 0.0]], [0.01]),
            settings=MonteCarloSettings_(),
        )


def test_pseudo_random_settings_are_accepted(dal):
    """mrg32/irn streams are not bit-compatible with native; both sides accept the setting."""
    dates, events = [(2024, 1, 2)], ["p pays MAX(spot()-100.0,0)"]
    point = [100.0, 0.2, 0.03, 0.01]
    n_paths = 2**14
    ours_sobol = BlackScholesMonteCarlo_Get_Risk(_ours(dates, events), point, n_paths).value
    payoff_std = 25.0
    tolerance = 6.0 * payoff_std * math.sqrt(2.0) / math.sqrt(n_paths)
    for rsg in ("mrg32", "irn"):
        ours = BlackScholesMonteCarlo_Get_Risk(
            _ours(dates, events), point, n_paths, settings=SegmentedMonteCarloSettings_(rsg=rsg)
        )
        theirs = dal.BlackScholesMonteCarlo_Get_Risk(
            _theirs(dal, dates, events), point, n_paths, settings=_settings(dal, rsg=rsg)
        )
        assert ours.settings.rsg == theirs.settings.rsg == rsg
        assert ours.value != ours_sobol
        assert ours.value == pytest.approx(theirs.value, abs=tolerance)
