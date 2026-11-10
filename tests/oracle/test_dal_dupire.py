"""Dupire risk inputs, calibration snapshots and model parity with dal-python 2026.11.1."""

import re

import numpy as np
import pytest
from oracle.test_dal_risk_api import _assert_risk_parity

from dal_jax.api import (
    IVS_,
    BSModelData_New,
    CorrelatedBSModelData_New,
    DupireCalibration_New,
    DupireModelData_New,
    DupireRiskInputs_,
    MertonIVS_,
    MonteCarlo_ValueWithRisk,
    MonteCarlo_ValueWithSettings,
    Product_New,
    ValuationSettings,
)
from dal_jax.dates import Date
from dal_jax.errors import (
    CalibrationRiskBudgetExceeded,
    DupireSnapshotMismatch,
    InvalidCalibrationRiskRequest,
    InvalidDupireCalibration,
    InvalidDupireInput,
    InvalidDupirePullback,
    InvalidDupireQuote,
    InvalidDupireRiskRequest,
    InvalidIndex,
    InvalidModelIndex,
    InvalidModelParameter,
    InvalidPathCount,
    InvalidSetting,
    ScriptError,
)
from dal_jax.models import DupireCalibrationSnapshot, LocalVolSurface

pytestmark = pytest.mark.oracle

NAN = float("nan")

#  The densified inclusion grid for the shared fixture: inclusion spots
#  [60, 100, 140] with spacing 10, inclusion times [0.5, 1.0] with spacing 0.5.
GRID_SPOTS = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0]
GRID_TIMES = [0.5, 1.0]

#  Native snapshot vols (dal-python 2026.11.1, DupireCentral1e-4FixedBaseBand/v1)
#  for the shared fixture; the visible deviation from 0.2 is the deterministic
#  finite-difference truncation of the native stencil.  Rows outside the stable
#  band repeat the band-edge value.
BS_VOLS = [
    [0.20000000608747995, 0.1999999365140388],
    [0.20000000608747995, 0.1999999932657005],
    [0.19999999696922605, 0.1999999981807106],
    [0.20000000244580848, 0.20000000355117212],
    [0.20000000422950104, 0.20000000132808238],
    [0.19999999950300149, 0.20000000112884214],
    [0.19999999240532834, 0.19999999765417856],
    [0.19999998277789474, 0.19999999542089344],
    [0.19999998277789474, 0.1999999917339414],
]
MERTON_VOLS = [
    [0.22296697353361203, 0.21621225805886637],
    [0.22296697353361203, 0.2104867381129218],
    [0.21058634378777508, 0.207529356745557],
    [0.20624946043039596, 0.20592474746208103],
    [0.2045051413923498, 0.20499179462993322],
    [0.20374435804502866, 0.20442544881740712],
    [0.2034861110147458, 0.20408131192215961],
    [0.20360122153084273, 0.20388675002069445],
    [0.20360122153084273, 0.20380399991795162],
]
SMILE_VOLS = [
    [0.20957996803269704, 0.21940446034505975],
    [0.2096586165779147, 0.21948288503602156],
    [0.20974449244245075, 0.21956884190865478],
    [0.20983675754325712, 0.21966135880153292],
    [0.20993473627237283, 0.2197597128078344],
    [0.21003785961180252, 0.21986332533533487],
    [0.21014567219149338, 0.21997172677591725],
    [0.2102577904617912, 0.22008452899593023],
    [0.21037389278685595, 0.22020139736893363],
]
BUMPED_SPREADS = [[0.01, 0.02], [-0.01, 0.03], [0.005, -0.02]]
BUMPED_VOLS = [
    [0.2174101710019903, 0.22966003566278087],
    [0.2174101710019903, 0.22966007684264916],
    [0.22896721610979837, 0.23412563580223375],
    [0.22457043579772254, 0.24689031179039],
    [0.21924657910823026, 0.2593528436346213],
    [0.21590458091424558, 0.23902511463071713],
    [0.20800671332928022, 0.1914172010265099],
    [0.19914049146501295, 0.1444879853894669],
    [0.19914049146501295, 0.15181710968364862],
]


def _fields():
    return dict(
        quote_strikes=[75.0, 105.0, 135.0],
        quote_maturities=[0.4, 1.2],
        quote_spreads=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        inclusion_spots=[60.0, 100.0, 140.0],
        max_spot_spacing=10.0,
        inclusion_times=[0.5, 1.0],
        max_time_spacing=0.5,
    )


def _ours_inputs(**overrides):
    fields = _fields()
    fields.update(overrides)
    return DupireRiskInputs_(**fields)


def _their_inputs(dal, **overrides):
    fields = _fields()
    fields.update(overrides)
    if not isinstance(fields["quote_spreads"], str):
        fields["quote_spreads"] = dal.DoubleMatrix_(fields["quote_spreads"])
    return dal.DupireRiskInputs_(**fields)


def _our_base():
    return BSModelData_New(100.0, 0.2, 0.05, 0.02)


def _their_base(dal):
    return dal.BSModelData_New(100.0, 0.2, 0.05, 0.02)


def _smile(strike, maturity):
    return 0.2 + 0.0005 * (strike - 100.0) / 100.0 + 0.01 * maturity


def _our_callable():
    return IVS_(spot=100.0, rate=0.05, dividend_yield=0.02, implied_vol=_smile)


def _their_callable(dal, function=_smile, spot=100.0, rate=0.05, dividend_yield=0.02):
    class TrampolineIVS(dal.IVS_):
        def __init__(self):
            super().__init__(spot=spot, rate=rate, dividend_yield=dividend_yield)

        def implied_vol(self, strike, maturity):
            return function(strike, maturity)

    return TrampolineIVS()


def _error_of(message):
    return {
        "InvalidDupireQuote": InvalidDupireQuote,
        "InvalidDupireCalibration": InvalidDupireCalibration,
        "InvalidDupirePullback": InvalidDupirePullback,
        "InvalidDupireInput": InvalidDupireInput,
        "InvalidModelIndex": InvalidModelIndex,
        "InvalidIndex": InvalidIndex,
        "InvalidHybridComponent": ScriptError,
        "InvalidCalibrationRiskRequest": InvalidCalibrationRiskRequest,
        "InvalidDupireRiskRequest": InvalidDupireRiskRequest,
        "InvalidPathCount": InvalidPathCount,
        "InvalidSetting": InvalidSetting,
        "CalibrationRiskBudgetExceeded": CalibrationRiskBudgetExceeded,
        "DupireSnapshotMismatch": DupireSnapshotMismatch,
        "InvalidCurvatureRequest": InvalidCurvatureRequest,
        "BumpOverAAD": ScriptError,
        "DupireScriptCurvature": ScriptError,
    }[message.split(":")[0]]


def _assert_both_raise(dal, their_call, our_call, message):
    with pytest.raises((RuntimeError, TypeError), match=re.escape(message.split(":")[0])):
        their_call()
    with pytest.raises(_error_of(message), match=re.escape(message)) as error:
        our_call()
    assert str(error.value) == message


def _assert_snapshot_parity(ours, theirs, vols):
    assert isinstance(ours, DupireCalibrationSnapshot)
    assert list(ours.spots) == theirs.spots == GRID_SPOTS
    assert list(ours.times) == theirs.times == GRID_TIMES
    rows = [list(row) for row in ours.vols]
    assert rows == theirs.vols.to_rows() == vols
    assert ours.algorithm == theirs.algorithm == "DupireCentral1e-4FixedBaseBand/v1"
    assert (ours.spot, ours.rate, ours.dividend_yield) == (
        theirs.spot,
        theirs.rate,
        theirs.dividend_yield,
    )
    assert ours.surface.name == theirs.surface.name


def test_snapshot_parity_black_scholes_base(dal):
    ours = DupireCalibration_New(_our_base(), _ours_inputs(), name="frozen")
    theirs = dal.DupireCalibration_New(_their_base(dal), _their_inputs(dal), name="frozen")
    _assert_snapshot_parity(ours, theirs, BS_VOLS)
    assert (ours.spot, ours.rate, ours.dividend_yield) == (100.0, 0.05, 0.02)


def test_snapshot_parity_merton_base(dal):
    kwargs = dict(spot=100.0, vol=0.2, intensity=0.08, average_jump=-0.1, jump_std=0.15)
    ours = DupireCalibration_New(MertonIVS_(**kwargs), _ours_inputs(), name="merton")
    theirs = dal.DupireCalibration_New(dal.MertonIVS_(**kwargs), _their_inputs(dal), name="merton")
    _assert_snapshot_parity(ours, theirs, MERTON_VOLS)
    assert (ours.spot, ours.rate, ours.dividend_yield) == (100.0, 0.0, 0.0)


def test_snapshot_parity_callable_base(dal):
    ours = DupireCalibration_New(_our_callable(), _ours_inputs(), name="pyivs")
    theirs = dal.DupireCalibration_New(_their_callable(dal), _their_inputs(dal), name="pyivs")
    _assert_snapshot_parity(ours, theirs, SMILE_VOLS)
    assert (ours.spot, ours.rate, ours.dividend_yield) == (100.0, 0.05, 0.02)


def test_snapshot_parity_bumped_quotes(dal):
    ours = DupireCalibration_New(_our_base(), _ours_inputs(quote_spreads=BUMPED_SPREADS))
    theirs = dal.DupireCalibration_New(
        _their_base(dal), _their_inputs(dal, quote_spreads=BUMPED_SPREADS)
    )
    _assert_snapshot_parity(ours, theirs, BUMPED_VOLS)


def test_snapshot_roundtrips_inputs_and_feeds_local_vol(dal):
    ours = DupireCalibration_New(_our_base(), _ours_inputs(), name="frozen")
    theirs = dal.DupireCalibration_New(_their_base(dal), _their_inputs(dal), name="frozen")
    assert list(ours.inputs.quote_strikes) == theirs.inputs.quote_strikes == [75.0, 105.0, 135.0]
    assert list(ours.inputs.quote_maturities) == theirs.inputs.quote_maturities == [0.4, 1.2]
    assert [list(row) for row in ours.inputs.quote_spreads] == theirs.inputs.quote_spreads.to_rows()
    assert (
        list(ours.inputs.inclusion_spots) == theirs.inputs.inclusion_spots == [60.0, 100.0, 140.0]
    )
    assert ours.inputs.max_spot_spacing == theirs.inputs.max_spot_spacing == 10.0
    assert list(ours.inputs.inclusion_times) == theirs.inputs.inclusion_times == [0.5, 1.0]
    assert ours.inputs.max_time_spacing == theirs.inputs.max_time_spacing == 0.5
    assert isinstance(ours.surface, LocalVolSurface)
    assert list(ours.surface.spots) == list(ours.spots)
    assert list(ours.surface.times) == list(ours.times)
    assert ours.surface.vols == ours.vols


def test_matches_ignores_name_but_not_inputs(dal):
    ours = DupireCalibration_New(_our_base(), _ours_inputs(), name="frozen")
    theirs = dal.DupireCalibration_New(_their_base(dal), _their_inputs(dal), name="frozen")
    cases = (
        ("frozen", {}, True),
        ("renamed", {}, True),
        ("frozen", {"quote_spreads": [[0.01, 0.0], [0.0, 0.0], [0.0, 0.0]]}, False),
        ("frozen", {"inclusion_spots": [60.0, 140.0]}, False),
    )
    for name, overrides, expected in cases:
        other = DupireCalibration_New(_our_base(), _ours_inputs(**overrides), name=name)
        native_other = dal.DupireCalibration_New(
            _their_base(dal), _their_inputs(dal, **overrides), name=name
        )
        assert ours.matches(other) is expected
        assert theirs.matches(native_other) is expected


@pytest.mark.parametrize(
    ("overrides", "spots"),
    [
        (
            {"max_spot_spacing": 15.0},
            [
                60.0,
                73.33333333333333,
                86.66666666666666,
                100.0,
                113.33333333333333,
                126.66666666666666,
                140.0,
            ],
        ),
        ({"max_spot_spacing": 5.0}, [60.0 + 5.0 * step for step in range(17)]),
        ({"max_spot_spacing": 40.0}, [60.0, 100.0, 140.0]),
    ],
)
def test_densified_spot_grid_parity(dal, overrides, spots):
    ours = DupireCalibration_New(_our_base(), _ours_inputs(**overrides))
    theirs = dal.DupireCalibration_New(_their_base(dal), _their_inputs(dal, **overrides))
    assert list(ours.spots) == theirs.spots == spots


@pytest.mark.parametrize(
    ("overrides", "times"),
    [
        ({"max_time_spacing": 0.2}, [0.2, 0.35, 0.5, 0.6666666666666666, 0.8333333333333333, 1.0]),
        ({"max_time_spacing": 0.3}, [0.3, 0.5, 0.75, 1.0]),
        ({"max_time_spacing": 1.0}, [0.5, 1.0]),
        ({"max_time_spacing": 2.0}, [0.5, 1.0, 2.0]),
    ],
)
def test_densified_time_grid_parity(dal, overrides, times):
    ours = DupireCalibration_New(_our_base(), _ours_inputs(**overrides))
    theirs = dal.DupireCalibration_New(_their_base(dal), _their_inputs(dal, **overrides))
    assert list(ours.times) == theirs.times == times


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"inclusion_spots": [60.0, 60.0, 140.0]},
            "InvalidDupireQuote: axis must be finite, positive and strictly increasing; field=inclusionSpots",
        ),
        (
            {"quote_maturities": [1.2, 0.4]},
            "InvalidDupireQuote: axis must be finite, positive and strictly increasing; field=quoteMaturities",
        ),
        (
            {"inclusion_times": [0.0, 1.0]},
            "InvalidDupireQuote: axis must be finite, positive and strictly increasing; field=inclusionTimes",
        ),
        (
            {"quote_strikes": [-75.0, 105.0, 135.0]},
            "InvalidDupireQuote: axis must be finite, positive and strictly increasing; field=quoteStrikes",
        ),
        ({"quote_strikes": []}, "InvalidDupireQuote: empty or oversized axis; field=quoteStrikes"),
        (
            {"quote_spreads": [[0.0, 0.0], [0.0, 0.0]]},
            "InvalidDupireQuote: spread matrix must match strike/maturity axes",
        ),
        ({"quote_spreads": [[NAN, NAN]] * 3}, "InvalidDupireQuote: non-finite spread"),
        (
            {"max_spot_spacing": 0.0},
            "InvalidDupireCalibration: grid spacing must be finite and positive",
        ),
        (
            {"max_spot_spacing": -10.0},
            "InvalidDupireCalibration: grid spacing must be finite and positive",
        ),
        (
            {"max_time_spacing": 0.0},
            "InvalidDupireCalibration: grid spacing must be finite and positive",
        ),
        (
            {"max_spot_spacing": 1e-16},
            "InvalidDupireCalibration: grid spacing is below coordinate precision",
        ),
        (
            {"max_time_spacing": 1e-16},
            "InvalidDupireCalibration: grid spacing is below coordinate precision",
        ),
        (
            {"max_spot_spacing": 1e-9},
            "InvalidDupireCalibration: grid spacing exceeds the supported fill extent",
        ),
        (
            {"max_time_spacing": 1e-11},
            "InvalidDupireCalibration: grid spacing exceeds the supported fill extent",
        ),
        (
            {"inclusion_spots": [10.0, 500.0], "max_spot_spacing": 300.0},
            "InvalidDupireCalibration: no spot nodes inside stable calibration band",
        ),
    ],
)
def test_calibration_validates_inputs_like_native(dal, overrides, message):
    _assert_both_raise(
        dal,
        lambda: dal.DupireCalibration_New(_their_base(dal), _their_inputs(dal, **overrides)),
        lambda: DupireCalibration_New(_our_base(), _ours_inputs(**overrides)),
        message,
    )


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (
            lambda strike, maturity: -0.2,
            "InvalidDupireCalibration: base implied volatility must be finite and nonnegative",
        ),
        (
            lambda strike, maturity: NAN,
            "InvalidDupireCalibration: IVS_; implied_vol; type=float, value=nan; "
            "expected finite int or float, excluding bool and enums",
        ),
        (
            lambda strike, maturity: "0.2",
            "InvalidDupireCalibration: IVS_; implied_vol; type=str, value='0.2'; "
            "expected finite int or float, excluding bool and enums",
        ),
        (
            lambda strike, maturity: 0.5 - 0.35 * maturity,
            "InvalidDupireCalibration: local variance is nonpositive or non-finite",
        ),
    ],
)
def test_calibration_validates_callable_base_like_native(dal, function, message):
    _assert_both_raise(
        dal,
        lambda: dal.DupireCalibration_New(_their_callable(dal, function), _their_inputs(dal)),
        lambda: DupireCalibration_New(
            IVS_(spot=100.0, rate=0.05, dividend_yield=0.02, implied_vol=function), _ours_inputs()
        ),
        message,
    )


def test_calibration_rejects_bad_names_like_native(dal):
    cases = [
        (
            3.5,
            "InvalidDupireCalibration: DupireCalibration_New; name; type=float, value=3.5; expected str or DAL String_ without NUL",
        ),
        (
            "a\x00b",
            "InvalidDupireCalibration: DupireCalibration_New; name; type=str, value='a\\x00b'; expected str or DAL String_ without NUL",
        ),
    ]
    for name, message in cases:
        _assert_both_raise(
            dal,
            lambda name=name: dal.DupireCalibration_New(
                _their_base(dal), _their_inputs(dal), name=name
            ),
            lambda name=name: DupireCalibration_New(_our_base(), _ours_inputs(), name=name),
            message,
        )


def test_calibration_rejects_bad_dispatch_like_native(dal):
    inputs_message = "InvalidDupirePullback: DupireCalibration_New; inputs; type=dict; expected DupireRiskInputs_"
    _assert_both_raise(
        dal,
        lambda: dal.DupireCalibration_New(_their_base(dal), {"quote_strikes": [1.0]}),
        lambda: DupireCalibration_New(_our_base(), {"quote_strikes": [1.0]}),
        inputs_message,
    )
    base_message = "InvalidDupireCalibration: DupireCalibration_New; base; type=str, value='BS'; expected BSModelData_ or IVS_"
    _assert_both_raise(
        dal,
        lambda: dal.DupireCalibration_New("BS", _their_inputs(dal)),
        lambda: DupireCalibration_New("BS", _ours_inputs()),
        base_message,
    )
    non_bs_message = (
        "InvalidDupireCalibration: DupireCalibration_New; base must be BSModelData_ or IVS_"
    )
    ours = CorrelatedBSModelData_New(["EQ[A]"], [100.0], [0.2], [0.0], 0.03, [[1.0]])
    _assert_both_raise(
        dal,
        lambda: dal.DupireCalibration_New(
            dal.CorrelatedBSModelData_New(
                ["EQ[A]"], [100.0], [0.2], [0.0], 0.03, dal.DoubleMatrix_(1, 1, 1.0)
            ),
            _their_inputs(dal),
        ),
        lambda: DupireCalibration_New(ours, _ours_inputs()),
        non_bs_message,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "quote_strikes",
            "bad",
            "InvalidDupireQuote: DupireRiskInputs_; quote_strikes; type=str, value='bad'; expected list, tuple or DoubleVector",
        ),
        (
            "inclusion_spots",
            [True, 100.0, 140.0],
            "InvalidDupireCalibration: DupireRiskInputs_; inclusion_spots; type=bool, value=True; "
            "expected finite int or float, excluding bool and enums",
        ),
        (
            "inclusion_times",
            ["0.5", 1.0],
            "InvalidDupireCalibration: DupireRiskInputs_; inclusion_times; type=str, value='0.5'; "
            "expected finite int or float, excluding bool and enums",
        ),
        (
            "inclusion_spots",
            [NAN, 100.0, 140.0],
            "InvalidDupireCalibration: DupireRiskInputs_; inclusion_spots; type=float, value=nan; "
            "expected finite int or float, excluding bool and enums",
        ),
        (
            "max_spot_spacing",
            True,
            "InvalidDupireCalibration: DupireRiskInputs_; max_spot_spacing; type=bool, value=True; "
            "expected finite int or float, excluding bool and enums",
        ),
        (
            "max_spot_spacing",
            "10",
            "InvalidDupireCalibration: DupireRiskInputs_; max_spot_spacing; type=str, value='10'; "
            "expected finite int or float, excluding bool and enums",
        ),
        (
            "max_time_spacing",
            NAN,
            "InvalidDupireCalibration: DupireRiskInputs_; max_time_spacing; type=float, value=nan; "
            "expected finite int or float, excluding bool and enums",
        ),
        (
            "quote_spreads",
            5.0,
            "InvalidDupirePullback: DupireRiskInputs_; quote_spreads; type=float, value=5.0; expected DoubleMatrix_",
        ),
    ],
)
def test_risk_inputs_type_errors_match_native(dal, field, value, message):
    fields = _fields()
    fields[field] = value
    if field != "quote_spreads":
        fields["quote_spreads"] = dal.DoubleMatrix_(3, 2)
    _assert_both_raise(
        dal,
        lambda: dal.DupireRiskInputs_(**fields),
        lambda: DupireRiskInputs_(**{**_fields(), field: value}),
        message,
    )


def test_risk_inputs_reject_positional_and_missing_arguments(dal):
    values = list(_fields().values())
    with pytest.raises(TypeError):
        dal.DupireRiskInputs_(*values)
    with pytest.raises(TypeError):
        DupireRiskInputs_(*values)  # pylint: disable=too-many-function-args,missing-kwoa
    with pytest.raises(TypeError):
        dal.DupireRiskInputs_(quote_strikes=[75.0])
    with pytest.raises(TypeError):
        DupireRiskInputs_(quote_strikes=[75.0])  # pylint: disable=missing-kwoa


def test_merton_inputs_validate_like_native(dal):
    kwargs = dict(spot=100.0, vol=0.2, intensity=0.08, average_jump=-0.1, jump_std=0.15)
    ours, theirs = MertonIVS_(**kwargs), dal.MertonIVS_(**kwargs)
    assert (ours.spot, ours.rate, ours.dividend_yield) == (
        theirs.spot,
        theirs.rate,
        theirs.dividend_yield,
    )
    positional = [100.0, 0.2, 0.08, -0.1, 0.15]
    with pytest.raises(TypeError):
        dal.MertonIVS_(*positional)
    with pytest.raises(TypeError):
        MertonIVS_(*positional)  # pylint: disable=too-many-function-args,missing-kwoa
    cases = [
        ("spot", -1.0, "InvalidDupireCalibration: MertonIVS_; spot must be finite and positive"),
        (
            "vol",
            NAN,
            "InvalidDupireCalibration: MertonIVS_; vol; type=float, value=nan; "
            "expected finite int or float, excluding bool and enums",
        ),
        ("vol", -0.1, "InvalidDupireCalibration: MertonIVS_; vol must be finite and nonnegative"),
        (
            "intensity",
            -1.0,
            "InvalidDupireCalibration: MertonIVS_; intensity must be finite and nonnegative",
        ),
        (
            "average_jump",
            float("inf"),
            "InvalidDupireCalibration: MertonIVS_; average_jump; type=float, value=inf; "
            "expected finite int or float, excluding bool and enums",
        ),
        (
            "jump_std",
            -0.2,
            "InvalidDupireCalibration: MertonIVS_; jump_std must be finite and nonnegative",
        ),
        (
            "spot",
            True,
            "InvalidDupireCalibration: MertonIVS_; spot; type=bool, value=True; "
            "expected finite int or float, excluding bool and enums",
        ),
    ]
    for field, value, message in cases:
        _assert_both_raise(
            dal,
            lambda field=field, value=value: dal.MertonIVS_(**{**kwargs, field: value}),
            lambda field=field, value=value: MertonIVS_(**{**kwargs, field: value}),
            message,
        )


def test_callable_ivs_validates_like_native(dal):
    ours = IVS_(spot=50.0, implied_vol=_smile)
    theirs = dal.IVS_(spot=50.0)
    assert (ours.spot, ours.rate, ours.dividend_yield) == (
        theirs.spot,
        theirs.rate,
        theirs.dividend_yield,
    )
    cases = [
        (
            "spot",
            "50",
            "InvalidDupireCalibration: IVS_; spot; type=str, value='50'; "
            "expected finite int or float, excluding bool and enums",
        ),
        ("spot", 0.0, "InvalidDupireCalibration: IVS_; spot must be positive"),
        (
            "rate",
            NAN,
            "InvalidDupireCalibration: IVS_; rate; type=float, value=nan; "
            "expected finite int or float, excluding bool and enums",
        ),
    ]
    for field, value, message in cases:
        _assert_both_raise(
            dal,
            lambda field=field, value=value: dal.IVS_(**{"spot": 50.0, field: value}),
            lambda field=field, value=value: IVS_(
                **{"spot": 50.0, "implied_vol": _smile, field: value}
            ),
            message,
        )
    override_message = "InvalidDupireCalibration: IVS_; implied_vol override is required"
    _assert_both_raise(
        dal,
        lambda: dal.DupireCalibration_New(dal.IVS_(spot=100.0), _their_inputs(dal)),
        lambda: DupireCalibration_New(IVS_(spot=100.0), _ours_inputs()),
        override_message,
    )


#  ---------------------------------------------------------------------------
#  DupireModelData_New: engine pricing and risk parity (the spec fixture).


VALUATION = ValuationSettings(evaluation_date=Date.ymd(2026, 9, 12))
INDEX = "EQ[LOCAL]"
AXIS_LABELS = (
    [f"spot:{INDEX}", f"div:{INDEX}"]
    + [f"lvol:{INDEX}:{i}:{j}" for i in range(len(GRID_SPOTS)) for j in range(len(GRID_TIMES))]
    + ["rate:USD"]
)
#  Native anchors (dal-python 2026.11.1): PV at 4096 paths and the AAD risk
#  result at 257 paths for pay PAYS FIX(EQ[LOCAL])^2 / 100 at T=1.
PV_ANCHOR = 105.040477399776
RISK_VALUE_ANCHOR = 104.36149296571121
JACOBIAN_ANCHOR = [
    2.087229838097331,
    -208.72298506454018,
    -0.001764818918493701,
    -0.001764818918493701,
    1.4192315218325229,
    0.27278357450313195,
    1.6032207177681372,
    0.46161200438291494,
    2.6593385859953456,
    0.04219775209870143,
    15.602249517503878,
    1.0778517587380083,
    5.155820819562145,
    1.1960117599900137,
    4.7628340285264255,
    2.1318296127421226,
    0.21429338508498674,
    0.43031470718152737,
    -1.6215939997989777,
    -1.3196185640440747,
    104.36149209882889,
]


def _our_model(**overrides):
    kwargs = dict(name="local", max_step=0.25)
    kwargs.update(overrides)
    return DupireModelData_New(_our_calibration(), INDEX, "USD", "F_LOCAL", **kwargs)


def _their_model(dal, **overrides):
    kwargs = dict(name="local", max_step=0.25)
    kwargs.update(overrides)
    return dal.DupireModelData_New(_their_calibration(dal), INDEX, "USD", "F_LOCAL", **kwargs)


def _our_calibration():
    return DupireCalibration_New(_our_base(), _ours_inputs(), name="frozen")


def _their_calibration(dal):
    return dal.DupireCalibration_New(_their_base(dal), _their_inputs(dal), name="frozen")


def _our_product():
    return Product_New([Date.ymd(2027, 9, 12)], [f"pay PAYS FIX({INDEX}) * FIX({INDEX}) / 100"])


def _their_product(dal):
    return dal.Product_New(
        [dal.Date_(2027, 9, 12)], [f"pay PAYS FIX({INDEX}) * FIX({INDEX}) / 100"]
    )


def _their_valuation(dal):
    return dal.ScriptValuationSettings_(evaluation_date=dal.Date_(2026, 9, 12))


def test_model_pv_parity_on_spec_fixture(dal):
    ours = MonteCarlo_ValueWithSettings(_our_product(), _our_model(), 4096, valuation=VALUATION)
    theirs = dal.MonteCarlo_ValueWithSettings(
        _their_product(dal), _their_model(dal), 4096, valuation=_their_valuation(dal)
    )
    assert ours["PV"] == pytest.approx(theirs["PV"], rel=1e-10)
    assert ours["PV"] == pytest.approx(PV_ANCHOR, rel=1e-10)


@pytest.mark.parametrize("max_step", [0.1, 0.5, 1.5])
def test_model_pv_parity_across_max_steps(dal, max_step):
    ours = MonteCarlo_ValueWithSettings(
        _our_product(), _our_model(max_step=max_step), 4096, valuation=VALUATION
    )
    theirs = dal.MonteCarlo_ValueWithSettings(
        _their_product(dal),
        _their_model(dal, max_step=max_step),
        4096,
        valuation=_their_valuation(dal),
    )
    assert ours["PV"] == pytest.approx(theirs["PV"], rel=1e-10)


def test_model_risk_axis_and_jacobian_parity(dal):
    ours = MonteCarlo_ValueWithRisk(_our_product(), _our_model(), 257, valuation=VALUATION)
    theirs = dal.MonteCarlo_ValueWithRisk(
        _their_product(dal), _their_model(dal), 257, valuation=_their_valuation(dal)
    )
    _assert_risk_parity(ours, theirs)
    assert ours.values[0] == pytest.approx(RISK_VALUE_ANCHOR, rel=1e-12)
    np.testing.assert_allclose(np.asarray(ours.jacobian)[0], JACOBIAN_ANCHOR, rtol=1e-9, atol=1e-12)


def test_model_risk_axis_reports_snapshot_coordinates(dal):
    ours = MonteCarlo_ValueWithRisk(_our_product(), _our_model(), 257, valuation=VALUATION)
    assert [c.label for c in ours.input_axis] == AXIS_LABELS
    assert [c.id for c in ours.input_axis] == [f"model:{n}" for n in range(21)]
    assert all(c.family == "model" for c in ours.input_axis)
    assert all(c.native_unit == "model-coordinate" for c in ours.input_axis)
    assert all(c.physical_unit is None for c in ours.input_axis)


def test_model_risk_axis_carries_snapshot_values(dal):
    ours = MonteCarlo_ValueWithRisk(_our_product(), _our_model(), 257, valuation=VALUATION)
    #  The axis carries the calibrated snapshot verbatim: spot, dividend, the
    #  row-major surface nodes, then the deterministic rate.
    expected_values = [100.0, 0.02] + [vol for row in BS_VOLS for vol in row] + [0.05]
    assert [c.value for c in ours.input_axis] == expected_values


def test_model_multi_event_risk_parity(dal):
    dates = [Date.ymd(2027, 3, 12), Date.ymd(2028, 3, 12)]
    events = [f"pay PAYS FIX({INDEX}) * 0.01", f"pay PAYS FIX({INDEX}) / 100"]
    ours = MonteCarlo_ValueWithRisk(
        Product_New(dates, events), _our_model(max_step=0.1), 257, valuation=VALUATION
    )
    theirs = dal.MonteCarlo_ValueWithRisk(
        dal.Product_New([dal.Date_(2027, 3, 12), dal.Date_(2028, 3, 12)], events),
        _their_model(dal, max_step=0.1),
        257,
        valuation=_their_valuation(dal),
    )
    _assert_risk_parity(ours, theirs)
    assert [c.label for c in ours.input_axis] == AXIS_LABELS


def test_model_ctor_validates_like_native(dal):
    calibration = _our_calibration()
    native_calibration = _their_calibration(dal)
    cases = [
        #  (overrides, message)
        (
            {"index": "EQ[X]", "currency": "USD", "factor": "F", "calibration": "cal"},
            "InvalidDupirePullback: DupireModelData_New; calibration; type=str, value='cal'; expected DupireCalibrationSnapshot_",
        ),
        (
            {"index": 5, "currency": "USD", "factor": "F"},
            "InvalidDupireInput: DupireModelData_New; index; type=int, value=5; expected str or DAL String_ without NUL",
        ),
        (
            {"index": INDEX, "currency": 7, "factor": "F"},
            "InvalidDupireInput: DupireModelData_New; currency; type=int, value=7; expected str or DAL String_ without NUL",
        ),
        (
            {"index": INDEX, "currency": "USD", "factor": None},
            "InvalidDupireInput: DupireModelData_New; factor; type=NoneType, value=None; expected str or DAL String_ without NUL",
        ),
        (
            {"index": INDEX, "currency": "USD", "factor": "F", "name": "a\x00b"},
            "InvalidDupireInput: DupireModelData_New; name; type=str, value='a\\x00b'; expected str or DAL String_ without NUL",
        ),
        (
            {"index": "", "currency": "USD", "factor": "F"},
            "InvalidModelIndex: expected an ordinary EQ index: ",
        ),
        (
            {"index": "SPAM", "currency": "USD", "factor": "F"},
            "InvalidModelIndex: expected an ordinary EQ index: SPAM",
        ),
        (
            {"index": "IR[USD,XYZ]", "currency": "USD", "factor": "F"},
            "InvalidModelIndex: expected an ordinary EQ index: IR[USD,XYZ]",
        ),
        (
            {"index": "FOO[BAR]", "currency": "USD", "factor": "F"},
            "InvalidModelIndex: expected an ordinary EQ index: FOO[BAR]",
        ),
        (
            {"index": "foo:bar", "currency": "USD", "factor": "F"},
            "InvalidModelIndex: expected an ordinary EQ index: foo:bar",
        ),
        (
            {"index": "COMM[GOLD]", "currency": "USD", "factor": "F"},
            "InvalidModelIndex: expected an ordinary EQ index: COMM[GOLD]",
        ),
        (
            {"index": "EQ[ABC", "currency": "USD", "factor": "F"},
            "InvalidIndex: EQ[ABC; equity index must contain a nonempty EQ[name]",
        ),
        (
            {"index": "EQ[]", "currency": "USD", "factor": "F"},
            "InvalidIndex: EQ[]; equity index must contain a nonempty EQ[name]",
        ),
        (
            {"index": "eq[local", "currency": "USD", "factor": "F"},
            "InvalidIndex: eq[local; equity index must contain a nonempty EQ[name]",
        ),
        (
            {"index": INDEX, "currency": "", "factor": "F"},
            "InvalidHybridComponent: local-vol equity name, currency, factor, and surface are required",
        ),
        (
            {"index": INDEX, "currency": "USD", "factor": ""},
            "InvalidHybridComponent: local-vol equity name, currency, factor, and surface are required",
        ),
        (
            {"index": INDEX, "currency": "USD", "factor": "F", "max_step": 0.0},
            "InvalidDupireInput: DupireModelData_New; max_step must be positive",
        ),
        (
            {"index": INDEX, "currency": "USD", "factor": "F", "max_step": -0.1},
            "InvalidDupireInput: DupireModelData_New; max_step must be positive",
        ),
        (
            {"index": INDEX, "currency": "USD", "factor": "F", "max_step": NAN},
            "InvalidDupireCalibration: DupireModelData_New; max_step; type=float, value=nan; "
            "expected finite int or float, excluding bool and enums",
        ),
        (
            {"index": INDEX, "currency": "USD", "factor": "F", "max_step": True},
            "InvalidDupireCalibration: DupireModelData_New; max_step; type=bool, value=True; "
            "expected finite int or float, excluding bool and enums",
        ),
        (
            {"index": INDEX, "currency": "USD", "factor": "F", "max_step": "0.25"},
            "InvalidDupireCalibration: DupireModelData_New; max_step; type=str, value='0.25'; "
            "expected finite int or float, excluding bool and enums",
        ),
    ]
    for overrides, message in cases:
        fields = dict(calibration=calibration, index=INDEX, currency="USD", factor="F")
        fields.update(overrides)
        native_fields = dict(fields)
        if native_fields["calibration"] is calibration:
            native_fields["calibration"] = native_calibration
        _assert_both_raise(
            dal,
            lambda native_fields=native_fields: dal.DupireModelData_New(**native_fields),
            lambda fields=fields: DupireModelData_New(**fields),
            message,
        )
    with pytest.raises(TypeError):
        dal.DupireModelData_New(native_calibration, INDEX, "USD", "F_LOCAL", 0.25)
    with pytest.raises(TypeError):
        DupireModelData_New(calibration, INDEX, "USD", "F_LOCAL", 0.25)  # pylint: disable=too-many-function-args


def test_model_defaults_and_lowercase_index(dal):
    plain = DupireModelData_New(_our_calibration(), "eq[local]", "USD", "F_LOCAL")
    theirs = dal.DupireModelData_New(_their_calibration(dal), "eq[local]", "USD", "F_LOCAL")
    ours = MonteCarlo_ValueWithSettings(_our_product(), plain, 4096, valuation=VALUATION)
    theirs_pv = dal.MonteCarlo_ValueWithSettings(
        _their_product(dal), theirs, 4096, valuation=_their_valuation(dal)
    )
    #  Default max_step (1/12) prices differently from the 0.25 anchor fixture;
    #  this pins canonicalisation and defaults, not the anchor.
    assert ours["PV"] == pytest.approx(theirs_pv["PV"], rel=1e-10)
    #  Canonicalisation upper-cases only the EQ prefix; the inner name keeps
    #  its case, as native's RiskLabels report it.
    risk = MonteCarlo_ValueWithRisk(_our_product(), plain, 129, valuation=VALUATION)
    native_risk = dal.MonteCarlo_ValueWithRisk(
        _their_product(dal), theirs, 129, valuation=_their_valuation(dal)
    )
    labels = [c.label for c in risk.input_axis]
    assert labels[:2] == ["spot:EQ[local]", "div:EQ[local]"]
    assert labels[-1] == "rate:USD"
    assert labels == [c.label for c in native_risk.input_axis]


def test_model_unsupported_currency_diverges_from_native(dal):
    #  Known divergence: native accepts any currency string end-to-end (a
    #  probed XYZ model even prices), while our hybrid reuses the LocalVol
    #  currency gate and rejects construction.  Do not "fix" one side without
    #  the other; revisit only if native starts validating currencies.
    assert dal.DupireModelData_New(_their_calibration(dal), INDEX, "XYZ", "F") is not None
    with pytest.raises(InvalidModelParameter, match="currency must be supported"):
        DupireModelData_New(_our_calibration(), INDEX, "XYZ", "F")


#  ---------------------------------------------------------------------------
#  Script-risk planners and the quote pullback (native DupireScriptRisk* chain).


from dal_jax.api import (  # noqa: E402
    CalibrationRiskRequest_,
    DupireDirectQuoteAdjoints_,
    DupireParameterAdjoints_,
    DupireParameterAdjoints_FromRisk,
    DupireQuoteBinding_,
    DupireQuoteRisk_New,
    DupireScriptQuoteRisk_New,
    DupireScriptRiskPlan_New,
    DupireScriptRiskRequest_,
    DupireScriptRiskResult_New,
    MonteCarloSettings_,
    RiskRequest_,
)

RISK_PRODUCT_EVENTS = ["0.0", f"pay PAYS FIX({INDEX}) * FIX({INDEX}) / 100 + 3 * QUOTE"]
QUOTE_JACOBIAN_ANCHOR = [22.25711331187233, 3.0437272799317725]
QUOTE_REPORTED_ANCHOR = [0.2225711331187233, 1.5218636399658863]
PULLBACK_ROW_ANCHOR = [3.0437272799317725, 7.320435909437947]


def _our_risk_product():
    return Product_New(["QUOTE", Date.ymd(2027, 9, 12)], RISK_PRODUCT_EVENTS)


def _their_risk_product(dal):
    return dal.Product_New(["QUOTE", dal.Date_(2027, 9, 12)], RISK_PRODUCT_EVENTS)


def _our_script_request(**overrides):
    kwargs = dict(num_paths=257, valuation=VALUATION)
    kwargs.update(overrides)
    return DupireScriptRiskRequest_(**kwargs)


def _their_script_request(dal, **overrides):
    kwargs = dict(num_paths=257, valuation=_their_valuation(dal))
    kwargs.update(overrides)
    return dal.DupireScriptRiskRequest_(**kwargs)


def _our_plan(**overrides):
    kwargs = dict(
        product=_our_risk_product(),
        modelData=_our_model(),
        calibration=_our_calibration(),
        component="equity",
        request=_our_script_request(),
    )
    kwargs.update(overrides)
    return DupireScriptRiskPlan_New(**kwargs)


def _their_plan(dal, **overrides):
    kwargs = dict(
        product=_their_risk_product(dal),
        modelData=_their_model(dal),
        calibration=_their_calibration(dal),
        component="equity",
        request=_their_script_request(dal),
    )
    kwargs.update(overrides)
    return dal.DupireScriptRiskPlan_New(**kwargs)


def _quote_records(plan):
    return [
        (c.id, c.label, c.ordinal, c.row, c.column, c.native_unit, c.report_scale, c.value)
        for c in plan.quote_plan.input_axis
    ]


def _assert_quote_plan_parity(ours, theirs):
    assert _quote_records(ours) == _quote_records(theirs)
    assert list(ours.quote_plan.selected_ordinals) == list(theirs.quote_plan.selected_ordinals)
    assert [c.id for c in ours.quote_plan.complete_input_axis] == [
        c.id for c in theirs.quote_plan.complete_input_axis
    ]


def test_script_risk_chain_parity(dal):
    request = dict(
        quotes=CalibrationRiskRequest_(inputs=["quote:3", "quote:0"], report_factors=[0.01, 0.5]),
        direct_bindings=[DupireQuoteBinding_(constant_ordinal=0, quote_id="quote:3")],
    )
    ours_plan = _our_plan(request=_our_script_request(**request))
    theirs_plan = _their_plan(
        dal, request=_their_script_request(dal, **_native_quotes(dal, request))
    )
    ours = DupireScriptRiskResult_New(ours_plan)
    theirs = dal.DupireScriptRiskResult_New(theirs_plan)
    _assert_risk_parity(ours.valuation, theirs.valuation)
    _assert_quote_plan_parity(ours_plan, theirs_plan)
    assert ours.method == theirs.method == "NativeAADThenNativeAADCalibrationVJP"
    assert ours.component == theirs.component == "equity"
    assert [c.id for c in ours_plan.complete_input_axis] == [
        c.id for c in theirs_plan.complete_input_axis
    ]
    assert [c.id for c in ours_plan.required_input_axis] == [
        c.id for c in theirs_plan.required_input_axis
    ]
    np.testing.assert_allclose(
        [list(row) for row in ours.quote_risk.jacobian],
        theirs.quote_risk.jacobian.to_rows(),
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        [list(row) for row in ours.quote_risk.jacobian], [QUOTE_JACOBIAN_ANCHOR], rtol=1e-7
    )
    np.testing.assert_allclose(
        [list(row) for row in ours.quote_risk.reported_jacobian],
        [QUOTE_REPORTED_ANCHOR],
        rtol=1e-7,
    )


def _native_quotes(dal, request):
    native = dict(request)
    if "quotes" in native:
        quotes = native["quotes"]
        native["quotes"] = dal.CalibrationRiskRequest_(
            inputs=list(quotes.inputs),
            report_factors=list(quotes.report_factors) if quotes.report_factors else None,
        )
    if "direct_bindings" in native:
        native["direct_bindings"] = [
            dal.DupireQuoteBinding_(constant_ordinal=b.constant_ordinal, quote_id=b.quote_id)
            for b in native["direct_bindings"]
        ]
    return native


def test_script_risk_default_request_parity(dal):
    ours = DupireScriptRiskResult_New(_our_plan())
    theirs = dal.DupireScriptRiskResult_New(_their_plan(dal))
    _assert_risk_parity(ours.valuation, theirs.valuation)
    assert list(_our_plan().quote_plan.selected_ordinals) == list(range(6))
    np.testing.assert_allclose(
        [list(row) for row in ours.quote_risk.jacobian],
        theirs.quote_risk.jacobian.to_rows(),
        rtol=1e-7,
        atol=1e-9,
    )
    assert ours.quote_risk.quote_risk.method == theirs.quote_risk.quote_risk.method


def test_script_risk_expired_and_retrained_methods(dal):
    expired_request = dict(
        num_paths=257, valuation=ValuationSettings(evaluation_date=Date.ymd(2028, 1, 1))
    )
    expired = DupireScriptRiskResult_New(
        _our_plan(request=DupireScriptRiskRequest_(**expired_request))
    )
    native_expired = dal.DupireScriptRiskResult_New(
        _their_plan(
            dal,
            request=dal.DupireScriptRiskRequest_(
                num_paths=257,
                valuation=dal.ScriptValuationSettings_(evaluation_date=dal.Date_(2028, 1, 1)),
            ),
        )
    )
    assert expired.method == native_expired.method == "ExpiredThenNativeAADCalibrationVJP"
    assert expired.valuation.values == native_expired.valuation.values == [0.0]
    np.testing.assert_allclose(
        [list(row) for row in expired.quote_risk.jacobian], [[0.0] * 6], atol=1e-12
    )
    retrained_request = dict(
        num_paths=64,
        valuation=VALUATION,
        simulation=MonteCarloSettings_(
            enable_aad=True, lsmc_training_paths=64, lsmc_policy_risk_mode="RetrainedBump"
        ),
    )
    events = [f"pay PAYS 1 EXERCISE MAX(100 - FIX({INDEX}), 0)"] * 2
    dates = [Date.ymd(2027, 3, 12), Date.ymd(2027, 9, 12)]
    ours = DupireScriptRiskResult_New(
        DupireScriptRiskPlan_New(
            Product_New(dates, events),
            _our_model(),
            _our_calibration(),
            "equity",
            DupireScriptRiskRequest_(**retrained_request),
        )
    )
    theirs = dal.DupireScriptRiskResult_New(
        dal.DupireScriptRiskPlan_New(
            dal.Product_New([dal.Date_(2027, 3, 12), dal.Date_(2027, 9, 12)], events),
            _their_model(dal),
            _their_calibration(dal),
            "equity",
            dal.DupireScriptRiskRequest_(
                num_paths=64,
                valuation=_their_valuation(dal),
                simulation=dal.MonteCarloSettings_(
                    enable_aad=True, lsmc_training_paths=64, lsmc_policy_risk_mode="RetrainedBump"
                ),
            ),
        )
    )
    assert ours.method == theirs.method


def test_pullback_names_parity(dal):
    valuation = MonteCarlo_ValueWithRisk(
        _our_risk_product(), _our_model(), 257, valuation=VALUATION
    )
    native_valuation = dal.MonteCarlo_ValueWithRisk(
        _their_risk_product(dal), _their_model(dal), 257, valuation=_their_valuation(dal)
    )
    ours_seeds = DupireParameterAdjoints_FromRisk(valuation, _our_calibration(), "equity")
    theirs_seeds = dal.DupireParameterAdjoints_FromRisk(
        native_valuation, _their_calibration(dal), "equity"
    )
    np.testing.assert_allclose(
        [list(row) for row in ours_seeds.adjoints],
        theirs_seeds.adjoints.to_rows(),
        rtol=1e-9,
        atol=1e-12,
    )
    assert ours_seeds.calibration.matches(_our_calibration())
    ours_risk = DupireQuoteRisk_New(_our_calibration(), ours_seeds)
    theirs_risk = dal.DupireQuoteRisk_New(_their_calibration(dal), theirs_seeds)
    assert ours_risk.method == theirs_risk.method == "NativeAADCalibrationVJP"
    assert ours_risk.unit == theirs_risk.unit == "decimal-vol"
    assert ours_risk.boundary == theirs_risk.boundary
    np.testing.assert_allclose(
        [list(row) for row in ours_risk.calibration_adjoints],
        theirs_risk.calibration_adjoints.to_rows(),
        rtol=1e-7,
        atol=1e-9,
    )
    assert list(ours_risk.calibration_adjoints[0]) == pytest.approx(PULLBACK_ROW_ANCHOR, rel=1e-7)
    script = DupireScriptQuoteRisk_New(valuation, _our_calibration(), "equity")
    native_script = dal.DupireScriptQuoteRisk_New(
        native_valuation, _their_calibration(dal), "equity"
    )
    assert script.method == native_script.method == "NativeAADThenNativeAADCalibrationVJP"
    assert script.component == native_script.component == "equity"
    direct = DupireDirectQuoteAdjoints_(
        calibration=_our_calibration(), adjoints=[[0.1] * 2 for _ in range(3)]
    )
    ours_direct = DupireScriptQuoteRisk_New(valuation, _our_calibration(), "equity", direct=direct)
    native_direct = dal.DupireScriptQuoteRisk_New(
        native_valuation,
        _their_calibration(dal),
        "equity",
        direct=dal.DupireDirectQuoteAdjoints_(
            calibration=_their_calibration(dal), adjoints=dal.DoubleMatrix_(3, 2, 0.1)
        ),
    )
    np.testing.assert_allclose(
        [list(row) for row in ours_direct.quote_risk.direct_adjoints],
        native_direct.quote_risk.direct_adjoints.to_rows(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        [list(row) for row in ours_direct.quote_risk.total_adjoints],
        native_direct.quote_risk.total_adjoints.to_rows(),
        rtol=1e-7,
        atol=1e-9,
    )


#  ---------------------------------------------------------------------------
#  Request, plan and pullback validation, both sides.


def _crr(**overrides):
    kwargs = dict(inputs=None, report_factors=None, numeric_payload_budget_bytes=None)
    kwargs.update(overrides)
    return CalibrationRiskRequest_(**kwargs)


def _their_crr(dal, **overrides):
    kwargs = dict(inputs=None, report_factors=None, numeric_payload_budget_bytes=None)
    kwargs.update(overrides)
    for key in ("inputs", "report_factors"):
        if isinstance(kwargs[key], (list, tuple)):
            kwargs[key] = list(kwargs[key])
    return dal.CalibrationRiskRequest_(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "inputs",
            "quote:3",
            "InvalidCalibrationRiskRequest: CalibrationRiskRequest_; inputs; type=str, value='quote:3'; expected list or tuple, or None",
        ),
        (
            "inputs",
            [3],
            "InvalidCalibrationRiskRequest: CalibrationRiskRequest_; inputs; type=int, value=3; expected str or DAL String_ without NUL",
        ),
        (
            "report_factors",
            "x",
            "InvalidCalibrationRiskRequest: CalibrationRiskRequest_; report_factors; type=str, value='x'; expected list or tuple, or None",
        ),
        (
            "report_factors",
            [True],
            "InvalidCalibrationRiskRequest: CalibrationRiskRequest_; report_factors; type=bool, value=True; expected int or float, excluding bool",
        ),
        (
            "numeric_payload_budget_bytes",
            -1,
            "InvalidCalibrationRiskRequest: CalibrationRiskRequest_; numeric_payload_budget_bytes; type=int, value=-1; expected nonnegative size_t integer or None, excluding bool",
        ),
        (
            "numeric_payload_budget_bytes",
            1.5,
            "InvalidCalibrationRiskRequest: CalibrationRiskRequest_; numeric_payload_budget_bytes; type=float, value=1.5; expected nonnegative size_t integer or None, excluding bool",
        ),
    ],
)
def test_calibration_risk_request_validates_like_native(dal, field, value, message):
    _assert_both_raise(
        dal,
        lambda: _their_crr(dal, **{field: value}),
        lambda: _crr(**{field: value}),
        message,
    )


@pytest.mark.parametrize(
    ("quotes_kwargs", "message"),
    [
        (
            {"inputs": ["quote:99"]},
            "InvalidCalibrationRiskRequest: unknown input ID; input=quote:99",
        ),
        (
            {"inputs": ["quote:3", "quote:3"]},
            "InvalidCalibrationRiskRequest: repeated input ID; input=quote:3",
        ),
        (
            {"inputs": ["quote:\x03"]},
            "InvalidCalibrationRiskRequest: unknown input ID; input=quote:\x03",
        ),
        (
            {"inputs": ["quote:3"], "report_factors": [1.0, 2.0]},
            "InvalidCalibrationRiskRequest: report factor count must match selected inputs; field=reportFactors",
        ),
        (
            {"inputs": ["quote:3"], "report_factors": [0.0]},
            "InvalidCalibrationRiskRequest: report factor must be finite and positive; input=quote:3",
        ),
        (
            {"inputs": ["quote:3"], "report_factors": [float("nan")]},
            "InvalidCalibrationRiskRequest: report factor must be finite and positive; input=quote:3",
        ),
    ],
)
def test_plan_validates_quote_selection_like_native(dal, quotes_kwargs, message):
    _assert_both_raise(
        dal,
        lambda: _their_plan(
            dal,
            request=_their_script_request(
                dal,
                quotes=dal.CalibrationRiskRequest_(
                    **{k: (list(v) if v else v) for k, v in quotes_kwargs.items()}
                ),
            ),
        ),
        lambda: _our_plan(
            request=_our_script_request(quotes=CalibrationRiskRequest_(**quotes_kwargs))
        ),
        message,
    )


def test_plan_budget_exceeded_like_native(dal):
    message = (
        "CalibrationRiskBudgetExceeded: combined valuation/quote payload exceeds "
        "numericPayloadBudgetBytes; requiredBytes=296"
    )
    _assert_both_raise(
        dal,
        lambda: _their_plan(
            dal,
            request=_their_script_request(
                dal, quotes=dal.CalibrationRiskRequest_(numeric_payload_budget_bytes=8)
            ),
        ),
        lambda: _our_plan(
            request=_our_script_request(
                quotes=CalibrationRiskRequest_(numeric_payload_budget_bytes=8)
            )
        ),
        message,
    )
    assert _our_plan().numeric_payload_bytes == 296
    bound = _our_plan(
        request=_our_script_request(
            direct_bindings=[DupireQuoteBinding_(constant_ordinal=0, quote_id="quote:3")]
        )
    )
    assert bound.numeric_payload_bytes == 304
    assert _our_plan().quote_plan.numeric_payload_bytes == 144


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"constant_ordinal": -1, "quote_id": "quote:3"},
            "InvalidDupireRiskRequest: DupireQuoteBinding_; constant_ordinal; type=int, value=-1; expected nonnegative size_t integer, excluding bool and enums",
        ),
        (
            {"constant_ordinal": 1.5, "quote_id": "quote:3"},
            "InvalidDupireRiskRequest: DupireQuoteBinding_; constant_ordinal; type=float, value=1.5; expected nonnegative size_t integer, excluding bool and enums",
        ),
        (
            {"constant_ordinal": True, "quote_id": "quote:3"},
            "InvalidDupireRiskRequest: DupireQuoteBinding_; constant_ordinal; type=bool, value=True; expected nonnegative size_t integer, excluding bool and enums",
        ),
        (
            {"constant_ordinal": 0, "quote_id": 3},
            "InvalidDupireRiskRequest: DupireQuoteBinding_; quote_id; type=int, value=3; expected str or DAL String_ without NUL",
        ),
    ],
)
def test_quote_binding_validates_like_native(dal, kwargs, message):
    _assert_both_raise(
        dal,
        lambda: dal.DupireQuoteBinding_(**kwargs),
        lambda: DupireQuoteBinding_(**kwargs),
        message,
    )
    with pytest.raises(TypeError):
        dal.DupireQuoteBinding_(0, "quote:3")
    with pytest.raises(TypeError):
        DupireQuoteBinding_(0, "quote:3")  # pylint: disable=too-many-function-args,missing-kwoa


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "num_paths",
            0,
            "InvalidPathCount: DupireScriptRiskRequest_; num_paths; type=int, value=0; expected integer in 1..2147483647, excluding bool and enums; number of Monte Carlo paths must be positive",
        ),
        (
            "num_paths",
            True,
            "InvalidPathCount: DupireScriptRiskRequest_; num_paths; type=bool, value=True; expected integer in 1..2147483647, excluding bool and enums",
        ),
        (
            "num_paths",
            "257",
            "InvalidPathCount: DupireScriptRiskRequest_; num_paths; type=str, value='257'; expected integer in 1..2147483647, excluding bool and enums",
        ),
        (
            "quotes",
            "REQUEST",
            "InvalidSetting: DupireScriptRiskRequest_; quotes; type=str, value='REQUEST'; expected CalibrationRiskRequest_ or None",
        ),
        (
            "valuation",
            "SETTINGS",
            "InvalidSetting: DupireScriptRiskRequest_; valuation; type=str, value='SETTINGS'; expected ScriptValuationSettings_ or None",
        ),
        (
            "simulation",
            "x",
            "InvalidSetting: DupireScriptRiskRequest_; simulation; type=str, value='x'; expected MonteCarloSettings_ or None",
        ),
        (
            "direct",
            "x",
            "InvalidDupireRiskRequest: DupireScriptRiskRequest_; direct; type=str, value='x'; expected CalibrationDirectQuoteAdjoints_ or None",
        ),
        (
            "direct_bindings",
            "x",
            "InvalidDupireRiskRequest: DupireScriptRiskRequest_; direct_bindings; type=str, value='x'; expected list or tuple, or None",
        ),
        (
            "direct_bindings",
            ["x"],
            "InvalidDupireRiskRequest: DupireScriptRiskRequest_; direct_bindings; type=str, value='x'; expected DupireQuoteBinding_",
        ),
    ],
)
def test_script_risk_request_validates_like_native(dal, field, value, message):
    def theirs():
        kwargs = dict(num_paths=257, valuation=_their_valuation(dal))
        kwargs[field] = value if field != "valuation" else value
        return dal.DupireScriptRiskRequest_(**kwargs)

    def ours():
        kwargs = dict(num_paths=257, valuation=VALUATION)
        kwargs[field] = value
        return DupireScriptRiskRequest_(**kwargs)

    _assert_both_raise(dal, theirs, ours, message)


def test_plan_validates_arguments_like_native(dal):
    cases = [
        (
            "product",
            "x",
            "InvalidDupireRiskRequest: DupireScriptRiskPlan_New; product; type=str, value='x'; expected ScriptProductData_",
        ),
        (
            "modelData",
            "x",
            "InvalidDupireRiskRequest: DupireScriptRiskPlan_New; modelData; type=str, value='x'; expected ModelData_",
        ),
        (
            "calibration",
            "x",
            "InvalidDupireRiskRequest: DupireScriptRiskPlan_New; calibration; type=str, value='x'; expected DupireCalibrationSnapshot_",
        ),
        (
            "component",
            1.5,
            "InvalidDupireRiskRequest: DupireScriptRiskPlan_New; component; type=float, value=1.5; expected str or DAL String_ without NUL",
        ),
        (
            "request",
            "x",
            "InvalidDupireRiskRequest: DupireScriptRiskPlan_New; request; type=str, value='x'; expected DupireScriptRiskRequest_",
        ),
    ]
    for field, value, message in cases:
        fields = dict(
            product=_our_risk_product(),
            modelData=_our_model(),
            calibration=_our_calibration(),
            component="equity",
            request=_our_script_request(),
        )
        native_fields = dict(
            product=_their_risk_product(dal),
            modelData=_their_model(dal),
            calibration=_their_calibration(dal),
            component="equity",
            request=_their_script_request(dal),
        )
        fields[field] = value
        native_fields[field] = value
        _assert_both_raise(
            dal,
            lambda native_fields=native_fields: dal.DupireScriptRiskPlan_New(**native_fields),
            lambda fields=fields: DupireScriptRiskPlan_New(**fields),
            message,
        )
    semantic = [
        (
            dict(modelData=BSModelData_New(100.0, 0.2, 0.05, 0.02)),
            dict(modelData=dal.BSModelData_New(100.0, 0.2, 0.05, 0.02)),
            "InvalidDupireRiskRequest: Hybrid model is required",
        ),
        (
            dict(
                simulation=MonteCarloSettings_(enable_aad=False),
                request=_our_script_request(simulation=MonteCarloSettings_(enable_aad=False)),
            )
            if False
            else dict(
                request=_our_script_request(simulation=MonteCarloSettings_(enable_aad=False))
            ),
            dict(
                request=_their_script_request(
                    dal, simulation=dal.MonteCarloSettings_(enable_aad=False)
                )
            ),
            "InvalidDupireRiskRequest: native AAD execution is required",
        ),
        (
            dict(component="rate"),
            dict(component="rate"),
            "InvalidDupireRiskRequest: named local-vol equity is required; component=rate",
        ),
    ]
    for overrides, native_overrides, message in semantic:
        _assert_both_raise(
            dal,
            lambda native_overrides=native_overrides: _their_plan(dal, **native_overrides),
            lambda overrides=overrides: _our_plan(**overrides),
            message,
        )


def test_plan_validates_carry_surface_and_bindings_like_native(dal):
    bumped = DupireCalibration_New(
        BSModelData_New(101.0, 0.2, 0.05, 0.02), _ours_inputs(), name="frozen"
    )
    native_bumped = dal.DupireCalibration_New(
        dal.BSModelData_New(101.0, 0.2, 0.05, 0.02), _their_inputs(dal), name="frozen"
    )
    carry_message = "DupireSnapshotMismatch: automatic request requires matching flat deterministic carry; component=equity"
    _assert_both_raise(
        dal,
        lambda: _their_plan(dal, calibration=native_bumped),
        lambda: _our_plan(calibration=bumped),
        carry_message,
    )
    regridded = DupireCalibration_New(
        _our_base(), _ours_inputs(inclusion_spots=[60.0, 101.0, 140.0]), name="frozen"
    )
    native_regridded = dal.DupireCalibration_New(
        _their_base(dal),
        _their_inputs(dal, inclusion_spots=[60.0, 101.0, 140.0]),
        name="frozen",
    )
    grids_message = "DupireSnapshotMismatch: component surface grids changed; component=equity"
    _assert_both_raise(
        dal,
        lambda: _their_plan(dal, calibration=native_regridded),
        lambda: _our_plan(calibration=regridded),
        grids_message,
    )
    other = DupireCalibration_New(
        BSModelData_New(100.0, 0.25, 0.05, 0.02), _ours_inputs(), name="frozen"
    )
    native_other = dal.DupireCalibration_New(
        dal.BSModelData_New(100.0, 0.25, 0.05, 0.02), _their_inputs(dal), name="frozen"
    )
    surface_message = "DupireSnapshotMismatch: component surface values changed; component=equity"
    _assert_both_raise(
        dal,
        lambda: _their_plan(dal, calibration=native_other),
        lambda: _our_plan(calibration=other),
        surface_message,
    )
    binding_cases = [
        (
            dict(constant_ordinal=5, quote_id="quote:3"),
            "InvalidDupireRiskRequest: unknown direct constant ordinal=5",
        ),
        (
            dict(constant_ordinal=0, quote_id="quote:99"),
            "InvalidDupireRiskRequest: unknown direct quote ID=quote:99",
        ),
        (
            dict(constant_ordinal=0, quote_id="quote:4"),
            "InvalidDupireRiskRequest: repeated direct constant; input=constant:0",
        ),
    ]
    for extra, message in binding_cases:
        repeated = extra["constant_ordinal"] == 0 and extra["quote_id"] == "quote:4"
        ours = (
            [
                DupireQuoteBinding_(constant_ordinal=0, quote_id="quote:3"),
                DupireQuoteBinding_(constant_ordinal=0, quote_id="quote:4"),
            ]
            if repeated
            else [DupireQuoteBinding_(**extra)]
        )
        native = (
            [
                dal.DupireQuoteBinding_(constant_ordinal=0, quote_id="quote:3"),
                dal.DupireQuoteBinding_(constant_ordinal=0, quote_id="quote:4"),
            ]
            if repeated
            else [dal.DupireQuoteBinding_(**extra)]
        )
        _assert_both_raise(
            dal,
            lambda native=native: _their_plan(
                dal, request=_their_script_request(dal, direct_bindings=native)
            ),
            lambda ours=ours: _our_plan(request=_our_script_request(direct_bindings=ours)),
            message,
        )
    #  A model built from a bumped-spread calibration surfaces the value check.
    spreads = [[0.0, 0.0], [0.0, 0.5], [0.0, 0.0]]
    bumped_model = DupireModelData_New(
        _bumped_calibration(spreads), INDEX, "USD", "F_LOCAL", max_step=0.25
    )
    native_bumped_cal = dal.DupireCalibration_New(
        _their_base(dal), _their_inputs(dal, quote_spreads=spreads), name="frozen"
    )
    native_bumped_model = dal.DupireModelData_New(
        native_bumped_cal, INDEX, "USD", "F_LOCAL", max_step=0.25
    )
    value_message = (
        "InvalidDupireRiskRequest: direct constant/quote values disagree; "
        "input=constant:0; quote=quote:3"
    )
    _assert_both_raise(
        dal,
        lambda: dal.DupireScriptRiskPlan_New(
            _their_risk_product(dal),
            native_bumped_model,
            native_bumped_cal,
            "equity",
            _their_script_request(
                dal,
                direct_bindings=[dal.DupireQuoteBinding_(constant_ordinal=0, quote_id="quote:3")],
            ),
        ),
        lambda: DupireScriptRiskPlan_New(
            _our_risk_product(),
            bumped_model,
            _bumped_calibration(spreads),
            "equity",
            _our_script_request(
                direct_bindings=[DupireQuoteBinding_(constant_ordinal=0, quote_id="quote:3")]
            ),
        ),
        value_message,
    )


def _bumped_calibration(spreads):
    return DupireCalibration_New(_our_base(), _ours_inputs(quote_spreads=spreads), name="frozen")


def test_result_and_pullback_validate_like_native(dal):
    valuation = MonteCarlo_ValueWithRisk(
        _our_risk_product(), _our_model(), 257, valuation=VALUATION
    )
    native_valuation = dal.MonteCarlo_ValueWithRisk(
        _their_risk_product(dal), _their_model(dal), 257, valuation=_their_valuation(dal)
    )
    other = DupireCalibration_New(
        BSModelData_New(100.0, 0.25, 0.05, 0.02), _ours_inputs(), name="frozen"
    )
    native_other = dal.DupireCalibration_New(
        dal.BSModelData_New(100.0, 0.25, 0.05, 0.02), _their_inputs(dal), name="frozen"
    )
    cases = [
        (
            lambda: dal.DupireScriptRiskResult_New("x"),
            lambda: DupireScriptRiskResult_New("x"),
            "InvalidDupireRiskRequest: DupireScriptRiskResult_New; plan; type=str, value='x'; expected DupireScriptRiskPlan_",
        ),
        (
            lambda: dal.DupireParameterAdjoints_FromRisk("x", _their_calibration(dal), "equity"),
            lambda: DupireParameterAdjoints_FromRisk("x", _our_calibration(), "equity"),
            "InvalidDupirePullback: DupireParameterAdjoints_FromRisk; valuation; type=str, value='x'; expected RiskResult_",
        ),
        (
            lambda: dal.DupireParameterAdjoints_FromRisk(native_valuation, "x", "equity"),
            lambda: DupireParameterAdjoints_FromRisk(valuation, "x", "equity"),
            "InvalidDupirePullback: DupireParameterAdjoints_FromRisk; calibration; type=str, value='x'; expected DupireCalibrationSnapshot_",
        ),
        (
            lambda: dal.DupireParameterAdjoints_FromRisk(
                native_valuation, _their_calibration(dal), "rate"
            ),
            lambda: DupireParameterAdjoints_FromRisk(valuation, _our_calibration(), "rate"),
            "InvalidDupirePullback: component must be local-vol equity; component=rate",
        ),
        (
            lambda: dal.DupireParameterAdjoints_FromRisk(native_valuation, native_other, "equity"),
            lambda: DupireParameterAdjoints_FromRisk(valuation, other, "equity"),
            "DupireSnapshotMismatch: component surface values changed; component=equity",
        ),
        (
            lambda: dal.DupireParameterAdjoints_FromRisk(
                dal.MonteCarlo_ValueWithRisk(
                    _their_risk_product(dal),
                    _their_model(dal),
                    257,
                    valuation=_their_valuation(dal),
                    request=dal.RiskRequest_(inputs=["model:2"]),
                ),
                _their_calibration(dal),
                "equity",
            ),
            lambda: DupireParameterAdjoints_FromRisk(
                MonteCarlo_ValueWithRisk(
                    _our_risk_product(),
                    _our_model(),
                    257,
                    valuation=VALUATION,
                    request=RiskRequest_(inputs=["model:2"]),
                ),
                _our_calibration(),
                "equity",
            ),
            "InvalidDupirePullback: missing selected surface risk; input=model:3",
        ),
        (
            lambda: dal.DupireParameterAdjoints_(calibration="x", adjoints=dal.DoubleMatrix_(9, 2)),
            lambda: DupireParameterAdjoints_(calibration="x", adjoints=[[1.0] * 2] * 9),
            "InvalidDupirePullback: DupireAdjoints; calibration; type=str, value='x'; expected DupireCalibrationSnapshot_",
        ),
        (
            lambda: dal.DupireParameterAdjoints_(calibration="x", adjoints=dal.DoubleMatrix_(9, 2)),
            lambda: DupireParameterAdjoints_(calibration="x", adjoints=[[1.0] * 2] * 9),
            "InvalidDupirePullback: DupireAdjoints; calibration; type=str, value='x'; expected DupireCalibrationSnapshot_",
        ),
    ]
    for their_call, our_call, message in cases:
        _assert_both_raise(dal, their_call, our_call, message)
    #  The adjoints container itself diverges by design: native insists on
    #  DoubleMatrix_, ours accepts nested numeric rows (as quote_spreads does).
    seeds = DupireParameterAdjoints_FromRisk(valuation, _our_calibration(), "equity")
    native_seeds = dal.DupireParameterAdjoints_FromRisk(
        native_valuation, _their_calibration(dal), "equity"
    )
    pullback_cases = [
        (
            lambda: dal.DupireQuoteRisk_New(
                _their_calibration(dal),
                dal.DupireParameterAdjoints_(
                    calibration=_their_calibration(dal), adjoints=dal.DoubleMatrix_(3, 2, 1.0)
                ),
            ),
            lambda: DupireQuoteRisk_New(
                _our_calibration(),
                DupireParameterAdjoints_(calibration=_our_calibration(), adjoints=[[1.0] * 2] * 3),
            ),
            "InvalidDupirePullback: seed dimensions disagree; field=parameters",
        ),
        (
            lambda: dal.DupireQuoteRisk_New(
                _their_calibration(dal),
                dal.DupireParameterAdjoints_(
                    calibration=_their_calibration(dal),
                    adjoints=dal.DoubleMatrix_(9, 2, float("nan")),
                ),
            ),
            lambda: DupireQuoteRisk_New(
                _our_calibration(),
                DupireParameterAdjoints_(
                    calibration=_our_calibration(), adjoints=[[float("nan")] * 2] * 9
                ),
            ),
            "InvalidDupirePullback: non-finite seed; field=parameters",
        ),
        (
            lambda: dal.DupireQuoteRisk_New(
                _their_calibration(dal),
                native_seeds,
                direct=dal.DupireDirectQuoteAdjoints_(
                    calibration=_their_calibration(dal), adjoints=dal.DoubleMatrix_(2, 2, 0.1)
                ),
            ),
            lambda: DupireQuoteRisk_New(
                _our_calibration(),
                seeds,
                direct=DupireDirectQuoteAdjoints_(
                    calibration=_our_calibration(), adjoints=[[0.1] * 2] * 2
                ),
            ),
            "InvalidDupirePullback: seed dimensions disagree; field=directQuotes",
        ),
        (
            lambda: dal.DupireQuoteRisk_New(native_other, native_seeds),
            lambda: DupireQuoteRisk_New(other, seeds),
            "DupireSnapshotMismatch: parameter seed calibration does not match",
        ),
        (
            lambda: dal.DupireScriptQuoteRisk_New(
                dal.MonteCarlo_ValueWithSettings(
                    _their_risk_product(dal),
                    _their_model(dal),
                    257,
                    valuation=_their_valuation(dal),
                ),
                _their_calibration(dal),
                "equity",
            ),
            lambda: DupireScriptQuoteRisk_New(
                MonteCarlo_ValueWithSettings(
                    _our_risk_product(), _our_model(), 257, valuation=VALUATION
                ),
                _our_calibration(),
                "equity",
            ),
            "InvalidDupirePullback: DupireScriptQuoteRisk_New; valuation; type=dict; expected RiskResult_",
        ),
    ]
    for their_call, our_call, message in pullback_cases:
        _assert_both_raise(dal, their_call, our_call, message)


#  ---------------------------------------------------------------------------
#  Quote-space curvature: bump the flattened spreads, recalibrate, revalue.

from dal_jax.api import (  # noqa: E402
    BumpOverAADRequest_,
    DupireScriptCurvaturePlan_New,
    DupireScriptCurvatureRequest_,
    DupireScriptCurvatureResult_New,
)
from dal_jax.errors import InvalidCurvatureRequest  # noqa: E402

#  Native anchors (dal-python 2026.11.1) at 257 paths, directions {quote:0,
#  quote:3} with steps 0.01.  Parity tolerance derivation: the hessian is the
#  central quotient (g+ - g-)/(2h) of the total quote gradients, and our
#  analytic VJP agrees with native's AAD replay to ~2e-9 relative on gradient
#  magnitudes up to ~20, i.e. ~4e-8 absolute per evaluation.  The quotient
#  amplifies that difference by 1/(2*0.01) = 50, bounding the disagreement by
#  ~4e-6 absolute; measured worst cells are ~6e-7 absolute and ~4e-8 relative
#  (native's own h-dependence at halved steps is ~1.4 ABSOLUTE, so parity only
#  holds at equal steps).  The tolerance below carries >10x headroom.
CURVATURE_HESSIAN_ANCHOR = [
    [
        -19.197883317247033,
        10.90832638728898,
        -13.619165022464585,
        9.914050975203281,
        3.2441534975077957,
        7.087305648019537,
    ],
    [
        9.660772961797193,
        1.8474208627594635,
        12.588976890401682,
        -50.91892433956673,
        49.03924013633514,
        31.803567546012346,
    ],
]
CURVATURE_TOLERANCE = dict(rtol=1e-6, atol=1e-5)


def _our_bumps(rows=None, steps=None, **kwargs):
    rows = (
        rows
        if rows is not None
        else [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    )
    return BumpOverAADRequest_(directions=rows, steps=steps or [0.01] * len(rows), **kwargs)


def _their_bumps(dal, rows=None, steps=None, **kwargs):
    rows = (
        rows
        if rows is not None
        else [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    )
    return dal.BumpOverAADRequest_(
        directions=dal.DoubleMatrix_(rows), steps=steps or [0.01] * len(rows), **kwargs
    )


def _our_curvature_request(**overrides):
    kwargs = dict(risk=_our_script_request(), bumps=_our_bumps())
    kwargs.update(overrides)
    return DupireScriptCurvatureRequest_(**kwargs)


def _their_curvature_request(dal, **overrides):
    kwargs = dict(risk=_their_script_request(dal), bumps=_their_bumps(dal))
    kwargs.update(overrides)
    return dal.DupireScriptCurvatureRequest_(**kwargs)


def _our_curvature_plan(**overrides):
    kwargs = dict(
        product=_our_risk_product(),
        modelData=_our_model(),
        calibration=_our_calibration(),
        component="equity",
        request=_our_curvature_request(),
    )
    kwargs.update(overrides)
    return DupireScriptCurvaturePlan_New(**kwargs)


def _their_curvature_plan(dal, **overrides):
    kwargs = dict(
        product=_their_risk_product(dal),
        modelData=_their_model(dal),
        calibration=_their_calibration(dal),
        component="equity",
        request=_their_curvature_request(dal),
    )
    kwargs.update(overrides)
    return dal.DupireScriptCurvaturePlan_New(**kwargs)


def test_curvature_chain_parity(dal):
    ours_plan = _our_curvature_plan()
    theirs_plan = _their_curvature_plan(dal)
    assert list(ours_plan.point) == theirs_plan.point == [0.0] * 6
    assert [list(row) for row in ours_plan.directions] == theirs_plan.directions.to_rows()
    assert list(ours_plan.steps) == theirs_plan.steps == [0.01, 0.01]
    assert ours_plan.numeric_payload_bytes == theirs_plan.numeric_payload_bytes == 608
    assert ours_plan.base_plan.component == theirs_plan.base_plan.component == "equity"
    assert ours_plan.base_plan.num_paths == theirs_plan.base_plan.num_paths == 257
    ours = DupireScriptCurvatureResult_New(ours_plan)
    theirs = dal.DupireScriptCurvatureResult_New(theirs_plan)
    assert (
        ours.execution.method
        == theirs.execution.method
        == "BumpOverRecalibratedNativeDupireMonteCarloAAD"
    )
    assert (
        ours.execution.quote_gradient_evaluations
        == theirs.execution.quote_gradient_evaluations
        == 5
    )
    assert ours.execution.paths_per_evaluation == theirs.execution.paths_per_evaluation == 257
    assert ours.execution.numeric_payload_bytes == theirs.execution.numeric_payload_bytes == 608
    assert ours.base.method == theirs.base.method
    assert ours.base.valuation.values == pytest.approx(theirs.base.valuation.values, rel=1e-12)
    np.testing.assert_allclose(list(ours.gradient), theirs.gradient, rtol=1e-7, atol=1e-8)
    assert [c.id for c in ours.input_axis] == [c.id for c in theirs.input_axis]
    np.testing.assert_allclose(
        [list(row) for row in ours.hessian_products],
        theirs.hessian_products.to_rows(),
        **CURVATURE_TOLERANCE,
    )
    np.testing.assert_allclose(
        [list(row) for row in ours.hessian_products],
        CURVATURE_HESSIAN_ANCHOR,
        **CURVATURE_TOLERANCE,
    )


def test_curvature_detaches_directions(dal):
    rows = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    ours_bumps = _our_bumps(rows=rows)
    theirs_bumps = _their_bumps(dal, rows=rows)
    rows[0][1] = 5.0
    assert [list(row) for row in tuple(ours_bumps.directions)] == [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ]
    assert theirs_bumps.directions.to_rows()[0][1] == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "steps",
            [0.01, 0.02],
            "InvalidCurvatureRequest: BumpOverAADRequest_; steps requires one value per direction row",
        ),
        (
            "steps",
            [0.0],
            "InvalidCurvatureRequest: BumpOverAADRequest_; steps must be finite and positive",
        ),
        (
            "steps",
            [float("nan")],
            "InvalidCurvatureRequest: BumpOverAADRequest_; steps must be finite and positive",
        ),
        (
            "steps",
            ["0.01"],
            "InvalidCurvatureRequest: BumpOverAADRequest_; steps; type=str, value='0.01'; expected int or float, excluding bool",
        ),
        (
            "steps",
            "x",
            "InvalidCurvatureRequest: BumpOverAADRequest_; steps; type=str, value='x'; expected list or tuple, or None",
        ),
    ],
)
def test_bump_request_steps_validate_like_native(dal, field, value, message):
    _assert_both_raise(
        dal,
        lambda: _their_curvature_request(
            dal, bumps=_their_bumps(dal, rows=[[1.0] * 6], steps=value)
        ),
        lambda: _our_curvature_request(
            bumps=BumpOverAADRequest_(directions=[[1.0] * 6], steps=value)
        ),
        message,
    )


def test_bump_request_directions_validate_like_native(dal):
    nan_rows = [[float("nan")] + [0.0] * 5]
    _assert_both_raise(
        dal,
        lambda: _their_curvature_request(dal, bumps=_their_bumps(dal, rows=nan_rows)),
        lambda: _our_curvature_request(bumps=_our_bumps(rows=nan_rows)),
        "InvalidCurvatureRequest: BumpOverAADRequest_; directions must be finite",
    )
    #  Directions with a bad entry type surface native's DoubleMatrix binding
    #  error; ours accepts nested numeric rows and flags the entry itself.
    with pytest.raises(InvalidCurvatureRequest, match="expected int or float, excluding bool"):
        _our_curvature_request(bumps=_our_bumps(rows=[[True] + [0.0] * 5]))


def test_curvature_request_validates_types_like_native(dal):
    _assert_both_raise(
        dal,
        lambda: dal.DupireScriptCurvatureRequest_(risk="x", bumps=_their_bumps(dal)),
        lambda: DupireScriptCurvatureRequest_(risk="x", bumps=_our_bumps()),
        "InvalidCurvatureRequest: DupireScriptCurvatureRequest_; risk; type=str, value='x'; expected DupireScriptRiskRequest_",
    )
    _assert_both_raise(
        dal,
        lambda: dal.DupireScriptCurvatureRequest_(risk=_their_script_request(dal), bumps="x"),
        lambda: DupireScriptCurvatureRequest_(risk=_our_script_request(), bumps="x"),
        "InvalidCurvatureRequest: DupireScriptCurvatureRequest_; bumps; type=str, value='x'; expected BumpOverAADRequest_",
    )


def test_curvature_plan_validates_like_native(dal):
    cases = [
        (
            dict(product="x"),
            dict(product="x"),
            "InvalidCurvatureRequest: DupireScriptCurvaturePlan_New; product; type=str, value='x'; expected ScriptProductData_",
        ),
        (
            dict(component=1.5),
            dict(component=1.5),
            "InvalidCurvatureRequest: DupireScriptCurvaturePlan_New; component; type=float, value=1.5; expected str or DAL String_ without NUL",
        ),
        (
            dict(request="x"),
            dict(request="x"),
            "InvalidCurvatureRequest: DupireScriptCurvaturePlan_New; request; type=str, value='x'; expected DupireScriptCurvatureRequest_",
        ),
    ]
    for overrides, native_overrides, message in cases:
        _assert_both_raise(
            dal,
            lambda native_overrides=native_overrides: _their_curvature_plan(
                dal, **native_overrides
            ),
            lambda overrides=overrides: _our_curvature_plan(**overrides),
            message,
        )
    semantic = [
        #  One column per flattened quote entry: seven columns do not fit.
        (
            dict(request=_our_curvature_request(bumps=_our_bumps(rows=[[1.0] * 7]))),
            dict(request=_their_curvature_request(dal, bumps=_their_bumps(dal, rows=[[1.0] * 7]))),
            "BumpOverAAD: directions must have nonnegative rows and one column per input",
        ),
        (
            dict(request=_our_curvature_request(bumps=_our_bumps(rows=[[0.0] * 6]))),
            dict(request=_their_curvature_request(dal, bumps=_their_bumps(dal, rows=[[0.0] * 6]))),
            "BumpOverAAD: direction=0; direction must be nonzero",
        ),
        (
            dict(
                request=_our_curvature_request(
                    bumps=_our_bumps(rows=[[0.0, 0.0, 0.0, -1.0, 0.0, 0.0]], steps=[0.3])
                )
            ),
            dict(
                request=_their_curvature_request(
                    dal,
                    bumps=_their_bumps(dal, rows=[[0.0, 0.0, 0.0, -1.0, 0.0, 0.0]], steps=[0.3]),
                )
            ),
            "DupireScriptCurvature: admission; direction=0; plus; InvalidDupireCalibration: local variance is nonpositive or non-finite",
        ),
        (
            dict(
                request=_our_curvature_request(
                    bumps=BumpOverAADRequest_(
                        directions=[[1.0] * 6], steps=[0.01], recording_capacity_budget_bytes=8
                    )
                )
            ),
            dict(
                request=_their_curvature_request(
                    dal,
                    bumps=dal.BumpOverAADRequest_(
                        directions=dal.DoubleMatrix_(1, 6, 1.0),
                        steps=[0.01],
                        recording_capacity_budget_bytes=8,
                    ),
                )
            ),
            "DupireScriptCurvature: worker recording capacity limits are unsupported",
        ),
        (
            dict(
                request=_our_curvature_request(
                    bumps=BumpOverAADRequest_(
                        directions=[[1.0] * 6], steps=[0.01], numeric_payload_budget_bytes=8
                    )
                )
            ),
            dict(
                request=_their_curvature_request(
                    dal,
                    bumps=dal.BumpOverAADRequest_(
                        directions=dal.DoubleMatrix_(1, 6, 1.0),
                        steps=[0.01],
                        numeric_payload_budget_bytes=8,
                    ),
                )
            ),
            "BumpOverAAD: numeric payload budget exceeded",
        ),
    ]
    for overrides, native_overrides, message in semantic:
        _assert_both_raise(
            dal,
            lambda native_overrides=native_overrides: _their_curvature_plan(
                dal, **native_overrides
            ),
            lambda overrides=overrides: _our_curvature_plan(**overrides),
            message,
        )


def test_curvature_rejects_exercise_products_like_native(dal):
    events = [f"pay PAYS 1 EXERCISE MAX(100 - FIX({INDEX}), 0)"] * 2
    message = (
        "DupireScriptCurvature: rebuild; base; RecalibrateDupireScriptRisk: "
        "exercise-policy curvature is unsupported"
    )
    _assert_both_raise(
        dal,
        lambda: _their_curvature_plan(
            dal,
            product=dal.Product_New([dal.Date_(2027, 3, 12), dal.Date_(2027, 9, 12)], events),
        ),
        lambda: _our_curvature_plan(
            product=Product_New([Date.ymd(2027, 3, 12), Date.ymd(2027, 9, 12)], events)
        ),
        message,
    )
    _assert_both_raise(
        dal,
        lambda: dal.DupireScriptCurvatureResult_New("x"),
        lambda: DupireScriptCurvatureResult_New("x"),
        "InvalidCurvatureRequest: DupireScriptCurvatureResult_New; plan; type=str, value='x'; expected DupireScriptCurvaturePlan_",
    )


def test_curvature_with_direct_bindings_parity(dal):
    ours = DupireScriptCurvatureResult_New(
        _our_curvature_plan(
            request=_our_curvature_request(
                risk=_our_script_request(
                    direct_bindings=[DupireQuoteBinding_(constant_ordinal=0, quote_id="quote:3")]
                )
            )
        )
    )
    theirs = dal.DupireScriptCurvatureResult_New(
        _their_curvature_plan(
            dal,
            request=_their_curvature_request(
                dal,
                risk=_their_script_request(
                    dal,
                    direct_bindings=[
                        dal.DupireQuoteBinding_(constant_ordinal=0, quote_id="quote:3")
                    ],
                ),
            ),
        )
    )
    assert ours.execution.method == theirs.execution.method
    np.testing.assert_allclose(list(ours.gradient), theirs.gradient, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(
        [list(row) for row in ours.hessian_products],
        theirs.hessian_products.to_rows(),
        **CURVATURE_TOLERANCE,
    )
