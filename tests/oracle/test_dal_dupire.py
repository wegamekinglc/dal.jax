"""Dupire risk inputs and calibration snapshots parity with dal-python 2026.11.1."""

import re

import pytest

from dal_jax.api import (
    IVS_,
    BSModelData_New,
    CorrelatedBSModelData_New,
    DupireCalibration_New,
    DupireRiskInputs_,
    MertonIVS_,
)
from dal_jax.errors import (
    InvalidDupireCalibration,
    InvalidDupirePullback,
    InvalidDupireQuote,
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
