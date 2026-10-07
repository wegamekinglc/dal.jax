"""Script front end against dal-python: byte-identical dumps and matching errors (issue #1, P1).

dal-python 2026.9.25 predates vectors, ``FOR``, ``PAYS ... ON``, the IR index
parser and the ``30U/360`` basis, so the corpus sticks to features it has; the
newer ones are pinned by the ported DAL unit tests.  ``Product_Describe`` of the
newer DAL also adds ``regression_features``, which is dropped before comparing.
DAL error messages carry C++ source locations, so errors compare on the
exception code and the core message.
"""

import pytest

from dal_jax.api import EvaluationDate_Get, EvaluationDate_Set, Product_Debug, Product_DebugJson, Product_DebugTree, Product_Describe, Product_New
from dal_jax.dates import Date
from dal_jax.errors import DalError

pytestmark = pytest.mark.oracle
EVALUATION = (2022, 9, 15)


def D(y, m, d):
    return ("date", y, m, d)


CORPUS = {
    "european": (["STRIKE", D(2025, 9, 14)], ["120.0", "call pays MAX(spot() - STRIKE, 0.0)"]),
    "barrier": (["STRIKE", "BARRIER", D(2022, 9, 15), "START: 2022-09-15 END: 2025-09-14 FREQ: 1M", D(2025, 9, 14)],
                ["120.00", "150.00", "alive = 1", "if spot() >= BARRIER:0.1 then alive = 0 end",
                 "if spot() >= BARRIER:0.1 then alive = 0 end\ncall pays alive * MAX(spot() - STRIKE, 0.0)"]),
    "past_and_future": ([D(2021, 1, 4), D(2022, 9, 15), D(2023, 3, 1)], ["x = 1", "y = x + 2", "z pays y * 2"]),
    "macros": (["PAYOFF", "K", D(2024, 1, 2)], ["MAX(spot() - K, 0)", "100", "c pays PAYOFF + payoff"]),
    "arith": ([D(2024, 1, 2)], ["x = -(1 + 2) * 3 ^ 2 ^ 0.5 / 4 - +5\ny = LOG(EXP(x)) + SQRT(2) - MIN(x, 1, 2) + MAX(1, 2)\np pays -x"]),
    "conditions": ([D(2024, 1, 2)], ["if spot() > 100 and spot() < 200 or spot() = 150 then a = 1 else a = 2 end\n"
                                      "if spot() != 1;0.5 then b = 1 end\nif (spot() >= 1 and spot() <= 2) then c = 1 end\np pays a + b + c"]),
    "nested_if": ([D(2024, 1, 2)], ["if spot() > 1 then if spot() > 2 then x = 1 else x = 2 end y = 3 else x = 4 end p pays x + y"]),
    "dcf": (["START: 2023-01-02 END: 2023-07-02 FREQ: 3M"], ["acc = DCF(ACT/365F, PeriodBegin, PeriodEnd)\ncpn pays acc * 0.05"]),
    "dcf_bases": ([D(2024, 1, 2)], ["a = DCF(ACT/360, 2023-01-31, 2023-03-31) + DCF(30/360, 2023-01-31, 2023-03-31) + "
                                     "DCF(ACT/ACT, 2023-06-01, 2024-06-01)\np pays a"]),
    "schedule_fixing_begin": (["START: 2023-01-02 END: 2023-07-02 FREQ: 2M FIXING: BEGIN"], ["p pays spot()"]),
    "schedule_calendar": (["START: 2022-05-07 END: 2023-05-07 FREQ: 1M CALENDAR: CN.SSE BizRule: ModifiedFollowing"], ["p pays spot()"]),
    "same_date": ([D(2024, 1, 2), D(2024, 1, 2), D(2024, 1, 2)], ["x = 1", "y = 2", "p pays x + y"]),
    "case_insensitive": (["Strike", D(2024, 1, 2)], ["100", "If SPOT() > strike Then X = 1 Else x = 2 END\nP PAYS x"]),
    "fix": ([D(2025, 9, 14)], ["x = FIX(EQ[Aapl], 2025-09-01) + FIX(FX[eur/usd]) + FIX(EQ[x]@2026-12-31) + FIX(EQ[y]>3M)\np pays x"]),
    "multiline_sources": ([D(2024, 1, 2), D(2024, 1, 2)], ["a = 1\n  b = 2", "if a > b then\n c = 1\nend\np pays c"]),
    "long_tree": ([D(2024, 1, 2)], ["p pays MAX(spot() - 100, 0) * MAX(spot() - 110, 0) + MAX(spot() - 120, 0) * MAX(spot() - 130, 0) + "
                                     "MAX(spot() - 140, 0) * MAX(spot() - 150, 0) + MAX(spot() - 160, 0) * LOG(SQRT(EXP(spot()))) / 2"]),
    "long_if": ([D(2024, 1, 2)], ["if spot() > 100 and spot() < 200 and spot() > 120 and spot() < 180 and spot() > 130 and spot() < 170 "
                                   "and spot() > 140 then alive = alive * 1 + 0 else alive = 0 end p pays alive"]),
    "special_numbers": ([D(2024, 1, 2)], ["x = 0.1 + 1e5 + 123456789.123 + .5 + 5. + 0.000001234 + 1E7 + 0x10\np pays x"]),
    "neg_and_pow_parens": ([D(2024, 1, 2)], ["x = -(-spot()) + (-spot()) ^ 2 + 2 ^ (3 ^ 2) + (2 ^ 3) ^ 2 - (1 - (2 - 3))\np pays x"]),
    "empty_else": ([D(2024, 1, 2)], ["if spot() > 1 then x = 1 else end p pays x"]),
}

ERROR_CASES = [
    "x = ", "x", "x = 1 +", "x = (1 + 2", "x = 1 + 2)", "if spot() > 1 x = 1 end", "if spot() > 1 then x = 1", "if spot() then x = 1 end",
    "x = LOG(1, 2)", "x = SPOT", "x = 1 y", "1 = 2", "if = 2", "FIX = 1", "x = FIX(1)", "x = FIX", "x = FIX(EQ[a], 2024-1-1)",
    "x = FIX(EQ[a], 2024 -01-01)", "x = FIX(EQ[a], 2023-13-01)", "x = FIX(QQ[a])", "x = FIX(EQ[a]", "if spot() > 1:0 then x = 1 end",
    "if spot() > 1:-2 then x = 1 end", "if spot() > 1:abc then x = 1 end", "x = 1 $ 2", "x = 1.5e-3", "x = DCF(ACT/365F, 2023-01-01)",
    "x = DCF(XYZ, 2023-01-01, 2024-01-01)", "x = DCF(ACT/365L, 2023-01-01, 2024-01-01)", "x = DCF ACT", "x = MAX(1 2)", "x pays",
    "x = EQ[a", "x = FIX(EQ[x]@2026-01-01)", "x = FIX(EQ[a], 2025-01-01)", "x = FIX(FX[EUR/XYZ])", "x = FIX(EQ[IBM]>2147483648M, 2024-01-01)",
]
DEFINITION_ERRORS = [
    (["FIX"], ["1"]), (["EXERCISE"], ["1"]), (["K", "K"], ["1", "2"]), ([D(2024, 1, 2), "K"], ["x = 1", "2"]),
    (["START: 2023-01-01 END: 2023-06-01 FREQ: 1M BOGUS: 1"], ["x = 1"]), (["START: 2023-01-01 END: 2023-06-01 FREQ: 1Q"], ["x = 1"]),
    (["START: 2023-01-01 END: 2023-06-01 FREQ: 1M FIXING: MIDDLE"], ["x = 1"]), (["START: 2023-01-01 END: 2023-06-01 FREQ: 1M CALENDAR: XX"], ["x = 1"]),
]


@pytest.fixture(scope="module")
def dal_evaluation(dal):
    dal.EvaluationDate_Set(dal.Date_(*EVALUATION))
    previous = EvaluationDate_Get()
    EvaluationDate_Set(Date.ymd(*EVALUATION))
    yield dal
    EvaluationDate_Set(previous)


def _cells(dates, dal=None):
    def convert(cell):
        if isinstance(cell, tuple):
            return dal.Date_(*cell[1:]) if dal else Date.ymd(*cell[1:])
        return cell

    return [convert(c) for c in dates]


def _without_regression_features(described: dict) -> dict:
    return {k: v for k, v in described.items() if k != "regression_features"}


@pytest.mark.parametrize("name", list(CORPUS))
def test_dumps_are_byte_identical(dal_evaluation, name):
    dal = dal_evaluation
    dates, events = CORPUS[name]
    theirs = dal.Product_New(_cells(dates, dal), events)
    ours = Product_New(_cells(dates), events)
    assert _without_regression_features(Product_Describe(ours)) == dal.Product_Describe(theirs)
    if not any("FIX(" in text for text in events):
        assert Product_DebugJson(ours) == dal.Product_DebugJson(theirs)
    assert Product_DebugTree(ours) == dal.Product_DebugTree(theirs)
    assert Product_DebugTree(ours, True, 80) == dal.Product_DebugTree(theirs, True, 80)
    assert Product_Debug(ours) == dal.Product_Debug(theirs)


def _core(message: str) -> str:
    """The DAL message without C++ source locations and NOTICE context lines."""
    lines = [line for line in str(message).splitlines()
             if line.strip() and not line.startswith("/project") and not line.startswith(("name = ", "src = ", "center = ", "Reading date"))]
    return lines[-1] if lines else str(message)


def _compare_errors(dal, dates, events):
    try:
        dal.Product_Describe(dal.Product_New(_cells(dates, dal), events))
        theirs = None
    except Exception as error:  # noqa: BLE001 - DAL raises several exception types
        theirs = _core(str(error))
    try:
        Product_Describe(Product_New(_cells(dates), events))
        ours = None
    except DalError as error:
        ours = str(error)
    assert (ours is None) == (theirs is None), (ours, theirs)
    if theirs is not None:
        detail = theirs.split(": ", 1)[-1] if theirs.startswith(("InvalidIndex: ", "UnknownIndex: ")) else theirs
        assert detail.split("; line=")[0] in ours or theirs in ours, (ours, theirs)


@pytest.mark.parametrize("script", ERROR_CASES)
def test_script_errors_match(dal_evaluation, script):
    _compare_errors(dal_evaluation, [D(2024, 1, 2)], [script])


@pytest.mark.parametrize("dates,events", DEFINITION_ERRORS)
def test_definition_errors_match(dal_evaluation, dates, events):
    _compare_errors(dal_evaluation, dates, events)


def test_describe_rejects_look_ahead_fixings(dal_evaluation):
    product = Product_New([Date.ymd(2025, 9, 14)], ["x = FIX(EQ[a], 2025-09-20)\np pays x"])
    with pytest.raises(DalError, match="LookAheadObservation: expected fixing <= event; index=EQ\\[a\\]; fixing=2025-09-20 00:00:00"):
        Product_Describe(product)
