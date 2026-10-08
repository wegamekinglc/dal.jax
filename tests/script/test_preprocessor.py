"""Port of DAL's ``tests/script/test_preprocessor.cpp`` plus schedule parsing."""

import re

import pytest

from dal_jax.dates import Date
from dal_jax.errors import InvalidVectorDefinition, ReservedIdentifier, ScriptError
from dal_jax.script.lexer import tokenize
from dal_jax.script.preprocessor import Preprocessor, parse_schedule

D = Date.ymd(2023, 12, 1)


def process(rows):
    return Preprocessor().process(rows)


def test_const_variable_definition():
    result = process([("STRIKE", "110.0")])
    assert dict(result.const_variables) == {"STRIKE": 110.0} and not result.events


def test_numeric_vector_definition():
    result = process(
        [("STRIKES", "[100, 120, 150]"), (Date.ymd(2026, 10, 1), "pay PAYS STRIKES[1]")]
    )
    assert result.numeric_vectors["STRIKES"] == (100.0, 120.0, 150.0)
    assert result.events[Date.ymd(2026, 10, 1)] == "pay PAYS STRIKES[1]"
    assert process([("E", "[]")]).numeric_vectors["E"] == ()
    for bad in ("[1, x]", "[1,,2]", "[inf]"):
        with pytest.raises(InvalidVectorDefinition):
            process([("V", bad)])


@pytest.mark.parametrize("keyword", ["FOR", "APPEND", "SUM", "AVERAGE", "FIX", "EXERCISE", "fix"])
def test_reserved_definition_names(keyword):
    with pytest.raises(ReservedIdentifier):
        process([(keyword, "1")])


def test_macro_expansion_into_dated_event():
    result = process([("PAYOFF", "MAX(spot() - 100.0, 0.0)"), (D, "call PAYS PAYOFF")])
    assert not result.const_variables
    statement = result.events[D]
    assert "MAX(spot() - 100.0, 0.0)" in statement and "PAYOFF" not in statement


def test_const_variable_is_not_expanded():
    result = process([("STRIKE", "110.0"), (D, "x = STRIKE")])
    assert result.const_variables["STRIKE"] == 110.0 and "STRIKE" in result.events[D]


def test_events_on_the_same_date_are_joined_with_origins():
    result = process([(D, "x = 1"), (D, "y = 2")])
    assert result.events[D] == "x = 1\ny = 2"
    assert [(o.offset, o.row) for o in result.sources[D]] == [(0, 1), (6, 2)]


def test_definitions_must_come_first_and_be_unique():
    with pytest.raises(ScriptError, match="macros should always at the front"):
        process([(D, "call PAYS MAX(spot() - STRIKE, 0.0)"), ("STRIKE", "110.0")])
    with pytest.raises(ScriptError, match="already registered"):
        process([("STRIKE", "110.0"), ("strike", "120.0")])


def test_schedule_expansion_and_placeholders():
    result = process(
        [
            (
                "START: 2022-05-07 END: 2023-05-07 FREQ: 1m CALENDAR: CN.SSE",
                "acc = DCF(ACT365F, PeriodBegin, PeriodEnd)",
            )
        ]
    )
    assert len(result.events) == 12
    assert all(
        "PeriodBegin" not in text and "PeriodEnd" not in text for text in result.events.values()
    )


def test_schedule_placeholders_are_case_insensitive_and_skip_index_literals():
    result = process(
        [
            (
                "START: 2023-01-02 END: 2023-03-02 FREQ: 1M",
                "acc = DCF(ACT365F, periodbegin, PERIODEND) + EQ[PeriodBegin]",
            )
        ]
    )
    assert (
        result.events[Date.ymd(2023, 2, 2)]
        == "acc = DCF(ACT365F, 2023-01-02, 2023-02-02) + EQ[PeriodBegin]"
    )
    assert (
        result.events[Date.ymd(2023, 3, 2)]
        == "acc = DCF(ACT365F, 2023-02-02, 2023-03-02) + EQ[PeriodBegin]"
    )


def test_extensibility_via_override():
    class NoConstPreprocessor(Preprocessor):
        def is_const_variable(self, value: str) -> bool:
            return False

    result = NoConstPreprocessor().process([("RATE", "0.05"), (D, "x = RATE")])
    assert not result.const_variables
    assert "0.05" in result.events[D] and "RATE" not in result.events[D]


def _regex_replacement(text: str, pattern: str, replacement: str) -> str:
    """Python twin of std::regex_replace(icase) for the patterns used below."""
    return re.sub(
        pattern, replacement.replace("$&", r"\g<0>").replace("$$", "$"), text, flags=re.IGNORECASE
    )


def test_macro_expansion_matches_regex_replacement_outside_indices():
    statement = "x PAYS PAYOFF + payoff+PAYOFFPAYOFF y = PayOff2 z = EQ[PAYOFF] w = FIX(EQ[PAYOFF]@2023-11-30)"
    expanded = process([("PAYOFF", "MAX(spot() - 100, 0)"), (D, statement)]).events[D]
    index = statement.find("z = EQ[")
    expected = (
        _regex_replacement(statement[:index], "PAYOFF", "MAX(spot() - 100, 0)")
        + "z = EQ[PAYOFF] w = FIX(EQ[PAYOFF]@2023-11-30)"
    )
    assert expanded == expected


def test_macro_expansion_keeps_regex_semantics_for_special_characters():
    assert process([("M", "$&_$$"), (D, "x = M + m")]).events[D] == "x = M_$ + m_$"
    assert process([("A.B", "B.A"), (D, "x = AxB + A.B")]).events[D] == "x = B.A + B.A"


def test_macros_expand_in_case_insensitive_name_order():
    #  std::map<String_> folds case, so "a" expands before "B" (ASCII order would do "B" first and leave "x = B")
    assert process([("a", "B"), ("B", "c"), (D, "x = a")]).events[D] == "x = c"


def test_schedule_parameters():
    def dates(text):
        return [(str(b), str(e), str(f)) for b, e, f in parse_schedule(tokenize(text))]

    assert dates("START: 2022-05-07 END: 2022-08-07 FREQ: 1M")[0] == (
        "2022-05-07",
        "2022-06-07",
        "2022-06-07",
    )
    assert dates("START: 2022-05-07 END: 2022-08-07 FREQ: 1M FIXING: BEGIN")[0] == (
        "2022-05-07",
        "2022-06-07",
        "2022-05-07",
    )
    assert dates("START: 05/07/2022 END: 08/07/2022 FREQ: 1M")[-1][1] == "2022-08-07"
    assert [
        d[2]
        for d in dates(
            "START: 2022-01-29 END: 2022-04-30 FREQ: 1M CALENDAR: TARGET BizRule: Following"
        )
    ] == ["2022-02-28", "2022-03-28", "2022-04-28", "2022-05-02"]
    with pytest.raises(ScriptError, match="not followed by `:`"):
        dates("START: 2022-05-07 END: 2022-08-07 FREQ: 1M FIXING: BEGIN CALENDAR: TARGET")
    with pytest.raises(ScriptError, match="unknown token"):
        dates("START: 2022-05-07 END: 2022-08-07 FREQ: 1M BOGUS: 1")
    with pytest.raises(ScriptError, match="requires START, END and FREQ"):
        dates("START: 2022-05-07 FREQ: 1M")
