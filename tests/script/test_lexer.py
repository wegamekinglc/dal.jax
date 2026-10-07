"""Port of DAL's ``tests/script/test_lexer.cpp``."""

import pytest

from dal_jax.dates import Date
from dal_jax.errors import InvalidIndex, InvalidScript
from dal_jax.script.lexer import SourceOrigin, index_literal_ranges, lex, tokenize


def test_tokenize_assignment():
    assert tokenize("x = 2") == ["x", "=", "2"]


def test_tokenize_operators_and_parentheses():
    assert tokenize("MAX(spot()-K,0.0)") == ["MAX", "(", "spot", "(", ")", "-", "K", ",", "0.0", ")"]


def test_tokenize_comparators():
    assert tokenize("a >= b != c <= d") == ["a", ">=", "b", "!=", "c", "<=", "d"]


def test_tokenize_schedule_colon():
    assert tokenize("START: 2022-05-07") == ["START", ":", "2022", "-", "05", "-", "07"]


def test_index_literal_tokens():
    tokens = tokenize("x = FIX(FX[EUR/USD]) + FIX(EQ[Aapl]@2026-12-31, 2026-09-11)")
    assert tokens[4] == "FX[EUR/USD]"
    assert tokens[9] == "EQ[Aapl]@2026-12-31"
    positioned = lex("FIX(FX[EUR/USD])")
    assert positioned[2].is_index and positioned[2].text == "FX[EUR/USD]" and positioned[2].source.offset == 4


def test_index_suffix_requires_at_or_greater_than():
    #  DAL HEAD: text after "]" belongs to the literal only after '@' or '>'
    assert tokenize("FIX(EQ[a] 2)") == ["FIX", "(", "EQ[a]", "2", ")"]
    assert tokenize("FIX(EQ[a]>3M, 2026-01-02)")[2] == "EQ[a]>3M"


def test_non_ascii_characters_are_not_word_characters():
    with pytest.raises(InvalidScript, match="unexpected character"):
        tokenize("x = é")


@pytest.mark.parametrize("text", ["x = EQ[a", 'x = EQ[a"b]', "x = FIX(EQ[a]@2026[)"])
def test_malformed_index_literals(text):
    with pytest.raises(InvalidIndex):
        tokenize(text)


def test_source_positions_and_origins():
    date = Date.ymd(2024, 1, 2)
    tokens = lex("a = 1\n  b = 2", [SourceOrigin(0, 3, date), SourceOrigin(6, 4, date)])
    b = tokens[3]
    assert (b.text, b.source.line, b.source.column, b.source.offset, b.source.row) == ("b", 2, 3, 8, 4)
    assert b.source.describe() == "line=2, column=3, offset=8, row=4, event=2024-01-02"


def test_index_literal_ranges():
    text = "z = EQ[PAYOFF] w = FIX(EQ[PAYOFF]@2023-11-30)"
    assert [text[a:b] for a, b in index_literal_ranges(text)] == ["EQ[PAYOFF]", "EQ[PAYOFF]@2023-11-30"]
    assert index_literal_ranges("no brackets") == []
