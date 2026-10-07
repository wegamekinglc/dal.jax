"""Ports of DAL's ``test_parser.cpp`` and the parsing half of ``test_exercise_parse.cpp``."""

import pytest

from dal_jax.errors import (
    DuplicateElse,
    DuplicateExercise,
    ImmutableVector,
    InvalidExerciseCondition,
    InvalidFor,
    InvalidPaymentDate,
    InvalidSmoothing,
    InvalidVectorEntry,
    ReservedIdentifier,
    ScriptError,
    UnsupportedExerciseNesting,
    VectorIndexOutOfRange,
)
from dal_jax.script import ast as A
from dal_jax.script.parser import Parser
from dal_jax.strings import CIMap


def parse(text, consts=None, vectors=None):
    return Parser(CIMap(consts or {}), CIMap(vectors or {})).parse(text)


def assert_sub_order(comparison, const_first, c, var):
    sub = comparison.args[0]
    assert isinstance(sub, A.Sub)
    const, variable = (sub.args[0], sub.args[1]) if const_first else (sub.args[1], sub.args[0])
    assert isinstance(const, A.Const) and const.const_val == c
    assert isinstance(variable, A.Var) and variable.name == var


def test_assign_and_functions():
    (statement,) = parse("x = 2")
    assert isinstance(statement, A.Assign) and isinstance(statement.args[0], A.Var) and statement.args[1].const_val == 2.0
    for name, cls in (("Log", A.Log), ("Exp", A.Exp), ("Sqrt", A.Sqrt)):
        assert isinstance(parse(f"y = 2.0\nx = {name}(y)")[1].args[1], cls)


def test_dcf_folds_to_a_constant():
    assert parse("x = DCF(ACT365F, 2023-04-23, 2024-04-23)")[0].args[1].const_val == pytest.approx(1.00274, abs=1e-5)
    with pytest.raises(ScriptError):
        parse("x = DCF(ACT365F, 2023-04-23, 2024-04-23, 2025-04-23)")


def test_if_with_and_without_else():
    (node,) = parse("IF x >= 2 THEN y = 3 + x END")
    assert isinstance(node, A.If) and node.first_else == -1
    cond = node.args[0]
    assert isinstance(cond, A.SupEqual) and isinstance(cond.args[0].args[0], A.Var) and isinstance(cond.args[0].args[1], A.Const)
    assign = node.args[1]
    assert isinstance(assign.args[1], A.Add) and isinstance(assign.args[1].args[0], A.Const)
    (node,) = parse("IF x >= 2 THEN y = 3 + x ELSE y = x END")
    assert node.first_else == 2 and isinstance(node.args[2].args[1], A.Var)


def test_for_unrolls_constant_bounds():
    (collected,) = parse("FOR(i, 0, COUNT) x = x + i END", {"COUNT": 3.0})
    assert isinstance(collected, A.Collect) and len(collected.args) == 3
    assert [s.args[1].args[1].const_val for s in collected.args] == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "text",
    ["FOR(i, 0.5, 3) x = i END", "FOR(i, 0, n) x = i END", "FOR(i, 0, 2) x = i", "FOR(i, 0, 2) i = 5 END", "FOR(i, 0, 2) i[0] = 5 END",
     "FOR(i, 0, 2) APPEND(i, 5) END", "FOR(i, 0, 2) x PAYS SUM(i) END", "FOR(i, 0, 1) EXERCISE 1 END", "FOR(i, 0, 0) EXERCISE 1 END",
     "FOR(i, 3, 1) x = i END", "FOR(i, 0, 10001) x = i END"],
)
def test_for_rejects_invalid_loops(text):
    with pytest.raises(ScriptError):
        parse(text)


def test_for_nested_and_empty_ranges():
    outer, empty = parse("FOR(i, 0, 2) FOR(j, i, 2) x = i + j END END FOR(k, 3, 3) x = 99 END")
    assert [len(c.args) for c in outer.args] == [2, 1]
    assert isinstance(empty, A.Collect) and not empty.args
    parser = Parser()
    parser.parse("FOR(i, 0, 0) x PAYS FIX(EQ[AAPL]) END")
    assert not parser.has_pays and not parser.preparation_error


def test_vector_append_entry_and_average():
    append, loop, payment = parse("APPEND(fixings, SPOT()) FOR(i, 0, 2) fixings[i] = fixings[i] + 1 END payoff PAYS AVERAGE(fixings)")
    assert isinstance(append, A.VectorAppend) and len(loop.args) == 2
    assignment = loop.args[1]
    assert isinstance(assignment, A.VectorAssign) and assignment.args[0].entry == 1
    assert isinstance(payment, A.Pays) and isinstance(payment.args[1], A.VectorReduce) and payment.args[1].kind == "Average"


def test_predefined_numeric_vectors_are_immutable():
    vectors = {"STRIKES": (100.0, 120.0, 150.0)}
    entry, mean = parse("x = STRIKES[1] y = AVERAGE(STRIKES)", vectors=vectors)
    assert entry.args[1].const_val == 120.0 and mean.args[1].const_val == pytest.approx(370.0 / 3.0)
    for text, error in (("APPEND(STRIKES, 1)", ImmutableVector), ("STRIKES[0] = 99", ScriptError), ("STRIKES = 99", ImmutableVector),
                        ("x = STRIKES[3]", VectorIndexOutOfRange)):
        with pytest.raises(error):
            parse(text, vectors=vectors)


@pytest.mark.parametrize("text", ["x = 1bad[0]", "x = _bad[0]", "x = v[1.5]", "x = v[k]", "x = v[2000000]"])
def test_vector_entry_rejects_invalid_forms(text):
    with pytest.raises(InvalidVectorEntry):
        parse(text)


@pytest.mark.parametrize("text", ["0X12 = 2.0;", "PAYS = 2.0;", "^ = 1", "x = ^", "_x = 1", "x = 2 +", "x = (2 + 3", "x = DCF", "x = LOG",
                                  "IF x > 1", "IF x > 2 y = 1 END", "x 2", "IF x + 2 THEN y = 1 END", "x = MIN(2)"])
def test_invalid_scripts(text):
    with pytest.raises(ScriptError):
        parse(text)


def test_valid_identifiers():
    parse("x_1 = 1")
    parse("Zeta = zeta + 1")


def test_precedence():
    assert isinstance(parse("x = 2 + 3 * 4")[0].args[1].args[1], A.Mul)
    assert isinstance(parse("x = 2 * 3 + 4")[0].args[1].args[0], A.Mul)
    assert isinstance(parse("x = 2 * 3 ^ 4")[0].args[1].args[1], A.Pow)
    assert isinstance(parse("x = (2 + 3) * 4")[0].args[1].args[0], A.Add)
    assert isinstance(parse("x = -3")[0].args[1], A.UMinus) and isinstance(parse("x = +3")[0].args[1], A.UPlus)
    assert len(parse("x = MIN(2, 3)")[0].args[1].args) == 2 and len(parse("x = MAX(2, 3, 4)")[0].args[1].args) == 3
    pow_node = parse("x = 2 ^ 3")[0].args[1]
    assert isinstance(pow_node, A.Pow) and all(isinstance(a, A.Const) for a in pow_node.args)


def test_comparisons():
    assert isinstance(parse("IF x > 2 THEN y = 1 END")[0].args[0], A.Sup)
    assert_sub_order(parse("IF x < 2 THEN y = 1 END")[0].args[0], True, 2.0, "x")
    assert_sub_order(parse("IF x <= 2 THEN y = 1 END")[0].args[0], True, 2.0, "x")
    assert isinstance(parse("IF x = 2 THEN y = 1 END")[0].args[0], A.Equal)
    neq = parse("IF x != 2 THEN y = 1 END")[0].args[0]
    assert isinstance(neq, A.Not) and isinstance(neq.args[0], A.Equal)
    both = parse("IF x > 2 AND x < 5 THEN y = 1 END")[0].args[0]
    assert isinstance(both, A.And)
    assert_sub_order(both.args[0], False, 2.0, "x")
    assert_sub_order(both.args[1], True, 5.0, "x")
    assert isinstance(parse("IF x > 2 OR x < 5 THEN y = 1 END")[0].args[0], A.Or)
    mixed = parse("IF x > 1 OR x > 2 AND x > 3 THEN y = 1 END")[0].args[0]
    assert isinstance(mixed, A.Or) and isinstance(mixed.args[0], A.Sup) and isinstance(mixed.args[1], A.And)


def test_if_rejects_repeated_else():
    with pytest.raises(DuplicateElse):
        parse("IF x > 2 THEN y = 1 ELSE y = 2 ELSE y = 3 END")


def test_smoothing_option():
    assert parse("IF spot() > 1:0.5 THEN x = 1 END")[0].args[0].eps == 0.5
    assert parse("IF spot() > 1 THEN x = 1 END")[0].args[0].eps == -1.0
    for text in ("IF spot() > 100; 0 THEN y = 2 END", "EXERCISE 1.0 IF spot() > 100: 0", "EXERCISE 1.0 IF spot() > 100; 0.0"):
        with pytest.raises(InvalidSmoothing):
            parse(text)
    with pytest.raises(ScriptError, match="^stod$"):
        parse("IF spot() > 1:-2 THEN x = 1 END")


def test_payment_dates():
    pays = parse("y pays 1 on 2025-12-01 z = 2")
    assert str(pays[0].payment_date) == "2025-12-01" and isinstance(pays[1], A.Assign)
    for text in ("y pays 1 on 2025-1-01", "y pays 1 on", "y pays 1 on 2025-13-01", "y pays 1 on 2025 -12-01"):
        with pytest.raises(InvalidPaymentDate):
            parse(text)


def test_fix_nodes():
    (statement,) = parse("x = FIX(EQ[Aapl], 2025-09-01) + FIX(fx[eur/usd])")
    first, second = statement.args[1].args
    assert (first.literal, first.canonical, str(first.fixing_date)) == ("EQ[Aapl]", "EQ[Aapl]", "2025-09-01")
    assert (second.canonical, second.fixing_date) == ("FX[EUR/USD]", None)
    parser = Parser()
    parser.parse("x = FIX(EQ[a])")
    assert parser.preparation_error.startswith("PreparationRequired: FIX requires a prepared observation; index=EQ[a]")


# --- EXERCISE -------------------------------------------------------------------------------


def exercise_of(event):
    assert isinstance(event[0], A.Exercise)
    return event[0]


def test_exercise_value_and_condition():
    plain = exercise_of(parse("EXERCISE 1.5"))
    assert len(plain.args) == 1 and plain.args[0].const_val == 1.5 and plain.eps == -1.0
    conditional = exercise_of(parse("EXERCISE MAX(spot() - 100, 0) IF spot() > 100"))
    assert isinstance(conditional.args[0], A.Max) and isinstance(conditional.args[1], A.Sup) and conditional.eps == -1.0
    shared = exercise_of(parse("EXERCISE 1.0 IF spot() > 100; 0.02"))
    assert shared.eps == 0.02 and shared.args[1].eps == 0.02


def test_exercise_eps_follows_first_comparison():
    first = exercise_of(parse("EXERCISE 1.0 IF spot() > 100 AND spot() < 200; 0.05"))
    assert first.eps == -1.0 and first.args[1].args[0].eps == -1.0 and first.args[1].args[1].eps == 0.05
    assert exercise_of(parse("EXERCISE 1.0 IF spot() > 100; 0.02 AND spot() < 200")).eps == 0.02


def test_exercise_keyword_rules():
    for text in ("exercise 1.5", "Exercise 1.5", "ExErCiSe 1.5"):
        exercise_of(parse(text))
    for text in ("x = exercise + 1", "exercise = 2", "exercise PAYS 2"):
        with pytest.raises(ReservedIdentifier, match="EXERCISE is a statement; rename the variable"):
            parse(text)
    for text in ("IF x > 1 THEN EXERCISE 1 END", "IF x > 1 THEN y = 2 ELSE EXERCISE 1 END"):
        with pytest.raises(UnsupportedExerciseNesting):
            parse(text)
    with pytest.raises(DuplicateExercise, match="line=2"):
        parse("EXERCISE 1\nEXERCISE 2")
    for text, keyword in (("EXERCISE 1 IF x > 1 THEN y = 2 END", "THEN"), ("EXERCISE 1 IF x > 1 END", "END"),
                          ("EXERCISE 1 IF x > 1\nIF y > 2 THEN z = 3 END", "IF")):
        with pytest.raises(InvalidExerciseCondition, match=keyword):
            parse(text)
    event = parse("EXERCISE 1 IF x > 1\ny = 2")
    assert len(event) == 2 and isinstance(event[1], A.Assign)


def test_errors_carry_source_positions():
    with pytest.raises(InvalidFor, match=r"line=1, column=1, offset=0"):
        parse("FOR(i, 0, 2) x = i")
