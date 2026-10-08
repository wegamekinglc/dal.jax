"""Ports of DAL's ``test_constprocessor.cpp`` / ``test_constcondprocessor.cpp`` plus IF metadata and domain checks."""

import pytest

from dal_jax.script import ast as A
from dal_jax.script.parser import Parser
from dal_jax.script.passes.constcond import process_const_conditions
from dal_jax.script.passes.constfold import process_constants
from dal_jax.script.passes.domain import process_domains
from dal_jax.script.passes.ifmeta import process_ifs
from dal_jax.script.passes.varindex import index_variables


def indexed(text):
    (events,), table = index_variables([[Parser().parse(text)]])
    return events[0], table


def const_processed(text):
    event, table = indexed(text)
    ((event,),) = process_constants([[event]], len(table.var_names))
    return event


def rhs(statement):
    return statement.args[1]


# --- constant marking ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,value",
    [
        ("x = 2 + 3", 5.0),
        ("x = 7 - 4", 3.0),
        ("x = 3 * 4", 12.0),
        ("x = 6 / 2", 3.0),
        ("x = 2 ^ 3", 8.0),
        ("x = MAX(2, 3)", 3.0),
        ("x = MIN(2, 3)", 2.0),
        ("x = SQRT(4)", 2.0),
        ("x = EXP(0)", 1.0),
        ("x = LOG(1)", 0.0),
        ("x = -5", -5.0),
        ("x = (2 + 3) * 4 - 1", 19.0),
        ("x = MAX(1, 3, 2)", 3.0),
    ],
)
def test_literal_subtrees_fold(text, value):
    value_node = rhs(const_processed(text)[0])
    assert value_node.is_const and value_node.const_val == pytest.approx(value)


def test_constant_variables_propagate():
    event = const_processed("y = 2\nx = y + 1")
    add = rhs(event[1])
    assert (
        add.is_const
        and add.const_val == 3.0
        and add.args[0].is_const
        and add.args[0].const_val == 2.0
    )


def test_non_constant_sources():
    assert not rhs(const_processed("x = spot() + 1")[0]).is_const
    mixed = const_processed("x = spot()\ny = x + 2")
    assert not rhs(mixed[1]).is_const and not rhs(mixed[1]).args[0].is_const
    assert not rhs(const_processed("x = 2\nx = spot()\ny = x")[2]).is_const
    assert not rhs(const_processed("IF spot() >= 1 THEN\nx = 2\nEND\ny = x")[1]).is_const
    stays = rhs(const_processed("x = 2\ny = x")[1])
    assert stays.is_const and stays.const_val == 2.0
    assert not rhs(const_processed("p pays 1\ny = p")[1]).is_const  # numeraire-deflated


def test_named_constants_never_fold():
    from dal_jax.strings import CIMap

    (events,), table = index_variables([[Parser(CIMap({"K": 2.0})).parse("x = K * 3")]])
    ((event,),) = process_constants([events], len(table.var_names))
    assert not rhs(event[0]).is_const and not rhs(event[0]).args[0].is_const


# --- IF metadata ---------------------------------------------------------------------------------


def test_if_metadata_collects_nested_writes():
    event, _ = indexed(
        "a = 0 b = 0 c = 0 IF spot() > 1 THEN a = 1 IF spot() > 2 THEN b = 2 END ELSE p pays 1 END APPEND(v, 1)"
    )
    (annotated,), depth = process_ifs([event])
    outer = annotated[3]
    inner = outer.args[2]
    assert depth == 2 and outer.affected_vars == (0, 1, 3) and inner.affected_vars == (1,)
    (vec_event,), _ = process_ifs([indexed("IF spot() > 1 THEN APPEND(v, 1) w[2] = 3 END")[0]])
    assert vec_event[0].affected_vectors == (0, 1)


# --- domains ----------------------------------------------------------------------------------------


def domain_processed(text, fuzzy=False):
    event, table = indexed(text)
    (event,), _ = process_ifs([event])
    ((event,),) = process_domains([[event]], len(table.var_names), fuzzy)
    return event


def test_domain_flags_constant_conditions():
    assert domain_processed("IF 2 >= 1 THEN x = 1 END")[0].always_true
    assert domain_processed("IF 2 < 1 THEN x = 1 END")[0].always_false
    node = domain_processed("IF spot() >= 1 THEN x = 1 END")[0]
    assert not node.always_true and not node.always_false
    #  x is {0} or {1} after the first IF, so x = 2 can never hold
    later = domain_processed("IF spot() > 1 THEN x = 1 END IF x = 2 THEN y = 1 END")[1]
    assert later.always_false


def test_fuzzy_domain_sets_discrete_bounds():
    equal = domain_processed(
        "IF spot() > 1 THEN x = 1 ELSE x = 3 END IF x = 1 THEN y = 1 END", fuzzy=True
    )[1].args[0]
    assert equal.is_discrete and (equal.lb, equal.rb) == (-0.5, 2.0)
    sup = domain_processed(
        "IF spot() > 1 THEN x = 1 ELSE x = 3 END IF x > 2 THEN y = 1 END", fuzzy=True
    )[1].args[0]
    assert sup.is_discrete and (sup.lb, sup.rb) == (-1.0, 1.0)
    continuous = domain_processed("IF spot() > 1 THEN y = 1 END", fuzzy=True)[0].args[0]
    assert not continuous.is_discrete
    #  a constant operand stays fuzzy (kept as true-or-false) in the legacy pipeline
    kept = domain_processed("IF 2 > 1 THEN y = 1 END", fuzzy=True)[0]
    assert not kept.always_true and not kept.args[0].is_discrete


def test_domain_requires_prepared_fixings():
    with pytest.raises(Exception, match="PreparationRequired"):
        domain_processed("x = FIX(EQ[a])")


# --- constant-condition folding -------------------------------------------------------------------


def const_cond_processed(text):
    return process_const_conditions([domain_processed(text)])[0]


def evaluate(event, n_vars=1):
    """Tiny exact evaluator for the folded statements of these tests (constants, arithmetic, IF)."""
    values = [0.0] * n_vars

    def expr(node):
        match node:
            case A.Const():
                return node.const_val
            case A.Var():
                return values[node.index]
            case A.Add():
                return expr(node.args[0]) + expr(node.args[1])
            case A.Mul():
                return expr(node.args[0]) * expr(node.args[1])
        raise NotImplementedError(node)

    def run(statement):
        match statement:
            case A.Assign():
                values[statement.args[0].index] = expr(statement.args[1])
            case A.Collect():
                for s in statement.args:
                    run(s)
            case _:
                raise NotImplementedError(statement)

    for statement in event:
        run(statement)
    return values


def test_always_true_if_becomes_collection():
    (collect,) = const_cond_processed("IF 2 >= 1 THEN x = 1 END")
    assert (
        isinstance(collect, A.Collect)
        and len(collect.args) == 1
        and isinstance(collect.args[0], A.Assign)
    )


def test_always_false_if_keeps_else_branch():
    (collect,) = const_cond_processed("IF 2 < 1 THEN x = 1 ELSE x = 2 END")
    assert isinstance(collect, A.Collect) and collect.args[0].args[1].const_val == 2.0
    (empty,) = const_cond_processed("IF 2 < 1 THEN x = 1 END")
    assert isinstance(empty, A.Collect) and not empty.args


def test_eager_booleans_keep_the_if_and_fold_children():
    (node,) = const_cond_processed("IF (2 >= 1) AND spot() >= 0 THEN x = 1 END")
    assert (
        isinstance(node, A.If)
        and isinstance(node.args[0], A.And)
        and isinstance(node.args[0].args[0], A.TrueNode)
    )
    (node,) = const_cond_processed("IF (2 < 1) OR spot() >= 0 THEN x = 1 END")
    assert isinstance(node.args[0], A.Or) and isinstance(node.args[0].args[0], A.FalseNode)
    assert isinstance(const_cond_processed("IF spot() >= 1 THEN x = 1 END")[0], A.If)


def test_nested_folding_preserves_statement_order():
    event = const_cond_processed("""
        IF 2 < 1 THEN x = 90 x = 99
        ELSE
            x = 1
            IF 3 < 2 THEN x = 80 x = 88 ELSE x = 10 * x + 2 x = 10 * x + 3 END
            x = 10 * x + 4
        END
        x = 10 * x + 5""")
    assert (
        len(event) == 2
        and isinstance(event[0], A.Collect)
        and len(event[0].args) == 3
        and isinstance(event[0].args[1], A.Collect)
    )
    assert evaluate(event)[0] == 12345.0
    event = const_cond_processed("""
        x = 1
        IF 2 > 1 THEN x = 10 * x + 2 IF 3 < 2 THEN x = 80 x = 88 END x = 10 * x + 3 END
        IF 2 < 1 THEN x = 90 x = 99 END
        x = 10 * x + 4""")
    assert (
        len(event) == 4
        and len(event[1].args) == 3
        and isinstance(event[1].args[1], A.Collect)
        and not event[1].args[1].args
    )
    assert evaluate(event)[0] == 1234.0


def test_division_by_zero_constant_domain_is_reported():
    with pytest.raises(Exception, match=r"Division by \{0\}"):
        domain_processed("x = 1 / 0")
