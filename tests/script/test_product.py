"""Ports of DAL's ``test_event.cpp`` and the dump goldens of ``test_exercise_parse.cpp``."""

import json

import pytest

from dal_jax.api import Product_Debug, Product_DebugJson, Product_DebugTree, Product_Describe, Product_New
from dal_jax.dates import Date, DayBasis, Holidays, make_schedule, parse_increment
from dal_jax.errors import DebugSchemaUnsupported, DuplicateExercise, ReservedIdentifier, ScriptError
from dal_jax.script import diagnostics
from dal_jax.script.debug import UNICODE_STYLE, debug_node, debug_node_text, debug_node_tree, node_json
from dal_jax.script.parser import Parser
from dal_jax.script.product import ScriptProduct, ScriptProductData

EVAL = Date.ymd(2026, 9, 12)


def product(rows, payoff=""):
    return ScriptProduct(rows, payoff)


def test_event_with_macro_and_ordering_rules():
    assert len(product([("STRIKE", "110.0"), (Date.ymd(2023, 12, 1), "call PAYS MAX(spot() - STRIKE, 0.0)")]).event_dates) == 1
    with pytest.raises(ScriptError):
        product([(Date.ymd(2023, 12, 1), "call PAYS MAX(spot() - STRIKE, 0.0)"), ("STRIKE", "110.0")])
    with pytest.raises(ScriptError):
        product([("STRIKE", "110.0"), ("STRIKE", "120.0")])


def test_event_with_schedules():
    assert len(product([("START: 2022-05-07 END: 2023-05-07 FREQ: 1m CALENDAR: CN.SSE", "STRIKE = 110.0")]).event_dates) == 12
    begin = product([("START: 2022-05-07 END: 2023-05-07 FREQ: 1m CALENDAR: CN.SSE FIXING: BEGIN", "STRIKE = 110.0")])
    end = product([("START: 2022-05-07 END: 2023-05-07 FREQ: 1m CALENDAR: CN.SSE FIXING: END", "STRIKE = 110.0")])
    assert begin.event_dates[0] == Date.ymd(2022, 5, 7) and end.event_dates[0] == Date.ymd(2022, 6, 7)


def test_schedule_placeholder_feeds_dcf():
    scripted = product([("START: 2022-05-07 END: 2023-05-07 FREQ: 1m CALENDAR: CN.SSE", "acc = DCF(ACT365F, 2022-05-07, PeriodEnd)")])
    schedule = make_schedule(Date.ymd(2022, 5, 7), Date.ymd(2023, 5, 7), Holidays("CN.SSE"), parse_increment("1m"))
    basis = DayBasis.parse("ACT365F")
    for i, event in enumerate(scripted.events):
        assert event[0].args[1].const_val == pytest.approx(basis.year_fraction(schedule[0], schedule[i + 1]), abs=1e-12)


def test_partition_and_variable_numbering():
    scripted = product([(Date.ymd(2021, 1, 4), "x = 1"), (Date.ymd(2023, 3, 1), "z PAYS x * 2")])
    scripted.partition_events(Date.ymd(2022, 9, 15))
    scripted.index_variables()
    assert scripted.past_event_dates == [Date.ymd(2021, 1, 4)] and scripted.event_dates == [Date.ymd(2023, 3, 1)]
    assert scripted.vars.var_names == ("x", "z") and scripted.payoff_index == 1
    with pytest.raises(ScriptError):
        scripted.partition_events(Date.ymd(2022, 9, 15))


def test_payoff_receiver_follows_pays():
    d = Date.ymd(2026, 9, 22)
    routed = product([(d, "x = 1"), (d, "y PAYS x"), (d, "EXERCISE y")])
    routed.index_variables()
    assert routed.payoff_index == 1
    exercise_only = product([(d, "x = 1\nEXERCISE x")])
    exercise_only.index_variables()
    assert exercise_only.payoff_index == -1 and exercise_only.has_payoff and not exercise_only.has_pays
    plain = product([(d, "x = 1"), (d, "y = x + 1")])
    plain.index_variables()
    assert plain.payoff_index == 1
    named = product([(d, "x = 1"), (d, "y = x + 1")], payoff="X")
    named.index_variables()
    assert named.payoff_index == 0


def test_variables_are_case_insensitive_and_keep_first_spelling():
    scripted = product([(Date.ymd(2024, 1, 2), "Alive = 1\nALIVE = alive + 1\np pays aLiVe")])
    scripted.index_variables()
    assert scripted.vars.var_names == ("Alive", "p")


def test_vector_capacities():
    scripted = product([(Date.ymd(2024, 1, 2), "v[3] = 1 APPEND(v, 2) APPEND(v, 3) APPEND(w, 1) x = SUM(v) + w[0]")])
    scripted.index_variables()
    assert scripted.vars.vector_names == ("v", "w") and scripted.vars.vector_capacities == (6, 2)
    with pytest.raises(ScriptError, match="VectorNameConflict"):
        bad = product([(Date.ymd(2024, 1, 2), "v = 1 APPEND(v, 2)")])
        bad.index_variables()


def test_reserved_definitions_and_duplicate_exercise_rows():
    with pytest.raises(ReservedIdentifier, match="rename the definition; row=1"):
        product([("exercise", "2"), (Date.ymd(2026, 9, 22), "pay PAYS 1")])
    d = Date.ymd(2026, 9, 22)
    with pytest.raises(DuplicateExercise, match="row=2, event=2026-09-22"):
        product([(d, "EXERCISE 1"), (d, "EXERCISE 2")])


# --- dumps ------------------------------------------------------------------------------------


def debugged(text):
    return debug_node(Parser().parse(text)[0])


def test_exercise_text_json_and_tree_goldens():
    assert debug_node_text(debugged("EXERCISE 1.5")) == "EXERCISE[CONT,EPS=-1.000000](\n\tCONST[1.500000]\n)\n"
    assert debug_node_text(debugged("EXERCISE 1.5 IF spot() > 100; 0.02")) == (
        "EXERCISE[CONT,EPS=0.020000](\n\tCONST[1.500000]\n,\n\tGTZERO[CONT,EPS=0.020000](\n\t\tSUBTRACT(\n\t\t\tSPOT\n\t\t,\n"
        "\t\t\tCONST[100.000000]\n\t\t)\n\t)\n)\n")
    assert node_json(debugged("EXERCISE 1.5 IF spot() > 100; 0.02"))[0] == (
        '{"id":"n0","kind":"exercise","mode":"continuous","eps":0.02,"children":[{"id":"n1","kind":"const","value":1.5},'
        '{"id":"n2","kind":"gt0","mode":"continuous","eps":0.02,"children":[{"id":"n3","kind":"sub","children":['
        '{"id":"n4","kind":"spot"},{"id":"n5","kind":"const","value":100}]}]}]}')
    assert node_json(debugged("EXERCISE 1.5"))[0] == (
        '{"id":"n0","kind":"exercise","mode":"continuous","eps":-1,"children":[{"id":"n1","kind":"const","value":1.5}]}')
    lines: list[str] = []
    debug_node_tree(debugged("EXERCISE 1.5 IF spot() > 100; 0.02"), "(1) ", "    ", UNICODE_STYLE, 125, lines)
    assert lines == ["(1) exercise 1.5 if spot() > 100 ⟨ε=0.02⟩"]
    lines = []
    debug_node_tree(debugged("EXERCISE MAX(spot() - 100, 0) IF spot() > 100"), "(1) ", "    ", UNICODE_STYLE, 20, lines)
    assert lines[0] == "(1) exercise" and "max" in lines[1] and any("if spot() > 100" in line for line in lines)


def test_describe_exercise_golden():
    data = ScriptProductData((Date.ymd(2026, 9, 22),), ("EXERCISE 1.5",), name="ex")
    assert diagnostics.describe(data) == (
        '{"schema":"dal.script-product/2","name":"ex","default_index":{"original":"","canonical":null},'
        '"regression_features":[],"input_rows":[{"row":1,"date_or_definition":"2026-09-22","text":"EXERCISE 1.5"}],'
        '"variables":[],"constants":[],"payoff_index":null,'
        '"events":[{"event_id":0,"date":"2026-09-22","origins":[{"row":1,"offset":0,"event_date":"2026-09-22"}],'
        '"statements":[{"id":"n0","kind":"exercise","mode":"continuous","eps":-1,"children":['
        '{"id":"n1","kind":"const","value":1.5}]}]}]}')
    described = diagnostics.describe(ScriptProductData((Date.ymd(2026, 9, 22),), ("x = 1\nEXERCISE x",), name="ex"))
    assert '"variables":[{"index":0,"name":"x"}]' in described and '"payoff_index":null' in described


def test_tree_and_schema_one_gate_for_exercise():
    data = ScriptProductData((Date.ymd(2026, 9, 22),), ("EXERCISE 1.5",))
    assert diagnostics.debug_tree(data, EVAL) == "📅 1 · 2026-09-22 · future\n└── (1) exercise 1.5\n\n"
    with pytest.raises(DebugSchemaUnsupported, match="EXERCISE"):
        diagnostics.debug_json(data, EVAL)
    with pytest.raises(DebugSchemaUnsupported, match="FIX"):
        diagnostics.debug_json(ScriptProductData((Date.ymd(2026, 9, 22),), ("x = FIX(EQ[a])",)), EVAL)


def test_vector_nodes_render():
    node = debugged("APPEND(v, 2)")
    assert (node.kind, node.name) == ("vector_append", "v")
    data = ScriptProductData((Date.ymd(2026, 9, 22),), ("APPEND(v, 2) v[0] = v[0] + 1 x = SUM(v)",))
    tree = diagnostics.debug_tree(data, EVAL)
    assert "APPEND(v, 2)" in tree and "v[0] ← v[0] + 1" in tree and "x ← SUM(v)" in tree
    dump = json.loads(diagnostics.debug_json(data, EVAL))
    append, assign, _ = dump["events"][0]["statements"]
    assert append == {"id": "n0", "kind": "vector_append", "name": "v", "index": 0, "children": [{"id": "n1", "kind": "const", "value": 2}]}
    assert assign["target"] == {"id": "n3", "kind": "vector_entry", "name": "v", "index": 0, "entry": 0}


def test_api_round_trip():
    from dal_jax.api import EvaluationDate_Get, EvaluationDate_Set

    previous = EvaluationDate_Get()
    try:
        EvaluationDate_Set(Date.ymd(2022, 9, 15))
        p = Product_New(["STRIKE", Date.ymd(2025, 9, 14)], ["120.0", "call pays MAX(spot() - STRIKE, 0.0)"])
        assert Product_Describe(p)["constants"] == [{"index": 0, "name": "STRIKE", "value": 120}]
        assert Product_DebugJson(p).startswith('{"schema":"dal.script-product/1"')
        assert Product_DebugTree(p, True, 80).startswith("Variables: call*")
        assert Product_Debug(p).startswith("EventTime_: 2025-09-14\tEvent_: 1\n")
        with pytest.raises(TypeError):
            Product_New([1.5], ["x = 1"])
    finally:
        EvaluationDate_Set(previous)
