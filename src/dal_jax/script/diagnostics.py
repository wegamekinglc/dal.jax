"""Product dumps, ports of DAL's ``ScriptProduct_::DebugJson/DebugTree/Debug`` and ``DescribeScriptProductData``.

* ``debug_json`` - schema ``dal.script-product/1``: events partitioned at the
  evaluation date (past first), variables numbered, no folding.
* ``describe`` - schema ``dal.script-product/2``: every event with its origins,
  observations resolved against the default index, no partition.
* ``debug_tree`` / ``debug_text`` - the human-friendly tree and the legacy
  s-expression text.
"""

from dal_jax.dates.date import Date
from dal_jax.errors import DalError, script_error
from dal_jax.index import parse_index
from dal_jax.script.debug import (
    ASCII_STYLE,
    UNICODE_STYLE,
    DescribeContext,
    debug_node,
    debug_node_text,
    debug_node_tree,
    json_string,
    node_json,
)
from dal_jax.script.product import ScriptProduct, ScriptProductData
from dal_jax.strings import shortest_repr


def _product_for_dump(data: ScriptProductData, evaluation_date: Date, index_variables: bool = True) -> ScriptProduct:
    product = data.product()
    product.partition_events(evaluation_date)
    if index_variables:
        product.index_variables()
    return product


def _events_json(dates, events, phase: str, event_id: int, node_id: int) -> tuple[list[str], int, int]:
    out = []
    for date, event in zip(dates, events):
        statements = []
        for statement in event:
            text, node_id = node_json(debug_node(statement), node_id)
            statements.append(text)
        out.append(f'{{"index":{event_id},"date":"{date}","phase":"{phase}","statements":[' + ",".join(statements) + "]}")
        event_id += 1
    return out, event_id, node_id


def _variables_json(table) -> str:
    return ",".join(f'{{"index":{i},"name":{json_string(name)}}}' for i, name in enumerate(table.var_names))


def _constants_json(table) -> str:
    return ",".join(f'{{"index":{i},"name":{json_string(name)},"value":{shortest_repr(value)}}}'
                    for i, (name, value) in enumerate(zip(table.const_names, table.const_values)))


def debug_json(data: ScriptProductData, evaluation_date: Date) -> str:
    if data.settings.default_index:
        raise script_error("DebugSchemaUnsupported: dal.script-product/1 does not support default_index; use DescribeScriptProduct (dal.script-product/2)")
    product = _product_for_dump(data, evaluation_date)
    if product.preparation_error:
        raise script_error("DebugSchemaUnsupported: dal.script-product/1 does not support FIX; use DescribeScriptProduct (dal.script-product/2)")
    if product.has_exercise:
        raise script_error("DebugSchemaUnsupported: dal.script-product/1 does not support EXERCISE; use DescribeScriptProduct (dal.script-product/2)")
    out = ['{"schema":"dal.script-product/1"']
    table = product.vars
    if table.var_names:
        out.append(f',"variables":[{_variables_json(table)}],"payoff_index":{product.payoff_index}')
    if table.const_names:
        out.append(f',"constants":[{_constants_json(table)}]')
    past, event_id, node_id = _events_json(product.past_event_dates, product.past_events, "past", 0, 0)
    future, _, _ = _events_json(product.event_dates, product.events, "future", event_id, node_id)
    out.append(',"events":[' + ",".join(past + future) + "]}")
    return "".join(out)


def _cell_json(cell) -> str:
    if isinstance(cell, Date):
        return json_string(str(cell))
    if isinstance(cell, bool):
        return "true" if cell else "false"
    if isinstance(cell, float):
        return shortest_repr(cell)
    return json_string(str(cell))


def _canonical_default_index(name: str) -> str:
    if not name:
        return ""
    try:
        return parse_index(name).name
    except DalError as error:
        raise script_error(f"InvalidIndex: {error}; product.defaultIndex_={name}; expected a non-empty, fully parsed index name") from error


def _described_events(product: ScriptProduct, default_name: str, default_canonical: str) -> list[str]:
    events, node_id = [], 0
    for event_id, (date, event) in enumerate(zip(product.event_dates, product.events)):
        origins = ",".join(f'{{"row":{o.row},"offset":{o.offset},"event_date":{json_string(str(o.event_date))}}}'
                           for o in product.parsed_event_sources[event_id])
        statements = []
        for statement_id, statement in enumerate(event):
            text, node_id = node_json(debug_node(statement, DescribeContext(default_name, default_canonical, statement_id)), node_id)
            statements.append(text)
        events.append(f'{{"event_id":{event_id},"date":{json_string(str(date))},"origins":[{origins}],"statements":[' + ",".join(statements) + "]}")
    return events


def describe(data: ScriptProductData) -> str:
    product = data.product()
    product.index_variables()
    default_name = data.settings.default_index
    default_canonical = _canonical_default_index(default_name)
    table = product.vars
    rows = ",".join(f'{{"row":{i + 1},"date_or_definition":{_cell_json(cell)},"text":{json_string(text)}}}'
                    for i, (cell, text) in enumerate(zip(data.dates, data.events)))
    payoff = str(product.payoff_index) if product.has_pays and table.var_names else "null"
    events = _described_events(product, default_name, default_canonical)
    features = ",".join(json_string(name) for name in data.settings.regression_features)
    default_json = json_string(default_canonical) if default_canonical else "null"
    return (f'{{"schema":"dal.script-product/2","name":{json_string(data.name)},'
            f'"default_index":{{"original":{json_string(default_name)},"canonical":{default_json}}},'
            f'"regression_features":[{features}],"input_rows":[{rows}],"variables":[{_variables_json(table)}],'
            f'"constants":[{_constants_json(table)}],"payoff_index":{payoff},"events":[' + ",".join(events) + "]}")


def _tree_header(product: ScriptProduct) -> list[str]:
    table = product.vars
    if not table.var_names:
        return []
    out = ["Variables:" + ",".join(f" {name}" + ("*" if i == product.payoff_index else "") for i, name in enumerate(table.var_names)) + "\n"]
    if table.const_names:
        out.append("Constants:" + ",".join(f" {name}={shortest_repr(value)}" for name, value in zip(table.const_names, table.const_values)) + "\n")
    out.append("\n")
    return out


def _tree_event(event, st, width: int) -> list[str]:
    out = []
    for s, statement in enumerate(event):
        last = s + 1 == len(event)
        lines: list[str] = []
        first = (st.elbow if last else st.tee) + f"({s + 1}) "
        debug_node_tree(debug_node(statement), first, st.blank if last else st.pipe, st, width, lines)
        out.extend(line + "\n" for line in lines)
    return out


def debug_tree(data: ScriptProductData, evaluation_date: Date, ascii: bool = False, width: int = 125) -> str:
    st = ASCII_STYLE if ascii else UNICODE_STYLE
    product = _product_for_dump(data, evaluation_date)
    out = _tree_header(product)
    phases = ((product.past_event_dates, product.past_events, "past"), (product.event_dates, product.events, "future"))
    dated = [(date, event, phase) for dates, events, phase in phases for date, event in zip(dates, events)]
    for event_id, (date, event, phase) in enumerate(dated, start=1):
        out.append(f"{st.event} {event_id} {st.dot} {date} {st.dot} {phase}\n")
        out.extend(_tree_event(event, st, width))
        out.append("\n")
    return "".join(out)


def debug_text(data: ScriptProductData, evaluation_date: Date) -> str:
    """Legacy ``Product_Debug``: future events only, variables not numbered."""
    product = _product_for_dump(data, evaluation_date, index_variables=False)
    out = [f"Var[{v}] = {name}\n" for v, name in enumerate(product.vars.var_names)]
    for e, (date, event) in enumerate(zip(product.event_dates, product.events), start=1):
        out.append(f"EventTime_: {date}\tEvent_: {e}\n")
        for s, statement in enumerate(event, start=1):
            out.append(f"Statement_: {s}\n{debug_node_text(debug_node(statement))}\n")
    text = "".join(out)
    if not text:
        raise script_error("empty script product description")
    return text
