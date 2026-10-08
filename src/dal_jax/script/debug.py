"""Debug renderings of script statements, a port of DAL's ``visitor/debugger.hpp``."""

from dataclasses import dataclass, field
from types import MappingProxyType

from dal_jax.dates.date import Date, datetime_string
from dal_jax.errors import script_error
from dal_jax.script import ast as A
from dal_jax.script.lexer import SourceLocation
from dal_jax.strings import shortest_repr as debug_number


@dataclass(slots=True)
class DebugObservation:
    original: str
    canonical: str
    fixing_date: Date | None
    source: SourceLocation
    statement_id: int


@dataclass(slots=True)
class DebugNode:
    label: str
    kind: str
    name: str = ""
    index: int = -1
    entry: int = 0
    first_else: int = -1
    number: float = 0.0
    lb: float = 0.0
    rb: float = 0.0
    discrete: bool = False
    children: list["DebugNode"] = field(default_factory=list)
    observation: DebugObservation | None = None


def _f(value: float) -> str:
    """``std::to_string(double)``."""
    return f"{value:f}"


_SIMPLE = MappingProxyType(
    {
        A.Collect: ("COLLECT", "collect"),
        A.UPlus: ("UPLUS", "uplus"),
        A.UMinus: ("UMINUS", "neg"),
        A.Add: ("ADD", "add"),
        A.Sub: ("SUBTRACT", "sub"),
        A.Mul: ("MULT", "mul"),
        A.Div: ("DIV", "div"),
        A.Pow: ("POW", "pow"),
        A.Log: ("LOG", "log"),
        A.Exp: ("EXP", "exp"),
        A.Sqrt: ("SQRT", "sqrt"),
        A.Max: ("MAX", "max"),
        A.Min: ("MIN", "min"),
        A.Not: ("NOT", "not"),
        A.And: ("AND", "and"),
        A.Or: ("OR", "or"),
        A.Assign: ("ASSIGN", "assign"),
        A.VectorAssign: ("VECTOR_ASSIGN", "vector_assign"),
        A.Pays: ("PAYS", "pays"),
        A.TrueNode: ("TRUE", "true"),
        A.FalseNode: ("FALSE", "false"),
    }
)
_COMPARISONS = MappingProxyType(
    {
        A.Equal: ("EQUALZERO", "eq0"),
        A.Sup: ("GTZERO", "gt0"),
        A.SupEqual: ("GTEQUALZERO", "ge0"),
    }
)
_REDUCE = MappingProxyType(
    {
        "Sum": ("SUM", "vector_sum"),
        "Average": ("AVERAGE", "vector_average"),
        "Minimum": ("MIN", "vector_min"),
        "Maximum": ("MAX", "vector_max"),
    }
)


@dataclass(frozen=True, slots=True)
class DescribeContext:
    """Describe mode (schema /2) attaches observations: ``SPOT()`` binds to the default index."""

    default_original: str
    default_canonical: str
    statement_id: int


def _comparison_node(node: A.Comparison, children: list[DebugNode], _describe) -> DebugNode:
    label, kind = _COMPARISONS[type(node)]
    if node.is_discrete:
        return DebugNode(
            f"{label}[DISCRETE,BOUNDS={_f(node.lb)},{_f(node.rb)}]",
            kind,
            discrete=True,
            lb=node.lb,
            rb=node.rb,
            children=children,
        )
    return DebugNode(f"{label}[CONT,EPS={_f(node.eps)}]", kind, number=node.eps, children=children)


def _spot_node(
    node: A.Spot, children: list[DebugNode], describe: DescribeContext | None
) -> DebugNode:
    observation = None
    if describe is not None:
        observation = DebugObservation(
            describe.default_original,
            describe.default_canonical,
            None,
            node.source,
            describe.statement_id,
        )
    return DebugNode("SPOT", "spot", children=children, observation=observation)


def _fix_node(
    node: A.Fix, children: list[DebugNode], describe: DescribeContext | None
) -> DebugNode:
    label = (
        f"FIX({node.literal}"
        + (f", {node.fixing_date}" if node.fixing_date is not None else "")
        + ")"
    )
    observation = None
    if describe is not None:
        observation = DebugObservation(
            node.literal, node.canonical, node.fixing_date, node.source, describe.statement_id
        )
    return DebugNode(label, "fix", children=children, observation=observation)


def _vector_reduce_node(node: A.VectorReduce, children: list[DebugNode], _describe) -> DebugNode:
    function, kind = _REDUCE[node.kind]
    return DebugNode(
        f"{function}[{node.name}]", kind, name=node.name, index=node.index, children=children
    )


_BUILDERS = MappingProxyType(
    {
        A.Equal: _comparison_node,
        A.Sup: _comparison_node,
        A.SupEqual: _comparison_node,
        A.VectorAppend: lambda node, children, _: DebugNode(
            f"VECTOR_APPEND[{node.name}]",
            "vector_append",
            name=node.name,
            index=node.index,
            children=children,
        ),
        A.VectorEntry: lambda node, children, _: DebugNode(
            f"VECTOR_ENTRY[{node.name},{node.entry}]",
            "vector_entry",
            name=node.name,
            index=node.index,
            entry=node.entry,
            children=children,
        ),
        A.VectorReduce: _vector_reduce_node,
        A.Exercise: lambda node, children, _: DebugNode(
            f"EXERCISE[CONT,EPS={_f(node.eps)}]", "exercise", number=node.eps, children=children
        ),
        A.Spot: _spot_node,
        A.Fix: _fix_node,
        A.If: lambda node, children, _: DebugNode(
            f"IF[FIRSTELSE={node.first_else}]", "if", first_else=node.first_else, children=children
        ),
        A.Const: lambda node, children, _: DebugNode(
            f"CONST[{_f(node.const_val)}]", "const", number=node.const_val, children=children
        ),
        A.Var: lambda node, children, _: DebugNode(
            f"VAR[{node.name},{node.index},{_f(node.const_val)}]",
            "var",
            name=node.name,
            index=node.index,
            number=node.const_val,
            children=children,
        ),
        A.ConstVar: lambda node, children, _: DebugNode(
            f"CONST_VAR[{node.name},{node.index},{_f(node.const_val)}]",
            "const_var",
            name=node.name,
            index=node.index,
            number=node.const_val,
            children=children,
        ),
    }
)


def debug_node(node: A.Node, describe: DescribeContext | None = None) -> DebugNode:
    children = [debug_node(arg, describe) for arg in node.args]
    if (builder := _BUILDERS.get(type(node))) is not None:
        return builder(node, children, describe)
    label, kind = _SIMPLE[type(node)]
    return DebugNode(label, kind, children=children)


_JSON_ESCAPES = MappingProxyType(
    {code: f"\\u00{code:02x}" for code in range(0x20)}
    | {
        ord('"'): '\\"',
        ord("\\"): "\\\\",
        ord("\n"): "\\n",
        ord("\r"): "\\r",
        ord("\t"): "\\t",
    }
)


def json_string(text: str) -> str:
    return '"' + text.translate(_JSON_ESCAPES) + '"'


def json_source(source: SourceLocation) -> str:
    event_date = json_string(str(source.event_date)) if source.event_date is not None else "null"
    return f'{{"row":{source.row},"offset":{source.offset},"line":{source.line},"column":{source.column},"event_date":{event_date}}}'


def json_fixing_date(date: Date | None) -> str:
    mode = "Explicit" if date is not None else "EventDate"
    literal = json_string(str(date)) if date is not None else "null"
    return f',"fixing_date_mode":"{mode}","fixing_date_literal":{literal}'


def look_ahead_message(
    original: str,
    canonical: str,
    fixing: Date,
    source: SourceLocation,
    statement_id: int,
    node_id: int,
) -> str:
    return (
        f"LookAheadObservation: expected fixing <= event; index={canonical}; fixing={datetime_string(fixing)}; {source.describe()}; "
        f"original={original}; canonical={canonical}; statement={statement_id}; node=n{node_id}"
    )


def _json_observation(observation: DebugObservation, kind: str, node_id: int) -> str:
    source = observation.source
    if source.event_date is None:
        raise script_error("InvalidFixingDate: observation source has no event date")
    fixing = observation.fixing_date if observation.fixing_date is not None else source.event_date
    if fixing > source.event_date:
        raise script_error(
            look_ahead_message(
                observation.original,
                observation.canonical,
                fixing,
                source,
                observation.statement_id,
                node_id,
            )
        )
    original = json_string(observation.original) if observation.original else "null"
    canonical = json_string(observation.canonical) if observation.canonical else "null"
    return (
        f',"type":"{"Fix" if kind == "fix" else "Spot"}","index_original":{original},"index_canonical":{canonical}'
        f"{json_fixing_date(observation.fixing_date)}"
        f',"fixing_date":{json_string(str(fixing))},"fixing_time":{json_string(datetime_string(fixing))},"source":{json_source(source)}'
    )


_VECTOR_REDUCTIONS = ("vector_sum", "vector_average", "vector_min", "vector_max")


class _JsonWriter:
    def __init__(self, first_id: int = 0) -> None:
        self.next_id = first_id
        self._writers = {
            "if": self._if_fields,
            "assign": self._target_fields,
            "pays": self._target_fields,
            "vector_assign": self._target_fields,
            "var": lambda node: self._named(node) + ',"const_value":' + debug_number(node.number),
            "const_var": lambda node: self._named(node) + ',"value":' + debug_number(node.number),
            "vector_entry": lambda node: self._named(node) + f',"entry":{node.entry}',
            "vector_append": lambda node: self._named(node) + self._children(node.children),
            "const": lambda node: f',"value":{debug_number(node.number)}',
            **{kind: self._named for kind in _VECTOR_REDUCTIONS},
            **{
                kind: lambda node: self._fuzzy(node) + self._children(node.children)
                for kind in ("eq0", "gt0", "ge0", "exercise")
            },
        }

    def node(self, node: DebugNode) -> str:
        node_id = self.next_id
        self.next_id += 1
        out = [f'{{"id":"n{node_id}","kind":{json_string(node.kind)}']
        if node.observation is not None:
            out.append(_json_observation(node.observation, node.kind, node_id))
        if (writer := self._writers.get(node.kind)) is not None:
            out.append(writer(node))
        elif node.children:
            out.append(self._children(node.children))
        out.append("}")
        return "".join(out)

    def _children(self, children: list[DebugNode]) -> str:
        return ',"children":[' + ",".join(self.node(child) for child in children) + "]"

    @staticmethod
    def _named(node: DebugNode) -> str:
        return f',"name":{json_string(node.name)},"index":{node.index}'

    @staticmethod
    def _fuzzy(node: DebugNode) -> str:
        if node.discrete:
            return f',"mode":"discrete","lb":{debug_number(node.lb)},"rb":{debug_number(node.rb)}'
        return f',"mode":"continuous","eps":{debug_number(node.number)}'

    def _if_fields(self, node: DebugNode) -> str:
        first_else = len(node.children) if node.first_else < 0 else node.first_else
        condition = self.node(node.children[0])
        then = ",".join(self.node(child) for child in node.children[1:first_else])
        other = ",".join(self.node(child) for child in node.children[first_else:])
        return f',"condition":{condition},"then":[{then}],"else":[{other}]'

    def _target_fields(self, node: DebugNode) -> str:
        target = self.node(node.children[0])
        return f',"target":{target},"value":{self.node(node.children[1])}'


def node_json(node: DebugNode, first_id: int = 0) -> tuple[str, int]:
    """JSON of one statement and the next free node id."""
    writer = _JsonWriter(first_id)
    return writer.node(node), writer.next_id


@dataclass(frozen=True, slots=True)
class TreeStyle:
    tee: str
    elbow: str
    pipe: str
    blank: str
    plus: str
    minus: str
    times: str
    over: str
    power: str
    negate: str
    log: str
    exp: str
    sqrt: str
    max: str
    min: str
    eq: str
    gt: str
    ge: str
    and_: str
    or_: str
    not_: str
    true: str
    false: str
    assign: str
    pays: str
    then: str
    else_: str
    cond: str
    event: str
    dot: str
    l_ang: str
    r_ang: str
    eps: str


UNICODE_STYLE = TreeStyle(
    "├── ",
    "└── ",
    "│   ",
    "    ",
    "+",
    "−",
    "×",
    "÷",
    "^",
    "−",
    "ln",
    "exp",
    "√",
    "max",
    "min",
    "=",
    ">",
    "≥",
    "∧",
    "∨",
    "¬",
    "⊤",
    "⊥",
    "←",
    "⇐",
    "▶ ",
    "▷ ",
    "? ",
    "📅",
    "·",
    "⟨",
    "⟩",
    "ε",
)
ASCII_STYLE = TreeStyle(
    "|-- ",
    "`-- ",
    "|   ",
    "    ",
    "+",
    "-",
    "*",
    "/",
    "^",
    "-",
    "ln",
    "exp",
    "sqrt",
    "max",
    "min",
    "=",
    ">",
    ">=",
    "and",
    "or",
    "not",
    "true",
    "false",
    "<-",
    "<=",
    "> ",
    ". ",
    "? ",
    "#",
    "@",
    "<",
    ">",
    "eps",
)

_WIDE = (
    (0x1100, 0x115F),
    (0x2E80, 0xA4CF),
    (0xAC00, 0xD7A3),
    (0xF900, 0xFAFF),
    (0xFE30, 0xFE6F),
    (0xFF00, 0xFF60),
    (0x1F300, 0x1F64F),
    (0x1F900, 0x1F9FF),
)


def display_width(text: str) -> int:
    return sum(2 if any(lo <= ord(c) <= hi for lo, hi in _WIDE) else 1 for c in text)


_PRECEDENCE = MappingProxyType(
    {
        "assign": 0,
        "vector_assign": 0,
        "vector_append": 0,
        "pays": 0,
        "if": 0,
        "collect": 0,
        "exercise": 0,
        "or": 1,
        "and": 2,
        "eq0": 3,
        "gt0": 3,
        "ge0": 3,
        "add": 4,
        "sub": 4,
        "mul": 5,
        "div": 5,
        "not": 6,
        "neg": 6,
        "uplus": 6,
        "pow": 7,
    }
)


def _prec(kind: str) -> int:
    return _PRECEDENCE.get(kind, 8)


def _paren(child: DebugNode, st: TreeStyle, context: int) -> str:
    inner = tree_inline(child, st)
    return inner if _prec(child.kind) >= context else f"({inner})"


def _fuzzy_suffix(node: DebugNode, st: TreeStyle) -> str:
    if node.discrete:
        return f" {st.l_ang}[{debug_number(node.lb)}, {debug_number(node.rb)}]{st.r_ang}"
    if node.number > 0.0:
        return f" {st.l_ang}{st.eps}={debug_number(node.number)}{st.r_ang}"
    return ""


def _function_symbol(kind: str, st: TreeStyle) -> str:
    return st.log if kind == "log" else st.exp if kind == "exp" else st.sqrt


def _binary_symbol(kind: str, st: TreeStyle) -> str:
    return {"add": st.plus, "sub": st.minus, "mul": st.times, "div": st.over}.get(kind, st.power)


def _compare_symbol(kind: str, st: TreeStyle) -> str:
    return st.eq if kind == "eq0" else st.gt if kind == "gt0" else st.ge


_REDUCE_NAMES = MappingProxyType(
    {
        "vector_sum": "SUM",
        "vector_average": "AVERAGE",
        "vector_min": "MIN",
        "vector_max": "MAX",
    }
)


def _first_else(node: DebugNode) -> int:
    return len(node.children) if node.first_else < 0 else node.first_else


def _inline_neg(node: DebugNode, st: TreeStyle) -> str:
    operand = node.children[0]
    if operand.kind == "neg":
        return f"{st.negate}({tree_inline(operand, st)})"
    return st.negate + _paren(operand, st, 8)


def _inline_binary(node: DebugNode, st: TreeStyle) -> str:
    prec = _prec(node.kind)
    left = _paren(node.children[0], st, 8 if node.kind == "pow" else prec)
    return f"{left} {_binary_symbol(node.kind, st)} {_paren(node.children[1], st, prec + 1)}"


def _inline_logical(node: DebugNode, st: TreeStyle) -> str:
    symbol = st.and_ if node.kind == "and" else st.or_
    return f" {symbol} ".join(_paren(child, st, _prec(node.kind)) for child in node.children)


def _inline_comparison(node: DebugNode, st: TreeStyle) -> str:
    operand, op = node.children[0], _compare_symbol(node.kind, st)
    if operand.kind == "sub":
        text = f"{_paren(operand.children[0], st, 3)} {op} {_paren(operand.children[1], st, 3)}"
    else:
        text = f"{_paren(operand, st, 3)} {op} 0"
    return text + _fuzzy_suffix(node, st)


def _inline_target(node: DebugNode, st: TreeStyle) -> str:
    arrow = st.pays if node.kind == "pays" else st.assign
    return f"{tree_inline(node.children[0], st)} {arrow} {tree_inline(node.children[1], st)}"


def _inline_if(node: DebugNode, st: TreeStyle) -> str:
    children, first_else = node.children, _first_else(node)
    text = f"if {tree_inline(children[0], st)} then"
    text += "".join(" " + tree_inline(child, st) for child in children[1:first_else])
    return text + "".join(" else " + tree_inline(child, st) for child in children[first_else:])


def _inline_exercise(node: DebugNode, st: TreeStyle) -> str:
    text = "exercise " + tree_inline(node.children[0], st)
    return text + (" if " + tree_inline(node.children[1], st) if len(node.children) > 1 else "")


def _inline_collect(node: DebugNode, st: TreeStyle) -> str:
    return "; ".join(tree_inline(child, st) for child in node.children)


_INLINE = MappingProxyType(
    {
        "const": lambda node, st: debug_number(node.number),
        "var": lambda node, st: node.name,
        "const_var": lambda node, st: node.name,
        "vector_entry": lambda node, st: f"{node.name}[{node.entry}]",
        **{
            kind: lambda node, st: f"{_REDUCE_NAMES[node.kind]}({node.name})"
            for kind in _REDUCE_NAMES
        },
        "spot": lambda node, st: "spot()",
        "fix": lambda node, st: node.label,
        "true": lambda node, st: st.true,
        "false": lambda node, st: st.false,
        **{
            kind: lambda node, st: (
                f"{_function_symbol(node.kind, st)}({tree_inline(node.children[0], st)})"
            )
            for kind in ("log", "exp", "sqrt")
        },
        "uplus": lambda node, st: _paren(node.children[0], st, 6),
        "neg": _inline_neg,
        **{
            kind: lambda node, st: (
                f"{st.max if node.kind == 'max' else st.min}("
                + ", ".join(tree_inline(c, st) for c in node.children)
                + ")"
            )
            for kind in ("max", "min")
        },
        **{kind: _inline_binary for kind in ("add", "sub", "mul", "div", "pow")},
        "not": lambda node, st: st.not_ + _paren(node.children[0], st, 7),
        "and": _inline_logical,
        "or": _inline_logical,
        **{kind: _inline_comparison for kind in ("eq0", "gt0", "ge0")},
        **{kind: _inline_target for kind in ("assign", "vector_assign", "pays")},
        "vector_append": lambda node, st: (
            f"APPEND({node.name}, {tree_inline(node.children[0], st)})"
        ),
        "if": _inline_if,
        "exercise": _inline_exercise,
    }
)


def tree_inline(node: DebugNode, st: TreeStyle) -> str:
    """Best-effort single-line form with minimal parentheses."""
    return _INLINE.get(node.kind, _inline_collect)(node, st)


Branches = tuple[str, list[tuple[DebugNode, str, bool]]]


def _target_branches(node: DebugNode, first: str, st: TreeStyle, _width: int) -> Branches:
    arrow = st.pays if node.kind == "pays" else st.assign
    return f"{first}{tree_inline(node.children[0], st)} {arrow}", [(node.children[1], "", True)]


def _exercise_branches(node: DebugNode, first: str, _st: TreeStyle, _width: int) -> Branches:
    children = node.children
    return first + "exercise", [(children[0], "", True)] + (
        [(children[1], "if ", False)] if len(children) > 1 else []
    )


def _if_branches(node: DebugNode, first: str, st: TreeStyle, width: int) -> Branches:
    children, first_else = node.children, _first_else(node)
    condition = tree_inline(children[0], st)
    branches = []
    if display_width(f"{first}if {condition} then") <= width:
        header = f"{first}if {condition} then"
    else:
        header = first + "if"
        branches.append((children[0], st.cond, False))
    branches += [(child, st.then, False) for child in children[1:first_else]]
    branches += [(child, st.else_, False) for child in children[first_else:]]
    return header, branches


def _unary_branches(node: DebugNode, first: str, st: TreeStyle, _width: int) -> Branches:
    symbol = {"not": st.not_, "neg": st.negate, "uplus": "+"}.get(node.kind) or _function_symbol(
        node.kind, st
    )
    return first + symbol, [(node.children[0], "", True)]


def _nary_branches(node: DebugNode, first: str, st: TreeStyle, _width: int) -> Branches:
    symbol = {"max": st.max, "min": st.min, "collect": "", "pow": st.power}.get(node.kind)
    return first + (symbol if symbol is not None else _binary_symbol(node.kind, st)), [
        (child, "", True) for child in node.children
    ]


_BRANCHES = MappingProxyType(
    {
        **{kind: _target_branches for kind in ("assign", "vector_assign", "pays")},
        "vector_append": lambda node, first, st, _: (
            f"{first}APPEND({node.name})",
            [(node.children[0], "", True)],
        ),
        "exercise": _exercise_branches,
        "if": _if_branches,
        **{
            kind: lambda node, first, st, _: (
                first + _compare_symbol(node.kind, st) + _fuzzy_suffix(node, st),
                [(node.children[0], "", True)],
            )
            for kind in ("eq0", "gt0", "ge0")
        },
        **{kind: _unary_branches for kind in ("not", "neg", "uplus", "log", "exp", "sqrt")},
    }
)


def _branches(node: DebugNode, first: str, st: TreeStyle, width: int) -> Branches:
    return _BRANCHES.get(node.kind, _nary_branches)(node, first, st, width)


def debug_node_tree(
    node: DebugNode, first: str, cont: str, st: TreeStyle, width: int, out: list[str]
) -> None:
    whole = first + tree_inline(node, st)
    if display_width(whole) <= width or not node.children:
        out.append(whole)
        return
    header, branches = _branches(node, first, st, width)
    out.append(header)
    for i, (child, marker, connected) in enumerate(branches):
        last = i + 1 == len(branches)
        branch_first = (
            cont + (st.elbow if last else st.tee) + marker if connected else cont + marker
        )
        branch_cont = (
            cont + (st.blank if last else st.pipe)
            if connected
            else cont + " " * display_width(marker)
        )
        debug_node_tree(child, branch_first, branch_cont, st, width, out)


def debug_node_text(node: DebugNode, depth: int = 0) -> str:
    tabs = "\t" * depth
    if not node.children:
        return f"{tabs}{node.label}\n"
    out = [f"{tabs}{node.label}(\n"]
    for i, child in enumerate(node.children):
        out.append(debug_node_text(child, depth + 1))
        if i + 1 < len(node.children):
            out.append(f"{tabs},\n")
    out.append(f"{tabs})\n")
    return "".join(out)
