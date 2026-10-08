"""Recursive-descent parser, a port of DAL's ``script/parser.cpp``."""

import math
import re
from collections.abc import Callable, Sequence
from types import MappingProxyType

from dal_jax.dates.date import Date
from dal_jax.dates.daybasis import DayBasis
from dal_jax.errors import DalError, ScriptError, script_error
from dal_jax.index import parse_index
from dal_jax.script import ast as A
from dal_jax.script.lexer import SourceLocation, SourceOrigin, Token, lex
from dal_jax.strings import CIMap, ci_eq, ci_in, is_number, stod, stod_error

RESERVED_KEY_WORDS = (
    "IF",
    "END",
    "THEN",
    "ELSE",
    "DCF",
    "PAYS",
    "AND",
    "OR",
    "SPOT",
    "MAX",
    "MIN",
    "LOG",
    "SQRT",
    "EXP",
    "FIX",
    "EXERCISE",
    "FOR",
    "APPEND",
    "SUM",
    "AVERAGE",
    "ON",
)
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_NAME_CHARS = _LETTERS + "0123456789_."
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_MAX_LOOP_ITERATIONS = 10000
_MAX_EXPANDED_STATEMENTS = 100000
_MAX_INDEX = 1000000.0
#  name -> (min args, max args, node); SPOT and DCF are built specially
_FUNCTIONS = MappingProxyType(
    {
        "SPOT": (0, 0, A.Spot),
        "LOG": (1, 1, A.Log),
        "SQRT": (1, 1, A.Sqrt),
        "EXP": (1, 1, A.Exp),
        "MIN": (2, 1000, A.Min),
        "MAX": (2, 1000, A.Max),
        "DCF": (3, 3, None),
    }
)


def _fail(message: str) -> ScriptError:
    return script_error(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _fail(message)


def _to_double(text: str) -> float:
    value = stod(text)
    if value is None:
        raise _fail(stod_error(text))
    return value


def _function_name(text: str) -> str | None:
    return next((name for name in _FUNCTIONS if ci_eq(text, name)), None)


def is_vector_identifier(name: str) -> bool:
    return bool(name) and name[0] in _LETTERS and all(c in _NAME_CHARS for c in name)


def _is_reserved(text: str) -> bool:
    return ci_in(text, RESERVED_KEY_WORDS)


def read_vector_entry(values: Sequence[float], entry: int, context: str) -> float:
    if entry >= len(values):
        raise _fail("VectorIndexOutOfRange: " + context)
    return values[entry]


def reduce_vector_values(values: Sequence[float], kind: A.ReduceKind, context: str) -> float:
    if kind == "Sum":
        total = 0.0
        for value in values:
            total += value
        return total
    if not values:
        raise _fail("EmptyVectorReduction: " + context)
    if kind == "Average":
        return reduce_vector_values(values, "Sum", context) / len(values)
    result = values[0]
    for value in values[1:]:
        if (value < result) if kind == "Minimum" else (value > result):
            result = value
    return result


class Parser:
    """Parses one event text at a time; ``const_variables`` / ``numeric_vectors`` come from the preprocessor."""

    def __init__(
        self, const_variables: CIMap | None = None, numeric_vectors: CIMap | None = None
    ) -> None:
        self.const_variables = const_variables if const_variables is not None else CIMap()
        self.numeric_vectors = numeric_vectors if numeric_vectors is not None else CIMap()
        self.loop_indices = CIMap()
        self.preparation_error = ""
        self.has_exercise = False
        self.has_pays = False
        self.if_level = 0
        self.for_level = 0
        self.expanded_statements = 0
        self.tokens: list[Token] = []
        self.cur = 0

    def _tok(self, i: int | None = None) -> Token:
        return self.tokens[self.cur if i is None else i]

    def _text(self, i: int | None = None) -> str:
        return self._tok(i).text

    def _first(self, i: int | None = None) -> str:
        return self._text(i)[0]

    def _is(self, word: str, i: int | None = None) -> bool:
        return ci_eq(self._text(i), word)

    def _find_match(self, start: int, end: int) -> int:
        """Index of the ``)`` matching the ``(`` at ``start``."""
        opens, cur = 1, start + 1
        while cur != end and opens > 0:
            first = self._first(cur)
            opens += (first == "(") - (first == ")")
            cur += 1
        if cur == end and opens > 0:
            raise _fail("opening ( has no matching closing )")
        return cur - 1

    def _parentheses(
        self, on_match: Callable[[int], A.Node], on_no_match: Callable[[int], A.Node], end: int
    ) -> A.Node:
        _require(self.cur != end, "unexpected end of expression")
        if self._text() == "(":
            close = self._find_match(self.cur, end)
            self.cur += 1
            tree = on_match(close)
            _require(self.cur == close, "unexpected trailing tokens in parentheses")
            self.cur = close + 1
            return tree
        return on_no_match(end)

    def parse_expr(self, end: int) -> A.Node:
        lhs = self._parse_expr_l2(end)
        while self.cur != end and self._first() in "+-":
            op = self._first()
            self.cur += 1
            _require(self.cur != end, "unexpected end of statement")
            rhs = self._parse_expr_l2(end)
            lhs = (A.Add if op == "+" else A.Sub)(args=(lhs, rhs))
        return lhs

    def _parse_expr_l2(self, end: int) -> A.Node:
        lhs = self._parse_expr_l3(end)
        while self.cur != end and self._first() in "*/":
            op = self._first()
            self.cur += 1
            _require(self.cur != end, "unexpected end of statement")
            rhs = self._parse_expr_l3(end)
            lhs = (A.Mul if op == "*" else A.Div)(args=(lhs, rhs))
        return lhs

    def _parse_expr_l3(self, end: int) -> A.Node:
        lhs = self._parse_expr_l4(end)
        while self.cur != end and self._first() == "^":
            self.cur += 1
            _require(self.cur != end, "unexpected end of statement")
            rhs = self._parse_expr_l4(end)
            lhs = A.Pow(args=(lhs, rhs))
        return lhs

    def _parse_expr_l4(self, end: int) -> A.Node:
        if self.cur != end and self._first() in "+-":
            op = self._first()
            self.cur += 1
            _require(self.cur != end, "unexpected end of statement")
            rhs = self._parse_expr_l4(end)
            return (A.UPlus if op == "+" else A.UMinus)(args=(rhs,))
        return self._parentheses(self.parse_expr, self._parse_var_const_func, end)

    def _is_vector_reduction(self, end: int) -> bool:
        argument = self.cur + 1
        if argument == end or self._text(argument) != "(":
            return False
        argument += 1
        if argument == end or self._tok(argument).is_index:
            return False
        argument += 1
        return argument != end and self._text(argument) == ")"

    def _parse_var_const_func(self, end: int) -> A.Node:
        _require(self.cur != end, "unexpected end of expression")
        if self._is("FIX"):
            return self._parse_fix(end)
        if self._tok().is_index:
            return self._parse_vector_entry()
        if ci_in(self._text(), ("SUM", "AVERAGE", "MIN", "MAX")) and self._is_vector_reduction(end):
            return self._parse_vector_reduction(end)
        if self._first() in ".0123456789":
            return self._parse_const()
        name = _function_name(self._text())
        return self._parse_var() if name is None else self._parse_function(end, name)

    def _parse_function(self, end: int, name: str) -> A.Node:
        token = self._tok()
        self.cur += 1
        if name == "DCF":
            return A.Const(const_val=self._parse_dcf(end))
        args = tuple(self._parse_func_args(end))
        lo, hi, node = _FUNCTIONS[name]
        if not lo <= len(args) <= hi:
            raise _fail(f"Function {token.text}: wrong number of arguments_")
        return A.Spot(args=args, source=token.source) if name == "SPOT" else node(args=args)

    def _parse_const(self) -> A.Node:
        value = _to_double(self._text())
        self.cur += 1
        return A.Const(const_val=value)

    def _parse_var(self) -> A.Node:
        if self._tok().is_index:
            return self._parse_vector_entry()
        token = self._tok()
        _require(
            not ci_eq(token.text, "FIX"),
            "ReservedIdentifier: FIX is a function; rename the variable; "
            + token.source.describe(),
        )
        _require(
            not ci_eq(token.text, "EXERCISE"),
            "ReservedIdentifier: EXERCISE is a statement; rename the variable; "
            + token.source.describe(),
        )
        _require(token.text[0] in _LETTERS, f"Variable name {token.text} is invalid")
        _require(
            not _is_reserved(token.text),
            f"Variable name {token.text} is conflicted with an existing key word",
        )
        _require(
            token.text not in self.numeric_vectors,
            "ImmutableVector: predefined vector requires indexed access; "
            + token.source.describe(),
        )
        self.cur += 1
        if token.text in self.loop_indices:
            return A.Const(const_val=self.loop_indices[token.text])
        if token.text in self.const_variables:
            return A.ConstVar(name=token.text, const_val=self.const_variables[token.text])
        return A.Var(name=token.text)

    def _is_bare_name(self, token: Token) -> bool:
        return (
            not token.is_index and is_vector_identifier(token.text) and not _is_reserved(token.text)
        )

    def _is_fresh_loop_index(self, token: Token) -> bool:
        name = token.text
        return (
            self._is_bare_name(token)
            and name not in self.const_variables
            and name not in self.numeric_vectors
            and name not in self.loop_indices
        )

    def _numeric_constant(self, key: str, error: str) -> float:
        if is_number(key):
            return _to_double(key)
        if key in self.const_variables:
            return self.const_variables[key]
        if key in self.loop_indices:
            return self.loop_indices[key]
        raise _fail(error)

    @staticmethod
    def _nonnegative_integer(value: float, error: str) -> int:
        if (
            not math.isfinite(value)
            or value < 0.0
            or value > _MAX_INDEX
            or math.floor(value) != value
        ):
            raise _fail(error)
        return int(value)

    def _parse_vector_name(self, end: int, source: SourceLocation, operation: str) -> str:
        _require(
            self.cur != end and self._is_bare_name(self._tok()),
            f"{operation}: expected a vector name; {source.describe()}",
        )
        name = self._text()
        _require(
            name not in self.loop_indices,
            "InvalidFor: loop index cannot name a vector; " + source.describe(),
        )
        _require(
            name not in self.const_variables,
            f"{operation}: scalar constant is not a vector; {source.describe()}",
        )
        self.cur += 1
        return name

    def _parse_vector_entry(self) -> A.Node:
        raw, source = self._text(), self._tok().source
        opening, closing = raw.find("["), raw.find("]")
        _require(
            opening > 0 and closing == len(raw) - 1,
            "InvalidVectorEntry: expected name[constant-index]; " + source.describe(),
        )
        name, key = raw[:opening], raw[opening + 1 : closing]
        _require(
            is_vector_identifier(name),
            "InvalidVectorEntry: invalid vector name; " + source.describe(),
        )
        _require(
            name not in self.loop_indices,
            "InvalidFor: loop index cannot name a vector; " + source.describe(),
        )
        _require(
            not _is_reserved(name), "InvalidVectorEntry: reserved vector name; " + source.describe()
        )
        _require(
            name not in self.const_variables,
            "InvalidVectorEntry: scalar constant is not a vector; " + source.describe(),
        )
        value = self._numeric_constant(
            key, "InvalidVectorEntry: index must be an integer constant; " + source.describe()
        )
        entry = self._nonnegative_integer(
            value,
            "InvalidVectorEntry: index must be a nonnegative integer at most 1000000; "
            + source.describe(),
        )
        self.cur += 1
        if name in self.numeric_vectors:
            return A.Const(
                const_val=read_vector_entry(
                    self.numeric_vectors[name], entry, f"{name}; {source.describe()}"
                )
            )
        return A.VectorEntry(name=name, entry=entry, source=source)

    def _parse_vector_reduction(self, end: int) -> A.Node:
        function, source = self._text(), self._tok().source
        self.cur += 1
        _require(
            self.cur != end and self._text() == "(",
            "InvalidVectorReduction: expected '('; " + source.describe(),
        )
        self.cur += 1
        name = self._parse_vector_name(end, source, "InvalidVectorReduction")
        _require(
            self.cur != end and self._text() == ")",
            "InvalidVectorReduction: expected ')'; " + source.describe(),
        )
        self.cur += 1
        kind = (
            "Sum"
            if ci_eq(function, "SUM")
            else "Average"
            if ci_eq(function, "AVERAGE")
            else "Minimum"
            if ci_eq(function, "MIN")
            else "Maximum"
        )
        if name in self.numeric_vectors:
            return A.Const(
                const_val=reduce_vector_values(
                    self.numeric_vectors[name], kind, f"{name}; {source.describe()}"
                )
            )
        return A.VectorReduce(name=name, kind=kind, source=source)

    def _parse_func_args(self, end: int) -> list[A.Node]:
        _require(self.cur != end and self._first() == "(", "No opening ( following function name")
        close = self._find_match(self.cur, end)
        args = []
        self.cur += 1
        while self.cur != close:
            args.append(self.parse_expr(end))
            if self.cur != end and self._first() == ",":
                self.cur += 1
            elif self.cur != close:
                raise _fail("Arguments must be separated by commas")
        self.cur = close + 1
        return args

    def _dcf_field(self, close: int) -> str:
        text = ""
        while self.cur != close and self._first() != ",":
            text += self._text()
            self.cur += 1
        return text

    def _skip_commas(self, close: int, missing: str) -> None:
        _require(self.cur != close, missing)
        self.cur += 1
        while self.cur != close and self._first() == ",":
            self.cur += 1
        _require(self.cur != close, missing)

    def _parse_dcf(self, end: int) -> float:
        _require(self.cur != end and self._first() == "(", "missing opening '(' after `DCF`")
        close = self._find_match(self.cur, end)
        self.cur += 1
        _require(self.cur != close, "missing `basis` for `DCF`")
        basis = self._dcf_field(close)
        self._skip_commas(close, "missing `start` for `DCF`")
        start = self._dcf_field(close)
        self._skip_commas(close, "missing `end` for `DCF`")
        stop = self._dcf_field(close)
        _require(self.cur == close, "too many arguments for `DCF`")
        self.cur = close + 1
        try:
            return DayBasis.parse(basis).year_fraction(
                Date.from_string(start), Date.from_string(stop)
            )
        except DalError as error:
            raise _fail(error.detail) from error

    def _parse_fix(self, end: int) -> A.Node:
        function_source = self._tok().source
        self.cur += 1
        _require(
            self.cur != end and self._text() == "(",
            "ReservedIdentifier: FIX is a function; use FIX(index[,date]) or rename the variable; "
            + function_source.describe(),
        )
        self.cur += 1
        _require(
            self.cur != end and self._tok().is_index,
            "InvalidIndex: FIX requires an unquoted index literal; " + function_source.describe(),
        )
        literal, source = self._text(), self._tok().source
        try:
            index = parse_index(literal)
        except DalError as error:
            raise _fail(f"{error}; {source.describe()}; input={literal}") from error
        self.cur += 1
        fixing_date = None
        if self.cur != end and self._text() == ",":
            self.cur += 1
            fixing_date = self._parse_date_literal(
                end,
                source,
                "InvalidFixingDate",
                "FIX requires a strict YYYY-MM-DD date literal",
                lambda token: token.text != ")",
                stop_on_gap=False,
            )
        _require(
            self.cur != end and self._text() == ")",
            "InvalidIndex: FIX requires one index and an optional date; "
            + function_source.describe(),
        )
        self.cur += 1
        node = A.Fix(literal=literal, canonical=index.name, fixing_date=fixing_date, source=source)
        if not self.preparation_error:
            self.preparation_error = node.preparation_error()
        return node

    def _parse_date_literal(
        self,
        end: int,
        fallback: SourceLocation,
        code: str,
        expectation: str,
        accept,
        stop_on_gap: bool,
    ) -> Date:
        """Contiguous tokens forming ``yyyy-mm-dd``; ``stop_on_gap`` ends the literal at whitespace (``ON``)."""
        date_source = fallback if self.cur == end else self._tok().source
        text, contiguous = self._adjacent_tokens(end, date_source.offset, accept, stop_on_gap)
        _require(
            contiguous and bool(_ISO_DATE.fullmatch(text)),
            f"{code}: {expectation}; input={text}; {date_source.describe()}",
        )
        try:
            return Date.from_string(text)
        except DalError as error:
            raise _fail(f"{code}: {text}; {date_source.describe()}; {error.detail}") from error

    def _adjacent_tokens(
        self, end: int, offset: int, accept, stop_on_gap: bool
    ) -> tuple[str, bool]:
        """Concatenated accepted tokens and whether they touched without gaps."""
        text, contiguous = "", True
        while self.cur != end and accept(self._tok()):
            token = self._tok()
            if stop_on_gap and token.source.offset != offset:
                break
            contiguous = contiguous and token.source.offset == offset
            text += token.text
            offset = token.source.offset + len(token.text)
            self.cur += 1
        return text, contiguous

    def parse_cond(self, end: int) -> A.Node:
        lhs = self._parse_cond_l2(end)
        while self.cur != end and self._is("OR"):
            self.cur += 1
            _require(self.cur != end, "unexpected end of statement")
            rhs = self._parse_cond_l2(end)
            lhs = A.Or(args=(lhs, rhs))
        return lhs

    def _parse_cond_l2(self, end: int) -> A.Node:
        lhs = self._parentheses(self.parse_cond, self._parse_cond_elem, end)
        while self.cur != end and self._is("AND"):
            self.cur += 1
            _require(self.cur != end, "unexpected end of statement")
            rhs = self._parentheses(self.parse_cond, self._parse_cond_elem, end)
            lhs = A.And(args=(lhs, rhs))
        return lhs

    def _parse_cond_elem(self, end: int) -> A.Node:
        lhs = self.parse_expr(end)
        _require(self.cur != end, "unexpected end of statement")
        comparator = self._text()
        self.cur += 1
        _require(self.cur != end, "unexpected end of statement")
        rhs = self.parse_expr(end)
        eps = self._parse_cond_optionals(end)
        match comparator:
            case "=":
                return A.Equal(args=(A.Sub(args=(lhs, rhs)),), eps=eps)
            case "!=":
                return A.Not(args=(A.Equal(args=(A.Sub(args=(lhs, rhs)),), eps=eps),))
            case "<":
                return A.Sup(args=(A.Sub(args=(rhs, lhs)),), eps=eps)
            case ">":
                return A.Sup(args=(A.Sub(args=(lhs, rhs)),), eps=eps)
            case "<=":
                return A.SupEqual(args=(A.Sub(args=(rhs, lhs)),), eps=eps)
            case ">=":
                return A.SupEqual(args=(A.Sub(args=(lhs, rhs)),), eps=eps)
        raise _fail("elementary condition has no valid comparator")

    def _parse_cond_optionals(self, end: int) -> float:
        eps = -1.0
        while self.cur != end and self._text() in (";", ":"):
            self.cur += 1
            _require(self.cur != end, "unexpected end of statement")
            eps = _to_double(self._text())
            _require(
                math.isfinite(eps) and eps > 0.0,
                f"InvalidSmoothing: the ;eps option expects a finite positive width, got '{self._text()}'; {self._tok().source.describe()}",
            )
            self.cur += 1
        return eps

    def _parse_if(self, end: int) -> A.Node:
        self.cur += 1
        _require(self.cur != end, "`if` is not followed by `then`")
        cond = self.parse_cond(end)
        if self.cur == end or not self._is("then"):
            raise _fail("`if` is not followed by `then`")
        self.cur += 1
        else_statements = []
        self.if_level += 1
        then_statements = self._parse_block(end)
        _require(self.cur != end, "`if/then` is not followed by `else` or `end`")
        first_else = -1
        if self._is("ELSE"):
            self.cur += 1
            else_statements = self._parse_block(end)
            _require(self.cur != end, "`if/then/else` is not followed by `end`")
            _require(
                not self._is("ELSE"),
                "DuplicateElse: `if/then/else` admits a single `else` clause; "
                + self._tok().source.describe(),
            )
            first_else = len(then_statements) + 1
        self.if_level -= 1
        self.cur += 1
        return A.If(args=(cond, *then_statements, *else_statements), first_else=first_else)

    def _parse_block(self, end: int) -> list[A.Node]:
        """Statements up to the next ``ELSE`` / ``END``."""
        statements = []
        while self.cur != end and not self._is("ELSE") and not self._is("END"):
            statements.append(self.parse_statement(end))
        return statements

    def _parse_for_bound(self, end: int, context: str) -> int:
        _require(self.cur != end, "InvalidFor: missing loop bound" + context)
        sign = 1
        if self._text() in ("-", "+"):
            sign = -1 if self._text() == "-" else 1
            self.cur += 1
        _require(self.cur != end, "InvalidFor: missing loop bound" + context)
        value = sign * self._numeric_constant(
            self._text(), "InvalidFor: bound must be an integer constant" + context
        )
        self.cur += 1
        return self._nonnegative_integer(
            value, "InvalidFor: bound must be a nonnegative integer at most 1000000" + context
        )

    def _parse_for_header(self, end: int, context: str) -> tuple[str, int, int]:
        self.cur += 1
        _require(self.cur != end and self._text() == "(", "InvalidFor: expected '('" + context)
        self.cur += 1
        _require(
            self.cur != end and self._is_fresh_loop_index(self._tok()),
            "InvalidFor: expected a fresh loop index" + context,
        )
        index_name = self._text()
        self.cur += 1
        _require(
            self.cur != end and self._text() == ",",
            "InvalidFor: expected ',' after loop index" + context,
        )
        self.cur += 1
        first = self._parse_for_bound(end, context)
        _require(
            self.cur != end and self._text() == ",",
            "InvalidFor: expected ',' between bounds" + context,
        )
        self.cur += 1
        last = self._parse_for_bound(end, context)
        _require(self.cur != end and self._text() == ")", "InvalidFor: expected ')'" + context)
        _require(
            last >= first and last - first <= _MAX_LOOP_ITERATIONS,
            "InvalidFor: range must contain at most 10000 iterations" + context,
        )
        self.cur += 1
        return index_name, first, last

    def _parse_for_iteration(
        self, body: int, end: int, emit: bool, collected: list, context: str
    ) -> int:
        self.cur = body
        while self.cur != end and not self._is("END"):
            statement = self.parse_statement(end)
            if emit:
                _require(
                    self.expanded_statements < _MAX_EXPANDED_STATEMENTS,
                    "InvalidFor: expanded program exceeds 100000 statements" + context,
                )
                self.expanded_statements += 1
                collected.append(statement)
        _require(self.cur != end, "InvalidFor: missing END" + context)
        return self.cur

    def _parse_for(self, end: int) -> A.Node:
        context = "; " + self._tok().source.describe()
        index_name, first, last = self._parse_for_header(end, context)
        body_start, body_end = self.cur, None
        collected: list[A.Node] = []
        saved = (self.has_pays, self.has_exercise, self.preparation_error, self.expanded_statements)
        self.for_level += 1
        for i in range(first, max(last, first + 1)):
            self.loop_indices[index_name] = float(i)
            body = self._parse_for_iteration(body_start, end, i < last, collected, context)
            if body_end is None:
                body_end = body
            else:
                _require(body == body_end, "InvalidFor: inconsistent loop body" + context)
        self.for_level -= 1
        del self.loop_indices[index_name]
        if first == last:
            self.has_pays, self.has_exercise, self.preparation_error, self.expanded_statements = (
                saved
            )
        self.cur = body_end + 1
        return A.Collect(args=tuple(collected))

    def _parse_vector_append(self, end: int) -> A.Node:
        source = self._tok().source
        self.cur += 1
        _require(
            self.cur != end and self._text() == "(",
            "InvalidVectorAppend: expected '('; " + source.describe(),
        )
        close = self._find_match(self.cur, end)
        self.cur += 1
        name = self._parse_vector_name(close, source, "InvalidVectorAppend")
        _require(
            name not in self.numeric_vectors,
            "ImmutableVector: APPEND cannot modify a predefined vector; " + source.describe(),
        )
        _require(
            self.cur != close and self._text() == ",",
            "InvalidVectorAppend: expected ','; " + source.describe(),
        )
        self.cur += 1
        _require(self.cur != close, "InvalidVectorAppend: expected a value; " + source.describe())
        value = self.parse_expr(close)
        _require(
            self.cur == close,
            "InvalidVectorAppend: unexpected tokens after value; " + source.describe(),
        )
        self.cur = close + 1
        return A.VectorAppend(args=(value,), name=name, source=source)

    def _parse_exercise(self, end: int) -> A.Node:
        source = self._tok().source
        self.has_exercise = True
        self.cur += 1
        if self.cur != end and (self._text() == "=" or self._is("PAYS")):
            raise _fail(
                "ReservedIdentifier: EXERCISE is a statement; rename the variable; "
                + source.describe()
            )
        _require(
            self.cur != end,
            "unexpected end of statement; EXERCISE requires a value expression; "
            + source.describe(),
        )
        value = self.parse_expr(end)
        if self.cur != end and self._is("IF"):
            return self._parse_exercise_condition(end, source, value)
        return A.Exercise(args=(value,), source=source)

    def _parse_exercise_condition(self, end: int, source: SourceLocation, value: A.Node) -> A.Node:
        """``EXERCISE value IF cond``; the exercise shares the eps of the condition's first comparison."""
        self.cur += 1
        _require(
            self.cur != end,
            "unexpected end of statement; EXERCISE requires a condition after IF; "
            + source.describe(),
        )
        cond = self.parse_cond(end)
        if self.cur != end and _is_reserved(self._text()):
            raise _fail(
                f"InvalidExerciseCondition: EXERCISE condition ends on the statement keyword '{self._text()}'; "
                + self._tok().source.describe()
            )
        comparison = A.find_node(cond, lambda n: isinstance(n, A.Comparison))
        return A.Exercise(
            args=(value, cond),
            source=source,
            eps=comparison.eps if comparison is not None else -1.0,
        )

    def _parse_assignment(self, end: int, lhs: A.Node, kind) -> A.Node:
        self.cur += 1
        _require(self.cur != end, "unexpected end of statement")
        return kind(args=(lhs, self.parse_expr(end)))

    def _parse_pays(self, end: int, lhs: A.Node) -> A.Node:
        source = self._tok().source
        self.has_pays = True
        self.cur += 1
        _require(self.cur != end, "unexpected end of statement")
        rhs = self.parse_expr(end)
        payment_date = None
        if self.cur != end and self._is("ON"):
            on_source = self._tok().source
            self.cur += 1
            _require(
                self.cur != end,
                "InvalidPaymentDate: PAYS expects a YYYY-MM-DD date literal after ON; "
                + on_source.describe(),
            )
            payment_date = self._parse_date_literal(
                end,
                on_source,
                "InvalidPaymentDate",
                "PAYS expects a strict YYYY-MM-DD date literal after ON",
                lambda token: bool(token.text) and all(c in "0123456789-" for c in token.text),
                stop_on_gap=True,
            )
        return A.Pays(args=(lhs, rhs), source=source, payment_date=payment_date)

    def parse_statement(self, end: int) -> A.Node:
        if self._is("IF"):
            return self._parse_if(end)
        if self._is("FOR"):
            return self._parse_for(end)
        if self._is("APPEND"):
            return self._parse_vector_append(end)
        if self._is("EXERCISE"):
            _require(
                self.if_level == 0 and self.for_level == 0,
                "UnsupportedExerciseNesting: EXERCISE must be a top-level statement outside IF/FOR; "
                + self._tok().source.describe(),
            )
            _require(
                not self.has_exercise,
                "DuplicateExercise: an event admits at most one EXERCISE statement; "
                + self._tok().source.describe(),
            )
            return self._parse_exercise(end)
        return self._parse_target_statement(end)

    def _parse_target_statement(self, end: int) -> A.Node:
        """``target = value`` or ``target PAYS value``."""
        source = self._tok().source
        lhs = self._parse_var()
        _require(
            isinstance(lhs, (A.Var, A.VectorEntry)),
            "InvalidAssignmentTarget: expected a mutable variable or vector entry; "
            + source.describe(),
        )
        _require(self.cur != end, "unexpected end of statement")
        if self._text() == "=":
            return self._parse_assignment(
                end, lhs, A.VectorAssign if isinstance(lhs, A.VectorEntry) else A.Assign
            )
        if self._is("PAYS"):
            _require(
                isinstance(lhs, A.Var),
                "InvalidPaymentTarget: expected a scalar variable; " + source.describe(),
            )
            return self._parse_pays(end, lhs)
        raise _fail("statement without an instruction")

    def parse(self, event: str, origins: Sequence[SourceOrigin] = ()) -> A.Event:
        self.preparation_error = ""
        self.has_exercise = self.has_pays = False
        self.if_level = self.for_level = self.expanded_statements = 0
        self.loop_indices = CIMap()
        self.tokens = lex(event, origins)
        self.cur = 0
        statements = []
        while self.cur != len(self.tokens):
            statements.append(self.parse_statement(len(self.tokens)))
        return tuple(statements)
