"""Event-table front end, a port of DAL's ``script/preprocessor.cpp`` and ``event/schedule.cpp``.

Rows whose date column is a :class:`~dal_jax.dates.Date` are events.  Other
rows are definitions: a ``START: ... END: ... FREQ: ...`` schedule (expanded
into one event per period, with ``PeriodBegin`` / ``PeriodEnd`` replaced), a
numeric vector ``[a, b, ...]``, a numeric constant, or otherwise a textual
macro.  Macro names are replaced case-insensitively outside index literals;
events on the same date are joined with newlines and remember their origins.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from dal_jax.dates.date import Date
from dal_jax.dates.holidays import NO_HOLIDAYS, Holidays
from dal_jax.dates.increment import parse_increment
from dal_jax.dates.schedule import make_schedule
from dal_jax.errors import ScriptError, script_error
from dal_jax.script.lexer import SourceOrigin, index_literal_ranges, tokenize
from dal_jax.strings import CIMap, ci_eq, ci_find, is_number, stod, to_double

type Cell = Date | str
type EventRow = tuple[Cell, str]

_IDENTIFIER = re.compile(r"[A-Za-z0-9_]+")
_TRIM = " \t\r\n"


@dataclass(slots=True)
class PreprocessedEvents:
    const_variables: CIMap = field(default_factory=CIMap)  # name -> float
    numeric_vectors: CIMap = field(default_factory=CIMap)  # name -> tuple[float, ...]
    events: dict[Date, str] = field(default_factory=dict)  # ordered by date
    sources: dict[Date, list[SourceOrigin]] = field(default_factory=dict)


# --- macro replacement --------------------------------------------------------------


def _replace_literal(text: str, pattern: str, replacement: str) -> str:
    out, start = [], 0
    while (found := ci_find(text, pattern, start)) != -1:
        out.append(text[start:found])
        out.append(replacement)
        start = found + len(pattern)
    out.append(text[start:])
    return "".join(out)


def _ecma_group(replacement: str, i: int, match: re.Match) -> tuple[str, int]:
    """``$n`` / ``$nn`` at ``i``: the text and the characters consumed; two digits only when they name a group."""
    groups = match.re.groups or 0
    digits = replacement[i + 1]
    if i + 2 < len(replacement) and replacement[i + 2].isdigit() and int(digits + replacement[i + 2]) <= groups:
        digits += replacement[i + 2]
    n = int(digits)
    return ((match.group(n) or "") if 0 < n <= groups else "$" + digits), 1 + len(digits)


def _ecma_escape(replacement: str, i: int, match: re.Match, text: str) -> tuple[str, int]:
    """The ``$`` escape at ``i``: its text and the characters consumed."""
    nxt = replacement[i + 1]
    if nxt.isdigit():
        return _ecma_group(replacement, i, match)
    simple = {"$": "$", "&": match.group(0), "`": text[: match.start()], "'": text[match.end() :]}
    return (simple[nxt], 2) if nxt in simple else ("$", 1)


def _ecma_replacement(replacement: str, match: re.Match, text: str) -> str:
    """``std::regex_replace`` format escapes: ``$&``, ``$$``, `` $` ``, ``$'`` and ``$n``."""
    out, i = [], 0
    while i < len(replacement):
        if replacement[i] == "$" and i + 1 < len(replacement):
            piece, used = _ecma_escape(replacement, i, match, text)
        else:
            piece, used = replacement[i], 1
        out.append(piece)
        i += used
    return "".join(out)


def replace_outside_indices(statement: str, pattern: str, replacement: str) -> str:
    """DAL's ``ReplaceOutsideIndices``: literal case-insensitive replacement when the
    pattern is an identifier and the replacement has no ``$``; otherwise an
    ECMAScript case-insensitive ``regex_replace``.  Index literals are skipped."""
    literal = bool(_IDENTIFIER.fullmatch(pattern)) and "$" not in replacement
    if literal and ci_find(statement, pattern) == -1:
        return statement
    if literal:
        replace = lambda text: _replace_literal(text, pattern, replacement)  # noqa: E731
    else:
        expression = re.compile(pattern, re.IGNORECASE)
        replace = lambda text: expression.sub(lambda m: _ecma_replacement(replacement, m, text), text)  # noqa: E731
    out, start = [], 0
    for begin, end in index_literal_ranges(statement):
        out.append(replace(statement[start:begin]))
        out.append(statement[begin:end])
        start = end
    out.append(replace(statement[start:]))
    return "".join(out)


# --- definitions -----------------------------------------------------------------------


def _vector_value(token: str, row: int) -> float:
    token = token.strip(_TRIM)
    value = stod(token) if token else None
    if value is None:
        raise script_error(f"InvalidVectorDefinition: expected finite numeric entries; row={row}")
    if not math.isfinite(value):
        raise script_error(f"InvalidVectorDefinition: entries must be finite; row={row}")
    return value


def _parse_numeric_vector(definition: str, row: int) -> tuple[float, ...]:
    text = definition.strip(_TRIM)
    if not (len(text) >= 2 and text[0] == "[" and text[-1] == "]"):
        raise script_error(f"InvalidVectorDefinition: expected [number, ...]; row={row}")
    if len(text) == 2:
        return ()
    return tuple(_vector_value(token, row) for token in text[1:-1].split(","))


def _with_source(expand, row: int, date: Date | None) -> str:
    try:
        return expand()
    except ScriptError as error:
        context = f"; row={row}" + (f", event={date}" if date is not None else "")
        raise script_error(str(error) + context) from error


def _schedule_date(tokens: Sequence[str], start: int) -> Date:
    if start + 5 > len(tokens):
        raise script_error("InvalidScript: schedule date expects five tokens such as 2022-09-15")
    return Date.from_string("".join(tokens[start : start + 5]))


def _schedule_fixing(tokens: Sequence[str], i: int) -> bool:
    if ci_eq(tokens[i], "BEGIN"):
        return False
    if ci_eq(tokens[i], "END"):
        return True
    raise script_error("unknown token for fixing")


#  name, setting, parser of the value at a token position, tokens consumed after the name
#  (FIXING consumes only its name, as in DAL, so it must come last)
_SCHEDULE_PARAMETERS = (
    ("START", "start", _schedule_date, 7),
    ("END", "end", _schedule_date, 7),
    ("FREQ", "tenor", lambda tokens, i: parse_increment(tokens[i]), 3),
    ("CALENDAR", "holidays", lambda tokens, i: Holidays(tokens[i]), 3),
    ("BizRule", "convention", lambda tokens, i: tokens[i], 3),
    ("FIXING", "fix_at_end", _schedule_fixing, 1),
)


def _schedule_parameter(name: str) -> tuple:
    parameter = next((p for p in _SCHEDULE_PARAMETERS if ci_eq(name, p[0])), None)
    if parameter is None:
        raise script_error("unknown token")
    return parameter


def parse_schedule(tokens: Sequence[str]) -> list[tuple[Date, Date, Date]]:
    """``ParseSchedule``: ``(period begin, period end, fixing date)`` per period.

    Like DAL, ``FIXING: BEGIN|END`` does not advance past its value, so it must
    be the last parameter.
    """
    settings = {"start": None, "end": None, "tenor": None, "holidays": NO_HOLIDAYS, "convention": "Unadjusted", "fix_at_end": True}
    i = 0
    while i < len(tokens) - 2:
        if tokens[i + 1] != ":":
            raise script_error("schedule parameter name not followed by `:`")
        _, setting, parse, consumed = _schedule_parameter(tokens[i])
        settings[setting] = parse(tokens, i + 2)
        i += consumed
    if any(settings[key] is None for key in ("start", "end", "tenor")):
        raise script_error("InvalidScript: a schedule requires START, END and FREQ")
    dates = make_schedule(settings["start"], settings["end"], settings["holidays"], settings["tenor"], "Forward", settings["convention"])
    fix_at_end = settings["fix_at_end"]
    return [(dates[k - 1], dates[k], dates[k] if fix_at_end else dates[k - 1]) for k in range(1, len(dates))]


_RESERVED_DEFINITIONS = (
    ("FIX", "ReservedIdentifier: FIX is a function; rename the definition; row={row}"),
    ("EXERCISE", "ReservedIdentifier: EXERCISE is a statement; rename the definition; row={row}"),
)


class Preprocessor:
    """Overridable like DAL's ``Preprocessor_`` (``is_schedule``, ``is_const_variable``, expansions)."""

    def is_schedule(self, description: str) -> bool:
        return ":" in description

    def is_const_variable(self, value: str) -> bool:
        return is_number(value)

    def expand_macros(self, statement: str, macros: CIMap) -> str:
        for name, body in macros.items():
            statement = replace_outside_indices(statement, name, body)
        return statement

    def expand_schedule_placeholders(self, statement: str, begin: Date, end: Date) -> str:
        for placeholder, date in (("PeriodBegin", begin), ("PeriodEnd", end)):
            if ci_find(statement, placeholder) != -1:
                statement = replace_outside_indices(statement, placeholder, str(date))
        return statement

    def process(self, rows: Sequence[EventRow]) -> PreprocessedEvents:
        result = PreprocessedEvents()
        macros = CIMap()
        events: dict[Date, str] = {}
        for row, (cell, text) in enumerate(rows, start=1):
            if isinstance(cell, Date):
                expanded = _with_source(lambda: self.expand_macros(text, macros), row, cell)
                self._append(result, events, cell, expanded, row)
            elif self.is_schedule(cell):
                schedule = parse_schedule(tokenize(cell))
                expanded = _with_source(lambda: self.expand_macros(text, macros), row, None)
                for begin, end, fixing in schedule:
                    final = _with_source(lambda: self.expand_schedule_placeholders(expanded, begin, end), row, fixing)
                    self._append(result, events, fixing, final, row)
            else:
                self._define(result, macros, events, cell, text, row)
        result.events = dict(sorted(events.items()))
        result.sources = {date: result.sources[date] for date in result.events}
        return result

    @staticmethod
    def _check_definition(result: PreprocessedEvents, macros: CIMap, events: dict, name: str, row: int) -> None:
        for reserved, message in _RESERVED_DEFINITIONS:
            if ci_eq(name, reserved):
                raise script_error(message.format(row=row))
        if any(ci_eq(name, keyword) for keyword in ("FOR", "APPEND", "SUM", "AVERAGE")):
            raise script_error(f"ReservedIdentifier: vector/loop keyword cannot name a definition; row={row}")
        for registry, message in ((macros, "macro name has already registered"), (result.const_variables, "const macro name has already registered"),
                                  (result.numeric_vectors, "vector name has already registered")):
            if name in registry:
                raise script_error(message)
        if events:
            raise script_error("macros should always at the front")

    def _define(self, result: PreprocessedEvents, macros: CIMap, events: dict, name: str, text: str, row: int) -> None:
        self._check_definition(result, macros, events, name, row)
        definition = text.strip(_TRIM)
        if definition.startswith("["):
            result.numeric_vectors[name] = _parse_numeric_vector(definition, row)
        elif self.is_const_variable(text):
            result.const_variables[name] = to_double(text)
        else:
            macros[name] = text

    @staticmethod
    def _append(result: PreprocessedEvents, events: dict[Date, str], date: Date, statement: str, row: int) -> None:
        offset = 0 if date not in events else len(events[date]) + 1
        result.sources.setdefault(date, []).append(SourceOrigin(offset, row, date))
        events[date] = statement if date not in events else events[date] + "\n" + statement


__all__ = ["Cell", "EventRow", "PreprocessedEvents", "Preprocessor", "parse_schedule", "replace_outside_indices"]
