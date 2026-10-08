"""DAL string semantics: case-insensitive ``String_`` and C ``stod`` number parsing.

DAL's ``String_`` compares, searches and sorts case-insensitively through the
``CI_ORDER`` table: ASCII letters fold to upper case, and ``{|}~`` plus DEL sort
just after ``Z`` ... ``_``.  Script identifiers, keywords, macro names and the
``std::map`` orderings that number constants all inherit these rules, so they
are reproduced here exactly.
"""

import math
import re
from collections.abc import Iterator, Mapping
from typing import Any


def _ci_char(code: int) -> int:
    if 97 <= code <= 122:  # a-z -> A-Z
        return code - 32
    if 123 <= code <= 127:  # {|}~DEL sort right after the letters
        return code - 26
    return code


_CI_TABLE = {code: _ci_char(code) for code in range(128) if _ci_char(code) != code}


def ci_key(text: str) -> str:
    """Sort/equality key of DAL's ``ci_traits`` (bytes >= 0x80 keep their value)."""
    return text.translate(_CI_TABLE)


def ci_eq(lhs: str, rhs: str) -> bool:
    return ci_key(lhs) == ci_key(rhs)


def ci_find(text: str, pattern: str, start: int = 0) -> int:
    """``String_::find``: case-insensitive substring search, -1 when absent."""
    return ci_key(text).find(ci_key(pattern), start)


def ci_in(text: str, choices) -> bool:
    key = ci_key(text)
    return any(key == ci_key(choice) for choice in choices)


class CIMap(Mapping[str, Any]):
    """``std::map<String_, V>``: case-insensitive keys, iterated in ``CI_ORDER``.

    The first spelling inserted for a key is the one reported back, as in DAL.
    """

    __slots__ = ("_items",)

    def __init__(self, items=()) -> None:
        self._items: dict[str, tuple[str, Any]] = {}
        for key, value in dict(items).items() if isinstance(items, Mapping) else items:
            self[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._items[ci_key(key)][1]

    def __setitem__(self, key: str, value: Any) -> None:
        folded = ci_key(key)
        original = self._items[folded][0] if folded in self._items else key
        self._items[folded] = (original, value)

    def __delitem__(self, key: str) -> None:
        del self._items[ci_key(key)]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and ci_key(key) in self._items

    def __iter__(self) -> Iterator[str]:
        return (self._items[k][0] for k in sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"CIMap({dict(self.items())!r})"


#  std::stod: optional leading C whitespace, then a decimal or hexadecimal float,
#  INF/INFINITY or NAN[(chars)], the whole string must be consumed (DAL checks idx).
_C_SPACE = " \t\n\v\f\r"
_STOD = re.compile(
    r"[+-]?(?:"
    r"(?P<hex>0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?)"
    r"|(?P<dec>(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
    r"|(?P<inf>[iI][nN][fF](?:[iI][nN][iI][tT][yY])?)"
    r"|(?P<nan>[nN][aA][nN](?:\([0-9A-Za-z_]*\))?)"
    r")"
)
_MIN_NORMAL = 2.2250738585072014e-308


def _finite_value(match: re.Match, body: str) -> tuple[float, str]:
    """Value of a decimal or hexadecimal match and the digits of its mantissa."""
    if match.group("hex"):
        mantissa = match.group("hex")
        sign = "-" if body.startswith("-") else ""
        value = float.fromhex(
            sign + (mantissa if re.search(r"[pP]", mantissa) else mantissa + "p0")
        )
        return value, re.split(r"[pP]", mantissa[2:])[0]
    return float(body.lstrip("+")), re.split(r"[eE]", match.group("dec"))[0]


def _out_of_range(value: float, digits: str) -> bool:
    """``strtod``'s ERANGE: overflow, subnormal results, or a nonzero mantissa that underflows to zero."""
    if math.isinf(value) or (value != 0.0 and abs(value) < _MIN_NORMAL):
        return True
    return value == 0.0 and any(c not in "0." for c in digits)


def stod(text: str) -> float | None:
    """Value of ``String::ToDouble`` or ``None`` where DAL throws (garbage, trailing text, ERANGE)."""
    body = text.lstrip(_C_SPACE)
    match = _STOD.fullmatch(body)
    if match is None:
        return None
    if match.group("inf"):
        return -math.inf if body.startswith("-") else math.inf
    if match.group("nan"):
        return math.nan
    value, digits = _finite_value(match, body)
    return None if _out_of_range(value, digits) else value


def stod_error(text: str) -> str:
    """The message DAL reports when ``String::ToDouble(text)`` fails.

    ``std::stod`` itself throws (``what() == "stod"``) when nothing converts or
    the value is out of range; a valid prefix followed by more text fails DAL's
    own check instead."""
    body = text.lstrip(_C_SPACE)
    match = _STOD.match(body)
    if match is None or match.end() == len(body):
        return "stod"
    return "Not a valid number string"


def is_number(text: str) -> bool:
    return stod(text) is not None


def to_double(text: str) -> float:
    value = stod(text)
    if value is None:
        raise ValueError(f"Not a valid number string: {text!r}")
    return value


def condensed(text: str) -> str:
    """``String::Condensed``: drop space, tab and underscore, upper-case ASCII."""
    return "".join(c for c in text if c not in " \t_").upper()


def equivalent(lhs: str, rhs: str) -> bool:
    """``String::Equivalent`` used by DAL enum parsers: ``rhs`` is already condensed."""
    return condensed(lhs) == rhs


def shortest_repr(value: float) -> str:
    """``DebugNumber``: the shortest of %.6g/%.9g/%.12g/%.17g that round-trips; ``null`` if not finite."""
    if not math.isfinite(value):
        return "null"
    for precision in (6, 9, 12, 17):
        text = f"{value:.{precision}g}"
        if stod(text) == value:
            return text
    return repr(value)
