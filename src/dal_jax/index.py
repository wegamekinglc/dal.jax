"""Index names used by ``FIX(...)`` (``dal/indice``): parsing and canonical names.

``parse_index("EQ[Aapl]@2026-12-31")`` returns an index whose ``name`` is
DAL's canonical ``Index_::Name()``.  Parsers are chosen by the text before the
first ``:`` or ``[`` (case-insensitive), as DAL's parser registry does.
"""

from dataclasses import dataclass

from dal_jax.dates.date import Date
from dal_jax.dates.increment import parse_increment
from dal_jax.errors import DalError, InvalidIndex, UnknownIndex, script_error
from dal_jax.strings import ci_eq, ci_key, equivalent

CURRENCIES = ("USD", "EUR", "GBP", "JPY", "AUD", "CHF", "CAD", "CNY")
TRADED_RATES = (
    ("LIBOR3MCME", "LIBOR_3M_CME"),
    ("LIBOR3MLCH", "LIBOR_3M_LCH"),
    ("LIBOR3MFUT", "LIBOR_3M_FUT"),
    ("LIBOR6MCME", "LIBOR_6M_CME"),
    ("LIBOR6MLCH", "LIBOR_6M_LCH"),
)


class _IndexParseError(Exception):
    """A DAL ``REQUIRE`` failure inside an index parser: ``str()`` is the bare message."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _IndexParseError(message)


def currency(text: str) -> str:
    for code in CURRENCIES:
        if equivalent(text, code):
            return code
    raise _IndexParseError(f"'{text}' is not a recognizable Ccy")


def traded_rate(text: str) -> str:
    for alias, name in TRADED_RATES:
        if equivalent(text, alias):
            return name
    raise _IndexParseError(f"'{text}' is not a recognizable TradedRate")


def _valid_increment(text: str) -> bool:
    try:
        parse_increment(text)
    except DalError as error:
        raise _IndexParseError(error.detail) from error
    return True


def _is_libor_tenor(tenor: str) -> bool:
    return "y" not in tenor.lower()


@dataclass(frozen=True, slots=True)
class Index:
    """A parsed index: ``kind`` is EQ, FX or IR; ``name`` is DAL's canonical name."""

    kind: str
    name: str


# --- equity -----------------------------------------------------------------------------


def _equity_parts(name: str) -> tuple[str, str]:
    """The name inside ``EQ[...]`` and the text after the bracket."""
    stop = name.find("]")
    _require(
        ci_eq(name[:3], "EQ[") and stop != -1 and stop > 3,
        "equity index must contain a nonempty EQ[name]",
    )
    _require(
        not any(c in name[3:stop] for c in "[]\"'\r\n") and name.find("]", stop + 1) == -1,
        "malformed equity index brackets or name",
    )
    return name[3:stop], name[stop + 1 :]


def _equity_increment(increment: str) -> str:
    _require(
        bool(increment) and increment[0] != "&" and increment[-1] != "&" and "&&" not in increment,
        "equity delivery increment contains an empty component",
    )
    _valid_increment(increment)
    return f">{increment}"


def _equity_delivery(tail: str) -> str:
    """Canonical ``@date`` / ``>increment`` delivery suffix."""
    if not tail:
        return ""
    if tail[0] == ">":
        return _equity_increment(tail[1:])
    _require(tail[0] == "@", "unexpected trailing equity index characters")
    try:
        return f"@{Date.from_string(tail[1:])}"
    except DalError as error:
        raise _IndexParseError(error.detail) from error


def _equity(name: str) -> Index:
    try:
        eq_name, tail = _equity_parts(name)
        return Index("EQ", f"EQ[{eq_name}]{_equity_delivery(tail)}")
    except _IndexParseError as error:
        raise _IndexParseError(f"InvalidIndex: {name}; {error}") from error


# --- FX -----------------------------------------------------------------------------------


def _fx(name: str) -> Index:
    start, stop, sep = name.find("["), name.find("]"), name.find("/")
    brackets = (
        ci_eq(name[:3], "FX[")
        and start == 2
        and stop + 1 == len(name)
        and name.find("[", start + 1) == -1
    )
    separator = start + 1 < sep < stop - 1 and name.find("/", sep + 1) == -1
    _require(
        brackets and separator, "InvalidIndex: FX requires the complete FX[foreign/domestic] name"
    )
    foreign, domestic = currency(name[start + 1 : sep]), currency(name[sep + 1 : stop])
    return Index("FX", f"FX[{foreign}/{domestic}]")


# --- IR -------------------------------------------------------------------------------------


def _parts(body: str) -> list[str]:
    parts = body.split(",")
    _require(all(parts), "InvalidIndex: empty IR index field")
    return parts


def _date_or_increment(value: str) -> str:
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        try:
            return str(Date.from_string(value))
        except DalError as error:
            raise _IndexParseError(error.detail) from error
    _valid_increment(value)
    return value


def _discount(values: list[str]) -> Index:
    _require(
        len(values) in (2, 3),
        "InvalidIndex: IR discount requires currency, maturity and optional start",
    )
    ccy = currency(values[0])
    if len(values) == 2:
        return Index("IR", f"IR[DF]:{ccy},{_date_or_increment(values[1])}")
    start = _date_or_increment(values[1])
    return Index("IR", f"IR[DF]:{ccy},{start},{_date_or_increment(values[2])}")


def _rate(values: list[str]) -> Index:
    _require(
        len(values) in (2, 3), "InvalidIndex: IR rate requires currency, tenor and optional start"
    )
    ccy, tenor = currency(values[0]), values[1]
    start = f",{_date_or_increment(values[2])}" if len(values) == 3 else ""
    if not _is_libor_tenor(tenor):
        return Index("IR", f"IR:{ccy},{tenor}{start}")
    return Index("IR", f"IR:{ccy},{traded_rate(tenor)}{start}")


def _bracketed(values: list[str]) -> Index:
    _require(len(values) >= 2, "InvalidIndex: incomplete IR index")
    if ci_eq(values[1], "DF"):
        _require(
            len(values) in (3, 4), "InvalidIndex: discount requires maturity and optional start"
        )
        return _discount([values[0]] + ([values[3]] if len(values) == 4 else []) + [values[2]])
    if ci_eq(values[1], "SWAP"):
        _require(len(values) in (3, 4), "InvalidIndex: swap requires a tenor and optional start")
        _valid_increment(values[2])
        ccy, tenor = currency(values[0]), values[2]
        start = f",{_date_or_increment(values[3])}" if len(values) == 4 else ""
        return Index(
            "IR",
            f"IR:{ccy},{tenor}{start}"
            if not _is_libor_tenor(tenor)
            else f"IR[{ccy},SWAP,{tenor}{start}]",
        )
    return _rate(values)


def _ir(name: str) -> Index:
    if ci_eq(name[:7], "IR[DF]:"):
        return _discount(_parts(name[7:]))
    if ci_eq(name[:3], "IR:"):
        return _rate(_parts(name[3:]))
    _require(
        ci_eq(name[:3], "IR[") and name.endswith("]"),
        "InvalidIndex: expected IR[currency,DF,maturity], IR[currency,rate] or a canonical IR name",
    )
    return _bracketed(_parts(name[3:-1]))


_PARSERS = {ci_key("EQ"): _equity, ci_key("FX"): _fx, ci_key("IR"): _ir}


def parse_index(name: str) -> Index:
    """``Index::Parse``; failures raise :class:`InvalidIndex` / :class:`UnknownIndex` with DAL's text."""
    stops = [i for i in (name.find(":"), name.find("[")) if i != -1]
    if not stops:
        raise InvalidIndex(f"no index parsed from '{name}'")
    parser = _PARSERS.get(ci_key(name[: min(stops)]))
    if parser is None:
        raise UnknownIndex(f"no parser for '{name}'")
    try:
        return parser(name)
    except _IndexParseError as error:
        raise script_error(str(error)) from error
