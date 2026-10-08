"""Day-count fractions (``dal/time/daybasis.cpp``)."""

from dataclasses import dataclass
from types import MappingProxyType

from dal_jax.dates.date import Date, days_in_month, is_leap_year
from dal_jax.errors import InvalidDate
from dal_jax.strings import equivalent

#  Condensed aliases of DAL's DayBasis enum, in declaration order.
_ALIASES = (
    ("ACT365F", "ACT_365F"),
    ("ACT/365F", "ACT_365F"),
    ("ACT365FIXED", "ACT_365F"),
    ("ACT/365FIXED", "ACT_365F"),
    ("ACT365L", "ACT_365L"),
    ("ACT/365L", "ACT_365L"),
    ("ISMAYEAR", "ACT_365L"),
    ("ACT360", "ACT_360"),
    ("ACT/360", "ACT_360"),
    ("MONEY", "ACT_360"),
    ("ACTUAL/360", "ACT_360"),
    ("ACTACT", "ACT_ACT"),
    ("ACT/ACT", "ACT_ACT"),
    ("ACTUAL/ACTUAL", "ACT_ACT"),
    ("BOND", "BOND"),
    ("30360", "BOND"),
    ("30/360", "BOND"),
    ("BONDBASIS", "BOND"),
    ("THIRTY360US", "THIRTY_360_US"),
    ("30360US", "THIRTY_360_US"),
    ("30U/360", "THIRTY_360_US"),
)


@dataclass(frozen=True, slots=True)
class Context:
    """``DayBasis::Context_``: coupon period information needed by ACT/365L."""

    is_last: bool
    nominal_start: Date
    nominal_end: Date
    coupon_months: int


def _days_in_year(year: int) -> float:
    return 366.0 if is_leap_year(year) else 365.0


def _act_act_isda(start: Date, end: Date) -> float:
    if end < start:
        return -_act_act_isda(end, start)
    denominator = _days_in_year(start.year)
    if end.year <= start.year:
        return (end - start) / denominator
    next_year = Date.ymd(start.year + 1, 1, 1)
    return (next_year - start) / denominator + _act_act_isda(next_year, end)


def _annual_days_l(start: Date, end: Date) -> float:
    for yy in range(start.year, end.year + 1):
        if is_leap_year(yy) and start < Date.ymd(yy, 2, 29) <= end:
            return 366.0
    return 365.0


def _thirty_360(y1: int, m1: int, d1: int, y2: int, m2: int, d2: int) -> float:
    return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.0


def _bond_basis(start: Date, end: Date) -> float:
    d1 = 30 if start.day == 31 else start.day
    d2 = 30 if end.day == 31 and d1 == 30 else end.day
    return _thirty_360(start.year, start.month, d1, end.year, end.month, d2)


def _last_of_february(date: Date) -> bool:
    return date.month == 2 and date.day == days_in_month(date.year, 2)


def _thirty_360_us(start: Date, end: Date) -> float:
    d1, d2 = start.day, end.day
    if _last_of_february(start):
        if _last_of_february(end):
            d2 = 30
        d1 = 30
    if d2 == 31 and d1 >= 30:
        d2 = 30
    if d1 == 31:
        d1 = 30
    return _thirty_360(start.year, start.month, d1, end.year, end.month, d2)


def _act_365l(start: Date, end: Date, context: Context | None) -> float:
    if context is None:
        raise InvalidDate("ACT/365L day-count requires nominal end date")
    if context.coupon_months == 12:
        return (end - start) / _annual_days_l(start, context.nominal_end)
    return (end - start) / _days_in_year(context.nominal_end.year)


_FRACTIONS = MappingProxyType(
    {
        "ACT_360": lambda start, end, _: (end - start) / 360.0,
        "ACT_365F": lambda start, end, _: (end - start) / 365.0,
        "ACT_ACT": lambda start, end, _: _act_act_isda(start, end),
        "ACT_365L": _act_365l,
        "BOND": lambda start, end, _: _bond_basis(start, end),
        "THIRTY_360_US": lambda start, end, _: _thirty_360_us(start, end),
    }
)


@dataclass(frozen=True, slots=True)
class DayBasis:
    name: str  # canonical: ACT_365F, ACT_365L, ACT_360, ACT_ACT, BOND, THIRTY_360_US

    @classmethod
    def parse(cls, text: str) -> "DayBasis":
        for alias, name in _ALIASES:
            if equivalent(text, alias):
                return cls(name)
        raise InvalidDate(f"'{text}' is not a recognizable DayBasis")

    def year_fraction(self, start: Date, end: Date, context: Context | None = None) -> float:
        return _FRACTIONS[self.name](start, end, context)
