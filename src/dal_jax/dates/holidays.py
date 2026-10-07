"""Holiday calendars and business-day adjustment (``dal/time/holidays.cpp``).

Built-in centers mirror DAL's ``Calendars_::Init``: ``CN.SSE``, ``CN.IB``
(SSE holidays without 2024-02-09, plus working weekends) and ``TARGET``
(2000-2100).  ``Holidays("A B")`` combines centers; ``Holidays("")`` has none.
"""

import bisect
import functools
from dataclasses import dataclass

from dal_jax.dates.calendar_data import CN_IB_WORK_WEEKENDS, CN_SSE_HOLIDAYS
from dal_jax.dates.date import Date
from dal_jax.errors import InvalidDate
from dal_jax.strings import ci_key, equivalent


def _easter_sunday(year: int) -> Date:
    """Meeus/Jones/Butcher Gregorian computus."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    return Date.ymd(year, (h + l - 7 * m + 114) // 31, ((h + l - 7 * m + 114) % 31) + 1)


def _target_holidays(start: int, end: int) -> tuple[Date, ...]:
    days = set()
    for y in range(start, end + 1):
        easter = _easter_sunday(y)
        for d in (Date.ymd(y, 1, 1), easter.add_days(-2), easter.add_days(1), Date.ymd(y, 5, 1), Date.ymd(y, 12, 25), Date.ymd(y, 12, 26)):
            if not d.is_weekend():
                days.add(d)
    return tuple(sorted(days))


@functools.cache
def _centers() -> dict[str, tuple[tuple[Date, ...], tuple[Date, ...]]]:
    sse = tuple(Date.ymd(*ymd) for ymd in CN_SSE_HOLIDAYS)
    ib = tuple(d for d in sse if d != Date.ymd(2024, 2, 9))
    return {
        ci_key("CN.SSE"): (sse, ()),
        ci_key("CN.IB"): (ib, tuple(Date.ymd(*ymd) for ymd in CN_IB_WORK_WEEKENDS)),
        ci_key("TARGET"): (_target_holidays(2000, 2100), ()),
    }


_CENTER_NAMES = {ci_key(name): name for name in ("CN.SSE", "CN.IB", "TARGET")}


def _contains(sorted_dates: tuple[Date, ...], date: Date) -> bool:
    i = bisect.bisect_left(sorted_dates, date)
    return i < len(sorted_dates) and sorted_dates[i] == date


@dataclass(frozen=True, slots=True)
class Holidays:
    """``Holidays_``: space-separated centers, deduplicated and sorted like DAL's ``Unique``."""

    name: str = ""

    def __post_init__(self) -> None:
        centers = sorted({ci_key(c): c for c in self.name.split(" ") if c}.items())
        for folded, original in centers:
            if folded not in _centers():
                raise InvalidDate("Invalid holiday center")
        object.__setattr__(self, "name", " ".join(_CENTER_NAMES[folded] for folded, _ in centers))

    def _parts(self):
        return [_centers()[ci_key(c)] for c in self.name.split(" ") if c]

    def is_holiday(self, date: Date) -> bool:
        return any(_contains(holidays, date) for holidays, _ in self._parts())

    def is_work_weekend(self, date: Date) -> bool:
        return any(_contains(weekends, date) for _, weekends in self._parts())

    def is_business_day(self, date: Date) -> bool:
        return self.is_work_weekend(date) or (not date.is_weekend() and not self.is_holiday(date))

    def next_bus(self, date: Date) -> Date:
        while not self.is_business_day(date):
            date = date.add_days(1)
        return date

    def prev_bus(self, date: Date) -> Date:
        while not self.is_business_day(date):
            date = date.add_days(-1)
        return date


NO_HOLIDAYS = Holidays("")

BIZ_DAY_CONVENTIONS = {"UNADJUSTED": "Unadjusted", "FOLLOWING": "Following", "MODIFIEDFOLLOWING": "ModifiedFollowing", "PRECEDING": "Preceding"}


def biz_day_convention(text: str) -> str:
    """``BizDayConvention_``: canonical name of a convention, matched like DAL's enum parser."""
    for alias, name in BIZ_DAY_CONVENTIONS.items():
        if equivalent(text, alias):
            return name
    raise InvalidDate(f"'{text}' is not a recognizable BizDayConvention")


def adjust(holidays: Holidays, date: Date, convention: str) -> Date:
    match biz_day_convention(convention):
        case "Unadjusted":
            return date
        case "Preceding":
            return holidays.prev_bus(date)
        case "Following":
            return holidays.next_bus(date)
        case _:  # ModifiedFollowing
            following = holidays.next_bus(date)
            return following if following.month == date.month else holidays.prev_bus(date)
