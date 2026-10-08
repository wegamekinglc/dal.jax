"""Date increments (``dal/time/dateincrement.cpp``): ``"3M"``, ``"2BD;TARGET"``, ``"IMM"``, ``"1Y&EOM"``.

``parse_increment`` returns an object with ``fwd_from`` / ``back_from``.  A
multistep increment ``n<unit>[;centers]`` moves by years, months, weeks,
calendar or business days and, with holidays, rolls onto a business day; a
special day (IMM, IMM1, CDS, EOM) jumps to the next/previous such date;
``a&b`` applies its parts in order.
"""

from dataclasses import dataclass

from dal_jax.dates.date import Date
from dal_jax.dates.holidays import NO_HOLIDAYS, Holidays
from dal_jax.errors import InvalidDate
from dal_jax.strings import equivalent

#  Condensed aliases of DAL's Machinist enums, in declaration order.
_SPECIAL_DAYS = {
    "IMM": "IMM",
    "IMM3": "IMM",
    "IMMQUARTERLY": "IMM",
    "IMM1": "IMM1",
    "IMMMONTHLY": "IMM1",
    "CDS": "CDS",
    "CDS3": "CDS",
    "CDSQUARTERLY": "CDS",
    "EOM": "EOM",
}
_STEP_SIZES = {
    "Y": "Y",
    "YEAR": "Y",
    "YEARS": "Y",
    "M": "M",
    "MONTH": "M",
    "MONTHS": "M",
    "W": "W",
    "WEEK": "W",
    "WEEKS": "W",
    "BD": "BD",
    "BUSDAY": "BD",
    "BUSINESSDAY": "BD",
    "CD": "CD",
    "CALDAY": "CD",
    "CALENDARDAY": "CD",
}


def _enum(text: str, aliases: dict[str, str], what: str) -> str:
    for alias, value in aliases.items():
        if equivalent(text, alias):
            return value
    raise InvalidDate(f"'{text}' is not a recognizable {what}")


def _month_in_range(y: int, m: int) -> tuple[int, int]:
    while m > 12:
        m, y = m - 12, y + 1
    while m < 1:
        m, y = m + 12, y - 1
    return y, m


def _to_cds(date: Date, forward: bool) -> Date:
    yy, mm, dd = date.year, date.month, date.day
    if (dd >= 20) if forward else (dd <= 20):
        while True:
            mm += 1 if forward else -1
            if mm % 3 == 0:
                break
    yy, mm = _month_in_range(yy, mm)
    return Date.ymd(yy, mm, 20)


def _imm_day(yy: int, mm: int) -> int:
    return 21 - Date.ymd(yy, mm, 18).day_of_week()


def _to_imm(date: Date, forward: bool, imm_months: int) -> Date:
    yy, mm, dd = date.year, date.month, date.day
    if (dd >= _imm_day(yy, mm)) if forward else (dd <= _imm_day(yy, mm)):
        while True:
            mm += 1 if forward else -1
            if mm % imm_months == 0:
                break
    yy, mm = _month_in_range(yy, mm)
    return Date.ymd(yy, mm, _imm_day(yy, mm))


@dataclass(frozen=True, slots=True)
class SpecialDay:
    kind: str  # IMM, IMM1, CDS, EOM

    def _step(self, date: Date, forward: bool) -> Date:
        match self.kind:
            case "CDS":
                return _to_cds(date, forward)
            case "IMM":
                return _to_imm(date, forward, 3)
            case "IMM1":
                return _to_imm(date, forward, 1)
            case _:
                return (
                    date.end_of_month()
                    if forward
                    else Date.ymd(date.year, date.month, 1).add_days(-1)
                )

    def fwd_from(self, date: Date) -> Date:
        return self._step(date, True)

    def back_from(self, date: Date) -> Date:
        return self._step(date, False)


_CALENDAR_STEPS = {
    "Y": lambda date, n: date.add_months(12 * n),
    "M": lambda date, n: date.add_months(n),
    "W": lambda date, n: date.add_days(7 * n),
    "CD": lambda date, n: date.add_days(n),
}


@dataclass(frozen=True, slots=True)
class Multistep:
    n: int
    unit: str  # Y, M, W, BD, CD
    holidays: Holidays = NO_HOLIDAYS

    def _business_days(self, date: Date, forward: bool) -> Date:
        for _ in range(self.n):
            date = (
                self.holidays.next_bus(date.add_days(1))
                if forward
                else self.holidays.prev_bus(date.add_days(-1))
            )
        return date

    def _step(self, date: Date, forward: bool) -> Date:
        sign = 1 if forward else -1
        if self.unit == "BD":
            date = self._business_days(date, forward)
        else:
            date = _CALENDAR_STEPS[self.unit](date, self.n * sign)
        if self.holidays == NO_HOLIDAYS:
            return date
        return self.holidays.next_bus(date) if forward else self.holidays.prev_bus(date)

    def fwd_from(self, date: Date) -> Date:
        return self._step(date, True)

    def back_from(self, date: Date) -> Date:
        return self._step(date, False)


@dataclass(frozen=True, slots=True)
class Compound:
    parts: tuple

    def fwd_from(self, date: Date) -> Date:
        for part in self.parts:
            date = part.fwd_from(date)
        return date

    def back_from(self, date: Date) -> Date:
        for part in self.parts:
            date = part.back_from(date)
        return date


type Increment = SpecialDay | Multistep | Compound
_INT_MAX = 2**31 - 1


def parse_increment(text: str) -> Increment:
    if "&" in text:
        return Compound(tuple(parse_increment(part) for part in text.split("&")))
    if not text:
        raise InvalidDate("Increment should not be empty")
    numeric = len(text) - len(text.lstrip("0123456789"))
    if numeric == 0:
        return SpecialDay(_enum(text, _SPECIAL_DAYS, "SpecialDay"))
    semicolon = text.find(";")
    holidays = Holidays(text[semicolon + 1 :]) if semicolon >= 0 else NO_HOLIDAYS
    unit = text[numeric : semicolon if semicolon >= 0 else len(text)]
    count = int(text[:numeric])
    if count > _INT_MAX:
        raise InvalidDate("stoi")  # String::ToInt is std::stoi, which rejects counts beyond int
    return Multistep(count, _enum(unit, _STEP_SIZES, "DateStepSize"), holidays)
