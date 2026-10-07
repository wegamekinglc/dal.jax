"""DAL's ``Date_``: a day in [1970-01-01, 2149-06-05], stored as an Excel serial.

Arithmetic follows ``dal/time/date.cpp``: differences are in days, adding days
outside the range raises, ``AddMonths`` clamps to the month end (optionally
preserving an end-of-month date), and ``DayOfWeek`` is 0 for Sunday.
"""

import datetime as _dt
import re
from dataclasses import dataclass

from dal_jax.errors import InvalidDate

EXCEL_OFFSET = 25568  # Excel serial of DAL's invalid serial 0
MAX_SERIAL = 65535
_EXCEL_EPOCH = _dt.date(1899, 12, 30)
_KNOWN_SUNDAY = 974
SUPPORTED_RANGE = "Date out of supported range [1970-01-01, 2149-06-05]"

_US_FORMAT = re.compile(r"([0-9]+)/([0-9]+)/([0-9]+)")
_YMD_FORMAT = re.compile(r"([0-9]+)-([0-9]+)-([0-9]+)")


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_month(year: int, month: int) -> int:
    if not 1 <= month <= 12:
        raise InvalidDate("Month out of range [1, 12]")
    return (31, 29 if is_leap_year(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]


@dataclass(frozen=True, slots=True, order=True)
class Date:
    """Construct with :meth:`ymd`, :meth:`from_string` or :meth:`from_excel`."""

    excel: int  # Excel serial; valid dates have EXCEL_OFFSET < excel <= EXCEL_OFFSET + MAX_SERIAL

    def __post_init__(self) -> None:
        if not EXCEL_OFFSET < self.excel <= EXCEL_OFFSET + MAX_SERIAL:
            raise InvalidDate(SUPPORTED_RANGE)

    @classmethod
    def ymd(cls, year: int, month: int, day: int) -> "Date":
        if not 1970 <= year <= 2149:
            raise InvalidDate(SUPPORTED_RANGE)
        if not 1 <= month <= 12:
            raise InvalidDate("Month out of range [1, 12]")
        if not 1 <= day <= 31:
            raise InvalidDate("Day out of range [1, 31]")
        if day > days_in_month(year, month):
            raise InvalidDate("Day out of range for the given year and month")
        return cls((_dt.date(year, month, day) - _EXCEL_EPOCH).days)

    @classmethod
    def from_python(cls, value: _dt.date) -> "Date":
        return cls.ymd(value.year, value.month, value.day)

    @classmethod
    def from_excel(cls, serial: int) -> "Date | None":
        """``Date::FromExcel``: ``None`` (DAL's invalid date) outside the supported range."""
        return cls(serial) if EXCEL_OFFSET < serial <= EXCEL_OFFSET + MAX_SERIAL else None

    @classmethod
    def from_string(cls, text: str) -> "Date":
        """``Date::FromString``: ``mm/dd/yy[yy]`` or ``yyyy-mm-dd`` (any digit counts)."""
        if match := _US_FORMAT.fullmatch(text):
            mm, dd, yy = (int(g) for g in match.groups())
            return cls.ymd(yy + (2000 if len(match.group(3)) == 2 else 0), mm, dd)
        if match := _YMD_FORMAT.fullmatch(text):
            yyyy, mm, dd = (int(g) for g in match.groups())
            return cls.ymd(yyyy, mm, dd)
        raise InvalidDate("Unrecognizable date source")

    @staticmethod
    def is_date_string(text: str) -> bool:
        return bool(_US_FORMAT.fullmatch(text) or _YMD_FORMAT.fullmatch(text))

    @classmethod
    def minimum(cls) -> "Date":
        return cls(EXCEL_OFFSET + 1)

    @classmethod
    def maximum(cls) -> "Date":
        return cls(EXCEL_OFFSET + MAX_SERIAL)

    def to_python(self) -> _dt.date:
        return _EXCEL_EPOCH + _dt.timedelta(days=self.excel)

    @property
    def year(self) -> int:
        return self.to_python().year

    @property
    def month(self) -> int:
        return self.to_python().month

    @property
    def day(self) -> int:
        return self.to_python().day

    def day_of_week(self) -> int:
        """0 = Sunday ... 6 = Saturday."""
        return (self.excel - _KNOWN_SUNDAY) % 7

    def is_weekend(self) -> bool:
        return self.day_of_week() % 6 == 0

    def add_days(self, days: int) -> "Date":
        return Date(self.excel + days)

    def end_of_month(self) -> "Date":
        return Date.ymd(self.year, self.month, days_in_month(self.year, self.month))

    def add_months(self, months: int, preserve_eom: bool = False) -> "Date":
        yy, mm, dd = self.year, self.month, self.day
        to_eom = preserve_eom and dd == days_in_month(yy, mm)
        ny = int(months / 12)  # C++ integer division truncates toward zero
        yy += ny
        mm += months - 12 * ny
        if mm > 12:
            mm, yy = mm - 12, yy + 1
        if mm < 1:
            mm, yy = mm + 12, yy - 1
        d_max = days_in_month(yy, mm)
        if to_eom or dd > d_max:
            dd = d_max
        return Date.ymd(yy, mm, dd)

    def __sub__(self, other: "Date") -> int:
        return self.excel - other.excel

    def __str__(self) -> str:
        return f"{self.year:4d}-{self.month:02d}-{self.day:02d}"

    def __repr__(self) -> str:
        return f"Date({self})"


def datetime_string(date: Date, frac: float = 0.0) -> str:
    """``DateTime::ToString``: ``yyyy-mm-dd HH:MM:SS`` for a day fraction in [0, 1)."""
    h = int(24 * frac)
    m = int(1440 * frac) % 60
    s = int(86400 * frac) % 60
    return f"{date} {h:02d}:{m:02d}:{s:02d}"
