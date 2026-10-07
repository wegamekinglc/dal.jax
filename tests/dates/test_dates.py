"""Dates, increments, calendars, schedules and day counts (DAL ``dal/time`` subset)."""

import datetime as dt

import pytest

from dal_jax.dates import Date, DayBasis, Holidays, adjust, date_generate, make_schedule, parse_increment
from dal_jax.dates.date import datetime_string
from dal_jax.errors import InvalidDate


def test_range_parsing_and_formatting():
    assert str(Date.minimum()) == "1970-01-01" and str(Date.maximum()) == "2149-06-05"
    assert Date.from_string("2022-09-15") == Date.ymd(2022, 9, 15) == Date.from_string("09/15/22") == Date.from_string("9/15/2022")
    assert Date.ymd(2022, 9, 15).excel == 44819
    assert Date.from_excel(25568) is None and Date.from_excel(44819) == Date.ymd(2022, 9, 15)
    for bad in ("2022-13-01", "2022-02-30", "1969-12-31", "2150-01-01", "15.09.2022"):
        with pytest.raises(InvalidDate):
            Date.from_string(bad)
    with pytest.raises(InvalidDate):
        Date.maximum().add_days(1)


def test_calendar_arithmetic():
    d = Date.ymd(2024, 1, 31)
    assert str(d.add_months(1)) == "2024-02-29" and str(d.add_months(-2)) == "2023-11-30" and str(d.add_months(14)) == "2025-03-31"
    assert str(Date.ymd(2024, 2, 29).add_months(12)) == "2025-02-28"
    assert str(Date.ymd(2023, 2, 28).add_months(1, preserve_eom=True)) == "2023-03-31"
    assert Date.ymd(2022, 9, 18).day_of_week() == 0 and Date.ymd(2022, 9, 17).is_weekend() and not Date.ymd(2022, 9, 16).is_weekend()
    assert Date.ymd(2025, 9, 14) - Date.ymd(2022, 9, 15) == 1095
    assert datetime_string(Date.ymd(2022, 10, 15)) == "2022-10-15 00:00:00"
    assert Date.from_python(dt.date(2030, 5, 6)).to_python() == dt.date(2030, 5, 6)


def test_increments():
    d = Date.ymd(2022, 9, 15)
    assert str(parse_increment("3M").fwd_from(d)) == "2022-12-15" and str(parse_increment("1y").back_from(d)) == "2021-09-15"
    assert str(parse_increment("2W").fwd_from(d)) == "2022-09-29" and str(parse_increment("10CD").fwd_from(d)) == "2022-09-25"
    assert str(parse_increment("3BD").fwd_from(Date.ymd(2022, 9, 16))) == "2022-09-21"
    assert str(parse_increment("1M;TARGET").fwd_from(Date.ymd(2022, 3, 25))) == "2022-04-25"
    assert str(parse_increment("IMM").fwd_from(d)) == "2022-09-21" and str(parse_increment("IMM").fwd_from(Date.ymd(2022, 9, 22))) == "2022-12-21"
    assert str(parse_increment("CDS").fwd_from(d)) == "2022-09-20" and str(parse_increment("EOM").fwd_from(d)) == "2022-09-30"
    assert str(parse_increment("1M&EOM").fwd_from(d)) == "2022-10-31"
    for bad in ("", "1Q", "XYZ"):
        with pytest.raises(InvalidDate):
            parse_increment(bad)


def test_holidays_and_adjustment():
    sse, ib, target = Holidays("CN.SSE"), Holidays("cn.ib"), Holidays("TARGET")
    assert sse.is_holiday(Date.ymd(2024, 2, 9)) and not ib.is_holiday(Date.ymd(2024, 2, 9))
    assert ib.is_business_day(Date.ymd(2024, 2, 4)) and not sse.is_business_day(Date.ymd(2024, 2, 4))  # working Sunday
    assert target.is_holiday(Date.ymd(2024, 3, 29)) and target.is_holiday(Date.ymd(2024, 4, 1))  # Good Friday, Easter Monday
    assert Holidays("TARGET CN.SSE").name == "CN.SSE TARGET"
    saturday = Date.ymd(2022, 4, 30)
    assert str(adjust(target, saturday, "Following")) == "2022-05-02"
    assert str(adjust(target, saturday, "ModifiedFollowing")) == "2022-04-29"
    assert str(adjust(target, saturday, "Preceding")) == "2022-04-29"
    assert adjust(target, saturday, "Unadjusted") == saturday
    with pytest.raises(InvalidDate):
        Holidays("XX")
    with pytest.raises(InvalidDate):
        adjust(target, saturday, "Nearest")


def test_schedules():
    start, end = Date.ymd(2022, 1, 31), Date.ymd(2022, 5, 31)
    forward = [str(d) for d in date_generate(start, end, parse_increment("1M"))]
    assert forward == ["2022-01-31", "2022-02-28", "2022-03-28", "2022-04-28", "2022-05-28", "2022-05-31"]  # DAL chains month steps
    backward = [str(d) for d in date_generate(start, end, parse_increment("1M"), "Backward")]
    assert backward == ["2022-01-31", "2022-02-28", "2022-03-30", "2022-04-30", "2022-05-31"]
    adjusted = make_schedule(Date.ymd(2022, 5, 7), Date.ymd(2022, 8, 7), Holidays("CN.SSE"), parse_increment("1M"), "Forward", "Following")
    assert [str(d) for d in adjusted] == ["2022-05-09", "2022-06-07", "2022-07-07", "2022-08-08"]


@pytest.mark.parametrize(
    "basis,start,end,expected",
    [
        ("ACT/365F", (2023, 4, 23), (2024, 4, 23), 366 / 365),
        ("money", (2023, 1, 1), (2023, 7, 1), 181 / 360),
        ("ACT/ACT", (2023, 7, 1), (2024, 7, 1), 184 / 365 + 182 / 366),
        ("30/360", (2023, 1, 31), (2023, 3, 31), 60 / 360),
        ("30U/360", (2023, 2, 28), (2023, 8, 31), 180 / 360),
    ],
)
def test_day_counts(basis, start, end, expected):
    assert DayBasis.parse(basis).year_fraction(Date.ymd(*start), Date.ymd(*end)) == pytest.approx(expected, rel=1e-15)


def test_day_count_errors():
    with pytest.raises(InvalidDate, match="not a recognizable DayBasis"):
        DayBasis.parse("XYZ")
    with pytest.raises(InvalidDate, match="ACT/365L"):
        DayBasis.parse("ACT/365L").year_fraction(Date.ymd(2023, 1, 1), Date.ymd(2024, 1, 1))
