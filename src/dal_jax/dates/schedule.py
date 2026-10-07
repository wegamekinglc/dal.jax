"""Schedule generation (``dal/time/schedules.cpp``): ``DateGenerate`` and ``MakeSchedule``."""

from dal_jax.dates.date import Date
from dal_jax.dates.holidays import Holidays, adjust
from dal_jax.dates.increment import Increment
from dal_jax.errors import InvalidDate
from dal_jax.strings import equivalent


def date_generation(text: str) -> str:
    for name in ("Backward", "Forward"):
        if equivalent(text, name.upper()):
            return name
    raise InvalidDate(f"'{text}' is not a recognizable DateGeneration")


def date_generate(start: Date, maturity: Date, tenor: Increment, method: str = "Forward") -> list[Date]:
    """Pin dates chained from ``start`` (Forward) or ``maturity`` (Backward), both ends included."""
    if date_generation(method) == "Forward":
        dates = [start]
        while (pin := tenor.fwd_from(dates[-1])) <= maturity:
            dates.append(pin)
        if dates[-1] != maturity:
            dates.append(maturity)
        return dates
    dates = [maturity]
    while (pin := tenor.back_from(dates[-1])) >= start:
        dates.append(pin)
    if dates[-1] != start:
        dates.append(start)
    return dates[::-1]


def make_schedule(start: Date, maturity: Date, holidays: Holidays, tenor: Increment, method: str = "Forward", convention: str = "Unadjusted") -> list[Date]:
    """Adjusted pin dates, sorted and deduplicated like DAL's ``Unique``."""
    return sorted({adjust(holidays, pin, convention) for pin in date_generate(start, maturity, tenor, method)})
