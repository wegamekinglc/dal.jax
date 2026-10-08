"""Immutable historical fixing snapshots and valuation policy (host-side only)."""

import datetime as dt
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from dal_jax.dates import Date
from dal_jax.errors import InvalidSetting, script_error
from dal_jax.index import parse_index
from dal_jax.strings import ci_key


class TodayFixingPolicy(StrEnum):
    MODEL = "Model"
    REQUIREHISTORICAL = "RequireHistorical"


def fixing_time(value):
    """Exact timestamps; Date means midnight, datetime retains its time of day."""
    if isinstance(value, Date):
        return value, 0
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            raise InvalidSetting("fixing timestamps must have no timezone")
        micros = (
            (value.hour * 60 + value.minute) * 60 + value.second
        ) * 1_000_000 + value.microsecond
        return Date.from_python(value.date()), micros
    if isinstance(value, dt.date):
        return Date.from_python(value), 0
    raise InvalidSetting("fixing timestamp must be a Date or datetime")


def _reverse_fx(name):
    if not name.startswith("FX["):
        return None
    foreign, domestic = name[3:-1].split("/")
    return f"FX[{domestic}/{foreign}]"


def _validated_quote(name, value):
    number = float(value)
    if not math.isfinite(number):
        raise script_error(f"InvalidFixingSnapshot: expected finite value for {name}")
    if _reverse_fx(name) is not None and number <= 0:
        raise script_error(f"InvalidFixingSnapshot: expected positive FX value for {name}")
    return number


@dataclass(frozen=True, slots=True, init=False)
class FixingSnapshot:
    """Copy an index -> timestamp -> value mapping; aliases share canonical identity."""

    entries: tuple[tuple[str, Date, int, float], ...]

    def __init__(self, values: Mapping[str, Mapping] = ()):
        records = {}
        for name, history in dict(values).items():
            canonical = ci_key(parse_index(name).name)
            for timestamp, value in history.items():
                date, micros = fixing_time(timestamp)
                records[canonical, date, micros] = _validated_quote(canonical, value)
        _validate_reciprocals(records)
        entries = tuple((*key, value) for key, value in sorted(records.items()))
        object.__setattr__(self, "entries", entries)

    def find(self, index, timestamp):
        name = ci_key(parse_index(index).name)
        date, micros = fixing_time(timestamp)
        values = {(n, d, t): v for n, d, t, v in self.entries}
        direct = values.get((name, date, micros))
        if direct is not None:
            return direct
        reverse = _reverse_fx(name)
        value = values.get((reverse, date, micros))
        return None if value is None else 1.0 / value


def _validate_reciprocals(records):
    for (name, date, micros), value in records.items():
        reverse = records.get((_reverse_fx(name), date, micros))
        if reverse is not None and abs(value * reverse - 1.0) > 1e-10:
            raise script_error(
                f"InvalidFixingSnapshot: inconsistent direct/reverse FX fixings for {name} at {date}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ValuationSettings:
    evaluation_date: Date | None = None
    fixings: FixingSnapshot | None = None
    today_fixing_policy: TodayFixingPolicy = TodayFixingPolicy.MODEL

    def __post_init__(self):
        try:
            policy = TodayFixingPolicy(self.today_fixing_policy)
        except ValueError as error:
            raise InvalidSetting(
                "today_fixing_policy must be Model or RequireHistorical"
            ) from error
        object.__setattr__(self, "today_fixing_policy", policy)
        if self.evaluation_date is not None:
            date, micros = fixing_time(self.evaluation_date)
            if micros:
                raise InvalidSetting("evaluation_date must be a date")
            object.__setattr__(self, "evaluation_date", date)
        if self.fixings is not None and not isinstance(self.fixings, FixingSnapshot):
            raise InvalidSetting("fixings must be a FixingSnapshot")

    def historical(self, date, evaluation_date):
        return date < evaluation_date or (
            date == evaluation_date
            and self.today_fixing_policy == TodayFixingPolicy.REQUIREHISTORICAL
        )


_global_lock = threading.Lock()
_global_snapshot = FixingSnapshot()


def set_global_fixings(snapshot: FixingSnapshot):
    """Replace the in-process fixing snapshot; preparation captures it once."""
    if not isinstance(snapshot, FixingSnapshot):
        raise InvalidSetting("global fixings must be a FixingSnapshot")
    global _global_snapshot
    with _global_lock:
        _global_snapshot = snapshot


def global_fixings():
    with _global_lock:
        return _global_snapshot
