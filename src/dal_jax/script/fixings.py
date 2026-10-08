"""Immutable historical fixing snapshots and valuation policy (host-side only)."""

import datetime as dt
import math
import threading
import warnings
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
    _keys: tuple[tuple[str, Date, int], ...] = field(repr=False, compare=False)

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
        object.__setattr__(self, "_keys", tuple(row[:3] for row in entries))

    def find(self, index, timestamp):
        name = ci_key(parse_index(index).name)
        date, micros = fixing_time(timestamp)
        direct = self._find((name, date, micros))
        if direct is not None:
            return direct
        reverse = _reverse_fx(name)
        value = None if reverse is None else self._find((reverse, date, micros))
        return None if value is None else 1.0 / value

    def _find(self, key):
        index = bisect_left(self._keys, key)
        return (
            self.entries[index][3] if index < len(self._keys) and self._keys[index] == key else None
        )


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ValuationContext(ValuationSettings):
    """Resolved valuation inputs, independent of session updates."""

    evaluation_date: Date = field()
    fixings: FixingSnapshot = field(default_factory=FixingSnapshot)

    def __post_init__(self):
        ValuationSettings.__post_init__(self)
        if self.evaluation_date is None or self.fixings is None:
            raise InvalidSetting(
                "a valuation context requires an evaluation date and fixing snapshot"
            )


class ValuationSession:
    """Own a replaceable valuation configuration and publish immutable snapshots."""

    def __init__(self, settings: ValuationSettings | None = None):
        if settings is not None and not isinstance(settings, ValuationSettings):
            raise InvalidSetting("session settings must be ValuationSettings")
        self._lock = threading.Lock()
        self._settings = settings or ValuationSettings()

    def snapshot(self) -> ValuationContext:
        with self._lock:
            settings = self._settings
        return ValuationContext(
            evaluation_date=settings.evaluation_date or Date.from_python(dt.date.today()),
            fixings=settings.fixings if settings.fixings is not None else FixingSnapshot(),
            today_fixing_policy=settings.today_fixing_policy,
        )

    def update(self, settings: ValuationSettings) -> None:
        if not isinstance(settings, ValuationSettings):
            raise InvalidSetting("session settings must be ValuationSettings")
        with self._lock:
            self._settings = settings

    def _set_date(self, date):
        with self._lock:
            self._settings = replace(self._settings, evaluation_date=date)

    def _set_fixings(self, snapshot):
        with self._lock:
            self._settings = replace(self._settings, fixings=snapshot)


# The deprecated DAL compatibility interface alone owns shared business state.
_LEGACY_SESSION = ValuationSession()


def legacy_context() -> ValuationContext:
    return _LEGACY_SESSION.snapshot()


def set_legacy_date(date) -> None:
    warnings.warn(
        "EvaluationDate_Set is deprecated; pass a ValuationContext",
        DeprecationWarning,
        stacklevel=3,
    )
    _LEGACY_SESSION._set_date(date)


def resolve_valuation(settings: ValuationSettings) -> ValuationContext:
    fallback = (
        legacy_context()
        if settings.evaluation_date is None or settings.fixings is None
        else settings
    )
    return ValuationContext(
        evaluation_date=settings.evaluation_date or fallback.evaluation_date,
        fixings=fallback.fixings if settings.fixings is None else settings.fixings,
        today_fixing_policy=settings.today_fixing_policy,
    )


def set_global_fixings(snapshot: FixingSnapshot):
    """Deprecated DAL-compatible fixing setter."""
    if not isinstance(snapshot, FixingSnapshot):
        raise InvalidSetting("global fixings must be a FixingSnapshot")
    warnings.warn(
        "set_global_fixings is deprecated; pass a ValuationContext",
        DeprecationWarning,
        stacklevel=2,
    )
    _LEGACY_SESSION._set_fixings(snapshot)


def global_fixings():
    return legacy_context().fixings
