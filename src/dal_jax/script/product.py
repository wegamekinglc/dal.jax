"""Event-table products, a port of DAL's ``ScriptProduct_`` / ``ScriptProductData_`` front end.

``ScriptProductData`` is the immutable input (dates/definitions column, event
texts, settings).  ``ScriptProduct`` parses it: preprocessing, one parsed event
per date, then optionally the partition into past and future events at an
evaluation date and variable numbering.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from dal_jax.dates.date import Date
from dal_jax.errors import script_error
from dal_jax.script import ast as A
from dal_jax.script.lexer import SourceOrigin
from dal_jax.script.parser import Parser
from dal_jax.script.passes.varindex import VarTable, index_variables
from dal_jax.script.preprocessor import Cell, Preprocessor
from dal_jax.strings import ci_eq


@dataclass(frozen=True, slots=True, kw_only=True)
class ScriptProductSettings:
    """DAL's ``ScriptProductSettings_``: default index for ``SPOT()`` and LSMC regression features."""

    default_index: str = ""
    regression_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "regression_features", tuple(self.regression_features))


@dataclass(frozen=True, slots=True)
class ScriptProductData:
    """The event table: ``dates[i]`` is a :class:`Date` (event) or a definition/schedule string."""

    dates: tuple[Cell, ...]
    events: tuple[str, ...]
    settings: ScriptProductSettings = field(default_factory=ScriptProductSettings)
    name: str = "ScriptProductData_"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dates", tuple(self.dates))
        object.__setattr__(self, "events", tuple(self.events))
        if len(self.dates) != len(self.events):
            raise script_error(
                f"InvalidSetting: dates.size={len(self.dates)}; events.size={len(self.events)}; expected equal lengths"
            )

    def product(self) -> "ScriptProduct":
        return ScriptProduct(list(zip(self.dates, self.events)))


class ScriptProduct:
    def __init__(self, rows: Sequence[tuple[Cell, str]], payoff: str = "") -> None:
        self.payoff = payoff
        self.payoff_index = -1
        self.has_pays = False
        self.has_exercise = False
        self.preparation_error = ""
        self.evaluation_date: Date | None = None
        self.parsed_event_dates: list[Date] = []
        self.parsed_event_sources: list[list[SourceOrigin]] = []
        self.past_event_dates: list[Date] = []
        self.past_events: list[A.Event] = []
        self.event_dates: list[Date] = []
        self.events: list[A.Event] = []
        self.vars = VarTable()
        self._parse_events(rows)

    def _parse_events(self, rows: Sequence[tuple[Cell, str]]) -> None:
        preprocessed = Preprocessor().process(rows)
        parser = Parser(preprocessed.const_variables, preprocessed.numeric_vectors)
        for date, text in preprocessed.events.items():
            sources = preprocessed.sources[date]
            event = parser.parse(text, sources)
            if not self.preparation_error:
                self.preparation_error = parser.preparation_error
            self.has_pays = self.has_pays or parser.has_pays
            self.has_exercise = self.has_exercise or parser.has_exercise
            self.parsed_event_dates.append(date)
            self.parsed_event_sources.append(sources)
            self.event_dates.append(date)
            self.events.append(event)

    @property
    def has_payoff(self) -> bool:
        return self.has_pays or self.has_exercise

    def partition_events(self, evaluation_date: Date) -> None:
        """Events strictly before ``evaluation_date`` become past events."""
        if self.evaluation_date is not None:
            raise script_error("script events are already partitioned")
        future_dates, future_events = [], []
        for date, event in zip(self.event_dates, self.events):
            if date < evaluation_date:
                self.past_event_dates.append(date)
                self.past_events.append(event)
            else:
                future_dates.append(date)
                future_events.append(event)
        self.event_dates, self.events = future_dates, future_events
        self.evaluation_date = evaluation_date

    def index_variables(self) -> None:
        (self.past_events, self.events), self.vars = index_variables(
            [self.past_events, self.events]
        )
        names = self.vars.var_names
        self.payoff_index = next(
            (i for i, name in enumerate(names) if ci_eq(name, self.payoff)), -1
        )
        #  The default receiver is the last variable; an EXERCISE-only product has none.
        if self.payoff_index == -1 and names and (self.has_pays or not self.has_exercise):
            self.payoff_index = len(names) - 1

    def statements(self, past: bool = True, future: bool = True):
        if past:
            yield from (statement for event in self.past_events for statement in event)
        if future:
            yield from (statement for event in self.events for statement in event)
