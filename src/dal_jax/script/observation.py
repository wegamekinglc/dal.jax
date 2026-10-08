"""Model-aware FIX/SPOT and delayed-payment planning, before JAX tracing."""

import math
from dataclasses import dataclass, replace

from dal_jax.errors import (
    MissingDefaultIndex,
    PreparationRequired,
    UnboundHistoricalSpot,
    script_error,
)
from dal_jax.index import TRADED_RATES, parse_index
from dal_jax.models.base import SampleDef
from dal_jax.script import ast as A
from dal_jax.script.fixings import global_fixings
from dal_jax.strings import ci_key


@dataclass(frozen=True, slots=True)
class Observation:
    date: object
    sample_id: int | None = None
    value: float | None = None
    index_name: str = ""
    output_id: int | None = None
    historical: bool = False
    source: object = None


@dataclass(frozen=True, slots=True)
class ObservationPlan:
    past: tuple
    future: tuple
    observations: tuple[Observation, ...]
    sample_dates: tuple
    definitions: tuple[SampleDef, ...]
    event_to_sample: tuple[int, ...]
    event_observations: tuple[tuple[int, ...], ...]
    historical_spots: tuple[float, ...]


class _Collector:
    def __init__(self, date, valuation, default_index, spots, model):
        self.date, self.valuation, self.spots, self.model = date, valuation, spots, model
        self.default = parse_index(default_index).name if default_index else ""
        self.records = []
        self.ids = {}
        self.unbound = []
        self.discounts = {}

    def bind(self, node, event_date):
        if isinstance(node, A.Spot):
            if self.default:
                node = A.Fix(
                    literal=self.default,
                    canonical=self.default,
                    fixing_date=None,
                    source=node.source,
                )
            else:
                return self._spot(node, event_date)
        if isinstance(node, A.Fix):
            return self._fix(node, event_date)
        if isinstance(node, A.Pays):
            node = self._payment(node, event_date)
        return node.with_args(tuple(self.bind(arg, event_date) for arg in node.args))

    def _spot(self, node, date):
        historical = self.valuation.historical(date, self.date)
        self.unbound.append((node, date, historical))
        value = None
        if historical:
            if date not in self.spots:
                raise UnboundHistoricalSpot(
                    f"SPOT() requires a historical observation; event={date}; {node.source.describe()}"
                )
            value = float(self.spots[date])
            if not math.isfinite(value):
                raise script_error(f"MissingFixing: non-finite historical SPOT; event={date}")
        return replace(node, observation_id=self._request("", date, historical, value, node.source))

    def _request(self, name, date, historical, value=None, source=None):
        key = ci_key(name), date
        if key not in self.ids:
            self.ids[key] = len(self.records)
            self.records.append(
                Observation(
                    date, value=value, index_name=name, historical=historical, source=source
                )
            )
        return self.ids[key]

    def _fix(self, node, event_date):
        fixing_date = node.fixing_date or event_date
        context = f"index={node.literal}; canonical={node.canonical}; fixing={fixing_date} 00:00:00; {node.source.describe()}"
        if fixing_date > event_date:
            raise script_error(f"LookAheadObservation: {context}")
        historical = self.valuation.historical(fixing_date, self.date)
        index = parse_index(node.canonical)
        if historical and not _historical_index(index):
            raise script_error(f"UnsupportedHistoricalIndex: {context}")
        return replace(
            node,
            observation_id=self._request(
                node.canonical, fixing_date, historical, source=node.source
            ),
        )

    def _payment(self, node, event_date):
        payment = node.payment_date
        if payment is None:
            return node
        if payment < event_date:
            raise script_error(
                f"InvalidPaymentDate: payment {payment} precedes its event {event_date}; {node.source.describe()}"
            )
        if payment == event_date:
            return replace(node, payment_date=None)
        if event_date < self.date:
            if payment >= self.date:
                raise script_error(
                    f"UnsettledDelayedPayment: event {event_date}; payment {payment}; evaluation date {self.date}; {node.source.describe()}"
                )
            return replace(node, payment_date=None)
        mats = self.discounts.setdefault(event_date, [])
        maturity = (payment - self.date) / 365.0
        if maturity not in mats:
            mats.append(maturity)
        return replace(node, discount_id=mats.index(maturity))

    def _validate_plain_spots(self):
        has_fix = any(record.index_name for record in self.records)
        ambiguous = self.model is not None and getattr(self.model, "num_assets", 1) > 1
        if self.unbound and (has_fix or ambiguous):
            raise MissingDefaultIndex(
                "SPOT() requires product.default_index when combined with FIX or a multi-asset model"
            )

    def _live_names(self):
        return tuple(
            dict.fromkeys(
                ci_key(r.index_name) for r in self.records if r.index_name and not r.historical
            )
        )

    def validate(self):
        self._validate_plain_spots()
        live_names = self._live_names()
        if live_names and self.model is None:
            raise PreparationRequired("FIX requires model-aware preparation; pass model=...")
        _validate_model(self.model, live_names, self.discounts)

    def resolve_history(self, expired):
        if expired:
            return
        snapshot = global_fixings() if self.valuation.fixings is None else self.valuation.fixings
        source = "GlobalSnapshot" if self.valuation.fixings is None else "ExplicitSnapshot"
        for i, record in enumerate(self.records):
            if not record.historical or not record.index_name:
                continue
            value = snapshot.find(record.index_name, record.date)
            if value is None:
                raise script_error(
                    f"MissingFixing: index={record.index_name}; fixing={record.date} 00:00:00; "
                    f"source={source}; {record.source.describe()}; exact historical fixing required; no model fallback"
                )
            self.records[i] = replace(record, value=value)


def _historical_index(index):
    if index.kind in ("EQ", "FX"):
        return True
    return index.name.startswith("IR:") and index.name.split(",")[1] in {
        name for _, name in TRADED_RATES
    }


def _validate_model(model, names, discounts):
    if model is not None:
        if len(names) > model.max_observed_indices:
            raise script_error(
                f"MultipleModelIndices: first={names[0]}; expected no more model-observed indices than the model supports"
            )
        for name in names:
            if not model.supports_index(parse_index(name)):
                raise script_error(
                    f"UnsupportedModelObservation: expected a model-supported plain EQ; index={name}"
                )
    if discounts and (model is None or not model.supports_discount_factors):
        raise script_error(
            "UnsupportedDelayedPayment: PAYS ... ON requires a model providing discount factors"
        )


def _sample_records(collector, sample_ids, names):
    for i, record in enumerate(collector.records):
        if record.historical:
            continue
        sample = sample_ids[record.date]
        output = None
        if record.index_name:
            output = len(names[sample])
            names[sample].append(record.index_name)
        collector.records[i] = replace(record, sample_id=sample, output_id=output)


def _definitions(dates, names, event_dates, discounts):
    return tuple(
        SampleDef(
            numeraire=date in event_dates,
            index_names=tuple(row),
            discount_mats=tuple(discounts.get(date, ())),
        )
        for date, row in zip(dates, names)
    )


def _samples(collector, event_dates):
    dates = tuple(
        sorted(set(event_dates) | {r.date for r in collector.records if not r.historical})
    )
    sample_ids = {date: i for i, date in enumerate(dates)}
    names = [[] for _ in dates]
    _sample_records(collector, sample_ids, names)
    return (
        dates,
        _definitions(dates, names, event_dates, collector.discounts),
        tuple(sample_ids[date] for date in event_dates),
    )


def _bind_events(collector, dates, events):
    return tuple(
        tuple(collector.bind(node, day) for node in event) for day, event in zip(dates, events)
    )


def _event_requests(future):
    def requests(event):
        return tuple(
            dict.fromkeys(
                node.observation_id
                for statement in event
                for node in A.walk(statement)
                if isinstance(node, A.Fix)
            )
        )

    return tuple(requests(event) for event in future)


def _past_spots(records, dates):
    replay = {r.date: r.value for r in records if r.historical and not r.index_name}
    return tuple(replay.get(day, 0.0) for day in dates)


def bind_observations(product, data, date, valuation, spots, model):
    collector = _Collector(date, valuation, data.settings.default_index, spots, model)
    past = _bind_events(collector, product.past_event_dates, product.past_events)
    future = _bind_events(collector, product.event_dates, product.events)
    collector.validate()
    collector.resolve_history(not future)
    dates, defs, event_samples = _samples(collector, product.event_dates)
    return ObservationPlan(
        past,
        future,
        tuple(collector.records),
        dates,
        defs,
        event_samples,
        _event_requests(future),
        _past_spots(collector.records, product.past_event_dates),
    )


def local_observations(events, event_indices):
    def remap(node, indices):
        if isinstance(node, A.Fix):
            return replace(node, observation_id=indices.index(node.observation_id))
        return node.with_args(tuple(remap(arg, indices) for arg in node.args))

    return tuple(
        tuple(remap(node, indices) for node in event)
        for event, indices in zip(events, event_indices)
    )


def compact_lsmc_outputs(plan, events, features):
    """Drop dead output slots after validating requests; preserve dates/Sobol dimensions."""
    from dal_jax.script.lsmcprep import bind_regression_outputs

    ids = _event_requests(events)
    live = {index for row in ids for index in row}
    records = list(plan.observations)
    definitions = [replace(definition, index_names=()) for definition in plan.definitions]
    _compact_outputs(records, definitions, live)
    definitions, features = bind_regression_outputs(
        events, plan.event_to_sample, definitions, features
    )
    return replace(
        plan, observations=tuple(records), definitions=definitions, event_observations=ids
    ), features


def _compact_outputs(records, definitions, live):
    for i, record in enumerate(records):
        if record.historical or not record.index_name:
            continue
        if i not in live:
            records[i] = replace(record, sample_id=None, output_id=None)
            continue
        sample = record.sample_id
        names = definitions[sample].index_names
        records[i] = replace(record, output_id=len(names))
        definitions[sample] = replace(definitions[sample], index_names=names + (record.index_name,))
