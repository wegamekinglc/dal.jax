"""Prepare scalar scripts for exact and fuzzy Monte Carlo valuation.

The input event table stays immutable.  Preparation partitions dates, binds
SPOT observations and replays history on the host. Exact events additionally
use domain/condition folding; fuzzy events retain continuous comparisons,
following DAL's model-aware preparation. Both lower to grouped JAX events.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import jax.numpy as jnp

from dal_jax.dates import Date
from dal_jax.errors import InvalidSetting, InvalidScriptStructure, MissingFixing, UnboundHistoricalSpot, UnsupportedExecutionMode, script_error
from dal_jax.mc.engine import PathProduct
from dal_jax.models.base import Sample, SampleDef
from dal_jax.script import ast as A
from dal_jax.script.lower.exact import lower_event, replay_events
from dal_jax.script.lower.events import lower_events
from dal_jax.script.passes.constcond import process_const_conditions
from dal_jax.script.passes.constfold import ConstProcessor
from dal_jax.script.passes.domain import DomainProcessor
from dal_jax.script.passes.ifmeta import process_ifs
from dal_jax.script.passes.eventgroup import group_events
from dal_jax.script.passes.intervals import Domain
from dal_jax.script.passes.varindex import VarTable
from dal_jax.script.product import ScriptProductData

_VECTOR_NODES = (A.VectorEntry, A.VectorReduce, A.VectorAssign, A.VectorAppend)


@dataclass(frozen=True, slots=True)
class SpotObservation:
    """One shared SPOT observation per event date, historical or simulated."""

    date: Date
    sample_id: int | None
    value: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedProduct:
    evaluation_date: Date
    event_dates: tuple[Date, ...]
    events: tuple[A.Event, ...]
    fuzzy_events: tuple[A.Event, ...]
    past_event_dates: tuple[Date, ...]
    past_events: tuple[A.Event, ...]
    timeline: tuple[float, ...]
    sample_defs: tuple[SampleDef, ...]
    variables: VarTable
    payoff_index: int
    observations: tuple[SpotObservation, ...]
    initial_values: tuple[float, ...]
    historical_spots: tuple[float, ...]
    max_nested_ifs: int

    @property
    def script_params(self) -> tuple[tuple[str, float], ...]:
        return tuple(zip(self.variables.const_names, self.variables.const_values))

    @property
    def expired(self) -> bool:
        return not self.events

    def event_groups(self, *, fuzzy: bool = False, threshold: int = 4):
        """Static event templates, per-event constants and scan spans for diagnostics."""
        return group_events(self.fuzzy_events if fuzzy else self.events, threshold)

    def path_product(self) -> PathProduct:
        """Lower the prepared events to a pure single-path payoff for P0's engine."""
        history = tuple(lower_event(event, self.variables.const_names, historical=True) for event in self.past_events)
        live_history = any(isinstance(node, A.ConstVar) for event in self.past_events for statement in event for node in A.walk(statement))

        def initial_state(params):
            state = jnp.asarray(self.initial_values, dtype=jnp.float64)
            if live_history:
                state = jnp.zeros_like(state)
                for event, spot in zip(history, self.historical_spots):
                    sample = Sample(jnp.asarray(spot, state.dtype), jnp.asarray(1.0, state.dtype), jnp.empty(0), jnp.empty(0))
                    state = event(state, sample, params["script"])
            return state

        def payoff(params, scenario, ctx, initial=None):
            state = initial_state(params) if initial is None else initial
            state = state.astype(scenario.spot.dtype)
            events = self.fuzzy_events if ctx.fuzzy else self.events
            state = lower_events(events, self.variables.const_names, ctx)(state, scenario, params["script"])
            return state[self.payoff_index]

        return PathProduct(timeline=self.timeline, payoff=payoff, payoff_names=(self.variables.var_names[self.payoff_index],),
                           sample_defs=self.sample_defs, script_params=self.script_params, initial_state=initial_state)


def _validate_node(node: A.Node) -> None:
    if isinstance(node, A.Fix):
        raise script_error(node.preparation_error())
    if isinstance(node, _VECTOR_NODES):
        raise UnsupportedExecutionMode("vector evaluation requires P5")
    if isinstance(node, A.Exercise):
        raise UnsupportedExecutionMode("EXERCISE statements require the LSMC simulation driver")
    if isinstance(node, A.Pays) and node.payment_date is not None and node.payment_date != node.source.event_date:
        raise script_error(f"PreparationRequired: PAYS ... ON {node.payment_date} requires model-aware preparation; {node.source.describe()}")


def _bind_event(event: A.Event, date: Date, sample_id: int | None, observations: list[SpotObservation],
                historical_spots: Mapping[Date, float]) -> A.Event:
    uses = [node for statement in event for node in A.walk(statement) if isinstance(node, A.Spot)]
    observation_id = None
    if uses:
        value = _historical_spot(date, uses[0], historical_spots) if sample_id is None else None
        observation_id = len(observations)
        observations.append(SpotObservation(date, sample_id, value))

    def bind(node):
        if isinstance(node, A.Spot):
            return replace(node, observation_id=observation_id)
        if isinstance(node, A.Pays):
            node = replace(node, payment_date=None)
        return node.with_args(tuple(bind(arg) for arg in node.args))

    return tuple(bind(statement) for statement in event)


def _historical_spot(date: Date, use: A.Spot, spots: Mapping[Date, float]) -> float:
    if date not in spots:
        raise UnboundHistoricalSpot(f"SPOT() requires a historical observation; event={date}; {use.source.describe()}")
    value = float(spots[date])
    if not math.isfinite(value):
        raise MissingFixing(f"non-finite historical SPOT; event={date}")
    return value


class _ScalarDomains(DomainProcessor):
    """Discard past payments and keep invalid arithmetic domains conservative."""

    historical = True

    def __init__(self, n_vars: int, known_observations) -> None:
        super().__init__(n_vars, fuzzy=False, known_observations=known_observations)
        self._handlers[A.Div] = lambda node: self._binary(node, _division_domain)

    def visit(self, node: A.Node) -> A.Node:
        # Literal NaNs/infinities in an unused branch must survive analysis.
        # They cannot participate in DAL's tolerance-based interval algebra.
        if isinstance(node, A.Expr) and node.is_const and not math.isfinite(node.const_val):
            return self._unknown(node)
        return super().visit(node)

    def _pays(self, node: A.Pays) -> A.Node:
        if not self.historical:
            return super()._pays(node)
        value = self.visit(node.args[1])
        self.doms.pop()
        return node.with_args((node.args[0], value))


def _division_domain(left: Domain, right: Domain) -> Domain:
    # A zero denominator may belong to an unused branch.  Keep its domain
    # unknown and let exact evaluation decide whether the path is valid.
    return Domain.real() if right.is_constant and right.can_be_zero() else left / right


def _process(events, processor):
    return tuple(tuple(processor.visit(statement) for statement in event) for event in events)


def _analyse(past, future, n_vars: int, observations: tuple[SpotObservation, ...], *, fuzzy: bool = False):
    future, _ = process_ifs(future)
    known = {i: observation.value for i, observation in enumerate(observations) if observation.value is not None}
    constants = ConstProcessor(n_vars, known_observations=known, historical=True)
    past = _process(past, constants)
    constants.start_future()
    future = _process(future, constants)
    # Model-aware DAL preparation retains parsed branches and continuous
    # fuzzy kernels. Discrete domains belong to the legacy pass pipeline.
    if fuzzy:
        future, depth = process_ifs(future)
        return tuple(future), depth
    domains = _ScalarDomains(n_vars, known_observations=known)
    _process(past, domains)
    domains.historical = False
    future = _process(future, domains)
    future = process_const_conditions(future)
    future, depth = process_ifs(future)
    return tuple(future), depth


def _parse_product(data: ScriptProductData, evaluation_date: Date):
    if not isinstance(evaluation_date, Date):
        raise InvalidSetting("evaluation_date must be a dal_jax Date")
    product = data.product()
    if not product.events:
        raise InvalidScriptStructure("script has no dated events")
    if not product.has_payoff:
        raise InvalidScriptStructure("dates/events has no PAYS payoff")
    for statement in product.statements():
        for node in A.walk(statement):
            _validate_node(node)
    product.partition_events(evaluation_date)
    product.index_variables()
    return product


def _bind_events(product, spots):
    observations: list[SpotObservation] = []
    past = tuple(_bind_event(event, date, None, observations, spots) for date, event in zip(product.past_event_dates, product.past_events))
    future = tuple(_bind_event(event, date, i, observations, spots) for i, (date, event) in enumerate(zip(product.event_dates, product.events)))
    observed = {observation.date: observation.value for observation in observations if observation.value is not None}
    replay_spots = tuple(observed.get(date, 0.0) for date in product.past_event_dates)
    return past, future, tuple(observations), replay_spots


def prepare(data: ScriptProductData, evaluation_date: Date, *, historical_spots: Mapping[Date, float] | None = None) -> PreparedProduct:
    """Prepare scalar SPOT scripts; explicit historical spots are keyed by event date.

    FIX, mutable vectors, delayed payments and EXERCISE keep their front-end
    support but require later milestones for valuation.
    """
    product = _parse_product(data, evaluation_date)
    past, future, observations, replay_spots = _bind_events(product, {} if historical_spots is None else historical_spots)
    annotated, _ = process_ifs(past)
    past = tuple(annotated)
    table = product.vars
    initial = replay_events(past, replay_spots, len(table.var_names), table.const_names, dict(zip(table.const_names, table.const_values)))
    fuzzy_future, fuzzy_depth = _analyse(past, future, len(table.var_names), observations, fuzzy=True)
    future, depth = _analyse(past, future, len(table.var_names), observations)
    return PreparedProduct(evaluation_date=evaluation_date, event_dates=tuple(product.event_dates), events=future,
                           fuzzy_events=fuzzy_future, past_event_dates=tuple(product.past_event_dates), past_events=past,
                           timeline=tuple((date - evaluation_date) / 365.0 for date in product.event_dates),
                           sample_defs=tuple(SampleDef(numeraire=True) for _ in product.event_dates), variables=table,
                           payoff_index=product.payoff_index, observations=observations, initial_values=initial,
                           historical_spots=replay_spots, max_nested_ifs=max(depth, fuzzy_depth))
