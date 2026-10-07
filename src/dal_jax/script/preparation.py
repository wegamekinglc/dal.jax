"""Prepare scalar/vector scripts for exact and fuzzy Monte Carlo valuation.

The input event table stays immutable.  Preparation partitions dates, binds
SPOT/FIX observations and replays history on the host. Exact events additionally
use domain/condition folding; fuzzy events retain continuous comparisons,
following DAL's model-aware preparation. Both lower to grouped JAX events.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from dal_jax.dates import Date
from dal_jax.errors import InvalidSetting, InvalidScriptStructure, UnsupportedExecutionMode
from dal_jax.mc.engine import PathProduct
from dal_jax.models.base import Sample, SampleDef
from dal_jax.script import ast as A
from dal_jax.script.lower.exact import lower_event, replay_events
from dal_jax.script.lower.events import lower_events
from dal_jax.script.lower.state import ScriptState, scalars, initial_state as empty_state
from dal_jax.script.passes.constcond import process_const_conditions
from dal_jax.script.passes.constfold import ConstProcessor
from dal_jax.script.passes.domain import DomainProcessor
from dal_jax.script.passes.ifmeta import process_ifs
from dal_jax.script.passes.eventgroup import group_events
from dal_jax.script.passes.intervals import Domain
from dal_jax.script.passes.varindex import VarTable
from dal_jax.script.product import ScriptProductData
from dal_jax.script.fixings import ValuationSettings
from dal_jax.script.observation import Observation, bind_observations, local_observations

SpotObservation = Observation


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
    initial_vectors: tuple[tuple[float, ...], ...] = ()
    initial_lengths: tuple[int, ...] = ()
    event_to_sample: tuple[int, ...] = ()
    event_observations: tuple[tuple[int, ...], ...] = ()
    historical_observations: tuple[float, ...] = ()

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
            if self.variables.vector_names:
                state = ScriptState(state, tuple(jnp.asarray(v) for v in self.initial_vectors),
                                    jnp.asarray(self.initial_lengths, dtype=jnp.int32), jnp.zeros(2*len(self.initial_vectors), dtype=bool))
            if live_history:
                state = empty_state(len(self.initial_values), self.variables.vector_capacities, jnp)
                for event, spot in zip(history, self.historical_spots):
                    sample = Sample(jnp.asarray(spot, state.dtype), jnp.asarray(1.0, state.dtype), jnp.asarray(self.historical_observations), jnp.empty(0))
                    state = event(state, sample, params["script"])
            return state

        def payoff(params, scenario, ctx, initial=None):
            state = initial_state(params) if initial is None else initial
            state = jax.tree.map(lambda x: x.astype(scenario.spot.dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, state)
            events = self.fuzzy_events if ctx.fuzzy else self.events
            samples = self._event_scenario(scenario)
            state = lower_events(events, self.variables.const_names, ctx)(state, samples, params["script"])
            value = scalars(state)[self.payoff_index]
            return jnp.concatenate((value[None], state.errors.astype(state.dtype))) if isinstance(state, ScriptState) else value

        return PathProduct(timeline=self.timeline, payoff=payoff, payoff_names=(self.variables.var_names[self.payoff_index],),
                           sample_defs=self.sample_defs, script_params=self.script_params, initial_state=initial_state,
                           path_state_size=len(self.initial_values)+sum(self.variables.vector_capacities)+3*len(self.initial_vectors),
                           error_messages=tuple(f"{code}: {name}" for name in self.variables.vector_names
                                                for code in ("VectorIndexOutOfRange", "EmptyVectorReduction")))

    def _event_scenario(self, scenario):
        from dal_jax.models.base import Scenario
        ids = jnp.asarray(self.event_to_sample, dtype=jnp.int32)
        fields = tuple(jnp.take(field, ids, axis=0) for field in (scenario.spot, scenario.numeraire, scenario.discounts))
        return Scenario(fields[0], fields[1], self._event_observations(scenario), fields[2])

    def _event_observations(self, scenario):
        width = max((len(row) for row in self.event_observations), default=0)
        shape = (len(self.events), width)
        if not width:
            return jnp.empty(shape, dtype=scenario.spot.dtype)
        samples, outputs, known, live = _observation_arrays(self.observations, self.event_observations, width)
        values = jnp.asarray(known, dtype=scenario.spot.dtype)
        if any(any(row) for row in live):
            simulated = scenario.observations[jnp.asarray(samples), jnp.asarray(outputs)]
            values = jnp.where(jnp.asarray(live), simulated, values)
        return values


def _or_zero(value):
    return 0 if value is None else value

def _observation_arrays(observations, event_ids, width):
    samples, outputs, known, live = [], [], [], []
    for ids in event_ids:
        row = [observations[i] for i in ids]
        row += [Observation(date=None, value=0., historical=True)]*(width-len(row))
        samples.append([_or_zero(r.sample_id) for r in row])
        outputs.append([_or_zero(r.output_id) for r in row])
        known.append([_or_zero(r.value) for r in row])
        live.append([not r.historical for r in row])
    return samples, outputs, known, live


def _validate_node(node: A.Node) -> None:
    if isinstance(node, A.Exercise):
        raise UnsupportedExecutionMode("EXERCISE statements require the LSMC simulation driver")


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


def _seed_metadata(initial):
    metadata = {"initial_values": tuple(float(x) for x in scalars(initial))}
    if isinstance(initial, ScriptState):
        metadata["initial_vectors"] = tuple(tuple(float(x) for x in v) for v in initial.vectors)
        metadata["initial_lengths"] = tuple(int(x) for x in initial.lengths)
    return metadata



def _with_snapshot(valuation, fixings):
    if valuation.fixings is not None and valuation.fixings is not fixings:
        raise InvalidSetting("expected one explicit fixing snapshot")
    return replace(valuation, fixings=fixings)


def _with_date(valuation, date):
    if valuation.evaluation_date is not None and date != valuation.evaluation_date:
        raise InvalidSetting("evaluation_date and valuation.evaluation_date disagree")
    return replace(valuation, evaluation_date=date)


def _valuation(evaluation_date, valuation, fixings, today_fixing_policy):
    valuation = valuation or ValuationSettings()
    if evaluation_date is not None:
        valuation = _with_date(valuation, evaluation_date)
    if fixings is not None:
        valuation = _with_snapshot(valuation, fixings)
    if today_fixing_policy is not None:
        valuation = replace(valuation, today_fixing_policy=today_fixing_policy)
    if valuation.evaluation_date is None:
        from dal_jax.api import EvaluationDate_Get
        valuation = replace(valuation, evaluation_date=EvaluationDate_Get())
    return valuation


def prepare(data: ScriptProductData, evaluation_date: Date | None = None, *, model=None, valuation=None,
            historical_spots: Mapping[Date, float] | None = None, fixings=None, today_fixing_policy=None) -> PreparedProduct:
    """Prepare vectors, observations, history and discounts; EXERCISE requires P6.

    FIX and delayed payments need ``model=...``. An immutable valuation setting
    or explicit fixing snapshot selects history; future quotes are always model
    outputs. Legacy SPOT history can also be supplied by event date.
    """
    valuation = _valuation(evaluation_date, valuation, fixings, today_fixing_policy)
    date = valuation.evaluation_date
    product = _parse_product(data, date)
    plan = bind_observations(product, data, date, valuation, historical_spots or {}, model)
    past, _ = process_ifs(plan.past)
    past = tuple(past)
    table = product.vars
    history_values = tuple(o.value if o.value is not None else 0. for o in plan.observations)
    initial = _replay_initial(past, plan, table, history_values)
    fuzzy_future, fuzzy_depth = _analyse(past, plan.future, len(table.var_names), plan.observations, fuzzy=True)
    future, depth = _analyse(past, plan.future, len(table.var_names), plan.observations)
    return PreparedProduct(evaluation_date=date, event_dates=tuple(product.event_dates),
                           events=local_observations(future, plan.event_observations),
                           fuzzy_events=local_observations(fuzzy_future, plan.event_observations),
                           past_event_dates=tuple(product.past_event_dates), past_events=past,
                           timeline=tuple((day-date)/365. for day in plan.sample_dates), sample_defs=plan.definitions, variables=table,
                           payoff_index=product.payoff_index, observations=plan.observations, **_seed_metadata(initial),
                           historical_spots=plan.historical_spots, historical_observations=history_values,
                           event_to_sample=plan.event_to_sample, event_observations=plan.event_observations,
                           max_nested_ifs=max(depth, fuzzy_depth))


def _replay_initial(past, plan, table, history_values):
    if not plan.future:
        import numpy as np
        return empty_state(len(table.var_names), table.vector_capacities, np)
    return replay_events(past, plan.historical_spots, len(table.var_names), table.const_names,
                         dict(zip(table.const_names, table.const_values)), capacities=table.vector_capacities,
                         observations=history_values)
