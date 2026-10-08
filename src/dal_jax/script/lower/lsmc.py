"""Recording exercise values, conditions, features and event-date cash flows."""

from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from dal_jax.script import ast as A
from dal_jax.script.lower.events import EventSample
from dal_jax.script.lower.exact import _Lowerer
from dal_jax.script.lower.fuzzy import _FuzzyLowerer
from dal_jax.script.lower.state import ScriptState, scalars, write
from dal_jax.script.passes.eventgroup import group_events


class Records(NamedTuple):
    payments: Array  # [events]
    exercise_values: Array  # [events], zero on non-exercise days
    conditions: Array  # [events]
    features: Array  # [events, features]
    numeraires: Array  # [events]
    errors: Array | None


# The mixin obtains evaluator methods from _Lowerer/_FuzzyLowerer through MRO.
class _Recording:  # pylint: disable=no-member
    def configure(self, n_vars, payoff_index, features, observation_width):
        self.n_vars, self.payoff_index = n_vars, payoff_index
        self.features, self.observation_width = features, observation_width
        self._statements.update({A.Exercise: self._exercise, A.Pays: self._pays})

    def _pays(self, node):
        index = node.args[0].index
        if index != self.payoff_index:
            return super()._pays(node)
        expression = self.expression(node.args[1])

        def evaluate(state, sample, params, active):
            amount = jnp.where(active, expression(state, sample, params, active), 0.0)
            if node.discount_id is not None:
                amount = amount * sample.discounts[node.discount_id]
            values = scalars(state)
            state = self._write(state, self.n_vars, values[self.n_vars] + amount)
            return self._write(state, index, values[index] + amount / sample.numeraire)

        return evaluate

    def _exercise(self, node):
        value = self.expression(node.args[0])
        condition = (
            self.expression(node.args[1])
            if len(node.args) > 1
            else lambda state, sample, params, active: 1.0
        )

        def evaluate(state, sample, params, active):
            h, cond = value(state, sample, params, active), condition(state, sample, params, active)
            features = []
            for i, feature in enumerate(self.features):
                if feature.variable_index is not None:
                    features.append(scalars(state)[feature.variable_index])
                elif feature.name == "SPOT":
                    features.append(sample.spot)
                else:
                    features.append(sample.observations[self.observation_width + i])
            return self._write(
                state,
                jnp.arange(self.n_vars + 1, self.n_vars + 3 + len(features)),
                jnp.stack((h, jnp.asarray(cond, state.dtype), *features)),
            )

        return evaluate

    def _if(self, node):
        records_payment = any(
            isinstance(child, A.Pays) and child.args[0].index == self.payoff_index
            for child in A.walk(node)
        )
        if records_payment:
            node = replace(node, affected_vars=node.affected_vars + (self.n_vars,))
        return super()._if(node)


class _ExactRecording(_Recording, _Lowerer):
    pass


class _FuzzyRecording(_Recording, _FuzzyLowerer):
    pass


def _extend_state(state, extra):
    current = scalars(state)
    padding = jnp.zeros(extra, dtype=state.dtype)
    values = jnp.concatenate((current, padding)) if current.shape[0] else padding
    return state._replace(values=values) if isinstance(state, ScriptState) else values


def _varying(tree, axis_name):
    if axis_name is None:
        return tree
    return jax.tree.map(
        lambda leaf: (
            leaf
            if axis_name in jax.typeof(leaf).manual_axis_type.varying
            else jax.lax.pcast(leaf, (axis_name,), to="varying")
        ),
        tree,
    )


def _unroll_records(step, carry, inputs, ids):
    samples = tuple(jnp.unstack(field) for field in inputs)
    rows = []
    for sample, event_id in zip(zip(*samples), jnp.unstack(ids)):
        carry, row = step(carry, (EventSample(*sample), event_id))
        rows.append(row)
    return carry, jnp.stack(rows)


def lower_records(prepared, ctx, policy=None):
    """``(initial, event_scenario, script_params) -> (records, hard_price)``.

    A hard pricing policy masks all events after its first exercise, including
    unsafe arithmetic and vector error flags. Training records every event.
    """
    events = prepared.fuzzy_events if ctx.fuzzy else prepared.events
    groups = group_events(events, ctx.scan_group_threshold)
    n_vars, n_features = len(prepared.variables.var_names), len(prepared.regression_features)
    width = max((len(row) for row in prepared.event_observations), default=0)
    functions = []
    for group in groups:
        lowerer = (
            _FuzzyRecording(prepared.variables.const_names, ctx.smooth, ctx.smoothing_kernel)
            if ctx.fuzzy
            else _ExactRecording(prepared.variables.const_names, historical=False, backend=jnp)
        )
        lowerer.configure(n_vars, prepared.payoff_index, prepared.regression_features, width)
        functions.append(lowerer.event(group.template))

    def evaluate(initial, scenario, params):
        state = initial
        carry = _varying((state, jnp.asarray(True), jnp.asarray(0.0, state.dtype)), ctx.axis_name)
        rows = []
        for group, event in zip(groups, functions):
            fields = tuple(field[group.start : group.stop] for field in scenario)
            inputs = EventSample(*fields, jnp.asarray(group.constants, dtype=state.dtype))
            ids = jnp.arange(group.start, group.stop)

            def step(previous, item):
                current, alive, exercise_price = previous
                sample, event_id = item
                before = (
                    scalars(current)[prepared.payoff_index] if prepared.payoff_index >= 0 else 0.0
                )
                current = write(
                    current,
                    jnp.arange(n_vars, n_vars + 3 + n_features),
                    jnp.zeros(3 + n_features, current.dtype),
                    jnp,
                )
                current = event(current, sample, params, alive)
                values = scalars(current)[n_vars:]
                if policy is not None:
                    h, cond, x = values[1], values[2], values[3:]
                    exercise = (
                        alive
                        & policy.exercise_mask[event_id]
                        & (cond > 0.0)
                        & (h > 0.0)
                        & (h > policy.predict(event_id, x))
                    )
                    exercise_price = jnp.where(
                        exercise, before + h / sample.numeraire, exercise_price
                    )
                    alive = alive & ~exercise
                return (current, alive, exercise_price), jnp.concatenate(
                    (values, sample.numeraire[None])
                )

            if group.scanned:
                body = jax.checkpoint(step, prevent_cse=False) if ctx.checkpoint else step
                carry, group_rows = jax.lax.scan(body, carry, (inputs, ids))
            else:
                carry, group_rows = _unroll_records(step, carry, inputs, ids)
            rows.append(group_rows)
        final, alive, exercise_price = carry
        data = jnp.concatenate(rows)
        errors = final.errors if isinstance(final, ScriptState) else jnp.empty(0, dtype=bool)
        recorded = Records(data[:, 0], data[:, 1], data[:, 2], data[:, 3:-1], data[:, -1], errors)
        last = (
            scalars(final)[prepared.payoff_index]
            if prepared.payoff_index >= 0
            else jnp.asarray(0.0, final.dtype)
        )
        return recorded, jnp.where(alive, last, exercise_price)

    return evaluate


def regression_scenario(prepared, scenario):
    """Append the model's regression outputs after the localized FIX slots."""
    events = prepared._event_scenario(scenario)
    columns = []
    for feature in prepared.regression_features:
        if feature.output_by_event and feature.variable_index is None:
            values = scenario.observations[
                jnp.asarray(prepared.event_to_sample), jnp.asarray(feature.output_by_event)
            ]
        else:
            values = jnp.zeros_like(events.spot)
        columns.append(values)
    observations = jnp.concatenate((events.observations, jnp.stack(columns, axis=1)), axis=1)
    return events._replace(observations=observations)
