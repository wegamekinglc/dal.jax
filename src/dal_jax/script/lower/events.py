"""Execute normalized event groups, scanning long runs with one traced body."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from dal_jax.script.lower import exact, fuzzy
from dal_jax.script.passes.eventgroup import group_events


class EventSample(NamedTuple):
    spot: object
    numeraire: object
    observations: object
    discounts: object
    constants: object


def _scan_event(event, state, inputs, params, checkpoint):
    step = lambda carry, sample: (event(carry, sample, params), None)
    step = jax.checkpoint(step, prevent_cse=False) if checkpoint else step
    return jax.lax.scan(step, state, inputs)[0]


def lower_events(events, const_names, ctx):
    groups = group_events(events, ctx.scan_group_threshold)
    lower = (
        (
            lambda event: fuzzy.lower_event(
                event, const_names, smooth=ctx.smooth, kernel=ctx.smoothing_kernel
            )
        )
        if ctx.fuzzy
        else (lambda event: exact.lower_event(event, const_names))
    )
    functions = tuple(lower(group.template) for group in groups)

    def evaluate(state, scenario, params):
        # Parameter-dependent history is already varying under shard_map;
        # JAX rejects a second varying-to-varying pcast.
        if ctx.axis_name is not None:
            state = jax.tree.map(
                lambda leaf: (
                    leaf
                    if ctx.axis_name in jax.typeof(leaf).manual_axis_type.varying
                    else jax.lax.pcast(leaf, (ctx.axis_name,), to="varying")
                ),
                state,
            )
        for group, event in zip(groups, functions):
            fields = tuple(field[group.start : group.stop] for field in scenario)
            constants = jnp.asarray(group.constants, dtype=state.dtype)
            inputs = EventSample(*fields, constants)
            if group.scanned:
                state = _scan_event(event, state, inputs, params, ctx.checkpoint)
            else:
                samples = tuple(jnp.unstack(field) for field in inputs)
                for sample in zip(*samples):
                    state = event(state, EventSample(*sample), params)
        return state

    return evaluate
