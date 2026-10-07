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


def lower_events(events, const_names, ctx):
    groups = group_events(events, ctx.scan_group_threshold)
    lower = (lambda event: fuzzy.lower_event(event, const_names, smooth=ctx.smooth, kernel=ctx.smoothing_kernel)) if ctx.fuzzy else (
        lambda event: exact.lower_event(event, const_names))
    functions = tuple(lower(group.template) for group in groups)

    def evaluate(state, scenario, params):
        # Parameter-dependent history is already varying under shard_map;
        # JAX rejects a second varying-to-varying pcast.
        if ctx.axis_name is not None and ctx.axis_name not in jax.typeof(state).manual_axis_type.varying:
            state = jax.lax.pcast(state, (ctx.axis_name,), to="varying")
        for group, event in zip(groups, functions):
            fields = tuple(field[group.start:group.stop] for field in scenario)
            constants = jnp.asarray(group.constants, dtype=state.dtype)
            inputs = EventSample(*fields, constants)
            if group.scanned:
                state, _ = jax.lax.scan(lambda carry, sample: (event(carry, sample, params), None), state, inputs)
            else:
                samples = tuple(jnp.unstack(field) for field in inputs)
                for sample in zip(*samples):
                    state = event(state, EventSample(*sample), params)
        return state

    return evaluate
