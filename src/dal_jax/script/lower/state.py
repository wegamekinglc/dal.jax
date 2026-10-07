"""Fixed-capacity path state and endpoint-aware scalar/vector branch merging."""

from typing import NamedTuple

import numpy as np


class ScriptState(NamedTuple):
    values: object
    vectors: tuple
    lengths: object
    errors: object

    @property
    def dtype(self):
        return self.values.dtype


def scalars(state):
    return state.values if isinstance(state, ScriptState) else state


def write(state, index, value, xp):
    values = scalars(state)
    if xp is np:
        result = values.copy()
        result[index] = value
    else:
        result = values.at[index].set(value)
    return state._replace(values=result) if isinstance(state, ScriptState) else result


def initial_state(n_vars, capacities, xp):
    values = xp.zeros(n_vars, dtype=xp.float64)
    if not capacities:
        return values
    return ScriptState(values, tuple(xp.zeros(cap, dtype=xp.float64) for cap in capacities),
                       xp.zeros(len(capacities), dtype=xp.int32), xp.zeros(2*len(capacities), dtype=bool))


def merge_vectors(state, left, right, affected, choose, xp, *, blend=None):
    """Length is branch-selected at endpoints and max-length inside a fuzzy IF."""
    if not isinstance(state, ScriptState):
        return state
    vectors = list(state.vectors)
    lengths = state.lengths
    for index in affected:
        positions = xp.arange(vectors[index].shape[0])
        a = xp.where(positions < left.lengths[index], left.vectors[index], 0.)
        b = xp.where(positions < right.lengths[index], right.vectors[index], 0.)
        value = choose(a, b) if blend is None else choose(a, b, blend(a, b))
        length = choose(left.lengths[index], right.lengths[index]) if blend is None else choose(
            left.lengths[index], right.lengths[index], xp.maximum(left.lengths[index], right.lengths[index]))
        vectors[index] = value
        lengths = lengths.at[index].set(length)
    return state._replace(vectors=tuple(vectors), lengths=lengths, errors=left.errors | right.errors)
