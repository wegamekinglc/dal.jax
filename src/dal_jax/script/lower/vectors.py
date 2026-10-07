"""Mutable vector reads, writes and reductions with masked runtime checks."""

from collections.abc import Callable
from typing import Any

import numpy as np

from dal_jax.errors import script_error


class VectorLowering:
    """Mixin used by exact and fuzzy lowerers; checks follow statement activity."""

    xp: Any
    _checks: list
    expression: Callable

    def _vector_entry(self, node):
        check = lambda state, sample, params, active: self.xp.logical_and(active, node.entry >= state.lengths[node.index])
        self._checks.append((2*node.index, check, f"VectorIndexOutOfRange: {node.name}; {node.source.describe()}"))
        return lambda state, sample, params, active: self.xp.where(active, state.vectors[node.index][node.entry], 0.)

    def _vector_reduce(self, node):
        if node.kind != "Sum":
            check = lambda state, sample, params, active: self.xp.logical_and(active, state.lengths[node.index] == 0)
            self._checks.append((2*node.index+1, check, f"EmptyVectorReduction: {node.name}; {node.source.describe()}"))

        def evaluate(state, sample, params, active):
            values = state.vectors[node.index]
            length = state.lengths[node.index]
            mask = self.xp.arange(values.shape[0]) < length
            return self.xp.where(active, self._reduce_values(values, mask, length, node.kind), 0.)

        return evaluate

    def _reduce_values(self, values, mask, length, kind):
        # XLA 0.11.2 CPU constant folding crashes on a zero-size reduction.
        # A physically empty buffer has no active entries for any reduction.
        if not values.shape[0]:
            return self.xp.asarray(0., values.dtype)
        total = self.xp.sum(self.xp.where(mask, values, 0.))
        if kind == "Sum":
            return total
        if kind == "Average":
            return total/self.xp.maximum(length, 1)
        minimum = kind == "Minimum"
        masked = self.xp.where(mask, values, self.xp.inf if minimum else -self.xp.inf)
        index = (self.xp.argmin if minimum else self.xp.argmax)(masked)
        return self.xp.where(length > 0, values[index], 0.)

    def _vector_assign(self, node):
        entry, value = node.args[0], self.expression(node.args[1])
        return lambda state, sample, params, active: self._vector_write(
            state, entry.index, entry.entry, value(state, sample, params, active))

    def _vector_append(self, node):
        value = self.expression(node.args[0])
        return lambda state, sample, params, active: self._vector_write(
            state, node.index, state.lengths[node.index], value(state, sample, params, active))

    def _vector_write(self, state, index, entry, value):
        vectors = list(state.vectors)
        positions = self.xp.arange(vectors[index].shape[0])
        vectors[index] = self.xp.where(positions == entry, value, vectors[index])
        lengths = self.xp.maximum(state.lengths[index], entry+1)
        if self.xp is np:
            result = state.lengths.copy()
            result[index] = lengths
        else:
            result = state.lengths.at[index].set(lengths)
        return state._replace(vectors=tuple(vectors), lengths=result)

    def _checked_statement(self, statement, checks):
        def evaluate(state, sample, params, active):
            errors = state.errors
            for index, check, message in checks:
                invalid = check(state, sample, params, active)
                if self.xp is np:
                    if invalid:
                        raise script_error(message)
                else:
                    errors = errors.at[index].set(errors[index] | invalid)
            return statement(state._replace(errors=errors), sample, params, active)
        return evaluate
