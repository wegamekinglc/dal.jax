"""Variable numbering, a port of DAL's ``visitor/varindexer.hpp``.

Variables, named constants and vectors are numbered in pre-order of first
appearance across all events (past first, then future), case-insensitively,
keeping the first spelling.  A vector's capacity is its largest indexed write
plus one, plus the number of ``APPEND`` statements.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from dal_jax.errors import script_error
from dal_jax.script import ast as A
from dal_jax.strings import CIMap


@dataclass(frozen=True, slots=True)
class VarTable:
    var_names: tuple[str, ...] = ()
    const_names: tuple[str, ...] = ()
    const_values: tuple[float, ...] = ()
    vector_names: tuple[str, ...] = ()
    vector_capacities: tuple[int, ...] = ()


class _Indexer:
    def __init__(self) -> None:
        self.vars = CIMap()  # name -> index
        self.consts = CIMap()  # name -> (index, value)
        self.vectors = CIMap()  # name -> [index, largest_entry, appends]
        self._handlers = {
            A.Var: self._var,
            A.ConstVar: self._const_var,
            A.VectorEntry: self._entry,
            A.VectorReduce: lambda n: replace(n, index=self._vector(n.name, n.source)[0]),
            A.VectorAppend: self._append,
        }

    def _vector(self, name: str, source) -> list[int]:
        if name in self.vars or name in self.consts:
            raise script_error(
                "VectorNameConflict: vector name also names a scalar; " + source.describe()
            )
        if name not in self.vectors:
            self.vectors[name] = [len(self.vectors), 0, 0]
        return self.vectors[name]

    def _scalar_name(self, name: str) -> None:
        if name in self.vectors:
            raise script_error("VectorNameConflict: scalar name also names a vector")

    def _var(self, node: A.Var) -> A.Node:
        self._scalar_name(node.name)
        if node.name not in self.vars:
            self.vars[node.name] = len(self.vars)
        return replace(node, index=self.vars[node.name])

    def _const_var(self, node: A.ConstVar) -> A.Node:
        self._scalar_name(node.name)
        if node.name not in self.consts:
            self.consts[node.name] = (len(self.consts), node.const_val)
        return replace(node, index=self.consts[node.name][0])

    def _entry(self, node: A.VectorEntry) -> A.Node:
        info = self._vector(node.name, node.source)
        info[1] = max(info[1], node.entry + 1)
        return replace(node, index=info[0])

    def _append(self, node: A.VectorAppend) -> A.Node:
        info = self._vector(node.name, node.source)
        info[2] += 1
        return replace(node, index=info[0], args=tuple(self.visit(arg) for arg in node.args))

    def visit(self, node: A.Node) -> A.Node:
        handler = self._handlers.get(type(node))
        return (
            handler(node)
            if handler
            else node.with_args(tuple(self.visit(arg) for arg in node.args))
        )

    def table(self) -> VarTable:
        def by_index(mapping, index_of):
            names = [""] * len(mapping)
            for name, value in mapping.items():
                names[index_of(value)] = name
            return tuple(names)

        const_values = [0.0] * len(self.consts)
        for index, value in self.consts.values():
            const_values[index] = value
        capacities = [0] * len(self.vectors)
        for index, largest, appends in self.vectors.values():
            capacities[index] = largest + appends
        return VarTable(
            var_names=by_index(self.vars, lambda v: v),
            const_names=by_index(self.consts, lambda v: v[0]),
            const_values=tuple(const_values),
            vector_names=by_index(self.vectors, lambda v: v[0]),
            vector_capacities=tuple(capacities),
        )


def index_variables(
    event_groups: Sequence[Sequence[A.Event]],
) -> tuple[list[list[A.Event]], VarTable]:
    """Number every event of every group (e.g. past then future) with one shared table."""
    indexer = _Indexer()
    indexed = [
        [tuple(indexer.visit(statement) for statement in event) for event in events]
        for events in event_groups
    ]
    return indexed, indexer.table()
