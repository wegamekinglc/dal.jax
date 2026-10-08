"""IF metadata, a port of DAL's ``visitor/ifprocessor.hpp``.

For every ``If`` it records the (sorted) indices of variables and vectors
written in its branches, nested ``If``s included; fuzzy evaluation blends only
those.  It also reports the deepest ``If`` nesting.
"""

from collections.abc import Sequence
from dataclasses import replace

from dal_jax.script import ast as A


class _IfProcessor:
    def __init__(self) -> None:
        self.var_stack: list[set[int]] = []
        self.vector_stack: list[set[int]] = []
        self.max_nested_ifs = 0
        self._handlers = {
            A.If: self._if,
            A.Assign: self._target,
            A.Pays: self._target,
            A.VectorAssign: lambda node: self._vector(node, node.args[0].index),
            A.VectorAppend: lambda node: self._vector(node, node.index),
            A.Var: self._var,
        }

    def visit(self, node: A.Node) -> A.Node:
        handler = self._handlers.get(type(node))
        return (
            handler(node)
            if handler
            else node.with_args(tuple(self.visit(arg) for arg in node.args))
        )

    def _if(self, node: A.If) -> A.Node:
        self.var_stack.append(set())
        self.vector_stack.append(set())
        self.max_nested_ifs = max(self.max_nested_ifs, len(self.var_stack))
        args = (node.args[0],) + tuple(self.visit(arg) for arg in node.args[1:])
        affected_vars, affected_vectors = self.var_stack.pop(), self.vector_stack.pop()
        if self.var_stack:
            self.var_stack[-1] |= affected_vars
            self.vector_stack[-1] |= affected_vectors
        return replace(
            node,
            args=args,
            affected_vars=tuple(sorted(affected_vars)),
            affected_vectors=tuple(sorted(affected_vectors)),
        )

    def _target(self, node: A.Node) -> A.Node:
        """Assignments and payments: only the written variable matters."""
        if self.var_stack:
            self.visit(node.args[0])
        return node

    def _vector(self, node: A.Node, index: int) -> A.Node:
        if self.vector_stack:
            self.vector_stack[-1].add(index)
        return node

    def _var(self, node: A.Var) -> A.Node:
        if self.var_stack:
            self.var_stack[-1].add(node.index)
        return node


def process_ifs(events: Sequence[A.Event]) -> tuple[list[A.Event], int]:
    """Annotated events and the maximum number of nested ``If``s."""
    processor = _IfProcessor()
    annotated = [tuple(processor.visit(statement) for statement in event) for event in events]
    return annotated, processor.max_nested_ifs
