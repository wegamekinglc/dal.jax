"""Constant marking, a port of DAL's ``visitor/constprocessor.hpp``.

Marks expression nodes whose value is known before simulation (``is_const`` /
``const_val``).  Literal subtrees fold; variables are constant only after an
unconditional constant assignment; ``PAYS`` (numeraire-deflated) and named
constants (differentiable parameters) are never constant; ``SPOT``/``FIX`` are
constant only when ``known_observations`` supplies their fixing.  Nodes are
annotated, never replaced.

DAL's compiled mode folds ``MAX``/``MIN`` from their first two arguments only;
this port folds all arguments, matching DAL's tree evaluator.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace

from dal_jax.script import ast as A
from dal_jax.script.passes.intervals import _c_div, c_exp, c_log, c_pow, c_sqrt

_BINARY = {
    A.Add: lambda x, y: x + y,
    A.Sub: lambda x, y: x - y,
    A.Mul: lambda x, y: x * y,
    A.Div: _c_div,
    A.Pow: c_pow,
}
_UNARY = {A.UPlus: lambda x: x, A.UMinus: lambda x: -x, A.Log: c_log, A.Sqrt: c_sqrt, A.Exp: c_exp}
_REDUCTIONS = {A.Max: max, A.Min: min}


class ConstProcessor:
    def __init__(
        self,
        n_vars: int,
        known_observations: Mapping[int, float] | None = None,
        historical: bool = False,
    ) -> None:
        self.var_const = [True] * n_vars
        self.var_value = [0.0] * n_vars
        self.in_conditional = False
        self.known = known_observations
        self.historical = historical
        self._handlers = {
            A.If: self._if,
            A.Assign: self._assign,
            A.Pays: self._pays,
            A.Var: self._var,
            A.Spot: self._observation,
            A.Fix: self._observation,
            A.ConstVar: lambda node: replace(node, is_const=False),
        }

    def start_future(self) -> None:
        self.historical = False

    @staticmethod
    def _folded(node: A.Node, args: tuple[A.Node, ...], op) -> A.Node:
        if all(isinstance(arg, A.Expr) and arg.is_const for arg in args):
            return replace(
                node, args=args, is_const=True, const_val=op(*(arg.const_val for arg in args))
            )
        return replace(node, args=args, is_const=False)

    def visit(self, node: A.Node) -> A.Node:
        kind = type(node)
        if kind in _BINARY:
            return self._folded(node, self._args(node), _BINARY[kind])
        if kind in _UNARY:
            return self._folded(node, self._args(node), _UNARY[kind])
        if kind in _REDUCTIONS:
            pick = _REDUCTIONS[kind]
            return self._folded(node, self._args(node), lambda *values: pick(values))
        handler = self._handlers.get(kind)
        return handler(node) if handler else node.with_args(self._args(node))

    def _if(self, node: A.If) -> A.Node:
        nested = self.in_conditional
        self.in_conditional = True
        args = self._args(node)
        self.in_conditional = nested
        return node.with_args(args)

    def _assign(self, node: A.Assign) -> A.Node:
        index = node.args[0].index
        value = self.visit(node.args[1])
        constant = not self.in_conditional and isinstance(value, A.Expr) and value.is_const
        self.var_const[index] = constant
        if constant:
            self.var_value[index] = value.const_val
        return node.with_args((node.args[0], value))

    def _pays(self, node: A.Pays) -> A.Node:
        if not self.historical:
            self.var_const[node.args[0].index] = False
        return node.with_args((node.args[0], self.visit(node.args[1])))

    def _var(self, node: A.Var) -> A.Node:
        if self.var_const[node.index]:
            return replace(node, is_const=True, const_val=self.var_value[node.index])
        return replace(node, is_const=False)

    def _observation(self, node: A.Node) -> A.Node:
        """``SPOT`` / ``FIX``: constant only when the fixing is already known."""
        known = None
        if self.known is not None and node.observation_id is not None:
            known = self.known.get(node.observation_id)
        return (
            replace(node, is_const=False)
            if known is None
            else replace(node, is_const=True, const_val=known)
        )

    def _args(self, node: A.Node) -> tuple[A.Node, ...]:
        return tuple(self.visit(arg) for arg in node.args)


def process_constants(
    event_groups: Sequence[Sequence[A.Event]], n_vars: int
) -> list[list[A.Event]]:
    """Mark constants across event groups visited in order (past, then future)."""
    processor = ConstProcessor(n_vars)
    return [
        [tuple(processor.visit(statement) for statement in event) for event in events]
        for events in event_groups
    ]
