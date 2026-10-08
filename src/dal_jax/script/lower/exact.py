"""Lower scalar script events to single-path JAX functions.

An event maps ``(state, sample, script_params)`` to a fresh variable array.
IF branches start from the same state and merge only ``affected_vars``.
The active-path mask reaches unsafe arithmetic before it is evaluated, so
an unused branch cannot inject NaNs into a reverse-mode derivative.

The NumPy backend executes hard branches on the host for historical replay;
historical PAYS evaluates its expression and leaves the receiver unchanged.
"""

import operator
from collections.abc import Callable, Mapping
from functools import reduce

import jax.numpy as jnp
import numpy as np

from dal_jax.errors import UnsupportedExecutionMode
from dal_jax.script import ast as A
from dal_jax.script.lower.state import initial_state, merge_vectors, scalars, write
from dal_jax.script.lower.vectors import VectorLowering

_BINARY = {A.Add: operator.add, A.Sub: operator.sub, A.Mul: operator.mul}
_COMPARISONS = {A.Equal: operator.eq, A.Sup: operator.gt, A.SupEqual: operator.ge}


class _Lowerer(VectorLowering):
    def __init__(self, const_names: tuple[str, ...], *, historical: bool, backend) -> None:
        self.const_names = const_names
        self.historical = historical
        self.xp = backend
        self._checks = []
        self._expressions = {
            A.Const: self._constant,
            A.EventConst: lambda n: (
                lambda state, sample, params, active: self.xp.where(
                    active, sample.constants[n.index], 0.0
                )
            ),
            A.Var: lambda n: (
                lambda state, sample, params, active: self.xp.where(
                    active, scalars(state)[n.index], 0.0
                )
            ),
            A.ConstVar: lambda n: (
                lambda state, sample, params, active: self.xp.where(
                    active, params[self.const_names[n.index]], 0.0
                )
            ),
            A.Fix: lambda n: (
                lambda state, sample, params, active: self.xp.where(
                    active, sample.observations[n.observation_id], 0.0
                )
            ),
            A.Spot: lambda n: (
                lambda state, sample, params, active: self.xp.where(active, sample.spot, 0.0)
            ),
            A.VectorEntry: self._vector_entry,
            A.VectorReduce: self._vector_reduce,
            A.Div: self._divide,
            A.Pow: self._power,
            A.Max: lambda n: self._extremum(n, lambda a, b: self.xp.where(a < b, b, a)),
            A.Min: lambda n: self._extremum(n, lambda a, b: self.xp.where(a > b, b, a)),
            A.UPlus: lambda n: self.expression(n.args[0]),
            A.UMinus: lambda n: self._unary(n, operator.neg),
            A.Log: lambda n: self._unary(n, self.xp.log, safe=1.0),
            A.Sqrt: lambda n: self._unary(n, self.xp.sqrt, safe=1.0),
            A.Exp: lambda n: self._unary(n, self.xp.exp, safe=0.0),
            A.And: lambda n: self._binary(n, self.xp.logical_and),
            A.Or: lambda n: self._binary(n, self.xp.logical_or),
            A.Not: lambda n: self._unary(n, self.xp.logical_not),
            A.TrueNode: lambda n: lambda state, sample, params, active: True,
            A.FalseNode: lambda n: lambda state, sample, params, active: False,
        }
        self._statements = {
            A.VectorAssign: self._vector_assign,
            A.VectorAppend: self._vector_append,
            A.Assign: self._assign,
            A.Pays: self._pays,
            A.If: self._if,
            A.Collect: lambda n: self.event(n.args),
        }

    def expression(self, node: A.Node) -> Callable:
        kind = type(node)
        if (
            isinstance(node, A.Expr)
            and node.is_const
            and not isinstance(node, (A.Var, A.ConstVar, A.Spot))
        ):
            return self._constant(A.Const(const_val=node.const_val))
        if kind in _BINARY:
            return self._binary(node, _BINARY[kind])
        if kind in _COMPARISONS:
            operand, op = self.expression(node.args[0]), _COMPARISONS[kind]
            return lambda state, sample, params, active: op(
                operand(state, sample, params, active), 0.0
            )
        handler = self._expressions.get(kind)
        if handler is None:
            raise UnsupportedExecutionMode(
                f"exact scalar lowering does not support {kind.__name__}"
            )
        return handler(node)

    def _constant(self, node: A.Const) -> Callable:
        return lambda state, sample, params, active: self.xp.where(
            active, self.xp.asarray(node.const_val, dtype=state.dtype), 0.0
        )

    def _binary(self, node: A.Node, op) -> Callable:
        left, right = (self.expression(arg) for arg in node.args)
        return lambda state, sample, params, active: op(
            left(state, sample, params, active), right(state, sample, params, active)
        )

    def _unary(self, node: A.Node, op, *, safe: float | None = None) -> Callable:
        operand = self.expression(node.args[0])

        def evaluate(state, sample, params, active):
            value = operand(state, sample, params, active)
            return op(value if safe is None else self.xp.where(active, value, safe))

        return evaluate

    def _divide(self, node: A.Div) -> Callable:
        numerator, denominator = (self.expression(arg) for arg in node.args)

        def evaluate(state, sample, params, active):
            top = self.xp.where(active, numerator(state, sample, params, active), 0.0)
            bottom = self.xp.where(active, denominator(state, sample, params, active), 1.0)
            return top / bottom

        return evaluate

    def _power(self, node: A.Pow) -> Callable:
        base, exponent = (self.expression(arg) for arg in node.args)

        def evaluate(state, sample, params, active):
            left = self.xp.where(active, base(state, sample, params, active), 1.0)
            right = self.xp.where(active, exponent(state, sample, params, active), 1.0)
            return self.xp.power(left, right)

        return evaluate

    def _extremum(self, node: A.Node, op) -> Callable:
        operands = tuple(self.expression(arg) for arg in node.args)
        return lambda state, sample, params, active: reduce(
            op, (f(state, sample, params, active) for f in operands)
        )

    def _write(self, state, index, value):
        return write(state, index, value, self.xp)

    def _assign(self, node: A.Assign) -> Callable:
        index, value = node.args[0].index, self.expression(node.args[1])
        return lambda state, sample, params, active: self._write(
            state, index, value(state, sample, params, active)
        )

    def _pays(self, node: A.Pays) -> Callable:
        index, value = node.args[0].index, self.expression(node.args[1])

        def evaluate(state, sample, params, active):
            amount = self.xp.where(active, value(state, sample, params, active), 0.0)
            if self.historical:
                return state
            numeraire = self.xp.where(active, sample.numeraire, 1.0)
            if node.discount_id is not None:
                amount = amount * self.xp.where(active, sample.discounts[node.discount_id], 1.0)
            return self._write(state, index, scalars(state)[index] + amount / numeraire)

        return evaluate

    def _if(self, node: A.If) -> Callable:
        if isinstance(node.condition, (A.TrueNode, A.FalseNode)):
            return self.event(
                node.then_branch if isinstance(node.condition, A.TrueNode) else node.else_branch
            )
        condition = self.expression(node.condition)
        then, otherwise = self.event(node.then_branch), self.event(node.else_branch)
        indices = np.asarray(node.affected_vars, dtype=np.int32)

        def evaluate(state, sample, params, active):
            take_then = condition(state, sample, params, active)
            if self.xp is np:
                return (then if take_then else otherwise)(state, sample, params, active)
            left = then(state, sample, params, self.xp.logical_and(active, take_then))
            right = otherwise(
                state, sample, params, self.xp.logical_and(active, self.xp.logical_not(take_then))
            )
            values = self.xp.where(take_then, scalars(left)[indices], scalars(right)[indices])
            state = self._write(state, indices, values)
            return merge_vectors(
                state,
                left,
                right,
                node.affected_vectors,
                lambda a, b: self.xp.where(take_then, a, b),
                self.xp,
            )

        return evaluate

    def event(self, event: A.Event) -> Callable:
        statements = tuple(self._statement(node) for node in event)

        def evaluate(state, sample, params, active=True):
            for statement in statements:
                state = statement(state, sample, params, active)
            return state

        return evaluate

    def _statement(self, node: A.Node) -> Callable:
        handler = self._statements.get(type(node))
        if handler is None:
            raise UnsupportedExecutionMode(
                f"exact scalar lowering does not support {type(node).__name__}"
            )
        outer_checks = self._checks
        self._checks = []
        statement = handler(node)
        checks = self._checks
        self._checks = outer_checks
        return self._checked_statement(statement, checks) if checks else statement


def lower_event(
    event: A.Event, const_names: tuple[str, ...] = (), *, historical: bool = False
) -> Callable:
    """``event_k(state, sample_k, script_params) -> state``; pure and jittable."""
    return _Lowerer(const_names, historical=historical, backend=jnp).event(event)


def replay_events(
    events: tuple[A.Event, ...],
    spots: tuple[float, ...],
    n_vars: int,
    const_names: tuple[str, ...],
    params: Mapping[str, float],
    *,
    capacities=(),
    observations=(),
):
    """Replay one historical path on the host, using DAL's hard branch semantics."""
    from dal_jax.models.base import Sample  # avoid coupling the lowering to the MC engine

    lowerer = _Lowerer(const_names, historical=True, backend=np)
    state = initial_state(n_vars, capacities, np)
    with np.errstate(all="ignore"):
        for event, spot in zip(events, spots):
            sample = Sample(
                np.asarray(spot), np.asarray(1.0), np.asarray(observations), np.empty(0)
            )
            state = lowerer.event(event)(state, sample, params)
    return state if capacities else tuple(float(value) for value in state)
