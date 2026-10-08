"""Domain analysis, a port of DAL's ``visitor/domainproc.hpp``."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from enum import Enum

from dal_jax.errors import script_error
from dal_jax.script import ast as A
from dal_jax.script.passes.intervals import (
    POSITIVE_HALF_LINE,
    REAL_LINE,
    Domain,
    c_exp,
    c_log,
    c_pow,
    c_sqrt,
)


class Cond(Enum):
    ALWAYS_TRUE = 1
    ALWAYS_FALSE = 2
    TRUE_OR_FALSE = 3


def _flags(cond: Cond) -> dict:
    return {"always_true": cond is Cond.ALWAYS_TRUE, "always_false": cond is Cond.ALWAYS_FALSE}


class DomainProcessor:
    def __init__(
        self,
        n_vars: int,
        fuzzy: bool,
        initial_domains: Sequence[Domain] | None = None,
        known_observations: Mapping[int, float] | None = None,
    ) -> None:
        self.fuzzy = fuzzy
        self.var_domains = (
            [d.copy() for d in initial_domains]
            if initial_domains is not None
            else [Domain.value(0.0) for _ in range(n_vars)]
        )
        self.known = known_observations
        self.doms: list[Domain] = []
        self.conds: list[Cond] = []
        self._handlers = self._handler_table()

    def _retain_fuzzy(self, node: A.Comparison, domain: Domain) -> A.Node | None:
        if not self.fuzzy or (self.known is None and not domain.is_constant):
            return None
        self.conds.append(Cond.TRUE_OR_FALSE)
        self.doms.pop()
        return replace(node, is_discrete=False, always_true=False, always_false=False)

    def _push_observation(self, observation_id: int | None) -> None:
        if observation_id is not None and self.known is not None and observation_id in self.known:
            self.doms.append(Domain.value(self.known[observation_id]))
        else:
            self.doms.append(Domain.real())

    def _args(self, node: A.Node) -> tuple[A.Node, ...]:
        return tuple(self.visit(arg) for arg in node.args)

    def _binary(self, node: A.Node, op) -> A.Node:
        args = self._args(node)
        rhs, lhs = self.doms.pop(), self.doms.pop()
        self.doms.append(op(lhs, rhs))
        return node.with_args(args)

    def _unary(self, node: A.Node, func, func_domain) -> A.Node:
        args = self._args(node)
        self.doms.append(self.doms.pop().apply(func, func_domain))
        return node.with_args(args)

    def _extrema(self, node: A.Node, maximum: bool) -> A.Node:
        args = self._args(node)
        result = self.doms.pop()
        for _ in range(1, len(node.args)):
            result = result.dmax(self.doms.pop()) if maximum else result.dmin(self.doms.pop())
        self.doms.append(result)
        return node.with_args(args)

    def _compare(self, node: A.Comparison, never_true, never_false, fuzzy_bounds) -> A.Node:
        """Shared skeleton of ``Equal`` / ``Sup`` / ``SupEqual`` on the operand's domain."""
        node = node.with_args(self._args(node))
        domain = self.doms[-1]
        if (retained := self._retain_fuzzy(node, domain)) is not None:
            return retained
        self.doms.pop()
        if never_true(domain):
            cond = Cond.ALWAYS_FALSE
        elif never_false(domain):
            cond = Cond.ALWAYS_TRUE
        else:
            cond = Cond.TRUE_OR_FALSE
        self.conds.append(cond)
        node = replace(node, **_flags(cond))
        if cond is Cond.TRUE_OR_FALSE and self.fuzzy:
            node = fuzzy_bounds(node, domain)
        return node

    def _equal(self, node: A.Equal) -> A.Node:
        return self._compare(
            node, lambda d: not d.can_be_zero(), lambda d: not d.can_be_non_zero(), _equal_bounds
        )

    def _sup(self, node: A.Comparison, strict: bool) -> A.Node:
        return self._compare(
            node,
            lambda d: not d.can_be_positive(strict),
            lambda d: not d.can_be_negative(not strict),
            lambda n, d: _sup_bounds(n, d, strict),
        )

    def _logical(self, node: A.Node) -> A.Node:
        args = self._args(node)
        if isinstance(node, A.Not):
            cp = self.conds.pop()
            cond = {Cond.ALWAYS_TRUE: Cond.ALWAYS_FALSE, Cond.ALWAYS_FALSE: Cond.ALWAYS_TRUE}.get(
                cp, Cond.TRUE_OR_FALSE
            )
        else:
            cp1, cp2 = self.conds.pop(), self.conds.pop()
            if isinstance(node, A.And):
                cond = (
                    Cond.ALWAYS_TRUE
                    if cp1 is cp2 is Cond.ALWAYS_TRUE
                    else Cond.ALWAYS_FALSE
                    if Cond.ALWAYS_FALSE in (cp1, cp2)
                    else Cond.TRUE_OR_FALSE
                )
            else:
                cond = (
                    Cond.ALWAYS_TRUE
                    if Cond.ALWAYS_TRUE in (cp1, cp2)
                    else Cond.ALWAYS_FALSE
                    if cp1 is cp2 is Cond.ALWAYS_FALSE
                    else Cond.TRUE_OR_FALSE
                )
        self.conds.append(cond)
        return replace(node, args=args, **_flags(cond))

    def _visit_range(self, node: A.If, args: list[A.Node], start: int, stop: int) -> None:
        for i in range(start, stop):
            args[i] = self.visit(node.args[i])

    def _visit_else(self, node: A.If, args: list[A.Node]) -> None:
        if node.has_else:
            self._visit_range(node, args, node.first_else, len(node.args))

    def _visit_both_branches(self, node: A.If, args: list[A.Node]) -> None:
        """Run both branches from the same start and join the affected variables' domains."""
        before = [self.var_domains[i].copy() for i in node.affected_vars]
        self._visit_range(node, args, 1, node.last_true_index + 1)
        after_then = [self.var_domains[i] for i in node.affected_vars]
        for k, i in enumerate(node.affected_vars):
            self.var_domains[i] = before[k]
        self._visit_else(node, args)
        for k, i in enumerate(node.affected_vars):
            self.var_domains[i].add_domain(after_then[k])

    def _if(self, node: A.If) -> A.Node:
        args = list(node.args)
        args[0] = self.visit(node.args[0])
        cond = self.conds.pop()
        if cond is Cond.ALWAYS_TRUE:
            self._visit_range(node, args, 1, node.last_true_index + 1)
        elif cond is Cond.ALWAYS_FALSE:
            self._visit_else(node, args)
        else:
            self._visit_both_branches(node, args)
        return replace(node, args=tuple(args), **_flags(cond))

    def _assign(self, node: A.Assign) -> A.Node:
        value = self.visit(node.args[1])
        self.var_domains[node.args[0].index] = self.doms.pop()
        return node.with_args((node.args[0], value))

    def _pays(self, node: A.Pays) -> A.Node:
        value = self.visit(node.args[1])
        index = node.args[0].index
        self.var_domains[index] = self.var_domains[index] + self.doms.pop() / Domain(
            POSITIVE_HALF_LINE
        )
        return node.with_args((node.args[0], value))

    #  vector writes: the value is analysed but vectors carry no domain

    def _vector_append(self, node: A.VectorAppend) -> A.Node:
        args = self._args(node)
        self.doms.pop()
        return node.with_args(args)

    def _vector_assign(self, node: A.VectorAssign) -> A.Node:
        value = self.visit(node.args[1])
        self.doms.pop()
        return node.with_args((node.args[0], value))

    def _exercise(self, node: A.Exercise) -> A.Node:
        args = self._args(node)
        self.doms.pop()
        if len(node.args) > 1:
            self.conds.pop()
        return node.with_args(args)

    def _unknown(self, node: A.Node) -> A.Node:
        self.doms.append(Domain.real())
        return node

    def _var(self, node: A.Var) -> A.Node:
        self.doms.append(self.var_domains[node.index].copy())
        return node

    def _const(self, node: A.Const) -> A.Node:
        self.doms.append(Domain.value(node.const_val))
        return node

    def _fix(self, node: A.Fix) -> A.Node:
        if node.observation_id is None or self.known is None:
            raise script_error(node.preparation_error())
        self._push_observation(node.observation_id)
        return node

    def _spot(self, node: A.Spot) -> A.Node:
        self._push_observation(node.observation_id)
        return node

    def _negate(self, node: A.UMinus) -> A.Node:
        args = self._args(node)
        self.doms.append(-self.doms.pop())
        return node.with_args(args)

    def _handler_table(self) -> dict:
        return {
            A.Add: lambda n: self._binary(n, Domain.__add__),
            A.Sub: lambda n: self._binary(n, Domain.__sub__),
            A.Mul: lambda n: self._binary(n, Domain.__mul__),
            A.Div: lambda n: self._binary(n, Domain.__truediv__),
            A.Pow: lambda n: self._binary(n, lambda x, y: x.apply2(c_pow, y, REAL_LINE)),
            A.UMinus: self._negate,
            A.Log: lambda n: self._unary(n, c_log, POSITIVE_HALF_LINE),
            A.Sqrt: lambda n: self._unary(n, c_sqrt, POSITIVE_HALF_LINE),
            A.Exp: lambda n: self._unary(n, c_exp, REAL_LINE),
            A.Max: lambda n: self._extrema(n, True),
            A.Min: lambda n: self._extrema(n, False),
            A.Equal: self._equal,
            A.Sup: lambda n: self._sup(n, True),
            A.SupEqual: lambda n: self._sup(n, False),
            A.Not: self._logical,
            A.And: self._logical,
            A.Or: self._logical,
            A.If: self._if,
            A.Assign: self._assign,
            A.Pays: self._pays,
            A.VectorAppend: self._vector_append,
            A.VectorAssign: self._vector_assign,
            A.VectorEntry: self._unknown,
            A.VectorReduce: self._unknown,
            A.ConstVar: self._unknown,
            A.Exercise: self._exercise,
            A.Var: self._var,
            A.Const: self._const,
            A.Fix: self._fix,
            A.Spot: self._spot,
        }

    def visit(self, node: A.Node) -> A.Node:
        handler = self._handlers.get(type(node))
        return handler(node) if handler else node.with_args(self._args(node))


def _either(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


def _equal_bounds(node: A.Equal, domain: Domain) -> A.Node:
    """Butterfly bounds of ``x = 0`` when 0 is an isolated point of the domain."""
    if not domain.zero_is_discrete():
        return replace(node, is_discrete=False)
    rb, lb = domain.smallest_pos_lb(True), domain.biggest_neg_rb(True)
    return replace(node, is_discrete=True, rb=_either(rb, 0.5), lb=_either(lb, -0.5))


def _sup_bounds(node: A.Comparison, domain: Domain, strict: bool) -> A.Node:
    """Call-spread bounds of ``x > 0`` / ``x >= 0`` around a gap of the domain."""
    if domain.can_be_zero() and not domain.zero_is_discrete():
        return replace(node, is_discrete=False)
    positive, negative = domain.smallest_pos_lb(True), domain.biggest_neg_rb(True)
    lb, rb = node.lb, node.rb
    if not domain.can_be_zero():
        lb, rb = _either(negative, lb), _either(positive, rb)
    elif strict:
        lb, rb = 0.0, _either(positive, rb)
    else:
        lb, rb = _either(negative, lb), 0.0
    return replace(node, is_discrete=True, lb=lb, rb=rb)


def process_domains(
    event_groups: Sequence[Sequence[A.Event]], n_vars: int, fuzzy: bool
) -> list[list[A.Event]]:
    processor = DomainProcessor(n_vars, fuzzy)
    return [
        [tuple(processor.visit(statement) for statement in event) for event in events]
        for events in event_groups
    ]
