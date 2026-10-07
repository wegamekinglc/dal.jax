"""Script syntax tree: one frozen dataclass per DAL ``Node*_`` (``script/node.hpp``).

Children live in ``args`` in DAL's ``arguments_`` order, so visitors port line
by line: binary operators hold ``(lhs, rhs)``, comparisons hold the single
operand ``lhs - rhs`` (``x > y`` is ``Sup(x - y)``, ``x < y`` is
``Sup(y - x)``), ``If`` holds ``(condition, *then, *else)`` with
``first_else`` (``-1`` without else), assignments hold ``(target, value)``.
Passes never mutate a tree: they return a rebuilt one (``dataclasses.replace``).
"""

from dataclasses import dataclass, replace
from typing import Literal

from dal_jax.dates.date import Date
from dal_jax.script.lexer import SourceLocation


@dataclass(frozen=True, slots=True, kw_only=True)
class Node:
    args: tuple["Node", ...] = ()

    def with_args(self, args: tuple["Node", ...]) -> "Node":
        return self if args == self.args else replace(self, args=args)


# --- expressions -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Expr(Node):
    """``ExprNode_``: ``is_const`` / ``const_val`` are filled by the constant processor."""

    is_const: bool = False
    const_val: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class Add(Expr):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Sub(Expr):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Mul(Expr):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Div(Expr):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Pow(Expr):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Max(Expr):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Min(Expr):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class UPlus(Expr):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class UMinus(Expr):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Log(Expr):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Sqrt(Expr):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Exp(Expr):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Const(Expr):
    is_const: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class Var(Expr):
    """A script variable; DAL's constructor marks it constant 0 until the constant processor runs."""

    name: str
    index: int = -1
    is_const: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class ConstVar(Expr):
    """A named numeric constant of the event table (``STRIKE``); stays live so it is differentiable."""

    name: str
    index: int = -1


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorEntry(Expr):
    name: str
    entry: int
    source: SourceLocation
    index: int = -1


type ReduceKind = Literal["Sum", "Average", "Minimum", "Maximum"]


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorReduce(Expr):
    name: str
    kind: ReduceKind
    source: SourceLocation
    index: int = -1


@dataclass(frozen=True, slots=True, kw_only=True)
class Spot(Expr):
    source: SourceLocation
    observation_id: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Fix(Expr):
    literal: str  # raw index literal as written
    canonical: str  # DAL's Index_::Name()
    fixing_date: Date | None
    source: SourceLocation
    observation_id: int | None = None

    def preparation_error(self) -> str:
        return f"PreparationRequired: FIX requires a prepared observation; index={self.literal}; {self.source.describe()}"


# --- conditions --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class BoolNode(Node):
    always_true: bool = False
    always_false: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class Comparison(BoolNode):
    """``CompNode_``: ``eps`` is the ``:eps`` option (``-1`` when unset); discrete bounds come from the domain processor."""

    is_discrete: bool = False
    eps: float = 0.0
    lb: float = 0.0
    rb: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class Equal(Comparison):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Sup(Comparison):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class SupEqual(Comparison):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class And(BoolNode):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Or(BoolNode):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Not(BoolNode):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class TrueNode(BoolNode):
    always_true: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class FalseNode(BoolNode):
    always_false: bool = True


# --- statements ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Statement(Node):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Assign(Statement):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorAssign(Statement):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorAppend(Statement):
    name: str
    source: SourceLocation
    index: int = -1


@dataclass(frozen=True, slots=True, kw_only=True)
class Pays(Statement):
    """``var PAYS expr [ON date]``; preparation clears ``payment_date`` for same-date payments."""

    source: SourceLocation
    payment_date: Date | None = None
    discount_id: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Exercise(Statement):
    """``EXERCISE value [IF condition]``; ``eps`` is the condition's ``;eps`` (``-1`` falls back to the default)."""

    source: SourceLocation
    eps: float = -1.0


@dataclass(frozen=True, slots=True, kw_only=True)
class If(Statement):
    first_else: int = -1
    affected_vars: tuple[int, ...] = ()
    affected_vectors: tuple[int, ...] = ()
    always_true: bool = False
    always_false: bool = False

    @property
    def has_else(self) -> bool:
        return self.first_else != -1

    @property
    def last_true_index(self) -> int:
        return self.first_else - 1 if self.has_else else len(self.args) - 1

    @property
    def condition(self) -> Node:
        return self.args[0]

    @property
    def then_branch(self) -> tuple[Node, ...]:
        return self.args[1 : self.last_true_index + 1]

    @property
    def else_branch(self) -> tuple[Node, ...]:
        return self.args[self.first_else :] if self.has_else else ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Collect(Statement):
    pass


type Event = tuple[Node, ...]


def find_node(node: Node, predicate) -> Node | None:
    """First node in pre-order satisfying ``predicate`` (DAL's ``FindNode``)."""
    if predicate(node):
        return node
    for arg in node.args:
        if (found := find_node(arg, predicate)) is not None:
            return found
    return None


def walk(node: Node):
    """Pre-order traversal."""
    yield node
    for arg in node.args:
        yield from walk(arg)
