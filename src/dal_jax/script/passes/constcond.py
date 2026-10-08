"""Constant-condition folding, a port of DAL's ``visitor/constcondprocessor.hpp``."""

from collections.abc import Sequence

from dal_jax.script import ast as A


def _has_eager_boolean(node: A.Node) -> bool:
    return A.find_node(node, lambda n: isinstance(n, (A.And, A.Or))) is not None


def _decided(node: A.Node, condition: A.Node) -> bool:
    """A condition proved always true/false that holds no eagerly evaluated AND/OR."""
    return (node.always_true or node.always_false) and not _has_eager_boolean(condition)


def _fold_children(node: A.Node) -> A.Node:
    return node.with_args(tuple(fold_conditions(arg) for arg in node.args))


def fold_conditions(node: A.Node) -> A.Node:
    if isinstance(node, (A.Comparison, A.Not, A.And, A.Or)) and _decided(node, node):
        return A.TrueNode() if node.always_true else A.FalseNode()
    if isinstance(node, A.If) and _decided(node, node.args[0]):
        kept = node.then_branch if node.always_true else node.else_branch
        return A.Collect(args=tuple(fold_conditions(arg) for arg in kept))
    return _fold_children(node)


def process_const_conditions(events: Sequence[A.Event]) -> list[A.Event]:
    return [tuple(fold_conditions(statement) for statement in event) for event in events]
