"""Normalize event literals and group adjacent equal templates for lax.scan."""

from dataclasses import dataclass, replace

from dal_jax.script import ast as A
from dal_jax.script.lexer import SourceLocation

_SOURCE = SourceLocation()
_LIVE_LEAVES = (A.Var, A.ConstVar, A.Spot)


def normalize_event(event: A.Event) -> tuple[A.Event, tuple[float, ...]]:
    """Normalize event identities; folded literals become per-date slots."""
    constants = []

    def normalize(node):
        if isinstance(node, A.Expr) and node.is_const and not isinstance(node, _LIVE_LEAVES):
            index = len(constants)
            constants.append(node.const_val)
            return A.EventConst(index=index)
        changes = _identity_fields(node)
        changes["args"] = tuple(normalize(arg) for arg in node.args)
        return replace(node, **changes)

    return tuple(normalize(statement) for statement in event), tuple(constants)


def _identity_fields(node):
    if isinstance(node, (A.Var, A.ConstVar)):
        return {"name": str(node.index), "is_const": False, "const_val": 0.0}
    if isinstance(node, A.Spot):
        return {"source": _SOURCE, "observation_id": None, "is_const": False, "const_val": 0.0}
    if isinstance(node, (A.VectorEntry, A.VectorReduce)):
        return {"source": _SOURCE, "name": str(node.index), "is_const": False, "const_val": 0.0}
    if isinstance(node, A.VectorAppend):
        return {"source": _SOURCE, "name": str(node.index)}
    if isinstance(node, A.Fix):
        return {"source": _SOURCE, "fixing_date": None, "is_const": False, "const_val": 0.0}
    if isinstance(node, A.Pays):
        return {"source": _SOURCE, "payment_date": None}
    return _expression_identity(node)


def _expression_identity(node):
    if isinstance(node, A.Exercise):
        return {"source": _SOURCE}
    if isinstance(node, A.Expr):
        return {"is_const": False, "const_val": 0.0}
    return {}


@dataclass(frozen=True, slots=True)
class EventGroup:
    start: int
    stop: int
    template: A.Event
    constants: tuple[tuple[float, ...], ...]
    scanned: bool

    @property
    def size(self) -> int:
        return self.stop - self.start


def group_events(events: tuple[A.Event, ...], threshold: int = 4) -> tuple[EventGroup, ...]:
    """Group runs of equal templates; ``threshold=0`` disables scanning."""
    normalized = tuple(normalize_event(event) for event in events)
    groups = []
    start = 0
    while start < len(events):
        template, _ = normalized[start]
        stop = start + 1
        while stop < len(events) and normalized[stop][0] == template:
            stop += 1
        constants = tuple(row for _, row in normalized[start:stop])
        groups.append(
            EventGroup(
                start, stop, template, constants, threshold > 0 and stop - start >= threshold
            )
        )
        start = stop
    return tuple(groups)
