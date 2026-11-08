"""Request-driven risk machinery mirroring DAL 2026.11.1's script risk API."""

from collections.abc import Iterator
from dataclasses import dataclass

from dal_jax.errors import script_error
from dal_jax.script import ast as A
from dal_jax.script.product import ScriptProduct, ScriptProductData
from dal_jax.strings import ci_key


@dataclass(frozen=True, slots=True)
class RiskCoordinate:
    family: str
    id: str
    label: str
    ordinal: int
    native_unit: str
    physical_unit: str | None
    report_scale: float
    value: float


@dataclass(frozen=True, slots=True)
class RiskOutputCoordinate:
    id: str
    label: str
    slot: int


@dataclass(slots=True)
class RiskRequest:
    inputs: object = None
    outputs: object = None
    report_factors: object = None
    numeric_payload_budget_bytes: int | None = None


@dataclass(slots=True)
class JacobianRiskRequest(RiskRequest):
    max_block_width: int = 1
    recording_capacity_budget_bytes: int | None = None
    scratch_capacity_budget_bytes: int | None = None


@dataclass(slots=True)
class WeightedRiskRequest(RiskRequest):
    weights: object = None


@dataclass(slots=True)
class BumpOverAADRequest:
    directions: object
    steps: object
    numeric_payload_budget_bytes: int | None = None
    recording_capacity_budget_bytes: int | None = None


@dataclass(slots=True)
class RiskExecutionSnapshot:
    paths_per_replicate: int
    pricing_replicates: int
    method: str
    block_width: int


@dataclass(slots=True)
class RiskResultProvenance:
    engine: str
    method: str
    normalization: str
    calibration: str
    evaluation_date: str
    execution: RiskExecutionSnapshot


@dataclass(slots=True)
class RiskResult:
    values: list
    output_ids: list
    input_axis: list
    complete_input_axis: list
    output_axis: list
    jacobian: object
    reported_jacobian: object
    legacy_values: dict
    provenance: RiskResultProvenance


@dataclass(slots=True)
class WeightedRiskResult:
    weighted_value: float
    component_means: list
    weights: list
    values: list
    output_axis: list
    input_axis: list
    complete_input_axis: list
    jacobian: object
    legacy_values: dict
    provenance: RiskResultProvenance


def _scalar_targets(node: A.Node) -> Iterator[str]:
    """Scalar ``=`` / ``PAYS`` target names in statement order, descending into IF/FOR."""
    if isinstance(node, (A.Assign, A.Pays)):
        yield node.args[0].name
    for child in node.args:
        yield from _scalar_targets(child)


def _distinct_labels(product: ScriptProduct) -> list[str]:
    """Scalar ``=`` / ``PAYS`` target names, deduplicated keeping the first spelling."""
    labels: list[str] = []
    seen: set[str] = set()
    for statement in product.statements():
        for name in _scalar_targets(statement):
            key = ci_key(name)
            if key not in seen:
                seen.add(key)
                labels.append(name)
    return labels


def payoff_outputs(data: ScriptProductData | ScriptProduct) -> list[RiskOutputCoordinate]:
    """One coordinate per distinct scalar ``=``/``PAYS`` target, the last one ``payoff``.

    Mirrors ``Product_Get_RiskOutputs``: events in date order (same-date rows in input
    order), targets deduplicated case-insensitively keeping the first spelling.  Vector
    targets, FOR loop indices and table constants are not payment outputs; with no
    scalar receiver at all (empty, vector-only, or ``EXERCISE`` without ``PAYS``)
    DAL rejects the request.
    """
    product = data.product() if isinstance(data, ScriptProductData) else data
    labels = _distinct_labels(product)
    receiverless = product.has_exercise and not product.has_pays
    if not labels or receiverless:
        raise script_error(
            "InvalidWeightedRiskRequest: indexed scalar default receiver is required; output=payoff"
        )
    return [
        RiskOutputCoordinate(
            id="payoff" if slot == len(labels) - 1 else f"output:{slot}",
            label=label,
            slot=slot,
        )
        for slot, label in enumerate(labels)
    ]
