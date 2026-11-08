"""Request-driven risk machinery mirroring DAL 2026.11.1's script risk API."""

from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

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


@dataclass(slots=True)
class JacobianRiskExecution:
    actual_widths: list
    executed_paths: int
    replay_attempts: int


@dataclass(slots=True)
class JacobianRiskResult:
    values: list
    output_axis: list
    complete_output_axis: list
    input_axis: list
    complete_input_axis: list
    jacobian: object
    reported_jacobian: object
    provenance: RiskResultProvenance
    execution: JacobianRiskExecution


def _scalar_targets(node: A.Node) -> Iterator[str]:
    """Scalar ``=`` / ``PAYS`` target names in statement order, descending into IF/FOR.

    Only ``If`` branches and ``Collect`` (the unrolled FOR body) hold nested statements;
    expressions never contain statements, so their subtrees are skipped.
    """
    if isinstance(node, (A.Assign, A.Pays)):
        yield node.args[0].name
    elif isinstance(node, (A.If, A.Collect)):
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


_PHYSICAL_UNITS = (("vol", "year^-1/2"), ("rate", "year^-1"), ("div", "year^-1"))


def _physical_unit(label: str) -> str | None:
    return next((unit for prefix, unit in _PHYSICAL_UNITS if label.startswith(prefix)), None)


def input_axis(model, script_params=()) -> list[RiskCoordinate]:
    """Model coordinates followed by script constants, mirroring DAL's risk axis.

    Model coordinates are ``model:<ordinal>`` in ``param_labels`` order with
    ``vol``/``rate``/``div`` prefixes carrying a physical unit; script constants are
    ``constant:<ordinal>`` with per-family ordinals and no physical unit.
    """
    params = model.default_params()
    axis = [
        RiskCoordinate(
            family="model",
            id=f"model:{ordinal}",
            label=label,
            ordinal=ordinal,
            native_unit="model-coordinate",
            physical_unit=_physical_unit(label),
            report_scale=1.0,
            value=float(params[label]),
        )
        for ordinal, label in enumerate(model.param_labels)
    ]
    return axis + [
        RiskCoordinate(
            family="constant",
            id=f"constant:{ordinal}",
            label=label,
            ordinal=ordinal,
            native_unit="script-number",
            physical_unit=None,
            report_scale=1.0,
            value=float(value),
        )
        for ordinal, (label, value) in enumerate(script_params)
    ]


def _selector_keys(selector, field, code, ctor) -> list[str]:
    if not isinstance(selector, (list, tuple)):
        raise script_error(
            f"{code}: {ctor}; {field}; type={type(selector).__name__}, "
            f"value={selector!r}; expected list or tuple, or None"
        )
    for key in selector:
        if not isinstance(key, str):
            raise script_error(
                f"{code}: {ctor}; {field}; type={type(key).__name__}, "
                f"value={key!r}; expected str or DAL String_ without NUL"
            )
    return list(selector)


_REQUEST_CODES = {  # subclasses first: isinstance matches the most specific request
    JacobianRiskRequest: ("InvalidJacobianRiskRequest", "JacobianRiskRequest_"),
    RiskRequest: ("InvalidRiskRequest", "RiskRequest_"),
}


def _request_codes(request):
    return next(
        (codes for cls, codes in _REQUEST_CODES.items() if isinstance(request, cls)),
        ("InvalidRiskRequest", "RiskRequest_"),
    )


def _selected(axis, selector, request) -> tuple[list[RiskCoordinate], list[int]]:
    """Coordinates chosen by case-insensitive ID in request order, as DAL's ``SelectedInputs``."""
    if selector is None:
        return list(axis), list(range(len(axis)))
    code, ctor = _request_codes(request)
    chosen, indices = [], []
    for key in _selector_keys(selector, "inputs", code, ctor):
        match = ci_key(key)
        found = next((i for i, c in enumerate(axis) if ci_key(c.id) == match), None)
        if found is None:
            raise script_error(f"InvalidRiskRequest: unknown input ID; input={key}")
        if found in indices:
            raise script_error(f"InvalidRiskRequest: repeated input ID; input={key}")
        chosen.append(axis[found])
        indices.append(found)
    return chosen, indices


def _selected_payoff(outputs, selector, request) -> list[RiskOutputCoordinate]:
    """DAL values the single supported payoff; only ``payoff`` (any spelling) selects it."""
    if selector is None:
        return outputs[-1:]
    code, ctor = _request_codes(request)
    keys = _selector_keys(selector, "outputs", code, ctor)
    if len(keys) != 1 or ci_key(keys[0]) != ci_key(outputs[-1].id):
        raise script_error("InvalidRiskRequest: outputs must select the single supported payoff")
    return outputs[-1:]


def _selected_outputs(outputs, selector, request) -> tuple[list[RiskOutputCoordinate], list[int]]:
    """Output slots chosen by case-insensitive ID, several allowed (the Jacobian variant)."""
    if selector is None:
        return outputs[-1:], [len(outputs) - 1]
    code, ctor = _request_codes(request)
    chosen, indices = [], []
    for key in _selector_keys(selector, "outputs", code, ctor):
        match = ci_key(key)
        found = next((i for i, c in enumerate(outputs) if ci_key(c.id) == match), None)
        if found is None:
            raise script_error(
                f"InvalidJacobianRiskRequest: unknown scalar output ID; output={key}"
            )
        if found in indices:
            raise script_error(f"InvalidJacobianRiskRequest: repeated output ID; output={key}")
        chosen.append(outputs[found])
        indices.append(found)
    if not chosen:
        raise script_error(
            "InvalidJacobianRiskRequest: output selection must be nonempty; field=outputs"
        )
    return chosen, indices


def _provenance(prepared, settings, n_paths, method, engine) -> RiskResultProvenance:
    return RiskResultProvenance(
        engine="dal-jax",
        method=method,
        normalization="mean",
        calibration="fixed",
        evaluation_date=str(prepared.evaluation_date),
        execution=RiskExecutionSnapshot(
            paths_per_replicate=n_paths,
            pricing_replicates=settings.lsmc_rqmc_replicates or 1,
            method=method,
            block_width=engine.layout(n_paths).block_size,
        ),
    )


def _risk_fields(engine, prepared, data, n_paths, request) -> dict:
    """Run the AAD engine once and shape the result fields like native ``RiskResult_``."""
    legacy = engine.value(n_paths)
    axis = input_axis(engine.model, prepared.script_params)
    outputs = payoff_outputs(data)
    chosen_outputs = _selected_payoff(outputs, getattr(request, "outputs", None), request)
    chosen_inputs, _ = _selected(axis, getattr(request, "inputs", None), request)
    jacobian = np.asarray([[float(legacy[f"d_{c.label}"]) for c in chosen_inputs]], dtype=float)
    return dict(
        values=[float(legacy["PV"])] * len(chosen_outputs),
        output_ids=[c.id for c in chosen_outputs],
        input_axis=chosen_inputs,
        complete_input_axis=axis,
        output_axis=chosen_outputs,
        jacobian=jacobian,
        reported_jacobian=jacobian,
        legacy_values=legacy,
    )


def value_with_risk(prepared, data, model, settings, n_paths, request=None) -> dict:
    """Risk result fields as a plain dict: one AAD run shaped like ``RiskResult_``."""
    engine = prepared.engine(model, replace(settings, enable_aad=True))
    return _risk_fields(engine, prepared, data, int(n_paths), request)


def risk_result(prepared, data, model, settings, n_paths, request=None, method="NativeAAD"):
    """A :class:`RiskResult` with native-matching provenance for the AAD risk variant."""
    settings = replace(settings, enable_aad=True)
    n_paths = int(n_paths)
    engine = prepared.engine(model, settings)
    return RiskResult(
        provenance=_provenance(prepared, settings, n_paths, method, engine),
        **_risk_fields(engine, prepared, data, n_paths, request),
    )


def jacobian_risk_result(
    prepared, data, model, settings, n_paths, request=None, method="NativeAAD"
):
    """A :class:`JacobianRiskResult`: one AAD replay per selected output slot, as native."""
    settings = replace(settings, enable_aad=True)
    n_paths = int(n_paths)
    outputs = payoff_outputs(data)
    chosen_outputs, slots = _selected_outputs(outputs, getattr(request, "outputs", None), request)
    axis = input_axis(model, prepared.script_params)
    chosen_inputs, _ = _selected(axis, getattr(request, "inputs", None), request)
    engine = None
    values, rows = [], []
    for slot in slots:
        engine = replace(prepared, payoff_index=slot).engine(model, settings)
        legacy = engine.value(n_paths)
        values.append(float(legacy["PV"]))
        rows.append([float(legacy[f"d_{c.label}"]) for c in chosen_inputs])
    jacobian = np.asarray(rows, dtype=float)
    return JacobianRiskResult(
        values=values,
        output_axis=chosen_outputs,
        complete_output_axis=outputs,
        input_axis=chosen_inputs,
        complete_input_axis=axis,
        jacobian=jacobian,
        reported_jacobian=jacobian,
        provenance=_provenance(prepared, settings, n_paths, method, engine),
        execution=JacobianRiskExecution(
            actual_widths=[1] * len(values),
            executed_paths=n_paths * len(values),
            replay_attempts=len(values),
        ),
    )
