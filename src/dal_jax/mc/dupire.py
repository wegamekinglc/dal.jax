"""Dupire script-risk planners and the quote pullback, mirroring dal-python 2026.11.1.

Mirrors the native ``DupireScriptRiskRequest_``/``DupireQuoteBinding_``/
``CalibrationRiskRequest_`` surface and the ``DupireScriptRiskPlan_New``/
``DupireScriptRiskResult_New`` chain: the plan selects the required surface
nodes plus bound constants, the result values them through the standard AAD
engine and pulls the surface adjoints back into quote space through the
calibration VJP.  The pullback math lives next to the calibration in
:mod:`dal_jax.models.dupire`.
"""

import enum
import math
from dataclasses import dataclass

from dal_jax.errors import (
    CalibrationRiskBudgetExceeded,
    DupireSnapshotMismatch,
    InvalidCalibrationRiskRequest,
    InvalidDupirePullback,
    InvalidDupireRiskRequest,
    InvalidPathCount,
    InvalidSetting,
)
from dal_jax.mc.risk import RiskRequest, RiskResult, input_axis, risk_result
from dal_jax.mc.settings import MonteCarloSettings
from dal_jax.models.base import Model
from dal_jax.models.dupire import (
    DupireCalibrationSnapshot,
    DupireDirectQuoteAdjoints,
    DupireParameterAdjoints,
    DupireQuoteRisk,
    pullback_dupire,
)
from dal_jax.models.hybrid import (
    Hybrid,
    HybridBSEquity,
    HybridDeterministicRate,
    HybridLocalVolEquity,
)
from dal_jax.script.fixings import ValuationSettings
from dal_jax.script.preparation import prepare
from dal_jax.script.product import ScriptProductData

_QUOTE_UNIT = "decimal-vol"
_QUOTE_BYTES_PER_CELL = 24
_INT_MAX = 2147483647


def _detail(value, field, expected):
    """Native InputContext detail; the raising error class prefixes its code."""
    detail = f"type={type(value).__name__}"
    if value is None or isinstance(value, (str, int, float, bytes)):
        detail += f", value={value!r}"
    return f"{field}; {detail}; expected {expected}"


def _text_input(value, field, error):
    if not isinstance(value, str) or "\0" in value:
        raise error(_detail(value, field, "str or DAL String_ without NUL"))
    return value


def _unsigned_input(value, field, error, expected):
    if isinstance(value, bool) or isinstance(value, enum.Enum) or not isinstance(value, int):
        raise error(_detail(value, field, expected))
    if value < 0:
        raise error(_detail(value, field, expected))
    return int(value)


def _ids_input(value, field, error):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise error(_detail(value, field, "list or tuple, or None"))
    return tuple(_text_input(item, field, error) for item in value)


def _factors_input(value, field, error):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise error(_detail(value, field, "list or tuple, or None"))
    factors = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise error(_detail(item, field, "int or float, excluding bool"))
        factors.append(float(item))
    return tuple(factors)


def _path_count(value, field):
    expected = "integer in 1..2147483647, excluding bool and enums"
    if isinstance(value, bool) or isinstance(value, enum.Enum) or not isinstance(value, int):
        raise InvalidPathCount(_detail(value, field, expected))
    if not 1 <= value <= _INT_MAX:
        raise InvalidPathCount(
            f"{_detail(value, field, expected)}; number of Monte Carlo paths must be positive"
        )
    return int(value)


#  ---------------------------------------------------------------------------
#  The calibration quote-selection plan.


@dataclass(frozen=True, slots=True)
class CalibrationQuoteCoordinate:
    """One quote-spread cell, mirroring native ``CalibrationQuoteCoordinate_``."""

    id: str
    label: str
    ordinal: int
    row: int
    column: int
    native_unit: str
    report_scale: float
    value: float
    strike: float
    maturity: float


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationRiskRequest:
    """Quote-grid selection, mirroring native ``CalibrationRiskRequest_``."""

    inputs: object = None
    report_factors: object = None
    numeric_payload_budget_bytes: object = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "inputs",
            _ids_input(
                self.inputs, "CalibrationRiskRequest_; inputs", InvalidCalibrationRiskRequest
            ),
        )
        object.__setattr__(
            self,
            "report_factors",
            _factors_input(
                self.report_factors,
                "CalibrationRiskRequest_; report_factors",
                InvalidCalibrationRiskRequest,
            ),
        )
        object.__setattr__(
            self,
            "numeric_payload_budget_bytes",
            None
            if self.numeric_payload_budget_bytes is None
            else _unsigned_input(
                self.numeric_payload_budget_bytes,
                "CalibrationRiskRequest_; numeric_payload_budget_bytes",
                InvalidCalibrationRiskRequest,
                "nonnegative size_t integer or None, excluding bool",
            ),
        )


@dataclass(frozen=True, slots=True)
class CalibrationRiskPlan:
    """The selected quote axis, mirroring native ``CalibrationRiskPlan_``."""

    calibration: DupireCalibrationSnapshot
    complete_input_axis: tuple
    input_axis: tuple
    selected_ordinals: tuple
    numeric_payload_bytes: int


def _quote_axis(calibration):
    inputs = calibration.inputs
    axis = []
    for row in range(len(inputs.quote_spreads)):
        for column in range(len(inputs.quote_maturities)):
            ordinal = len(axis)
            axis.append(
                CalibrationQuoteCoordinate(
                    id=f"quote:{ordinal}",
                    label=f"spread:{row}:{column}",
                    ordinal=ordinal,
                    row=row,
                    column=column,
                    native_unit=_QUOTE_UNIT,
                    report_scale=1.0,
                    value=inputs.quote_spreads[row][column],
                    strike=inputs.quote_strikes[row],
                    maturity=inputs.quote_maturities[column],
                )
            )
    return tuple(axis)


def _selected_ordinals(axis, request):
    if request.inputs is None:
        return tuple(range(len(axis)))
    available = {coordinate.id: coordinate.ordinal for coordinate in axis}
    seen = set()
    ordinals = []
    for identifier in request.inputs:
        if identifier in seen:
            raise InvalidCalibrationRiskRequest(f"repeated input ID; input={identifier}")
        seen.add(identifier)
        if identifier not in available:
            raise InvalidCalibrationRiskRequest(f"unknown input ID; input={identifier}")
        ordinals.append(available[identifier])
    return tuple(ordinals)


def _selected_axis(axis, ordinals, request):
    if request.report_factors is not None and len(request.report_factors) != len(ordinals):
        raise InvalidCalibrationRiskRequest(
            "report factor count must match selected inputs; field=reportFactors"
        )
    selected = []
    for column, ordinal in enumerate(ordinals):
        coordinate = axis[ordinal]
        scale = (
            coordinate.report_scale
            if request.report_factors is None
            else request.report_factors[column]
        )
        if not math.isfinite(scale) or scale <= 0.0:
            raise InvalidCalibrationRiskRequest(
                f"report factor must be finite and positive; input={coordinate.id}"
            )
        selected.append(
            CalibrationQuoteCoordinate(
                id=coordinate.id,
                label=coordinate.label,
                ordinal=coordinate.ordinal,
                row=coordinate.row,
                column=coordinate.column,
                native_unit=coordinate.native_unit,
                report_scale=scale,
                value=coordinate.value,
                strike=coordinate.strike,
                maturity=coordinate.maturity,
            )
        )
    return tuple(selected)


def plan_quote_selection(calibration, request, *, check_budget=True):
    """Native PlanCalibrationRiskRequest over the flattened quote-spread grid."""
    inputs = calibration.inputs
    rows, columns = len(inputs.quote_spreads), len(inputs.quote_maturities)
    if rows * columns > _INT_MAX:
        raise InvalidCalibrationRiskRequest("quote count exceeds the matrix column limit")
    bytes_ = rows * columns * _QUOTE_BYTES_PER_CELL
    budget = getattr(request, "numeric_payload_budget_bytes", None)
    if check_budget and budget is not None and bytes_ > budget:
        raise CalibrationRiskBudgetExceeded(
            "full numeric contribution payload exceeds numericPayloadBudgetBytes"
        )
    axis = _quote_axis(calibration)
    ordinals = _selected_ordinals(axis, request)
    return CalibrationRiskPlan(
        calibration=calibration,
        complete_input_axis=axis,
        input_axis=_selected_axis(axis, ordinals, request),
        selected_ordinals=ordinals,
        numeric_payload_bytes=bytes_,
    )


#  ---------------------------------------------------------------------------
#  The script-risk request and plan.


@dataclass(frozen=True, slots=True, kw_only=True)
class DupireQuoteBinding:
    """A constant bound to a quote, mirroring native ``DupireQuoteBinding_``."""

    constant_ordinal: int
    quote_id: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "constant_ordinal",
            _unsigned_input(
                self.constant_ordinal,
                "DupireQuoteBinding_; constant_ordinal",
                InvalidDupireRiskRequest,
                "nonnegative size_t integer, excluding bool and enums",
            ),
        )
        object.__setattr__(
            self,
            "quote_id",
            _text_input(self.quote_id, "DupireQuoteBinding_; quote_id", InvalidDupireRiskRequest),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DupireScriptRiskRequest:
    """Script-risk configuration, mirroring native ``DupireScriptRiskRequest_``."""

    num_paths: int
    quotes: object = None
    direct_bindings: object = None
    direct: object = None
    valuation: object = None
    simulation: object = None

    def __post_init__(self):
        object.__setattr__(
            self, "num_paths", _path_count(self.num_paths, "DupireScriptRiskRequest_; num_paths")
        )
        object.__setattr__(
            self,
            "quotes",
            _optional_input(
                self.quotes,
                CalibrationRiskRequest,
                "DupireScriptRiskRequest_; quotes",
                "CalibrationRiskRequest_ or None",
            ),
        )
        object.__setattr__(self, "direct_bindings", _bindings_input(self.direct_bindings))
        #  Divergence: the external-direct path needs the generic
        #  CalibrationDirectQuoteAdjoints_ (shared with the rate-curve side);
        #  until that layer exists any non-None direct is rejected with
        #  native's binding message.
        if self.direct is not None:
            raise InvalidDupireRiskRequest(
                _detail(
                    self.direct,
                    "DupireScriptRiskRequest_; direct",
                    "CalibrationDirectQuoteAdjoints_ or None",
                )
            )
        object.__setattr__(
            self,
            "valuation",
            _optional_input(
                self.valuation,
                ValuationSettings,
                "DupireScriptRiskRequest_; valuation",
                "ScriptValuationSettings_ or None",
            )
            or ValuationSettings(),
        )
        object.__setattr__(
            self,
            "simulation",
            _optional_input(
                self.simulation,
                MonteCarloSettings,
                "DupireScriptRiskRequest_; simulation",
                "MonteCarloSettings_ or None",
            )
            or MonteCarloSettings(enable_aad=True),
        )


def _optional_input(value, accepted, field, expected):
    if value is None or isinstance(value, accepted):
        return value
    raise InvalidSetting(_detail(value, field, expected))


def _bindings_input(value):
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise InvalidDupireRiskRequest(
            _detail(value, "DupireScriptRiskRequest_; direct_bindings", "list or tuple, or None")
        )
    for item in value:
        if not isinstance(item, DupireQuoteBinding):
            raise InvalidDupireRiskRequest(
                _detail(item, "DupireScriptRiskRequest_; direct_bindings", "DupireQuoteBinding_")
            )
    return tuple(value)


@dataclass(frozen=True, slots=True)
class DupireScriptRiskPlan:
    """A frozen script-risk plan, mirroring native ``DupireScriptRiskPlan_``."""

    product: ScriptProductData
    model: Model
    quote_plan: CalibrationRiskPlan
    component: str
    complete_input_axis: tuple
    required_input_axis: tuple
    direct_bindings: tuple
    num_paths: int
    valuation: ValuationSettings
    simulation: MonteCarloSettings
    numeric_payload_bytes: int
    prepared: object = None

    @property
    def valuation_settings(self):
        return self.valuation

    @property
    def simulation_settings(self):
        return self.simulation


def _component_parameters(component):
    return len(component.param_labels)


def _local_vol_equity(model, component):
    """Native SurfaceOffset: locate the component, validate, return its node offset."""
    offset = 0
    for item in sorted(model.components, key=lambda c: c.name.lower()):
        if item.name.lower() == component.lower():
            if not isinstance(item, HybridLocalVolEquity):
                raise InvalidDupirePullback(
                    f"component must be local-vol equity; component={component}"
                )
            return item, offset + 2
        offset += _component_parameters(item)
    raise InvalidDupirePullback(f"unknown component; component={component}")


def _validate_surface(equity, calibration, component):
    surface = equity.surface
    if surface.spots != calibration.surface.spots or surface.times != calibration.surface.times:
        raise DupireSnapshotMismatch(f"component surface grids changed; component={component}")
    if surface.vols != calibration.surface.vols:
        raise DupireSnapshotMismatch(f"component surface values changed; component={component}")


def _validate_carry(model, calibration, component):
    target = _named_local_vol_equity(model, component)
    rate = next(
        (item for item in model.components if isinstance(item, HybridDeterministicRate)), None
    )
    if (
        rate is None
        or rate.rate != calibration.rate
        or target.spot != calibration.spot
        or target.div != calibration.dividend_yield
    ):
        raise DupireSnapshotMismatch(
            f"automatic request requires matching flat deterministic carry; component={component}"
        )
    return target


def _named_local_vol_equity(model, component):
    target = None
    for item in model.components:
        if item.name.lower() == component.lower():
            target = item if isinstance(item, HybridLocalVolEquity) else None
    if target is None:
        raise InvalidDupireRiskRequest(f"named local-vol equity is required; component={component}")
    return target


def _validate_seal(model):
    if not isinstance(model, Hybrid):
        raise InvalidDupireRiskRequest("Hybrid model is required")
    for item in model.components:
        if not isinstance(item, (HybridBSEquity, HybridLocalVolEquity, HybridDeterministicRate)):
            raise InvalidDupireRiskRequest(
                f"unsupported Hybrid component type; component={item.name}"
            )


def _combined_payload(calibration, bindings, budget):
    """Native CombinedPayload: valuation bytes plus the quote payload."""
    nodes = len(calibration.surface.vols) * len(calibration.surface.vols[0])
    quote_bytes = (
        len(calibration.inputs.quote_spreads)
        * len(calibration.inputs.quote_maturities)
        * _QUOTE_BYTES_PER_CELL
    )
    required = 8 * (1 + nodes + len(bindings)) + quote_bytes
    if budget is not None and required > budget:
        raise CalibrationRiskBudgetExceeded(
            "combined valuation/quote payload exceeds numericPayloadBudgetBytes; "
            f"requiredBytes={required}"
        )
    return required


def _with_direct_bindings(complete, required, quote_plan, bindings):
    seen_constants = set()
    seen_quotes = set()
    for binding in bindings:
        constant = _bound_constant(complete, binding)
        quote = _bound_quote(quote_plan, binding)
        if binding.constant_ordinal in seen_constants:
            raise InvalidDupireRiskRequest(f"repeated direct constant; input={constant.id}")
        if quote.id in seen_quotes:
            raise InvalidDupireRiskRequest(f"repeated direct quote; input={quote.id}")
        seen_constants.add(binding.constant_ordinal)
        seen_quotes.add(quote.id)
        if constant.value != quote.value:
            raise InvalidDupireRiskRequest(
                f"direct constant/quote values disagree; input={constant.id}; quote={quote.id}"
            )
        required = required + (constant,)
    return required


def _bound_constant(complete, binding):
    constant = next(
        (
            coordinate
            for coordinate in complete
            if coordinate.family == "constant" and coordinate.ordinal == binding.constant_ordinal
        ),
        None,
    )
    if constant is None:
        raise InvalidDupireRiskRequest(
            f"unknown direct constant ordinal={binding.constant_ordinal}"
        )
    return constant


def _bound_quote(quote_plan, binding):
    quote = next((c for c in quote_plan.complete_input_axis if c.id == binding.quote_id), None)
    if quote is None:
        raise InvalidDupireRiskRequest(f"unknown direct quote ID={binding.quote_id}")
    return quote


def plan_dupire_script_risk(product, model, calibration, component, request):
    """Mirror native PlanDupireScriptRisk."""
    _require(product, ScriptProductData, "DupireScriptRiskPlan_New; product", "ScriptProductData_")
    _require(model, Model, "DupireScriptRiskPlan_New; modelData", "ModelData_")
    _require(
        calibration,
        DupireCalibrationSnapshot,
        "DupireScriptRiskPlan_New; calibration",
        "DupireCalibrationSnapshot_",
    )
    _text_input(component, "DupireScriptRiskPlan_New; component", InvalidDupireRiskRequest)
    _require(
        request,
        DupireScriptRiskRequest,
        "DupireScriptRiskPlan_New; request",
        "DupireScriptRiskRequest_",
    )
    if not request.simulation.enable_aad:
        raise InvalidDupireRiskRequest("native AAD execution is required")
    quotes_request = request.quotes or CalibrationRiskRequest()
    payload = _combined_payload(
        calibration,
        request.direct_bindings,
        quotes_request.numeric_payload_budget_bytes,
    )
    _validate_seal(model)
    _validate_carry(model, calibration, component)
    equity, offset = _local_vol_equity(model, component)
    _validate_surface(equity, calibration, component)
    prepared = prepare(product, model=model, valuation=request.valuation)
    complete = tuple(input_axis(model, prepared.script_params))
    nodes = len(calibration.surface.vols) * len(calibration.surface.vols[0])
    required = tuple(complete[offset : offset + nodes])
    quote_plan = plan_quote_selection(
        calibration,
        CalibrationRiskRequest(
            inputs=quotes_request.inputs, report_factors=quotes_request.report_factors
        ),
        check_budget=False,
    )
    required = _with_direct_bindings(complete, required, quote_plan, request.direct_bindings)
    return DupireScriptRiskPlan(
        product=product,
        model=model,
        quote_plan=quote_plan,
        component=component,
        complete_input_axis=complete,
        required_input_axis=required,
        direct_bindings=request.direct_bindings,
        num_paths=request.num_paths,
        valuation=request.valuation,
        simulation=request.simulation,
        numeric_payload_bytes=payload,
        prepared=prepared,
    )


def _require(value, accepted, field, expected):
    if not isinstance(value, accepted):
        raise InvalidDupireRiskRequest(_detail(value, field, expected))
    return value


#  ---------------------------------------------------------------------------
#  The result chain and the pullback names.


@dataclass(frozen=True, slots=True)
class CalibrationRiskResult:
    """Quote-space rows projected onto the selected quotes, native's wrapper."""

    plan: CalibrationRiskPlan
    quote_risk: DupireQuoteRisk

    @property
    def jacobian(self):
        return _project(self.plan, self.quote_risk.total_adjoints, False)

    @property
    def calibration_jacobian(self):
        return _project(self.plan, self.quote_risk.calibration_adjoints, False)

    @property
    def direct_jacobian(self):
        return _project(self.plan, self.quote_risk.direct_adjoints, False)

    @property
    def reported_jacobian(self):
        return _project(self.plan, self.quote_risk.total_adjoints, True)


def _project(plan, matrix, reported):
    row = []
    for coordinate in plan.input_axis:
        value = matrix[coordinate.row][coordinate.column] * (
            coordinate.report_scale if reported else 1.0
        )
        if not math.isfinite(value):
            raise InvalidCalibrationRiskRequest(
                f"non-finite reported contribution; input={coordinate.id}"
            )
        row.append(value)
    return (tuple(row),)


@dataclass(frozen=True, slots=True)
class DupireScriptRiskResult:
    """The valued plan, mirroring native ``DupireScriptRiskResult_``."""

    valuation: RiskResult
    quote_risk: CalibrationRiskResult
    component: str
    method: str
    numeric_payload_bytes: int


def _surface_seeds(valuation, calibration, offset):
    """Native SurfaceSeeds: map the model-node jacobian columns to seeds."""
    columns = {}
    for column, coordinate in enumerate(valuation.input_axis):
        if coordinate.family != "model":
            continue
        if coordinate.ordinal in columns:
            raise InvalidDupirePullback("repeated selected model ordinal")
        columns[coordinate.ordinal] = column
    rows = len(calibration.surface.vols)
    columns_count = len(calibration.surface.vols[0])
    seeds = [[0.0] * columns_count for _ in range(rows)]
    for row in range(rows):
        for column in range(columns_count):
            ordinal = offset + row * columns_count + column
            if ordinal not in columns:
                raise InvalidDupirePullback(f"missing selected surface risk; input=model:{ordinal}")
            seeds[row][column] = float(valuation.jacobian[0][columns[ordinal]])
    return seeds


def _bound_direct(valuation, quote_plan, bindings):
    if not bindings:
        return None
    calibration = quote_plan.calibration
    seeds = [
        [0.0] * len(calibration.inputs.quote_maturities) for _ in calibration.inputs.quote_spreads
    ]
    start = len(valuation.input_axis) - len(bindings)
    for ordinal, binding in enumerate(bindings):
        quote = next(c for c in quote_plan.complete_input_axis if c.id == binding.quote_id)
        seeds[quote.row][quote.column] = float(valuation.jacobian[0][start + ordinal])
    return DupireDirectQuoteAdjoints(calibration=calibration, adjoints=seeds)


def value_dupire_script_risk(plan):
    """Mirror native ValueByMonteCarloWithDupireRisk."""
    _require(
        plan, DupireScriptRiskPlan, "DupireScriptRiskResult_New; plan", "DupireScriptRiskPlan_"
    )
    request = RiskRequest(inputs=[coordinate.id for coordinate in plan.required_input_axis])
    valuation = risk_result(
        plan.prepared,
        plan.product,
        plan.model,
        plan.simulation,
        plan.num_paths,
        request,
    )
    calibration = plan.quote_plan.calibration
    equity, offset = _local_vol_equity(plan.model, plan.component)
    _validate_surface(equity, calibration, plan.component)
    seeds = _surface_seeds(valuation, calibration, offset)
    direct = _bound_direct(valuation, plan.quote_plan, plan.direct_bindings)
    quote_risk = pullback_dupire(
        calibration, DupireParameterAdjoints(calibration=calibration, adjoints=seeds), direct
    )
    result = CalibrationRiskResult(plan=plan.quote_plan, quote_risk=quote_risk)
    return DupireScriptRiskResult(
        valuation=valuation,
        quote_risk=result,
        component=plan.component,
        method=f"{valuation.provenance.method}Then{quote_risk.method}",
        numeric_payload_bytes=plan.numeric_payload_bytes,
    )


#  ---------------------------------------------------------------------------
#  The standalone pullback names.


@dataclass(frozen=True, slots=True)
class DupireScriptQuoteRisk:
    """A valued pullback, mirroring native ``DupireScriptQuoteRisk_``."""

    valuation: RiskResult
    quote_risk: DupireQuoteRisk
    component: str
    method: str


def dupire_parameter_adjoints_from_risk(valuation, calibration, component):
    """Mirror native ExtractDupireParameterAdjoints."""
    _pullback_input(
        valuation, "DupireParameterAdjoints_FromRisk; valuation", "RiskResult_", RiskResult
    )
    _pullback_input(
        calibration,
        "DupireParameterAdjoints_FromRisk; calibration",
        "DupireCalibrationSnapshot_",
        DupireCalibrationSnapshot,
    )
    _text_input(component, "DupireParameterAdjoints_FromRisk; component", InvalidDupirePullback)
    model = _retained_model(valuation)
    _validate_model_axis(model, valuation.complete_input_axis)
    equity, offset = _local_vol_equity(model, component)
    _validate_surface(equity, calibration, component)
    return DupireParameterAdjoints(
        calibration=calibration,
        adjoints=_surface_seeds(valuation, calibration, offset),
    )


def _pullback_input(value, field, expected, accepted):
    if not isinstance(value, accepted):
        raise InvalidDupirePullback(_detail(value, field, expected))
    return value


def _retained_model(valuation):
    """Native CheckedExecution/ValidateMethod/RestoreModel, condensed."""
    provenance = valuation.provenance
    if provenance.normalization != "mean" or provenance.calibration != "fixed":
        raise InvalidDupirePullback("expected native mean model risk with fixed calibration")
    _require_scalar_payoff(valuation)
    model = provenance.execution.model
    if model is None:
        raise InvalidDupirePullback("modelSnapshotJson is required")
    if not isinstance(model, Hybrid):
        raise InvalidDupirePullback("retained model must be HybridModelData")
    supported = ("NativeAAD", "NativeAADWithRetrainedPolicySecant", "Expired")
    if provenance.method not in supported:
        raise InvalidDupirePullback(f"unsupported valuation method; method={provenance.method}")
    return model


def _require_scalar_payoff(valuation):
    if valuation.output_ids != ["payoff"]:
        raise InvalidDupirePullback(
            "expected one scalar payoff with a matching selected input axis"
        )
    if len(valuation.values) != 1 or len(valuation.jacobian) != 1:
        raise InvalidDupirePullback(
            "expected one scalar payoff with a matching selected input axis"
        )


def _validate_model_axis(model, axis):
    labels = model.param_labels
    params = model.default_params()
    if len(labels) > len(axis):
        raise DupireSnapshotMismatch("complete model parameter extent changed")
    for ordinal, label in enumerate(labels):
        coordinate = axis[ordinal]
        if coordinate.id != f"model:{ordinal}" or coordinate.label != label:
            raise DupireSnapshotMismatch(f"model ordinal identity changed; input={coordinate.id}")
        if coordinate.value != float(params[label]):
            raise DupireSnapshotMismatch(
                f"model coordinate disagrees with retained model; input={coordinate.id}"
            )
    for coordinate in axis[len(labels) :]:
        if coordinate.family != "constant":
            raise DupireSnapshotMismatch("unexpected extra model coordinate")


def dupire_quote_risk(calibration, parameters, direct=None):
    """Mirror native DupireQuoteRisk_New."""
    _pullback_input(
        calibration,
        "DupireQuoteRisk_New; calibration",
        "DupireCalibrationSnapshot_",
        DupireCalibrationSnapshot,
    )
    if not isinstance(parameters, DupireParameterAdjoints):
        raise InvalidDupirePullback(
            _detail(
                parameters, "DupireQuoteRisk_New; parameter_adjoints", "DupireParameterAdjoints_"
            )
        )
    if direct is not None and not isinstance(direct, DupireDirectQuoteAdjoints):
        raise InvalidDupirePullback(
            _detail(direct, "DupireQuoteRisk; direct", "DupireDirectQuoteAdjoints_ or None")
        )
    return pullback_dupire(calibration, parameters, direct)


def dupire_script_quote_risk(valuation, calibration, component, direct=None):
    """Mirror native PullbackDupireScriptRisk."""
    _pullback_input(valuation, "DupireScriptQuoteRisk_New; valuation", "RiskResult_", RiskResult)
    parameters = dupire_parameter_adjoints_from_risk(valuation, calibration, component)
    quote_risk = pullback_dupire(calibration, parameters, direct)
    return DupireScriptQuoteRisk(
        valuation=valuation,
        quote_risk=quote_risk,
        component=component,
        method=f"{valuation.provenance.method}Then{quote_risk.method}",
    )
