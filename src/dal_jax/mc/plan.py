"""BlackScholesMonteCarlo plan API: prepare once, revalue at many model points."""

import math
import numbers
from dataclasses import dataclass, replace

import numpy as np

from dal_jax.dates.date import Date
from dal_jax.errors import script_error
from dal_jax.mc.engine import MonteCarloEngine, PathProduct
from dal_jax.mc.risk import BumpOverAADRequest, validate_request_metadata
from dal_jax.mc.settings import MonteCarloSettings
from dal_jax.models import BlackScholes
from dal_jax.script.fixings import ValuationSettings, resolve_valuation
from dal_jax.script.preparation import PreparedProduct, prepare
from dal_jax.script.product import ScriptProductData, ScriptProductSettings

MODEL_LABELS = BlackScholes.param_labels
RSG_CHOICES = ("sobol", "mrg32", "irn")
PRECISION_CHOICES = ("Default", "Fast", "Precise")
_MAX_PATHS = 2147483647
_DIRECTION_MESSAGE = "BumpOverAAD: directions must have nonnegative rows and one column per input"
#  Native's plan prepares without a model instance; dal_jax's FIX binding is
#  model-aware, so the plan prepares against a canonical Black-Scholes.  The
#  placeholder's parameters never reach a valuation: every revaluation passes
#  the point's parameters explicitly.  Built lazily: constructing a model runs
#  JAX operations, and importing this module must not initialise backends
#  before the caller has configured jax_num_cpu_devices.
_CANONICAL_MODEL: BlackScholes | None = None


def _canonical_model() -> BlackScholes:
    global _CANONICAL_MODEL
    if _CANONICAL_MODEL is None:
        _CANONICAL_MODEL = BlackScholes(spot=1.0, vol=0.1)
    return _CANONICAL_MODEL


@dataclass(frozen=True, slots=True)
class SegmentedMonteCarloSettings:
    """Execution settings of the plan API, mirroring ``SegmentedMonteCarloSettings_``.

    ``first_path`` offsets the Sobol point IDs (the engine's path offset) and
    ``scramble_key`` is a Sobol digital shift.  ``segment_steps`` and the two
    capacity budgets are native execution-plan knobs: validated and echoed,
    but they do not change a valuation — native's values are invariant to them.
    """

    rsg: str = "sobol"
    use_bb: bool = False
    normal_precision: str = "Default"
    scramble_key: int | None = None
    first_path: int = 0
    segment_steps: int = 64
    recording_capacity_budget_bytes: int | None = None
    checkpoint_capacity_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        #  rsg keeps its given spelling (native echoes it back); validation at
        #  valuation time is case-insensitive.
        self._check_bool("use_bb")
        self._check_unsigned("scramble_key", high=2**64 - 1, kind="uint64")
        self._check_unsigned("first_path")
        self._check_unsigned("segment_steps")
        self._check_unsigned("recording_capacity_budget_bytes")
        self._check_unsigned("checkpoint_capacity_budget_bytes")
        if self.segment_steps < 1:
            raise script_error("InvalidSegmentedMonteCarloRequest: segment_steps must be positive")

    def _check_bool(self, name: str) -> None:
        if not isinstance(getattr(self, name), bool):
            self._type_error(name, "expected bool")

    def _check_unsigned(self, name, *, high=math.inf, kind="nonnegative size_t") -> None:
        value = getattr(self, name)
        if value is None:
            return
        integral = isinstance(value, numbers.Integral) and not isinstance(value, bool)
        if not (integral and 0 <= value <= high):
            self._type_error(name, f"expected {kind} integer or None, excluding bool")
        object.__setattr__(self, name, int(value))

    def _type_error(self, name, expected) -> None:
        value = getattr(self, name)
        raise script_error(
            f"InvalidSegmentedMonteCarloRequest: SegmentedMonteCarloSettings_; {name}; "
            f"type={type(value).__name__}, value={value!r}; {expected}"
        )


@dataclass(frozen=True, slots=True)
class RiskObservationSnapshot:
    """A named-index observation of the plan, mirroring ``RiskObservationSnapshot_``."""

    fixing_time: Date
    historical: bool
    index: str
    value: float | None


@dataclass(frozen=True, slots=True)
class BlackScholesMonteCarloPlan:
    """A prepared product plus its valuation context, revalued at model points.

    ``parameter_labels`` are the four Black-Scholes parameters followed by the
    product's script constants; a risk point carries one value per label and
    the returned gradient follows the same order.
    """

    product: ScriptProductData
    valuation: ValuationSettings
    smoothing: float
    parameter_labels: tuple[str, ...]
    contract_dates: tuple
    contract_events: tuple[str, ...]
    contract_settings: ScriptProductSettings
    observations: tuple[RiskObservationSnapshot, ...]
    script_constants: tuple[float, ...]
    prepared: PreparedProduct


@dataclass(frozen=True, slots=True)
class SegmentedMonteCarloExecution:
    """The executed simulation snapshot, mirroring ``SegmentedMonteCarloExecution_``.

    The ``max_path_*_bytes`` fields report native's AAD tape telemetry and stay
    ``None``; the counts describe this engine's block layout.
    """

    path_count: int
    first_path: int
    rsg: str
    scramble_key: int | None
    use_bb: bool
    normal_precision: str
    segment_steps: int
    batches: int
    batch_size: int
    lanes: int
    max_path_checkpoint_bytes: int | None = None
    max_path_cleanup_reserve_bytes: int | None = None
    max_path_tape_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class SegmentedMonteCarloResult:
    value: float
    gradient: list[float]
    parameter_labels: tuple[str, ...]
    execution: SegmentedMonteCarloExecution


@dataclass(frozen=True, slots=True)
class BlackScholesMonteCarloResult:
    """The Get_Risk result, mirroring ``BlackScholesMonteCarloResult_``."""

    value: float
    gradient: list[float]
    parameter_labels: tuple[str, ...]
    point: list[float]
    plan: BlackScholesMonteCarloPlan
    settings: SegmentedMonteCarloSettings
    execution: SegmentedMonteCarloExecution


@dataclass(frozen=True, slots=True)
class MonteCarloCurvatureExecution:
    """The curvature execution snapshot, mirroring ``MonteCarloCurvatureExecution_``."""

    method: str
    gradient_evaluations: int
    recording_capacity_budget_bytes: int | None
    numeric_payload_bytes: int | None = None
    max_path_checkpoint_bytes: int | None = None
    max_path_cleanup_reserve_bytes: int | None = None
    max_path_tape_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class MonteCarloCurvatureResult:
    """The Get_Curvature result, mirroring ``MonteCarloCurvatureResult_``.

    ``hessian_products`` holds one row per direction — the central gradient
    difference ``(g(point + h*d) - g(point - h*d)) / (2h)``, the Hessian times
    each direction, one column per parameter label.
    """

    base: SegmentedMonteCarloResult
    directions: np.ndarray
    hessian_products: np.ndarray
    steps: list[float]
    point: list[float]
    plan: BlackScholesMonteCarloPlan
    settings: SegmentedMonteCarloSettings
    execution: MonteCarloCurvatureExecution


def _require_number(function, field, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise script_error(
            f"InvalidSegmentedMonteCarloRequest: {function}; {field}; "
            f"type={type(value).__name__}, value={value!r}; "
            "expected int or float, excluding bool"
        )
    return float(value)


def _check_point_entries(function, point) -> None:
    if not isinstance(point, (list, tuple)):
        raise script_error(
            f"InvalidSegmentedMonteCarloRequest: {function}; point; "
            f"type={type(point).__name__}; expected list or tuple, or None"
        )
    for value in point:
        _require_number(function, "point", value)


def _check_point_length(point, labels, where=None) -> None:
    """Wrong-size points name the segmented path; curvature wraps it with its context."""
    if len(point) != len(labels):
        message = "BlackScholesSegmentedPath: parameter size must equal ParameterLabels"
        raise script_error(f"MonteCarloCurvature: {where}; {message}" if where else message)


def _check_path_count(function, num_path) -> int:
    #  Intentionally not api._path_count: the plan API reports native's
    #  function-named messages with the 1..2^31-1 range, pinned by the oracle
    #  tests; the generic api-level message must not shadow them.
    if isinstance(num_path, bool) or not isinstance(num_path, numbers.Integral):
        raise script_error(
            f"InvalidPathCount: {function}; num_path / numPath; "
            f"type={type(num_path).__name__}, value={num_path!r}; "
            f"expected a positive integer in 1..{_MAX_PATHS}"
        )
    if not 1 <= num_path <= _MAX_PATHS:
        raise script_error(
            f"InvalidPathCount: {function}; num_path / numPath; "
            f"type={type(num_path).__name__}, value={num_path!r}; "
            f"expected a positive integer in 1..{_MAX_PATHS}; "
            "number of Monte Carlo paths must be positive"
        )
    return int(num_path)


def _validated_plan(function, plan) -> BlackScholesMonteCarloPlan:
    """The bindings-level plan type check: plan is the first argument, so it fires first."""
    if isinstance(plan, BlackScholesMonteCarloPlan):
        return plan
    #  Scalars (str, numbers, bool, None) carry a value segment; other objects do not.
    described = (
        f", value={plan!r}"
        if plan is None or isinstance(plan, (str, numbers.Integral, float))
        else ""
    )
    raise script_error(
        f"InvalidSegmentedMonteCarloRequest: {function}; plan; "
        f"type={type(plan).__name__}{described}; expected BlackScholesMonteCarloPlan_"
    )


def _resolved_settings(function, settings) -> SegmentedMonteCarloSettings:
    if settings is None:
        return SegmentedMonteCarloSettings()
    if type(settings) is not SegmentedMonteCarloSettings:
        raise script_error(
            f"InvalidSegmentedMonteCarloRequest: {function}; settings; "
            f"type={type(settings).__name__}; expected SegmentedMonteCarloSettings_"
        )
    return settings


def _engine_settings(plan, settings: SegmentedMonteCarloSettings) -> MonteCarloSettings:
    """Map segmented settings onto the engine's, with native's validation errors.

    ``rsg`` is case-insensitive and ``normal_precision`` is not; both are
    checked here, at valuation, as in native — never at construction.
    """
    if settings.rsg.lower() not in RSG_CHOICES:
        raise script_error(
            f"InvalidSetting: simulation.rsg_={settings.rsg}; "
            f"expected {', '.join(RSG_CHOICES[:-1])} or {RSG_CHOICES[-1]}; "
            "rng method is not known"
        )
    if settings.normal_precision not in PRECISION_CHOICES:
        raise script_error(
            f"InvalidSetting: simulation.normalPrecision_={settings.normal_precision}; "
            f"expected {', '.join(PRECISION_CHOICES[:-1])} or {PRECISION_CHOICES[-1]}"
        )
    return MonteCarloSettings(
        rsg=settings.rsg.lower(),
        use_bb=settings.use_bb,
        normal_precision=settings.normal_precision,
        sobol_shift_key=settings.scramble_key,
        smooth=plan.smoothing,
        enable_aad=True,
    )


def _model_at(point, where=None) -> BlackScholes:
    """The point's Black-Scholes model; curvature wraps validity errors with its context."""
    try:
        return BlackScholes(**dict(zip(MODEL_LABELS, point[:4])))
    except Exception as error:
        if where is None:
            raise
        raise script_error(f"MonteCarloCurvature: {where}; {error}") from error


def _params(plan: BlackScholesMonteCarloPlan, point) -> dict:
    """Model parameters followed by script constants, in ``parameter_labels`` order."""
    return {
        "model": dict(zip(MODEL_LABELS, (float(value) for value in point[:4]))),
        "script": dict(zip(plan.parameter_labels[4:], (float(value) for value in point[4:]))),
    }


def _engine(plan, model, settings) -> MonteCarloEngine:
    product: PathProduct = plan.prepared.path_product()
    if settings.first_path:
        product = replace(product, path_offset=settings.first_path)
    return MonteCarloEngine(product, model, _engine_settings(plan, settings))


def _execution(settings, num_path, engine) -> SegmentedMonteCarloExecution:
    layout = engine.layout(num_path)
    return SegmentedMonteCarloExecution(
        path_count=num_path,
        first_path=settings.first_path,
        rsg=settings.rsg,
        scramble_key=settings.scramble_key,
        use_bb=settings.use_bb,
        normal_precision=settings.normal_precision,
        segment_steps=settings.segment_steps,
        batches=layout.n_blocks,
        batch_size=layout.block_size,
        lanes=layout.n_blocks,
    )


def _gradient(result, labels) -> list[float]:
    return [float(result[f"d_{label}"]) for label in labels]


def _validate_plan_arguments(product, valuation, smoothing) -> None:
    """The bindings-level checks of ``BlackScholesMonteCarloPlan_New``, in order."""
    if not isinstance(product, ScriptProductData):
        raise script_error(
            "InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarloPlan_New; product; "
            f"type={type(product).__name__}, value={product!r}; expected ScriptProductData_"
        )
    if valuation is not None and not isinstance(valuation, ValuationSettings):
        raise script_error(
            "InvalidSetting: BlackScholesMonteCarloPlan_New; valuation; "
            f"type={type(valuation).__name__}, value={valuation!r}; "
            "expected ScriptValuationSettings_ or None"
        )
    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)):
        raise script_error(
            "InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarloPlan_New; smoothing; "
            f"type={type(smoothing).__name__}, value={smoothing!r}; "
            "expected int or float, excluding bool"
        )
    if not (math.isfinite(smoothing) and smoothing > 0.0):
        raise script_error(
            f"InvalidSetting: InvalidSmoothing; simulation.smooth_={smoothing:.6f}; "
            "expected a finite positive width"
        )


def _observation_snapshots(prepared) -> tuple[RiskObservationSnapshot, ...]:
    """Named-index observations only; the anonymous SPOT() is not a native entry."""
    return tuple(
        RiskObservationSnapshot(
            fixing_time=observation.date,
            historical=observation.historical,
            index=observation.index_name,
            value=observation.value,
        )
        for observation in prepared.observations
        if observation.index_name
    )


def black_scholes_plan(
    product: ScriptProductData, valuation: ValuationSettings | None = None, smoothing: float = 0.01
) -> BlackScholesMonteCarloPlan:
    """Prepare a script product for revaluation at Black-Scholes model points."""
    _validate_plan_arguments(product, valuation, smoothing)
    if product.product().has_exercise:
        raise script_error("BlackScholesSegmentedPath: EXERCISE is unsupported")
    resolved = resolve_valuation(valuation or ValuationSettings())
    prepared = prepare(product, model=_canonical_model(), valuation=resolved)
    return BlackScholesMonteCarloPlan(
        product=product,
        valuation=resolved,
        smoothing=float(smoothing),
        parameter_labels=MODEL_LABELS + tuple(name for name, _ in prepared.script_params),
        contract_dates=tuple(product.dates),
        contract_events=tuple(product.events),
        contract_settings=product.settings,
        observations=_observation_snapshots(prepared),
        script_constants=tuple(float(value) for _, value in prepared.script_params),
        prepared=prepared,
    )


def get_risk(plan, point, num_path, settings=None) -> BlackScholesMonteCarloResult:
    """Value and differentiate the plan at one point, as native Get_Risk."""
    plan = _validated_plan("BlackScholesMonteCarlo_Get_Risk", plan)
    _check_point_entries("BlackScholesMonteCarlo_Get_Risk", point)
    num_path = _check_path_count("BlackScholesMonteCarlo_Get_Risk", num_path)
    settings = _resolved_settings("BlackScholesMonteCarlo_Get_Risk", settings)
    _check_point_length(point, plan.parameter_labels)
    model = _model_at(point)
    engine = _engine(plan, model, settings)
    result = engine.value(num_path, _params(plan, point))
    return BlackScholesMonteCarloResult(
        value=float(result["PV"]),
        gradient=_gradient(result, plan.parameter_labels),
        parameter_labels=plan.parameter_labels,
        point=[float(value) for value in point],
        plan=plan,
        settings=settings,
        execution=_execution(settings, num_path, engine),
    )


def get_curvature(plan, point, num_path, bumps, settings=None) -> MonteCarloCurvatureResult:
    """Bump-over-AAD curvature: one Hessian-times-direction row per bump."""
    function = "BlackScholesMonteCarlo_Get_Curvature"
    plan = _validated_plan(function, plan)
    _check_point_entries(function, point)
    num_path = _check_path_count(function, num_path)
    settings = _resolved_settings(function, settings)
    steps, directions = _validated_bumps(bumps, len(point))
    _check_point_length(point, plan.parameter_labels, where="base")
    model = _model_at(point, "base")
    for ordinal, (direction, step) in enumerate(zip(directions, steps)):
        _model_at(_bumped(point, direction, step), f"direction={ordinal}; plus")
        _model_at(_bumped(point, direction, -step), f"direction={ordinal}; minus")

    engine = _engine(plan, model, settings)
    base_run = engine.value(num_path, _params(plan, point))
    rows = []
    for direction, step in zip(directions, steps):
        g_up = engine.value(num_path, _params(plan, _bumped(point, direction, step)))
        g_down = engine.value(num_path, _params(plan, _bumped(point, direction, -step)))
        rows.append(
            (
                np.asarray(_gradient(g_up, plan.parameter_labels))
                - np.asarray(_gradient(g_down, plan.parameter_labels))
            )
            / (2.0 * step)
        )
    execution = _execution(settings, num_path, engine)
    base = SegmentedMonteCarloResult(
        value=float(base_run["PV"]),
        gradient=_gradient(base_run, plan.parameter_labels),
        parameter_labels=plan.parameter_labels,
        execution=execution,
    )
    return MonteCarloCurvatureResult(
        base=base,
        directions=np.asarray(directions, dtype=float),
        hessian_products=np.asarray(rows, dtype=float).reshape(len(directions), len(point)),
        steps=steps,
        point=[float(value) for value in point],
        plan=plan,
        settings=settings,
        execution=MonteCarloCurvatureExecution(
            method="BumpOverSegmentedNativeAAD",
            gradient_evaluations=1 + 2 * len(directions),
            recording_capacity_budget_bytes=settings.recording_capacity_budget_bytes,
        ),
    )


def _bumped(point, direction, step) -> list[float]:
    return [value + step * entry for value, entry in zip(point, direction)]


def _curvature_number(field, value) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise script_error(
            f"InvalidCurvatureRequest: BumpOverAADRequest_; {field}; "
            f"type={type(value).__name__}, value={value!r}; "
            "expected int or float, excluding bool"
        )


def _validated_steps(steps) -> list[float]:
    """Steps: numbers excluding bool, each finite and positive, as native's Steps."""
    if not isinstance(steps, (list, tuple)):
        raise script_error(
            "InvalidCurvatureRequest: BumpOverAADRequest_; steps; "
            f"type={type(steps).__name__}, value={steps!r}; expected list or tuple, or None"
        )
    for step in steps:
        _curvature_number("steps", step)
        if not math.isfinite(step) or step <= 0.0:
            raise script_error(
                "InvalidCurvatureRequest: BumpOverAADRequest_; steps must be finite and positive"
            )
    return [float(step) for step in steps]


def _numeric_rows(directions) -> None:
    """Directions must be rows (sequences) of numbers, excluding bool."""
    if not isinstance(directions, (list, tuple)) or not all(
        isinstance(row, (list, tuple)) for row in directions
    ):
        raise script_error(_DIRECTION_MESSAGE)
    for row in directions:
        for entry in row:
            _curvature_number("directions", entry)


def _validated_directions(directions, n_steps, n_inputs) -> list[list[float]]:
    """Direction rows of numbers, one per step and one column per input."""
    _numeric_rows(directions)
    if len(directions) != n_steps:
        raise script_error(
            "InvalidCurvatureRequest: BumpOverAADRequest_; steps requires one value per "
            "direction row"
        )
    if not directions or any(len(row) != n_inputs for row in directions):
        raise script_error(_DIRECTION_MESSAGE)
    return [[float(entry) for entry in row] for row in directions]


def _validated_bumps(bumps, n_inputs) -> tuple[list[float], list[list[float]]]:
    """Validate a BumpOverAADRequest as native's NewBumps: steps then directions.

    Steps are checked for type, positivity, then one-per-row; direction rows
    need one column per input, where the count is the point's own size.
    """
    if type(bumps) is not BumpOverAADRequest:
        raise script_error(
            f"InvalidSegmentedMonteCarloRequest: BlackScholesMonteCarlo_Get_Curvature; bumps; "
            f"type={type(bumps).__name__}; expected BumpOverAADRequest_"
        )
    #  Native type-checks the capacity budgets in the constructor; our request
    #  is a plain dataclass, so the checks ride along with its content checks.
    validate_request_metadata(bumps)
    steps = _validated_steps(bumps.steps)
    directions = _validated_directions(bumps.directions, len(steps), n_inputs)
    return steps, directions
