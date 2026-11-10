"""Portfolio-level script risk mirroring DAL 2026.11.1's portfolio risk API.

``ScriptPortfolio_New`` plus the jacobian/weighted portfolio variants, built
from the per-trade risk machinery in :mod:`dal_jax.mc.risk`.  Every behavior
below was probed against the native wheel; the axis grammar is
``model:<owner>:parameter:<ordinal>`` for model coordinates (owners are
identity-deduplicated models in first-occurrence order) followed by
``trade:<position>:constant:<ordinal>`` per trade, and
``trade:<position>:output:<slot>`` / ``trade:<position>:payoff`` for outputs.
"""

import math
import numbers
from dataclasses import dataclass, replace

import numpy as np

from dal_jax.dates.date import Date
from dal_jax.errors import InvalidWeightedRiskRequest, script_error
from dal_jax.mc.risk import (
    _BUDGET_FIELDS,
    _PHYSICAL_UNITS,
    RiskCoordinate,
    RiskExecutionSnapshot,
    RiskOutputCoordinate,
    _described,
    _physical_unit,
    _selector_keys,
    _slot_runs,
    payoff_outputs,
)
from dal_jax.mc.settings import MonteCarloSettings
from dal_jax.models.base import Model
from dal_jax.script.fixings import ValuationSettings, resolve_valuation
from dal_jax.script.preparation import prepare
from dal_jax.script.product import ScriptProductData
from dal_jax.strings import ci_key

#  Native's ``ModelData_`` type names, printed in per-trade provenance.
_NATIVE_MODEL_TYPES = {
    "BlackScholes": "BSModelData_",
    "CorrelatedBlackScholes": "CorrelatedBSModelData_",
    "LocalVol": "BSLocalVolModelData_",
    "GSR": "GSRModelData_",
    "GSRSLV": "GSRSLVModelData_",
    "Hybrid": "HybridModelData_",
}

_MAX_BLOCK_WIDTH = 32768


@dataclass(slots=True)
class ScriptPortfolioData:
    """Field-compatible with native ``ScriptPortfolioData_`` plus the inputs.

    ``products`` and ``models`` are the per-trade lists passed to
    :func:`ScriptPortfolio_New`; native exposes only ``trade_ids``,
    ``model_owners``, ``name`` and ``type``.
    """

    trade_ids: list
    products: list
    models: list
    model_owners: list
    name: str = ""

    @property
    def type(self) -> str:  # noqa: A003 - native attribute name
        return "ScriptPortfolioData"


@dataclass(slots=True)
class PortfolioJacobianRiskRequest:
    inputs: object = None
    outputs: object = None
    report_factors: object = None
    numeric_payload_budget_bytes: int | None = None
    max_block_width: int = 1
    recording_capacity_budget_bytes: int | None = None
    scratch_capacity_budget_bytes: int | None = None


@dataclass(slots=True)
class PortfolioWeightedRiskRequest:
    inputs: object = None
    outputs: object = None
    weights: object = None
    report_factors: object = None
    numeric_payload_budget_bytes: int | None = None
    recording_capacity_budget_bytes: int | None = None
    scratch_capacity_budget_bytes: int | None = None


@dataclass(slots=True)
class PortfolioSampleDefinition:
    """Field-for-field with native ``PortfolioSampleDefinition_``."""

    numeraire: bool
    index_names: tuple
    discount_maturities: tuple
    forward_maturities: tuple
    libor_definitions: tuple


@dataclass(slots=True)
class PortfolioGroupExecution:
    """Field-for-field with native ``PortfolioGroupExecution_``.

    One group per model owner per distinct timeline; trades sharing both are
    simulated on one scenario stream, so native merges them into a single
    group with shared ``trade_positions``.  ``suffix_reversals`` counts one
    tape reversal per path (``generated_scenarios``); ``prefix_reversals``
    counts one per executed replay batch.  Native's batch decomposition
    scales with its thread count (32 threads split 64 paths into 32 batches,
    a 4-vCPU runner into 4), so it reports thread-scaled counts where this
    engine reports its own block layout's batch count — cross-engine equality
    is undefined, the same class of telemetry as the tape byte peaks and the
    plan API's lanes.
    """

    actual_widths: list
    deterministic_numeraire: bool
    evaluator_calls: int
    factors: int
    generated_scenarios: int
    model_owner: int
    prefix_reversals: int
    random_dimension: int
    replay_attempts: int
    sample_dates: list
    sample_definitions: list
    simulation: MonteCarloSettings
    suffix_reversals: int
    timeline: tuple
    trade_positions: list


@dataclass(slots=True)
class PortfolioRiskExecution:
    """Field-for-field with native ``PortfolioRiskExecution_``."""

    groups: list
    peak_recording_bytes: int
    peak_scratch_bytes: int
    requested_max_block_width: int


@dataclass(slots=True)
class PortfolioTradeProvenance:
    """Field-for-field with native ``RiskResultProvenance_`` inside a portfolio."""

    engine: str
    method: str
    normalization: str
    calibration: str
    evaluation_date: Date
    execution: RiskExecutionSnapshot
    model_type: str


@dataclass(slots=True)
class PortfolioRiskProvenance:
    """Field-for-field with native ``PortfolioRiskProvenance_``."""

    engine: str
    method: str
    normalization: str
    calibration: str
    evaluation_date: Date
    trade_ids: list
    model_owners: list
    trades: list


@dataclass(slots=True)
class PortfolioJacobianRiskResult:
    values: list
    output_axis: list
    complete_output_axis: list
    input_axis: list
    complete_input_axis: list
    jacobian: object
    reported_jacobian: object
    provenance: PortfolioRiskProvenance
    execution: PortfolioRiskExecution


@dataclass(slots=True)
class PortfolioWeightedRiskResult:
    weighted_value: float
    component_means: list
    weights: list
    output_axis: list
    complete_output_axis: list
    input_axis: list
    complete_input_axis: list
    jacobian: object
    reported_jacobian: object
    provenance: PortfolioRiskProvenance
    execution: PortfolioRiskExecution


def _native_type_name(value) -> str:
    """The pybind wrapper name native prints for wrapped objects, else ours.

    Models report their registered holder ``ModelData_`` (native's messages do
    too, whatever the concrete model class); concrete names appear only in
    per-trade provenance via ``_NATIVE_MODEL_TYPES``.
    """
    if isinstance(value, Model):
        return "ModelData_"
    if isinstance(value, ScriptProductData):
        return "ScriptProductData_"
    return type(value).__name__


def _sequence(value, field: str, *, none_as_empty: bool = False) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None and none_as_empty:
        return []
    raise script_error(
        f"InvalidScriptPortfolio: ScriptPortfolio_New; {field}; "
        f"type={_native_type_name(value)}{_described(value)}; expected list or tuple, or None"
    )


def _string_ids(trade_ids) -> list:
    ids = _sequence(trade_ids, "trade_ids", none_as_empty=True)
    for trade_id in ids:
        if not isinstance(trade_id, str) or "\0" in trade_id:
            raise script_error(
                f"InvalidScriptPortfolio: ScriptPortfolio_New; trade_ids; "
                f"type={type(trade_id).__name__}{_described(trade_id)}; "
                "expected str or DAL String_ without NUL"
            )
    return ids


def _typed_elements(values, ids, expected, cls, field: str) -> None:
    """Per-element type check naming the offending trade, as native's bindings.

    Callers must already have verified equal lengths (native's bindings check
    the three lengths before per-element conversion of products/modelData).
    """
    for position, value in enumerate(values):
        if not isinstance(value, cls):
            raise script_error(
                f"InvalidScriptPortfolio: ScriptPortfolio_New; trade={ids[position]}; {field}; "
                f"type={_native_type_name(value)}{_described(value)}; expected {expected}"
            )


def _require_equal_lengths(ids, prods, models) -> None:
    if not ids or len(ids) != len(prods) or len(ids) != len(models):
        raise script_error(
            "InvalidScriptPortfolio: trade_ids, products and modelData must have "
            "equal nonzero lengths"
        )


def _require_unique_ids(ids) -> None:
    seen = set()
    for trade_id in ids:
        key = ci_key(trade_id)
        if not trade_id or key in seen:
            raise script_error(
                f"InvalidScriptPortfolio: trade ID must be nonempty and unique; "
                f"field=tradeIds; trade={trade_id}"
            )
        seen.add(key)


def _owner_indices(models) -> list:
    """Owner index per model, deduplicating by identity in first-occurrence order."""
    owners, owner_of = [], {}
    model_owners = []
    for model in models:
        if id(model) not in owner_of:
            owner_of[id(model)] = len(owners)
            owners.append(model)
        model_owners.append(owner_of[id(model)])
    return model_owners


def ScriptPortfolio_New(trade_ids, products, model_data) -> ScriptPortfolioData:  # noqa: N802
    """Build a portfolio, validating like native's ``ScriptPortfolio_New``.

    ``model_data`` is one model per trade; identical model objects are deduped
    into shared owners by identity.  Native's bindings accept ``trade_ids=None``
    as empty (the length check then rejects it) but reject ``None`` for the
    other two fields, and reject a ``None``/scalar ``modelData`` while the
    message offers "or None" — asymmetries kept verbatim.
    """
    ids = _string_ids(trade_ids)
    prods = _sequence(products, "products")
    models = _sequence(model_data, "modelData")
    _require_equal_lengths(ids, prods, models)
    _typed_elements(prods, ids, "ScriptProductData_", ScriptProductData, "products")
    _typed_elements(models, ids, "ModelData_", Model, "modelData")
    _require_unique_ids(ids)
    return ScriptPortfolioData(ids, prods, models, _owner_indices(models))


def _portfolio_message(field: str, value, detail: str, ctor: str = "portfolio request") -> None:
    """Raise native's constructor-shaped metadata error for one bad field."""
    raise script_error(
        f"InvalidPortfolioRiskRequest: {ctor}; {field}; "
        f"type={type(value).__name__}{_described(value)}; {detail}"
    )


def _numeric_sequence(request, field: str, ctor: str) -> None:
    """A list-of-numbers field: sequence shape first, then element types."""
    raw = getattr(request, field, None)
    if raw is None:
        return
    if not isinstance(raw, (list, tuple)):
        _portfolio_message(field, raw, "expected list or tuple, or None", ctor)
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _portfolio_message(field, value, "expected int or float, excluding bool", ctor)


def _metadata_budgets(request) -> None:
    for field in _BUDGET_FIELDS:
        value = getattr(request, field, None)
        valid = value is not None and (
            isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 0
        )
        if value is not None and not valid:
            _portfolio_message(
                field, value, "expected nonnegative size_t integer or None, excluding bool"
            )


def _metadata_width(request) -> None:
    width = getattr(request, "max_block_width", None)
    if width is None:
        return
    if isinstance(width, numbers.Integral) and not isinstance(width, bool):
        if not 1 <= width <= _MAX_BLOCK_WIDTH:
            raise script_error(
                f"InvalidPortfolioRiskRequest: max_block_width must be in [1, {_MAX_BLOCK_WIDTH}]"
            )
        return
    _portfolio_message(
        "max_block_width",
        width,
        "expected nonnegative size_t integer or None, excluding bool",
        "PortfolioJacobianRiskRequest_",
    )


def _portfolio_metadata(request) -> None:
    """Native's constructor-time checks on a portfolio request, at our boundary.

    The generic sequence and budget fields report the literal ctor name
    ``portfolio request`` (native's own quirk); ``weights`` and
    ``max_block_width`` name their request class, and the width range check
    carries no ctor context at all.
    """
    if request is None:
        return
    for field in ("inputs", "outputs"):
        value = getattr(request, field, None)
        if value is not None:
            _selector_keys(value, field, "InvalidPortfolioRiskRequest", "portfolio request")
    _numeric_sequence(request, "report_factors", "portfolio request")
    _metadata_budgets(request)
    _numeric_sequence(request, "weights", "PortfolioWeightedRiskRequest_")
    _metadata_width(request)


def _portfolio_weights(request, outputs) -> list[float]:
    """Weights per selected output, defaulting to units; count and finiteness checks.

    Types were already validated at the metadata boundary; native checks the
    count and finiteness at valuation, a non-finite weight naming its output.
    """
    raw = getattr(request, "weights", None)
    if raw is None:
        return [1.0] * len(outputs)
    if len(raw) != len(outputs):
        raise script_error(
            "InvalidPortfolioWeightedRequest: weight count must match selected outputs; "
            "field=weights"
        )
    for weight, output in zip(raw, outputs):
        if not math.isfinite(float(weight)):
            raise script_error(
                f"InvalidPortfolioWeightedRequest: weight must be finite; output={output.id}"
            )
    return [float(weight) for weight in raw]


def _owner_axis(owner: int, model) -> list[RiskCoordinate]:
    params = model.default_params()
    units = tuple(getattr(model, "parameter_physical_units", _PHYSICAL_UNITS))
    return [
        RiskCoordinate(
            family="model",
            id=f"model:{owner}:parameter:{ordinal}",
            label=label,
            ordinal=ordinal,
            native_unit="model-coordinate",
            physical_unit=_physical_unit(label, units),
            report_scale=1.0,
            value=float(params[label]),
        )
        for ordinal, label in enumerate(model.param_labels)
    ]


def _constant_axis(position: int, script_params) -> list[RiskCoordinate]:
    return [
        RiskCoordinate(
            family="constant",
            id=f"trade:{position}:constant:{ordinal}",
            label=label,
            ordinal=ordinal,
            native_unit="script-number",
            physical_unit=None,
            report_scale=1.0,
            value=float(value),
        )
        for ordinal, (label, value) in enumerate(script_params)
    ]


def _portfolio_output_axis(position: int, outputs) -> list[RiskOutputCoordinate]:
    last = len(outputs) - 1
    return [
        RiskOutputCoordinate(
            id=f"trade:{position}:payoff" if slot == last else f"trade:{position}:output:{slot}",
            label=coordinate.label,
            slot=slot,
        )
        for slot, coordinate in enumerate(outputs)
    ]


class _NoPayoffTrade(Exception):
    """A trade with neither PAYS nor EXERCISE; native's first-party axis check."""


class _TradePlan:
    """One trade's output coordinates, constant coordinates and axis columns."""

    def __init__(self, position, data, owner):
        self.position = position
        self.owner = owner
        product = data.product()
        product.index_variables()
        if not (product.has_pays or product.has_exercise):
            raise _NoPayoffTrade("product must contain a PAYS or EXERCISE payoff")
        self.outputs = _portfolio_output_axis(position, payoff_outputs(data))
        self.constants = _constant_axis(
            position, zip(product.vars.const_names, product.vars.const_values)
        )
        self.labels = ()  # owner parameter labels + constant labels, set with columns
        self.columns = ()  # indices of this trade's inputs in the portfolio axis


def _trade_axes(portfolio, owner_models):
    """Every trade's plan plus the concatenated portfolio input axis.

    Owner parameter blocks come first in owner order, then each trade's
    constants in trade order.  A product with no PAYS or EXERCISE raises the
    axis guard's own cause; an EXERCISE-without-PAYS product passes that check
    but the delegated risk-output axis rejects it, so its cause is the
    delegated message verbatim — both wrapped with the trade context, before
    any selector validation.
    """
    owner_coordinates = {
        owner: _owner_axis(owner, model) for owner, model in enumerate(owner_models)
    }
    trades = []
    for position, (data, owner) in enumerate(zip(portfolio.products, portfolio.model_owners)):
        try:
            trades.append(_TradePlan(position, data, owner))
        except (InvalidWeightedRiskRequest, _NoPayoffTrade) as error:
            raise script_error(
                f"InvalidScriptPortfolio: trade axes cannot be indexed; field=products; "
                f"trade={portfolio.trade_ids[position]}; cause={error}"
            ) from error
    axis, owner_offsets = [], {}
    for owner in sorted(owner_coordinates):
        owner_offsets[owner] = len(axis)
        axis.extend(owner_coordinates[owner])
    for trade in trades:
        start = len(axis)
        axis.extend(trade.constants)
        owner_block = owner_coordinates[trade.owner]
        trade.labels = tuple(coordinate.label for coordinate in owner_block + trade.constants)
        trade.columns = tuple(
            [owner_offsets[trade.owner] + coordinate.ordinal for coordinate in owner_block]
            + list(range(start, len(axis)))
        )
    return trades, axis


def _matched_ids(complete, keys, code, unknown, repeated):
    """Coordinates for case-insensitive ID keys, in request order.

    ``unknown``/``repeated`` format the two semantic errors with the key.
    """
    chosen, indices = [], []
    for key in keys:
        match = ci_key(key)
        found = next((i for i, c in enumerate(complete) if ci_key(c.id) == match), None)
        if found is None:
            raise script_error(unknown.format(code=code, key=key))
        if found in indices:
            raise script_error(repeated.format(code=code, key=key))
        chosen.append(complete[found])
        indices.append(found)
    return chosen, indices


def _selected_portfolio_outputs(complete, selector, code):
    """Outputs chosen by case-insensitive global ID; default is per-trade payoffs."""
    if selector is None:
        indices = [
            index for index, coordinate in enumerate(complete) if coordinate.id.endswith("payoff")
        ]
        return [complete[index] for index in indices], indices
    keys = _selector_keys(selector, "outputs", code, "portfolio request")
    chosen, indices = _matched_ids(
        complete,
        keys,
        code,
        "{code}: unknown scalar output ID; output={key}",
        "{code}: repeated output ID; output={key}",
    )
    if not chosen:
        raise script_error(f"{code}: output selection must be nonempty; field=outputs")
    return chosen, indices


def _selected_portfolio_inputs(complete, selector):
    """Coordinates chosen by case-insensitive global ID, in request order."""
    if selector is None:
        return list(complete), list(range(len(complete)))
    keys = _selector_keys(selector, "inputs", "InvalidPortfolioRiskRequest", "portfolio request")
    return _matched_ids(
        complete,
        keys,
        "InvalidRiskRequest",
        "{code}: unknown input ID; input={key}",
        "{code}: repeated input ID; input={key}",
    )


def _report_factors(request, count):
    """The per-input report factors, count-checked against the selected inputs.

    Native writes each factor into its coordinate's ``report_scale`` and scales
    the reported jacobian column by it; the plain jacobian stays unscaled.
    """
    factors = getattr(request, "report_factors", None)
    if factors is not None and len(factors) != count:
        raise script_error("InvalidRiskRequest: report factor count must match selected inputs")
    return factors


def _scaled_inputs(chosen, factors):
    if factors is None:
        return chosen
    return [
        replace(coordinate, report_scale=float(factor))
        for coordinate, factor in zip(chosen, factors)
    ]


def _guard_trades(portfolio, valuation) -> None:
    """Native plans every trade, so exercise/expired trades reject unconditionally.

    The guard runs after all selector validation (native's order) and applies
    to every trade even one whose outputs were not selected.
    """
    date = resolve_valuation(valuation or ValuationSettings()).evaluation_date
    for data in portfolio.products:
        product = data.product()
        product.index_variables()
        if product.has_exercise or all(day < date for day in product.event_dates):
            raise script_error(
                "UnsupportedPortfolioRisk: exercise and fully expired trades are unsupported"
            )


class _TradeRun:
    """One trade's replay results: slot to (mean, row over the trade's inputs)."""

    def __init__(self, prepared, engine, settings, results):
        self.prepared = prepared
        self.engine = engine
        self.settings = settings
        self.results = results


def _run_trades(portfolio, trades, chosen_outputs, valuation, simulation, n_paths):
    """Prepare every trade and replay each of its selected output slots once."""
    settings = replace(simulation or MonteCarloSettings(), enable_aad=True)
    runs = []
    for trade, data, model in zip(trades, portfolio.products, portfolio.models):
        prepared = prepare(data, model=model, valuation=valuation)
        slots = [
            coordinate.slot
            for coordinate in chosen_outputs
            if _trade_position(coordinate) == trade.position
        ]
        if slots:
            means, rows, engine = _slot_runs(
                prepared, model, settings, n_paths, slots, list(trade.labels)
            )
            results = dict(zip(slots, zip(means, rows)))
        else:
            engine = prepared.engine(model, settings)
            results = {}
        runs.append(_TradeRun(prepared, engine, settings, results))
    return runs


def _trade_position(coordinate) -> int:
    """The owning trade's position, parsed from the ``trade:<position>:`` prefix."""
    return int(coordinate.id.split(":")[1])


def _global_rows(trades, chosen, runs, width):
    """Selected outputs' values and Jacobian rows over the concatenated axis."""
    values, rows = [], []
    for coordinate in chosen:
        trade = trades[_trade_position(coordinate)]
        mean, row = runs[trade.position].results[coordinate.slot]
        values.append(mean)
        full = [0.0] * width
        for column, value in zip(trade.columns, row):
            full[column] = value
        rows.append(full)
    return values, rows


def _group_counts(chosen, trade, weighted):
    """Whether a trade contributes to its group, and with how many replays."""
    selected = [c for c in chosen if _trade_position(c) == trade.position]
    replays = 0 if weighted else len(selected)
    return bool(selected), replays


def _accumulate_groups(trades, runs, chosen, n_paths, weighted):
    """Group trades by (owner, timeline), accumulating positions and counts.

    ``batches`` totals the replay batches this engine executes for the group:
    one block-layout decomposition per replay (per selected trade once for
    the weighted variant, which never replays).
    """
    groups, by_key = [], {}
    for trade, run in zip(trades, runs):
        selected, replays = _group_counts(chosen, trade, weighted)
        blocks = run.engine.layout(n_paths).n_blocks
        key = (trade.owner, run.prepared.timeline)
        if key not in by_key:
            by_key[key] = {
                "owner": trade.owner,
                "positions": [],
                "replays": 0,
                "active": 0,
                "batches": 0,
                "run": run,
            }
            groups.append(by_key[key])
        group = by_key[key]
        group["positions"].append(trade.position)
        group["replays"] += replays
        group["active"] += selected
        if weighted:
            group["batches"] = max(group["batches"], blocks if selected else 0)
        else:
            group["batches"] += blocks * replays
    return groups


def _sample_definitions(prepared) -> list:
    return [
        PortfolioSampleDefinition(
            numeraire=definition.numeraire,
            index_names=tuple(definition.index_names),
            discount_maturities=tuple(definition.discount_mats),
            forward_maturities=(),
            libor_definitions=(),
        )
        for definition in prepared.sample_defs
    ]


def _group_execution(portfolio, group, n_paths, weighted) -> PortfolioGroupExecution:
    """One group's telemetry: replay counts scale the scenario/reversal totals."""
    run = group["run"]
    n_factors = int(getattr(portfolio.models[group["positions"][0]], "n_factors", 1))
    if weighted:
        generated = n_paths if group["active"] else 0
        evaluator = n_paths * group["active"]
    else:
        generated = evaluator = n_paths * group["replays"]
    return PortfolioGroupExecution(
        actual_widths=[] if weighted else [1] * group["replays"],
        deterministic_numeraire=True,
        evaluator_calls=evaluator,
        factors=n_factors,
        generated_scenarios=generated,
        model_owner=group["owner"],
        prefix_reversals=group["batches"],
        random_dimension=len(run.prepared.timeline) * n_factors,
        replay_attempts=0 if weighted else group["replays"],
        sample_dates=list(run.prepared.event_dates),
        sample_definitions=_sample_definitions(run.prepared),
        simulation=run.settings,
        suffix_reversals=generated,
        timeline=run.prepared.timeline,
        trade_positions=group["positions"],
    )


def _execution_groups(portfolio, trades, runs, chosen, n_paths, weighted):
    """One :class:`PortfolioGroupExecution` per (owner, timeline), in trade order.

    Jacobian groups count one replay per selected output (zeroing out when a
    trade has none); weighted groups never replay, and value each selected
    trade once — an unselected trade zeroes its group's counters.
    """
    return [
        _group_execution(portfolio, group, n_paths, weighted)
        for group in _accumulate_groups(trades, runs, chosen, n_paths, weighted)
    ]


def _provenance(portfolio, runs, method, n_paths):
    """Portfolio provenance with one native-shaped entry per trade."""
    trades = []
    for run, model in zip(runs, portfolio.models):
        name = _NATIVE_MODEL_TYPES.get(type(model).__name__, type(model).__name__)
        trades.append(
            PortfolioTradeProvenance(
                engine="dal-jax",
                method="NativeAAD",
                normalization="mean",
                calibration="fixed",
                evaluation_date=run.prepared.evaluation_date,
                execution=RiskExecutionSnapshot(
                    paths_per_replicate=n_paths,
                    pricing_replicates=run.settings.lsmc_rqmc_replicates or 1,
                    method="NativeAAD",
                    block_width=run.engine.layout(n_paths).block_size,
                ),
                model_type=name,
            )
        )
    return PortfolioRiskProvenance(
        engine="dal-jax",
        method=method,
        normalization="mean",
        calibration="fixed",
        evaluation_date=runs[0].prepared.evaluation_date,
        trade_ids=list(portfolio.trade_ids),
        model_owners=list(portfolio.model_owners),
        trades=trades,
    )


def _owner_models(portfolio) -> list:
    owners = {}
    for model, owner in zip(portfolio.models, portfolio.model_owners):
        owners.setdefault(owner, model)
    return [owners[owner] for owner in sorted(owners)]


def _axes_and_outputs(portfolio, request, output_code):
    """Metadata, trade axes and the validated output selection.

    Native's order: request construction (metadata, mirrored at our boundary),
    trade-axis indexing, then the output selector.  The input selector and the
    weights come after, in each variant's own order.
    """
    _portfolio_metadata(request)
    trades, complete_axis = _trade_axes(portfolio, _owner_models(portfolio))
    complete_outputs = [coordinate for trade in trades for coordinate in trade.outputs]
    chosen_outputs, _ = _selected_portfolio_outputs(
        complete_outputs, getattr(request, "outputs", None), output_code
    )
    return trades, complete_axis, complete_outputs, chosen_outputs


def portfolio_jacobian_result(
    portfolio, n_paths, *, request=None, valuation=None, simulation=None
) -> PortfolioJacobianRiskResult:
    """Per-trade values and one concatenated Jacobian row per selected output.

    Native validates the output selector, then the input selector; cross-trade
    blocks are zero by construction (trades value independently) and trades
    sharing a model object share that owner's columns.
    """
    n_paths = int(n_paths)
    trades, complete_axis, complete_outputs, chosen_outputs = _axes_and_outputs(
        portfolio, request, "InvalidPortfolioJacobianRequest"
    )
    chosen_inputs, input_indices = _selected_portfolio_inputs(
        complete_axis, getattr(request, "inputs", None)
    )
    factors = _report_factors(request, len(chosen_inputs))
    chosen_inputs = _scaled_inputs(chosen_inputs, factors)
    _guard_trades(portfolio, valuation)
    runs = _run_trades(portfolio, trades, chosen_outputs, valuation, simulation, n_paths)
    values, rows = _global_rows(trades, chosen_outputs, runs, len(complete_axis))
    jacobian = np.asarray(rows, dtype=float)[:, input_indices]
    reported = jacobian if factors is None else jacobian * np.asarray(factors, dtype=float)
    return PortfolioJacobianRiskResult(
        values=values,
        output_axis=chosen_outputs,
        complete_output_axis=complete_outputs,
        input_axis=chosen_inputs,
        complete_input_axis=complete_axis,
        jacobian=jacobian,
        reported_jacobian=reported,
        provenance=_provenance(portfolio, runs, "NativeAADJacobianPortfolio", n_paths),
        execution=PortfolioRiskExecution(
            groups=_execution_groups(portfolio, trades, runs, chosen_outputs, n_paths, False),
            peak_recording_bytes=0,
            peak_scratch_bytes=0,
            requested_max_block_width=getattr(request, "max_block_width", None) or 1,
        ),
    )


def portfolio_weighted_result(
    portfolio, n_paths, *, request=None, valuation=None, simulation=None
) -> PortfolioWeightedRiskResult:
    """Weight-aggregated portfolio view: one value and one Jacobian row.

    Native validates the output selector, then the weights, then the input
    selector; the Jacobian is the weight-aggregated sum of the per-output rows.
    """
    n_paths = int(n_paths)
    trades, complete_axis, complete_outputs, chosen_outputs = _axes_and_outputs(
        portfolio, request, "InvalidPortfolioWeightedRequest"
    )
    weights = _portfolio_weights(request, chosen_outputs)
    chosen_inputs, input_indices = _selected_portfolio_inputs(
        complete_axis, getattr(request, "inputs", None)
    )
    factors = _report_factors(request, len(chosen_inputs))
    chosen_inputs = _scaled_inputs(chosen_inputs, factors)
    _guard_trades(portfolio, valuation)
    runs = _run_trades(portfolio, trades, chosen_outputs, valuation, simulation, n_paths)
    values, rows = _global_rows(trades, chosen_outputs, runs, len(complete_axis))
    selected = np.asarray(rows, dtype=float)[:, input_indices]
    jacobian = (np.asarray(weights, dtype=float) @ selected).reshape(1, -1)
    reported = jacobian if factors is None else jacobian * np.asarray(factors, dtype=float)
    return PortfolioWeightedRiskResult(
        weighted_value=float(np.dot(weights, values)),
        component_means=values,
        weights=weights,
        output_axis=chosen_outputs,
        complete_output_axis=complete_outputs,
        input_axis=chosen_inputs,
        complete_input_axis=complete_axis,
        jacobian=jacobian,
        reported_jacobian=reported,
        provenance=_provenance(portfolio, runs, "NativeAADWeightedPortfolio", n_paths),
        execution=PortfolioRiskExecution(
            groups=_execution_groups(portfolio, trades, runs, chosen_outputs, n_paths, True),
            peak_recording_bytes=0,
            peak_scratch_bytes=0,
            requested_max_block_width=0,
        ),
    )
