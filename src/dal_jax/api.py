"""dal-python compatible functions, so event-table scripts move over unchanged."""

import datetime as _dt
import json
import numbers
import warnings
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from dal_jax.dates.date import Date
from dal_jax.errors import InvalidPathCount, InvalidSetting, script_error
from dal_jax.mc import MonteCarloSettings
from dal_jax.mc.plan import (
    SegmentedMonteCarloSettings,
    _check_path_count,
    black_scholes_plan,
    get_curvature,
    get_risk,
)
from dal_jax.mc.portfolio import (
    PortfolioJacobianRiskRequest,
    PortfolioJacobianRiskResult,
    PortfolioWeightedRiskRequest,
    PortfolioWeightedRiskResult,
    ScriptPortfolio_New,
    ScriptPortfolioData,
    portfolio_jacobian_result,
    portfolio_weighted_result,
)
from dal_jax.mc.risk import (
    BumpOverAADRequest,
    JacobianRiskRequest,
    JacobianRiskResult,
    RiskCoordinate,
    RiskOutputCoordinate,
    RiskRequest,
    RiskResult,
    WeightedRiskRequest,
    WeightedRiskResult,
    jacobian_risk_result,
    payoff_outputs,
    risk_result,
    validate_risk_product,
    weighted_result,
)
from dal_jax.mc.settings import DEFAULT_SMOOTH
from dal_jax.models import (
    IVS,
    BlackScholes,
    CorrelatedBlackScholes,
    DupireCalibrationSnapshot,
    DupireRiskInputs,
    LocalVol,
    LocalVolSurface,
    MertonIVS,
    calibrate_dupire,
)
from dal_jax.models.base import Model
from dal_jax.models.gsr import GSR, GSRCurve, GSRVol, MultiFactorGSRVol
from dal_jax.models.gsrslv import GSRSLV, GSRLeverage, GSRSLVSettings
from dal_jax.models.hybrid import (
    Hybrid,
    HybridBSEquity,
    HybridCorrelation,
    HybridDeterministicRate,
    HybridGSRRate,
    HybridGSRSLVRate,
    HybridLocalVolEquity,
    HybridLogDfRate,
    assemble_correlation,
)
from dal_jax.random.pseudorng import PseudoRSG
from dal_jax.script import diagnostics
from dal_jax.script.fixings import (
    FixingSnapshot,
    TodayFixingPolicy,
    ValuationSettings,
    legacy_context,
    resolve_valuation,
    set_legacy_date,
)
from dal_jax.script.preparation import prepare
from dal_jax.script.product import ScriptProductData, ScriptProductSettings


def EvaluationDate_Set(date: Date | _dt.date) -> None:  # noqa: N802 - dal-python name
    set_legacy_date(_to_date(date))


def EvaluationDate_Get() -> Date:  # noqa: N802
    """Legacy session date; without an explicit date, resolve today on each call."""
    warnings.warn(
        "EvaluationDate_Get is deprecated; use ValuationSession.snapshot",
        DeprecationWarning,
        stacklevel=2,
    )
    return legacy_context().evaluation_date


def _to_date(value: Date | _dt.date) -> Date:
    if isinstance(value, Date):
        return value
    if isinstance(value, _dt.date):
        return Date.from_python(value)
    raise TypeError(f"expected a Date or datetime.date, got {type(value).__name__}")


def _cell(value, row: int):
    if isinstance(value, (Date, _dt.date)):
        return _to_date(value)
    if isinstance(value, str):
        if "\0" in value:
            raise script_error(
                f"InvalidSetting: Product_New; events_dates / dates/events row={row}; value={value!r}; expected text without NUL"
            )
        return value
    raise TypeError(
        f"InvalidSetting: Product_New; events_dates / dates/events row={row}; type={type(value).__name__}; "
        "expected a date or a definition string"
    )


def Product_New(
    events_dates: Sequence, events: Sequence[str], *, settings: ScriptProductSettings | None = None
) -> ScriptProductData:  # noqa: N802
    cells = tuple(_cell(value, row) for row, value in enumerate(events_dates, start=1))
    return ScriptProductData(cells, tuple(events), settings or ScriptProductSettings())


def Product_Describe(product: ScriptProductData) -> dict:  # noqa: N802
    return json.loads(diagnostics.describe(product))


def Product_DebugJson(product: ScriptProductData, *, valuation=None) -> str:  # noqa: N802
    return diagnostics.debug_json(
        product, resolve_valuation(valuation or ValuationSettings()).evaluation_date
    )


def Product_DebugTree(
    product: ScriptProductData, ascii: bool = False, width: int = 125, *, valuation=None
) -> str:  # noqa: N802
    return diagnostics.debug_tree(
        product, resolve_valuation(valuation or ValuationSettings()).evaluation_date, ascii, width
    )


def Product_Debug(product: ScriptProductData, *, valuation=None) -> str:  # noqa: N802
    return diagnostics.debug_text(
        product, resolve_valuation(valuation or ValuationSettings()).evaluation_date
    )


def Product_Get_RiskOutputs(product: ScriptProductData) -> list[RiskOutputCoordinate]:  # noqa: N802
    return payoff_outputs(product)


def BSModelData_New(spot: float, vol: float, rate: float = 0.0, div: float = 0.0) -> BlackScholes:  # noqa: N802
    """DAL-compatible Black-Scholes constructor."""
    return BlackScholes(spot=spot, vol=vol, rate=rate, div=div)


def CorrelatedBSModelData_New(indices, spots, vols, divs, rate, correlations):  # noqa: N802
    return CorrelatedBlackScholes(
        indices=indices, spots=spots, vols=vols, divs=divs, rate=rate, correlations=correlations
    )


def LocalVolSurfaceData_New(name, spots, times, vols):  # noqa: N802
    return LocalVolSurface(name=name, spots=spots, times=times, vols=vols)


def BSLocalVolModelData_New(name, index, currency, factor, bs, surface, max_step=1.0 / 12.0):  # noqa: N802
    return LocalVol(
        name=name,
        index=index,
        currency=currency,
        factor=factor,
        spot=bs.spot,
        rate=bs.rate,
        div=bs.div,
        surface=surface,
        max_step=max_step,
    )


def GSRCurveData_New(
    name,
    evaluation_date,
    currency,
    node_dates,
    discount_log_df,
    projection_tenors=(),
    projection_log_df=(),
):  # noqa: N802
    return GSRCurve(
        name=name,
        evaluation_date=evaluation_date,
        currency=currency,
        node_dates=node_dates,
        discount_log_df=discount_log_df,
        projection_tenors=projection_tenors,
        projection_log_df=projection_log_df,
    )


def GSRVolData_New(name, g_knot_dates, g_values, h_knot_dates, h_values):  # noqa: N802
    return GSRVol(
        name=name,
        g_knot_dates=g_knot_dates,
        g_values=g_values,
        h_knot_dates=h_knot_dates,
        h_values=h_values,
    )


def MultiFactorGSRVolData_New(
    name, factor_names, g_knot_dates, g_values, h_knot_dates, h_values, correlations
):  # noqa: N802
    return MultiFactorGSRVol(
        name=name,
        factor_names=factor_names,
        g_knot_dates=g_knot_dates,
        g_values=g_values,
        h_knot_dates=h_knot_dates,
        h_values=h_values,
        correlations=correlations,
    )


def GSRModelData_New(name, curve, vol):  # noqa: N802
    return GSR(name=name, curve=curve, vol=vol)


MultiFactorGSRModelData_New = GSRModelData_New


def GSRLeverageData_New(name, rate_shifts, times, values):  # noqa: N802
    return GSRLeverage(name=name, rate_shifts=rate_shifts, times=times, values=values)


class GSRSLVSettings_:  # noqa: N801
    """Mutable DAL-compatible input settings, snapshotted by the model factory."""

    def __init__(self):
        self.kappa, self.vol_of_vol, self.variance_correlations, self.max_step = (
            1.0,
            0.5,
            [],
            1.0 / 52.0,
        )


def GSRSLVModelData_New(name, gaussian, leverage, settings=None):  # noqa: N802
    settings = settings or GSRSLVSettings_()
    return GSRSLV(
        name=name,
        gaussian=gaussian,
        leverage=leverage,
        settings=GSRSLVSettings(
            kappa=settings.kappa,
            vol_of_vol=settings.vol_of_vol,
            variance_correlations=tuple(settings.variance_correlations),
            max_step=settings.max_step,
        ),
    )


def HybridBSEquityData_New(name, index, currency, factor, spot, vol, div):  # noqa: N802
    return HybridBSEquity(
        name=name, index=index, currency=currency, factor=factor, spot=spot, vol=vol, div=div
    )


def HybridLocalVolEquityData_New(
    name, index, currency, factor, spot, div, surface, max_step=1.0 / 12.0
):  # noqa: N802
    return HybridLocalVolEquity(
        name=name,
        index=index,
        currency=currency,
        factor=factor,
        spot=spot,
        div=div,
        surface=surface,
        max_step=max_step,
    )


def HybridDeterministicRateData_New(name, currency, rate):  # noqa: N802
    return HybridDeterministicRate(name=name, currency=currency, rate=rate)


def HybridLogDfRateData_New(name, currency, times, log_df, scheme="LOG_LINEAR"):  # noqa: N802
    return HybridLogDfRate(
        name=name, currency=currency, times=times, log_df_values=log_df, scheme=scheme
    )


def HybridGSRRateData_New(name, factor, curve, vol):  # noqa: N802
    return HybridGSRRate(name=name, factors=(factor,), model=GSR(curve=curve, vol=vol))


def HybridGSRRateDataMulti_New(name, factors, curve, multi_vol):  # noqa: N802
    return HybridGSRRate(
        name=name,
        factors=tuple(factors) or multi_vol.factor_names,
        model=GSR(curve=curve, vol=multi_vol),
    )


class HybridFactorLink_(NamedTuple):  # noqa: N801
    factor_a: str
    factor_b: str
    correlation: float = 0.0


def HybridGSRSLVRateData_New(name, vol_factor, bridge_factor, model):  # noqa: N802
    return HybridGSRSLVRate(
        name=name, vol_factor=vol_factor, bridge_factor=bridge_factor, model=model
    )


def HybridConstantCorrelationData_New(name, factor_names, correlations):  # noqa: N802
    return HybridCorrelation(name=name, factor_names=factor_names, correlations=correlations)


def HybridCorrelation_Assemble(name, components, links=()):  # noqa: N802
    return assemble_correlation(components, links, name)


def HybridModelData_New(name, domestic_currency, components, correlation):  # noqa: N802
    return Hybrid(
        name=name,
        domestic_currency=domestic_currency,
        components=tuple(components),
        correlation=correlation,
    )


def MonteCarloSettings_(*, method="sobol", compiled=True, **settings):  # noqa: N802
    return MonteCarloSettings(rsg=method, compiled=compiled, **settings)


DupireRiskInputs_ = DupireRiskInputs
DupireCalibrationSnapshot_ = DupireCalibrationSnapshot
IVS_ = IVS
MertonIVS_ = MertonIVS


def DupireCalibration_New(base, inputs, *, name=""):  # noqa: N802
    """Calibrate the Dupire local-volatility snapshot, as native DupireCalibration_New."""
    return calibrate_dupire(base, inputs, name)


def MonteCarlo_Value(
    product: ScriptProductData,
    model: Model,
    n_paths: int,
    rsg: str = "sobol",
    use_bb: bool = False,  # noqa: N802
    enable_aad: bool = False,
    smooth: float = DEFAULT_SMOOTH,
    compiled: bool | None = None,
    *,
    evaluation_date: Date | _dt.date | None = None,
    historical_spots: Mapping[Date | _dt.date, float] | None = None,
    method: str | None = None,
    valuation: ValuationSettings | None = None,
    fixings: FixingSnapshot | None = None,
    today_fixing_policy: TodayFixingPolicy | str | None = None,
    **execution_settings,
) -> dict[str, float]:
    """Value prepared scripts, returning PV and, with AAD, all ``d_<label>`` risks."""
    _path_count(n_paths)
    rsg = _random_sequence(rsg, method)
    _compiled_option(compiled)
    settings = MonteCarloSettings(
        rsg=rsg,
        use_bb=use_bb,
        enable_aad=enable_aad,
        smooth=smooth,
        compiled=compiled if compiled is not None else True,
        **execution_settings,
    )
    date = None if evaluation_date is None else _to_date(evaluation_date)
    spots = (
        None
        if historical_spots is None
        else {_to_date(day): value for day, value in historical_spots.items()}
    )
    prepared = prepare(
        product,
        date,
        model=model,
        valuation=valuation,
        fixings=fixings,
        today_fixing_policy=today_fixing_policy,
        historical_spots=spots,
    )
    return prepared.engine(model, settings).value(int(n_paths))


TodayFixingPolicy_ = TodayFixingPolicy


def MarketFixingSnapshot_New(values) -> FixingSnapshot:  # noqa: N802
    return FixingSnapshot(values)


def ScriptValuationSettings_(
    *, evaluation_date=None, fixings=None, today_fixing="Model"
) -> ValuationSettings:  # noqa: N802
    return ValuationSettings(
        evaluation_date=evaluation_date, fixings=fixings, today_fixing_policy=today_fixing
    )


def MonteCarlo_ValueWithSettings(product, model, n_paths, *, valuation=None, simulation=None):  # noqa: N802
    """Snapshot valuation settings before preparing and running a Monte Carlo simulation."""
    _path_count(n_paths)
    prepared = prepare(product, model=model, valuation=valuation)
    return prepared.engine(model, simulation).value(int(n_paths))


#  API-level request type names for the InvalidSetting messages; subclasses come first
#  so isinstance matches the most specific request.  Mirrors mc.risk._REQUEST_CODES,
#  which maps the same classes to their constructor-error codes and names.
_REQUEST_NAMES = {
    PortfolioJacobianRiskRequest: "PortfolioJacobianRiskRequest_",
    PortfolioWeightedRiskRequest: "PortfolioWeightedRiskRequest_",
    JacobianRiskRequest: "JacobianRiskRequest_",
    WeightedRiskRequest: "WeightedRiskRequest_",
    BumpOverAADRequest: "BumpOverAADRequest_",
    RiskRequest: "RiskRequest_",
}


def _require_request_type(request, accepted, function, expected) -> None:
    if request is not None and type(request) is not accepted:
        name = next((n for cls, n in _REQUEST_NAMES.items() if isinstance(request, cls)), None)
        raise script_error(
            f"InvalidSetting: {function}; request; type={name or type(request).__name__}; "
            f"expected {expected} or None"
        )


def MonteCarlo_ValueWithRisk(  # noqa: N802
    product, model, n_paths, *, request=None, valuation=None, simulation=None
) -> RiskResult:
    """Value the supported payoff with AAD risks, shaped like native ``RiskResult_``."""
    _path_count(n_paths)
    _require_request_type(request, RiskRequest, "MonteCarlo_ValueWithRisk", "RiskRequest_")
    prepared = prepare(product, model=model, valuation=valuation)
    return risk_result(
        prepared, product, model, simulation or MonteCarloSettings(), int(n_paths), request
    )


def MonteCarlo_ValueWithJacobianRisk(  # noqa: N802
    product, model, n_paths, *, request=None, valuation=None, simulation=None
) -> JacobianRiskResult:
    """Per-slot values and Jacobian rows, as native's Jacobian-risk variant."""
    _path_count(n_paths)
    _require_request_type(
        request, JacobianRiskRequest, "MonteCarlo_ValueWithJacobianRisk", "JacobianRiskRequest_"
    )
    #  Native guards the product before binding, so prepare errors cannot shadow it.
    validate_risk_product(product, valuation, "UnsupportedJacobianRisk", "Jacobian")
    prepared = prepare(product, model=model, valuation=valuation)
    return jacobian_risk_result(
        prepared, product, model, simulation or MonteCarloSettings(), int(n_paths), request
    )


def MonteCarlo_ValueWithWeightedRisk(  # noqa: N802
    product, model, n_paths, *, request=None, valuation=None, simulation=None
) -> WeightedRiskResult:
    """Weighted aggregation over selected output slots, as native's weighted variant."""
    _path_count(n_paths)
    _require_request_type(
        request, WeightedRiskRequest, "MonteCarlo_ValueWithWeightedRisk", "WeightedRiskRequest_"
    )
    #  Native guards the product before binding, so prepare errors cannot shadow it.
    validate_risk_product(product, valuation, "UnsupportedWeightedRisk", "weighted")
    prepared = prepare(product, model=model, valuation=valuation)
    return weighted_result(
        prepared, product, model, simulation or MonteCarloSettings(), int(n_paths), request
    )


def PortfolioMonteCarlo_ValueWithJacobianRisk(  # noqa: N802
    portfolio, n_paths, *, request=None, valuation=None, simulation=None
) -> PortfolioJacobianRiskResult:
    """Per-trade values and one concatenated Jacobian row per selected output."""
    _check_path_count("PortfolioMonteCarlo_ValueWithJacobianRisk", n_paths)
    _require_request_type(
        request,
        PortfolioJacobianRiskRequest,
        "PortfolioMonteCarlo_ValueWithJacobianRisk",
        "PortfolioJacobianRiskRequest_",
    )
    return portfolio_jacobian_result(
        portfolio,
        int(n_paths),
        request=request,
        valuation=valuation,
        simulation=simulation,
    )


def PortfolioMonteCarlo_ValueWithWeightedRisk(  # noqa: N802
    portfolio, n_paths, *, request=None, valuation=None, simulation=None
) -> PortfolioWeightedRiskResult:
    """Weight-aggregated portfolio view: one value and one Jacobian row."""
    _check_path_count("PortfolioMonteCarlo_ValueWithWeightedRisk", n_paths)
    _require_request_type(
        request,
        PortfolioWeightedRiskRequest,
        "PortfolioMonteCarlo_ValueWithWeightedRisk",
        "PortfolioWeightedRiskRequest_",
    )
    return portfolio_weighted_result(
        portfolio,
        int(n_paths),
        request=request,
        valuation=valuation,
        simulation=simulation,
    )


def PseudoRSG_New(seed, ndim=1):  # noqa: N802
    return PseudoRSG(seed, ndim)


def BlackScholesMonteCarloPlan_New(  # noqa: N802
    product: ScriptProductData, *, valuation=None, smoothing: float = 0.01
):
    """Prepare a script product once for revaluation at Black-Scholes model points."""
    return black_scholes_plan(product, valuation=valuation, smoothing=smoothing)


def BlackScholesMonteCarlo_Get_Risk(  # noqa: N802
    plan, point, num_path: int, *, settings=None
):
    """Value and differentiate a plan at one model point, as native Get_Risk."""
    return get_risk(plan, point, num_path, settings=settings)


def BlackScholesMonteCarlo_Get_Curvature(  # noqa: N802
    plan, point, num_path: int, bumps, *, settings=None
):
    """Bump-over-AAD curvature of a plan at one model point, as native Get_Curvature."""
    return get_curvature(plan, point, num_path, bumps, settings=settings)


def PseudoRSG_Get_Uniform(rsg, n_rows):  # noqa: N802
    return rsg.fill_uniform(n_rows)


def PseudoRSG_Get_Normal(rsg, n_rows):  # noqa: N802
    return rsg.fill_normal(n_rows)


RiskCoordinate_ = RiskCoordinate
RiskOutputCoordinate_ = RiskOutputCoordinate
RiskRequest_ = RiskRequest
JacobianRiskRequest_ = JacobianRiskRequest
WeightedRiskRequest_ = WeightedRiskRequest
BumpOverAADRequest_ = BumpOverAADRequest
ScriptPortfolioData_ = ScriptPortfolioData
PortfolioJacobianRiskRequest_ = PortfolioJacobianRiskRequest
PortfolioWeightedRiskRequest_ = PortfolioWeightedRiskRequest
SegmentedMonteCarloSettings_ = SegmentedMonteCarloSettings


def ScriptValuation_Explain(product, model, *, valuation=None):  # noqa: N802
    from dal_jax.script.explain import valuation_explain

    return valuation_explain(product, model, valuation)


def ScriptSimulation_Explain(product, model, n_paths, *, valuation=None, simulation=None):  # noqa: N802
    from dal_jax.script.explain import simulation_explain

    _path_count(n_paths)
    return simulation_explain(product, model, int(n_paths), valuation, simulation)


def _path_count(n_paths: int) -> None:
    if not isinstance(n_paths, numbers.Integral) or isinstance(n_paths, bool) or n_paths <= 0:
        raise InvalidPathCount("number of paths must be a positive integer")


def _random_sequence(rsg: str, method: str | None) -> str:
    if method is None:
        return rsg
    if rsg != "sobol" and rsg != method:
        raise InvalidSetting("method and rsg must name the same random sequence")
    return method


def _compiled_option(compiled: bool | None) -> None:
    if compiled is None:
        return
    if not isinstance(compiled, bool):
        raise TypeError("compiled must be bool or None")
    warnings.warn(
        "compiled only changes reported diagnostics: JAX always compiles the script with XLA",
        UserWarning,
        stacklevel=3,
    )


__all__ = (
    "BSModelData_New",
    "EvaluationDate_Get",
    "EvaluationDate_Set",
    "MonteCarlo_Value",
    "Product_Debug",
    "Product_DebugJson",
    "Product_DebugTree",
    "Product_Describe",
    "Product_New",
    "MarketFixingSnapshot_New",
    "ScriptValuationSettings_",
    "TodayFixingPolicy_",
    "MonteCarlo_ValueWithSettings",
    "CorrelatedBSModelData_New",
    "LocalVolSurfaceData_New",
    "BSLocalVolModelData_New",
)
__all__ += (
    "GSRCurveData_New",
    "GSRVolData_New",
    "MultiFactorGSRVolData_New",
    "GSRModelData_New",
    "MultiFactorGSRModelData_New",
    "GSRLeverageData_New",
    "GSRSLVSettings_",
    "GSRSLVModelData_New",
    "HybridBSEquityData_New",
    "HybridLocalVolEquityData_New",
    "HybridDeterministicRateData_New",
    "HybridLogDfRateData_New",
    "HybridGSRRateData_New",
    "HybridGSRRateDataMulti_New",
    "HybridGSRSLVRateData_New",
    "HybridConstantCorrelationData_New",
    "HybridCorrelation_Assemble",
    "HybridModelData_New",
    "MonteCarloSettings_",
)
__all__ += ("ScriptValuation_Explain", "ScriptSimulation_Explain")
__all__ += ("HybridFactorLink_",)
__all__ += ("PseudoRSG_New", "PseudoRSG_Get_Uniform", "PseudoRSG_Get_Normal")
__all__ += (
    "Product_Get_RiskOutputs",
    "RiskCoordinate_",
    "RiskOutputCoordinate_",
    "RiskRequest_",
    "JacobianRiskRequest_",
    "WeightedRiskRequest_",
    "BumpOverAADRequest_",
)
__all__ += (
    "DupireRiskInputs_",
    "DupireCalibration_New",
    "DupireCalibrationSnapshot_",
    "IVS_",
    "MertonIVS_",
)
__all__ += (
    "MonteCarlo_ValueWithRisk",
    "MonteCarlo_ValueWithJacobianRisk",
    "MonteCarlo_ValueWithWeightedRisk",
)
__all__ += (
    "ScriptPortfolio_New",
    "ScriptPortfolioData_",
    "PortfolioJacobianRiskRequest_",
    "PortfolioWeightedRiskRequest_",
    "PortfolioMonteCarlo_ValueWithJacobianRisk",
    "PortfolioMonteCarlo_ValueWithWeightedRisk",
)
__all__ += (
    "BlackScholesMonteCarloPlan_New",
    "BlackScholesMonteCarlo_Get_Risk",
    "BlackScholesMonteCarlo_Get_Curvature",
    "SegmentedMonteCarloSettings_",
)
