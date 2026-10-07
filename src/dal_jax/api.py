"""dal-python compatible functions, so event-table scripts move over unchanged.

Dates may be :class:`dal_jax.dates.Date` or :class:`datetime.date`; strings are
definitions or schedules exactly as in DAL (a date *string* is a definition
name, not an event date).  ``Product_Describe`` returns the parsed JSON object
like dal-python; the other dumps return strings.
"""

import datetime as _dt
import json
import numbers
import threading
import warnings
from collections.abc import Mapping
from collections.abc import Sequence

from dal_jax.dates.date import Date
from dal_jax.errors import InvalidPathCount, InvalidSetting, script_error
from dal_jax.mc import MonteCarloEngine, MonteCarloSettings
from dal_jax.mc.settings import DEFAULT_SMOOTH
from dal_jax.models import BlackScholes
from dal_jax.models.base import Model
from dal_jax.script import diagnostics
from dal_jax.script.preparation import prepare
from dal_jax.script.product import ScriptProductData, ScriptProductSettings

_lock = threading.Lock()
_evaluation_date: Date | None = None


def EvaluationDate_Set(date: Date | _dt.date) -> None:  # noqa: N802 - dal-python name
    global _evaluation_date
    with _lock:
        _evaluation_date = _to_date(date)


def EvaluationDate_Get() -> Date:  # noqa: N802
    """The global evaluation date; like DAL, the first read without a set fixes it to today."""
    global _evaluation_date
    with _lock:
        if _evaluation_date is None:
            _evaluation_date = Date.from_python(_dt.date.today())
        return _evaluation_date


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
            raise script_error(f"InvalidSetting: Product_New; events_dates / dates/events row={row}; value={value!r}; expected text without NUL")
        return value
    raise TypeError(f"InvalidSetting: Product_New; events_dates / dates/events row={row}; type={type(value).__name__}; "
                    "expected a date or a definition string")


def Product_New(events_dates: Sequence, events: Sequence[str], *, settings: ScriptProductSettings | None = None) -> ScriptProductData:  # noqa: N802
    cells = tuple(_cell(value, row) for row, value in enumerate(events_dates, start=1))
    return ScriptProductData(cells, tuple(events), settings or ScriptProductSettings())


def Product_Describe(product: ScriptProductData) -> dict:  # noqa: N802
    return json.loads(diagnostics.describe(product))


def Product_DebugJson(product: ScriptProductData) -> str:  # noqa: N802
    return diagnostics.debug_json(product, EvaluationDate_Get())


def Product_DebugTree(product: ScriptProductData, ascii: bool = False, width: int = 125) -> str:  # noqa: N802
    return diagnostics.debug_tree(product, EvaluationDate_Get(), ascii, width)


def Product_Debug(product: ScriptProductData) -> str:  # noqa: N802
    return diagnostics.debug_text(product, EvaluationDate_Get())


def BSModelData_New(spot: float, vol: float, rate: float = 0.0, div: float = 0.0) -> BlackScholes:  # noqa: N802
    """DAL-compatible Black-Scholes constructor."""
    return BlackScholes(spot=spot, vol=vol, rate=rate, div=div)


def MonteCarlo_Value(product: ScriptProductData, model: Model, n_paths: int, rsg: str = "sobol", use_bb: bool = False,  # noqa: N802
                    enable_aad: bool = False, smooth: float = DEFAULT_SMOOTH, compiled: bool | None = None, *,
                    evaluation_date: Date | _dt.date | None = None, historical_spots: Mapping[Date | _dt.date, float] | None = None,
                    method: str | None = None, **execution_settings) -> dict[str, float]:
    """Value scalar scripts, returning PV and, with AAD, all ``d_<label>`` risks.

    ``method`` aliases ``rsg`` for dal-python callers.  Execution options such
    as ``block_size``, ``parallel`` and ``devices`` go to MonteCarloSettings.
    Historical SPOT values may be supplied by event date.  AAD evaluates
    future events in fuzzy mode; historical assignments always use hard IF.
    """
    _path_count(n_paths)
    rsg = _random_sequence(rsg, method)
    _compiled_option(compiled)
    settings = MonteCarloSettings(rsg=rsg, use_bb=use_bb, enable_aad=enable_aad, smooth=smooth, **execution_settings)
    date = EvaluationDate_Get() if evaluation_date is None else _to_date(evaluation_date)
    spots = None if historical_spots is None else {_to_date(day): value for day, value in historical_spots.items()}
    prepared = prepare(product, date, historical_spots=spots)
    return MonteCarloEngine(prepared.path_product(), model, settings).value(int(n_paths))


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
    warnings.warn("compiled has no effect: JAX always compiles the script with XLA", UserWarning, stacklevel=3)


__all__ = ["BSModelData_New", "EvaluationDate_Get", "EvaluationDate_Set", "MonteCarlo_Value", "Product_Debug", "Product_DebugJson",
           "Product_DebugTree", "Product_Describe", "Product_New"]
