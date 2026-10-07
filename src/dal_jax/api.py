"""dal-python compatible functions, so event-table scripts move over unchanged.

Dates may be :class:`dal_jax.dates.Date` or :class:`datetime.date`; strings are
definitions or schedules exactly as in DAL (a date *string* is a definition
name, not an event date).  ``Product_Describe`` returns the parsed JSON object
like dal-python; the other dumps return strings.
"""

import datetime as _dt
import json
import threading
from collections.abc import Sequence

from dal_jax.dates.date import Date
from dal_jax.errors import script_error
from dal_jax.script import diagnostics
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


__all__ = ["EvaluationDate_Get", "EvaluationDate_Set", "Product_Debug", "Product_DebugJson", "Product_DebugTree", "Product_Describe", "Product_New"]
