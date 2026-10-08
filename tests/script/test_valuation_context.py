from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from dal_jax import FixingSnapshot, ValuationContext, ValuationSession, prepare
from dal_jax.api import Product_DebugJson, Product_New
from dal_jax.dates import Date


def test_interleaved_sessions_preserve_date_and_history():
    today = Date.ymd(2026, 1, 1)
    past = today.add_days(-1)
    product = Product_New([past, today.add_days(5)], ["x=FIX(EQ[A])", "pay PAYS x"])
    barrier = Barrier(2)

    def request(offset, quote):
        context = ValuationContext(
            evaluation_date=today.add_days(offset), fixings=FixingSnapshot({"EQ[A]": {past: quote}})
        )
        session = ValuationSession(context)
        captured = session.snapshot()
        barrier.wait()
        session.update(ValuationContext(evaluation_date=today.add_days(3)))
        prepared = prepare(product, valuation=captured)
        return (
            prepared.evaluation_date,
            prepared.initial_values[0],
            Product_DebugJson(
                Product_New([today, today.add_days(5)], ["x=1", "pay PAYS x"]), valuation=captured
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(request, 0, 80.0), executor.submit(request, 1, 90.0)]
        first, second = (future.result() for future in futures)
    assert first[:2] == (today, 80.0)
    assert second[:2] == (today.add_days(1), 90.0)
    assert first[2] != second[2]


def test_large_snapshot_lookup_preserves_aliases_time_and_reverse_fx():
    import datetime as dt

    day = Date.ymd(2026, 1, 1)
    dates = tuple(day.add_days(i) for i in range(3000))
    snapshot = FixingSnapshot(
        {
            "EQ[A]": {date: float(i) for i, date in enumerate(dates)},
            "FX[EUR/USD]": {dt.datetime(2026, 1, 1, 12): 1.25},
        }
    )
    assert snapshot.find("eq[a]", dates[0]) == 0.0
    assert snapshot.find("EQ[A]", dates[-1]) == 2999.0
    assert snapshot.find("EQ[A]", day.add_days(-1)) is None
    assert snapshot.find("FX[USD/EUR]", dt.datetime(2026, 1, 1, 12)) == 0.8
    assert snapshot.find("FX[EUR/USD]", day) is None
