"""Scalar event-table products for reproducible CPU/GPU and DAL benchmarks."""

from dal_jax import BlackScholes, prepare
from dal_jax.api import Product_New
from dal_jax.dates import Date

TODAY = Date.ymd(2022, 9, 15)
MATURITY = TODAY.add_days(1095)


def model():
    return BlackScholes(spot=100., vol=.15, rate=.05, div=.03)


def barrier(freq):
    dates = ["STRIKE", "BARRIER", TODAY, f"START: {TODAY} END: {MATURITY} FREQ: {freq}", MATURITY]
    events = ["120", "150", "alive=1", "IF SPOT()>=BARRIER:0.1 THEN alive=0 END",
              "IF SPOT()>=BARRIER:0.1 THEN alive=0 END call PAYS alive*MAX(SPOT()-STRIKE,0)"]
    return dates, events


def event_tables():
    monthly = f"START: {TODAY} END: {MATURITY} FREQ: 1M"
    return {
        "european": (["STRIKE", MATURITY], ["120", "call PAYS MAX(SPOT()-STRIKE,0)"]),
        "barrier_1m": barrier("1M"),
        "barrier_1w": barrier("1W"),
        "asian": (["STRIKE", TODAY, monthly, MATURITY],
                  ["120", "total=0 count=0", "total=total+SPOT() count=count+1", "call PAYS MAX(total/count-STRIKE,0)"]),
        "autocall": (["LEVEL", "COUPON", TODAY, monthly, MATURITY],
                     ["100", "5", "alive=1", "IF alive=1 THEN IF SPOT()>=LEVEL THEN pay PAYS 100+COUPON alive=0 "
                      "ELSE pay PAYS COUPON END END", "IF alive=1 THEN pay PAYS MIN(SPOT(),100) END"]),
    }


def prepared_products():
    return {name: prepare(Product_New(dates, events), TODAY) for name, (dates, events) in event_tables().items()}


def dal_products(dal):
    dal.EvaluationDate_Set(dal.Date_(TODAY.year, TODAY.month, TODAY.day))
    products = {}
    for name, (dates, events) in event_tables().items():
        dates = [dal.Date_(day.year, day.month, day.day) if isinstance(day, Date) else day for day in dates]
        products[name] = dal.Product_New(dates, events)
    return products, dal.BSModelData_New(100., .15, .05, .03)
