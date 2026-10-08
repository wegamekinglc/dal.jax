"""Scalar event tables shared by the P2 simulation and DAL oracle tests."""

from dal_jax.dates import Date

EVALUATION = Date.ymd(2022, 9, 15)
MATURITY = EVALUATION.add_days(365)
QUARTERLY = f"START: {EVALUATION} END: {MATURITY} FREQ: 3M"

CASES = {
    "european_call": (["STRIKE", MATURITY], ["100", "call PAYS MAX(SPOT() - STRIKE, 0)"]),
    "european_put": (["STRIKE", MATURITY], ["100", "put PAYS MAX(STRIKE - SPOT(), 0)"]),
    "asian": (
        ["STRIKE", EVALUATION, QUARTERLY, MATURITY],
        [
            "100",
            "total = 0 count = 0",
            "total = total + SPOT() count = count + 1",
            "call PAYS MAX(total/count - STRIKE, 0)",
        ],
    ),
    "autocall": (
        ["LEVEL", "COUPON", EVALUATION, QUARTERLY, MATURITY],
        [
            "100",
            "5",
            "alive = 1",
            "IF alive = 1 THEN IF SPOT() >= LEVEL THEN pay PAYS 100 + COUPON alive = 0 "
            "ELSE pay PAYS COUPON END END",
            "IF alive = 1 THEN pay PAYS MIN(SPOT(), 100) END",
        ],
    ),
    "arithmetic": (
        ["SCALE", MATURITY],
        [
            "2",
            "x = -(1 + 2) * 3 ^ 2 ^ 0.5 / 4 - +5 "
            "pay PAYS SCALE * (LOG(EXP(x)) + SQRT(SPOT()) - MIN(x, 1) + MAX(1, 2))",
        ],
    ),
    "nested_if": (
        [MATURITY],
        [
            "x = 2 y = 3 IF SPOT() > 100 THEN x = x + y "
            "IF SPOT() > 110 THEN y = x ELSE y = y + 2 END ELSE y = x END pay PAYS x + y"
        ],
    ),
    "conditions": (
        [MATURITY],
        [
            "IF SPOT() > 100 AND SPOT() < 200 OR SPOT() = 150 THEN x = 1 ELSE x = 2 END "
            "IF SPOT() != 100 THEN x = x + 1 END IF SPOT() <= 100 THEN x = x + 2 END pay PAYS x"
        ],
    ),
    "history": (
        ["SCALE", EVALUATION.add_days(-10), MATURITY],
        ["2", "x = SCALE x PAYS 100", "pay PAYS x * SPOT() / 100"],
    ),
    "discarded_history_payment": (
        [EVALUATION.add_days(-1), MATURITY],
        ["pay PAYS 10", "IF pay = 0 THEN pay PAYS SPOT() ELSE pay PAYS 1 END"],
    ),
    "same_date": ([MATURITY, MATURITY, MATURITY], ["x = 1", "y = 2", "pay PAYS x + y"]),
    "case_insensitive": (
        ["Strike", MATURITY],
        ["100", "IF SPOT() > strike THEN X = 1 ELSE x = 2 END P PAYS x"],
    ),
    "dcf": ([QUARTERLY], ["pay PAYS 5 * DCF(ACT365F, PeriodBegin, PeriodEnd)"]),
    "today": ([EVALUATION], ["pay PAYS SPOT()"]),
    "expired": ([EVALUATION.add_days(-1)], ["pay PAYS 7"]),
}


def oracle_dates(dates, dal):
    return [
        dal.Date_(day.year, day.month, day.day) if isinstance(day, Date) else day for day in dates
    ]
