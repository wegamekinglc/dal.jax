"""Immutable fixing snapshots, today's policy, dated FIX and delayed PAYS ON."""

import dal
from _common import TODAY, VALUATION, arguments, compare, finish, model, require_p5_oracle, settings

import dal_jax as dj
from dal_jax.api import Product_New


def main():
    args = arguments(__doc__)
    require_p5_oracle()
    day = dal.Date_(TODAY.year, TODAY.month, TODAY.day)
    end = TODAY.add_days(365)
    snapshot = dj.FixingSnapshot({"EQ[A]": {TODAY: 97.0}})
    native_snapshot = dal.MarketFixingSnapshot_New({"EQ[A]": {dal.DateTime_(day, 0): 97.0}})
    rows = ([TODAY], ["pay PAYS FIX(EQ[A])"])
    comparisons = []
    for policy in ("Model", "RequireHistorical"):
        valuation = dj.ValuationContext(
            evaluation_date=TODAY, fixings=snapshot, today_fixing_policy=policy
        )
        product = dj.prepare(Product_New(*rows), model=model(), valuation=valuation)
        engine = product.engine(model(), settings(args))
        native = dal.ScriptValuationSettings_(
            evaluation_date=day, fixings=native_snapshot, today_fixing=policy
        )
        comparisons.append(compare(f"Today's FIX: {policy}", engine, rows, args, valuation=native))
    observation = TODAY.add_days(180)
    payment = TODAY.add_days(730)
    rows = (["K", end], ["100", f"pay PAYS MAX(FIX(EQ[A],{observation})-K,0) ON {payment}"])
    product = dj.prepare(Product_New(*rows), valuation=VALUATION, model=model())
    engine = product.engine(model(), settings(args))
    comparisons.append(compare("Dated equity FIX and two-year payment", engine, rows, args))
    print(f"\nObservation/sample timeline: {product.timeline}")
    print(f"Discount maturity slots: {[s.discount_mats for s in product.sample_defs]}")
    finish(
        args,
        comparisons,
        timeline=product.timeline,
        observation_requests=len(product.observations),
        discount_maturities=[s.discount_mats for s in product.sample_defs],
    )


if __name__ == "__main__":
    main()
