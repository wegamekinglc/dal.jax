"""Two-asset basket and worst-of, factor-aware Brownian bridge and default index."""

import dal
from _common import TODAY, VALUATION, arguments, compare, finish, require_p5_oracle, settings

import dal_jax as dj
from dal_jax.api import Product_New
from dal_jax.script.product import ScriptProductSettings


def main():
    args = arguments(__doc__)
    require_p5_oracle()
    fields = dict(
        indices=("EQ[A]", "EQ[B]"),
        spots=(100.0, 95.0),
        vols=(0.15, 0.2),
        divs=(0.03, 0.02),
        rate=0.05,
        correlations=((1.0, 0.4), (0.4, 1.0)),
    )
    bs = dj.CorrelatedBlackScholes(**fields)
    native = dal.CorrelatedBSModelData_New(
        **(fields | {"correlations": dal.DoubleMatrix_(fields["correlations"])})
    )
    end = TODAY.add_days(365)
    comparisons = []
    for label, expression, bridge in [
        ("Basket call", "(FIX(EQ[A])+FIX(EQ[B]))/2", False),
        ("Worst-of with Brownian bridge", "MIN(FIX(EQ[A]),FIX(EQ[B]))", True),
    ]:
        rows = (["K", end], ["100", f"pay PAYS MAX({expression}-K,0)"])
        product = dj.prepare(Product_New(*rows), valuation=VALUATION, model=bs)
        engine = product.engine(bs, settings(args, use_bb=bridge))
        comparisons.append(compare(label, engine, rows, args, bs=native))
    rows = ([end], ["pay PAYS SPOT()+FIX(EQ[B])"])
    product = dj.prepare(
        Product_New(*rows, settings=ScriptProductSettings(default_index="EQ[B]")),
        valuation=VALUATION,
        model=bs,
    )
    native_product = dal.Product_New(
        [dal.Date_(end.year, end.month, end.day)],
        rows[1],
        settings=dal.ScriptProductSettings_(default_index="EQ[B]"),
    )
    engine = product.engine(bs, settings(args))
    comparisons.append(
        compare(
            "SPOT bound to the second equity", engine, rows, args, bs=native, product=native_product
        )
    )
    finish(
        args,
        comparisons,
        factors=bs.n_factors,
        param_labels=bs.param_labels,
        correlation_is_passive=True,
    )


if __name__ == "__main__":
    main()
