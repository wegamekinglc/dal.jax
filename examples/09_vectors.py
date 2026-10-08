"""Mutable vectors, monthly Asian, expanded FOR and fuzzy unequal branch lengths."""

import dal
from _common import TODAY, arguments, compare, finish, model, require_p5_oracle, settings

import dal_jax as dj
from dal_jax.api import Product_New
from dal_jax.errors import EmptyVectorReduction


def main():
    args = arguments(__doc__)
    require_p5_oracle()
    end = TODAY.add_days(365)
    rows = (
        ["K", f"START: {TODAY} END: {end} FREQ: 1M", end],
        ["100", "APPEND(v,SPOT())", "pay PAYS MAX(AVERAGE(v)-K,0)"],
    )
    comparisons = []
    for threshold in (0, 4):
        product = dj.prepare(Product_New(*rows), TODAY)
        engine = product.engine(model(), settings(args, scan_group_threshold=threshold))
        comparisons.append(
            compare(f"Monthly vector Asian; scan threshold {threshold}", engine, rows, args)
        )
    rows = ([end], ["FOR(i,0,3) APPEND(v,SPOT()+i) END v[1]=2 pay PAYS AVERAGE(v)+MAX(v)+MIN(v)"])
    product = dj.prepare(Product_New(*rows), TODAY)
    engine = product.engine(model(), settings(args))
    comparisons.append(compare("FOR, indexed writes and reductions", engine, rows, args))
    rows = (
        ["K", end],
        [
            "100",
            "IF SPOT()>K:4 THEN APPEND(v,SPOT()) APPEND(v,10) ELSE APPEND(v,2) END pay PAYS SUM(v)+AVERAGE(v)",
        ],
    )
    product = dj.prepare(Product_New(*rows), TODAY)
    engine = product.engine(model(), settings(args))
    comparisons.append(compare("Fuzzy branch vector lengths", engine, rows, args))
    bad = dj.prepare(Product_New([TODAY], ["pay PAYS AVERAGE(v)"]), TODAY)
    invalid = bad.engine(model(), settings(args))
    try:
        invalid.value(16)
    except EmptyVectorReduction as error:
        print(f"\nNamed runtime error: {error}")
    else:
        raise AssertionError("empty vector must raise EmptyVectorReduction")
    finish(
        args,
        comparisons,
        vector_capacity=product.variables.vector_capacities,
        error="EmptyVectorReduction",
        dal_source_capability=hasattr(dal, "CorrelatedBSModelData_New"),
    )


if __name__ == "__main__":
    main()
