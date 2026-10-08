"""JAX transforms: Jacobians, JVP, Hessian and a spot ladder, with DAL bump references."""

import jax
import jax.numpy as jnp
import numpy as np
from _common import (
    MATURITY,
    arguments,
    compare,
    dal,
    finish,
    model,
    oracle_model,
    oracle_product,
    prepare,
    settings,
    table,
    timed,
)


def multiple_comparisons(args):
    names = ("call", "put", "forward")
    bodies = ("MAX(SPOT()-STRIKE,0)", "MAX(STRIKE-SPOT(),0)", "SPOT()+0*STRIKE")
    engines = []
    comparisons = []
    for name, body in zip(names, bodies):
        rows = (["STRIKE", MATURITY], ["120", f"{name} PAYS {body}"])
        engine = prepare(rows).engine(model(), settings(args))
        engines.append(engine)
        comparisons.append(compare(f"Multiple-script Jacobian: {name}", engine, rows, args))
    # Each contract is evaluated by the script engine; combine their prices
    # into a vector before applying JAX's Jacobian transform.
    pricers = tuple(engine.pricer(args.paths) for engine in engines)

    def prices(params):
        return jnp.concatenate([price(params) for price in pricers])

    jacobian = jax.jit(jax.jacrev(prices))(engines[0].default_params())
    np.testing.assert_allclose(
        jacobian["model"]["spot"],
        [record["dal"]["result"]["d_spot"] for record in comparisons],
        rtol=1e-8,
        atol=1e-10,
    )
    table(
        ["payoff", "Jacobian delta", "DAL delta"],
        [
            [name, float(jacobian["model"]["spot"][i]), record["dal"]["result"]["d_spot"]]
            for i, (name, record) in enumerate(zip(names, comparisons))
        ],
    )
    return comparisons


def main():
    args = arguments(__doc__)
    # C1 smoothing gives a stable local gamma for the DAL delta-bump comparison.
    rows = (
        ["STRIKE", MATURITY],
        ["120", "IF SPOT()>STRIKE:4 THEN call PAYS SPOT()-STRIKE ELSE call PAYS 0 END"],
    )
    reference = (
        ["STRIKE", MATURITY],
        [
            "120",
            "d=SPOT()-STRIKE y=(d+2)/4 IF d>=2:0.000000000001 THEN call PAYS d "
            "ELSE IF d>-2:0.000000000001 THEN call PAYS d*y*y*(3-2*y) ELSE call PAYS 0 END END",
        ],
    )
    engine = prepare(rows).engine(model(), settings(args, smoothing_kernel="smoothstep"))
    comparisons = [compare("C1 fuzzy call, width 4 vs DAL explicit cubic", engine, reference, args)]
    params = engine.default_params()
    price = engine.pricer(args.paths)

    def scalar(spot):
        return price(params | {"model": params["model"] | {"spot": spot}})[0]

    forward = float(jax.jacfwd(scalar)(100.0))
    reverse = float(jax.jacrev(scalar)(100.0))
    _, tangent = jax.jvp(scalar, (100.0,), (1.0,))
    np.testing.assert_allclose([forward, float(tangent)], reverse, rtol=1e-13, atol=1e-13)
    gamma = float(jax.jit(jax.hessian(scalar))(100.0))
    product = oracle_product(reference)
    h = 1e-5
    bumps = [
        dict(
            dal.MonteCarlo_Value(
                product, oracle_model(spot=100.0 + sign * h), args.paths, "sobol", False, True
            )
        )
        for sign in (-1, 1)
    ]
    dal_gamma = (bumps[1]["d_spot"] - bumps[0]["d_spot"]) / (2 * h)
    np.testing.assert_allclose(gamma, dal_gamma, rtol=1e-5, atol=1e-7)
    levels = [90.0, 95.0, 100.0, 105.0, 110.0]
    spots = jax.device_put(jnp.asarray(levels), params["model"]["spot"].sharding)
    ladder = jax.jit(jax.vmap(jax.value_and_grad(scalar)))
    ours, ours_time = timed(lambda: ladder(spots), args.repeat)

    def dal_ladder():
        return [
            dict(
                dal.MonteCarlo_Value(
                    product, oracle_model(spot=s), args.paths, "sobol", False, True
                )
            )
            for s in levels
        ]

    theirs, theirs_time = timed(dal_ladder, args.repeat)
    table(
        ["spot", "JAX PV", "DAL PV", "JAX delta", "DAL delta"],
        [
            [float(s), float(p), d["PV"], float(g), d["d_spot"]]
            for s, p, g, d in zip(spots, *ours, theirs)
        ],
    )
    np.testing.assert_allclose(ours[0], [r["PV"] for r in theirs], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ours[1], [r["d_spot"] for r in theirs], rtol=1e-8, atol=1e-10)
    table(
        ["Jacfwd delta", "Jacrev delta", "JVP", "Hessian gamma", "DAL delta-bump gamma"],
        [[forward, reverse, float(tangent), gamma, dal_gamma]],
    )
    table(
        ["ladder backend", "warm min (ms)"],
        [
            ["JAX vmap", ours_time["warm_min_seconds"] * 1000],
            ["DAL sequential calls", theirs_time["warm_min_seconds"] * 1000],
        ],
    )
    comparisons.extend(multiple_comparisons(args))
    finish(
        args,
        comparisons,
        delta=reverse,
        gamma=gamma,
        dal_gamma=dal_gamma,
        ladder_spots=[float(s) for s in spots],
        ladder_jax=[[float(v) for v in values] for values in ours],
        ladder_dal=theirs,
        ladder_jax_timing=ours_time,
        ladder_dal_timing=theirs_time,
    )


if __name__ == "__main__":
    main()
