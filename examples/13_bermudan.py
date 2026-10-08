"""Bermudan put: normalized regression, frozen risk, transforms and phase timings."""

from dataclasses import replace

import dal
import jax
import numpy as np
from _common import (
    TODAY,
    arguments,
    compare,
    finish,
    model,
    oracle_model,
    oracle_product,
    require_p5_oracle,
    settings,
    table,
    timed,
)

import dal_jax as dj
from dal_jax.api import Product_New, ScriptSimulation_Explain


def main():
    args = arguments(__doc__)
    require_p5_oracle()
    dates = [TODAY.add_days(days) for days in (180, 365, 545, 730)]
    rows = (["K"] + dates, ["100"] + ["EXERCISE MAX(K-SPOT(),0)"] * len(dates))
    bs = model()
    prepared = dj.prepare(Product_New(*rows), TODAY, model=bs)
    simulation = settings(
        args, lsmc_training_paths=args.paths, lsmc_basis_degree=3, use_bb=True, block_size=1024
    )
    engine = prepared.engine(bs, simulation)
    result = compare("Bermudan frozen policy", engine, rows, args)
    params = engine.default_params()
    policy, training = timed(lambda: engine.train(args.paths, params), args.repeat)
    _, full = timed(lambda: engine.value(args.paths, params), args.repeat)
    print("\nPhase A/B training and full valuation (milliseconds)")
    table(
        ["work", "first", "warm median"],
        [
            ["training", training["first_seconds"] * 1000, training["warm_median_seconds"] * 1000],
            [
                "training + pricing + risks",
                full["first_seconds"] * 1000,
                full["warm_median_seconds"] * 1000,
            ],
        ],
    )
    prices = engine.pricer(args.paths, policy=policy)
    reverse = jax.jit(jax.jacrev(prices))(params)
    forward = jax.jit(jax.jacfwd(prices))(params)
    for a, b in zip(jax.tree.leaves(reverse), jax.tree.leaves(forward)):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11)
    ladder = jax.jit(
        jax.vmap(lambda spot: prices(params | {"model": params["model"] | {"spot": spot}})[0])
    )(jax.numpy.asarray([80.0, 100.0, 120.0]))
    diagnostic_settings = replace(simulation, enable_aad=False)
    diagnostics = ScriptSimulation_Explain(
        Product_New(*rows), bs, args.paths, simulation=diagnostic_settings
    )
    native = dal.ScriptSimulation_Explain(
        oracle_product(rows),
        oracle_model(),
        args.paths,
        simulation=dal.MonteCarloSettings_(use_bb=True, lsmc_training_paths=args.paths),
    )
    for a, b in zip(diagnostics["exercise_events"], native["exercise_events"]):
        np.testing.assert_allclose(a["coefficients"], b["coefficients"], rtol=1e-8, atol=1e-10)
    table(
        ["date", "degree", "solver", "exercise rate"],
        [
            [e["date"], e["basis_degree"], e["solver"], e["exercise_rate"]]
            for e in diagnostics["exercise_events"]
        ],
    )
    finish(
        args,
        [result],
        training=training,
        end_to_end=full,
        exercise=diagnostics,
        spot_ladder=np.asarray(ladder).tolist(),
    )


if __name__ == "__main__":
    main()
