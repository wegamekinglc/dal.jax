"""Script barrier: exact/fuzzy, Brownian bridge, all DAL Greeks and finite differences."""

import jax
import numpy as np
from _common import arguments, barrier_rows, compare, finish, model, prepare, settings, table


def gradient_diagnostics(engine, args):
    params = engine.default_params()

    def with_barrier(barrier):
        return params | {"script": params["script"] | {"BARRIER": barrier}}

    hard = jax.jit(jax.grad(lambda b: engine.pricer(args.paths, fuzzy=False)(with_barrier(b))[0]))(
        150.0
    )
    fuzzy = jax.jit(lambda b: engine.pricer(args.paths, fuzzy=True)(with_barrier(b))[0])
    slope = float(jax.grad(fuzzy)(150.0))
    bump = 1e-6
    fd = float((fuzzy(150.0 + bump) - fuzzy(150.0 - bump)) / (2 * bump))
    np.testing.assert_allclose(slope, fd, rtol=1e-6, atol=1e-8)
    assert float(hard) == 0.0  # nosec B101
    table(
        ["method", "d_BARRIER"],
        [["hard pathwise derivative", float(hard)], ["fuzzy grad", slope], ["common-path FD", fd]],
    )
    return {"hard_d_barrier": float(hard), "fuzzy_d_barrier": slope, "finite_difference": fd}


def main():
    args = arguments(__doc__)
    rows = barrier_rows()
    product = prepare(rows)
    comparisons = []
    for bridge in (False, True):
        for aad in (False, True):
            engine = product.engine(model(), settings(args, use_bb=bridge, enable_aad=aad))
            comparisons.append(
                compare(f"Monthly barrier: bridge={bridge}, aad={aad}", engine, rows, args)
            )
    engine = product.engine(model(), settings(args))
    finish(args, comparisons, **gradient_diagnostics(engine, args))


if __name__ == "__main__":
    main()
