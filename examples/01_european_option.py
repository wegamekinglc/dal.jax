"""European call script: all Greeks, closed form and timed DAL comparison."""

from copy import copy

import jax
import jax.numpy as jnp
from _common import BS, arguments, compare, european_rows, finish, model, prepare, settings, table
from jax.scipy.stats import norm


def closed_call(spot, vol, rate, div, strike):
    t = 3.0
    d1 = (jnp.log(spot / strike) + (rate - div + 0.5 * vol**2) * t) / (vol * jnp.sqrt(t))
    return spot * jnp.exp(-div * t) * norm.cdf(d1) - strike * jnp.exp(-rate * t) * norm.cdf(
        d1 - vol * jnp.sqrt(t)
    )


def closed_form():
    values = tuple(BS[name] for name in ("spot", "vol", "rate", "div")) + (120.0,)
    result = {"PV": float(closed_call(*values))}
    gradients = jax.grad(closed_call, argnums=(0, 1, 2, 3, 4))(*values)
    result.update(zip(("d_spot", "d_vol", "d_rate", "d_div", "d_STRIKE"), map(float, gradients)))
    return result


def convergence(product, args):
    comparisons = []
    for n in sorted({min(1024, args.paths), min(4096, args.paths), args.paths}):
        local = copy(args)
        local.paths = n
        for rsg in ("sobol", "mrg32"):
            engine = product.engine(model(), settings(args, rsg=rsg, enable_aad=False))
            comparisons.append(
                compare(
                    f"Convergence: {rsg}, {n} paths",
                    engine,
                    european_rows(),
                    local,
                    check=rsg == "sobol",
                )
            )
    print("MRG32 names use different JAX/DAL streams; their convergence comparison is statistical.")
    return comparisons


def main():
    args = arguments(__doc__)
    product = prepare(european_rows())
    comparisons = []
    for aad in (False, True):
        engine = product.engine(model(), settings(args, enable_aad=aad))
        comparisons.append(compare(f"European: enable_aad={aad}", engine, european_rows(), args))
    analytic = closed_form()
    table(
        ["quantity", "Monte Carlo", "closed form", "quadrature difference"],
        [
            [
                name,
                comparisons[-1]["jax"]["result"][name],
                value,
                comparisons[-1]["jax"]["result"][name] - value,
            ]
            for name, value in analytic.items()
        ],
    )
    comparisons.extend(convergence(product, args))
    finish(args, comparisons, closed_form=analytic)


if __name__ == "__main__":
    main()
