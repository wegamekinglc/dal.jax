"""Threefry/RBG statistics and parallel stream consistency, with timed DAL MRG32 references."""

import jax.numpy as jnp
import numpy as np
from _common import BS, MATURITY, arguments, compare, finish, model, prepare, settings, table


def standard_errors(n):
    s, v, q, t = BS["spot"], BS["vol"], BS["div"], 3.0
    variance = s * s * np.exp(-2 * q * t) * np.expm1(v * v * t)
    vega_variance = s * s * np.exp(-2 * q * t + v * v * t) * (t + v * v * t * t)
    return {
        "PV": np.sqrt(variance / n),
        "d_spot": np.sqrt(variance / n) / s,
        "d_div": t * np.sqrt(variance / n),
        "d_vol": np.sqrt(vega_variance / n),
        "d_rate": 0.0,
    }


def main():
    args = arguments(__doc__)
    rows = ([MATURITY], ["pay PAYS SPOT()"])
    product = prepare(rows)
    comparisons = []
    errors = standard_errors(args.paths)
    for impl in ("threefry2x32", "rbg"):
        engine = product.engine(model(), settings(args, rsg="mrg32", prng_impl=impl))
        record = compare(
            f"{impl}: statistical comparison (different streams)", engine, rows, args, check=False
        )
        standardized = []
        for name, se in errors.items():
            difference = abs(record["jax"]["result"][name] - record["dal"]["result"][name])
            assert difference <= 3 * np.sqrt(2) * se + 1e-10, (impl, name, difference)  # nosec B101
            standardized.append([name, difference, np.sqrt(2) * se])
        table(["quantity", "difference", "combined standard error"], standardized)
        parallel = product.engine(
            model(), settings(args, rsg="mrg32", prng_impl=impl, parallel="auto")
        )
        actual = parallel.value(args.paths)
        for name, value in actual.items():
            np.testing.assert_allclose(value, record["jax"]["result"][name], rtol=1e-13, atol=1e-10)
        comparisons.append(record)
    expected = float(100 * jnp.exp(-0.03 * 3))
    table(["analytic discounted forward", "PRNG single-estimate SE"], [[expected, errors["PV"]]])
    print(
        "Fixed block size preserves streams across devices/strategies; changing it changes PRNG paths."
    )
    finish(args, comparisons, analytic_pv=expected, standard_errors=errors)


if __name__ == "__main__":
    main()
