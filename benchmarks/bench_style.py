"""Measure model precomputations and isolated loop alternatives on one device."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dal_jax import BlackScholes, MonteCarloSettings, prepare
from dal_jax.api import Product_New
from dal_jax.dates import Date
from dal_jax.mc.lsmc import Policy, _backward_device
from dal_jax.mc.regression_device import select_device
from dal_jax.models.gsr import GSR, GSRCurve, MultiFactorGSRVol
from dal_jax.models.gsrslv import GSRSLV, GSRLeverage, GSRSLVSettings
from dal_jax.models.hybrid import (
    Hybrid,
    HybridBSEquity,
    HybridGSRRate,
    HybridGSRSLVRate,
    assemble_correlation,
)
from dal_jax.random.sobol import Sobol, directions
from dal_jax.script.lower.lsmc import Records

TODAY = Date.ymd(2026, 10, 2)


def model(kind):
    curve = GSRCurve(
        evaluation_date=TODAY,
        currency="USD",
        node_dates=(TODAY, TODAY.add_days(365), TODAY.add_days(2190)),
        discount_log_df=(0.0, -0.03, -0.18),
    )
    vol = MultiFactorGSRVol(
        factor_names=("level", "slope"),
        g_knot_dates=(TODAY,),
        g_values=((0.02,), (0.01,)),
        h_knot_dates=(TODAY,),
        h_values=((1.0,), (-0.4,)),
        correlations=((1.0, 0.3), (0.3, 1.0)),
    )
    gaussian = GSR(curve=curve, vol=vol)
    smile = GSRSLV(
        gaussian=gaussian,
        leverage=GSRLeverage(
            rate_shifts=(-0.05, 0.05), times=(0.0, 1.0), values=((0.9, 0.95), (1.2, 1.1))
        ),
        settings=GSRSLVSettings(max_step=0.25, variance_correlations=(0.2, -0.1)),
    )
    if kind == "gsr":
        return gaussian
    if kind == "slv":
        return smile
    equity = HybridBSEquity(
        name="A", index="EQ[A]", currency="USD", factor="FA", spot=100.0, vol=0.2, div=0.01
    )
    rate = (
        HybridGSRRate(name="R", model=gaussian, factors=("level", "slope"))
        if kind == "hybrid"
        else HybridGSRSLVRate(name="R", model=smile, vol_factor="FV", bridge_factor="FB")
    )
    return Hybrid(
        domestic_currency="USD",
        components=(rate, equity),
        correlation=assemble_correlation((rate, equity), (("FA", "level", 0.2),)),
    )


def timed(function, inputs, repeat):
    compiled = jax.jit(function)
    started = time.perf_counter()
    lowered = compiled.lower(*inputs)
    lowering = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compilation = time.perf_counter() - started
    output = jax.block_until_ready(executable(*inputs))
    runs = []
    for _ in range(repeat):
        started = time.perf_counter()
        output = jax.block_until_ready(executable(*inputs))
        runs.append(time.perf_counter() - started)
    memory = executable.memory_analysis()
    return output, dict(
        lowering_seconds=lowering,
        compilation_seconds=compilation,
        warm_median_seconds=statistics.median(runs),
        temporary_bytes=memory.temp_size_in_bytes,
    )


def models(args):
    records = []
    for kind in ("gsr", "slv", "hybrid", "hybrid_slv"):
        kernel = model(kind)
        dates = tuple(
            TODAY.add_days(round(365 * i / args.events)) for i in range(1, args.events + 1)
        )
        event = "pay PAYS FIX(IR[USD,LIBOR_3M_CME])+FIX(IR[USD,DF,2028-10-01])"
        if kind.startswith("hybrid"):
            event += "+.01*FIX(EQ[A])"
        settings = MonteCarloSettings(
            platform=args.platform, parallel="none", block_size=128, enable_aad=True
        )
        started = time.perf_counter()
        engine = prepare(Product_New(dates, (event,) * len(dates)), TODAY, model=kernel).engine(
            kernel, settings
        )
        preparation = time.perf_counter() - started
        price = engine.pricer(args.paths)
        output, times = timed(
            jax.value_and_grad(lambda params: price(params)[0]),
            (engine.default_params(),),
            args.repeat,
        )
        pv, risks = output
        values = {"PV": float(pv)} | {
            f"d_{name}": float(value) for group in risks.values() for name, value in group.items()
        }
        records.append(dict(kind=kind, preparation_seconds=preparation, values=values, **times))
        print(kind, times, flush=True)
    return records


def xor_reduce(path_ids, directions):
    points = (path_ids + 1).astype(jnp.uint32)
    gray = points ^ (points >> 1)
    bits = ((gray[:, None] >> jnp.arange(32, dtype=jnp.uint32)) & 1).astype(bool)
    return jnp.bitwise_xor.reduce(
        jnp.where(bits[:, :, None], directions[None, :, :], jnp.uint32(0)), axis=1
    )


def welford_host(x, y):
    means, m2, target = np.zeros(x.shape[1]), np.zeros(x.shape[1]), 0.0
    for count, (row, value) in enumerate(zip(x, y, strict=True), 1):
        target += (value - target) / count
        delta = row - means
        means += delta / count
        m2 += delta * (row - means)
    return means, m2, target


def welford_scan(x, y):
    from dal_jax.mc.regression import ordered_moments

    _, means, m2, target = ordered_moments(x, y, jnp.ones(y.shape, dtype=bool))
    return means, m2, target


def backward_unrolled(records):
    count, features = records.payments.shape[1], records.features.shape[-1]
    coefficients = jnp.zeros((count, 4))
    means, sigmas = jnp.zeros((count, features)), jnp.ones((count, features))
    values = jnp.zeros(records.payments.shape[0])
    for event in reversed(range(count)):
        ratio = (
            records.numeraires[:, event] / records.numeraires[:, event + 1]
            if event + 1 < count
            else 0.0
        )
        targets = records.payments[:, event] + ratio * values
        x, exercise = records.features[:, event], records.exercise_values[:, event]
        included = (exercise > 0.0) & (records.conditions[:, event] > 0.0)
        fit = select_device(x, targets, included, 3)
        coefficients = coefficients.at[event].set(fit.coefficients)
        means, sigmas = means.at[event].set(fit.means), sigmas.at[event].set(fit.sigmas)
        values = jnp.where(included & (exercise > fit.predict(x)), exercise, targets)
    return Policy(coefficients, means, sigmas, jnp.ones(count, dtype=bool))


def experiments(args):
    sequence = Sobol(dim=128)
    ids = jnp.arange(4096)
    dirs = jnp.asarray(directions(sequence.dim))
    reference, original = timed(jax.vmap(sequence.state), (ids,), args.repeat)
    candidate, reduced = timed(xor_reduce, (ids, dirs), args.repeat)
    np.testing.assert_array_equal(reference, candidate)
    rng = np.random.default_rng(1024)
    x, y = rng.normal(size=(16384, 3)), rng.normal(size=16384)
    started = time.perf_counter()
    host = welford_host(x, y)
    host_seconds = time.perf_counter() - started
    scanned, scan_times = timed(welford_scan, (jnp.asarray(x), jnp.asarray(y)), args.repeat)
    for a, b in zip(host, scanned, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    dates = tuple(TODAY.add_days(7 * i) for i in range(1, 17))
    prepared = prepare(
        Product_New(dates, ("EXERCISE MAX(100-SPOT(),0)",) * len(dates)),
        TODAY,
        model=BlackScholes(spot=100.0, vol=0.2),
    )
    features = jnp.asarray(100.0 + 20.0 * rng.normal(size=(512, len(dates), 1)))
    values = jnp.maximum(100.0 - features[:, :, 0], 0.0)
    records = Records(
        jnp.zeros_like(values), values, jnp.ones_like(values), features, jnp.ones_like(values), None
    )
    unrolled, unrolled_times = timed(lambda r: backward_unrolled(r), (records,), args.repeat)
    scanned, backward_times = timed(
        lambda r: _backward_device(prepared, r, None, 3), (records,), args.repeat
    )
    for a, b in zip(unrolled, scanned, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)
    return dict(
        sobol=dict(original=original, xor_reduce=reduced),
        welford=dict(host_seconds=host_seconds, scan=scan_times),
        backward=dict(unrolled=unrolled_times, scan=backward_times),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--paths", type=int, default=257)
    parser.add_argument("--events", type=int, default=8)
    parser.add_argument("--experiments", action="store_true")
    args = parser.parse_args()
    if min(args.repeat, args.paths, args.events) < 1 or args.events > 365:
        parser.error("positive counts and at most 365 events are required")
    device = jax.local_devices(backend=args.platform)[0]
    with jax.default_device(device):
        report = dict(
            jax=jax.__version__,
            python=platform.python_version(),
            device=str(device),
            device_kind=device.device_kind,
            platform=args.platform,
            paths=args.paths,
            events=args.events,
            dtype="float64",
            block_size=128,
            repeat=args.repeat,
            results=models(args),
        )
        if args.experiments:
            report["experiments"] = experiments(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
