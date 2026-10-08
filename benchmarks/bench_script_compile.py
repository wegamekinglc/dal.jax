"""Measure script event graph size and compilation with/without scan grouping.

This isolates event lowering from random generation and model allocation.
Inputs are a supplied single-path Scenario and live script parameters; price
and parameter gradients are compiled together. Timings are descriptive, not
test assertions. Long unrolled graphs are measured but only compiled with
``--unrolled-limit DAYS`` (default 64), since their compilation can take minutes.
Run ``uv run python benchmarks/bench_script_compile.py``.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from dal_jax import EvalContext, config, prepare
from dal_jax.api import Product_New
from dal_jax.dates import Date
from dal_jax.models.base import Scenario


def count_equations(graph):
    """Count equations including nested control-flow and jit bodies."""
    graph = getattr(graph, "jaxpr", graph)
    total = len(graph.eqns)
    for equation in graph.eqns:
        for value in equation.params.values():
            candidates = value if isinstance(value, (tuple, list)) else (value,)
            total += sum(
                count_equations(child)
                for child in candidates
                if hasattr(child, "jaxpr") or hasattr(child, "eqns")
            )
    return total


def product(n):
    today = Date.ymd(2022, 9, 15)
    end = today.add_days(n)
    dates = ["BARRIER", "STRIKE", today, f"START: {today} END: {end} FREQ: 1CD", end]
    events = [
        "150",
        "120",
        "alive=1",
        "IF SPOT()>=BARRIER:0.1 THEN alive=0 END",
        "call PAYS alive*MAX(SPOT()-STRIKE,0)",
    ]
    return prepare(Product_New(dates, events), today)


def measure(prepared, threshold, *, compile_enabled=True):
    payoff = prepared.path_product().payoff
    ctx = EvalContext(fuzzy=True, scan_group_threshold=threshold)
    fn = jax.value_and_grad(lambda p, s: payoff({"script": p}, s, ctx))
    params = {name: jnp.asarray(value) for name, value in prepared.script_params}
    n = len(prepared.events)
    samples = Scenario(
        jnp.linspace(100.0, 149.99, n), jnp.ones(n), jnp.empty((n, 0)), jnp.empty((n, 0))
    )
    graph = jax.make_jaxpr(fn)(params, samples)
    start = time.perf_counter()
    lowered = jax.jit(fn).lower(params, samples)
    lower_seconds = time.perf_counter() - start
    hlo_chars = len(str(lowered.compiler_ir()))
    row = {
        "events": n,
        "threshold": threshold,
        "scan_spans": [
            [g.start, g.stop]
            for g in prepared.event_groups(fuzzy=True, threshold=threshold)
            if g.scanned
        ],
        "top_level_equations": len(graph.jaxpr.eqns),
        "total_equations": count_equations(graph),
        "stablehlo_chars": hlo_chars,
        "lower_seconds": lower_seconds,
        "compile_seconds": None,
        "warm_seconds": None,
        "pv": None,
        "gradients": None,
    }
    if compile_enabled:
        row.update(time_compiled(lowered, params, samples))
    return row


def time_compiled(lowered, params, samples):
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - start
    jax.block_until_ready(compiled(params, samples))
    start = time.perf_counter()
    result = jax.block_until_ready(compiled(params, samples))
    return {
        "compile_seconds": compile_seconds,
        "warm_seconds": time.perf_counter() - start,
        "pv": float(result[0]),
        "gradients": {name: float(value) for name, value in result[1].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, nargs="+", default=[36, 365, 750])
    parser.add_argument(
        "--unrolled-limit",
        type=int,
        default=64,
        help="compile unrolled baselines only up to this many days",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config.configure(num_cpu_devices=1)
    report = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "measurements": [],
    }
    for n in args.days:
        prepared = product(n)
        for threshold in (0, 4):
            row = measure(
                prepared, threshold, compile_enabled=threshold > 0 or n <= args.unrolled_limit
            )
            report["measurements"].append(row)
            print(json.dumps(row), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
