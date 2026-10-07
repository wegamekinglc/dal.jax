"""Console examples: synchronized timing, mandatory DAL comparison and JSON output."""

import argparse
import json
import os
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import dal
import jax
import numpy as np

import dal_jax as dj
from dal_jax.api import Product_New
from dal_jax.dates import Date

TODAY = Date.ymd(2022, 9, 15)
MATURITY = TODAY.add_days(1095)
BS = {"spot": 100., "vol": .15, "rate": .05, "div": .03}


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--paths", type=int, default=2**16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--devices", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.paths, args.repeat, args.devices) < 1:
        parser.error("paths, repeat and devices must be positive")
    dj.config.configure(num_cpu_devices=args.devices)
    dal.EvaluationDate_Set(dal.Date_(TODAY.year, TODAY.month, TODAY.day))
    print(f"JAX {jax.__version__}; {args.platform}; {args.paths:,} paths; DAL {version('dal-python')}")
    return args


def settings(args, **overrides):
    return dj.MonteCarloSettings(**({"platform": args.platform, "enable_aad": True} | overrides))


def model(**overrides):
    return dj.BlackScholes(**(BS | overrides))


def oracle_model(**overrides):
    values = BS | overrides
    return dal.BSModelData_New(*(values[name] for name in ("spot", "vol", "rate", "div")))


def oracle_product(rows):
    dates, events = rows
    dates = [dal.Date_(d.year, d.month, d.day) if isinstance(d, Date) else d for d in dates]
    return dal.Product_New(dates, events)


def prepare(rows):
    return dj.prepare(Product_New(*rows), TODAY)


def european_rows():
    return ["STRIKE", MATURITY], ["120", "call PAYS MAX(SPOT()-STRIKE,0)"]


def barrier_rows(width=.1):
    return ["STRIKE", "BARRIER", TODAY, f"START: {TODAY} END: {MATURITY} FREQ: 1M", MATURITY], [
        "120", "150", "alive=1", f"IF SPOT()>=BARRIER:{width} THEN alive=0 END",
        f"IF SPOT()>=BARRIER:{width} THEN alive=0 END call PAYS alive*MAX(SPOT()-STRIKE,0)",
    ]


def table(headers, rows):
    def cell(value):
        return f"{value:.10g}" if isinstance(value, (float, np.floating)) else str(value)
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(cell(value) for value in row) + " |")


def timed(fn, repeat):
    start = time.perf_counter()
    output = jax.block_until_ready(fn())
    first = time.perf_counter() - start
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        output = jax.block_until_ready(fn())
        durations.append(time.perf_counter() - start)
    return output, {"first_seconds": first, "warm_min_seconds": min(durations),
                    "warm_median_seconds": statistics.median(durations), "warm_runs_seconds": durations}


def scalar_results(output, greeks):
    if not greeks:
        return {"PV": float(output)}
    pv, risks = output
    values = {"PV": float(pv)}
    for group in risks.values():
        values.update({f"d_{name}": float(value) for name, value in group.items()})
    return values


def measure_jax(engine, args, payoff_index=0):
    params = engine.default_params()
    price = engine.pricer(args.paths)
    greeks = engine.settings.enable_aad
    scalar = lambda p: price(p)[payoff_index]
    fn = jax.value_and_grad(scalar) if greeks else scalar
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(params).compile()
    compilation = time.perf_counter() - start
    output, timing = timed(lambda: compiled(params), args.repeat)
    return timing | {"compile_seconds": compilation, "result": scalar_results(output, greeks),
                     "backend": engine.devices[0].platform, "devices": len(engine.devices),
                     "dtype": str(engine.dtype), "block_size": engine.layout(args.paths).block_size}


def measure_dal(product, bs, args, mc):
    run = lambda: dict(dal.MonteCarlo_Value(product, bs, args.paths, mc.rsg, mc.use_bb, mc.enable_aad, mc.smooth))
    output, timing = timed(run, args.repeat)
    return timing | {"result": output}


def check_results(ours, theirs, dtype="float64"):
    assert ours.keys() == theirs.keys()  # nosec B101: executable numerical validation
    tolerances = {"float64": (1e-10, 1e-8, 1e-10), "float32": (2e-5, 5e-3, 2e-4)}
    pv_rtol, risk_rtol, atol = tolerances[dtype]
    for name in ours:
        np.testing.assert_allclose(ours[name], theirs[name], rtol=pv_rtol if name == "PV" else risk_rtol,
                                   atol=atol, err_msg=name)


def compare(label, engine, rows, args, *, bs=None, check=True, payoff_index=0):
    ours = measure_jax(engine, args, payoff_index)
    theirs = measure_dal(oracle_product(rows), oracle_model() if bs is None else bs, args, engine.settings)
    if check:
        check_results(ours["result"], theirs["result"], ours["dtype"])
    print(f"\n{label}")
    table(["quantity", "JAX", "DAL", "absolute difference"],
          [[name, value, theirs["result"][name], abs(value-theirs["result"][name])] for name, value in ours["result"].items()])
    table(["backend", "compile (ms)", "first (ms)", "warm min (ms)", "warm median (ms)"], [
        [f"JAX {ours['backend']} {ours['devices']} devices / {ours['dtype']}", ours["compile_seconds"]*1000,
         ours["first_seconds"]*1000, ours["warm_min_seconds"]*1000, ours["warm_median_seconds"]*1000],
        ["DAL CPU", "precompiled C++", theirs["first_seconds"]*1000, theirs["warm_min_seconds"]*1000,
         theirs["warm_median_seconds"]*1000],
    ])
    return {"label": label, "paths": args.paths, "jax": ours, "dal": theirs}


def finish(args, comparisons, **diagnostics):
    report = {"environment": {"jax": jax.__version__, "dal_python": version("dal-python")},
              "configuration": {"paths": args.paths, "repeat": args.repeat, "platform": args.platform,
                                "devices": args.devices}, "comparisons": comparisons, "diagnostics": diagnostics}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nSaved {args.output}")
