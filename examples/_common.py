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
from dal_jax.api import Product_New, EvaluationDate_Set
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
    EvaluationDate_Set(TODAY)
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


def require_p5_oracle():
    if not hasattr(dal, "CorrelatedBSModelData_New"):
        raise RuntimeError("This example needs the pinned DAL source oracle. Run scripts/build_dal_oracle.sh as documented in examples/README.md.")


def prepare(rows):
    return dj.prepare(Product_New(*rows), TODAY)


def european_rows():
    return ["STRIKE", MATURITY], ["120", "call PAYS MAX(SPOT()-STRIKE,0)"]


def barrier_rows(width=.1):
    return ["STRIKE", "BARRIER", TODAY, f"START: {TODAY} END: {MATURITY} FREQ: 1M", MATURITY], [
        "120", "150", "alive=1", f"IF SPOT()>=BARRIER:{width} THEN alive=0 END",
        f"IF SPOT()>=BARRIER:{width} THEN alive=0 END call PAYS alive*MAX(SPOT()-STRIKE,0)",
    ]


def _table_cell(value):
    return f"{value:.10g}" if isinstance(value, (float, np.floating)) else str(value)


def _table_alignment(column):
    numeric = (int, float, np.integer, np.floating)
    return ">" if any(isinstance(value, numeric) and not isinstance(value, (bool, np.bool_))
                      for value in column) else "<"


def _table_row(row, widths, alignments):
    return "  ".join(f"{value:{alignment}{width}}" for value, width, alignment
                     in zip(row, widths, alignments, strict=True))


def table(headers, rows):
    """Print a DAL-style table with aligned columns and full-width rules."""
    if not headers:
        return
    columns = list(zip(headers, *rows, strict=True))
    alignments = [_table_alignment(column[1:]) for column in columns]
    cells = [[_table_cell(value) for value in column] for column in columns]
    widths = [max(map(len, column)) for column in cells]
    formatted_rows = iter(zip(*cells, strict=True))
    rule = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print(_table_row(next(formatted_rows), widths, alignments))
    print(rule)
    for row in formatted_rows:
        print(_table_row(row, widths, alignments))
    print(rule)


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
    start = time.perf_counter()
    price = engine.pricer(args.paths)
    training = time.perf_counter()-start if isinstance(engine,dj.LsmcEngine) else None
    greeks = engine.settings.enable_aad
    scalar = lambda p: price(p)[payoff_index]
    fn = jax.value_and_grad(scalar) if greeks else scalar
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(params).compile()
    compilation = time.perf_counter() - start
    output, timing = timed(lambda: compiled(params), args.repeat)
    return timing | {"compile_seconds": compilation, "training_first_seconds":training, "result": scalar_results(output, greeks),
                     "backend": engine.devices[0].platform, "devices": len(engine.devices),
                     "dtype": str(engine.dtype), "block_size": engine.layout(args.paths).block_size}


def measure_dal(product, bs, args, mc, valuation=None):
    fields = ("lsmc_basis_degree","lsmc_training_paths","lsmc_validation_paths","lsmc_rqmc_replicates",
              "lsmc_training_seed","lsmc_pricing_seed","lsmc_policy_risk_mode","lsmc_policy_bump_relative")
    exercise = mc.lsmc_training_paths is not None or mc.lsmc_rqmc_replicates is not None
    if valuation is None and not exercise:
        run = lambda: dict(dal.MonteCarlo_Value(product, bs, args.paths, mc.rsg, mc.use_bb, mc.enable_aad, mc.smooth))
    else:
        extra = {name:getattr(mc,name) for name in fields if getattr(mc,name) is not None} if exercise else {}
        simulation = dal.MonteCarloSettings_(method=mc.rsg, use_bb=mc.use_bb, enable_aad=mc.enable_aad, smooth=mc.smooth,**extra)
        run = lambda: dict(dal.MonteCarlo_ValueWithSettings(product, bs, args.paths, valuation=valuation, simulation=simulation))
    output, timing = timed(run, args.repeat)
    return timing | {"result": output}


def check_results(ours, theirs, dtype="float64", *, exercise=False):
    assert ours.keys() == theirs.keys()  # nosec B101: executable numerical validation
    tolerances = {"float64": (1e-10, 1e-8, 1e-10), "float32": (2e-5, 5e-3, 2e-4)}
    pv_rtol, risk_rtol, atol = tolerances[dtype]
    if exercise and dtype == "float64":
        pv_rtol = 1e-6
    for name in ours:
        np.testing.assert_allclose(ours[name], theirs[name], rtol=pv_rtol if name == "PV" else risk_rtol,
                                   atol=atol, err_msg=name)


def compare(label, engine, rows, args, *, bs=None, check=True, payoff_index=0, valuation=None, product=None):
    ours = measure_jax(engine, args, payoff_index)
    theirs = measure_dal(oracle_product(rows) if product is None else product, oracle_model() if bs is None else bs,
                         args, engine.settings, valuation)
    if check:
        check_results(ours["result"], theirs["result"], ours["dtype"],exercise=isinstance(engine,dj.LsmcEngine))
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
