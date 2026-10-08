"""End-to-end scalar script benchmark: compile, synchronized warm runs and memory.

Use separate processes for different virtual CPU device counts. Examples:
    uv run python benchmarks/bench_suite.py --devices 8 --dal --output cpu.json
    python benchmarks/bench_suite.py --platform gpu --dtype float32 --output gpu.json
    python benchmarks/bench_suite.py --platform gpu --rsg mrg32 --prng-impl rbg

Default workloads are 2**20 paths, exact price and fuzzy price plus all Greeks.
GPU tests should install the cuda12/cuda13 extra in a separate environment.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax
from products import dal_products, model, prepared_products

from dal_jax import MonteCarloEngine, MonteCarloSettings, config


def block_size(text):
    if text == "auto":
        return text
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("block size must be positive or auto")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--platform", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument(
        "--parallel", choices=("none", "shard_map", "auto", "pmap"), default="shard_map"
    )
    parser.add_argument("--dtype", choices=("float64", "float32", "auto"), default="float64")
    parser.add_argument("--paths", type=int, default=2**20)
    parser.add_argument("--block-size", type=block_size, default="auto")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cases", default="european,barrier_1m,barrier_1w,asian,autocall")
    parser.add_argument("--mode", choices=("price", "greeks", "both"), default="both")
    parser.add_argument("--rsg", choices=("sobol", "mrg32", "irn"), default="sobol")
    parser.add_argument("--prng-impl", choices=("threefry2x32", "rbg", "unsafe_rbg"))
    parser.add_argument("--use-bb", action="store_true")
    parser.add_argument("--smooth", type=float, default=0.01)
    parser.add_argument("--smoothing-kernel", choices=("dal", "smoothstep"), default="dal")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--scan-threshold", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dal", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.dal and args.smoothing_kernel != "dal":
        parser.error("DAL comparison requires --smoothing-kernel dal")
    return args


def cpu_name():
    info = Path("/proc/cpuinfo")
    if info.exists():
        return next(
            (
                line.split(":", 1)[1].strip()
                for line in info.read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
    return platform.processor()


def configuration(args):
    return {
        "platform": args.platform,
        "devices": args.devices,
        "parallel": args.parallel,
        "dtype": args.dtype,
        "paths": args.paths,
        "block_size": args.block_size,
        "rsg": args.rsg,
        "prng_impl": args.prng_impl,
        "use_bb": args.use_bb,
        "smooth": args.smooth,
        "smoothing_kernel": args.smoothing_kernel,
        "checkpoint": not args.no_checkpoint,
        "scan_group_threshold": args.scan_threshold,
        "deterministic_reduction": args.deterministic,
        "repeat": args.repeat,
    }


def result_dict(output, greeks):
    if not greeks:
        return {"PV": float(output[0])}
    pv, risks = output
    result = {"PV": float(pv)}
    for group in risks.values():
        result.update({f"d_{name}": float(value) for name, value in group.items()})
    return result


def memory_analysis(compiled):
    stats = compiled.memory_analysis()
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    return {name: getattr(stats, name) for name in names} if stats is not None else None


def measure_jax(prepared, args, greeks):
    settings = MonteCarloSettings(
        enable_aad=greeks,
        platform=args.platform,
        devices=config.devices(args.platform)[: args.devices],
        parallel=args.parallel,
        dtype=args.dtype,
        block_size=args.block_size,
        rsg=args.rsg,
        prng_impl=args.prng_impl,
        use_bb=args.use_bb,
        checkpoint=not args.no_checkpoint,
        smooth=args.smooth,
        smoothing_kernel=args.smoothing_kernel,
        scan_group_threshold=args.scan_threshold,
        deterministic_reduction=args.deterministic,
    )
    engine = MonteCarloEngine(prepared.path_product(), model(), settings)
    params = engine.default_params()
    price = engine.pricer(args.paths)
    fn = jax.value_and_grad(lambda p: price(p)[0]) if greeks else price
    start = time.perf_counter()
    lowered = jax.jit(fn).lower(params)
    lower_seconds = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    output = jax.block_until_ready(compiled(params))
    first_seconds = time.perf_counter() - start
    durations = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        output = jax.block_until_ready(compiled(params))
        durations.append(time.perf_counter() - start)
    return {
        "backend": engine.devices[0].platform,
        "device_kind": engine.devices[0].device_kind,
        "n_devices": len(engine.devices),
        "dtype": str(engine.dtype),
        "block_size": engine.layout(args.paths).block_size,
        "sim_dim": engine.sim_dim,
        "scan_spans": [
            [group.start, group.stop]
            for group in prepared.event_groups(fuzzy=greeks, threshold=args.scan_threshold)
            if group.scanned
        ],
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "first_run_seconds": first_seconds,
        "warm_min_seconds": min(durations),
        "warm_median_seconds": statistics.median(durations),
        "warm_runs_seconds": durations,
        "compiled_memory": memory_analysis(compiled),
        "allocator_stats_after_run": engine.devices[0].memory_stats(),
        "result": result_dict(output, greeks),
    }


def measure_dal(product, bs, args, greeks, dal):
    run = lambda: dict(
        dal.MonteCarlo_Value(product, bs, args.paths, args.rsg, args.use_bb, greeks, args.smooth)
    )
    run()
    durations = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        result = run()
        durations.append(time.perf_counter() - start)
    return {
        "warm_min_seconds": min(durations),
        "warm_median_seconds": statistics.median(durations),
        "result": result,
    }


def main():
    args = parse_args()
    config.configure(num_cpu_devices=args.devices)
    products = prepared_products()
    modes = {"price": (False,), "greeks": (True,), "both": (False, True)}[args.mode]
    report = {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "os": platform.platform(),
            "cpu": cpu_name(),
        },
        "configuration": configuration(args),
        "measurements": [],
    }
    dal, reference, bs = None, {}, None
    if args.dal:
        import dal as oracle

        dal = oracle
        reference, bs = dal_products(dal)
    for name in args.cases.split(","):
        for greeks in modes:
            row = {
                "case": name,
                "mode": "greeks" if greeks else "price",
                "jax": measure_jax(products[name], args, greeks),
            }
            if dal is not None:
                row["dal"] = measure_dal(reference[name], bs, args, greeks, dal)
            report["measurements"].append(row)
            print(json.dumps(row), flush=True)
            if args.output:
                args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
