"""Monte Carlo engine benchmarks: compile time and warm run time.

Cases (2**20 Sobol paths by default): European call, and an up-and-out call
observed monthly (36 dates) or weekly (156 dates), each priced alone and with
all Greeks (fuzzy mode + ``value_and_grad``).  ``barrier_*`` unroll the
barrier test date by date like DAL's event loop (the shape the script layer's
scan grouping will target); ``barrier_*_vec`` evaluate the same survival
product over all dates at once.  ``--dal`` times dal-python on the same
machine for comparison.

    python benchmarks/bench_mc.py --devices 8
    python benchmarks/bench_mc.py --platform gpu --dtype float32
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", type=int, default=1, help="virtual CPU devices (jax_num_cpu_devices)")
    parser.add_argument("--platform", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--parallel", choices=("shard_map", "auto", "pmap", "none"), default="shard_map")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--paths", type=int, default=2**20)
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cases", default="european,barrier_1m,barrier_1w,barrier_1w_vec")
    parser.add_argument("--dal", action="store_true", help="also time dal-python")
    return parser.parse_args()


def weekly_timeline(years: int = 3) -> tuple[float, ...]:
    end = 365 * years
    return (0.0,) + tuple(d / 365.0 for d in range(7, end, 7)) + (end / 365.0,)


def timed(fn, repeat: int) -> tuple[float, float]:
    start = time.perf_counter()
    fn()
    first = time.perf_counter() - start
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return first, best


def dal_cases(dal):
    from support import BARRIER, DIV, MATURITY, RATE, SPOT, STRIKE, VOL

    today = dal.Date_(2022, 9, 15)
    dal.EvaluationDate_Set(today)
    maturity = today.AddDays(int(365 * MATURITY))
    model = dal.BSModelData_New(SPOT, VOL, RATE, DIV)
    european = dal.Product_New(["STRIKE", maturity], [f"{STRIKE}", "call pays MAX(spot() - STRIKE, 0.0)"])

    def barrier(freq):
        dates = ["STRIKE", "BARRIER", today, f"START: {today} END: {maturity} FREQ: {freq}", maturity]
        events = [f"{STRIKE:.2f}", f"{BARRIER:.2f}", "alive = 1", "if spot() >= BARRIER:0.1 then alive = 0 end",
                  "if spot() >= BARRIER:0.1 then alive = 0 end\ncall pays alive * MAX(spot() - STRIKE, 0.0)"]
        return dal.Product_New(dates, events)

    monthly, weekly = barrier("1M"), barrier("1W")
    return {"european": european, "barrier_1m": monthly, "barrier_1w": weekly, "barrier_1m_vec": monthly, "barrier_1w_vec": weekly}, model


def jax_products():
    from support import european_call, monthly_barrier_timeline, up_and_out_call

    def barrier(timeline, vectorized):
        last = len(timeline) - 1
        return up_and_out_call(timeline, tuple(range(1, last + 1)) + (last,), vectorized=vectorized)

    return {
        "european": european_call(),
        "barrier_1m": barrier(monthly_barrier_timeline(), False),
        "barrier_1w": barrier(weekly_timeline(), False),
        "barrier_1m_vec": barrier(monthly_barrier_timeline(), True),
        "barrier_1w_vec": barrier(weekly_timeline(), True),
    }


def bench_jax(args, cases) -> None:
    import jax
    from support import bs_model

    import dal_jax
    from dal_jax import MonteCarloEngine, MonteCarloSettings

    products = jax_products()
    devices = dal_jax.config.devices(args.platform)
    print(f"jax {jax.__version__}, platform={args.platform}, devices={len(devices)} ({devices[0].device_kind}), "
          f"parallel={args.parallel}, dtype={args.dtype}, paths={args.paths}, block={args.block_size}")
    print()
    print("| case | mode | compile + first run (s) | warm (ms) | PV |")
    print("|---|---|---:|---:|---:|")
    for case in cases:
        for greeks in (False, True):
            settings = MonteCarloSettings(enable_aad=greeks, platform=args.platform, parallel=args.parallel,
                                          dtype=args.dtype, block_size=args.block_size)
            engine = MonteCarloEngine(products[case], bs_model(), settings)
            result = {}
            first, best = timed(lambda engine=engine, result=result: result.update(engine.value(args.paths)), args.repeat)
            print(f"| {case} | {'price + grad' if greeks else 'price'} | {first:.2f} | {best * 1e3:.1f} | {result['PV']:.10f} |")


def bench_dal(args, cases) -> None:
    import dal

    products, model = dal_cases(dal)
    print()
    print("| case | mode | DAL warm (ms) | DAL PV |")
    print("|---|---|---:|---:|")
    for case in cases:
        for greeks in (False, True):
            result = {}

            def run(product=products[case], greeks=greeks, result=result):
                result.update(dict(dal.MonteCarlo_Value(product, model, args.paths, "sobol", False, greeks)))

            _, best = timed(run, args.repeat)
            print(f"| {case} | {'price + grad' if greeks else 'price'} | {best * 1e3:.1f} | {result['PV']:.10f} |")


def main() -> int:
    args = parse_args()
    import dal_jax

    if args.platform == "cpu":
        dal_jax.config.configure(num_cpu_devices=args.devices)
    cases = [c for c in args.cases.split(",") if c]
    bench_jax(args, cases)
    if args.dal:
        bench_dal(args, cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
