"""Run on CPU or --platform gpu: resolved auto blocks, float64/32 and DAL prices/Greeks."""

from _common import arguments, barrier_rows, compare, european_rows, finish, model, prepare, settings, table


def main():
    args = arguments(__doc__)
    comparisons = []
    layouts = []
    for name, rows in (("European", european_rows()), ("Monthly barrier", barrier_rows())):
        product = prepare(rows)
        for dtype in ("float64", "float32", "auto"):
            engine = product.engine(model(), settings(args, dtype=dtype))
            comparisons.append(compare(f"{name}: dtype={dtype}, block_size=auto", engine, rows, args))
            memory = engine.devices[0].memory_stats() or {}
            layouts.append([name, dtype, str(engine.dtype), engine.block_size, memory.get("bytes_limit", "CPU / unknown")])
    table(["product", "requested dtype", "resolved dtype", "auto block upper limit", "allocator bytes limit"], layouts)
    print("float64 stays the default. Narrow autocall Greeks can have large float32 errors even when PV agrees.")
    print("The report in docs/performance.md records that stress case; widening smoothing changes the valuation.")
    finish(args, comparisons, layouts=layouts)


if __name__ == "__main__":
    main()
