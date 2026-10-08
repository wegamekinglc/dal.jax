"""Parallel strategies, bitwise deterministic reduction, float32 and bucketed tail paths."""

from copy import copy

import jax
from _common import arguments, compare, european_rows, finish, model, prepare, settings, table

import dal_jax as dj


def main():
    args = arguments(__doc__)
    rows = european_rows()
    product = prepare(rows)
    comparisons = []
    for strategy in ("none", "shard_map", "auto", "pmap"):
        engine = product.engine(model(), settings(args, parallel=strategy))
        comparisons.append(compare(strategy, engine, rows, args))
    deterministic = []
    devices = dj.config.devices(args.platform)
    for selected in (devices[:1], devices):
        engine = product.engine(
            model(), settings(args, devices=selected, block_size=1024, deterministic_reduction=True)
        )
        record = compare(f"Deterministic, {len(selected)} devices", engine, rows, args)
        deterministic.append(record["jax"]["result"])
        comparisons.append(record)
    assert deterministic[0] == deterministic[1]  # nosec B101: executable numerical validation
    engine = product.engine(model(), settings(args, dtype="float32"))
    comparisons.append(compare("float32 paths, float64 block accumulation", engine, rows, args))
    bucketed = product.engine(model(), settings(args, block_size=1024, block_bucketing=True))
    for n in (args.paths + 1, args.paths + 2):
        local = copy(args)
        local.paths = n
        comparisons.append(compare(f"Bucketed tail, {n} paths", bucketed, rows, local))
    table(
        ["deterministic PV hex", "delta hex", "bitwise equal"],
        [
            [
                deterministic[0]["PV"].hex(),
                deterministic[0]["d_spot"].hex(),
                deterministic[0] == deterministic[1],
            ]
        ],
    )
    print("Selected devices:", jax.tree.map(str, list(devices)))
    finish(args, comparisons, deterministic_bitwise_equal=True)


if __name__ == "__main__":
    main()
