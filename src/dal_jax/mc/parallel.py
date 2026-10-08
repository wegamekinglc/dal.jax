"""Spreading Monte Carlo blocks over devices."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

AXIS = "paths"

#  local(params, block_ids, n_paths, axis_name) -> pytree.  ``axis_name`` is the
#  manual mesh axis when running under ``shard_map`` (None otherwise), so the
#  callee can mark loop carries as device-varying.
type LocalFn = Callable[..., object]


def make_mesh(devices: Sequence[jax.Device]) -> jax.sharding.Mesh:
    return jax.make_mesh(
        (len(devices),), (AXIS,), axis_types=(AxisType.Auto,), devices=tuple(devices)
    )


def _varying(tree: object) -> object:
    """Mark replicated parameters varying once at the shard_map boundary."""
    # Recasting inside the block scan adds an all-reduce per block in reverse mode.
    return jax.tree.map(lambda x: jax.lax.pcast(x, (AXIS,), to="varying"), tree)


def _device_rows(n_blocks: int, n_devices: int, mesh: jax.sharding.Mesh | None) -> Array:
    rows = jnp.arange(n_blocks, dtype=jnp.int64).reshape(n_devices, n_blocks // n_devices)
    if mesh is not None:
        rows = jax.lax.with_sharding_constraint(rows, NamedSharding(mesh, P(AXIS, None)))
    return rows


def sum_blocks(
    local: LocalFn,
    params: object,
    n_paths: Array,
    *,
    strategy: str,
    devices: Sequence[jax.Device],
    n_blocks: int,
) -> Array:
    """Sum of ``local`` over all blocks; ``local`` returns its partial sum."""
    n_devices = len(devices)
    match strategy:
        case "none":
            return local(params, jnp.arange(n_blocks, dtype=jnp.int64), n_paths, None)
        case "shard_map":
            per_device = n_blocks // n_devices

            def run(p, n):
                ids = jax.lax.axis_index(AXIS) * per_device + jnp.arange(
                    per_device, dtype=jnp.int64
                )
                return jax.lax.psum(local(_varying(p), ids, n, AXIS), AXIS)

            return jax.shard_map(run, mesh=make_mesh(devices), in_specs=(P(), P()), out_specs=P())(
                params, n_paths
            )
        case "auto":
            rows = _device_rows(n_blocks, n_devices, make_mesh(devices))
            return jax.vmap(lambda ids: local(params, ids, n_paths, None))(rows).sum(axis=0)
        case "pmap":
            rows = _device_rows(n_blocks, n_devices, None)
            run = jax.pmap(
                lambda p, ids, n: local(p, ids, n, None),
                in_axes=(None, 0, None),
                devices=list(devices),
            )
            return run(params, rows, n_paths).sum(axis=0)
        case _:
            raise ValueError(f"unknown parallel strategy {strategy!r}")


def gather_blocks(
    local: LocalFn,
    params: object,
    n_paths: Array,
    *,
    strategy: str,
    devices: Sequence[jax.Device],
    n_blocks: int,
) -> object:
    """Per-block results stacked in global block order; ``local`` returns one row per block."""
    n_devices = len(devices)

    def flatten_rows(tree):
        return jax.tree.map(lambda x: x.reshape((n_blocks,) + x.shape[2:]), tree)

    match strategy:
        case "none":
            return local(params, jnp.arange(n_blocks, dtype=jnp.int64), n_paths, None)
        case "shard_map":
            per_device = n_blocks // n_devices

            def run(p, n):
                ids = jax.lax.axis_index(AXIS) * per_device + jnp.arange(
                    per_device, dtype=jnp.int64
                )
                return local(_varying(p), ids, n, AXIS)

            return jax.shard_map(
                run, mesh=make_mesh(devices), in_specs=(P(), P()), out_specs=P(AXIS)
            )(params, n_paths)
        case "auto":
            rows = _device_rows(n_blocks, n_devices, make_mesh(devices))
            return flatten_rows(jax.vmap(lambda ids: local(params, ids, n_paths, None))(rows))
        case "pmap":
            rows = _device_rows(n_blocks, n_devices, None)
            run = jax.pmap(
                lambda p, ids, n: local(p, ids, n, None),
                in_axes=(None, 0, None),
                devices=list(devices),
            )
            return flatten_rows(run(params, rows, n_paths))
        case _:
            raise ValueError(f"unknown parallel strategy {strategy!r}")


def ordered_sum(x: Array) -> Array:
    """Sum over the leading axis strictly in index order (a sequential ``scan``)."""
    total, _ = jax.lax.scan(lambda acc, row: (acc + row, None), jnp.zeros_like(x[0]), x)
    return total
