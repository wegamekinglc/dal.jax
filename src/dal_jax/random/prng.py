"""Pseudo-random normals for DAL's ``mrg32`` / ``irn`` generators.

Both names map to ``jax.random``; streams are not bit-compatible with DAL and
are compared statistically only.  Each block draws from
``fold_in(key, block_id)``, so with a fixed block size the numbers a path sees
do not depend on how blocks are spread over devices.
"""

from functools import cache

import jax
import jax.numpy as jnp
from jax import Array
from jax.custom_batching import sequential_vmap

from dal_jax.errors import InvalidSetting

PRNG_NAMES = ("mrg32", "irn")
_STREAM_SALT = {"mrg32": 0, "irn": 1}


def prng_key(rsg: str, seed: int, impl: str | None = None) -> Array:
    if rsg not in _STREAM_SALT:
        raise InvalidSetting(
            f"simulation.rsg_={rsg}; expected mrg32 or irn for a pseudo-random stream"
        )
    return jax.random.fold_in(jax.random.key(seed, impl=impl), _STREAM_SALT[rsg])


@cache
def _sequential_blocks(block_size: int, dim: int):
    # RBG's native batching uses only the first key. Sequential key mapping
    # preserves each global block's stream under GSPMD and business vmaps.
    return sequential_vmap(
        lambda key, block_id: jax.random.normal(
            jax.random.fold_in(key, block_id), (block_size, dim), dtype=jnp.float64
        )
    )


def block_normals(key: Array, block_id: Array, block_size: int, dim: int) -> Array:
    """``float64[block_size, dim]`` standard normals for one block."""
    if str(jax.random.key_impl(key)) in ("rbg", "unsafe_rbg"):
        return _sequential_blocks(block_size, dim)(key, block_id)
    return jax.random.normal(
        jax.random.fold_in(key, block_id), (block_size, dim), dtype=jnp.float64
    )
