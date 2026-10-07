"""Process-wide JAX configuration: x64, platform, PRNG, compilation cache, CPU devices.

``dal_jax`` enables ``jax_enable_x64`` on import.  DAL computes in double
precision and the parity tests compare to 1e-10, so float64 must be available
even when a float32 simulation is requested (the float32 mode only casts the
path arrays, see :class:`dal_jax.mc.MonteCarloSettings`).
"""

from typing import Literal

import jax

type Platform = Literal["cpu", "gpu", "auto"]


def enable_x64() -> None:
    jax.config.update("jax_enable_x64", True)


def configure(
    *,
    num_cpu_devices: int | None = None,
    prng_impl: str | None = None,
    compilation_cache_dir: str | None = None,
) -> None:
    """Apply process-wide settings.

    ``num_cpu_devices`` splits the host into that many virtual CPU devices so
    ``shard_map`` can spread blocks across cores.  JAX only accepts it before
    the first JAX operation runs, so call this right after importing ``dal_jax``.
    ``prng_impl`` sets the default key implementation (``"threefry2x32"``,
    ``"rbg"`` or ``"unsafe_rbg"``).  ``compilation_cache_dir`` enables the
    persistent compilation cache.
    """
    if num_cpu_devices is not None:
        if num_cpu_devices < 1:
            raise ValueError("num_cpu_devices must be positive")
        jax.config.update("jax_num_cpu_devices", num_cpu_devices)
    if prng_impl is not None:
        jax.config.update("jax_default_prng_impl", prng_impl)
    if compilation_cache_dir is not None:
        jax.config.update("jax_compilation_cache_dir", compilation_cache_dir)


def devices(platform: Platform = "auto") -> tuple[jax.Device, ...]:
    """Local devices for ``platform``; ``"auto"`` is JAX's default backend."""
    match platform:
        case "auto":
            return tuple(jax.local_devices())
        case "cpu" | "gpu":
            return tuple(jax.local_devices(backend=platform))
        case _:
            raise ValueError(f"unknown platform {platform!r}; expected cpu, gpu or auto")
