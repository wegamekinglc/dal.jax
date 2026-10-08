"""Process-wide JAX configuration: x64, platform, PRNG, compilation cache, CPU devices."""

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
    """Configure JAX before its first operation."""
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
