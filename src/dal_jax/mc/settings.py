"""Monte Carlo settings: DAL's ``MonteCarloSettings_`` plus JAX execution fields."""

import math
from dataclasses import dataclass
from typing import Literal

import jax

from dal_jax import config
from dal_jax.errors import InvalidSetting, InvalidSmoothing

type Rsg = Literal["sobol", "mrg32", "irn"]
type Parallel = Literal["shard_map", "auto", "pmap", "none"]
type InverseNormal = Literal["acklam", "acklam_polish", "acklam_polish_precise", "ndtri"]
type DType = Literal["float64", "float32"]

DEFAULT_SMOOTH = 0.01
DEFAULT_BLOCK_SIZE = 8192
DEFAULT_PRNG_SEED = 1024

_RSGS = ("sobol", "mrg32", "irn")
_PARALLEL = ("shard_map", "auto", "pmap", "none")
_INVERSE_NORMALS = ("acklam", "acklam_polish", "acklam_polish_precise", "ndtri")
_DTYPES = ("float64", "float32")


@dataclass(frozen=True, slots=True, kw_only=True)
class MonteCarloSettings:
    """Static configuration of a simulation; hashable, so it can key compilation caches.

    DAL fields: ``rsg``, ``use_bb``, ``enable_aad`` (fuzzy evaluation plus
    gradients), ``smooth`` (default smoothing width).

    Random numbers: ``inverse_normal`` selects DAL's ``InverseNCDF`` variant
    (``acklam`` is what DAL's Sobol uses; ``ndtri`` is more accurate but not
    path-compatible with DAL); ``sobol_shift_key`` applies DAL's digital shift;
    ``seed`` / ``prng_impl`` drive the ``mrg32`` / ``irn`` streams.

    Execution: paths run in blocks of ``block_size``; ``parallel`` spreads
    blocks over ``devices`` (default: every device of ``platform``);
    ``dtype="float32"`` casts path arrays while block sums accumulate in
    float64; ``deterministic_reduction`` sums per-block values and gradients in
    block order so results are bitwise independent of the device count;
    ``block_bucketing`` rounds the block count up to a power of two so nearby
    path counts reuse one compilation; ``checkpoint`` rematerialises each block
    in the backward pass so gradient memory is one block, not all paths.
    """

    rsg: Rsg = "sobol"
    use_bb: bool = False
    enable_aad: bool = False
    smooth: float = DEFAULT_SMOOTH
    inverse_normal: InverseNormal = "acklam"
    sobol_shift_key: int | None = None
    seed: int = DEFAULT_PRNG_SEED
    prng_impl: str | None = None
    block_size: int = DEFAULT_BLOCK_SIZE
    parallel: Parallel = "shard_map"
    platform: config.Platform = "auto"
    devices: tuple[jax.Device, ...] | None = None
    dtype: DType = "float64"
    deterministic_reduction: bool = False
    block_bucketing: bool = False
    checkpoint: bool = True

    def __post_init__(self) -> None:
        if self.rsg not in _RSGS:
            raise InvalidSetting(f"simulation.rsg_={self.rsg}; expected sobol, mrg32 or irn; rng method is not known")
        if not (math.isfinite(self.smooth) and self.smooth > 0.0):
            raise InvalidSmoothing(f"simulation.smooth_={self.smooth}; expected a finite positive width")
        if self.inverse_normal not in _INVERSE_NORMALS:
            raise InvalidSetting(f"inverse_normal={self.inverse_normal}; expected one of {', '.join(_INVERSE_NORMALS)}")
        if self.sobol_shift_key is not None and self.rsg != "sobol":
            raise InvalidSetting("a Sobol digital shift requires simulation.rsg_=sobol")
        if self.block_size <= 0:
            raise InvalidSetting(f"block_size={self.block_size}; expected a positive integer")
        if self.parallel not in _PARALLEL:
            raise InvalidSetting(f"parallel={self.parallel}; expected one of {', '.join(_PARALLEL)}")
        if self.dtype not in _DTYPES:
            raise InvalidSetting(f"dtype={self.dtype}; expected float64 or float32")
        if self.devices is not None:
            object.__setattr__(self, "devices", tuple(self.devices))
            if not self.devices:
                raise InvalidSetting("devices must not be empty")

    def resolved_devices(self) -> tuple[jax.Device, ...]:
        if self.devices is not None:
            return self.devices
        found = config.devices(self.platform)
        if not found:
            raise InvalidSetting(f"no {self.platform} devices available")
        return found
