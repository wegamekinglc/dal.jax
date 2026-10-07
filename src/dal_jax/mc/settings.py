"""Monte Carlo settings: DAL's ``MonteCarloSettings_`` plus JAX execution fields."""

import math
import numbers
from dataclasses import dataclass
from typing import Literal

import jax

from dal_jax import config
from dal_jax.errors import InvalidSetting, InvalidSmoothing

type Rsg = Literal["sobol", "mrg32", "irn"]
type Parallel = Literal["shard_map", "auto", "pmap", "none"]
type InverseNormal = Literal["acklam", "acklam_polish", "acklam_polish_precise", "ndtri"]
type DType = Literal["float64", "float32"]
type SmoothingKernel = Literal["dal", "smoothstep"]

DEFAULT_SMOOTH = 0.01
DEFAULT_BLOCK_SIZE = 8192
DEFAULT_PRNG_SEED = 1024

#  Allowed values per field; ``rsg`` is reported under DAL's field name.
_CHOICES = {
    "rsg": ("sobol", "mrg32", "irn"),
    "inverse_normal": ("acklam", "acklam_polish", "acklam_polish_precise", "ndtri"),
    "parallel": ("shard_map", "auto", "pmap", "none"),
    "dtype": ("float64", "float32"),
    "smoothing_kernel": ("dal", "smoothstep"),
}
_DAL_FIELD_NAMES = {"rsg": "simulation.rsg_"}


def _check_choices(settings: "MonteCarloSettings") -> None:
    for name, allowed in _CHOICES.items():
        value = getattr(settings, name)
        if value not in allowed:
            raise InvalidSetting(f"{_DAL_FIELD_NAMES.get(name, name)}={value}; expected one of {', '.join(allowed)}")


def _check_integer(name: str, value: object, low: float = -math.inf, high: float = math.inf) -> None:
    """Integers only: ``True`` and ``8192.0`` would fail later as JAX shapes or seeds."""
    is_integer = isinstance(value, numbers.Integral) and not isinstance(value, bool)
    if not (is_integer and low <= value <= high):
        raise InvalidSetting(f"{name}={value!r}; expected an integer in [{low}, {high}]")


@dataclass(frozen=True, slots=True, kw_only=True)
class MonteCarloSettings:
    """Static configuration of a simulation; hashable, so it can key compilation caches.

    DAL fields: ``rsg``, ``use_bb``, ``enable_aad`` (fuzzy evaluation plus
    gradients), ``smooth`` (default smoothing width).
    Scripts use DAL's piecewise-linear kernels by default; ``smoothing_kernel``
    can select ``smoothstep`` for C1 transitions. ``scan_group_threshold``
    groups adjacent equal event templates (default 4; 0 disables scanning).

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
    smoothing_kernel: SmoothingKernel = "dal"
    scan_group_threshold: int = 4
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
        _check_choices(self)
        if not (math.isfinite(self.smooth) and self.smooth > 0.0):
            raise InvalidSmoothing(f"simulation.smooth_={self.smooth}; expected a finite positive width")
        _check_integer("block_size", self.block_size, low=1)
        _check_integer("seed", self.seed)
        _check_integer("scan_group_threshold", self.scan_group_threshold, low=0)
        object.__setattr__(self, "scan_group_threshold", int(self.scan_group_threshold))
        object.__setattr__(self, "block_size", int(self.block_size))
        object.__setattr__(self, "seed", int(self.seed))
        if self.sobol_shift_key is not None:
            _check_integer("sobol_shift_key", self.sobol_shift_key, low=0, high=2**64 - 1)
            object.__setattr__(self, "sobol_shift_key", int(self.sobol_shift_key))
            if self.rsg != "sobol":
                raise InvalidSetting("a Sobol digital shift requires simulation.rsg_=sobol")
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
