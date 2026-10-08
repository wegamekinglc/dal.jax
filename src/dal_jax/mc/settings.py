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
type DType = Literal["float64", "float32", "auto"]
type BlockSize = int | Literal["auto"]
type SmoothingKernel = Literal["dal", "smoothstep"]

DEFAULT_SMOOTH = 0.01
DEFAULT_BLOCK_SIZE = 8192
DEFAULT_PRNG_SEED = 1024

#  Allowed values per field; ``rsg`` is reported under DAL's field name.
_CHOICES = {
    "rsg": ("sobol", "mrg32", "irn"),
    "inverse_normal": ("acklam", "acklam_polish", "acklam_polish_precise", "ndtri"),
    "parallel": ("shard_map", "auto", "pmap", "none"),
    "dtype": ("float64", "float32", "auto"),
    "smoothing_kernel": ("dal", "smoothstep"),
    "lsmc_policy_risk_mode": ("Frozen", "RetrainedBump"),
}
_DAL_FIELD_NAMES = {"rsg": "simulation.rsg_"}


def _check_choices(settings: "MonteCarloSettings") -> None:
    for name, allowed in _CHOICES.items():
        value = getattr(settings, name)
        if value not in allowed:
            raise InvalidSetting(
                f"{_DAL_FIELD_NAMES.get(name, name)}={value}; expected one of {', '.join(allowed)}"
            )


def _check_integer(
    name: str, value: object, low: float = -math.inf, high: float = math.inf
) -> None:
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

    Execution: ``block_size="auto"`` keeps 8192 on CPU and estimates a GPU
    block from its allocator memory limit; explicit positive sizes override it.
    Paths run in these blocks; ``parallel`` spreads
    blocks over ``devices`` (default: every device of ``platform``);
    ``dtype="float32"`` casts path arrays while block sums accumulate in
    float64. ``dtype="auto"`` chooses float64 CPU / float32 GPU paths;
    the default remains float64. ``deterministic_reduction`` sums per-block values and gradients in
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
    block_size: BlockSize = "auto"
    parallel: Parallel = "shard_map"
    platform: config.Platform = "auto"
    devices: tuple[jax.Device, ...] | None = None
    dtype: DType = "float64"
    deterministic_reduction: bool = False
    block_bucketing: bool = False
    checkpoint: bool = True
    lsmc_basis_degree: int = 3
    lsmc_training_paths: int | None = None
    lsmc_validation_paths: int | None = None
    lsmc_rqmc_replicates: int | None = None
    lsmc_training_seed: int | None = None
    lsmc_pricing_seed: int | None = None
    lsmc_policy_risk_mode: str = "Frozen"
    lsmc_policy_bump_relative: float = 1e-3

    def __post_init__(self) -> None:
        _check_choices(self)
        if not (math.isfinite(self.smooth) and self.smooth > 0.0):
            raise InvalidSmoothing(
                f"simulation.smooth_={self.smooth}; expected a finite positive width"
            )
        if self.block_size != "auto":
            _check_integer("block_size", self.block_size, low=1)
            object.__setattr__(self, "block_size", int(self.block_size))
        _check_integer("seed", self.seed)
        _check_integer("scan_group_threshold", self.scan_group_threshold, low=0)
        object.__setattr__(self, "scan_group_threshold", int(self.scan_group_threshold))
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
        self._check_lsmc()

    def _check_lsmc(self) -> None:
        _check_integer("lsmc_basis_degree", self.lsmc_basis_degree, 1, 8)
        for name, low in (
            ("lsmc_training_paths", 1),
            ("lsmc_validation_paths", 0),
            ("lsmc_rqmc_replicates", 2),
            ("lsmc_training_seed", 0),
            ("lsmc_pricing_seed", 0),
        ):
            value = getattr(self, name)
            if value is not None:
                _check_integer(name, value, low, 2**31 - 1)
        if (
            not math.isfinite(self.lsmc_policy_bump_relative)
            or not 0 < self.lsmc_policy_bump_relative <= 0.1
        ):
            raise InvalidSetting("lsmc_policy_bump_relative must be finite and in (0, 0.1]")
        self._check_lsmc_modes()

    def _check_lsmc_modes(self) -> None:
        if self.lsmc_policy_risk_mode == "RetrainedBump" and not self.enable_aad:
            raise InvalidSetting("lsmc_policy_risk_mode=RetrainedBump requires enable_aad=True")
        if self.lsmc_rqmc_replicates is not None and self.rsg != "sobol":
            raise InvalidSetting("lsmc_rqmc_replicates requires rsg=sobol")
        if self.lsmc_rqmc_replicates is None and (
            self.lsmc_training_seed is not None or self.lsmc_pricing_seed is not None
        ):
            raise InvalidSetting("LSMC seeds require lsmc_rqmc_replicates")

    def resolved_devices(self) -> tuple[jax.Device, ...]:
        if self.devices is not None:
            return self.devices
        found = config.devices(self.platform)
        if not found:
            raise InvalidSetting(f"no {self.platform} devices available")
        return found
