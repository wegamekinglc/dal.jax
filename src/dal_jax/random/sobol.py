"""Random-access Sobol sequence, point-for-point compatible with DAL's ``SobolSet_``."""

from dataclasses import dataclass, field
from importlib import resources

import jax.numpy as jnp
import numpy as np
from jax import Array

from dal_jax.errors import InvalidRandomSequence
from dal_jax.random.inverse_normal import inverse_ncdf

N_BITS = 32
N_KNOWN = 21201
MUL = 2.3283064365386963e-10  # 2**-32
MAX_POINTS = 2**N_BITS - 1  # largest Sobol point index a 32-bit state reaches

_MASK64 = (1 << 64) - 1


def _read_only(array: np.ndarray) -> np.ndarray:
    """Publish an owned table without writable views."""
    array.setflags(write=False)
    return array


def _direction_table() -> np.ndarray:
    with resources.files("dal_jax.random").joinpath("directions.npy").open("rb") as stream:
        return _read_only(np.load(stream))


def directions(dim: int) -> np.ndarray:
    """``uint32`` array of shape ``(32, dim)``: ``directions[bit, coordinate]``."""
    if not 0 < dim < N_KNOWN:
        raise InvalidRandomSequence(
            f"Sobol dimension {dim} outside [1, {N_KNOWN - 1}]; "
            "not enough primitive polynomials available to generate Sobol sequences"
        )
    return _read_only(np.ascontiguousarray(_direction_table()[:dim].T))


def digital_shifts(dim: int, key: int) -> np.ndarray:
    """One XOR mask per coordinate: the high 32 bits of successive SplitMix64(key) outputs."""
    states = np.uint64(key & _MASK64) + np.arange(1, dim + 1, dtype=np.uint64) * np.uint64(
        0x9E3779B97F4A7C15
    )
    values = (states ^ (states >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return _read_only(((values ^ (values >> np.uint64(31))) >> np.uint64(32)).astype(np.uint32))


def sobol_state(path_id: Array, dirs: Array) -> Array:
    """``uint32[dim]`` Gray-code state DAL holds after generating global path ``path_id``."""
    n = (jnp.asarray(path_id) + 1).astype(jnp.uint32)
    gray = n ^ (n >> 1)
    state = jnp.zeros(dirs.shape[1], dtype=jnp.uint32)
    for bit in range(N_BITS):
        set_bit = ((gray >> bit) & 1).astype(jnp.bool_)
        state = state ^ jnp.where(set_bit, dirs[bit], jnp.uint32(0))
    return state


@dataclass(frozen=True, slots=True, kw_only=True)
class Sobol:
    """Single-path Sobol generator; ``vmap`` over ``path_id`` for a batch.

    ``precise``/``polish`` are DAL's ``InverseNCDF`` flags (both off in DAL's
    Monte Carlo).  ``shift_key`` enables the digital shift.
    """

    dim: int
    shift_key: int | None = None
    precise: bool = False
    polish: bool = False
    _directions: np.ndarray = field(init=False, repr=False, compare=False)
    _shift: np.ndarray | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_directions", directions(self.dim))
        object.__setattr__(
            self,
            "_shift",
            None if self.shift_key is None else digital_shifts(self.dim, self.shift_key),
        )

    def state(self, path_id: Array) -> Array:
        return sobol_state(path_id, jnp.asarray(self._directions))

    def uniform(self, path_id: Array) -> Array:
        state = self.state(path_id)
        if self.shift_key is None:
            return state.astype(jnp.float64) * MUL
        shifted = state ^ jnp.asarray(self._shift)
        return (shifted.astype(jnp.float64) + 0.5) * MUL

    def normal(self, path_id: Array) -> Array:
        return inverse_ncdf(self.uniform(path_id), self.precise, self.polish)
