"""Random-access Sobol sequence, point-for-point compatible with DAL's ``SobolSet_``.

DAL advances a Gray-code state: ``FillUniform`` first increments the path
counter, so global path ``n`` (0-based) is Sobol point ``n + 1`` and its state
is the XOR of ``dir[j]`` over the set bits ``j`` of ``gray(n + 1)``.  Computing
that state directly lets any device produce any path without ``SkipTo``.

Uniforms are ``state * 2**-32``; with a digital shift (DAL's
``NewDigitallyShiftedSobol``) they are ``((state ^ shift) + 0.5) * 2**-32``.
"""

import functools
from dataclasses import dataclass
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


@functools.cache
def _direction_table() -> np.ndarray:
    with resources.files("dal_jax.random").joinpath("directions.npy").open("rb") as stream:
        return np.load(stream)


@functools.cache
def directions(dim: int) -> np.ndarray:
    """``uint32`` array of shape ``(32, dim)``: ``directions[bit, coordinate]``."""
    if not 0 < dim < N_KNOWN:
        raise InvalidRandomSequence(
            f"Sobol dimension {dim} outside [1, {N_KNOWN - 1}]; "
            "not enough primitive polynomials available to generate Sobol sequences"
        )
    return np.ascontiguousarray(_direction_table()[:dim].T)


def _next_split_mix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return state, value ^ (value >> 31)


@functools.cache
def digital_shifts(dim: int, key: int) -> np.ndarray:
    """One XOR mask per coordinate: the high 32 bits of successive SplitMix64(key) outputs."""
    state = key & _MASK64
    shifts = np.empty(dim, dtype=np.uint32)
    for i in range(dim):
        state, value = _next_split_mix64(state)
        shifts[i] = value >> 32
    return shifts


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

    def __post_init__(self) -> None:
        directions(self.dim)  # validate eagerly

    def state(self, path_id: Array) -> Array:
        return sobol_state(path_id, jnp.asarray(directions(self.dim)))

    def uniform(self, path_id: Array) -> Array:
        state = self.state(path_id)
        if self.shift_key is None:
            return state.astype(jnp.float64) * MUL
        shifted = state ^ jnp.asarray(digital_shifts(self.dim, self.shift_key))
        return (shifted.astype(jnp.float64) + 0.5) * MUL

    def normal(self, path_id: Array) -> Array:
        return inverse_ncdf(self.uniform(path_id), self.precise, self.polish)
