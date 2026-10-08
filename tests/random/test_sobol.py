import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dal_jax.errors import InvalidRandomSequence
from dal_jax.random.sobol import MUL, N_KNOWN, Sobol, digital_shifts, directions


def sequential_states(dim: int, n_points: int, start: int = 0) -> np.ndarray:
    """DAL's ``Seek`` + repeated ``FillUniform``, transcribed with Python ints."""
    dirs = directions(dim).astype(np.uint64)
    state = np.zeros(dim, dtype=np.uint64)
    ip, jj = start, 0
    while ip:
        if (ip ^ (ip >> 1)) & 1:
            state ^= dirs[jj]
        ip >>= 1
        jj += 1
    out = np.empty((n_points, dim), dtype=np.uint64)
    i_path = start
    for row in range(n_points):
        i_path += 1
        k = (i_path & -i_path).bit_length() - 1
        state ^= dirs[k]
        out[row] = state
    return out


def test_first_points_match_known_sequence():
    u = jax.vmap(Sobol(dim=3).uniform)(jnp.arange(5))
    expected = [
        [0.5, 0.5, 0.5],
        [0.75, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.375, 0.375, 0.625],
        [0.875, 0.875, 0.125],
    ]
    np.testing.assert_array_equal(np.asarray(u), expected)


@pytest.mark.parametrize(
    "dim,start", [(1, 0), (7, 0), (40, 1000), (300, 2**20 - 17), (5, 2**31 - 70)]
)
def test_random_access_equals_gray_code_recurrence(dim, start):
    seq = Sobol(dim=dim)
    states = jax.vmap(seq.state)(jnp.arange(start, start + 64))
    np.testing.assert_array_equal(
        np.asarray(states, dtype=np.uint64), sequential_states(dim, 64, start)
    )


def test_uniforms_are_states_scaled_by_two_to_minus_32():
    seq = Sobol(dim=11)
    ids = jnp.arange(100, 130)
    u = jax.vmap(seq.uniform)(ids)
    s = jax.vmap(seq.state)(ids)
    np.testing.assert_array_equal(np.asarray(u), np.asarray(s, dtype=np.float64) * MUL)
    assert MUL == 2.0**-32


def test_cached_tables_are_read_only():
    for table in (directions(5), digital_shifts(5, 42)):
        with pytest.raises(ValueError):
            table[0] = 1
    np.testing.assert_array_equal(
        np.asarray(jax.vmap(Sobol(dim=3).uniform)(jnp.arange(2)))[:, 0], [0.5, 0.75]
    )


def test_dimension_limits():
    assert directions(N_KNOWN - 1).shape == (32, N_KNOWN - 1)
    for bad in (0, N_KNOWN):
        with pytest.raises(InvalidRandomSequence):
            Sobol(dim=bad)


def test_digital_shift_uses_split_mix64_high_words():
    # SplitMix64 seeded with 0 starts 0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, ...
    np.testing.assert_array_equal(
        digital_shifts(2, 0), np.array([0xE220A839, 0x6E789E6A], dtype=np.uint32)
    )
    seq = Sobol(dim=2, shift_key=0)
    ids = jnp.arange(8)
    shifted = jax.vmap(seq.uniform)(ids)
    states = np.asarray(jax.vmap(seq.state)(ids))
    expected = ((states ^ digital_shifts(2, 0)).astype(np.float64) + 0.5) * MUL
    np.testing.assert_array_equal(np.asarray(shifted), expected)
    assert np.all((np.asarray(shifted) > 0.0) & (np.asarray(shifted) < 1.0))


def test_normals_are_finite_and_roughly_standard():
    z = np.asarray(jax.vmap(Sobol(dim=4).normal)(jnp.arange(2**14)))
    assert np.all(np.isfinite(z))
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=2e-3)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=2e-3)
