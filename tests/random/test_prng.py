import numpy as np
import pytest

from dal_jax.errors import InvalidSetting
from dal_jax.random.prng import block_normals, prng_key


def test_blocks_are_reproducible_and_independent():
    key = prng_key("mrg32", 1024)
    a = np.asarray(block_normals(key, 3, 4096, 5))
    np.testing.assert_array_equal(a, np.asarray(block_normals(prng_key("mrg32", 1024), 3, 4096, 5)))
    b = np.asarray(block_normals(key, 4, 4096, 5))
    assert abs(np.corrcoef(a.ravel(), b.ravel())[0, 1]) < 0.02
    assert a.dtype == np.float64
    np.testing.assert_allclose(a.mean(), 0.0, atol=0.03)
    np.testing.assert_allclose(a.std(), 1.0, atol=0.03)


def test_generator_names_select_distinct_streams():
    a = np.asarray(block_normals(prng_key("mrg32", 7), 0, 16, 2))
    b = np.asarray(block_normals(prng_key("irn", 7), 0, 16, 2))
    assert not np.array_equal(a, b)


def test_alternative_key_implementation():
    z = np.asarray(block_normals(prng_key("irn", 1, impl="rbg"), 0, 1024, 3))
    assert np.all(np.isfinite(z))


def test_unknown_generator():
    with pytest.raises(InvalidSetting):
        prng_key("sobol", 1)
