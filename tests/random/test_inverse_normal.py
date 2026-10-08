import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from dal_jax.random.inverse_normal import inverse_ncdf, inverse_ncdf_ndtri, ncdf

GRID = np.concatenate(
    [
        np.geomspace(1e-12, 0.02425, 200),
        np.linspace(0.02425, 0.97575, 401),
        1.0 - np.geomspace(1e-12, 0.02425, 200),
    ]
)


def test_acklam_accuracy_matches_its_published_bound():
    z = np.asarray(inverse_ncdf(GRID))
    exact = ndtri(GRID)
    assert np.max(np.abs(z - exact) / np.maximum(np.abs(exact), 1.0)) < 1.2e-9


def test_polish_with_erfc_reaches_double_precision():
    #  DAL caps the Newton step's exp(z^2 / 2) at exp(8), so the polish only fully applies for |z| <= 4.
    x = GRID[(GRID > 1e-4) & (GRID < 1 - 1e-4)]
    np.testing.assert_allclose(
        np.asarray(inverse_ncdf(x, precise=True, polish=True)), ndtri(x), rtol=1e-12, atol=1e-12
    )


def test_spline_ncdf_is_a_coarse_but_continuous_approximation():
    z = np.linspace(-6.0, 6.0, 2001)
    approx = np.asarray(ncdf(z, precise=False))
    assert np.max(np.abs(approx - ndtr(z))) < 3e-5
    assert np.all(np.diff(approx) >= 0.0)
    np.testing.assert_allclose(approx + approx[::-1], 1.0, atol=1e-15)


def test_precise_ncdf_is_erfc():
    z = np.linspace(-8.0, 8.0, 101)
    np.testing.assert_allclose(np.asarray(ncdf(z)), ndtr(z), rtol=1e-13, atol=1e-300)


@pytest.mark.parametrize("precise,polish", [(False, False), (False, True), (True, True)])
def test_inverse_is_antisymmetric(precise, polish):
    x = np.linspace(0.001, 0.499, 50)
    lo = np.asarray(inverse_ncdf(x, precise, polish))
    hi = np.asarray(inverse_ncdf(1.0 - x, precise, polish))
    np.testing.assert_allclose(lo, -hi, atol=5e-9)


def test_ndtri_variant():
    np.testing.assert_allclose(np.asarray(inverse_ncdf_ndtri(GRID)), ndtri(GRID), rtol=1e-14)
