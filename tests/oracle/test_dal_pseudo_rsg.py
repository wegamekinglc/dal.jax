"""PseudoRSG parity with dal-python's MRG32 stream."""

import numpy as np
import pytest

from dal_jax.api import PseudoRSG_Get_Normal, PseudoRSG_Get_Uniform, PseudoRSG_New

pytestmark = pytest.mark.oracle


@pytest.mark.parametrize("seed,ndim", [(7, 3), (42, 1), (1024, 5)])
def test_uniforms_match_native_stream(dal, seed, ndim):
    ours, theirs = PseudoRSG_New(seed, ndim), dal.PseudoRSG_New(seed, ndim)
    np.testing.assert_allclose(
        PseudoRSG_Get_Uniform(ours, 9),
        np.asarray(dal.PseudoRSG_Get_Uniform(theirs, 9).to_rows()),
        atol=0,
        rtol=1e-15,
    )
    # the antithetic alternation and the stream position survive across calls
    np.testing.assert_allclose(
        PseudoRSG_Get_Uniform(ours, 4),
        np.asarray(dal.PseudoRSG_Get_Uniform(theirs, 4).to_rows()),
        atol=0,
        rtol=1e-15,
    )


@pytest.mark.parametrize("seed,ndim", [(7, 3), (42, 2)])
def test_normals_match_native_stream(dal, seed, ndim):
    ours, theirs = PseudoRSG_New(seed, ndim), dal.PseudoRSG_New(seed, ndim)
    np.testing.assert_allclose(
        PseudoRSG_Get_Normal(ours, 8),
        np.asarray(dal.PseudoRSG_Get_Normal(theirs, 8).to_rows()),
        rtol=1e-14,
        atol=1e-15,
    )


def test_interleaved_calls_match_native_stream(dal):
    ours, theirs = PseudoRSG_New(11, 3), dal.PseudoRSG_New(11, 3)
    np.testing.assert_allclose(
        PseudoRSG_Get_Uniform(ours, 3),
        np.asarray(dal.PseudoRSG_Get_Uniform(theirs, 3).to_rows()),
        atol=0,
        rtol=1e-15,
    )
    # normals must not disturb the antithetic alternation or the stream position
    np.testing.assert_allclose(
        PseudoRSG_Get_Normal(ours, 2),
        np.asarray(dal.PseudoRSG_Get_Normal(theirs, 2).to_rows()),
        rtol=1e-14,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        PseudoRSG_Get_Uniform(ours, 2),
        np.asarray(dal.PseudoRSG_Get_Uniform(theirs, 2).to_rows()),
        atol=0,
        rtol=1e-15,
    )
