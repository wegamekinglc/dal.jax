"""Exact MRG32k3a uniforms matching DAL's PseudoRSG stream."""

import numpy as np
import pytest

from dal_jax.errors import InvalidSetting
from dal_jax.random.pseudorng import PseudoRSG, mrg32_state, mrg32_uniforms


def test_stream_matches_reference_values():
    state = mrg32_state(7)
    draws, state = mrg32_uniforms(state, 6)
    assert np.allclose(
        draws[:3],
        [0.0025416781494089066, 0.4468918780222327, 0.3718702828858557],
        atol=0,
        rtol=1e-9,
    )
    assert np.allclose(
        draws[3:],
        [0.8711512142791069, 0.1083147333770675, 0.9690315049510806],
        atol=0,
        rtol=1e-9,
    )
    # resuming from the carried state continues the exact stream
    more, _ = mrg32_uniforms(state, 3)
    assert np.allclose(
        more,
        [0.6944798374203514, 0.08077386994868632, 0.29431771748189006],
        atol=0,
        rtol=1e-9,
    )


def test_seed_zero_matches_native():
    draws, _ = mrg32_uniforms(mrg32_state(0), 3)
    assert np.allclose(
        draws, [0.0002015431043507917, 0.5557426818633633, 0.27246401940298176], rtol=1e-9
    )


def test_pseudo_rsg_rejects_invalid_settings():
    with pytest.raises(InvalidSetting):
        PseudoRSG(-1)
    with pytest.raises(InvalidSetting):
        PseudoRSG(1.5)
    with pytest.raises(InvalidSetting):
        PseudoRSG(7, ndim=0)
    with pytest.raises(InvalidSetting):
        PseudoRSG(7, ndim=2.5)
