"""Exact MRG32k3a uniforms matching DAL's PseudoRSG stream."""

import numpy as np
import pytest

from dal_jax.api import PseudoRSG_Get_Normal, PseudoRSG_Get_Uniform, PseudoRSG_New
from dal_jax.errors import InvalidSetting
from dal_jax.random.inverse_normal import inverse_ncdf
from dal_jax.random.pseudorng import M1, PseudoRSG, _mrg32_ks, mrg32_state, mrg32_uniforms


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
    with pytest.raises(InvalidSetting, match="expected a nonnegative MRG32 seed"):
        PseudoRSG(-1)
    with pytest.raises(InvalidSetting, match="expected an integer"):
        PseudoRSG(1.5)
    with pytest.raises(InvalidSetting, match="expected a positive integer"):
        PseudoRSG(7, ndim=0)
    with pytest.raises(InvalidSetting, match="expected a positive integer"):
        PseudoRSG(7, ndim=2.5)


def test_pseudo_rsg_rejects_bool_seed():
    with pytest.raises(InvalidSetting, match="expected an integer"):
        PseudoRSG_New(True, 1)


def test_pseudo_rsg_zero_rows_return_empty_matrices():
    rsg = PseudoRSG(7, 3)
    assert rsg.fill_uniform(0).shape == (0, 3)
    assert rsg.fill_normal(0).shape == (0, 3)


def test_pseudo_rsg_rejects_invalid_row_counts():
    rsg = PseudoRSG(7, 3)
    with pytest.raises(InvalidSetting, match="expected a nonnegative integer"):
        rsg.fill_uniform(-2)
    with pytest.raises(InvalidSetting, match="expected a nonnegative integer"):
        rsg.fill_normal(-2)
    with pytest.raises(InvalidSetting, match="expected a nonnegative integer"):
        PseudoRSG_Get_Uniform(rsg, 2.7)
    with pytest.raises(InvalidSetting, match="expected a nonnegative integer"):
        PseudoRSG_Get_Normal(rsg, True)


def test_fill_normal_precise_false_uses_fast_inverse():
    rsg = PseudoRSG(7, 2, precise=False)
    ks, _ = _mrg32_ks(mrg32_state(7), 8)
    expected = inverse_ncdf(np.asarray(ks) / (M1 + 1), precise=False, polish=False)
    np.testing.assert_array_equal(rsg.fill_normal(4), np.asarray(expected).reshape(4, 2))


def test_batched_fill_uniform_matches_row_at_a_time():
    batched, stepped = PseudoRSG(7, 3), PseudoRSG(7, 3)
    batched_rows = [
        batched.fill_uniform(1),  # fresh entry, odd length -> antithetic entry next
        batched.fill_uniform(1),  # antithetic entry: reuses the cached fresh row
        batched.fill_uniform(2),  # fresh entry
        batched.fill_uniform(3),  # fresh entry, odd length -> antithetic entry next
    ]
    batched_normals = batched.fill_normal(2)  # must not disturb the alternation
    batched_rows.append(batched.fill_uniform(2))  # antithetic entry

    stepped_rows = [stepped.fill_uniform(1) for _ in range(1 + 1 + 2 + 3)]
    stepped_normals = stepped.fill_normal(2)
    stepped_rows += [stepped.fill_uniform(1) for _ in range(2)]

    np.testing.assert_array_equal(np.vstack(batched_rows), np.vstack(stepped_rows))
    np.testing.assert_array_equal(batched_normals, stepped_normals)
