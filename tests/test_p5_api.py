"""The new DAL-compatible factories support full valuation through settings."""

import numpy as np

from dal_jax import MonteCarloSettings
from dal_jax.api import (
    BSLocalVolModelData_New,
    BSModelData_New,
    CorrelatedBSModelData_New,
    LocalVolSurfaceData_New,
    MarketFixingSnapshot_New,
    MonteCarlo_ValueWithSettings,
    Product_New,
    ScriptValuationSettings_,
)
from dal_jax.dates import Date

TODAY = Date.ymd(2022, 9, 15)


def test_fixing_and_valuation_factories_capture_explicit_date():
    snapshot = MarketFixingSnapshot_New({"EQ[A]": {TODAY: 97.0}})
    valuation = ScriptValuationSettings_(
        evaluation_date=TODAY, fixings=snapshot, today_fixing="RequireHistorical"
    )
    product = Product_New([TODAY], ["pay PAYS FIX(EQ[A])"])
    result = MonteCarlo_ValueWithSettings(
        product,
        BSModelData_New(100.0, 0.15),
        16,
        valuation=valuation,
        simulation=MonteCarloSettings(enable_aad=True),
    )
    np.testing.assert_allclose([result["PV"], result["d_spot"]], [97.0, 0.0], atol=0)


def test_correlated_factory_exposes_every_model_risk():
    model = CorrelatedBSModelData_New(
        ["EQ[A]", "EQ[B]"], [100.0, 95.0], [0.15, 0.2], [0.03, 0.02], 0.05, [[1.0, 0.4], [0.4, 1.0]]
    )
    product = Product_New([TODAY], ["pay PAYS FIX(EQ[A])+FIX(EQ[B])"])
    result = MonteCarlo_ValueWithSettings(
        product,
        model,
        16,
        valuation=ScriptValuationSettings_(evaluation_date=TODAY),
        simulation=MonteCarloSettings(enable_aad=True),
    )
    assert set(result) == {"PV"} | {f"d_{label}" for label in model.param_labels}
    np.testing.assert_allclose(
        [result["PV"], result["d_spot:EQ[A]"], result["d_spot:EQ[B]"]], [195.0, 1.0, 1.0], atol=0
    )


def test_localvol_factory_replaces_bs_vol_and_preserves_surface_axes():
    surface = LocalVolSurfaceData_New(
        "surface", [70.0, 100.0, 140.0], [0.0, 1.0], [[0.15, 0.15]] * 3
    )
    model = BSLocalVolModelData_New(
        "local_vol", "EQ[A]", "USD", "W_EQ", BSModelData_New(100.0, 0.9, 0.05, 0.03), surface, 0.25
    )
    assert model.surface == surface and model.max_step == 0.25
    product = Product_New([TODAY], ["pay PAYS FIX(EQ[A])"])
    result = MonteCarlo_ValueWithSettings(
        product,
        model,
        16,
        valuation=ScriptValuationSettings_(evaluation_date=TODAY),
        simulation=MonteCarloSettings(enable_aad=True),
    )
    assert "d_vol" not in result
    assert all(
        result[f"d_{label}"] == 0.0 for label in model.param_labels if label.startswith("lvol:")
    )
    assert result["PV"] == 100.0
