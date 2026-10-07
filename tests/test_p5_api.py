"""The new DAL-compatible factories support full valuation through settings."""

import numpy as np

from dal_jax.api import (BSModelData_New, BSLocalVolModelData_New, CorrelatedBSModelData_New,
                         LocalVolSurfaceData_New, MarketFixingSnapshot_New, MonteCarlo_ValueWithSettings,
                         Product_New, ScriptValuationSettings_)
from dal_jax import MonteCarloSettings
from dal_jax.dates import Date

TODAY = Date.ymd(2022, 9, 15)


def test_fixing_and_valuation_factories_capture_explicit_date():
    snapshot = MarketFixingSnapshot_New({"EQ[A]": {TODAY: 97.}})
    valuation = ScriptValuationSettings_(evaluation_date=TODAY, fixings=snapshot, today_fixing="RequireHistorical")
    product = Product_New([TODAY], ["pay PAYS FIX(EQ[A])"])
    result = MonteCarlo_ValueWithSettings(product, BSModelData_New(100., .15), 16,
                                         valuation=valuation, simulation=MonteCarloSettings(enable_aad=True))
    np.testing.assert_allclose([result["PV"], result["d_spot"]], [97., 0.], atol=0)


def test_correlated_factory_exposes_every_model_risk():
    model = CorrelatedBSModelData_New(["EQ[A]", "EQ[B]"], [100., 95.], [.15, .2], [.03, .02], .05, [[1., .4], [.4, 1.]])
    product = Product_New([TODAY], ["pay PAYS FIX(EQ[A])+FIX(EQ[B])"])
    result = MonteCarlo_ValueWithSettings(product, model, 16, valuation=ScriptValuationSettings_(evaluation_date=TODAY),
                                         simulation=MonteCarloSettings(enable_aad=True))
    assert set(result) == {"PV"} | {f"d_{label}" for label in model.param_labels}
    np.testing.assert_allclose([result["PV"], result["d_spot:EQ[A]"], result["d_spot:EQ[B]"]], [195., 1., 1.], atol=0)


def test_localvol_factory_replaces_bs_vol_and_preserves_surface_axes():
    surface = LocalVolSurfaceData_New("surface", [70., 100., 140.], [0., 1.], [[.15, .15]]*3)
    model = BSLocalVolModelData_New("local_vol", "EQ[A]", "USD", "W_EQ", BSModelData_New(100., .9, .05, .03), surface, .25)
    assert model.surface == surface and model.max_step == .25
    product = Product_New([TODAY], ["pay PAYS FIX(EQ[A])"])
    result = MonteCarlo_ValueWithSettings(product, model, 16, valuation=ScriptValuationSettings_(evaluation_date=TODAY),
                                         simulation=MonteCarloSettings(enable_aad=True))
    assert "d_vol" not in result
    assert all(result[f"d_{label}"] == 0. for label in model.param_labels if label.startswith("lvol:"))
    assert result["PV"] == 100.
