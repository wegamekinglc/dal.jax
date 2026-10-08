"""Multi-asset model contracts, factor order, observation slots and sensitivities."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dal_jax import (
    BlackScholes,
    CorrelatedBlackScholes,
    MonteCarloEngine,
    MonteCarloSettings,
    SampleDef,
    prepare,
)
from dal_jax.api import Product_New
from dal_jax.dates import Date
from dal_jax.errors import (
    DuplicateModelIndex,
    InvalidCorrelation,
    InvalidModelIndex,
    InvalidModelParameter,
    MissingDefaultIndex,
)
from dal_jax.models import Model
from dal_jax.script.product import ScriptProductSettings


def model(**kwargs):
    fields = dict(
        indices=("EQ[A]", "EQ[B]"),
        spots=(100.0, 95.0),
        vols=(0.15, 0.2),
        divs=(0.03, 0.02),
        rate=0.05,
        correlations=((1.0, 0.4), (0.4, 1.0)),
    )
    return CorrelatedBlackScholes(**(fields | kwargs))


def test_protocol_and_scalar_parameter_labels():
    bs = model()
    assert isinstance(bs, Model) and bs.n_factors == bs.num_assets == 2
    assert bs.param_labels == (
        "spot:EQ[A]",
        "vol:EQ[A]",
        "div:EQ[A]",
        "spot:EQ[B]",
        "vol:EQ[B]",
        "div:EQ[B]",
        "rate",
    )
    assert hash(bs) == hash(model())


def test_step_major_factors_and_per_date_observation_order():
    bs = model()
    defs = (
        SampleDef(index_names=("EQ[B]", "EQ[A]")),
        SampleDef(index_names=("EQ[A]",)),
        SampleDef(index_names=("EQ[B]",), discount_mats=(2.0,)),
    )
    plan = bs.allocate((0.0, 0.5, 1.0), defs)
    normals = [0.3, -0.8, 0.7, 1.1]
    scenario = jax.jit(lambda z: bs.generate(bs.init(bs.default_params(), plan), plan, z))(
        jnp.asarray(normals)
    )
    logs = np.log([100.0, 95.0])
    expected = [[100.0, 95.0]]
    lower = np.linalg.cholesky([[1.0, 0.4], [0.4, 1.0]])
    for i in range(2):
        correlated = lower @ np.asarray(normals[2 * i : 2 * i + 2])
        logs += (
            0.05 - np.asarray([0.03, 0.02]) - 0.5 * np.asarray([0.15, 0.2]) ** 2
        ) * 0.5 + np.asarray([0.15, 0.2]) * math.sqrt(0.5) * correlated
        expected.append(np.exp(logs).tolist())
    np.testing.assert_allclose(scenario.spot, np.asarray(expected)[:, 0], rtol=1e-15)
    np.testing.assert_allclose(
        scenario.observations[:, 0], [95.0, expected[1][0], expected[2][1]], rtol=1e-15
    )
    assert float(scenario.observations[0, 1]) == 100.0
    assert bs.sim_dim(plan) == 4
    np.testing.assert_allclose(scenario.discounts[-1, 0], math.exp(-0.05), rtol=1e-15)


def test_single_asset_reduces_to_black_scholes_on_identical_normals():
    corr = CorrelatedBlackScholes(
        indices=("EQ[A]",),
        spots=(100.0,),
        vols=(0.15,),
        divs=(0.03,),
        rate=0.05,
        correlations=((1.0,),),
    )
    bs = BlackScholes(spot=100.0, vol=0.15, div=0.03, rate=0.05)
    defs = (SampleDef(index_names=("EQ[A]",)),) * 3
    times = (0.2, 0.7, 1.0)
    z = jnp.asarray([0.1, -0.3, 0.7])
    a, b = corr.allocate(times, defs), bs.allocate(times, defs)
    np.testing.assert_allclose(
        corr.generate(corr.init(corr.default_params(), a), a, z).spot,
        bs.generate(bs.init(bs.default_params(), b), b, z).spot,
        rtol=1e-15,
    )


def test_default_index_selects_second_asset_and_ambiguous_spot_is_rejected():
    today = Date.ymd(2022, 9, 15)
    rows = ([today], ["pay PAYS SPOT()+FIX(EQ[B])"])
    bs = model()
    with pytest.raises(MissingDefaultIndex):
        prepare(Product_New(*rows), today, model=bs)
    data = Product_New(*rows, settings=ScriptProductSettings(default_index="eq[b]"))
    prepared = prepare(data, today, model=bs)
    assert len(prepared.observations) == 1
    result = MonteCarloEngine(
        prepared.path_product(), bs, MonteCarloSettings(enable_aad=True)
    ).value(16)
    np.testing.assert_allclose(
        [result["PV"], result["d_spot:EQ[A]"], result["d_spot:EQ[B]"]], [190.0, 0.0, 2.0], atol=0
    )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"indices": ("EQ[A]", "eq[a]")}, DuplicateModelIndex),
        ({"indices": ("EQ[A]>3M", "EQ[B]")}, InvalidModelIndex),
        ({"spots": (0.0, 95.0)}, InvalidModelParameter),
        ({"vols": (0.15,)}, InvalidModelParameter),
        ({"correlations": ((1.0, 1.0), (1.0, 1.0))}, InvalidCorrelation),
        ({"correlations": ((1.0, 0.4), (0.2, 1.0))}, InvalidCorrelation),
        ({"correlations": ((0.9, 0.4), (0.4, 1.0))}, InvalidCorrelation),
        ({"correlations": ((1.0, np.nan), (np.nan, 1.0))}, InvalidCorrelation),
        ({"correlations": ((1.0,),)}, InvalidCorrelation),
    ],
)
def test_invalid_model_inputs(kwargs, error):
    with pytest.raises(error):
        model(**kwargs)
