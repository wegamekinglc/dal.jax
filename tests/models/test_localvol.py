"""Local-vol interpolation, subdivision, Euler paths and complete bucket gradients."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dal_jax import (
    LocalVol,
    LocalVolSurface,
    MonteCarloEngine,
    MonteCarloSettings,
    SampleDef,
    prepare,
)
from dal_jax.api import Product_New
from dal_jax.dates import Date
from dal_jax.errors import InvalidLocalVolSurface, InvalidModelParameter
from dal_jax.models import Model


def surface(**kw):
    return LocalVolSurface(
        **(dict(spots=(80.0, 120.0), times=(0.0, 1.0), vols=((0.1, 0.2), (0.3, 0.4))) | kw)
    )


def test_surface_interpolates_log_spot_and_time_with_flat_extrapolation():
    surf = surface()
    midpoint = math.sqrt(80 * 120)
    np.testing.assert_allclose(surf.volatility(0.25, midpoint), 0.225, atol=1e-15)
    np.testing.assert_allclose(
        [surf.volatility(-1.0, 70.0), surf.volatility(2.0, 130.0)], [0.1, 0.4], atol=0
    )
    derivative = jax.grad(lambda s: surf.volatility(0.25, s))
    np.testing.assert_allclose(
        derivative(midpoint), 0.2 / (midpoint * math.log(120 / 80)), rtol=1e-14
    )
    assert float(derivative(80.0)) == float(derivative(120.0)) == 0.0


def test_single_point_surface_and_bucket_weight_partition():
    surf = surface(spots=(100.0,), times=(0.0,), vols=((0.2,),))
    assert float(surf.volatility(1.0, 200.0)) == 0.2
    surf = surface()
    gradients = jax.grad(lambda v: surf.volatility(0.25, math.sqrt(80 * 120), vols=v))(
        jnp.asarray(surf.vols)
    )
    np.testing.assert_allclose(gradients, [[0.375, 0.125], [0.375, 0.125]], atol=1e-14)
    assert float(jnp.sum(gradients)) == 1.0


def test_grid_equal_subdivisions_and_today_zero_random_dimension():
    model = LocalVol(spot=100.0, surface=surface(), max_step=0.4)
    assert isinstance(model, Model)
    plan = model.allocate((0.0, 0.5, 1.0), (SampleDef(),) * 3)
    assert plan.grid == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert plan.sample_indices == (0, 2, 4)
    assert model.sim_dim(plan) == 4
    today = model.allocate((0.0,), (SampleDef(),))
    assert model.sim_dim(today) == 0
    scenario = jax.jit(lambda p: model.generate(model.init(p, today), today, jnp.empty(0)))(
        model.default_params()
    )
    assert float(scenario.spot[0]) == 100.0


def test_constant_surface_matches_explicit_log_euler_and_parameter_adjoint():
    model = LocalVol(
        spot=100.0,
        surface=surface(vols=((0.2, 0.2), (0.2, 0.2))),
        rate=0.05,
        div=0.02,
        max_step=0.25,
    )
    plan = model.allocate((1.0,), (SampleDef(),))
    z = jnp.asarray([0.1, -0.5, 0.7, 0.3])

    def terminal(p):
        return model.generate(model.init(p, plan), plan, z).spot[-1]

    value, gradients = jax.jit(jax.value_and_grad(terminal))(model.default_params())
    expected = 100.0 * math.exp((0.05 - 0.02 - 0.5 * 0.2**2) + 0.2 * 0.5 * float(jnp.sum(z)))
    np.testing.assert_allclose(value, expected, rtol=1e-14)
    np.testing.assert_allclose(gradients["spot:EQ[spot]"], expected / 100, rtol=1e-14)
    # Bumping all surface nodes together equals a single flat volatility bump.
    np.testing.assert_allclose(
        jnp.sum(model.vol_grid(gradients)), expected * (-0.2 + 0.5 * float(jnp.sum(z))), rtol=1e-13
    )


def test_nonflat_bucket_vegas_match_common_path_differences():
    today = Date.ymd(2022, 9, 15)
    model = LocalVol(
        spot=100.0, index="EQ[A]", surface=surface(), rate=0.05, div=0.02, max_step=0.25
    )
    prepared = prepare(
        Product_New([today.add_days(365)], ["pay PAYS MAX(FIX(EQ[A])-100,0)"]), today, model=model
    )
    engine = MonteCarloEngine(prepared.path_product(), model, MonteCarloSettings(enable_aad=True))
    params = engine.default_params()
    price = jax.jit(lambda p: engine.pricer(4096)(p)[0])
    gradients = jax.jit(jax.grad(price))(params)
    for name in model.param_labels:
        step = 1e-5 if name.startswith("spot:") else 1e-7
        up = params | {"model": params["model"] | {name: params["model"][name] + step}}
        down = params | {"model": params["model"] | {name: params["model"][name] - step}}
        finite_difference = float((price(up) - price(down)) / (2 * step))
        np.testing.assert_allclose(
            gradients["model"][name], finite_difference, rtol=2e-6, atol=1e-7, err_msg=name
        )


@pytest.mark.parametrize("aad", [False, True])
def test_float32_euler_carry_and_bucket_risks(aad, cpu_devices):
    from p5_cases import TODAY, case

    rows, model = case("localvol_skew")
    product = prepare(Product_New(*rows), TODAY, model=model).path_product()
    settings = dict(enable_aad=aad, devices=cpu_devices, block_size=512)
    engine = MonteCarloEngine(product, model, MonteCarloSettings(**settings, dtype="float32"))
    reference = MonteCarloEngine(product, model, MonteCarloSettings(**settings, dtype="float64"))
    ours, theirs = engine.value(4097), reference.value(4097)
    assert engine.path_payoffs(engine.default_params(), 0, 4097)[1].dtype == jnp.float32
    for name in ours:
        np.testing.assert_allclose(
            ours[name], theirs[name], rtol=2e-5 if name == "PV" else 5e-3, atol=2e-4, err_msg=name
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spots": ()},
        {"spots": (120.0, 80.0)},
        {"spots": (0.0, 120.0)},
        {"times": (-1.0, 1.0)},
        {"times": (0.0, 0.0)},
        {"vols": ((0.2,), (0.2,))},
        {"vols": ((0.1, np.nan), (0.3, 0.4))},
        {"vols": ((-0.1, 0.2), (0.3, 0.4))},
    ],
)
def test_invalid_surfaces(kwargs):
    with pytest.raises(InvalidLocalVolSurface):
        surface(**kwargs)


@pytest.mark.parametrize("maximum", [0.0, -0.1, np.inf])
def test_invalid_step(maximum):
    with pytest.raises(InvalidModelParameter):
        LocalVol(spot=100.0, surface=surface(), max_step=maximum)
