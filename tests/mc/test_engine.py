import math
from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from support import (
    DIV,
    MATURITY,
    RATE,
    SPOT,
    STRIKE,
    VOL,
    bs_call_price,
    bs_model,
    call_payoff,
    european_call,
    monthly_barrier_timeline,
    up_and_out_call,
)

from dal_jax import BlackScholes, MonteCarloEngine, MonteCarloSettings, PathProduct
from dal_jax.errors import (
    InvalidPathCount,
    InvalidPayoff,
    InvalidSetting,
    InvalidSmoothing,
    ReservedIdentifier,
    UnsupportedBrownianBridge,
)


def test_compiled_engine_configuration_is_frozen(one_cpu):
    from dal_jax.api import Product_New
    from dal_jax.dates import Date
    from dal_jax.script.preparation import prepare

    today = Date.ymd(2026, 1, 1)
    product = prepare(
        Product_New([today.add_days(365)], ["IF SPOT()>105 THEN pay PAYS 1 ELSE pay PAYS 0 END"]),
        today,
    )
    model = BlackScholes(spot=100.0, vol=0.2)
    settings = MonteCarloSettings(devices=one_cpu, block_size=256, enable_aad=True)
    original = product.engine(model, settings)
    before = original.value(1024)
    for field, value in (
        ("settings", replace(settings, smooth=60.0)),
        ("model", model),
        ("product", product.path_product()),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(original, field, value)
    changed = product.engine(model, replace(settings, smooth=60.0)).value(1024)
    assert original.value(1024) == before
    assert abs(changed["PV"] - before["PV"]) > 0.01


@pytest.fixture
def one_cpu(cpu_devices):
    return cpu_devices[:1]


def engine(product=None, model=None, devices=None, **settings) -> MonteCarloEngine:
    return MonteCarloEngine(
        product or european_call(),
        model or bs_model(),
        MonteCarloSettings(devices=devices, **settings),
    )


def test_sobol_price_converges_to_black_scholes(one_cpu):
    pv = engine(devices=one_cpu).value(2**18)["PV"]
    exact = bs_call_price(SPOT, STRIKE, VOL, RATE, DIV, MATURITY)
    assert abs(pv - exact) / exact < 2e-4


@pytest.mark.parametrize("rsg", ["mrg32", "irn"])
def test_prng_price_within_three_standard_errors(one_cpu, rsg):
    eng = engine(devices=one_cpu, rsg=rsg)
    n = 2**16
    params = eng.default_params()
    values = np.concatenate(
        [np.asarray(eng.path_payoffs(params, b, n)[1][:, 0]) for b in range(eng.layout(n).n_blocks)]
    )[:n]
    se = values.std() / math.sqrt(n)
    exact = bs_call_price(SPOT, STRIKE, VOL, RATE, DIV, MATURITY)
    assert abs(eng.value(n)["PV"] - exact) < 3.0 * se


def test_greeks_match_closed_form(one_cpu):
    result = engine(devices=one_cpu, enable_aad=True).value(2**18)
    assert set(result) == {"PV", "d_spot", "d_vol", "d_rate", "d_div", "d_STRIKE"}

    def price(spot=SPOT, vol=VOL, rate=RATE, div=DIV, strike=STRIKE):
        return bs_call_price(spot, strike, vol, rate, div, MATURITY)

    h = 1e-5
    for key, arg, base in [
        ("d_spot", "spot", SPOT),
        ("d_vol", "vol", VOL),
        ("d_rate", "rate", RATE),
        ("d_div", "div", DIV),
        ("d_STRIKE", "strike", STRIKE),
    ]:
        fd = (price(**{arg: base + h}) - price(**{arg: base - h})) / (2 * h)
        assert abs(result[key] - fd) <= 2e-3 * max(1.0, abs(fd)), key


def test_gradient_equals_finite_difference_with_common_random_numbers(one_cpu):
    product = up_and_out_call(monthly_barrier_timeline(), tuple(range(1, 37)) + (36,))
    eng = engine(product, devices=one_cpu, enable_aad=True)
    f = jax.jit(lambda p: eng.pricer(2**14)(p)[0])
    params = eng.default_params()
    grads = jax.grad(f)(params)
    #  Steps small enough that almost no path crosses a kink of CSpr or MAX.
    for group, name, h in [
        ("model", "spot", 1e-6),
        ("model", "vol", 1e-7),
        ("script", "BARRIER", 1e-6),
        ("script", "STRIKE", 1e-6),
    ]:
        up = jax.tree.map(lambda x: x, params)
        dn = jax.tree.map(lambda x: x, params)
        up[group][name] = params[group][name] + h
        dn[group][name] = params[group][name] - h
        fd = (float(f(up)) - float(f(dn))) / (2 * h)
        np.testing.assert_allclose(float(grads[group][name]), fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("enable_aad", [False, True])
def test_vectorized_barrier_matches_date_by_date(one_cpu, enable_aad):
    timeline = monthly_barrier_timeline(1)
    dates = tuple(range(1, 13)) + (12,)
    loop = engine(up_and_out_call(timeline, dates), devices=one_cpu, enable_aad=enable_aad).value(
        2**12
    )
    vec = engine(
        up_and_out_call(timeline, dates, vectorized=True), devices=one_cpu, enable_aad=enable_aad
    ).value(2**12)
    for key in loop:
        np.testing.assert_allclose(vec[key], loop[key], rtol=1e-12, atol=1e-15)


def test_pricer_composes_with_jax_transforms(one_cpu):
    eng = engine(devices=one_cpu)
    f = eng.pricer(2**12, fuzzy=True)
    params = eng.default_params()
    pv = f(params)
    assert pv.shape == (1,)
    np.testing.assert_allclose(jax.jit(f)(params), pv, rtol=1e-15)
    rev = jax.jacrev(f)(params)
    fwd = jax.jacfwd(f)(params)
    for group in ("model", "script"):
        for name in params[group]:
            np.testing.assert_allclose(rev[group][name], fwd[group][name], rtol=1e-12)
    gamma = jax.hessian(
        lambda s: f({"model": params["model"] | {"spot": s}, "script": params["script"]})[0]
    )(params["model"]["spot"])
    assert np.isfinite(float(gamma))

    ladder = jnp.linspace(80.0, 120.0, 5)
    by_spot = jax.vmap(
        lambda s: f({"model": params["model"] | {"spot": s}, "script": params["script"]})[0]
    )(ladder)
    one_by_one = [
        float(f({"model": params["model"] | {"spot": s}, "script": params["script"]})[0])
        for s in ladder
    ]
    np.testing.assert_allclose(np.asarray(by_spot), one_by_one, rtol=1e-13)
    assert np.all(np.diff(np.asarray(by_spot)) > 0.0)


def test_multiple_payoffs_and_jacobian(one_cpu):
    def payoffs(params, s, ctx):
        k = params["script"]["STRIKE"]
        st, df = s.spot[-1], 1.0 / s.numeraire[-1]
        return jnp.stack([jnp.maximum(st - k, 0.0) * df, jnp.maximum(k - st, 0.0) * df, st * df])

    product = PathProduct(
        timeline=(1.0,),
        payoff=payoffs,
        payoff_names=("call", "put", "fwd"),
        script_params={"STRIKE": 100.0},
    )
    eng = engine(product, devices=one_cpu)
    f = eng.pricer(2**14)
    params = eng.default_params()
    call, put, fwd = (float(v) for v in f(params))
    np.testing.assert_allclose(call - put, fwd - 100.0 * math.exp(-RATE), rtol=1e-12)
    np.testing.assert_allclose(fwd, SPOT * math.exp(-DIV), rtol=1e-3)
    jac = jax.jacrev(f)(params)
    assert jac["model"]["spot"].shape == (3,)
    assert eng.value(2**14, payoff="put")["PV"] == pytest.approx(put, rel=1e-14)


def test_partial_last_block_is_masked(one_cpu):
    n = 5000
    blocked = engine(devices=one_cpu, block_size=1024)
    params = blocked.default_params()
    values = np.concatenate(
        [np.asarray(blocked.path_payoffs(params, b, n)[1][:, 0]) for b in range(5)]
    )
    assert values.shape == (5120,)
    np.testing.assert_allclose(blocked.value(n)["PV"], values[:n].mean(), rtol=1e-13)
    np.testing.assert_allclose(
        engine(devices=one_cpu).value(n)["PV"], values[:n].mean(), rtol=1e-13
    )


def test_block_bucketing_reuses_compilation(one_cpu):
    bucketed = engine(devices=one_cpu, block_size=256, block_bucketing=True)
    assert bucketed.layout(3000) == bucketed.layout(4000)
    plain = engine(devices=one_cpu, block_size=256)
    for n in (3000, 4000):
        np.testing.assert_allclose(bucketed.value(n)["PV"], plain.value(n)["PV"], rtol=1e-13)
    assert len(bucketed._compiled) == 1


def test_checkpoint_does_not_change_results(one_cpu):
    product = up_and_out_call(monthly_barrier_timeline(1), tuple(range(1, 13)))
    a = engine(product, devices=one_cpu, enable_aad=True, checkpoint=True).value(2**12)
    b = engine(product, devices=one_cpu, enable_aad=True, checkpoint=False).value(2**12)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-13, atol=1e-15)


def test_float32_paths_stay_close_to_float64(one_cpu):
    f64 = engine(devices=one_cpu, enable_aad=True).value(2**14)
    f32 = engine(devices=one_cpu, enable_aad=True, dtype="float32").value(2**14)
    for key in f64:
        np.testing.assert_allclose(f32[key], f64[key], rtol=1e-4, atol=1e-5)


def test_inverse_normal_variants_agree(one_cpu):
    base = engine(devices=one_cpu).value(2**14)["PV"]
    #  Acklam is accurate to ~1e-9; DAL's spline NCDF (used by the plain polish) only to ~1e-5.
    for variant, rtol in (
        ("acklam_polish_precise", 1e-7),
        ("ndtri", 1e-7),
        ("acklam_polish", 1e-3),
    ):
        np.testing.assert_allclose(
            engine(devices=one_cpu, inverse_normal=variant).value(2**14)["PV"], base, rtol=rtol
        )


def test_digital_shift_changes_points_but_not_the_price(one_cpu):
    plain = engine(devices=one_cpu).value(2**14)["PV"]
    shifted = engine(devices=one_cpu, sobol_shift_key=42).value(2**14)["PV"]
    assert plain != shifted
    np.testing.assert_allclose(shifted, plain, rtol=5e-3)


def test_bridge_keeps_terminal_distribution(one_cpu):
    product = up_and_out_call(monthly_barrier_timeline(1), ())
    exact = bs_call_price(SPOT, STRIKE, VOL, RATE, DIV, monthly_barrier_timeline(1)[-1])
    for use_bb in (False, True):
        pv = engine(product, devices=one_cpu, use_bb=use_bb).value(2**16)["PV"]
        assert abs(pv - exact) / exact < 5e-3


def test_today_only_product(one_cpu):
    product = PathProduct(timeline=(0.0,), payoff=lambda p, s, ctx: s.spot[0] - 90.0)
    assert engine(product, devices=one_cpu).value(100)["PV"] == pytest.approx(10.0, rel=1e-15)


def test_expired_product_returns_zero(one_cpu):
    product = PathProduct(timeline=(), payoff=call_payoff, script_params={"STRIKE": 1.0})
    result = engine(product, devices=one_cpu, enable_aad=True).value(10)
    assert result == {
        "PV": 0.0,
        "d_spot": 0.0,
        "d_vol": 0.0,
        "d_rate": 0.0,
        "d_div": 0.0,
        "d_STRIKE": 0.0,
    }


def test_non_finite_payoff_raises(one_cpu):
    product = PathProduct(timeline=(1.0,), payoff=lambda p, s, ctx: jnp.log(s.spot[-1] - 100.0))
    with pytest.raises(InvalidPayoff):
        engine(product, devices=one_cpu).value(1024)


def test_invalid_inputs(one_cpu):
    eng = engine(devices=one_cpu)
    for bad in (0, -5, 1.5):
        with pytest.raises(InvalidPathCount):
            eng.value(bad)
    with pytest.raises(InvalidPathCount):
        eng.layout(2**32)
    with pytest.raises(InvalidSetting):
        MonteCarloSettings(rsg="halton")
    with pytest.raises(InvalidSmoothing):
        MonteCarloSettings(smooth=0.0)
    with pytest.raises(InvalidSetting):
        MonteCarloSettings(rsg="mrg32", sobol_shift_key=1)
    with pytest.raises(InvalidSetting):
        MonteCarloSettings(parallel="threads")
    with pytest.raises(InvalidSetting):
        PathProduct(timeline=(1.0,), payoff=call_payoff, payoff_names=())


def test_record_collection_validates_counts_after_cache_warmup(one_cpu):
    eng = engine(devices=one_cpu)
    with pytest.raises(InvalidPathCount):
        eng.path_collector(1.0)
    records = eng.path_collector(1)(eng.default_params())
    assert records.shape == (1, 1)
    with pytest.raises(InvalidPathCount):
        eng.path_collector(1.0)


@pytest.mark.parametrize(
    "settings",
    [
        {"block_size": 8192.0},
        {"block_size": True},
        {"seed": 1.5},
        {"sobol_shift_key": -1},
        {"sobol_shift_key": 2**64},
        {"scan_group_threshold": -1},
        {"scan_group_threshold": True},
        {"scan_group_threshold": 4.0},
    ],
)
def test_integer_settings_reject_other_types(settings):
    with pytest.raises(InvalidSetting):
        MonteCarloSettings(**settings)


def test_integer_settings_accept_numpy_integers():
    settings = MonteCarloSettings(
        block_size=np.int64(1024), seed=np.int32(7), scan_group_threshold=np.int64(0)
    )
    assert (
        type(settings.block_size) is int
        and type(settings.seed) is int
        and type(settings.scan_group_threshold) is int
    )


def test_unknown_smoothing_kernel_is_rejected():
    with pytest.raises(InvalidSetting, match="smoothing_kernel"):
        MonteCarloSettings(smoothing_kernel="cubic")


def test_script_parameters_cannot_shadow_model_labels(one_cpu):
    product = PathProduct(
        timeline=(1.0,), payoff=call_payoff, script_params={"STRIKE": 100.0, "spot": 1.0}
    )
    with pytest.raises(ReservedIdentifier):
        engine(product, devices=one_cpu)
    with pytest.raises(InvalidSetting):
        PathProduct(
            timeline=(1.0,), payoff=call_payoff, script_params=(("STRIKE", 100.0), ("STRIKE", 90.0))
        )


def test_bridge_requires_model_support(one_cpu):
    class TwoFactor(BlackScholes):
        n_factors = 2

        @property
        def supports_bb(self) -> bool:
            return False

    with pytest.raises(UnsupportedBrownianBridge):
        engine(model=TwoFactor(spot=SPOT, vol=VOL), devices=one_cpu, use_bb=True)
