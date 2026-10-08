import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dal_jax import BlackScholes, SampleDef
from dal_jax.errors import InvalidModelParameter, InvalidModelTimeline, UnsupportedModelObservation
from dal_jax.models import Model


def make(**kw) -> BlackScholes:
    return BlackScholes(**({"spot": 100.0, "vol": 0.2, "rate": 0.03, "div": 0.01} | kw))


def defs(n: int, **kw) -> tuple[SampleDef, ...]:
    return tuple(SampleDef(**kw) for _ in range(n))


def test_satisfies_protocol_and_labels():
    model = make()
    assert isinstance(model, Model)
    assert model.param_labels == ("spot", "vol", "rate", "div")
    assert (
        model.supports_bb and model.numeraire_is_deterministic and model.supports_discount_factors
    )


def test_allocate_builds_steps_from_positive_times():
    model = make()
    plan = model.allocate([0.0, 0.5, 1.25], defs(3))
    assert plan.today_on_timeline
    assert plan.dts == (0.5, 0.75)
    assert model.sim_dim(plan) == 2
    plan = model.allocate([0.5, 1.25], defs(2))
    assert not plan.today_on_timeline and plan.dts == (0.5, 0.75)


def test_generate_matches_dal_recurrence():
    model = make()
    times = [0.0, 0.5, 1.25, 2.0]
    plan = model.allocate(times, defs(4, index_names=("EQ[spot]",), discount_mats=(3.0,)))
    params = model.default_params()
    state = model.init(params, plan)
    z = jnp.asarray([0.3, -1.2, 0.7])
    scenario = model.generate(state, plan, z)

    log_spot, spots = math.log(100.0), [100.0]
    for dt, g in zip((0.5, 0.75, 0.75), (0.3, -1.2, 0.7)):
        log_spot += (0.03 - 0.01 - 0.5 * 0.2 * 0.2) * dt + 0.2 * math.sqrt(dt) * g
        spots.append(math.exp(log_spot))
    np.testing.assert_allclose(np.asarray(scenario.spot), spots, rtol=1e-15)
    np.testing.assert_allclose(
        np.asarray(scenario.numeraire), [math.exp(0.03 * t) for t in times], rtol=1e-15
    )
    np.testing.assert_allclose(
        np.asarray(scenario.discounts[:, 0]),
        [math.exp(-0.03 * (3.0 - t)) for t in times],
        rtol=1e-15,
    )
    np.testing.assert_array_equal(
        np.asarray(scenario.observations[:, 0]), np.asarray(scenario.spot)
    )
    samples = scenario.samples()
    assert len(samples) == 4 and float(samples[2].spot) == float(scenario.spot[2])


def test_padding_and_numeraire_flags():
    model = make()
    sample_defs = (SampleDef(numeraire=False), SampleDef(discount_mats=(2.0, 3.0)))
    plan = model.allocate([1.0, 2.0], sample_defs)
    scenario = model.generate(model.init(model.default_params(), plan), plan, jnp.zeros(2))
    assert float(scenario.numeraire[0]) == 1.0
    assert scenario.discounts.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(scenario.discounts[0]), [1.0, 1.0])
    assert scenario.observations.shape == (2, 0)


def test_deterministic_path_without_volatility():
    model = make(vol=0.0)
    plan = model.allocate([1.0, 2.0], defs(2))
    scenario = model.generate(
        model.init(model.default_params(), plan), plan, jnp.asarray([5.0, -5.0])
    )
    np.testing.assert_allclose(
        np.asarray(scenario.spot), [100.0 * math.exp(0.02), 100.0 * math.exp(0.04)], rtol=1e-14
    )


def test_today_only_timeline_has_no_random_dimension():
    model = make()
    plan = model.allocate([0.0], defs(1))
    assert model.sim_dim(plan) == 0
    scenario = model.generate(model.init(model.default_params(), plan), plan, jnp.zeros(0))
    assert float(scenario.spot[0]) == 100.0


def test_init_is_differentiable():
    model = make()
    plan = model.allocate([1.0], defs(1))

    def terminal(params):
        return model.generate(model.init(params, plan), plan, jnp.asarray([0.5])).spot[-1]

    grads = jax.grad(terminal)(model.default_params())
    s_t = float(terminal(model.default_params()))
    np.testing.assert_allclose(float(grads["spot"]), s_t / 100.0, rtol=1e-14)
    np.testing.assert_allclose(float(grads["vol"]), s_t * (-0.2 + 0.5), rtol=1e-13)


@pytest.mark.parametrize(
    "timeline,sample_defs",
    [
        ([], ()),
        ([1.0, 0.5], defs(2)),
        ([-0.1, 1.0], defs(2)),
        ([1.0, 1.0], defs(2)),
        ([1.0], defs(2)),
        ([1.0], (SampleDef(discount_mats=(0.5,)),)),
        ([1.0, float("nan")], defs(2)),
    ],
)
def test_invalid_timelines(timeline, sample_defs):
    with pytest.raises(InvalidModelTimeline):
        make().allocate(timeline, sample_defs)


def test_observation_budget():
    with pytest.raises(UnsupportedModelObservation):
        make().allocate([1.0, 2.0], (SampleDef(index_names=("A",)), SampleDef(index_names=("B",))))
    with pytest.raises(UnsupportedModelObservation):
        make().allocate([1.0], (SampleDef(index_names=("A", "A")),))


@pytest.mark.parametrize(
    "kw",
    [
        {"spot": 0.0},
        {"spot": float("inf")},
        {"vol": -0.1},
        {"rate": float("nan")},
        {"div": float("inf")},
    ],
)
def test_invalid_parameters(kw):
    with pytest.raises(InvalidModelParameter):
        make(**kw)
