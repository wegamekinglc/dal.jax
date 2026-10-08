"""Parity with dal-python on identical Sobol points (issue #1, P0 acceptance).

DAL builds products from event tables; the timelines below are taken from
``ScriptValuation_Explain`` so the hand-written JAX payoffs see exactly DAL's
sample times.  Tolerances follow the issue: PV relative error <= 1e-10, Greeks
<= 1e-8 (absolute for Greeks near zero).
"""

import json
import math

import jax
import numpy as np
import pytest
from support import (
    BARRIER,
    DIV,
    MATURITY,
    RATE,
    SPOT,
    STRIKE,
    VOL,
    bs_model,
    european_call,
    up_and_out_call,
)

from dal_jax import MonteCarloEngine, MonteCarloSettings
from dal_jax.random.sobol import Sobol

pytestmark = pytest.mark.oracle

PV_RTOL = 1e-10
GREEK_RTOL = 1e-8
GREEK_ATOL = 1e-10


def as_array(matrix) -> np.ndarray:
    return np.asarray(matrix.to_rows())


def explain_timeline(dal, product, model) -> tuple[float, ...]:
    explained = dal.ScriptValuation_Explain(product, model)
    explained = json.loads(explained) if isinstance(explained, str) else explained
    return tuple(float(t) for t in explained["timeline"])


def assert_parity(ours: dict, theirs: dict) -> None:
    assert set(ours) == set(theirs)
    np.testing.assert_allclose(ours["PV"], theirs["PV"], rtol=PV_RTOL)
    for key in ours.keys() - {"PV"}:
        np.testing.assert_allclose(
            ours[key], theirs[key], rtol=GREEK_RTOL, atol=GREEK_ATOL, err_msg=key
        )


@pytest.mark.parametrize("dim,start", [(1, 0), (3, 5), (37, 1023), (500, 2**20 - 9), (21200, 7)])
def test_sobol_uniforms_and_normals(dal, dim, start):
    seq = Sobol(dim=dim)
    ids = jax.numpy.arange(start, start + 8)
    np.testing.assert_array_equal(
        np.asarray(jax.vmap(seq.uniform)(ids)),
        as_array(dal.SobolRSG_Get_Uniform(dal.SobolRSG_New(start, dim), 8)),
    )
    np.testing.assert_allclose(
        np.asarray(jax.vmap(seq.normal)(ids)),
        as_array(dal.SobolRSG_Get_Normal(dal.SobolRSG_New(start, dim), 8)),
        rtol=1e-14,
        atol=1e-15,
    )


@pytest.mark.parametrize("precise,polish", [(False, True), (True, True)])
def test_sobol_polished_normals(dal, precise, polish):
    seq = Sobol(dim=16, precise=precise, polish=polish)
    ids = jax.numpy.arange(0, 64)
    theirs = as_array(dal.SobolRSG_Get_Normal(dal.SobolRSG_New(0, 16, precise, polish), 64))
    np.testing.assert_allclose(
        np.asarray(jax.vmap(seq.normal)(ids)), theirs, rtol=1e-13, atol=1e-14
    )


@pytest.fixture(scope="module")
def dal_european(dal):
    today = dal.EvaluationDate_Get()
    product = dal.Product_New(
        ["STRIKE", today.AddDays(int(365 * MATURITY))],
        [f"{STRIKE}", "call pays MAX(spot() - STRIKE, 0.0)"],
    )
    return product, dal.BSModelData_New(SPOT, VOL, RATE, DIV)


@pytest.mark.parametrize("enable_aad", [False, True])
def test_european_pv_and_five_greeks(dal, dal_european, cpu_devices, enable_aad):
    product, model = dal_european
    assert explain_timeline(dal, product, model) == (MATURITY,)
    n = 2**18
    ours = MonteCarloEngine(
        european_call(), bs_model(), MonteCarloSettings(enable_aad=enable_aad, devices=cpu_devices)
    ).value(n)
    assert_parity(ours, dict(dal.MonteCarlo_Value(product, model, n, "sobol", False, enable_aad)))


@pytest.fixture(scope="module")
def dal_barrier(dal):
    today = dal.EvaluationDate_Get()
    maturity = today.AddDays(int(365 * MATURITY))
    dates = ["STRIKE", "BARRIER", today, f"START: {today} END: {maturity} FREQ: 1M", maturity]
    events = [
        f"{STRIKE:.2f}",
        f"{BARRIER:.2f}",
        "alive = 1",
        "if spot() >= BARRIER:0.1 then alive = 0 end",
        "if spot() >= BARRIER:0.1 then alive = 0 end\ncall pays alive * MAX(spot() - STRIKE, 0.0)",
    ]
    product = dal.Product_New(dates, events)
    model = dal.BSModelData_New(SPOT, VOL, RATE, DIV)
    timeline = explain_timeline(dal, product, model)
    #  The schedule's last date is the maturity, so the barrier is tested twice there.
    last = len(timeline) - 1
    return product, model, up_and_out_call(timeline, tuple(range(1, last + 1)) + (last,))


@pytest.mark.parametrize("use_bb", [False, True])
@pytest.mark.parametrize("enable_aad", [False, True])
def test_monthly_barrier_exact_and_fuzzy(dal, dal_barrier, cpu_devices, use_bb, enable_aad):
    product, model, ours_product = dal_barrier
    n = 2**16
    settings = MonteCarloSettings(enable_aad=enable_aad, use_bb=use_bb, devices=cpu_devices)
    ours = MonteCarloEngine(ours_product, bs_model(), settings).value(n)
    assert_parity(ours, dict(dal.MonteCarlo_Value(product, model, n, "sobol", use_bb, enable_aad)))


def test_barrier_greeks_match_dal_reference(dal, dal_barrier, cpu_devices):
    """DAL's AAD Greeks at 2**20 paths; a hard-condition ``jax.grad`` would give d_BARRIER = 0."""
    _, _, ours_product = dal_barrier
    ours = MonteCarloEngine(
        ours_product, bs_model(), MonteCarloSettings(enable_aad=True, devices=cpu_devices)
    ).value(2**20)
    assert ours["d_BARRIER"] == pytest.approx(0.08925165480978217, rel=1e-6)
    assert ours["d_vol"] == pytest.approx(-7.227015534889115, rel=1e-6)


@pytest.mark.parametrize("rsg", ["mrg32", "irn"])
def test_pseudo_random_streams_agree_statistically(dal, dal_european, cpu_devices, rsg):
    """Pseudo-random streams are not bit-compatible, so compare within 3 standard errors.

    The reference is DAL's converged Sobol price: DAL's own mrg32 / irn results
    drift by several standard errors even at 2**20 paths (irn ~5.28 against a
    converged 5.2018), so they make a poor statistical yardstick.
    """
    product, model = dal_european
    reference = dict(dal.MonteCarlo_Value(product, model, 2**20, "sobol", False, False))["PV"]
    n = 2**16
    eng = MonteCarloEngine(
        european_call(), bs_model(), MonteCarloSettings(rsg=rsg, devices=cpu_devices)
    )
    params = eng.default_params()
    values = np.concatenate(
        [np.asarray(eng.path_payoffs(params, b, n)[1][:, 0]) for b in range(eng.layout(n).n_blocks)]
    )[:n]
    se = values.std() / math.sqrt(n)
    assert abs(values.mean() - reference) < 3.0 * se
    np.testing.assert_allclose(eng.value(n)["PV"], values.mean(), rtol=1e-12)
