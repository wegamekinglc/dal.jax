"""Event normalization, adjacent scans and numerical/graph-size invariants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from script_cases import EVALUATION as TODAY
from script_cases import QUARTERLY

from dal_jax import EvalContext, prepare
from dal_jax.api import Product_New
from dal_jax.models.base import Scenario
from dal_jax.script import ast as A
from dal_jax.script.passes.eventgroup import group_events


def daily_barrier(n):
    dates = [
        "BARRIER",
        "STRIKE",
        TODAY,
        f"START: {TODAY} END: {TODAY.add_days(n)} FREQ: 1CD",
        TODAY.add_days(n),
    ]
    events = [
        "100",
        "90",
        "alive=1",
        "IF SPOT()>BARRIER:0.25 THEN alive=0 END",
        "pay PAYS alive*(SPOT()-STRIKE)",
    ]
    return prepare(Product_New(dates, events), TODAY)


def scenario(spots):
    spots = jnp.asarray(spots)
    n = spots.size
    return Scenario(spots, jnp.ones(n), jnp.empty((n, 0)), jnp.empty((n, 0)))


def test_adjacent_runs_threshold_and_distinct_templates():
    product = daily_barrier(12)
    groups = product.event_groups(fuzzy=True)
    assert [(g.start, g.stop, g.scanned) for g in groups] == [
        (0, 1, False),
        (1, 12, True),
        (12, 13, False),
    ]
    assert not any(g.scanned for g in product.event_groups(fuzzy=True, threshold=0))
    assert not any(g.scanned for g in product.event_groups(fuzzy=True, threshold=12))
    repeated = (groups[1].template,) * 4
    broken = repeated + (groups[0].template,) + repeated
    assert [(g.size, g.scanned) for g in group_events(broken)] == [(4, True), (1, False), (4, True)]


def test_date_literals_and_folded_counters_become_event_constants():
    data = Product_New(
        ["SCALE", TODAY, QUARTERLY],
        ["2", "count=0", "count=count+1 pay PAYS SCALE*count*DCF(ACT365F,PeriodBegin,PeriodEnd)"],
    )
    product = prepare(data, TODAY)
    group = product.event_groups()[1]
    assert group.size == 4 and group.scanned
    assert len(set(group.constants)) == 4
    nodes = [node for statement in group.template for node in A.walk(statement)]
    assert any(isinstance(node, A.EventConst) for node in nodes)
    assert any(isinstance(node, A.ConstVar) for node in nodes)
    payoff = product.path_product().payoff
    path = scenario(jnp.ones(5))
    evaluate = lambda scale, threshold: payoff(
        {"script": {"SCALE": scale}}, path, EvalContext(scan_group_threshold=threshold)
    )
    a = jax.jit(jax.value_and_grad(lambda scale: evaluate(scale, 0)))(2.0)
    b = jax.jit(jax.value_and_grad(lambda scale: evaluate(scale, 4)))(2.0)
    expected = sum(
        (i + 1) * (end - start) / 365
        for i, (start, end) in enumerate(zip(product.event_dates, product.event_dates[1:]))
    )
    np.testing.assert_allclose(a, [2 * expected, expected], rtol=0, atol=1e-14)
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-14)


def test_source_locations_and_identifier_case_do_not_split_a_run():
    dates = [TODAY.add_days(i) for i in range(1, 5)]
    events = ["pay PAYS SPOT()*1", "PAY pays spot()*2", "Pay Pays SPOT()*3", "pay PAYS spot()*4"]
    product = prepare(Product_New(dates, events), TODAY)
    (group,) = product.event_groups()
    assert group.scanned and group.constants == ((1.0,), (2.0,), (3.0,), (4.0,))
    payoff = product.path_product().payoff
    assert (
        float(
            jax.jit(lambda s: payoff({"script": {}}, s, EvalContext()))(
                scenario([1.0, 2.0, 3.0, 4.0])
            )
        )
        == 30.0
    )


@pytest.mark.parametrize("aad", [False, True])
def test_delayed_payment_dates_share_template_but_keep_discount_slots(aad, cpu_devices):
    from p5_cases import case, compare

    from dal_jax import MonteCarloEngine, MonteCarloSettings

    rows, model = case("payment_schedule")
    product = prepare(Product_New(*rows), TODAY, model=model)
    (group,) = product.event_groups(fuzzy=aad)
    assert group.scanned and group.size == 8
    payments = [
        n for statement in group.template for n in A.walk(statement) if isinstance(n, A.Pays)
    ]
    assert payments[0].payment_date is None and payments[0].discount_id == 0
    # Slot ids carry runtime meaning even after payment dates are normalized.
    from dataclasses import replace

    changed = (replace(group.template[0], discount_id=1),)
    assert len(group_events((group.template, changed))) == 2
    values = []
    for threshold in (0, 4):
        settings = MonteCarloSettings(
            enable_aad=aad, devices=cpu_devices, block_size=128, scan_group_threshold=threshold
        )
        values.append(MonteCarloEngine(product.path_product(), model, settings).value(513))
    compare(values[1], values[0])


@pytest.mark.parametrize("fuzzy", [False, True])
@pytest.mark.parametrize("n", [36, 64])
def test_scan_and_unrolled_payoffs_and_gradients_agree(fuzzy, n):
    product = daily_barrier(n)
    spots = jnp.full(n + 1, 90.0).at[jnp.asarray([2, 3, 4])].set(99.9375).at[-1].set(99.0)
    payoff = product.path_product().payoff

    def evaluate(barrier, threshold):
        return payoff(
            {"script": {"BARRIER": barrier, "STRIKE": 90.0}},
            scenario(spots),
            EvalContext(fuzzy=fuzzy, scan_group_threshold=threshold),
        )

    a = jax.jit(jax.value_and_grad(lambda barrier: evaluate(barrier, 0)))(100.0)
    b = jax.jit(jax.value_and_grad(lambda barrier: evaluate(barrier, 4)))(100.0)
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-14)
    assert float(b[1]) > 0 if fuzzy else float(b[1]) == 0


def test_scan_graph_size_stays_constant_for_a_longer_schedule():
    sizes = []
    for n in (36, 365):
        payoff = daily_barrier(n).path_product().payoff
        f = lambda s: payoff(
            {"script": {"BARRIER": 100.0, "STRIKE": 90.0}}, s, EvalContext(fuzzy=True)
        )
        graph = jax.make_jaxpr(f)(scenario(jnp.ones(n + 1)))
        scans = [eq for eq in graph.jaxpr.eqns if eq.primitive.name == "scan"]
        assert len(scans) == 1 and scans[0].params["length"] == n - 1
        sizes.append((len(graph.jaxpr.eqns), len(scans[0].params["jaxpr"].jaxpr.eqns)))
    assert sizes[0] == sizes[1]
