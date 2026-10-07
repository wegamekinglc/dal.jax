"""Prepared scalar scripts: named parameter risks, historical replay, DCF and schedules."""

import jax
import jax.numpy as jnp
import numpy as np

import dal_jax as dj
from _common import MATURITY, TODAY, arguments, compare, finish, model, prepare, settings, table


def delayed_call(params, scenario, _ctx):
    value = jnp.maximum(scenario.spot[-1]-params["script"]["STRIKE"], 0.)
    return value*scenario.discounts[-1, 0]/scenario.numeraire[-1]


def delayed_comparison(args):
    # The published DAL wheel lacks PAYS ON. Record the payoff at exercise time
    # and pay it later: the same contract with one extra DAL simulation date.
    delay = 182/365
    product = dj.PathProduct(timeline=(3.,), payoff=delayed_call,
                             sample_defs=(dj.SampleDef(numeraire=True, discount_mats=(3.+delay,)),),
                             script_params={"STRIKE": 120.})
    engine = dj.MonteCarloEngine(product, model(), settings(args))
    rows = (["STRIKE", MATURITY, MATURITY.add_days(182)], ["120", "x=MAX(SPOT()-STRIKE,0)", "pay PAYS x"])
    return compare("Delayed native payment vs DAL stored payoff paid later", engine, rows, args)


def main():
    args = arguments(__doc__)
    historical = (["SCALE", TODAY.add_days(-10), MATURITY],
                  ["2", "x=SCALE x PAYS 100", "pay PAYS x*SPOT()/100"])
    monthly = f"START: {TODAY} END: {MATURITY} FREQ: 1M"
    asian = (["STRIKE", TODAY, monthly, MATURITY],
             ["100", "total=0 count=0", "total=total+SPOT() count=count+1", "call PAYS MAX(total/count-STRIKE,0)"])
    coupon = (["COUPON", monthly], ["5", "pay PAYS COUPON*DCF(ACT365F,PeriodBegin,PeriodEnd)"])
    comparisons = []
    for label, rows in (("Historical SCALE assignment; past PAYS discarded", historical),
                        ("Scalar monthly Asian", asian), ("DCF coupon schedule", coupon)):
        engine = dj.MonteCarloEngine(prepare(rows).path_product(), model(), settings(args))
        comparisons.append(compare(label, engine, rows, args))
    product = prepare(historical)
    engine = dj.MonteCarloEngine(product.path_product(), model(), settings(args))
    params = engine.default_params()
    def value(scale):
        return engine.pricer(args.paths)(params | {"script": {"SCALE": scale}})[0]
    value = jax.jit(value)
    gradient = float(jax.grad(value)(2.))
    difference = float((value(2.+1e-5)-value(2.-1e-5))/(2e-5))
    np.testing.assert_allclose(gradient, difference, rtol=1e-8, atol=1e-10)
    table(["history parameter", "JAX grad", "common-path FD"], [["SCALE", gradient, difference]])
    print("Prepared initial values:", product.initial_values)
    comparisons.append(delayed_comparison(args))
    finish(args, comparisons, historical_scale_gradient=gradient, finite_difference=difference)


if __name__ == "__main__":
    main()
