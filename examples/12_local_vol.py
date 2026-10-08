"""Flat/nonflat local-volatility surfaces, Euler subdivision and all bucket vegas."""

import dal
import jax
import numpy as np
import dal_jax as dj
from dal_jax.api import Product_New

from _common import TODAY, arguments, compare, finish, oracle_model, require_p5_oracle, settings, table


def main():
    args = arguments(__doc__)
    require_p5_oracle()
    rows = (["K", TODAY.add_days(365)], ["100", "pay PAYS MAX(FIX(EQ[A])-K,0)"])
    comparisons = []
    for label, vols in [("Flat surface", ((.15, .15),)*3),
                        ("Nonflat surface", ((.18, .19), (.15, .16), (.17, .18)))]:
        surf = dj.LocalVolSurface(name="surface", spots=(70., 100., 140.), times=(0., 1.), vols=vols)
        bs = dj.LocalVol(spot=100., index="EQ[A]", rate=.05, div=.03, surface=surf, max_step=.25)
        native_surf = dal.LocalVolSurfaceData_New("surface", surf.spots, surf.times, dal.DoubleMatrix_(vols))
        native = dal.BSLocalVolModelData_New("local_vol", "EQ[A]", "USD", "W_EQ", oracle_model(), native_surf, .25)
        product = dj.prepare(Product_New(*rows), TODAY, model=bs)
        engine = product.engine(bs, settings(args))
        comparisons.append(compare(label, engine, rows, args, bs=native))
    params = engine.default_params()
    price = jax.jit(lambda p: engine.pricer(args.paths)(p)[0])
    risks = jax.jit(jax.grad(price))(params)
    buckets = []
    for name in bs.param_labels:
        if not name.startswith("lvol:"):
            continue
        step = 1e-7
        up = params | {"model": params["model"] | {name: params["model"][name]+step}}
        down = params | {"model": params["model"] | {name: params["model"][name]-step}}
        fd = float((price(up)-price(down))/(2*step))
        adjoint = float(risks["model"][name])
        np.testing.assert_allclose(adjoint, fd, rtol=2e-6, atol=1e-7, err_msg=name)
        buckets.append([name, adjoint, fd])
    print("\nAll surface nodes: common-path differences")
    table(["bucket", "AAD", "finite difference"], buckets)
    print(f"\nVega matrix (spot rows, time columns):\n{np.asarray(bs.vol_grid(risks['model']))}")
    finish(args, comparisons, simulation_grid=engine.plan.grid, bucket_checks=buckets,
           vega_matrix=np.asarray(bs.vol_grid(risks["model"])).tolist())


if __name__ == "__main__":
    main()
