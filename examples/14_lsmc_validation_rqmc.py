"""Held-out degree selection, independent Sobol replicas and retrained-policy risks."""

from dataclasses import replace

import dal
import numpy as np

import dal_jax as dj
from dal_jax.api import Product_New, ScriptSimulation_Explain
from _common import TODAY, arguments, compare, finish, model, oracle_product, oracle_model, require_p5_oracle, settings, table, timed


def main():
    args = arguments(__doc__)
    require_p5_oracle()
    rows = ([TODAY.add_days(365),TODAY.add_days(545)],["EXERCISE MAX(100-SPOT(),0)"]*2)
    bs = model()
    data = Product_New(*rows)
    prepared = dj.prepare(data,TODAY,model=bs)
    simulation = settings(args,lsmc_training_paths=min(args.paths,4096),lsmc_validation_paths=min(args.paths,1024),
                          lsmc_rqmc_replicates=3,lsmc_training_seed=17,lsmc_pricing_seed=29,use_bb=True,block_size=1024)
    engine = prepared.engine(bs,simulation)
    result = compare("Validated RQMC frozen policy",engine,rows,args)
    diagnostic = ScriptSimulation_Explain(data,bs,args.paths,simulation=replace(simulation,enable_aad=False))
    native = dal.ScriptSimulation_Explain(oracle_product(rows),oracle_model(),args.paths,
        simulation=dal.MonteCarloSettings_(use_bb=True,lsmc_training_paths=simulation.lsmc_training_paths,
            lsmc_validation_paths=simulation.lsmc_validation_paths,lsmc_rqmc_replicates=3,lsmc_training_seed=17,lsmc_pricing_seed=29))
    np.testing.assert_allclose(diagnostic["uncertainty"]["replicate_means"],native["uncertainty"]["replicate_means"],rtol=1e-10,atol=1e-10)
    table(["replicate","JAX PV","DAL PV"],[[i,a,b] for i,(a,b) in enumerate(zip(diagnostic["uncertainty"]["replicate_means"],native["uncertainty"]["replicate_means"]))])
    small_paths = min(args.paths,4096)
    retrained_settings = replace(simulation,lsmc_training_paths=min(args.paths,1024),lsmc_validation_paths=None,
                                lsmc_rqmc_replicates=None,lsmc_training_seed=None,lsmc_pricing_seed=None,lsmc_policy_risk_mode="RetrainedBump")
    retrained = prepared.engine(bs,retrained_settings)
    ours,our_times = timed(lambda:retrained.value(small_paths),args.repeat)
    ns = dal.MonteCarloSettings_(enable_aad=True,use_bb=True,lsmc_training_paths=retrained_settings.lsmc_training_paths,lsmc_policy_risk_mode="RetrainedBump")
    theirs,their_times = timed(lambda:dict(dal.MonteCarlo_ValueWithSettings(oracle_product(rows),oracle_model(),small_paths,simulation=ns)),args.repeat)
    for name in ours:
        np.testing.assert_allclose(ours[name],theirs[name],rtol=1e-8,atol=1e-9,err_msg=name)
    table(["risk","JAX retrained","DAL retrained"],[[name,value,theirs[name]] for name,value in ours.items()])
    finish(args,[result],uncertainty=diagnostic["uncertainty"],exercise=diagnostic["exercise_events"],
           retrained={"paths":small_paths,"jax":our_times|{"result":ours},"dal":their_times|{"result":theirs}})


if __name__ == "__main__":
    main()
