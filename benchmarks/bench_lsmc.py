"""Bermudan phase timings on disjoint training and million-path pricing sets.

Run separate processes for CPU 1, CPU 4 and an actual GPU. The DAL CPU
reference includes training on every call. Frozen-policy JAX timings and
training-plus-pricing timings are both reported to make that difference clear.
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import dal
import jax
import numpy as np

import dal_jax as dj
import dal_jax.api as api
from dal_jax.dates import Date


def timed(function,repeat):
    start = time.perf_counter()
    output = jax.block_until_ready(function())
    first = time.perf_counter()-start
    runs = []
    for _ in range(repeat):
        start = time.perf_counter()
        output = jax.block_until_ready(function())
        runs.append(time.perf_counter()-start)
    return output,{"first_seconds":first,"warm_median_seconds":statistics.median(runs),"warm_runs_seconds":runs}


def results(output,greeks):
    if not greeks:
        return {"PV":float(output[0])}
    pv,risks = output
    return {"PV":float(pv),**{f"d_{name}":float(value) for group in risks.values() for name,value in group.items()}}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths",type=int,default=2**20)
    parser.add_argument("--training-paths",type=int,default=2**16)
    parser.add_argument("--devices",type=int,default=1)
    parser.add_argument("--platform",choices=("cpu","gpu"),default="cpu")
    parser.add_argument("--repeat",type=int,default=3)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if min(args.paths,args.training_paths,args.devices,args.repeat) < 1:
        parser.error("path counts, devices and repeat must be positive")
    return args


def measure(args,prepared,model,native_product,native_model,greeks):
    settings = dj.MonteCarloSettings(platform=args.platform,devices=dj.config.devices(args.platform)[:args.devices],
        block_size=8192,use_bb=True,enable_aad=greeks,lsmc_training_paths=args.training_paths)
    engine = prepared.engine(model,settings)
    params = engine.default_params()
    policy,training = timed(lambda:engine.train(args.paths,params),args.repeat)
    price = engine.pricer(args.paths,policy=policy)
    function = jax.value_and_grad(lambda p:price(p)[0]) if greeks else price
    start = time.perf_counter()
    executable = jax.jit(function).lower(params).compile()
    compilation = time.perf_counter()-start
    output,pricing = timed(lambda:executable(params),args.repeat)
    full,complete = timed(lambda:engine.value(args.paths,params),args.repeat)
    native_settings = dal.MonteCarloSettings_(use_bb=True,enable_aad=greeks,lsmc_training_paths=args.training_paths)
    native,reference = timed(lambda:dict(dal.MonteCarlo_ValueWithSettings(native_product,native_model,args.paths,simulation=native_settings)),args.repeat)
    computed = results(output,greeks)
    for name,value in computed.items():
        np.testing.assert_allclose(value,native[name],rtol=1e-6 if name == "PV" else 1e-8,atol=1e-10,err_msg=name)
        np.testing.assert_allclose(full[name],value,rtol=1e-12,atol=1e-12,err_msg=name)
    return {"mode":"greeks" if greeks else "price","jax":{"training":training,"compile_seconds":compilation,
            "frozen_pricing":pricing,"end_to_end":complete,"result":computed,"backend":engine.devices[0].platform,
            "device_kind":engine.devices[0].device_kind,"devices":len(engine.devices),"block_size":engine.layout(args.paths).block_size},
            "dal":reference|{"result":native}}


def main():
    args = arguments()
    dj.config.configure(num_cpu_devices=args.devices)
    today = Date.ymd(2022,9,15)
    dates = [today.add_days(days) for days in (180,365,545,730)]
    events = ["EXERCISE MAX(K-SPOT(),0)"]*4
    model = dj.BlackScholes(spot=100.,vol=.15,rate=.05,div=.03)
    prepared = dj.prepare(api.Product_New(["K"]+dates,["100"]+events),today,model=model)
    dal.EvaluationDate_Set(dal.Date_(today.year,today.month,today.day))
    native_product = dal.Product_New(["K"]+[dal.Date_(d.year,d.month,d.day) for d in dates],["100"]+events)
    native_model = dal.BSModelData_New(100.,.15,.05,.03)
    report = {"environment":{"python":platform.python_version(),"jax":jax.__version__,"os":platform.platform()},
              "configuration":vars(args)|{"output":str(args.output)},"measurements":[]}
    for greeks in (False,True):
        row = measure(args,prepared,model,native_product,native_model,greeks)
        report["measurements"].append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(row),flush=True)


if __name__ == "__main__":
    main()
