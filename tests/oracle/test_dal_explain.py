"""Diagnostic schema, source locations, regression choices and uncertainty parity."""

import math
from dataclasses import replace

import numpy as np
import pytest

import dal_jax.api as api
from dal_jax import MonteCarloSettings
from dal_jax.script.fixings import FixingSnapshot, ValuationSettings
from dal_jax.script.product import ScriptProductSettings
from test_dal_rates import TODAY, native_date


def compare(ours,theirs,path=""):
    if isinstance(theirs,dict):
        assert ours.keys() == theirs.keys(),path
        for key in theirs:
            compare(ours[key],theirs[key],f"{path}.{key}")
    elif isinstance(theirs,list):
        assert len(ours) == len(theirs),path
        for i,(a,b) in enumerate(zip(ours,theirs)):
            compare(a,b,f"{path}[{i}]")
    elif isinstance(theirs,float):
        np.testing.assert_allclose(ours,theirs,rtol=1e-8,atol=1e-10,err_msg=path)
    else:
        assert ours == theirs,path


@pytest.mark.oracle
@pytest.mark.parametrize("case",["legacy","named","historical","delayed","expired","dead","features","feature_only"])
def test_preparation_explanation_matches_native_schema_and_sources(dal,case):
    if not hasattr(dal,"HybridModelData_New"):
        pytest.skip("full diagnostic parity requires pinned DAL source")
    dates = (TODAY.add_days(180),TODAY.add_days(365))
    events = ("x=SPOT()","pay PAYS x + SPOT()")
    default,features = "",()
    fixing = None
    if case == "named":
        events = ("x=FIX(EQ[A],2027-03-29)","pay PAYS x + SPOT()")
        default = "EQ[A]"
    elif case == "historical":
        dates = (TODAY.add_days(-1),)+dates
        events = ("x=FIX(EQ[A])","y=SPOT()","pay PAYS x+y+FIX(EQ[A])")
        default = "EQ[A]"
        fixing = {"EQ[A]":{dates[0]:98.}}
    elif case == "delayed":
        events = ("x=FIX(EQ[A])","pay PAYS x ON 2028-10-01")
    elif case == "expired":
        dates = (TODAY.add_days(-2),TODAY.add_days(-1))
        events = ("x=FIX(EQ[A])","pay PAYS x")
    elif case == "dead":
        events = ("dead=FIX(EQ[A],2027-03-29) EXERCISE 1","EXERCISE 2")
    elif case == "features":
        events = ("EXERCISE MAX(100-FIX(EQ[A]),0)",)*2
        features = ("EQ[A]",)
    elif case == "feature_only":
        events = ("EXERCISE 1", "EXERCISE 2")
        features = ("EQ[A]",)
    product = api.Product_New(dates,events,settings=ScriptProductSettings(default_index=default,regression_features=features))
    nproduct = dal.Product_New([native_date(dal,d) for d in dates],events,
                              settings=dal.ScriptProductSettings_(default_index=default,regression_features=list(features)))
    valuation = ValuationSettings(evaluation_date=TODAY,fixings=FixingSnapshot(fixing) if fixing is not None else None)
    nfix = dal.MarketFixingSnapshot_New({name:{dal.DateTime_(native_date(dal,d),0):v for d,v in history.items()} for name,history in fixing.items()}) if fixing else None
    nvaluation = dal.ScriptValuationSettings_(evaluation_date=native_date(dal,TODAY),fixings=nfix)
    ours = api.ScriptValuation_Explain(product,api.BSModelData_New(100.,.2,.03),valuation=valuation)
    theirs = dal.ScriptValuation_Explain(nproduct,dal.BSModelData_New(100.,.2,.03,0.),valuation=nvaluation)
    compare(ours,theirs)


@pytest.mark.oracle
@pytest.mark.parametrize("case",["plain","expired","put","zero","rqmc","variable"])
def test_simulation_diagnostics_match_native(dal,case):
    if not hasattr(dal,"HybridModelData_New"):
        pytest.skip("full diagnostic parity requires pinned DAL source")
    dates = (TODAY.add_days(180),TODAY.add_days(365))
    events = ("pay PAYS 1 EXERCISE MAX(100-SPOT(),0)",)*2
    settings = MonteCarloSettings(parallel="none",block_size=128,lsmc_training_paths=128,lsmc_validation_paths=64)
    features = ()
    if case == "plain":
        events = ("pay PAYS 1",)*2
    elif case == "expired":
        dates = (TODAY.add_days(-2),TODAY.add_days(-1))
        events = ("pay PAYS 1",)*2
    elif case == "zero":
        events = ("EXERCISE 0",)*2
    elif case == "rqmc":
        settings = replace(settings,lsmc_rqmc_replicates=3,lsmc_training_seed=17,lsmc_pricing_seed=29)
    elif case == "variable":
        events = ("x=SPOT() EXERCISE MAX(100-x,0)",)*2
        features = ("VAR[x]",)
    product = api.Product_New(dates,events,settings=ScriptProductSettings(regression_features=features))
    nproduct = dal.Product_New([native_date(dal,d) for d in dates],events,settings=dal.ScriptProductSettings_(regression_features=list(features)))
    valuation = ValuationSettings(evaluation_date=TODAY)
    nvaluation = dal.ScriptValuationSettings_(evaluation_date=native_date(dal,TODAY))
    ns = dal.MonteCarloSettings_(lsmc_training_paths=128,lsmc_validation_paths=64,
            **({"lsmc_rqmc_replicates":3,"lsmc_training_seed":17,"lsmc_pricing_seed":29} if case == "rqmc" else {}))
    ours = api.ScriptSimulation_Explain(product,api.BSModelData_New(100.,.2,.03),129,valuation=valuation,simulation=settings)
    theirs = dal.ScriptSimulation_Explain(nproduct,dal.BSModelData_New(100.,.2,.03,0.),129,valuation=nvaluation,simulation=ns)
    compare(ours,theirs)
