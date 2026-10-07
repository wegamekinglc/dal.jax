"""Single/multi-factor GSR: projection curves, bond/caplet/swap and rate exercise."""

import dal
import numpy as np

import dal_jax as dj
import dal_jax.api as api
from dal_jax.script.product import ScriptProductSettings
from _common import TODAY, arguments, compare, finish, require_p5_oracle, settings, table


def model_pair(multi):
    dates = [TODAY,TODAY.add_days(365),TODAY.add_days(365*8)]
    curve = api.GSRCurveData_New("curve",TODAY,"USD",dates,[0.,-.03,-.25],["3M"],[[0.,-.034,-.29]])
    native_curve = dal.GSRCurveData_New("curve",dal.Date_(TODAY.year,TODAY.month,TODAY.day),"USD",
                     [dal.Date_(d.year,d.month,d.day) for d in dates],[0.,-.03,-.25],["3M"],dal.DoubleMatrix_([[0.,-.034,-.29]]))
    if multi:
        vol = api.MultiFactorGSRVolData_New("vol",["level","slope"],[TODAY],[[.02],[.01]],[TODAY],[[1.],[-.4]],[[1.,.3],[.3,1.]])
        nv = dal.MultiFactorGSRVolData_New("vol",["level","slope"],[dal.Date_(TODAY.year,TODAY.month,TODAY.day)],dal.DoubleMatrix_([[.02],[.01]]),
                [dal.Date_(TODAY.year,TODAY.month,TODAY.day)],dal.DoubleMatrix_([[1.],[-.4]]),dal.DoubleMatrix_([[1.,.3],[.3,1.]]))
        native = dal.MultiFactorGSRModelData_New("rates",native_curve,nv)
    else:
        vol = api.GSRVolData_New("vol",[TODAY],[.02],[TODAY],[1.])
        nv = dal.GSRVolData_New("vol",[dal.Date_(TODAY.year,TODAY.month,TODAY.day)],[.02],[dal.Date_(TODAY.year,TODAY.month,TODAY.day)],[1.])
        native = dal.GSRModelData_New("rates",native_curve,nv)
    return api.GSRModelData_New("rates",curve,vol),native


def main():
    args = arguments(__doc__)
    require_p5_oracle()
    comparisons = []
    date = TODAY.add_days(365)
    maturity = TODAY.add_days(730)
    rows = (["K",date],[".03",f"pay PAYS FIX(IR[USD,DF,{maturity}]) + MAX(FIX(IR[USD,LIBOR_3M_CME])-K,0) + FIX(IR[USD,SWAP,5Y])"])
    for multi in (False,True):
        model,native = model_pair(multi)
        product = api.Product_New(*rows)
        prepared = dj.prepare(product,TODAY,model=model)
        engine = prepared.engine(model,settings(args,use_bb=True,block_size=1024))
        comparisons.append(compare("Two-factor GSR" if multi else "Single-factor GSR",engine,rows,args,bs=native))
    exercise_rows = ([TODAY.add_days(180),date],["EXERCISE MAX(FIX(IR[USD,LIBOR_3M_CME])-.03,0)"]*2)
    product_settings = ScriptProductSettings(regression_features=("IR[USD,LIBOR_3M_CME]",))
    product = api.Product_New(*exercise_rows,settings=product_settings)
    np_settings = dal.ScriptProductSettings_(regression_features=["IR[USD,LIBOR_3M_CME]"])
    nproduct = dal.Product_New([dal.Date_(d.year,d.month,d.day) for d in exercise_rows[0]],exercise_rows[1],settings=np_settings)
    engine = dj.prepare(product,TODAY,model=model).engine(model,settings(args,use_bb=True,block_size=1024,lsmc_training_paths=min(args.paths,4096)))
    comparisons.append(compare("Rate Bermudan with IR regressor",engine,exercise_rows,args,bs=native,product=nproduct))
    diagnostics = api.ScriptValuation_Explain(product,model)
    print("\nModel observation slots")
    table(["sample","date","indices"],[[d["sample_id"],diagnostics["sample_dates"][d["sample_id"]],d["index_names"]] for d in diagnostics["sample_definitions"]])
    finish(args,comparisons,preparation=diagnostics)


if __name__ == "__main__":
    main()
