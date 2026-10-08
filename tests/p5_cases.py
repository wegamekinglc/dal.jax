"""P5 contracts and native model constructors shared by CPU/GPU parity tests."""

from dal_jax import BlackScholes, CorrelatedBlackScholes, LocalVol, LocalVolSurface
from dal_jax.dates import Date

TODAY = Date.ymd(2022, 9, 15)
END = TODAY.add_days(365)


def case(name):
    bs = BlackScholes(spot=100.0, vol=0.15, rate=0.05, div=0.03)
    if name == "vector_asian":
        rows = (
            ["K", f"START: {TODAY} END: {END} FREQ: 1M", END],
            ["100", "APPEND(v,SPOT())", "pay PAYS MAX(AVERAGE(v)-K,0)"],
        )
    elif name == "vector_fuzzy":
        rows = (
            ["K", END],
            [
                "100",
                "IF SPOT()>K:4 THEN APPEND(v,SPOT()) APPEND(v,10) "
                "ELSE APPEND(v,2) END pay PAYS SUM(v)+AVERAGE(v)",
            ],
        )
    elif name == "dated_fix_payment":
        rows = (
            ["K", END],
            ["100", f"pay PAYS MAX(FIX(EQ[A],{TODAY.add_days(180)})-K,0) ON {TODAY.add_days(730)}"],
        )
    elif name == "payment_schedule":
        rows = _payment_rows()
    elif name.startswith("correlated"):
        return _correlated_case(name)
    elif name.startswith("localvol"):
        vols = (
            ((0.15, 0.15),) * 3
            if name == "localvol_flat"
            else ((0.18, 0.19), (0.15, 0.16), (0.17, 0.18))
        )
        surf = LocalVolSurface(spots=(70.0, 100.0, 140.0), times=(0.0, 1.0), vols=vols)
        bs = LocalVol(spot=100.0, index="EQ[A]", rate=0.05, div=0.03, surface=surf, max_step=0.25)
        rows = (["K", END], ["100", "pay PAYS MAX(FIX(EQ[A])-K,0)"])
    else:
        raise ValueError(name)
    return rows, bs


def _payment_rows():
    dates = [TODAY.add_days(30 * i) for i in range(1, 9)]
    return ["SCALE"] + dates, ["2"] + [
        f"pay PAYS SCALE*FIX(EQ[A]) ON {d.add_days(30)}" for d in dates
    ]


def _correlated_case(name):
    model = CorrelatedBlackScholes(
        indices=("EQ[A]", "EQ[B]"),
        spots=(100.0, 95.0),
        vols=(0.15, 0.2),
        divs=(0.03, 0.02),
        rate=0.05,
        correlations=((1.0, 0.4), (0.4, 1.0)),
    )
    observation = (
        "(FIX(EQ[A])+FIX(EQ[B]))/2" if name == "correlated_basket" else "MIN(FIX(EQ[A]),FIX(EQ[B]))"
    )
    rows = (
        ["K", TODAY.add_days(180), END],
        ["100", f"first={observation}", f"pay PAYS MAX((first+{observation})/2-K,0)"],
    )
    return rows, model


def native_model(dal, bs):
    if isinstance(bs, CorrelatedBlackScholes):
        return dal.CorrelatedBSModelData_New(
            bs.indices, bs.spots, bs.vols, bs.divs, bs.rate, dal.DoubleMatrix_(bs.correlations)
        )
    if isinstance(bs, LocalVol):
        surf = bs.surface
        native_surface = dal.LocalVolSurfaceData_New(
            surf.name, surf.spots, surf.times, dal.DoubleMatrix_(surf.vols)
        )
        base = dal.BSModelData_New(bs.spot, 0.15, bs.rate, bs.div)
        return dal.BSLocalVolModelData_New(
            bs.name, bs.index, bs.currency, bs.factor, base, native_surface, bs.max_step
        )
    return dal.BSModelData_New(bs.spot, bs.vol, bs.rate, bs.div)


def native_dates(dal, dates):
    return [dal.Date_(d.year, d.month, d.day) if isinstance(d, Date) else d for d in dates]


def compare(ours, theirs):
    import numpy as np

    assert ours.keys() == theirs.keys()
    for name in ours:
        np.testing.assert_allclose(
            ours[name], theirs[name], rtol=1e-10 if name == "PV" else 1e-8, atol=1e-10, err_msg=name
        )
