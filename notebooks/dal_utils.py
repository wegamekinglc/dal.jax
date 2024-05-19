def print_result(result, names=("rate", "div", "vol", "spot", "STRIKE")):
    res = dict()
    if not isinstance(result, tuple):
        res["PV"] = float(result)
    else:
        res["PV"] = float(result[0])
        for n, v in zip(names, result[1]):
            res["d_" + n] = float(v)
    return res