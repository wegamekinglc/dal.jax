"""Repeated event scans and cubic C1 smoothing; DAL uses an explicit equivalent polynomial."""

import numpy as np

import dal_jax as dj
from _common import arguments, barrier_rows, compare, finish, model, prepare, settings, table


def cubic_oracle_rows(width):
    dates, events = barrier_rows(width)
    half = width/2
    # Near-hard guards only select the cubic's interval. Their boundary slopes
    # are zero; no sampled spot in this example lies within their 1e-12 guards.
    hit = (f"x=SPOT()-BARRIER IF x<-{half}:0.000000000001 THEN hit=0 ELSE "
           f"IF x>{half}:0.000000000001 THEN hit=1 ELSE y=x/{width}+0.5 hit=y*y*(3-2*y) END END "
           "alive=alive*(1-hit)")
    return dates, [events[0], events[1], events[2], hit, hit+" call PAYS alive*MAX(SPOT()-STRIKE,0)"]


def main():
    args = arguments(__doc__)
    rows = barrier_rows(.25)
    prepared = prepare(rows)
    comparisons = []
    for threshold in (0, 4):
        engine = dj.MonteCarloEngine(prepared.path_product(), model(), settings(args, scan_group_threshold=threshold))
        comparisons.append(compare(f"DAL kernel, scan threshold {threshold}", engine, rows, args))
    for name in comparisons[0]["jax"]["result"]:
        np.testing.assert_allclose(comparisons[0]["jax"]["result"][name], comparisons[1]["jax"]["result"][name],
                                   rtol=1e-13, atol=1e-13)
    engine = dj.MonteCarloEngine(prepared.path_product(), model(), settings(args, smoothing_kernel="smoothstep"))
    comparisons.append(compare("C1 smoothstep vs DAL explicit cubic payoff", engine, cubic_oracle_rows(.25), args))
    groups = [[group.start, group.stop, group.size, group.scanned] for group in prepared.event_groups(fuzzy=True)]
    table(["event start", "event stop", "count", "scan"], groups)
    print("C1 changes the smoothing profile; its DAL reference implements the same cubic, not DAL's linear kernel.")
    finish(args, comparisons, event_groups=groups)


if __name__ == "__main__":
    main()
