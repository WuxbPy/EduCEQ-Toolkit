# -*- coding: utf-8 -*-
"""Calibration of the effective-size threshold (Table 10 of the paper).

For each object count N, independent two-category patterns are drawn in a square
domain and the second-order term is estimated. The script reports three support
measures and the Monte Carlo standard deviation s of the second-order estimator
across replicates, in per cent.

Version 1.1.0 reports the effective size alongside the raw count, because the
threshold is applied to the former (Section 4.2). The last column, s * sqrt(e),
should be roughly constant: it is the coefficient of the 1/sqrt(e) law on which
the choice of 30 rests.

    cd code && python calibration.py            # default: R = 200 replicates
    cd code && python calibration.py --reps 500

Output: printed table + results/calibration.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from educeq import kernels, estimator, diagnostics  # noqa: E402

BANDWIDTH = 5.0
DOMAIN = 200.0
COUNTS = [4000, 5000, 6000, 7000, 7800, 9000]
SEED = 2024


def one_replicate(n, seed):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, DOMAIN, size=(n, 2))
    types = rng.integers(0, 2, n)
    W = kernels.distance(coords, bandwidth=BANDWIDTH)
    res = estimator._decompose_unchecked(W, coords, types, bandwidth=BANDWIDTH,
                                         grid_n=40)
    c2 = float(res["C2"][0, 1])
    s = diagnostics.support(W, types)
    return (s["raw"][0, 1], s["weighted"][0, 1], s["ess"][0, 1], c2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    print("Calibration of the effective-size threshold (Table 10)")
    print("bandwidth b = %.1f km, domain = %.0f x %.0f km, replicates R = %d, seed = %d"
          % (BANDWIDTH, DOMAIN, DOMAIN, args.reps, SEED))
    print()
    print("%-8s %-9s %-9s %-9s %-10s %-10s" %
          ("N", "n_B", "S_B", "e_B", "sd s (%)", "s*sqrt(e)"))

    rows = []
    for n in COUNTS:
        raw, sw, es, c2 = [], [], [], []
        for r in range(args.reps):
            a, b, c, v = one_replicate(n, SEED + 1000 * n + r)
            raw.append(a); sw.append(b); es.append(c); c2.append(v)
        raw_m, sw_m, es_m = (float(np.median(raw)), float(np.median(sw)),
                             float(np.median(es)))
        sd = float(np.std(c2, ddof=1)) * 100.0
        rows.append((n, raw_m, sw_m, es_m, sd, sd / 100.0 * np.sqrt(es_m)))
        print("%-8d %-9.1f %-9.1f %-9.1f %-10.2f %-10.4f"
              % (n, raw_m, sw_m, es_m, sd, sd / 100.0 * np.sqrt(es_m)))

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "calibration.csv")
    with open(out, "w", encoding="utf8") as f:
        f.write("N,n_B,S_B,e_B,sd_percent,s_times_sqrt_e,replicates\n")
        for n, r_, s_, e_, sd, k in rows:
            f.write("%d,%.1f,%.1f,%.1f,%.3f,%.4f,%d\n"
                    % (n, r_, s_, e_, sd, k, args.reps))
    print("\nwritten: %s" % out)


if __name__ == "__main__":
    main()
