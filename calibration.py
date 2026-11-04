# -*- coding: utf-8 -*-
"""Calibration of the count threshold (Table 7 of the paper).

For each object count N, independent two-category patterns are drawn in a square
domain and the second-order term is estimated. The script reports the median
neighbourhood count kappa_B and the Monte Carlo standard deviation s of the
second-order estimator across replicates, in per cent.

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

BANDWIDTH = 7.0
DOMAIN = 200.0
COUNTS = [4000, 5000, 6000, 7000, 7800, 9000]
SEED = 2024


def one_replicate(n, seed):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, DOMAIN, size=(n, 2))
    types = rng.integers(0, 2, n)
    W = kernels.distance(coords, bandwidth=BANDWIDTH)
    res = estimator.decompose(W, coords, types, bandwidth=BANDWIDTH,
                              grid_n=40, check=False)
    c2 = float(res["C2"][0, 1])
    kappa = float(np.nanmedian(diagnostics.neighbourhood_count(W, types)[0, 1]))
    return kappa, c2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    print("Calibration of the count threshold (Table 7)")
    print("bandwidth b = %.1f km, domain = %.0f x %.0f km, replicates R = %d, seed = %d"
          % (BANDWIDTH, DOMAIN, DOMAIN, args.reps, SEED))
    print()
    print("%-12s %-10s %-10s %-10s" % ("N", "kappa_B", "SD s (%)", "replicates"))

    rows = []
    for n in COUNTS:
        kap, c2 = [], []
        for r in range(args.reps):
            k, v = one_replicate(n, SEED + 1000 * n + r)
            kap.append(k)
            c2.append(v)
        kap_m = float(np.median(kap))
        sd = float(np.std(c2, ddof=1)) * 100.0
        rows.append((n, kap_m, sd))
        print("%-12d %-10.1f %-10.2f %-10d" % (n, kap_m, sd, args.reps))

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "calibration.csv")
    with open(out, "w", encoding="utf8") as f:
        f.write("N,kappa_B,sd_percent,replicates\n")
        for n, k, s in rows:
            f.write("%d,%.2f,%.3f,%d\n" % (n, k, s, args.reps))
    print("\nwritten: %s" % out)


if __name__ == "__main__":
    main()
