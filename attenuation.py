# -*- coding: utf-8 -*-
"""Attenuation experiment with known ground truth (Table 8 of the paper).

Two configurations are simulated with a known value of the attenuation rate lambda
returned by validity.proxy_test:

  complete proxy   X has no direct effect on Y   -> true lambda = 1
  independent      X has a direct effect on Y    -> true lambda = 0

For each replicate a fresh sample is drawn and lambda is recovered. The script
reports mean and standard deviation over replicates.

    cd code && python attenuation.py              # default: 1000 replicates
    cd code && python attenuation.py --reps 200

Output: printed table + results/attenuation.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from educeq import validity  # noqa: E402

N = 2000
SEED = 12345


def one_replicate(seed):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=N)
    X = 0.8 * Z + rng.normal(size=N)
    Y_proxy = 0.6 * Z + rng.normal(size=N)          # complete proxy: no direct effect
    Y_indep = 0.7 * X + rng.normal(size=N)          # independent effect
    a = validity.proxy_test(Y_proxy, X, Z=Z[:, None])
    b = validity.proxy_test(Y_indep, X, Z=Z[:, None])
    return a["lam"], b["lam"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()

    print("Attenuation experiment (Table 8)")
    print("n = %d per replicate, replicates R = %d, seed = %d" % (N, args.reps, SEED))
    print()
    lam_a, lam_b = [], []
    for r in range(args.reps):
        a, b = one_replicate(SEED + r)
        lam_a.append(a)
        lam_b.append(b)

    lam_a, lam_b = np.array(lam_a), np.array(lam_b)
    print("%-34s %-10s %-22s" % ("configuration", "true", "recovered (mean +- SD)"))
    print("%-34s %-10s %-22s" % ("complete proxy (no direct effect)", "1.0",
                                 "%.2f +- %.2f" % (lam_a.mean(), lam_a.std(ddof=1))))
    print("%-34s %-10s %-22s" % ("independent effect", "0.0",
                                 "%.2f +- %.2f" % (lam_b.mean(), lam_b.std(ddof=1))))

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "attenuation.csv")
    with open(out, "w", encoding="utf8") as f:
        f.write("configuration,true,mean,sd,replicates,seed\n")
        f.write("complete_proxy,1.0,%.4f,%.4f,%d,%d\n"
                % (lam_a.mean(), lam_a.std(ddof=1), args.reps, SEED))
        f.write("independent,0.0,%.4f,%.4f,%d,%d\n"
                % (lam_b.mean(), lam_b.std(ddof=1), args.reps, SEED))
    print("\nwritten: %s" % out)


if __name__ == "__main__":
    main()
