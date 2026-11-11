# -*- coding: utf-8 -*-
"""Attenuation with analytic population values (Table 13 of the paper).

Why this script exists
----------------------
Two attenuation rates are easy to misread: 0.95 against a true value of 1, and
0.14 against a true value of 0. The first is a finite-sample bias, the second is
an error in the labelling of the design — for the configuration actually
simulated the population attenuation is 0.1409, so the recovered 0.141 was
correct. This script derives each analytic value in closed form before simulating
anything, and adds the design whose attenuation is exactly zero.

Designs (all linear Gaussian, unit error variances, n = 2,000)
--------------------------------------------------------------
A  complete proxy      Z ~ N(0,1), X = aZ + e_x, Y = bZ + e_y
                       X carries no direct effect, so r_net = 0 and lambda = 1
B  independent effect  X = aZ + e_x, Y = cX + e_y
                       sigma_X^2 = a^2 + 1, cov(X,Y) = c sigma_X^2
                       r_base = c sigma_X / sigma_Y,  r_net = c / sqrt(c^2 + 1)
                       lambda = 1 - r_net / r_base
                       with a = 0.8, c = 0.7:  lambda = 0.1409
C  X orthogonal to Z   X ~ N(0,1) independent of Z, Y = cX + e_y
                       r_base = r_net, so lambda = 0 exactly

The finite-sample bias of design A
----------------------------------
lambda is defined on absolute correlations, lambda = 1 - |r_net| / |r_base|.
When the population value of r_net is zero, E|r_net| > 0, so lambda is biased
downwards by approximately (2/pi)^0.5 * sigma_r / |r_base|. With sigma_r = 0.0218
(measured over the replicates) and the population |r_base| = 0.3214 this predicts
0.9459 against a recovered 0.9458, a gap of 0.0001 on a Monte Carlo standard
error of 0.0013; the script prints the prediction next to the estimate. The prediction
uses the population r_base, never a single sample draw, otherwise the comparison
mixes approximation error with sampling error.

    cd code && python attenuation.py                # default 1,000 replicates
    cd code && python attenuation.py --reps 200

Output: printed table + results/attenuation.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                    # noqa: E402

from educeq import validity                           # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(ROOT), "results", "attenuation.csv")

A, B, C = 0.8, 0.6, 0.7        # X = aZ + ex ; Y_proxy = bZ + ey ; Y_indep = cX + ey
N, SEED = 2000, 12345


def analytic():
    """Population attenuation for the three designs, in closed form.

    r_base is the population correlation of Y on X with no controls, which is
    what proxy_test reports when C is None. It differs by design: in design A
    both Y and X are driven by Z, so r_base = b a / sqrt((a^2+1)(b^2+1)); in
    designs B and C, Y = c X + e_y, so r_base = c sigma_X / sigma_Y.
    """
    lam_proxy = 1.0                                  # r_net = 0 by construction
    sx2 = A * A + 1.0                                # var(X) = a^2 + 1
    sy2 = B * B + 1.0                                # var(Y) = b^2 + 1 in design A
    r_base_proxy = A * B / np.sqrt(sx2 * sy2)        # corr(X, Y) in design A
    sy2b = C * C * sx2 + 1.0                         # var(Y) = c^2 var(X) + 1
    r_base_indep = C * sx2 / np.sqrt(sx2 * sy2b)
    r_net_indep = C / np.sqrt(C * C + 1.0)
    lam_indep = 1.0 - r_net_indep / r_base_indep
    return (("complete proxy", lam_proxy, r_base_proxy),
            ("independent effect", lam_indep, r_base_indep),
            ("X orthogonal to Z", 0.0, r_net_indep))


def simulate(design, seed, n=N):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    if design == "X orthogonal to Z":
        X = rng.normal(size=n)
    else:
        X = A * Z + rng.normal(size=n)
    if design == "complete proxy":
        Y = B * Z + rng.normal(size=n)
    else:
        Y = C * X + rng.normal(size=n)
    out = validity.proxy_test(Y, X, Z=Z[:, None])
    return out["lam"], out["r_base"], out["r_net"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()

    print("Attenuation with analytic population values (Table 13)")
    print("n = %d per replicate, R = %d, seed = %d\n" % (N, args.reps, SEED))
    head = "%-22s %9s %10s %10s %10s %9s" % (
        "design", "analytic", "recovered", "sd", "bias", "r_base")
    print(head)
    print("-" * len(head))

    rows, nets = [], {}
    for name, lam, r_base_theory in analytic():
        res = [simulate(name, SEED + r) for r in range(args.reps)]
        vals = np.array([r[0] for r in res])
        nets[name] = np.array([r[2] for r in res])
        mean, sd = vals.mean(), vals.std(ddof=1)
        print("%-22s %9.4f %10.4f %10.4f %10.4f %9.3f"
              % (name, lam, mean, sd, mean - lam, r_base_theory))
        rows.append((name, lam, mean, sd, mean - lam, args.reps, SEED, r_base_theory))

    r_base_A = analytic()[0][2]
    sigma_r = nets["complete proxy"].std(ddof=1)
    pred = 1.0 - np.sqrt(2.0 / np.pi) * sigma_r / r_base_A
    print("\nComplete proxy: bias of a rate defined on absolute correlations")
    print("  sigma(r_net) = %.4f (empirical over R replicates)" % sigma_r)
    print("  |r_base|     = %.4f (population value, not a single draw)" % r_base_A)
    print("  predicted E[lambda] = 1 - (2/pi)^0.5 * sigma / |r_base| = %.4f" % pred)
    print("  recovered = %.4f ; gap = %.4f against a Monte Carlo s.e. of %.4f,"
          % (rows[0][2], rows[0][2] - pred, rows[0][3] / np.sqrt(args.reps)))
    print("  so the departure from 1.0000 is definitional, not simulation noise.")
    print("  Section 5.6 of the paper reports it rather than hiding it.")

    with open(OUT, "w", encoding="utf8") as fh:
        fh.write("design,analytic,recovered_mean,recovered_sd,bias,replicates,seed,n,r_base_pop\n")
        for r in rows:
            fh.write("%s,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%.4f\n"
                     % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], N, r[7]))
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
