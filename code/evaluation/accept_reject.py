# -*- coding: utf-8 -*-
"""Controlled acceptance and rejection with known association structure (Tables 8 and 9).

Why this script exists
----------------------
A checker is only tested if both branches are exercised. Testing the refusal alone,
on a public dataset whose largest category (1,765) is below the population
threshold (3,000) at every bandwidth, would be passed by a checker that always
answered "undecidable". This script supplies the acceptance branch and a known
truth for it.

Designs, all with an analytic population value of CEQ
-----------------------------------------------------
G1  null          categories independent                      truth CEQ = 1 exactly
G2  attraction    a fraction theta of B points is duplicated at A points;
                  the duplicate carries weight 1 while the background carries
                  p_B * S, which gives  CEQ = 1 + theta / (S + theta),
                  with S the mean total weight of an A neighbourhood
G3  displacement  a fraction theta of B points is moved outside the domain, where
                  they contribute nothing to either the numerator or the
                  neighbourhood total, so CEQ = 2(1-theta)/(2-theta)
G4  violation     three families that must be refused: (a) N_B < 3000,
                  (b) N_B >= 3000 but effective size < 30, (c) rho > 0.01
G5  counterfactual  family (b) estimated with the unchecked internal path, to
                  measure what the refusal buys

Everything runs on synthetic data with fixed seeds.

Run:
    python evaluation/accept_reject.py            # default 200 replicates
    python evaluation/accept_reject.py --reps 50

Output: printed table + results/accept_reject.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402

from educeq import diagnostics, estimator, inference, kernels  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(ROOT), "results", "accept_reject.csv")

DOMAIN = 100.0
N = 9000
B = 2.7                 # bandwidth at which all three criteria hold
SEED = 20261109
N_PERM = 99


def pattern(rng, n=N, theta=0.0, mode="null", b=B):
    """Return coordinates, labels and the analytic value of CEQ_{A->B}."""
    a = rng.uniform(0, DOMAIN, size=(n // 2, 2))
    if mode == "null":
        bb = rng.uniform(0, DOMAIN, size=(n // 2, 2))
        truth = 1.0
    elif mode == "attract":
        # a fraction theta of B is placed on top of A points, jittered by 0.05 km
        # so that the pairs are not dropped as zero-distance duplicates
        k = int(round(theta * (n // 2)))
        src = rng.choice(len(a), size=k, replace=True)
        near = a[src] + rng.normal(scale=0.05, size=(k, 2))
        bb = np.vstack([near, rng.uniform(0, DOMAIN, size=(n // 2 - k, 2))])
        truth = None                       # filled in below, needs S
    elif mode == "displace":
        # a fraction theta of B is parked well outside the domain and therefore
        # outside every A neighbourhood, which lowers CEQ by exactly theta
        k = int(round(theta * (n // 2)))
        bb = rng.uniform(0, DOMAIN, size=(n // 2, 2))
        bb[:k] = np.column_stack([np.full(k, DOMAIN + 60.0),
                                  rng.uniform(0, DOMAIN, size=k)])
        # the departed mass leaves both the numerator and the neighbourhood
        # total, so with equal category sizes CEQ = 2(1-theta)/(2-theta)
        truth = 2.0 * (1.0 - theta) / (2.0 - theta)
    else:
        raise ValueError(mode)
    coords = np.vstack([a, bb])
    types = np.concatenate([np.zeros(len(a), int), np.ones(len(bb), int)])
    return coords, types, truth


def mean_weight(W, types, target=0):
    """Mean total weight of a neighbourhood of the target category."""
    tot = np.asarray(W.sum(1)).ravel()
    return float(tot[types == target].mean())


def run_design(mode, theta, reps, b=B, seed=SEED, n_perm=N_PERM):
    obs, truth, pvals, decided = [], [], [], 0
    for r in range(reps):
        rng = np.random.default_rng(seed + 17 * r)
        coords, types, tr = pattern(rng, theta=theta, mode=mode, b=b)
        W = kernels.distance(coords, b)
        dec = diagnostics.check(W, types, target=0)
        if not dec.decidable:
            continue
        decided += 1
        res = estimator.decompose(W, coords, types, bandwidth=2 * b,
                                  grid_n=40, target=0, contract=dec)
        obs.append(float(res["C"][0, 1]))
        if tr is None:
            tr = 1.0 + theta / (mean_weight(W, types) + theta)
        truth.append(float(tr))
        p = inference.permutation(W, types, contract=dec, n_perm=n_perm,
                                  seed=r)
        pvals.append(_pair_p(W, types, res, p))
    obs = np.array(obs)
    return dict(mode=mode, theta=theta, reps=reps, decided=decided,
                truth=float(np.mean(truth)) if truth else np.nan,
                mean=float(obs.mean()), sd=float(obs.std(ddof=1)),
                bias=float(obs.mean() - np.mean(truth)),
                rmse=float(np.sqrt(np.mean((obs - np.mean(truth)) ** 2))),
                reject=float(np.mean(np.array(pvals) < 0.05)) if pvals else np.nan)


def _pair_p(W, types, res, perm_out):
    """Two-sided permutation p-value for the (0,1) cell, from the null draws."""
    obs = res["C"][0, 1]
    null = perm_out["null"][:, 0, 1]
    return float((np.sum(np.abs(null - 1.0) >= abs(obs - 1.0)) + 1)
                 / (len(null) + 1))


def run_violation(family, reps, seed=SEED):
    """How often is a violated configuration refused, and at the right step?"""
    refused, right_step, forced = 0, 0, []
    for r in range(reps):
        rng = np.random.default_rng(seed + 31 * r)
        if family == "a":                     # N_B < 3000
            coords, types, _ = pattern(rng, n=4000)
            b = 5.0
        elif family == "b":                   # effective size below 30
            coords, types, _ = pattern(rng, n=N)
            b = 2.0
        else:                                 # coverage above 0.01
            coords, types, _ = pattern(rng, n=N)
            b = 4.0
        W = kernels.distance(coords, b)
        dec = diagnostics.check(W, types, target=0)
        if not dec.decidable:
            refused += 1
            if dec.failed and dec.failed[0].startswith("step%d" % {"a": 1, "b": 2, "c": 3}[family]):
                right_step += 1
            out = estimator._decompose_unchecked(W, coords, types,
                                                 bandwidth=2 * b, grid_n=40)
            forced.append(float(out["C"][0, 1]))
    return dict(family=family, reps=reps, refused=refused / reps,
                right_step=right_step / reps,
                forced_sd=float(np.std(forced, ddof=1)) if len(forced) > 1 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    print("Controlled acceptance and rejection (Tables 8 and 9)")
    print("n = %d, bandwidth = %.1f km, replicates = %d, permutations = %d, seed = %d\n"
          % (N, B, args.reps, N_PERM, SEED))

    rows = []
    print("-- acceptance: known association structure ------------------------")
    print("%-22s %8s %8s %9s %9s %9s %9s"
          % ("design", "truth", "mean", "bias", "RMSE", "rejection", "decided"))
    for mode, theta in (("null", 0.0), ("attract", 0.10), ("attract", 0.25),
                        ("displace", 0.25)):
        d = run_design(mode, theta, args.reps)
        label = "%s (theta=%.2f)" % (mode, theta) if mode != "null" else "null"
        print("%-22s %8.4f %8.4f %9.4f %9.4f %9.3f %9d"
              % (label, d["truth"], d["mean"], d["bias"], d["rmse"],
                 d["reject"], d["decided"]))
        rows.append(d)

    print("\n-- rejection: violated configurations ------------------------------")
    print("%-24s %10s %12s %12s" % ("family", "refused", "right step", "forced sd"))
    viol = []
    for fam, name in (("a", "N_B < 3000"), ("b", "effective size < 30"),
                      ("c", "rho > 0.01")):
        v = run_violation(fam, args.reps)
        print("%-24s %10.3f %12.3f %12.4f"
              % (name, v["refused"], v["right_step"], v["forced_sd"]))
        viol.append(v)

    sd_ok = next(r for r in rows if r["mode"] == "null")["sd"]
    v_b = next(v for v in viol if v["family"] == "b")
    print("\nG5 counterfactual: estimating where the contract refuses gives")
    print("  sd = %.4f against sd = %.4f in the decidable null, a factor of %.1f"
          % (v_b["forced_sd"], sd_ok, v_b["forced_sd"] / sd_ok))

    with open(OUT, "w", encoding="utf8") as f:
        f.write("design,theta,truth,mean,sd,bias,rmse,reject_rate,replicates\n")
        for r in rows:
            f.write("%s,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%d\n"
                    % (r["mode"], r["theta"], r["truth"], r["mean"], r["sd"],
                       r["bias"], r["rmse"], r["reject"], r["reps"]))
        f.write("\nfamily,refused_rate,right_step_rate,forced_sd\n")
        for v in viol:
            f.write("%s,%.3f,%.3f,%.4f\n"
                    % (v["family"], v["refused"], v["right_step"], v["forced_sd"]))
    print("\nwritten: %s" % OUT)


if __name__ == "__main__":
    main()
