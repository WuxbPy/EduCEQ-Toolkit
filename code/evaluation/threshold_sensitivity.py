# -*- coding: utf-8 -*-
"""Sensitivity of the decidability criterion to distribution, imbalance and kernel.

The question is whether the thresholds of 30 effective neighbours, 1 per cent
coverage and 3,000 objects survive across different distributions, class
imbalances and kernel configurations. The script answers with a measured standard
deviation rather than with a restatement of the rule.

Design
------
Three point patterns on the same 100 x 100 km window, n = 20,000 objects:

    csr        homogeneous Poisson (complete spatial randomness)
    clustered  Thomas process: 200 parent locations, 100 children each,
               Gaussian displacement with sigma = 1.5 km
    regular    jittered lattice, spacing 0.71 km, uniform jitter of 0.15 km

Three imbalances for the target category B, pi_B in {0.5, 0.2, 0.05}, so that
N_B is 10,000, 4,000 and 1,000; three weight profiles (binary, gauss, sharp);
three bandwidths, 1.5, 2.0 and 2.5 km. Labels are redrawn 100 times in every
cell, which is the null of no association beyond the marginal distribution, and
the standard deviation of CEQ_AB across the redraws is recorded next to the
effective size e_B that the checker reads.

What the criterion has to do
-----------------------------
The variance of a ratio of weighted sums obeys sd ~ c / sqrt(e_B). If that holds
in every cell, then a threshold on e_B is a threshold on sd, whatever the point
pattern, the imbalance or the kernel. The script reports c = sd * sqrt(e_B) per
cell and its range over all cells.

Run:
    python evaluation/threshold_sensitivity.py
Output: results/threshold_sensitivity.csv and results/threshold_sensitivity.json
"""
import json
import os
import sys

import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from educeq import kernels, diagnostics  # noqa: E402
from educeq._core import clq_global  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")
N = 20000
SIDE = 100.0
PI = [0.5, 0.2, 0.05]
BAND = [1.5, 2.0, 2.5]
PROFILES = ["binary", "gauss", "sharp"]
REPS = 100
SEED = 20261111


def pattern(kind, rng):
    if kind == "csr":
        return rng.uniform(0, SIDE, size=(N, 2))
    if kind == "clustered":
        parents = rng.uniform(0, SIDE, size=(200, 2))
        which = rng.integers(0, 200, N)
        return parents[which] + rng.normal(0, 1.5, size=(N, 2))
    g = int(np.ceil(np.sqrt(N)))
    axis = np.linspace(0, SIDE, g)
    gx, gy = np.meshgrid(axis, axis)
    base = np.column_stack([gx.ravel(), gy.ravel()])[:N]
    return base + rng.uniform(-0.15, 0.15, size=base.shape)


def profile(W, kind, rng):
    W = W.copy()
    if kind == "binary":
        W.data[:] = 1.0
    elif kind == "sharp":
        W.data = W.data ** 4
    elif kind == "jitter":
        W.data = W.data * rng.lognormal(0.0, 1.5, size=W.nnz)
    return W


def ess_row(W, types, a):
    idx = np.where(types == a)[0]
    onehot = sparse.csr_matrix((np.ones(len(types)), (np.arange(len(types)), types)),
                               shape=(len(types), 2))
    S = np.asarray((W[idx] @ onehot).todense())
    S2 = np.asarray((W[idx].multiply(W[idx]) @ onehot).todense())
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(S2 > 0, S ** 2 / S2, 0.0)
    return float(np.median(e[:, 1]))


def main():
    os.makedirs(RES, exist_ok=True)
    rows = []
    for kind in ["csr", "clustered", "regular"]:
        rng = np.random.default_rng(SEED + {"csr": 1, "clustered": 2, "regular": 3}[kind])
        xy = pattern(kind, rng)
        for b in BAND:
            W0 = kernels.distance(xy, bandwidth=b)
            for prof in PROFILES:
                W = profile(W0, prof, np.random.default_rng(SEED))
                for pi in PI:
                    rng2 = np.random.default_rng(SEED + 7)
                    vals = np.empty(REPS)
                    for r in range(REPS):
                        types = (rng2.random(N) < pi).astype(int)   # 1 = target B
                        C, _ = clq_global(W, types, [0, 1])
                        vals[r] = C[0, 1]
                    types = (np.random.default_rng(SEED + 7).random(N) < pi).astype(int)
                    d = diagnostics.check(W, types, target=1, pair=(0, 1))
                    sd = float(vals.std(ddof=1))
                    rows.append(dict(process=kind, bandwidth=b, profile=prof,
                                     pi_B=pi, n_B=d.kappa_raw, S_B=d.kappa_weighted,
                                     e_B=d.ess, rho_e=d.rho_ess, sd=sd,
                                     sd_sqrt_e=sd * np.sqrt(max(d.ess, 1e-12)),
                                     decidable=bool(d.decidable),
                                     failed=" ".join(d.failed)))
                    print("%-10s b=%.1f %-6s pi=%.2f  n_B=%7.1f e_B=%8.1f "
                          "rho_e=%.4f  sd=%.5f  sd*sqrt(e)=%.4f  %s"
                          % (kind, b, prof, pi, d.kappa_raw, d.ess, d.rho_ess, sd,
                             sd * np.sqrt(max(d.ess, 1e-12)),
                             "accept" if d.decidable else "refuse"), flush=True)
            del W0

    import csv
    with open(os.path.join(RES, "threshold_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    c = np.array([r["sd_sqrt_e"] for r in rows])
    acc = [r for r in rows if r["decidable"]]
    summary = dict(cells=len(rows), c_median=float(np.median(c)),
                   c_min=float(c.min()), c_max=float(c.max()),
                   ratio=float(c.max() / c.min()),
                   sd_at_30=float(np.median(c) / np.sqrt(30)),
                   accepted=len(acc),
                   sd_accepted_max=float(max(r["sd"] for r in acc)) if acc else None,
                   sd_refused_min=float(min(r["sd"] for r in rows if not r["decidable"])))
    print(json.dumps(summary, indent=1), flush=True)
    with open(os.path.join(RES, "threshold_sensitivity.json"), "w") as f:
        json.dump(dict(summary=summary, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
