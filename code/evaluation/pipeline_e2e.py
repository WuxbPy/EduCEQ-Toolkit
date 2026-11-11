# -*- coding: utf-8 -*-
"""End-to-end pipeline on three datasets (Table 12).

Why this script exists
----------------------
An evaluation of validity.proxy_test alone exercises neither the checker, nor the
decomposition, nor the inference channel: that module operates on tabular data and
never touches the quotient pipeline. Testing the components separately therefore
does not show the pipeline as a pipeline. This script runs all five stages of
Section 3.1 on three datasets and prints, stage by stage, the input, the expected
output and the observed output, together with the returned object itself.

The three datasets are chosen so that the two verdicts and the impossible case
all occur:
    null      synthetic, criteria met, no association      -> estimate near 1
    planted   synthetic, criteria met, known association   -> estimate above 1
    panel     College Scorecard, criteria not met          -> no estimate at all

Run:
    python evaluation/pipeline_e2e.py

Output: printed trace + results/pipeline_e2e.log (via tee in the Makefile)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402

from educeq import diagnostics, estimator, inference, kernels, contract  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "scorecard.npz")
SEED = 20261109
B = 2.7


def project(lonlat):
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    lat0 = np.deg2rad(np.nanmean(lat))
    return np.column_stack([np.deg2rad(lon) * 6371.0 * np.cos(lat0),
                            np.deg2rad(lat) * 6371.0])


def datasets():
    rng = np.random.default_rng(SEED)
    # 1. null
    a = rng.uniform(0, 100, size=(4500, 2))
    b = rng.uniform(0, 100, size=(4500, 2))
    null = (np.vstack([a, b]),
            np.concatenate([np.zeros(4500, int), np.ones(4500, int)]))
    # 2. planted: 40 per cent of B placed on top of A, jittered below the bandwidth
    k = 1800
    near = a[rng.choice(4500, size=k, replace=False)] + rng.normal(scale=0.05,
                                                                   size=(k, 2))
    b2 = np.vstack([near, rng.uniform(0, 100, size=(4500 - k, 2))])
    planted = (np.vstack([a, b2]),
               np.concatenate([np.zeros(4500, int), np.ones(4500, int)]))
    # 3. public panel
    d = np.load(DATA)
    panel = (project(d["coords"]), d["types"])
    return [("null", null, B), ("planted", planted, B),
            ("panel", panel, 150.0)]


def main():
    print("End-to-end pipeline (Table 12)")
    print("Five stages of Section 3.1 on three datasets, seed = %d\n" % SEED)
    head = "%-8s %-22s %-34s %s"
    print(head % ("dataset", "stage", "expected", "observed"))
    print("-" * 110)

    for name, (coords, types), bw in datasets():
        # ---------------- stage 1: kernel
        W = kernels.distance(coords, bandwidth=bw)
        import scipy.sparse as sp
        print(head % (name, "1 kernel",
                      "sparse, non-negative",
                      "nnz=%d, min w=%.3f, rows=%d"
                      % (W.nnz, W.data.min() if W.nnz else 0.0, W.shape[0])))

        # ---------------- stage 2: verdict
        dec = diagnostics.check(W, types, target=0)
        print(head % ("", "2 verdict",
                      "accept for synthetic, refuse for panel",
                      "decidable=%s  n_B=%.0f e_B=%.1f rho_e=%.4f"
                      % (dec.decidable, dec.kappa_raw, dec.ess, dec.rho_ess)))

        # ---------------- stage 3: estimate
        if dec.decidable:
            res = estimator.decompose(W, coords, types, bandwidth=2 * bw,
                                      grid_n=40, target=0, contract=dec)
            print(head % ("", "3 estimate",
                          "C, C1, C2, identity residual",
                          "C01=%.4f  C1(01)=%.4f  C2(01)=%.4f  identity=%s"
                          % (res["C"][0, 1], res["C1"][0, 1], res["C2"][0, 1],
                             res["identity_ok"])))
            p = inference.permutation(W, types, contract=dec, n_perm=199,
                                      seed=0)
            pv = (np.sum(np.abs(p["null"][:, 0, 1] - 1.0)
                         >= abs(res["C"][0, 1] - 1.0)) + 1) / 200.0
            print(head % ("", "4 inference",
                          "p available, conditional permutation",
                          "z01=%.2f  p01=%.3f" % (p["z"][0, 1], pv)))
        else:
            res = estimator.decompose(W, coords, types, bandwidth=2 * bw,
                                      grid_n=40, target=0)
            print(head % ("", "3 estimate",
                          "no estimate; status object only",
                          "'C' in result: %s  failed=%s"
                          % ("C" in res, res["failed"])))
            try:
                estimator.ceq_matrix(W, types, contract=dec)
                observed = "returned a number (this would be the defect)"
            except Exception as exc:
                observed = type(exc).__name__
            print(head % ("", "4 inference",
                          "unreachable without a passing verdict",
                          "ceq_matrix -> %s" % observed))

        # ---------------- stage 5: the object itself
        keys = sorted(res.keys())
        print(head % ("", "5 contract",
                      "estimate and conditions in one object",
                      "keys=%s" % keys))
        print()

    print("The panel row is the point of the table: the estimate does not exist,")
    print("and the only way to obtain one is the private arithmetic of")
    print("estimator._decompose_unchecked, which the paper does not treat as an")
    print("estimate and which no public entry point exposes.")


if __name__ == "__main__":
    main()
