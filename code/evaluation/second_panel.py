# -*- coding: utf-8 -*-
"""Second public panel: a county police force area, one year (Section 5.3).

The primary panel of the paper is refused by the contract. This script supplies
the complementary evidence: a real application in which the contract accepts and
the estimate is interpretable, produced end to end from one public file:

    data.police.uk street-level crime, Cambridgeshire Constabulary, 2025
    N = 81,959 offences, 14 categories, longitude and latitude to six decimals.

Pipeline, in the order of Section 3.1: distance kernel -> three-step check ->
enrichment matrix -> exact first/second-order decomposition -> conditional
permutation -> false discovery rate control. Nothing is estimated at a bandwidth
the check refuses.

Outputs (written into ../results, a sibling of code/):
    second_panel_sweep.csv     support measures and verdict along the sweep
    second_panel_matrix.csv    the 14 x 14 matrix at the working bandwidth
    second_panel_top.csv       the enriched pairs with p and q
    second_panel.json          summary values quoted in the paper

Run:
    python evaluation/second_panel.py
The archive is not redistributed; see data/PREPARE_CRIME.md for the download and
the filter that reproduces it byte for byte.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from educeq import kernels, diagnostics, estimator, inference  # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "..", "results"))
DEFAULT = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "..", "..", "data", "uk_police_all.npz"))
NPZ = os.environ.get("EDUCEQ_CRIME", DEFAULT)

REGION = 3          # Cambridgeshire Constabulary
PERIODS = (24, 36)  # 2025
SWEEP = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.47]
WORK = 0.25
N_PERM = 199
SEED = 20261110

NAMES = ["Anti-social behaviour", "Bicycle theft", "Burglary",
         "Criminal damage and arson", "Drugs", "Other crime", "Other theft",
         "Possession of weapons", "Public order", "Robbery", "Shoplifting",
         "Theft from the person", "Vehicle crime",
         "Violence and sexual offences"]


def load():
    z = np.load(NPZ)
    m = (z["region_id"] == REGION) & (z["time_periods"] >= PERIODS[0]) \
        & (z["time_periods"] < PERIODS[1])
    c = z["coords"][m]
    t = z["types"][m].astype(int) - 1
    lon0, lat0 = c[:, 0].mean(), c[:, 1].mean()
    k = np.cos(np.deg2rad(lat0))
    xy = np.column_stack([(c[:, 0] - lon0) * 111.320 * k,
                          (c[:, 1] - lat0) * 110.574])
    return xy, t


def main():
    os.makedirs(RES, exist_ok=True)
    xy, t = load()
    n = len(t)
    counts = np.bincount(t)
    target = int(counts.argmax())
    print("N = %d, categories = %d, largest = %s (%d)"
          % (n, len(counts), NAMES[target], counts[target]), flush=True)

    rows, mats, t0 = [], {}, time.time()
    for b in SWEEP:
        ws = time.time()
        W = kernels.distance(xy, bandwidth=b)
        d = diagnostics.check(W, t, target=target)
        rows.append(dict(b=b, nnz=int(W.nnz), n_B=d.kappa_raw,
                         S_B=d.kappa_weighted, e_B=d.ess,
                         rho_n=d.rho_raw, rho_S=d.rho_weighted,
                         rho_e=d.rho_ess, decidable=bool(d.decidable),
                         failed=" ".join(d.failed), seconds=time.time() - ws))
        print("b=%.3f  n_B=%6.1f  S_B=%7.1f  e_B=%7.1f  rho_e=%.5f  %s"
              % (b, d.kappa_raw, d.kappa_weighted, d.ess, d.rho_ess,
                 "accept" if d.decidable else "refuse " + str(d.failed)),
              flush=True)
        if abs(b - WORK) < 1e-9:
            dec, wg = d, W
        del W, d
    import csv as _csv
    with open(os.path.join(RES, "second_panel_sweep.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("working bandwidth %.3f km, verdict decidable = %s"
          % (WORK, dec.decidable), flush=True)
    out = estimator.decompose(wg, xy, t, WORK, grid_n=600, contract=dec)
    print("identity_ok =", out["identity_ok"], flush=True)
    C, C1, C2 = out["C"], out["C1"], out["C2"]

    perm = inference.permutation(wg, t, contract=dec, n_perm=N_PERM, seed=SEED)
    null = perm["null"]
    p = np.ones((C.shape))
    for a in range(C.shape[0]):
        for b2 in range(C.shape[1]):
            p[a, b2] = (1.0 + np.sum(np.abs(null[:, a, b2] - 1.0)
                                     >= abs(C[a, b2] - 1.0))) / (N_PERM + 1.0)

    iu = np.triu_indices(len(counts), 1)
    pv = p[iu]
    q, k_bh = inference.fdr(pv)
    qb, k_by = inference.fdr_by(pv)
    print("pairs tested %d, BH rejections %d, BY rejections %d"
          % (len(pv), k_bh, k_by), flush=True)

    # table of pairs, sorted by quotient
    pairs = []
    for i, (a, b2) in enumerate(zip(*iu)):
        pairs.append(dict(a=NAMES[a], b=NAMES[b2], ceq=float(C[a, b2]),
                          c1=float(C1[a, b2]), c2=float(C2[a, b2]),
                          p=float(pv[i]), q_bh=float(q[i]),
                          q_by=float(qb[i])))
    pairs.sort(key=lambda r: -r["ceq"])
    with open(os.path.join(RES, "second_panel_top.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)
    for r in pairs[:6]:
        print("  %-28s -> %-28s CEQ=%.3f  C1=%.3f  C2=%.3f  q_BH=%.3g"
              % (r["a"], r["b"], r["ceq"], r["c1"], r["c2"], r["q_bh"]), flush=True)

    np.savetxt(os.path.join(RES, "second_panel_matrix.csv"), C, delimiter=",",
               fmt="%.4f")

    # stability of the matrix across the admissible range
    from scipy.stats import spearmanr
    stab = {}
    for b in (0.15, 0.40):
        Wb = kernels.distance(xy, bandwidth=b)
        db = diagnostics.check(Wb, t, target=target)
        if db.decidable:
            Cb = estimator.ceq_matrix(Wb, t, contract=db)
            stab[b] = float(spearmanr(C[iu], Cb[iu]).statistic)
        del Wb
    print("Spearman against working bandwidth:", stab, flush=True)

    summary = dict(n=int(n), n_categories=int(len(counts)),
                   largest=NAMES[target], n_largest=int(counts[target]),
                   working_b=WORK, e_B=dec.ess, rho_e=dec.rho_ess,
                   n_B=dec.kappa_raw, S_B=dec.kappa_weighted,
                   identity_ok=bool(out["identity_ok"]),
                   normalization=estimator.normalization_ok(C, t),
                   n_perm=N_PERM, pairs=int(len(pv)),
                   bh_rejections=int(k_bh), by_rejections=int(k_by),
                   stability=stab, sweep=rows,
                   elapsed=time.time() - t0)
    with open(os.path.join(RES, "second_panel.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("elapsed %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
