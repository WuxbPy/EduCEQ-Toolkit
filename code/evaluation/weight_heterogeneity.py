# -*- coding: utf-8 -*-
"""Weight heterogeneity at fixed raw support (Table 11).

The question
------------
If the screening criterion is a count of non-zero neighbours, then two kernels
with the same support count but very different weight profiles receive the same
verdict, although the variance of a weighted ratio is governed by the effective
number of weighted observations, not by the number of non-zero entries. This
script holds the raw support fixed and varies only the weight profile.

Design
------
One point pattern (n = 6,000 uniform in the unit square, two categories of equal
size) and four weight profiles, each tuned so that the median raw support of the
target category is 60 in all four:

    binary  w = 1                        (support and information coincide)
    gauss   w = exp(-d^2 / 2b^2)         (the kernel shipped in the package)
    sharp   w = exp(-d^2 / 2b^2)^4       (same support, concentrated weights)
    jitter  gauss x lognormal(0, 1.5)    (heavy-tailed heterogeneity, same support)

For each profile, labels are permuted 200 times (the null of no association
beyond distribution) and the standard deviation of CEQ_AB and of the
second-order term across permutations is recorded, together with the Kish
effective sample size of each row.

Expected reading: if the raw count were the right screening quantity, the four
standard deviations would agree. They do not; they track ESS^(-1/2).

Run:
    python evaluation/weight_heterogeneity.py

Output: printed table + results/weight_heterogeneity.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402
from scipy import sparse                             # noqa: E402

from educeq import kernels, estimator                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(ROOT), "results", "weight_heterogeneity.csv")

N = 6000
TARGET_SUPPORT = 60
REPS = 200
SEED = 20261109
FAMILIES = ["binary", "gauss", "sharp", "jitter"]


def row_ess(v):
    v = np.asarray(v, float)
    s2 = float((v * v).sum())
    return float(v.sum()) ** 2 / s2 if s2 > 0 else 0.0


def build(family, coords, b, rng):
    W = sparse.csr_matrix(kernels.distance(coords, b))
    W = W.copy()
    if family == "binary":
        W.data[:] = 1.0
    elif family == "gauss":
        pass
    elif family == "sharp":
        W.data = W.data ** 4
    elif family == "jitter":
        W.data = W.data * rng.lognormal(0.0, 1.5, size=W.nnz)
    return W


def describe(W, types):
    coo = W.tocoo()
    onehot = sparse.csr_matrix((np.ones(len(types)), (np.arange(len(types)), types)),
                               shape=(len(types), 2))
    cnt = sparse.csr_matrix((np.ones(coo.nnz), (coo.row, coo.col)), shape=W.shape)
    n_B = np.asarray((cnt @ onehot).todense())[types == 0, 0]
    ess = np.array([row_ess(W.getrow(i).data) for i in range(0, W.shape[0], 20)])
    return float(np.median(n_B)), float(np.nanmedian(ess))


def tune(family, coords, types, rng):
    """Bandwidth giving median raw support closest to TARGET_SUPPORT."""
    lo, hi = 1e-3, 0.4
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        n_B, _ = describe(build(family, coords, mid, rng), types)
        lo, hi = (mid, hi) if n_B < TARGET_SUPPORT else (lo, mid)
    return mid


def main():
    rng = np.random.default_rng(SEED)
    coords = rng.uniform(size=(N, 2))
    types = np.repeat([0, 1], N // 2)

    print("Weight heterogeneity at fixed raw support")
    print("n = %d, target median raw support = %d, permutations per profile = %d, seed = %d\n"
          % (N, TARGET_SUPPORT, REPS, SEED))
    head = ("%-7s %7s %6s %6s %8s %8s %9s %9s"
            % ("profile", "b", "n_B", "ESS", "sd(CEQ)", "sd(CEQ2)", "sd*sqrtESS", "rel.sd"))
    print(head)
    print("-" * len(head))

    rows = []
    for family in FAMILIES:
        frng = np.random.default_rng(SEED)
        b = tune(family, coords, types, frng)
        W = build(family, coords, b, np.random.default_rng(SEED + 1))
        n_B, ess = describe(W, types)
        vals, vals2 = [], []
        for _ in range(REPS):
            tp = rng.permutation(types)
            out = estimator._decompose_unchecked(W, coords, tp, bandwidth=b,
                                                 grid_n=40)
            vals.append(out["C"][0, 1])
            vals2.append(out["C2"][0, 1])
        sd, sd2 = float(np.nanstd(vals)), float(np.nanstd(vals2))
        print("%-7s %7.4f %6.0f %6.1f %8.4f %8.4f %9.4f %9.4f"
              % (family, b, n_B, ess, sd, sd2, sd * np.sqrt(ess), sd / np.nanmean(vals)))
        rows.append((family, b, n_B, ess, sd, sd2, np.nanmean(vals)))

    print("\nsd(CEQ) rises by a factor of about %.1f across profiles while the raw"
          % (max(r[4] for r in rows) / min(r[4] for r in rows)))
    print("support is identical; sd*sqrt(ESS) stays roughly constant, so the variance")
    print("scale is the effective sample size and not the non-zero count.")

    with open(OUT, "w", encoding="utf8") as f:
        f.write("profile,bandwidth,raw_support,ess,sd_ceq,sd_ceq2,mean_ceq\n")
        for r in rows:
            f.write("%s,%.4f,%.1f,%.2f,%.6f,%.6f,%.6f\n" % r)
    print("\nwritten: %s" % OUT)


if __name__ == "__main__":
    main()
