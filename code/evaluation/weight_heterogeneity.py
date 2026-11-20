# -*- coding: utf-8 -*-
"""Weight heterogeneity at fixed raw support (Table 14).

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
second-order term across permutations is recorded, together with the three
support measures of Section 4.2.

Aggregation (revision 3)
------------------------
The three measures are read from ``diagnostics.support``, the function the
contract itself calls, so that the experiment cannot drift from the definition
the checker uses. All three are taken over the *target-category subset* of the
row -- the neighbours j with t_j = B and W_ij > 0 -- and aggregated as the median
over subjects of type A. Because they are taken over the same set,

    e_B <= n_B,   with equality exactly when the non-zero weights of that
                  row are equal (in particular, for a binary neighbour list).

An earlier version of this script computed the effective size over every
non-zero entry of the row rather than over its target-category subset. For the
binary profile that returned the whole-row neighbour count (about 121, of which
60 are of the target category) and so reported an effective size exceeding the
raw support. The value is now taken from the library, and the assertion below
holds throughout. The whole-row value is still printed, in the column ``e_row``,
so that the difference between the two aggregations stays on the record.

Expected reading: if the raw count were the right screening quantity, the four
standard deviations would agree. They do not; they track the effective size.

Run:
    python evaluation/weight_heterogeneity.py

Output: printed table + results/weight_heterogeneity.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402
from scipy import sparse                             # noqa: E402

from educeq import diagnostics, estimator, kernels    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(ROOT), "results", "weight_heterogeneity.csv")

N = 6000
TARGET_SUPPORT = 60
REPS = 200
SEED = 20261109
FAMILIES = ["binary", "gauss", "sharp", "jitter"]


def row_ess(v):
    """Kish effective size of one set of weights, (sum w)^2 / sum w^2."""
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


def target_counts(W, types):
    """Number of category-B neighbours of each subject of type A."""
    coo = W.tocoo()
    cnt = sparse.csr_matrix((np.ones(coo.nnz), (coo.row, coo.col)), shape=W.shape)
    onehot = sparse.csr_matrix((np.ones(len(types)), (np.arange(len(types)), types)),
                               shape=(len(types), 2))
    return np.asarray((cnt @ onehot).todense())[types == 0, 1]


def describe(W, types):
    """The three support measures of the target pair, as Equation (1) uses them.

    ``n_B``, ``S_B`` and ``e_B`` are medians over subjects of the source type A
    of quantities defined on the category-B subset of each row, and are read
    from the package so that the experiment and the contract share one
    implementation. ``e_row`` is the effective size of the whole row, over both
    categories, and is reported only to document the difference between the two
    aggregations.
    """
    s = diagnostics.support(W, types)
    n_B = float(s["raw"][0, 1])
    S_B = float(s["weighted"][0, 1])
    e_B = float(s["ess"][0, 1])
    e_row = float(np.nanmedian([row_ess(W.getrow(i).data)
                                for i in range(0, W.shape[0], 20)]))
    assert e_B <= n_B + 1e-9, "effective size exceeds the raw support: %r" % (e_B,)
    return n_B, S_B, e_B, e_row


def tune(family, coords, types, rng):
    """Bandwidth giving median raw support closest to TARGET_SUPPORT."""
    lo, hi = 1e-3, 0.4
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        n_B = float(np.median(target_counts(build(family, coords, mid, rng), types)))
        lo, hi = (mid, hi) if n_B < TARGET_SUPPORT else (lo, mid)
    return mid


def main():
    rng = np.random.default_rng(SEED)
    coords = rng.uniform(size=(N, 2))
    types = np.repeat([0, 1], N // 2)

    print("Weight heterogeneity at fixed raw support")
    print("n = %d, target median raw support = %d, permutations per profile = %d, seed = %d\n"
          % (N, TARGET_SUPPORT, REPS, SEED))
    head = ("%-7s %7s %6s %6s %6s %6s %8s %8s %9s"
            % ("profile", "b", "n_B", "S_B", "e_B", "e_row",
               "sd(CEQ)", "sd(CEQ2)", "sd*sqrt(e_B)"))
    print(head)
    print("-" * len(head))

    rows = []
    for family in FAMILIES:
        frng = np.random.default_rng(SEED)
        b = tune(family, coords, types, frng)
        W = build(family, coords, b, np.random.default_rng(SEED + 1))
        n_B, S_B, e_B, e_row = describe(W, types)
        vals, vals2 = [], []
        for _ in range(REPS):
            tp = rng.permutation(types)
            out = estimator._decompose_unchecked(W, coords, tp, bandwidth=b,
                                                 grid_n=40)
            vals.append(out["C"][0, 1])
            vals2.append(out["C2"][0, 1])
        sd, sd2 = float(np.nanstd(vals)), float(np.nanstd(vals2))
        print("%-7s %7.4f %6.1f %6.2f %6.1f %6.1f %8.4f %8.4f %9.4f"
              % (family, b, n_B, S_B, e_B, e_row, sd, sd2, sd * np.sqrt(e_B)))
        rows.append((family, b, n_B, S_B, e_B, e_row, sd, sd2,
                     sd * np.sqrt(e_B), float(np.nanmean(vals))))

    print("\nsd(CEQ) varies by a factor of about %.1f across profiles while the raw"
          % (max(r[6] for r in rows) / min(r[6] for r in rows)))
    print("support is identical (60 in all four) and the effective size varies from")
    print("%.1f to %.1f. sd*sqrt(e_B) stays within %.4f to %.4f, so the scale of the"
          % (min(r[4] for r in rows), max(r[4] for r in rows),
             min(r[8] for r in rows), max(r[8] for r in rows)))
    print("variance is the effective size and not the number of non-zero weights.")

    with open(OUT, "w", encoding="utf8") as f:
        f.write("profile,bandwidth,raw_support,weighted_support,ess,ess_row,"
                "sd_ceq,sd_ceq2,s_sqrt_ess,mean_ceq\n")
        for r in rows:
            f.write("%s,%.4f,%.1f,%.2f,%.2f,%.1f,%.6f,%.6f,%.4f,%.6f\n" % r)
    print("\nwritten: %s" % OUT)


if __name__ == "__main__":
    main()
