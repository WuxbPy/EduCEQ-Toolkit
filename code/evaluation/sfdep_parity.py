# -*- coding: utf-8 -*-
"""Parity with the published colocation-quotient definition (Section 5.6).

Why this script exists
----------------------
The relationship with existing co-location software has to be checkable, not
asserted. The R package sfdep implements the colocation quotients of Leslie and
Kronenfeld from a neighbour list: a binary, row-standardised neighbourhood, no
arbitrary weighting, and the quotient formed as the observed share of the target
category within the neighbours of the source category, divided by that category's
global share.

This script reimplements that definition from the published formula, without
using the package, and compares it with estimator.ceq_matrix driven by a binary
kernel built from the same neighbour list.

Two neighbourhood designs, and why both (revision 3)
----------------------------------------------------
The two estimators normalise inside each subject -- the local share of the target
category is formed from the row, so a subject with a large total weight does not
contribute a large numerator -- and they differ only in how subjects are
combined. The published quotient averages the per-subject shares with equal
weight:

    CLQ_AB = (1/m_A) sum_{i in A} ( sum_{j in N(i), t_j = B} w_ij
                                    / sum_{j in N(i)} w_ij ) / p_B

whereas Equation (1) of the paper is a ratio of sums, which pools subjects in
proportion to the total weight of their row:

    CEQ_AB = sum_{i in A} sum_{j in N(i), t_j = B} w_ij
             / ( p_B * sum_{i in A} sum_{j in N(i)} w_ij ).

The two are the same expression when every row has the same total weight, and
they are not otherwise. The script therefore reports both designs:

  (a) an 80-nearest-neighbour list, whose rows all sum to 80: the classical
      configuration, where the agreement is expected to be at machine precision;
  (b) a 4.5 km distance band on the same pattern, whose row totals range from
      about 30 to about 290: the configuration in which the two conventions
      separate, together with the same comparison after row-standardising the
      weights, which recovers the published quotient exactly.

Reading (b) matters for the interpretation of (a): agreement at 2.2e-16 there is
an algebraic identity evaluated in a different order of summation, not an
independent numerical validation. The estimate of the present estimator is
homogeneous of degree zero in a common rescaling of every weight (the property
test asserts this), but a row-specific rescaling moves the row's share of the
pooled ratio; row-standardising the weights removes the difference and makes the
two definitions coincide for any neighbour list.

R could not be run in the review environment, so this is a reimplementation of the
definition rather than a call into sfdep; the script prints the definition it
evaluates so that the comparison can be checked line by line.

    cd code && python evaluation/sfdep_parity.py

Output: printed comparison + results/sfdep_parity.csv
        + results/sfdep_parity_normalisation.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                    # noqa: E402
from scipy.sparse import csr_matrix                   # noqa: E402
from scipy.spatial import cKDTree                     # noqa: E402

from educeq import contract, diagnostics, estimator   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(ROOT), "results", "sfdep_parity.csv")
OUT_NORM = os.path.join(os.path.dirname(ROOT), "results",
                        "sfdep_parity_normalisation.csv")

N = 12000
K = 80                 # neighbours per subject, above the effective-size floor
BAND = 4.5             # km, the varying-row-total design of block (b)
SEED = 20261109


def data():
    """Two categories, B clustered around three centres so that CLQ != 1."""
    rng = np.random.default_rng(SEED)
    a = rng.uniform(0, 100, size=(N // 2, 2))
    centres = np.array([[30.0, 30.0], [70.0, 40.0], [45.0, 75.0]])
    pick = rng.integers(0, 3, size=N // 2)
    b = centres[pick] + rng.normal(scale=12.0, size=(N // 2, 2))
    coords = np.vstack([a, b])
    types = np.concatenate([np.zeros(N // 2, int), np.ones(N // 2, int)])
    return coords, types


def neighbour_list(coords, k=K):
    """k nearest neighbours, self excluded, returned as an (N, k) index array."""
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    return idx[:, 1:]


def knn_weights(nb, n):
    rows = np.repeat(np.arange(n), nb.shape[1])
    W = csr_matrix((np.ones(len(rows)), (rows, nb.ravel())), shape=(n, n))
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def band_weights(coords, radius):
    """Binary distance-band weights; rows differ in the number of neighbours."""
    tree = cKDTree(coords)
    d = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    n = len(coords)
    W = csr_matrix((np.ones(d.nnz), (d.row, d.col)), shape=(n, n))
    W.setdiag(0.0)
    W.eliminate_zeros()
    W.data[:] = 1.0
    return W


def clq_from_neighbours(types, nb):
    """Leslie and Kronenfeld's quotient, written from the definition.

    For source category A and target category B:

        CLQ_AB = [ sum_{i in A} (1/k_i) sum_{j in N(i)} w_ij * 1(t_j = B) ]
                 / [ m_A * p_B ]

    with w_ij = 1 for j in N(i). This is the unweighted average, over subjects of
    type A, of the within-neighbourhood share of B, divided by the global share.
    """
    n = len(types)
    p = np.bincount(types, minlength=2) / n
    share = np.zeros((2, 2))
    for A in (0, 1):
        idx = np.flatnonzero(types == A)
        lab = types[nb[idx]]                      # (m_A, k) labels of the neighbours
        for B in (0, 1):
            share[A, B] = (lab == B).mean(1).mean()   # mean over subjects, then over k
    return share / p[None, :], p


def clq_from_weights(types, W):
    """The same quotient for a weight matrix with rows of unequal total.

    Row standardisation is applied explicitly, one row at a time, and the
    per-subject shares are then averaged with equal weight. Subjects with an
    empty row carry no share and are dropped, as sfdep drops them.
    """
    n = len(types)
    p = np.bincount(types, minlength=2) / n
    tot = np.asarray(W.sum(1)).ravel()
    keep = tot > 0
    onehot = csr_matrix((np.ones(n), (np.arange(n), types)), shape=(n, 2))
    num = np.asarray((W @ onehot).todense())
    share = np.zeros((2, 2))
    for A in (0, 1):
        idx = np.flatnonzero((types == A) & keep)
        share[A] = (num[idx] / tot[idx][:, None]).mean(0)
    return share / p[None, :], p, int((~keep).sum())


def main():
    coords, types = data()
    nb = neighbour_list(coords)
    W = knn_weights(nb, len(types))
    ref, p = clq_from_neighbours(types, nb)

    dec = diagnostics.check(W, types, target=0, n_total=len(types))
    got = estimator.ceq_matrix(W, types, contract=dec)

    print("Parity with the published colocation-quotient definition")
    print("n = %d, neighbours per subject k = %d, seed = %d" % (len(types), K, SEED))
    print("global shares p = [%.4f, %.4f] ; verdict decidable = %s"
          % (p[0], p[1], dec.decidable))
    print()
    print("(a) 80-nearest-neighbour list, every row sums to 80")
    print("%-12s %18s %18s %12s" % ("pair", "reference (formula)", "ceq_matrix", "abs diff"))
    print("-" * 64)
    worst = 0.0
    out = []
    for A in (0, 1):
        for B in (0, 1):
            d = abs(got[A, B] - ref[A, B])
            worst = max(worst, d)
            print("%-12s %18.12f %18.12f %12.3e"
                  % ("%d -> %d" % (A, B), ref[A, B], got[A, B], d))
            out.append((A, B, ref[A, B], got[A, B], d))
    print("\nlargest absolute difference: %.3e" % worst)
    print("Every row of this design carries the same total weight, 80, so the")
    print("pooled form of Equation (1) and the subject-averaged form of the")
    print("published definition are the same expression; the residual is the order")
    print("of the floating-point summation.")

    with open(OUT, "w", encoding="utf8") as fh:
        fh.write("source,target,reference,ceq_matrix,abs_diff\n")
        for r in out:
            fh.write("%d,%d,%.12f,%.12f,%.3e\n" % r)

    # (b) the varying-row-total design
    Wb = band_weights(coords, BAND)
    tot = np.asarray(Wb.sum(1)).ravel()
    keep = tot > 0
    decb = diagnostics.check(Wb, types, target=0, n_total=len(types))
    refb, _, dropped = clq_from_weights(types, Wb)
    gotb = estimator.ceq_matrix(Wb, types, contract=decb)

    # row-standardised weights: recovers the published convention
    Wn = Wb.multiply(1.0 / np.maximum(tot, 1e-12)[:, None]).tocsr()
    Wn.eliminate_zeros()
    decn = diagnostics.check(Wn, types, target=0, n_total=len(types))
    gotn = estimator.ceq_matrix(Wn, types, contract=decn)

    print()
    print("(b) %.1f km distance band, rows of unequal total" % BAND)
    print("row totals: min %d, median %d, max %d over %d subjects with a non-empty"
          % (tot[keep].min(), np.median(tot[keep]), tot[keep].max(), keep.sum()))
    print("neighbourhood (%d subjects have none and are dropped, as sfdep does);"
          % dropped)
    print("decidable = %s" % decb.decidable)
    print()
    print("%-8s %16s %16s %12s %16s %12s"
          % ("pair", "published", "ceq raw W", "rel.diff",
             "ceq row-std W", "abs diff"))
    print("-" * 88)
    rows = []
    for A in (0, 1):
        for B in (0, 1):
            rel = (gotb[A, B] - refb[A, B]) / refb[A, B]
            d = abs(gotn[A, B] - refb[A, B])
            print("%-8s %16.6f %16.6f %11.2f%% %16.6f %12.3e"
                  % ("%d -> %d" % (A, B), refb[A, B], gotb[A, B], 100 * rel,
                     gotn[A, B], d))
            rows.append((A, B, refb[A, B], gotb[A, B], rel, gotn[A, B], d))
    print()
    print("With rows of unequal total the two conventions separate: the pooled form")
    print("weights a subject by the total weight of its row, the published form")
    print("weights every subject equally. Row-standardising the weights before the")
    print("call removes the difference and reproduces the published quotient to")
    print("machine precision in every pair.")

    with open(OUT_NORM, "w", encoding="utf8") as fh:
        fh.write("source,target,published,ceq_raw,rel_diff,ceq_row_standardised,"
                 "abs_diff_row_standardised\n")
        for r in rows:
            fh.write("%d,%d,%.12f,%.12f,%.6f,%.12f,%.3e\n" % r)
    print("\nwritten: %s" % OUT)
    print("written: %s" % OUT_NORM)


if __name__ == "__main__":
    main()
