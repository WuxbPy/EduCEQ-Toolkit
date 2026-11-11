# -*- coding: utf-8 -*-
"""EduCEQ attributive kernels and kernel composition

This module supplies the attributive side, which the two kernel families used in
the paper's empirical work do not cover:

  * ordinal_kernel     ordinal attribute kernel w = max(0, 1 - |dz| / r)
  * categorical_kernel categorical attribute kernel w = I(z_i = z_j)
  * combine            multiplicative composition; a component set to the
                       constant one leaves that dimension unconstrained

Design rationale (Section 3.2 of the paper)
-------------------------------------------
The kernel is only required to be non-negative; it need not be symmetric and need
not be a distance. An attributive kernel can therefore be built without any
coordinates, and that is what makes EduCEQ (attribute-similarity kernel, with no
geometric component) possible in the first place.

Division of labour with _core
-----------------------------
_core.build_weights  builds the geometric (distance-decay) kernel from coordinates
this module          builds attributive kernels and their composition

Empirical use (the same-data, two-neighbourhood comparison of Section 7)
------------------------------------------------------------------------
On a single POI dataset:
  W_geo  = _core.build_weights(coords, b)    geometric neighbourhood
  W_attr = ordinal_kernel(z, r)              attributive neighbourhood
  W_both = combine(W_geo, W_attr)            combined neighbourhood
Feeding all three to _core.clq_global gives comparable enrichment matrices.
"""
import numpy as np
from scipy import sparse


def ordinal_kernel(z, r, dtype=np.float64):
    """Ordinal attribute kernel w_ij = max(0, 1 - |z_i - z_j| / r), zero diagonal.

    z : (n,) numeric attribute, continuous or ordinal
    r : attribute radius; as r grows the weights tend to one, leaving that
        dimension unconstrained
    """
    z = np.asarray(z, dtype=dtype)
    r = float(r)
    n = len(z)
    if r <= 0:
        raise ValueError("attribute radius r must be positive")
    d = np.abs(z[:, None] - z[None, :])
    W = np.maximum(0.0, 1.0 - d / r)
    np.fill_diagonal(W, 0.0)
    return sparse.csr_matrix(W)


def categorical_kernel(codes):
    """Categorical attribute kernel w_ij = I(c_i = c_j), zero diagonal.

    codes : (n,) integer codes of the categorical attribute
    Note: if the attribute is fully nested within the target type, meaning each
    category contains a single type, then every cross-type weight is zero and the
    kernel is degenerate. Check before building it.
    """
    c = np.asarray(codes)
    n = len(c)
    W = (c[:, None] == c[None, :]).astype(np.float64)
    np.fill_diagonal(W, 0.0)
    return sparse.csr_matrix(W)


def combine(*mats):
    """Multiplicative (element-wise) composition of several kernels.

    A component set to the constant one leaves that dimension unconstrained, so
    the composition reduces to the product of the remaining components.
    """
    if not mats:
        raise ValueError("at least one kernel component is required")
    out = mats[0]
    for m in mats[1:]:
        out = out.multiply(m) if sparse.issparse(out) else np.multiply(out, m)
    return sparse.csr_matrix(out) if not sparse.issparse(out) else out.tocsr()


def is_degenerate(W, types, tol=1e-14):
    """Degeneracy check: is every cross-type weight of some source type zero?

    Returns (is_degenerate, explanation). A nested attributive kernel is the
    typical trigger.
    """
    W = sparse.csr_matrix(W)
    K = int(types.max()) + 1
    bad = []
    for a in range(K):
        idx = np.where(types == a)[0]
        if len(idx) == 0:
            continue
        sub = W[idx]
        others = [b for b in range(K) if b != a]
        if others and float(np.abs(sub[:, others]).sum()) < tol:
            bad.append(a)
    return (len(bad) > 0), ("degenerate: every cross-type weight of source type %s is zero" % bad if bad else "not degenerate")


def support_matrices(W, types, K=None):
    """The three support measures of Section 4.2, as K x K matrices of medians.

    For a source type A and a target category B the manuscript distinguishes

        raw support       n_AB = median over i in A of #{j in B : W_ij > 0}
        weighted support  S_AB = median over i in A of sum_{j in B} W_ij
        effective size    e_AB = median over i in A of (sum w)^2 / sum w^2

    Only the third is the scale on which the variance of the ratio depends
    (Section 4.2 and Table 11). The first two are reported because they are what
    users see when they think in neighbours, and because the weight sum is what
    earlier versions of this function returned; keeping all three visible is what
    lets a reader see where the three differ.

    Returns a dict with the three matrices, the three coverage proportions (each
    a median over all subjects divided by the number of objects) and the median
    share of the single largest weight in a neighbourhood.
    """
    W = sparse.csr_matrix(W).tocsr()
    n = W.shape[0]
    K = int(types.max()) + 1 if K is None else K
    onehot = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), types)), shape=(n, K))

    # raw counts: replace every stored weight by one
    B = W.copy()
    B.data = np.ones_like(B.data)
    raw = np.asarray((B @ onehot).todense())            # (n, K) neighbour counts
    # weighted sums, and sums of squares, per target category
    S = np.asarray((W @ onehot).todense())
    W2 = W.multiply(W)
    S2 = np.asarray((W2 @ onehot).todense())
    with np.errstate(divide="ignore", invalid="ignore"):
        E = np.where(S2 > 0, S ** 2 / S2, 0.0)          # (n, K) row effective size

    def med(X):
        M = np.zeros((K, K))
        for a in range(K):
            idx = np.where(types == a)[0]
            if len(idx) == 0:
                continue
            for b in range(K):
                M[a, b] = float(np.median(X[idx, b]))
        return M

    n_mat, S_mat, e_mat = med(raw), med(S), med(E)

    tot_raw = raw.sum(1)                                # (n,) all categories
    tot_S = S.sum(1)
    tot_e = E.sum(1)
    row_max = np.asarray(W.max(axis=1).todense()).ravel()
    row_sum = np.asarray(W.sum(1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        conc = np.where(row_sum > 0, row_max / row_sum, np.nan)

    return dict(raw=n_mat, weighted=S_mat, ess=e_mat,
                rho_raw=float(np.median(tot_raw)) / n,
                rho_weighted=float(np.median(tot_S)) / n,
                rho_ess=float(np.median(tot_e)) / n,
                concentration=float(np.nanmedian(conc)))


def neighbour_diagnostics(W, types, K=None):
    """Deprecated: use :func:`support_matrices`.

    Retained so that external scripts written against version 1.0.0 do not break
    silently. It returns ``(weighted support matrix, coverage proportion)``,
    i.e. the quantities earlier versions called kappa and rho, and it is not
    exported in the public API.
    """
    s = support_matrices(W, types, K=K)
    return s["weighted"], s["rho_weighted"]
