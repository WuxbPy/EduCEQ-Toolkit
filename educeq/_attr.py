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


def neighbour_diagnostics(W, types, K=None):
    """Neighbourhood count kappa and coverage proportion rho from a weight matrix
    (the decidability criteria of Section 4 of the paper).

    kappa : K x K matrix of the median number of target-type objects in the
            neighbourhood of each source point
    rho   : median count of all types in a neighbourhood, divided by the number
            of objects
    """
    W = sparse.csr_matrix(W).tocsr()
    n = W.shape[0]
    K = int(types.max()) + 1 if K is None else K
    onehot = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), types)), shape=(n, K))
    R = np.asarray((W @ onehot).todense())          # (n, K) weight sums per type
    tot = np.asarray(W.sum(1)).ravel()              # total weight per point
    kap = np.zeros((K, K))
    for a in range(K):
        idx = np.where(types == a)[0]
        if len(idx) == 0:
            continue
        for b in range(K):
            kap[a, b] = float(np.median(R[idx, b]))
    rho = float(np.median(tot)) / n
    return kap, rho
