# -*- coding: utf-8 -*-
"""
Core of the CEQ measure (the general form of the co-enrichment quotient and of
the co-location quotient)
================================================================================

Where this sits in the paper
----------------------------
Section 2.1   general form. W is only required to be non-negative; it need not
              be symmetric and need not be a distance. One implementation thus
              covers geometric and attributive neighbourhoods alike, the
              co-location quotient being the special case of a distance kernel.
Sections 2.1 and 2.3   clq_global gives the enrichment matrix and
              clq_first_order the first-order term; their quotient is the
              second-order term, an exact decomposition.
Sections 2.4, 2.5 and 4   neighborhood_fraction and kappa_matrix give the
              neighbourhood count kappa and the coverage proportion rho, the
              decidability criteria; permutation_test supports stratified
              permutation and bh_fdr controls multiple comparison.

A note on naming
----------------
The functions keep the clq_* prefix so that they can be matched one-to-one
against the original analysis scripts. In the terminology of the paper the
result under a distance kernel is the co-location quotient (CLQ), while the
general form is called the co-enrichment quotient (CEQ).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse


def standardize(X):
    X = np.asarray(X, dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def build_weights(coords, b, cut=3.0):
    """Return the sparse weight matrix W: a Gaussian kernel truncated at cut*b,
    with no diagonal."""
    tree = cKDTree(coords)
    D = tree.sparse_distance_matrix(tree, max_distance=cut * b, output_type='coo_matrix')
    r, c, d = D.row, D.col, D.data
    mask = (r != c) & (d > 0)
    r, c, d = r[mask], c[mask], d[mask]
    w = np.exp(-(d ** 2) / (2 * b ** 2))
    return sparse.csr_matrix((w, (r, c)), shape=(len(coords), len(coords)))


def neighborhood_fraction(coords, b, cut=1.0):
    """rho, the mean neighbourhood capture proportion: mean count within radius b,
    divided by N."""
    tree = cKDTree(coords)
    k = np.array([len(x) - 1 for x in tree.query_ball_point(coords, b)])
    return k.mean() / len(coords), k.mean()


def clq_global(W, types, labels, alpha=0.0, attr_sim=None):
    """
    Compute the CEQ matrix.
    W: (N, N) sparse spatial weights
    types: (N,) integer type codes
    labels: list of type labels, of length K
    when alpha > 0 an attributive kernel is applied:
    W_eff = W * attr_sim**alpha, with attr_sim (N, N), sparse or dense
    Returns the (K, K) matrix and the per-source total weight used for row
    normalisation.
    """
    K = len(labels)
    N = len(types)
    Wl = W
    if alpha > 0 and attr_sim is not None:
        Wl = W.multiply(attr_sim ** alpha) if sparse.issparse(attr_sim) else W.multiply(attr_sim ** alpha)
    # denominator: total weight of each source point
    Wl = Wl.tocsr()
    tot = np.asarray(Wl.sum(1)).ravel()          # total weight per point
    # weight from each point to each type = Wl @ onehot
    onehot = sparse.csr_matrix((np.ones(N), (np.arange(N), types)), shape=(N, K))
    # (N, K): weight from point i to type B
    contrib = Wl @ onehot
    contrib = np.asarray(contrib.todense()) if sparse.issparse(contrib) else np.asarray(contrib)
    p = np.bincount(types, minlength=K) / N
    C = np.zeros((K, K))
    for A in range(K):
        idx = (types == A)
        num = contrib[idx].sum(0)               # summed over B
        den = tot[idx].sum() * p
        with np.errstate(divide='ignore', invalid='ignore'):
            C[A] = np.where(den > 0, num / den, np.nan)
    return C, tot


def clq_first_order(coords, types, K, b, grid_n=120, bounds=None):
    """
    First-order term: estimated by kernel smoothing of intensity fields, from a
    gridded count field convolved with a Gaussian.
    Returns the (K, K) first-order matrix.
    """
    coords = np.asarray(coords, float)
    lo = coords.min(0) if bounds is None else bounds[0]
    hi = coords.max(0) if bounds is None else bounds[1]
    lo = lo - 3 * b; hi = hi + 3 * b
    d = coords.shape[1]
    # grid centres
    axes = [np.linspace(lo[k], hi[k], grid_n) for k in range(d)]
    # gridded count field
    digits = []
    for k in range(d):
        idx = np.clip(((coords[:, k] - lo[k]) / (hi[k] - lo[k]) * (grid_n - 1)).round().astype(int), 0, grid_n - 1)
        digits.append(idx)
    shape = tuple([grid_n] * d)
    lam = np.zeros((K,) + shape)
    lam_tot = np.zeros(shape)
    flat = np.ravel_multi_index(digits, shape)
    for A in range(K):
        cnt = np.bincount(flat[types == A], minlength=grid_n ** d).reshape(shape)
        lam[A] = cnt
    lam_tot = lam.sum(0)
    # Gaussian convolution, applied separably
    lam_s = [gauss_blur_sep(lam[A], b, axes) for A in range(K)]
    tot_s = gauss_blur_sep(lam_tot, b, axes)
    dx = np.prod([(hi[k] - lo[k]) / (grid_n - 1) for k in range(d)])
    A_mat = np.zeros((K, K))
    for a in range(K):
        w_a = lam[a] * dx
        den = (w_a * tot_s).sum()
        for bb in range(K):
            A_mat[a, bb] = (w_a * lam_s[bb]).sum() / den if den > 0 else np.nan
    # first-order term
    p = np.array([ (types == A).sum() for A in range(K) ], float); p /= p.sum()
    return A_mat, lam_s, tot_s, dx, lo, hi, grid_n


def clq_first_order_grouped(coords, types, K, b, groups=None, grid_n=50, bounds=None):
    """
    First-order term, from a count field convolved with a Gaussian. Any number
    of dimensions d is supported.
    groups: (N,) group labels. When given, the kernel acts within groups, that is,
    an indicator kernel multiplied by a Gaussian kernel.
    """
    from scipy.ndimage import gaussian_filter
    coords = np.asarray(coords, float)
    N, d = coords.shape
    if bounds is None:
        lo = coords.min(0) - 3*b; hi = coords.max(0) + 3*b
    else:
        lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
    shape = (grid_n,)*d
    digs = np.stack([np.clip(((coords[:,k]-lo[k])/(hi[k]-lo[k])*(grid_n-1)).round().astype(int),
                              0, grid_n-1) for k in range(d)], 1)
    flat = np.ravel_multi_index(digs.T, shape)
    dx = np.prod([(hi[k]-lo[k])/(grid_n-1) for k in range(d)])
    sigma = [b/((hi[k]-lo[k])/(grid_n-1)) for k in range(d)]
    A_sum = np.zeros((K, K)); D_a = np.zeros(K)
    gs = [None] if groups is None else np.unique(groups)
    ncell = grid_n**d
    for g in gs:
        mg = np.ones(N, bool) if g is None else (groups == g)
        lam = np.zeros((K,)+shape)
        for a in range(K):
            lam[a] = np.bincount(flat[mg & (types == a)], minlength=ncell).reshape(shape)
        tot_s = gaussian_filter(lam.sum(0), sigma=sigma, mode='constant')
        lam_s = np.array([gaussian_filter(lam[a], sigma=sigma, mode='constant') for a in range(K)])
        for a in range(K):
            wa = lam[a]*dx
            D_a[a] += (wa*tot_s).sum()
            for b2 in range(K):
                A_sum[a, b2] += (wa*lam_s[b2]).sum()
    p = np.bincount(types, minlength=K)/len(types)
    return A_sum/(D_a[:, None]*p[None, :])


def gauss_blur_sep(field, b, axes):
    """Separable Gaussian convolution, with the scale taken from the grid axes."""
    from scipy.ndimage import gaussian_filter
    sig = [b / (axes[k][1] - axes[k][0]) for k in range(len(axes))]
    return gaussian_filter(field, sigma=sig, mode='constant')


def kappa_matrix(coords, types, K, b, cut=1.0):
    """kappa_AB(b): median expected neighbourhood count of the target type within
    radius b, computed from a sparse binary adjacency matrix."""
    tree = cKDTree(coords)
    D = tree.sparse_distance_matrix(tree, max_distance=b, output_type='coo_matrix')
    r, c, d = D.row, D.col, D.data
    mask = (d > 0)
    Bm = sparse.csr_matrix((np.ones(mask.sum()), (r[mask], c[mask])), shape=(len(coords), len(coords)))
    onehot = sparse.csr_matrix((np.ones(len(types)), (np.arange(len(types)), types)),
                               shape=(len(coords), K))
    R = np.asarray((Bm @ onehot).todense())   # (N,K)
    kap = np.zeros((K, K))
    for A in range(K):
        idx = np.where(types == A)[0]
        for B in range(K):
            kap[A, B] = np.median(R[idx, B]) if len(idx) else np.nan
    return kap


def permutation_test(W, types, labels, n_perm=999, seed=0, block=None):
    """
    Global label permutation test.
    block: when (N,) group labels are given, permutation is confined within
    groups, which is the stratified permutation.
    """
    rng = np.random.default_rng(seed)
    K = len(labels)
    C_obs, _ = clq_global(W, types, labels)
    N = len(types)
    null = np.zeros((n_perm, K, K))
    for t in range(n_perm):
        if block is None:
            tp = rng.permutation(types)
        else:
            tp = types.copy()
            for g in np.unique(block):
                idx = np.where(block == g)[0]
                tp[idx] = rng.permutation(types[idx])
        Ct, _ = clq_global(W, tp, labels)
        null[t] = Ct
    mu = null.mean(0); sd = null.std(0)
    z = (C_obs - mu) / np.where(sd > 0, sd, np.nan)
    return C_obs, mu, sd, z, null


def bh_fdr(p, alpha=0.05):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1)) / n
    sig = ranked <= thresh
    k = np.max(np.where(sig)[0]) + 1 if sig.any() else 0
    q = np.empty(n)
    q[order] = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.clip(q, 0, 1)
    return q, k
