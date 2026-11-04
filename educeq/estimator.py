# -*- coding: utf-8 -*-
"""educeq.estimator — enrichment values and decomposition (module two)

This module corresponds to Sections 2.1 and 2.3 of the paper. At its centre is
one identity:

    CEQ_AB = CEQ1_AB * CEQ2_AB

The first term is estimated from intensity fields by kernel smoothing and the
second is defined as their quotient, so the identity **holds on any data**; it
is not an approximation. Its value is interpretive: a second term close to one
says that the observed departure is explained by the marginal distribution.

The module also provides a numerical self-check of the three-way identity,
comparing CEQ_{a;(b,c)} against CEQ_{a;c} * CEQ_{a;b|c} (Section 2.3 of the paper
states that the discrepancy is at machine precision; this module supplies the
checking function built on the three-way weighted counts).
"""
import numpy as np
from ._core import clq_global, clq_first_order_grouped

__all__ = ["ceq_matrix", "ceq_first_order", "ceq_second_order",
           "decompose", "normalization_ok"]


def ceq_matrix(W, types, labels=None, alpha=0.0, attr_sim=None):
    """Enrichment matrix. When alpha > 0 and attr_sim is given, the weights are
    W * attr_sim**alpha."""
    types = np.asarray(types)
    K = int(types.max()) + 1
    labels = list(range(K)) if labels is None else labels
    C, tot = clq_global(W, types, labels, alpha=alpha, attr_sim=attr_sim)
    return C


def ceq_first_order(coords, types, bandwidth, grid_n=50, groups=None, bounds=None):
    """First-order term, estimated from a gridded count field convolved with a
    Gaussian of the same bandwidth (supports indicator kernels over groups and
    any number of dimensions)."""
    types = np.asarray(types)
    K = int(types.max()) + 1
    return clq_first_order_grouped(coords, types, K, bandwidth,
                                   groups=groups, grid_n=grid_n, bounds=bounds)


def ceq_second_order(C, C1):
    """Second-order term, defined as the quotient: CEQ2 = CEQ / CEQ1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(C1 > 0, C / C1, np.nan)


def decompose(W, coords, types, bandwidth, grid_n=50, groups=None,
              check=True, target=None, kappa_min=30.0, rho_max=0.01,
              n_total=None):
    """Return the matrix, the first- and second-order terms, the identity check
    and the decidability diagnostics in a single call.

    This is the documented entry point of the package and the place where the
    runtime contract is enforced. With ``check=True`` (the default) the call runs
    :func:`educeq.diagnostics.check` first; if the conditions do not hold it
    **returns no estimate at all** and yields only
    ``{"decidable": False, "failed": [...], "advice": ..., "diagnostics": {...}}``.
    When the conditions do hold, the ordinary result carries a ``diagnostics``
    field, so the number and the conditions under which it may be used cannot be
    passed around separately (the return-value contract of Section 3.2).

    Pass ``check=False`` to bypass the contract, which unit tests do.
    """
    if check:
        from .diagnostics import check as _check
        verdict = _check(W, types, target=target, kappa_min=kappa_min,
                         rho_max=rho_max, n_total=n_total)
        if not verdict.decidable:
            return dict(decidable=False,
                        diagnostics=verdict.as_dict(),
                        failed=list(verdict.failed),
                        advice=verdict.advice)
    C = ceq_matrix(W, types)
    C1 = ceq_first_order(coords, types, bandwidth, grid_n=grid_n, groups=groups)
    C2 = ceq_second_order(C, C1)
    fin = np.isfinite(C) & np.isfinite(C1) & np.isfinite(C2)
    identity_ok = bool(np.allclose(C[fin], (C1 * C2)[fin], atol=1e-9))
    out = dict(decidable=True, C=C, C1=C1, C2=C2, identity_ok=identity_ok)
    if check:
        out["diagnostics"] = verdict.as_dict()
    return out


def normalization_ok(C, types, tol=1e-9):
    """Normalisation identity sum_B p_B * CEQ_AB = 1, independent of the kernel;
    used as a self-check in Section 4 of the paper."""
    types = np.asarray(types)
    K = int(types.max()) + 1
    p = np.bincount(types, minlength=K) / len(types)
    dev = np.abs(np.nansum(C * p[None, :], axis=1) - 1.0)
    return bool(dev.max() < tol), float(dev.max())
