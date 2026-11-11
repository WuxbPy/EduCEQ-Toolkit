# -*- coding: utf-8 -*-
"""educeq.estimator — enrichment values and decomposition (module two)

This module corresponds to Sections 2.1 and 2.3 of the paper. At its centre is
one identity:

    CEQ_AB = CEQ1_AB * CEQ2_AB

The first term is estimated from intensity fields by kernel smoothing and the
second is defined as their quotient, so the identity **holds on any data**; it
is not an approximation. Its value is interpretive: a second term close to one
says that the observed departure is explained by the marginal distribution.

Every entry point that can return an estimate requires a verdict
---------------------------------------------------------------
Section 3.2 of the paper claims that no call returns a number without its
conditions of use. Version 1.0.0 did not honour that claim: ``ceq_matrix`` took
no verdict and ``decompose`` accepted ``check=False``. Both are gone. The public
functions here take a ``contract`` argument, which must be the object returned by
:func:`educeq.diagnostics.check` for the same data; see :mod:`educeq.contract`
for the exception types. ``ceq_matrix`` passes through
:func:`educeq.contract.require` as its first statement, so an omitted verdict
raises ``ContractRequired``. ``decompose`` has one further path, because it is
also the documented single-call entry point of Section 3.2: when ``contract`` is
omitted it issues a verdict itself and then behaves identically, returning the
diagnostics and no estimate when that verdict is negative. Neither path reaches
the arithmetic without a verdict. The arithmetic is available with no verdict at
all only through :func:`_decompose_unchecked`, which is private, absent from
``__all__``, and used by the tests and the evaluation scripts that need an
unchecked baseline rather than an estimate.

The module also provides a numerical self-check of the three-way identity,
comparing CEQ_{a;(b,c)} against CEQ_{a;c} * CEQ_{a;b|c} (Section 2.3 of the paper
states that the discrepancy is at machine precision; this module supplies the
checking function built on the three-way weighted counts).
"""
import numpy as np
from ._core import clq_global, clq_first_order_grouped
from .contract import ContractMismatch, require

__all__ = ["ceq_matrix", "ceq_first_order", "ceq_second_order",
           "decompose", "normalization_ok"]


def _matrix(W, types, labels=None, alpha=0.0, attr_sim=None):
    """Arithmetic only: the enrichment matrix. Not a public entry point."""
    types = np.asarray(types)
    K = int(types.max()) + 1
    labels = list(range(K)) if labels is None else labels
    C, tot = clq_global(W, types, labels, alpha=alpha, attr_sim=attr_sim)
    return C


def ceq_matrix(W, types, *, contract=None, labels=None, alpha=0.0, attr_sim=None):
    """Enrichment matrix.

    Parameters
    ----------
    contract : Decidability
        The verdict issued by :func:`educeq.diagnostics.check` for ``W`` and
        ``types``. Required: this call performs no check of its own, so the
        verdict must already exist. A missing verdict raises
        :class:`educeq.contract.ContractRequired`, one issued for other data
        raises :class:`ContractMismatch`, and a negative one raises
        :class:`ContractRefused`.
    """
    require(contract, W, types)
    return _matrix(W, types, labels=labels, alpha=alpha, attr_sim=attr_sim)


def ceq_first_order(coords, types, bandwidth, grid_n=50, groups=None, bounds=None):
    """First-order term, estimated from a gridded count field convolved with a
    Gaussian of the same bandwidth (supports indicator kernels over groups and
    any number of dimensions).

    This operates on intermediate quantities (coordinates and labels) and returns
    a term of the decomposition rather than a stand-alone estimate of association,
    so it does not require a verdict; passing its output to a table without the
    second-order term would be a misuse described in Section 5.7.
    """
    types = np.asarray(types)
    K = int(types.max()) + 1
    return clq_first_order_grouped(coords, types, K, bandwidth,
                                   groups=groups, grid_n=grid_n, bounds=bounds)


def ceq_second_order(C, C1):
    """Second-order term, defined as the quotient: CEQ2 = CEQ / CEQ1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(C1 > 0, C / C1, np.nan)


def _decompose_unchecked(W, coords, types, bandwidth, grid_n=50, groups=None):
    """Arithmetic of the decomposition with no verdict. Private on purpose.

    Used by the tests of the identities and by the counterfactual experiment of
    Section 5.3, which has to estimate in a regime the contract refuses in order
    to show what the refusal buys. Anything else that calls this has stepped
    outside the contract and the paper says so.
    """
    C = _matrix(W, types)
    C1 = ceq_first_order(coords, types, bandwidth, grid_n=grid_n, groups=groups)
    C2 = ceq_second_order(C, C1)
    fin = np.isfinite(C) & np.isfinite(C1) & np.isfinite(C2)
    identity_ok = bool(np.allclose(C[fin], (C1 * C2)[fin], atol=1e-9))
    return dict(decidable=True, C=C, C1=C1, C2=C2, identity_ok=identity_ok)


def decompose(W, coords, types, bandwidth, grid_n=50, groups=None,
              target=None, contract=None, ess_min=30.0, rho_max=0.01,
              n_total=None):
    """Return the matrix, the first- and second-order terms, the identity check
    and the decidability diagnostics in a single call.

    This is the documented entry point of the package and the place where the
    runtime contract is enforced. When ``contract`` is None the call runs
    :func:`educeq.diagnostics.check` itself and, if the conditions do not hold,
    **returns no estimate at all**, yielding only
    ``{"decidable": False, "failed": [...], "advice": ..., "diagnostics": {...}}``.
    When the conditions do hold, the ordinary result carries a ``diagnostics``
    field, so the number and the conditions under which it may be used cannot be
    passed around separately (the return-value contract of Section 3.2).

    When ``contract`` is supplied — for instance to reuse a verdict already
    computed for a bandwidth sweep — it is validated against the data and against
    ``target``, so a verdict computed on one configuration cannot authorise an
    estimate on another.
    """
    types = np.asarray(types)
    if contract is None:
        from .diagnostics import check as _check
        verdict = _check(W, types, target=target, ess_min=ess_min,
                         rho_max=rho_max, n_total=n_total)
    else:
        if target is not None and int(target) != int(contract.target):
            raise ContractMismatch(
                "the verdict was issued for target=%s, not target=%s"
                % (contract.target, target))
        verdict = require(contract, W, types)

    if not verdict.decidable:
        return dict(decidable=False,
                    diagnostics=verdict.as_dict(),
                    failed=list(verdict.failed),
                    advice=verdict.advice)

    out = _decompose_unchecked(W, coords, types, bandwidth, grid_n=grid_n,
                              groups=groups)
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
