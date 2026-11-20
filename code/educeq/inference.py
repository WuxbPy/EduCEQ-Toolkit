# -*- coding: utf-8 -*-
"""educeq.inference — significance testing and multiple comparison (module three)

This module corresponds to Section 2.3 of the paper.

**The released code provides one channel: the permutation test, with support for
stratified (conditional) permutation.**

Stratified permutation is not optional. When the kernel uses attribute
information, permuting labels globally destroys the existing relation between
labelling and attributes, which pushes the actual type-one error far above the
nominal level: Section 2.3 of the paper reports 0.29 against a nominal 0.05
for the attribute configuration, and 1.00 for a composition design in which the
permutation is unconditional. Section 5.8 measures both. Stratified permutation permutes independently within each conditioning
stratum, preserves the joint margins and breaks only the association of
interest.

Section 2.3 also describes an asymptotic-null channel. That channel is **not
implemented in the released code**; see Section 7, Limitations. This module
reports the number of available channels honestly through ``channel_report`` and
never substitutes one silently for another.

A p-value is a statement about an estimate, so the public entry point requires
the verdict that authorised that estimate (Section 3.2, Section 4.7): testing a
number the contract refused would attach a significance claim to an
uninterpretable quantity. The arithmetic is reachable without a verdict only via
``_permutation_unchecked``, which is private.
"""
import numpy as np
from ._core import permutation_test, bh_fdr, by_fdr
from .contract import require

__all__ = ["permutation", "fdr", "fdr_by", "channel_report"]


def _permutation_unchecked(W, types, n_perm=999, seed=0, strata=None):
    """Arithmetic of the permutation test, with no verdict. Private."""
    types = np.asarray(types)
    K = int(types.max()) + 1
    labels = list(range(K))
    C_obs, mu, sd, z, null = permutation_test(
        W, types, labels, n_perm=n_perm, seed=seed, block=strata)
    return dict(observed=C_obs, null_mean=mu, null_sd=sd, z=z, null=null)


def permutation(W, types, *, contract=None, n_perm=999, seed=0, strata=None):
    """Permutation test. When strata is given, permutation is confined within
    strata (conditional permutation).

    Returns a dict with the observed matrix, the null mean and standard
    deviation, the z value and the permutation distribution.

    ``contract`` is mandatory: the verdict returned by
    :func:`educeq.diagnostics.check` for the same ``W`` and ``types``.
    """
    require(contract, W, types)
    return _permutation_unchecked(W, types, n_perm=n_perm, seed=seed,
                                  strata=strata)


def fdr(p_values, alpha=0.05):
    """Benjamini-Hochberg false discovery rate control; returns (q values, number
    of rejections). Valid when the tests are independent or positively dependent;
    use fdr_by when that cannot be assumed."""
    q, k = bh_fdr(np.asarray(p_values, float), alpha=alpha)
    return q, int(k)


def fdr_by(p_values, alpha=0.05):
    """Benjamini-Yekutieli false discovery rate control; returns (q values, number
    of rejections). Valid under arbitrary dependence among the tests, at the cost
    of the harmonic factor in the threshold. Section 5.8 of the paper reports
    empirical size and power of both under spatially dependent tests."""
    q, k = by_fdr(np.asarray(p_values, float), alpha=alpha)
    return q, int(k)


def channel_report():
    """Report honestly which inference channels the released code provides."""
    return dict(available=["conditional_permutation"],
                not_implemented=["asymptotic_null"],
                note=("The asymptotic-null channel described in the paper "
                      "is part of the general formulation and is not included "
                      "in the released code; no result in this paper relies on it."))
