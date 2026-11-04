# -*- coding: utf-8 -*-
"""educeq.diagnostics — neighbourhood diagnostics and decidability (module four)

This module corresponds to Section 4 of the paper, and it is the one place where
the tool works against its own purpose: **it does not promise an answer**.

Two preconditions pull in opposite directions:
  * condition one   the median number of target-type objects in a neighbourhood
                    is at least 30 (too few neighbours makes the estimate
                    unstable)
  * condition two   the population coverage proportion of a neighbourhood is at
                    most 0.01 (too wide a neighbourhood compresses the ratio
                    mechanically towards one)

The two are linked by kappa_B ~ N_B * rho, and taken together they give the
feasibility condition N_B >= 3000 of the paper, which depends on neither the
study area nor the bandwidth.

Failure semantics: when the conditions do not hold the call **does not raise**;
it returns a status object. Undecidability is a research finding, not a program
error, and the user should be able to put it into a report rather than be
interrupted by the library.
"""
import numpy as np
from ._core import neighborhood_fraction, kappa_matrix
from ._attr import neighbour_diagnostics

__all__ = ["neighbourhood_count", "coverage", "check", "Decidability",
           "spectral_shape"]


class Decidability:
    """Outcome of the decidability check. Nothing is raised; the verdict lives in
    the decidable field."""

    def __init__(self, decidable, n_target, kappa, rho, failed=None, advice=None):
        self.decidable = bool(decidable)
        self.n_target = int(n_target)
        self.kappa = float(kappa)
        self.rho = float(rho)
        self.failed = failed or []
        self.advice = advice or ""

    def __repr__(self):
        if self.decidable:
            return ("Decidability(decidable=True, N_B=%d, kappa_B=%.1f, rho=%.5f)"
                    % (self.n_target, self.kappa, self.rho))
        return ("Decidability(decidable=False, N_B=%d, kappa_B=%.1f, rho=%.5f, "
                "failed=%s, advice=%s)"
                % (self.n_target, self.kappa, self.rho, self.failed, self.advice))

    def as_dict(self):
        return dict(decidable=self.decidable, n_target=self.n_target,
                    kappa=self.kappa, rho=self.rho,
                    failed=list(self.failed), advice=self.advice)


def neighbourhood_count(W, types):
    """The kappa_AB matrix: median number of target-type B objects in the
    neighbourhood of a source-type A object."""
    kap, _ = neighbour_diagnostics(W, np.asarray(types))
    return kap


def coverage(W):
    """rho: population coverage proportion = median neighbourhood size divided by
    the number of objects."""
    n = W.shape[0]
    tot = np.asarray(W.sum(axis=1)).ravel()
    return float(np.median(tot)) / n


def check(W, types, target=None, kappa_min=30.0, rho_max=0.01, n_total=None):
    """Three-step decidability check.

    Step 1  the target category has N_B >= kappa_min / rho_max objects
            (3000 by default). If this fails the check stops here.
    Step 2  the median neighbourhood count is at least 30.
    Step 3  the empirical coverage proportion is at most 0.01, measured on the
            data rather than estimated from an area ratio.

    Returns a Decidability object; when decidable is False it also carries the
    list of failed steps and an advice string.
    """
    types = np.asarray(types)
    n = len(types) if n_total is None else int(n_total)
    kap, rho = neighbour_diagnostics(W, types)
    K = int(types.max()) + 1

    failed = []
    # step 1: size of the target category
    n_target = int((types == target).sum()) if target is not None else int(types.size)
    threshold = int(kappa_min / rho_max)
    if n_target < threshold:
        failed.append("step1:n_target")
        return Decidability(
            False, n_target, float(np.nanmedian(kap)), rho, failed,
            "target category has %d objects, below the threshold of %d; "
            "widen the time span or the study area. No parameter setting can bypass this." % (n_target, threshold))

    # steps 2 and 3
    k_med = float(np.nanmedian(kap))
    if k_med < kappa_min:
        failed.append("step2:kappa")
    if rho > rho_max:
        failed.append("step3:rho")

    if failed:
        if "step2:kappa" in failed and "step3:rho" in failed:
            advice = ("both conditions fail: satisfying the neighbourhood count requires a larger "
                      "neighbourhood, satisfying the coverage proportion requires a smaller one, "
                      "so the feasible interval is empty.")
        elif "step2:kappa" in failed:
            advice = "insufficient neighbourhood count: increase the bandwidth or collect more data."
        else:
            advice = "coverage proportion too high: reduce the bandwidth."
        return Decidability(False, n_target, k_med, rho, failed, advice)

    return Decidability(True, n_target, k_med, rho)


def spectral_shape(bandwidths, values):
    """Spectral shape check: does the scanned curve have an interior maximum?

    Returns a dict with the monotonicity flag, whether the maximum sits at an
    endpoint of the grid, and the bandwidth at which it occurs. On a monotone
    spectrum the maximum reflects only the extent of the grid, not a property of
    the data (Section 4.5 of the paper).
    """
    b = list(bandwidths)
    v = np.asarray(values, float)
    i = int(np.nanargmax(v))
    interior = 0 < i < len(v) - 1
    # monotonicity: the differences share a sign, allowing for numerical noise
    d = np.diff(v)
    mono = bool(np.all(d >= -1e-12) or np.all(d <= 1e-12))
    return dict(monotone=mono, interior_maximum=bool(interior and not mono),
                argmax_bandwidth=b[i], at_endpoint=bool(i == 0 or i == len(v) - 1))
