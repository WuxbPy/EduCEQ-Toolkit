# -*- coding: utf-8 -*-
"""educeq.diagnostics — support diagnostics and decidability (module four)

This module corresponds to Section 4 of the paper, and it is the one place where
the tool works against its own purpose: **it does not promise an answer**.

Two preconditions pull in opposite directions:
  * condition one   the effective number of target-type neighbours of a subject
                    is at least `ess_min` (too few effective observations makes
                    the estimate unstable)
  * condition two   the population coverage proportion of a neighbourhood is at
                    most `rho_max` (too wide a neighbourhood compresses the ratio
                    mechanically towards one)

The two are linked by e_B ~ N_B * rho, and taken together they give the
feasibility condition N_B >= ess_min / rho_max = 3000 of the paper, which depends
on neither the study area nor the bandwidth.

Why the effective size and not the count
----------------------------------------
Version 1.0.0 of this module summed the weights (and the text described them as
raw counts). Neither is the scale on which the variance of a ratio of weighted
sums depends. For weights w_1..w_m the variance of a weighted mean is governed by
sum(w^2)/sum(w)^2, whose reciprocal is the Kish effective sample size
e = (sum w)^2 / sum w^2. At a fixed number of non-zero neighbours, e varies by an
order of magnitude between an indicator kernel and a heavy-tailed one, and the
standard deviation of the estimate varies with it (Table 11 of the paper). The
contract is therefore applied to e, and the raw count and the weight sum are
reported alongside so that the difference is visible.

Failure semantics: when the conditions do not hold the call **does not raise**;
it returns a status object carrying a fingerprint of the data. Undecidability is
a research finding, not a program error, and the user should be able to put it
into a report rather than be interrupted by the library. The functions that can
return an estimate, by contrast, do raise: see educeq.contract.
"""
import numpy as np

from ._attr import support_matrices
from .contract import fingerprint

__all__ = ["support", "neighbourhood_count", "effective_size", "coverage",
           "check", "Decidability", "spectral_shape"]


class Decidability:
    """Outcome of the decidability check. Nothing is raised; the verdict lives in
    the decidable field, and the fingerprint binds it to the data it describes.
    """

    def __init__(self, decidable, target, n_target, kappa_raw, kappa_weighted,
                 ess, rho_raw, rho_weighted, rho_ess, concentration=float("nan"),
                 failed=None, advice=None, ess_min=30.0, rho_max=0.01,
                 fingerprint=None, pair=None):
        self.decidable = bool(decidable)
        self.target = target
        self.pair = pair if pair is not None else (target, target)
        self.n_target = int(n_target)
        self.kappa_raw = float(kappa_raw)
        self.kappa_weighted = float(kappa_weighted)
        self.ess = float(ess)
        self.rho_raw = float(rho_raw)
        self.rho_weighted = float(rho_weighted)
        self.rho_ess = float(rho_ess)
        self.concentration = float(concentration)
        self.failed = failed or []
        self.advice = advice or ""
        self.ess_min = float(ess_min)
        self.rho_max = float(rho_max)
        self.fingerprint = fingerprint

    # -- names kept from version 1.0.0 ----------------------------------------
    @property
    def kappa(self):
        """Effective neighbourhood size (the quantity the criterion uses)."""
        return self.ess

    @property
    def rho(self):
        """Coverage proportion measured in effective size."""
        return self.rho_ess

    def __repr__(self):
        head = ("Decidability(decidable=%s, target=%s, N_B=%d, "
                % (self.decidable, self.target, self.n_target))
        body = ("n_B=%.1f, S_B=%.1f, e_B=%.1f, rho_n=%.5f, rho_S=%.5f, "
                "rho_e=%.5f" % (self.kappa_raw, self.kappa_weighted, self.ess,
                                self.rho_raw, self.rho_weighted, self.rho_ess))
        if self.decidable:
            return head + body + ")"
        return head + body + ", failed=%s, advice=%s)" % (self.failed, self.advice)

    def as_dict(self):
        return dict(decidable=self.decidable, target=self.target, pair=self.pair,
                    n_target=self.n_target, kappa=self.ess, rho=self.rho_ess,
                    support_raw=self.kappa_raw,
                    support_weighted=self.kappa_weighted,
                    ess=self.ess, rho_raw=self.rho_raw,
                    rho_weighted=self.rho_weighted, rho_ess=self.rho_ess,
                    concentration=self.concentration,
                    ess_min=self.ess_min, rho_max=self.rho_max,
                    failed=list(self.failed), advice=self.advice,
                    fingerprint=self.fingerprint)


# ---------------------------------------------------------------- public helpers
def support(W, types):
    """The three support matrices of Section 4.2 (raw, weighted, effective)."""
    return support_matrices(W, np.asarray(types))


def neighbourhood_count(W, types):
    """Raw support matrix: median number of target objects with non-zero weight."""
    return support_matrices(W, np.asarray(types))["raw"]


def effective_size(W, types):
    """Effective-size matrix e_AB = median_i (sum w)^2 / sum w^2."""
    return support_matrices(W, np.asarray(types))["ess"]


def coverage(W, types=None):
    """Coverage proportion in effective size: median neighbourhood e divided by
    the number of objects."""
    n = W.shape[0]
    if types is None:
        types = np.zeros(n, dtype=int)
    return support_matrices(W, np.asarray(types))["rho_ess"]


def check(W, types, target=None, ess_min=30.0, rho_max=0.01, n_total=None,
          kappa_min=None, pair=None):
    """Three-step decidability check, evaluated on the target pair (A, B).

    Step 1  the target category has N_B >= ess_min / rho_max objects
            (3000 by default). If this fails the check stops here.
    Step 2  the effective neighbourhood size of the target pair is at least
            ess_min.
    Step 3  the empirical coverage proportion, measured in effective size, is at
            most rho_max.

    ``target`` is the source type A; when it is None the largest category is
    used, and the choice is recorded in the verdict. ``pair`` overrides the
    target pair (A, B) used for steps 2 and 3; by default B = A.

    ``kappa_min`` is accepted as a deprecated alias of ``ess_min`` so that scripts
    written against version 1.0.0 keep working; the value is read as an effective
    size, which is the quantity the criterion now means.

    Returns a Decidability object; when decidable is False it also carries the
    list of failed steps and an advice string.
    """
    types = np.asarray(types)
    if kappa_min is not None:
        ess_min = float(kappa_min)
    ess_min, rho_max = float(ess_min), float(rho_max)
    n = len(types) if n_total is None else int(n_total)
    if target is None:
        target = int(np.bincount(types).argmax())
    target = int(target)
    a, b = (target, target) if pair is None else (int(pair[0]), int(pair[1]))

    s = support_matrices(W, types)
    n_target = int((types == target).sum())
    threshold = int(round(ess_min / rho_max))
    fp = fingerprint(W, types, target=target, pair=(a, b),
                     ess_min=ess_min, rho_max=rho_max)

    failed = []
    if n_target < threshold:
        failed.append("step1:n_target")
        return Decidability(
            False, target, n_target, s["raw"][a, b], s["weighted"][a, b],
            s["ess"][a, b], s["rho_raw"], s["rho_weighted"], s["rho_ess"],
            s["concentration"], failed,
            "target category has %d objects, below the threshold of %d; "
            "widen the time span or the study area. No parameter setting can "
            "bypass this." % (n_target, threshold),
            ess_min, rho_max, fp, pair=(a, b))

    # steps 2 and 3
    if s["ess"][a, b] < ess_min:
        failed.append("step2:ess")
    if s["rho_ess"] > rho_max:
        failed.append("step3:rho")

    advice = ""
    if failed:
        if "step2:ess" in failed and "step3:rho" in failed:
            advice = ("both conditions fail: satisfying the effective-size "
                      "condition requires a wider neighbourhood, satisfying the "
                      "coverage condition requires a narrower one, so the "
                      "feasible interval is empty.")
        elif "step2:ess" in failed:
            advice = ("insufficient effective support (%d effective neighbours "
                      "against a minimum of %d): widen the bandwidth, use a "
                      "flatter kernel, or collect more data."
                      % (int(s["ess"][a, b]), int(ess_min)))
        else:
            advice = ("coverage proportion too high in effective size (%.4f "
                      "against %.4f): narrow the bandwidth."
                      % (s["rho_ess"], rho_max))
    return Decidability(
        not failed, target, n_target, s["raw"][a, b], s["weighted"][a, b],
        s["ess"][a, b], s["rho_raw"], s["rho_weighted"], s["rho_ess"],
        s["concentration"], failed, advice, ess_min, rho_max, fp, pair=(a, b))


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
