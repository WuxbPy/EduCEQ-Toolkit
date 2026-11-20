# -*- coding: utf-8 -*-
"""Inference under spatial dependence: size, power, and BH against BY (Section 5.8)

Two questions are asked here, and they are different.

(a) Exchangeability. Conditional permutation is justified only if the labels are
exchangeable within the conditioning stratum. If the labels carry spatial
autocorrelation that is not in the stratum, permuting globally destroys it, the
null distribution is too narrow, and the test rejects far more often than the
nominal level. The first panel measures the empirical rejection rate of the two
channels on data where the null is true by construction: objects drawn from an
inhomogeneous density, labels assigned independently of location, so that A and B
share a spatial trend with no interaction. Global permutation is expected to
over-reject; whether permutation within blocks repairs that depends on the block
size, and the block sizes in the grid are chosen to bracket the answer rather
than to illustrate it.

(b) False discovery rate control under dependent tests. The second panel plants
associations among three of six categories, so that the fifteen pairwise tests
share types and are therefore correlated, and compares the empirical false
discovery rate of Benjamini-Hochberg with that of Benjamini-Yekutieli, which is
valid under arbitrary dependence at the cost of the harmonic factor.

Run:
    python evaluation/inference_dependence.py
Output: results/inference_dependence.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from educeq import kernels, inference  # noqa: E402
from educeq._core import permutation_test  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")
SIDE = 10.0
N = 4000
BAND = 0.6
N_PERM = 49
REPS = 200
ALPHA = 0.05
SEED = 20261113


def pattern(rng):
    m = int(0.7 * N)
    core = rng.normal([SIDE / 2, SIDE / 2], 1.8, size=(m, 2))
    rest = rng.uniform(0, SIDE, size=(N - m, 2))
    xy = np.vstack([core, rest])
    xy[:, 0] = np.clip(xy[:, 0], 0, SIDE)
    xy[:, 1] = np.clip(xy[:, 1], 0, SIDE)
    return xy


def pvalue(obs, null, a, b):
    """Two-sided permutation p value for one cell of the matrix."""
    stat = np.abs(null[:, a, b] - 1.0)
    return (1.0 + np.sum(stat >= abs(obs[a, b] - 1.0))) / (N_PERM + 1.0)


def pvalue_z(obs, null, a, b):
    """Two-sided p value from the standardised departure.

    Used where a step-up procedure has to reach below the resolution of 99
    permutations: with 99 draws the smallest attainable empirical p value is
    0.01, and Benjamini-Hochberg over fifteen tests at 0.05 needs 0.0033. The
    standardisation is the one the module already reports as z.
    """
    from scipy.stats import norm
    sd = null[:, a, b].std(ddof=1)
    if sd <= 0:
        return 1.0
    z = (obs[a, b] - null[:, a, b].mean()) / sd
    return float(2.0 * norm.sf(abs(z)))


def main():
    os.makedirs(RES, exist_ok=True)
    rng = np.random.default_rng(SEED)
    xy = pattern(rng)
    W = kernels.distance(xy, bandwidth=BAND)
    blocks = (np.floor(xy[:, 0] / 2.0) * 5 + np.floor(xy[:, 1] / 2.0)).astype(int)

    # ---- (a) size of the two channels under a true null -------------------
    # The composition of the neighbourhood varies smoothly: in the core, A and B
    # each hold a quarter of the objects; outside, a tenth part of that. A and B
    # therefore co-locate, but only because the composition of the area changes,
    # which is a first-order effect with no interaction. Under the null of no
    # association beyond composition both channels should reject at 5 per cent;
    # global permutation removes the composition with the labels and should
    # reject far more often.
    core = np.hypot(xy[:, 0] - SIDE / 2, xy[:, 1] - SIDE / 2) < 3.0
    nblock = {2.0: 5, 1.0: 10, 0.5: 20}
    rej = {"global": 0, "conditional_2km": 0, "conditional_1km": 0,
           "conditional_0.5km": 0}
    for r in range(REPS):
        r3 = np.random.default_rng(SEED + r)
        u = r3.random(N)
        inside = np.where(core)[0]
        outside = np.where(~core)[0]
        types = np.empty(N, int)
        types[inside] = np.where(u[inside] < 0.30, 0,
                                 np.where(u[inside] < 0.60, 1, 2))
        types[outside] = np.where(u[outside] < 0.03, 0,
                                  np.where(u[outside] < 0.06, 1, 2))
        obs, _, _, _, null = permutation_test(W, types, [0, 1, 2], n_perm=N_PERM,
                                              seed=SEED + r)
        rej["global"] += pvalue(obs, null, 0, 1) < ALPHA
        for side, nb in nblock.items():
            bl = (np.floor(xy[:, 0] / side).clip(0, nb - 1) * nb
                  + np.floor(xy[:, 1] / side).clip(0, nb - 1)).astype(int)
            obs2, _, _, _, null2 = permutation_test(W, types, [0, 1, 2],
                                                    n_perm=N_PERM,
                                                    seed=SEED + r, block=bl)
            rej["conditional_%gkm" % side] += pvalue(obs2, null2, 0, 1) < ALPHA
        if (r + 1) % 50 == 0:
            print("size rep %d: %s" % (r + 1, {k: round(v / (r + 1), 3)
                                               for k, v in rej.items()}),
                  flush=True)
    # (a2) the same question when the nuisance is a categorical attribute, so
    # that the stratum coincides with it: four quadrants, each with its own
    # composition. Permutation within a quadrant is then exact by construction.
    quad = ((xy[:, 0] >= SIDE / 2).astype(int) * 2
            + (xy[:, 1] >= SIDE / 2).astype(int))
    share = {0: (0.30, 0.60), 1: (0.05, 0.10), 2: (0.20, 0.40), 3: (0.40, 0.80)}
    rej_a2 = {"global": 0, "conditional_quadrant": 0}
    reps_a2 = 100
    for r in range(reps_a2):
        r4 = np.random.default_rng(SEED + 5000 + r)
        u = r4.random(N)
        types = np.empty(N, int)
        for q, (lo, hi) in share.items():
            m = quad == q
            types[m] = np.where(u[m] < lo, 0, np.where(u[m] < hi, 1, 2))
        obs, _, _, _, null = permutation_test(W, types, [0, 1, 2], n_perm=N_PERM,
                                              seed=SEED + r)
        obs2, _, _, _, null2 = permutation_test(W, types, [0, 1, 2], n_perm=N_PERM,
                                                seed=SEED + r, block=quad)
        rej_a2["global"] += pvalue(obs, null, 0, 1) < ALPHA
        rej_a2["conditional_quadrant"] += pvalue(obs2, null2, 0, 1) < ALPHA
    size_exact = {k: v / reps_a2 for k, v in rej_a2.items()}
    print("exact strata (quadrants):", size_exact, flush=True)

    size = {k: v / REPS for k, v in rej.items()}

    # ---- (b) FDR and power under dependent tests --------------------------
    K = 6
    pairs = [(a, b) for a in range(K) for b in range(K) if a < b]
    true_alt = {(0, 1), (0, 2), (1, 2)}
    fdp_bh, fdp_by, pow_bh, pow_by = [], [], [], []
    for r in range(100):
        r2 = np.random.default_rng(SEED + 1000 + r)
        types = r2.integers(0, K, N)
        xyr = xy.copy()
        # 30 per cent of the objects of type 1 and of type 2 are moved onto
        # objects of type 0, which plants a fine-scale association between the
        # three and leaves the other three categories independent
        idx0 = np.where(types == 0)[0]
        for tgt in (1, 2):
            idx_t = np.where(types == tgt)[0]
            n_move = int(0.3 * len(idx_t))
            movers = r2.choice(idx_t, n_move, replace=False)
            hosts = r2.choice(idx0, n_move, replace=False)
            xyr[movers] = xy[hosts]
        Wr = kernels.distance(xyr, bandwidth=BAND)
        obs, _, _, _, null = permutation_test(Wr, types, list(range(K)),
                                              n_perm=N_PERM, seed=SEED + r,
                                              block=blocks)
        del Wr
        p = np.array([pvalue_z(obs, null, a, b) for a, b in pairs])
        for fn, fdp, pw in ((inference.fdr, fdp_bh, pow_bh),
                            (inference.fdr_by, fdp_by, pow_by)):
            q, k = fn(p, alpha=ALPHA)
            rej_set = {pairs[i] for i in np.where(q <= ALPHA)[0]}
            false = len(rej_set - true_alt)
            fdp.append(false / max(len(rej_set), 1))
            pw.append(len(rej_set & true_alt) / len(true_alt))
        if (r + 1) % 25 == 0:
            print("fdr rep %d: BH %.3f BY %.3f" % (r + 1, np.mean(fdp_bh),
                                                   np.mean(fdp_by)), flush=True)

    out = dict(n=N, bandwidth=BAND, n_perm=N_PERM, reps=REPS, alpha=ALPHA,
               size=size, size_exact_strata=size_exact, fdr_bh=float(np.mean(fdp_bh)),
               fdr_by=float(np.mean(fdp_by)),
               power_bh=float(np.mean(pow_bh)), power_by=float(np.mean(pow_by)))
    print(json.dumps(out, indent=1), flush=True)
    with open(os.path.join(RES, "inference_dependence.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
