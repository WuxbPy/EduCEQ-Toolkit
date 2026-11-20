# -*- coding: utf-8 -*-
"""Does the decomposition attribute a departure to the right term? (Section 5.4)

The identity CEQ = CEQ1 * CEQ2 holds by construction and therefore says nothing
about correctness on its own. What can be checked is whether the first- and
second-order terms carry the departure to the right side when the generating
process is known:

    null        three categories, all uniform and independent
                -> CEQ ~ 1, CEQ1 ~ 1, CEQ2 ~ 1
    composition the composition of the area changes smoothly: in a core of
                radius 2.5 km the three categories are equally frequent, outside
                it the third category takes 90 per cent of the objects and A and
                B take 5 per cent each. A and B then share a location pattern
                with no pairing at any scale.
                -> CEQ > 1, CEQ1 > 1, CEQ2 ~ 1
    interaction A and B uniformly spread, but every B object is moved onto an
                A object, so the smoothed field of B is unchanged and only the
                fine-scale pairing is new.
                -> CEQ > 1, CEQ1 ~ 1, CEQ2 > 1

If the decomposition is informative, the third case must load on the second term
and the second case on the first. An estimator that merely satisfied the identity
would not have to do that.

Run:
    python evaluation/decomposition_attribution.py
Output: results/decomposition_attribution.csv
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from educeq import estimator  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")
SIDE = 10.0
N = 9000
BAND = 0.25
GRID = 120
REPS = 60
SEED = 20261112


def core_and_out(n_core, r_core, rng):
    """Points inside a disk of radius r_core and outside it, uniform within each."""
    inside = np.empty((0, 2))
    while len(inside) < n_core:
        cand = rng.uniform(-r_core, r_core, size=(3 * n_core, 2))
        keep = cand[np.hypot(cand[:, 0], cand[:, 1]) < r_core]
        inside = np.vstack([inside, keep])[:n_core]
    inside += [SIDE / 2, SIDE / 2]
    return inside


def make(kind, rng):
    """Three generating processes, all with the composition of the area known."""
    n = N // 3
    r = 2.5
    if kind == "null":
        xy = rng.uniform(0, SIDE, size=(N, 2))
        types = rng.permutation(np.arange(N) % 3)
        return xy, types
    if kind == "composition":
        # A and B are drawn from the same inhomogeneous density, 60 per cent of
        # them in the core; C is mostly outside. A and B therefore share a
        # location pattern, with no pairing at any scale.
        parts, kinds = [], []
        for cat, share_core in ((0, 0.6), (1, 0.6), (2, 0.1)):
            n_core = int(share_core * n)
            parts.append(core_and_out(n_core, r, rng))
            parts.append(rng.uniform(0, SIDE, size=(n - n_core, 2)))
            kinds += [cat] * n
        xy = np.vstack(parts)
        types = np.array(kinds)
        order = rng.permutation(len(types))
        return xy[order], types[order]
    # interaction: A and B both uniform, 40 per cent of the B objects moved onto
    # A objects, so only the fine-scale pairing is new
    xy = rng.uniform(0, SIDE, size=(N, 2))
    types = rng.permutation(np.arange(N) % 2)
    b_idx = np.where(types == 1)[0]
    a_idx = np.where(types == 0)[0]
    movers = b_idx          # every B object is placed on an A object
    hosts = rng.choice(a_idx, len(movers), replace=False)
    xy[movers] = xy[hosts]
    return xy, types


def main():
    os.makedirs(RES, exist_ok=True)
    rows = []
    for kind in ["null", "composition", "interaction"]:
        for r in range(REPS):
            rng = np.random.default_rng(SEED + r)
            coords, types = make(kind, rng)
            from educeq import kernels
            W = kernels.distance(coords, bandwidth=BAND)
            out = estimator._decompose_unchecked(W, coords, types, BAND,
                                                 grid_n=GRID)
            rows.append(dict(regime=kind, rep=r, ceq=out["C"][0, 1],
                             ceq1=out["C1"][0, 1], ceq2=out["C2"][0, 1],
                             identity_ok=out["identity_ok"]))
            del W
    import csv
    with open(os.path.join(RES, "decomposition_attribution.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for kind in ["null", "composition", "interaction"]:
        sub = [r for r in rows if r["regime"] == kind]
        f = lambda k: (np.mean([s[k] for s in sub]), np.std([s[k] for s in sub], ddof=1))
        print("%-12s CEQ = %.4f (%.4f)  CEQ1 = %.4f (%.4f)  CEQ2 = %.4f (%.4f)"
              % (kind, *f("ceq"), *f("ceq1"), *f("ceq2")), flush=True)


if __name__ == "__main__":
    main()
