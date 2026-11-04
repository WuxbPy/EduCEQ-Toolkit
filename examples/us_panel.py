# -*- coding: utf-8 -*-
"""Primary validation: institution-level panel for U.S. higher education.

This is the main validation example of the paper (Section 5.2). The data are
institution-level records from College Scorecard, published by the U.S. Department
of Education. The largest category in the analytic sample contains about 1,765
objects, below the count threshold of 3,000, so the feasible bandwidth interval is
empty and the toolkit declines to return an estimate.

Pipeline
--------
    1. read the prepared sample (data/scorecard.npz)
    2. lon/lat -> local planar coordinates (equirectangular, longitude scaled by mean latitude)
    3. decidability check -> expected: undecidable
    4. show the status object returned by the toolkit (no exception, no estimate)

Run
---
    python data/prepare_scorecard.py --raw <download dir> --out data/scorecard.npz
    python examples/us_panel.py

If the prepared sample is absent, the script falls back to a synthetic sample of the
same size and says so, so that the pipeline can be checked anywhere.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from educeq import kernels, diagnostics, estimator  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "scorecard.npz")

KAPPA_MIN = 30.0
RHO_MAX = 0.01


def project(lonlat):
    """Lon/lat to a local plane (equirectangular), longitude scaled by mean latitude."""
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    lat0 = np.deg2rad(np.nanmean(lat))
    x = np.deg2rad(lon) * 6371.0 * np.cos(lat0)
    y = np.deg2rad(lat) * 6371.0
    return np.column_stack([x, y])          # units: kilometres


def main():
    print("=" * 68)
    print("Primary validation: U.S. higher-education panel")
    print("=" * 68)

    if os.path.exists(DATA):
        d = np.load(DATA)
        coords_raw, types = d["coords"], d["types"]
        src = "College Scorecard (public data)"
    else:
        rng = np.random.default_rng(3)
        n = 4785
        coords_raw = np.column_stack([rng.uniform(-125, -66, n), rng.uniform(25, 49, n)])
        types = rng.choice([0, 1, 2], n, p=[0.37, 0.33, 0.30])
        src = "synthetic sample of the same size (raw data absent, pipeline check only)"

    print("\ndata source: %s" % src)
    print("objects: %d   categories: %d" % (len(types), int(types.max()) + 1))
    print("composition: %s" % np.bincount(types, minlength=3).tolist())

    coords = project(coords_raw)
    print("projection: lon/lat -> local plane in km, longitude scaled by mean latitude")

    # ---- one check at the working bandwidth ----
    print("\n[check] bandwidth 150 km (working value used in Section 5.2)")
    W = kernels.distance(coords, bandwidth=150.0)
    d150 = diagnostics.check(W, types, target=0)
    print("  N_B=%d  kappa_B=%.1f  rho=%.5f" % (d150.n_target, d150.kappa, d150.rho))
    print("  decidable = %s" % d150.decidable)
    if not d150.decidable:
        print("  failed step: %s" % d150.failed)
        print("  advice: %s" % d150.advice)

    # ---- bandwidth sweep: confirm the feasible interval is empty ----
    print("\n[bandwidth sweep] is the feasible interval empty?")
    print("  %-10s %-10s %-10s %s" % ("bandwidth", "kappa_B", "rho", "verdict"))
    feasible = []
    for b in [50, 100, 150, 200, 300, 400, 500]:
        Wb = kernels.distance(coords, bandwidth=float(b))
        db = diagnostics.check(Wb, types, target=0)
        ok_b = db.decidable
        if ok_b:
            feasible.append(b)
        print("  %-10d %-10.1f %-10.5f %s" % (b, db.kappa, db.rho, "decidable" if ok_b else "undecidable"))

    print("\n[conclusion]")
    if feasible:
        print("  feasible bandwidths: %s" % feasible)
    else:
        print("  no bandwidth in 50-500 km satisfies both criteria: the feasible interval is empty.")
        print("  the tool declines to estimate on real public data, which is the behaviour described in Section 4.")
    print("  note: this is insufficient data size, not an insensitive measure; Section 5.3 separates the two.")

    # ---- the convention used in the paper: size of the largest category ----
    cnt = np.bincount(types, minlength=3)
    print("\n[link to the paper] largest category = %d, below the threshold of %d" % (cnt.max(), int(KAPPA_MIN / RHO_MAX)))
    print("  Section 5.2 reports non-decidability on this basis; the tool returns the same verdict.")

    print("\n" + "=" * 68)
    print("primary validation finished.")
    print("=" * 68)


if __name__ == "__main__":
    main()
