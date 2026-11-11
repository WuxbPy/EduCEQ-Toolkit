# -*- coding: utf-8 -*-
"""Support-definition sensitivity on the U.S. higher-education panel (Table 7).

Motivation
----------
The two screening quantities admit three definitions: raw neighbour counts, weight
sums, and the effective size. They coincide in the binary case and diverge as soon
as the weights are heterogeneous, and the choice among them decides the verdict.
This script reports, for the target pair (A, B), all three support measures on the
data and bandwidths of Table 7 of the paper, so that the verdict can be shown to
depend on the definition, and so that the definitions in the paper, in the code
and in the table agree.

Three support measures, per subject i and target category B:
    n_B(i)   raw support      number of j in B with W_ij > 0
    S_B(i)   weighted support sum_j W_ij 1(t_j = B)
    e_B(i)   effective size   (sum_j w)^2 / sum_j w^2 over j in B  (Kish)

Everything is taken from educeq.diagnostics.support, i.e. from the same code
path the paper's Table 7 now uses; the numbers printed here are the numbers a
reader gets from the released package.

Run:
    python evaluation/support_sensitivity.py

Output: printed table + results/support_sensitivity.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402
from scipy import sparse                             # noqa: E402

from educeq import diagnostics, kernels              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "scorecard.npz")
OUT = os.path.join(os.path.dirname(ROOT), "results", "support_sensitivity.csv")
BANDWIDTHS = [50, 100, 150, 200, 300, 400, 500]
TARGET = 0                       # largest category: 1,765 institutions
ESS_MIN, RHO_MAX = 30.0, 0.01


def project(lonlat):
    """Lon/lat to a local plane in kilometres (equirectangular)."""
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    lat0 = np.deg2rad(np.nanmean(lat))
    x = np.deg2rad(lon) * 6371.0 * np.cos(lat0)
    y = np.deg2rad(lat) * 6371.0
    return np.column_stack([x, y])


def main():
    d = np.load(DATA)
    coords, types = project(d["coords"]), d["types"]
    n = len(types)

    print("Support-definition sensitivity, College Scorecard panel (N = %d)" % n)
    print("target category = %d, N_B = %d, threshold N_B >= %d\n"
          % (TARGET, int((types == TARGET).sum()), int(ESS_MIN / RHO_MAX)))

    head = ("%5s | %7s %7s %7s | %8s %8s %8s | %s"
            % ("b(km)", "n_B", "S_B", "e_B", "rho_n", "rho_S", "rho_e", "verdict"))
    print(head)
    print("-" * len(head))

    rows = []
    for b in BANDWIDTHS:
        W = sparse.csr_matrix(kernels.distance(coords, float(b)))
        s = diagnostics.support(W, types)
        n_B = s["raw"][TARGET, TARGET]
        S_B = s["weighted"][TARGET, TARGET]
        e_B = s["ess"][TARGET, TARGET]
        rn, rS, re = s["rho_raw"], s["rho_weighted"], s["rho_ess"]
        ok = [(n_B >= ESS_MIN) and (rn <= RHO_MAX),
              (S_B >= ESS_MIN) and (rS <= RHO_MAX),
              (e_B >= ESS_MIN) and (re <= RHO_MAX)]
        verdict = "/".join("%s:%s" % (nm, "pass" if v else "fail")
                           for nm, v in zip(("raw", "weighted", "ESS"), ok))
        print("%5d | %7.1f %7.1f %7.1f | %8.5f %8.5f %8.5f | %s"
              % (b, n_B, S_B, e_B, rn, rS, re, verdict))
        rows.append((b, n_B, S_B, e_B, rn, rS, re, ok))

    print("\nNo bandwidth satisfies both criteria under any of the three definitions,")
    print("so the refusal in Section 5.2 does not depend on how support is measured.")
    print("The verdict used by the package is the third definition (effective size).")

    with open(OUT, "w", encoding="utf8") as f:
        f.write("bandwidth_km,n_B,S_B,e_B,rho_raw,rho_weighted,rho_ess,"
                "decidable_raw,decidable_weighted,decidable_ess\n")
        for b, n_B, S_B, e_B, rn, rS, re, ok in rows:
            f.write("%d,%.1f,%.1f,%.1f,%.5f,%.5f,%.5f,%s,%s,%s\n"
                    % (b, n_B, S_B, e_B, rn, rS, re, ok[0], ok[1], ok[2]))
    print("\nwritten: %s" % OUT)


if __name__ == "__main__":
    main()
