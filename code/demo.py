# -*- coding: utf-8 -*-
"""EduCEQ-Toolkit self-check and worked examples (Section 5 of the paper).

    cd code && python demo.py

Example 1  large sample, decidable    -> enrichment matrix, decomposition and diagnostics
Example 2  small sample, undecidable  -> no estimate, status object with remedies
Example 3  synthetic data with known ground truth -> the measure is sensitive when conditions allow

All data are synthetic and the script reproduces Sections 5.3 and 5.5 of the paper
without downloading anything.

Note on Example 1: the bandwidth is 5 km on a 200 x 200 km domain. Under the
support measures of version 1.1.0 this gives a median raw support of 77 and an
effective size of 33.8, i.e. the same configuration that version 1.0.0 described
as "kappa about 33" is now reported as 77 raw / 33.8 effective. The difference is
the point of Table 11: the count and the information are not the same quantity.
"""
import numpy as np
from educeq import kernels, estimator, inference, diagnostics, validity

rng = np.random.default_rng(7)

print("=" * 66)
print("EduCEQ-Toolkit self-check")
print("=" * 66)

# ---------------------------------------------------------------- example 1
print("\n[Example 1] Large sample, decidable")
# Objects spread uniformly; the bandwidth is chosen so that all three criteria hold:
# the target category has >= 3000 objects, the effective size >= 30 and rho <= 0.01.
n = 9000
coords = rng.uniform(0, 200, size=(n, 2))
types = rng.integers(0, 2, n)
W = kernels.distance(coords, bandwidth=5.0)
dec = diagnostics.check(W, types, target=0)
print("  verdict: N_B=%d  n_B=%.0f  S_B=%.1f  e_B=%.1f  rho_e=%.5f  decidable=%s"
      % (dec.n_target, dec.kappa_raw, dec.kappa_weighted, dec.ess,
         dec.rho_ess, dec.decidable))
if dec.decidable:
    C = estimator.ceq_matrix(W, types, contract=dec)
    ok, dev = estimator.normalization_ok(C, types)
    print("  enrichment matrix:")
    for a in range(C.shape[0]):
        print("     " + "".join("%9.4f" % C[a, b] for b in range(C.shape[1])))
    print("  normalisation identity: %s (max deviation %.2e)" % (ok, dev))
    res = estimator.decompose(W, coords, types, bandwidth=10.0, grid_n=40,
                              target=0, contract=dec)
    print("  second-order range: %.4f - %.4f, identity=%s"
          % (np.nanmin(res["C2"]), np.nanmax(res["C2"]), res["identity_ok"]))
    p = inference.permutation(W, types, contract=dec, n_perm=199, seed=0)
    print("  permutation z (A->A): %.2f" % p["z"][0, 0])
    print("  diagnostics carried by the result: e_B=%.1f  rho_e=%.5f"
          % (res["diagnostics"]["ess"], res["diagnostics"]["rho_ess"]))
else:
    print("  undecidable: %s" % dec.advice)

# ---------------------------------------------------------------- example 2
print("\n[Example 2] Small sample, undecidable")
n2 = 900
coords2 = rng.uniform(0, 100, size=(n2, 2))
types2 = rng.integers(0, 3, n2)
W2 = kernels.distance(coords2, bandwidth=40.0)       # coverage far too high
res2 = estimator.decompose(W2, coords2, types2, bandwidth=40.0, target=0)
print("  decidable = %s" % res2["decidable"])
if not res2["decidable"]:
    print("  failed step: %s" % res2["failed"])
    print("  advice: %s" % res2["advice"])
    print("  no key 'C' in the returned object: %s" % ("C" not in res2))
print("  note: the tool returns no estimate here, only the undecidable status.")

# ---------------------------------------------------------------- example 2b
print("\n[Example 2b] Large bandwidth, undecidable")
n2b = 9000
coords2b = rng.uniform(0, 100, size=(n2b, 2))
types2b = rng.integers(0, 2, n2b)
W2b = kernels.distance(coords2b, bandwidth=40.0)     # neighbourhood covers most of the domain
dec2b = diagnostics.check(W2b, types2b, target=0)
print("  decidable = %s   failed = %s" % (dec2b.decidable, dec2b.failed))
print("  e_B=%.1f  rho_e=%.4f (limit 0.01)" % (dec2b.ess, dec2b.rho_ess))
print("  advice: %s" % dec2b.advice)

# ------------------------------------------------------------ example 2c
print("\n[Example 2c] A verdict cannot be borrowed")
try:
    estimator.ceq_matrix(W2, types2)                       # no verdict at all
except Exception as exc:
    print("  without a verdict: %s" % type(exc).__name__)
W3 = kernels.distance(rng.uniform(0, 80, size=(400, 2)), bandwidth=40.0)
try:
    estimator.ceq_matrix(W3, np.zeros(400, dtype=int), contract=dec)
except Exception as exc:
    print("  with a verdict issued for other data: %s" % type(exc).__name__)

# ---------------------------------------------------------------- example 3
print("\n[Example 3] Attenuation with known ground truth (analytic values)")
n3 = 4000
Z = rng.normal(size=n3)
X = 0.8 * Z + rng.normal(size=n3)
pt_a = validity.proxy_test(0.6 * Z + rng.normal(size=n3), X, Z=Z[:, None])
pt_b = validity.proxy_test(0.7 * X + rng.normal(size=n3), X, Z=Z[:, None])
X0 = rng.normal(size=n3)
pt_c = validity.proxy_test(0.7 * X0 + rng.normal(size=n3), X0, Z=Z[:, None])
print("  complete proxy   (analytic 1.0000): recovered %.4f  class=%s"
      % (pt_a["lam"], pt_a["cls"]))
print("  independent      (analytic 0.1409): recovered %.4f  class=%s"
      % (pt_b["lam"], pt_b["cls"]))
print("  X orthogonal to Z (analytic 0.0000): recovered %.4f  class=%s"
      % (pt_c["lam"], pt_c["cls"]))
print("  the middle row is often mislabelled: its population attenuation is")
print("  0.1409, not zero. The zero-attenuation design is the third row, where")
print("  X is drawn independently of Z.")

# ---------------------------------------------------------------- channels
print("\n[Channel report]")
rep = inference.channel_report()
print("  available: %s" % rep["available"])
print("  not implemented: %s" % rep["not_implemented"])
print("  %s" % rep["note"])

print("\n" + "=" * 66)
print("Self-check finished. All data above are synthetic and reproducible.")
print("=" * 66)
