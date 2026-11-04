# -*- coding: utf-8 -*-
"""EduCEQ-Toolkit self-check and worked examples (Section 5 of the paper).

    cd code && python demo.py

Example 1  large sample, decidable    -> enrichment matrix, decomposition and diagnostics
Example 2  small sample, undecidable  -> no estimate, status object with remedies
Example 3  synthetic data with known ground truth -> the measure is sensitive when conditions allow

All data are synthetic and the script reproduces Sections 5.3 and 5.5 of the paper
without downloading anything.
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
# the target category has >= 3000 objects, kappa_B >= 30 and rho <= 0.01.
n = 9000
coords = rng.uniform(0, 200, size=(n, 2))
types = rng.integers(0, 2, n)
W = kernels.distance(coords, bandwidth=7.0)
d = diagnostics.check(W, types, target=0)
print("  diagnostics: N_B=%d  kappa_B=%.1f  rho=%.5f  decidable=%s"
      % (d.n_target, d.kappa, d.rho, d.decidable))
if d.decidable:
    C = estimator.ceq_matrix(W, types)
    ok, dev = estimator.normalization_ok(C, types)
    print("  enrichment matrix:")
    for a in range(C.shape[0]):
        print("     " + "".join("%9.4f" % C[a, b] for b in range(C.shape[1])))
    print("  normalisation identity: %s (max deviation %.2e)" % (ok, dev))
    res = estimator.decompose(W, coords, types, bandwidth=7.0, grid_n=40, target=0)
    print("  second-order range: %.4f - %.4f, identity=%s"
          % (np.nanmin(res["C2"]), np.nanmax(res["C2"]), res["identity_ok"]))
    print("  diagnostics carried by the result: kappa_B=%.1f rho=%.5f"
          % (res["diagnostics"]["kappa"], res["diagnostics"]["rho"]))
else:
    print("  undecidable: %s" % d.advice)

# ---------------------------------------------------------------- example 2
print("\n[Example 2] Small sample, undecidable")
n2 = 900
coords2 = rng.uniform(0, 100, size=(n2, 2))
types2 = rng.integers(0, 3, n2)
W2 = kernels.distance(coords2, bandwidth=40.0)       # bandwidth too large => coverage too high
res2 = estimator.decompose(W2, coords2, types2, bandwidth=40.0, target=0)
print("  decidable = %s" % res2["decidable"])
if not res2["decidable"]:
    print("  failed step: %s" % res2["failed"])
    print("  advice: %s" % res2["advice"])
    print("  no key 'C' in the returned object: %s" % ("C" not in res2))
print("  note: the tool returns no estimate here, only the undecidable status.")

# ---------------------------------------------------------------- example 3
print("\n[Example 3] Synthetic data with known ground truth")
n3 = 4000
Z = rng.normal(size=n3)
X = 0.8 * Z + rng.normal(size=n3)
# case A: X has no direct effect on Y (complete proxy) => lambda -> 1
Y_a = 0.6 * Z + rng.normal(size=n3)
pt_a = validity.proxy_test(Y_a, X, Z=Z[:, None])
# case B: X has a direct effect (independent effect) => lambda -> 0
Y_b = 0.7 * X + rng.normal(size=n3)
pt_b = validity.proxy_test(Y_b, X, Z=Z[:, None])
print("  complete proxy (true lambda=1):  recovered lambda=%.4f  class=%s"
      % (pt_a["lam"], pt_a["cls"]))
print("  independent effect (true 0):     recovered lambda=%.4f  class=%s"
      % (pt_b["lam"], pt_b["cls"]))
print("  the two ground truths are separated, so a second-order term near one"
      " should be read as 'not detected' rather than 'absent'.")

# ---------------------------------------------------------------- channels
print("\n[Channel report]")
rep = inference.channel_report()
print("  available: %s" % rep["available"])
print("  not implemented: %s" % rep["not_implemented"])
print("  %s" % rep["note"])

print("\n" + "=" * 66)
print("Self-check finished. All data above are synthetic and reproducible.")
print("=" * 66)
