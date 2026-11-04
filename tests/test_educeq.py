# -*- coding: utf-8 -*-
"""EduCEQ-Toolkit test suite (synthetic data, runs standalone)

    cd code && python tests/test_educeq.py
    cd code && python -m pytest tests -q

Covers the five modules and the key properties of the paper, with emphasis on
the non-decidability behaviour (Section 4).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import educeq  # noqa: E402
from educeq import kernels, estimator, inference, diagnostics, validity  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (("  | " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)


def synth(n=900, K=3, seed=0, cluster=True):
    rng = np.random.default_rng(seed)
    if cluster:
        cen = rng.uniform(0, 100, size=(6, 2))
        coords = cen[rng.integers(0, 6, n)] + rng.normal(scale=3.0, size=(n, 2))
    else:
        coords = rng.uniform(0, 100, size=(n, 2))
    types = rng.integers(0, K, n)
    return coords, types


# ---------------------------------------------------------------- module 1: kernels
def test_kernels():
    coords, types = synth()
    Wd = kernels.distance(coords, bandwidth=8.0)
    z = rng_attr(coords, types)
    Wo = kernels.ordinal(z, radius=2.0)
    Wc = kernels.categorical(rng_cat(types))
    check("kernels: distance kernel has zero diagonal", bool(np.allclose(Wd.diagonal(), 0)))
    check("kernels: ordinal kernel weights non-negative", bool((Wo.data >= 0).all()))
    check("kernels: categorical kernel weights non-negative", bool((Wc.data >= 0).all()))
    Wc2 = kernels.compose(Wd, Wo)
    A = np.asarray(Wd.todense()); B = np.asarray(Wo.todense())
    check("kernels: composition is elementwise product",
          bool(np.allclose(np.asarray(Wc2.todense()), A * B, atol=1e-12)))
    # degenerate: a categorical kernel nested within the target type
    Wn = kernels.categorical(types)
    deg, msg = kernels.degeneracy_check(Wn, types)
    check("kernels: degeneracy check catches nested attribute", deg, msg[:40])
    # as r grows the kernel degenerates to the constant one
    Wbig = kernels.ordinal(z, radius=1e12)
    diff = abs(np.asarray(kernels.compose(Wd, Wbig).todense()) - A).max()
    check("kernels: ordinal kernel tends to constant one as r grows", diff < 1e-6, "max difference %.2e" % diff)


def rng_attr(coords, types):
    r = np.random.default_rng(11)
    return r.uniform(0, 10, len(types))


def rng_cat(types):
    r = np.random.default_rng(12)
    return r.integers(0, 5, len(types))


# ---------------------------------------------------------------- module 2: estimator
def test_estimator():
    coords, types = synth(n=600, seed=2)
    b = 12.0
    W = kernels.distance(coords, bandwidth=b)
    d = estimator.decompose(W, coords, types, bandwidth=b, grid_n=40, check=False)
    C, C1, C2 = d["C"], d["C1"], d["C2"]
    fin = np.isfinite(C) & np.isfinite(C1) & np.isfinite(C2)
    check("estimator: decomposition identity exact", bool(np.allclose(C[fin], (C1 * C2)[fin], atol=1e-9)))
    ok, dev = estimator.normalization_ok(C, types)
    check("estimator: normalisation identity", ok, "max deviation %.2e" % dev)


# ---------------------------------------------------------------- module 3: inference
def test_inference():
    coords, types = synth(n=300, seed=3)
    W = kernels.distance(coords, bandwidth=12.0)
    r = inference.permutation(W, types, n_perm=19, seed=0)
    check("inference: permutation output shape",
          r["null"].shape == (19, int(types.max()) + 1, int(types.max()) + 1))
    # stratified permutation
    strata = np.arange(len(types)) % 5
    r2 = inference.permutation(W, types, n_perm=9, seed=0, strata=strata)
    check("inference: stratified permutation runs", np.isfinite(r2["observed"]).all())
    q, k = inference.fdr(np.array([0.001, 0.02, 0.2, 0.6, 0.9]))
    check("inference: BH correction in [0,1]", 0 <= q.min() and q.max() <= 1, "q=%s" % np.round(q, 4).tolist())
    rep = inference.channel_report()
    check("inference: unavailable channel is reported, not hidden",
          "asymptotic_null" in rep["not_implemented"], str(rep["available"]))


# ---------------------------------------------------------------- module 4: diagnostics
def test_contract_enforced():
    """The contract is enforced inside decompose: no estimate is returned on failure."""
    rng = np.random.default_rng(6)
    n = 900
    coords = rng.uniform(0, 100, size=(n, 2))
    types = rng.integers(0, 2, n)
    W = kernels.distance(coords, bandwidth=30.0)
    res = estimator.decompose(W, coords, types, bandwidth=30.0, target=0)
    check("contract: failure flag set", res["decidable"] is False)
    check("contract: no estimate in the returned object", "C" not in res)
    check("contract: failing step reported", bool(res["failed"]), str(res["failed"]))
    check("contract: diagnostics travel with the status",
          "kappa" in res["diagnostics"] and "rho" in res["diagnostics"])
    res2 = estimator.decompose(W, coords, types, bandwidth=30.0, target=0, check=False)
    check("contract: check=False returns the bare decomposition", "C" in res2)


def test_diagnostics_decidable():
    """Small sample, but all three criteria hold."""
    rng = np.random.default_rng(4)
    n = 3000
    coords = rng.uniform(0, 60, size=(n, 2))
    types = rng.integers(0, 2, n)
    W = kernels.distance(coords, bandwidth=6.0)
    d = diagnostics.check(W, types, target=0)
    check("diagnostics: coverage proportion computable", 0 < d.rho < 1, "ρ=%.5f, κ=%.1f" % (d.rho, d.kappa))
    check("diagnostics: returns a Decidability object", isinstance(d, diagnostics.Decidability))
    check("diagnostics: exposes as_dict", set(d.as_dict()) >= {"decidable", "kappa", "rho"})


def test_diagnostics_undecidable():
    """Criteria unmet: no exception, a decidable=False status with advice."""
    rng = np.random.default_rng(5)
    n = 900
    coords = rng.uniform(0, 100, size=(n, 2))
    types = rng.integers(0, 3, n)
    # coverage proportion too large (bandwidth too large)
    W = kernels.distance(coords, bandwidth=40.0)
    d = diagnostics.check(W, types, target=0)
    check("diagnostics: no exception on non-decidability", isinstance(d, diagnostics.Decidability))
    if not d.decidable:
        check("diagnostics: failure carries failed step and advice",
              bool(d.failed) and bool(d.advice), "%s / %s" % (d.failed, d.advice[:40]))
    else:
        check("diagnostics: this case must be undecidable", False, "returned decidable=True")
    # target category below the count threshold
    d2 = diagnostics.check(W, types, target=2)
    check("diagnostics: undersized category stops at step 1",
          (not d2.decidable) and ("step1:n_target" in d2.failed), str(d2.failed))


def test_spectral_shape():
    b = [5, 20, 50, 100, 200, 500]
    v = [1.731, 1.463, 1.375, 1.325, 1.294, 1.230]      # monotonically decreasing
    s = diagnostics.spectral_shape(b, v)
    check("diagnostics: monotone spectrum detected", bool(s["monotone"]))
    check("diagnostics: extremum lies on the grid endpoint", bool(s["at_endpoint"]), "argmax=%s" % s["argmax_bandwidth"])
    check("diagnostics: no interior maximum on a monotone spectrum", not bool(s["interior_maximum"]))


# ---------------------------------------------------------------- module 5: validity
def test_validity():
    rng = np.random.default_rng(0)
    n = 3000
    Z = rng.normal(size=n)
    X = 0.8 * Z + rng.normal(size=n)
    Y = 0.6 * Z + rng.normal(size=n)                    # X has no direct effect => complete proxy
    pt = validity.proxy_test(Y, X, Z=Z[:, None])
    check("validity: complete proxy recovers lambda near 1", pt["lam"] > 0.9, "λ=%.4f" % pt["lam"])
    # P3 closed form
    X2 = 0.8 * Z + rng.normal(size=n)
    Y2 = 0.5 * X2 + 0.6 * Z + rng.normal(size=n)
    rxy = np.corrcoef(X2, Y2)[0, 1]
    rxz = np.corrcoef(X2, Z)[0, 1]
    ryz = np.corrcoef(Y2, Z)[0, 1]
    f = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    pt2 = validity.proxy_test(Y2, X2, Z=Z[:, None])
    check("validity: partial correlation matches closed form", abs(pt2["r_net"] - f) < 1e-12,
          "difference %.2e" % abs(pt2["r_net"] - f))
    # TOST
    a = validity.tost_corr(0.01, n, delta=0.10)
    b_ = validity.tost_corr(0.40, n, delta=0.10)
    check("validity: TOST declares equivalence for a small effect", bool(a["equivalent"]), "p=%.3g" % a["tost_p"])
    check("validity: TOST withholds equivalence for a large effect", not bool(b_["equivalent"]), "p=%.3g" % b_["tost_p"])
    # D1 discrimination decomposition
    n2 = 2500
    cell = rng.integers(0, 40, n2)
    zc = rng.normal(size=40)[cell]
    y = (rng.uniform(size=n2) < 1 / (1 + np.exp(-(1.2 * zc + 2.0 * rng.normal(size=n2))))).astype(int)
    if y.sum() in (0, n2):
        y[:5] = 1 - y[:5]
    dd = validity.discrimination_decomposition(y, zc, cell)
    check("validity: within-cell AUC is exactly 0.5", abs(dd["auc_within"] - 0.5) < 1e-9,
          "AUC_within=%.6f" % dd["auc_within"])
    check("validity: AUC decomposition identity holds", bool(dd["identity_ok"]))


# ---------------------------------------------------------------- package level
def test_package():
    check("package: version string present", isinstance(educeq.__version__, str))
    check("package: five modules importable",
          all(hasattr(educeq, m) for m in
              ["kernels", "estimator", "inference", "diagnostics", "validity"]))


if __name__ == "__main__":
    CASES = [test_package, test_kernels, test_estimator, test_inference,
             test_contract_enforced, test_diagnostics_decidable,
             test_diagnostics_undecidable, test_spectral_shape, test_validity]
    print("=" * 68)
    print("EduCEQ-Toolkit test suite (synthetic data)")
    print("=" * 68)
    for fn in CASES:
        try:
            fn()
        except Exception as e:
            check(fn.__name__, False, "exception: %s" % e)
    print("-" * 68)
    print("result: %s (%d failures / %d cases)" %
          ("all passed" if not FAIL else "failures present", len(FAIL), len(CASES)))
    for x in FAIL:
        print("   FAIL:", x)
