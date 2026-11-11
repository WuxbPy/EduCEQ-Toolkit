# -*- coding: utf-8 -*-
"""
_infer.py — shared inferential utilities for the CEQ layer
=========================================================
For descriptive association measures such as the co-enrichment quotient, three
inferential questions follow:

  Q1 does the association exist?           partial_corr / mde_corr / tost_corr
  Q2 is it the association of interest?    proxy_test + mediation_boot
  Q3 can it discriminate individuals?      discrimination_decomposition

Propositions (proved and simulated in the companion methodological work)
-----------------------------------------------------------------------
P1 (zero residual under a complete proxy): if X is independent of Y given (Z, C)
   and X = f(Z, C) + e, then plim r_net = 0 and lambda tends to 1.
P2 (zero attenuation under independence): if X is independent of Z given C and Y
   depends on X, then plim lambda = 0.
P3 (exact closed form for the attenuation rate): in a linear three-variable
   system, after standardisation,
      r_net = (r_XY - r_XZ*r_YZ) / sqrt[(1-r_XZ^2)(1-r_YZ^2)],
   hence
      lambda = 1 - |r_XY - r_XZ*r_YZ| / (|r_XY|*sqrt[(1-r_XZ^2)(1-r_YZ^2)]).
   (Numerically the empirical value matches the formula to about 1e-15.)
   Corollaries: (i) when Z fully confounds, so X has no direct effect and
   r_XY = r_XZ*r_YZ, then r_net = 0 and lambda = 1; (ii) when X is independent of
   Z, then r_net = r_XY and lambda = 0; (iii) when r_XZ*r_YZ has the opposite sign
   to r_XY and is larger in magnitude, r_net changes sign and lambda < 0, which is
   suppression.
D1 (zero-contribution proposition): if the score s depends only on a
   within-unit constant, s_i = g(z_{c(i)}), then AUC_within equals 0.5 exactly.
D2 (linear-model invariance): if s is the linear model b'x_ind + g'z_cell, then
   the within-unit ordering is driven by b'x_ind alone and AUC_within is exactly
   independent of whether z_cell is included; any observed change proves the model
   has learned interaction terms.
Identity (exact, not approximate):
   AUC_total = (N_w*AUC_within + N_b*AUC_between) / (N_w + N_b).
"""
import numpy as np
from scipy import stats as st

# ---------------------------------------------------------------- basics

def _as2d(C):
    if C is None:
        return None
    C = np.asarray(C, float)
    if C.ndim == 1:
        C = C[:, None]
    return C if C.shape[1] > 0 else None


def _ncol(C):
    C = _as2d(C)
    return 0 if C is None else C.shape[1]


def _resid(v, C):
    """Residuals after regressing on the columns of C plus an intercept; with
    C=None the values are only centred."""
    v = np.asarray(v, float)
    C = _as2d(C)
    if C is None:
        return v - v.mean()
    D = np.column_stack([np.ones(len(v)), C])
    return v - D @ np.linalg.lstsq(D, v, rcond=None)[0]


def _corr_p(a, b):
    n = len(a)
    if np.std(a) < 1e-14 or np.std(b) < 1e-14:
        return 0.0, 1.0
    r = float(np.corrcoef(a, b)[0, 1])
    if abs(r) >= 1 - 1e-15:
        return r, 0.0
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return r, float(2 * (1 - st.t.cdf(abs(t), n - 2)))


def partial_corr(y, x, C=None):
    """Partial correlation r(x, y | C); returns (r, p, df)."""
    r, p = _corr_p(_resid(x, C), _resid(y, C))
    return r, p, len(y) - 2 - _ncol(C)


def partial_r2(x, Z, C=None):
    """Partial R-squared of Z on x after controlling for C, that is, the share of
    the variance of x that Z explains."""
    e0 = _resid(x, C)
    Zc = _as2d(Z)
    if Zc is None:
        return 0.0
    CZ = Zc if C is None else np.column_stack([_as2d(C), Zc])
    e1 = _resid(x, CZ)
    v0 = np.var(e0)
    return float(0.0 if v0 < 1e-14 else max(0.0, 1 - np.var(e1) / v0))


def mde_corr(n, k_ctrl=0, alpha=0.05, power=0.80):
    """Minimum detectable effect for a partial correlation, by the Fisher-z
    approximation."""
    za = st.norm.ppf(1 - alpha / 2)
    zb = st.norm.ppf(power)
    return float(np.tanh((za + zb) / np.sqrt(max(n - 3 - k_ctrl, 1))))


def tost_corr(r, n, k_ctrl=0, delta=0.10, alpha=0.05):
    """Equivalence test for a correlation (TOST): H0 |rho| >= delta against
    H1 |rho| < delta."""
    se = 1 / np.sqrt(max(n - 3 - k_ctrl, 1))
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    zd = np.arctanh(delta)
    p_low = 1 - st.norm.cdf((z + zd) / se)      # rejects rho <= -delta
    p_high = st.norm.cdf((z - zd) / se)         # rejects rho >= +delta
    return dict(tost_p=max(p_low, p_high), ci90=(np.tanh(z - 1.645 * se), np.tanh(z + 1.645 * se)),
                ci95=(np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)),
                equivalent=bool(max(p_low, p_high) < alpha))


# ---------------------------------------------------------------- tool one: proxy test

def classify_proxy(lam, r2xz, p_base, p_net, alpha=0.05, r_base=None, r_net=None):
    """
    Classification rule for the proxy test. The thresholds were calibrated by
    simulation over six scenarios with 500 replicates each.
    Note 1: suppression is defined by a sign reversal of r_net relative to
    r_base, not by lambda < 0. Under pure noise lambda fluctuates below zero near
    zero, so treating lambda < 0 as suppression produces many false positives
    (agreement with the reference rule only 0.504 in simulation).
    Note 2: lambda and R-squared of X on Z are not equal, since lambda is built on
    correlation coefficients and carries the residual variance ratio. The two are
    parallel diagnostics and should not be conflated.
    """
    if p_base >= alpha or not np.isfinite(lam):
        return 'Q1:no association'
    # suppression = sign reversal together with a significant net association,
    # since otherwise jitter around r_net = 0 would create spurious suppression
    if (r_base is not None and r_net is not None and np.isfinite(r_net)
            and np.sign(r_net) != np.sign(r_base) and p_net < alpha):
        return 'Q2:suppression (sign reversal)'
    if lam <= 0.30:
        return 'Q2:independent effect'
    if lam <= 0.70:
        return 'Q2:partial proxy'
    if lam <= 1.20:
        return 'Q2:proxy'
    return 'Q2:strong proxy / amplification'


def proxy_test(y, x, C=None, Z=None, alpha=0.05, delta=0.10):
    """
    Proxy test: does the association between X and Y survive control for the
    structural variable Z?
    Returns r_base, r_net, the attenuation rate lambda, R-squared of X on Z, the
    equivalence test and the classification.
    """
    C, Z = _as2d(C), _as2d(Z)
    r_base, p_base, df = partial_corr(y, x, C)
    out = dict(r_base=float(r_base), p_base=float(p_base), df=int(df), n=int(len(y)))
    if Z is None:
        out.update(r_net=np.nan, p_net=np.nan, lam=np.nan, r2xz=np.nan, cls=classify_proxy(r_base, 0, p_base, p_base))
        return out
    CZ = Z if C is None else np.column_stack([C, Z])
    r_net, p_net, _ = partial_corr(y, x, CZ)
    lam = float(1 - abs(r_net) / abs(r_base)) if abs(r_base) > 1e-12 else np.nan
    r2xz = partial_r2(x, Z, C)
    out.update(r_net=float(r_net), p_net=float(p_net), lam=lam, r2xz=float(r2xz),
               eq_net=tost_corr(r_net, len(y), _ncol(CZ), delta=delta),
               cls=classify_proxy(lam, r2xz, p_base, p_net, alpha, r_base=r_base, r_net=r_net),
               lam_minus_r2=float(lam - r2xz) if np.isfinite(lam) else np.nan)
    return out


def mediation_boot(y, x, m, C=None, n_boot=1000, seed=0, alpha=0.05):
    """
    Single-mediator bootstrap mediation analysis for x -> m -> y, reporting a, b,
    the indirect, direct and total effects, and a suppression flag.
    Suppression means the direct and indirect effects carry opposite signs.
    """
    y, x, m = np.asarray(y, float), np.asarray(x, float), np.asarray(m, float)
    rng = np.random.default_rng(seed)

    def _fit(idx):
        yy, xx, mm = y[idx], x[idx], m[idx]
        cc = None if C is None else np.asarray(C)[idx]
        ex = _resid(xx, cc); em = _resid(mm, cc); ey = _resid(yy, cc)
        if np.std(ex) < 1e-14 or np.std(em) < 1e-14:
            return np.nan, np.nan, np.nan, np.nan
        a = np.polyfit(ex, em, 1)[0]
        # joint regression y ~ ex + em
        D = np.column_stack([np.ones(len(ex)), ex, em])
        coef = np.linalg.lstsq(D, ey, rcond=None)[0]
        return a, coef[2], coef[1], np.polyfit(ex, ey, 1)[0]

    idx0 = np.arange(len(y))
    a, b, cp, c = _fit(idx0)
    boots = np.full((n_boot, 3), np.nan)
    for t in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        try:
            aa, bb, ccp, _ = _fit(i)
            boots[t] = [aa, bb, ccp]
        except Exception:
            pass
    ok = ~np.isnan(boots).any(1)
    ind = boots[ok, 0] * boots[ok, 1]
    lo, hi = np.percentile(ind, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if ok.sum() > 10 else (np.nan, np.nan)
    return dict(a=float(a), b=float(b), indirect=float(a * b), indirect_ci=(float(lo), float(hi)),
                direct=float(cp), total=float(c), suppression=bool(np.isfinite(a * b) and np.isfinite(cp) and a * b * cp < 0),
                sig_indirect=bool(np.isfinite(lo) and np.isfinite(hi) and not (lo <= 0 <= hi)))


# ---------------------------------------------------------------- tool two: discrimination decomposition

def discrimination_decomposition(y, s, cell, chunk=64):
    """
    Discrimination decomposition, an exact identity:
        AUC_total = (N_w*AUC_within + N_b*AUC_between) / (N_w + N_b)
    Discordant pairs (i, j) are split by whether the two cases share a unit;
    ties count 0.5.
    """
    y = np.asarray(y).astype(int); s = np.asarray(s, float); cell = np.asarray(cell)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError('both positive and negative cases are required')
    nw = db = nb = dw = 0.0
    for i0 in range(0, len(pos), chunk):
        idx = pos[i0:i0 + chunk]
        D = s[idx][:, None] - s[neg][None, :]
        c = (D > 0).astype(float) + 0.5 * (D == 0)
        same = cell[idx][:, None] == cell[neg][None, :]
        nw += c[same].sum(); dw += same.sum()
        nb += c[~same].sum(); db += (~same).sum()
    Nw, Nb = float(dw), float(db)
    aw = nw / Nw if Nw > 0 else np.nan
    ab = nb / Nb if Nb > 0 else np.nan
    tot = (nw + nb) / (Nw + Nb)
    return dict(auc_total=float(tot), auc_within=float(aw), auc_between=float(ab),
                n_pairs_within=int(Nw), n_pairs_between=int(Nb),
                w_within=float(Nw / (Nw + Nb)),
                identity_ok=bool(abs(tot - (Nw * aw + Nb * ab) / (Nw + Nb)) < 1e-12))


def dd_bootstrap(y, s, cell, n_boot=200, seed=0, chunk=256):
    """Cluster-level bootstrap giving 95% confidence intervals for AUC_within and
    AUC_between."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float); cell = np.asarray(cell)
    uc = np.unique(cell)
    buckets = {c: np.where(cell == c)[0] for c in uc}
    rng = np.random.default_rng(seed)
    aw, ab = [], []
    for _ in range(n_boot):
        pick = rng.choice(len(uc), len(uc), replace=True)
        idx = np.concatenate([buckets[uc[k]] for k in pick])
        newc = np.concatenate([np.full(len(buckets[uc[k]]), k) for k in pick])
        if y[idx].sum() in (0, len(idx)):
            continue
        try:
            r = discrimination_decomposition(y[idx], s[idx], newc, chunk=chunk)
        except Exception:
            continue
        aw.append(r['auc_within']); ab.append(r['auc_between'])
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 10 else (np.nan, np.nan)
    return dict(auc_within_ci=ci(aw), auc_between_ci=ci(ab), n_valid=len(aw))


# ---------------------------------------------------------------- summary report

def infer_report(name, y, x, C=None, Z=None, label=None):
    """Return a one-row summary answering Q1 (does the association exist) and Q2
    (what explains it)."""
    pt = proxy_test(y, x, C=C, Z=Z)
    n, k = pt['n'], _ncol(C) + (0 if Z is None else _ncol(Z))
    return dict(variable=name if label is None else label, n=n,
                r_base=round(pt['r_base'], 4), p_base=round(pt['p_base'], 4),
                r_net=round(pt['r_net'], 4) if np.isfinite(pt['r_net']) else np.nan,
                p_net=round(pt['p_net'], 4) if np.isfinite(pt['p_net']) else np.nan,
                attenuation_lambda=round(pt['lam'], 3) if np.isfinite(pt['lam']) else np.nan,
                R2_XZ=round(pt['r2xz'], 3) if np.isfinite(pt['r2xz']) else np.nan,
                lambda_minus_r2=round(pt['lam_minus_r2'], 3) if np.isfinite(pt.get('lam_minus_r2', np.nan)) else np.nan,
                MDE=round(mde_corr(n, k), 3), verdict=pt['cls'])


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n = 3000
    # -- P1 self-check: Z fully confounds, so X has no direct effect and Y is
    #    driven by Z; r_net should be about 0 and lambda should tend to 1.
    #    Note: if Y keeps a direct effect on X, as in Y = 0.5X + e, then lambda no
    #    longer tends to 1 and is not equal to R-squared of X on Z either; the two
    #    are parallel diagnostics and should not be conflated.
    Z = rng.normal(size=(n, 1))
    X = 0.8 * Z[:, 0] + rng.normal(size=n)
    Y = 0.6 * Z[:, 0] + rng.normal(size=n)
    pt = proxy_test(Y, X, Z=Z)
    print(f"P1 self-check: lambda={pt['lam']:.4f} (should tend to 1), "
          f"r_net={pt['r_net']:.4f} (should be about 0), R2_XZ={pt['r2xz']:.4f}")
    # -- P3 self-check: the exact closed form of the partial correlation; the
    #    empirical value and the formula should differ at machine precision
    X2 = 0.8 * Z[:, 0] + rng.normal(size=n)
    Y2 = 0.5 * X2 + 0.6 * Z[:, 0] + rng.normal(size=n)
    pt2 = proxy_test(Y2, X2, Z=Z)
    rxy = np.corrcoef(X2, Y2)[0, 1]
    rxz = np.corrcoef(X2, Z[:, 0])[0, 1]
    ryz = np.corrcoef(Y2, Z[:, 0])[0, 1]
    formula = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    print(f"P3 self-check: r_net={pt2['r_net']:.10f}, closed form={formula:.10f}, "
          f"difference={abs(pt2['r_net'] - formula):.2e}")
    # -- D1 self-check: the score depends only on a within-unit constant, so
    #    AUC_within should be exactly 0.5
    ncell = 50
    cell = rng.integers(0, ncell, n)
    zc = rng.normal(size=ncell)[cell]                     # unit-level constant
    yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-(1.2 * zc + 2.0 * rng.normal(size=n))))).astype(int)
    dd = discrimination_decomposition(yb, zc, cell)
    print(f"D1 self-check: AUC_within={dd['auc_within']:.4f} (should be exactly 0.5), "
          f"AUC_between={dd['auc_between']:.4f}, "
          f"identity={'holds' if dd['identity_ok'] else 'fails'}")
