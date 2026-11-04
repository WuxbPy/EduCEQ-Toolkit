# -*- coding: utf-8 -*-
"""EduCEQ-Toolkit

A compact Python toolkit for co-enrichment quotient (CEQ) structure diagnosis.
The core design decision is that **the decidability check is the default
behaviour**: when the conditions do not hold, the call returns "undecidable"
rather than an uninterpretable number.

Five modules (matching Section 3.1 of the paper):

    educeq.kernels       construction and composition of kernels
                         (distance, ordinal attribute, categorical attribute;
                         composed multiplicatively)
    educeq.estimator     enrichment matrix, first-order term, second-order
                         term, identity self-check
    educeq.inference     permutation test (including stratified/conditional
                         permutation), false discovery rate control
    educeq.diagnostics   neighbourhood count kappa, coverage proportion rho,
                         the decidability checker, spectral shape check
    educeq.validity      partial correlation, minimum detectable effect,
                         equivalence test, proxy test, discrimination
                         decomposition

Dependencies: numpy, scipy. Every example uses synthetic data; no internal or
sensitive data are included.

Typical use:

    from educeq import kernels, estimator, diagnostics, inference

    W = kernels.distance(coords, bandwidth=150.0)
    d = diagnostics.check(W, types, target=0)
    if not d.decidable:
        print(d)          # reports why, and what to change; no estimate
    else:
        C = estimator.ceq_matrix(W, types)
        p = inference.permutation(W, types, n_perm=999)

Scope (Section 7 of the paper): temporal kernels, random-walk kernels,
embedding kernels and the asymptotic-null inference channel belong to the
general formulation but are not implemented in the released code, and no
empirical validation is claimed for them here.
"""
from . import kernels, estimator, inference, diagnostics, validity

__version__ = "1.0.0"

__all__ = ["kernels", "estimator", "inference", "diagnostics", "validity",
           "__version__"]
