# -*- coding: utf-8 -*-
"""EduCEQ-Toolkit

A compact Python toolkit for co-enrichment quotient (CEQ) structure diagnosis.
The core design decision is that **a verdict precedes the estimate**: when the
conditions for an interpretable estimate do not hold, the software says so and
returns no coefficient.

That is a property of the API, not only of the default call. Since version 1.1.0
the functions that can return an estimate require the object issued by
``diagnostics.check`` for the same data, so there is no argument a caller can set
to obtain a number without a verdict. See ``educeq.contract``.

Five modules (matching Section 3.1 of the paper):

    educeq.kernels       construction and composition of kernels
                         (distance, ordinal attribute, categorical attribute;
                         composed multiplicatively)
    educeq.estimator     enrichment matrix, first-order term, second-order
                         term, identity self-check
    educeq.inference     permutation test (including stratified/conditional
                         permutation), false discovery rate control
    educeq.diagnostics   support measures (raw count, weight sum, effective
                         size), the decidability checker, spectral shape check
    educeq.validity      partial correlation, minimum detectable effect,
                         equivalence test, proxy test, discrimination
                         decomposition

Dependencies: numpy, scipy. Every example uses synthetic data; no internal or
sensitive data are included.

Typical use:

    from educeq import kernels, diagnostics, estimator, inference

    W = kernels.distance(coords, bandwidth=150.0)
    dec = diagnostics.check(W, types, target=0)     # the verdict
    if not dec.decidable:
        print(dec)                                   # why, and what to change
    else:
        C = estimator.ceq_matrix(W, types, contract=dec)
        p = inference.permutation(W, types, contract=dec, n_perm=999)

Migrations from 1.0.0: ``estimator.ceq_matrix(W, types)`` now requires
``contract=``; ``estimator.decompose(..., check=False)`` is removed and the
unchecked arithmetic lives in the private ``_decompose_unchecked``;
``inference.permutation`` likewise requires ``contract=``; the verdict reports
the effective neighbourhood size ``ess`` in place of a count of non-zero weights.

Scope (Section 7 of the paper): temporal kernels, random-walk kernels,
embedding kernels and the asymptotic-null inference channel belong to the
general formulation but are not implemented in the released code, and no
empirical validation is claimed for them here.
"""
from . import kernels, estimator, inference, diagnostics, validity, contract

__version__ = "1.1.0"

__all__ = ["kernels", "estimator", "inference", "diagnostics", "validity",
           "contract", "__version__"]
