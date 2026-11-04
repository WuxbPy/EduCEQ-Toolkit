# -*- coding: utf-8 -*-
"""educeq.kernels — construction and composition of kernels (module one)

This module corresponds to Section 2.2 of the paper. The kernels available in
the implementation are listed below; they compose multiplicatively:

| kernel | component | form | parameter |
|---|---|---|---|
| distance | geometric | $\\exp(-d^2/2b^2)$, truncated at 3b | bandwidth b |
| ordinal | attribute | $\\max(0, 1-|\\Delta z|/r)$ | radius r |
| categorical | attribute | $\\mathbb I(c_i=c_j)$ | none |

Any of the three may be built on its own or composed multiplicatively. When a
component is set to the constant one, that dimension carries no constraint and
the composition reduces to the product of the remaining components. This is why
a single formula covers both purely geometric and purely attributive
neighbourhoods.

Two checks run at construction time:
  * degeneracy check    the component is constant on the given data, so it
                        cancels from numerator and denominator and has no effect
  * collinearity check  two components are strongly related (0.7 by default),
                        so the second adds no information

Kernel families not implemented in the released code (temporal decay, random
walk, embedding radial basis) belong to the general formulation; see Section 7,
Limitations, of the paper. This module does not provide them.
"""
from ._core import build_weights
from ._attr import ordinal_kernel, categorical_kernel, combine, is_degenerate

__all__ = [
    "distance", "ordinal", "categorical",
    "compose", "degeneracy_check", "collinearity_check",
]


def distance(coords, bandwidth, cut=3.0):
    """Gaussian distance-decay kernel built from coordinates; sparse, zero diagonal."""
    return build_weights(coords, b=bandwidth, cut=cut)


def ordinal(attr, radius):
    """Ordinal or continuous attribute kernel; tends to the constant one as the
    radius grows, at which point that dimension carries no constraint."""
    return ordinal_kernel(attr, r=radius)


def categorical(codes):
    """Categorical attribute kernel; if the attribute is nested within the target
    type, every cross-type weight vanishes and the kernel is degenerate."""
    return categorical_kernel(codes)


def compose(*mats):
    """Multiplicative composition of several kernels."""
    return combine(*mats)


def degeneracy_check(W, types):
    """Degeneracy check: returns (is_degenerate, explanation)."""
    return is_degenerate(W, types)


def collinearity_check(W1, W2, threshold=0.7):
    """Collinearity check: association of two kernels over their non-zero positions.

    Returns (is_collinear, correlation). The correlation is approximated by the
    overlap rate of the non-zero positions of the two kernels, because an
    element-wise correlation of sparse kernels is too costly at large n, and the
    overlap rate moves in the same direction as the correlation.
    """
    A = W1.tocsr() if hasattr(W1, "tocsr") else W1
    B = W2.tocsr() if hasattr(W2, "tocsr") else W2
    nz1 = set(zip(*A.nonzero()))
    nz2 = set(zip(*B.nonzero()))
    if not nz1 or not nz2:
        return False, 0.0
    inter = len(nz1 & nz2)
    overlap = inter / min(len(nz1), len(nz2))
    return (overlap >= threshold), float(overlap)
