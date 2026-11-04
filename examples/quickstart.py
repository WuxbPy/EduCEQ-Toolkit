# -*- coding: utf-8 -*-
"""Minimal working example of Section 3.4 (reproduced verbatim in the paper).

    cd code && python examples/quickstart.py

The fragment in Section 3.4 of the paper is the block between the two markers below.
The surrounding lines only generate a synthetic dataset of the size at which the
contract passes, so that the example runs anywhere in a few seconds.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from educeq import kernels, estimator, inference  # noqa: E402

# ---------------------------------------------------------------- synthetic data
rng = np.random.default_rng(0)
n = 9000
xy = rng.uniform(0, 3000, size=(n, 2))        # a domain of continental scale, in km
z = rng.normal(size=n)                        # an ordinal attribute
types = rng.integers(0, 2, n)                 # two categories, ~4500 subjects each
strata = np.digitize(z, np.quantile(z, [0.2, 0.4, 0.6, 0.8]))

# ---------------------------------------------------------------- BEGIN fragment
import numpy as np
from educeq import kernels, estimator, inference

# 1) Construct the kernel (geometric + attributive component)
W = kernels.compose(
    kernels.distance(coords=xy, bandwidth=150.0),
    kernels.ordinal(attr=z, radius=2.0),
)

# 2) Estimate. The contract is evaluated inside the call, before any estimate
#    exists, so the failure path returns a status object, not a coefficient.
res = estimator.decompose(W, xy, types,
                          bandwidth=300.0,      # intensity scale, must differ
                          target=1)
if not res["decidable"]:
    print(res["failed"], res["diagnostics"]["kappa"], res["diagnostics"]["rho"])
else:
    C, C1, C2 = res["C"], res["C1"], res["C2"]
    p = inference.permutation(W, types, n_perm=999, strata=strata)
    print(C, C2, p["z"], res["diagnostics"])
# ---------------------------------------------------------------- END fragment
