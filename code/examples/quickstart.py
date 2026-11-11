# -*- coding: utf-8 -*-
"""Minimal working example of Section 3.4 (reproduced verbatim in the paper).

    cd code && python examples/quickstart.py

The fragment in Section 3.4 of the paper is the block between the two markers
below. The surrounding lines only generate a synthetic dataset of the size at
which the contract passes, so that the example runs anywhere in a few seconds.

Note the shape of the fragment: the verdict comes first and is then passed in.
There is no argument that suppresses it, which is the property Section 3.2
claims and Section 4.7 enforces.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from educeq import kernels, diagnostics, estimator, inference  # noqa: E402

# ---------------------------------------------------------------- synthetic data
rng = np.random.default_rng(0)
n = 12000
xy = rng.uniform(0, 3000, size=(n, 2))        # a domain of continental scale, in km
z = rng.normal(size=n)                        # an ordinal attribute
types = rng.integers(0, 2, n)                 # two categories, ~6000 subjects each
strata = np.digitize(z, np.quantile(z, [0.2, 0.4, 0.6, 0.8]))

# ---------------------------------------------------------------- BEGIN fragment
import numpy as np
from educeq import kernels, diagnostics, estimator, inference

# 1) Construct the kernel (geometric + attributive component)
W = kernels.compose(
    kernels.distance(coords=xy, bandwidth=80.0),
    kernels.ordinal(attr=z, radius=2.0),
)

# 2) Ask whether an estimate is possible. The verdict is the permission to
#    compute: it records the support measures and a fingerprint of these data.
dec = diagnostics.check(W, types, target=1)
if not dec.decidable:
    print(dec.failed, dec.ess, dec.rho_ess)     # no estimate exists yet
else:
    # 3) Estimate, decompose and test. Each call requires the verdict, and a
    #    verdict issued for other data is rejected rather than ignored.
    res = estimator.decompose(W, xy, types, bandwidth=160.0,
                              target=1, contract=dec)
    p = inference.permutation(W, types, contract=dec, n_perm=999, strata=strata)
    print(res["C"], res["C1"], res["C2"], res["identity_ok"], p["z"])
# ---------------------------------------------------------------- END fragment
