# EduCEQ-Toolkit

A Python framework for co-enrichment structure diagnosis with built-in runtime
decidability contracts. Version 1.0.0, MIT licence, dependencies: `numpy` and
`scipy` only.

The toolkit estimates the co-enrichment quotient (CEQ), an exact first- and
second-order decomposition of it, and conditional permutation inference. Its design
feature is that the conditions under which a ratio-type measure can be read are
evaluated **inside the call**: when they are not met, the function returns a
non-decidability status object instead of a coefficient.

## Install

```bash
pip install numpy scipy
pip install -e .          # optional, editable install
```

`educeq` is pure Python and needs no compilation; adding this directory to
`PYTHONPATH` also works.

## Quick start

```python
from educeq import kernels, estimator

W = kernels.compose(
    kernels.distance(coords=xy, bandwidth=150.0),
    kernels.ordinal(attr=z, radius=2.0),
)
res = estimator.decompose(W, xy, types, bandwidth=300.0, target=1)
if not res["decidable"]:
    print(res["failed"], res["diagnostics"]["kappa"], res["diagnostics"]["rho"])
else:
    print(res["C"], res["C1"], res["C2"], res["diagnostics"])
```

The returned dictionary always contains a `decidable` flag and a `diagnostics`
record. When the contract passes it also contains the CEQ matrix `C`, the
first-order term `C1`, the second-order term `C2` and the identity check
`identity_ok`. When the contract fails, no estimate is present at all, so a
downstream table cannot be filled by accident.

## Modules

| Module | Responsibility | Main entry points |
|---|---|---|
| `kernels` | kernel construction and composition | `distance`, `ordinal`, `categorical`, `compose`, `degeneracy_check`, `collinearity_check` |
| `estimator` | enrichment matrix and decomposition | `decompose` (enforces the contract), `ceq_matrix`, `ceq_first_order`, `ceq_second_order`, `normalization_ok` |
| `inference` | permutation testing and multiplicity | `permutation` (with `strata`), `fdr`, `channel_report` |
| `diagnostics` | neighbourhood diagnostics and the contract | `neighbourhood_count`, `coverage`, `check` -> `Decidability`, `spectral_shape` |
| `validity` | whether a conclusion can be used | `partial_corr`, `mde_corr`, `tost_corr`, `proxy_test`, `mediation_boot`, `discrimination_decomposition` |

## Scripts

| Script | Reproduces |
|---|---|
| `examples/quickstart.py` | the minimal working example of Section 3.4 |
| `demo.py` | the decidable example, the undecidable example and the sensitivity example (Sections 5.3 and 5.5) |
| `examples/us_panel.py` | the bandwidth sweep on the College Scorecard panel (Table 6) |
| `calibration.py` | Monte Carlo calibration of the count threshold (Table 7) |
| `attenuation.py` | the attenuation experiment with known ground truth (Table 8) |

All scripts run from this directory:

```bash
python demo.py
python examples/quickstart.py
python examples/us_panel.py
python calibration.py --reps 200
python attenuation.py --reps 1000
```

Console logs of the runs reported in the paper are in `results/`.

## The contract in one place

```python
from educeq import kernels, diagnostics

W = kernels.distance(coords, bandwidth=40.0)
d = diagnostics.check(W, types, target=0)
print(d.decidable)   # False
print(d.failed)      # ['step1:n_target']
print(d.advice)      # target category has 306 objects, below the threshold of 3000; ...
```

| Step | Criterion | On failure |
|---|---|---|
| 1 | target-category count N_B >= kappa_min / rho_max (3000 by default) | stop, nothing downstream runs |
| 2 | median neighbourhood count >= 30 | advise a larger bandwidth or more data |
| 3 | empirical coverage proportion <= 0.01 (measured, not an area ratio) | advise a smaller bandwidth |

## Data

`data/scorecard.npz` holds the analytic sample used in Section 5.2: 4,785 U.S.
post-secondary institutions, one row per institution, with projected coordinates and
a three-level type label. It was derived from the public College Scorecard files of
the U.S. Department of Education; `data/prepare_scorecard.py` regenerates it from the
raw download and documents the field mapping. The raw source files (several hundred
megabytes) are not redistributed here; the derived sample is, so that Table 6 can be
re-checked without downloading anything.

All other scripts generate their own synthetic data and need no download.

## Tests

```bash
python -m pytest tests -q
python tests/test_educeq.py     # the same checks, no pytest required
```

Nine cases, thirty-five checks, all passing. The suite covers kernel composition,
the degeneracy and collinearity checks, the decomposition identity at machine
precision, conditional permutation, the three-step check, the spectral-shape
classification, the validity measures, and the enforcement of the contract inside
`decompose`.

## Scope of the release

The package does not implement temporal, embedding or random-walk kernels, the
asymptotic-null inference channel, or the migration-criteria bridge. These belong to
the general formulation discussed in the paper but are outside the scope of version
1.0.0, and no result in the paper depends on them. `inference.channel_report()`
returns the list of channels that are and are not available, rather than falling back
silently.

Docstrings, inline comments and console output are in English throughout the release.

## Citation

If the toolkit is useful in your work, please cite:

> Wu, J.; Li, F.; He, W.; Wu, X. EduCEQ-Toolkit: A Python Framework for
> Co-Enrichment Structure Diagnosis with Runtime Decidability Contracts.
> *Software* **2026**, submitted.

A versioned archive will be deposited at Zenodo under a DOI upon acceptance.

## Licence

MIT. See `LICENSE`.
