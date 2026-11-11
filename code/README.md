# EduCEQ-Toolkit

A Python framework for co-enrichment structure diagnosis with runtime
decidability contracts. **Version 1.1.0**, MIT licence, dependencies: `numpy`
and `scipy` only.

The toolkit estimates the co-enrichment quotient (CEQ), an exact first- and
second-order decomposition of it, and conditional permutation inference. Its
design feature is that the conditions under which a ratio-type measure can be
read are evaluated **before the estimate exists**, and the verdict is required
by every function that can return one: when the conditions are not met, the call
returns a non-decidability object instead of a coefficient, and no argument
suppresses it.

## Install

```bash
pip install numpy scipy
pip install -e .          # optional, editable install
```

`educeq` is pure Python and needs no compilation; adding this directory to
`PYTHONPATH` also works.

## Quick start

```python
from educeq import diagnostics, estimator, kernels

W = kernels.compose(
    kernels.distance(coords=xy, bandwidth=80.0),
    kernels.ordinal(attr=z, radius=2.0),
)

dec = diagnostics.check(W, types, target=1)      # the verdict
if not dec.decidable:
    print(dec.failed, dec.ess, dec.rho_ess)      # no estimate exists yet
else:
    res = estimator.decompose(W, xy, types, bandwidth=160.0,
                              target=1, contract=dec)
    print(res["C"], res["C1"], res["C2"], res["identity_ok"])
```

The verdict is bound to the data it was computed on: a token issued for another
bandwidth, another label vector or other thresholds is rejected with
`ContractMismatch` rather than silently reused. Passing no verdict raises
`ContractRequired`, and a negative verdict raises `ContractRefused`.

## The three-step check

| Step | Quantity | Criterion |
|---|---|---|
| 1 | `N_B`, size of the target category | ≥ 3,000 (reported; derived from steps 2–3) |
| 2 | `e_B`, median effective size of the target-pair support, `e = (Σw)²/Σw²` | ≥ 30 |
| 3 | `ρ_e`, median neighbourhood effective size divided by the median over all categories | ≤ 0.01 |

The criteria are stated on the effective sample size, not on a count of non-zero
weights: `diagnostics.support()` reports the raw count `n_B`, the weighted sum
`S_B` and the effective size `e_B` side by side, and Table 11 of the paper shows that
four weight profiles with an identical count differ by a factor of 3.3 in the
standard deviation of the estimate.

## Modules

| Module | Responsibility |
|---|---|
| `kernels` | three kernel families, multiplicative composition, degeneracy and collinearity checks |
| `contract` | the verdict object, its fingerprint, and the check that a verdict was issued for these data |
| `estimator` | CEQ matrix, first-order and second-order terms, identity self-check; requires a verdict |
| `inference` | conditional (stratified) permutation, FDR control, channel report; requires a verdict |
| `diagnostics` | the three support measures, the three-step check, spectral-shape check |
| `validity` | partial correlation, minimum detectable effect, equivalence testing, proxy attenuation, mediation |

## Tests

```bash
python -m pytest tests -q        # 14 cases, 49 checks
```

## Reproducing the paper

`REPRODUCING.md` maps every table of the paper to a script, a command line and a
seed. `make all` runs the test suite and every evaluation script and writes the
console logs to `../results/`.

## Migration from 1.0.0

* `estimator.decompose(..., check=False)` no longer exists; the argument has been
  removed rather than deprecated, because a switch that disables a precondition
  turns the contract into documentation.
* `estimator.decompose` keeps a single-call form: with no `contract` it issues the
  verdict itself, and a negative verdict yields the diagnostics with no estimate.
  This is not a bypass — the arithmetic is reached only after a verdict — but if
  you want the verdict in hand before estimating, call `diagnostics.check` first,
  as the quick start above does.
* `estimator.decompose`, `estimator.ceq_matrix` and `inference.permutation` now
  take a `contract` keyword argument, which must be the object returned by
  `diagnostics.check` for the same data.
* `diagnostics.check` now takes `ess_min` (previously `kappa_min`) and evaluates
  steps 2 and 3 on the effective size. `kappa_min` is still accepted as a
  deprecated alias.
* The `Decidability` object exposes `n_target`, `ess`, `rho_ess`, `kappa_raw`,
  `kappa_weighted`, `rho_raw`, `rho_weighted` and `fingerprint`; the old
  `kappa` and `rho` attributes remain as aliases of `ess` and `rho_ess`.

## How to cite

If you use the software, please cite the paper:

> Wu, J.; Li, F.; He, W.; Wu, X. EduCEQ-Toolkit: A Python Framework for
> Co-Enrichment Structure Diagnosis with Runtime Decidability Contracts.
> *Software* (MDPI), 2026, submitted (manuscript ID software-4700273).

If you need to point at the code itself, cite the release tag `v1.1.0`.

## Language

All source files, comments, docstrings, console output and documentation are in
English, so that the package can be read without translation.

## Licence

MIT. See `LICENSE`.
