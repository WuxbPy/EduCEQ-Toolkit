# EduCEQ-Toolkit

A Python framework for co-enrichment structure diagnosis with runtime
decidability contracts. **Version 1.2.1**, MIT licence, dependencies: `numpy`
and `scipy` only.

Repository: <https://github.com/WuxbPy/EduCEQ-Toolkit>. This file describes tag
`v1.2.1`, released on 20 November 2026. The changes since 1.2.0 are listed under
*Changes in 1.2.1* below, and those of 1.2.0 since 1.1.0 under *Migration from
1.1.0*.

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

Or straight from the tag, without cloning (the package sits in the `code/`
subdirectory of the repository):

```bash
pip install "educeq @ git+https://github.com/WuxbPy/EduCEQ-Toolkit@v1.2.1#subdirectory=code"
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
`S_B` and the effective size `e_B` side by side, and Table 14 of the paper shows that
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
python -m pytest tests -q        # 29 cases, 67 checks
```

`tests/test_educeq.py` holds fourteen cases written as named checks; it also runs
standalone, with `python tests/test_educeq.py`, and prints one line per check.
`tests/test_properties.py` holds fifteen cases, six of which are property based and
need `hypothesis`; each generated case is run over 100 examples. Nine are boundary
cases: an empty neighbourhood, a single category, one object, a target of 2,999
against 3,000, an effective size of exactly 30 with a coverage of exactly 1 per
cent and one neighbour more, duplicate coordinates, a target category with no
objects, and an estimator called without a verdict.

## Reproducing the paper

`REPRODUCING.md` maps every table of the paper to a script, a command line and a
seed. `make all` runs the test suite and every evaluation script and writes the
console logs to `../results/`. The table below lists the scripts one by one.

| Script | What it reproduces |
|---|---|
| `examples/quickstart.py` | the minimal working example of §3.4 |
| `examples/us_panel.py` | the College Scorecard panel of §5.2 |
| `evaluation/support_sensitivity.py` | Table 7, the bandwidth sweep on that panel |
| `evaluation/second_panel.py` | Table 8, the crime panel of §5.3 (needs `EDUCEQ_CRIME`) |
| `evaluation/accept_reject.py` | Tables 9 and 10, controlled acceptance and rejection |
| `evaluation/decomposition_attribution.py` | Table 11, which term carries which mechanism |
| `evaluation/threshold_sensitivity.py` | Table 12, the 81-cell sensitivity grid |
| `calibration.py` | Table 13, Monte Carlo standard deviation against effective size |
| `evaluation/weight_heterogeneity.py` | Table 14, four profiles at a fixed raw support |
| `evaluation/pipeline_e2e.py` | Table 15, the five stages on three datasets |
| `attenuation.py` | Table 16, attenuation against analytic population values |
| `evaluation/inference_dependence.py` | Table 17, size and false discovery rate under dependence |
| `evaluation/benchmarks.py` | Table 19, time and peak memory by object count |
| `evaluation/sfdep_parity.py` | §5.6, agreement with the published quotient, and the two normalisation conventions |
| `demo.py` | the worked examples of §§5.4 and 5.7, on synthetic data |

## Changes in 1.2.1

Release 1.2.1 changes no library code. It corrects one reproduction script and
adds a second case to another, and it is the version described in the revised
manuscript.

* `evaluation/weight_heterogeneity.py` (Table 14) now reads the raw support, the
  weighted support and the effective size from `diagnostics.support`, the
  function the contract calls. The previous version computed the effective size
  over every non-zero entry of the row, including neighbours of the other
  category, and so reported for the binary profile an effective size of about
  121, the whole-row neighbour count, against a raw support of 60; the effective
  size of the target-category support, which is the quantity the checker reads
  and which cannot exceed the raw support, is 60. The script now asserts
  `e <= n` on every profile and prints the whole-row value beside the
  target-pair value so that the two aggregations are distinguishable in the
  output. The standard deviations of the estimate are unchanged, because they do
  not depend on the support measures; the effective-size column and the derived
  product `sd * sqrt(e)` are corrected.
* `evaluation/sfdep_parity.py` (§5.6) now reports two neighbourhood designs
  rather than one: the constant-row-total design of the 80-nearest-neighbour
  list, where the estimator and the published quotient agree to machine
  precision, and a distance band with rows of unequal total, where the pooled
  form of Equation (1) and the subject-averaged form of the published definition
  separate, together with the row-standardised weights that recover the
  published values exactly. The point estimate itself, and every other table, is
  unaffected.

Both changes are confined to the evaluation scripts, to the results they write
and to the documentation. The public API, the test suite (twenty-nine cases) and
every number outside Table 14 are those of 1.2.0.

## Migration from 1.1.0

* `educeq.inference` now implements both false discovery procedures. `fdr` is
  Benjamini-Hochberg and `fdr_by` is Benjamini-Yekutieli; the default remains
  Benjamini-Hochberg. Table 5 had described both since 1.0.0 while the module
  implemented one, and the table is now the description of the code.
* The weight matrix no longer drops a pair at zero distance. `build_weights` used
  to exclude coincident objects from one another, contradicting the weight the
  kernel assigns at distance zero; such a pair now carries that weight. The change
  is visible only on data with duplicate coordinates — 18 shared coordinates among
  4,785 objects in the U.S. sample, 10,152 distinct locations behind 81,959
  offences in the crime panel — and it changes the numbers of §5.3 only.

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
> *Software* (MDPI), 2026, under review (manuscript ID software-4700273).

If you need to point at the code itself, cite the release tag `v1.2.1`.

## Language

All source files, comments, docstrings, console output and documentation are in
English, so that the package can be read without translation.

## Licence

MIT. See `LICENSE`.
