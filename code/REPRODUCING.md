# Reproducing the tables and figures of the paper

Every number in Section 5 of the paper is produced by one of the fifteen scripts
below, with the seeds given here. The console output of each run is stored in
`../results/`, and the command line that produced it is written at the top of the
corresponding log.

Environment: Python 3.11 or later, `numpy` and `scipy` only (see
`requirements.lock`); `hypothesis` is needed for the property tests alone.
Install with

```bash
pip install -r requirements.lock
```

or run the whole set, including the test suite, with

```bash
make all
```

## Table to script

| Paper item | Script | Command | Seed |
|---|---|---|---|
| Table 1, positioning | — | documentation only | — |
| Table 2, kernel families | `educeq/kernels.py` | `pytest tests -k kernels` | — |
| Table 3, module boundaries | — | documentation only | — |
| Table 4, three-step check | `educeq/diagnostics.py` | `pytest tests -k diagnostics` | — |
| Table 5, default parameters | `educeq/diagnostics.py`, `educeq/inference.py` | signatures of `check()` and `fdr_by()` | — |
| Table 6, enforcement ledger | `tests/test_educeq.py` | `pytest tests -k contract` | — |
| Table 7, bandwidth sweep, U.S. panel | `examples/us_panel.py`, `evaluation/support_sensitivity.py` | `python evaluation/support_sensitivity.py` | data only, no seed |
| Table 8, bandwidth sweep, crime panel | `evaluation/second_panel.py` | `EDUCEQ_CRIME=<path> python evaluation/second_panel.py` | 20261110 |
| Table 9, controlled acceptance | `evaluation/accept_reject.py` | `python evaluation/accept_reject.py --reps 200` | 20261109 |
| Table 10, rejection and counterfactual | `evaluation/accept_reject.py` | as above | 20261109 |
| Table 11, attribution of the two terms | `evaluation/decomposition_attribution.py` | `python evaluation/decomposition_attribution.py` | 20261112 |
| Table 12, sensitivity grid, 81 cells | `evaluation/threshold_sensitivity.py` | `python evaluation/threshold_sensitivity.py` | 20261111 |
| Table 13, Monte Carlo standard deviation | `calibration.py` | `python calibration.py --reps 200` | 2024 |
| Table 14, weight heterogeneity | `evaluation/weight_heterogeneity.py` | `python evaluation/weight_heterogeneity.py` | 20261109 |
| §5.6 normalisation conventions | `evaluation/sfdep_parity.py`, block (b) | same command; output `results/sfdep_parity_normalisation.csv` | 20261109 |
| Table 15, end-to-end trace | `evaluation/pipeline_e2e.py` | `python evaluation/pipeline_e2e.py` | 20261109 |
| Table 16, attenuation | `attenuation.py` | `python attenuation.py --reps 1000` | 12345 |
| Table 17, size and FDR under dependence | `evaluation/inference_dependence.py` | `python evaluation/inference_dependence.py` | 20261113 |
| Table 18, manual workflow | — | documentation only | — |
| Table 19, time and peak memory | `evaluation/benchmarks.py` | `python evaluation/benchmarks.py` | 20261114 |
| Table 20, notation | — | documentation only | — |
| Table 21, API | `educeq/*.py` | docstrings | — |

## Section to script

| Paper item | Script | Command |
|---|---|---|
| §3.4 listing | `examples/quickstart.py` | `python examples/quickstart.py` |
| §5.2 first panel | `examples/us_panel.py` | `python examples/us_panel.py` |
| §5.4 decidable example, §5.7 attenuation | `demo.py` | `python demo.py` |
| §5.6 reduction to the published quotient | `evaluation/sfdep_parity.py` | `python evaluation/sfdep_parity.py` |

## The five scripts added for the second revision

| Table | Script | Command | Seed |
|---|---|---|---|
| 8  | `evaluation/second_panel.py` | `python evaluation/second_panel.py` | 20261110 |
| 11 | `evaluation/decomposition_attribution.py` | `python evaluation/decomposition_attribution.py` | 20261112 |
| 12 | `evaluation/threshold_sensitivity.py` | `python evaluation/threshold_sensitivity.py` | 20261111 |
| 17 | `evaluation/inference_dependence.py` | `python evaluation/inference_dependence.py` | 20261113 |
| 19 | `evaluation/benchmarks.py` | `python evaluation/benchmarks.py` | 20261114 |

Table 8 needs the crime archive, which is not redistributed; see
`data/PREPARE_CRIME.md` for the download and the filter that rebuilds it, and set
`EDUCEQ_CRIME` to its path. The other four run on synthetic data alone.

## Tests

`pytest tests -q` reports 29 cases. `tests/test_educeq.py` holds fourteen cases
written as named checks and runs standalone as well
(`python tests/test_educeq.py`, which prints one line per check).
`tests/test_properties.py` holds fifteen cases; six are property based and need
`hypothesis` (6.170.15 at the time of writing), and each generated case is run
over 100 examples.

## Figures

The two figures are drawn in `../figures/src/*.svg` and rasterised by
`../figures/src/build_figures.py` (`rsvg-convert`). They contain no computed
data: every label states a criterion or a value that appears in the text.

## Data

`data/scorecard.npz` is the derived analytic sample of 4,785 U.S. higher
education institutions used in Tables 7 and 15. It is regenerated from the
public College Scorecard download by `data/prepare_scorecard.py`; the field
mapping is documented in `data/SOURCE.md`. The raw source files are not
redistributed. The crime panel behind Table 8 is derived from the monthly files
of `data.police.uk` as described in `data/PREPARE_CRIME.md`; those files are not
redistributed either.

## Tolerances

The test suite asserts the decomposition identity to 1e-9 and the parity check
of §5.6 to 1e-12. Monte Carlo results are reproducible exactly for a given
`numpy` version, because every script draws from `default_rng` with an explicit
seed; across `numpy` versions the last digit of a Monte Carlo standard deviation
may move.
