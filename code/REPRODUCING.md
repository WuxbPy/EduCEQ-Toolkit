# Reproducing the tables and figures of the paper

Every number in Section 5 of the paper is produced by one of the scripts below,
with the seeds given here. The console output of each run is stored in
`../results/`, and the command line that produced it is written at the top of the
corresponding log.

Environment: Python 3.11 or later, `numpy` and `scipy` only (see
`requirements.lock`). Install with

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
| Table 5, default parameters | `educeq/diagnostics.py` | signatures of `check()` | — |
| Table 6, enforcement ledger | `tests/test_educeq.py` | `pytest tests -k contract` | — |
| Table 7, bandwidth sweep | `examples/us_panel.py`, `evaluation/support_sensitivity.py` | `python evaluation/support_sensitivity.py` | data only, no seed |
| Table 8, controlled acceptance | `evaluation/accept_reject.py` | `python evaluation/accept_reject.py --reps 200` | 20261109 |
| Table 9, rejection and counterfactual | `evaluation/accept_reject.py` | as above | 20261109 |
| Table 10, calibration | `calibration.py` | `python calibration.py --reps 200` | 2024 |
| Table 11, weight heterogeneity | `evaluation/weight_heterogeneity.py` | `python evaluation/weight_heterogeneity.py` | 20261109 |
| Table 12, end-to-end trace | `evaluation/pipeline_e2e.py` | `python evaluation/pipeline_e2e.py` | 20261109 |
| Table 13, attenuation | `attenuation.py` | `python attenuation.py --reps 1000` | 12345 |
| Table 14, manual workflow | — | documentation only | — |
| Table 15, notation | — | documentation only | — |
| Table 16, API | `educeq/*.py` | docstrings | — |

## Section to script

| Paper item | Script | Command |
|---|---|---|
| §3.4 listing | `examples/quickstart.py` | `python examples/quickstart.py` |
| §5.3 decidable example, §5.6 attenuation | `demo.py` | `python demo.py` |
| §5.5 reduction to the published quotient | `evaluation/sfdep_parity.py` | `python evaluation/sfdep_parity.py` |

## Figures

The two figures are drawn in `../figures/src/*.svg` and rasterised by
`../figures/src/build_figures.py` (`rsvg-convert`). They contain no computed
data: every label states a criterion or a value that appears in the text.

## Data

`data/scorecard.npz` is the derived analytic sample of 4,785 U.S. higher
education institutions used in Tables 7 and 12. It is regenerated from the
public College Scorecard download by `data/prepare_scorecard.py`; the field
mapping is documented in `data/SOURCE.md`. The raw source files are not
redistributed.

## Tolerances

The test suite asserts the decomposition identity to 1e-9 and the parity check
of §5.5 to 1e-12. Monte Carlo results are reproducible exactly for a given
`numpy` version, because every script draws from `default_rng` with an explicit
seed; across `numpy` versions the last digit of a Monte Carlo standard deviation
may move.
