# Console logs of the runs reported in the paper

Every number in the evaluation section of the paper comes from one of the files
below. They were produced on the same machine, in the order listed, with the
versions in `environment.txt`; `educeq` itself imports nothing beyond `numpy` and
`scipy`, and `hypothesis` is needed by the property tests only.

| File | Command | Reproduces |
|---|---|---|
| `pytest.log` | `python -m pytest tests -q` | the test suite: twenty-nine cases, sixty-seven checks |
| `demo.log` | `python demo.py` | §5.4 decidable example, the three refusal examples of §4, and the attenuation example of §5.7 |
| `quickstart.log` | `python examples/quickstart.py` | the minimal working example of §3.4 |
| `us_panel.log` | `python examples/us_panel.py` | §5.2: the College Scorecard panel, verdict negative at every bandwidth |
| `support_sensitivity.log`, `.csv` | `python evaluation/support_sensitivity.py` | Table 7: raw, weighted and effective support on that panel |
| `second_panel.log`, `.json`, `_sweep.csv`, `_top.csv`, `_matrix.csv` | `python evaluation/second_panel.py` | Table 8 and §5.3: the crime panel, twelve bandwidths, the pair ranking and the quotient matrix |
| `accept_reject.log`, `.csv` | `python evaluation/accept_reject.py --reps 200` | Tables 9 and 10: acceptance, rejection, and the estimate the checker forbids |
| `decomposition_attribution.log`, `.csv` | `python evaluation/decomposition_attribution.py` | Table 11: which term carries which mechanism, on three generating processes |
| `threshold_sensitivity.log`, `.csv`, `.json` | `python evaluation/threshold_sensitivity.py` | Table 12: 81 cells, three patterns by three imbalances by three weight profiles by three bandwidths |
| `calibration.log`, `.csv` | `python calibration.py --reps 200` | Table 13: Monte Carlo standard deviation against effective size |
| `weight_heterogeneity.log`, `.csv` | `python evaluation/weight_heterogeneity.py` | Table 14: four weight profiles at a fixed raw support; the CSV reports the target-pair support measures and, beside them, the whole-row effective size |
| `pipeline_e2e.log` | `python evaluation/pipeline_e2e.py` | Table 15: the five stages on three datasets |
| `attenuation.log`, `.csv` | `python attenuation.py --reps 1000` | Table 16: attenuation against analytic population values |
| `inference_dependence.log`, `.json` | `python evaluation/inference_dependence.py` | Table 17: size of the two permutation channels and false discovery rate of the two procedures |
| `benchmarks.log`, `.csv`, `.json` | `python evaluation/benchmarks.py` | Table 19: wall-clock time and peak resident memory by object count |
| `sfdep_parity.log`, `.csv`, `_normalisation.csv` | `python evaluation/sfdep_parity.py` | §5.6: agreement with the published quotient to machine precision, and the two normalisation conventions on a band with unequal row totals |

Notes on the Monte Carlo tables:

- `calibration.py` uses R = 200 replicates with seeds `2024 + 1000*N + r`; the seed
  of the run is printed in the header of `calibration.log`.
- `accept_reject.py` uses R = 200 replicates per design and per family, seed
  20261109, and 99 permutations per replicate.
- `decomposition_attribution.py` uses 60 replicates per process, seed 20261112, at
  a bandwidth of 250 m.
- `threshold_sensitivity.py` uses 100 redraws of the labels per cell, seed
  20261111, on 20,000 points at three bandwidths.
- `inference_dependence.py` uses 200 replicates for the size measurement and 100
  for the false-discovery-rate measurement, 49 permutations per test, seed
  20261113.
- `weight_heterogeneity.py` uses 200 permutations per profile, seed 20261109.
- `attenuation.py` uses R = 1000 replicates with seeds `12345 + r`, n = 2000 per
  replicate; both are printed in the header of `attenuation.log`.
- `pipeline_e2e.py`, `sfdep_parity.py` and `second_panel.py` use seed 20261109,
  20261109 and 20261110 respectively.
- `benchmarks.py` uses seed 20261114. The times are wall-clock and machine
  dependent; the ratios and the growth rates are what the paper reads.

The data in `demo.log` and `quickstart.log` are synthetic and generated inside the
scripts. `us_panel.log` and `support_sensitivity.log` use `data/scorecard.npz`, the
derived analytic sample of 4,785 institutions; the raw College Scorecard files are
not redistributed. `second_panel.log` uses the crime archive of
`code/data/PREPARE_CRIME.md`, which is not redistributed either; set `EDUCEQ_CRIME`
to its path and run `make crime` to regenerate that log alone.
