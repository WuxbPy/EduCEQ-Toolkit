# Console logs of the runs reported in the paper

Every number in the evaluation section of the paper comes from one of the files
below. They were produced on the same machine, in the order listed, with the
versions in `environment.txt`, which records the three direct dependencies only:
the package imports nothing else.

| File | Command | Reproduces |
|---|---|---|
| `pytest.log` | `python -m pytest tests -q` | the test suite: fourteen cases, forty-nine checks |
| `demo.log` | `python demo.py` | §5.3 decidable example, the three refusal examples of §4, and the attenuation example of §5.6 |
| `quickstart.log` | `python examples/quickstart.py` | the minimal working example of §3.4 |
| `us_panel.log` | `python examples/us_panel.py` | §5.2: the College Scorecard panel, verdict negative at every bandwidth |
| `support_sensitivity.log`, `.csv` | `python evaluation/support_sensitivity.py` | Table 7: raw, weighted and effective support on the panel |
| `accept_reject.log`, `.csv` | `python evaluation/accept_reject.py --reps 200` | Tables 8 and 9: acceptance, rejection, and the estimate the checker forbids |
| `calibration.log`, `.csv` | `python calibration.py --reps 200` | Table 10: Monte Carlo standard deviation against effective size |
| `weight_heterogeneity.log`, `.csv` | `python evaluation/weight_heterogeneity.py` | Table 11: four weight profiles at a fixed raw support |
| `pipeline_e2e.log` | `python evaluation/pipeline_e2e.py` | Table 12: the five stages on three datasets |
| `attenuation.log`, `.csv` | `python attenuation.py --reps 1000` | Table 13: attenuation against analytic population values |
| `sfdep_parity.log`, `.csv` | `python evaluation/sfdep_parity.py` | §5.5: agreement with `sfdep` to machine precision |

Notes on the Monte Carlo tables:

- `calibration.py` uses R = 200 replicates with seeds `2024 + 1000*N + r`; the seed
  of the run is printed in the header of `calibration.log`.
- `accept_reject.py` uses R = 200 replicates per design and per family, seed
  20261109, and 99 permutations per replicate.
- `weight_heterogeneity.py` uses 200 permutations per profile, seed 20261109.
- `attenuation.py` uses R = 1000 replicates with seeds `12345 + r`, n = 2000 per
  replicate; both are printed in the header of `attenuation.log`.
- `pipeline_e2e.py` and `sfdep_parity.py` use seed 20261109.

The data in `demo.log` and `quickstart.log` are synthetic and generated inside the
scripts. `us_panel.log` and `support_sensitivity.log` use `data/scorecard.npz`, the
derived analytic sample of 4,785 institutions; the raw College Scorecard files are
not redistributed.
