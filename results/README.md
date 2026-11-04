# Console logs of the runs reported in the paper

Every number in the evaluation section of the paper comes from one of the files below.
They were produced on the same machine, in the order listed, with the versions in
`pip_freeze.txt`.

| File | Command | Reproduces |
|---|---|---|
| `pytest.log` | `python -m pytest tests -q` | the test suite: nine cases, thirty-five checks |
| `demo.log` | `python demo.py` | Sections 5.3 and 5.5: decidable example, undecidable example, sensitivity example |
| `quickstart.log` | `python examples/quickstart.py` | the minimal working example of Section 3.4 |
| `us_panel.log` | `python examples/us_panel.py` | Table 6: bandwidth sweep on the College Scorecard panel |
| `calibration.log`, `calibration.csv` | `python calibration.py --reps 200` | Table 7: Monte Carlo calibration of the count threshold |
| `attenuation.log`, `attenuation.csv` | `python attenuation.py --reps 1000` | Table 8: attenuation experiment with known ground truth |

Notes on the two Monte Carlo tables:

- `calibration.py` uses R = 200 replicates with seeds `2024 + 1000*N + r`; the seed of
  the run is printed in the header of `calibration.log`.
- `attenuation.py` uses R = 1000 replicates with seeds `12345 + r`, n = 2000 per
  replicate; both are printed in the header of `attenuation.log`.

The data in `demo.log` and `quickstart.log` are synthetic and generated inside the
scripts. `us_panel.log` uses `data/scorecard.npz`, the derived analytic sample of
4,785 institutions; the raw College Scorecard files are not redistributed.
