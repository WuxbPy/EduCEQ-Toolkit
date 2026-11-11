# EduCEQ-Toolkit — Supplementary Material

This archive accompanies the manuscript

> Wu, J.; Li, F.; He, W.; Wu, X. EduCEQ-Toolkit: A Python Framework for
> Co-Enrichment Structure Diagnosis with Runtime Decidability Contracts.
> *Software* (MDPI), 2026, manuscript ID software-4700273.

## Contents

| Path | What it is |
|---|---|
| `code/` | source of `educeq` v1.1.0 (MIT licence), the test suite, the ten reproduction scripts and the examples |
| `results/` | console logs and CSV output of every run reported in Section 5, plus `environment.txt` |

## Running it

```bash
cd code
pip install -r requirements.lock
python -m pytest tests -q      # 14 cases, 49 checks
python demo.py                 # the examples of Sections 5.3 and 5.6
make all                       # regenerates every table of Section 5
```

`code/REPRODUCING.md` maps each table of the paper to a script, a command line
and a random seed; `results/README.md` does the same for the logs.

## Data

`code/data/scorecard.npz` is the derived analytic sample of 4,785 U.S.
higher-education institutions used in Tables 7 and 12 (about 0.3 MB).
`code/data/prepare_scorecard.py` regenerates it from the public College
Scorecard download; the raw files (about 422 MB) are not redistributed.
`code/data/SOURCE.md` documents the field mapping.

## Licence

MIT. See `code/LICENSE`.
