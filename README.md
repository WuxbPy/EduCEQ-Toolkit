# EduCEQ-Toolkit — Supplementary Material

This archive accompanies the manuscript

> Wu, J.; Li, F.; He, W.; Wu, X. EduCEQ-Toolkit: A Python Framework for
> Co-Enrichment Structure Diagnosis with Runtime Decidability Contracts.
> *Software* (MDPI), 2026, manuscript ID software-4700273.

Latest release: **`v1.2.1`** (20 November 2026), the version described in the
revised manuscript. Release 1.2.1 changes no library code: it corrects the
aggregation of the effective size in the script behind Table 14 and adds the
two normalisation cases to the parity script of §5.6. `code/README.md` records
that change, and the ones of 1.2.0 and 1.1.0 before it.

## Contents

| Path | What it is |
|---|---|
| `code/` | source of `educeq` v1.2.1 (MIT licence), the test suite, the fifteen reproduction scripts and the examples |
| `code/data/` | the derived U.S. analytic sample, its preparation script, the field mapping and the note that rebuilds the crime panel of Section 5.3 |
| `code/tests/` | twenty-nine cases: fourteen named checks, six property-based cases and nine boundary cases |
| `results/` | console logs and CSV output of every run reported in Section 5, plus `environment.txt` |

## Running it

```bash
cd code
pip install -r requirements.lock
python -m pytest tests -q      # 29 cases, 67 checks
python demo.py                 # the examples of Sections 5.4 and 5.7
make all                       # every table of Section 5 that needs no external archive
```

`code/REPRODUCING.md` maps each table of the paper to a script, a command line
and a random seed; `results/README.md` does the same for the logs.

## Getting the code without cloning

The package lives in `code/`, so a pip install from the tag names the
subdirectory:

```bash
pip install "educeq @ git+https://github.com/WuxbPy/EduCEQ-Toolkit@v1.2.1#subdirectory=code"
```

`code/README.md` documents the API, the three-step check and the release history
(1.0.0 → 1.1.0 → 1.2.0 → 1.2.1).

## Data

`code/data/scorecard.npz` is the derived analytic sample of 4,785 U.S.
higher-education institutions used in Tables 7 and 15 (about 0.3 MB).
`code/data/prepare_scorecard.py` regenerates it from the public College
Scorecard download; the raw files (about 422 MB) are not redistributed.
`code/data/SOURCE.md` documents the field mapping.

The crime panel of Section 5.3 (Table 8) comes from the monthly street-level
crime files of `data.police.uk`, published under the Open Government Licence
v3.0. Those files are not redistributed either: `code/data/PREPARE_CRIME.md`
records the force area, the period, the filter and the record counts, and the
script reads the archive from the path in `EDUCEQ_CRIME`.

## Licence

MIT. See `code/LICENSE`.
