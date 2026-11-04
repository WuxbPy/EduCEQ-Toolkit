# Data source and reproduction notes (main validation)

## 1. Dataset

**College Scorecard**, the institution-level public dataset released by the
U.S. Department of Education.

| Item | Value |
|---|---|
| Publisher | U.S. Department of Education |
| Public location | <https://collegescorecard.ed.gov/data/> |
| Contents | institution-level annual files plus the most recent cross-section file |
| Unit of observation | institution |
| Version | download package of 10 June 2026 (`College_Scorecard_Raw_Data_06032026`) |
| Licence | public data, freely obtainable and usable for research |

## 2. Contents of this directory

| File | Description | Size |
|---|---|---|
| `prepare_scorecard.py` | builds the analysis sample from the raw CSVs; holds the field mapping and the five filters | — |
| `scorecard.npz` | the analysis-ready sample produced by that script, and the file every result in the paper is computed from | about 0.3 MB |
| `SOURCE.md` | this file | — |

The raw source files are not redistributed here. The derived analysis sample is,
because it is small and it lets a reader re-check the main validation without
downloading several hundred megabytes of source files. To rebuild the sample
from scratch, download the raw files as described in Section 3 and run
`prepare_scorecard.py`.

## 3. Reproduction steps

```bash
# 1. download the raw data (about 469 MB compressed, several GB unpacked)
#    https://collegescorecard.ed.gov/data/
#    take "Institution-level data files" and "Most-Recent-Cohorts-Institution"

# 2. point the script at the unpacked directory
cd code
python data/prepare_scorecard.py --raw <unpacked-dir> --out data/scorecard.npz

# 3. run the main validation
python examples/us_panel.py
```

If the raw data have not been downloaded, steps 2 and 3 fall back to a synthetic
sample of the same size, and the output says so explicitly, so that the pipeline
can be checked without network access.

## 4. Filtering (matching Section 5.1 of the paper)

Five steps take the raw records to the analysis sample:

| Step | Operation | Records |
|---|---|---|
| 1 | raw records | 6,273 |
| 2 | the 48 contiguous states plus DC; Alaska, Hawaii and the outlying territories are excluded | — |
| 3 | main campus only | — |
| 4 | currently operating only | — |
| 5 | valid coordinates and valid category labels | **4,785** |

Excluding Alaska, Hawaii and the outlying territories is a technical trade-off:
those areas lie far from the mainland, so including them would enlarge the study
area enormously while leaving institution density very low, and neighbourhoods
would contain too few neighbours.

## 5. Category labels

| Dimension | Category | Count |
|---|---|---|
| Sector | public | 1,765 |
| | private non-profit | 1,586 |
| | private for-profit | 1,434 |
| Level | four-year and above | 2,391 |
| | two-year | 1,177 |
| | less than two-year | 1,217 |

## 6. Coordinate handling

1. **Projection**: longitude and latitude to a local plane (equirectangular, with
   longitude scaled by the mean latitude of the study area).
2. **Back-linking across years**: the coordinate fields of the annual files are
   missing in some years. Where they are, the record is linked back from ZIP code
   to the most recent year centroid, with a small jitter added to break exact
   coincidences. Over all years the back-link covers about 87.5 per cent of
   records.
3. The cross-sectional validation in this package uses **the most recent year
   only** (`Most-Recent-Cohorts-Institution.csv`) and therefore involves no
   back-linking; back-linking is needed only to build the multi-year panel.

## 7. Main validation result (as measured)

Running `examples/us_panel.py` on `scorecard.npz` gives:

| Bandwidth (km) | kappa_B | rho | Verdict |
|---|---|---|---|
| 50 | 10.3 | 0.00546 | undecidable |
| 100 | 23.5 | 0.01376 | undecidable |
| **150** | **43.5** | **0.02752** | **undecidable** |
| 200 | 68.1 | 0.04398 | undecidable |
| 300 | 119.5 | 0.07960 | undecidable |
| 400 | 189.0 | 0.13460 | undecidable |
| 500 | 274.8 | 0.18880 | undecidable |

**Conclusion**: the largest category holds 1,765 objects, below the threshold of
3,000, and over 50 to 500 km no bandwidth satisfies both criteria at once. The
feasible interval is empty and the tool declines to return an estimate, which is
the result reported in Section 5.1 of the paper.

## 8. Data availability

All data used in the paper are either public (College Scorecard) or synthetic.
No internal, licensed or personal data are needed to reproduce any result.
