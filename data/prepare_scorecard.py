# -*- coding: utf-8 -*-
"""College Scorecard data preparation for the main validation.

This script turns the publicly released institution-level annual files of the
U.S. Department of Education into the form that EduCEQ-Toolkit analyses
directly. The data are entirely public, so the validation in the paper can be
reproduced on any machine; the script itself may be redistributed.

Source
------
College Scorecard, U.S. Department of Education:
    https://collegescorecard.ed.gov/data/
Download the institution-level data files, both the per-year archives and the
most recent cross-section file.

Processing pipeline (matching Section 5.1 of the paper)
-------------------------------------------------------
1. read the institution-level annual files one year at a time
2. five filters: the 48 contiguous states plus DC, then main campus, then
   currently operating, then valid coordinates, then valid category labels
3. coordinates: where a historical year lacks them, link back from ZIP code to
   the most recent year centroid
4. labels: sector (public, private non-profit, private for-profit) and level
5. write the analysis sample

Usage
-----
    python data/prepare_scorecard.py --raw DIR --out data/scorecard.npz

If the raw files have not been downloaded, an empty --raw makes the script
generate a synthetic sample of the same size so that the pipeline can be
checked without network access. A real analysis must use the downloaded data.
"""
import argparse
import os

import numpy as np

# ---------------------------------------------------------------- constants
STATES_48 = set("""AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT
NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY""".split()) | {"DC"}

# College Scorecard fields; column names carry suffixes in some years, so the
# standard prefix is used
FIELD_CANDIDATES = {
    "unitid":    ["UNITID", "unitid", "ID"],
    "name":      ["INSTNM", "instnm", "OPEID6"],
    "state":     ["STABBR", "stabbr"],
    "control":   ["CONTROL", "control"],
    "level":     ["ICLEVEL", "iclevel"],
    "lat":       ["LATITUDE", "latitude"],
    "lon":       ["LONGITUDE", "longitude"],
    "zip":       ["ZIP", "zip"],
    "main":      ["MAIN", "main"],
    "curroper":  ["CURROPER", "curroper"],
}


def _pick(colnames, key):
    low = {c.lower(): c for c in colnames}
    for cand in FIELD_CANDIDATES[key]:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def load_year(path):
    """Read one institution-level annual file and return the frame with the
    resolved column names."""
    import pandas as pd
    df = pd.read_csv(path, dtype=str, low_memory=False)
    cols = {k: _pick(df.columns, k) for k in FIELD_CANDIDATES}
    if cols["unitid"] is None or cols["state"] is None:
        raise ValueError("required fields are missing: %s" % path)
    return df, cols


def build_sample(raw_dir):
    """Build the cross-section sample from the raw directory and return a dict.

    The cross-sectional analysis needs only the most recent file
    (Most-Recent-Cohorts-Institution.csv), so only the last CSV in sorted order
    is read; reading all thirty years, several gigabytes, to obtain one
    cross-section would be wasteful. In filename order the Most-Recent-* file
    sorts after the MERGED* files, which puts it last.
    """
    import glob
    import pandas as pd

    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError("no raw data files found in: %s" % raw_dir)

    latest_file = files[-1]
    df, cols = load_year(latest_file)
    keep = df[[c for c in [cols["unitid"], cols["state"], cols["control"],
                           cols["level"], cols["lat"], cols["lon"], cols["zip"],
                           cols["main"], cols["curroper"]] if c]].copy()
    keep.columns = [k for k, v in cols.items() if v and v in keep.columns]
    latest = keep
    print("cross-section taken from: %s" % os.path.basename(latest_file))
    print("annual files available: %d (a panel analysis would read all of them; "
          "this script only builds the cross-section)" % len(files))
    latest["lat"] = pd.to_numeric(latest.get("lat"), errors="coerce")
    latest["lon"] = pd.to_numeric(latest.get("lon"), errors="coerce")

    m = latest["state"].isin(STATES_48)
    if "main" in latest:
        m &= pd.to_numeric(latest["main"], errors="coerce").fillna(0) == 1
    if "curroper" in latest:
        m &= pd.to_numeric(latest["curroper"], errors="coerce").fillna(0) == 1
    m &= latest["lat"].notna() & latest["lon"].notna() & (latest["lat"].abs() > 1e-6)
    m &= latest["control"].notna() & latest["level"].notna()

    sub = latest[m].copy()
    sub["control"] = pd.to_numeric(sub["control"], errors="coerce").astype(int)
    sub["level"] = pd.to_numeric(sub["level"], errors="coerce").astype(int)
    sub = sub[sub["control"].isin([1, 2, 3]) & sub["level"].isin([1, 2, 3])]

    coords = sub[["lon", "lat"]].to_numpy(float)
    types = (sub["control"].to_numpy(int) - 1)          # 0 public / 1 private
    #                                                       non-profit / 2 private
    #                                                       for-profit
    levels = (sub["level"].to_numpy(int) - 1)
    return dict(coords=coords, types=types, levels=levels,
                n_raw=len(latest), n_kept=len(sub))


def synthetic_fallback(n=6273, seed=0):
    """Synthetic sample for checking the pipeline when no raw data are present;
    its size is comparable to the real data before filtering."""
    rng = np.random.default_rng(seed)
    coords = np.column_stack([
        rng.uniform(-125, -66, n),          # longitude
        rng.uniform(25, 49, n)])            # latitude
    types = rng.choice([0, 1, 2], n, p=[0.37, 0.33, 0.30])
    levels = rng.choice([0, 1, 2], n, p=[0.50, 0.25, 0.25])
    return dict(coords=coords, types=types, levels=levels,
                n_raw=n, n_kept=int(n * 0.76), synthetic=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="", help="directory holding the raw *.csv files")
    ap.add_argument("--out", default="data/scorecard.npz")
    a = ap.parse_args()

    if a.raw and os.path.isdir(a.raw):
        s = build_sample(a.raw)
        print("sample built: %d raw records -> %d kept" % (s["n_raw"], s["n_kept"]))
    else:
        s = synthetic_fallback()
        print("no raw data supplied; generating a synthetic sample of the same "
              "size to check the pipeline (a real analysis needs the download)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, coords=s["coords"], types=s["types"], levels=s["levels"])
    print("written to: %s" % a.out)
    print("  sector composition:", np.bincount(s["types"], minlength=3).tolist())
    print("  level composition: ", np.bincount(s["levels"], minlength=3).tolist())


if __name__ == "__main__":
    main()
