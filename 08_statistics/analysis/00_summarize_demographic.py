#!/usr/bin/env python3
"""Participant-level demographic summary and Shapiro–Wilk normality checks."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import shapiro

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normality_diagnostics import show_shapiro_diagnostic_plots

# Path to the csv file
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_path = os.path.join(project_root, "data", "participants", "IXI_HH_participants.csv")

# Load the data
df = pd.read_csv(csv_path)
n_total = int(len(df))
print(f"Participants, n = {n_total}")

################################################################################
# Categorical: counts and percentages
################################################################################

SEX_LEVELS = ["male", "female"]
# Use `ethnicity` (white / asian / black / other / unknown), not the collapsed
# `ethnicity_group` (white / asian / black/other/unknown).
ETHNICITY_LEVELS = ["white", "asian", "black", "other", "unknown"]


def _print_count_pct(
    series: pd.Series,
    levels: list[str],
    *,
    title: str,
) -> None:
    s = series.astype(str).str.strip().str.lower()
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    counts = s.value_counts(dropna=False)
    for level in levels:
        n = int(counts.get(level, 0))
        pct = 100.0 * n / n_total if n_total else np.nan
        print(f"  {level:10s}  n = {n:3d}  ({pct:5.1f}%)")
    other = sorted(set(counts.index.astype(str)) - set(levels))
    if other:
        print("  (unexpected levels not in requested list)")
        for level in other:
            n = int(counts[level])
            pct = 100.0 * n / n_total if n_total else np.nan
            print(f"  {level:10s}  n = {n:3d}  ({pct:5.1f}%)")


_print_count_pct(df["sex"], SEX_LEVELS, title="Sex")
_print_count_pct(
    df["ethnicity"],
    ETHNICITY_LEVELS,
    title="Ethnic group (ethnicity: white / asian / black / other / unknown)",
)

################################################################################
# Continuous: mean (SD), median (IQR) + Shapiro–Wilk
################################################################################

CONTINUOUS_SPECS = [
    ("Age, years", "AGE"),
    ("Body mass index, kg/m^2", "bmi"),
    ("Total intracranial volume, cm^3", "TIV"),
    ("Cerebrospinal fluid volume, cm^3", "CSF_absolute_volume"),
    ("White matter hyperintensities volume, cm^3", "vol_abs_WMH"),
]
ALPHA = 0.05


def _continuous_summary(series: pd.Series) -> dict:
    s = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    n = int(s.shape[0])
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "W": np.nan,
            "p": np.nan,
            "normal": False,
            "note": "no valid values",
        }
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    out = {
        "n": n,
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)) if n >= 2 else np.nan,
        "median": float(s.median()),
        "q1": q1,
        "q3": q3,
        "W": np.nan,
        "p": np.nan,
        "normal": False,
        "note": "",
    }
    if n < 3:
        out["note"] = "n < 3 (Shapiro–Wilk not run)"
        return out
    # SciPy Shapiro–Wilk supports n up to 5000
    x = s.to_numpy()
    if n > 5000:
        rng = np.random.default_rng(0)
        x = rng.choice(x, size=5000, replace=False)
        out["note"] = "subsampled to n=5000 for Shapiro–Wilk"
        out["n"] = 5000
    W, p = shapiro(x)
    out["W"] = float(W)
    out["p"] = float(p)
    out["normal"] = bool(p >= ALPHA)
    return out


print("\n" + "=" * 72)
print("Continuous variables: mean (SD), median (IQR)")
print("=" * 72)
rows = []
for label, col in CONTINUOUS_SPECS:
    if col not in df.columns:
        print(f"\n{label} [{col}]: column missing")
        continue
    r = _continuous_summary(df[col])
    rows.append({"variable": label, "column": col, **r})
    print(f"\n{label} [{col}]")
    print(f"  n      = {r['n']}")
    print(f"  mean   = {r['mean']:.4g}  (SD = {r['sd']:.4g})")
    print(
        f"  median = {r['median']:.4g}  "
        f"(IQR = {r['q1']:.4g}–{r['q3']:.4g})"
    )

print("\n" + "=" * 72)
print(f"Shapiro–Wilk normality tests (α = {ALPHA:g})")
print("=" * 72)
print("H0: samples are drawn from a normal distribution. Reject H0 if p < α.")
for r in rows:
    if np.isnan(r["W"]):
        print(f"\n{r['variable']}")
        print(f"  n = {r['n']}; W/p unavailable ({r['note'] or 'NA'})")
        continue
    verdict = "normal" if r["normal"] else "non-normal"
    print(f"\n{r['variable']}")
    print(f"  n = {r['n']}; W = {r['W']:.4f}; p = {r['p']:.4g}  →  {verdict}")
    if r["note"]:
        print(f"  note: {r['note']}")

                                        
    show_shapiro_diagnostic_plots(
        df[r["column"]],
        r["variable"],
        mean=r["mean"],
        median=r["median"],
    )
