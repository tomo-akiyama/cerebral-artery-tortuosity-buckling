#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 100 vessel mesh statistics."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import shapiro

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normality_diagnostics import show_shapiro_diagnostic_plots

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

participants_csv = os.path.join(
    project_root, "data", "participants", "IXI_HH_participants.csv"
)
vessel_meshes_root = os.path.join(
    project_root, "data", "preprocessed", "MRA_preprocessed", "vessel_meshes"
)
output_dir = os.path.join(
    project_root, "results", "morphometrics", "vessel_mesh_data"
)
output_csv = os.path.join(output_dir, "IXI_HH_vessel_mesh_features.csv")

ALPHA = 0.05
RATIO_COL = "total_n_faces_per_surface_area_mm2"
EDGE_MEDIAN_COL = "total_edge_length_median_mm"
RT_VESSEL_LENGTH_COL = "rt_vessel_length_mm"
LT_VESSEL_LENGTH_COL = "lt_vessel_length_mm"
EDGE_LENGTH_COLS = [
    "total_edge_length_mean_vox",
    "total_edge_length_sd_vox",
    "total_edge_length_median_vox",
    "total_edge_length_iqr_vox",
    "total_edge_length_mean_mm",
    "total_edge_length_sd_mm",
    "total_edge_length_median_mm",
    "total_edge_length_iqr_mm",
]


def per_case_csv_path(directory_id: str) -> str:
    return os.path.join(
        vessel_meshes_root, directory_id, f"{directory_id}_vessel_mesh_features.csv"
    )


def merge_vessel_mesh_features(directory_ids):
    frames = []
    missing = []

    for directory_id in directory_ids:
        csv_path = per_case_csv_path(directory_id)
        if not os.path.exists(csv_path):
            missing.append(directory_id)
            continue

        df_case = pd.read_csv(csv_path)
        if "directory_id" not in df_case.columns:
            df_case.insert(0, "directory_id", directory_id)
        else:
            df_case["directory_id"] = directory_id
        frames.append(df_case)

    if missing:
        print(f"Warning: cases with missing CSV: {len(missing)}")
        for did in missing:
            print(f"  - {did}")

    if not frames:
        raise FileNotFoundError(
            f"No CSVs to merge under: {vessel_meshes_root}"
        )

    df = pd.concat(frames, ignore_index=True)

                                             
    cols = list(df.columns)
    front = [c for c in ("directory_id", "IXI_ID") if c in cols]
    edge_cols = [c for c in EDGE_LENGTH_COLS if c in cols]
    rest = [c for c in cols if c not in front and c not in edge_cols]
    df = df[front + edge_cols + rest]

    missing_edge = [c for c in EDGE_LENGTH_COLS if c not in df.columns]
    if missing_edge:
        print(
            "Warning: merged CSV lacks edge-length columns (regenerate case CSVs): "
            + ", ".join(missing_edge)
        )
    else:
        n_with_edge = int(df[EDGE_MEDIAN_COL].notna().sum())
        print(f"Included edge-length columns in merge: {len(EDGE_LENGTH_COLS)} cols, valid cases n = {n_with_edge}")

    return df, missing


def add_face_density(df: pd.DataFrame) -> pd.DataFrame:
    n_faces = pd.to_numeric(df["total_n_faces"], errors="coerce")
    area = pd.to_numeric(df["total_surface_area_mm2"], errors="coerce")
    ratio = n_faces / area.replace(0, np.nan)
    df = df.copy()
    df[RATIO_COL] = ratio
    return df


def summarize_continuous(series: pd.Series) -> dict:
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
            "iqr": np.nan,
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
        "iqr": q3 - q1,
        "W": np.nan,
        "p": np.nan,
        "normal": False,
        "note": "",
    }

    if n < 3:
        out["note"] = "n < 3 (Shapiro–Wilk not run)"
        return out

    x = s.to_numpy()
    if n > 5000:
        rng = np.random.default_rng(0)
        x = rng.choice(x, size=5000, replace=False)
        out["note"] = "subsampled to n=5000 for Shapiro–Wilk"

    W, p = shapiro(x)
    out["W"] = float(W)
    out["p"] = float(p)
    out["normal"] = bool(p >= ALPHA)
    return out


               
ANALYSIS_VARS = [
    ("total_n_faces", "total_n_faces"),
    ("total_surface_area_mm2", "total_surface_area_mm2"),
    ("total_n_faces / total_surface_area_mm2", RATIO_COL),
                                  
    (
        "per-case edge length median (mm) [= median of medians]",
        EDGE_MEDIAN_COL,
    ),
]


def print_edge_median_of_medians(df: pd.DataFrame) -> None:
    if EDGE_MEDIAN_COL not in df.columns:
        print(f"\nWarning: {EDGE_MEDIAN_COL}  missing; skipping median-of-medians")
        return

    r = summarize_continuous(df[EDGE_MEDIAN_COL])
    print("\n" + "=" * 72)
    print("Edge length: median of per-case medians (mm)")
    print("=" * 72)
    print(f"  Columns: {EDGE_MEDIAN_COL}")
    print(f"  n (cases) = {r['n']}")
    print(
        f"  median of medians = {r['median']:.6g} mm  "
        f"(IQR = {r['q1']:.6g}–{r['q3']:.6g} mm; IQR width = {r['iqr']:.6g} mm)"
    )


def print_vessel_length_per_hemisphere(df: pd.DataFrame) -> None:
    missing = [
        c for c in (RT_VESSEL_LENGTH_COL, LT_VESSEL_LENGTH_COL) if c not in df.columns
    ]
    if missing:
        print(
            "\nWarning: vessel length  columns missing; skipping hemisphere summary: "
            + ", ".join(missing)
        )
        return

    rt = pd.to_numeric(df[RT_VESSEL_LENGTH_COL], errors="coerce")
    lt = pd.to_numeric(df[LT_VESSEL_LENGTH_COL], errors="coerce")
    per_hemi = pd.concat([rt, lt], ignore_index=True)
    r = summarize_continuous(per_hemi)

    n_cases = int(len(df))
    print("\n" + "=" * 72)
    print("Vessel length per hemisphere (mm): pooled Rt + Lt")
    print("=" * 72)
    print(f"  Columns: {RT_VESSEL_LENGTH_COL}, {LT_VESSEL_LENGTH_COL}")
    print(f"  n (cases) = {n_cases}")
    print(f"  n (hemispheres) = {r['n']}  (expected {n_cases * 2})")
    print(
        f"  median = {r['median']:.6g} mm / hemisphere  "
        f"(IQR = {r['q1']:.6g}–{r['q3']:.6g} mm; IQR width = {r['iqr']:.6g} mm)"
    )


def print_and_plot_statistics(df: pd.DataFrame) -> None:
    for label, col in ANALYSIS_VARS:
        if col not in df.columns:
            print(f"\nWarning: required column missing; skipping: {col}")
            continue

        print("\n" + "=" * 72)
        print(f"{label}: descriptive statistics")
        print("=" * 72)

        r = summarize_continuous(df[col])
        print(f"  n      = {r['n']}")
        print(f"  mean   = {r['mean']:.6g}  (SD = {r['sd']:.6g})")
        print(
            f"  median = {r['median']:.6g}  "
            f"(IQR = {r['q1']:.6g}–{r['q3']:.6g}; IQR width = {r['iqr']:.6g})"
        )

        print("\n" + "-" * 72)
        print(f"Shapiro–Wilk normality test (α = {ALPHA:g})")
        print("H0: samples are drawn from a normal distribution. Reject H0 if p < α.")

        if np.isnan(r["W"]):
            print(f"  n = {r['n']}; W/p unavailable ({r['note'] or 'NA'})")
            print("  → Summary to use: indeterminate (insufficient data)")
            continue

        verdict = "normal" if r["normal"] else "non-normal"
        print(f"  n = {r['n']}; W = {r['W']:.4f}; p = {r['p']:.4g}  →  {verdict}")
        if r["note"]:
            print(f"  note: {r['note']}")
        if r["normal"]:
            print(
                "  → Summary to use: mean (SD)  "
                f"= {r['mean']:.6g} ({r['sd']:.6g})"
            )
        else:
            print(
                "  → Summary to use: median (IQR)  "
                f"= {r['median']:.6g} ({r['q1']:.6g}–{r['q3']:.6g})"
            )

                                            
        show_shapiro_diagnostic_plots(
            df[col],
            label,
            mean=r["mean"],
            median=r["median"],
        )


def main():
    if not os.path.exists(participants_csv):
        raise FileNotFoundError(f"participants CSV not found: {participants_csv}")

    df_participants = pd.read_csv(participants_csv)
    directory_ids = (
        df_participants["directory_id"].dropna().astype(str).unique().tolist()
    )
    print(f"participants directory_id count: {len(directory_ids)}")

    df, missing = merge_vessel_mesh_features(directory_ids)
    print(f"Merged case count: {len(df)}")
    print(f"Missing case count: {len(missing)}")

    df = add_face_density(df)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    print(f"Merged CSV column count: {df.shape[1]} (including edge-length columns)")

    print_edge_median_of_medians(df)
    print_vessel_length_per_hemisphere(df)
    print_and_plot_statistics(df)


if __name__ == "__main__":
    main()
