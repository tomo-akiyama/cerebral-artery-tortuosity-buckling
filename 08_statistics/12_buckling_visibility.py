#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 12 buckling visibility."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import statsmodels.formula.api as smf

STATISTICS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STATISTICS_ROOT.parent
_SIN_TREND_V2 = PROJECT_ROOT / "06_periodic_wave"
if str(_SIN_TREND_V2) not in sys.path:
    sys.path.insert(0, str(_SIN_TREND_V2))

from sin_trend_paths import FRAME_BASELINE, analysis_csv_path  # noqa: E402

DEFAULT_RUN_ID = "run_154_baseline_v2_r0"


def formal_csv_path(project_root: Optional[Path] = None) -> Path:
    root = PROJECT_ROOT if project_root is None else Path(project_root)
    release = root / "data" / "periodic_wave" / "IXI_HH_sin_trend_analysis_dataset.csv"
    if release.is_file():
        return release
    release = root / "data" / "periodic_wave" / "IXI_HH_sin_trend_analysis_dataset.csv"
    if release.is_file():
        return release
    return analysis_csv_path(DEFAULT_RUN_ID, FRAME_BASELINE, root)


def _match_level(series: pd.Series, target: str) -> Optional[str]:
    for value in series.dropna().unique():
        if str(value).strip().lower() == target.strip().lower():
            return str(value)
    return None


def _first_value(series: pd.Series):
    kept = series.dropna()
    return kept.iloc[0] if len(kept) else pd.NA


def aggregate_subject_level(df: pd.DataFrame) -> pd.DataFrame:
    required = ["IXI_ID", "AGE", "total_vessel_length"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    sub = df.copy()
    sub["IXI_ID"] = sub["IXI_ID"].astype(str)
    sub["_row"] = 1
    parts = {
        "AGE": ("AGE", _first_value),
        "n_total": ("_row", "sum"),
        "total_vessel_length": ("total_vessel_length", _first_value),
    }
    for column in ("sex", "ethnicity_group", "TIV_cuberoot"):
        if column in sub.columns:
            parts[column] = (column, _first_value)

    out = sub.groupby("IXI_ID", dropna=False).agg(**parts).reset_index()
    out["AGE"] = pd.to_numeric(out["AGE"], errors="coerce")
    out["n_total"] = pd.to_numeric(out["n_total"], errors="coerce")
    out["total_vessel_length"] = pd.to_numeric(out["total_vessel_length"], errors="coerce")
    if "TIV_cuberoot" in out.columns:
        out["TIV_cuberoot"] = pd.to_numeric(out["TIV_cuberoot"], errors="coerce")
    out = out.dropna(subset=["AGE", "n_total", "total_vessel_length"])
    return out.loc[out["n_total"] > 0].copy()


def load_participant_frame(project_root: Optional[Path] = None) -> tuple[pd.DataFrame, Path]:
    csv_path = formal_csv_path(project_root)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Formal analysis CSV not found: {csv_path}")
    raw = pd.read_csv(csv_path, low_memory=False)
    return aggregate_subject_level(raw), csv_path


def covariate_terms(df: pd.DataFrame, *, include_vessel_length: bool) -> list[str]:
    terms: list[str] = []
    if "sex" in df.columns:
        sex_ref = _match_level(df["sex"], "Female")
        terms.append(
            f"C(sex, Treatment(reference='{sex_ref}'))" if sex_ref is not None else "C(sex)"
        )
    if "ethnicity_group" in df.columns:
        eth_ref = _match_level(df["ethnicity_group"], "White")
        terms.append(
            f"C(ethnicity_group, Treatment(reference='{eth_ref}'))"
            if eth_ref is not None
            else "C(ethnicity_group)"
        )
    if "TIV_cuberoot" in df.columns:
        terms.append("TIV_cuberoot")
    if include_vessel_length and "total_vessel_length" in df.columns:
        terms.append("total_vessel_length")
    return terms


def _columns_in_terms(terms: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for term in terms:
        if term in {"TIV_cuberoot", "total_vessel_length"}:
            columns.append(term)
        elif term.startswith("C(sex"):
            columns.append("sex")
        elif term.startswith("C(ethnicity_group"):
            columns.append("ethnicity_group")
    return columns


def fit_age_ols(df: pd.DataFrame, y_col: str, cov_terms: Iterable[str]):
    terms = list(cov_terms)
    cov_str = " + ".join(terms)
    formula = f"{y_col} ~ AGE + {cov_str}" if cov_str else f"{y_col} ~ AGE"
    subset = ["AGE", y_col, *_columns_in_terms(terms)]
    work = df.dropna(subset=subset).copy()
    if len(work) < 3 or work["AGE"].nunique() < 2:
        raise ValueError(f"{y_col}: insufficient complete cases")
    return smf.ols(formula, data=work).fit(), work


def main() -> None:
    agg, csv_path = load_participant_frame(PROJECT_ROOT)
    print(f"input: {csv_path}")
    print(f"participants: {len(agg)}; intervals: {int(agg['n_total'].sum())}")
    print("columns: " + ", ".join(agg.columns.astype(str)))


if __name__ == "__main__":
    main()
