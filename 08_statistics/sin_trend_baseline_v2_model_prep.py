#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare sin_trend analysis CSV columns for legacy model-validation scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_COVERAGE_MIN = 1.5
OUTPUT_TAG = "sin_trend"


def _as_bool_series(s: pd.Series) -> pd.Series:
    def _one(x: Any) -> bool:
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return False
        if x in (1, 1.0, "1", "True", "true", "TRUE"):
            return True
        if x in (0, 0.0, "0", "False", "false", "FALSE", ""):
            return False
        try:
            return bool(int(x))
        except (TypeError, ValueError):
            return False

    return s.map(_one)


def _positive_log(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = np.log(x)
    out.loc[~(x > 0)] = np.nan
    out = out.where(np.isfinite(out))
    return out


def _ensure_age_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "AGE" in out.columns:
        out["AGE"] = pd.to_numeric(out["AGE"], errors="coerce")
        return out
    for alt in ("age", "Age"):
        if alt in out.columns:
            out["AGE"] = pd.to_numeric(out[alt], errors="coerce")
            return out
    return out


def add_legacy_log_geometry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "radius_median_mm" not in out.columns:
        if "diameter_ref_mm" in out.columns:
            out["radius_median_mm"] = pd.to_numeric(out["diameter_ref_mm"], errors="coerce") / 2.0
        elif "radius_mm" in out.columns:
            out["radius_median_mm"] = pd.to_numeric(out["radius_mm"], errors="coerce")

    # observed-arc wavelength is primary for legacy bridge
    if "lambda_hat_mm" not in out.columns and "lambda_hat_s_mm" in out.columns:
        out["lambda_hat_mm"] = pd.to_numeric(out["lambda_hat_s_mm"], errors="coerce")
    elif "lambda_hat_s_mm" in out.columns:
        # keep both; prefer s for log if present
        pass

    lam_src = None
    if "lambda_hat_s_mm" in out.columns:
        lam_src = out["lambda_hat_s_mm"]
    elif "lambda_hat_mm" in out.columns:
        lam_src = out["lambda_hat_mm"]
    if lam_src is not None and "log_lambda" not in out.columns:
        out["log_lambda"] = _positive_log(lam_src)

    if "radius_median_mm" in out.columns and "log_r" not in out.columns:
        out["log_r"] = _positive_log(out["radius_median_mm"])
    if "log_lambda" in out.columns and "log_r" in out.columns and "log_lambda_over_r" not in out.columns:
        out["log_lambda_over_r"] = out["log_lambda"] - out["log_r"]

    if "amp_hat_mm" not in out.columns and "amp_comb_mm" in out.columns:
        out["amp_hat_mm"] = pd.to_numeric(out["amp_comb_mm"], errors="coerce")
    return out


def add_legacy_cohort_columns(
    df: pd.DataFrame,
    coverage_min: float = DEFAULT_COVERAGE_MIN,
) -> pd.DataFrame:
    out = df.copy()
    tag = coverage_min
    col_in = f"core_inplane_coverage_{tag}"
    col_out = f"core_outofplane_coverage_{tag}"

    cov = pd.to_numeric(out["coverage_hat"], errors="coerce") if "coverage_hat" in out.columns else None

    if "core_inplane" in out.columns:
        inplane = _as_bool_series(out["core_inplane"])
        if cov is not None and float(coverage_min) > DEFAULT_COVERAGE_MIN:
            inplane = inplane & (cov >= float(coverage_min))
        out[col_in] = inplane
    elif col_in not in out.columns:
        out[col_in] = False

    if "core_outofplane" in out.columns:
        outplane = _as_bool_series(out["core_outofplane"])
        if cov is not None and float(coverage_min) > DEFAULT_COVERAGE_MIN:
            outplane = outplane & (cov >= float(coverage_min))
        out[col_out] = outplane
    elif col_out not in out.columns:
        out[col_out] = False

    # v2 does not use R2>=0.60 core; legacy_* isolation columns only if present
    for r2_tag in ("060", "070", "080"):
        src = f"legacy_core_inplane_r2_{r2_tag}"
        if src not in out.columns and r2_tag == "060" and "legacy_core_inplane_r2_060" in out.columns:
            src = "legacy_core_inplane_r2_060"
        dst = f"core_inplane_r2_{r2_tag}_coverage_{tag}"
        if src in out.columns:
            flag = _as_bool_series(out[src])
            if cov is not None and float(coverage_min) > DEFAULT_COVERAGE_MIN:
                flag = flag & (cov >= float(coverage_min))
            out[dst] = flag

    return out


def filter_analysis_rows(
    df: pd.DataFrame,
    *,
    exclude_label_conflicts: bool = True,
    require_mca_for_strata: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    meta: Dict[str, Any] = {"n_input": int(len(out))}
    if exclude_label_conflicts and "mca_label_conflict" in out.columns:
        mask = _as_bool_series(out["mca_label_conflict"])
        meta["n_label_conflicts_excluded"] = int(mask.sum())
        out = out.loc[~mask].copy()
    else:
        meta["n_label_conflicts_excluded"] = 0
    if require_mca_for_strata and "mca_segment" in out.columns:
        ok = out["mca_segment"].notna() & ~out["mca_segment"].astype(str).str.strip().isin(
            {"", "nan", "none", "null"}
        )
        meta["n_without_mca_segment"] = int((~ok).sum())
    else:
        meta["n_without_mca_segment"] = None
    meta["n_after_filters"] = int(len(out))
    return out, meta


def prepare_model_validation_dataframe(
    df: pd.DataFrame,
    *,
    coverage_min: float = DEFAULT_COVERAGE_MIN,
    exclude_label_conflicts: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if "directory_id" not in df.columns:
        raise ValueError("analysis dataframe missing directory_id")
    work = df.copy()
    work["directory_id"] = work["directory_id"].astype(str)
    work = _ensure_age_column(work)
    work, fmeta = filter_analysis_rows(
        work, exclude_label_conflicts=exclude_label_conflicts
    )
    work = add_legacy_log_geometry(work)
    work = add_legacy_cohort_columns(work, coverage_min=coverage_min)

    required = ["log_r", "log_lambda", "log_lambda_over_r", "AGE", "directory_id"]
    missing = [c for c in required if c not in work.columns]
    meta = {
        **fmeta,
        "coverage_min": float(coverage_min),
        "missing_required_columns": missing,
        "n_core_inplane": int(work[f"core_inplane_coverage_{coverage_min}"].sum())
        if f"core_inplane_coverage_{coverage_min}" in work.columns
        else None,
        "n_core_outofplane": int(work[f"core_outofplane_coverage_{coverage_min}"].sum())
        if f"core_outofplane_coverage_{coverage_min}" in work.columns
        else None,
        "output_tag": str(work["output_tag"].iloc[0])
        if "output_tag" in work.columns and len(work)
        else OUTPUT_TAG,
        "measurement_version": str(work["measurement_version"].iloc[0])
        if "measurement_version" in work.columns and len(work)
        else None,
        "fit_method": str(work["fit_method"].iloc[0])
        if "fit_method" in work.columns and len(work)
        else None,
    }
    return work, meta


def load_analysis_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def cohort_summary(df: pd.DataFrame, coverage_mins: Sequence[float] = (1.5,)) -> pd.DataFrame:
    rows = []
    for cov in coverage_mins:
        col_in = f"core_inplane_coverage_{cov}"
        col_out = f"core_outofplane_coverage_{cov}"
        if col_in in df.columns:
            rows.append({"cohort": col_in, "n_segments": int(df[col_in].sum())})
        if col_out in df.columns:
            rows.append({"cohort": col_out, "n_segments": int(df[col_out].sum())})
    return pd.DataFrame(rows)
