#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend dataset."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

TRUE_TOKENS = {"1", "true", "True", "TRUE", "yes", "YES", "y", "Y"}
FALSE_TOKENS = {"0", "false", "False", "FALSE", "no", "NO", "n", "N", "0.0"}


def parse_bool_series(s: pd.Series, *, unknown_as: Optional[bool] = None) -> pd.Series:
    from sin_trend_classify import parse_tri_bool

    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    for i, v in s.items():
        tb = parse_tri_bool(v)
        if tb is None:
            if unknown_as is not None:
                out.at[i] = unknown_as
            continue
        out.at[i] = tb
    return out


def _pos_log(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    out = np.full(len(v), np.nan, float)
    m = np.isfinite(v.to_numpy(dtype=float, na_value=np.nan)) & (v.to_numpy(dtype=float, na_value=np.nan) > 0)
    out[m] = np.log(v.to_numpy(dtype=float, na_value=np.nan)[m])
    return pd.Series(out, index=x.index)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    r = pd.to_numeric(out.get("radius_median_mm"), errors="coerce") if "radius_median_mm" in out.columns else pd.Series(np.nan, index=out.index)
    D = pd.to_numeric(out.get("diameter_ref_mm"), errors="coerce") if "diameter_ref_mm" in out.columns else (2.0 * r)
                                                                                           
    lam_primary = pd.to_numeric(out.get("lambda_hat_mm"), errors="coerce") if "lambda_hat_mm" in out.columns else pd.Series(np.nan, index=out.index)
    if not np.any(np.isfinite(lam_primary.to_numpy(dtype=float, na_value=np.nan))):
        lam_primary = pd.to_numeric(out.get("lambda_hat_s_mm"), errors="coerce") if "lambda_hat_s_mm" in out.columns else lam_primary
    lam_s = lam_primary
    lam_b = pd.to_numeric(out.get("lambda_baseline_eff_mm"), errors="coerce") if "lambda_baseline_eff_mm" in out.columns else lam_primary
    if not np.any(np.isfinite(lam_b.to_numpy(dtype=float, na_value=np.nan))):
        lam_b = lam_primary
    amp = pd.to_numeric(out.get("amp_comb_mm"), errors="coerce") if "amp_comb_mm" in out.columns else pd.Series(np.nan, index=out.index)
    if "amp_major_mm" in out.columns:
        sig1 = pd.to_numeric(out["amp_major_mm"], errors="coerce")
    elif "sigma1_mm" in out.columns:
        sig1 = pd.to_numeric(out["sigma1_mm"], errors="coerce")
    else:
        sig1 = pd.Series(np.nan, index=out.index)
    L_obs = pd.to_numeric(out.get("L_trimmed_mm"), errors="coerce") if "L_trimmed_mm" in out.columns else pd.Series(np.nan, index=out.index)
    if "L_base_mm" in out.columns:
        L_base = pd.to_numeric(out["L_base_mm"], errors="coerce")
    elif "L_baseline_fit_mm" in out.columns:
        L_base = pd.to_numeric(out["L_baseline_fit_mm"], errors="coerce")
    else:
        L_base = pd.Series(np.nan, index=out.index)

    def ratio(num, den):
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
        res = np.full(len(num), np.nan, float)
        nn = num.to_numpy(dtype=float, na_value=np.nan)
        dd = den.to_numpy(dtype=float, na_value=np.nan)
        m = np.isfinite(nn) & np.isfinite(dd) & (dd > 0)
        res[m] = nn[m] / dd[m]
        return pd.Series(res, index=out.index)

    out["lambda_s_over_r"] = ratio(lam_s, r)
    out["lambda_base_eff_over_r"] = ratio(lam_b, r)
    out["amp_comb_over_r"] = ratio(amp, r)
    out["sigma1_over_r"] = ratio(sig1, r)
    out["lambda_s_over_D"] = ratio(lam_s, D)
    out["L_obs_over_r"] = ratio(L_obs, r)
    out["L_base_over_r"] = ratio(L_base, r)
    out["log_radius_mm"] = _pos_log(r)
    out["log_lambda_hat_s_mm"] = _pos_log(lam_s)
    out["log_lambda_baseline_eff_mm"] = _pos_log(lam_b)
    out["log_baseline_to_observed_length_ratio"] = _pos_log(ratio(L_base, L_obs))
    out["log_lambda_s_over_r"] = _pos_log(out["lambda_s_over_r"])
    out["log_lambda_base_eff_over_r"] = _pos_log(out["lambda_base_eff_over_r"])
    return out


                                                                 
MCA_CONFIRMED_STATUSES = frozenset({"ok", "inherited_shape_verified"})
MCA_UNVERIFIED_STATUSES = frozenset(
    {
        "unset",
        "inherited_id_only_shape_unverified",
        "inherited_stats_match_unverified",
        "pending_recheck",
        "shape_mismatch",
    }
)


def mca_is_m2m4(label: Any) -> bool:
    if label is None or (isinstance(label, float) and not np.isfinite(label)):
        return False
    s = str(label).strip()
    return s in {"M2-3", "M4"}


def mca_label_confirmed(status: Any) -> bool:
    if status is None or (isinstance(status, float) and not np.isfinite(status)):
        return False
    return str(status).strip() in MCA_CONFIRMED_STATUSES


def mca_needs_review(row_or_status: Any, label: Any = None) -> bool:
    if isinstance(row_or_status, dict) or isinstance(row_or_status, pd.Series):
        status = row_or_status.get("label_status") if hasattr(row_or_status, "get") else row_or_status["label_status"] if "label_status" in row_or_status.index else ""
        label = row_or_status.get("mca_segment") if hasattr(row_or_status, "get") else row_or_status.get("mca_segment", label)
        conflict = row_or_status.get("mca_label_conflict", False) if hasattr(row_or_status, "get") else False
        if parse_bool_series(pd.Series([conflict])).fillna(False).iloc[0]:
            return True
    else:
        status = row_or_status
    st = "" if status is None else str(status).strip()
    if st in MCA_UNVERIFIED_STATUSES or st == "":
        return True
    if mca_is_set_label(label) and not mca_label_confirmed(st):
        return True
    return False


def mca_is_set_label(label: Any) -> bool:
    if label is None or (isinstance(label, float) and not np.isfinite(label)):
        return False
    s = str(label).strip()
    return s not in {"", "nan", "None", "null", "NULL"}


def filter_mca_m2m4_core_periodic(df: pd.DataFrame) -> pd.DataFrame:
    core = parse_bool_series(df["core_periodic"]) if "core_periodic" in df.columns else parse_bool_series(df.get("core_all", pd.Series(False, index=df.index)))
    mca = df["mca_segment"].map(mca_is_m2m4) if "mca_segment" in df.columns else pd.Series(False, index=df.index)
    conflict = parse_bool_series(df["mca_label_conflict"]).fillna(False) if "mca_label_conflict" in df.columns else pd.Series(False, index=df.index)
    confirmed = df["label_status"].map(mca_label_confirmed) if "label_status" in df.columns else pd.Series(False, index=df.index)
    return df.loc[core.fillna(False) & mca & ~conflict & confirmed.fillna(False)].copy()


def filter_mca_m2m4_core_inplane(df: pd.DataFrame) -> pd.DataFrame:
    core = parse_bool_series(df["core_inplane"]) if "core_inplane" in df.columns else pd.Series(False, index=df.index)
    mca = df["mca_segment"].map(mca_is_m2m4) if "mca_segment" in df.columns else pd.Series(False, index=df.index)
    conflict = parse_bool_series(df["mca_label_conflict"]).fillna(False) if "mca_label_conflict" in df.columns else pd.Series(False, index=df.index)
    confirmed = df["label_status"].map(mca_label_confirmed) if "label_status" in df.columns else pd.Series(False, index=df.index)
    return df.loc[core.fillna(False) & mca & ~conflict & confirmed.fillna(False)].copy()


def filter_mca_needs_review(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    mask = []
    for _, row in df.iterrows():
        mask.append(mca_needs_review(row))
    return df.loc[pd.Series(mask, index=df.index)].copy()


def filter_paired_wavelength(df: pd.DataFrame) -> pd.DataFrame:
    lam_s = pd.to_numeric(df.get("lambda_hat_s_mm", df.get("lambda_hat_mm")), errors="coerce")
    lam_b = pd.to_numeric(df.get("lambda_baseline_eff_mm"), errors="coerce")
    ok = np.isfinite(lam_s) & (lam_s > 0) & np.isfinite(lam_b) & (lam_b > 0)
    return df.loc[ok].copy()


def filter_paired_amplitude(df: pd.DataFrame) -> pd.DataFrame:
    a = pd.to_numeric(df.get("amp_comb_mm"), errors="coerce")
    if "amp_major_mm" in df.columns:
        s1 = pd.to_numeric(df["amp_major_mm"], errors="coerce")
    else:
        s1 = pd.to_numeric(df.get("sigma1_mm"), errors="coerce")
    ok = np.isfinite(a) & (a >= 0) & np.isfinite(s1) & (s1 >= 0)
    return df.loc[ok].copy()
