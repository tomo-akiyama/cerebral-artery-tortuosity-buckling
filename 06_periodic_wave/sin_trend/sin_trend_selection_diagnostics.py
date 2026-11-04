# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend selection diagnostics."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from sin_trend_classify import parse_tri_bool


def _as_bool_scalar(x: Any) -> bool:
    tb = parse_tri_bool(x)
    return tb is True


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"required boolean column missing: {col}")
    return df[col].map(_as_bool_scalar)


def flow_counts(df: pd.DataFrame, *, participant_col: str = "directory_id") -> Dict[str, Any]:
    required = ("interval_eligible", "fit_valid", "core_periodic", "surface_evaluable", "core_inplane")
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"n_requested": len(df), "status": "missing_columns", "missing_columns": missing}

    elig = _bool_col(df, "interval_eligible")
    fit = _bool_col(df, "fit_valid")
    core_p = _bool_col(df, "core_periodic")
    surf = _bool_col(df, "surface_evaluable")
    core_i = _bool_col(df, "core_inplane")

                 
    fit_status = df["fit_status"].astype(str) if "fit_status" in df.columns else pd.Series("", index=df.index)
    n_fit_ok = int((fit_status == "ok").sum())
    n_fit_exception = int(fit_status.isin(["exception", "ambiguous_groupid"]).sum())
    n_fit_failed_other = int((~fit_status.isin(["ok", "exception", "ambiguous_groupid", ""])).sum())

    stages: Dict[str, Any] = {
        "n_requested": int(len(df)),
                      
        "n_interval_eligible": int(elig.sum()),
        "n_fit_valid": int(fit.sum()),
        "n_core_periodic": int(core_p.sum()),
        "n_surface_evaluable": int(surf.sum()),
        "n_core_inplane": int(core_i.sum()),
                      
        "cumulative_eligible": int(elig.sum()),
        "cumulative_fit_valid": int((elig & fit).sum()),
        "cumulative_core_periodic": int((elig & fit & core_p).sum()),
        "cumulative_surface_evaluable": int((elig & fit & core_p & surf).sum()),
        "cumulative_core_inplane": int((elig & fit & core_p & surf & core_i).sum()),
                   
        "n_fit_ok": n_fit_ok,
        "n_fit_exception_or_ambiguous": n_fit_exception,
        "n_fit_failed_other": n_fit_failed_other,
        "status": "ok",
    }

    if "core_all" in df.columns:
        stages["n_core_all"] = int(_bool_col(df, "core_all").sum())

                        
    if "label_status" in df.columns:
        from sin_trend_dataset import mca_is_m2m4, mca_label_confirmed, mca_needs_review

        st = df["label_status"].fillna("").astype(str).str.strip()
        stages["n_label_confirmed"] = int(st.map(mca_label_confirmed).sum())
        stages["n_label_needs_review"] = int(
            df.apply(lambda r: mca_needs_review(r), axis=1).sum()
        ) if len(df) else 0
        if "mca_segment" in df.columns:
            m2m4 = df["mca_segment"].map(mca_is_m2m4)
            stages["n_mca_m2m4_any_status"] = int(m2m4.sum())
            stages["n_mca_m2m4_confirmed"] = int((m2m4 & st.map(mca_label_confirmed)).sum())
            stages["label_status_counts"] = st.value_counts(dropna=False).astype(int).to_dict()

    if participant_col in df.columns:
        stages["n_participants_requested"] = int(df[participant_col].nunique())
        stages["n_participants_core_periodic"] = int(df.loc[core_p, participant_col].nunique())
        stages["n_participants_core_inplane"] = int(df.loc[core_i, participant_col].nunique())
    return stages


def reject_reason_table(df: pd.DataFrame) -> pd.DataFrame:
    if "reject_reasons" not in df.columns:
        return pd.DataFrame(columns=["reason", "n"])
    counts: Dict[str, int] = {}
    for text in df["reject_reasons"].fillna("").astype(str):
        if not text.strip():
            continue
        for r in text.split("|"):
            r = r.strip()
            if r:
                counts[r] = counts.get(r, 0) + 1
    return pd.DataFrame([{"reason": k, "n": v} for k, v in sorted(counts.items())])
