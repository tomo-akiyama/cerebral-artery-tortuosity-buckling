#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend classify."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)

from sin_trend_paths import (  # noqa: E402
    BOUNDARY_GATE_ENABLED,
    BOUNDARY_POLICY_QC_ONLY,
    CLASSIFICATION_FINALIZED,
    CLASSIFICATION_REVISION,
    CLASSIFICATION_RULE_FROZEN,
    CLASSIFICATION_VALIDATION_STATUS,
    CORE_COVERAGE_MIN,
    CORE_PROFILE_DISPLAY,
    DEFAULT_R2_THRESHOLD,
    PARTIAL_R2_GATE_ENABLED,
    PARTIAL_R2_POLICY,
    PARTIAL_R2_THRESHOLD,
    R2_THRESHOLDS,
    r2_tag,
)

                                                        
SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL = "Predominantly tangential"
SURFACE_LABEL_SUBSTANTIAL_NORMAL = "Substantial normal component"
SURFACE_EVALUABLE_LABELS = frozenset(
    {
        SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL,
        SURFACE_LABEL_SUBSTANTIAL_NORMAL,
    }
)
           
SURFACE_EVALUABLE_LABELS_LEGACY = frozenset({"In-plane Buckling", "Out-of-plane"})

COLUMN_CORRESPONDENCE = {
    "core_periodic": (
        "eligible + fit_valid + partial_R2>=0.50 + coverage>=1.5 "
        "+ not boundary_exact_low/high"
    ),
    "core_all": "alias of core_periodic",
    "core_inplane": "core_periodic AND predominantly tangential (R_tan>0.75 / R_out<0.25)",
    "periodicity": "Core if core_periodic else NotCore/InvalidFit/NotEligible",
    "surface_constraint": (
        "Predominantly tangential (R_tan>0.75) / Substantial normal component (R_tan≤0.75) "
        "only when Core and surface eval ok "
        "(constraint_coverage>=0.50, squared_displacement_coverage>=0.50, finite R_out; "
        "cortex_coverage is diagnostic only, not a gate)"
    ),
}

                                                   
FORBIDDEN_GATE_SUBSTRINGS = (
    "amp_hat_mm<",
    "amp<0.5",
    "SNR<",
    "SNR<1.5",
    "hr_amp>",
    "hr>0.2",
    "boundary_hit_high_2bins",
    "R2<0.60",
    "R2<0.6",
    "lambda_over_diameter_not_gt_2",
)


def _finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _finite_positive(x: Any) -> bool:
    return _finite(x) and float(x) > 0.0


def parse_tri_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    try:
        if x is pd.NA:  # type: ignore[comparison-overlap]
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, str) and x.strip() == "":
        return None                         
    if x in (1, 1.0, "1", "1.0", "True", "true", "TRUE"):
        return True
    if x in (0, 0.0, "0", "0.0", "False", "false", "FALSE"):
        return False
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    # integer 2 etc. are unknown, not True
    return None


def _as_bool(x: Any) -> bool:
    v = parse_tri_bool(x)
    return False if v is None else bool(v)


def _get(row: Dict[str, Any], key: str, default: Any = np.nan) -> Any:
    if isinstance(row, pd.Series):
        return row[key] if key in row.index else default
    return row.get(key, default)


def fit_valid_from_row(row: Dict[str, Any]) -> bool:
    if str(_get(row, "fit_status", "")) != "ok":
        return False
    need = (
        "R2",
        "partial_R2",
        "lambda_hat_mm",
        "k_hat_mm",
        "amp_comb_mm",
        "coef_uU_cos_mm",
        "coef_uU_sin_mm",
        "coef_uV_cos_mm",
        "coef_uV_sin_mm",
    )
    return all(_finite(_get(row, k)) for k in need) and _finite_positive(_get(row, "lambda_hat_mm"))


def good_fit(row: Dict[str, Any], t: float) -> bool:
    r2 = _get(row, "R2")
    return fit_valid_from_row(row) and _finite(r2) and float(r2) >= float(t)


_COVERAGE_FLOAT_RTOL = 1e-12
_COVERAGE_FLOAT_ATOL = 1e-12


def coverage_meets_minimum(cov: float, min_cov: float = CORE_COVERAGE_MIN) -> bool:
    c = float(cov)
    m = float(min_cov)
    if c >= m:
        return True
    return bool(np.isclose(c, m, rtol=_COVERAGE_FLOAT_RTOL, atol=_COVERAGE_FLOAT_ATOL))


def coverage_ok(row: Dict[str, Any], min_cov: float = CORE_COVERAGE_MIN) -> bool:
    cov = _get(row, "coverage_hat")
    if _finite(cov):
        return coverage_meets_minimum(float(cov), min_cov)
                                          
    L = _get(row, "L_base_mm")
    if not _finite_positive(L):
        L = _get(row, "L_baseline_fit_mm")
    lam = _get(row, "lambda_hat_mm")
    if not _finite_positive(L) or not _finite_positive(lam):
        return False
    return coverage_meets_minimum(float(L) / float(lam), min_cov)


def interval_eligible_from_row(row: Dict[str, Any]) -> bool:
    v = _get(row, "interval_eligible", None)
    if v is not None and not (isinstance(v, float) and not np.isfinite(v)):
        if isinstance(v, (bool, np.bool_, int)) or str(v) in {"0", "1", "True", "False", "true", "false"}:
            return _as_bool(v)
    L = _get(row, "L_full_mm")
    return _finite(L) and float(L) > 0.0


def partial_r2_ok(row: Dict[str, Any], thr: float = PARTIAL_R2_THRESHOLD) -> bool:
    p = _get(row, "partial_R2")
    if not _finite(p):
        return False
    return float(p) >= float(thr)


def boundary_identity_status(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = row.index if isinstance(row, pd.Series) else row
    has_low = "boundary_exact_low" in keys
    has_high = "boundary_exact_high" in keys
    if not has_low or not has_high:
        # try reconstruct from selected_index / n_grid / k vs k_min/k_max
        sel = _get(row, "selected_index", None)
        n_grid = _get(row, "n_grid", None)
        k_hat = _get(row, "k_hat_mm", _get(row, "k_hat", None))
        k_min = _get(row, "k_min_mm", _get(row, "k_min", None))
        k_max = _get(row, "k_max_mm", _get(row, "k_max", None))
        recon_low = recon_high = None
        if _finite(sel) and _finite(n_grid) and int(n_grid) >= 1:
            recon_low = int(sel) == 0
            recon_high = int(sel) == int(n_grid) - 1
        elif _finite(k_hat) and _finite(k_min) and _finite(k_max):
            recon_low = bool(np.isclose(float(k_hat), float(k_min), rtol=0, atol=1e-12))
            recon_high = bool(np.isclose(float(k_hat), float(k_max), rtol=0, atol=1e-12))
        if recon_low is None or recon_high is None:
            return {"excluded": True, "reason": "boundary_identity_missing", "low": None, "high": None}
        if recon_low:
            return {"excluded": True, "reason": "boundary_exact_low", "low": True, "high": recon_high}
        if recon_high:
            return {"excluded": True, "reason": "boundary_exact_high", "low": recon_low, "high": True}
        return {"excluded": False, "reason": "", "low": False, "high": False}

    low = parse_tri_bool(_get(row, "boundary_exact_low", None))
    high = parse_tri_bool(_get(row, "boundary_exact_high", None))
    if low is None or high is None:
        return {"excluded": True, "reason": "boundary_identity_missing", "low": low, "high": high}
    if low:
        return {"excluded": True, "reason": "boundary_exact_low", "low": True, "high": high}
    if high:
        return {"excluded": True, "reason": "boundary_exact_high", "low": False, "high": True}
    return {"excluded": False, "reason": "", "low": False, "high": False}


def boundary_excluded(row: Dict[str, Any]) -> bool:
    return bool(boundary_identity_status(row)["excluded"])


def core_periodic_from_row(row: Dict[str, Any]) -> bool:
    if not interval_eligible_from_row(row):
        return False
    if not fit_valid_from_row(row):
        return False
    if not partial_r2_ok(row):
        return False
    if not coverage_ok(row):
        return False
    if BOUNDARY_GATE_ENABLED and boundary_excluded(row):
        return False
    return True


def surface_eligible_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    status = str(_get(row, "surface_evaluation_status", ""))
    # prefer latent R_out for primary gate; fall back to legacy column
    r_out = _get(row, "R_out_latent", _get(row, "R_out_surface"))
    cons = _get(row, "constraint_coverage")
                                       
    energy = _get(
        row,
        "squared_displacement_coverage_latent",
        _get(
            row,
            "squared_displacement_coverage",
            _get(row, "energy_coverage_latent", _get(row, "energy_coverage")),
        ),
    )
    def threshold_or_default(key: str, default: float) -> float:
        value = _get(row, key, None)
        if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
            return float(default)
        return float(value)

    # Zero is an explicit threshold, not a request to restore the default.
    thr_k = threshold_or_default("constraint_coverage_thresh", 0.50)
    thr_e = threshold_or_default(
        "squared_displacement_coverage_thresh", threshold_or_default("energy_coverage_thresh", 0.50)
    )
                                                                       
                                                                 
    thr_r = threshold_or_default("R_out_thresh", 0.25)
    thr_tan_derived = 1.0 - thr_r
    thr_tan_row = _get(row, "R_tan_thresh", None)
    thresh_pair_inconsistent = False
    if thr_tan_row is not None and _finite(thr_tan_row):
        if abs(float(thr_tan_row) - thr_tan_derived) > 1e-12:
            thresh_pair_inconsistent = True
        thr_tan = thr_tan_derived
    else:
        thr_tan = thr_tan_derived
    r_tan = _get(row, "R_tan_latent", _get(row, "R_tan_surface"))
    if not _finite(r_tan) and _finite(r_out):
        r_tan = 1.0 - float(r_out)
    eval_ok = status == "ok" and _finite(r_out)
                                              
    cov_ok = (
        eval_ok
        and _finite(cons)
        and _finite(energy)
        and float(cons) >= thr_k
        and float(energy) >= thr_e
    )
                                                                                               
    r_tan_ok = eval_ok and _finite(r_tan) and float(r_tan) > thr_tan
    r_in = eval_ok and float(r_out) < thr_r
                                       
    eligible = bool(cov_ok and r_in)
    substantial_normal_possible = bool(cov_ok and eval_ok and not r_in)
    return {
        "surface_eligible": eligible,
        "surface_eval_ok": eval_ok,
        "surface_coverage_ok": cov_ok,
        "surface_evaluable": eval_ok,
        "surface_inplane_ok": eligible,                                 
        "surface_out_of_plane_possible": substantial_normal_possible,       
        "surface_predominantly_tangential": eligible,
        "surface_substantial_normal_possible": substantial_normal_possible,
        "R_tan": float(r_tan) if _finite(r_tan) else float("nan"),
        "R_tan_ok": bool(r_tan_ok),
        "R_tan_thresh_used": float(thr_tan),
        "R_out_thresh_used": float(thr_r),
        "thresh_pair_inconsistent": bool(thresh_pair_inconsistent),
    }


def _periodicity_label(row: Dict[str, Any], is_core: bool) -> str:
    if not interval_eligible_from_row(row):
        return "NotEligible"
    if not fit_valid_from_row(row):
        return "InvalidFit"
    if is_core:
        return "Core"
    return "NotCore"


def _reject_reasons_default(row: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if not interval_eligible_from_row(row):
        reasons.append("not_interval_eligible")
    if not fit_valid_from_row(row):
        reasons.append(f"not_fit_valid:{_get(row, 'fit_status', '')}")
    elif not partial_r2_ok(row):
        p = _get(row, "partial_R2")
        if not _finite(p):
            reasons.append("partial_R2_not_finite")
        else:
            reasons.append(f"partial_R2<{PARTIAL_R2_THRESHOLD:g}")
    if not coverage_ok(row):
        reasons.append("coverage_hat<1.5")
    if BOUNDARY_GATE_ENABLED:
        bstat = boundary_identity_status(row)
        if bstat["reason"]:
            reasons.append(str(bstat["reason"]))
    for r in reasons:
        for bad in FORBIDDEN_GATE_SUBSTRINGS:
            if bad in r:
                raise RuntimeError(f"forbidden gate substring {bad!r} in reject reason {r!r}")
    return reasons


def classify_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["classification_revision"] = CLASSIFICATION_REVISION
    out["classification_finalized"] = bool(CLASSIFICATION_FINALIZED)
    out["classification_rule_frozen"] = bool(CLASSIFICATION_RULE_FROZEN)
    out["classification_validation_status"] = CLASSIFICATION_VALIDATION_STATUS
    out["core_profile"] = CORE_PROFILE_DISPLAY
    out["partial_r2_policy"] = PARTIAL_R2_POLICY
    out["partial_r2_threshold"] = float(PARTIAL_R2_THRESHOLD)
    out["amplitude_gate_enabled"] = False
    out["snr_gate_enabled"] = False
    out["harmonic_ratio_gate_enabled"] = False
    out["partial_r2_gate_enabled"] = bool(PARTIAL_R2_GATE_ENABLED)
    out["boundary_gate_enabled"] = bool(BOUNDARY_GATE_ENABLED)
    out["boundary_policy"] = str(_get(row, "boundary_policy", BOUNDARY_POLICY_QC_ONLY) or BOUNDARY_POLICY_QC_ONLY)

    out["fit_valid"] = int(fit_valid_from_row(row))
    out["interval_eligible"] = int(interval_eligible_from_row(row))
    out["partial_r2_ok"] = int(partial_r2_ok(row))
    out["coverage_ok_flag"] = int(coverage_ok(row))
    out["boundary_excluded"] = int(boundary_excluded(row)) if BOUNDARY_GATE_ENABLED else 0

    surf = surface_eligible_from_row(row)
    out["surface_eligible"] = int(surf["surface_eligible"])
    out["surface_evaluable"] = int(surf["surface_evaluable"])
    out["surface_coverage_ok"] = int(surf["surface_coverage_ok"])
    out["surface_inplane_ok"] = int(surf["surface_inplane_ok"])

                            
    for t in R2_THRESHOLDS:
        tag = r2_tag(t)
        out[f"good_fit_r2_{tag}"] = int(good_fit(row, t))

    is_core = core_periodic_from_row(row)
    out["core_periodic"] = int(is_core)
    out["core_all"] = int(is_core)  # exact alias
    core_in = bool(is_core and surf["surface_eligible"])
    out["core_inplane"] = int(core_in)
    out["periodicity"] = _periodicity_label(row, is_core)
    if is_core and surf["surface_eligible"]:
        out["surface_constraint"] = SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL
    elif is_core and surf["surface_out_of_plane_possible"]:
        out["surface_constraint"] = SURFACE_LABEL_SUBSTANTIAL_NORMAL
    else:
        out["surface_constraint"] = None
    out["plane_class"] = out["surface_constraint"]
    reasons = _reject_reasons_default(row)
    out["reject_reasons"] = "|".join(reasons)
    out["reject_reason_primary"] = reasons[0] if reasons else ""
    return out


def classification_column_names() -> List[str]:
    cols = [
        "classification_revision",
        "classification_finalized",
        "classification_rule_frozen",
        "classification_validation_status",
        "core_profile",
        "partial_r2_policy",
        "partial_r2_threshold",
        "amplitude_gate_enabled",
        "snr_gate_enabled",
        "harmonic_ratio_gate_enabled",
        "partial_r2_gate_enabled",
        "boundary_gate_enabled",
        "boundary_policy",
        "fit_valid",
        "interval_eligible",
        "partial_r2_ok",
        "coverage_ok_flag",
        "boundary_excluded",
        "surface_eligible",
        "surface_evaluable",
        "surface_coverage_ok",
        "surface_inplane_ok",
    ]
    for t in R2_THRESHOLDS:
        tag = r2_tag(t)
        cols.append(f"good_fit_r2_{tag}")
    cols.extend(
        [
            "core_periodic",
            "core_all",
            "core_inplane",
            "periodicity",
            "surface_constraint",
            "plane_class",
            "reject_reasons",
            "reject_reason_primary",
        ]
    )
    return cols


def apply_classification(row: Dict[str, Any]) -> Dict[str, Any]:
    return classify_row(row)


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        for c in classification_column_names():
            if c not in df.columns:
                df[c] = pd.Series(dtype=object)
        return df
    rows = [classify_row(rec) for rec in df.to_dict(orient="records")]
    return pd.DataFrame(rows)


def gui_unlabeled_mask(df: pd.DataFrame, *, mca_col: str = "mca_segment") -> pd.Series:
    core = df["core_periodic"] if "core_periodic" in df.columns else df.get("core_all", pd.Series(False, index=df.index))
    if mca_col not in df.columns:
        unlabeled = pd.Series(True, index=df.index)
    else:
        unlabeled = df[mca_col].isna() | (df[mca_col].astype(str).str.strip() == "") | (df[mca_col].astype(str) == "nan")
    return core.fillna(False).astype(bool) & unlabeled


def stats_extraction_mask(df: pd.DataFrame) -> pd.Series:
    col = "core_inplane"
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def reject_reasons_contain_forbidden_gates(reasons: Any) -> bool:
    text = str(reasons or "")
    return any(s in text for s in FORBIDDEN_GATE_SUBSTRINGS)
