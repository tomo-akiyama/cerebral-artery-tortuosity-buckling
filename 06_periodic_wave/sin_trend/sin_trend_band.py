#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend band."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
# Prefer this package directory on sys.path (same basename modules must not collide).
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)

from sin_trend_paths import (  # noqa: E402
    BAND_GEOMETRY_2D,
    BAND_LEGACY_FIXED_10MM,
    BOUNDARY_POLICY_QC_ONLY,
    DIAMETER_WAVELENGTH_FACTOR,
)
from sin_trend_fitting_helpers import (  # noqa: E402
    K_SELECTION_METHOD,
    R2_DEFINITION,
    choose_k_from_tie_set_legacy_diagnostic,
    choose_k_min_sse,
    fit_sin_trend_two_component,
    harmonic_ratio_from_fit,
)


def _finite_positive(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v) and v > 0.0)


def make_k_grid_bounds(k_min: float, k_max: float, dk: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "ks": np.array([], float),
        "k_min": float(k_min) if np.isfinite(k_min) else float("nan"),
        "k_max": float(k_max) if np.isfinite(k_max) else float("nan"),
        "dk": float(dk) if np.isfinite(dk) else float("nan"),
        "n_grid": 0,
        "appended_k_max": False,
        "final_step": float("nan"),
        "final_step_shorter_than_dk": False,
        "status": "ok",
        "reason": "",
    }
    if not (np.isfinite(k_min) and np.isfinite(k_max) and np.isfinite(dk)):
        out.update(status="invalid_band", reason="non-finite k_min/k_max/dk")
        return out
    if dk <= 0.0:
        out.update(status="invalid_band", reason=f"dk must be positive (got {dk})")
        return out
    if k_max <= k_min:
        out.update(status="invalid_band", reason=f"k_max <= k_min ({k_max} <= {k_min})")
        return out
    n = int(np.floor((k_max - k_min) / dk)) + 1
    ks = k_min + dk * np.arange(max(n, 1), dtype=float)
                               
    ks = ks[ks <= k_max + 1e-12]
    if ks.size == 0:
        ks = np.array([k_min], float)
    appended = False
    final_step = float(dk)
    if float(ks[-1]) < k_max - 1e-12:
        final_step = float(k_max - ks[-1])
        ks = np.append(ks, float(k_max))
        appended = True
                       
    if ks.size >= 2 and abs(float(ks[-1]) - float(ks[-2])) <= 1e-15:
        ks = ks[:-1]
        appended = False
        final_step = float(dk)
    out.update(
        ok=True,
        ks=np.asarray(ks, float),
        n_grid=int(ks.size),
        appended_k_max=bool(appended),
        final_step=final_step,
        final_step_shorter_than_dk=bool(appended and final_step < dk - 1e-15),
        status="ok",
    )
    return out


def geometry_longwave_2D_band(
    *,
    diameter_ref_mm: float,
    voxel_size_mm: float,
    L_fit_mm: Optional[float] = None,
    N_cyc_search: float = 0.5,
    q_fine: int = 8,
    diameter_wavelength_factor: float = DIAMETER_WAVELENGTH_FACTOR,
    L_trimmed_mm: Optional[float] = None,
) -> Dict[str, Any]:
    if L_fit_mm is None and L_trimmed_mm is not None:
        L_fit_mm = float(L_trimmed_mm)
    if not _finite_positive(voxel_size_mm):
        return {"ok": False, "status": "invalid_band", "reason": "voxel_size_mm must be positive"}
    if L_fit_mm is None or not _finite_positive(L_fit_mm):
        return {"ok": False, "status": "invalid_band", "reason": "L_fit_mm (baseline trimmed length) is not finite positive"}
    if not _finite_positive(q_fine) or int(q_fine) < 1:
        return {"ok": False, "status": "invalid_band", "reason": f"q_fine must be >= 1 (got {q_fine})"}
    if not _finite_positive(N_cyc_search):
        return {"ok": False, "status": "invalid_band", "reason": "N_cyc_search must be positive"}
    if not _finite_positive(diameter_wavelength_factor):
        return {"ok": False, "status": "invalid_band", "reason": "diameter_wavelength_factor must be positive"}
    if not _finite_positive(diameter_ref_mm):
        return {
            "ok": False,
            "status": "missing_diameter",
            "reason": "diameter_ref_mm is missing/non-positive; geometry band cannot be formed (no 10 mm fallback)",
        }

    D = float(diameter_ref_mm)
    L = float(L_fit_mm)
    vs = float(voxel_size_mm)
    factor = float(diameter_wavelength_factor)
    lambda_min_mm = factor * D
    lambda_max_mm = L / float(N_cyc_search)
                                                                  
    k_min_mm = 2.0 * np.pi * float(N_cyc_search) / L
    k_max_mm = np.pi / D  # factor=2 → 2π/(2D) = π/D
    if abs(factor - 2.0) > 1e-12:
        k_max_mm = 2.0 * np.pi / lambda_min_mm
    dk_mm = np.pi / (float(q_fine) * L)

                                                                                
    k_min_vox = k_min_mm * vs
    k_max_vox = k_max_mm * vs
    dk_vox = dk_mm * vs
    grid = make_k_grid_bounds(k_min_vox, k_max_vox, dk_vox)
    out: Dict[str, Any] = {
        "ok": bool(grid["ok"]),
        "status": grid["status"],
        "reason": grid["reason"],
        "search_band_mode": BAND_GEOMETRY_2D,
        "diameter_ref_mm": D,
        "diameter_wavelength_factor": factor,
        "lambda_min_geometry_mm": lambda_min_mm,
        "lambda_min_search_mm": lambda_min_mm,
        "lambda_max_search_mm": lambda_max_mm,
        "N_cyc_search": float(N_cyc_search),
        "q_fine": int(q_fine),
        "k_min_mm": float(k_min_mm),
        "k_max_mm": float(k_max_mm),
        "dk_mm": float(dk_mm),
        "k_min": float(k_min_vox),
        "k_max": float(k_max_vox),
        "dk": float(dk_vox),
        "ks": grid["ks"],
        "n_grid": grid["n_grid"],
        "appended_k_max": grid["appended_k_max"],
        "final_step": grid["final_step"],
        "final_step_shorter_than_dk": grid["final_step_shorter_than_dk"],
        "legacy_N_pp_role": "unused_in_geometry_longwave_2D",
        "legacy_ds_eff_role": "unused_in_geometry_longwave_2D",
        "grid_length_basis": "trimmed_baseline_arclength",
        "L_fit_mm": float(L),
    }
    if not grid["ok"]:
        return out
    return out


def legacy_fixed_10mm_band(
    *,
    L_trimmed_mm: float,
    voxel_size_mm: float,
    N_cyc_search: float = 0.5,
    q_fine: int = 8,
    N_pp: float = 20.0,
    ds_eff_vox: float = 1.0,
) -> Dict[str, Any]:
    if not _finite_positive(L_trimmed_mm) or not _finite_positive(voxel_size_mm):
        return {"ok": False, "status": "invalid_band", "reason": "L_trimmed_mm/voxel_size_mm invalid"}
    if not _finite_positive(N_pp) or not _finite_positive(ds_eff_vox) or not _finite_positive(q_fine):
        return {"ok": False, "status": "invalid_band", "reason": "legacy N_pp/ds_eff/q_fine invalid"}
    vs = float(voxel_size_mm)
    L_vox = float(L_trimmed_mm) / vs
    lambda_min_vox = float(N_pp) * float(ds_eff_vox)
    lambda_min_mm = lambda_min_vox * vs
    k_max_vox = 2.0 * np.pi / lambda_min_vox
    k_min_vox = 2.0 * np.pi * float(N_cyc_search) / L_vox
    dk_vox = np.pi / (float(q_fine) * L_vox)
    grid = make_k_grid_bounds(k_min_vox, k_max_vox, dk_vox)
    return {
        "ok": bool(grid["ok"]),
        "status": grid["status"],
        "reason": grid["reason"],
        "search_band_mode": BAND_LEGACY_FIXED_10MM,
        "lambda_min_search_mm": lambda_min_mm,
        "lambda_max_search_mm": float(L_trimmed_mm) / float(N_cyc_search),
        "lambda_min_geometry_mm": float("nan"),
        "N_cyc_search": float(N_cyc_search),
        "q_fine": int(q_fine),
        "k_min_mm": float(k_min_vox / vs),
        "k_max_mm": float(k_max_vox / vs),
        "dk_mm": float(dk_vox / vs),
        "k_min": float(k_min_vox),
        "k_max": float(k_max_vox),
        "dk": float(dk_vox),
        "ks": grid["ks"],
        "n_grid": grid["n_grid"],
        "appended_k_max": grid["appended_k_max"],
        "final_step": grid["final_step"],
        "final_step_shorter_than_dk": grid["final_step_shorter_than_dk"],
        "legacy_N_pp": float(N_pp),
        "legacy_ds_eff": float(ds_eff_vox),
        "legacy_N_pp_role": "used_in_legacy_fixed_10mm",
        "legacy_ds_eff_role": "used_in_legacy_fixed_10mm",
    }


def build_search_band(
    cfg: Dict[str, Any],
    *,
    diameter_ref_mm: Any,
    L_fit_mm: Optional[float] = None,
    L_trimmed_mm: Optional[float] = None,
) -> Dict[str, Any]:
    mode = str(cfg.get("search_band_mode", BAND_GEOMETRY_2D))
    vs = float(cfg["voxel_size"])
    q_fine = int(cfg["q_fine"])
    n_cyc = float(cfg.get("N_cyc_search", cfg.get("N_cyc", 0.5)))
    if L_fit_mm is None and L_trimmed_mm is not None:
        L_fit_mm = float(L_trimmed_mm)
    if mode == BAND_GEOMETRY_2D:
        return geometry_longwave_2D_band(
            L_fit_mm=float(L_fit_mm) if L_fit_mm is not None else None,
            diameter_ref_mm=float(diameter_ref_mm) if diameter_ref_mm is not None else float("nan"),
            voxel_size_mm=vs,
            N_cyc_search=n_cyc,
            q_fine=q_fine,
            diameter_wavelength_factor=float(cfg.get("diameter_wavelength_factor", DIAMETER_WAVELENGTH_FACTOR)),
        )
    if mode == BAND_LEGACY_FIXED_10MM:
        L_leg = float(L_trimmed_mm) if L_trimmed_mm is not None else float(L_fit_mm) if L_fit_mm is not None else float("nan")
        return legacy_fixed_10mm_band(
            L_trimmed_mm=L_leg,
            voxel_size_mm=vs,
            N_cyc_search=n_cyc,
            q_fine=q_fine,
            N_pp=float(cfg.get("legacy_N_pp", 20.0)),
            ds_eff_vox=float(cfg.get("legacy_ds_eff", 1.0)),
        )
    return {"ok": False, "status": "invalid_band", "reason": f"unknown search_band_mode {mode!r}"}


def boundary_diagnostics(
    *,
    selected_index: Optional[int],
    n_grid: int,
    k_hat: float,
    k_min: float,
    k_max: float,
    voxel_size_mm: float,
    lambda_hat_mm: float,
    lambda_min_search_mm: float,
    diameter_ref_mm: float,
    margin_bins: int = 2,
) -> Dict[str, Any]:
    nan = float("nan")
    n = int(n_grid)
    idx = int(selected_index) if selected_index is not None and np.isfinite(selected_index) else None
    exact_low = bool(idx is not None and idx == 0)
    exact_high = bool(idx is not None and n >= 1 and idx == n - 1)
    hit_low_2 = bool(idx is not None and idx <= int(margin_bins))
    hit_high_2 = bool(idx is not None and n >= 1 and idx >= n - 1 - int(margin_bins))
    k_hat_mm = float(k_hat) / voxel_size_mm if np.isfinite(k_hat) and voxel_size_mm > 0 else nan
    k_max_mm = float(k_max) / voxel_size_mm if np.isfinite(k_max) and voxel_size_mm > 0 else nan
    dist_high = k_max_mm - k_hat_mm if np.isfinite(k_max_mm) and np.isfinite(k_hat_mm) else nan
    dist_rel = dist_high / k_max_mm if np.isfinite(dist_high) and np.isfinite(k_max_mm) and k_max_mm != 0 else nan
    lam_over_min = (
        float(lambda_hat_mm) / float(lambda_min_search_mm)
        if _finite_positive(lambda_hat_mm) and _finite_positive(lambda_min_search_mm)
        else nan
    )
    lam_over_d = (
        float(lambda_hat_mm) / float(diameter_ref_mm)
        if _finite_positive(lambda_hat_mm) and _finite_positive(diameter_ref_mm)
        else nan
    )
    return {
        "boundary_policy": BOUNDARY_POLICY_QC_ONLY,
        "boundary_gate_enabled": False,
        "boundary_margin_bins": int(margin_bins),
        "boundary_exact_low": int(exact_low),
        "boundary_exact_high": int(exact_high),
        "boundary_hit_low_2bins": int(hit_low_2),
        "boundary_hit_high_2bins": int(hit_high_2),
        "boundary_hit_low": int(hit_low_2),                      
        "boundary_hit_high": int(hit_high_2),
        "boundary_distance_high_k_mm": dist_high,
        "boundary_distance_high_relative": dist_rel,
        "lambda_over_search_min": lam_over_min,
        "lambda_over_diameter": lam_over_d,
    }


def search_k_grid(
    uU,
    uV,
    s,
    ks: np.ndarray,
    *,
    k_min: float,
    k_max: float,
    dk: float,
    ds: float,
    weights=None,
    trend_degree: int = 1,
    verbose: bool = False,
    voxel_size_mm: float = 0.5,
    compute_legacy_tie_diagnostic: bool = True,
) -> Dict[str, Any]:
    uU = np.asarray(uU, float)
    uV = np.asarray(uV, float)
    s = np.asarray(s, float)
    ks = np.asarray(ks, float)
    L = float(s[-1] - s[0]) if len(s) >= 2 else float("nan")
    meta = dict(
        r2_definition=R2_DEFINITION,
        k_selection_method=K_SELECTION_METHOD,
        q_fine_note="dk supplied by caller; not recomputed from N_pp*ds_eff",
        voxel_size_mm=float(voxel_size_mm),
        SSE_unit="voxel^2",
    )
    empty = dict(
        status="invalid_band",
        status_reason="",
        k_hat=float("nan"),
        lambda_hat=float("nan"),
        R2=float("nan"),
        partial_R2=float("nan"),
        SSE_full=float("nan"),
        SSE_tr=float("nan"),
        SST=float("nan"),
        candidates=[],
        ks=ks,
        k_min=float(k_min),
        k_max=float(k_max),
        dk=float(dk),
        n_grid=int(len(ks)),
        selected_index=None,
        n_exact_sse_ties=0,
        **meta,
    )
    if len(s) < 2 or not np.isfinite(L) or L <= 0.0:
        empty.update(status="nonpositive_length", status_reason="invalid s / non-positive length")
        return empty
    if not (np.isfinite(uU).all() and np.isfinite(uV).all() and np.isfinite(s).all()):
        empty.update(status="nonfinite_input", status_reason="non-finite uU/uV/s")
        return empty
    if len(ks) < 1 or not np.isfinite(ks).all():
        empty.update(status="invalid_band", status_reason="empty or non-finite k grid")
        return empty

    candidates: List[Dict[str, Any]] = []
    for cand_i, k in enumerate(ks):
        fit = fit_sin_trend_two_component(uU, uV, s, k, weights=weights, trend_degree=trend_degree)
        hr_amp = float("nan")
        hr_energy = float("nan")
        hr_status = "ok"
        hr_reason = ""
        try:
            hr = harmonic_ratio_from_fit(uU=uU, uV=uV, s=s, k=k, ds=ds, weights=weights, trend_degree=trend_degree)
            hr_amp = float(hr["hr_amp"])
            hr_energy = float(hr["hr_energy"])
            if not np.isfinite(hr_amp):
                hr_status = "nonfinite"
                hr_reason = "hr_amp non-finite"
        except Exception as exc:  # noqa: BLE001
            hr_status = "failed"
            hr_reason = str(exc)
        candidates.append(
            {
                "candidate_index": int(cand_i),
                "k": float(k),
                "lambda": float(fit["lamb"]) if np.isfinite(fit.get("lamb", np.nan)) else float("nan"),
                "R2": float(fit["R2"]) if fit["R2"] is not None else float("nan"),
                "partial_R2": float(fit["partial_R2"]) if fit["partial_R2"] is not None else float("nan"),
                "SSE_full": float(fit["SSE_full"]) if fit["SSE_full"] is not None else float("nan"),
                "SSE_tr": float(fit["SSE_tr"]) if fit["SSE_tr"] is not None else float("nan"),
                "SST": float(fit["SST"]) if fit["SST"] is not None else float("nan"),
                "hr_energy": hr_energy,
                "hr_amp": hr_amp,
                "hr_status": hr_status,
                "hr_status_reason": hr_reason,
                "fit_status": fit["fit_status"],
                "fit_status_reason": fit.get("fit_status_reason", ""),
                "rank_full": fit.get("rank_full", np.nan),
                "rank_tr": fit.get("rank_tr", np.nan),
                "rcond": fit.get("rcond", np.nan),
                "fit": fit,
            }
        )

    try:
        best, selected_index, n_exact_ties = choose_k_min_sse(candidates)
    except ValueError as exc:
        empty.update(
            status="no_valid_candidate",
            status_reason=str(exc),
            candidates=candidates,
        )
        return empty

    fit_best = best["fit"]
    k_hat = float(best["k"])
    legacy_diag = None
    if compute_legacy_tie_diagnostic:
        try:
            legacy_best, legacy_tie = choose_k_from_tie_set_legacy_diagnostic(candidates)
            legacy_diag = dict(
                k_hat=float(legacy_best["k"]),
                lambda_hat=float(legacy_best["lambda"]),
                R2=float(legacy_best["R2"]),
                partial_R2=float(legacy_best["partial_R2"]),
                SSE_full=float(legacy_best["SSE_full"]),
                n_tie=len(legacy_tie),
                note="diagnostic_only_corrected_R2_with_legacy_tie_rule",
            )
        except Exception as exc:  # noqa: BLE001
            legacy_diag = dict(status="failed", status_reason=str(exc))

    if verbose:
        print(f"  L = {L:.4f} (same unit as s)")
        print(f"  dk = {dk:.6g}, n_grid = {len(ks)}")
        print(
            f"  selected (grid optimum) k = {k_hat:.6g}, lambda = {best['lambda']:.4f}, "
            f"R2 = {best['R2']:.4f}, partial_R2 = {best['partial_R2']:.4f}, "
            f"SSE_full = {best['SSE_full']:.6g}, exact_sse_ties = {n_exact_ties}"
        )

    status = fit_best.get("fit_status", "ok")
    status_reason = fit_best.get("fit_status_reason", "")
    if status == "ok" and not np.isfinite(best["R2"]):
        status = "nonfinite_r2"
        status_reason = "selected candidate has non-finite R2"

    return dict(
        status=status,
        status_reason=status_reason,
        k_hat=k_hat,
        lambda_hat=float(best["lambda"]),
        R2=float(best["R2"]),
        partial_R2=float(best["partial_R2"]),
        hr_amp=float(best["hr_amp"]),
        hr_energy=float(best["hr_energy"]),
        hr_status=best.get("hr_status", ""),
        hr_status_reason=best.get("hr_status_reason", ""),
        A_comb=float(fit_best["A_comb"]),
        A_U=float(fit_best["A_U"]),
        A_V=float(fit_best["A_V"]),
        uU_cos=float(fit_best["uU_cos"]),
        uU_sin=float(fit_best["uU_sin"]),
        uV_cos=float(fit_best["uV_cos"]),
        uV_sin=float(fit_best["uV_sin"]),
        coef_U=fit_best["coef_U"],
        coef_V=fit_best["coef_V"],
        yhat_U=fit_best["yhat_U"],
        yhat_V=fit_best["yhat_V"],
        fit=fit_best,
        SSE_full=float(best["SSE_full"]),
        SSE_tr=float(best["SSE_tr"]),
        SST=float(best["SST"]),
        k_min=float(k_min),
        k_max=float(k_max),
        ks=ks,
        dk=float(dk),
        L=float(L),
        n_grid=int(len(ks)),
        selected_index=int(selected_index),
        n_exact_sse_ties=int(n_exact_ties),
        candidates=candidates,
        legacy_tie_diagnostic=legacy_diag,
        trend_degree=int(trend_degree),
        **meta,
    )
