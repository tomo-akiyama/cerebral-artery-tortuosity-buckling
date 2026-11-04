#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend length selection validation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sin_trend_classify import classify_row, core_periodic_from_row
from sin_trend_measure import default_measurement_config, measure_interval
from sin_trend_paths import (
    BAND_GEOMETRY_2D,
    FRAME_BASELINE,
    PARTIAL_R2_THRESHOLD,
    CORE_COVERAGE_MIN,
    MEASUREMENT_SOURCE_FILES,
    atomic_to_csv,
    atomic_write_json,
    metadata_dict,
    package_versions,
    project_root_from_here,
    sha256_file,
    sha256_json,
    source_code_hashes,
)
from sin_trend_synthetic import (
    make_sinusoid_centerline,
    n_cycles_input_for_target_remaining_after_trim,
    remaining_cycles_true_from_xi,
)


_HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _HERE / "configs" / "length_selection_validation_default.json"
DEFAULT_CLINICAL_CSV_REL = Path(
    "results/analysis/sin_trend/run_154_baseline_v2_r0/baseline/"
    "IXI_HH_sin_trend_analysis_dataset.csv"
)

VALIDATION_SOURCE_FILES = tuple(sorted(set(MEASUREMENT_SOURCE_FILES) | {
    "sin_trend_length_selection_validation.py",
    "40_length_selection_validation.py",
    "sin_trend_synthetic.py",
    "sin_trend_measure.py",
    "sin_trend_classify.py",
    "sin_trend_band.py",
    "sin_trend_paths.py",
    "36_measurement_validation.py",
}))

RESULT_NUMERIC_COLUMNS = (
    "N_true", "N_hat", "n_true_target", "L0_true_trimmed_mm", "L0_true_window_mm",
    "L0_target_mm", "L_base_mm", "L_trimmed_mm", "lambda_hat_mm", "amp_comb_mm",
    "partial_R2", "R2", "coverage_hat", "N_fit", "lambda_min_search_mm",
    "lambda_max_search_mm", "actual_ds_mm", "actual_ds_median_mm",
    "lambda_error_signed_mm", "lambda_abs_rel_error", "lambda_hat_over_true",
    "L_base_over_L0_true_trimmed", "N_hat_over_N_true",
    "N_hat_inflation_from_lambda_underest", "N_hat_inflation_from_L_base_overest",
    "length_error_mm", "length_abs_rel_error", "length_tolerance_mm",
    "phase_at_trim_start_rad",
    "A_hat_mm", "A_hat_over_true", "A_error_signed_mm", "A_abs_rel_error",
)
RESULT_FLAG_COLUMNS = (
    "accepted_core_all", "core_all", "core_periodic", "core_inplane", "fit_valid",
    "accepted_N_true_lt_1p5", "at_search_low_boundary", "at_search_high_boundary",
)


def empty_result_row() -> Dict[str, Any]:
    """Stable schema even when generation fails before measurement starts."""
    return {
        **{k: float("nan") for k in RESULT_NUMERIC_COLUMNS},
        **{k: 0 for k in RESULT_FLAG_COLUMNS},
        "generation_status": "pending", "generation_failure_reason": "",
        "execution_status": "not_run", "execution_failure_reason": "",
        "fit_status": "not_run", "fit_status_reason": "", "reject_reasons": "",
        "lambda_true_in_search_band": "unknown",
    }


def ensure_result_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for key, default in empty_result_row().items():
        if key not in out:
            out[key] = default
    return out


# ---------------------------------------------------------------------------
# config / paths
# ---------------------------------------------------------------------------
def length_selection_validation_root(
    run_id: str, project_root: Optional[Path] = None
) -> Path:
    root = Path(project_root) if project_root is not None else project_root_from_here(__file__)
    return root / "results" / "analysis" / "sin_trend" / "length_selection_validation" / run_id


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(p.resolve())
    cfg["_config_sha256"] = sha256_file(p)
    return cfg


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_smoke_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ov = cfg.get("smoke_overrides") or {}
    out = copy.deepcopy(cfg)
    for key in ("suite_A", "suite_B"):
        if key in ov:
            out[key] = deep_update(out.get(key, {}), ov[key])
    return out


def measurement_config_from_validation_cfg(vcfg: Dict[str, Any]) -> Dict[str, Any]:
    if vcfg.get("frame_reference", FRAME_BASELINE) != FRAME_BASELINE:
        raise ValueError("This validation requires frame_reference=baseline")
    if vcfg.get("acceptance_flag", "core_all") not in ("core_all", "core_periodic"):
        raise ValueError("Only periodic-fit acceptance is supported (no surface selection)")
    if "n_true_match_atol" in vcfg.get("suite_B", {}):
        raise ValueError("Replace suite_B.n_true_match_atol with length_atol_mm and length_rtol")
    cfg = default_measurement_config()
    cfg["verbose"] = False
    cfg["enable_plots"] = False
    cfg["voxel_size"] = float(vcfg.get("voxel_size_mm", 0.5))
    cfg["removal"] = int(vcfg.get("removal_points", 10))
    cfg["N_cyc_search"] = float(vcfg.get("N_cyc_search", 0.5))
    cfg["q_fine"] = int(vcfg.get("q_fine", 8))
    cfg["search_band_mode"] = str(vcfg.get("search_band_mode", BAND_GEOMETRY_2D))
                             
    if abs(float(cfg["N_cyc_search"]) - 0.5) > 1e-12:
        raise ValueError(
            f"N_cyc_search must remain 0.5 for this validation; got {cfg['N_cyc_search']!r}"
        )
    return cfg


def background_dict(bg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not bg or bg.get("type") in (None, "", "null", "straight"):
        return None
    return {"type": str(bg["type"]), "curvature_mm": float(bg.get("curvature_mm", 0.0))}


def stable_case_id(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def rng_for(*parts: Any) -> np.random.Generator:
    material = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()[:8]
    seed = int.from_bytes(digest, "little")
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# truth / window helpers
# ---------------------------------------------------------------------------
def L0_true_trimmed_mm(xi_mm: np.ndarray, removal: int) -> float:
    xi = np.asarray(xi_mm, float)
    rem = int(removal)
    if len(xi) <= 2 * rem:
        return float("nan")
    return float(xi[-rem - 1] - xi[rem])


def N_true_from_xi(xi_mm: np.ndarray, lambda_true_mm: float, removal: int) -> float:
    return remaining_cycles_true_from_xi(
        np.asarray(xi_mm, float), float(lambda_true_mm), removal_points=int(removal)
    )


def make_parent_centerline(
    *,
    amplitude_mm: float,
    lambda_true_mm: float,
    phase_rad: float,
    diameter_mm: float,
    target_ds_mm: float,
    removal: int,
    max_n_true: float,
    parent_extra_cycles: float,
    background: Optional[Dict[str, Any]],
    noise_sd_mm: float,
    seed: int,
    voxel_size_mm: float,
    length_atol_mm: float = 0.05,
) -> Dict[str, Any]:
    target_remaining = float(max_n_true) + float(parent_extra_cycles)
    if not (np.isfinite(length_atol_mm) and float(length_atol_mm) > 0):
        raise ValueError(f"length_atol_mm must be positive finite; got {length_atol_mm!r}")
    atol_cycles = float(length_atol_mm) / float(lambda_true_mm)
    n_in = n_cycles_input_for_target_remaining_after_trim(
        target_remaining=target_remaining,
        amplitude_mm=float(amplitude_mm),
        lambda_x_mm=float(lambda_true_mm),
        phase=float(phase_rad),
        target_ds_mm=float(target_ds_mm),
        removal_points=int(removal),
        voxel_size_mm=float(voxel_size_mm),
        background=background,
        diameter_mm=float(diameter_mm),
        atol=float(atol_cycles),
    )
    syn = make_sinusoid_centerline(
        amplitude_mm=float(amplitude_mm),
        lambda_x_mm=float(lambda_true_mm),
        n_cycles=float(n_in),
        phase=float(phase_rad),
        voxel_size_mm=float(voxel_size_mm),
        target_ds_mm=float(target_ds_mm),
        background=background,
        noise_sd_mm=float(noise_sd_mm),
        seed=int(seed),
        diameter_mm=float(diameter_mm),
    )
    syn["n_cycles_target_parent_after_trim"] = target_remaining
    syn["n_cycles_remaining_parent_true"] = N_true_from_xi(
        syn["xi_mm"], syn["lambda_baseline_true_mm"], removal
    )
    return syn


def cut_prefix_window_for_n_true(
    parent: Dict[str, Any],
    *,
    n_true_target: float,
    removal: int,
    start_pad_cycles: float = 0.05,
    atol: float = 0.02,
) -> Dict[str, Any]:
    xi = np.asarray(parent["xi_mm"], float)
    lam = float(parent["lambda_baseline_true_mm"])
    rem = int(removal)
    n = len(xi)
    if n <= 2 * rem + 2 or not (np.isfinite(lam) and lam > 0):
        return {"ok": False, "reason": "parent_too_short_or_invalid_lambda"}

                                    
    xi0 = float(xi[0]) + float(start_pad_cycles) * lam
    i0 = int(np.searchsorted(xi, xi0, side="left"))
    i0 = max(0, min(i0, n - (2 * rem + 3)))

    target_dxi = float(n_true_target) * lam
                                                      
    lo = i0 + 2 * rem + 1
    hi = n - 1
    if lo > hi:
        return {"ok": False, "reason": "insufficient_points_for_window"}

    best_i1 = None
    best_err = float("inf")
    best_n = float("nan")
                             
    for i1 in range(lo, hi + 1):
        n_true = (float(xi[i1 - rem]) - float(xi[i0 + rem])) / lam
        err = abs(n_true - float(n_true_target))
        if err < best_err:
            best_err = err
            best_i1 = i1
            best_n = n_true

    if best_i1 is None:
        return {"ok": False, "reason": "window_search_failed"}

    sl = slice(i0, best_i1 + 1)
    xi_w = xi[sl]
    n_ach = N_true_from_xi(xi_w, lam, rem)
    L0_trim = L0_true_trimmed_mm(xi_w, rem)
    ok = bool(np.isfinite(n_ach) and abs(n_ach - float(n_true_target)) <= float(atol))
    out = {
        "ok": ok,
        "reason": "" if ok else (
            f"n_true_target_unmet:target={n_true_target:.6g},achieved={n_ach:.6g},"
            f"|err|={abs(n_ach - float(n_true_target)):.3e},atol={atol:g}"
        ),
        "i0": int(i0),
        "i1": int(best_i1),
        "P_vox": np.asarray(parent["P_vox"], float)[sl],
        "xi_mm": xi_w,
        "wave_phase_rad": np.asarray(parent["wave_phase_rad"], float)[sl],
        "radius_vox": (
            None
            if parent.get("radius_vox") is None
            else np.asarray(parent["radius_vox"], float)[sl]
        ),
        "n_true_target": float(n_true_target),
        "n_true_achieved": float(n_ach),
        "L0_true_trimmed_mm": float(L0_trim),
        "L0_true_window_mm": float(xi_w[-1] - xi_w[0]) if len(xi_w) else float("nan"),
        "phase_at_trim_start_rad": float(parent["wave_phase_rad"][i0 + rem])
        if (i0 + rem) < len(parent["wave_phase_rad"])
        else float("nan"),
        "target_dxi_mm": float(target_dxi),
        "best_abs_err_cycles": float(best_err),
    }
    return out


# ---------------------------------------------------------------------------
# measure + classify (production path)
# ---------------------------------------------------------------------------
def measure_and_classify_centerline(
    P_vox: np.ndarray,
    *,
    cfg: Dict[str, Any],
    radius_vox: Optional[np.ndarray] = None,
    ids: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    res = measure_interval(
        np.asarray(P_vox, float),
        frame_reference=FRAME_BASELINE,
        cfg=cfg,
        observed_tangent=None,
        radius_vox=radius_vox,
        ids=ids or {"directory_id": "SYNTHETIC_LS", "segment_id": 0, "segment_index": 0},
        verbose=False,
        compute_pca=False,
    )
    row = classify_row(res["row"])
                                       
                                                               
    meta = {
        "fit_status": row.get("fit_status"),
        "fit_status_reason": row.get("fit_status_reason"),
        "lambda_hat_mm": row.get("lambda_hat_mm"),
        "amp_comb_mm": row.get("amp_comb_mm"),
        "partial_R2": row.get("partial_R2"),
        "R2": row.get("R2"),
        "coverage_hat": row.get("coverage_hat"),
        "N_fit": row.get("N_fit"),
        "L_base_mm": row.get("L_base_mm"),
        "L_trimmed_mm": row.get("L_trimmed_mm"),
        "L_full_mm": row.get("L_full_mm"),
        "lambda_min_search_mm": row.get("lambda_min_search_mm"),
        "lambda_max_search_mm": row.get("lambda_max_search_mm"),
        "boundary_exact_low": row.get("boundary_exact_low"),
        "boundary_exact_high": row.get("boundary_exact_high"),
        "boundary_hit_low": row.get("boundary_hit_low"),
        "boundary_hit_high": row.get("boundary_hit_high"),
        "fit_valid": row.get("fit_valid"),
        "interval_eligible": row.get("interval_eligible"),
        "partial_r2_ok": row.get("partial_r2_ok"),
        "coverage_ok_flag": row.get("coverage_ok_flag"),
        "boundary_excluded": row.get("boundary_excluded"),
        "core_periodic": row.get("core_periodic"),
        "core_all": row.get("core_all"),
        "core_inplane": row.get("core_inplane"),
        "reject_reasons": row.get("reject_reasons"),
        "periodicity": row.get("periodicity"),
        "surface_constraint": row.get("surface_constraint"),
        "diameter_ref_mm": row.get("diameter_ref_mm"),
        "pts": row.get("pts"),
        "pts_trimmed": row.get("pts_trimmed"),
        "actual_ds_median_mm": row.get("ds_mm"),
    }
                               
    meta["core_periodic_recheck"] = int(core_periodic_from_row(row))
    del res
    return meta


def _finite_pos(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v) and v > 0)


def lambda_true_in_search_band(
    lambda_true_mm: float, lam_min: Any, lam_max: Any
) -> str:
    if not _finite_pos(lambda_true_mm):
        return "unknown"
    if not _finite_pos(lam_min) or not _finite_pos(lam_max):
        return "band_unknown"
    if float(lambda_true_mm) < float(lam_min) - 1e-9:
        return "below_min"
    if float(lambda_true_mm) > float(lam_max) + 1e-9:
        return "above_max"
    return "inside"


def enrichment_from_measure(
    *,
    lambda_true_mm: float,
    L0_true_trimmed_mm: float,
    N_true: float,
    meas: Dict[str, Any],
    A_true_mm: Optional[float] = None,
) -> Dict[str, Any]:
    lam_hat = meas.get("lambda_hat_mm")
    L_base = meas.get("L_base_mm")
    N_hat = meas.get("coverage_hat")
    A_hat = meas.get("amp_comb_mm")
    if not _finite_pos(N_hat) and _finite_pos(L_base) and _finite_pos(lam_hat):
        N_hat = float(L_base) / float(lam_hat)

    err_signed = (
        float(lam_hat) - float(lambda_true_mm)
        if _finite_pos(lam_hat) and _finite_pos(lambda_true_mm)
        else float("nan")
    )
    err_abs_rel = (
        abs(err_signed) / float(lambda_true_mm)
        if np.isfinite(err_signed) and _finite_pos(lambda_true_mm)
        else float("nan")
    )
    ratio = (
        float(lam_hat) / float(lambda_true_mm)
        if _finite_pos(lam_hat) and _finite_pos(lambda_true_mm)
        else float("nan")
    )
    L_base_over_L0 = (
        float(L_base) / float(L0_true_trimmed_mm)
        if _finite_pos(L_base) and _finite_pos(L0_true_trimmed_mm)
        else float("nan")
    )
                                   
    # N_hat / N_true ≈ (L_base/L0_true) / (λ_hat/λ_true)
    n_ratio = (
        float(N_hat) / float(N_true)
        if _finite_pos(N_hat) and _finite_pos(N_true)
        else float("nan")
    )
    contrib_lambda_underest = (
        float(lambda_true_mm) / float(lam_hat)
        if _finite_pos(lam_hat) and _finite_pos(lambda_true_mm)
        else float("nan")
    )
    A_ratio = float("nan")
    A_err = float("nan")
    A_abs_rel = float("nan")
    if A_true_mm is not None and _finite_pos(A_true_mm) and _finite_pos(A_hat):
        A_ratio = float(A_hat) / float(A_true_mm)
        A_err = float(A_hat) - float(A_true_mm)
        A_abs_rel = abs(A_err) / float(A_true_mm)
    accepted = int(meas.get("core_all", 0) or 0) == 1
    return {
        "N_hat": float(N_hat) if _finite_pos(N_hat) else float("nan"),
        "lambda_error_signed_mm": err_signed,
        "lambda_abs_rel_error": err_abs_rel,
        "lambda_hat_over_true": ratio,
        "A_hat_mm": float(A_hat) if _finite_pos(A_hat) else float("nan"),
        "A_hat_over_true": A_ratio,
        "A_error_signed_mm": A_err,
        "A_abs_rel_error": A_abs_rel,
        "L_base_over_L0_true_trimmed": L_base_over_L0,
        "N_hat_over_N_true": n_ratio,
        "N_hat_inflation_from_lambda_underest": contrib_lambda_underest,
        "N_hat_inflation_from_L_base_overest": L_base_over_L0,
        "lambda_true_in_search_band": lambda_true_in_search_band(
            float(lambda_true_mm),
            meas.get("lambda_min_search_mm"),
            meas.get("lambda_max_search_mm"),
        ),
        "accepted_core_all": int(accepted),
                                                  
        "accepted_N_true_lt_1p5": int(
            accepted and np.isfinite(N_true) and float(N_true) < 1.5 - 1e-9
        ),
        "at_search_low_boundary": int(bool(meas.get("boundary_exact_low"))),
        "at_search_high_boundary": int(bool(meas.get("boundary_exact_high"))),
    }


def ensure_amplitude_ratio_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    A_true = pd.to_numeric(out.get("A_true_mm", pd.Series(np.nan, index=out.index)), errors="coerce")
    A_hat = pd.to_numeric(out.get("amp_comb_mm", pd.Series(np.nan, index=out.index)), errors="coerce")
    if "A_hat_mm" not in out.columns:
        out["A_hat_mm"] = A_hat
    else:
        existing = pd.to_numeric(out["A_hat_mm"], errors="coerce")
        out["A_hat_mm"] = existing.where(np.isfinite(existing), A_hat)
    ratio = A_hat / A_true
    ratio = ratio.where(np.isfinite(A_true) & (A_true > 0) & np.isfinite(A_hat) & (A_hat > 0))
    out["A_hat_over_true"] = ratio
    out["A_error_signed_mm"] = A_hat - A_true
    out["A_abs_rel_error"] = (A_hat - A_true).abs() / A_true
    return out


# ---------------------------------------------------------------------------
# (3) Accuracy across cycle counts (config key: suite_A)
# ---------------------------------------------------------------------------
def iter_suite_A_jobs(vcfg: Dict[str, Any], *, seed: int) -> List[Dict[str, Any]]:
    a = vcfg["suite_A"]
    jobs: List[Dict[str, Any]] = []
    max_n = float(max(a["n_true_targets"]))
    for D in a["diameter_mm"]:
        for lam in a["lambda_true_mm"]:
            for A in a["amplitude_mm"]:
                for ph in a["phase_rad"]:
                    for bg in a["backgrounds"]:
                        parent_id = stable_case_id(
                            "A_parent", seed, D, lam, A, ph, bg.get("name"), a.get("noise_sd_mm", 0.0)
                        )
                        for n_t in a["n_true_targets"]:
                            case_id = stable_case_id(
                                "A", seed, parent_id, n_t, D, lam, A, ph, bg.get("name")
                            )
                            jobs.append(
                                {
                                    "suite": "A",
                                    "case_id": case_id,
                                    "parent_curve_id": parent_id,
                                    "scenario": f"window_Ntrue_{n_t:g}",
                                    "diameter_mm": float(D),
                                    "lambda_true_mm": float(lam),
                                    "amplitude_mm": float(A),
                                    "phase_rad": float(ph),
                                    "n_true_target": float(n_t),
                                    "background": bg,
                                    "noise_sd_mm": float(a.get("noise_sd_mm", 0.0)),
                                    "max_n_true": max_n,
                                    "parent_extra_cycles": float(a.get("parent_extra_cycles", 0.75)),
                                    "window_start_pad_cycles": float(
                                        a.get("window_start_pad_cycles", 0.05)
                                    ),
                                    "n_true_match_atol": float(a.get("n_true_match_atol", 0.02)),
                                    "parent_length_atol_mm": float(
                                        a.get("parent_length_atol_mm", 0.05)
                                    ),
                                    "seed": int(seed),
                                }
                            )
    return jobs


def _suite_A_base_row(job: Dict[str, Any], *, target_ds_mm: float) -> Dict[str, Any]:
    return {
        **empty_result_row(),
        "case_id": job["case_id"],
        "parent_curve_id": job["parent_curve_id"],
        "suite": "A",
        "scenario": job["scenario"],
        "replicate": 0,
        "seed": job["seed"],
        "D_mm": job["diameter_mm"],
        "A_true_mm": job["amplitude_mm"],
        "lambda_true_mm": job["lambda_true_mm"],
        "phase_rad": job["phase_rad"],
        "background_name": job["background"].get("name"),
        "background_curvature_mm": float(job["background"].get("curvature_mm", 0.0) or 0.0),
        "noise_sd_mm": job["noise_sd_mm"],
        "n_true_target": job["n_true_target"],
        "target_ds_mm": target_ds_mm,
    }


def run_suite_A_job(job: Dict[str, Any], meas_cfg: Dict[str, Any], vcfg: Dict[str, Any]) -> Dict[str, Any]:
    parent_cache: Dict[str, Dict[str, Any]] = {}
    return run_suite_A_job_with_parent_cache(job, meas_cfg, vcfg, parent_cache)


def run_suite_A_job_with_parent_cache(
    job: Dict[str, Any],
    meas_cfg: Dict[str, Any],
    vcfg: Dict[str, Any],
    parent_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    removal = int(vcfg.get("removal_points", 10))
    ds = float(vcfg.get("target_ds_mm", 0.1))
    vs = float(vcfg.get("voxel_size_mm", 0.5))
    bg = background_dict(job["background"])
    base = _suite_A_base_row(job, target_ds_mm=ds)
    try:
        pid = str(job["parent_curve_id"])
        if pid not in parent_cache:
            parent_cache[pid] = make_parent_centerline(
                amplitude_mm=job["amplitude_mm"],
                lambda_true_mm=job["lambda_true_mm"],
                phase_rad=job["phase_rad"],
                diameter_mm=job["diameter_mm"],
                target_ds_mm=ds,
                removal=removal,
                max_n_true=job["max_n_true"],
                parent_extra_cycles=job["parent_extra_cycles"],
                background=bg,
                noise_sd_mm=job["noise_sd_mm"],
                seed=int(rng_for(job["seed"], job["parent_curve_id"]).integers(0, 2**31 - 1)),
                voxel_size_mm=vs,
                length_atol_mm=float(job.get("parent_length_atol_mm", 0.05)),
            )
        parent = parent_cache[pid]
        win = cut_prefix_window_for_n_true(
            parent,
            n_true_target=job["n_true_target"],
            removal=removal,
            start_pad_cycles=job["window_start_pad_cycles"],
            atol=job["n_true_match_atol"],
        )
        if not win["ok"]:
            base.update(
                {
                    "generation_status": "failed",
                    "generation_failure_reason": win.get("reason", "window_failed"),
                    "N_true": win.get("n_true_achieved", float("nan")),
                    "L0_true_trimmed_mm": win.get("L0_true_trimmed_mm", float("nan")),
                    "actual_ds_mm": parent.get("actual_ds_mm", float("nan")),
                    "accepted_core_all": 0,
                }
            )
            return base
        N_true = float(win["n_true_achieved"])
        L0_trim = float(win["L0_true_trimmed_mm"])
        base.update(
            {
                "generation_status": "ok",
                "N_true": N_true,
                "L0_true_trimmed_mm": L0_trim,
                "L0_true_window_mm": win["L0_true_window_mm"],
                "actual_ds_mm": parent.get("actual_ds_mm", float("nan")),
                "n_cycles_input_parent": parent.get("n_cycles_input"),
                "phase_at_trim_start_rad": win.get("phase_at_trim_start_rad"),
            }
        )
        meas = measure_and_classify_centerline(
            win["P_vox"],
            cfg=meas_cfg,
            radius_vox=win.get("radius_vox"),
            ids={
                "directory_id": f"LS_A_{job['case_id']}",
                "segment_id": 0,
                "segment_index": 0,
            },
        )
        base.update(meas)
        base["execution_status"] = "ok"
        base.update(
            enrichment_from_measure(
                lambda_true_mm=job["lambda_true_mm"],
                L0_true_trimmed_mm=L0_trim,
                N_true=N_true,
                meas=meas,
                A_true_mm=float(job["amplitude_mm"]),
            )
        )
        del win
        return base
    except Exception as exc:  # noqa: BLE001
        if base["generation_status"] != "ok":
            base["generation_status"] = "failed"
            base["generation_failure_reason"] = f"{type(exc).__name__}: {exc}"
        else:
            base["execution_status"] = "exception"
            base["execution_failure_reason"] = f"{type(exc).__name__}: {exc}"
        base["accepted_core_all"] = 0
        return base


def run_suite_A_jobs_batched(
    jobs: List[Dict[str, Any]],
    meas_cfg: Dict[str, Any],
    vcfg: Dict[str, Any],
    *,
    done_ids: Optional[set] = None,
    quiet: bool = False,
    on_result=None,
) -> List[Dict[str, Any]]:
    done_ids = done_ids or set()
    pending = [j for j in jobs if j["case_id"] not in done_ids]
    if not pending:
        return []
                                              
    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for j in pending:
        by_parent.setdefault(str(j["parent_curve_id"]), []).append(j)
    for pid in by_parent:
        by_parent[pid].sort(key=lambda x: (float(x["n_true_target"]), str(x["case_id"])))

    rows: List[Dict[str, Any]] = []
    parent_ids = sorted(by_parent.keys())
    for pi, pid in enumerate(parent_ids, 1):
        cache: Dict[str, Dict[str, Any]] = {}
        group = by_parent[pid]
        for j in group:
            row = run_suite_A_job_with_parent_cache(j, meas_cfg, vcfg, cache)
            if on_result is not None:
                on_result(row)
            rows.append(row)
        cache.clear()
        if not quiet and (pi % 10 == 0 or pi == len(parent_ids)):
            print(f"  suiteA parents {pi}/{len(parent_ids)} cases_done={len(rows)}")
    return rows


# ---------------------------------------------------------------------------
# (4) Procedure-induced bias in wavelength–diameter scaling (config key: suite_B)
# ---------------------------------------------------------------------------
@dataclass
class ClinicalPool:
    D_mm: np.ndarray
    L0_proxy_mm: np.ndarray
    source_csv: str
    source_sha256: str
    n_total: int
    n_usable: int
    drop_reasons: Dict[str, int]


def load_clinical_DL_pool(
    csv_path: Path,
    *,
    population_mode: str = "pre_selection",
) -> ClinicalPool:
    path = Path(csv_path)
    usecols = [
        "radius_median_mm",
        "L_base_mm",
        "L_trimmed_mm",
        "periodicity",
        "core_all",
        "core_periodic",
        "surface_constraint",
        "fit_status",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    n_total = len(df)
    drops: Dict[str, int] = {}
    r = pd.to_numeric(df["radius_median_mm"], errors="coerce")
    L = pd.to_numeric(df["L_base_mm"], errors="coerce")
    miss_r = ~np.isfinite(r.to_numpy(dtype=float))
    miss_L = ~np.isfinite(L.to_numpy(dtype=float))
    nonpos_r = np.isfinite(r.to_numpy(dtype=float)) & (r.to_numpy(dtype=float) <= 0)
    nonpos_L = np.isfinite(L.to_numpy(dtype=float)) & (L.to_numpy(dtype=float) <= 0)
    drops["radius_missing"] = int(miss_r.sum())
    drops["L_base_missing"] = int(miss_L.sum())
    drops["radius_nonpositive"] = int(nonpos_r.sum())
    drops["L_base_nonpositive"] = int(nonpos_L.sum())
    ok = np.isfinite(r.to_numpy(dtype=float)) & (r.to_numpy(dtype=float) > 0)
    ok &= np.isfinite(L.to_numpy(dtype=float)) & (L.to_numpy(dtype=float) > 0)
                                                
    if population_mode != "pre_selection":
        raise ValueError(
            f"unsupported population_mode {population_mode!r}; "
            "only pre_selection is implemented (do not filter by periodic/surface)"
        )
    D = 2.0 * r.to_numpy(dtype=float)[ok]
    L0 = L.to_numpy(dtype=float)[ok]
    return ClinicalPool(
        D_mm=D,
        L0_proxy_mm=L0,
        source_csv=str(path.resolve()),
        source_sha256=sha256_file(path),
        n_total=n_total,
        n_usable=int(ok.sum()),
        drop_reasons=drops,
    )


def assign_lambda_true(
    D_mm: np.ndarray,
    *,
    scenario: Dict[str, Any],
    rng: np.random.Generator,
    lambda_pool: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    kind = str(scenario["kind"])
    if kind == "proportional":
        k = float(scenario["k_lambda_over_D"])
        sd = float(scenario.get("epsilon_sd", 0.0))
        eps = rng.normal(0.0, sd, size=len(D_mm)) if sd > 0 else np.zeros(len(D_mm))
        lam = k * np.asarray(D_mm, float) * np.exp(eps)
        meta = _relation_meta(D_mm, lam)
        meta.update({"k_lambda_over_D": k, "epsilon_sd": sd})
        return lam, meta
    if kind == "independent_permute":
        if lambda_pool is None:
            raise ValueError("independent_permute requires lambda_pool")
        lam = np.asarray(lambda_pool, float).copy()
        rng.shuffle(lam)
        meta = _relation_meta(D_mm, lam)
        return lam, meta
    raise ValueError(f"unknown lambda scenario kind {kind!r}")


def _relation_meta(D: np.ndarray, lam: np.ndarray) -> Dict[str, float]:
    d = np.asarray(D, float)
    y = np.asarray(lam, float)
    m = np.isfinite(d) & np.isfinite(y) & (d > 0) & (y > 0)
    out = {
        "gen_n": float(m.sum()),
        "gen_slope_loglog": float("nan"),
        "gen_corr_loglog": float("nan"),
        "gen_lambda_over_D_median": float("nan"),
    }
    if int(m.sum()) < 3:
        return out
    ld = np.log(d[m])
    ly = np.log(y[m])
    if float(np.var(ld)) <= 0:
        return out
    slope = float(np.polyfit(ld, ly, 1)[0])
    corr = float(np.corrcoef(ld, ly)[0, 1]) if int(m.sum()) >= 2 else float("nan")
    out["gen_slope_loglog"] = slope
    out["gen_corr_loglog"] = corr
    out["gen_lambda_over_D_median"] = float(np.median(y[m] / d[m]))
    return out


def sample_DL(
    pool: ClinicalPool,
    *,
    n: int,
    pairing: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pool = len(pool.D_mm)
    if n_pool < 1:
        raise ValueError("clinical D/L pool is empty")
    idx = rng.integers(0, n_pool, size=int(n))
    D = pool.D_mm[idx].copy()
    L = pool.L0_proxy_mm[idx].copy()
    if pairing == "paired":
        return D, L, idx
    if pairing == "shuffled_DL":
        L2 = L.copy()
        rng.shuffle(L2)
        return D, L2, idx
    raise ValueError(f"unknown dl_pairing {pairing!r}")


def generate_population_centerline(
    *,
    D_mm: float,
    L0_true_trimmed_target_mm: float,
    lambda_true_mm: float,
    amplitude_mm: float,
    phase_rad: float,
    target_ds_mm: float,
    removal: int,
    voxel_size_mm: float,
    background: Optional[Dict[str, Any]],
    noise_sd_mm: float,
    seed: int,
    length_atol_mm: float = 0.05,
    length_rtol: float = 0.0,
) -> Dict[str, Any]:
    if not (_finite_pos(D_mm) and _finite_pos(L0_true_trimmed_target_mm) and _finite_pos(lambda_true_mm)):
        return {"ok": False, "reason": "nonpositive_D_L_or_lambda"}
    n_target = float(L0_true_trimmed_target_mm) / float(lambda_true_mm)
    if not np.isfinite(length_atol_mm) or not np.isfinite(length_rtol) or length_atol_mm <= 0 or length_rtol < 0:
        raise ValueError("length_atol_mm must be positive and length_rtol nonnegative")
    length_tol = float(length_atol_mm) + float(length_rtol) * float(L0_true_trimmed_target_mm)
    # Convert the same length tolerance to cycles; longer waves get no looser length tolerance.
    atol_cycles = length_tol / float(lambda_true_mm)
    try:
        n_in = n_cycles_input_for_target_remaining_after_trim(
            target_remaining=n_target,
            amplitude_mm=float(amplitude_mm),
            lambda_x_mm=float(lambda_true_mm),
            phase=float(phase_rad),
            target_ds_mm=float(target_ds_mm),
            removal_points=int(removal),
            voxel_size_mm=float(voxel_size_mm),
            background=background,
            diameter_mm=float(D_mm),
            atol=float(atol_cycles),
        )
    except ValueError as exc:
        return {
            "ok": False,
            "reason": f"n_cycles_solve_failed:{exc}",
            "n_true_target": n_target,
        }
    syn = make_sinusoid_centerline(
        amplitude_mm=float(amplitude_mm),
        lambda_x_mm=float(lambda_true_mm),
        n_cycles=float(n_in),
        phase=float(phase_rad),
        voxel_size_mm=float(voxel_size_mm),
        target_ds_mm=float(target_ds_mm),
        background=background,
        noise_sd_mm=float(noise_sd_mm),
        seed=int(seed),
        diameter_mm=float(D_mm),
    )
    n_true = N_true_from_xi(syn["xi_mm"], syn["lambda_baseline_true_mm"], removal)
    L0_trim = L0_true_trimmed_mm(syn["xi_mm"], removal)
    length_error = float(L0_trim - L0_true_trimmed_target_mm)
    ok = bool(np.isfinite(L0_trim) and abs(length_error) <= length_tol)
    return {
        "ok": ok,
        "reason": ""
        if ok
        else (
            f"n_true_unmet:target={n_target:.6g},achieved={n_true:.6g},"
            f"|err|={abs(n_true - n_target):.3e}"
        ),
        "syn": syn,
        "n_true_target": n_target,
        "N_true": float(n_true),
        "L0_true_trimmed_mm": float(L0_trim),
        "n_cycles_input": float(n_in),
        "length_error_mm": length_error,
        "length_abs_rel_error": abs(length_error) / float(L0_true_trimmed_target_mm),
        "length_tolerance_mm": length_tol,
    }


def build_suite_B_replicate_jobs(
    vcfg: Dict[str, Any],
    *,
    seed: int,
    replicate: int,
    n_per_replicate: int,
    pool: ClinicalPool,
) -> List[Dict[str, Any]]:
    b = vcfg["suite_B"]
    jobs: List[Dict[str, Any]] = []
    scenarios = list(b["lambda_scenarios"])
                                                
    prop_names = {
        s["name"] for s in scenarios if str(s["kind"]).startswith("proportional")
    }
    for pairing in b["dl_pairing"]:
        for A in b["amplitude_mm"]:
            for bg in b["backgrounds"]:
                rng_dl = rng_for(seed, "B_DL", replicate, pairing, A, bg.get("name"))
                D, L, src_idx = sample_DL(
                    pool, n=n_per_replicate, pairing=str(pairing), rng=rng_dl
                )
                rng_phase = rng_for(seed, "B_phase", replicate, pairing, A, bg.get("name"))
                phases = rng_phase.uniform(0.0, 2.0 * np.pi, size=n_per_replicate)
                                                     
                lambda_by_name: Dict[str, np.ndarray] = {}
                meta_by_name: Dict[str, Dict[str, float]] = {}
                for sc in scenarios:
                    if sc["name"] in prop_names or str(sc["kind"]) == "proportional":
                        rng_l = rng_for(
                            seed, "B_lam", replicate, pairing, A, bg.get("name"), sc["name"]
                        )
                        lam, meta = assign_lambda_true(D, scenario=sc, rng=rng_l)
                        lambda_by_name[sc["name"]] = lam
                        meta_by_name[sc["name"]] = meta
                for sc in scenarios:
                    if str(sc["kind"]) == "independent_permute":
                        src = sc.get("source_scenario")
                        if src not in lambda_by_name:
                            raise ValueError(
                                f"independent scenario {sc['name']!r} needs source {src!r}"
                            )
                        rng_l = rng_for(
                            seed, "B_lam", replicate, pairing, A, bg.get("name"), sc["name"]
                        )
                        lam, meta = assign_lambda_true(
                            D, scenario=sc, rng=rng_l, lambda_pool=lambda_by_name[src]
                        )
                        lambda_by_name[sc["name"]] = lam
                        meta_by_name[sc["name"]] = meta
                for sc in scenarios:
                    lam_arr = lambda_by_name[sc["name"]]
                    gen_meta = meta_by_name[sc["name"]]
                    for i in range(n_per_replicate):
                        case_id = stable_case_id(
                            "B",
                            seed,
                            replicate,
                            pairing,
                            A,
                            bg.get("name"),
                            sc["name"],
                            i,
                            float(D[i]),
                            float(L[i]),
                            float(lam_arr[i]),
                            float(phases[i]),
                        )
                        jobs.append(
                            {
                                "suite": "B",
                                "case_id": case_id,
                                "parent_curve_id": case_id,        
                                "scenario": sc["name"],
                                "lambda_kind": sc["kind"],
                                "replicate": int(replicate),
                                "seed": int(seed),
                                "unit_index": int(i),
                                "dl_pairing": str(pairing),
                                "diameter_mm": float(D[i]),
                                "L0_proxy_mm": float(L[i]),
                                "lambda_true_mm": float(lam_arr[i]),
                                "amplitude_mm": float(A),
                                "phase_rad": float(phases[i]),
                                "background": bg,
                                "noise_sd_mm": 0.0,
                                "source_pool_index": int(src_idx[i]),
                                "gen_slope_loglog": gen_meta.get("gen_slope_loglog"),
                                "gen_corr_loglog": gen_meta.get("gen_corr_loglog"),
                                "gen_lambda_over_D_median": gen_meta.get(
                                    "gen_lambda_over_D_median"
                                ),
                                "length_atol_mm": float(b.get("length_atol_mm", 0.05)),
                                "length_rtol": float(b.get("length_rtol", 0.0)),
                            }
                        )
    return jobs


def run_suite_B_job(job: Dict[str, Any], meas_cfg: Dict[str, Any], vcfg: Dict[str, Any]) -> Dict[str, Any]:
    removal = int(vcfg.get("removal_points", 10))
    ds = float(vcfg.get("target_ds_mm", 0.1))
    vs = float(vcfg.get("voxel_size_mm", 0.5))
    bg = background_dict(job["background"])
    base = {
        **empty_result_row(),
        "case_id": job["case_id"],
        "parent_curve_id": job["parent_curve_id"],
        "suite": "B",
        "scenario": job["scenario"],
        "lambda_kind": job.get("lambda_kind"),
        "replicate": job["replicate"],
        "seed": job["seed"],
        "unit_index": job["unit_index"],
        "dl_pairing": job["dl_pairing"],
        "D_mm": job["diameter_mm"],
        "A_true_mm": job["amplitude_mm"],
        "lambda_true_mm": job["lambda_true_mm"],
        "phase_rad": job["phase_rad"],
        "background_name": job["background"].get("name"),
        "background_curvature_mm": float(job["background"].get("curvature_mm", 0.0) or 0.0),
        "noise_sd_mm": job["noise_sd_mm"],
        "L0_proxy_from_clinical_L_base_mm": job["L0_proxy_mm"],
        "L0_target_mm": job["L0_proxy_mm"],
        "target_ds_mm": ds,
        "source_pool_index": job["source_pool_index"],
        "gen_slope_loglog": job.get("gen_slope_loglog"),
        "gen_corr_loglog": job.get("gen_corr_loglog"),
        "gen_lambda_over_D_median": job.get("gen_lambda_over_D_median"),
        "n_true_target": (job["L0_proxy_mm"] / job["lambda_true_mm"]
                          if _finite_pos(job["lambda_true_mm"]) else float("nan")),
    }
    try:
        gen = generate_population_centerline(
            D_mm=job["diameter_mm"],
            L0_true_trimmed_target_mm=job["L0_proxy_mm"],
            lambda_true_mm=job["lambda_true_mm"],
            amplitude_mm=job["amplitude_mm"],
            phase_rad=job["phase_rad"],
            target_ds_mm=ds,
            removal=removal,
            voxel_size_mm=vs,
            background=bg,
            noise_sd_mm=job["noise_sd_mm"],
            seed=int(rng_for(job["seed"], job["case_id"], "gen").integers(0, 2**31 - 1)),
            length_atol_mm=job["length_atol_mm"],
            length_rtol=job["length_rtol"],
        )
        if not gen["ok"]:
            base.update(
                {
                    "generation_status": "failed",
                    "generation_failure_reason": gen.get("reason", "gen_failed"),
                    "N_true": gen.get("N_true", float("nan")),
                    "n_true_target": gen.get("n_true_target", base["n_true_target"]),
                    "length_error_mm": gen.get("length_error_mm", float("nan")),
                    "length_abs_rel_error": gen.get("length_abs_rel_error", float("nan")),
                    "length_tolerance_mm": gen.get("length_tolerance_mm", float("nan")),
                    "L0_true_trimmed_mm": gen.get("L0_true_trimmed_mm", float("nan")),
                    "accepted_core_all": 0,
                }
            )
            return base
        syn = gen["syn"]
        N_true = float(gen["N_true"])
        L0_trim = float(gen["L0_true_trimmed_mm"])
        base.update(
            {
                "generation_status": "ok",
                "N_true": N_true,
                "n_true_target": gen["n_true_target"],
                "L0_true_trimmed_mm": L0_trim,
                "actual_ds_mm": syn.get("actual_ds_mm", float("nan")),
                "n_cycles_input": gen["n_cycles_input"],
                "length_error_mm": gen["length_error_mm"],
                "length_abs_rel_error": gen["length_abs_rel_error"],
                "length_tolerance_mm": gen["length_tolerance_mm"],
            }
        )
        meas = measure_and_classify_centerline(
            syn["P_vox"],
            cfg=meas_cfg,
            radius_vox=syn.get("radius_vox"),
            ids={
                "directory_id": f"LS_B_{job['case_id']}",
                "segment_id": 0,
                "segment_index": 0,
            },
        )
        base.update(meas)
        base["execution_status"] = "ok"
        base.update(
            enrichment_from_measure(
                lambda_true_mm=job["lambda_true_mm"],
                L0_true_trimmed_mm=L0_trim,
                N_true=N_true,
                meas=meas,
                A_true_mm=float(job["amplitude_mm"]),
            )
        )
        del syn
        return base
    except Exception as exc:  # noqa: BLE001
        if base["generation_status"] != "ok":
            base["generation_status"] = "failed"
            base["generation_failure_reason"] = f"{type(exc).__name__}: {exc}"
        else:
            base["execution_status"] = "exception"
            base["execution_failure_reason"] = f"{type(exc).__name__}: {exc}"
        base["accepted_core_all"] = 0
        return base


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _mask_finite_pos(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return np.isfinite(x) & (x > 0)


def loglog_ols(D: np.ndarray, lam: np.ndarray) -> Dict[str, Any]:
    d = np.asarray(D, float)
    y = np.asarray(lam, float)
    m = np.isfinite(d) & np.isfinite(y) & (d > 0) & (y > 0)
    n = int(m.sum())
    out = {
        "n": n,
        "slope": float("nan"),
        "intercept": float("nan"),
        "corr": float("nan"),
        "status": "ok",
        "denominator": "finite_positive_D_and_lambda",
    }
    if n < 3:
        out["status"] = "insufficient_n"
        return out
    ld = np.log(d[m])
    ly = np.log(y[m])
    if float(np.var(ld)) <= 1e-18:
        out["status"] = "insufficient_D_variance"
        return out
    slope, intercept = np.polyfit(ld, ly, 1)
    out["slope"] = float(slope)
    out["intercept"] = float(intercept)
    out["corr"] = float(np.corrcoef(ld, ly)[0, 1])
    return out


def median_iqr(x: np.ndarray) -> Tuple[float, float, float]:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return float(med), float(q1), float(q3)


def summarize_relation_block(
    df: pd.DataFrame,
    *,
    D_col: str,
    lam_col: str,
    L_col: str,
    label: str,
    denominator_note: str,
) -> Dict[str, Any]:
    if len(df) == 0:
        return {
            "relation_label": label,
            "denominator_note": denominator_note,
            "n": 0,
            "slope_log_lambda_vs_log_D": float("nan"),
            "slope_status": "empty",
            "lambda_over_D_median": float("nan"),
            "lambda_over_D_q1": float("nan"),
            "lambda_over_D_q3": float("nan"),
            "corr_D_L0": float("nan"),
            "corr_lambda_L0": float("nan"),
        }
    D = pd.to_numeric(df[D_col], errors="coerce").to_numpy(dtype=float)
    lam = pd.to_numeric(df[lam_col], errors="coerce").to_numpy(dtype=float)
    L = pd.to_numeric(df[L_col], errors="coerce").to_numpy(dtype=float)
    ols = loglog_ols(D, lam)
    ratio = lam / D
    med, q1, q3 = median_iqr(ratio)
    def _corr(a, b):
        m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if int(m.sum()) < 3:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    return {
        "relation_label": label,
        "denominator_note": denominator_note,
        "n": ols["n"],
        "slope_log_lambda_vs_log_D": ols["slope"],
        "slope_status": ols["status"],
        "corr_loglog": ols["corr"],
        "lambda_over_D_median": med,
        "lambda_over_D_q1": q1,
        "lambda_over_D_q3": q3,
        "corr_D_L0": _corr(D, L),
        "corr_lambda_L0": _corr(lam, L),
    }


def aggregate_suite_A(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    df = ensure_amplitude_ratio_columns(ensure_result_schema(df))
    rows = []
    keys = [
        "lambda_true_mm",
        "A_true_mm",
        "background_name",
        "n_true_target",
        "phase_rad",
    ]
    for key_vals, g in df.groupby(keys, dropna=False):
        n_all = len(g)
        gen_ok = g["generation_status"].astype(str).eq("ok")
        exec_ok = g["execution_status"].astype(str).eq("ok")
        measured = g[gen_ok & exec_ok]
        n_meas = len(measured)
        n_true = pd.to_numeric(g["N_true"], errors="coerce")
        lt_15 = n_true < 1.5 - 1e-9
        accepted = pd.to_numeric(g.get("accepted_core_all"), errors="coerce").fillna(0).astype(int) == 1
              
        n_lt15_all = int(lt_15.sum())
        n_lt15_meas = int((lt_15 & gen_ok & exec_ok).sum())
        n_accepted_lt15 = int((lt_15 & accepted).sum())
        accepted_lt15_rate_all = (n_accepted_lt15 / n_lt15_all) if n_lt15_all else float("nan")
        accepted_lt15_rate_meas = (n_accepted_lt15 / n_lt15_meas) if n_lt15_meas else float("nan")
        acc_rate = float(accepted.mean()) if n_all else float("nan")
        acc_rate_meas = (
            float(accepted[gen_ok & exec_ok].mean()) if n_meas else float("nan")
        )
        lam_ratio = pd.to_numeric(measured.get("lambda_hat_over_true"), errors="coerce")
        err = pd.to_numeric(measured.get("lambda_abs_rel_error"), errors="coerce")
                                                  
        if "A_hat_over_true" in measured.columns:
            A_ratio = pd.to_numeric(measured["A_hat_over_true"], errors="coerce")
        else:
            A_hat = pd.to_numeric(measured.get("amp_comb_mm"), errors="coerce")
            A_true = pd.to_numeric(measured.get("A_true_mm"), errors="coerce")
            A_ratio = A_hat / A_true
        if n_meas:
            acc_in_meas = (
                pd.to_numeric(measured.get("accepted_core_all"), errors="coerce").fillna(0).astype(int)
                == 1
            )
            accepted_meas = measured.loc[acc_in_meas]
        else:
            accepted_meas = measured.iloc[0:0]
        if len(accepted_meas):
            if "A_hat_over_true" in accepted_meas.columns:
                A_ratio_acc = pd.to_numeric(accepted_meas["A_hat_over_true"], errors="coerce")
            else:
                A_ratio_acc = (
                    pd.to_numeric(accepted_meas.get("amp_comb_mm"), errors="coerce")
                    / pd.to_numeric(accepted_meas.get("A_true_mm"), errors="coerce")
                )
        else:
            A_ratio_acc = pd.Series(dtype=float)
        rows.append(
            {
                **{k: v for k, v in zip(keys, key_vals)},
                "n_all_inputs": n_all,
                "n_generation_failed": int((~gen_ok).sum()),
                "n_execution_failed": int((gen_ok & ~exec_ok).sum()),
                "n_N_true_unknown": int(n_true.isna().sum()),
                "n_target_lt_1p5_inputs": int((pd.to_numeric(g["n_true_target"], errors="coerce") < 1.5 - 1e-9).sum()),
                "n_generation_ok_and_measured": n_meas,
                "acceptance_rate_all_inputs": acc_rate,
                "acceptance_rate_measured": acc_rate_meas,
                "n_N_true_lt_1p5_all_inputs": n_lt15_all,
                "n_N_true_lt_1p5_measured": n_lt15_meas,
                "n_accepted_N_true_lt_1p5": n_accepted_lt15,
                "accepted_N_true_lt_1p5_rate_denom_all_inputs": accepted_lt15_rate_all,
                "accepted_N_true_lt_1p5_rate_denom_measured": accepted_lt15_rate_meas,
                "lambda_hat_over_true_median_measured": float(lam_ratio.median())
                if n_meas
                else float("nan"),
                "lambda_abs_rel_error_median_measured": float(err.median())
                if n_meas
                else float("nan"),
                "A_hat_over_true_median_measured": float(A_ratio.median())
                if n_meas and np.isfinite(A_ratio).any()
                else float("nan"),
                "A_hat_over_true_p10_measured": float(A_ratio.quantile(0.1))
                if n_meas and np.isfinite(A_ratio).any()
                else float("nan"),
                "A_hat_over_true_p90_measured": float(A_ratio.quantile(0.9))
                if n_meas and np.isfinite(A_ratio).any()
                else float("nan"),
                "A_hat_over_true_median_accepted": float(A_ratio_acc.median())
                if len(A_ratio_acc) and np.isfinite(A_ratio_acc).any()
                else float("nan"),
                "L_base_over_L0_median_measured": float(
                    pd.to_numeric(measured.get("L_base_over_L0_true_trimmed"), errors="coerce").median()
                )
                if n_meas
                else float("nan"),
                "N_hat_over_N_true_median_measured": float(
                    pd.to_numeric(measured.get("N_hat_over_N_true"), errors="coerce").median()
                )
                if n_meas
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def aggregate_suite_B_replicate(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    df = ensure_result_schema(df)
    rows = []
    group_cols = ["replicate", "scenario", "dl_pairing", "A_true_mm", "background_name"]
    for key_vals, g in df.groupby(group_cols, dropna=False):
        meta = {k: v for k, v in zip(group_cols, key_vals)}
        gen_ok = g["generation_status"].astype(str).eq("ok")
        exec_ok = g["execution_status"].astype(str).eq("ok")
        true_all = g[gen_ok].copy()
        accepted = (
            pd.to_numeric(g.get("accepted_core_all"), errors="coerce").fillna(0).astype(int) == 1
        )
        true_acc = g[gen_ok & accepted].copy()
        fit_valid = (
            pd.to_numeric(g.get("fit_valid"), errors="coerce").fillna(0).astype(int) == 1
        )
        est_pre = g[gen_ok & exec_ok & fit_valid].copy()
        est_acc = g[gen_ok & exec_ok & accepted].copy()

        blocks = [
            summarize_relation_block(
                g, D_col="D_mm", lam_col="lambda_true_mm", L_col="L0_target_mm",
                label="0_all_planned_true_lambda_vs_D",
                denominator_note="all planned inputs, including generation failures; L is the target",
            ),
            summarize_relation_block(
                true_all,
                D_col="D_mm",
                lam_col="lambda_true_mm",
                L_col="L0_true_trimmed_mm",
                label="1_all_generated_true_lambda_vs_D",
                denominator_note="generation_status==ok",
            ),
            summarize_relation_block(
                true_acc,
                D_col="D_mm",
                lam_col="lambda_true_mm",
                L_col="L0_true_trimmed_mm",
                label="2_accepted_true_lambda_vs_D",
                denominator_note="generation_ok AND core_all==1",
            ),
            summarize_relation_block(
                est_pre,
                D_col="D_mm",
                lam_col="lambda_hat_mm",
                L_col="L0_true_trimmed_mm",
                label="3_preaccept_fit_valid_estimated_lambda_vs_D",
                denominator_note="generation_ok AND execution_ok AND fit_valid==1 (not yet core gate)",
            ),
            summarize_relation_block(
                est_acc,
                D_col="D_mm",
                lam_col="lambda_hat_mm",
                L_col="L0_true_trimmed_mm",
                label="4_accepted_estimated_lambda_vs_D",
                denominator_note="generation_ok AND core_all==1",
            ),
        ]
        N_hat = pd.to_numeric(est_acc.get("N_hat"), errors="coerce")
        n_hat_15_2 = int(((N_hat >= 1.5 - 1e-9) & (N_hat <= 2.0 + 1e-9)).sum()) if len(est_acc) else 0
        reject = g.get("reject_reasons")
        reject_counts = (
            reject.fillna("").astype(str).value_counts().head(8).to_dict()
            if reject is not None
            else {}
        )
        for b in blocks:
            row = {
                **meta,
                **b,
                "n_inputs": len(g),
                "n_generation_ok": int(gen_ok.sum()),
                "n_generation_failed": int((~gen_ok).sum()),
                "n_execution_failed": int((gen_ok & ~exec_ok).sum()),
                "max_length_abs_error_mm": float(pd.to_numeric(g["length_error_mm"], errors="coerce").abs().max()),
                "max_length_abs_rel_error": float(pd.to_numeric(g["length_abs_rel_error"], errors="coerce").max()),
                "n_accepted": int(accepted.sum()),
                "acceptance_rate": float(accepted.mean()) if len(g) else float("nan"),
                "N_hat_median_accepted": float(N_hat.median()) if len(est_acc) else float("nan"),
                "frac_N_hat_in_1p5_to_2_accepted": (
                    n_hat_15_2 / len(est_acc) if len(est_acc) else float("nan")
                ),
                "gen_slope_loglog_design": float(
                    pd.to_numeric(g["gen_slope_loglog"], errors="coerce").iloc[0]
                )
                if "gen_slope_loglog" in g.columns and len(g)
                else float("nan"),
                "top_reject_reasons_json": json.dumps(reject_counts, ensure_ascii=False),
                "percentile_range_note": (
                    "Monte Carlo distribution under the configured generation settings; "
                    "not a clinical 95% CI"
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_suite_B_across_replicates(rep_df: pd.DataFrame) -> pd.DataFrame:
    if len(rep_df) == 0:
        return pd.DataFrame()
    keys = ["scenario", "dl_pairing", "A_true_mm", "background_name", "relation_label"]
    rows = []
    for key_vals, g in rep_df.groupby(keys, dropna=False):
        slopes = pd.to_numeric(g["slope_log_lambda_vs_log_D"], errors="coerce")
        finite = slopes[np.isfinite(slopes)]
        rows.append(
            {
                **{k: v for k, v in zip(keys, key_vals)},
                "n_replicates": len(g),
                "n_replicates_with_finite_slope": int(len(finite)),
                "slope_median": float(finite.median()) if len(finite) else float("nan"),
                "slope_p05": float(np.percentile(finite, 5)) if len(finite) else float("nan"),
                "slope_p95": float(np.percentile(finite, 95)) if len(finite) else float("nan"),
                "percentile_range_note": (
                    "Monte Carlo distribution under the configured generation settings; "
                    "not a clinical 95% CI"
                ),
                "acceptance_rate_median": float(
                    pd.to_numeric(g["acceptance_rate"], errors="coerce").median()
                ),
                "denominator_note": str(g["denominator_note"].iloc[0])
                if "denominator_note" in g.columns
                else "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# plots (summary only)
# ---------------------------------------------------------------------------
def save_summary_figures(df: pd.DataFrame, out_dir: Path) -> List[str]:
    """Separate backgrounds/pairings; MC summaries are not clinical confidence intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = ensure_result_schema(df)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name):
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))

    A = df[df["suite"].eq("A")]
    for fi, (bg, g) in enumerate(A.groupby("background_name", dropna=False)):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for (diam, lam, amp), h in g.groupby(["D_mm", "lambda_true_mm", "A_true_mm"]):
            label = f"D={diam:g}, wavelength={lam:g}, A={amp:g} mm"
            axes[0, 0].scatter(h.N_true, h.lambda_hat_over_true, s=16, alpha=.7, label=label)
            axes[0, 1].scatter(h.L0_true_trimmed_mm, h.lambda_hat_mm, s=16, alpha=.7)
            axes[1, 0].scatter(h.N_true, h.N_hat, s=16, alpha=.7)
            # Pool only phases within each fixed geometry/target condition.
            rate = h.groupby("n_true_target").accepted_core_all.mean()
            axes[1, 1].plot(rate.index, rate.values, ".-", label=label)
        axes[0, 0].axhline(1, color=".4", ls="--")
        axes[0, 0].set(xlabel="True retained cycles", ylabel="Estimated / true wavelength")
        axes[0, 1].set(xlabel="True retained baseline length (mm)", ylabel="Estimated wavelength (mm)")
        nh = pd.to_numeric(g.N_hat, errors="coerce")
        nt = pd.to_numeric(g.N_true, errors="coerce")
        hi = max(3., float(nh.max()) if nh.notna().any() else 3.,
                 float(nt.max()) if nt.notna().any() else 3.)
        axes[1, 0].plot([0, hi], [0, hi], "k--", lw=1)
        axes[1, 0].axhline(1.5, color=".5", ls=":")
        axes[1, 0].axvline(1.5, color=".5", ls=":")
        axes[1, 0].set(xlabel="True retained cycles", ylabel="Estimated fitted cycles")
        axes[1, 1].set(xlabel="Target retained cycles", ylabel="Acceptance fraction (all inputs)", ylim=(-.05, 1.05))
        for ax in axes.flat:
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"(3) Accuracy across cycle counts — {bg}", fontsize=12)
        save(fig, f"figA_{fi:02d}_recovery_and_acceptance.png")

    if len(A):
                            
        A_amp = ensure_amplitude_ratio_columns(A)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (lam, Aamp), g in A_amp.groupby(["lambda_true_mm", "A_true_mm"]):
            x = pd.to_numeric(g["N_true"], errors="coerce")
            y = pd.to_numeric(g["A_hat_over_true"], errors="coerce")
            ax.scatter(x, y, s=18, alpha=0.7, label=f"λ={lam:g},A={Aamp:g}")
        ax.axhline(1.0, color="0.3", ls="--", lw=1)
        ax.axvline(1.5, color="0.5", ls=":", lw=1)
        ax.set_xlabel("N_true (post-trim)")
        ax.set_ylabel("A_hat / A_true")
        ax.set_title("(3) Amplitude ratio vs true cycles")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        p = out_dir / "figA_amplitude_ratio_vs_Ntrue.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))

                    
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        acc_flag = pd.to_numeric(A_amp.get("accepted_core_all"), errors="coerce").fillna(0).astype(int) == 1
        for ax, mask, title in (
            (axes[0], ~acc_flag, "not accepted"),
            (axes[1], acc_flag, "accepted (core_all=1)"),
        ):
            sub = A_amp.loc[mask]
            x = pd.to_numeric(sub["N_true"], errors="coerce")
            y = pd.to_numeric(sub["A_hat_over_true"], errors="coerce")
            ax.scatter(x, y, s=16, alpha=0.65, c="C0")
            ax.axhline(1.0, color="0.3", ls="--", lw=1)
            ax.axvline(1.5, color="0.5", ls=":", lw=1)
            ax.set_xlabel("N_true")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("A_hat / A_true")
        fig.suptitle("(3) Amplitude recovery by acceptance", fontsize=11)
        fig.tight_layout()
        p = out_dir / "figA_amplitude_ratio_by_acceptance.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))

    B = df[df["suite"].eq("B")]
    if not len(B):
        return written
    rep = aggregate_suite_B_replicate(B)
    labels = {
        "proportional_sigma0": "Proportional (no scatter)",
        "proportional_sigma0p2": "Proportional (log SD 0.2)",
        "independent_perm_of_proportional_sigma0": "Independent (permuted)",
    }
    blocks = [
        "0_all_planned_true_lambda_vs_D", "1_all_generated_true_lambda_vs_D",
        "2_accepted_true_lambda_vs_D", "3_preaccept_fit_valid_estimated_lambda_vs_D",
        "4_accepted_estimated_lambda_vs_D",
    ]
    ticklabels = ["Planned\ntrue", "Generated\ntrue", "Accepted\ntrue", "Fit-valid\nestimated", "Accepted\nestimated"]
    for fi, ((pairing, amp, bg), g) in enumerate(B.groupby(
            ["dl_pairing", "A_true_mm", "background_name"], dropna=False)):
        title = f"(4) Wavelength–diameter scaling — {pairing}; A={amp:g} mm; {bg}"
        scenarios = list(g.scenario.unique())
        colors = plt.get_cmap("tab10")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for si, sc in enumerate(scenarios):
            h = g[g.scenario.eq(sc)]
            acc = h[h.accepted_core_all.eq(1)]
            for ax, subset, col in [(axes[0], h, "lambda_true_mm"), (axes[1], acc, "lambda_hat_mm")]:
                x = pd.to_numeric(subset.D_mm, errors="coerce")
                y = pd.to_numeric(subset[col], errors="coerce")
                ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                if ok.any():
                    ax.scatter(x[ok], y[ok], s=12, alpha=.45, color=colors(si),
                               label=labels.get(sc, sc))
        for ax, subtitle in zip(axes, ["All planned inputs (true)", "Accepted (estimated)"]):
            if ax.collections:
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.legend(fontsize=7)
            else:
                ax.text(.5, .5, "No finite cases", ha="center", transform=ax.transAxes)
            ax.set(xlabel="Diameter (mm)", ylabel="Wavelength (mm)", title=subtitle)
            ax.grid(alpha=.2)
        fig.suptitle(title)
        save(fig, f"figB_{fi:02d}_wavelength_vs_D.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        rg = rep[rep.dl_pairing.eq(pairing) & rep.A_true_mm.eq(amp) & rep.background_name.eq(bg)]
        for si, sc in enumerate(scenarios):
            h = rg[rg.scenario.eq(sc)]
            for bi, block in enumerate(blocks):
                vals = pd.to_numeric(h.loc[h.relation_label.eq(block), "slope_log_lambda_vs_log_D"],
                                     errors="coerce").dropna().to_numpy()
                vals = vals[np.isfinite(vals)]
                if not len(vals):
                    continue
                x = bi + (si - (len(scenarios) - 1) / 2) * .18
                lo, med, hi = np.percentile(vals, [5, 50, 95])
                ax.plot([x, x], [lo, hi], color=colors(si), lw=2)
                ax.scatter([x], [med], color=colors(si), s=30,
                           label=labels.get(sc, sc) if bi == 0 else None)
        ax.axhline(0, color=".5", ls=":")
        ax.axhline(1, color=".5", ls="--")
        ax.set_xticks(range(len(blocks)), ticklabels)
        ax.set_ylabel("log wavelength–log diameter slope")
        ax.set_title(title + "\nMedian and 5th–95th percentiles across replicates (not clinical CIs)", fontsize=10)
        if ax.collections:
            ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)
        save(fig, f"figB_{fi:02d}_slopes_before_after.png")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for si, sc in enumerate(scenarios):
            h = g[g.scenario.eq(sc)]
            acc = h[h.accepted_core_all.eq(1)]
            label = labels.get(sc, sc)
            for ax, values in [
                (axes[0], pd.to_numeric(acc.lambda_hat_mm, errors="coerce") / pd.to_numeric(acc.D_mm, errors="coerce")),
                (axes[1], pd.to_numeric(acc.N_hat, errors="coerce")),
            ]:
                values = values[np.isfinite(values)]
                if len(values):
                    # ECDFs avoid bin-dependent comparisons and count imbalance.
                    v = np.sort(values.to_numpy())
                    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                            label=label, color=colors(si))
            axes[2].bar(si, h.accepted_core_all.mean(), color=colors(si))
        axes[0].set(xlabel="Accepted estimated wavelength / diameter", ylabel="Cumulative fraction")
        axes[1].axvline(1.5, color=".5", ls=":")
        axes[1].axvline(2., color=".5", ls=":")
        axes[1].set(xlabel="Accepted fitted cycle count", ylabel="Cumulative fraction")
        axes[2].set(ylim=(0, 1), ylabel="Acceptance fraction (all inputs)")
        axes[2].set_xticks(range(len(scenarios)), [labels.get(sc, sc).replace(" (", "\n(") for sc in scenarios],
                          fontsize=7)
        for ax in axes[:2]:
            if ax.lines:
                handles, labs = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labs, fontsize=7)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=.2)
        fig.suptitle(title)
        save(fig, f"figB_{fi:02d}_ratios_cycles_acceptance.png")
    return written


def save_example_waveforms(jobs, df, meas_cfg, vcfg, out_dir):
    """Regenerate a few exact cases for display; never replace measured case results."""
    diag = vcfg.get("diagnostics", {})
    if not diag.get("save_example_waveforms", False):
        return []
    maximum = int(diag.get("max_example_plots", 3))
    if maximum < 0:
        raise ValueError("max_example_plots must be nonnegative")
    by_id = {j["case_id"]: j for j in jobs}
    requested = list(diag.get("example_case_ids") or [])
    if set(requested) - set(by_id):
        raise ValueError("example_case_ids contains a case outside the current plan")
    if not requested:
        # Deterministic representatives of both accepted and rejected generated cases.
        good = df[df.generation_status.eq("ok")].sort_values("case_id")
        picks = [g.case_id.iloc[0] for _, g in good.groupby(["suite", "accepted_core_all"])]
        requested = list(dict.fromkeys(picks + good.case_id.tolist()))
    chosen = requested[:maximum]
    if not chosen:
        return []
    from importlib import import_module
    plot = import_module("36_measurement_validation").plot_original_and_baseline_centerlines
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    ds, vs, removal = float(vcfg["target_ds_mm"]), float(vcfg["voxel_size_mm"]), int(vcfg["removal_points"])
    indexed = df.set_index("case_id")
    for cid in chosen:
        job = by_id[cid]
        if indexed.loc[cid, "generation_status"] != "ok":
            continue
        bg = background_dict(job["background"])
        if job["suite"] == "A":
            parent = make_parent_centerline(
                amplitude_mm=job["amplitude_mm"], lambda_true_mm=job["lambda_true_mm"],
                phase_rad=job["phase_rad"], diameter_mm=job["diameter_mm"],
                target_ds_mm=ds, removal=removal, max_n_true=job["max_n_true"],
                parent_extra_cycles=job["parent_extra_cycles"], background=bg,
                noise_sd_mm=job["noise_sd_mm"],
                seed=int(rng_for(job["seed"], job["parent_curve_id"]).integers(0, 2**31 - 1)),
                voxel_size_mm=vs,
                length_atol_mm=float(job.get("parent_length_atol_mm", 0.05)),
            )
            syn = cut_prefix_window_for_n_true(
                parent, n_true_target=job["n_true_target"], removal=removal,
                start_pad_cycles=job["window_start_pad_cycles"], atol=job["n_true_match_atol"])
            if not syn["ok"]:
                raise RuntimeError(f"Example regeneration failed: {cid}")
        else:
            gen = generate_population_centerline(
                D_mm=job["diameter_mm"], L0_true_trimmed_target_mm=job["L0_proxy_mm"],
                lambda_true_mm=job["lambda_true_mm"], amplitude_mm=job["amplitude_mm"],
                phase_rad=job["phase_rad"], target_ds_mm=ds, removal=removal,
                voxel_size_mm=vs, background=bg, noise_sd_mm=job["noise_sd_mm"],
                seed=int(rng_for(job["seed"], cid, "gen").integers(0, 2**31 - 1)),
                length_atol_mm=job["length_atol_mm"], length_rtol=job["length_rtol"])
            if not gen["ok"]:
                raise RuntimeError(f"Example regeneration failed: {cid}")
            syn = gen["syn"]
        res = measure_interval(syn["P_vox"], frame_reference=FRAME_BASELINE, cfg=meas_cfg,
                               radius_vox=syn.get("radius_vox"), compute_pca=False, verbose=False)
        p0 = (res.get("baseline") or {}).get("P0")
        if p0 is None:
            continue
        row = indexed.loc[cid]
        path = out_dir / f"{cid}.png"
        plot(P_mm=np.asarray(syn["P_vox"]) * vs, P0_mm=np.asarray(p0) * vs,
             out_path=path, title=f"{cid}\ntrue wavelength={job['lambda_true_mm']:.3g} mm; "
             f"estimated={row.lambda_hat_mm:.3g} mm; accepted={int(row.accepted_core_all)}")
        written.append(str(path))
    return written


def write_report(
    path: Path,
    *,
    vcfg: Dict[str, Any],
    summary_A: pd.DataFrame,
    summary_B_rep: pd.DataFrame,
    summary_B_mc: pd.DataFrame,
    pool_info: Optional[Dict[str, Any]],
    n_rows: int,
    elapsed_s: float,
) -> None:
    lines = [
        "# Synthetic accuracy validation — (3)(4)",
        "",
        "- (3) Accuracy across cycle counts",
        "- (4) Procedure-induced bias in wavelength–diameter scaling",
        "",
        "## Scope and limits",
        "",
        "- Validates wavelength estimation and periodic-fit acceptance (`core_all` / `core_periodic`) only.",
        "- Does **not** reproduce brain-surface mesh, normal estimation, coverage, or R_tan selection.",
        "- Therefore this artificial population does **not** reproduce the full clinical path to the final 275 predominantly-tangential intervals.",
        "- Artificial planar curves are **not** labeled Predominantly tangential merely for being planar.",
        "- Search lower cycle count remains `N_cyc_search=0.5` (not changed to 1.5).",
        "- Acceptance still requires estimated cycles `coverage_hat >= 1.5` via production `classify_row`.",
        "",
        f"- Elapsed: {elapsed_s:.1f} s; rows: {n_rows}",
        f"- Config: {vcfg.get('_config_path')}",
        f"- Config sha256: {vcfg.get('_config_sha256')}",
        "",
        "## Length definitions",
        "",
        "- `L0_true_trimmed_mm`: true known-baseline arc length after end trim (from generation ξ).",
        "- `L_base_mm`: estimated quadratic baseline arc length after trim (production measure).",
        "- `L_trimmed_mm`: observed centerline arc length after trim.",
        "- `N_true = Δξ / lambda_true` (not from estimated baseline / lambda_hat).",
        "- `N_hat = L_base_mm / lambda_hat_mm` (production coverage_hat).",
        "",
    ]
    if pool_info:
        lines += [
            "## Clinical D/L pool ((4) inputs)",
            "",
            f"- CSV: `{pool_info.get('source_csv')}`",
            f"- sha256: `{pool_info.get('source_sha256')}`",
            f"- n_total={pool_info.get('n_total')}, n_usable={pool_info.get('n_usable')}",
            f"- drop_reasons: {pool_info.get('drop_reasons')}",
            "- `L_base_mm` is itself an estimate; used only as a proxy distribution for artificial true length.",
            "- Population is pre-selection (not filtered to Core / surface).",
            "",
        ]
    lines += ["## (3) Accuracy across cycle counts — summary (head)", ""]
    if len(summary_A):
        lines.append(summary_A.head(20).to_string(index=False))
    else:
        lines.append("(empty)")
    lines += ["", "## (4) Procedure-induced bias in wavelength–diameter scaling — replicate slopes (head)", ""]
    if len(summary_B_rep):
        cols = [
            c
            for c in [
                "replicate",
                "scenario",
                "dl_pairing",
                "relation_label",
                "n",
                "slope_log_lambda_vs_log_D",
                "slope_status",
                "acceptance_rate",
            ]
            if c in summary_B_rep.columns
        ]
        lines.append(summary_B_rep[cols].head(40).to_string(index=False))
    else:
        lines.append("(empty)")
    lines += ["", "## (4) Procedure-induced bias in wavelength–diameter scaling — Monte Carlo slope percentiles", ""]
    if len(summary_B_mc):
        lines.append(summary_B_mc.head(40).to_string(index=False))
        lines.append("")
        lines.append(
            "Percentile ranges are Monte Carlo distributions under the configured "
            "generation settings, not clinical coefficient 95% CIs."
        )
    else:
        lines.append("(empty)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resume_fingerprint(
    vcfg: Dict[str, Any], *, seed: int, clinical_sha: Optional[str],
    run_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Hash the actual merged config, not only the original JSON file.
    effective = {k: v for k, v in vcfg.items()
                 if not k.startswith("_") and k != "smoke_overrides"}
    return {
        "schema_version": 2,
        "effective_config_sha256": sha256_json(effective),
        "seed": int(seed),
        "clinical_csv_sha256": clinical_sha,
        "run_spec": run_spec or {},
        "code_hashes": source_code_hashes(VALIDATION_SOURCE_FILES, here=_HERE),
        "measurement_cfg_sha256": sha256_json(measurement_config_from_validation_cfg(vcfg)),
        "package_versions": package_versions(),
        "partial_R2_threshold": float(PARTIAL_R2_THRESHOLD),
        "coverage_min": float(CORE_COVERAGE_MIN),
    }


def check_resume_consistency(out_dir: Path, fp: Dict[str, Any]) -> None:
    meta_path = Path(out_dir) / "run_fingerprint.json"
    if not meta_path.is_file():
        raise RuntimeError("Cannot resume without run_fingerprint.json; use a new output directory")
    prev = json.loads(meta_path.read_text(encoding="utf-8"))
    if prev.get("schema_version") != 2:
        raise RuntimeError("Legacy validation run cannot be resumed safely; use a new run_id")
    if prev != fp:
        changed = sorted(k for k in set(prev) | set(fp) if prev.get(k) != fp.get(k))
        raise RuntimeError("resume fingerprint mismatch: " + ", ".join(changed))


def crosscheck_with_36_short_coverage(
    meas_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    from importlib import import_module

    m36 = import_module("36_measurement_validation")
    cond = {
        "amplitude_mm": 1.0,
        "lambda_x_mm": 20.0,
        "phase": 0.0,
        "target_ds_mm": 0.10,
        "diameter_mm": 1.4,
        "variant": "short_coverage",
        "n_cycles_target_after_trim": 1.5,
    }
                      
    from sin_trend_synthetic import n_cycles_input_for_target_remaining_after_trim

    cond["n_cycles"] = n_cycles_input_for_target_remaining_after_trim(
        target_remaining=1.5,
        amplitude_mm=1.0,
        lambda_x_mm=20.0,
        phase=0.0,
        target_ds_mm=0.10,
        removal_points=10,
        diameter_mm=1.4,
    )
    rows36 = m36.run_condition(cond, meas_cfg, verbose=False, plot_path=None)
    r36 = rows36[0]
              
    syn = make_sinusoid_centerline(
        amplitude_mm=1.0,
        lambda_x_mm=20.0,
        n_cycles=float(cond["n_cycles"]),
        phase=0.0,
        target_ds_mm=0.10,
        diameter_mm=1.4,
        voxel_size_mm=float(meas_cfg["voxel_size"]),
    )
    meas = measure_and_classify_centerline(
        syn["P_vox"], cfg=meas_cfg, radius_vox=syn.get("radius_vox")
    )
    keys = ["lambda_hat_mm", "partial_R2", "coverage_hat", "core_all", "L_base_mm"]
    diff = {}
    ok = True
    for k in keys:
        a = r36.get(k)
        b = meas.get(k)
        try:
            fa, fb = float(a), float(b)
            match = bool(np.isfinite(fa) and np.isfinite(fb) and abs(fa - fb) <= 1e-9 * max(1.0, abs(fa)))
            if not match and np.isfinite(fa) and np.isfinite(fb):
                match = abs(fa - fb) < 1e-6
        except (TypeError, ValueError):
            match = str(a) == str(b)
        diff[k] = {"from_36": a, "from_40": b, "match": match}
        ok = ok and match
    return {"ok": ok, "keys": diff, "n_cycles_input": cond["n_cycles"]}
