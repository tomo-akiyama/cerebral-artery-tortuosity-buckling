#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend measure."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
# Prefer this package directory on sys.path (same basename modules must not collide).
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)

from sin_trend_frames import (  # noqa: E402
    BASELINE_METHOD,
    CORRESPONDENCE_METHOD,
    FITTING_COORDINATE,
    FRAME_TRANSPORT_METHOD,
    FRAME_TRANSPORT_METHOD_LEGACY,
    TANGENT_SOURCE_BASELINE,
    TANGENT_SOURCE_FRENET,
    TANGENT_SOURCE_NUMERIC,
    analytic_baseline_tangent,
    arclength,
    lateral_components_signed,
    legacy_projection_frame,
    minimal_rotation_transport,
    numeric_tangent,
    quadratic_baseline,
    tangent_angle_deg,
    trim_bundle,
)
from sin_trend_band import (  # noqa: E402
    boundary_diagnostics,
    build_search_band,
)
from sin_trend_classify import apply_classification, classification_column_names  # noqa: E402
from sin_trend_paths import (  # noqa: E402
    BAND_GEOMETRY_2D,
    BAND_LEGACY_FIXED_10MM,
    BOUNDARY_POLICY_QC_ONLY,
    CLASSIFICATION_REVISION,
    DIAMETER_WAVELENGTH_FACTOR,
    FRAME_BASELINE,
    FRAME_OBSERVED,
    FRAME_OBSERVED_LEGACY,
    FRAME_REFERENCES,
    MEASUREMENT_REVISION,
    MEASUREMENT_VERSION,
    OUTPUT_TAG,
    SEARCH_BAND_MODES,
    compute_patch_radius_vox,
    validate_frame_reference,
)
from sin_trend_forward_fit import (  # noqa: E402
    FIT_METHOD,
    K_SELECTION_METHOD,
    R2_DEFINITION,
    search_k_forward_3d,
)
from sin_trend_wavelength_coordinates import cumulative_baseline_arclength  # noqa: E402

DIAMETER_DEFINITION = (
    "2*radius_median_mm (observed lumen radius median over trimmed interval; "
    "no wall thickness; not baseline-derived). "
    "Full-interval medians are in radius_*_full_mm columns."
)
PHASE_COORDINATE = "baseline_arclength"
TREND_COORDINATE = "baseline_arclength"
QUADRATIC_DETREND_COORDINATE = "observed_arclength"
GRID_LENGTH_BASIS = "trimmed_baseline_arclength"
RADIUS_BASIS = "trimmed_observed_lumen"
SLENDERNESS_DEFINITION = "lambda_hat_mm / (2 * radius_median_mm)"
ELIGIBILITY_BASIS_FULL = "L_full_pre_trim_finite_positive"
ELIGIBILITY_BASIS_LEGACY = "L_trimmed_ge_legacy_min_trimmed_length_mm"

                                          
RESIDUAL_VS_INPUT_SCALE_FACTOR = 10.0

                                      
DEPRECATED_MEASUREMENT_CONFIG_KEYS = (
    "lambda_min_resolution_mm",
    "lambda_min_resolution",
)


def validate_measurement_config(cfg: Dict[str, Any]) -> None:
    for key in DEPRECATED_MEASUREMENT_CONFIG_KEYS:
        if key not in cfg:
            continue
        val = cfg[key]
        if val is None:
            continue
        if isinstance(val, float) and np.isnan(val):
            continue
        raise ValueError(
            f"config key {key!r} is deprecated and must not be set "
            f"(geometry_longwave_2D uses lambda_min_search = 2*D only; got {val!r})"
        )
    bd = cfg.get("baseline_degree", 2)
    td = cfg.get("trend_degree", 1)
    if int(bd) != 2:
        raise ValueError(f"baseline_degree must be 2 for formal v2 method (got {bd!r})")
    if int(td) != 1:
        raise ValueError(f"trend_degree must be 1 for formal v2 method (got {td!r})")
    w = cfg.get("weights", None)
    if w is not None:
        raise ValueError(f"non-None weights are not supported in formal v2 method (got type {type(w)!r})")


def input_coordinate_scale(P: np.ndarray) -> float:
    P = np.asarray(P, float)
    if P.size == 0:
        return 0.0
    Pc = P - np.mean(P, axis=0, keepdims=True)
    return float(max(np.linalg.norm(Pc), np.linalg.norm(P)))


def residual_numerically_zero_vs_input_scale(
    residual: np.ndarray,
    P: np.ndarray,
    *,
    c: float = RESIDUAL_VS_INPUT_SCALE_FACTOR,
) -> Dict[str, Any]:
    r = np.asarray(residual, float).ravel()
    n = float(r.size)
    rnorm = float(np.linalg.norm(r)) if n > 0 else 0.0
    scale = input_coordinate_scale(P)
    eps = float(np.finfo(float).eps)
    if n <= 0:
        tol = 0.0
        is_zero = True
    elif scale == 0.0:
        tol = 0.0
        is_zero = rnorm == 0.0
    else:
        tol = float(c) * np.sqrt(n) * eps * scale
        is_zero = rnorm <= tol
    return {
        "is_zero": bool(is_zero),
        "residual_norm": rnorm,
        "input_scale": scale,
        "tol": tol,
        "n_elem": int(n),
        "c": float(c),
        "eps": eps,
    }


# ---------------------------------------------------------------------------
    
# ---------------------------------------------------------------------------
def default_measurement_config() -> Dict[str, Any]:
    from sin_trend_paths import PARTIAL_R2_GATE_ENABLED, BOUNDARY_GATE_ENABLED

    return dict(
        voxel_size=0.5,
        removal=10,                            
        trend_degree=1,
        baseline_degree=2,
                           
        search_band_mode=BAND_GEOMETRY_2D,
        diameter_wavelength_factor=DIAMETER_WAVELENGTH_FACTOR,
        N_cyc_search=0.5,
        q_fine=8,
        weights=None,
        boundary_policy=BOUNDARY_POLICY_QC_ONLY,
        boundary_gate_enabled=bool(BOUNDARY_GATE_ENABLED),
        boundary_margin_bins=2,                             
                                                               
        legacy_N_pp=20,
        legacy_ds_eff=1.0,  # voxel
        legacy_min_trimmed_length_mm=10.0,
                                   
        amplitude_gate_enabled=False,
        snr_gate_enabled=False,
        harmonic_ratio_gate_enabled=False,
        partial_r2_gate_enabled=bool(PARTIAL_R2_GATE_ENABLED),
                                                   
        coh_thr=0.75,
        patch_r_ratio=3.0,
        r_min_mm=2.5,
        r_max_mm=5.0,
        stride=10,                         
        min_neighbors=30,
                                        
        gap_fill_size=3,
                                            
        bridge_max_gap_mm=5.0,
                                                         
                                                       
        bridge_axis_max_deg=45.0,
                                                                          
                                                       
        cortex_coverage_thresh=0.30,
        constraint_coverage_thresh=0.50,
        squared_displacement_coverage_thresh=0.50,
                                                     
        # Predominantly tangential: R_out < 0.25 ⇔ R_tan > 0.75（θ_eq=30°）
        # Substantial normal component: R_out ≥ 0.25 ⇔ R_tan ≤ 0.75
        R_out_thresh=0.25,
        R_tan_thresh=0.75,
                                                           
        observed_tangent_source="frenet",
        enable_plots=False,
        verbose=False,
                                              
        fitting_coordinate=PHASE_COORDINATE,
        phase_coordinate=PHASE_COORDINATE,
        trend_coordinate=TREND_COORDINATE,
        quadratic_detrend_coordinate=QUADRATIC_DETREND_COORDINATE,
        grid_length_basis=GRID_LENGTH_BASIS,
        radius_basis=RADIUS_BASIS,
        slenderness_definition=SLENDERNESS_DEFINITION,
    )


# ---------------------------------------------------------------------------
                                   
# ---------------------------------------------------------------------------
def measurement_row_schema() -> Dict[str, Any]:
    nan = float("nan")
    return {
        # ID
        "IXI_ID": nan,
        "directory_id": "",
        "segment_id": nan,
        "segment_index": nan,
        "mca_segment": None,
        "side": "",
        # version / method metadata
        "output_tag": OUTPUT_TAG,
        "measurement_version": MEASUREMENT_VERSION,
        "measurement_revision": MEASUREMENT_REVISION,
        "frame_reference": "",
        "frame_transport_method": "",
        "tangent_source": "",
        "baseline_method": BASELINE_METHOD,
        "correspondence_method": CORRESPONDENCE_METHOD,
        "fitting_coordinate": FITTING_COORDINATE,
        "r2_definition": R2_DEFINITION,
        "k_selection_method": K_SELECTION_METHOD,
        "fit_method": FIT_METHOD,
        "amp_comb_mm": nan,
        "amp_major_mm": nan,
        "amp_minor_mm": nan,
        "sigma1_mm": nan,
        "sigma2_mm": nan,
        "anisotropy_1_minus_s2_over_s1": nan,
        "anisotropy_status": "",
        "R2_trend_only": nan,
        "SSE_full_mm2": nan,
        "SSE_tr_mm2": nan,
        "SST_mm2": nan,
        "N_fit": nan,
        "lambda_baseline_eff_mm": nan,  # alias of lambda_hat_mm (baseline-phase primary)
        "lambda_baseline_mm": nan,  # alias of lambda_hat_mm
        "L_base_mm": nan,
        "L_baseline_fit_mm": nan,  # alias of L_base_mm
        "L_base_over_L_obs": nan,
        "phase_coordinate": "",
        "trend_coordinate": "",
        "quadratic_detrend_coordinate": "",
        "grid_length_basis": "",
        "radius_basis": "",
        "ds_phase_median_vox": nan,
        "slenderness_definition": "",
        "wavelength_conversion_status": "",
        "wavelength_conversion_reason": "",
        "R_out_latent": nan,
        "R_tan_latent": nan,
        "R_out_response": nan,
        "R_tan_response": nan,
        "squared_displacement_coverage_latent": nan,
        "squared_displacement_coverage_response": nan,
        "surface_displacement_representation": "",
        "rank_X": nan,
        "rank_Z": nan,
        "cond_X": nan,
        "rcond": nan,
        "L_baseline_fit_internal": nan,
        "baseline_integral_abserr": nan,
        "baseline_integral_q_a": nan,
        "baseline_integral_q_b": nan,
        "directionality_legacy_s2_over_s1_sq": nan,
        "lambda_base_eff_pct_diff_vs_lambda_s": nan,
        "phase_origin_s_vox": nan,
        "trend_t_mean_s_vox": nan,
        "principal_axis_unique": nan,
        "R_dir_evaluable": nan,
        "coef_trend_U_c_mm": nan,
        "coef_trend_U_d": nan,
        "coef_trend_V_c_mm": nan,
        "coef_trend_V_d": nan,
        # sampling / trim
        "pts": nan,
        "pts_trimmed": nan,
        "ds_mm": nan,
        "ds_median_vox": nan,
        "L_full_mm": nan,
        "L_trimmed_mm": nan,
        "removal_points": nan,
        "removed_mm_start": nan,
        "removed_mm_end": nan,
        "stride_points": nan,
        "stride_mm": nan,
        "voxel_size_mm": nan,
        "coordinate_unit_input": "voxel",
                                                    
        "radius_median_mm": nan,
        "radius_mean_mm": nan,
        "radius_std_mm": nan,
        "radius_status": "missing",
        "radius_median_full_mm": nan,
        "radius_mean_full_mm": nan,
        "radius_std_full_mm": nan,
        "radius_status_full": "missing",
        "euclid_distance_mm": nan,
        "centerline_geometry_sha256": "",
        "TI": nan,
        "rho": nan,
        "kappa_median_mm": nan,
        "kappa_mean_mm": nan,
        "kappa_std_mm": nan,
        "kappa_rms_mm": nan,
        "kappa_p95_mm": nan,
        "kappa_max_mm": nan,
        "kappa_min_mm": nan,
        "curvature_status": "missing",
        # slenderness = lambda_hat_mm / (2 * radius_median_mm)
        "slenderness": nan,
        "eigval1_3d_mm": nan,
        "eigval2_3d_mm": nan,
        "eigval3_3d_mm": nan,
        "linearity": nan,
        "planarity": nan,
        "scattering": nan,
        "omnivariance": nan,
        "anisotropy": nan,
        "change_curvature": nan,
        "qc_pca_status": "not_run",
        "qc_pca_reason": "",
        "min_length_mm": nan,                                            
        "eligibility_length_basis": "",
        "interval_eligible": 0,
        "search_band_mode": "",
        "diameter_ref_mm": nan,
        "diameter_definition": "",
        "diameter_wavelength_factor": nan,
        "lambda_min_geometry_mm": nan,
        "lambda_over_radius_gt4_boundary": nan,                                
        "ds_max_gap_mm": nan,
        "ds_max_gap_vox": nan,
        "n_samples_per_cycle_hat": nan,
        # baseline
        "baseline_degree": nan,
        "baseline_status": "",
        "baseline_status_reason": "",
        "baseline_c2_x": nan,
        "baseline_c2_y": nan,
        "baseline_c2_z": nan,
        "baseline_c1_x": nan,
        "baseline_c1_y": nan,
        "baseline_c1_z": nan,
        "baseline_c0_x": nan,
        "baseline_c0_y": nan,
        "baseline_c0_z": nan,
        "baseline_coef_unit": "voxel (q dimensionless)",
        "baseline_q_min": nan,
        "baseline_q_max": nan,
        "baseline_s_first_vox": nan,
        "baseline_s_last_vox": nan,
        "baseline_dnorm_min": nan,
        "baseline_dnorm_median": nan,
        "baseline_dnorm_max": nan,
        "baseline_dnorm_min_continuous": nan,
        "baseline_tangent_rel_tol": nan,
        # frame
        "frame_status": "",
        "frame_status_reason": "",
        "frame_orth_err_max": nan,
        "frame_adjacent_angle_max_deg": nan,
        "frame_adjacent_angle_median_deg": nan,
        "observed_tangent_available": 0,
                       
        "deltaP_mean_mm": nan,
        "deltaP_p95_mm": nan,
        "deltaP_max_mm": nan,
        "resid_parallel_ss_mm2": nan,
        "resid_perp_ss_mm2": nan,
        "resid_parallel_over_perp": nan,
        "residual_input_scale_vox": nan,
        "residual_norm_vs_input_vox": nan,
        "residual_zero_tol_vs_input_vox": nan,
        "residual_numerically_zero_vs_input": 0,
        "fit_signal_norm_vs_input_vox": nan,
        "fit_signal_zero_tol_vs_input_vox": nan,
        "fit_signal_numerically_zero_vs_input": 0,
        "perp_recon_err_max_mm": nan,
        "perp_dot_T_max_mm": nan,
        "angle_T0_Tobs_mean_deg": nan,
        "angle_T0_Tobs_median_deg": nan,
        "angle_T0_Tobs_p95_deg": nan,
        "angle_T0_Tobs_max_deg": nan,
        "angle_T0_Tobs_frac_gt90": nan,
                
        "N_cyc": nan,
        "N_cyc_search": nan,
        "N_pp": nan,                                 
        "ds_eff": nan,
        "legacy_N_pp_role": "",
        "legacy_ds_eff_role": "",
        "q_fine": nan,
        "boundary_margin_bins": nan,
        "boundary_policy": "",
        "boundary_gate_enabled": 0,
        "trend_degree": nan,
        "lambda_min_search_mm": nan,
        "lambda_max_search_mm": nan,
        "k_min_mm": nan,
        "k_max_mm": nan,
        "dk_mm": nan,
        "appended_k_max": 0,
        "final_grid_step_shorter_than_dk": 0,
        "SSE_full_unit": "voxel^2",
        "SST_unit": "voxel^2",
        "k_unit": "rad/voxel (k_hat_mm: rad/mm)",
        "lambda_unit": "voxel in fit; lambda_hat_mm is baseline-arclength wavelength in mm",
                
        "fit_status": "unknown",
        "fit_status_reason": "",
        "failure_stage": "",
        "k_hat_mm": nan,
        "lambda_hat_mm": nan,
        "lambda_hat_s_mm": nan,
        "amp_hat_U_mm": nan,
        "amp_hat_V_mm": nan,
        "coef_uU_cos_mm": nan,
        "coef_uU_sin_mm": nan,
        "coef_uV_cos_mm": nan,
        "coef_uV_sin_mm": nan,
        "eigval1_2d_mm": nan,
        "eigval2_2d_mm": nan,
        "sigma1_mm": nan,
        "sigma2_mm": nan,
        "sigma2_over_sigma1": nan,
        "A_comb_over_sigma1": nan,
        "phase_variance1_mm2": nan,
        "phase_variance2_mm2": nan,
        "directionality": nan,
        "amp_comb_over_L": nan,
        "lambda_hat_over_radius": nan,
        "amp_comb_over_r": nan,
        "R2": nan,
        "partial_R2": nan,
        "SSE_full": nan,
        "SSE_tr": nan,
        "SST": nan,
        "hr_amp": nan,
        "hr_energy": nan,
        "hr_status": "",
        "hr_status_reason": "",
        "snr_status": "",
        "coverage_hat": nan,
        "SNR": nan,
        "rms_residual_full_mm": nan,
        "rms_residual_tr_mm": nan,
        "boundary_hit_low": 0,
        "boundary_hit_high": 0,
        "boundary_exact_low": 0,
        "boundary_exact_high": 0,
        "boundary_hit_low_2bins": 0,
        "boundary_hit_high_2bins": 0,
        "boundary_distance_high_k_mm": nan,
        "boundary_distance_high_relative": nan,
        "lambda_over_search_min": nan,
        "lambda_over_diameter": nan,
        "selected_index": nan,
        "n_exact_sse_ties": nan,
        "n_valid_candidates": nan,
        "k_min": nan,
        "k_max": nan,
        "dk": nan,
        "n_grid": nan,
        "legacy_tie_k_hat_mm": nan,
        "legacy_tie_lambda_hat_mm": nan,
        "legacy_tie_R2": nan,
                                   
        "patch_r_ratio": nan,
        "r_patch_definition": "",
        "r_patch_mm_raw": nan,
        "r_patch_mm": nan,
        "r_patch_vox": nan,
        "coh_thr": nan,
        "cortex_coverage_thresh": nan,
        "constraint_coverage_thresh": nan,
        "squared_displacement_coverage_thresh": nan,
        "R_out_thresh": nan,
        "R_tan_thresh": nan,
        "min_neighbors": nan,
        "gap_fill_size": nan,
        "bridge_max_gap_mm": nan,
        "bridge_axis_max_deg": nan,
        "surface_method_revision": "",
        "normals_cache_status": "",
        "R_out_surface": nan,
        "R_tan_surface": nan,
        "R_dir_surface": nan,
        "R_out_upper_bound": nan,
        "R_out_within_upper_bound": nan,
        "cortex_coverage": nan,
        "constraint_coverage": nan,
        "squared_displacement_coverage": nan,
        "total_energy": nan,
        "energy_constrained": nan,
        "perp_energy": nan,
        "evaluated_tangential_fraction": nan,
        "abs_T0_dot_n_mean": nan,
        "abs_T0_dot_n_median": nan,
        "abs_Tobs_dot_n_mean": nan,
        "abs_Tobs_dot_n_median": nan,
        "n_points_total": nan,
        "n_eval": nan,
                                                    
        "n_eval_good": nan,
        "n_eval_bridged": nan,
        "good_coverage": nan,
        "bridged_coverage": nan,
        "squared_displacement_coverage_good": nan,
        "squared_displacement_coverage_bridged": nan,
        "n_anchors": nan,
        "n_anchors_observed_normal": nan,
        "surface_evaluation_status": "skipped",
        "surface_evaluation_reason": "",
            
        "candidate_csv_path": "",
        "candidates_save_status": "not_attempted",
        "pointwise_npz_path": "",
        "pointwise_save_status": "not_attempted",
                                                 
        **{k: ("" if k in {"core_profile", "partial_r2_policy", "periodicity", "reject_reasons", "classification_revision", "boundary_policy"} else (None if k in {"surface_constraint", "plane_class"} else (False if "finalized" in k else 0 if k.endswith(("enabled", "eligible", "valid")) or k.startswith(("good_fit", "core_")) else nan))) for k in classification_column_names()},
    }


def failure_row(
    base: Dict[str, Any],
    *,
    fit_status: str,
    fit_status_reason: str,
    failure_stage: str = "",
    surface_evaluation_status: str = "skipped",
    surface_evaluation_reason: str = "",
) -> Dict[str, Any]:
    row = measurement_row_schema()
    row.update(base)
    row["fit_status"] = fit_status
    row["fit_status_reason"] = fit_status_reason
    row["failure_stage"] = failure_stage
    row["surface_evaluation_status"] = surface_evaluation_status
    row["surface_evaluation_reason"] = surface_evaluation_reason
    return apply_classification(row)


def frame_metadata(frame_reference: str, cfg: Dict[str, Any]) -> Dict[str, str]:
    fr = validate_frame_reference(frame_reference)
    if fr == FRAME_BASELINE:
        return {
            "frame_reference": fr,
            "frame_transport_method": FRAME_TRANSPORT_METHOD,
            "tangent_source": TANGENT_SOURCE_BASELINE,
        }
    src = str(cfg.get("observed_tangent_source", "frenet"))
    tsrc = TANGENT_SOURCE_FRENET if src == "frenet" else TANGENT_SOURCE_NUMERIC
    if fr == FRAME_OBSERVED:
        return {"frame_reference": fr, "frame_transport_method": FRAME_TRANSPORT_METHOD, "tangent_source": tsrc}
    return {"frame_reference": fr, "frame_transport_method": FRAME_TRANSPORT_METHOD_LEGACY, "tangent_source": tsrc}


# ---------------------------------------------------------------------------
         
# ---------------------------------------------------------------------------
def ixi_id_from_directory(directory_id: str) -> float:
    try:
        return float(int(str(directory_id)[3:6].lstrip("0") or "0"))
    except (ValueError, IndexError):
        return float("nan")


def run_shape_pca(P_vox: np.ndarray, voxel_size: float) -> Tuple[Dict[str, float], str, str]:
    try:
        from sklearn.decomposition import PCA
    except Exception as exc:  # noqa: BLE001
        return {}, "unavailable", f"sklearn import failed: {exc}"
    try:
        pca = PCA(n_components=3)
        pca.fit(np.asarray(P_vox, float) * voxel_size)
        e1, e2, e3 = map(float, pca.explained_variance_)
        tot = e1 + e2 + e3
        return (
            {
                "eigval1_3d_mm": e1,
                "eigval2_3d_mm": e2,
                "eigval3_3d_mm": e3,
                "linearity": (e1 - e2) / e1 if e1 > 0 else float("nan"),
                "planarity": (e2 - e3) / e1 if e1 > 0 else float("nan"),
                "scattering": e3 / e1 if e1 > 0 else float("nan"),
                "omnivariance": float((e1 * e2 * e3) ** (1 / 3)) if e1 * e2 * e3 >= 0 else float("nan"),
                "anisotropy": (e1 - e3) / e1 if e1 > 0 else float("nan"),
                "change_curvature": e3 / tot if tot > 0 else float("nan"),
            },
            "ok",
            "",
        )
    except Exception as exc:  # noqa: BLE001
        return {}, "failed", str(exc)


def svd_coef_qc(search: Dict[str, Any], voxel_size: float) -> Tuple[Dict[str, float], np.ndarray]:
    keys = ("uU_cos", "uU_sin", "uV_cos", "uV_sin")
    out: Dict[str, float] = {}
    if not all(np.isfinite(search.get(k, np.nan)) for k in keys):
        out["directionality"] = float("nan")
        return out, np.array([1.0, 0.0])
    M_mm = np.array(
        [
            [search["uU_cos"] * voxel_size, search["uU_sin"] * voxel_size],
            [search["uV_cos"] * voxel_size, search["uV_sin"] * voxel_size],
        ],
        float,
    )
    U_mat, S_vals, _ = np.linalg.svd(M_mm, full_matrices=True)
    s1, s2 = float(S_vals[0]), float(S_vals[1])
    g1 = U_mat[:, 0].copy()
    if g1[0] < 0:
        g1 *= -1
    A_mm = float(search.get("A_comb", np.nan)) * voxel_size
                                                                 
    principal_unique = 1
    if s1 > 0 and np.isfinite(s2):
        anis = float((s1 - s2) / s1)
        anis_status = "ok"
        if np.isclose(s1, s2, rtol=1e-9, atol=1e-15):
            principal_unique = 0
            anis_status = "ok_circular_principal_axis_not_unique"
    else:
        anis = float("nan")
        anis_status = "undefined"
        principal_unique = 0
    out.update(
        eigval1_2d_mm=s1,
        eigval2_2d_mm=s2,
        sigma1_mm=s1,
        sigma2_mm=s2,
        sigma2_over_sigma1=(s2 / s1) if s1 > 0 else float("nan"),
        A_comb_over_sigma1=(A_mm / s1) if s1 > 0 and np.isfinite(A_mm) else float("nan"),
        phase_variance1_mm2=0.5 * s1**2,
        phase_variance2_mm2=0.5 * s2**2,
        anisotropy_1_minus_s2_over_s1=anis,
        anisotropy_status=anis_status,
        directionality=anis,
        directionality_legacy_s2_over_s1_sq=(s2 / s1) ** 2 if s1 > 0 else float("nan"),
        principal_axis_unique=int(principal_unique),
        R_dir_evaluable=int(principal_unique),
    )
    return out, g1


def candidate_table(search: Dict[str, Any], voxel_size: float, ids: Dict[str, Any], frame_reference: str) -> pd.DataFrame:
    cols = [
        "directory_id",
        "segment_id",
        "segment_index",
        "frame_reference",
        "measurement_version",
        "measurement_revision",
        "fit_method",
        "candidate_index",
        "k_rad_per_voxel",
        "lambda_voxel",
        "k_rad_per_mm",
        "lambda_mm",
        "R2",
        "partial_R2",
        "R2_trend_only",
        "SSE_full",
        "SSE_tr",
        "SST",
        "SSE_unit",
        "a1",
        "b1",
        "a2",
        "b2",
        "c1",
        "d1",
        "c2",
        "d2",
        "A_comb",
        "sigma1",
        "sigma2",
        "hr_amp",
        "fit_status",
        "fit_status_reason",
        "rank_full",
        "rank_tr",
        "rank_X",
        "rank_Z",
        "cond_X",
        "rcond",
        "boundary_exact_low",
        "boundary_exact_high",
        "is_selected",
    ]
    selected = search.get("selected_index")
    rows: List[Dict[str, Any]] = []
    for c in search.get("candidates") or []:
        k_vox = float(c.get("k", np.nan))
        lam_vox = float(c.get("lambda", np.nan))
        cand_i = c.get("candidate_index", c.get("index", -1))
        rows.append(
            {
                **{k: ids.get(k) for k in ("directory_id", "segment_id", "segment_index")},
                "frame_reference": frame_reference,
                "measurement_version": MEASUREMENT_VERSION,
                "measurement_revision": MEASUREMENT_REVISION,
                "fit_method": FIT_METHOD,
                "candidate_index": int(cand_i) if cand_i is not None else -1,
                "k_rad_per_voxel": k_vox,
                "lambda_voxel": lam_vox,
                "k_rad_per_mm": k_vox / voxel_size if np.isfinite(k_vox) else np.nan,
                "lambda_mm": lam_vox * voxel_size if np.isfinite(lam_vox) else np.nan,
                "R2": c.get("R2", np.nan),
                "partial_R2": c.get("partial_R2", np.nan),
                "R2_trend_only": c.get("R2_trend_only", np.nan),
                "SSE_full": c.get("SSE_full", np.nan),
                "SSE_tr": c.get("SSE_tr", np.nan),
                "SST": c.get("SST", np.nan),
                "SSE_unit": search.get("SSE_unit", "voxel^2"),
                "a1": c.get("a1", np.nan),
                "b1": c.get("b1", np.nan),
                "a2": c.get("a2", np.nan),
                "b2": c.get("b2", np.nan),
                "c1": c.get("c1", np.nan),
                "d1": c.get("d1", np.nan),
                "c2": c.get("c2", np.nan),
                "d2": c.get("d2", np.nan),
                "A_comb": c.get("A_comb", np.nan),
                "sigma1": c.get("sigma1", np.nan),
                "sigma2": c.get("sigma2", np.nan),
                "hr_amp": c.get("hr_amp", np.nan),
                "fit_status": c.get("fit_status", ""),
                "fit_status_reason": c.get("fit_status_reason", ""),
                "rank_full": c.get("rank_X", c.get("rank_full", np.nan)),
                "rank_tr": c.get("rank_Z", c.get("rank_tr", np.nan)),
                "rank_X": c.get("rank_X", np.nan),
                "rank_Z": c.get("rank_Z", np.nan),
                "cond_X": c.get("cond_X", np.nan),
                "rcond": c.get("rcond", search.get("rcond", np.nan)),
                "boundary_exact_low": int(bool(c.get("boundary_exact_low", False))),
                "boundary_exact_high": int(bool(c.get("boundary_exact_high", False))),
                "is_selected": int(
                    selected is not None
                    and cand_i is not None
                    and int(cand_i) == int(selected)
                ),
            }
        )
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
       
# ---------------------------------------------------------------------------
def select_observed_tangent(
    P_vox: np.ndarray,
    s_vox: np.ndarray,
    observed_tangent: Optional[np.ndarray],
    cfg: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], str]:
    src = str(cfg.get("observed_tangent_source", "frenet")).lower()
    if src == "numeric":
        return numeric_tangent(P_vox, s_vox), TANGENT_SOURCE_NUMERIC
    if observed_tangent is None:
        return None, "missing"
    return np.asarray(observed_tangent, float), TANGENT_SOURCE_FRENET


def build_frame(
    frame_reference: str,
    *,
    baseline: Dict[str, Any],
    observed_tangent: Optional[np.ndarray],
    P_vox: np.ndarray,
    s_vox: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    fr = validate_frame_reference(frame_reference)
    if fr == FRAME_BASELINE:
        T0, st, reason = analytic_baseline_tangent(baseline)
        if T0 is None:
            return {"status": f"baseline_tangent_{st}", "status_reason": reason, "T": None, "U": None, "V": None}
        return minimal_rotation_transport(T0)

    # observed / observed_legacy
    T_obs, _src_tag = select_observed_tangent(P_vox, s_vox, observed_tangent, cfg)
    if T_obs is None:
        return {
            "status": "observed_tangent_missing",
            "status_reason": "observed tangent (FrenetTangent) unavailable; numeric alternative not enabled",
            "T": None,
            "U": None,
            "V": None,
        }
    T_obs = np.asarray(T_obs, float)
    if T_obs.shape != P_vox.shape or not np.isfinite(T_obs).all():
        return {
            "status": "observed_tangent_invalid",
            "status_reason": f"observed tangent shape {T_obs.shape} or non-finite",
            "T": None,
            "U": None,
            "V": None,
        }
    out: Dict[str, Any]
    if fr == FRAME_OBSERVED:
        out = minimal_rotation_transport(T_obs)
    else:
        out = legacy_projection_frame(T_obs)
    out["T_obs_selected"] = T_obs
    out["observed_tangent_source_resolved"] = _src_tag
    return out


def measure_interval(
    P_vox: np.ndarray,
    *,
    frame_reference: str,
    cfg: Optional[Dict[str, Any]] = None,
    observed_tangent: Optional[np.ndarray] = None,
    radius_vox: Optional[np.ndarray] = None,
    curvature_vox: Optional[np.ndarray] = None,
    point_ids: Optional[np.ndarray] = None,
    ids: Optional[Dict[str, Any]] = None,
    verbose: Optional[bool] = None,
    compute_pca: bool = True,
) -> Dict[str, Any]:
    cfg = {**default_measurement_config(), **(cfg or {})}
    validate_measurement_config(cfg)
    verbose = bool(cfg["verbose"]) if verbose is None else bool(verbose)
    voxel_size = float(cfg["voxel_size"])
    removal = int(cfg["removal"])
    fr = validate_frame_reference(frame_reference)
    ids = dict(ids or {})

    band_mode = str(cfg.get("search_band_mode", BAND_GEOMETRY_2D))
    n_cyc_search = float(cfg.get("N_cyc_search", 0.5))
    base: Dict[str, Any] = {
        "directory_id": ids.get("directory_id", ""),
        "segment_id": ids.get("segment_id", np.nan),
        "segment_index": ids.get("segment_index", np.nan),
        "IXI_ID": ids.get("IXI_ID", ixi_id_from_directory(str(ids.get("directory_id", "")))),
        "side": ids.get("side", ""),
        "min_length_mm": float(cfg.get("legacy_min_trimmed_length_mm", 10.0)),
        "eligibility_length_basis": ELIGIBILITY_BASIS_LEGACY if band_mode == BAND_LEGACY_FIXED_10MM else ELIGIBILITY_BASIS_FULL,
        "search_band_mode": band_mode,
        "diameter_definition": DIAMETER_DEFINITION,
        "diameter_wavelength_factor": float(cfg.get("diameter_wavelength_factor", DIAMETER_WAVELENGTH_FACTOR)),
        "voxel_size_mm": voxel_size,
        "removal_points": removal,
        "stride_points": int(cfg["stride"]),
        "baseline_degree": int(cfg["baseline_degree"]),
        "N_cyc": n_cyc_search,
        "N_cyc_search": n_cyc_search,
        "N_pp": float("nan"),
        "ds_eff": float("nan"),
        "legacy_N_pp_role": "unused_in_geometry_longwave_2D",
        "legacy_ds_eff_role": "unused_in_geometry_longwave_2D",
        "q_fine": int(cfg["q_fine"]),
        "boundary_margin_bins": int(cfg["boundary_margin_bins"]),
        "boundary_policy": str(cfg.get("boundary_policy", BOUNDARY_POLICY_QC_ONLY)),
        "boundary_gate_enabled": int(bool(cfg.get("boundary_gate_enabled", False))),
        "trend_degree": int(cfg["trend_degree"]),
        "coh_thr": float(cfg["coh_thr"]),
                                                   
        "cortex_coverage_thresh": float(cfg["cortex_coverage_thresh"]),
        "constraint_coverage_thresh": float(cfg["constraint_coverage_thresh"]),
        "squared_displacement_coverage_thresh": float(cfg["squared_displacement_coverage_thresh"]),
        "R_out_thresh": float(cfg["R_out_thresh"]),
        "R_tan_thresh": float(cfg.get("R_tan_thresh", 1.0 - float(cfg["R_out_thresh"]))),
        "min_neighbors": int(cfg["min_neighbors"]),
        "gap_fill_size": int(cfg["gap_fill_size"]),
        "bridge_max_gap_mm": float(cfg.get("bridge_max_gap_mm", 5.0)),
        "bridge_axis_max_deg": float(cfg.get("bridge_axis_max_deg", 45.0)),
        "patch_r_ratio": float(cfg["patch_r_ratio"]),
        "amplitude_gate_enabled": False,
        "snr_gate_enabled": False,
        "harmonic_ratio_gate_enabled": False,
        "partial_r2_gate_enabled": int(bool(cfg.get("partial_r2_gate_enabled", True))),
        **frame_metadata(fr, cfg),
    }
    result: Dict[str, Any] = {
        "row": None,
        "search": None,
        "pointwise": None,
        "candidates": pd.DataFrame(),
        "ok_for_surface": False,
        "frame": None,
        "baseline": None,
    }

                    
    try:
        P = np.asarray(P_vox, float)
    except (TypeError, ValueError) as exc:
        result["row"] = failure_row(base, fit_status="invalid_geometry", fit_status_reason=str(exc), failure_stage="validate_points")
        return result
    if P.ndim != 2 or P.shape[1] != 3:
        result["row"] = failure_row(
            base, fit_status="invalid_geometry", fit_status_reason=f"points shape {P.shape}", failure_stage="validate_points"
        )
        return result
    if P.shape[0] < 2:
        result["row"] = failure_row(
            base, fit_status="insufficient_points", fit_status_reason="fewer than 2 centerline points", failure_stage="validate_points"
        )
        return result
    if not np.isfinite(P).all():
        result["row"] = failure_row(
            base, fit_status="nonfinite_input", fit_status_reason="non-finite coordinates", failure_stage="validate_points"
        )
        return result
    n_pts = int(P.shape[0])
    base["pts"] = n_pts
    if point_ids is None:
        point_ids = np.arange(n_pts)
    point_ids = np.asarray(point_ids)
    if len(point_ids) != n_pts:
        result["row"] = failure_row(
            base, fit_status="invalid_geometry", fit_status_reason="point_ids length mismatch", failure_stage="validate_points"
        )
        return result

                  
    s_vox, ds_arr = arclength(P)
    if len(s_vox) < 2 or not np.isfinite(s_vox[-1]) or float(s_vox[-1]) <= 0.0:
        result["row"] = failure_row(
            base, fit_status="insufficient_points", fit_status_reason="arclength produced invalid s", failure_stage="arclength"
        )
        return result
    L_vox = float(s_vox[-1])
    ds_vox = float(np.median(ds_arr))
    ds_max_vox = float(np.max(ds_arr)) if len(ds_arr) else float("nan")
    euclid = float(np.linalg.norm(P[0] - P[-1]))
    TI = L_vox / euclid if euclid > 0 else float("nan")
    base.update(
        ds_mm=ds_vox * voxel_size,
        ds_median_vox=ds_vox,
        ds_max_gap_vox=ds_max_vox,
        ds_max_gap_mm=ds_max_vox * voxel_size,
        L_full_mm=L_vox * voxel_size,
        euclid_distance_mm=euclid * voxel_size,
        TI=TI,
        rho=TI - 1.0 if np.isfinite(TI) else float("nan"),
        stride_mm=float(cfg["stride"]) * ds_vox * voxel_size,
    )
    try:
        from sin_trend_mca_labels import compute_centerline_geometry_hash

        base["centerline_geometry_sha256"] = compute_centerline_geometry_hash(P, voxel_size_mm=voxel_size)
    except Exception:  # noqa: BLE001
        base["centerline_geometry_sha256"] = ""

                                  
                                                               
    radius_median_vox = float("nan")
    r_lumen: Optional[np.ndarray] = None
    if radius_vox is not None:
        r_lumen = np.asarray(radius_vox, float)
        if len(r_lumen) != n_pts:
            base["radius_status"] = "invalid"
            base["radius_status_full"] = "invalid"
        elif not np.isfinite(r_lumen).all():
                                                          
            base["radius_status_full"] = "invalid"
        else:
            base.update(
                radius_median_full_mm=float(np.median(r_lumen)) * voxel_size,
                radius_mean_full_mm=float(np.mean(r_lumen)) * voxel_size,
                radius_std_full_mm=(float(np.std(r_lumen, ddof=1)) if n_pts > 1 else float("nan")) * voxel_size,
                radius_status_full="ok",
            )
    if curvature_vox is not None:
        kap = np.asarray(curvature_vox, float)
        if len(kap) == n_pts and np.isfinite(kap).all():
            kt = kap[removal:-removal] if n_pts > 2 * removal and removal > 0 else kap
            if len(kt) == 0:
                kt = kap
            base.update(
                kappa_median_mm=float(np.median(kt)) / voxel_size,
                kappa_mean_mm=float(np.mean(kt)) / voxel_size,
                kappa_std_mm=(float(np.std(kt, ddof=1)) if len(kt) > 1 else float("nan")) / voxel_size,
                kappa_rms_mm=float(np.sqrt(np.mean(kt**2))) / voxel_size,
                kappa_p95_mm=float(np.percentile(kt, 95)) / voxel_size,
                kappa_max_mm=float(np.max(kt)) / voxel_size,
                kappa_min_mm=float(np.min(kt)) / voxel_size,
                curvature_status="ok",
            )
        else:
            base["curvature_status"] = "invalid"
    base["observed_tangent_available"] = int(
        observed_tangent is not None
        and np.asarray(observed_tangent).shape == P.shape
        and np.isfinite(np.asarray(observed_tangent, float)).all()
    )

                        
    if n_pts - 2 * removal < 2:
        base["pts_trimmed"] = max(n_pts - 2 * removal, 0)
        result["row"] = failure_row(
            base,
            fit_status="insufficient_points",
            fit_status_reason="trimmed segment has fewer than 2 points",
            failure_stage="trim",
            surface_evaluation_status="skipped_insufficient_points",
        )
        return result
    idx_trim = np.arange(removal, n_pts - removal) if removal > 0 else np.arange(n_pts)
    base["pts_trimmed"] = int(len(idx_trim))
    L_trim_vox = float(s_vox[idx_trim[-1]] - s_vox[idx_trim[0]])
    L_trim_mm = L_trim_vox * voxel_size
    base["L_trimmed_mm"] = L_trim_mm
    base["removed_mm_start"] = float(s_vox[idx_trim[0]] - s_vox[0]) * voxel_size
    base["removed_mm_end"] = float(s_vox[-1] - s_vox[idx_trim[-1]]) * voxel_size
    if not np.isfinite(L_trim_mm) or L_trim_mm <= 0.0:
        result["row"] = failure_row(
            base,
            fit_status="nonpositive_trimmed_length",
            fit_status_reason="L_trimmed_mm is not finite positive (distinct from interval_eligible)",
            failure_stage="trim",
            surface_evaluation_status="skipped_insufficient_points",
        )
        return result
    base["L_trimmed_mm"] = L_trim_mm
    base["removed_mm_start"] = float(s_vox[idx_trim[0]] - s_vox[0]) * voxel_size
    base["removed_mm_end"] = float(s_vox[-1] - s_vox[idx_trim[-1]]) * voxel_size

                                                                     
    if r_lumen is not None and len(r_lumen) == n_pts:
        rt = r_lumen[idx_trim]
        n_rt = int(len(rt))
        if n_rt > 0 and np.isfinite(rt).all():
            radius_median_vox = float(np.median(rt))
            base.update(
                radius_median_mm=radius_median_vox * voxel_size,
                radius_mean_mm=float(np.mean(rt)) * voxel_size,
                radius_std_mm=(float(np.std(rt, ddof=1)) if n_rt > 1 else float("nan")) * voxel_size,
                radius_status="ok",
                diameter_ref_mm=2.0 * radius_median_vox * voxel_size,
            )
        else:
            base["radius_status"] = "invalid"

                                                         
    L_full_mm = float(base["L_full_mm"])
    if band_mode == BAND_LEGACY_FIXED_10MM:
        eligible = bool(np.isfinite(L_trim_mm) and L_trim_mm >= float(cfg.get("legacy_min_trimmed_length_mm", 10.0)))
        eligible_reason = "" if eligible else "L_trimmed_mm < legacy_min_trimmed_length_mm"
    else:
        eligible = bool(np.isfinite(L_full_mm) and L_full_mm > 0.0)
        if not np.isfinite(L_full_mm):
            eligible_reason = "L_full_mm not finite (not classified as too_short)"
        elif L_full_mm <= 0.0:
            eligible_reason = f"L_full_mm not positive ({L_full_mm:g})"
        else:
            eligible_reason = ""
    base["interval_eligible"] = int(eligible)

    row = measurement_row_schema()
    row.update(base)
    row["fit_status"] = "pending"
    row["surface_evaluation_status"] = "pending"

    if not eligible:
        row.update(
            fit_status="skipped_not_eligible",
            fit_status_reason=eligible_reason,
            failure_stage="interval_eligibility",
            surface_evaluation_status="skipped_not_eligible",
            surface_evaluation_reason=eligible_reason,
        )
        result["row"] = apply_classification(row)
        return result

    if compute_pca:
        shape_stats, st_pca, reason_pca = run_shape_pca(P, voxel_size)
        row["qc_pca_status"] = st_pca
        row["qc_pca_reason"] = reason_pca
        row.update(shape_stats)

                                         
    baseline = quadratic_baseline(P, s_vox, degree=int(cfg["baseline_degree"]))
    result["baseline"] = baseline
    row["baseline_status"] = baseline["status"]
    row["baseline_status_reason"] = baseline["status_reason"]
    row["baseline_tangent_rel_tol"] = baseline.get("tangent_rel_tol", np.nan)
    if "coef" in baseline:
        coef = np.asarray(baseline["coef"], float)
        names = ["c2", "c1", "c0"] if coef.shape[0] == 3 else [f"c{coef.shape[0]-1-i}" for i in range(coef.shape[0])]
        for i, nm in enumerate(names):
            for j, ax in enumerate("xyz"):
                key = f"baseline_{nm}_{ax}"
                if key in row:
                    row[key] = float(coef[i, j])
        row["baseline_q_min"] = float(np.min(baseline["q"]))
        row["baseline_q_max"] = float(np.max(baseline["q"]))
        row["baseline_s_first_vox"] = baseline["s_first"]
        row["baseline_s_last_vox"] = baseline["s_last"]
        row["baseline_dnorm_min"] = baseline["dnorm_min"]
        row["baseline_dnorm_median"] = baseline["dnorm_median"]
        row["baseline_dnorm_max"] = baseline["dnorm_max"]
        row["baseline_dnorm_min_continuous"] = baseline["dnorm_min_continuous"]
    if baseline["status"] in {"nonfinite", "degenerate_length"}:
        row.update(
            fit_status="baseline_failed",
            fit_status_reason=baseline["status_reason"],
            failure_stage="baseline",
            surface_evaluation_status="skipped_invalid_fit",
        )
        result["row"] = apply_classification(row)
        return result
    P0 = np.asarray(baseline["P0"], float)

                                           
    s_base_pack = cumulative_baseline_arclength(
        np.asarray(baseline["coef"], float),
        s_vox,
        s_first=float(baseline["s_first"]),
        s_last=float(baseline["s_last"]),
    )
    if not s_base_pack["ok"]:
        row.update(
            fit_status="baseline_arclength_failed",
            fit_status_reason=s_base_pack.get("status_reason", ""),
            failure_stage="baseline_arclength",
            surface_evaluation_status="skipped_invalid_fit",
        )
        result["row"] = apply_classification(row)
        return result
    s_base_full = np.asarray(s_base_pack["s_base_full"], float)

    # ---- frame ----
    frame = build_frame(fr, baseline=baseline, observed_tangent=observed_tangent, P_vox=P, s_vox=s_vox, cfg=cfg)
    result["frame"] = frame
    row["frame_status"] = frame.get("status", "")
    row["frame_status_reason"] = frame.get("status_reason", "")
    if frame.get("status") != "ok" or frame.get("T") is None:
        row.update(
            fit_status="frame_failed",
            fit_status_reason=f"{frame.get('status')}: {frame.get('status_reason')}",
            failure_stage="frame",
            surface_evaluation_status="skipped_invalid_fit",
        )
        result["row"] = apply_classification(row)
        return result
    T, U, V = frame["T"], frame["U"], frame["V"]
    row["frame_orth_err_max"] = frame.get("frame_orth_err_max", np.nan)
    row["frame_adjacent_angle_max_deg"] = frame.get("adjacent_angle_max_deg", np.nan)
    row["frame_adjacent_angle_median_deg"] = frame.get("adjacent_angle_median_deg", np.nan)

                                     
    lat = lateral_components_signed(P, P0, T, U, V)
    T0_full, _, _ = analytic_baseline_tangent(baseline) if baseline["status"] == "ok" else (None, "", "")
                                                                 
                                         
    T_obs_full = frame.get("T_obs_selected")
    if T_obs_full is None:
        T_obs_full, _ = select_observed_tangent(P, s_vox, observed_tangent, cfg)
    T_obs_frenet_full = None
    if observed_tangent is not None and np.asarray(observed_tangent).shape == P.shape:
        T_obs_frenet_full = np.asarray(observed_tangent, float)

    tb = trim_bundle(
        {
            "P": P,
            "P0": P0,
            "point_id": point_ids,
            "T": T,
            "U": U,
            "V": V,
            "Delta": lat["Delta"],
            "Delta_parallel": lat["Delta_parallel"],
            "d_perp": lat["d_perp"],
            "uU": lat["uU"],
            "uV": lat["uV"],
            "T0": T0_full,
            "T_obs": T_obs_full,
            "T_obs_frenet": T_obs_frenet_full,
        },
        removal,
        s_vox,
    )
    if not tb["ok"]:
        row.update(fit_status="insufficient_points", fit_status_reason=tb["reason"], failure_stage="trim")
        result["row"] = apply_classification(row)
        return result
    s_obs_trim = tb["s_fit"]                                            
                                               
    s_check, _ = arclength(tb["P"])
    if not np.allclose(s_check, s_obs_trim, rtol=0, atol=1e-9 * max(1.0, float(s_obs_trim[-1]))):
        row.update(fit_status="internal_error", fit_status_reason="trimmed s_obs mismatch", failure_stage="trim")
        result["row"] = apply_classification(row)
        return result
    J = np.asarray(tb["index"], int)
                                     
    s_phase_full = s_base_full - float(s_base_full[J[0]])
    s_phase_trim = s_phase_full[J]
    L_base_vox = float(s_base_full[J[-1]] - s_base_full[J[0]])
    if not np.isfinite(L_base_vox) or L_base_vox <= 0.0:
        row.update(
            fit_status="baseline_arclength_failed",
            fit_status_reason="L_base (trimmed baseline arclength) not finite positive",
            failure_stage="baseline_arclength",
            surface_evaluation_status="skipped_invalid_fit",
        )
        result["row"] = apply_classification(row)
        return result
    if abs(float(s_phase_trim[-1] - s_phase_trim[0]) - L_base_vox) > 1e-9 * max(1.0, L_base_vox):
        row.update(
            fit_status="internal_error",
            fit_status_reason="s_phase_trim span != L_base",
            failure_stage="baseline_arclength",
            surface_evaluation_status="skipped_invalid_fit",
        )
        result["row"] = apply_classification(row)
        return result
    L_base_mm = L_base_vox * voxel_size
    row["L_base_mm"] = L_base_mm
    row["L_baseline_fit_mm"] = L_base_mm  # alias
    row["L_base_over_L_obs"] = (L_base_vox / L_trim_vox) if L_trim_vox > 0 else float("nan")
    row["fitting_coordinate"] = PHASE_COORDINATE                                                
    row["phase_coordinate"] = PHASE_COORDINATE
    row["trend_coordinate"] = TREND_COORDINATE
    row["quadratic_detrend_coordinate"] = QUADRATIC_DETREND_COORDINATE
    row["grid_length_basis"] = GRID_LENGTH_BASIS
    row["radius_basis"] = RADIUS_BASIS
    row["slenderness_definition"] = SLENDERNESS_DEFINITION
    ds_phase = np.diff(s_phase_trim)
    row["ds_phase_median_vox"] = float(np.median(ds_phase)) if len(ds_phase) else float("nan")

    dn = tb["Delta"]
    dnorm_t = np.linalg.norm(dn, axis=1)
    ss_par = float(np.sum(tb["Delta_parallel"] ** 2))
    ss_perp = float(np.sum(tb["d_perp"] ** 2))
    row.update(
        deltaP_mean_mm=float(np.mean(dnorm_t)) * voxel_size,
        deltaP_p95_mm=float(np.percentile(dnorm_t, 95)) * voxel_size,
        deltaP_max_mm=float(np.max(dnorm_t)) * voxel_size,
        resid_parallel_ss_mm2=ss_par * voxel_size**2,
        resid_perp_ss_mm2=ss_perp * voxel_size**2,
        resid_parallel_over_perp=(ss_par / ss_perp) if ss_perp > 0 else float("nan"),
        perp_recon_err_max_mm=lat["recon_err_max"] * voxel_size,
        perp_dot_T_max_mm=lat["perp_dot_T_max"] * voxel_size,
    )
                                                  
                                                  
    P_trim = np.asarray(tb["P"], float)
    geom_zero = residual_numerically_zero_vs_input_scale(tb["Delta"], P_trim)
    fit_sig = np.concatenate([np.asarray(tb["uU"], float), np.asarray(tb["uV"], float)])
    fit_zero = residual_numerically_zero_vs_input_scale(fit_sig, P_trim)
    row.update(
        residual_input_scale_vox=geom_zero["input_scale"],
        residual_norm_vs_input_vox=geom_zero["residual_norm"],
        residual_zero_tol_vs_input_vox=geom_zero["tol"],
        residual_numerically_zero_vs_input=int(geom_zero["is_zero"]),
        fit_signal_norm_vs_input_vox=fit_zero["residual_norm"],
        fit_signal_zero_tol_vs_input_vox=fit_zero["tol"],
        fit_signal_numerically_zero_vs_input=int(fit_zero["is_zero"]),
    )
    if geom_zero["is_zero"] or fit_zero["is_zero"]:
        which = []
        if geom_zero["is_zero"]:
            which.append("DeltaP")
        if fit_zero["is_zero"]:
            which.append("uU_uV")
        row.update(
            fit_status="numerically_flat",
            fit_status_reason=(
                "residual indistinguishable from zero at float64 precision relative to input "
                f"coordinate scale ({'+'.join(which)}); not a physical amplitude gate"
            ),
            failure_stage="residual_vs_input_scale",
            surface_evaluation_status="skipped_invalid_fit",
        )
        result["pointwise"] = {
            "frame_reference": fr,
            "index_full_to_trim": tb["index"],
            "s_obs_full": s_vox,
            "s_base_full": s_base_full,
            "s_phase_full": s_phase_full,
            "s_full": s_vox,  # alias: observed
            "P_full": P,
            "P0_full": P0,
            "point_id_full": point_ids,
            "P": tb["P"],
            "P0": tb["P0"],
            "point_id": tb["point_id"],
            "s_obs_trim": s_obs_trim,
            "s_phase_trim": s_phase_trim,
            "s_trend_trim": s_phase_trim,  # linear trend on baseline phase
            "s_fit": s_phase_trim,  # primary phase for latent (baseline)
            "T": tb["T"],
            "U": tb["U"],
            "V": tb["V"],
            "Delta": tb["Delta"],
            "Delta_parallel": tb["Delta_parallel"],
            "d_perp": tb["d_perp"],
            "uU": tb["uU"],
            "uV": tb["uV"],
            "T0": tb["T0"],
            "T_obs": tb["T_obs"],
            "T_obs_frenet": tb.get("T_obs_frenet"),
        }
        result["row"] = apply_classification(row)
        return result
    if tb.get("T0") is not None and tb.get("T_obs") is not None:
        ang = tangent_angle_deg(tb["T0"], tb["T_obs"])
        row.update(
            angle_T0_Tobs_mean_deg=float(np.mean(ang)),
            angle_T0_Tobs_median_deg=float(np.median(ang)),
            angle_T0_Tobs_p95_deg=float(np.percentile(ang, 95)),
            angle_T0_Tobs_max_deg=float(np.max(ang)),
            angle_T0_Tobs_frac_gt90=float(np.mean(ang > 90.0)),
        )

                                                     
    diameter_ref_mm = float(row.get("diameter_ref_mm", np.nan))
    band = build_search_band(
        cfg,
        L_fit_mm=L_base_mm,
        L_trimmed_mm=L_trim_mm,
        diameter_ref_mm=diameter_ref_mm,
    )
    for k in (
        "search_band_mode",
        "lambda_min_search_mm",
        "lambda_max_search_mm",
        "lambda_min_geometry_mm",
        "k_min_mm",
        "k_max_mm",
        "dk_mm",
        "k_min",
        "k_max",
        "dk",
        "n_grid",
        "appended_k_max",
        "legacy_N_pp_role",
        "legacy_ds_eff_role",
        "grid_length_basis",
        "L_fit_mm",
    ):
        if k in band:
            row[k] = band[k]
    row["final_grid_step_shorter_than_dk"] = int(bool(band.get("final_step_shorter_than_dk", False)))
    if band.get("legacy_N_pp") is not None:
        row["N_pp"] = band.get("legacy_N_pp", np.nan)
        row["ds_eff"] = band.get("legacy_ds_eff", np.nan)
    if not band.get("ok", False):
        row.update(
            fit_status=str(band.get("status", "invalid_band")),
            fit_status_reason=str(band.get("reason", "")),
            failure_stage="search_band",
            surface_evaluation_status="skipped_invalid_fit",
        )
        result["row"] = apply_classification(row)
        return result

    q_full = np.asarray(baseline["q"], float)

    search = search_k_forward_3d(
        u1=tb["uU"],
        u2=tb["uV"],
        s_phase_full=s_phase_full,
        s_phase_trim=s_phase_trim,
        s_trend_trim=s_phase_trim,
        q_full=q_full,
        U_full=U,
        V_full=V,
        J=J,
        ks=band["ks"],
        k_min=float(band["k_min"]),
        k_max=float(band["k_max"]),
        dk=float(band["dk"]),
        verbose=verbose,
    )
    search["SSE_unit"] = "voxel^2"
    search["k_min"] = float(band["k_min"])
    search["k_max"] = float(band["k_max"])
    search["dk"] = float(band["dk"])
    search["n_grid"] = int(band.get("n_grid", len(band["ks"])))
    result["search"] = search
    row["fit_method"] = FIT_METHOD
    row["k_selection_method"] = K_SELECTION_METHOD
    row["r2_definition"] = R2_DEFINITION
    row["fit_status"] = str(search.get("status", "unknown"))
    row["fit_status_reason"] = str(search.get("status_reason", ""))
    if row["fit_status"] != "ok":
        row["failure_stage"] = "fit"

    k_hat = search.get("k_hat", np.nan)
    lam = search.get("lambda_hat", np.nan)
    row["k_hat_mm"] = k_hat / voxel_size if np.isfinite(k_hat) else np.nan
    row["lambda_hat_mm"] = lam * voxel_size if np.isfinite(lam) else np.nan
    row["lambda_hat_s_mm"] = row["lambda_hat_mm"]
                                                                           
    v_comb = search.get("A_comb", np.nan)
    row["amp_comb_mm"] = v_comb * voxel_size if np.isfinite(v_comb) else np.nan
    A_U = np.hypot(search.get("uU_cos", np.nan), search.get("uU_sin", np.nan))
    A_V = np.hypot(search.get("uV_cos", np.nan), search.get("uV_sin", np.nan))
    row["amp_hat_U_mm"] = A_U * voxel_size if np.isfinite(A_U) else np.nan
    row["amp_hat_V_mm"] = A_V * voxel_size if np.isfinite(A_V) else np.nan
    for src, dst in [
        ("uU_cos", "coef_uU_cos_mm"),
        ("uU_sin", "coef_uU_sin_mm"),
        ("uV_cos", "coef_uV_cos_mm"),
        ("uV_sin", "coef_uV_sin_mm"),
    ]:
        v = search.get(src, np.nan)
        row[dst] = v * voxel_size if np.isfinite(v) else np.nan
    fit = search.get("fit") or {}
    for src, dst in [("c1", "coef_trend_U_c_mm"), ("d1", "coef_trend_U_d"), ("c2", "coef_trend_V_c_mm"), ("d2", "coef_trend_V_d")]:
        v = fit.get(src, np.nan)
        if dst.endswith("_mm"):
            row[dst] = v * voxel_size if np.isfinite(v) else np.nan
        else:
            row[dst] = v if np.isfinite(v) else np.nan
    for k in ("R2", "partial_R2", "R2_trend_only", "SSE_full", "SSE_tr", "SST", "hr_amp", "hr_energy", "k_min", "k_max", "dk", "n_grid"):
        row[k] = search.get(k, np.nan)
                                                                
    for k in ("rank_X", "rank_Z", "cond_X", "rcond"):
        v = search.get(k, None)
        if v is None and isinstance(fit, dict):
            v = fit.get(k, np.nan)
        try:
            fv = float(v) if v is not None else float("nan")
            row[k] = fv if np.isfinite(fv) else np.nan
        except (TypeError, ValueError):
            row[k] = np.nan
    row["SSE_full_mm2"] = row["SSE_full"] * voxel_size**2 if np.isfinite(row.get("SSE_full", np.nan)) else np.nan
    row["SSE_tr_mm2"] = row["SSE_tr"] * voxel_size**2 if np.isfinite(row.get("SSE_tr", np.nan)) else np.nan
    row["SST_mm2"] = row["SST"] * voxel_size**2 if np.isfinite(row.get("SST", np.nan)) else np.nan
                                               
    row["coverage_hat"] = L_base_vox / lam if np.isfinite(lam) and lam > 0 else np.nan
    row["N_fit"] = row["coverage_hat"]
                                    
    ds_ph = float(row.get("ds_phase_median_vox", np.nan))
    if np.isfinite(lam) and lam > 0 and np.isfinite(ds_ph) and ds_ph > 0:
        row["n_samples_per_cycle_hat"] = float(lam) / ds_ph
    else:
        row["n_samples_per_cycle_hat"] = np.nan
    sel = search.get("selected_index")
    row["selected_index"] = sel if sel is not None else np.nan
    row["n_exact_sse_ties"] = search.get("n_exact_sse_ties", np.nan)
    row["n_valid_candidates"] = int(
        sum(1 for c in (search.get("candidates") or []) if str(c.get("fit_status")) == "ok" and np.isfinite(c.get("SSE_full", np.nan)))
    )
    row["phase_origin_s_vox"] = search.get("phase_origin_s", np.nan)
    row["trend_t_mean_s_vox"] = search.get("t_mean_s", np.nan)
    row.update(
        boundary_diagnostics(
            selected_index=sel if sel is not None else np.nan,
            n_grid=int(search.get("n_grid", band.get("n_grid", 0)) or 0),
            k_hat=float(k_hat) if np.isfinite(k_hat) else float("nan"),
            k_min=float(band["k_min"]),
            k_max=float(band["k_max"]),
            voxel_size_mm=voxel_size,
            lambda_hat_mm=float(row["lambda_hat_mm"]),
            lambda_min_search_mm=float(band.get("lambda_min_search_mm", np.nan)),
            diameter_ref_mm=diameter_ref_mm,
            margin_bins=int(cfg["boundary_margin_bins"]),
        )
    )
    if np.isfinite(row.get("lambda_over_diameter", np.nan)):
        row["lambda_over_radius_gt4_boundary"] = float(row["lambda_over_diameter"]) * 2.0
    row["hr_status"] = search.get("hr_status", "")
    row["hr_status_reason"] = search.get("hr_status_reason", "")

    resid_full = fit.get("resid_full")
    resid_tr = fit.get("resid_tr")
    if resid_full is not None and resid_tr is not None:
        rms_full = float(np.sqrt(np.mean(np.asarray(resid_full) ** 2)))
        rms_tr = float(np.sqrt(np.mean(np.asarray(resid_tr) ** 2)))
        row["rms_residual_full_mm"] = rms_full * voxel_size
        row["rms_residual_tr_mm"] = rms_tr * voxel_size
        A = search.get("A_comb", np.nan)
        if np.isfinite(A) and rms_full > 0:
            row["SNR"] = (A / np.sqrt(2.0)) / rms_full
            row["snr_status"] = "ok"
        elif np.isfinite(A) and rms_full == 0:
            row["SNR"] = float("inf")
            row["snr_status"] = "undefined_zero_residual"
        else:
            row["SNR"] = np.nan
            row["snr_status"] = "undefined"

    svd_stats, g1 = svd_coef_qc(search, voxel_size)
    row.update(svd_stats)
                                               
    row["amp_major_mm"] = row.get("sigma1_mm", np.nan)
    row["amp_minor_mm"] = row.get("sigma2_mm", np.nan)
    A_comb = search.get("A_comb", np.nan)
    row["amp_comb_over_L"] = A_comb / L_trim_vox if np.isfinite(A_comb) and L_trim_vox > 0 else np.nan
    if np.isfinite(radius_median_vox) and radius_median_vox > 0:
        row["lambda_hat_over_radius"] = lam / radius_median_vox if np.isfinite(lam) else np.nan
        row["amp_comb_over_r"] = A_comb / radius_median_vox if np.isfinite(A_comb) else np.nan

                                                                      
    row["lambda_baseline_eff_mm"] = row["lambda_hat_mm"]                  
    row["lambda_baseline_mm"] = row["lambda_hat_mm"]
    row["wavelength_conversion_status"] = "not_applicable"
    row["wavelength_conversion_reason"] = "primary lambda is direct baseline-phase fit; no L_base/L_trim post-hoc"

    # slenderness = λ_hat / (2 r_trim)
    lam_mm = row.get("lambda_hat_mm", np.nan)
    r_mm = row.get("radius_median_mm", np.nan)
    if np.isfinite(lam_mm) and np.isfinite(r_mm) and float(r_mm) > 0:
        row["slenderness"] = float(lam_mm) / (2.0 * float(r_mm))
    else:
        row["slenderness"] = np.nan

                                   
    patch = compute_patch_radius_vox(
        radius_median_vox,
        patch_r_ratio=float(cfg["patch_r_ratio"]),
        r_min_mm=float(cfg["r_min_mm"]),
        r_max_mm=float(cfg["r_max_mm"]),
        voxel_size=voxel_size,
    )
    row["patch_r_ratio"] = patch.get("patch_r_ratio", np.nan)
    row["r_patch_definition"] = patch.get("r_patch_definition", "")
    row["r_patch_mm_raw"] = patch.get("r_patch_mm_raw", np.nan)
    row["r_patch_mm"] = patch.get("r_patch_mm", np.nan)
    row["r_patch_vox"] = patch.get("r_patch_vox", np.nan)
    result["patch"] = patch

                     
    pw: Dict[str, Any] = {
        "frame_reference": fr,
        "phase_coordinate": "baseline_arclength",
        "trend_coordinate": "baseline_arclength",
        "index_full_to_trim": tb["index"],
        "s_obs_full": s_vox,
        "s_base_full": s_base_full,
        "s_phase_full": s_phase_full,
        "s_full": s_vox,  # alias: observed
        "q_full": q_full,
        "P_full": P,
        "P0_full": P0,
        "point_id_full": point_ids,
        "U_full": U,
        "V_full": V,
        "T_full": T,
        "P": tb["P"],
        "P0": tb["P0"],
        "point_id": tb["point_id"],
        "s_obs_trim": s_obs_trim,
        "s_phase_trim": s_phase_trim,
        "s_trend_trim": s_phase_trim,  # linear trend on baseline phase
        "s_fit": s_phase_trim,  # primary phase (baseline)
        "T": tb["T"],
        "U": tb["U"],
        "V": tb["V"],
        "Delta": tb["Delta"],
        "Delta_parallel": tb["Delta_parallel"],
        "d_perp": tb["d_perp"],
        "uU": tb["uU"],
        "uV": tb["uV"],
        "T0": tb["T0"],
        "T_obs": tb["T_obs"],
        "T_obs_frenet": tb.get("T_obs_frenet"),
    }
    ok_fit = row["fit_status"] == "ok" and all(
        np.isfinite(search.get(k, np.nan)) for k in ("k_hat", "uU_cos", "uU_sin", "uV_cos", "uV_sin")
    )
    if ok_fit:
        resp_U = np.asarray(fit["response_U"], float)
        resp_V = np.asarray(fit["response_V"], float)
        trend_U = np.asarray(fit["trend_U"], float)
        trend_V = np.asarray(fit["trend_V"], float)
        lat_U = np.asarray(fit["latent_U"], float)
        lat_V = np.asarray(fit["latent_V"], float)
        yU = np.asarray(fit["yhat_U"], float)
        yV = np.asarray(fit["yhat_V"], float)
        d_latent = lat_U[:, None] * tb["U"] + lat_V[:, None] * tb["V"]
        d_response = resp_U[:, None] * tb["U"] + resp_V[:, None] * tb["V"]
        pw.update(
            fit_U=yU,
            fit_V=yV,
            trend_U=trend_U,
            trend_V=trend_V,
            response_U=resp_U,
            response_V=resp_V,
            latent_U=lat_U,
            latent_V=lat_V,
            # legacy aliases: sin_* = latent (pre-detrending periodic)
            sin_U=lat_U,
            sin_V=lat_V,
            d_sin=d_latent,
            d_latent=d_latent,
            d_response=d_response,
            g1_3d=_normalize_rows(g1[0] * tb["U"] + g1[1] * tb["V"]),
            g1_uv=g1,
            coef=np.asarray(fit.get("coef"), float),
        )
        result["surface_displacement"] = d_latent
        result["surface_displacement_representation"] = "latent_pre_detrending_periodic"
    result["pointwise"] = pw
    result["candidates"] = candidate_table(search, voxel_size, ids, fr)
    result["ok_for_surface"] = bool(ok_fit)
    if not ok_fit:
        row["surface_evaluation_status"] = "skipped_invalid_fit"
        row["surface_evaluation_reason"] = f"fit_status={row['fit_status']}"
    result["row"] = apply_classification(row)
    return result


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


# ---------------------------------------------------------------------------
                                         
# ---------------------------------------------------------------------------
def build_groupid_map(
    group_ids: Sequence[Any],
    pointids: Sequence[Any],
    points: Optional[np.ndarray] = None,
) -> Tuple[Dict[Any, Any], Dict[Any, int], pd.DataFrame]:
    groups: Dict[Any, List[int]] = {}
    for i, gid in enumerate(group_ids):
        groups.setdefault(gid, []).append(i)
    mapping: Dict[Any, Any] = {}
    cell_index: Dict[Any, int] = {}
    diag_rows: List[Dict[str, Any]] = []
    for gid, idxs in groups.items():
        if len(idxs) == 1:
            mapping[gid] = pointids[idxs[0]]
            cell_index[gid] = int(idxs[0])
            continue
        shapes = []
        for i in idxs:
            pid = np.asarray(pointids[i])
            if points is not None:
                shapes.append(np.asarray(points[pid], float))
            else:
                shapes.append(pid.astype(float))
        same = all(s.shape == shapes[0].shape and np.allclose(s, shapes[0]) for s in shapes[1:])
        diag_rows.append(
            {
                "group_id": gid,
                "n_cells": len(idxs),
                "cell_indices": "|".join(map(str, idxs)),
                "shape_identical": bool(same),
                "resolution": "collapsed_identical_duplicates" if same else "ambiguous_excluded",
            }
        )
        if same:
            mapping[gid] = pointids[idxs[0]]
            cell_index[gid] = int(idxs[0])
    cols = ["group_id", "n_cells", "cell_indices", "shape_identical", "resolution"]
    return mapping, cell_index, pd.DataFrame(diag_rows, columns=cols)
