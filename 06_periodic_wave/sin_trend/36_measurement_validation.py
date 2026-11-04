#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 36 measurement validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from sin_trend_classify import classify_row  # noqa: E402
from sin_trend_measure import default_measurement_config, measure_interval  # noqa: E402
from sin_trend_paths import (  # noqa: E402
    BAND_GEOMETRY_2D,
    BAND_LEGACY_FIXED_10MM,
    FRAME_BASELINE,
    atomic_to_csv,
    atomic_write_json,
    metadata_dict,
    new_run_id,
    project_root_from_here,
    sha256_json,
    source_code_hashes,
    validation_root,
)
from sin_trend_synthetic import (  # noqa: E402
    BASE_PLANE_CONDITIONS,
    CURVED_BASELINE_STRESS,
    CURVED_SHORT_STRESS,
    SHORT_COVERAGE_STRESS,
    VALIDATION_DIAMETERS_MM,
    expand_grid_conditions,
    make_sinusoid_centerline,
    random_rotation,
    stress_validation_conditions,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic validation (frame_reference=baseline only)")
    p.add_argument("--run-id", default=None, help="共有 run_id（省略時は validation 専用の新 run_id）")
    p.add_argument("--out-dir", type=Path, default=None, help="出力 directory（既定: run の synthetic_validation/）")
    p.add_argument(
        "--n-cycles",
        nargs="+",
        type=float,
        default=None,
        help="基本平面の trim 後目標残存周期（既定 3.0; 例: 1.5 2 3 4）",
    )
    p.add_argument("--extra", action="store_true", help="背景曲線・面外波・ノイズ・剛体回転の任意検証を追加")
    p.add_argument(
        "--curved-baseline",
        action="store_true",
        help="曲がった二次 baseline に周期偏位を載せた条件を追加（または --stress-only で単独）",
    )
    p.add_argument(
        "--short-coverage",
        action="store_true",
        help="trim 後に約 1.5〜2 周期残る条件を追加",
    )
    p.add_argument(
        "--stress",
        action="store_true",
        help="--curved-baseline + --short-coverage + 曲がり×短周期の交差をまとめて追加",
    )
    p.add_argument(
        "--stress-only",
        action="store_true",
        help="基本平面を走らず stress 系のみ（--stress / --curved-baseline / --short-coverage と併用）",
    )
    p.add_argument("--q-fine", type=int, default=None, help="診断用の波数 grid 細分（既定は臨床既定 8 を変更しない）")
    p.add_argument("--search-band-mode", default=BAND_GEOMETRY_2D, choices=[BAND_GEOMETRY_2D, BAND_LEGACY_FIXED_10MM])
    p.add_argument(
        "--diameter-mm",
        type=float,
        nargs="*",
        default=None,
        help=f"人工血管径 D [mm]（省略時 {VALIDATION_DIAMETERS_MM}）",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="中心線プロット（original vs estimated baseline）を保存しない",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _safe_plot_stem(cond: Dict[str, Any], index: int) -> str:
    variant = str(cond.get("variant", "cond")).replace("/", "_")
    A = float(cond.get("amplitude_mm", float("nan")))
    lam = float(cond.get("lambda_x_mm", float("nan")))
    ph = float(cond.get("phase", float("nan")))
    nc = float(cond.get("n_cycles", float("nan")))
    D = float(cond.get("diameter_mm", float("nan")))
    return f"{index:03d}_{variant}_D{D:g}_A{A:g}_lam{lam:g}_ph{ph:.3f}_nc{nc:g}"


def plot_original_and_baseline_centerlines(
    *,
    P_mm: np.ndarray,
    P0_mm: np.ndarray,
    out_path: Path,
    title: str,
    B_known_mm: Optional[np.ndarray] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    P = np.asarray(P_mm, float)
    P0 = np.asarray(P0_mm, float)
    Bk = np.asarray(B_known_mm, float) if B_known_mm is not None else None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8))
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    ax3.plot(P[:, 0], P[:, 1], P[:, 2], color="C0", lw=1.4, label="original P")
    ax3.plot(P0[:, 0], P0[:, 1], P0[:, 2], color="C3", lw=1.4, label="baseline P0")
    if Bk is not None and Bk.ndim == 2 and Bk.shape[1] == 3:
        ax3.plot(Bk[:, 0], Bk[:, 1], Bk[:, 2], color="0.35", ls="--", lw=1.0, label="known baseline B")
    ax3.set_xlabel("x [mm]")
    ax3.set_ylabel("y [mm]")
    ax3.set_zlabel("z [mm]")
    ax3.set_title("3D")
    ax3.legend(loc="upper left", fontsize=8)

    for ax, i, j, xlab, ylab, ttl in (
        (ax_xy, 0, 1, "x [mm]", "y [mm]", "XY"),
        (ax_xz, 0, 2, "x [mm]", "z [mm]", "XZ"),
        (ax_yz, 1, 2, "y [mm]", "z [mm]", "YZ"),
    ):
        ax.plot(P[:, i], P[:, j], color="C0", lw=1.4, label="original P")
        ax.plot(P0[:, i], P0[:, j], color="C3", lw=1.4, label="baseline P0")
        if Bk is not None and Bk.ndim == 2 and Bk.shape[1] == 3:
            ax.plot(Bk[:, i], Bk[:, j], color="0.35", ls="--", lw=1.0, label="known B")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    ax_xy.legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _conditions(args: argparse.Namespace) -> List[Dict[str, Any]]:
    suite = BASE_PLANE_CONDITIONS
                                                   
    ds_fixed = float(suite["target_ds_mm"][0])
    nc_list = list(args.n_cycles) if args.n_cycles else list(suite["n_cycles_target_after_trim"])
    if args.diameter_mm is not None and len(args.diameter_mm) > 0:
        D_sel: Any = list(args.diameter_mm)
    else:
        D_sel = None                                               
    want_curved = bool(args.curved_baseline or args.stress)
    want_short = bool(args.short_coverage or args.stress)
    want_cross = bool(args.stress)
    stress_only = bool(args.stress_only)
    if stress_only and not (want_curved or want_short or want_cross):
        want_curved = want_short = want_cross = True

    conds: List[Dict[str, Any]] = []
    if not stress_only:
        base_spec = dict(suite)
        base_spec["n_cycles_target_after_trim"] = nc_list
        base_spec.pop("n_cycles", None)
        if "variant" not in base_spec:
            base_spec["variant"] = "pure_sinusoid"
        conds.extend(expand_grid_conditions(base_spec, diameter_mm=D_sel))
    if args.extra and not stress_only:
        D_extra = float(D_sel[0]) if isinstance(D_sel, list) else (
            float(D_sel) if D_sel is not None else float(VALIDATION_DIAMETERS_MM[0])
        )
        base = dict(
            amplitude_mm=1.0,
            lambda_x_mm=20.0,
            n_cycles=3.0,
            phase=0.0,
            target_ds_mm=ds_fixed,
            diameter_mm=D_extra,
        )
        conds.append({**base, "variant": "background_quadratic", "background": {"type": "quadratic", "curvature_mm": 0.005}})
        conds.append({**base, "variant": "out_of_plane_wave", "out_of_plane_amplitude_mm": 0.5})
        conds.append({**base, "variant": "noise_0.1mm", "noise_sd_mm": 0.1, "seed": 1})
        conds.append({**base, "variant": "rigid_rotation", "rotation": random_rotation(3), "translation_mm": np.array([10.0, -5.0, 3.0])})
        conds.append({**base, "variant": "aperiodic_quadratic_only", "amplitude_mm": 0.0, "background": {"type": "quadratic", "curvature_mm": 0.01}})
    if want_curved and want_short and want_cross:
        conds.extend(stress_validation_conditions(diameter_mm=D_sel))
    else:
        if want_curved:
            conds.extend(expand_grid_conditions(CURVED_BASELINE_STRESS, diameter_mm=D_sel))
        if want_short:
            conds.extend(expand_grid_conditions(SHORT_COVERAGE_STRESS, diameter_mm=D_sel))
        if want_cross:
            conds.extend(expand_grid_conditions(CURVED_SHORT_STRESS, diameter_mm=D_sel))
    return conds


def run_condition(
    cond: Dict[str, Any],
    cfg: Dict[str, Any],
    verbose: bool,
    *,
    plot_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    gen_keys = {
        "amplitude_mm",
        "lambda_x_mm",
        "n_cycles",
        "phase",
        "target_ds_mm",
        "diameter_mm",
        "background",
        "out_of_plane_amplitude_mm",
        "noise_sd_mm",
        "rotation",
        "translation_mm",
        "seed",
        "dense_dx_mm",
        "voxel_size_mm",
    }
    gen_kwargs = {k: v for k, v in cond.items() if k in gen_keys}
    voxel_size = float(cfg.get("voxel_size", 0.5))
    syn = make_sinusoid_centerline(
        voxel_size_mm=voxel_size,
        **{k: v for k, v in gen_kwargs.items() if k != "voxel_size_mm"},
    )
    fr = FRAME_BASELINE
    res = measure_interval(
        syn["P_vox"],
        frame_reference=fr,
        cfg=cfg,
        observed_tangent=None,
        radius_vox=syn.get("radius_vox"),
        ids={"directory_id": "SYNTHETIC", "segment_id": 0, "segment_index": 0},
        verbose=verbose,
        compute_pca=False,
    )
    if plot_path is not None:
        bl = res.get("baseline") or {}
        P0 = bl.get("P0")
        if P0 is not None:
            P_mm = np.asarray(syn["P_vox"], float) * voxel_size
            P0_mm = np.asarray(P0, float) * voxel_size
            B_mm = None
            if syn.get("B_vox") is not None:
                B_mm = np.asarray(syn["B_vox"], float) * voxel_size
            title = (
                f"{cond.get('variant')}  D={cond.get('diameter_mm')}  "
                f"A={cond.get('amplitude_mm')}  "
                f"λ={cond.get('lambda_x_mm')}  φ={cond.get('phase'):.3f}  "
                f"n_cycles={cond.get('n_cycles')}"
            )
            try:
                plot_original_and_baseline_centerlines(
                    P_mm=P_mm,
                    P0_mm=P0_mm,
                    out_path=plot_path,
                    title=title,
                    B_known_mm=B_mm,
                )
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"centerline plot failed: {exc}")
    row = classify_row(res["row"])
    A_hat = row["amp_comb_mm"]
    lam_hat = row["lambda_hat_mm"]
    A_true = syn["A_true_mm"]
    lx = syn["lambda_x_true_mm"]
    if "lambda_s_polyline_mm" not in syn:
        raise KeyError(
            "make_sinusoid_centerline must provide lambda_s_polyline_mm "
            "(old lambda_s_true_mm was removed; regenerate synthetic outputs)"
        )
    ls = syn["lambda_s_polyline_mm"]
    l_base = syn.get(
        "lambda_baseline_true_mm",
        syn.get("lambda_phase_true_mm", lx),         
    )
    L_trim = row["L_trimmed_mm"]
    D = syn.get("diameter_mm", np.nan)
    two_D = 2.0 * D if np.isfinite(D) else np.nan
    lam_min = row.get("lambda_min_search_mm")

                                                  
    removal = int(cfg.get("removal", 10))
    xi = np.asarray(syn.get("xi_mm"), float) if syn.get("xi_mm") is not None else None
    n_cycles_remaining_true = float("nan")
    if xi is not None and len(xi) > 2 * removal and np.isfinite(l_base) and l_base > 0:
        xi_a = float(xi[removal])
        xi_b = float(xi[-removal - 1])
        n_cycles_remaining_true = (xi_b - xi_a) / float(l_base)

    def _vs_2d_pos(lam: float) -> str:
        if not (np.isfinite(two_D) and two_D > 0 and np.isfinite(lam)):
            return ""
        if np.isclose(lam, two_D, rtol=0, atol=1e-4):
            return "at_boundary"
        if lam < two_D:
            return "outside_band"
        return "inside_band"

    err_vs_baseline = (
        100.0 * (lam_hat - l_base) / l_base
        if np.isfinite(lam_hat) and np.isfinite(l_base) and l_base > 0
        else np.nan
    )
    out = [
            {
                "variant": cond.get("variant", ""),
                "frame_reference": fr,
                "search_band_mode": row.get("search_band_mode"),
                "diameter_mm_input": D,
                "diameter_ref_mm": row.get("diameter_ref_mm"),
                "two_D_mm": two_D,
                "lambda_min_search_mm": lam_min,
                "lambda_over_diameter": row.get("lambda_over_diameter"),
                "lambda_x_over_2D_true": (lx / two_D) if np.isfinite(two_D) and two_D > 0 and np.isfinite(lx) else np.nan,
                "lambda_x_vs_2D_position": _vs_2d_pos(lx),
                "lambda_s_polyline_over_diameter": syn.get("lambda_s_polyline_over_diameter"),
                "lambda_s_polyline_over_2D": (ls / two_D) if np.isfinite(two_D) and two_D > 0 else np.nan,
                "lambda_s_polyline_status": syn.get("lambda_s_polyline_status", ""),
                                                      
                "lambda_s_polyline_vs_2D_position": _vs_2d_pos(ls),
                "tangent_source": row["tangent_source"],
                "frame_transport_method": row["frame_transport_method"],
                "A_y_true_mm": syn.get("A_y_true_mm"),
                "A_z_true_mm": syn.get("A_z_true_mm"),
                "A_comb_true_mm": syn.get("A_comb_true_mm"),
                "A_true_mm": A_true,
                "lambda_x_true_mm": lx,
                "lambda_baseline_true_mm": l_base,                          
                "lambda_baseline_true_basis": syn.get(
                    "lambda_baseline_true_basis",
                    syn.get("lambda_phase_true_basis", ""),
                ),
                "lambda_phase_true_mm": l_base,            
                "lambda_phase_true_basis": syn.get(
                    "lambda_baseline_true_basis",
                    syn.get("lambda_phase_true_basis", ""),
                ),
                "lambda_s_polyline_mm": ls,
                "n_cycles_input": syn["n_cycles_input"],
                "n_cycles_target_after_trim": cond.get("n_cycles_target_after_trim", np.nan),
                "phase_rad": syn["phase_rad"],
                "phase_coordinate_generation": syn.get("phase_coordinate_generation", ""),
                "target_ds_mm": syn["target_ds_mm"],
                "actual_ds_mm": syn["actual_ds_mm"],
                "L_input_mm": syn["L_input_mm"],
                "L0_known_baseline_mm": syn.get("L0_known_baseline_mm"),
                "x_max_mm": syn.get("x_max_mm"),
                "pts": row["pts"],
                "pts_trimmed": row["pts_trimmed"],
                "removed_mm_start": row["removed_mm_start"],
                "removed_mm_end": row["removed_mm_end"],
                "L_full_mm": row.get("L_full_mm"),
                "L_trimmed_mm": L_trim,
                "L_base_mm": row.get("L_base_mm"),
                "phase_coordinate": row.get("phase_coordinate"),
                "n_cycles_observed_trimmed_vs_lambda_s": (L_trim / ls) if np.isfinite(L_trim) and ls > 0 else np.nan,
                                                              
                "n_cycles_remaining_true": n_cycles_remaining_true,
                "n_cycles_remaining_vs_baseline_true": (
                    float(row.get("L_base_mm")) / float(l_base)
                    if np.isfinite(row.get("L_base_mm", np.nan)) and np.isfinite(l_base) and l_base > 0
                    else np.nan
                ),
                "n_cycles_remaining_vs_lambda_hat": (
                    float(row.get("L_base_mm")) / float(lam_hat)
                    if np.isfinite(row.get("L_base_mm", np.nan)) and np.isfinite(lam_hat) and lam_hat > 0
                    else np.nan
                ),
                "background_type": (cond.get("background") or {}).get("type", ""),
                "background_curvature_mm": (cond.get("background") or {}).get("curvature_mm", np.nan),
                "coverage_hat": row["coverage_hat"],
                "fit_status": row["fit_status"],
                "fit_status_reason": row["fit_status_reason"],
                "fit_valid": row.get("fit_valid"),
                "interval_eligible": row.get("interval_eligible"),
                "good_fit_r2_060": row.get("good_fit_r2_060"),                     
                "core_all": row.get("core_all"),
                "core_periodic": row.get("core_periodic"),
                "reject_reasons": row.get("reject_reasons"),
                "boundary_hit_high_2bins": row.get("boundary_hit_high_2bins"),
                "boundary_exact_high": row.get("boundary_exact_high"),
                "baseline_status": row["baseline_status"],
                "frame_status": row["frame_status"],
                "A_hat_mm": A_hat,
                "amp_comb_mm": row.get("amp_comb_mm", A_hat),
                "amp_major_mm": row.get("amp_major_mm"),
                "amp_minor_mm": row.get("amp_minor_mm"),
                "A_error_pct": 100.0 * (A_hat - A_true) / A_true if A_true > 0 and np.isfinite(A_hat) else np.nan,
                "A_error_mm": A_hat - A_true if np.isfinite(A_hat) else np.nan,
                "lambda_hat_mm": row.get("lambda_hat_mm", lam_hat),
                "lambda_hat_s_mm": lam_hat,  # alias
                "lambda_baseline_eff_mm": row.get("lambda_baseline_eff_mm"),
                "L_base_over_L_obs": row.get("L_base_over_L_obs"),
                "wavelength_conversion_status": row.get("wavelength_conversion_status"),
                "wavelength_conversion_reason": row.get("wavelength_conversion_reason"),
                                                         
                "lambda_error_vs_baseline_pct": err_vs_baseline,
                "lambda_error_vs_phase_pct": err_vs_baseline,            
                "lambda_diff_vs_x_pct": 100.0 * (lam_hat - lx) / lx if np.isfinite(lam_hat) and lx > 0 else np.nan,
                "lambda_hat_vs_x_pct": 100.0 * (lam_hat - lx) / lx if np.isfinite(lam_hat) and lx > 0 else np.nan,
                "lambda_error_vs_s_pct": 100.0 * (lam_hat - ls) / ls if np.isfinite(lam_hat) and ls > 0 else np.nan,
                "lambda_error_vs_s_is_diagnostic_only": 1,
                "R2": row["R2"],
                "partial_R2": row["partial_R2"],
                "R2_trend_only": row.get("R2_trend_only"),
                "SNR": row["SNR"],
                "hr_amp": row["hr_amp"],
                "boundary_hit_low": row["boundary_hit_low"],
                "boundary_hit_high": row["boundary_hit_high"],
                "sigma1_mm": row["sigma1_mm"],
                "sigma2_mm": row["sigma2_mm"],
                "sigma1_true_mm": syn.get("A_comb_true_mm", A_true),
                "sigma2_true_mm": 0.0 if np.isfinite(syn.get("A_comb_true_mm", A_true)) else np.nan,
                "deltaP_max_mm": row["deltaP_max_mm"],
                "resid_parallel_over_perp": row["resid_parallel_over_perp"],
                "angle_T0_Tobs_mean_deg": row["angle_T0_Tobs_mean_deg"],
                "frame_orth_err_max": row["frame_orth_err_max"],
                "fit_method": row.get("fit_method"),
                "q_fine": cfg["q_fine"],
            }
    ]
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in df.groupby("variant"):
        ok = g[g["fit_status"] == "ok"]
        if "core_all" in ok.columns:
            core = ok[pd.to_numeric(ok["core_all"], errors="coerce").fillna(0).astype(int) == 1]
        else:
            core = ok.iloc[0:0]
        n_core = int(len(core))

        def _abs_med(series: pd.Series) -> float:
            return float(series.abs().median()) if len(series) else float("nan")

        def _abs_max(series: pd.Series) -> float:
            return float(series.abs().max()) if len(series) else float("nan")

        rows.append(
            {
                "variant": variant,
                "frame_reference": FRAME_BASELINE,
                         
                "n_conditions": len(g),
                "n_fit_ok": len(ok),
                "n_core_all": n_core,
                "n_core_reject": int(len(ok) - n_core),
                                       
                "A_error_pct_abs_median_core": _abs_med(core["A_error_pct"])
                if "A_error_pct" in core.columns
                else np.nan,
                "A_error_pct_abs_max_core": _abs_max(core["A_error_pct"])
                if "A_error_pct" in core.columns
                else np.nan,
                "lambda_error_vs_baseline_pct_abs_median_core": _abs_med(
                    core["lambda_error_vs_baseline_pct"]
                )
                if "lambda_error_vs_baseline_pct" in core.columns
                else np.nan,
                "lambda_error_vs_baseline_pct_abs_max_core": _abs_max(
                    core["lambda_error_vs_baseline_pct"]
                )
                if "lambda_error_vs_baseline_pct" in core.columns
                else np.nan,
                "R2_min_core": float(core["R2"].min()) if len(core) and "R2" in core.columns else np.nan,
                "coverage_hat_median_core": float(core["coverage_hat"].median())
                if len(core) and "coverage_hat" in core.columns
                else np.nan,
                                
                "n_cycles_remaining_true_median": ok["n_cycles_remaining_true"].median()
                if len(ok) and "n_cycles_remaining_true" in ok.columns
                else np.nan,
                "n_cycles_remaining_true_min": ok["n_cycles_remaining_true"].min()
                if len(ok) and "n_cycles_remaining_true" in ok.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root_from_here(__file__)
    run_id = args.run_id or new_run_id("validation")
    out_dir = Path(args.out_dir) if args.out_dir else validation_root(run_id, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = default_measurement_config()
    cfg["verbose"] = False
    cfg["search_band_mode"] = args.search_band_mode
    if args.q_fine is not None:
        cfg["q_fine"] = int(args.q_fine)                        
    conds = _conditions(args)
    if not args.quiet:
        print(
            f"run_id={run_id} conditions={len(conds)} frame_reference={FRAME_BASELINE} "
            f"band={cfg['search_band_mode']} q_fine={cfg['q_fine']} -> {out_dir}"
        )

    plot_dir = None if args.no_plots else (out_dir / "centerline_plots")
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for i, cond in enumerate(conds, 1):
        if not args.quiet:
            print(
                f"[{i}/{len(conds)}] {cond.get('variant')}: D={cond.get('diameter_mm')}, "
                f"A={cond['amplitude_mm']}, "
                f"lambda_x={float(cond['lambda_x_mm']):.4f}, "
                f"phase={cond['phase']:.3f}, n_cycles={cond['n_cycles']}, ds={cond['target_ds_mm']}"
            )
        plot_path = None
        if plot_dir is not None:
            plot_path = plot_dir / f"{_safe_plot_stem(cond, i)}.png"
        try:
            rows.extend(run_condition(cond, cfg, verbose=False, plot_path=plot_path))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "variant": cond.get("variant"),
                    "frame_reference": FRAME_BASELINE,
                    "fit_status": "exception",
                    "fit_status_reason": str(exc),
                    **{k: cond.get(k) for k in ("amplitude_mm", "lambda_x_mm", "phase", "n_cycles", "target_ds_mm")},
                }
            )

    df = pd.DataFrame(rows)
    atomic_to_csv(df, out_dir / "synthetic_validation_results.csv", index=False)
    summ = summarize(df) if len(df) else pd.DataFrame()
    atomic_to_csv(summ, out_dir / "synthetic_validation_summary.csv", index=False)

    meta = metadata_dict(
        run_id=run_id,
        config=cfg,
        config_hash=sha256_json(cfg),
        source_code_hashes=source_code_hashes(),
        n_conditions=len(conds),
        frame_reference=FRAME_BASELINE,
        conditions_suite=(
            "stress_only"
            if args.stress_only
            else (
                "base_plane_plus_stress"
                if (args.stress or args.curved_baseline or args.short_coverage)
                else "base_plane"
            )
        ),
        diameters_mm=sorted(
            {
                float(c["diameter_mm"])
                for c in conds
                if c.get("diameter_mm") is not None and np.isfinite(float(c["diameter_mm"]))
            }
        ),
        base_plane_note=(
            "pure_sinusoid 72 (A∈{0.25,0.5,1,2}×λ∈{10,20,30,40,50,60}×3 phases), "
            "curved 54, short 24 (λ∈{10,20,40}), curved_short 6. D=1.4, ds=0.1 (fixed). "
            "Plane/curved/short set n_cycles_target_after_trim (plane/curved: 3; short: 1.5/2); "
            "input n_cycles is solved so true remaining cycles after resample+removal=10 match the target."
        ),
        stress_note=(
            "Generation phase is sin(2π ξ/λ+φ) on known baseline arclength ξ (code: xi_mm); "
            "lambda_baseline_true=λ (constant). Amplitude is perpendicular offset to known baseline. "
            "Cycle counts are post-trim targets matched via ξ. x_max is the Cartesian x-span of B. "
            "Flags: --stress / --curved-baseline / --short-coverage / --stress-only."
        ),
        centerline_plots=(
            None
            if args.no_plots
            else "centerline_plots/: original P vs estimated quadratic baseline P0 (known B dashed when present)"
        ),
    )
    atomic_write_json(meta, out_dir / "synthetic_validation_metadata.json")
    if not args.quiet:
        print(summ.to_string(index=False) if len(summ) else "(no rows)")
        print(f"saved: {out_dir}")
        if plot_dir is not None:
            print(f"centerline plots: {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
