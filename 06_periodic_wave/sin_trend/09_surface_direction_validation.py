#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 09 surface direction validation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_classify import (  # noqa: E402
    SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL,
    SURFACE_LABEL_SUBSTANTIAL_NORMAL,
)
from sin_trend_io_utils import atomic_to_csv  # noqa: E402
from sin_trend_measure import default_measurement_config, measure_interval  # noqa: E402
from sin_trend_paths import FRAME_BASELINE, MEASUREMENT_REVISION, SURFACE_METHOD_REVISION  # noqa: E402
from sin_trend_surface_direction_validation import (  # noqa: E402
    TOL_GEOM_ABS,
    TOL_NORMAL_AXIS_DEG,
    TOL_PLANE_DIRECT_ABS,
    TOL_PLANE_FIT_ABS,
    VAL_R_OUT_THRESH,
    VAL_R_TAN_THRESH,
    apply_rigid,
    centerline_along_x,
    centerline_follow_valley_surface,
    classify_direction,
    constant_unit_tangent,
    geometric_isotropic_R_tan_mean,
    known_periodic_displacement,
    known_periodic_displacement_perp_T,
    make_plane_tri_mesh,
    make_valley_tri_mesh,
    metrics_vs_truth,
    random_rotation,
    run_surface_on_arrays,
    summarize_row,
    surface_height_mm,
    theoretical_R,
    unit_tangents_from_polyline,
    validation_config,
)
from sin_trend_synthetic import resample_uniform_arclength_with_attrs  # noqa: E402


def _shlex_join(parts: List[str]) -> str:
    try:
        import shlex

        return shlex.join(parts)
    except Exception:
        return " ".join(f'"{p}"' if (" " in p or "\t" in p) else p for p in parts)


def expected_label_for_theta(theta_plane_deg: float) -> str:
                                                
    th = float(theta_plane_deg)
    if abs(th - 30.0) < 1e-9:
        return SURFACE_LABEL_SUBSTANTIAL_NORMAL
    R_out_t, _ = theoretical_R(th)
    if R_out_t < VAL_R_OUT_THRESH:
        return SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL
    return SURFACE_LABEL_SUBSTANTIAL_NORMAL


def pass_plane_direct_row(met: Dict[str, Any], cls: Dict[str, Any], theta: float) -> Dict[str, Any]:
    err = met.get("err_R_out_vs_theory")
    pass_R = bool(np.isfinite(err) and abs(float(err)) <= TOL_PLANE_DIRECT_ABS)
    axis = met.get("normal_axis_err_deg_median")
    pass_axis = bool(np.isfinite(axis) and float(axis) <= TOL_NORMAL_AXIS_DEG)
    expect = expected_label_for_theta(theta)
    got = cls.get("label")
                                                                 
    boundary30 = abs(float(theta) - 30.0) < 1e-9
    if boundary30:
        pass_class = got in (
            SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL,
            SURFACE_LABEL_SUBSTANTIAL_NORMAL,
        ) and bool(cls.get("evaluable"))
        class_note = "boundary_30deg_float_may_differ_from_exact_0.25_label"
    else:
        pass_class = bool(cls.get("evaluable")) and got == expect
        class_note = ""
    pass_ub = bool(met.get("pass_R_out_le_upper_bound", False))
    return dict(
        expected_direction_label=expect,
        pass_R=pass_R,
        pass_axis=pass_axis,
        pass_class=pass_class,
        class_note=class_note,
        pass_all=bool(pass_R and pass_axis and pass_class and pass_ub),
    )


def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _prod_vs_val_thresh() -> Dict[str, Any]:
    cfg = default_measurement_config()
    prod = float(cfg.get("R_out_thresh", float("nan")))
    return dict(
        production_R_out_thresh=prod,
        validation_R_out_thresh=VAL_R_OUT_THRESH,
        validation_R_tan_thresh=VAL_R_TAN_THRESH,
        differs=bool(np.isfinite(prod) and abs(prod - VAL_R_OUT_THRESH) > 1e-15),
    )


def run_plane_direct(out_dir: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    thetas = [0.0, 29.0, 30.0, 31.0, 45.0, 90.0]
    A_mm, lam_mm, phase = 0.5, 20.0, 0.0
    L_mm, ds_mm, z_off = 60.0, float(cfg["target_ds_mm"]), 0.5
    radius_mm = 0.7
    rows: List[Dict[str, Any]] = []
    mesh = make_plane_tri_mesh(
        x_min_mm=-5.0,
        x_max_mm=L_mm + 5.0,
        y_min_mm=-12.0,
        y_max_mm=12.0,
        edge_mm=0.33,
        z_mm=0.0,
        voxel_size_mm=float(cfg["voxel_size"]),
    )
    cl = centerline_along_x(L_mm=L_mm, ds_mm=ds_mm, y_mm=0.0, z_mm=z_off, voxel_size_mm=float(cfg["voxel_size"]))
    n_true0 = np.array([0.0, 0.0, 1.0])
    T0 = constant_unit_tangent(len(cl["P_vox"]), (1.0, 0.0, 0.0))                     

    for th in thetas:
        d = known_periodic_displacement(
            cl["x_mm"], amplitude_mm=A_mm, lambda_mm=lam_mm, phase=phase, theta_plane_deg=th, voxel_size_mm=float(cfg["voxel_size"])
        )
        res = run_surface_on_arrays(cl["P_vox"], d, mesh, cfg, radius_mm=radius_mm, T0=T0, T_frame=T0)
        if not res["ok"]:
            rows.append(dict(mode="direct", theta_plane_deg=th, ok=False, reason=res.get("reason")))
            continue
        met = metrics_vs_truth(res["ev"], res["est"], d, n_true0, theta_plane_deg=th, T0_for_theory=T0)
        cls = classify_direction(
            R_out=met["R_out"],
            constraint_coverage=met["constraint_coverage"],
            squared_displacement_coverage=met["squared_displacement_coverage"],
            status=met["status"],
            R_out_thresh=float(cfg["R_out_thresh"]),
        )
        flags = pass_plane_direct_row(met, cls, th)
        row = summarize_row(
            dict(
                mode="direct",
                ok=True,
                theta_plane_deg=th,
                amplitude_mm=A_mm,
                lambda_mm=lam_mm,
                phase=phase,
                L_mm=L_mm,
                z_offset_mm=z_off,
                radius_mm=radius_mm,
                rigid_rotation=False,
                tol_abs=TOL_PLANE_DIRECT_ABS,
                **met,
                **flags,
            ),
            cls,
        )
        rows.append(row)

                     
    th = 45.0
    d = known_periodic_displacement(
        cl["x_mm"], amplitude_mm=A_mm, lambda_mm=lam_mm, phase=phase, theta_plane_deg=th, voxel_size_mm=float(cfg["voxel_size"])
    )
    R = random_rotation(7)
    P2, d2, mesh2, T02 = apply_rigid(cl["P_vox"], d, mesh, R, T_vox=T0)
    n_true2 = R @ n_true0
    res = run_surface_on_arrays(P2, d2, mesh2, cfg, radius_mm=radius_mm, T0=T02, T_frame=T02)
    if res["ok"]:
        met = metrics_vs_truth(res["ev"], res["est"], d2, n_true2, theta_plane_deg=th, T0_for_theory=T02)
        cls = classify_direction(
            R_out=met["R_out"],
            constraint_coverage=met["constraint_coverage"],
            squared_displacement_coverage=met["squared_displacement_coverage"],
            status=met["status"],
            R_out_thresh=float(cfg["R_out_thresh"]),
        )
        flags = pass_plane_direct_row(met, cls, th)
        rows.append(
            summarize_row(
                dict(
                    mode="direct_rigid",
                    ok=True,
                    theta_plane_deg=th,
                    amplitude_mm=A_mm,
                    lambda_mm=lam_mm,
                    rigid_rotation=True,
                    tol_abs=TOL_PLANE_DIRECT_ABS,
                    **met,
                    **flags,
                ),
                cls,
            )
        )

    df = pd.DataFrame(rows)
    atomic_to_csv(df, out_dir / "plane_direct_known_displacement.csv", index=False)
    return df


def run_plane_fit_through(out_dir: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    cases = [
        dict(name="inplane_A0.5", A=0.5, Az=0.0, theta_approx=0.0),
        dict(name="theta30_A0.5", A=0.5 * np.cos(np.deg2rad(30)), Az=0.5 * np.sin(np.deg2rad(30)), theta_approx=30.0),
        dict(name="theta90_Az0.5", A=0.0, Az=0.5, theta_approx=90.0),
    ]
    L_mm = 60.0
    vs = float(cfg["voxel_size"])
    ds_mm = float(cfg["target_ds_mm"])
    ds_vox = ds_mm / vs
    mesh = make_plane_tri_mesh(
        x_min_mm=-5.0,
        x_max_mm=L_mm + 5.0,
        y_min_mm=-12.0,
        y_max_mm=12.0,
        edge_mm=0.33,
        z_mm=0.0,
        voxel_size_mm=vs,
    )
    n_true = np.array([0.0, 0.0, 1.0])
    for c in cases:
                                                        
        cl_dense = centerline_along_x(L_mm=L_mm, ds_mm=ds_mm * 0.5, y_mm=0.0, z_mm=0.5, voxel_size_mm=vs)
        x = cl_dense["x_mm"]
        s = np.sin(2.0 * np.pi * x / 20.0)
        d_mm = np.column_stack([np.zeros_like(s), float(c["A"]) * s, float(c["Az"]) * s])
        d_vox_dense = d_mm / vs
        P_obs_dense = cl_dense["P_vox"] + d_vox_dense
        P_obs, attrs = resample_uniform_arclength_with_attrs(
            P_obs_dense,
            ds_vox,
            attrs={
                "d_true_x": d_vox_dense[:, 0],
                "d_true_y": d_vox_dense[:, 1],
                "d_true_z": d_vox_dense[:, 2],
            },
        )
        d_true_res = np.column_stack([attrs["d_true_x"], attrs["d_true_y"], attrs["d_true_z"]])
                                
        if len(P_obs) >= 2:
            seg_mm = np.linalg.norm(np.diff(P_obs, axis=0), axis=1) * vs
            ds_med = float(np.median(seg_mm))
            ds_min, ds_max = float(np.min(seg_mm)), float(np.max(seg_mm))
        else:
            ds_med = ds_min = ds_max = float("nan")
        r_vox = np.full(len(P_obs), 0.7 / vs)
        meas = measure_interval(
            P_obs,
            frame_reference=FRAME_BASELINE,
            cfg=cfg,
            radius_vox=r_vox,
            ids={"directory_id": "SYN_SURF_DIR", "segment_id": 0, "segment_index": 0},
            verbose=False,
            compute_pca=False,
        )
        pw = meas.get("pointwise") or {}
        if pw.get("d_latent") is None or pw.get("P") is None:
            rows.append(
                dict(
                    mode="fit_through",
                    case=c["name"],
                    ok=False,
                    reason="no_pointwise_d_latent",
                    fit_status=meas["row"].get("fit_status"),
                )
            )
            continue
        P_trim = np.asarray(pw["P"], float)
        d_lat = np.asarray(pw["d_latent"], float)
        idx = np.asarray(pw.get("index_full_to_trim", np.arange(len(P_obs))), int)
        d_true_trim = d_true_res[idx]
        T0_trim = np.asarray(pw.get("T0"), float) if pw.get("T0") is not None else None
        T_frame = np.asarray(pw.get("T"), float) if pw.get("T") is not None else T0_trim
        if T0_trim is None or len(T0_trim) != len(P_trim):
            T0_trim = unit_tangents_from_polyline(P_trim)
            T_frame = T0_trim
        res = run_surface_on_arrays(
            P_trim, d_lat, mesh, cfg, radius_mm=0.7, T0=T0_trim, T_frame=T_frame
        )
        if not res["ok"]:
            rows.append(dict(mode="fit_through", case=c["name"], ok=False, reason=res.get("reason")))
            continue
                                          
        met = metrics_vs_truth(
            res["ev"],
            res["est"],
            d_true_trim,
            n_true,
            theta_plane_deg=float(c["theta_approx"]),
            d_est_vox=d_lat,
            T0_for_theory=T0_trim,
        )
        met["pass_R"] = (
            (abs(met["err_R_out_vs_theory"]) <= TOL_PLANE_FIT_ABS)
            if np.isfinite(met["err_R_out_vs_theory"])
            else False
        ) and bool(met.get("pass_R_out_le_upper_bound"))
        cls = classify_direction(
            R_out=met["R_out"],
            constraint_coverage=met["constraint_coverage"],
            squared_displacement_coverage=met["squared_displacement_coverage"],
            status=met["status"],
            R_out_thresh=float(cfg["R_out_thresh"]),
        )
        rows.append(
            summarize_row(
                dict(
                    mode="fit_through",
                    case=c["name"],
                    ok=True,
                    theta_approx_deg=c["theta_approx"],
                    fit_status=meas["row"].get("fit_status"),
                    amp_comb_mm=meas["row"].get("amp_comb_mm"),
                    lambda_hat_mm=meas["row"].get("lambda_hat_mm"),
                    resampled_ds_mm_median=ds_med,
                    resampled_ds_mm_min=ds_min,
                    resampled_ds_mm_max=ds_max,
                    tol_abs=TOL_PLANE_FIT_ABS,
                    **met,
                ),
                cls,
            )
        )
    df = pd.DataFrame(rows)
    atomic_to_csv(df, out_dir / "plane_fit_through.csv", index=False)
    return df


def run_geometric_T(out_dir: Path) -> pd.DataFrame:
    n = np.array([0.0, 0.0, 1.0])
    cases = [
        ("T_parallel_to_plane", np.array([1.0, 0.0, 0.0]), 0.5),
        ("T_45_to_plane", np.array([1.0, 0.0, 1.0]) / np.sqrt(2), 0.75),
        ("T_perp_to_plane", np.array([0.0, 0.0, 1.0]), 1.0),
    ]
    rows = []
    for name, T, expect in cases:
        g = geometric_isotropic_R_tan_mean(T, n)
        rows.append(
            dict(
                case=name,
                ok=True,
                expect_mean_R_tan=expect,
                mean_R_tan_analytic=g["mean_R_tan_analytic"],
                mean_R_tan_numeric=g["mean_R_tan_numeric"],
                T_dot_n=g["T_dot_n"],
                pass_analytic=abs(g["mean_R_tan_analytic"] - expect) <= TOL_GEOM_ABS,
                pass_numeric=abs(g["mean_R_tan_numeric"] - expect) <= 1e-6,
                pass_abs=bool(
                    abs(g["mean_R_tan_analytic"] - expect) <= TOL_GEOM_ABS
                    and abs(g["mean_R_tan_numeric"] - expect) <= 1e-6
                ),
                note="known_normal_isotropic_in_T_perp_plane; not a mesh validation substitute",
            )
        )
    df = pd.DataFrame(rows)
    atomic_to_csv(df, out_dir / "geometric_T_isotropy.csv", index=False)
    return df


def run_valley_bridge(out_dir: Path, cfg: Dict[str, Any]):
    rows: List[Dict[str, Any]] = []
    vs = float(cfg["voxel_size"])
    L_mm, ds_mm = 40.0, float(cfg["target_ds_mm"])
    theta_disp_deg = 30.0
    true_top_n = np.array([0.0, 0.0, 1.0], float)
    mesh_specs = {
                                             
        "short_trench": dict(
            peak1_x_mm=14.0,
            peak2_x_mm=22.0,
            peak_halfwidth_mm=2.0,
            peak_z_mm=3.0,
            mode="flat_plateaus_trench",
            trench_halfwidth_mm=1.5,
            wall_ramp_mm=1.0,
        ),
                                              
        "long_trench": dict(
            peak1_x_mm=10.0,
            peak2_x_mm=30.0,
            peak_halfwidth_mm=2.0,
            peak_z_mm=3.0,
            mode="flat_plateaus_trench",
            trench_halfwidth_mm=5.0,
            wall_ramp_mm=0.0,
        ),
        "gaussian_long": dict(
            peak1_x_mm=12.0,
            peak2_x_mm=28.0,
            peak_halfwidth_mm=2.0,
            peak_z_mm=3.5,
            mode="gaussian_peaks",
        ),
    }
    meshes = {}
    for name, spec in mesh_specs.items():
        meshes[name] = make_valley_tri_mesh(
            x_min_mm=-2.0,
            x_max_mm=L_mm + 2.0,
            y_min_mm=-6.0,
            y_max_mm=6.0,
            edge_mm=0.33,
            voxel_size_mm=vs,
            **spec,
        )

    path_defs = [
        dict(
            path="follow_surface_gaussian",
            mesh="gaussian_long",
            z_mode="follow",
            offset_mm=0.6,
            z_const_mm=None,
            intent="cross_peak_valley_peak_follow_surface",
        ),
        dict(
            path="air_ridge_short_trench",
            mesh="short_trench",
            z_mode="const",
            offset_mm=0.5,
            z_const_mm=3.4,
            intent="expect_possible_short_bridge_between_plateaus",
        ),
        dict(
            path="air_ridge_long_trench",
            mesh="long_trench",
            z_mode="const",
            offset_mm=0.5,
            z_const_mm=3.4,
            intent="expect_gap_too_long_for_5mm_bridge",
        ),
        dict(
                                               
            path="one_sided_into_trench",
            mesh="long_trench",
            z_mode="const",
            offset_mm=0.5,
            z_const_mm=3.4,
            intent="one_sided_good_should_not_fill_end",
            x_start_mm=0.0,
            x_end_mm=18.0,
        ),
    ]

    fig_cache: Dict[str, Any] = {}
    for pd_ in path_defs:
        spec = mesh_specs[pd_["mesh"]]
        x0 = float(pd_.get("x_start_mm", 0.0))
        x1 = float(pd_.get("x_end_mm", L_mm))
        mode = str(spec.get("mode", "gaussian_peaks"))
        cl = centerline_follow_valley_surface(
            L_mm=x1 - x0,
            ds_mm=ds_mm,
            offset_mm=float(pd_["offset_mm"]),
            peak1_x_mm=float(spec["peak1_x_mm"]) - x0,
            peak2_x_mm=float(spec["peak2_x_mm"]) - x0,
            peak_halfwidth_mm=float(spec["peak_halfwidth_mm"]),
            peak_z_mm=float(spec["peak_z_mm"]),
            voxel_size_mm=vs,
            z_mode=pd_["z_mode"],
            z_const_mm=pd_.get("z_const_mm"),
            mode=mode,
            trench_halfwidth_mm=float(spec.get("trench_halfwidth_mm", 2.0)),
            wall_ramp_mm=float(spec.get("wall_ramp_mm", 1.0)),
        )
        cl["P_mm"][:, 0] = cl["P_mm"][:, 0] + x0
        cl["P_vox"] = cl["P_mm"] / vs
        cl["x_mm"] = cl["P_mm"][:, 0]
                                      
        cl["z_surf_mm"] = surface_height_mm(
            cl["x_mm"],
            np.zeros_like(cl["x_mm"]),
            mode=mode,
            peak1_x_mm=float(spec["peak1_x_mm"]),
            peak2_x_mm=float(spec["peak2_x_mm"]),
            peak_halfwidth_mm=float(spec["peak_halfwidth_mm"]),
            peak_z_mm=float(spec["peak_z_mm"]),
            trench_halfwidth_mm=float(spec.get("trench_halfwidth_mm", 2.0)),
            wall_ramp_mm=float(spec.get("wall_ramp_mm", 1.0)),
        )
        cl["mesh_mode"] = mode
        cl["mesh_spec"] = dict(spec)
        T0_v = unit_tangents_from_polyline(cl["P_vox"])
                                                                       
                                                          
        d = known_periodic_displacement_perp_T(
            cl["x_mm"],
            T0_v,
            amplitude_mm=0.4,
            lambda_mm=15.0,
            phase=0.0,
            theta_plane_deg=theta_disp_deg,
            voxel_size_mm=vs,
        )
        mesh = meshes[pd_["mesh"]]
        for gap_mm, axis_deg, tag in (
            (5.0, 45.0, "bridge_cfg_5mm_45deg"),
            (3.0, 45.0, "bridge_cfg_3mm_45deg"),
            (8.0, 45.0, "bridge_cfg_8mm_45deg"),
            (5.0, 30.0, "bridge_cfg_5mm_30deg"),
            (5.0, 60.0, "bridge_cfg_5mm_60deg"),
        ):
            cfg_i = dict(cfg)
            cfg_i["bridge_max_gap_mm"] = float(gap_mm)
            cfg_i["bridge_axis_max_deg"] = float(axis_deg)
            res = run_surface_on_arrays(cl["P_vox"], d, mesh, cfg_i, radius_mm=0.7, T0=T0_v, T_frame=T0_v)
            if not res["ok"]:
                rows.append(
                    dict(
                        path=pd_["path"],
                        mesh_spec=pd_["mesh"],
                        tag=tag,
                        ok=False,
                        reason=res.get("reason"),
                        intent=pd_["intent"],
                        bridge_max_gap_mm=gap_mm,
                        bridge_axis_max_deg=axis_deg,
                        theta_plane_deg=theta_disp_deg,
                        pass_intent=False,
                    )
                )
                continue
            est, ev = res["est"], res["ev"]
            good = np.asarray(est.get("anchor_good"), bool)
            coh = np.asarray(est.get("anchor_coherence"), float)
            P = np.asarray(cl["P_vox"], float)
            if len(P) >= 2:
                s_mm = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1) * vs)])
            else:
                s_mm = np.zeros(len(P))
            aidx = np.asarray(est["anchor_indices"], int)
            s_a = s_mm[aidx]
            gidx = np.flatnonzero(good)
            gaps = []
            axes = []
            n_g = np.asarray(est.get("anchor_normals_good", est["anchor_normals"]), float)
            for k in range(len(gidx) - 1):
                i, j = int(gidx[k]), int(gidx[k + 1])
                if j > i + 1:
                    gaps.append(abs(float(s_a[j] - s_a[i])))
                    axes.append(unsigned_axis_angle_safe(n_g[i], n_g[j]))
                                                   
            top_errs = []
            for i in gidx:
                top_errs.append(unsigned_axis_angle_safe(n_g[int(i)], true_top_n))
            src = np.asarray(est.get("eval_normal_source"), int)
            from sin_trend_fitting_helpers import EVAL_NORMAL_SOURCE_BRIDGED, EVAL_NORMAL_SOURCE_GOOD, EVAL_NORMAL_SOURCE_NONE

            n_src_none = int(np.sum(src == EVAL_NORMAL_SOURCE_NONE)) if src.size else 0
            n_src_good = int(np.sum(src == EVAL_NORMAL_SOURCE_GOOD)) if src.size else 0
            n_src_bridged = int(np.sum(src == EVAL_NORMAL_SOURCE_BRIDGED)) if src.size else 0
                                            
            end_none = bool(src.size and src[-1] == EVAL_NORMAL_SOURCE_NONE)
            mask_truth = np.asarray(est["mask_eval"], bool)
            d2_truth = np.sum(d * d, axis=1)
            normal2_truth = (d @ true_top_n) ** 2
            denom_truth = float(np.sum(d2_truth[mask_truth]))
            denom_full = float(np.sum(d2_truth))
            R_out_t = float(np.sum(normal2_truth[mask_truth]) / denom_truth) if denom_truth > 0 else float("nan")
            R_out_t_full = float(np.sum(normal2_truth) / denom_full) if denom_full > 0 else float("nan")
            R_out_nominal, _ = theoretical_R(theta_disp_deg)
            row = dict(
                path=pd_["path"],
                mesh_spec=pd_["mesh"],
                mesh_mode=mode,
                tag=tag,
                ok=True,
                intent=pd_["intent"],
                note="reference_plane_is_peak_connecting_approx_horizontal_not_sidewall",
                peak1_x_mm=spec["peak1_x_mm"],
                peak2_x_mm=spec["peak2_x_mm"],
                peak_sep_mm=abs(float(spec["peak2_x_mm"]) - float(spec["peak1_x_mm"])),
                wall_ramp_mm=float(spec.get("wall_ramp_mm", 1.0)),
                trench_halfwidth_mm=float(spec.get("trench_halfwidth_mm", np.nan)),
                bridge_max_gap_mm=gap_mm,
                bridge_axis_max_deg=axis_deg,
                theta_plane_deg=theta_disp_deg,
                n_anchors=int(est.get("n_anchors", 0)),
                n_anchors_good=int(est.get("n_anchors_good", 0)),
                frac_anchor_good=float(np.mean(good)) if good.size else float("nan"),
                coh_median=float(np.nanmedian(coh)) if coh.size else float("nan"),
                good_bridge_candidate_gap_mm_min=float(np.min(gaps)) if gaps else float("nan"),
                good_bridge_candidate_gap_mm_max=float(np.max(gaps)) if gaps else float("nan"),
                good_bridge_candidate_axis_deg_max=float(np.nanmax(axes)) if axes else float("nan"),
                n_bridge_candidate_pairs=int(len(gaps)),
                n_eval_good=ev.get("n_eval_good"),
                n_eval_bridged=ev.get("n_eval_bridged"),
                n_src_none=n_src_none,
                n_src_good=n_src_good,
                n_src_bridged=n_src_bridged,
                end_eval_source_none=end_none,
                good_coverage=ev.get("good_coverage"),
                bridged_coverage=ev.get("bridged_coverage"),
                constraint_coverage=ev.get("constraint_coverage"),
                R_out=ev.get("R_out"),
                R_tan=ev.get("R_tan"),
                R_out_theory=R_out_t,
                R_out_truth_full=R_out_t_full,
                R_out_nominal_fixed_direction=R_out_nominal,
                truth_basis="generated_displacement_projected_to_known_top_normal",
                abs_T0_dot_n_mean=ev.get("abs_T0_dot_n_mean"),
                abs_T0_dot_n_median=ev.get("abs_T0_dot_n_median"),
                R_out_upper_bound=ev.get("R_out_upper_bound"),
                R_out_within_upper_bound=ev.get("R_out_within_upper_bound"),
                true_top_n_axis_err_deg_median=float(np.nanmedian(top_errs)) if top_errs else float("nan"),
                true_top_n_axis_err_deg_max=float(np.nanmax(top_errs)) if top_errs else float("nan"),
                status=ev.get("status"),
                forced_mask=False,
            )
            rows.append(row)
            if tag == "bridge_cfg_5mm_45deg" and pd_["path"] not in fig_cache:
                fig_cache[pd_["path"]] = dict(
                    cl=cl, d=d, est=est, ev=ev, mesh=mesh, row=row, mesh_spec=dict(spec), mesh_mode=mode
                )

    df = pd.DataFrame(rows)
    df = _annotate_valley_intent_pass(df)
    atomic_to_csv(df, out_dir / "valley_bridge_observation.csv", index=False)
    return df, fig_cache


def _finite_number(value: Any) -> float:
    """Unknown diagnostics stay unknown; never turn NaN into a valid count/flag."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _true_flag(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating)) and _finite_number(value) == 1.0


def _count_or_none(value: Any) -> Optional[int]:
    number = _finite_number(value)
    return int(number) if np.isfinite(number) and number >= 0 and number.is_integer() else None


def _annotate_valley_intent_pass(df: pd.DataFrame) -> pd.DataFrame:
    """Require measured evidence for each intended bridge behavior."""
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    passes = []
    notes = []
    for _, r in out.iterrows():
        if not _true_flag(r.get("ok")):
            passes.append(False)
            notes.append("run_failed_or_unknown")
            continue
        intent, tag = str(r.get("intent", "")), str(r.get("tag", ""))
        bridged = _count_or_none(r.get("n_eval_bridged"))
        n_pairs = _count_or_none(r.get("n_bridge_candidate_pairs"))
        n_good = _count_or_none(r.get("n_anchors_good"))
        gap_min = _finite_number(r.get("good_bridge_candidate_gap_mm_min"))
        axis_max = _finite_number(r.get("good_bridge_candidate_axis_deg_max"))
        top_err = _finite_number(r.get("true_top_n_axis_err_deg_median"))
        R_out = _finite_number(r.get("R_out"))
        R_th = _finite_number(r.get("R_out_theory"))
        if intent == "expect_possible_short_bridge_between_plateaus" and tag == "bridge_cfg_5mm_45deg":
            passes.append(bool(
                bridged is not None and bridged > 0
                and np.isfinite(top_err) and top_err <= 25.0
                and np.isfinite(R_out) and np.isfinite(R_th) and abs(R_out - R_th) <= 0.15
            ))
            notes.append("short_bridge_and_top_normal")
        elif intent == "expect_possible_short_bridge_between_plateaus" and tag == "bridge_cfg_5mm_30deg":
            passes.append(bridged == 0)
            notes.append("axis_reject_at_30deg")
        elif intent == "expect_gap_too_long_for_5mm_bridge" and tag == "bridge_cfg_5mm_45deg":
            passes.append(bool(
                bridged == 0 and n_pairs is not None and n_pairs > 0
                and np.isfinite(gap_min) and gap_min > 5.0
                and np.isfinite(axis_max) and axis_max <= 40.0
            ))
            notes.append("length_reject_with_small_axis")
        elif intent == "one_sided_good_should_not_fill_end" and tag == "bridge_cfg_5mm_45deg":
            passes.append(bool(
                n_good is not None and n_good > 0 and bridged == 0 and n_pairs == 0
                and _true_flag(r.get("end_eval_source_none"))
            ))
            notes.append("one_sided_end_none")
        else:
            passes.append(True)
            notes.append("observational")
    out["pass_intent"] = passes
    out["intent_check"] = notes
    return out



def unsigned_axis_angle_safe(n0, n1) -> float:
    from sin_trend_fitting_helpers import unsigned_axis_angle_deg

    try:
        return float(unsigned_axis_angle_deg(n0, n1))
    except Exception:
        return float("nan")


def run_classification_boundary_table(out_dir: Path) -> pd.DataFrame:
    rows = []
    for r in (0.249999, 0.25, 0.250001):
        cls = classify_direction(
            R_out=r,
            constraint_coverage=0.8,
            squared_displacement_coverage=0.8,
            status="ok",
        )
        rows.append(dict(R_out=r, R_tan=1.0 - r, label=cls["label"], evaluable=cls["evaluable"]))
               
    cls2 = classify_direction(R_out=0.1, constraint_coverage=0.4, squared_displacement_coverage=0.8, status="ok")
    rows.append(dict(R_out=0.1, R_tan=0.9, label=cls2["label"], evaluable=cls2["evaluable"], note="low_constraint"))
    df = pd.DataFrame(rows)
    atomic_to_csv(df, out_dir / "classification_boundary_direct.csv", index=False)
    return df


def _is_placeholder_directory_id(did: str) -> bool:
    s = str(did).strip().upper()
    return (not s) or s.startswith("EXAMPLE_") or s in {"PLACEHOLDER", "TODO"}


def run_clinical_sensitivity(out_dir: Path, targets_csv: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    from sin_trend_paths import brain_mesh_path, centerline_vtp_path, sha256_file, to_python_scalar
    from sin_trend_io_utils import project_root_from_here
    from sin_trend_measure import build_groupid_map
    from sin_trend_surface import run_surface_evaluation, surface_settings

    if not targets_csv.is_file():
        guide = dict(
            status="skipped_targets_csv_missing",
            reason="targets_csv_missing",
            path=str(targets_csv),
            clinical_mesh_eval_implemented=True,
            message="Targets CSV missing. Provide the expected format to run real-mesh evaluation.",
        )
        _write_json(guide, out_dir / "clinical_sensitivity_status.json")
        return guide

    df = pd.read_csv(targets_csv)
    need = {"directory_id", "segment_id"}
    missing = need - set(df.columns)
    if missing:
        guide = dict(status="error", reason=f"missing_columns:{sorted(missing)}", clinical_mesh_eval_implemented=True)
        _write_json(guide, out_dir / "clinical_sensitivity_status.json")
        return guide
    atomic_to_csv(df, out_dir / "clinical_targets_loaded.csv", index=False)

    sens = []
    for gap in (3.0, 5.0, 8.0):
        sens.append(dict(param="bridge_max_gap_mm", value=gap, note="one_at_a_time"))
    for ax in (30.0, 45.0, 60.0):
        sens.append(dict(param="bridge_axis_max_deg", value=ax, note="one_at_a_time"))
    for thr in (0.20, 0.25, 0.30):
        sens.append(dict(param="R_out_thresh", value=thr, R_tan_gt=1.0 - thr, note="reclassify_from_saved_R_out_only"))
    atomic_to_csv(pd.DataFrame(sens), out_dir / "clinical_sensitivity_grid.csv", index=False)

    placeholders = [str(r.directory_id) for r in df.itertuples(index=False) if _is_placeholder_directory_id(r.directory_id)]
    if len(placeholders) == len(df):
        guide = dict(
            status="skipped_placeholder_targets",
            n_targets=int(len(df)),
            clinical_mesh_eval_implemented=True,
            message=(
                "Targets are EXAMPLE_/placeholders only, so real-mesh evaluation was not run. "
                "Pass a CSV with real directory_id/segment_id via --targets-csv to evaluate. "
                "List loading without evaluation is intentional, not a missing implementation."
            ),
            primary_settings=dict(bridge_max_gap_mm=5.0, bridge_axis_max_deg=45.0, R_out_thresh=0.25),
        )
        _write_json(guide, out_dir / "clinical_sensitivity_status.json")
        return guide

                                               
    root = project_root_from_here(__file__)
    try:
        from centerline_extraction import load_vtp, vmtk_convert2numpy  # type: ignore
        import vessel_mesh_features  # type: ignore
    except Exception as exc:  # noqa: BLE001
        guide = dict(
            status="error_import",
            reason=str(exc),
            clinical_mesh_eval_implemented=True,
            message="Cannot import centerline_extraction / vessel_mesh_features required for clinical evaluation.",
        )
        _write_json(guide, out_dir / "clinical_sensitivity_status.json")
        return guide

    target_rows: List[Dict[str, Any]] = []
    sens_rows: List[Dict[str, Any]] = []
    fig_dir = out_dir / "figures" / "clinical"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _persist_clinical_tables() -> None:
        atomic_to_csv(pd.DataFrame(target_rows), out_dir / "clinical_primary_results.csv", index=False)
        if sens_rows:
            atomic_to_csv(pd.DataFrame(sens_rows), out_dir / "clinical_sensitivity_results.csv", index=False)

    def _sens_metrics(ev_d: Dict[str, Any], cls_d: Dict[str, Any]) -> Dict[str, Any]:
        return dict(
            R_out=ev_d.get("R_out"),
            R_tan=ev_d.get("R_tan"),
            constraint_coverage=ev_d.get("constraint_coverage"),
            squared_displacement_coverage=ev_d.get("squared_displacement_coverage"),
            squared_displacement_coverage_good=ev_d.get("squared_displacement_coverage_good"),
            squared_displacement_coverage_bridged=ev_d.get("squared_displacement_coverage_bridged"),
            good_coverage=ev_d.get("good_coverage"),
            bridged_coverage=ev_d.get("bridged_coverage"),
            cortex_coverage=ev_d.get("cortex_coverage"),
            n_eval_good=ev_d.get("n_eval_good"),
            n_eval_bridged=ev_d.get("n_eval_bridged"),
            abs_T0_dot_n_mean=ev_d.get("abs_T0_dot_n_mean"),
            abs_T0_dot_n_median=ev_d.get("abs_T0_dot_n_median"),
            R_out_upper_bound=ev_d.get("R_out_upper_bound"),
            R_out_within_upper_bound=ev_d.get("R_out_within_upper_bound"),
            evaluable=cls_d["evaluable"],
            direction_label=cls_d["label"],
            surface_status=ev_d.get("status"),
            surface_reason=ev_d.get("reason", ""),
            class_reason=cls_d.get("reason", ""),
        )

    for r in df.itertuples(index=False):
        did = str(r.directory_id)
        sid = to_python_scalar(r.segment_id)
        gnote = getattr(r, "geometry_note", "") if hasattr(r, "geometry_note") else ""
        base = dict(directory_id=did, segment_id=sid, geometry_note=gnote)
        primary_row: Optional[Dict[str, Any]] = None
        try:
            if _is_placeholder_directory_id(did):
                target_rows.append({**base, "status": "skipped_placeholder", "reason": "placeholder_directory_id"})
                _persist_clinical_tables()
                continue
            vtp = centerline_vtp_path(did, root)
            if not vtp.is_file():
                target_rows.append({**base, "status": "missing_vtp", "reason": f"missing:{vtp}", "input_vtp": str(vtp)})
                _persist_clinical_tables()
                continue
            try:
                mesh_cl = load_vtp(str(vtp))
                celldata, pointids, points, pointdata = vmtk_convert2numpy(mesh_cl)
                group_ids = [to_python_scalar(g) for g in celldata["GroupIds"]]
                mapping, cell_index_of, gid_diag = build_groupid_map(group_ids, pointids, points)
            except Exception as exc:  # noqa: BLE001
                target_rows.append({**base, "status": "vtp_load_failed", "reason": str(exc), "input_vtp": str(vtp)})
                _persist_clinical_tables()
                continue
            if sid not in mapping:
                amb = False
                if len(gid_diag) and "group_id" in gid_diag.columns:
                    amb = bool(((gid_diag["group_id"] == sid) & (gid_diag["resolution"] == "ambiguous_excluded")).any())
                target_rows.append(
                    {
                        **base,
                        "status": "segment_not_found_or_ambiguous",
                        "reason": "ambiguous_groupid" if amb else "segment_id_not_in_GroupIds",
                        "input_vtp": str(vtp),
                        "available_group_ids": "|".join(
                            str(g) for g in sorted(set(group_ids), key=lambda x: (str(type(x)), x))
                        ),
                    }
                )
                _persist_clinical_tables()
                continue
            cell_index = int(to_python_scalar(cell_index_of[sid]))
            pid = np.asarray(mapping[sid])
            P = np.asarray(points[pid], float)
                                                                             
            radius = None
            frenet = None
            if isinstance(pointdata, dict):
                if "MaximumInscribedSphereRadius" in pointdata:
                    radius = np.asarray(pointdata["MaximumInscribedSphereRadius"], float)[pid]
                if "FrenetTangent" in pointdata:
                    frenet = np.asarray(pointdata["FrenetTangent"], float)[pid]
            right_cells, *_ = vessel_mesh_features.classify_centerline_left_right_by_kmeans(
                points, pointids, n_iter=100, verbose=False
            )
            right_set = set(int(x) for x in np.asarray(right_cells).ravel().tolist())
            side = "Rt" if cell_index in right_set else "Lt"
            mp = brain_mesh_path(did, side, root)
            if not mp.is_file():
                target_rows.append(
                    {
                        **base,
                        "status": "missing_brain_mesh",
                        "reason": f"missing:{mp}",
                        "segment_index": cell_index,
                        "side": side,
                        "input_vtp": str(vtp),
                    }
                )
                _persist_clinical_tables()
                continue

            meas = measure_interval(
                P,
                frame_reference=FRAME_BASELINE,
                cfg=cfg,
                radius_vox=radius,
                observed_tangent=frenet,
                ids={"directory_id": did, "segment_id": sid, "segment_index": cell_index},
                verbose=False,
                compute_pca=False,
            )
            pw = meas.get("pointwise") or {}
            if pw.get("d_latent") is None or pw.get("P") is None:
                target_rows.append(
                    {
                        **base,
                        "status": "measure_failed",
                        "reason": str(meas["row"].get("fit_status")),
                        "segment_index": cell_index,
                        "side": side,
                        "radius_status": meas["row"].get("radius_status"),
                    }
                )
                _persist_clinical_tables()
                continue
                                                          
            patch = meas.get("patch") or {}
            if not patch.get("ok"):
                target_rows.append(
                    {
                        **base,
                        "status": "invalid_radius",
                        "reason": patch.get("reason", "patch_not_ok"),
                        "segment_index": cell_index,
                        "side": side,
                        "radius_status": meas["row"].get("radius_status"),
                        "radius_median_mm": meas["row"].get("radius_median_mm"),
                    }
                )
                _persist_clinical_tables()
                continue
            P_trim = np.asarray(pw["P"], float)
            P_full = np.asarray(pw.get("P_full", P_trim), float)
            d_lat = np.asarray(pw["d_latent"], float)
                                                    
            ds_vox = float(meas["row"].get("ds_median_vox", np.nan))
            if not np.isfinite(ds_vox) or ds_vox <= 0:
                raise ValueError("measurement ds_median_vox must be finite and positive")
            T0_c = np.asarray(pw.get("T0"), float)
            if T0_c.shape != P_trim.shape or not np.isfinite(T0_c).all():
                raise ValueError("measurement baseline T0 is missing or invalid")
            T_fr = np.asarray(pw.get("T", T0_c), float)
            if T_fr.shape != P_trim.shape or not np.isfinite(T_fr).all():
                raise ValueError("measurement frame tangent is missing or invalid")
            T_obs = np.asarray(pw["T_obs"], float) if pw.get("T_obs") is not None else None

            primary = dict(cfg)
            primary["bridge_max_gap_mm"] = 5.0
            primary["bridge_axis_max_deg"] = 45.0
            settings = surface_settings(primary)
                                                                 
            ev, est, cache_status = run_surface_evaluation(
                P_full_vox=P_full,
                P_trim=P_trim,
                d_sin=d_lat,
                ds_vox=float(ds_vox),
                r_patch_vox=float(patch["r_patch_vox"]),
                mesh_path=mp,
                settings=settings,
                cache_path=None,
                g1_3d=pw.get("g1_3d"),
                T_frame=T_fr,
                T0=T0_c,
                T_obs=T_obs,
                verbose=False,
            )
            cls = classify_direction(
                R_out=float(ev.get("R_out", np.nan)),
                constraint_coverage=float(ev.get("constraint_coverage", np.nan)),
                squared_displacement_coverage=float(ev.get("squared_displacement_coverage", np.nan)),
                status=str(ev.get("status", "")),
                R_out_thresh=0.25,
            )
                                                                
            primary_row = {
                **base,
                "status": "primary_ok_sensitivity_pending",
                "sensitivity_complete": False,
                "segment_index": cell_index,
                "side": side,
                "input_vtp": str(vtp),
                "brain_mesh": str(mp),
                "input_vtp_sha256": sha256_file(vtp),
                "brain_mesh_sha256": sha256_file(mp),
                "measurement_origin": "recomputed_from_centerline_vtp",
                "fit_status": meas["row"].get("fit_status"),
                "radius_status": meas["row"].get("radius_status"),
                "radius_median_mm": meas["row"].get("radius_median_mm"),
                "r_patch_mm": patch.get("r_patch_mm"),
                "r_patch_vox": patch.get("r_patch_vox"),
                "ds_median_vox": ds_vox,
                "normals_cache_status": cache_status,
                "R_out": ev.get("R_out"),
                "R_tan": ev.get("R_tan"),
                "constraint_coverage": ev.get("constraint_coverage"),
                "squared_displacement_coverage": ev.get("squared_displacement_coverage"),
                "squared_displacement_coverage_good": ev.get("squared_displacement_coverage_good"),
                "squared_displacement_coverage_bridged": ev.get("squared_displacement_coverage_bridged"),
                "cortex_coverage": ev.get("cortex_coverage"),
                "good_coverage": ev.get("good_coverage"),
                "bridged_coverage": ev.get("bridged_coverage"),
                "n_eval_good": ev.get("n_eval_good"),
                "n_eval_bridged": ev.get("n_eval_bridged"),
                "surface_status": ev.get("status"),
                "evaluable": cls["evaluable"],
                "direction_label": cls["label"],
                "bridge_max_gap_mm": 5.0,
                "bridge_axis_max_deg": 45.0,
                "R_out_thresh": 0.25,
                "R_out_upper_bound": ev.get("R_out_upper_bound"),
                "R_out_within_upper_bound": ev.get("R_out_within_upper_bound"),
                "abs_T0_dot_n_mean": ev.get("abs_T0_dot_n_mean"),
                "abs_T0_dot_n_median": ev.get("abs_T0_dot_n_median"),
            }

            for param, values in (
                ("bridge_max_gap_mm", (3.0, 5.0, 8.0)),
                ("bridge_axis_max_deg", (30.0, 45.0, 60.0)),
            ):
                for val in values:
                    cfg_i = dict(primary)
                    cfg_i[param] = float(val)
                    settings_i = surface_settings(cfg_i)
                    ev_i, _est_i, _ = run_surface_evaluation(
                        P_full_vox=P_full,
                        P_trim=P_trim,
                        d_sin=d_lat,
                        ds_vox=float(ds_vox),
                        r_patch_vox=float(patch["r_patch_vox"]),
                        mesh_path=mp,
                        settings=settings_i,
                        cache_path=None,
                        g1_3d=pw.get("g1_3d"),
                        T_frame=T_fr,
                        T0=T0_c,
                        T_obs=T_obs,
                        verbose=False,
                    )
                    cls_i = classify_direction(
                        R_out=float(ev_i.get("R_out", np.nan)),
                        constraint_coverage=float(ev_i.get("constraint_coverage", np.nan)),
                        squared_displacement_coverage=float(ev_i.get("squared_displacement_coverage", np.nan)),
                        status=str(ev_i.get("status", "")),
                        R_out_thresh=0.25,
                    )
                    sens_rows.append(
                        dict(
                            directory_id=did,
                            segment_id=sid,
                            segment_index=cell_index,
                            param=param,
                            value=val,
                            **_sens_metrics(ev_i, cls_i),
                        )
                    )
            R_saved = float(ev.get("R_out", np.nan))
            cons = float(ev.get("constraint_coverage", np.nan))
            sq = float(ev.get("squared_displacement_coverage", np.nan))
            st = str(ev.get("status", ""))
            for thr in (0.20, 0.25, 0.30):
                cls_t = classify_direction(
                    R_out=R_saved,
                    constraint_coverage=cons,
                    squared_displacement_coverage=sq,
                    status=st,
                    R_out_thresh=thr,
                )
                m = _sens_metrics(ev, cls_t)
                                                              
                m["R_out"] = R_saved
                m["R_tan"] = (1.0 - R_saved) if np.isfinite(R_saved) else float("nan")
                m["note"] = "reclassify_from_saved_R_out_only"
                sens_rows.append(
                    dict(
                        directory_id=did,
                        segment_id=sid,
                        segment_index=cell_index,
                        param="R_out_thresh",
                        value=thr,
                        **m,
                    )
                )

            try:
                import matplotlib.pyplot as plt
                from sin_trend_fitting_helpers import EVAL_NORMAL_SOURCE_BRIDGED, EVAL_NORMAL_SOURCE_GOOD

                src = np.asarray(est.get("eval_normal_source"), int)
                coh = np.asarray(est.get("coherence_full"), float)
                fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
                s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P_trim, axis=0), axis=1))] * float(cfg["voxel_size"])
                axes[0].plot(s, coh, lw=0.8, label="coherence")
                axes[0].axhline(0.75, color="k", ls="--", lw=0.7)
                axes[0].set_ylabel("coherence")
                axes[0].legend(fontsize=7)
                colors = np.full(len(src), "0.8")
                colors[src == EVAL_NORMAL_SOURCE_GOOD] = "C0"
                colors[src == EVAL_NORMAL_SOURCE_BRIDGED] = "C1"
                axes[1].scatter(s, np.zeros_like(s), c=colors, s=8, marker="|")
                axes[1].set_xlabel("centerline arc length [mm]")
                axes[1].set_yticks([])
                axes[1].set_title(f"{did} sid={sid} idx={cell_index} (GOOD=blue BRIDGED=orange)")
                fig.tight_layout()
                fig.savefig(fig_dir / f"{did}_sid{sid}_eval.png", dpi=120)
                plt.close(fig)
                primary_row["coherence_plot_status"] = "ok"
            except Exception as exc:
                primary_row["coherence_plot_status"] = f"failed:{exc}"

            try:
                import pyvista as pv

                brain = pv.read(str(mp))
                if "Normals" not in brain.point_data:
                    brain = brain.compute_normals(
                        point_normals=True, cell_normals=False, inplace=False, auto_orient_normals=True
                    )
                pv_files = write_clinical_pyvista_plots(
                    fig_dir=fig_dir,
                    directory_id=did,
                    segment_id=sid,
                    segment_index=cell_index,
                    brain=brain,
                    P_trim_vox=P_trim,
                    d_lat_vox=d_lat,
                    est=est,
                    ev=ev,
                    voxel_size_mm=float(cfg["voxel_size"]),
                    primary_row=primary_row,
                )
                primary_row["pyvista_files"] = "|".join(pv_files)
            except Exception as exc:  # noqa: BLE001
                primary_row["pyvista_files"] = f"failed:{exc}"

            primary_row["status"] = "ok"
            primary_row["sensitivity_complete"] = True
            target_rows.append(primary_row)
            _persist_clinical_tables()
        except Exception as exc:  # noqa: BLE001
                                                        
            if primary_row is not None and primary_row.get("status") == "primary_ok_sensitivity_pending":
                primary_row["status"] = "sensitivity_incomplete"
                primary_row["sensitivity_complete"] = False
                primary_row["reason"] = str(exc)
                target_rows.append(primary_row)
            else:
                target_rows.append({**base, "status": "target_exception", "reason": str(exc)})
            _persist_clinical_tables()
            continue

    _persist_clinical_tables()
    n_ok = sum(1 for t in target_rows if t.get("status") == "ok" and t.get("sensitivity_complete", True))
    n_sens_incomplete = sum(1 for t in target_rows if t.get("status") == "sensitivity_incomplete")
    n_target_exception = sum(1 for t in target_rows if t.get("status") == "target_exception")
    n_placeholder = sum(1 for t in target_rows if t.get("status") == "skipped_placeholder")
    n_plot_failed = sum(
        1 for t in target_rows
        if str(t.get("coherence_plot_status", "")).startswith("failed:")
        or str(t.get("pyvista_files", "")).startswith("failed:")
    )
    n_other_fail = sum(
        1
        for t in target_rows
        if t.get("status")
        not in {"ok", "skipped_placeholder", "sensitivity_incomplete", "target_exception", "primary_ok_sensitivity_pending"}
    )
    n_real_attempted = int(len(df)) - int(n_placeholder)
    if n_sens_incomplete or n_target_exception or n_plot_failed:
        guide_status = "ran_with_failures"
    elif n_ok and n_ok == n_real_attempted:
        guide_status = "ran"
    elif n_ok:
                                    
        guide_status = "ran_partial"
    else:
        guide_status = "ran_no_successful_targets"
    guide = dict(
        status=guide_status,
        n_targets=int(len(df)),
        n_real_attempted=int(n_real_attempted),
        n_ok=int(n_ok),
        n_sensitivity_incomplete=int(n_sens_incomplete),
        n_target_exception=int(n_target_exception),
        n_other_fail=int(n_other_fail),
        n_plot_failed=int(n_plot_failed),
        clinical_mesh_eval_implemented=True,
        primary_settings=dict(bridge_max_gap_mm=5.0, bridge_axis_max_deg=45.0, R_out_thresh=0.25),
        message=(
            "Evaluate listed targets only; missing/ambiguous GroupIds were not substituted. "
            "Patch radius follows production: median trim radius from measure_interval. "
            "Surface uses production run_surface_evaluation (mesh crop + normals). "
            "Radius from MaximumInscribedSphereRadius; ds from ds_median_vox. "
            "Do not set status=ok until sensitivity finishes. Small-n sensitivity only; not cohort-representative."
        ),
    )
    _write_json(guide, out_dir / "clinical_sensitivity_status.json")
    return guide


def write_clinical_pyvista_plots(
    *,
    fig_dir: Path,
    directory_id: str,
    segment_id: Any,
    segment_index: int,
    brain,
    P_trim_vox: np.ndarray,
    d_lat_vox: np.ndarray,
    est: Dict[str, Any],
    ev: Dict[str, Any],
    voxel_size_mm: float,
    primary_row: Dict[str, Any],
) -> List[str]:
    import pyvista as pv
    from sin_trend_fitting_helpers import EVAL_NORMAL_SOURCE_BRIDGED, EVAL_NORMAL_SOURCE_GOOD

    fig_dir.mkdir(parents=True, exist_ok=True)
    vs = float(voxel_size_mm)
    P = np.asarray(P_trim_vox, float) * vs  # mm
    d = np.asarray(d_lat_vox, float) * vs
    nrm = np.asarray(est["normals_full"], float)
    src = np.asarray(est.get("eval_normal_source"), int)
    mask = np.asarray(est["mask_eval"], bool)
    coh = np.asarray(est.get("coherence_full"), float)

                        
    brain_mm = brain.copy(deep=True)
    brain_mm.points = np.asarray(brain_mm.points, float) * vs
    bounds = np.array(
        [
            P[:, 0].min() - 8,
            P[:, 0].max() + 8,
            P[:, 1].min() - 8,
            P[:, 1].max() + 8,
            P[:, 2].min() - 8,
            P[:, 2].max() + 8,
        ],
        float,
    )
    cropped = brain_mm.clip_box(bounds, invert=False)

    cl_line = pv.lines_from_points(P, close=False)
    from sin_trend_fitting_helpers import EVAL_NORMAL_SOURCE_NONE

    good_pts = pv.PolyData(P[src == EVAL_NORMAL_SOURCE_GOOD]) if np.any(src == EVAL_NORMAL_SOURCE_GOOD) else None
    bridged_pts = pv.PolyData(P[src == EVAL_NORMAL_SOURCE_BRIDGED]) if np.any(src == EVAL_NORMAL_SOURCE_BRIDGED) else None
    none_mask = src == EVAL_NORMAL_SOURCE_NONE
    none_pts = pv.PolyData(P[none_mask]) if np.any(none_mask) else None

                
    step = max(1, len(P) // 60)
    idx = np.arange(0, len(P), step)
    idx_eval = idx[mask[idx]] if np.any(mask[idx]) else idx
    glyphs_d = pv.PolyData(P[idx_eval])
    glyphs_d["vectors"] = d[idx_eval]
    arrows_d = glyphs_d.glyph(orient="vectors", scale="vectors", factor=3.0, geom=pv.Arrow())
    glyphs_n = pv.PolyData(P[idx_eval])
    nn = nrm[idx_eval]
    nn_ok = np.all(np.isfinite(nn), axis=1) & (np.linalg.norm(nn, axis=1) > 0)
    if np.any(nn_ok):
        glyphs_n = pv.PolyData(P[idx_eval][nn_ok])
        glyphs_n["vectors"] = nn[nn_ok] * 2.0              
        arrows_n = glyphs_n.glyph(orient="vectors", scale=False, factor=1.0, geom=pv.Arrow())
    else:
        arrows_n = None

    title = (
        f"{directory_id} sid={segment_id} idx={segment_index} | "
        f"R_tan={primary_row.get('R_tan')} | {primary_row.get('direction_label')}"
    )
    files: List[str] = []
    stem = f"{directory_id}_sid{segment_id}"

    legend_text = (
        "Arrow legend\n"
        "Red: d_latent = periodic displacement (fitted sin component)\n"
        "Green: n_eval = evaluation cortex normal (GOOD/BRIDGED)\n"
        "Black: observed centerline / gray: inevaluable (NONE)"
    )
    if arrows_n is None:
        legend_text += "\nNote: no evaluation normals in this case (no green arrows)"

    def _add_common_actors(pl, *, with_coherence_points: bool = False) -> None:
        pl.add_mesh(cropped, color="lightgray", opacity=0.35, smooth_shading=True, label="brain surface (clip)")
        pl.add_mesh(cl_line, color="black", line_width=4, label="centerline (observed)")
        if with_coherence_points:
            pts = pv.PolyData(P)
            pts["coherence"] = coh
            pl.add_mesh(
                pts,
                scalars="coherence",
                cmap="viridis",
                clim=[0, 1],
                point_size=8,
                render_points_as_spheres=True,
                label="coherence (0–1)",
            )
        if good_pts is not None and good_pts.n_points:
            pl.add_mesh(good_pts, color="#1f77b4", point_size=10, render_points_as_spheres=True, label="GOOD anchor/eval")
        if bridged_pts is not None and bridged_pts.n_points:
            pl.add_mesh(bridged_pts, color="#ff7f0e", point_size=10, render_points_as_spheres=True, label="BRIDGED eval")
        if none_pts is not None and none_pts.n_points:
            pl.add_mesh(none_pts, color="#bbbbbb", point_size=6, render_points_as_spheres=True, label="NONE (not evaluable)")
        pl.add_mesh(
            arrows_d,
            color="#d62728",
            label="RED arrow: d_latent (periodic displacement)",
        )
        if arrows_n is not None:
            pl.add_mesh(
                arrows_n,
                color="#2ca02c",
                label="GREEN arrow: n_eval (surface normal)",
            )
        pl.add_text(title, font_size=9, position="upper_left")
        pl.add_text(legend_text, font_size=8, position="lower_left")
        pl.add_legend(bcolor="white", face=None, size=(0.28, 0.28))

    def _render(path: Path, camera: str = "iso") -> None:
        pl = pv.Plotter(off_screen=True, window_size=(1400, 1000))
        pl.set_background("white")
        _add_common_actors(pl)
        pl.camera_position = camera
        pl.reset_camera()
        if camera == "xy":
            pl.view_xy()
        elif camera == "xz":
            pl.view_xz()
        elif camera == "yz":
            pl.view_yz()
        pl.screenshot(str(path))
        pl.close()

    for cam, suffix in (("iso", "iso"), ("xy", "xy"), ("xz", "xz")):
        fname = f"{stem}_pyvista_{suffix}.png"
        _render(fig_dir / fname, camera=cam if cam != "iso" else "iso")
        files.append(fname)

                                           
    try:
        pl = pv.Plotter(off_screen=True, window_size=(1400, 1000))
        pl.set_background("white")
        _add_common_actors(pl, with_coherence_points=True)
        html_name = f"{stem}_pyvista.html"
        html_path = fig_dir / html_name
        pl.export_html(str(html_path))
        pl.close()
                                                 
        try:
            html = html_path.read_text(encoding="utf-8")
            panel = """
<div id="arrow-legend-panel" style="position:fixed;left:12px;bottom:12px;z-index:9999;
background:rgba(255,255,255,0.92);border:1px solid #333;border-radius:8px;padding:10px 12px;
font:13px/1.45 sans-serif;color:#111;max-width:420px;box-shadow:0 2px 8px rgba(0,0,0,.15)">
<strong>Arrow meaning</strong><br>
<span style="color:#d62728">■ Red</span>: <code>d_latent</code> = periodic displacement (3D direction of fitted sine)<br>
<span style="color:#2ca02c">■ Green</span>: <code>n_eval</code> = evaluation cortex normal (GOOD/BRIDGED only)<br>
<span style="color:#000">■ Black</span>: observed centerline<br>
<span style="color:#888">■ Gray</span>: inevaluable (NONE); colored points show coherence
</div>
"""
            if "</body>" in html:
                html = html.replace("</body>", panel + "\n</body>")
            else:
                html += panel
            html_path.write_text(html, encoding="utf-8")
        except Exception:
            pass
        files.append(html_name)
    except Exception:
        pass

            
    _write_json(
        {
            "R_out": primary_row.get("R_out"),
            "R_tan": primary_row.get("R_tan"),
            "abs_T0_dot_n_median": primary_row.get("abs_T0_dot_n_median"),
            "R_out_upper_bound": primary_row.get("R_out_upper_bound"),
            "direction_label": primary_row.get("direction_label"),
            "n_eval_good": ev.get("n_eval_good"),
            "n_eval_bridged": ev.get("n_eval_bridged"),
            "files": files,
        },
        fig_dir / f"{stem}_pyvista_meta.json",
    )
    return files


def write_figures_plane(out_dir: Path, cfg: Dict[str, Any], valley_fig_cache: Optional[Dict[str, Any]] = None) -> None:
    files = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        _write_json({"figures": "skipped", "reason": str(exc)}, out_dir / "figures_status.json")
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    th = np.linspace(0, 90, 181)
    ro = np.sin(np.deg2rad(th)) ** 2
    rt = np.cos(np.deg2rad(th)) ** 2
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(th, ro, label="R_out=sin²θ")
    ax.plot(th, rt, label="R_tan=cos²θ")
    ax.axhline(0.25, color="k", ls="--", lw=0.8, label="R_out=0.25")
    ax.axhline(0.75, color="gray", ls=":", lw=0.8, label="R_tan=0.75")
    ax.set_xlabel("θ plane angle [deg]")
    ax.set_ylabel("ratio")
    ax.legend(fontsize=8)
    ax.set_title("Theory (not data)")
    fig.tight_layout()
    fig.savefig(fig_dir / "theory_R_out_R_tan.png", dpi=120)
    plt.close(fig)
    files.append("theory_R_out_R_tan.png")

                                        
    if valley_fig_cache:
        from sin_trend_fitting_helpers import EVAL_NORMAL_SOURCE_BRIDGED, EVAL_NORMAL_SOURCE_GOOD

        for path_name, pack in valley_fig_cache.items():
            try:
                cl = pack["cl"]
                est = pack["est"]
                P = np.asarray(cl["P_mm"], float)
                src = np.asarray(est.get("eval_normal_source"), int)
                coh = np.asarray(est.get("coherence_full"), float)
                nrm = np.asarray(est.get("normals_full"), float)
                fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [2, 1]})
                axes[0].plot(P[:, 0], P[:, 2], "k-", lw=1.2, label="centerline (x-z)")
                                                                
                z_surf = cl.get("z_surf_mm")
                mesh_mode = str(cl.get("mesh_mode") or pack.get("mesh_mode") or "gaussian_peaks")
                mesh_spec = pack.get("mesh_spec") or cl.get("mesh_spec") or {}
                if z_surf is None and mesh_spec:
                    z_surf = surface_height_mm(
                        np.asarray(cl["x_mm"], float),
                        np.zeros_like(np.asarray(cl["x_mm"], float)),
                        mode=mesh_mode,
                        peak1_x_mm=float(mesh_spec.get("peak1_x_mm", 0.0)),
                        peak2_x_mm=float(mesh_spec.get("peak2_x_mm", 0.0)),
                        peak_halfwidth_mm=float(mesh_spec.get("peak_halfwidth_mm", 2.5)),
                        peak_z_mm=float(mesh_spec.get("peak_z_mm", 3.0)),
                        trench_halfwidth_mm=float(mesh_spec.get("trench_halfwidth_mm", 2.0)),
                        wall_ramp_mm=float(mesh_spec.get("wall_ramp_mm", 1.0)),
                    )
                if z_surf is not None:
                    axes[0].fill_between(cl["x_mm"], z_surf, color="C2", alpha=0.25, label=f"mesh height ({mesh_mode})")
                    axes[0].plot(cl["x_mm"], z_surf, "C2--", lw=0.8)
                step = max(1, len(P) // 40)
                for i in range(0, len(P), step):
                    if not np.all(np.isfinite(nrm[i])):
                        continue
                    col = "C0" if src[i] == EVAL_NORMAL_SOURCE_GOOD else ("C1" if src[i] == EVAL_NORMAL_SOURCE_BRIDGED else "0.7")
                    axes[0].arrow(P[i, 0], P[i, 2], nrm[i, 0] * 0.8, nrm[i, 2] * 0.8, color=col, head_width=0.05, length_includes_head=True)
                axes[0].set_xlabel("x [mm]")
                axes[0].set_ylabel("z [mm]")
                axes[0].legend(fontsize=7)
                axes[0].set_title(f"valley path={path_name} mode={mesh_mode} (GOOD=blue BRIDGED=orange)")
                axes[1].plot(P[:, 0], coh, lw=0.9)
                axes[1].axhline(0.75, color="k", ls="--", lw=0.7)
                axes[1].set_ylabel("coherence")
                axes[1].set_xlabel("x [mm]")
                fig.tight_layout()
                fname = f"valley_{path_name}.png"
                fig.savefig(fig_dir / fname, dpi=120)
                plt.close(fig)
                files.append(fname)
            except Exception as exc:  # noqa: BLE001
                files.append(f"valley_{path_name}_failed:{exc}")

    _write_json({"figures": "ok", "files": files}, out_dir / "figures_status.json")


def write_japanese_report(
    out_dir: Path,
    *,
    thresh_cmp: Dict[str, Any],
    plane_direct: pd.DataFrame,
    plane_fit: pd.DataFrame,
    geom: pd.DataFrame,
    valley: pd.DataFrame,
    clinical: Dict[str, Any],
    existing_tests: Dict[str, Any],
    overall: Dict[str, Any],
) -> None:
    lines = []
    lines.append("# Surface-direction validation report\n")
    lines.append(f"- Output: `{out_dir}`\n")
    lines.append(f"- MEASUREMENT_REVISION: `{MEASUREMENT_REVISION}`\n")
    lines.append(f"- SURFACE_METHOD_REVISION: `{SURFACE_METHOD_REVISION}`\n")
    lines.append(f"- overall_status: `{overall.get('status')}`\n")
    lines.append("\n## Tolerances (fixed before run)\n")
    lines.append(f"- Plane + known displacement (direct): |ΔR| ≤ {TOL_PLANE_DIRECT_ABS}, normal-axis median ≤ {TOL_NORMAL_AXIS_DEG}°, plus expected class (pass_class)\n")
    lines.append(f"- Via fit: |ΔR| ≤ {TOL_PLANE_FIT_ABS}\n")
    lines.append(f"- Geometric T check: analytic |Δ| ≤ {TOL_GEOM_ABS} and numerical mean check\n")
    lines.append("\n## Relation to production thresholds\n")
    lines.append(f"- Production R_out_thresh: {thresh_cmp.get('production_R_out_thresh')}\n")
    lines.append(f"- Validation R_out_thresh: {thresh_cmp.get('validation_R_out_thresh')} (R_tan > {thresh_cmp.get('validation_R_tan_thresh')})\n")
    lines.append(f"- Differs: {thresh_cmp.get('differs')}\n")
    lines.append("\n## Known deviations from the written spec\n")
    lines.append("- Production column rename not applied globally (validation outputs also report R_tan).\n")
    lines.append("- evaluated_tangential_fraction uses all-interval denominator × R_tan; it is not R_tan itself.\n")
    lines.append("- Valley mesh is peak→valley→peak along x; missing regions are not force-masked.\n")
    lines.append("- Valley displacement uses θ=30° (includes z). Figure height matches compute mesh surface_height_mm.\n")
    lines.append("- Overall verdict fails on empty tables, ok=False, or missing required cases (not only pass=False).\n")
    lines.append("- Real-data patch from measure_interval trim radius; missing radius → invalid_radius (no imputation).\n")
    lines.append("- Bridging distance uses cumulative arc length (production estimate_n_eff).\n")
    lines.append("\n## Plane + known displacement (direct)\n")
    if len(plane_direct):
        cols = [
            c
            for c in (
                "theta_plane_deg",
                "R_out",
                "R_tan",
                "abs_T0_dot_n_median",
                "R_out_upper_bound",
                "R_out_within_upper_bound",
                "R_out_theory",
                "err_R_out_vs_theory",
                "direction_label",
                "expected_direction_label",
                "pass_R",
                "pass_class",
                "pass_R_out_le_upper_bound",
                "pass_all",
            )
            if c in plane_direct.columns
        ]
        sub = plane_direct[plane_direct["mode"] == "direct"] if "mode" in plane_direct.columns else plane_direct
        lines.append(sub[cols].to_string(index=False))
        lines.append("\n")
        lines.append(
            "\n(Pass baseline tangent T0 into evaluate; report |T0·n| and R_out_upper_bound alongside R_tan. "
            " For straight×plane, |T0·n|≈0 and upper bound≈1. Confirm R_out ≤ bound via pass_R_out_le_upper_bound.)\n"
        )
    lines.append("\n## Plane via fit (separate table)\n")
    if len(plane_fit):
        cols = [
            c
            for c in (
                "case",
                "theta_approx_deg",
                "R_out",
                "R_tan",
                "abs_T0_dot_n_median",
                "R_out_upper_bound",
                "R_out_true_on_mask",
                "err_R_out_vs_theory",
                "err_R_out_vs_mask_truth",
                "direction_label",
                "pass_R",
                "fit_status",
            )
            if c in plane_fit.columns
        ]
        lines.append(plane_fit[cols].to_string(index=False))
        lines.append("\n")
    lines.append("\n## Geometric T check (known normals; not a null distribution)\n")
    lines.append(geom.to_string(index=False) if len(geom) else "(empty)")
    lines.append("\n\n## Valley / bridging observation\n")
    lines.append(f"Rows: {len(valley)} (details in valley_bridge_observation.csv)\n")
    if len(valley) and "n_eval_bridged" in valley.columns:
        n_bridge = int(np.nansum(pd.to_numeric(valley["n_eval_bridged"], errors="coerce").fillna(0) > 0))
        lines.append(f"- Conditions with BRIDGED>0: {n_bridge} / {len(valley)}\n")
    lines.append("\n## Real-data sensitivity\n")
    lines.append(json.dumps(clinical, ensure_ascii=False, indent=2))
    lines.append("\n\n## Existing tests\n")
    lines.append(json.dumps(existing_tests, ensure_ascii=False, indent=2))
    lines.append("\n\n## Overall verdict\n")
    lines.append(json.dumps(overall, ensure_ascii=False, indent=2))
    lines.append("\n\n## How to read the conclusions\n")
    lines.append("1. Numerical correctness: R_out/R_tan error and boundary class for known displacement × plane mesh (direct).\n")
    lines.append("2. Recovery of known direction: via-fit reported with separate tolerances (fit_through); truth columns are generated displacements.\n")
    lines.append("3. Real-data stability: skipped if no real IDs provided. Implementation exists but placeholders do not run.\n")
    lines.append("\nDo not claim from these results that "0.25 is optimal" or that mechanical constraint is proven.\n")
    (out_dir / "REPORT_JA.md").write_text("\n".join(str(x) for x in lines), encoding="utf-8")


def run_existing_unit_tests() -> Dict[str, Any]:
    import subprocess

    py = sys.executable
    tests = [
        "tests/test_good_anchor_bridge.py",
        "tests/test_surface_eligibility_no_cortex_gate.py",
        "tests/test_eval_source_coverage_split.py",
        "tests/test_surface_direction_validation_extras.py",
        "tests/test_surface_direction_validation_workflow.py",
    ]
    results = {}
    for t in tests:
        path = _HERE / t
        if not path.is_file():
            results[t] = dict(status="missing")
            continue
        p = subprocess.run([py, "-m", "pytest", str(path), "-q"], cwd=str(_HERE), capture_output=True, text=True)
        if p.returncode != 0:
            p2 = subprocess.run([py, str(path)], cwd=str(_HERE), capture_output=True, text=True)
            results[t] = dict(
                status="ok" if p2.returncode == 0 else "fail",
                via="script" if p2.returncode == 0 else "pytest_and_script_failed",
                stdout=(p2.stdout or "")[-500:],
                stderr=(p2.stderr or p.stderr or "")[-500:],
            )
        else:
            results[t] = dict(status="ok", via="pytest", stdout=(p.stdout or "")[-300:])
    return results


def _table_required_failures(
    df: Optional[pd.DataFrame],
    name: str,
    *,
    pass_cols: Tuple[str, ...],
    require_nonempty: bool = True,
    min_ok_rows: Optional[int] = None,
) -> List[str]:
    fails: List[str] = []
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        if require_nonempty:
            fails.append(f"{name}:empty")
        return fails
    if "ok" in df.columns:
        ok_mask = df["ok"].eq(True).fillna(False)
        n_bad = int((~ok_mask).sum())
        if n_bad:
            fails.append(f"{name}:ok=False_or_missing_n={n_bad}")
        n_ok = int(ok_mask.sum())
        if min_ok_rows is not None and n_ok < int(min_ok_rows):
            fails.append(f"{name}:ok_rows={n_ok}<required_{min_ok_rows}")
    elif min_ok_rows is not None and len(df) < int(min_ok_rows):
        fails.append(f"{name}:rows={len(df)}<required_{min_ok_rows}")
    for col in pass_cols:
        if col not in df.columns:
                                 
            if "ok" not in df.columns or bool((df["ok"] == True).any()):  # noqa: E712
                fails.append(f"{name}:missing_col:{col}")
            continue
        mask = ~df[col].eq(True).fillna(False)
        if "ok" in df.columns:
            mask = mask & df["ok"].eq(True).fillna(False)
        n_fail = int(mask.sum())
        if n_fail:
            fails.append(f"{name}:{col}=False_or_missing_n={n_fail}")
    return fails


def _valley_required_failures(valley: Optional[pd.DataFrame]) -> List[str]:
    fails: List[str] = []
    fails.extend(
        _table_required_failures(
            valley,
            "valley",
            pass_cols=("pass_intent",),
            require_nonempty=True,
            min_ok_rows=1,
        )
    )
    if valley is None or not isinstance(valley, pd.DataFrame) or len(valley) == 0:
        return fails

    def _pick(path: str, tag: str) -> Optional[pd.Series]:
        if "path" not in valley.columns or "tag" not in valley.columns:
            return None
        sub = valley[(valley["path"] == path) & (valley["tag"] == tag)]
        if len(sub) == 0:
            return None
        return sub.iloc[0]

    short45 = _pick("air_ridge_short_trench", "bridge_cfg_5mm_45deg")
    short30 = _pick("air_ridge_short_trench", "bridge_cfg_5mm_30deg")
    long45 = _pick("air_ridge_long_trench", "bridge_cfg_5mm_45deg")
    one45 = _pick("one_sided_into_trench", "bridge_cfg_5mm_45deg")
    if short45 is None:
        fails.append("valley:required_missing:short_5mm_45deg")
    elif not _true_flag(short45.get("ok")) or (_count_or_none(short45.get("n_eval_bridged")) or 0) <= 0:
        fails.append("valley:short_bridge_not_observed")
    if short30 is None:
        fails.append("valley:required_missing:short_5mm_30deg")
    elif not _true_flag(short30.get("ok")) or _count_or_none(short30.get("n_eval_bridged")) != 0:
        fails.append("valley:axis_reject_30deg_failed")
    if long45 is None:
        fails.append("valley:required_missing:long_5mm_45deg")
    else:
        axis_max = _finite_number(long45.get("good_bridge_candidate_axis_deg_max"))
        bridged = _count_or_none(long45.get("n_eval_bridged"))
        if not _true_flag(long45.get("ok")) or bridged != 0 or not np.isfinite(axis_max) or axis_max > 40.0:
            fails.append("valley:length_reject_not_isolated")
    if one45 is None:
        fails.append("valley:required_missing:one_sided_5mm_45deg")
    elif (
        not _true_flag(one45.get("ok"))
        or _count_or_none(one45.get("n_eval_bridged")) != 0
        or not _true_flag(one45.get("end_eval_source_none"))
    ):
        fails.append("valley:one_sided_end_fill_failed")
    return fails


def aggregate_overall_status(
    plane_direct: pd.DataFrame,
    plane_fit: pd.DataFrame,
    geom: pd.DataFrame,
    existing: Dict[str, Any],
    *,
    valley: Optional[pd.DataFrame] = None,
    clinical: Optional[Dict[str, Any]] = None,
    expect_plane_fit: bool = True,
) -> Dict[str, Any]:
    fails: List[str] = []
    fails.extend(
        _table_required_failures(
            plane_direct,
            "plane_direct",
            pass_cols=("pass_all",),
            require_nonempty=True,
            min_ok_rows=7,
        )
    )
    fails.extend(
        _table_required_failures(
            plane_fit,
            "plane_fit",
            pass_cols=("pass_R",),
            require_nonempty=bool(expect_plane_fit),
            min_ok_rows=3 if expect_plane_fit else None,
        )
    )
    fails.extend(
        _table_required_failures(
            geom,
            "geom",
            pass_cols=("pass_abs",),
            require_nonempty=True,
            min_ok_rows=3,
        )
    )
    fails.extend(_valley_required_failures(valley))
    for t, st in (existing or {}).items():
        if not isinstance(st, dict):
            fails.append(f"test:{t}:bad_status")
            continue
        if st.get("status") != "ok":
            fails.append(f"test:{t}:{st.get('status')}")
    if clinical is not None and isinstance(clinical, dict):
        st = str(clinical.get("status", ""))
        n_ok_c = int(clinical.get("n_ok") or 0)
        n_sens_inc = int(clinical.get("n_sensitivity_incomplete") or 0)
        n_texc = int(clinical.get("n_target_exception") or 0)
        n_real = int(clinical.get("n_real_attempted") or 0)
        n_other_fail = int(clinical.get("n_other_fail") or 0)
        n_plot_failed = int(clinical.get("n_plot_failed") or 0)
                                                  
        if st.startswith("error"):
            fails.append(f"clinical:{st}")
        elif st in {"ran_with_failures", "ran_no_successful_targets", "ran_partial"}:
            fails.append(f"clinical:{st}")
        elif n_other_fail > 0 or n_plot_failed > 0:
            fails.append(f"clinical:missing_targets_or_plots:targets={n_other_fail}:plots={n_plot_failed}")
        elif n_sens_inc > 0 or n_texc > 0:
            fails.append(f"clinical:incomplete_or_exception:sens={n_sens_inc}:exc={n_texc}")
        elif n_real > 0 and n_ok_c == 0 and st not in {
            "skipped_placeholder_targets",
            "skipped_targets_csv_missing",
        }:
            fails.append(f"clinical:no_successful_targets:attempted={n_real}")
    status = "pass" if not fails else "fail"
    return dict(status=status, n_failures=len(fails), failures=fails)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Surface direction (R_out/R_tan) validation")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: results/analysis/sin_trend/surface_direction_validation/surface_direction_<timestamp>)",
    )
    p.add_argument(
        "--targets-csv",
        type=Path,
        default=_HERE / "surface_direction_validation_targets.example.csv",
        help="real-data target list (directory_id,segment_id[,geometry_note])",
    )
    p.add_argument("--skip-existing-tests", action="store_true")
    p.add_argument("--skip-fit-through", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    argv_list = list(argv) if argv is not None else list(sys.argv[1:])
    args = parse_args(argv_list)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                                                         
    from sin_trend_io_utils import project_root_from_here

    project_root = project_root_from_here(__file__)
    default_out = (
        project_root
        / "results"
        / "analysis"
        / "sin_trend"
        / "surface_direction_validation"
        / f"surface_direction_{ts}"
    )
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_out
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = validation_config()
    thresh_cmp = _prod_vs_val_thresh()
    cmd_parts = [sys.executable, str(Path(__file__).resolve())] + argv_list
    _write_json(
        dict(
            config=cfg,
            tolerances=dict(
                plane_direct_abs=TOL_PLANE_DIRECT_ABS,
                plane_fit_abs=TOL_PLANE_FIT_ABS,
                normal_axis_deg=TOL_NORMAL_AXIS_DEG,
                geometric_abs=TOL_GEOM_ABS,
            ),
            thresh_compare=thresh_cmp,
            command=_shlex_join(cmd_parts),
            argv=argv_list,
            code_refs=[
                "sin_trend_surface.py:estimate_normals,evaluate_displacement",
                "sin_trend_fitting_helpers.py:interpolate_good_anchor_normals_with_bridge,estimate_n_eff_along_centerline",
                "sin_trend_classify.py:surface_eligible_from_row,SURFACE_LABEL_*",
                "sin_trend_io_utils.py:compute_patch_radius_vox",
            ],
        ),
        out_dir / "run_settings.json",
    )

    ex = _HERE / "surface_direction_validation_targets.example.csv"
    if ex.is_file():
        (out_dir / "targets_format_example.csv").write_text(ex.read_text(encoding="utf-8"), encoding="utf-8")

    plane_direct = run_plane_direct(out_dir, cfg)
    plane_fit = pd.DataFrame() if args.skip_fit_through else run_plane_fit_through(out_dir, cfg)
    geom = run_geometric_T(out_dir)
    valley, valley_fig = run_valley_bridge(out_dir, cfg)
    boundary = run_classification_boundary_table(out_dir)
    clinical = run_clinical_sensitivity(out_dir, Path(args.targets_csv), cfg)
    write_figures_plane(out_dir, cfg, valley_fig_cache=valley_fig)

    existing = {} if args.skip_existing_tests else run_existing_unit_tests()
    _write_json(existing, out_dir / "existing_tests_status.json")

    overall = aggregate_overall_status(
        plane_direct,
        plane_fit,
        geom,
        existing,
        valley=valley,
        clinical=clinical,
        expect_plane_fit=not bool(args.skip_fit_through),
    )
    _write_json(overall, out_dir / "overall_status.json")

    write_japanese_report(
        out_dir,
        thresh_cmp=thresh_cmp,
        plane_direct=plane_direct,
        plane_fit=plane_fit,
        geom=geom,
        valley=valley,
        clinical=clinical,
        existing_tests=existing,
        overall=overall,
    )
    if overall["status"] == "pass":
        print(f"[ok] validation PASS; wrote outputs under {out_dir}")
        return 0
    print(f"[fail] validation FAIL ({overall['n_failures']}); wrote outputs under {out_dir}")
    for f in overall.get("failures", []):
        print(f"  - {f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
