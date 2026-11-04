#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend surface direction validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sin_trend_classify import (
    SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL,
    SURFACE_LABEL_SUBSTANTIAL_NORMAL,
)
from sin_trend_fitting_helpers import (
    EVAL_NORMAL_SOURCE_BRIDGED,
    EVAL_NORMAL_SOURCE_GOOD,
    EVAL_NORMAL_SOURCE_NONE,
    unsigned_axis_angle_deg,
)
from sin_trend_io_utils import compute_patch_radius_vox
from sin_trend_measure import default_measurement_config
from sin_trend_surface import estimate_normals, evaluate_displacement, surface_settings

# ---------------------------------------------------------------------------
                         
# ---------------------------------------------------------------------------
                              
TOL_PLANE_DIRECT_ABS = 0.02  # |R_out - sin²θ|, |R_tan - cos²θ|
TOL_NORMAL_AXIS_DEG = 5.0  # arccos(|n·n_true|) [deg]
                       
TOL_PLANE_FIT_ABS = 0.08
                   
TOL_GEOM_ABS = 1e-9

                      
VAL_R_OUT_THRESH = 0.25
VAL_R_TAN_THRESH = 0.75  # Predominantly tangential: R_tan > this


def validation_config(
    *,
    voxel_size_mm: float = 0.5,
    target_ds_mm: float = 0.1,
    coh_thr: float = 0.75,
    min_neighbors: int = 30,
    bridge_max_gap_mm: float = 5.0,
    bridge_axis_max_deg: float = 45.0,
    stride: int = 10,
    patch_r_ratio: float = 3.0,
    r_min_mm: float = 2.5,
    r_max_mm: float = 5.0,
    R_out_thresh: float = VAL_R_OUT_THRESH,
    R_tan_thresh: float = VAL_R_TAN_THRESH,
) -> Dict[str, Any]:
    cfg = default_measurement_config()
    cfg.update(
        dict(
            voxel_size=float(voxel_size_mm),
            coh_thr=float(coh_thr),
            min_neighbors=int(min_neighbors),
            bridge_max_gap_mm=float(bridge_max_gap_mm),
            bridge_axis_max_deg=float(bridge_axis_max_deg),
            stride=int(stride),
            patch_r_ratio=float(patch_r_ratio),
            r_min_mm=float(r_min_mm),
            r_max_mm=float(r_max_mm),
            R_out_thresh=float(R_out_thresh),
            R_tan_thresh=float(R_tan_thresh),
            constraint_coverage_thresh=0.50,
            squared_displacement_coverage_thresh=0.50,
            target_ds_mm=float(target_ds_mm),
        )
    )
    return cfg


def classify_direction(
    *,
    R_out: float,
    constraint_coverage: float,
    squared_displacement_coverage: float,
    status: str,
    R_out_thresh: float = VAL_R_OUT_THRESH,
    constraint_thresh: float = 0.50,
    sq_thresh: float = 0.50,
) -> Dict[str, Any]:
    nan = float("nan")
    R_tan = (1.0 - float(R_out)) if np.isfinite(R_out) else nan
    ok_calc = status == "ok" and np.isfinite(R_out) and np.isfinite(R_tan)
    ok_cov = (
        ok_calc
        and np.isfinite(constraint_coverage)
        and np.isfinite(squared_displacement_coverage)
        and float(constraint_coverage) >= float(constraint_thresh)
        and float(squared_displacement_coverage) >= float(sq_thresh)
    )
    if not ok_cov:
        return dict(
            evaluable=False,
            label=None,
            reason="not_evaluable",
            R_out=float(R_out) if np.isfinite(R_out) else nan,
            R_tan=R_tan,
        )
    if float(R_out) < float(R_out_thresh):
        label = SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL
    else:
        label = SURFACE_LABEL_SUBSTANTIAL_NORMAL
    return dict(evaluable=True, label=label, reason="", R_out=float(R_out), R_tan=float(R_tan))


def theoretical_R(theta_plane_deg: float) -> Tuple[float, float]:
    th = np.deg2rad(float(theta_plane_deg))
    return float(np.sin(th) ** 2), float(np.cos(th) ** 2)


def make_plane_tri_mesh(
    *,
    x_min_mm: float,
    x_max_mm: float,
    y_min_mm: float,
    y_max_mm: float,
    edge_mm: float = 0.33,
    z_mm: float = 0.0,
    voxel_size_mm: float = 0.5,
):
    import pyvista as pv

    vs = float(voxel_size_mm)
    dx = float(edge_mm)
    xs = np.arange(x_min_mm, x_max_mm + 0.5 * dx, dx)
    ys = np.arange(y_min_mm, y_max_mm + 0.5 * dx, dx)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.full_like(xx, float(z_mm))
    # StructuredGrid → triangulate
    grid = pv.StructuredGrid(
        xx / vs,
        yy / vs,
        zz / vs,
    )
    mesh = grid.extract_surface().triangulate()
    mesh = mesh.compute_normals(point_normals=True, cell_normals=False, inplace=False, auto_orient_normals=True)
               
    n = np.asarray(mesh.point_data["Normals"], float)
    if float(np.mean(n[:, 2])) < 0:
        mesh.point_data["Normals"] = -n
    return mesh


def valley_height_mm(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    *,
    peak1_x_mm: float,
    peak2_x_mm: float,
    peak_halfwidth_mm: float,
    peak_z_mm: float,
    floor_z_mm: float = 0.0,
) -> np.ndarray:
    x = np.asarray(x_mm, float)
    y = np.asarray(y_mm, float)
    w = max(float(peak_halfwidth_mm), 1e-6)
    g1 = np.exp(-((x - float(peak1_x_mm)) ** 2) / w**2)
    g2 = np.exp(-((x - float(peak2_x_mm)) ** 2) / w**2)
    gy = np.exp(-(y**2) / (2.5 * w) ** 2)
    return float(floor_z_mm) + float(peak_z_mm) * (g1 + g2) * gy


def trench_height_mm(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    *,
    peak1_x_mm: float,
    peak2_x_mm: float,
    peak_z_mm: float,
    floor_z_mm: float = 0.0,
    trench_halfwidth_mm: float = 2.0,
    wall_ramp_mm: float = 1.0,
) -> np.ndarray:
    x = np.asarray(x_mm, float)
    _ = y_mm
    mid = 0.5 * (float(peak1_x_mm) + float(peak2_x_mm))
    half = float(trench_halfwidth_mm)
    ramp = max(float(wall_ramp_mm), 0.0)
    zz = np.full_like(x, float(peak_z_mm), dtype=float)
    dist = np.abs(x - mid)
    in_trench = dist <= half
    zz = np.where(in_trench, float(floor_z_mm), zz)
    if ramp > 0:
        edge = (dist > half) & (dist < half + ramp)
        t = (dist - half) / ramp
        zz = np.where(edge, float(floor_z_mm) + t * (float(peak_z_mm) - float(floor_z_mm)), zz)
    return zz


def surface_height_mm(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    *,
    mode: str = "gaussian_peaks",
    peak1_x_mm: float,
    peak2_x_mm: float,
    peak_halfwidth_mm: float = 2.5,
    peak_z_mm: float = 3.0,
    floor_z_mm: float = 0.0,
    trench_halfwidth_mm: float = 2.0,
    wall_ramp_mm: float = 1.0,
) -> np.ndarray:
    if mode == "flat_plateaus_trench":
        return trench_height_mm(
            x_mm,
            y_mm,
            peak1_x_mm=peak1_x_mm,
            peak2_x_mm=peak2_x_mm,
            peak_z_mm=peak_z_mm,
            floor_z_mm=floor_z_mm,
            trench_halfwidth_mm=trench_halfwidth_mm,
            wall_ramp_mm=wall_ramp_mm,
        )
    return valley_height_mm(
        x_mm,
        y_mm,
        peak1_x_mm=peak1_x_mm,
        peak2_x_mm=peak2_x_mm,
        peak_halfwidth_mm=peak_halfwidth_mm,
        peak_z_mm=peak_z_mm,
        floor_z_mm=floor_z_mm,
    )


def make_valley_tri_mesh(
    *,
    x_min_mm: float,
    x_max_mm: float,
    y_min_mm: float,
    y_max_mm: float,
    edge_mm: float = 0.33,
    peak1_x_mm: float = 12.0,
    peak2_x_mm: float = 28.0,
    peak_halfwidth_mm: float = 2.5,
    peak_z_mm: float = 3.0,
    floor_z_mm: float = 0.0,
    voxel_size_mm: float = 0.5,
    peak_y_mm: Optional[float] = None,
    mode: str = "gaussian_peaks",
    trench_halfwidth_mm: float = 2.0,
    wall_ramp_mm: float = 1.0,
):
    import pyvista as pv

    _ = peak_y_mm
    vs = float(voxel_size_mm)
    dx = float(edge_mm)
    xs = np.arange(x_min_mm, x_max_mm + 0.5 * dx, dx)
    ys = np.arange(y_min_mm, y_max_mm + 0.5 * dx, dx)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = surface_height_mm(
        xx,
        yy,
        mode=mode,
        peak1_x_mm=peak1_x_mm,
        peak2_x_mm=peak2_x_mm,
        peak_halfwidth_mm=peak_halfwidth_mm,
        peak_z_mm=peak_z_mm,
        floor_z_mm=floor_z_mm,
        trench_halfwidth_mm=trench_halfwidth_mm,
        wall_ramp_mm=wall_ramp_mm,
    )
    grid = pv.StructuredGrid(xx / vs, yy / vs, zz / vs)
    mesh = grid.extract_surface().triangulate()
    mesh = mesh.compute_normals(point_normals=True, cell_normals=False, inplace=False, auto_orient_normals=True)
    return mesh


def centerline_along_x(
    *,
    L_mm: float,
    ds_mm: float,
    y_mm: float = 0.0,
    z_mm: float = 0.5,
    voxel_size_mm: float = 0.5,
) -> Dict[str, np.ndarray]:
    vs = float(voxel_size_mm)
    xs = np.arange(0.0, float(L_mm) + 0.5 * float(ds_mm), float(ds_mm))
    P_mm = np.column_stack([xs, np.full_like(xs, y_mm), np.full_like(xs, z_mm)])
    P_vox = P_mm / vs
    return dict(P_mm=P_mm, P_vox=P_vox, x_mm=xs, ds_mm=float(ds_mm), voxel_size_mm=vs)


def centerline_follow_valley_surface(
    *,
    L_mm: float,
    ds_mm: float,
    offset_mm: float,
    peak1_x_mm: float,
    peak2_x_mm: float,
    peak_halfwidth_mm: float,
    peak_z_mm: float,
    floor_z_mm: float = 0.0,
    y_mm: float = 0.0,
    voxel_size_mm: float = 0.5,
    z_mode: str = "follow",
    z_const_mm: Optional[float] = None,
    mode: str = "gaussian_peaks",
    trench_halfwidth_mm: float = 2.0,
    wall_ramp_mm: float = 1.0,
) -> Dict[str, np.ndarray]:
    vs = float(voxel_size_mm)
    xs = np.arange(0.0, float(L_mm) + 0.5 * float(ds_mm), float(ds_mm))
    ys = np.full_like(xs, float(y_mm))
    z_surf = surface_height_mm(
        xs,
        ys,
        mode=mode,
        peak1_x_mm=peak1_x_mm,
        peak2_x_mm=peak2_x_mm,
        peak_halfwidth_mm=peak_halfwidth_mm,
        peak_z_mm=peak_z_mm,
        floor_z_mm=floor_z_mm,
        trench_halfwidth_mm=trench_halfwidth_mm,
        wall_ramp_mm=wall_ramp_mm,
    )
    if z_mode == "const":
        zz = np.full_like(xs, float(z_const_mm if z_const_mm is not None else peak_z_mm + offset_mm))
    else:
        zz = z_surf + float(offset_mm)
    P_mm = np.column_stack([xs, ys, zz])
    return dict(
        P_mm=P_mm,
        P_vox=P_mm / vs,
        x_mm=xs,
        z_surf_mm=z_surf,
        ds_mm=float(ds_mm),
        voxel_size_mm=vs,
        z_mode=z_mode,
        mesh_mode=mode,
    )


def known_periodic_displacement(
    x_mm: np.ndarray,
    *,
    amplitude_mm: float,
    lambda_mm: float,
    phase: float,
    theta_plane_deg: float,
    voxel_size_mm: float = 0.5,
) -> np.ndarray:
    vs = float(voxel_size_mm)
    th = np.deg2rad(float(theta_plane_deg))
    s = np.sin(2.0 * np.pi * np.asarray(x_mm, float) / float(lambda_mm) + float(phase))
    direction = np.array([0.0, np.cos(th), np.sin(th)], float)
    d_mm = (float(amplitude_mm) * s)[:, None] * direction[None, :]
    return d_mm / vs


def known_periodic_displacement_perp_T(
    x_mm: np.ndarray,
    T_vox: np.ndarray,
    *,
    amplitude_mm: float,
    lambda_mm: float,
    phase: float,
    theta_plane_deg: float,
    voxel_size_mm: float = 0.5,
) -> np.ndarray:
    vs = float(voxel_size_mm)
    th = np.deg2rad(float(theta_plane_deg))
    pref = np.array([0.0, np.cos(th), np.sin(th)], float)
    T = np.asarray(T_vox, float)
    n = len(T)
    if n == 0:
        return np.zeros((0, 3), dtype=float)
    dirs = np.zeros((n, 3), dtype=float)
    for i in range(n):
        ti = T[i]
        nt = float(np.linalg.norm(ti))
        if nt <= 0:
            ti = np.array([1.0, 0.0, 0.0])
        else:
            ti = ti / nt
        v = pref - float(np.dot(pref, ti)) * ti
        nv = float(np.linalg.norm(v))
        if nv <= 1e-12:
                                      
            helper = np.array([0.0, 1.0, 0.0]) if abs(ti[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
            v = helper - float(np.dot(helper, ti)) * ti
            nv = float(np.linalg.norm(v))
        dirs[i] = v / max(nv, 1e-12)
    s = np.sin(2.0 * np.pi * np.asarray(x_mm, float) / float(lambda_mm) + float(phase))
    d_mm = (float(amplitude_mm) * s)[:, None] * dirs
    return d_mm / vs


def unit_tangents_from_polyline(P_vox: np.ndarray) -> np.ndarray:
    P = np.asarray(P_vox, float)
    n = len(P)
    T = np.zeros((n, 3), dtype=float)
    if n == 0:
        return T
    if n == 1:
        T[0] = np.array([1.0, 0.0, 0.0])
        return T
    d = np.diff(P, axis=0)
    nn = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.where(nn > 0, nn, 1.0)
    T[0] = d[0]
    T[-1] = d[-1]
    if n > 2:
        mid = d[:-1] + d[1:]
        mn = np.linalg.norm(mid, axis=1, keepdims=True)
        T[1:-1] = mid / np.where(mn > 0, mn, 1.0)
              
    for i in range(n):
        if np.linalg.norm(T[i]) <= 0:
            T[i] = T[i - 1] if i > 0 else np.array([1.0, 0.0, 0.0])
        else:
            T[i] = T[i] / np.linalg.norm(T[i])
    return T


def constant_unit_tangent(n_points: int, direction: Sequence[float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    v = np.asarray(direction, float).ravel()
    nrm = np.linalg.norm(v)
    if nrm <= 0:
        raise ValueError("direction must be non-zero")
    v = v / nrm
    return np.tile(v, (int(n_points), 1))


def apply_rigid(
    P_vox: np.ndarray,
    d_vox: np.ndarray,
    mesh,
    R: np.ndarray,
    t_vox: Optional[np.ndarray] = None,
    T_vox: Optional[np.ndarray] = None,
):
    import pyvista as pv

    R = np.asarray(R, float).reshape(3, 3)
    t = np.zeros(3) if t_vox is None else np.asarray(t_vox, float).ravel()
    P2 = (P_vox @ R.T) + t
    d2 = d_vox @ R.T
    pts = np.asarray(mesh.points, float)
    pts2 = (pts @ R.T) + t
    n = np.asarray(mesh.point_data["Normals"], float)
    n2 = n @ R.T
    nn = np.linalg.norm(n2, axis=1, keepdims=True)
    n2 = n2 / np.where(nn > 0, nn, 1.0)
    mesh2 = pv.PolyData(pts2, mesh.faces)
    mesh2.point_data["Normals"] = n2
    if T_vox is None:
        return P2, d2, mesh2
    T2 = np.asarray(T_vox, float) @ R.T
    tn = np.linalg.norm(T2, axis=1, keepdims=True)
    T2 = T2 / np.where(tn > 0, tn, 1.0)
    return P2, d2, mesh2, T2


def random_rotation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    a = rng.normal(size=3)
    a /= np.linalg.norm(a)
    ang = float(rng.uniform(0.2, 1.2))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], float)
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return R


def patch_radius_for_radius_mm(radius_mm: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    vs = float(cfg["voxel_size"])
    return compute_patch_radius_vox(
        float(radius_mm) / vs,
        patch_r_ratio=float(cfg["patch_r_ratio"]),
        r_min_mm=float(cfg["r_min_mm"]),
        r_max_mm=float(cfg["r_max_mm"]),
        voxel_size=vs,
    )


def run_surface_on_arrays(
    P_vox: np.ndarray,
    d_vox: np.ndarray,
    mesh,
    cfg: Dict[str, Any],
    *,
    radius_mm: float = 0.7,
    T0: Optional[np.ndarray] = None,
    T_frame: Optional[np.ndarray] = None,
    T_obs: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    patch = patch_radius_for_radius_mm(radius_mm, cfg)
    if not patch.get("ok"):
        return dict(ok=False, reason=str(patch.get("reason")), patch=patch)
    settings = surface_settings(cfg)
    ds_vox = float(cfg.get("target_ds_mm", 0.1)) / float(cfg["voxel_size"])
    P = np.asarray(P_vox, float)
    est = estimate_normals(
        P,
        mesh,
        r_patch_vox=float(patch["r_patch_vox"]),
        ds_vox=ds_vox,
        settings=settings,
        verbose=verbose,
    )
    if T0 is None:
        T0 = unit_tangents_from_polyline(P)
    T_use = T_frame if T_frame is not None else T0
    ev = evaluate_displacement(
        np.asarray(d_vox, float),
        est,
        T_frame=T_use,
        T0=T0,
        T_obs=T_obs,
    )
    true_n = np.array([0.0, 0.0, 1.0])                     
    return dict(
        ok=True,
        est=est,
        ev=ev,
        patch=patch,
        ds_vox=ds_vox,
        true_n_default=true_n,
        T0=np.asarray(T0, float),
        T_frame=np.asarray(T_use, float),
    )


def _project_R_on_known_normal(d_vox: np.ndarray, n_true: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    d = np.asarray(d_vox, float)
    nt = np.asarray(n_true, float).ravel()
    nn = np.linalg.norm(nt)
    if nn <= 0 or d.size == 0:
        return float("nan"), float("nan")
    nt = nt / nn
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.size != len(d) or not np.any(m):
            return float("nan"), float("nan")
        d = d[m]
    E = float(np.sum(d * d))
    if E <= 0:
        return float("nan"), float("nan")
    R_out = float(np.sum((d @ nt) ** 2)) / E
    return R_out, 1.0 - R_out


def metrics_vs_truth(
    ev: Dict[str, Any],
    est: Dict[str, Any],
    d_true_vox: np.ndarray,
    n_true: np.ndarray,
    *,
    theta_plane_deg: float,
    d_est_vox: Optional[np.ndarray] = None,
    T0_for_theory: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    R_out_t, R_tan_t = theoretical_R(theta_plane_deg)
    mask = np.asarray(est["mask_eval"], bool)
    n_est = np.asarray(est["normals_full"], float)
    nt = np.asarray(n_true, float).ravel()
    nt = nt / np.linalg.norm(nt)

    R_out_true_mask, R_tan_true_mask = _project_R_on_known_normal(d_true_vox, nt, mask)
    R_out_true_all, R_tan_true_all = _project_R_on_known_normal(d_true_vox, nt, None)

    R_out_est_proj_mask = R_tan_est_proj_mask = float("nan")
    if d_est_vox is not None:
        R_out_est_proj_mask, R_tan_est_proj_mask = _project_R_on_known_normal(d_est_vox, nt, mask)

    axis_errs = []
    for i in np.flatnonzero(mask):
        ni = n_est[i]
        if not np.all(np.isfinite(ni)):
            continue
        nn = np.linalg.norm(ni)
        if nn <= 0:
            continue
        axis_errs.append(unsigned_axis_angle_deg(ni / nn, nt))
    axis_err = float(np.median(axis_errs)) if axis_errs else float("nan")

    R_out = float(ev.get("R_out", np.nan))
    R_tan = float(ev.get("R_tan", np.nan))
    etf = float(ev.get("evaluated_tangential_fraction", np.nan))
    sq = float(ev.get("squared_displacement_coverage", np.nan))
    etf_id = (sq * R_tan) if (np.isfinite(sq) and np.isfinite(R_tan)) else float("nan")

                                                         
    abs_T0_mean = float(ev.get("abs_T0_dot_n_mean", np.nan))
    abs_T0_med = float(ev.get("abs_T0_dot_n_median", np.nan))
    R_ub = float(ev.get("R_out_upper_bound", np.nan))
    within_ub = ev.get("R_out_within_upper_bound", np.nan)
    within_ub_f = float(within_ub) if within_ub is not None and np.isfinite(within_ub) else float("nan")
                                                
    R_ub_theory = float("nan")
    if T0_for_theory is not None:
        Trep = np.asarray(T0_for_theory, float)
        if Trep.ndim == 2 and len(Trep):
            Trep = Trep[0]
        Trep = np.asarray(Trep, float).ravel()
        tn = np.linalg.norm(Trep)
        if tn > 0:
            Trep = Trep / tn
            R_ub_theory = 1.0 - float(np.dot(Trep, nt) ** 2)

    src = np.asarray(est.get("eval_normal_source"), int)
    return dict(
        R_out=R_out,
        R_tan=R_tan,
        R_out_theory=R_out_t,
        R_tan_theory=R_tan_t,
        R_out_true_on_mask=R_out_true_mask,
        R_tan_true_on_mask=R_tan_true_mask,
        R_out_true_all=R_out_true_all,
        R_tan_true_all=R_tan_true_all,
        R_out_est_proj_known_n_on_mask=R_out_est_proj_mask,
        R_tan_est_proj_known_n_on_mask=R_tan_est_proj_mask,
        err_R_out_vs_theory=(R_out - R_out_t) if np.isfinite(R_out) else float("nan"),
        err_R_tan_vs_theory=(R_tan - R_tan_t) if np.isfinite(R_tan) else float("nan"),
        err_R_out_vs_mask_truth=(R_out - R_out_true_mask) if np.isfinite(R_out) and np.isfinite(R_out_true_mask) else float("nan"),
        normal_axis_err_deg_median=axis_err,
        abs_T0_dot_n_mean=abs_T0_mean,
        abs_T0_dot_n_median=abs_T0_med,
        R_out_upper_bound=R_ub,
        R_out_within_upper_bound=within_ub_f,
        R_out_upper_bound_theory_constTn=R_ub_theory,
        pass_R_out_le_upper_bound=bool(np.isfinite(R_out) and np.isfinite(R_ub) and R_out <= R_ub + 1e-9),
        constraint_coverage=float(ev.get("constraint_coverage", np.nan)),
        squared_displacement_coverage=sq,
        cortex_coverage=float(ev.get("cortex_coverage", np.nan)),
        good_coverage=float(ev.get("good_coverage", np.nan)),
        bridged_coverage=float(ev.get("bridged_coverage", np.nan)),
        squared_displacement_coverage_good=float(ev.get("squared_displacement_coverage_good", np.nan)),
        squared_displacement_coverage_bridged=float(ev.get("squared_displacement_coverage_bridged", np.nan)),
        evaluated_tangential_fraction=etf,
        etf_equals_sq_times_R_tan=bool(np.isfinite(etf) and np.isfinite(etf_id) and abs(etf - etf_id) < 1e-12),
        n_eval=int(ev.get("n_eval", 0)),
        n_eval_good=ev.get("n_eval_good", float("nan")),
        n_eval_bridged=ev.get("n_eval_bridged", float("nan")),
        n_points=int(ev.get("n_points_total", 0)),
        status=str(ev.get("status", "")),
        reason=str(ev.get("reason", "")),
        n_src_good=int(np.sum(src == EVAL_NORMAL_SOURCE_GOOD)) if src.size else 0,
        n_src_bridged=int(np.sum(src == EVAL_NORMAL_SOURCE_BRIDGED)) if src.size else 0,
        n_src_none=int(np.sum(src == EVAL_NORMAL_SOURCE_NONE)) if src.size else 0,
    )


def geometric_isotropic_R_tan_mean(
    T: np.ndarray,
    n: np.ndarray,
    *,
    n_dirs: int = 720,
) -> Dict[str, Any]:
    T = np.asarray(T, float).ravel()
    n = np.asarray(n, float).ravel()
    T = T / np.linalg.norm(T)
    n = n / np.linalg.norm(n)
            
    tmp = np.array([1.0, 0.0, 0.0]) if abs(T[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(T, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(T, e1)
    phis = np.linspace(0.0, 2.0 * np.pi, int(n_dirs), endpoint=False)
    rtans = []
    for ph in phis:
        d = np.cos(ph) * e1 + np.sin(ph) * e2
        d = d / np.linalg.norm(d)
        r_out = float(np.dot(d, n) ** 2)
        rtans.append(1.0 - r_out)
                                                                       
    n_par = float(np.dot(n, T))
    n_perp2 = 1.0 - n_par**2
    mean_analytic = 1.0 - 0.5 * n_perp2
    return dict(
        mean_R_tan_numeric=float(np.mean(rtans)),
        mean_R_tan_analytic=float(mean_analytic),
        T_dot_n=n_par,
        angle_T_plane_deg=float(np.degrees(np.arcsin(np.clip(abs(n_par), 0, 1)))),  # T vs plane: asin(|T·n|)
    )


def summarize_row(base: Dict[str, Any], cls: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(
        {
            "evaluable": bool(cls["evaluable"]),
            "direction_label": cls["label"],
            "class_reason": cls["reason"],
            "R_out_thresh_used": VAL_R_OUT_THRESH,
            "R_tan_thresh_used": VAL_R_TAN_THRESH,
        }
    )
    return out
