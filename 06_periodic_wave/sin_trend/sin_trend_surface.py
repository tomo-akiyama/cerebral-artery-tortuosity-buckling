#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend surface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
# Prefer this package directory on sys.path (same basename modules must not collide).
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)

from sin_trend_fitting_helpers import (  # noqa: E402
    EVAL_NORMAL_SOURCE_BRIDGED,
    EVAL_NORMAL_SOURCE_GOOD,
    EVAL_NORMAL_SOURCE_NONE,
)
from sin_trend_paths import SURFACE_METHOD_REVISION, sha256_array, sha256_json  # noqa: E402

NORMAL_SOURCE_OBSERVED = 1                                    
NORMAL_SOURCE_CARRIED = 2                                    
NORMAL_SOURCE_DEFAULT = 3                                           

                            
EVAL_SOURCE_NONE = EVAL_NORMAL_SOURCE_NONE
EVAL_SOURCE_GOOD = EVAL_NORMAL_SOURCE_GOOD
EVAL_SOURCE_BRIDGED = EVAL_NORMAL_SOURCE_BRIDGED


def surface_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        coh_thr=float(cfg["coh_thr"]),
        stride=int(cfg["stride"]),
        min_neighbors=int(cfg["min_neighbors"]),
                                                                    
        gap_fill_size=int(cfg.get("gap_fill_size", 3)),
        bridge_max_gap_mm=float(cfg.get("bridge_max_gap_mm", 5.0)),
                                                 
        bridge_axis_max_deg=float(cfg.get("bridge_axis_max_deg", 45.0)),
        voxel_size=float(cfg["voxel_size"]),
        patch_r_ratio=float(cfg["patch_r_ratio"]),
        r_min_mm=float(cfg["r_min_mm"]),
        r_max_mm=float(cfg["r_max_mm"]),
        clip_margin_vox=10,
        surface_method_revision=SURFACE_METHOD_REVISION,
    )


def normals_cache_key(P_trim: np.ndarray, mesh_identity: Dict[str, Any], r_patch_vox: float, ds_vox: float, settings: Dict[str, Any]) -> str:
    return sha256_json(
        {
            "P_trim": sha256_array(P_trim),
            "mesh": mesh_identity,
            "r_patch_vox": float(r_patch_vox),
            "ds_vox": float(ds_vox),
            "settings": settings,
        }
    )


def mesh_identity_from_path(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    st = p.stat()
    return {"path": str(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


# ---------------------------------------------------------------------------
                 
# ---------------------------------------------------------------------------
def estimate_normals(
    P_trim: np.ndarray,
    cropped_mesh,
    *,
    r_patch_vox: float,
    ds_vox: float,
    settings: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    from sin_trend_fitting_helpers import estimate_n_eff_along_centerline

    P_trim = np.asarray(P_trim, float)
    est = estimate_n_eff_along_centerline(
        P_trim,
        cropped_mesh,
        float(r_patch_vox),
        stride=int(settings["stride"]),
        min_neighbors=int(settings["min_neighbors"]),
        coh_thr=float(settings["coh_thr"]),
        gap_fill_size=int(settings.get("gap_fill_size", 3)),
        ds_vox=float(ds_vox),
        voxel_size_mm=float(settings["voxel_size"]),
        bridge_max_gap_mm=float(settings.get("bridge_max_gap_mm", 5.0)),
        bridge_axis_max_deg=float(settings.get("bridge_axis_max_deg", 45.0)),
        verbose=verbose,
    )
    anchor_idx = np.asarray(est["anchor_indices"], int)
    anchor_coh = np.asarray(est["anchor_coherence"], float)
    anchor_observed = np.asarray(est.get("anchor_observed"), bool)
    anchor_good = np.asarray(est.get("anchor_good"), bool)
    if anchor_observed.size != len(anchor_idx):
        anchor_observed = np.isfinite(anchor_coh)
    if anchor_good.size != len(anchor_idx):
        anchor_good = anchor_observed & np.isfinite(anchor_coh) & (anchor_coh >= float(settings["coh_thr"]))
    anchor_has_cortex = np.asarray(est.get("anchor_has_cortex", anchor_observed), bool)
    if anchor_has_cortex.size != len(anchor_idx):
        anchor_has_cortex = anchor_observed.copy()
    anchor_constraint_raw = anchor_good.copy()
                              
    has_constraint_full = np.asarray(est["has_constraint_full"], bool)
    anchor_constraint_filled = has_constraint_full[anchor_idx]
                                      
    anchor_source = np.full(len(anchor_idx), NORMAL_SOURCE_OBSERVED, int)
    seen_obs = False
    for i, obs in enumerate(anchor_observed):
        if obs:
            seen_obs = True
            anchor_source[i] = NORMAL_SOURCE_OBSERVED
        else:
            anchor_source[i] = NORMAL_SOURCE_CARRIED if seen_obs else NORMAL_SOURCE_DEFAULT
    n_points = int(est["n_points"])
    normals_full = np.asarray(est["normals_full"], float)
    eval_normal_source = np.asarray(
        est.get("eval_normal_source", np.full(n_points, EVAL_SOURCE_NONE, dtype=int)), int
    )
    mask_eval = np.asarray(est["mask_eval"], bool)
    bracket_ok = np.asarray(est.get("bracket_observed_both", mask_eval), bool)
    constraint_raw_full = np.asarray(
        est.get("has_constraint_full_raw", np.zeros(n_points, dtype=bool)), bool
    )

    out = dict(
        normals_full=normals_full,
        normals_display_full=np.asarray(est.get("normals_display_full", normals_full), float),
        coherence_full=np.asarray(est["coherence_full"], float),
        observed_full=np.asarray(est.get("observed_full", np.zeros(n_points, dtype=bool)), bool),
        has_cortex_full=np.asarray(est["has_cortex_full"], bool),
        has_constraint_full_filled=has_constraint_full,
        has_constraint_full_raw=constraint_raw_full,
        mask_eval=mask_eval,
        bracket_observed_both=bracket_ok,
        eval_normal_source=eval_normal_source,
        cortex_coverage=float(est["cortex_coverage"]),
        constraint_coverage=float(np.mean(mask_eval)) if n_points else float("nan"),
        n_eval_good=int(np.sum((eval_normal_source == EVAL_SOURCE_GOOD) & mask_eval)),
        n_eval_bridged=int(np.sum((eval_normal_source == EVAL_SOURCE_BRIDGED) & mask_eval)),
        n_eval_none=int(np.sum(eval_normal_source == EVAL_SOURCE_NONE)),
        anchor_indices=anchor_idx,
        anchor_normals=np.asarray(est["anchor_normals"], float),
        anchor_normals_good=np.asarray(est.get("anchor_normals_good", est["anchor_normals"]), float),
        anchor_normals_display=np.asarray(est.get("anchor_normals_display", est["anchor_normals"]), float),
        anchor_coherence=anchor_coh,
        anchor_observed=anchor_observed,
        anchor_good=anchor_good,
        anchor_has_cortex=anchor_has_cortex,
        anchor_constraint_raw=anchor_constraint_raw,
        anchor_constraint_filled=anchor_constraint_filled,
        anchor_normal_source=anchor_source,
        n_points=n_points,
        n_anchors=int(len(anchor_idx)),
        n_anchors_observed_normal=int(anchor_observed.sum()),
        n_anchors_good=int(anchor_good.sum()),
        r_patch_vox=float(r_patch_vox),
        ds_vox=float(ds_vox),
        bridge_max_gap_mm=float(est.get("bridge_max_gap_mm", settings.get("bridge_max_gap_mm", 5.0))),
        bridge_axis_max_deg=float(
            est.get("bridge_axis_max_deg", settings.get("bridge_axis_max_deg", 45.0))
        ),
        settings=dict(settings),
        surface_method_revision=SURFACE_METHOD_REVISION,
        status="ok",
    )
                                               
    P_disp = np.asarray(P_trim, float)
    if n_points <= 1:
        s_full = np.zeros(n_points, dtype=float)
    else:
        seg = np.linalg.norm(np.diff(P_disp, axis=0), axis=1)
        seg = np.where(np.isfinite(seg), seg, 0.0)
        s_full = np.concatenate([[0.0], np.cumsum(seg)])
        if not np.all(np.isfinite(s_full)) or float(s_full[-1]) <= 0.0:
            s_full = np.arange(n_points, dtype=float) * float(ds_vox)
    s_anchor = s_full[anchor_idx]
    nearest = np.abs(s_full[:, None] - s_anchor[None, :]).argmin(axis=1)
    out["normal_source_full"] = anchor_source[nearest]
    out["s_full_vox"] = s_full
    return out


def save_normals_cache(path: Path, key: str, est: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in est.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in est.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path, cache_key=np.array(key), scalars_json=np.array(json.dumps(scalars, default=str)), **arrays)


def load_normals_cache(path: Path, key: str) -> Optional[Dict[str, Any]]:
    if not Path(path).is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["cache_key"]) != key:
                return None
            out: Dict[str, Any] = json.loads(str(z["scalars_json"]))
            for k in z.files:
                if k in {"cache_key", "scalars_json"}:
                    continue
                out[k] = z[k]
            return out
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
                                 
# ---------------------------------------------------------------------------
def evaluate_displacement(
    d_sin: np.ndarray,
    est: Dict[str, Any],
    *,
    g1_3d: Optional[np.ndarray] = None,
    T_frame: Optional[np.ndarray] = None,
    T0: Optional[np.ndarray] = None,
    T_obs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    d = np.asarray(d_sin, float)
    n_all = np.asarray(est["normals_full"], float)
    mask = np.asarray(est["mask_eval"], bool)
    if d.shape != n_all.shape:
        raise ValueError(f"d_sin shape {d.shape} != normals shape {n_all.shape}")
    n_points = int(len(d))
    nan = float("nan")
    out: Dict[str, Any] = dict(
        R_out=nan,
        R_tan=nan,
        R_dir=nan,
        squared_displacement_coverage=nan,
        total_energy=nan,
        energy_constrained=nan,
        perp_energy=nan,
        evaluated_tangential_fraction=nan,
        R_out_upper_bound=nan,
        R_out_within_upper_bound=nan,
        abs_T0_dot_n_mean=nan,
        abs_T0_dot_n_median=nan,
        abs_Tobs_dot_n_mean=nan,
        abs_Tobs_dot_n_median=nan,
        n_points_total=n_points,
        n_eval=int(mask.sum()),
        n_eval_good=nan,
        n_eval_bridged=nan,
        good_coverage=nan,
        bridged_coverage=nan,
        squared_displacement_coverage_good=nan,
        squared_displacement_coverage_bridged=nan,
        cortex_coverage=float(est["cortex_coverage"]),
        constraint_coverage=float(est["constraint_coverage"]),
        status="ok",
        reason="",
        r_out_local=np.full(n_points, np.nan),
    )

                                                             
    src_raw = est.get("eval_normal_source", None)
    mask_g = mask_b = None
    if src_raw is not None:
        src = np.asarray(src_raw, int).ravel()
        if src.size != n_points:
            raise ValueError(f"eval_normal_source length {src.size} != n_points {n_points}")
        mask_g = mask & (src == EVAL_SOURCE_GOOD)
        mask_b = mask & (src == EVAL_SOURCE_BRIDGED)
        n_g = int(np.sum(mask_g))
        n_b = int(np.sum(mask_b))
        out["n_eval_good"] = n_g
        out["n_eval_bridged"] = n_b
        if n_points > 0:
            out["good_coverage"] = float(n_g) / float(n_points)
            out["bridged_coverage"] = float(n_b) / float(n_points)

    E_tot = float(np.sum(d * d))
    out["total_energy"] = E_tot
    if src_raw is not None and np.isfinite(E_tot) and E_tot > 0.0 and mask_g is not None and mask_b is not None:
        d2 = np.einsum("ij,ij->i", d, d)
        out["squared_displacement_coverage_good"] = float(np.sum(d2[mask_g])) / E_tot
        out["squared_displacement_coverage_bridged"] = float(np.sum(d2[mask_b])) / E_tot
                                                                                 

    if not np.isfinite(E_tot) or E_tot <= 0.0:
        out.update(status="zero_displacement_energy", reason="total periodic displacement energy <= 0")
        return out
    if not np.any(mask):
        out.update(status="no_evaluable_points", reason="evaluation mask empty", squared_displacement_coverage=0.0)
        return out
    dm = d[mask]
    nm = n_all[mask]
                          
    nn = np.linalg.norm(nm, axis=1, keepdims=True)
    if np.any(~np.isfinite(nn)) or np.any(nn <= 0):
        out.update(status="invalid_normals", reason="non-unit / non-finite normals in evaluation region")
        return out
    nm = nm / nn
    dd = np.einsum("ij,ij->i", dm, nm)
    E_c = float(np.sum(dm * dm))
    E_p = float(np.sum(dd * dd))
    out["energy_constrained"] = E_c
    out["perp_energy"] = E_p
    out["squared_displacement_coverage"] = E_c / E_tot
    if E_c <= 0.0:
        out.update(status="zero_constrained_energy", reason="periodic displacement is zero on evaluation region; R_out undefined")
        return out
    R_out = E_p / E_c
    out["R_out"] = R_out
    out["R_tan"] = 1.0 - R_out                    
    out["evaluated_tangential_fraction"] = out["squared_displacement_coverage"] * out["R_tan"]

                              
    d2 = np.einsum("ij,ij->i", d, d)
    dot_all = np.einsum("ij,ij->i", d, n_all / np.where(np.linalg.norm(n_all, axis=1, keepdims=True) > 0, np.linalg.norm(n_all, axis=1, keepdims=True), 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rl = np.where(d2 > 0, (dot_all * dot_all) / d2, np.nan)
    rl[~mask] = np.nan
    out["r_out_local"] = rl

    if g1_3d is not None:
        g = np.asarray(g1_3d, float)[mask]
        cos2 = np.einsum("ij,ij->i", g, nm) ** 2
        out["R_dir"] = float(np.mean(cos2)) if len(cos2) else float("nan")
    w = np.einsum("ij,ij->i", dm, dm)
    if T0 is not None:
        t0 = np.asarray(T0, float)[mask]
        a0 = np.abs(np.einsum("ij,ij->i", t0, nm))
        out["abs_T0_dot_n_mean"] = float(np.mean(a0))
        out["abs_T0_dot_n_median"] = float(np.median(a0))
    if T_obs is not None:
        t = np.asarray(T_obs, float)[mask]
        a = np.abs(np.einsum("ij,ij->i", t, nm))
        out["abs_Tobs_dot_n_mean"] = float(np.mean(a))
        out["abs_Tobs_dot_n_median"] = float(np.median(a))
                                                                                  
    T_ub = T_frame
    if T_ub is None:
        T_ub = T_obs if T_obs is not None else T0
    if T_ub is not None:
        t = np.asarray(T_ub, float)[mask]
        a = np.abs(np.einsum("ij,ij->i", t, nm))
        ub = float(np.sum(w * (1.0 - a**2)) / np.sum(w)) if np.sum(w) > 0 else float("nan")
        out["R_out_upper_bound"] = ub
        out["R_out_within_upper_bound"] = float(R_out <= ub + 1e-9) if np.isfinite(ub) else float("nan")
    return out


# ---------------------------------------------------------------------------
                        
# ---------------------------------------------------------------------------
_MESH_CACHE: Dict[str, Any] = {}


def load_brain_mesh(path: Path | str):
    import pyvista as pv

    p = Path(path)
    ident = mesh_identity_from_path(p)
    key = json.dumps(ident, sort_keys=True)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key], ident
    mesh = pv.read(str(p))
    if "Normals" not in mesh.point_data:
        mesh = mesh.compute_normals(point_normals=True, cell_normals=False, inplace=False)
    _MESH_CACHE.clear()                 
    _MESH_CACHE[key] = mesh
    return mesh, ident


def crop_mesh(P_full_vox: np.ndarray, mesh, margin: int = 10):
    from sin_trend_fitting_helpers import clip_mesh_to_coords

    return clip_mesh_to_coords(np.asarray(P_full_vox, float), mesh, margin=margin)


def run_surface_evaluation(
    *,
    P_full_vox: np.ndarray,
    P_trim: np.ndarray,
    d_sin: np.ndarray,
    ds_vox: float,
    r_patch_vox: float,
    mesh_path: Path,
    settings: Dict[str, Any],
    cache_path: Optional[Path] = None,
    g1_3d: Optional[np.ndarray] = None,
    T_frame: Optional[np.ndarray] = None,
    T0: Optional[np.ndarray] = None,
    T_obs: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    # The cropped mesh must contain the entire requested search sphere, even
    # when a sensitivity run increases r_max_mm or uses a smaller voxel size.
    settings = dict(settings)
    settings["clip_margin_vox"] = max(
        int(settings.get("clip_margin_vox", 10)), int(np.ceil(float(r_patch_vox)))
    )
    ident = mesh_identity_from_path(mesh_path)
    key = normals_cache_key(P_trim, ident, r_patch_vox, ds_vox, settings)
    est = load_normals_cache(cache_path, key) if cache_path is not None else None
    cache_status = "hit" if est is not None else "miss"
    if est is None:
        mesh, _ = load_brain_mesh(mesh_path)
        cropped = crop_mesh(P_full_vox, mesh, margin=int(settings.get("clip_margin_vox", 10)))
        est = estimate_normals(P_trim, cropped, r_patch_vox=r_patch_vox, ds_vox=ds_vox, settings=settings, verbose=verbose)
        if cache_path is not None:
            try:
                save_normals_cache(cache_path, key, est)
                cache_status = "miss_saved"
            except Exception as exc:  # noqa: BLE001
                cache_status = f"miss_save_failed:{exc}"
    ev = evaluate_displacement(d_sin, est, g1_3d=g1_3d, T_frame=T_frame, T0=T0, T_obs=T_obs)
    return ev, est, cache_status
