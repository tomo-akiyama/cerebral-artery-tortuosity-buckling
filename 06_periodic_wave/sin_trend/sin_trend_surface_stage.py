#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend surface stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
import sys

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_classify import (  # noqa: E402
    SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL,
    SURFACE_LABEL_SUBSTANTIAL_NORMAL,
    surface_eligible_from_row,
)
from sin_trend_io_utils import compute_patch_radius_vox  # noqa: E402
from sin_trend_paths import (  # noqa: E402
    SURFACE_METHOD_REVISION,
    atomic_to_csv,
    brain_mesh_path,
    normals_cache_dir,
    per_case_csv_path,
    surface_per_case_csv_path,
    resolve_surface_config,
    pointwise_dir,
    project_root_from_here,
    sha256_file,
    to_python_scalar,
    validate_run_id,
    validate_frame_reference,
    validate_surface_tag,
    write_surface_config,
)
from sin_trend_surface import run_surface_evaluation, surface_settings  # noqa: E402

                               
SURFACE_RESULT_KEYS = (
    "patch_r_ratio",
    "r_patch_definition",
    "r_patch_mm_raw",
    "r_patch_mm",
    "r_patch_vox",
    "coh_thr",
    "constraint_coverage_thresh",
    "squared_displacement_coverage_thresh",
    "R_out_thresh",
    "R_tan_thresh",
    "bridge_max_gap_mm",
    "bridge_axis_max_deg",
    "min_neighbors",
    "surface_method_revision",
    "normals_cache_status",
    "surface_tag",
    "surface_displacement_representation",
    "R_out_surface",
    "R_tan_surface",
    "R_dir_surface",
    "R_dir_status",
    "R_out_latent",
    "R_tan_latent",
    "squared_displacement_coverage_latent",
    "R_out_response",
    "R_tan_response",
    "squared_displacement_coverage_response",
    "cortex_coverage",
    "constraint_coverage",
    "squared_displacement_coverage",
    "total_energy",
    "energy_constrained",
    "perp_energy",
    "evaluated_tangential_fraction",
    "R_out_upper_bound",
    "R_out_within_upper_bound",
    "abs_T0_dot_n_mean",
    "abs_T0_dot_n_median",
    "abs_Tobs_dot_n_mean",
    "abs_Tobs_dot_n_median",
    "n_points_total",
    "n_eval",
    "n_eval_good",
    "n_eval_bridged",
    "good_coverage",
    "bridged_coverage",
    "squared_displacement_coverage_good",
    "squared_displacement_coverage_bridged",
    "n_anchors",
    "n_anchors_observed_normal",
    "surface_evaluation_status",
    "surface_evaluation_reason",
    "response_surface_diag_reason",
    "surface_config_hash",
    "surface_code_hash",
    "surface_source_measurement_csv_sha256",
    "surface_source_pointwise_sha256",
    "surface_source_pointwise_path",
    "surface_ds_vox",
    "viz_surface_html_status",
    "viz_surface_html_path",
)

SURFACE_CLASSIFICATION_KEYS = (
    "surface_eligible", "surface_evaluable", "surface_coverage_ok", "surface_inplane_ok",
    "surface_predominantly_tangential", "surface_substantial_normal_possible",
    "R_tan_ok", "R_tan_thresh_used", "R_out_thresh_used", "thresh_pair_inconsistent",
    "core_inplane", "surface_constraint", "plane_class",
)
# Historical surface aliases must not survive a failed/new evaluation.
LEGACY_SURFACE_KEYS = (
    "R_out", "R_tan", "R_dir", "energy_coverage", "energy_coverage_latent",
    "energy_coverage_response", "surface_out_of_plane_possible", "surface_eval_ok",
)


def _is_surface_key(key: str) -> bool:
    return key in SURFACE_RESULT_KEYS + SURFACE_CLASSIFICATION_KEYS + LEGACY_SURFACE_KEYS


def _clear_surface(row: Dict[str, Any], tag: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: (np.nan if _is_surface_key(k) else v) for k, v in row.items()}
    for key in SURFACE_RESULT_KEYS + LEGACY_SURFACE_KEYS:
        out[key] = np.nan
    out.update(
        surface_tag=tag, surface_method_revision=SURFACE_METHOD_REVISION,
        surface_evaluation_status="not_run", surface_evaluation_reason="",
        response_surface_diag_reason="",
        viz_surface_html_status="not_requested", viz_surface_html_path="",
        coh_thr=float(cfg.get("coh_thr", .75)),
        constraint_coverage_thresh=float(cfg.get("constraint_coverage_thresh", .5)),
        squared_displacement_coverage_thresh=float(cfg.get("squared_displacement_coverage_thresh", .5)),
        R_out_thresh=float(cfg.get("R_out_thresh", .25)),
        R_tan_thresh=1.0 - float(cfg.get("R_out_thresh", .25)),
        bridge_max_gap_mm=float(cfg.get("bridge_max_gap_mm", 5)),
        bridge_axis_max_deg=float(cfg.get("bridge_axis_max_deg", 45)),
        min_neighbors=int(cfg.get("min_neighbors", 30)),
    )
    return out


def _surface_classification_only(row: Dict[str, Any]) -> Dict[str, Any]:
    """Retain measured fit/Core decisions; update only their surface-dependent outputs."""
    out = dict(row)
    surf = surface_eligible_from_row(out)
    for key in SURFACE_CLASSIFICATION_KEYS:
        if key in surf:
            out[key] = int(surf[key]) if isinstance(surf[key], (bool, np.bool_)) else surf[key]
    saved_core = out.get("core_periodic", out.get("core_all"))
    is_core = str(saved_core).strip().lower() in {"1", "1.0", "true"}
    out["core_inplane"] = int(is_core and surf["surface_eligible"])
    label = None
    if is_core and surf["surface_eligible"]:
        label = SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL
    elif is_core and surf["surface_out_of_plane_possible"]:
        label = SURFACE_LABEL_SUBSTANTIAL_NORMAL
    out["surface_constraint"] = label
    out["plane_class"] = label
                                      
    for key in [k for k in out if str(k).startswith("core_inplane_r2_")]:
        out.pop(key, None)
    return out


def _integer_id(value: Any, name: str) -> int:
    try:
        f = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"invalid {name}: {value!r}") from None
    if not np.isfinite(f) or f != int(f):
        raise ValueError(f"invalid {name}: {value!r}")
    return int(f)


def _validate_pointwise(pw: Dict[str, np.ndarray], scalars: Dict[str, Any], row: Dict[str, Any], *, run_id: str, frame_reference: str, voxel_size: float) -> None:
    for key, expected in (("run_id", run_id), ("frame_reference", frame_reference), ("directory_id", str(row.get("directory_id", "")))):
        for source in (row, scalars):
            got = source.get(key)
            if got is not None and not (isinstance(got, float) and np.isnan(got)) and str(got) != expected:
                raise ValueError(f"pointwise/measurement {key} mismatch")
    for key in ("segment_id", "segment_index"):
        if key in scalars and _integer_id(scalars[key], key) != _integer_id(row.get(key), key):
            raise ValueError(f"pointwise {key} mismatch")
    for source in (row, scalars):
        value = source.get("voxel_size_mm")
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            if not np.isfinite(float(value)) or float(value) != voxel_size:
                raise ValueError("pointwise/measurement voxel_size_mm mismatch")
    if "P" not in pw or ("d_latent" not in pw and "d_sin" not in pw):
        raise ValueError("P and d_latent/d_sin are required")
    P = np.asarray(pw["P"], float)
    if P.ndim != 2 or P.shape[1:] != (3,) or len(P) < 2 or not np.isfinite(P).all():
        raise ValueError("invalid P shape/values")
    for key in ("d_latent", "d_sin", "d_response", "T0", "T", "T_obs"):
        if key in pw:
            a = np.asarray(pw[key], float)
            if a.shape != P.shape or not np.isfinite(a).all():
                raise ValueError(f"invalid {key} shape/values")
    full = np.asarray(pw.get("P_full", P), float)
    if full.ndim != 2 or full.shape[1:] != (3,) or len(full) < len(P) or not np.isfinite(full).all():
        raise ValueError("invalid P_full shape/values")
    if "index_full_to_trim" in pw:
        idx = np.asarray(pw["index_full_to_trim"])
        if idx.shape != (len(P),) or not np.isfinite(idx).all() or not np.equal(idx, np.floor(idx)).all():
            raise ValueError("invalid index_full_to_trim")
        idx = idx.astype(int)
        if np.any(idx < 0) or np.any(idx >= len(full)) or not np.allclose(full[idx], P, rtol=0, atol=1e-8):
            raise ValueError("P/index_full_to_trim does not match P_full")
    if len(full) < 2 or not np.isfinite(np.median(np.linalg.norm(np.diff(full, axis=0), axis=1))) or np.median(np.linalg.norm(np.diff(full, axis=0), axis=1)) <= 0:
        raise ValueError("pointwise geometry has no positive median spacing")



def load_pointwise_npz(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files if k != "scalars_json"}
        scalars: Dict[str, Any] = {}
        if "scalars_json" in z.files:
            raw = z["scalars_json"]
            text = raw.item() if getattr(raw, "shape", ()) == () else str(raw)
            scalars = json.loads(str(text))
            if not isinstance(scalars, dict):
                raise ValueError("pointwise scalars_json must be a JSON object")
    return arrays, scalars


def pointwise_path_for_row(
    row: Dict[str, Any],
    *,
    run_id: str,
    frame_reference: str,
    project_root: Path,
) -> Optional[Path]:
    # Resolve only the requested run/frame/segment; never follow another run's pointer.
    did = str(row.get("directory_id", ""))
    if not did or Path(did).name != did or did in {".", ".."}:
        raise ValueError("invalid directory_id")
    sid = _integer_id(row.get("segment_id"), "segment_id")
    idx = _integer_id(row.get("segment_index"), "segment_index")
    cand = (pointwise_dir(run_id, frame_reference, project_root) / f"{did}_seg{sid}_idx{idx}_pointwise.npz").resolve()
    p = row.get("pointwise_npz_path")
    if p is not None and str(p).strip() not in {"", "nan", "None"}:
        supplied = Path(str(p))
        if not supplied.is_absolute():
            supplied = project_root / supplied
        if supplied.resolve() != cand:
            raise ValueError("pointwise_npz_path does not belong to the requested run/frame/segment")
    return cand if cand.is_file() else None


def radius_median_vox_from_row(row: Dict[str, Any], voxel_size: float) -> float:
    r_mm = row.get("radius_median_mm")
    try:
        if r_mm is not None and np.isfinite(float(r_mm)) and float(r_mm) > 0:
            return float(r_mm) / float(voxel_size)
    except (TypeError, ValueError):
        pass
    return float("nan")



def _save_surface_stage_plot(
    row: Dict[str, Any], pw: Dict[str, np.ndarray], est: Dict[str, Any], *,
    run_id: str, frame_reference: str, surface_tag: str, project_root: Path,
    mesh_path: Path,
) -> Tuple[str, str]:
    """Render from copies of saved geometry and the newly computed evaluation normals."""
    try:
        from sin_trend_visualize import save_surface_constraint_html

        view = {key: np.array(value, copy=True) for key, value in pw.items()}
        if "P0_full" not in view:
            return "failed:missing_pointwise:P0_full", ""
        P = view["P"]
        normals = np.array(est["normals_full"], dtype=float, copy=True)
        mask = np.asarray(est["mask_eval"], bool)
        if normals.shape != P.shape or mask.shape != (len(P),):
            return "failed:invalid_evaluation_normals", ""
        # Non-evaluable locations have no normal arrow; do not display carry/default as evidence.
        normals[~mask | ~np.isfinite(normals).all(axis=1)] = 0.0
        anchors = np.asarray(est.get("anchor_indices", []), int).copy()
        anchors = anchors[(anchors >= 0) & (anchors < len(P))]
        did = str(row["directory_id"])
        sid = _integer_id(row["segment_id"], "segment_id")
        idx = _integer_id(row["segment_index"], "segment_index")
        destination = surface_per_case_csv_path(
            run_id, frame_reference, did, surface_tag=surface_tag, project_root=project_root
        ).parent.parent / "figures" / did / f"{did}_buckling_surface_constraint_visualization_{sid}_{idx}.html"
        status = save_surface_constraint_html(
            out_path=destination,
            P_full=view.get("P_full", P), P0_full=view["P0_full"], P_trim=P,
            d_sin=view.get("d_latent", view.get("d_sin")),
            g1_3d=view.get("g1_3d"), n_eff=normals,
            anchor_indices=anchors, anchor_constraint=mask[anchors].copy(),
            mesh_path=mesh_path, segment_id=sid, cell_index=idx,
            P0_trim=view.get("P0"), U=view.get("U"), V=view.get("V"),
            fit_U=view.get("fit_U"), fit_V=view.get("fit_V"),
        )
        if status == "ok" and not destination.is_file():
            return "failed:export_did_not_create_file", ""
        return str(status), str(destination) if status == "ok" else ""
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return f"failed:{type(exc).__name__}:{exc}", ""

def apply_surface_eval_to_row(
    row: Dict[str, Any],
    *,
    pw: Dict[str, np.ndarray],
    surface_cfg: Dict[str, Any],
    project_root: Path,
    run_id: str,
    surface_tag: str,
    verbose: bool = False,
    enable_plots: bool = False,
    frame_reference: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    save_status: Dict[str, str] = {}
    tag = validate_surface_tag(surface_tag)
    out = _clear_surface(row, tag, surface_cfg)
    if enable_plots:
        out["viz_surface_html_status"] = "skipped_surface_not_evaluated"
    vs = float(surface_cfg.get("voxel_size", row.get("voxel_size_mm", 0.5)))
    directory_id = str(out.get("directory_id", ""))
    segment_id = out.get("segment_id")
    cell_index = out.get("segment_index")
    side = str(out.get("side", "") or "")

    if str(out.get("fit_status", "")) != "ok":
        out["surface_evaluation_status"] = "skipped_invalid_fit"
        out["surface_evaluation_reason"] = f"fit_status={out.get('fit_status')}"
        save_status["surface"] = out["surface_evaluation_status"]
        return _surface_classification_only(out), save_status

    if "d_latent" not in pw and "d_sin" not in pw:
        out["surface_evaluation_status"] = "missing_pointwise"
        out["surface_evaluation_reason"] = "d_latent/d_sin missing in pointwise"
        save_status["surface"] = out["surface_evaluation_status"]
        return _surface_classification_only(out), save_status
    if "P" not in pw:
        out["surface_evaluation_status"] = "missing_pointwise"
        out["surface_evaluation_reason"] = "P missing in pointwise"
        save_status["surface"] = out["surface_evaluation_status"]
        return _surface_classification_only(out), save_status

    try:
        _validate_pointwise(pw, {}, row, run_id=run_id, frame_reference=str(row.get("frame_reference", "baseline")), voxel_size=vs)
    except (ValueError, TypeError) as exc:
        out["surface_evaluation_status"] = "invalid_pointwise"
        out["surface_evaluation_reason"] = str(exc)
        save_status["surface"] = out["surface_evaluation_status"]
        return _surface_classification_only(out), save_status

    r_med_vox = radius_median_vox_from_row(out, vs)
    patch = compute_patch_radius_vox(
        r_med_vox,
        patch_r_ratio=float(surface_cfg.get("patch_r_ratio", 3.0)),
        r_min_mm=float(surface_cfg.get("r_min_mm", 2.5)),
        r_max_mm=float(surface_cfg.get("r_max_mm", 5.0)),
        voxel_size=vs,
    )
    out["patch_r_ratio"] = patch.get("patch_r_ratio", np.nan)
    out["r_patch_definition"] = patch.get("r_patch_definition", "")
    out["r_patch_mm_raw"] = patch.get("r_patch_mm_raw", np.nan)
    out["r_patch_mm"] = patch.get("r_patch_mm", np.nan)
    out["r_patch_vox"] = patch.get("r_patch_vox", np.nan)
    out["coh_thr"] = float(surface_cfg.get("coh_thr", 0.75))
    out["constraint_coverage_thresh"] = float(surface_cfg.get("constraint_coverage_thresh", 0.50))
    out["squared_displacement_coverage_thresh"] = float(
        surface_cfg.get("squared_displacement_coverage_thresh", 0.50)
    )
    out["R_out_thresh"] = float(surface_cfg.get("R_out_thresh", 0.25))
    out["R_tan_thresh"] = float(surface_cfg.get("R_tan_thresh", 1.0 - float(out["R_out_thresh"])))
    out["bridge_max_gap_mm"] = float(surface_cfg.get("bridge_max_gap_mm", 5.0))
    out["bridge_axis_max_deg"] = float(surface_cfg.get("bridge_axis_max_deg", 45.0))
    out["min_neighbors"] = int(surface_cfg.get("min_neighbors", 30))

    if not patch.get("ok"):
        out["surface_evaluation_status"] = "invalid_radius"
        out["surface_evaluation_reason"] = patch.get("reason", "invalid patch radius")
        save_status["surface"] = out["surface_evaluation_status"]
        return _surface_classification_only(out), save_status

    if not side:
        out["surface_evaluation_status"] = "missing_brain_mesh"
        out["surface_evaluation_reason"] = "side missing on measurement row"
        save_status["surface"] = out["surface_evaluation_status"]
        return _surface_classification_only(out), save_status

    mesh_p = brain_mesh_path(directory_id, side, project_root)
    if not mesh_p.is_file():
        out["surface_evaluation_status"] = "missing_brain_mesh"
        out["surface_evaluation_reason"] = str(mesh_p)
        save_status["surface"] = out["surface_evaluation_status"]
        return _surface_classification_only(out), save_status

    settings = surface_settings(surface_cfg)
    stem = f"{directory_id}_seg{segment_id}_idx{int(cell_index)}"
                                    
    cache_path = normals_cache_dir(run_id, project_root) / directory_id / tag / f"{stem}_normals.npz"
    P_trim = np.asarray(pw["P"], float)
    P_full = np.asarray(pw.get("P_full", P_trim), float)
    d_latent = np.asarray(pw.get("d_latent", pw.get("d_sin")), float)
    ds_vox = float(np.median(np.linalg.norm(np.diff(P_full, axis=0), axis=1)))
    out["surface_ds_vox"] = ds_vox

    T0 = np.asarray(pw["T0"], float) if "T0" in pw else None
    T_frame = np.asarray(pw["T"], float) if "T" in pw else T0
    T_obs = np.asarray(pw["T_obs"], float) if "T_obs" in pw else None
    g1_3d = np.asarray(pw["g1_3d"], float) if "g1_3d" in pw else None

    try:
        out["surface_displacement_representation"] = "latent_pre_detrending_periodic"
        ev, est, cache_status = run_surface_evaluation(
            P_full_vox=P_full,
            P_trim=P_trim,
            d_sin=d_latent,
            ds_vox=float(ds_vox),
            r_patch_vox=float(patch["r_patch_vox"]),
            mesh_path=mesh_p,
            settings=settings,
            cache_path=cache_path,
            g1_3d=g1_3d,
            T_frame=T_frame,
            T0=T0,
            T_obs=T_obs,
            verbose=verbose,
        )
        out["surface_method_revision"] = SURFACE_METHOD_REVISION
        out["normals_cache_status"] = cache_status
        for k in (
            "R_out",
            "R_tan",
            "R_dir",
            "cortex_coverage",
            "constraint_coverage",
            "squared_displacement_coverage",
            "total_energy",
            "energy_constrained",
            "perp_energy",
            "evaluated_tangential_fraction",
            "R_out_upper_bound",
            "R_out_within_upper_bound",
            "abs_T0_dot_n_mean",
            "abs_T0_dot_n_median",
            "abs_Tobs_dot_n_mean",
            "abs_Tobs_dot_n_median",
            "n_points_total",
            "n_eval",
            "n_eval_good",
            "n_eval_bridged",
            "good_coverage",
            "bridged_coverage",
            "squared_displacement_coverage_good",
            "squared_displacement_coverage_bridged",
        ):
            dst = {"R_out": "R_out_surface", "R_tan": "R_tan_surface", "R_dir": "R_dir_surface"}.get(k, k)
            out[dst] = ev.get(k, np.nan)
        r_dir_ok = out.get("R_dir_evaluable")
        try:
            r_dir_ok = int(r_dir_ok) == 1 if r_dir_ok is not None and str(r_dir_ok) != "" else False
        except (TypeError, ValueError):
            r_dir_ok = False
        if not r_dir_ok:
            out["R_dir_surface"] = np.nan
            out["R_dir_status"] = "invalid_principal_axis_not_unique"
        else:
            out["R_dir_status"] = "ok"
        out["R_out_latent"] = out.get("R_out_surface", np.nan)
        out["R_tan_latent"] = out.get("R_tan_surface", np.nan)
        out["squared_displacement_coverage_latent"] = out.get("squared_displacement_coverage", np.nan)
        if "d_response" in pw and est.get("mask_eval") is not None:
            try:
                from sin_trend_surface import evaluate_displacement

                ev_resp = evaluate_displacement(
                    np.asarray(pw["d_response"], float),
                    est,
                    g1_3d=g1_3d,
                    T_frame=T_frame,
                    T0=T0,
                    T_obs=T_obs,
                )
                out["R_out_response"] = ev_resp.get("R_out", np.nan)
                out["R_tan_response"] = ev_resp.get("R_tan", np.nan)
                out["squared_displacement_coverage_response"] = ev_resp.get(
                    "squared_displacement_coverage", np.nan
                )
            except Exception as exc:  # noqa: BLE001
                out["R_out_response"] = np.nan
                out["R_tan_response"] = np.nan
                out["squared_displacement_coverage_response"] = np.nan
                out["response_surface_diag_reason"] = str(exc)
        out["n_anchors"] = est.get("n_anchors", np.nan)
        out["n_anchors_observed_normal"] = est.get("n_anchors_observed_normal", np.nan)
        out["surface_evaluation_status"] = "ok" if ev["status"] == "ok" else ev["status"]
        out["surface_evaluation_reason"] = ev.get("reason", "")
        if enable_plots:
            if out["surface_evaluation_status"] == "ok":
                plot_frame = validate_frame_reference(frame_reference or str(row.get("frame_reference") or "baseline"))
                plot_status, plot_path = _save_surface_stage_plot(
                    out, pw, est, run_id=run_id, frame_reference=plot_frame,
                    surface_tag=tag, project_root=project_root, mesh_path=mesh_p,
                )
                out["viz_surface_html_status"] = plot_status
                out["viz_surface_html_path"] = plot_path
                save_status["surface_constraint_html"] = plot_status
            else:
                out["viz_surface_html_status"] = "skipped_surface_not_evaluable"
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        out = _clear_surface(row, tag, surface_cfg)
        out["surface_evaluation_status"] = "surface_eval_failed"
        out["surface_evaluation_reason"] = str(exc)
        if verbose:
            print(f"surface evaluation failed (fit retained): {exc}")

    save_status["surface"] = str(out["surface_evaluation_status"])
    return _surface_classification_only(out), save_status


def run_surface_stage_on_dataframe(
    df: pd.DataFrame,
    *,
    run_id: str,
    frame_reference: str,
    surface_cfg: Dict[str, Any],
    project_root: Optional[Path] = None,
    surface_tag: str = "primary",
    verbose: bool = False,
    enable_plots: bool = False,
    segment_ids: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    root = Path(project_root) if project_root is not None else project_root_from_here(__file__)
    run_id = validate_run_id(run_id)
    frame_reference = validate_frame_reference(frame_reference)
    tag = validate_surface_tag(surface_tag)
    surface_cfg = resolve_surface_config(run_id, surface_cfg, project_root=root)
    cfg_record = write_surface_config(run_id, surface_cfg, surface_tag=tag, project_root=root, overwrite=False)
    rows_out: List[Dict[str, Any]] = []
    n_missing_pw = 0
    want = {_integer_id(s, "segment_id") for s in segment_ids} if segment_ids is not None else None
    for rec in df.to_dict(orient="records"):
        out = _clear_surface(rec, tag, surface_cfg)
        try:
            selected = want is None or _integer_id(rec.get("segment_id"), "segment_id") in want
            if not selected:
                out["surface_evaluation_reason"] = "not_selected_for_surface_stage"
            elif str(rec.get("fit_status", "")) != "ok":
                out["surface_evaluation_status"] = "skipped_invalid_fit"
                out["surface_evaluation_reason"] = f"fit_status={rec.get('fit_status')}"
            else:
                pw_path = pointwise_path_for_row(rec, run_id=run_id, frame_reference=frame_reference, project_root=root)
                if pw_path is None:
                    n_missing_pw += 1
                    out["surface_evaluation_status"] = "missing_pointwise"
                    out["surface_evaluation_reason"] = "pointwise npz not found"
                else:
                    pw, scalars = load_pointwise_npz(pw_path)
                    _validate_pointwise(pw, scalars, rec, run_id=run_id, frame_reference=frame_reference, voxel_size=float(surface_cfg["voxel_size"]))
                    out, _ = apply_surface_eval_to_row(
                        rec, pw=pw, surface_cfg=surface_cfg, project_root=root,
                        run_id=run_id, surface_tag=tag, verbose=verbose,
                        enable_plots=enable_plots, frame_reference=frame_reference,
                    )
                    out["surface_source_pointwise_path"] = str(pw_path)
                    out["surface_source_pointwise_sha256"] = sha256_file(pw_path)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            out = _clear_surface(rec, tag, surface_cfg)
            out["surface_evaluation_status"] = "invalid_pointwise"
            out["surface_evaluation_reason"] = str(exc)
        if enable_plots and out.get("viz_surface_html_status") == "not_requested":
            out["viz_surface_html_status"] = "skipped_surface_not_evaluated"
        out["surface_config_hash"] = cfg_record.get("config_hash", "")
        out["surface_code_hash"] = cfg_record.get("surface_source_code_hash_combined", "")
        rows_out.append(_surface_classification_only(out))
    out_df = pd.DataFrame(rows_out)
    meta = dict(
        surface_tag=tag, n_rows=int(len(out_df)), n_missing_pointwise=int(n_missing_pw),
        surface_config_hash=cfg_record.get("config_hash", ""),
        surface_plot_status_counts=out_df["viz_surface_html_status"].value_counts(dropna=False).to_dict() if len(out_df) else {},
        surface_status_counts=out_df["surface_evaluation_status"].value_counts(dropna=False).to_dict() if len(out_df) else {},
    )
    return out_df, meta


def run_surface_stage_for_directory(
    directory_id: str,
    run_id: str,
    *,
    frame_reference: str,
    surface_cfg: Dict[str, Any],
    project_root: Optional[Path] = None,
    surface_tag: str = "primary",
    verbose: bool = False,
    enable_plots: bool = False,
    segment_ids: Optional[Sequence[int]] = None,
    write_csv: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read immutable measurement artifacts; save every surface tag separately."""
    root = Path(project_root) if project_root is not None else project_root_from_here(__file__)
    run_id = validate_run_id(run_id)
    tag = validate_surface_tag(surface_tag)
    csv_path = per_case_csv_path(run_id, frame_reference, directory_id, root)
    if not csv_path.is_file():
        raise FileNotFoundError(f"measurement CSV not found (run measure stage first): {csv_path}")
    measurement_hash = sha256_file(csv_path)
    df = pd.read_csv(csv_path)
    if len(df) and not df["directory_id"].astype(str).eq(str(directory_id)).all():
        raise ValueError("measurement CSV directory_id does not match requested directory")
    alt = surface_per_case_csv_path(run_id, frame_reference, directory_id, surface_tag=tag, project_root=root)
    previous = pd.read_csv(alt) if segment_ids is not None and alt.is_file() else None
    out_df, meta = run_surface_stage_on_dataframe(
        df, run_id=run_id, frame_reference=frame_reference, surface_cfg=surface_cfg,
        project_root=root, surface_tag=tag, verbose=verbose, enable_plots=enable_plots, segment_ids=segment_ids,
    )
    out_df["surface_source_measurement_csv_sha256"] = measurement_hash
    # Preserve only same-tag results of unselected rows with unchanged source artifacts.
    if previous is not None and len(previous) and len(out_df):
        want = {_integer_id(v, "segment_id") for v in segment_ids}
        for i, rec in out_df.iterrows():
            sid = _integer_id(rec["segment_id"], "segment_id")
            if sid in want:
                continue
            matches = previous[(pd.to_numeric(previous["segment_id"], errors="coerce") == sid) & (pd.to_numeric(previous["segment_index"], errors="coerce") == _integer_id(rec["segment_index"], "segment_index"))]
            if len(matches) != 1:
                continue
            old = matches.iloc[0]
            if old.get("surface_source_measurement_csv_sha256") != measurement_hash or old.get("surface_config_hash") != meta["surface_config_hash"]:
                continue
            try:
                old_pw = pointwise_path_for_row(rec.to_dict(), run_id=run_id, frame_reference=frame_reference, project_root=root)
                if old_pw is None or old.get("surface_source_pointwise_sha256") != sha256_file(old_pw):
                    continue
            except (ValueError, OSError):
                continue
            for key in previous.columns:
                if _is_surface_key(key):
                    out_df.at[i, key] = old[key]
    meta.update(
        directory_id=directory_id, run_id=run_id, frame_reference=frame_reference,
        per_case_csv=str(csv_path), measurement_csv_sha256=measurement_hash,
        per_case_csv_surface=str(alt),
        surface_plot_status_counts=out_df["viz_surface_html_status"].value_counts(dropna=False).to_dict() if len(out_df) else {},
        surface_status_counts=out_df["surface_evaluation_status"].value_counts(dropna=False).to_dict() if len(out_df) else {},
    )
    if write_csv:
        atomic_to_csv(out_df, alt, index=False)
        meta["per_case_csv_sha256"] = sha256_file(alt)
    return out_df, meta
