#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 31 run sin trend fitting."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
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

from sin_trend_classify import apply_classification  # noqa: E402
from sin_trend_measure import (  # noqa: E402
    build_groupid_map,
    default_measurement_config,
    failure_row,
    ixi_id_from_directory,
    measure_interval,
    measurement_row_schema,
)
from sin_trend_paths import (  # noqa: E402
    EXPECTED_FIT_METHOD,
    FRAME_BASELINE,
    FRAME_OBSERVED,
    FRAME_REFERENCES,
    MEASUREMENT_REVISION,
    MEASUREMENT_VERSION,
    OUTPUT_TAG,
    SURFACE_METHOD_REVISION,
    SURFACE_INCOMPLETE_STATUSES,
    atomic_to_csv,
    atomic_write_json,
    brain_mesh_path,
    candidate_dir,
    centerline_vtp_path,
    figures_root,
    interval_key_set_hash,
    metadata_dict,
    participant_outcome_from_counts,
    participant_status_path,
    per_case_csv_path,
    per_case_dir,
    pointwise_dir,
    project_root_from_here,
    sha256_file,
    split_measure_surface_config,
    to_python_scalar,
    validate_frame_reference,
    validate_run_id,
    validate_surface_tag,
    write_run_config,
    write_surface_config,
    surface_config_for_run,
    read_surface_config,
    read_participant_status,
    surface_participant_status_path,
)
from sin_trend_surface_stage import (  # noqa: E402
    apply_surface_eval_to_row,
    run_surface_stage_for_directory,
)
from sin_trend_visualize import save_interval_visualizations  # noqa: E402


def _setup_import_paths(project_root: Path) -> None:
    ac = project_root / "analysis_code"
    for p in (str(ac), str(ac / "sin_trend"), str(ac / "utils"), str(ac / "mesh"), str(ac / "morphometrics")):
        if p not in sys.path:
            sys.path.insert(0, p)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Run baseline-Bishop sin+trend fitting ({OUTPUT_TAG}) for one directory_id")
    p.add_argument("--directory-id", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--frame-reference", default=FRAME_BASELINE, choices=list(FRAME_REFERENCES))
    p.add_argument("--segment-id", action="append", type=int, default=None, help="limit to given segment_id(s) (smoke test)")
    stage = p.add_mutually_exclusive_group()
    stage.add_argument(
        "--measure-only",
        action="store_true",
        help="run measurement stage only (1–3); leave surface_evaluation_status=not_run",
    )
    stage.add_argument(
        "--surface-only",
        action="store_true",
        help="run surface stage only (4+) from existing per-case CSV + pointwise.npz",
    )
    stage.add_argument("--no-surface", action="store_true", help="alias of --measure-only (compat)")
    p.add_argument(
        "--surface-tag",
        default="primary",
        help="surface config / cache tag (default: primary; non-primary writes *_surface_<tag>.csv)",
    )
    p.add_argument(
        "--surface-config-json",
        type=Path,
        default=None,
        help="optional JSON object of surface settings (patch_r_ratio, coh_thr, bridge_*, R_out_thresh, ...)",
    )
    p.add_argument("--observed-tangent-source", default="frenet", choices=["frenet", "numeric"])
    p.add_argument("--enable-plots", action="store_true", default=True, help="save PNG/HTML visualizations (default: on)")
    p.add_argument("--no-plots", action="store_true", help="disable PNG/HTML visualizations")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
    
# ---------------------------------------------------------------------------
def save_candidates(df: pd.DataFrame, path: Path) -> str:
    try:
        atomic_to_csv(df, path, index=False)
        return "ok" if len(df) else "empty"
    except Exception as exc:  # noqa: BLE001
        return f"failed:{exc}"


def save_pointwise(pw: Dict[str, Any], scalars: Dict[str, Any], path: Path) -> str:
    import os
    import tempfile

    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.asarray(v) for k, v in pw.items() if isinstance(v, np.ndarray)}
        # Use explicit .npz suffix so savez_compressed does not append another .npz
        fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=str(path.parent))
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            np.savez_compressed(tmp_path, scalars_json=np.array(json.dumps(scalars, default=str)), **arrays)
            with np.load(tmp_path, allow_pickle=False) as z:
                for k, arr in arrays.items():
                    if k not in z.files:
                        raise ValueError(f"npz missing key after write: {k}")
                    if np.asarray(z[k]).shape != np.asarray(arr).shape:
                        raise ValueError(f"npz reload shape mismatch for {k}")
            os.replace(tmp_path, path)
            return "ok"
        finally:
            if tmp_path.exists() and tmp_path != path:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    except Exception as exc:  # noqa: BLE001
        return f"failed:{exc}"


# ---------------------------------------------------------------------------
# 1 interval
# ---------------------------------------------------------------------------
def process_interval(
    *,
    directory_id: str,
    segment_id: Any,
    cell_index: int,
    pid: np.ndarray,
    points: np.ndarray,
    pointdata: Dict[str, Any],
    side: str,
    frame_reference: str,
    cfg: Dict[str, Any],
    project_root: Path,
    run_id: str,
    do_surface: bool,
    verbose: bool,
    surface_cfg: Optional[Dict[str, Any]] = None,
    surface_tag: str = "primary",
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    ids = {
        "directory_id": directory_id,
        "segment_id": segment_id,
        "segment_index": int(cell_index),
        "IXI_ID": ixi_id_from_directory(directory_id),
        "side": side,
    }
    P_vox = np.asarray(points[pid], float)

    def _pd(key: str) -> Optional[np.ndarray]:
        if key not in pointdata:
            return None
        try:
            arr = np.asarray(pointdata[key][pid], float)
        except Exception:  # noqa: BLE001
            return None
        return arr

    radius = _pd("MaximumInscribedSphereRadius")
    curvature = _pd("Curvature")
    frenet = _pd("FrenetTangent")

    res = measure_interval(
        P_vox,
        frame_reference=frame_reference,
        cfg=cfg,
        observed_tangent=frenet,
        radius_vox=radius,
        curvature_vox=curvature,
        point_ids=np.asarray(pid),
        ids=ids,
        verbose=verbose,
    )
    row = res["row"]
    save_status: Dict[str, str] = {}

         
    stem = f"{directory_id}_seg{segment_id}_idx{cell_index}"
    cand_path = candidate_dir(run_id, frame_reference, project_root) / f"{stem}_candidates.csv"
    if len(res["candidates"]):
        st = save_candidates(res["candidates"], cand_path)
        row["candidate_csv_path"] = str(cand_path) if st in {"ok", "empty"} else ""
        row["candidates_save_status"] = st
    else:
        row["candidates_save_status"] = "no_candidates"
    save_status["candidates"] = row["candidates_save_status"]

    pw = res["pointwise"] or {}
                                           
                          
    if pw:
        pw_path = pointwise_dir(run_id, frame_reference, project_root) / f"{stem}_pointwise.npz"
        scalars = {
            **ids,
            "frame_reference": frame_reference,
            "measurement_version": MEASUREMENT_VERSION,
            "measurement_revision": MEASUREMENT_REVISION,
            "voxel_size_mm": float(cfg["voxel_size"]),
            "removal_points": int(cfg["removal"]),
            "fit_status": row["fit_status"],
            "fitting_coordinate": row.get("fitting_coordinate") or pw.get("phase_coordinate") or "baseline_arclength",
            "phase_coordinate": row.get("phase_coordinate") or pw.get("phase_coordinate") or "baseline_arclength",
            "trend_coordinate": (
                row.get("trend_coordinate") or pw.get("trend_coordinate") or "baseline_arclength"
            ),
            "quadratic_detrend_coordinate": row.get("quadratic_detrend_coordinate") or "observed_arclength",
            "grid_length_basis": row.get("grid_length_basis") or "trimmed_baseline_arclength",
            "radius_basis": row.get("radius_basis") or "trimmed_observed_lumen",
            "array_keys_note": {
                "s_phase_full/s_phase_trim/s_trend_trim/s_fit": "baseline_arclength (trim origin 0)",
                "s_obs_full/s_obs_trim/s_full": "observed_arclength",
                "s_base_full": "cumulative baseline arclength along P0",
            },
            "k_hat_vox": (
                float(res["search"]["k_hat"])
                if res.get("search") and np.isfinite(res["search"].get("k_hat", np.nan))
                else None
            ),
            "coef_vox": (
                {k: float(res["search"][k]) for k in ("uU_cos", "uU_sin", "uV_cos", "uV_sin")}
                if res.get("search")
                and all(np.isfinite(res["search"].get(k, np.nan)) for k in ("uU_cos", "uU_sin", "uV_cos", "uV_sin"))
                else None
            ),
            "surface_evaluation_status": "not_run",
            "units": "voxel (multiply by voxel_size_mm for mm)",
        }
        st = save_pointwise(pw, scalars, pw_path)
        row["pointwise_npz_path"] = str(pw_path) if st == "ok" else ""
        row["pointwise_save_status"] = st
    else:
        row["pointwise_save_status"] = "no_pointwise"
    save_status["pointwise"] = row["pointwise_save_status"]

                  
    if do_surface and res.get("ok_for_surface") and pw:
        scfg = dict(surface_cfg or cfg)
        row, sst = apply_surface_eval_to_row(
            row,
            pw=pw,
            surface_cfg=scfg,
            project_root=project_root,
            run_id=run_id,
            surface_tag=surface_tag,
            verbose=verbose,
        )
        save_status.update(sst)
    elif do_surface and not res.get("ok_for_surface"):
                                                               
        if not str(row.get("surface_evaluation_status") or ""):
            row["surface_evaluation_status"] = "skipped_invalid_fit"
            row["surface_evaluation_reason"] = f"fit_status={row.get('fit_status')}"
        save_status["surface"] = str(row["surface_evaluation_status"])
        row = apply_classification(row)
    else:
        row["surface_evaluation_status"] = "not_run"
        row["surface_evaluation_reason"] = "measure_only_or_no_surface"
        save_status["surface"] = "not_run"
        row = apply_classification(row)

    if cfg.get("enable_plots") and row.get("fit_status") == "ok":
        try:
            viz_dir = figures_root(run_id, frame_reference, project_root) / directory_id
            vs = float(cfg["voxel_size"])
            L_trim_vox = float(row["L_trimmed_mm"]) / vs if np.isfinite(row.get("L_trimmed_mm", np.nan)) else float("nan")
            L_base_vox = (
                float(row["L_base_mm"]) / vs
                if np.isfinite(row.get("L_base_mm", np.nan))
                else float("nan")
            )
            mesh_p = brain_mesh_path(directory_id, side, project_root)
            viz_st = save_interval_visualizations(
                out_dir=viz_dir,
                directory_id=directory_id,
                segment_id=segment_id,
                cell_index=cell_index,
                search=res.get("search"),
                candidates=res.get("candidates") if isinstance(res.get("candidates"), pd.DataFrame) else pd.DataFrame(),
                pointwise=pw,
                row=row,
                mesh_path=mesh_p if mesh_p.is_file() else None,
                L_trimmed_vox=L_trim_vox,
                L_base_vox=L_base_vox,
            )
            row["viz_grid_search_status"] = viz_st.get("grid_search_png", "")
            row["viz_buckling_html_status"] = viz_st.get("buckling_html", "")
            row["viz_surface_html_status"] = viz_st.get("surface_constraint_html", "")
            if verbose:
                print(
                    "  viz:",
                    f"grid={row['viz_grid_search_status']}",
                    f"buckling={row['viz_buckling_html_status']}",
                    f"surface={row['viz_surface_html_status']}",
                )
        except Exception as exc:  # noqa: BLE001
            row["viz_grid_search_status"] = f"failed:{exc}"
            if verbose:
                print(f"plot warning (fit retained): {exc}")
                                                        
    return row, save_status


# ---------------------------------------------------------------------------
       
# ---------------------------------------------------------------------------
def run_directory(
    directory_id: str,
    run_id: str,
    frame_reference: str = FRAME_BASELINE,
    project_root: Optional[Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
    segment_ids: Optional[Sequence[int]] = None,
    do_surface: bool = True,
    surface_cfg: Optional[Dict[str, Any]] = None,
    surface_tag: str = "primary",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    root = Path(project_root) if project_root is not None else project_root_from_here(__file__)
    _setup_import_paths(root)
    run_id = validate_run_id(run_id)
    frame_reference = validate_frame_reference(frame_reference)
    cfg = {**default_measurement_config(), **(cfg or {})}
    verbose = bool(cfg.get("verbose", False))
    tag = validate_surface_tag(surface_tag)
    t_start = time.time()

    cfg_for_hash = {k: v for k, v in cfg.items() if k not in {"verbose", "enable_plots"}}
    measure_cfg, surface_cfg_default = split_measure_surface_config(cfg_for_hash)
                                                       
    write_run_config(run_id, measure_cfg, project_root=root)
    scfg = {}
    if do_surface:
        overrides = dict(surface_cfg or surface_cfg_default)
        if surface_cfg and "R_out_thresh" in surface_cfg and "R_tan_thresh" not in surface_cfg:
            overrides["R_tan_thresh"] = 1.0 - float(surface_cfg["R_out_thresh"])
        scfg = surface_config_for_run(run_id, overrides, surface_tag=tag, project_root=root)
        write_surface_config(run_id, scfg, surface_tag=tag, project_root=root)

    import centerline_extraction
    import vessel_mesh_features

    out_root = per_case_dir(run_id, frame_reference, root)
    out_case = out_root / directory_id
    out_case.mkdir(parents=True, exist_ok=True)
    status_path = participant_status_path(run_id, frame_reference, directory_id, root)
    status: Dict[str, Any] = metadata_dict(
        directory_id=directory_id,
        run_id=run_id,
        frame_reference=frame_reference,
        status="running",
        fit_method=EXPECTED_FIT_METHOD,
        measurement_version=MEASUREMENT_VERSION,
        measurement_revision=MEASUREMENT_REVISION,
        surface_enabled=False,
        surface_tag="",
        started_utc=datetime.now(timezone.utc).isoformat(),
    )
                                                 
    atomic_write_json(status, status_path)

    vtp = centerline_vtp_path(directory_id, root)
    if not vtp.is_file():
        status.update(status="failed", reason=f"missing centerline VTP: {vtp}", input_vtp=str(vtp))
        atomic_write_json(status, status_path)
        raise FileNotFoundError(str(vtp))
    status["input_vtp"] = str(vtp)
    status["input_vtp_sha256"] = sha256_file(vtp)

    try:
        mesh = centerline_extraction.load_vtp(str(vtp))
        celldata, pointids, points, pointdata = centerline_extraction.vmtk_convert2numpy(mesh)
    except (KeyboardInterrupt, SystemExit):
        status.update(
            status="interrupted",
            completeness_reason="interrupted_before_intervals",
            finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_sec=float(time.time() - t_start),
        )
        atomic_write_json(status, status_path)
        raise
    except Exception as exc:  # noqa: BLE001
        status.update(
            status="failed",
            reason=str(exc),
            finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_sec=float(time.time() - t_start),
        )
        atomic_write_json(status, status_path)
        raise
                                                                     
    group_ids = [to_python_scalar(g) for g in celldata["GroupIds"]]
    mapping, cell_index_of, gid_diag = build_groupid_map(group_ids, pointids, points)
    if len(gid_diag):
        atomic_to_csv(gid_diag, out_case / f"{directory_id}_groupid_diagnostics.csv", index=False)
    ambiguous = [
        to_python_scalar(g)
        for g in (gid_diag[gid_diag["resolution"] == "ambiguous_excluded"]["group_id"].tolist() if len(gid_diag) else [])
    ]
    all_gids = sorted(set(group_ids), key=lambda x: (str(type(x)), x))
    if segment_ids is not None:
        want = set(int(s) for s in segment_ids)
        all_gids = [g for g in all_gids if int(g) in want]

    right_cells, _left, *_ = vessel_mesh_features.classify_centerline_left_right_by_kmeans(points, pointids, n_iter=100, verbose=False)
    right_set = set(int(x) for x in np.asarray(right_cells).ravel().tolist())

    checkpoint_path = out_case / f"{directory_id}_vessel_segmentdata.checkpoint.csv"
    final_path = per_case_csv_path(run_id, frame_reference, directory_id, root)
    rows: List[Dict[str, Any]] = []
    exceptions: List[Dict[str, Any]] = []
    save_failures: List[Dict[str, Any]] = []
    required_artifacts: List[Dict[str, Any]] = []
    brain_meshes: Dict[str, Dict[str, str]] = {}
    meta_cols = {
        "voxel_size_mm": float(cfg["voxel_size"]),
        "frame_reference": frame_reference,
        "IXI_ID": ixi_id_from_directory(directory_id),
    }

    for gid in all_gids:
        gid = to_python_scalar(gid)
        if gid in ambiguous or gid not in mapping:
            rows.append(
                failure_row(
                    {**meta_cols, "directory_id": directory_id, "segment_id": gid, "segment_index": np.nan},
                    fit_status="ambiguous_groupid",
                    fit_status_reason="duplicate GroupId with non-identical shapes; not measured",
                    failure_stage="groupid_map",
                )
            )
            atomic_to_csv(pd.DataFrame(rows), checkpoint_path, index=False)
            continue
        cell_index = int(to_python_scalar(cell_index_of[gid]))
        pid = np.asarray(mapping[gid])
        side = "Rt" if cell_index in right_set else "Lt"
        if verbose:
            print("\n" + "=" * 80 + f"\n{directory_id} segment_id={gid} index={cell_index} frame={frame_reference}")
        try:
            row, sst = process_interval(
                directory_id=directory_id,
                segment_id=gid,
                cell_index=cell_index,
                pid=pid,
                points=points,
                pointdata=pointdata,
                side=side,
                frame_reference=frame_reference,
                cfg=cfg,
                project_root=root,
                run_id=run_id,
                do_surface=False,
                verbose=verbose,
                surface_cfg=scfg,
                surface_tag=tag,
            )
            for k, v in sst.items():
                if str(v).startswith("failed"):
                    save_failures.append({"segment_id": gid, "what": k, "status": str(v)})
            if str(row.get("pointwise_save_status", "")) == "ok" and row.get("pointwise_npz_path"):
                pw_path = Path(str(row["pointwise_npz_path"]))
                art_pw: Dict[str, Any] = {
                    "kind": "pointwise",
                    "path": str(pw_path),
                    "segment_id": gid,
                    "segment_index": cell_index,
                }
                try:
                    art_pw["sha256"] = sha256_file(pw_path)
                except OSError:
                    pass
                required_artifacts.append(art_pw)
            if str(row.get("candidates_save_status", "")) == "ok" and row.get("candidate_csv_path"):
                cand_path = Path(str(row["candidate_csv_path"]))
                art_c: Dict[str, Any] = {
                    "kind": "candidates",
                    "path": str(cand_path),
                    "segment_id": gid,
                    "segment_index": cell_index,
                }
                try:
                    art_c["sha256"] = sha256_file(cand_path)
                except OSError:
                    pass
                required_artifacts.append(art_c)
        except (KeyboardInterrupt, SystemExit):
            if rows:
                atomic_to_csv(pd.DataFrame(rows), checkpoint_path, index=False)
            status.update(
                status="interrupted",
                completeness_reason="interrupted_during_intervals",
                n_intervals_partial=len(rows),
                finished_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_sec=float(time.time() - t_start),
            )
            atomic_write_json(status, status_path)
            raise
        except Exception as exc:  # noqa: BLE001
            exceptions.append({"segment_id": gid, "segment_index": cell_index, "error": str(exc)})
            if verbose:
                traceback.print_exc()
            row = failure_row(
                {**meta_cols, "directory_id": directory_id, "segment_id": gid, "segment_index": cell_index, "side": side},
                fit_status="exception",
                fit_status_reason=str(exc),
                failure_stage="process_interval",
                surface_evaluation_status="skipped_exception",
            )
        rows.append(row)
        atomic_to_csv(pd.DataFrame(rows), checkpoint_path, index=False)

    # Keep the stable measurement schema first, but do not discard runtime
    # columns (for example PNG/HTML export statuses added by process_interval).
    # Empty runs still produce the complete measurement schema.
    schema_cols = list(measurement_row_schema().keys())
    df = pd.DataFrame(rows)
    extra_cols = [c for c in df.columns if c not in schema_cols]
    df = df.reindex(columns=schema_cols + extra_cols)
    atomic_to_csv(df, final_path, index=False)

    fit_counts = df["fit_status"].value_counts(dropna=False).to_dict() if len(df) else {}
    surf_counts = df["surface_evaluation_status"].value_counts(dropna=False).to_dict() if len(df) else {}
    n_intervals = int(len(df))
    outcome, outcome_reason = participant_outcome_from_counts(
        n_exceptions=len(exceptions),
        n_save_failures=len(save_failures),
        n_ambiguous=len(ambiguous),
        surface_enabled=False,
        surface_status_counts={str(k): int(v) for k, v in surf_counts.items()},
    )
    status.update(
        status=outcome,
        completeness_reason=outcome_reason,
        n_segments_in_vtp=len(set(group_ids)),
        n_intervals=n_intervals,
        n_fit_ok=int((df["fit_status"] == "ok").sum()) if n_intervals else 0,
        fit_status_counts={str(k): int(v) for k, v in fit_counts.items()},
        surface_status_counts={str(k): int(v) for k, v in surf_counts.items()},
        n_exceptions=len(exceptions),
        exceptions=exceptions,
        n_save_failures=len(save_failures),
        save_failures=save_failures,
        ambiguous_group_ids=[to_python_scalar(g) for g in ambiguous],
        duplicate_groupid_diagnostics=int(len(gid_diag)),
        segment_filter=[int(s) for s in segment_ids] if segment_ids is not None else None,
        surface_enabled=False,
        surface_tag="",
        input_brain_meshes=brain_meshes,
        required_artifacts=required_artifacts,
        measurement_version=MEASUREMENT_VERSION,
        measurement_revision=MEASUREMENT_REVISION,
        surface_method_revision=SURFACE_METHOD_REVISION,
        per_case_csv=str(final_path),
                                                         
        interval_key_set_sha256=interval_key_set_hash(df),
        per_case_csv_sha256=sha256_file(final_path) if final_path.is_file() else "",
        elapsed_sec=float(time.time() - t_start),
        finished_utc=datetime.now(timezone.utc).isoformat(),
        completeness_notes={
            "surface_incomplete_statuses": sorted(SURFACE_INCOMPLETE_STATUSES),
            "eligibility_and_no_evaluable_are_not_partial": True,
        },
    )
    atomic_write_json(status, participant_status_path(run_id, frame_reference, directory_id, root))
    if verbose:
        print(f"saved: {final_path} (n={n_intervals}; status={status['status']}; elapsed={status['elapsed_sec']:.1f}s)")
    if do_surface:
        return run_saved_surface(
            directory_id, run_id, frame_reference=frame_reference,
            surface_cfg=scfg, project_root=root, surface_tag=tag,
            verbose=verbose, segment_ids=segment_ids,
            enable_plots=bool(cfg.get("enable_plots", True)),
        )
    return df, status


def run_saved_surface(
    directory_id, run_id, *, frame_reference, surface_cfg, project_root,
    surface_tag="primary", verbose=False, segment_ids=None, enable_plots=False,
):
    """Surface completion has its own record; saved measurement artifacts are read-only."""
    root = Path(project_root)
    tag = validate_surface_tag(surface_tag)
    # Reject a changed tag before replacing its existing completion record.
    write_surface_config(run_id, surface_cfg, surface_tag=tag, project_root=root)
    status_path = surface_participant_status_path(
        run_id, frame_reference, directory_id, surface_tag=tag, project_root=root
    )
    status = metadata_dict(
        directory_id=directory_id, run_id=run_id, frame_reference=frame_reference,
        surface_tag=tag, surface_only=True, surface_enabled=True, status="running",
        started_utc=datetime.now(timezone.utc).isoformat(),
    )
    atomic_write_json(status, status_path)
    try:
        df, meta = run_surface_stage_for_directory(
            directory_id, run_id, frame_reference=frame_reference,
            surface_cfg=surface_cfg, project_root=root, surface_tag=tag,
            verbose=verbose, segment_ids=segment_ids, write_csv=True,
            enable_plots=enable_plots,
        )
        measured = read_participant_status(run_id, frame_reference, directory_id, root) or {}
        counts = {str(k): int(v) for k, v in (meta.get("surface_status_counts") or {}).items()}
        outcome, reason = participant_outcome_from_counts(
            n_exceptions=int(measured.get("n_exceptions", 0)),
            n_save_failures=int(measured.get("n_save_failures", 0)),
            n_ambiguous=len(measured.get("ambiguous_group_ids") or []),
            surface_enabled=True, surface_status_counts=counts,
        )
        status.update(meta)
        status.update(
            status=outcome, completeness_reason=reason,
            n_intervals=int(len(df)), measurement_version=MEASUREMENT_VERSION,
            measurement_revision=MEASUREMENT_REVISION,
            surface_method_revision=SURFACE_METHOD_REVISION,
            surface_config_hash=read_surface_config(run_id, tag, root)["config_hash"],
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        atomic_write_json(status, status_path)
        return df, status
    except (KeyboardInterrupt, SystemExit):
        status.update(status="interrupted", finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_write_json(status, status_path)
        raise
    except Exception as exc:
        status.update(status="failed", reason=str(exc), finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_write_json(status, status_path)
        raise


def _load_surface_cfg_overrides(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"--surface-config-json must be a JSON object: {p}")
    return obj


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = default_measurement_config()
    cfg["enable_plots"] = not bool(args.no_plots)
    cfg["verbose"] = not bool(args.quiet)
    cfg["observed_tangent_source"] = args.observed_tangent_source
    measure_only = bool(args.measure_only or args.no_surface)
    surface_only = bool(args.surface_only)
    tag = validate_surface_tag(args.surface_tag)
    try:
        surface_overrides = _load_surface_cfg_overrides(args.surface_config_json)
    except Exception as exc:  # noqa: BLE001
        print(f"surface-config-json failed: {exc}", file=sys.stderr)
        return 1

    try:
        if surface_only:
            root = project_root_from_here(__file__)
            _setup_import_paths(root)
            scfg = surface_config_for_run(
                args.run_id, surface_overrides, surface_tag=tag, project_root=root
            )
            df, status = run_saved_surface(
                args.directory_id, args.run_id, frame_reference=args.frame_reference,
                surface_cfg=scfg, project_root=root, surface_tag=tag,
                verbose=bool(cfg["verbose"]), segment_ids=args.segment_id,
                enable_plots=bool(cfg["enable_plots"]),
            )
            if cfg["verbose"]:
                print(f"surface-only done: tag={tag}; status={status['status']}; n={len(df)}")
            return 0 if status["status"] == "complete" else 2

        _, status = run_directory(
            args.directory_id,
            args.run_id,
            frame_reference=args.frame_reference,
            cfg=cfg,
            segment_ids=args.segment_id,
            do_surface=not measure_only,
            surface_cfg=surface_overrides or None,
            surface_tag=tag,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"participant failed: {args.directory_id}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0 if status["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
