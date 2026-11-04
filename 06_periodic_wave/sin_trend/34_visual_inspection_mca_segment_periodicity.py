#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 34 visual inspection mca segment periodicity."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_mca_labels import (  # noqa: E402
    GEOMETRY_HASH_COL,
    SHAPE_FINGERPRINT_COLS,
    analysis_revision_from_df,
    build_work_from_source,
    compute_centerline_geometry_hash,
    load_manual_labels,
    manual_labels_fingerprint,
    mca_is_set,
    save_manual_labels,
    should_refresh_work,
    source_content_fingerprint,
    sync_confirmed_labels_to_manual,
    upsert_manual_label_row,
)
from sin_trend_measure import build_groupid_map  # noqa: E402
from sin_trend_paths import (  # noqa: E402
    CLASSIFICATION_REVISION,
    FRAME_BASELINE,
    FRAME_REFERENCES,
    INTERVAL_KEY,
    OUTPUT_TAG,
    analysis_csv_path,
    atomic_to_csv,
    atomic_write_json,
    mca_manual_labels_csv,
    mca_work_csv_path,
    mca_work_frame_root,
    project_root_from_here,
    validate_frame_reference,
    validate_run_id,
)
from sin_trend_io_utils import (  # noqa: E402
    find_duplicate_keys,
    normalize_id_columns,
    same_file,
)

SURFACE_EVALUABLE = {
    "Predominantly tangential",
    "Substantial normal component",
    "In-plane Buckling",  # legacy
    "Out-of-plane",  # legacy
}
VALID_MCA_LABELS = {"M1", "M2-3", "M4"}
ANALYSIS_REVISION = CLASSIFICATION_REVISION

PROFILE_COLUMNS = {
    "longwave_r2": ("periodicity", "surface_constraint"),
}

PLOTTER_WINDOW_SIZE = (2560, 1440)
PLOTTER_POSITION = (500, 500)
CHANGE_HISTORY_NAME = "mca_label_change_history.csv"
WORK_META_NAME = "work_csv_metadata.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Interactive MCA classification ({OUTPUT_TAG})")
    p.add_argument("--run-id", required=True)
    p.add_argument("--frame-reference", default=FRAME_BASELINE, choices=list(FRAME_REFERENCES))
    p.add_argument("--source-csv", type=Path, default=None)
    p.add_argument("--work-csv", type=Path, default=None)
    p.add_argument(
        "--core-profile",
        choices=list(PROFILE_COLUMNS.keys()),
        default="longwave_r2",
        help="Which periodicity/surface columns to filter",
    )
    p.add_argument("--mode", choices=["add", "recheck"], default="add")
    p.add_argument("--dry-run", action="store_true", help="Print targets only; no GUI")
    p.add_argument(
        "--include-surface-unevaluable",
        action="store_true",
        help="also include Core-periodic intervals that are surface-inevaluable (default: surface-evaluable only)",
    )
    p.add_argument(
        "--core-inplane-only",
        action="store_true",
        help="restrict anatomical review to core_inplane (Predominantly tangential) only",
    )
    p.add_argument("--start-directory-id", default=None)
    p.add_argument("--start-segment-id", type=int, default=None)
    return p.parse_args(argv)


def profile_column_names(core_profile: str) -> Tuple[str, str]:
    if core_profile not in PROFILE_COLUMNS:
        raise ValueError(f"unknown core_profile: {core_profile}")
    return PROFILE_COLUMNS[core_profile]


def _mca_is_set(val: Any) -> bool:
    return mca_is_set(val)


def _analysis_revision_from_df(df: pd.DataFrame) -> str:
    return analysis_revision_from_df(df)


def list_targets(
    df: pd.DataFrame,
    core_profile: str,
    *,
    mode: str = "add",
    start_directory_id: Optional[str] = None,
    start_segment_id: Optional[int] = None,
    require_surface: bool = True,
    core_inplane_only: bool = False,
) -> pd.DataFrame:
    per_col, surf_col = profile_column_names(core_profile)
    if per_col not in df.columns:
        raise ValueError(f"DataFrame missing columns for {core_profile}: {per_col}")

    missing_keys = [c for c in INTERVAL_KEY if c not in df.columns]
    if missing_keys:
        raise ValueError(f"DataFrame missing interval key columns: {missing_keys}")

    # periodicity Core: support both string "Core" and bool core_periodic/core_all
    from sin_trend_dataset import parse_bool_series

    if per_col in df.columns and df[per_col].dtype == object and per_col not in ("core_periodic", "core_all"):
        core_mask = df[per_col].astype(str) == "Core"
    elif "core_periodic" in df.columns:
        core_mask = parse_bool_series(df["core_periodic"]).fillna(False)
    elif "core_all" in df.columns:
        core_mask = parse_bool_series(df["core_all"]).fillna(False)
    else:
        core_mask = parse_bool_series(df[per_col]).fillna(False)

    if require_surface:
        if surf_col not in df.columns:
            raise ValueError(f"DataFrame missing surface column {surf_col}")
        mask = core_mask & (df[surf_col].isin(SURFACE_EVALUABLE))
    else:
        mask = core_mask
    if core_inplane_only:
        if "core_inplane" not in df.columns:
            raise ValueError("DataFrame missing column core_inplane (required by --core-inplane-only)")
        mask = mask & parse_bool_series(df["core_inplane"]).fillna(False)
    sub = df.loc[mask].copy()
    if mode == "add":
        from sin_trend_dataset import mca_label_confirmed, mca_needs_review
        from sin_trend_mca_labels import _row_is_unresolved_conflict

        pending = sub.apply(_row_is_unresolved_conflict, axis=1) if len(sub) else pd.Series(False, index=sub.index)
        needs = sub.apply(mca_needs_review, axis=1) if len(sub) else pd.Series(False, index=sub.index)
        unset = ~sub["mca_segment"].map(_mca_is_set) if "mca_segment" in sub.columns else pd.Series(True, index=sub.index)
        # include unset OR unverified non-empty labels OR unresolved conflicts
        sub = sub[unset | needs | pending]
    if len(sub) == 0:
        return sub.reindex(columns=list(df.columns))
    sub = sub.sort_values(list(INTERVAL_KEY))

    if mode != "recheck" or (start_directory_id is None and start_segment_id is None):
        return sub

    dir_ids = sorted(sub["directory_id"].dropna().unique().tolist())
    start_dir_pos = (
        dir_ids.index(start_directory_id)
        if start_directory_id is not None and start_directory_id in dir_ids
        else 0
    )
    keep_rows: List[int] = []
    for idx, row in sub.sort_values(list(INTERVAL_KEY)).iterrows():
        did = row["directory_id"]
        sid = row["segment_id"]
        if start_directory_id is not None and did in dir_ids:
            if dir_ids.index(did) < start_dir_pos:
                continue
        if (
            start_directory_id is not None
            and str(did) == str(start_directory_id)
            and start_segment_id is not None
            and int(sid) < int(start_segment_id)
        ):
            continue
        keep_rows.append(idx)
    return sub.loc[keep_rows]



def apply_mca_label(
    df: pd.DataFrame,
    key: Dict[str, Any],
    label: Optional[str],
    *,
    manual_labels_path: Optional[Path] = None,
    P_vox: Optional[np.ndarray] = None,
    voxel_size_mm: Optional[float] = None,
) -> bool:
    """Confirm one displayed interval, persist first, then change the work table.

    A stale view may not close a newer episode. Same-label confirmation records
    the action, while window dismissal never calls this function.
    """
    from sin_trend_mca_labels import (
        MCA_CONFLICT_TOKEN_COL, MCA_CONFLICT_AT_COL, LAST_RESOLVED_TOKEN_COL,
        ensure_persistent_conflict_tokens, _row_is_unresolved_conflict,
    )
    from sin_trend_mca_registry import (
        ConflictRegistry, REGISTRY_TOKEN_COL, StaleLabelViewError,
        label_write_lock, registry_path_for_manual, text,
    )
    if label not in VALID_MCA_LABELS:
        raise ValueError(f"invalid mca label: {label!r}")
    mask = pd.Series(True, index=df.index)
    for k in INTERVAL_KEY:
        if k not in key or k not in df.columns:
            return False
        mask &= df[k] == key[k]
    if not mask.any():
        return False
    if int(mask.sum()) != 1:
        raise ValueError(f"Expected one interval for key {key}, found {int(mask.sum())}")
    proposed = ensure_persistent_conflict_tokens(df)
    idx = proposed.index[mask][0]
    prior = proposed.loc[idx].copy()
    geom = text(prior.get(GEOMETRY_HASH_COL))
    if P_vox is not None:
        if voxel_size_mm is None or not np.isfinite(float(voxel_size_mm)) or float(voxel_size_mm) <= 0:
            raise ValueError("A positive voxel_size_mm is required for displayed geometry")
        P = np.asarray(P_vox, float)
        if not np.isfinite(P).all():
            raise ValueError("Displayed geometry contains nonfinite coordinates")
        actual = compute_centerline_geometry_hash(P, voxel_size_mm=float(voxel_size_mm))
        if geom and geom.lower() != actual.lower():
            raise StaleLabelViewError("Displayed centerline differs from the measured geometry; rerun measurement before labeling")
        geom = actual
        proposed.at[idx, GEOMETRY_HASH_COL] = geom
    expected = text(prior.get(MCA_CONFLICT_TOKEN_COL)) if _row_is_unresolved_conflict(prior) else ""
    view = text(prior.get(REGISTRY_TOKEN_COL))

    def update_work(token: str) -> None:
        proposed.at[idx, "mca_segment"] = label
        proposed.at[idx, "mca_label_conflict"] = False
        proposed.at[idx, "label_status"] = "ok"
        proposed.at[idx, "mca_conflict_labels"] = ""
        proposed.at[idx, MCA_CONFLICT_TOKEN_COL] = ""
        proposed.at[idx, MCA_CONFLICT_AT_COL] = ""
        if token:
            proposed.at[idx, LAST_RESOLVED_TOKEN_COL] = token
            proposed.at[idx, REGISTRY_TOKEN_COL] = token

    if manual_labels_path is not None:
        manual_labels_path = Path(manual_labels_path)
        with label_write_lock(manual_labels_path):
            with ConflictRegistry(registry_path_for_manual(manual_labels_path)) as registry:
                # Legacy pending CSVs are adopted once. Existing registry state
                # is never replaced just because a stale GUI was still open.
                registry.observe_existing(prior)
                registry.attach_displayed_geometry(prior, proposed.loc[idx])
                token = registry.confirm(proposed.loc[idx], str(label), expected,
                                         expected_view_token=view)
                update_work(token)
                row = proposed.loc[idx]
                shape = {c: row.get(c, np.nan) for c in SHAPE_FINGERPRINT_COLS}
                shape[GEOMETRY_HASH_COL] = geom
                labels = load_manual_labels(manual_labels_path)
                labels = upsert_manual_label_row(labels, key, str(label), shape,
                                                origin="gui", label_status="ok",
                                                resolved_conflict_token=token or None)
                save_manual_labels(labels, manual_labels_path)
                # A save failure rolls back the registry and leaves df unchanged.
    else:
        update_work(expected)
    for col in proposed.columns:
        df[col] = proposed[col]
    return True


def _unique_mca_labels(work: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sin_trend_mca_labels import unique_labels_by_key

    return unique_labels_by_key(work)



def prepare_work_csv(
    source: Path, work: Path, *, manual_labels_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Refresh measurements from source and labels from shared persistent state.

    Rebuild labels even when source bytes are unchanged: another frame may have
    created a newer conflict in the shared registry. Do not lock during display.
    """
    from sin_trend_mca_labels import _confirmed_label_table
    from sin_trend_mca_registry import (
        ConflictRegistry, LABEL_STATE_REVISION, label_write_lock,
        registry_path_for_manual, reconcile_registered, source_identity,
    )
    if same_file(source, work):
        raise ValueError("source-csv and work-csv must not be the same file")
    labels_path = (Path(manual_labels_path) if manual_labels_path is not None
                   else work.parent.parent / "mca_manual_labels.csv")
    if manual_labels_path is None and (work.parent / "mca_manual_labels.csv").is_file():
        labels_path = work.parent / "mca_manual_labels.csv"
    with label_write_lock(labels_path):
        src_df = normalize_id_columns(pd.read_csv(source, low_memory=False))
        if len(find_duplicate_keys(src_df)):
            raise ValueError("Duplicate interval keys in source")
        prev = normalize_id_columns(pd.read_csv(work, low_memory=False)) if work.is_file() else None
        if prev is not None and len(find_duplicate_keys(prev)):
            raise ValueError("Duplicate interval keys in work")
        manual = load_manual_labels(labels_path)
        eid, sha = source_identity(labels_path) if labels_path.is_file() else (None,None)
        with ConflictRegistry(registry_path_for_manual(labels_path)) as registry:
            out, conflicts = reconcile_registered(
                src_df, update_labels=manual if len(manual) else None,
                fill_labels=_confirmed_label_table(prev) if prev is not None else None,
                registry=registry, event_id=eid, source_fingerprint=sha,
            )
            atomic_to_csv(out, work, index=False)
        atomic_to_csv(conflicts, work.parent / "mca_work_inherit_conflicts.csv", index=False)
        _, n_synced = sync_confirmed_labels_to_manual(out, labels_path, origin="sync_from_work", fill_only=True)
        meta = {
            "label_state_revision": LABEL_STATE_REVISION,
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "refresh_reason": "source_manual_and_shared_registry",
            "source_path": str(source), "source_sha256": source_content_fingerprint(source),
            "source_revision": _analysis_revision_from_df(src_df) or ANALYSIS_REVISION,
            "manual_labels_path": str(labels_path),
            "manual_labels_sha256": manual_labels_fingerprint(labels_path),
            "n_rows": len(out), "n_labeled": int(out["mca_segment"].map(_mca_is_set).sum()),
            "n_manual_synced": n_synced,
        }
        atomic_write_json(meta, work.parent / WORK_META_NAME)
    print(f"refreshed work and shared label state: {work}")
    return out


def append_change_history(
    work_dir: Path,
    directory_id: str,
    segment_id: Any,
    segment_index: Any,
    old_val: Any,
    new_val: Any,
) -> None:
    path = work_dir / CHANGE_HISTORY_NAME
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "directory_id": directory_id,
        "segment_id": segment_id,
        "segment_index": segment_index,
        "old_mca_segment": old_val,
        "new_mca_segment": new_val,
    }
    if path.is_file():
        hist = pd.read_csv(path)
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    atomic_to_csv(hist, path, index=False)


def _save_work(df: pd.DataFrame, work: Path) -> None:
    atomic_to_csv(df, work, index=False)


def finalize_mca_gui_choice(
    *,
    user_acted: bool,
    chosen_label: Any,
    current_label: Any,
    mode: str,
) -> Tuple[str, str]:
    if not user_acted:
        if mode == "add" or not _mca_is_set(current_label):
            return "abort_missing", ""
        return "dismissed", ""
    if not _mca_is_set(chosen_label):
        if mode == "add" or not _mca_is_set(current_label):
            return "abort_missing", ""
        return "apply", str(current_label).strip()
    return "apply", str(chosen_label).strip()


def visualize_and_update_mca_segment(
    *,
    project_root: Path,
    segment_id: Any,
    row: pd.Series,
    df_all: pd.DataFrame,
    directory_id: str,
    work_csv: Path,
    mode: str,
    periodicity_col: str,
    case_context: dict,
    manual_labels_path: Optional[Path] = None,
) -> Tuple[bool, bool, bool]:
    import pyvista as pv                

    from sin_trend_fitting_helpers import clip_mesh_to_coords

    cell_index = row["segment_index"]
    mca_raw = row.get("mca_segment", None)
    current_mca_segment = None if not _mca_is_set(mca_raw) else str(mca_raw).strip()

    print("\n" + "=" * 80)
    print(f"Processing directory_id: {directory_id}, segment_id: {segment_id}, segment_index: {cell_index}")
    print(f"periodicity: {row.get(periodicity_col, 'N/A')}")
    print(f"Current MCA label: {current_mca_segment}")
    print("=" * 80)

    pointids_dict = case_context["pointids_dict"]
    if segment_id not in pointids_dict:
        print(f"display skip: segment_id {segment_id} not in centerline GroupIds")
        return False, False, False

    points = case_context["points"]
    P_vox_3darr = points[pointids_dict[segment_id]]
    right_cells = case_context["right_cells"]
    brain_mesh_dir = case_context["brain_mesh_dir"]
    vessel_mesh = case_context["vessel_mesh"]
    m1_mesh = case_context["m1_mesh"]

    if cell_index in right_cells:
        side = "Rt"
        brain_mesh_path = os.path.join(brain_mesh_dir, f"{directory_id}_rt_pial_remesh.ply")
    else:
        side = "Lt"
        brain_mesh_path = os.path.join(brain_mesh_dir, f"{directory_id}_lt_pial_remesh.ply")

    if not os.path.exists(brain_mesh_path):
        print(f"display skip: brain mesh not found: {brain_mesh_path}")
        return False, False, False

    brain_mesh = pv.read(brain_mesh_path)
    cropped_mesh = clip_mesh_to_coords(P_vox_3darr, brain_mesh, margin=20)

    mca_segment = current_mca_segment if mode == "recheck" else None
    user_acted = False                              
    pl = None

    def _close_plotter():
        nonlocal pl
        if pl is not None:
            try:
                pl.close()
            except Exception:  # noqa: BLE001
                pass

    def set_m1():
        nonlocal mca_segment, user_acted
        mca_segment = "M1"
        user_acted = True
        print("mca_segment set to 'M1'")
        _close_plotter()

    def set_m2_3():
        nonlocal mca_segment, user_acted
        mca_segment = "M2-3"
        user_acted = True
        print("mca_segment set to 'M2-3'")
        _close_plotter()

    def set_m4():
        nonlocal mca_segment, user_acted
        mca_segment = "M4"
        user_acted = True
        print("mca_segment set to 'M4'")
        _close_plotter()

    def set_no_change():
        nonlocal mca_segment, user_acted
        mca_segment = current_mca_segment
        user_acted = True
        print(f"No change (current value: {current_mca_segment})")
        _close_plotter()

    pl = pv.Plotter(shape=(1, 2), window_size=PLOTTER_WINDOW_SIZE)
    focus = np.asarray(P_vox_3darr).mean(axis=0)
    mid_pt = P_vox_3darr[len(P_vox_3darr) // 2]

    pl.add_text("Press key: M1 = 1, M2-3 = 2 or 3, M4 = 4", position="lower_left", font_size=18, color="red")
    pl.add_text(f"Current: {current_mca_segment}", position="upper_right", font_size=18, color="blue")
    pl.add_text(f"Periodicity: {row.get(periodicity_col, 'N/A')}", position="upper_left", font_size=14, color="green")

    pl.subplot(0, 0)
    pl.add_mesh(vessel_mesh, color="lightblue", opacity=0.5)
    pl.add_mesh(m1_mesh, color="blue", opacity=0.3)
    pl.add_mesh(brain_mesh, color="ivory", opacity=0)
    pl.add_points(P_vox_3darr, color="red", point_size=5)
    pl.add_points(mid_pt, color="red", opacity=0.8, point_size=30, render_points_as_spheres=False, ambient=1.0)
    pl.add_mesh(cropped_mesh, color="ivory", opacity=0.8)
    pl.camera_position = "xy"
    pl.camera.azimuth = 180
    pl.set_focus(focus)
    pl.camera.zoom(3)
    pl.show_bounds(grid="front", location="outer", all_edges=True, ticks="outside")

    pl.subplot(0, 1)
    pl.add_mesh(vessel_mesh, color="lightblue", opacity=0.5)
    pl.add_mesh(m1_mesh, color="blue", opacity=0.3)
    pl.add_mesh(brain_mesh, color="ivory", opacity=1)
    pl.add_mesh(cropped_mesh, color="lightgreen", opacity=1)
    pl.add_points(P_vox_3darr, color="red", point_size=5)
    pl.add_points(mid_pt, color="red", opacity=0.8, point_size=30, render_points_as_spheres=False, ambient=1.0)
    pl.camera_position = "zy"
    pl.camera.azimuth = 0 if side == "Lt" else 180
    pl.camera.zoom(2.0)
    pl.set_focus(focus)
    pl.show_bounds(grid="front", location="outer", all_edges=True, ticks="outside")
    pl.add_title(
        f"MCA Classification Check, {directory_id}, segment ID: {segment_id}, segment index: {cell_index}",
        font_size=18,
    )

    for key, cb in [("1", set_m1), ("2", set_m2_3), ("3", set_m2_3), ("4", set_m4)]:
        pl.add_key_event(key, cb)
        pl.add_key_event(f"KP_{key}", cb)
    if mode == "recheck":
        pl.add_key_event("space", set_no_change)

    pl.render()
    if hasattr(pl, "render_window") and pl.render_window is not None:
        pl.render_window.SetPosition(PLOTTER_POSITION[0], PLOTTER_POSITION[1])
    try:
        pl.show()
    except (AttributeError, RuntimeError):
        pass

    action, new_label = finalize_mca_gui_choice(
        user_acted=user_acted,
        chosen_label=mca_segment,
        current_label=current_mca_segment,
        mode=mode,
    )
    if action == "abort_missing":
        print("Error: mca_segment unset (add mode requires 1/2/3/4)")
        return False, False, False
    if action == "dismissed":
        print("  Window closed only (no confirm/change)")
        return True, False, False

    old_label = str(current_mca_segment).strip() if _mca_is_set(current_mca_segment) else ""
    label_changed = new_label != old_label

    key = {k: row[k] for k in INTERVAL_KEY}
    voxel_mm = float(row["voxel_size_mm"]) if "voxel_size_mm" in row.index and pd.notna(row.get("voxel_size_mm")) else None
    if voxel_mm is None or not np.isfinite(voxel_mm):
        voxel_mm = 0.5
                                                        
    if not apply_mca_label(
        df_all,
        key,
        new_label,
        manual_labels_path=manual_labels_path,
        P_vox=P_vox_3darr,
        voxel_size_mm=voxel_mm,
    ):
        print(f"Error: segment not found: {key}")
        return False, False, False

    _save_work(df_all, work_csv)
    if label_changed:
        append_change_history(work_csv.parent, directory_id, segment_id, cell_index, old_label, new_label)
        print(f"  saved mca_segment: {old_label} -> {new_label}")
    else:
        print(f"  confirmed mca_segment (unchanged): {new_label} -> label_status=ok")
    return True, label_changed, True


def load_case_context(project_root: Path, directory_id: str) -> Optional[dict]:
    utils_path = project_root / "lib"
    mesh_path = project_root / "03_mesh"
    for p in (project_root / "lib", project_root / "03_mesh", utils_path, mesh_path):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import centerline_extraction
    import mesh_generation
    import vessel_mesh_features

    vessel_centerlines_dir = project_root / "data" / "derived" / "vessel_centerlines" / directory_id
    cl_path = vessel_centerlines_dir / f"{directory_id}_merged_centerlines.vtp"
    if not cl_path.is_file():
        print(f"  skip case {directory_id}: centerline missing: {cl_path}")
        return None

    merged = centerline_extraction.load_vtp(str(cl_path))
    celldata, pointids, points, _pointdata = centerline_extraction.vmtk_convert2numpy(merged)
    group_ids = list(celldata["GroupIds"])
                                     
    pointids_dict, _cell_index_of, gid_diag = build_groupid_map(group_ids, pointids, points)
    if len(gid_diag):
        amb = gid_diag[gid_diag["resolution"] == "ambiguous_excluded"]
        if len(amb):
            print(f"  warning: {len(amb)} ambiguous GroupId(s) excluded from GUI map (same rule as measurement)")
    right_cells, _left_cells, _tx, _ir, _il = vessel_mesh_features.classify_centerline_left_right_by_kmeans(
        points, pointids, n_iter=100, verbose=False
    )

    vessel_mesh_dir = project_root / "data" / "preprocessed" / "MRA_preprocessed" / "vessel_meshes" / directory_id
    vessel_mesh_path = vessel_mesh_dir / f"{directory_id}_vessel_mesh.ply"
    m1_mesh_path = vessel_mesh_dir / f"{directory_id}_M1_vessel_mesh.ply"
    if not vessel_mesh_path.is_file() or not m1_mesh_path.is_file():
        print(f"  skip case {directory_id}: vessel mesh missing")
        return None

    return {
        "pointids_dict": pointids_dict,
        "points": points,
        "right_cells": right_cells,
        "vessel_mesh": mesh_generation.load_ply_to_polydata(str(vessel_mesh_path)),
        "m1_mesh": mesh_generation.load_ply_to_polydata(str(m1_mesh_path)),
        "brain_mesh_dir": project_root / "data" / "preprocessed" / "T1_preprocessed" / "brain_meshes" / directory_id,
        "groupid_diagnostics": gid_diag,
    }


def run_interactive_gui(args: argparse.Namespace) -> int:
    root = project_root_from_here(__file__)
    run_id = validate_run_id(args.run_id)
    frame = validate_frame_reference(args.frame_reference)
    source = Path(args.source_csv) if args.source_csv else analysis_csv_path(run_id, frame, root)
    work = Path(args.work_csv) if args.work_csv else mca_work_csv_path(run_id, frame, root)
    labels_path = mca_manual_labels_csv(run_id, root)

    if not source.is_file():
        print(f"Error: missing source CSV: {source}")
        return 1
    if same_file(source, work):
        print("Error: source-csv and work-csv must not be the same file")
        return 1

    df_all = prepare_work_csv(source, work, manual_labels_path=labels_path)
    dup = find_duplicate_keys(df_all)
    if len(dup):
        print(f"Error: duplicate interval keys in work CSV: {len(dup)}")
        return 1

    per_col, surf_col = profile_column_names(args.core_profile)
    require_surface = not bool(args.include_surface_unevaluable)
    core_inplane_only = bool(args.core_inplane_only)
    targets = list_targets(
        df_all,
        args.core_profile,
        mode=args.mode,
        start_directory_id=args.start_directory_id,
        start_segment_id=args.start_segment_id,
        require_surface=require_surface,
        core_inplane_only=core_inplane_only,
    )
    print(f"core_profile={args.core_profile} mode={args.mode} targets={len(targets)}")
    print(
        f"  columns: {per_col}, {surf_col}; require_surface={require_surface}; "
        f"core_inplane_only={core_inplane_only}"
    )
    print(f"  work CSV (frame={frame}): {work}")
    print(f"  manual labels (shared): {labels_path}")

    if len(targets) == 0:
        print("No targets to process.")
        return 0

    directory_ids = sorted(targets["directory_id"].dropna().unique().tolist())
    total_changed = 0
    total_confirmed = 0
    total_failed = 0

    print("\nKeys: 1→M1, 2/3→M2-3, 4→M4", end="")
    if args.mode == "recheck":
        print(", space→no change (save as reviewed even if label unchanged)")
    else:
        print(" (space disabled in add mode)")

    for directory_id in directory_ids:
        df_dir = targets[targets["directory_id"] == directory_id]
        try:
            case_ctx = load_case_context(root, str(directory_id))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip case {directory_id}: load failed: {exc}")
            total_failed += len(df_dir)
            continue
        if case_ctx is None:
            total_failed += len(df_dir)
            continue

        for _, row in df_dir.sort_values(list(INTERVAL_KEY)).iterrows():
            segment_id = row["segment_id"]
            try:
                ok, changed, confirmed = visualize_and_update_mca_segment(
                    project_root=root,
                    segment_id=segment_id,
                    row=row,
                    df_all=df_all,
                    directory_id=str(directory_id),
                    work_csv=work,
                    mode=args.mode,
                    periodicity_col=per_col,
                    case_context=case_ctx,
                    manual_labels_path=labels_path,
                )
                if ok and changed:
                    total_changed += 1
                if ok and confirmed:
                    total_confirmed += 1
                if not ok:
                    total_failed += 1
            except KeyboardInterrupt:
                print("\nInterrupted.")
                return 130
            except Exception as exc:  # noqa: BLE001
                total_failed += 1
                print(
                    f"  Error: {directory_id}, segment_id={segment_id}, "
                    f"segment_index={row.get('segment_index')}: {exc}"
                )

    print(
        f"\nDone. changed={total_changed}, confirmed={total_confirmed}, "
        f"failed/skipped={total_failed}"
    )
                                    
    from sin_trend_mca_registry import label_write_lock
    with label_write_lock(labels_path):
        _, n_synced = sync_confirmed_labels_to_manual(df_all, labels_path, origin="sync_after_gui", fill_only=True)
    if n_synced:
        print(f"synced {n_synced} unchanged confirmed labels into manual table")
    print(f"work CSV: {work}")
    print(f"manual labels: {labels_path}")
    script_23 = _HERE / "33_prepare_core_inplane_dataset.py"
    print(
        "Re-integrate labels into formal analysis dataset "
        "(previous formal labels are kept as base; manual updates overlay):\n"
        f'  python "{script_23}" --run-id {run_id} --frame-reference {frame} --mca-source-csv "{labels_path}"'
    )
    return 2 if total_failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root_from_here(__file__)
    run_id = validate_run_id(args.run_id)
    frame = validate_frame_reference(args.frame_reference)
    source = Path(args.source_csv) if args.source_csv else analysis_csv_path(run_id, frame, root)
    work = Path(args.work_csv) if args.work_csv else mca_work_csv_path(run_id, frame, root)
    labels_path = mca_manual_labels_csv(run_id, root)

    if not source.is_file():
        print(f"Error: missing source CSV: {source}")
        return 1
    if same_file(source, work):
        print("Error: source-csv and work-csv must not be the same file")
        return 1

    if args.dry_run:
        if work.is_file():
            base = prepare_work_csv(source, work, manual_labels_path=labels_path)
        else:
            base = normalize_id_columns(pd.read_csv(source, low_memory=False))
        require_surface = not bool(args.include_surface_unevaluable)
        core_inplane_only = bool(args.core_inplane_only)
        targets = list_targets(
            base,
            args.core_profile,
            mode=args.mode,
            start_directory_id=args.start_directory_id,
            start_segment_id=args.start_segment_id,
            require_surface=require_surface,
            core_inplane_only=core_inplane_only,
        )
        per_col, surf_col = profile_column_names(args.core_profile)
        print(f"[dry-run] output_tag={OUTPUT_TAG} analysis_revision={ANALYSIS_REVISION}")
        print(f"[dry-run] source={source}")
        print(f"[dry-run] work={work}")
        print(f"[dry-run] manual_labels={labels_path}")
        print(f"[dry-run] core_profile={args.core_profile} mode={args.mode}")
        if require_surface:
            print(f"[dry-run] filter: {per_col}==Core and {surf_col} in {sorted(SURFACE_EVALUABLE)}")
        else:
            print(f"[dry-run] filter: {per_col}==Core (surface unevaluable included)")
        if core_inplane_only:
            print("[dry-run] filter: core_inplane==True")
        print(
            f"[dry-run] require_surface={require_surface} "
            f"core_inplane_only={core_inplane_only} target count: {len(targets)}"
        )
        if len(targets):
            ids = targets[list(INTERVAL_KEY)].head(20)
            print("[dry-run] first targets (up to 20):")
            print(ids.to_string(index=False))
        print("[dry-run] GUI not run.")
        return 0

    return 0


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.dry_run:
        raise SystemExit(main())
    raise SystemExit(run_interactive_gui(cli_args))
