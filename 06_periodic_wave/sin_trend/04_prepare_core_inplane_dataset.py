#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 33 prepare core inplane dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_classify import classify_dataframe as apply_classification_dataframe  # noqa: E402
from sin_trend_mca_labels import (  # noqa: E402
    LABEL_STATUS_UNSET,
    apply_manual_labels,
    ensure_label_status_columns,
    inherit_mca_labels_layered,
    mca_is_set,
)
from sin_trend_paths import (  # noqa: E402
    CLASSIFICATION_REVISION,
    FRAME_BASELINE,
    FRAME_REFERENCES,
    INTERVAL_KEY,
    OUTPUT_TAG,
    analysis_csv_path,
    analysis_root,
    atomic_to_csv,
    atomic_write_json,
    legacy_v1_mca_label_csv,
    long_csv_path,
    surface_long_csv_path,
    validate_surface_tag,
    mca_manual_labels_csv,
    metadata_dict,
    parse_json_bool,
    participants_csv_path,
    project_root_from_here,
    validate_frame_reference,
    validate_run_id,
)
from sin_trend_selection_diagnostics import flow_counts, reject_reason_table  # noqa: E402
from sin_trend_io_utils import (  # noqa: E402
    find_duplicate_keys,
    normalize_id_columns,
    same_file,
)

SURFACE_EVALUABLE = {
    "Predominantly tangential",
    "Substantial normal component",
                       
    "In-plane Buckling",
    "Out-of-plane",
}
SURFACE_SUBSTANTIAL_NORMAL = {"Substantial normal component", "Out-of-plane"}
UNSET_MCA_COLUMNS: Sequence[str] = (
    "directory_id",
    "segment_id",
    "segment_index",
    "core_all",
    "core_inplane",
    "periodicity",
    "surface_constraint",
    "R2",
    "lambda_hat_mm",
    "amp_comb_mm",
    "mca_label_conflict",
    "label_status",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Prepare analysis dataset ({OUTPUT_TAG})")
    p.add_argument("--run-id", required=True)
    p.add_argument("--frame-reference", default=FRAME_BASELINE, choices=list(FRAME_REFERENCES))
    p.add_argument("--morphometrics-csv", type=Path, default=None)
    p.add_argument("--surface-tag", default=None, help="read this surface-tag long CSV; non-primary analyses stay in separate directories")
    p.add_argument("--participants-csv", type=Path, default=None)
    p.add_argument(
        "--mca-source-csv",
        type=Path,
        default=None,
        help=(
            "Authoritative MCA label update CSV. "
            "Default when omitted: shared mca_manual_labels.csv if present; "
            "else legacy v1 only on first migration (no previous analysis). "
            "Legacy is never the default overwrite source once confirmed labels exist."
        ),
    )
    p.add_argument("--mca-event-id", default=None,
                   help="Stable id for a NEW external import/audit; reuse the same id for both frames. "
                        "Default: source-content SHA256 (identical bytes mean retry, not new audit).")
    p.add_argument("--reclassify", action="store_true", help="re-apply classification columns from stored measurements")
    p.add_argument("--skip-mca", action="store_true")
    return p.parse_args(argv)


def resolve_mca_update_source(
    *,
    run_id: str,
    project_root: Path,
    explicit: Optional[Path],
    previous_analysis_exists: bool,
) -> Tuple[Optional[Path], str]:
    if explicit is not None:
        return Path(explicit), "explicit"
    manual = mca_manual_labels_csv(run_id, project_root)
    if manual.is_file():
        return manual, "shared_manual"
    if not previous_analysis_exists:
        legacy = legacy_v1_mca_label_csv(project_root)
        if legacy.is_file():
            return legacy, "legacy_initial"
    return None, "none"


def _mca_is_set(val: Any) -> bool:
    return mca_is_set(val)


def _assert_no_duplicate_keys(df: pd.DataFrame, label: str) -> None:
    dup = find_duplicate_keys(df)
    if len(dup):
        raise ValueError(f"duplicate interval keys in {label}: {len(dup)} rows")


def _normalize_id_value(v: Any) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, float) and np.isfinite(v) and float(v).is_integer():
        return str(int(v))
    s = str(v).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    # numeric-looking floats as strings
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return s


def _values_compatible(a: Any, b: Any, *, col: str) -> bool:
    id_like = col.lower() in {"ixi_id", "participant_id", "directory_id", "subject_id"}
    if id_like:
        return _normalize_id_value(a) == _normalize_id_value(b)
    na = pd.to_numeric(pd.Series([a]), errors="coerce").iloc[0]
    nb = pd.to_numeric(pd.Series([b]), errors="coerce").iloc[0]
    if np.isfinite(na) and np.isfinite(nb):
        return bool(np.isclose(float(na), float(nb), rtol=0, atol=0) or float(na) == float(nb))
    return str(a).strip() == str(b).strip()


def merge_participants(df: pd.DataFrame, parts: pd.DataFrame) -> pd.DataFrame:
    parts = normalize_id_columns(parts, keys=("directory_id",))
    if parts.duplicated(subset=["directory_id"]).any():
        raise ValueError("participants CSV has duplicate directory_id")
    n_before = len(df)
    attr_cols = [c for c in parts.columns if c != "directory_id"]
    merged = df.merge(
        parts[["directory_id"] + attr_cols],
        on="directory_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_from_participants"),
    )
    if len(merged) != n_before:
        raise ValueError(f"participants merge changed row count: {n_before} -> {len(merged)}")

    def _is_missing(series: pd.Series) -> pd.Series:
        if series.dtype == object:
            return series.isna() | (series.astype(str).str.strip() == "") | (series.astype(str).str.lower() == "nan")
        return series.isna()

    for c in attr_cols:
        src = f"{c}_from_participants"
        if src not in merged.columns:
            continue
        if c not in df.columns:
            merged[c] = merged[src]
        else:
            miss = _is_missing(merged[c])
            both = ~miss & ~_is_missing(merged[src])
            if both.any():
                for i in merged.index[both]:
                    if not _values_compatible(merged.at[i, c], merged.at[i, src], col=c):
                        raise ValueError(
                            f"participants attribute conflict on column {c!r}: "
                            f"{merged.at[i, c]!r} vs {merged.at[i, src]!r} (directory_id={merged.at[i, 'directory_id']!r})"
                        )
            merged.loc[miss, c] = merged.loc[miss, src]
        merged.drop(columns=[src], inplace=True)

    # age: coalesce across aliases per row; conflict if two finite different ages
    age_aliases = [c for c in ("AGE", "age", "Age", "age_years") if c in merged.columns]
    age_years = pd.Series(np.nan, index=merged.index, dtype=float)
    for i in merged.index:
        vals = []
        for c in age_aliases:
            v = pd.to_numeric(merged.at[i, c], errors="coerce")
            if np.isfinite(v):
                vals.append(float(v))
        if not vals:
            continue
        if max(vals) - min(vals) > 1e-9:
            raise ValueError(f"conflicting age aliases for directory_id={merged.at[i, 'directory_id']!r}: {vals}")
        age_years.at[i] = vals[0]
    merged["age_years"] = age_years
    merged["age_available"] = age_years.notna().astype(int)
    merged["participant_matched"] = merged["directory_id"].isin(set(parts["directory_id"].astype(str))).astype(int)
    return merged


def inherit_mca_segment(df: pd.DataFrame, mca_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = ensure_label_status_columns(normalize_id_columns(df))
    src = normalize_id_columns(pd.read_csv(mca_csv, low_memory=False))
    if "mca_segment" not in src.columns:
        raise ValueError(f"MCA source missing mca_segment: {mca_csv}")
    _assert_no_duplicate_keys(out, "analysis dataframe")
    from sin_trend_mca_labels import shape_fingerprint_available

    require_shape = shape_fingerprint_available(src)
    merged, conflicts = apply_manual_labels(
        out,
        src,
        require_shape=require_shape,
        allow_resolve_conflicts=True,
        authoritative=True,
    )
    return merged, conflicts


def build_mca_unset_table(df: pd.DataFrame) -> pd.DataFrame:
    core = df["core_all"].fillna(0).astype(bool) if "core_all" in df.columns else pd.Series(False, index=df.index)
    missing = ~df["mca_segment"].map(_mca_is_set) if "mca_segment" in df.columns else pd.Series(True, index=df.index)
    conflict = df["mca_label_conflict"].fillna(False).astype(bool) if "mca_label_conflict" in df.columns else False
                                    
    sub = df.loc[core & (missing | conflict)].copy()
    cols = [c for c in UNSET_MCA_COLUMNS if c in sub.columns]
    return sub[cols] if cols else pd.DataFrame(columns=list(UNSET_MCA_COLUMNS))


def _metric_summary(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    finite = s[np.isfinite(s)]
    return {
        "n_total": int(len(s)),
        "n_valid": int(np.isfinite(s).sum()),
        "n_nonfinite": int((~np.isfinite(s.fillna(np.nan))).sum()) if len(s) else 0,
        "mean": float(finite.mean()) if len(finite) else np.nan,
        "median": float(finite.median()) if len(finite) else np.nan,
        "q1": float(finite.quantile(0.25)) if len(finite) else np.nan,
        "q3": float(finite.quantile(0.75)) if len(finite) else np.nan,
    }


def surface_coverage_qc(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    if "core_all" not in df.columns:
        return pd.DataFrame(rows)
    core = df[df["core_all"].fillna(0).astype(bool)]
    surf_col = "surface_constraint"
    surf_eval = core[core[surf_col].isin(SURFACE_EVALUABLE)] if surf_col in core.columns else core.iloc[0:0]
    inplane = df[df["core_inplane"].fillna(0).astype(bool)] if "core_inplane" in df.columns else core.iloc[0:0]
    mca_ok = inplane[inplane["mca_segment"].map(_mca_is_set)] if "mca_segment" in inplane.columns else inplane.iloc[0:0]
    groups = [
        ("measured_all", df),
        ("fit_valid", df[df["fit_valid"].fillna(0).astype(bool)] if "fit_valid" in df.columns else df.iloc[0:0]),
        ("core_all", core),
        ("core_surface_evaluable", surf_eval),
        ("core_inplane", inplane),
        ("core_inplane_with_mca", mca_ok),
    ]
    metrics = ["squared_displacement_coverage", "R_out_surface", "evaluated_tangential_fraction", "partial_R2", "amp_comb_mm", "SNR", "hr_amp"]
    for gname, sub in groups:
        for metric in metrics:
            if metric not in sub.columns:
                continue
            rows.append({"group": gname, "metric": metric, **_metric_summary(sub[metric])})
    return pd.DataFrame(rows)


def _main_unlocked(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root_from_here(__file__)
    run_id = validate_run_id(args.run_id)
    frame = validate_frame_reference(args.frame_reference)
    surface_tag = validate_surface_tag(args.surface_tag) if args.surface_tag is not None else None
    out_dir = analysis_root(run_id, frame, root)
    if surface_tag is not None and surface_tag != "primary":
        out_dir = out_dir / "surface" / surface_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    default_morph_csv = (
        surface_long_csv_path(run_id, frame, surface_tag, root)
        if surface_tag is not None and surface_tag != "primary"
        else long_csv_path(run_id, frame, root)
    )
    morph_csv = Path(args.morphometrics_csv) if args.morphometrics_csv else default_morph_csv
    selected_analysis_csv = out_dir / analysis_csv_path(run_id, frame, root).name
    part_csv = Path(args.participants_csv) if args.participants_csv else participants_csv_path(root)

    if not morph_csv.is_file():
        print(f"Error: missing morphometrics CSV: {morph_csv}")
        return 1
    if not part_csv.is_file():
        print(f"Error: missing participants CSV: {part_csv}")
        return 1

    df = normalize_id_columns(pd.read_csv(morph_csv, low_memory=False))
    n_morph = len(df)
    if surface_tag is not None and len(df):
        if "surface_tag" not in df or not df["surface_tag"].notna().all() or not df["surface_tag"].astype(str).eq(surface_tag).all():
            print(f"Error: morphometrics CSV does not contain only surface_tag={surface_tag}")
            return 3
        if "surface_config_hash" not in df or not df["surface_config_hash"].notna().all() or df["surface_config_hash"].astype(str).nunique() != 1:
            print("Error: morphometrics CSV has missing or mixed surface_config_hash")
            return 3
    _assert_no_duplicate_keys(df, "morphometrics long CSV")
    df = merge_participants(df, pd.read_csv(part_csv, low_memory=False))

    if args.reclassify:
        df = apply_classification_dataframe(df)

    from sin_trend_dataset import (
        add_derived_columns,
        filter_mca_m2m4_core_inplane,
        filter_mca_m2m4_core_periodic,
        filter_mca_needs_review,
        filter_paired_amplitude,
        filter_paired_wavelength,
    )
    from sin_trend_paths import read_run_config, version_revision_consistency

    try:
        run_cfg = read_run_config(run_id, root)
    except FileNotFoundError as exc:
        atomic_write_json(
            {"reason": "run_config_missing", "detail": str(exc)},
            out_dir / "rejected_input_run_config.json",
        )
        print(f"Error: run_config required for formal 33 path: {exc}")
        return 3
    except Exception as exc:  # noqa: BLE001
        atomic_write_json(
            {"reason": "run_config_unreadable", "detail": str(exc)},
            out_dir / "rejected_input_run_config.json",
        )
        print(f"Error: run_config unreadable: {exc}")
        return 3
    cons = version_revision_consistency(
        df, expected_frame=frame, run_config=run_cfg, require_run_config=True
    )
    if not cons["ok"]:
        atomic_write_json(
            {"consistency": cons, "expected_frame": frame},
            out_dir / "rejected_input_version_revision_frame.json",
        )
        print(f"Error: morphometrics CSV failed version/method/unit check: {cons.get('reasons')}")
        return 3

                                    
                                                    
                                                                                 
    agg_meta: Dict[str, Any] = {}
    upstream_aggregation_incomplete: Optional[bool] = None
    upstream_all_requests_complete: Optional[bool] = None
    long_meta_path = morph_csv.parent / "long_format_metadata.json"
    if args.morphometrics_csv:
        upstream_aggregation_incomplete = None  # unknown
        upstream_all_requests_complete = None
    elif long_meta_path.is_file():
        try:
            import json

            with open(long_meta_path, "r", encoding="utf-8") as f:
                agg_meta = json.load(f)
            if "aggregation_incomplete" in agg_meta:
                upstream_aggregation_incomplete = parse_json_bool(agg_meta.get("aggregation_incomplete"))
            elif "incomplete" in agg_meta:
                upstream_aggregation_incomplete = parse_json_bool(agg_meta.get("incomplete"))
            if "all_requests_complete" in agg_meta:
                upstream_all_requests_complete = parse_json_bool(agg_meta.get("all_requests_complete"))
        except Exception:  # noqa: BLE001
            upstream_aggregation_incomplete = None
            upstream_all_requests_complete = None
    else:
        upstream_aggregation_incomplete = None
        upstream_all_requests_complete = None

    df = add_derived_columns(df)

    if "core_all" in df.columns and "surface_constraint" in df.columns:
        from sin_trend_dataset import parse_bool_series as _pbs

        _core_all = _pbs(df["core_all"]).fillna(False)
        df["core_outofplane"] = _core_all & df["surface_constraint"].isin(SURFACE_SUBSTANTIAL_NORMAL)
        df["core_surface_evaluable"] = _core_all & df["surface_constraint"].isin(SURFACE_EVALUABLE)
    df["output_tag"] = OUTPUT_TAG
    df["analysis_revision"] = CLASSIFICATION_REVISION

    conflicts = pd.DataFrame()
    prev_analysis_path = selected_analysis_csv
                                       
    prev_labels_df: Optional[pd.DataFrame] = None
    update_df: Optional[pd.DataFrame] = None
    fill_df: Optional[pd.DataFrame] = None
    mca_csv: Optional[Path] = None
    mca_source_role = "skipped"
    legacy_fill_used = False
    if not args.skip_mca:
        if prev_analysis_path.is_file() and not same_file(morph_csv, prev_analysis_path):
            try:
                prev_labels_df = normalize_id_columns(pd.read_csv(prev_analysis_path, low_memory=False))
                print(f"using previous analysis labels as base: {prev_analysis_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Error: previous analysis CSV is unreadable ({exc}); existing results were not overwritten")
                return 1

        mca_csv, mca_source_role = resolve_mca_update_source(
            run_id=run_id,
            project_root=root,
            explicit=Path(args.mca_source_csv) if args.mca_source_csv else None,
            previous_analysis_exists=prev_labels_df is not None,
        )
        print(f"mca update source role={mca_source_role}: {mca_csv}")

        if mca_csv is not None:
            if same_file(morph_csv, mca_csv):
                print("Error: mca-source-csv must not be the same file as morphometrics CSV")
                return 1
            if prev_labels_df is not None and same_file(prev_analysis_path, mca_csv):
                                                       
                update_df = None
                mca_source_role = "previous_analysis_base_only"
                print("mca-source-csv is the previous analysis CSV; using it as base only")
            elif mca_csv.is_file():
                update_df = normalize_id_columns(pd.read_csv(mca_csv, low_memory=False))
                if "mca_segment" not in update_df.columns:
                    print(f"Error: MCA source missing mca_segment: {mca_csv}")
                    return 1
            else:
                print(f"MCA source missing: {mca_csv}")
                if mca_source_role == "explicit":
                    return 1
                update_df = None
                mca_csv = None

                                         
                                                       
        legacy_path = legacy_v1_mca_label_csv(root)
        if (
            mca_source_role != "legacy_initial"
            and legacy_path.is_file()
            and (mca_csv is None or not same_file(mca_csv, legacy_path))
            and (prev_labels_df is None or not same_file(prev_analysis_path, legacy_path))
        ):
            try:
                fill_df = normalize_id_columns(pd.read_csv(legacy_path, low_memory=False))
                if "mca_segment" in fill_df.columns:
                    legacy_fill_used = True
                    print(f"using legacy MCA labels as fill-only (unset rows): {legacy_path}")
                else:
                    fill_df = None
            except Exception as exc:  # noqa: BLE001
                print(f"warning: could not read legacy MCA CSV for fill ({exc})")
                fill_df = None

    from sin_trend_mca_registry import (
        ConflictRegistry, LABEL_STATE_REVISION, registry_path_for_manual,
        reconcile_registered, source_identity,
    )
    event_id, event_sha = None, None
    if args.mca_event_id is not None and args.mca_source_csv is None:
        raise ValueError("--mca-event-id requires an explicit --mca-source-csv")
    if update_df is not None and mca_csv is not None:
        event_id, event_sha = source_identity(mca_csv, args.mca_event_id)
    conflict_history = pd.DataFrame(columns=list(INTERVAL_KEY) + ["labels", "conflict_type"])
    if not args.skip_mca:
        _assert_no_duplicate_keys(df, "analysis dataframe")
        with ConflictRegistry(registry_path_for_manual(mca_manual_labels_csv(run_id, root))) as registry:
            df, conflicts = reconcile_registered(
                df, base_labels=prev_labels_df, update_labels=update_df,
                fill_labels=fill_df, registry=registry,
                event_id=event_id, source_fingerprint=event_sha,
            )
        hist = df.attrs.get("conflict_history")
        if isinstance(hist, pd.DataFrame):
            conflict_history = hist
        active_events = df.attrs.get("conflict_active_events")
        n_active_events = (
            int(len(active_events))
            if isinstance(active_events, pd.DataFrame)
            else int(len(conflicts))
        )
        if len(conflicts):
            atomic_to_csv(conflicts, out_dir / "mca_label_conflicts.csv", index=False)
            print(
                f"MCA label conflicts: {len(conflicts)} unresolved interval(s) "
                f"({n_active_events} diagnostic event row(s); not auto-resolved)"
            )
        else:
            atomic_to_csv(
                pd.DataFrame(columns=list(INTERVAL_KEY) + ["labels", "conflict_type"]),
                out_dir / "mca_label_conflicts.csv",
                index=False,
            )
            n_active_events = 0
    else:
        n_active_events = 0
        atomic_to_csv(
            pd.DataFrame(columns=list(INTERVAL_KEY) + ["labels", "conflict_type"]),
            out_dir / "mca_label_conflicts.csv",
            index=False,
        )
    atomic_to_csv(conflict_history, out_dir / "mca_conflict_history.csv", index=False)
    if isinstance(df.attrs.get("conflict_active_events"), pd.DataFrame):
        atomic_to_csv(
            df.attrs["conflict_active_events"],
            out_dir / "mca_conflict_active_events.csv",
            index=False,
        )
    if len(conflict_history) or n_active_events:
        print(
            f"MCA conflict counts: unresolved_intervals={len(conflicts)} "
            f"active_events={n_active_events} history_records={len(conflict_history)}"
        )

    if "mca_segment" not in df.columns:
        df["mca_segment"] = np.nan
    df = ensure_label_status_columns(df)
                    
    unset = ~df["mca_segment"].map(_mca_is_set) & ~df["mca_label_conflict"].fillna(False).astype(bool)
    df.loc[unset, "label_status"] = df.loc[unset, "label_status"].replace("", LABEL_STATUS_UNSET)
    df.loc[unset & (df["label_status"].isna() | (df["label_status"].astype(str).str.strip() == "")), "label_status"] = LABEL_STATUS_UNSET

    unset = build_mca_unset_table(df)
    atomic_to_csv(unset, out_dir / "mca_segment_unset_for_core.csv", index=False)
    print(f"Core rows without mca_segment: {len(unset)}")

    qc = surface_coverage_qc(df)
    atomic_to_csv(qc, out_dir / "qc_surface_coverage_by_cohort.csv", index=False)

    out_csv = selected_analysis_csv
    atomic_to_csv(df, out_csv, index=False)

    # purpose-specific subset tables (do not silently drop from the full table)
    partials = {
        "MCA_M2M4_core_periodic.csv": filter_mca_m2m4_core_periodic(df),
        "MCA_M2M4_core_inplane.csv": filter_mca_m2m4_core_inplane(df),
        "mca_needs_review.csv": filter_mca_needs_review(df),
        "paired_wavelength_coordinate_intervals.csv": filter_paired_wavelength(df),
        "paired_amplitude_definition_intervals.csv": filter_paired_amplitude(df),
        "core_periodic.csv": (
            df[df["core_periodic"].fillna(0).astype(bool)].copy()
            if "core_periodic" in df.columns
            else (
                df[df["core_all"].fillna(0).astype(bool)].copy()
                if "core_all" in df.columns
                else df.iloc[0:0].copy()
            )
        ),
        "core_inplane.csv": (
            df[df["core_inplane"].fillna(0).astype(bool)].copy()
            if "core_inplane" in df.columns
            else df.iloc[0:0].copy()
        ),
    }
    partial_counts = {}
    for name, sub in partials.items():
        atomic_to_csv(sub, out_dir / name, index=False)
        partial_counts[name] = int(len(sub))

    # reject reason counts
    rr = reject_reason_table(df)
    if len(rr):
        atomic_to_csv(rr, out_dir / "reject_reasons_counts.csv", index=False)

    selection = flow_counts(df)
    atomic_write_json(selection, out_dir / "selection_flow.json")

    meta = metadata_dict(
        run_id=run_id,
        frame_reference=frame,
        surface_tag=surface_tag or agg_meta.get("surface_tag"),
        morphometrics_csv=str(morph_csv),
        long_format_metadata_path=str(long_meta_path),
        participants_csv=str(part_csv),
        mca_source_csv=str(mca_csv) if mca_csv else None,
        mca_source_role=mca_source_role,
        mca_legacy_fill_only=bool(legacy_fill_used),
        previous_analysis_labels_base=str(prev_analysis_path) if prev_labels_df is not None else None,
        output_csv=str(out_csv),
        n_rows=int(len(df)),
        n_morph_rows=int(n_morph),
        n_core_periodic=int(df["core_periodic"].fillna(0).sum()) if "core_periodic" in df.columns else None,
        n_core_all=int(df["core_all"].fillna(0).sum()) if "core_all" in df.columns else None,
        n_core_inplane=int(df["core_inplane"].fillna(0).sum()) if "core_inplane" in df.columns else None,
        n_mca_unset_core=int(len(unset)),
        n_mca_conflicts=int(len(conflicts)),  # unique unresolved intervals
        n_mca_conflict_active_events=int(n_active_events),
        n_mca_conflict_history=int(len(conflict_history)),
        partial_table_counts=partial_counts,
        selection_flow=selection,
        paired_table_note=(
            "paired / both-sides tables use all eligible rows with attributes; "
            "MCA Core confirmed subsets are separate partial CSVs (not the paired denominator)."
        ),
        upstream_aggregation_incomplete=upstream_aggregation_incomplete,
        upstream_all_requests_complete=upstream_all_requests_complete,
        analysis_csv_written_ok=True,
        all_requests_complete=upstream_all_requests_complete,
        reclassify=bool(args.reclassify),
        label_state_revision=LABEL_STATE_REVISION,
        mca_import_event_id=event_id,
        mca_import_source_sha256=event_sha,
        mca_registry_path=str(registry_path_for_manual(mca_manual_labels_csv(run_id, root))),
        mca_legacy_source_note="results/analysis/sin_trend/IXI_HH_sin_trend_analysis_dataset.csv (labels only)",
    )
    atomic_write_json(meta, out_dir / "analysis_dataset_metadata.json")
    atomic_write_json(meta, out_dir / f"analysis_metadata_{CLASSIFICATION_REVISION}.json")
    print(
        f"saved: {out_csv} rows={len(df)} "
        f"core_periodic={meta.get('n_core_periodic')} core_inplane={meta['n_core_inplane']}"
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Serialize the full read/merge/write operation across frames of one run."""
    from sin_trend_mca_registry import label_write_lock
    args = parse_args(argv)
    root = project_root_from_here(__file__)
    with label_write_lock(mca_manual_labels_csv(validate_run_id(args.run_id), root)):
        return _main_unlocked(argv)


if __name__ == "__main__":
    raise SystemExit(main())
