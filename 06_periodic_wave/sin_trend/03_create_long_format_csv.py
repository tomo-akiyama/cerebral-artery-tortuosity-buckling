#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 32 create long format csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_HERE = Path(__file__).resolve().parent
# Prefer this package directory on sys.path (same basename modules must not collide).
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)

from sin_trend_measure import measurement_row_schema  # noqa: E402
from sin_trend_paths import (  # noqa: E402
    FRAME_BASELINE,
    FRAME_REFERENCES,
    OUTPUT_TAG,
    atomic_to_csv,
    atomic_write_json,
    build_run_manifest,
    ensure_empty_csv,
    long_csv_path,
    metadata_dict,
    morphometrics_root,
    participants_csv_path,
    per_case_csv_path,
    project_root_from_here,
    read_participant_status,
    read_run_config,
    read_surface_config,
    read_surface_participant_status,
    sha256_file,
    surface_per_case_dir,
    surface_per_case_csv_path,
    surface_long_csv_path,
    validate_surface_tag,
    validate_frame_reference,
    validate_run_id,
    version_revision_consistency,
    write_run_manifest,
)
from sin_trend_selection_diagnostics import flow_counts, reject_reason_table  # noqa: E402
from sin_trend_io_utils import (                            
    INTERVAL_KEY,
    assert_interval_keys_present,
    find_duplicate_keys,
    normalize_id_columns,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Create long-format CSV ({OUTPUT_TAG})")
    p.add_argument("--run-id", required=True)
    p.add_argument("--frame-reference", default=FRAME_BASELINE, choices=list(FRAME_REFERENCES))
    p.add_argument("--directory-id", action="append", default=None, help="restrict to these participants (smoke)")
    stage = p.add_mutually_exclusive_group()
    stage.add_argument("--surface-tag", default=None, help="aggregate only this surface tag (default: primary if available)")
    stage.add_argument("--measurement-only", action="store_true", help="aggregate immutable measurement CSVs only")
    return p.parse_args(argv)


def create_long_format(
    run_id: str,
    frame_reference: str,
    project_root: Optional[Path] = None,
    directory_ids: Optional[List[str]] = None,
    surface_tag: Optional[str] = None,
    measurement_only: bool = False,
) -> Dict[str, Any]:
    root = Path(project_root) if project_root is not None else project_root_from_here(__file__)
    run_id = validate_run_id(run_id)
    frame_reference = validate_frame_reference(frame_reference)
    cfg = read_run_config(run_id, root)                  

    if surface_tag is not None and measurement_only:
        raise ValueError("surface_tag and measurement_only are mutually exclusive")
    tag = validate_surface_tag(surface_tag) if surface_tag is not None else None
    if tag is None and not measurement_only:
        primary_dir = surface_per_case_dir(run_id, frame_reference, "primary", root)
        if any(primary_dir.glob("*_vessel_segmentdata.csv")):
            tag = "primary"
    surface_record = None
    if tag is not None:
        surface_record = read_surface_config(run_id, tag, root)
    out_csv = (
        surface_long_csv_path(run_id, frame_reference, tag, root)
        if tag is not None and tag != "primary"
        else long_csv_path(run_id, frame_reference, root)
    )
    stage_meta = dict(
        aggregation_stage="surface" if tag is not None else "measurement",
        surface_tag=tag,
        surface_config_hash=surface_record.get("config_hash") if surface_record else None,
    )
    diag_dir = out_csv.parent
    diag_dir.mkdir(parents=True, exist_ok=True)
    schema_cols = list(measurement_row_schema().keys())

    participants = pd.read_csv(participants_csv_path(root))
    all_ids = sorted(participants["directory_id"].dropna().astype(str).unique().tolist())
    target_ids = [d for d in all_ids if d in set(directory_ids)] if directory_ids else all_ids
    unknown_ids = sorted(set(directory_ids or []) - set(all_ids))

    frames: List[pd.DataFrame] = []
    failed: List[Dict[str, Any]] = []
    missing_participants: List[str] = []
    ambiguous_rows: List[pd.DataFrame] = []
    status_summary: List[Dict[str, Any]] = []
    for did in target_ids:
        try:
            st = (
                read_surface_participant_status(run_id, frame_reference, did, tag, root)
                if tag is not None else read_participant_status(run_id, frame_reference, did, root)
            )
        except (OSError, ValueError) as exc:
            st = {"status": "status_unreadable", "status_read_error": str(exc)}
        if tag is not None and st is not None and st.get("status") != "status_unreadable":
            if st.get("surface_tag") != tag or st.get("surface_config_hash") != surface_record["config_hash"]:
                return {"rc": 6, "reason": f"surface status tag/config mismatch: {did}"}
        if st is None:
            case_status = "missing_status"
        else:
            case_status = str(st.get("status", "unknown"))
            if case_status == "status_unreadable":
                # keep same vocabulary as reader; optional detail in reason field
                err = st.get("status_read_error", "")
                if err:
                    case_status = f"status_unreadable:{err}"
        status_summary.append({"directory_id": did, "participant_status": case_status})

        path = (
            surface_per_case_csv_path(run_id, frame_reference, did, tag, root)
            if tag is not None else per_case_csv_path(run_id, frame_reference, did, root)
        )
        if not path.is_file():
            missing_participants.append(did)
            failed.append({"directory_id": did, "reason": "missing", "participant_status": case_status})
            continue
        measurement_hash = None
        if tag is not None:
            measurement_path = per_case_csv_path(run_id, frame_reference, did, root)
            provenance = {
                "surface_tag": tag, "directory_id": did, "input_csv": str(path),
                "measurement_csv": str(measurement_path),
            }
            reason = None
            if not isinstance(st, dict) or not st.get("per_case_csv_sha256"):
                reason = "surface status is missing the surface CSV hash"
            elif not measurement_path.is_file():
                reason = "source measurement CSV is missing"
            else:
                try:
                    surface_hash = sha256_file(path)
                    measurement_hash = sha256_file(measurement_path)
                    provenance.update(
                        surface_csv_sha256=surface_hash,
                        status_surface_csv_sha256=st["per_case_csv_sha256"],
                        current_measurement_csv_sha256=measurement_hash,
                    )
                    if st["per_case_csv_sha256"] != surface_hash:
                        reason = "surface CSV hash differs from its status record"
                except OSError as exc:
                    reason = f"surface/source CSV hash read failed: {exc}"
            if reason is not None:
                atomic_write_json({**provenance, "reason": reason}, diag_dir / "invalid_surface_provenance.json")
                return {"rc": 6, "reason": f"{reason}: {did}", **stage_meta}
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            failed.append({"directory_id": did, "reason": str(exc), "participant_status": case_status})
            continue
        if len(df) == 0:
            failed.append({"directory_id": did, "reason": "empty", "participant_status": case_status})
            continue
        if tag is not None:
            source_key = "surface_source_measurement_csv_sha256"
            if source_key not in df or not df[source_key].notna().all() or not df[source_key].astype(str).eq(measurement_hash).all():
                atomic_write_json(
                    {**provenance, "reason": "surface rows were not generated from the current measurement CSV", "failed_column": source_key},
                    diag_dir / "invalid_surface_provenance.json",
                )
                return {"rc": 6, "reason": f"source measurement CSV hash mismatch in {path}", **stage_meta}
            expected = {"surface_tag": tag, "surface_config_hash": surface_record["config_hash"]}
            code_hash = surface_record.get("surface_source_code_hash_combined")
            if code_hash:
                expected["surface_code_hash"] = code_hash
            for key, value in expected.items():
                if key not in df or not df[key].notna().all() or not df[key].astype(str).eq(str(value)).all():
                    atomic_write_json(
                        {"surface_tag": tag, "expected_config_hash": surface_record["config_hash"], "failed_column": key, "input_csv": str(path)},
                        diag_dir / "mixed_surface_tag_config.json",
                    )
                    return {"rc": 6, "reason": f"surface tag/config/code mismatch in {path}: {key}"}
        if "directory_id" in df.columns:
            mism = df["directory_id"].astype(str) != did
            if bool(mism.any()):
                atomic_to_csv(df.loc[mism, ["directory_id"]].drop_duplicates(), diag_dir / f"id_mismatch_{did}.csv", index=False)
                return {"rc": 2, "reason": f"directory_id mismatch in {path}"}
        amb = df["fit_status"].astype(str) == "ambiguous_groupid" if "fit_status" in df.columns else pd.Series(False, index=df.index)
        if bool(amb.any()):
            ambiguous_rows.append(df.loc[amb])
            df = df.loc[~amb]
        frames.append(df)

    result: Dict[str, Any] = {"rc": 0, **stage_meta}
    fail_cols = ["directory_id", "reason", "participant_status"]
    status_cols = ["directory_id", "participant_status"]
    atomic_to_csv(
        pd.DataFrame({"directory_id": missing_participants}) if missing_participants else pd.DataFrame(columns=["directory_id"]),
        diag_dir / "missing_participants.csv",
        index=False,
    )
    atomic_to_csv(pd.DataFrame(failed, columns=fail_cols) if failed else pd.DataFrame(columns=fail_cols), diag_dir / "per_case_read_failures.csv", index=False)
    atomic_to_csv(
        pd.DataFrame(status_summary, columns=status_cols) if status_summary else pd.DataFrame(columns=status_cols),
        diag_dir / "requested_participant_status.csv",
        index=False,
    )
    amb_df = pd.concat(ambiguous_rows, ignore_index=True) if ambiguous_rows else pd.DataFrame(columns=schema_cols)
    atomic_to_csv(amb_df, diag_dir / "ambiguous_groupid_rows.csv", index=False)

    incomplete_status_tokens = {
        "partial",
        "interrupted",
        "running",
        "failed",
        "missing",
        "missing_status",
        "status_unreadable",
    }

    def _status_incomplete(st: str) -> bool:
        s = str(st)
        return (
            s in incomplete_status_tokens
            or s.startswith("status_read_failed")
            or s.startswith("status_unreadable")
        )

    n_status_complete = sum(1 for r in status_summary if r["participant_status"] == "complete")
    n_status_incomplete = sum(1 for r in status_summary if _status_incomplete(r["participant_status"]))
                                             
    aggregation_incomplete = bool(
        missing_participants or failed or unknown_ids or n_status_incomplete > 0 or n_status_complete < len(target_ids)
    )
    all_requests_complete = (not aggregation_incomplete) and n_status_complete == len(target_ids) and len(target_ids) > 0

    if not frames:
        empty_df = pd.DataFrame(columns=schema_cols)
        cons0 = version_revision_consistency(
            empty_df, expected_frame=frame_reference, run_config=cfg, require_run_config=True
        )
        if not cons0["ok"]:
            atomic_write_json(
                {"consistency": cons0, "expected_frame": frame_reference},
                diag_dir / "mixed_version_revision_frame.json",
            )
            return {"rc": 3, "reason": f"run_config/schema rejected on empty aggregation: {cons0}"}
        ensure_empty_csv(out_csv, schema_cols)
        ensure_empty_csv(diag_dir / "reject_reasons_counts.csv", ["reason", "n"])
        meta = metadata_dict(
            run_id=run_id,
            frame_reference=frame_reference,
            **stage_meta,
            output_csv=str(out_csv),
            n_rows=0,
            n_participants_requested=len(target_ids),
            n_participants_ok=0,
            n_participants_missing=len(missing_participants),
            n_participants_unknown=len(unknown_ids),
            n_participants_status_complete=n_status_complete,
            n_participants_status_incomplete=n_status_incomplete,
            unknown_directory_ids=unknown_ids,
            n_ambiguous_groupid_rows=int(len(amb_df)),
            incomplete=True,
            aggregation_incomplete=True,
            csv_written_ok=True,
            all_requests_complete=False,
            failed=failed,
            consistency=cons0,
            selection_flow=None,
        )
        atomic_write_json(meta, diag_dir / "long_format_metadata.json")
        if tag is None:
                                                    
            manifest = build_run_manifest(run_id, target_ids, [frame_reference], root)
            try:
                import json

                from sin_trend_paths import run_manifest_path

                mp = run_manifest_path(run_id, root)
                if mp.is_file():
                    with open(mp, "r", encoding="utf-8") as f:
                        old = json.load(f)
                    old.setdefault("frames", {}).update(manifest["frames"])
                    old["updated_by_32"] = meta.get("created_utc")
                    manifest = old
            except Exception:  # noqa: BLE001
                pass
            write_run_manifest(manifest, run_id, root)
        result.update(n_rows=0, out_csv=str(out_csv), incomplete=True, aggregation_incomplete=True)
        return result

    long_df = pd.concat(frames, ignore_index=True)
    for c in schema_cols:
        if c not in long_df.columns:
            long_df[c] = pd.NA
    long_df = long_df[schema_cols + [c for c in long_df.columns if c not in schema_cols]]
    long_df = normalize_id_columns(long_df, INTERVAL_KEY)

    cons = version_revision_consistency(
        long_df, expected_frame=frame_reference, run_config=cfg, require_run_config=True
    )
    if not cons["ok"]:
        atomic_write_json({"consistency": cons, "expected_frame": frame_reference}, diag_dir / "mixed_version_revision_frame.json")
        return {"rc": 3, "reason": f"mixed measurement_version/revision/frame/unit: {cons}"}

    try:
        assert_interval_keys_present(long_df, INTERVAL_KEY, where="(long_df)")
    except ValueError as exc:
        return {"rc": 4, "reason": str(exc)}
    dups = find_duplicate_keys(long_df, INTERVAL_KEY)
    if len(dups):
        atomic_to_csv(dups, diag_dir / "duplicate_interval_ids.csv", index=False)
        return {"rc": 5, "reason": f"duplicate interval IDs n_rows={len(dups)}; formal long CSV NOT written"}

    atomic_to_csv(long_df, out_csv, index=False)
    fit_counts = long_df["fit_status"].astype(str).value_counts().to_dict()
    surf_counts = long_df["surface_evaluation_status"].astype(str).value_counts().to_dict()
    selection = flow_counts(long_df)
    rr = reject_reason_table(long_df)
    atomic_to_csv(rr if len(rr) else pd.DataFrame(columns=["reason", "n"]), diag_dir / "reject_reasons_counts.csv", index=False)
    meta = metadata_dict(
        run_id=run_id,
        frame_reference=frame_reference,
        **stage_meta,
        output_csv=str(out_csv),
        n_rows=int(len(long_df)),
        n_participants_requested=len(target_ids),
        n_participants_ok=len(frames),
        n_participants_missing=len(missing_participants),
        n_participants_unknown=len(unknown_ids),
        n_participants_status_complete=n_status_complete,
        n_participants_status_incomplete=n_status_incomplete,
        unknown_directory_ids=unknown_ids,
        n_ambiguous_groupid_rows=int(len(amb_df)),
        incomplete=aggregation_incomplete,
        aggregation_incomplete=aggregation_incomplete,
        csv_written_ok=True,
        all_requests_complete=all_requests_complete,
        failed=failed,
        consistency=cons,
        fit_status_counts=fit_counts,
        surface_status_counts=surf_counts,
        selection_flow=selection,
        paired_table_note="long CSV keeps all readable intervals; MCA Core subsets are built in 33, not here.",
        note_eligibility_reject_is_not_failure="numerical Core rejection and surface unevaluable are kept as rows; they do not alone mark aggregation incomplete",
    )
    atomic_write_json(meta, diag_dir / "long_format_metadata.json")

    if tag is None:
                                     
        manifest = build_run_manifest(run_id, target_ids, [frame_reference], root)
        try:
            import json

            from sin_trend_paths import run_manifest_path

            mp = run_manifest_path(run_id, root)
            if mp.is_file():
                with open(mp, "r", encoding="utf-8") as f:
                    old = json.load(f)
                old.setdefault("frames", {}).update(manifest["frames"])
                old["updated_by_32"] = meta.get("created_utc")
                manifest = old
        except Exception:  # noqa: BLE001
            pass
        write_run_manifest(manifest, run_id, root)

    result.update(n_rows=int(len(long_df)), out_csv=str(out_csv), incomplete=meta["incomplete"], fit_status_counts=fit_counts)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    res = create_long_format(
        args.run_id, args.frame_reference, directory_ids=args.directory_id,
        surface_tag=args.surface_tag, measurement_only=args.measurement_only,
    )
    if res["rc"] != 0:
        print(f"Error: {res.get('reason')}")
        return int(res["rc"])
    print(f"saved: {res['out_csv']} rows={res['n_rows']} incomplete={res['incomplete']} fit_status={res.get('fit_status_counts')}")
    if res["incomplete"]:
        print("Warning: incomplete aggregation; see missing_participants.csv / per_case_read_failures.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
