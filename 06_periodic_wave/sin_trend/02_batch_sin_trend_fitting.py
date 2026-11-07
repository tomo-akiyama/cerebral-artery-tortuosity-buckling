#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 31 batch sin trend fitting."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from sin_trend_measure import default_measurement_config  # noqa: E402
from sin_trend_paths import (  # noqa: E402
    FRAME_BASELINE,
    FRAME_REFERENCES,
    OUTPUT_TAG,
    atomic_to_csv,
    atomic_write_json,
    build_run_manifest,
    centerline_vtp_path,
    per_case_csv_path,
    per_case_dir,
    surface_config_for_run,
    new_run_id,
    participants_csv_path,
    project_root_from_here,
    read_participant_status,
    read_surface_participant_status,
    run_root,
    split_measure_surface_config,
    validate_frame_reference,
    validate_run_id,
    validate_surface_tag,
    verify_participant_complete_for_resume,
    write_run_config,
    write_run_manifest,
    write_surface_config,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Batch run ({OUTPUT_TAG})")
    p.add_argument("--run-id", default=None, help="run_id; omitted -> new timestamped id")
    p.add_argument("--frame-reference", action="append", default=None, choices=list(FRAME_REFERENCES))
    p.add_argument("--directory-id", action="append", default=None)
    p.add_argument("--start-directory-id", default=None)
    p.add_argument("--limit", type=int, default=None, help="process at most N participants (smoke)")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--timeout-sec", type=int, default=3 * 3600)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--all", action="store_true", help="explicitly allow running the full participant list")
    p.add_argument("--execute", action="store_true", help="actually launch subprocess workers")
    p.add_argument("--dry-run", action="store_true", help="plan only (default when --execute is omitted)")
    stage = p.add_mutually_exclusive_group()
    stage.add_argument("--measure-only", action="store_true", help="measurement stage only (pass-through to runner)")
    stage.add_argument("--surface-only", action="store_true", help="surface stage only from existing measurement outputs")
    stage.add_argument("--no-surface", action="store_true", help="alias of --measure-only (compat)")
    p.add_argument("--surface-tag", default="primary", help="surface config tag (default: primary)")
    p.add_argument(
        "--surface-config-json",
        type=Path,
        default=None,
        help="optional surface settings JSON passed to each runner",
    )
    plot_group = p.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--enable-plots",
        dest="plots",
        action="store_true",
        help="save PNG/HTML visualizations (default)",
    )
    plot_group.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="skip PNG/HTML visualizations",
    )
    p.set_defaults(plots=True)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _format_elapsed(sec: float) -> str:
    total = max(0, int(round(float(sec))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        human = f"{h}h {m}m"
    elif m > 0:
        human = f"{m}m {s}s"
    else:
        human = f"{s}s"
    return f"{total}s ({human})"


def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    env.setdefault("PYVISTA_OFF_SCREEN", "true")
    return env


def run_one_participant(
    *,
    runner: Path,
    project_root: Path,
    directory_id: str,
    run_id: str,
    frames: List[str],
    args: argparse.Namespace,
    log_dir: Path,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fr in frames:
        resume_recheck: Optional[str] = None
        measure_only = bool(args.measure_only or args.no_surface)
        surface_only = bool(args.surface_only)
        if args.resume and not surface_only:
            st = read_participant_status(run_id, fr, directory_id, project_root)
            if measure_only:
                ok, reason = verify_participant_complete_for_resume(
                    st,
                    run_id=run_id,
                    frame_reference=fr,
                    directory_id=directory_id,
                    project_root=project_root,
                    expect_surface=False,
                    expect_segment_ids=None,
                )
                if ok:
                    out.append(
                        {
                            "directory_id": directory_id,
                            "frame_reference": fr,
                            "outcome": "skipped_complete",
                            "returncode": None,
                            "elapsed_sec": 0.0,
                            "resume_reason": reason,
                        }
                    )
                    continue
                resume_recheck = reason
            else:
                                                           
                ok_full, reason_full = verify_participant_complete_for_resume(
                    st,
                    run_id=run_id,
                    frame_reference=fr,
                    directory_id=directory_id,
                    project_root=project_root,
                    expect_surface=True,
                    expect_segment_ids=None,
                )
                if ok_full:
                    out.append(
                        {
                            "directory_id": directory_id,
                            "frame_reference": fr,
                            "outcome": "skipped_complete",
                            "returncode": None,
                            "elapsed_sec": 0.0,
                            "resume_reason": reason_full,
                        }
                    )
                    continue
                ok_meas, reason_meas = verify_participant_complete_for_resume(
                    st,
                    run_id=run_id,
                    frame_reference=fr,
                    directory_id=directory_id,
                    project_root=project_root,
                    expect_surface=False,
                    expect_segment_ids=None,
                )
                if ok_meas:
                    surface_only = True
                    resume_recheck = "measurement_complete_surface_only"
                else:
                    resume_recheck = reason_meas
        cmd = [sys.executable, str(runner), "--directory-id", directory_id, "--run-id", run_id, "--frame-reference", fr]
        if args.plots:
            cmd.append("--enable-plots")
        else:
            cmd.append("--no-plots")
        if args.quiet:
            cmd.append("--quiet")
        if surface_only:
            cmd.append("--surface-only")
        elif measure_only:
            cmd.append("--measure-only")
        cmd.extend(["--surface-tag", str(args.surface_tag)])
        if args.surface_config_json is not None:
            cmd.extend(["--surface-config-json", str(args.surface_config_json)])
        t0 = time.time()
        log_path = log_dir / f"{directory_id}__{fr}.log"
        rec: Dict[str, Any] = {"directory_id": directory_id, "frame_reference": fr, "log": str(log_path)}
        if resume_recheck is not None:
            rec["resume_recheck_reason"] = resume_recheck
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(cmd, cwd=str(project_root), stdout=logf, stderr=subprocess.STDOUT, text=True, timeout=args.timeout_sec, env=_child_env())
            rc = proc.returncode
            rec.update(returncode=rc, outcome={0: "complete", 2: "partial"}.get(rc, "failed"))
        except subprocess.TimeoutExpired:
            rec.update(returncode=None, outcome="timeout")
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            rec.update(returncode=None, outcome="failed", reason=str(exc))
        rec["elapsed_sec"] = float(time.time() - t0)
        out.append(rec)
        if rec["outcome"] in {"failed", "timeout"}:
                                                     
            continue
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.execute and args.dry_run:
        print("Error: --execute and --dry-run are mutually exclusive", file=sys.stderr)
        return 1
    do_execute = bool(args.execute)
    project_root = project_root_from_here(__file__)
    runner = _HERE / "01_run_sin_trend_fitting.py"
    frames = [validate_frame_reference(f) for f in (args.frame_reference or [FRAME_BASELINE])]
    if args.surface_only and not args.run_id:
        print("Error: --surface-only requires the existing --run-id", file=sys.stderr)
        return 1
    run_id = validate_run_id(args.run_id) if args.run_id else new_run_id("run")
    n_workers = max(1, int(args.n_workers))

                                                          
    cfg = default_measurement_config()
    cfg_for_hash = {k: v for k, v in cfg.items() if k not in {"verbose", "enable_plots"}}
    measure_cfg, surface_cfg = split_measure_surface_config(cfg_for_hash)
    tag = validate_surface_tag(args.surface_tag)
    overrides = {}
    if args.surface_config_json is not None:
        import json
        # Child processes run from project_root, so keep caller-relative paths valid.
        args.surface_config_json = Path(args.surface_config_json).resolve()
        overrides = json.loads(Path(args.surface_config_json).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            print("Error: --surface-config-json must be a JSON object", file=sys.stderr)
            return 1
    # Surface-only needs saved measurements, not the original VTP/participant list.
    if args.surface_only:
        surface_cfg = surface_config_for_run(run_id, overrides, surface_tag=tag, project_root=project_root)
        if args.directory_id:
            all_ids = sorted(set(args.directory_id))
        else:
            suffix = "_vessel_segmentdata.csv"
            all_ids = sorted({
                p.name[:-len(suffix)]
                for fr in frames for p in per_case_dir(run_id, fr, project_root).glob("*" + suffix)
            })
    else:
        participants = participants_csv_path(project_root)
        df = pd.read_csv(participants)
        all_ids = sorted(df["directory_id"].dropna().astype(str).unique().tolist())
    if args.directory_id:
        want = set(args.directory_id)
        directory_ids = [d for d in all_ids if d in want]
        unknown = sorted(want - set(all_ids))
    else:
        if not args.all and args.limit is None:
            print("Error: full cohort requires --all; or use --directory-id / --limit.", file=sys.stderr)
            return 1
        directory_ids, unknown = all_ids, []
    if args.start_directory_id is not None:
        if args.start_directory_id not in directory_ids:
            print(f"Error: start-directory-id not in target list: {args.start_directory_id}")
            return 1
        directory_ids = directory_ids[directory_ids.index(args.start_directory_id) :]
    if args.limit is not None:
        directory_ids = directory_ids[: int(args.limit)]

    expected_exclusions: Dict[str, str] = {}
    for d in unknown:
        expected_exclusions[d] = "not in participants csv"
    runnable = []
    for d in directory_ids:
        if args.surface_only and not all(per_case_csv_path(run_id, fr, d, project_root).is_file() for fr in frames):
            expected_exclusions[d] = "missing saved measurement CSV"
        elif not args.surface_only and not centerline_vtp_path(d, project_root).is_file():
            expected_exclusions[d] = "missing centerline VTP"
        else:
            runnable.append(d)

    out_dir = run_root(run_id, project_root)
    if args.surface_only:
        out_dir = out_dir / "surface_batches" / tag
    log_dir = out_dir / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    mode = "EXECUTE" if do_execute else "DRY-RUN"
    print(f"[{mode}] run_id={run_id} tag={OUTPUT_TAG} frames={frames} targets={len(runnable)} excluded={len(expected_exclusions)} n_workers={n_workers}")
    if not do_execute:
        plan = {
            "run_id": run_id,
            "mode": "dry-run",
            "frames": frames,
            "n_targets": len(runnable),
            "directory_ids": runnable,
            "expected_exclusions": expected_exclusions,
            "child_runner": str(runner),
        }
        atomic_write_json(plan, out_dir / "batch_dry_run_plan.json")
        print(f"dry-run plan written: {out_dir / 'batch_dry_run_plan.json'}")
        return 0

    # Config writes occur only for execution, not when displaying a dry-run plan.
    if not args.surface_only:
        write_run_config(run_id, measure_cfg, project_root=project_root)
    if not (args.measure_only or args.no_surface):
        surface_cfg = surface_config_for_run(run_id, overrides, surface_tag=tag, project_root=project_root)
        write_surface_config(run_id, surface_cfg, surface_tag=tag, project_root=project_root)

    records: List[Dict[str, Any]] = []
    t_batch = time.time()
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {
                ex.submit(
                    run_one_participant,
                    runner=runner,
                    project_root=project_root,
                    directory_id=d,
                    run_id=run_id,
                    frames=frames,
                    args=args,
                    log_dir=log_dir,
                ): d
                for d in runnable
            }
            done = 0
            for fut in as_completed(futs):
                d = futs[fut]
                done += 1
                try:
                    recs = fut.result()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    recs = [{"directory_id": d, "frame_reference": fr, "outcome": "failed", "reason": str(exc)} for fr in frames]
                records.extend(recs)
                summary = ", ".join(f"{r['frame_reference']}={r['outcome']}({r.get('elapsed_sec', 0):.0f}s)" for r in recs)
                print(f"({done}/{len(runnable)}) {d}: {summary}", flush=True)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: waiting for running children to stop; partial batch tables will be written.", file=sys.stderr)
        raise
    finally:
        rec_df = pd.DataFrame(records)
        for name, outcomes in (
            ("batch_completed", {"complete", "skipped_complete"}),
            ("batch_partial", {"partial"}),
            ("batch_failed", {"failed"}),
            ("batch_timeout", {"timeout"}),
        ):
            sub = rec_df[rec_df["outcome"].isin(outcomes)] if len(rec_df) else pd.DataFrame(columns=["directory_id", "frame_reference", "outcome"])
            atomic_to_csv(sub, out_dir / f"{name}.csv", index=False)
        atomic_to_csv(
            pd.DataFrame([{"directory_id": k, "reason": v} for k, v in expected_exclusions.items()], columns=["directory_id", "reason"]),
            out_dir / "batch_expected_exclusions.csv",
            index=False,
        )
        manifest = build_run_manifest(run_id, directory_ids + unknown, frames, project_root, expected_exclusions=expected_exclusions)
        manifest["batch"] = {
            "n_workers": n_workers,
            "elapsed_sec": float(time.time() - t_batch),
            "n_targets": len(runnable),
            "outcome_counts": rec_df["outcome"].value_counts().to_dict() if len(rec_df) else {},
            "argv": sys.argv[1:],
        }
        if args.surface_only:
            # Preserve the measurement batch manifest.
            manifest["kind"] = "surface_batch"
            manifest["surface_tag"] = tag
            for fr in frames:
                participants = {}
                for did in sorted(set(directory_ids + unknown)):
                    if did in expected_exclusions:
                        participants[did] = {"status": "expected_exclusion", "reason": expected_exclusions[did]}
                    else:
                        try:
                            participants[did] = read_surface_participant_status(run_id, fr, did, tag, project_root) or {"status": "missing"}
                        except Exception as exc:
                            participants[did] = {"status": "status_unreadable", "reason": str(exc)}
                surface_counts = {}
                for st in participants.values():
                    key = str(st.get("status", "failed"))
                    surface_counts[key] = surface_counts.get(key, 0) + 1
                manifest["frames"][fr] = {"participants": participants, "counts": surface_counts}
            atomic_write_json(manifest, out_dir / "surface_batch_manifest.json")
        else:
            write_run_manifest(manifest, run_id, project_root)

    counts = rec_df["outcome"].value_counts().to_dict() if len(rec_df) else {}
    elapsed_sec = float(time.time() - t_batch)
    print(f"\nrun_id={run_id} outcomes={counts} elapsed={_format_elapsed(elapsed_sec)}")
    bad = sum(counts.get(k, 0) for k in ("failed", "timeout", "partial"))
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
