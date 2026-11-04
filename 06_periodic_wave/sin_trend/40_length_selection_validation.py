#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(3) Accuracy across cycle counts
(4) Procedure-induced bias in wavelength–diameter scaling

Unified entry: 39_synthetic_accuracy_validation.py --part 3|4|3+4
Docs: SYNTHETIC_ACCURACY_VALIDATION.md

Durable per-case results and strict resume.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)

from sin_trend_paths import (
    atomic_to_csv, atomic_write_json, metadata_dict, new_run_id, package_versions,
    project_root_from_here, sha256_json, source_code_hashes,
)
from sin_trend_length_selection_validation import (
    DEFAULT_CLINICAL_CSV_REL, DEFAULT_CONFIG_PATH, VALIDATION_SOURCE_FILES,
    aggregate_suite_A, aggregate_suite_B_across_replicates, aggregate_suite_B_replicate,
    apply_smoke_overrides, build_suite_B_replicate_jobs, check_resume_consistency,
    crosscheck_with_36_short_coverage, ensure_result_schema, iter_suite_A_jobs,
    length_selection_validation_root, load_clinical_DL_pool, load_config,
    measurement_config_from_validation_cfg, resume_fingerprint, run_suite_A_job,
    run_suite_A_jobs_batched, run_suite_B_job, save_summary_figures,
    save_example_waveforms, write_report,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="all", choices=["all", "A", "B", "smoke", "crosscheck36"])
    p.add_argument("--config", type=Path)
    p.add_argument("--seed", type=int, default=20261027)
    # None distinguishes omitted defaults from an explicitly requested 20 / 200 in smoke.
    p.add_argument("--n-replicates", type=int, default=None)
    p.add_argument("--n-per-replicate", type=int, default=None)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--run-id")
    p.add_argument("--clinical-csv", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Read-only planning; creates no files")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _worker_A(payload):
    job, meas_cfg, vcfg = payload
    return run_suite_A_job(job, meas_cfg, vcfg)


def _worker_B(payload):
    job, meas_cfg, vcfg = payload
    return run_suite_B_job(job, meas_cfg, vcfg)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class CaseJournal:
    """Append + fsync each completed case; recover only an incomplete final record."""

    def __init__(self, path: Path, planned_ids: set):
        self.path = Path(path)
        self.planned_ids = set(planned_ids)
        self.records: Dict[str, Dict[str, Any]] = {}
        self.recovered_tail_bytes = 0
        if self.path.exists():
            with self.path.open("rb+") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        self.recovered_tail_bytes = len(line)
                        f.truncate(offset)
                        f.flush()
                        os.fsync(f.fileno())
                        break
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise RuntimeError(f"Corrupt complete journal record at byte {offset}") from exc
                    self._validate(row)
                    self.records[str(row["case_id"])] = row
        self.file = self.path.open("ab")

    def _validate(self, row):
        cid = str(row.get("case_id", ""))
        if cid not in self.planned_ids:
            raise RuntimeError(f"Journal case is not in the current plan: {cid}")
        if cid in self.records:
            raise RuntimeError(f"Duplicate journal case_id: {cid}")

    def append(self, row):
        clean = _json_safe(row)
        self._validate(clean)
        data = (json.dumps(clean, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        self.file.write(data)
        self.file.flush()
        os.fsync(self.file.fileno())
        self.records[str(clean["case_id"])] = clean

    def close(self):
        self.file.close()


def _run_jobs(jobs, *, worker, meas_cfg, vcfg, n_workers, done_ids, quiet, on_result):
    pending = [j for j in jobs if j["case_id"] not in done_ids]
    if not pending:
        return
    if not quiet:
        print(f"pending={len(pending)} workers={n_workers}", flush=True)
    if n_workers == 1:
        for i, job in enumerate(pending, 1):
            on_result(worker((job, meas_cfg, vcfg)))
            if not quiet and (i % 25 == 0 or i == len(pending)):
                print(f"  saved {i}/{len(pending)}", flush=True)
        return
    # Keep only a bounded number of tasks in flight. An interrupt cannot leave
    # tens of thousands of queued tasks running before the executor shuts down.
    iterator = iter(pending)
    ex = ProcessPoolExecutor(max_workers=n_workers)
    futures = {}

    def submit_next():
        job = next(iterator, None)
        if job is not None:
            futures[ex.submit(worker, (job, meas_cfg, vcfg))] = job["case_id"]

    try:
        for _ in range(2 * n_workers):
            submit_next()
        count = 0
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in finished:
                expected = futures.pop(future)
                row = future.result()
                if row.get("case_id") != expected:
                    raise RuntimeError("Worker returned the wrong case_id")
                on_result(row)
                count += 1
                submit_next()
                if not quiet and (count % 25 == 0 or count == len(pending)):
                    print(f"  saved {count}/{len(pending)}", flush=True)
    finally:
        for future in futures:
            future.cancel()
        ex.shutdown(wait=True, cancel_futures=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root_from_here(__file__)
    t0 = time.monotonic()
    vcfg = load_config(args.config or DEFAULT_CONFIG_PATH)
    smoke = bool(args.smoke or args.suite == "smoke")
    if smoke:
        vcfg = apply_smoke_overrides(vcfg)
    suite = "all" if smoke else args.suite
    nr = args.n_replicates if args.n_replicates is not None else (2 if smoke else 20)
    npop = args.n_per_replicate if args.n_per_replicate is not None else (12 if smoke else 200)
    if min(nr, npop, args.n_workers) < 1:
        raise ValueError("n-replicates, n-per-replicate and n-workers must be positive")
    meas_cfg = measurement_config_from_validation_cfg(vcfg)
    if args.suite == "crosscheck36":
        cross = crosscheck_with_36_short_coverage(meas_cfg)
        print(json.dumps(cross, indent=2))
        return 0 if cross["ok"] else 2

    run_id = args.run_id or (args.output_dir.name if args.output_dir else new_run_id("lengthsel"))
    if not args.output_dir and (Path(run_id).name != run_id or run_id in (".", "..")):
        raise ValueError("run-id must be a directory name")
    out_dir = (args.output_dir or length_selection_validation_root(run_id, root)).resolve()
    need_A = suite in ("all", "A") and vcfg.get("suite_A", {}).get("enabled", True)
    need_B = suite in ("all", "B") and vcfg.get("suite_B", {}).get("enabled", True)
    pool = None
    pool_info = None
    if need_B:
        clinical = args.clinical_csv or Path(vcfg["suite_B"].get("clinical_csv") or DEFAULT_CLINICAL_CSV_REL)
        if not clinical.is_absolute():
            clinical = root / clinical
        pool = load_clinical_DL_pool(clinical, population_mode=vcfg["suite_B"].get("population_mode", "pre_selection"))
        pool_info = {
            "source_csv": pool.source_csv, "source_sha256": pool.source_sha256,
            "n_total": pool.n_total, "n_usable": pool.n_usable, "drop_reasons": pool.drop_reasons,
        }
    jobs_A = iter_suite_A_jobs(vcfg, seed=args.seed) if need_A else []
    jobs_B = []
    if need_B:
        for rep in range(nr):
            jobs_B.extend(build_suite_B_replicate_jobs(
                vcfg, seed=args.seed, replicate=rep, n_per_replicate=npop, pool=pool))
    jobs = jobs_A + jobs_B
    ids = {j["case_id"] for j in jobs}
    if not jobs or len(ids) != len(jobs):
        raise ValueError("Plan must contain at least one case and unique case_ids")
    run_spec = {
        "suite": suite, "smoke": smoke,
        "n_replicates": nr if need_B else 0, "n_per_replicate": npop if need_B else 0,
        "n_jobs_A": len(jobs_A), "n_jobs_B": len(jobs_B),
        "planned_jobs_sha256": sha256_json(jobs),
    }
    fp = resume_fingerprint(vcfg, seed=args.seed,
                            clinical_sha=pool.source_sha256 if pool else None, run_spec=run_spec)
    if args.resume:
        check_resume_consistency(out_dir, fp)
    if args.dry_run:
        print(json.dumps({
            "run_id": run_id, "output_dir": str(out_dir), **run_spec, "pool": pool_info,
            "read_only": True, "output_exists": out_dir.exists(),
            "would_refuse_nonresume_overwrite": out_dir.exists() and not args.resume,
        }, indent=2, ensure_ascii=False))
        return 0

    # Exclusive creation protects both existing results and accidentally reused run IDs.
    if not args.resume:
        out_dir.mkdir(parents=True, exist_ok=False)
    lock = (out_dir / ".run.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another process is using this validation output directory")

    journal = None
    try:
        if args.resume:
            check_resume_consistency(out_dir, fp)
        else:
            atomic_write_json(fp, out_dir / "run_fingerprint.json")
            atomic_write_json(vcfg, out_dir / "config_used.json")
        journal_path = out_dir / "case_results.jsonl"
        if args.resume and (out_dir / "all_cases.csv").exists() and not journal_path.exists():
            raise RuntimeError("Missing per-case journal; do not reconstruct a resume from CSV")
        journal = CaseJournal(journal_path, ids)
        done_ids = set(journal.records)
        atomic_write_json({
            "status": "running", "n_planned": len(jobs), "n_saved": len(done_ids),
            "recovered_tail_bytes": journal.recovered_tail_bytes,
        }, out_dir / "run_state.json")
        if not args.quiet:
            print(f"run_id={run_id} A={len(jobs_A)} B={len(jobs_B)} saved={len(done_ids)}", flush=True)

        if jobs_A:
            if args.n_workers == 1:
                run_suite_A_jobs_batched(jobs_A, meas_cfg, vcfg, done_ids=done_ids,
                                        quiet=args.quiet, on_result=journal.append)
            else:
                _run_jobs(jobs_A, worker=_worker_A, meas_cfg=meas_cfg, vcfg=vcfg,
                          n_workers=args.n_workers, done_ids=done_ids,
                          quiet=args.quiet, on_result=journal.append)
        _run_jobs(jobs_B, worker=_worker_B, meas_cfg=meas_cfg, vcfg=vcfg,
                  n_workers=args.n_workers, done_ids=done_ids,
                  quiet=args.quiet, on_result=journal.append)
        if set(journal.records) != ids:
            raise RuntimeError("Completed case set does not equal the planned case set")

        df = ensure_result_schema(pd.DataFrame(journal.records.values()))
        df = df.sort_values("case_id").reset_index(drop=True)
        atomic_to_csv(df, out_dir / "all_cases.csv", index=False)
        summary_A = aggregate_suite_A(df[df.suite.eq("A")])
        B = df[df.suite.eq("B")]
        summary_B_rep = aggregate_suite_B_replicate(B)
        summary_B_mc = aggregate_suite_B_across_replicates(summary_B_rep)
        atomic_to_csv(summary_A, out_dir / "summary_suite_A.csv", index=False)
        atomic_to_csv(summary_B_rep, out_dir / "summary_suite_B_by_replicate.csv", index=False)
        atomic_to_csv(summary_B_mc, out_dir / "summary_suite_B_monte_carlo.csv", index=False)
        if pool_info:
            atomic_write_json(pool_info, out_dir / "clinical_pool_info.json")
        figures = []
        if not args.no_plots:
            figures = save_summary_figures(df, out_dir / "figures")
            figures += save_example_waveforms(jobs, df, meas_cfg, vcfg, out_dir / "figures/examples")

        try:
            cross = crosscheck_with_36_short_coverage(meas_cfg)
        except Exception as exc:
            cross = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        atomic_write_json(cross, out_dir / "crosscheck_36_short_coverage.json")
        n_gen_fail = int(df.generation_status.ne("ok").sum())
        n_exec_fail = int((df.generation_status.eq("ok") & df.execution_status.ne("ok")).sum())
        status = "complete" if not (n_gen_fail or n_exec_fail) and cross["ok"] else "partial"
        elapsed = time.monotonic() - t0
        write_report(out_dir / "REPORT.md", vcfg=vcfg, summary_A=summary_A,
                     summary_B_rep=summary_B_rep, summary_B_mc=summary_B_mc,
                     pool_info=pool_info, n_rows=len(df), elapsed_s=elapsed)
        with (out_dir / "REPORT.md").open("a") as f:
            f.write(f"\n## Execution status\n\n- Status: {status}\n"
                    f"- Generation failures: {n_gen_fail}; measurement exceptions: {n_exec_fail}\n"
                    f"- Crosscheck with 36: {cross['ok']}\n"
                    "- Completed cases are durably recorded in case_results.jsonl.\n"
                    "- Rejected periodic fits are valid results, not execution failures.\n"
                    "- Cases with unknown achieved N_true are excluded from the achieved-N denominator "
                    "and counted separately; their target cycle count is not substituted for truth.\n")
        meta = metadata_dict(
            run_id=run_id, validation="length_selection", status=status, seed=args.seed,
            n_replicates=nr, n_per_replicate=npop, n_workers=args.n_workers, smoke=smoke,
            n_rows=len(df), n_jobs_A=len(jobs_A), n_jobs_B=len(jobs_B),
            n_generation_failed=n_gen_fail, n_execution_failed=n_exec_fail,
            clinical_pool=pool_info, measurement_config=meas_cfg,
            source_code_hashes=fp["code_hashes"], package_versions=fp["package_versions"],
            effective_config_sha256=fp["effective_config_sha256"], run_spec=run_spec,
            crosscheck_36=cross, figures=figures, elapsed_s=elapsed,
            limits=["No brain-surface / R_tan selection", "Clinical L_base is a proxy input distribution"],
        )
        atomic_write_json(meta, out_dir / "metadata.json")
        atomic_write_json({"status": status, "n_saved": len(df), "n_planned": len(jobs)},
                          out_dir / "run_state.json")
        if not args.quiet:
            print(f"{status}: saved {len(df)} cases; elapsed {elapsed:.1f}s; {out_dir}", flush=True)
        return 0 if status == "complete" else 2
    except BaseException as exc:
        # Preserve the original exception even if a full disk prevents a state update.
        try:
            atomic_write_json({
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "n_saved": len(journal.records) if journal else 0, "n_planned": len(jobs),
                "error": f"{type(exc).__name__}: {exc}",
            }, out_dir / "run_state.json")
        except Exception:
            pass
        if isinstance(exc, KeyboardInterrupt):
            print("Interrupted. Completed cases are saved; resume with identical generation settings.",
                  file=sys.stderr)
            return 130
        raise
    finally:
        if journal:
            journal.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
