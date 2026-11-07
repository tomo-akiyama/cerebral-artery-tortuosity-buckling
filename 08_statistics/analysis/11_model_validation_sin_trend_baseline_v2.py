#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 11 model validation sin trend baseline v2."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

_STATISTICS_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _STATISTICS_DIR.parent
_SIN_TREND_V2 = _PROJECT_ROOT / "06_periodic_wave"
_AC = _PROJECT_ROOT / "06_periodic_wave"
for _p in (str(_AC), str(_STATISTICS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sin_trend.sin_trend_paths import (  # noqa: E402
    FRAME_BASELINE,
    FRAME_REFERENCES,
    analysis_csv_path,
    analysis_root,
    validate_frame_reference,
    validate_run_id,
)

_VALIDATION = _STATISTICS_DIR / "11_model_validation.py"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Physical model validation (sin_trend formal dataset)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--frame-reference", default=FRAME_BASELINE, choices=list(FRAME_REFERENCES))
    p.add_argument("--analysis-csv", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="resolve the formal CSV and stop")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    root = _PROJECT_ROOT
    run_id = validate_run_id(args.run_id)
    frame = validate_frame_reference(args.frame_reference)
    csv_path = Path(args.analysis_csv) if args.analysis_csv else analysis_csv_path(run_id, frame, root)
    if not csv_path.is_file():
        print(f"Error: missing analysis CSV (run 33 first): {csv_path}")
        return 1
    print(f"formal analysis CSV: {csv_path}")
    if args.dry_run:
        print("dry-run: skipped 11_model_validation.py")
        return 0
    if not _VALIDATION.is_file():
        print(f"Error: missing {_VALIDATION}")
        return 1
    out_dir = analysis_root(run_id, frame, root) / "physical_validation"
    proc = subprocess.run(
        [
            sys.executable,
            str(_VALIDATION),
            "--run-id",
            run_id,
            "--frame-reference",
            frame,
            "--analysis-csv",
            str(csv_path),
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(_PROJECT_ROOT),
        check=False,
    )
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
