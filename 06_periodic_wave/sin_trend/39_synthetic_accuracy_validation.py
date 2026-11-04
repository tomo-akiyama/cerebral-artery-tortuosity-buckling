#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic accuracy validation — unified entry for parts (1)–(4).

(1) Accuracy across amplitudes, wavelengths, and phases
(2) Accuracy across background curvatures
(3) Accuracy across cycle counts
(4) Procedure-induced bias in wavelength–diameter scaling

See SYNTHETIC_ACCURACY_VALIDATION.md. Implementation remains in 36 / 42 / 40.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

PART_ALIASES = {
    "1": "1",
    "amplitudes-wavelengths-phases": "1",
    "awp": "1",
    "2": "2",
    "background-curvatures": "2",
    "curvature": "2",
    "3": "3",
    "cycle-counts": "3",
    "4": "4",
    "wavelength-diameter": "4",
    "scaling-bias": "4",
    "3+4": "3+4",
    "length-selection": "3+4",
    "all": "all",
}

PART_LABELS = {
    "1": "(1) Accuracy across amplitudes, wavelengths, and phases",
    "2": "(2) Accuracy across background curvatures",
    "3": "(3) Accuracy across cycle counts",
    "4": "(4) Procedure-induced bias in wavelength–diameter scaling",
    "3+4": "(3)(4) Accuracy across cycle counts + wavelength–diameter scaling bias",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--part",
        default="all",
        help=(
            "Which part(s): 1|2|3|4|3+4|all "
            "(aliases: amplitudes-wavelengths-phases, background-curvatures, "
            "cycle-counts, wavelength-diameter, length-selection)"
        ),
    )
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--n-replicates", type=int, default=None)
    p.add_argument("--n-per-replicate", type=int, default=None)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--clinical-csv", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def _normalize_part(raw: str) -> str:
    key = str(raw).strip()
    if key not in PART_ALIASES:
        raise SystemExit(
            f"unknown --part {raw!r}; use 1|2|3|4|3+4|all "
            f"(or aliases listed in --help)"
        )
    return PART_ALIASES[key]


def _run_part1(args: argparse.Namespace) -> int:
    mod = importlib.import_module("36_measurement_validation")
    argv: List[str] = ["--stress"]
    if args.run_id:
        argv.extend(["--run-id", args.run_id])
    if args.no_plots:
        argv.append("--no-plots")
    return int(mod.main(argv))


def _run_part2(args: argparse.Namespace) -> int:
    mod = importlib.import_module("42_curvature_sensitivity_grid")
    argv: List[str] = []
    if args.run_id:
        argv.extend(["--run-id", args.run_id])
    if args.quiet:
        argv.append("--quiet")
    return int(mod.main(argv))


def _run_length_selection(args: argparse.Namespace, suite: str) -> int:
    mod = importlib.import_module("40_length_selection_validation")
    argv: List[str] = ["--suite", suite, "--seed", str(args.seed), "--n-workers", str(args.n_workers)]
    if args.run_id:
        argv.extend(["--run-id", args.run_id])
    if args.output_dir is not None:
        argv.extend(["--output-dir", str(args.output_dir)])
    if args.clinical_csv is not None:
        argv.extend(["--clinical-csv", str(args.clinical_csv)])
    if args.config is not None:
        argv.extend(["--config", str(args.config)])
    if args.n_replicates is not None:
        argv.extend(["--n-replicates", str(args.n_replicates)])
    if args.n_per_replicate is not None:
        argv.extend(["--n-per-replicate", str(args.n_per_replicate)])
    if args.smoke:
        argv.append("--smoke")
    if args.dry_run:
        argv.append("--dry-run")
    if args.resume:
        argv.append("--resume")
    if args.no_plots:
        argv.append("--no-plots")
    if args.quiet:
        argv.append("--quiet")
    return int(mod.main(argv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    part = _normalize_part(args.part)
    rc = 0
    if part in ("1", "all"):
        if not args.quiet:
            print(f"=== {PART_LABELS['1']} ===", flush=True)
        rc = max(rc, _run_part1(args))
    if part in ("2", "all"):
        if not args.quiet:
            print(f"=== {PART_LABELS['2']} ===", flush=True)
        if part == "all" and args.run_id:
            args2 = argparse.Namespace(**vars(args))
            args2.run_id = f"{args.run_id}_curv"
            rc = max(rc, _run_part2(args2))
        else:
            rc = max(rc, _run_part2(args))
    if part in ("3", "3+4", "all"):
        suite = "A" if part == "3" else "all"
        label = PART_LABELS["3"] if part == "3" else PART_LABELS["3+4"]
        if not args.quiet:
            print(f"=== {label} ===", flush=True)
        rc = max(rc, _run_length_selection(args, suite=suite if part != "all" else "all"))
    if part == "4":
        if not args.quiet:
            print(f"=== {PART_LABELS['4']} ===", flush=True)
        rc = max(rc, _run_length_selection(args, suite="B"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
