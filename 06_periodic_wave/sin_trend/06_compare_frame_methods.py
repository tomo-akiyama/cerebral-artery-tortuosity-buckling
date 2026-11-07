#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 06 compare frame methods."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_paths import (  # noqa: E402
    COMPARISON_KEY,
    FRAME_BASELINE,
    FRAME_OBSERVED,
    INTERVAL_KEY,
    OUTPUT_TAG,
    analysis_csv_path,
    atomic_to_csv,
    atomic_write_json,
    comparison_root,
    metadata_dict,
    project_root_from_here,
    validate_run_id,
)
from sin_trend_io_utils import find_duplicate_keys, normalize_id_columns, safe_merge  # noqa: E402

COMPARE_METRICS = (
    "lambda_hat_mm",
    "amp_comb_mm",
    "amp_major_mm",
    "sigma1_mm",
    "sigma2_mm",
    "R2",
    "partial_R2",
    "R_out_surface",
    "squared_displacement_coverage",
    "coverage_hat",
    "L_base_mm",
    "slenderness",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Compare baseline vs observed ({OUTPUT_TAG})")
    p.add_argument("--run-id", required=True)
    p.add_argument("--baseline-csv", type=Path, default=None)
    p.add_argument("--observed-csv", type=Path, default=None)
    return p.parse_args(argv)


def _load(path: Path, frame: str) -> pd.DataFrame:
    df = normalize_id_columns(pd.read_csv(path, low_memory=False))
    dups = find_duplicate_keys(df, INTERVAL_KEY)
    if len(dups):
        raise ValueError(f"duplicate keys in {frame}: {len(dups)}")
    if "frame_reference" in df.columns:
        frames = sorted(df["frame_reference"].dropna().astype(str).unique().tolist())
        if frames and frames != [frame]:
            raise ValueError(f"{path} has unexpected frame_reference={frames}")
    return df


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root_from_here(__file__)
    run_id = validate_run_id(args.run_id)
    out_dir = comparison_root(run_id, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.baseline_csv) if args.baseline_csv else analysis_csv_path(run_id, FRAME_BASELINE, root)
    obs_path = Path(args.observed_csv) if args.observed_csv else analysis_csv_path(run_id, FRAME_OBSERVED, root)
    if not base_path.is_file() or not obs_path.is_file():
        print(f"Error: need both analysis CSVs\n  baseline={base_path}\n  observed={obs_path}")
        return 1

    base = _load(base_path, FRAME_BASELINE)
    obs = _load(obs_path, FRAME_OBSERVED)
    keys = list(INTERVAL_KEY)

    measurable = []
    for name, df in ((FRAME_BASELINE, base), (FRAME_OBSERVED, obs)):
        ok = df["fit_status"].astype(str) == "ok" if "fit_status" in df.columns else pd.Series(False, index=df.index)
        measurable.append({"frame": name, "n_rows": len(df), "n_fit_ok": int(ok.sum()), "n_core_all": int(df["core_all"].fillna(0).sum()) if "core_all" in df.columns else None})
    atomic_to_csv(pd.DataFrame(measurable), out_dir / "frame_measurable_counts.csv", index=False)

    metrics = [c for c in COMPARE_METRICS if c in base.columns and c in obs.columns]
    right_cols = keys + metrics + [c for c in ("core_all", "core_inplane", "fit_status", "periodicity", "surface_constraint") if c in obs.columns]
    joined = safe_merge(
        base,
        obs[right_cols],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_observed"),
    )
    # delta columns
    for m in metrics:
        bcol, ocol = f"{m}_baseline", f"{m}_observed"
        if bcol in joined.columns and ocol in joined.columns:
            joined[f"delta_{m}"] = pd.to_numeric(joined[ocol], errors="coerce") - pd.to_numeric(joined[bcol], errors="coerce")
    atomic_to_csv(joined, out_dir / "frame_pair_measurement_join.csv", index=False)

    # Core set comparison (independent selection)
    def _core_keys(df: pd.DataFrame) -> set:
        if "core_all" not in df.columns:
            return set()
        sub = df[df["core_all"].fillna(0).astype(bool)]
        return set(map(tuple, sub[keys].itertuples(index=False, name=None)))

    kb, ko = _core_keys(base), _core_keys(obs)
    set_rows = [
        {"set": "core_baseline", "n": len(kb)},
        {"set": "core_observed", "n": len(ko)},
        {"set": "intersection", "n": len(kb & ko)},
        {"set": "baseline_only", "n": len(kb - ko)},
        {"set": "observed_only", "n": len(ko - kb)},
    ]
    atomic_to_csv(pd.DataFrame(set_rows), out_dir / "frame_core_set_counts.csv", index=False)
    for name, idxs in (
        ("intersection", sorted(kb & ko)),
        ("baseline_only", sorted(kb - ko)),
        ("observed_only", sorted(ko - kb)),
    ):
        atomic_to_csv(pd.DataFrame(idxs, columns=keys), out_dir / f"frame_core_{name}_ids.csv", index=False)

    # numeric summary on paired measurable rows
    both_ok = joined
    if "fit_status_baseline" in joined.columns and "fit_status_observed" in joined.columns:
        both_ok = joined[(joined["fit_status_baseline"].astype(str) == "ok") & (joined["fit_status_observed"].astype(str) == "ok")]
    summary = []
    for m in metrics:
        dcol = f"delta_{m}"
        if dcol not in both_ok.columns:
            continue
        s = pd.to_numeric(both_ok[dcol], errors="coerce")
        finite = s[np.isfinite(s)]
        summary.append(
            {
                "metric": m,
                "n_paired_ok": int(len(both_ok)),
                "n_finite_delta": int(len(finite)),
                "delta_median": float(finite.median()) if len(finite) else np.nan,
                "delta_mean": float(finite.mean()) if len(finite) else np.nan,
                "delta_q1": float(finite.quantile(0.25)) if len(finite) else np.nan,
                "delta_q3": float(finite.quantile(0.75)) if len(finite) else np.nan,
            }
        )
    atomic_to_csv(pd.DataFrame(summary), out_dir / "frame_paired_delta_summary.csv", index=False)

    meta = metadata_dict(
        run_id=run_id,
        baseline_csv=str(base_path),
        observed_csv=str(obs_path),
        n_inner_join=int(len(joined)),
        n_paired_fit_ok=int(len(both_ok)),
        comparison_key=list(COMPARISON_KEY),
        note="Higher R2 is not treated as higher accuracy across frames.",
    )
    atomic_write_json(meta, out_dir / "frame_comparison_metadata.json")
    print(f"saved comparison under {out_dir} (inner={len(joined)}, paired_ok={len(both_ok)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
