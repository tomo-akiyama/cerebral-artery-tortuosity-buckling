#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 37 run clinical models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_paths import (  # noqa: E402
    FRAME_BASELINE,
    FRAME_REFERENCES,
    OUTPUT_TAG,
    analysis_csv_path,
    atomic_to_csv,
    atomic_write_json,
    metadata_dict,
    models_root,
    project_root_from_here,
    validate_frame_reference,
    validate_run_id,
)

REQUIRED_SPEC_KEYS = (
    "models",  # list of model blocks
)
REQUIRED_MODEL_KEYS = (
    "name",
    "outcome",
    "predictors",
    "filters",
    "cluster_id",
    "status",  # "confirmed" | "unresolved"
)


DEFAULT_TEMPLATE: Dict[str, Any] = {
    "description": "Fill from existing primary analysis scripts/notebooks. Do not invent covariates.",
    "unresolved_items": [
        "exact formula for log(lambda) ~ log(radius)",
        "exact formula for log(lambda/radius) ~ age",
        "amplitude~age specification (raw/log, covariates, lambda adjustment)",
        "MCA site inclusion rules",
        "missing-data handling",
        "participant-cluster SE / small-sample correction",
    ],
    "models": [
        {
            "name": "log_lambda_vs_log_radius",
            "outcome": "log_lambda_hat_mm",
            "predictors": ["log_radius_mm"],
            "filters": {"core_inplane": 1, "mca_segment_required": True, "exclude_label_conflicts": True},
            "cluster_id": "directory_id",
            "status": "unresolved",
            "notes": "Inherit exact adjustment from prior primary analysis when available.",
        },
        {
            "name": "log_lambda_over_radius_vs_age",
            "outcome": "log_lambda_over_radius",
            "predictors": ["age"],
            "filters": {"core_inplane": 1, "mca_segment_required": True},
            "cluster_id": "directory_id",
            "status": "unresolved",
        },
        {
            "name": "amplitude_vs_age",
            "outcome": "amp_comb_mm",
            "predictors": ["age"],
            "filters": {"core_inplane": 1, "mca_segment_required": True},
            "cluster_id": "directory_id",
            "status": "unresolved",
        },
    ],
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Clinical models connector ({OUTPUT_TAG})")
    p.add_argument("--run-id", required=True)
    p.add_argument("--frame-reference", default=FRAME_BASELINE, choices=list(FRAME_REFERENCES))
    p.add_argument("--analysis-csv", type=Path, default=None)
    p.add_argument("--spec", type=Path, default=None, help="model specification JSON")
    p.add_argument("--write-template", action="store_true", help="write template JSON and exit")
    p.add_argument("--allow-exploratory", action="store_true", help="run unadjusted exploratory OLS only")
    return p.parse_args(argv)


def _mca_ok(s: pd.Series) -> pd.Series:
    return s.notna() & ~s.astype(str).str.strip().str.lower().isin({"", "nan", "none", "null"})


def _prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "lambda_hat_mm" in out.columns:
        out["log_lambda_hat_mm"] = np.log(pd.to_numeric(out["lambda_hat_mm"], errors="coerce"))
    if "diameter_ref_mm" in out.columns:
        out["radius_mm"] = pd.to_numeric(out["diameter_ref_mm"], errors="coerce") / 2.0
        out["log_radius_mm"] = np.log(out["radius_mm"])
        if "lambda_hat_mm" in out.columns:
            out["log_lambda_over_radius"] = out["log_lambda_hat_mm"] - out["log_radius_mm"]
    return out


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    sub = df
    if filters.get("core_inplane") and "core_inplane" in sub.columns:
        sub = sub[sub["core_inplane"].fillna(0).astype(bool)]
    if filters.get("core_all") and "core_all" in sub.columns:
        sub = sub[sub["core_all"].fillna(0).astype(bool)]
    if filters.get("mca_segment_required") and "mca_segment" in sub.columns:
        sub = sub[_mca_ok(sub["mca_segment"])]
    if filters.get("exclude_label_conflicts") and "mca_label_conflict" in sub.columns:
        sub = sub[~sub["mca_label_conflict"].fillna(False).astype(bool)]
    return sub


def _exploratory_ols(df: pd.DataFrame, outcome: str, predictors: List[str], cluster_id: str) -> Dict[str, Any]:
    cols = [outcome] + predictors + ([cluster_id] if cluster_id in df.columns else [])
    use = df[cols].dropna()
    y = use[outcome].to_numpy(float)
    X = np.column_stack([np.ones(len(use))] + [use[p].to_numpy(float) for p in predictors])
    # ordinary least squares; SE are naive (not cluster-robust) — exploratory only
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = float(resid @ resid / max(n - k, 1))
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    names = ["intercept"] + predictors
    coefs = []
    for i, name in enumerate(names):
        coefs.append({"term": name, "coef": float(beta[i]), "se_naive": float(se[i]), "n_intervals": int(n), "n_participants": int(use[cluster_id].nunique()) if cluster_id in use.columns else None})
    return {"status": "exploratory_unadjusted", "coefficients": coefs, "note": "Not primary analysis; cluster-robust SE not applied."}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root_from_here(__file__)
    run_id = validate_run_id(args.run_id)
    frame = validate_frame_reference(args.frame_reference)
    out_dir = models_root(run_id, frame, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_path = out_dir / "model_spec_template.json"
    if args.write_template:
        atomic_write_json(DEFAULT_TEMPLATE, template_path)
        print(f"wrote template: {template_path}")
        print("schema/template only; clinical regression is a separate stage (not executed).")
        return 0
    if args.spec is None and not args.allow_exploratory:
        atomic_write_json(DEFAULT_TEMPLATE, template_path)
        print(f"wrote template: {template_path}")
        print(
            "status=regression_not_implemented_in_this_stage; "
            "use --write-template for schema, or --allow-exploratory for non-primary OLS only."
        )
        return 3

    spec_path = Path(args.spec) if args.spec else template_path
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    for k in REQUIRED_SPEC_KEYS:
        if k not in spec:
            print(f"Error: spec missing {k}")
            return 1

    csv_path = Path(args.analysis_csv) if args.analysis_csv else analysis_csv_path(run_id, frame, root)
    if not csv_path.is_file():
        print(f"Error: missing analysis CSV: {csv_path}")
        return 1
    df = _prepare_columns(pd.read_csv(csv_path, low_memory=False))

    results = []
    confirmed_any = False
    for model in spec.get("models", []):
        for k in REQUIRED_MODEL_KEYS:
            if k not in model:
                print(f"Error: model missing {k}: {model}")
                return 1
        status = str(model["status"])
        filtered = _apply_filters(df, model.get("filters") or {})
        atomic_to_csv(filtered, out_dir / f"analysis_subset_{model['name']}.csv", index=False)
        if status == "confirmed":
            confirmed_any = True
            print(f"Error: confirmed model runner is not fully wired for '{model['name']}' yet; provide inherited formula implementation.")
            return 2
        if status == "unresolved":
            if not args.allow_exploratory:
                results.append({"name": model["name"], "status": "skipped_unresolved"})
                continue
            missing = [c for c in [model["outcome"], *model["predictors"]] if c not in filtered.columns]
            if missing:
                results.append({"name": model["name"], "status": "missing_columns", "missing": missing})
                continue
            fit = _exploratory_ols(filtered, model["outcome"], list(model["predictors"]), model.get("cluster_id", "directory_id"))
            atomic_write_json(fit, out_dir / f"exploratory_{model['name']}.json")
            results.append({"name": model["name"], **{k: fit[k] for k in ("status", "note")}})
        else:
            results.append({"name": model["name"], "status": f"unknown_status:{status}"})

    meta = metadata_dict(
        run_id=run_id,
        frame_reference=frame,
        analysis_csv=str(csv_path),
        spec=str(spec_path),
        confirmed_any=confirmed_any,
        results=results,
        unresolved_items=spec.get("unresolved_items", DEFAULT_TEMPLATE["unresolved_items"]),
    )
    atomic_write_json(meta, out_dir / "clinical_models_metadata.json")
    print(f"saved under {out_dir}; exploratory={args.allow_exploratory}; results={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
