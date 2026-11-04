#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 13 age related visibility."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

STATISTICS_ROOT = Path(__file__).resolve().parent


def _load_visibility():
    path = STATISTICS_ROOT / "12_buckling_visibility.py"
    spec = importlib.util.spec_from_file_location("buckling_visibility", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _print_age(res, *, n_participants: int) -> None:
    beta = float(res.params["AGE"])
    ci_low, ci_high = res.conf_int().loc["AGE"]
    p_value = float(res.pvalues["AGE"])
    print(
        f"  AGE: beta={beta:.6f} CI=[{float(ci_low):.6f}, {float(ci_high):.6f}] "
        f"p={p_value:.5g} N={n_participants}"
    )


def _fit_and_print(mod, agg, *, model_id: str, y_col: str, include_vessel_length: bool) -> None:
    terms = mod.covariate_terms(agg, include_vessel_length=include_vessel_length)
    for label, cov_terms in (("univariable", []), ("multivariable", terms)):
        res, work = mod.fit_age_ols(agg, y_col, cov_terms)
        covariates = ", ".join(cov_terms) if cov_terms else "(none)"
        print(f"{model_id} {label}")
        print(f"  outcome: {y_col}")
        print(f"  Covariates: {covariates}")
        print(f"  formula: {res.model.formula}")
        _print_age(res, n_participants=len(work))
        if not np.isfinite(float(res.params["AGE"])):
            raise RuntimeError(f"{model_id} {label}: AGE coefficient unavailable")


def main() -> None:
    mod = _load_visibility()
    agg, csv_path = mod.load_participant_frame(mod.PROJECT_ROOT)
    print(f"input: {csv_path}")
    print(f"participants: {len(agg)}; intervals: {int(agg['n_total'].sum())}")
    _fit_and_print(
        mod,
        agg,
        model_id="a_total_vessel_length",
        y_col="total_vessel_length",
        include_vessel_length=False,
    )
    _fit_and_print(
        mod,
        agg,
        model_id="b_segment_count",
        y_col="n_total",
        include_vessel_length=True,
    )


if __name__ == "__main__":
    main()
