#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load predominantly tangential intervals from the formal analysis CSV for figure scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_VALIDATION = Path(__file__).resolve().parent / "11_model_validation.py"
_RELEASE_CSV = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "periodic_wave"
    / "IXI_HH_sin_trend_analysis_dataset.csv"
)


def _validation_module():
    spec = importlib.util.spec_from_file_location("model_validation_11", _VALIDATION)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {_VALIDATION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_predominantly_tangential(project_root: Path) -> tuple[pd.DataFrame, Path]:
    """
    Return Core + predominantly tangential intervals from the formal analysis CSV.

    Columns are aligned for existing figure code:
      log_r, log_lambda, log_lambda_over_r, amp_hat_mm, lambda_hat_mm, radius_median_mm
    Clustering column: IXI_ID.
    """
    module = _validation_module()
    release = Path(project_root) / "data" / "periodic_wave" / "IXI_HH_sin_trend_analysis_dataset.csv"
    if release.is_file():
        csv_path = release
    elif _RELEASE_CSV.is_file():
        csv_path = _RELEASE_CSV
    else:
        csv_path = module.analysis_csv_path(
            module.DEFAULT_RUN_ID, module.FRAME_BASELINE, project_root
        )
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    raw = pd.read_csv(csv_path, low_memory=False)
    df = module.select_predominantly_tangential(module.prepare_analysis_frame(raw))
    df["log_r"] = df["log_radius"]
    df["amp_hat_mm"] = df["amplitude_mm"]
    df["lambda_hat_mm"] = df["lambda_mm"]
    df["radius_median_mm"] = df["radius_mm"]
    df["IXI_ID"] = df["IXI_ID"].astype(str)
    return df, csv_path
