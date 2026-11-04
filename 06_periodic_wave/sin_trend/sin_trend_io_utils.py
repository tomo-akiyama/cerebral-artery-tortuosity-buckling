#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend io utils."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

OUTPUT_TAG = "sin_trend_11_patch1"
ANALYSIS_VERSION = "sin_trend_11"
ANALYSIS_REVISION = "patch1"
INTERVAL_KEY = ("directory_id", "segment_id", "segment_index")


def project_root_from_here(here: Optional[str] = None) -> Path:
    base = Path(here or __file__).resolve()
    # .../analysis_code/morphometrics/this.py -> project root
    return base.parents[2]


def morphometrics_root(project_root: Optional[Path] = None) -> Path:
    root = Path(project_root) if project_root is not None else project_root_from_here()
    return root / "results" / "morphometrics" / OUTPUT_TAG


def analysis_root(project_root: Optional[Path] = None) -> Path:
    root = Path(project_root) if project_root is not None else project_root_from_here()
    return root / "results" / "analysis" / OUTPUT_TAG


def figures_root(project_root: Optional[Path] = None) -> Path:
    root = Path(project_root) if project_root is not None else project_root_from_here()
    return root / "results" / "figures" / OUTPUT_TAG


def mca_work_root(project_root: Optional[Path] = None) -> Path:
    root = Path(project_root) if project_root is not None else project_root_from_here()
    return root / "results" / "analysis" / f"{OUTPUT_TAG}_mca_classification"


def per_case_dir(project_root: Optional[Path] = None) -> Path:
    return morphometrics_root(project_root) / "per_case"


def candidate_dir(project_root: Optional[Path] = None) -> Path:
    return morphometrics_root(project_root) / "candidates"


def long_csv_path(project_root: Optional[Path] = None) -> Path:
    return morphometrics_root(project_root) / "IXI_HH_sin_trend_morphometrics_long.csv"


def analysis_csv_path(project_root: Optional[Path] = None) -> Path:
    return analysis_root(project_root) / "IXI_HH_sin_trend_analysis_dataset.csv"


def metadata_dict(**extra: Any) -> Dict[str, Any]:
    d = {
        "output_tag": OUTPUT_TAG,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_revision": ANALYSIS_REVISION,
        "interval_key": list(INTERVAL_KEY),
    }
    d.update(extra)
    return d


def atomic_to_csv(df: pd.DataFrame, path: Path | str, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        df.to_csv(tmp_path, **kwargs)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def to_python_scalar(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    return x


def json_default(o: Any):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(to_python_scalar(x) for x in o)
    if isinstance(o, tuple):
        return [to_python_scalar(x) for x in o]
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def atomic_write_json(obj: Dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def normalize_id_columns(df: pd.DataFrame, keys: Sequence[str] = INTERVAL_KEY) -> pd.DataFrame:
    out = df.copy()
    for k in keys:
        if k not in out.columns:
            continue
        if k in ("segment_id", "segment_index"):
                                                  
            converted = pd.to_numeric(out[k], errors="coerce")
            mixed = out[k].copy()
            mask = converted.notna() & out[k].notna()
            mixed.loc[mask] = converted.loc[mask]
            out[k] = mixed
        else:
            out[k] = out[k].astype("string")
    return out


def assert_interval_keys_present(df: pd.DataFrame, keys: Sequence[str] = INTERVAL_KEY, where: str = "") -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"missing interval key columns {missing} {where}".strip())
    for k in keys:
        n_na = int(df[k].isna().sum())
        if n_na:
            raise ValueError(f"interval key {k!r} has {n_na} missing values {where}".strip())


def find_duplicate_keys(df: pd.DataFrame, keys: Sequence[str] = INTERVAL_KEY) -> pd.DataFrame:
    assert_interval_keys_present(df, keys)
    dup_mask = df.duplicated(subset=list(keys), keep=False)
    return df.loc[dup_mask, list(keys)].copy()


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: Sequence[str],
    how: str,
    validate: str,
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    assert_interval_keys_present(left, on, where="(left)")
    assert_interval_keys_present(right, on, where="(right)")
    n_left = len(left)
    out = left.merge(right, on=list(on), how=how, validate=validate, suffixes=suffixes)
    if how == "left" and len(out) != n_left:
        raise ValueError(
            f"merge altered row count: left={n_left}, result={len(out)}, validate={validate}"
        )
    return out


def same_file(a: Path | str, b: Path | str) -> bool:
    pa, pb = Path(a).resolve(), Path(b).resolve()
    try:
        return os.path.samefile(pa, pb)
    except OSError:
        return pa == pb


def compute_patch_radius_vox(
    radius_median_vox: float,
    *,
    patch_r_ratio: float = 3.0,
    r_min_mm: float = 2.5,
    r_max_mm: float = 5.0,
    voxel_size: float = 0.5,
) -> Dict[str, Any]:
    out = {
        "patch_r_ratio": float(patch_r_ratio),
        "r_patch_definition": "r_mm + r_mm*patch_r_ratio = (1+patch_r_ratio)*r_mm; clip(r_min_mm,r_max_mm)",
        "r_patch_mm_raw": float("nan"),
        "r_patch_mm": float("nan"),
        "r_patch_vox": float("nan"),
        "ok": False,
        "reason": "",
    }
    if radius_median_vox is None or not np.isfinite(radius_median_vox) or float(radius_median_vox) <= 0.0:
        out["reason"] = "radius_median non-positive or non-finite"
        return out
    r_mm = float(radius_median_vox) * float(voxel_size)
    raw = r_mm + r_mm * float(patch_r_ratio)
    clipped = float(np.clip(raw, float(r_min_mm), float(r_max_mm)))
    out["r_patch_mm_raw"] = float(raw)
    out["r_patch_mm"] = clipped
    out["r_patch_vox"] = clipped / float(voxel_size)
    out["ok"] = True
    return out
