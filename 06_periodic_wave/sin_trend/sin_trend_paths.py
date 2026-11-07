#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend paths."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
# Prefer this package over analysis_code/sin_trend (same basename modules must not collide).
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)

from sin_trend_io_utils import (  # noqa: E402,F401  (re-export)
    INTERVAL_KEY,
    assert_interval_keys_present,
    atomic_to_csv,
    compute_patch_radius_vox,
    find_duplicate_keys,
    normalize_id_columns,
    project_root_from_here,
    safe_merge,
    same_file,
)


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


# ---------------------------------------------------------------------------
                                         
# ---------------------------------------------------------------------------
OUTPUT_TAG = "sin_trend"
MEASUREMENT_VERSION = "sin_trend"
                                                                                           
MEASUREMENT_REVISION = "v2_forward3d_r2_baseline_phase_trend"
                                                                
                                                         
                                                        
                                                                                              
CLASSIFICATION_REVISION = "v2_partialR2_050_c4"
                                             
                                                       
SURFACE_METHOD_REVISION = "coherence_normals_v3_good_bridge_mm_evalsplit"

           
BAND_GEOMETRY_2D = "geometry_longwave_2D"
BAND_LEGACY_FIXED_10MM = "legacy_fixed_10mm"             
SEARCH_BAND_MODES = (BAND_GEOMETRY_2D, BAND_LEGACY_FIXED_10MM)

FRAME_BASELINE = "baseline"
FRAME_OBSERVED = "observed"
FRAME_OBSERVED_LEGACY = "observed_legacy"
FRAME_REFERENCES = (FRAME_BASELINE, FRAME_OBSERVED, FRAME_OBSERVED_LEGACY)
PRIMARY_FRAME_REFERENCES = (FRAME_BASELINE, FRAME_OBSERVED)

                          
COMPARISON_KEY = tuple(INTERVAL_KEY) + ("frame_reference",)

                                                              
CORE_PROFILE_V2 = "forward3d_partial_r2_050"
CORE_PROFILES = (CORE_PROFILE_V2,)
DEFAULT_CORE_PROFILE = CORE_PROFILE_V2
R2_THRESHOLDS = (0.60, 0.70, 0.80)                               
DEFAULT_R2_THRESHOLD = 0.60
PARTIAL_R2_THRESHOLD = 0.50
CORE_COVERAGE_MIN = 1.5
DIAMETER_WAVELENGTH_FACTOR = 2.0
BOUNDARY_POLICY_QC_ONLY = "record_2bins_qc_only"
BOUNDARY_POLICY_LEGACY_2BINS = "legacy_2bins_provisional"
BOUNDARY_GATE_ENABLED = True                                      
PARTIAL_R2_GATE_ENABLED = True
CLASSIFICATION_FINALIZED = False
CLASSIFICATION_RULE_FROZEN = True
CLASSIFICATION_VALIDATION_STATUS = "pending"
CORE_PROFILE_DISPLAY = "forward3d_partial_r2_050"
PARTIAL_R2_POLICY = "gate_ge_0.50"


def r2_tag(t: float) -> str:
    return f"{int(round(float(t) * 100)):03d}"

                                                                     
BASELINE_SOURCE_FILES = (
    "sin_trend_paths.py",
    "sin_trend_frames.py",
    "sin_trend_band.py",
    "sin_trend_measure.py",
    "sin_trend_classify.py",
    "sin_trend_surface.py",
    "sin_trend_synthetic.py",
    "sin_trend_forward_fit.py",
    "sin_trend_wavelength_coordinates.py",
    "sin_trend_fitting_helpers.py",
    "sin_trend_fitting.py",
    "sin_trend_io_utils.py",
    "sin_trend_dataset.py",
    "sin_trend_mca_labels.py",
    "sin_trend_mca_registry.py",
    "sin_trend_visualize.py",
    "01_run_sin_trend_fitting.py",
    "02_batch_sin_trend_fitting.py",
    "03_create_long_format_csv.py",
    "04_prepare_core_inplane_dataset.py",
)
# Stage manifests retain only the executable dependencies of their own stage.
# Shared modules remain in both sets: changing them cannot silently claim the
# same measurement implementation when measurements are actually rerun.
MEASUREMENT_SOURCE_FILES = tuple(
    name for name in BASELINE_SOURCE_FILES
    if name not in {
        "sin_trend_surface.py", "sin_trend_synthetic.py", "sin_trend_dataset.py",
        "sin_trend_mca_labels.py", "sin_trend_mca_registry.py", "sin_trend_visualize.py",
        "03_create_long_format_csv.py", "04_prepare_core_inplane_dataset.py",
    }
)
SURFACE_SOURCE_FILES = (
    "sin_trend_paths.py", "sin_trend_surface.py", "sin_trend_surface_stage.py",
    "sin_trend_fitting_helpers.py", "sin_trend_fitting.py", "sin_trend_classify.py",
    "sin_trend_io_utils.py",
)
                         
EXPECTED_FIT_METHOD = "forward3d_processed_template_joint_ols"
EXPECTED_OUTPUT_TAG = OUTPUT_TAG
EXPECTED_MEASUREMENT_VERSION = MEASUREMENT_VERSION
EXPECTED_MEASUREMENT_REVISION = MEASUREMENT_REVISION
EXPECTED_COORDINATE_UNIT_INPUT = "voxel"
EXPECTED_VOXEL_SIZE_MM = 0.5


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------
def _root(project_root: Optional[Path]) -> Path:
    return Path(project_root) if project_root is not None else project_root_from_here(__file__)


def validate_frame_reference(frame_reference: str) -> str:
    if frame_reference not in FRAME_REFERENCES:
        raise ValueError(f"unknown frame_reference {frame_reference!r}; choose from {FRAME_REFERENCES}")
    return frame_reference


def validate_run_id(run_id: str) -> str:
    if run_id is None:
        raise ValueError("run_id is required")
    s = str(run_id).strip()
    if not s or any(ch in s for ch in "/\\ \t\n"):
        raise ValueError(f"invalid run_id {run_id!r}: must be non-empty and contain no path separators or spaces")
    return s


def new_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def run_root(run_id: str, project_root: Optional[Path] = None) -> Path:
    return _root(project_root) / "results" / "morphometrics" / OUTPUT_TAG / validate_run_id(run_id)


def morphometrics_root(run_id: str, frame_reference: str, project_root: Optional[Path] = None) -> Path:
    return run_root(run_id, project_root) / validate_frame_reference(frame_reference)


def analysis_root(run_id: str, frame_reference: str, project_root: Optional[Path] = None) -> Path:
    return (
        _root(project_root)
        / "results"
        / "analysis"
        / OUTPUT_TAG
        / validate_run_id(run_id)
        / validate_frame_reference(frame_reference)
    )


def analysis_run_root(run_id: str, project_root: Optional[Path] = None) -> Path:
    return _root(project_root) / "results" / "analysis" / OUTPUT_TAG / validate_run_id(run_id)


def figures_root(run_id: str, frame_reference: str, project_root: Optional[Path] = None) -> Path:
    return morphometrics_root(run_id, frame_reference, project_root) / "figures"


def mca_work_root(run_id: str, project_root: Optional[Path] = None) -> Path:
    return analysis_run_root(run_id, project_root) / "mca_classification"


def mca_manual_labels_csv(run_id: str, project_root: Optional[Path] = None) -> Path:
    return mca_work_root(run_id, project_root) / "mca_manual_labels.csv"


def mca_work_frame_root(run_id: str, frame_reference: str, project_root: Optional[Path] = None) -> Path:
    return mca_work_root(run_id, project_root) / validate_frame_reference(frame_reference)


def mca_work_csv_path(run_id: str, frame_reference: str, project_root: Optional[Path] = None) -> Path:
    return mca_work_frame_root(run_id, frame_reference, project_root) / "IXI_HH_sin_trend_work.csv"


def validation_root(run_id: str, project_root: Optional[Path] = None) -> Path:
    return analysis_run_root(run_id, project_root) / "synthetic_validation"


def comparison_root(run_id: str, project_root: Optional[Path] = None) -> Path:
    return analysis_run_root(run_id, project_root) / "frame_comparison"


def models_root(run_id: str, frame_reference: str, project_root: Optional[Path] = None) -> Path:
    return analysis_root(run_id, frame_reference, project_root) / "clinical_models"


def per_case_dir(run_id: str, frame_reference: str, project_root: Optional[Path] = None) -> Path:
    return morphometrics_root(run_id, frame_reference, project_root) / "per_case"


def candidate_dir(run_id: str, frame_reference: str, project_root: Optional[Path] = None) -> Path:
                                                  
    return morphometrics_root(run_id, frame_reference, project_root) / "candidates"


def pointwise_dir(run_id: str, frame_reference: str, project_root: Optional[Path] = None) -> Path:
    return morphometrics_root(run_id, frame_reference, project_root) / "pointwise"


def normals_cache_dir(run_id: str, project_root: Optional[Path] = None) -> Path:
                                                   
    return run_root(run_id, project_root) / "normals_cache"


def long_csv_path(run_id: str, frame_reference: str, project_root: Optional[Path] = None) -> Path:
    return morphometrics_root(run_id, frame_reference, project_root) / "IXI_HH_sin_trend_morphometrics_long.csv"


def analysis_csv_path(run_id: str, frame_reference: str, project_root: Optional[Path] = None) -> Path:
    return analysis_root(run_id, frame_reference, project_root) / "IXI_HH_sin_trend_analysis_dataset.csv"


def run_manifest_path(run_id: str, project_root: Optional[Path] = None) -> Path:
    return run_root(run_id, project_root) / "run_manifest.json"


def run_config_path(run_id: str, project_root: Optional[Path] = None) -> Path:
    return run_root(run_id, project_root) / "run_config.json"


                                                                     
SURFACE_CONFIG_KEYS: frozenset = frozenset(
    {
        "coh_thr",
        "patch_r_ratio",
        "r_min_mm",
        "r_max_mm",
        "stride",
        "min_neighbors",
        "gap_fill_size",
        "bridge_max_gap_mm",
        "bridge_axis_max_deg",
        "cortex_coverage_thresh",
        "constraint_coverage_thresh",
        "squared_displacement_coverage_thresh",
        "R_out_thresh",
        "R_tan_thresh",
        "voxel_size",                              
    }
)


def validate_surface_tag(tag: str) -> str:
    s = str(tag or "").strip()
    if not s or s in {".", ".."}:
        raise ValueError("surface_tag must be non-empty")
                    
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(c not in allowed for c in s):
        raise ValueError(f"surface_tag has invalid characters: {tag!r}")
    return s


def surface_config_path(
    run_id: str, surface_tag: str = "primary", project_root: Optional[Path] = None
) -> Path:
    tag = validate_surface_tag(surface_tag)
    return run_root(run_id, project_root) / f"surface_config_{tag}.json"


def surface_per_case_dir(
    run_id: str, frame_reference: str, surface_tag: str = "primary", project_root: Optional[Path] = None
) -> Path:
    return morphometrics_root(run_id, frame_reference, project_root) / "surface" / validate_surface_tag(surface_tag) / "per_case"


def surface_per_case_csv_path(
    run_id: str, frame_reference: str, directory_id: str,
    surface_tag: str = "primary", project_root: Optional[Path] = None,
) -> Path:
    return surface_per_case_dir(run_id, frame_reference, surface_tag, project_root) / f"{directory_id}_vessel_segmentdata.csv"


def surface_long_csv_path(
    run_id: str, frame_reference: str, surface_tag: str = "primary", project_root: Optional[Path] = None
) -> Path:
    return surface_per_case_dir(run_id, frame_reference, surface_tag, project_root).parent / "IXI_HH_sin_trend_morphometrics_long.csv"


def surface_participant_status_path(
    run_id: str, frame_reference: str, directory_id: str,
    surface_tag: str = "primary", project_root: Optional[Path] = None,
) -> Path:
    return surface_per_case_dir(run_id, frame_reference, surface_tag, project_root) / directory_id / f"{directory_id}_surface_status.json"


def read_surface_participant_status(
    run_id: str, frame_reference: str, directory_id: str,
    surface_tag: str = "primary", project_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    path = surface_participant_status_path(run_id, frame_reference, directory_id, surface_tag, project_root)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        raise ValueError(f"surface participant status is not a dict: {path}")
    return obj


def split_measure_surface_config(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    measure: Dict[str, Any] = {}
    surface: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k in SURFACE_CONFIG_KEYS:
            surface[k] = v
        else:
            measure[k] = v
                                            
    if "voxel_size" in cfg:
        measure["voxel_size"] = cfg["voxel_size"]
        surface["voxel_size"] = cfg["voxel_size"]
    return measure, surface


def _surface_threshold_complement(value: Any) -> float:
    """Preserve decimal threshold pairs, e.g. user 0.70 corresponds to 0.30."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("surface direction threshold must be numeric") from exc
    return float(Decimal("1") - Decimal(str(number)))


def resolve_surface_config(
    run_id: str, surface_cfg: Dict[str, Any], project_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Validate surface-only overrides against the immutable measurement coordinates."""
    if not isinstance(surface_cfg, dict):
        raise ValueError("surface_cfg must be a dict")
    unknown = sorted(set(surface_cfg) - SURFACE_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"non-surface configuration keys cannot be overridden: {unknown}")
    record = read_run_config(run_id, project_root)
    checked = validate_run_config_contents(record, require=True)
    if not checked["ok"]:
        raise ValueError(f"invalid measurement run_config: {checked['reasons']}")
    saved_cfg = record["config"]
    if record.get("config_hash") != sha256_json(saved_cfg):
        raise ValueError("measurement run_config config_hash does not match its contents")
    expected_vs = float(checked["voxel_size"])
    result = dict(surface_cfg)
    if "voxel_size" in result:
        try:
            got_vs = float(result["voxel_size"])
        except (ValueError, TypeError) as exc:
            raise ValueError("surface voxel_size must match the measurement voxel_size") from exc
        if not np.isfinite(got_vs) or got_vs != expected_vs:
            raise ValueError(
                f"surface voxel_size={got_vs!r} differs from immutable measurement voxel_size={expected_vs!r}"
            )
    result["voxel_size"] = expected_vs
    unit_interval_keys = {
        "coh_thr", "constraint_coverage_thresh", "squared_displacement_coverage_thresh",
        "cortex_coverage_thresh", "R_out_thresh", "R_tan_thresh",
    }
    numeric_keys = unit_interval_keys | {
        "patch_r_ratio", "r_min_mm", "r_max_mm", "bridge_max_gap_mm",
        "bridge_axis_max_deg", "stride", "min_neighbors", "gap_fill_size",
    }
    for key in numeric_keys & set(result):
        try:
            value = float(result[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a finite number") from exc
        if not np.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if key in unit_interval_keys and not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be in [0, 1]")
        if key == "bridge_axis_max_deg" and not 0.0 <= value <= 90.0:
            raise ValueError("bridge_axis_max_deg must be in [0, 90]")
        if key in {"patch_r_ratio", "bridge_max_gap_mm"} and value < 0.0:
            raise ValueError(f"{key} must be nonnegative")
        if key in {"r_min_mm", "r_max_mm"} and value <= 0.0:
            raise ValueError(f"{key} must be positive")
        if key in {"stride", "min_neighbors", "gap_fill_size"}:
            minimum = 0 if key == "gap_fill_size" else 1
            if value != int(value) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
            result[key] = int(value)
        else:
            result[key] = value
    if "r_min_mm" in result and "r_max_mm" in result and result["r_min_mm"] > result["r_max_mm"]:
        raise ValueError("r_min_mm must not exceed r_max_mm")
    if "R_out_thresh" in result:
        derived = _surface_threshold_complement(result["R_out_thresh"])
        if "R_tan_thresh" in result and not np.isclose(float(result["R_tan_thresh"]), derived, atol=1e-12, rtol=0):
            raise ValueError("R_tan_thresh must equal 1 - R_out_thresh")
        result["R_tan_thresh"] = derived
    elif "R_tan_thresh" in result:
        result["R_out_thresh"] = _surface_threshold_complement(result["R_tan_thresh"])
    return result


def surface_config_for_run(
    run_id: str, overrides: Optional[Dict[str, Any]] = None, *,
    surface_tag: str = "primary", project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build surface settings without replacing the saved measurement coordinate scale."""
    from sin_trend_measure import default_measurement_config

    _measure, settings = split_measure_surface_config(default_measurement_config())
    path = surface_config_path(run_id, surface_tag, project_root)
    if path.is_file():
        settings.update(read_surface_config(run_id, surface_tag, project_root)["config"])
    measurement = read_run_config(run_id, project_root)
    checked = validate_run_config_contents(measurement, require=True)
    if not checked["ok"]:
        raise ValueError(f"invalid measurement run_config: {checked['reasons']}")
    settings["voxel_size"] = checked["voxel_size"]
    requested = dict(overrides or {})
    settings.update(requested)
    if "R_out_thresh" in requested and "R_tan_thresh" not in requested:
        settings["R_tan_thresh"] = _surface_threshold_complement(requested["R_out_thresh"])
    elif "R_tan_thresh" in requested and "R_out_thresh" not in requested:
        settings["R_out_thresh"] = _surface_threshold_complement(requested["R_tan_thresh"])
    return resolve_surface_config(run_id, settings, project_root)


def write_surface_config(
    run_id: str,
    surface_cfg: Dict[str, Any],
    *,
    surface_tag: str = "primary",
    project_root: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    tag = validate_surface_tag(surface_tag)
    path = surface_config_path(run_id, tag, project_root)
    _ = overwrite
    slim = resolve_surface_config(run_id, surface_cfg, project_root)
    cfg_hash = sha256_json(slim)
    measurement_record = read_run_config(run_id, project_root)
    code_hashes = source_code_hashes(SURFACE_SOURCE_FILES)
    record = metadata_dict(
        run_id=validate_run_id(run_id),
        surface_tag=tag,
        config=slim,
        config_hash=cfg_hash,
        measurement_config_hash=measurement_record["config_hash"],
        surface_source_code_hashes=code_hashes,
        surface_source_code_hash_combined=sha256_json(code_hashes),
        surface_method_revision=SURFACE_METHOD_REVISION,
        kind="surface_config",
    )
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            old = json.load(f)
        if not isinstance(old, dict) or old.get("config_hash") != sha256_json(old.get("config")):
            raise ValueError(f"surface_config contents do not match their config_hash: {path}")
        identity_keys = ("config_hash", "measurement_config_hash", "surface_source_code_hash_combined")
        mismatches = [key for key in identity_keys if old.get(key) != record.get(key)]
        if mismatches:
            raise RuntimeError(
                f"surface_tag {tag!r} already exists with different {mismatches}; "
                "use a new surface_tag to keep existing results unambiguous."
            )
        return old
    atomic_write_json(record, path)
    return record


def read_surface_config(
    run_id: str, surface_tag: str = "primary", project_root: Optional[Path] = None
) -> Dict[str, Any]:
    path = surface_config_path(run_id, surface_tag, project_root)
    if not path.is_file():
        raise FileNotFoundError(f"surface_config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"surface_config is not a dict: {type(obj).__name__}")
    if obj.get("run_id") != validate_run_id(run_id) or obj.get("surface_tag") != validate_surface_tag(surface_tag):
        raise ValueError(f"surface_config identity mismatch: {path}")
    if not isinstance(obj.get("config"), dict) or obj.get("config_hash") != sha256_json(obj["config"]):
        raise ValueError(f"surface_config contents do not match their config_hash: {path}")
    measurement = read_run_config(run_id, project_root)
    if obj.get("measurement_config_hash") != measurement.get("config_hash"):
        raise ValueError(f"surface_config measurement_config_hash mismatch: {path}")
    return obj


def participant_status_path(
    run_id: str, frame_reference: str, directory_id: str, project_root: Optional[Path] = None
) -> Path:
    return per_case_dir(run_id, frame_reference, project_root) / directory_id / f"{directory_id}_run_status.json"


def per_case_csv_path(run_id: str, frame_reference: str, directory_id: str, project_root: Optional[Path] = None) -> Path:
    return per_case_dir(run_id, frame_reference, project_root) / f"{directory_id}_vessel_segmentdata.csv"


def centerline_vtp_path(directory_id: str, project_root: Optional[Path] = None) -> Path:
    return (
        _root(project_root)
        / "data"
        / "derived"
        / "vessel_centerlines"
        / directory_id
        / f"{directory_id}_merged_centerlines.vtp"
    )


def brain_mesh_path(directory_id: str, side: str, project_root: Optional[Path] = None) -> Path:
    tag = "rt" if str(side).lower().startswith("r") else "lt"
    return (
        _root(project_root)
        / "data"
        / "preprocessed"
        / "T1_preprocessed"
        / "brain_meshes"
        / directory_id
        / f"{directory_id}_{tag}_pial_remesh.ply"
    )


def participants_csv_path(project_root: Optional[Path] = None) -> Path:
    return _root(project_root) / "data" / "participants" / "IXI_HH_participants.csv"


def legacy_v1_mca_label_csv(project_root: Optional[Path] = None) -> Path:
    return legacy_v1_analysis_csv(project_root)


def legacy_v1_analysis_csv(project_root: Optional[Path] = None) -> Path:
    return _root(project_root) / "results" / "analysis" / "sin_trend" / "IXI_HH_sin_trend_analysis_dataset.csv"


def legacy_mca_classification_csv(project_root: Optional[Path] = None) -> Path:
    return (
        _root(project_root)
        / "results"
        / "analysis"
        / "sin_trend_mca_classification"
        / "IXI_HH_sin_trend_analysis_dataset.csv"
    )


# ---------------------------------------------------------------------------
# hashes / metadata
# ---------------------------------------------------------------------------
def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _json_default(o: Any):
    try:
        return json_default(o)
    except TypeError:
        return str(o)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr, float))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def source_code_hashes(files: Sequence[str] = BASELINE_SOURCE_FILES, here: Optional[Path] = None) -> Dict[str, str]:
    here = Path(here) if here is not None else _HERE
    out: Dict[str, str] = {}
    for name in files:
        cand = here / name
        if cand.is_file():
            out[name] = sha256_file(cand)
        else:
            out[name] = "missing"
    return out


def package_versions() -> Dict[str, str]:
    out: Dict[str, str] = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "pandas", "pyvista", "vtk", "sklearn", "statsmodels", "matplotlib"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", None)
            if ver is None and mod == "vtk":
                ver = m.vtkVersion.GetVTKVersion()
            out[mod] = str(ver)
        except Exception:  # noqa: BLE001
            out[mod] = "not_available"
    return out


def metadata_dict(**extra: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "output_tag": OUTPUT_TAG,
        "measurement_version": MEASUREMENT_VERSION,
        "measurement_revision": MEASUREMENT_REVISION,
        "classification_revision": CLASSIFICATION_REVISION,
        "surface_method_revision": SURFACE_METHOD_REVISION,
        "interval_key": list(INTERVAL_KEY),
        "comparison_key": list(COMPARISON_KEY),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
                                                             
# ---------------------------------------------------------------------------
def write_run_config(run_id: str, cfg: Dict[str, Any], project_root: Optional[Path] = None) -> Dict[str, Any]:
    path = run_config_path(run_id, project_root)
    # The measurement contract excludes surface settings and surface-only code.
    # Surface-only reruns read this record; they never call this writer.
    cfg, _surface_cfg = split_measure_surface_config(cfg)
    code_hashes = source_code_hashes(MEASUREMENT_SOURCE_FILES)
    cfg_hash = sha256_json(cfg)
                                                         
    coord_keys = (
        "fitting_coordinate",
        "phase_coordinate",
        "trend_coordinate",
        "quadratic_detrend_coordinate",
        "grid_length_basis",
        "radius_basis",
        "slenderness_definition",
    )
    coord_meta = {k: cfg.get(k) for k in coord_keys if k in cfg and cfg.get(k) not in (None, "")}
    record = metadata_dict(
        run_id=validate_run_id(run_id),
        config=cfg,
        config_hash=cfg_hash,
        source_code_hashes=code_hashes,
        source_code_hash_combined=sha256_json(code_hashes),
        source_code_hash_scope="measurement_stage_v1",
        package_versions=package_versions(),
        **coord_meta,
    )
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            old = json.load(f)
        mism = []
        if old.get("config_hash") != cfg_hash:
            mism.append("config_hash")
        if old.get("source_code_hash_combined") != record["source_code_hash_combined"]:
            mism.append("source_code_hash")
        if old.get("measurement_version") != MEASUREMENT_VERSION or old.get("measurement_revision") != MEASUREMENT_REVISION:
            mism.append("measurement_version_or_revision")
        if mism:
            raise RuntimeError(
                f"run_id {run_id!r} already exists with different {mism}; "
                "resume requires identical config/code. Use a new run_id."
            )
        return old
    atomic_write_json(record, path)
    return record


def read_run_config(run_id: str, project_root: Optional[Path] = None) -> Dict[str, Any]:
    path = run_config_path(run_id, project_root)
    if not path.is_file():
        raise FileNotFoundError(f"run_config.json not found for run_id={run_id}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"run_config is not a dict: {type(obj).__name__}")
    return obj


def validate_run_config_contents(
    run_config: Any,
    *,
    require: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "reasons": [], "voxel_size": None}
    if run_config is None:
        if require:
            out["ok"] = False
            out["reasons"].append("run_config_missing")
        return out
    if not isinstance(run_config, dict) or len(run_config) == 0:
        out["ok"] = False
        out["reasons"].append("run_config_not_a_nonempty_dict")
        return out

                                                                                
    for key, expected in (
        ("measurement_version", EXPECTED_MEASUREMENT_VERSION),
        ("measurement_revision", EXPECTED_MEASUREMENT_REVISION),
        ("output_tag", EXPECTED_OUTPUT_TAG),
    ):
        got = run_config.get(key)
        if got is None or str(got).strip() == "":
            # older records may nest only under config; still require top-level for formal path
            out["ok"] = False
            out["reasons"].append(f"run_config_{key}_missing")
        elif str(got) != str(expected):
            out["ok"] = False
            out["reasons"].append(f"run_config_{key}_expected_{expected}_got_{got}")

    cfg = run_config.get("config")
    if not isinstance(cfg, dict):
        out["ok"] = False
        out["reasons"].append("run_config_config_missing_or_not_dict")
        return out
    if "voxel_size" not in cfg:
        out["ok"] = False
        out["reasons"].append("run_config_voxel_size_missing")
        return out
    try:
        vs = float(cfg["voxel_size"])
    except (TypeError, ValueError):
        out["ok"] = False
        out["reasons"].append(f"run_config_voxel_size_non_numeric:{cfg.get('voxel_size')!r}")
        return out
    if not np.isfinite(vs) or vs <= 0:
        out["ok"] = False
        out["reasons"].append(f"run_config_voxel_size_not_positive_finite:{vs}")
        return out
    out["voxel_size"] = float(vs)
    return out


def read_participant_status(
    run_id: str, frame_reference: str, directory_id: str, project_root: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    p = participant_status_path(run_id, frame_reference, directory_id, project_root)
    if not p.is_file():
        return None

    def _unreadable(kind: str, detail: Any) -> Dict[str, Any]:
        return {
            "status": "status_unreadable",
            "status_read_error": f"{kind}:{detail}",
            "directory_id": str(directory_id),
            "run_id": str(run_id),
            "frame_reference": str(frame_reference),
            "status_path": str(p),
        }

    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except UnicodeDecodeError as exc:
        return _unreadable("UnicodeDecodeError", exc)
    except json.JSONDecodeError as exc:
        return _unreadable("JSONDecodeError", exc)
    except OSError as exc:
        return _unreadable("OSError", exc)
    if not isinstance(obj, dict):
        return _unreadable("not_a_dict", type(obj).__name__)
    return obj


def _segment_filter_equal(stored: Any, expected: Optional[Sequence[int]]) -> bool:
    def _norm(v: Any) -> Optional[frozenset]:
        if v is None:
            return None
        if isinstance(v, (list, tuple, set, frozenset)):
            if len(v) == 0:
                return None
            return frozenset(int(x) for x in v)
        return frozenset([int(v)])

    return _norm(stored) == _norm(expected)


                                     
                                                          
SURFACE_INCOMPLETE_STATUSES: frozenset = frozenset(
    {
        "missing_brain_mesh",
        "surface_eval_failed",
        "skipped_exception",
        "invalid_radius",
        "missing_pointwise",
        "invalid_pointwise",
        "not_run",              
    }
)


def surface_incomplete_reason(
    surf_counts: Optional[Dict[str, Any]],
    *,
    surface_enabled: bool,
) -> Optional[str]:
    if not surface_enabled:
        return None
    counts = surf_counts or {}
    for bad in sorted(SURFACE_INCOMPLETE_STATUSES):
        if int(counts.get(bad, 0) or 0) > 0:
            return str(bad)
    return None


def participant_outcome_from_counts(
    *,
    n_exceptions: int = 0,
    n_save_failures: int = 0,
    n_ambiguous: int = 0,
    surface_enabled: bool = True,
    surface_status_counts: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    if int(n_exceptions) > 0:
        return "partial", "exceptions"
    if int(n_save_failures) > 0:
        return "partial", "save_failures"
    if int(n_ambiguous) > 0:
        return "partial", "ambiguous_groupid"
    bad = surface_incomplete_reason(surface_status_counts, surface_enabled=surface_enabled)
    if bad is not None:
        return "partial", f"surface_incomplete:{bad}"
    return "complete", "ok"


def verify_brain_mesh_inputs(
    status: Dict[str, Any],
    *,
    expect_surface: bool,
) -> Tuple[bool, str]:
    if not expect_surface:
        return True, "ok"
    if not bool(status.get("surface_enabled", False)):
        return True, "ok"
    if "input_brain_meshes" not in status:
                                                              
        return False, "missing_brain_mesh_sha_metadata"
    meshes = status.get("input_brain_meshes") or {}
    for side, info in meshes.items():
        if not isinstance(info, dict):
            return False, f"invalid_brain_mesh_manifest:{side}"
        path_s = info.get("path")
        stored = info.get("sha256")
        if not path_s or not stored:
            return False, f"incomplete_brain_mesh_manifest:{side}"
        path = Path(path_s)
        if not path.is_file():
            return False, f"missing_brain_mesh_file:{side}"
        try:
            cur = sha256_file(path)
        except OSError as exc:
            return False, f"brain_mesh_unreadable:{side}:{exc}"
        if str(cur) != str(stored):
            return False, f"brain_mesh_sha256_mismatch:{side}"
    return True, "ok"


def verify_required_artifacts(
    status: Dict[str, Any],
    *,
    csv_path: Optional[Path] = None,
) -> Tuple[bool, str]:
    arts = status.get("required_artifacts")
    if arts is None:
                                   
        if csv_path is None or not Path(csv_path).is_file():
            return False, "missing_required_artifacts_metadata"
        try:
            import pandas as pd

            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            return False, f"artifacts_csv_unreadable:{exc}"
        arts = []
        if "pointwise_save_status" in df.columns and "pointwise_npz_path" in df.columns:
            for _, r in df.iterrows():
                if str(r.get("pointwise_save_status", "")) == "ok":
                    p = str(r.get("pointwise_npz_path", "") or "").strip()
                    if p:
                        arts.append({"kind": "pointwise", "path": p})
        if "candidates_save_status" in df.columns and "candidate_csv_path" in df.columns:
            for _, r in df.iterrows():
                if str(r.get("candidates_save_status", "")) == "ok":
                    p = str(r.get("candidate_csv_path", "") or "").strip()
                    if p:
                        arts.append({"kind": "candidates", "path": p})
    for art in arts or []:
        if not isinstance(art, dict):
            continue
        if art.get("required") is False:
            continue
        p = Path(str(art.get("path", "") or ""))
        kind = art.get("kind", "artifact")
        if not p.is_file():
            return False, f"missing_artifact:{kind}:{p.name or str(p)}"
                                             
        stored_sha = art.get("sha256")
        if stored_sha:
            try:
                if str(sha256_file(p)) != str(stored_sha):
                    return False, f"artifact_sha256_mismatch:{kind}:{p.name}"
            except OSError as exc:
                return False, f"artifact_unreadable:{kind}:{p.name}:{exc}"
        try:
            if kind == "pointwise" or p.suffix == ".npz":
                with np.load(p, allow_pickle=False) as z:
                    names = list(z.files)
                    if len(names) == 0:
                        return False, f"empty_artifact:{kind}:{p.name}"
                    for key in names:
                                                  
                        _ = np.asarray(z[key])
            elif kind == "candidates" or p.suffix == ".csv":
                import pandas as pd

                pd.read_csv(p, nrows=1)
        except Exception as exc:  # noqa: BLE001
            return False, f"unreadable_artifact:{kind}:{p.name}:{exc}"
    return True, "ok"


def count_csv_data_rows(path: Path | str) -> int:
    import csv

    path = Path(path)
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def verify_participant_complete_for_resume(
    status: Optional[Dict[str, Any]],
    *,
    run_id: str,
    frame_reference: str,
    directory_id: str,
    project_root: Optional[Path] = None,
    expect_surface: bool = True,
    expect_segment_ids: Optional[Sequence[int]] = None,
) -> Tuple[bool, str]:
    root = _root(project_root)
    if status is None:
        return False, "missing_status"

                                                
    for key, expected in (
        ("run_id", run_id),
        ("directory_id", directory_id),
        ("frame_reference", frame_reference),
        ("measurement_version", EXPECTED_MEASUREMENT_VERSION),
        ("measurement_revision", EXPECTED_MEASUREMENT_REVISION),
        ("fit_method", EXPECTED_FIT_METHOD),
    ):
        got = status.get(key)
        if got is None or str(got).strip() == "" or str(got).lower() == "nan":
            return False, f"status_{key}_missing"
        if str(got) != str(expected):
            return False, f"status_{key}_mismatch:{got}"
    got_tag = status.get("output_tag")
    if got_tag is not None and str(got_tag).strip() != "" and str(got_tag) != str(EXPECTED_OUTPUT_TAG):
        return False, f"status_output_tag_mismatch:{got_tag}"

                                                         
    outcome, outcome_reason = participant_outcome_from_counts(
        n_exceptions=int(status.get("n_exceptions", 0) or 0),
        n_save_failures=int(status.get("n_save_failures", 0) or 0),
        n_ambiguous=len(status.get("ambiguous_group_ids") or []),
        surface_enabled=bool(status.get("surface_enabled", False)),
        surface_status_counts=status.get("surface_status_counts") or {},
    )
    if str(status.get("status", "")) != "complete" or outcome != "complete":
        if str(status.get("status", "")) != "complete":
            return False, f"status_not_complete:{status.get('status')}"
        return False, outcome_reason

    csv_path = Path(status.get("per_case_csv") or per_case_csv_path(run_id, frame_reference, directory_id, root))
    if not csv_path.is_file():
        return False, "missing_per_case_csv"

    vtp = centerline_vtp_path(directory_id, root)
    if not vtp.is_file():
        return False, "missing_input_vtp"
    stored_sha = status.get("input_vtp_sha256")
    if not stored_sha:
        return False, "missing_input_vtp_sha256_in_status"
    try:
        cur_sha = sha256_file(vtp)
    except OSError as exc:
        return False, f"input_vtp_unreadable:{exc}"
    if str(stored_sha) != str(cur_sha):
        return False, "input_vtp_sha256_mismatch"

    if not _segment_filter_equal(status.get("segment_filter"), expect_segment_ids):
        return False, "segment_filter_mismatch"

    surface_enabled = bool(status.get("surface_enabled", False))
    if expect_surface and not surface_enabled:
        return False, "surface_was_disabled"
                                                          

                                                              
    if expect_surface:
        bad = surface_incomplete_reason(status.get("surface_status_counts") or {}, surface_enabled=True)
        if bad is not None:
            return False, f"surface_incomplete:{bad}"

    ok_mesh, mesh_reason = verify_brain_mesh_inputs(status, expect_surface=expect_surface)
    if not ok_mesh:
        return False, mesh_reason

    ok_art, art_reason = verify_required_artifacts(status, csv_path=csv_path)
    if not ok_art:
        return False, art_reason

    n_intervals = status.get("n_intervals")
    if n_intervals is None:
        return False, "missing_n_intervals"
    try:
        n_csv = count_csv_data_rows(csv_path)
    except OSError as exc:
        return False, f"per_case_csv_unreadable:{exc}"
    if int(n_csv) != int(n_intervals):
        return False, f"per_case_csv_row_mismatch:csv={n_csv},status={n_intervals}"

                                                                     
    try:
        import pandas as pd

        cdf = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        return False, f"per_case_csv_parse_failed:{exc}"

    required_csv_cols = (
        "directory_id",
        "segment_id",
        "segment_index",
        "frame_reference",
        "fit_method",
        "measurement_version",
        "measurement_revision",
    )
    for col in required_csv_cols:
        if col not in cdf.columns:
            return False, f"per_case_csv_missing_key:{col}"

    def _nonempty(s: pd.Series) -> pd.Series:
        return s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str).str.lower() != "nan")

                                                                
    if len(cdf) > 0:
        for col in ("directory_id", "segment_id", "segment_index"):
            if not bool(_nonempty(cdf[col]).all()):
                return False, f"per_case_csv_{col}_missing_values"
        mism = cdf["directory_id"].astype(str) != str(directory_id)
        if bool(mism.any()):
            return False, "per_case_csv_directory_id_mismatch"

        key_df = cdf[["directory_id", "segment_id", "segment_index"]].copy()
        if bool(key_df.duplicated().any()):
            return False, "per_case_csv_duplicate_interval_keys"

        for col, expected in (
            ("frame_reference", frame_reference),
            ("fit_method", EXPECTED_FIT_METHOD),
            ("measurement_version", EXPECTED_MEASUREMENT_VERSION),
            ("measurement_revision", EXPECTED_MEASUREMENT_REVISION),
        ):
            if not bool(_nonempty(cdf[col]).all()):
                return False, f"per_case_csv_{col}_missing_values"
            uniq = sorted(cdf[col].astype(str).unique().tolist())
            if uniq != [str(expected)]:
                return False, f"per_case_csv_{col}_mismatch:{uniq}"

                                                            
    key_hash = interval_key_set_hash(cdf)
    stored_keys = status.get("interval_key_set_sha256")
    if stored_keys is None or str(stored_keys).strip() == "" or str(stored_keys).lower() == "nan":
        return False, "missing_interval_key_set_sha256"
    if str(stored_keys) != str(key_hash):
        return False, "per_case_csv_interval_key_set_mismatch"
    stored_csv_sha = status.get("per_case_csv_sha256")
    if stored_csv_sha is None or str(stored_csv_sha).strip() == "" or str(stored_csv_sha).lower() == "nan":
        return False, "missing_per_case_csv_sha256"
    try:
        if str(sha256_file(csv_path)) != str(stored_csv_sha):
            return False, "per_case_csv_sha256_mismatch"
    except OSError as exc:
        return False, f"per_case_csv_sha_unreadable:{exc}"

                                              
                                                         
                                                          
                                                         
    if expect_segment_ids is None:
        n_vtp = status.get("n_segments_in_vtp")
        if n_vtp is not None and int(n_intervals) < int(n_vtp):
            return False, f"incomplete_interval_coverage:n_intervals={n_intervals},n_vtp={n_vtp}"

    return True, "ok"


def parse_json_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if type(value) is bool:
        return value
    return None


def interval_key_set_hash(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                str(r.get("directory_id", "")).strip(),
                str(r.get("segment_id", "")).strip(),
                str(r.get("segment_index", "")).strip(),
            )
        )
    rows_sorted = sorted(set(rows))
    payload = "\n".join("|".join(t) for t in rows_sorted)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_run_manifest(
    run_id: str,
    directory_ids: Iterable[str],
    frame_references: Sequence[str] = PRIMARY_FRAME_REFERENCES,
    project_root: Optional[Path] = None,
    expected_exclusions: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    expected_exclusions = expected_exclusions or {}
    manifest: Dict[str, Any] = metadata_dict(run_id=validate_run_id(run_id), frames={})
    for fr in frame_references:
        validate_frame_reference(fr)
        per: Dict[str, Any] = {}
        counts = {
            "complete": 0,
            "partial": 0,
            "failed": 0,
            "missing": 0,
            "expected_exclusion": 0,
            "status_unreadable": 0,
            "interrupted": 0,
            "running": 0,
        }
        for did in sorted(set(map(str, directory_ids))):
            if did in expected_exclusions:
                per[did] = {"status": "expected_exclusion", "reason": expected_exclusions[did]}
                counts["expected_exclusion"] += 1
                continue
            st = read_participant_status(run_id, fr, did, project_root)
            if st is None:
                per[did] = {"status": "missing"}
                counts["missing"] += 1
                continue
            s = str(st.get("status", "failed"))
            if s not in counts:
                # unknown vocabulary → failed bucket, keep raw status in payload
                counts["failed"] += 1
            else:
                counts[s] += 1
            per[did] = st
        manifest["frames"][fr] = {"participants": per, "counts": counts}
    return manifest


def write_run_manifest(manifest: Dict[str, Any], run_id: str, project_root: Optional[Path] = None) -> Path:
    path = run_manifest_path(run_id, project_root)
    atomic_write_json(manifest, path)
    return path


def ensure_empty_csv(path: Path | str, columns: Sequence[str]) -> None:
    atomic_to_csv(pd.DataFrame(columns=list(columns)), path, index=False)


def version_revision_consistency(
    df: pd.DataFrame,
    *,
    expected_frame: Optional[str] = None,
    expected_voxel_size_mm: Optional[float] = None,
    expected_coordinate_unit: Optional[str] = EXPECTED_COORDINATE_UNIT_INPUT,
    run_config: Optional[Dict[str, Any]] = None,
    require_run_config: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": True,
        "n_rows": int(len(df)),
        "versions": [],
        "revisions": [],
        "frames": [],
        "fit_methods": [],
        "output_tags": [],
        "coordinate_units": [],
        "voxel_sizes": [],
        "reasons": [],
    }
    required = (
        "output_tag",
        "measurement_version",
        "measurement_revision",
        "frame_reference",
        "fit_method",
        "directory_id",
        "segment_id",
        "segment_index",
        "coordinate_unit_input",
        "voxel_size_mm",
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        out["ok"] = False
        out["reasons"].append(f"missing_columns:{','.join(missing_cols)}")
        return out

    def _nonempty_str(s: pd.Series) -> pd.Series:
        return s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str).str.lower() != "nan")

                                                        
    cfg_check = validate_run_config_contents(
        run_config, require=bool(require_run_config or run_config is not None)
    )
    if run_config is not None or require_run_config:
        if not cfg_check["ok"]:
            out["ok"] = False
            out["reasons"].extend(list(cfg_check.get("reasons") or []))
                                                
            if require_run_config or run_config is not None:
                out["run_config_check"] = cfg_check
                if len(df) == 0:
                    out["empty_table"] = True
                return out

    if expected_voxel_size_mm is None:
        if cfg_check.get("voxel_size") is not None:
            expected_voxel_size_mm = float(cfg_check["voxel_size"])
        elif require_run_config:
            out["ok"] = False
            out["reasons"].append("run_config_voxel_size_missing")
            return out
        else:
            expected_voxel_size_mm = EXPECTED_VOXEL_SIZE_MM
    else:
                             
        try:
            ev = float(expected_voxel_size_mm)
        except (TypeError, ValueError):
            out["ok"] = False
            out["reasons"].append(f"expected_voxel_size_mm_non_numeric:{expected_voxel_size_mm!r}")
            return out
        if not np.isfinite(ev) or ev <= 0:
            out["ok"] = False
            out["reasons"].append(f"expected_voxel_size_mm_not_positive_finite:{ev}")
            return out
        expected_voxel_size_mm = float(ev)

                                                   
    if len(df) == 0:
        out["empty_table"] = True
        out["run_config_check"] = cfg_check
        return out

    for col, key, expected in (
        ("measurement_version", "versions", EXPECTED_MEASUREMENT_VERSION),
        ("measurement_revision", "revisions", EXPECTED_MEASUREMENT_REVISION),
        ("frame_reference", "frames", expected_frame),
        ("fit_method", "fit_methods", EXPECTED_FIT_METHOD),
        ("output_tag", "output_tags", EXPECTED_OUTPUT_TAG),
    ):
        vals = df[col]
        nonempty = _nonempty_str(vals)
        if not bool(nonempty.all()):
            out["ok"] = False
            out["reasons"].append(f"{col}_has_missing_or_empty")
        uniq = sorted(vals.loc[nonempty].astype(str).unique().tolist()) if nonempty.any() else []
        out[key] = uniq
        if len(uniq) > 1:
            out["ok"] = False
            out["reasons"].append(f"{col}_mixed:{uniq}")
        if expected is not None and (len(uniq) != 1 or uniq[0] != expected):
            out["ok"] = False
            out["reasons"].append(f"{col}_expected_{expected}_got_{uniq}")

    for col in ("directory_id", "segment_id", "segment_index"):
        if not bool(_nonempty_str(df[col]).all()):
            out["ok"] = False
            out["reasons"].append(f"{col}_incomplete")

                   
    umask = _nonempty_str(df["coordinate_unit_input"])
    units = sorted(df.loc[umask, "coordinate_unit_input"].astype(str).unique().tolist()) if umask.any() else []
    out["coordinate_units"] = units
    if not bool(umask.all()):
        out["ok"] = False
        out["reasons"].append("coordinate_unit_input_partial_or_missing")
    if expected_coordinate_unit is not None and units != [str(expected_coordinate_unit)]:
        out["ok"] = False
        out["reasons"].append(f"coordinate_unit_input_expected_{expected_coordinate_unit}_got_{units}")

                                                
    vs = pd.to_numeric(df["voxel_size_mm"], errors="coerce")
    vs_arr = vs.to_numpy(dtype=float, na_value=np.nan)
    finite_pos = np.isfinite(vs_arr) & (vs_arr > 0)
    if not bool(np.all(finite_pos)):
        out["ok"] = False
        out["reasons"].append("voxel_size_mm_nonfinite_or_nonpositive_or_missing")
    uniq_vs = sorted({round(float(x), 12) for x in vs_arr[finite_pos].tolist()})
    out["voxel_sizes"] = uniq_vs
    if expected_voxel_size_mm is not None:
        if not uniq_vs or len(uniq_vs) != 1 or abs(uniq_vs[0] - float(expected_voxel_size_mm)) > 1e-12:
            out["ok"] = False
            out["reasons"].append(f"voxel_size_mm_expected_{expected_voxel_size_mm}_got_{uniq_vs}")
    return out
