#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 08 render core inplane plots."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)

from sin_trend_classify import (  # noqa: E402
    SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL,
    SURFACE_LABEL_SUBSTANTIAL_NORMAL,
)
from sin_trend_paths import (  # noqa: E402
    FRAME_BASELINE,
    FRAME_REFERENCES,
    OUTPUT_TAG,
    analysis_csv_path,
    analysis_run_root,
    atomic_to_csv,
    brain_mesh_path,
    candidate_dir,
    figures_root,
    project_root_from_here,
    surface_config_for_run,
    validate_frame_reference,
    validate_run_id,
    validate_surface_tag,
)
from sin_trend_surface_stage import (  # noqa: E402
    apply_surface_eval_to_row,
    load_pointwise_npz,
    pointwise_path_for_row,
)
from sin_trend_visualize import save_interval_visualizations  # noqa: E402

PERIODICITY_NOT_CORE = "NotCore"

# CLI token -> formal surface_constraint label
_SURFACE_CONSTRAINT_ALIASES: Dict[str, str] = {
    "predominantly_tangential": SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL,
    "pt": SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL,
    "core_inplane": SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL,
    "tangential": SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL,
    SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL.lower(): SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL,
    "substantial_normal": SURFACE_LABEL_SUBSTANTIAL_NORMAL,
    "sn": SURFACE_LABEL_SUBSTANTIAL_NORMAL,
    "substantial": SURFACE_LABEL_SUBSTANTIAL_NORMAL,
    SURFACE_LABEL_SUBSTANTIAL_NORMAL.lower(): SURFACE_LABEL_SUBSTANTIAL_NORMAL,
}

                                                             
_PERIODICITY_ALIASES: Dict[str, str] = {
    "not_core": PERIODICITY_NOT_CORE,
    "notcore": PERIODICITY_NOT_CORE,
    "not-core": PERIODICITY_NOT_CORE,
    PERIODICITY_NOT_CORE.lower(): PERIODICITY_NOT_CORE,
}


@dataclass
class TargetSpec:
    surface_labels: List[str] = field(default_factory=list)
    periodicity_labels: List[str] = field(default_factory=list)


def _unique_append(out: List[str], seen: Set[str], label: str) -> None:
    if label not in seen:
        seen.add(label)
        out.append(label)


def resolve_target_spec(
    surface_tokens: Optional[Sequence[str]],
    periodicity_tokens: Optional[Sequence[str]],
) -> TargetSpec:
    """
    Build selection unions.
    Default (both omitted): Predominantly tangential only.
    --periodicity not_core alone: NotCore only (no default PT).
    """
    spec = TargetSpec()
    surf_seen: Set[str] = set()
    per_seen: Set[str] = set()

    if surface_tokens is None and periodicity_tokens is None:
        spec.surface_labels = [SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL]
        return spec

    for tok in surface_tokens or []:
        key = str(tok).strip().lower()
        # Convenience: allow not_core on --surface-constraint as well.
        if key in _PERIODICITY_ALIASES:
            _unique_append(spec.periodicity_labels, per_seen, _PERIODICITY_ALIASES[key])
            continue
        if key not in _SURFACE_CONSTRAINT_ALIASES:
            allowed = "predominantly_tangential|pt|core_inplane, substantial_normal|sn, not_core"
            raise ValueError(f"unknown --surface-constraint {tok!r}; use one of: {allowed}")
        _unique_append(spec.surface_labels, surf_seen, _SURFACE_CONSTRAINT_ALIASES[key])

    for tok in periodicity_tokens or []:
        key = str(tok).strip().lower()
        if key not in _PERIODICITY_ALIASES:
            raise ValueError(
                f"unknown --periodicity {tok!r}; use not_core "
                "(InvalidFit/NotEligible are not supported: no saved pointwise/candidates)"
            )
        _unique_append(spec.periodicity_labels, per_seen, _PERIODICITY_ALIASES[key])

    if not spec.surface_labels and not spec.periodicity_labels:
        raise ValueError("no target classes resolved")
    return spec


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            f"Backfill visualizations from saved artifacts ({OUTPUT_TAG}); "
            "default class = Predominantly tangential (core_inplane)"
        )
    )
    p.add_argument("--run-id", required=True)
    p.add_argument("--frame-reference", default=FRAME_BASELINE, choices=list(FRAME_REFERENCES))
    p.add_argument(
        "--source-csv",
        type=Path,
        default=None,
        help="interval table (default: formal analysis CSV)",
    )
    p.add_argument("--directory-id", action="append", default=None)
    p.add_argument("--limit", type=int, default=None, help="process at most N intervals (smoke)")
    p.add_argument(
        "--surface-constraint",
        action="append",
        default=None,
        metavar="CLASS",
        help=(
            "repeatable; predominantly_tangential|pt|core_inplane, "
            "substantial_normal|sn, and/or not_core "
            "(default if both this and --periodicity omitted: predominantly_tangential)"
        ),
    )
    p.add_argument(
        "--periodicity",
        action="append",
        default=None,
        metavar="CLASS",
        help="repeatable; currently not_core only (fit_ok segments with saved artifacts)",
    )
    p.add_argument("--surface-tag", default="primary")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--force", action="store_true", help="overwrite existing figure files")
    p.add_argument("--quiet", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="list targets only (default)")
    mode.add_argument("--execute", action="store_true", help="actually render and save figures")
    return p.parse_args(argv)


def _configure_runtime() -> None:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(k, "1")


def _voxel_size_mm(row: Dict[str, Any], surface_cfg: Dict[str, Any]) -> float:
    for key in ("voxel_size_mm", "voxel_size"):
        try:
            v = float(row.get(key, np.nan))
            if np.isfinite(v) and v > 0:
                return v
        except (TypeError, ValueError):
            pass
    try:
        v = float(surface_cfg.get("voxel_size", np.nan))
        if np.isfinite(v) and v > 0:
            return v
    except (TypeError, ValueError):
        pass
    return 0.5


def _figure_paths(fig_dir: Path, directory_id: str, segment_id: int, cell_index: int) -> Dict[str, Path]:
    sid, idx = int(segment_id), int(cell_index)
    return {
        "grid_search_png": fig_dir / f"{directory_id}_grid_search_result_{sid}_{idx}.png",
        "buckling_html": fig_dir / f"{directory_id}_buckling_visualization_{sid}_{idx}.html",
        "surface_constraint_html": fig_dir
        / f"{directory_id}_buckling_surface_constraint_visualization_{sid}_{idx}.html",
    }


def _all_exist_ok(paths: Dict[str, Path]) -> bool:
    return all(p.is_file() and p.stat().st_size > 0 for p in paths.values())


def _candidate_path(
    run_id: str, frame_reference: str, directory_id: str, segment_id: int, cell_index: int, project_root: Path
) -> Path:
    stem = f"{directory_id}_seg{int(segment_id)}_idx{int(cell_index)}"
    return candidate_dir(run_id, frame_reference, project_root) / f"{stem}_candidates.csv"


def _reconstruct_search(candidates: pd.DataFrame, row: Dict[str, Any], voxel_size_mm: float) -> Dict[str, Any]:
    search: Dict[str, Any] = {}
    if len(candidates) and "k_rad_per_voxel" in candidates.columns:
        ks = pd.to_numeric(candidates["k_rad_per_voxel"], errors="coerce")
        if ks.notna().any():
            search["k_min"] = float(ks.min())
            search["k_max"] = float(ks.max())
        sel = candidates
        if "is_selected" in candidates.columns:
            m = pd.to_numeric(candidates["is_selected"], errors="coerce").fillna(0).astype(int) == 1
            if m.any():
                sel = candidates.loc[m]
        k_hat = pd.to_numeric(sel["k_rad_per_voxel"], errors="coerce")
        if k_hat.notna().any():
            search["k_hat"] = float(k_hat.iloc[0])
    if "k_hat" not in search or not np.isfinite(search.get("k_hat", np.nan)):
        try:
            k_mm = float(row.get("k_hat_mm", np.nan))
        except (TypeError, ValueError):
            k_mm = float("nan")
        if np.isfinite(k_mm) and voxel_size_mm > 0:
            search["k_hat"] = float(k_mm) * float(voxel_size_mm)
    return search


def select_targets(
    source: pd.DataFrame,
    *,
    spec: TargetSpec,
    directory_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    if "fit_status" not in source.columns:
        raise ValueError("source CSV missing column fit_status")
    fit_ok = source["fit_status"].astype(str).str.strip().eq("ok")
    class_mask = pd.Series(False, index=source.index)
    if spec.surface_labels:
        if "surface_constraint" not in source.columns:
            raise ValueError("source CSV missing column surface_constraint")
        class_mask |= source["surface_constraint"].astype(str).isin({str(x) for x in spec.surface_labels})
    if spec.periodicity_labels:
        if "periodicity" not in source.columns:
            raise ValueError("source CSV missing column periodicity")
        class_mask |= source["periodicity"].astype(str).isin({str(x) for x in spec.periodicity_labels})
    mask = fit_ok & class_mask
    if directory_ids:
        want = {str(x) for x in directory_ids}
        mask &= source["directory_id"].astype(str).isin(want)
    sub = source.loc[mask].copy()
    for col in ("directory_id", "segment_id", "segment_index"):
        if col not in sub.columns:
            raise ValueError(f"source CSV missing column {col}")
    sub = sub.sort_values(["directory_id", "segment_id", "segment_index"]).reset_index(drop=True)
    if limit is not None:
        sub = sub.iloc[: max(0, int(limit))].copy()
    return sub


def render_one(
    row: Dict[str, Any],
    *,
    run_id: str,
    frame_reference: str,
    project_root: Path,
    surface_cfg: Dict[str, Any],
    surface_tag: str,
    force: bool,
    quiet: bool,
) -> Dict[str, Any]:
    did = str(row["directory_id"])
    sid = int(row["segment_id"])
    idx = int(row["segment_index"])
    fig_dir = figures_root(run_id, frame_reference, project_root) / did
    paths = _figure_paths(fig_dir, did, sid, idx)
    rec: Dict[str, Any] = {
        "directory_id": did,
        "segment_id": sid,
        "segment_index": idx,
        "surface_constraint": str(row.get("surface_constraint", "") or ""),
        "periodicity": str(row.get("periodicity", "") or ""),
        "outcome": "",
        "grid_search_png": "",
        "buckling_html": "",
        "surface_constraint_html": "",
        "grid_path": str(paths["grid_search_png"]),
        "buckling_path": str(paths["buckling_html"]),
        "surface_path": str(paths["surface_constraint_html"]),
        "reason": "",
        "elapsed_sec": 0.0,
    }
    t0 = time.time()
    try:
        if not force and _all_exist_ok(paths):
            rec["outcome"] = "skipped_exists"
            rec["grid_search_png"] = "skipped_exists"
            rec["buckling_html"] = "skipped_exists"
            rec["surface_constraint_html"] = "skipped_exists"
            return rec

        cand_path = _candidate_path(run_id, frame_reference, did, sid, idx, project_root)
        pw_path = pointwise_path_for_row(
            row, run_id=run_id, frame_reference=frame_reference, project_root=project_root
        )
        side = str(row.get("side", "") or "").strip()
        mesh_p = brain_mesh_path(did, side, project_root) if side else None

        missing: List[str] = []
        if not cand_path.is_file():
            missing.append(f"candidates:{cand_path.name}")
        if pw_path is None or not pw_path.is_file():
            missing.append("pointwise")
        if mesh_p is None or not mesh_p.is_file():
            missing.append(f"mesh:side={side or 'missing'}")
        if missing:
            rec["outcome"] = "missing_artifacts"
            rec["reason"] = ";".join(missing)
            return rec

        candidates = pd.read_csv(cand_path)
        pw, _scalars = load_pointwise_npz(pw_path)
        vs = _voxel_size_mm(row, surface_cfg)
        search = _reconstruct_search(candidates, row, vs)
        L_trim_vox = (
            float(row["L_trimmed_mm"]) / vs
            if np.isfinite(float(row.get("L_trimmed_mm", np.nan)))
            else float("nan")
        )
        L_base_mm = row.get("L_base_mm", row.get("L_baseline_fit_mm", np.nan))
        try:
            L_base_mm_f = float(L_base_mm)
        except (TypeError, ValueError):
            L_base_mm_f = float("nan")
        L_base_vox = L_base_mm_f / vs if np.isfinite(L_base_mm_f) and vs > 0 else float("nan")

        # Remove existing outputs when forcing so status reflects a fresh write.
        if force:
            for p in paths.values():
                if p.is_file():
                    p.unlink()

        viz_st = save_interval_visualizations(
            out_dir=fig_dir,
            directory_id=did,
            segment_id=sid,
            cell_index=idx,
            search=search,
            candidates=candidates,
            pointwise=pw,
            row={**row, "voxel_size_mm": vs},
            mesh_path=mesh_p,
            L_trimmed_vox=L_trim_vox,
            L_base_vox=L_base_vox,
        )
        rec["grid_search_png"] = str(viz_st.get("grid_search_png", ""))
        rec["buckling_html"] = str(viz_st.get("buckling_html", ""))

        # Surface HTML needs n_eff; NPZ from --no-plots does not store it.
        # Recompute surface eval in-memory (no measurement/surface CSV rewrite).
        surf_row, _sst = apply_surface_eval_to_row(
            dict(row),
            pw=pw,
            surface_cfg=surface_cfg,
            project_root=project_root,
            run_id=run_id,
            surface_tag=surface_tag,
            verbose=False,
            enable_plots=True,
            frame_reference=frame_reference,
        )
        surf_status = str(surf_row.get("viz_surface_html_status", "") or "")
        surf_written = str(surf_row.get("viz_surface_html_path", "") or "").strip()
        target_surf = paths["surface_constraint_html"]
        if surf_status == "ok" and surf_written:
            src = Path(surf_written)
            if src.is_file():
                target_surf.parent.mkdir(parents=True, exist_ok=True)
                if src.resolve() != target_surf.resolve():
                    shutil.copy2(src, target_surf)
                surf_status = "ok" if target_surf.is_file() else "failed:copy_to_figures_root"
            else:
                surf_status = "failed:surface_plot_missing"
        rec["surface_constraint_html"] = surf_status

        statuses = (
            rec["grid_search_png"],
            rec["buckling_html"],
            rec["surface_constraint_html"],
        )
        if all(s == "ok" for s in statuses):
            rec["outcome"] = "complete"
        elif any(str(s).startswith("failed") for s in statuses):
            rec["outcome"] = "failed"
        elif any(s == "ok" for s in statuses):
            rec["outcome"] = "partial"
        else:
            rec["outcome"] = "failed"
            if not rec["reason"]:
                rec["reason"] = "no_plot_written"

        if not quiet:
            print(
                f"{did} seg={sid} idx={idx}: {rec['outcome']} "
                f"grid={rec['grid_search_png']} buckling={rec['buckling_html']} "
                f"surface={rec['surface_constraint_html']}"
            )
        return rec
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        rec["outcome"] = "failed"
        rec["reason"] = f"{type(exc).__name__}:{exc}"
        if not quiet:
            print(f"{did} seg={sid} idx={idx}: failed {rec['reason']}")
        return rec
    finally:
        rec["elapsed_sec"] = float(time.time() - t0)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _configure_runtime()
    root = project_root_from_here(__file__)
    run_id = validate_run_id(args.run_id)
    frame_reference = validate_frame_reference(args.frame_reference)
    surface_tag = validate_surface_tag(args.surface_tag)
    source_path = Path(args.source_csv) if args.source_csv is not None else analysis_csv_path(run_id, frame_reference, root)
    if not source_path.is_file():
        print(f"error: source CSV not found: {source_path}", file=sys.stderr)
        return 2

    try:
        spec = resolve_target_spec(args.surface_constraint, args.periodicity)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    df = pd.read_csv(source_path, low_memory=False)
    targets = select_targets(
        df,
        spec=spec,
        directory_ids=args.directory_id,
        limit=args.limit,
    )
    print(f"source={source_path}")
    print(f"run_id={run_id} frame={frame_reference} surface_tag={surface_tag}")
    print(f"surface_constraint filter: {spec.surface_labels or '(none)'}")
    print(f"periodicity filter: {spec.periodicity_labels or '(none)'}")
    print(f"fit_ok targets: {len(targets)}")
    if len(targets) and "surface_constraint" in targets.columns:
        print("by surface_constraint:", targets["surface_constraint"].astype(str).value_counts(dropna=False).to_dict())
    if len(targets) and "periodicity" in targets.columns:
        print("by periodicity:", targets["periodicity"].astype(str).value_counts(dropna=False).to_dict())
    if len(targets):
        cols = ["directory_id", "segment_id", "segment_index"]
        for c in ("surface_constraint", "periodicity"):
            if c in targets.columns:
                cols.append(c)
        head = targets[cols].head(10)
        print("first targets:")
        print(head.to_string(index=False))

    do_execute = bool(args.execute) and not bool(args.dry_run)
    if not do_execute:
        print("[dry-run] pass --execute to render figures")
        return 0

    surface_cfg = surface_config_for_run(run_id, surface_tag=surface_tag, project_root=root)
    rows = targets.to_dict(orient="records")
    results: List[Dict[str, Any]] = []
    n_workers = max(1, int(args.n_workers))

    if n_workers == 1:
        for rec in rows:
            results.append(
                render_one(
                    rec,
                    run_id=run_id,
                    frame_reference=frame_reference,
                    project_root=root,
                    surface_cfg=surface_cfg,
                    surface_tag=surface_tag,
                    force=bool(args.force),
                    quiet=bool(args.quiet),
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [
                ex.submit(
                    render_one,
                    rec,
                    run_id=run_id,
                    frame_reference=frame_reference,
                    project_root=root,
                    surface_cfg=surface_cfg,
                    surface_tag=surface_tag,
                    force=bool(args.force),
                    quiet=bool(args.quiet),
                )
                for rec in rows
            ]
            for fut in as_completed(futs):
                results.append(fut.result())

    out_df = pd.DataFrame(results)
    if len(out_df):
        out_df = out_df.sort_values(["directory_id", "segment_id", "segment_index"]).reset_index(drop=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    status_dir = analysis_run_root(run_id, root) / "plot_backfill"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"surface_plots_status_{stamp}.csv"
    atomic_to_csv(out_df, status_path, index=False)

    counts = out_df["outcome"].value_counts(dropna=False).to_dict() if len(out_df) else {}
    print(f"status CSV: {status_path}")
    print(f"outcome counts: {counts}")
    n_fail = int((out_df["outcome"] == "failed").sum()) if len(out_df) else 0
    n_miss = int((out_df["outcome"] == "missing_artifacts").sum()) if len(out_df) else 0
    return 2 if (n_fail or n_miss) else 0


if __name__ == "__main__":
    raise SystemExit(main())
