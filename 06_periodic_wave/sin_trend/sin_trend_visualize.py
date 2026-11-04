#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend visualize."""

from __future__ import annotations

import html as html_lib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


PLOT_WINDOW = (1600, 1000)

# (label_ja, color_css, kind)  kind: point | arrow | mesh
BUCKLING_LEGEND_ITEMS: List[Tuple[str, str, str]] = [
    ("Observed centerline (original)", "#32CD32", "point"),
    ("Quadratic baseline P0", "#FF00FF", "point"),
    ("sin+trend fit curve (deep-sky-blue line)", "#00BFFF", "line"),
    ("Periodic component only P0+d_sin (white line)", "#FFFFFF", "line"),
    ("Periodic displacement d_sin (orange arrows)", "#FFA500", "arrow"),
    ("Principal displacement direction g1 (yellow arrows)", "#FFFF00", "arrow"),
    ("Cortex mesh (cropped)", "#FAFAE1", "mesh"),
    ("Cortex mesh normals (navy arrows)", "#000080", "arrow"),
]

SURFACE_LEGEND_ITEMS: List[Tuple[str, str, str]] = [
    ("Observed centerline after trim", "#32CD32", "point"),
    ("Quadratic baseline P0", "#FF00FF", "point"),
    ("sin+trend fit curve (deep-sky-blue line)", "#00BFFF", "line"),
    ("Periodic component only P0+d_sin (white line)", "#FFFFFF", "line"),
    ("Displaced points P + d_sin (reference)", "#FF0000", "point"),
    ("Periodic displacement d_sin (orange arrows)", "#FFA500", "arrow"),
    ("Principal displacement direction g1 (yellow arrows)", "#FFFF00", "arrow"),
    ("Effective cortex normal n_eff (cyan arrows)", "#00FFFF", "arrow"),
    ("Normal-estimation anchors", "#008000", "point"),
    ("Anchors with constraint", "#FF0000", "point_large"),
    ("Anchors without constraint", "#0000FF", "point_large"),
    ("Cortex mesh (cropped)", "#FAFAE1", "mesh"),
]


def _legend_plain_text(items: Sequence[Tuple[str, str, str]], *, title: str) -> str:
    lines = [title]
    for label, _color, kind in items:
        prefix = {
            "point": "●",
            "point_large": "⬤",
            "arrow": "→",
            "line": "─",
            "mesh": "▨",
        }.get(kind, "•")
        lines.append(f"{prefix} {label}")
    return "\n".join(lines)


BUCKLING_LEGEND_JA = _legend_plain_text(BUCKLING_LEGEND_ITEMS, title="Legend — Buckling Visualization")
SURFACE_LEGEND_JA = _legend_plain_text(SURFACE_LEGEND_ITEMS, title="Legend — Surface Constraint Visualization")


def _add_legend_panel(pl, text: str, *, font_size: int = 12) -> None:
    pl.add_text(text, position="upper_left", font_size=font_size, color="black", name="legend_panel")


def _write_legend_sidecar(html_path: Path, text: str) -> None:
    side = html_path.with_suffix(".legend.txt")
    side.write_text(text + "\n", encoding="utf-8")


def _kind_swatch_html(kind: str, color: str) -> str:
    c = html_lib.escape(color)
    if kind == "arrow":
        return (
            f'<span style="display:inline-block;width:22px;text-align:center;'
            f'color:{c};font-weight:700;font-size:16px;line-height:1;">→</span>'
        )
    if kind == "line":
        return (
            f'<span style="display:inline-block;width:18px;height:3px;margin:0 4px;'
            f'background:{c};border:1px solid #333;vertical-align:middle;"></span>'
        )
    if kind == "mesh":
        return (
            f'<span style="display:inline-block;width:14px;height:14px;margin:0 4px;'
            f'background:{c};border:1px solid #666;opacity:0.85;vertical-align:middle;"></span>'
        )
    size = "14px" if kind == "point_large" else "10px"
    return (
        f'<span style="display:inline-block;width:{size};height:{size};margin:0 4px;'
        f'border-radius:50%;background:{c};border:1px solid #333;vertical-align:middle;"></span>'
    )


def _polyline_mesh(pv, points: np.ndarray):
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
        return None
    if not np.all(np.isfinite(pts)):
        return None
    return pv.lines_from_points(pts, close=False)


def _fitted_curves_on_baseline(
    P0_trim: np.ndarray,
    U: Optional[np.ndarray],
    V: Optional[np.ndarray],
    fit_U: Optional[np.ndarray],
    fit_V: Optional[np.ndarray],
    d_sin: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    P0 = np.asarray(P0_trim, float)
    if d_sin is not None:
        ds = np.asarray(d_sin, float)
        if ds.shape == P0.shape and np.all(np.isfinite(ds)):
            out["P_fit_sin"] = P0 + ds
    if U is not None and V is not None and fit_U is not None and fit_V is not None:
        Ua = np.asarray(U, float)
        Va = np.asarray(V, float)
        fU = np.asarray(fit_U, float).ravel()
        fV = np.asarray(fit_V, float).ravel()
        if (
            Ua.shape == P0.shape
            and Va.shape == P0.shape
            and len(fU) == len(P0)
            and len(fV) == len(P0)
            and np.all(np.isfinite(fU))
            and np.all(np.isfinite(fV))
        ):
            out["P_fit_full"] = P0 + fU[:, None] * Ua + fV[:, None] * Va
    return out


def _add_fitted_curve_actors(pl, pv, curves: Dict[str, np.ndarray]) -> None:
    if "P_fit_full" in curves:
        line = _polyline_mesh(pv, curves["P_fit_full"])
        if line is not None:
            pl.add_mesh(line, color="#00BFFF", line_width=4, label="deep-sky-blue: sin+trend fit")
    if "P_fit_sin" in curves:
        line = _polyline_mesh(pv, curves["P_fit_sin"])
        if line is not None:
            pl.add_mesh(line, color="white", line_width=3, label="white: periodic component P0+d_sin")


def inject_html_legend_overlay(
    html_path: Path,
    items: Sequence[Tuple[str, str, str]],
    *,
    title: str,
) -> None:
    path = Path(html_path)
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if "sin-trend-legend-overlay" in text:
                         
        start = text.find("<!-- sin-trend-legend-overlay -->")
        end = text.find("<!-- /sin-trend-legend-overlay -->")
        if start >= 0 and end > start:
            text = text[:start] + text[end + len("<!-- /sin-trend-legend-overlay -->") :]

    rows = []
    for label, color, kind in items:
        rows.append(
            "<div style='margin:4px 0;display:flex;align-items:center;gap:6px;'>"
            f"{_kind_swatch_html(kind, color)}"
            f"<span>{html_lib.escape(label)}</span>"
            "</div>"
        )
    panel = (
        "<!-- sin-trend-legend-overlay -->\n"
        "<style>\n"
        "#sin-trend-legend-overlay{"
        "position:fixed;top:12px;left:12px;z-index:99999;"
        "max-width:min(420px,46vw);max-height:80vh;overflow:auto;"
        "background:rgba(255,255,255,0.94);color:#111;"
        "border:1px solid #333;border-radius:8px;padding:10px 12px;"
        "font:13px/1.35 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "box-shadow:0 2px 10px rgba(0,0,0,0.25);}"
        "#sin-trend-legend-overlay h3{margin:0 0 8px 0;font-size:14px;}"
        "#sin-trend-legend-overlay .hint{margin-top:8px;font-size:11px;color:#444;}"
        "</style>\n"
        "<div id='sin-trend-legend-overlay'>\n"
        f"<h3>{html_lib.escape(title)}</h3>\n"
        + "".join(rows)
        + "<div class='hint'>Deep-sky-blue: sin+trend fit on baseline. "
        "White: periodic component only P0+d_sin. "
        "Orange arrows: periodic displacement. "
        "Yellow arrows: principal direction g1. "
        "Cyan/navy: cortex normals.</div>\n"
        "</div>\n"
        "<!-- /sin-trend-legend-overlay -->\n"
    )
    marker = "</body>"
    idx = text.lower().rfind(marker)
    if idx < 0:
        text = text + "\n" + panel
    else:
        # preserve original casing of </body>
        end_tag_len = len(marker)
        text = text[:idx] + panel + text[idx:]
    path.write_text(text, encoding="utf-8")


def save_grid_search_png(
    *,
    out_path: Path,
    search: Dict[str, Any],
    candidates: pd.DataFrame,
    L_trimmed_vox: float,
    k_hat_vox: float,
    title: str = "",
    voxel_size_mm: float = 0.5,
    L_base_vox: Optional[float] = None,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return f"failed:matplotlib:{exc}"

    try:
        ks = np.asarray(search.get("ks", []), float) if search is not None else np.array([], float)
        if len(candidates) == 0:
            return "skipped_no_candidates"
        if "k_rad_per_voxel" in candidates.columns:
            order = np.argsort(candidates["k_rad_per_voxel"].to_numpy(float))
            cand = candidates.iloc[order]
            ks_plot = cand["k_rad_per_voxel"].to_numpy(float)
        elif "k" in candidates.columns:
            order = np.argsort(candidates["k"].to_numpy(float))
            cand = candidates.iloc[order]
            ks_plot = cand["k"].to_numpy(float)
        elif len(ks) > 0:
            cand = candidates
            ks_plot = ks[: len(cand)]
        else:
            return "skipped_no_candidates"
        R2s = pd.to_numeric(cand["R2"], errors="coerce").to_numpy(float) if "R2" in cand.columns else np.full(len(cand), np.nan)
        pR2s = pd.to_numeric(cand["partial_R2"], errors="coerce").to_numpy(float) if "partial_R2" in cand.columns else np.full(len(cand), np.nan)
        k_min = float(search.get("k_min", np.nanmin(ks_plot)))
        k_max = float(search.get("k_max", np.nanmax(ks_plot)))
        vs = float(voxel_size_mm) if np.isfinite(voxel_size_mm) and voxel_size_mm > 0 else 0.5

        def k_to_lambda_mm(k: np.ndarray) -> np.ndarray:
            k = np.asarray(k, float)
            out = np.full_like(k, np.nan, dtype=float)
            ok = np.isfinite(k) & (k > 0)
            out[ok] = (2.0 * np.pi / k[ok]) * vs
            return out

        def lambda_mm_to_k(lam: np.ndarray) -> np.ndarray:
            lam = np.asarray(lam, float)
            out = np.full_like(lam, np.nan, dtype=float)
            ok = np.isfinite(lam) & (lam > 0)
            out[ok] = (2.0 * np.pi * vs) / lam[ok]
            return out

        fig = plt.figure(figsize=(9, 6.0))
        ax1 = fig.add_subplot(111)
        if np.isfinite(k_min) and np.isfinite(k_max):
            ax1.axvspan(k_min, k_max, color="grey", alpha=0.15, label="search band")
        ax1.plot(ks_plot, R2s, "o-", label="R2 (fit quality)", color="C0", markersize=4)
        ax1.plot(ks_plot, pR2s, "x-", label="partial R2 (periodic contrib.)", color="C1", markersize=4)
        if np.isfinite(k_hat_vox):
            ax1.axvline(k_hat_vox, color="b", linestyle="-", alpha=0.6, label="selected k*")
            ymax = ax1.get_ylim()[1] if ax1.get_ylim()[1] else 1.0
            lam_star = float(k_to_lambda_mm(np.array([k_hat_vox]))[0])
            ax1.annotate(
                f"selected k* = {k_hat_vox:.3f}\n(lambda* = {lam_star:.2f} mm)",
                xy=(k_hat_vox, ymax * 0.92),
                xytext=(8, 8),
                textcoords="offset points",
                color="b",
                fontsize=9,
            )
        L_for_kL = float(L_base_vox) if L_base_vox is not None and np.isfinite(L_base_vox) and L_base_vox > 0 else float(L_trimmed_vox)
        if np.isfinite(L_for_kL) and L_for_kL > 0:
            k_L = 2.0 * np.pi / L_for_kL
            ax1.axvline(k_L, color="purple", linestyle=":", alpha=0.8, label="k_L = 2*pi/L_base (one cycle over baseline length)")
            ymax = ax1.get_ylim()[1] if ax1.get_ylim()[1] else 1.0
            ax1.annotate(
                f"k_L = 2*pi/L_base = {k_L:.3f}\n(one full cycle over L_base)",
                xy=(k_L, ymax * 0.78),
                xytext=(8, -4),
                textcoords="offset points",
                color="purple",
                fontsize=8,
            )
        ax1.set_xlabel("wave number k [rad/voxel]")
        ax1.set_ylabel("R2 / partial R2")
                                                                
                                          
        pos = ks_plot[np.isfinite(ks_plot) & (ks_plot > 0)]
        xlim_lo = float(k_min) if np.isfinite(k_min) and k_min > 0 else (float(np.min(pos)) if len(pos) else 1e-6)
        xlim_hi = float(k_max) if np.isfinite(k_max) and k_max > xlim_lo else (float(np.max(pos)) if len(pos) else xlim_lo * 2.0)
        if not (np.isfinite(xlim_lo) and np.isfinite(xlim_hi) and xlim_hi > xlim_lo > 0):
            return "failed:non_positive_k_axis_limits"
        pad = 0.02 * (xlim_hi - xlim_lo)
        x0 = max(xlim_lo - pad, xlim_lo * 0.5)
        x1 = xlim_hi + pad
        ax1.set_xlim(x0, x1)
                                                                 
                                                                                  
        ax_lam = ax1.secondary_xaxis(-0.18, functions=(k_to_lambda_mm, lambda_mm_to_k))
        lam_lo = float(k_to_lambda_mm(np.array([x1]))[0])
        lam_hi = float(k_to_lambda_mm(np.array([x0]))[0])
        if np.isfinite(lam_lo) and np.isfinite(lam_hi) and lam_hi > lam_lo > 0:
            candidates_mm = np.array(
                [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0, 200.0, 250.0]
            )
            ticks_mm = candidates_mm[(candidates_mm >= lam_lo * 0.98) & (candidates_mm <= lam_hi * 1.02)]
            if len(ticks_mm) < 3:
                ticks_mm = np.linspace(lam_lo, lam_hi, 5)
            ticks_mm = np.asarray(ticks_mm, float)
            ok_t = np.isfinite(ticks_mm) & (ticks_mm >= lam_lo) & (ticks_mm <= lam_hi)
            ticks_mm = ticks_mm[ok_t]
            if len(ticks_mm):
                                            
                ks_t = lambda_mm_to_k(ticks_mm)
                order = np.argsort(ks_t)                       
                ticks_mm = ticks_mm[order]
                ks_t = ks_t[order]
                span = max(float(x1 - x0), 1e-12)
                kept_idx: list[int] = []
                last_frac = -1e9
                min_frac = 0.12
                for i, kk in enumerate(ks_t):
                    if not (np.isfinite(kk) and kk > 0):
                        continue
                    frac = (float(kk) - float(x0)) / span
                    if not kept_idx or (frac - last_frac) >= min_frac:
                        kept_idx.append(i)
                        last_frac = frac
                                
                if len(kept_idx) > 6:
                    keep = np.unique(np.round(np.linspace(0, len(kept_idx) - 1, 6)).astype(int))
                    kept_idx = [kept_idx[j] for j in keep]
                ticks_use = ticks_mm[kept_idx]
                ax_lam.set_xticks(ticks_use)
                ax_lam.set_xticklabels([f"{v:.0f}" for v in ticks_use])
        ax_lam.set_xlabel(r"wavelength $\lambda = 2\pi/k$ [mm]  (same points as k above)")
        ax1.legend(loc="best", fontsize=8, framealpha=0.9)
        if title:
            ax1.set_title(title)
        fig.text(
            0.01,
            0.01,
            "Axes: top-bottom pair = k and lambda=2*pi/k; blue=selected k*; purple dotted=k_L; grey=search band",
            fontsize=8,
            ha="left",
            va="bottom",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out_path, format="png", dpi=300)
        plt.close(fig)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"failed:{exc}"


def _safe_close(pl) -> None:
    try:
        pl.close()
    except Exception:  # noqa: BLE001
        pass


def save_buckling_html(
    *,
    out_path: Path,
    P_full: np.ndarray,
    P0_full: np.ndarray,
    P_trim: np.ndarray,
    d_sin: np.ndarray,
    g1_3d: Optional[np.ndarray],
    mesh_path: Path,
    segment_id: Any,
    cell_index: int,
    P0_trim: Optional[np.ndarray] = None,
    U: Optional[np.ndarray] = None,
    V: Optional[np.ndarray] = None,
    fit_U: Optional[np.ndarray] = None,
    fit_V: Optional[np.ndarray] = None,
) -> str:
    try:
        import pyvista as pv
        from sin_trend_surface import crop_mesh, load_brain_mesh
    except Exception as exc:  # noqa: BLE001
        return f"failed:import:{exc}"
    if not mesh_path.is_file():
        return f"failed:missing_mesh:{mesh_path}"
    try:
        mesh, _ = load_brain_mesh(mesh_path)
        cropped = crop_mesh(P_full, mesh, margin=20)
        pl = pv.Plotter(off_screen=True, window_size=PLOT_WINDOW)
        pl.add_points(
            np.asarray(P_full, float),
            color="lime",
            point_size=5,
            render_points_as_spheres=True,
            label="lime: observed centerline",
        )
        pl.add_points(
            np.asarray(P0_full, float),
            color="magenta",
            point_size=5,
            render_points_as_spheres=True,
            label="magenta: baseline P0",
        )
        P0t = np.asarray(P0_trim if P0_trim is not None else P_trim, float)
        curves = _fitted_curves_on_baseline(P0t, U, V, fit_U, fit_V, d_sin)
        _add_fitted_curve_actors(pl, pv, curves)
        pl.add_arrows(
            np.asarray(P_trim, float),
            np.asarray(d_sin, float),
            color="orange",
            mag=1,
            label="orange arrows: periodic displacement d_sin",
        )
        if g1_3d is not None:
            pl.add_arrows(
                np.asarray(P_trim, float),
                np.asarray(g1_3d, float),
                mag=1,
                color="yellow",
                label="yellow arrows: principal direction g1",
            )
        pl.add_mesh(cropped, color="#fafae1", opacity=0.5, label="ivory: cortex mesh")
        if "Normals" in cropped.point_data:
            pl.add_mesh(
                cropped.glyph(orient="Normals", scale=False, factor=0.5),
                color="navy",
                opacity=0.7,
                label="navy arrows: cortex normals",
            )
        pl.camera_position = "xy"
        pl.camera.azimuth = 180
        pl.camera.zoom(1.2)
        pl.add_legend(bcolor=(1.0, 1.0, 1.0), border=True, size=(0.28, 0.28))
        _add_legend_panel(pl, BUCKLING_LEGEND_JA, font_size=11)
        pl.add_title(f"Buckling Visualization, segment ID: {segment_id}, segment index: {cell_index}", font_size=14)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pl.export_html(str(out_path))
        _write_legend_sidecar(out_path, BUCKLING_LEGEND_JA)
        inject_html_legend_overlay(
            out_path,
            BUCKLING_LEGEND_ITEMS,
            title="Legend — Buckling Visualization",
        )
        _safe_close(pl)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"failed:{exc}"


def save_surface_constraint_html(
    *,
    out_path: Path,
    P_full: np.ndarray,
    P0_full: np.ndarray,
    P_trim: np.ndarray,
    d_sin: np.ndarray,
    g1_3d: Optional[np.ndarray],
    n_eff: Optional[np.ndarray],
    anchor_indices: Optional[np.ndarray],
    anchor_constraint: Optional[np.ndarray],
    mesh_path: Path,
    segment_id: Any,
    cell_index: int,
    P0_trim: Optional[np.ndarray] = None,
    U: Optional[np.ndarray] = None,
    V: Optional[np.ndarray] = None,
    fit_U: Optional[np.ndarray] = None,
    fit_V: Optional[np.ndarray] = None,
) -> str:
    try:
        import pyvista as pv
        from sin_trend_surface import crop_mesh, load_brain_mesh
    except Exception as exc:  # noqa: BLE001
        return f"failed:import:{exc}"
    if not mesh_path.is_file():
        return f"failed:missing_mesh:{mesh_path}"
    try:
        mesh, _ = load_brain_mesh(mesh_path)
        cropped = crop_mesh(P_full, mesh, margin=20)
        P_trim = np.asarray(P_trim, float)
        d_sin = np.asarray(d_sin, float)
        Psin = P_trim + d_sin
        pl = pv.Plotter(off_screen=True, window_size=PLOT_WINDOW)
        pl.add_points(P_trim, color="lime", point_size=5, label="lime: observed centerline (after trim)")
        pl.add_points(np.asarray(P0_full, float), color="magenta", point_size=5, label="magenta: baseline P0")
        P0t = np.asarray(P0_trim if P0_trim is not None else P_trim, float)
        curves = _fitted_curves_on_baseline(P0t, U, V, fit_U, fit_V, d_sin)
        _add_fitted_curve_actors(pl, pv, curves)
        pl.add_mesh(cropped, color="#fafae1", opacity=0.5, label="ivory: cortex mesh")
        pl.add_points(Psin, color="red", point_size=5, label="red: P + d_sin (reference)")
        pl.add_arrows(P_trim, d_sin, color="orange", mag=1, label="orange arrows: periodic displacement d_sin")
        if g1_3d is not None:
            pl.add_arrows(P_trim, np.asarray(g1_3d, float), mag=1.0, color="yellow", label="yellow arrows: principal direction g1")
        if n_eff is not None:
            pl.add_arrows(P_trim, np.asarray(n_eff, float), mag=1.0, color="cyan", label="cyan arrows: effective cortex normal n_eff")
        if anchor_indices is not None and len(anchor_indices):
            aidx = np.asarray(anchor_indices, int)
            aidx = aidx[(aidx >= 0) & (aidx < len(P_trim))]
            anchors = P_trim[aidx]
            pl.add_points(anchors, color="green", point_size=10, label="green: normal-estimation anchors")
            if anchor_constraint is not None and len(anchor_constraint) == len(aidx):
                ac = np.asarray(anchor_constraint, bool)
                if np.any(ac):
                    pl.add_points(anchors[ac], color="red", point_size=20, label="large red: constrained anchors")
                if np.any(~ac):
                    pl.add_points(anchors[~ac], color="blue", point_size=20, label="large blue: unconstrained anchors")
        pl.add_legend(bcolor=(1.0, 1.0, 1.0), border=True, size=(0.30, 0.32))
        _add_legend_panel(pl, SURFACE_LEGEND_JA, font_size=10)
        pl.add_title(
            f"Surface Constraint Visualization, segment ID: {segment_id}, segment index: {cell_index}",
            font_size=14,
        )
        pl.camera_position = "xy"
        pl.camera.azimuth = 180
        pl.camera.zoom(1.2)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pl.export_html(str(out_path))
        _write_legend_sidecar(out_path, SURFACE_LEGEND_JA)
        inject_html_legend_overlay(
            out_path,
            SURFACE_LEGEND_ITEMS,
            title="Legend — Surface Constraint Visualization",
        )
        _safe_close(pl)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"failed:{exc}"


def save_interval_visualizations(
    *,
    out_dir: Path,
    directory_id: str,
    segment_id: Any,
    cell_index: int,
    search: Optional[Dict[str, Any]],
    candidates: pd.DataFrame,
    pointwise: Dict[str, Any],
    row: Dict[str, Any],
    mesh_path: Optional[Path],
    L_trimmed_vox: float,
    L_base_vox: Optional[float] = None,
) -> Dict[str, str]:
    status: Dict[str, str] = {
        "grid_search_png": "skipped",
        "buckling_html": "skipped",
        "surface_constraint_html": "skipped",
    }
    sid, cidx = segment_id, int(cell_index)
    if L_base_vox is None:
        lb_mm = row.get("L_base_mm", row.get("L_baseline_fit_mm", np.nan))
        vs = float(row.get("voxel_size_mm", 0.5) or 0.5)
        if np.isfinite(lb_mm) and vs > 0:
            L_base_vox = float(lb_mm) / vs
    if search and len(candidates) and np.isfinite(search.get("k_hat", np.nan)):
        png = out_dir / f"{directory_id}_grid_search_result_{sid}_{cidx}.png"
        status["grid_search_png"] = save_grid_search_png(
            out_path=png,
            search=search,
            candidates=candidates,
            L_trimmed_vox=L_trimmed_vox,
            L_base_vox=L_base_vox,
            k_hat_vox=float(search["k_hat"]),
            title=f"{directory_id} seg {sid} idx {cidx}",
            voxel_size_mm=float(row.get("voxel_size_mm", 0.5) or 0.5),
        )
        status["grid_search_png_path"] = str(png) if status["grid_search_png"] == "ok" else ""

    need = ("P_full", "P0_full", "P", "d_sin")
    if mesh_path is not None and all(k in pointwise for k in need):
        html1 = out_dir / f"{directory_id}_buckling_visualization_{sid}_{cidx}.html"
        status["buckling_html"] = save_buckling_html(
            out_path=html1,
            P_full=pointwise["P_full"],
            P0_full=pointwise["P0_full"],
            P_trim=pointwise["P"],
            d_sin=pointwise["d_sin"],
            g1_3d=pointwise.get("g1_3d"),
            mesh_path=Path(mesh_path),
            segment_id=sid,
            cell_index=cidx,
            P0_trim=pointwise.get("P0"),
            U=pointwise.get("U"),
            V=pointwise.get("V"),
            fit_U=pointwise.get("fit_U"),
            fit_V=pointwise.get("fit_V"),
        )
        status["buckling_html_path"] = str(html1) if status["buckling_html"] == "ok" else ""

        if pointwise.get("n_eff") is not None:
            html2 = out_dir / f"{directory_id}_buckling_surface_constraint_visualization_{sid}_{cidx}.html"
            aidx = pointwise.get("anchor_indices")
            # filled constraint on anchors if available
            ac = None
            if aidx is not None and pointwise.get("constraint_filled") is not None:
                filled = np.asarray(pointwise["constraint_filled"], bool)
                aidx_arr = np.asarray(aidx, int)
                ok = (aidx_arr >= 0) & (aidx_arr < len(filled))
                ac = filled[aidx_arr[ok]] if np.any(ok) else None
                aidx = aidx_arr[ok] if np.any(ok) else aidx_arr
            status["surface_constraint_html"] = save_surface_constraint_html(
                out_path=html2,
                P_full=pointwise["P_full"],
                P0_full=pointwise["P0_full"],
                P_trim=pointwise["P"],
                d_sin=pointwise["d_sin"],
                g1_3d=pointwise.get("g1_3d"),
                n_eff=pointwise.get("n_eff"),
                anchor_indices=aidx,
                anchor_constraint=ac,
                mesh_path=Path(mesh_path),
                segment_id=sid,
                cell_index=cidx,
                P0_trim=pointwise.get("P0"),
                U=pointwise.get("U"),
                V=pointwise.get("V"),
                fit_U=pointwise.get("fit_U"),
                fit_V=pointwise.get("fit_V"),
            )
            status["surface_constraint_html_path"] = str(html2) if status["surface_constraint_html"] == "ok" else ""
        else:
            status["surface_constraint_html"] = "skipped_no_normals"
    return status
