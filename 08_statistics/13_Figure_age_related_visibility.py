#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate paper figure: 13_Figure_age_related_visibility."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

                                     
_here = Path(__file__).resolve()
_mpl_dir = _here.parent.parent / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib.pyplot as plt

try:
    import statsmodels.api as sm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    sm = None  # type: ignore


DEFAULT_RUN_ID = "run_154_baseline_v2_r0"
_CM_TO_INCH = 1.0 / 2.54
_FIG_WIDTH_CM = 9.0
_FIG_WIDTH_IN = _FIG_WIDTH_CM * _CM_TO_INCH
                                    
_PANEL_LABEL_PAD_PT = 8.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "statistics" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure project root is importable (so we can import figures/plot_config.py)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SIN_TREND_V2 = PROJECT_ROOT / "06_periodic_wave"
if str(_SIN_TREND_V2) not in sys.path:
    sys.path.insert(0, str(_SIN_TREND_V2))

from sin_trend_paths import FRAME_BASELINE, analysis_csv_path  # noqa: E402

# Apply common plotting rules (seaborn theme, rcParams, colors, etc.)
from figures.plot_config import (  # noqa: E402
    add_panel_label,
    alpha,
    blue_color,
    ms,
    save_figure,
)


def _length_mm_column(df: pd.DataFrame) -> str | None:
    for c in ("L_trimmed_mm", "L_full_mm"):
        if c in df.columns:
            return c
    return None


def _is_true_like(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    s2 = s.astype(str).str.strip().str.lower()
    return s2.isin({"true", "1", "t", "yes", "y"})


def aggregate_subject_level(df: pd.DataFrame) -> pd.DataFrame:
    required = ["IXI_ID", "AGE", "core_inplane"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    sub = df.copy()
    sub["IXI_ID"] = sub["IXI_ID"].astype(str)
    sub["_is_core_inplane"] = _is_true_like(sub["core_inplane"])
    sub["_row"] = 1
    length_col = _length_mm_column(sub)
    if length_col is None:
        length = pd.Series(np.nan, index=sub.index)
    else:
        length = pd.to_numeric(sub[length_col], errors="coerce")
    sub["_L_inplane"] = np.where(sub["_is_core_inplane"] & length.notna(), length, 0.0)

    agg_parts = {
        "AGE": ("AGE", "first"),
        "n_total": ("_row", "sum"),
        "n_core_inplane": ("_is_core_inplane", "sum"),
        "sum_L_inplane": ("_L_inplane", "sum"),
    }
    if "total_vessel_length" in sub.columns:
        agg_parts["total_vessel_length"] = ("total_vessel_length", "first")

    out = sub.groupby("IXI_ID", dropna=False).agg(**agg_parts).reset_index(drop=True)
    out["AGE"] = pd.to_numeric(out["AGE"], errors="coerce")
    out["n_total"] = pd.to_numeric(out["n_total"], errors="coerce")
    out["n_core_inplane"] = pd.to_numeric(out["n_core_inplane"], errors="coerce")
    out["sum_L_inplane"] = pd.to_numeric(out["sum_L_inplane"], errors="coerce")
    out = out.dropna(subset=["AGE", "n_total"])
    out = out.loc[out["n_total"] > 0].copy()
    out["ratio_core_inplane"] = out["n_core_inplane"] / out["n_total"]
    return out


def _scatter(ax, x, y) -> None:
    ax.plot(
        x,
        y,
        linestyle="None",
        marker="o",
        color=blue_color,
        markeredgewidth=0,
        markersize=ms,
        alpha=alpha,
        zorder=2,
    )


def _add_unadjusted_ols_fit(
    ax: plt.Axes,
    age: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
) -> None:
    """Unadjusted OLS fit line: y ~ AGE."""
    work = pd.DataFrame({"AGE": age, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 3 or work["AGE"].nunique() < 2:
        return
    try:
        if sm is None:
            xg = np.linspace(float(work["AGE"].min()), float(work["AGE"].max()), 200)
            slope, intercept = np.polyfit(work["AGE"].to_numpy(dtype=float), work["y"].to_numpy(dtype=float), 1)
            yhat = intercept + slope * xg
            ax.plot(xg, yhat, color="k", lw=1.2, label=label)
            return
        x = sm.add_constant(work["AGE"].to_numpy(dtype=float))
        res = sm.OLS(work["y"].to_numpy(dtype=float), x).fit()
        xg = np.linspace(float(work["AGE"].min()), float(work["AGE"].max()), 200)
        pr = res.get_prediction(sm.add_constant(xg))
        sf = pr.summary_frame(alpha=0.05)
        mean = sf["mean"].to_numpy(dtype=float)
        lo = sf["mean_ci_lower"].to_numpy(dtype=float)
        hi = sf["mean_ci_upper"].to_numpy(dtype=float)
        ax.plot(xg, mean, color="k", lw=1.2, label=label)
        ax.fill_between(xg, lo, hi, color="k", alpha=0.25, linewidth=0.0)
    except Exception:
        return


def _fit_height_to_square_panels(fig: plt.Figure, axes) -> None:
    width_in = _FIG_WIDTH_IN
    fig.set_size_inches(width_in, width_in, forward=True)
    engine = fig.get_layout_engine()
    for _ in range(5):
        engine.set(rect=(0.0, 0.0, 1.0, 1.0))
        fig.draw_without_rendering()
        renderer = fig.canvas.get_renderer()
        dpi = fig.dpi
        ax_bbs = [ax.get_window_extent(renderer) for ax in axes]
        side_px = max(bb.width for bb in ax_bbs)
        top_extra = 0.0
        bot_extra = 0.0
        for ax, abb in zip(axes, ax_bbs):
            artists = [ax.xaxis.label, ax.yaxis.label, *ax.texts]
            artists.extend(ax.get_xticklabels())
            artists.extend(ax.get_yticklabels())
            for artist in artists:
                if not artist.get_visible() or not str(getattr(artist, "get_text", lambda: "")()).strip():
                    continue
                bb = artist.get_window_extent(renderer)
                top_extra = max(top_extra, bb.y1 - abb.y1)
                bot_extra = max(bot_extra, abb.y0 - bb.y0)
        pad_px = 4.0 / 72.0 * dpi
        target_h_in = (bot_extra + side_px + top_extra + 2.0 * pad_px) / dpi
        top_frac = (top_extra + pad_px) / (target_h_in * dpi)
        top_frac = float(min(max(top_frac, 0.0), 0.3))
        fig.set_size_inches(width_in, target_h_in, forward=True)
        engine.set(rect=(0.0, 0.0, 1.0, 1.0 - top_frac))
    fig.draw_without_rendering()


def make_age_burden_figure(agg: pd.DataFrame) -> plt.Figure:
    required = {"AGE", "n_total", "total_vessel_length"}
    missing = sorted(required - set(agg.columns))
    if missing:
        raise ValueError(f"Missing summary columns: {missing}")

    work = agg.copy()
    work["AGE"] = pd.to_numeric(work["AGE"], errors="coerce")
    work["n_total"] = pd.to_numeric(work["n_total"], errors="coerce")
    work["total_vessel_length"] = pd.to_numeric(work["total_vessel_length"], errors="coerce")
    work = work.dropna(subset=["AGE", "n_total", "total_vessel_length"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(_FIG_WIDTH_IN, _FIG_WIDTH_IN), constrained_layout=True, sharex=True)
    ax_a, ax_b = axes

    _scatter(ax_a, work["AGE"], work["total_vessel_length"])
    _add_unadjusted_ols_fit(
        ax_a,
        work["AGE"].to_numpy(),
        work["total_vessel_length"].to_numpy(),
        label="Unadjusted OLS fit",
    )
    ax_a.set_ylabel("Total MCA length (mm)")

    _scatter(ax_b, work["AGE"], work["n_total"])
    _add_unadjusted_ols_fit(
        ax_b,
        work["AGE"].to_numpy(),
        work["n_total"].to_numpy(),
        label="Unadjusted OLS fit",
    )
    ax_b.set_ylabel("Number of MCA intervals")

    for ax in (ax_a, ax_b):
        ax.set_xlabel("Age (yr)")
        ax.set_xlim(15, 85)
        ax.set_xticks([20, 30, 40, 50, 60, 70, 80])
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.grid(True)
        ax.set_box_aspect(1)
        ax.tick_params(axis="x", labelbottom=True)

    for ax, label in zip((ax_a, ax_b), ("a", "b")):
        add_panel_label(ax, label, pad_pt=_PANEL_LABEL_PAD_PT)
    _fit_height_to_square_panels(fig, (ax_a, ax_b))
    for ax, label in ((ax_a, "a"), (ax_b, "b")):
        for text in list(ax.texts):
            if text.get_text() == label:
                text.remove()
        add_panel_label(ax, label, pad_pt=_PANEL_LABEL_PAD_PT)
    return fig


def main() -> None:
    csv_path = PROJECT_ROOT / 'data' / 'periodic_wave' / 'IXI_HH_sin_trend_analysis_dataset.csv'
    if not csv_path.is_file():
        csv_path = PROJECT_ROOT / 'data' / 'periodic_wave' / 'IXI_HH_sin_trend_analysis_dataset.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(f"Formal analysis CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    agg = aggregate_subject_level(df)
    n_core = int(pd.to_numeric(agg["n_core_inplane"], errors="coerce").sum())
    print(f"input: {csv_path}")
    print(f"intervals: {len(df)}; participants: {len(agg)}; core_inplane intervals: {n_core}")

    fig = make_age_burden_figure(agg)
    width_in, height_in = fig.get_size_inches()
    print(
        f"Figure size: {width_in * 2.54:.2f} × {height_in * 2.54:.2f} cm "
        f"({width_in:.3f} × {height_in:.3f} in)"
    )

    out_path = OUT_DIR / "Figure_age_related_visibility.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
