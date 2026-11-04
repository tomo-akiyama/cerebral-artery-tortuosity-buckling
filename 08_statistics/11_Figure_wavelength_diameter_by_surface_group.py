#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figure: 11_Figure_wavelength_diameter_by_surface_group."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT / "data" / "periodic_wave" / "IXI_HH_sin_trend_analysis_dataset.csv"
    / "run_154_baseline_v2_r0"
    / "baseline"
    / "IXI_HH_sin_trend_analysis_dataset.csv"
)
OUT_DIR = PROJECT_ROOT / "statistics" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figures.plot_config import (  # noqa: E402
    add_panel_label,
    alpha,
    blue_color,
    gray_color,
    ms,
    orange_color,
    purple_color,
    save_figure,
)

FIG_FIGSIZE = (7.0, 2.4)
                                    
_SCATTER_ALPHA = 0.4

GROUP_ORDER = [
    "Predominantly tangential",
    "Substantial normal component",
    "Unclassified",
]
GROUP_COLORS = {
    "Predominantly tangential": blue_color,
    "Substantial normal component": orange_color,
    "Unclassified": purple_color,
}
GROUP_LABELS = {
    "Predominantly tangential": r"Predominantly tangential",
    "Substantial normal component": r"Substantial normal component",
    "Unclassified": r"Unclassified",
}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def load_core_groups(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    mask = raw["periodicity"].astype(str).eq("Core")
    if "mca_label_conflict" in raw.columns:
        mask = mask & ~_as_bool(raw["mca_label_conflict"])
    df = raw.loc[mask].copy()

    lam = pd.to_numeric(df["lambda_hat_mm"], errors="coerce")
    radius = pd.to_numeric(df["radius_median_mm"], errors="coerce")
    df["inner_diameter_mm"] = 2.0 * radius
    df["lambda_hat_mm"] = lam
    sc = df["surface_constraint"]
    group = np.full(len(df), "Unclassified", dtype=object)
    is_pt = sc.astype(str).eq("Predominantly tangential")
    is_sn = sc.astype(str).eq("Substantial normal component")
    group[is_pt.to_numpy()] = "Predominantly tangential"
    group[is_sn.to_numpy()] = "Substantial normal component"
    df["surface_group"] = group

    ok = (
        np.isfinite(df["inner_diameter_mm"].to_numpy(dtype=float))
        & np.isfinite(df["lambda_hat_mm"].to_numpy(dtype=float))
        & (df["inner_diameter_mm"].to_numpy(dtype=float) > 0)
        & (df["lambda_hat_mm"].to_numpy(dtype=float) > 0)
    )
    return df.loc[ok].copy()


def _log_limits(values: np.ndarray, pad_decades: float = 0.08) -> tuple[float, float]:
    log_values = np.log10(np.asarray(values, dtype=float))
    lo = float(np.nanmin(log_values)) - pad_decades
    hi = float(np.nanmax(log_values)) + pad_decades
    return 10.0**lo, 10.0**hi


def _style_log_scatter(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1.0, 2.0]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2"]))
    ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50]))
    ax.yaxis.set_major_formatter(FixedFormatter(["5", "10", "20", "50"]))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) / 10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) / 10.0))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(True, which="major")
    ax.set_box_aspect(1)


def _scatter_group(ax, x, y, *, color, label, zorder: int) -> None:
    ax.plot(
        x,
        y,
        linestyle="None",
        marker="o",
        color=color,
        markeredgewidth=0,
        markersize=ms,
        alpha=_SCATTER_ALPHA,
        label=label,
        zorder=zorder,
    )


def _box_by_group(ax, df: pd.DataFrame, ycol: str) -> None:
    data = [df.loc[df["surface_group"] == g, ycol].to_numpy(dtype=float) for g in GROUP_ORDER]
    bp = ax.boxplot(
        data,
        tick_labels=[f"n={len(v)}" for v in data],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": gray_color, "linewidth": 0.8},
        capprops={"color": gray_color, "linewidth": 0.8},
        boxprops={"linewidth": 0.8, "edgecolor": gray_color},
    )
    for patch, g in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, (g, vals) in enumerate(zip(GROUP_ORDER, data), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.plot(
            np.full(len(vals), i) + jitter,
            vals,
            linestyle="None",
            marker="o",
            color=GROUP_COLORS[g],
            markeredgewidth=0,
            markersize=max(ms - 1.0, 1.5),
            alpha=min(alpha, 0.35),
            zorder=2,
        )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["PT", "SN", "Uncl."])
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(True, axis="y")
    ax.set_box_aspect(1)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)
    df = load_core_groups(DATA_PATH)
    counts = df["surface_group"].value_counts()
    print(f"input: {DATA_PATH}")
    for g in GROUP_ORDER:
        print(f"  {g}: {int(counts.get(g, 0))}")

    fig, axes = plt.subplots(1, 3, figsize=FIG_FIGSIZE, constrained_layout=True)
    ax_sc, ax_d, ax_l = axes

    # Draw unclassified / SN first so PT stays on top.
    draw_order = [
        "Unclassified",
        "Substantial normal component",
        "Predominantly tangential",
    ]
    for z, g in enumerate(draw_order, start=1):
        sub = df.loc[df["surface_group"] == g]
        n = len(sub)
        _scatter_group(
            ax_sc,
            sub["inner_diameter_mm"],
            sub["lambda_hat_mm"],
            color=GROUP_COLORS[g],
            label=f"{GROUP_LABELS[g]} (n={n})",
            zorder=z,
        )

    x_lo, x_hi = _log_limits(df["inner_diameter_mm"].to_numpy())
    y_lo, y_hi = _log_limits(df["lambda_hat_mm"].to_numpy())
    ax_sc.set_xlim(x_lo, x_hi)
    ax_sc.set_ylim(y_lo, y_hi)
    # slope = 1 guide through geometric mean
    gmean_d = float(np.exp(np.mean(np.log(df["inner_diameter_mm"]))))
    gmean_l = float(np.exp(np.mean(np.log(df["lambda_hat_mm"]))))
    slope_one = gmean_l / gmean_d
    x_line = np.array([x_lo, x_hi], dtype=float)
    ax_sc.plot(x_line, slope_one * x_line, color=gray_color, linestyle="--", linewidth=1.0, zorder=0)
    _style_log_scatter(ax_sc)
    ax_sc.set_xlabel(r"Inner diameter, $2r_{\mathrm{i}}$ (mm)")
    ax_sc.set_ylabel(r"Wavelength, $\hat{\lambda}$ (mm)")
    ax_sc.legend(loc="best", frameon=True, fontsize=6, handlelength=1.0, borderpad=0.3)

    _box_by_group(ax_d, df, "inner_diameter_mm")
    ax_d.set_ylabel(r"Inner diameter, $2r_{\mathrm{i}}$ (mm)")
    ax_d.set_xlabel("Surface group")

    _box_by_group(ax_l, df, "lambda_hat_mm")
    ax_l.set_ylabel(r"Wavelength, $\hat{\lambda}$ (mm)")
    ax_l.set_xlabel("Surface group")

    for ax, label in zip(axes, ("a", "b", "c")):
        add_panel_label(ax, label)

    out_pdf = OUT_DIR / "Figure_wavelength_diameter_by_surface_group.pdf"
    save_figure(fig, out_pdf)
    print(f"Saved: {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
