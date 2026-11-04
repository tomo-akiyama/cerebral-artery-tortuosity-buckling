#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figure: Figure_R2_vs_partial_R2_all."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figures.plot_config import configure_interactive_backend  # noqa: E402

configure_interactive_backend()

import matplotlib.pyplot as plt  # noqa: E402

from figures.plot_config import (  # noqa: E402
    blue_color,
    gray_color,
    lightgray_color,
    orange_color,
    red_color,
    save_figure_and_show,
)

DATA_PATH = (
    PROJECT_ROOT / "data" / "periodic_wave" / "IXI_HH_sin_trend_analysis_dataset.csv"
)
OUT_PATH = PROJECT_ROOT / "statistics" / "figures" / "Figure_R2_vs_partial_R2_all.pdf"
FIGSIZE = (4.5, 4.0)

CLASS_STYLES = {
    "Undetectable": (lightgray_color, 0.35, 7),
    "Secondary": (orange_color, 0.45, 8),
    "Core": (blue_color, 0.55, 9),
}


def _true_like(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})


def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
    df["partial_R2"] = pd.to_numeric(df["partial_R2"], errors="coerce")
    plot_df = df.dropna(subset=["R2", "partial_R2"]).copy()

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    for class_name, (color, point_alpha, point_size) in CLASS_STYLES.items():
        subset = plot_df[plot_df["periodicity"] == class_name]
        ax.scatter(
            subset["R2"],
            subset["partial_R2"],
            s=point_size,
            color=color,
            alpha=point_alpha,
            edgecolors="none",
            rasterized=True,
            label=f"{class_name} (N={len(subset):,})",
        )

    if "core_inplane" in plot_df.columns:
        core_inplane = plot_df[_true_like(plot_df["core_inplane"])]
        ax.scatter(
            core_inplane["R2"],
            core_inplane["partial_R2"],
            s=22,
            facecolors="none",
            edgecolors=red_color,
            linewidths=0.7,
            rasterized=True,
            label=f"core_inplane (N={len(core_inplane):,})",
        )

                        
    ax.axvline(0.60, color=gray_color, linestyle="--", linewidth=0.8)
    ax.axhline(0.15, color=gray_color, linestyle="--", linewidth=0.8)
    ax.text(0.605, 0.02, r"Core: $R^2 \geq 0.60$", color=gray_color)
    ax.text(
        0.02,
        0.158,
        r"Core: partial $R^2 \geq 0.15$",
        color=gray_color,
    )

    ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel=r"$R^2$",
        ylabel=r"Partial $R^2$",
    )
    ax.grid(True, color=lightgray_color, alpha=0.2, linewidth=0.5)
    ax.legend(loc="lower right", frameon=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_figure_and_show(fig, OUT_PATH)
    print(f"有効データ: {len(plot_df):,}/{len(df):,}")
    print(f"保存: {OUT_PATH}")


if __name__ == "__main__":
    main()
