#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate paper figure: 11_Figure_wavelength_ratio_distributions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAT_DIR = Path(__file__).resolve().parents[1]
if str(STAT_DIR) not in sys.path:
    sys.path.insert(0, str(STAT_DIR))
OUT_DIR = PROJECT_ROOT / "statistics" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "statistics") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "statistics"))

import figures.plot_config  # noqa: E402, F401
from figures.plot_config import alpha, blue_color, gray_color, ms, save_figure  # noqa: E402
from figure_cohort import load_predominantly_tangential  # noqa: E402

_FIG_WIDTH_IN = 7.0
_FIG_HEIGHT_IN = 2.15
FIG_FIGSIZE = (_FIG_WIDTH_IN, _FIG_HEIGHT_IN)

_AXIS_LABEL_FS = 10
_TICK_LABEL_FS = 8

_SEARCH_FLOOR = 1.0                          
_COVERAGE_MIN = 1.5                  


def _style_panel(ax, *, xlabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=_AXIS_LABEL_FS)
    ax.set_ylabel("Number of intervals", fontsize=_AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=_TICK_LABEL_FS)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(True, which="major", color="0.8", linewidth=0.6, zorder=0)
    ax.set_box_aspect(1)


def _hist(ax, values: np.ndarray, bins: np.ndarray) -> None:
    ax.hist(
        values,
        bins=bins,
        color=blue_color,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )


def _reference_line(ax, x: float) -> None:
    ax.axvline(x, color=gray_color, linestyle="--", linewidth=1.0, zorder=3)


def main() -> None:
    df, csv_path = load_predominantly_tangential(PROJECT_ROOT)
    need = ["lambda_hat_mm", "radius_median_mm", "L_base_mm"]
    work = df.dropna(subset=need).copy()
    work = work.loc[
        (work["lambda_hat_mm"] > 0) & (work["radius_median_mm"] > 0) & (work["L_base_mm"] > 0)
    ].copy()
    lam = work["lambda_hat_mm"].to_numpy(dtype=float)
    radius = work["radius_median_mm"].to_numpy(dtype=float)
    ell_0 = work["L_base_mm"].to_numpy(dtype=float)
    ratio_radius = lam / radius
    ratio_floor = lam / (4.0 * radius)
    ratio_coverage = ell_0 / lam

    print(f"input: {csv_path}")
    print(f"N={len(work)}; predominantly tangential")
    print(
        f"  λ/r: min={ratio_radius.min():.3f}, median={np.median(ratio_radius):.3f}, "
        f"max={ratio_radius.max():.3f}"
    )
    print(
        f"  λ/(4r): min={ratio_floor.min():.3f}, median={np.median(ratio_floor):.3f}, "
        f"max={ratio_floor.max():.3f}; "
        f"n within 0.05 of {_SEARCH_FLOOR:g} = {int(np.sum(np.abs(ratio_floor - _SEARCH_FLOOR) <= 0.05))}"
    )
    print(
        f"  ℓ0/λ: min={ratio_coverage.min():.3f}, median={np.median(ratio_coverage):.3f}, "
        f"max={ratio_coverage.max():.3f}; "
        f"n within 0.02 of {_COVERAGE_MIN:g} = "
        f"{int(np.sum(np.abs(ratio_coverage - _COVERAGE_MIN) <= 0.02))}"
    )

    fig, axes = plt.subplots(1, 5, figsize=FIG_FIGSIZE, constrained_layout=True)

    _hist(axes[0], ratio_radius, np.arange(0.0, 80.0, 5.0))
    _style_panel(axes[0], xlabel=r"$\hat{\lambda}/r_{\mathrm{i}}$")
    axes[0].set_xlim(0.0, 75.0)

    _hist(axes[1], ratio_floor, np.arange(1.0, 19.0, 1.0))
    _reference_line(axes[1], _SEARCH_FLOOR)
    _style_panel(axes[1], xlabel=r"$\hat{\lambda}/(4r_{\mathrm{i}})$")
    axes[1].set_xlim(0.5, 18.0)

    coverage_step = 0.0625
    coverage_bins = np.arange(
        _COVERAGE_MIN - coverage_step / 2.0,
        float(ratio_coverage.max()) + coverage_step,
        coverage_step,
    )
    _hist(axes[2], ratio_coverage, coverage_bins)
    _reference_line(axes[2], _COVERAGE_MIN)
    _style_panel(axes[2], xlabel=r"$\ell_0/\hat{\lambda}$")
    axes[2].set_xlim(1.4, 3.2)

    ax_scatter = axes[3]
    ax_scatter.plot(
        ell_0,
        lam,
        linestyle="None",
        marker="o",
        color=blue_color,
        markeredgewidth=0,
        markersize=ms,
        alpha=alpha,
        zorder=2,
    )
    ell_line = np.array([0.0, 80.0])
    ax_scatter.plot(
        ell_line,
        (2.0 / 3.0) * ell_line,
        color=gray_color,
        linestyle="--",
        linewidth=1.0,
        zorder=3,
    )
    ax_scatter.set_xlim(0.0, 80.0)
    ax_scatter.set_ylim(0.0, 55.0)
    ax_scatter.set_xlabel(r"$\ell_0$ (mm)", fontsize=_AXIS_LABEL_FS)
    ax_scatter.set_ylabel(r"$\hat{\lambda}$ (mm)", fontsize=_AXIS_LABEL_FS)
    ax_scatter.tick_params(axis="both", labelsize=_TICK_LABEL_FS)
    ax_scatter.spines["top"].set_visible(True)
    ax_scatter.spines["right"].set_visible(True)
    ax_scatter.grid(True, which="major", color="0.8", linewidth=0.6, zorder=0)
    ax_scatter.set_box_aspect(1)

    ax_radius = axes[4]
    ax_radius.plot(
        radius,
        ell_0,
        linestyle="None",
        marker="o",
        color=blue_color,
        markeredgewidth=0,
        markersize=ms,
        alpha=alpha,
        zorder=2,
    )
    ax_radius.set_xlim(0.2, 1.1)
    ax_radius.set_ylim(0.0, 80.0)
    ax_radius.set_xlabel(r"$r_{\mathrm{i}}$ (mm)", fontsize=_AXIS_LABEL_FS)
    ax_radius.set_ylabel(r"$\ell_0$ (mm)", fontsize=_AXIS_LABEL_FS)
    ax_radius.tick_params(axis="both", labelsize=_TICK_LABEL_FS)
    ax_radius.spines["top"].set_visible(True)
    ax_radius.spines["right"].set_visible(True)
    ax_radius.grid(True, which="major", color="0.8", linewidth=0.6, zorder=0)
    ax_radius.set_box_aspect(1)

    out_pdf = OUT_DIR / "Figure_wavelength_ratio_distributions.pdf"
    save_figure(fig, out_pdf)
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
