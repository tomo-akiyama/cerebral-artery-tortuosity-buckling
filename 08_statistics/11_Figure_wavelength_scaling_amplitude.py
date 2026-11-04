#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Three-panel scatter for predominantly tangential intervals.

  left:   ln(wavelength/mm) vs ln(radius/mm)
  middle: ln(wavelength/radius) vs age
  right:  ln(amplitude/mm) vs age

Style aligned with statistics/03_Figure_M1_Length_EuclidDistance_rho.py:
  filled blue circles, unadjusted OLS (black) + 95% CI (cluster = IXI_ID),
  bold titles, square panels, no panel letters.
  The left panel also draws the slope = 1 reference.

Run from project root:
  python statistics/11_Figure_wavelength_scaling_amplitude.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from patsy import dmatrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "statistics" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from figure_cohort import load_predominantly_tangential  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figures.plot_config import (  # noqa: E402
    alpha,
    ms,
    blue_color,
    gray_color,
    legend_fontsize,
    save_figure,
)

                                                    
_FIG_WIDTH_IN = 7.0
_FIG_HEIGHT_IN = 2.6
FIG_FIGSIZE = (_FIG_WIDTH_IN, _FIG_HEIGHT_IN)

_OLS_COLOR = "k"
_OLS_SHADE_ALPHA = 0.25

_AXIS_LABEL_FS = 10
_TICK_LABEL_FS = 8
_TITLE_FS = 10


def fit_ols_cluster(df: pd.DataFrame, formula: str):
    model = smf.ols(formula=formula, data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["IXI_ID"]})


def predict_mean_ci(res, pred_df: pd.DataFrame):
    design = dmatrix(res.model.data.design_info, pred_df, return_type="dataframe")
    X = design.to_numpy(dtype=float)
    beta = res.params.to_numpy(dtype=float)
    cov = res.cov_params().to_numpy(dtype=float)
    yhat = X @ beta
    var = np.einsum("ij,jk,ik->i", X, cov, X)
    se = np.sqrt(np.maximum(var, 0.0))
    z = 1.96
    lo = yhat - z * se
    hi = yhat + z * se
    return yhat, lo, hi


def _scatter_open(ax, x, y) -> None:
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


def _overlay_ols_ci(ax, x_grid, yhat, lo, hi) -> None:
    ax.plot(x_grid, yhat, color=_OLS_COLOR, lw=1.2, zorder=3)
    ax.fill_between(
        x_grid,
        lo,
        hi,
        color=_OLS_COLOR,
        alpha=_OLS_SHADE_ALPHA,
        linewidth=0,
        zorder=1,
    )


def _style_panel(ax, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=_TITLE_FS, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=_AXIS_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=_AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=_TICK_LABEL_FS)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(True)
    ax.set_box_aspect(1)


def main() -> None:
    df, csv_path = load_predominantly_tangential(PROJECT_ROOT)
    print(f"input: {csv_path}")
    print(f"N={len(df)}; predominantly tangential")
    print(
        f"Figure size: {FIG_FIGSIZE[0]:.3f} × {FIG_FIGSIZE[1]:.3f} in "
        f"({FIG_FIGSIZE[0] * 2.54:.2f} × {FIG_FIGSIZE[1] * 2.54:.2f} cm)"
    )

    fig, axes = plt.subplots(1, 3, figsize=FIG_FIGSIZE, constrained_layout=True)
    ax_scale, ax_ratio, ax_amp = axes

    # -------------------------------------------------------------------------
    # left: ln(wavelength/mm) vs ln(radius/mm)
    # -------------------------------------------------------------------------
    subset_scale = df.dropna(subset=["log_lambda", "log_r", "IXI_ID"]).copy()
    _scatter_open(ax_scale, subset_scale["log_r"], subset_scale["log_lambda"])
    res_scale = fit_ols_cluster(subset_scale, "log_lambda ~ log_r")
    x_scale = np.linspace(
        float(subset_scale["log_r"].min()),
        float(subset_scale["log_r"].max()),
        241,
    )
    y_scale, lo_scale, hi_scale = predict_mean_ci(
        res_scale, pd.DataFrame({"log_r": x_scale})
    )
    _overlay_ols_ci(ax_scale, x_scale, y_scale, lo_scale, hi_scale)
    ax_scale.plot(
        x_scale,
        float(subset_scale["log_lambda"].mean())
        + 1.0 * (x_scale - float(subset_scale["log_r"].mean())),
        color=gray_color,
        linestyle="--",
        linewidth=1.0,
        label="Slope = 1 (reference)",
        zorder=3,
    )
    ax_scale.plot([], [], color=_OLS_COLOR, lw=1.2, label="Unadjusted OLS fit")
    print(
        f"  [left] log_lambda ~ log_r: slope={float(res_scale.params['log_r']):.4f}, "
        f"p={float(res_scale.pvalues['log_r']):.2e}, N={int(res_scale.nobs)}"
    )
    _style_panel(
        ax_scale,
        title="Wavelength scaling",
        xlabel=r"$\ln(\mathrm{radius}/\mathrm{mm})$",
        ylabel=r"$\ln(\mathrm{wavelength}/\mathrm{mm})$",
    )
    ax_scale.set_xlim(-1.11, 0.2)
    ax_scale.set_ylim(2.25, 4.5)
    ax_scale.legend(
        loc="upper left", fontsize=legend_fontsize, frameon=True, framealpha=0.7
    )

    # -------------------------------------------------------------------------
    # middle: ln(wavelength/radius) vs age
    # -------------------------------------------------------------------------
    subset_ratio = df.dropna(subset=["log_lambda_over_r", "AGE", "IXI_ID"]).copy()
    _scatter_open(ax_ratio, subset_ratio["AGE"], subset_ratio["log_lambda_over_r"])
    res_ratio = fit_ols_cluster(subset_ratio, "log_lambda_over_r ~ AGE")
    x_age = np.linspace(float(subset_ratio["AGE"].min()), float(subset_ratio["AGE"].max()), 241)
    y_ratio, lo_ratio, hi_ratio = predict_mean_ci(res_ratio, pd.DataFrame({"AGE": x_age}))
    _overlay_ols_ci(ax_ratio, x_age, y_ratio, lo_ratio, hi_ratio)
    print(
        f"  [middle] log_lambda_over_r ~ AGE: slope={float(res_ratio.params['AGE']):.6f}, "
        f"p={float(res_ratio.pvalues['AGE']):.2e}, N={int(res_ratio.nobs)}"
    )
    _style_panel(
        ax_ratio,
        title="Wavelength-to-radius ratio",
        xlabel="Age (years)",
        ylabel=r"$\ln(\mathrm{wavelength}/\mathrm{radius})$",
    )
    ax_ratio.set_xlim(15, 85)
    ax_ratio.set_xticks([20, 30, 40, 50, 60, 70, 80])
    ax_ratio.set_ylim(2.5, 5.0)

    # -------------------------------------------------------------------------
    # right: ln(amplitude/mm) vs age
    # -------------------------------------------------------------------------
    subset_amp = df.dropna(subset=["log_amplitude", "AGE", "IXI_ID"]).copy()
    _scatter_open(ax_amp, subset_amp["AGE"], subset_amp["log_amplitude"])
    res_amp = fit_ols_cluster(subset_amp, "log_amplitude ~ AGE")
    x_amp = np.linspace(float(subset_amp["AGE"].min()), float(subset_amp["AGE"].max()), 241)
    y_amp, lo_amp, hi_amp = predict_mean_ci(res_amp, pd.DataFrame({"AGE": x_amp}))
    _overlay_ols_ci(ax_amp, x_amp, y_amp, lo_amp, hi_amp)
    print(
        f"  [right] log_amplitude ~ AGE: slope={float(res_amp.params['AGE']):.6f}, "
        f"p={float(res_amp.pvalues['AGE']):.2e}, N={int(res_amp.nobs)}"
    )
    _style_panel(
        ax_amp,
        title="Amplitude",
        xlabel="Age (years)",
        ylabel=r"$\ln(\mathrm{amplitude}/\mathrm{mm})$",
    )
    ax_amp.set_xlim(15, 85)
    ax_amp.set_xticks([20, 30, 40, 50, 60, 70, 80])
    ax_amp.set_ylim(-2.0, 3.0)

    out_pdf = OUT_DIR / "Figure_wavelength_scaling_amplitude.pdf"
    save_figure(fig, out_pdf)
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
