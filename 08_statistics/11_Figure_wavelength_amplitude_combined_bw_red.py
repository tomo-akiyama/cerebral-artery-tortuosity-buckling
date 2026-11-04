#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate paper figure: 11_Figure_wavelength_amplitude_combined_bw_red."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullFormatter
import statsmodels.formula.api as smf
from patsy import dmatrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "statistics" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from figure_cohort import load_predominantly_tangential  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import figures.plot_config  # noqa: E402, F401
from figures.plot_config import (  # noqa: E402
    alpha,
    ms,
    red_color,
    save_figure,
)

_CM_TO_INCH = 1.0 / 2.54
FIG_FIGSIZE = (16.0 * _CM_TO_INCH, 6.0 * _CM_TO_INCH)

_MARKER_FACE = "white"
_MARKER_EDGE = "black"
_OLS_COLOR = red_color
_OLS_SHADE_ALPHA = 0.25

_AXIS_LABEL_FS = 10
_TICK_LABEL_FS = 8
_TITLE_FS = 10
_CLUSTER_COL = "IXI_ID"
_AGE_TICKS = [20, 40, 60, 80]


def fit_ols_cluster(df: pd.DataFrame, formula: str, group_col: str = _CLUSTER_COL):
    model = smf.ols(formula=formula, data=df, missing="drop")
    row_labels = model.data.row_labels
    groups = df.loc[row_labels, group_col]
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


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
        markerfacecolor=_MARKER_FACE,
        markeredgecolor=_MARKER_EDGE,
        markeredgewidth=0.8,
        markersize=ms,
        alpha=alpha,
        zorder=2,
    )


def _style_axes(ax) -> None:
    ax.tick_params(axis="both", labelsize=_TICK_LABEL_FS)
    ax.tick_params(top=False, right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _ols_band(ax, x, yhat, lo, hi) -> None:
    ax.plot(x, yhat, color=_OLS_COLOR, lw=1.2, zorder=3)
    ax.fill_between(
        x,
        lo,
        hi,
        color=_OLS_COLOR,
        alpha=_OLS_SHADE_ALPHA,
        linewidth=0,
        zorder=1,
    )


def _plot_left_wavelength_radius(ax, df: pd.DataFrame) -> None:
    sub_wr = df.dropna(
        subset=["radius_median_mm", "lambda_hat_mm", "log_r", "log_lambda", _CLUSTER_COL]
    ).copy()
    sub_wr = sub_wr.loc[
        (sub_wr["radius_median_mm"] > 0) & (sub_wr["lambda_hat_mm"] > 0)
    ].copy()
    _scatter_open(ax, sub_wr["radius_median_mm"], sub_wr["lambda_hat_mm"])

    res_wr = fit_ols_cluster(sub_wr, "log_lambda ~ log_r")
    log_r_grid = np.linspace(
        float(sub_wr["log_r"].min()), float(sub_wr["log_r"].max()), 241
    )
    y_log, lo_log, hi_log = predict_mean_ci(
        res_wr, pd.DataFrame({"log_r": log_r_grid})
    )
    r_grid = np.exp(log_r_grid)
    _ols_band(ax, r_grid, np.exp(y_log), np.exp(lo_log), np.exp(hi_log))
    print(
        f"  [left] log_lambda ~ log_r: slope={float(res_wr.params['log_r']):.4f}, "
        f"p={float(res_wr.pvalues['log_r']):.2e}, N={int(res_wr.nobs)}"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        "Wavelength vs. lumen radius", fontsize=_TITLE_FS, fontweight="bold"
    )
    ax.set_xlabel(r"Lumen radius, $r_{\mathrm{i}}$ (mm)", fontsize=_AXIS_LABEL_FS)
    ax.set_ylabel(r"Wavelength, $\lambda$ (mm)", fontsize=_AXIS_LABEL_FS)
    _style_axes(ax)
    ax.set_xlim(0.35, 1.2)
    ax.set_ylim(1e1, 1e2)
                                        
    ax.xaxis.set_major_locator(FixedLocator([5e-1, 1e0]))
    ax.xaxis.set_major_formatter(
        FixedFormatter([r"$5\times10^{-1}$", r"$10^{0}$"])
    )
    ax.yaxis.set_major_locator(FixedLocator([1e1, 1e2]))
    ax.yaxis.set_major_formatter(
        FixedFormatter([r"$10^{1}$", r"$10^{2}$"])
    )
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) / 10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) / 10.0))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="major", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2, width=0.6)


def _plot_right_upper_ratio(ax, df: pd.DataFrame, *, y_scale: str) -> None:
    if y_scale not in {"log", "linear"}:
        raise ValueError(f"y_scale must be 'log' or 'linear', got {y_scale!r}")

    sub_ratio = df.dropna(
        subset=["AGE", "lambda_over_r", "log_lambda_over_r", _CLUSTER_COL]
    ).copy()
    sub_ratio = sub_ratio.loc[sub_ratio["lambda_over_r"] > 0].copy()
    _scatter_open(ax, sub_ratio["AGE"], sub_ratio["lambda_over_r"])

    age_grid = np.linspace(
        float(sub_ratio["AGE"].min()), float(sub_ratio["AGE"].max()), 241
    )

    if y_scale == "log":
        res_ratio = fit_ols_cluster(sub_ratio, "log_lambda_over_r ~ AGE")
        y_log_r, lo_log_r, hi_log_r = predict_mean_ci(
            res_ratio, pd.DataFrame({"AGE": age_grid})
        )
        _ols_band(
            ax, age_grid, np.exp(y_log_r), np.exp(lo_log_r), np.exp(hi_log_r)
        )
        print(
            f"  [right-upper/log] log_lambda_over_r ~ AGE: "
            f"slope={float(res_ratio.params['AGE']):.6f}, "
            f"p={float(res_ratio.pvalues['AGE']):.2e}, N={int(res_ratio.nobs)}"
        )
        ax.set_yscale("log")
        ax.set_ylim(10, 150)
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        res_ratio = fit_ols_cluster(sub_ratio, "lambda_over_r ~ AGE")
        yhat, lo, hi = predict_mean_ci(res_ratio, pd.DataFrame({"AGE": age_grid}))
        _ols_band(ax, age_grid, yhat, lo, hi)
        print(
            f"  [right-upper/linear] lambda_over_r ~ AGE: "
            f"slope={float(res_ratio.params['AGE']):.4f}, "
            f"p={float(res_ratio.pvalues['AGE']):.2e}, N={int(res_ratio.nobs)}"
        )
        ax.set_ylim(0, 100)

    ax.set_title(
        r"$\lambda/r_{\mathrm{i}}$ vs. age",
        fontsize=_TITLE_FS,
        fontweight="bold",
        loc="left",
    )
    ax.set_xlabel("Age (years)", fontsize=_AXIS_LABEL_FS)
    ax.set_ylabel(r"$\lambda/r_{\mathrm{i}}$", fontsize=_AXIS_LABEL_FS)
    _style_axes(ax)
    ax.set_xlim(15, 85)
    ax.set_xticks(_AGE_TICKS)


def _plot_right_lower_amplitude(ax, df: pd.DataFrame) -> None:
    sub_amp = df.dropna(subset=["AGE", "amp_hat_mm", _CLUSTER_COL]).copy()
    sub_amp = sub_amp.loc[sub_amp["amp_hat_mm"] > 0].copy()
    _scatter_open(ax, sub_amp["AGE"], sub_amp["amp_hat_mm"])

    res_amp = fit_ols_cluster(sub_amp, "amp_hat_mm ~ AGE")
    age_grid_a = np.linspace(
        float(sub_amp["AGE"].min()), float(sub_amp["AGE"].max()), 241
    )
    y_a, lo_a, hi_a = predict_mean_ci(
        res_amp, pd.DataFrame({"AGE": age_grid_a})
    )
    _ols_band(ax, age_grid_a, y_a, lo_a, hi_a)
    print(
        f"  [right-lower] amp_hat_mm ~ AGE: "
        f"slope={float(res_amp.params['AGE']):.6f}, "
        f"p={float(res_amp.pvalues['AGE']):.2e}, N={int(res_amp.nobs)}"
    )

    ax.set_title(
        "Amplitude vs. age",
        fontsize=_TITLE_FS,
        fontweight="bold",
        loc="left",
    )
    ax.set_xlabel("Age (years)", fontsize=_AXIS_LABEL_FS)
    ax.set_ylabel("Amplitude (mm)", fontsize=_AXIS_LABEL_FS)
    _style_axes(ax)
    ax.set_xlim(15, 85)
    ax.set_xticks(_AGE_TICKS)
    ax.set_ylim(0, 10)
    ax.set_yticks([0, 4, 8])


def _make_combined_figure(df: pd.DataFrame, *, ratio_y_scale: str, out_pdf: Path) -> None:
    fig = plt.figure(figsize=FIG_FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0])
    ax_left = fig.add_subplot(gs[:, 0])
    ax_ru = fig.add_subplot(gs[0, 1])
    ax_rl = fig.add_subplot(gs[1, 1])

    _plot_left_wavelength_radius(ax_left, df)
    _plot_right_upper_ratio(ax_ru, df, y_scale=ratio_y_scale)
    _plot_right_lower_amplitude(ax_rl, df)

    save_figure(fig, out_pdf)
    print(f"Saved: {out_pdf}")
    plt.close(fig)


def main() -> None:
    df, csv_path = load_predominantly_tangential(PROJECT_ROOT)
    print(f"input: {csv_path}")
    df["lambda_over_r"] = df["lambda_hat_mm"] / df["radius_median_mm"]
    df.loc[~(df["radius_median_mm"] > 0), "lambda_over_r"] = np.nan
    print(f"N={len(df)}; predominantly tangential")

    print(
        f"Figure size: {FIG_FIGSIZE[0] * 2.54:.2f} cm (W) × "
        f"{FIG_FIGSIZE[1] * 2.54:.2f} cm (H)"
    )

                          
    _make_combined_figure(
        df,
        ratio_y_scale="log",
        out_pdf=OUT_DIR / "Figure_wavelength_amplitude_combined_bw_red.pdf",
    )
                     
    _make_combined_figure(
        df,
        ratio_y_scale="linear",
        out_pdf=OUT_DIR / "Figure_wavelength_amplitude_combined_bw_red_linear_ratio.pdf",
    )


if __name__ == "__main__":
    main()
