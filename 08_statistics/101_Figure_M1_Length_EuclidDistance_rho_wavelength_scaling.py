#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Four-panel row, width 7 inch.

  1–3: M1 centerline length, Euclidean distance, excess-length ratio ρ vs age
  4:   wavelength vs inner diameter 2 r_i on log–log axes (panel d)

Panel titles are omitted. Panel letters a–d use figures.plot_config.add_panel_label.
Y labels: Centerline length (mm), Euclidean distance (mm),
Excess-length ratio ρ, and Wavelength.

Style matches Figure_M1_Length_EuclidDistance_rho, with typography from
figures/plot_config.py (axis labels and tick labels at axes_label_fontsize
and tick_label_fontsize).

Run from project root:
  python statistics/101_Figure_M1_Length_EuclidDistance_rho_wavelength_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, FormatStrFormatter, LogLocator, NullFormatter
import statsmodels.formula.api as smf
from patsy import dmatrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
M1_DATA_PATH = PROJECT_ROOT / "data" / "m1" / "IXI_HH_m1_features_long.csv"
OUT_DIR = PROJECT_ROOT / "statistics" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "statistics") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "statistics"))

import figures.plot_config  # noqa: E402, F401
from figures.plot_config import (  # noqa: E402
    add_panel_label,
    alpha,
    ms,
    blue_color,
    gray_color,
    save_figure,
)
from figure_cohort import load_predominantly_tangential  # noqa: E402

                                     
_FIG_WIDTH_IN = 7.0
_FIG_HEIGHT_IN = 1.92
FIG_FIGSIZE = (_FIG_WIDTH_IN, _FIG_HEIGHT_IN)

_OLS_COLOR = "k"
_OLS_SHADE_ALPHA = 0.25


def fit_ols_cluster(df: pd.DataFrame, formula: str):
    model = smf.ols(formula=formula, data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["IXI_ID"]})


def make_prediction_df(age_grid: np.ndarray, ref_cov: dict) -> pd.DataFrame:
    d = {"AGE": age_grid}
    d.update(ref_cov)
    return pd.DataFrame(d)


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


def _style_panel(ax, *, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(True)
    ax.set_box_aspect(1)


def _centered_log_limits(values: np.ndarray, pad_decades: float = 0.10) -> tuple[float, float]:
    log_values = np.log10(np.asarray(values, dtype=float))
    mid = 0.5 * (float(log_values.min()) + float(log_values.max()))
    half = 0.5 * (float(log_values.max()) - float(log_values.min())) + pad_decades
    return 10.0 ** (mid - half), 10.0 ** (mid + half)


def _log_tick_marks(ax) -> None:
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1.0, 2.0]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2"]))
    ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50]))
    ax.yaxis.set_major_formatter(FixedFormatter(["5", "10", "20", "50"]))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) / 10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) / 10.0))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="major", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2, width=0.6)
    ax.grid(True, which="major", color="0.75", linewidth=0.6, zorder=0)
    ax.grid(True, which="minor", color="0.90", linewidth=0.4, zorder=0)


def _load_m1() -> pd.DataFrame:
    if not M1_DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {M1_DATA_PATH}")
    df_raw = pd.read_csv(M1_DATA_PATH)
    required = [
        "IXI_ID",
        "side",
        "AGE",
        "m1_length",
        "m1_euclid",
        "m1_ti",
        "sex",
        "ethnicity_group",
        "branch_pattern",
        "TIV_cuberoot",
    ]
    core = ["IXI_ID", "side", "AGE", "m1_length", "m1_euclid", "m1_ti"]
    missing_core = [c for c in core if c not in df_raw.columns]
    if missing_core:
        raise ValueError(f"Missing required columns: {missing_core}")
    use_cols = [c for c in required if c in df_raw.columns]
    df = (
        df_raw[use_cols]
        .dropna(subset=["side", "AGE", "m1_length", "m1_euclid", "m1_ti"])
        .copy()
    )
    df = df.loc[df["m1_ti"] > 0].copy()
    df["m1_rho"] = df["m1_ti"] - 1.0
    return df


def main() -> None:
    df = _load_m1()
    wave, wave_csv = load_predominantly_tangential(PROJECT_ROOT)
    print(f"M1 rows: {len(df)}")
    print(f"wavelength input: {wave_csv}")
    print(f"wavelength N={len(wave)}; predominantly tangential")
    print(
        f"Figure size: {FIG_FIGSIZE[0]:.3f} × {FIG_FIGSIZE[1]:.3f} in "
        f"({FIG_FIGSIZE[0] * 2.54:.2f} × {FIG_FIGSIZE[1] * 2.54:.2f} cm)"
    )

    fig, axes = plt.subplots(1, 4, figsize=FIG_FIGSIZE, constrained_layout=True)
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])

    age_grid = np.linspace(df["AGE"].min(), df["AGE"].max(), 241)
    pred_df = make_prediction_df(age_grid, {})

    res_len = fit_ols_cluster(df, "m1_length ~ AGE")
    res_euc = fit_ols_cluster(df, "m1_euclid ~ AGE")
    res_rho = fit_ols_cluster(df, "m1_rho ~ AGE")
    pred_len, lo_len, hi_len = predict_mean_ci(res_len, pred_df)
    pred_euc, lo_euc, hi_euc = predict_mean_ci(res_euc, pred_df)
    pred_rho, lo_rho, hi_rho = predict_mean_ci(res_rho, pred_df)

    panels = [
        ("m1_length", r"Centerline length (mm)"),
        ("m1_euclid", r"Euclidean distance (mm)"),
        ("m1_rho", r"Excess-length ratio, $\rho$"),
    ]
    for ax, (ycol, ylabel) in zip(axes[:3], panels):
        _scatter_open(ax, df["AGE"], df[ycol])
        _style_panel(ax, xlabel="Age (yr)", ylabel=ylabel)
        ax.set_xlim(15, 85)
        ax.set_xticks([20, 30, 40, 50, 60, 70, 80])

    _overlay_ols_ci(axes[0], age_grid, pred_len, lo_len, hi_len)
    axes[0].set_ylim(10, 50)
    axes[0].set_yticks([10, 20, 30, 40, 50])

    _overlay_ols_ci(axes[1], age_grid, pred_euc, lo_euc, hi_euc)
    axes[1].set_ylim(10, 50)
    axes[1].set_yticks([10, 20, 30, 40, 50])

    _overlay_ols_ci(axes[2], age_grid, pred_rho, lo_rho, hi_rho)
    axes[2].set_ylim(0, 1.0)
    axes[2].set_yticks([0, 0.20, 0.40, 0.60, 0.80, 1.00])
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    subset = wave.dropna(
        subset=["log_lambda", "log_r", "radius_median_mm", "lambda_hat_mm", "IXI_ID"]
    ).copy()
    subset = subset.loc[
        (subset["radius_median_mm"] > 0) & (subset["lambda_hat_mm"] > 0)
    ].copy()
    subset["inner_diameter_mm"] = 2.0 * subset["radius_median_mm"]
    subset["log_d"] = np.log(subset["inner_diameter_mm"])
    ax_w = axes[3]
    _scatter_open(ax_w, subset["inner_diameter_mm"], subset["lambda_hat_mm"])
    res_w = fit_ols_cluster(subset, "log_lambda ~ log_d")
    log_d_grid = np.linspace(float(subset["log_d"].min()), float(subset["log_d"].max()), 241)
    y_log, lo_log, hi_log = predict_mean_ci(res_w, pd.DataFrame({"log_d": log_d_grid}))
    d_grid = np.exp(log_d_grid)
    _overlay_ols_ci(ax_w, d_grid, np.exp(y_log), np.exp(lo_log), np.exp(hi_log))
    slope_one = float(np.exp(subset["log_lambda"].mean() - subset["log_d"].mean()))
    ax_w.plot(d_grid, slope_one * d_grid, color=gray_color, linestyle="--", linewidth=1.0, zorder=3)
    print(
        f"  wavelength log_lambda ~ log(2 r_i): slope={float(res_w.params['log_d']):.4f}, "
        f"p={float(res_w.pvalues['log_d']):.2e}, N={int(res_w.nobs)}"
    )
    ax_w.set_xscale("log")
    ax_w.set_yscale("log")
    _style_panel(
        ax_w,
        xlabel=r"Inner diameter, $2r_{\mathrm{i}}$ (mm)",
        ylabel=r"Wavelength, $\hat{\lambda}$ (mm)",
    )
    x_lo, x_hi = _centered_log_limits(subset["inner_diameter_mm"].to_numpy())
    y_lo, y_hi = _centered_log_limits(subset["lambda_hat_mm"].to_numpy())
    ax_w.set_xlim(x_lo, x_hi)
    ax_w.set_ylim(y_lo, y_hi)
    _log_tick_marks(ax_w)
    print(
        f"  panel d limits: inner diameter [{x_lo:.3f}, {x_hi:.3f}] mm, "
        f"wavelength [{y_lo:.2f}, {y_hi:.2f}] mm"
    )

    for ax, label in zip(axes, ("a", "b", "c", "d")):
        add_panel_label(ax, label)

    out_pdf = OUT_DIR / "Figure_M1_Length_EuclidDistance_rho_wavelength_scaling.pdf"
    save_figure(fig, out_pdf)
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
