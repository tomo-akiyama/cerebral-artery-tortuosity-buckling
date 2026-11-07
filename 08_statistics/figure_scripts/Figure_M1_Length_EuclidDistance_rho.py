#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Three-panel scatter (M1 Length / Euclid / rho vs Age).

Style (aligned with Figure_M1_Length_EuclidDistance_rho_bw_red):
  - no left/right color split
  - filled blue circles (no marker edge)
  - OLS fit: black line; CI: light gray shade
  - no panel labels / legends inside panels
  - axis limits match bw_red

Run from project root:
  python statistics/03_Figure_M1_Length_EuclidDistance_rho.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import statsmodels.formula.api as smf
from patsy import dmatrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "m1" / "IXI_HH_m1_features_long.csv"
OUT_DIR = PROJECT_ROOT / "statistics" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import figures.plot_config  # noqa: E402, F401
from figures.plot_config import (  # noqa: E402
    alpha,
    ms,
    blue_color,
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


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {DATA_PATH}")

    df_raw = pd.read_csv(DATA_PATH)
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

    print(
        f"Figure size: {FIG_FIGSIZE[0]:.3f} × {FIG_FIGSIZE[1]:.3f} in "
        f"({FIG_FIGSIZE[0] * 2.54:.2f} × {FIG_FIGSIZE[1] * 2.54:.2f} cm)"
    )
    fig, axes = plt.subplots(
        1, 3, figsize=FIG_FIGSIZE, constrained_layout=True, sharex=True
    )

    age_grid = np.linspace(df["AGE"].min(), df["AGE"].max(), 241)
    pred_df = make_prediction_df(age_grid, {})

    res_len = fit_ols_cluster(df, "m1_length ~ AGE")
    res_euc = fit_ols_cluster(df, "m1_euclid ~ AGE")
    res_rho = fit_ols_cluster(df, "m1_rho ~ AGE")

    pred_len, lo_len, hi_len = predict_mean_ci(res_len, pred_df)
    pred_euc, lo_euc, hi_euc = predict_mean_ci(res_euc, pred_df)
    pred_rho, lo_rho, hi_rho = predict_mean_ci(res_rho, pred_df)

    panels = [
        ("m1_length", "Length (mm)", "Centerline length"),
        ("m1_euclid", "Length (mm)", "Euclidean distance"),
        ("m1_rho", r"$\rho$", r"Excess-length ratio, $\rho$"),
    ]

    for ax, (ycol, ylabel, title) in zip(axes, panels):
        _scatter_open(ax, df["AGE"], df[ycol])
        ax.set_title(title, fontsize=_TITLE_FS, fontweight="bold")
        ax.set_xlabel("Age (years)", fontsize=_AXIS_LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=_AXIS_LABEL_FS)
        ax.tick_params(axis="both", labelsize=_TICK_LABEL_FS)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.grid(True)
        ax.set_box_aspect(1)
        ax.set_xlim(15, 85)
        ax.set_xticks([20, 30, 40, 50, 60, 70, 80])

                                              
    axes[0].plot(age_grid, pred_len, color=_OLS_COLOR, lw=1.2, zorder=3)
    axes[0].fill_between(
        age_grid,
        lo_len,
        hi_len,
        color=_OLS_COLOR,
        alpha=_OLS_SHADE_ALPHA,
        linewidth=0,
        zorder=1,
    )
    axes[0].set_ylim(10, 50)
    axes[0].set_yticks([10, 20, 30, 40, 50])

    axes[1].plot(age_grid, pred_euc, color=_OLS_COLOR, lw=1.2, zorder=3)
    axes[1].fill_between(
        age_grid,
        lo_euc,
        hi_euc,
        color=_OLS_COLOR,
        alpha=_OLS_SHADE_ALPHA,
        linewidth=0,
        zorder=1,
    )
    axes[1].set_ylim(10, 50)
    axes[1].set_yticks([10, 20, 30, 40, 50])

    axes[2].plot(age_grid, pred_rho, color=_OLS_COLOR, lw=1.2, zorder=3)
    axes[2].fill_between(
        age_grid,
        lo_rho,
        hi_rho,
        color=_OLS_COLOR,
        alpha=_OLS_SHADE_ALPHA,
        linewidth=0,
        zorder=1,
    )
    axes[2].set_ylim(0, 1.0)
    axes[2].set_yticks([0, 0.20, 0.40, 0.60, 0.80, 1.00])
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    out_pdf = OUT_DIR / "Figure_M1_Length_EuclidDistance_rho.pdf"
    save_figure(fig, out_pdf)
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
