#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate paper figure: 11_Figure_amplitude_vs_age_bw_red."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from patsy import dmatrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAT_DIR = Path(__file__).resolve().parents[1]
if str(STAT_DIR) not in sys.path:
    sys.path.insert(0, str(STAT_DIR))
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
FIG_FIGSIZE = ((16.5 / 3.0) * _CM_TO_INCH, 6.0 * _CM_TO_INCH)

_MARKER_FACE = "white"
_MARKER_EDGE = "black"
_OLS_COLOR = red_color
_OLS_SHADE_ALPHA = 0.25

_AXIS_LABEL_FS = 10
_TICK_LABEL_FS = 8
_TITLE_FS = 10
_CLUSTER_COL = "IXI_ID"


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


def main() -> None:
    df, csv_path = load_predominantly_tangential(PROJECT_ROOT)
    print(f"input: {csv_path}")
    df["A_mm"] = pd.to_numeric(df["amp_hat_mm"], errors="coerce")
    df = df.dropna(subset=["AGE", "A_mm", _CLUSTER_COL]).copy()
    df = df.loc[df["A_mm"] > 0].copy()

    print(f"N={len(df)}; predominantly tangential")
    print(
        f"Figure size: {FIG_FIGSIZE[0] * 2.54:.2f} cm (W) × "
        f"{FIG_FIGSIZE[1] * 2.54:.2f} cm (H)"
    )

    fig, ax = plt.subplots(figsize=FIG_FIGSIZE, constrained_layout=True)

    ax.plot(
        df["AGE"],
        df["A_mm"],
        linestyle="None",
        marker="o",
        markerfacecolor=_MARKER_FACE,
        markeredgecolor=_MARKER_EDGE,
        markeredgewidth=0.8,
        markersize=ms,
        alpha=alpha,
        zorder=2,
    )

    res = fit_ols_cluster(df, "A_mm ~ AGE")
    age_grid = np.linspace(float(df["AGE"].min()), float(df["AGE"].max()), 241)
    pred, lo, hi = predict_mean_ci(res, pd.DataFrame({"AGE": age_grid}))
    print(
        f"  A_mm ~ AGE: slope={float(res.params['AGE']):.6f}, "
        f"p={float(res.pvalues['AGE']):.2e}, N={int(res.nobs)}"
    )

    ax.plot(age_grid, pred, color=_OLS_COLOR, lw=1.2, zorder=3)
    ax.fill_between(
        age_grid,
        lo,
        hi,
        color=_OLS_COLOR,
        alpha=_OLS_SHADE_ALPHA,
        linewidth=0,
        zorder=1,
    )

    ax.set_title("Undulation amplitude", fontsize=_TITLE_FS, fontweight="bold")
    ax.set_xlabel("Age (years)", fontsize=_AXIS_LABEL_FS)
    ax.set_ylabel("Amplitude (mm)", fontsize=_AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=_TICK_LABEL_FS)
    ax.tick_params(top=False, right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_box_aspect(1)
    ax.set_xlim(15, 85)
    ax.set_xticks([20, 30, 40, 50, 60, 70, 80])
    ax.set_ylim(0, 8)
    ax.set_yticks([0, 2, 4, 6, 8])

    out_pdf = OUT_DIR / "Figure_amplitude_vs_age_bw_red.pdf"
    save_figure(fig, out_pdf)
    print(f"Saved: {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
