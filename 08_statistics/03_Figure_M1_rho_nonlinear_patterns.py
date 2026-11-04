#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Three-panel figure for non-linear patterns in rho (= m1_ti - 1):
  (a) Covariate-adjusted spline curve (mean + 95% CI)
  (b) Spline-derived age-specific slope d rho / d Age (pointwise 95% CI)
  (c) Cluster bootstrap breakpoint-selection distribution (working OLS AIC; exploratory)

Figure style follows:
  - statistics/03_Figure_M1_Length_EuclidDistance_rho.py
  - figures/plot_config.py

Run from project root:
  python statistics/03_Figure_M1_rho_nonlinear_patterns.py
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
                      
# -----------------------------------------------------------------------------
FIG_FIGSIZE = (7.0, 2.5)  # (width, height)

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from patsy import dmatrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "m1" / "IXI_HH_m1_features_long.csv"
OUT_DIR = PROJECT_ROOT / "statistics" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure project root is importable (so we can import figures/plot_config.py)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import figures.plot_config  # noqa: E402, F401
from figures.plot_config import (  # noqa: E402
    add_panel_label,
    blue_color,
    gray_color,
    save_figure,
)


# Use the same defaults as statistics/03_m1_ols_cluster_multivariable_analysis.py
SPLINE_DF = 2
CANDIDATE_KNOTS = [30, 35, 40, 45, 50, 55]
DO_CLUSTER_BOOTSTRAP = True
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 0
_NOLEGEND = "_nolegend_"


def _remove_axis_legend(ax) -> None:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.legend_ = None


def fit_ols_cluster(df: pd.DataFrame, formula: str):
    model = smf.ols(formula=formula, data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["IXI_ID"]})


def _match_reference_level(series: pd.Series, target: str):
    levels = [str(v) for v in series.dropna().unique()]
    for lv in levels:
        if lv.strip().lower() == target.strip().lower():
            return lv
    return None


def build_covariate_terms(df: pd.DataFrame) -> List[str]:
    sex_ref = _match_reference_level(df["sex"], "Female") if "sex" in df.columns else None
    eth_ref = (
        _match_reference_level(df["ethnicity_group"], "White")
        if "ethnicity_group" in df.columns
        else None
    )
    branch_ref = (
        _match_reference_level(df["branch_pattern"], "Single")
        if "branch_pattern" in df.columns
        else None
    )

    sex_term = f"C(sex, Treatment(reference='{sex_ref}'))" if sex_ref is not None else "C(sex)"
    eth_term = (
        f"C(ethnicity_group, Treatment(reference='{eth_ref}'))"
        if eth_ref is not None
        else "C(ethnicity_group)"
    )
    branch_term = (
        f"C(branch_pattern, Treatment(reference='{branch_ref}'))"
        if branch_ref is not None
        else "C(branch_pattern)"
    )

    candidate_terms = [
        ("sex", sex_term),
        ("side", "C(side)"),
        ("ethnicity_group", eth_term),
        ("branch_pattern", branch_term),
        ("TIV_cuberoot", "TIV_cuberoot"),
    ]
    terms: List[str] = []
    for col, term in candidate_terms:
        if col in df.columns:
            terms.append(term)
    return terms


def reference_row_for_prediction(df: pd.DataFrame) -> Dict[str, object]:
    ref: Dict[str, object] = {}
    for col in ["sex", "side", "ethnicity_group", "branch_pattern"]:
        if col in df.columns:
            ref[col] = df[col].mode(dropna=True).iloc[0]
    if "TIV_cuberoot" in df.columns:
        ref["TIV_cuberoot"] = float(df["TIV_cuberoot"].median())
    return ref


def make_prediction_df(age_grid: np.ndarray, ref_cov: Dict[str, object]) -> pd.DataFrame:
    d: Dict[str, object] = {"AGE": age_grid}
    d.update(ref_cov)
    return pd.DataFrame(d)


def predict_mean_ci(res, pred_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = dmatrix(res.model.data.design_info, pred_df, return_type="dataframe")
    X = design.to_numpy(dtype=float)
    beta = res.params.to_numpy(dtype=float)
    cov = res.cov_params().to_numpy(dtype=float)  # cluster-robust
    yhat = X @ beta
    var = np.einsum("ij,jk,ik->i", X, cov, X)
    se = np.sqrt(np.maximum(var, 0.0))
    z = 1.96
    lo = yhat - z * se
    hi = yhat + z * se
    return yhat, lo, hi


def spline_slope_pointwise_ci(
    res, age_grid: np.ndarray, ref_cov: Dict[str, object], delta: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    d rho / d Age via central difference of the linear predictor.
    Pointwise 95% CI uses cluster-robust covariance.
    """
    pred_a = make_prediction_df(age_grid - delta, ref_cov)
    pred_b = make_prediction_df(age_grid + delta, ref_cov)
    X_a = dmatrix(res.model.data.design_info, pred_a, return_type="dataframe").to_numpy(dtype=float)
    X_b = dmatrix(res.model.data.design_info, pred_b, return_type="dataframe").to_numpy(dtype=float)
    G = (X_b - X_a) / float(2.0 * delta)

    beta = res.params.to_numpy(dtype=float)
    cov = res.cov_params().to_numpy(dtype=float)
    slope = G @ beta
    var = np.einsum("ij,jk,ik->i", G, cov, G)
    se = np.sqrt(np.maximum(var, 0.0))
    lo = slope - 1.96 * se
    hi = slope + 1.96 * se
    return slope, lo, hi


def piecewise(df: pd.DataFrame, breakpoint: float) -> pd.DataFrame:
    out = df.copy()
    out["AGE_post"] = np.maximum(out["AGE"] - breakpoint, 0.0)
    return out


def cluster_bootstrap_breakpoint_selection(
    df_rho: pd.DataFrame,
    *,
    cov_str: str,
    knots: np.ndarray,
    n_boot: int,
    seed: int,
) -> pd.Series:
    rng = np.random.default_rng(int(seed))
    ids = df_rho["IXI_ID"].dropna().unique()
    by_id = {i: df_rho.loc[df_rho["IXI_ID"] == i] for i in ids}

    best = []
    for _ in range(int(n_boot)):
        sampled_ids = rng.choice(ids, size=len(ids), replace=True)
        dboot = pd.concat([by_id[i] for i in sampled_ids], ignore_index=True)
        dboot = dboot.dropna(subset=["IXI_ID", "AGE", "m1_rho"]).copy()

        aics = []
        for k in knots:
            dk = piecewise(dboot, float(k))
            try:
                res = fit_ols_cluster(dk, f"m1_rho ~ AGE + AGE_post + {cov_str}")
                aics.append(float(res.aic))
            except Exception:
                aics.append(float("nan"))
        best_idx = int(np.nanargmin(np.asarray(aics, dtype=float)))
        best.append(float(knots[best_idx]))

    return pd.Series(best, name="best_breakpoint_aic")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {DATA_PATH}")

    df_raw = pd.read_csv(DATA_PATH)
    required = [
        "IXI_ID",
        "AGE",
        "m1_ti",
        "sex",
        "side",
        "ethnicity_group",
        "branch_pattern",
        "TIV_cuberoot",
    ]
    core = ["IXI_ID", "AGE", "m1_ti"]
    missing_core = [c for c in core if c not in df_raw.columns]
    if missing_core:
        raise ValueError(f"Missing required columns: {missing_core}")

    use_cols = [c for c in required if c in df_raw.columns]
    df = df_raw[use_cols].dropna(subset=["IXI_ID", "AGE", "m1_ti"]).copy()
    df = df.loc[df["m1_ti"] > 0].copy()
    df["m1_rho"] = df["m1_ti"] - 1.0

    cov_terms = build_covariate_terms(df)
    cov_str = " + ".join(cov_terms) if cov_terms else "1"
    ref_cov = reference_row_for_prediction(df)

    formula = f"m1_rho ~ cr(AGE, df={int(SPLINE_DF)}, constraints='center') + {cov_str}"
    res_spline = fit_ols_cluster(df, formula)

    age_grid = np.linspace(df["AGE"].min(), df["AGE"].max(), 241)
    pred_df = make_prediction_df(age_grid, ref_cov)
    yhat, ylo, yhi = predict_mean_ci(res_spline, pred_df)

    slope, slope_lo, slope_hi = spline_slope_pointwise_ci(res_spline, age_grid, ref_cov, delta=0.25)

    knots = np.asarray([float(k) for k in CANDIDATE_KNOTS], dtype=float)
    counts = None
    if bool(DO_CLUSTER_BOOTSTRAP):
        best = cluster_bootstrap_breakpoint_selection(
            df,
            cov_str=cov_str,
            knots=knots,
            n_boot=int(BOOTSTRAP_N),
            seed=int(BOOTSTRAP_SEED),
        )
        counts = best.value_counts().sort_index()

    fig, axes = plt.subplots(1, 3, figsize=FIG_FIGSIZE, constrained_layout=True)

    ax = axes[0]
    ax.plot(age_grid, yhat, color=blue_color, label=_NOLEGEND)
    ax.fill_between(
        age_grid,
        ylo,
        yhi,
        color=blue_color,
        alpha=0.15,
        linewidth=0.0,
        label=_NOLEGEND,
    )
    ax.set_xlabel("Age (yr)")
    ax.set_ylabel(r"Predicted $\rho$")
    ax.set_xlim(15, 85)
    ax.set_xticks([20, 30, 40, 50, 60, 70, 80])
    ax.set_box_aspect(1)

    ax = axes[1]
    ax.plot(age_grid, slope, color=blue_color, label=_NOLEGEND)
    ax.fill_between(
        age_grid,
        slope_lo,
        slope_hi,
        color=blue_color,
        alpha=0.15,
        linewidth=0.0,
        label=_NOLEGEND,
    )
    ax.axhline(0.0, color=gray_color, lw=0.8, label=_NOLEGEND)
    ax.set_xlabel("Age (yr)")
                                        
    ax.set_ylabel(r"$d\rho/d\mathrm{Age}$ ($\mathregular{yr^{-1}}$)")
    ax.set_xlim(15, 85)
    ax.set_xticks([20, 30, 40, 50, 60, 70, 80])
    ax.set_box_aspect(1)

    ax = axes[2]
    if counts is not None and len(counts) > 0:
        ax.bar(counts.index.astype(float), counts.values, width=2.8, color=blue_color, alpha=0.85)
        ax.set_xlabel("Selected breakpoint (yr)")
        ax.set_ylabel("Number of bootstrap replicates")
        ax.set_xticks([float(k) for k in knots])
        ax.set_box_aspect(1)
    else:
        ax.text(0.5, 0.5, "Bootstrap disabled", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    for ax, lbl in zip(axes, ["a", "b", "c"]):
        add_panel_label(ax, lbl)

    for ax in axes:
        _remove_axis_legend(ax)

    out_pdf = OUT_DIR / "Figure_M1_rho_nonlinear_patterns.pdf"
    save_figure(fig, out_pdf)
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
