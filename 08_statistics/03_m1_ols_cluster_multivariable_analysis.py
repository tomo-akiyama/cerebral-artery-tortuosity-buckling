#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public pipeline module: 03 m1 ols cluster multivariable analysis."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
from patsy import dmatrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "m1" / "IXI_HH_m1_features_long.csv"
OUT_RESULTS = PROJECT_ROOT / "statistics" / "results"
OUT_FIGURES = PROJECT_ROOT / "statistics" / "figures"

OUT_RESULTS.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

# =============================================================================
      
# =============================================================================
                               
ALWAYS_SHOW_WINDOWS = True

                                             
SPLINE_DF = 3

                                               
CANDIDATE_KNOTS = [30, 35, 40, 45, 50, 55]

                                             
DO_CLUSTER_BOOTSTRAP = True
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 0

REQUIRED_COLUMNS = [
    "IXI_ID",
    "side",
    "trunk",
    "m1_euclid",
    "m1_length",
    "m1_ti",
    "branch_pattern",
    "bifurcation_location",
    "sex",
    "AGE",
    "ethnicity_group",
    "HEIGHT",
    "WEIGHT",
    "TIV_cuberoot",
]


def fit_ols_cluster(df: pd.DataFrame, formula: str):
    """OLS with cluster-robust SE by IXI_ID."""
    model = smf.ols(formula=formula, data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["IXI_ID"]})


def summarize_model(res, term: str) -> Dict[str, float]:
    """Extract coefficient summary with robust SE CI."""
    ci = res.conf_int().loc[term]
    return {
        "term": term,
        "beta": float(res.params[term]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p": float(res.pvalues[term]),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "llf": float(res.llf),
        "nobs": float(res.nobs),
    }


def _match_reference_level(series: pd.Series, target: str) -> Optional[str]:
    """Return the observed level matching target (case-insensitive)."""
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

    sex_term = (
        f"C(sex, Treatment(reference='{sex_ref}'))" if sex_ref is not None else "C(sex)"
    )
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


def piecewise(df: pd.DataFrame, breakpoint: float) -> pd.DataFrame:
    out = df.copy()
    out["AGE_post"] = np.maximum(out["AGE"] - breakpoint, 0.0)
    return out


def wald_test_zero(res, term: str) -> Tuple[float, float]:
    """Wald test H0: term = 0."""
    wt = res.t_test(f"{term} = 0")
    stat = float(np.asarray(wt.tvalue).squeeze() ** 2)  # Wald chi2(1) ~= t^2
    p = float(np.asarray(wt.pvalue).squeeze())
    return stat, p


def candidate_breakpoint_piecewise_table(
    df: pd.DataFrame,
    outcome: str,
    *,
    candidate_knots: List[float],
    cov_str: str,
) -> pd.DataFrame:
    """
    Continuous piecewise-linear models at fixed candidate ages.

    Model: outcome ~ AGE + max(0, AGE - k) + covariates
    Inference: cluster-robust SE by IXI_ID.
    Wald p for AGE_post is exploratory (no multiplicity adjustment across k).
    Numeric values are returned unrounded for CSV storage.
    """
    knots = [float(k) for k in candidate_knots]
    rows: List[Dict[str, object]] = []
    for bp in knots:
        lt_mask = df["AGE"] < float(bp)
        ge_mask = ~lt_mask
        n_lt = int(lt_mask.sum())
        n_ge = int(ge_mask.sum())
        p_lt = int(df.loc[lt_mask, "IXI_ID"].nunique())
        p_ge = int(df.loc[ge_mask, "IXI_ID"].nunique())
        counts_str = f"{n_lt}/{n_ge} ({p_lt}/{p_ge})"

        dpk = piecewise(df, float(bp))
        res_pw = fit_ols_cluster(dpk, f"{outcome} ~ AGE + AGE_post + {cov_str}")
        b_pre = float(res_pw.params["AGE"])
        b_change = float(res_pw.params["AGE_post"])
        b_post = b_pre + b_change
        wald_stat, wald_p = wald_test_zero(res_pw, "AGE_post")
        rows.append(
            {
                "breakpoint": float(bp),
                "pre_breakpoint_slope_beta1": b_pre,
                "slope_change_beta2": b_change,
                "post_breakpoint_slope_beta1_plus_beta2": b_post,
                "wald_chi2_beta2_unadjusted_exploratory": wald_stat,
                "wald_p_beta2_unadjusted_exploratory": wald_p,
                "working_ols_aic": float(res_pw.aic),
                "n_<breakpoint>/n_>=breakpoint (participants)": counts_str,
            }
        )
    return pd.DataFrame(rows).sort_values("breakpoint").reset_index(drop=True)


def summarize_inference(res, term: str) -> Dict[str, float]:
    """
    Inference-facing summary (no AIC/BIC/llf).
    Note: AIC/BIC/llf are intentionally excluded to avoid misinterpretation as
    formal likelihood-based model selection under cluster-robust inference.
    """
    ci = res.conf_int().loc[term]
    return {
        "term": term,
        "beta": float(res.params[term]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p": float(res.pvalues[term]),
        "nobs": float(res.nobs),
    }


def coef_table_cluster_robust(res) -> pd.DataFrame:
    """
    Full coefficient table using cluster-robust covariance.
    (No AIC/BIC/llf to avoid model-selection misinterpretation.)
    """
    params = res.params.copy()
    ci = res.conf_int()
    out = pd.DataFrame(
        {
            "term": params.index.astype(str),
            "beta": params.to_numpy(dtype=float),
            "se": res.bse.to_numpy(dtype=float),
            "ci_low": ci.iloc[:, 0].to_numpy(dtype=float),
            "ci_high": ci.iloc[:, 1].to_numpy(dtype=float),
            "p": res.pvalues.to_numpy(dtype=float),
        }
    )
    return out


def spline_nonlinearity_wald_p(
    df_lnti: pd.DataFrame, *, cov_str: str, spline_df: int
) -> Tuple[float, int]:
    """
    Test whether the AGE effect is nonlinear (beyond a linear AGE term).

    Strategy:
    - Build a natural spline basis B = cr(AGE, df=spline_df) (no intercept)
    - Residualize B with respect to [1, AGE] to keep only the nonlinear component
    - Fit: ln_m1_ti ~ AGE + B_nl + covariates (cluster-robust)
    - Wald test H0: all B_nl coefficients = 0
    """
    if int(spline_df) < 3:
        raise ValueError("spline_df must be >= 3")

    work = df_lnti.copy()
    work = work.dropna(subset=["IXI_ID", "AGE", "ln_m1_ti"]).copy()

    # spline basis (no intercept)
    B = dmatrix(f"cr(AGE, df={int(spline_df)}) - 1", work, return_type="dataframe")
    Z = np.column_stack([np.ones(len(work), dtype=float), work["AGE"].to_numpy(dtype=float)])

    # residualize each spline column on [1, AGE]
    coef, *_ = np.linalg.lstsq(Z, B.to_numpy(dtype=float), rcond=None)  # (2 x k)
    B_hat = Z @ coef
    R = B.to_numpy(dtype=float) - B_hat

    # drop near-zero columns (numerical rank deficiency)
    keep = np.nanstd(R, axis=0) > 1e-10
    R = R[:, keep]
    df_nl = int(R.shape[1])
    if df_nl == 0:
        return float("nan"), 0

    # add residualized nonlinear basis to dataframe
    nl_cols = []
    for j in range(df_nl):
        col = f"AGE_nl{j+1}"
        work[col] = R[:, j]
        nl_cols.append(col)

    formula = "ln_m1_ti ~ AGE + " + " + ".join(nl_cols) + f" + {cov_str}"
    res = fit_ols_cluster(work, formula)

    constraints = ", ".join([f"{c} = 0" for c in nl_cols])
    wt = res.wald_test(constraints, scalar=True)
    p = float(np.asarray(wt.pvalue).squeeze())
    return p, df_nl


def reference_row_for_prediction(df: pd.DataFrame) -> Dict[str, object]:
    """Reference covariates for adjusted line plotting."""
    ref: Dict[str, object] = {}
    for col in ["sex", "side", "ethnicity_group", "branch_pattern"]:
        if col in df.columns:
            ref[col] = df[col].mode(dropna=True).iloc[0]
    if "TIV_cuberoot" in df.columns:
        ref["TIV_cuberoot"] = float(df["TIV_cuberoot"].median())
    return ref


def make_prediction_df(
    age_grid: np.ndarray, ref_cov: Dict[str, object], breakpoint: Optional[float] = None
) -> pd.DataFrame:
    d = {"AGE": age_grid}
    d.update(ref_cov)
    pred_df = pd.DataFrame(d)
    if breakpoint is not None:
        pred_df["AGE_post"] = np.maximum(pred_df["AGE"] - breakpoint, 0.0)
    return pred_df


def predict_mean_ci(res, pred_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _cluster_bootstrap_breakpoint_selection(
    df_lnti: pd.DataFrame,
    *,
    cov_str: str,
    knots: np.ndarray,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """
    Cluster bootstrap (resample IXI_ID with replacement).
    For each bootstrap sample, fit candidate-breakpoint piecewise models and record the
    breakpoint with the smallest working OLS AIC (exploratory residual-based fit measure).
    """
    rng = np.random.default_rng(int(seed))
    ids = df_lnti["IXI_ID"].dropna().unique()
    if len(ids) == 0:
        raise ValueError("No IXI_ID found for bootstrap.")

    def _piecewise_local(d: pd.DataFrame, breakpoint: float) -> pd.DataFrame:
        out = d.copy()
        out["AGE_post"] = np.maximum(out["AGE"] - breakpoint, 0.0)
        return out

    rows = []
    # Pre-split by cluster id to speed up repeated sampling
    by_id = {i: df_lnti.loc[df_lnti["IXI_ID"] == i] for i in ids}
    for b in range(int(n_boot)):
        sampled_ids = rng.choice(ids, size=len(ids), replace=True)
        dboot = pd.concat([by_id[i] for i in sampled_ids], ignore_index=True)
        # keep only columns needed; dropna for model
        dboot = dboot.dropna(subset=["IXI_ID", "AGE", "ln_m1_ti"]).copy()

        aics = []
        for k in knots:
            dk = _piecewise_local(dboot, float(k))
            try:
                res = fit_ols_cluster(dk, f"ln_m1_ti ~ AGE + AGE_post + {cov_str}")
                aics.append(float(res.aic))
            except Exception:
                aics.append(float("nan"))
        best_idx = int(np.nanargmin(np.asarray(aics, dtype=float)))
        rows.append({"boot": b, "best_breakpoint_aic": float(knots[best_idx])})

    return pd.DataFrame(rows)


def main(
    *,
    spline_df: int,
    candidate_knots: List[float],
    do_bootstrap: bool,
    bootstrap_n: int,
    bootstrap_seed: int,
) -> None:
    print(f"Loading: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)

    print("\nColumns in CSV:")
    for c in df_raw.columns:
        print(f"- {c}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only necessary columns for this analysis
    use_cols = list(dict.fromkeys(REQUIRED_COLUMNS))
    df = df_raw[use_cols].dropna().copy()
    print(f"\nRows after dropna on required columns: {len(df)}")

    covariates = build_covariate_terms(df)
    cov_str = " + ".join(covariates) if covariates else "1"
    print(f"Covariates used: {covariates if covariates else ['(none)']}")

    # ---------------------------------------------------------------------
    # Outcomes: ln(T), T, and rho = T - 1
    # ---------------------------------------------------------------------
    df_lnti = df.loc[df["m1_ti"] > 0].copy()
    df_lnti["ln_m1_ti"] = np.log(df_lnti["m1_ti"])
    print(f"Rows for ln(T) (T>0): {len(df_lnti)}")

    # ρ uses the same complete-case analysis rows as the multivariable sample
    # (currently identical to T>0 for this dataset).
    df_rho = df.copy()
    df_rho["rho"] = df_rho["m1_ti"] - 1.0
    print(
        f"Rows for rho = T - 1: {len(df_rho)} "
        f"(participants={int(df_rho['IXI_ID'].nunique())})"
    )

    # ===============================================================
    # Step 1: linear model for ln(T)
    # ===============================================================
    formula_lnti_lin = f"ln_m1_ti ~ AGE + {cov_str}"
    res_lnti_lin = fit_ols_cluster(df_lnti, formula_lnti_lin)
    sum_lnti_lin = summarize_inference(res_lnti_lin, "AGE")

    # Linear model for T
    formula_ti_lin = f"m1_ti ~ AGE + {cov_str}"
    res_ti_lin = fit_ols_cluster(df, formula_ti_lin)
    sum_ti_lin = summarize_inference(res_ti_lin, "AGE")

    # Also report length and euclid distance linear models (as in the previous code)
    formula_len_lin = f"m1_length ~ AGE + {cov_str}"
    res_len_lin = fit_ols_cluster(df, formula_len_lin)
    sum_len_lin = summarize_inference(res_len_lin, "AGE")

    formula_euclid_lin = f"m1_euclid ~ AGE + {cov_str}"
    res_euclid_lin = fit_ols_cluster(df, formula_euclid_lin)
    sum_euclid_lin = summarize_inference(res_euclid_lin, "AGE")

    # ===============================================================
    # Step 2: spline model for ln(T) + age-specific slope
    # ===============================================================
    # Natural cubic spline via patsy: cr(AGE, df=spline_df)
    # Note: df here is degrees of freedom for spline basis.
    if int(spline_df) < 3:
        raise ValueError("SPLINE_DF must be >= 3")

    formula_lnti_spline = f"ln_m1_ti ~ cr(AGE, df={int(spline_df)}) + {cov_str}"
    res_lnti_spline = fit_ols_cluster(df_lnti, formula_lnti_spline)

    # Prediction grid and reference covariates
    ref_cov = reference_row_for_prediction(df_lnti)
    age_grid = np.linspace(df_lnti["AGE"].min(), df_lnti["AGE"].max(), 241)
    pred_df_spline = make_prediction_df(age_grid, ref_cov)
    yhat_spline, ylo_spline, yhi_spline = predict_mean_ci(res_lnti_spline, pred_df_spline)

    # Age-specific slope d ln(T) / d Age (central difference + robust pointwise CI)
    delta = 0.25  # years
    age_mid = age_grid.copy()
    pred_df_a = make_prediction_df(age_mid - delta, ref_cov)
    pred_df_b = make_prediction_df(age_mid + delta, ref_cov)
    X_a = dmatrix(
        res_lnti_spline.model.data.design_info, pred_df_a, return_type="dataframe"
    ).to_numpy(dtype=float)
    X_b = dmatrix(
        res_lnti_spline.model.data.design_info, pred_df_b, return_type="dataframe"
    ).to_numpy(dtype=float)
    G = (X_b - X_a) / float(2.0 * delta)  # d(eta)/dAge by central difference
    beta = res_lnti_spline.params.to_numpy(dtype=float)
    cov = res_lnti_spline.cov_params().to_numpy(dtype=float)
    slope_hat = G @ beta
    slope_var = np.einsum("ij,jk,ik->i", G, cov, G)
    slope_se = np.sqrt(np.maximum(slope_var, 0.0))
    slope_lo = slope_hat - 1.96 * slope_se
    slope_hi = slope_hat + 1.96 * slope_se

    # ===============================================================
    # Step 3: exploratory candidate-breakpoint piecewise (ln(T))
    # ===============================================================
    knots = np.asarray([float(k) for k in candidate_knots], dtype=float)
    pw_df = candidate_breakpoint_piecewise_table(
        df_lnti,
        "ln_m1_ti",
        candidate_knots=list(knots),
        cov_str=cov_str,
    )

    # ===============================================================
    # Step 3b: exploratory candidate-breakpoint piecewise (rho = T - 1)
    # ===============================================================
    # Same candidate ages / covariates / cluster-robust inference as ln(T).
    # Outcome is estimated as rho directly (not transformed from ln(T) results).
    pw_rho_df = candidate_breakpoint_piecewise_table(
        df_rho,
        "rho",
        candidate_knots=list(knots),
        cov_str=cov_str,
    )

    # --------------------------------------------------
    # Save tabular results
    # --------------------------------------------------
    summary_rows = [
        {"analysis": "lnT_linear_beta_AGE", **sum_lnti_lin},
        {"analysis": "T_linear_beta_AGE", **sum_ti_lin},
        {"analysis": "m1_length_linear_beta_AGE", **sum_len_lin},
        {"analysis": "m1_euclid_linear_beta_AGE", **sum_euclid_lin},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_out_csv = OUT_RESULTS / "m1_T_lnT_linear_spline_piecewise_summary.csv"
    summary_df.to_csv(summary_out_csv, index=False)

    # Full coefficient tables (multivariable covariates; cluster-robust inference)
    coef_tables = []
    coef_tables.append(
        coef_table_cluster_robust(res_lnti_lin).assign(model="lnTI_linear", dataset="TI>0")
    )
    coef_tables.append(
        coef_table_cluster_robust(res_ti_lin).assign(model="TI_linear", dataset="all")
    )
    coef_tables.append(
        coef_table_cluster_robust(res_len_lin).assign(model="length_linear", dataset="all")
    )
    coef_tables.append(
        coef_table_cluster_robust(res_euclid_lin).assign(model="euclid_linear", dataset="all")
    )
    coef_all = pd.concat(coef_tables, ignore_index=True)
    coef_out_csv = OUT_RESULTS / "m1_multivariable_full_coefficients_cluster_robust.csv"
    coef_all.to_csv(coef_out_csv, index=False)

    # Save with new name (breakpoint) and legacy name (knot) for compatibility
    pw_out_csv = OUT_RESULTS / "lnTI_candidate_breakpoint_piecewise_table.csv"
    pw_df.to_csv(pw_out_csv, index=False)
    pw_out_csv_legacy = OUT_RESULTS / "lnTI_candidate_knot_piecewise_table.csv"
    pw_df.rename(columns={"breakpoint": "knot"}).to_csv(pw_out_csv_legacy, index=False)

    # ρ piecewise table for statistics/03_Table_piecewise_models_rho.py
    # Values are stored unrounded; display rounding is applied only in the table script.
    pw_rho_out_csv = OUT_RESULTS / "rho_candidate_breakpoint_piecewise_table.csv"
    pw_rho_df.to_csv(pw_rho_out_csv, index=False)

    # --------------------------------------------------
    # Figures (recommended set)
    #   Fig1: scatter ln(T) vs Age + LOESS (visualization only)
    #   Fig2: adjusted spline curve (predicted ln(T) vs Age)
    #   Fig3: spline-derived age-specific slope d ln(T) / d Age
    # --------------------------------------------------
    figs_to_show: List[plt.Figure] = []

    # Requested scatter plots (length / euclid / T / log(T))
    fig0a, ax0a = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax0a.scatter(df["AGE"], df["m1_length"], s=14, alpha=0.45, color="#1f77b4")
    ax0a.set_xlabel("Age (years)")
    ax0a.set_ylabel("Vessel length (mm)")
    ax0a.set_title("Vessel length vs Age")
    ax0a.grid(alpha=0.3)
    fig0a.savefig(OUT_FIGURES / "scatter_vessel_length_mm_vs_age_years.pdf", bbox_inches="tight")
    figs_to_show.append(fig0a)

    fig0b, ax0b = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax0b.scatter(df["AGE"], df["m1_euclid"], s=14, alpha=0.45, color="#1f77b4")
    ax0b.set_xlabel("Age (years)")
    ax0b.set_ylabel("Euclid distance (mm)")
    ax0b.set_title("Euclid distance vs Age")
    ax0b.grid(alpha=0.3)
    fig0b.savefig(OUT_FIGURES / "scatter_euclid_distance_mm_vs_age_years.pdf", bbox_inches="tight")
    figs_to_show.append(fig0b)

    fig0c, ax0c = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax0c.scatter(df["AGE"], df["m1_ti"], s=14, alpha=0.45, color="#1f77b4")
    ax0c.set_xlabel("Age (years)")
    ax0c.set_ylabel("Tortuosity (T)")
    ax0c.set_title("Tortuosity (T) vs Age")
    ax0c.grid(alpha=0.3)
    fig0c.savefig(OUT_FIGURES / "scatter_tortuosity_index_vs_age_years.pdf", bbox_inches="tight")
    figs_to_show.append(fig0c)

    fig0d, ax0d = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax0d.scatter(df_lnti["AGE"], df_lnti["ln_m1_ti"], s=14, alpha=0.45, color="#1f77b4")
    ax0d.set_xlabel("Age (years)")
    ax0d.set_ylabel("ln(T)")
    ax0d.set_title("ln(T) vs Age")
    ax0d.grid(alpha=0.3)
    fig0d.savefig(OUT_FIGURES / "scatter_log_tortuosity_index_vs_age_years.pdf", bbox_inches="tight")
    figs_to_show.append(fig0d)

    # Figure 1: scatter + LOESS
    fig1, ax1 = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax1.scatter(df_lnti["AGE"], df_lnti["ln_m1_ti"], s=14, alpha=0.45, color="#1f77b4")
    lo = lowess(df_lnti["ln_m1_ti"].to_numpy(), df_lnti["AGE"].to_numpy(), frac=0.35, return_sorted=True)
    ax1.plot(lo[:, 0], lo[:, 1], color="#ff7f0e", lw=2.0, label="LOESS (visualization only)")
    ax1.set_xlabel("Age (years)")
    ax1.set_ylabel("ln(T)")
    ax1.set_title("Scatter of ln(T) vs Age")
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False)
    fig1.savefig(OUT_FIGURES / "fig1_scatter_lnTI_vs_age_loess.pdf", bbox_inches="tight")
    figs_to_show.append(fig1)

    # Figure 2: adjusted spline curve (mean prediction + 95% CI)
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax2.plot(age_grid, yhat_spline, color="#1f77b4", lw=2.2, label="Adjusted spline mean")
    ax2.fill_between(age_grid, ylo_spline, yhi_spline, color="#1f77b4", alpha=0.15, linewidth=0.0, label="95% CI (mean)")
    ax2.set_xlabel("Age (years)")
    ax2.set_ylabel("Predicted ln(T)")
    ax2.set_title(f"Covariate-adjusted age trend (natural spline, df={int(spline_df)})")
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False)
    fig2.savefig(OUT_FIGURES / f"fig2_adjusted_spline_lnTI_df{int(spline_df)}.pdf", bbox_inches="tight")
    figs_to_show.append(fig2)

    # Figure 3: age-specific slope
    fig3, ax3 = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax3.plot(age_mid, slope_hat, color="#2ca02c", lw=2.2, label=r"$d\,\ln(T)/d\,Age$")
    ax3.fill_between(
        age_mid,
        slope_lo,
        slope_hi,
        color="#2ca02c",
        alpha=0.15,
        linewidth=0.0,
        label="pointwise 95% CI",
    )
    ax3.axhline(0.0, color="gray", lw=1.0)
    ax3.set_xlabel("Age (years)")
    ax3.set_ylabel(r"Age-specific slope of ln(T)")
    ax3.set_title("Spline-derived age-specific increase rate")
    ax3.grid(alpha=0.3)
    ax3.legend(frameon=False)
    fig3.savefig(OUT_FIGURES / f"fig3_spline_slope_lnTI_df{int(spline_df)}.pdf", bbox_inches="tight")
    figs_to_show.append(fig3)

    # Bootstrap breakpoint selection distribution (exploratory)
                                                               
    boot_out_csv = None
    best_boot_breakpoint = None
    if bool(do_bootstrap):
        boot_df = _cluster_bootstrap_breakpoint_selection(
            df_lnti,
            cov_str=cov_str,
            knots=knots,
            n_boot=int(bootstrap_n),
            seed=int(bootstrap_seed),
        )
        boot_out_csv = OUT_RESULTS / "lnTI_candidate_breakpoint_cluster_bootstrap.csv"
        boot_df.to_csv(boot_out_csv, index=False)
        # legacy name
        boot_out_csv_legacy = OUT_RESULTS / "lnTI_candidate_knot_cluster_bootstrap.csv"
        boot_df.rename(columns={"best_breakpoint_aic": "best_knot_aic"}).to_csv(
            boot_out_csv_legacy, index=False
        )

        counts = boot_df["best_breakpoint_aic"].value_counts().sort_index()
        if len(counts) > 0:
            # Mode breakpoint; in ties, pick the smallest breakpoint (sort_index already applied)
            max_count = int(counts.max())
            best_boot_breakpoint = float(counts.loc[counts == max_count].index.min())
        fig4, ax4 = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        ax4.bar(counts.index.astype(float), counts.values, width=2.8, color="#9467bd", alpha=0.85)
        ax4.set_xlabel("Best breakpoint (years) by working OLS AIC")
        ax4.set_ylabel("Count")
        ax4.set_title(f"Cluster bootstrap breakpoint selection (n={int(bootstrap_n)})")
        ax4.grid(axis="y", alpha=0.3)
        fig4.savefig(OUT_FIGURES / "fig4_bootstrap_breakpoint_selection_hist.pdf", bbox_inches="tight")
        figs_to_show.append(fig4)

    if ALWAYS_SHOW_WINDOWS and figs_to_show:
        plt.show()
    for f in figs_to_show:
        plt.close(f)

    # --------------------------------------------------
    # Export data for unified plotting (no re-fitting)
    # --------------------------------------------------
    scatter_df = df.copy()
    scatter_df["ln_m1_ti"] = np.where(scatter_df["m1_ti"] > 0, np.log(scatter_df["m1_ti"]), np.nan)
    scatter_out = OUT_RESULTS / "m1_scatter_with_lnTI.csv"
    scatter_df[["IXI_ID", "side", "AGE", "m1_ti", "ln_m1_ti"]].to_csv(scatter_out, index=False)

    pred_df_out = pd.DataFrame(
        {
            "AGE": age_grid.astype(float),
            "lnTI_spline_yhat": yhat_spline.astype(float),
            "lnTI_spline_ci_low": ylo_spline.astype(float),
            "lnTI_spline_ci_high": yhi_spline.astype(float),
            "lnTI_spline_slope": slope_hat.astype(float),
            "lnTI_spline_slope_ci_low": slope_lo.astype(float),
            "lnTI_spline_slope_ci_high": slope_hi.astype(float),
            "spline_df": int(spline_df),
        }
    )
    pred_out = OUT_RESULTS / f"lnTI_spline_predictions_and_slopes_df{int(spline_df)}.csv"
    pred_df_out.to_csv(pred_out, index=False)

    # --------------------------------------------------
    # Print final summary
    # --------------------------------------------------
                                          

    # Show full multivariable coefficient tables (cluster-robust)
    def _print_coef_table(title: str, res, *, drop_intercept: bool = True) -> None:
        t = coef_table_cluster_robust(res)
        if drop_intercept:
            t = t.loc[t["term"] != "Intercept"].copy()
        # compact rounding for terminal readability
        t["beta"] = t["beta"].round(6)
        t["se"] = t["se"].round(6)
        t["ci_low"] = t["ci_low"].round(6)
        t["ci_high"] = t["ci_high"].round(6)
        t["p"] = t["p"].map(lambda x: float(f"{x:.4g}"))
        print(f"\n{title} (all terms; cluster-robust)")
        print(t.to_string(index=False))

    _print_coef_table("M1 length linear model", res_len_lin)
    _print_coef_table("Euclid distance linear model", res_euclid_lin)
    _print_coef_table("T linear model", res_ti_lin)
    _print_coef_table("ln(T) linear model", res_lnti_lin)

    # ln(T) spline model (df=...): term-wise Wald tests
    try:
        wt_terms = res_lnti_spline.wald_test_terms(skip_single=False).table
        idx = [i for i in wt_terms.index.astype(str) if "cr(" in i and "AGE" in i]
        print(f"\nln(T) spline model (df={int(spline_df)}): term-wise Wald tests")
        if idx:
            print(wt_terms.loc[idx].to_string())
        else:
            # Fallback: show full table if no spline row could be detected
            print(wt_terms.to_string())
    except Exception as e:
        print(f"\nln(T) spline model (df={int(spline_df)}): Wald test failed: {e}")
    print("\nExploratory candidate-breakpoint piecewise (lnT):")
    print(
        "  Counts n_<breakpoint / n_>=breakpoint refer to vessel-level observations "
        "(values in parentheses denote unique participants)."
    )
    print(
        pw_df[
            [
                "breakpoint",
                "pre_breakpoint_slope_beta1",
                "slope_change_beta2",
                "post_breakpoint_slope_beta1_plus_beta2",
                "wald_p_beta2_unadjusted_exploratory",
                "n_<breakpoint>/n_>=breakpoint (participants)",
            ]
        ].to_string(index=False)
    )

    print("\nExploratory candidate-breakpoint piecewise (rho = T - 1):")
    print(
        "  Same candidate ages, covariates, and cluster-robust Wald tests as ln(T); "
        "p values are exploratory (no multiplicity adjustment across breakpoints)."
    )
    print(
        pw_rho_df[
            [
                "breakpoint",
                "pre_breakpoint_slope_beta1",
                "slope_change_beta2",
                "post_breakpoint_slope_beta1_plus_beta2",
                "wald_p_beta2_unadjusted_exploratory",
                "n_<breakpoint>/n_>=breakpoint (participants)",
            ]
        ].to_string(index=False)
    )

    # M1 length piecewise at the most frequent bootstrap breakpoint (Fig.4)
    if best_boot_breakpoint is not None and np.isfinite(best_boot_breakpoint):
        dlen = piecewise(df, float(best_boot_breakpoint))
        res_len_pw = fit_ols_cluster(dlen, f"m1_length ~ AGE + AGE_post + {cov_str}")
        b_pre = float(res_len_pw.params["AGE"])
        b_change = float(res_len_pw.params["AGE_post"])
        b_post = b_pre + b_change
        ci = res_len_pw.conf_int()
        _, p_change = wald_test_zero(res_len_pw, "AGE_post")

        lt_mask = df["AGE"] < float(best_boot_breakpoint)
        ge_mask = ~lt_mask
        n_lt = int(lt_mask.sum())
        n_ge = int(ge_mask.sum())
        p_lt = int(df.loc[lt_mask, "IXI_ID"].nunique())
        p_ge = int(df.loc[ge_mask, "IXI_ID"].nunique())

        print(
            f"\nM1 length piecewise using most frequent bootstrap breakpoint (Fig.4): breakpoint = {best_boot_breakpoint:.0f}"
        )
        print(
            "  Counts n_<breakpoint / n_>=breakpoint refer to vessel-level observations "
            f"(participants in parentheses): {n_lt}/{n_ge} ({p_lt}/{p_ge})"
        )
        print(
            f"  pre-breakpoint slope (AGE) = {b_pre:.6f} "
            f"(95% CI [{ci.loc['AGE', 0]:.6f}, {ci.loc['AGE', 1]:.6f}])"
        )
        print(
            f"  slope change (AGE_post) = {b_change:.6f} "
            f"(95% CI [{ci.loc['AGE_post', 0]:.6f}, {ci.loc['AGE_post', 1]:.6f}]), "
            f"Wald p = {p_change:.4g}"
        )
        print(f"  post-breakpoint slope = {b_post:.6f}")
    else:
        print(
            "\nM1 length piecewise at bootstrap-mode breakpoint: not shown (bootstrap disabled or no bootstrap results)."
        )
    print("\nSaved files:")
    print(f"- {summary_out_csv}")
    print(f"- {pw_out_csv}")
    print(f"- {pw_out_csv_legacy}")
    print(f"- {pw_rho_out_csv}")
    print(f"- {coef_out_csv}")
    print(f"- {scatter_out}")
    print(f"- {pred_out}")
    print(f"- {OUT_FIGURES / 'fig1_scatter_lnTI_vs_age_loess.pdf'}")
    print(f"- {OUT_FIGURES / f'fig2_adjusted_spline_lnTI_df{int(spline_df)}.pdf'}")
    print(f"- {OUT_FIGURES / f'fig3_spline_slope_lnTI_df{int(spline_df)}.pdf'}")
    if boot_out_csv is not None:
        print(f"- {boot_out_csv}")
        print(f"- {boot_out_csv_legacy}")
        print(f"- {OUT_FIGURES / 'fig4_bootstrap_breakpoint_selection_hist.pdf'}")
    print("=" * 72)


if __name__ == "__main__":
    main(
        spline_df=int(SPLINE_DF),
        candidate_knots=list(CANDIDATE_KNOTS),
        do_bootstrap=bool(DO_CLUSTER_BOOTSTRAP),
        bootstrap_n=int(BOOTSTRAP_N),
        bootstrap_seed=int(BOOTSTRAP_SEED),
    )

