#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public pipeline module: 02 m1 ols cluster univariate analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from statsmodels.stats.outliers_influence import variance_inflation_factor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "m1" / "IXI_HH_m1_features_long.csv"
OUT_RESULTS = PROJECT_ROOT / "data" / "statistics"

OUT_RESULTS.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "IXI_ID",
    "side",
    "trunk",
    "m1_euclid",
    "m1_length",
    "m1_ti",
    "m1_radius_median",
    "branch_pattern",
    "sex",
    "AGE",
    "ethnicity_group",
    "HEIGHT",
    "WEIGHT",
    "TIV",
    "TIV_cuberoot",
    "bmi",
    "bsa_dubois",
    "vol_abs_WMH",
    "CSF_absolute_volume",
]


def fit_ols_cluster(df: pd.DataFrame, formula: str):
    """OLS with cluster-robust SE by IXI_ID."""
    model = smf.ols(formula=formula, data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["IXI_ID"]})


def coef_table_cluster_robust(res) -> pd.DataFrame:
    """Full coefficient table using cluster-robust covariance."""
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


def _match_reference_level(series: pd.Series, target: str) -> Optional[str]:
    """Return the observed level matching target (case-insensitive)."""
    levels = [str(v) for v in series.dropna().unique()]
    for lv in levels:
        if lv.strip().lower() == target.strip().lower():
            return lv
    return None


def add_univariate_predictor_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "m1_radius_median" in out.columns:
        out["m1_radius_median"] = pd.to_numeric(out["m1_radius_median"], errors="coerce")
    if "vol_abs_WMH" in out.columns and "TIV" in out.columns:
        v = pd.to_numeric(out["vol_abs_WMH"], errors="coerce")
        tiv = pd.to_numeric(out["TIV"], errors="coerce")
        out["vol_rel_WMH"] = v / tiv
    if "CSF_absolute_volume" in out.columns and "TIV" in out.columns:
        v = pd.to_numeric(out["CSF_absolute_volume"], errors="coerce")
        tiv = pd.to_numeric(out["TIV"], errors="coerce")
        out["CSF_relative_volume"] = v / tiv
    return out


def _multivariable_covariate_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
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
    pairs: List[Tuple[str, str]] = []
    for col, term in candidate_terms:
        if col in df.columns:
            pairs.append((col, term))
    return pairs


def build_covariate_terms(df: pd.DataFrame) -> List[str]:
    return [term for _, term in _multivariable_covariate_pairs(df)]


def univariate_predictor_specs(df: pd.DataFrame) -> List[Tuple[str, str]]:
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

    specs: List[Tuple[str, str]] = [
        ("AGE", "AGE"),
        ("sex", sex_term),
        ("ethnicity_group", eth_term),
        ("side", "C(side)"),
        ("TIV_cuberoot", "TIV_cuberoot"),
        ("branch_pattern", branch_term),
        ("m1_radius_median", "m1_radius_median"),
        ("HEIGHT", "HEIGHT"),
        ("bmi", "bmi"),
        ("bsa_dubois", "bsa_dubois"),
    ]
    # Derived columns created by add_univariate_predictor_columns
    if "vol_rel_WMH" in df.columns:
        specs.append(("vol_rel_WMH", "vol_rel_WMH"))
    if "CSF_relative_volume" in df.columns:
        specs.append(("CSF_relative_volume", "CSF_relative_volume"))
    return specs


def run_univariate_cluster_ols(
    df_model: pd.DataFrame,
    outcome_col: str,
    specs: List[Tuple[str, str]],
) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []
    for block_label, rhs in specs:
        formula = f"{outcome_col} ~ {rhs}"
        work = df_model.dropna(subset=["IXI_ID", outcome_col]).copy()
        try:
            res = fit_ols_cluster(work, formula)
        except Exception:
            continue
        tab = coef_table_cluster_robust(res)
        tab = tab.loc[tab["term"] != "Intercept"].copy()
        if tab.empty:
            continue
        tab.insert(0, "predictor_block", block_label)
        tab.insert(0, "outcome", outcome_col)
        tab.insert(0, "model", "univariate_cluster_robust")
        tables.append(tab)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def compute_multicollinearity_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # numeric correlation (vessel-level)
    numeric_cols = [
        "AGE",
        "TIV_cuberoot",
        "m1_radius_median",
        "HEIGHT",
        "bmi",
        "bsa_dubois",
        "vol_rel_WMH",
        "CSF_relative_volume",
    ]
    num_present = [c for c in numeric_cols if c in df.columns]
    df_num = df[num_present].apply(pd.to_numeric, errors="coerce")
    corr = df_num.corr(method="pearson", min_periods=3)
    corr_long = (
        corr.stack(dropna=False)
        .rename("corr")
        .reset_index()
        .rename(columns={"level_0": "var1", "level_1": "var2"})
    )

    # VIF on full predictor design matrix (dummy-expanded)
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

    rhs_terms = [
        "AGE",
        sex_term,
        eth_term,
        "C(side)",
        "TIV_cuberoot",
        branch_term,
        "m1_radius_median",
        "HEIGHT",
        "bmi",
        "bsa_dubois",
        "vol_rel_WMH",
        "CSF_relative_volume",
    ]
    rhs = " + ".join(rhs_terms)
    work = df.dropna(subset=[c for c in ["IXI_ID"] + num_present if c in df.columns]).copy()
                                                            
                                              
    X = dmatrix(f"1 + {rhs}", work, return_type="dataframe")
                               
    if "Intercept" in X.columns:
        X = X.drop(columns=["Intercept"])

    # drop near-constant columns to avoid singularities
    keep = X.apply(lambda s: np.nanstd(s.to_numpy(dtype=float)) > 1e-12)
    X = X.loc[:, keep]

    vif_rows = []
    X_np = X.to_numpy(dtype=float)
    for i, name in enumerate(X.columns.astype(str).tolist()):
        try:
            v = float(variance_inflation_factor(X_np, i))
        except Exception:
            v = float("nan")
        vif_rows.append({"term": name, "vif": v})
    vif_df = pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)
    return corr_long, vif_df


def main() -> None:
    print(f"Loading: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    use_cols = list(dict.fromkeys(REQUIRED_COLUMNS))
    df = df_raw[use_cols].dropna().copy()
    print(f"Rows after dropna on required columns: {len(df)}")

    df = add_univariate_predictor_columns(df)
    predictor_specs = univariate_predictor_specs(df)
    print(f"Univariate predictor blocks (one-at-a-time): {[b for b, _ in predictor_specs]}")

    # multicollinearity diagnostics
    corr_long, vif_df = compute_multicollinearity_tables(df)
    corr_out_csv = OUT_RESULTS / "m1_univariate_predictors_numeric_correlations.csv"
    vif_out_csv = OUT_RESULTS / "m1_univariate_predictors_vif.csv"
    corr_long.to_csv(corr_out_csv, index=False)
    vif_df.to_csv(vif_out_csv, index=False)
    print(f"\nSaved: {corr_out_csv}")
    print(f"Saved: {vif_out_csv}")
    print("\nTop VIF terms (descending):")
    print(vif_df.head(25).to_string(index=False))

    df_lnti = df.loc[df["m1_ti"] > 0].copy()
    df_lnti["ln_m1_ti"] = np.log(df_lnti["m1_ti"])
    df_lnti["m1_rho"] = df_lnti["m1_ti"] - 1.0
    print(f"Rows for ln(T) and rho (T>0): {len(df_lnti)}")

    uni_parts = [
        run_univariate_cluster_ols(df_lnti, "ln_m1_ti", predictor_specs),
        run_univariate_cluster_ols(df_lnti, "m1_rho", predictor_specs),
        run_univariate_cluster_ols(df, "m1_ti", predictor_specs),
        run_univariate_cluster_ols(df, "m1_length", predictor_specs),
        run_univariate_cluster_ols(df, "m1_euclid", predictor_specs),
    ]
    uni_nonempty = [u for u in uni_parts if not u.empty]
    univariate_df = (
        pd.concat(uni_nonempty, ignore_index=True) if uni_nonempty else pd.DataFrame()
    )

    univar_out_csv = OUT_RESULTS / "m1_univariate_cluster_robust_before_multivariable.csv"
    if not univariate_df.empty:
        univariate_df.to_csv(univar_out_csv, index=False)
        print(f"\nSaved: {univar_out_csv}")
    else:
        print("\nNo univariate results (all fits failed or empty).")

    if not univariate_df.empty:
        print("\nUnivariate OLS + cluster-robust SE (one predictor block at a time)")
        udisp = univariate_df.copy()
        for c in ("beta", "se", "ci_low", "ci_high"):
            udisp[c] = udisp[c].round(6)
        udisp["p"] = udisp["p"].map(lambda x: float(f"{x:.4g}"))
        for oc in ["ln_m1_ti", "m1_rho", "m1_ti", "m1_length", "m1_euclid"]:
            sub = udisp.loc[udisp["outcome"] == oc]
            if sub.empty:
                continue
            print(f"\nOutcome: {oc}")
            print(
                sub[
                    ["predictor_block", "term", "beta", "se", "ci_low", "ci_high", "p"]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()
