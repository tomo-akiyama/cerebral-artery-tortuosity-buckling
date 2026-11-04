#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate paper table outputs: 03_Table_m1_multivariable_analysis."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

STATISTICS_ROOT = Path(__file__).resolve().parent
TABLES_ROOT = STATISTICS_ROOT / "tables"
if str(TABLES_ROOT) not in sys.path:
    sys.path.insert(0, str(TABLES_ROOT))

from table_format import format_beta_cell, format_ci_cell, format_p_value  # noqa: E402

PROJECT_ROOT = STATISTICS_ROOT.parent
DATA_PATH = PROJECT_ROOT / "data" / "m1" / "IXI_HH_m1_features_long.csv"
OUT_TEX = TABLES_ROOT / "Table_m1_multivariable.tex"

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


def extract_coef_ci_p_cluster(res: Any, term: str) -> Dict[str, float]:
    if term not in res.params.index:
        raise KeyError(f"term not in model: {term!r}")
    ci = res.conf_int().loc[term]
    return {
        "beta": float(res.params[term]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p": float(res.pvalues[term]),
    }


def _match_reference_level(series: pd.Series, target: str) -> Optional[str]:
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


def fit_ols_cluster(df: pd.DataFrame, formula: str) -> Any:
    model = smf.ols(formula=formula, data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["IXI_ID"]})


def build_multivariable_formula(outcome: str, cov_str: str) -> str:
    return f"{outcome} ~ AGE + {cov_str}"


def patsy_term_sex_male(df: pd.DataFrame) -> str:
    sex_ref = _match_reference_level(df["sex"], "Female")
    if sex_ref is None:
        raise ValueError("Cannot resolve sex reference (Female).")
    male = _match_reference_level(df["sex"], "Male")
    if male is None:
        raise ValueError("Cannot resolve sex level (Male).")
    return f"C(sex, Treatment(reference='{sex_ref}'))[T.{male}]"


def patsy_term_side_rt(df: pd.DataFrame) -> str:
    lt = _match_reference_level(df["side"], "lt")
    rt = _match_reference_level(df["side"], "rt")
    if lt is None or rt is None:
        raise ValueError("Expected side levels lt and rt.")
                         
    return f"C(side)[T.{rt}]"


def patsy_term_ethnicity(df: pd.DataFrame, level_target: str) -> str:
    eth_ref = _match_reference_level(df["ethnicity_group"], "White")
    if eth_ref is None:
        raise ValueError("Cannot resolve ethnicity reference (White).")
    lv = _match_reference_level(df["ethnicity_group"], level_target)
    if lv is None:
        raise ValueError(f"Cannot resolve ethnicity level matching {level_target!r}.")
    return f"C(ethnicity_group, Treatment(reference='{eth_ref}'))[T.{lv}]"


def patsy_term_branch(df: pd.DataFrame, level: str) -> str:
    br_ref = _match_reference_level(df["branch_pattern"], "Single")
    if br_ref is None:
        raise ValueError("Cannot resolve branch_pattern reference (Single).")
    lv = _match_reference_level(df["branch_pattern"], level)
    if lv is None:
        raise ValueError(f"Cannot resolve branch_pattern level {level!r}.")
    return f"C(branch_pattern, Treatment(reference='{br_ref}'))[T.{lv}]"


# ---------------------------------------------------------------------------
                 
# ---------------------------------------------------------------------------
def latex_escape_text(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


@dataclass(frozen=True)
class TableRowSpec:
    row_key: str
    term: str
    label_main: str              
    label_level: str
    label_main_is_raw_latex: bool = False


def build_row_specs_from_df(df: pd.DataFrame) -> List[TableRowSpec]:
    return [
        TableRowSpec("AGE", "AGE", "Age (per year)", "", False),
        TableRowSpec(
            "sex_male",
            patsy_term_sex_male(df),
            "Sex (vs female)",
            "Male",
            False,
        ),
        TableRowSpec(
            "ethnicity_asian",
            patsy_term_ethnicity(df, "Asian"),
            "Ethnic group (vs White)",
            "Asian",
            False,
        ),
        TableRowSpec(
            "ethnicity_black_other",
            patsy_term_ethnicity(df, "black/other/unknown"),
            "",
            "Black/Other/Unknown (combined)",
            False,
        ),
        TableRowSpec(
            "side_rt",
            patsy_term_side_rt(df),
            "Side (vs left)",
            "Right",
            False,
        ),
        TableRowSpec(
            "TIV_cuberoot",
            "TIV_cuberoot",
            r"TIV$^{1/3}$ (mm)",
            "",
            True,
        ),
        TableRowSpec(
            "branch_codominant",
            patsy_term_branch(df, "codominant"),
            "Major branch type (vs single-trunk type)",
            "Codominant-trunk type",
            False,
        ),
        TableRowSpec(
            "branch_dominant",
            patsy_term_branch(df, "dominant"),
            "",
            "Dominant-trunk type",
            False,
        ),
    ]


def build_latex_regression_table_booktabs(
    *,
    row_specs: Sequence[TableRowSpec],
    results_by_outcome: Mapping[str, Any],
    outcome_columns: Sequence[Tuple[str, str]],
) -> str:
    lines: List[str] = []
    lines.append("% Auto-generated by statistics/03_Table_m1_multivariable_analysis.py")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r" &  & $\beta$ & \multicolumn{2}{c}{95\% CI} & $p$ \\")
    lines.append(r"\midrule")

    for i, (ok, h) in enumerate(outcome_columns):
        if i > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{h}}}}} \\")
        res = results_by_outcome[ok]
        for spec in row_specs:
            main = (
                spec.label_main
                if spec.label_main_is_raw_latex
                else latex_escape_text(spec.label_main)
            )
            lvl = latex_escape_text(spec.label_level)
            d = extract_coef_ci_p_cluster(res, spec.term)
            lines.append(
                f"{main} & {lvl} & {format_beta_cell(d['beta'])} & "
                f"{format_ci_cell(d['ci_low'], d['ci_high'])} & {format_p_value(d['p'])} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(DATA_PATH)
    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    use_cols = list(dict.fromkeys(REQUIRED_COLUMNS))
    df = df_raw[use_cols].dropna().copy()
    cov_str = " + ".join(build_covariate_terms(df)) if build_covariate_terms(df) else "1"

    df_rho = df.loc[df["m1_ti"] > 0].copy()
    df_rho["m1_rho"] = df_rho["m1_ti"] - 1.0

    res_length = fit_ols_cluster(df, build_multivariable_formula("m1_length", cov_str))
    res_euclid = fit_ols_cluster(df, build_multivariable_formula("m1_euclid", cov_str))
    res_rho = fit_ols_cluster(df_rho, build_multivariable_formula("m1_rho", cov_str))

    row_specs = build_row_specs_from_df(df)
    results_by_outcome = {
        "m1_length": res_length,
        "m1_euclid": res_euclid,
        "m1_rho": res_rho,
    }
    outcome_columns = [
        ("m1_length", "Centerline length"),
        ("m1_euclid", "Euclidean distance"),
        ("m1_rho", r"Excess-length ratio, $\rho$"),
    ]
    tex = build_latex_regression_table_booktabs(
        row_specs=row_specs,
        results_by_outcome=results_by_outcome,
        outcome_columns=outcome_columns,
    )
    OUT_TEX.write_text(tex, encoding="utf-8")
    print(f"Wrote: {OUT_TEX}")


if __name__ == "__main__":
    main()
