#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate paper table outputs: 02_Table_m1_univariable_analysis."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
OUT_TEX = TABLES_ROOT / "Table_m1_univariable.tex"

REQUIRED_COLUMNS = [
    "IXI_ID",
    "side",
    "trunk",
    "m1_euclid",
    "m1_length",
    "m1_ti",
    "branch_pattern",
    "sex",
    "AGE",
    "ethnicity_group",
    "TIV_cuberoot",
]


def latex_escape_text(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------
def fit_ols_cluster(df: pd.DataFrame, formula: str) -> Any:
    model = smf.ols(formula=formula, data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["IXI_ID"]})


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


def build_terms(df: pd.DataFrame) -> Dict[str, str]:
    sex_ref = _match_reference_level(df["sex"], "Female")
    eth_ref = _match_reference_level(df["ethnicity_group"], "White")
    br_ref = _match_reference_level(df["branch_pattern"], "Single")
    if sex_ref is None or eth_ref is None or br_ref is None:
        raise ValueError("Cannot resolve reference levels (Female/White/Single).")

    male = _match_reference_level(df["sex"], "Male")
    if male is None:
        raise ValueError("Cannot resolve sex level (Male).")

    rt = _match_reference_level(df["side"], "rt")
    if rt is None:
        raise ValueError("Cannot resolve side level (rt).")

    asian = _match_reference_level(df["ethnicity_group"], "Asian")
    black_other = _match_reference_level(df["ethnicity_group"], "black/other/unknown")
    if asian is None or black_other is None:
        raise ValueError("Cannot resolve ethnicity levels (Asian, black/other/unknown).")

    codom = _match_reference_level(df["branch_pattern"], "codominant")
    dom = _match_reference_level(df["branch_pattern"], "dominant")
    if codom is None or dom is None:
        raise ValueError("Cannot resolve branch_pattern levels (codominant, dominant).")

    sex_term = f"C(sex, Treatment(reference='{sex_ref}'))[T.{male}]"
    side_term = f"C(side)[T.{rt}]"
    eth_asian_term = f"C(ethnicity_group, Treatment(reference='{eth_ref}'))[T.{asian}]"
    eth_black_other_term = (
        f"C(ethnicity_group, Treatment(reference='{eth_ref}'))[T.{black_other}]"
    )
    br_codom_term = f"C(branch_pattern, Treatment(reference='{br_ref}'))[T.{codom}]"
    br_dom_term = f"C(branch_pattern, Treatment(reference='{br_ref}'))[T.{dom}]"
    return {
        "sex_male": sex_term,
        "side_rt": side_term,
        "eth_asian": eth_asian_term,
        "eth_black_other": eth_black_other_term,
        "br_codom": br_codom_term,
        "br_dom": br_dom_term,
    }


@dataclass(frozen=True)
class UniRowSpec:
    label_main: str
    label_level: str
    rhs: str  # formula RHS
    term: str  # coefficient term to extract
    label_main_is_raw_latex: bool = False


def build_univariable_row_specs(df: pd.DataFrame) -> List[UniRowSpec]:
    t = build_terms(df)
    sex_ref = _match_reference_level(df["sex"], "Female")
    eth_ref = _match_reference_level(df["ethnicity_group"], "White")
    br_ref = _match_reference_level(df["branch_pattern"], "Single")
    assert sex_ref is not None and eth_ref is not None and br_ref is not None

    return [
        UniRowSpec("Age (per year)", "", "AGE", "AGE"),
        UniRowSpec(
            "Sex (vs female)",
            "Male",
            f"C(sex, Treatment(reference='{sex_ref}'))",
            t["sex_male"],
        ),
        UniRowSpec(
            "Ethnic group (vs White)",
            "Asian",
            f"C(ethnicity_group, Treatment(reference='{eth_ref}'))",
            t["eth_asian"],
        ),
        UniRowSpec(
            "",
            "Black/Other/Unknown (combined)",
            f"C(ethnicity_group, Treatment(reference='{eth_ref}'))",
            t["eth_black_other"],
        ),
        UniRowSpec("Side (vs left)", "Right", "C(side)", t["side_rt"]),
        UniRowSpec(
            r"TIV$^{1/3}$ (mm)",
            "",
            "TIV_cuberoot",
            "TIV_cuberoot",
            True,
        ),
        UniRowSpec(
            "Major branch type (vs single-trunk type)",
            "Codominant-trunk type",
            f"C(branch_pattern, Treatment(reference='{br_ref}'))",
            t["br_codom"],
        ),
        UniRowSpec(
            "",
            "Dominant-trunk type",
            f"C(branch_pattern, Treatment(reference='{br_ref}'))",
            t["br_dom"],
        ),
    ]


def fit_univariable_and_extract(
    df: pd.DataFrame, *, outcome: str, spec: UniRowSpec
) -> Dict[str, float]:
    formula = f"{outcome} ~ {spec.rhs}"
    needed = ["IXI_ID", outcome]
                        
    if spec.rhs in df.columns:
        needed.append(spec.rhs)
    work = df.dropna(subset=list(dict.fromkeys(needed))).copy()
    res = fit_ols_cluster(work, formula)
    return extract_coef_ci_p_cluster(res, spec.term)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OutcomeSpec:
    key: str
    title_latex: str
    title_is_raw_latex: bool = True


def build_table_tex(
    *, df: pd.DataFrame, outcomes: Sequence[OutcomeSpec], row_specs: Sequence[UniRowSpec]
) -> str:
    lines: List[str] = []
    lines.append("% Auto-generated by statistics/02_Table_m1_univariable_analysis.py")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r" &  & $\beta$ & \multicolumn{2}{c}{95\% CI} & $p$ \\")
    lines.append(r"\midrule")

    for out in outcomes:
        title = out.title_latex if out.title_is_raw_latex else latex_escape_text(out.title_latex)
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{title}}}}} \\")
        for spec in row_specs:
            d = fit_univariable_and_extract(df, outcome=out.key, spec=spec)
            main = spec.label_main if spec.label_main_is_raw_latex else latex_escape_text(spec.label_main)
            lvl = latex_escape_text(spec.label_level)
            beta_cell = format_beta_cell(d["beta"])
            ci_cell = format_ci_cell(d["ci_low"], d["ci_high"])
            p_cell = format_p_value(d["p"])
            lines.append(f"{main} & {lvl} & {beta_cell} & {ci_cell} & {p_cell} \\\\")
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(DATA_PATH)
    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df_raw[list(dict.fromkeys(REQUIRED_COLUMNS))].copy()

    # rho (T > 0; same rows as former ln(T) block)
    df_rho = df.loc[df["m1_ti"] > 0].copy()
    df_rho["m1_rho"] = df_rho["m1_ti"] - 1.0

    # row specs built from df (levels)
    row_specs = build_univariable_row_specs(df)

    # For ln(T), use df_ln; others use df
    parts: List[str] = []
    parts.append("% Auto-generated by statistics/02_Table_m1_univariable_analysis.py")
    parts.append(r"\begin{tabular}{llcccc}")
    parts.append(r"\toprule")
    parts.append(r" &  & $\beta$ & \multicolumn{2}{c}{95\% CI} & $p$ \\")
    parts.append(r"\midrule")

    # length
    parts.append(r"\multicolumn{6}{l}{\textbf{Centerline length}} \\")
    for spec in row_specs:
        d = fit_univariable_and_extract(df, outcome="m1_length", spec=spec)
        main = spec.label_main if spec.label_main_is_raw_latex else latex_escape_text(spec.label_main)
        lvl = latex_escape_text(spec.label_level)
        parts.append(
            f"{main} & {lvl} & {format_beta_cell(d['beta'])} & {format_ci_cell(d['ci_low'], d['ci_high'])} & {format_p_value(d['p'])} \\\\"
        )
    parts.append(r"\midrule")

    # euclid
    parts.append(r"\multicolumn{6}{l}{\textbf{Euclidean distance}} \\")
    for spec in row_specs:
        d = fit_univariable_and_extract(df, outcome="m1_euclid", spec=spec)
        main = spec.label_main if spec.label_main_is_raw_latex else latex_escape_text(spec.label_main)
        lvl = latex_escape_text(spec.label_level)
        parts.append(
            f"{main} & {lvl} & {format_beta_cell(d['beta'])} & {format_ci_cell(d['ci_low'], d['ci_high'])} & {format_p_value(d['p'])} \\\\"
        )
    parts.append(r"\midrule")

    # rho
    parts.append(r"\multicolumn{6}{l}{\textbf{Excess-length ratio, $\rho$}} \\")
    for spec in row_specs:
        d = fit_univariable_and_extract(df_rho, outcome="m1_rho", spec=spec)
        main = spec.label_main if spec.label_main_is_raw_latex else latex_escape_text(spec.label_main)
        lvl = latex_escape_text(spec.label_level)
        parts.append(
            f"{main} & {lvl} & {format_beta_cell(d['beta'])} & {format_ci_cell(d['ci_low'], d['ci_high'])} & {format_p_value(d['p'])} \\\\"
        )
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")

    tex = "\n".join(parts) + "\n"
    OUT_TEX.write_text(tex, encoding="utf-8")
    print(f"Wrote: {OUT_TEX}")


if __name__ == "__main__":
    main()
