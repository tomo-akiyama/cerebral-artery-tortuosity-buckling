#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 11 model validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

STATISTICS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STATISTICS_ROOT.parent
TABLES_ROOT = STATISTICS_ROOT / "tables"
if str(TABLES_ROOT) not in sys.path:
    sys.path.insert(0, str(TABLES_ROOT))
_SIN_TREND_V2 = PROJECT_ROOT / "06_periodic_wave"
if str(_SIN_TREND_V2) not in sys.path:
    sys.path.insert(0, str(_SIN_TREND_V2))

from table_format import format_beta_cell, format_ci_cell, format_p_value  # noqa: E402
from sin_trend_paths import (  # noqa: E402
    FRAME_BASELINE,
    FRAME_REFERENCES,
    analysis_csv_path,
    validate_frame_reference,
    validate_run_id,
)

DATA_PERIODIC_DEFAULT = PROJECT_ROOT / "data" / "periodic_wave" / "IXI_HH_sin_trend_analysis_dataset.csv"
DEFAULT_RUN_ID = "run_154_baseline_v2_r0"
SURFACE_PREDOMINANTLY_TANGENTIAL = "Predominantly tangential"
PERIODICITY_CORE = "Core"
COHORT_TANGENTIAL = "predominantly_tangential"

GEOMETRY_COLUMNS = [
    "log_lambda",
    "log_inner_diameter",
    "log_lambda_over_2r",
    "lambda_over_inner_diameter",
    "lambda_mm",
    "inner_diameter_mm",
    "log_amplitude",
    "AGE",
    "IXI_ID",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Buckling model tests for Predominantly tangential intervals"
    )
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--frame-reference", default=FRAME_BASELINE, choices=list(FRAME_REFERENCES))
    p.add_argument("--analysis-csv", type=Path, default=None, help="Formal analysis CSV from step 33; if omitted, resolve from run-id")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "statistics",
        help="Output path for coefficients CSV",
    )
    p.add_argument(
        "--table-path",
        type=Path,
        default=TABLES_ROOT / "Table_model_validation.tex",
    )
    return p.parse_args(argv)


def _as_bool(series: pd.Series) -> pd.Series:
    def one(x: Any) -> bool:
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return False
        text = str(x).strip().lower()
        if text in {"1", "true", "t", "yes"}:
            return True
        if text in {"0", "false", "f", "no", "", "nan", "none"}:
            return False
        try:
            return bool(int(float(text)))
        except (TypeError, ValueError):
            return False

    return series.map(one)


def _match_level(series: pd.Series, target: str) -> Optional[str]:
    for value in series.dropna().unique():
        if str(value).strip().lower() == target.strip().lower():
            return str(value)
    return None


def _positive_log(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = np.log(x.to_numpy(dtype=float))
    out = np.where(np.isfinite(x.to_numpy(dtype=float)) & (x.to_numpy(dtype=float) > 0), out, np.nan)
    return pd.Series(out, index=series.index)


def prepare_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "lambda_hat_mm" in out.columns:
        lam = pd.to_numeric(out["lambda_hat_mm"], errors="coerce")
    else:
        lam = pd.to_numeric(out["lambda_hat_s_mm"], errors="coerce")
    radius = pd.to_numeric(out["radius_median_mm"], errors="coerce")
    amp = pd.to_numeric(out["amp_comb_mm"], errors="coerce")
    inner_diameter = 2.0 * radius
    out["lambda_mm"] = lam
    out["radius_mm"] = radius
    out["inner_diameter_mm"] = inner_diameter
    out["amplitude_mm"] = amp
    out["log_lambda"] = _positive_log(lam)
    out["log_radius"] = _positive_log(radius)
    out["log_inner_diameter"] = _positive_log(inner_diameter)
    out["log_amplitude"] = _positive_log(amp)
                      
    out["log_lambda_over_r"] = _positive_log(lam / radius)
    out["lambda_over_inner_diameter"] = lam / inner_diameter
    out["log_lambda_over_2r"] = _positive_log(out["lambda_over_inner_diameter"])
    out["AGE"] = pd.to_numeric(out["AGE"], errors="coerce")
    out["TIV_cuberoot"] = pd.to_numeric(out["TIV_cuberoot"], errors="coerce")
    if "mca_segment" in out.columns:
        blank = out["mca_segment"].isna() | out["mca_segment"].astype(str).str.strip().isin(
            {"", "nan", "None", "null", "NULL"}
        )
        out.loc[blank, "mca_segment"] = np.nan
    return out


def _periodic_fit_mask(df: pd.DataFrame) -> pd.Series:
    if "periodicity" not in df.columns:
        raise ValueError("analysis CSV is missing periodicity")
    mask = df["periodicity"].astype(str).eq(PERIODICITY_CORE)
    if "mca_label_conflict" in df.columns:
        mask = mask & ~_as_bool(df["mca_label_conflict"])
    return mask


def select_predominantly_tangential(df: pd.DataFrame) -> pd.DataFrame:
    if "surface_constraint" not in df.columns:
        raise ValueError("analysis CSV is missing surface_constraint")
    mask = _periodic_fit_mask(df) & df["surface_constraint"].astype(str).eq(
        SURFACE_PREDOMINANTLY_TANGENTIAL
    )
    return df.loc[mask].copy()


ADJUSTMENT_COLUMNS = [
    "sex",
    "side",
    "ethnicity_group",
    "TIV_cuberoot",
    "mca_segment",
]


def _require_adjustment_columns(df: pd.DataFrame) -> None:
    missing = [c for c in ADJUSTMENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing covariate columns: {missing}")


def adjustment_reference_levels(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    _require_adjustment_columns(df)
    return {
        "sex": _match_level(df["sex"], "Female"),
        "side": _match_level(df["side"], "lt") or _match_level(df["side"], "left"),
        "ethnicity_group": _match_level(df["ethnicity_group"], "White"),
        "mca_segment": _match_level(df["mca_segment"], "M2-3"),
    }


def covariate_formula(df: pd.DataFrame) -> str:
    """sex, side, ethnic group, TIV^{1/3}, anatomical MCA segment。"""
    refs = adjustment_reference_levels(df)
    sex_ref = refs["sex"]
    eth_ref = refs["ethnicity_group"]
    side_ref = refs["side"]
    mca_ref = refs["mca_segment"]
    sex = (
        f"C(sex, Treatment(reference='{sex_ref}'))" if sex_ref is not None else "C(sex)"
    )
    eth = (
            f"C(ethnicity_group, Treatment(reference='{eth_ref}'))"
            if eth_ref is not None
            else "C(ethnicity_group)"
        )
    side = (
        f"C(side, Treatment(reference='{side_ref}'))" if side_ref is not None else "C(side)"
    )
    mca = (
        f"C(mca_segment, Treatment(reference='{mca_ref}'))"
        if mca_ref is not None
            else "C(mca_segment)"
        )
    return " + ".join([sex, side, eth, "TIV_cuberoot", mca])


def fit_ols_cluster(df: pd.DataFrame, formula: str):
    model = smf.ols(formula=formula, data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["IXI_ID"]})


def wald_p(res, term: str, null_value: float) -> float:
    tested = res.t_test(f"{term} = {null_value}")
    return float(np.asarray(tested.pvalue).squeeze())


def coefficient_rows(res, *, model_id: str, formula: str, n_participants: int) -> List[Dict[str, Any]]:
    ci = res.conf_int()
    rows: List[Dict[str, Any]] = []
    for term in res.params.index.astype(str):
        rows.append(
            {
                "model_id": model_id,
                "formula": formula,
                "term": term,
                "beta": float(res.params[term]),
                "se": float(res.bse[term]),
                "ci_low": float(ci.loc[term, 0]),
                "ci_high": float(ci.loc[term, 1]),
                "p_equals_0": float(res.pvalues[term]),
                "n_intervals": int(res.nobs),
                "n_participants": int(n_participants),
            }
        )
    return rows


def focus_record(
    res,
    *,
    model_id: str,
    section: str,
    model_label: str,
    term: str,
    term_label: str,
    null_value: float,
    n_participants: int,
) -> Dict[str, Any]:
    ci = res.conf_int().loc[term]
    return {
        "section": section,
        "model_id": model_id,
        "model_label": model_label,
        "term": term,
        "term_label": term_label,
        "beta": float(res.params[term]),
        "se": float(res.bse[term]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "null_value": float(null_value),
        "p_null": wald_p(res, term, null_value),
        "p_equals_0": float(res.pvalues[term]),
        "n_intervals": int(res.nobs),
        "n_participants": int(n_participants),
    }


def _complete(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    needed = list(dict.fromkeys(["IXI_ID", *columns]))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns for model: {missing}")
    return df.dropna(subset=needed).copy()


def print_outcomes_and_covariates(df: pd.DataFrame, specs: Sequence[Dict[str, Any]]) -> None:
    n_intervals = len(df)
    n_participants = int(df["IXI_ID"].nunique()) if len(df) else 0
    refs = adjustment_reference_levels(df)
    print(
        f"{SURFACE_PREDOMINANTLY_TANGENTIAL}: "
        f"{n_intervals} intervals, {n_participants} participants"
    )
    print("Adjustment covariates (shared across multivariate models):")
    print(f"  sex (reference: {refs['sex']})")
    print(f"  side (reference: {refs['side']})")
    print(f"  ethnicity_group (reference: {refs['ethnicity_group']})")
    print("  TIV_cuberoot")
    print(f"  mca_segment (reference: {refs['mca_segment']})")
    print("Outcome and covariates per model:")
    for spec in specs:
        outcome = spec["columns"][0]
        covariates = ", ".join(str(name) for name in spec["columns"][1:])
        print(f"{spec['model_id']}")
        print(f"  outcome: {outcome}")
        print(f"  Covariates: {covariates}")


def run_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cov = covariate_formula(df)
    covariates = ADJUSTMENT_COLUMNS
    specs: List[Dict[str, Any]] = [
        {
            "model_id": "A_univariable",
            "section": "Wavelength scaling",
            "model_label": "Univariable",
            "formula": "log_lambda ~ log_inner_diameter",
            "columns": ["log_lambda", "log_inner_diameter"],
            "term": "log_inner_diameter",
            "term_label": r"$\log(2r_{\mathrm{i}})$",
            "null_value": 1.0,                       
        },
        {
            "model_id": "A_multivariable",
            "section": "Wavelength scaling",
            "model_label": "Multivariable",
            "formula": f"log_lambda ~ log_inner_diameter + AGE + {cov}",
            "columns": ["log_lambda", "log_inner_diameter", "AGE", *covariates],
            "term": "log_inner_diameter",
            "term_label": r"$\log(2r_{\mathrm{i}})$",
            "null_value": 1.0,                       
        },
        {
            "model_id": "B_univariable",
            "section": "Wavelength-to-inner-diameter ratio",
            "model_label": "Univariable",
            "formula": "log_lambda_over_2r ~ AGE",
            "columns": ["log_lambda_over_2r", "AGE"],
            "term": "AGE",
            "term_label": "Age (per year)",
            "null_value": 0.0,
        },
        {
            "model_id": "B_multivariable",
            "section": "Wavelength-to-inner-diameter ratio",
            "model_label": "Multivariable",
            "formula": f"log_lambda_over_2r ~ AGE + {cov}",
            "columns": ["log_lambda_over_2r", "AGE", *covariates],
            "term": "AGE",
            "term_label": "Age (per year)",
            "null_value": 0.0,
        },
        {
            "model_id": "D_univariable",
            "section": "Log amplitude",
            "model_label": "Univariable",
            "formula": "log_amplitude ~ AGE",
            "columns": ["log_amplitude", "AGE"],
            "term": "AGE",
            "term_label": "Age (per year)",
            "null_value": 0.0,
        },
        {
            "model_id": "D_multivariable",
            "section": "Log amplitude",
            "model_label": r"Adjusted for $\log\hat{\lambda}$ and $\log(2r_{\mathrm{i}})$",
            "formula": f"log_amplitude ~ AGE + log_lambda + log_inner_diameter + {cov}",
            "columns": ["log_amplitude", "AGE", "log_lambda", "log_inner_diameter", *covariates],
            "term": "AGE",
            "term_label": "Age (per year)",
            "null_value": 0.0,
        },
    ]
    print_outcomes_and_covariates(df, specs)

    full_rows: List[Dict[str, Any]] = []
    focus_rows: List[Dict[str, Any]] = []
    for spec in specs:
        work = _complete(df, spec["columns"])
        if len(work) == 0 or work["IXI_ID"].nunique() < 2:
            raise ValueError(f"{spec['model_id']}: not enough complete observations")
        res = fit_ols_cluster(work, spec["formula"])
        n_participants = int(work["IXI_ID"].nunique())
        full_rows.extend(
            coefficient_rows(
                res,
                model_id=spec["model_id"],
                formula=spec["formula"],
                n_participants=n_participants,
            )
        )
        focus_rows.append(
            focus_record(
                res,
                model_id=spec["model_id"],
                section=spec["section"],
                model_label=spec["model_label"],
                term=spec["term"],
                term_label=spec["term_label"],
                null_value=spec["null_value"],
                n_participants=n_participants,
            )
        )
    return pd.DataFrame(focus_rows), pd.DataFrame(full_rows)


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def ratio_median_iqr(df: pd.DataFrame) -> Tuple[float, float, float, int]:
    values = pd.to_numeric(df["lambda_over_inner_diameter"], errors="coerce").dropna()
    q1, median, q3 = np.percentile(values.to_numpy(dtype=float), [25, 50, 75])
    return float(median), float(q1), float(q3), int(len(values))


def build_latex(
    focus: pd.DataFrame,
    *,
    cohort_title: str,
    n_intervals: int,
    ratio_summary: Tuple[float, float, float, int],
) -> str:
    median, q1, q3, n_ratio = ratio_summary
    section_title = {
        "Wavelength scaling": (
            r"Wavelength scaling, $\log\hat{\lambda} \sim \log(2r_{\mathrm{i}})$ "
            r"(null: $\beta_{\log(2r_{\mathrm{i}})}=0$)"
        ),
        "Wavelength-to-inner-diameter ratio": (
            r"Wavelength-to-inner-diameter ratio, "
            r"$\log(\hat{\lambda}/(2r_{\mathrm{i}})) \sim$ age "
            r"(null: $\beta_{\mathrm{age}}=0$)"
        ),
        "Log amplitude": (
            r"Log amplitude "
            r"(null: $\beta_{\mathrm{age}}=0$)"
        ),
    }
    lines = [
        "% Auto-generated by statistics/11_model_validation.py",
        f"% cohort: {cohort_title}: {n_intervals} intervals",
        "% multivariable adjustment: sex, side, ethnic group, TIV^{1/3}, anatomical MCA segment",
        "% wavelength scaling multivariable also adjusts for age",
        "% amplitude multivariable also includes log wavelength and log inner diameter",
        (
            f"% wavelength-to-inner-diameter ratio median (IQR), N={n_ratio}: "
            f"{median:.6f} ({q1:.6f}--{q3:.6f})"
        ),
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r" &  & $\beta$ & \multicolumn{2}{c}{95\% CI} & $p$ \\",
        r"\midrule",
    ]
    first = True
    for section, block in focus.groupby("section", sort=False):
        if not first:
            lines.append(r"\midrule")
        first = False
        title = section_title.get(str(section), _latex_escape(str(section)))
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{title}}}}} \\")
        if str(section) == "Wavelength-to-inner-diameter ratio":
            lines.append(
                r"\multicolumn{6}{l}{Ratio $\hat{\lambda}/(2r_{\mathrm{i}})$, median (IQR): "
                + rf"\num{{{median:.6f}}} (\num{{{q1:.6f}}}--\num{{{q3:.6f}}})"
                + r"} \\"
            )
        for row in block.itertuples(index=False):
            if "\\" in str(row.model_label) or "$" in str(row.model_label):
                label = str(row.model_label)
            else:
                label = _latex_escape(str(row.model_label))
            term = str(row.term_label) if ("\\" in str(row.term_label) or "$" in str(row.term_label)) else _latex_escape(str(row.term_label))
            p_for_table = row.p_equals_0 if str(section) == "Wavelength scaling" else row.p_null
            lines.append(
                f"{label} & {term} & {format_beta_cell(row.beta)} & "
                f"{format_ci_cell(row.ci_low, row.ci_high)} & {format_p_value(p_for_table)} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_id = validate_run_id(args.run_id)
    frame = validate_frame_reference(args.frame_reference)
    if args.analysis_csv is not None:
        csv_path = Path(args.analysis_csv)
    elif DATA_PERIODIC_DEFAULT.is_file():
        csv_path = DATA_PERIODIC_DEFAULT
    else:
        csv_path = analysis_csv_path(run_id, frame, PROJECT_ROOT)
    if not csv_path.is_file():
        print(f"Error: formal analysis CSV not found: {csv_path}")
        return 1

    raw = pd.read_csv(csv_path, low_memory=False)
    prepared = prepare_analysis_frame(raw)
    print(f"input: {csv_path}")
    print(f"rows in formal CSV: {len(raw)}")

    cohort = select_predominantly_tangential(prepared)
    analytic = _complete(cohort, [*GEOMETRY_COLUMNS, *ADJUSTMENT_COLUMNS])
    if len(analytic) < len(cohort):
        print(
            f"dropped for missing covariates or geometry: {len(cohort) - len(analytic)}"
        )
    ratio_summary = ratio_median_iqr(analytic)
    ratio_median, ratio_q1, ratio_q3, ratio_n = ratio_summary
    print(
        "wavelength-to-inner-diameter ratio "
        f"median={ratio_median:.6f} IQR=[{ratio_q1:.6f}, {ratio_q3:.6f}] N={ratio_n}"
    )
    focus, full = run_models(analytic)
    focus.insert(0, "cohort", COHORT_TANGENTIAL)
    focus.insert(1, "surface_constraint", SURFACE_PREDOMINANTLY_TANGENTIAL)
    full.insert(0, "cohort", COHORT_TANGENTIAL)
    full.insert(1, "surface_constraint", SURFACE_PREDOMINANTLY_TANGENTIAL)
    args.table_path.parent.mkdir(parents=True, exist_ok=True)
    args.table_path.write_text(
        build_latex(
            focus,
            cohort_title=SURFACE_PREDOMINANTLY_TANGENTIAL,
            n_intervals=len(analytic),
            ratio_summary=ratio_summary,
        ),
        encoding="utf-8",
    )
    print(f"table: {args.table_path}")
    for row in focus.itertuples(index=False):
        print(
            f"  {row.model_id}: beta={row.beta:.6f} "
            f"CI=[{row.ci_low:.6f}, {row.ci_high:.6f}] "
            f"p(null={row.null_value:g})={row.p_null:.5g} "
            f"N={row.n_intervals}/{row.n_participants}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    focus_path = args.output_dir / "model_validation_primary_coefficients.csv"
    full_path = args.output_dir / "model_validation_full_coefficients.csv"
    focus.to_csv(focus_path, index=False)
    full.to_csv(full_path, index=False)
    print(f"primary coefficients: {focus_path}")
    print(f"full coefficients: {full_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
