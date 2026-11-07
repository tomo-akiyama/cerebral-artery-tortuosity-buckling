#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Age and sex distribution (stacked bars by 10-year age group).

Uses data/participants/IXI_HH_participants.csv.

Figure style follows figures/plot_config.py (same palette as the manuscript figure).

Run from project root:
  python statistics/00_Figure_age_sex_distribution.py
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
                    
# -----------------------------------------------------------------------------
FIG_WIDTH_CM = 9.0
FIG_HEIGHT_CM = 6.0
_CM_TO_INCH = 1.0 / 2.54
FIG_FIGSIZE = (FIG_WIDTH_CM * _CM_TO_INCH, FIG_HEIGHT_CM * _CM_TO_INCH)

AGE_BINS = [20, 30, 40, 50, 60, 70, 100]
AGE_LABELS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
Y_MAX = 40
Y_TICKS = list(range(0, Y_MAX + 1, 5))
BAR_LABEL_FONTSIZE = 6

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "participants" / "IXI_HH_participants.csv"
OUT_DIR = PROJECT_ROOT / "statistics" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "Figure_age_sex_distribution.pdf"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import figures.plot_config  # noqa: E402, F401
from figures.plot_config import (  # noqa: E402
    alpha,
    blue_color,
    orange_color,
    save_figure,
)

def _sex_id_series(df: pd.DataFrame) -> pd.Series:
    """Return SEX_ID (1=male, 2=female), tolerating column-name variants."""
    if "SEX_ID (1=m, 2=f)" in df.columns:
        return pd.to_numeric(df["SEX_ID (1=m, 2=f)"], errors="coerce")
    if "SEX_ID" in df.columns:
        return pd.to_numeric(df["SEX_ID"], errors="coerce")
    if "sex" in df.columns:
        mapped = (
            df["sex"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"male": 1, "m": 1, "female": 2, "f": 2})
        )
        return pd.to_numeric(mapped, errors="coerce")
    raise KeyError("No SEX_ID / sex column found in participants CSV")


def load_counts(path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    if "AGE" not in df.columns:
        raise KeyError(f"AGE column missing in {path}")

    sex_id = _sex_id_series(df)
    age_group = pd.cut(
        df["AGE"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    )
    work = pd.DataFrame({"Age Group": age_group, "SEX_ID": sex_id}).dropna()

    males = (
        work.loc[work["SEX_ID"] == 1, "Age Group"]
        .value_counts(sort=False)
        .reindex(AGE_LABELS, fill_value=0)
    )
    females = (
        work.loc[work["SEX_ID"] == 2, "Age Group"]
        .value_counts(sort=False)
        .reindex(AGE_LABELS, fill_value=0)
    )
    return males, females


def plot_age_sex_distribution(males: pd.Series, females: pd.Series):
    fig, ax = plt.subplots(figsize=FIG_FIGSIZE, constrained_layout=True)

    x = list(range(len(AGE_LABELS)))
    male_vals = [int(males[lab]) for lab in AGE_LABELS]
    female_vals = [int(females[lab]) for lab in AGE_LABELS]

    ax.bar(
        x,
        male_vals,
        color=blue_color,
        label="Males",
        alpha=alpha,
        edgecolor="black",
    )
    ax.bar(
        x,
        female_vals,
        bottom=male_vals,
        color=orange_color,
        label="Females",
        alpha=alpha,
        edgecolor="black",
    )

    for i, (male_count, female_count) in enumerate(zip(male_vals, female_vals)):
        if male_count > 0:
            ax.text(
                i,
                male_count / 2,
                str(male_count),
                ha="center",
                va="center",
                color="black",
                fontsize=BAR_LABEL_FONTSIZE,
                fontweight="bold",
            )
        if female_count > 0:
            ax.text(
                i,
                male_count + female_count / 2,
                str(female_count),
                ha="center",
                va="center",
                color="black",
                fontsize=BAR_LABEL_FONTSIZE,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(AGE_LABELS)
    ax.set_xlabel("Age group (yr)")
    ax.set_ylabel("Number of participants")
    ax.legend()

    ax.set_yticks(Y_TICKS)
    ax.set_ylim(0, Y_MAX)
    ax.set_axisbelow(True)                    
    ax.grid(which="major", axis="x", visible=False)
    ax.grid(which="major", axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis="x", bottom=False, top=False)

    return fig


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing participants CSV: {DATA_PATH}")

    males, females = load_counts(DATA_PATH)
    n_total = int(males.sum() + females.sum())
    print(f"Loaded {DATA_PATH.name}: n={n_total}")
    print("Males by age group:", males.to_dict())
    print("Females by age group:", females.to_dict())

    fig = plot_age_sex_distribution(males, females)
    save_figure(fig, OUT_PDF)
    print(f"Saved: {OUT_PDF}")
    plt.close(fig)


if __name__ == "__main__":
    main()
