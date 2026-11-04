#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 10 summarize sin trend."""


# ===== USER SETTINGS =====
                                                                               
COVERAGE_HAT_CORE_MIN_VALUES = ["formal"]
PERIODICITY_LEVELS = ["Core", "NotCore", "InvalidFit", "NotEligible"]
SURFACE_LEVELS = ["Predominantly tangential", "Substantial normal component"]
# =========================

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def combine_count_and_percentage(count_df, pct_df):
    result_df = count_df.copy().astype(object)
    
                     
    def get_value(df, idx, col):
        try:
            val = df.loc[idx, col]
            if isinstance(val, pd.Series):
                if len(val) > 0:
                    return val.iloc[0]
                else:
                    return np.nan
            return val
        except (KeyError, IndexError):
            return np.nan
    
                               
                              
    for idx in pct_df.index:
        for col in pct_df.columns:
            if col == 'Total':
                                                         
                                   
                if 'All' not in count_df.columns:
                    pct_val = get_value(pct_df, idx, col)
                    if pd.notna(pct_val):
                        if idx in result_df.index:
                            if 'Total' not in result_df.columns:
                                result_df['Total'] = ''
                            result_df.loc[idx, 'Total'] = f"({pct_val:.2f}%)"
            else:
                               
                if idx in count_df.index and col in count_df.columns:
                    count_val = get_value(count_df, idx, col)
                    pct_val = get_value(pct_df, idx, col)
                    if pd.notna(count_val) and pd.notna(pct_val):
                        result_df.loc[idx, col] = f"{int(count_val)} ({pct_val:.2f}%)"
                    elif pd.notna(count_val):
                        result_df.loc[idx, col] = f"{int(count_val)}"
                    elif pd.notna(pct_val):
                        result_df.loc[idx, col] = f"({pct_val:.2f}%)"
    
                                
    if 'All' in count_df.index and 'All' in count_df.columns:
        total_all = get_value(count_df, 'All', 'All')
        if pd.notna(total_all) and total_all > 0:
            for col in count_df.columns:
                if col != 'Total' and col != 'All':               
                    count_val = get_value(count_df, 'All', col)
                    if pd.notna(count_val):
                        pct_val = (count_val / total_all) * 100
                        result_df.loc['All', col] = f"{int(count_val)} ({pct_val:.2f}%)"
                elif col == 'All':
                                               
                    count_val = get_value(count_df, 'All', 'All')
                    if pd.notna(count_val):
                        result_df.loc['All', 'All'] = f"{int(count_val)} (100.00%)"
    
                                
    if 'All' in count_df.columns and 'All' in count_df.index:
        total_all = get_value(count_df, 'All', 'All')
        if pd.notna(total_all) and total_all > 0:
            for idx in count_df.index:
                if idx != 'All':           
                    count_val = get_value(count_df, idx, 'All')
                    if pd.notna(count_val):
                        pct_val = (count_val / total_all) * 100
                        result_df.loc[idx, 'All'] = f"{int(count_val)} ({pct_val:.2f}%)"
    
                      
    if 'All' in result_df.columns and 'Total' in result_df.columns:
        result_df = result_df.drop('Total', axis=1)
    
    return result_df


def core_surface_mask(frame: pd.DataFrame, coverage_min: str, surface_level: str):
    col_p = f"periodicity_coverage_{coverage_min}"
    col_s = f"surface_constraint_coverage_{coverage_min}"
    if col_p not in frame.columns or col_s not in frame.columns:
        return None
    return frame[col_p].astype(str).eq("Core") & frame[col_s].astype(str).eq(surface_level)


def align_crosstab_columns(table: pd.DataFrame, preferred: list) -> pd.DataFrame:
    cols = [c for c in preferred if c in table.columns]
    cols.extend(c for c in table.columns if c not in cols and c != "All")
    if "All" in table.columns:
        cols.append("All")
    return table.loc[:, cols]

              
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                                    
sys.path.insert(0, project_root)
from figures.plot_config import (
    alpha as plot_alpha,
    ms as plot_ms,
    blue_color,
    orange_color,
    gray_color,
    lightblue_color,
    text_fontsize,
    text_fontsize_large,
)

                                                       
csv_path = os.path.join(project_root, "data", "periodic_wave", "IXI_HH_sin_trend_analysis_dataset.csv")

              
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found: {csv_path}")
    sys.exit(1)

print("="*80)
print(csv_path)
print("Loading...")
print("="*80)
df = pd.read_csv(csv_path)

print(f"Total records: {len(df)}")
print(f"Column count: {len(df.columns)}")
print(f"\nColumn names:")
print(df.columns.tolist())
print("\n" + "="*80)
print("First 5 rows:")
print("="*80)
print(df.head())
print("\n" + "="*80)
print("Basic statistics:")
print("="*80)
print(df.describe())
print("\n" + "="*80)
print("Missing-value check:")
print("="*80)
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values")

                                                                        
print("\n" + "="*80)
print("Checking classification columns (periodicity, surface_constraint)")
print("="*80)

required = ["periodicity", "surface_constraint"]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    print(f"Error: columns missing from dataset: {missing_cols}")
    sys.exit(1)

tag = COVERAGE_HAT_CORE_MIN_VALUES[0]
df[f"periodicity_coverage_{tag}"] = df["periodicity"]
df[f"surface_constraint_coverage_{tag}"] = df["surface_constraint"]

for coverage_min in COVERAGE_HAT_CORE_MIN_VALUES:
    col_p = f"periodicity_coverage_{coverage_min}"
    vc = df[col_p].value_counts(dropna=False)
    print(f"  {col_p}: {dict(vc)}")

default_coverage_min = COVERAGE_HAT_CORE_MIN_VALUES[0]
df["periodicity"] = df[f"periodicity_coverage_{default_coverage_min}"]
df["surface_constraint"] = df[f"surface_constraint_coverage_{default_coverage_min}"]
print(f"\n  Reference columns: periodicity / surface_constraint (tag {default_coverage_min})")

                                              
_pt_mask = df["periodicity"].astype(str).eq("Core") & df["surface_constraint"].astype(str).eq(
    "Predominantly tangential"
)
_pt = df.loc[_pt_mask]
_pt_id = "IXI_ID" if "IXI_ID" in _pt.columns else "directory_id"
print("\n" + "=" * 80)
print("Predominantly tangential（Core）")
print("=" * 80)
print(
    f"  Segments: {len(_pt)}, participants: {_pt[_pt_id].nunique() if len(_pt) else 0}"
)


def _print_median_iqr(label: str, series: pd.Series) -> None:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        print(f"  {label}: no non-missing values")
        return
    q1, q3 = values.quantile([0.25, 0.75])
    print(
        f"  {label}: N={len(values)}, "
        f"median={float(values.median()):.4f}, "
        f"IQR=[{float(q1):.4f}, {float(q3):.4f}]"
    )


if "sigma2_over_sigma1" in _pt.columns:
    _print_median_iqr("sigma2_over_sigma1", _pt["sigma2_over_sigma1"])
else:
    print("  sigma2_over_sigma1: column missing")

_lambda_col = "lambda_hat_mm" if "lambda_hat_mm" in _pt.columns else "lambda_hat_s_mm"
if _lambda_col in _pt.columns and "radius_median_mm" in _pt.columns:
    _lam = pd.to_numeric(_pt[_lambda_col], errors="coerce")
    _rad = pd.to_numeric(_pt["radius_median_mm"], errors="coerce")
    _print_median_iqr("lambda/(2*radius)", _lam / (2.0 * _rad))
else:
    print("  lambda/(2*radius): lambda or radius_median_mm column missing")

                                                       
if 'age_group' not in df.columns:
    df['age_group'] = None
    print("Warning: 'age_group' column missing; skipping age-band summaries.")
elif df['age_group'].notna().any():
    print(f"  age_group: {df['age_group'].value_counts(dropna=False).sort_index().to_dict()}")

                
if 'mca_segment' in df.columns:
    mca_segment_missing = df['mca_segment'].isna().sum()
    if mca_segment_missing > 0:
        print(f"  mca_segment: missing {mca_segment_missing}")
    print(f"  mca_segment values: {df['mca_segment'].value_counts(dropna=False).to_dict()}")
else:
    print("Warning: 'mca_segment' column not found; skipping segment summaries.")

                                           
                     
for coverage_min in COVERAGE_HAT_CORE_MIN_VALUES:
    col_name_periodicity = f'periodicity_coverage_{coverage_min}'
    col_name_surface = f'surface_constraint_coverage_{coverage_min}'
    
    print("\n" + "="*80)
    print(f"A. Periodicity (sin+trend) stats (COVERAGE_HAT_CORE_MIN={coverage_min})")
    print("="*80)
    
    periodicities_coverage = df[col_name_periodicity].tolist()
    categories_coverage = df[col_name_surface].tolist()
    
    periodicity_counts = pd.Series(periodicities_coverage).value_counts()
    periodicity_percentages = (periodicity_counts / len(periodicities_coverage) * 100).round(2)

                
    periodicity_order = PERIODICITY_LEVELS
    for periodicity in periodicity_order:
        count = periodicity_counts.get(periodicity, 0)
        percentage = periodicity_percentages.get(periodicity, 0.0)
        print(f"  {periodicity:20s}: {count:6d} ({percentage:6.2f}%)")

    print(f"\n  Total: {len(periodicities_coverage)}")

                                 
    if 'age_group' in df.columns and df['age_group'].notna().any():
        print("\n【age_group × Periodicity（sin+trend）】")
        periodicity_by_age_a = pd.crosstab(
            df['age_group'],
            df[col_name_periodicity],
            margins=True
        )
        periodicity_order_a = PERIODICITY_LEVELS
        periodicity_by_age_a = align_crosstab_columns(periodicity_by_age_a, periodicity_order_a)
        periodicity_by_age_a_pct = periodicity_by_age_a.copy()
        if 'All' in periodicity_by_age_a_pct.index:
            periodicity_by_age_a_pct = periodicity_by_age_a_pct.drop('All')
        if 'All' in periodicity_by_age_a_pct.columns:
            periodicity_by_age_a_pct = periodicity_by_age_a_pct.drop('All', axis=1)
        periodicity_by_age_a_pct = periodicity_by_age_a_pct.div(periodicity_by_age_a_pct.sum(axis=1), axis=0) * 100
        if 'All' not in periodicity_by_age_a.columns:
            row_sums = periodicity_by_age_a_pct.sum(axis=1)
            periodicity_by_age_a_pct['Total'] = row_sums.round(2)
        periodicity_by_age_a_combined = combine_count_and_percentage(periodicity_by_age_a, periodicity_by_age_a_pct)
        print("Counts (pct%):")
        print(periodicity_by_age_a_combined)

                                                 
    print("\n" + "="*80)
    print(f"B. Surface constraint stats (COVERAGE_HAT_CORE_MIN={coverage_min})")
    print("="*80)

            
    core_mask = df[col_name_periodicity] == "Core"
    core_categories = df.loc[core_mask, col_name_surface].tolist()

    if len(core_categories) > 0:
        print("\n[Core only]")
        category_counts = pd.Series(core_categories).value_counts()
        category_percentages = (category_counts / len(core_categories) * 100).round(2)
        
        category_order = SURFACE_LEVELS
        for category in category_order:
            count = category_counts.get(category, 0)
            percentage = category_percentages.get(category, 0.0)
            print(f"  {category:25s}: {count:6d} ({percentage:6.2f}%)")
        
        print(f"  Total: {len(core_categories)}")
                                                                                               
        if 'age_group' in df.columns and df['age_group'].notna().any():
            core_df_b = df[df[col_name_periodicity] == "Core"].copy()
            if len(core_df_b) > 0:
                print("\n[Core only: age_group × In-plane/Out-of-plane/Surface-NotEvaluable]")
                surface_by_age_core_b = pd.crosstab(
                    core_df_b['age_group'],
                    core_df_b[col_name_surface],
                    margins=True
                )
                cat_order = SURFACE_LEVELS
                surface_by_age_core_b = align_crosstab_columns(surface_by_age_core_b, cat_order)
                surface_by_age_core_b_pct = surface_by_age_core_b.copy()
                if 'All' in surface_by_age_core_b_pct.index:
                    surface_by_age_core_b_pct = surface_by_age_core_b_pct.drop('All')
                if 'All' in surface_by_age_core_b_pct.columns:
                    surface_by_age_core_b_pct = surface_by_age_core_b_pct.drop('All', axis=1)
                surface_by_age_core_b_pct = surface_by_age_core_b_pct.div(surface_by_age_core_b_pct.sum(axis=1), axis=0) * 100
                if 'All' not in surface_by_age_core_b.columns:
                    row_sums = surface_by_age_core_b_pct.sum(axis=1)
                    surface_by_age_core_b_pct['Total'] = row_sums.round(2)
                surface_by_age_core_b_combined = combine_count_and_percentage(surface_by_age_core_b, surface_by_age_core_b_pct)
                print("Counts (pct%):")
                print(surface_by_age_core_b_combined)
    else:
        print("\n[Core only]")
        print("  No Core rows; no surface-constraint labels")

                                                                                           
    if 'mca_segment' in df.columns:
        print("\n" + "="*80)
        print(f"C. By segment (mca_segment) stats (COVERAGE_HAT_CORE_MIN={coverage_min})")
        print("="*80)
        print("(mca_segment is Core-only; reporting Core-only surface-constraint summaries)")
        
        core_df = df[df[col_name_periodicity] == "Core"].copy()
        if len(core_df) > 0:
            print("\n[Core only: mca_segment × surface constraint]")
            surface_by_segment_core = pd.crosstab(
                core_df['mca_segment'],
                core_df[col_name_surface],
                margins=True
            )
            category_order = SURFACE_LEVELS
            surface_by_segment_core = align_crosstab_columns(surface_by_segment_core, category_order)
            surface_by_segment_core_pct = surface_by_segment_core.copy()
            if 'All' in surface_by_segment_core_pct.index:
                surface_by_segment_core_pct = surface_by_segment_core_pct.drop('All')
            if 'All' in surface_by_segment_core_pct.columns:
                surface_by_segment_core_pct = surface_by_segment_core_pct.drop('All', axis=1)
            surface_by_segment_core_pct = surface_by_segment_core_pct.div(surface_by_segment_core_pct.sum(axis=1), axis=0) * 100
            if 'All' not in surface_by_segment_core.columns:
                row_sums = surface_by_segment_core_pct.sum(axis=1)
                surface_by_segment_core_pct['Total'] = row_sums.round(2)
            
            surface_by_segment_core_combined = combine_count_and_percentage(surface_by_segment_core, surface_by_segment_core_pct)
            print("Counts (pct%):")
            print(surface_by_segment_core_combined)
        else:
            print("\n[Core only]")
            print("No Core rows; no by-segment summary")

                            
    if 'age_group' in df.columns and df['age_group'].notna().any():
        print("\n" + "="*80)
        print(f"D. By age-band stats (COVERAGE_HAT_CORE_MIN={coverage_min})")
        print("="*80)
        
                         
        print("\n[Periodicity (sin+trend)]")
        periodicity_by_age = pd.crosstab(
            df['age_group'],
            df[col_name_periodicity],
            margins=True
        )
               
        periodicity_order = PERIODICITY_LEVELS
        periodicity_by_age = align_crosstab_columns(periodicity_by_age, periodicity_order)
        
                             
        periodicity_by_age_pct = periodicity_by_age.copy()
        if 'All' in periodicity_by_age_pct.index:
            periodicity_by_age_pct = periodicity_by_age_pct.drop('All')
        if 'All' in periodicity_by_age_pct.columns:
            periodicity_by_age_pct = periodicity_by_age_pct.drop('All', axis=1)
        periodicity_by_age_pct = periodicity_by_age_pct.div(periodicity_by_age_pct.sum(axis=1), axis=0) * 100
                                     
        if 'All' not in periodicity_by_age.columns:
            row_sums = periodicity_by_age_pct.sum(axis=1)
            periodicity_by_age_pct['Total'] = row_sums.round(2)
        
                        
        periodicity_by_age_combined = combine_count_and_percentage(periodicity_by_age, periodicity_by_age_pct)
        print("\nCounts (pct%):")
        print(periodicity_by_age_combined)
        
                                
        print("\n[Surface constraint]")
        
                
        core_df = df[df[col_name_periodicity] == "Core"].copy()
        if len(core_df) > 0:
            print("\n[Core only]")
            surface_by_age_core = pd.crosstab(
                core_df['age_group'],
                core_df[col_name_surface],
                margins=True
            )
            category_order = SURFACE_LEVELS
            surface_by_age_core = align_crosstab_columns(surface_by_age_core, category_order)
            surface_by_age_core_pct = surface_by_age_core.copy()
            if 'All' in surface_by_age_core_pct.index:
                surface_by_age_core_pct = surface_by_age_core_pct.drop('All')
            if 'All' in surface_by_age_core_pct.columns:
                surface_by_age_core_pct = surface_by_age_core_pct.drop('All', axis=1)
            surface_by_age_core_pct = surface_by_age_core_pct.div(surface_by_age_core_pct.sum(axis=1), axis=0) * 100
                                         
            if 'All' not in surface_by_age_core.columns:
                row_sums = surface_by_age_core_pct.sum(axis=1)
                surface_by_age_core_pct['Total'] = row_sums.round(2)
            
                            
            surface_by_age_core_combined = combine_count_and_percentage(surface_by_age_core, surface_by_age_core_pct)
            print("Counts (pct%):")
            print(surface_by_age_core_combined)
        else:
            print("\n[Core only]")
            print("No Core rows; no surface-constraint summary")

                                                                                   
    if 'mca_segment' in df.columns and 'age_group' in df.columns and df['age_group'].notna().any():
        print("\n" + "="*80)
        print(f"E. By segment × age-band stats (COVERAGE_HAT_CORE_MIN={coverage_min})")
        print("="*80)
        print("(mca_segment is Core-only; reporting Core-only mca_segment × age-band × surface constraint)")
        
        core_df = df[df[col_name_periodicity] == "Core"].copy()
        if len(core_df) > 0:
            print("\n[Core only: mca_segment × age_group × surface constraint]")
            surface_by_segment_age_core = pd.crosstab(
                [core_df['mca_segment'], core_df['age_group']],
                core_df[col_name_surface],
                margins=True
            )
            category_order = SURFACE_LEVELS
            surface_by_segment_age_core = align_crosstab_columns(surface_by_segment_age_core, category_order)
            surface_by_segment_age_core_pct = surface_by_segment_age_core.copy()
            if isinstance(surface_by_segment_age_core_pct.index, pd.MultiIndex):
                surface_by_segment_age_core_pct = surface_by_segment_age_core_pct[~surface_by_segment_age_core_pct.index.get_level_values(-1).isin(['All'])]
            elif 'All' in surface_by_segment_age_core_pct.index:
                surface_by_segment_age_core_pct = surface_by_segment_age_core_pct.drop('All')
            if 'All' in surface_by_segment_age_core_pct.columns:
                surface_by_segment_age_core_pct = surface_by_segment_age_core_pct.drop('All', axis=1)
            surface_by_segment_age_core_pct = surface_by_segment_age_core_pct.div(surface_by_segment_age_core_pct.sum(axis=1), axis=0) * 100
            if 'All' not in surface_by_segment_age_core.columns:
                row_sums = surface_by_segment_age_core_pct.sum(axis=1)
                surface_by_segment_age_core_pct['Total'] = row_sums.round(2)
            
            surface_by_segment_age_core_combined = combine_count_and_percentage(surface_by_segment_age_core, surface_by_segment_age_core_pct)
            print("Counts (pct%):")
            print(surface_by_segment_age_core_combined)
        else:
            print("\n[Core only]")
            print("No Core rows; no segment × age-band summary")

def show_figure_with_space_to_close(fig):
    closed = [False]

    def on_key(event):
        if event.key == ' ':
            closed[0] = True
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=False)
    while not closed[0]:
        if fig.number not in plt.get_fignums():
            break
        plt.pause(0.1)


def _median_iqr_str(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "NA"
    med = float(s.median())
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    return f"{med:.2f} ({q1:.2f}–{q3:.2f})"


def _plot_age_histogram_overlay(ax, ages_all: np.ndarray, ages_sel: np.ndarray, *, title: str, label_sel: str):
    ages_all = np.asarray(ages_all, dtype=float)
    ages_sel = np.asarray(ages_sel, dtype=float)
    ages_all = ages_all[np.isfinite(ages_all)]
    ages_sel = ages_sel[np.isfinite(ages_sel)]
    if len(ages_all) == 0:
        ax.text(0.5, 0.5, "No valid AGE (all)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    lo = float(np.min(ages_all))
    hi = float(np.max(ages_all))
    if len(ages_sel) > 0:
        lo = min(lo, float(np.min(ages_sel)))
        hi = max(hi, float(np.max(ages_sel)))
    span = max(hi - lo, 1e-6)
    n_bins = int(np.clip(round(span), 8, 35))
    bins = np.linspace(lo, hi, n_bins + 1)
    ax.hist(
        ages_all,
        bins=bins,
        alpha=0.45,
        color="gray",
        edgecolor="black",
        linewidth=0.6,
        label="All participants (unique)",
    )
    if len(ages_sel) > 0:
        ax.hist(
            ages_sel,
            bins=bins,
            alpha=0.55,
            color="darkorange",
            edgecolor="black",
            linewidth=0.6,
            label=label_sel,
        )
    else:
        ax.text(0.02, 0.98, "Cohort: no AGE", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.set_xlabel("AGE (years)")
    ax.set_ylabel("Number of participants")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


                                                                                   
if 'directory_id' in df.columns:
    n_cohorts = len(COVERAGE_HAT_CORE_MIN_VALUES)
                                                                     
    fig, axes = plt.subplots(
        2, n_cohorts,
        figsize=(5 * n_cohorts, 4 * 2),
        constrained_layout=True
    )
    if n_cohorts == 1:
        axes = np.reshape(axes, (2, 1))

    def draw_hist_for_cohort(ax, df_cohort, coverage_min, group_label, color='steelblue'):
        seg_per_subject = df_cohort.groupby('directory_id').size()
        if len(seg_per_subject) == 0:
            ax.text(0.5, 0.5, f'coverage_{coverage_min}\n{group_label}\nNo data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Segments per subject')
            ax.set_ylabel('Number of subjects')
            ax.set_title(f'coverage_{coverage_min}\n{group_label} (N=0 subjects)')
            return
        max_seg = int(seg_per_subject.max())
        ax.hist(seg_per_subject, bins=range(1, max_seg + 2), align='left', color=color, edgecolor='black', alpha=0.8)
        ax.set_xticks(range(1, max_seg + 1))
        n_subjects = int(seg_per_subject.shape[0])
        n_segments = int(seg_per_subject.sum())
        mean_k = float(seg_per_subject.mean())
        median_k = float(seg_per_subject.median())
        q1, q3 = seg_per_subject.quantile([0.25, 0.75])
        max_k = int(seg_per_subject.max())
        pct_ge2 = 100.0 * float((seg_per_subject >= 2).mean())
        stats_txt = (
            f"Segments: {n_segments}\n"
            f"Mean: {mean_k:.2f}\n"
            f"Median [IQR]: {median_k:.0f} [{q1:.0f}–{q3:.0f}]\n"
            f"Max: {max_k}\n"
            f"% ≥2 segments: {pct_ge2:.1f}%"
        )
        ax.text(0.98, 0.98, stats_txt, transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='black'))
        ax.set_xlabel('Segments per subject')
        ax.set_ylabel('Number of subjects')
        ax.set_title(f'coverage_{coverage_min}\n{group_label} (N={n_subjects} subjects)')

    for i, coverage_min in enumerate(COVERAGE_HAT_CORE_MIN_VALUES):
                                                   
        col_p = f'periodicity_coverage_{coverage_min}'
        if col_p not in df.columns:
            axes[0, i].text(0.5, 0.5, f'coverage_{coverage_min}\nColumn not found: {col_p}', ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].set_axis_off()
        else:
            df_core = df[df[col_p] == "Core"].copy()
            draw_hist_for_cohort(axes[0, i], df_core, coverage_min, 'Core', color='steelblue')

                                                             
        surface_pt = "Predominantly tangential"
        mask_pt = core_surface_mask(df, coverage_min, surface_pt)
        if mask_pt is None:
            axes[1, i].text(
                0.5, 0.5,
                f'coverage_{coverage_min}\nperiodicity / surface_constraint missing',
                ha='center', va='center', transform=axes[1, i].transAxes,
            )
            axes[1, i].set_axis_off()
        else:
            draw_hist_for_cohort(
                axes[1, i], df.loc[mask_pt].copy(), coverage_min, surface_pt, color='darkorange',
            )

    show_figure_with_space_to_close(fig)
else:
    print("\nWarning: 'directory_id' column missing; skipped segment-count histogram")

                                                                                  
if "directory_id" in df.columns:
    _subject_col = "directory_id"
elif "IXI_ID" in df.columns:
    _subject_col = "IXI_ID"
else:
    _subject_col = None

print("\n" + "="*80)
print("Predominantly tangential / Substantial normal component: segment and subject counts (by coverage)")
print("="*80)
if _subject_col is None:
    print(
        "Warning: neither 'directory_id' nor 'IXI_ID' present; "
        "cannot count subjects (showing segment counts only)."
    )
for _cov in COVERAGE_HAT_CORE_MIN_VALUES:
    _col_core = f"periodicity_coverage_{_cov}"
    # --- Core (periodicity) ---
    if _col_core in df.columns:
        _m_core = df[_col_core] == "Core"
        _n_seg_core = int(_m_core.sum())
        if _subject_col is not None:
            _n_sub_core = int(df.loc[_m_core, _subject_col].nunique())
            print(
                f"  coverage_{_cov} Core: "
                f"{_n_seg_core} segments from {_n_sub_core} participants ({_subject_col})"
            )
            _sp_core = df.loc[_m_core].groupby(_subject_col).size()
            print(
                f"      Segments per participant, median (IQR): "
                f"{_median_iqr_str(_sp_core)} (n_participants={int(_sp_core.shape[0])})"
            )
        else:
            print(f"  coverage_{_cov} Core: {_n_seg_core} segments")
    else:
        print(f"  coverage_{_cov} Core: column not found: {_col_core}")
    for _surface in SURFACE_LEVELS:
        _m_surface = core_surface_mask(df, _cov, _surface)
        if _m_surface is None:
            print(f"  coverage_{_cov} {_surface}: periodicity / surface_constraint missing")
            continue
        _n_seg_surface = int(_m_surface.sum())
        if _subject_col is not None:
            _n_sub_surface = int(df.loc[_m_surface, _subject_col].nunique())
            print(
                f"  coverage_{_cov} {_surface}: "
                f"{_n_seg_surface} segments from {_n_sub_surface} participants ({_subject_col})"
            )
            _sp_surface = df.loc[_m_surface].groupby(_subject_col).size()
            print(
                f"      Segments per participant, median (IQR): "
                f"{_median_iqr_str(_sp_surface)} (n_participants={int(_sp_surface.shape[0])})"
            )
        else:
            print(f"  coverage_{_cov} {_surface}: {_n_seg_surface} segments")

                                                                                                
if _subject_col is None:
    print(
        "\nWarning: participant ID column missing; "
        "skipped age histogram for surface-constraint cohort."
    )
elif "AGE" not in df.columns:
    print("\nWarning: 'AGE' column missing; skipped age histogram.")
else:
    _age_df = df[[_subject_col, "AGE"]].copy()
    _age_df["AGE"] = pd.to_numeric(_age_df["AGE"], errors="coerce")
    _age_df = _age_df.dropna(subset=[_subject_col, "AGE"]).drop_duplicates(subset=[_subject_col])
    _ages_all = _age_df["AGE"].to_numpy(dtype=float)

    _n_cov = len(COVERAGE_HAT_CORE_MIN_VALUES)
    _fig_age, _axes_age = plt.subplots(
        len(SURFACE_LEVELS),
        _n_cov,
        figsize=(5 * _n_cov, 3.5 * len(SURFACE_LEVELS)),
        constrained_layout=True,
    )
    if _n_cov == 1:
        _axes_age = np.reshape(_axes_age, (len(SURFACE_LEVELS), 1))

    for _j, _cov in enumerate(COVERAGE_HAT_CORE_MIN_VALUES):
        for _row, _surface in enumerate(SURFACE_LEVELS):
            _ax = _axes_age[_row, _j]
            _mask = core_surface_mask(df, _cov, _surface)
            if _mask is None:
                _ax.text(
                    0.5, 0.5,
                    f"coverage_{_cov}\nperiodicity / surface_constraint missing",
                    ha="center", va="center", transform=_ax.transAxes,
                )
                _ax.set_title(f"coverage_{_cov}\nAGE: {_surface}")
                continue
            _ids = df.loc[_mask, _subject_col].dropna().unique()
            _ages = _age_df.loc[_age_df[_subject_col].isin(_ids), "AGE"].to_numpy(dtype=float)
            _plot_age_histogram_overlay(
                _ax,
                _ages_all,
                _ages,
                title=f"coverage_{_cov}\nAGE: {_surface} vs all",
                label_sel=f"{_surface}",
            )

    _fig_age.suptitle(
        "Participant age: surface cohort vs all (unique per participant ID)",
        fontsize=11,
    )
    show_figure_with_space_to_close(_fig_age)

                                                                                          
                                                                   
if "sigma2_over_sigma1" in df.columns:
    _ellipticity = pd.to_numeric(df["sigma2_over_sigma1"], errors="coerce")
elif "amp_minor_mm" in df.columns and "amp_major_mm" in df.columns:
    _minor = pd.to_numeric(df["amp_minor_mm"], errors="coerce")
    _major = pd.to_numeric(df["amp_major_mm"], errors="coerce")
    _ellipticity = _minor / _major.replace(0, np.nan)
else:
    _ellipticity = None

if _ellipticity is not None:
    print("\n" + "=" * 80)
    print("ellipticity (sigma2_over_sigma1): median and IQR (per group)")
    print("=" * 80)
    df["_ellipticity"] = _ellipticity.replace([np.inf, -np.inf], np.nan)

    n_cohorts = len(COVERAGE_HAT_CORE_MIN_VALUES)
    n_surfaces = len(SURFACE_LEVELS)
    fig, axes = plt.subplots(
        n_surfaces, n_cohorts,
        figsize=(5 * n_cohorts, 4 * n_surfaces),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    if n_cohorts == 1:
        axes = np.reshape(axes, (n_surfaces, 1))
    _ellip_bins = np.linspace(0.0, 1.0, 21)

    def _draw_ellipticity_hist(ax, mask, coverage_min, group_label, color):
        if mask is None:
            ax.text(
                0.5, 0.5, f"coverage_{coverage_min}\n{group_label}\nColumn not found",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_axis_off()
            return
        vals = df.loc[mask, "_ellipticity"].dropna()
        print(f"  coverage_{coverage_min} / {group_label}")
        _print_median_iqr("ellipticity", vals)
        if len(vals) == 0:
            ax.text(
                0.5, 0.5, f"coverage_{coverage_min}\n{group_label}\nNo data",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title(f"coverage_{coverage_min}\n{group_label} (N=0)")
            return
        ax.hist(vals, bins=_ellip_bins, color=color, edgecolor="black", alpha=0.8)
        q1, q3 = vals.quantile([0.25, 0.75])
        stats_txt = (
            f"N={len(vals)}\n"
            f"Median [IQR]: {float(vals.median()):.3f}\n"
            f"[{float(q1):.3f}–{float(q3):.3f}]"
        )
        ax.text(
            0.98, 0.98, stats_txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="black"),
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"ellipticity ($\sigma_2/\sigma_1$)")
        ax.set_ylabel("Number of segments")
        ax.set_title(f"coverage_{coverage_min}\n{group_label} (N={len(vals)})")
        ax.grid(True, axis="y", alpha=0.3)

    _surface_colors = [orange_color, blue_color]
    for i, coverage_min in enumerate(COVERAGE_HAT_CORE_MIN_VALUES):
        for row, surface_level in enumerate(SURFACE_LEVELS):
            color = _surface_colors[row % len(_surface_colors)]
            _draw_ellipticity_hist(
                axes[row, i],
                core_surface_mask(df, coverage_min, surface_level),
                coverage_min,
                surface_level,
                color,
            )

    fig.suptitle(r"Ellipticity ($\sigma_2/\sigma_1$ = amp_minor / amp_major)", fontsize=11)
    show_figure_with_space_to_close(fig)
    df.drop(columns=["_ellipticity"], inplace=True)
else:
    print(
        "\nWarning: neither sigma2_over_sigma1 nor amp_minor_mm / amp_major_mm present; "
        "skipped ellipticity histogram"
    )

print("\n" + "="*80)
print("Done")
print("="*80)
