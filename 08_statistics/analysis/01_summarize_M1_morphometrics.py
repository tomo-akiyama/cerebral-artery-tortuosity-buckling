#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

# Path to the csv file
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_path = os.path.join(project_root, 'results', 'participants', 'IXI_HH_participants.csv')

# Load the data
df = pd.read_csv(csv_path)

import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

# Import plot configuration
import sys
sys.path.insert(0, os.path.join(project_root, 'figures'))
import plot_config

               
plt.figure()
sns.scatterplot(x=df["AGE"], y=df["TIV"])
sns.regplot(x=df["AGE"], y=df["TIV"], scatter=False, color='red', label='Linear fit')
plt.xlabel("Age (years)")
plt.ylabel("TIV")
plt.title("Scatter plot of Age vs TIV")
plt.legend()
plt.show()

             
corr, pval = pearsonr(df["AGE"], df["TIV"])
print(f"Pearson correlation: r = {corr:.3f}, p-value = {pval:.5f}")

                                                                                              
model = smf.ols("TIV ~ AGE + C(sex, Treatment(reference='male')) + HEIGHT", data=df).fit()
print(model.summary())

                                      
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = model.model.exog
vif = pd.DataFrame({
    "variable": model.model.exog_names,
    "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
})
print(vif)


################################################################################
                                                    

                    
import re
                      
long_len = (
    df.melt(
        id_vars='IXI_ID',                                               
        value_vars=[c for c in df.columns                                       
                    if re.match(r'^[rl]t_m1_length', c)],
        var_name='variable',
        value_name='value'
    )
)

          
pat_len = (
    r'^(?P<side>[rl]t)_m1_length'                          # side: rt / lt
    r'(?P<trunk>[12])'                                     # trunk: 1 or 2
    r'(?:_(?P<measure>euclid|ti))?$'                       # measure: euclid / ti / None
)

long_len = (
    long_len
    .assign(**long_len['variable'].str.extract(pat_len))
    .drop(columns='variable')
    .assign(measure=lambda d: d['measure'].fillna('length'))
)

# 3. pivot
long_len = (
    long_len
    .pivot_table(index=['IXI_ID', 'side', 'trunk'], columns='measure', values='value', aggfunc='first')
    .reset_index()
    .rename(columns={'length': 'm1_length', 'euclid': 'm1_euclid', 'ti': 'm1_ti'})
)
                   
long_rad = (
    df.melt(
        id_vars='IXI_ID',
        value_vars=[c for c in df.columns
                    if re.match(r'^[rl]t_m1_radius', c)],
        var_name='variable',
        value_name='value'
    )
)
pat_rad = (
    r'^(?P<side>[rl]t)_m1_radius'
    r'(?P<trunk>[12])_'
    r'(?P<stat>median|q1|q3|iqr|mean|std)$'
)
long_rad = (
    long_rad
    .assign(**long_rad['variable'].str.extract(pat_rad))
    .drop(columns='variable')
)
long_rad = (
    long_rad
    .pivot_table(index=['IXI_ID', 'side', 'trunk'],
                 columns='stat', values='value', aggfunc='first')
    .reset_index()
)

long_rad = long_rad.rename(
    columns={c: f'm1_radius_{c}'
             for c in ['median', 'q1', 'q3', 'iqr', 'mean', 'std']}
)

                        
long_df = long_len.merge(long_rad, on=['IXI_ID', 'side', 'trunk'], how='left')

        
                         
#   m1_length, m1_euclid, m1_ti,
#   m1_radius_median, m1_radius_q1, ... , m1_radius_std
                      

                            
branch = (
    df.set_index('IXI_ID')[['rt_m1_branch_pattern', 'lt_m1_branch_pattern']]
      .rename(columns={'rt_m1_branch_pattern': 'rt', 'lt_m1_branch_pattern': 'lt'})
      .stack()
      .rename_axis(index=['IXI_ID', 'side'])
      .reset_index(name='branch_pattern')
)
bif = (
    df.set_index('IXI_ID')[['rt_m1_bifurcation_location', 'lt_m1_bifurcation_location']]
      .rename(columns={'rt_m1_bifurcation_location': 'rt', 'lt_m1_bifurcation_location': 'lt'})
      .stack()
      .rename_axis(index=['IXI_ID', 'side'])
      .reset_index(name='bifurcation_location')
)
long_df = long_df.merge(branch, on=['IXI_ID', 'side'], how='left')
long_df = long_df.merge(bif, on=['IXI_ID', 'side'], how='left')

# 7. subject-level covariates
subject_cov = [
    'sex', 'HEIGHT', 'WEIGHT',
    'ethnicity', 'ethnicity_group',
    'AGE', 'bmi', 'bsa_dubois',
    'CSF_absolute_volume', 'GM_absolute_volume', 'WM_absolute_volume',
    'GM_and_WM_absolute_volume', 'vol_abs_WMH',
    'surf_TSA', 'Average_thickness_FS', 'Average_thickness_std',
    'TIV', 'TIV_cuberoot'
]

long_df = long_df.merge(
    df[['IXI_ID'] + subject_cov].drop_duplicates(),
    on='IXI_ID', how='left'
)

print(long_df.head())
# Save to csv
long_df.to_csv(os.path.join(project_root, 'results', 'participants', 'IXI_HH_m1_features_long.csv'), index=False)

################################################################################
# Vessel-level summary table (M1 segments; clustered within participants)
#
                                                   
                                        
#
                                                 
                                                    
#
                             
# - M1 length
# - Euclidean distance
# - T
# - rho (= T - 1)
# - M1 radius median
# - M1 radius IQR
################################################################################

def _median_iqr(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return {"n": 0, "median": np.nan, "q1": np.nan, "q3": np.nan}
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    med = float(s.median())
    return {"n": int(len(s)), "median": med, "q1": q1, "q3": q3}


def print_rho_summary_under_age(
    df: pd.DataFrame,
    age_years: float,
    *,
    rho_col: str = "m1_rho",
    age_col: str = "AGE",
) -> None:
    sub = df.loc[pd.to_numeric(df[age_col], errors="coerce") < age_years].copy()
    sub[rho_col] = pd.to_numeric(sub[rho_col], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[rho_col])
    rho = sub[rho_col]
    print("\n" + "=" * 80)
    print(f"M1 excess-length ratio rho (segments with {age_col} < {age_years:g} years)")
    print("=" * 80)
    if rho.empty:
        print("  No valid segments.")
        return
    print(f"  n segments = {int(rho.shape[0])}")
    if "IXI_ID" in sub.columns:
        print(f"  n participants = {int(sub['IXI_ID'].nunique())}")
    print(f"  mean   = {float(rho.mean()):.4f}")
    print(f"  median = {float(rho.median()):.4f}")
    print(f"  Q1     = {float(rho.quantile(0.25)):.4f}")
    print(f"  Q3     = {float(rho.quantile(0.75)):.4f}")


print("\n" + "=" * 80)
print("M1 vessel-level (segment-level) descriptive summary")
print("=" * 80)
n_participants = int(long_df["IXI_ID"].nunique())
n_segments = int(long_df.shape[0])
print(f"M1 segments, n={n_segments}")
print(f"{n_segments} M1 segments from {n_participants} participants (clustered by IXI_ID)")

ti = pd.to_numeric(long_df["m1_ti"], errors="coerce")
long_df["m1_rho"] = ti - 1.0

print_rho_summary_under_age(long_df, age_years=35.0)

specs = [
    ("M1 centerline length, mm", "m1_length"),
    ("M1 Euclidean distance, mm", "m1_euclid"),
    ("Tortuosity index (T)", "m1_ti"),
    ("Excess-length ratio (rho)", "m1_rho"),
    ("M1 radius median, mm", "m1_radius_median"),
    ("M1 radius IQR, mm", "m1_radius_iqr"),
]

rows = []
for label, col in specs:
    if col not in long_df.columns:
        rows.append({"variable": label, "column": col, "n": 0, "median(IQR)": "NA"})
        continue
    s = _median_iqr(long_df[col])
    rows.append(
        {
            "variable": label,
            "column": col,
            "n": s["n"],
            "median(IQR)": f'{s["median"]:.4g} ({s["q1"]:.4g}–{s["q3"]:.4g})',
        }
    )

tbl = pd.DataFrame(rows)
print("\n--- median (IQR) across M1 segments ---")
print(tbl.to_string(index=False))

################################################################################
# M1 radius / diameter summary (subject-level → group-level)
#
# Hierarchy of averaging:
#   (i)  Within-M1 (subject × side): simple mean over trunks (no length-weighting)
#   (ii) Left/right: combine trunk 1 and trunk 2 per side
#   (iii) Subject-level: average of left and right
################################################################################

# Check required columns (produced in long_df)
required_cols_radius = ["IXI_ID", "side", "trunk", "m1_length", "m1_radius_mean"]
missing_radius = [c for c in required_cols_radius if c not in long_df.columns]
if missing_radius:
    print("\n[WARN] Skip M1 radius summary: missing columns:", missing_radius)
else:
    def _summarise_radius_by_side(group: pd.DataFrame) -> pd.Series:
        length = group["m1_length"].to_numpy()
        radius_mean = group["m1_radius_mean"].to_numpy()

        valid = np.isfinite(radius_mean) & np.isfinite(length)
        if not np.any(valid):
            return pd.Series(
                {
                    "radius_M1": np.nan,
                    "length_M1": np.nan,
                    "n_trunk": group.shape[0],
                }
            )

        r = radius_mean[valid]
        radius_M1 = float(r.mean())
        length_M1 = float(length[valid].sum())

        return pd.Series(
            {
                "radius_M1": radius_M1,
                "length_M1": length_M1,
                "n_trunk": group.shape[0],
            }
        )

    by_subject_side = (
                                                                             
                                                                   
        long_df.groupby(["IXI_ID", "side"])[["m1_length", "m1_radius_mean"]]
        .apply(_summarise_radius_by_side)
        .reset_index()
    )

    side_map = {"lt": "left", "rt": "right"}
    by_subject_side["side_label"] = by_subject_side["side"].map(side_map)

    radius_wide = (
        by_subject_side.pivot(index="IXI_ID", columns="side_label", values="radius_M1")
        .rename(columns={"left": "radius_M1_left", "right": "radius_M1_right"})
    )

    length_wide = (
        by_subject_side.pivot(index="IXI_ID", columns="side_label", values="length_M1")
        .rename(columns={"left": "length_M1_left", "right": "length_M1_right"})
    )

    ntrunk_wide = (
        by_subject_side.pivot(index="IXI_ID", columns="side_label", values="n_trunk")
        .rename(columns={"left": "n_trunk_left", "right": "n_trunk_right"})
    )

    subject_df = radius_wide.join([length_wide, ntrunk_wide])
    subject_df["radius_M1_LR_mean"] = subject_df[["radius_M1_left", "radius_M1_right"]].mean(axis=1)

    for col_r in ["radius_M1_left", "radius_M1_right", "radius_M1_LR_mean"]:
        dcol = col_r.replace("radius", "diameter")
        subject_df[dcol] = 2.0 * subject_df[col_r]

    covariate_cols = ["sex", "AGE", "HEIGHT", "WEIGHT", "bmi", "bsa_dubois"]
    covariate_cols = [c for c in covariate_cols if c in long_df.columns]
    if covariate_cols:
        cov = (
            long_df[["IXI_ID"] + covariate_cols]
            .drop_duplicates(subset="IXI_ID")
            .set_index("IXI_ID")
        )
        subject_df = subject_df.join(cov, how="left")

    subject_df = subject_df.reset_index().rename(columns={"IXI_ID": "subject_id"})

    # Group-level statistics
    cols_to_summarise = [
        "radius_M1_left",
        "radius_M1_right",
        "radius_M1_LR_mean",
        "diameter_M1_left",
        "diameter_M1_right",
        "diameter_M1_LR_mean",
    ]

    stats = []
    for col in cols_to_summarise:
        if col not in subject_df.columns:
            continue
        series = subject_df[col].dropna()
        if series.empty:
            continue
        stats.append(
            {
                "variable": col,
                "n": int(series.shape[0]),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "min": float(series.min()),
                "q1": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q3": float(series.quantile(0.75)),
                "max": float(series.max()),
            }
        )
    group_stats = pd.DataFrame(stats)

    cols_display = [
        "subject_id",
        "radius_M1_left",
        "radius_M1_right",
        "radius_M1_LR_mean",
        "diameter_M1_left",
        "diameter_M1_right",
        "diameter_M1_LR_mean",
    ]
    cols_display = [c for c in cols_display if c in subject_df.columns]

    print("\n=== Subject-level M1 radius / diameter (first 10 subjects) ===")
    if cols_display:
        print(subject_df[cols_display].head(10).to_string(index=False))
    else:
        print("No subject-level columns to display.")

    print("\n=== Group-level statistics of M1 radius / diameter ===")
    if not group_stats.empty:
        print(group_stats.to_string(index=False))
    else:
        print("No group-level statistics available.")