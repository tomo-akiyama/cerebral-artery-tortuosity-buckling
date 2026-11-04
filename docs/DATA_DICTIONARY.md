# Data dictionary

## `data/participants/IXI_HH_participants.csv`
Participant-level demographics and CAT12 morphometrics used as covariates.
Key columns (units in brackets):

| column | meaning | unit |
|--------|---------|------|
| IXI_ID | IXI participant identifier | — |
| AGE | age | years |
| SEX_ID / SEX | sex coding as in IXI / analysis | — |
| ETHNIC_ID | ethnicity code (IXI); analysis uses collapsed groups | — |
| TIV | total intracranial volume (CAT12) | mm^3 |
| directory_id | local case folder name (`IXI###-HH-####`) | — |

## `data/m1/IXI_HH_m1_features_long.csv`
Anatomically defined M1 segment measurements (long format; one or more paths per hemisphere).

| column | meaning | unit |
|--------|---------|------|
| IXI_ID / directory_id | participant keys | — |
| side | left/right | — |
| length / m1_length | centerline length \(L\) | mm |
| euclid / euclid_distance | endpoint Euclidean distance \(L_0\) | mm |
| TI | tortuosity index \(L/L_0\) | — |
| m1_rho / rho | excess-length ratio \(\rho=TI-1\) | — |
| branch type fields | M1 major branch type used in regressions | — |

## `data/periodic_wave/IXI_HH_sin_trend_analysis_dataset.csv`
Interval-level sinusoid-plus-trend results for all analyzable MCA intervals (`n=11808`).
**Public release note:** `mca_segment` (and related MCA label metadata) are retained only for
intervals with `core_inplane==1` (predominantly tangential periodic deviations; `n=275`).
Labels on all other intervals were cleared for release.

Selected columns:

| column | meaning | unit |
|--------|---------|------|
| IXI_ID, directory_id, segment_id | keys | — |
| mca_segment | M1 / M2-3 / M4 (only if `core_inplane==1`) | — |
| radius_median_mm | median lumen radius along fitting samples | mm |
| lambda_hat_mm / lambda_hat_s_mm | estimated wavelength | mm |
| amp_hat_mm | combined amplitude \(\hat A\) | mm |
| partial_R2 | partial \(R^2\) of sinusoid beyond linear trend | — |
| coverage_hat / N_cyc-like fields | fitted cycle count \(\hat N_{cyc}\) | — |
| R_tan / R_out | tangential / normal squared-displacement fractions | — |
| plane_class | directional class vs brain surface | — |
| core_periodic, core_inplane | qualification flags used in the paper | — |
| AGE, SEX, ethnicity, TIV^{1/3}, side | merged covariates | mixed |

## `data/periodic_wave/core_inplane.csv`
Subset with `core_inplane==1` (`n=275`), used for primary wavelength–diameter and amplitude analyses.

## `data/simulation/.../P_coupling.npz`
Saved ACTIVE simulation trajectory (`L0=100` mm, `N=1000`, `dt=0.01` yr, `g=0.35` mm/yr, `mu=0.005`).
Arrays include displacement field snapshots, axial force \(P(t)\), and metadata.

## `data/statistics/*.csv`
Regression coefficients, piecewise/spline/bootstrap summaries for Main/SI tables.

## `data/validation/*`
Summary-only synthetic-centerline validation outputs (large per-case CSVs omitted).
