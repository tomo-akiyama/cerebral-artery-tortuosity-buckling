# OUTPUT_SCHEMA (sin_trend)

Canonical full table: `IXI_HH_sin_trend_analysis_dataset.csv`

## Columns needed for wavelength–radius / normalized wavelength–age / amplitude–age

| Use | Columns |
|---|---|
| ID | directory_id, segment_id, segment_index, IXI_ID, side |
| Age | AGE/age (raw), age_years |
| Radius (primary = after trim) | radius_median_mm, diameter_ref_mm (=2× post-trim median) |
| Radius (full reference) | radius_median_full_mm, radius_mean_full_mm, radius_std_full_mm |
| Wavelength | lambda_hat_mm (estimated directly from baseline arc-length phase). lambda_hat_s_mm / lambda_baseline_eff_mm are compatibility aliases |
| Interval length | L_base_mm (trim-matched baseline arc length), L_trimmed_mm / L_full_mm (observed) |
| Slenderness | slenderness = lambda_hat_mm / (2 * radius_median_mm) |
| Coordinate metadata | fitting/phase/trend=baseline_arclength; quadratic_detrend=observed_arclength; grid_length_basis=trimmed_baseline_arclength (row / run_config / NPZ scalars_json) |
| Normalization | lambda_s_over_r, lambda_base_eff_over_r, amp_comb_over_L, amp_comb_over_r, log_* |
| Amplitude | amp_comb_mm, amp_major_mm (=sigma1_mm), amp_minor_mm (=sigma2_mm) |
| Anatomy | mca_segment, label_status, mca_label_conflict |
| Acceptance | core_periodic, core_all(=alias), core_inplane, reject_reasons |
| Fit | R2, partial_R2, SSE_*_mm2, fit_status, fit_method |

## Partial tables

- `MCA_M2M4_core_periodic.csv` / `MCA_M2M4_core_inplane.csv`
- `paired_wavelength_coordinate_intervals.csv`
- `paired_amplitude_definition_intervals.csv`

## Missing values / bools

- `"False"`/`"0"` are False (`sin_trend_dataset.parse_bool_series`)
- Logs use positive finite values only (0/negative → NaN)

## Measurement products and tag-specific surface products

Let `M = results/morphometrics/sin_trend/<run-id>`, `F = M/<frame-reference>`, `A = results/analysis/sin_trend/<run-id>/<frame-reference>`.

| Product | Location |
|---|---|
| Measurement config / manifest | `M/run_config.json`, `M/run_manifest.json` |
| Measurement CSV | `F/per_case/<directory_id>_vessel_segmentdata.csv` |
| Measurement pointwise arrays | `F/pointwise/<directory_id>_seg<segment_id>_idx<segment_index>_pointwise.npz` |
| Fixed tag settings and hashes | `M/surface_config_<tag>.json` |
| Surface CSV (primary also stored separately) | `F/surface/<tag>/per_case/<directory_id>_vessel_segmentdata.csv` |
| Surface status | `F/surface/<tag>/per_case/<directory_id>/<directory_id>_surface_status.json` |
| Surface normals cache NPZ | `M/normals_cache/<directory_id>/<tag>/` |
| Surface HTML | `F/surface/<tag>/figures/<directory_id>/` |
| Measurement or primary long CSV | `F/IXI_HH_sin_trend_morphometrics_long.csv` |
| Non-primary long CSV | `F/surface/<tag>/IXI_HH_sin_trend_morphometrics_long.csv` |
| Primary analysis outputs | `A/` |
| Non-primary analysis outputs | `A/surface/<tag>/` |

The surface stage does not overwrite measurement CSV / NPZ / config / status / manifest. Surface CSV keeps measurement columns and adds/updates the fields below. `core_periodic` / `core_all` keep measurement values; `core_inplane` reflects tag-specific direction decisions.

| Kind | Key columns / fields |
|---|---|
| Surface provenance | `surface_tag`, `surface_config_hash`, `surface_code_hash` |
| Source verification | `surface_source_measurement_csv_sha256`, `surface_source_pointwise_path`, `surface_source_pointwise_sha256` |
| Surface processing state | `surface_evaluation_status`, `surface_evaluation_reason`, `surface_evaluable`, `surface_constraint` |
| Direction and evaluable span | `R_out_surface`, `R_tan_surface`, `constraint_coverage`, `squared_displacement_coverage` |
| GOOD / BRIDGED breakdown | `n_eval_good`, `n_eval_bridged`, `good_coverage`, `bridged_coverage`, `squared_displacement_coverage_good`, `squared_displacement_coverage_bridged` |
| Relation to baseline tangent | `abs_T0_dot_n_mean`, `abs_T0_dot_n_median`, `R_out_upper_bound`, `R_out_within_upper_bound` |

`eval_normal_source` (NONE / GOOD / BRIDGED) and evaluation normals/masks are stored in the tag-specific normals cache NPZ. Surface status `per_case_csv_sha256` is the surface CSV hash; `measurement_csv_sha256` is the source measurement CSV hash.

`long_format_metadata.json` is stored beside the long CSV and records `aggregation_stage`, `surface_tag`, `surface_config_hash`, and whether anything is incomplete. Script 32 compares hashes of surface CSV/status/current measurement CSV and rejects stale products. If primary is only partially present, missing targets are not filled from measurement rows; incompleteness is recorded. Script 33 uses metadata that matches the actual input long CSV.
