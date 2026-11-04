# CONFIG_AND_METHODS (sin_trend)

## Measurement settings (inherited)

| Item | Value |
|---|---|
| voxel_size | 0.5 |
| removal | 10 points / end |
| baseline_degree | 2 |
| trend_degree | 1 (linear; no quadratic trend term) |
| search_band_mode | geometry_longwave_2D |
| diameter_wavelength_factor | 2.0 |
| N_cyc_search | 0.5 |
| q_fine | 8 |

## Cortex (legacy default operating values)

| Item | Value |
|---|---:|
| patch_r_ratio | 3.0 |
| stride | 10 |
| min_neighbors | 30 |
| gap_fill_size | 3 (legacy boolean closing; unused on the evaluation path) |
| bridge_max_gap_mm | 5.0 (max centerline gap between good endpoints; provisional) |
| bridge_axis_max_deg | 45.0 (unsigned angle between endpoint normal axes; provisional/unvalidated; distinct from coherence-equivalent angle) |
| coh_thr | 0.75 |
| r_min_mm / r_max_mm | 2.5 / 5.0 |
| constraint_coverage / squared_displacement_coverage (acceptance) | 0.50 / 0.50 |
| cortex_coverage | computed and stored only (diagnostic; not an interval acceptance gate) |
| cortex_coverage_thresh | row-metadata compatibility only; **unused** (not an active acceptance setting) |
| good_coverage / bridged_coverage | GOOD/BRIDGED counts inside mask_eval / total points (diagnostic; no acceptance threshold) |
| squared_displacement_coverage_good / squared_displacement_coverage_bridged | periodic displacement energy of those subsets / total interval energy (diagnostic; no acceptance threshold) |
| n_eval_good / n_eval_bridged | GOOD/BRIDGED counts among evaluation points |
| R_out / R_tan | \(R_{\mathrm{out}}=\) normal-component energy / periodic-displacement energy; \(R_{\mathrm{tan}}=1-R_{\mathrm{out}}\) (tangential) |
| R_out_thresh / R_tan_thresh | 0.25 / 0.75 (with surface angle θ, \(R_{\mathrm{out}}=\sin^2\theta\), θ_eq=30°; \(R_{\mathrm{tan}}=1-R_{\mathrm{out}}\)) |

Cortex-direction classification (`CLASSIFICATION_REVISION = v2_partialR2_050_c4`):

- Core and surface evaluation ok (finite R_out) and `constraint_coverage≥0.50` and `squared_displacement_coverage≥0.50`
- \(R_{\mathrm{out}}&lt;0.25\) (\(R_{\mathrm{tan}}&gt;0.75\)) → **Predominantly tangential**
- \(R_{\mathrm{out}}\ge0.25\) (\(R_{\mathrm{tan}}\le0.75\)) → **Substantial normal component**
- Legacy labels In-plane Buckling / Out-of-plane were retired in c4 (read-compatible only). c3 moved thresh 0.30→0.25; c2 removed the cortex gate

## New fit

1. Fixed U,V → 3D deflection field `Ucos, Usin, Vcos, Vsin`
2. Full Cartesian quadratic residualization → project onto fixed frame → trim
3. Joint two-component 8-coefficient OLS (`a1,b1,a2,b2,c1,d1,c2,d2`)
4. Trend-only is refit on Z alone
5. Best candidate = minimum SSE among valid candidates (no runner-up swap)

## response / latent

- `fit = response + trend` (goodness-of-fit comes from here)
- `latent` = pre-processed periodic deflection (not claimed as the true physical displacement)
- Primary cortex decision uses `d_latent`

## Amplitude

- `amp_comb_mm` (combined amplitude)
- `amp_major_mm = sigma1_mm`, `amp_minor_mm = sigma2_mm`

## Phase / wavelength (primary analysis)

- `fitting_coordinate` = `phase_coordinate` = **baseline_arclength** (sin/cos phase; not legacy `observed_arclength_s_trimmed_zero_start`)
- Extra linear trend `trend_coordinate` = **baseline_arclength** (`t = s_phase_trim − mean(s_phase_trim)`)
- Quadratic residualization Q `quadratic_detrend_coordinate` = observed_arclength (normalized `q`)
- `grid_length_basis` = trimmed_baseline_arclength (search length L_base)
- Primary wavelength: `lambda_hat = 2*pi/k_hat` (directly from baseline phase)
- coverage: `N_fit = L_base / lambda_hat`
- slenderness: `lambda_hat / (2 * radius_median_mm)`
- Legacy post-hoc `lambda * L_base/L_obs` is not computed
- These definitions are stored on measurement rows, `run_config`, and pointwise NPZ `scalars_json`

## New Core (`core_periodic`)

```
eligible AND fit_valid
AND finite(partial_R2) AND partial_R2 >= 0.50
AND coverage N >= 1.5  (L_base / lambda_hat)
AND NOT boundary_exact_low/high
```

Ordinary R2≥0.60, independent λ/D, amplitude/SNR/hr, ±20% uncertainty, and secondary peaks are not used for acceptance. `core_all` is an exact alias of `core_periodic`.

Classification revision: `CLASSIFICATION_REVISION=v2_partialR2_050_c4` (c4: Predominantly tangential / Substantial normal component; c3: R_out_thresh=0.25; c2: cortex gate removed).

## Stage separation and reproducibility

- **Measurement stage**: centerline processing, baseline/frame estimation, trim, periodic fitting, and measurement Core decision; writes measurement CSV and pointwise NPZ. With `--measure-only`, surface evaluation is `not_run`.
- **Surface stage**: reads saved `P`, `P_full`, `d_latent`, `T0` and post-trim radius; runs the same mesh crop, normal estimation, and classification as production. With `--surface-only`, fitting is not repeated. Estimators and `core_periodic` / `core_all` stay fixed; only surface classification is added into tag-specific products.
- **full (default in 31)**: after saving the measurement stage, run the surface stage. Primary products are also written to separate CSV/status paths.

Measurement `voxel_size` is fixed in the surface stage and cannot be overridden. Surface JSON may specify surface settings only. Settings are checked for finiteness and allowed ranges. Specifying only `R_tan_thresh` resolves `R_out_thresh=1−R_tan_thresh`; specifying both requires consistency.

`surface_config_<tag>.json` records the resolved settings, measurement config hash, and surface analysis code hash. Settings/code hashes for a given tag cannot change; a new tag is required (even with an overwrite flag). Example: save `{"r_min_mm": 3.0}` as `patch_min3` and compare to primary. Surface re-runs do not update measurement run_config / CSV / NPZ / status / manifest.

Search radius in mm is `r_patch=clip((1+patch_r_ratio)×r_trim, r_min_mm, r_max_mm)`, so default ratio=3 is 4× before clamps. Mesh-crop padding per axis is `max(10 voxel, ceil(r_patch_vox))` so larger search radii do not truncate the search ball. Anchor spacing along the centerline uses the saved centerline sample spacing.

Surface aggregation in 32 uses tag-specific status completion, compares SHA-256 of surface CSV/status and each row’s source measurement CSV against the current measurement CSV, and rejects mismatches or missing source measurement CSV. 33 reads metadata beside the chosen long CSV and isolates non-primary analysis outputs per tag.
