# Length / selection validation report

## Scope and limits

- Validates wavelength estimation and periodic-fit acceptance (`core_all` / `core_periodic`) only.
- Does **not** reproduce brain-surface mesh, normal estimation, coverage, or R_tan selection.
- Therefore this artificial population does **not** reproduce the full clinical path to the final 275 predominantly-tangential intervals.
- Artificial planar curves are **not** labeled Predominantly tangential merely for being planar.
- Search lower cycle count remains `N_cyc_search=0.5` (not changed to 1.5).
- Acceptance still requires estimated cycles `coverage_hat >= 1.5` via production `classify_row`.

- Elapsed: 3598.7 s; rows: 120000
- Config: [local-path-removed] 系統解剖研究室/10 Geometric analysis 220401-/cerebral_artery_bucling_project/analysis_code/sin_trend/configs/length_selection_validation_default.json
- Config sha256: f7c72e031bd83aeeb06c3b80da4410ec09aac1dca61ea870b93e612659d3b088

## Length definitions

- `L0_true_trimmed_mm`: true known-baseline arc length after end trim (from generation ξ).
- `L_base_mm`: estimated quadratic baseline arc length after trim (production measure).
- `L_trimmed_mm`: observed centerline arc length after trim.
- `N_true = Δξ / lambda_true` (not from estimated baseline / lambda_hat).
- `N_hat = L_base_mm / lambda_hat_mm` (production coverage_hat).

## Clinical D/L pool (Suite B inputs)

- CSV: `[local-path-removed] 系統解剖研究室/10 Geometric analysis 220401-/cerebral_artery_bucling_project/results/analysis/sin_trend/run_154_baseline_v2_r0/baseline/IXI_HH_sin_trend_analysis_dataset.csv`
- sha256: `185eea914ea61c5c32376d7c49fab90b249616d7188df682e3398bc15aed4d0f`
- n_total=11808, n_usable=11368
- drop_reasons: {'radius_missing': 440, 'L_base_missing': 440, 'radius_nonpositive': 0, 'L_base_nonpositive': 0}
- `L_base_mm` is itself an estimate; used only as a proxy distribution for artificial true length.
- Population is pre-selection (not filtered to Core / surface).

## Suite A summary (head)

(empty)

## Suite B replicate slopes (head)

 replicate                                scenario  dl_pairing                              relation_label   n  slope_log_lambda_vs_log_D slope_status  acceptance_rate
         0 independent_perm_of_proportional_sigma0      paired              0_all_planned_true_lambda_vs_D 500                  -0.088546           ok            0.170
         0 independent_perm_of_proportional_sigma0      paired            1_all_generated_true_lambda_vs_D 500                  -0.088546           ok            0.170
         0 independent_perm_of_proportional_sigma0      paired                 2_accepted_true_lambda_vs_D  85                  -0.078026           ok            0.170
         0 independent_perm_of_proportional_sigma0      paired 3_preaccept_fit_valid_estimated_lambda_vs_D 473                   0.245501           ok            0.170
         0 independent_perm_of_proportional_sigma0      paired            4_accepted_estimated_lambda_vs_D  85                  -0.077343           ok            0.170
         0 independent_perm_of_proportional_sigma0      paired              0_all_planned_true_lambda_vs_D 500                  -0.049752           ok            0.210
         0 independent_perm_of_proportional_sigma0      paired            1_all_generated_true_lambda_vs_D 500                  -0.049752           ok            0.210
         0 independent_perm_of_proportional_sigma0      paired                 2_accepted_true_lambda_vs_D 105                   0.149424           ok            0.210
         0 independent_perm_of_proportional_sigma0      paired 3_preaccept_fit_valid_estimated_lambda_vs_D 477                   0.254270           ok            0.210
         0 independent_perm_of_proportional_sigma0      paired            4_accepted_estimated_lambda_vs_D 105                   0.145736           ok            0.210
         0 independent_perm_of_proportional_sigma0 shuffled_DL              0_all_planned_true_lambda_vs_D 500                  -0.031941           ok            0.204
         0 independent_perm_of_proportional_sigma0 shuffled_DL            1_all_generated_true_lambda_vs_D 500                  -0.031941           ok            0.204
         0 independent_perm_of_proportional_sigma0 shuffled_DL                 2_accepted_true_lambda_vs_D 102                  -0.053152           ok            0.204
         0 independent_perm_of_proportional_sigma0 shuffled_DL 3_preaccept_fit_valid_estimated_lambda_vs_D 471                   0.072713           ok            0.204
         0 independent_perm_of_proportional_sigma0 shuffled_DL            4_accepted_estimated_lambda_vs_D 102                  -0.051729           ok            0.204
         0 independent_perm_of_proportional_sigma0 shuffled_DL              0_all_planned_true_lambda_vs_D 500                  -0.003581           ok            0.194
         0 independent_perm_of_proportional_sigma0 shuffled_DL            1_all_generated_true_lambda_vs_D 500                  -0.003581           ok            0.194
         0 independent_perm_of_proportional_sigma0 shuffled_DL                 2_accepted_true_lambda_vs_D  97                  -0.066624           ok            0.194
         0 independent_perm_of_proportional_sigma0 shuffled_DL 3_preaccept_fit_valid_estimated_lambda_vs_D 482                   0.087668           ok            0.194
         0 independent_perm_of_proportional_sigma0 shuffled_DL            4_accepted_estimated_lambda_vs_D  97                  -0.068104           ok            0.194
         0                     proportional_sigma0      paired              0_all_planned_true_lambda_vs_D 500                   1.000000           ok            0.196
         0                     proportional_sigma0      paired            1_all_generated_true_lambda_vs_D 500                   1.000000           ok            0.196
         0                     proportional_sigma0      paired                 2_accepted_true_lambda_vs_D  98                   1.000000           ok            0.196
         0                     proportional_sigma0      paired 3_preaccept_fit_valid_estimated_lambda_vs_D 473                   0.664561           ok            0.196
         0                     proportional_sigma0      paired            4_accepted_estimated_lambda_vs_D  98                   0.992593           ok            0.196
         0                     proportional_sigma0      paired              0_all_planned_true_lambda_vs_D 500                   1.000000           ok            0.222
         0                     proportional_sigma0      paired            1_all_generated_true_lambda_vs_D 500                   1.000000           ok            0.222
         0                     proportional_sigma0      paired                 2_accepted_true_lambda_vs_D 111                   1.000000           ok            0.222
         0                     proportional_sigma0      paired 3_preaccept_fit_valid_estimated_lambda_vs_D 476                   0.702689           ok            0.222
         0                     proportional_sigma0      paired            4_accepted_estimated_lambda_vs_D 111                   0.993411           ok            0.222
         0                     proportional_sigma0 shuffled_DL              0_all_planned_true_lambda_vs_D 500                   1.000000           ok            0.184
         0                     proportional_sigma0 shuffled_DL            1_all_generated_true_lambda_vs_D 500                   1.000000           ok            0.184
         0                     proportional_sigma0 shuffled_DL                 2_accepted_true_lambda_vs_D  92                   1.000000           ok            0.184
         0                     proportional_sigma0 shuffled_DL 3_preaccept_fit_valid_estimated_lambda_vs_D 470                   0.746233           ok            0.184
         0                     proportional_sigma0 shuffled_DL            4_accepted_estimated_lambda_vs_D  92                   0.999073           ok            0.184
         0                     proportional_sigma0 shuffled_DL              0_all_planned_true_lambda_vs_D 500                   1.000000           ok            0.184
         0                     proportional_sigma0 shuffled_DL            1_all_generated_true_lambda_vs_D 500                   1.000000           ok            0.184
         0                     proportional_sigma0 shuffled_DL                 2_accepted_true_lambda_vs_D  92                   1.000000           ok            0.184
         0                     proportional_sigma0 shuffled_DL 3_preaccept_fit_valid_estimated_lambda_vs_D 482                   0.760719           ok            0.184
         0                     proportional_sigma0 shuffled_DL            4_accepted_estimated_lambda_vs_D  92                   1.000670           ok            0.184

## Suite B Monte Carlo slope percentiles

                               scenario  dl_pairing  A_true_mm background_name                              relation_label  n_replicates  n_replicates_with_finite_slope  slope_median  slope_p05  slope_p95                                                                    percentile_range_note  acceptance_rate_median                                                    denominator_note
independent_perm_of_proportional_sigma0      paired        0.5        straight              0_all_planned_true_lambda_vs_D            20                              20     -0.008601  -0.091701   0.056165 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.201  all planned inputs, including generation failures; L is the target
independent_perm_of_proportional_sigma0      paired        0.5        straight            1_all_generated_true_lambda_vs_D            20                              20     -0.008601  -0.091701   0.056165 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.201                                               generation_status==ok
independent_perm_of_proportional_sigma0      paired        0.5        straight                 2_accepted_true_lambda_vs_D            20                              20      0.054993  -0.102743   0.345408 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.201                                       generation_ok AND core_all==1
independent_perm_of_proportional_sigma0      paired        0.5        straight 3_preaccept_fit_valid_estimated_lambda_vs_D            20                              20      0.254012   0.150504   0.402335 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.201 generation_ok AND execution_ok AND fit_valid==1 (not yet core gate)
independent_perm_of_proportional_sigma0      paired        0.5        straight            4_accepted_estimated_lambda_vs_D            20                              20      0.055641  -0.103796   0.343758 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.201                                       generation_ok AND core_all==1
independent_perm_of_proportional_sigma0      paired        1.0        straight              0_all_planned_true_lambda_vs_D            20                              20      0.005502  -0.085271   0.057564 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.206  all planned inputs, including generation failures; L is the target
independent_perm_of_proportional_sigma0      paired        1.0        straight            1_all_generated_true_lambda_vs_D            20                              20      0.005502  -0.085271   0.057564 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.206                                               generation_status==ok
independent_perm_of_proportional_sigma0      paired        1.0        straight                 2_accepted_true_lambda_vs_D            20                              20      0.046432  -0.203249   0.291974 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.206                                       generation_ok AND core_all==1
independent_perm_of_proportional_sigma0      paired        1.0        straight 3_preaccept_fit_valid_estimated_lambda_vs_D            20                              20      0.258748   0.092894   0.388104 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.206 generation_ok AND execution_ok AND fit_valid==1 (not yet core gate)
independent_perm_of_proportional_sigma0      paired        1.0        straight            4_accepted_estimated_lambda_vs_D            20                              20      0.044888  -0.200235   0.289937 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.206                                       generation_ok AND core_all==1
independent_perm_of_proportional_sigma0 shuffled_DL        0.5        straight              0_all_planned_true_lambda_vs_D            20                              20     -0.006437  -0.064909   0.041251 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.208  all planned inputs, including generation failures; L is the target
independent_perm_of_proportional_sigma0 shuffled_DL        0.5        straight            1_all_generated_true_lambda_vs_D            20                              20     -0.006437  -0.064909   0.041251 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.208                                               generation_status==ok
independent_perm_of_proportional_sigma0 shuffled_DL        0.5        straight                 2_accepted_true_lambda_vs_D            20                              20      0.036273  -0.134967   0.156707 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.208                                       generation_ok AND core_all==1
independent_perm_of_proportional_sigma0 shuffled_DL        0.5        straight 3_preaccept_fit_valid_estimated_lambda_vs_D            20                              20      0.103354  -0.006799   0.172702 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.208 generation_ok AND execution_ok AND fit_valid==1 (not yet core gate)
independent_perm_of_proportional_sigma0 shuffled_DL        0.5        straight            4_accepted_estimated_lambda_vs_D            20                              20      0.035896  -0.132572   0.158438 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.208                                       generation_ok AND core_all==1
independent_perm_of_proportional_sigma0 shuffled_DL        1.0        straight              0_all_planned_true_lambda_vs_D            20                              20      0.004964  -0.046191   0.050444 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.196  all planned inputs, including generation failures; L is the target
independent_perm_of_proportional_sigma0 shuffled_DL        1.0        straight            1_all_generated_true_lambda_vs_D            20                              20      0.004964  -0.046191   0.050444 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.196                                               generation_status==ok
independent_perm_of_proportional_sigma0 shuffled_DL        1.0        straight                 2_accepted_true_lambda_vs_D            20                              20      0.024991  -0.119978   0.160079 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.196                                       generation_ok AND core_all==1
independent_perm_of_proportional_sigma0 shuffled_DL        1.0        straight 3_preaccept_fit_valid_estimated_lambda_vs_D            20                              20      0.072183  -0.017500   0.170478 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.196 generation_ok AND execution_ok AND fit_valid==1 (not yet core gate)
independent_perm_of_proportional_sigma0 shuffled_DL        1.0        straight            4_accepted_estimated_lambda_vs_D            20                              20      0.025358  -0.117791   0.159061 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.196                                       generation_ok AND core_all==1
                    proportional_sigma0      paired        0.5        straight              0_all_planned_true_lambda_vs_D            20                              20      1.000000   1.000000   1.000000 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.197  all planned inputs, including generation failures; L is the target
                    proportional_sigma0      paired        0.5        straight            1_all_generated_true_lambda_vs_D            20                              20      1.000000   1.000000   1.000000 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.197                                               generation_status==ok
                    proportional_sigma0      paired        0.5        straight                 2_accepted_true_lambda_vs_D            20                              20      1.000000   1.000000   1.000000 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.197                                       generation_ok AND core_all==1
                    proportional_sigma0      paired        0.5        straight 3_preaccept_fit_valid_estimated_lambda_vs_D            20                              20      0.627437   0.531790   0.760881 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.197 generation_ok AND execution_ok AND fit_valid==1 (not yet core gate)
                    proportional_sigma0      paired        0.5        straight            4_accepted_estimated_lambda_vs_D            20                              20      0.996480   0.992418   1.001529 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.197                                       generation_ok AND core_all==1
                    proportional_sigma0      paired        1.0        straight              0_all_planned_true_lambda_vs_D            20                              20      1.000000   1.000000   1.000000 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.219  all planned inputs, including generation failures; L is the target
                    proportional_sigma0      paired        1.0        straight            1_all_generated_true_lambda_vs_D            20                              20      1.000000   1.000000   1.000000 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.219                                               generation_status==ok
                    proportional_sigma0      paired        1.0        straight                 2_accepted_true_lambda_vs_D            20                              20      1.000000   1.000000   1.000000 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.219                                       generation_ok AND core_all==1
                    proportional_sigma0      paired        1.0        straight 3_preaccept_fit_valid_estimated_lambda_vs_D            20                              20      0.639021   0.529759   0.749025 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.219 generation_ok AND execution_ok AND fit_valid==1 (not yet core gate)
                    proportional_sigma0      paired        1.0        straight            4_accepted_estimated_lambda_vs_D            20                              20      0.997139   0.993654   1.004017 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.219                                       generation_ok AND core_all==1
                    proportional_sigma0 shuffled_DL        0.5        straight              0_all_planned_true_lambda_vs_D            20                              20      1.000000   1.000000   1.000000 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.203  all planned inputs, including generation failures; L is the target
                    proportional_sigma0 shuffled_DL        0.5        straight            1_all_generated_true_lambda_vs_D            20                              20      1.000000   1.000000   1.000000 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.203                                               generation_status==ok
                    proportional_sigma0 shuffled_DL        0.5        straight                 2_accepted_true_lambda_vs_D            20                              20      1.000000   1.000000   1.000000 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.203                                       generation_ok AND core_all==1
                    proportional_sigma0 shuffled_DL        0.5        straight 3_preaccept_fit_valid_estimated_lambda_vs_D            20                              20      0.766232   0.695659   0.850405 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.203 generation_ok AND execution_ok AND fit_valid==1 (not yet core gate)
                    proportional_sigma0 shuffled_DL        0.5        straight            4_accepted_estimated_lambda_vs_D            20                              20      0.999327   0.996974   1.003025 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.203                                       generation_ok AND core_all==1
                    proportional_sigma0 shuffled_DL        1.0        straight              0_all_planned_true_lambda_vs_D            20                              20      1.000000   1.000000   1.000000 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.200  all planned inputs, including generation failures; L is the target
                    proportional_sigma0 shuffled_DL        1.0        straight            1_all_generated_true_lambda_vs_D            20                              20      1.000000   1.000000   1.000000 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.200                                               generation_status==ok
                    proportional_sigma0 shuffled_DL        1.0        straight                 2_accepted_true_lambda_vs_D            20                              20      1.000000   1.000000   1.000000 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.200                                       generation_ok AND core_all==1
                    proportional_sigma0 shuffled_DL        1.0        straight 3_preaccept_fit_valid_estimated_lambda_vs_D            20                              20      0.742370   0.661122   0.799034 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.200 generation_ok AND execution_ok AND fit_valid==1 (not yet core gate)
                    proportional_sigma0 shuffled_DL        1.0        straight            4_accepted_estimated_lambda_vs_D            20                              20      0.999573   0.995458   1.002366 Monte Carlo distribution under the configured generation settings; not a clinical 95% CI                   0.200                                       generation_ok AND core_all==1

Percentile ranges are Monte Carlo distributions under the configured generation settings, not clinical coefficient 95% CIs.

## Execution status

- Status: complete
- Generation failures: 0; measurement exceptions: 0
- Crosscheck with 36: True
- Completed cases are durably recorded in case_results.jsonl.
- Rejected periodic fits are valid results, not execution failures.
- Cases with unknown achieved N_true are excluded from the achieved-N denominator and counted separately; their target cycle count is not substituted for truth.
