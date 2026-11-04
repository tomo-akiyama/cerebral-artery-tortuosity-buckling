# OUTPUT_SCHEMA（sin_trend）

正式全表: `IXI_HH_sin_trend_analysis_dataset.csv`

## 波長–半径 / 規格化波長–年齢 / 振幅–年齢に必要な列

| 用途 | 列 |
|---|---|
| ID | directory_id, segment_id, segment_index, IXI_ID, side |
| 年齢 | AGE/age（raw）, age_years |
| 半径（主=trim 後） | radius_median_mm, diameter_ref_mm (=2×trim 中央値) |
| 半径（full 参照） | radius_median_full_mm, radius_mean_full_mm, radius_std_full_mm |
| 波長 | lambda_hat_mm（baseline 弧長位相から直接推定）。lambda_hat_s_mm / lambda_baseline_eff_mm は互換 alias |
| 区間長 | L_base_mm（trim 対応 baseline 弧長）、L_trimmed_mm / L_full_mm（観測） |
| 細長比 | slenderness = lambda_hat_mm / (2 * radius_median_mm) |
| 座標 metadata | fitting/phase/trend=baseline_arclength; quadratic_detrend=observed_arclength; grid_length_basis=trimmed_baseline_arclength（行・run_config·NPZ scalars_json） |
| 規格化 | lambda_s_over_r, lambda_base_eff_over_r, amp_comb_over_L, amp_comb_over_r, log_* |
| 振幅 | amp_comb_mm, amp_major_mm (=sigma1_mm), amp_minor_mm (=sigma2_mm) |
| 部位 | mca_segment, label_status, mca_label_conflict |
| 採否 | core_periodic, core_all(=alias), core_inplane, reject_reasons |
| 適合 | R2, partial_R2, SSE_*_mm2, fit_status, fit_method |

## 部分表

- `MCA_M2M4_core_periodic.csv` / `MCA_M2M4_core_inplane.csv`
- `paired_wavelength_coordinate_intervals.csv`
- `paired_amplitude_definition_intervals.csv`

## 欠測・bool

- `"False"`/`"0"` は False（`sin_trend_dataset.parse_bool_series`）
- 対数は正の有限値のみ（0/負は NaN）

## 測定成果と tag 別表面成果

以下で `M = results/morphometrics/sin_trend/<run-id>`、`F = M/<frame-reference>`、`A = results/analysis/sin_trend/<run-id>/<frame-reference>` とする。

| 成果 | 保存先 |
|---|---|
| 測定設定 / manifest | `M/run_config.json`, `M/run_manifest.json` |
| 測定 CSV | `F/per_case/<directory_id>_vessel_segmentdata.csv` |
| 測定 pointwise 配列 | `F/pointwise/<directory_id>_seg<segment_id>_idx<segment_index>_pointwise.npz` |
| tag の固定設定と hash | `M/surface_config_<tag>.json` |
| 表面 CSV（primary も別保存） | `F/surface/<tag>/per_case/<directory_id>_vessel_segmentdata.csv` |
| 表面 status | `F/surface/<tag>/per_case/<directory_id>/<directory_id>_surface_status.json` |
| 表面法線 cache NPZ | `M/normals_cache/<directory_id>/<tag>/` |
| 表面 HTML | `F/surface/<tag>/figures/<directory_id>/` |
| 測定または primary の long CSV | `F/IXI_HH_sin_trend_morphometrics_long.csv` |
| 非 primary の long CSV | `F/surface/<tag>/IXI_HH_sin_trend_morphometrics_long.csv` |
| primary の解析出力 | `A/` |
| 非 primary の解析出力 | `A/surface/<tag>/` |

表面 stage は測定 CSV / NPZ / config / status / manifest を上書きしない。表面 CSV は測定列を保持し、以下を追加・更新する。`core_periodic` / `core_all` は測定値を保持し、`core_inplane` は tag ごとの方向判定を反映する。

| 種類 | 主要列・項目 |
|---|---|
| 表面の出典 | `surface_tag`, `surface_config_hash`, `surface_code_hash` |
| 生成元の照合 | `surface_source_measurement_csv_sha256`, `surface_source_pointwise_path`, `surface_source_pointwise_sha256` |
| 表面処理状態 | `surface_evaluation_status`, `surface_evaluation_reason`, `surface_evaluable`, `surface_constraint` |
| 方向と評価範囲 | `R_out_surface`, `R_tan_surface`, `constraint_coverage`, `squared_displacement_coverage` |
| GOOD / BRIDGED の内訳 | `n_eval_good`, `n_eval_bridged`, `good_coverage`, `bridged_coverage`, `squared_displacement_coverage_good`, `squared_displacement_coverage_bridged` |
| baseline 接線との関係 | `abs_T0_dot_n_mean`, `abs_T0_dot_n_median`, `R_out_upper_bound`, `R_out_within_upper_bound` |

`eval_normal_source`（NONE / GOOD / BRIDGED）と評価用法線・マスクは tag 別法線 cache NPZ に保存する。表面 status の `per_case_csv_sha256` は表面 CSV の hash を表し、`measurement_csv_sha256` は生成元測定 CSV の hash を表す。

`long_format_metadata.json` は long CSV と同じディレクトリに保存し、`aggregation_stage`、`surface_tag`、`surface_config_hash`、未完了の有無を記録する。32 は表面 CSV / status / 現測定 CSV の hash を照合し、古い成果を拒否する。primary が一部だけ存在する場合も、欠測対象を測定行で埋めず未完了を記録する。33 は実際の入力 long CSV に対応する metadata を使用する。
