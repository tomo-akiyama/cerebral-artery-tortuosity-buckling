# CONFIG_AND_METHODS（sin_trend）

## 測定設定（継承）

| 項目 | 値 |
|---|---|
| voxel_size | 0.5 |
| removal | 10 点/端 |
| baseline_degree | 2 |
| trend_degree | 1（線形; 二次 trend は追加しない） |
| search_band_mode | geometry_longwave_2D |
| diameter_wavelength_factor | 2.0 |
| N_cyc_search | 0.5 |
| q_fine | 8 |

## 脳表（旧 default 実値）

| 項目 | 値 |
|---|---:|
| patch_r_ratio | 3.0 |
| stride | 10 |
| min_neighbors | 30 |
| gap_fill_size | 3（旧 boolean closing。評価経路では未使用） |
| bridge_max_gap_mm | 5.0（良好両端間の中心線距離上限; 暫定） |
| bridge_axis_max_deg | 45.0（両端法線軸の符号無視角度差; 暫定・未検証。coherence 換算角とは別） |
| coh_thr | 0.75 |
| r_min_mm / r_max_mm | 2.5 / 5.0 |
| constraint_coverage / squared_displacement_coverage（採否） | 0.50 / 0.50 |
| cortex_coverage | 計算・保存のみ（確認用。区間全体の採否ゲートではない） |
| cortex_coverage_thresh | 行 metadata 互換用・**未使用**（有効な採否設定ではない） |
| good_coverage / bridged_coverage | mask_eval 内 GOOD・BRIDGED 点数／全点数（確認用・採否閾値なし） |
| squared_displacement_coverage_good / squared_displacement_coverage_bridged | 同上部分集合の周期変位二乗和／全区間二乗和（確認用・採否閾値なし） |
| n_eval_good / n_eval_bridged | 評価対象内の GOOD・BRIDGED 点数 |
| R_out / R_tan | \(R_{\mathrm{out}}=\)法線成分二乗和／周期変位二乗和、\(R_{\mathrm{tan}}=1-R_{\mathrm{out}}\)（接線成分） |
| R_out_thresh / R_tan_thresh | 0.25 / 0.75（面との角 θ で \(R_{\mathrm{out}}=\sin^2\theta\)、θ_eq=30°。\(R_{\mathrm{tan}}=1-R_{\mathrm{out}}\) から導出） |

脳表方向分類（`CLASSIFICATION_REVISION = v2_partialR2_050_c4`）:

- Core かつ surface 評価 ok（有限 R_out）かつ `constraint_coverage≥0.50` かつ `squared_displacement_coverage≥0.50`
- \(R_{\mathrm{out}}&lt;0.25\)（\(R_{\mathrm{tan}}&gt;0.75\)）→ **Predominantly tangential**
- \(R_{\mathrm{out}}\ge0.25\)（\(R_{\mathrm{tan}}\le0.75\)）→ **Substantial normal component**
- 旧ラベル In-plane Buckling / Out-of-plane は c4 で廃止（読取互換のみ）。c3 で thresh 0.30→0.25、c2 で cortex ゲート廃止

## 新フィット

1. 固定 U,V で `Ucos, Usin, Vcos, Vsin` の 3D 偏位場
2. full Cartesian 二次残差化 → 固定 frame 射影 → trim
3. 二成分同時 8 係数 OLS（`a1,b1,a2,b2,c1,d1,c2,d2`）
4. trend-only は Z だけで再フィット
5. 最良候補 = 有効候補の最小 SSE（次善乗り換えなし）

## response / latent

- `fit = response + trend`（適合度はここから）
- `latent` = 加工前の周期偏位（真の物理変位とは呼ばない）
- 脳表主判定は `d_latent`

## 振幅

- `amp_comb_mm`（合成振幅）
- `amp_major_mm = sigma1_mm`、`amp_minor_mm = sigma2_mm`

## 位相・波長（主解析）

- `fitting_coordinate` = `phase_coordinate` = **baseline_arclength**（sin/cos の位相。旧 `observed_arclength_s_trimmed_zero_start` ではない）
- 追加線形トレンド `trend_coordinate` = **baseline_arclength**（`t = s_phase_trim − mean(s_phase_trim)`）
- 二次残差化 Q `quadratic_detrend_coordinate` = observed_arclength（正規化 `q`）
- `grid_length_basis` = trimmed_baseline_arclength（探索長 L_base）
- 主波長: `lambda_hat = 2*pi/k_hat`（baseline 位相から直接）
- coverage: `N_fit = L_base / lambda_hat`
- slenderness: `lambda_hat / (2 * radius_median_mm)`
- 旧 post-hoc `lambda * L_base/L_obs` は計算しない
- 上記定義は測定行・`run_config`・pointwise NPZ の `scalars_json` に保存する

## 新 Core（core_periodic）

```
eligible AND fit_valid
AND finite(partial_R2) AND partial_R2 >= 0.50
AND coverage N >= 1.5  （L_base / lambda_hat）
AND NOT boundary_exact_low/high
```

通常 R2>=0.60・独立 λ/D・振幅/SNR/hr・±20% 不確かさ・副ピークは採否に使わない。`core_all` は `core_periodic` の完全一致 alias。

分類版: `CLASSIFICATION_REVISION=v2_partialR2_050_c4`（c4: Predominantly tangential / Substantial normal component；c3: R_out_thresh=0.25；c2: cortex ゲート撤廃）。

## ステージ分離と再現性

- **測定 stage**: 中心線処理、baseline / frame 推定、trim、周期 fitting、測定 Core 判定までを行い、測定 CSV と pointwise NPZ を保存する。`--measure-only` では表面評価は `not_run`。
- **表面 stage**: 保存済みの `P`、`P_full`、`d_latent`、`T0` と trim 後半径を読み、本番と同じ mesh 切り出し・法線推定・分類を行う。`--surface-only` で fitting を繰り返さない。推定量と `core_periodic` / `core_all` は固定し、表面分類だけを tag 別成果へ追加する。
- **full（31 の既定）**: 測定 stage の保存後に表面 stage を実行する。primary も別 CSV / status へ保存する。

測定設定の `voxel_size` は表面 stage でも固定であり、異なる値を override できない。表面 JSON には表面設定のみを指定できる。設定値は有限性・許容範囲を検査する。`R_tan_thresh` のみを指定した場合も `R_out_thresh=1−R_tan_thresh` を解決し、両方を指定する場合は整合性を要求する。

`surface_config_<tag>.json` は、解決後の設定、測定 config hash、表面解析コード hash を記録する。同じ tag の設定・コード hash は変更できず、変更時は新しい tag が必要（上書き指定でも回避しない）。例: `{"r_min_mm": 3.0}` を `patch_min3` として保存し、primary と比較する。表面再実行は測定 run_config / CSV / NPZ / status / manifest を更新しない。

探索半径は mm 単位で `r_patch=clip((1+patch_r_ratio)×r_trim, r_min_mm, r_max_mm)`。したがって既定の ratio=3 は上限・下限適用前の 4 倍である。mesh 切り出し余白は各軸で `max(10 voxel, ceil(r_patch_vox))` とし、探索半径を増やした場合も探索球を切断しない。中心線上のアンカー間隔には保存された中心線の実際の標本間隔を用いる。

32 の表面集約は tag 別 status の完了状況を用い、表面 CSV と status の SHA-256、および各行の生成元測定 CSV SHA-256 と現測定 CSV を照合する。不一致・元測定 CSV 欠測は拒否する。33 は選択した long CSV の隣の metadata を読み、非 primary の解析出力を tag ごとに分離する。
