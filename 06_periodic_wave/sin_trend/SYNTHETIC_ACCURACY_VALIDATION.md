# Synthetic accuracy validation

人工中心線での振幅・波長の精度検証と、区間長／採択が波長–内径関係に及ぼす影響の検証一式。  
実行入口: `39_synthetic_accuracy_validation.py`（実装は 36 / 42 / 40 を呼び出す）。

共通測定設定: \(D=1.4\,\mathrm{mm}\)（内腔）、\(ds=0.1\,\mathrm{mm}\)、両端 10 点 trim、\(N_{\mathrm{cyc,search}}=0.5\)、探索帯 `geometry_longwave_2D`（\(\lambda_{\min}=2D\)）、frame = baseline arclength、採択 `core_all`。

| # | Name | Script |
|---|------|--------|
| (1) | Accuracy across amplitudes, wavelengths, and phases | `36_measurement_validation.py` |
| (2) | Accuracy across background curvatures | `42_curvature_sensitivity_grid.py` |
| (3) | Accuracy across cycle counts | `40_length_selection_validation.py --suite A` |
| (4) | Procedure-induced bias in wavelength–diameter scaling | `40_length_selection_validation.py --suite B` |

---

## (1) Accuracy across amplitudes, wavelengths, and phases

既知の \(A,\lambda,\varphi\) と trim 後目標周期のもとで、測定パイプラインが波長・振幅を精度よく推定できるかを確認する（基本平面 + stress）。

| variant | \(n\) | \(A\) [mm] | \(\lambda\) [mm] | \(\varphi\) | \(N_{\mathrm{target}}\)（trim 後） | background |
|---------|------:|------------|------------------|-------------|-------------------------------------|------------|
| `pure_sinusoid` | 72 | 0.25, 0.5, 1.0, 2.0 | 10, 20, 30, 40, 50, 60 | \(0,\pi/4,\pi/2\) | 3.0 | straight |
| `curved_baseline_sinusoid` | 54 | 0.5, 1.0, 2.0 | 10, 20, 30, 40, 50, 60 | \(0,\pi/4,\pi/2\) | 3.0 | quadratic \(c=0.005\) |
| `short_coverage` | 24 | 0.5, 1.0 | 10, 20, 40 | \(0,\pi/2\) | 1.5, 2.0 | straight |
| `curved_short_coverage` | 6 | 1.0 | 10, 20, 40 | \(0\) | 1.5, 2.0 | quadratic \(c=0.005\) |

入力 \(n_{\mathrm{cycles}}\) は trim 後 \(N_{\mathrm{target}}\) になるよう逆算。noise = 0。合計 156 例。

```bash
python analysis_code/sin_trend/39_synthetic_accuracy_validation.py --part 1
# 同等: python analysis_code/sin_trend/36_measurement_validation.py --stress
```

出力: `results/analysis/sin_trend/<run_id>/synthetic_validation/`

---

## (2) Accuracy across background curvatures

本体 (1) の曲がった baseline は \(c=0.005\) のみ。同じ \(A,\lambda,\varphi\)、trim 後目標 3 周期の代表 18 条件について \(c=0.005\) と \(c=0.010\) を比較する（本体 156 には含めない）。

| 項目 | 値 |
|------|-----|
| \(A\) [mm] | 0.5, 1.0, 2.0 |
| \(\lambda\) [mm] | 10, 20, 40 |
| \(\varphi\) | \(0,\pi/2\) |
| \(N_{\mathrm{target}}\)（trim 後） | 3.0 |
| \(c\) | 0.005 vs 0.010 |
| 格子 | \(3\times3\times2=18\) × 2\(c\) = 36 |

```bash
python analysis_code/sin_trend/39_synthetic_accuracy_validation.py --part 2
# 同等: python analysis_code/sin_trend/42_curvature_sensitivity_grid.py
```

出力: `results/analysis/sin_trend/<run_id>/curvature_sensitivity_grid/`

---

## (3) Accuracy across cycle counts

同じ長い親曲線から長さの異なる窓を切り出し、真の周期数 \(N\) と位相を変えながら推定・採択を比較する。

| 項目 | 値 |
|------|-----|
| \(D\) [mm] | 1.4 |
| \(\lambda\) [mm] | 10, 20, 40 |
| \(A\) [mm] | 0.5, 1.0 |
| \(\varphi\) | \(0,\pi/4,\pi/2,3\pi/4\) |
| \(N_{\mathrm{true,target}}\) | 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0 |
| background | straight; quadratic \(c=0.005\) |
| noise | 0 |
| 格子サイズ | \(3\times2\times4\times9\times2=432\) |

親曲線生成の長さ許容は `suite_A.parent_length_atol_mm`（既定 0.05 mm）。

```bash
python analysis_code/sin_trend/39_synthetic_accuracy_validation.py --part 3 \
  --seed 0 --run-id lengthsel_A_<tag>
# 同等: python analysis_code/sin_trend/40_length_selection_validation.py --suite A ...
```

---

## (4) Procedure-induced bias in wavelength–diameter scaling

臨床の選別前集団から有効な \(D\) と \(L_{\mathrm{base}}\) を抽出し、真の \(\lambda\) を割り当てた人工集団で、採択前後の \(\log\lambda\)–\(\log D\) 傾きを Monte Carlo 比較する。臨床 \(L_{\mathrm{base}}\) は長さ分布の代理のみ。

| 項目 | 値 |
|------|-----|
| \(D,L\) | IXI-HH run_154 pre-selection プール（usable 11368） |
| \(A\) [mm] | 0.5, 1.0 |
| D–L pairing | `paired`, `shuffled_DL` |
| \(\lambda\) シナリオ | (i) \(\lambda=kD\)（\(k=17.4\), \(Z=0\)）; (ii) \(\lambda=kD\,e^{Z}\), \(Z\sim\mathcal N(0,0.2^2)\); (iii) (i) の独立 permute |
| background | straight のみ |
| MC | 既定 20 replicates × 200（本計算設定に依存; 例: 500×20） |

生成長の許容は `length_atol_mm + length_rtol * L0_target`（既定 0.05 mm）。許容内に入れない場合は生成失敗として記録する。

```bash
python analysis_code/sin_trend/39_synthetic_accuracy_validation.py --part 4 \
  --seed 0 --n-replicates 20 --n-per-replicate 500 --n-workers 4 \
  --run-id lengthsel_B_<tag>
# 同等: python analysis_code/sin_trend/40_length_selection_validation.py --suite B ...
```

(3)(4) の出力先: `results/analysis/sin_trend/length_selection_validation/<run_id>/`  
中断・再開・fingerprint の詳細は従来どおり（`--resume`、JSONL 追記、設定不一致時は停止）。

---

## 一括・smoke

```bash
python analysis_code/sin_trend/39_synthetic_accuracy_validation.py --part all --seed 0

python analysis_code/sin_trend/39_synthetic_accuracy_validation.py --part 3+4 --smoke --seed 0
```

## 対象外・限界

- 脳表メッシュ・法線・coverage・\(R_{\tan}\) 選別は再現しない（脳表方向は `38_surface_direction_validation.py`）。
- 最終 275 区間の選別全体や座屈の因果起源の検証ではない。
- 人工平面という理由では Predominantly tangential と分類しない。

## テスト

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover \
  -s analysis_code/sin_trend/tests \
  -p 'test_*validation*.py' -v
```
