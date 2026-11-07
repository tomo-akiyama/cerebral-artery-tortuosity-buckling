# Synthetic accuracy validation

End-to-end checks of amplitude/wavelength accuracy on synthetic centerlines, and of how interval length / acceptance affect the wavelength–diameter relation.  
Entry point: `11_synthetic_accuracy_validation.py` (implementation calls 36 / 42 / 40).

Shared measurement settings: \(D=1.4\,\mathrm{mm}\) (lumen), \(ds=0.1\,\mathrm{mm}\), 10-point trim at each end, \(N_{\mathrm{cyc,search}}=0.5\), search band `geometry_longwave_2D` (\(\lambda_{\min}=2D\)), frame = baseline arclength, acceptance `core_all`.

| # | Name | Script |
|---|------|--------|
| (1) | Accuracy across amplitudes, wavelengths, and phases | `10_measurement_validation.py` |
| (2) | Accuracy across background curvatures | `14_curvature_sensitivity_grid.py` |
| (3) | Accuracy across cycle counts | `12_length_selection_validation.py --suite A` |
| (4) | Procedure-induced bias in wavelength–diameter scaling | `12_length_selection_validation.py --suite B` |

---

## (1) Accuracy across amplitudes, wavelengths, and phases

With known \(A,\lambda,\varphi\) and a post-trim target period count, check that the measurement pipeline recovers wavelength and amplitude accurately (base planes + stress).

| variant | \(n\) | \(A\) [mm] | \(\lambda\) [mm] | \(\varphi\) | \(N_{\mathrm{target}}\) (after trim) | background |
|---------|------:|------------|------------------|-------------|-------------------------------------|------------|
| `pure_sinusoid` | 72 | 0.25, 0.5, 1.0, 2.0 | 10, 20, 30, 40, 50, 60 | \(0,\pi/4,\pi/2\) | 3.0 | straight |
| `curved_baseline_sinusoid` | 54 | 0.5, 1.0, 2.0 | 10, 20, 30, 40, 50, 60 | \(0,\pi/4,\pi/2\) | 3.0 | quadratic \(c=0.005\) |
| `short_coverage` | 24 | 0.5, 1.0 | 10, 20, 40 | \(0,\pi/2\) | 1.5, 2.0 | straight |
| `curved_short_coverage` | 6 | 1.0 | 10, 20, 40 | \(0\) | 1.5, 2.0 | quadratic \(c=0.005\) |

Input \(n_{\mathrm{cycles}}\) is back-calculated so the post-trim count equals \(N_{\mathrm{target}}\). noise = 0. Total 156 cases.

```bash
python 06_periodic_wave/sin_trend/11_synthetic_accuracy_validation.py --part 1
# equivalent: python 06_periodic_wave/sin_trend/10_measurement_validation.py --stress
```

Output: `results/analysis/sin_trend/<run_id>/synthetic_validation/`

---

## (2) Accuracy across background curvatures

The main suite (1) uses only \(c=0.005\) for curved baselines. For the same representative 18 conditions of \(A,\lambda,\varphi\) with post-trim target 3 cycles, compare \(c=0.005\) vs \(c=0.010\) (not included in the main 156).

| Item | Value |
|------|-----|
| \(A\) [mm] | 0.5, 1.0, 2.0 |
| \(\lambda\) [mm] | 10, 20, 40 |
| \(\varphi\) | \(0,\pi/2\) |
| \(N_{\mathrm{target}}\) (after trim) | 3.0 |
| \(c\) | 0.005 vs 0.010 |
| Grid | \(3\times3\times2=18\) × 2\(c\) = 36 |

```bash
python 06_periodic_wave/sin_trend/11_synthetic_accuracy_validation.py --part 2
# equivalent: python 06_periodic_wave/sin_trend/14_curvature_sensitivity_grid.py
```

Output: `results/analysis/sin_trend/<run_id>/curvature_sensitivity_grid/`

---

## (3) Accuracy across cycle counts

From the same long parent curve, cut windows of different lengths and compare estimation/acceptance while varying true cycle count \(N\) and phase.

| Item | Value |
|------|-----|
| \(D\) [mm] | 1.4 |
| \(\lambda\) [mm] | 10, 20, 40 |
| \(A\) [mm] | 0.5, 1.0 |
| \(\varphi\) | \(0,\pi/4,\pi/2,3\pi/4\) |
| \(N_{\mathrm{true,target}}\) | 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0 |
| background | straight; quadratic \(c=0.005\) |
| noise | 0 |
| Grid size | \(3\times2\times4\times9\times2=432\) |

Parent-curve length tolerance uses `suite_A.parent_length_atol_mm` (default 0.05 mm).

```bash
python 06_periodic_wave/sin_trend/11_synthetic_accuracy_validation.py --part 3 \
  --seed 0 --run-id lengthsel_A_<tag>
# equivalent: python 06_periodic_wave/sin_trend/12_length_selection_validation.py --suite A ...
```

---

## (4) Procedure-induced bias in wavelength–diameter scaling

From the clinical pre-selection pool, extract usable \(D\) and \(L_{\mathrm{base}}\), assign true \(\lambda\) in a synthetic cohort, and Monte Carlo–compare pre- vs post-acceptance \(\log\lambda\)–\(\log D\) slopes. Clinical \(L_{\mathrm{base}}\) is used only as a length-distribution proxy.

| Item | Value |
|------|-----|
| \(D,L\) | IXI-HH run_154 pre-selection pool (usable 11368) |
| \(A\) [mm] | 0.5, 1.0 |
| D–L pairing | `paired`, `shuffled_DL` |
| \(\lambda\) scenarios | (i) \(\lambda=kD\) (\(k=17.4\), \(Z=0\)); (ii) \(\lambda=kD\,e^{Z}\), \(Z\sim\mathcal N(0,0.2^2)\); (iii) independent permute of (i) |
| background | straight only |
| MC | default 20 replicates × 200 (depends on production settings; e.g. 500×20) |

Generation-length tolerance is `length_atol_mm + length_rtol * L0_target` (default 0.05 mm). Failures outside tolerance are recorded as generation failures.

```bash
python 06_periodic_wave/sin_trend/11_synthetic_accuracy_validation.py --part 4 \
  --seed 0 --n-replicates 20 --n-per-replicate 500 --n-workers 4 \
  --run-id lengthsel_B_<tag>
# equivalent: python 06_periodic_wave/sin_trend/12_length_selection_validation.py --suite B ...
```

Outputs for (3)(4): `results/analysis/sin_trend/length_selection_validation/<run_id>/`  
Interrupt/resume/fingerprint behavior is unchanged (`--resume`, JSONL append, stop on config mismatch).

---

## Batch / smoke

```bash
python 06_periodic_wave/sin_trend/11_synthetic_accuracy_validation.py --part all --seed 0

python 06_periodic_wave/sin_trend/11_synthetic_accuracy_validation.py --part 3+4 --smoke --seed 0
```

## Out of scope / limits

- Does not reproduce cortex mesh, normals, coverage, or \(R_{\tan}\) selection (surface direction: `09_surface_direction_validation.py`).
- Not a validation of the full selection of the final 275 intervals, nor of the causal origin of buckling.
- Does not classify as Predominantly tangential merely because the synthetic surface is planar.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover \
  -s 06_periodic_wave/sin_trend/tests \
  -p 'test_*validation*.py' -v
```
