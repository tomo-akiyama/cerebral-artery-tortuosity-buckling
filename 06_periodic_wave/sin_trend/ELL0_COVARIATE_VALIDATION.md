# Validation (5): estimated baseline length ℓ̂0 as covariate

Standalone from accuracy validations (1)–(4). Entry point:

```bash
python analysis_code/sin_trend/43_ell0_covariate_scaling_validation.py --part 5a+clinical
python analysis_code/sin_trend/43_ell0_covariate_scaling_validation.py --part 5b+5c --n-workers 4
```

- **5a**: reanalyze Suite B `all_cases.csv` (no new generation); uses pipeline `L_base_mm` as ℓ̂0.
- **5b**: new sims with log-normal scatter σ ∈ {σ_real, 0.3, 0.4}, A=1.0 mm, paired / shuffled_DL.
- **5c**: length-determined λ = ℓ0 / n, n ~ U(1.5, 3).
- **clinical**: 275 predominantly tangential regressions ± ℓ̂0, cluster-robust SE.

Outputs: `results/analysis/sin_trend/ell0_covariate_validation/<run_id>/`.
