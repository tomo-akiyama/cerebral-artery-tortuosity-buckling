# Cerebral artery tortuosity and growth-driven buckling

Custom Python pipelines and **processed measurements** accompanying the manuscript:

> Akiyama T., Yoshimoto K., Miura T.  
> *Growth-driven mechanical buckling as a mechanism of age-related tortuosity in human cerebral arteries.*

This repository does **not** redistribute IXI MRI volumes. Download IXI separately, then run the
preprocessing stages locally if you need to regenerate meshes/centerlines.

Repository (intended): https://github.com/tomo-akiyama/cerebral-artery-tortuosity-buckling

## Licenses

| Content | License |
|---------|---------|
| Source code | MIT (`LICENSE`) |
| IXI-derived CSVs (`data/participants`, `data/m1`, `data/periodic_wave`) | CC BY-SA 3.0 (`LICENSE-DATA`) |
| Simulation outputs & analysis summaries (`data/simulation`, `data/statistics`, `data/validation`) | CC BY 4.0 (`LICENSE-DATA`) |

## Environment

- Python **3.10**
- Key packages: NumPy, SciPy, pandas, statsmodels, scikit-image, VTK, **VMTK 1.5.0**, PyMeshLab, PyVista, ANTsPy, brainextractor
- External (not in Conda env): **MATLAB + SPM12/CAT12 12.9**, **3D Slicer 5.6**

```bash
conda env create -f environment.yml
conda activate cerebral-tortuosity-buckling
```

VMTK requires an **x86_64** Conda stack (use Rosetta on Apple Silicon).

## Repository layout

| Directory | Role |
|-----------|------|
| `01_preprocessing/` | Resample, MRA denoising/N4/BET, registration, hysteresis segmentation |
| `02_segmentation/` | Notes for manual 3D Slicer QC (no automated code) |
| `03_mesh/` | Brain-surface remesh; vessel marching cubes / Taubin / Butterfly |
| `04_centerline/` | VMTK centerline extraction, smoothing, resampling |
| `05_m1_analysis/` | Limen-insulae M1 definition and M1 morphometrics |
| `06_periodic_wave/sin_trend/` | Bishop-frame sinusoid-plus-trend pipeline (**paper analysis code**) |
| `07_validation/` | Pointers to synthetic accuracy / procedure-bias scripts + summary data |
| `08_statistics/` | OLS / spline / piecewise / bootstrap analyses and paper figure/table scripts |
| `09_simulation/` | Growth-driven buckling PDE simulation and simulation figures |
| `10_figures/` | Shared plotting config |
| `data/` | Processed tables + ACTIVE simulation `.npz` (no raw MRI) |
| `lib/` | Shared utilities |
| `docs/` | CAT12 notes, data dictionary |

Symlinks `simulation` → `09_simulation`, `statistics` → `08_statistics`, and `figures` → `10_figures`
preserve historical import names.

## Pipeline order

1. Download IXI Hammersmith Hospital MRA + T1WI.  
2. `01_preprocessing/run_mri_preprocessing.py` — 0.5 mm B-spline resample; MRA denoise (`p=1,r=3`), N4, BET (`bt=0.45`), TRSAA→SyNOnly; hysteresis 99.9/98.5.  
3. CAT12 on resampled T1 (GUI; see `docs/CAT12_NOTES.md`).  
4. `03_mesh/run_surface_remesh.py` — isotropic remesh (10 iter, 0.33 mm).  
5. Manual mask QC in 3D Slicer (`02_segmentation/README.md`).  
6. `03_mesh/run_mesh_generation.py` — discrete marching cubes; remove ≤62.5 mm³; Taubin; Butterfly×2.  
7. `04_centerline/run_centerline_extraction.py` — VMTK; smooth 1.2×100; resample 0.1 mm.  
8. `05_m1_analysis/` — M1 endpoints and \(L\), \(L_0\), \(\rho\).  
9. `06_periodic_wave/sin_trend/31_*.py` → `32_*.py` → `33_*.py` — periodic-wave quantification.  
10. `07_validation` / `36_*`–`44_*` — synthetic accuracy tests (optional regenerate).  
11. `08_statistics/` — regressions and Main/SI figure–table scripts.  
12. `09_simulation/buckling_simulation_P_coupling.py` — numerical model; figures via `simulation_figure_config.py`.

**Shipped data already include steps 8–12 outputs** needed to reproduce statistics/figures without raw MRI.

## Paper figure/table map

| Script | Paper item |
|--------|------------|
| `08_statistics/00_Figure_age_sex_distribution.py` | SI age/sex distribution |
| `08_statistics/01_summarize_M1_morphometrics.py` + `02_*`/`03_*` tables | SI M1 morphology & regression tables |
| `08_statistics/03_Figure_M1_Length_EuclidDistance_rho*.py` / `101_Figure_*wavelength_scaling*.py` | Main geometric figure (M1 panels + wavelength scaling) |
| `08_statistics/03_Figure_M1_rho_nonlinear_patterns.py` / `03_Table_piecewise_models_rho.py` | SI ρ nonlinearity / piecewise table |
| `08_statistics/11_model_validation*.py` + wavelength/amplitude figure scripts | Main wavelength–diameter / amplitude analyses; SI model tables |
| `08_statistics/13_Figure_age_related_visibility.py` | SI age-related visibility |
| `09_simulation/simulation_figure_config.py` (+ `Figure_*.py`) | Main numerical simulation figure |
| Schematic Main/SI illustrations (MIP, anatomy, flowcharts, etc.) | Created outside this repo (not included) |

## Reproducing statistics with shipped CSVs

From the repository root:

```bash
python 08_statistics/00_Figure_age_sex_distribution.py
python 08_statistics/101_Figure_M1_Length_EuclidDistance_rho_wavelength_scaling.py
python 08_statistics/11_model_validation.py
python 09_simulation/simulation_figure_config.py
```

Default analysis CSV: `data/periodic_wave/IXI_HH_sin_trend_analysis_dataset.csv`  
ACTIVE simulation: `data/simulation/buckling_simulation_P_coupling_age_20to80years_mu0p005_g0p35/P_coupling.npz`

## IXI data access

https://brain-development.org/ixi-dataset/  
Use the Hammersmith Hospital (HH) 3 T subset with paired MRA and T1WI, under CC BY-SA 3.0.

## Citation

See `CITATION.cff`. Please also cite the IXI Dataset when using IXI-derived tables.

## Contact

Tomoaki Akiyama — corresponding contact listed in the manuscript.

## Notes

- MCA anatomical labels (`mca_segment`) in the public analysis CSV are retained only for `core_inplane==1` intervals (n=275).
- We used Cursor (Anysphere) with automatic model selection to prepare the analysis code for public release.
