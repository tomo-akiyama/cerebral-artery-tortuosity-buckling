# Cerebral Artery Tortuosity and Buckling Analysis

## Overview
This repository contains all custom Python code used in the study **“Geometric analysis and mechanical buckling model of human cerebral artery tortuosity”**.  

This study aims to quantify age-related changes in cerebral arterial tortuosity and to develop a mathematical model based on a mechanical buckling framework, using human brain MRI data. The analyses include MRI preprocessing, vessel centerline extraction, quantification of vascular tortuosity, numerical simulations based on a mechanical buckling model, and statistical analyses.  


本リポジトリは**脳動脈弯曲の形態変化と座屈メカニズムの数理解析**に関する研究で使用した**すべてのカスタム Python コード**を収録している。本研究ではヒトの**脳主幹動脈および皮質動脈**の加齢に伴う血管蛇行の変化を対象とし、頭部MRIデータの前処理、血管中心線の抽出、血管弯曲の定量評価、座屈モデルに基づく数値シミュレーション、および統計解析を統合的に行っている。  

---

## Datasets

All MRI data used in this study are publicly available human datasets, including T1-weighted images and MR angiography. 

The IXI Dataset can be downloaded from the following URL:  [IXI Dataset](https://brain-development.org/ixi-dataset/)

---

## Directory Structure

Below is the directory structure of the project:

    cerebral-artery-tortuosity-buckling/
        data/
            raw/                               # raw MRI data (nii.gz files)
            preprocessed/                      # preprocessed MRI data
                MRA_preprocessed/
                    segmentations/
                    vessel_meshes/
                    vessel_centerlines/
                T1_preprocessed/
                    post_cat12/
                    brain_meshes/

        notebooks/            # 探索用 Jupyter notebook (最終的には不要)

        analysis_code/
            preprocessing/
                run_mri_preprocessing.py        # pipeline for preprocessing
                preprocessing.py                # function modules
            mesh/
                run_mesh_generation.py          # pipeline for mesh generation
                mesh_generation.py              # function modules

                run_centerline_extraction.py    # pipeline for centerline extraction
                centerline_extraction.py        # function modules

                run_vessel_mesh_statistics.py   # pipeline for calculating the statistics of vessel mesh
                compute_mesh_statistics.py   # compute_* 系
                split_left_right.py             # 左右分割ロジック

            morphometrics/                      # segmentごとの形態計測
                run_sin_trend_fitting.py
                compute_sin_trend_metrics.py
                sin_trend_utils.py
            
            utils/                              # common utility

        simulation/
            run_buckling_simulation_semi_implicit.py    # semi-implicit Euler simulation of cerebral artery buckling dynamics
            run_bifurcation_and_wavelength_selection.py # Pi_P sweep for bifurcation and wavelength selection
            parameters.py                               # parameters for buckling simulations

        statistics/           # LMM, GAM, GAMM の統計解析コード
            summarize_demographic.py            # subjects / Demographics
            summarize_basic_morphometrics.py    # basic morphometrics
            regression_*.py                # Statistical analysis

        figures/              # 論文で使用した図

        results/              # 中間・最終解析結果（CSV など）
            participants/
                IXI_HH_subjects.csv         # participant-level integrated analysis table
            basic_morphometrics/            # 何が計算されたか（解析入力）
                basic_morphometrics.csv
                per_case/                   # 症例ごとのログ）
            morphometrics/
                sin_trend/
            tables/                         # 論文に出す最終形
                Table1_demographics.csv
                Table2_basic_morphometrics.csv

        environment.yml       # conda 環境（再現性の確保）
        README.md             # このファイル  
        **日本語部分は修正予定**  

---

## Requirements

The Vascular Modeling Toolkit (VMTK, http://www.vmtk.org/) requires an x86_64 (Intel / AMD) Conda environment.

- Windows / Linux with Intel or AMD CPUs are supported.
- On Apple Silicon (arm64) Macs, VMTK must be run under Rosetta using an x86_64 Conda distribution.

```bash
conda env create -f environment.yml
conda activate cerebral-tortuosity-buckling
```

---

## Steps for Analysis

The analysis presented in this study can be reproduced by executing the following Python scripts in order.

### 1. Image processing
- `analysis_code/preprocessing.py`
- `analysis_code/segmentation.py`
- `analysis_code/centerline_extraction.py`

### 2. Geometric analysis
- `analysis_code/curvature_torsion.py`
- `analysis_code/feature_extraction.py`

### 3. Buckling model and numerical simulation
- `simulation/buckling_solver.py`
- `simulation/amplitude_estimation.py`
- `simulation/wavelength_radius_scaling.py`

### 4. Statistical analysis
- `statistics/stat_LMM.py`   (Linear mixed-effects model)
- `statistics/stat_GAM.py`   (Generalized additive model)
- `statistics/stat_GAMM.py`  (GAMM using R mgcv; optional)

### 5. Figure generation
- `analysis_code/plotting.py`

Generated figures and tables are saved in the `figures/` and `results/` directories.

---

##  Statistical Analysis

- LMM（線形混合効果モデル）  
- GAM（一般化加法モデル）  
- GAMM（mgcv を用いた非線形＋ランダム効果モデル）  
- AIC / BIC  
- 尤度比検定（LRT）  
- ブートストラップ信頼区間  
**日本語部分は修正予定**

---

##  Mechanical Buckling Model

`simulation/` includes the following:

- Euler buckling モデル  
- 弱非線形近似（amplitude equation）  
- 座屈波長 λ と血管半径 r のスケーリング  
- 血圧・tethering・伸長 mismatch の負荷分解  
- 加齢による tortuosity 増大の時間発展モデル  
**日本語部分は修正予定**  

---

## Dependencies

See `environment.yml` for details of the Python/Conda environment used in this study.  

---

## Citation

When using these codes, please cite the following reports:   
Akiyama T., et al. (2025) Geometric analysis and mechanical buckling model of human cerebral artery tortuosity. Journal.  

---

## Contacts

**First Author**  
Tomoaki Akiyama (秋山 智明), MD  
Email: akiyama.tomoaki@kyudai.jp  
Affiliation:  
- Department of Neurosurgery, Graduate School of Medical Sciences, Kyushu University, Fukuoka, Fukuoka, Japan (九州大学大学院医学研究院 脳神経外科)  
- Department of Anatomy and Cell Biology, Graduate School of Medical Sciences, Kyushu University, Fukuoka, Fukuoka, Japan (九州大学大学院医学研究院 系統解剖学分野)   

**Corresponding Author**  
Prof. Takashi Miura (三浦 岳), MD, PhD  
Email: miura.takashi.869@m.kyushu-u.ac.jp  
Affiliation:  
- Department of Anatomy and Cell Biology, Graduate School of Medical Sciences, Kyushu University, Fukuoka, Fukuoka, Japan (九州大学大学院医学研究院 系統解剖学分野)  