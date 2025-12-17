# 脳動脈座屈（Cerebral Artery Buckling）解析
Cerebral Artery Buckling Project

本リポジトリは、研究  
**「脳動脈の形態変化と座屈メカニズムの数理解析」**  
**「Geometric analysis and mechanical buckling model of human cerebral artery tortuosity」**
で使用したすべての **カスタム Python コード**を含んでいます。

脳 MRA データの処理、血管中心線の抽出、曲率・ねじれ計測、  
座屈モデルの数値シミュレーション、統計解析（LMM, GAM, GAMM）など、  
論文に掲載された解析を再現できるよう構成されています。

---

## 使用したデータセット
IXI Dataset(https://brain-development.org/ixi-dataset/)

---

## ディレクトリ構成（Directory structure）

以下はプロジェクトのフォルダ構成です：

    cerebral_artery_buckling_project/
        data/                 # MRA, セグメンテーション, 中心線データ
        notebooks/            # 探索用 Jupyter notebook
        analysis_code/        # 論文用に整理した Python コード
        simulation/           # 座屈モデル（Euler buckling, 非線形ソルバー）
        statistics/           # LMM, GAM, GAMM の統計解析コード
        figures/              # 論文で使用した図
        results/              # 中間・最終解析結果（CSV など）
        environment.yml       # conda 環境（再現性の確保）
        README.md             # このファイル

---

## 🔧 インストール手順（Installation）

1. Anaconda または Miniconda をインストール  
2. 以下のコマンドを実行：
conda env create -f environment.yml
conda activate cerebral-buckling

---

## ▶ 解析の再現方法（How to reproduce the analysis）

本研究の解析は、以下の Python スクリプトを順に実行することで再現できます。

### 1. 画像処理（Image processing）
analysis_code/preprocessing.py
analysis_code/segmentation.py
analysis_code/centerline_extraction.py

### 2. 幾何学解析（Geometry analysis）
analysis_code/curvature_torsion.py
analysis_code/feature_extraction.py

### 3. 座屈モデル・シミュレーション（Buckling model）
simulation/buckling_solver.py
simulation/amplitude_estimation.py
simulation/wavelength_radius_scaling.py

### 4. 統計解析（Statistical analysis）
statistics/stat_LMM.py     # 線形混合効果モデル
statistics/stat_GAM.py     # 一般化加法モデル
statistics/stat_GAMM.py    # R mgcv を用いた GAMM（任意）

### 5. 図の作成（Figures）
analysis_code/plotting.py
生成された図や表は `figures/` および `results/` に保存されます。

---

##  統計解析（Statistical analysis）

- LMM（線形混合効果モデル）  
- GAM（一般化加法モデル）  
- GAMM（mgcv を用いた非線形＋ランダム効果モデル）  
- AIC / BIC  
- 尤度比検定（LRT）  
- ブートストラップ信頼区間  

---

##  座屈モデル（Buckling model）

`simulation/` には以下の解析が含まれます：

- Euler buckling モデル  
- 弱非線形近似（amplitude equation）  
- 座屈波長 λ と血管半径 r のスケーリング  
- 血圧・tethering・伸長 mismatch の負荷分解  
- 加齢による tortuosity 増大の時間発展モデル  

---

## 依存パッケージ（Dependencies）

- Python 3.10+  
- NumPy / SciPy  
- pandas  
- scikit-learn  
- statsmodels  
- matplotlib / seaborn  
- pyvista / VMTK / vtk  
- optional: rpy2 + R mgcv（GAMM 用）  

詳細は `environment.yml` を参照してください。

---

## 引用（Citation）

本コードを利用する際は以下を引用してください：
Akiyama T., et al. (2025)
Geometric analysis and mechanical buckling model of human cerebral artery tortuosity.
Journal.

---

## 連絡先（Contacts）
**筆頭著者（First Author）**  
秋山 智明 (Tomoaki Akiyama), MD  
Email: akiyama.tomoaki@kyudai.jp  
所属：  
・九州大学大学院医学研究院 脳神経外科  
・九州大学大学院医学研究院 系統解剖学分野

**責任著者（Corresponding Author）**  
三浦 岳 (Prof. Takashi Miura), MD, PhD  
Email: miura.takashi.869@m.kyushu-u.ac.jp  
所属：  
・九州大学大学院医学研究院 系統解剖学分野