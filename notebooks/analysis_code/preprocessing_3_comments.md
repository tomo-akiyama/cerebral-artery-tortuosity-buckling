# Comments from Jupyter Notebook: 0 前処理3.ipynb

## Cell 1

# Preprocessing

## 1 T1 1mm * 1mm * 1mm
 ### 1_1  Isotropic resampling 1mm * 1mm * 1mm　→ 0.5mm * 0.5mm * 0.5mm
 ### 1_2 CAT12 or FS or BrainSuite  automatic segmentation and surface reconstruction

## 2 MRA 0.5mm * 0.5mm * 0.8mm
 ### 2_1 MRA 0.5mm * 0.5mm * 0.8mm　→ 0.5mm * 0.5mm * 0.5mm
 ### 2_2 Denoising (noise reduction) - ANML, gausian (SNR↑なのでgaussianでOK)
 ### 2_3 N4 bias field correction - Antspy
 ### 2_4 Skull stripping - FSL-BE

## 3 T1 and MRA registration 
 ### 3_1 Linear registration (TRSAA)
 ### 3_2 Non linear registration (SyNOnLy)

## 4 Hysteresis vessel segmentation

## Cell 6

## 3 T1 and MRA registration 
 ### 3_1 Linear registration (TRSAA); Reference T1 (moving, MRA)

## Cell 9

>> CAT12のMRIディレクトリについて  (https://andysbrainbook.readthedocs.io/en/latest/CAT12/CAT12_03_Preprocessing.html)  
•	mwp1: 灰白質のセグメンテーション（mwp1_ファイル名）  
•	mwp2: 白質のセグメンテーション（mwp2_ファイル名）  
•	p: 頭蓋骨と顔が除去された解剖画像のセグメンテーション（p_ファイル名）  
•	wm: MNI空間にワープされた解剖スキャンのセグメンテーション（wm_ファイル名）  
•	y: フィールド不均一性を識別し補正するために使用される逆変形場（y_ファイル名）  
•	mi: Native空間のpreprocessing後画像

## Cell 18

 ### 3_2 Non linear registration (SyNOnLy)
