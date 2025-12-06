# Comments from Jupyter Notebook: 0 前処理4.ipynb

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

## 4 Hysteresis vessel segmentation
