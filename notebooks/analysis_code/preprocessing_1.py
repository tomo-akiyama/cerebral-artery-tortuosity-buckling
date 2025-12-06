#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: 0 前処理1.ipynb

# Cell 2
# Select the case for preprocessing

case = "IXI202-HH-1526"

# Cell 3
import ants
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pyvista as pv

import sys
# 0 Scriptデイレクトリに移動する相対パス
sys.path.append(" 系統解剖研究室/10 Geometric analysis 220401-/Dataset/IXI HH/0 Script")

import tool
import preprocessing
import mesh_generation

import time
start = time.time() # 現在時刻（処理開始前）を取得

# Cell 4
# 共通項目
dataset = " 系統解剖研究室/10 Geometric analysis 220401-/Dataset/IXI HH"

voxel_size = 0.5


# 画像フォルダ
import os
case_path = os.path.join(dataset, case)
print(f"case_path = {case_path}")

# 画像フォルダ
import os
case_path = os.path.join(dataset, case)
data_path = os.path.join(case_path, 'Data')
intermediate_data0_path = os.path.join(case_path, 'Intermediate_data0'); tool.create_dir(intermediate_data0_path)
intermediate_data1_path = os.path.join(case_path, 'Intermediate_data1'); tool.create_dir(intermediate_data1_path)

intermediate_data1_3ds_path = os.path.join(case_path, 'Intermediate_data1 3d slicer'); tool.create_dir(intermediate_data1_3ds_path)


mca_intermediate_data1_3ds_path = os.path.join(case_path, 'MCA Intermediate_data1 3d slicer'); tool.create_dir(mca_intermediate_data1_3ds_path)

# Cell 5
# Load the images
input_MRA_path = os.path.join(case_path, case + "-MRA.nii.gz")
input_T1_path = os.path.join(case_path, case + "-T1.nii.gz")
print(f"input_MRA_path = {input_MRA_path}")
print(f"input_T1_path = {input_T1_path}")

img_MRA = ants.image_read(input_MRA_path); img_mra_array = img_MRA.numpy()
img_T1 = ants.image_read(input_T1_path); img_T1_array = img_T1.numpy()
print("MRA image header", img_MRA)
print("T1w image header", img_T1)

# Show original image



# Cell 7
# Isotropic resampling  (https://antspy.readthedocs.io/en/latest/_modules/ants/registration/resample_image.html)
# resample_image(image, resample_params, use_voxels=False, interp_type=1)
#    ---------
#    image : ANTsImage  input image
#    resample_params : tuple/list  vector of size dimension with numeric values
#    use_voxels : boolean  True means interpret resample params as voxel counts
#    interp_type : integer  one of 0 (linear), 1 (nearest neighbor), 2 (gaussian), 3 (windowed sinc), 4 (bspline)
#    ---------
# Returns    ANTsImage

print(f"spacing = {voxel_size} mm")
resampled_img_T1 = ants.resample_image(img_T1, (voxel_size, voxel_size, voxel_size),False,4)
print("T1 image header after resampling", resampled_img_T1)

# Cell 8
# Resamplingの影響でT1画像のbackgoundでマイナス値となったボクセル値のところを0にする
resampled_img_T1[resampled_img_T1.numpy() < 0] = 0
resampled_img_T1_array = resampled_img_T1.numpy()

# Save the image
resampled_img_T1_path = os.path.join(intermediate_data0_path, "1_1_resampled_T1.nii.gz")
ants.image_write(resampled_img_T1, resampled_img_T1_path, ri=False)


# Cell 10
end = time.time()  # 現在時刻（処理完了後）を取得
# 処理完了後の時刻（終了時間）を表示
終了時間 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))
print("終了時間:", 終了時間)

time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する

# 秒を「時：分」に変換
hours, remainder = divmod(time_diff, 3600)
minutes, seconds = divmod(remainder, 60)
# 処理にかかった時間を「時：分」形式で表示
print("処理時間: {:02}時間{:02}分".format(int(hours), int(minutes)))

# Cell 11
tool.notify(case)
