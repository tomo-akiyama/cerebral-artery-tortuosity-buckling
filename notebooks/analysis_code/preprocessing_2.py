#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: 0 前処理2.ipynb

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

img_MRA = ants.image_read(input_MRA_path); img_mra_array = img_MRA.numpy()
img_T1 = ants.image_read(input_T1_path); img_T1_array = img_T1.numpy()

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

resampled_img_MRA = ants.resample_image(img_MRA, (voxel_size, voxel_size, voxel_size),False,4)

# Add pad
matrix_size = max(resampled_img_MRA.shape)
resampled_img_MRA = ants.pad_image(resampled_img_MRA, shape=(matrix_size, matrix_size, matrix_size)) 

# Cell 8
# Resamplingの影響でbackgoundのマイナス値となったボクセルを0にする
resampled_img_MRA[resampled_img_MRA.numpy() < 0] = 0
resampled_img_MRA_array = resampled_img_MRA.numpy()

# Save the image
resampled_img_MRA_path = os.path.join(intermediate_data0_path, "2_1_resampled_MRA.nii.gz")
ants.image_write(resampled_img_MRA, resampled_img_MRA_path, ri=False)


# Cell 10
# Adaptive Non-Local Means Denoising with Gaussian noise model
denoised_img_MRA = ants.denoise_image(resampled_img_MRA, 
                                    mask=None, 
                                    shrink_factor=1, 
                                    p=1, r=3, 
                                    noise_model='Gaussian', v=0)

# Save the image
denoised_img_MRA_path = os.path.join(intermediate_data0_path, "2_2_denoised_MRA.nii.gz")
ants.image_write(denoised_img_MRA, denoised_img_MRA_path, ri=False)

# Cell 11
denoised_img_MRA = tool.match_origin_and_direction(original_img=img_MRA, target_img=denoised_img_MRA)

# Cell 13
# rescale_intensities=Trueで信号強度の大きなずれを防ぐ
corrected_img_MRA = ants.n4_bias_field_correction(denoised_img_MRA, 
                                                  rescale_intensities=True, 
                                                  shrink_factor=2); corrected_img_MRA_array = corrected_img_MRA.numpy()

# Save the image
corrected_img_MRA_path = os.path.join(intermediate_data0_path, "2_3_corrected_MRA.nii.gz")
ants.image_write(corrected_img_MRA, corrected_img_MRA_path, ri=False)

# Cell 14
corrected_img_MRA = tool.match_origin_and_direction(original_img=img_MRA, target_img=corrected_img_MRA)

# Cell 16
# FLS-BEがうまくいかない場合は、ここから新規で始めるとiteration counterが動く

# Cell 17
corrected_img_MRA_path = os.path.join(intermediate_data0_path, "2_3_corrected_MRA.nii.gz")

# Cell 18
from brainextractor import BrainExtractor
import nibabel as nib

# read in the image file
img = nib.load(corrected_img_MRA_path)

# create a BrainExtractor object using the input_img as input
# we just use the default arguments here, but look at the
# BrainExtractor class in the code for the full argument list
# parameteについて：https://github.com/vanandrew/brainextractor/blob/main/brainextractor/main.py
bet_mask = BrainExtractor(img=img, bt=0.45) # default, bt=0.5

# run the brain extraction
# this will by default run for 1000 iterations
# I recommend looking at the run method to see how it works
bet_mask.run()

# save the brain extracted image
bet_mask_MRA_path = os.path.join(intermediate_data0_path, "2_4_1_bet_MRA.nii.gz")
bet_mask.save_mask(bet_mask_MRA_path)

# Cell 19
# Brain extractionの結果、originが変わるので、元のoriginに戻す
bet_mask_MRA = ants.image_read(bet_mask_MRA_path)
bet_mask_MRA = tool.match_origin_and_direction(original_img=img_MRA, target_img=bet_mask_MRA)
ants.image_write(bet_mask_MRA, bet_mask_MRA_path, ri=False)

# Cell 20
bet_mask_MRA = ants.image_read(bet_mask_MRA_path); bet_mask_MRA_array = bet_mask_MRA.numpy()
brain_MRA = corrected_img_MRA * bet_mask_MRA_array; brain_MRA_array = brain_MRA.numpy()

bet_mask_MRA, brain_MRA, corrected_img_MRA

# Cell 21

brain_MRA_path = os.path.join(intermediate_data0_path, "2_4_2_brain_MRA.nii.gz")
ants.image_write(brain_MRA, brain_MRA_path, ri=False)

# Cell 22
end = time.time()  # 現在時刻（処理完了後）を取得
# 処理完了後の時刻（終了時間）を表示
終了時間 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))

time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する

# 秒を「時：分」に変換
hours, remainder = divmod(time_diff, 3600)
minutes, seconds = divmod(remainder, 60)
# 処理にかかった時間を「時：分」形式で表示

# Cell 23
tool.notify(case)

# Cell 24
# Skull stripping AntspyNetの場合 →MRAには使えない

# https://antsx.github.io/ANTsPyNet/docs/build/html/utilities.html?highlight=brain%20extraction#antspynet.utilities.brain_extraction
# https://antsx.github.io/ANTsPyNet/docs/build/html/_modules/antspynet/utilities/brain_extraction.html
