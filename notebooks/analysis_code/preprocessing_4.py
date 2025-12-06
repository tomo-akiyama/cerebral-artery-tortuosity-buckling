#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: 0 前処理4.ipynb

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
import display

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
TRSAA_SyNOnly_moving_img_MRA_path = os.path.join(intermediate_data0_path, "3_2_1_TRSAA_SyNOnly_MRA.nii.gz")
registerd_MRA = ants.image_read(TRSAA_SyNOnly_moving_img_MRA_path); registerd_MRA_array = registerd_MRA.numpy()

mapped_TRSAA_SyNOnly_bet_mask_MRA_path = os.path.join(intermediate_data0_path, "3_2_2_mapped_TRSAA_SyNOnly_bet_mask_MRA.nii.gz")
mask_MRA = ants.image_read(mapped_TRSAA_SyNOnly_bet_mask_MRA_path); mask_MRA_array = mask_MRA.numpy()

brain_MRA = registerd_MRA * mask_MRA
brain_MRA_array = brain_MRA.numpy()

registerd_MRA, mask_MRA, brain_MRA

# Cell 8

# Cell 9
brain_MRA_signals = brain_MRA_array.flatten()
brain_MRA_signals = brain_MRA_signals[brain_MRA_signals > 0]

high_threshold_percentile = 99.9
low_threshold_percentile = 98.5




# Cell 10
# 注意 hysteresis閾値処理内にfill holesは入っている
vessel_hysterisis_thresholding = preprocessing.apply_hysteresis_threshold(image=brain_MRA_array, 
                                                     low=low_threshold, 
                                                     high=high_threshold)


# Cell 11
# remove small islands
# vessel_hysterisis_thresholding  = preprocessing.remove_small_islands(vessel_hysterisis_thresholding , volume_threshold)

from skimage.morphology import remove_small_objects
vessel_hysterisis_thresholding= remove_small_objects(vessel_hysterisis_thresholding.astype(bool), min_size=volume_threshold, connectivity=3).astype('float32')

# Cell 12

# Cell 13
# Save the image of vessel_hysterisis_thresholding
vessel_hysterisis_path = os.path.join(intermediate_data0_path, "4_1_vessel_hysterisis.nii.gz")

origin = registerd_MRA.origin
spacing = registerd_MRA.spacing
direction = registerd_MRA.direction

vessel = ants.from_numpy(vessel_hysterisis_thresholding)
vessel.set_origin(origin)
vessel.set_spacing(spacing)
vessel.set_direction(direction)

ants.image_write(vessel, vessel_hysterisis_path, ri=False)


# 3D slicerで選択しやすいように、ディレクトリの一番上になるようにする
vessel_hysterisis_path = os.path.join(intermediate_data0_path, "0 4_1_vessel_hysterisis.nii.gz")
ants.image_write(vessel, vessel_hysterisis_path , ri=False)

# Cell 14
# Pyvista表示(表示はVTK座標に合わせる x,y,z -> z,y,x)
vessel_hysterisis_thresholding_pv = pv.wrap(
    np.transpose(vessel_hysterisis_thresholding, (2, 1, 0))
    )
vessel_hysterisis_thresholding_pv_outline = vessel_hysterisis_thresholding_pv.contour()

p = pv.Plotter()
p.background_color = 'white'
p.camera_position = 'zx'
p.add_title("Hysterisis thresholding", color='k', font_size=10)
p.add_axes(line_width=5, labels_off=False)

# Save the 3D images
vessel_html_path = os.path.join(intermediate_data0_path, "4_2_vessel_hysterisis.html")

# Cell 15
end = time.time()  # 現在時刻（処理完了後）を取得
# 処理完了後の時刻（終了時間）を表示
終了時間 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))

time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する

# 秒を「時：分」に変換
hours, remainder = divmod(time_diff, 3600)
minutes, seconds = divmod(remainder, 60)
# 処理にかかった時間を「時：分」形式で表示

# Cell 16
tool.notify(case)
