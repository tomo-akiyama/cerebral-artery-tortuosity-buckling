#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: 0 前処理3.ipynb

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
# Load the MRA images
corrected_img_MRA_path = os.path.join(intermediate_data0_path, "2_3_corrected_MRA.nii.gz")
corrected_img_MRA = ants.image_read(corrected_img_MRA_path); corrected_img_MRA_array = corrected_img_MRA.numpy()
bet_mask_MRA_path = os.path.join(intermediate_data0_path, "2_4_1_bet_MRA.nii.gz")
bet_mask_MRA = ants.image_read(bet_mask_MRA_path); bet_mask_MRA_array = bet_mask_MRA.numpy()

# moving_mask = ants.get_mask(bet_mask_MRA)としてもよいが、bet_mask_MRAはそもそもmaskなので、そのまま使う
moving_mask = bet_mask_MRA; moving_mask_array = moving_mask.numpy()


# Cell 8
corrected_img_MRA = tool.match_origin_and_direction(original_img=img_MRA, target_img=corrected_img_MRA)
moving_mask = tool.match_origin_and_direction(original_img=img_MRA, target_img=moving_mask)

# Cell 10
# Load the T1 images
preprocessed_img_T1_path = os.path.join(intermediate_data0_path, "mri/m1_1_resampled_T1.nii")
preprocessed_img_T1 = ants.image_read(preprocessed_img_T1_path); preprocessed_img_T1_array = preprocessed_img_T1.numpy()

segmented_brain_T1_path = os.path.join(intermediate_data0_path, "mri/p01_1_resampled_T1.nii")
segmented_brain_T1 = ants.image_read(segmented_brain_T1_path); segmented_brain_T1_array = segmented_brain_T1.numpy()



# CAT12のp: 頭蓋骨と顔が除去された解剖画像のセグメンテーションをfixed maskとして使用する
fixed_mask = ants.get_mask(segmented_brain_T1); fixed_mask_array = fixed_mask.numpy()


# Cell 11
# 例 similarity_check
mi_t1_same = ants.image_mutual_information(preprocessed_img_T1, preprocessed_img_T1)
mi_mra_same = ants.image_mutual_information(corrected_img_MRA, corrected_img_MRA)

similarity_t1_t1_mask = ants.image_similarity(fixed_image=preprocessed_img_T1,
                                                moving_image=preprocessed_img_T1,
                                                metric_type='MattesMutualInformation', 
                                                fixed_mask=fixed_mask, 
                                                moving_mask=fixed_mask, 
                                                sampling_strategy='regular',
                                                sampling_percentage=1.)

similarity_t1_t1_nomask = ants.image_similarity(fixed_image=preprocessed_img_T1,
                                                 moving_image=preprocessed_img_T1,
                                                 metric_type='MattesMutualInformation', 
                                                 sampling_strategy='regular',
                                                 sampling_percentage=1.)

similarity_mra_mra_mask= ants.image_similarity(fixed_image=corrected_img_MRA,
                                                moving_image=corrected_img_MRA,
                                                metric_type='MattesMutualInformation', 
                                                fixed_mask=moving_mask, 
                                                moving_mask=moving_mask, 
                                                sampling_strategy='regular',
                                                sampling_percentage=1.)

similarity_mra_mra_nomask = ants.image_similarity(fixed_image=corrected_img_MRA,
                                                    moving_image=corrected_img_MRA,
                                                    metric_type='MattesMutualInformation', 
                                                    sampling_strategy='regular',
                                                    sampling_percentage=1.)

# Cell 12
# 位置合わせ前の類似度
mi_before = ants.image_mutual_information(preprocessed_img_T1, corrected_img_MRA)

"""
  metric_type : string
        image metric to calculate
            MeanSquares
            Correlation
            ANTSNeighborhoodCorrelation
            MattesMutualInformation
            JointHistogramMutualInformation
            Demons
"""
similarity_before = ants.image_similarity(fixed_image=preprocessed_img_T1,
                                                moving_image=corrected_img_MRA,
                                                metric_type='MattesMutualInformation', 
                                                fixed_mask=fixed_mask, 
                                                moving_mask=moving_mask,
                                                sampling_strategy='regular',
                                                sampling_percentage=1.)

# Cell 13
preprocessed_img_T1, fixed_mask, corrected_img_MRA, moving_mask

# Cell 14
# スレッド数を1に設定して完全な再現性を確保
import os
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

# TRSAA変換による初期位置合わせ
TRSAA_registration = ants.registration(
    fixed=preprocessed_img_T1, 
    moving=corrected_img_MRA, 
    type_of_transform='TRSAA',
    mask=fixed_mask, 
    moving_mask=moving_mask, 
    aff_random_sampling_rate=0.2,
    aff_sampling=256,
    aff_iterations=(3000, 2000, 1500, 500), 
    aff_shrink_factors=(8, 4, 2, 1), # 追加
    reg_iterations = (1000,1000,100,20), 
    random_seed=12345, 
    verbose=False
)

# TRSAA_registration = ants.registration(
#     fixed=preprocessed_img_T1, 
#     moving=corrected_img_MRA, 
#     type_of_transform='TRSAA',
#     mask=fixed_mask, 
#     moving_mask=moving_mask, 
#     aff_random_sampling_rate=0.5,  # サンプリングレートを上げる
#     aff_sampling=512,  # サンプリングポイント数を増やす
#     aff_iterations=(5000, 3000, 2000, 1000),  # イテレーション数を増やす
#     reg_iterations = (2000, 1500, 1000, 500),  # イテレーション数を増やす
#     random_seed=12345, 
#     verbose=False
# )


# 位置合わせ後の画像
TRSAA_moving_img_MRA = TRSAA_registration['warpedmovout']; TRSAA_moving_img_MRA_array = TRSAA_moving_img_MRA.numpy()
TRSAA_moving_img_MRA_path = os.path.join(intermediate_data0_path, "3_1_1_TRSAA_MRA.nii.gz")
ants.image_write(TRSAA_moving_img_MRA, TRSAA_moving_img_MRA_path)

# Cell 15
mapped_TRSAA_bet_mask_MRA = ants.apply_transforms(
    fixed=preprocessed_img_T1, 
    moving=moving_mask, 
    transformlist=TRSAA_registration['fwdtransforms'], 
    interpolator='nearestNeighbor', 
    verbose=False)
mapped_TRSAA_bet_mask_MRA_path = os.path.join(intermediate_data0_path, "3_1_2_mapped_TRSAA_bet_mask_MRA.nii.gz")
ants.image_write(mapped_TRSAA_bet_mask_MRA, mapped_TRSAA_bet_mask_MRA_path)


display.overlay_axial_slices(volume1=preprocessed_img_T1_array, volume2=TRSAA_moving_img_MRA_array, 
                             step=10, alpha1=1, alpha2=0.5,
                             output_file=os.path.join(intermediate_data0_path, "3_1_3_TRSAA_overlay_axial_slices.png"))
display.overlay_sagittal_slices(volume1=preprocessed_img_T1_array, volume2=TRSAA_moving_img_MRA_array, 
                                step=10, alpha1=1, alpha2=0.5,
                                output_file=os.path.join(intermediate_data0_path, "3_1_4_TRSAA_overlay_sagittal_slices.png"))

# Cell 16

# Cell 17
# 類似度
mi_after_TRSAA = ants.image_mutual_information(preprocessed_img_T1, TRSAA_moving_img_MRA)
similarity_after_TRSAA = ants.image_similarity(fixed_image=preprocessed_img_T1,
                                                moving_image=TRSAA_moving_img_MRA,
                                                metric_type='MattesMutualInformation', 
                                                fixed_mask=fixed_mask, 
                                                moving_mask=mapped_TRSAA_bet_mask_MRA,
                                                sampling_strategy='regular',
                                                sampling_percentage=1.)

# Cell 19
# TRSAA変換による初期位置合わせ後にSyNOnly変換を実行
# 初期位置合わせの変換をinitial_transformとして利用し、その変換を基に非線形変換を実行

TRSAA_SyNOnly_registration = ants.registration(
    fixed=preprocessed_img_T1, 
    moving=corrected_img_MRA, 
    type_of_transform='SyNOnly',
    mask=fixed_mask, 
    moving_mask=moving_mask, 
    initial_transform=TRSAA_registration['fwdtransforms'],
    syn_sampling=256, 
    reg_iterations = (1000,1000,100,20),
    random_seed=12345,  # 再現性を確保するためのランダムシード
    verbose=False
)

# Cell 20
# 位置合わせ後の画像
TRSAA_SyNOnly_moving_img_MRA = TRSAA_SyNOnly_registration['warpedmovout']; TRSAA_SyNOnly_moving_img_MRA_array = TRSAA_SyNOnly_moving_img_MRA.numpy()
TRSAA_SyNOnly_moving_img_MRA_path = os.path.join(intermediate_data0_path, "3_2_1_TRSAA_SyNOnly_MRA.nii.gz")
ants.image_write(TRSAA_SyNOnly_moving_img_MRA, TRSAA_SyNOnly_moving_img_MRA_path)

# 3D slicerで選択しやすいように、ディレクトリの一番上になるようにする
TRSAA_SyNOnly_moving_img_MRA_path = os.path.join(intermediate_data0_path, "0 3_2_1_TRSAA_SyNOnly_MRA.nii.gz")
ants.image_write(TRSAA_SyNOnly_moving_img_MRA, TRSAA_SyNOnly_moving_img_MRA_path)

# 類似度
mi_after_TRSAA_SyNOnly = ants.image_mutual_information(preprocessed_img_T1, TRSAA_SyNOnly_moving_img_MRA)
similarity_after_TRSAA_SyNOnly = ants.image_similarity(fixed_image=preprocessed_img_T1,
                                                moving_image=TRSAA_SyNOnly_moving_img_MRA,
                                                metric_type='MattesMutualInformation', 
                                                fixed_mask=fixed_mask, 
                                                moving_mask=ants.apply_transforms(preprocessed_img_T1, 
                                                                                  moving_mask, 
                                                                                  transformlist=TRSAA_SyNOnly_registration['fwdtransforms'], 
                                                                                  interpolator='nearestNeighbor'),
                                                sampling_strategy='regular',
                                                sampling_percentage=1.)

# Cell 21
mapped_TRSAA_SyNOnly_bet_mask_MRA = ants.apply_transforms(
    fixed=preprocessed_img_T1, 
    moving=moving_mask, 
    transformlist=TRSAA_SyNOnly_registration['fwdtransforms'], 
    interpolator='nearestNeighbor', 
    verbose=False)

mapped_TRSAA_SyNOnly_bet_mask_MRA_path = os.path.join(intermediate_data0_path, "3_2_2_mapped_TRSAA_SyNOnly_bet_mask_MRA.nii.gz")
ants.image_write(mapped_TRSAA_SyNOnly_bet_mask_MRA, mapped_TRSAA_SyNOnly_bet_mask_MRA_path)

display.overlay_axial_slices(volume1=preprocessed_img_T1_array, volume2=TRSAA_SyNOnly_moving_img_MRA_array, 
                             step=10, alpha1=1, alpha2=0.5,
                             output_file=os.path.join(intermediate_data0_path, "3_2_overlay_TRSAA_SyNOnly_axial.png"))
display.overlay_sagittal_slices(volume1=preprocessed_img_T1_array, volume2=TRSAA_SyNOnly_moving_img_MRA_array, 
                                step=10, alpha1=1, alpha2=0.5,
                                output_file=os.path.join(intermediate_data0_path, "3_2_overlay_TRSAA_SyNOnly_sagittal.png"))

# Cell 22
# 最終結果をマスクオーバーレイで可視化

# Cell 23
import pandas as pd
# Calculate the similarity using cosine distance
similarity_matrix = pd.DataFrame(columns=['Variable', 'Similarity'])
similarity_matrix['Variable'] = ['相互情報量(同一画像 t1)', '相互情報量(同一画像 MRA)', 
                                 '相互情報量(前)', '相互情報量(TRSAA後)', "相互情報量(TRSAA_SyNOnly後)",

                                 'similarity t1_t1 with mask', 'similarity t1_t1 without mask','similarity mra_mra with mask', 'similarity mra_mra without mask',
                                 'similarity_before', "similarity_TRSAA", "similarity_after_TRSAA_SyNOnly"
                                    ]
similarity_matrix['Similarity'] = [mi_t1_same, mi_mra_same, 
                                      mi_before, mi_after_TRSAA, mi_after_TRSAA_SyNOnly,


                                   similarity_t1_t1_mask, similarity_t1_t1_nomask, similarity_mra_mra_mask, similarity_mra_mra_nomask,
                                   similarity_before, similarity_after_TRSAA, similarity_after_TRSAA_SyNOnly]

# Save the similarity matrix as a dataframe
similarity_matrix.to_csv(os.path.join(intermediate_data0_path, '3_3_similarity_matrix.csv'), index=False)

similarity_matrix 

# Cell 24
end = time.time()  # 現在時刻（処理完了後）を取得
# 処理完了後の時刻（終了時間）を表示
終了時間 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))

time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する

# 秒を「時：分」に変換
hours, remainder = divmod(time_diff, 3600)
minutes, seconds = divmod(remainder, 60)
# 処理にかかった時間を「時：分」形式で表示

tool.notify(case)
