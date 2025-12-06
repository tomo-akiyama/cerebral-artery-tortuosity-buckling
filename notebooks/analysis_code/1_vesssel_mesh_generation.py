#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: 1 vesssel mesh generation.ipynb

# Cell 2
# Select the case for preprocessing

case = "IXI546-HH-2450"

# Cell 3
import ants
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pyvista as pv

import sys
# 0 Scriptデイレクトリに移動する相対パス
sys.path.append(" 系統解剖研究室/10 Geometric analysis 220401-/Dataset/IXI HH/0 Script")

import vmtk_centerlines as vmtk_CL
import tool
import preprocessing
import mesh_generation
import display

import time
start = time.time() # 現在時刻（処理開始前）を取得

# Cell 4
# 共通項目
dataset = " 系統解剖研究室/10 Geometric analysis 220401-/Dataset/IXI HH"

# 画像フォルダ
import os
case_path = os.path.join(dataset, case)

voxel_size = 0.5

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



# Cell 6
# Load the images
 # T1 images
preprocessed_img_T1_path = os.path.join(intermediate_data0_path, "mri/m1_1_resampled_T1.nii")
preprocessed_img_T1 = ants.image_read(preprocessed_img_T1_path); preprocessed_img_T1_array = preprocessed_img_T1.numpy()
 # TRSAA + SyNOnly
TRSAA_SyNOnly_moving_img_MRA_path = os.path.join(intermediate_data0_path, "3_2_1_TRSAA_SyNOnly_MRA.nii.gz")
registerd_MRA = ants.image_read(TRSAA_SyNOnly_moving_img_MRA_path); registerd_MRA_array = registerd_MRA.numpy()


# Hysterisis閾値処理後(manual editing前)
vessel_hysterisis_path = os.path.join(intermediate_data0_path, "4_1_vessel_hysterisis.nii.gz")
vessel_hysterisis = ants.image_read(vessel_hysterisis_path); vessel_hysterisis_array = vessel_hysterisis.numpy()

# 3d slicerによるmanual editing後
 # whole artery
# vessel = ants.image_read(vessel_path)
 # mca
mca_path = os.path.join(intermediate_data1_3ds_path, "mca Segmentation-Segment_1-label_1.nii.gz")
mca = ants.image_read(mca_path)

# Headerの確認
# vessel = tool.match_origin_and_direction(original_img=registerd_MRA, target_img=vessel); vessel_array = vessel.numpy()
mca = tool.match_origin_and_direction(original_img=registerd_MRA, target_img=mca); mca_array = mca.numpy()

# Cell 7
# 画像の確認


# Cell 8
# Meshlabにvolume dataからmcする関数がなかったので、vtkでmcを行う
# vessel_mc = mesh_generation.vtk_mc(mesh_generation.np2vtk(vessel_array), threshold=1)
mca_mc = mesh_generation.vtk_mc(mesh_generation.np2vtk(mca_array), threshold=1)

# 3D表示
pl = pv.Plotter()
pl.camera.zoom(1.5)

# Cell 9
import pymeshlab
import pandas as pd

def create_mesh (Case, vessel_mc, vessel_mc_name, directry_path, volume_threshold = 5000, voxel_size = 0.5,
                 l=0.6307, m=-0.6732, s=10, i=2):
    # vessel meshのvolumeを計算
    vol_list = mesh_generation.vol_distribution(vessel_mc)

    # 小さなislandを再度削除
    vessel_mc = mesh_generation.filter_by_volume(vessel_mc, volume_threshold)

    # vtkをplyに変換
    vessel_mc_filename = f"1_1_{vessel_mc_name}.ply"
    vessel_mc_path = os.path.join(directry_path, vessel_mc_filename)
    mesh_generation.export_vtkpolydata_to_ply(vessel_mc, vessel_mc_path)


    # 細かいメッシュ
    # メッシュセットの作成
    ms = pymeshlab.MeshSet()
    # メッシュファイルの読み込み
    ms.load_new_mesh(vessel_mc_path)
    vertex_count1 = ms.current_mesh().vertex_number()

    # Taubin smoothing (lambda=0.6307, mu=-0.6732は原著で使われている値、pass band = 0.1と同じ)
    ms.apply_coord_taubin_smoothing(lambda_=l, mu=m, stepsmoothnum=s)
    vertex_count2 = ms.current_mesh().vertex_number()

    # Butterfly subdivision
    # iter 1で4倍に分割, iter 2では16倍に分割 → 3回繰り返すと256倍に分割
    # thresholdは特定の長さ以上のエッジを持つすべてのエッジを細分化するために使用される。0%の場合、すべてのエッジが細分化される。
    ms.meshing_surface_subdivision_butterfly(iterations = i, threshold = pymeshlab.PercentageValue(0.000000))
    vertex_count3 = ms.current_mesh().vertex_number()

    # 処理後のメッシュを保存
    vessel_mc_taubin_thick_butterfly_filename = f"1_2_{vessel_mc_name}_taubin_thick_butterfly16.ply"
    vessel_mc_taubin_thick_butterfly_path = os.path.join(directry_path, vessel_mc_taubin_thick_butterfly_filename)
    ms.save_current_mesh(vessel_mc_taubin_thick_butterfly_path)
    
    
    # vessel meshのvolumeを再計算
    vessel_mc_taubin_thick_butterfly = mesh_generation.load_ply_to_polydata(vessel_mc_taubin_thick_butterfly_path)
    vol_list = mesh_generation.vol_distribution(vessel_mc_taubin_thick_butterfly)
    surface_vased_volume = np.sum(vol_list) 

    # DataFrameを作成するためのデータ
    index1 = [Case]
    data = {
        "Case": [Case],
        "number of faces": [vessel_mc_taubin_thick_butterfly.GetNumberOfCells()],
        "mesh-based volume": [surface_vased_volume],
        "mesh-based volume (mm^3)": [np.sum(surface_vased_volume) * (voxel_size ** 3)],
        "face density": [vessel_mc_taubin_thick_butterfly.GetNumberOfCells()/(np.sum(surface_vased_volume)*(voxel_size**3))]
    }

    # indexとcolumnsを設定してDataFrameを作成
    df1 = pd.DataFrame(data)
    # DataFrameを表示

    # meshの統計量をcsvに保存
    meshdata_filename = f"3_0_{vessel_mc_name}_vessel_meshdata.csv"
    meshdata_path = os.path.join(directry_path, meshdata_filename)
    df1.to_csv(meshdata_path, index=None) # 圧縮無し

    return vessel_mc_taubin_thick_butterfly

# Cell 10
# vessel_mesh = create_mesh (Case=case, 
#                                  vessel_mc=vessel_mc, 
#                                  vessel_mc_name="vessel_mc", 
#                                  volume_threshold = 5000, voxel_size = 0.5,
#                                  l=0.6307, m=-0.6732, s=10, i=2)

# Cell 12
# pl = pv.Plotter()

# Cell 13
"""
最初はvolume_threshold = 5000としていたが、mca duplicationの症例があり、
volume_threshold = 500へ変更した
"""

mca_mesh = create_mesh (Case=case, 
                                     vessel_mc=mca_mc, 
                                     vessel_mc_name="mca_mc", 
                                     directry_path=intermediate_data1_path, 
                                     volume_threshold = 500, voxel_size = 0.5,
                                     l=0.6307, m=-0.6732, s=10, i=2)

# Cell 14
pl = pv.Plotter()
pl.camera.zoom(1.5)

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

# Cell 17
pl = pv.Plotter()
pl.camera.zoom(1.5)
