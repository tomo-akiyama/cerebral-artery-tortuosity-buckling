#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: 3 centerline extraction with 3d slicer.ipynb

# Cell 2
# Select the case for preprocessing

case = "IXI546-HH-2450"

# Cell 3

# Cell 4
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

import time
start = time.time() # 現在時刻（処理開始前）を取得
# 開始時間を表示
開始時間 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))

# Cell 5
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
intermediate_data2_path = os.path.join(case_path, 'Intermediate_data2'); tool.create_dir(intermediate_data2_path)
intermediate_data3_path = os.path.join(case_path, 'Intermediate_data3'); tool.create_dir(intermediate_data3_path)

intermediate_data1_3ds_path = os.path.join(case_path, 'Intermediate_data1 3d slicer'); tool.create_dir(intermediate_data1_3ds_path)


mca_intermediate_data1_3ds_path = os.path.join(case_path, 'MCA Intermediate_data1 3d slicer'); tool.create_dir(mca_intermediate_data1_3ds_path)
mca_intermediate_data3_3ds_path = os.path.join(case_path, 'MCA Intermediate_data3 3d slicer'); tool.create_dir(mca_intermediate_data3_3ds_path)

# Cell 6
# Load the images
input_MRA_path = os.path.join(case_path, case + "-MRA.nii.gz")
input_T1_path = os.path.join(case_path, case + "-T1.nii.gz")

img_MRA = ants.image_read(input_MRA_path); img_mra_array = img_MRA.numpy()
img_T1 = ants.image_read(input_T1_path); img_T1_array = img_T1.numpy()

# Show original image



# Cell 7
vessel_mesh_path = os.path.join(intermediate_data1_path, "1_2_mca_mc_taubin_thick_butterfly16.ply")
vessel_mesh = mesh_generation.load_ply_to_polydata(vessel_mesh_path)

pv.wrap(vessel_mesh)

# Cell 8
import json
sources_on_mesh_path = os.path.join(intermediate_data1_3ds_path, "Inlets.mrk.json")
targets_on_mesh_path = os.path.join(intermediate_data1_3ds_path, "Outlets.mrk.json")

with open(sources_on_mesh_path, 'r') as f:
    sources_data = json.load(f)

with open(targets_on_mesh_path, 'r') as f:
    targets_data = json.load(f)

# Cell 9
sources_on_mesh = sources_data.get('markups', [])[0].get('controlPoints', [])
# 位置データを抽出してnumpy配列に変換
positions = [point['position'] for point in sources_on_mesh ]
sources_on_mesh = np.array(positions)
# 位置データを表示

targets_on_mesh = targets_data.get('markups', [])[0].get('controlPoints', [])
# 位置データを抽出してnumpy配列に変換
positions = [point['position'] for point in targets_on_mesh ]
targets_on_mesh = np.array(positions)
# 位置データを表示

# Cell 10
pl = pv.Plotter()
pl.camera.zoom(1)

# Cell 11
pv_path = os.path.join(mca_intermediate_data3_3ds_path, "11 vessel_surface_and_epts.html")
pl.export_html(pv_path)

# Cell 12
# centerlines抽出
centerlines = vmtk_CL.vmtk_extract_centerline(vessel_mesh_path, sources_on_mesh, targets_on_mesh);

# Cell 14
# https://discourse.slicer.org/t/cannot-obtain-the-center-line/18112

# Cell 15
# centerlinesの保存

# mesh_generation.export_vtkpolydata_to_ply(centerlines, centerlines_path)

centerlines_path = os.path.join(mca_intermediate_data3_3ds_path, "1 vmtk_centerlines.vtp")
mesh_generation.save_vtp(centerlines, centerlines_path)

# Cell 16
pl = pv.Plotter()
pl.camera.zoom(1)

# Cell 17
pv_path = os.path.join(mca_intermediate_data3_3ds_path, "12 vessel_surface_and_centerlines.html")
pl.export_html(pv_path)

# Cell 18
end = time.time()  # 現在時刻（処理完了後）を取得
# 処理完了後の時刻（終了時間）を表示
終了時間 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))

time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する

# 秒を「時：分」に変換
hours, remainder = divmod(time_diff, 3600)
minutes, seconds = divmod(remainder, 60)
# 処理にかかった時間を「時：分」形式で表示

# Cell 19
tool.notify(case)
