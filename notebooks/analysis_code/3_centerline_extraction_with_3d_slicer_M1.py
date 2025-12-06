#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: 3 centerline extraction with 3d slicer M1.ipynb

# Cell 2
# Select the case for preprocessing

case = "IXI451-HH-2212"

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

import time
start = time.time() # 現在時刻（処理開始前）を取得
# 開始時間を表示
開始時間 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))

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
intermediate_data2_path = os.path.join(case_path, 'Intermediate_data2'); tool.create_dir(intermediate_data2_path)
intermediate_data3_path = os.path.join(case_path, 'Intermediate_data3'); tool.create_dir(intermediate_data3_path)

intermediate_data1_3ds_path = os.path.join(case_path, 'Intermediate_data1 3d slicer'); tool.create_dir(intermediate_data1_3ds_path)


mca_intermediate_data1_3ds_path = os.path.join(case_path, 'MCA Intermediate_data1 3d slicer'); tool.create_dir(mca_intermediate_data1_3ds_path)
mca_intermediate_data3_3ds_path = os.path.join(case_path, 'MCA Intermediate_data3 3d slicer'); tool.create_dir(mca_intermediate_data3_3ds_path)

# Cell 5
# Load the images
input_MRA_path = os.path.join(case_path, case + "-MRA.nii.gz")
input_T1_path = os.path.join(case_path, case + "-T1.nii.gz")

img_MRA = ants.image_read(input_MRA_path); img_mra_array = img_MRA.numpy()
img_T1 = ants.image_read(input_T1_path); img_T1_array = img_T1.numpy()

# Show original image



# Cell 6
vessel_mesh_path = os.path.join(intermediate_data1_path, "1_2_mca_mc_taubin_thick_butterfly16.ply")
vessel_mesh = mesh_generation.load_ply_to_polydata(vessel_mesh_path)

pv.wrap(vessel_mesh)

# Cell 7
import json
sources_on_mesh_path = os.path.join(intermediate_data3_path, "Inlets.mrk.json")
targets_on_mesh_path = os.path.join(intermediate_data3_path, "Outlets m1.mrk.json")


with open(sources_on_mesh_path, 'r') as f:
    sources_data = json.load(f)

with open(targets_on_mesh_path, 'r') as f:
    targets_data = json.load(f)

# Cell 8
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

# Cell 9
pl = pv.Plotter()
pl.camera.zoom(1)

# Cell 10
# centerlines抽出
centerlines = vmtk_CL.vmtk_extract_centerline(vessel_mesh_path, sources_on_mesh, targets_on_mesh);

# Cell 12
# https://discourse.slicer.org/t/cannot-obtain-the-center-line/18112

# Cell 13
# centerlinesの保存

# mesh_generation.export_vtkpolydata_to_ply(centerlines, centerlines_path)

centerlines_path = os.path.join(intermediate_data3_path, "1 m1 vmtk_centerlines.vtp")
mesh_generation.save_vtp(centerlines, centerlines_path)

# Cell 14
centerlines_properties = vmtk_CL.vmtk_centerline_attributes(centerlines)
centerlines_properties = vmtk_CL.vmtk_compute_geometry(centerlines_properties)

# Cell 15
celldata, pointids, points, pointdata = vmtk_CL.vmtk_convert2numpy(centerlines_properties)
celldata, pointids, points, pointdata

# Cell 16
pointdata['MaximumInscribedSphereRadius']

# Cell 17
try:
    m1p1 = points[pointids[0][0]]
    m1p1_r1 = pointdata['MaximumInscribedSphereRadius'][[pointids[0][0]]][0]  # スカラー値を抽出
    sphere1 = pv.Sphere(radius=m1p1_r1, center=m1p1)
except IndexError:
    m1p1 = None
    m1p1_r1 = None
    sphere1 = None

try:
    m1p2 = points[pointids[1][0]]
    m1p2_r1 = pointdata['MaximumInscribedSphereRadius'][[pointids[1][0]]][0]  # スカラー値を抽出
    sphere2 = pv.Sphere(radius=m1p2_r1, center=m1p2)
except IndexError:
    m1p2 = None
    m1p2_r1 = None
    sphere2 = None


# Cell 18
pl = pv.Plotter()


if sphere1 is not None:  # sphere1がNoneでない場合に追加

if m1p1 is not None:  # m1p1がNoneでない場合に追加

if sphere2 is not None:  # sphere2がNoneでない場合に追加

if m1p2 is not None:  # m1p2がNoneでない場合に追加

pl.camera.zoom(1)

# Cell 20
pv_path = os.path.join(intermediate_data3_path, "m1 vmtk_centerlines.html")
pl.export_html(pv_path)

# Cell 22
m1_length1 = m1p1_r1 + celldata['Length'][0]
m1_length1 = m1_length1 * voxel_size

# Cell 23
try:
    # celldata['Length'][1] が存在するか確認
    length_value = celldata['Length'][1] if len(celldata['Length']) > 1 else np.nan
    m1p2_r1_value = m1p2_r1 if m1p2_r1 is not None else np.nan
    # 計算を実行
    m1_length2 = m1p2_r1_value + length_value
    m1_length2 = m1_length2 * voxel_size
except (IndexError, TypeError):
    # エラーが発生した場合はnanを返す
    m1_length2 = np.nan


# Cell 24
m1_centerlines_properties_path = os.path.join(intermediate_data3_path, "2 centerlines_properties.vtp")
mesh_generation.save_vtp(centerlines_properties, m1_centerlines_properties_path)

# Cell 25
end = time.time()  # 現在時刻（処理完了後）を取得
# 処理完了後の時刻（終了時間）を表示
終了時間 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))

time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する

# 秒を「時：分」に変換
hours, remainder = divmod(time_diff, 3600)
minutes, seconds = divmod(remainder, 60)
# 処理にかかった時間を「時：分」形式で表示

# Cell 26
tool.notify(case)
