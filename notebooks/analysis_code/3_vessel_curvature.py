#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: 3 vessel curvature.ipynb

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
intermediate_data2_path = os.path.join(case_path, 'Intermediate_data2'); tool.create_dir(intermediate_data2_path)
intermediate_data3_path = os.path.join(case_path, 'Intermediate_data3'); tool.create_dir(intermediate_data3_path)

intermediate_data1_3ds_path = os.path.join(case_path, 'Intermediate_data1 3d slicer'); tool.create_dir(intermediate_data1_3ds_path)


mca_intermediate_data1_3ds_path = os.path.join(case_path, 'MCA Intermediate_data1 3d slicer'); tool.create_dir(mca_intermediate_data1_3ds_path)
mca_intermediate_data3_3ds_path = os.path.join(case_path, 'MCA Intermediate_data3 3d slicer'); tool.create_dir(mca_intermediate_data3_3ds_path)
mca_intermediate_data4_3ds_path = os.path.join(case_path, 'MCA Intermediate_data4 3d slicer'); tool.create_dir(mca_intermediate_data4_3ds_path)

# Cell 5
# 血管とcenterlineの読み込み
vessel_mesh_path = os.path.join(intermediate_data1_path, "1_2_mca_mc_taubin_thick_butterfly16.ply")
vessel_mesh = mesh_generation.load_ply_to_polydata(vessel_mesh_path)

centerlines_path = os.path.join(mca_intermediate_data3_3ds_path, "1 vmtk_centerlines.vtp")
centerlines = mesh_generation.load_vtp(centerlines_path)

pv.wrap(vessel_mesh), pv.wrap(centerlines)

# Cell 6
vmtk_CL.check_steplength(centerlines)

# Cell 7
pl = pv.Plotter()
pl.camera.zoom(1)

# Cell 9
# Split centerlines
splitted_centerlines = tool.measure_time(vmtk_CL.vmtk_branch_splitting, centerlines) # vmtk_CL.vmtk_branch_splitting(centerlines)

# Cell 10
# Save splitted centerlines
splitted_centerlines_path = os.path.join(mca_intermediate_data3_3ds_path, "2 splitted_centerlines.vtp")
mesh_generation.save_vtp(splitted_centerlines, splitted_centerlines_path)

# Cell 12
s_factor = 1.2
num_iter = 100

smoothed_centerlines = vmtk_CL.vmtk_centerline_smoothing(splitted_centerlines, num_iteration=num_iter, smooth_factor=s_factor)
merged_splitted_centerlines = vmtk_CL.vmtk_centerlinemerge(smoothed_centerlines, step_length=s_length)
merged_splitted_centerlines = vmtk_CL.vmtk_centerline_attributes(merged_splitted_centerlines)
merged_splitted_centerlines = vmtk_CL.vmtk_compute_geometry(merged_splitted_centerlines)

# Cell 13
pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
pl.camera.zoom(1)
pl.subplot(0, 1)
pl.camera.zoom(1)

# Cell 14
vmtk_CL.check_steplength(merged_splitted_centerlines)

# Cell 15
celldata, pointids, points, pointdata = vmtk_CL.vmtk_convert2numpy(merged_splitted_centerlines)
kappa = pointdata["Curvature"]
tau = pointdata["Torsion"]

celldata, pointids, points, pointdata

# Cell 16
merged_splitted_centerlines_path = os.path.join(mca_intermediate_data3_3ds_path, "3 merged_splitted_centerlines.vtp")
mesh_generation.save_vtp(merged_splitted_centerlines, merged_splitted_centerlines_path)

# Cell 17

# Cell 18
# カラーマップの中心を0にする
max_abs_curvature = np.max(np.abs(kappa))
max_abs_torsion = np.max(np.abs(tau))

pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)
            clim=(0, max_abs_curvature),
            scalar_bar_args={
    "title": "Curvature",
    "n_labels": 10,
    "fmt": '%.2f'})
pl.camera.zoom(2)
pl.subplot(0, 1)
            clim=(-max_abs_torsion, max_abs_torsion),
            scalar_bar_args={
    "title": "Torsion",
    "n_labels": 10,
    "fmt": '%.2f'})
pl.camera.zoom(2)

# Cell 19
removal = 10

# Cell 20
pl = pv.Plotter(shape=(1,2))
pl.subplot(0,0)
for i in range(len(pointids)):
    if len(points[pointids[i]][removal:-removal])==0:
        continue
    else:
                    scalars=kappa[pointids[i]][removal:-removal], 
                    opacity=1, line_width=5, cmap="jet")
                      render_points_as_spheres=True, 
                      point_size=4,
                      opacity=0.5)
pl.camera.zoom(1)
pl.subplot(0,1)
for i in range(len(pointids)):
                scalars=pointdata["Curvature"][pointids[i]], 
                opacity=1, line_width=5, cmap="jet")
                      render_points_as_spheres=True,
                      point_size=5,
                      opacity=0.5)

pl.camera.zoom(1)

# Cell 21
pl = pv.Plotter()
for i in range(len(pointids)):
                scalars=pointdata["Curvature"][pointids[i]], 
                opacity=1, line_width=5, cmap="jet")
    
    if len(points[pointids[i]][removal:-removal])==0:
        continue
    else:
                      render_points_as_spheres=True, 
                      point_size=5,
                      opacity=0.2,
                      color='cyan')
                        render_points_as_spheres=True, 
                        point_size=10,
                        opacity=0.5,
                        color='magenta')
pl.camera.zoom(1)

# Cell 22
remove_ends_pointids = []
for id_group in pointids:
    if len(points[id_group][removal:-removal])==0:
        continue
    else:
        new_id_group = id_group[removal:-removal]
        remove_ends_pointids.append(new_id_group)
remove_ends_pointids

# Cell 23
"""まちがい
remove_ends_pointids = []
for id_group in pointids:
    if len(points[id_group][removal:-removal])==0:
        continue
    else:
        new_id_group = id_group[5:-5]
        remove_ends_pointids.append(new_id_group)
remove_ends_pointids
"""

# Cell 24

pl = pv.Plotter(shape=(1, 2))
for i in range(len(remove_ends_pointids)):
    if len(remove_ends_pointids)==0:
        continue
    else:
        kappa_ = kappa[remove_ends_pointids[i]]
        tau_ = tau[remove_ends_pointids[i]]
        x = np.arange(len(kappa_)) * s_length
        pl.subplot(0, 0)
                    scalars=kappa_, 
                    opacity=1, line_width=5, cmap="jet")
        pl.camera.zoom(2)       
        pl.subplot(0, 1)
                    scalars=tau_, 
                    opacity=1, 
                    line_width=5, cmap="jet")
        pl.camera.zoom(2) 
        
        




# Cell 25
import pandas as pd
pd.DataFrame(celldata)

# Cell 26
for i in range(len(remove_ends_pointids)):

    if len(remove_ends_pointids)==0:
        continue
    else:
        kappa_ = kappa[remove_ends_pointids[i]]
        tau_ = tau[remove_ends_pointids[i]]
        p = points[remove_ends_pointids[i]]
        x,y,z = p.T

        pl = pv.Plotter()
        pl.camera.zoom(2)

        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(x, y, z, "o-")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_box_aspect([1,1,1])
        
        ax2 = fig.add_subplot(132, projection='3d')
        sc2 = ax2.scatter(x, y, z, c=kappa_, cmap='jet', s=10)
        # カラーバーを追加
        cb2 = fig.colorbar(sc2, ax=ax2, label="Curvature")
        # ラベルと軸比
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.set_box_aspect([1, 1, 1])  # 各軸の比率を均一に設定

        ax3 = fig.add_subplot(133, projection='3d')
        sc3 = ax3.scatter(x, y, z, c=tau_, cmap='jet', s=10)
        # カラーバーを追加
        cb3 = fig.colorbar(sc3, ax=ax3, label="Torision")
        # ラベルと軸比
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax3.set_box_aspect([1, 1, 1])



        x = np.arange(len(kappa_)) * s_length

        # CurvatureとTorsionの相関解析(Curvatureが小さい部分でtorsionがスパイクするかどうかを確認)
        curvature = kappa_
        torsion = tau_
        

        # 条件付きのCurvatureでのデータ抽出
        low_curvature_indices = curvature < 0.02
        low_curvature_torsion = torsion[low_curvature_indices]
        # Low curvatureのヒストグラム

# Cell 27
for i in range(len(remove_ends_pointids)):

    if len(remove_ends_pointids)==0:
        continue
    else:
        kappa_ = kappa[remove_ends_pointids[i]]
        tau_ = tau[remove_ends_pointids[i]]

        x = np.arange(len(kappa_)) * s_length

# Cell 28
end = time.time()  # 現在時刻（処理完了後）を取得
# 処理完了後の時刻（終了時間）を表示
終了時間 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))

time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する

# 秒を「時：分」に変換
hours, remainder = divmod(time_diff, 3600)
minutes, seconds = divmod(remainder, 60)
# 処理にかかった時間を「時：分」形式で表示

# Cell 29
tool.notify(case)
