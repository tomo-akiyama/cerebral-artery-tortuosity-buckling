#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: 2 surface mesh spm12:cat12 remesh専用.ipynb

# Cell 2
# Select the case for preprocessing

case = "IXI057-HH-1342"

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
t1_path = os.path.join(intermediate_data0_path, '1_1_resampled_T1.nii.gz')
t1 = ants.image_read(t1_path)

preprocessed_img_T1_path = os.path.join(intermediate_data0_path, "mri/m1_1_resampled_T1.nii")
preprocessed_img_T1 = ants.image_read(preprocessed_img_T1_path); preprocessed_img_T1_array = preprocessed_img_T1.numpy()


t1, preprocessed_img_T1

# Cell 7
lt_central_path = os.path.join(intermediate_data0_path, "surf/lh.central.1_1_resampled_T1.gii")
lt_pial_path = os.path.join(intermediate_data0_path, "surf/lh.pial.1_1_resampled_T1.gii")
lt_white_path = os.path.join(intermediate_data0_path, "surf/lh.white.1_1_resampled_T1.gii")

rt_central_path = os.path.join(intermediate_data0_path, "surf/rh.central.1_1_resampled_T1.gii")
rt_pial_path = os.path.join(intermediate_data0_path, "surf/rh.pial.1_1_resampled_T1.gii")
rt_white_path = os.path.join(intermediate_data0_path, "surf/rh.white.1_1_resampled_T1.gii")

lt_central = mesh_generation.gifti_to_vtkpolydata(lt_central_path, preprocessed_img_T1_path)
lt_pial = mesh_generation.gifti_to_vtkpolydata(lt_pial_path, preprocessed_img_T1_path)
lt_white = mesh_generation.gifti_to_vtkpolydata(lt_white_path, preprocessed_img_T1_path)

rt_central = mesh_generation.gifti_to_vtkpolydata(rt_central_path, preprocessed_img_T1_path)
rt_pial = mesh_generation.gifti_to_vtkpolydata(rt_pial_path, preprocessed_img_T1_path)
rt_white = mesh_generation.gifti_to_vtkpolydata(rt_white_path, preprocessed_img_T1_path)

mesh_generation.export_vtkpolydata_to_ply(lt_central, os.path.join(intermediate_data2_path , '01_lt_central.ply'))
mesh_generation.export_vtkpolydata_to_ply(lt_pial, os.path.join(intermediate_data2_path , '01_lt_pial.ply'))
mesh_generation.export_vtkpolydata_to_ply(lt_white, os.path.join(intermediate_data2_path , '01_lt_white.ply'))

mesh_generation.export_vtkpolydata_to_ply(rt_central, os.path.join(intermediate_data2_path , '01_rt_central.ply'))
mesh_generation.export_vtkpolydata_to_ply(rt_pial, os.path.join(intermediate_data2_path , '01_rt_pial.ply'))
mesh_generation.export_vtkpolydata_to_ply(rt_white, os.path.join(intermediate_data2_path , '01_rt_white.ply'))

# Cell 8
lt_brain_surface = pv.wrap(lt_pial)
rt_brain_surface = pv.wrap(rt_pial)

p = pv.Plotter(shape=(1, 2), window_size=[1024*2, 1024])
p.subplot(0, 0)
p.camera.zoom(1.5)
p.subplot(0, 1)
p.camera.zoom(1.5)


pv_path = os.path.join(intermediate_data2_path, "Surface reconstruction by cat12.html")

# Cell 9
# spm12/cat12のmeshのedgeは平均1mmだが、実際には幅がある

# メッシュのユニークなエッジとその長さを計算する関数
def compute_unique_edges_and_lengths(mesh):
    edges = {}
    faces = mesh.faces.reshape(-1, 4)  # 各セルの頂点インデックスを取得
    
    for face in faces:
        # 各三角形の3辺をセットに追加（ソートしてエッジをユニークにする）
        for i, j in [(1, 2), (2, 3), (3, 1)]:
            edge = tuple(sorted([face[i], face[j]]))
            if edge not in edges:
                point1 = mesh.points[edge[0]]
                point2 = mesh.points[edge[1]]
                length = np.linalg.norm(point1 - point2)
                edges[edge] = length
    
    return edges

# 左半球
# メッシュのユニークなエッジとその長さを取得
unique_edges_and_lengths = compute_unique_edges_and_lengths(lt_brain_surface)
num_edges = len(unique_edges_and_lengths)
edge_lengths = list(unique_edges_and_lengths.values())
# 辺の統計情報を表示


# 右半球
# メッシュのユニークなエッジとその長さを取得
unique_edges_and_lengths = compute_unique_edges_and_lengths(rt_brain_surface)
num_edges = len(unique_edges_and_lengths)
edge_lengths = list(unique_edges_and_lengths.values())
# 辺の統計情報を表示

# Cell 10
import pymeshlab

lt_ms = pymeshlab.MeshSet()
lt_ms.load_new_mesh(os.path.join(intermediate_data2_path , '01_lt_pial.ply'));

# Cell 11
# 絶対値を与える方法 PureValue; https://pymeshlab.readthedocs.io/en/latest/classes/pure_value.html#pure-value
lt_ms.meshing_isotropic_explicit_remeshing(iterations = 10, # default = 10
                                           targetlen = pymeshlab.PureValue(0.66), # 0.66 * 0.5 = 0.33mm幅のedgeに統一
                                           featuredeg = 30, # default = 30
                                           maxsurfdist = pymeshlab.PercentageValue(1),
                                           ) # メッシュの単位系の1は、もとのvoxel size = 0.5mmに対応するので、edge長が1mmに統一

# Cell 12
lt_remesh_path = os.path.join(intermediate_data2_path , '01_lt_pial_remesh.ply')
lt_ms.save_current_mesh(lt_remesh_path)

lt_remesh = pv.read(lt_remesh_path)
lt_brain_surface_remesh = pv.wrap(lt_remesh)

p = pv.Plotter(shape=(1, 2), window_size=[1024*2, 1024]) # lt_ms.show_polyscope()
p.subplot(0, 0)
p.camera.zoom(10)
p.subplot(0, 1)
p.camera.zoom(10)

# Cell 13
# メッシュのユニークなエッジとその長さを取得
unique_edges_and_lengths = compute_unique_edges_and_lengths(lt_brain_surface_remesh)

# エッジの数と長さを表示
num_edges = len(unique_edges_and_lengths)
edge_lengths = list(unique_edges_and_lengths.values())

# 辺の統計情報を表示

# Cell 14
                        vert=False,  # 横向きにする
            patch_artist=True,  # 箱を塗りつぶす
            boxprops=dict(facecolor='skyblue', color='blue'),  # 箱の色
            whiskerprops=dict(color='blue'),  # 髭の色
            capprops=dict(color='blue'),  # キャップの色
            flierprops=dict(marker='o', color='red', alpha=0.5),  # 外れ値のマーカー
            medianprops=dict(color='red')  # 中央値の色
           )

# Cell 15
rt_ms = pymeshlab.MeshSet()
rt_ms.load_new_mesh(os.path.join(intermediate_data2_path , '01_rt_pial.ply'));

# 絶対値を与える方法 PureValue; https://pymeshlab.readthedocs.io/en/latest/classes/pure_value.html#pure-value
rt_ms.meshing_isotropic_explicit_remeshing(iterations = 10, # default = 10
                                           targetlen = pymeshlab.PureValue(0.66), # 0.66 * 0.5 = 0.33mm幅のedgeに統一
                                           featuredeg = 30, # default = 30
                                           maxsurfdist = pymeshlab.PercentageValue(1),
                                           ) # メッシュの単位系の1は、もとのvoxel size = 0.5mmに対応するので、edge長が1mmに統一


# remesh後のメッシュを保存, 再読み込み
rt_remesh_path = os.path.join(intermediate_data2_path , '01_rt_pial_remesh.ply')
rt_ms.save_current_mesh(rt_remesh_path)
rt_remesh = pv.read(rt_remesh_path)
rt_brain_surface_remesh = pv.wrap(rt_remesh)

# Cell 16
# メッシュのユニークなエッジとその長さを取得
unique_edges_and_lengths = compute_unique_edges_and_lengths(rt_brain_surface_remesh)

# エッジの数と長さを表示
num_edges = len(unique_edges_and_lengths)
edge_lengths = list(unique_edges_and_lengths.values())

# 辺の統計情報を表示


                        vert=False,  # 横向きにする
            patch_artist=True,  # 箱を塗りつぶす
            boxprops=dict(facecolor='skyblue', color='blue'),  # 箱の色
            whiskerprops=dict(color='blue'),  # 髭の色
            capprops=dict(color='blue'),  # キャップの色
            flierprops=dict(marker='o', color='red', alpha=0.5),  # 外れ値のマーカー
            medianprops=dict(color='red')  # 中央値の色
           )

# Cell 17
tool.notify(case)
