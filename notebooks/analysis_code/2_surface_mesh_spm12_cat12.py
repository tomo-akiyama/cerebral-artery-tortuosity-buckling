#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: 2 surface mesh spm12:cat12.ipynb

# Cell 2
# Select the case for preprocessing

case = "IXI056-HH-1327"

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
import pymeshlab

lt_ms = pymeshlab.MeshSet()
lt_ms.load_new_mesh(os.path.join(intermediate_data2_path , '01_lt_pial.ply'));

rt_ms = pymeshlab.MeshSet()
rt_ms.load_new_mesh(os.path.join(intermediate_data2_path , '01_rt_pial.ply'));


# lt_ms = pymeshlab.MeshSet()

# rt_ms = pymeshlab.MeshSet()

# Cell 18
# バウンディングボックスの確認

# メッシュのバウンディングボックスの最小座標と最大座標を取得
bounding_box_min = lt_ms.current_mesh().bounding_box().min()
bounding_box_max = lt_ms.current_mesh().bounding_box().max()

# バウンディングボックスの対角線の長さを計算
import math
diagonal_length = math.sqrt(
    (bounding_box_max[0] - bounding_box_min[0])**2 +
    (bounding_box_max[1] - bounding_box_min[1])**2 +
    (bounding_box_max[2] - bounding_box_min[2])**2
)


scale = 0.05

# Cell 19
method_name = 'Scale Dependent Quadric Fitting'
patch =10

# Cell 21
lt_ms.compute_curvature_principal_directions_per_vertex(method=method_name, 
    curvcolormethod = 0, # 0: mean curvature   
    scale=pymeshlab.PureValue(patch),  # スケール依存させずに、20(=0.5mm*20=10mm)の長さに設定
    # scale=pymeshlab.PercentageValue(scale),  # スケール依存の方法のためのスケール（5%）
    autoclean=True  # 参照されていない頂点を削除
)
lt_mean_curvature = lt_ms.current_mesh().vertex_scalar_array()
np.save(os.path.join(intermediate_data2_path, "03_lt_mean_curvature.npy"), lt_mean_curvature)

# Cell 22
rt_ms.compute_curvature_principal_directions_per_vertex(method=method_name,
    curvcolormethod = 0, # 0: mean curvature                                                         
    scale=pymeshlab.PureValue(patch),  # スケール依存させずに、20(=0.5mm*20=10mm)の長さに設定
    # scale=pymeshlab.PercentageValue(scale),  # スケール依存の方法のためのスケール（5%）
    autoclean=True  # 参照されていない頂点を削除
)
rt_mean_curvature = rt_ms.current_mesh().vertex_scalar_array()
np.save(os.path.join(intermediate_data2_path, "03_rt_mean_curvature.npy"), rt_mean_curvature)

# Cell 23


####################################################################################################
# 左側を例に確認する
data = lt_mean_curvature
brain_surface = lt_brain_surface

max = data.max()
min = data.min()

percentile_10 = np.percentile(data, 10)
percentile_90 = np.percentile(data, 90)

# 箱ヒゲ図
                        vert=False,  # 横向きにする
            patch_artist=True,  # 箱を塗りつぶす
            boxprops=dict(facecolor='skyblue', color='blue'),  # 箱の色
            whiskerprops=dict(color='blue'),  # 髭の色
            capprops=dict(color='blue'),  # キャップの色
            flierprops=dict(marker='o', color='red', alpha=0.5),  # 外れ値のマーカー
            medianprops=dict(color='red')  # 中央値の色
           )
# グラフを表示

filtered_MC_10 = tool.remove_outliers_10(data)
filtered_MC_iqr = tool.remove_outliers_iqr(data)

brain_surface.point_data["MC"] = data
brain_surface.point_data["filtered_MC_10"] = filtered_MC_10
brain_surface.point_data["filtered_MC_iqr"] = filtered_MC_iqr


pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)

# 最大絶対値を取得
max = np.max(np.abs(filtered_MC_10))

            clim=(-max, max))
pl.camera.zoom(2)
pl.subplot(0, 1)

# 最大絶対値を取得
max = np.max(np.abs(filtered_MC_iqr))

            clim=(-max, max))
pl.camera.zoom(2)

pv_path = os.path.join(intermediate_data2_path, "02_lt_mean_curvature.html")
pl.export_html(pv_path)


####################################################################################################
# 追加
rt_brain_surface.point_data["MC"] = rt_mean_curvature

# Cell 24
lt_ms.compute_curvature_principal_directions_per_vertex(method=method_name, 
    curvcolormethod = 1, # 1: gaussian curvature
    scale=pymeshlab.PureValue(patch),  # スケール依存させずに、20(=0.5mm*20=10mm)の長さに設定
    autoclean=True  # 参照されていない頂点を削除
)
lt_gaussian_curvature = lt_ms.current_mesh().vertex_scalar_array()
np.save(os.path.join(intermediate_data2_path, "03_lt_gaussian_curvature.npy"), lt_gaussian_curvature)

rt_ms.compute_curvature_principal_directions_per_vertex(method=method_name,
    curvcolormethod = 1, # 1: gaussian curvature
    scale=pymeshlab.PureValue(patch),  # スケール依存させずに、20(=0.5mm*20=10mm)の長さに設定
    autoclean=True # 参照されていない頂点を削除
)
rt_gaussian_curvature = rt_ms.current_mesh().vertex_scalar_array()
np.save(os.path.join(intermediate_data2_path, "03_rt_gaussian_curvature.npy"), rt_gaussian_curvature)


####################################################################################################
# 左側を例に確認する
data = lt_gaussian_curvature

max = data.max()
min = data.min()

percentile_10 = np.percentile(data, 10)
percentile_90 = np.percentile(data, 90)

# 箱ヒゲ図
                        vert=False,  # 横向きにする
            patch_artist=True,  # 箱を塗りつぶす
            boxprops=dict(facecolor='skyblue', color='blue'),  # 箱の色
            whiskerprops=dict(color='blue'),  # 髭の色
            capprops=dict(color='blue'),  # キャップの色
            flierprops=dict(marker='o', color='red', alpha=0.5),  # 外れ値のマーカー
            medianprops=dict(color='red')  # 中央値の色
           )
# グラフを表示

filtered_GC_10 = tool.remove_outliers_10(data)
filtered_GC_iqr = tool.remove_outliers_iqr(data)

brain_surface.point_data["GC"] = data
brain_surface.point_data["filtered_GC_10"] = filtered_GC_10
brain_surface.point_data["filtered_GC_iqr"] = filtered_GC_iqr


pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)

# 最大絶対値を取得
max = np.max(np.abs(filtered_GC_10))

            clim=(-max, max))
pl.camera.zoom(2)
pl.subplot(0, 1)

# 最大絶対値を取得
max = np.max(np.abs(filtered_GC_iqr))

            clim=(-max, max))
pl.camera.zoom(2)

pv_path = os.path.join(intermediate_data2_path, "02_lt_gaussian_curvature .html")
pl.export_html(pv_path)


####################################################################################################
# 追加
rt_brain_surface.point_data["GC"] = rt_gaussian_curvature

# Cell 25
lt_ms.compute_curvature_principal_directions_per_vertex(method=method_name, 
    curvcolormethod = 3, # 3: max curvature
    scale=pymeshlab.PureValue(patch),  # スケール依存させずに、20(=0.5mm*20=10mm)の長さに設定
    autoclean=True  # 参照されていない頂点を削除
)
lt_k1 = lt_ms.current_mesh().vertex_scalar_array()
np.save(os.path.join(intermediate_data2_path, "03_lt_k1.npy"), lt_k1)

rt_ms.compute_curvature_principal_directions_per_vertex(method=method_name,
    curvcolormethod = 3, # 3: max curvature
    scale=pymeshlab.PureValue(patch),  # スケール依存させずに、20(=0.5mm*20=10mm)の長さに設定
    autoclean=True  # 参照されていない頂点を削除
)
rt_k1 = rt_ms.current_mesh().vertex_scalar_array()
np.save(os.path.join(intermediate_data2_path, "03_rt_k1.npy"), rt_k1)


####################################################################################################
# 左側を例に確認する
data = lt_k1

max = data.max()
min = data.min()

percentile_10 = np.percentile(data, 10)
percentile_90 = np.percentile(data, 90)

# 箱ヒゲ図
                        vert=False,  # 横向きにする
            patch_artist=True,  # 箱を塗りつぶす
            boxprops=dict(facecolor='skyblue', color='blue'),  # 箱の色
            whiskerprops=dict(color='blue'),  # 髭の色
            capprops=dict(color='blue'),  # キャップの色
            flierprops=dict(marker='o', color='red', alpha=0.5),  # 外れ値のマーカー
            medianprops=dict(color='red')  # 中央値の色
           )
# グラフを表示

filtered_k1_10 = tool.remove_outliers_10(data)
filtered_k1_iqr = tool.remove_outliers_iqr(data)

brain_surface.point_data["k1"] = data
brain_surface.point_data["filtered_k1_10"] = filtered_k1_10
brain_surface.point_data["filtered_k1_iqr"] = filtered_k1_iqr


pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)

# 最大絶対値を取得
max = np.max(np.abs(filtered_k1_10))

            clim=(-max, max))
pl.camera.zoom(2)
pl.subplot(0, 1)

# 最大絶対値を取得
max = np.max(np.abs(filtered_k1_iqr))

            clim=(-max, max))
pl.camera.zoom(2)

pv_path = os.path.join(intermediate_data2_path, "02_lt_k1.html")
pl.export_html(pv_path)


####################################################################################################
# 追加
rt_brain_surface.point_data["k1"] = rt_k1

# Cell 26
lt_ms.compute_curvature_principal_directions_per_vertex(method=method_name, 
    curvcolormethod = 2, # 2: min curvature
    scale=pymeshlab.PureValue(patch),  # スケール依存させずに、20(=0.5mm*20=10mm)の長さに設定
    autoclean=True  # 参照されていない頂点を削除
)
lt_k2 = lt_ms.current_mesh().vertex_scalar_array()
np.save(os.path.join(intermediate_data2_path, "03_lt_k2.npy"), lt_k2)

rt_ms.compute_curvature_principal_directions_per_vertex(method=method_name,
    curvcolormethod = 2, # 2: min curvature
    scale=pymeshlab.PureValue(patch),  # スケール依存させずに、20(=0.5mm*20=10mm)の長さに設定
    autoclean=True  # 参照されていない頂点を削除
)
rt_k2 = rt_ms.current_mesh().vertex_scalar_array()
np.save(os.path.join(intermediate_data2_path, "03_rt_k2.npy"), rt_k2)


####################################################################################################
# 左側を例に確認する
data = lt_k2

max = data.max()
min = data.min()

percentile_10 = np.percentile(data, 10)
percentile_90 = np.percentile(data, 90)

# 箱ヒゲ図
                        vert=False,  # 横向きにする
            patch_artist=True,  # 箱を塗りつぶす
            boxprops=dict(facecolor='skyblue', color='blue'),  # 箱の色
            whiskerprops=dict(color='blue'),  # 髭の色
            capprops=dict(color='blue'),  # キャップの色
            flierprops=dict(marker='o', color='red', alpha=0.5),  # 外れ値のマーカー
            medianprops=dict(color='red')  # 中央値の色
           )
# グラフを表示

filtered_k2_10 = tool.remove_outliers_10(data)
filtered_k2_iqr = tool.remove_outliers_iqr(data)

brain_surface.point_data["k2"] = data
brain_surface.point_data["filtered_k2_10"] = filtered_k2_10
brain_surface.point_data["filtered_k2_iqr"] = filtered_k2_iqr


pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)

# 最大絶対値を取得
max = np.max(np.abs(filtered_k2_10))

            clim=(-max, max))
pl.camera.zoom(2)
pl.subplot(0, 1)

# 最大絶対値を取得
max = np.max(np.abs(filtered_k2_iqr))

            clim=(-max, max))
pl.camera.zoom(2)

pv_path = os.path.join(intermediate_data2_path, "02_lt_k2.html")
pl.export_html(pv_path)


####################################################################################################
# 追加
rt_brain_surface.point_data["k2"] = rt_k2

# Cell 27
lt_ms.compute_curvature_principal_directions_per_vertex(method=method_name, 
    curvcolormethod = 4, # 4: shape index
    scale=pymeshlab.PureValue(patch),  # スケール依存させずに、20(=0.5mm*20=10mm)の長さに設定
    autoclean=True  # 参照されていない頂点を削除
)
lt_shape_index = lt_ms.current_mesh().vertex_scalar_array()
np.save(os.path.join(intermediate_data2_path, "03_lt_shape_index.npy"), lt_shape_index)

rt_ms.compute_curvature_principal_directions_per_vertex(method=method_name,
    curvcolormethod = 4, # 4: shape index
    scale=pymeshlab.PureValue(patch),  # スケール依存させずに、20(=0.5mm*20=10mm)の長さに設定
    autoclean=True  # 参照されていない頂点を削除
)
rt_shape_index = rt_ms.current_mesh().vertex_scalar_array()
np.save(os.path.join(intermediate_data2_path, "03_rt_shape_index.npy"), rt_shape_index)


####################################################################################################
# 左側を例に確認する
data = lt_shape_index # -1から1の範囲なので、外れ値の基準を変える?

max = data.max()
min = data.min()

percentile_10 = np.percentile(data, 10)
percentile_90 = np.percentile(data, 90)

# 箱ヒゲ図
                        vert=False,  # 横向きにする
            patch_artist=True,  # 箱を塗りつぶす
            boxprops=dict(facecolor='skyblue', color='blue'),  # 箱の色
            whiskerprops=dict(color='blue'),  # 髭の色
            capprops=dict(color='blue'),  # キャップの色
            flierprops=dict(marker='o', color='red', alpha=0.5),  # 外れ値のマーカー
            medianprops=dict(color='red')  # 中央値の色
           )
# グラフを表示

# filtered_shape_index_10 = tool.remove_outliers_10(data)
# filtered_shape_index_iqr = tool.remove_outliers_iqr(data)

brain_surface.point_data["shape_index"] = data
# brain_surface.point_data["filtered_shape_index_10"] = filtered_shape_index_10
# brain_surface.point_data["filtered_shape_index_iqr"] = filtered_shape_index_iqr


pl = pv.Plotter()
            clim=(-1, 1))
pl.camera.zoom(2)

pv_path = os.path.join(intermediate_data2_path, "02_lt_shape_index.html")
pl.export_html(pv_path)


####################################################################################################
# 追加
rt_brain_surface.point_data["shape_index"] = rt_shape_index

# Cell 28
lt_ms.compute_curvature_principal_directions_per_vertex(method=method_name, 
    curvcolormethod = 5, # 5: curvedness
    scale=pymeshlab.PureValue(patch),  # スケール依存させずに、20(=0.5mm*20=10mm)の長さに設定
    autoclean=True  # 参照されていない頂点を削除
)
lt_curvedness = lt_ms.current_mesh().vertex_scalar_array()
np.save(os.path.join(intermediate_data2_path, "03_lt_curvedness.npy"), lt_curvedness)

rt_ms.compute_curvature_principal_directions_per_vertex(method=method_name,
    curvcolormethod = 5, # 5: curvedness
    scale=pymeshlab.PureValue(patch),  # スケール依存させずに、20(=0.5mm*20=10mm)の長さに設定
    autoclean=True  # 参照されていない頂点を削除
)
rt_curvedness = rt_ms.current_mesh().vertex_scalar_array()
np.save(os.path.join(intermediate_data2_path, "03_rt_curvedness.npy"), rt_curvedness)


####################################################################################################
# 左側を例に確認する
data = lt_curvedness

max = data.max()
min = data.min()

percentile_10 = np.percentile(data, 10)
percentile_90 = np.percentile(data, 90)

# 箱ヒゲ図
                        vert=False,  # 横向きにする
            patch_artist=True,  # 箱を塗りつぶす
            boxprops=dict(facecolor='skyblue', color='blue'),  # 箱の色
            whiskerprops=dict(color='blue'),  # 髭の色
            capprops=dict(color='blue'),  # キャップの色
            flierprops=dict(marker='o', color='red', alpha=0.5),  # 外れ値のマーカー
            medianprops=dict(color='red')  # 中央値の色
           )
# グラフを表示

filtered_curvedness_10 = tool.remove_outliers_10(data)
filtered_curvedness_iqr = tool.remove_outliers_iqr(data)

brain_surface.point_data["curvedness"] = data
brain_surface.point_data["filtered_curvedness_10"] = filtered_curvedness_10
brain_surface.point_data["filtered_curvedness_iqr"] = filtered_curvedness_iqr


pl = pv.Plotter(shape=(1, 2))
pl.subplot(0, 0)

# 最大絶対値を取得
max = np.max(np.abs(filtered_curvedness_10))

            clim=(0, max))
pl.camera.zoom(2)
pl.subplot(0, 1)

# 最大絶対値を取得
max = np.max(np.abs(filtered_curvedness_iqr))

            clim=(0, max))
pl.camera.zoom(2)

pv_path = os.path.join(intermediate_data2_path, "02_lt_curvedness.html")
pl.export_html(pv_path)


####################################################################################################
# 追加
rt_brain_surface.point_data["curvedness"] = rt_curvedness

# Cell 29
end = time.time()  # 現在時刻（処理完了後）を取得
# 処理完了後の時刻（終了時間）を表示
終了時間 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))

time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する

# 秒を「時：分」に変換
hours, remainder = divmod(time_diff, 3600)
minutes, seconds = divmod(remainder, 60)
# 処理にかかった時間を「時：分」形式で表示

# Cell 30
tool.notify(case)

# Cell 31
"""

# Cell 33
# 血管とcenterlineの読み込み
intermediate_data10_mca_path = os.path.join(case_path, 'MCA Intermediate_data10 3d slicer')
vessel_mesh_path = os.path.join(intermediate_data10_mca_path, "vessel_mc_taubin_thick_butterfly.ply")
vessel_mesh = mesh_generation.load_ply_to_polydata(vessel_mesh_path)

intermediate_data20_mca_path = os.path.join(case_path, 'MCA Intermediate_data20 3d slicer')
centerlines_path = os.path.join(intermediate_data20_mca_path, "vmtk_centerlines.vtp")
centerlines = mesh_generation.load_vtp(centerlines_path)

intermediate_data30_mca_path = os.path.join(case_path, 'MCA Intermediate_data30 3d slicer')
merged_splitted_centerlines_path = os.path.join(intermediate_data30_mca_path, "merged_splitted_centerlines.vtp")
merged_splitted_centerlines = mesh_generation.load_vtp(merged_splitted_centerlines_path)

# Cell 34
pl = pv.Plotter(shape=(1, 2), window_size=[1024*2, 1024])
pl.subplot(0, 0)
pl.camera.zoom(1.5)
pl.subplot(0, 1)
pl.camera.zoom(1.5)

# Cell 35
celldata, pointids, points, pointdata = vmtk_CL.vmtk_convert2numpy(merged_splitted_centerlines)

EdgeArray = pointdata["EdgeArray"]
EdgePCoordArray = pointdata["EdgePCoordArray"]
radius = pointdata["MaximumInscribedSphereRadius"]
TCoords = pointdata["TCoords"]
abscissas = pointdata["abscissas"]
normals = pointdata["normals"]

kappa = pointdata["Curvature"]
tau = pointdata["Torsion"]

FrenetTangent = pointdata["FrenetTangent"]
FrenetNormal = pointdata["FrenetNormal"]
FrenetBinormal = pointdata["FrenetBinormal"]

# Cell 36
s_factor = 1.2
num_iter = 100

removal = 10

removed_pointids = tool.remove_end_points(pointids, removal)

# Cell 37
removed_pts = []
removed_kappa = []
removed_tau = []
removed_radius = []
removed_FrenetTangent = []
removed_FrenetNormal = []
removed_FrenetBinormal = []


for ids in removed_pointids:
    new_pts = points[ids]; removed_pts.append(new_pts)
    new_kappa = kappa[ids]; removed_kappa.append(new_kappa)
    new_tau = tau[ids]; removed_tau.append(new_tau)
    new_radius = radius[ids]; removed_radius.append(new_radius)
    new_FrenetTangent = FrenetTangent[ids]; removed_FrenetTangent.append(new_FrenetTangent)
    new_FrenetNormal = FrenetNormal[ids]; removed_FrenetNormal.append(new_FrenetNormal)
    new_FrenetBinormal = FrenetBinormal[ids]; removed_FrenetBinormal.append(new_FrenetBinormal)   

# Cell 38
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.hist(np.concatenate(removed_kappa), bins=100, color='blue', alpha=0.7)
ax1.set_title('Curvature')

ax2.hist(np.concatenate(removed_tau), bins=100, color='green', alpha=0.7)
ax2.set_title('Torsion')

ax3.hist(np.concatenate(removed_radius), bins=100, color='red', alpha=0.7)
ax3.set_title('Radius')


# Cell 39
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

for ids in removed_pointids:
    new_kappa = kappa[ids]
    new_tau = tau[ids]
    new_radius = radius[ids]

    arch  = np.arange(len(ids)) * s_length * 0.5 # 0.5mmはspacing

    ax1.plot(arch, new_kappa)
    ax1.set_title("Curvature")
    ax2.plot(arch, new_tau)
    ax2.set_title("Torsion")
    ax3.plot(arch, new_radius)
    ax3.set_title("Radius")


# Cell 40
pl = pv.Plotter()
for ids in removed_pointids:
    new_kappa = kappa[ids]    
    new_pts = points[ids]
    segment = pv.PolyData(new_pts)
    segment["Curvature"] = new_kappa


pv_path = os.path.join(intermediate_data11_path, "04_curvature.html")
pl.export_html(pv_path)

# Cell 41
pointids_combined = np.concatenate(pointids)
pointids_sorted = np.sort(pointids_combined)
pointids_unique = np.unique(pointids_sorted)

new_pointids_combined = np.concatenate(removed_pointids)
new_pointids_sorted = np.sort(new_pointids_combined)
new_pointids_unique = np.unique(new_pointids_sorted)

new_ids = np.arange(len(new_pointids_unique))

mask = np.isin(pointids_unique, new_pointids_unique)

# maskを使ってkappa, tau,...の対応する値を抽出
n_EdgeArray = EdgeArray[mask]
n_EdgePCoordArray = EdgePCoordArray[mask]
n_radius = radius[mask]
n_TCoords = TCoords[mask]
n_abscissas = abscissas[mask]
n_normals = normals[mask]

n_kappa = kappa[mask]
n_tau= tau[mask]

n_FrenetTangent = FrenetTangent[mask]
n_FrenetNormal = FrenetNormal[mask]
n_FrenetBinormal = FrenetBinormal[mask]

n_points = points[mask]

# Cell 42
# # 脳表の頂点と三角形を取得
# lt_vertices, rt_triangles = mesh_generation.get_vertices_triangles(lt_brain_surface)
# rt_vertices, rt_triangles = mesh_generation.get_vertices_triangles(rt_brain_surface)

# Cell 43
lt_brain_surface

# Cell 44
rt_brain_surface

# Cell 45
# Extract vertices
lt_vertices = lt_brain_surface.points
rt_vertices = rt_brain_surface.points

# Extract curvature values in the left hemisphere
lt_MC_values = lt_brain_surface.point_data['MC']
lt_GC_values = lt_brain_surface.point_data['GC']
lt_k1_value = lt_brain_surface.point_data['k1']
lt_k2_value = lt_brain_surface.point_data['k2']
lt_shape_index = lt_brain_surface.point_data['shape_index']
lt_curvedness = lt_brain_surface.point_data['curvedness']

# Extract curvature values in the right hemisphere
rt_MC_values = rt_brain_surface.point_data['MC']
rt_GC_values = rt_brain_surface.point_data['GC']
rt_k1_value = rt_brain_surface.point_data['k1']
rt_k2_value = rt_brain_surface.point_data['k2']
rt_shape_index = rt_brain_surface.point_data['shape_index']
rt_curvedness = rt_brain_surface.point_data['curvedness']

# Concentrate the curvature values
whole_vertices = np.concatenate([lt_vertices, rt_vertices])
whole_MC_values = np.concatenate([lt_MC_values, rt_MC_values])
whole_GC_values = np.concatenate([lt_GC_values, rt_GC_values])
whole_k1_value = np.concatenate([lt_k1_value, rt_k1_value])
whole_k2_value = np.concatenate([lt_k2_value, rt_k2_value])
whole_shape_index = np.concatenate([lt_shape_index, rt_shape_index])
whole_curvedness = np.concatenate([lt_curvedness, rt_curvedness])

# Cell 46
# 脳表のメッシュ点と血管の中心線点のマッピング
from scipy.spatial import KDTree

# Create a KDTree for efficient nearest-neighbor search
brain_tree = KDTree(whole_vertices)

# Find the nearest brain surface vertex for each vessel centerline point
distances, indices = brain_tree.query(n_points)

# Map brain surface curvatures to vessel centerline points
mapped_brain_mean_curvatures = whole_MC_values[indices]
mapped_brain_gaussian_curvatures = whole_GC_values[indices]
mapped_brain_k1 = whole_k1_value[indices]
mapped_brain_k2 = whole_k2_value[indices]
mapped_brain_shape_index = whole_shape_index[indices]
mapped_brain_curvedness = whole_curvedness[indices]

# Cell 47

# Cell 48
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Calculate the Pearson correlation coefficient
correlation_MC, p_value_MC = pearsonr(n_kappa, mapped_brain_mean_curvatures)
correlation_GC, p_value_GC = pearsonr(n_kappa, mapped_brain_gaussian_curvatures)
correlation_k1, p_value_k1 = pearsonr(n_kappa, mapped_brain_k1)
correlation_k2, p_value_k2 = pearsonr(n_kappa, mapped_brain_k2)
correlation_shape_index, p_value_shape_index = pearsonr(n_kappa, mapped_brain_shape_index)
correlation_curvedness, p_value_curvedness = pearsonr(n_kappa, mapped_brain_curvedness)













# Cell 49
# 重みづけ平均する半径のイメージを掴む
radius = 10 # 10*0.5(spacing) = 5mm in millimeters
sphere = pv.Sphere(center=[286, 220, 372], radius=radius)
representative_pt101 = n_points[101]; sphere101 = pv.Sphere(center=representative_pt101, radius=radius)
representative_pt17533 = n_points[17533]; sphere17533 = pv.Sphere(center=representative_pt17533, radius=radius)


pl = pv.Plotter()
pl.camera.zoom(1.5)

# Cell 50
# Create a KDTree for efficient nearest-neighbor search
brain_tree = KDTree(whole_vertices)

# Find the nearest brain surface vertex for each vessel centerline point
distances, indices = brain_tree.query(n_points)
# Nearest brain surface vertex list
nearest_brain_surface_vertices = whole_vertices[indices]

# Map brain surface curvatures to vessel centerline points
mapped_brain_mean_curvatures = whole_MC_values[indices]
mapped_brain_gaussian_curvatures = whole_GC_values[indices]
mapped_brain_k1 = whole_k1_value[indices]
mapped_brain_k2 = whole_k2_value[indices]
mapped_brain_shape_index = whole_shape_index[indices]
mapped_brain_curvedness = whole_curvedness[indices]

# Cell 51
#  距離重み付け平均
# Create a KDTree for efficient nearest-neighbor search
brain_tree = KDTree(whole_vertices)

# Define the radius for local surface approximation
radius = 10  # 10*0.5(spacing) = 5mm in millimeters

# Initialize lists to store matched curvatures
mapped_brain_weighted_mean_curvatures = []
mapped_brain_weighted_gaussian_curvatures = []
mapped_brain_weighted_k1 = []
mapped_brain_weighted_k2 = []
mapped_brain_weighted_shape_index = []
mapped_brain_weighted_curvedness = []

# Iterate over each vessel centerline point
for i, point in enumerate(n_points):
    # Find all brain surface vertices within the specified radius
    indices = brain_tree.query_ball_point(point, radius)
    
    if indices:
        # Calculate the distances to the centerline point
        distances = np.linalg.norm(whole_vertices[indices] - point, axis=1)
        
        # Calculate weights based on distances (e.g., inverse distance)
        weights = 1 / (distances + 1e-5)  # Adding a small constant to avoid division by zero
        
        # Normalize the weights
        weights /= np.sum(weights)
        
        # Calculate the weighted curvature of the brain surface within the radius
        local_mean_curvatures = whole_MC_values[indices]
        weighted_mean_brain_curvature = np.sum(local_mean_curvatures * weights)
        local_gaussian_curvatures = whole_GC_values[indices]
        weighted_gaussian_brain_curvature = np.sum(local_gaussian_curvatures * weights)
        local_k1 = whole_k1_value[indices]
        weighted_k1 = np.sum(local_k1 * weights)
        local_k2 = whole_k2_value[indices]
        weighted_k2 = np.sum(local_k2 * weights)
        local_shape_index = whole_shape_index[indices]
        weighted_shape_index = np.sum(local_shape_index * weights)
        local_curvedness = whole_curvedness[indices]
        weighted_curvedness = np.sum(local_curvedness * weights)
        
        # Store the weighted mean curvature and the vessel curvature
        mapped_brain_weighted_mean_curvatures.append(weighted_mean_brain_curvature)
        mapped_brain_weighted_gaussian_curvatures.append(weighted_gaussian_brain_curvature)
        mapped_brain_weighted_k1.append(weighted_k1)
        mapped_brain_weighted_k2.append(weighted_k2)
        mapped_brain_weighted_shape_index.append(weighted_shape_index)
        mapped_brain_weighted_curvedness.append(weighted_curvedness)

# Convert lists to numpy arrays for analysis
mapped_brain_weighted_mean_curvatures = np.array(mapped_brain_weighted_mean_curvatures)
mapped_brain_weighted_gaussian_curvatures = np.array(mapped_brain_weighted_gaussian_curvatures)
mapped_brain_weighted_k1 = np.array(mapped_brain_weighted_k1)
mapped_brain_weighted_k2 = np.array(mapped_brain_weighted_k2)
mapped_brain_weighted_shape_index = np.array(mapped_brain_weighted_shape_index)
mapped_brain_weighted_curvedness = np.array(mapped_brain_weighted_curvedness)

# Cell 52
# Calculate the Pearson correlation coefficient
correlation_weighted_mean_curvature, p_value_weighted_mean_curvature = pearsonr(n_kappa, mapped_brain_weighted_mean_curvatures)
correlation_weighted_gaussian_curvature, p_value_weighted_gaussian_curvature = pearsonr(n_kappa, mapped_brain_weighted_gaussian_curvatures)
correlation_weighted_k1, p_value_weighted_k1 = pearsonr(n_kappa, mapped_brain_weighted_k1)
correlation_weighted_k2, p_value_weighted_k2 = pearsonr(n_kappa, mapped_brain_weighted_k2)
correlation_weighted_shape_index, p_value_weighted_shape_index = pearsonr(n_kappa, mapped_brain_weighted_shape_index)
correlation_weighted_curvedness, p_value_weighted_curvedness = pearsonr(n_kappa, mapped_brain_weighted_curvedness)












# Cell 54
# 統計的検定を用いて、血管の蛇行の強度（高い曲率を持つ部位）と脳表面の特定の形状の間に有意な差があるかどうかを評価

# 高曲率領域と低曲率領域に分割
high_curvature_indices = n_kappa > np.median(n_kappa)
low_curvature_indices = n_kappa <= np.median(n_kappa)

from scipy.stats import ttest_ind

# 高曲率領域と低曲率領域の血管曲率を比較
# ここでは加重平均の曲率を使用
t_stat_MC, p_value_MC = ttest_ind(mapped_brain_weighted_mean_curvatures[high_curvature_indices], mapped_brain_weighted_mean_curvatures[low_curvature_indices])

t_stat_GC, p_value_GC = ttest_ind(mapped_brain_weighted_gaussian_curvatures[high_curvature_indices], mapped_brain_weighted_gaussian_curvatures[low_curvature_indices])

t_stat_k1, p_value_k1 = ttest_ind(mapped_brain_weighted_k1[high_curvature_indices], mapped_brain_weighted_k1[low_curvature_indices])

t_stat_k2, p_value_k2 = ttest_ind(mapped_brain_weighted_k2[high_curvature_indices], mapped_brain_weighted_k2[low_curvature_indices])

t_stat_shape_index, p_value_shape_index = ttest_ind(mapped_brain_weighted_shape_index[high_curvature_indices], mapped_brain_weighted_shape_index[low_curvature_indices])

t_stat_curvedness, p_value_curvedness = ttest_ind(mapped_brain_weighted_curvedness[high_curvature_indices], mapped_brain_weighted_curvedness[low_curvature_indices])


# Cell 55
high_curvature = n_kappa[high_curvature_indices]
low_curvature = n_kappa[low_curvature_indices]

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(low_curvature, mapped_brain_weighted_mean_curvatures[low_curvature_indices], s=1, alpha=0.5, color='blue')
ax1.set_title('low curvature')
ax1.set_xlabel('Vessel Curvature')
ax1.set_ylabel('Mean Curvature')

ax2.scatter(high_curvature, mapped_brain_weighted_mean_curvatures[high_curvature_indices], s=1, alpha=0.5, color='red')
ax2.set_title('high curvature')
ax2.set_xlabel('Vessel Curvature')
ax2.set_ylabel('Mean Curvature')


ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(low_curvature, mapped_brain_weighted_gaussian_curvatures[low_curvature_indices], s=1, alpha=0.5, color='blue')
ax1.set_title('low curvature')
ax1.set_xlabel('Vessel Curvature')
ax1.set_ylabel('Gaussian Curvature')

ax2.scatter(high_curvature, mapped_brain_weighted_gaussian_curvatures[high_curvature_indices], s=1, alpha=0.5, color='red')
ax2.set_title('high curvature')
ax2.set_xlabel('Vessel Curvature')
ax2.set_ylabel('Gaussian Curvature')


ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(low_curvature, mapped_brain_weighted_k1[low_curvature_indices], s=1, alpha=0.5, color='blue')
ax1.set_title('low curvature')
ax1.set_xlabel('Vessel Curvature')
ax1.set_ylabel('k1 Max Curvature')

ax2.scatter(high_curvature, mapped_brain_weighted_k1[high_curvature_indices], s=1, alpha=0.5, color='red')
ax2.set_title('high curvature')
ax2.set_xlabel('Vessel Curvature')
ax2.set_ylabel('k1 Max Curvature')


ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(low_curvature, mapped_brain_weighted_k2[low_curvature_indices], s=1, alpha=0.5, color='blue')
ax1.set_title('low curvature')
ax1.set_xlabel('Vessel Curvature')
ax1.set_ylabel('k2 Min Curvature')

ax2.scatter(high_curvature, mapped_brain_weighted_k2[high_curvature_indices], s=1, alpha=0.5, color='red')
ax2.set_title('high curvature')
ax2.set_xlabel('Vessel Curvature')
ax2.set_ylabel('k2 Min Curvature')


ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(low_curvature, mapped_brain_weighted_shape_index[low_curvature_indices], s=1, alpha=0.5, color='blue')
ax1.set_title('low curvature')
ax1.set_xlabel('Vessel Curvature')
ax1.set_ylabel('Shape Index')

ax2.scatter(high_curvature, mapped_brain_weighted_shape_index[high_curvature_indices], s=1, alpha=0.5, color='red')
ax2.set_title('high curvature')
ax2.set_xlabel('Vessel Curvature')
ax2.set_ylabel('Shape Index')


# Cell 56
pl = pv.Plotter()

# Cell 57
# ヒストグラムをみて高曲率領域と低曲率領域に分割
threshold = 0.2
high_curvature_indices = n_kappa > threshold
low_curvature_indices = n_kappa <= threshold

pl = pv.Plotter()

# Cell 59
# Centerlinesの軸方向の曲率を計算
pl = pv.Plotter()

# 最近傍の点を共有している可能性が高いので、間引く
thinning = 10   

# Cell 60
# https://github.com/CristianoPizzamiglio/circle-fitting-3d
from circle_fitting_3d import Circle3D 
axial_radius = []
axial_center = []
axial_curv = []

thinning_ids = new_ids[::thinning]

for id in thinning_ids:
    if id <= thinning_ids[-3]:
        p0 = nearest_brain_surface_vertices[id]
        p1 = nearest_brain_surface_vertices[id+thinning]
        p2 = nearest_brain_surface_vertices[id+thinning*2]

        P0 = n_points[id]
        P1 = n_points[id+thinning]
        P2 = n_points[id+thinning*2]

        if np.linalg.norm(P0 -  P1) > 5 or np.linalg.norm(P1 -  P2) > 5: # 別のsegmentを跨ぐ場合はnanをあてる
            axial_radius.append(np.nan)
            axial_center.append(np.nan)
            axial_curv.append(np.nan)
        elif any([(p0 == p1).all(), (p1 == p2).all(), (p0 == p1).all()]) == True: # 最近傍の点が同じ場合はnanをあてる
            axial_radius.append(np.nan)
            axial_center.append(np.nan)
            axial_curv.append(np.nan)
        else:
            circle_3d = Circle3D([p0, p1, p2])
            axial_radius.append(circle_3d.radius)
            axial_center.append(circle_3d.center)
            axial_curv.append(1/circle_3d.radius)
    elif id == thinning_ids[-2] or thinning_ids[-1]:
        axial_radius.append(np.nan)
        axial_center.append(np.nan)
        axial_curv.append(np.nan)

axial_radius = np.array(axial_radius)
axial_curv = np.array(axial_curv)

# Cell 61
len(nearest_brain_surface_vertices[::thinning])

# Cell 62
len(n_kappa[::thinning])

# Cell 63
import numpy as np
from scipy.stats import pearsonr

# axial_curvのnanを無視して相関係数を計算
valid_indices = ~np.isnan(axial_curv)
thinning_n_kappa = n_kappa[::thinning]

correlation_axial_curvature, p_value_axial_curvature = pearsonr(thinning_n_kappa[valid_indices], axial_curv[valid_indices])

