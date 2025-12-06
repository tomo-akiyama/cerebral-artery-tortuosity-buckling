#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: 251129 sin+trend fitting データを1被験者ずつ抽出のコピー.ipynb

# Cell 1
import numpy as np
import pandas as pd

# Cell 2
# Select the case for preprocessing

case = "IXI013-HH-1212"


case_number = case[3:6].lstrip("0") ; case_number = int(case_number)

# Cell 3
# Excelファイルのパス
excel_path = ' 系統解剖研究室/10 Geometric analysis 220401-/Dataset/IXI HH/1 Data/HH_samples 抜粋.xlsx'
# データの読み込み
df = pd.read_excel(excel_path)

# 性別の値を文字列に変換
df.rename(columns={"SEX_ID (1=m, 2=f)": "SEX_ID"}, inplace=True)

# 人種の再分類を実行
def simplify_race(row):
    if row['ETHNIC_ID'] == 1:
        return 'White'
    elif row['ETHNIC_ID'] == 2:
        return 'Unknown'
    elif row['ETHNIC_ID'] == 3:
        return 'Asian'
    elif row['ETHNIC_ID'] == 4:
        return 'Black'
    elif row['ETHNIC_ID'] == 5:
        return 'Asian'
    elif row['ETHNIC_ID'] == 6:
        return 'Other'
df['race'] = df.apply(simplify_race, axis=1)
ethnic_counts = df['race'].value_counts().sort_index()

def simplify_sex(row):
    if row['SEX_ID'] == 1:
        return "Male"
    elif row['SEX_ID'] == 2:
        return "Female"

# 性別の再分類を実行
df["SEX"] = df.apply(simplify_sex, axis=1)
male_count = df[df["SEX"] == "Male"].shape[0]
female_count = df[df["SEX"] == "Female"].shape[0]

# 必要な列だけ残す
cols_keep = [
    'IXI_ID', 
    'SEX_ID', 'SEX', 'HEIGHT', 'WEIGHT',
    'ETHNIC_ID', 'race', 
    'AGE', 
    'BMI', 'BSA_DuBois',
    'CSF_absolute_volume', 'GM_absolute_volume', 'WM_absolute_volume',
    'GM_and_WM_absolute_volume', 'vol_abs_WMH', 'surf_TSA',
    'Average_thickness_FS', 'Average_thickness_std', 'TIV', 'TIV^(1/3)'
]

# make a working copy
df_ = df.copy()

# ensure all requested columns exist (create NaN if missing) and select in order
for c in cols_keep:
    if c not in df_.columns:
        df_[c] = np.nan

df_reduced = df_[cols_keep]

# Filter rows where case_number matches IXI_ID
matching_rows = df_reduced[df_reduced['IXI_ID'] == case_number]

matching_rows

# Cell 4
import ants
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pyvista as pv

import pandas as pd
import matplotlib.pyplot as plt

import sys
# 0 Scriptデイレクトリに移動する相対パス
sys.path.append(" 系統解剖研究室/10 Geometric analysis 220401-/Dataset/IXI HH/0 Script")

import vmtk_centerlines as vmtk_CL
import tool
import preprocessing
import mesh_generation

# Cell 5
lt_poly = pv.read(f' 系統解剖研究室/10 Geometric analysis 220401-/Dataset/IXI HH/{case}/Intermediate_data2/01_lt_pial_remesh.ply')
rt_poly = pv.read(f' 系統解剖研究室/10 Geometric analysis 220401-/Dataset/IXI HH/{case}/Intermediate_data2/01_rt_pial_remesh.ply')

# Cell 6
# データの基本設定
voxel_size = 0.5 # mm
s_factor = 1.2
num_iter = 100

# 共通項目
dataset = " 系統解剖研究室/10 Geometric analysis 220401-/Dataset/IXI HH"
# 画像フォルダ
import os
case_path = os.path.join(dataset, case)
# データを保存するフォルダ
entire_data_path = os.path.join(dataset, "4 Data 250821")
import os
case_path = os.path.join(dataset, case)
data_path = os.path.join(case_path, 'Data')
intermediate_data0_path = os.path.join(case_path, 'Intermediate_data0'); tool.create_dir(intermediate_data0_path)
intermediate_data1_path = os.path.join(case_path, 'Intermediate_data1'); tool.create_dir(intermediate_data1_path)
intermediate_data2_path = os.path.join(case_path, 'Intermediate_data2'); tool.create_dir(intermediate_data2_path)
intermediate_data3_path = os.path.join(case_path, 'Intermediate_data3'); tool.create_dir(intermediate_data3_path)
intermediate_data4_path = os.path.join(case_path, 'Intermediate_data4'); tool.create_dir(intermediate_data4_path)
intermediate_data5_path = os.path.join(case_path, 'Intermediate_data5'); tool.create_dir(intermediate_data5_path)

intermediate_data1_3ds_path = os.path.join(case_path, 'Intermediate_data1 3d slicer'); tool.create_dir(intermediate_data1_3ds_path)


mca_intermediate_data1_3ds_path = os.path.join(case_path, 'MCA Intermediate_data1 3d slicer'); tool.create_dir(mca_intermediate_data1_3ds_path)
mca_intermediate_data3_3ds_path = os.path.join(case_path, 'MCA Intermediate_data3 3d slicer'); tool.create_dir(mca_intermediate_data3_3ds_path)
mca_intermediate_data4_3ds_path = os.path.join(case_path, 'MCA Intermediate_data4 3d slicer'); tool.create_dir(mca_intermediate_data4_3ds_path)


# Load the data
vessel_vtk_mc_taubin_butterfly_path = os.path.join(intermediate_data1_path, "1_2_mca_mc_taubin_thick_butterfly16.ply")
vessel_vtk_mc_taubin_butterfly = mesh_generation.load_ply_to_polydata(vessel_vtk_mc_taubin_butterfly_path)

# geo_centerlines = mesh_generation.load_vtp(geo_centerlines_path)

merged_splitted_centerlines_path = os.path.join(mca_intermediate_data3_3ds_path, "3 merged_splitted_centerlines.vtp")
merged_splitted_centerlines = mesh_generation.load_vtp(merged_splitted_centerlines_path)

pl = pv.Plotter()
pl.camera.azimuth = 180
pl.camera.zoom(1.2)
pl.show_axes()
pl.show_bounds(grid='front', location='outer', all_edges=True, ticks='outside')

# Cell 7
celldata, pointids, points, pointdata = vmtk_CL.vmtk_convert2numpy(merged_splitted_centerlines)

# Cell 9
removal = 10

def _arclength(P_3darr):
    """点列 P (N,3) の弧長パラメータ s と各ステップ長 ds を返す"""
    ds = np.linalg.norm(np.diff(P_3darr, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s, ds

def _trime_ends(x, removal=10):
    return x[removal: -removal]

# Cell 11

n_faces = vessel_vtk_mc_taubin_butterfly.GetNumberOfPolys()

# Cell 12
def compute_edge_lengths(polydata):
    """
    vtkPolyData から全エッジ長を抽出して np.ndarray を返す関数
    長さの単位は polydata の座標系（＝今なら voxel 単位）のまま
    """
    mesh = pv.wrap(polydata)

    # 全エッジを抽出
    edges = mesh.extract_all_edges()

    # エッジの頂点インデックス（PyVista形式: [2, i, j, 2, k, l, ...]）
    lines = edges.lines.reshape(-1, 3)[:, 1:]  # shape = (n_edges, 2)

    pts = edges.points
    p0 = pts[lines[:, 0]]
    p1 = pts[lines[:, 1]]

    # 各エッジのユークリッド距離
    edge_lengths = np.linalg.norm(p1 - p0, axis=1)

    return edge_lengths

import vtk
def compute_mesh_volume(polydata):
    # まず全ポリゴンを三角形化
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.Update()

    # 体積計算
    mass = vtk.vtkMassProperties()
    mass.SetInputData(tri.GetOutput())
    mass.Update()

    volume = mass.GetVolume()
    area = mass.GetSurfaceArea()

    return  volume, area

# Cell 13
edge_lengths_vox = compute_edge_lengths(vessel_vtk_mc_taubin_butterfly)




# Cell 14
volume_vox, area_vox = compute_mesh_volume(vessel_vtk_mc_taubin_butterfly)



# Cell 15
index = [case_number]
data_vessel_meshdata = {
    "case": [case_number], 
    "n_segments": [len(pointids)],    
    "n_edges": [edge_lengths_vox.size],
    "n_faces": [n_faces],
    "mesh_volume_mm": volume_vox * (voxel_size ** 3),
    "surface_area_mm": area_vox * (voxel_size ** 2),
    "total_vessel_length_mm": [celldata['Length'].sum() * voxel_size]

}

# indexとcolumnsを設定してDataFrameを作成
df_vessel_meshdata = pd.DataFrame(data_vessel_meshdata)

# meshの統計量をcsvに保存
meshdata_filename = f"{case_number}_vessel_meshdata.csv"
meshdata_path = os.path.join("", meshdata_filename)
df_vessel_meshdata.to_csv(meshdata_path, index=None) # 圧縮無し

# DataFrameを表示
df_vessel_meshdata

# Cell 18
# ===== USER SETTINGS =====
segment_id = 5

# Cell 19
P_vox_3darr = points[pointids[segment_id]]

radius_vox_arr = pointdata['MaximumInscribedSphereRadius'][pointids[segment_id]]
radius_median_vox = np.median(radius_vox_arr)
radius_mean_vox = np.mean(radius_vox_arr)
radius_std_vox= np.std(radius_vox_arr, ddof=1)


s_vox_arr, ds_vox_arr = _arclength(P_vox_3darr)
ds_vox = np.median(ds_vox_arr)

L_vox = s_vox_arr[-1]

euclid_d_vox = np.linalg.norm(P_vox_3darr[0] - P_vox_3darr[-1])



TI = L_vox / euclid_d_vox
rho = TI - 1


P_vox_3darr_trimmed = _trime_ends(P_vox_3darr)
radius_vox_arr_trimmed = _trime_ends(radius_vox_arr)
s_vox_arr_trimmed, ds_vox_arr_trimmed = _arclength(P_vox_3darr_trimmed) # Rmks; s_vox_arr_trimmed = _trime_ends(s_vox_arr)とする場合は、弧長を0始まりに調整が必要。s_vox_arr_trimmed = s_vox_arr_trimmed - s_vox_arr_trimmed[0]
L_vox_trimmed = s_vox_arr_trimmed[-1]


# Cell 20
arc = []
for i in range(len(pointids)):
    def _arclength(P_3darr):
        """点列 P (N,3) の弧長パラメータ s と各ステップ長 ds を返す"""
        ds = np.linalg.norm(np.diff(P_3darr, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        return s, ds
    P_vox_3darr_ = points[pointids[i]]
    s_, ds_ = _arclength(P_vox_3darr_)
    l_ = s_[-1]
    arc.append(s_[-1] * voxel_size)

    radius_vox_arr_ = pointdata['MaximumInscribedSphereRadius'][pointids[i]]
    radius_mean_vox_ = np.median(radius_vox_arr_)


# Cell 22
import json
sources_on_mesh_path = os.path.join(intermediate_data5_path, "Inlets.mrk.json")
targets_on_mesh_path = os.path.join(intermediate_data5_path, "m1 outlets.mrk.json")

with open(sources_on_mesh_path, 'r') as f:
    sources_data = json.load(f)

with open(targets_on_mesh_path, 'r') as f:
    targets_data = json.load(f)

# 始点・終点について位置データを抽出してnumpy配列に変換 
sources_on_mesh = sources_data.get('markups', [])[0].get('controlPoints', [])
positions = [point['position'] for point in sources_on_mesh ]
sources_on_mesh = np.array(positions)

targets_on_mesh = targets_data.get('markups', [])[0].get('controlPoints', [])
positions = [point['position'] for point in targets_on_mesh ]
targets_on_mesh = np.array(positions)

# Cell 23
m1_mesh_path = os.path.join(intermediate_data5_path, "2_m1_mc_taubin_thick_butterfly16.ply")
m1_mesh = mesh_generation.load_ply_to_polydata(m1_mesh_path)

# Cell 24
# 着目したい中心点（P_vox_3darrの重心）
focus = np.asarray(P_vox_3darr).mean(axis=0) # 弧長に対するcentroidになる
# 弧長のほぼ真ん中の点
mid_pt = P_vox_3darr[len(P_vox_3darr)//2]

pl = pv.Plotter()
pl.subplot(0, 0)
pl.camera.azimuth = 180
pl.set_focus(focus) # 焦点を見たい位置に移す
pl.camera.zoom(3)
pl.show_bounds(grid='front', location='outer', all_edges=True, ticks='outside')

pl = pv.Plotter()
pl.camera.azimuth = 180
pl.set_focus(focus) # 焦点を見たい位置に移す
pl.camera.zoom(3)
pl.show_bounds(grid='front', location='outer', all_edges=True, ticks='outside')

# Cell 25
pl = pv.Plotter()
pl.camera.azimuth = 180
pl.set_focus(focus) # 焦点を見たい位置に移す
pl.camera.zoom(3)
pl.show_bounds(grid='front', location='outer', all_edges=True, ticks='outside')

# Cell 27
# ===== USER SETTINGS =====
# "M1"
# "M2-3"
# "M4"
mca_segment = "M2-3"

# Cell 28
# Validate mca_segment
if 'mca_segment' not in globals():
    raise NameError("mca_segment is not defined. Please set mca_segment to one of: 'M1', 'M2-3', 'M4'.")

_allowed_mca = {"M1", "M2-3", "M4"}
if mca_segment not in _allowed_mca:
    raise ValueError(f"Invalid mca_segment: {mca_segment!r}. Allowed values are: {sorted(_allowed_mca)}")

# Cell 29

# Cell 31
# 1D buckling modelの前提 (rod-like)
slenderness = L_vox / (2*radius_median_vox)

# Cell 32
kappa_vox_arr = pointdata['Curvature'][pointids[segment_id]]
tau_vox_arr = pointdata['Torsion'][pointids[segment_id]]

frenettangent_vox_arr = pointdata['FrenetTangent'][pointids[segment_id]]
frenetnormal_vox_arr = pointdata['FrenetNormal'][pointids[segment_id]]
frenetbinormal_vox_arr = pointdata['FrenetBinormal'][pointids[segment_id]]

fig, axes = plt.subplots(2, 1)
axes[0].plot(s_vox_arr * voxel_size, kappa_vox_arr / voxel_size, label="Curvature")
axes[0].set_ylabel("Curvature [/mm]")
axes[0].legend()
axes[0].grid(True)
axes[1].plot(s_vox_arr * voxel_size, tau_vox_arr / voxel_size, label="Torsion", color="tab:orange")
axes[1].set_xlabel("Arc length [mm]")
axes[1].set_ylabel("Torsion [/mm]")
axes[1].legend()
axes[1].grid(True)
fig.tight_layout()

pl = pv.Plotter()
pl.camera.azimuth = 180
pl.camera.zoom(1.2)

# Cell 34
# points: shape (N, 3)  ;  pointids: list-like (n_cells,) で、各セルの頂点インデックス列
xs = points[:, 0].astype(float)

# 各セルの代表 x（中央値）を計算
cell_median_x = []
for cell in pointids:
    xm = np.median(xs[np.asarray(cell, dtype=int)])
    cell_median_x.append(xm)
cell_median_x = np.asarray(cell_median_x)

# ---- 1次元 K-means (k=2)  ----
def kmeans_1d_two_clusters(x, n_iter=50):
    x = x.reshape(-1, 1)
    # 初期中心：分位点でざっくり
    c1, c2 = np.percentile(x, [33, 66]).reshape(2,1)
    for _ in range(n_iter):
        # 割当
        d1 = np.abs(x - c1)
        d2 = np.abs(x - c2)
        labels = (d2 < d1).astype(int)  # 0 or 1
        # 更新
        if np.any(labels == 0):
            c1_new = x[labels==0].mean()
        else:
            c1_new = c1
        if np.any(labels == 1):
            c2_new = x[labels==1].mean()
        else:
            c2_new = c2
        # 収束チェック
        if np.isclose(c1, c1_new).all() and np.isclose(c2, c2_new).all():
            break
        c1, c2 = np.array([[c1_new]]), np.array([[c2_new]])
    centers = np.array([c1.item(), c2.item()])
    labels = labels.ravel()
    return labels, centers

labels, centers = kmeans_1d_two_clusters(cell_median_x, n_iter=100)
# 大きい中心を「右」、小さい中心を「左」と定義
right_label = int(np.argmax(centers))
left_label  = 1 - right_label

# しきい値 x0（2中心の中点）
x0 = centers.mean()

# 結果：cellごとの左右フラグ
is_right_cell = (labels == right_label)
is_left_cell  = (labels == left_label)


# さらに「各点」を左右に分けたい場合は、推定した x0 を使って判定
is_right_point = xs >= x0
is_left_point  = xs <  x0

# Cell 35
# セルIDのリストを出力
right_cells = np.where(is_right_cell)[0]
left_cells  = np.where(is_left_cell)[0]


# 左右を画像で可視化
pl = pv.Plotter()
pl.camera.azimuth = 180
pl.camera.zoom(1.2)
pl.show_axes()
pl.show_bounds(grid='front', location='outer', all_edges=True, ticks='outside')

# Cell 36
if segment_id in right_cells:
    side = "Rt"
else:
    side = "Rt"

# Cell 38
# Sklearnを使えば、PCA分析は簡単
from sklearn.decomposition import PCA

# PCAを実行
pca = PCA(n_components=3) # PCAを行ったり、PCAの結果を格納したりするための変数を、pcaとして宣言
pca.fit(P_vox_3darr * voxel_size) # PCAを実行

# 固有値（＝各主成分の分散）: eigval
# 脳表を走行する脳血管での考え方
# PCA 固有値の意味
  # λ1：最も大きい分散 → 血管の“進行方向”
  # λ2, 3：その主方向に垂直な 2 次元平面内の 2 本の正規直交軸 -> 「より強い揺れ」＝ λ₂、「より弱い揺れ」＝ λ₃
# eigval1; 主方向のばらつき 本来なら一直線に伸びる方向が、脳表のアップダウン（法線方向の傾き）に従って蛇行し、主方向に沿った一貫性が低下する → 直線性が弱まり、λ1 の“圧倒的な優位性”が下がる
# eigval2; 面内方向 平坦面にある場合よりλ2が確実に増える
# eigval3; 面外方向＝法線方向のばらつき 血管が脳表の凸凹に従って “上下動（面外方向の揺れ）” すると、それがλ3 
eigvals = pca.explained_variance_
eigval1, eigval2, eigval3 = eigvals

# 固有ベクトル（＝主成分軸）: eigvecs
eigvecs = pca.components_
eigvec1, eigvec2, eigvec3 = eigvecs

# linearity/planarity/scattering: 3D点群解析（LiDAR, point cloud）では標準的な指標
# linearity = (λ₁−λ₂)/λ₁ → 第1主成分に対してどれだけ「線っぽいか」（λ₂が小さく、ほぼ1本の線状に伸びていると大きくなる）
# planarity = (λ₂−λ₃)/λ₁ → 第3軸が小さく、1次元ではなく「薄い2次元平面」に広がっていると大きくなる
# scattering = λ₃/λ₁ → 第3軸方向にもそこそこ広がっていれば大きくなる（＝3Dにバラけている）
# この3つの和は1

linearity  = (eigval1 - eigval2) / eigval1
planarity  = (eigval2 - eigval3) / eigval1
scattering = eigval2 / eigval1

# Classification of Boulders in Coastal Environments Using Random Forest Machine Learning on Topo-Bathymetric LiDAR Data　を参考に他の指標も求めておく
omni =  (eigval1 * eigval2 * eigval3) **(1/3)
aniso = (eigval1 - eigval3) / eigval1 
c_curvature = eigval3 / np.sum(eigvals)

# Cell 41
%matplotlib widget

# 主成分軸を可視化（平均点からベクトルを描く）
mean_P = P_vox_3darr.mean(axis=0)

ax = fig.add_subplot(111, projection='3d')
ax.scatter(P_vox_3darr[:,0], P_vox_3darr[:,1], P_vox_3darr[:,2], alpha=0.4, label="points")

scale = 3 * np.sqrt(eigvals)
colors = ['r', 'g', 'b']
labels = ['PC1', 'PC2', 'PC3']
for i in range(3):
    vec = eigvecs[i] * scale[i]
    ax.plot([mean_P[0], mean_P[0]+vec[0]],
            [mean_P[1], mean_P[1]+vec[1]],
            [mean_P[2], mean_P[2]+vec[2]],
            color=colors[i], label=labels[i], linewidth=3)

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()

# Cell 44
# exclusion
if L_vox * voxel_size < 10:
    class_periodicity = 'Too short'
    undetectable_reason = None
    
    data_vessel_segmentdata = {
    "case": [case_number], 
    "segment_id": [segment_id],
    "mca_segment": [mca_segment],
    "side": [side],
    "pts": [len(P_vox_3darr)],
    "ds_mm": [ds_vox * voxel_size],
    "L_full_mm": [s_vox_arr[-1] * voxel_size],
    "L_trimmed_mm": [np.nan],
    "radius_median_mm": [radius_median_vox * voxel_size],
    "radius_mean_mm": [radius_mean_vox * voxel_size],
    "radius_std_mm": [radius_std_vox * voxel_size],
    "euclid_distance_mm": [euclid_d_vox * voxel_size],
    "TI": [TI],
    "rho": [rho],

    # κ 系 → NaN
    "kappa_median_mm": [np.nan],
    "kappa_mean_mm":   [np.nan],
    "kappa_std_mm":    [np.nan],
    "kappa_rms_mm":    [np.nan],
    "kappa_p95_mm":    [np.nan],
    "kappa_max_mm":    [np.nan],
    "kappa_min_mm":    [np.nan],

    "slenderness": [np.nan],

    # 3D_PCAはrawなので残してよい
    "eigval1_3d_mm": [eigval1],
    "eigval2_3d_mm": [eigval2],
    "eigval3_3d_mm": [eigval3],
    "linearity":     [linearity],
    "planarity":     [planarity],
    "scattering":    [scattering],
    "omnivariance":  [omni],
    "anisotropy":    [aniso],
    "change_curvature": [c_curvature],

    # prior系も残さない
    "alpha_prior": [np.nan],
    "lambda_prior_mm": [np.nan],
    "pts_trimmed": [np.nan],
    "class_periodicity": [class_periodicity],    
    "undetectable_reason": [undetectable_reason], 

    # ---- sin+trend / 波形系すべて NaN ----
    "k_hat_mm": [np.nan],
    "lambda_hat_mm": [np.nan],
    "amp_hat_mm": [np.nan],
    "amp_hat_U_mm": [np.nan],
    "amp_hat_V_mm": [np.nan],

    "coef_uU_cos_mm": [np.nan], 
    "coef_uU_sin_mm": [np.nan], 
    "coef_uV_cos_mm": [np.nan], 
    "coef_uV_sin_mm": [np.nan],

    "eigval1_2d_mm": [np.nan],
    "eigval2_2d_mm": [np.nan],
    "directionality": [np.nan],
    "minor_major_amp_ratio": [np.nan],
    "amp_hat_over_L": [np.nan],
    "lambda_hat_over_radius": [np.nan],
    "amp_hat_over_radius": [np.nan],

    "R2": [np.nan],
    "partial_R2": [np.nan],
    "SSE_full": [np.nan],
    "SSE_tr": [np.nan],
    "hr_amp": [np.nan],
    "hr_energy": [np.nan],
    "alpha_hat": [np.nan],
    "coverage_hat": [np.nan],
    "SNR": [np.nan],
    "rms_residual_full_mm": [np.nan],
    "rms_residual_tr_mm": [np.nan],

    "boundary_hit_low": [np.nan],
    "boundary_hit_high": [np.nan],
    }
    # indexとcolumnsを設定してDataFrameを作成
    df_vessel_segmentdata = pd.DataFrame(data_vessel_segmentdata)

    # meshの統計量をcsvに保存
    segmentdata_filename = f"{case_number}_vessel_segmentdata_{segment_id}.csv"
    segment_path = os.path.join("", segmentdata_filename)
    df_vessel_segmentdata.to_csv(segment_path, index=None) # 圧縮無し

    tool.notify(f"{case}, segment ID {segment_id}")
    sys.exit()

else:
    pass

# Cell 45
# 下のplotをみてわかるようにkappaは端点が不安定になるので、端で暴れているκは「端点効果」として切り捨てる
kappa_arr_trimmed = _trime_ends(kappa_vox_arr)


kappa_median_vox = np.median(kappa_arr_trimmed)
kappa_mean_vox = np.mean(kappa_arr_trimmed) # もともと0以上の値しかとらないので、Average (Absolute) Curvature = Mean Curvature
kappa_std_vox = np.std(kappa_arr_trimmed, ddof=1)


kappa_q1_vox = np.percentile(kappa_arr_trimmed, 25)
kappa_q3_vox = np.percentile(kappa_arr_trimmed, 75)
kappa_iqr_vox = kappa_q3_vox - kappa_q1_vox

kappa_rms_vox = np.sqrt(np.mean(kappa_arr_trimmed**2))
kappa_max_vox = np.max(kappa_arr_trimmed)
kappa_min_vox = np.min(kappa_arr_trimmed)
kappa_p95_vox  = np.percentile(kappa_arr_trimmed, 95)


# Cell 47
# 共通ユーティリティ（正規化・接線）
def _normalize(vec):
    return vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12) # + 1e-12 はゼロ除算対策


# 以下のユーティリティは不要かも

# サンプル間隔が等しい」前提
def _tangent(P_3darr):
    """中心差分で接線（単位ベクトル）""" # 3点中心差分（2次精度）
    T = np.zeros_like(P_3darr)
    T[1:-1] = P_3darr[2:] - P_3darr[:-2] # 3点差分（中心差分）
    T[0]    = P_3darr[1] - P_3darr[0] # 最初の点は前方差分
    T[-1]   = P_3darr[-1] - P_3darr[-2] # 最後の点は後方差分
    T = _normalize(T)
    return T

# Helpers: 単位変換
def mm_to_vox(x_mm, voxel_size_mm=voxel_size):
    return np.asarray(x_mm, dtype=float) / float(voxel_size_mm)

def vox_to_mm(x_vox, voxel_size_mm=voxel_size):
    return np.asarray(x_vox, dtype=float) * float(voxel_size_mm)

# Cell 53
# alphaの決め方を変更
# 既報からM1の形は基本的にC型が増える! (S型という報告はなかった)
alpha_prior = 25.27038792824841

# lambda = alpha_prior * r
lambda_prior_vox = alpha_prior * radius_median_vox

# ------------- α_prior と ±50% ガード ------------- => を考えていたが、λ_priorにおいてguardをあとで導入する!

# Cell 55
# n次の全体多項式
 # 目的1: Baseline centerlineの作成
 # 目的2: 信号y(s)の低次のトレンド(平均・一次・必要に応じ二次)を取り除く。短区間で DCT/FFT をかける前の漏れ・リープ抑制が目的。

def poly_trend(Y_arr,
               s_arr,
               degree, 
               return_residual=True):
    """
    多項式トレンドを推定して返す func。
    - Y: shape (N,) でも (N,D) でもOK（1D: u_U/u_V、3D: centerline）
    - s: shape (N,) 弧長（等間隔でなくてもよい）※内部で 0..1 に正規化
    - degree: 0,1,2,... （0:平均, 1:一次, 2:二次...）
    戻り値:
      trend:  Y と同じ shape（トレンド成分）
      info:   {"coef": 係数配列, "degree": degree, "s_minmax": (s0,s1)}
              ※return_residual=True のときは info["residual"] も含める
    """
    Y = np.asarray(Y_arr, float)
    s = np.asarray(s_arr, float)

    # 1D(N,) / 2D(N,D) 判定
    is_1d = (Y.ndim == 1)
    if is_1d:
        Y = Y[:, None]  # (N,) → (N,1)

    # -------- degree = 0: 平均値だけの特別処理 --------
    if degree == 0:
        mean_val = np.mean(Y, axis=0)          # shape (1,) or (D,)
        trend = np.tile(mean_val, (len(Y), 1)) # shape (N,D)

        info = {
            "coef": mean_val,
            "degree": 0,
            "s_minmax": (float(s[0]), float(s[-1]))
        }

        if return_residual:
            resid = Y - trend                  # shape (N,D)
            info["residual"] = resid[:, 0] if is_1d else resid

        trend_out = trend[:, 0] if is_1d else trend
        return trend_out, info

    # -------- degree >= 1: 通常の polyfit 処理 --------

    # sを0..1に正規化
    s0 = (s - s[0]) / max(s[-1] - s[0], 1e-12)

    # 多項式フィット（Yが(N,1)でも(N,D)でもOK）
    coef = np.polyfit(s0, Y, deg=degree)  # shape: (degree+1,) or (degree+1,D)

    # トレンド計算
    if is_1d:
        # 1Dのときは(N,)で受けてから(N,1)にしてYと合わせる
        trend_1d = np.polyval(coef, s0)       # (N,)
        trend = trend_1d[:, None]             # (N,1)
    else:
        # D次元（例: centerline (N,3)）
        trend = np.polyval(coef, s0[:, None]) # (N,D)

    info = {
        "coef": coef,             # 注意：高次数→低次数の並び（polyfit仕様）
        "degree": degree,
        "s_minmax": (float(s[0]), float(s[-1]))
    }

    if return_residual:
        resid = Y - trend
        info["residual"] = resid[:, 0] if is_1d else resid

    trend_out = trend[:, 0] if is_1d else trend
    return trend_out, info

# Cell 56
P0_vox_3darr, info_poly_trend = poly_trend(Y_arr=P_vox_3darr,
               s_arr=s_vox_arr,
               degree = 2, 
               return_residual = True)

# Cell 57
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P_vox_3darr[:,0], P_vox_3darr[:,1], P_vox_3darr[:,2], alpha=0.4, label="points")
ax.scatter(P0_vox_3darr[:,0], P0_vox_3darr[:,1], P0_vox_3darr[:,2], alpha=0.4, label="p0")

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()

# Cell 58
def bishop_frame(T_3darr):
    T = np.asarray(T_3darr, float)
    N = len(T)
    U = np.zeros_like(T)
    seed = np.array([1.0, 0.0, 0.0])
    # 接線T[0]がx方向とほぼ平行（内積>0.9）なら、y方向を使う（直交性を保つため）
    if np.abs(np.dot(seed, T[0])) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u0 = seed - np.dot(seed, T[0]) * T[0] # seedを接線Tに垂直投影したベクトル(Tに直交し、U・V成分にはまだ分解されていない）をseedから引く
    U[0] = u0 / (np.linalg.norm(u0) + 1e-12) # u0の時点では「Tに直交する平面（U–V平面）」上のどこかの方向ベクトル -> そのu0の方向を“U軸”と定義
    for i in range(1, N):
        # 前のUのうち、新しい接線T[i]の方向を向いている成分だけを取り除き、残った法平面内の成分を新しいU[i]とする(=これはずれたTの分だけuはずれるだけでそれ以外は変えないということ[最小回転, parallel transport])
        u = U[i - 1] - np.dot(U[i - 1], T[i]) * T[i] # U[i−1]を新しい接線T[i]に新しく垂直投影したベクトル(Tに直交し、U・V成分にはまだ分解されていない）(「前の点でのU[i−1]を、新しい接線ベクトルT[i に直交する平面（＝新しい U–V 平面）に再射影する」=> U は急に回転したりねじれたりせず、最小限の角度で更新される)
        U[i] = u / (np.linalg.norm(u) + 1e-12)
    V = np.cross(T, U) # 残りのVは自動的に外積で直交的に決まる
    V = _normalize(V)
    # UとVのベクトルの向きをそろえ、Uベクトルの符号を反転するときにVベクトルも同時に反転(ベクトル三つ（T, U, V）が右手の向きの関係（直交座標系の正しい回転方向）を保つようにする)
    for i in range(1, N):
        if np.dot(U[i], U[i - 1]) < 0:
            U[i] = -U[i]
            V[i] = -V[i]   # U と V を同時に反転
    return T, U, V

# Cell 59
T_vox_3darr, U_vox_3darr, V_vox_3darr = bishop_frame(frenettangent_vox_arr)

# Cell 60
T, U, V = bishop_frame(P_vox_3darr)

# 観測された centerline：P_vox_3darr
# 大域トレンド（多項式）：P0_vox_3darr
# => Δ = P_vox_3darr − P0_vox_3darr（observed − trend） で統一
Delta_vox_3darr = P_vox_3darr - P0_vox_3darr

pl = pv.Plotter()
pl.camera.azimuth = 180
pl.camera.zoom(1.2)

# Cell 61
# 血管中心線Pと平滑基準線P_0の差\DeltaをBishop frame(T, U, V)に射影し、横方向の変位成分u_U, u_Vを取り出す
# 中心線に対して「横方向（sideways）」の変位成分: lateral_components
def lateral_components(P_3darr,
                       P0_3darr,
                       T_3darr, U_3darr, V_3darr):
    """
    Args:
        P_vox_3darr: 元の血管中心線の座標（N, 3）
        P0_vox_3darr: Savitzky-Golay平滑で得た基準線（N, 3）
        T_vox_3darr, U_vox_3darr, V_vox_3darr: Bishop frame（三直交単位ベクトル、各点に対応）

    Returns:
        uU: (N,) 平面内横方向変位量（符号付き） ※buckling（座屈波）の主成分として解析(しかしこの時点ではsurfaceとの位置合わせされていない)
        uV: (N,) 面外方向変位量（符号付き） ※最終的にはねじれや脳表起伏の影響を反映（地形的影響）
        Delta: (N, 3) 差ベクトル
        Delta_parT: (N, 3) 接線方向変位、つまり、軸方向
        Delta_perpT: (N, 3) 横方向変位（Tに垂直）形状変化の主要成分
        DeltaU: (N, 3) U方向変位ベクトル ※φ角解析・可視化用
        DeltaV: (N, 3) V方向変位ベクトル ※φ角解析・可視化用
    """
    # 差ベクトルDelta(s)
    Delta_3darr = P_3darr - P0_3darr
    # DeltaのT方向成分
    Delta_parT_3darr = np.einsum('ij,ij->i', Delta_3darr, T_3darr)[:, np.newaxis] * T_3darr # 内積の公式通り(np.einsum('ij,ij->i', Delta_3darr, T) (N,)-> [:, np.newaxis](N,1) (各点を縦ベクトル化（列ベクトル))
    # DeltaのT方向と直行成分(U-V平面成分)
    Delta_perpT_3darr = Delta_3darr - Delta_parT_3darr

    # スカラー射影
    # u_U：面内の横方向変位の大きさ（符号付き、主なbuckling成分）
    uU_arr = np.einsum('ij,ij->i', Delta_perpT_3darr, U_3darr)
    # u_V：面外方向変位の大きさ（符号付き、ねじれや脳表起伏の寄与[ただしn_Sとこの時点では合致しているわけではない])
    uV_arr = np.einsum('ij,ij->i', Delta_perpT_3darr, V_3darr)

    # 各方向ベクトル成分
    DeltaU_3darr = uU_arr[:, None] * U_3darr
    DeltaV_3darr = uV_arr[:, None] * V_3darr
    return uU_arr, uV_arr, Delta_3darr, Delta_parT_3darr, Delta_perpT_3darr, DeltaU_3darr, DeltaV_3darr

# Cell 62
uU_vox_arr, uV_vox_arr, Delta_vox_3darr, Delta_parT_vox_3darr, Delta_perpT_vox_3darr, DeltaU_vox_3darr, DeltaV_vox_3darr = lateral_components(P_vox_3darr, P0_vox_3darr,
                                                                            T_vox_3darr, U_vox_3darr, V_vox_3darr )

# Cell 63
pl = pv.Plotter()

# U方向変位ベクトル
# V方向変位ベクトル

pl.camera.azimuth = 180
pl.camera.zoom(1.2)

# Cell 65
# φ(s) の算出
phi = np.arctan2(uV_vox_arr, uU_vox_arr)

# φ分散（平面性の指標）
phi_var = np.var(phi[removal:-removal])  

# arctan2 は角度を −π〜+π の範囲に「折りたたむ」関数 
# =>つまり、実際に φ(s) が 190°, 200°… と滑らかに回転しているのに、+π（180°）を超えた瞬間に −π に戻ってしまう という性質がある 

phi_unwrap = np.unwrap(phi)

# φ分散（平面性の指標）--- 連続位相で評価
phi_var = np.var(phi_unwrap[removal:-removal])

# =>「地形追従（surface following）」と「torsional buckling（ねじれ座屈）」
# ゆっくりした“傾き（ドリフト）”と混同：表面追従で平面がゆっくり回ると、分散は増えるが ねじれ（torsion）とは限らない

# 可視化

# =>Bishop フレームの U–V 平面は滑らかに回転しており、急激なトーション（ねじれ）はない。

# Cell 70
from scipy.fft import dct

def dct_power_sum(uU_arr, uV_arr, detrend_deg=2):
    """
    DCT-II (orthonormal) を uU,uV に別々に適用し、パワー和 S[m] を返す。
    戻り: m（DCTインデックス）, S（パワー和）, U_dct, V_dct, メタ（トレンド係数など）
    """
    uU_arr = np.asarray(uU_arr, float) # 不安定な端点近くは除いておく(_trime_endsを使ったuU,uVを渡すこと)
    uV_arr = np.asarray(uV_arr, float)
    N  = len(uU_arr)

    # デトレンド（平均・一次/二次など）※短区間では一次で十分
    uU_detr, metaU = poly_trend(uU_arr, np.arange(N), degree=detrend_deg) # detr = detrended の略
    uV_detr, metaV = poly_trend(uV_arr, np.arange(N), degree=detrend_deg)

    # DCT-II（実数・端点滑らか／漏れ低減）※窓は通常不要
    Uc_arr = dct(uU_detr, type=2, norm='ortho')
    Vc_arr = dct(uV_detr, type=2, norm='ortho')

    S_arr = Uc_arr**2 + Vc_arr**2 # パワー和 S(k)=|U(k)|^2+|V(k)|^2 
    m_arr = np.arange(N)
    return m_arr, S_arr, Uc_arr, Vc_arr, N, dict(metaU=metaU, metaV=metaV)

# Cell 71
def dct_index_to_k(m_arr, N, ds):
    """DCT-II のインデックスmを波数k[rad/length] に換算（目安） ※目安と書いた理由は以下参照
    m_arr: DCTののインデックス番号（整数）
    k_arr: 実空間に換算した波数

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html 
    DCT-IIの余弦の中身は$\frac{\pi m(2n+1)}{2N}$ ※kの記号は波数と混乱するので、m(出力（周波数）側のモード番号（0,1,…,N−1 の整数）)を使う
    実際の空間座標は s_n = n * ds （n番目の点の位置, nは自然数）なので$n = \frac{s_n}{ds}$
    これを上の式に代入すると\frac{\pi m}{N} * n = \frac{\pi m}{N} * \frac{s_n}{ds} = \frac{\pi m}{N ds} * s_n.
    これは実空間座標sに対する角周波数（波数）の係数(rad/vox, 1voxあたり何ラジアン進むか)

    DCT は理論的には“偶対称（端点で勾配ゼロ）”な信号を仮定している。
    実データ（あなたの血管中心線など）はその仮定と少し違う（端点で傾きゼロではない）。
    そのため、DCT の数式から導かれる波数 k_m = \pi*m / N*ds は、理想条件下の理論値であり、実際のデータの“真の空間波数”とはズレる可能性がある。
    しかし、主波長などを求める初期推定としては十分使える。
    """
    return np.pi * m_arr / (N * ds)

def pick_peak_in_band(m_arr, S_arr, k_arr, lambda_prior=None, mult_range=(0.5, 2.0), exclude_dc=True):
    """
    バンド制限（k_band=(kmin,kmax)）内で S のピークを取る。
    バンド未指定なら m=1..N-1 の最大。

    波数kと波長\lambdaの関係はk=\frac{2 \pi}{\lambda} 
    =>k_band には、理論波数 k_{prior} の+-2倍（<=\lambda_priorの1/2〜2倍）に相当する範囲を与える
    # \lambda_priorについて1/2~2*\lambdaの範囲にあるであろうという設定
    ※\lambda_priorの1/2倍 => もしS型があった場合まで含むことができるように
    ※\lambda_priorの2倍 => M1がC型と仮定したとしても、実は不完全なC型でlambdaがもっと長い場合に備える

    Arg: 
     m_arr: DCTののインデックス番号（整数）
     S_arr: パワー和 S(k)=|U(k)|^2+|V(k)|^2 
     k_arr: 各インデックスに対応する波数 [rad/length]
     lambda_prior : float, optional
        理論波長 λ_prior = alpha * r
     mult_range : tuple(float,float), default (0.5, 2.0) => 探索範囲 [a,b]*λ_prior。
     exclude_dc : bool, default True 直流 (m=0) を除外するか否か。
    """
    

    # λ_prior が指定されていない場合 → 全域で探す
    if lambda_prior is None:
        sel = (m_arr >= 1) if exclude_dc else np.ones_like(m_arr, dtype=bool) # exclude_dc=Trueの場合、直流除外。Falseの場合、全データを使う（直流含む）
    else:
        # λを基準に ±2倍の範囲を設定
        lam_min = mult_range[0] * lambda_prior
        lam_max = mult_range[1] * lambda_prior
        # λが大きいほどkは小さいので、逆順に変換
        k_min = 2 * np.pi / lam_max
        k_max = 2 * np.pi / lam_min
        sel = (k_arr >= k_min) & (k_arr <= k_max)
        if exclude_dc:
            sel = sel & (m_arr >= 1) # 実際にはk_max>0なのでm=0 は自動的に除外されるのでいらないかも

    if not np.any(sel):
        return None, dict(found=False, reason="no_selection")

    rel_idx = int(np.argmax(S_arr[sel]))
    idx = np.where(sel)[0][rel_idx]
    m_star = int(m_arr[idx])
    k_star = float(k_arr[idx])
    lam_star = float(2*np.pi / max(k_star, 1e-12))
    return m_star, dict(found=True, m=m_star, k=k_star, lambda_est=lam_star, idx=idx)

# Cell 72
uU_vox_arr_trimmed = _trime_ends(uU_vox_arr)
uV_vox_arr_trimmed = _trime_ends(uV_vox_arr)



# Cell 73
# DCT
detrend_deg = 2
m_arr, S_arr, Uc_arr, Vc_arr, N, meta_dct = dct_power_sum(uU_arr=uU_vox_arr_trimmed, uV_arr=uV_vox_arr_trimmed, detrend_deg=detrend_deg)
k_arr = dct_index_to_k(m_arr, N, ds=ds_vox)

# Cell 74
# ピーク
m_star, meta_peak = pick_peak_in_band(m_arr, S_arr, k_arr, lambda_prior=lambda_prior_vox)

# Cell 75
if meta_peak['found'] == False:
    class_periodicity = 'Long-wave / Straight-ish / undetectable'
    undetectable_reason = 'no DCT mode in prior band'

    data_vessel_segmentdata = {
    "case": [case_number], 
    "segment_id": [segment_id],
    "mca_segment": [mca_segment],
    "side": [side],
    "pts": [len(P_vox_3darr)],
    "ds_mm": [ds_vox * voxel_size],
    "L_full_mm": [s_vox_arr[-1] * voxel_size],
    "L_trimmed_mm": [np.nan],
    "radius_median_mm": [radius_median_vox * voxel_size],
    "radius_mean_mm": [radius_mean_vox * voxel_size],
    "radius_std_mm": [radius_std_vox * voxel_size],
    "euclid_distance_mm": [euclid_d_vox * voxel_size],
    "TI": [TI],
    "rho": [rho],

    # κ 系 → NaN
    "kappa_median_mm": [np.nan],
    "kappa_mean_mm":   [np.nan],
    "kappa_std_mm":    [np.nan],
    "kappa_rms_mm":    [np.nan],
    "kappa_p95_mm":    [np.nan],
    "kappa_max_mm":    [np.nan],
    "kappa_min_mm":    [np.nan],

    "slenderness": [np.nan],

    # 3D_PCAはrawなので残してよい
    "eigval1_3d_mm": [eigval1],
    "eigval2_3d_mm": [eigval2],
    "eigval3_3d_mm": [eigval3],
    "linearity":     [linearity],
    "planarity":     [planarity],
    "scattering":    [scattering],
    "omnivariance":  [omni],
    "anisotropy":    [aniso],
    "change_curvature": [c_curvature],

    # prior系も残さない
    "alpha_prior": [np.nan],
    "lambda_prior_mm": [np.nan],
    "pts_trimmed": [np.nan],
    "class_periodicity": [class_periodicity],   
    "undetectable_reason": [undetectable_reason], 

    # ---- sin+trend / 波形系すべて NaN ----
    "k_hat_mm": [np.nan],
    "lambda_hat_mm": [np.nan],
    "amp_hat_mm": [np.nan],
    "amp_hat_U_mm": [np.nan],
    "amp_hat_V_mm": [np.nan],

    "coef_uU_cos_mm": [np.nan], 
    "coef_uU_sin_mm": [np.nan], 
    "coef_uV_cos_mm": [np.nan], 
    "coef_uV_sin_mm": [np.nan],

    "eigval1_2d_mm": [np.nan],
    "eigval2_2d_mm": [np.nan],
    "directionality": [np.nan],
    "minor_major_amp_ratio": [np.nan],
    "amp_hat_over_L": [np.nan],
    "lambda_hat_over_radius": [np.nan],
    "amp_hat_over_radius": [np.nan],

    "R2": [np.nan],
    "partial_R2": [np.nan],
    "SSE_full": [np.nan],
    "SSE_tr": [np.nan],
    "hr_amp": [np.nan],
    "hr_energy": [np.nan],
    "alpha_hat": [np.nan],
    "coverage_hat": [np.nan],
    "SNR": [np.nan],
    "rms_residual_full_mm": [np.nan],
    "rms_residual_tr_mm": [np.nan],

    "boundary_hit_low": [np.nan],
    "boundary_hit_high": [np.nan],
    }
    # indexとcolumnsを設定してDataFrameを作成
    df_vessel_segmentdata = pd.DataFrame(data_vessel_segmentdata)

    # meshの統計量をcsvに保存
    segmentdata_filename = f"{case_number}_vessel_segmentdata_{segment_id}.csv"
    segment_path = os.path.join("", segmentdata_filename)
    df_vessel_segmentdata.to_csv(segment_path, index=None) # 圧縮無し

    tool.notify(f"{case}, segment ID {segment_id}")
    sys.exit()

else:
    pass


# Cell 76
from matplotlib.ticker import FixedLocator, FixedFormatter

# 作図
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
# U, V, Sum（m=0:DC は見栄えのため除く）
m1 = m_arr[1:]
k1 = k_arr[1:]
ax.plot(k1, Uc_arr[1:]**2, label='|U|^2 (DCT power)', marker='o')
ax.plot(k1, Vc_arr[1:]**2, label='|V|^2 (DCT power)', marker='o')
ax.plot(k1, S_arr[1:],      label='S = |U|^2 + |V|^2', linewidth=2, marker='o')
# 事前帯域の陰影
if lambda_prior_vox:
    mult_range=(0.5, 2.0)
    lam_min = mult_range[0] * lambda_prior_vox
    lam_max = mult_range[1] * lambda_prior_vox
    k_band = (2*np.pi/lam_max, 2*np.pi/lam_min)  # 小→大 
    ax.axvspan(k_band[0], k_band[1], color='grey', alpha=0.15, label='prior band')

# ピークに印
if meta_peak.get('found', False):
    ax.axvline(meta_peak['k'], color='k', linestyle='--', alpha=0.6)
    ax.annotate(f"peak λ≈{meta_peak['lambda_est']*voxel_size:.1f} mm",
                xy=(meta_peak['k'], S_arr[meta_peak['idx']]),
                xytext=(5, 10), textcoords='offset points',
                fontsize=9,
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='k', alpha=0.5)  # 矢印スタイルarrowprops=dict(arrowstyle='-|>', lw=0.8, color='r', alpha=0.5)  # 矢印スタイル
    )
ax.set_xlabel('wavenumber k [rad/voxel]')
ax.set_ylabel('power')
ax.legend(loc='upper right')

k_prior_vox = 2*np.pi / lambda_prior_vox # [rad/voxel]
ax.axvline(k_prior_vox, color='r', linestyle='--', alpha=0.6)
ax.annotate(
    f"k_prior={k_prior_vox:.3f} rad/vox\n"
    f"prior λ≈{lambda_prior_vox*voxel_size:.1f} mm",   # 表示する文字
    xy=(k_prior_vox, ax.get_ylim()[1]*0.9),                 # 矢印の基点（x=k_prior, y=縦軸の90%付近）
    xytext=(5, 10),                                     # 文字のオフセット [pixel単位]
    textcoords='offset points',
    color='r',
    fontsize=9,
    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='r', alpha=0.5)  # 矢印スタイル
)



# 表示に上限をつけてみる。人の目で見える範囲として5mmを基準にしてみる(脳血管の径が1–3mm程度とすると、波長5mmの曲がりは「血管径の1.5〜5倍」)
lambda_min_mm = 5     # 5 mm → 高周波（細かい）
k_max = 2*np.pi*voxel_size / lambda_min_mm
k_max = max(k_max, k_band[1]+0.5) # 5mmよりも小さい範囲までk_bandがカバーする場合に備える
k_min = 1e-2  # 表示範囲（0 から始めないため）
ax.set_xlim(k_min, k_max)

# 上部に λ[mm] の目盛り（補助目盛）
ax2 = ax.secondary_xaxis('top', functions=(
    lambda kk: 2*np.pi/np.maximum(kk, 1e-12)*voxel_size,   # k -> lambda_mm
    lambda ll: 2*np.pi/np.maximum(ll/voxel_size, 1e-12)    # lambda_mm -> k
))
ax2.set_xlabel('wavelength λ [mm]')

# λ目盛りを自分で指定
k_ticks = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
lambda_ticks = 2*np.pi*voxel_size / k_ticks  # 対応するλ[mm]
# Segment lengthの情報も入れておく
k_L = 2*np.pi / L_vox # [rad/voxel]
ax.axvline(k_L, color='purple', linestyle=':', alpha=0.7)
ax.annotate(
    f"segment length = {L_vox * voxel_size:.1f} mm\nk_L = {k_L:.3f} rad/vox",
    xy=(k_L, ax.get_ylim()[1] * 0.5),
    xytext=(5, 10),
    textcoords='offset points',
    color='purple',
    fontsize=9,
    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='purple', alpha=0.5)
)

# set_ticks / set_ticklabels は xaxis オブジェクトに対して行う
ax2.xaxis.set_major_locator(FixedLocator(lambda_ticks))
ax2.xaxis.set_major_formatter(FixedFormatter([f"{lt:.1f}" for lt in lambda_ticks]))

ax.set_title(f"segment ID. {segment_id}")

# Cell 77
from matplotlib.ticker import FixedLocator, FixedFormatter

# 作図
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
# U, V, Sum（m=0:DC は見栄えのため除く）
m1 = m_arr[1:]
k1 = k_arr[1:]
ax.plot(k1, Uc_arr[1:]**2, label='|U|^2 (DCT power)', marker='o')
ax.plot(k1, Vc_arr[1:]**2, label='|V|^2 (DCT power)', marker='o')
ax.plot(k1, S_arr[1:],      label='S = |U|^2 + |V|^2', linewidth=2, marker='o')
# 事前帯域の陰影
if lambda_prior_vox:
    mult_range=(0.5, 2.0)
    lam_min = mult_range[0] * lambda_prior_vox
    lam_max = mult_range[1] * lambda_prior_vox
    k_band = (2*np.pi/lam_max, 2*np.pi/lam_min)  # 小→大 
    ax.axvspan(k_band[0], k_band[1], color='grey', alpha=0.15, label='prior band')

# ピークに印
if meta_peak.get('found', False):
    ax.axvline(meta_peak['k'], color='k', linestyle='--', alpha=0.6)
    ax.annotate(f"peak λ≈{meta_peak['lambda_est']*voxel_size:.1f} mm",
                xy=(meta_peak['k'], S_arr[meta_peak['idx']]),
                xytext=(5, 10), textcoords='offset points',
                fontsize=9,
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='k', alpha=0.5)  # 矢印スタイルarrowprops=dict(arrowstyle='-|>', lw=0.8, color='r', alpha=0.5)  # 矢印スタイル
    )
ax.set_xlabel('wavenumber k [rad/voxel]')
ax.set_ylabel('power')
ax.legend(loc='upper right')

k_prior_vox = 2*np.pi / lambda_prior_vox # [rad/voxel]
ax.axvline(k_prior_vox, color='r', linestyle='--', alpha=0.6)
ax.annotate(
    f"k_prior={k_prior_vox:.3f} rad/vox\n"
    f"prior λ≈{lambda_prior_vox*voxel_size:.1f} mm",   # 表示する文字
    xy=(k_prior_vox, ax.get_ylim()[1]*0.9),                 # 矢印の基点（x=k_prior, y=縦軸の90%付近）
    xytext=(5, 10),                                     # 文字のオフセット [pixel単位]
    textcoords='offset points',
    color='r',
    fontsize=9,
    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='r', alpha=0.5)  # 矢印スタイル
)



# 表示に上限をつけてみる。人の目で見える範囲として5mmを基準にしてみる(脳血管の径が1–3mm程度とすると、波長5mmの曲がりは「血管径の1.5〜5倍」)
lambda_min_mm = 5     # 5 mm → 高周波（細かい）
k_max = 2*np.pi*voxel_size / lambda_min_mm
k_max = max(k_max, k_band[1]+0.5) # 5mmよりも小さい範囲までk_bandがカバーする場合に備える
k_min = 1e-2  # 表示範囲（0 から始めないため）
ax.set_xlim(k_min, k_max)

# 上部に λ[mm] の目盛り（補助目盛）
ax2 = ax.secondary_xaxis('top', functions=(
    lambda kk: 2*np.pi/np.maximum(kk, 1e-12)*voxel_size,   # k -> lambda_mm
    lambda ll: 2*np.pi/np.maximum(ll/voxel_size, 1e-12)    # lambda_mm -> k
))
ax2.set_xlabel('wavelength λ [mm]')

# λ目盛りを自分で指定
k_ticks = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
lambda_ticks = 2*np.pi*voxel_size / k_ticks  # 対応するλ[mm]
# Segment lengthの情報も入れておく
k_L = 2*np.pi / L_vox # [rad/voxel]
ax.axvline(k_L, color='purple', linestyle=':', alpha=0.7)
ax.annotate(
    f"segment length = {L_vox * voxel_size:.1f} mm\nk_L = {k_L:.3f} rad/vox",
    xy=(k_L, ax.get_ylim()[1] * 0.5),
    xytext=(5, 10),
    textcoords='offset points',
    color='purple',
    fontsize=9,
    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='purple', alpha=0.5)
)

# set_ticks / set_ticklabels は xaxis オブジェクトに対して行う
ax2.xaxis.set_major_locator(FixedLocator(lambda_ticks))
ax2.xaxis.set_major_formatter(FixedFormatter([f"{lt:.1f}" for lt in lambda_ticks]))

ax.set_yscale('log')
ax.set_title(f"segment ID. {segment_id}")

# Cell 82
# 帯域内の局所ピークを最大3個ほど抽出（パワー和 S=|U|^2+|V|^2 で上位）

from scipy.signal import find_peaks

def find_candidate_peaks_in_band(
    mode_arr, power_arr, k_arr,  # k[rad/voxel]
    lambda_prior,         # 事前波長 [voxel]
    mult_range=(0.5, 2.0),
    top_n=3,
    peak_power_frac=None, # 帯域内最大の30%以上のピーク（>= thr）だけを採用 => E_band × min_band_frac（global threshold）だけ残した方が合理的と考えて"None"としておく
    min_band_frac=0.10,   # E_bandの10%以上のピークだけを採用
    min_separation=2,     # 近接ピークの重複防止（DCT モード番号差）
    exclude_dc=True,
    include_edge=True,         # 帯域端（最初/最後）も候補に入れる
    force_take_argmax=True,    # 帯域内 argmax を必ず候補に入れる
):
    """
    DCTパワー和 S = |U|^2 + |V|^2xから、
    事前帯域内の局所ピークを“最大top_n個”抽出（端点&argmaxの取りこぼしを確実に回避）。

    Returns
    -------
    candidates : list[dict]
        各候補の情報を格納した辞書のリスト。power降順。
    meta : dict
        しきい値・帯域・検出数などのQC情報。
    """
    mode_arr  = np.asarray(mode_arr)
    power_arr = np.asarray(power_arr, float)
    k_arr     = np.asarray(k_arr, float)

    # ---- 1) 事前帯域 ----
    # λを基準に ±2倍の範囲を設定
    lam_min = mult_range[0] * lambda_prior
    lam_max = mult_range[1] * lambda_prior
    # λが大きいほどkは小さいので、逆順に変換
    k_min = 2 * np.pi / lam_max
    k_max = 2 * np.pi / lam_min
    sel = (k_arr >= k_min) & (k_arr <= k_max)
    if exclude_dc:
        sel = sel & (mode_arr >= 1) # 実際にはk_max>0なのでm=0 は自動的に除外されるのでいらないかも

    # --- 帯域内が空なら空返し ---
    if not np.any(sel):
        return [], dict(reason="no_band", band=None)

    # ---- 帯域内配列 ----
    band_idx = np.where(sel)[0]
    band_power = power_arr[band_idx]

    # ---- 帯域内最大と E_band ----
    max_power_band = float(np.max(band_power))
    total_power_band = float(np.sum(band_power))

    # ---- 閾値（local + global のどちらか有効な方） ----
    threshold_list = []
    if (peak_power_frac is not None) and (max_power_band > 0):
        threshold_local = max_power_band * peak_power_frac
        threshold_list.append(threshold_local)
    else:
        threshold_local = None

    if (min_band_frac is not None) and (total_power_band > 0):
        threshold_global = total_power_band * min_band_frac
        threshold_list.append(threshold_global)
    else:
        threshold_global = None

    # しきい値が一つも定義されない場合は 0 とする
    threshold_power = max(threshold_list) if len(threshold_list) > 0 else 0.0

    # --- 1) 内部ピーク （global/local 共通 threshold） ----
    idx_peaks_local, props = find_peaks(
        band_power, height=threshold_power, distance=min_separation
    )
    idx_peaks_abs = band_idx[idx_peaks_local] # 帯域->全体インデックスへ

    # --- 2) 端点ピークの手動追加 ---
    extra = []
    if include_edge and band_power.size >= 2:
        # 左端
        if (band_power[0] >= threshold_power) and (band_power[0] >= band_power[1]):
            extra.append(band_idx[0])
        # 右端
        if (band_power[-1] >= threshold_power) and (band_power[-1] >= band_power[-2]):
            extra.append(band_idx[-1])

    # --- 3) 帯域 argmax を必ず追加 ---
    if force_take_argmax:
        extra.append(band_idx[int(np.argmax(band_power))])

    # 統合（重複除去）
    if len(extra):
        idx_peaks_abs = np.unique(np.concatenate([idx_peaks_abs, extra]).astype(int))

    # --- 4) ピークがゼロの場合のフォールバック：閾値を満たす点の上位を採用 ---
    # 明確なピークが無かったときは、とりあえず一番大きい値（上位top_n点）を擬似ピークとして使用する
    if idx_peaks_abs.size == 0:
        # 閾値以上の点をパワー降順で上位抽出
        order = np.argsort(band_power)[::-1] # power降順
        idx_candidates = band_idx[order[:top_n]]
    else:
        idx_candidates = idx_peaks_abs

    # --- 5) 近接ピークの重複解消（強い方を残す） ---
    if idx_candidates.size > 1 and min_separation > 0: # 候補が2つ以上あり、重複除去の最小距離（min_separation）が設定されている場合にだけ実行
        order = np.argsort(power_arr[idx_candidates])[::-1] # power降順
        chosen = []
        # すでに選ばれたピーク（chosen）とのDCT モード番号差 |m[j]-m[q]| がmin_separation以上なら新しいピークとして採用
        # 近すぎる（差が小さい）場合は「同一山の中」とみなしてスキップ
        for j in idx_candidates[order]:
            if all(abs(mode_arr[j] - mode_arr[q]) >= min_separation for q in chosen):
                chosen.append(j)
        idx_candidates = np.array(sorted(chosen), dtype=int)

    # --- 6) 上位top_nに制限（powerを降順に並べ、上位top_n個だけ残す） ---
    idx_candidates = idx_candidates[np.argsort(power_arr[idx_candidates])[::-1]]
    if idx_candidates.size > top_n:
        idx_candidates = idx_candidates[:top_n]

    # ---- 出力 ----
    candidates = []
    for j in idx_candidates:
        lamb = (2 * np.pi / k_arr[j]) if k_arr[j] > 0 else np.inf
        candidates.append(dict(
            idx=int(j),
            mode=int(mode_arr[j]),
            k=float(k_arr[j]),
            lamb=float(lamb),
            power=float(power_arr[j]),
            power_frac_band=float(power_arr[j] / total_power_band),
            in_band=True
        ))

    meta = dict(
        k_band=(k_min, k_max),
        lambda_band=(lam_min, lam_max),
        max_power_band=max_power_band,
        total_power_band=total_power_band,
        threshold_local=threshold_local,
        threshold_global=threshold_global,
        threshold_used=threshold_power,
        n_in_band=len(band_power),
        n_candidates=len(candidates),
        min_separation=min_separation,
        top_n=top_n,
    )

    return candidates, meta

# Cell 83
cands, meta_cand = find_candidate_peaks_in_band(
    mode_arr=m_arr,
    power_arr=S_arr,
    k_arr=k_arr,
    lambda_prior=lambda_prior_vox,
    mult_range=(0.5, 2.0),
    top_n=3,
    peak_power_frac=None,
    min_band_frac=0.10,
    min_separation=2,
    exclude_dc=True,
    include_edge=True,
    force_take_argmax=True
) 

# Cell 84
# --- 事前帯域の設定 ---
mult_range = (0.5, 2.0)
lam_min = mult_range[0] * lambda_prior_vox
lam_max = mult_range[1] * lambda_prior_vox
k_band = (2*np.pi/lam_max, 2*np.pi/lam_min)  # (k_min, k_max)

# 作図
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
# U, V, Sum（m=0:DC は見栄えのため除く）
# --- m>=1 の範囲を band 内に絞るマスク ---
m1 = m_arr[1:]
k1 = k_arr[1:]
Uc1 = Uc_arr[1:]
Vc1 = Vc_arr[1:]
S1  = S_arr[1:]
ax.plot(k1, Uc_arr[1:]**2, alpha=0.3, marker='x', color="k")
ax.plot(k1, Vc_arr[1:]**2, alpha=0.3, marker='x', color="k")
ax.plot(k1, S_arr[1:], alpha=0.3, marker='x', color="k")

# band 内に入るインデックス
band_mask = (k1 >= k_band[0]) & (k1 <= k_band[1])
idx_band = np.where(band_mask)[0]

# band の前後1点ずつ拡張
start = max(idx_band[0] - 1, 0)
stop  = min(idx_band[-1] + 1, len(k1) - 1)

k_plot  = k1[start:stop+1]
Uc_plot = Uc1[start:stop+1]
Vc_plot = Vc1[start:stop+1]
S_plot  = S1[start:stop+1]

# U, V, Sum（band 内だけ）
ax.plot(k_plot, Uc_plot**2, label='|U|^2 (DCT power)', marker='o')
ax.plot(k_plot, Vc_plot**2, label='|V|^2 (DCT power)', marker='o')
ax.plot(k_plot, S_plot,      label='S = |U|^2 + |V|^2', linewidth=2, marker='o')

# 事前帯域の陰影
ax.axvspan(k_band[0], k_band[1], color='grey', alpha=0.15, label='prior band')

# ピークに印
if meta_peak.get('found', False):
    ax.axvline(meta_peak['k'], color='k', linestyle='--', alpha=0.6)
    ax.annotate(f"peak λ≈{meta_peak['lambda_est']*voxel_size:.1f} mm",
                xy=(meta_peak['k'], S_arr[meta_peak['idx']]),
                xytext=(5, 10), textcoords='offset points',
                fontsize=9,
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='k', alpha=0.5)  # 矢印スタイルarrowprops=dict(arrowstyle='-|>', lw=0.8, color='r', alpha=0.5)  # 矢印スタイル
    )
ax.set_xlabel('wavenumber k [rad/voxel]')
ax.set_ylabel('power')
ax.legend(loc='upper right')

k_prior_vox = 2*np.pi / lambda_prior_vox # [rad/voxel]
ax.axvline(k_prior_vox, color='r', linestyle='--', alpha=0.6)
ax.annotate(
    f"k_prior={k_prior_vox:.3f} rad/vox\n"
    f"prior λ≈{lambda_prior_vox*voxel_size:.1f} mm",   # 表示する文字
    xy=(k_prior_vox, ax.get_ylim()[1]*0.9),                 # 矢印の基点（x=k_prior, y=縦軸の90%付近）
    xytext=(5, 10),                                     # 文字のオフセット [pixel単位]
    textcoords='offset points',
    color='r',
    fontsize=9,
    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='r', alpha=0.5)  # 矢印スタイル
)


# 表示に上限をつけてみる。人の目で見える範囲として5mmを基準にしてみる(脳血管の径が1–3mm程度とすると、波長5mmの曲がりは「血管径の1.5〜5倍」)
lambda_min_mm = 5     # 5 mm → 高周波（細かい）
k_max = 2*np.pi*voxel_size / lambda_min_mm
k_max = max(k_max, k_band[1]+0.5) # 5mmよりも小さい範囲までk_bandがカバーする場合に備える
k_min = 1e-2  # 表示範囲（0 から始めないため）
ax.set_xlim(k_min, k_max)

# 上部に λ[mm] の目盛り（補助目盛）
ax2 = ax.secondary_xaxis('top', functions=(
    lambda kk: 2*np.pi/np.maximum(kk, 1e-12)*voxel_size,   # k -> lambda_mm
    lambda ll: 2*np.pi/np.maximum(ll/voxel_size, 1e-12)    # lambda_mm -> k
))
ax2.set_xlabel('wavelength λ [mm]')

# λ目盛りを自分で指定
k_ticks = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
lambda_ticks = 2*np.pi*voxel_size / k_ticks  # 対応するλ[mm]
# Segment lengthの情報も入れておく
k_L = 2*np.pi / L_vox # [rad/voxel]
ax.axvline(k_L, color='purple', linestyle=':', alpha=0.7)
ax.annotate(
    f"segment length = {L_vox * voxel_size:.1f} mm\nk_L = {k_L:.3f} rad/vox",
    xy=(k_L, ax.get_ylim()[1] * 0.5),
    xytext=(5, 10),
    textcoords='offset points',
    color='purple',
    fontsize=9,
    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='purple', alpha=0.5)
)
# set_ticks / set_ticklabels は xaxis オブジェクトに対して行う
ax2.xaxis.set_major_locator(FixedLocator(lambda_ticks))
ax2.xaxis.set_major_formatter(FixedFormatter([f"{lt:.1f}" for lt in lambda_ticks]))

# --- ここから追記：cands を印＋注釈 ---
if 'cands' in locals() and len(cands) > 0:
    # 1) 印（丸）と縦線
    xk = [c['k'] for c in cands]
    yS = [S_arr[c['idx']] for c in cands]
    ax.scatter(xk, yS, s=64, facecolors='none', edgecolors='k', zorder=5, label='peaks')
    for c in cands:
        ax.axvline(c['k'], color='k', linestyle='--', alpha=0.35)

    # 2) 注釈：m, k, power, λ を表示（重なり回避に上下交互オフセット）
    for i, c in enumerate(cands):
        dx, dy = 6, (12 if (i % 2 == 0) else -18)  # pixelオフセット
        text = (f"m={c['mode']}\n"
                f"k={c['k']:.3f} rad/vox\n"
                f"power={c['power']:.2e}\n"
                f"λ≈{c['lamb']*voxel_size:.2f} mm\n"
                f"power_frac_band={c['power_frac_band']*100:.1f}%")
        ax.annotate(text,
                    xy=(c['k'], S_arr[c['idx']]),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', alpha=0.95),
                    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='k', alpha=0.6))

    # 凡例重複を避ける（既に 'peaks' がある場合でもOK）
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc='upper right')

ax.set_title(f"segment ID. {segment_id}")

# Cell 86
# --- 1) DCT で得た候補 k を seeds にする ---
k_seeds_vox = [c['k'] for c in cands]         # rad/voxel
m_seeds = [c['mode'] for c in cands]         # DCT モード番号
#plot時に計算済み: k_prior = 2*np.pi / float(lambda_prior_vox)  # 事前（rad/voxel）



# Cell 90
from numpy.linalg import lstsq
from scipy.fft import rfft, rfftfreq

# ----------------------------
# Step 1：設計行列（sin+trend）; y = X\beta + \varepsilonのデザイン行列Xを作成
# ----------------------------
def _design_matrix(s_arr, k_arr, trend_degree=1):
    """
    Args:
    s_arr: 弧長
    k_arr: 波数
    trend_deg: 0:定数, 1:定数+一次, 2:定数+一次+二次
    Returns:
    [cos(ks), sin(ks), 1, s0, (s0^2)]（s0 = s - mean(s)）
    """
    s = np.asarray(s_arr, float)
    s0 = s - s.mean()  # 数値安定のため中心化(centerling)(sを中心化（平均0）するとトレンドの多重共線性を緩和)
    X = [np.cos(k_arr*s), np.sin(k_arr*s), np.ones_like(s)]
    if trend_degree >= 1:
        X.append(s0)
    if trend_degree >= 2:
        X.append(s0**2)
    return np.column_stack(X)

# ----------------------------
# Step 2：（加重）最小二乗；最小二乗で係数を求める
# ----------------------------
def _solve_linear(X, y, weights=None, return_info=False):
    """
    線形回帰（最小二乗）を解く。重みがあればWLS、なければOLS
    X: (N,p), y: (N,), w: (N,) or None
    OLS: weights=None、WLS: weights>0 -> sqrt(weights)で加重(※直接 W を作って行列積を計算するより、各行に \sqrt{w_i} を掛ける形に変換すると効率的 \sum_i w_i (y_i - X_i \beta)^2 = \sum_i (\sqrt{w_i}(y_i - X_i \beta))^2))
    """
    if weights is not None:
        weights = np.asarray(weights, float)
        W = np.sqrt(np.clip(weights, 0.0, np.inf))[:, None] # np.clip(w, 0.0, np.inf)：負の重みがもしあっても0〜∞の範囲に制限
        Xw = X * W
        yw = y * W.squeeze()
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef # @は行列積演算子

    # R2・SSE(残差平方和)
    resid = y - yhat
    sse = float(np.sum((resid if weights is None else np.sqrt(weights)*resid)**2))
    if return_info:  # n:データ点の個数, p:説明変数(パラメータ数)の個数 ex X = [cos(k s), sin(k s), 1, s] → p=4
        return coef, yhat, dict(sse=sse, n=X.shape[0], p=X.shape[1])
    
    return coef, yhat # 係数と当てはめ結果の両方を返す

# Cell 91
# R^2 functions
def r2_unweighted(y_true, y_pred):
    """未重みの決定係数 R^2"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1.0 - ss_res / ss_tot)

def r2_weighted(y_true, y_pred, weights):
    # https://kdsv.jp/news/archives/4324#:~:text=重み付き回帰分析と,方法の一つです%E3%80%82&text=どうしても観測データに不,場合に使われます%E3%80%82
    """重み付き R^2（TSS も RSS も重み付き）"""
    weights = np.asarray(weights, float)
    wsum = np.sum(weights)
    ybar_w = np.sum(weights * y_true) / wsum
    ss_res_w = np.sum(weights * (y_true - y_pred)**2)
    ss_tot_w = np.sum(weights * (y_true - ybar_w)**2)
    return float(1.0 - ss_res_w / ss_tot_w)

# Cell 92
def _design_matrix_trend_only(s_arr, trend_degree=1):
    """
    Args:
    s_arr: 弧長
    trend_degree: 0:定数, 1:定数+一次, 2:定数+一次+二次
    Returns:
    [1, s0, (s0^2)]（s0 = s - mean(s)）
    """
    s = np.asarray(s_arr, float)
    s0 = s - s.mean()  # 数値安定のため中心化(centerling)(sを中心化（平均0）するとトレンドの多重共線性を緩和)
    X = [np.ones_like(s)]
    if trend_degree >= 1:
        X.append(s0)
    if trend_degree >= 2:
        X.append(s0**2)
    return np.column_stack(X)

# Cell 93
# ----------------------------
# Step 3：成分同時フィット（同一k）；同じkでuUとuVを同時評価
# ----------------------------
def fit_sin_trend_two_component(uU_arr, uV_arr, s_arr, k_arr,
                                weights=None,
                                trend_degree=1):
    """
    あるひとつのkを固定し、同じ設計行列XでuUとuVを別々に回帰
    cos, sin 係数からそれぞれの振幅を計算し、両者をまとめた 回転不変の合成振幅A=\sqrt{a_U^2+b_U^2+a_V^2+b_V^2} を算出。
    合成R^2（UとVを縦連結したSSE/SSTで評価）を計算
    Args:
    uU, uV, s: 1D arrays（端点トリム済み）
    k
    weights: None -> OLS（Core 推奨）
             array -> WLS（端点等に重みをかけたい場合のみ）
    Return: 
    dict（A, lambda_vox, R2_tot, 各係数, etc）
    """
    uU = np.asarray(uU_arr, float)
    uV = np.asarray(uV_arr, float)
    s = np.asarray(s_arr, float)

    X = _design_matrix(s, k_arr, trend_degree=trend_degree)

    cU, yU = _solve_linear(X, uU, weights=weights)
    cV, yV = _solve_linear(X, uV, weights=weights)

    # 係数の意味： [cos, sin, 1, s, (s^2)]
    def _amp_from_coef(c):
        a, b = float(c[0]), float(c[1])
        return a, b, float(np.sqrt(a**2 + b**2)) # cos・sin係数a,bおよび振幅A 

    aU, bU, AU = _amp_from_coef(cU)
    aV, bV, AV = _amp_from_coef(cV)
    A = float(np.sqrt(aU**2 + bU**2 + aV**2 + bV**2))  # 合成振幅

    # 合成 R^2（UとVを縦に連結した合成残差で計算）
    y_true_cat = np.concatenate([uU, uV])
    y_pred_cat = np.concatenate([yU, yV])
    R2_tot = r2_unweighted(y_true_cat, y_pred_cat)

    # sin + trend vs trend onlyを比較
    # full model の残差平方和
    resid_full = y_true_cat - y_pred_cat
    if weights is None:
        SSE_full = float(np.sum(resid_full**2))
    else:
        w = np.asarray(weights, float)
        w_cat = np.concatenate([w, w])
        SSE_full = float(np.sum(w_cat * resid_full**2))

    # trendのみモデル
    X_tr = _design_matrix_trend_only(s, trend_degree=trend_degree)
    _, yU_tr = _solve_linear(X_tr, uU, weights=weights)
    _, yV_tr = _solve_linear(X_tr, uV, weights=weights)
    y_pred_tr_cat = np.concatenate([yU_tr, yV_tr])
    resid_tr = y_true_cat - y_pred_tr_cat
    if weights is None:
        SSE_tr = float(np.sum(resid_tr**2))
    else:
        # w_catは上で計算済み
        SSE_tr = float(np.sum(w_cat * resid_tr**2))

    # 部分決定係数(partial_R2): 1 - reduced model（単純なモデル）の誤差平方和SSE_reduced / full model（説明変数を追加したモデル）の誤差平方和SSE_full
    # 「トレンドだけ」のモデルでは説明できていなかった変動のうち、「トレンド＋sin」モデルが“新たに説明できた割合
    partial_R2 = 1.0 - SSE_full / SSE_tr

    R2_w = None
    if weights is not None:
        # w_catは上で計算済み
        R2_w = r2_weighted(y_true_cat, y_pred_cat, w_cat)

    return dict(
        k=float(k_arr),
        lamb=float(2.0*np.pi / max(k_arr, 1e-12)),
        A_comb=float(A), A_U=AU, A_V=AV,
        uU_cos=aU, uU_sin=bU, # uUのcos,sin係数
        uV_cos=aV, uV_sin=bV, # uVのcos,sin係数
        coef_U=cU, coef_V=cV,
        yhat_U=yU, yhat_V=yV,
        R2=R2_tot,             # 未重み合成R^2
        R2_weighted=R2_w,      # 重み付き合成R^2（weightsがあれば数値、無ければNone）
        partial_R2=partial_R2, # 部分決定計数
        resid_full=resid_full,
        resid_tr=resid_tr,
        SSE_full=SSE_full,
        SSE_tr=SSE_tr
    )


# Cell 96
# ----------------------------
# Step 4：k探索；最適な波数 k を「粗→細」で探す
# 細探索のときのgrid幅は疎探索のときの0.25binと同じ -> 中心を再設定して再評価することで結果が安定し、R^2カーブがきれいになるということが目的
# ----------------------------

# 疎・細探索のグリッド点を作成
# 前準備
def _dk_bin_from_s(N, ds):
    """1 binの波数幅 Δk_bin [rad/voxel] を計算"""
    dk_bin = np.pi / (N * ds)
    return dk_bin

def _grid_around(k0, N, ds, B_bins, q_rel, step_frac):
    """
    k_seedsの周りを"bin幅"と"％幅"の大きい方で半幅を決めて等間隔グリッドを返す。
    B_bins: サーチに使うbin幅の数（粗:1.5, 細:0.5）
    q_rel : サーチに使う相対％（粗:0.30, 細:0.10）
    step_frac: グリッド刻み=step_frac*dk_bin (1bin幅を細かく分割, 0.25を基本)
    """
    k0 = float(k0)
    dk_bin = _dk_bin_from_s(N, ds)

    half_bin = float(B_bins) * dk_bin
    half_pct = float(q_rel) * k0
    half = max(half_bin, half_pct)

    dk = max(step_frac * dk_bin, 1e-12)
    lo, hi = max(k0 - half, 1e-12), k0 + half
    # 刻み幅dkで等間隔に点を置くとき、必要な点数(端の浮動誤差で1点足りない/余るのを避ける)
    n = int(np.ceil((hi - lo)/dk)) + 1 # np.ceil(x)でx以上の最小の整数を返す
    ks = lo + np.arange(n) * dk
    return ks
def k_to_dct_index(k, N, ds):
    """k [rad/voxel] から DCT モード番号 m に変換"""
    m = k * N * ds / np.pi
    return int(np.floor(m + 0.5))

# def harmonic_ratio_dct(m, S)は作成済みだが再度掲載
def harmonic_ratio_dct(m, S):
    """
    DCT-IIの近似ビンでE3/E1を評価（CoreのQC用）。 ※E3=第3次高調波
    """
    m1 = int(m)
    m3 = 3 * m1

    if m1 <= 0 or m3 >= len(S): # ガード：範囲外
        return float('nan'), dict(m1=m1, m3=m3, reason="out_of_range")
    else:
        E1 = float(S[m1]) # 1st harmonic, fundamental frequency
        E3 = float(S[m3])
        return float(E3/(E1 + 1e-12)), dict(m1=m1, m3=m3, reason="ok")
# 当初このdef harmonic_ratio_dct(m, S)の使用を考えていたが、DCTのビンmへの丸めにより離散ビン依存のアーティファクトが出るため使用不可
# いま試しているkそのもので「基本波の振幅」と「3次高調波の振幅」を回帰で計算し、その比をhrとする。これならビン丸めやリークの影響を受けない（先ほどのdef harmonic_ratio_dct(m, S)はあくまでDCT時の初期のQC）
def harmonic_ratio_from_fit(uU, uV, s, k, ds, weights=None, trend_degree=1):
    """
    kを固定してsin+trend をまず当て、残差に3kの正弦・余弦を当ててA3/A1およびhr=E3/E1を返す。WLSウェイトも使用可。

    # ----------------------------
	# 良好 E3/E1 < 0.10 … きれいな正弦に近い（歪み小, 弱非線形の buckling-like に整合）→ Good
    # 注意 E3/E1 > 0.10-0.25 … 軽度の歪み（近接ピーク干渉や地形の影響もあり得る, 境界や局所拘束の影響か、複合成分の可能性）→ Acceptable
    # 警告 E3/E1 > 0.25 … 強い歪み or 別周波の混合（ただし短枝や端点影響が強いときは例外あり） →  歪み顕著。候補の見直し（他ピーク／デトレンド／窓）を推奨 → それでも良ければ Caution
    # ----------------------------
    """
    uU = np.asarray(uU, float)
    uV = np.asarray(uV, float)
    s = np.asarray(s, float)    

    # 1) 基本波kで二成分同時回帰（sin+trend）
    fit = fit_sin_trend_two_component(uU, uV, s, k,
                                      weights=weights,
                                      trend_degree=trend_degree)

    # 基本波の合成振幅（回転不変）
    A1 = float(fit['A_comb'])  # = sqrt(aU^2+bU^2+aV^2+bV^2)

    if 3.0 * k >= np.pi / ds:
        return dict(hr_energy=np.nan, hr_amp=np.nan, A1=A1, A3=np.nan, fit=fit)

    # 2) 残差に3k成分を当てて3次高調波の合成振幅を出す
    rU = uU - fit['yhat_U']
    rV = uV - fit['yhat_V']
    X3 = _design_matrix(s, 3*k, trend_degree=0) # [cos, sin, 1 ] -> 3倍波（3k成分）の回帰モデルを作るための設計行列

    (a3U, b3U, cU3), y3U = _solve_linear(X3, rU, weights=weights)
    (a3V, b3V, cV3), y3V = _solve_linear(X3, rV, weights=weights)

    A3 = float(np.sqrt(a3U**2 + b3U**2 + a3V**2 + b3V**2))  # 3次高調波の合成振幅

    # エネルギー比および振幅比
    hr_energy = (A3*A3) / (A1*A1 + 1e-12)
    hr_amp    =  A3 / (A1 + 1e-12)

    return dict(hr_energy=float(hr_energy), hr_amp=float(hr_amp), A1=A1, A3=A3, fit=fit)

# 統計学的に同等のkをもつTie setから1つに絞る => 優先順位（辞書順）を決める
# ΔR² ≤ 0.006 のTie set => hr小さい順 => k_priorに近い順 => k小さい順 とする (# temporary note(8. AICと決定係数R^2の関係を導出、同等と見做せる$\Delta R^2$を求める)にて、ΔAIC<2が同等、を参考に同等と見做せるΔR^2を0.06とした)
# hrはそのkで当てた波がどれくらいキレイな正弦波か、波形が座屈っぽい 1 周期に近いか？、それともギザギザ（高調波だらけ）なのか？の指標（純粋にデータから評価する指標）
# k_priorは理論的に期待される波数からのズレを抑えるための指標だが、“正解” ではなく“だいたいこのあたり”の目安に近い
def choose_k_from_tie_set(candidates, 
                          lambda_prior=None,
                          dR2_tol=0.006, 
                          dhr_amp_tol=0.05): # tolerance, 許容範囲
    """
    candidates: list of dict
      各 dict に {'k':..., 'lambda':..., 'R2':..., 'hr_energy':..., 'hr_amp':..., 'fit':...} が入っている
    """
    # 1) R2の最大値 & Tie setの抽出
    R2_vals = np.array([c['R2'] for c in candidates], float)
    R2_max = R2_vals.max()
    tie = [c for c in candidates if (R2_max - c['R2']) <= dR2_tol]

    # 2) Tie set内でのhr_ampの“最小値 ± dhr_amp_tol”を同等とみなす
    if len(tie) == 1:
        # 1個しか無ければそれで決まり
        return tie[0], tie
    
    hr_amp_vals = np.array([c['hr_amp'] for c in tie])
    hr_amp_min = float(hr_amp_vals.min())
    tie2 = [c for c in tie if (c['hr_amp'] - hr_amp_min) <= dhr_amp_tol]

    if len(tie2) == 0:
        # 万が一全部外れたら元の tie を使う
        tie2 = tie

    # 3) 優先順位づけ
    def _key(c):
        # λ_priorからの距離（log尺度）;「理論値からのズレ」は“何 mm 違うか”ではなく“何倍違うか”で測りたいのでlog尺度を使用
        if lambda_prior is None or not np.isfinite(lambda_prior):
            dlog = 0.0
        else:
            dlog = abs(np.log(c['lambda']) - np.log(lambda_prior))

        return (
            dlog,           # 1) 理論λ_priorに近い
            c['k'],         # 2) k小さい（低周波=長波長優先）
        )

    tie_sorted = sorted(tie2, key=_key) # 辞書順ソート(昇順)
    best = tie_sorted[0]
    return best, tie_sorted

# メイン関数
def grid_search_k(uU, uV, s, N, ds,
                  k_seeds,
                  pct_coarse=0.30, pct_fine=0.10,
                  B_bins_coarse=1.5, B_bins_fine=0.5,
                  step_frac=0.25,
                  weights=None,
                  trend_degree=1,
                  lambda_prior=None,       
                  ):
    """
    Args:
    uU, uV, s: 1D arrays（端点トリム済み）
    N: データ点数
    ds: サンプリング間隔 [voxel]
    k_seeds: 初期候補の波数（1個でも複数でも可）
    around_pct: 粗探索はk_seedの±30%、細探索はその最良解の±15%を均等分割してサーチ （割合）
    B_bins_coarse, B_bins_fine: 粗探索・細探索で使うbin幅の数（粗:1.5, 細:0.5）
    step_frac: グリッド刻み=step_frac*dk_bin (1bin幅を細かく分割, 0.25を基本)
    weights: None -> OLS（Core 推奨）
             array -> WLS（端点等に重みをかけたい場合のみ）

    trend_degree: 0:定数, 1:定数+一次, 2:定数+一次+二次
    lambda_prior: 事前波長（voxel単位）。

    Return:dict
    k_hat (float, rad/voxel): 最終的に選ばれた最良の波数 k^\*。粗探索→細探索の2段階で、スコア（既定ではR^2）が最大になったもの。
    lambda_vox (float, voxel): k^\*に対応する推定波長 λ=2π/k^\*
    R2 (float): 最終的な決定係数 R^2
    hr_amp (float): 最終的な高調波比 E3/E1（振幅比）※小さいほど「きれいな正弦（歪みが少ない）」。ナイキスト超過時はNaNに注意。
    hr_energy (float): 最終的な高調波比 E3/E1（エネルギー比）
    A_comb (float): 合成振幅
    A_U (float): U成分の振幅
    A_V (float): V成分の振幅
    coef_U (array): U成分の回帰係数（設計行列の列の順に対応）
    coef_V (array): V成分の回帰係数（設計行列の列の順に対応）
    yhat_U (array): U成分の当てはめ結果（可視化や残差解析に使う）
    yhat_V (array): V成分の当てはめ結果（可視化や残差解析に使う）
    """
    k_seeds = np.atleast_1d(np.asarray(k_seeds, float)) #np.atleast_1d()により、スカラ値も1次元配列化
    s = np.asarray(s, float)

    # 1) 粗探索（k_seedsの ±30% or ±1.5bin）
    # k_seedsの各seed k_0の周りを分割(_grid_around) -> 複数seedがあれば結合して重複を除去。
    # ※近い複数のピークなら粗グリッドが自然に重なってくれるので、そのままで問題なし。離れた複数のpeakでも、粗探索でどちらがデータに合うかを自動で判定。
    coarse_ks = np.unique(np.concatenate([_grid_around(k0=k0, 
                                                       N=N, 
                                                       ds=ds, 
                                                       B_bins=B_bins_coarse, 
                                                       q_rel=pct_coarse, 
                                                       step_frac=step_frac) for k0 in k_seeds])) # 粗探索では多少の k_grid の近接重複は完全に許容してOK
    cand = []
    for k in coarse_ks:
        fit = fit_sin_trend_two_component(uU, uV, s, k,
                                          weights=weights,
                                          trend_degree=trend_degree,
                                          )
        lam = fit["lamb"]
        r2 = fit['R2']
        partial_r2 = fit["partial_R2"]

        # 品質（高調波比）と prior 近さ
        hr = harmonic_ratio_from_fit(uU=uU, uV=uV, s=s, k=k, ds=ds,
                                     weights=weights,
                                     trend_degree=trend_degree)
        hr_energy = hr['hr_energy']
        hr_amp = hr['hr_amp'] 
        cand.append({'k': float(k),'lambda': float(lam),'R2': float(r2), "partial_R2": float(partial_r2), 'hr_energy': float(hr_energy), 'hr_amp': float(hr_amp), 'fit': fit})
    # 最大スコアの候補を選ぶのであれば score, k_best, fit_best, hr_best = max(cand, key=lambda z: z[0]) だが統計的にほぼ同等なkが複数ある場合に対応していない
    # temporary note(8. AICと決定係数R^2の関係を導出、同等と見做せる$\Delta R^2$を求める)にて、ΔAIC<2が同等、を参考に同等と見做せるΔR^2を0.06とした
    best_cand, _ = choose_k_from_tie_set(cand, 
                                        lambda_prior=lambda_prior,
                                        dR2_tol=0.006
                                        )
    k_best      = best_cand['k']
    lambda_best = best_cand['lambda']
    R2_best     = best_cand['R2']
    partial_R2_best = best_cand['partial_R2']
    hr_amp_best     = best_cand['hr_amp']
    hr_energy_best     = best_cand['hr_energy']

    # 2) 微調整（best の ±10% or ±0.5bin）
    fine_ks = _grid_around(k0=k_best, 
                           N=N, 
                           ds=ds, 
                           B_bins=B_bins_fine, 
                           q_rel=pct_fine, 
                           step_frac=step_frac)

    cand2 = []
    for k in fine_ks:
        fit = fit_sin_trend_two_component(uU, uV, s, k,
                                          weights=weights,
                                          trend_degree=trend_degree,
                                          )
        lam = fit["lamb"]
        r2 = fit['R2']
        partial_r2 = fit["partial_R2"]

        # 品質（高調波比）と prior 近さ
        hr = harmonic_ratio_from_fit(uU=uU, uV=uV, s=s, k=k, ds=ds,
                                     weights=weights,
                                     trend_degree=trend_degree)
        hr_energy = hr['hr_energy'] 
        hr_amp = hr['hr_amp']
        cand2.append({'k': float(k),'lambda': float(lam),'R2': float(r2), "partial_R2": float(partial_r2), 'hr_energy': float(hr_energy), 'hr_amp': float(hr_amp), 'fit': fit})

    best_cand2, _ = choose_k_from_tie_set(cand2, 
                                          lambda_prior=lambda_prior,
                                          dR2_tol=0.006
                                         )
    k_best2 = best_cand2['k']
    lambda_best2 = best_cand2['lambda']
    R2_best2 = best_cand2['R2']
    partial_R2_best2 = best_cand2['partial_R2']
    hr_amp_best2 = best_cand2['hr_amp']
    hr_energy_best2 = best_cand2['hr_energy']
    fit_best2 = best_cand2['fit']
    
    out = dict(
        k_hat=float(k_best2),
        lambda_hat=float(lambda_best2),
        R2=float(R2_best2),
        partial_R2=float(partial_R2_best2),
        hr_amp=float(hr_amp_best2),
        hr_energy=float(hr_energy_best2),
        A_comb=float(fit_best2['A_comb']),
        A_U=float(fit_best2['A_U']),
        A_V=float(fit_best2['A_V']),
        uU_cos=float(fit_best2['uU_cos']), uU_sin=float(fit_best2['uU_sin']),
        uV_cos=float(fit_best2['uV_cos']), uV_sin=float(fit_best2['uV_sin']), 
        coef_U=fit_best2['coef_U'],
        coef_V=fit_best2['coef_V'],
        yhat_U=fit_best2['yhat_U'],
        yhat_V=fit_best2['yhat_V'],
        fit=fit_best2,
          
        k_coarse=float(k_best),
        lambda_coarse=float(lambda_best),
        R2_coarse=float(R2_best),
        partial_R2_coarse=float(partial_R2_best),
        hr_amp_coarse=float(hr_amp_best),
        hr_energy_coarse=float(hr_energy_best),
        candidates_coarse=cand,
        candidates_fine=cand2,
    )
    return out

# Cell 97

result = grid_search_k(uU=uU_vox_arr_trimmed, uV=uV_vox_arr_trimmed, s=s_vox_arr_trimmed, N=N, ds=ds_vox,
                  k_seeds=k_seeds_vox,
                  pct_coarse=0.30, pct_fine=0.10,
                  B_bins_coarse=1.5, B_bins_fine=0.5,
                  step_frac=0.25,
                  weights=None,
                  trend_degree=1,
                  lambda_prior=lambda_prior_vox,
                  )

k_hat_vox = result['k_hat']
lambda_hat_vox = result['lambda_hat']
A_comb_hat_vox = result['A_comb']
A_u_hat_vox = result['A_U']
A_v_hat_vox = result['A_V']
uU_cos_vox = result['uU_cos']
uU_sin_vox = result['uU_sin']
uV_cos_vox = result['uV_cos']
uV_sin_vox = result['uV_sin']


R2 = result['R2']
partial_R2 = result['partial_R2']
hr_amp = result['hr_amp']
hr_energy = result['hr_energy']


cand1 = result['candidates_coarse']
cand2 = result['candidates_fine']

alpha_hat = result['lambda_hat'] / radius_median_vox

coverage_hat = L_vox_trimmed / result['lambda_hat']

SSE_full = result['fit']['SSE_full']
SSE_tr = result['fit']['SSE_tr']

# Cell 98
k_course = [c['k'] for c in cand1]
r2_course = [c['R2'] for c in cand1]
partial_r2_course = [c['partial_R2'] for c in cand1]
hr_amp_course = [c['hr_amp'] for c in cand1]
k_fine = [c['k'] for c in cand2]
r2_fine = [c['R2'] for c in cand2]
partial_r2_fine = [c['partial_R2'] for c in cand2]
hr_amp_fine = [c['hr_amp'] for c in cand2]

ax1 = fig.add_subplot(111) 
ax1.plot(k_course, r2_course, 'o-', label='R2 coarse search', color='C0')
ax1.plot(k_fine, r2_fine, 'o-', label='R2 fine search', color='C1')

# 事前帯域の陰影
ax1.axvspan(k_band[0], k_band[1], color='grey', alpha=0.15, label='prior band')
ax1.set_xlabel('Wave number k [rad/voxel]')
ax1.set_ylabel('R2')

ax1.plot(k_course, partial_r2_course, 'x-', label='partial R2 coarse search', color='C2')
ax1.plot(k_fine, partial_r2_fine, 'x-', label='partial R2 fine search', color='C3')

ax1.axvline(k_hat_vox, color='b', linestyle='-', alpha=0.5)
ax1.annotate(
    f"Selected k* = {k_hat_vox:.3f} rad/vox", 
    xy=(k_hat_vox, ax1.get_ylim()[1]*0.1), 
    xytext=(5, 10),                                     # 文字のオフセット [pixel単位]
    textcoords='offset points',
    color='b',
    fontsize=9,
    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='b', alpha=0.5)  # 矢印スタイル
)
ax1.axvline(k_prior_vox, color='r', linestyle='--', alpha=0.5)
ax1.annotate(
    f"k_prior={k_prior_vox:.3f} rad/vox",
    xy=(k_prior_vox, ax1.get_ylim()[1]*0.2),                 # 矢印の基点（x=k_prior, y=縦軸の90%付近）
    xytext=(5, 10),                                     # 文字のオフセット [pixel単位]
    textcoords='offset points',
    color='r',
    fontsize=9,
    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='r', alpha=0.5)  # 矢印スタイル
)
ax1.axvline(k_L, color='purple', linestyle=':', alpha=0.7)
ax1.annotate(
    f"k_L = {k_L:.3f} rad/vox",
    xy=(k_L, ax1.get_ylim()[1] * 0.3),
    xytext=(5, 10),
    textcoords='offset points',
    color='purple',
    fontsize=9,
    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='purple', alpha=0.5)
)
if meta_peak.get('found', False):
    ax1.axvline(meta_peak['k'], color='k', linestyle='--', alpha=0.6)
    ax1.annotate(f"peak DCT",
                xy=(meta_peak['k'], ax1.get_ylim()[1] * 0.4),
                xytext=(5, 10), textcoords='offset points',
                color='k',
                fontsize=9,
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='k', alpha=0.5)  # 矢印スタイルarrowprops=dict(arrowstyle='-|>', lw=0.8, color='r', alpha=0.5)  # 矢印スタイル
    )

ax2 = ax1.twinx()
ax2.plot(k_course, hr_amp_course, 's--', label='Harmonic ratio amp_(E3/E1) Coarse', color='C4')
ax2.plot(k_fine, hr_amp_fine, 's--', label='Harmonic ratio amp_(E3/E1) Fine', color='C5')
ax2.set_ylabel('Harmonic ratio amp_(E3/E1)', color='C0')
ax2.tick_params(axis='y', labelcolor='C0')

handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
ax1.legend(handler1 + handler2, label1 + label2, loc=1, borderaxespad=0.)



# Cell 99
# --- 事前帯域の設定 ---
mult_range = (0.5, 2.0)
lam_min = mult_range[0] * lambda_prior_vox
lam_max = mult_range[1] * lambda_prior_vox
k_band = (2*np.pi/lam_max, 2*np.pi/lam_min)  # (k_min, k_max)

# 作図
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
# U, V, Sum（m=0:DC は見栄えのため除く）
# --- m>=1 の範囲を band 内に絞るマスク ---
m1 = m_arr[1:]
k1 = k_arr[1:]
Uc1 = Uc_arr[1:]
Vc1 = Vc_arr[1:]
S1  = S_arr[1:]

ax.plot(k1, Uc_arr[1:]**2, alpha=0.3, marker='x', color="k")
ax.plot(k1, Vc_arr[1:]**2, alpha=0.3, marker='x', color="k")
ax.plot(k1, S_arr[1:], alpha=0.3, marker='x', color="k")

# band 内に入るインデックス
band_mask = (k1 >= k_band[0]) & (k1 <= k_band[1])
idx_band = np.where(band_mask)[0]

# band の前後1点ずつ拡張
start = max(idx_band[0] - 1, 0)
stop  = min(idx_band[-1] + 1, len(k1) - 1)

k_plot  = k1[start:stop+1]
Uc_plot = Uc1[start:stop+1]
Vc_plot = Vc1[start:stop+1]
S_plot  = S1[start:stop+1]

# U, V, Sum（band 内だけ）
ax.plot(k_plot, Uc_plot**2, label='|U|^2 (DCT power)', marker='o')
ax.plot(k_plot, Vc_plot**2, label='|V|^2 (DCT power)', marker='o')
ax.plot(k_plot, S_plot,      label='S = |U|^2 + |V|^2', linewidth=2, marker='o')


# 事前帯域の陰影
ax.axvspan(k_band[0], k_band[1], color='grey', alpha=0.15, label='prior band')

# ピークに印
if meta_peak.get('found', False):
    ax.axvline(meta_peak['k'], color='k', linestyle='--', alpha=0.6)
    ax.annotate(f"peak λ≈{meta_peak['lambda_est']*voxel_size:.1f} mm",
                xy=(meta_peak['k'], S_arr[meta_peak['idx']]),
                xytext=(5, 10), textcoords='offset points',
                fontsize=9,
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='k', alpha=0.5)  # 矢印スタイルarrowprops=dict(arrowstyle='-|>', lw=0.8, color='r', alpha=0.5)  # 矢印スタイル
    )
ax.set_xlabel('wavenumber k [rad/voxel]')
ax.set_ylabel('power')
ax.legend(loc='upper right')

k_prior_vox = 2*np.pi / lambda_prior_vox # [rad/voxel]
ax.axvline(k_prior_vox, color='r', linestyle='--', alpha=0.6)
ax.annotate(
    f"k_prior={k_prior_vox:.3f} rad/vox\n"
    f"prior λ≈{lambda_prior_vox*voxel_size:.1f} mm",   # 表示する文字
    xy=(k_prior_vox, ax.get_ylim()[1]*0.9),                 # 矢印の基点（x=k_prior, y=縦軸の90%付近）
    xytext=(5, 10),                                     # 文字のオフセット [pixel単位]
    textcoords='offset points',
    color='r',
    fontsize=9,
    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='r', alpha=0.5)  # 矢印スタイル
)


# 表示に上限をつけてみる。人の目で見える範囲として5mmを基準にしてみる(脳血管の径が1–3mm程度とすると、波長5mmの曲がりは「血管径の1.5〜5倍」)
lambda_min_mm = 5     # 5 mm → 高周波（細かい）
k_max = 2*np.pi*voxel_size / lambda_min_mm
k_max = max(k_max, k_band[1]+0.5) # 5mmよりも小さい範囲までk_bandがカバーする場合に備える
k_min = 1e-2  # 表示範囲（0 から始めないため）
ax.set_xlim(k_min, k_max)

# 上部に λ[mm] の目盛り（補助目盛）
ax2 = ax.secondary_xaxis('top', functions=(
    lambda kk: 2*np.pi/np.maximum(kk, 1e-12)*voxel_size,   # k -> lambda_mm
    lambda ll: 2*np.pi/np.maximum(ll/voxel_size, 1e-12)    # lambda_mm -> k
))
ax2.set_xlabel('wavelength λ [mm]')

# λ目盛りを自分で指定
k_ticks = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
lambda_ticks = 2*np.pi*voxel_size / k_ticks  # 対応するλ[mm]
# Segment lengthの情報も入れておく
k_L = 2*np.pi / L_vox # [rad/voxel]
ax.axvline(k_L, color='purple', linestyle=':', alpha=0.7)
ax.annotate(
    f"segment length = {L_vox * voxel_size:.1f} mm\nk_L = {k_L:.3f} rad/vox",
    xy=(k_L, ax.get_ylim()[1] * 0.5),
    xytext=(5, 10),
    textcoords='offset points',
    color='purple',
    fontsize=9,
    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='purple', alpha=0.5)
)
# set_ticks / set_ticklabels は xaxis オブジェクトに対して行う
ax2.xaxis.set_major_locator(FixedLocator(lambda_ticks))
ax2.xaxis.set_major_formatter(FixedFormatter([f"{lt:.1f}" for lt in lambda_ticks]))

# --- ここから追記：cands を印＋注釈 ---
if 'cands' in locals() and len(cands) > 0:
    # 1) 印（丸）と縦線
    xk = [c['k'] for c in cands]
    yS = [S_arr[c['idx']] for c in cands]
    ax.scatter(xk, yS, s=64, facecolors='none', edgecolors='k', zorder=5, label='peaks')
    for c in cands:
        ax.axvline(c['k'], color='k', linestyle='--', alpha=0.35)

    # 2) 注釈：m, k, power, λ を表示（重なり回避に上下交互オフセット）
    for i, c in enumerate(cands):
        dx, dy = 6, (12 if (i % 2 == 0) else -18)  # pixelオフセット
        text = (f"m={c['mode']}\n"
                f"k={c['k']:.3f} rad/vox\n"
                f"power={c['power']:.2e}\n"
                f"λ≈{c['lamb']*voxel_size:.2f} mm\n"
                f"power_frac_band={c['power_frac_band']*100:.1f}%")
        ax.annotate(text,
                    xy=(c['k'], S_arr[c['idx']]),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', alpha=0.95),
                    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='k', alpha=0.6))

    # 凡例重複を避ける（既に 'peaks' がある場合でもOK）
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc='upper right')

# 最終的に選ばれたk_hatを印
ax.axvline(k_hat_vox, color='blue', linestyle='-', alpha=0.7, label='selected k*')
ax.annotate(
    f"k*={k_hat_vox:.3f} rad/vox\n"
    f"λ*≈{result['lambda_hat']*voxel_size:.1f} mm",   # 表示する文字
    xy=(k_hat_vox, ax.get_ylim()[1]*0.8),             # 矢印の基点（x=k_hat_vox, y=縦軸の80%付近）
    xytext=(-80, -30),                                # 文字のオフセット [pixel単位]
    textcoords='offset points',
    color='blue',
    fontsize=9,
    arrowprops=dict(arrowstyle='-|>', lw=0.8, color='blue', alpha=0.5)  # 矢印スタイル
)

ax.set_title(f"segment ID. {segment_id}")

# Cell 100
if k_hat_vox <= k_band[0] + 1e-6:
    boundary_hit_low = True; boundary_hit_high = False
elif k_hat_vox >= k_band[1] - 1e-6:
    boundary_hit_low = False; boundary_hit_high = True
else:
    boundary_hit_low = False; boundary_hit_high = False

# Cell 101
# 1) 時間（弧長）方向の残差波形
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(s_vox_arr_trimmed, uU_vox_arr_trimmed, label='uU_trimmed (orig)')
axes[0].plot(s_vox_arr_trimmed, result['yhat_U'], label='uU_trimmed fit')
axes[0].plot(s_vox_arr_trimmed, uU_vox_arr_trimmed - result['yhat_U'], label='uU_trimmed residual')
axes[0].legend()
axes[0].set_ylabel('uU')

axes[1].plot(s_vox_arr_trimmed, uV_vox_arr_trimmed, label='uV_trimmed (orig)')
axes[1].plot(s_vox_arr_trimmed, result['yhat_V'], label='uV_trimmed fit')
axes[1].plot(s_vox_arr_trimmed, uV_vox_arr_trimmed - result['yhat_V'], label='uV_trimmed residual')
axes[1].legend()
axes[1].set_xlabel('s [vox]')
axes[1].set_ylabel('uV')


# 2) 残差スペクトル全体（DCT パワー）
m_o, S_o, *_ = dct_power_sum(uU_vox_arr_trimmed, uV_vox_arr_trimmed, detrend_deg=1)
m_r, S_r, *_ = dct_power_sum(uU_vox_arr_trimmed - result['yhat_U'],
                             uV_vox_arr_trimmed - result['yhat_V'],
                             detrend_deg=1)

from matplotlib.ticker import MaxNLocator

# Cell 103
from scipy.stats import f
from scipy.optimize import fsolve
df1 = 2
df2 = 100-4


Fcrit = f.ppf(0.95, dfn=df1, dfd=df2)

def func(x):
    return (x/(1-x))*(df2/df1) - Fcrit

sol = fsolve(func, 0.05)
partial_R2_thr = sol[0]

# Cell 104
# 残差をノイズの推定値として扱う
resid_full = result['fit']['resid_full']
resid_tr = result['fit']['resid_tr']

rms_residual_full = np.sqrt(np.mean(resid_full**2))
rms_residual_tr = np.sqrt(np.mean(resid_tr**2))




rms_A_comb_hat_vox = A_comb_hat_vox / np.sqrt(2)
SNR = rms_A_comb_hat_vox / rms_residual_full

# Cell 107
# inclusion (序盤でexclusion [<10mm]はすでに判定済み)

# 1) Long-wave / Straight-ish / detectable判定 (あやしい・座屈として語りたくない)
undetectable_reason = None  # ここで初期化
if boundary_hit_low and coverage_hat < 1.0: # coverageが1周未満で探索下限に張り付いた場合
    class_periodicity = 'Long-wave / Straight-ish / undetectable'
    undetectable_reason = "boundary_hit_low and coverage_hat"
elif coverage_hat < 0.5: # 半周期未満
    class_periodicity = 'Long-wave / Straight-ish / undetectable'
    undetectable_reason = "coverage_hat < 0.5"
elif A_comb_hat_vox < 1: # 解像度以下
    class_periodicity = 'Long-wave / Straight-ish / undetectable'
    undetectable_reason = "A_comb_hat_vox < 1"
elif SNR < 1.5: # ノイズに対して弱い
    class_periodicity = 'Long-wave / Straight-ish / undetectable'
    undetectable_reason = "SNR < 1.5"
elif partial_R2 < partial_R2_thr: # F検定的に「sinの寄与がほぼゼロ(p=0.05)」のライン
    class_periodicity = 'Long-wave / Straight-ish / undetectable'    
    undetectable_reason = "partial_R2 < partial_R2_thr"
elif R2 < 0.3: # Secondaryとして許せる最小R^2(モデル自体の当てはまりが悪い => Straight-ish行き)
    class_periodicity = 'Long-wave / Straight-ish / undetectable'  
    undetectable_reason = "R2 < 0.3"
    
# 2) Core判定 (かなり確かな座屈的なものだけを抽出)
elif coverage_hat >= 1.5 and partial_R2 >= 0.15 and R2 >= 0.6 and hr_amp <= 0.2: # hr_ampの基準は波形が“ほぼ正弦波”か否か
    class_periodicity = 'Core'
    undetectable_reason  = None

# 3) Secondary（残り、中途半端なもの）
else:
    class_periodicity = 'Secondary'
    undetectable_reason  = None


# Cell 108
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics

def save_result_as_pdf(result_text, output_path):
    """
    print結果（文字列）をPDFとして保存する関数
    日本語対応：HeiseiMin-W3 フォントを自動登録
    """

    # --- 日本語フォント登録 ---
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))

    # --- PDFスタイル設定 ---
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.fontName = "HeiseiKakuGo-W5"
    style.fontSize = 20
    style.leading=28

    # --- PDFドキュメント作成 ---
    doc = SimpleDocTemplate(output_path, pagesize=A4)

    # --- Paragraphは1行ずつ処理する必要がある ---
    story = []
    for line in result_text.split("\n"):
        story.append(Paragraph(line.replace(" ", "&nbsp;"), style))

    # --- PDF書き込み ---
    doc.build(story)
    return output_path

assessment_text = (
    "=======================" "\n"
    f"Class; {class_periodicity}\n"
    f"Undetectable reason; {undetectable_reason}\n"
    f"coverage_hat: {coverage_hat:.3f}\n"
    f"R2: {R2:.3f}\n"
    f"partial R2: {partial_R2:.3f}\n"
    f"hr_amp: {hr_amp:.3f}\n"
    f"alpha_hat: {alpha_hat:.3f}\n"
    f"lambda_hat: {lambda_hat_vox*voxel_size:.3f} mm\n"
    f"L_trimmed: {L_vox_trimmed*voxel_size:.3f} mm\n"
    f"SNR: {SNR:.2f}\n"
    f"振幅A: {A_comb_hat_vox*voxel_size:.2f} mm\n"
    f"振幅AのRMS: {rms_A_comb_hat_vox*voxel_size:.2f} mm\n"
)

# すでに作ってある PDF 出力関数に渡す
pdf_path = save_result_as_pdf(assessment_text, "")

# Cell 111
# "U,V が脳表と紐づいた(＝解剖学的に固定された)座標になっている場合にのみ"、minor/major振幅比<<1であれば、平面内つまり1D bucklingな性質をいえる
# 「変位はほとんど脳表接線平面の中で起きており，法線方向への飛び出しは小さい」 = buckling は脳表にほぼ平行な平面内で起きている」
minor_major_amp_ratio = min(A_u_hat_vox, A_v_hat_vox) / max(A_u_hat_vox, A_v_hat_vox)

# Cell 113
hU_vox = uU_cos_vox * np.cos(k_hat_vox * s_vox_arr_trimmed) + uU_sin_vox * np.sin(k_hat_vox * s_vox_arr_trimmed)
hV_vox = uV_cos_vox * np.cos(k_hat_vox * s_vox_arr_trimmed) + uV_sin_vox * np.sin(k_hat_vox * s_vox_arr_trimmed)

hU_mm = hU_vox * voxel_size
hV_mm = hV_vox * voxel_size

hUV_mm = np.column_stack([hU_mm, hV_mm])

# PCAを実行
pca_2d = PCA(n_components=2) # PCAを行ったり、PCAの結果を格納したりするための変数を、pcaとして宣言
pca_2d.fit(hUV_mm) # PCAを実行

eigenvalues = pca_2d.explained_variance_
eigenvalue1, eigenvalue2 = eigenvalues

directionality = eigenvalue2 / eigenvalue1        # 0〜1 の比


# PCAから固有ベクトルを取得
vecs = pca_2d.components_   # shape (2,2), 各行が固有ベクトル（λ1, λ2に対応）
vals = pca_2d.explained_variance_

tab10 = plt.get_cmap('tab10').colors  # 10色のタプルのリスト

center = hUV_mm.mean(axis=0)
for i in range(2):
    # 固有値の平方根をスケールとして使う
    length = 2.0 * np.sqrt(vals[i])
    v = vecs[i] * length
             [center[1], center[1] + v[1]],
             lw=2, color=tab10[i+1])


# Cell 114
data_vessel_segmentdata = {
    "case": [case_number], 
    "segment_id": [segment_id],
    "mca_segment": [mca_segment],
    "side": [side],
    "pts": [len(P_vox_3darr)],
    "ds_mm": [ds_vox * voxel_size],
    "L_full_mm": [s_vox_arr[-1] * voxel_size],
    "L_trimmed_mm": [L_vox_trimmed * voxel_size],
    "radius_median_mm": [radius_median_vox * voxel_size],
    "radius_mean_mm": [radius_mean_vox * voxel_size],
    "radius_std_mm": [radius_std_vox * voxel_size],
    "euclid_distance_mm": [euclid_d_vox * voxel_size],
    "TI": [TI],
    "rho": [rho],
    "kappa_median_mm": [kappa_median_vox / voxel_size],
    "kappa_mean_mm": [kappa_mean_vox / voxel_size],
    "kappa_std_mm": [kappa_std_vox / voxel_size],
    "kappa_rms_mm": [kappa_rms_vox / voxel_size],
    "kappa_p95_mm": [kappa_p95_vox / voxel_size],
    "kappa_max_mm": [kappa_max_vox / voxel_size],
    "kappa_min_mm": [kappa_min_vox / voxel_size],
    "slenderness": [slenderness],
    "eigval1_3d_mm": [eigval1],
    "eigval2_3d_mm": [eigval2],
    "eigval3_3d_mm": [eigval3],
    "linearity": [linearity],
    "planarity": [planarity],
    "scattering": [scattering],
    "omnivariance": [omni],
    "anisotropy": [aniso],
    "change_curvature": [c_curvature],

    "alpha_prior": [alpha_prior],
    "lambda_prior_mm": [lambda_prior_vox * voxel_size],
    "pts_trimmed": [len(P_vox_3darr_trimmed)],
    "class_periodicity": [class_periodicity],
    "undetectable_reason": [undetectable_reason],
    "k_hat_mm": [k_hat_vox / voxel_size],
    "lambda_hat_mm": [lambda_hat_vox * voxel_size],
    "amp_hat_mm": [A_comb_hat_vox * voxel_size],
    "amp_hat_U_mm": [A_u_hat_vox * voxel_size],
    "amp_hat_V_mm": [A_v_hat_vox * voxel_size],
    "coef_uU_cos_mm": [uU_cos_vox * voxel_size], 
    "coef_uU_sin_mm": [uU_sin_vox * voxel_size], 
    "coef_uV_cos_mm": [uV_cos_vox * voxel_size], 
    "coef_uV_sin_mm": [uV_sin_vox * voxel_size],
    "eigval1_2d_mm": [eigenvalue1],
    "eigval2_2d_mm": [eigenvalue2],    
    "directionality": [directionality],
    "minor_major_amp_ratio": [minor_major_amp_ratio],
    "amp_hat_over_L": [A_comb_hat_vox / L_vox_trimmed], # 無次元量
    "lambda_hat_over_radius": [lambda_hat_vox / radius_median_vox], # 無次元量
    "amp_hat_over_radius": [A_comb_hat_vox / radius_median_vox], # 無次元量
    "R2": [R2],
    "partial_R2": [partial_R2],
    "SSE_full": [SSE_full],
    "SSE_tr": [SSE_tr],
    "hr_amp": [hr_amp],
    "hr_energy": [hr_energy],
    "alpha_hat": [alpha_hat],
    "coverage_hat": [coverage_hat],
    "SNR": [SNR],
    "rms_residual_full_mm": [rms_residual_full * voxel_size],
    "rms_residual_tr_mm": [rms_residual_tr * voxel_size],
    "boundary_hit_low": [boundary_hit_low],
    "boundary_hit_high": [boundary_hit_high]
    }

# indexとcolumnsを設定してDataFrameを作成
df_vessel_segmentdata = pd.DataFrame(data_vessel_segmentdata)

# meshの統計量をcsvに保存
segmentdata_filename = f"{case_number}_vessel_segmentdata_{segment_id}.csv"
segment_path = os.path.join("", segmentdata_filename)
df_vessel_segmentdata.to_csv(segment_path, index=None) # 圧縮無し

# DataFrameを表示
df_vessel_segmentdata

# Cell 115
tool.notify(f"{case}, segment ID {segment_id}")
