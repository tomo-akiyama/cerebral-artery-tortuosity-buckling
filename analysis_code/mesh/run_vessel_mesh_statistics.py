#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate vessel mesh statistics
"""

########################################################
# ===== USER SETTINGS =====
directory_id = "IXI284-HH-2354"

voxel_size = 0.5  # mm
########################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import vtk
import pyvista as pv
from compute_vessel_mesh_statistics import (
    compute_edge_lengths, 
    compute_mesh_volume,
    classify_left_right_by_kmeans,
    split_mesh_left_right
)


def calculate_and_save_vessel_statistics(
    vessel_vtk_mc_taubin_butterfly,
    celldata,
    pointids,
    points,
    voxel_size,
    directory_id,
    csv_path=None,
    output_dir=None
):
    """
    血管の基本統計量を計算してExcelファイルに保存する
    
    Parameters
    ----------
    vessel_vtk_mc_taubin_butterfly : vtkPolyData
        血管メッシュ
    celldata : dict
        セルデータ
    pointids : list
        ポイントIDのリスト
    points : array-like
        ポイントの座標
    voxel_size : float
        ボクセルサイズ（mm）
    directory_id : str
        Directory_ID
    csv_path : str, optional
        CSVファイルのパス。指定された場合、directory_idが一致する行にデータを追加
    output_dir : str
        出力ディレクトリのパス（csv_pathが指定されない場合のみ使用）
    """
    print("directory_id", directory_id)
    case_id = directory_id[3:6].lstrip("0")
    case_id = str(case_id)
    print("case_id", case_id)

    n_faces = vessel_vtk_mc_taubin_butterfly.GetNumberOfPolys()
    edge_lengths_vox = compute_edge_lengths(vessel_vtk_mc_taubin_butterfly)
    volume_vox, area_vox = compute_mesh_volume(vessel_vtk_mc_taubin_butterfly)

    print(f"total length: {celldata['Length'].sum() * voxel_size:.3f} mm")
    print(f"number of segments: {len(pointids)}")
    print(f"number of faces: {n_faces}")
    print(f"number of edges: {edge_lengths_vox.size}")
    print(f"edge length median: {np.median(edge_lengths_vox)* voxel_size:.3f} mm")
    print(f"edge length mean: {edge_lengths_vox.mean() * voxel_size:.3f} mm")
    print(f"volume: {volume_vox * (voxel_size ** 3):.2f} mm^3")
    print(f"surface area: {area_vox * (voxel_size ** 2):.2f} mm^2")

    """
    fig, ax = plt.subplots()
    ax.hist(edge_lengths_vox*voxel_size, bins=50)
    ax.axvline(x = np.median(edge_lengths_vox)*voxel_size, c = "red")
    ax.axvline(x = edge_lengths_vox.mean()*voxel_size, c = "green")
    ax.set_xlabel("edge length [mm]")
    ax.set_ylabel("count")
    ax.set_title("Edge length distribution (mm)")
    

    # グラフ内の右上にテキストを配置（axes座標系を使用）
    text_lines = [
        f"case_id: {case_id}",
        f"total length: {celldata['Length'].sum() * voxel_size:.3f} mm",
        f"number of segments: {len(pointids)}",
        f"number of faces: {n_faces}",
        f"number of edges: {edge_lengths_vox.size}",
        f"edge length median: {np.median(edge_lengths_vox)* voxel_size:.3f} mm",
        f"edge length mean: {edge_lengths_vox.mean() * voxel_size:.3f} mm",
        f"volume: {volume_vox * (voxel_size ** 3):.2f} mm^3",
        f"surface area: {area_vox * (voxel_size ** 2):.2f} mm^2"
    ]
    
    
    # 右上から下に向かって配置
    y_start = 0.99
    y_spacing = 0.06
    for i, line in enumerate(text_lines):
        ax.text(0.99, y_start - i * y_spacing, line, 
                fontsize=10, ha='right', va='top',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    """

    # 左右に分けて情報を抽出
    is_right_cell, right_cells, left_cells, threshold_x, centers = classify_left_right_by_kmeans(
        points, pointids, n_iter=100, verbose=True
    )

    # メッシュを左右に分離
    right_vessel, left_vessel, right_center_x, left_center_x = split_mesh_left_right(
        vessel_vtk_mc_taubin_butterfly, threshold_x, verbose=True
    )

    rt_vessel_length =  celldata['Length'][right_cells].sum()
    lt_vessel_length =  celldata['Length'][left_cells].sum()

    n_faces_right = right_vessel.GetNumberOfPolys()
    n_faces_left = left_vessel.GetNumberOfPolys()
    edge_lengths_vox_right = compute_edge_lengths(right_vessel)
    edge_lengths_vox_left = compute_edge_lengths(left_vessel)
    volume_vox_right, area_vox_right = compute_mesh_volume(right_vessel)
    volume_vox_left, area_vox_left = compute_mesh_volume(left_vessel)

    print("================================================")
    print(f"Rt vessel total length: {rt_vessel_length * voxel_size:.3f} mm")
    print(f"Rt number of segments: {right_cells.size}")
    print(f"Rt number of faces: {n_faces_right}")
    print(f"Rt number of edges: {edge_lengths_vox_right.size}")
    print(f"Rt edge length median: {np.median(edge_lengths_vox_right)* voxel_size:.3f} mm")
    print(f"Rt edge length mean: {edge_lengths_vox_right.mean() * voxel_size:.3f} mm")
    print(f"Rt volume: {volume_vox_right * (voxel_size ** 3):.2f} mm^3")
    print(f"Rt surface area: {area_vox_right * (voxel_size ** 2):.2f} mm^2")
    print("================================================")
    print(f"Lt vessel total length: {lt_vessel_length * voxel_size:.3f} mm")
    print(f"Lt number of segments: {left_cells.size}")
    print(f"Lt number of faces: {n_faces_left}")
    print(f"Lt number of edges: {edge_lengths_vox_left.size}")
    print(f"Lt edge length median: {np.median(edge_lengths_vox_left)* voxel_size:.3f} mm")
    print(f"Lt edge length mean: {edge_lengths_vox_left.mean() * voxel_size:.3f} mm")
    print(f"Lt volume: {volume_vox_left * (voxel_size ** 3):.2f} mm^3")
    print(f"Lt surface area: {area_vox_left * (voxel_size ** 2):.2f} mm^2")

    pl = pv.Plotter()
    pl.add_mesh(vessel_vtk_mc_taubin_butterfly, color='grey', opacity=0.5)
    pl.add_mesh(right_vessel, color='green', opacity=0.8)
    pl.add_mesh(left_vessel, color='blue', opacity=0.8)
    
    # threshold_xを通るyz平面に平行な平面を追加
    # メッシュの範囲を取得
    mesh_pv = pv.wrap(vessel_vtk_mc_taubin_butterfly)
    bounds = mesh_pv.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
    
    # 平面のサイズ（yとzの範囲に合わせる）
    y_range = bounds[3] - bounds[2]
    z_range = bounds[5] - bounds[4]
    plane_size = max(y_range, z_range) * 1.5  # 少し大きめに
    y_center = (bounds[2] + bounds[3]) / 2
    z_center = (bounds[4] + bounds[5]) / 2
    
    # x=threshold_xの位置に平面を作成（法線ベクトルはx軸方向）
    plane_threshold = pv.Plane(
        center=[threshold_x, y_center, z_center],
        direction=[1, 0, 0],  # x軸方向の法線ベクトル
        i_size=plane_size,
        j_size=plane_size
    )
    pl.add_mesh(plane_threshold, color='red', opacity=0.3, show_edges=True)
    
    # 左右のvesselの中心地を通るyz平面に平行な平面を追加
    plane_right = pv.Plane(
        center=[right_center_x, y_center, z_center],
        direction=[1, 0, 0],
        i_size=plane_size,
        j_size=plane_size
    )
    pl.add_mesh(plane_right, color='lightgreen', opacity=0.2, show_edges=True)
    
    plane_left = pv.Plane(
        center=[left_center_x, y_center, z_center],
        direction=[1, 0, 0],
        i_size=plane_size,
        j_size=plane_size
    )
    pl.add_mesh(plane_left, color='lightblue', opacity=0.2, show_edges=True)
    
    pl.add_title("Right (green) and Left (blue) points on the centerlines")
    
    pl.camera_position = 'xy'
    pl.camera.azimuth = 180
    pl.camera.zoom(1.2)
    pl.show_axes()
    pl.show_bounds(grid='front', location='outer', all_edges=True, ticks='outside')
    pl.show(jupyter_backend="static")


    # meshデータの計算
    mesh_data = {
        "IXI_ID": case_id,

    # total
    "total_n_segments": len(pointids),
    "total_n_edges": edge_lengths_vox.size,
    "total_n_faces": n_faces,
    "total_mesh_volume": volume_vox * (voxel_size ** 3),
    "total_surface_area": area_vox * (voxel_size ** 2),
    "total_vessel_length": celldata["Length"].sum() * voxel_size,

    # right
    "rt_n_segments": right_cells.size,
    "rt_n_edges": edge_lengths_vox_right.size,
    "rt_n_faces": n_faces_right,
    "rt_mesh_volume": volume_vox_right * (voxel_size ** 3),
    "rt_surface_area": area_vox_right * (voxel_size ** 2),
    "rt_vessel_length": rt_vessel_length * voxel_size,

    # left
    "lt_n_segments": left_cells.size,
    "lt_n_edges": edge_lengths_vox_left.size,
    "lt_n_faces": n_faces_left,
    "lt_mesh_volume": volume_vox_left * (voxel_size ** 3),
    "lt_surface_area": area_vox_left * (voxel_size ** 2),
    "lt_vessel_length": lt_vessel_length * voxel_size,
    }

    # csvファイルに書き込む場合
    if csv_path is not None and directory_id is not None:
        # csvファイルを読み込む
        if os.path.exists(csv_path):
            df_csv = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"csvファイルが見つかりません: {csv_path}")
        
        # Directory_IDが一致する行を探す
        mask = df_csv['Directory_ID'] == directory_id
        matching_rows = df_csv[mask]
        
        # meshデータを追加/更新（IXI_IDは除外）
        for key, value in mesh_data.items():
            if key != "IXI_ID":
                df_csv.loc[mask, key] = value
        
        # csvファイルに保存
        df_csv.to_csv(csv_path, index=False)
        print(f"csvファイルを更新しました: {csv_path}")
        
        return df_csv[mask]
    
    
    # csvファイルに保存
    if output_dir is None:
        raise ValueError("output_dir must be specified when csv_path is None")
    
    # 出力ディレクトリが存在しない場合は作成
    utils.ensure_directory_exists(output_dir, verbose=True)
    
    df_vessel_meshdata = pd.DataFrame([mesh_data])
    csv_file_path = f"{output_dir}/{directory_id}_vessel_meshdata.csv"
    df_vessel_meshdata.to_csv(csv_file_path, index=None)
    print(f"csvファイルを保存しました: {csv_file_path}")
    return df_vessel_meshdata


# 必要なモジュールのインポート
# mesh_generationとvmtk_centerlinesは同じディレクトリ（mesh）にあるため、追加のパス設定は不要
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_path = os.path.join(project_root, 'analysis_code', 'utils')
sys.path.insert(0, os.path.join(project_root, 'analysis_code'))  # Add analysis_code to path for package imports
sys.path.append(utils_path)

import vmtk_centerlines as vmtk_CL
import utils
import mesh_generation


if __name__ == "__main__":
    """
    血管の基本統計量を計算してCSVに保存するスクリプト
    """

    dataset = " 系統解剖研究室/10 Geometric analysis 220401-/Dataset/IXI HH"
    case_path = os.path.join(dataset, directory_id)
    intermediate_data1_path = os.path.join(case_path, 'Intermediate_data1')
        
    # データの読み込み
    mca_intermediate_data3_3ds_path = os.path.join(case_path, 'MCA Intermediate_data3 3d slicer')
    vessel_vtk_mc_taubin_butterfly_path = os.path.join(intermediate_data1_path, "1_2_mca_mc_taubin_thick_butterfly16.ply")
    vessel_vtk_mc_taubin_butterfly = mesh_generation.load_ply_to_polydata(vessel_vtk_mc_taubin_butterfly_path)
    merged_splitted_centerlines_path = os.path.join(mca_intermediate_data3_3ds_path, "3 merged_splitted_centerlines.vtp")
    merged_splitted_centerlines = mesh_generation.load_vtp(merged_splitted_centerlines_path)
    celldata, pointids, points, pointdata = vmtk_CL.vmtk_convert2numpy(merged_splitted_centerlines)
    
    # csvファイルのパス
    csv_path = os.path.join(project_root, 'results', 'participants', 'IXI_HH_participants.csv')
    
    # 出力ディレクトリをスクリプトと同じ階層に作成
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, f"temporary_{directory_id}_vessel_statistics")
    utils.ensure_directory_exists(output_dir, verbose=True)
    
    # 統計量を計算してCSVファイルに保存
    df_vessel_meshdata = calculate_and_save_vessel_statistics(
        vessel_vtk_mc_taubin_butterfly=vessel_vtk_mc_taubin_butterfly,
        celldata=celldata,
        pointids=pointids,
        points=points,
        voxel_size=voxel_size,
        directory_id=directory_id,
        csv_path=csv_path,
        directory_id=directory_id,
        output_dir=output_dir
    )
    
    print(f"統計量の計算と保存が完了しました: {df_vessel_meshdata}")

