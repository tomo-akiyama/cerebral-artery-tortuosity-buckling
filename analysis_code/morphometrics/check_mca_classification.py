#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合CSVファイルからMCA分類を視覚的に確認・設定するスクリプト
"""

########################################################
# ===== USER SETTINGS =====
directory_id = "IXI000-HH-0000"

# 処理するセグメントIDのリスト（Noneの場合は全て）
target_segment_ids = None  # 例: [55, 56, 57] または None

# 開始するsegment_index（エラー時に途中から再開する場合に設定）
# Noneの場合は0から開始、特定の値が設定されている場合はその値以降から開始
start_segment_index = None # 例: 5 または None

# ===== Visualization settings =====
plotter_window_size = (2560, 1440)  # ウィンドウサイズ (width, height)
plotter_position = (500, 500)  # ウィンドウの位置 (x, y)
########################################################
print("="*80)
print("MCA分類の確認・設定を開始する...")
print("="*80)
print(f"directory_id: {directory_id}")
print(f"target_segment_ids: {target_segment_ids}")
print(f"start_segment_index: {start_segment_index}")
print("="*80)

import os
import sys
import numpy as np
import pandas as pd
import pyvista as pv

# Add the path to make the utils module importable
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_path = os.path.join(project_root, 'analysis_code', 'utils')
mesh_path = os.path.join(project_root, 'analysis_code', 'mesh')
sys.path.insert(0, os.path.join(project_root, 'analysis_code'))
sys.path.append(utils_path)
sys.path.append(mesh_path)

import utils
import mesh_generation
import centerline_extraction
import vessel_mesh_features
from sin_trend_fitting import clip_mesh_to_coords

# 統合CSVファイルのパス
output_dir_per_case = os.path.join(project_root, 'results', 'morphometrics', 'sin_trend', 'per_case')
csv_path = os.path.join(output_dir_per_case, f"{directory_id}_vessel_segmentdata.csv")

if not os.path.exists(csv_path):
    print(f"Error: 統合CSVファイルが見つからない: {csv_path}")
    sys.exit(1)

# 統合CSVファイルを読み込む
print(f"統合CSVファイルを読み込む: {csv_path}")
df_all = pd.read_csv(csv_path)

print(f"総レコード数: {len(df_all)}")
print()

# 処理対象のセグメントIDを決定
if target_segment_ids is None:
    segment_ids = sorted(df_all['segment_id'].unique())
else:
    segment_ids = [sid for sid in target_segment_ids if sid in df_all['segment_id'].values]

print(f"処理対象のsegment_id数: {len(segment_ids)}")
print(f"segment_idリスト: {segment_ids}")

# 開始するsegment_indexを設定
if start_segment_index is None:
    start_segment_index = 0
print(f"開始するsegment_index: {start_segment_index}")
print()

# Centerlineデータを読み込む
vessel_centerlines_dir = os.path.join(project_root, 'data', 'derived', 'vessel_centerlines', directory_id)
merged_splitted_centerlines_path = os.path.join(vessel_centerlines_dir, f"{directory_id}_merged_centerlines.vtp")
merged_splitted_centerlines = centerline_extraction.load_vtp(merged_splitted_centerlines_path)

celldata, pointids, points, pointdata = centerline_extraction.vmtk_convert2numpy(merged_splitted_centerlines)

# pointidsを辞書に変換
group_ids = celldata['GroupIds']
pointids_dict = {group_id: pointids[i] for i, group_id in enumerate(group_ids)}

# 左右の判定
right_cells, left_cells, threshold_x, is_right_point, is_left_point = vessel_mesh_features.classify_centerline_left_right_by_kmeans(
    points, pointids, n_iter=100, verbose=False)

# メッシュを読み込む
vessel_mesh_dir = os.path.join(project_root, 'data', 'preprocessed', 'MRA_preprocessed', 'vessel_meshes', directory_id)
vessel_mesh_path = os.path.join(vessel_mesh_dir, f"{directory_id}_vessel_mesh.ply")
vessel_mesh = mesh_generation.load_ply_to_polydata(vessel_mesh_path)
m1_mesh_path = os.path.join(vessel_mesh_dir, f"{directory_id}_M1_vessel_mesh.ply")
m1_mesh = mesh_generation.load_ply_to_polydata(m1_mesh_path)

brain_mesh_dir = os.path.join(project_root, 'data', 'preprocessed', 'T1_preprocessed', 'brain_meshes', directory_id)

def visualize_and_update_mca_segment(segment_id, row, df_all, csv_path, directory_id):
    """
    セグメントのMCA分類を視覚的に確認・設定する
    
    Args:
        segment_id: セグメントID
        row: DataFrameの行（セグメントのデータ）
        df_all: DataFrame全体
        csv_path: 統合CSVファイルのパス
        directory_id: ディレクトリID
    
    Returns:
        (success, changed): successは処理が成功したか、changedは変更があったか
    """
    # セグメントのデータを取得
    cell_index = row['segment_index']
    # CSVから読み込むとNoneはNaNになるため、pd.isna()でチェック
    mca_segment_raw = row.get('mca_segment', None)
    if pd.isna(mca_segment_raw):
        current_mca_segment = None
    else:
        current_mca_segment = mca_segment_raw
    
    print("\n" + "="*80)
    print(f"Processing segment_id: {segment_id}, segment_index: {cell_index}")
    print(f"現在のMCA分類: {current_mca_segment}")
    print("="*80)
    
    P_vox_3darr = points[pointids_dict[segment_id]]
    
    # 左右の判定
    if cell_index in right_cells:
        side = "Rt"
        brain_mesh_path = os.path.join(brain_mesh_dir, f"{directory_id}_rt_pial_remesh.ply")
    else:
        side = "Lt"
        brain_mesh_path = os.path.join(brain_mesh_dir, f"{directory_id}_lt_pial_remesh.ply")
    
    if not os.path.exists(brain_mesh_path):
        print(f"Error: Brain mesh file not found: {brain_mesh_path}")
        return False, False
    
    brain_mesh = pv.read(brain_mesh_path)
    cropped_mesh = clip_mesh_to_coords(P_vox_3darr, brain_mesh, margin=20)
    
    # キーボードイベントハンドラー
    mca_segment = current_mca_segment  # 現在の値を初期値として設定
    pl = None
    
    def set_m1():
        nonlocal mca_segment, pl
        mca_segment = "M1"
        print(f"mca_segment set to 'M1'")
        if pl is not None:
            try:
                pl.close()
            except:
                pass

    def set_m2_3():
        nonlocal mca_segment, pl
        mca_segment = "M2-3"
        print(f"mca_segment set to 'M2-3'")
        if pl is not None:
            try:
                pl.close()
            except:
                pass

    def set_m4():
        nonlocal mca_segment, pl
        mca_segment = "M4"
        print(f"mca_segment set to 'M4'")
        if pl is not None:
            try:
                pl.close()
            except:
                pass

    def set_no_change():
        nonlocal mca_segment, pl
        mca_segment = current_mca_segment  # 現在の値を維持
        print(f"変更なし（現在の値: {current_mca_segment}）")
        if pl is not None:
            try:
                pl.close()
            except:
                pass

    # PyVistaプロッターを作成
    pl = pv.Plotter(shape=(1,2), window_size=plotter_window_size)
    
    # 着目したい中心点
    focus = np.asarray(P_vox_3darr).mean(axis=0)
    mid_pt = P_vox_3darr[len(P_vox_3darr)//2]
    
    # 説明テキストを追加
    pl.add_text(f"Press key: M1 = 1, M2-3 = 2 or 3, M4 = 4", 
                position='lower_left', font_size=18, color='red')
    pl.add_text(f"Current: {current_mca_segment}", position='upper_right', font_size=18, color='blue')
    
    # 左側のサブプロット（拡大表示）
    pl.subplot(0,0)
    pl.add_mesh(vessel_mesh, color='lightblue', opacity=0.5)
    pl.add_mesh(m1_mesh, color='blue', opacity=0.3)
    pl.add_mesh(brain_mesh, color='ivory', opacity=0)
    pl.add_points(P_vox_3darr, color='red', point_size=5)
    pl.add_points(mid_pt, color='red', opacity=0.8, point_size=30, render_points_as_spheres=False, ambient=1.0)
    pl.add_mesh(cropped_mesh, color="#fafae1", opacity=0.8)
    pl.camera_position = 'xz'
    pl.camera.azimuth = 180
    pl.set_focus(focus)
    pl.camera.zoom(5.0)
    pl.show_bounds(grid='front', location='outer', all_edges=True, ticks='outside')
    
    # 右側のサブプロット（全体表示）
    pl.subplot(0,1)
    pl.add_mesh(vessel_mesh, color='lightblue', opacity=0.5)
    pl.add_mesh(m1_mesh, color='blue', opacity=0.3)
    pl.add_mesh(brain_mesh, color='ivory', opacity=1)
    pl.add_points(P_vox_3darr, color='red', point_size=5)
    pl.add_points(mid_pt, color='red', opacity=0.8, point_size=30, render_points_as_spheres=False, ambient=1.0)
    pl.camera_position = 'xz'
    pl.camera.azimuth = 180
    pl.camera.zoom(1.0)
    pl.show_bounds(grid='front', location='outer', all_edges=True, ticks='outside')
    pl.add_title(f"MCA Classification Check, segment ID: {segment_id}, segment index: {cell_index}", font_size=18)
    
    # キーボードイベントを追加（render()の前に登録）
    pl.add_key_event('1', set_m1)
    pl.add_key_event('2', set_m2_3)
    pl.add_key_event('3', set_m2_3)  # 3もM2-3に設定
    pl.add_key_event('4', set_m4)
    pl.add_key_event('space', set_no_change)  # spaceキーで変更なし
    
    # ウィンドウ位置を設定
    pl.render()
    if hasattr(pl, 'render_window') and pl.render_window is not None:
        pl.render_window.SetPosition(plotter_position[0], plotter_position[1])
    
    # キー入力待ち
    try:
        pl.show()
    except (AttributeError, RuntimeError):
        pass
    
    # キー入力後の確認
    if mca_segment is None:
        print("Error: mca_segmentが未設定。M1(1)、M2-3(2/3)、M4(4)のいずれかを選択する。")
        return False, False  # 失敗
    
    # MCA分類が追加/変更されたかチェック
    if mca_segment != current_mca_segment:
        if current_mca_segment is None:
            print(f"MCA分類を追加: {current_mca_segment} -> {mca_segment}")
        else:
            print(f"MCA分類が変更された: {current_mca_segment} -> {mca_segment}")
        
        # 統合CSVファイルを更新
        mask = (df_all['directory_id'] == directory_id) & (df_all['segment_id'] == segment_id) & (df_all['segment_index'] == cell_index)
        if mask.any():
            df_all.loc[mask, 'mca_segment'] = mca_segment
            # 変更があったら即座に統合CSVファイルに保存
            df_all.to_csv(csv_path, index=False)
            if current_mca_segment is None:
                print(f"  統合CSVファイルにmca_segmentを追加して保存した")
            else:
                print(f"  統合CSVファイルを更新して保存した")
        else:
            print(f"Error: セグメントが見つからない: {directory_id}, {segment_id}, {cell_index}")
            return False, False
        
        return True, True  # (success, changed)
    else:
        print(f"  MCA分類は変更されなかった: {mca_segment}")
        return True, False  # (success, not_changed)

# 各セグメントを処理
print("\n" + "="*80)
print("MCA分類の確認・設定を開始する...")
print("="*80)
print("キー操作:")
print("  1 → M1")
print("  2 or 3 → M2-3")
print("  4 → M4")
print("  spaceキー → 変更なし（現在の値を維持）で次のセグメントへ")
print("="*80)

updated_count = 0
skipped_count = 0
changed_segments = []  # 変更があったセグメントのリスト

# segment_idとsegment_indexの組み合わせで処理
# segment_indexでソートして、start_segment_index以降から処理
all_segments = []
for segment_id in segment_ids:
    segment_rows = df_all[df_all['segment_id'] == segment_id]
    for _, row in segment_rows.iterrows():
        cell_index = row['segment_index']
        all_segments.append((segment_id, cell_index, row))

# segment_indexでソート
all_segments.sort(key=lambda x: (x[0], x[1]))  # segment_id, segment_indexの順でソート

# start_segment_index以降のセグメントのみ処理
filtered_segments = []
for segment_id, cell_index, row in all_segments:
    if cell_index >= start_segment_index:
        filtered_segments.append((segment_id, cell_index, row))

print(f"処理対象のセグメント数: {len(filtered_segments)} (segment_index >= {start_segment_index})")
print()

for segment_id, cell_index, row in filtered_segments:
    # 変更前の値を保存
    old_mca_segment = row.get('mca_segment', None)
    
    try:
        result = visualize_and_update_mca_segment(segment_id, row, df_all, csv_path, directory_id)
        if isinstance(result, tuple):
            success, changed = result
            if success:
                updated_count += 1
                if changed:
                    # 変更後の値を統合CSVファイルから読み込む
                    df_updated = pd.read_csv(csv_path)
                    mask = (df_updated['directory_id'] == directory_id) & (df_updated['segment_id'] == segment_id) & (df_updated['segment_index'] == cell_index)
                    if mask.any():
                        new_mca_segment = df_updated.loc[mask, 'mca_segment'].iloc[0]
                        if pd.isna(new_mca_segment):
                            new_mca_segment = None
                    else:
                        new_mca_segment = "Unknown"
                    changed_segments.append((segment_id, cell_index, old_mca_segment, new_mca_segment))
            else:
                skipped_count += 1
                print(f"Warning: segment_id {segment_id}, segment_index {cell_index}の処理が失敗しました")
                print(f"  次のセグメントから再開する場合は start_segment_index = {cell_index + 1} を設定する")
        else:
            if result:
                updated_count += 1
            else:
                skipped_count += 1
                print(f"Warning: segment_id {segment_id}, segment_index {cell_index}の処理が失敗しました")
                print(f"  次のセグメントから再開する場合は start_segment_index = {cell_index + 1} を設定する")
    except KeyboardInterrupt:
        print(f"\n\n処理が中断されました")
        print(f"現在の位置: segment_id {segment_id}, segment_index {cell_index}")
        print(f"再開する場合は start_segment_index = {cell_index} を設定する")
        raise
    except Exception as e:
        skipped_count += 1
        print(f"\nError: segment_id {segment_id}, segment_index {cell_index}の処理中にエラーが発生しました: {e}")
        print(f"  次のセグメントから再開する場合は start_segment_index = {cell_index + 1} を設定する")
        import traceback
        traceback.print_exc()
        # エラーが発生しても続行するか、中断するか選択可能
        # 続行する場合は次のセグメントへ、中断する場合は以下のコメントを外す
        # raise

print("\n" + "="*80)
print("処理完了")
print(f"処理したセグメント数: {updated_count}")
print(f"スキップしたセグメント数: {skipped_count}")
if len(changed_segments) > 0:
    print(f"\n変更があったセグメント数: {len(changed_segments)}")
    print("変更内容:")
    for seg_id, cell_idx, old_val, new_val in changed_segments:
        print(f"  segment_id {seg_id:3d} (index {cell_idx:3d}): {old_val} -> {new_val}")
else:
    print("\n変更があったセグメントはなかった")
print("="*80)
