#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
periodicityがCoreまたはSecondaryのsegmentに対して、MCA分類を設定し、CSVファイルに追加するスクリプト

02_create_long_format_csv.py)では、IXI_HH_sin_trend_morphometrics_long.csvを作成済み
03_prepare_core_inplane_dataset.py)では、このCSVファイルとIXI_HH_participants.csvをマージして、IXI_HH_sin_trend_analysis_dataset.csvを作成済み
ここ(04_visual_inspection_mca_segment_periodicity.py)では、このCSVファイルを使用して、periodicityがCoreのsegmentに対して、MCA分類を設定する


参照元データ:元の解析データセット
  - results/analysis/sin_trend/IXI_HH_sin_trend_analysis_dataset.csv

出力先:作業用CSVファイル（mode='add'の場合は元のCSVファイルをコピーして作成、mode='recheck'の場合は既存の作業用CSVファイルを上書き）
  - results/analysis/sin_trend_mca_classification/IXI_HH_sin_trend_analysis_dataset.csv: 
"""

########################################################
# ===== USER SETTINGS =====
# 処理するdirectory_idのリスト（Noneの場合は全て）
target_directory_ids = None # 例: ["IXI000-HH-0000"] または None

# 使用するperiodicity列の選択（必須）
# 明示的に指定: 'periodicity', 'periodicity_coverage_1.5', 'periodicity_coverage_1.0' など
periodicity_column = 'periodicity_coverage_1.0'  # 必須: 'periodicity' または 'periodicity_coverage_1.5' または 'periodicity_coverage_1.0' など

# 対象とするperiodicityとplane_class
# 固定: periodicity ∈ {"Core","Secondary"}, plane_class ∈ {"In-plane","Out-of-plane"}
periodicity_filter = ["Core"] # ["Core", "Secondary"]
plane_class_filter = ["In-plane", "Out-of-plane"]

# 実行モード
# - "add": mca_segment が未設定（NaN/None）のものだけを対象に「追加」する（必ず入力）
# - "recheck": 既存の mca_segment を目視で再確認し、必要なら修正する（spaceで変更なし可）
mode = "add"  # "add" または "recheck"

# recheck開始位置（mode='recheck'のときのみ有効。指定したdirectory_id・segment_idから処理を開始）
recheck_start_directory_id = None  # 例: "IXI052-HH-1343"
recheck_start_segment_id = None  # 例: 5（segment_idの値）

# ===== Visualization settings =====
plotter_window_size = (2560, 1440)  # ウィンドウサイズ (width, height)
plotter_position = (500, 500)  # ウィンドウの位置 (x, y)
########################################################
print("="*80)
print(f"periodicityが{periodicity_filter} かつ plane_classが{plane_class_filter}のsegmentのmca_segmentを視覚的に確認・設定")
print(f"Coreのcoverage_hat基準: {periodicity_column}")
print("="*80)
print(f"mode: {mode}")
print(f"target_directory_ids: {target_directory_ids}")
if mode == "recheck" and (recheck_start_directory_id is not None or recheck_start_segment_id is not None):
    print(f"recheck開始位置: directory_id={recheck_start_directory_id}, segment_id={recheck_start_segment_id}")

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
import shutil

# 元のCSVファイルのパス
source_csv_path = os.path.join(project_root, 'results', 'analysis', 'sin_trend', 'IXI_HH_sin_trend_analysis_dataset.csv')

if not os.path.exists(source_csv_path):
    print(f"Error: 元のCSVファイルが見つからない: {source_csv_path}")
    sys.exit(1)

# sin_trend_mca_classification用の作業ディレクトリを作成
mca_classification_dir = os.path.join(project_root, 'results', 'analysis', 'sin_trend_mca_classification')
os.makedirs(mca_classification_dir, exist_ok=True)

# 作業用CSVファイルのパス
csv_path = os.path.join(mca_classification_dir, 'IXI_HH_sin_trend_analysis_dataset.csv')

# modeに応じた処理
if mode == "recheck":
    # recheckモード: 既存の作業用CSVファイルをそのまま使用（コピーしない）
    if not os.path.exists(csv_path):
        print(f"Error: 作業用CSVファイルが見つからない: {csv_path}")
        print(f"  recheckモードでは既存の作業用CSVファイルが必要")
        print(f"  まずmode='add'で実行するか、手動でファイルをコピーするように")
        sys.exit(1)
    print(f"\n作業用CSVファイルを使用（recheckモード）: {csv_path}")
else:
    # addモード: 作業用CSVファイルが存在しない場合、元のCSVファイルをコピー
    if not os.path.exists(csv_path):
        print(f"\n作業用CSVファイルを作成中: {csv_path}")
        shutil.copy2(source_csv_path, csv_path)
        print(f"  元のCSVファイルをコピーした: {source_csv_path}")
    else:
        print(f"\n作業用CSVファイルを使用: {csv_path}")
        print(f"  （既存のファイルを使用する。新しくコピーする場合は手動で削除するように）")

# CSVファイルを読み込む
print(f"\nCSVファイルを読み込み中: {csv_path}")
df_all = pd.read_csv(csv_path)
print(f"総レコード数: {len(df_all)}")

# periodicity列を選択（USER SETTINGSで指定された列を使用）
if periodicity_column is None:
    print("Error: periodicity_columnが指定されていない。")
    print("  periodicity_columnを明示的に指定する必要がある。")
    print("  例: 'periodicity', 'periodicity_coverage_1.5', 'periodicity_coverage_1.0'")
    sys.exit(1)

if periodicity_column not in df_all.columns:
    print(f"Error: 指定された'{periodicity_column}'列が見つからない。")
    # 利用可能な列を表示
    periodicity_coverage_cols = [col for col in df_all.columns if col.startswith('periodicity_coverage_')]
    if 'periodicity' in df_all.columns:
        periodicity_coverage_cols.insert(0, 'periodicity')
    print(f"  利用可能なperiodicity列: {periodicity_coverage_cols}")
    sys.exit(1)

periodicity_col = periodicity_column
print(f"  指定された'{periodicity_col}'列を使用する")

# periodicityとplane_classで対象をフィルタリング
mask_periodicity = df_all[periodicity_col].isin(periodicity_filter)

if 'plane_class' in df_all.columns:
    mask_plane = df_all['plane_class'].isin(plane_class_filter)
    df_filtered = df_all[mask_periodicity & mask_plane].copy()
    print(f"periodicity ∈ {periodicity_filter} かつ plane_class ∈ {plane_class_filter} のレコード数: {len(df_filtered)}")
else:
    df_filtered = df_all[mask_periodicity].copy()
    print("Warning: 'plane_class'列が見つからないので、periodicityだけでフィルタリングする")
    print(f"periodicity ∈ {periodicity_filter} のレコード数: {len(df_filtered)}")

core_count = len(df_filtered[df_filtered[periodicity_col] == 'Core'])
secondary_count = len(df_filtered[df_filtered[periodicity_col] == 'Secondary'])
print(f"  Core: {core_count}")
print(f"  Secondary: {secondary_count}")

# directory_idのリストを取得（recheck開始位置以降の件数計算で使用するためここで取得）
if target_directory_ids is None:
    directory_ids = sorted(df_filtered['directory_id'].dropna().unique().tolist())
else:
    if isinstance(target_directory_ids, str):
        target_directory_ids = [target_directory_ids]
    directory_ids = [did for did in sorted(df_filtered['directory_id'].dropna().unique().tolist())
                     if did in target_directory_ids]

# modeに応じた解析対象レコード数（target_directory_ids 指定時はその directory 内のみを対象とする）
df_target = df_filtered[df_filtered['directory_id'].isin(directory_ids)]  # 実際に処理する directory に限定
if mode == "add":
    n_to_analyze = df_target['mca_segment'].isna().sum()
    print(f"  mode='add': 解析対象レコード数（mca_segment未設定）: {n_to_analyze}")
else:
    # recheck: 開始位置指定時はその directory_id / segment_id 以降のみを対象とする
    if recheck_start_directory_id is None and recheck_start_segment_id is None:
        n_to_analyze = len(df_target)
        print(f"  mode='recheck': 解析対象レコード数: {n_to_analyze}")
    else:
        n_to_analyze = 0
        _started = False
        for _did in directory_ids:
            if recheck_start_directory_id is not None and recheck_start_directory_id in directory_ids and directory_ids.index(_did) < directory_ids.index(recheck_start_directory_id):
                continue
            _df_dir = df_filtered[df_filtered['directory_id'] == _did]
            _seg_list = _df_dir.sort_values(['segment_id', 'segment_index'])
            for _, _row in _seg_list.iterrows():
                _seg_id = _row['segment_id']
                if not _started:
                    if recheck_start_directory_id is not None and _did != recheck_start_directory_id:
                        _started = True
                    elif recheck_start_directory_id is not None and _did == recheck_start_directory_id and recheck_start_segment_id is not None and _seg_id != recheck_start_segment_id:
                        continue
                    elif recheck_start_directory_id is not None and _did == recheck_start_directory_id and recheck_start_segment_id is not None and _seg_id == recheck_start_segment_id:
                        _started = True
                    elif recheck_start_directory_id is None and recheck_start_segment_id is not None and _seg_id != recheck_start_segment_id:
                        continue
                    else:
                        _started = True
                if _started:
                    n_to_analyze += 1
        print(f"  mode='recheck': 解析対象レコード数（開始位置以降）: {n_to_analyze}")

if len(df_filtered) == 0:
    print("Error: 指定したperiodicity/plane_classに一致するレコードがない。")
    sys.exit(1)

print(f"\n処理対象のdirectory_id数: {len(directory_ids)}")
if len(directory_ids) > 0:
    print(f"directory_idリスト: {directory_ids[:10]}{'...' if len(directory_ids) > 10 else ''}")

# periodicity_colをグローバル変数として定義（関数内で使用するため）
def visualize_and_update_mca_segment(segment_id, row, df_all, directory_id, csv_path, mode):
    """
    セグメントのMCA分類を視覚的に確認・修正する
    
    Args:
        segment_id: セグメントID
        row: DataFrameの行（セグメントのデータ）
        df_all: DataFrame
        directory_id: ディレクトリID
        csv_path: CSVファイルのパス
        mode: "add" または "recheck"
    
    Returns:
        (success, changed): successは処理が成功したか、changedは変更があったか
    """
    global periodicity_col  # グローバル変数を使用
    # セグメントのデータを取得
    cell_index = row['segment_index']
    # CSVから読み込むとNoneはNaNになるため、pd.isna()でチェック
    mca_segment_raw = row.get('mca_segment', None)
    if pd.isna(mca_segment_raw):
        current_mca_segment = None
    else:
        current_mca_segment = mca_segment_raw
    
    print("\n" + "="*80)
    print(f"Processing directory_id: {directory_id}, segment_id: {segment_id}, segment_index: {cell_index}")
    print(f"periodicity: {row.get(periodicity_col, 'N/A')}")
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
    # recheck: 現在値を初期値にする
    # add: 未設定を前提に None を初期値にする（必須入力）
    if mode == "recheck":
        mca_segment = current_mca_segment
    else:
        mca_segment = None
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
        """変更なしで閉じる（現在値を維持）"""
        nonlocal mca_segment, pl
        mca_segment = current_mca_segment
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
    pl.add_text("Press key: M1 = 1, M2-3 = 2 or 3, M4 = 4",
                position='lower_left', font_size=18, color='red')
    pl.add_text(f"Current: {current_mca_segment}", position='upper_right', font_size=18, color='blue')
    pl.add_text(f"Periodicity: {row.get(periodicity_col, 'N/A')}", position='upper_left', font_size=14, color='green')
    
    # 左側のサブプロット（拡大表示）
    pl.subplot(0,0)
    pl.add_mesh(vessel_mesh, color='lightblue', opacity=0.5)
    pl.add_mesh(m1_mesh, color='blue', opacity=0.3)
    pl.add_mesh(brain_mesh, color='ivory', opacity=0)
    pl.add_points(P_vox_3darr, color='red', point_size=5)
    pl.add_points(mid_pt, color='red', opacity=0.8, point_size=30, render_points_as_spheres=False, ambient=1.0)
    pl.add_mesh(cropped_mesh, color='ivory', opacity=0.8)
    pl.camera_position = 'xy'
    pl.camera.azimuth = 180
    pl.set_focus(focus)
    pl.camera.zoom(3)
    pl.show_bounds(grid='front', location='outer', all_edges=True, ticks='outside')
    
    # 右側のサブプロット（全体表示）
    pl.subplot(0,1)
    pl.add_mesh(vessel_mesh, color='lightblue', opacity=0.5)
    pl.add_mesh(m1_mesh, color='blue', opacity=0.3)
    pl.add_mesh(brain_mesh, color='ivory', opacity=1)
    pl.add_mesh(cropped_mesh, color="lightgreen", opacity=1)
    pl.add_points(P_vox_3darr, color='red', point_size=5)
    pl.add_points(mid_pt, color='red', opacity=0.8, point_size=30, render_points_as_spheres=False, ambient=1.0)
    # 左右に応じてカメラ位置を設定
    pl.camera_position = 'zy'
    if side == "Lt":
        # Ltの場合：yzでxの大きい方から見る
        pl.camera.azimuth = 0
    else:  # Rtの場合
        # Rtの場合：yzでxの小さい方から見る
        pl.camera.azimuth = 180
    pl.camera.zoom(2.0)
    pl.set_focus(focus)
    pl.show_bounds(grid='front', location='outer', all_edges=True, ticks='outside')
    pl.add_title(f"MCA Classification Check, {directory_id}, segment ID: {segment_id}, segment index: {cell_index}", font_size=18)
    
    # キーボードイベントを追加（render()の前に登録）
    # メインキーボード
    pl.add_key_event('1', set_m1)
    pl.add_key_event('2', set_m2_3)
    pl.add_key_event('3', set_m2_3)  # 3もM2-3に設定
    pl.add_key_event('4', set_m4)
    # テンキー（同じコールバックを登録）
    pl.add_key_event('KP_1', set_m1)
    pl.add_key_event('KP_2', set_m2_3)
    pl.add_key_event('KP_3', set_m2_3)
    pl.add_key_event('KP_4', set_m4)
    if mode == "recheck":
        pl.add_key_event('space', set_no_change)  # recheckのみ、spaceで変更なし
    
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
        # add: 必須入力
        # recheck: current_mca_segment が None の場合のみ必須入力
        if mode == "add" or current_mca_segment is None:
            print("Error: mca_segmentが未設定。M1(1)、M2-3(2/3)、M4(4)のいずれかを選択する。")
            return False, False  # 失敗
        # recheckで既に値がある場合は安全側に倒して「変更なし」
        mca_segment = current_mca_segment
    
    # MCA分類が追加/変更されたかチェック
    # current_mca_segmentがNoneの場合は「追加」、それ以外は「変更」
    if mca_segment != current_mca_segment:
        if current_mca_segment is None:
            print(f"MCA分類を追加: {current_mca_segment} -> {mca_segment}")
        else:
            print(f"MCA分類が変更された: {current_mca_segment} -> {mca_segment}")
        
        # CSVファイルを更新/追加
        mask = (df_all['directory_id'] == directory_id) & (df_all['segment_id'] == segment_id) & (df_all['segment_index'] == cell_index)
        if mask.any():
            df_all.loc[mask, 'mca_segment'] = mca_segment
            # 変更があったら即座にCSVファイルに保存
            df_all.to_csv(csv_path, index=False)
            if current_mca_segment is None:
                print(f"  CSVファイルにmca_segmentを追加して保存した")
            else:
                print(f"  CSVファイルを更新して保存した")
        else:
            print(f"Error: セグメントが見つからない: {directory_id}, {segment_id}, {cell_index}")
            return False, False
        
        return True, True  # (success, changed)
    else:
        print(f"  MCA分類は変更されなかった: {mca_segment}")
        return True, False  # (success, not_changed)

# 各directory_idを処理
print("\n" + "="*80)
print("mca_segmentの確認/設定を開始")
print("="*80)
print("キー操作:")
print("  1 → M1")
print("  2 or 3 → M2-3")
print("  4 → M4")
if mode == "recheck":
    print("  spaceキー → 変更なし（現在の値を維持）")
else:
    print("  注意: mode='add' では 1/2/3/4 を入力する（spaceは無効）。")
print("="*80)

total_updated_count = 0
total_skipped_count = 0
all_changed_segments = []  # 変更があったセグメントのリスト
recheck_started = False  # recheck開始位置に到達したか（mode='recheck'で開始位置指定時のみ使用）

for directory_id in directory_ids:
    # recheck開始位置指定時：指定より前のdirectoryはスキップ
    if mode == "recheck" and recheck_start_directory_id is not None and recheck_start_directory_id in directory_ids:
        if directory_ids.index(directory_id) < directory_ids.index(recheck_start_directory_id):
            continue

    print("\n" + "="*80)
    print(f"Processing directory_id: {directory_id}")
    print("="*80)
    
    # このdirectory_idのデータをフィルタリング
    df_dir = df_filtered[df_filtered['directory_id'] == directory_id].copy()
    
    if len(df_dir) == 0:
        print(f"  Warning: {directory_id} に{periodicity_col}がCoreまたはSecondaryのレコードがない。スキップする")
        continue
    
    # 処理対象のセグメントIDを決定（全て）
    segment_ids = sorted(df_dir['segment_id'].unique())
    
    print(f"  処理対象のsegment_id数: {len(segment_ids)}")
    if len(segment_ids) > 0:
        print(f"  segment_idリスト: {segment_ids[:10]}{'...' if len(segment_ids) > 10 else ''}")
    
    # Centerlineデータを読み込む
    vessel_centerlines_dir = os.path.join(project_root, 'data', 'derived', 'vessel_centerlines', directory_id)
    merged_splitted_centerlines_path = os.path.join(vessel_centerlines_dir, f"{directory_id}_merged_centerlines.vtp")
    
    if not os.path.exists(merged_splitted_centerlines_path):
        print(f"  Error: Centerline file not found: {merged_splitted_centerlines_path}")
        total_skipped_count += len(df_dir)
        continue
    
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
    m1_mesh_path = os.path.join(vessel_mesh_dir, f"{directory_id}_M1_vessel_mesh.ply")
    
    if not os.path.exists(vessel_mesh_path) or not os.path.exists(m1_mesh_path):
        print(f"  Error: Mesh files not found for {directory_id}")
        total_skipped_count += len(df_dir)
        continue
    
    vessel_mesh = mesh_generation.load_ply_to_polydata(vessel_mesh_path)
    m1_mesh = mesh_generation.load_ply_to_polydata(m1_mesh_path)
    
    brain_mesh_dir = os.path.join(project_root, 'data', 'preprocessed', 'T1_preprocessed', 'brain_meshes', directory_id)
    
    # segment_idとsegment_indexの組み合わせで処理
    all_segments = []
    for segment_id in segment_ids:
        segment_rows = df_dir[df_dir['segment_id'] == segment_id]
        for _, row in segment_rows.iterrows():
            cell_index = row['segment_index']
            # mode='add' の場合、mca_segment未設定（NaN/None）のものだけを対象
            if mode == "add":
                mca_raw = row.get('mca_segment', None)
                if not pd.isna(mca_raw):
                    continue
            all_segments.append((segment_id, cell_index, row))
    
    # segment_indexでソート
    all_segments.sort(key=lambda x: (x[0], x[1]))  # segment_id, segment_indexの順でソート
    
    # 処理対象のセグメント（全て）
    filtered_segments = all_segments
    
    print(f"  処理対象のセグメント数: {len(filtered_segments)}")
    
    updated_count = 0
    skipped_count = 0
    
    for segment_id, cell_index, row in filtered_segments:
        # recheck開始位置指定時：指定の(directory_id, segment_id)に到達するまでスキップ
        if mode == "recheck" and (recheck_start_directory_id is not None or recheck_start_segment_id is not None) and not recheck_started:
            if recheck_start_directory_id is not None and directory_id != recheck_start_directory_id:
                recheck_started = True  # 開始directoryを過ぎたので以降は全て処理
            elif recheck_start_directory_id is not None and directory_id == recheck_start_directory_id and recheck_start_segment_id is not None:
                if segment_id != recheck_start_segment_id:
                    continue  # 開始segment_idに到達するまでスキップ
                recheck_started = True  # このセグメントから処理開始
            elif recheck_start_directory_id is None and recheck_start_segment_id is not None and segment_id != recheck_start_segment_id:
                continue
            else:
                recheck_started = True

        # 残りレコード数（全体）
        remaining = n_to_analyze - total_updated_count - total_skipped_count
        print(f"  残りレコード数: {remaining} / {n_to_analyze}")

        # 変更前の値を保存（CSVから読み込むとNoneはNaNになるため、pd.isna()でチェック）
        old_mca_segment_raw = row.get('mca_segment', None)
        if pd.isna(old_mca_segment_raw):
            old_mca_segment = None
        else:
            old_mca_segment = old_mca_segment_raw
        
        try:
            result = visualize_and_update_mca_segment(segment_id, row, df_all, directory_id, csv_path, mode)
            if isinstance(result, tuple):
                success, changed = result
                if success:
                    updated_count += 1
                    total_updated_count += 1
                    if changed:
                        # 変更後の値をCSVから読み込む（NaNの場合はNoneに変換）
                        mask = (df_all['directory_id'] == directory_id) & (df_all['segment_id'] == segment_id) & (df_all['segment_index'] == cell_index)
                        if mask.any():
                            new_mca_segment_raw = df_all.loc[mask, 'mca_segment'].iloc[0]
                            if pd.isna(new_mca_segment_raw):
                                new_mca_segment = None
                            else:
                                new_mca_segment = new_mca_segment_raw
                        else:
                            new_mca_segment = "Unknown"
                        all_changed_segments.append((directory_id, segment_id, cell_index, old_mca_segment, new_mca_segment))
                else:
                    skipped_count += 1
                    total_skipped_count += 1
                    print(f"  Warning: {directory_id}, segment_id {segment_id}, segment_index {cell_index}の処理が失敗した。")
            else:
                if result:
                    updated_count += 1
                    total_updated_count += 1
                else:
                    skipped_count += 1
                    total_skipped_count += 1
                    print(f"  Warning: {directory_id}, segment_id {segment_id}, segment_index {cell_index}の処理が失敗した。")
        except KeyboardInterrupt:
            print(f"\n\n処理が中断した。")
            print(f"現在の位置: {directory_id}, segment_id {segment_id}, segment_index {cell_index}")
            raise
        except Exception as e:
            skipped_count += 1
            total_skipped_count += 1
            print(f"\n  Error: {directory_id}, segment_id {segment_id}, segment_index {cell_index}の処理中にエラーが発生した: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n  {directory_id}: 処理したセグメント数: {updated_count}, スキップしたセグメント数: {skipped_count}")

print("\n" + "="*80)
print("処理完了")
print(f"処理したセグメント数: {total_updated_count}")
print(f"スキップしたセグメント数: {total_skipped_count}")
if len(all_changed_segments) > 0:
    print(f"\n変更があったセグメント数: {len(all_changed_segments)}")
    print("変更内容:")
    for dir_id, seg_id, cell_idx, old_val, new_val in all_changed_segments:
        print(f"  {dir_id}, segment_id {seg_id:3d} (index {cell_idx:3d}): {old_val} -> {new_val}")
else:
    print("\n変更があったセグメントはなかった")
print(f"\n作業用CSVファイルの保存場所: {csv_path}")
print(f"元のCSVファイル: {source_csv_path}")
print("="*80)
