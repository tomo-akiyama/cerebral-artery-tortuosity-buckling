#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全参加者の統合CSVファイルを1つのlong-format CSVに統合するスクリプト

ここ(02_create_long_format_csv.py)では、IXI_HH_sin_trend_morphometrics_long.csvを作成する
=> 03_prepare_core_inplane_dataset.pyでは、このCSVファイルとIXI_HH_participants.csvをマージして、IXI_HH_sin_trend_analysis_dataset.csvを作成する
=> 04_visual_inspection_mca_segment_periodicity.pyでは、このCSVファイルを使用して、periodicityがCoreのsegmentに対して、MCA分類を設定する
"""

import os
import sys
import glob
import pandas as pd

# プロジェクトルートの取得
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CSVファイルのパス
participants_csv_path = os.path.join(project_root, 'results', 'participants', 'IXI_HH_participants.csv')
output_dir_per_case = os.path.join(project_root, 'results', 'morphometrics', 'sin_trend', 'per_case')
output_dir_combined = os.path.join(project_root, 'results', 'morphometrics', 'sin_trend')
output_csv_path = os.path.join(output_dir_combined, 'IXI_HH_sin_trend_morphometrics_long.csv')

# 参加者情報を読み込む
if not os.path.exists(participants_csv_path):
    print(f"Error: 参加者情報CSVファイルが見つからない: {participants_csv_path}")
    sys.exit(1)

df_participants = pd.read_csv(participants_csv_path)
directory_ids = df_participants['directory_id'].dropna().unique()

print("="*80)
print("全参加者の統合CSVファイルをlong-format CSVに統合")
print("="*80)
print(f"抽出されたdirectory_id数: {len(directory_ids)}")
print(f"directory_id一覧: {sorted(directory_ids.tolist())}")
print()

# 各参加者の統合CSVファイルを読み込む
all_dataframes = []
successful_cases = []
failed_cases = []

for directory_id in sorted(directory_ids):
    csv_path = os.path.join(output_dir_per_case, f"{directory_id}_vessel_segmentdata.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSVファイルが見つからない: {csv_path}")
        failed_cases.append({"directory_id": directory_id, "reason": "CSVファイルが見つからない"})
        continue
    
    try:
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            all_dataframes.append(df)
            successful_cases.append(directory_id)
            print(f"✓ {directory_id}: {len(df)} レコード")
        else:
            print(f"Warning: {directory_id}: CSVファイルが空")
            failed_cases.append({"directory_id": directory_id, "reason": "CSVファイルが空"})
    except Exception as e:
        print(f"Error: {directory_id}: CSVファイルの読み込みに失敗: {e}")
        failed_cases.append({"directory_id": directory_id, "reason": str(e)})

if len(all_dataframes) == 0:
    print("Error: 読み込めるCSVファイルがない")
    sys.exit(1)

# すべてのDataFrameを統合
print("\n統合中...")
long_df = pd.concat(all_dataframes, ignore_index=True)

# 統合CSVファイルを保存
long_df.to_csv(output_csv_path, index=None)

print("\n" + "="*80)
print("処理完了")
print("="*80)
print(f"成功: {len(successful_cases)} 参加者")
print(f"失敗: {len(failed_cases)} 参加者")
print(f"総レコード数: {len(long_df)}")
print(f"出力ファイル: {output_csv_path}")

if failed_cases:
    print("\n失敗したケース:")
    for fc in failed_cases:
        print(f"  - {fc['directory_id']}: {fc['reason']}")

if successful_cases:
    print(f"\n成功したケース数: {len(successful_cases)}")
    print(f"成功したケース: {sorted(successful_cases)}")

print("="*80)
