#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_run_sin_trend_fitting.pyを実行するバッチスクリプト
"""

########################################################
# ===== USER SETTINGS =====
# 処理対象のdirectory_idを指定（個別解析にも対応）
# - None: すべてのdirectory_idを処理
# - 文字列: 単一のdirectory_idのみ処理（例: "IXI052-HH-1343"）
# - リスト: 指定したdirectory_idのみ処理（例: ["IXI052-HH-1343", "IXI048-HH-1326"]）
target_directory_ids = None  # 例: "IXI052-HH-1343" または ["IXI052-HH-1343", "IXI048-HH-1326"] または None

# 開始するdirectory_id（エラー時に途中から再開する場合に設定）
# Noneの場合は最初から開始、特定の値が設定されている場合はその値以降から開始
# 個別解析の場合はNoneにする
start_directory_id = None  # 例: "IXI034-HH-1260" または None
########################################################

import pandas as pd
import os
import sys
import subprocess
import re
import tempfile

# プロジェクトルートの取得
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CSVファイルのパス
csv_path = os.path.join(project_root, 'results', 'participants', 'IXI_HH_participants.csv')

# データの読み込み
df = pd.read_csv(csv_path)

# directory_id列を抽出
all_directory_ids = df['directory_id'].dropna().unique()

# 処理対象のdirectory_idを決定
if target_directory_ids is None:
    directory_ids = sorted(all_directory_ids.tolist())
else:
    # target_directory_idsが文字列の場合はリストに変換
    if isinstance(target_directory_ids, str):
        target_directory_ids = [target_directory_ids]
    # target_directory_idsに含まれるもののみ処理
    directory_ids = [did for did in sorted(all_directory_ids.tolist()) if did in target_directory_ids]

print(f"抽出されたdirectory_id数: {len(all_directory_ids)}")
print(f"処理対象のdirectory_id数: {len(directory_ids)}")
print(f"directory_id一覧: {directory_ids}")

# 開始するdirectory_idを設定
if start_directory_id is not None:
    # start_directory_id以降のdirectory_idのみ処理
    try:
        start_index = directory_ids.index(start_directory_id)
        directory_ids = directory_ids[start_index:]
        print(f"開始するdirectory_id: {start_directory_id} (index: {start_index})")
        print(f"処理対象のdirectory_id数（開始位置以降）: {len(directory_ids)}")
    except ValueError:
        print(f"警告: start_directory_id '{start_directory_id}' が処理対象リストに見つからない")
        print(f"  処理対象リスト: {directory_ids}")
        sys.exit(1)
else:
    print(f"開始するdirectory_id: 最初から")

# 01_run_sin_trend_fitting.pyのパス
script_dir = os.path.dirname(os.path.abspath(__file__))
run_sin_trend_fitting_path = os.path.join(script_dir, '01_run_sin_trend_fitting.py')

# 各directory_idに対して処理を実行
results = []
failed_cases = []

for idx, directory_id in enumerate(directory_ids, 1):
    print(f"\n{'='*80}")
    print(f"処理中 ({idx}/{len(directory_ids)}): {directory_id}")
    print(f"{'='*80}")
    
    try:
        # スクリプトファイルを読み込んで、directory_idを置換
        with open(run_sin_trend_fitting_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # directory_id = "..."の行を探して置換
        pattern = r'directory_id\s*=\s*"[^"]*"'
        replacement = f'directory_id = "{directory_id}"'
        modified_script = re.sub(pattern, replacement, script_content)
        
        # 一時ファイルを元のスクリプトと同じディレクトリに作成（パス解決のため）
        tmp_script_path = os.path.join(script_dir, f"tmp_run_sin_trend_fitting_{directory_id.replace('-', '_')}.py")
        with open(tmp_script_path, 'w', encoding='utf-8') as tmp_file:
            tmp_file.write(modified_script)
        
        try:
            # サブプロセスとして実行
            result = subprocess.run(
                [sys.executable, tmp_script_path],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1時間のタイムアウト
            )
            
            # 標準出力と標準エラーを表示
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            
            if result.returncode == 0:
                results.append({"directory_id": directory_id, "status": "success"})
                print(f"✓ 完了: {directory_id}")
            else:
                failed_cases.append({
                    "directory_id": directory_id,
                    "reason": f"スクリプトがエラーコード {result.returncode} で終了した"
                })
                print(f"✗ エラー: {directory_id} (エラーコード: {result.returncode})")
                
        finally:
            # 一時ファイルを削除
            if os.path.exists(tmp_script_path):
                os.remove(tmp_script_path)
        
    except subprocess.TimeoutExpired:
        failed_cases.append({
            "directory_id": directory_id,
            "reason": "タイムアウト（1時間を超えました）"
        })
        print(f"✗ タイムアウト: {directory_id}")
        
    except Exception as e:
        print(f"✗ エラー: {directory_id}")
        print(f"  エラー内容: {str(e)}")
        import traceback
        traceback.print_exc()
        failed_cases.append({"directory_id": directory_id, "reason": str(e)})

# 結果のサマリー
print(f"\n{'='*80}")
print("処理完了")
print(f"{'='*80}")
print(f"成功: {len(results)}件")
print(f"失敗: {len(failed_cases)}件")

if failed_cases:
    print("\n失敗したケース:")
    for fc in failed_cases:
        print(f"  - {fc['directory_id']}: {fc['reason']}")

if results:
    print(f"\n成功したケース:")
    for r in results:
        print(f"  - {r['directory_id']}")

print(f"\n各directory_idのCSVファイルは以下に保存された:")
print(f"  {os.path.join(project_root, 'results', 'morphometrics', 'sin_trend')}")
