#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析用データセット作成 + QC（分布・欠損・外れ値）

主解析に使う解析データセットを生成し、QC（分布・欠損・外れ値）を自動出力するスクリプト
02_create_long_format_csv.py)では、IXI_HH_sin_trend_morphometrics_long.csvを作成済み
ここ(03_prepare_core_inplane_dataset.py)では、このCSVファイルとIXI_HH_participants.csvをマージして、IXI_HH_sin_trend_analysis_dataset.csvを作成する
=> 04_visual_inspection_mca_segment_periodicity.pyでは、このCSVファイルを使用して、periodicityがCoreのsegmentに対して、MCA分類を設定する

出力ファイル:
  CSV (results/analysis/sin_trend/):
    - IXI_HH_sin_trend_analysis_dataset.csv: メインデータセット
    - qc_missingness_overall.csv: 全体の欠損率表
    - qc_counts_{cohort_name}_coverage_{value}.csv: コホート別N数集計（各coverage設定ごと）
    - qc_missingness_{cohort_name}_coverage_{value}.csv: コホート別欠損率（各coverage設定ごと）
    - qc_outliers_summary_{cohort_name}_coverage_{value}.csv: コホート別外れ値サマリー（各coverage設定ごと）
    - qc_outliers_rows_{cohort_name}_coverage_{value}.csv: コホート別外れ値行データ（各coverage設定ごと）
    - qc_cohort_comparison_counts_coverage_{value}.csv: コホート比較表
    - qc_comparison_stats_inplane_vs_outofplane_coverage_{value}.csv: 分布比較統計表
    - qc_logistic_regression_inplane_dependency_coverage_{value}.csv: ロジスティック回帰結果
  
  PNG (results/figures/sin_trend/):
    - hist_{var_name}_{cohort_name}_coverage_{value}.png: ヒストグラム（raw変数、各coverage設定ごと）
    - hist_log_{var_name}_{cohort_name}_coverage_{value}.png: ヒストグラム（log変数、各coverage設定ごと）
    - box_{var_name}_{cohort_name}_coverage_{value}.png: 箱ひげ図（raw変数、各coverage設定ごと）
    - box_log_{var_name}_{cohort_name}_coverage_{value}.png: 箱ひげ図（log変数、各coverage設定ごと）
    - comparison_hist_{var_name}_inplane_vs_outofplane_coverage_{value}.png: 比較用ヒストグラム
    - comparison_box_{var_name}_inplane_vs_outofplane_coverage_{value}.png: 比較用箱ひげ図

  {cohort_name}: core_all, core_inplane, core_outofplane, core_surface_evaluable
  {var_name}: radius, lambda, amplitude, length, A_over_L, lambda_over_r など
  {value}: COVERAGE_HAT_CORE_MIN_VALUES の値（例: 1.5, 1.0）
"""

import os
import sys
import pandas as pd
import numpy as np
import time
import matplotlib

########################################################
# ===== USER SETTINGS =====
SHOW_PLOTS = True
# True: 図をウィンドウで表示する（スペースキーで閉じる、1秒で自動で閉じる）
# False: 図を表示せず、ファイル保存のみ

# ===== 分類に使う閾値 =====
# Periodicity分類の閾値
MIN_LENGTH_MM = 10.0  # TooShort判定: L_trimmed_mm < min_length_mm
COVERAGE_HAT_BOUNDARY_THR = 1.0  # Undetectable判定: boundary_hit_low時、coverage_hat < 1.0
COVERAGE_HAT_UNDET_THR = 0.5  # Undetectable判定: coverage_hat < 0.5
AMP_HAT_MIN = 0.5  # Undetectable判定: amp_hat_mm < voxel_size (0.5mm)
SNR_THR = 1.5  # Undetectable判定: SNR < 1.5
PARTIAL_R2_THR = 0.061  # Undetectable判定: partial_R2 < partial_R2_thr (デフォルト値)
R2_THR = 0.3  # Undetectable判定: R2 < R2_thr (デフォルト値)
# Core判定の閾値（すべて満たす必要がある）
# 複数の設定で分類を実行する場合、COVERAGE_HAT_CORE_MIN_VALUESにリストで指定
COVERAGE_HAT_CORE_MIN_VALUES = [1.5, 1.0]  # Core判定: coverage_hat >= の値（複数設定可）
COVERAGE_HAT_CORE_MIN = COVERAGE_HAT_CORE_MIN_VALUES[0]  # デフォルト値（最初の値を使用）
PARTIAL_R2_CORE_MIN = 0.15  # Core判定: partial_R2 >= 0.15
R2_CORE_MIN = 0.6  # Core判定: R2 >= 0.6
HR_AMP_CORE_MAX = 0.2  # Core判定: hr_amp <= 0.2

# Surface constraint分類の閾値
CORTEX_COVERAGE_THR = 0.3  # Surface-NotEvaluable判定: cortex_coverage < 0.3
ENERGY_COVERAGE_THR = 0.5  # Surface-NotEvaluable判定: energy_coverage < energy_coverage_thresh (デフォルト値)
CONSTRAINT_COVERAGE_THR = 0.5  # Surface-NotEvaluable判定: constraint_coverage < constraint_coverage_thresh (デフォルト値)
R_OUT_THR = 0.30  # In-plane判定: R_out < R_out_thresh (デフォルト値)
########################################################

import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# プロジェクトルートの取得
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== 入力ファイルのパス =====
sin_trend_csv_path = os.path.join(project_root, 'results', 'morphometrics', 'sin_trend', 'IXI_HH_sin_trend_morphometrics_long.csv')
participants_csv_path = os.path.join(project_root, 'results', 'participants', 'IXI_HH_participants.csv')

# ===== 出力ディレクトリ =====
output_analysis_dir = os.path.join(project_root, 'results', 'analysis', 'sin_trend')
output_figures_dir = os.path.join(project_root, 'results', 'figures', 'sin_trend')

# 出力ディレクトリを作成
os.makedirs(output_analysis_dir, exist_ok=True)
os.makedirs(output_figures_dir, exist_ok=True)

print("="*80)
print("解析用データセット作成 + QC")
print("="*80)

# ===== 1) 入力ファイルの読み込み =====
print("\n【1. 入力ファイルの読み込み】")

# sin_trend morphometrics CSVの読み込み
if not os.path.exists(sin_trend_csv_path):
    print(f"Error: sin_trend morphometrics CSVファイルが見つからない: {sin_trend_csv_path}")
    sys.exit(1)

print(f"読み込み中: {sin_trend_csv_path}")
df = pd.read_csv(sin_trend_csv_path)
initial_row_count = len(df)
print(f"  読み込んだ行数: {initial_row_count}")
print(f"  列数: {len(df.columns)}")

# participants CSVの読み込み
if not os.path.exists(participants_csv_path):
    print(f"Error: participants CSVファイルが見つからない: {participants_csv_path}")
    sys.exit(1)

print(f"読み込み中: {participants_csv_path}")
df_participants = pd.read_csv(participants_csv_path)
print(f"  参加者数: {len(df_participants)}")

# ===== 2) 参加者の基本情報のマージ ===== 
# IXI_HH_participants.csv
print("\n【2. 参加者情報のマージ】")
if 'directory_id' in df.columns:
    # 既にdfに存在する列を除外してマージ（directory_idはキーなので除外）
    existing_cols = set(df.columns)
    participants_cols_to_merge = [col for col in df_participants.columns 
                                   if col != 'directory_id' and col not in existing_cols]
    
    if participants_cols_to_merge:
        print(f"  マージする列（重複を除外）: {participants_cols_to_merge}")
        
        # directory_idとマージ対象の列のみを選択
        merge_cols = ['directory_id'] + participants_cols_to_merge
        df = df.merge(
            df_participants[merge_cols],
            on='directory_id',
            how='left'
        )
        
        # マージ結果の確認
        merged_count = df['directory_id'].notna().sum()
        print(f"  マージ成功件数: {merged_count} / {len(df)} レコード")
        
        # 各列のマージ状況を確認
        for col in participants_cols_to_merge:
            if col in df.columns:
                merged_col_count = df[col].notna().sum()
                if merged_col_count < len(df):
                    print(f"    {col}: {merged_col_count} / {len(df)} レコードに情報あり")
                else:
                    print(f"    {col}: すべてのレコードに情報あり")
            else:
                print(f"    Warning: {col} がマージ後のデータフレームに存在しない")
    else:
        print(f"  マージする新しい列はない（すべて既存の列と重複）")
    
    # 重複していた列の情報
    duplicate_cols = [col for col in df_participants.columns 
                      if col != 'directory_id' and col in existing_cols]
    if duplicate_cols:
        print(f"  スキップした列（既に存在）: {duplicate_cols}")
else:
    print("  Error: 'directory_id'列が見つからない。参加者情報をマージできない")
    sys.exit(1)

# ===== 3) 前処理: boundary_hit_low/high の bool 化（堅牢版） =====
print("\n【3. 前処理: boundary_hit_low/high の bool 化】")

def robust_bool_convert(series):
    """
    文字列 "0"/"1"、数値 0/1、True/False いずれでも bool に変換
    "1","true","True",1 → True
    "0","false","False",0 → False
    欠損は NaN のまま
    """
    result = series.copy()
    
    # 文字列の場合
    if result.dtype == 'object':
        result = result.astype(str).str.strip().str.lower()
        # True に変換する値
        true_values = {'1', 'true', 't', 'yes', 'y'}
        # False に変換する値
        false_values = {'0', 'false', 'f', 'no', 'n', ''}
        
        result = result.replace({val: True for val in true_values})
        result = result.replace({val: False for val in false_values})
        # 上記に該当しないもの（'nan', 'none'など）は NaN のまま
        result = result.replace({'nan': np.nan, 'none': np.nan, 'null': np.nan})
    else:
        # 数値の場合
        result = result.astype('boolean')
    
    return result

for col in ['boundary_hit_low', 'boundary_hit_high']:
    if col in df.columns:
        df[col] = robust_bool_convert(df[col])
        print(f"  {col}: bool型に変換完了（欠損値: {df[col].isna().sum()}件）")
    else:
        print(f"  Warning: {col}列が見つからない")

# ===== 4) 分類ロジックの適用 =====
print("\n【4. 分類ロジックの適用】")

# 分類ロジック（01_run_sin_trend_fitting.pyのclassify_and_display_segmentと同じ）
CLASS_UNDET = 'Undetectable'
CLASS_CORE = 'Core'
CLASS_SEC = 'Secondary'
CLASS_EXCL = 'TooShort'

voxel_size = 0.5  # mm

def classify_row(row, voxel_size, coverage_hat_core_min=None):
    """
    1行のデータから判定を行う関数
    
    Args:
        row: DataFrameの行
        voxel_size: ボクセルサイズ
        coverage_hat_core_min: Core判定のcoverage_hat閾値（Noneの場合はUSER SETTINGSのデフォルト値を使用）
    
    Returns:
        periodicity: sin+trend判定結果（TooShort, Undetectable, Secondary, Core）
        category: 脳表拘束判定結果（Surface-NotEvaluable, In-plane Buckling, Out-of-plane, None）
    """
    # coverage_hat_core_minが指定されていない場合はデフォルト値を使用
    if coverage_hat_core_min is None:
        coverage_hat_core_min = COVERAGE_HAT_CORE_MIN
    # 1. まずsin+trend判定を行う
    L_trimmed_mm = row.get('L_trimmed_mm', np.nan)
    min_length_mm = row.get('min_length_mm', MIN_LENGTH_MM)  # USER SETTINGSの値を使用
    boundary_hit_low = row.get('boundary_hit_low', np.nan)
    boundary_hit_high = row.get('boundary_hit_high', np.nan)
    coverage_hat = row.get('coverage_hat', np.nan)
    amp_hat_mm = row.get('amp_hat_mm', np.nan)
    SNR = row.get('SNR', np.nan)
    partial_R2 = row.get('partial_R2', np.nan)
    R2 = row.get('R2', np.nan)
    hr_amp = row.get('hr_amp', np.nan)
    # CSVから閾値を取得できる場合はそれを使用、なければUSER SETTINGSのデフォルト値を使用
    partial_R2_thr = row.get('partial_R2_thr', PARTIAL_R2_THR)
    R2_thr = row.get('R2_thr', R2_THR)
    
    # TooShort判定
    if pd.notna(L_trimmed_mm) and pd.notna(min_length_mm) and L_trimmed_mm < min_length_mm:
        periodicity = CLASS_EXCL
        category = None
    # Undetectable判定
    elif pd.notna(boundary_hit_low) and boundary_hit_low and pd.notna(coverage_hat) and coverage_hat < COVERAGE_HAT_BOUNDARY_THR:
        periodicity = CLASS_UNDET
        category = None
    elif pd.notna(coverage_hat) and coverage_hat < COVERAGE_HAT_UNDET_THR:
        periodicity = CLASS_UNDET
        category = None
    elif pd.notna(amp_hat_mm) and amp_hat_mm < AMP_HAT_MIN:
        periodicity = CLASS_UNDET
        category = None
    elif pd.notna(SNR) and SNR < SNR_THR:
        periodicity = CLASS_UNDET
        category = None
    elif pd.notna(partial_R2) and pd.notna(partial_R2_thr) and partial_R2 < partial_R2_thr:
        periodicity = CLASS_UNDET
        category = None
    elif pd.notna(R2) and pd.notna(R2_thr) and R2 < R2_thr:
        periodicity = CLASS_UNDET
        category = None
    # Core判定
    elif (pd.notna(coverage_hat) and coverage_hat >= coverage_hat_core_min and 
          pd.notna(partial_R2) and partial_R2 >= PARTIAL_R2_CORE_MIN and 
          pd.notna(R2) and R2 >= R2_CORE_MIN and 
          pd.notna(hr_amp) and hr_amp <= HR_AMP_CORE_MAX):
        periodicity = CLASS_CORE
    # Secondary（残り）
    else:
        periodicity = CLASS_SEC
    
    # Coreになったが境界に張り付いているならSecondaryに落とす
    if periodicity == CLASS_CORE and pd.notna(boundary_hit_high) and boundary_hit_high:
        periodicity = CLASS_SEC
    
    # 2. Core/Secondaryの場合のみ、表面拘束判定を適用
    if periodicity in [CLASS_CORE, CLASS_SEC]:
        cortex_coverage = row.get('cortex_coverage', np.nan)
        energy_coverage = row.get('energy_coverage', np.nan)
        constraint_coverage = row.get('constraint_coverage', np.nan)
        R_out = row.get('R_out_surface', np.nan)
        # CSVから閾値を取得できる場合はそれを使用、なければUSER SETTINGSのデフォルト値を使用
        energy_coverage_thresh = row.get('energy_coverage_thresh', ENERGY_COVERAGE_THR)
        constraint_coverage_thresh = row.get('constraint_coverage_thresh', CONSTRAINT_COVERAGE_THR)
        R_out_thresh = row.get('R_out_thresh', R_OUT_THR)
        
        if pd.isna(cortex_coverage) or cortex_coverage < CORTEX_COVERAGE_THR:
            category = "Surface-NotEvaluable"
        elif pd.isna(energy_coverage) or energy_coverage < energy_coverage_thresh:
            category = "Surface-NotEvaluable"
        elif pd.isna(constraint_coverage) or constraint_coverage < constraint_coverage_thresh:
            category = "Surface-NotEvaluable"
        elif pd.notna(R_out) and R_out < R_out_thresh:
            category = "In-plane Buckling"
        else:
            category = "Out-of-plane"
    else:
        category = None
    
    return periodicity, category

# 各レコードに対して判定を実行
print("  分類を実行中")
print(f"  COVERAGE_HAT_CORE_MIN設定: {COVERAGE_HAT_CORE_MIN_VALUES}")

# すべての閾値設定で分類を実行
for coverage_min in COVERAGE_HAT_CORE_MIN_VALUES:
    periodicities = []
    categories = []
    for idx, row in df.iterrows():
        periodicity, category = classify_row(row, voxel_size, coverage_hat_core_min=coverage_min)
        periodicities.append(periodicity)
        categories.append(category)
    
    # 列名に閾値を含める（例: periodicity_coverage_1.5, periodicity_coverage_1.0）
    col_name_periodicity = f'periodicity_coverage_{coverage_min}'
    col_name_surface = f'surface_constraint_coverage_{coverage_min}'
    
    df[col_name_periodicity] = periodicities
    df[col_name_surface] = categories
    
    print(f"  分類完了（COVERAGE_HAT_CORE_MIN={coverage_min}）: {len(periodicities)} レコード")
    print(f"  Periodicity分布:")
    for periodicity in [CLASS_CORE, CLASS_SEC, CLASS_UNDET, CLASS_EXCL]:
        count = periodicities.count(periodicity)
        print(f"    {periodicity}: {count} ({count/len(periodicities)*100:.2f}%)")

# 後続のコードとの互換性のため、デフォルト設定（最初の値）の結果をperiodicityとsurface_constraintにも保存
default_coverage_min = COVERAGE_HAT_CORE_MIN_VALUES[0]
df['periodicity'] = df[f'periodicity_coverage_{default_coverage_min}']
df['surface_constraint'] = df[f'surface_constraint_coverage_{default_coverage_min}']
print(f"\n  デフォルト設定（COVERAGE_HAT_CORE_MIN={default_coverage_min}）の結果を'periodicity'と'surface_constraint'列にも保存しました。")

# ===== 5) 解析用の派生列の作成 =====
print("\n【5. 解析用の派生列の作成】")

# 5.1 Surface関連
print("  5.1 Surface関連の派生列")
df['surface_evaluable'] = df['surface_constraint'].isin(['In-plane Buckling', 'Out-of-plane'])
df['plane_class'] = df['surface_constraint'].map({
    'In-plane Buckling': 'In-plane',
    'Out-of-plane': 'Out-of-plane'
})
print(f"    surface_evaluable: {df['surface_evaluable'].sum()}件 (True)")

# 5.2 解析コホートフラグ（各設定ごとに作成）
print("  5.2 解析コホートフラグ")
for coverage_min in COVERAGE_HAT_CORE_MIN_VALUES:
    periodicity_col = f'periodicity_coverage_{coverage_min}'
    surface_constraint_col = f'surface_constraint_coverage_{coverage_min}'
    
    # 各設定ごとの派生列を作成
    df[f'core_all_coverage_{coverage_min}'] = (df[periodicity_col] == 'Core')
    df[f'core_inplane_coverage_{coverage_min}'] = (df[periodicity_col] == 'Core') & (df[surface_constraint_col] == 'In-plane Buckling')
    df[f'core_outofplane_coverage_{coverage_min}'] = (df[periodicity_col] == 'Core') & (df[surface_constraint_col] == 'Out-of-plane')
    df[f'surface_evaluable_coverage_{coverage_min}'] = df[surface_constraint_col].isin(['In-plane Buckling', 'Out-of-plane'])
    df[f'core_surface_evaluable_coverage_{coverage_min}'] = (df[periodicity_col] == 'Core') & (df[f'surface_evaluable_coverage_{coverage_min}'] == True)
    df[f'core_or_secondary_coverage_{coverage_min}'] = df[periodicity_col].isin(['Core', 'Secondary'])
    df[f'core_or_secondary_inplane_coverage_{coverage_min}'] = df[f'core_or_secondary_coverage_{coverage_min}'] & (df[surface_constraint_col] == 'In-plane Buckling')
    
    print(f"    COVERAGE_HAT_CORE_MIN={coverage_min}:")
    print(f"      core_all: {df[f'core_all_coverage_{coverage_min}'].sum()}件")
    print(f"      core_inplane: {df[f'core_inplane_coverage_{coverage_min}'].sum()}件")
    print(f"      core_outofplane: {df[f'core_outofplane_coverage_{coverage_min}'].sum()}件")
    print(f"      core_surface_evaluable: {df[f'core_surface_evaluable_coverage_{coverage_min}'].sum()}件")
    print(f"      core_or_secondary: {df[f'core_or_secondary_coverage_{coverage_min}'].sum()}件")
    print(f"      core_or_secondary_inplane: {df[f'core_or_secondary_inplane_coverage_{coverage_min}'].sum()}件")

# デフォルト設定（最初の値）の結果を後続コードとの互換性のためにも保存
default_coverage_min = COVERAGE_HAT_CORE_MIN_VALUES[0]
df['core_all'] = df[f'core_all_coverage_{default_coverage_min}']
df['core_inplane'] = df[f'core_inplane_coverage_{default_coverage_min}']
df['core_outofplane'] = df[f'core_outofplane_coverage_{default_coverage_min}']
df['core_surface_evaluable'] = df[f'core_surface_evaluable_coverage_{default_coverage_min}']
df['core_or_secondary'] = df[f'core_or_secondary_coverage_{default_coverage_min}']
df['core_or_secondary_inplane'] = df[f'core_or_secondary_inplane_coverage_{default_coverage_min}']
df['surface_evaluable'] = df[f'surface_evaluable_coverage_{default_coverage_min}']

# 5.3 主解析buckling検出フラグ（複数）
print("  5.3 主解析buckling検出フラグ")
df['buckling_detect_strict'] = df['core_inplane']
df['buckling_detect_relaxed'] = df['core_all']
df['buckling_detect_surface'] = df['core_surface_evaluable']

print(f"    buckling_detect_strict: {df['buckling_detect_strict'].sum()}件")
print(f"    buckling_detect_relaxed: {df['buckling_detect_relaxed'].sum()}件")
print(f"    buckling_detect_surface: {df['buckling_detect_surface'].sum()}件")

# ===== 6) 主要量の列名自動検出 =====
print("\n【6. 主要量の列名自動検出】")

def find_column(df, canonical_name, candidate_names):
    """
    候補名リストから存在する最初の列を検出
    
    Parameters:
    -----------
    df : pd.DataFrame
    canonical_name : str
        正規名（エラーメッセージ用）
    candidate_names : list
        候補列名のリスト
    
    Returns:
    --------
    str or None
        見つかった列名、見つからなければNone
    """
    for candidate in candidate_names:
        if candidate in df.columns:
            return candidate
    return None

column_map = {}
# 固定の列名を使用
column_candidates = {
    'radius': ['radius_median_mm'],
    'lambda': ['lambda_hat_mm'],
    'amplitude': ['amp_hat_mm'],
    'length': ['L_trimmed_mm']
}

missing_columns = []
for canonical_name, candidates in column_candidates.items():
    # 固定列名が存在するかチェック
    fixed_col = candidates[0]  # 最初（唯一）の候補が固定列名
    if fixed_col in df.columns:
        column_map[canonical_name] = fixed_col
        print(f"  {canonical_name}: '{fixed_col}' を採用（固定）")
    else:
        missing_columns.append(canonical_name)

if missing_columns:
    print(f"\n  Error: 以下の固定列名が見つからなかった: {missing_columns}")
    print(f"\n  期待される固定列名:")
    for canonical_name in missing_columns:
        print(f"    {canonical_name}: {column_candidates[canonical_name][0]}")
    print(f"\n  実データに存在する列一覧（先頭50件）:")
    for i, col in enumerate(sorted(df.columns)[:50]):
        print(f"    {i+1}. {col}")
    if len(df.columns) > 50:
        print(f"    ... (残り {len(df.columns) - 50} 件)")
    sys.exit(1)

# 採用された列名マップを表示
print(f"\n  採用された列名マップ:")
for canonical_name, actual_col in column_map.items():
    print(f"    {canonical_name}: {actual_col}")

# ===== 7) log変換用の列作成 =====
print("\n【7. log変換用の列作成】")

epsilon = 1e-9

# 主要量の列名を取得
radius_col = column_map['radius']
lambda_col = column_map['lambda']
amplitude_col = column_map['amplitude']
length_col = column_map['length']

# 0以下または欠損値のチェック
invalid_counts_summary = {}
for canonical_name, col_name in column_map.items():
    invalid_mask = (df[col_name] <= 0) | df[col_name].isna()
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        invalid_counts_summary[canonical_name] = invalid_count
        print(f"  Warning: {canonical_name} ({col_name}) に <=0 または欠損が {invalid_count}件ある")
        if invalid_count <= 20:
            id_cols_to_show = []
            for col in ['directory_id', 'segment_id', 'vessel_id']:
                if col in df.columns:
                    id_cols_to_show.append(col)
            if id_cols_to_show:
                invalid_ids = df.loc[invalid_mask, id_cols_to_show + [col_name]].to_dict('records')
                print(f"    該当ID（最大20件）:")
                for record in invalid_ids[:20]:
                    id_str = ", ".join([f"{k}={record[k]}" for k in id_cols_to_show])
                    print(f"      {id_str}, {col_name}={record[col_name]}")

if not invalid_counts_summary:
    print(f"  <=0値の検出件数: 0件（すべて正常）")
else:
    print(f"\n  <=0値の検出件数サマリー:")
    for canonical_name, count in invalid_counts_summary.items():
        print(f"    {canonical_name}: {count}件")

# 比率の計算
df['A_over_L'] = df[amplitude_col] / df[length_col]
df['lambda_over_r'] = df[lambda_col] / df[radius_col]

# log変換
df['log_r'] = np.log(df[radius_col] + epsilon)
df['log_lambda'] = np.log(df[lambda_col] + epsilon)
df['log_amplitude'] = np.log(df[amplitude_col] + epsilon)
df['log_length'] = np.log(df[length_col] + epsilon)
df['log_lambda_over_r'] = np.log(df['lambda_over_r'] + epsilon)
df['log_A_over_L'] = np.log(df['A_over_L'] + epsilon)

print(f"  log変換列を作成した")
print(f"    A_over_L: 欠損 {df['A_over_L'].isna().sum()}件")
print(f"    lambda_over_r: 欠損 {df['lambda_over_r'].isna().sum()}件")
print(f"    log_amplitude: 欠損 {df['log_amplitude'].isna().sum()}件")
print(f"    log_length: 欠損 {df['log_length'].isna().sum()}件")

# ===== 8) 年齢帯の作成 =====
print("\n【8. 年齢帯の作成】")
age_bins = [20, 30, 40, 50, 60, 70]  # 70+でまとめるため、80は削除
max_age = df['AGE'].max()
if pd.notna(max_age):
    # 70以上はすべて70+として扱う
    bins_with_max = age_bins + [max_age + 1]
    age_labels = []
    for i in range(len(age_bins)):
        if i == len(age_bins) - 1:
            age_labels.append(f"{age_bins[i]}+")
        else:
            age_labels.append(f"{age_bins[i]}-{age_bins[i+1]-1}")
    
    df['age_group'] = pd.cut(
        df['AGE'],
        bins=bins_with_max,
        labels=age_labels,
        right=False,
        include_lowest=True
    )
    print(f"  年齢帯を作成した: {df['age_group'].value_counts().to_dict()}")
else:
    print("  Warning: 年齢データがすべて欠測値のため、年齢帯を作成できない")
    df['age_group'] = None

# ===== 9) 解析データセットの保存 =====
print("\n【9. 解析データセットの保存】")
dataset_csv_path = os.path.join(output_analysis_dir, 'IXI_HH_sin_trend_analysis_dataset.csv')
df.to_csv(dataset_csv_path, index=False, encoding='utf-8')
print(f"  解析データセットを保存した: {dataset_csv_path}")

# ===== 10) QC出力 =====
print("\n【10. QC出力】")

def show_figure_with_space_to_close(fig, timeout=1.0):
    """
    図を表示し、スペースキーで閉じるか、1秒で自動で閉じる
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        表示する図
    timeout : float
        自動で閉じるまでの秒数（デフォルト: 1.0秒）
    """
    closed = [False]  # リストでラップして参照渡しにする
    start_time = time.time()
    
    def on_key(event):
        if event.key == ' ':
            closed[0] = True
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=False)
    # スペースキーが押されるか、タイムアウトするまで待機
    while not closed[0]:
        if fig.number not in plt.get_fignums():
            break
        elapsed_time = time.time() - start_time
        if elapsed_time >= timeout:
            plt.close(fig)
            break
        plt.pause(0.1)

def run_qc_for_cohort(df, cohort_name, mask, var_cols, out_dirs, id_cols, plot_vars_dict, coverage_min=None):
    """
    コホートごとのQC出力を実行する再利用可能な関数
    
    Parameters:
    -----------
    df : pd.DataFrame
        全体のデータフレーム
    cohort_name : str
        コホート名
    mask : pd.Series
        コホート抽出用のブールマスク
    var_cols : dict
        変数名と列名のマッピング（qc_vars用）
    out_dirs : dict
        出力ディレクトリ {'analysis': path, 'figures': path}
    id_cols : list
        ID列のリスト
    plot_vars_dict : dict
        プロット用変数辞書 {var_name: (raw_col, log_col)}
    coverage_min : float or None
        COVERAGE_HAT_CORE_MINの値（Noneの場合はファイル名に含めない）
    
    Returns:
    --------
    dict
        保存したファイルパスのリスト
    """
    cohort_df = df[mask].copy()
    saved_files = []
    
    if len(cohort_df) == 0:
        print(f"    {cohort_name}: データなし、スキップ")
        return saved_files
    
    # N集計表
    n_summary = []
    n_summary.append({
        'group': 'Overall',
        'subgroup': 'All',
        'N': len(cohort_df)
    })
    
    # 部位別N
    if 'mca_segment' in cohort_df.columns:
        segment_counts = cohort_df['mca_segment'].value_counts(dropna=False)
        for segment, count in segment_counts.items():
            n_summary.append({
                'group': 'BySegment',
                'subgroup': str(segment) if pd.notna(segment) else 'Missing',
                'N': count
            })
    else:
        print(f"    Warning: {cohort_name} - mca_segment列が存在しないため、部位別集計をスキップする")
    
    # 年齢帯別N
    if 'age_group' in cohort_df.columns and cohort_df['age_group'].notna().any():
        age_counts = cohort_df['age_group'].value_counts(dropna=False).sort_index()
        for age_group, count in age_counts.items():
            n_summary.append({
                'group': 'ByAgeGroup',
                'subgroup': str(age_group) if pd.notna(age_group) else 'Missing',
                'N': count
            })
    
    # 部位×年齢帯別N
    if 'mca_segment' in cohort_df.columns and 'age_group' in cohort_df.columns and cohort_df['age_group'].notna().any():
        cross_counts = pd.crosstab(cohort_df['mca_segment'], cohort_df['age_group'], margins=False)
        for segment in cross_counts.index:
            for age_group in cross_counts.columns:
                count = cross_counts.loc[segment, age_group]
                if count > 0:
                    n_summary.append({
                        'group': 'BySegmentAndAgeGroup',
                        'subgroup': f"{segment}_{age_group}",
                        'N': count
                    })
    
    n_summary_df = pd.DataFrame(n_summary)
    if coverage_min is not None:
        n_summary_path = os.path.join(out_dirs['analysis'], f'qc_counts_{cohort_name}_coverage_{coverage_min}.csv')
    else:
        n_summary_path = os.path.join(out_dirs['analysis'], f'qc_counts_{cohort_name}.csv')
    n_summary_df.to_csv(n_summary_path, index=False, encoding='utf-8')
    saved_files.append(n_summary_path)
    
    # 欠損率表
    missing_cohort = []
    for var in var_cols.values():  # var_colsはdictなので.values()を使用
        if var in cohort_df.columns:
            missing_count = cohort_df[var].isna().sum()
            missing_pct = (missing_count / len(cohort_df)) * 100 if len(cohort_df) > 0 else 0
            missing_cohort.append({
                'variable': var,
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            })
    
    missing_cohort_df = pd.DataFrame(missing_cohort)
    if coverage_min is not None:
        missing_cohort_path = os.path.join(out_dirs['analysis'], f'qc_missingness_{cohort_name}_coverage_{coverage_min}.csv')
    else:
        missing_cohort_path = os.path.join(out_dirs['analysis'], f'qc_missingness_{cohort_name}.csv')
    missing_cohort_df.to_csv(missing_cohort_path, index=False, encoding='utf-8')
    saved_files.append(missing_cohort_path)
    
    # 分布図
    for var_name, (raw_col, log_col) in plot_vars_dict.items():
        if raw_col not in cohort_df.columns or log_col not in cohort_df.columns:
            continue
        
        # ヒストグラム（raw）
        fig, ax = plt.subplots(figsize=(8, 6))
        valid_data = cohort_df[raw_col].dropna()
        if len(valid_data) > 0:
            ax.hist(valid_data, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel(var_name)
            ax.set_ylabel('Frequency')
            if coverage_min is not None:
                ax.set_title(f'Histogram: {var_name} ({cohort_name}, coverage={coverage_min}, N={len(valid_data)})')
            else:
                ax.set_title(f'Histogram: {var_name} ({cohort_name}, N={len(valid_data)})')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if coverage_min is not None:
            hist_path = os.path.join(out_dirs['figures'], f'hist_{var_name}_{cohort_name}_coverage_{coverage_min}.png')
        else:
            hist_path = os.path.join(out_dirs['figures'], f'hist_{var_name}_{cohort_name}.png')
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        if SHOW_PLOTS:
            print(f"      {var_name} (raw) ヒストグラムを表示中")
            show_figure_with_space_to_close(fig)
        else:
            plt.close(fig)
        saved_files.append(hist_path)
        
        # ヒストグラム（log）
        fig, ax = plt.subplots(figsize=(8, 6))
        valid_data = cohort_df[log_col].dropna()
        if len(valid_data) > 0:
            ax.hist(valid_data, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel(f'log({var_name})')
            ax.set_ylabel('Frequency')
            if coverage_min is not None:
                ax.set_title(f'Histogram: log({var_name}) ({cohort_name}, coverage={coverage_min}, N={len(valid_data)})')
            else:
                ax.set_title(f'Histogram: log({var_name}) ({cohort_name}, N={len(valid_data)})')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if coverage_min is not None:
            hist_path = os.path.join(out_dirs['figures'], f'hist_log_{var_name}_{cohort_name}_coverage_{coverage_min}.png')
        else:
            hist_path = os.path.join(out_dirs['figures'], f'hist_log_{var_name}_{cohort_name}.png')
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        if SHOW_PLOTS:
            print(f"      log({var_name}) ヒストグラムを表示中")
            show_figure_with_space_to_close(fig)
        else:
            plt.close(fig)
        saved_files.append(hist_path)
        
        # 箱ひげ（raw）
        fig, ax = plt.subplots(figsize=(6, 8))
        valid_data = cohort_df[raw_col].dropna()
        if len(valid_data) > 0:
            bp = ax.boxplot(valid_data, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            ax.set_ylabel(var_name)
            if coverage_min is not None:
                ax.set_title(f'Boxplot: {var_name} ({cohort_name}, coverage={coverage_min}, N={len(valid_data)})')
            else:
                ax.set_title(f'Boxplot: {var_name} ({cohort_name}, N={len(valid_data)})')
            ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        if coverage_min is not None:
            box_path = os.path.join(out_dirs['figures'], f'box_{var_name}_{cohort_name}_coverage_{coverage_min}.png')
        else:
            box_path = os.path.join(out_dirs['figures'], f'box_{var_name}_{cohort_name}.png')
        plt.savefig(box_path, dpi=300, bbox_inches='tight')
        if SHOW_PLOTS:
            print(f"      {var_name} (raw) 箱ひげ図を表示中")
            show_figure_with_space_to_close(fig)
        else:
            plt.close(fig)
        saved_files.append(box_path)
        
        # 箱ひげ（log）
        fig, ax = plt.subplots(figsize=(6, 8))
        valid_data = cohort_df[log_col].dropna()
        if len(valid_data) > 0:
            bp = ax.boxplot(valid_data, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            ax.set_ylabel(f'log({var_name})')
            if coverage_min is not None:
                ax.set_title(f'Boxplot: log({var_name}) ({cohort_name}, coverage={coverage_min}, N={len(valid_data)})')
            else:
                ax.set_title(f'Boxplot: log({var_name}) ({cohort_name}, N={len(valid_data)})')
            ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        if coverage_min is not None:
            box_path = os.path.join(out_dirs['figures'], f'box_log_{var_name}_{cohort_name}_coverage_{coverage_min}.png')
        else:
            box_path = os.path.join(out_dirs['figures'], f'box_log_{var_name}_{cohort_name}.png')
        plt.savefig(box_path, dpi=300, bbox_inches='tight')
        if SHOW_PLOTS:
            print(f"      log({var_name}) 箱ひげ図を表示中")
            show_figure_with_space_to_close(fig)
        else:
            plt.close(fig)
        saved_files.append(box_path)
    
    # 外れ値QC
    outlier_vars = []
    for key in ['radius', 'lambda', 'amplitude', 'length']:
        if key in var_cols:
            outlier_vars.append(var_cols[key])
    outlier_vars.extend(['A_over_L', 'lambda_over_r',
                        'log_r', 'log_lambda', 'log_amplitude', 'log_length', 
                        'log_lambda_over_r', 'log_A_over_L'])
    
    outlier_summary = []
    outlier_rows_list = []
    
    for var in outlier_vars:
        if var not in cohort_df.columns:
            continue
        
        valid_data = cohort_df[var].dropna()
        if len(valid_data) < 4:
            continue
        
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outlier_mask = (cohort_df[var] < lower_bound) | (cohort_df[var] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        outlier_summary.append({
            'variable': var,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': outlier_count,
            'outlier_percentage': (outlier_count / len(cohort_df)) * 100 if len(cohort_df) > 0 else 0
        })
        
        if outlier_count > 0:
            outlier_rows = cohort_df.loc[outlier_mask, id_cols + [var]].copy()
            outlier_rows = outlier_rows.rename(columns={var: 'value'})
            outlier_rows['variable'] = var
            cols = ['variable'] + id_cols + ['value']
            outlier_rows = outlier_rows[cols]
            outlier_rows_list.append(outlier_rows)
    
    if outlier_summary:
        outlier_summary_df = pd.DataFrame(outlier_summary)
        if coverage_min is not None:
            outlier_summary_path = os.path.join(out_dirs['analysis'], f'qc_outliers_summary_{cohort_name}_coverage_{coverage_min}.csv')
        else:
            outlier_summary_path = os.path.join(out_dirs['analysis'], f'qc_outliers_summary_{cohort_name}.csv')
        outlier_summary_df.to_csv(outlier_summary_path, index=False, encoding='utf-8')
        saved_files.append(outlier_summary_path)
    
    if outlier_rows_list:
        outlier_rows_df = pd.concat(outlier_rows_list, ignore_index=True)
        if coverage_min is not None:
            outlier_rows_path = os.path.join(out_dirs['analysis'], f'qc_outliers_rows_{cohort_name}_coverage_{coverage_min}.csv')
        else:
            outlier_rows_path = os.path.join(out_dirs['analysis'], f'qc_outliers_rows_{cohort_name}.csv')
        outlier_rows_df.to_csv(outlier_rows_path, index=False, encoding='utf-8')
        saved_files.append(outlier_rows_path)
    
    return saved_files

# 10.1 対象コホートの定義（各coverage設定ごとにループ内で定義）
# ここでは定義のみ（実際の使用は各coverage設定ごとのループ内で）

# ID列を自動検出
id_cols = []
for col in ['directory_id', 'segment_id', 'vessel_id', 'IXI_ID']:
    if col in df.columns:
        id_cols.append(col)

# 変数列のマッピング（qc_vars用）
var_cols_dict = {
    'AGE': 'AGE',
    'mca_segment': 'mca_segment',
    'periodicity': 'periodicity',
    'surface_constraint': 'surface_constraint',
    'radius': radius_col,
    'lambda': lambda_col,
    'amplitude': amplitude_col,
    'length': length_col,
    'A_over_L': 'A_over_L',
    'lambda_over_r': 'lambda_over_r',
    'log_r': 'log_r',
    'log_lambda': 'log_lambda',
    'log_amplitude': 'log_amplitude',
    'log_length': 'log_length',
    'log_lambda_over_r': 'log_lambda_over_r',
    'log_A_over_L': 'log_A_over_L'
}

qc_vars = list(var_cols_dict.values())

# プロット用変数辞書
plot_vars = {
    'radius': (radius_col, 'log_r'),
    'lambda': (lambda_col, 'log_lambda'),
    'lambda_over_r': ('lambda_over_r', 'log_lambda_over_r'),
    'A_over_L': ('A_over_L', 'log_A_over_L'),
    'amplitude': (amplitude_col, 'log_amplitude'),
    'length': (length_col, 'log_length')
}

# 出力ディレクトリ
out_dirs = {
    'analysis': output_analysis_dir,
    'figures': output_figures_dir
}

# 10.1.1 N集計表（CSV）
print("\n  10.1 N集計表の作成")

# 10.1.2 欠損率表（CSV）- 全体
print("\n  10.1.2 欠損率表の作成（全体）")
missing_overall = []
for var in qc_vars:
    if var in df.columns:
        missing_count = df[var].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_overall.append({
            'variable': var,
            'missing_count': missing_count,
            'missing_percentage': missing_pct
        })

missing_overall_df = pd.DataFrame(missing_overall)
missing_overall_path = os.path.join(output_analysis_dir, 'qc_missingness_overall.csv')
missing_overall_df.to_csv(missing_overall_path, index=False, encoding='utf-8')
print(f"    全体の欠損率表を保存した: {missing_overall_path}")

# 各コホートに対してQCを実行（各coverage設定ごとに）
all_saved_files = [missing_overall_path]
cohort_n_counts = {}

print("\n  10.1.2 欠損率表の作成（各コホート）")
print("  10.1.3 分布図の作成")
print("  10.1.4 外れ値QC（IQR法）")

# 各coverage設定ごとにQCを実行
for coverage_min in COVERAGE_HAT_CORE_MIN_VALUES:
    print(f"\n  COVERAGE_HAT_CORE_MIN={coverage_min} のQC処理中")
    
    # 各設定ごとのコホート列名
    core_all_col = f'core_all_coverage_{coverage_min}'
    core_surface_evaluable_col = f'core_surface_evaluable_coverage_{coverage_min}'
    core_inplane_col = f'core_inplane_coverage_{coverage_min}'
    core_outofplane_col = f'core_outofplane_coverage_{coverage_min}'
    
    # 各設定ごとのコホート定義
    cohorts_coverage = {
        'core_all': df[core_all_col] if core_all_col in df.columns else pd.Series([False]*len(df)),
        'core_surface_evaluable': df[core_surface_evaluable_col] if core_surface_evaluable_col in df.columns else pd.Series([False]*len(df)),
        'core_inplane': df[core_inplane_col] if core_inplane_col in df.columns else pd.Series([False]*len(df)),
        'core_outofplane': df[core_outofplane_col] if core_outofplane_col in df.columns else pd.Series([False]*len(df))
    }
    
    for cohort_name, cohort_mask in cohorts_coverage.items():
        print(f"\n    {cohort_name} のQC処理中")
        saved_files = run_qc_for_cohort(
            df, cohort_name, cohort_mask, 
            var_cols_dict, out_dirs, id_cols, plot_vars, coverage_min=coverage_min
        )
        all_saved_files.extend(saved_files)
        
        # コホートのNを記録（coverage設定ごとに）
        cohort_key = f'{cohort_name}_coverage_{coverage_min}'
        cohort_n_counts[cohort_key] = cohort_mask.sum()
        print(f"      {cohort_name}: N={cohort_n_counts[cohort_key]}, {len(saved_files)}ファイルを保存した")

# ===== 11) 実行ログの出力 =====
print("\n" + "="*80)
print("実行ログ")
print("="*80)
print(f"読み込んだ行数: {initial_row_count}")
# AGE列のマージ状況を確認（存在する場合）
if 'AGE' in df.columns:
    age_merged_count = df['AGE'].notna().sum()
    print(f"AGEマージ成功件数: {age_merged_count} / {len(df)} レコード")
print(f"\n採用された列名マップ:")
for canonical_name, actual_col in column_map.items():
    print(f"  {canonical_name}: {actual_col}")
print(f"\n各コホートのN（coverage設定ごと）:")
for cohort_key, n in sorted(cohort_n_counts.items()):
    print(f"  {cohort_key}: {n}件")
if invalid_counts_summary:
    print(f"\n<=0値の検出件数:")
    for canonical_name, count in invalid_counts_summary.items():
        print(f"  {canonical_name}: {count}件")
else:
    print(f"\n<=0値の検出件数: 0件（すべて正常）")
print(f"\n保存した主要ファイルのパス一覧:")
print(f"  解析データセット: {dataset_csv_path}")
print(f"  全体の欠損率表: {missing_overall_path}")
print(f"  各コホートのQCファイル: {len(all_saved_files)}件")
for i, file_path in enumerate(all_saved_files[:10], 1):
    print(f"    {i}. {file_path}")
if len(all_saved_files) > 10:
    print(f"    ... (残り {len(all_saved_files) - 10} 件)")

# ===== 12) core_inplane 主解析コホート妥当性の比較QC =====
# 各設定ごとに統計解析を実行
for coverage_min in COVERAGE_HAT_CORE_MIN_VALUES:
    print("\n" + "="*80)
    print(f"core_inplane 主解析コホート妥当性の比較QC (COVERAGE_HAT_CORE_MIN={coverage_min})")
    print("="*80)
    
    # 各設定ごとのコホート列名
    core_all_col = f'core_all_coverage_{coverage_min}'
    core_surface_evaluable_col = f'core_surface_evaluable_coverage_{coverage_min}'
    core_inplane_col = f'core_inplane_coverage_{coverage_min}'
    core_outofplane_col = f'core_outofplane_coverage_{coverage_min}'
    
    # 12.1 4コホートの件数・割合を出す
    print("\n【12.1 4コホートの件数・割合】")
    
    cohort_comparison = []
    total_n = len(df)
    
    for cohort_name_base in ['core_all', 'core_surface_evaluable', 'core_inplane', 'core_outofplane']:
        cohort_name = f'{cohort_name_base}_coverage_{coverage_min}'
        if cohort_name not in df.columns:
            continue
        
        cohort_n = df[cohort_name].sum()
        cohort_pct = (cohort_n / total_n) * 100 if total_n > 0 else 0
        
        cohort_comparison.append({
            'coverage_setting': coverage_min,
            'cohort': cohort_name_base,
            'group': 'Overall',
            'subgroup': 'All',
            'N': cohort_n,
            'percentage': cohort_pct
        })
        
        # mca_segment別
        if 'mca_segment' in df.columns:
            for segment in df['mca_segment'].dropna().unique():
                segment_mask = (df['mca_segment'] == segment) & df[cohort_name]
                segment_n = segment_mask.sum()
                segment_total = (df['mca_segment'] == segment).sum()
                segment_pct = (segment_n / segment_total) * 100 if segment_total > 0 else 0
                
                cohort_comparison.append({
                    'coverage_setting': coverage_min,
                    'cohort': cohort_name_base,
                    'group': 'BySegment',
                    'subgroup': str(segment),
                    'N': segment_n,
                    'percentage': segment_pct
                })
        
        # age_group別
        if 'age_group' in df.columns and df['age_group'].notna().any():
            for age_group in df['age_group'].dropna().unique():
                age_mask = (df['age_group'] == age_group) & df[cohort_name]
                age_n = age_mask.sum()
                age_total = (df['age_group'] == age_group).sum()
                age_pct = (age_n / age_total) * 100 if age_total > 0 else 0
                
                cohort_comparison.append({
                    'coverage_setting': coverage_min,
                    'cohort': cohort_name_base,
                    'group': 'ByAgeGroup',
                    'subgroup': str(age_group),
                    'N': age_n,
                    'percentage': age_pct
                })
    
    cohort_comparison_df = pd.DataFrame(cohort_comparison)
    cohort_comparison_path = os.path.join(output_analysis_dir, f'qc_cohort_comparison_counts_coverage_{coverage_min}.csv')
    cohort_comparison_df.to_csv(cohort_comparison_path, index=False, encoding='utf-8')
    print(f"  4コホート比較表を保存した: {cohort_comparison_path}")
    
    # 12.2 core_inplane と core_outofplane の分布比較
    print("\n【12.2 core_inplane vs core_outofplane 分布比較】")

    # 対象変数の候補リスト
    comparison_vars = {
        'AGE': 'AGE',
        'radius': radius_col,
        'length': length_col,
        'lambda': lambda_col,
        'lambda_over_r': 'lambda_over_r',
        'amplitude': amplitude_col,
        'A_over_L': 'A_over_L',
        'coverage_hat': 'coverage_hat',
        'SNR': 'SNR',
        'partial_R2': 'partial_R2',
        'R2': 'R2',
        'cortex_coverage': 'cortex_coverage',
        'energy_coverage': 'energy_coverage',
        'constraint_coverage': 'constraint_coverage',
        'R_out_surface': 'R_out_surface'
    }

    # 存在する変数のみを採用
    available_vars = {}
    for var_name, col_name in comparison_vars.items():
        if col_name in df.columns:
            available_vars[var_name] = col_name
        else:
            print(f"  Warning: {var_name} ({col_name}) が見つからない。スキップする")

    print(f"  比較対象変数: {len(available_vars)}個")

    # 両群のデータを抽出
    inplane_df = df[df[core_inplane_col]].copy()
    outofplane_df = df[df[core_outofplane_col]].copy()
    
    comparison_stats = []
    
    for var_name, col_name in available_vars.items():
        inplane_data = inplane_df[col_name].dropna()
        outofplane_data = outofplane_df[col_name].dropna()
        
        # 統計量を計算
        inplane_n = len(inplane_data)
        inplane_missing = inplane_df[col_name].isna().sum()
        outofplane_n = len(outofplane_data)
        outofplane_missing = outofplane_df[col_name].isna().sum()
        
        if inplane_n > 0:
            inplane_mean = inplane_data.mean()
            inplane_std = inplane_data.std()
            inplane_median = inplane_data.median()
            inplane_q1 = inplane_data.quantile(0.25)
            inplane_q3 = inplane_data.quantile(0.75)
            inplane_iqr = inplane_q3 - inplane_q1
        else:
            inplane_mean = inplane_std = inplane_median = inplane_iqr = np.nan
        
        if outofplane_n > 0:
            outofplane_mean = outofplane_data.mean()
            outofplane_std = outofplane_data.std()
            outofplane_median = outofplane_data.median()
            outofplane_q1 = outofplane_data.quantile(0.25)
            outofplane_q3 = outofplane_data.quantile(0.75)
            outofplane_iqr = outofplane_q3 - outofplane_q1
        else:
            outofplane_mean = outofplane_std = outofplane_median = outofplane_iqr = np.nan
        
        comparison_stats.append({
            'coverage_setting': coverage_min,
            'variable': var_name,
            'inplane_N': inplane_n,
            'inplane_missing': inplane_missing,
            'inplane_mean': inplane_mean,
            'inplane_SD': inplane_std,
            'inplane_median': inplane_median,
            'inplane_IQR': inplane_iqr,
            'outofplane_N': outofplane_n,
            'outofplane_missing': outofplane_missing,
            'outofplane_mean': outofplane_mean,
            'outofplane_SD': outofplane_std,
            'outofplane_median': outofplane_median,
            'outofplane_IQR': outofplane_iqr
        })
        
        # ヒストグラムと箱ひげ図を作成
        if inplane_n > 0 and outofplane_n > 0:
            # 共通のビンと軸範囲を決定
            all_data = pd.concat([inplane_data, outofplane_data])
            data_min = all_data.min()
            data_max = all_data.max()
            bins = np.linspace(data_min, data_max, 50)
            
            # ヒストグラム
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(inplane_data, bins=bins, alpha=0.6, label='core_inplane', color='blue', edgecolor='black')
            ax.hist(outofplane_data, bins=bins, alpha=0.6, label='core_outofplane', color='orange', edgecolor='black')
            ax.set_xlabel(var_name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram: {var_name} (coverage={coverage_min}, inplane N={inplane_n}, outofplane N={outofplane_n})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            hist_path = os.path.join(output_figures_dir, f'comparison_hist_{var_name}_inplane_vs_outofplane_coverage_{coverage_min}.png')
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 箱ひげ図
            fig, ax = plt.subplots(figsize=(8, 6))
            box_data = [inplane_data, outofplane_data]
            bp = ax.boxplot(box_data, tick_labels=['core_inplane', 'core_outofplane'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            ax.set_ylabel(var_name)
            ax.set_title(f'Boxplot: {var_name} (coverage={coverage_min}, inplane N={inplane_n}, outofplane N={outofplane_n})')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            box_path = os.path.join(output_figures_dir, f'comparison_box_{var_name}_inplane_vs_outofplane_coverage_{coverage_min}.png')
            plt.savefig(box_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    comparison_stats_df = pd.DataFrame(comparison_stats)
    comparison_stats_path = os.path.join(output_analysis_dir, f'qc_comparison_stats_inplane_vs_outofplane_coverage_{coverage_min}.csv')
    comparison_stats_df.to_csv(comparison_stats_path, index=False, encoding='utf-8')
    print(f"  分布比較統計表を保存した: {comparison_stats_path}")
    
    # 12.3 inplane判定の依存性チェック（ロジスティック回帰）
    print("\n【12.3 inplane判定の依存性チェック（ロジスティック回帰）】")

    # 参加者ID列を検出
    subject_id_col = None
    for col in ['directory_id', 'case_id', 'case']:
        if col in df.columns:
            subject_id_col = col
            break
    
    if subject_id_col is None:
        print("  Error: 参加者ID列（directory_id, case_id, case）が見つからない")
        sys.exit(1)
    
    print(f"  参加者ID列: {subject_id_col}")
    
    # core_inplane と core_outofplane のデータのみを使用
    model_df = df[(df[core_inplane_col] | df[core_outofplane_col])].copy()
    model_df = model_df.dropna(subset=['AGE', 'log_r', 'mca_segment'])
    
    if len(model_df) == 0:
        print("  Warning: モデル用データがない。スキップする")
    else:
        # 目的変数: 1=core_inplane, 0=core_outofplane
        y = model_df[core_inplane_col].astype(int)
        
        # 説明変数の準備
        X = pd.DataFrame()
        X['AGE'] = pd.to_numeric(model_df['AGE'], errors='coerce')
        X['log_r'] = pd.to_numeric(model_df['log_r'], errors='coerce')
        
        # mca_segmentをダミー変数化
        if 'mca_segment' in model_df.columns:
            segment_dummies = pd.get_dummies(model_df['mca_segment'], prefix='segment', drop_first=True)
            # 数値型に変換
            for col in segment_dummies.columns:
                segment_dummies[col] = segment_dummies[col].astype(float)
            X = pd.concat([X, segment_dummies], axis=1)
        
        # 欠損値を除外
        X = X.dropna()
        if len(X) == 0:
            print("  Warning: 説明変数に有効なデータがない。スキップする")
        else:
            # 目的変数とmodel_dfをXのインデックスに合わせる
            y = model_df.loc[X.index, core_inplane_col].astype(int)
            model_df_subset = model_df.loc[X.index]
            
            # 定数項を追加
            X = sm.add_constant(X)
            
            # 数値型であることを確認
            X = X.astype(float)
            
            # ロジスティック回帰を実行
            try:
                logit_model = sm.Logit(y, X)
                result = logit_model.fit(disp=0)
                
                # クラスタロバストSEを計算
                cluster_ids = model_df_subset[subject_id_col].values
                
                # クラスタロバスト共分散行列を計算
                # スコア関数（各観測値の勾配）を計算
                y_pred = result.predict(X)
                # ロジスティック回帰のスコア関数: X^T * (y - y_pred)
                score_matrix = X.values * (y.values[:, np.newaxis] - y_pred.values[:, np.newaxis])
                
                # クラスタごとにスコアを合計
                cluster_df = pd.DataFrame(score_matrix, columns=X.columns)
                cluster_df['cluster'] = cluster_ids
                cluster_scores = cluster_df.groupby('cluster')[X.columns].sum()
                
                # クラスタロバスト共分散行列を計算
                n_clusters = len(cluster_scores)
                n_params = len(result.params)
                cov_cluster_robust = cluster_scores.values.T @ cluster_scores.values
                
                # 調整係数を適用
                n = len(y)
                k = n_params
                if n_clusters > k:
                    adjustment = (n_clusters / (n_clusters - 1))
                else:
                    adjustment = 1.0
                cov_cluster_robust = adjustment * cov_cluster_robust
                
                # 標準誤差を計算
                se_cluster = np.sqrt(np.diag(cov_cluster_robust))
                
                # 結果をまとめる
                regression_results = []
                for i, var_name in enumerate(X.columns):
                    coef = result.params[var_name]
                    se = se_cluster[i]
                    z_score = coef / se if se > 0 else np.nan
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) if not np.isnan(z_score) else np.nan
                    
                    regression_results.append({
                        'coverage_setting': coverage_min,
                        'variable': var_name,
                        'coefficient': coef,
                        'SE_cluster_robust': se,
                        'z_score': z_score,
                        'p_value': p_value
                    })
                
                regression_results_df = pd.DataFrame(regression_results)
                regression_results_path = os.path.join(output_analysis_dir, f'qc_logistic_regression_inplane_dependency_coverage_{coverage_min}.csv')
                regression_results_df.to_csv(regression_results_path, index=False, encoding='utf-8')
                print(f"  ロジスティック回帰結果を保存した: {regression_results_path}")
                print(f"    サンプル数: {len(X)}")
                print(f"    core_inplane: {y.sum()}件, core_outofplane: {(y==0).sum()}件")
                
            except Exception as e:
                print(f"  Error: ロジスティック回帰の実行に失敗した: {e}")
                import traceback
                traceback.print_exc()

print("\n" + "="*80)
print("core_inplane 主解析コホート妥当性の比較QCが完了した（すべての設定）")
print("="*80)

print("\n" + "="*80)
print("QC出力が完了した")
print("="*80)
print(f"出力先:")
print(f"  CSV/Parquet: {output_analysis_dir}")
print(f"  図PNG: {output_figures_dir}")
if SHOW_PLOTS:
    print("\nすべての図の表示が完了した")
else:
    print("\nすべての図をファイルに保存した（ウィンドウ表示は無効）")
