#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sin+trend fitting
"""

########################################################
# ===== USER SETTINGS =====
directory_id = "IXI000-HH-0000"

# ===== 再計算が必要な設定（後から変更できない） =====
# 基本設定
voxel_size = 0.5  # mm

# 前処理パラメータ
s_factor = 1.2
num_iter = 100
s_length = 0.2  # ステップ長さ（voxel単位）
removal = 10  # 端点除去点数

# sin trend fittingのパラメータ（再計算が必要）
trend_degree = 1
N_cyc = 0.5
N_pp = 20
ds_eff = 1  # ds_eff: 効果的サンプリング間隔(ここでは1 voxel。もし実際の空間で行う場合はvoxelサイズを入れる)
q_fine = 8  # Δk = π/(q_fine*L).  q_fine=8 -> π/(8L): 最大位相誤差 ≤ 11.25°
dR2_tol = 0.006  # grid searchでのR2の許容誤差
dhr_amp_tol = 0.05  # tie set内でのhr_ampの許容誤差
boundary_margin_bins = 2  # 端に張り付いた判定幅（grid何点ぶん）

# 脳表拘束解析のパラメータ（再計算が必要）
coh_thr = 0.70  # 脳表拘束の成立判定 (完全にramdomならば1/3。0.75はtheta=30°、0.70はtheta=33°に相当)
patch_r_ratio = 3.0  # 血管の周囲の探索するマージン
stride = 10  # サンプリング間隔（anchor pointsの間隔）
min_neighbors = 30  # 近傍点の最小数
gap_fill_size = 3  # gap_fill_size-1のサイズの穴まで埋める（初期設定ではanchor points 2個(=2mm)までの穴を埋める）

# 除外条件（再計算が必要）
min_length_mm = 10.0  # 最小セグメント長さの閾値　※基本的なexclusionは先に判定済みだが、閾値を変更して再判定可能

# ===== 判定用の閾値（後から変更可能、CSVに保存される） =====
# sin+trend判定用の閾値
partial_R2_thr = 0.061
R2_thr = 0.3

# 脳表拘束判定用の閾値（CSVに保存される生データに基づいて後から判定可能）
constraint_coverage_thresh = 0.5  # 拘束が成立している区間の割合。セグメントの過半数を十分に超える (0.5で過半数、0.6はmajority + marginといった保守的な閾値)
energy_coverage_thresh = 0.5  # 脳表拘束あり区間で説明できるエネルギー割合。変形の大半が拘束区間で起きている (0.5で過半数、0.6はmajority + marginといった保守的な閾値)
R_out_thresh = 0.30  # 面外率の閾値（<0.30 なら in-plane）, 0.25はtheta=30°、0.3はtheta=33°に相当

# 脳表拘束判定用の閾値（（再計算が必要）
r_min_mm = 2.5
r_max_mm = 5.0


"""
# prior
alpha_prior = 25.27038792824841  # 251125 alphaの決め方 (251116 データを1参加者ずつ抽出 3の検証の続きの続き secondary修正).ipynb　を参照

# DCT
detrend_deg = 2
"""

# ===== Visualization settings =====
# PyVista plotter settings
plotter_window_size = (2048, 1152) # ウィンドウサイズ (width, height)
plotter_position = (100, 500)  # ウィンドウの位置 (x, y)
########################################################


case_id = directory_id[3:6].lstrip("0")
case_id = int(case_id)


import os
import sys
import glob
import numpy as np
import pandas as pd
import pyvista as pv
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

# Add the path to make the utils module importable
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
utils_path = os.path.join(project_root, 'analysis_code', 'utils')
mesh_path = os.path.join(project_root, 'analysis_code', 'mesh')
sys.path.insert(0, os.path.join(project_root, 'analysis_code'))  # Add analysis_code to path for package imports
sys.path.append(utils_path)
sys.path.append(mesh_path)

# Import function modules from the same directory
import utils
import utils.tool as tool
import mesh_generation
import centerline_extraction
import vessel_mesh_features
from sin_trend_fitting import trime_ends, arclength, normalize, poly_trend, bishop_frame, lateral_components, grid_search_k_band, clip_mesh_to_coords, analyze_surface_constraint_coherence

# save directory and dataframe output path
# 個別ファイルは morphometrics/sin_trend/per_case/{directory_id}/ に保存
output_dir_combined = os.path.join(project_root, 'results', 'morphometrics', 'sin_trend', 'per_case')
utils.ensure_directory_exists(output_dir_combined, verbose=True)
output_dir_per_case = os.path.join(output_dir_combined, directory_id)
utils.ensure_directory_exists(output_dir_per_case, verbose=True)
# 統合したCSVファイルは morphometrics/sin_trend/per_case/ に保存

########################################################
# Step 1（マクロな説明）
vessel_centerlines_dir = os.path.join(project_root, 'data', 'derived', 'vessel_centerlines', directory_id)
merged_splitted_centerlines_path = os.path.join(vessel_centerlines_dir, f"{directory_id}_merged_centerlines.vtp")
merged_splitted_centerlines = centerline_extraction.load_vtp(merged_splitted_centerlines_path)

### outcome: 
### κ_orig_RMS
### D, L, TI(=L/D) & 余剰長ρ(=L/D−1)
### linearity, planarity1, scattering, planarity2    
celldata, pointids, points, pointdata = centerline_extraction.vmtk_convert2numpy(merged_splitted_centerlines)

# pointidsをリストから辞書に変換（GroupIdsをキーとして使用）
# pointidsはリストで、各要素がセグメントのポイントIDのリスト
# celldata['GroupIds']は各セル（セグメント）のGroupIdを格納
group_ids = celldata['GroupIds']
pointids_dict = {group_id: pointids[i] for i, group_id in enumerate(group_ids)}

# すべてのsegment_idを取得
all_segment_ids = sorted(pointids_dict.keys())
print(f"処理対象のsegment_id数: {len(all_segment_ids)}")
print(f"segment_idリスト: {all_segment_ids}")

# P_vox_3darrが左右のどちらのセグメントに属するかを判定（全segment_idに対して一度だけ実行）
# classify_centerline_left_right_by_kmeansはリスト形式のpointidsを期待している
right_cells, left_cells, threshold_x, is_right_point, is_left_point = vessel_mesh_features.classify_centerline_left_right_by_kmeans(
    points, pointids, n_iter=100, verbose=False)

def classify_and_display_segment(csv_path, segment_id, cell_index, voxel_size):
    """
    CSVファイルを読み込んで判定を行い、結果を表示する関数
    
    Args:
        csv_path: CSVファイルのパス
        segment_id: セグメントID
        cell_index: セルインデックス
        voxel_size: ボクセルサイズ
    """
    try:
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            row = df.iloc[0]
            
            # 判定ロジック（表示用）
            CLASS_UNDET = 'Long-wave / Straight-ish / undetectable' # 周期として語れない（周期数不足・振幅不足・SNR不足・sin寄与なし等）
            CLASS_CORE = 'Core' # 周期性が強く、推定が安定（coverage も十分、R2/partial_R2 も十分、歪みも小さい）
            CLASS_SEC = 'Secondary' # 周期らしさはあるが Core ほど堅くない（弱い/曖昧/境界など）
            CLASS_EXCL = 'Excluded (Too short)' # 長さが短すぎる（<10mm）
            
            # 1. まずsin+trend判定を行う
            L_trimmed_mm = row.get('L_trimmed_mm', np.nan)
            min_length_mm = row.get('min_length_mm', 10.0)
            boundary_hit_low = row.get('boundary_hit_low', np.nan)
            boundary_hit_high = row.get('boundary_hit_high', np.nan)
            coverage_hat = row.get('coverage_hat', np.nan)
            amp_hat_mm = row.get('amp_hat_mm', np.nan)
            SNR = row.get('SNR', np.nan)
            partial_R2 = row.get('partial_R2', np.nan)
            R2 = row.get('R2', np.nan)
            hr_amp = row.get('hr_amp', np.nan)
            partial_R2_thr = row.get('partial_R2_thr', 0.061)
            R2_thr = row.get('R2_thr', 0.3)
            
            # Exclusion判定
            reason_periodicity = None
            if pd.notna(L_trimmed_mm) and pd.notna(min_length_mm) and L_trimmed_mm < min_length_mm:
                periodicity = CLASS_EXCL
                category = None  # Excluded (Too short)の場合は表面拘束判定は不要
                reason_periodicity = f"L_trimmed_mm ({L_trimmed_mm:.3f} mm) < min_length_mm ({min_length_mm:.3f} mm)"
            elif pd.notna(boundary_hit_low) and boundary_hit_low and pd.notna(coverage_hat) and coverage_hat < 1.0:
                periodicity = CLASS_UNDET
                category = None  # Undetectableの場合は表面拘束判定は不要
                reason_periodicity = f"boundary_hit_low=True and coverage_hat ({coverage_hat:.3f}) < 1.0"
            elif pd.notna(coverage_hat) and coverage_hat < 0.5:
                periodicity = CLASS_UNDET
                category = None
                reason_periodicity = f"coverage_hat ({coverage_hat:.3f}) < 0.5"
            elif pd.notna(amp_hat_mm) and amp_hat_mm < voxel_size:  # 1 voxel未満
                periodicity = CLASS_UNDET
                category = None
                reason_periodicity = f"amp_hat_mm ({amp_hat_mm:.3f} mm) < voxel_size ({voxel_size:.3f} mm)"
            elif pd.notna(SNR) and SNR < 1.5:
                periodicity = CLASS_UNDET
                category = None
                reason_periodicity = f"SNR ({SNR:.3f}) < 1.5"
            elif pd.notna(partial_R2) and pd.notna(partial_R2_thr) and partial_R2 < partial_R2_thr:
                periodicity = CLASS_UNDET
                category = None
                reason_periodicity = f"partial_R2 ({partial_R2:.3f}) < partial_R2_thr ({partial_R2_thr:.3f})"
            elif pd.notna(R2) and pd.notna(R2_thr) and R2 < R2_thr:
                periodicity = CLASS_UNDET
                category = None
                reason_periodicity = f"R2 ({R2:.3f}) < R2_thr ({R2_thr:.3f})"
            elif (pd.notna(coverage_hat) and coverage_hat >= 1.5 and 
                  pd.notna(partial_R2) and partial_R2 >= 0.15 and 
                  pd.notna(R2) and R2 >= 0.6 and 
                  pd.notna(hr_amp) and hr_amp <= 0.2):
                periodicity = CLASS_CORE
                reason_periodicity = f"coverage_hat ({coverage_hat:.3f}) >= 1.5, partial_R2 ({partial_R2:.3f}) >= 0.15, R2 ({R2:.3f}) >= 0.6, hr_amp ({hr_amp:.3f}) <= 0.2"
                # Coreの場合、表面拘束判定を適用
            else:
                periodicity = CLASS_SEC
                reason_periodicity = "Otherwise (not Core, not excluded, not undetectable)"
                # Secondaryの場合、表面拘束判定を適用
            # Coreになったが境界に張り付いているならSecondaryに落とす（波長推定が最短波長側で飽和していて、λが信用しにくい＝分解能限界・ノイズ・未モデル化高周波の疑い）
            if periodicity == CLASS_CORE and pd.notna(boundary_hit_high) and boundary_hit_high:
                periodicity = CLASS_SEC
                reason_periodicity += f" -> Secondary (boundary_hit_high=True)"
            
            # 2. Core/Secondaryの場合のみ、表面拘束判定を適用
            reason_category = None
            if periodicity in [CLASS_CORE, CLASS_SEC]:
                cortex_coverage = row.get('cortex_coverage', np.nan)
                energy_coverage = row.get('energy_coverage', np.nan)
                constraint_coverage = row.get('constraint_coverage', np.nan)
                R_out = row.get('R_out_surface', np.nan)
                energy_coverage_thresh = row.get('energy_coverage_thresh', 0.5)
                constraint_coverage_thresh = row.get('constraint_coverage_thresh', 0.5)
                R_out_thresh = row.get('R_out_thresh', 0.25)
                
                if pd.isna(cortex_coverage) or cortex_coverage < 0.3:
                    cortex_str = "NaN" if pd.isna(cortex_coverage) else f"{cortex_coverage:.3f}"
                    category = "Excluded (No cortex constraint evaluated)"
                    reason_category = f"cortex_coverage ({cortex_str}) < 0.3"
                elif pd.isna(energy_coverage) or energy_coverage < energy_coverage_thresh:
                    energy_str = "NaN" if pd.isna(energy_coverage) else f"{energy_coverage:.3f}"
                    category = "Excluded (Insufficient constrained region)"
                    reason_category = f"energy_coverage ({energy_str}) < energy_coverage_thresh ({energy_coverage_thresh:.3f})"
                elif pd.isna(constraint_coverage) or constraint_coverage < constraint_coverage_thresh:
                    constraint_str = "NaN" if pd.isna(constraint_coverage) else f"{constraint_coverage:.3f}"
                    category = "Excluded (Insufficient constraint coverage)"
                    reason_category = f"constraint_coverage ({constraint_str}) < constraint_coverage_thresh ({constraint_coverage_thresh:.3f})"
                elif pd.notna(R_out) and R_out < R_out_thresh:
                    category = "In-plane Buckling"
                    reason_category = f"R_out ({R_out:.3f}) < R_out_thresh ({R_out_thresh:.3f})"
                else:
                    R_out_str = "NaN" if pd.isna(R_out) else f"{R_out:.3f}"
                    category = "Terrain Following / Artifact"
                    reason_category = f"Otherwise (R_out ({R_out_str}) >= R_out_thresh ({R_out_thresh:.3f}))"
            else:
                # Excluded (Too short) または Undetectable の場合は表面拘束判定は不要
                category = None
                reason_category = "N/A (periodicity is Excluded or Undetectable)"
            
            # 判定結果を表示
            print("\n" + "="*60)
            print(f"segment_id {segment_id:3d} (segment_index: {cell_index}) の判定結果:")
            print(f"  sin+trend判定: {periodicity}")
            if reason_periodicity:
                print(f"    理由: {reason_periodicity}")
            if category is not None:
                print(f"  表面拘束判定: {category}")
                if reason_category:
                    print(f"    理由: {reason_category}")
            else:
                print(f"  表面拘束判定: -")
                if reason_category:
                    print(f"    理由: {reason_category}")
            print("="*60)
    except Exception as e:
        print(f"Warning: Failed to read and classify {csv_path}: {str(e)}")


def process_segment(segment_id):
    """個別のsegment_idに対する処理を実行する関数
    
    Returns:
        DataFrame: セグメントのデータを含むDataFrame、エラー時はNone
    """
    # segment_idに対応するセルインデックスを取得（group_idsは連番ではない可能性がある）
    # group_idsは配列で、group_ids[i]がセルインデックスiのGroupId（segment_id）を表す
    cell_index_arr = np.where(group_ids == segment_id)[0]
    if len(cell_index_arr) == 0:
        print(f"Warning: segment_id {segment_id}に対応するセルが見つからない")
        return None
    cell_index = cell_index_arr[0]  # スカラー値として取得

    print("\n" + "="*80)
    print(f"Processing segment_id: {segment_id}, segment_index: {cell_index}")
    print("="*80)
    
    try:
        # Original centerlinesの座標
        P_vox_3darr = points[pointids_dict[segment_id]]
        pts = len(P_vox_3darr)

        # Radius
        radius_vox_arr = pointdata['MaximumInscribedSphereRadius'][pointids_dict[segment_id]]
        radius_median_vox = np.median(radius_vox_arr)
        radius_mean_vox = np.mean(radius_vox_arr)
        radius_std_vox= np.std(radius_vox_arr, ddof=1)

        # Arc length (= segment length)
        s_vox_arr, ds_vox_arr = arclength(P_vox_3darr)
        L_vox = s_vox_arr[-1]
        # Median step length
        ds_vox = np.median(ds_vox_arr)

        # Euclidean distance between the first and last points
        euclid_d_vox = np.linalg.norm(P_vox_3darr[0] - P_vox_3darr[-1])

        # Tortuosity index
        TI = L_vox / euclid_d_vox
        rho = TI - 1

        print("segment_id:", segment_id)
        print("pts", pts)
        print(f"ds: {ds_vox:.3f} voxel, {ds_vox * voxel_size:.3f} mm")
        print(f"segment length, L: {L_vox:.3f} voxel, {L_vox * voxel_size:.3f} mm") 
        print("celldataでのlengthと一致しているかチェック:")
        length_from_celldata = celldata['Length'][cell_index]
        print(f"==> {np.round(length_from_celldata, decimals=3) == np.round(L_vox, decimals=3)}")
        print(f"radius, median: {radius_median_vox:.3f} voxel, {radius_median_vox * voxel_size:.3f} mm")
        print(f"radius, mean: {radius_mean_vox:.3f} voxel, {radius_mean_vox * voxel_size:.3f} mm")
        print(f"radius, std: {radius_std_vox:.3f} voxel, {radius_std_vox * voxel_size:.3f} mm")
        print(f"euclid distace, D: {euclid_d_vox:.3f} voxel, {euclid_d_vox * voxel_size:.3f} mm")
        print(f"tortuosity index: {TI:.3f}")
        print(f"rho (=TI-1): {rho:.3f}")
        print("\n" + "="*40)


        # Trimmed centerlinesの座標
        P_vox_3darr_trimmed = trime_ends(P_vox_3darr, removal)
        pts_trimmed = len(P_vox_3darr_trimmed)
        radius_vox_arr_trimmed = trime_ends(radius_vox_arr, removal)
        s_vox_arr_trimmed, ds_vox_arr_trimmed = arclength(P_vox_3darr_trimmed) # Rmks; s_vox_arr_trimmed = trime_ends(s_vox_arr)とする場合は、弧長を0始まりに調整が必要。s_vox_arr_trimmed = s_vox_arr_trimmed - s_vox_arr_trimmed[0]
        L_vox_trimmed = s_vox_arr_trimmed[-1]

        brain_mesh_dir = os.path.join(project_root, 'data', 'preprocessed', 'T1_preprocessed', 'brain_meshes', directory_id)
        # segment_idに対応するセルインデックスを取得済み
        if cell_index in right_cells:
            side = "Rt"
            brain_mesh_path = os.path.join(brain_mesh_dir, f"{directory_id}_rt_pial_remesh.ply")
            brain_mesh = pv.read(brain_mesh_path)
        else:
            side = "Lt"
            brain_mesh_path = os.path.join(brain_mesh_dir, f"{directory_id}_lt_pial_remesh.ply")
            brain_mesh = pv.read(brain_mesh_path)
        print(f"side: {side}")

        # mca_segmentは後で統合CSVファイルに対して可視化して決定するため、ここではNoneに設定
        mca_segment = None

        # 1D buckling modelの前提 (rod-like)
        slenderness = L_vox / (2*radius_median_vox)
        print(f"slenderness: {slenderness:.3f}")

        # kappa & tau & frenet vectors
        kappa_vox_arr = pointdata['Curvature'][pointids_dict[segment_id]]
        tau_vox_arr = pointdata['Torsion'][pointids_dict[segment_id]]
        frenettangent_vox_arr = pointdata['FrenetTangent'][pointids_dict[segment_id]]
        frenetnormal_vox_arr = pointdata['FrenetNormal'][pointids_dict[segment_id]]
        frenetbinormal_vox_arr = pointdata['FrenetBinormal'][pointids_dict[segment_id]]

        # kappaは端点が不安定になるので、端で暴れているκは「端点効果」として切り捨てる
        kappa_arr_trimmed = trime_ends(kappa_vox_arr, removal)
        
        # kappa_arr_trimmedが空でないことを確認
        if len(kappa_arr_trimmed) == 0:
            print(f"Warning: kappa_arr_trimmed is empty (original length: {len(kappa_vox_arr)}, removal: {removal})")
            print("Using original kappa_vox_arr for statistics")
            kappa_arr_trimmed = kappa_vox_arr
        
        if len(kappa_arr_trimmed) > 0:
            kappa_median_vox = np.median(kappa_arr_trimmed)
            kappa_mean_vox = np.mean(kappa_arr_trimmed) # もともと0以上の値しかとらないので、Average (Absolute) Curvature = Mean Curvature
            kappa_std_vox = np.std(kappa_arr_trimmed, ddof=1)
            print(f"kappa, median: {kappa_median_vox:.3f} /voxel, {kappa_median_vox / voxel_size:.3f} /mm")
            print(f"kappa, mean: {kappa_mean_vox:.3f} /voxel, {kappa_mean_vox / voxel_size:.3f} /mm")
            print(f"kappa, std: {kappa_std_vox:.3f} /voxel, {kappa_std_vox / voxel_size:.3f} /mm")
            kappa_q1_vox = np.percentile(kappa_arr_trimmed, 25)
            kappa_q3_vox = np.percentile(kappa_arr_trimmed, 75)
            kappa_iqr_vox = kappa_q3_vox - kappa_q1_vox
            print(f"kappa Q1: {kappa_q1_vox:.3f} /voxel, {kappa_q1_vox / voxel_size:.3f} /mm")
            print(f"kappa Q3: {kappa_q3_vox:.3f} /voxel, {kappa_q3_vox / voxel_size:.3f} /mm")
            print(f"kappa IQR: {kappa_iqr_vox:.3f} /voxel, {kappa_iqr_vox / voxel_size:.3f} /mm")
            kappa_rms_vox = np.sqrt(np.mean(kappa_arr_trimmed**2))
            kappa_max_vox = np.max(kappa_arr_trimmed)
            kappa_min_vox = np.min(kappa_arr_trimmed)
            kappa_p95_vox  = np.percentile(kappa_arr_trimmed, 95)
            print(f"kappa rms: {kappa_rms_vox:.3f} /voxel, {kappa_rms_vox / voxel_size:.3f} /mm")
            print(f"kappa p95: {kappa_p95_vox:.3f} /voxel, {kappa_p95_vox / voxel_size:.3f} /mm")
            print(f"kappa max: {kappa_max_vox:.3f} /voxel, {kappa_max_vox / voxel_size:.3f} /mm")
            print(f"kappa min: {kappa_min_vox:.3f} /voxel, {kappa_min_vox / voxel_size:.3f} /mm")
        else:
            # 完全に空の場合（元の配列も空）
            print("Warning: kappa_vox_arr is empty, setting all kappa statistics to NaN")
            kappa_median_vox = np.nan
            kappa_mean_vox = np.nan
            kappa_std_vox = np.nan
            kappa_q1_vox = np.nan
            kappa_q3_vox = np.nan
            kappa_iqr_vox = np.nan
            kappa_rms_vox = np.nan
            kappa_max_vox = np.nan
            kappa_min_vox = np.nan
            kappa_p95_vox = np.nan
            print(f"kappa, median: nan /voxel, nan /mm")
            print(f"kappa, mean: nan /voxel, nan /mm")
            print(f"kappa, std: nan /voxel, nan /mm")
    


        # segment内の平面性の情報 (全体を1枚の面に押し込めるとどう見えるか) (参考: 局所平面性はセグメント全体で見る必要は必ずしもない)
        # PCA analysis
        pca = PCA(n_components=3)
        pca.fit(P_vox_3darr * voxel_size)

        eigvals = pca.explained_variance_
        eigval1, eigval2, eigval3 = eigvals
        print(f"固有値 λ1: {eigval1:.3f} mm^2, 固有値 λ2: {eigval2:.3f} mm^2, 固有値 λ3: {eigval3:.3f} mm^2")

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
        scattering = eigval3 / eigval1
        print(f"Linearity = {linearity:.3f}, Planarity = {planarity:.3f}, Scattering = {scattering:.3f}")

        # Classification of Boulders in Coastal Environments Using Random Forest Machine Learning on Topo-Bathymetric LiDAR Data　を参考に他の指標も求めておく
        omni =  (eigval1 * eigval2 * eigval3) **(1/3)
        aniso = (eigval1 - eigval3) / eigval1 
        c_curvature = eigval3 / np.sum(eigvals)
        print(f"omnivariance = {omni:.3f}, anisotropy = {aniso:.3f}, change of curvature = {c_curvature:.3f}")
        print("\n" + "="*40)

        # =========================
        # Exclusion（基本的なexclusionは先に判定し、閾値は保存する）
        # 計算コストが高い処理をスキップするため、先に判定する
        if L_vox_trimmed * voxel_size < min_length_mm:
            print(f"=> Exclusion: Segment too short (L_trimmed = {L_vox_trimmed * voxel_size:.3f} mm < {min_length_mm} mm)")
            print("\n" + "="*60)

            data_vessel_segmentdata = {
            "IXI_ID": [case_id],
            "directory_id": [directory_id],
            "segment_id": [segment_id],
            "segment_index": [cell_index],
            "mca_segment": [mca_segment],
            "side": [side],
            "pts": [pts],
            "pts_trimmed": [pts_trimmed],
            "ds_mm": [ds_vox * voxel_size],
            "L_full_mm": [s_vox_arr[-1] * voxel_size],
            "L_trimmed_mm": [L_vox_trimmed * voxel_size],
            "radius_median_mm": [radius_median_vox * voxel_size],
            "radius_mean_mm": [radius_mean_vox * voxel_size],
            "radius_std_mm": [radius_std_vox * voxel_size],
            "euclid_distance_mm": [euclid_d_vox * voxel_size],
            "TI": [TI],
            "rho": [rho],

            # κ 系 → NaN
            "kappa_median_mm": [kappa_median_vox / voxel_size],
            "kappa_mean_mm": [kappa_mean_vox / voxel_size],
            "kappa_std_mm": [kappa_std_vox / voxel_size],
            "kappa_rms_mm": [kappa_rms_vox / voxel_size],
            "kappa_p95_mm": [kappa_p95_vox / voxel_size],
            "kappa_max_mm": [kappa_max_vox / voxel_size],
            "kappa_min_mm": [kappa_min_vox / voxel_size],
        
            "slenderness": [slenderness],

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

            # sin+trend（判定はCSV保存後に柔軟に行うため、判定結果は保存しない）
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
            "amp_hat_over_L": [np.nan],
            "lambda_hat_over_radius": [np.nan],
            "amp_hat_over_radius": [np.nan],

            "R2": [np.nan],
            "partial_R2": [np.nan],
            "SSE_full": [np.nan],
            "SSE_tr": [np.nan],
            "hr_amp": [np.nan],
            "hr_energy": [np.nan],
            "coverage_hat": [np.nan],
            "SNR": [np.nan],
            "rms_residual_full_mm": [np.nan],
            "rms_residual_tr_mm": [np.nan],

            "boundary_hit_low": [np.nan],
            "boundary_hit_high": [np.nan],

            # 追加
            # 3D Constraint Metrics (脳表拘束の証明用)
            "R_out_surface": [np.nan],
            "R_dir_surface": [np.nan],
            "cortex_coverage": [np.nan],
            "constraint_coverage": [np.nan],
            "energy_coverage": [np.nan],
            "total_energy": [np.nan],
            "energy_constrained": [np.nan],
            "perp_energy": [np.nan],
            "n_points_total": [np.nan],
            "n_eval": [np.nan],
            "coh_thr": [np.nan],
            "constraint_coverage_thresh": [np.nan],
            "energy_coverage_thresh": [np.nan],
            "R_out_thresh": [np.nan],
            "stride_mm": [np.nan],
            "min_neighbors": [np.nan],
            "r_patch_mm": [np.nan],
            "gap_fill_size": [np.nan],
            
            # 判定用の閾値（再現性のため）
            "min_length_mm": [min_length_mm],
            "partial_R2_thr": [np.nan],
            "R2_thr": [np.nan]

            }
            # indexとcolumnsを設定してDataFrameを作成
            df_vessel_segmentdata = pd.DataFrame(data_vessel_segmentdata)
            
            # DataFrameを返す（個別CSVファイルは保存しない）
            return df_vessel_segmentdata

        else:
            pass

        # =========================
        # Step 2（ミクロな buckling 成分）

        # Centerline を 2次多項式トレンドで近似し、baseline centerlineの作成
        # 目的としては信号の低次のトレンド(平均・一次・必要に応じ二次)を取り除き、P_orig(s)-P_0(s) を「振動（座屈候補）成分」として扱う = 信号処理で言えばDCT/DFTをかける前detrend(低次トレンド除去))。
        P0_vox_3darr, info_poly_trend = poly_trend(P_3darr=P_vox_3darr,
                                s_arr=s_vox_arr,
                                degree = 2, 
                                return_residual = True)

        # Original centerline上のBishop frameを作成する
        T_vox_3darr, U_vox_3darr, V_vox_3darr = bishop_frame(frenettangent_vox_arr)
        # 観測された centerline：P_vox_3darr
        # 大域トレンド（多項式）：P0_vox_3darr
        # => Δ = P_vox_3darr − P0_vox_3darr（observed − trend）とする
        # 血管中心線Pと平滑基準線P_0の差\DeltaをBishop frame(T, U, V)に射影し、横方向の変位成分u_U, u_Vを取り出す
        # 中心線に対して「横方向（sideways）」の変位成分: lateral_components
        # uU: (N,) 平面内横方向変位量（符号付き） ※buckling（座屈波）の主成分として解析(しかしこの時点ではsurfaceとの位置合わせされていない)
        # uV: (N,) 面外方向変位量（符号付き） ※最終的にはねじれや脳表起伏の影響を反映（地形的影響）
        # Delta: (N, 3) 差ベクトル
        # Delta_parT: (N, 3) 接線方向変位、つまり、軸方向
        # Delta_perpT: (N, 3) 横方向変位（Tに垂直）形状変化の主要成分
        # DeltaU: (N, 3) U方向変位ベクトル ※φ角解析・可視化用
        # DeltaV: (N, 3) V方向変位ベクトル ※φ角解析・可視化用
        uU_vox_arr, uV_vox_arr, Delta_vox_3darr, Delta_parT_vox_3darr, Delta_perpT_vox_3darr, DeltaU_vox_3darr, DeltaV_vox_3darr = lateral_components(P_vox_3darr, P0_vox_3darr,
                                                                            T_vox_3darr, U_vox_3darr, V_vox_3darr )

        """
        pl = pv.Plotter(window_size=plotter_window_size)
        pl.add_points(P_vox_3darr, color='lime', point_size=5)
        pl.add_points(P0_vox_3darr, color='magenta', point_size=5)
        pl.add_arrows(P_vox_3darr, T_vox_3darr, color='red', mag=1)
        pl.add_arrows(P_vox_3darr, U_vox_3darr, color='blue', mag=1)
        pl.add_arrows(P_vox_3darr, V_vox_3darr, color='green',mag=1)
        pl.add_arrows(P0_vox_3darr, Delta_vox_3darr, color='cyan', mag=1)
        pl.camera_position = 'xy'
        pl.camera.azimuth = 180
        pl.camera.zoom(1.2)
        pl.add_key_event('space', lambda: tool.close_window(pl))
        # ウィンドウ位置を設定（show()の前にrender()を呼び出してrender_windowを初期化）
        pl.render()
        if hasattr(pl, 'render_window') and pl.render_window is not None:
            pl.render_window.SetPosition(plotter_position[0], plotter_position[1])
        try:
            pl.show()
        except (AttributeError, RuntimeError):
            # キーボードイベントでpl.close()が呼ばれた場合、render_windowがNoneになることがある
            pass
        """

        """
        pl = pv.Plotter(window_size=plotter_window_size)
        pl.add_points(P_vox_3darr, color='lime', point_size=5)
        pl.add_points(P0_vox_3darr, color='magenta', point_size=5)
        pl.add_arrows(P_vox_3darr, T_vox_3darr, color='red', mag=1)
        pl.add_arrows(P_vox_3darr, U_vox_3darr, color='blue', mag=1)
        pl.add_arrows(P_vox_3darr, V_vox_3darr, color='green',mag=1)
        pl.add_arrows(P0_vox_3darr, Delta_vox_3darr, color='cyan', mag=1)
        pl.add_arrows(P_vox_3darr, Delta_parT_vox_3darr, color='yellow', mag=5)
        pl.add_arrows(P_vox_3darr, Delta_perpT_vox_3darr, color='black', mag=5)

        # U方向変位ベクトル
        pl.add_arrows(P_vox_3darr, DeltaU_vox_3darr, color='purple', mag=5)
        # V方向変位ベクトル
        pl.add_arrows(P_vox_3darr, DeltaV_vox_3darr, color='orange', mag=5)

        pl.camera_position = 'xy'
        pl.camera.azimuth = 180
        pl.camera.zoom(1.2)
        pl.add_key_event('space', lambda: tool.close_window(pl))
        # ウィンドウ位置を設定（show()の前にrender()を呼び出してrender_windowを初期化）
        pl.render()
        if hasattr(pl, 'render_window') and pl.render_window is not None:
            pl.render_window.SetPosition(plotter_position[0], plotter_position[1])
        try:
            pl.show()
        except (AttributeError, RuntimeError):
            # キーボードイベントでpl.close()が呼ばれた場合、render_windowがNoneになることがある
            pass
        """

        # 同定: sin + trend, 2成分同時に進む
        uU_vox_arr_trimmed = trime_ends(uU_vox_arr, removal)
        uV_vox_arr_trimmed = trime_ends(uV_vox_arr, removal)
 
        """
        # DCT
        m_arr, S_arr, Uc_arr, Vc_arr, N, meta_dct = dct_power_sum(uU_arr=uU_vox_arr_trimmed, uV_arr=uV_vox_arr_trimmed, detrend_deg=detrend_deg)
        k_arr = dct_index_to_k(m_arr, N, ds=ds_vox)

        # ピーク
        m_star, meta_peak = pick_peak_in_band(m_arr, S_arr, k_arr, lambda_prior=lambda_prior_vox)

        # 事前帯域を設定
        mult_range = (0.5, 2.0)
        lam_min = mult_range[0] * lambda_prior_vox
        lam_max = mult_range[1] * lambda_prior_vox
        k_band = (2*np.pi/lam_max, 2*np.pi/lam_min)  # (k_min, k_max)
        print(f"λの推定範囲: {lam_min*voxel_size:.1f} mm, {lam_max*voxel_size:.1f} mm")

        if meta_peak['found'] == False:
            print("===========================================================================")
            print("=> No peak found in prior band")
            class_periodicity = 'Long-wave / Straight-ish / undetectable'
            undetectable_reason = 'no DCT mode in prior band'
            print(f"=> {class_periodicity} / {undetectable_reason}")
            print("===========================================================================")

            data_vessel_segmentdata = {
            "IXI_ID": [case_id],
            "directory_id": [directory_id],
            "segment_id": [segment_id],
            "mca_segment": [mca_segment],
            "side": [side],
            "pts": [pts],
            "pts_trimmed": [pts_trimmed],
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

            # sin+trend（判定はCSV保存後に柔軟に行うため、判定結果は保存しない）
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

            # 追加
            # 3D Constraint Metrics (脳表拘束の証明用)
            "R_out_surface": [np.nan],
            "R_dir_surface": [np.nan],
            "cortex_coverage": [np.nan],
            "constraint_coverage": [np.nan],
            "energy_coverage": [np.nan],
            "total_energy": [np.nan],
            "energy_constrained": [np.nan],
            "perp_energy": [np.nan],
            "n_points_total": [np.nan],
            "n_eval": [np.nan],
            "coh_thr": [np.nan],
            "constraint_coverage_thresh": [np.nan],
            "energy_coverage_thresh": [np.nan],
            "R_out_thresh": [np.nan],
            "stride_mm": [np.nan],
            "min_neighbors": [np.nan],
            "r_patch_mm": [np.nan],
            "gap_fill_size": [np.nan],
            
            # 判定用の閾値（再現性のため）
            "min_length_mm": [np.nan],  # このケースではsin+trend fittingが実行されていないためNaN
            "partial_R2_thr": [np.nan],
            "R2_thr": [np.nan]
            }
        
            # indexとcolumnsを設定してDataFrameを作成
            df_vessel_segmentdata = pd.DataFrame(data_vessel_segmentdata)
            # 個別CSVファイルは保存しない（統合CSVファイルのみ保存）

            # Power Spectrum Plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            # U, V, Sum（m=0:DCは除く）
            m1 = m_arr[1:]
            k1 = k_arr[1:]
            ax.plot(k1, Uc_arr[1:]**2, label='|U|^2 (DCT power)', marker='o')
            ax.plot(k1, Vc_arr[1:]**2, label='|V|^2 (DCT power)', marker='o')
            ax.plot(k1, S_arr[1:],      label='S = |U|^2 + |V|^2', linewidth=2, marker='o')
            # 事前帯域の陰影
            if lambda_prior_vox:
                ax.axvspan(k_band[0], k_band[1], color='grey', alpha=0.15, label='prior band')

            ax.set_xlabel('wavenumber k [rad/voxel]')
            ax.set_ylabel('power')
            ax.legend(loc='upper right')

            k_prior_vox = 2*np.pi / lambda_prior_vox # [rad/voxel]
            ax.axvline(k_prior_vox, color='r', linestyle='--', alpha=0.6)
            ax.annotate(
                f"k_prior={k_prior_vox:.3f} rad/vox\n"
                f"prior λ≈{lambda_prior_vox*voxel_size:.1f} mm", 
                xy=(k_prior_vox, ax.get_ylim()[1]*0.9), 
                xytext=(5, 10),           
                textcoords='offset points',
                color='r',
                fontsize=9,
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color='r', alpha=0.5)
            )

            # 表示に上限をつけてみる。人の目で見える範囲として5mmを基準にする(脳血管の径が1–3mm程度とすると、波長5mmの曲がりは「血管径の1.5〜5倍」)
            lambda_min_mm = 5     # 5 mm → 高周波（細かい）
            k_max = 2*np.pi*voxel_size / lambda_min_mm
            k_max = max(k_max, k_band[1]+0.5) # 5mmよりも小さい範囲までk_bandがカバーする場合に備える
            k_min = 1e-2  # 表示範囲（0から始めない）
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
            plt.tight_layout()
            
            # spaceキーでウィンドウを閉じる
            def on_key(event):
                if event.key == ' ':
                    plt.close(fig)
            
            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.show()

            return True  # 成功

        else:
            print(f"Peak found")
            pass

        # DCT
        # 候補ピークの同定: 帯域内の局所ピークを最大3個ほど抽出（パワー和 S=|U|^2+|V|^2 で上位）
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

        # sin + trend（二成分同時回帰）
        # 信号を「正弦波成分（sin, cos）」＋「トレンド成分（平均、傾き、ゆるい曲線）」で表現するモデル
        # DCTで得た候補kをseedsにする
        k_seeds_vox = [c['k'] for c in cands]         # rad/voxel
        m_seeds = [c['mode'] for c in cands]         # DCT モード番号
        k_prior_vox = 2*np.pi / lambda_prior_vox # 事前（rad/voxel）

        print("=======================================")
        print("Number of seeds:", len(k_seeds_vox))
        print("k_seeds (降順):", k_seeds_vox)
        print("k_prior:", k_prior_vox)
        print("DCTモード番号 m_seeds:", m_seeds)

        # Grid search with sin + trend fitting
        # 最適な波数 k を「粗→細」で探す
        print("=======================================")
        print("Grid search with sin + trend fitting")
        k_grid_search_result = grid_search_k(uU=uU_vox_arr_trimmed, uV=uV_vox_arr_trimmed, s=s_vox_arr_trimmed, N=N, ds=ds_vox,
                  k_seeds=k_seeds_vox,
                  pct_coarse=0.30, pct_fine=0.10,
                  B_bins_coarse=1.5, B_bins_fine=0.5,
                  step_frac=0.25,
                  weights=None,
                  trend_degree=1,
                  lambda_prior=lambda_prior_vox
                  )
        k_hat_vox = k_grid_search_result['k_hat']
        lambda_hat_vox = k_grid_search_result['lambda_hat']
        A_comb_hat_vox = k_grid_search_result['A_comb']
        A_u_hat_vox = k_grid_search_result['A_U']
        A_v_hat_vox = k_grid_search_result['A_V']
        uU_cos_vox = k_grid_search_result['uU_cos']
        uU_sin_vox = k_grid_search_result['uU_sin']
        uV_cos_vox = k_grid_search_result['uV_cos']
        uV_sin_vox = k_grid_search_result['uV_sin']

        print("Grid search result:")
        print(f" k_hat: {k_hat_vox :.2f} rad/vox, {k_hat_vox / voxel_size :.2f} rad/mm")
        print(f" lambda_hat: {lambda_hat_vox:.2f} voxel, {lambda_hat_vox*voxel_size:.2f} mm")
        print(f" amplitude of wave: {A_comb_hat_vox :.2f} voxel, {A_comb_hat_vox *voxel_size:.2f} mm")

        R2 = k_grid_search_result['R2']
        partial_R2 = k_grid_search_result['partial_R2']
        hr_amp = k_grid_search_result['hr_amp']
        hr_energy = k_grid_search_result['hr_energy']

        print(f" R2: {R2:.2f}")
        print(f" partial R2: {partial_R2:.2f}")
        print(f" hr(E3/E1)_amp={hr_amp:2f}, hr(E3/E1)_energy={hr_energy:.2f}")

        cand1 = k_grid_search_result['candidates_coarse']
        cand2 = k_grid_search_result['candidates_fine']

        print("===========================================================================")
        alpha_hat = k_grid_search_result['lambda_hat'] / radius_median_vox
        print(f"alpha_hat: {alpha_hat:.4f}")
        print(f"alpha_prior: {alpha_prior:.4f}")

        coverage_hat = L_vox_trimmed / k_grid_search_result['lambda_hat']
        print(f"coverage_hat: {coverage_hat:.4f}")

        SSE_full = k_grid_search_result['fit']['SSE_full']
        SSE_tr = k_grid_search_result['fit']['SSE_tr']
        print(f"SSE_full (sin+trend): {SSE_full}")
        print(f"SSE_tr (trend-only): {SSE_tr}") 


        k_course = [c['k'] for c in cand1]
        r2_course = [c['R2'] for c in cand1]
        partial_r2_course = [c['partial_R2'] for c in cand1]
        hr_amp_course = [c['hr_amp'] for c in cand1]
        k_fine = [c['k'] for c in cand2]
        r2_fine = [c['R2'] for c in cand2]
        partial_r2_fine = [c['partial_R2'] for c in cand2]
        hr_amp_fine = [c['hr_amp'] for c in cand2]


        # Grid search result plot
        fig = plt.figure(figsize=(8,5))
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
        # Segment lengthの情報も入れておく
        k_L = 2*np.pi / L_vox_trimmed # [rad/voxel]
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


        plt.title(f"Grid Search Result for segment ID. {segment_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_per_case, f"{directory_id}_Grid_search_result_{segment_id}.png"), format='png', dpi=300)
        plt.close()



        # DCT with peak and grid search result plot
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
        ax.axvline(k_L, color='purple', linestyle=':', alpha=0.7)
        ax.annotate(
        f"segment length (trimmed)= {L_vox_trimmed * voxel_size:.1f} mm\nk_L = {k_L:.3f} rad/vox",
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
            f"λ*≈{k_grid_search_result['lambda_hat']*voxel_size:.1f} mm",   # 表示する文字
            xy=(k_hat_vox, ax.get_ylim()[1]*0.8),             # 矢印の基点（x=k_hat_vox, y=縦軸の80%付近）
            xytext=(-80, -30),                                # 文字のオフセット [pixel単位]
            textcoords='offset points',
            color='blue',
            fontsize=9,
            arrowprops=dict(arrowstyle='-|>', lw=0.8, color='blue', alpha=0.5)  # 矢印スタイル
        )

        ax.set_title(f"DCT with Grid Search Result for segment ID. {segment_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_per_case, f"{directory_id}_DCT_with_grid_search_result_{segment_id}.png"), format='png', dpi=300)
        plt.close()


        # 事前帯域の境界を超えたかどうかのチェック
        if k_hat_vox <= k_band[0] + 1e-6:
            print("Boundary hit at k_low")
            boundary_hit_low = True; boundary_hit_high = False
        elif k_hat_vox >= k_band[1] - 1e-6:
            print("Boundary hit at k_high")
            boundary_hit_low = False; boundary_hit_high = True
        else:
            print("No boundary hit")
            boundary_hit_low = False; boundary_hit_high = False
        """

        # Sinusoidal plus trend fitting using least squares (DCTは用いない)
        k_grid_search_result = grid_search_k_band(uU=uU_vox_arr_trimmed, uV=uV_vox_arr_trimmed, s=s_vox_arr_trimmed, ds=ds_vox,
                                                    N_cyc=N_cyc,
                                                    N_pp=N_pp,
                                                    ds_eff=ds_eff,
                                                    q_fine=q_fine,
                                                    weights=None,
                                                    trend_degree=trend_degree,
                                                    dR2_tol=dR2_tol,
                                                    dhr_amp_tol=dhr_amp_tol,
                                                    boundary_margin_bins=boundary_margin_bins,
                                                    verbose=True
                                                )
        k_hat_vox = k_grid_search_result['k_hat']
        lambda_hat_vox = k_grid_search_result['lambda_hat']

        A_comb_hat_vox = k_grid_search_result['A_comb']
        A_u_hat_vox  = k_grid_search_result['A_U']
        A_v_hat_vox = k_grid_search_result['A_V']
        uU_cos_vox = k_grid_search_result['uU_cos']
        uU_sin_vox = k_grid_search_result['uU_sin']
        uV_cos_vox = k_grid_search_result['uV_cos']
        uV_sin_vox = k_grid_search_result['uV_sin']
        
        R2 = k_grid_search_result['R2']
        partial_R2 = k_grid_search_result['partial_R2']
        hr_amp = k_grid_search_result['hr_amp']
        hr_energy = k_grid_search_result['hr_energy']


        coverage_hat = L_vox_trimmed / lambda_hat_vox 

        print("Grid search result:")
        print(f" k_hat: {k_hat_vox :.2f} rad/vox, {k_hat_vox / voxel_size :.2f} rad/mm")
        print(f" lambda_hat: {lambda_hat_vox:.2f} voxel, ***{lambda_hat_vox*voxel_size:.2f} mm***")
        print(f" amplitude of wave: {A_comb_hat_vox :.2f} voxel, {A_comb_hat_vox *voxel_size:.2f} mm")
        print(f" R2: {R2:.2f}")
        print(f" partial R2: {partial_R2:.2f}")
        print(f" hr(E3/E1)_amp={hr_amp:2f}, hr(E3/E1)_energy={hr_energy:.2f}")
        print(f" coverage_hat: {coverage_hat:.4f}")
        
        SSE_full = k_grid_search_result['fit']['SSE_full']
        SSE_tr = k_grid_search_result['fit']['SSE_tr']
        print(f"SSE_full (sin+trend): {SSE_full}")
        print(f"SSE_tr (trend-only): {SSE_tr}") 
        print("\n" + "="*40)


        # Boundary hit 判定
        boundary_hit_low = k_grid_search_result['boundary_hit_low']
        boundary_hit_high = k_grid_search_result['boundary_hit_high']
        if boundary_hit_low:
            print("Boundary hit at k_low")
        elif boundary_hit_high:
            print("Boundary hit at k_high")
        else:
            print("No boundary hit")
        print("\n" + "="*40)

        # 残差をノイズの推定値として扱う
        resid_full = k_grid_search_result['fit']['resid_full']
        resid_tr = k_grid_search_result['fit']['resid_tr']

        rms_residual_full = np.sqrt(np.mean(resid_full**2))
        rms_residual_tr = np.sqrt(np.mean(resid_tr**2))

        print("セグメントのノイズスケール")
        print(f"rms_residual_full [mm]: {rms_residual_full * voxel_size:.2f}") # sin+trendモデルでは説明できなかった残差のRMS
        print(f"rms_residual_tr [mm]: {rms_residual_tr * voxel_size:.2f}")

        print("つまり")
        print(f"trend-only だと「平均 ±{rms_residual_tr * voxel_size:.2f} mm くらいズレている」")
        print(f"sin+trend を入れると「そのズレが ±{rms_residual_full * voxel_size:.2f} mm くらいまで減る」")

        print(f"sin+trendで取れなかった残りの揺れは、 {rms_residual_full:.2f} voxel くらいの振幅")

        rms_A_comb_hat_vox = A_comb_hat_vox / np.sqrt(2)
        print(f"振幅A: {A_comb_hat_vox*voxel_size:.2f} mm, 振幅AのRMS: {rms_A_comb_hat_vox*voxel_size:.2f} mm")
        SNR = rms_A_comb_hat_vox / rms_residual_full
        print(f"SNR: {SNR:.2f}")
        print("\n" + "="*40)

        """
        # ここで言う「ノイズ」   
        MRA画像のボクセル解像度由来のギザギザ（0.5 mm ステップ）.  
        セグメンテーション誤差・centerline 抽出の揺れ・座標変換/補間の誤差.  
        frame（T,U,V）のわずかなjitter(信号の揺らぎ).  
        モデル化しなかった高調波・細かい蛇行.  
        本当にランダムな measurement noise.  
        が全部まとめて入っている.  

        # A のしきい値について.  
        振幅 A は「信号の大きさ」なので、SNR = A_RMS / noise_RMS を考えると.  
        SNR  0.5 未満 -> ノイズの1/2以下.  
        SNR  1.0 前後 -> ノイズと同じオーダー.  
        SNR  2.0 以上 -> ノイズの 2 倍以上で、目視しても“揺れている”のがわかりやすい.  
        => SNR > 1.5を基準とする.  
        A_hatが1 voxel(=0.5mm)未満の “揺れ”であった場合に、血管の形態とみなすか否か問題もあるので.  
        A_hat > 1 voxel (0.5mm)の条件もつける
        """

        # inclusion (序盤でexclusion [<10mm]はすでに判定済み)
        # 1) Long-wave / Straight-ish / detectable判定 (あやしい・座屈として語りたくない)
        """
        # partial_R2 の閾値をどう決めるか
        F検定を使用して求める
        temporary noteの"9. "trend-only model"と"trend + sin model"の比較をpatial F検定で行う"を参照

        $\boxed{
        partial \,{R^2} =
        \frac{F df_1}{F df_1 + df_2}}$

        df_1 = 第1自由度（分子の自由度）つまり 2.  
        df_2 = 第2自由度（分母の自由度）つまり N - 4.  

        from scipy.stats import f
        from scipy.optimize import fsolve
        df1 = 2
        df2 = 100-4


        Fcrit = f.ppf(0.95, dfn=df1, dfd=df2)
        print(f"F値alpha=0.05(分子df1={df1}, 分母df2={df2}): {Fcrit:.3f}")

        def func(x):
            return (x/(1-x))*(df2/df1) - Fcrit

        sol = fsolve(func, 0.05)
        partial_R2_thr = sol[0]
        print(f"この場合のpartial_R2_thr: {partial_R2_thr:.3f}")
        >>> F値alpha=0.05(分子df1=2, 分母df2=96): 3.091
        >>> この場合のpartial_R2_thr: 0.061
        """
        # 判定ロジック（表示用のみ。CSV保存後に柔軟に判定できるように、ここでは判定結果を保存しない）
        CLASS_UNDET = 'Long-wave / Straight-ish / undetectable'
        CLASS_CORE = 'Core'
        CLASS_SEC = 'Secondary'

        # 表示用の判定（CSVには保存しない）
        undetectable_reason = None
        if boundary_hit_low and coverage_hat < 1.0:
            class_periodicity_display = CLASS_UNDET
            undetectable_reason = "boundary_hit_low and coverage_hat"
            print("coverageが1周未満で探索下限に張り付いた")
        elif coverage_hat < 0.5:
            class_periodicity_display = CLASS_UNDET
            undetectable_reason = "coverage_hat < 0.5"
            print("半周期未満")
        elif A_comb_hat_vox < 1:
            class_periodicity_display = CLASS_UNDET
            undetectable_reason = "A_comb_hat_vox < 1"
            print("振幅Aが解像度未満")
        elif SNR < 1.5:
            class_periodicity_display = CLASS_UNDET
            undetectable_reason = "SNR < 1.5"
            print("SNRが1.5未満でノイズに弱い")
        elif partial_R2 < partial_R2_thr:
            class_periodicity_display = CLASS_UNDET    
            undetectable_reason = "partial_R2 < partial_R2_thr"
            print("F検定的に「sinの寄与がほぼゼロ」")
        elif R2 < R2_thr:
            class_periodicity_display = CLASS_UNDET  
            undetectable_reason = f"R2 < {R2_thr}"
            print("モデル自体の当てはまりが悪い")     
        elif coverage_hat >= 1.5 and partial_R2 >= 0.15 and R2 >= 0.6 and hr_amp <= 0.2:
            class_periodicity_display = CLASS_CORE
            undetectable_reason = None
            print("Core: ")
            print(f"coverage_hat: {coverage_hat:.3f}")
            print(f"R2: {R2:.3f}")
            print(f"partial R2: {partial_R2:.3f}")
            print(f"hr_amp: {hr_amp:.3f}")
        else:
            class_periodicity_display = CLASS_SEC
            undetectable_reason = None
            print("Secondary: ")
            print(f"coverage_hat: {coverage_hat:.3f}")
            print(f"R2: {R2:.3f}")
            print(f"partial R2: {partial_R2:.3f}")
            print(f"hr_amp: {hr_amp:.3f}")

        
        print(f"=> {class_periodicity_display}")
        print(f"undetectable reason: {undetectable_reason}")
        print("\n" + "="*60)

        # Plot the result
        ks = k_grid_search_result['ks']
        R2s = np.array([c['R2'] for c in k_grid_search_result['candidates']], float)
        partial_R2s = np.array([c['partial_R2'] for c in k_grid_search_result['candidates']], float)
        hr_amps = np.array([c['hr_amp'] for c in k_grid_search_result['candidates']], float)
        hr_energies = np.array([c['hr_energy'] for c in k_grid_search_result['candidates']], float)
        k_min = k_grid_search_result['k_min']
        k_max = k_grid_search_result['k_max']
        dk = k_grid_search_result['dk']

        # R2が同等の候補（tie set）を計算
        R2_max = R2s.max()
        tie_mask = (R2_max - R2s) <= dR2_tol
        tie_ks = ks[tie_mask]
        tie_R2s = R2s[tie_mask]
        tie_hr_amps = hr_amps[tie_mask]

        # hr_ampが同等の候補（tie2 set）を計算（tie set内で）
        if len(tie_ks) > 0:
            tie_hr_amps_valid = tie_hr_amps[np.isfinite(tie_hr_amps)]
            if len(tie_hr_amps_valid) > 0:
                hr_amp_min = np.nanmin(tie_hr_amps)
                tie2_mask = tie_mask & (np.isfinite(hr_amps)) & ((hr_amps - hr_amp_min) <= dhr_amp_tol)
                tie2_ks = ks[tie2_mask]
                tie2_R2s = R2s[tie2_mask]
                tie2_hr_amps = hr_amps[tie2_mask]
            else:
                tie2_ks = np.array([])
                tie2_R2s = np.array([])
                tie2_hr_amps = np.array([])
        else:
            tie2_ks = np.array([])
            tie2_R2s = np.array([])
            tie2_hr_amps = np.array([])

        print(f"k_min: {k_min:.6g} rad/voxel")
        print(f"k_max: {k_max:.6g} rad/voxel")
        print(f"dk: {dk:.6g} ")
        print("\n" + "="*40)

        fig = plt.figure(figsize=(8,5))
        ax1 = fig.add_subplot(111) 
        # 事前帯域の陰影
        ax1.axvspan(k_min, k_max, color='grey', alpha=0.15, label='prior band')
        ax1.set_xlabel('Wave number k [rad/voxel]')
        ax1.set_ylabel('R2')

        ax1.plot(ks, R2s, 'o-', label='R2 search', color='C0', markersize=4)
        ax1.plot(ks, partial_R2s, 'x-', label='partial R2 search', color='C1', markersize=4)
        
        # R2が同等の候補を視覚化
        if len(tie_ks) > 0:
            ax1.scatter(tie_ks, tie_R2s, s=100, facecolors='none', edgecolors='C0', linewidths=2, 
                       marker='o', label=f'R2 equivalent (ΔR² ≤ {dR2_tol})')
        
        # hr_ampが同等の候補を視覚化
        if len(tie2_ks) > 0:
            ax1.scatter(tie2_ks, tie2_R2s, s=150, facecolors='none', edgecolors='C2', linewidths=2.5, 
                       marker='s', label=f'hr_amp equivalent (Δhr_amp ≤ {dhr_amp_tol})')

        ax1.axvline(k_hat_vox, color='b', linestyle='-', alpha=0.5)
        ax1.annotate(
            f"Selected k* = {k_hat_vox:.3f} rad/vox", 
            xy=(k_hat_vox, ax1.get_ylim()[1]*0.95), 
            xytext=(5, 10),                                     # 文字のオフセット [pixel単位]
            textcoords='offset points',
            color='b',
            fontsize=9,
            arrowprops=dict(arrowstyle='-|>', lw=0.8, color='b', alpha=0.5)  # 矢印スタイル
        )

        # Segment lengthの情報も入れておく
        k_L = 2*np.pi / L_vox_trimmed # [rad/voxel]
        ax1.axvline(k_L, color='purple', linestyle=':', alpha=0.7)
        ax1.annotate(
            f"k_L = {k_L:.3f} rad/vox",
            xy=(k_L, ax1.get_ylim()[1] * 0.95),
            xytext=(5, 10),
            textcoords='offset points',
            color='purple',
            fontsize=9,
            arrowprops=dict(arrowstyle='-|>', lw=0.8, color='purple', alpha=0.5)
        )

        ax2 = ax1.twinx()
        ax2.plot(ks, hr_amps, 's--', label='Harmonic ratio amp_(E3/E1)', color='C2', markersize=4)
        
        # hr_ampが同等の候補をhr_amp軸でも視覚化
        if len(tie2_ks) > 0:
            ax2.scatter(tie2_ks, tie2_hr_amps, s=150, facecolors='none', edgecolors='C2', linewidths=2.5, 
                       marker='s')
        
        ax2.set_ylabel('Harmonic ratio amp_(E3/E1)', color='C0')
        ax2.tick_params(axis='y', labelcolor='C0')

        handler1, label1 = ax1.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()
        ax1.legend(handler1 + handler2, label1 + label2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)


        plt.title(f"Grid Search Result, segment ID: {segment_id}, segment index: {cell_index}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_per_case, f"{directory_id}_grid_search_result_{segment_id}_{cell_index}.png"), format='png', dpi=300)
        # tool.setup_matplotlib_close_on_space(fig)
        # plt.show()
        plt.close()


        """
        線形代数の基本事実: 「円に任意の線形変換（2×2行列）をかけると、図形は一般に “楕円” になる」.  

        sin+trend fittingの"sin成分だけ"をとりだすと.  
        $$
        h_U(s) = a_U\cos(ks) + b_U\sin(ks),\\
        h_V(s) = a_V\cos(ks) + b_V\sin(ks)
        $$
        これをまとめて

        $$
        \mathbf{h}(s)
        =
        \begin{pmatrix}
        h_U(s) \\
        h_V(s)
        \end{pmatrix}
        =
        \begin{pmatrix}
        a_U & b_U \\
        a_V & b_V
        \end{pmatrix}
        \begin{pmatrix}
        \cos(ks) \\
        \sin(ks)
        \end{pmatrix}
        =
        M
        \begin{pmatrix}
        \cos(ks) \\
        \sin(ks)
        \end{pmatrix}.
        $$

        と書ける（Mは2×2の行列）。右側の$\begin{pmatrix}
        \cos(ks) \\
        \sin(ks)
        \end{pmatrix}$は円なので、それに行列Mをかけると、「円を線形変換したもの」になる。つまり、線形変換された円=楕円。  

        $h(s)$はU–V 平面の中で 楕円状の軌跡を描き、「横ずれベクトルの先端が、その平面内でどう動くか」を見ている.  
        もし本物の 1D buckling（平面内で一方向だけに蛇行）なら.  
        => 「横ずれの向き」はほとんど変わらず、「大きさ（符号含む）」だけが変わる.  
        => つまり U–V 平面内でみるとほぼ 1 本の直線上を行ったり来たりする.  
        もし 3D っぽくグネグネ回る（軸の周りを回るヘリックスなど）なら. 
        => 横ずれの向き自体もぐるぐる回る.  
        => U–V 平面内では 円に近い軌跡になる.  
        2×2共分散行列の固有値比λ2/λ1で測っている、というのがdirectionality 
        """
        # hU, hVを計算(sin+trend fittingの"sin成分だけ"をとりだす)
        hU_vox = uU_cos_vox * np.cos(k_hat_vox * s_vox_arr_trimmed) + uU_sin_vox * np.sin(k_hat_vox * s_vox_arr_trimmed)
        hV_vox = uV_cos_vox * np.cos(k_hat_vox * s_vox_arr_trimmed) + uV_sin_vox * np.sin(k_hat_vox * s_vox_arr_trimmed)

        hU_mm = hU_vox * voxel_size
        hV_mm = hV_vox * voxel_size

        hUV_mm = np.column_stack([hU_mm, hV_mm])

    
        # ====================================================================================================
        # ====================================================================================================
        # PCAを利用した解析は保険
        # PCAを実行
        pca_2d = PCA(n_components=2) # PCAを行ったり、PCAの結果を格納したりするための変数を、pcaとして宣言
        pca_2d.fit(hUV_mm) # PCAを実行

        eigenvalues_2d = pca_2d.explained_variance_
        eigenvalue1, eigenvalue2 = eigenvalues_2d

        directionality = eigenvalue2 / eigenvalue1        # 0〜1 の比
        print(f"directionality: {directionality:.4f}")
        # directionality_2d = λ2 / λ1 << 1 は BishopのU–V 座標系の中では、sin成分がほぼ1本の線状に並んでいる => 1D buckling candidateの状態を表す
        print("\n" + "="*40)

        # PCAから固有ベクトルを取得
        vecs_2d = pca_2d.components_   # shape (2,2), 各行が固有ベクトル（λ1, λ2に対応）
        g1_uv = vecs_2d[0]         # これが U–V 平面内の g1_hat（単位ベクトル）
        g2_uv = vecs_2d[1]         # λ2 に対応する方向
        # ====================================================================================================
        # ====================================================================================================


        # --- SVD-based principal axes (equivalent to PCA for a single-frequency sinusoid) ---
        # Coefficient matrix of the sinusoidal component in mm:
        # h(s) = M [cos(ks), sin(ks)]^T
        M_mm = np.array([
            [uU_cos_vox * voxel_size, uU_sin_vox * voxel_size],
            [uV_cos_vox * voxel_size, uV_sin_vox * voxel_size],
            ], dtype=float)

        # Singular value decomposition: M = U Σ V^T
        # Columns of U give the major/minor axis directions in the U–V plane.
        U_mat, S_vals, Vt_mat = np.linalg.svd(M_mm, full_matrices=True)
        sigma1, sigma2 = float(S_vals[0]), float(S_vals[1])

        # Directions in the U–V plane (unit vectors)
        g1 = U_mat[:, 0].copy()
        g2 = U_mat[:, 1].copy()
        # Sign convention for stability (optional): make g1[0] non-negative
        if g1[0] < 0:
            g1 *= -1
            g2 *= -1

        # Directionality: ratio of minor/major variance in the U–V plane
        # For the sinusoidal ellipse, variance ratio = (sigma2^2) / (sigma1^2)
        directionality_svd = (sigma2 / sigma1)**2

        print(f"directionality (SVD-based): {directionality_svd:.4f}")
        print(f"sigma1={sigma1:.4f} mm, sigma2={sigma2:.4f} mm")

        tab10 = plt.get_cmap('tab10').colors  # 10色のタプルのリスト

        
        center = (0,0) # 原点中心に変更 # center = hUV_mm.mean(axis=0)

        # hU-hV sin component plot
        plt.figure(figsize=(10,10))
        plt.plot(hU_mm[0], hV_mm[0], "x", color="red", alpha=0.7)
        plt.plot(hU_mm[-1], hV_mm[-1], "o", color="red", alpha=0.7)
        plt.plot(hU_mm, hV_mm, alpha=0.7)
        for i in range(2):
            # 固有値の平方根をスケールとして使う
            length = 2.0 * np.sqrt(eigenvalues_2d[i])
            v = vecs_2d[i] * length
            plt.plot([center[0], center[0] + v[0]],
                     [center[1], center[1] + v[1]],
                     lw=2, color=tab10[i+1])
        
        # SVD-based principal axes plot
        plt.plot([center[0], center[0] + g1[0]],
         [center[1], center[1] + g1[1]],
         lw=2, color='orange', linestyle='--', label='SVD major axis')
        plt.plot([center[0], center[0] + g2[0]],
                [center[1], center[1] + g2[1]],
                lw=2, color='green', linestyle='--', label='SVD minor axis')

        # --- Major axis (±σ1 e1) ---
        max1_pos = sigma1 * g1
        max1_neg = -sigma1 * g1
        plt.plot(max1_pos[0], max1_pos[1], marker='*', markersize=12, color='black', label='Max amplitude major')
        plt.plot(max1_neg[0], max1_neg[1], marker='*', markersize=12, color='black', label='Max amplitude major')

        # --- Minor axis (±σ2 e2) ---
        max2_pos = sigma2 * g2
        max2_neg = -sigma2 * g2
        plt.plot(max2_pos[0], max2_pos[1], marker='^', markersize=12, color='black', label='Max amplitude minor')
        plt.plot(max2_neg[0], max2_neg[1], marker='^', markersize=12, color='black', label='Max amplitude minor')

        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel('hU [mm]')
        plt.ylabel('hV [mm]')
        plt.title(f'hU-hV Sin Component (directionality={directionality_svd:.3f}), segment ID: {segment_id}, segment index: {cell_index}')
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(output_dir_per_case, f"{directory_id}_hU-hV_sin_component_{segment_id}_{cell_index}.png"), format='png', dpi=300)
        plt.close()

        # ====================================================================================================
        # ====================================================================================================
        # To keep compatibility with previous PCA outputs (explained_variance_ has units mm^2):
        # If sampling over phase is approximately uniform, cov(h) = 0.5 * M M^T.
        eigval1_svd_mm2 = 0.5 * sigma1**2
        eigval2_svd_mm2 = 0.5 * sigma2**2

        print("\n" + "="*40)
        print(f"eigval1 (PCA): {eigenvalue1:.4f} mm^2, eigval2 (PCA): {eigenvalue2:.4f} mm^2, directionality (PCA): {directionality:.4f}")
        print(f"eigval1 (SVD): {eigval1_svd_mm2:.4f} mm^2, eigval2 (SVD): {eigval2_svd_mm2:.4f} mm^2, directionality (SVD): {directionality_svd:.4f}")
        print("\n" + "="*40)
        # ====================================================================================================
        # ====================================================================================================

        data_vessel_segmentdata = {
        "IXI_ID": [case_id],
        "directory_id": [directory_id],
        "segment_id": [segment_id],
        "segment_index": [cell_index],
        "mca_segment": [mca_segment],
        "side": [side],
        "pts": [pts],
        "pts_trimmed": [pts_trimmed],
        "ds_mm": [ds_vox * voxel_size],
        "L_full_mm": [s_vox_arr[-1] * voxel_size],
        "L_trimmed_mm": [L_vox_trimmed * voxel_size],
        "radius_median_mm": [radius_median_vox * voxel_size],
        "radius_mean_mm": [radius_mean_vox * voxel_size],
        "radius_std_mm": [radius_std_vox * voxel_size],
        "euclid_distance_mm": [euclid_d_vox * voxel_size],
        "TI": [TI],
        "rho": [rho],


        # Curvature(kappa)系
        "kappa_median_mm": [kappa_median_vox / voxel_size],
        "kappa_mean_mm": [kappa_mean_vox / voxel_size],
        "kappa_std_mm": [kappa_std_vox / voxel_size],
        "kappa_rms_mm": [kappa_rms_vox / voxel_size],
        "kappa_p95_mm": [kappa_p95_vox / voxel_size],
        "kappa_max_mm": [kappa_max_vox / voxel_size],
        "kappa_min_mm": [kappa_min_vox / voxel_size],

        # 局所的な形態量
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

        # sin+trend（判定はCSV保存後に行うため、判定結果は保存しない）
        "k_hat_mm": [k_hat_vox / voxel_size],
        "lambda_hat_mm": [lambda_hat_vox * voxel_size],
        "amp_hat_mm": [A_comb_hat_vox * voxel_size],
        "amp_hat_U_mm": [A_u_hat_vox * voxel_size],
        "amp_hat_V_mm": [A_v_hat_vox * voxel_size],
        "coef_uU_cos_mm": [uU_cos_vox * voxel_size], 
        "coef_uU_sin_mm": [uU_sin_vox * voxel_size], 
        "coef_uV_cos_mm": [uV_cos_vox * voxel_size], 
        "coef_uV_sin_mm": [uV_sin_vox * voxel_size],
        "eigval1_2d_mm": [sigma1],
        "eigval2_2d_mm": [sigma2],
        "directionality": [directionality_svd],
        "amp_hat_over_L": [A_comb_hat_vox / L_vox_trimmed], # 無次元量
        "lambda_hat_over_radius": [lambda_hat_vox / radius_median_vox], # 無次元量
        "amp_hat_over_radius": [A_comb_hat_vox / radius_median_vox], # 無次元量
        "R2": [R2],
        "partial_R2": [partial_R2],
        "SSE_full": [SSE_full],
        "SSE_tr": [SSE_tr],
        "hr_amp": [hr_amp],
        "hr_energy": [hr_energy],
        "coverage_hat": [coverage_hat],
        "SNR": [SNR],
        "rms_residual_full_mm": [rms_residual_full * voxel_size],
        "rms_residual_tr_mm": [rms_residual_tr * voxel_size],
        "boundary_hit_low": [int(boundary_hit_low)],
        "boundary_hit_high": [int(boundary_hit_high)],

        # 3D Constraint Metrics (脳表拘束の証明)
        "R_out_surface": [np.nan],
        "R_dir_surface": [np.nan],
        "cortex_coverage": [np.nan],
        "constraint_coverage": [np.nan],
        "energy_coverage": [np.nan],
        "total_energy": [np.nan],
        "energy_constrained": [np.nan],
        "perp_energy": [np.nan],
        "n_points_total": [np.nan],
        "n_eval": [np.nan],
        "coh_thr": [np.nan],
        "constraint_coverage_thresh": [np.nan],
        "energy_coverage_thresh": [np.nan],
        "R_out_thresh": [np.nan],
        "stride_mm": [np.nan],
        "min_neighbors": [np.nan],
        "r_patch_mm": [np.nan],
        "gap_fill_size": [np.nan],
        
        # 判定用の閾値（再現性のため）
        "min_length_mm": [min_length_mm],
        "partial_R2_thr": [partial_R2_thr],
        "R2_thr": [R2_thr]
        }

        # indexとcolumnsを設定してDataFrameを作成
        df_vessel_segmentdata = pd.DataFrame(data_vessel_segmentdata)

        # 判定ロジック（表示用）に基づいて、undetectableの場合は早期リターン
        # 判定はCSV保存後に柔軟に行えるが、処理効率のためここで判定
        if class_periodicity_display == CLASS_UNDET:
            # DataFrameを返す（個別CSVファイルは保存しない）
            return df_vessel_segmentdata

        # Core または Secondary の場合のみ、脳表拘束の証明を最後に行う
        # sin + trend fittingの"sin成分だけ"をとりだす
        U_vox_3darr_trimmed = trime_ends(U_vox_3darr, removal)
        V_vox_3darr_trimmed = trime_ends(V_vox_3darr, removal)

        # 3D変位（voxel）= (U方向の振幅) * (U方向ベクトル) + (V方向の振幅) * (V方向ベクトル)
        dPsin_vox_3darr_trimmed = (hU_vox[:, None] * U_vox_3darr_trimmed + hV_vox[:, None] * V_vox_3darr_trimmed)

        # 変位を元点列に足す（voxel空間の座標）
        Psin_vox_3darr_trimmed = P_vox_3darr_trimmed + dPsin_vox_3darr_trimmed

        """
        # UV平面での第一主成分方向に沿った変位成分も抽出する(PCAを再実行する必要はない。g1_uv, g2_uvは単位ベクトルなので、そのまま使える)
        gU, gV = g1_uv   # 2D PCA の第一主成分方向ベクトル成分
        g1_3d = gU * U_vox_3darr_trimmed + gV * V_vox_3darr_trimmed
        g1_3d = normalize(g1_3d)
        """
        
        # SVD-based principal axes
        gU, gV = g1[0], g1[1]
        g1_3d = gU * U_vox_3darr_trimmed + gV * V_vox_3darr_trimmed
        g1_3d = normalize(g1_3d) # 数値的安定性と安全性のためnormalize(理論上は正規化は不要)

        # 関心領域のみ抽出
        cropped_mesh = clip_mesh_to_coords(P_vox_3darr, brain_mesh)

        pl = pv.Plotter(window_size=plotter_window_size)
        pl.add_points(P_vox_3darr, color='lime', point_size=5, render_points_as_spheres=True, label='Original centerline')
        pl.add_points(P0_vox_3darr, color='magenta', point_size=5, render_points_as_spheres=True, label='Baseline centerline')
        pl.add_arrows(P_vox_3darr_trimmed, dPsin_vox_3darr_trimmed, color='orange', mag=1, label='Transverse displacement (buckling component)')
        pl.add_arrows(P_vox_3darr_trimmed, g1_3d, mag=1, color='yellow', label='Principal buckling direction (g1)')
        pl.add_mesh(cropped_mesh, color="#fafae1", opacity=0.5)
        # 法線表示
        normal_arrows = cropped_mesh.glyph(
            orient="Normals",
            scale=False,
            factor=0.5
        )
        pl.add_mesh(normal_arrows, color="navy", opacity=0.7)

        pl.camera_position = 'xy'
        pl.camera.azimuth = 180
        pl.camera.zoom(1.2)
        pl.add_legend()
        pl.add_title(f'Buckling Visualization, segment ID: {segment_id}, segment index: {cell_index}', font_size=18)
        pl.add_key_event('space', lambda: tool.close_window(pl))
        pl.render()
        if hasattr(pl, 'render_window') and pl.render_window is not None:
            pl.render_window.SetPosition(plotter_position[0], plotter_position[1])
        
        # HTMLファイルとして出力
        html_output_path = os.path.join(output_dir_per_case, f"{directory_id}_buckling_visualization_{segment_id}_{cell_index}.html")
        try:
            pl.export_html(html_output_path)
        except Exception as e:
            print(f"Warning: HTMLファイルの保存に失敗した: {str(e)}")
        
        try:
            # pl.show()
            pl.close()
        except (AttributeError, RuntimeError):
            pass

        # 脳表のパッチをつくる
        radius_median_mm = radius_median_vox * voxel_size
        print(f"血管半径の中央値: {radius_median_mm:.2f} mm")
        r_patch_mm_raw = radius_median_mm + radius_median_mm * patch_r_ratio

        # さらに過度に小さい/大きいのをクリップ
        r_patch_mm = np.clip(r_patch_mm_raw, r_min_mm, r_max_mm)
        r_patch_vox = r_patch_mm / voxel_size
        print(f"脳表パッチの半径: {r_patch_mm:.2f} mm (raw: {r_patch_mm_raw:.2f} mm)")

        # 脳表拘束の証明
        print("Analyzing surface constraint...")
        surface_constraint_result = analyze_surface_constraint_coherence(
            P_vox_3darr_trimmed,             # (N,3) voxel座標（trim後推奨）
            dPsin_vox_3darr_trimmed,         # (N,3) voxel座標の変位（trim後）
            g1_3d,
            ds_vox,                          # 1点あたり弧長ステップ（voxel）
            cropped_mesh,                    # compute_normals済みの脳表メッシュ
            r_patch_vox,
            stride=stride,
            min_neighbors=min_neighbors,
            coh_thr=coh_thr,
            constraint_coverage_thresh=constraint_coverage_thresh,
            energy_coverage_thresh=energy_coverage_thresh,
            R_out_thresh=R_out_thresh,
            gap_fill_size=gap_fill_size,
            verbose=True)

        # Visualization
        anchor_neighbors_idx = surface_constraint_result['anchor_neighbors_idx']
        anchor_neighbors_idx = np.concatenate([np.array(lst) for lst in anchor_neighbors_idx if len(lst)>0])
        anchor_neighbors_idx = np.unique(anchor_neighbors_idx) # 重複に対応

        anchor_neighbors_pts = cropped_mesh.points[anchor_neighbors_idx]
        anchor_neighbors_normals = cropped_mesh.point_data["Normals"][anchor_neighbors_idx]

        normals_coherence = surface_constraint_result['normals_full']
        anchor_pts = surface_constraint_result["anchor_pts"]
        anchor_has_constraint = surface_constraint_result["anchor_has_constraint_clean"]

        # Visualization of surface constraint
        pl = pv.Plotter(window_size=plotter_window_size)
        # 1. 基本の血管形状&脳表
        pl.add_points(P_vox_3darr_trimmed, color='lime', point_size=5, label='Centerline')
        pl.add_points(P0_vox_3darr, color='magenta', point_size=5)

        pl.add_mesh(cropped_mesh, color="#fafae1", opacity=0.5)
        pl.add_points(anchor_neighbors_pts, color='k', point_size=5, opacity=0.2)
        pl.add_arrows(anchor_neighbors_pts, anchor_neighbors_normals, color="navy", opacity=0.7, mag=0.5)


        # 2. 変位後の形状 (赤点) と 変位ベクトル (橙矢印)
        pl.add_points(Psin_vox_3darr_trimmed, color='red', point_size=5)
        pl.add_arrows(P_vox_3darr_trimmed, dPsin_vox_3darr_trimmed, color='orange', mag=1, label='Transverse displacement (buckling component)')

        # 主成分λ1の方向ベクトルも表示
        pl.add_arrows(P_vox_3darr_trimmed, g1_3d, mag=1.0, color='y', label='g1_hat_3d direction')
        # 壁のnormal vecsも表示
        pl.add_arrows(P_vox_3darr_trimmed, normals_coherence, mag=1.0, color='cyan', label='wall normal direction')

        # Anchorに使った点を表示
        pl.add_points(anchor_pts, color='green', point_size=10, label=f'anchor')
        if np.any(anchor_has_constraint):
            pl.add_points(anchor_pts[anchor_has_constraint], color='red', point_size=20, label=f'constraint')
        if np.any(~anchor_has_constraint):
            pl.add_points(anchor_pts[~anchor_has_constraint], color='blue', point_size=20, label=f'not constraint')

        pl.add_legend() 
        pl.add_title(f"Surface Constraint Visualization, segment ID: {segment_id}, segment index: {cell_index}", font_size=18)
        pl.camera_position = 'xy'
        pl.camera.azimuth = 180
        pl.camera.zoom(1.2)
        pl.add_key_event('space', lambda: tool.close_window(pl))
        pl.render()
        if hasattr(pl, 'render_window') and pl.render_window is not None:
            pl.render_window.SetPosition(plotter_position[0], plotter_position[1])
        
        # HTMLファイルとして出力
        html_output_path = os.path.join(output_dir_per_case, f"{directory_id}_buckling_surface_constraint_visualization_{segment_id}_{cell_index}.html")
        try:
            pl.export_html(html_output_path)
        except Exception as e:
            print(f"Warning: HTMLファイルの保存に失敗しました: {str(e)}")
        
        try:
            # pl.show()
            pl.close()
        except (AttributeError, RuntimeError):
            pass


        # data_vessel_segmentdataにsurface_constraintの情報を追加
        # 追加: 3D Constraint Metrics (脳表拘束の証明用)
        # 主要指標（判定はCSV保存後に柔軟に行う）
        df_vessel_segmentdata.loc[0, "R_out_surface"]   = surface_constraint_result["R_out"]
        df_vessel_segmentdata.loc[0, "R_dir_surface"]   = surface_constraint_result["R_dir"]
        # 幾何学的カバレッジ（信頼性）
        df_vessel_segmentdata.loc[0, "cortex_coverage"] = surface_constraint_result["cortex_coverage"]
        df_vessel_segmentdata.loc[0, "constraint_coverage"] = surface_constraint_result["constraint_coverage"]
        df_vessel_segmentdata.loc[0, "energy_coverage"] = surface_constraint_result["energy_coverage"]
        # エネルギー量
        df_vessel_segmentdata.loc[0, "total_energy"]    = surface_constraint_result["total_energy"]
        df_vessel_segmentdata.loc[0, "energy_constrained"] = surface_constraint_result["energy_constrained"]
        df_vessel_segmentdata.loc[0, "perp_energy"]     = surface_constraint_result["perp_energy"]
        # 点数（割合だけだと危ないので）
        df_vessel_segmentdata.loc[0, "n_points_total"] = surface_constraint_result["n_points_total"] # 全ての点数
        df_vessel_segmentdata.loc[0, "n_eval"]         = surface_constraint_result["n_eval"] # 評価できた点の数（脳表法線が定義できた点）

        # 再現性のための解析パラメータ（重要）
        df_vessel_segmentdata.loc[0, "coh_thr"]                = surface_constraint_result["coh_thr"]
        df_vessel_segmentdata.loc[0, "constraint_coverage_thresh"] = surface_constraint_result["constraint_coverage_thresh"]
        df_vessel_segmentdata.loc[0, "energy_coverage_thresh"] = surface_constraint_result["energy_coverage_thresh"]
        df_vessel_segmentdata.loc[0, "R_out_thresh"]           = surface_constraint_result["R_out_thresh"]
        df_vessel_segmentdata.loc[0, "stride_mm"]              = surface_constraint_result["stride"] * ds_vox * voxel_size
        df_vessel_segmentdata.loc[0, "min_neighbors"]          = surface_constraint_result["min_neighbors"]
        df_vessel_segmentdata.loc[0, "r_patch_mm"]             = surface_constraint_result["r_patch_vox"] * voxel_size
        df_vessel_segmentdata.loc[0, "gap_fill_size"]          = surface_constraint_result["gap_fill_size"]

        # DataFrameを返す（個別CSVファイルは保存しない）
        return df_vessel_segmentdata
        
    except Exception as e:
        # cell_indexが定義されていない場合（142行目以前でエラーが発生した場合）の処理
        cell_index_str = f"{cell_index}" if 'cell_index' in locals() else "N/A"
        print(f"Error processing segment_id {segment_id}, segment_index: {cell_index_str}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None  # 失敗

# すべてのsegment_idに対して処理を実行
all_dataframes = []
successful_segments = []
failed_segments = []

for segment_id in all_segment_ids:
    result = process_segment(segment_id)
    if result is not None:
        if isinstance(result, pd.DataFrame):
            all_dataframes.append(result)
            successful_segments.append(segment_id)
        else:
            failed_segments.append(segment_id)
    else:
        failed_segments.append(segment_id)

print("\n" + "="*80)
print("処理完了")
print(f"成功: {len(successful_segments)} segments")
print(f"失敗: {len(failed_segments)} segments")
if failed_segments:
    print(f"失敗したsegment_id: {failed_segments}")
print("="*80)

# 統合CSVファイルを保存
if len(all_dataframes) > 0:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_csv_path = os.path.join(output_dir_combined, f"{directory_id}_vessel_segmentdata.csv")
    combined_df.to_csv(combined_csv_path, index=None)
    print(f"\nCSVファイルを保存した: {combined_csv_path}")
    print(f"総レコード数: {len(combined_df)}")
else:
    print("\n警告: 保存するデータがない")