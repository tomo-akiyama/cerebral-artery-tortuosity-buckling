#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: 3 vessel curvature splitなし.ipynb

# Cell 2
# Select the case for preprocessing

case = "IXI538-HH-2411"

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
intermediate_data3_path = os.path.join(case_path, 'Intermediate_data3'); tool.create_dir(intermediate_data3_path)

intermediate_data1_3ds_path = os.path.join(case_path, 'Intermediate_data1 3d slicer'); tool.create_dir(intermediate_data1_3ds_path)


mca_intermediate_data1_3ds_path = os.path.join(case_path, 'MCA Intermediate_data1 3d slicer'); tool.create_dir(mca_intermediate_data1_3ds_path)
mca_intermediate_data3_3ds_path = os.path.join(case_path, 'MCA Intermediate_data3 3d slicer'); tool.create_dir(mca_intermediate_data3_3ds_path)
mca_intermediate_data4_3ds_path = os.path.join(case_path, 'MCA Intermediate_data4 3d slicer'); tool.create_dir(mca_intermediate_data4_3ds_path)


# 

# Cell 5
# 血管とcenterlineの読み込み
vessel_mesh_path = os.path.join(intermediate_data1_path, "1_2_mca_mc_taubin_thick_butterfly16.ply")
vessel_mesh = mesh_generation.load_ply_to_polydata(vessel_mesh_path)

centerlines_path = os.path.join(mca_intermediate_data3_3ds_path, "1 vmtk_centerlines.vtp")
centerlines = mesh_generation.load_vtp(centerlines_path)


# vmtk_CL.check_steplength(centerlines)

# Cell 6
pl = pv.Plotter()
pl.camera.zoom(1)

# Cell 8
s_factor = 1.2
num_iter = 100
s_length = 0.2

# Smoothing centerlines
smoothed_centerlines = vmtk_CL.vmtk_centerline_smoothing(centerlines, num_iteration=num_iter, smooth_factor=s_factor)

# Save smoothed centerlines
splitted_centerlines_path = os.path.join(mca_intermediate_data4_3ds_path, "2 smoothed_centerlines.vtp")
mesh_generation.save_vtp(smoothed_centerlines, splitted_centerlines_path)

# Cell 9
pl = pv.Plotter()
pl.camera.zoom(1)

# Cell 10
# Resampling centerlines
resampled_smoothed_centerlines = vmtk_CL.vmtk_centerline_resampling(smoothed_centerlines, step_length=s_length)

# Save centerlines
resampled_smoothed_centerlines_path = os.path.join(mca_intermediate_data4_3ds_path, "3 resampled_smoothed_centerlines.vtp")
mesh_generation.save_vtp(resampled_smoothed_centerlines, resampled_smoothed_centerlines_path)

# Cell 11
# vmtk_CL.check_steplength(resampled_smoothed_centerlines)

# Cell 12
# Post processing
post_processing_centerlines = vmtk_CL.vmtk_centerline_attributes(resampled_smoothed_centerlines)
post_processing_centerlines = vmtk_CL.vmtk_compute_geometry(resampled_smoothed_centerlines)

# Save centerlines
post_processing_centerlines_path = os.path.join(mca_intermediate_data4_3ds_path, "4 post_processing_centerlines.vtp")
mesh_generation.save_vtp(post_processing_centerlines, post_processing_centerlines_path)

# Cell 13
tool.notify(case)

# Cell 14
"""

# Cell 15
removal = 10

# Cell 16
pl = pv.Plotter(shape=(1,2))
pl.subplot(0,0)
for i in range(len(pointids)):
    if len(points[pointids[i]][removal:-removal])==0:
        continue
    else:
                    scalars=kappa[pointids[i]][removal:-removal], 
                    opacity=1, line_width=5, cmap="jet")
                      render_points_as_spheres=True, 
                      point_size=4,
                      opacity=0.5)
pl.camera.zoom(1)
pl.subplot(0,1)
for i in range(len(pointids)):
                scalars=pointdata["Curvature"][pointids[i]], 
                opacity=1, line_width=5, cmap="jet")
                      render_points_as_spheres=True,
                      point_size=5,
                      opacity=0.5)

pl.camera.zoom(1)

# Cell 17
# Remove ends
removed_pointids = tool.remove_end_points(pointids, removal)

# Cell 18
# FFT
# ウィンドウごとのFFT
actucal_window_length = 100 # (mm)  # 固定のウィンドウ長 
step = s_length * 0.5 # 0.5mmはspacing, サンプリング周期 ※サンプリング周波数は1/step = 10 (Hz, 1/mm)
window_length = int (actucal_window_length / step) # 固定のウィンドウのサンプル数

fs = 1 / step  # サンプリング周波数　→ サンプリング周期dt = 1/fs = 20mm

# オーバーラップして分割
overlap_rate = 0.5  # 50%のオーバーラップ
overlap = int(overlap_rate * window_length)  # 50%のオーバーラップ


# Zero padding
padded_length = window_length

if padded_length == window_length:
else:

# Cell 19
# FFTで検出可能な最小周波数
min_freq = 1 / actucal_window_length

# Cell 20
# DCオフセットを取り除く関数
def remove_dc_offset(curvature):
    mean_value = np.mean(curvature)
    curvature_without_dc = curvature - mean_value
    return curvature_without_dc, mean_value

# Cell 21
# 全体のFFT結果
fig1, axs1 = plt.subplots(1, 2, figsize=(20, 5), sharex=True)
total_Amp1 = np.zeros(padded_length) # 加算平均用の配列を用意
segment_count1 = 0  # セグメント数のカウンタ

# グループごとのFFT結果
# 加算平均FFT: diameter 0.5-1.5, 1.5-2.5, 2.5+
total_Amp2_1 = np.zeros(padded_length) # 0.5-1.5の加算平均用の配列を用意
segment_count2_1= 0  # 0.5-1.5のセグメント数のカウンタ
total_Amp2_2 = np.zeros(padded_length) # 1.5-2.5の加算平均用の配列を用意
segment_count2_2= 0  # 1.5-2.5のセグメント数のカウンタ
total_Amp2_3 = np.zeros(padded_length) # 2.5+の加算平均用の配列を用意
segment_count2_3= 0  # 2.5+のセグメント数のカウンタ
# 各FFT plot: diameter 0.5-1.5, 1.5-2.5, 2.5+
fig2, axs2 = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)
fig3, axs3 = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)
fig4, axs4 = plt.subplots(4, 1, figsize=(10, 40), sharex=True, sharey=True)


for i, ids in enumerate(removed_pointids):
    new_pts = points[ids] # native space
    new_kappa = kappa[ids] # native space
    new_tau = tau[ids] # native space
    new_radius = radius[ids] # native space
    new_FrenetTangent = FrenetTangent[ids] # native space
    new_FrenetNormal = FrenetNormal[ids] # native space
    new_FrenetBinormal = FrenetBinormal[ids] # native space
    arc = len(ids)

    # 

    Kappa = new_kappa / 0.5 # original space
    diameter = new_radius * 0.5 * 2 # original space
    median_diameter = np.median(diameter)

    # 各セグメントごとのamp
    each_segment_amp = np.zeros(padded_length)
    each_segment_count = 0   

    # mean_values を格納するリストを準備
    mean_values_segment1 = []
    mean_values_segment2 = []
    total_DC1 = []
    total_DC2 = []
    total_num_of_points = []

    for start in range(0, len(Kappa), window_length - overlap):
        # window_length分のデータを取り出す(DCオフセットを取り除く前)
        segment = Kappa[start:start+window_length]
        # DCオフセットを取り除く
        segment_without_dc, mean_value1 = remove_dc_offset(segment)
        num_of_points = len(segment)

        # 各セグメントの長さに応じてハミング窓を計算
        window = np.hamming(len(segment))
        # ハミング窓を適用
        windowed_segment = segment_without_dc * window

        # ハミング窓によって生じたDC成分を再度取り除く
        windowed_segment_without_dc, mean_value2 = remove_dc_offset(windowed_segment)

        # FFT
        F = np.fft.fft(windowed_segment_without_dc, n=padded_length)
        freq = np.fft.fftfreq(padded_length, d=1/fs)

        Amp = np.abs(F)
        Amp = Amp / (padded_length / 2)
        # DC成分の修正(DC成分は2倍にしないので正規化で2倍になったものを元に戻す)
        Amp[0] = Amp[0] / 2
        # ナイキスト周波数の修正(サンプル数Nが偶数の場合のN/2周期の値は2倍にしないので正規化で2倍になったものを元に戻す)
        if padded_length % 2 == 0:
            Amp[padded_length//2] = Amp[padded_length//2] / 2  # ナイキスト周波数の補正


        # 窓関数のスケーリング補正
        Amplitude_Correction_Factor = np.sum(window) / len(segment)
        Power_Correction_Factor = np.sum(window * window) / len(segment)
        Power_Equivalent_Bandwidth_Factor = Power_Correction_Factor / (Amplitude_Correction_Factor ** 2)

        # ゼロパディングによるスケーリング補正
        zero_padding_factor = padded_length / len(segment)

        # スケーリング
        Amp_scaled = (Amp / Amplitude_Correction_Factor) * zero_padding_factor

        # 全体のAmp加算
        total_Amp1 += Amp_scaled
        segment_count1 += 1  # セグメントのカウントを増やす

        # 各セグメントのAmp加算
        each_segment_amp += Amp_scaled
        each_segment_count += 1

        # mean_value1 と mean_value2 をリストに保存
        mean_values_segment1.append(mean_value1)
        mean_values_segment2.append(mean_value2)

        # 振幅スペクトルのプロット
        axs1[0].plot(freq[:int(padded_length/2)], Amp_scaled[:int(padded_length/2)], alpha=0.5)
        axs1[0].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
        axs1[0].set_title('Windowed FFT Analysis of All Segments')
        axs1[0].set_xlabel('Spacial Freqency[1/mm]')
        axs1[0].set_ylabel("Amplitude (scaled)")
        axs1[0].grid(True, axis="both", which="major", linewidth=1)
        axs1[0].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
        axs1[0].minorticks_on()

        if median_diameter < 1.5:
            total_Amp2_1 += Amp_scaled
            segment_count2_1 += 1  # セグメントのカウントを増やす

            # 振幅スペクトルのプロット
            axs2[0].plot(freq[:int(padded_length/2)], Amp_scaled[:int(padded_length/2)], alpha=0.5)
            axs2[0].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
            axs2[0].set_title('Windowed FFT Analysis of Diameter 0.5-1.5[mm]') 
            axs2[0].set_xlabel('Spacial Freqency[1/mm]')
            axs2[0].set_ylabel("Amplitude (scaled)")
            axs2[0].grid(True, axis="both", which="major", linewidth=1)
            axs2[0].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
            axs2[0].minorticks_on()

        elif 1.5 <= median_diameter < 2.5:
            total_Amp2_2 += Amp_scaled
            segment_count2_2 += 1

            # 振幅スペクトルのプロット
            axs2[1].plot(freq[:int(padded_length/2)], Amp_scaled[:int(padded_length/2)], alpha=0.5)
            axs2[1].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
            axs2[1].set_title('Windowed FFT Analysis of Diameter 1.5-2.5[mm]')
            axs2[1].set_xlabel('Spacial Freqency[1/mm]')
            axs2[1].set_ylabel("Amplitude (scaled)")
            axs2[1].grid(True, axis="both", which="major", linewidth=1)
            axs2[1].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
            axs2[1].minorticks_on()

        else: # 2.5+
            total_Amp2_3 += Amp_scaled
            segment_count2_3 += 1

            # 振幅スペクトルのプロット
            axs2[2].plot(freq[:int(padded_length/2)], Amp_scaled[:int(padded_length/2)], alpha=0.5)
            axs2[2].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
            axs2[2].set_title('Windowed FFT Analysis of Diameter 2.5+[mm]')
            axs2[2].set_xlabel('Spacial Freqency[1/mm]')
            axs2[2].set_ylabel("Amplitude (scaled)")
            axs2[2].grid(True, axis="both", which="major", linewidth=1)
            axs2[2].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
            axs2[2].minorticks_on()
    
    # セグメントごとの平均を計算
    if each_segment_count > 0:
        each_segment_average_Amp = each_segment_amp / each_segment_count

    total_DC = np.sum(total_DC1) + np.sum(total_DC2)
    total_num_of_points = np.sum(total_num_of_points)



# 平均を計算
if segment_count1 > 0:
    average_Amp = total_Amp1 / segment_count1

    # 平均FFTをプロット
    axs1[1].plot(freq[:int(padded_length/2)], average_Amp[:int(padded_length/2)], marker='o', ms=3, alpha=0.5)
    axs1[1].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
    axs1[1].set_title('Average FFT Analysis of All Segments')
    axs1[1].set_xlabel('Spacial Freqency[1/mm]')
    axs1[1].set_ylabel("Amplitude (scaled)")
    axs1[1].grid(True, axis="both", which="major", linewidth=1)
    axs1[1].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
    axs1[1].minorticks_on()

    axs4[0].plot(freq[:int(padded_length/2)], average_Amp[:int(padded_length/2)], marker='o', ms=3, alpha=0.5)
    axs4[0].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
    axs4[0].set_title('Average FFT Analysis of All Segments')
    axs4[0].set_xlabel('Spacial Freqency[1/mm]')
    axs4[0].set_ylabel("Amplitude (scaled)")
    axs4[0].grid(True, axis="both", which="major", linewidth=1)
    axs4[0].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
    axs4[0].minorticks_on()

# グループごとの平均を計算
if segment_count2_1 > 0:
    average_Amp2_1 = total_Amp2_1 / segment_count2_1

    # 平均FFTをプロット
    axs3[0].plot(freq[:int(padded_length/2)], average_Amp2_1[:int(padded_length/2)], marker='o', ms=3, alpha=0.5)
    axs3[0].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
    axs3[0].set_title('Average FFT Analysis of Diameter 0.5-1.5[mm]')
    axs3[0].set_xlabel('Spacial Freqency[1/mm]')
    axs3[0].set_ylabel("Amplitude (scaled)")
    axs3[0].grid(True, axis="both", which="major", linewidth=1)
    axs3[0].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
    axs3[0].minorticks_on()

    axs4[1].plot(freq[:int(padded_length/2)], average_Amp2_1[:int(padded_length/2)], marker='o', ms=3, alpha=0.5)
    axs4[1].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
    axs4[1].set_title('Average FFT Analysis of Diameter 0.5-1.5[mm]')
    axs4[1].set_xlabel('Spacial Freqency[1/mm]')
    axs4[1].set_ylabel("Amplitude (scaled)")
    axs4[1].grid(True, axis="both", which="major", linewidth=1)
    axs4[1].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
    axs4[1].minorticks_on()

if segment_count2_2 > 0:
    average_Amp2_2 = total_Amp2_2 / segment_count2_2

    # 平均FFTをプロット
    axs3[1].plot(freq[:int(padded_length/2)], average_Amp2_2[:int(padded_length/2)], marker='o', ms=3, alpha=0.5)
    axs3[1].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
    axs3[1].set_title('Average FFT Analysis of Diameter 1.5-2.5[mm]')
    axs3[1].set_xlabel('Spacial Freqency[1/mm]')
    axs3[1].set_ylabel("Amplitude (scaled)")
    axs3[1].grid(True, axis="both", which="major", linewidth=1)
    axs3[1].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
    axs3[1].minorticks_on()

    axs4[2].plot(freq[:int(padded_length/2)], average_Amp2_2[:int(padded_length/2)], marker='o', ms=3, alpha=0.5)
    axs4[2].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
    axs4[2].set_title('Average FFT Analysis of Diameter 1.5-2.5[mm]')
    axs4[2].set_xlabel('Spacial Freqency[1/mm]')
    axs4[2].set_ylabel("Amplitude (scaled)")
    axs4[2].grid(True, axis="both", which="major", linewidth=1)
    axs4[2].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
    axs4[2].minorticks_on()

if segment_count2_3 > 0:
    average_Amp2_3 = total_Amp2_3 / segment_count2_3
    
    # 平均FFTをプロット
    axs3[2].plot(freq[:int(padded_length/2)], average_Amp2_3[:int(padded_length/2)], marker='o', ms=3, alpha=0.5)
    axs3[2].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
    axs3[2].set_title('Average FFT Analysis of Diameter 2.5+[mm]')
    axs3[2].set_xlabel('Spacial Freqency[1/mm]')
    axs3[2].set_ylabel("Amplitude (scaled)")
    axs3[2].grid(True, axis="both", which="major", linewidth=1)
    axs3[2].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
    axs3[2].minorticks_on()

    axs4[3].plot(freq[:int(padded_length/2)], average_Amp2_3[:int(padded_length/2)], marker='o', ms=3, alpha=0.5)
    axs4[3].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
    axs4[3].set_title('Average FFT Analysis of Diameter 2.5+[mm]')
    axs4[3].set_xlabel('Spacial Freqency[1/mm]')
    axs4[3].set_ylabel("Amplitude (scaled)")
    axs4[3].grid(True, axis="both", which="major", linewidth=1)
    axs4[3].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
    axs4[3].minorticks_on()

# グラフの表示
axs1[0].set_ylim(0, 0.5)
axs1[1].set_ylim(0, 0.1)
axs2[0].set_ylim(0, 0.5)
axs3[0].set_ylim(0, 0.1)
axs4[0].set_ylim(0, 0.1)

axs4[0].set_xlim(0, 1)

fig1.suptitle(f"{case}", fontsize=16)
fig2.suptitle(f"{case}", fontsize=16)
fig3.suptitle(f"{case}", fontsize=16)
fig4.suptitle(f"{case}", fontsize=16)

    

# Cell 22
1/freq[:int(padded_length/2)]

# Cell 23
fig4, axs4 = plt.subplots(4, 1, figsize=(10, 40), sharex=True, sharey=True)

# 平均を計算
if segment_count1 > 0:
    average_Amp = total_Amp1 / segment_count1

    # 平均FFTをプロット
    axs4[0].plot(1/freq[:int(padded_length/2)], average_Amp[:int(padded_length/2)], marker='o', ms=3, alpha=0.5)
    axs4[0].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
    axs4[0].set_title('Average FFT Analysis of All Segments')
    axs4[0].set_xlabel('Wavelength[mm]')
    axs4[0].set_ylabel("Amplitude (scaled)")
    axs4[0].grid(True, axis="both", which="major", linewidth=1)
    axs4[0].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
    axs4[0].minorticks_on()

# グループごとの平均を計算
if segment_count2_1 > 0:
    average_Amp2_1 = total_Amp2_1 / segment_count2_1

    # 平均FFTをプロット
    axs4[1].plot(freq[:int(padded_length/2)], average_Amp2_1[:int(padded_length/2)], marker='o', ms=3, alpha=0.5)
    axs4[1].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
    axs4[1].set_title('Average FFT Analysis of Diameter 0.5-1.5[mm]')
    axs4[1].set_xlabel('Spacial Freqency[1/mm]')
    axs4[1].set_ylabel("Amplitude (scaled)")
    axs4[1].grid(True, axis="both", which="major", linewidth=1)
    axs4[1].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
    axs4[1].minorticks_on()

if segment_count2_2 > 0:
    average_Amp2_2 = total_Amp2_2 / segment_count2_2

    # 平均FFTをプロット
    axs4[2].plot(freq[:int(padded_length/2)], average_Amp2_2[:int(padded_length/2)], marker='o', ms=3, alpha=0.5)
    axs4[2].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
    axs4[2].set_title('Average FFT Analysis of Diameter 1.5-2.5[mm]')
    axs4[2].set_xlabel('Spacial Freqency[1/mm]')
    axs4[2].set_ylabel("Amplitude (scaled)")
    axs4[2].grid(True, axis="both", which="major", linewidth=1)
    axs4[2].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
    axs4[2].minorticks_on()

if segment_count2_3 > 0:
    average_Amp2_3 = total_Amp2_3 / segment_count2_3
    
    # 平均FFTをプロット
    axs4[3].plot(freq[:int(padded_length/2)], average_Amp2_3[:int(padded_length/2)], marker='o', ms=3, alpha=0.5)
    axs4[3].axvline(x=min_freq, color='red', linestyle='--', alpha=0.5)
    axs4[3].set_title('Average FFT Analysis of Diameter 2.5+[mm]')
    axs4[3].set_xlabel('Spacial Freqency[1/mm]')
    axs4[3].set_ylabel("Amplitude (scaled)")
    axs4[3].grid(True, axis="both", which="major", linewidth=1)
    axs4[3].grid(True, axis="both", which="minor", linestyle="--", linewidth=0.5)
    axs4[3].minorticks_on()

# グラフの表示
# axs4[0].set_ylim(0, 0.1)

# axs4[0].set_xlim(0, 1)

fig4.suptitle(f"{case}", fontsize=16)

    
