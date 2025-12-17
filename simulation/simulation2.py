# Pitchfork bifurcation diagram
# Pi_ksを固定し、Pi_Pを変化させて分岐図を作成

import numpy as np
import matplotlib.pyplot as plt
from simulation import run_semi_implicit_simulation
from figures.plot_config import (
    alpha, ms, blue_color, lightblue_color, orange_color, 
    lightorange_color, gray_color
)

if __name__ == "__main__":
# パラメータ設定
    from parameters import L0, N, dt, t_max, Pi_ks, noise_std, seed, save_interval_t

# Pi_Pの範囲を設定（分岐点付近を重点的に）
# 理論的には、線形安定性解析から分岐点が決まる
Pi_P_min = 0.0
Pi_P_max = 2.0
    n_Pi_P = 80  # Pi_Pのサンプル数（計算時間を考慮して調整）

Pi_P_values = np.linspace(Pi_P_min, Pi_P_max, n_Pi_P)

    # 臨界荷重は Pi_P_cr = 2 * sqrt(Pi_ks)
    Pi_P_cr = 2 * (Pi_ks)**(1/2)

print(f"Pi_ks = {Pi_ks} を固定して、Pi_P を {Pi_P_min} から {Pi_P_max} まで変化させる...")
    print(f"臨界荷重: Pi_P_cr = {Pi_P_cr:.4f}")
print(f"シミュレーション回数: {n_Pi_P}")

    # 各Pi_Pでの定常状態の最大変位と波長を記録
max_displacements_steady = []
    dominant_wavelengths = []
Pi_P_results = []

# 定常状態を判定するための閾値（最後の20%の時間での変動が小さいことを確認）
steady_state_threshold = 0.01

    def calculate_dominant_wavelength(x, h):
        """支配的な波長を計算（FFT使用）"""
        # FFTで周波数解析
        fft_h = np.fft.fft(h)
        freqs = np.fft.fftfreq(len(h), x[1] - x[0])
        power = np.abs(fft_h)**2
        
        # ゼロ周波数と負の周波数を除外
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        # 最大パワーの周波数を見つける（ゼロ周波数以外）
        if len(positive_freqs) > 1:
            max_idx = np.argmax(positive_power[1:]) + 1
            dominant_freq = positive_freqs[max_idx]
            if dominant_freq > 0:
                wavelength = 1.0 / dominant_freq
            else:
                wavelength = L0  # デフォルト値
        else:
            wavelength = L0
        
        return wavelength

for i, Pi_P in enumerate(Pi_P_values):
    if (i + 1) % 5 == 0 or i == 0:
        print(f"Progress: {i+1}/{n_Pi_P}, Pi_P = {Pi_P:.3f}")
    
    # シミュレーション実行
    x, t_saved, h_saved = run_semi_implicit_simulation(
        Pi_P=Pi_P,
        Pi_ks=Pi_ks,
        L0=L0,
        N=N,
        t_max=t_max,
        dt=dt,
        noise_std=noise_std,
        seed=seed,
            save_every=int(save_interval_t / dt),  # 定常状態を確認するための保存間隔
    )
    
    # 定常状態の最大変位を計算（最後の20%の時間での平均）
    max_displacements = np.array([np.max(np.abs(h)) for h in h_saved])
    
    # 最後の20%の時間での平均と標準偏差
    n_last = int(len(max_displacements) * 0.2)
    if n_last > 0:
        steady_max_disp = np.mean(max_displacements[-n_last:])
        steady_std = np.std(max_displacements[-n_last:])
        
        # 定常状態かどうかを確認（変動が小さい）
        if steady_std < steady_state_threshold:
            max_displacements_steady.append(steady_max_disp)
            Pi_P_results.append(Pi_P)
                # 定常状態のパターンから波長を計算
                h_steady = np.mean(h_saved[-n_last:], axis=0)  # 最後のn_last個の平均
                wavelength = calculate_dominant_wavelength(x, h_steady)
                dominant_wavelengths.append(wavelength)
        else:
            # 変動が大きい場合は最後の値を採用
            max_displacements_steady.append(max_displacements[-1])
            Pi_P_results.append(Pi_P)
                h_steady = h_saved[-1]
                wavelength = calculate_dominant_wavelength(x, h_steady)
                dominant_wavelengths.append(wavelength)
    else:
        # データが少ない場合は最後の値を使用
        max_displacements_steady.append(max_displacements[-1])
        Pi_P_results.append(Pi_P)
            h_steady = h_saved[-1]
            wavelength = calculate_dominant_wavelength(x, h_steady)
            dominant_wavelengths.append(wavelength)

# 結果をnumpy配列に変換
Pi_P_results = np.array(Pi_P_results)
max_displacements_steady = np.array(max_displacements_steady)
    dominant_wavelengths = np.array(dominant_wavelengths)

    # 臨界波数と臨界波長を計算
    k_cr = (Pi_ks)**(1/4)
    lambda_cr = 2 * np.pi / k_cr if k_cr > 0 else 0
    print(f"臨界波数: k_cr = {k_cr:.4f}")
    print(f"臨界波長: lambda_cr = {lambda_cr:.4f}")

    # 分岐図を作成（2つのサブプロット）
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(7, 3),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05},
        constrained_layout=True
    )

    # 左: pitchfork図（振幅 vs Pi_P）
    ax1.plot(Pi_P_results, max_displacements_steady, 'o-', markersize=ms, color=blue_color, clip_on=False)
    ax1.set_xlabel(r'Axial load parameter $\Pi_P$')
    ax1.set_ylabel(r'Steady-state maximum displacement $\max_x |h(x)|$')
    ax1.yaxis.set_label_coords(-0.1,0.42) # y軸ラベルのy方向の位置を少し下げる
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(max_displacements_steady)*1.1)
    # 目盛りを軸の位置に表示
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.plot(Pi_P_cr, 0, 'o', markersize=ms, color=orange_color, clip_on=False)

    # 右: 支配的な波長 vs Pi_P
    # 臨界点以降のデータのみ抽出
    mask_post_critical = Pi_P_results >= Pi_P_cr
    Pi_P_post_critical = Pi_P_results[mask_post_critical]
    wavelengths_post_critical = dominant_wavelengths[mask_post_critical]

    # 臨界点以降のみプロット
    if len(Pi_P_post_critical) > 0:
        ax2.plot(Pi_P_post_critical, wavelengths_post_critical, 'o-', 
                    markersize=ms, color=blue_color)

    # 臨界点前の領域を灰色帯で塗る
    ax2.axvspan(Pi_P_min, Pi_P_cr, alpha=0.3, color='lightgray', zorder=0)

    # "Undefined"テキストを追加（x軸の上）
    y_lim_max = ax2.get_ylim()[1] if len(Pi_P_post_critical) > 0 else 1.0
    ax2.text(Pi_P_cr / 2, y_lim_max * 0.05,
                'Undefined', fontsize=10, ha='center', va='bottom', 
                style='italic', color=gray_color)

    # k_crに対応する波長の横線を追加
    ax2.axhline(y=lambda_cr, color=orange_color, linestyle='--', linewidth=1.2, alpha=0.7)
    
    # 横線の上にラベルを追加
    ax2.text(Pi_P_max * 0.95, lambda_cr * 1.025, 
             r'$\lambda_{\mathrm{cr}} = 2\pi/k_{\mathrm{cr}}$',
             fontsize=8, ha='right', va='bottom', 
             color=orange_color, style='italic')

    ax2.set_xlabel(r'Axial load parameter $\Pi_P$')
    ax2.set_ylabel(r'Dominant wavelength $\lambda^*$')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(Pi_P_min, Pi_P_max)
    if len(wavelengths_post_critical) > 0:
        ax2.set_ylim(0, max(wavelengths_post_critical) * 1.5)

# 保存
    output_path = "simulation/buckling_bifurcation_and_wavelength_selection.pdf"
    fig.savefig(output_path, bbox_inches='tight')
print(f"\nBifurcation diagram saved to: {output_path}")

plt.show()

# 分岐点の推定（最大変位が閾値を超える最初のPi_P）
    threshold = 0.1
above_threshold = max_displacements_steady > threshold
if np.any(above_threshold):
    bifurcation_idx = np.argmax(above_threshold)
    Pi_P_critical = Pi_P_results[bifurcation_idx]
    print(f"\nEstimated bifurcation point: Pi_P ≈ {Pi_P_critical:.4f}")
    print(f"  Maximum displacement at this point: {max_displacements_steady[bifurcation_idx]:.6f}")
else:
    print("\nNo bifurcation point found (adjust the threshold)")
