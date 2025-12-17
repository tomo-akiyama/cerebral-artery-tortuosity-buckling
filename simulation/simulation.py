# Simulation of cerebral artery buckling
# Semi-implicit Euler scheme

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu
from figures.plot_config import (
    alpha, ms, blue_color, lightblue_color, orange_color, 
    lightorange_color, gray_color, colors, label_fontsize
)

def run_semi_implicit_simulation(
    Pi_P=1.0,
    Pi_ks=0.1,
    L0=100.0,
    N=1000,
    t_max=1000.0,
    dt=0.01,
    noise_std=0.01,
    seed=1,
    save_every=1000,
):
    """
    半陰Euler法で
        h_t = - Pi_P h_xx - h_xxxx - Pi_ks h - h^3
    を周期境界条件のもとで数値的に解く。

    Parameters
    ----------
    Pi_P : float 無次元張力パラメータ Pi_P.
    Pi_ks : float 無次元線形拘束パラメータ Pi_ks.
    L0 : float 初期長さ（空間ドメイン長）.
    N : int 空間グリッド数.
    t_max : float 総シミュレーション時間.
    dt : float 時間刻み.
    noise_std : float 初期条件ノイズの標準偏差.
    seed : int 乱数シード.
    save_every : int 何ステップごとに結果を保存するか（可視化・解析用）.

    Returns
    -------
    x : (N,) array 空間座標.
    t_saved : (n_save,) array 保存した時刻.
    h_saved : (n_save, N) array 各保存時刻における h(x,t).
    """
    rng = np.random.default_rng(seed)

    # --- 空間離散化 ---
    dx = L0 / N
    x = np.linspace(0.0, L0 - dx, N)  # 0..L0-dx, 周期境界を想定

    # 周期境界付き 2階差分 D2
    # 対角（-2）, 上下対角（1）, さらに周期を表す角成分を追加
    main_diag = -2.0 * np.ones(N)
    off_diag = 1.0 * np.ones(N - 1)
    D2 = diags(
        diagonals=[off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="lil",
    )

    # 周期境界: j=0 の左隣は j=N-1, j=N-1 の右隣は j=0
    D2[0, -1] = 1.0
    D2[-1, 0] = 1.0
    D2 = (1.0 / dx**2) * D2.tocsr()

    # 4階微分オペレータ D4 = D2 @ D2
    D4 = D2 @ D2

    # 単位行列（疎行列）
    I = eye(N, format="csr")

    # --- 線形作用素 L と A 行列の構築 ---
    # L h = - Pi_P D2 h - D4 h - Pi_ks h
    L = -Pi_P * D2 - D4 - Pi_ks * I

    # 半陰Euler: (I - dt L) h^{n+1} = h^{n} - dt (h^{n})^3
    A = (I - dt * L).tocsr()

    # A をLU分解しておけば毎ステップ速く解ける
    # 注意: 周期境界条件により、Aは常に可逆である必要がある
    A_lu = splu(A)

    # --- 初期条件 h(x,0) = 0 + epsilon ---
    h = rng.normal(loc=0.0, scale=noise_std, size=N)

    # --- 時間ループの準備 ---
    n_steps = int(np.round(t_max / dt))

    # 保存用
    t_saved = []
    h_saved = []

    def save_state(step, h_vec):
        t_now = step * dt
        t_saved.append(t_now)
        h_saved.append(h_vec.copy())

    # 初期状態を保存
    save_state(0, h)

    # --- 時間発展 ---
    for n in range(1, n_steps + 1):
        # 右辺: b = h^n - dt * (h^n)^3
        # 非線形項 h^3 は陽的に扱う（半陰Euler法）
        b = h - dt * (h**3)

        # A h^{n+1} = b を解く（LU分解済み）
        # 線形項は陰的に扱われる（安定性向上）
        h = A_lu.solve(b)

        # 必要なステップごとに保存
        if n % save_every == 0:
            save_state(n, h)

    return x, np.array(t_saved), np.array(h_saved)

if __name__ == "__main__":
# パラメータ設定
    Pi_P = 1      # 適当な値
    from parameters import L0, N, dt, t_max, Pi_ks, noise_std, seed, save_interval_t

    print(f"パラメータ: Pi_P = {Pi_P}, Pi_ks = {Pi_ks}, L0 = {L0}, N = {N}, t_max = {t_max}, dt = {dt}")
    print(f"k_max = (Pi_P / 2)**(1/2) = {(Pi_P / 2)**(1/2)}")

print("シミュレーション開始...")
x, t_saved, h_saved = run_semi_implicit_simulation(
    Pi_P=Pi_P,
    Pi_ks=Pi_ks,
    L0=L0,
    N=N,
    t_max=t_max,
    dt=dt,
        noise_std=noise_std,
        seed=seed,
        save_every=int(save_interval_t / dt),  # tの保存間隔
)

print(f"シミュレーション完了。保存された時刻数: {len(t_saved)}")

# 最大変位 max_x |h(x,t)| を計算
max_displacements = [np.max(np.abs(h)) for h in h_saved]

    # 初期状態を真っ直ぐに見せるために、初期状態を0にする
    h_saved_vis = h_saved.copy()
    h_saved_vis[0] = np.zeros_like(h_saved_vis[0])
    max_displacements_vis = max_displacements.copy()
    max_displacements_vis[0] = 0.0

# 2つのプロットを作成
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(7, 3),
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.05},
        constrained_layout=True
    ) # "wspace": 余白を調整

# 左側: 空間パターンの時間発展
    # 成長を見る代表時刻と、飽和後の代表時刻だけ選ぶ
    times_to_plot = [0, 50, 100, 150, 200]  # 200 でほぼ飽和とわかっている

    indices_to_plot = []
    for target_t in times_to_plot:
        closest_idx = np.argmin(np.abs(t_saved - target_t))
        if closest_idx not in indices_to_plot:
            indices_to_plot.append(closest_idx)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "time_blue", colors, N=len(indices_to_plot)
    )

    for j, i in enumerate(indices_to_plot):
        t = t_saved[i]
        h = h_saved_vis[i] # 初期状態を真っ直ぐに見せるために、初期状態を0にしたもの

        # この時刻に対応する色をカラーマップから取得
        col = cmap(j)

        # 飽和後（t≈200）は少し太くして「最終形」とわかるように
        lw = 1.2 if t < 200 else 1.6

        ax1.plot(x, h, color=col, alpha=0.9, lw=lw)

    ax1.set_xlabel(r"Spatial coordinate $x$")
    ax1.set_ylabel(r"Displacement $h(x, t)$")
ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-max(max_displacements_vis) * 1.1,
                  max(max_displacements_vis) * 1.1)
    ax1.set_xlim(0, L0)

# 右側: 最大変位の時間発展（成長→飽和）
    ax2.plot(t_saved, max_displacements_vis, 'o-', markersize=ms, color=blue_color)
    ax2.set_xlabel(r"Time $t$")
    ax2.set_ylabel(r"Maximum displacement $\max_x |h(x,t)|$")
    # ax2.set_title("Maximum displacement vs time (growth → saturation)")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(left=0)

# 画像ファイルとしても保存
    output_path = "simulation/buckling_pattern_evolution.pdf"
    plt.savefig(output_path, bbox_inches='tight')
print(f"プロットの保存完了: {output_path}")
plt.show()

    plt.show()

    # ===== 新しい図: 中央30ユニットだけを表示 =====
    print("\n中央30ユニットを抽出して図示...")

    # 中央30ユニットの範囲を定義
    center_x = L0 / 2
    display_width = 30
    x_min_display = center_x - display_width / 2
    x_max_display = center_x + display_width / 2

    # 表示範囲内のインデックスを取得
    mask_display = (x >= x_min_display) & (x <= x_max_display)
    x_display = x[mask_display]

    # ローカル座標(0,0)を中央30ユニットの中心に設定
    local_x = x_display - x_min_display

    # 新しい図を作成
    fig_extended, (ax1_ext, ax2_ext) = plt.subplots(
        1, 2,
        figsize=(7, 3),
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.05},
        constrained_layout=True
    )

    # 左側: 中央30ユニットだけを表示
    times_to_plot_ext = [0, 50, 100, 150, 200]

    indices_to_plot_ext = []
    for target_t in times_to_plot_ext:
        closest_idx = np.argmin(np.abs(t_saved - target_t))
        if closest_idx not in indices_to_plot_ext:
            indices_to_plot_ext.append(closest_idx)

    for j, i in enumerate(indices_to_plot_ext):
        t = t_saved[i]
        h = h_saved_vis[i]
        
        # 表示範囲内のデータだけを抽出
        h_display = h[mask_display]
        
        col = cmap(j)
        lw = 1.2 if t < 200 else 1.6
        
        ax1_ext.plot(local_x, h_display, color=col, alpha=0.9, lw=lw)


    ax1_ext.set_xlabel(r"Spatial coordinate $x$")
    ax1_ext.set_ylabel(r"Displacement $h(x, t)$")
    ax1_ext.grid(True, alpha=0.3)
    ax1_ext.set_ylim(-max(max_displacements_vis) * 1.1,
                      max(max_displacements_vis) * 1.1)
    ax1_ext.set_xlim(0, display_width*1.16)             # 0〜30 に固定した表示範囲に合わせる

    # 右側: 最大変位の時間発展（既存と同じ）
    ax2_ext.plot(t_saved, max_displacements_vis, 'o-', markersize=ms, color=blue_color)
    ax2_ext.set_xlabel(r"Time $t$")
    ax2_ext.set_ylabel(r"Maximum displacement $\max_x |h(x,t)|$")
    ax2_ext.grid(True, alpha=0.3)
    ax2_ext.set_xlim(left=0)

    # 保存
    output_path_extended = "simulation/buckling_pattern_evolution_central_30_units.pdf"
    fig_extended.savefig(output_path_extended, bbox_inches='tight')
    print(f"中央30ユニット表示のプロットの保存完了: {output_path_extended}")
    plt.show()

    # ===== FFT解析: 飽和状態のパワースペクトラム =====
    print("\nFFT解析を実行中...")
    h_steady = h_saved[-1]  # 飽和状態

    # FFTを実行
    fft_h = np.fft.fft(h_steady)
    n = len(h_steady)
    dx = x[1] - x[0]  # 空間刻み

    # 周波数（波数）を計算
    # 空間周波数: k = 2π/λ (波長λに対応)
    freqs = np.fft.fftfreq(n, dx)
    # 波数 k = 2π * 周波数
    k = 2 * np.pi * freqs

    # パワースペクトラムを計算
    power_spectrum = np.abs(fft_h)**2

    # 正の周波数（波数）のみを抽出（対称性のため）
    positive_idx = k > 0
    k_positive = k[positive_idx]
    power_positive = power_spectrum[positive_idx]

    # 特徴周波数（最大パワーを持つ波数）を抽出
    if len(power_positive) > 0:
        max_power_idx = np.argmax(power_positive)
        dominant_k = k_positive[max_power_idx]
        dominant_wavelength = 2 * np.pi / dominant_k if dominant_k > 0 else L0
        
        print(f"特徴波数: k = {dominant_k:.4f}")
        print(f"特徴波長: λ = {dominant_wavelength:.4f}")
        print(f"最大パワー: {power_positive[max_power_idx]:.2e}")
    else:
        print("特徴周波数の抽出なし")

    # パワースペクトラムをプロット
    fig_fft, ax_fft = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)

    # 波数に対してプロット
    ax_fft.plot(k_positive[:30], power_positive[:30], 'o-', markersize=ms, color=blue_color, linewidth=1.2)
    ax_fft.set_xlabel(r'Wavenumber $k$')
    ax_fft.set_ylabel(r'Power spectrum $|\hat{h}(k)|^2$')
    ax_fft.set_title('Power spectrum of steady-state buckling pattern')
    ax_fft.grid(True, alpha=0.3)
    ax_fft.set_xlim(left=0)

    # 特徴周波数を強調表示
    if len(power_positive) > 0:
        ax_fft.axvline(x=dominant_k, color=orange_color, linestyle='--', 
                       linewidth=1.5, alpha=0.7, label=f'Dominant wavenumber$k$ = {dominant_k:.3f}')
        ax_fft.legend(loc='best', fontsize=8)

    # 保存
    output_path_fft = "simulation/power_spectrum_of_steady_state_buckling_pattern.pdf"
    fig_fft.savefig(output_path_fft, bbox_inches='tight')
    print(f"パワースペクトラムの保存完了: {output_path_fft}")
    plt.show()

# アニメーションの作成
print("アニメーションを作成中...")
    fig_anim, (ax_anim1, ax_anim2) = plt.subplots(1, 2, figsize=(7.0, 4.5), constrained_layout=True)

# y軸の範囲を事前に設定（安定した表示のため）
    y_min = np.min(h_saved_vis) * 1.1
    y_max = np.max(h_saved_vis) * 1.1
ax_anim1.set_xlim(0, L0)
ax_anim1.set_ylim(y_min, y_max)
    ax_anim1.set_xlabel(r"Spatial coordinate $x$")
    ax_anim1.set_ylabel(r"Displacement $h(x, t)$")
ax_anim1.set_title("Buckling pattern evolution")
ax_anim1.grid(True, alpha=0.3)

# 最大変位のプロット用
ax_anim2.set_xlim(0, t_max)
    ax_anim2.set_ylim(0, max(max_displacements_vis) * 1.1)
    ax_anim2.set_xlabel(r"Time $t$")
    ax_anim2.set_ylabel(r"Maximum displacement $\max_x |h(x,t)|$")
ax_anim2.set_title("Maximum displacement vs time")
ax_anim2.grid(True, alpha=0.3)

# 初期化
    line1, = ax_anim1.plot([], [], color=blue_color, label=r'$h(x,t)$')
time_text = ax_anim1.text(0.02, 0.95, '', transform=ax_anim1.transAxes, 
                          fontsize=label_fontsize, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
line2, = ax_anim2.plot([], [], 'o-', markersize=ms, color=blue_color)
max_disp_line, = ax_anim2.plot([], [], 'o', markersize=ms, color=orange_color)

def animate(frame):
        # アニメーションの各フレームを更新
    # 左側: 現在のパターン
        line1.set_data(x, h_saved_vis[frame])
    time_text.set_text(f't = {t_saved[frame]:.2f}')
    
    # 右側: 最大変位の時系列（現在時刻まで）
    t_current = t_saved[frame]
        max_disp_current = max_displacements_vis[frame]
    
    # 時系列プロット（現在時刻まで）
    t_plot = t_saved[:frame+1]
        max_disp_plot = max_displacements_vis[:frame+1]
    line2.set_data(t_plot, max_disp_plot)
    
    # 現在の点を強調表示
    max_disp_line.set_data([t_current], [max_disp_current])
    
    return line1, time_text, line2, max_disp_line

# アニメーション作成
anim = FuncAnimation(fig_anim, animate, frames=len(t_saved), 
                    interval=200, blit=True, repeat=True)

# アニメーションをGIFとして保存
    anim_path = "simulation/fixed_Pi_P_animation.gif"
print(f"アニメーションを保存中: {anim_path}")
anim.save(anim_path, writer=PillowWriter(fps=5), dpi=600)
print(f"アニメーションの保存完了: {anim_path}")

# アニメーションを表示
plt.show()