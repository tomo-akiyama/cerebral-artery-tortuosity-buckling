# Figure: Schematic dispersion relationの作成

"""
### 数理モデル
$ \mu \frac{\delta h}{\delta t}= -P \frac{\delta^2 h}{\delta x^2} - EI \frac{\delta^4 h}{\delta x^4} -k_s h -\alpha h^3$  
左辺は減衰項（粘性項）、右辺は順に圧縮荷重項、曲げ弾性項、線形復元力項、非線形復元力項   

#### 線形安定性解析  
摂動(変化)が微小という仮定において、高次の項を無視（$h^3\approx0$）すると  微分方程式は線形化して解析可能となる   
このとき固有値$\lambda(k)$は    
$\lambda(k) = \frac{1}{\mu} (P k^2 - EI k^4 - k_s)$
"""


def calculate_P(t, alpha):
    return alpha * t

def calculate_eigenvalue(t, k, alpha, beta, gamma):
    P = calculate_P(t, alpha)
    eigenvalue = P * k**2 - beta * k**4 - gamma
    return eigenvalue


import plot_config  # 共通のプロット設定をインポート
from plot_config import text_fontsize, text_fontsize_large
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import matplotlib.cm as cm
import os

# パラメータの設定
alpha = 10
beta = 1.0
gamma = 1410

num = 5

# 波数kの範囲
k_values = np.linspace(0, 10, 100)
# 時間tのリスト
t_values = np.linspace(0, 10, 5)

fig, ax = plt.subplots(figsize=(7, 3), dpi=600, constrained_layout=True)

for i, t in enumerate(t_values):
    eigenvalues = calculate_eigenvalue(t, k_values, alpha, beta, gamma)
    ax.plot(k_values, eigenvalues, color=cm.Blues((i+1)/num))

    # 極値を検出
    peaks, _ = find_peaks(eigenvalues)
    peak_k_values = k_values[peaks]
    peak_eigenvalues = eigenvalues[peaks]
    ax.plot(peak_k_values, peak_eigenvalues, 'x', markersize=6, color='k')

    if i == 0:
        ax.text(k_values[-1] + 0.2, eigenvalues[-1], r'$P=P_{\mathrm{0}}$', 
                ha='left', va='center', fontsize=plt.rcParams["axes.labelsize"])
    if i == 3:
        ax.text(k_values[-1] + 0.2, eigenvalues[-1], r'$P=P_{\mathrm{cr}}$', 
                ha='left', va='center', fontsize=plt.rcParams["axes.labelsize"])

# 軸を原点を通るように設定
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


# 最後のtの値で計算したeigenvaluesのmaxの上にテキストを追加
last_eigenvalues = calculate_eigenvalue(t_values[-1], peak_k_values, alpha, beta, gamma)
ax.text(peak_k_values[-1] + 0.2, np.max(last_eigenvalues) + 600, r'$\lambda_{\mathrm{max}}$', 
        ha='center', va='center', fontsize=plot_config.text_fontsize_large)

# k=0の時の固有値の位置にテキストを追加
# 最初のtの値で計算したeigenvalues[0]を使用
first_eigenvalues = calculate_eigenvalue(t_values[0], k_values, alpha, beta, gamma)
ax.text(k_values[0] - 0.1, first_eigenvalues[0], r'$-k_s/\mu$', 
        ha='right', va='center', fontsize=plot_config.text_fontsize_large)

# 目盛りを非表示
ax.set_xticks([])
ax.set_yticks([])

# x, y軸の範囲を取得
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# y軸の上端に$\lambda$を追加
ax.text(-0.1, ylim[1]+400, r'$\lambda$', 
        ha='right', va='center', fontsize=plt.rcParams["axes.labelsize"])
# x軸の右端の$k$を追加
ax.text(xlim[1]+0.1, 400, r'$k$', 
        ha='left', va='center', fontsize=plt.rcParams["axes.labelsize"])

# Text挿入
ax.text(peak_k_values[-1] + 0.5, np.max(last_eigenvalues)-7000, r'Increasing axial load $P$', 
        ha='right', va='center', fontsize=plt.rcParams["axes.labelsize"])

# 保存
output_path = "figures/schematic_dispersion_relation.pdf"   
fig.savefig(output_path, bbox_inches='tight')
print(f"Schematic dispersion relationの保存完了: {output_path}")
plt.show()
