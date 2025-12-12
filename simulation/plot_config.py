# Plot configuration for simulation scripts
# 共通のプロット設定をまとめたファイル

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import colormaps

# Seaborn推奨のテーマ設定
sns.set_theme(
    style="whitegrid",
    context="paper",
    font="Arial",
    rc={
        "axes.edgecolor": "black",           # 枠線を黒に
        "axes.linewidth": 0.6,               # 枠線の太さ
        "xtick.direction": "out",            # 突起を外向き
        "ytick.direction": "out",
        "xtick.major.size": 4,               # 突起の長さ
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,            # 突起の太さ
        "ytick.major.width": 0.8
    }
)

# Tickの表示設定
plt.rcParams.update({
    "xtick.bottom": True,
    "xtick.top": False, # 上の目盛りは非表示
    "ytick.left": True,
    "ytick.right": False, # 右の目盛りは非表示
})

# さらに細かく調整
plt.rcParams.update({
    "lines.linewidth": 1.2, # 折れ線などの太さ
    "axes.titlesize": 10, # タイトル（ax.set_title()）の文字サイズ（基本）
    "axes.labelsize": 10, # x軸・y軸ラベルのサイズ（基本）
    "xtick.labelsize": 8, # x軸目盛（tickの数字）のサイズ（小さめ）
    "ytick.labelsize": 8, # y軸目盛（tickの数字）のサイズ（小さめ）
    "legend.fontsize": 8, # 凡例（legend）の文字サイズ（小さめ）
    "mathtext.fontset": "cm",  # 数学記号をComputer Modernフォントに（TeX風）
    "mathtext.default": "it",  # デフォルトでイタリック体
})

# 共通パラメータ
alpha = 0.8
ms = 3

# color rules
tab20 = colormaps["tab20"]  # 新しい推奨スタイル
blue_color = tab20(0)  # 青（視認性が高い）
lightblue_color = tab20(1)
orange_color = tab20(2)   # オレンジ（赤寄りで区別しやすい）
lightorange_color = tab20(3)

gray_color = '#4D4D4D'  # グレー（視認性が高い）

# グラデーション用のカラーマップを作成
colors = [
    mcolors.to_rgba("#cfe8ff"),  # 明るい水色（初期）
    mcolors.to_rgba("#08306b")   # 濃い紺（後期）
]

# subplotごとのラベル "a", "b", "c" を左上に追加（bold & fontsize=12）
label_fontsize = plt.rcParams["axes.titlesize"]  # タイトルと同じサイズに合わせる

