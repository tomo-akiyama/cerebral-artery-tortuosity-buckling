#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figure: Gif_dispersion_P_coupling."""

from __future__ import annotations

import os
import sys

import matplotlib

if not os.environ.get("MPLBACKEND"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from simulation.simulation_output import figure_path, get_figures_dir
from simulation.simulation_results import load_simulation_run
from simulation.Figure_dispersion_P_coupling import (
    DEFAULT_MODE_NS,
    collect_mode_points,
    compute_dispersion,
    dispersion_ylim_tight,
    get_dispersion_k_grid,
    mu_from_meta,
    set_mode_marker_scatter_data,
)
from simulation.buckling_simulation_P_coupling import (
    _DEFAULT_EI as EI,
    _DEFAULT_K_S as k_s,
    _DEFAULT_L0 as L0,
)

import figures.plot_config  # noqa: F401
from figures.plot_config import (
    annotation_fontsize,
    gray_color,
    lightgray_color,
    tick_label_fontsize,
    ms,
)

_GIF_FIGSIZE = (7.0, 3.5)
_GIF_LINE_LW = 1.5
_GIF_MODE_NS = DEFAULT_MODE_NS  # n = 1 … 6
_GIF_MODE_LABEL_ROTATION = 35                                              
_MS_MODE_NORMAL = ms * 1.4
_MS_MODE_PEAK = ms * 2.6


def make_dispersion_animation(
    t_saved,
    P_saved,
    EI_val,
    k_s_val,
    L0_val,
    mu,
    out_gif_path,
) -> None:
    k_vals, k_max_disp = get_dispersion_k_grid(EI_val, k_s_val, L0_val)

    sigma_samples = [
        compute_dispersion(k_vals, float(P_saved[i]), EI_val, k_s_val, mu)
        for i in range(0, len(t_saved), max(1, len(t_saved) // 50))
    ]
    ylim_gif = dispersion_ylim_tight(np.concatenate(sigma_samples))
    if ylim_gif is None:
        ylim_gif = (-0.015 / mu, 0.015 / mu)

    fig, ax = plt.subplots(figsize=_GIF_FIGSIZE)
    (line,) = ax.plot(
        [],
        [],
        color=lightgray_color,
        lw=_GIF_LINE_LW,
        linestyle="--",
        zorder=3,
    )

    scatter_neg = ax.scatter(
        [],
        [],
        s=[],
        facecolors="none",
        edgecolors=gray_color,
        linewidths=0.8,
        marker="o",
        zorder=4,
        clip_on=False,
    )
    scatter_pos = ax.scatter(
        [],
        [],
        s=[],
        facecolors=gray_color,
        edgecolors=gray_color,
        linewidths=0.8,
        marker="o",
        zorder=4,
        clip_on=False,
    )

    for n_mode in _GIF_MODE_NS:
        k_n = 2.0 * np.pi * n_mode / L0_val
        if 0.0 <= k_n <= k_max_disp:
            ax.axvline(k_n, color=gray_color, linestyle=":", linewidth=0.8, alpha=0.7)
            ax.text(
                k_n,
                0.0,
                rf"$n={n_mode}$",
                ha="left",
                va="bottom",
                rotation=_GIF_MODE_LABEL_ROTATION,
                rotation_mode="anchor",
                fontsize=tick_label_fontsize,
                color=gray_color,
                clip_on=False,
                zorder=5,
            )

    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.3)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top")

    ax.set_ylim(ylim_gif[0], ylim_gif[1])
    ax.set_xlim(0.0, k_max_disp)
    xlim_disp = ax.get_xlim()
    ylim_disp = ax.get_ylim()
    ax.text(
        xlim_disp[1] * 1.02,
        0.0,
        r"$k$",
        ha="left",
        va="center",
        fontsize=annotation_fontsize,
    )
    ax.text(
        0.0,
        ylim_disp[1] * 1.02,
        r"$\sigma$",
        ha="center",
        va="bottom",
        fontsize=annotation_fontsize,
    )

    def update(frame):
        P_i = float(P_saved[frame])
        sigma_vals = compute_dispersion(k_vals, P_i, EI_val, k_s_val, mu)
        line.set_data(k_vals, sigma_vals)
        mode_pts, n_peak = collect_mode_points(
            k_vals, sigma_vals, L0_val, mode_ns=_GIF_MODE_NS
        )
        set_mode_marker_scatter_data(
            scatter_neg,
            scatter_pos,
            mode_pts,
            n_peak,
            ms_normal=_MS_MODE_NORMAL,
            ms_peak=_MS_MODE_PEAK,
        )
        time_text.set_text(rf"$t={t_saved[frame]:.1f}$")
        return line, scatter_neg, scatter_pos, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(t_saved), interval=50, blit=False
    )
    try:
        ani.save(out_gif_path, writer="pillow", fps=20, dpi=150)
        print(f"保存: {out_gif_path}")
    except Exception as e:
        print(f"dispersion GIF 保存に失敗した（Pillow が必要）: {e}")
    plt.close(fig)


def main() -> None:
    print(f"GIF の保存先: {get_figures_dir()}")
    print("Loading simulation results...")
    _x, t_saved, h_saved, P_saved, meta = load_simulation_run()
    mu = mu_from_meta(meta)

    gif_path = figure_path("Gif_buckling_P_coupling_dispersion_evolution.gif")
    make_dispersion_animation(t_saved, P_saved, EI, k_s, L0, mu, gif_path)


if __name__ == "__main__":
    main()
