#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figure: Figure_max_displacement_P_coupling."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from figures.plot_config import configure_interactive_backend

configure_interactive_backend()

from simulation.simulation_output import figure_path, get_figures_dir
from simulation.simulation_results import load_simulation_run
from simulation.buckling_simulation_P_coupling import (
    P_cr_theory_linear,
    _align_twin_y_zero,
    compute_H_series,
    estimate_t_cr_sim_from_H_minimum,
    estimate_t_slow_sim_from_H_growth_rate,
    t_cr_theory_linear,
    _DEFAULT_ES,
    _DEFAULT_EI as EI,
    _DEFAULT_NATURAL_ELONGATION_RATE,
    _DEFAULT_K_S as k_s,
    _DEFAULT_L0 as L0,
)

import figures.plot_config  # noqa: F401
from figures.plot_config import (
    add_age_equivalent_top_xaxis,
    add_panel_label,
    age_offset_from_meta,
    align_axes_to_max_displacement_p_coupling_box,
    as_simulation_time,
    blue_color,
    inline_label_bbox,
    orange_color,
    save_figure_and_show,
    text_fontsize,
    twin_time_series_axes_bbox,
)

                                           
FIG_FIGSIZE = (7.0, 2.5)
                                                                                  
_CONSTRAINED_LAYOUT_W_PAD = 0.06
_CONSTRAINED_LAYOUT_H_PAD = 0.10
                                                                                                              
_LEFT_YLABEL_LABELPAD = 6.0
_RIGHT_YLABEL_LABELPAD = 6.0

_left_ylabel_coords_cache: tuple[float, tuple[float, float]] | None = None
_right_ylabel_coords_cache: tuple[float, tuple[float, float]] | None = None

                              
_INLINE_LABEL_T = 55.0
                                                       
_THREE_STAGES_YMAX_HEADROOM = 0.12
# Pre-buckling / Super-exponential growth / Slow growth
_STAGE_COLORS = ("#dbeaf4", "#f6e5c8", "#dfeedd")
_STAGE_LABELS = (
    "Pre-buckling",
    "Super-exponential growth",
    "Slow growth",
)
_STAGE_BOUNDARY_COLOR = "#555555"
_STAGE_BACKGROUND_ALPHA = 0.32


def _add_three_stage_background(
    ax: plt.Axes,
    t_start: float,
    t_cr_sim: float,
    t_slow_sim: float,
    t_end: float,
    *,
    show_labels: bool = True,
    stage_colors: tuple[str, str, str] | None = None,
    background_alpha: float | None = None,
) -> None:
    bounds = (t_start, t_cr_sim, t_slow_sim, t_end)
    if not np.all(np.isfinite(bounds)) or not (
        t_start < t_cr_sim < t_slow_sim < t_end
    ):
        return
    colors = _STAGE_COLORS if stage_colors is None else stage_colors
    alpha = (
        _STAGE_BACKGROUND_ALPHA if background_alpha is None else background_alpha
    )
    transform = ax.get_xaxis_transform()
    for left, right, color, label in zip(
        bounds[:-1], bounds[1:], colors, _STAGE_LABELS
    ):
        ax.axvspan(left, right, color=color, alpha=alpha, zorder=0)
        if show_labels:
            ax.text(
                0.5 * (left + right),
                0.97,
                label,
                transform=transform,
                ha="center",
                va="top",
                fontsize=text_fontsize - 1,
                color="black",
                zorder=5,
            )
    for boundary in (t_cr_sim, t_slow_sim):
        ax.axvline(
            boundary,
            color=_STAGE_BOUNDARY_COLOR,
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )

def apply_constrained_layout_pads(fig: plt.Figure) -> None:
    fig.set_constrained_layout_pads(
        w_pad=_CONSTRAINED_LAYOUT_W_PAD,
        h_pad=_CONSTRAINED_LAYOUT_H_PAD,
    )
    import figures.plot_config as _pc

    _pc._twin_time_series_axes_bbox_cache = None
    global _left_ylabel_coords_cache, _right_ylabel_coords_cache
    _left_ylabel_coords_cache = None
    _right_ylabel_coords_cache = None


def reference_left_ylabel_coords() -> tuple[float, float]:
    global _left_ylabel_coords_cache
    if (
        _left_ylabel_coords_cache is not None
        and _left_ylabel_coords_cache[0] == _LEFT_YLABEL_LABELPAD
    ):
        return _left_ylabel_coords_cache[1]

    fig, ax = plt.subplots(figsize=FIG_FIGSIZE, constrained_layout=True)
    apply_constrained_layout_pads(fig)
    ax_twin = ax.twinx()
    ax.set_xlabel(r"Time, $t$ (year)")
    ax.set_ylabel(
        r"Global deformation, $H(t)$ (mm)",
        color=blue_color,
        labelpad=_LEFT_YLABEL_LABELPAD,
    )
    ax_twin.set_ylabel(
        r"Axial load $P(t)$ (N)",
        color=orange_color,
        labelpad=_RIGHT_YLABEL_LABELPAD,
    )
    add_age_equivalent_top_xaxis(ax)
    align_axes_to_max_displacement_p_coupling_box(fig, ax, ax_twin)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ylab_bb = ax.yaxis.label.get_window_extent(renderer)
    inv_ax = ax.transAxes.inverted()
    y_mid = (ylab_bb.y0 + ylab_bb.y1) * 0.5
    x_axes = float(inv_ax.transform((ylab_bb.x1, y_mid))[0])
    coords = (x_axes, 0.5)
    _left_ylabel_coords_cache = (_LEFT_YLABEL_LABELPAD, coords)
    plt.close(fig)
    return coords


def apply_shared_left_ylabel_coords(ax: plt.Axes) -> None:
    x, y = reference_left_ylabel_coords()
    ref_w = float(twin_time_series_axes_bbox()[2])
    cur_w = float(ax.get_position().width)
    if cur_w > 0.0 and abs(cur_w - ref_w) > 1e-6:
        x = float(x) * (ref_w / cur_w)
    ax.yaxis.set_label_coords(x, y)


def reference_right_ylabel_coords() -> tuple[float, float]:
    global _right_ylabel_coords_cache
    if (
        _right_ylabel_coords_cache is not None
        and _right_ylabel_coords_cache[0] == _RIGHT_YLABEL_LABELPAD
    ):
        return _right_ylabel_coords_cache[1]

    fig, ax = plt.subplots(figsize=FIG_FIGSIZE, constrained_layout=True)
    apply_constrained_layout_pads(fig)
    ax_twin = ax.twinx()
    ax.set_xlabel(r"Time, $t$ (year)")
    ax.set_ylabel(
        r"Global deformation, $H(t)$ (mm)",
        color=blue_color,
        labelpad=_LEFT_YLABEL_LABELPAD,
    )
    ax_twin.set_ylabel(
        r"Axial load $P(t)$ (N)",
        color=orange_color,
        labelpad=_RIGHT_YLABEL_LABELPAD,
    )
    add_age_equivalent_top_xaxis(ax)
    align_axes_to_max_displacement_p_coupling_box(fig, ax, ax_twin)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ylab_bb = ax_twin.yaxis.label.get_window_extent(renderer)
    inv_ax = ax_twin.transAxes.inverted()
    y_mid = (ylab_bb.y0 + ylab_bb.y1) * 0.5
    x_axes = float(inv_ax.transform((ylab_bb.x0, y_mid))[0])
    coords = (x_axes, 0.5)
    _right_ylabel_coords_cache = (_RIGHT_YLABEL_LABELPAD, coords)
    plt.close(fig)
    return coords


def apply_shared_right_ylabel_coords(ax_twin: plt.Axes) -> None:
    x, y = reference_right_ylabel_coords()
    ref_w = float(twin_time_series_axes_bbox()[2])
    cur_w = float(ax_twin.get_position().width)
    if cur_w > 0.0 and abs(cur_w - ref_w) > 1e-6:
        x = 1.0 + (float(x) - 1.0) * (ref_w / cur_w)
    ax_twin.yaxis.set_label_coords(x, y)


def make_max_displacement_figure(
    t_saved: np.ndarray,
    H_series: np.ndarray,
    P_saved: np.ndarray,
    P_cr: float,
    out_pdf_path: str,
    *,
    age_offset: float | None = None,
    P_cr_sim: float | None = None,
    t_cr_sim: float | None = None,
    t_slow_sim: float | None = None,
    show_three_stages: bool = False,
    panel_label: str | None = "c",
    match_horizontal_spacetime_box: bool = False,
) -> None:
    t = as_simulation_time(t_saved, age_offset=age_offset)
    H = np.asarray(H_series, dtype=float)
    P = np.asarray(P_saved, dtype=float)

    if match_horizontal_spacetime_box:
        from simulation.Figure_h_evolution_and_spacetime import (
            horizontal_spacetime_figsize,
        )

        figsize = horizontal_spacetime_figsize()
    else:
        figsize = FIG_FIGSIZE
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    apply_constrained_layout_pads(fig)
    ax_twin = ax.twinx()
    if (
        show_three_stages
        and len(t) > 0
        and t_cr_sim is not None
        and t_slow_sim is not None
    ):
        _add_three_stage_background(
            ax,
            float(t[0]),
            float(t_cr_sim),
            float(t_slow_sim),
            float(t[-1]),
        )
    ax.plot(
        t,
        H,
        color=blue_color,
        lw=1.2,
        label=r"$H(t)$" if show_three_stages else None,
    )
    ax_twin.plot(
        t,
        P,
        color=orange_color,
        lw=1.2,
        linestyle="-",
        label=r"$P(t)$" if show_three_stages else None,
    )

    ax.set_xlabel(r"Time, $t$ (yr)" if show_three_stages else r"Time, $t$ (year)")
    ax.set_ylabel(
        r"Global deformation, $H(t)$ (mm)",
        color=blue_color,
        labelpad=_LEFT_YLABEL_LABELPAD,
    )
    ax_twin.set_ylabel(
        r"Axial force $P(t)$ (N)" if show_three_stages else r"Axial load $P(t)$ (N)",
        color=orange_color,
        labelpad=_RIGHT_YLABEL_LABELPAD,
    )
    ax.grid(True, alpha=0.3)
    ax_twin.grid(False)
    _align_twin_y_zero(ax, ax_twin, H, P)
    if show_three_stages:
        for axis in (ax, ax_twin):
            y_lo, y_hi = axis.get_ylim()
            axis.set_ylim(
                y_lo,
                y_hi + _THREE_STAGES_YMAX_HEADROOM * (y_hi - y_lo),
            )

    if np.isfinite(P_cr):
        ax_twin.axhline(
            y=P_cr,
            color=orange_color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
            label=r"$P_{\mathrm{cr}}$" if show_three_stages else None,
        )
        if not show_three_stages:
            ax_twin.annotate(
                r"$P_{\mathrm{cr}}$",
                xy=(t[-1], P_cr),
                xytext=(-2, -10),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=text_fontsize,
                color=orange_color,
                clip_on=False,
            )

    if len(t) > 0:
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.margins(x=0)
        add_age_equivalent_top_xaxis(
            ax,
            age_offset=age_offset,
            label=(
                r"Age-equivalent time, $20+t$ (yr)"
                if show_three_stages
                else None
            ),
        )
        if show_three_stages:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_twin.get_legend_handles_labels()
            ax.legend(
                h1 + h2,
                l1 + l2,
                loc="upper left",
                bbox_to_anchor=(0.01, 0.86),
                borderaxespad=0.0,
                fontsize=text_fontsize - 1,
                ncol=1,
            )
        else:
            t_label = _INLINE_LABEL_T
            idx = int(np.argmin(np.abs(t - t_label)))
            H_label = float(H[idx])
            P_label = float(P[idx])
            ax.annotate(
                r"$H(t)$",
                (t_label, H_label),
                xytext=(0, -2),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=text_fontsize,
                color=blue_color,
                clip_on=False,
                bbox=inline_label_bbox,
            )
            ax_twin.annotate(
                r"$P(t)$",
                (t_label, P_label),
                xytext=(0, 2),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=text_fontsize,
                color=orange_color,
                clip_on=False,
                bbox=inline_label_bbox,
            )

    if match_horizontal_spacetime_box:
        from simulation.Figure_h_evolution_and_spacetime import (
            align_axes_to_horizontal_spacetime_box,
            apply_shared_horizontal_left_ylabel,
        )

        align_axes_to_horizontal_spacetime_box(fig, ax, ax_twin)
        apply_shared_horizontal_left_ylabel(ax)
    else:
        align_axes_to_max_displacement_p_coupling_box(fig, ax, ax_twin)
        apply_shared_left_ylabel_coords(ax)
        apply_shared_right_ylabel_coords(ax_twin)
    if panel_label:
        add_panel_label(ax, panel_label)
    if show_three_stages:
        from simulation.Figure_h_evolution_and_spacetime import (
            apply_five_figure_label_sizes,
        )

        apply_five_figure_label_sizes(fig)
    save_figure_and_show(fig, out_pdf_path)


def main() -> None:
    print(f"図の保存先: {get_figures_dir()}")
    print("Loading simulation results...")

    P_cr = P_cr_theory_linear(EI, k_s)
    t_cr = t_cr_theory_linear(
        ES=_DEFAULT_ES,
        L0=L0,
        P_cr=P_cr,
        natural_elongation_rate=_DEFAULT_NATURAL_ELONGATION_RATE,
    )

    _x, t_saved, h_saved, P_saved, meta = load_simulation_run()
    age_offset = age_offset_from_meta(meta)

    t_saved = np.asarray(t_saved, dtype=float)
    P_saved = np.asarray(P_saved, dtype=float)
    H_series = compute_H_series(h_saved)

    cr_sim = estimate_t_cr_sim_from_H_minimum(t_saved, H_series)
    t_cr_sim = float(cr_sim["t_cr_sim"]) if cr_sim["ok"] else float("nan")
    idx_cr_sim = int(cr_sim["idx_min"]) if cr_sim["ok"] else -1
    P_cr_sim = (
        float(P_saved[idx_cr_sim])
        if cr_sim["ok"] and 0 <= idx_cr_sim < len(P_saved)
        else float("nan")
    )
    if cr_sim["ok"]:
        print(f"  t_cr_sim (H minimum) = {t_cr_sim:.1f} year  (saved series)")
        print(f"  P(t) at t_cr_sim = {P_cr_sim:.2f} N")
    t_plot = as_simulation_time(t_saved, age_offset=age_offset)
    t_cr_plot = (
        float(t_plot[idx_cr_sim])
        if cr_sim["ok"] and 0 <= idx_cr_sim < len(t_plot)
        else float("nan")
    )
    slow_sim = estimate_t_slow_sim_from_H_growth_rate(
        t_saved,
        H_series,
        idx_cr_sim=idx_cr_sim,
    )
    idx_slow_sim = int(slow_sim["idx_slow"]) if slow_sim["ok"] else -1
    t_slow_sim = (
        float(t_plot[idx_slow_sim])
        if slow_sim["ok"] and 0 <= idx_slow_sim < len(t_plot)
        else float("nan")
    )
    if slow_sim["ok"]:
        print(
            f"  t_slow_sim (min d2(log H)/dt2 after growth-rate peak) = "
            f"{t_slow_sim:.1f} year, acceleration = "
            f"{slow_sim['relative_growth_acceleration']:.4f} /year^2"
        )

    if np.isfinite(t_cr):
        print(f"  t_cr (theory)  = {t_cr:.4f} year")
    if np.isfinite(P_cr):
        print(f"  P_cr (theory) = {P_cr:.2f} N")


    out_pdf = figure_path("Figure_max_displacement_P_coupling.pdf")
    make_max_displacement_figure(
        t_saved,
        H_series,
        P_saved,
        P_cr,
        out_pdf,
        age_offset=age_offset,
        P_cr_sim=P_cr_sim,
        t_cr_sim=t_cr_sim,
    )
    make_max_displacement_figure(
        t_saved,
        H_series,
        P_saved,
        P_cr,
        figure_path("Figure_max_displacement_P_coupling_three_stages.pdf"),
        age_offset=age_offset,
        P_cr_sim=P_cr_sim,
        t_cr_sim=t_cr_plot,
        t_slow_sim=t_slow_sim,
        show_three_stages=True,
        panel_label=None,
        match_horizontal_spacetime_box=True,
    )


if __name__ == "__main__":
    main()
