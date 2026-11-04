#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figure: Figure_h_evolution_and_spacetime."""

from __future__ import annotations

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from figures.plot_config import configure_interactive_backend

configure_interactive_backend()

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from simulation.simulation_output import figure_path, get_figures_dir
from simulation.simulation_results import load_simulation_run
from simulation.buckling_simulation_P_coupling import (
    compute_H_series,
    estimate_t_cr_sim_from_H_minimum,
    estimate_t_rise_sim_from_H_rise,
    estimate_t_slow_sim_from_H_growth_rate,
    _times_to_saved_indices,
    _DEFAULT_H_RISE,
    _DEFAULT_NOISE_STD,
    _DEFAULT_PLOT_TIME_INTERVAL,
)
import figures.plot_config  # noqa: F401
from figures.plot_config import (
    add_age_equivalent_top_xaxis,
    add_age_equivalent_right_yaxis,
    add_panel_label,
    age_offset_from_meta,
    align_axes_to_max_displacement_p_coupling_box,
    as_simulation_time,
    gray_colors,
    save_figure_and_show,
    text_fontsize,
)

                      
_FIG_FIGSIZE = (7.0, 2.5)
                                    
_HORIZONTAL_SPACETIME_FIGSIZE = (4.5, 3.0 * 4.5 / 8.4)
                               
_CONSTRAINED_LAYOUT_W_PAD = 0.04
_CONSTRAINED_LAYOUT_H_PAD = 0.20
                          
_LEFT_YLABEL_LABELPAD = 6.0
_EVO_XLIM = (0.0, 100.0)         
_ST_XLIM = (0.0, 100.0)         
_EVO_XTICKS = np.arange(0.0, 101.0, 20.0)             
_EVO_YLIM_MARGIN = 1.1                              
_WIDTH_RATIOS = (1.82, 1.0)                       
_PANEL_AB_WSPACE = 0.13           
_HORIZONTAL_COLORBAR_GAP = 0.008
_HORIZONTAL_COLORBAR_WIDTH = 0.015
_HORIZONTAL_COLORBAR_LABELPAD = 4.0
_HORIZONTAL_COLORBAR_FIGURE_PAD = 0.006
_THREE_SNAPSHOT_PANEL_WSPACE = 0.04
_THREE_SNAPSHOT_LAYOUT_RECT = (0.0, 0.0, 0.999, 1.0)
                                                            
                                    
_THREE_SNAPSHOT_EQUAL_FIGSIZE = (4.5, 3.8 * 4.5 / 8.4)
_SNAPSHOT_LINE_LW = 2.0
_SNAPSHOT_PANEL_FACECOLOR = "black"
_SNAPSHOT_LINE_COLOR = "white"
_SNAPSHOT_AXIS_COLOR = "black"
_SNAPSHOT_GRID_COLOR = (1.0, 1.0, 1.0, 0.25)
_THREE_SNAPSHOT_H_XLIM_FIXED = (-50.0, 50.0)
_THREE_SNAPSHOT_H_XTICKS_FIXED = np.arange(-50.0, 51.0, 20.0)
_STAGE_COLORS = ("#dbeaf4", "#f6e5c8", "#dfeedd")
_STAGE_BOUNDARY_COLOR = "#555555"


def _add_three_stage_background_to_spacetime(
    ax: plt.Axes,
    t_start: float,
    t_cr_sim: float,
    t_slow_sim: float,
    t_end: float,
) -> None:
    bounds = (t_start, t_cr_sim, t_slow_sim, t_end)
    if not np.all(np.isfinite(bounds)) or not (
        t_start < t_cr_sim < t_slow_sim < t_end
    ):
        return
    for bottom, top, color in zip(bounds[:-1], bounds[1:], _STAGE_COLORS):
        ax.axhspan(bottom, top, color=color, alpha=0.14, zorder=2)
    for boundary in (t_cr_sim, t_slow_sim):
        ax.axhline(
            boundary,
            color=_STAGE_BOUNDARY_COLOR,
            linestyle="--",
            linewidth=1.0,
            zorder=3,
        )


def apply_constrained_layout_pads(fig: plt.Figure) -> None:
    fig.set_constrained_layout_pads(
        w_pad=_CONSTRAINED_LAYOUT_W_PAD,
        h_pad=_CONSTRAINED_LAYOUT_H_PAD,
    )


def apply_three_snapshot_equal_layout_pads(fig: plt.Figure) -> None:
    fig.set_constrained_layout_pads(
        w_pad=_CONSTRAINED_LAYOUT_W_PAD,
        h_pad=_CONSTRAINED_LAYOUT_H_PAD,
        wspace=_THREE_SNAPSHOT_PANEL_WSPACE,
        rect=_THREE_SNAPSHOT_LAYOUT_RECT,
    )


_three_snapshot_equal_axes_bbox_cache: list[list[float]] | None = None


def three_snapshot_equal_axes_bboxes() -> list[list[float]]:
    global _three_snapshot_equal_axes_bbox_cache
    if _three_snapshot_equal_axes_bbox_cache is not None:
        return [list(b) for b in _three_snapshot_equal_axes_bbox_cache]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=_THREE_SNAPSHOT_EQUAL_FIGSIZE,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )
    apply_three_snapshot_equal_layout_pads(fig)
    for i_ax, ax in enumerate(axes):
        ax.set_facecolor(_SNAPSHOT_PANEL_FACECOLOR)
        ax.set_xlabel(
            r"Displacement, $h(x,t)$ (mm)",
            color=_SNAPSHOT_AXIS_COLOR,
        )
        if i_ax == 0:
            ax.set_ylabel(
                r"Spatial coordinate, $x$ (mm)",
                labelpad=_LEFT_YLABEL_LABELPAD,
                color=_SNAPSHOT_AXIS_COLOR,
            )
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlim(*_THREE_SNAPSHOT_H_XLIM_FIXED)
        ax.set_xticks(_THREE_SNAPSHOT_H_XTICKS_FIXED)
        ax.set_ylim(*_EVO_XLIM)
        ax.set_yticks(_EVO_XTICKS)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.grid(True, color=_SNAPSHOT_GRID_COLOR)
        ax.set_title(r"$t=0$ yr", fontsize=text_fontsize, pad=6, color=_SNAPSHOT_AXIS_COLOR)
        ax.tick_params(
            axis="both",
            colors=_SNAPSHOT_AXIS_COLOR,
            which="both",
        )
        for spine in ax.spines.values():
            spine.set_color(_SNAPSHOT_AXIS_COLOR)
    fig.canvas.draw()
    _three_snapshot_equal_axes_bbox_cache = [
        list(ax.get_position().bounds) for ax in axes
    ]
    plt.close(fig)
    return [list(b) for b in _three_snapshot_equal_axes_bbox_cache]


def align_axes_to_three_snapshot_equal_boxes(
    fig: plt.Figure,
    axes,
) -> None:
    fig.canvas.draw()
    bboxes = three_snapshot_equal_axes_bboxes()
    _disable_figure_layout(fig)
    for ax, bbox in zip(axes, bboxes):
        _exclude_from_layout(ax)
        ax.set_aspect("auto")
        ax.set_box_aspect(None)
        ax.set_position(bbox)


def _exclude_from_layout(ax: plt.Axes) -> None:
    setter = getattr(ax, "set_in_layout", None)
    if callable(setter):
        setter(False)


def _disable_figure_layout(fig: plt.Figure) -> None:
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)
    else:
        fig.set_constrained_layout(False)


def _place_colorbar_right_of_aligned_axes(
    fig: plt.Figure,
    ax: plt.Axes,
    mappable,
    label: str,
):
    fig.canvas.draw()
    _disable_figure_layout(fig)
    _exclude_from_layout(ax)
    cax = fig.add_axes([0.0, 0.0, 0.01, 0.01])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, labelpad=_HORIZONTAL_COLORBAR_LABELPAD)
    _exclude_from_layout(cbar.ax)

    def _set_cbar_position(panel) -> None:
        cbar.ax.set_position(
            [
                panel.x1 + _HORIZONTAL_COLORBAR_GAP,
                panel.y0,
                _HORIZONTAL_COLORBAR_WIDTH,
                panel.height,
            ]
        )

    fig.canvas.draw()
    _set_cbar_position(ax.get_position())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_right = float(fig.bbox.x1)
    right_edges = [cbar.ax.yaxis.label.get_window_extent(renderer).x1]
    right_edges.extend(
        tick.get_window_extent(renderer).x1
        for tick in cbar.ax.get_yticklabels()
        if tick.get_visible() and tick.get_text()
    )
    overflow_px = max(right_edges) - fig_right + (
        _HORIZONTAL_COLORBAR_FIGURE_PAD * float(fig.bbox.width)
    )
    if overflow_px > 0:
        overflow = overflow_px / float(fig.bbox.width)
        panel = ax.get_position()
        ax.set_position(
            [panel.x0, panel.y0, max(panel.width - overflow, 0.1), panel.height]
        )
        _set_cbar_position(ax.get_position())
    return cbar


_horizontal_spacetime_axes_bbox_cache: list[float] | None = None
_horizontal_spacetime_figsize_cache: tuple[float, float] | None = None
_horizontal_left_ylabel_axes_x_cache: float | None = None
_horizontal_left_ylabel_offset_pt_cache: float | None = None


def horizontal_spacetime_figsize() -> tuple[float, float]:
    horizontal_spacetime_axes_bbox()
    return (
        float(_horizontal_spacetime_figsize_cache[0]),
        float(_horizontal_spacetime_figsize_cache[1]),
    )


def horizontal_spacetime_axes_bbox() -> list[float]:
    global _horizontal_spacetime_axes_bbox_cache
    global _horizontal_spacetime_figsize_cache
    global _horizontal_left_ylabel_axes_x_cache
    global _horizontal_left_ylabel_offset_pt_cache
    if (
        _horizontal_spacetime_axes_bbox_cache is not None
        and _horizontal_spacetime_figsize_cache is not None
        and _horizontal_left_ylabel_axes_x_cache is not None
        and _horizontal_left_ylabel_offset_pt_cache is not None
    ):
        return list(_horizontal_spacetime_axes_bbox_cache)

    fig, ax = plt.subplots(
        figsize=_HORIZONTAL_SPACETIME_FIGSIZE,
        constrained_layout=True,
    )
    apply_constrained_layout_pads(fig)
    t_dummy = np.linspace(0.0, 60.0, 16)
    x_dummy = np.linspace(0.0, 100.0, 16)
    z_dummy = np.linspace(-10.0, 10.0, 16).reshape(-1, 1) * np.ones((1, 16))
    norm = mcolors.SymLogNorm(linthresh=0.01, vmin=-10.0, vmax=10.0, base=10)
    pc = ax.pcolormesh(
        t_dummy,
        x_dummy,
        z_dummy,
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
    )
    ax.set_xlabel(r"Time, $t$ (year)")
    ax.set_ylabel(
        r"Spatial coordinate, $x$ (mm)",
        labelpad=_LEFT_YLABEL_LABELPAD,
    )
    ax.set_xlim(0.0, 60.0)
    add_age_equivalent_top_xaxis(ax)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(_EVO_XTICKS)
    _place_colorbar_right_of_aligned_axes(
        fig,
        ax,
        pc,
        r"Displacement, $h(x,t)$ (mm)",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ylab_bb = ax.yaxis.label.get_window_extent(renderer)
    y_mid = (ylab_bb.y0 + ylab_bb.y1) * 0.5
    spine_x = float(ax.transAxes.transform((0.0, 0.5))[0])
    _horizontal_left_ylabel_axes_x_cache = float(
        ax.transAxes.inverted().transform((ylab_bb.x1, y_mid))[0]
    )
    _horizontal_left_ylabel_offset_pt_cache = (
        (spine_x - float(ylab_bb.x1)) / float(fig.dpi) * 72.0
    )
    base_w, base_h = _HORIZONTAL_SPACETIME_FIGSIZE
    pos = ax.get_position()
    bottom_in = float(pos.y0) * base_h
    top_in = (1.0 - float(pos.y0) - float(pos.height)) * base_h
    snap_box = three_snapshot_equal_axes_bboxes()[0]
    panel_h = float(snap_box[3]) * float(_THREE_SNAPSHOT_EQUAL_FIGSIZE[1])
    fig_h = bottom_in + panel_h + top_in
    _horizontal_spacetime_figsize_cache = (float(base_w), fig_h)
    _horizontal_spacetime_axes_bbox_cache = [
        float(pos.x0),
        bottom_in / fig_h,
        float(pos.width),
        panel_h / fig_h,
    ]
    plt.close(fig)
    return list(_horizontal_spacetime_axes_bbox_cache)


def reference_horizontal_left_ylabel_axes_x() -> float:
    if _horizontal_left_ylabel_axes_x_cache is None:
        horizontal_spacetime_axes_bbox()
    return float(_horizontal_left_ylabel_axes_x_cache)


def reference_horizontal_left_ylabel_offset_pt() -> float:
    if _horizontal_left_ylabel_offset_pt_cache is None:
        horizontal_spacetime_axes_bbox()
    return float(_horizontal_left_ylabel_offset_pt_cache)


def apply_shared_horizontal_left_ylabel(ax: plt.Axes) -> None:
    fig = ax.figure
    fig.canvas.draw()
    offset_pt = (
        reference_horizontal_left_ylabel_offset_pt()
        - _FIVE_FIGURE_LEFT_YLABEL_CLOSER_PT
    )
    spine_x = float(ax.transAxes.transform((0.0, 0.5))[0])
    target_x = spine_x - offset_pt / 72.0 * float(fig.dpi)
    x_axes = float(ax.transAxes.inverted().transform((target_x, 0.0))[0])
    ax.yaxis.set_label_coords(x_axes, 0.5)


_FIVE_FIGURE_LABEL_FONTSIZE = 6.0
                                               
_FIVE_FIGURE_LEFT_YLABEL_CLOSER_PT = 4.0
                                  
_FIVE_FIGURE_XLABEL_CLOSER_PT = 2.0
                            
_FIVE_FIGURE_XLABEL_MIN_TICK_GAP_PT = 1.5


def _iter_figure_axes(fig: plt.Figure):
    pending = list(fig.axes)
    seen: set[int] = set()
    while pending:
        ax = pending.pop()
        if id(ax) in seen:
            continue
        seen.add(id(ax))
        pending.extend(getattr(ax, "child_axes", ()))
        yield ax


def _xlabel_gap_to_ticks_pt(ax: plt.Axes, renderer) -> float | None:
    label = ax.xaxis.label
    if not label.get_text().strip():
        return None
    ticks = [
        tick.get_window_extent(renderer)
        for tick in ax.get_xticklabels()
        if tick.get_text().strip()
    ]
    if not ticks:
        return None
    box = label.get_window_extent(renderer)
    dpi = float(ax.figure.dpi)
    if ax.xaxis.get_label_position() == "bottom":
        return (min(tick.y0 for tick in ticks) - box.y1) / dpi * 72.0
    if ax.xaxis.get_label_position() == "top":
        return (box.y0 - max(tick.y1 for tick in ticks)) / dpi * 72.0
    return None


def _move_xlabels_closer_to_axes(fig: plt.Figure) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in _iter_figure_axes(fig):
        gap_pt = _xlabel_gap_to_ticks_pt(ax, renderer)
        if gap_pt is None:
            continue
        move_pt = min(
            _FIVE_FIGURE_XLABEL_CLOSER_PT,
            max(0.0, gap_pt - _FIVE_FIGURE_XLABEL_MIN_TICK_GAP_PT),
        )
        if move_pt <= 0.0:
            continue
        pad = ax.xaxis.labelpad
        if pad is None:
            pad = float(plt.rcParams.get("axes.labelpad", 4.0))
        ax.xaxis.labelpad = float(pad) - move_pt


def apply_five_figure_label_sizes(fig: plt.Figure) -> None:
    size = _FIVE_FIGURE_LABEL_FONTSIZE
    for ax in _iter_figure_axes(fig):
        ax.xaxis.label.set_fontsize(size)
        ax.yaxis.label.set_fontsize(size)
        ax.title.set_fontsize(size)
        ax.tick_params(axis="both", which="both", labelsize=size)
        for text in ax.texts:
            text.set_fontsize(size)
        legend = ax.get_legend()
        if legend is None:
            continue
        if hasattr(legend, "set_fontsize"):
            legend.set_fontsize(size)
        for text in legend.get_texts():
            text.set_fontsize(size)
    _move_xlabels_closer_to_axes(fig)


def align_axes_to_horizontal_spacetime_box(
    fig: plt.Figure,
    ax: plt.Axes,
    *extra_axes: plt.Axes,
) -> None:
    fig.set_size_inches(*horizontal_spacetime_figsize(), forward=True)
    fig.canvas.draw()
    bbox = horizontal_spacetime_axes_bbox()
    _disable_figure_layout(fig)
    _exclude_from_layout(ax)
    ax.set_position(bbox)
    for other in extra_axes:
        if other is not None and other is not ax:
            _exclude_from_layout(other)
            other.set_position(bbox)


def fit_yaxis_labels_inside_figure(
    fig: plt.Figure,
    *axes: plt.Axes,
    pad: float = 0.004,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w = float(fig.bbox.width)
    dpi = float(fig.dpi)
    fig_left = float(fig.bbox.x0) + pad * fig_w
    fig_right = float(fig.bbox.x1) - pad * fig_w
    for ax in axes:
        if ax is None:
            continue
        lab = ax.yaxis.label
        if not lab.get_text():
            continue
        bb = lab.get_window_extent(renderer)
        overflow_px = 0.0
        if bb.x0 < fig_left:
            overflow_px = fig_left - bb.x0
        elif bb.x1 > fig_right:
            overflow_px = bb.x1 - fig_right
        if overflow_px < 0.5:
            continue
        current = ax.yaxis.labelpad
        if current is None:
            current = plt.rcParams.get("axes.labelpad", 4.0)
        ax.yaxis.labelpad = float(current) - overflow_px / dpi * 72.0
    fig.canvas.draw()


def _three_snapshot_indices(
    t_plot: np.ndarray,
    *,
    idx_rise_sim: int,
) -> list[tuple[int, str, str]]:
    t = np.asarray(t_plot, dtype=float)
    n = t.size
    if n == 0:
        raise ValueError("empty t_plot")

    idx_pre = 0
    idx_end = n - 1

    def _t_title(t_val: float) -> str:
        t_f = float(t_val)
        if abs(t_f) < 1e-9:
            return r"$t=0$ yr"
        return rf"$t={t_f:.1f}$ yr"

    return [
        (idx_pre, "Pre-buckling", _t_title(0.0)),
        (idx_rise_sim, "Super-exponent growth", _t_title(t[idx_rise_sim])),
        (idx_end, "Slow growth", _t_title(t[idx_end])),
    ]


def _profile_indices(t_saved: np.ndarray, plot_time_interval: float) -> np.ndarray:
    t_end = float(t_saved[-1])
    times_to_plot = np.arange(0.0, t_end + 0.01, plot_time_interval)
    if t_end > 0 and (
        len(times_to_plot) == 0
        or not np.isclose(times_to_plot[-1], t_end, rtol=0, atol=1e-6)
    ):
        if len(times_to_plot) == 0 or abs(times_to_plot[-1] - t_end) > 1e-9:
            times_to_plot = np.unique(np.append(times_to_plot, t_end))
    return np.asarray(_times_to_saved_indices(t_saved, times_to_plot), dtype=int)


def make_h_evolution_and_spacetime_figure(
    x: np.ndarray,
    t_saved: np.ndarray,
    h_saved,
    noise_std: float,
    plot_time_interval: float,
    out_pdf_path: str,
    *,
    age_offset: float | None = None,
    t_cr_sim: float | None = None,
    t_slow_sim: float | None = None,
    show_three_stages: bool = False,
) -> None:
    t_saved = np.asarray(t_saved, dtype=float)
    t_plot = as_simulation_time(t_saved, age_offset=age_offset)
    h_arr = np.asarray(h_saved, dtype=float)
    indices = _profile_indices(t_saved, plot_time_interval)
    idx_last = len(h_saved) - 1
    n_profiles = max(len(indices), 1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "time_gray", gray_colors, N=n_profiles
    )

    fig, (ax_evo, ax_st) = plt.subplots(
        1,
        2,
        figsize=_FIG_FIGSIZE,
        constrained_layout=True,
        gridspec_kw={"width_ratios": list(_WIDTH_RATIOS), "wspace": _PANEL_AB_WSPACE},
    )
    apply_constrained_layout_pads(fig)

    for j_e, i in enumerate(indices):
        is_final = i == idx_last
        t_norm = j_e / max(n_profiles - 1, 1)
        ax_evo.plot(
            x,
            h_saved[i],
            color=cmap(t_norm),
            lw=2.6 if is_final else 1.2,
            linestyle="-",
            zorder=10 if is_final else 2,
        )
    ax_evo.set_xlabel(r"Spatial coordinate, $x$ (mm)")
    ax_evo.set_ylabel(
        r"Displacement, $h(x,t)$ (mm)",
        labelpad=_LEFT_YLABEL_LABELPAD,
    )
    ax_evo.set_xlim(left=_EVO_XLIM[0], right=_EVO_XLIM[1])
    ax_evo.set_xticks(_EVO_XTICKS)
    h_profiles = np.concatenate(
        [np.asarray(h_saved[i], dtype=float).ravel() for i in indices]
    )
    v_evo = float(np.max(np.abs(h_profiles))) if h_profiles.size else 1.0
    y_lim = _EVO_YLIM_MARGIN * v_evo
    ax_evo.set_ylim(-y_lim, y_lim)
    ax_evo.grid(True, alpha=0.3)

    v = float(np.max(np.abs(h_arr)))
    norm = mcolors.SymLogNorm(linthresh=noise_std, vmin=-v, vmax=v, base=10)
    pc = ax_st.pcolormesh(x, t_plot, h_arr, cmap="RdBu_r", norm=norm, shading="auto")
    if (
        show_three_stages
        and len(t_plot) > 0
        and t_cr_sim is not None
        and t_slow_sim is not None
    ):
        _add_three_stage_background_to_spacetime(
            ax_st,
            float(t_plot[0]),
            float(t_cr_sim),
            float(t_slow_sim),
            float(t_plot[-1]),
        )
    ax_st.set_xlim(left=_ST_XLIM[0], right=_ST_XLIM[1])
    ax_st.set_xlabel(r"Spatial coordinate, $x$ (mm)")
    ax_st.set_ylabel(r"Time, $t$ (year)")
    if len(t_plot) > 0:
        ax_st.set_ylim(float(t_plot[0]), float(t_plot[-1]))
        add_age_equivalent_right_yaxis(ax_st, age_offset=age_offset)
    fig.colorbar(
        pc,
        ax=ax_st,
        label=r"Displacement, $h(x,t)$ (mm)",
        location="left",
    )

    ax_st.set_xticks(_EVO_XTICKS)

    for ax, lbl in zip((ax_evo, ax_st), ("a", "b")):
        add_panel_label(ax, lbl)

    save_figure_and_show(fig, out_pdf_path)


def make_horizontal_spacetime_figure(
    x: np.ndarray,
    t_saved: np.ndarray,
    h_saved,
    noise_std: float,
    out_pdf_path: str,
    *,
    age_offset: float | None = None,
) -> None:
    t_saved = np.asarray(t_saved, dtype=float)
    t_plot = as_simulation_time(t_saved, age_offset=age_offset)
    x = np.asarray(x, dtype=float)
    h_arr = np.asarray(h_saved, dtype=float)

    fig, ax = plt.subplots(
        figsize=horizontal_spacetime_figsize(),
        constrained_layout=True,
    )
    apply_constrained_layout_pads(fig)

    v = float(np.max(np.abs(h_arr)))
                                                   
    v_lim = max(v, 10.0)
    norm = mcolors.SymLogNorm(linthresh=noise_std, vmin=-v_lim, vmax=v_lim, base=10)
    pc = ax.pcolormesh(
        t_plot,
        x,
        h_arr.T,
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
    )

    ax.set_xlabel(r"Time, $t$ (yr)")
    ax.set_ylabel(
        r"Spatial coordinate, $x$ (mm)",
        labelpad=_LEFT_YLABEL_LABELPAD,
    )
    if len(t_plot) > 0:
        ax.set_xlim(float(t_plot[0]), float(t_plot[-1]))
        add_age_equivalent_top_xaxis(
            ax,
            age_offset=age_offset,
            label=r"Age-equivalent time, $20+t$ (yr)",
        )
    ax.set_ylim(float(x[0]), float(x[-1]))
    ax.set_yticks(_EVO_XTICKS)
    cbar = _place_colorbar_right_of_aligned_axes(
        fig,
        ax,
        pc,
        r"Displacement, $h(x,t)$ (mm)",
    )
    align_axes_to_horizontal_spacetime_box(fig, ax)
    apply_shared_horizontal_left_ylabel(ax)
    panel = ax.get_position()
    cbar.ax.set_position(
        [
            panel.x1 + _HORIZONTAL_COLORBAR_GAP,
            panel.y0,
            _HORIZONTAL_COLORBAR_WIDTH,
            panel.height,
        ]
    )
                                                  
    for tick in cbar.ax.get_yticklabels():
        tick.set_clip_on(False)
    apply_five_figure_label_sizes(fig)
    save_figure_and_show(fig, out_pdf_path)


def make_h_evolution_three_snapshots_figure(
    x: np.ndarray,
    t_saved: np.ndarray,
    h_saved,
    out_pdf_path: str,
    *,
    age_offset: float | None = None,
    idx_rise_sim: int,
    h_xlim: tuple[float, float] | None = None,
    h_xticks: np.ndarray | None = None,
    equal_aspect: bool = False,
) -> None:
    t_saved = np.asarray(t_saved, dtype=float)
    t_plot = as_simulation_time(t_saved, age_offset=age_offset)
    snapshots = _three_snapshot_indices(
        t_plot,
        idx_rise_sim=idx_rise_sim,
    )

    if h_xlim is None:
        h_profiles = np.concatenate(
            [
                np.asarray(h_saved[idx], dtype=float).ravel()
                for idx, _, _ in snapshots
            ]
        )
        v_evo = float(np.max(np.abs(h_profiles))) if h_profiles.size else 1.0
        h_xlim_use = (-_EVO_YLIM_MARGIN * v_evo, _EVO_YLIM_MARGIN * v_evo)
    else:
        h_xlim_use = (float(h_xlim[0]), float(h_xlim[1]))

    figsize = _THREE_SNAPSHOT_EQUAL_FIGSIZE if equal_aspect else _FIG_FIGSIZE
    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )
    if equal_aspect:
        apply_three_snapshot_equal_layout_pads(fig)
    else:
        fig.set_constrained_layout_pads(
            w_pad=_CONSTRAINED_LAYOUT_W_PAD,
            h_pad=_CONSTRAINED_LAYOUT_H_PAD,
            wspace=_THREE_SNAPSHOT_PANEL_WSPACE,
            rect=_THREE_SNAPSHOT_LAYOUT_RECT,
        )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    print(f"\n--- h evolution three snapshots ({out_pdf_path}) ---")
    for i_ax, (ax, (idx, phase_name, t_subtitle)) in enumerate(
        zip(axes, snapshots)
    ):
        ax.set_facecolor(_SNAPSHOT_PANEL_FACECOLOR)
        ax.plot(
            h_saved[idx],
            x,
            color=_SNAPSHOT_LINE_COLOR,
            lw=_SNAPSHOT_LINE_LW,
            linestyle="-",
            zorder=2,
        )
        ax.set_xlabel(
            r"Displacement, $h(x,t)$ (mm)",
            color=_SNAPSHOT_AXIS_COLOR,
        )
        if i_ax == 0:
            ax.set_ylabel(
                r"Spatial coordinate, $x$ (mm)",
                labelpad=_LEFT_YLABEL_LABELPAD,
                color=_SNAPSHOT_AXIS_COLOR,
            )
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlim(h_xlim_use[0], h_xlim_use[1])
        if h_xticks is not None:
            ax.set_xticks(h_xticks)
        ax.set_ylim(_EVO_XLIM[0], _EVO_XLIM[1])
        ax.set_yticks(_EVO_XTICKS)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(1)
        ax.grid(True, color=_SNAPSHOT_GRID_COLOR)
        ax.set_title(
            t_subtitle,
            fontsize=text_fontsize,
            pad=6,
            color=_SNAPSHOT_AXIS_COLOR,
        )
        ax.tick_params(
            axis="both",
            colors=_SNAPSHOT_AXIS_COLOR,
            which="both",
        )
        for spine in ax.spines.values():
            spine.set_color(_SNAPSHOT_AXIS_COLOR)
        print(
            f"  [{phase_name}] index={idx}, t={float(t_plot[idx]):.1f}"
        )

    if equal_aspect:
        align_axes_to_three_snapshot_equal_boxes(fig, axes)
        apply_shared_horizontal_left_ylabel(axes[0])
        fig.canvas.draw()
        for i_ax, ax in enumerate(axes):
            bbox = ax.get_window_extent()
            aspect = float(bbox.width / bbox.height) if bbox.height else float("nan")
            print(
                f"  panel {i_ax} display size {bbox.width:.1f} x {bbox.height:.1f} px "
                f"(aspect={aspect:.3f})"
            )
        apply_five_figure_label_sizes(fig)

    save_figure_and_show(fig, out_pdf_path)


def make_horizontal_h_evolution_figure(
    x: np.ndarray,
    t_saved: np.ndarray,
    h_saved,
    plot_time_interval: float,
    out_pdf_path: str,
) -> None:
    t_saved = np.asarray(t_saved, dtype=float)
    indices = _profile_indices(t_saved, plot_time_interval)
    idx_last = len(h_saved) - 1
    n_profiles = max(len(indices), 1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "time_gray_horizontal", gray_colors, N=n_profiles
    )

    fig, ax = plt.subplots(figsize=_FIG_FIGSIZE, constrained_layout=True)
    apply_constrained_layout_pads(fig)

    for j_e, i in enumerate(indices):
        is_final = i == idx_last
        t_norm = j_e / max(n_profiles - 1, 1)
        ax.plot(
            x,
            h_saved[i],
            color=cmap(t_norm),
            lw=2.6 if is_final else 1.2,
            linestyle="-",
            zorder=10 if is_final else 2,
        )

    ax.set_xlabel(r"Spatial coordinate, $x$ (mm)")
    ax.set_ylabel(
        r"Displacement, $h(x,t)$ (mm)",
        labelpad=_LEFT_YLABEL_LABELPAD,
    )
    ax.set_xlim(left=_EVO_XLIM[0], right=_EVO_XLIM[1])
    ax.set_xticks(_EVO_XTICKS)
    h_profiles = np.concatenate(
        [np.asarray(h_saved[i], dtype=float).ravel() for i in indices]
    )
    v_evo = float(np.max(np.abs(h_profiles))) if h_profiles.size else 1.0
    y_lim = _EVO_YLIM_MARGIN * v_evo
    ax.set_ylim(-y_lim, y_lim)
    ax.grid(True, alpha=0.3)

    align_axes_to_max_displacement_p_coupling_box(fig, ax)
    save_figure_and_show(fig, out_pdf_path)


def main() -> None:
    print("Loading simulation results...")
    print(f"Figures saved to: {get_figures_dir()}")
    x, t_saved, h_saved, _P_saved, meta = load_simulation_run()
    age_offset = age_offset_from_meta(meta)
    t_saved = np.asarray(t_saved, dtype=float)
    t_plot = as_simulation_time(t_saved, age_offset=age_offset)
    H_series = compute_H_series(h_saved)
    cr_sim = estimate_t_cr_sim_from_H_minimum(
        t_saved,
        H_series,
    )
    idx_cr_sim = int(cr_sim["idx_min"]) if cr_sim["ok"] else -1
    t_cr_sim = (
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
    rise_sim = estimate_t_rise_sim_from_H_rise(
        t_saved,
        H_series,
        _DEFAULT_H_RISE,
    )
    idx_rise_sim = int(rise_sim["idx_cross"]) if rise_sim["ok"] else -1
    t_rise_sim = (
        float(t_plot[idx_rise_sim])
        if rise_sim["ok"] and 0 <= idx_rise_sim < len(t_plot)
        else float("nan")
    )
    if rise_sim["ok"]:
        print(
            f"  t_rise_sim (H > {_DEFAULT_H_RISE:g} mm) = {t_rise_sim:.1f} year"
        )
    out_pdf = figure_path("Figure_buckling_P_coupling_h_evolution_and_spacetime.pdf")
    make_h_evolution_and_spacetime_figure(
        x,
        t_saved,
        h_saved,
        _DEFAULT_NOISE_STD,
        _DEFAULT_PLOT_TIME_INTERVAL,
        out_pdf,
        age_offset=age_offset,
    )
    make_h_evolution_and_spacetime_figure(
        x,
        t_saved,
        h_saved,
        _DEFAULT_NOISE_STD,
        _DEFAULT_PLOT_TIME_INTERVAL,
        figure_path(
            "Figure_buckling_P_coupling_h_evolution_and_spacetime_three_stages.pdf"
        ),
        age_offset=age_offset,
        t_cr_sim=t_cr_sim,
        t_slow_sim=t_slow_sim,
        show_three_stages=True,
    )
    if rise_sim["ok"]:
        make_h_evolution_three_snapshots_figure(
            x,
            t_saved,
            h_saved,
            figure_path(
                "Figure_buckling_P_coupling_h_evolution_three_snapshots.pdf"
            ),
            age_offset=age_offset,
            idx_rise_sim=idx_rise_sim,
        )
        make_h_evolution_three_snapshots_figure(
            x,
            t_saved,
            h_saved,
            figure_path(
                "Figure_buckling_P_coupling_h_evolution_three_snapshots_hxlim_pm50.pdf"
            ),
            age_offset=age_offset,
            idx_rise_sim=idx_rise_sim,
            h_xlim=_THREE_SNAPSHOT_H_XLIM_FIXED,
            h_xticks=_THREE_SNAPSHOT_H_XTICKS_FIXED,
            equal_aspect=True,
        )
    else:
        print(
            "  Skipped h evolution three snapshots: "
            f"t_rise_sim could not be estimated (H never exceeded {_DEFAULT_H_RISE:g} mm)."
        )
    make_horizontal_h_evolution_figure(
        x,
        t_saved,
        h_saved,
        _DEFAULT_PLOT_TIME_INTERVAL,
        figure_path("Figure_buckling_P_coupling_h_evolution_horizontal.pdf"),
    )
    out_horizontal_pdf = figure_path(
        "Figure_buckling_P_coupling_spacetime_horizontal.pdf"
    )
    make_horizontal_spacetime_figure(
        x,
        t_saved,
        h_saved,
        _DEFAULT_NOISE_STD,
        out_horizontal_pdf,
        age_offset=age_offset,
    )


if __name__ == "__main__":
    main()
