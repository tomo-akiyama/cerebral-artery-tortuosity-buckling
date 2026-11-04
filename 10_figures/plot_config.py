# Plot configuration for simulation scripts
                    

# font: "Arial"
# dpi: 600
#
                                                          
                                                               
                                                     

from __future__ import annotations

import os
from typing import Any, Mapping

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps
from matplotlib.transforms import blended_transform_factory

# Typography (pt)
axes_label_fontsize = 8
tick_label_fontsize = 8
legend_fontsize = 8
label_fontsize = 12                                              
annotation_fontsize = 10                           
text_fontsize = 8                     
text_fontsize_large = annotation_fontsize        

                                                
                                         
                                   
                                             
                                                                          
                                                 
                                                                       
panel_label_pad_pt = 3.0
panel_label_fig_margin = 0.004                         

figure_dpi = 600
                                           
figure_show_seconds = 1.0

                                         
savefig_bbox_inches = None
savefig_pad_inches = 0.02


def configure_interactive_backend() -> None:
    if os.environ.get("MPLBACKEND"):
        return
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")


def save_figure(fig, fname, /, **kwargs) -> None:
    fig.savefig(
        fname,
        bbox_inches=savefig_bbox_inches,
        pad_inches=savefig_pad_inches,
        **kwargs,
    )


def show_figure(
    fig=None,
    *,
    auto_close_seconds: float | None = None,
) -> None:
    if matplotlib.get_backend().lower() == "agg":
        return
    secs = (
        float(auto_close_seconds)
        if auto_close_seconds is not None
        else None
    )
    if fig is not None:
        plt.figure(fig.number)
    if secs is None or secs <= 0:
        plt.show(block=True)
        return
    plt.show(block=False)
    plt.pause(secs)
    if fig is not None:
        plt.close(fig)
    else:
        plt.close("all")


def save_figure_and_show(
    fig,
    fname,
    /,
    *,
    show: bool = True,
    close: bool = True,
    auto_close_seconds: float | None = None,
    **kwargs,
) -> None:
    save_figure(fig, fname, **kwargs)
    print(f"保存: {fname}")
    if show:
        show_figure(fig, auto_close_seconds=auto_close_seconds)
    if close:
        plt.close(fig)


                                                                        
                                                          
REFERENCE_AXES_BBOX = [0.050, 0.131, 0.9248, 0.6811]

                                                               
TIME_SERIES_FIGSIZE = (7.0, 2.5)


def reference_axes_bbox() -> list[float]:
    return list(REFERENCE_AXES_BBOX)


def apply_reference_axes_bbox(ax: plt.Axes) -> None:
    ax.set_position(REFERENCE_AXES_BBOX)


                                                     
default_age_offset = 20.0                          
_AGE_EQUIVALENT_LABEL = r"Age-equivalent time, $20+t$ (year)"
_AGE_EQUIVALENT_TICKS = np.arange(20.0, 81.0, 10.0)


def age_offset_from_meta(meta: Mapping[str, Any] | None = None) -> float:
    if meta and "age_offset" in meta:
        return float(meta["age_offset"])
    if meta and "t_min" in meta:
        t0 = float(meta["t_min"])
        if t0 >= 18.0:
            return t0
    return default_age_offset


def as_simulation_time(
    t_calendar: np.ndarray,
    *,
    age_offset: float | None = None,
    meta: Mapping[str, Any] | None = None,
) -> np.ndarray:
    t0 = age_offset_from_meta(meta) if age_offset is None else float(age_offset)
    t = np.asarray(t_calendar, dtype=float)
    if t.size == 0:
        return t
    if float(t[0]) >= t0 - 1e-6:
        return t - t0
    return t


def add_age_equivalent_top_xaxis(
    ax: plt.Axes,
    *,
    age_offset: float | None = None,
    meta: Mapping[str, Any] | None = None,
    label: str | None = None,
) -> plt.Axes:
    t0 = age_offset_from_meta(meta) if age_offset is None else float(age_offset)
    ax_top = ax.secondary_xaxis(
        "top",
        functions=(lambda t, _t0=t0: _t0 + t, lambda age, _t0=t0: age - _t0),
    )
    ax_top.set_xlabel(_AGE_EQUIVALENT_LABEL if label is None else label)
    ax_top.set_xticks(_AGE_EQUIVALENT_TICKS)
    ax._panel_label_top_axis = ax_top
    return ax_top


def add_age_equivalent_right_yaxis(
    ax: plt.Axes,
    *,
    age_offset: float | None = None,
    meta: Mapping[str, Any] | None = None,
) -> plt.Axes:
    t0 = age_offset_from_meta(meta) if age_offset is None else float(age_offset)
    ax_right = ax.secondary_yaxis(
        "right",
        functions=(lambda t, _t0=t0: _t0 + t, lambda age, _t0=t0: age - _t0),
    )
    ax_right.set_ylabel(_AGE_EQUIVALENT_LABEL)
    ax_right.set_yticks(_AGE_EQUIVALENT_TICKS)
    return ax_right


def _top_xlabel_baseline_fig_y(ax, renderer, fig_inv) -> float | None:
    axes_bb = ax.get_window_extent(renderer).transformed(fig_inv)
    fig = ax.figure
    candidates: list = []
    tagged = getattr(ax, "_panel_label_top_axis", None)
    if tagged is not None:
        candidates.append(tagged)
    candidates.extend(fig.axes)

    best_y: float | None = None
    best_gap: float | None = None
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or candidate is ax:
            continue
        cid = id(candidate)
        if cid in seen:
            continue
        seen.add(cid)

        xlab = candidate.xaxis.label
        if not xlab.get_text().strip() or not xlab.get_visible():
            continue

        cand_bb = candidate.get_window_extent(renderer).transformed(fig_inv)
        xlab_bb = xlab.get_window_extent(renderer).transformed(fig_inv)
        x_overlap = min(cand_bb.x1, axes_bb.x1) - max(cand_bb.x0, axes_bb.x0)
        if x_overlap <= 0.01 * axes_bb.width:
            continue

        gap = xlab_bb.y0 - axes_bb.y1
        if gap < -0.02 * axes_bb.height:
            continue

        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_y = xlab_bb.y0

    return best_y


def _panel_label_annotation(
    ax,
    label: str,
    x_fig: float,
    y_fig: float | None,
    *,
    ha: str,
    pad_pt: float,
    use_axes_y_offset: bool,
):
    fig = ax.figure
    if use_axes_y_offset:
        trans = blended_transform_factory(fig.transFigure, ax.transAxes)
        return ax.annotate(
            label,
            xy=(x_fig, 1.0),
            xycoords=trans,
            xytext=(0.0, pad_pt),
            textcoords="offset points",
            fontsize=label_fontsize,
            fontweight="bold",
            ha=ha,
            va="bottom",
            clip_on=False,
            in_layout=False,
        )
    return ax.annotate(
        label,
        xy=(x_fig, y_fig),
        xycoords=fig.transFigure,
        fontsize=label_fontsize,
        fontweight="bold",
        ha=ha,
        va="bottom",
        clip_on=False,
        in_layout=False,
    )


def _measure_panel_label_anchor(
    ax,
    renderer,
    fig_inv,
    *,
    pad_in: float,
) -> tuple[float, str, float | None]:
    ylab = ax.yaxis.label
    axes_bb = ax.get_window_extent(renderer).transformed(fig_inv)
    if ylab.get_text().strip():
        ylab_bb = ylab.get_window_extent(renderer).transformed(fig_inv)
        x_fig = ylab_bb.x0 + 0.5 * ylab_bb.width
        ha = "center"
    else:
        x_fig = axes_bb.x0 + pad_in
        ha = "left"
    top_x_y = _top_xlabel_baseline_fig_y(ax, renderer, fig_inv)
    return x_fig, ha, top_x_y


def _place_panel_label(
    ax,
    label: str,
    x_fig: float,
    y_fig: float | None,
    *,
    ha: str,
    pad_pt: float,
    use_axes_y_offset: bool,
):
    fig = ax.figure
    txt = _panel_label_annotation(
        ax,
        label,
        x_fig,
        y_fig,
        ha=ha,
        pad_pt=pad_pt,
        use_axes_y_offset=use_axes_y_offset,
    )

    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    fig_inv = fig.transFigure.inverted()
    x_lo = panel_label_fig_margin
    tbb = txt.get_window_extent(renderer).transformed(fig_inv)
    if tbb.x0 >= x_lo:
        return txt

    txt.remove()
    if ha == "center":
        x_try = x_lo + 0.5 * tbb.width
        ha_try = "center"
    else:
        x_try = x_lo
        ha_try = "left"
    txt = _panel_label_annotation(
        ax,
        label,
        x_try,
        y_fig,
        ha=ha_try,
        pad_pt=pad_pt,
        use_axes_y_offset=use_axes_y_offset,
    )
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    fig_inv = fig.transFigure.inverted()
    tbb = txt.get_window_extent(renderer).transformed(fig_inv)
    if tbb.x0 < x_lo:
        txt.remove()
        txt = _panel_label_annotation(
            ax,
            label,
            x_lo,
            y_fig,
            ha="left",
            pad_pt=pad_pt,
            use_axes_y_offset=use_axes_y_offset,
        )
    return txt


def add_panel_label(
    ax,
    label: str,
    *,
    pad_pt: float | None = None,
) -> None:
    if pad_pt is None:
        pad_pt = panel_label_pad_pt

    fig = ax.figure
    pad_in = pad_pt / 72.0

    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    fig_inv = fig.transFigure.inverted()
    x_fig, ha, top_x_y = _measure_panel_label_anchor(
        ax, renderer, fig_inv, pad_in=pad_in
    )
    _place_panel_label(
        ax,
        label,
        x_fig,
        top_x_y,
        ha=ha,
        pad_pt=pad_pt,
        use_axes_y_offset=top_x_y is None,
    )

                 
sns.set_theme(
    style="whitegrid",
    context="paper",
    font="Arial",
    rc={
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    },
)

           
plt.rcParams.update({
    "xtick.bottom": True,
    "xtick.top": False,
    "ytick.left": True,
    "ytick.right": False,
})

          
plt.rcParams.update({
    "lines.linewidth": 1.2,
    "axes.titlesize": 10,
    "axes.labelsize": axes_label_fontsize,
    "xtick.labelsize": tick_label_fontsize,
    "ytick.labelsize": tick_label_fontsize,
    "legend.fontsize": legend_fontsize,
    "figure.dpi": figure_dpi,
    "savefig.dpi": figure_dpi,
    "savefig.bbox": "standard",
    "savefig.pad_inches": savefig_pad_inches,
    "mathtext.fontset": "cm",
    "mathtext.default": "it",
})

         
alpha = 0.8
ms = 3

# color rules
                         
                                                                              
                                
tab20 = colormaps["tab20"]
blue_color = tab20(0)
orange_color = tab20(2)
green_color = tab20(4)
red_color = tab20(6)
purple_color = tab20(8)
brown_color = tab20(10)
pink_color = tab20(12)
olive_color = tab20(16)
cyan_color = tab20(18)

                       
lightblue_color = tab20(1)
lightorange_color = tab20(3)
lightgreen_color = tab20(5)
lightred_color = tab20(7)
lightpurple_color = tab20(9)



gray_color = "#4d4d4d"
lightgray_color = "#8c8c8c"

                                                                       

                            
inline_label_bbox = dict(
    facecolor="white",
    edgecolor="none",
    alpha=0.7,
    pad=0.2,
)

                                                           
colors = [
    mcolors.to_rgba("#cfe8ff"),
    mcolors.to_rgba("#08306b"),
]
gray_colors = [
    mcolors.to_rgba("#e6e6e6"),
    mcolors.to_rgba("#000000"),
]

_twin_time_series_axes_bbox_cache: list[float] | None = None                      


def twin_time_series_axes_bbox() -> list[float]:
    global _twin_time_series_axes_bbox_cache
    if _twin_time_series_axes_bbox_cache is not None:
        return _twin_time_series_axes_bbox_cache

    fig, ax = plt.subplots(figsize=TIME_SERIES_FIGSIZE, constrained_layout=True)
    from simulation.Figure_max_displacement_P_coupling import (
        apply_constrained_layout_pads,
    )

    apply_constrained_layout_pads(fig)
    ax_twin = ax.twinx()
    ax.set_xlabel(r"Time, $t$ (year)")
    ax.set_ylabel(
        r"Global deformation, $H(t)$ (mm)",
        color=blue_color,
    )
    ax_twin.set_ylabel(r"Axial load $P(t)$ (N)", color=orange_color)
    add_age_equivalent_top_xaxis(ax)
    fig.canvas.draw()
    _twin_time_series_axes_bbox_cache = list(ax.get_position().bounds)
    plt.close(fig)
    return _twin_time_series_axes_bbox_cache


def align_axes_to_max_displacement_p_coupling_box(
    fig: plt.Figure,
    ax: plt.Axes,
    *extra_axes: plt.Axes,
) -> None:
    fig.canvas.draw()
    bbox = twin_time_series_axes_bbox()
    ax.set_position(bbox)
    for other in extra_axes:
        if other is not None and other is not ax:
            other.set_position(bbox)
