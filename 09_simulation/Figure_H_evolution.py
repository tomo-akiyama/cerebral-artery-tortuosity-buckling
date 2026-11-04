#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figure: Figure_H_evolution."""

from __future__ import annotations

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from figures.plot_config import configure_interactive_backend

configure_interactive_backend()

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from simulation.simulation_output import figure_path, get_figures_dir
from simulation.simulation_results import load_simulation_run
from simulation.buckling_simulation_P_coupling import compute_H_series
from simulation.Figure_modal_amplitude_evolution import (
    _AMPLITUDE_EPS,
)
import figures.plot_config  # noqa: F401
from figures.plot_config import (
    age_offset_from_meta,
    as_simulation_time,
    blue_color,
    save_figure_and_show,
)

         
_H_LINEWIDTH = 2.0
            
_YLIM: tuple[float, float] = (1.0e-4, 2.0e1)
                                  
_STAGE_BOUNDARY_TIMES: tuple[float, ...] = (14.0, 33.0)
_STAGE_BOUNDARY_COLOR = "#555555"
_STAGE_BOUNDARY_LW = 1.0

                                    
_AXES_LABEL_FONTSIZE = 10              
_TICK_LABEL_FONTSIZE = 8               

                                                       
_CM_TO_INCH = 1.0 / 2.54
_FIG_WIDTH_CM = 10.5
FIGSIZE = (_FIG_WIDTH_CM * _CM_TO_INCH, 7.5 * _CM_TO_INCH)
FIGSIZE_NO_AGE = (_FIG_WIDTH_CM * _CM_TO_INCH, 5.0 * _CM_TO_INCH)
                                                     
_LEFT_MARGIN_IN = 0.72
_RIGHT_MARGIN_IN = 0.18
_BOTTOM_MARGIN_IN = 0.95
_BOTTOM_MARGIN_IN_NO_AGE = 0.55
_TOP_MARGIN_IN = 0.12
_LEFT_YLABEL_LABELPAD = 6.0

                                                         
_AGE_XAXIS_AXES_Y = -0.28
_AGE_EQUIVALENT_LABEL = r"Age-equivalent time, $20+t$ (year)"
_AGE_EQUIVALENT_TICKS = np.arange(20.0, 81.0, 10.0)


def _set_panel_position(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    bottom_margin_in: float = _BOTTOM_MARGIN_IN,
) -> None:
    fw, fh = fig.get_size_inches()
    ax.set_position(
        [
            _LEFT_MARGIN_IN / float(fw),
            bottom_margin_in / float(fh),
            (float(fw) - _LEFT_MARGIN_IN - _RIGHT_MARGIN_IN) / float(fw),
            (float(fh) - bottom_margin_in - _TOP_MARGIN_IN) / float(fh),
        ]
    )


def _apply_local_fontsizes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=_TICK_LABEL_FONTSIZE)
    ax.xaxis.label.set_fontsize(_AXES_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(_AXES_LABEL_FONTSIZE)


def _disable_x_minor_ticks(ax: plt.Axes) -> None:
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)


def _style_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_age_equivalent_below_xaxis(
    ax: plt.Axes,
    *,
    age_offset: float | None = None,
) -> plt.Axes:
    t0 = (
        age_offset_from_meta(None)
        if age_offset is None
        else float(age_offset)
    )
    ax_age = ax.secondary_xaxis(
        _AGE_XAXIS_AXES_Y,
        functions=(
            lambda t, _t0=t0: _t0 + t,
            lambda age, _t0=t0: age - _t0,
        ),
    )
    ax_age.set_xlabel(
        _AGE_EQUIVALENT_LABEL,
        fontsize=_AXES_LABEL_FONTSIZE,
    )
    ax_age.set_xticks(_AGE_EQUIVALENT_TICKS)
    ax_age.tick_params(axis="x", which="major", labelsize=_TICK_LABEL_FONTSIZE)
    _disable_x_minor_ticks(ax_age)
    ax_age.spines["top"].set_visible(False)
    ax_age.spines["right"].set_visible(False)
    return ax_age


def make_H_evolution_figure(
    t_saved: np.ndarray,
    h_saved,
    out_pdf_path: str,
    *,
    age_offset: float | None = None,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    show_age_equivalent: bool = True,
) -> None:
    t = as_simulation_time(t_saved, age_offset=age_offset)
    H_t = np.maximum(compute_H_series(h_saved), _AMPLITUDE_EPS)

    size = FIGSIZE if figsize is None else figsize
    fig, ax = plt.subplots(figsize=size)
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)

    ax.plot(
        t,
        H_t,
        color=blue_color,
        lw=_H_LINEWIDTH,
        linestyle="-",
        zorder=2,
    )

    if len(t) > 0:
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.margins(x=0)

    ax.set_yscale("log")
    ax.set_ylim(*_YLIM)
    _disable_x_minor_ticks(ax)
                                               
    ax.tick_params(axis="y", which="minor", left=True, right=False)
    ax.set_xlabel(
        r"Time, $t$ (years)"
        if not show_age_equivalent
        else r"Time, $t$ (year)",
        fontsize=_AXES_LABEL_FONTSIZE,
    )
    ax.set_ylabel(
        r"Global deformation, $H(t)$ (mm)",
        fontsize=_AXES_LABEL_FONTSIZE,
        labelpad=_LEFT_YLABEL_LABELPAD,
    )
    ax.grid(False)
    _style_spines(ax)
    _apply_local_fontsizes(ax)

    for t_b in _STAGE_BOUNDARY_TIMES:
        ax.axvline(
            float(t_b),
            color=_STAGE_BOUNDARY_COLOR,
            linestyle="--",
            linewidth=_STAGE_BOUNDARY_LW,
            zorder=1,
        )

    if show_age_equivalent and len(t) > 0:
        _add_age_equivalent_below_xaxis(ax, age_offset=age_offset)

    bottom = (
        _BOTTOM_MARGIN_IN if show_age_equivalent else _BOTTOM_MARGIN_IN_NO_AGE
    )
    _set_panel_position(fig, ax, bottom_margin_in=bottom)
    save_figure_and_show(fig, out_pdf_path, show=show)


def main() -> None:
    print(f"図の保存先: {get_figures_dir()}")
    print(
        f"Figure size (standard): {FIGSIZE[0] * 2.54:.2f} cm (W) × "
        f"{FIGSIZE[1] * 2.54:.2f} cm (H)"
    )
    print(
        f"Figure size (no-age): {FIGSIZE_NO_AGE[0] * 2.54:.2f} cm (W) × "
        f"{FIGSIZE_NO_AGE[1] * 2.54:.2f} cm (H)"
    )
    print(
        f"Fonts: axes labels {_AXES_LABEL_FONTSIZE} pt, "
        f"ticks {_TICK_LABEL_FONTSIZE} pt"
    )
    print("Loading simulation results...")
    _x, t_saved, h_saved, _P_saved, meta = load_simulation_run()
    age_offset = age_offset_from_meta(meta)
    t_arr = np.asarray(t_saved, dtype=float)

    make_H_evolution_figure(
        t_arr,
        h_saved,
        figure_path("Figure_H_evolution.pdf"),
        age_offset=age_offset,
    )
    make_H_evolution_figure(
        t_arr,
        h_saved,
        figure_path("Figure_H_evolution_no_age.pdf"),
        age_offset=age_offset,
        figsize=FIGSIZE_NO_AGE,
        show_age_equivalent=False,
    )


if __name__ == "__main__":
    main()
