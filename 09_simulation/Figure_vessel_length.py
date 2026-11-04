#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figure: Figure_vessel_length."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from figures.plot_config import configure_interactive_backend

configure_interactive_backend()

from simulation.simulation_output import figure_path, get_figures_dir
from simulation.simulation_results import load_simulation_run
from simulation.buckling_simulation_P_coupling import (
    _DEFAULT_L0 as L0,
    build_L_series,
)

import simulation.Figure_max_displacement_P_coupling as _p_coupling_layout
from simulation.Figure_max_displacement_P_coupling import (
    FIG_FIGSIZE,
    apply_constrained_layout_pads,
    apply_shared_left_ylabel_coords,
)
import figures.plot_config  # noqa: F401
from figures.plot_config import (
    add_age_equivalent_top_xaxis,
    age_offset_from_meta,
    align_axes_to_max_displacement_p_coupling_box,
    as_simulation_time,
    blue_color,
    save_figure_and_show,
)


def make_vessel_length_figure(
    t_saved: np.ndarray,
    L_series: np.ndarray,
    out_pdf_path: str,
    *,
    age_offset: float | None = None,
) -> None:
    """Centerline length L(t)。"""
    t = as_simulation_time(t_saved, age_offset=age_offset)
    L = np.asarray(L_series, dtype=float)

    fig, ax = plt.subplots(figsize=FIG_FIGSIZE, constrained_layout=True)
    apply_constrained_layout_pads(fig)
    ax.plot(t, L, color=blue_color, lw=1.2)

    ax.set_xlabel(r"Time, $t$ (year)")
    ax.set_ylabel(
        r"Vessel length, $L(t)$ (mm)",
        labelpad=_p_coupling_layout._LEFT_YLABEL_LABELPAD,
    )
    ax.grid(True, alpha=0.3)

    if len(t) > 0:
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.margins(x=0)
        add_age_equivalent_top_xaxis(ax, age_offset=age_offset)

    align_axes_to_max_displacement_p_coupling_box(fig, ax)
    apply_shared_left_ylabel_coords(ax)
    save_figure_and_show(fig, out_pdf_path)


def main() -> None:
    print(f"Figures saved to: {get_figures_dir()}")
    print("Loading simulation results...")
    print(f"  L0 (straight reference) = {L0:.1f} mm")

    x, t_saved, h_saved, _P_saved, meta = load_simulation_run()
    age_offset = age_offset_from_meta(meta)
    L_series = build_L_series(h_saved, x)

    L_start = float(L_series[0]) if L_series.size else float("nan")
    L_end = float(L_series[-1]) if L_series.size else float("nan")
    print(f"  L(t_start) = {L_start:.6f} mm")
    print(f"  L(t_end)   = {L_end:.6f} mm")

    make_vessel_length_figure(
        t_saved,
        L_series,
        figure_path("Figure_vessel_length.pdf"),
        age_offset=age_offset,
    )


if __name__ == "__main__":
    main()
