#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figure: Figure_max_displacement_rho."""

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
    _align_twin_y_zero,
    build_rho_approx_series,
    build_rho_series,
    compute_H_series,
)

import simulation.Figure_max_displacement_P_coupling as _p_coupling_layout
from simulation.Figure_max_displacement_P_coupling import (
    FIG_FIGSIZE,
    apply_constrained_layout_pads,
    apply_shared_left_ylabel_coords,
    apply_shared_right_ylabel_coords,
)
import figures.plot_config  # noqa: F401
from figures.plot_config import (
    add_age_equivalent_top_xaxis,
    age_offset_from_meta,
    align_axes_to_max_displacement_p_coupling_box,
    as_simulation_time,
    alpha,
    blue_color,
    legend_fontsize,
    red_color,
    save_figure_and_show,
)


def make_max_displacement_rho_figure(
    t_saved: np.ndarray,
    H_series: np.ndarray,
    rho_geom_series: np.ndarray,
    rho_approx_series: np.ndarray,
    out_pdf_path: str,
    *,
    age_offset: float | None = None,
) -> None:
    """Global amplitude H(t) + rho_geom(t), rho_approx(t)（twin y-axis）。"""
    t = as_simulation_time(t_saved, age_offset=age_offset)
    H = np.asarray(H_series, dtype=float)
    rho_geom = np.asarray(rho_geom_series, dtype=float)
    rho_approx = np.asarray(rho_approx_series, dtype=float)

    fig, ax = plt.subplots(figsize=FIG_FIGSIZE, constrained_layout=True)
    apply_constrained_layout_pads(fig)
    ax_twin = ax.twinx()
    (line_H,) = ax.plot(t, H, color=blue_color, lw=1.2, label=r"$H(t)$")
    (line_geom,) = ax_twin.plot(
        t,
        rho_geom,
        color=red_color,
        alpha=alpha,
        lw=1.2,
        linestyle="-",
        label=r"$\rho_{\mathrm{geom}}(t)$",
    )
    (line_approx,) = ax_twin.plot(
        t,
        rho_approx,
        color=red_color,
        alpha=alpha,
        lw=1.2,
        linestyle="--",
        label=r"$\rho_{\mathrm{approx}}(t)$",
    )

    ax.set_xlabel(r"Time, $t$ (year)")
    ax.set_ylabel(
        r"Global deformation, $H(t)$ (mm)",
        color=blue_color,
        labelpad=_p_coupling_layout._LEFT_YLABEL_LABELPAD,
    )
    ax_twin.set_ylabel(
        r"Excess-length ratio, $\rho(t)$",
        color=red_color,
        labelpad=_p_coupling_layout._RIGHT_YLABEL_LABELPAD,
    )
    ax.grid(True, alpha=0.3)
    ax_twin.grid(False)
    _align_twin_y_zero(ax, ax_twin, H, np.concatenate([rho_geom, rho_approx]))

    if len(t) > 0:
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.margins(x=0)
        add_age_equivalent_top_xaxis(ax, age_offset=age_offset)

    ax.legend(
        handles=[line_H, line_geom, line_approx],
        loc="best",
        fontsize=legend_fontsize,
        ncol=1,
    )

    align_axes_to_max_displacement_p_coupling_box(fig, ax, ax_twin)
    apply_shared_left_ylabel_coords(ax)
    apply_shared_right_ylabel_coords(ax_twin)
    save_figure_and_show(fig, out_pdf_path)


def main() -> None:
    print(f"図の保存先: {get_figures_dir()}")
    print("Loading simulation results...")
    print(f"  L0 (straight reference) = {L0:.1f} mm")

    x, t_saved, h_saved, _P_saved, meta = load_simulation_run()
    age_offset = age_offset_from_meta(meta)
    H_series = compute_H_series(h_saved)
    rho_geom_series = build_rho_series(h_saved, x, L0)
    rho_approx_series = build_rho_approx_series(h_saved, x, L0)

    H_final = float(H_series[-1]) if H_series.size else float("nan")
    rho_geom_final = (
        float(rho_geom_series[-1]) if rho_geom_series.size else float("nan")
    )
    rho_approx_final = (
        float(rho_approx_series[-1]) if rho_approx_series.size else float("nan")
    )
    print(f"  H(t_end)          = {H_final:.6f} mm")
    print(f"  rho_geom(t_end)   = {rho_geom_final:.6f}  (L/L0 - 1)")
    print(f"  rho_approx(t_end) = {rho_approx_final:.6f}  ((1/(2 L0)) ∫ h_x^2 dx)")
    if np.isfinite(rho_geom_final) and abs(rho_geom_final) > 0:
        rel = (rho_approx_final - rho_geom_final) / rho_geom_final
        print(f"  (approx - geom) / geom = {rel:.4e}")

    out_pdf = figure_path("Figure_max_displacement_rho.pdf")
    make_max_displacement_rho_figure(
        t_saved,
        H_series,
        rho_geom_series,
        rho_approx_series,
        out_pdf,
        age_offset=age_offset,
    )


if __name__ == "__main__":
    main()
