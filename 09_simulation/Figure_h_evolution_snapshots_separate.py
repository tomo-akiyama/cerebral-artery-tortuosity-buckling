#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figure: Figure_h_evolution_snapshots_separate."""

from __future__ import annotations

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from figures.plot_config import configure_interactive_backend

configure_interactive_backend()

import matplotlib.pyplot as plt
import numpy as np

from simulation.simulation_output import figure_path, get_figures_dir
from simulation.simulation_results import load_simulation_run
from simulation.buckling_simulation_P_coupling import (
    compute_H_series,
    estimate_t_rise_sim_from_H_rise,
    _DEFAULT_H_RISE,
)
from simulation.Figure_h_evolution_and_spacetime import (
    _EVO_XLIM,
    _EVO_YLIM_MARGIN,
    _three_snapshot_indices,
)
import figures.plot_config  # noqa: F401
from figures.plot_config import (
    age_offset_from_meta,
    as_simulation_time,
    pink_color,
    save_figure_and_show,
)

                        
_FIGSIZE = (3.2, 3.2)
    
_LINEWIDTH = 5.0

_OUTPUT_SPECS = (
    ("Figure_buckling_P_coupling_h_evolution_snapshot_01_t0.pdf", "t0"),
    ("Figure_buckling_P_coupling_h_evolution_snapshot_02_trise.pdf", "trise"),
    ("Figure_buckling_P_coupling_h_evolution_snapshot_03_tend.pdf", "tend"),
)


def _style_blank_panel(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")
    ax.grid(False)
    ax.set_axis_off()


def make_single_h_snapshot_figure(
    x: np.ndarray,
    h_profile: np.ndarray,
    out_pdf_path: str,
    *,
    h_ylim: tuple[float, float],
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)

    ax.plot(
        np.asarray(x, dtype=float),
        np.asarray(h_profile, dtype=float),
        color=pink_color,
        lw=_LINEWIDTH,
        linestyle="-",
        zorder=2,
    )
    ax.set_xlim(_EVO_XLIM[0], _EVO_XLIM[1])
    ax.set_ylim(float(h_ylim[0]), float(h_ylim[1]))
    ax.set_aspect("equal", adjustable="box")
    _style_blank_panel(ax)

                          
    ax.set_position([0.02, 0.02, 0.96, 0.96])
    save_figure_and_show(fig, out_pdf_path, show=show)


def main() -> None:
    print(f"Figures saved to: {get_figures_dir()}")
    print("Loading simulation results...")
    x, t_saved, h_saved, _P_saved, meta = load_simulation_run()
    age_offset = age_offset_from_meta(meta)
    t_saved = np.asarray(t_saved, dtype=float)
    t_plot = as_simulation_time(t_saved, age_offset=age_offset)

    H_series = compute_H_series(h_saved)
    rise_sim = estimate_t_rise_sim_from_H_rise(
        t_saved, H_series, _DEFAULT_H_RISE
    )
    if not rise_sim.get("ok"):
        raise RuntimeError(
            f"t_rise_sim could not be estimated "
            f"(H never exceeded {_DEFAULT_H_RISE:g} mm)."
        )
    idx_rise_sim = int(rise_sim["idx_cross"])
    snapshots = _three_snapshot_indices(
        t_plot, idx_rise_sim=idx_rise_sim
    )

    h_profiles = np.concatenate(
        [
            np.asarray(h_saved[idx], dtype=float).ravel()
            for idx, _, _ in snapshots
        ]
    )
    v_evo = float(np.max(np.abs(h_profiles))) if h_profiles.size else 1.0
    h_ylim = (-_EVO_YLIM_MARGIN * v_evo, _EVO_YLIM_MARGIN * v_evo)

    print("\n--- separate h evolution snapshots ---")
    for (out_name, _tag), (idx, phase_name, t_subtitle) in zip(
        _OUTPUT_SPECS, snapshots
    ):
        out_pdf = figure_path(out_name)
        print(
            f"  [{phase_name}] index={idx}, "
            f"t={float(t_plot[idx]):.1f}, title={t_subtitle}"
        )
        make_single_h_snapshot_figure(
            x,
            h_saved[idx],
            out_pdf,
            h_ylim=h_ylim,
            show=False,
        )


if __name__ == "__main__":
    main()
