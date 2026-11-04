#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select ACTIVE simulation NPZ and batch-generate main simulation figures."""
from __future__ import annotations

import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
                          
# ---------------------------------------------------------------------------
                          
ACTIVE_NPZ_PATH = (
    "data/simulation/"
    "buckling_simulation_P_coupling_age_20to80years_mu0p005_g0p35/"
    "P_coupling.npz"
)

# ---------------------------------------------------------------------------

SELECTED_FIGURES = (
    "Figure_buckling_P_coupling_h_evolution_three_snapshots_hxlim_pm50.pdf",
    "Figure_dispersion_snapshots_three_stages.pdf",
    "Figure_buckling_P_coupling_spacetime_horizontal.pdf",
    "Figure_max_displacement_P_coupling_three_stages.pdf",
    "Figure_modal_amplitude_evolution_three_stages.pdf",
)
OUTPUT_SUBDIR = "Final"


def _abs_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def npz_path_for_active_run() -> str:
    return os.path.abspath(_abs_path(ACTIVE_NPZ_PATH))


def active_run_dir() -> str:
    return os.path.dirname(npz_path_for_active_run())


def load_active_simulation():
    from simulation.simulation_results import load_simulation_run

    return load_simulation_run(npz_path_for_active_run())


def run_all_figures() -> None:
    from figures.plot_config import age_offset_from_meta, as_simulation_time
    from simulation.simulation_output import figure_path, figures_subdir
    from simulation.buckling_simulation_P_coupling import (
        P_cr_theory_linear,
        compute_H_series,
        estimate_t_cr_sim_from_H_minimum,
        estimate_t_rise_sim_from_H_rise,
        estimate_t_slow_sim_from_H_growth_rate,
        _DEFAULT_EI as EI,
        _DEFAULT_H_RISE,
        _DEFAULT_K_S as k_s,
        _DEFAULT_L0 as L0,
        _DEFAULT_N,
        _DEFAULT_NOISE_STD,
    )
    from simulation.Figure_dispersion_P_coupling import (
        make_dispersion_snapshots_three_stages_figure,
        mu_from_meta,
    )
    from simulation.Figure_h_evolution_and_spacetime import (
        make_h_evolution_three_snapshots_figure,
        make_horizontal_spacetime_figure,
        _THREE_SNAPSHOT_H_XLIM_FIXED,
        _THREE_SNAPSHOT_H_XTICKS_FIXED,
    )
    from simulation.Figure_max_displacement_P_coupling import (
        make_max_displacement_figure,
    )
    from simulation.Figure_modal_amplitude_evolution import (
        _THREE_STAGES_MODE_LINE_STYLES,
        _THREE_STAGES_PLOT_MODES,
        make_modal_amplitude_figure,
    )

    path = npz_path_for_active_run()
    print(f"ACTIVE_NPZ_PATH: {ACTIVE_NPZ_PATH}")
    tag = "present" if os.path.isfile(path) else "missing"
    print(f"  {path}: {tag}")
    out_dir = figures_subdir(OUTPUT_SUBDIR)
    print(f"Figures saved to: {out_dir}")
    print("Figures to generate:")
    for name in SELECTED_FIGURES:
        print(f"  - {name}")

    x, t_saved, h_saved, P_saved, meta = load_active_simulation()
    age_offset = age_offset_from_meta(meta)
    t_saved = np.asarray(t_saved, dtype=float)
    P_saved = np.asarray(P_saved, dtype=float)
    t_plot = as_simulation_time(t_saved, age_offset=age_offset)
    H_series = compute_H_series(h_saved)
    P_cr = P_cr_theory_linear(EI, k_s)

    cr_sim = estimate_t_cr_sim_from_H_minimum(t_saved, H_series)
    idx_cr_sim = int(cr_sim["idx_min"]) if cr_sim["ok"] else -1
    t_cr_plot = (
        float(t_plot[idx_cr_sim])
        if cr_sim["ok"] and 0 <= idx_cr_sim < len(t_plot)
        else float("nan")
    )
    P_cr_sim = (
        float(P_saved[idx_cr_sim])
        if cr_sim["ok"] and 0 <= idx_cr_sim < len(P_saved)
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

    print(f"\n{'=' * 60}")
    print(SELECTED_FIGURES[0])
    print("=" * 60)
    if rise_sim["ok"]:
        make_h_evolution_three_snapshots_figure(
            x,
            t_saved,
            h_saved,
            figure_path(SELECTED_FIGURES[0], subdir=OUTPUT_SUBDIR),
            age_offset=age_offset,
            idx_rise_sim=idx_rise_sim,
            h_xlim=_THREE_SNAPSHOT_H_XLIM_FIXED,
            h_xticks=_THREE_SNAPSHOT_H_XTICKS_FIXED,
            equal_aspect=True,
        )
    else:
        print(
            "  Skipped: t_rise_sim could not be estimated "
            f"(H never exceeded {_DEFAULT_H_RISE:g} mm)."
        )

    print(f"\n{'=' * 60}")
    print(SELECTED_FIGURES[1])
    print("=" * 60)
    make_dispersion_snapshots_three_stages_figure(
        t_saved,
        P_saved,
        h_saved,
        EI,
        k_s,
        L0,
        P_cr,
        figure_path(SELECTED_FIGURES[1], subdir=OUTPUT_SUBDIR),
        mu=mu_from_meta(meta),
        meta=meta,
    )

    print(f"\n{'=' * 60}")
    print(SELECTED_FIGURES[2])
    print("=" * 60)
    make_horizontal_spacetime_figure(
        x,
        t_saved,
        h_saved,
        _DEFAULT_NOISE_STD,
        figure_path(SELECTED_FIGURES[2], subdir=OUTPUT_SUBDIR),
        age_offset=age_offset,
    )

    print(f"\n{'=' * 60}")
    print(SELECTED_FIGURES[3])
    print("=" * 60)
    make_max_displacement_figure(
        t_saved,
        H_series,
        P_saved,
        P_cr,
        figure_path(SELECTED_FIGURES[3], subdir=OUTPUT_SUBDIR),
        age_offset=age_offset,
        P_cr_sim=P_cr_sim,
        t_cr_sim=t_cr_plot,
        t_slow_sim=t_slow_sim,
        show_three_stages=True,
        panel_label=None,
        match_horizontal_spacetime_box=True,
    )

    print(f"\n{'=' * 60}")
    print(SELECTED_FIGURES[4])
    print("=" * 60)
    make_modal_amplitude_figure(
        t_saved,
        h_saved,
        int(meta.get("N", _DEFAULT_N)),
        figure_path(SELECTED_FIGURES[4], subdir=OUTPUT_SUBDIR),
        P_saved=P_saved,
        P_cr=P_cr,
        mark_phase_markers=False,
        panel_label=None,
        age_offset=age_offset,
        meta=meta,
        t_cr_sim=t_cr_plot,
        t_slow_sim=t_slow_sim,
        show_three_stages=True,
        modes=_THREE_STAGES_PLOT_MODES,
        mode_line_styles=_THREE_STAGES_MODE_LINE_STYLES,
    )


def main() -> None:
    run_all_figures()


if __name__ == "__main__":
    main()
