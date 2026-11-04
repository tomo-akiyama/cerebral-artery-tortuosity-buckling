#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figure: Figure_q_max_rho."""

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
    build_rho_approx_series,
    build_rho_series,
    compute_H_series,
    compute_q_max_series,
    estimate_t_cr_sim_from_H_minimum,
    estimate_t_slow_sim_from_H_growth_rate,
)

import simulation.Figure_max_displacement_P_coupling as _p_coupling_layout
from simulation.Figure_max_displacement_P_coupling import (
    _add_three_stage_background,
    apply_constrained_layout_pads,
)
import figures.plot_config  # noqa: F401
from figures.plot_config import (
    add_age_equivalent_top_xaxis,
    age_offset_from_meta,
    as_simulation_time,
    alpha,
    blue_color,
    legend_fontsize,
    red_color,
    save_figure_and_show,
)

_FIG_FIGSIZE = (7.0, 5.0)
# Pre-buckling / Super-exponential growth / Slow growth
_STAGE_COLORS = ("#dbeaf4", "#f6e5c8", "#dfeedd")
_STAGE_BACKGROUND_ALPHA = 0.32


def make_q_max_rho_figure(
    t_saved: np.ndarray,
    q_max_series: np.ndarray,
    rho_geom_series: np.ndarray,
    rho_approx_series: np.ndarray,
    out_pdf_path: str,
    *,
    age_offset: float | None = None,
    t_cr_sim: float | None = None,
    t_slow_sim: float | None = None,
    show_three_stages: bool = False,
) -> None:
    t = as_simulation_time(t_saved, age_offset=age_offset)
    q_max = np.asarray(q_max_series, dtype=float)
    rho_geom = np.asarray(rho_geom_series, dtype=float)
    rho_approx = np.asarray(rho_approx_series, dtype=float)

    fig, (ax_a, ax_b) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=_FIG_FIGSIZE,
        constrained_layout=True,
    )
    apply_constrained_layout_pads(fig)

    if (
        show_three_stages
        and len(t) > 0
        and t_cr_sim is not None
        and t_slow_sim is not None
    ):
        t0 = float(t[0])
        t1 = float(t[-1])
        _add_three_stage_background(
            ax_a,
            t0,
            float(t_cr_sim),
            float(t_slow_sim),
            t1,
            show_labels=True,
            stage_colors=_STAGE_COLORS,
            background_alpha=_STAGE_BACKGROUND_ALPHA,
        )
        _add_three_stage_background(
            ax_b,
            t0,
            float(t_cr_sim),
            float(t_slow_sim),
            t1,
            show_labels=True,
            stage_colors=_STAGE_COLORS,
            background_alpha=_STAGE_BACKGROUND_ALPHA,
        )

    q_color = "black" if show_three_stages else blue_color
    rho_color = "black" if show_three_stages else red_color
    line_alpha = 1.0 if show_three_stages else alpha

    ax_a.plot(t, q_max, color=q_color, lw=1.2, zorder=2)
    ax_a.set_ylabel(
        (
            r"$\max_{x} \left|\partial h/\partial x\right|$"
            if show_three_stages
            else r"$q_{\mathrm{max}}(t)=\max |h_x|$"
        ),
        labelpad=_p_coupling_layout._LEFT_YLABEL_LABELPAD,
    )
    ax_a.grid(True, alpha=0.3)
    if show_three_stages:
        ax_a.tick_params(axis="x", labelbottom=True)
        ax_a.set_xlabel(r"Time, $t$ (yr)")
    else:
        ax_a.tick_params(axis="x", labelbottom=False)

    if not show_three_stages:
        ax_b.fill_between(
            t,
            rho_geom,
            rho_approx,
            color=rho_color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
    ax_b.plot(
        t,
        rho_geom,
        color=rho_color,
        alpha=line_alpha,
        lw=1.2,
        linestyle="-",
        label=(
            r"$\rho(t)$" if show_three_stages else r"$\rho_{\mathrm{geom}}(t)$"
        ),
        zorder=2,
    )
    ax_b.plot(
        t,
        rho_approx,
        color=rho_color,
        alpha=line_alpha,
        lw=1.2,
        linestyle="--",
        label=r"$\rho_{\mathrm{approx}}(t)$",
        zorder=2,
    )
    ax_b.set_ylabel(
        (
            r"Excess-length ratio, $\rho$"
            if show_three_stages
            else r"Excess-length ratio, $\rho(t)$"
        ),
        labelpad=_p_coupling_layout._LEFT_YLABEL_LABELPAD,
    )
    ax_b.set_xlabel(
        r"Time, $t$ (yr)" if show_three_stages else r"Time, $t$ (year)"
    )
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="best", fontsize=legend_fontsize, ncol=1)

    _YLIM_NEG_FRAC = 0.06
    if q_max.size:
        q_hi = float(np.nanmax(q_max))
        if q_hi > 0:
            ax_a.set_ylim(-q_hi * _YLIM_NEG_FRAC, q_hi * 1.05)
        else:
            ax_a.set_ylim(-0.05, 1.0)
    rho_hi = float(np.nanmax(np.concatenate([rho_geom, rho_approx])))
    if np.isfinite(rho_hi):
        if rho_hi > 0:
            ax_b.set_ylim(-rho_hi * _YLIM_NEG_FRAC, rho_hi * 1.05)
        else:
            ax_b.set_ylim(-0.05, 1.0)

    if len(t) > 0:
        ax_a.set_xlim(float(t[0]), float(t[-1]))
        ax_a.margins(x=0)
        age_label = (
            r"Age-equivalent time, $20+t$ (yr)" if show_three_stages else None
        )
        add_age_equivalent_top_xaxis(
            ax_a,
            age_offset=age_offset,
            label=age_label,
        )
        if show_three_stages:
            add_age_equivalent_top_xaxis(
                ax_b,
                age_offset=age_offset,
                label=age_label,
            )

    save_figure_and_show(fig, out_pdf_path)


def main() -> None:
    print(f"Figures saved to: {get_figures_dir()}")
    print("Loading simulation results...")
    print(f"  L0 (straight reference) = {L0:.1f} mm")

    x, t_saved, h_saved, _P_saved, meta = load_simulation_run()
    age_offset = age_offset_from_meta(meta)
    t_saved = np.asarray(t_saved, dtype=float)
    H_series = compute_H_series(h_saved)
    q_max_series = compute_q_max_series(h_saved, x)
    rho_geom_series = build_rho_series(h_saved, x, L0)
    rho_approx_series = build_rho_approx_series(h_saved, x, L0)

    cr_sim = estimate_t_cr_sim_from_H_minimum(t_saved, H_series)
    idx_cr_sim = int(cr_sim["idx_min"]) if cr_sim["ok"] else -1
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
    if cr_sim["ok"]:
        print(f"  t_cr_sim (H minimum) = {t_cr_plot:.1f} year")
    if slow_sim["ok"]:
        print(f"  t_slow_sim = {t_slow_sim:.1f} year")

    q_max_final = float(q_max_series[-1]) if q_max_series.size else float("nan")
    q_max_peak = float(np.max(q_max_series)) if q_max_series.size else float("nan")
    rho_geom_final = (
        float(rho_geom_series[-1]) if rho_geom_series.size else float("nan")
    )
    rho_approx_final = (
        float(rho_approx_series[-1]) if rho_approx_series.size else float("nan")
    )
    print(f"  q_max(t_end)      = {q_max_final:.6f}")
    print(f"  q_max peak        = {q_max_peak:.6f}")
    print(f"  rho_geom(t_end)   = {rho_geom_final:.6f}  (L/L0 - 1)")
    print(f"  rho_approx(t_end) = {rho_approx_final:.6f}  ((1/(2 L0)) ∫ h_x^2 dx)")
    if np.isfinite(rho_geom_final) and abs(rho_geom_final) > 0:
        rel = (rho_approx_final - rho_geom_final) / rho_geom_final
        print(f"  (approx - geom) / geom = {rel:.4e}")

    make_q_max_rho_figure(
        t_saved,
        q_max_series,
        rho_geom_series,
        rho_approx_series,
        figure_path("Figure_q_max_rho.pdf"),
        age_offset=age_offset,
    )
    make_q_max_rho_figure(
        t_saved,
        q_max_series,
        rho_geom_series,
        rho_approx_series,
        figure_path("Figure_q_max_rho_three_stages.pdf"),
        age_offset=age_offset,
        t_cr_sim=t_cr_plot,
        t_slow_sim=t_slow_sim,
        show_three_stages=True,
    )


if __name__ == "__main__":
    main()
