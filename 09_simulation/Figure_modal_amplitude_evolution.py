#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figure: Figure_modal_amplitude_evolution."""

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
from simulation.buckling_simulation_P_coupling import (
    P_cr_theory_linear,
    compute_H_series,
    estimate_t_cr_sim_from_H_minimum,
    estimate_t_slow_sim_from_H_growth_rate,
    _DEFAULT_EI as EI,
    _DEFAULT_K_S as k_s,
    _DEFAULT_L0 as L0,
    _DEFAULT_N,
)
from simulation.Figure_dispersion_P_coupling import (
    find_dispersion_snapshot_indices,
    get_dispersion_k_grid,
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
    add_panel_label,
    age_offset_from_meta,
    alpha,
    as_simulation_time,
    blue_color,
    align_axes_to_max_displacement_p_coupling_box,
    green_color,
    ms,
    purple_color,
    red_color,
    save_figure_and_show,
    text_fontsize,
)

                                                         
_DEFAULT_PLOT_MODES: tuple[int, ...] = (2, 3)
                                                                             
_MODE_LINE_STYLES: tuple[dict[str, object], ...] = (
    {"color": green_color, "lw": 1.2, "linestyle": "--"},          
    {"color": purple_color, "lw": 1.2, "linestyle": "-."},            
)
                          
_THREE_STAGES_PLOT_MODES: tuple[int, ...] = (1, 2, 3)
_THREE_STAGES_MODE_LINE_STYLES: tuple[dict[str, object], ...] = (
    {"color": green_color, "lw": 1.2, "linestyle": ":"},          
    {"color": red_color, "lw": 1.2, "linestyle": "--"},          
    {"color": purple_color, "lw": 1.2, "linestyle": "-."},            
)
_THREE_STAGES_YLIM: tuple[float, float] = (5.0e-5, 3.0e1)
_AMPLITUDE_EPS = 1e-12             
# Pre-buckling / Super-exponential growth / Slow growth
_STAGE_COLORS = ("#dbeaf4", "#f6e5c8", "#dfeedd")
_STAGE_LABELS = (
    "Pre-buckling",
    "Super-exponential growth",
    "Slow growth",
)
_STAGE_BOUNDARY_COLOR = "#555555"
_STAGE_BACKGROUND_ALPHA = 0.32


def _add_three_stage_annotations(
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

    transform = ax.get_xaxis_transform()
    for left, right, color, label in zip(
        bounds[:-1], bounds[1:], _STAGE_COLORS, _STAGE_LABELS
    ):
        ax.axvspan(
            left, right, color=color, alpha=_STAGE_BACKGROUND_ALPHA, zorder=0
        )
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


def modal_amplitude_rfft(h: np.ndarray, n: int, N: int) -> float:
    a = np.fft.rfft(h - np.mean(h))
    if n >= a.size:
        return 0.0
    return float(np.abs(a[n]) * 2.0 / float(N))


def build_mode_amplitude_series(
    h_saved, N: int, modes: tuple[int, ...] = _DEFAULT_PLOT_MODES
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for n in modes:
        out[n] = np.array(
            [modal_amplitude_rfft(h, n, N) for h in h_saved], dtype=float
        )
    return out


def _mode_amplitude_line_style(
    n: int,
    modes: tuple[int, ...],
    *,
    style_hints: tuple[dict[str, object], ...] | None = _MODE_LINE_STYLES,
) -> dict:
    idx = modes.index(int(n)) if int(n) in modes else (int(n) - 1)
    hint: dict[str, object] = {}
    if style_hints is not None and 0 <= idx < len(style_hints):
        hint = style_hints[idx]

    color = hint.get("color")
    if color is None:
        color = plt.cm.tab10((int(n) - 1) % 10)
    lw = float(hint.get("lw", 1.2))
    linestyle = str(hint.get("linestyle", "-"))

    return {
        "color": color,
        "lw": lw,
        "label": rf"mode $n={n}$",
        "linestyle": linestyle,
        "alpha": alpha,
    }


def _add_phase_markers_on_H(
    ax,
    t_plot: np.ndarray,
    H_plot: np.ndarray,
    t_saved: np.ndarray,
    P_saved: np.ndarray,
    h_saved,
    P_cr: float,
    *,
    EI_val: float,
    k_s_val: float,
    L0_val: float,
    age_offset: float | None = None,
    meta: dict | None = None,
) -> None:
    k_vals, _ = get_dispersion_k_grid(EI_val, k_s_val, L0_val)
    snapshots = find_dispersion_snapshot_indices(
        t_saved,
        P_saved,
        h_saved,
        k_vals,
        EI_val,
        k_s_val,
        L0_val,
        P_cr,
        age_offset=age_offset,
        meta=meta,
    )
    t = np.asarray(t_plot, dtype=float)
    H = np.asarray(H_plot, dtype=float)
    marker_ms = ms * 1.8
    for idx, _, _ in snapshots:
        t_i = float(t[idx])
        y_i = float(H[idx])
        ax.plot(
            t_i,
            y_i,
            marker="o",
            markersize=marker_ms,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.9,
            linestyle="none",
            zorder=6,
            clip_on=False,
        )


def make_modal_amplitude_figure(
    t_saved: np.ndarray,
    h_saved,
    N: int,
    out_pdf_path: str,
    *,
    P_saved: np.ndarray | None = None,
    P_cr: float | None = None,
    mark_phase_markers: bool = False,
    panel_label: str | None = None,
    age_offset: float | None = None,
    meta: dict | None = None,
    show: bool = True,
    modes: tuple[int, ...] | None = None,
    mode_line_styles: tuple[dict[str, object], ...] | None = None,
    t_cr_sim: float | None = None,
    t_slow_sim: float | None = None,
    show_three_stages: bool = False,
) -> None:
    plot_modes = tuple(modes) if modes is not None else _DEFAULT_PLOT_MODES
    style_hints = (
        mode_line_styles
        if mode_line_styles is not None
        else _MODE_LINE_STYLES
    )

    t = as_simulation_time(t_saved, age_offset=age_offset)
    amps_by_mode = build_mode_amplitude_series(h_saved, N, plot_modes)
    H_raw = compute_H_series(h_saved)
    H_t = np.maximum(H_raw, _AMPLITUDE_EPS)

    if show_three_stages:
        from simulation.Figure_h_evolution_and_spacetime import (
            horizontal_spacetime_figsize,
        )

        figsize = horizontal_spacetime_figsize()
    else:
        figsize = FIG_FIGSIZE
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    apply_constrained_layout_pads(fig)
    if (
        show_three_stages
        and len(t) > 0
        and t_cr_sim is not None
        and t_slow_sim is not None
    ):
        _add_three_stage_annotations(
            ax,
            float(t[0]),
            float(t_cr_sim),
            float(t_slow_sim),
            float(t[-1]),
        )

    ax.plot(
        t,
        H_t,
        color=blue_color,
        lw=1.2,
        linestyle="-",
        label=r"$H(t)$",
        zorder=2,
    )

    if mark_phase_markers and P_saved is not None:
        p_cr = (
            float(P_cr)
            if P_cr is not None and np.isfinite(P_cr)
            else float(P_cr_theory_linear(EI_val=EI, k_s_val=k_s))
        )
        _add_phase_markers_on_H(
            ax,
            t,
            H_t,
            t_saved,
            np.asarray(P_saved, dtype=float),
            h_saved,
            p_cr,
            EI_val=EI,
            k_s_val=k_s,
            L0_val=L0,
            age_offset=age_offset,
            meta=meta,
        )

    for n in plot_modes:
        amps = np.maximum(amps_by_mode[n], _AMPLITUDE_EPS)
        ax.plot(
            t,
            amps,
            zorder=3,
            **_mode_amplitude_line_style(
                int(n), plot_modes, style_hints=style_hints
            ),
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

    ax.set_yscale("log")
    if show_three_stages:
        ax.set_ylim(*_THREE_STAGES_YLIM)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="y", which="minor", left=False)
    ax.set_xlabel(r"Time, $t$ (yr)" if show_three_stages else r"Time, $t$ (year)")
    ax.set_ylabel(
        r"Deformation magnitude (mm)",
        labelpad=_p_coupling_layout._LEFT_YLABEL_LABELPAD,
    )
    ax.grid(True, alpha=0.3, axis="y", which="major")

    if show_three_stages:
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 0.14),
            fontsize=text_fontsize - 1,
            ncol=1,
        )
    else:
        ax.legend(loc="best", fontsize=text_fontsize - 1, ncol=1)

    if show_three_stages:
        from simulation.Figure_h_evolution_and_spacetime import (
            align_axes_to_horizontal_spacetime_box,
            apply_shared_horizontal_left_ylabel,
        )

        align_axes_to_horizontal_spacetime_box(fig, ax)
        apply_shared_horizontal_left_ylabel(ax)
    else:
        align_axes_to_max_displacement_p_coupling_box(fig, ax)
        apply_shared_left_ylabel_coords(ax)
    if panel_label:
        add_panel_label(ax, panel_label)
    if show_three_stages:
        from simulation.Figure_h_evolution_and_spacetime import (
            apply_five_figure_label_sizes,
        )

        apply_five_figure_label_sizes(fig)
    save_figure_and_show(fig, out_pdf_path, show=show)


def main() -> None:
    print(f"Figures saved to: {get_figures_dir()}")
    print("Loading simulation results...")
    _x, t_saved, h_saved, P_saved, meta = load_simulation_run()
    age_offset = age_offset_from_meta(meta)
    t_saved = np.asarray(t_saved, dtype=float)
    P_saved = np.asarray(P_saved, dtype=float)
    P_cr = P_cr_theory_linear(EI, k_s)
    H_series = compute_H_series(h_saved)
    cr_sim = estimate_t_cr_sim_from_H_minimum(t_saved, H_series)
    t_cr_sim = float(cr_sim["t_cr_sim"]) if cr_sim["ok"] else float("nan")
    t_plot = as_simulation_time(t_saved, age_offset=age_offset)
    idx_cr_sim = int(cr_sim["idx_min"]) if cr_sim["ok"] else -1
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
    if np.isfinite(P_cr):
        print(f"  P_cr (theory) = {P_cr:.6f} N")
    if cr_sim["ok"]:
        print(
            f"  t_cr_sim (H minimum) = {t_cr_sim:.4f} year, "
            f"H_min = {cr_sim['H_min']:.6e}"
        )
    make_modal_amplitude_figure(
        t_saved,
        h_saved,
        int(meta.get("N", _DEFAULT_N)),
        figure_path("Figure_modal_amplitude_evolution.pdf"),
        P_saved=P_saved,
        P_cr=P_cr,
        mark_phase_markers=False,
        panel_label="d",
        age_offset=age_offset,
        meta=meta,
    )
    make_modal_amplitude_figure(
        t_saved,
        h_saved,
        int(meta.get("N", _DEFAULT_N)),
        figure_path("Figure_modal_amplitude_evolution_three_stages.pdf"),
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


if __name__ == "__main__":
    main()
