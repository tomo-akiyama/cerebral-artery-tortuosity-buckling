#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper figure: Figure_dispersion_P_coupling."""

import os
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from figures.plot_config import configure_interactive_backend

configure_interactive_backend()

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from simulation.simulation_output import figure_path, get_figures_dir
from simulation.simulation_results import load_simulation_run
from simulation.buckling_simulation_P_coupling import (
    compute_H_series,
    compute_sigma,
    estimate_t_cr_sim_from_H_minimum,
    estimate_t_rise_sim_from_H_rise,
    _DEFAULT_EI as EI,
    _DEFAULT_H_RISE,
    _DEFAULT_K_S as k_s,
    _DEFAULT_L0 as L0,
    _DEFAULT_MU,
)

import figures.plot_config  # noqa: F401 — sns theme + rcParams
from figures.plot_config import (
    add_panel_label,
    age_offset_from_meta,
    as_simulation_time,
    gray_color,
    label_fontsize,
    lightgray_color,
    ms,
    panel_label_pad_pt,
    save_figure_and_show,
    text_fontsize,
)
from matplotlib.transforms import blended_transform_factory


def _add_panel_label_above_yaxis(ax, label: str) -> None:
    fig = ax.figure
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    fig_inv = fig.transFigure.inverted()
    axes_bb = ax.get_window_extent(renderer).transformed(fig_inv)
    x_fig = axes_bb.x0
    trans = blended_transform_factory(fig.transFigure, ax.transAxes)
    ax.annotate(
        label,
        xy=(x_fig, 1.0),
        xycoords=trans,
        xytext=(0.0, panel_label_pad_pt),
        textcoords="offset points",
        fontsize=label_fontsize,
        fontweight="bold",
        ha="center",
        va="bottom",
        clip_on=False,
    )

                      
_FIG_FIGSIZE = (7.0, 2.5)
                               
_CONSTRAINED_LAYOUT_W_PAD = 0.06
_CONSTRAINED_LAYOUT_H_PAD = 0.10
                             
_LEFT_YLABEL_LABELPAD = 6.0
_SNAPSHOT_LINE_LW = 2.0
_SNAPSHOT_XTICK_STEP = 0.1
DEFAULT_MODE_NS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
_SNAPSHOT_MODE_NS = DEFAULT_MODE_NS
_SNAPSHOT_TRANSIENT_T = 20.0                                                       
_SNAPSHOT_PANEL_TITLES: Tuple[str, ...] = (
    "Phase I",
    "Phase II",
    "Phase III",
    "Phase IV",
)
_THREE_STAGE_PANEL_TITLES: Tuple[str, ...] = (
    "Pre-buckling",
    "Super-exponent growth",
    "Slow growth",
)
_MS_MODE_NORMAL = ms * 1.4
_MS_MODE_PEAK = ms * 2.6
_SNAPSHOT_MODE_LABEL_ROTATION = 35
_SNAPSHOT_PANEL_WSPACE = 0.04                                    
_SNAPSHOT_LAYOUT_RECT = (0.0, 0.0, 0.999, 1.0)                   

                                  
_SNAPSHOT_XLIM: tuple[float, float] = (0.0, 0.35)
                                                               
_SNAPSHOT_YLIM: tuple[float, float] = (-6.0, 2.0)


def apply_constrained_layout_pads(fig: plt.Figure) -> None:
    fig.set_constrained_layout_pads(
        w_pad=_CONSTRAINED_LAYOUT_W_PAD,
        h_pad=_CONSTRAINED_LAYOUT_H_PAD,
        wspace=_SNAPSHOT_PANEL_WSPACE,
        rect=_SNAPSHOT_LAYOUT_RECT,
    )


                                                 
_P_NEAR_P_CR_REL_TOL = 0.08
_POST_PEAK_TIME_FRAC = 0.25


def mu_from_meta(meta, *, default: float = _DEFAULT_MU) -> float:
    if meta and "mu" in meta:
        return float(meta["mu"])
    return float(default)


def compute_dispersion(k_vals, P_val, EI_val, k_s_val, mu_val):
    """σ(k,t) = D(k,t) / μ,  D = P k² - EI k⁴ - k_s"""
    return compute_sigma(k_vals, P_val, EI_val, k_s_val, mu_val)


def get_dispersion_k_grid(EI_val, k_s_val, L0_val, n_pts=1000):
    k_cr_local = (k_s_val / EI_val) ** 0.25
    k_max_disp = 2.5 * k_cr_local
    k_vals = np.linspace(0.0, k_max_disp, n_pts)
    return k_vals, k_max_disp


@dataclass(frozen=True)
class ModePoint:
    n: int
    k: float
    sigma: float


def collect_mode_points(
    k_vals: np.ndarray,
    sigma: np.ndarray,
    L0_val: float,
    mode_ns: Sequence[int] = DEFAULT_MODE_NS,
    y_range: Optional[Tuple[float, float]] = None,
) -> Tuple[list[ModePoint], Optional[int]]:
    k_max = float(k_vals[-1]) if k_vals.size else 0.0
    points: list[ModePoint] = []
    for n_mode in mode_ns:
        k_n = 2.0 * np.pi * float(n_mode) / float(L0_val)
        if k_n > k_max:
            continue
        sigma_n = float(np.interp(k_n, k_vals, sigma))
        if y_range is not None:
            y_lo, y_hi = y_range
            if not (y_lo <= sigma_n <= y_hi):
                continue
        points.append(ModePoint(n=n_mode, k=k_n, sigma=sigma_n))
    if not points:
        return [], None
    growing = [p for p in points if p.sigma >= 0.0]
    n_peak = max(growing, key=lambda p: p.sigma).n if growing else None
    return points, n_peak


def plot_dispersion_curve(
    ax,
    k_vals: np.ndarray,
    sigma: np.ndarray,
    *,
    color: str,
    lw: float,
    linestyle: str = "-",
    zorder: int = 3,
) -> None:
    ax.plot(k_vals, sigma, color=color, lw=lw, linestyle=linestyle, zorder=zorder)


def plot_mode_markers(
    ax,
    points: Iterable[ModePoint],
    n_peak: Optional[int],
    *,
    color: str,
    ms_normal: float,
    ms_peak: float,
    edgewidth: float = 0.8,
    zorder: int = 4,
) -> None:
    neg_k, neg_y, neg_s = [], [], []
    pos_k, pos_y, pos_s = [], [], []
    for p in points:
        highlight = n_peak is not None and p.n == n_peak and p.sigma >= 0.0
        ms_val = ms_peak if highlight else ms_normal
        if p.sigma < 0.0:
            neg_k.append(p.k)
            neg_y.append(p.sigma)
            neg_s.append(ms_val)
        else:
            pos_k.append(p.k)
            pos_y.append(p.sigma)
            pos_s.append(ms_val)

    if neg_k:
        ax.scatter(
            neg_k,
            neg_y,
            s=[m * m for m in neg_s],
            facecolors=color,
            edgecolors=color,
            linewidths=edgewidth,
            marker="o",
            zorder=zorder,
            clip_on=False,
        )
    if pos_k:
        ax.scatter(
            pos_k,
            pos_y,
            s=[m * m for m in pos_s],
            facecolors="none",
            edgecolors=color,
            linewidths=edgewidth,
            marker="o",
            zorder=zorder,
            clip_on=False,
        )


def set_mode_marker_scatter_data(
    scatter_neg,
    scatter_pos,
    points: Iterable[ModePoint],
    n_peak: Optional[int],
    *,
    ms_normal: float,
    ms_peak: float,
) -> None:
    neg_offsets, neg_sizes = [], []
    pos_offsets, pos_sizes = [], []
    for p in points:
        highlight = n_peak is not None and p.n == n_peak and p.sigma >= 0.0
        ms_val = ms_peak if highlight else ms_normal
        if p.sigma < 0.0:
            neg_offsets.append([p.k, p.sigma])
            neg_sizes.append(ms_val * ms_val)
        else:
            pos_offsets.append([p.k, p.sigma])
            pos_sizes.append(ms_val * ms_val)

    if neg_offsets:
        scatter_neg.set_offsets(np.asarray(neg_offsets))
        scatter_neg.set_sizes(np.asarray(neg_sizes))
        scatter_neg.set_visible(True)
    else:
        scatter_neg.set_offsets(np.empty((0, 2)))
        scatter_neg.set_sizes(np.empty(0))
        scatter_neg.set_visible(False)

    if pos_offsets:
        scatter_pos.set_offsets(np.asarray(pos_offsets))
        scatter_pos.set_sizes(np.asarray(pos_sizes))
        scatter_pos.set_visible(True)
    else:
        scatter_pos.set_offsets(np.empty((0, 2)))
        scatter_pos.set_sizes(np.empty(0))
        scatter_pos.set_visible(False)


def dispersion_ylim_tight(sigma_flat, lo_pct=1.5, hi_pct=98.5, pad_frac=0.10):
    a = np.asarray(sigma_flat, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    lo = float(np.percentile(a, lo_pct))
    hi = float(np.percentile(a, hi_pct))
    span = max(hi - lo, 1e-30)
    pad = pad_frac * span
    return lo - pad, hi + pad


def sigma_max_series(t_saved, P_saved, k_vals, EI_val, k_s_val, mu):
    out = np.empty(len(t_saved), dtype=float)
    for i, P_i in enumerate(P_saved):
        sig = compute_dispersion(k_vals, float(P_i), EI_val, k_s_val, mu)
        out[i] = float(np.nanmax(sig))
    return out


def find_dispersion_snapshot_indices(
    t_saved,
    P_saved,
    h_saved,
    k_vals,
    EI_val,
    k_s_val,
    L0_val,
    P_cr_theory,
    *,
    age_offset: float | None = None,
    meta=None,
):
    t = as_simulation_time(
        np.asarray(t_saved, dtype=float),
        age_offset=age_offset,
        meta=meta,
    )
    P = np.asarray(P_saved, dtype=float)
    n = t.size
    if n == 0:
        raise ValueError("empty t_saved")

    idx_pre = 0

    H_series = compute_H_series(h_saved)
    cr_sim = estimate_t_cr_sim_from_H_minimum(t, H_series)
    if cr_sim["ok"]:
        idx_cr_sim = int(cr_sim["idx_min"])
    else:
        idx_cr_sim = 0

    idx_pmax = int(np.argmax(P))
    idx_transient = int(np.argmin(np.abs(t - _SNAPSHOT_TRANSIENT_T)))

    idx_relaxed = n - 1
    if np.isfinite(P_cr_theory) and P_cr_theory > 0:
        idx_peak = idx_pmax
        t_peak = float(t[idx_peak])
        t_end = float(t[-1])
        margin = _POST_PEAK_TIME_FRAC * max(t_end - t_peak, 1e-6)
        rel_p = np.abs(P - P_cr_theory) / float(P_cr_theory)
        candidates = [
            i
            for i in range(n)
            if i > idx_peak
            and (t[i] - t_peak) >= margin
            and rel_p[i] <= _P_NEAR_P_CR_REL_TOL
        ]
        if candidates:
            idx_relaxed = int(candidates[-1])
        else:
            late = np.arange(idx_peak + 1, n) if idx_peak + 1 < n else np.array([n - 1])
            if late.size > 0:
                late_ok = late[t[late] - t_peak >= margin]
                pool = late_ok if late_ok.size > 0 else late
                idx_relaxed = int(pool[np.argmin(rel_p[pool])])

    return [
        (idx_pre, _SNAPSHOT_PANEL_TITLES[0], r"$t=0$"),
        (idx_cr_sim, _SNAPSHOT_PANEL_TITLES[1], rf"$t={t[idx_cr_sim]:.1f}$"),
        (idx_transient, _SNAPSHOT_PANEL_TITLES[2], rf"$t={t[idx_transient]:.1f}$"),
        (idx_relaxed, _SNAPSHOT_PANEL_TITLES[3], rf"$t={t[idx_relaxed]:.1f}$"),
    ]


def _snapshot_time_title(t: float) -> str:
    t_f = float(t)
    if abs(t_f) < 1e-9:
        return r"$t=0$ yr"
    return rf"$t={t_f:.1f}$ yr"


def find_dispersion_three_stage_snapshot_indices(
    t_saved,
    h_saved,
    *,
    age_offset: float | None = None,
    meta=None,
):
    t = as_simulation_time(
        np.asarray(t_saved, dtype=float),
        age_offset=age_offset,
        meta=meta,
    )
    n = t.size
    if n == 0:
        raise ValueError("empty t_saved")

    H_series = compute_H_series(h_saved)
    rise_sim = estimate_t_rise_sim_from_H_rise(
        t_saved,
        H_series,
        _DEFAULT_H_RISE,
    )
    if not rise_sim["ok"]:
        raise RuntimeError(
            f"t_rise_simをH(t)>{_DEFAULT_H_RISE:g} mm の初回 crossing から推定できなかった。"
        )
    idx_rise = int(rise_sim["idx_cross"])

    idx_pre = 0
    idx_end = n - 1

    return [
        (idx_pre, _THREE_STAGE_PANEL_TITLES[0], _snapshot_time_title(0.0)),
        (idx_rise, _THREE_STAGE_PANEL_TITLES[1], _snapshot_time_title(t[idx_rise])),
        (idx_end, _THREE_STAGE_PANEL_TITLES[2], _snapshot_time_title(t[idx_end])),
    ]


def _style_snapshot_panel_like_fig13(
    ax,
    xlim: tuple[float, float],
    ylim=None,
    *,
    show_ylabel: bool = True,
    xlabel: str = r"Wave number, $k$",
    ylabel: str = r"Growth rate, $\sigma$",
):
    ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xticks(
        np.arange(
            xlim[0],
            xlim[1] + 0.5 * _SNAPSHOT_XTICK_STEP,
            _SNAPSHOT_XTICK_STEP,
        )
    )
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.35, zorder=1)
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(
            ylabel,
            labelpad=_LEFT_YLABEL_LABELPAD,
        )
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(True, alpha=0.3)             
    ax.set_box_aspect(1)


def _add_snapshot_mode_xlabels(
    ax,
    L0_val: float,
    k_max_disp: float,
    k_vals: np.ndarray,
    sigma: np.ndarray,
) -> None:
    y_on_axis = float(ax.get_ylim()[0])
    y_lo, y_hi = ax.get_ylim()
    k_max = float(k_vals[-1]) if k_vals.size else 0.0
    for n_mode in _SNAPSHOT_MODE_NS:
        k_n = 2.0 * np.pi * n_mode / float(L0_val)
        if k_n > k_max or not (0.0 <= k_n <= k_max_disp):
            continue
        sigma_n = float(np.interp(k_n, k_vals, sigma))
        if not (y_lo <= sigma_n <= y_hi):
            continue
        ax.text(
            k_n+0.01,                                     
            y_on_axis+0.50,                               
            rf"$n={n_mode}$",
            ha="center",
            va="bottom",
            rotation=_SNAPSHOT_MODE_LABEL_ROTATION,
            rotation_mode="anchor",
            fontsize=text_fontsize,
            color=gray_color,
            clip_on=False,
            zorder=5,
        )


def _plot_snapshot_curve(
    ax, k_vals: np.ndarray, sigma: np.ndarray, L0_val: float, *, y_range
) -> None:
    plot_dispersion_curve(
        ax,
        k_vals,
        sigma,
        color=lightgray_color,
        lw=_SNAPSHOT_LINE_LW,
        linestyle="--",
        zorder=3,
    )
    mode_pts, n_peak = collect_mode_points(
        k_vals,
        sigma,
        L0_val,
        mode_ns=_SNAPSHOT_MODE_NS,
        y_range=y_range,
    )
    plot_mode_markers(
        ax,
        mode_pts,
        n_peak,
        color="black",
        ms_normal=_MS_MODE_NORMAL,
        ms_peak=_MS_MODE_PEAK,
        edgewidth=0.8,
        zorder=4,
    )

def _prepare_dispersion_snapshot_data(
    t_saved,
    P_saved,
    h_saved,
    EI_val,
    k_s_val,
    L0_val,
    P_cr_theory,
    snapshot_indices_fn,
    *,
    age_offset: float | None = None,
    meta=None,
    mu: float,
):
    if age_offset is None and meta is not None:
        age_offset = age_offset_from_meta(meta)
    t_sim = as_simulation_time(
        np.asarray(t_saved, dtype=float),
        age_offset=age_offset,
        meta=meta,
    )
    k_vals, k_max_grid = get_dispersion_k_grid(EI_val, k_s_val, L0_val)
    xlim_snap = _SNAPSHOT_XLIM
    if xlim_snap[1] > k_max_grid:
        k_vals = np.linspace(0.0, xlim_snap[1], k_vals.size)
    snapshots = snapshot_indices_fn(
        t_saved,
        P_saved,
        h_saved,
        k_vals,
        EI_val,
        k_s_val,
        L0_val,
        P_cr_theory,
        age_offset=age_offset,
        meta=meta,
    )
    sigma_all = [
        compute_dispersion(k_vals, float(P_saved[idx]), EI_val, k_s_val, mu)
        for idx, _, _ in snapshots
    ]
    return t_sim, k_vals, xlim_snap, snapshots, sigma_all


def _render_dispersion_snapshot_panels(
    snapshots,
    sigma_all,
    *,
    t_sim,
    k_vals,
    L0_val,
    xlim_snap,
    ylim_snap,
    n_panels: int,
    log_heading: str,
    P_cr_theory: float,
    panel_labels: Sequence[str] | None = None,
    match_three_snapshot_equal_boxes: bool = False,
    xlabel: str = r"Wave number, $k$",
    ylabel: str = r"Growth rate, $\sigma$",
) -> plt.Figure:
    if match_three_snapshot_equal_boxes:
        from simulation.Figure_h_evolution_and_spacetime import (
            _THREE_SNAPSHOT_EQUAL_FIGSIZE,
            align_axes_to_three_snapshot_equal_boxes,
            apply_shared_horizontal_left_ylabel,
            apply_three_snapshot_equal_layout_pads,
        )

        figsize = _THREE_SNAPSHOT_EQUAL_FIGSIZE
    else:
        figsize = _FIG_FIGSIZE

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1] * n_panels},
    )
    if match_three_snapshot_equal_boxes:
        apply_three_snapshot_equal_layout_pads(fig)
    else:
        apply_constrained_layout_pads(fig)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    print(log_heading)
    if np.isfinite(P_cr_theory):
        print(f"  P_cr (theory) = {P_cr_theory:.6f}")
    for i_ax, (ax, (idx, phase_name, t_subtitle), sig) in enumerate(
        zip(axes, snapshots, sigma_all)
    ):
        t_i = float(t_sim[idx])
        _style_snapshot_panel_like_fig13(
            ax,
            xlim_snap,
            ylim=ylim_snap,
            show_ylabel=(i_ax == 0),
            xlabel=xlabel,
            ylabel=ylabel,
        )
        _plot_snapshot_curve(
            ax, k_vals, sig, L0_val, y_range=ylim_snap
        )
        _add_snapshot_mode_xlabels(ax, L0_val, xlim_snap[1], k_vals, sig)
        ax.set_title(
            t_subtitle,
            fontsize=text_fontsize,
            pad=6,
        )
        print(
            f"  [{phase_name}] index={idx}, t={t_i:.1f}"
        )

    if match_three_snapshot_equal_boxes:
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

    if panel_labels is not None:
        add_panel_label(axes[0], panel_labels[0])
        for ax, label in zip(axes[1:], panel_labels[1:]):
            _add_panel_label_above_yaxis(ax, label)
    return fig


def make_dispersion_snapshots_figure(
    t_saved,
    P_saved,
    h_saved,
    EI_val,
    k_s_val,
    L0_val,
    P_cr_theory,
    out_pdf_path,
    *,
    mu: float,
    age_offset: float | None = None,
    meta=None,
):
    t_sim, k_vals, xlim_snap, snapshots, sigma_all = _prepare_dispersion_snapshot_data(
        t_saved,
        P_saved,
        h_saved,
        EI_val,
        k_s_val,
        L0_val,
        P_cr_theory,
        find_dispersion_snapshot_indices,
        age_offset=age_offset,
        meta=meta,
        mu=mu,
    )
    fig = _render_dispersion_snapshot_panels(
        snapshots,
        sigma_all,
        t_sim=t_sim,
        k_vals=k_vals,
        L0_val=L0_val,
        xlim_snap=xlim_snap,
        ylim_snap=_SNAPSHOT_YLIM,
        n_panels=4,
        log_heading="\n--- Dispersion snapshots ---",
        P_cr_theory=P_cr_theory,
        panel_labels=("e", "d", "f", "g"),
    )
    save_figure_and_show(fig, out_pdf_path)
    return snapshots


def make_dispersion_snapshots_three_stages_figure(
    t_saved,
    P_saved,
    h_saved,
    EI_val,
    k_s_val,
    L0_val,
    P_cr_theory,
    out_pdf_path,
    *,
    mu: float,
    age_offset: float | None = None,
    meta=None,
):
    if age_offset is None and meta is not None:
        age_offset = age_offset_from_meta(meta)

    t_sim = as_simulation_time(
        np.asarray(t_saved, dtype=float),
        age_offset=age_offset,
        meta=meta,
    )
    k_vals, k_max_grid = get_dispersion_k_grid(EI_val, k_s_val, L0_val)
    xlim_snap = _SNAPSHOT_XLIM
    if xlim_snap[1] > k_max_grid:
        k_vals = np.linspace(0.0, xlim_snap[1], k_vals.size)
    snapshots = find_dispersion_three_stage_snapshot_indices(
        t_saved,
        h_saved,
        age_offset=age_offset,
        meta=meta,
    )
    sigma_all = [
        compute_dispersion(k_vals, float(P_saved[idx]), EI_val, k_s_val, mu)
        for idx, _, _ in snapshots
    ]

    fig = _render_dispersion_snapshot_panels(
        snapshots,
        sigma_all,
        t_sim=t_sim,
        k_vals=k_vals,
        L0_val=L0_val,
        xlim_snap=xlim_snap,
        ylim_snap=_SNAPSHOT_YLIM,
        n_panels=3,
        log_heading="\n--- Dispersion snapshots (three stages) ---",
        P_cr_theory=P_cr_theory,
        panel_labels=None,
        match_three_snapshot_equal_boxes=True,
        xlabel=r"Wave number, $k$ (mm$^{-1}$)",
        ylabel=r"Growth rate, $\sigma$ (yr$^{-1}$)",
    )
    from simulation.Figure_h_evolution_and_spacetime import (
        apply_five_figure_label_sizes,
    )

    apply_five_figure_label_sizes(fig)
    save_figure_and_show(fig, out_pdf_path)
    return snapshots


def main():
    print("Loading simulation results...")
    _x, t_saved, h_saved, P_saved, meta = load_simulation_run()

    P_cr = 2.0 * (k_s * EI) ** 0.5

    fig_dir = get_figures_dir()
    print(f"図の保存先: {fig_dir}")

    mu = mu_from_meta(meta)

    snapshots_pdf = figure_path("Figure_dispersion_snapshots.pdf")
    make_dispersion_snapshots_figure(
        t_saved,
        P_saved,
        h_saved,
        EI,
        k_s,
        L0,
        P_cr,
        snapshots_pdf,
        mu=mu,
        meta=meta,
    )

    three_stages_pdf = figure_path("Figure_dispersion_snapshots_three_stages.pdf")
    make_dispersion_snapshots_three_stages_figure(
        t_saved,
        P_saved,
        h_saved,
        EI,
        k_s,
        L0,
        P_cr,
        three_stages_pdf,
        mu=mu,
        meta=meta,
    )


if __name__ == "__main__":
    main()

