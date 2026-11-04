#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: grid search g mu P coupling."""

from __future__ import annotations

import math
import os
import sys

                                                      
os.environ.setdefault("MPLBACKEND", "Agg")
from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simulation.buckling_simulation_P_coupling import (  # noqa: E402
    P_cr_theory_linear,
    compute_H_series,
    estimate_t_cr_sim_from_H_minimum,
    estimate_t_rise_sim_from_H_rise,
    excess_length_ratio_rho,
    excess_length_ratio_rho_approx,
    relative_rho_discrepancy,
    run_buckling_simulation_with_P_coupling,
    t_cr_theory_linear,
    _DEFAULT_ES,
    _DEFAULT_EI,
    _DEFAULT_K_S,
    _DEFAULT_L0,
    _DEFAULT_MU,
    _DEFAULT_N,
    _DEFAULT_NATURAL_ELONGATION_RATE,
    _DEFAULT_NOISE_STD,
    _DEFAULT_SAVE_EVERY,
    _DEFAULT_SEED,
    _DEFAULT_T_MAX,
    _DEFAULT_T_MIN,
    _DEFAULT_DT,
)
from simulation.Figure_modal_amplitude_evolution import (  # noqa: E402
    build_mode_amplitude_series,
    make_modal_amplitude_figure,
)
from simulation.simulation_output import result_subdir  # noqa: E402


MODAL_AMPLITUDE_FLOOR = 1e-12

                                                 
REFERENCE_G = float(_DEFAULT_NATURAL_ELONGATION_RATE)  # 0.35 mm/year
REFERENCE_MU = float(_DEFAULT_MU)  # 0.005 N year/mm^2

G_CANDIDATES = sorted(
    {
        0.1,
        0.2,
        0.3,
        REFERENCE_G,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    }
)
MU_CANDIDATES = [0.001, 0.005, 0.010, 0.020, 0.025, 0.030]

T_RISE_THRESHOLD_MM = 0.5

GRID_SEARCH_RESULTS_SUBDIR = "grid_search"

                                                                            
MODE_N_MAX: int | None = 6


def split_modes_by_amplitude_floor(
    h_saved,
    N: int,
    modes: tuple[int, ...],
    *,
    floor: float = MODAL_AMPLITUDE_FLOOR,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    amps_by_mode = build_mode_amplitude_series(h_saved, N, modes)
    valid: list[int] = []
    excluded: list[int] = []
    f = float(floor)
    for n in modes:
        if float(np.min(amps_by_mode[int(n)])) < f:
            excluded.append(int(n))
        else:
            valid.append(int(n))
    return tuple(valid), tuple(excluded)


def mode_ns_rfft(N: int, *, n_max: int | None = MODE_N_MAX) -> tuple[int, ...]:
    n_hi = int(N) // 2
    if n_max is not None:
        n_hi = min(n_hi, int(n_max))
    if n_hi < 1:
        return ()
    return tuple(range(1, n_hi + 1))


def modal_amplitudes_rfft(
    h: np.ndarray, *, N: int, mode_ns: tuple[int, ...]
) -> dict[int, float]:
    a = np.fft.rfft(np.asarray(h, dtype=float) - float(np.mean(h)))
    scale = 2.0 / float(N)
    out: dict[int, float] = {}
    for n in mode_ns:
        out[int(n)] = float(np.abs(a[n]) * scale) if n < a.size else 0.0
    return out


def modal_amplitude_rfft(h: np.ndarray, n: int, N: int) -> float:
    return modal_amplitudes_rfft(h, N=N, mode_ns=(int(n),))[int(n)]


def t_cr_theory(*, ES: float, EI: float, k_s: float, natural_elongation_rate: float) -> float:
    P_cr = P_cr_theory_linear(EI, k_s)
    return t_cr_theory_linear(
        ES=float(ES),
        L0=_DEFAULT_L0,
        P_cr=P_cr,
        natural_elongation_rate=float(natural_elongation_rate),
    )


def t_cr_n_for_mode(
    t_saved: np.ndarray, h_saved: np.ndarray, *, n: int, N: int
) -> float:
    amps = np.array(
        [modal_amplitude_rfft(h, n, N) for h in h_saved], dtype=float
    )
    idx_min = int(np.argmin(amps))
    return float(np.asarray(t_saved, dtype=float)[idx_min])


def earliest_t_cr_n(
    t_saved: np.ndarray, h_saved: np.ndarray, *, N: int, mode_ns: tuple[int, ...]
) -> tuple[float, int]:
    best_t = float("inf")
    best_n = -1
    for n in mode_ns:
        t_n = t_cr_n_for_mode(t_saved, h_saved, n=n, N=N)
        if t_n < best_t or (np.isclose(t_n, best_t) and (best_n < 0 or n < best_n)):
            best_t = t_n
            best_n = int(n)
    if best_n < 0:
        return float("nan"), -1
    return best_t, best_n


def dominant_mode_at_time(
    h: np.ndarray, *, N: int, mode_ns: tuple[int, ...]
) -> int:
    amps_n = modal_amplitudes_rfft(h, N=N, mode_ns=mode_ns)
    return int(max(amps_n, key=amps_n.get))


def dominant_modes_over_time(
    h_saved: np.ndarray, *, N: int, mode_ns: tuple[int, ...]
) -> List[int]:
    return [dominant_mode_at_time(h, N=N, mode_ns=mode_ns) for h in h_saved]


def n_peak_dominant_mode_after_t_cr_n(
    t_saved: np.ndarray,
    h_saved: np.ndarray,
    *,
    t_cr_n: float,
    N: int,
    mode_ns: tuple[int, ...],
) -> int:
    if not np.isfinite(t_cr_n):
        return -1
    t_arr = np.asarray(t_saved, dtype=float)
    h_arr = np.asarray(h_saved)
    mask = t_arr >= float(t_cr_n)
    if not np.any(mask):
        return -1
    n_dom = dominant_modes_over_time(h_arr[mask], N=N, mode_ns=mode_ns)
    return int(max(n_dom)) if n_dom else -1


def mode_sequence_summary(
    *, n_cr_n: int, n_peak: int, n_final: int
) -> str:
    return f"{int(n_cr_n)}→{int(n_peak)}→{int(n_final)}"


def print_modal_amplitude_v_turn_times(
    t_saved: np.ndarray,
    h_saved: np.ndarray,
    *,
    N: int,
    mode_ns: tuple[int, ...],
    excluded_modes: tuple[int, ...] = (),
    g: float | None = None,
    mu: float | None = None,
) -> None:
    excluded_set = {int(n) for n in excluded_modes}
    if g is not None and mu is not None:
        print(f"\n[g={float(g):g}, mu={float(mu):g}]")
    print("================================================")
    print("Eigenvalues around t_cr vs amplitude turnaround times")
    print("================================================")
    for n in mode_ns:
        t_n = t_cr_n_for_mode(t_saved, h_saved, n=int(n), N=N)
        suffix = "  [excluded: min|a-hat|<floor]" if int(n) in excluded_set else ""
        print(
            f"mode n={int(n)}: time of min amplitude (turnaround) t_cr_n = {t_n:.2f}{suffix}"
        )


@dataclass(frozen=True)
class GridResult:
    g: float
    mu: float
    t_cr: float
    t_cr_n: float
    n_cr_n: int
    t_cr_sim: float
    t_rise_sim: float
    delay: float
    n_final: int
    mode_seq: str
    t_end: float
    rho_geom_t_end: float
    rho_approx_t_end: float
    relative_rho_discrepancy_t_end: float


def _fmt(x: float, *, width: int = 10) -> str:
    if x is None or not np.isfinite(x):
        return f"{'nan':>{width}s}"
    return f"{x:>{width}.4f}"


def _fmt_rho(x: float, *, width: int = 10) -> str:
    if x is None or not np.isfinite(x):
        return f"{'nan':>{width}s}"
    return f"{x:>{width}.4f}"


def _fmt_pct(x: float, *, width: int = 10) -> str:
    if x is None or not np.isfinite(x):
        return f"{'nan':>{width}s}"
    return f"{100.0 * x:>{width - 1}.2f}%"


def _fmt_mu(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def grid_search_modal_amplitude_pdf_path(*, g: float, mu: float) -> str:
    """data/simulation/grid_search/modal_amplitude_g{g}_mu{mu}.pdf"""
    name = f"modal_amplitude_g{float(g):g}_mu{_fmt_mu(mu)}.pdf"
    return os.path.join(result_subdir(GRID_SEARCH_RESULTS_SUBDIR), name)


def save_grid_search_modal_amplitude_figure(
    t_saved: np.ndarray,
    h_saved: np.ndarray,
    *,
    g: float,
    mu: float,
    N: int,
    mode_ns: tuple[int, ...],
) -> str:
    out_pdf = grid_search_modal_amplitude_pdf_path(g=g, mu=mu)
    make_modal_amplitude_figure(
        np.asarray(t_saved, dtype=float),
        h_saved,
        int(N),
        out_pdf,
        panel_label=None,
        age_offset=None,
        show=False,
        modes=mode_ns,
    )
    return out_pdf


def run_one(*, g: float, mu: float) -> GridResult:
    EI = float(_DEFAULT_EI)
    k_s = float(_DEFAULT_K_S)
    ES = float(_DEFAULT_ES)

    x, t_saved, h_saved, P_saved, _epsilon_saved = run_buckling_simulation_with_P_coupling(
        mu=float(mu),
        EI=EI,
        k_s=k_s,
        ES=ES,
        L0=float(_DEFAULT_L0),
        N=int(_DEFAULT_N),
        t_min=float(_DEFAULT_T_MIN),
        t_max=float(_DEFAULT_T_MAX),
        dt=float(_DEFAULT_DT),
        natural_elongation_rate=float(g),
        noise_std=float(_DEFAULT_NOISE_STD),
        seed=int(_DEFAULT_SEED),
        save_every=int(_DEFAULT_SAVE_EVERY),
    )

    t_saved = np.asarray(t_saved, dtype=float)
    h_stack = np.asarray(h_saved, dtype=float)
    N = int(_DEFAULT_N)
    mode_ns_all = mode_ns_rfft(N)
    mode_ns, mode_ns_excluded = split_modes_by_amplitude_floor(
        h_stack, N, mode_ns_all, floor=MODAL_AMPLITUDE_FLOOR
    )

    t_cr = t_cr_theory(
        ES=ES, EI=EI, k_s=k_s, natural_elongation_rate=float(g)
    )

                                            
    H_series = compute_H_series(h_stack)
    cr_sim = estimate_t_cr_sim_from_H_minimum(t_saved, H_series, exclude_t0=True)
    t_cr_sim = float(cr_sim["t_cr_sim"]) if cr_sim.get("ok") else float("nan")

                                 
    if mode_ns:
        t_cr_n, n_cr_n = earliest_t_cr_n(
            t_saved, h_stack, N=N, mode_ns=mode_ns
        )
    else:
        t_cr_n, n_cr_n = float("nan"), -1

    max_amp = np.array([float(np.max(np.abs(h))) for h in h_stack], dtype=float)
    rise = estimate_t_rise_sim_from_H_rise(
        t_saved, max_amp, H_rise=T_RISE_THRESHOLD_MM
    )
    t_rise_sim = float(rise["t_rise_sim"]) if rise.get("ok") else float("nan")
    delay = (
        float(t_rise_sim - t_cr_sim)
        if (np.isfinite(t_rise_sim) and np.isfinite(t_cr_sim))
        else float("nan")
    )

    if mode_ns:
        n_dom_series = dominant_modes_over_time(h_stack, N=N, mode_ns=mode_ns)
        n_peak = n_peak_dominant_mode_after_t_cr_n(
            t_saved, h_stack, t_cr_n=t_cr_n, N=N, mode_ns=mode_ns
        )
        n_final = int(n_dom_series[-1]) if n_dom_series else -1
    else:
        n_final = -1
        n_peak = -1
    if n_cr_n >= 0 and n_peak >= 0 and n_final >= 0:
        mode_seq = mode_sequence_summary(
            n_cr_n=n_cr_n, n_peak=n_peak, n_final=n_final
        )
    else:
        mode_seq = "—"

    print_modal_amplitude_v_turn_times(
        t_saved,
        h_stack,
        N=N,
        mode_ns=mode_ns_all,
        excluded_modes=mode_ns_excluded,
        g=g,
        mu=mu,
    )

    save_grid_search_modal_amplitude_figure(
        t_saved,
        h_stack,
        g=g,
        mu=mu,
        N=N,
        mode_ns=mode_ns_all,
    )

                                
    idx_end = int(np.argmin(np.abs(t_saved - float(_DEFAULT_T_MAX))))
    t_end = float(t_saved[idx_end])
    h_end = h_stack[idx_end]
    L0 = float(_DEFAULT_L0)
    rho_geom_t_end = excess_length_ratio_rho(h_end, x, L0)
    rho_approx_t_end = excess_length_ratio_rho_approx(h_end, x, L0)
    rel_disc_t_end = relative_rho_discrepancy(rho_approx_t_end, rho_geom_t_end)

    return GridResult(
        g=float(g),
        mu=float(mu),
        t_cr=t_cr,
        t_cr_n=t_cr_n,
        n_cr_n=int(n_cr_n),
        t_cr_sim=t_cr_sim,
        t_rise_sim=t_rise_sim,
        delay=delay,
        n_final=n_final,
        mode_seq=mode_seq,
        t_end=t_end,
        rho_geom_t_end=float(rho_geom_t_end),
        rho_approx_t_end=float(rho_approx_t_end),
        relative_rho_discrepancy_t_end=float(rel_disc_t_end),
    )


def _iter_grid() -> Iterable[tuple[float, float]]:
    for g in G_CANDIDATES:
        for mu in MU_CANDIDATES:
            yield float(g), float(mu)


def _print_grid_table(
    results_by_key: dict[tuple[float, float], GridResult],
    *,
    title: str,
    values: Callable[[GridResult], str],
    g_values: list[float],
    mu_values: list[float],
) -> None:
    mu_header = "  ".join(f"{_fmt_mu(mu):>10s}" for mu in mu_values)
    print(f"\n--- {title} ---")
    _corner = "g\\mu"
    print(f"{_corner:>6s}  {mu_header}")
    print("-" * (6 + 2 + len(mu_header)))
    for g in g_values:
        row = []
        for mu in mu_values:
            r = results_by_key.get((g, mu))
            row.append(values(r) if r is not None else f"{'nan':>10s}")
        print(f"{g:6.2f}  " + "  ".join(row))


def _mode_ns_label() -> str:
    ns = mode_ns_rfft(int(_DEFAULT_N))
    if not ns:
        return "n=?"
    if len(ns) == 1:
        return f"n={ns[0]}"
    return f"n={ns[0]}..{ns[-1]}"


def print_tables(results: list[GridResult]) -> None:
    g_values = [float(g) for g in G_CANDIDATES]
    mu_values = [float(mu) for mu in MU_CANDIDATES]
    results_by_key = {(float(r.g), float(r.mu)): r for r in results}
    _modes = _mode_ns_label()

    _print_grid_table(
        results_by_key,
        title="t_cr (year)  [theory: P_cr/((ES/L)*g)]",
        values=lambda r: _fmt(r.t_cr),
        g_values=g_values,
        mu_values=mu_values,
    )
    _print_grid_table(
        results_by_key,
        title=f"t_cr_n (year)  [earliest modal-amplitude V-turn, {_modes}]",
        values=lambda r: _fmt(r.t_cr_n),
        g_values=g_values,
        mu_values=mu_values,
    )
    _print_grid_table(
        results_by_key,
        title="n_cr_n  [mode at t_cr_n]",
        values=lambda r: f"{int(r.n_cr_n):>10d}" if r.n_cr_n >= 0 else f"{'—':>10s}",
        g_values=g_values,
        mu_values=mu_values,
    )
    _print_grid_table(
        results_by_key,
        title="t_cr_sim (year)  [H_min time]",
        values=lambda r: _fmt(r.t_cr_sim),
        g_values=g_values,
        mu_values=mu_values,
    )
    _print_grid_table(
        results_by_key,
        title=f"t_rise_sim (year)  [max|h| > H_rise={T_RISE_THRESHOLD_MM:g} mm]",
        values=lambda r: _fmt(r.t_rise_sim),
        g_values=g_values,
        mu_values=mu_values,
    )
    _print_grid_table(
        results_by_key,
        title=(
            f"mode sequence  [n_cr_n→n_peak→n_final; "
            f"n_peak = max argmax {_modes} for t>=t_cr_n]"
        ),
        values=lambda r: f"{r.mode_seq:>11s}",
        g_values=g_values,
        mu_values=mu_values,
    )
    _print_grid_table(
        results_by_key,
        title="delay (year)  [t_rise_sim - t_cr_sim]",
        values=lambda r: _fmt(r.delay),
        g_values=g_values,
        mu_values=mu_values,
    )
    t_end_label = float(_DEFAULT_T_MAX)
    _print_grid_table(
        results_by_key,
        title=(
            f"rho_geom(t≈{t_end_label:g} year)  "
            r"[(1/L0)∫(√(1+h_x²)-1) dx]"
        ),
        values=lambda r: _fmt_rho(r.rho_geom_t_end),
        g_values=g_values,
        mu_values=mu_values,
    )
    _print_grid_table(
        results_by_key,
        title=(
            f"rho_approx(t≈{t_end_label:g} year)  "
            r"[(1/(2 L0))∫ h_x² dx]"
        ),
        values=lambda r: _fmt_rho(r.rho_approx_t_end),
        g_values=g_values,
        mu_values=mu_values,
    )
    _print_grid_table(
        results_by_key,
        title=(
            f"relative discrepancy(t≈{t_end_label:g} year)  "
            r"[|ρ_approx-ρ_geom|/ρ_geom]  (%)"
        ),
        values=lambda r: _fmt_pct(r.relative_rho_discrepancy_t_end),
        g_values=g_values,
        mu_values=mu_values,
    )


def main() -> None:
    print("Grid search: P-coupling buckling (g, mu)")
    print(
        f"reference simulation: g={REFERENCE_G:g} mm/year, "
        f"mu={REFERENCE_MU:g} N year/mm^2"
    )
    print(f"g candidates : {G_CANDIDATES}")
    print(f"mu candidates: {MU_CANDIDATES}")
    print(f"H_rise (t_rise_sim threshold): {T_RISE_THRESHOLD_MM} mm (max|h|)")
    print(f"modal modes (candidates): {_mode_ns_label()}")
    print(f"amplitude floor for mode evaluation: {MODAL_AMPLITUDE_FLOOR:g}")
    print(
        f"modal amplitude PDFs: "
        f"{result_subdir(GRID_SEARCH_RESULTS_SUBDIR)}/modal_amplitude_g<g>_mu<mu>.pdf"
    )
    print("")

    results: list[GridResult] = []
    for g, mu in _iter_grid():
        tag = "  [REFERENCE]" if (
            math.isclose(g, REFERENCE_G) and math.isclose(mu, REFERENCE_MU)
        ) else ""
        print(f"Running: g={g:g}, mu={mu:g}{tag} ...", flush=True)
        results.append(run_one(g=g, mu=mu))

    print("\n=== RESULTS ===")
    print_tables(results)


if __name__ == "__main__":
    main()

