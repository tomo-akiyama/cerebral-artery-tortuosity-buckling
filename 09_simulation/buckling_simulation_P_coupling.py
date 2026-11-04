#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: buckling simulation P coupling."""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import figures.plot_config  # noqa: F401 — sns theme + rcParams
from figures.plot_config import (
    blue_color,
    lightgray_color,
    orange_color,
    gray_color,
    red_color,
    figure_show_seconds,
    ms,
    colors,
    lightblue_color,
    annotation_fontsize,
    show_figure,
    text_fontsize,
)

           
_DEFAULT_MU = 0.005                                                   
_DEFAULT_NATURAL_ELONGATION_RATE = 0.35  # mm/year
_DEFAULT_H_RISE = 0.5 # mm


                                                                        
_DEFAULT_R_O = 1.8   # mm
_DEFAULT_R_I = 1.6   # mm
_DEFAULT_E = 2.0                                                          
_DEFAULT_K_S = 0.001 # N/mm^2
_DEFAULT_L0 = 100    # mm

_DEFAULT_I = (np.pi / 4) * (_DEFAULT_R_O**4 - _DEFAULT_R_I**4)  # mm^4
_DEFAULT_EI = _DEFAULT_E * _DEFAULT_I
_DEFAULT_S = np.pi * (_DEFAULT_R_O**2 - _DEFAULT_R_I**2)                   
_DEFAULT_ES = _DEFAULT_E * _DEFAULT_S

_DEFAULT_N = 1000               
_DEFAULT_AGE_OFFSET = 20                                         
_DEFAULT_T_MIN = 0                                
_DEFAULT_T_MAX = 60                                            
_DEFAULT_DT = 0.01                      
_DEFAULT_NOISE_STD = 0.01                                                
_DEFAULT_SEED = 1                
_DEFAULT_SAVE_INTERVAL = 0.1              
_DEFAULT_SAVE_EVERY = int(_DEFAULT_SAVE_INTERVAL / _DEFAULT_DT)                    

                                
                                     
_DEFAULT_PLOT_TIME_INTERVAL = 10.0

                
_DEFAULT_MAKE_GIF = False

         
                                       
                                                     
                                                                           
_DEFAULT_INITIAL_CONDITION_TYPE = "fourier_superposition"

                                    
                                                              
_DEFAULT_INITIAL_MODE_LOWER_BUFFER = 2
_DEFAULT_INITIAL_MODE_UPPER_BUFFER = 4
_DEFAULT_INITIAL_MODE_SEARCH_MAX = 50

                                                                    
_DEFAULT_INITIAL_DELTA_L_ARC_OVER_L0 = 1.0e-4  # 0.01 %


def _times_to_saved_indices(t_saved, times_to_plot):
    indices = []
    seen = set()
    for t in times_to_plot:
        i = int(np.argmin(np.abs(t_saved - t)))
        if i not in seen:
            seen.add(i)
            indices.append(i)
    return indices


def _sigma_ylim_tight(y_flat, lo_pct=1.5, hi_pct=98.5, pad_frac=0.06):
    a = np.asarray(y_flat, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    lo = float(np.percentile(a, lo_pct))
    hi = float(np.percentile(a, hi_pct))
    span = max(hi - lo, 1e-30)
    pad = pad_frac * span
    return lo - pad, hi + pad


def dispersion_D(k_vals, P_val, EI_val, k_s_val) -> np.ndarray:
    k = np.asarray(k_vals, dtype=float)
    return float(P_val) * k**2 - float(EI_val) * k**4 - float(k_s_val)


def D_at_mode(P_val, n_mode, L0_val, EI_val, k_s_val) -> float:
    k_n = 2.0 * np.pi * float(n_mode) / float(L0_val)
    return float(dispersion_D(k_n, P_val, EI_val, k_s_val))


def compute_sigma(k_vals, P_val, EI_val, k_s_val, mu_val) -> np.ndarray:
    mu_f = float(mu_val)
    if mu_f <= 0:
        return np.full(np.asarray(k_vals).shape, np.nan, dtype=float)
    return dispersion_D(k_vals, P_val, EI_val, k_s_val) / mu_f

def discrete_critical_mode_info(
    EI_val: float,
    k_s_val: float,
    L0_val: float,
    *,
    N_grid: int | None = None,
    n_search_max: int = 50,
) -> dict:
    out = {
        "ok": False,
        "n_star": -1,
        "k_star": np.nan,
        "lambda_star": np.nan,
        "P_cr_discrete": np.nan,
    }

    EI_f = float(EI_val)
    k_s_f = float(k_s_val)
    L0_f = float(L0_val)

    if EI_f <= 0.0 or k_s_f <= 0.0 or L0_f <= 0.0:
        return out

    n_cap = int(n_search_max)

    if N_grid is not None:
                                                              
        n_cap = min(n_cap, max(1, int(N_grid) // 2))

    if n_cap < 1:
        return out

    ns = np.arange(1, n_cap + 1, dtype=int)
    k_ns = 2.0 * np.pi * ns / L0_f

    P_cr_ns = EI_f * k_ns**2 + k_s_f / k_ns**2
    j = int(np.nanargmin(P_cr_ns))

    n_star = int(ns[j])
    k_star = float(k_ns[j])

    out["ok"] = True
    out["n_star"] = n_star
    out["k_star"] = k_star
    out["lambda_star"] = float(L0_f / n_star)
    out["P_cr_discrete"] = float(P_cr_ns[j])

    return out


def select_initial_modes_around_critical_mode(
    EI_val: float,
    k_s_val: float,
    L0_val: float,
    *,
    N_grid: int | None = None,
    n_search_max: int = 50,
    lower_buffer: int = 2,
    upper_buffer: int = 4,
    fallback_n_max: int = 6,
) -> tuple[int, ...]:
    info = discrete_critical_mode_info(
        EI_val,
        k_s_val,
        L0_val,
        N_grid=N_grid,
        n_search_max=n_search_max,
    )

    if not info["ok"]:
        n_cap = int(fallback_n_max)
        if N_grid is not None:
            n_cap = min(n_cap, max(1, int(N_grid) // 2))
        return tuple(range(1, n_cap + 1))

    n_star = int(info["n_star"])

    n_min = max(1, n_star - int(lower_buffer))
    n_max = n_star + int(upper_buffer)

    if N_grid is not None:
        n_max = min(n_max, max(1, int(N_grid) // 2))

    return tuple(range(n_min, n_max + 1))


def max_fourier_mode_from_geometry(L0: float, r_o: float) -> int:
    L0_f = float(L0)
    r_o_f = float(r_o)
    if L0_f <= 0.0 or r_o_f <= 0.0:
        return 1
    return max(1, int(np.floor(L0_f / (2.0 * r_o_f))))


def select_fourier_modes_geometric(
    L0: float,
    r_o: float,
    *,
    N_grid: int | None = None,
) -> tuple[int, ...]:
    n_geom = max_fourier_mode_from_geometry(L0, r_o)
    n_max = n_geom
    if N_grid is not None:
        n_max = min(n_max, max(1, int(N_grid) // 2))
    return tuple(range(1, n_max + 1))


def make_equal_low_mode_initial_condition(
    x: np.ndarray,
    L0: float,
    rng: np.random.Generator,
    H0: float,
    modes: tuple[int, ...],
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    L0_f = float(L0)
    H0_f = float(H0)

    modes_clean = tuple(sorted({int(n) for n in modes if int(n) > 0}))

    h = np.zeros_like(x_arr, dtype=float)

    if H0_f <= 0.0 or len(modes_clean) == 0:
        return h

    for n in modes_clean:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        h += np.cos(2.0 * np.pi * n * x_arr / L0_f + phase)

    h -= np.mean(h)

    H = float(np.max(np.abs(h)))
    if H > 0.0:
        h *= H0_f / H

    return h


def make_fourier_superposition_initial_condition(
    x: np.ndarray,
    L0: float,
    rng: np.random.Generator,
    modes: tuple[int, ...],
    *,
    delta_L_arc_over_L0: float = _DEFAULT_INITIAL_DELTA_L_ARC_OVER_L0,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    L0_f = float(L0)
    ratio = float(delta_L_arc_over_L0)
    modes_clean = tuple(sorted({int(n) for n in modes if int(n) > 0}))

    h = np.zeros_like(x_arr, dtype=float)
    if ratio <= 0.0 or len(modes_clean) == 0:
        return h

    for n in modes_clean:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        h += np.cos(2.0 * np.pi * n * x_arr / L0_f + phase)

    h -= np.mean(h)
    dL_arc = compute_shortening_integral(h, x_arr)
    if dL_arc <= 0.0:
        raise ValueError(
            f"Fourier superposition initial condition has non-positive ΔL_arc = {dL_arc:g}; "
            "cannot rescale by delta_L_arc_over_L0."
        )

    target_dL_arc = ratio * L0_f
    h *= np.sqrt(target_dL_arc / dL_arc)
    return h


def make_initial_condition(
    x: np.ndarray,
    L0: float,
    rng: np.random.Generator,
    *,
    initial_condition_type: str,
    noise_std: float,
    EI: float,
    k_s: float,
    N: int,
    r_o: float = _DEFAULT_R_O,
    initial_modes: tuple[int, ...] | None = None,
    initial_mode_search_max: int = 50,
    initial_mode_lower_buffer: int = 2,
    initial_mode_upper_buffer: int = 4,
    initial_delta_L_arc_over_L0: float = _DEFAULT_INITIAL_DELTA_L_ARC_OVER_L0,
) -> tuple[np.ndarray, tuple[int, ...] | None]:
    ic_type = str(initial_condition_type).lower().replace("-", "_")

    if ic_type == "white_noise":
        h = rng.normal(loc=0.0, scale=float(noise_std), size=int(N))

                                      
                                             
        h -= np.mean(h)

        return h, None

    if ic_type == "low_mode":
        if initial_modes is None:
            modes_used = select_initial_modes_around_critical_mode(
                EI,
                k_s,
                L0,
                N_grid=N,
                n_search_max=initial_mode_search_max,
                lower_buffer=initial_mode_lower_buffer,
                upper_buffer=initial_mode_upper_buffer,
            )
        else:
            modes_used = tuple(sorted({int(n) for n in initial_modes if int(n) > 0}))

        h = make_equal_low_mode_initial_condition(
            x=x,
            L0=L0,
            rng=rng,
            H0=float(noise_std),
            modes=modes_used,
        )

        return h, modes_used

    if ic_type == "fourier_superposition":
        modes_used = (
            tuple(sorted({int(n) for n in initial_modes if int(n) > 0}))
            if initial_modes is not None
            else select_fourier_modes_geometric(L0, r_o, N_grid=N)
        )
        h = make_fourier_superposition_initial_condition(
            x=x,
            L0=L0,
            rng=rng,
            modes=modes_used,
            delta_L_arc_over_L0=float(initial_delta_L_arc_over_L0),
        )
        return h, modes_used

    raise ValueError(
        "initial_condition_type must be one of "
        "'white_noise', 'low_mode', or 'fourier_superposition'. "
        f"Got {initial_condition_type!r}."
    )


def estimate_t_cr_sim_from_H_minimum(
    t_saved,
    H_series,
    *,
    exclude_t0: bool = True,
) -> dict:
    t = np.asarray(t_saved, dtype=float)
    H = np.asarray(H_series, dtype=float)
    out = {
        "ok": False,
        "t_cr_sim": np.nan,
        "H_min": np.nan,
        "idx_min": -1,
    }
    if t.size == 0 or H.size == 0 or t.size != H.size:
        return out
    mask = np.ones(t.shape, dtype=bool)
    if exclude_t0 and t.size > 1:
        mask = t > 0.0
        if not np.any(mask):
            mask = np.ones(t.shape, dtype=bool)
    idx_rel = int(np.argmin(H[mask]))
    idx = int(np.flatnonzero(mask)[idx_rel])
    out["ok"] = True
    out["t_cr_sim"] = float(t[idx])
    out["H_min"] = float(H[idx])
    out["idx_min"] = idx
    return out


def estimate_t_slow_sim_from_H_growth_rate(
    t_saved,
    H_series,
    *,
    idx_cr_sim: int,
    smoothing_window: int = 31,
    polyorder: int = 3,
) -> dict:
    t = np.asarray(t_saved, dtype=float)
    H = np.asarray(H_series, dtype=float)
    out = {
        "ok": False,
        "t_slow_sim": np.nan,
        "idx_slow": -1,
        "relative_growth_rate": np.nan,
        "relative_growth_acceleration": np.nan,
        "smoothing_window": -1,
    }
    if t.size < 5 or H.size != t.size or not np.all(np.diff(t) > 0):
        return out

    window = min(int(smoothing_window), int(t.size))
    if window % 2 == 0:
        window -= 1
    min_window = int(polyorder) + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window:
        return out

    dt = float(np.median(np.diff(t)))
    log_H = np.log(np.maximum(H, np.finfo(float).tiny))
    relative_growth_rate = savgol_filter(
        log_H,
        window_length=window,
        polyorder=int(polyorder),
        deriv=1,
        delta=dt,
        mode="interp",
    )
    relative_growth_acceleration = savgol_filter(
        log_H,
        window_length=window,
        polyorder=int(polyorder),
        deriv=2,
        delta=dt,
        mode="interp",
    )

    half_window = window // 2
    idx_start = max(int(idx_cr_sim) + 1, half_window)
    idx_stop = int(t.size) - half_window
    if idx_start >= idx_stop:
        return out
    search_indices = np.arange(idx_start, idx_stop, dtype=int)
    finite = np.isfinite(relative_growth_rate[search_indices])
    if not np.any(finite):
        return out
    search_indices = search_indices[finite]
    idx_growth_peak = int(
        search_indices[np.argmax(relative_growth_rate[search_indices])]
    )
    deceleration_indices = search_indices[search_indices >= idx_growth_peak]
    finite_deceleration = np.isfinite(
        relative_growth_acceleration[deceleration_indices]
    )
    if not np.any(finite_deceleration):
        return out
    deceleration_indices = deceleration_indices[finite_deceleration]
    idx_slow = int(
        deceleration_indices[
            np.argmin(relative_growth_acceleration[deceleration_indices])
        ]
    )

    out["ok"] = True
    out["t_slow_sim"] = float(t[idx_slow])
    out["idx_slow"] = idx_slow
    out["relative_growth_rate"] = float(relative_growth_rate[idx_slow])
    out["relative_growth_acceleration"] = float(
        relative_growth_acceleration[idx_slow]
    )
    out["smoothing_window"] = window
    return out


def estimate_t_rise_sim_from_H_rise(
    t_saved,
    amplitude_series,
    H_rise: float,
) -> dict:
    t = np.asarray(t_saved, dtype=float)
    m = np.asarray(amplitude_series, dtype=float)
    H_rise_f = float(H_rise)
    out = {
        "ok": False,
        "t_rise_sim": np.nan,
        "idx_cross": -1,
    }
    if t.size == 0 or m.size == 0 or t.size != m.size:
        return out
    if not np.isfinite(H_rise_f) or H_rise_f <= 0:
        return out

    above = np.where(m > H_rise_f)[0]
    if above.size == 0:
        return out

    i1 = int(above[0])
    out["idx_cross"] = i1
    out["t_rise_sim"] = float(t[i1])
    out["ok"] = True
    return out


def compute_dh_dx(h, dx):
    h_arr = np.asarray(h, dtype=float)
    return (np.roll(h_arr, -1) - np.roll(h_arr, 1)) / (2.0 * float(dx))


def compute_shortening_integral(h, x):
    x_arr = np.asarray(x, dtype=float)
    h_arr = np.asarray(h, dtype=float)
    if x_arr.size < 2:
        return 0.0
    dx = float(x_arr[1] - x_arr[0])
    dh = compute_dh_dx(h_arr, dx)
    return 0.5 * float(np.sum(dh ** 2) * dx)


def compute_axial_strain(L0, delta_L_t, h, x):
    L_n = float(L0) + float(delta_L_t)
    if L_n <= 0.0:
        return float("nan")
    L_bar = float(L0) + compute_shortening_integral(h, x)
    return (L_n - L_bar) / L_n


def compute_P(ES, L0, delta_L_t, h, x):
    eps = compute_axial_strain(L0, delta_L_t, h, x)
    if not np.isfinite(eps):
        return float("nan")
    return float(ES) * eps


def P_cr_theory_linear(EI_val: float, k_s_val: float) -> float:
    if EI_val > 0 and k_s_val > 0:
        return float(2.0 * np.sqrt(k_s_val * EI_val))
    return float("nan")


def t_cr_theory_linear(
    *,
    ES: float,
    L0: float,
    P_cr: float,
    natural_elongation_rate: float,
) -> float:
    g = float(natural_elongation_rate)
    es = float(ES)
    l0 = float(L0)
    pcr = float(P_cr)
    if g <= 0 or not np.isfinite(pcr) or es <= pcr:
        return float("nan")
    return pcr * l0 / (g * (es - pcr))


def compute_H_series(h_saved) -> np.ndarray:
    return np.array(
        [float(np.max(np.abs(h - np.mean(h)))) for h in h_saved], dtype=float
    )


def compute_q_max_series(h_saved, x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.size < 2:
        return np.zeros(len(h_saved), dtype=float)
    dx = float(x_arr[1] - x_arr[0])
    return np.array(
        [float(np.max(np.abs(compute_dh_dx(h, dx)))) for h in h_saved],
        dtype=float,
    )


def modal_amplitude_rfft(h: np.ndarray, n: int, N: int) -> float:
    a = np.fft.rfft(h - np.mean(h))
    if n >= a.size:
        return 0.0
    return float(np.abs(a[n]) * 2.0 / float(N))


def build_mode_amplitude_series(
    h_saved,
    N: int,
    modes: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
) -> dict[int, np.ndarray]:
    return {
        int(n): np.array(
            [modal_amplitude_rfft(h, int(n), N) for h in h_saved], dtype=float
        )
        for n in modes
    }


def print_modal_amplitudes_at_index(
    h_saved,
    idx: int,
    N: int,
    *,
    indent: str = "    ",
    header: str | None = None,
) -> None:
    if not (0 <= idx < len(h_saved)):
        return
    if header is not None:
        print(header)
    h_at = h_saved[idx]
    for n_mode in range(1, 7):
        a_n = modal_amplitude_rfft(h_at, n_mode, N)
        print(f"{indent}n={n_mode}: {a_n:.6e} mm")


def print_modal_v_turn(
    t_saved,
    h_saved,
    N: int,
    *,
    modes: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
) -> None:
    amps_by_mode = build_mode_amplitude_series(h_saved, N, modes)
    print("  per-mode amplitude minimum (V-turn; rFFT, DC removed):")
    for n in modes:
        turn = estimate_t_cr_sim_from_H_minimum(
            t_saved, amps_by_mode[int(n)], exclude_t0=True
        )
        if not turn["ok"]:
            print(f"    mode n={n}: could not estimate t_cr_n")
            continue
        a_n_min = float(turn["H_min"])
        t_cr_n = float(turn["t_cr_sim"])
        print(
            f"    mode n={n}: t_cr_n = {t_cr_n:.4f} year, "
            f"|â_{n}|_min = {a_n_min:.6e} mm"
        )


def estimate_first_t_D_positive_at_mode(
    t_saved,
    P_saved,
    *,
    n_mode: int,
    L0_val: float,
    EI_val: float,
    k_s_val: float,
) -> dict:
    t = np.asarray(t_saved, dtype=float)
    P = np.asarray(P_saved, dtype=float)
    out = {
        "ok": False,
        "t_cross": np.nan,
        "idx_cross": -1,
        "D_at_cross": np.nan,
    }
    if t.size == 0 or P.size != t.size:
        return out
    D_series = np.array(
        [
            D_at_mode(float(P[i]), int(n_mode), L0_val, EI_val, k_s_val)
            for i in range(t.size)
        ],
        dtype=float,
    )
    idx_pos = np.flatnonzero(D_series > 0.0)
    if idx_pos.size == 0:
        return out
    idx = int(idx_pos[0])
    out["ok"] = True
    out["t_cross"] = float(t[idx])
    out["idx_cross"] = idx
    out["D_at_cross"] = float(D_series[idx])
    return out


def print_mode_D_first_positive_crossings(
    t_saved,
    P_saved,
    *,
    L0_val: float,
    EI_val: float,
    k_s_val: float,
    modes: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
) -> None:
    print("  per-mode first D_n > 0 crossing (D_n = P k_n^2 - EI k_n^4 - k_s):")
    for n in modes:
        cross = estimate_first_t_D_positive_at_mode(
            t_saved,
            P_saved,
            n_mode=int(n),
            L0_val=L0_val,
            EI_val=EI_val,
            k_s_val=k_s_val,
        )
        if not cross["ok"]:
            print(f"    mode n={n}: D_n never > 0 in saved series")
            continue
        print(
            f"    mode n={n}: t_D_pos = {cross['t_cross']:.4f} year, "
            f"D_n = {cross['D_at_cross']:.6e}"
        )


def print_P_peak_time_and_H(
    t_saved: np.ndarray, h_saved, P_saved: np.ndarray
) -> dict[str, float | int]:
    t_arr = np.asarray(t_saved, dtype=float)
    P_arr = np.asarray(P_saved, dtype=float)
    H_arr = compute_H_series(h_saved)
    if P_arr.size == 0:
        print("P(t) peak (on saved series)")
        print("  Cannot compute: saved series is empty.")
        return {}
    idx = int(np.argmax(P_arr))
    t_peak = float(t_arr[idx])
    P_peak = float(P_arr[idx])
    H_peak = float(H_arr[idx])
    print("P(t) peak (on saved series)")
    print(f"  t_peak = {t_peak:.4f} year")
    print(f"  P_peak = {P_peak:.6f}")
    print(f"  H_peak = max_x|h - hbar| = {H_peak:.6e} mm")
    return {
        "idx": idx,
        "t_peak": t_peak,
        "P_peak": P_peak,
        "H_peak": H_peak,
    }


def relative_rho_discrepancy(rho_approx: float, rho_geom: float) -> float:
    rg = float(rho_geom)
    ra = float(rho_approx)
    if not np.isfinite(rg) or not np.isfinite(ra) or abs(rg) == 0.0:
        return float("nan")
    return float(abs(ra - rg) / abs(rg))


def print_q_max_and_rho_discrepancy_checkpoints(
    t_saved: np.ndarray,
    h_saved,
    x: np.ndarray,
    L0_ref: float,
    checkpoints: list[tuple[str, int]],
) -> None:
    t_arr = np.asarray(t_saved, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    print("q_max and relative rho discrepancy at checkpoints:")
    print(
        "  relative discrepancy := |ρ_approx(t) - ρ_geom(t)| / ρ_geom(t)"
    )
    n_saved = len(h_saved)
    for label, idx in checkpoints:
        if idx < 0 or idx >= n_saved:
            print(f"  [{label}] skipped (invalid index {idx})")
            continue
        h = h_saved[idx]
        q_max = float(np.max(np.abs(compute_dh_dx(h, float(x_arr[1] - x_arr[0])))))
        rho_geom = excess_length_ratio_rho(h, x_arr, L0_ref)
        rho_approx = excess_length_ratio_rho_approx(h, x_arr, L0_ref)
        rel = relative_rho_discrepancy(rho_approx, rho_geom)
        print(f"  [{label}] t = {float(t_arr[idx]):.4f} year (idx={idx})")
        print(f"    max_x |∂h/∂x| = {q_max:.6f}")
        print(f"    ρ_geom = {rho_geom:.6e}")
        print(f"    ρ_approx = {rho_approx:.6e}")
        if np.isfinite(rel):
            print(
                f"    relative discrepancy = {rel:.6e}  ({100.0 * rel:.3f} %)"
            )
        else:
            print("    relative discrepancy = nan  (ρ_geom ≈ 0)")



def centerline_arc_length(h: np.ndarray, x_1d: np.ndarray) -> float:
    h_arr = np.asarray(h, dtype=float)
    x_arr = np.asarray(x_1d, dtype=float)
    if h_arr.shape != x_arr.shape:
        raise ValueError("h and x must have the same shape for arc-length computation")
    if x_arr.size < 2:
        return 0.0
    dx = float(x_arr[1] - x_arr[0])
    dhdx = compute_dh_dx(h_arr, dx)
    return float(np.sum(np.sqrt(1.0 + dhdx**2)) * dx)


def build_L_series(h_saved, x: np.ndarray) -> np.ndarray:
    return np.array(
        [centerline_arc_length(h, x) for h in h_saved], dtype=float
    )


def excess_length_ratio_rho(h: np.ndarray, x_1d: np.ndarray, L0_ref: float) -> float:
    h_arr = np.asarray(h, dtype=float)
    x_arr = np.asarray(x_1d, dtype=float)
    if h_arr.shape != x_arr.shape:
        raise ValueError("h and x must have the same shape for rho computation")
    l0 = float(L0_ref)
    if x_arr.size < 2 or l0 == 0.0:
        return 0.0
    dx = float(x_arr[1] - x_arr[0])
    dhdx = compute_dh_dx(h_arr, dx)
    integrand = np.sqrt(1.0 + dhdx**2) - 1.0
    return float(np.sum(integrand) * dx / l0)


def excess_length_ratio_rho_approx(
    h: np.ndarray, x_1d: np.ndarray, L0_ref: float
) -> float:
    l0 = float(L0_ref)
    if l0 == 0.0:
        return 0.0
    return float(compute_shortening_integral(h, x_1d) / l0)


def build_rho_series(h_saved, x: np.ndarray, L0_ref: float) -> np.ndarray:
    return np.array(
        [excess_length_ratio_rho(h, x, L0_ref) for h in h_saved], dtype=float
    )


def build_rho_approx_series(h_saved, x: np.ndarray, L0_ref: float) -> np.ndarray:
    return np.array(
        [excess_length_ratio_rho_approx(h, x, L0_ref) for h in h_saved],
        dtype=float,
    )


def run_buckling_simulation_with_P_coupling(
    mu=_DEFAULT_MU,
    EI=_DEFAULT_EI,
    k_s=_DEFAULT_K_S,
    ES=_DEFAULT_ES,
    L0=_DEFAULT_L0,
    N=_DEFAULT_N,
    t_min=_DEFAULT_T_MIN,
    t_max=_DEFAULT_T_MAX,
    dt=_DEFAULT_DT,
    delta_L_func=None,
    natural_elongation_rate=_DEFAULT_NATURAL_ELONGATION_RATE,
    noise_std=_DEFAULT_NOISE_STD,
    seed=_DEFAULT_SEED,
    save_every=_DEFAULT_SAVE_EVERY,
    initial_condition_type=_DEFAULT_INITIAL_CONDITION_TYPE,
    initial_modes=None,
    r_o=_DEFAULT_R_O,
    initial_mode_search_max=_DEFAULT_INITIAL_MODE_SEARCH_MAX,
    initial_mode_lower_buffer=_DEFAULT_INITIAL_MODE_LOWER_BUFFER,
    initial_mode_upper_buffer=_DEFAULT_INITIAL_MODE_UPPER_BUFFER,
    initial_delta_L_arc_over_L0=_DEFAULT_INITIAL_DELTA_L_ARC_OVER_L0,
):
    ES = float(ES)
    rng = np.random.default_rng(seed)
    t_min = float(t_min)
    t_max = float(t_max)
    if delta_L_func is None:
        delta_L_func = lambda t, _t0=t_min: natural_elongation_rate * (t - _t0)

    dx = L0 / N
    x = np.linspace(0.0, L0 - dx, N)

    main_diag = -2.0 * np.ones(N)
    off_diag = 1.0 * np.ones(N - 1)
    D2 = diags(
        diagonals=[off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="lil",
    )
    D2[0, -1] = 1.0
    D2[-1, 0] = 1.0
    D2 = (1.0 / dx**2) * D2.tocsr()
    D4 = D2 @ D2
    I_mat = eye(N, format="csr")

    h, _initial_modes_used = make_initial_condition(
        x=x,
        L0=L0,
        rng=rng,
        initial_condition_type=initial_condition_type,
        noise_std=noise_std,
        EI=EI,
        k_s=k_s,
        N=N,
        r_o=r_o,
        initial_modes=initial_modes,
        initial_mode_search_max=initial_mode_search_max,
        initial_mode_lower_buffer=initial_mode_lower_buffer,
        initial_mode_upper_buffer=initial_mode_upper_buffer,
        initial_delta_L_arc_over_L0=initial_delta_L_arc_over_L0,
    )

    n_steps = int(np.round((t_max - t_min) / dt))
    t_saved = []
    h_saved = []
    P_saved = []
    epsilon_saved = []

    def save_state(step, h_vec, P_val, eps_val):
        t_saved.append(t_min + step * dt)
        h_saved.append(h_vec.copy())
        P_saved.append(P_val)
        epsilon_saved.append(eps_val)

    delta_L_0 = delta_L_func(t_min)
    eps0 = compute_axial_strain(L0, delta_L_0, h, x)
    P0 = float(ES) * eps0 if np.isfinite(eps0) else float("nan")
    save_state(0, h, P0, eps0)

    for n in range(1, n_steps + 1):
        t_now = t_min + n * dt
        delta_L_t = delta_L_func(t_now)
        eps = compute_axial_strain(L0, delta_L_t, h, x)
        P = float(ES) * eps if np.isfinite(eps) else float("nan")
        L = -P * D2 - EI * D4 - k_s * I_mat
        A = (mu * I_mat - dt * L).tocsr()
        b = mu * h
        try:
            A_lu = splu(A.tocsc())
            h = A_lu.solve(b)
        except RuntimeError:
            raise RuntimeError(
                f"Time step {n}: matrix singular. Try smaller dt or check parameters."
            )
        if n % save_every == 0:
            eps_saved_val = compute_axial_strain(L0, delta_L_t, h, x)
            P_saved_val = (
                float(ES) * eps_saved_val
                if np.isfinite(eps_saved_val)
                else float("nan")
            )
            save_state(n, h, P_saved_val, eps_saved_val)

    return (
        x,
        np.array(t_saved),
        np.array(h_saved),
        np.array(P_saved),
        np.array(epsilon_saved),
    )


def save_buckling_animation(x, t_saved, h_saved, P_saved, filename="buckling_evolution.gif"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [2, 1]})
    fig.subplots_adjust(hspace=0.3)

                
    v_max = np.max(np.abs(h_saved))
    line_h, = ax1.plot(x, h_saved[0], color=blue_color, lw=2)
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(-v_max * 1.1, v_max * 1.1)
    ax1.set_ylabel(r"Displacement $h(x,t)$")
    ax1.grid(True, alpha=0.3)
    time_text = ax1.text(0.02, 0.9, "", transform=ax1.transAxes, fontsize=annotation_fontsize, weight="bold")

             
    ax2.plot(t_saved, P_saved, color=orange_color, lw=1, alpha=0.3)
    line_p_trail, = ax2.plot([], [], color=orange_color, lw=2)
    point_p, = ax2.plot([], [], "o", color=orange_color, ms=5)
    ax2.set_xlim(t_saved[0], t_saved[-1])
    P_min = float(np.min(P_saved))
    P_max = float(np.max(P_saved))
    P_span = max(P_max - P_min, 1e-12)
    ax2.set_ylim(P_min - 0.05 * P_span, P_max + 0.05 * P_span)
    ax2.set_xlabel(r"Time, $t$ (year)")
    ax2.set_ylabel(r"Axial load $P(t)$ (N)")
    ax2.grid(True, alpha=0.3)

    def update(i):
        line_h.set_ydata(h_saved[i])
        time_text.set_text(f"Time: {t_saved[i]:.1f}")
        line_p_trail.set_data(t_saved[: i + 1], P_saved[: i + 1])
        point_p.set_data([t_saved[i]], [P_saved[i]])
        return line_h, time_text, line_p_trail, point_p

    ani = animation.FuncAnimation(fig, update, frames=len(t_saved), interval=30, blit=True)
    try:
        ani.save(filename, writer="pillow", fps=30, dpi=150)
        print(f"Saved movie: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"Save failed (Pillow required): {e}")
    plt.close()


def save_dispersion_animation(t_saved, P_saved, EI, k_s, L0, mu, filename="buckling_P_coupling_dispersion_evolution.gif"):
                     
    if EI > 0 and k_s > 0:
        k_cr_local = (k_s / EI) ** 0.25
        k_max_disp = 2.5 * k_cr_local
    else:
        k_max_disp = np.pi * 6 / L0
    k_vals = np.linspace(0.0, k_max_disp, 200)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    line, = ax.plot([], [], color=blue_color, lw=1.5)

                                                                         
    for n_mode in [1, 2, 3, 4, 5]:
        k_n = 2.0 * np.pi * n_mode / L0
        if 0.0 <= k_n <= k_max_disp:
            ax.axvline(k_n, color=gray_color, linestyle=":", linewidth=0.8, alpha=0.7)
            ax.text(
                k_n,
                0.0,
                rf"$n={n_mode}$",
                ha="center",
                va="top",
                fontsize=text_fontsize,
                rotation=0,
            )

                                               
    peak_marker, = ax.plot(
        [],
        [],
        linestyle="none",
        marker="x",
        color=orange_color,
        markersize=9,
        markeredgewidth=1.6,
        zorder=15,
        clip_on=False,
    )

           
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.3)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top")

                             
    sigma_samples = []
    for i in range(0, len(t_saved), max(1, len(t_saved) // 50)):
        P_i = P_saved[i]
        sigma_samples.append(
            compute_sigma(k_vals, P_i, EI, k_s, mu)
        )
    if sigma_samples:
        sigma_all = np.concatenate(sigma_samples)
        ylim_t = _sigma_ylim_tight(sigma_all)
        if ylim_t is not None:
            ax.set_ylim(ylim_t[0], ylim_t[1])
    ax.set_xlim(0.0, k_max_disp)

                        
    xlim_disp = ax.get_xlim()
    ylim_disp = ax.get_ylim()
    ax.text(xlim_disp[1] * 1.02, 0.0, r"$k$", ha="left", va="center")
    ax.text(0.0, ylim_disp[1] * 1.02, r"$\sigma$", ha="center", va="bottom")

    ax.set_title(r"Time evolution of dispersion relation $\sigma(k,t)$")

    def update(frame):
        P_i = P_saved[frame]
        sigma_vals = compute_sigma(k_vals, P_i, EI, k_s, mu)
        line.set_data(k_vals, sigma_vals)
        j_max = int(np.nanargmax(sigma_vals))
        peak_marker.set_data([k_vals[j_max]], [sigma_vals[j_max]])
        time_text.set_text(f"t = {t_saved[frame]:.1f}")
        return line, peak_marker, time_text

                                                      
    ani = animation.FuncAnimation(
        fig, update, frames=len(t_saved), interval=50, blit=False
    )
    try:
        ani.save(filename, writer="pillow", fps=20, dpi=150)
        print(f"Saved dispersion animation: {filename}")
    except Exception as e:
        print(f"Failed to save dispersion animation (Pillow required): {e}")
    plt.close(fig)


def _align_twin_y_zero(
    ax: plt.Axes,
    ax_twin: plt.Axes,
    y_left: np.ndarray,
    y_right: np.ndarray,
) -> None:
    yl = np.asarray(y_left, dtype=float)
    yr = np.asarray(y_right, dtype=float)
    yl = yl[np.isfinite(yl)]
    yr = yr[np.isfinite(yr)]
    if yl.size == 0 or yr.size == 0:
        return
    lo_l, hi_l = float(np.min(yl)), float(np.max(yl))
    lo_r, hi_r = float(np.min(yr)), float(np.max(yr))

    if lo_l >= 0 and lo_r >= 0:
        pad_l = 0.05 * max(hi_l, 1e-30)
        pad_r = 0.05 * max(hi_r, 1e-30)
        ax.set_ylim(0.0, hi_l + pad_l)
        ax_twin.set_ylim(0.0, hi_r + pad_r)
        return

    if lo_l >= 0 or hi_l <= 0 or lo_r >= 0 or hi_r <= 0:
        return
    frac_l = -lo_l / (hi_l - lo_l)
    frac_r = -lo_r / (hi_r - lo_r)
    if not np.isfinite(frac_l) or not np.isfinite(frac_r) or frac_l <= 0 or frac_r <= 0:
        return
    span_l = hi_l - lo_l
    span_r = hi_r - lo_r
    if frac_l >= frac_r:
        new_lo_r = -frac_l * span_r / (1.0 - frac_l)
        ax_twin.set_ylim(new_lo_r, hi_r)
    else:
        new_lo_l = -frac_r * span_l / (1.0 - frac_r)
        ax.set_ylim(new_lo_l, hi_l)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    make_gifs = _DEFAULT_MAKE_GIF

    def calculate_dominant_wavelength(x_arr, h_arr, L_domain, amplitude_threshold):
        h_centered = h_arr - np.mean(h_arr)
        if np.max(np.abs(h_centered)) < amplitude_threshold:
            return np.nan
                   
        fft_h = np.fft.fft(h_centered)

        dx = x_arr[1] - x_arr[0] if len(x_arr) > 1 else L_domain / len(h_arr)
        freqs = np.fft.fftfreq(len(h_arr), dx)
        power = np.abs(fft_h) ** 2

                    
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_power = power[positive_mask]

        if len(positive_freqs) == 0:
            return np.nan
                                 
        max_idx = np.argmax(positive_power)
        dominant_freq = positive_freqs[max_idx]
        if dominant_freq > 0:
            return 1.0 / dominant_freq
        return np.nan


               
    E = _DEFAULT_E
    k_s = _DEFAULT_K_S
    r_o, r_i = _DEFAULT_R_O, _DEFAULT_R_I
    I = _DEFAULT_I
    S = _DEFAULT_S
    mu = _DEFAULT_MU
    EI = _DEFAULT_EI
    L0 = _DEFAULT_L0
    N = _DEFAULT_N
    t_min = _DEFAULT_T_MIN
    t_max = _DEFAULT_T_MAX
    dt = _DEFAULT_DT
    ES = _DEFAULT_ES
    natural_elongation_rate = _DEFAULT_NATURAL_ELONGATION_RATE
    age_offset = _DEFAULT_AGE_OFFSET
    delta_L_func = lambda t, _t0=t_min: natural_elongation_rate * (t - _t0)

    seed = _DEFAULT_SEED
    noise_std = _DEFAULT_NOISE_STD
    save_every = _DEFAULT_SAVE_EVERY

    initial_condition_type = _DEFAULT_INITIAL_CONDITION_TYPE
    initial_mode_lower_buffer = _DEFAULT_INITIAL_MODE_LOWER_BUFFER
    initial_mode_upper_buffer = _DEFAULT_INITIAL_MODE_UPPER_BUFFER
    initial_mode_search_max = _DEFAULT_INITIAL_MODE_SEARCH_MAX
    initial_delta_L_arc_over_L0 = _DEFAULT_INITIAL_DELTA_L_ARC_OVER_L0

    plot_time_interval = _DEFAULT_PLOT_TIME_INTERVAL

    H_rise = _DEFAULT_H_RISE

    print(f"** Default parameters **")
    print(f"Anatomical and material parameters:")
    print(f"    r_i = {r_i} mm")
    print(f"    r_o = {r_o} mm")
    print(f"    E = {E} N/mm^2")

    print(f"Model and effective parameters:")
    print(f"    k_s={k_s} N/mm^2")
    print(f"    natural_elongation_rate g = {natural_elongation_rate} mm/year")
    print(f"    mu = {mu} N year/mm^2")

    print(f"Derived geometric parameters:")
    print(f"    I = {I} mm^4")
    print(f"    S = {S} mm^2")
    print(f"    EI = {EI} N mm^2")
    print(f"    ES = {ES:.6f} N")

    print(f"Simulation parameters:")
    print(f"    L0 = {L0} mm")
    print(f"    N = {N}")
    print(f"    t_min = {t_min} (simulation time start)")
    print(f"    t_max = {t_max} (simulation time end)")
    print(f"    => age-equivalent window = {age_offset}-{age_offset + t_max} year (20+t axis on figures)")
    print(f"    dt = {dt}")
    print(f"    H_rise = {H_rise} mm (H(t)=max|h-hbar| threshold for t_rise_sim)")
    print(f"    seed = {seed}")
    print(f"    save_every = {save_every}")
    print(f"    plot_time_interval (h-evolution, dispersion) = {plot_time_interval}")
    print(f"    initial_condition_type = {initial_condition_type}")
    if initial_condition_type == "white_noise":
        print(f"    noise_std = {noise_std}")
    elif initial_condition_type == "low_mode":
        print(f"    initial_mode_lower_buffer = {initial_mode_lower_buffer}")
        print(f"    initial_mode_upper_buffer = {initial_mode_upper_buffer}")
        print(f"    initial_mode_search_max = {initial_mode_search_max}")
        print(f"    H0 scale = {noise_std}")
        dx_preview = L0 / N
        x_preview = np.linspace(0.0, L0 - dx_preview, N)
        rng_preview = np.random.default_rng(seed)
        modes_preview = select_initial_modes_around_critical_mode(
            EI,
            k_s,
            L0,
            N_grid=N,
            n_search_max=initial_mode_search_max,
            lower_buffer=initial_mode_lower_buffer,
            upper_buffer=initial_mode_upper_buffer,
        )
        h_preview = make_equal_low_mode_initial_condition(
            x_preview,
            L0,
            rng_preview,
            noise_std,
            modes_preview,
        )
        dL_arc0_preview = compute_shortening_integral(h_preview, x_preview)
        print(
            f"    ΔL_arc(0)/L0 = {100.0 * dL_arc0_preview / L0:g} %"
        )
        delta_L_t0 = delta_L_func(t_min)
        P0 = compute_P(ES, L0, delta_L_t0, h_preview, x_preview)
        P_cr_preview = P_cr_theory_linear(EI, k_s)
        abs_P0 = abs(P0)
        if np.isfinite(P_cr_preview) and P_cr_preview > 0.0:
            p0_over_pcr = abs_P0 / P_cr_preview
        else:
            p0_over_pcr = float("nan")
        print(f"    P(0) = {P0:.6f} N")
        print(
            f"    |P(0)| / P_cr = {p0_over_pcr:.6e}  "
            f"(P_cr = {P_cr_preview:.6f} N, linear theory)"
        )
    elif initial_condition_type == "fourier_superposition":
        n_geom = max_fourier_mode_from_geometry(L0, r_o)
        print(f"    n_max = floor(L0/(2 r_o)) = {n_geom}")
        print(
            f"    ΔL_arc(0)/L0 = {100.0 * initial_delta_L_arc_over_L0:g} %"
        )
        dx_preview = L0 / N
        x_preview = np.linspace(0.0, L0 - dx_preview, N)
        rng_preview = np.random.default_rng(seed)
        modes_preview = select_fourier_modes_geometric(L0, r_o, N_grid=N)
        h_preview = make_fourier_superposition_initial_condition(
            x_preview,
            L0,
            rng_preview,
            modes_preview,
            delta_L_arc_over_L0=initial_delta_L_arc_over_L0,
        )
        delta_L_t0 = delta_L_func(t_min)
        P0 = compute_P(ES, L0, delta_L_t0, h_preview, x_preview)
        P_cr_preview = P_cr_theory_linear(EI, k_s)
        abs_P0 = abs(P0)
        if np.isfinite(P_cr_preview) and P_cr_preview > 0.0:
            p0_over_pcr = abs_P0 / P_cr_preview
        else:
            p0_over_pcr = float("nan")
        print(f"    P(0) = {P0:.6f} N")
        print(
            f"    |P(0)| / P_cr = {p0_over_pcr:.6e}  "
            f"(P_cr = {P_cr_preview:.6f} N, linear theory)"
        )
    else:
        raise ValueError(
            "initial_condition_type must be one of "
            "'white_noise', 'low_mode', or 'fourier_superposition'. "
            f"Got {initial_condition_type!r}."
        )


    print("================================================================================================")
    print("================================================================================================")
    print("Linear stability analysis, analytical solution:")
    k_cr = (k_s / EI) ** (1 / 4) if EI > 0 else 0.0
    lambda_cr = 2 * np.pi / k_cr if k_cr > 0 else np.nan
    P_cr = P_cr_theory_linear(EI, k_s)
    t_cr = t_cr_theory_linear(
        ES=ES, L0=L0, P_cr=P_cr, natural_elongation_rate=natural_elongation_rate
    )

    print(f"    Critical wavenumber k_cr (theory) = {k_cr:.4f}")
    print(f"    Critical wavelength λ_cr (theory) = {lambda_cr:.4f}")
    print(f"    Critical load P_cr (theory) = {P_cr:.6f}  (2√(k_s·EI); linear critical axial load)")
    print(f"    Critical time t_cr (theory) = {t_cr:.4f}  (theory, P=ES/(L0+g t)*g t, shortening ignored)")

    print("================================================================================================")
    print("================================================================================================")
    print("Initial condition:")
    initial_modes = None
    ic_type = str(initial_condition_type).lower().replace("-", "_")

    if ic_type == "low_mode":
        disc_info = discrete_critical_mode_info(
            EI,
            k_s,
            L0,
            N_grid=N,
            n_search_max=initial_mode_search_max,
        )

        initial_modes = select_initial_modes_around_critical_mode(
            EI,
            k_s,
            L0,
            N_grid=N,
            n_search_max=initial_mode_search_max,
            lower_buffer=initial_mode_lower_buffer,
            upper_buffer=initial_mode_upper_buffer,
        )

        print("Low-mode initial condition:")
        if disc_info["ok"]:
            print(
                f"    discrete critical mode n* = {disc_info['n_star']}, "
                f"k_* = {disc_info['k_star']:.4f}, "
                f"lambda_* = {disc_info['lambda_star']:.4f} mm, "
                f"P_cr,discrete = {disc_info['P_cr_discrete']:.6f} N"
            )
        print(f"    initial_modes = {initial_modes}")
        print(
            f"    initial H0 = {noise_std:g} mm "
            f"(max_x |h(x,0) - mean(h)| target)"
        )
    elif ic_type == "fourier_superposition":
        n_geom = max_fourier_mode_from_geometry(L0, r_o)
        initial_modes = select_fourier_modes_geometric(L0, r_o, N_grid=N)
        print("Fourier-superposition initial condition:")
        print(f"    n = 1 .. {n_geom}  (floor(L0/(2 r_o)); grid cap N/2={N // 2})")
        print(f"    n_modes used = {len(initial_modes)}")
        if len(initial_modes) <= 12:
            print(f"    initial_modes = {initial_modes}")
        else:
            print(
                f"    initial_modes = {initial_modes[:6]} ... {initial_modes[-3:]}"
            )
        print(
            f"    target ΔL_arc(0)/L0 = {100.0 * initial_delta_L_arc_over_L0:g} %)"
        )
    print("================================================================================================")
    print("================================================================================================")
    print("Simulation with P-coupling...")

    x, t_saved, h_saved, P_saved, epsilon_saved = run_buckling_simulation_with_P_coupling(
        mu=mu,
        EI=EI,
        k_s=k_s,
        ES=ES,
        L0=L0,
        N=N,
        t_min=t_min,
        t_max=t_max,
        dt=dt,
        delta_L_func=delta_L_func,
        noise_std=noise_std,
        seed=seed,
        save_every=save_every,
        initial_condition_type=initial_condition_type,
        initial_modes=initial_modes,
        r_o=r_o,
        initial_mode_search_max=initial_mode_search_max,
        initial_mode_lower_buffer=initial_mode_lower_buffer,
        initial_mode_upper_buffer=initial_mode_upper_buffer,
        initial_delta_L_arc_over_L0=initial_delta_L_arc_over_L0,
    )

    print(f"Done. Saved time points: {len(t_saved)}")
    print(f"P(t) range (simulation) = [{P_saved.min():.4f}, {P_saved.max():.4f}]")
    print("Initial modal amplitudes |â_n(0)|:")
    for n_mode in range(1, 7):
        a0_n = modal_amplitude_rfft(h_saved[0], n_mode, N)
        print(f"    n={n_mode}: {a0_n:.6e} mm")

    H0_actual = float(np.max(np.abs(h_saved[0] - np.mean(h_saved[0]))))
    dL_arc0 = compute_shortening_integral(h_saved[0], x)
    print(f"Initial H(0) = max_x|h(x,0)-hbar(0)| = {H0_actual:.6e} mm")
    print(f"Initial ΔL_arc(0)/L0 = {dL_arc0 / L0:.6e}  (ΔL_arc = {dL_arc0:.6e} mm)")

    from simulation.simulation_results import (
        RESULT_FILENAME,
        buckling_P_coupling_run_dir,
        save_simulation_run,
    )
    run_dir = buckling_P_coupling_run_dir(
        mu=mu,
        natural_elongation_rate=natural_elongation_rate,
        age_offset=age_offset,
        t_max=t_max,
    )
    npz_path = os.path.join(run_dir, RESULT_FILENAME)
    print(f"Results (.npz) saved to: {os.path.abspath(run_dir)}")
    save_simulation_run(
        npz_path,
        x,
        t_saved,
        h_saved,
        P_saved,
        epsilon_saved=epsilon_saved,
        params={
            "mu": mu,
            "EI": EI,
            "k_s": k_s,
            "ES": ES,
            "L0": L0,
            "N": N,
            "t_min": t_min,
            "t_max": t_max,
            "age_offset": age_offset,
            "dt": dt,
            "natural_elongation_rate": natural_elongation_rate,
            "noise_std": noise_std,
            "seed": seed,
            "save_every": save_every,
            "initial_condition_type": initial_condition_type,
            "r_o": r_o,
            "initial_modes": list(initial_modes) if initial_modes is not None else None,
            "initial_mode_search_max": initial_mode_search_max,
            "initial_mode_lower_buffer": initial_mode_lower_buffer,
            "initial_mode_upper_buffer": initial_mode_upper_buffer,
            "initial_delta_L_arc_over_L0": (
                initial_delta_L_arc_over_L0
                if ic_type == "fourier_superposition"
                else None
            ),
        },
    )

    print("================================================")
    print("Mode first D_n > 0 crossing:")
    print_mode_D_first_positive_crossings(
        t_saved,
        P_saved,
        L0_val=L0,
        EI_val=EI,
        k_s_val=k_s,
    )

    print("================================================")
    print("Critical time t_cr_sim:")
    print("Global deformation H(t) = max_x |h(x,t) - hbar(t)|")
                                                            
    max_displacements = np.array([np.max(np.abs(h)) for h in h_saved])
    H_series = compute_H_series(h_saved)

                            
    H_min = np.nan
    t_cr_sim = np.nan
    idx_cr_sim = -1
    P_cr_sim = np.nan

    cr_sim = estimate_t_cr_sim_from_H_minimum(t_saved, H_series)
    if cr_sim["ok"]:
        H_min = float(cr_sim["H_min"])
        t_cr_sim = float(cr_sim["t_cr_sim"])
        idx_cr_sim = int(cr_sim["idx_min"])
        if 0 <= idx_cr_sim < len(P_saved):
            P_cr_sim = float(P_saved[idx_cr_sim])

        print(
            f"  H_min = {H_min:.6e} mm"
            f"  t_cr_sim = {t_cr_sim:.4f} year (on saved series)"
        )
        if np.isfinite(P_cr_sim):
            print(f"  P(t_cr_sim) = {P_cr_sim:.6f} N")

        idx_h = idx_cr_sim
        if 0 <= idx_h < len(max_displacements):
            print(
                f"  max|h|_min = {max_displacements[idx_h]:.6e} mm  "
                f"(H_min / max|h|_min = {H_min / max(max_displacements[idx_h], 1e-30):.4f})"
            )
        if 0 <= idx_h < len(h_saved):
            print("  modal amplitudes |â_n| at t_cr_sim (rFFT, DC removed):")
            print_modal_amplitudes_at_index(h_saved, int(idx_h), N)
    else:
        print("  could not estimate t_cr_sim from H(t) minimum")

    print_modal_v_turn(t_saved, h_saved, N)

                                                         
    t_rise_sim = np.nan
    idx_rise_sim = -1
    print("================================================")
    print("Critical time t_rise_sim:")
    print(f"H(t) threshold crossing (H_rise = {H_rise:g} mm)")
    rise = estimate_t_rise_sim_from_H_rise(
        t_saved,
        H_series,
        H_rise,
    )
    if rise["ok"]:
        t_rise_sim = float(rise["t_rise_sim"])
        idx_rise_sim = int(rise["idx_cross"])
        print(f"  t_rise_sim = {t_rise_sim:.4f} year (on saved series)")
        print(f"  delay = t_rise_sim - t_cr_sim = {t_rise_sim - t_cr_sim:+.4f} year")
        
        print("  modal amplitudes |â_n| at t_rise_sim (rFFT, DC removed):")
        print_modal_amplitudes_at_index(
            h_saved,
            idx_rise_sim,
            N,
        )
    else:
        print(
            f"  could not estimate (H(t) never exceeded H_rise = {H_rise:g} mm "
            f"in saved series; extend t_max or lower H_rise)"
        )

    print("Check consistency with t_rise from linear stability analysis:")
    if cr_sim["ok"] and np.isfinite(H_min) and H_min > 0 and H_rise > H_min:
        R_eff = H_rise / H_min
        beta_cr = (ES * natural_elongation_rate * L0 * k_cr**2) / (mu * (L0 + natural_elongation_rate * t_cr)**2)
        t_rise = t_cr + np.sqrt(2 * np.log(R_eff) / beta_cr)
        print(f"    H_rise (threshold) = {H_rise:g} mm")
        print(f"    H_min (simulation) = {H_min:g} mm")
        print(f"    effective amplification factor R_eff = {R_eff:.4f}")
        print(f"    beta_cr β = {beta_cr:.4f} year^-2")
        print(f"    t_rise (theory) = {t_rise:.4f} year")
    else:
        print("    t_rise theory estimate could not be computed.")

    print("================================================")
    print("Peak time and H(t) at P peak:")
    peak_info = print_P_peak_time_and_H(t_saved, h_saved, P_saved)
    idx_P_peak = int(peak_info["idx"]) if peak_info else -1

    print("================================================")
    print("Super-exponential / Slow growth boundary t_slow_sim:")
    print(
        "  Definition: after max d(log H)/dt, the saved time that minimizes "
        "d^2(log H)/dt^2 (Savitzky-Golay)."
    )
    t_slow_sim = np.nan
    idx_slow_sim = -1
    if cr_sim["ok"] and idx_cr_sim >= 0:
        slow_sim = estimate_t_slow_sim_from_H_growth_rate(
            t_saved,
            H_series,
            idx_cr_sim=idx_cr_sim,
        )
        if slow_sim["ok"]:
            t_slow_sim = float(slow_sim["t_slow_sim"])
            idx_slow_sim = int(slow_sim["idx_slow"])
            print(f"  t_slow_sim = {t_slow_sim:.4f} year (on saved series)")
            print(
                f"  d(log H)/dt at t_slow_sim = "
                f"{float(slow_sim['relative_growth_rate']):.6f} /year"
            )
            print(
                f"  d^2(log H)/dt^2 at t_slow_sim = "
                f"{float(slow_sim['relative_growth_acceleration']):.6f} /year^2"
            )
            print(
                f"  smoothing_window = {int(slow_sim['smoothing_window'])}"
            )
            if np.isfinite(t_cr_sim):
                print(
                    f"  Super-exponential duration = "
                    f"t_slow_sim - t_cr_sim = {t_slow_sim - t_cr_sim:+.4f} year"
                )
            if 0 <= idx_slow_sim < len(H_series):
                print(
                    f"  H(t_slow_sim) = {float(H_series[idx_slow_sim]):.6e} mm"
                )
            if 0 <= idx_slow_sim < len(P_saved):
                print(
                    f"  P(t_slow_sim) = {float(P_saved[idx_slow_sim]):.6f} N"
                )
        else:
            print("  could not estimate t_slow_sim from H(t) growth rate")
    else:
        print("  skipped: t_cr_sim unavailable")

    print("================================================")
    idx_final = len(h_saved) - 1
    checkpoints: list[tuple[str, int]] = []
    if idx_P_peak >= 0:
        checkpoints.append(("axial-load peak", idx_P_peak))
    if idx_rise_sim >= 0:
        checkpoints.append(
            (f"H(t)=H_rise crossing (H_rise={H_rise:g} mm)", idx_rise_sim)
        )
    if idx_slow_sim >= 0:
        checkpoints.append(("onset of slow-growth stage", idx_slow_sim))
    if idx_final >= 0:
        checkpoints.append(("final time", idx_final))
    print_q_max_and_rho_discrepancy_checkpoints(
        t_saved,
        h_saved,
        x,
        L0,
        checkpoints,
    )

    eps_max = float(np.nanmax(epsilon_saved)) if len(epsilon_saved) else float("nan")
    eps_final = float(epsilon_saved[-1]) if len(epsilon_saved) else float("nan")
    print("================================================")
    print("Axial strain ε(t) = (L_n(t) - L_bar(t)) / L_n(t)")
    print("  L_n = L0 + ΔL,  L_bar = L0 + ΔL_arc[h]")
    print(f"  ε_max = {eps_max:.6e}")
    print(f"  ε_final = {eps_final:.6e}")

    print("================================================================================================")
    print("================================================================================================")
    print("Figures:")    
                                                         
    sim_dir = run_dir
    print(f"Figures/GIFs saved to: {os.path.abspath(sim_dir)}")

    # Fig1: h(x,t) profiles
    fig1, ax1 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    t_end_plot = float(t_saved[-1])
    times_to_plot = np.arange(t_min, t_end_plot + 0.01, plot_time_interval)
                           
    if t_end_plot > 0 and (len(times_to_plot) == 0 or not np.isclose(times_to_plot[-1], t_end_plot, rtol=0, atol=1e-6)):
        if len(times_to_plot) == 0 or abs(times_to_plot[-1] - t_end_plot) > 1e-9:
            times_to_plot = np.unique(np.append(times_to_plot, t_end_plot))
    indices = _times_to_saved_indices(t_saved, times_to_plot)
    idx_last = len(h_saved) - 1
    n_non_final = max(len([i for i in indices if i != idx_last]), 1)
    cmap = mcolors.LinearSegmentedColormap.from_list("time", colors, N=n_non_final)
    j_cmap = 0
    for i in indices:
        if i == idx_last:
            continue
        ax1.plot(x, h_saved[i], color=cmap(j_cmap), lw=1.2)
        j_cmap += 1
                                    
    ax1.plot(
        x,
        h_saved[idx_last],
        color=orange_color,
        lw=2.6,
        linestyle="-",
        zorder=10,
    )
    ax1.set_xlabel(r"Spatial coordinate $x$ (mm)")
    ax1.set_ylabel(r"Displacement $h(x,t)$ (mm)")
    ax1.set_xlim(left=-5.0, right=115.0)
    ax1.grid(True, alpha=0.3)
    _out_fig1 = os.path.join(sim_dir, "Fig1_h_evolution.pdf")
    fig1.savefig(_out_fig1, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(_out_fig1)}")
    show_figure(fig1, auto_close_seconds=figure_show_seconds)

                                                                         
    #   σ(k,t) = (P(t) k^2 - EI k^4 - k_s) / μ
    fig2, ax2 = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
                            
    if EI > 0 and k_s > 0:
        k_cr_local = (k_s / EI) ** 0.25
        k_max_disp = 2.5 * k_cr_local
    else:
        k_max_disp = np.pi * 6 / L0                   
    k_vals = np.linspace(0.0, k_max_disp, 200)

                          
    sigma_all_list = []

    times_to_plot_disp = np.arange(t_min, t_end_plot + 0.01, plot_time_interval)
    if t_end_plot > 0 and (len(times_to_plot_disp) == 0 or not np.isclose(times_to_plot_disp[-1], t_end_plot, rtol=0, atol=1e-6)):
        if len(times_to_plot_disp) == 0 or abs(times_to_plot_disp[-1] - t_end_plot) > 1e-9:
            times_to_plot_disp = np.unique(np.append(times_to_plot_disp, t_end_plot))
    indices_disp = _times_to_saved_indices(t_saved, times_to_plot_disp)
    cmap_disp = mcolors.LinearSegmentedColormap.from_list("time_disp", colors, N=max(len(indices_disp), 1))

    for j, i in enumerate(indices_disp):
        t_i = t_saved[i]
        P_i = P_saved[i]
        sigma_vals = compute_sigma(k_vals, P_i, EI, k_s, mu)
        sigma_all_list.append(sigma_vals)
        ax2.plot(k_vals, sigma_vals, color=cmap_disp(j), lw=1.0, label=f"t={t_i:.0f}")
                              
        j_max = int(np.nanargmax(sigma_vals))
        ax2.plot(
            k_vals[j_max],
            sigma_vals[j_max],
            marker="x",
            color=cmap_disp(j),
            markersize=7,
            markeredgewidth=1.4,
            linestyle="none",
            clip_on=False,
            zorder=15,
        )

                                 
    if cr_sim["ok"] and idx_cr_sim >= 0:
        t_cr_sim_actual = float(t_saved[idx_cr_sim])
        P_at_t_cr_sim = float(P_saved[idx_cr_sim])
        sigma_tcr = compute_sigma(k_vals, P_at_t_cr_sim, EI, k_s, mu)
        sigma_all_list.append(sigma_tcr)
        ax2.plot(
            k_vals,
            sigma_tcr,
            color=orange_color,
            lw=2.0,
            linestyle="-",
            label=rf"$t_{{\mathrm{{cr}}}}^{{\mathrm{{sim}}}} = {t_cr_sim_actual:.0f}$",
        )
        j_max_tcr = int(np.nanargmax(sigma_tcr))
        ax2.plot(
            k_vals[j_max_tcr],
            sigma_tcr[j_max_tcr],
            marker="x",
            color=orange_color,
            markersize=9,
            markeredgewidth=1.6,
            linestyle="none",
            clip_on=False,
            zorder=16,
        )

                                 
    ax2.spines["left"].set_position("zero")
    ax2.spines["bottom"].set_position("zero")
    ax2.spines["right"].set_color("none")
    ax2.spines["top"].set_color("none")
    ax2.xaxis.set_ticks_position("bottom")
    ax2.yaxis.set_ticks_position("left")

                               
    ax2.set_xticks([])

                         
    ax2.axhline(0.0, color="k", linewidth=0.8, alpha=0.3)

                                                  
    k_n_list = []
    for n_mode in [1, 2, 3, 4, 5]:
        k_n = 2.0 * np.pi * n_mode / L0
        if 0.0 <= k_n <= k_max_disp:
            k_n_list.append((n_mode, k_n))
            ax2.axvline(k_n, color=gray_color, linestyle=":", linewidth=0.8, alpha=0.7)
            ax2.text(
                k_n,
                0.0,
                rf"$n={n_mode}$",
                ha="center",
                va="top",
                fontsize=text_fontsize,
                rotation=0,
            )

                                         
    if sigma_all_list:
        sigma_all = np.concatenate(sigma_all_list)
        ylim_t = _sigma_ylim_tight(sigma_all)
        if ylim_t is not None:
            ax2.set_ylim(ylim_t[0], ylim_t[1])

                       
    xlim_disp = ax2.get_xlim()
    ylim_disp = ax2.get_ylim()
    ax2.text(xlim_disp[1] * 1.02, 0.0, r"$k$", ha="left", va="center")
    ax2.text(0.0, ylim_disp[1] * 1.02, r"$\sigma$", ha="center", va="bottom")

    ax2.set_title(r"Time evolution of dispersion relation $\sigma(k,t)$")
    _out_fig2 = os.path.join(sim_dir, "Fig2_dispersion_evolution.pdf")
    fig2.savefig(_out_fig2, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(_out_fig2)}")
    show_figure(fig2, auto_close_seconds=figure_show_seconds)

                                         
    fig3, ax3 = plt.subplots(figsize=(6, 4), constrained_layout=True)
    v = np.max(np.abs(h_saved))
                                              
                              
    norm = mcolors.SymLogNorm(linthresh=noise_std, vmin=-v, vmax=v, base=10)
    pc = ax3.pcolormesh(x, t_saved, h_saved, cmap="RdBu_r", norm=norm, shading="auto")
    ax3.set_xlabel(r"Spatial coordinate $x$ (mm)")
    ax3.set_ylabel(r"Time, $t$ (year)")
    cbar = fig3.colorbar(pc, ax=ax3, label=r"Displacement $h(x,t)$ (mm)")
    _out_fig3 = os.path.join(sim_dir, "Fig3_h_spacetime.pdf")
    fig3.savefig(_out_fig3, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(_out_fig3)}")
    show_figure(fig3, auto_close_seconds=figure_show_seconds)

    # Fig4: global deformation H(t) + P(t)
    t_min_plot = float(t_saved[0])
    t_max_plot = float(t_saved[-1])
    fig4, ax4 = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax4_twin = ax4.twinx()
    ax4.plot(t_saved, H_series, color=blue_color, lw=1.2)
    ax4_twin.plot(t_saved, P_saved, color=orange_color, lw=1.2, linestyle="-")
    ax4.set_xlabel(r"Time, $t$ (year)")
    ax4.set_ylabel(
        r"Global deformation, $H(t)$ (mm)",
        color=blue_color,
    )
    ax4_twin.set_ylabel(r"Axial load $P(t)$ (N)", color=orange_color)
    ax4.grid(True, alpha=0.3)
    ax4_twin.grid(False)
    _align_twin_y_zero(ax4, ax4_twin, np.asarray(H_series), np.asarray(P_saved))
    if not np.isnan(P_cr):
        ax4_twin.axhline(
            y=P_cr, color=orange_color, linestyle=":", linewidth=1.2, alpha=0.7,
            label=r"$P_{\mathrm{cr}}$ (theory)",
        )

                                                     
                                            
    _xax_tr = ax4.get_xaxis_transform()
    if np.isfinite(t_cr_sim):
        ax4.axvline(t_cr_sim, color=gray_color, linestyle="--", linewidth=1.2, alpha=0.9, zorder=0)
        ax4.text(
            t_cr_sim,
            -0.18,
            r"$t_{\mathrm{cr}}^{\mathrm{sim}}$",
            transform=_xax_tr,
            ha="center",
            va="top",
            color=gray_color,
            fontsize=annotation_fontsize,
            clip_on=False,
        )
    if np.isfinite(t_rise_sim):
        ax4.axvline(t_rise_sim, color=gray_color, linestyle=":", linewidth=1.2, alpha=0.9, zorder=0)
        ax4.text(
            t_rise_sim,
            -0.18,
            r"$t_{\mathrm{rise}}^{\mathrm{sim}}$",
            transform=_xax_tr,
            ha="center",
            va="top",
            color=gray_color,
            fontsize=annotation_fontsize,
            clip_on=False,
        )
                                    
    if cr_sim["ok"] and np.isfinite(P_cr_sim):
        ax4_twin.plot(
            t_cr_sim,
            P_cr_sim,
            "o",
            markersize=ms * 2,
            color=orange_color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            clip_on=False,
            zorder=10,
            label=r"$P(t_{\mathrm{cr}}^{\mathrm{sim}})$",
        )
    ax4.set_xlim(t_min_plot, t_max_plot)
    _out_fig4 = os.path.join(sim_dir, "Fig4_max_disp_and_P.pdf")
    fig4.savefig(_out_fig4, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(_out_fig4)}")
    show_figure(fig4, auto_close_seconds=figure_show_seconds)

    # Fig5: H(t) + rho(t)
    rho_series = build_rho_series(h_saved, x, L0)
    fig5, ax5 = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax5_twin = ax5.twinx()
    ax5.plot(t_saved, H_series, color=blue_color, lw=1.2)
    ax5_twin.plot(t_saved, rho_series, color=red_color, lw=1.2)
    ax5.set_xlabel(r"Time, $t$ (year)")
    ax5.set_ylabel(r"Global deformation, $H(t)$ (mm)", color=blue_color)
    ax5_twin.set_ylabel(r"Excess-length ratio, $\rho(t)$", color=red_color)
    ax5.grid(True, alpha=0.3)
    _align_twin_y_zero(
        ax5, ax5_twin, np.asarray(H_series, dtype=float), np.asarray(rho_series, dtype=float)
    )
    ax5.set_xlim(t_min_plot, t_max_plot)
    _out_fig5 = os.path.join(sim_dir, "Fig5_max_disp_and_rho.pdf")
    fig5.savefig(_out_fig5, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(_out_fig5)}")
    show_figure(fig5, auto_close_seconds=figure_show_seconds)

                                                   
    fig6, ax6 = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    max_disp_arr = np.asarray(max_displacements, dtype=float)
    eps_log = 1e-12
    max_disp_safe = np.maximum(max_disp_arr, eps_log)
    ax6.plot(t_saved, max_disp_safe, color=blue_color, lw=1.2)
    ax6.set_yscale("log")
    ax6.set_xlabel(r"Time, $t$ (year)")
    ax6.set_ylabel(r"Maximum displacement, $\max |h(x,t)|$ (mm)")
    ax6.set_xlim(t_min_plot, t_max_plot)
    ax6.grid(True, alpha=0.3, which="both")
    _out_fig6 = os.path.join(sim_dir, "Fig6_max_disp_vs_time_logy.pdf")
    fig6.savefig(_out_fig6, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(_out_fig6)}")
    show_figure(fig6, auto_close_seconds=figure_show_seconds)

                                                            
                                                         
    if np.isfinite(H_min) and H_min > 0:
        amplitude_threshold_abs = 1.5 * H_min
    else:
        amplitude_threshold_abs = 1.5 * noise_std

    dominant_wavelengths_abs = np.array(
        [calculate_dominant_wavelength(x, h, L0, amplitude_threshold_abs) for h in h_saved]
    )
                                                            
    dominant_wavelengths_incr = np.full(len(h_saved), np.nan)
    if np.isfinite(t_cr_sim) and idx_cr_sim >= 0 and np.isfinite(H_min) and H_min > 0:
        h_cr_ref = h_saved[idx_cr_sim]
        amplitude_threshold_incr = 1.5 * H_min
        for i, h in enumerate(h_saved):
            h_incr = h - h_cr_ref
            dominant_wavelengths_incr[i] = calculate_dominant_wavelength(
                x, h_incr, L0, amplitude_threshold_incr
            )

    mask_post_critical = (t_saved >= t_cr_sim) if not np.isnan(t_cr_sim) else np.ones(len(t_saved), dtype=bool)
    t_post_critical = t_saved[mask_post_critical]
    fig7, ax7 = plt.subplots(figsize=(7, 4), constrained_layout=True)
    # Absolute FFT
    wl_abs_post = dominant_wavelengths_abs[mask_post_critical]
    valid_abs = np.isfinite(wl_abs_post)
    if np.any(valid_abs):
        ax7.plot(
            t_post_critical[valid_abs], wl_abs_post[valid_abs],
            "o-", markersize=ms, color=blue_color, label="Absolute (Original)")
                                             
    wl_incr_post = dominant_wavelengths_incr[mask_post_critical]
    valid_incr = np.isfinite(wl_incr_post)
    if np.any(valid_incr):
        ax7.plot(
            t_post_critical[valid_incr], wl_incr_post[valid_incr],
            "x-", markersize=ms, alpha=0.3, color=orange_color, label="Incremental (Onset tracking)")
    if not np.isnan(t_cr_sim):
        ax7.axvspan(t_min_plot, t_cr_sim, alpha=0.25, color=gray_color, zorder=0)
        ax7.text(t_cr_sim / 2, ax7.get_ylim()[1] * 0.05 if (np.any(valid_abs) or np.any(valid_incr)) else 1.0,
                  "Undefined", fontsize=annotation_fontsize, ha="center", va="bottom", style="italic", color=gray_color)
    ax7.axhline(y=lambda_cr, color=gray_color, linestyle="--", linewidth=1.0, alpha=0.7)
                              
    for n in [1, 2, 3, 4, 5, 6]:
        lambda_n = L0 / n
        ax7.axhline(y=lambda_n, color=lightgray_color, linestyle="-", linewidth=1.2, alpha=0.8)
        ax7.text(t_max_plot * 0.95, lambda_n * 1.02, f"n={n}",
                 fontsize=text_fontsize, ha="right", va="bottom", color=lightgray_color)
    ax7.set_xlabel(r"Time, $t$ (year)")
    ax7.set_ylabel(r"Dominant wavelength $\lambda^*$")
    ax7.legend(loc="upper right")
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(t_min_plot, t_max_plot)
    all_valid = np.concatenate([wl_abs_post[valid_abs] if np.any(valid_abs) else [],
                                wl_incr_post[valid_incr] if np.any(valid_incr) else []])
    if len(all_valid) > 0:
        ax7.set_ylim(0, np.max(all_valid) * 1.5)
    _out_fig7 = os.path.join(sim_dir, "Fig7_dominant_wavelength.pdf")
    fig7.savefig(_out_fig7, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(_out_fig7)}")
    show_figure(fig7, auto_close_seconds=figure_show_seconds)


                                                                              
    fig8, ax8 = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for n in [1, 2, 3, 4, 5]:
        amps = np.array(
            [modal_amplitude_rfft(h, n, N) for h in h_saved],
            dtype=float,
        )
        eps = 1e-12
        amps_safe = np.maximum(amps, eps)
        ax8.plot(t_saved, amps_safe, label=f"mode n={n}", linewidth=1.2)
    ax8.set_yscale("log")
    ax8.set_xlabel(r"Time, $t$ (year)")
    ax8.set_ylabel(r"Amplitude (log scale)")
    ax8.plot(t_saved, H_series, color="black", lw=1.2, label=r"$H(t)$")
    if cr_sim["ok"] and idx_cr_sim >= 0:
        ax8.axvline(t_cr_sim, color="black", linestyle="--", linewidth=1.5, alpha=0.9, label=r"$t_{\mathrm{cr}}$")
        ax8.plot(t_cr_sim, H_min, "x", color="black", markersize=4, markeredgecolor="white", zorder=10)
        ax8.text(t_cr_sim + 1, H_min * 0.8, r"$H(t)$ turns upward", fontsize=text_fontsize, color="black", style="italic")
    ax8.legend()
    ax8.set_title("Modal Amplitude Evolution (Log-y)")
    ax8.grid(True, alpha=0.3)
    _out_fig8 = os.path.join(sim_dir, "Fig8_modal_amplitude.pdf")
    fig8.savefig(_out_fig8, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(_out_fig8)}")
    show_figure(fig8, auto_close_seconds=figure_show_seconds)

                                                           
    delta_L_saved = np.array([delta_L_func(t) for t in t_saved])
    fig9, ax9 = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax9.plot(delta_L_saved, P_saved, color=blue_color, lw=1.2, label=r"$P(t)$")
    if not np.isnan(P_cr):
        ax9.axhline(y=P_cr, color=orange_color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax9.text(delta_L_saved[-1] * 0.98, P_cr * 1.02, r"$P_{\mathrm{cr}}$",
                 fontsize=annotation_fontsize, ha="right", va="bottom", color=orange_color, style="italic")
    ax9.set_xlabel(r"$\Delta L(t) = \mathrm{growth\_rate} \times (t - t_{\min})$")
    ax9.set_ylabel(r"Axial load $P(t)$")
    ax9.set_title("P(t) vs " + r"$\Delta L(t)$ (saturation near P_cr)")
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    _out_fig9 = os.path.join(sim_dir, "Fig9_P_vs_deltaL.pdf")
    fig9.savefig(_out_fig9, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(_out_fig9)}")
    show_figure(fig9, auto_close_seconds=figure_show_seconds)

    # Fig10: axial strain ε(t)
    fig10, ax10 = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax10.plot(t_saved, epsilon_saved, color=blue_color, lw=1.2)
    if np.isfinite(eps_max):
        ax10.axhline(
            eps_max,
            color=orange_color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=rf"$\varepsilon_{{\max}}={eps_max:.3e}$",
        )
        ax10.legend(loc="best")
    ax10.set_xlabel(r"Time, $t$ (year)")
    ax10.set_ylabel(r"Axial strain, $\varepsilon(t)$")
    ax10.set_xlim(t_min_plot, t_max_plot)
    ax10.grid(True, alpha=0.3)
    _out_fig10 = os.path.join(sim_dir, "Fig10_epsilon_vs_time.pdf")
    fig10.savefig(_out_fig10, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(_out_fig10)}")
    show_figure(fig10, auto_close_seconds=figure_show_seconds)

    if make_gifs:
                                             
        anim_path = os.path.join(sim_dir, "Gif1_buckling_evolution.gif")
        save_buckling_animation(x, t_saved, h_saved, P_saved, filename=anim_path)
                                                     
        anim_disp_path = os.path.join(sim_dir, "Gif2_dispersion_evolution.gif")
        save_dispersion_animation(t_saved, P_saved, EI, k_s, L0, mu, filename=anim_disp_path)