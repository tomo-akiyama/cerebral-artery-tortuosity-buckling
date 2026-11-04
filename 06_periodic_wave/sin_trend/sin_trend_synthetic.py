#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend synthetic."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
# Prefer this package over analysis_code/sin_trend (same basename modules must not collide).
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)

from sin_trend_frames import arclength, numeric_tangent, rotation_matrix_from_axis_angle  # noqa: E402


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if fn is not None:
        return float(fn(y, x))
    dx = np.diff(x)
    return float(np.sum(0.5 * dx * (y[1:] + y[:-1])))


def combined_amplitude_mm(A_y_mm: float, A_z_mm: float = 0.0) -> float:
    return float(np.sqrt(float(A_y_mm) ** 2 + float(A_z_mm) ** 2))


def polyline_arclength_first_complete_cycle_mm(
    s_vox: np.ndarray,
    xi_mm: np.ndarray,
    lambda_baseline_mm: float,
    voxel_size_mm: float,
) -> Tuple[float, str]:
    s = np.asarray(s_vox, float).reshape(-1)
    xi = np.asarray(xi_mm, float).reshape(-1)
    lam = float(lambda_baseline_mm)
    vs = float(voxel_size_mm)
    if s.size < 2 or xi.size != s.size:
        return float("nan"), "insufficient_or_mismatched_samples"
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(vs) or vs <= 0:
        return float("nan"), "invalid_lambda_or_voxel_size"
    if not np.isfinite(s).all() or not np.isfinite(xi).all():
        return float("nan"), "non_finite_s_or_xi"
    xi0 = float(xi[0])
    xi_end = xi0 + lam
    if xi_end > float(xi[-1]) + 1e-12:
        return float("nan"), "incomplete_first_cycle_on_polyline"
                                 
    s0 = float(np.interp(xi0, xi, s))
    s1 = float(np.interp(xi_end, xi, s))
    out = (s1 - s0) * vs
    if not np.isfinite(out) or out <= 0:
        return float("nan"), "non_positive_polyline_cycle_length"
    return float(out), "ok"


def arc_length_per_cycle(
    A_y_mm: float,
    lambda_x_mm: float,
    A_z_mm: float = 0.0,
    n_dense: int = 200001,
) -> float:
    lx = float(lambda_x_mm)
    if not np.isfinite(lx) or lx <= 0:
        raise ValueError(f"lambda_x_mm must be positive finite; got {lambda_x_mm!r}")
    x = np.linspace(0.0, lx, int(n_dense))
    k = 2.0 * np.pi / lx
    c = np.cos(k * x)
    dydx = float(A_y_mm) * k * c
    dzdx = float(A_z_mm) * k * c
    return _trapz(np.sqrt(1.0 + dydx**2 + dzdx**2), x)


def lambda_x_for_target_lambda_s(
    A_y_mm: float,
    lambda_s_target_mm: float,
    *,
    A_z_mm: float = 0.0,
    n_dense: int = 200001,
    xtol: float = 1e-10,
    max_iter: int = 80,
) -> float:
    ls_t = float(lambda_s_target_mm)
    if not np.isfinite(ls_t) or ls_t <= 0:
        raise ValueError(f"lambda_s_target_mm must be positive finite; got {lambda_s_target_mm!r}")
    Ay = float(A_y_mm)
    Az = float(A_z_mm)
    if combined_amplitude_mm(Ay, Az) == 0.0:
        return ls_t

    def ls_of(lx: float) -> float:
        return arc_length_per_cycle(Ay, lx, Az, n_dense=n_dense)

                                                      
    hi = ls_t
    ls_hi = ls_of(hi)
    if ls_hi < ls_t - 1e-9:
        raise ValueError(f"unexpected: ls({hi})={ls_hi} < target {ls_t}")
                                  
    lo = min(ls_t * 0.5, ls_t - 1e-6)
    lo = max(lo, 1e-6)
    ls_lo = ls_of(lo)
    shrink = 0
    while ls_lo > ls_t and shrink < 60:
        lo *= 0.5
        if lo < 1e-8:
            break
        ls_lo = ls_of(lo)
        shrink += 1
    if ls_lo > ls_t:
        raise ValueError(
            f"cannot reach lambda_s={ls_t:g} with A_y={Ay:g}, A_z={Az:g}: "
            f"min explored ls({lo:g})={ls_lo:g} still above target"
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ls_mid = ls_of(mid)
        if abs(ls_mid - ls_t) <= max(xtol, 1e-9 * ls_t) or (hi - lo) <= xtol:
            return float(mid)
        if ls_mid < ls_t:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def resample_uniform_arclength(P_dense: np.ndarray, ds_target: float) -> np.ndarray:
    s_dense, _ = arclength(P_dense)
    L = float(s_dense[-1])
    n = int(np.floor(L / ds_target)) + 1
    s_new = np.linspace(0.0, L, n)
    return np.column_stack([np.interp(s_new, s_dense, P_dense[:, j]) for j in range(3)])


def resample_uniform_arclength_with_attrs(
    P_dense: np.ndarray,
    ds_target: float,
    attrs: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    s_dense, _ = arclength(P_dense)
    L = float(s_dense[-1])
    n = int(np.floor(L / ds_target)) + 1
    s_new = np.linspace(0.0, L, n)
    P_new = np.column_stack([np.interp(s_new, s_dense, P_dense[:, j]) for j in range(3)])
    out_attrs: Dict[str, np.ndarray] = {}
    if attrs:
        for key, val in attrs.items():
            out_attrs[key] = np.interp(s_new, s_dense, np.asarray(val, float))
    return P_new, out_attrs


def quadratic_baseline_arclength_mm(curvature_mm: float, x_max_mm: float, n_dense: int = 100001) -> float:
    x_max = float(x_max_mm)
    c = float(curvature_mm)
    if x_max <= 0:
        return 0.0
    if abs(c) < 1e-15:
        return x_max
    x = np.linspace(0.0, x_max, int(n_dense))
    dzdx = 2.0 * c * (x - 0.5 * x_max)
    return _trapz(np.sqrt(1.0 + dzdx**2), x)


def x_span_for_quadratic_baseline_arclength(
    curvature_mm: float,
    L0_mm: float,
    *,
    n_dense: int = 100001,
    xtol: float = 1e-10,
    max_iter: int = 80,
) -> float:
    L0 = float(L0_mm)
    c = float(curvature_mm)
    if not np.isfinite(L0) or L0 <= 0:
        raise ValueError(f"L0_mm must be positive finite; got {L0_mm!r}")
    if abs(c) < 1e-15:
        return L0
                                      
    lo, hi = max(L0 * 1e-6, 1e-8), L0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s_mid = quadratic_baseline_arclength_mm(c, mid, n_dense=n_dense)
        if abs(s_mid - L0) <= max(xtol, 1e-9 * L0) or (hi - lo) <= xtol:
            return float(mid)
        if s_mid < L0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def remaining_cycles_true_from_xi(
    xi_mm: np.ndarray,
    lambda_mm: float,
    *,
    removal_points: int = 10,
) -> float:
    xi = np.asarray(xi_mm, float)
    lam = float(lambda_mm)
    rem = int(removal_points)
    if rem < 0:
        raise ValueError(f"removal_points must be >= 0; got {removal_points!r}")
    if xi.ndim != 1 or len(xi) <= 2 * rem:
        return float("nan")
    if not (np.isfinite(lam) and lam > 0):
        return float("nan")
    return (float(xi[-rem - 1]) - float(xi[rem])) / lam


def trim_pad_cycles_for_target(
    *,
    lambda_mm: float,
    target_ds_mm: float,
    removal_points: int = 10,
    safety_cycles: float = 0.0,
) -> float:
    lam = float(lambda_mm)
    ds = float(target_ds_mm)
    if not (np.isfinite(lam) and lam > 0 and np.isfinite(ds) and ds > 0):
        return float(safety_cycles)
    return 2.0 * float(removal_points) * ds / lam + float(safety_cycles)


def n_cycles_input_for_target_remaining_after_trim(
    *,
    target_remaining: float,
    amplitude_mm: float,
    lambda_x_mm: float,
    phase: float,
    target_ds_mm: float,
    removal_points: int = 10,
    voxel_size_mm: float = 0.5,
    background: Optional[Dict[str, Any]] = None,
    diameter_mm: Optional[float] = None,
    out_of_plane_amplitude_mm: float = 0.0,
    dense_dx_mm: float = 0.01,
    max_iter: int = 40,
    atol: float = 1e-10,
) -> float:
    tgt = float(target_remaining)
    lam = float(lambda_x_mm)
    if not (np.isfinite(tgt) and tgt > 0 and np.isfinite(lam) and lam > 0):
        raise ValueError(
            f"target_remaining and lambda_x_mm must be positive finite; got {target_remaining!r}, {lambda_x_mm!r}"
        )
    rem = int(removal_points)
    ds = float(target_ds_mm)

    def remaining_of(n_in: float) -> float:
        syn = make_sinusoid_centerline(
            amplitude_mm=float(amplitude_mm),
            lambda_x_mm=lam,
            n_cycles=float(n_in),
            phase=float(phase),
            voxel_size_mm=float(voxel_size_mm),
            target_ds_mm=ds,
            dense_dx_mm=float(dense_dx_mm),
            background=background,
            out_of_plane_amplitude_mm=float(out_of_plane_amplitude_mm),
            diameter_mm=diameter_mm,
        )
        return remaining_cycles_true_from_xi(
            syn["xi_mm"],
            syn["lambda_baseline_true_mm"],
            removal_points=rem,
        )

    def _unmet(n_in: float, achieved: float) -> ValueError:
        return ValueError(
            "post-trim remaining cycles target not met "
            f"(discrete resampling; ds={ds:g} mm may be too coarse): "
            f"target={tgt:.12g}, achieved={achieved:.12g}, "
            f"|err|={abs(achieved - tgt):.3e}, atol={atol:g}, "
            f"n_cycles_input={n_in:.12g}, A={float(amplitude_mm):g}, "
            f"lambda={lam:g}, phase={float(phase):.6g}, removal={rem}"
        )

                         
    n_in = tgt + trim_pad_cycles_for_target(
        lambda_mm=lam, target_ds_mm=ds, removal_points=rem, safety_cycles=0.0
    )
    for _ in range(12):
        r = remaining_of(n_in)
        if not np.isfinite(r) or r <= 0:
            n_in *= 1.25
            continue
        if abs(r - tgt) <= atol:
            return float(n_in)
        n_in *= tgt / r

    lo = max(n_in * 0.9, tgt * 0.5, 1e-6)
    hi = max(n_in * 1.1, tgt + 1.0)
    r_lo, r_hi = remaining_of(lo), remaining_of(hi)
    expand = 0
    while np.isfinite(r_lo) and r_lo > tgt and expand < 20:
        lo *= 0.5
        r_lo = remaining_of(lo)
        expand += 1
    expand = 0
    while (not np.isfinite(r_hi) or r_hi < tgt) and expand < 20:
        hi *= 1.5
        r_hi = remaining_of(hi)
        expand += 1

    best_n = float(n_in)
    best_r = remaining_of(best_n)
    best_err = abs(best_r - tgt) if np.isfinite(best_r) else float("inf")
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        r_mid = remaining_of(mid)
        if not np.isfinite(r_mid):
            lo = mid
            continue
        err = abs(r_mid - tgt)
        if err < best_err:
            best_err, best_n, best_r = err, float(mid), float(r_mid)
        if err <= atol:
            return float(mid)
        if r_mid < tgt:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * max(1.0, tgt):
            break

                                  
    achieved = remaining_of(best_n)
    if not np.isfinite(achieved) or abs(achieved - tgt) > atol:
        raise _unmet(best_n, achieved if np.isfinite(achieved) else float("nan"))
    return float(best_n)


def _wave_phase_with_slow_drift(
    xi_mm: np.ndarray,
    *,
    lambda0_mm: float,
    phase0_rad: float,
    L0_mm: float,
    chirp_alpha: float,
    phase_quad_cycles: float,
) -> np.ndarray:
    xi = np.asarray(xi_mm, float)
    lam0 = float(lambda0_mm)
    L0 = float(L0_mm)
    alpha = float(chirp_alpha)
    beta = float(phase_quad_cycles)
    if not (np.isfinite(lam0) and lam0 > 0 and np.isfinite(L0) and L0 > 0):
        raise ValueError("lambda0_mm and L0_mm must be positive finite")
    if abs(alpha) >= 1.0:
        raise ValueError(f"chirp_alpha must satisfy |α|<1; got {alpha!r}")
    u = xi / L0
    # ∫ ds/λ with λ=λ0(a + b ξ), a=1-α, b=2α/L0
    if abs(alpha) < 1e-15:
        integ = xi / lam0
    else:
        a = 1.0 - alpha
        b = 2.0 * alpha / L0
        # (1/(λ0 b)) log((a+bξ)/a)
        integ = np.log((a + b * xi) / a) / (lam0 * b)
    psi = 2.0 * np.pi * integ + float(phase0_rad) + 2.0 * np.pi * beta * (u ** 2)
    return psi


def make_sinusoid_centerline(
    *,
    amplitude_mm: float,
    lambda_x_mm: float,
    n_cycles: float,
    phase: float,
    voxel_size_mm: float = 0.5,
    target_ds_mm: float = 0.5,
    dense_dx_mm: float = 0.01,
    background: Optional[Dict[str, Any]] = None,
    out_of_plane_amplitude_mm: float = 0.0,
    noise_sd_mm: float = 0.0,
    rotation: Optional[np.ndarray] = None,
    translation_mm: Optional[np.ndarray] = None,
    seed: int = 0,
    diameter_mm: Optional[float] = None,
    chirp_alpha: float = 0.0,
    phase_quad_cycles: float = 0.0,
) -> Dict[str, Any]:
    Ay = float(amplitude_mm)
    Az = float(out_of_plane_amplitude_mm)
    lam = float(lambda_x_mm)
    n_cyc = float(n_cycles)
    phi = float(phase)
    chirp_a = float(chirp_alpha)
    phase_b = float(phase_quad_cycles)
    if not (np.isfinite(lam) and lam > 0):
        raise ValueError(f"lambda_x_mm must be positive finite; got {lambda_x_mm!r}")
    if not (np.isfinite(n_cyc) and n_cyc > 0):
        raise ValueError(f"n_cycles must be positive finite; got {n_cycles!r}")

    L0 = n_cyc * lam                       
    c_bg = 0.0
    if background:
        if background.get("type") != "quadratic":
            raise ValueError(f"unknown background type {background!r}")
        c_bg = float(background.get("curvature_mm", 0.0))

    x_max = x_span_for_quadratic_baseline_arclength(c_bg, L0) if abs(c_bg) > 0 else L0
    n_dense = max(5000, int(max(x_max, L0) / float(dense_dx_mm)) + 1)
    x = np.linspace(0.0, x_max, n_dense)
    z_b = c_bg * (x - 0.5 * x_max) ** 2
    B_mm = np.column_stack([x, np.zeros_like(x), z_b])
    xi_mm, _ = arclength(B_mm)
                                 
    if float(xi_mm[-1]) > 0:
        xi_mm = xi_mm * (L0 / float(xi_mm[-1]))

    wave_phase_dense = _wave_phase_with_slow_drift(
        xi_mm,
        lambda0_mm=lam,
        phase0_rad=phi,
        L0_mm=L0,
        chirp_alpha=chirp_a,
        phase_quad_cycles=phase_b,
    )
    wave = np.sin(wave_phase_dense)
                                                   
    P_mm = B_mm.copy()
    P_mm[:, 1] = P_mm[:, 1] + Ay * wave
    if Az:
        P_mm[:, 2] = P_mm[:, 2] + Az * wave

    if rotation is not None:
        R = np.asarray(rotation, float)
        P_mm = P_mm @ R.T
        B_mm = B_mm @ R.T
    if translation_mm is not None:
        t = np.asarray(translation_mm, float)[None, :]
        P_mm = P_mm + t
        B_mm = B_mm + t

    ds_vox = float(target_ds_mm) / float(voxel_size_mm)
    P_vox, attrs = resample_uniform_arclength_with_attrs(
        P_mm / voxel_size_mm,
        ds_vox,
        attrs={"xi_mm": xi_mm, "wave_phase_rad": wave_phase_dense},
    )
                                                           
    B_vox, _ = resample_uniform_arclength_with_attrs(B_mm / voxel_size_mm, ds_vox)
    if noise_sd_mm:
        rng = np.random.default_rng(seed)
        P_vox = P_vox + rng.normal(0.0, noise_sd_mm / voxel_size_mm, size=P_vox.shape)
    s_vox, _ = arclength(P_vox)
    xi_on_obs_mm = attrs["xi_mm"]
    wave_phase_rad = attrs["wave_phase_rad"]
    radius_vox = None
    if diameter_mm is not None:
        radius_vox = np.full(len(P_vox), 0.5 * float(diameter_mm) / voxel_size_mm, dtype=float)

    A_comb = combined_amplitude_mm(Ay, Az)
    periodic = abs(Ay) > 0.0 or abs(Az) > 0.0
    if periodic:
        lambda_s_poly, lambda_s_poly_status = polyline_arclength_first_complete_cycle_mm(
            s_vox, xi_on_obs_mm, lam, float(voxel_size_mm)
        )
    else:
        lambda_s_poly, lambda_s_poly_status = float("nan"), "non_periodic"

    basis = "known_baseline_arclength_xi"
    return {
        "P_vox": P_vox,
        "B_vox": B_vox,
        "B_vox_role": "independent_baseline_polyline_for_display",
        "B_vox_pointwise_with_P": False,
        "s_vox": s_vox,
        "xi_mm": xi_on_obs_mm,
        "wave_phase_rad": wave_phase_rad,
        "radius_vox": radius_vox,
        "diameter_mm": (float(diameter_mm) if diameter_mm is not None else float("nan")),
        "lambda_x_over_diameter": (lam / float(diameter_mm) if diameter_mm and periodic else float("nan")),
        "lambda_s_polyline_over_diameter": (
            lambda_s_poly / float(diameter_mm)
            if diameter_mm and periodic and np.isfinite(lambda_s_poly)
            else float("nan")
        ),
        "T_numeric": numeric_tangent(P_vox, s_vox),
        "A_y_true_mm": Ay,
        "A_z_true_mm": Az,
        "A_comb_true_mm": A_comb,
        "A_true_mm": A_comb,
        "lambda_x_input_mm": lam,
                                                                     
        "lambda_x_true_mm": lam if periodic else float("nan"),
        "lambda_baseline_true_mm": lam if periodic else float("nan"),
        "lambda_baseline_true_basis": basis if periodic else "",
        "lambda_phase_true_mm": lam if periodic else float("nan"),
        "lambda_phase_true_basis": basis if periodic else "",
        "lambda_s_polyline_mm": float(lambda_s_poly) if periodic else float("nan"),
        "lambda_s_polyline_status": str(lambda_s_poly_status),
        "L0_known_baseline_mm": float(L0),
        "x_max_mm": float(x_max),
        "periodic_component_present": bool(periodic),
        "n_cycles_input": n_cyc,
        "phase_rad": phi,
        "chirp_alpha": float(chirp_a),
        "phase_quad_cycles": float(phase_b),
        "phase_coordinate_generation": "known_baseline_arclength_xi",
        "target_ds_mm": float(target_ds_mm),
        "actual_ds_mm": float(np.median(np.diff(s_vox))) * voxel_size_mm if len(s_vox) > 1 else float("nan"),
        "L_input_mm": float(s_vox[-1]) * voxel_size_mm,
        "voxel_size_mm": float(voxel_size_mm),
        "background": background,
        "out_of_plane_amplitude_mm": Az,
        "noise_sd_mm": float(noise_sd_mm),
    }


def random_rotation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    angle = rng.uniform(0.1, 3.0)
    return rotation_matrix_from_axis_angle(axis, angle)


                                     
VALIDATION_DIAMETERS_MM = [1.4]

                                              
                                                                
BASE_PLANE_CONDITIONS = dict(
    A_mm=[0.25, 0.5, 1.0, 2.0],
    lambda_x_mm=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    phase=[0.0, np.pi / 4, np.pi / 2],
    n_cycles_target_after_trim=[3.0],
    target_ds_mm=[0.10],
    diameter_mm=list(VALIDATION_DIAMETERS_MM),
    variant="pure_sinusoid",
    removal_points=10,
)
                                                       
                                                 
                                                                                    
CURVED_BASELINE_STRESS = dict(
    A_mm=[0.5, 1.0, 2.0],
    lambda_x_mm=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    phase=[0.0, np.pi / 4, np.pi / 2],
    n_cycles_target_after_trim=[3.0],
    target_ds_mm=[0.10],
    diameter_mm=list(VALIDATION_DIAMETERS_MM),
    background={"type": "quadratic", "curvature_mm": 0.005},
    variant="curved_baseline_sinusoid",
    removal_points=10,
)
                                     
SHORT_COVERAGE_STRESS = dict(
    A_mm=[0.5, 1.0],
    lambda_x_mm=[10.0, 20.0, 40.0],
    phase=[0.0, np.pi / 2],
    n_cycles_target_after_trim=[1.5, 2.0],
    target_ds_mm=[0.10],
    diameter_mm=list(VALIDATION_DIAMETERS_MM),
    variant="short_coverage",
    removal_points=10,
)
                            
CURVED_SHORT_STRESS = dict(
    A_mm=[1.0],
    lambda_x_mm=[10.0, 20.0, 40.0],
    phase=[0.0],
    n_cycles_target_after_trim=[1.5, 2.0],
    target_ds_mm=[0.10],
    diameter_mm=list(VALIDATION_DIAMETERS_MM),
    background={"type": "quadratic", "curvature_mm": 0.005},
    variant="curved_short_coverage",
    removal_points=10,
)


def _as_float_list(val: Any, *, default: Optional[List[float]] = None) -> List[float]:
    if val is None:
        return list(default or VALIDATION_DIAMETERS_MM)
    if isinstance(val, (list, tuple, np.ndarray)):
        return [float(x) for x in val]
    return [float(val)]


def expand_grid_conditions(
    spec: Dict[str, Any],
    *,
    diameter_mm: Optional[Any] = None,
    target_ds_mm: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    import itertools

    if diameter_mm is not None:
        D_list = _as_float_list(diameter_mm)
    else:
        D_list = _as_float_list(spec.get("diameter_mm"), default=VALIDATION_DIAMETERS_MM)
    if target_ds_mm is not None:
        ds_list = _as_float_list(target_ds_mm)
    else:
        ds_list = _as_float_list(spec.get("target_ds_mm"), default=[0.10])
    if "backgrounds" in spec and spec.get("backgrounds") is not None:
        bg_list: List[Optional[Dict[str, Any]]] = [
            (dict(bg) if bg is not None else None) for bg in list(spec["backgrounds"])
        ]
    elif "background" in spec:
        bg = spec.get("background")
        bg_list = [dict(bg) if bg is not None else None]
    else:
        bg_list = [None]
    variant = str(spec.get("variant", "synthetic"))
    removal = int(spec.get("removal_points", 10))
    targets = spec.get("n_cycles_target_after_trim")
    n_list = list(spec["n_cycles"]) if targets is None else list(targets)
    out: List[Dict[str, Any]] = []
    for D, A, lam, ph, nc_or_tgt, ds, bg in itertools.product(
        D_list,
        spec["A_mm"],
        spec["lambda_x_mm"],
        spec["phase"],
        n_list,
        ds_list,
        bg_list,
    ):
        if targets is None:
            n_cycles = float(nc_or_tgt)
            n_target = float("nan")
        else:
            n_target = float(nc_or_tgt)
            n_cycles = n_cycles_input_for_target_remaining_after_trim(
                target_remaining=n_target,
                amplitude_mm=float(A),
                lambda_x_mm=float(lam),
                phase=float(ph),
                target_ds_mm=float(ds),
                removal_points=removal,
                background=bg,
                diameter_mm=float(D),
            )
        row: Dict[str, Any] = dict(
            amplitude_mm=float(A),
            lambda_x_mm=float(lam),
            n_cycles=n_cycles,
            phase=float(ph),
            target_ds_mm=float(ds),
            diameter_mm=float(D),
            variant=variant,
        )
        if np.isfinite(n_target):
            row["n_cycles_target_after_trim"] = n_target
        if bg is not None:
            row["background"] = dict(bg)
        out.append(row)
    return out


def stress_validation_conditions(
    *,
    diameter_mm: Optional[Any] = None,
    target_ds_mm: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    return (
        expand_grid_conditions(
            CURVED_BASELINE_STRESS, diameter_mm=diameter_mm, target_ds_mm=target_ds_mm
        )
        + expand_grid_conditions(
            SHORT_COVERAGE_STRESS, diameter_mm=diameter_mm, target_ds_mm=target_ds_mm
        )
        + expand_grid_conditions(
            CURVED_SHORT_STRESS, diameter_mm=diameter_mm, target_ds_mm=target_ds_mm
        )
    )
