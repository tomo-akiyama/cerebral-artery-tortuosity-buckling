#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend frames."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
# Prefer this package directory on sys.path (same basename modules must not collide).
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)

import sin_trend_fitting as _legacy                                       

arclength = _legacy.arclength
trime_ends = _legacy.trime_ends
poly_trend = _legacy.poly_trend
lateral_components = _legacy.lateral_components
legacy_bishop_frame = _legacy.bishop_frame

BASELINE_METHOD = "quadratic_poly_in_normalized_observed_arclength_full_interval"
CORRESPONDENCE_METHOD = "same_observed_arclength_index_P_minus_P0"
                                        
                                             
FITTING_COORDINATE = "baseline_arclength"
FRAME_TRANSPORT_METHOD = "minimal_rotation_rodrigues_between_adjacent_unit_tangents"
FRAME_TRANSPORT_METHOD_LEGACY = "legacy_projection_update_bishop_frame"
TANGENT_SOURCE_BASELINE = "analytic_derivative_of_quadratic_baseline"
TANGENT_SOURCE_FRENET = "vmtk_FrenetTangent_pointdata"
TANGENT_SOURCE_NUMERIC = "central_difference_of_observed_points"

                                         
                                                      
TANGENT_NORM_REL_TOL = 1e-8
                                                         
ANTIPARALLEL_COS_TOL = 1e-10
                                                         
PARALLEL_CROSS_TOL = 1e-14


# ---------------------------------------------------------------------------
             
# ---------------------------------------------------------------------------
def quadratic_baseline(P_3darr: np.ndarray, s_arr: np.ndarray, degree: int = 2) -> Dict[str, Any]:
    P = np.asarray(P_3darr, float)
    s = np.asarray(s_arr, float)
    out: Dict[str, Any] = {
        "baseline_method": BASELINE_METHOD,
        "degree": int(degree),
        "tangent_rel_tol": float(TANGENT_NORM_REL_TOL),
        "status": "ok",
        "status_reason": "",
    }
    if P.ndim != 2 or P.shape[1] != 3 or len(P) != len(s):
        out.update(status="nonfinite", status_reason=f"invalid shapes P={P.shape}, s={s.shape}")
        return out
    if not (np.isfinite(P).all() and np.isfinite(s).all()):
        out.update(status="nonfinite", status_reason="non-finite P or s")
        return out
    if len(s) < degree + 2:
        out.update(status="degenerate_length", status_reason=f"n={len(s)} < degree+2")
        return out
    s_first, s_last = float(s[0]), float(s[-1])
    s_range = s_last - s_first
    if not np.isfinite(s_range) or s_range <= 0.0:
        out.update(status="degenerate_length", status_reason="non-positive arc-length range")
        return out

                                      
    P0, info = poly_trend(P_3darr=P, s_arr=s, degree=degree, return_residual=False)
    coef = np.asarray(info["coef"], float)                        
    q = (s - s_first) / max(s_range, 1e-12)                                                        

                          
                                                                       
    if degree >= 1:
        powers = np.arange(degree, 0, -1, dtype=float)  # degree, ..., 1
        dcoef = coef[:-1] * powers[:, None]  # (degree, 3)
        dP0_dq = np.polyval(dcoef, q[:, None])  # (N, 3)
    else:
        dP0_dq = np.zeros_like(P)
    dP0_ds = dP0_dq / s_range
    dnorm = np.linalg.norm(dP0_ds, axis=1)

                                                       
    if degree == 2:
        a = 2.0 * coef[0]  # (3,)
        b = coef[1]
        aa = float(a @ a)
        q_star = float(-(a @ b) / aa) if aa > 0 else 0.0
        q_star = min(1.0, max(0.0, q_star))
        dmin_cont = float(np.linalg.norm(a * q_star + b) / s_range)
    else:
        q_star = float("nan")
        dmin_cont = float(np.min(dnorm)) if len(dnorm) else float("nan")

    out.update(
        P0=P0,
        coef=coef,
        q=q,
        s_first=s_first,
        s_last=s_last,
        s_range=float(s_range),
        dP0_ds=dP0_ds,
        dnorm=dnorm,
        dnorm_min=float(np.min(dnorm)),
        dnorm_median=float(np.median(dnorm)),
        dnorm_max=float(np.max(dnorm)),
        dnorm_min_continuous=dmin_cont,
        q_at_min_continuous=q_star,
    )
    if not np.isfinite(P0).all() or not np.isfinite(dP0_ds).all():
        out.update(status="nonfinite", status_reason="non-finite baseline or derivative")
        return out
    scale = out["dnorm_median"]
    tol = TANGENT_NORM_REL_TOL * scale
    if scale <= 0.0 or out["dnorm_min"] <= tol:
        out.update(
            status="zero_tangent_at_sample",
            status_reason=(
                f"baseline derivative norm {out['dnorm_min']:.3e} <= rel tol {TANGENT_NORM_REL_TOL:g} * median {scale:.3e}"
            ),
        )
        return out
    if dmin_cont <= tol:
        out.update(
            status="zero_tangent_between_samples",
            status_reason=(
                f"baseline derivative vanishes between samples (q*={q_star:.4f}, min norm {dmin_cont:.3e})"
            ),
        )
    return out


def analytic_baseline_tangent(baseline: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str, str]:
    st = str(baseline.get("status", "nonfinite"))
    if st not in {"ok"}:
        return None, st, str(baseline.get("status_reason", ""))
    d = np.asarray(baseline["dP0_ds"], float)
    n = np.asarray(baseline["dnorm"], float)
    T0 = d / n[:, None]
    if not np.isfinite(T0).all():
        return None, "nonfinite", "non-finite unit tangent"
    return T0, "ok", ""


# ---------------------------------------------------------------------------
                  
# ---------------------------------------------------------------------------
def _initial_lateral_axis(T0: np.ndarray) -> np.ndarray:
    axes = np.eye(3)
    dots = np.abs(axes @ T0)
    e = axes[int(np.argmin(dots))]
    u = e - float(e @ T0) * T0
    nu = float(np.linalg.norm(u))
    if nu <= 0.0 or not np.isfinite(nu):
        raise ValueError("cannot construct initial lateral axis")
    return u / nu


def rodrigues_rotation_between(t_prev: np.ndarray, t_next: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    v = np.cross(t_prev, t_next)
    c = float(t_prev @ t_next)
    if c <= -1.0 + ANTIPARALLEL_COS_TOL:
        return None, "antiparallel"
    if float(np.linalg.norm(v)) < PARALLEL_CROSS_TOL:
        return np.eye(3), "parallel"
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    R = np.eye(3) + K + (K @ K) / (1.0 + c)
    return R, "rotated"


def minimal_rotation_transport(
    T_3darr: np.ndarray,
    *,
    initial_U: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    T = np.asarray(T_3darr, float)
    res: Dict[str, Any] = {
        "frame_transport_method": FRAME_TRANSPORT_METHOD,
        "status": "ok",
        "status_reason": "",
        "T": None,
        "U": None,
        "V": None,
    }
    if T.ndim != 2 or T.shape[1] != 3 or len(T) < 1:
        res.update(status="invalid_input", status_reason=f"T shape {T.shape}")
        return res
    if not np.isfinite(T).all():
        res.update(status="nonfinite_tangent", status_reason="non-finite tangent")
        return res
    norms = np.linalg.norm(T, axis=1)
    if np.any(norms <= 0.0):
        res.update(status="zero_tangent", status_reason="zero-norm tangent present; not filled")
        return res
    Tn = T / norms[:, None]
    N = len(Tn)
    U = np.zeros_like(Tn)
    V = np.zeros_like(Tn)
    try:
        if initial_U is None:
            U[0] = _initial_lateral_axis(Tn[0])
        else:
            u0 = np.asarray(initial_U, float)
            u0 = u0 - float(u0 @ Tn[0]) * Tn[0]
            nu = float(np.linalg.norm(u0))
            if nu <= 0.0:
                raise ValueError("initial_U parallel to T[0]")
            U[0] = u0 / nu
    except ValueError as exc:
        res.update(status="initial_axis_failed", status_reason=str(exc))
        return res
    V[0] = np.cross(Tn[0], U[0])

    adj_angles = np.zeros(max(N - 1, 0))
    for i in range(1, N):
        R, kind = rodrigues_rotation_between(Tn[i - 1], Tn[i])
        if R is None:
            res.update(
                status="antiparallel_ambiguous",
                status_reason=f"adjacent tangents antiparallel at index {i}; rotation axis undefined",
                failed_index=int(i),
            )
            return res
        u = R @ U[i - 1]
                     
        u = u - float(u @ Tn[i]) * Tn[i]
        nu = float(np.linalg.norm(u))
        if nu <= 0.0 or not np.isfinite(nu):
            res.update(status="transport_failed", status_reason=f"degenerate U at index {i}")
            return res
        U[i] = u / nu
        V[i] = np.cross(Tn[i], U[i])
        adj_angles[i - 1] = float(np.degrees(np.arccos(np.clip(Tn[i - 1] @ Tn[i], -1.0, 1.0))))

    chk = frame_orthonormality_check(Tn, U, V)
    res.update(
        T=Tn,
        U=U,
        V=V,
        input_norm_min=float(norms.min()),
        input_norm_max=float(norms.max()),
        adjacent_angle_max_deg=float(adj_angles.max()) if len(adj_angles) else 0.0,
        adjacent_angle_median_deg=float(np.median(adj_angles)) if len(adj_angles) else 0.0,
        **chk,
    )
    return res


def frame_orthonormality_check(T: np.ndarray, U: np.ndarray, V: np.ndarray) -> Dict[str, float]:
    T = np.asarray(T, float)
    U = np.asarray(U, float)
    V = np.asarray(V, float)
    norm_err = np.max(
        np.abs(
            np.concatenate(
                [np.linalg.norm(T, axis=1) - 1, np.linalg.norm(U, axis=1) - 1, np.linalg.norm(V, axis=1) - 1]
            )
        )
    )
    dot_err = np.max(
        np.abs(
            np.concatenate(
                [np.einsum("ij,ij->i", T, U), np.einsum("ij,ij->i", T, V), np.einsum("ij,ij->i", U, V)]
            )
        )
    )
    hand_err = float(np.max(np.linalg.norm(np.cross(T, U) - V, axis=1)))
    return {
        "frame_norm_err_max": float(norm_err),
        "frame_dot_err_max": float(dot_err),
        "frame_handedness_err_max": hand_err,
        "frame_orth_err_max": float(max(norm_err, dot_err, hand_err)),
    }


def legacy_projection_frame(T_3darr: np.ndarray) -> Dict[str, Any]:
    T = np.asarray(T_3darr, float)
    res: Dict[str, Any] = {"frame_transport_method": FRAME_TRANSPORT_METHOD_LEGACY, "status": "ok", "status_reason": ""}
    if T.ndim != 2 or T.shape[1] != 3 or not np.isfinite(T).all():
        res.update(status="invalid_input", status_reason="invalid tangent for legacy frame", T=None, U=None, V=None)
        return res
    Tl, Ul, Vl = legacy_bishop_frame(T)
    res.update(T=Tl, U=Ul, V=Vl, **frame_orthonormality_check(Tl, Ul, Vl))
    return res


def numeric_tangent(P_3darr: np.ndarray, s_arr: np.ndarray) -> np.ndarray:
    P = np.asarray(P_3darr, float)
    s = np.asarray(s_arr, float)
    d = np.empty_like(P)
    d[1:-1] = (P[2:] - P[:-2]) / (s[2:] - s[:-2])[:, None]
    d[0] = (P[1] - P[0]) / (s[1] - s[0])
    d[-1] = (P[-1] - P[-2]) / (s[-1] - s[-2])
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.where(n > 0, n, np.nan)


# ---------------------------------------------------------------------------
           
# ---------------------------------------------------------------------------
def lateral_components_signed(
    P_3darr: np.ndarray, P0_3darr: np.ndarray, T: np.ndarray, U: np.ndarray, V: np.ndarray
) -> Dict[str, Any]:
    uU, uV, Delta, Delta_par, d_perp, DU, DV = lateral_components(P_3darr, P0_3darr, T, U, V)
    recon = DU + DV
    recon_err = np.linalg.norm(d_perp - recon, axis=1)
    perp_dot_T = np.einsum("ij,ij->i", d_perp, T)
    return {
        "uU": uU,
        "uV": uV,
        "Delta": Delta,
        "Delta_parallel": Delta_par,
        "d_perp": d_perp,
        "recon_err_max": float(np.max(recon_err)) if len(recon_err) else 0.0,
        "perp_dot_T_max": float(np.max(np.abs(perp_dot_T))) if len(perp_dot_T) else 0.0,
        "deltaP_norm": np.linalg.norm(Delta, axis=1),
        "ss_parallel": float(np.sum(Delta_par * Delta_par)),
        "ss_perp": float(np.sum(d_perp * d_perp)),
    }


def tangent_angle_deg(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    c = np.einsum("ij,ij->i", np.asarray(T_a, float), np.asarray(T_b, float))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


# ---------------------------------------------------------------------------
                                                      
# ---------------------------------------------------------------------------
def trim_bundle(arrays: Dict[str, np.ndarray], removal: int, s_full: np.ndarray) -> Dict[str, Any]:
    removal = int(removal)
    n = len(s_full)
    if removal < 0:
        raise ValueError("removal must be >= 0")
    if n - 2 * removal < 1:
        return {"ok": False, "reason": f"n={n} too short for removal={removal}", "index": np.arange(0)}
    idx = np.arange(removal, n - removal) if removal > 0 else np.arange(n)
    out: Dict[str, Any] = {"ok": True, "reason": "", "index": idx, "removal": removal}
    for k, a in arrays.items():
        if a is None:
            out[k] = None
            continue
        a = np.asarray(a)
        if len(a) != n:
            raise ValueError(f"array {k!r} length {len(a)} != n {n}")
        out[k] = a[idx]
    s_full = np.asarray(s_full, float)
    s_fit = s_full[idx] - s_full[idx[0]]
    out["s_fit"] = s_fit
    out["s_full_trim"] = s_full[idx]
    out["removed_mm_start_in_s_unit"] = float(s_full[idx[0]] - s_full[0])
    out["removed_end_in_s_unit"] = float(s_full[-1] - s_full[idx[-1]])
    return out


def rotation_matrix_from_axis_angle(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
