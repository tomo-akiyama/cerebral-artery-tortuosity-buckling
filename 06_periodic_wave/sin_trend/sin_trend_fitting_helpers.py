#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend fitting helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

                                                     
                                         


def _legacy():
    import sin_trend_fitting as _stf

    return _stf


def arclength(*args, **kwargs):
    return _legacy().arclength(*args, **kwargs)


def trime_ends(*args, **kwargs):
    return _legacy().trime_ends(*args, **kwargs)


def normalize(*args, **kwargs):
    return _legacy().normalize(*args, **kwargs)


def bishop_frame(*args, **kwargs):
    return _legacy().bishop_frame(*args, **kwargs)


def lateral_components(*args, **kwargs):
    return _legacy().lateral_components(*args, **kwargs)


def poly_trend(*args, **kwargs):
    return _legacy().poly_trend(*args, **kwargs)


def clip_mesh_to_coords(*args, **kwargs):
    return _legacy().clip_mesh_to_coords(*args, **kwargs)


def analyze_surface_constraint_coherence(*args, **kwargs):
    return _legacy().analyze_surface_constraint_coherence(*args, **kwargs)


def design_matrix(s_arr, k_arr, trend_degree=1):
    s = np.asarray(s_arr, float)
    s0 = s - s.mean()
    X = [np.cos(k_arr * s), np.sin(k_arr * s), np.ones_like(s)]
    if trend_degree >= 1:
        X.append(s0)
    if trend_degree >= 2:
        X.append(s0**2)
    return np.column_stack(X)


def design_matrix_trend_only(s_arr, trend_degree=1):
    s = np.asarray(s_arr, float)
    s0 = s - s.mean()
    X = [np.ones_like(s)]
    if trend_degree >= 1:
        X.append(s0)
    if trend_degree >= 2:
        X.append(s0**2)
    return np.column_stack(X)


def _lstsq_rcond(Xw: np.ndarray) -> float:
    """LAPACK-like default: max(m,n)*eps*max(s)."""
    m, n = Xw.shape
    # s estimated later; provisional rcond uses eps scale
    return float(max(m, n) * np.finfo(float).eps)


def solve_linear(X, y, weights=None, return_info=False):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if weights is not None:
        w = np.asarray(weights, float)
        if w.shape != y.shape:
            raise ValueError("weights shape mismatch")
        if np.any(~np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be finite and non-negative")
        W = np.sqrt(w)[:, None]
        Xw = X * W
        yw = y * W.squeeze()
    else:
        w = None
        Xw = X
        yw = y

    rcond = _lstsq_rcond(Xw)
    coef, residuals, rank, svals = np.linalg.lstsq(Xw, yw, rcond=rcond)
    yhat = X @ coef
    resid = y - yhat
    if w is None:
        sse = float(np.sum(resid**2))
    else:
        sse = float(np.sum(w * resid**2))
    info = dict(
        sse=sse,
        n=int(X.shape[0]),
        p=int(X.shape[1]),
        rank=int(rank),
        singular_values=np.asarray(svals, float),
        rcond=float(rcond),
        full_rank=bool(rank >= X.shape[1]),
    )
    if return_info:
        return coef, yhat, info
    return coef, yhat


def residual_norm_numerically_zero(
    y: np.ndarray,
    yhat: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> bool:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    n = float(y.size)
    if n <= 0:
        return True
    if weights is None:
        r = y - yhat
        rnorm = float(np.linalg.norm(r))
        ynorm = float(np.linalg.norm(y))
        yhnorm = float(np.linalg.norm(yhat))
    else:
        w = np.asarray(weights, float)
        sw = np.sqrt(w)
        rnorm = float(np.linalg.norm(sw * (y - yhat)))
        ynorm = float(np.linalg.norm(sw * y))
        yhnorm = float(np.linalg.norm(sw * yhat))
    scale = max(ynorm, yhnorm)
    if scale == 0.0:
        return True
                                            
    tol = 10.0 * np.sqrt(n) * np.finfo(float).eps * scale
    return rnorm <= tol


def make_k_grid(L, N_cyc=0.5, N_pp=20, ds_eff=1, q_fine=4, verbose=True):
    L = float(L)
    k_max = 2.0 * np.pi / (N_pp * ds_eff)
    k_min = 2 * np.pi * N_cyc / L
    if k_max <= k_min:
        raise ValueError("Invalid band: k_max must be > k_min.")
    dk = np.pi / (q_fine * L)
    n = int(np.floor((k_max - k_min) / dk)) + 1
    ks = k_min + dk * np.arange(n, dtype=float)
    if ks[-1] < k_max - 1e-12:
        ks = np.append(ks, k_max)
    if verbose:
        print(
            f"探索波数 grid: k_min={k_min:.4f}, k_max={k_max:.4f}, dk={dk:.4f}, 点数={len(ks)}"
        )
    return ks, k_min, k_max, dk

ANALYSIS_VERSION = "sin_trend_11"
ANALYSIS_REVISION = "patch1"
R2_DEFINITION = "two_component_centered_sst"
K_SELECTION_METHOD = "argmin_sse_full_over_search_grid"
BASELINE_METHOD = "quadratic_poly_on_observed_centerline"
FRAME_REFERENCE = "bishop_on_observed_centerline"
FITTING_COORDINATE = "observed_arclength_s"
                                             
SELECTABLE_FIT_STATUSES = frozenset({"ok"})

                                   
EXACT_TIE_ATOL = 0.0
EXACT_TIE_RTOL = 0.0


def _finite_arrays(*arrays: np.ndarray) -> bool:
    return all(np.isfinite(a).all() for a in arrays)


def sse_sst_two_component(
    uU: np.ndarray,
    yU: np.ndarray,
    uV: np.ndarray,
    yV: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    uU = np.asarray(uU, float)
    uV = np.asarray(uV, float)
    yU = np.asarray(yU, float)
    yV = np.asarray(yV, float)

    if weights is None:
        sse = float(np.sum((uU - yU) ** 2) + np.sum((uV - yV) ** 2))
        sst = float(np.sum((uU - np.mean(uU)) ** 2) + np.sum((uV - np.mean(uV)) ** 2))
        return sse, sst

    w = np.asarray(weights, float)
    if w.shape != uU.shape:
        raise ValueError("weights must have the same shape as each component array")
    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0.0:
        return float("nan"), float("nan")
    mu_u = float(np.sum(w * uU) / wsum)
    mu_v = float(np.sum(w * uV) / wsum)
    sse = float(np.sum(w * (uU - yU) ** 2) + np.sum(w * (uV - yV) ** 2))
    sst = float(np.sum(w * (uU - mu_u) ** 2) + np.sum(w * (uV - mu_v) ** 2))
    return sse, sst


def _validate_weights(weights, n: int) -> Optional[str]:
    if weights is None:
        return None
    w = np.asarray(weights, float)
    if w.shape != (n,):
        return "invalid_weights: shape mismatch"
    if np.any(~np.isfinite(w)):
        return "invalid_weights: non-finite"
    if np.any(w < 0.0):
        return "invalid_weights: negative (not clipped)"
    if int(np.sum(w > 0.0)) < 2:
        return "invalid_weights: too few positive weights"
    return None


def fit_sin_trend_two_component(
    uU_arr,
    uV_arr,
    s_arr,
    k_arr,
    weights=None,
    trend_degree: int = 1,
) -> Dict[str, Any]:
    uU = np.asarray(uU_arr, float)
    uV = np.asarray(uV_arr, float)
    s = np.asarray(s_arr, float)
    k = float(k_arr)

    n_need = max(4, 2 + max(0, int(trend_degree)) + 2)
    if len(uU) != len(uV) or len(uU) != len(s):
        return _failed_fit(k, "length_mismatch", "uU/uV/s length mismatch")
    if len(uU) < n_need:
        return _failed_fit(k, "insufficient_points", "too few points for sin+trend model")
    if not _finite_arrays(uU, uV, s) or not np.isfinite(k):
        return _failed_fit(k, "nonfinite_input", "non-finite values in inputs")
    if float(s[-1] - s[0]) <= 0.0:
        return _failed_fit(k, "nonpositive_length", "non-positive arc-length range")
    if k <= 0.0:
        return _failed_fit(k, "nonpositive_k", "k must be positive")
    w_err = _validate_weights(weights, len(uU))
    if w_err is not None:
        return _failed_fit(k, "invalid_weights", w_err)

    try:
        X = design_matrix(s, k, trend_degree=trend_degree)
        X_tr = design_matrix_trend_only(s, trend_degree=trend_degree)
        cU, yU, info_u = solve_linear(X, uU, weights=weights, return_info=True)
        cV, yV, info_v = solve_linear(X, uV, weights=weights, return_info=True)
        _, yU_tr, info_ut = solve_linear(X_tr, uU, weights=weights, return_info=True)
        _, yV_tr, info_vt = solve_linear(X_tr, uV, weights=weights, return_info=True)
    except ValueError as exc:
        msg = str(exc)
        st = "invalid_weights" if "weight" in msg else "fit_failed"
        return _failed_fit(k, st, msg)
    except Exception as exc:  # noqa: BLE001
        return _failed_fit(k, "fit_failed", f"linear solve failed: {exc}")

    rank_full = min(int(info_u["rank"]), int(info_v["rank"]))
    rank_tr = min(int(info_ut["rank"]), int(info_vt["rank"]))
    need_full = int(X.shape[1])
    need_tr = int(X_tr.shape[1])
    if (not info_u["full_rank"]) or (not info_v["full_rank"]):
        return _failed_fit(
            k,
            "rank_deficient",
            f"full model rank-deficient (rank={rank_full}, p={need_full})",
            extra=dict(
                rank_full=rank_full,
                rank_tr=rank_tr,
                rcond=float(info_u["rcond"]),
                singular_values_full_U=info_u["singular_values"],
            ),
        )
    if (not info_ut["full_rank"]) or (not info_vt["full_rank"]):
        return _failed_fit(
            k,
            "rank_deficient",
            f"trend-only rank-deficient (rank={rank_tr}, p={need_tr})",
            extra=dict(rank_full=rank_full, rank_tr=rank_tr, rcond=float(info_ut["rcond"])),
        )

    aU, bU = float(cU[0]), float(cU[1])
    aV, bV = float(cV[0]), float(cV[1])
    AU = float(np.sqrt(aU**2 + bU**2))
    AV = float(np.sqrt(aV**2 + bV**2))
    A = float(np.sqrt(aU**2 + bU**2 + aV**2 + bV**2))

    sse_full, sst = sse_sst_two_component(uU, yU, uV, yV, weights=weights)
    sse_tr, _ = sse_sst_two_component(uU, yU_tr, uV, yV_tr, weights=weights)

                               
    mu_u = np.average(uU, weights=weights) if weights is not None else float(np.mean(uU))
    mu_v = np.average(uV, weights=weights) if weights is not None else float(np.mean(uV))
    sst_deg = residual_norm_numerically_zero(
        np.concatenate([uU - mu_u, uV - mu_v]),
        np.zeros(2 * len(uU)),
        weights=None if weights is None else np.concatenate([weights, weights]),
    )
                                       
                                        
    w_cat = None if weights is None else np.concatenate([weights, weights])
    trend_explained = residual_norm_numerically_zero(
        np.concatenate([uU, uV]),
        np.concatenate([yU_tr, yV_tr]),
        weights=w_cat,
    )

    status = "ok"
    status_reason = ""
    if not np.isfinite(sse_full) or not np.isfinite(sst) or not np.isfinite(sse_tr):
        status = "nonfinite_sse"
        status_reason = "non-finite SSE/SST"
        r2 = float("nan")
        partial_r2 = float("nan")
    elif sst_deg or sst <= 0.0:
        status = "degenerate_sst"
        status_reason = "SST numerically degenerate (no variance to explain)"
        r2 = float("nan")
        partial_r2 = float("nan")
    elif trend_explained:
        status = "trend_only_explained"
        status_reason = (
            "trend-only explains data within float64 residual-norm tolerance; "
            "periodic component not identifiable (partial_R2 undefined)"
        )
        r2 = float(1.0 - sse_full / sst) if sst > 0 else float("nan")
        partial_r2 = float("nan")
    else:
        r2 = float(1.0 - sse_full / sst)
        partial_r2 = float(1.0 - sse_full / sse_tr)

    resid_full = np.concatenate([uU - yU, uV - yV])
    resid_tr = np.concatenate([uU - yU_tr, uV - yV_tr])

    return dict(
        k=k,
        lamb=float(2.0 * np.pi / max(k, 1e-12)),
        A_comb=A,
        A_U=AU,
        A_V=AV,
        uU_cos=aU,
        uU_sin=bU,
        uV_cos=aV,
        uV_sin=bV,
        coef_U=cU,
        coef_V=cV,
        yhat_U=yU,
        yhat_V=yV,
        R2=r2,
        partial_R2=partial_r2,
        resid_full=resid_full,
        resid_tr=resid_tr,
        SSE_full=sse_full,
        SSE_tr=sse_tr,
        SST=sst,
        fit_status=status,
        fit_status_reason=status_reason,
        r2_definition=R2_DEFINITION,
        rank_full=rank_full,
        rank_tr=rank_tr,
        rcond=float(info_u["rcond"]),
        singular_values_full_U=info_u["singular_values"],
        singular_values_full_V=info_v["singular_values"],
        analysis_revision=ANALYSIS_REVISION,
    )


def _failed_fit(k: float, status: str, reason: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = dict(
        k=float(k) if np.isfinite(k) else float("nan"),
        lamb=float("nan"),
        A_comb=float("nan"),
        A_U=float("nan"),
        A_V=float("nan"),
        uU_cos=float("nan"),
        uU_sin=float("nan"),
        uV_cos=float("nan"),
        uV_sin=float("nan"),
        coef_U=None,
        coef_V=None,
        yhat_U=None,
        yhat_V=None,
        R2=float("nan"),
        partial_R2=float("nan"),
        resid_full=None,
        resid_tr=None,
        SSE_full=float("nan"),
        SSE_tr=float("nan"),
        SST=float("nan"),
        fit_status=status,
        fit_status_reason=reason,
        r2_definition=R2_DEFINITION,
        rank_full=np.nan,
        rank_tr=np.nan,
        rcond=np.nan,
        analysis_revision=ANALYSIS_REVISION,
    )
    if extra:
        out.update(extra)
    return out


def harmonic_ratio_from_fit(
    uU,
    uV,
    s,
    k,
    ds,
    weights=None,
    trend_degree: int = 1,
) -> Dict[str, float]:
    fit = fit_sin_trend_two_component(
        uU, uV, s, k, weights=weights, trend_degree=trend_degree
    )
    A1 = float(fit["A_comb"])
    if not np.isfinite(A1) or A1 <= 0.0:
        return dict(hr_energy=float("nan"), hr_amp=float("nan"), A1=A1, A3=float("nan"))
    if not np.isfinite(ds) or ds <= 0.0 or 3.0 * float(k) >= np.pi / ds:
        return dict(hr_energy=float("nan"), hr_amp=float("nan"), A1=A1, A3=float("nan"))

    rU = np.asarray(uU, float) - fit["yhat_U"]
    rV = np.asarray(uV, float) - fit["yhat_V"]
    X3 = design_matrix(s, 3.0 * float(k), trend_degree=0)
    coef3U, _ = solve_linear(X3, rU, weights=weights)
    coef3V, _ = solve_linear(X3, rV, weights=weights)
    a3U, b3U = float(coef3U[0]), float(coef3U[1])
    a3V, b3V = float(coef3V[0]), float(coef3V[1])
    A3 = float(np.sqrt(a3U**2 + b3U**2 + a3V**2 + b3V**2))
    return dict(
        hr_energy=float((A3 * A3) / (A1 * A1 + 1e-12)),
        hr_amp=float(A3 / (A1 + 1e-12)),
        A1=A1,
        A3=A3,
    )


def candidate_is_selectable(c: Dict[str, Any]) -> bool:
    if str(c.get("fit_status", "")) not in SELECTABLE_FIT_STATUSES:
        return False
    if not np.isfinite(c.get("SSE_full", np.nan)):
        return False
    if not np.isfinite(c.get("R2", np.nan)):
        return False
    if not np.isfinite(c.get("k", np.nan)) or float(c["k"]) <= 0.0:
        return False
    if not np.isfinite(c.get("lambda", np.nan)) or float(c["lambda"]) <= 0.0:
        return False
    return True


def choose_k_min_sse(
    candidates: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int, int]:
    valid = [(i, c) for i, c in enumerate(candidates) if candidate_is_selectable(c)]
    if not valid:
        raise ValueError("No numerically valid selectable candidate")

    sse_vals = np.array([c["SSE_full"] for _, c in valid], float)
    sse_min = float(np.min(sse_vals))
    tie_idx = [
        i
        for i, c in valid
        if np.isclose(c["SSE_full"], sse_min, rtol=EXACT_TIE_RTOL, atol=EXACT_TIE_ATOL)
    ]
                                   
    best_i = min(tie_idx, key=lambda i: float(candidates[i]["k"]))
    return candidates[best_i], best_i, len(tie_idx)


def choose_k_from_tie_set_legacy_diagnostic(
    candidates: Sequence[Dict[str, Any]],
    dR2_tol: float = 0.006,
    dhr_amp_tol: float = 0.05,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    r2_vals = np.array([c["R2"] for c in candidates], float)
    if not np.any(np.isfinite(r2_vals)):
        raise ValueError("No finite R2 in candidates")
    r2_max = float(np.nanmax(r2_vals))
    tie = [c for c in candidates if np.isfinite(c["R2"]) and (r2_max - c["R2"]) <= dR2_tol]
    if len(tie) == 1:
        return tie[0], tie

    hr_vals = np.array([c.get("hr_amp", np.nan) for c in tie], float)
    if np.all(~np.isfinite(hr_vals)):
        tie2 = tie
    else:
        hr_min = float(np.nanmin(hr_vals))
        tie2 = [c for c in tie if np.isfinite(c.get("hr_amp", np.nan)) and (c["hr_amp"] - hr_min) <= dhr_amp_tol]
        if len(tie2) == 0:
            tie2 = tie
    tie_sorted = sorted(tie2, key=lambda c: float(c["k"]))
    return tie_sorted[0], tie_sorted


def grid_search_k_band(
    uU,
    uV,
    s,
    ds,
    N_cyc: float = 0.5,
    N_pp: float = 20,
    ds_eff: float = 1.0,
    q_fine: int = 8,
    weights=None,
    trend_degree: int = 1,
    boundary_margin_bins: int = 2,
    verbose: bool = True,
    voxel_size_mm: float = 0.5,
    compute_legacy_tie_diagnostic: bool = True,
    dR2_tol_diagnostic: float = 0.006,
    dhr_amp_tol_diagnostic: float = 0.05,
) -> Dict[str, Any]:
    uU = np.asarray(uU, float)
    uV = np.asarray(uV, float)
    s = np.asarray(s, float)

    L = float(s[-1] - s[0]) if len(s) >= 2 else float("nan")
    lambda_min_search = float(N_pp * ds_eff)               
    lambda_min_search_mm = lambda_min_search * float(voxel_size_mm)

    meta = dict(
        analysis_version=ANALYSIS_VERSION,
        analysis_revision=ANALYSIS_REVISION,
        r2_definition=R2_DEFINITION,
        k_selection_method=K_SELECTION_METHOD,
        baseline_method=BASELINE_METHOD,
        frame_reference=FRAME_REFERENCE,
        fitting_coordinate=FITTING_COORDINATE,
        N_cyc=float(N_cyc),
        N_pp=float(N_pp),
        ds_eff=float(ds_eff),
        q_fine=int(q_fine),
        voxel_size_mm=float(voxel_size_mm),
        lambda_min_search=lambda_min_search,
        lambda_min_search_mm=lambda_min_search_mm,
        SSE_unit="voxel^2",
        min_interval_length_mm_note=(
            "min interval length (e.g. 10 mm) is a separate exclusion rule; "
            "not the same as shortest searchable wavelength"
        ),
    )

    if verbose:
        print("Fine-only grid search (series 11: argmin SSE_full)")
        print(f"  shortest searchable wavelength = {lambda_min_search:.4f} (unit of s)"
              f" = {lambda_min_search_mm:.4f} mm "
              f"(N_pp={N_pp}, ds_eff={ds_eff}, voxel_size={voxel_size_mm} mm)")
        print("  note: min interval length and shortest searchable wavelength are distinct settings")

    if len(s) < 2 or not np.isfinite(L) or L <= 0.0:
        return dict(
            status="nonpositive_length",
            status_reason="invalid s / non-positive length",
            k_hat=float("nan"),
            lambda_hat=float("nan"),
            R2=float("nan"),
            partial_R2=float("nan"),
            SSE_full=float("nan"),
            SSE_tr=float("nan"),
            SST=float("nan"),
            candidates=[],
            **meta,
        )
    if not _finite_arrays(uU, uV, s):
        return dict(
            status="nonfinite_input",
            status_reason="non-finite uU/uV/s",
            k_hat=float("nan"),
            lambda_hat=float("nan"),
            R2=float("nan"),
            partial_R2=float("nan"),
            SSE_full=float("nan"),
            SSE_tr=float("nan"),
            SST=float("nan"),
            candidates=[],
            **meta,
        )

    try:
        ks, k_min, k_max, dk = make_k_grid(
            L,
            N_cyc=N_cyc,
            N_pp=N_pp,
            ds_eff=ds_eff,
            q_fine=q_fine,
            verbose=verbose,
        )
    except ValueError as exc:
        return dict(
            status="invalid_band",
            status_reason=str(exc),
            k_hat=float("nan"),
            lambda_hat=float("nan"),
            R2=float("nan"),
            partial_R2=float("nan"),
            SSE_full=float("nan"),
            SSE_tr=float("nan"),
            SST=float("nan"),
            candidates=[],
            **meta,
        )

    candidates: List[Dict[str, Any]] = []
    for cand_i, k in enumerate(ks):
        fit = fit_sin_trend_two_component(
            uU, uV, s, k, weights=weights, trend_degree=trend_degree
        )
        try:
            hr = harmonic_ratio_from_fit(
                uU=uU, uV=uV, s=s, k=k, ds=ds, weights=weights, trend_degree=trend_degree
            )
            hr_amp = float(hr["hr_amp"])
            hr_energy = float(hr["hr_energy"])
        except Exception:  # noqa: BLE001
                                
            hr_amp = float("nan")
            hr_energy = float("nan")

        cand = {
            "candidate_index": int(cand_i),                              
            "k": float(k),
            "lambda": float(fit["lamb"]) if np.isfinite(fit.get("lamb", np.nan)) else float("nan"),
            "R2": float(fit["R2"]) if fit["R2"] is not None else float("nan"),
            "partial_R2": float(fit["partial_R2"]) if fit["partial_R2"] is not None else float("nan"),
            "SSE_full": float(fit["SSE_full"]) if fit["SSE_full"] is not None else float("nan"),
            "SSE_tr": float(fit["SSE_tr"]) if fit["SSE_tr"] is not None else float("nan"),
            "SST": float(fit["SST"]) if fit["SST"] is not None else float("nan"),
            "hr_energy": hr_energy,
            "hr_amp": hr_amp,
            "fit_status": fit["fit_status"],
            "fit_status_reason": fit.get("fit_status_reason", ""),
            "rank_full": fit.get("rank_full", np.nan),
            "rank_tr": fit.get("rank_tr", np.nan),
            "rcond": fit.get("rcond", np.nan),
            "fit": fit,
        }
        candidates.append(cand)

    try:
        best, selected_index, n_exact_ties = choose_k_min_sse(candidates)
    except ValueError as exc:
        return dict(
            status="no_valid_candidate",
            status_reason=str(exc),
            k_hat=float("nan"),
            lambda_hat=float("nan"),
            R2=float("nan"),
            partial_R2=float("nan"),
            SSE_full=float("nan"),
            SSE_tr=float("nan"),
            SST=float("nan"),
            candidates=candidates,
            ks=ks,
            k_min=float(k_min),
            k_max=float(k_max),
            dk=float(dk),
            n_grid=int(len(ks)),
            selected_index=None,
            n_exact_sse_ties=0,
            **meta,
        )

    fit_best = best["fit"]
    k_hat = float(best["k"])
    idx_hat = int(selected_index)
    boundary_hit_low = bool(idx_hat <= boundary_margin_bins)
    boundary_hit_high = bool(idx_hat >= (len(ks) - 1 - boundary_margin_bins))

    legacy_diag = None
    if compute_legacy_tie_diagnostic:
        try:
            legacy_best, legacy_tie = choose_k_from_tie_set_legacy_diagnostic(
                candidates,
                dR2_tol=dR2_tol_diagnostic,
                dhr_amp_tol=dhr_amp_tol_diagnostic,
            )
            legacy_diag = dict(
                k_hat=float(legacy_best["k"]),
                lambda_hat=float(legacy_best["lambda"]),
                R2=float(legacy_best["R2"]),
                partial_R2=float(legacy_best["partial_R2"]),
                SSE_full=float(legacy_best["SSE_full"]),
                n_tie=len(legacy_tie),
                note="diagnostic_only_corrected_R2_with_legacy_tie_rule",
            )
        except Exception as exc:  # noqa: BLE001
            legacy_diag = dict(status="failed", status_reason=str(exc))

    if verbose:
        print(f"  L = {L:.4f} (same unit as s)")
        print(f"  dk = {dk:.6g}, n_grid = {len(ks)}")
        print(
            f"  selected (grid optimum) k = {k_hat:.6g}, lambda = {best['lambda']:.4f}, "
            f"R2 = {best['R2']:.4f}, partial_R2 = {best['partial_R2']:.4f}, "
            f"SSE_full = {best['SSE_full']:.6g}, exact_sse_ties = {n_exact_ties}"
        )
        if boundary_hit_low:
            print("  boundary_hit_low = True")
        if boundary_hit_high:
            print("  boundary_hit_high = True")

    status = fit_best.get("fit_status", "ok")
    status_reason = fit_best.get("fit_status_reason", "")
    if status == "ok" and not np.isfinite(best["R2"]):
        status = "nonfinite_r2"
        status_reason = "selected candidate has non-finite R2"

    return dict(
        status=status,
        status_reason=status_reason,
        k_hat=k_hat,
        lambda_hat=float(best["lambda"]),
        R2=float(best["R2"]),
        partial_R2=float(best["partial_R2"]),
        hr_amp=float(best["hr_amp"]),
        hr_energy=float(best["hr_energy"]),
        A_comb=float(fit_best["A_comb"]),
        A_U=float(fit_best["A_U"]),
        A_V=float(fit_best["A_V"]),
        uU_cos=float(fit_best["uU_cos"]),
        uU_sin=float(fit_best["uU_sin"]),
        uV_cos=float(fit_best["uV_cos"]),
        uV_sin=float(fit_best["uV_sin"]),
        coef_U=fit_best["coef_U"],
        coef_V=fit_best["coef_V"],
        yhat_U=fit_best["yhat_U"],
        yhat_V=fit_best["yhat_V"],
        fit=fit_best,
        SSE_full=float(best["SSE_full"]),
        SSE_tr=float(best["SSE_tr"]),
        SST=float(best["SST"]),
        k_min=float(k_min),
        k_max=float(k_max),
        ks=ks,
        dk=float(dk),
        L=float(L),
        n_grid=int(len(ks)),
        selected_index=int(selected_index),
        n_exact_sse_ties=int(n_exact_ties),
        boundary_hit_low=boundary_hit_low,
        boundary_hit_high=boundary_hit_high,
        boundary_margin_bins=int(boundary_margin_bins),
        candidates=candidates,
        legacy_tie_diagnostic=legacy_diag,
        trend_degree=int(trend_degree),
        **meta,
    )


def reconstruct_sin_displacement_uv(
    s_arr,
    k: float,
    uU_cos: float,
    uU_sin: float,
    uV_cos: float,
    uV_sin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s_arr, float)
    hU = uU_cos * np.cos(k * s) + uU_sin * np.sin(k * s)
    hV = uV_cos * np.cos(k * s) + uV_sin * np.sin(k * s)
    return hU, hV


# ---------------------------------------------------------------------------
# Constrained-1D pathway: B = normalize(T × n_eff), fit on scalar u_B
# ---------------------------------------------------------------------------

FRAME_REFERENCE_CONSTRAINED1D = "surface_binormal_T_cross_n_eff"
R2_DEFINITION_1D = "one_component_centered_sst"
MIN_BINORMAL_CROSS_NORM = 1e-3
MIN_VALID_B_FRACTION = 0.5


def build_surface_binormal(
    T_3darr,
    n_eff_3darr,
    *,
    min_cross_norm: float = MIN_BINORMAL_CROSS_NORM,
) -> Tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T_3darr, float)
    n = np.asarray(n_eff_3darr, float)
    if T.shape != n.shape or T.ndim != 2 or T.shape[1] != 3:
        raise ValueError(f"T and n_eff must be (N,3); got {T.shape}, {n.shape}")
    cross = np.cross(T, n)
    cross_norm = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(cross_norm) & (cross_norm >= float(min_cross_norm))
    B = np.zeros_like(T)
    if np.any(valid):
        B[valid] = cross[valid] / cross_norm[valid, None]
            
    for i in range(1, len(B)):
        if valid[i] and valid[i - 1] and np.dot(B[i], B[i - 1]) < 0.0:
            B[i] = -B[i]
    return B, valid


def project_lateral_onto_B(
    Delta_perp_3darr,
    B_3darr,
    valid_mask=None,
) -> np.ndarray:
    D = np.asarray(Delta_perp_3darr, float)
    B = np.asarray(B_3darr, float)
    if D.shape != B.shape:
        raise ValueError(f"Delta_perp and B shape mismatch: {D.shape} vs {B.shape}")
    u_B = np.einsum("ij,ij->i", D, B)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, bool)
        u_B = u_B.astype(float, copy=True)
        u_B[~mask] = np.nan
    return u_B


def project_uv_onto_B(uU, uV, U_3darr, V_3darr, B_3darr, valid_mask=None) -> np.ndarray:
    """(uU U + uV V) · B。"""
    uU = np.asarray(uU, float)
    uV = np.asarray(uV, float)
    U = np.asarray(U_3darr, float)
    V = np.asarray(V_3darr, float)
    B = np.asarray(B_3darr, float)
    Delta_perp = uU[:, None] * U + uV[:, None] * V
    return project_lateral_onto_B(Delta_perp, B, valid_mask=valid_mask)


                                   
EVAL_NORMAL_SOURCE_NONE = 0        
EVAL_NORMAL_SOURCE_GOOD = 1              
EVAL_NORMAL_SOURCE_BRIDGED = 2                   


def unsigned_axis_angle_deg(n0: np.ndarray, n1: np.ndarray) -> float:
    a = np.asarray(n0, float).ravel()
    b = np.asarray(n1, float).ravel()
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if not (np.isfinite(an) and an > 0.0 and np.isfinite(bn) and bn > 0.0):
        return float("nan")
    c = abs(float(np.dot(a / an, b / bn)))
    return float(np.degrees(np.arccos(float(np.clip(c, 0.0, 1.0)))))


def _unit_or_none(n: np.ndarray) -> Optional[np.ndarray]:
    v = np.asarray(n, float).ravel()
    if v.size != 3 or not np.all(np.isfinite(v)):
        return None
    nn = float(np.linalg.norm(v))
    if not (np.isfinite(nn) and nn > 0.0):
        return None
    return v / nn


def interpolate_observed_normal_brackets(
    s_full: np.ndarray,
    s_anchors: np.ndarray,
    anchor_normals: np.ndarray,
    anchor_observed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    s_full = np.asarray(s_full, float).ravel()
    s_anchors = np.asarray(s_anchors, float).ravel()
    n_anchors = np.asarray(anchor_normals, float)
    obs = np.asarray(anchor_observed, bool).ravel()
    n_points = int(s_full.size)
    normals_eval = np.full((n_points, 3), np.nan, dtype=float)
    bracket_ok = np.zeros(n_points, dtype=bool)
    if n_anchors.ndim != 2 or n_anchors.shape[1] != 3:
        raise ValueError(f"anchor_normals must be (M,3); got {n_anchors.shape}")
    if len(s_anchors) != len(obs) or len(s_anchors) != len(n_anchors):
        raise ValueError("s_anchors, anchor_normals, anchor_observed length mismatch")
    if n_points == 0 or len(s_anchors) == 0:
        return normals_eval, bracket_ok

    for i in range(len(s_anchors) - 1):
        if not (obs[i] and obs[i + 1]):
            continue
        s0 = float(s_anchors[i])
        s1 = float(s_anchors[i + 1])
        n0 = _unit_or_none(n_anchors[i])
        n1 = _unit_or_none(n_anchors[i + 1])
        if n0 is None or n1 is None:
            continue
        if float(np.dot(n0, n1)) < 0.0:
            n1 = -n1
        seg = (s_full >= s0) & (s_full <= s1)
        if not np.any(seg):
            continue
        if np.isclose(s1, s0):
            normals_eval[seg] = n0
        else:
            w = ((s_full[seg] - s0) / (s1 - s0))[:, None]
            normals_eval[seg] = (1.0 - w) * n0 + w * n1
        bracket_ok[seg] = True

    for i, is_obs in enumerate(obs):
        if not is_obs:
            continue
        idx = int(np.argmin(np.abs(s_full - s_anchors[i])))
        n = _unit_or_none(n_anchors[i])
        if n is None:
            continue
        normals_eval[idx] = n
        bracket_ok[idx] = True

    okn = np.isfinite(normals_eval).all(axis=1)
    nn = np.linalg.norm(normals_eval[okn], axis=1, keepdims=True)
    good = (nn > 0).ravel()
    idx_ok = np.flatnonzero(okn)
    normals_eval[idx_ok[good]] = normals_eval[idx_ok[good]] / nn[good]
    return normals_eval, bracket_ok


def interpolate_good_anchor_normals_with_bridge(
    s_full: np.ndarray,
    s_anchors: np.ndarray,
    anchor_normals: np.ndarray,
    anchor_good: np.ndarray,
    *,
    voxel_size_mm: float,
    bridge_max_gap_mm: float = 5.0,
    bridge_axis_max_deg: float = 45.0,
) -> Dict[str, Any]:
    s_full = np.asarray(s_full, float).ravel()
    s_anchors = np.asarray(s_anchors, float).ravel()
    n_anchors = np.asarray(anchor_normals, float)
    good = np.asarray(anchor_good, bool).ravel()
    n_points = int(s_full.size)
    normals_eval = np.full((n_points, 3), np.nan, dtype=float)
    eval_source = np.full(n_points, EVAL_NORMAL_SOURCE_NONE, dtype=int)
    constraint = np.zeros(n_points, dtype=bool)
    vs = float(voxel_size_mm)
    max_gap = float(bridge_max_gap_mm)
    max_axis = float(bridge_axis_max_deg)
    if n_anchors.ndim != 2 or n_anchors.shape[1] != 3:
        raise ValueError(f"anchor_normals must be (M,3); got {n_anchors.shape}")
    if len(s_anchors) != len(good) or len(s_anchors) != len(n_anchors):
        raise ValueError("s_anchors, anchor_normals, anchor_good length mismatch")
    if n_points == 0 or len(s_anchors) == 0 or not (np.isfinite(vs) and vs > 0.0):
        return dict(
            normals_eval=normals_eval,
            eval_normal_source=eval_source,
            has_constraint_full=constraint,
            bracket_ok=constraint.copy(),
            bridge_max_gap_mm=max_gap,
            bridge_axis_max_deg=max_axis,
        )

    good_idx = np.flatnonzero(good)

    def _place_segment(i: int, j: int, source_code: int) -> bool:
        n0 = _unit_or_none(n_anchors[i])
        n1 = _unit_or_none(n_anchors[j])
        if n0 is None or n1 is None:
            return False
        if float(np.dot(n0, n1)) < 0.0:
            n1 = -n1
        s0 = float(s_anchors[i])
        s1 = float(s_anchors[j])
        seg = (s_full >= s0) & (s_full <= s1)
        if not np.any(seg):
            return False
        if np.isclose(s1, s0):
            normals_eval[seg] = n0
        else:
            w = ((s_full[seg] - s0) / (s1 - s0))[:, None]
            mid = (1.0 - w) * n0 + w * n1
            nn = np.linalg.norm(mid, axis=1, keepdims=True)
            ok = (nn > 0).ravel()
            mid[ok] = mid[ok] / nn[ok]
            normals_eval[seg] = mid
                                                       
        eval_source[seg] = int(source_code)
        for end_s, end_n in ((s0, n0), (s1, n1)):
            idx = int(np.argmin(np.abs(s_full - end_s)))
            normals_eval[idx] = end_n
            eval_source[idx] = EVAL_NORMAL_SOURCE_GOOD
        constraint[seg] = True
        return True

              
    for i in good_idx:
        n = _unit_or_none(n_anchors[i])
        if n is None:
            continue
        idx = int(np.argmin(np.abs(s_full - s_anchors[i])))
        normals_eval[idx] = n
        eval_source[idx] = EVAL_NORMAL_SOURCE_GOOD
        constraint[idx] = True

    for k in range(len(good_idx) - 1):
        i = int(good_idx[k])
        j = int(good_idx[k + 1])
        gap_mm = abs(float(s_anchors[j]) - float(s_anchors[i])) * vs
        axis_deg = unsigned_axis_angle_deg(n_anchors[i], n_anchors[j])
        if j == i + 1:
                                      
            _place_segment(i, j, EVAL_NORMAL_SOURCE_GOOD)
            continue
                                   
        if not (np.isfinite(gap_mm) and gap_mm <= max_gap):
            continue
        if not (np.isfinite(axis_deg) and axis_deg <= max_axis):
            continue
        _place_segment(i, j, EVAL_NORMAL_SOURCE_BRIDGED)

    okn = np.isfinite(normals_eval).all(axis=1)
    nn = np.linalg.norm(normals_eval[okn], axis=1, keepdims=True)
    ok_u = (nn > 0).ravel()
    idx_ok = np.flatnonzero(okn)
    normals_eval[idx_ok[ok_u]] = normals_eval[idx_ok[ok_u]] / nn[ok_u]
    return dict(
        normals_eval=normals_eval,
        eval_normal_source=eval_source,
        has_constraint_full=constraint,
        bracket_ok=constraint & okn,
        bridge_max_gap_mm=max_gap,
        bridge_axis_max_deg=max_axis,
    )


def estimate_n_eff_along_centerline(
    P_3darr,
    cropped_mesh,
    r_patch_vox: float,
    *,
    stride: int = 10,
    min_neighbors: int = 30,
    coh_thr: float = 0.75,
    gap_fill_size: int = 3,                                    
    ds_vox: float = 1.0,
    voxel_size_mm: float = 0.5,
    bridge_max_gap_mm: float = 5.0,
    bridge_axis_max_deg: float = 45.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    from scipy.interpolate import interp1d
    from scipy.spatial import cKDTree

    leg = _legacy()
    estimate_fn = leg.estimate_effective_normal_from_mesh_normals
    norm_fn = leg.normalize
    _ = gap_fill_size  # unused; kept for call-site compatibility

    P = np.asarray(P_3darr, float)
    n_points = len(P)
    if cropped_mesh is None:
        raise ValueError("cropped_mesh is required")

    anchor_indices = np.arange(0, n_points, int(stride))
    if len(anchor_indices) == 0 or anchor_indices[-1] != n_points - 1:
        anchor_indices = np.append(anchor_indices, n_points - 1)

                                               
                                              
    if n_points <= 1:
        s_full = np.zeros(n_points, dtype=float)
    else:
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        seg = np.where(np.isfinite(seg), seg, 0.0)
        s_full = np.concatenate([[0.0], np.cumsum(seg)])
        if not np.all(np.isfinite(s_full)) or float(s_full[-1]) <= 0.0:
            s_full = np.arange(n_points, dtype=float) * float(ds_vox)
    s_anchors = s_full[anchor_indices]
    anchor_pts = P[anchor_indices]

    surf_pts = cropped_mesh.points
    if "Normals" in cropped_mesh.point_data:
        surf_normals = cropped_mesh.point_data["Normals"]
    else:
        surf_normals = cropped_mesh.point_normals
    tree_pts = cKDTree(surf_pts)

    anchor_normals_obs = np.full((len(anchor_indices), 3), np.nan, dtype=float)
    anchor_normals_good = np.full((len(anchor_indices), 3), np.nan, dtype=float)
    anchor_normals_display = np.full((len(anchor_indices), 3), np.nan, dtype=float)
    anchor_coherence = []
    anchor_has_cortex = []
    anchor_good = []
    anchor_observed = []
    prev_obs_n = None
    prev_display_n = None
    default_n = np.array([0.0, 0.0, 1.0])

    for i, pt in enumerate(anchor_pts):
        n_eff, coh, nnb, _idx = estimate_fn(
            pt, tree_pts, surf_normals, r_patch_vox, min_neighbors=min_neighbors, prev_n=prev_obs_n
        )
        has_cortex = nnb >= min_neighbors
        n_unit = _unit_or_none(n_eff) if (has_cortex and n_eff is not None) else None
        observed = n_unit is not None
        anchor_has_cortex.append(bool(has_cortex))
        anchor_observed.append(bool(observed))
        if not observed:
            n_disp = prev_display_n.copy() if prev_display_n is not None else default_n.copy()
            coh_use = float("nan")
            n_obs = None
            is_good = False
        else:
            n_obs = n_unit.copy()
            if prev_obs_n is not None and float(np.dot(prev_obs_n, n_obs)) < 0.0:
                n_obs = -n_obs
            n_disp = n_obs.copy()
            coh_use = float(coh) if np.isfinite(coh) else float("nan")
            prev_obs_n = n_obs.copy()
            is_good = bool(np.isfinite(coh_use) and coh_use >= float(coh_thr))
        if prev_display_n is not None and float(np.dot(prev_display_n, n_disp)) < 0.0:
            n_disp = -n_disp
        prev_display_n = n_disp.copy()
        if n_obs is not None:
            anchor_normals_obs[i] = n_obs
            if is_good:
                anchor_normals_good[i] = n_obs
        anchor_normals_display[i] = n_disp
        anchor_coherence.append(coh_use)
        anchor_good.append(bool(is_good))

    anchor_coherence = np.asarray(anchor_coherence, float)
    anchor_has_cortex = np.asarray(anchor_has_cortex, bool)
    anchor_observed = np.asarray(anchor_observed, bool)
    anchor_good = np.asarray(anchor_good, bool)
                                        
    anchor_has_constraint = anchor_good.copy()

                              
    interp_display = interp1d(
        s_anchors, anchor_normals_display, axis=0, kind="linear", fill_value="extrapolate"
    )
    normals_display_full = np.asarray(interp_display(s_full), float)
    okn_disp = np.isfinite(normals_display_full).all(axis=1)
    normals_display_full[okn_disp] = norm_fn(normals_display_full[okn_disp])

                                    
    bridged = interpolate_good_anchor_normals_with_bridge(
        s_full,
        s_anchors,
        anchor_normals_good,
        anchor_good,
        voxel_size_mm=float(voxel_size_mm),
        bridge_max_gap_mm=float(bridge_max_gap_mm),
        bridge_axis_max_deg=float(bridge_axis_max_deg),
    )
    normals_full = np.asarray(bridged["normals_eval"], float)
    eval_normal_source = np.asarray(bridged["eval_normal_source"], int)
    has_constraint_full = np.asarray(bridged["has_constraint_full"], bool)
    bracket_ok = np.asarray(bridged["bracket_ok"], bool)

                                                               
    interp_cortex = interp1d(
        s_anchors, anchor_has_cortex.astype(float), kind="nearest", fill_value="extrapolate"
    )
    interp_coherence = interp1d(
        s_anchors, anchor_coherence, kind="nearest", fill_value="extrapolate"
    )
    interp_observed = interp1d(
        s_anchors, anchor_observed.astype(float), kind="nearest", fill_value="extrapolate"
    )
    interp_good = interp1d(
        s_anchors, anchor_good.astype(float), kind="nearest", fill_value="extrapolate"
    )
    has_cortex_full = interp_cortex(s_full) > 0.5
    coherence_full = interp_coherence(s_full)
    observed_full = interp_observed(s_full) > 0.5
    good_full_nearest = interp_good(s_full) > 0.5

    okn = np.isfinite(normals_full).all(axis=1)
    mask_eval = (
        has_constraint_full
        & okn
        & (
            (eval_normal_source == EVAL_NORMAL_SOURCE_GOOD)
            | (eval_normal_source == EVAL_NORMAL_SOURCE_BRIDGED)
        )
    )
                                                    
    cortex_coverage = float(np.mean(has_cortex_full)) if n_points else float("nan")
    constraint_coverage = float(np.mean(mask_eval)) if n_points else float("nan")

    if verbose:
        n_good = int(np.sum(eval_normal_source == EVAL_NORMAL_SOURCE_GOOD))
        n_br = int(np.sum(eval_normal_source == EVAL_NORMAL_SOURCE_BRIDGED))
        print(
            f"n_eff along centerline: cortex_coverage={cortex_coverage:.2%}, "
            f"constraint_coverage={constraint_coverage:.2%} "
            f"(eval: good={n_good}, bridged={n_br}; "
            f"bridge_max_gap_mm={float(bridge_max_gap_mm):g}, "
            f"bridge_axis_max_deg={float(bridge_axis_max_deg):g})"
        )

    return dict(
        normals_full=normals_full,
        normals_display_full=normals_display_full,
        coherence_full=coherence_full,
        has_cortex_full=has_cortex_full,
        has_constraint_full=has_constraint_full,
        has_constraint_full_raw=good_full_nearest,
        observed_full=observed_full,
        mask_eval=mask_eval,
        bracket_observed_both=bracket_ok,
        eval_normal_source=eval_normal_source,
        cortex_coverage=cortex_coverage,
        constraint_coverage=constraint_coverage,
        anchor_indices=anchor_indices,
        anchor_normals=anchor_normals_obs,
        anchor_normals_good=anchor_normals_good,
        anchor_normals_display=anchor_normals_display,
        anchor_coherence=anchor_coherence,
        anchor_observed=anchor_observed,
        anchor_good=anchor_good,
        anchor_has_constraint=anchor_has_constraint,
        bridge_max_gap_mm=float(bridge_max_gap_mm),
        bridge_axis_max_deg=float(bridge_axis_max_deg),
        n_points=int(n_points),
    )


def sse_sst_one_component(
    u: np.ndarray,
    yhat: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    u = np.asarray(u, float)
    yhat = np.asarray(yhat, float)
    if weights is None:
        sse = float(np.sum((u - yhat) ** 2))
        sst = float(np.sum((u - np.mean(u)) ** 2))
        return sse, sst
    w = np.asarray(weights, float)
    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0.0:
        return float("nan"), float("nan")
    mu = float(np.sum(w * u) / wsum)
    sse = float(np.sum(w * (u - yhat) ** 2))
    sst = float(np.sum(w * (u - mu) ** 2))
    return sse, sst


def _failed_fit_1d(k: float, status: str, reason: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = dict(
        k=float(k) if np.isfinite(k) else float("nan"),
        lamb=float("nan"),
        A_comb=float("nan"),
        A_U=float("nan"),
        A_V=0.0,
        uU_cos=float("nan"),
        uU_sin=float("nan"),
        uV_cos=0.0,
        uV_sin=0.0,
        coef_U=None,
        coef_V=None,
        yhat_U=None,
        yhat_V=None,
        R2=float("nan"),
        partial_R2=float("nan"),
        resid_full=None,
        resid_tr=None,
        SSE_full=float("nan"),
        SSE_tr=float("nan"),
        SST=float("nan"),
        fit_status=status,
        fit_status_reason=reason,
        r2_definition=R2_DEFINITION_1D,
        rank_full=np.nan,
        rank_tr=np.nan,
        rcond=np.nan,
        analysis_revision="constrained1d",
        n_components=1,
    )
    if extra:
        out.update(extra)
    return out


def fit_sin_trend_one_component(
    u_arr,
    s_arr,
    k_arr,
    weights=None,
    trend_degree: int = 1,
) -> Dict[str, Any]:
    u = np.asarray(u_arr, float)
    s = np.asarray(s_arr, float)
    k = float(k_arr)

    n_need = max(4, 2 + max(0, int(trend_degree)) + 2)
    if len(u) != len(s):
        return _failed_fit_1d(k, "length_mismatch", "u/s length mismatch")
    if len(u) < n_need:
        return _failed_fit_1d(k, "insufficient_points", "too few points for sin+trend model")
    if not _finite_arrays(u, s) or not np.isfinite(k):
        return _failed_fit_1d(k, "nonfinite_input", "non-finite values in inputs")
    if float(s[-1] - s[0]) <= 0.0:
        return _failed_fit_1d(k, "nonpositive_length", "non-positive arc-length range")
    if k <= 0.0:
        return _failed_fit_1d(k, "nonpositive_k", "k must be positive")
    w_err = _validate_weights(weights, len(u))
    if w_err is not None:
        return _failed_fit_1d(k, "invalid_weights", w_err)

    try:
        X = design_matrix(s, k, trend_degree=trend_degree)
        X_tr = design_matrix_trend_only(s, trend_degree=trend_degree)
        coef, yhat, info = solve_linear(X, u, weights=weights, return_info=True)
        _, yhat_tr, info_tr = solve_linear(X_tr, u, weights=weights, return_info=True)
    except ValueError as exc:
        msg = str(exc)
        st = "invalid_weights" if "weight" in msg else "fit_failed"
        return _failed_fit_1d(k, st, msg)
    except Exception as exc:  # noqa: BLE001
        return _failed_fit_1d(k, "fit_failed", f"linear solve failed: {exc}")

    if not info["full_rank"]:
        return _failed_fit_1d(
            k,
            "rank_deficient",
            f"full model rank-deficient (rank={info['rank']}, p={X.shape[1]})",
            extra=dict(rank_full=int(info["rank"]), rcond=float(info["rcond"])),
        )
    if not info_tr["full_rank"]:
        return _failed_fit_1d(
            k,
            "rank_deficient",
            f"trend-only rank-deficient (rank={info_tr['rank']}, p={X_tr.shape[1]})",
            extra=dict(rank_tr=int(info_tr["rank"]), rcond=float(info_tr["rcond"])),
        )

    a, b = float(coef[0]), float(coef[1])
    A = float(np.sqrt(a**2 + b**2))
    sse_full, sst = sse_sst_one_component(u, yhat, weights=weights)
    sse_tr, _ = sse_sst_one_component(u, yhat_tr, weights=weights)

    mu = np.average(u, weights=weights) if weights is not None else float(np.mean(u))
    sst_deg = residual_norm_numerically_zero(u - mu, np.zeros_like(u), weights=weights)
    trend_explained = residual_norm_numerically_zero(u, yhat_tr, weights=weights)

    status = "ok"
    status_reason = ""
    if not np.isfinite(sse_full) or not np.isfinite(sst) or not np.isfinite(sse_tr):
        status = "nonfinite_sse"
        status_reason = "non-finite SSE/SST"
        r2 = float("nan")
        partial_r2 = float("nan")
    elif sst_deg or sst <= 0.0:
        status = "degenerate_sst"
        status_reason = "SST numerically degenerate (no variance to explain)"
        r2 = float("nan")
        partial_r2 = float("nan")
    elif trend_explained:
        status = "trend_only_explained"
        status_reason = (
            "trend-only explains data within float64 residual-norm tolerance; "
            "periodic component not identifiable (partial_R2 undefined)"
        )
        r2 = float(1.0 - sse_full / sst) if sst > 0 else float("nan")
        partial_r2 = float("nan")
    else:
        r2 = float(1.0 - sse_full / sst)
        partial_r2 = float(1.0 - sse_full / sse_tr)

    return dict(
        k=k,
        lamb=float(2.0 * np.pi / max(k, 1e-12)),
        A_comb=A,
        A_U=A,
        A_V=0.0,
        uU_cos=a,
        uU_sin=b,
        uV_cos=0.0,
        uV_sin=0.0,
        coef_U=coef,
        coef_V=None,
        yhat_U=yhat,
        yhat_V=None,
        R2=r2,
        partial_R2=partial_r2,
        resid_full=u - yhat,
        resid_tr=u - yhat_tr,
        SSE_full=sse_full,
        SSE_tr=sse_tr,
        SST=sst,
        fit_status=status,
        fit_status_reason=status_reason,
        r2_definition=R2_DEFINITION_1D,
        rank_full=int(info["rank"]),
        rank_tr=int(info_tr["rank"]),
        rcond=float(info["rcond"]),
        singular_values_full_U=info["singular_values"],
        analysis_revision="constrained1d",
        n_components=1,
    )


def harmonic_ratio_from_fit_1d(
    u,
    s,
    k,
    ds,
    weights=None,
    trend_degree: int = 1,
) -> Dict[str, float]:
    fit = fit_sin_trend_one_component(u, s, k, weights=weights, trend_degree=trend_degree)
    A1 = float(fit["A_comb"])
    if not np.isfinite(A1) or A1 <= 0.0:
        return dict(hr_energy=float("nan"), hr_amp=float("nan"), A1=A1, A3=float("nan"))
    if not np.isfinite(ds) or ds <= 0.0 or 3.0 * float(k) >= np.pi / ds:
        return dict(hr_energy=float("nan"), hr_amp=float("nan"), A1=A1, A3=float("nan"))
    r = np.asarray(u, float) - fit["yhat_U"]
    X3 = design_matrix(s, 3.0 * float(k), trend_degree=0)
    coef3, _ = solve_linear(X3, r, weights=weights)
    a3, b3 = float(coef3[0]), float(coef3[1])
    A3 = float(np.sqrt(a3**2 + b3**2))
    return dict(
        hr_energy=float((A3 * A3) / (A1 * A1 + 1e-12)),
        hr_amp=float(A3 / (A1 + 1e-12)),
        A1=A1,
        A3=A3,
    )


def reconstruct_sin_displacement_1d(s_arr, k: float, a_cos: float, a_sin: float) -> np.ndarray:
    s = np.asarray(s_arr, float)
    return a_cos * np.cos(k * s) + a_sin * np.sin(k * s)


def grid_search_k_band_1d(
    u,
    s,
    ds,
    N_cyc: float = 0.5,
    N_pp: float = 20,
    ds_eff: float = 1.0,
    q_fine: int = 8,
    weights=None,
    trend_degree: int = 1,
    boundary_margin_bins: int = 2,
    verbose: bool = True,
    voxel_size_mm: float = 0.5,
    compute_legacy_tie_diagnostic: bool = True,
    dR2_tol_diagnostic: float = 0.006,
    dhr_amp_tol_diagnostic: float = 0.05,
) -> Dict[str, Any]:
    u = np.asarray(u, float)
    s = np.asarray(s, float)

    L = float(s[-1] - s[0]) if len(s) >= 2 else float("nan")
    lambda_min_search = float(N_pp * ds_eff)
    lambda_min_search_mm = lambda_min_search * float(voxel_size_mm)

    meta = dict(
        analysis_version=ANALYSIS_VERSION,
        analysis_revision="constrained1d",
        r2_definition=R2_DEFINITION_1D,
        k_selection_method=K_SELECTION_METHOD,
        baseline_method=BASELINE_METHOD,
        frame_reference=FRAME_REFERENCE_CONSTRAINED1D,
        fitting_coordinate=FITTING_COORDINATE,
        N_cyc=float(N_cyc),
        N_pp=float(N_pp),
        ds_eff=float(ds_eff),
        q_fine=int(q_fine),
        voxel_size_mm=float(voxel_size_mm),
        lambda_min_search=lambda_min_search,
        lambda_min_search_mm=lambda_min_search_mm,
        SSE_unit="voxel^2",
        n_components=1,
    )

    if verbose:
        print("Fine-only grid search (constrained1d: argmin SSE_full, 1-component)")
        print(
            f"  shortest searchable wavelength = {lambda_min_search:.4f} "
            f"= {lambda_min_search_mm:.4f} mm"
        )

    empty = dict(
        k_hat=float("nan"),
        lambda_hat=float("nan"),
        R2=float("nan"),
        partial_R2=float("nan"),
        SSE_full=float("nan"),
        SSE_tr=float("nan"),
        SST=float("nan"),
        candidates=[],
        **meta,
    )

    if len(s) < 2 or not np.isfinite(L) or L <= 0.0:
        return dict(status="nonpositive_length", status_reason="invalid s / non-positive length", **empty)
    if not _finite_arrays(u, s):
        return dict(status="nonfinite_input", status_reason="non-finite u/s", **empty)

    try:
        ks, k_min, k_max, dk = make_k_grid(
            L, N_cyc=N_cyc, N_pp=N_pp, ds_eff=ds_eff, q_fine=q_fine, verbose=verbose
        )
    except ValueError as exc:
        return dict(status="invalid_band", status_reason=str(exc), **empty)

    candidates: List[Dict[str, Any]] = []
    for cand_i, k in enumerate(ks):
        fit = fit_sin_trend_one_component(u, s, k, weights=weights, trend_degree=trend_degree)
        try:
            hr = harmonic_ratio_from_fit_1d(
                u=u, s=s, k=k, ds=ds, weights=weights, trend_degree=trend_degree
            )
            hr_amp = float(hr["hr_amp"])
            hr_energy = float(hr["hr_energy"])
        except Exception:  # noqa: BLE001
            hr_amp = float("nan")
            hr_energy = float("nan")
        candidates.append(
            {
                "candidate_index": int(cand_i),
                "k": float(k),
                "lambda": float(fit["lamb"]) if np.isfinite(fit.get("lamb", np.nan)) else float("nan"),
                "R2": float(fit["R2"]) if fit["R2"] is not None else float("nan"),
                "partial_R2": float(fit["partial_R2"]) if fit["partial_R2"] is not None else float("nan"),
                "SSE_full": float(fit["SSE_full"]) if fit["SSE_full"] is not None else float("nan"),
                "SSE_tr": float(fit["SSE_tr"]) if fit["SSE_tr"] is not None else float("nan"),
                "SST": float(fit["SST"]) if fit["SST"] is not None else float("nan"),
                "hr_energy": hr_energy,
                "hr_amp": hr_amp,
                "fit_status": fit["fit_status"],
                "fit_status_reason": fit.get("fit_status_reason", ""),
                "rank_full": fit.get("rank_full", np.nan),
                "rank_tr": fit.get("rank_tr", np.nan),
                "rcond": fit.get("rcond", np.nan),
                "fit": fit,
            }
        )

    try:
        best, selected_index, n_exact_ties = choose_k_min_sse(candidates)
    except ValueError as exc:
        return dict(
            status="no_valid_candidate",
            status_reason=str(exc),
            k_hat=float("nan"),
            lambda_hat=float("nan"),
            R2=float("nan"),
            partial_R2=float("nan"),
            SSE_full=float("nan"),
            SSE_tr=float("nan"),
            SST=float("nan"),
            candidates=candidates,
            ks=ks,
            k_min=float(k_min),
            k_max=float(k_max),
            dk=float(dk),
            n_grid=int(len(ks)),
            selected_index=None,
            n_exact_sse_ties=0,
            **meta,
        )

    fit_best = best["fit"]
    k_hat = float(best["k"])
    idx_hat = int(selected_index)
    boundary_hit_low = bool(idx_hat <= boundary_margin_bins)
    boundary_hit_high = bool(idx_hat >= (len(ks) - 1 - boundary_margin_bins))

    legacy_diag = None
    if compute_legacy_tie_diagnostic:
        try:
            legacy_best, legacy_tie = choose_k_from_tie_set_legacy_diagnostic(
                candidates, dR2_tol=dR2_tol_diagnostic, dhr_amp_tol=dhr_amp_tol_diagnostic
            )
            legacy_diag = dict(
                k_hat=float(legacy_best["k"]),
                lambda_hat=float(legacy_best["lambda"]),
                R2=float(legacy_best["R2"]),
                partial_R2=float(legacy_best["partial_R2"]),
                SSE_full=float(legacy_best["SSE_full"]),
                n_tie=len(legacy_tie),
                note="diagnostic_only_corrected_R2_with_legacy_tie_rule",
            )
        except Exception as exc:  # noqa: BLE001
            legacy_diag = dict(status="failed", status_reason=str(exc))

    if verbose:
        print(
            f"  selected (1D) k = {k_hat:.6g}, lambda = {best['lambda']:.4f}, "
            f"R2 = {best['R2']:.4f}, partial_R2 = {best['partial_R2']:.4f}"
        )

    status = fit_best.get("fit_status", "ok")
    status_reason = fit_best.get("fit_status_reason", "")
    if status == "ok" and not np.isfinite(best["R2"]):
        status = "nonfinite_r2"
        status_reason = "selected candidate has non-finite R2"

    return dict(
        status=status,
        status_reason=status_reason,
        k_hat=k_hat,
        lambda_hat=float(best["lambda"]),
        R2=float(best["R2"]),
        partial_R2=float(best["partial_R2"]),
        hr_amp=float(best["hr_amp"]),
        hr_energy=float(best["hr_energy"]),
        A_comb=float(fit_best["A_comb"]),
        A_U=float(fit_best["A_U"]),
        A_V=0.0,
        uU_cos=float(fit_best["uU_cos"]),
        uU_sin=float(fit_best["uU_sin"]),
        uV_cos=0.0,
        uV_sin=0.0,
        coef_U=fit_best["coef_U"],
        coef_V=None,
        yhat_U=fit_best["yhat_U"],
        yhat_V=None,
        fit=fit_best,
        SSE_full=float(best["SSE_full"]),
        SSE_tr=float(best["SSE_tr"]),
        SST=float(best["SST"]),
        k_min=float(k_min),
        k_max=float(k_max),
        ks=ks,
        dk=float(dk),
        L=float(L),
        n_grid=int(len(ks)),
        selected_index=int(selected_index),
        n_exact_sse_ties=int(n_exact_ties),
        boundary_hit_low=boundary_hit_low,
        boundary_hit_high=boundary_hit_high,
        boundary_margin_bins=int(boundary_margin_bins),
        candidates=candidates,
        legacy_tie_diagnostic=legacy_diag,
        trend_degree=int(trend_degree),
        **meta,
    )
