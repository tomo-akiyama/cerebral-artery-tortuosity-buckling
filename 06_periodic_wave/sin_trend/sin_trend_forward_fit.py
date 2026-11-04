#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend forward fit."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FIT_METHOD = "forward3d_processed_template_joint_ols"
K_SELECTION_METHOD = "min_SSE_among_valid_candidates_deterministic_tie"
R2_DEFINITION = "1 - SSE_full / SST; SST = sum_c sum_i (u_c,i - mean(u_c))^2"
PARTIAL_R2_DEFINITION = "1 - SSE_full / SSE_trend_only; trend-only refit on Z only"
                                                   
RESIDUAL_VS_SIGNAL_SCALE_FACTOR = 10.0
DEFAULT_RCOND = np.finfo(float).eps * 100  # explicit; recorded in fit metadata


def residual_numerically_zero_vs_signal(
    residual: np.ndarray,
    signal: np.ndarray,
    *,
    c: float = RESIDUAL_VS_SIGNAL_SCALE_FACTOR,
) -> Dict[str, Any]:
    r = np.asarray(residual, float).ravel()
    s = np.asarray(signal, float).ravel()
    n = float(r.size)
    rnorm = float(np.linalg.norm(r)) if n > 0 else 0.0
    scale = float(np.linalg.norm(s)) if s.size else 0.0
    eps = float(np.finfo(float).eps)
    if n <= 0:
        tol = 0.0
        is_zero = True
    elif scale == 0.0:
        tol = 0.0
        is_zero = rnorm == 0.0
    else:
        tol = float(c) * np.sqrt(n) * eps * scale
        is_zero = rnorm <= tol
    return {"is_zero": bool(is_zero), "residual_norm": rnorm, "signal_scale": scale, "tol": tol}


def residualize_quadratic_cartesian(
    F: np.ndarray,
    q: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    F = np.asarray(F, float)
    q = np.asarray(q, float).reshape(-1)
    meta: Dict[str, Any] = {"ok": False, "rank": 0, "reason": ""}
    if F.ndim == 1:
        F = F.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    n = int(F.shape[0])
    if q.shape[0] != n:
        meta["reason"] = f"q length {q.shape[0]} != F rows {n}"
        return np.full_like(F, np.nan), meta
    if not np.isfinite(F).all() or not np.isfinite(q).all():
        meta["reason"] = "non-finite F or q"
        return np.full_like(F, np.nan), meta
    B2 = np.column_stack([np.ones(n), q, q * q])
    if weights is not None:
        w = np.asarray(weights, float).reshape(-1)
        if w.shape[0] != n or not np.isfinite(w).all() or np.any(w < 0):
            meta["reason"] = "invalid weights"
            return np.full_like(F, np.nan), meta
        sw = np.sqrt(w)
        B2w = B2 * sw[:, None]
        Fw = F * sw[:, None]
        Q, R = np.linalg.qr(B2w, mode="reduced")
        rank = int(np.sum(np.abs(np.diag(R)) > 1e-12 * max(1.0, np.abs(R[0, 0]))))
        proj = Q @ (Q.T @ Fw)
        resid = (Fw - proj) / np.where(sw[:, None] > 0, sw[:, None], 1.0)
        # zero-weight rows: leave residual as F (no residualization contribution)
        resid = np.where(sw[:, None] > 0, resid, F)
    else:
        Q, R = np.linalg.qr(B2, mode="reduced")
        rank = int(np.sum(np.abs(np.diag(R)) > 1e-12 * max(1.0, float(np.abs(R[0, 0])) if R.size else 1.0)))
        if rank < 1:
            meta["reason"] = "B2 rank 0"
            return np.full_like(F, np.nan), meta
        Qr = Q[:, :rank]
        resid = F - Qr @ (Qr.T @ F)
    meta.update(ok=True, rank=rank, reason="")
    if squeeze:
        return resid.reshape(-1), meta
    return resid, meta


def build_processed_design(
    *,
    U_full: np.ndarray,
    V_full: np.ndarray,
    s_phase_full: np.ndarray,
    q_full: np.ndarray,
    k: float,
    J: np.ndarray,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    U = np.asarray(U_full, float)
    V = np.asarray(V_full, float)
    s = np.asarray(s_phase_full, float).reshape(-1)
    q = np.asarray(q_full, float).reshape(-1)
    J = np.asarray(J, int).reshape(-1)
    info: Dict[str, Any] = {"ok": False, "reason": ""}
    n = s.shape[0]
    if U.shape != (n, 3) or V.shape != (n, 3) or q.shape[0] != n:
        info["reason"] = "U/V/s/q shape mismatch"
        return None, info
    if not np.isfinite(k) or k <= 0:
        info["reason"] = f"invalid k={k}"
        return None, info
    if J.size < 2 or J.min() < 0 or J.max() >= n:
        info["reason"] = "invalid trim index J"
        return None, info
    # contiguous trim expected for clinical path; do not invent indices
    if not np.all(np.diff(J) == 1):
        info["reason"] = "J not contiguous"
        return None, info

    phase = k * s
    c = np.cos(phase)
    sn = np.sin(phase)
    templates = (
        U * c[:, None],
        U * sn[:, None],
        V * c[:, None],
        V * sn[:, None],
    )
    cols_U: List[np.ndarray] = []
    cols_V: List[np.ndarray] = []
    for F in templates:
        H, meta = residualize_quadratic_cartesian(F, q)
        if not meta["ok"]:
            info["reason"] = f"residualize failed: {meta['reason']}"
            return None, info
        gU = np.sum(H * U, axis=1)[J]
        gV = np.sum(H * V, axis=1)[J]
        cols_U.append(gU)
        cols_V.append(gV)
    Gu = np.column_stack(cols_U)
    Gv = np.column_stack(cols_V)
    W = np.vstack([Gu, Gv])
    if not np.isfinite(W).all():
        info["reason"] = "non-finite W"
        return None, info
    info.update(ok=True, reason="", n_trim=int(J.size))
    return W, info


def _block_diag_trend(t: np.ndarray) -> np.ndarray:
    """Z = block_diag([1,t],[1,t]) -> (2n, 4)."""
    t = np.asarray(t, float).reshape(-1)
    n = t.shape[0]
    ones = np.ones(n)
    Z1 = np.column_stack([ones, t])
    Z = np.zeros((2 * n, 4), float)
    Z[:n, :2] = Z1
    Z[n:, 2:] = Z1
    return Z


def fit_joint_sin_trend(
    u1: np.ndarray,
    u2: np.ndarray,
    t: np.ndarray,
    W: np.ndarray,
    *,
    rcond: Optional[float] = None,
) -> Dict[str, Any]:
    u1 = np.asarray(u1, float).reshape(-1)
    u2 = np.asarray(u2, float).reshape(-1)
    t = np.asarray(t, float).reshape(-1)
    W = np.asarray(W, float)
    n = u1.shape[0]
    if rcond is None:
        rcond = float(DEFAULT_RCOND)
    out: Dict[str, Any] = {
        "ok": False,
        "status": "failed",
        "status_reason": "",
        "coef": np.full(8, np.nan),
        "a1": np.nan,
        "b1": np.nan,
        "a2": np.nan,
        "b2": np.nan,
        "c1": np.nan,
        "d1": np.nan,
        "c2": np.nan,
        "d2": np.nan,
        "SSE_full": np.nan,
        "SSE_tr": np.nan,
        "SST": np.nan,
        "R2": np.nan,
        "partial_R2": np.nan,
        "R2_trend_only": np.nan,
        "rank_X": np.nan,
        "rank_Z": np.nan,
        "rcond": rcond,
        "cond_X": np.nan,
        "response_U": np.full(n, np.nan),
        "response_V": np.full(n, np.nan),
        "trend_U": np.full(n, np.nan),
        "trend_V": np.full(n, np.nan),
        "yhat_U": np.full(n, np.nan),
        "yhat_V": np.full(n, np.nan),
        "resid_full": None,
        "resid_tr": None,
        "trend_only_numerically_complete": False,
        "sst_numerically_zero": False,
    }
    if not (u2.shape[0] == n and t.shape[0] == n and W.shape == (2 * n, 4)):
        out["status_reason"] = "shape mismatch"
        return out
    if not (np.isfinite(u1).all() and np.isfinite(u2).all() and np.isfinite(t).all() and np.isfinite(W).all()):
        out["status_reason"] = "non-finite inputs"
        return out

    y = np.concatenate([u1, u2])
    Z = _block_diag_trend(t)
    X = np.hstack([W, Z])

    # SST: center each component separately
    u1c = u1 - np.mean(u1)
    u2c = u2 - np.mean(u2)
    y_centered = np.concatenate([u1c, u2c])
    SST = float(np.sum(u1c ** 2) + np.sum(u2c ** 2))
    sst_zero = residual_numerically_zero_vs_signal(y_centered, y)

    # trend-only refit on Z (lstsq: solution, residuals, rank, singular_values)
    coef_tr, _resid_unused, rank_Z_lstsq, s_Z = np.linalg.lstsq(Z, y, rcond=rcond)
    yhat_tr = Z @ coef_tr
    resid_tr = y - yhat_tr
    SSE_tr = float(np.sum(resid_tr ** 2))
    s_Z = np.asarray(s_Z, float)
    rank_Z = int(rank_Z_lstsq)

    coef, _resid_unused2, rank_X_lstsq, s_X = np.linalg.lstsq(X, y, rcond=rcond)
    yhat = X @ coef
    resid = y - yhat
    SSE_full = float(np.sum(resid ** 2))
    s_X = np.asarray(s_X, float)
    rank_X = int(rank_X_lstsq)

    if not np.isfinite(coef).all() or not np.isfinite(yhat).all() or not np.isfinite(SSE_full):
        out["status_reason"] = "non-finite fit"
        return out
    if rank_Z < 4:
        out["status_reason"] = f"Z rank {rank_Z} < 4"
        out.update(rank_Z=rank_Z, rank_X=rank_X, SSE_full=SSE_full, SSE_tr=SSE_tr, SST=SST)
        return out
    if rank_X < 8:
        out["status_reason"] = f"X rank {rank_X} < 8"
        out.update(rank_Z=rank_Z, rank_X=rank_X, SSE_full=SSE_full, SSE_tr=SSE_tr, SST=SST)
        return out

    try:
        cond_X = float(np.linalg.cond(X))
    except np.linalg.LinAlgError:
        cond_X = float("nan")

    # Good zero-residual fit is valid; SST numerical zero is a separate QC flag
    if sst_zero["is_zero"] and SSE_full == 0.0:
        R2 = 1.0
    elif SST > 0:
        R2 = 1.0 - SSE_full / SST
    else:
        R2 = float("nan")

    # trend-only explains signal within float precision → partial R2 undefined
    tr_zero = residual_numerically_zero_vs_signal(resid_tr, y)
    if not np.isfinite(SSE_tr) or SSE_tr <= 0.0 or tr_zero["is_zero"]:
        partial = float("nan")
        R2_tr = 1.0 if (sst_zero["is_zero"] or SSE_tr <= 0.0 or tr_zero["is_zero"]) else float("nan")
        if SST > 0 and not tr_zero["is_zero"] and SSE_tr > 0:
            R2_tr = 1.0 - SSE_tr / SST
        partial_reason = "trend_only_explains_signal_within_float_precision"
        tr_complete = True
    else:
        partial = 1.0 - SSE_full / SSE_tr
        R2_tr = 1.0 - SSE_tr / SST if SST > 0 else float("nan")
        partial_reason = ""
        tr_complete = False

    resp = W @ coef[:4]
    trend = Z @ coef[4:]
    out.update(
        ok=True,
        status="ok",
        status_reason=partial_reason,
        coef=np.asarray(coef, float),
        a1=float(coef[0]),
        b1=float(coef[1]),
        a2=float(coef[2]),
        b2=float(coef[3]),
        c1=float(coef[4]),
        d1=float(coef[5]),
        c2=float(coef[6]),
        d2=float(coef[7]),
        SSE_full=SSE_full,
        SSE_tr=SSE_tr,
        SST=SST,
        R2=float(R2) if np.isfinite(R2) else float("nan"),
        partial_R2=float(partial) if np.isfinite(partial) else float("nan"),
        R2_trend_only=float(R2_tr) if np.isfinite(R2_tr) else float("nan"),
        rank_X=rank_X,
        rank_Z=rank_Z,
        cond_X=cond_X,
        rcond=float(rcond),
        singular_values_X=np.asarray(s_X, float),
        singular_values_Z=np.asarray(s_Z, float),
        response_U=resp[:n],
        response_V=resp[n:],
        trend_U=trend[:n],
        trend_V=trend[n:],
        yhat_U=yhat[:n],
        yhat_V=yhat[n:],
        resid_full=resid,
        resid_tr=resid_tr,
        coef_trend_only=np.asarray(coef_tr, float),
        trend_only_numerically_complete=bool(tr_complete),
        sst_numerically_zero=bool(sst_zero["is_zero"]),
        trend_only_zero_tol=tr_zero["tol"],
        trend_only_resid_norm=tr_zero["residual_norm"],
    )
    return out


def amplitudes_from_coef(a1: float, b1: float, a2: float, b2: float) -> Dict[str, float]:
    C = np.array([[a1, b1], [a2, b2]], float)
    A_comb = float(np.sqrt(a1 * a1 + b1 * b1 + a2 * a2 + b2 * b2))
    try:
        svals = np.linalg.svd(C, compute_uv=False)
    except np.linalg.LinAlgError:
        svals = np.array([np.nan, np.nan])
    sigma1 = float(svals[0]) if svals.size else float("nan")
    sigma2 = float(svals[1]) if svals.size > 1 else float("nan")
    ratio = sigma2 / sigma1 if np.isfinite(sigma1) and sigma1 > 0 and np.isfinite(sigma2) else float("nan")
                                                                            
    principal_unique = False
    if np.isfinite(sigma1) and np.isfinite(sigma2) and sigma1 > 0:
        anisotropy = float((sigma1 - sigma2) / sigma1)
        anisotropy_status = "ok"
        principal_unique = not np.isclose(sigma1, sigma2, rtol=1e-9, atol=1e-15)
        if not principal_unique:
            anisotropy_status = "ok_circular_principal_axis_not_unique"
    else:
        anisotropy = float("nan")
        anisotropy_status = "undefined"
    return {
        "A_comb": A_comb,
        "amp_peak": sigma1,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma2_over_sigma1": ratio,
        "anisotropy_1_minus_s2_over_s1": anisotropy,
        "anisotropy_status": anisotropy_status,
        "principal_axis_unique": int(principal_unique),
        # legacy alias kept for QC; same as anisotropy (not (s2/s1)^2)
        "directionality": anisotropy,
    }


def latent_uv(s_fit: np.ndarray, k: float, a1: float, b1: float, a2: float, b2: float) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s_fit, float)
    # s_fit is already phase-origin shifted in clinical path (s_phase[J])
    c = np.cos(k * s)
    sn = np.sin(k * s)
    return a1 * c + b1 * sn, a2 * c + b2 * sn


def search_k_forward_3d(
    *,
    u1: np.ndarray,
    u2: np.ndarray,
    s_phase_full: np.ndarray,
    s_phase_trim: np.ndarray,
    s_trend_trim: np.ndarray,
    q_full: np.ndarray,
    U_full: np.ndarray,
    V_full: np.ndarray,
    J: np.ndarray,
    ks: Sequence[float],
    k_min: float,
    k_max: float,
    dk: float,
    verbose: bool = False,
    rcond: Optional[float] = None,
) -> Dict[str, Any]:
    s_phase_trim = np.asarray(s_phase_trim, float).reshape(-1)
    s_trend_trim = np.asarray(s_trend_trim, float).reshape(-1)
    ks_arr = np.asarray(list(ks), float)
    t = s_trend_trim - float(np.mean(s_trend_trim))
    phase_origin = float(s_phase_full[int(J[0])]) if len(J) else float("nan")
    candidates: List[Dict[str, Any]] = []
    best_idx: Optional[int] = None
    best_sse = float("inf")
    best_tie_indices: List[int] = []

    for i, k in enumerate(ks_arr):
        cand: Dict[str, Any] = {
            "index": int(i),
            "k": float(k),
            "lambda": (2.0 * np.pi / float(k)) if np.isfinite(k) and k > 0 else float("nan"),
            "fit_status": "failed",
            "fit_status_reason": "",
            "SSE_full": float("nan"),
            "SSE_tr": float("nan"),
            "SST": float("nan"),
            "R2": float("nan"),
            "partial_R2": float("nan"),
            "R2_trend_only": float("nan"),
            "rank_X": float("nan"),
            "cond_X": float("nan"),
            "boundary_exact_low": bool(np.isfinite(k_min) and np.isclose(k, k_min, rtol=0, atol=1e-12)),
            "boundary_exact_high": bool(np.isfinite(k_max) and np.isclose(k, k_max, rtol=0, atol=1e-12)),
            "selected": False,
            "a1": float("nan"),
            "b1": float("nan"),
            "a2": float("nan"),
            "b2": float("nan"),
            "c1": float("nan"),
            "d1": float("nan"),
            "c2": float("nan"),
            "d2": float("nan"),
            "A_comb": float("nan"),
            "sigma1": float("nan"),
            "sigma2": float("nan"),
        }
        W, winfo = build_processed_design(
            U_full=U_full, V_full=V_full, s_phase_full=s_phase_full, q_full=q_full, k=float(k), J=J
        )
        if W is None:
            cand["fit_status_reason"] = winfo.get("reason", "W failed")
            candidates.append(cand)
            continue
        fit = fit_joint_sin_trend(u1, u2, t, W, rcond=rcond)
        cand.update(
            fit_status=fit["status"],
            fit_status_reason=fit.get("status_reason", ""),
            SSE_full=fit["SSE_full"],
            SSE_tr=fit["SSE_tr"],
            SST=fit["SST"],
            R2=fit["R2"],
            partial_R2=fit["partial_R2"],
            R2_trend_only=fit["R2_trend_only"],
            rank_X=fit["rank_X"],
            rank_Z=fit["rank_Z"],
            cond_X=fit["cond_X"],
            rcond=fit.get("rcond", np.nan),
            a1=fit["a1"],
            b1=fit["b1"],
            a2=fit["a2"],
            b2=fit["b2"],
            c1=fit["c1"],
            d1=fit["d1"],
            c2=fit["c2"],
            d2=fit["d2"],
        )
        if fit["ok"]:
            amp = amplitudes_from_coef(fit["a1"], fit["b1"], fit["a2"], fit["b2"])
            cand.update(amp)
            sse = float(fit["SSE_full"])
            if np.isfinite(sse):
                if sse < best_sse - 0.0:
                    best_sse = sse
                    best_idx = i
                    best_tie_indices = [i]
                elif abs(sse - best_sse) <= 0.0 or (np.isfinite(best_sse) and abs(sse - best_sse) <= 1e-15 * max(1.0, best_sse)):
                    # exact / near-exact tie: keep deterministic set; choose later
                    if abs(sse - best_sse) == 0.0:
                        best_tie_indices.append(i)
        candidates.append(cand)

    result: Dict[str, Any] = {
        "status": "no_valid_candidate",
        "status_reason": "",
        "k_hat": float("nan"),
        "lambda_hat": float("nan"),
        "selected_index": None,
        "n_grid": int(ks_arr.size),
        "ks": ks_arr,
        "n_exact_sse_ties": 0,
        "k_min": float(k_min),
        "k_max": float(k_max),
        "dk": float(dk),
        "candidates": candidates,
        "fit": None,
        "A_comb": float("nan"),
        "sigma1": float("nan"),
        "sigma2": float("nan"),
        "uU_cos": float("nan"),
        "uU_sin": float("nan"),
        "uV_cos": float("nan"),
        "uV_sin": float("nan"),
        "phase_origin_s": phase_origin,
        "t_mean_s": float(np.mean(s_trend_trim)) if len(s_trend_trim) else float("nan"),
        "phase_coordinate": "baseline_arclength",
        "trend_coordinate": "baseline_arclength",
        "fit_method": FIT_METHOD,
        "k_selection_method": K_SELECTION_METHOD,
        "R2": float("nan"),
        "partial_R2": float("nan"),
        "SSE_full": float("nan"),
        "SSE_tr": float("nan"),
        "SST": float("nan"),
    }

    valid = [c for c in candidates if c["fit_status"] == "ok" and np.isfinite(c["SSE_full"])]
    if not valid:
        result["status_reason"] = "no numerically valid candidates"
        return result

    # deterministic tie-break: min SSE, then min index
    min_sse = min(float(c["SSE_full"]) for c in valid)
    ties = [c for c in valid if float(c["SSE_full"]) == min_sse or abs(float(c["SSE_full"]) - min_sse) <= 0.0]
    # use exact equality for SSE floats from same computation path; fallback to np.isclose
    if len(ties) == 0:
        ties = [c for c in valid if np.isclose(float(c["SSE_full"]), min_sse, rtol=0, atol=0)]
        if not ties:
            ties = sorted(valid, key=lambda c: (float(c["SSE_full"]), int(c["index"])))[:1]
    ties_sorted = sorted(ties, key=lambda c: int(c["index"]))
    chosen = ties_sorted[0]
    result["n_exact_sse_ties"] = int(len(ties_sorted))

    # recompute full fit for chosen k (needed arrays)
    W, winfo = build_processed_design(
        U_full=U_full, V_full=V_full, s_phase_full=s_phase_full, q_full=q_full, k=float(chosen["k"]), J=J
    )
    fit = fit_joint_sin_trend(u1, u2, t, W, rcond=rcond) if W is not None else {"ok": False, "status": "failed", "status_reason": winfo.get("reason", "")}
    if not fit.get("ok"):
        result["status"] = "best_refit_failed"
        result["status_reason"] = fit.get("status_reason", "refit failed")
        return result

    for c in candidates:
        c["selected"] = int(c["index"]) == int(chosen["index"])

    amp = amplitudes_from_coef(fit["a1"], fit["b1"], fit["a2"], fit["b2"])
    lat_U, lat_V = latent_uv(s_phase_trim, float(chosen["k"]), fit["a1"], fit["b1"], fit["a2"], fit["b2"])
    fit["latent_U"] = lat_U
    fit["latent_V"] = lat_V

    result.update(
        status="ok",
        status_reason="",
        k_hat=float(chosen["k"]),
        lambda_hat=float(chosen["lambda"]),
        selected_index=int(chosen["index"]),
        ks=ks_arr,
        fit=fit,
        A_comb=amp["A_comb"],
        sigma1=amp["sigma1"],
        sigma2=amp["sigma2"],
        sigma2_over_sigma1=amp["sigma2_over_sigma1"],
        anisotropy_1_minus_s2_over_s1=amp["anisotropy_1_minus_s2_over_s1"],
        anisotropy_status=amp["anisotropy_status"],
        principal_axis_unique=amp.get("principal_axis_unique", 0),
        directionality=amp["directionality"],
        amp_peak=amp["amp_peak"],
        uU_cos=fit["a1"],
        uU_sin=fit["b1"],
        uV_cos=fit["a2"],
        uV_sin=fit["b2"],
        R2=fit["R2"],
        partial_R2=fit["partial_R2"],
        R2_trend_only=fit["R2_trend_only"],
        SSE_full=fit["SSE_full"],
        SSE_tr=fit["SSE_tr"],
        SST=fit["SST"],
        rank_X=fit.get("rank_X", np.nan),
        rank_Z=fit.get("rank_Z", np.nan),
        cond_X=fit.get("cond_X", np.nan),
        rcond=fit.get("rcond", np.nan),
        hr_amp=float("nan"),
        hr_energy=float("nan"),
        hr_status="not_computed_in_forward3d",
        hr_status_reason="harmonic ratio not part of v2 primary path",
    )
    if verbose:
        print(
            f"[forward3d] k={result['k_hat']:.6g} lambda={result['lambda_hat']:.6g} "
            f"SSE={result['SSE_full']:.6g} R2={result['R2']:.4f} partial_R2={result['partial_R2']:.4f}"
        )
    return result
