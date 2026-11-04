#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: sin trend wavelength coordinates."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.integrate import quad


def _q_of_s(s: float, s_first: float, s_last: float) -> float:
    denom = float(s_last - s_first)
    if denom == 0.0:
        return float("nan")
    return (float(s) - float(s_first)) / denom


def baseline_speed_norm(coef: np.ndarray, q: float) -> float:
    """P0(q)=c2*q^2+c1*q+c0 → ||dP0/dq|| = ||2*c2*q + c1||."""
    coef = np.asarray(coef, float)
    if coef.shape != (3, 3):
        raise ValueError(f"baseline coef shape must be (3,3), got {coef.shape}")
    c2, c1, _c0 = coef[0], coef[1], coef[2]
    dPdq = 2.0 * c2 * q + c1
    return float(np.linalg.norm(dPdq))


def baseline_arclength_between(
    coef: np.ndarray,
    s_first: float,
    s_last: float,
    s_a: float,
    s_b: float,
    *,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    limit: int = 200,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "status": "failed",
        "status_reason": "",
        "L_base": float("nan"),
        "q_a": float("nan"),
        "q_b": float("nan"),
        "abserr": float("nan"),
        "neval": float("nan"),
        "integral_message": "",
    }
    try:
        coef = np.asarray(coef, float)
    except (TypeError, ValueError) as exc:
        out["status_reason"] = f"coef not array: {exc}"
        return out
    if coef.shape != (3, 3):
        out["status_reason"] = f"coef shape {coef.shape} != (3,3)"
        return out
    if not np.isfinite(coef).all():
        out["status_reason"] = "non-finite baseline coefficients"
        return out
    if not all(np.isfinite(x) for x in (s_first, s_last, s_a, s_b)):
        out["status_reason"] = "non-finite s bounds"
        return out
    if float(s_last) <= float(s_first):
        out["status_reason"] = "s_last <= s_first"
        return out
    if float(s_b) <= float(s_a):
        out["status_reason"] = "s_b <= s_a"
        return out

    q_a = _q_of_s(s_a, s_first, s_last)
    q_b = _q_of_s(s_b, s_first, s_last)
    out["q_a"] = q_a
    out["q_b"] = q_b
    if not (np.isfinite(q_a) and np.isfinite(q_b)):
        out["status_reason"] = "non-finite q"
        return out
    if q_b <= q_a:
        out["status_reason"] = "q_b <= q_a"
        return out

    def integrand(q: float) -> float:
        return baseline_speed_norm(coef, q)

    try:
        val, abserr = quad(integrand, q_a, q_b, epsabs=epsabs, epsrel=epsrel, limit=limit)
    except Exception as exc:  # noqa: BLE001
        out["status_reason"] = f"quad failed: {exc}"
        return out

    if not np.isfinite(val) or val <= 0.0:
        out.update(
            status="nonpositive_or_nonfinite",
            status_reason=f"integral value invalid: {val}",
            L_base=float(val) if np.isfinite(val) else float("nan"),
            abserr=float(abserr) if np.isfinite(abserr) else float("nan"),
        )
        return out

    out.update(
        ok=True,
        status="ok",
        status_reason="",
        L_base=float(val),
        abserr=float(abserr),
        integral_message="scipy.integrate.quad",
    )
    return out


def cumulative_baseline_arclength(
    coef: np.ndarray,
    s_obs: np.ndarray,
    *,
    s_first: float,
    s_last: float,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "status": "failed",
        "status_reason": "",
        "s_base_full": None,
        "L_base_full": float("nan"),
    }
    s_obs = np.asarray(s_obs, float).reshape(-1)
    n = int(s_obs.size)
    if n < 2:
        out["status_reason"] = "fewer than 2 samples"
        return out
    if not np.isfinite(s_obs).all():
        out["status_reason"] = "non-finite s_obs"
        return out
                    
    if np.any(np.diff(s_obs) < -1e-15):
        out["status_reason"] = "s_obs not non-decreasing"
        return out

    ell = np.zeros(n, dtype=float)
    abserr_sum = 0.0
    for i in range(1, n):
        if float(s_obs[i]) <= float(s_obs[i - 1]):
                                                        
            ell[i] = ell[i - 1]
            continue
        seg = baseline_arclength_between(
            coef, s_first, s_last, float(s_obs[i - 1]), float(s_obs[i]), epsabs=epsabs, epsrel=epsrel
        )
        if not seg["ok"]:
            out["status_reason"] = f"segment {i-1}->{i}: {seg.get('status_reason', '')}"
            out["status"] = str(seg.get("status", "failed"))
            return out
        ell[i] = ell[i - 1] + float(seg["L_base"])
        if np.isfinite(seg.get("abserr", np.nan)):
            abserr_sum += float(seg["abserr"])
    if not np.isfinite(ell).all():
        out["status_reason"] = "non-finite cumulative baseline arclength"
        return out
    if float(ell[-1]) <= 0.0:
        out["status_reason"] = "non-positive full baseline arclength"
        return out
                
    if np.any(np.diff(ell) < -1e-15):
        out["status_reason"] = "s_base_full not non-decreasing"
        return out
    out.update(
        ok=True,
        status="ok",
        status_reason="",
        s_base_full=ell,
        L_base_full=float(ell[-1]),
        abserr_sum=float(abserr_sum),
    )
    return out


def effective_baseline_wavelength(
    lambda_s: float,
    L_obs: float,
    L_base: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "status": "failed",
        "status_reason": "",
        "lambda_base_eff": float("nan"),
        "L_base_over_L_obs": float("nan"),
        "N_fit": float("nan"),
        "pct_diff_vs_lambda_s": float("nan"),
    }
    if not all(np.isfinite(x) for x in (lambda_s, L_obs, L_base)):
        out["status_reason"] = "non-finite inputs"
        return out
    if lambda_s <= 0 or L_obs <= 0 or L_base <= 0:
        out["status_reason"] = "non-positive lambda_s/L_obs/L_base"
        return out
    N_fit = L_obs / lambda_s
    lam_b = lambda_s * (L_base / L_obs)
    ratio = L_base / L_obs
    pct = 100.0 * (lam_b - lambda_s) / lambda_s
    out.update(
        ok=True,
        status="ok",
        status_reason="",
        lambda_base_eff=float(lam_b),
        L_base_over_L_obs=float(ratio),
        N_fit=float(N_fit),
        pct_diff_vs_lambda_s=float(pct),
    )
    return out


def convert_wavelengths_for_interval(
    *,
    baseline_coef: np.ndarray,
    s_first: float,
    s_last: float,
    s_a: float,
    s_b: float,
    lambda_s: float,
    L_obs: float,
    voxel_size: float = 1.0,
) -> Dict[str, Any]:
    integ = baseline_arclength_between(baseline_coef, s_first, s_last, s_a, s_b)
    v = float(voxel_size)
    row: Dict[str, Any] = {
        "wavelength_conversion_status": integ["status"],
        "wavelength_conversion_reason": integ["status_reason"],
        "L_baseline_fit_internal": integ["L_base"],
        "L_baseline_fit_mm": integ["L_base"] * v if integ["ok"] else float("nan"),
        "baseline_integral_q_a": integ["q_a"],
        "baseline_integral_q_b": integ["q_b"],
        "baseline_integral_abserr": integ["abserr"],
        "lambda_baseline_eff_mm": float("nan"),
        "L_base_over_L_obs": float("nan"),
        "lambda_base_eff_pct_diff_vs_lambda_s": float("nan"),
    }
    if not integ["ok"]:
        return row
    L_base = float(integ["L_base"])
    conv = effective_baseline_wavelength(lambda_s, L_obs, L_base)
    row["wavelength_conversion_status"] = conv["status"]
    row["wavelength_conversion_reason"] = conv["status_reason"]
    if conv["ok"]:
        row["lambda_baseline_eff_mm"] = float(conv["lambda_base_eff"]) * v
        row["L_base_over_L_obs"] = conv["L_base_over_L_obs"]
        row["lambda_base_eff_pct_diff_vs_lambda_s"] = conv["pct_diff_vs_lambda_s"]
    return row
