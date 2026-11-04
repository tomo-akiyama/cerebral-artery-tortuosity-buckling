#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: test forward fit core."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_forward_fit import (  # noqa: E402
    amplitudes_from_coef,
    build_processed_design,
    fit_joint_sin_trend,
    residualize_quadratic_cartesian,
    search_k_forward_3d,
)
from sin_trend_classify import (  # noqa: E402
    core_periodic_from_row,
    partial_r2_ok,
)
from sin_trend_wavelength_coordinates import (  # noqa: E402
    baseline_arclength_between,
    effective_baseline_wavelength,
)


def test_residualize_removes_quadratic():
    n = 80
    q = np.linspace(0, 1, n)
    # F lies in span{1,q,q^2}
    F = np.column_stack([1 + 2 * q + 3 * q**2, -q + 0.5 * q**2, np.ones(n)])
    resid, meta = residualize_quadratic_cartesian(F, q)
    assert meta["ok"]
    assert np.linalg.norm(resid) < 1e-10


def test_amplitudes_identity():
    a1, b1, a2, b2 = 0.3, -0.4, 0.1, 0.2
    amp = amplitudes_from_coef(a1, b1, a2, b2)
    assert abs(amp["A_comb"] ** 2 - (amp["sigma1"] ** 2 + amp["sigma2"] ** 2)) < 1e-12


def test_joint_fit_recovers_known_coef_constant_frame():
    n = 120
    s = np.linspace(0, 40.0, n)
    k = 2 * np.pi / 20.0
    a1, b1, a2, b2 = 0.5, -0.2, 0.3, 0.1
    c1, d1, c2, d2 = 0.05, 0.001, -0.02, -0.0005
    t = s - s.mean()
    u1 = a1 * np.cos(k * s) + b1 * np.sin(k * s) + c1 + d1 * t
    u2 = a2 * np.cos(k * s) + b2 * np.sin(k * s) + c2 + d2 * t
    # constant frame: 3D residualize of U*cos etc. with U=e_x, V=e_y equals scalar residualize of cos/sin
    U = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    V = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    q = (s - s[0]) / (s[-1] - s[0])
    J = np.arange(n)
    W, info = build_processed_design(U_full=U, V_full=V, s_phase_full=s, q_full=q, k=k, J=J)
    assert info["ok"]
    fit = fit_joint_sin_trend(u1, u2, t, W)
    assert fit["ok"]
    for got, want in zip(fit["coef"][:4], [a1, b1, a2, b2]):
        assert abs(got - want) < 1e-6
    assert fit["SSE_full"] <= fit["SSE_tr"] + 1e-9
    assert fit["SSE_tr"] <= fit["SST"] + 1e-9


def test_search_selects_true_k_on_grid():
    n = 100
    s = np.linspace(0, 30.0, n)
    k_true = 2 * np.pi / 15.0
    a1, b1, a2, b2 = 0.8, 0.0, 0.0, 0.6
    t = s - s.mean()
    u1 = a1 * np.cos(k_true * s) + b1 * np.sin(k_true * s)
    u2 = a2 * np.cos(k_true * s) + b2 * np.sin(k_true * s)
    U = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    V = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    q = (s - s[0]) / (s[-1] - s[0])
    J = np.arange(n)
    ks = np.array([k_true * 0.7, k_true, k_true * 1.3])
    out = search_k_forward_3d(
        u1=u1, u2=u2,
        s_phase_full=s, s_phase_trim=s, s_trend_trim=s,
        q_full=q, U_full=U, V_full=V, J=J,
        ks=ks, k_min=float(ks[0]), k_max=float(ks[-1]), dk=float(ks[1] - ks[0]),
    )
    assert out["status"] == "ok"
    assert out["selected_index"] == 1
    assert abs(out["k_hat"] - k_true) < 1e-12


def test_core_partial_r2_gate():
    base = {
        "fit_status": "ok",
        "interval_eligible": 1,
        "R2": 0.58,
        "partial_R2": 0.55,
        "lambda_hat_mm": 20.0,
        "k_hat_mm": 0.3,
        "amp_comb_mm": 0.5,
        "coef_uU_cos_mm": 0.1,
        "coef_uU_sin_mm": 0.1,
        "coef_uV_cos_mm": 0.1,
        "coef_uV_sin_mm": 0.1,
        "coverage_hat": 2.0,
        "boundary_exact_low": 0,
        "boundary_exact_high": 0,
        "L_trimmed_mm": 40.0,
    }
    assert core_periodic_from_row(base)
    fail = dict(base)
    fail["partial_R2"] = 0.40
    fail["R2"] = 0.94
    assert not core_periodic_from_row(fail)
    edge = dict(base)
    edge["partial_R2"] = 0.50
    assert partial_r2_ok(edge)
    assert core_periodic_from_row(edge)
    # N=1.5 rounding ok; 1.49 not
    cov_ok = dict(base)
    cov_ok["coverage_hat"] = 1.5
    assert core_periodic_from_row(cov_ok)
    cov_bad = dict(base)
    cov_bad["coverage_hat"] = 1.49
    assert not core_periodic_from_row(cov_bad)
    # boundary excluded
    bnd = dict(base)
    bnd["boundary_exact_high"] = 1
    assert not core_periodic_from_row(bnd)


def test_wavelength_conversion_ratio():
    # straight-ish quadratic with nearly constant speed
    coef = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # P0 = c1*q with c1=(10,0,0), s in [0,10] -> q=s/10, dP/dq=c1, L=10
    integ = baseline_arclength_between(coef, 0.0, 10.0, 1.0, 9.0)
    assert integ["ok"]
    assert abs(integ["L_base"] - 8.0) < 1e-6
    conv = effective_baseline_wavelength(lambda_s=4.0, L_obs=8.0, L_base=integ["L_base"])
    assert conv["ok"]
    assert abs(conv["lambda_base_eff"] - 4.0) < 1e-9
    assert abs(conv["L_base_over_L_obs"] - 1.0) < 1e-9


if __name__ == "__main__":
    _failed = []
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _exc:  # noqa: BLE001
                _failed.append((_name, _exc))
                print(f"FAIL {_name}: {_exc}")
    raise SystemExit(1 if _failed else 0)
