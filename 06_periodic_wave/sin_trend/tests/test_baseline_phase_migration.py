#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: test baseline phase migration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_band import build_search_band, geometry_longwave_2D_band  # noqa: E402
from sin_trend_classify import coverage_ok, classify_row  # noqa: E402
from sin_trend_forward_fit import search_k_forward_3d  # noqa: E402
from sin_trend_measure import default_measurement_config, measure_interval  # noqa: E402
from sin_trend_paths import MEASUREMENT_REVISION  # noqa: E402
from sin_trend_wavelength_coordinates import (  # noqa: E402
    baseline_arclength_between,
    cumulative_baseline_arclength,
)


def test_straight_baseline_arclength_matches_analytic():
    # P0 = (s, 0, 0): coef rows [c2,c1,c0], c1=(L,0,0), c2=c0=0
    L = 40.0
    coef = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [0.0, 0.0, 0.0]])
    s_obs = np.linspace(0.0, L, 81)
    pack = cumulative_baseline_arclength(coef, s_obs, s_first=0.0, s_last=L)
    assert pack["ok"], pack.get("status_reason")
    np.testing.assert_allclose(pack["s_base_full"], s_obs, atol=1e-8)
    rem = 10
    J = np.arange(rem, len(s_obs) - rem)
    L_base = float(pack["s_base_full"][J[-1]] - pack["s_base_full"][J[0]])
    expect = float(s_obs[J[-1]] - s_obs[J[0]])
    assert abs(L_base - expect) < 1e-8


def test_curved_baseline_trim_phase_origin():
    # strong curvature: P0(q)=c2 q^2+c1 q, coef rows = [c2,c1,c0]
    s_first, s_last = 0.0, 10.0
    coef = np.array([[0.0, 40.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    s_obs = np.linspace(0.0, 10.0, 51)
    pack = cumulative_baseline_arclength(coef, s_obs, s_first=s_first, s_last=s_last)
    assert pack["ok"], pack.get("status_reason")
    full = baseline_arclength_between(coef, s_first, s_last, 0.0, 10.0)
    assert full["ok"]
    assert abs(float(pack["L_base_full"]) - float(full["L_base"])) < 1e-5
    rem = 5
    J = np.arange(rem, len(s_obs) - rem)
    s_base = pack["s_base_full"]
    s_phase = s_base - float(s_base[J[0]])
    L_base = float(s_base[J[-1]] - s_base[J[0]])
    assert abs(float(s_phase[J[0]])) < 1e-12
    assert abs(float(s_phase[J[-1]]) - L_base) < 1e-10
    L_obs = float(s_obs[J[-1]] - s_obs[J[0]])
    assert abs(L_base - L_obs) > 1.0, (L_base, L_obs)
    # independent trim-range integral
    trim_int = baseline_arclength_between(
        coef, s_first, s_last, float(s_obs[J[0]]), float(s_obs[J[-1]])
    )
    assert trim_int["ok"]
    assert abs(L_base - float(trim_int["L_base"])) < 1e-5


def test_grid_uses_L_base_not_L_trim_scaling():
    L_base, L_trim, D = 25.0, 40.0, 3.0
    band = geometry_longwave_2D_band(
        diameter_ref_mm=D, voxel_size_mm=0.5, L_fit_mm=L_base, N_cyc_search=0.5, q_fine=8
    )
    assert band["ok"]
    assert abs(band["L_fit_mm"] - L_base) < 1e-12
    assert abs(band["k_min_mm"] - (np.pi / L_base)) < 1e-12
    assert abs(band["dk_mm"] - (np.pi / (8.0 * L_base))) < 1e-12
    assert abs(band["k_max_mm"] - (np.pi / D)) < 1e-12
    assert abs(band["lambda_max_search_mm"] - (2.0 * L_base)) < 1e-12
                                                                           
    band_trim = geometry_longwave_2D_band(
        diameter_ref_mm=D, voxel_size_mm=0.5, L_fit_mm=L_trim, N_cyc_search=0.5, q_fine=8
    )
    ratio = L_trim / L_base
    assert abs(band["k_min_mm"] / band_trim["k_min_mm"] - ratio) < 1e-10
    # k_max is from D only — identical, not scaled by L ratio
    assert abs(band["k_max_mm"] - band_trim["k_max_mm"]) < 1e-12


def test_forward_recovers_k_on_baseline_phase_with_base_trend():
    from sin_trend_forward_fit import build_processed_design

    n = 160
    s_obs = np.linspace(0.0, 40.0, n)
    s_base = s_obs + 0.02 * s_obs**2  # nonlinear, strictly increasing
    rem = 10
    J = np.arange(rem, n - rem)
    s_phase_full = s_base - float(s_base[J[0]])
    s_phase_trim = s_phase_full[J]
    s_trend = s_phase_trim  # trend also on baseline phase
    L_base = float(s_phase_trim[-1] - s_phase_trim[0])
    k_true = 2.0 * np.pi / (L_base / 2.0)
    coef_true = np.array([0.6, -0.25, 0.15, 0.4, 0.05, 0.002, -0.02, -0.001])
    U = np.tile([1.0, 0.0, 0.0], (n, 1))
    V = np.tile([0.0, 1.0, 0.0], (n, 1))
    q = (s_obs - s_obs[0]) / (s_obs[-1] - s_obs[0])
    W, winfo = build_processed_design(
        U_full=U, V_full=V, s_phase_full=s_phase_full, q_full=q, k=k_true, J=J
    )
    assert winfo["ok"] and W is not None
    t = s_trend - float(np.mean(s_trend))
    n_trim = len(J)
    Z = np.zeros((2 * n_trim, 4))
    Z[:n_trim, 0] = 1.0
    Z[:n_trim, 1] = t
    Z[n_trim:, 2] = 1.0
    Z[n_trim:, 3] = t
    X = np.hstack([W, Z])
    y = X @ coef_true
    u1, u2 = y[:n_trim], y[n_trim:]
    ks = np.array([k_true * 0.75, k_true, k_true * 1.25])
    out = search_k_forward_3d(
        u1=u1,
        u2=u2,
        s_phase_full=s_phase_full,
        s_phase_trim=s_phase_trim,
        s_trend_trim=s_trend,
        q_full=q,
        U_full=U,
        V_full=V,
        J=J,
        ks=ks,
        k_min=float(ks[0]),
        k_max=float(ks[-1]),
        dk=float(ks[1] - ks[0]),
    )
    assert out["status"] == "ok"
    assert out["selected_index"] == 1
    assert abs(out["k_hat"] - k_true) < 1e-6
    assert abs(float(out["t_mean_s"]) - float(np.mean(s_trend))) < 1e-12
    assert out.get("trend_coordinate") == "baseline_arclength"
    fit = out["fit"]
    for i, key in enumerate(("a1", "b1", "a2", "b2")):
        assert abs(fit[key] - coef_true[i]) < 1e-6
    np.testing.assert_allclose(
        fit["response_U"] + fit["trend_U"], fit["yhat_U"], atol=1e-10
    )
                                 
    out_lin = search_k_forward_3d(
        u1=u1,
        u2=u2,
        s_phase_full=s_obs - float(s_obs[J[0]]),
        s_phase_trim=s_obs[J] - float(s_obs[J[0]]),
        s_trend_trim=s_obs[J] - float(s_obs[J[0]]),
        q_full=q,
        U_full=U,
        V_full=V,
        J=J,
        ks=ks,
        k_min=float(ks[0]),
        k_max=float(ks[-1]),
        dk=float(ks[1] - ks[0]),
    )
    assert out_lin["selected_index"] != 1 or abs(out_lin["k_hat"] - k_true) > 1e-3 or out_lin["SSE_full"] > out["SSE_full"] * 1.01


def test_coverage_fallback_uses_L_base_not_L_trim():
    row = {
        "coverage_hat": np.nan,
        "L_base_mm": 30.0,
        "L_trimmed_mm": 50.0,
        "lambda_hat_mm": 20.0,
        "fit_status": "ok",
        "interval_eligible": 1,
        "partial_R2": 0.6,
        "R2": 0.7,
        "boundary_exact_low": 0,
        "boundary_exact_high": 0,
        "amp_comb_mm": 0.5,
        "coef_uU_cos_mm": 0.1,
        "coef_uU_sin_mm": 0.1,
        "coef_uV_cos_mm": 0.1,
        "coef_uV_sin_mm": 0.1,
        "k_hat_mm": 0.3,
    }
    # L_base/λ = 1.5 → ok; L_trim/λ = 2.5 would also pass but we must use L_base
    assert coverage_ok(row)
    row_short = dict(row)
    row_short["L_base_mm"] = 20.0  # 20/20=1.0 < 1.5; L_trim would still be 2.5
    assert not coverage_ok(row_short)


def test_measure_interval_baseline_phase_end_to_end():
    n = 120
    x = np.linspace(0, 40.0, n)
    k = 2 * np.pi / 20.0
    P = np.column_stack([x, 0.5 * np.sin(k * x), 0.25 * np.cos(k * x)])
    cfg = default_measurement_config()
    cfg["verbose"] = False
    r = np.full(n, 2.0)
    res = measure_interval(P, frame_reference="baseline", cfg=cfg, radius_vox=r, compute_pca=False)
    row = res["row"]
    pw = res["pointwise"]
    assert row["measurement_revision"] == MEASUREMENT_REVISION
    assert row["phase_coordinate"] == "baseline_arclength"
    assert row["trend_coordinate"] == "baseline_arclength"
    assert row["grid_length_basis"] == "trimmed_baseline_arclength"
    assert np.allclose(pw["s_trend_trim"], pw["s_phase_trim"])
    assert np.isfinite(row["L_base_mm"]) and row["L_base_mm"] > 0
    assert abs(row["L_baseline_fit_mm"] - row["L_base_mm"]) < 1e-12
    if row["fit_status"] == "ok":
        assert np.isfinite(row["lambda_hat_mm"])
        assert abs(row["coverage_hat"] - row["L_base_mm"] / row["lambda_hat_mm"]) < 1e-8
        expect_sl = row["lambda_hat_mm"] / (2.0 * row["radius_median_mm"])
        assert abs(row["slenderness"] - expect_sl) < 1e-12
        assert abs(row["lambda_baseline_eff_mm"] - row["lambda_hat_mm"]) < 1e-12
    assert "s_phase_trim" in pw and "s_obs_trim" in pw and "s_base_full" in pw
    assert abs(float(pw["s_phase_trim"][0])) < 1e-9
    assert abs(float(pw["s_phase_trim"][-1]) - float(row["L_base_mm"]) / float(cfg["voxel_size"])) < 1e-6
    classified = classify_row(row)
    if np.isfinite(row.get("coverage_hat", np.nan)):
        # reclassify must not flip coverage basis to L_trim
        assert abs(float(classified["coverage_hat"]) - float(row["coverage_hat"])) < 1e-12


def test_build_search_band_alias_L_fit():
    cfg = default_measurement_config()
    band = build_search_band(cfg, L_fit_mm=30.0, L_trimmed_mm=45.0, diameter_ref_mm=2.0)
    assert band["ok"]
    assert abs(band["L_fit_mm"] - 30.0) < 1e-12
    assert band["grid_length_basis"] == "trimmed_baseline_arclength"


def test_fitting_coordinate_matches_phase_not_old_observed():
    from sin_trend_frames import FITTING_COORDINATE

    cfg = default_measurement_config()
    assert cfg["fitting_coordinate"] == "baseline_arclength"
    assert cfg["phase_coordinate"] == "baseline_arclength"
    assert cfg["trend_coordinate"] == "baseline_arclength"
    assert FITTING_COORDINATE == "baseline_arclength"
    assert FITTING_COORDINATE != "observed_arclength_s_trimmed_zero_start"

    n = 80
    x = np.linspace(0, 40.0, n)
    k = 2 * np.pi / 20.0
    P = np.column_stack([x, 0.4 * np.sin(k * x), 0.2 * np.cos(k * x)])
    cfg["verbose"] = False
    row = measure_interval(
        P, frame_reference="baseline", cfg=cfg, radius_vox=np.full(n, 2.0), compute_pca=False
    )["row"]
    assert row["fitting_coordinate"] == "baseline_arclength"
    assert row["phase_coordinate"] == "baseline_arclength"
    assert row["fitting_coordinate"] == row["phase_coordinate"]


def test_duplicate_s_obs_preserves_cumulative_baseline_arclength():
    L = 10.0
    # coef rows [c2,c1,c0]: straight P0=(L q, 0, 0)
    coef = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [0.0, 0.0, 0.0]])
    s_obs = np.array([0.0, 2.0, 2.0, 5.0, 10.0])  # duplicate at index 2
    pack = cumulative_baseline_arclength(coef, s_obs, s_first=0.0, s_last=L)
    assert pack["ok"], pack.get("status_reason")
    ell = pack["s_base_full"]
    assert abs(float(ell[1]) - 2.0) < 1e-8
    assert abs(float(ell[2]) - float(ell[1])) < 1e-12                 
    assert abs(float(ell[3]) - 5.0) < 1e-8
    assert abs(float(ell[-1]) - 10.0) < 1e-8

                                                                 
    n = 60
    x = np.linspace(0, 30.0, n)
    k = 2 * np.pi / 15.0
    P = np.column_stack([x, 0.3 * np.sin(k * x), 0.0 * x])
    P_dup = np.vstack([P[:30], P[29:30], P[30:]])         
    cfg = default_measurement_config()
    cfg["verbose"] = False
    row = measure_interval(
        P_dup,
        frame_reference="baseline",
        cfg=cfg,
        radius_vox=np.full(len(P_dup), 2.0),
        compute_pca=False,
    )["row"]
    assert row.get("fit_status") != "baseline_arclength_failed", row.get("fit_status_reason")
    assert np.isfinite(row.get("L_base_mm", np.nan)) and float(row["L_base_mm"]) > 0


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
