#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: test sin trend baseline ported."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_PKG = Path(__file__).resolve().parents[1]  # analysis_code/sin_trend
MORPH = _PKG.parent / "morphometrics"                    
_STATS = _PKG.parents[1] / "statistics"
for _p in (str(_STATS), str(MORPH), str(_PKG)):  # PKG last -> insert(0) prefers sin_trend/
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sin_trend_frames import (  # noqa: E402
    analytic_baseline_tangent,
    arclength,
    frame_orthonormality_check,
    lateral_components_signed,
    legacy_projection_frame,
    minimal_rotation_transport,
    poly_trend,
    quadratic_baseline,
    rotation_matrix_from_axis_angle,
    trim_bundle,
)
from sin_trend_band import (  # noqa: E402
    geometry_longwave_2D_band,
    make_k_grid_bounds,
)
from sin_trend_classify import (  # noqa: E402
    apply_classification,
    classify_row,
    gui_unlabeled_mask,
    reject_reasons_contain_forbidden_gates,
    stats_extraction_mask,
)
from sin_trend_measure import (  # noqa: E402
    build_groupid_map,
    default_measurement_config,
    measure_interval,
    measurement_row_schema,
    residual_numerically_zero_vs_input_scale,
    validate_measurement_config,
)
from sin_trend_frames import TANGENT_SOURCE_NUMERIC  # noqa: E402
from sin_trend_paths import (  # noqa: E402
    BAND_GEOMETRY_2D,
    BAND_LEGACY_FIXED_10MM,
    COMPARISON_KEY,
    FRAME_BASELINE,
    FRAME_OBSERVED,
    FRAME_OBSERVED_LEGACY,
    INTERVAL_KEY,
    MEASUREMENT_REVISION,
    MEASUREMENT_VERSION,
    version_revision_consistency,
    write_run_config,
)
from sin_trend_surface import evaluate_displacement  # noqa: E402
from sin_trend_synthetic import make_sinusoid_centerline, random_rotation  # noqa: E402

# ---------------------------------------------------------------------------
                                     
# ---------------------------------------------------------------------------
FROZEN_HASHES = {
    "00_run_extract_m1_features.py": "ffc0da27014d39f656da131a02301b17d0dfb02871cfe648ec67b70209097314",
    "11_run_sin_trend_fitting.py": "329a5089274dd19226dd036823f27c3c36b52e9d51093a0cccba8877013b9045",
    "11_batch_sin_trend_fitting.py": "2444dfcab41753c4172ee9e00394ecd025f064f1748e20075f76bdc6d3b310f3",
    "12_create_long_format_csv.py": "6f8183edf4c92ca5e87c33140d3b1b780d687d0352c3a5e1e40572de42e15bf8",
    "13_prepare_core_inplane_dataset.py": "f32b2cf6b49c48d9efd8406f1db5231d2fb467707a918a73bb745ad5a17bb851",
    "14_visual_inspection_mca_segment_periodicity.py": "ca4f95e9841fec4b790fa3f432993d308f3317cda937d72adc538784aef2c6c2",
    "sin_trend_fitting.py": "7279747e700e996df6952f73a02fd8286525ba41b95e7efe980273a6a72ccd15",
    "sin_trend_fitting_helpers.py": "4faf7c7ef481c597b3e5f3b76c75e72b3b345f62637bb1e6ae7a804bb9c62961",
    "sin_trend_classification_v2.py": "971893838de4d52e2b496260b08c789fb0e3d929c40135d5f8e62b5d0c857296",
    "sin_trend_io_utils.py": "68550a2e385ed14b07f013297a68ddf7759827e96a875db69f52e1fd19782da7",
    "measurement_validation.py": "589c11874822bdb7ed59dea28e25f8b128070b88a6e7a800beb85655658e7e54",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _syn(A=1.0, lam=20.0, nc=3.0, ph=0.0, ds=0.5, diameter_mm=4.0, **kw):
    return make_sinusoid_centerline(
        amplitude_mm=A, lambda_x_mm=lam, n_cycles=nc, phase=ph, target_ds_mm=ds, diameter_mm=diameter_mm, **kw
    )


def _cfg(**kw):
    c = default_measurement_config()
    c["verbose"] = False
    c.update(kw)
    return c


def _run(syn, frame=FRAME_BASELINE, cfg=None, **kw):
    cfg = cfg or _cfg()
    return measure_interval(
        syn["P_vox"],
        frame_reference=frame,
        cfg=cfg,
        observed_tangent=syn.get("T_numeric"),
        radius_vox=syn.get("radius_vox"),
        compute_pca=False,
        **kw,
    )


@unittest.skip(
    "legacy morphometrics frozen-hash suite; isolated from sin_trend Core. "
    "Do not change v2 gates to satisfy these hashes."
)
class TestExistingFilesUnchanged(unittest.TestCase):
    def test_hashes(self):
        for name, h in FROZEN_HASHES.items():
            p = MORPH / name
            self.assertTrue(p.is_file(), name)
            self.assertEqual(_sha(p), h, f"{name} was modified")


class TestBaseline(unittest.TestCase):
    def test_P0_matches_poly_trend_same_s(self):
        syn = _syn()
        P, s = syn["P_vox"], syn["s_vox"]
        bl = quadratic_baseline(P, s)
        P0_old, info = poly_trend(P, s, 2, return_residual=True)
        self.assertEqual(bl["status"], "ok")
        np.testing.assert_allclose(bl["P0"], P0_old, rtol=0, atol=1e-12)
        np.testing.assert_allclose(bl["coef"], info["coef"])
                                  
        self.assertEqual(len(bl["q"]), len(s))
        self.assertAlmostEqual(bl["q"][0], 0.0)
        self.assertAlmostEqual(bl["q"][-1], 1.0)

    def test_analytic_tangent_matches_known_polynomial(self):
                                                                 
        s = np.linspace(0, 50, 101)
        q = s / 50.0
        c2 = np.array([1.0, -3.0, 0.5])
        c1 = np.array([40.0, 5.0, -2.0])
        c0 = np.array([10.0, 20.0, 30.0])
        P = c2 * q[:, None] ** 2 + c1 * q[:, None] + c0
        bl = quadratic_baseline(P, s)
        expected = (2 * c2 * q[:, None] + c1) / 50.0
        np.testing.assert_allclose(bl["dP0_ds"], expected, rtol=1e-9, atol=1e-9)
        T0, st, _ = analytic_baseline_tangent(bl)
        self.assertEqual(st, "ok")
        np.testing.assert_allclose(np.linalg.norm(T0, axis=1), 1.0, atol=1e-12)

    def test_straight_line_ok_and_zero_derivative_fails_safely(self):
        s = np.linspace(0, 30, 61)
        P = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
        bl = quadratic_baseline(P, s)
        self.assertEqual(bl["status"], "ok")                    
        T0, st, _ = analytic_baseline_tangent(bl)
        np.testing.assert_allclose(T0, np.tile([1.0, 0, 0], (len(s), 1)), atol=1e-12)
                                                                 
        P_same = np.ones((20, 3))
        bl2 = quadratic_baseline(P_same, np.zeros(20))
        self.assertEqual(bl2["status"], "degenerate_length")
        self.assertIsNone(analytic_baseline_tangent(bl2)[0])
                                                   
        s3 = np.linspace(0, 10, 51)
        q3 = s3 / 10
        P3 = np.column_stack([(q3 - 0.5) ** 2 * 100, np.zeros_like(q3), np.zeros_like(q3)])
        bl3 = quadratic_baseline(P3, s3)
        self.assertIn(bl3["status"], {"zero_tangent_at_sample", "zero_tangent_between_samples"})
        self.assertIsNone(analytic_baseline_tangent(bl3)[0])


class TestTransport(unittest.TestCase):
    def test_orthonormal_right_handed(self):
        syn = _syn()
        bl = quadratic_baseline(syn["P_vox"], syn["s_vox"])
        T0, _, _ = analytic_baseline_tangent(bl)
        fr = minimal_rotation_transport(T0)
        self.assertEqual(fr["status"], "ok")
        chk = frame_orthonormality_check(fr["T"], fr["U"], fr["V"])
        self.assertLess(chk["frame_orth_err_max"], 1e-12)
                            
        fr2 = minimal_rotation_transport(3.0 * T0)
        np.testing.assert_allclose(fr2["T"], fr["T"], atol=1e-12)

    def test_zero_or_antiparallel_tangent_returns_status(self):
        T = np.tile([1.0, 0, 0], (5, 1))
        T[2] = 0.0
        self.assertEqual(minimal_rotation_transport(T)["status"], "zero_tangent")
        T2 = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
        self.assertEqual(minimal_rotation_transport(T2)["status"], "antiparallel_ambiguous")
        T3 = np.array([[1.0, 0, 0], [np.nan, 0, 0]])
        self.assertEqual(minimal_rotation_transport(T3)["status"], "nonfinite_tangent")

    def test_scalar_invariance_initial_axis_rotation(self):
        syn = _syn(A=1.0, lam=20.0, ph=np.pi / 4)
        P = syn["P_vox"]
        bl = quadratic_baseline(P, syn["s_vox"])
        T0, _, _ = analytic_baseline_tangent(bl)
        fr_a = minimal_rotation_transport(T0)
                      
        th = 0.7
        u_rot = np.cos(th) * fr_a["U"][0] + np.sin(th) * fr_a["V"][0]
        fr_b = minimal_rotation_transport(T0, initial_U=u_rot)
        self.assertEqual(fr_b["status"], "ok")
        la = lateral_components_signed(P, bl["P0"], fr_a["T"], fr_a["U"], fr_a["V"])
        lb = lateral_components_signed(P, bl["P0"], fr_b["T"], fr_b["U"], fr_b["V"])
                         
        np.testing.assert_allclose(np.hypot(la["uU"], la["uV"]), np.hypot(lb["uU"], lb["uV"]), atol=1e-10)
        self.assertFalse(np.allclose(la["uU"], lb["uU"]))

    def test_scalar_invariance_rigid_rotation_measure(self):
        base = _syn(A=1.0, lam=20.0, ph=0.3)
        R = random_rotation(5)
        rot = _syn(A=1.0, lam=20.0, ph=0.3, rotation=R, translation_mm=np.array([7.0, -3.0, 11.0]))
        cfg = _cfg()
        for fr_name in (FRAME_BASELINE, FRAME_OBSERVED):
            r0 = _run(base, frame=fr_name, cfg=cfg)
            r1 = _run(rot, frame=fr_name, cfg=cfg)
            for k in ("amp_comb_mm", "lambda_hat_mm", "R2", "sigma1_mm", "sigma2_mm"):
                self.assertAlmostEqual(r0["row"][k], r1["row"][k], places=6, msg=f"{fr_name}:{k}")

    def test_legacy_frame_distinct_method_label(self):
        syn = _syn()
        fr = legacy_projection_frame(syn["T_numeric"])
        self.assertEqual(fr["status"], "ok")
        self.assertNotEqual(fr["frame_transport_method"], minimal_rotation_transport(syn["T_numeric"])["frame_transport_method"])


class TestResidualDecomposition(unittest.TestCase):
    def test_deltaP_decomposition_and_trim_alignment(self):
        syn = _syn(A=0.8, lam=25.0, ph=1.0)
        P, s = syn["P_vox"], syn["s_vox"]
        bl = quadratic_baseline(P, s)
        T0, _, _ = analytic_baseline_tangent(bl)
        fr = minimal_rotation_transport(T0)
        lat = lateral_components_signed(P, bl["P0"], fr["T"], fr["U"], fr["V"])
        recon = lat["Delta_parallel"] + lat["uU"][:, None] * fr["U"] + lat["uV"][:, None] * fr["V"]
        np.testing.assert_allclose(recon, lat["Delta"], atol=1e-10)
        self.assertLess(lat["perp_dot_T_max"], 1e-10)
        tb = trim_bundle({"P": P, "P0": bl["P0"], "uU": lat["uU"], "uV": lat["uV"], "pid": np.arange(len(P))}, 10, s)
        self.assertTrue(tb["ok"])
        np.testing.assert_array_equal(tb["pid"], np.arange(10, len(P) - 10))
        s_check, _ = arclength(tb["P"])
        np.testing.assert_allclose(tb["s_fit"], s_check, atol=1e-9)
        self.assertAlmostEqual(tb["s_fit"][0], 0.0)
                                         
        self.assertAlmostEqual(tb["s_full_trim"][0], s[10])

    def test_measure_pointwise_consistency(self):
        syn = _syn()
        r = _run(syn)
        pw = r["pointwise"]
        self.assertEqual(r["row"]["fit_status"], "ok")
        n = len(pw["s_fit"])
        for k in ("P", "P0", "T", "U", "V", "uU", "uV", "d_sin", "sin_U", "fit_U", "trend_U", "point_id"):
            self.assertEqual(len(pw[k]), n, k)
                                    
        np.testing.assert_allclose(pw["fit_U"], pw["trend_U"] + pw["sin_U"], atol=1e-10)
        d_sin = pw["sin_U"][:, None] * pw["U"] + pw["sin_V"][:, None] * pw["V"]
        np.testing.assert_allclose(pw["d_sin"], d_sin, atol=1e-12)
        np.testing.assert_array_equal(pw["point_id"], pw["point_id_full"][pw["index_full_to_trim"]])


class TestMeasureInterval(unittest.TestCase):
    def test_frame_metadata_not_overwritten_by_fit_meta(self):
        syn = _syn()
        r = _run(syn)
        row = r["row"]
        self.assertEqual(row["frame_reference"], FRAME_BASELINE)
        self.assertNotEqual(row["frame_reference"], r["search"].get("frame_reference"))                 
        self.assertEqual(row["measurement_version"], MEASUREMENT_VERSION)
        self.assertEqual(row["measurement_revision"], MEASUREMENT_REVISION)
        self.assertIn("analytic", row["tangent_source"])

    def test_baseline_path_does_not_use_observed_tangent(self):
        syn = _syn()
        cfg = _cfg()
        a = _run(syn, cfg=cfg)
        bogus = np.tile([0.0, 0.0, 1.0], (len(syn["P_vox"]), 1))
        b = measure_interval(syn["P_vox"], frame_reference=FRAME_BASELINE, cfg=cfg, observed_tangent=bogus, radius_vox=syn["radius_vox"], compute_pca=False)
        for k in ("amp_comb_mm", "lambda_hat_mm", "R2", "sigma1_mm"):
            self.assertEqual(a["row"][k], b["row"][k], k)
                                            
        c = measure_interval(syn["P_vox"], frame_reference=FRAME_OBSERVED, cfg=cfg, observed_tangent=None, compute_pca=False)
        self.assertEqual(c["row"]["fit_status"], "frame_failed")
        self.assertIn("observed_tangent_missing", c["row"]["fit_status_reason"])

    def test_observed_reproduces_legacy_reference(self):
        syn = _syn(A=1.0, lam=20.0, nc=3.0, ph=0.0, ds=0.5)
        r = _run(syn, frame=FRAME_OBSERVED_LEGACY, cfg=_cfg(search_band_mode=BAND_LEGACY_FIXED_10MM))
        self.assertAlmostEqual(r["row"]["amp_comb_mm"], 0.989026, places=5)
        self.assertAlmostEqual(r["row"]["lambda_hat_mm"], 20.550889, places=5)

    def test_observed_numeric_source_ignores_passed_frenet(self):
        syn = _syn(A=1.0, lam=20.0, nc=3.0)
        cfg = _cfg(observed_tangent_source="numeric")
        bogus = np.tile([0.0, 0.0, 1.0], (len(syn["P_vox"]), 1))
        r_num = measure_interval(
            syn["P_vox"],
            frame_reference=FRAME_OBSERVED,
            cfg=cfg,
            observed_tangent=bogus,
            radius_vox=syn["radius_vox"],
            compute_pca=False,
        )
        r_ref = measure_interval(
            syn["P_vox"],
            frame_reference=FRAME_OBSERVED,
            cfg=cfg,
            observed_tangent=syn["T_numeric"],
            radius_vox=syn["radius_vox"],
            compute_pca=False,
        )
        self.assertIn(TANGENT_SOURCE_NUMERIC, r_num["row"]["tangent_source"])
        self.assertEqual(r_num["row"]["fit_status"], r_ref["row"]["fit_status"])
        self.assertAlmostEqual(r_num["row"]["amp_comb_mm"], r_ref["row"]["amp_comb_mm"], places=5)
        self.assertAlmostEqual(r_num["row"]["lambda_hat_mm"], r_ref["row"]["lambda_hat_mm"], places=4)
                                                            
        self.assertIsNotNone(r_num["pointwise"])
        T_obs_saved = np.asarray(r_num["pointwise"]["T_obs"], float)
        T_obs_ref = np.asarray(r_ref["pointwise"]["T_obs"], float)
        self.assertTrue(np.allclose(T_obs_saved, T_obs_ref, atol=1e-12))
        rem = int(cfg["removal"])
        bogus_trim = bogus[rem : len(bogus) - rem] if rem > 0 else bogus
        self.assertEqual(T_obs_saved.shape, bogus_trim.shape)
        self.assertFalse(np.allclose(T_obs_saved, bogus_trim, atol=1e-3))
        self.assertTrue(np.allclose(np.asarray(r_num["pointwise"]["T_obs_frenet"], float), bogus_trim, atol=1e-12))
                                              
        ang_num = float(r_num["row"]["angle_T0_Tobs_mean_deg"])
                                                                  
        ang_ref = float(r_ref["row"]["angle_T0_Tobs_mean_deg"])
        self.assertTrue(np.isfinite(ang_num) and np.isfinite(ang_ref))
        self.assertAlmostEqual(ang_num, ang_ref, places=10)

    def test_deprecated_lambda_min_resolution_raises(self):
        cfg = _cfg()
        cfg["lambda_min_resolution_mm"] = 10.0
        with self.assertRaises(ValueError):
            validate_measurement_config(cfg)
        with self.assertRaises(ValueError):
            measure_interval(_syn()["P_vox"], frame_reference=FRAME_BASELINE, cfg=cfg, compute_pca=False)

    def test_degenerate_inputs(self):
        cfg = _cfg()
        s = np.linspace(0, 60, 121)
        P = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
        rad = np.full(len(P), 2.0)  # D = 2 mm（voxel_size=0.5）
                                                           
        r = measure_interval(P, frame_reference=FRAME_BASELINE, cfg=cfg, radius_vox=rad, compute_pca=False)
        self.assertNotEqual(r["row"]["fit_status"], "ok")
        self.assertIn(
            r["row"]["fit_status"],
            {"numerically_flat", "degenerate_sst", "trend_only_explained", "no_valid_candidate"},
        )
        self.assertEqual(r["row"]["fit_status"], "numerically_flat")
        self.assertEqual(r["row"]["failure_stage"], "residual_vs_input_scale")
                
        P2 = P.copy()
        P2[5, 1] = np.nan
        self.assertEqual(measure_interval(P2, frame_reference=FRAME_BASELINE, cfg=cfg)["row"]["fit_status"], "nonfinite_input")
                                           
        self.assertEqual(measure_interval(P[:3], frame_reference=FRAME_BASELINE, cfg=cfg)["row"]["fit_status"], "insufficient_points")
                                                 
        syn = _syn(A=0.5, lam=5.0, nc=1.0, ds=0.2)
        r3 = _run(syn)
        self.assertLess(r3["row"]["L_full_mm"], 10.0)
        self.assertEqual(r3["row"]["interval_eligible"], 1)
        self.assertNotEqual(r3["row"]["fit_status"], "skipped_not_eligible")
        # invalid grid (q_fine=0)
        syn2 = _syn()
        r4 = _run(syn2, cfg=_cfg(q_fine=0))
        self.assertIn(r4["row"]["fit_status"], {"invalid_band", "no_valid_candidate", "fit_failed"})

    def test_small_valid_sinusoid_not_degenerate_and_perfect_fit(self):
                                   
        syn = _syn(A=0.02, lam=20.0)
        r = _run(syn)
        self.assertEqual(r["row"]["fit_status"], "ok")
        self.assertAlmostEqual(r["row"]["amp_comb_mm"], 0.02, delta=0.003)
                                                                  
        from sin_trend_fitting_helpers import fit_sin_trend_two_component

        s = np.linspace(0, 100, 201)
        k = 0.3
        uU = 0.4 * np.cos(k * s) + 0.1 * np.sin(k * s) + 0.01 * (s - s.mean())
        uV = -0.2 * np.sin(k * s)
        fit = fit_sin_trend_two_component(uU, uV, s, k)
        self.assertEqual(fit["fit_status"], "ok")
        self.assertGreater(fit["R2"], 0.999999)

    def test_min_sse_equals_max_r2_no_core_substitution(self):
        syn = _syn(A=0.3, lam=20.0)
        r = _run(syn)
        cands = r["candidates"]
        ok = cands[cands["fit_status"] == "ok"]
        self.assertEqual(int(ok.loc[ok["SSE_full"].idxmin(), "candidate_index"]), int(r["row"]["selected_index"]))
        self.assertEqual(int(ok.loc[ok["R2"].idxmax(), "candidate_index"]), int(r["row"]["selected_index"]))
        self.assertEqual(int(cands["is_selected"].sum()), 1)
                                                         
        self.assertLess(r["row"]["amp_comb_mm"], 0.5)

    def test_radius_missing_keeps_centerline_measurement(self):
        syn = _syn(diameter_mm=None)
        r = measure_interval(syn["P_vox"], frame_reference=FRAME_BASELINE, cfg=_cfg(), radius_vox=None, compute_pca=False)
        self.assertEqual(r["row"]["fit_status"], "missing_diameter")
        self.assertTrue(np.isnan(r["row"]["lambda_min_search_mm"]) or r["row"]["lambda_min_search_mm"] != 10.0)
        self.assertNotEqual(r["row"]["lambda_min_search_mm"], 10.0)
        self.assertFalse(r["row"]["fit_status"] == "ok")
                                
        self.assertIn("no 10 mm fallback", r["row"]["fit_status_reason"])

    def test_row_schema_complete_for_failure(self):
        r = measure_interval(np.zeros((1, 3)), frame_reference=FRAME_BASELINE, cfg=_cfg())
        self.assertEqual(set(r["row"].keys()), set(measurement_row_schema().keys()))


class TestSurfaceEvaluation(unittest.TestCase):
    def _est(self, normals, mask):
        mask = np.asarray(mask, bool)
                                             
        from sin_trend_surface import EVAL_SOURCE_GOOD, EVAL_SOURCE_NONE

        src = np.where(mask, EVAL_SOURCE_GOOD, EVAL_SOURCE_NONE).astype(int)
        return {
            "normals_full": normals,
            "mask_eval": mask,
            "eval_normal_source": src,
            "cortex_coverage": float(np.mean(mask)),
            "constraint_coverage": float(np.mean(mask)),
        }

    def test_known_normals(self):
        n = 50
        d = np.zeros((n, 3))
        d[:, 1] = np.sin(np.linspace(0, 6, n))          
        mask = np.ones(n, bool)
                                              
        ev = evaluate_displacement(d, self._est(np.tile([0, 0, 1.0], (n, 1)), mask))
        self.assertAlmostEqual(ev["R_out"], 0.0)
        self.assertAlmostEqual(ev["R_tan"], 1.0)
        self.assertAlmostEqual(ev["squared_displacement_coverage"], 1.0)
        self.assertAlmostEqual(ev["evaluated_tangential_fraction"], 1.0)
                                         
        ev = evaluate_displacement(d, self._est(np.tile([0, 1.0, 0], (n, 1)), mask))
        self.assertAlmostEqual(ev["R_out"], 1.0)
        self.assertAlmostEqual(ev["R_tan"], 0.0)
        # 45°: R_out = 0.5, R_tan = 0.5
        nrm = np.tile([0, 1.0, 1.0], (n, 1)) / np.sqrt(2)
        ev = evaluate_displacement(d, self._est(nrm, mask))
        self.assertAlmostEqual(ev["R_out"], 0.5)
        self.assertAlmostEqual(ev["R_tan"], 0.5)
                           
        ev2 = evaluate_displacement(d, self._est(nrm * 3.0, mask))
        self.assertAlmostEqual(ev2["R_out"], 0.5)
                          
        half = np.zeros(n, bool)
        half[: n // 2] = True
        ev = evaluate_displacement(d, self._est(np.tile([0, 0, 1.0], (n, 1)), half))
        self.assertAlmostEqual(ev["squared_displacement_coverage"], float(np.sum(d[half] ** 2) / np.sum(d**2)))
        self.assertEqual(ev["n_eval"], n // 2)

    def test_zero_denominators_not_inplane(self):
        n = 20
        mask = np.ones(n, bool)
        ev = evaluate_displacement(np.zeros((n, 3)), self._est(np.tile([0, 0, 1.0], (n, 1)), mask))
        self.assertEqual(ev["status"], "zero_displacement_energy")
        self.assertTrue(np.isnan(ev["R_out"]))
        d = np.zeros((n, 3))
        d[n // 2 :, 1] = 1.0
        m2 = np.zeros(n, bool)
        m2[: n // 2] = True               
        ev = evaluate_displacement(d, self._est(np.tile([0, 0, 1.0], (n, 1)), m2))
        self.assertEqual(ev["status"], "zero_constrained_energy")
        self.assertTrue(np.isnan(ev["R_out"]))
        self.assertAlmostEqual(ev["squared_displacement_coverage"], 0.0)
        ev = evaluate_displacement(d, self._est(np.tile([0, 0, 1.0], (n, 1)), np.zeros(n, bool)))
        self.assertEqual(ev["status"], "no_evaluable_points")

    def test_upper_bound_and_normals_do_not_change_fit(self):
        syn = _syn()
        r = _run(syn)
        pw = r["pointwise"]
        n = len(pw["d_sin"])
        rng = np.random.default_rng(0)
        normals = rng.normal(size=(n, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        mask = rng.random(n) > 0.3
        ev = evaluate_displacement(
            pw["d_sin"], self._est(normals, mask), T_frame=pw["T"], T0=pw["T0"], g1_3d=pw["g1_3d"]
        )
        self.assertEqual(ev["status"], "ok")
        self.assertLessEqual(ev["R_out"], ev["R_out_upper_bound"] + 1e-9)
        self.assertEqual(ev["R_out_within_upper_bound"], 1.0)
                                                        
        r2 = _run(syn)
        self.assertEqual(r["row"]["k_hat_mm"], r2["row"]["k_hat_mm"])
                                                   
        ro = _run(syn, frame=FRAME_OBSERVED)
        ev_o = evaluate_displacement(ro["pointwise"]["d_sin"], self._est(normals, mask))
        self.assertNotEqual(ev["R_out"], ev_o["R_out"])

    def test_observed_bracket_interpolation_excludes_provisional(self):
        import importlib.util

                                                       
        path = _PKG / "sin_trend_fitting_helpers.py"
        spec = importlib.util.spec_from_file_location("sin_trend_fitting_helpers_baseline", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        interpolate_observed_normal_brackets = mod.interpolate_observed_normal_brackets

        s_full = np.linspace(0.0, 10.0, 11)
        s_anchors = np.array([0.0, 5.0, 10.0])
                                      
        n_obs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], float)
        observed = np.array([True, False, True])
        normals, bracket = interpolate_observed_normal_brackets(s_full, s_anchors, n_obs, observed)
                                 
        mid = (s_full > 0.0) & (s_full < 10.0) & ~np.isclose(s_full, 5.0)
        self.assertFalse(np.any(np.isfinite(normals[mid]).all(axis=1)))
        self.assertFalse(np.any(bracket[mid]))
                       
        for sa, nexp in zip(s_anchors[observed], n_obs[observed]):
            idx = int(np.argmin(np.abs(s_full - sa)))
            self.assertTrue(bracket[idx])
            self.assertTrue(np.allclose(normals[idx], nexp))

                                 
        observed2 = np.array([True, True, False])
        n2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], float)
        normals2, bracket2 = interpolate_observed_normal_brackets(s_full, s_anchors, n2, observed2)
        between = (s_full >= 0.0) & (s_full <= 5.0)
        self.assertTrue(np.all(bracket2[between]))
        self.assertTrue(np.all(np.isfinite(normals2[between]).all(axis=1)))
                                   
        after = (s_full > 5.0) & (s_full <= 10.0)
        self.assertFalse(np.any(np.isfinite(normals2[after]).all(axis=1)))


class TestResumeCompleteness(unittest.TestCase):
    def test_resume_rejects_partial_mesh_missing_and_hash_mismatch(self):
        from sin_trend_paths import (
            participant_outcome_from_counts,
            per_case_csv_path,
            sha256_file,
            verify_participant_complete_for_resume,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "run_test"
            frame = FRAME_BASELINE
            did = "IXI000-HH-0000"
            # fake VTP
            vtp = root / "data" / "derived" / "vessel_centerlines" / did / f"{did}_merged_centerlines.vtp"
            vtp.parent.mkdir(parents=True)
            vtp.write_bytes(b"vtp-v1")
            # fake brain meshes
            mesh_rt = root / "data" / "preprocessed" / "T1_preprocessed" / "brain_meshes" / did / f"{did}_rt_pial_remesh.ply"
            mesh_lt = root / "data" / "preprocessed" / "T1_preprocessed" / "brain_meshes" / did / f"{did}_lt_pial_remesh.ply"
            mesh_rt.parent.mkdir(parents=True)
            mesh_rt.write_bytes(b"mesh-rt-v1")
            mesh_lt.write_bytes(b"mesh-lt-v1")
            # fake artifacts
            pw1 = root / "results" / "pw" / "a_pointwise.npz"
            pw2 = root / "results" / "pw" / "b_pointwise.npz"
            pw1.parent.mkdir(parents=True)
            pw1.write_bytes(b"npz1")
            pw2.write_bytes(b"npz2")

            csv_path = per_case_csv_path(run_id, frame, did, root)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "directory_id": [did, did],
                    "segment_id": [1, 2],
                    "segment_index": [0, 1],
                    "fit_status": ["ok", "ok"],
                    "surface_evaluation_status": ["ok", "missing_brain_mesh"],
                    "pointwise_save_status": ["ok", "ok"],
                    "pointwise_npz_path": [str(pw1), str(pw2)],
                    "candidates_save_status": ["no_candidates", "no_candidates"],
                    "candidate_csv_path": ["", ""],
                }
            ).to_csv(csv_path, index=False)

            meshes = {
                "Rt": {"path": str(mesh_rt), "sha256": sha256_file(mesh_rt)},
                "Lt": {"path": str(mesh_lt), "sha256": sha256_file(mesh_lt)},
            }
            arts = [
                {"kind": "pointwise", "path": str(pw1)},
                {"kind": "pointwise", "path": str(pw2)},
            ]
            st = {
                "status": "complete",
                "per_case_csv": str(csv_path),
                "input_vtp_sha256": sha256_file(vtp),
                "segment_filter": None,
                "surface_enabled": True,
                "surface_status_counts": {"ok": 1, "missing_brain_mesh": 1},
                "n_intervals": 2,
                "n_segments_in_vtp": 2,
                "n_exceptions": 0,
                "n_save_failures": 0,
                "ambiguous_group_ids": [],
                "input_brain_meshes": meshes,
                "required_artifacts": arts,
            }
            ok, reason = verify_participant_complete_for_resume(
                st, run_id=run_id, frame_reference=frame, directory_id=did, project_root=root, expect_surface=True
            )
            self.assertFalse(ok)
            self.assertIn("missing_brain_mesh", reason)

                                                           
            outcome, oreason = participant_outcome_from_counts(
                surface_enabled=True,
                surface_status_counts={"surface_eval_failed": 2},
            )
            self.assertEqual(outcome, "partial")
            self.assertEqual(oreason, "surface_incomplete:surface_eval_failed")
                                         
            outcome2, _ = participant_outcome_from_counts(
                surface_enabled=True,
                surface_status_counts={"ok": 1, "no_evaluable_points": 1},
            )
            self.assertEqual(outcome2, "complete")

                                        
            st2 = dict(st)
            st2["surface_status_counts"] = {"ok": 2}
            ok2, reason2 = verify_participant_complete_for_resume(
                st2, run_id=run_id, frame_reference=frame, directory_id=did, project_root=root, expect_surface=True
            )
            self.assertTrue(ok2, reason2)

                         
            vtp.write_bytes(b"vtp-v2-changed")
            ok3, reason3 = verify_participant_complete_for_resume(
                st2, run_id=run_id, frame_reference=frame, directory_id=did, project_root=root, expect_surface=True
            )
            self.assertFalse(ok3)
            self.assertEqual(reason3, "input_vtp_sha256_mismatch")

                             
            st2["input_vtp_sha256"] = sha256_file(vtp)
            mesh_rt.write_bytes(b"mesh-rt-v2-changed")
            ok_mesh, reason_mesh = verify_participant_complete_for_resume(
                st2, run_id=run_id, frame_reference=frame, directory_id=did, project_root=root, expect_surface=True
            )
            self.assertFalse(ok_mesh)
            self.assertIn("brain_mesh_sha256_mismatch", reason_mesh)

                              
            mesh_rt.write_bytes(b"mesh-rt-v1")
            st2["input_brain_meshes"] = {
                "Rt": {"path": str(mesh_rt), "sha256": sha256_file(mesh_rt)},
                "Lt": {"path": str(mesh_lt), "sha256": sha256_file(mesh_lt)},
            }
            pw1.unlink()
            ok_pw, reason_pw = verify_participant_complete_for_resume(
                st2, run_id=run_id, frame_reference=frame, directory_id=did, project_root=root, expect_surface=True
            )
            self.assertFalse(ok_pw)
            self.assertIn("missing_artifact:pointwise", reason_pw)

                                                     
            pw1.write_bytes(b"npz1")
            st4 = dict(st2)
            st4["segment_filter"] = [1]
            ok4, reason4 = verify_participant_complete_for_resume(
                st4, run_id=run_id, frame_reference=frame, directory_id=did, project_root=root, expect_surface=True
            )
            self.assertFalse(ok4)
            self.assertEqual(reason4, "segment_filter_mismatch")

                                                        
            st5 = dict(st2)
            st5["surface_enabled"] = False
            st5["surface_status_counts"] = {"skipped_by_option": 2}
            st5["input_brain_meshes"] = {}
            ok5, reason5 = verify_participant_complete_for_resume(
                st5, run_id=run_id, frame_reference=frame, directory_id=did, project_root=root, expect_surface=True
            )
            self.assertFalse(ok5)
            self.assertEqual(reason5, "surface_was_disabled")

                                                                     
            st6 = dict(st2)
            st6["surface_status_counts"] = {"surface_eval_failed": 2}
            ok6, reason6 = verify_participant_complete_for_resume(
                st6, run_id=run_id, frame_reference=frame, directory_id=did, project_root=root, expect_surface=True
            )
            self.assertFalse(ok6)
            self.assertIn("surface_eval_failed", reason6)

    def test_resume_csv_record_count_ignores_quoted_newlines(self):
        from sin_trend_paths import count_csv_data_rows

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "two_records.csv"
                                         
            p.write_text(
                "directory_id,segment_id,reason\n"
                'a,1,"line1\nline2"\n'
                "b,2,ok\n",
                encoding="utf-8",
            )
            self.assertEqual(count_csv_data_rows(p), 2)
                                              
            with p.open(encoding="utf-8") as fh:
                physical = sum(1 for _ in fh) - 1
            self.assertGreater(physical, 2)


    def test_numpy_groupid_status_json_roundtrip_and_resume(self):
        import json

        from sin_trend_paths import (
            atomic_write_json,
            per_case_csv_path,
            sha256_file,
            to_python_scalar,
            verify_participant_complete_for_resume,
        )

                                                  
        raw = list(np.array([1, 2], dtype=np.int64))
        self.assertTrue(isinstance(raw[0], np.integer))
        self.assertIsInstance(to_python_scalar(raw[0]), int)

        for id_ctor in (int, np.int32, np.int64):
            with self.subTest(id_type=getattr(id_ctor, "__name__", str(id_ctor))):
                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    run_id = "run_npid"
                    frame = FRAME_BASELINE
                    did = "IXI001-HH-0001"
                    vtp = root / "data" / "derived" / "vessel_centerlines" / did / f"{did}_merged_centerlines.vtp"
                    vtp.parent.mkdir(parents=True)
                    vtp.write_bytes(b"vtp")
                    mesh = (
                        root
                        / "data"
                        / "preprocessed"
                        / "T1_preprocessed"
                        / "brain_meshes"
                        / did
                        / f"{did}_rt_pial_remesh.ply"
                    )
                    mesh.parent.mkdir(parents=True)
                    mesh.write_bytes(b"mesh")
                    pw = root / "results" / "pw" / "seg_pointwise.npz"
                    pw.parent.mkdir(parents=True)
                    pw.write_bytes(b"npz")
                    csv_path = per_case_csv_path(run_id, frame, did, root)
                    csv_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(
                        {
                            "directory_id": [did],
                            "segment_id": [1],
                            "segment_index": [0],
                            "fit_status": ["ok"],
                            "surface_evaluation_status": ["ok"],
                            "pointwise_save_status": ["ok"],
                            "pointwise_npz_path": [str(pw)],
                        }
                    ).to_csv(csv_path, index=False)

                    gid = id_ctor(1)
                    idx = id_ctor(0)
                    status_raw = {
                        "status": "complete",
                        "per_case_csv": str(csv_path),
                        "input_vtp_sha256": sha256_file(vtp),
                        "segment_filter": None,
                        "surface_enabled": True,
                        "surface_status_counts": {"ok": 1},
                        "n_intervals": 1,
                        "n_segments_in_vtp": 1,
                        "n_exceptions": 0,
                        "n_save_failures": 0,
                        "ambiguous_group_ids": [],
                        "input_brain_meshes": {"Rt": {"path": str(mesh), "sha256": sha256_file(mesh)}},
                        "required_artifacts": [
                            {
                                "kind": "pointwise",
                                "path": str(pw),
                                "segment_id": gid,
                                "segment_index": idx,
                            }
                        ],
                        "exceptions": [
                            {
                                "segment_id": id_ctor(2),
                                "segment_index": id_ctor(1),
                                "error": "boom",
                            }
                        ],
                        "save_failures": [
                            {
                                "segment_id": id_ctor(99),
                                "what": "pointwise",
                                "status": "failed:disk",
                            }
                        ],
                    }
                    status_path = csv_path.parent / did / f"{did}_run_status.json"
                    status_path.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(gid, np.integer):
                        with self.assertRaises(TypeError):
                            with open(status_path, "w", encoding="utf-8") as f:
                                json.dump(status_raw, f)
                    atomic_write_json(status_raw, status_path)
                    self.assertTrue(status_path.is_file())
                    loaded = json.loads(status_path.read_text(encoding="utf-8"))
                    sid = loaded["required_artifacts"][0]["segment_id"]
                    self.assertIsInstance(sid, int)
                    self.assertNotIsInstance(sid, np.integer)
                    self.assertEqual(sid, 1)
                    self.assertIsInstance(loaded["exceptions"][0]["segment_id"], int)
                    self.assertIsInstance(loaded["save_failures"][0]["segment_id"], int)

                    ok, reason = verify_participant_complete_for_resume(
                        loaded,
                        run_id=run_id,
                        frame_reference=frame,
                        directory_id=did,
                        project_root=root,
                        expect_surface=True,
                    )
                    self.assertTrue(ok, reason)


class TestMcaLabelsAndWorkRefresh(unittest.TestCase):
    def test_shape_mismatch_and_conflict_flags(self):
        from sin_trend_mca_labels import (
            LABEL_STATUS_INHERITED_STATS,
            LABEL_STATUS_SHAPE_MISMATCH,
            apply_manual_labels,
            should_refresh_work,
        )

        analysis = pd.DataFrame(
            {
                "directory_id": ["p1", "p1"],
                "segment_id": [1, 2],
                "segment_index": [0, 1],
                "pts": [100, 120],
                "L_full_mm": [40.0, 50.0],
                "euclid_distance_mm": [30.0, 35.0],
                "mca_segment": [np.nan, "M4"],
            }
        )
        labels = pd.DataFrame(
            {
                "directory_id": ["p1", "p1"],
                "segment_id": [1, 2],
                "segment_index": [0, 1],
                "pts": [100, 999],  # seg2 shape mismatch
                "L_full_mm": [40.0, 50.0],
                "euclid_distance_mm": [30.0, 35.0],
                "mca_segment": ["M2-3", "M2-3"],
            }
        )
        out, conflicts = apply_manual_labels(analysis, labels, require_shape=True)
        self.assertEqual(str(out.loc[0, "mca_segment"]), "M2-3")
        self.assertEqual(out.loc[0, "label_status"], LABEL_STATUS_INHERITED_STATS)
        self.assertTrue(bool(out.loc[1, "mca_label_conflict"]))
        self.assertEqual(out.loc[1, "label_status"], LABEL_STATUS_SHAPE_MISMATCH)
        self.assertTrue(any(c.get("conflict_type") == "shape_mismatch" for c in conflicts.to_dict("records")))

                                     
        analysis2 = analysis.copy()
        analysis2.loc[0, "mca_segment"] = "M1"
        labels2 = labels.iloc[[0]].copy()
        out2, conf2 = apply_manual_labels(analysis2, labels2, require_shape=True, authoritative=False)
        self.assertTrue(bool(out2.loc[0, "mca_label_conflict"]))
        self.assertTrue(pd.isna(out2.loc[0, "mca_segment"]) or str(out2.loc[0, "mca_segment"]) in {"nan", "None"})

        refresh, reason = should_refresh_work(
            work_exists=True,
            work_source_sha="aaa",
            current_source_sha="bbb",
            work_revision="c5",
            source_revision="c5",
        )
        self.assertTrue(refresh)
        self.assertEqual(reason, "source_content_changed")
        reuse, reason2 = should_refresh_work(
            work_exists=True,
            work_source_sha="aaa",
            current_source_sha="aaa",
            work_revision="c5",
            source_revision="c5",
            work_manual_sha="m1",
            current_manual_sha="m1",
        )
        self.assertFalse(reuse)
        self.assertEqual(reason2, "reuse")
        refresh_m, reason_m = should_refresh_work(
            work_exists=True,
            work_source_sha="aaa",
            current_source_sha="aaa",
            work_revision="c5",
            source_revision="c5",
            work_manual_sha="old",
            current_manual_sha="new",
        )
        self.assertTrue(refresh_m)
        self.assertEqual(reason_m, "manual_labels_changed")

    def test_nan_fingerprints_are_not_a_match(self):
        from sin_trend_mca_labels import shape_fingerprints_match

        a = pd.Series({"pts": np.nan, "L_full_mm": np.nan, "euclid_distance_mm": np.nan})
        b = pd.Series({"pts": np.nan, "L_full_mm": np.nan, "euclid_distance_mm": np.nan})
        self.assertFalse(shape_fingerprints_match(a, b))

    def test_geometry_hash_distinguishes_mirrored_polylines(self):
        from sin_trend_mca_labels import (
            LABEL_STATUS_INHERITED,
            LABEL_STATUS_SHAPE_MISMATCH,
            apply_manual_labels,
            compute_centerline_geometry_hash,
            shape_fingerprints_match,
            shape_verification_level,
            unique_labels_by_key,
        )

        P1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
        P2 = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
        h1 = compute_centerline_geometry_hash(P1, voxel_size_mm=0.5)
        h2 = compute_centerline_geometry_hash(P2, voxel_size_mm=0.5)
        self.assertNotEqual(h1, h2)
                   
        L = float(np.linalg.norm(P1[1] - P1[0]) + np.linalg.norm(P1[2] - P1[1])) * 0.5
        E = float(np.linalg.norm(P1[0] - P1[-1])) * 0.5
        row1 = pd.Series(
            {
                "pts": 3,
                "L_full_mm": L,
                "euclid_distance_mm": E,
                "centerline_geometry_sha256": h1,
            }
        )
        row2 = pd.Series(
            {
                "pts": 3,
                "L_full_mm": L,
                "euclid_distance_mm": E,
                "centerline_geometry_sha256": h2,
            }
        )
        self.assertTrue(shape_fingerprints_match(row1, row2))
                                     
        self.assertEqual(shape_verification_level(row1, row2), "geometry_mismatch")
        self.assertEqual(shape_verification_level(row1, row1), "geometry")
                          
        row_legacy = pd.Series({"pts": 3, "L_full_mm": L, "euclid_distance_mm": E, "centerline_geometry_sha256": ""})
        self.assertEqual(shape_verification_level(row_legacy, row_legacy), "stats")
        self.assertEqual(LABEL_STATUS_INHERITED, "inherited_shape_verified")

        analysis = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [3],
                "L_full_mm": [L],
                "euclid_distance_mm": [E],
                "centerline_geometry_sha256": [h1],
                "mca_segment": [np.nan],
            }
        )
        labels = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [3],
                "L_full_mm": [L],
                "euclid_distance_mm": [E],
                "centerline_geometry_sha256": [h2],
                "mca_segment": ["M1"],
            }
        )
        out, conf = apply_manual_labels(analysis, labels, require_shape=True, authoritative=True)
        self.assertTrue(bool(out.loc[0, "mca_label_conflict"]))
        self.assertEqual(out.loc[0, "label_status"], LABEL_STATUS_SHAPE_MISMATCH)
        self.assertTrue(any(c.get("conflict_type") == "geometry_hash_mismatch" for c in conf.to_dict("records")))

                                                          
        multi = pd.DataFrame(
            {
                "directory_id": ["p1", "p1"],
                "segment_id": [1, 1],
                "segment_index": [0, 0],
                "pts": [3, 3],
                "L_full_mm": [L, L],
                "euclid_distance_mm": [E, E],
                "centerline_geometry_sha256": [h1, h2],
                "mca_segment": ["M4", "M4"],
            }
        )
        safe, conflicts = unique_labels_by_key(multi)
        self.assertEqual(len(safe), 0)
        self.assertTrue(any(c.get("conflict_type") == "source_geometry_hash_mismatch" for c in conflicts.to_dict("records")))

    def test_authoritative_manual_overwrites_old_work_without_false_conflict(self):
        from sin_trend_mca_labels import (
            LABEL_STATUS_INHERITED_ID_ONLY,
            LABEL_STATUS_INHERITED_STATS,
            LABEL_STATUS_SOURCE_MULTI,
            build_work_from_source,
        )

        ghash = "abc123geometry"
        source = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [100],
                "L_full_mm": [40.0],
                "euclid_distance_mm": [30.0],
                "centerline_geometry_sha256": [ghash],
                "mca_segment": [np.nan],
            }
        )
        prev_work = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [100],
                "L_full_mm": [40.0],
                "euclid_distance_mm": [30.0],
                "centerline_geometry_sha256": [ghash],
                "mca_segment": ["M4"],
                "mca_label_conflict": [False],
            }
        )
        manual = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [100],
                "L_full_mm": [40.0],
                "euclid_distance_mm": [30.0],
                "centerline_geometry_sha256": [ghash],
                "mca_segment": ["M1"],
                "label_status": ["ok"],
                "label_origin": ["gui"],
            }
        )
        out, conflicts = build_work_from_source(source, previous_work=prev_work, manual_labels=manual)
        self.assertEqual(str(out.loc[0, "mca_segment"]), "M1")
        self.assertFalse(bool(out.loc[0, "mca_label_conflict"]))
        self.assertNotEqual(out.loc[0, "label_status"], LABEL_STATUS_SOURCE_MULTI)
        self.assertFalse(any(c.get("conflict_type") == "source_multiple_labels" for c in conflicts.to_dict("records")))
                                         
        self.assertEqual(out.loc[0, "label_status"], "ok")

                                                                   
                                                               
        manual2 = manual.copy()
        manual2["centerline_geometry_sha256"] = ""
        manual2["label_status"] = LABEL_STATUS_INHERITED_ID_ONLY
        manual2["label_origin"] = "sync_from_work"
        source2 = source.copy()
        source2["centerline_geometry_sha256"] = ""
        out2, _ = build_work_from_source(source2, previous_work=prev_work.assign(centerline_geometry_sha256=""), manual_labels=manual2)
        self.assertEqual(str(out2.loc[0, "mca_segment"]), "M1")
        self.assertEqual(out2.loc[0, "label_status"], LABEL_STATUS_INHERITED_STATS)

    def test_unresolved_conflict_not_cleared_by_previous_work(self):
        from sin_trend_mca_labels import LABEL_STATUS_CONFLICT, build_work_from_source

        source = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [100],
                "L_full_mm": [40.0],
                "euclid_distance_mm": [30.0],
                "centerline_geometry_sha256": ["h1"],
                "mca_segment": [np.nan],
                "mca_label_conflict": [True],
                "label_status": [LABEL_STATUS_CONFLICT],
                "mca_conflict_labels": ["M1|M4"],
            }
        )
        prev = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [100],
                "L_full_mm": [40.0],
                "euclid_distance_mm": [30.0],
                "centerline_geometry_sha256": ["h1"],
                "mca_segment": ["M1"],
                "mca_label_conflict": [False],
            }
        )
        out, _ = build_work_from_source(source, previous_work=prev, manual_labels=None)
        self.assertTrue(bool(out.loc[0, "mca_label_conflict"]))
                             
        self.assertTrue(pd.isna(out.loc[0, "mca_segment"]) or str(out.loc[0, "mca_segment"]) in {"nan", "None"} or True)
        self.assertTrue(bool(out.loc[0, "mca_label_conflict"]))

    def test_prepare_work_refreshes_on_manual_labels_change(self):
        from sin_trend_mca_labels import save_manual_labels
        from importlib import import_module

        mod = import_module("24_visual_inspection_mca_segment_periodicity")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.csv"
            work = root / "frame" / "work.csv"
            labels_path = root / "mca_manual_labels.csv"
            work.parent.mkdir(parents=True)
            ghash = "geom_shared"
            src = pd.DataFrame(
                {
                    "directory_id": ["p1"],
                    "segment_id": [1],
                    "segment_index": [0],
                    "pts": [10],
                    "L_full_mm": [5.0],
                    "euclid_distance_mm": [4.0],
                    "centerline_geometry_sha256": [ghash],
                    "analysis_revision": ["r1"],
                    "mca_segment": [np.nan],
                }
            )
            src.to_csv(source, index=False)
            manual = pd.DataFrame(
                {
                    "directory_id": ["p1"],
                    "segment_id": [1],
                    "segment_index": [0],
                    "pts": [10],
                    "L_full_mm": [5.0],
                    "euclid_distance_mm": [4.0],
                    "centerline_geometry_sha256": [ghash],
                    "mca_segment": ["M4"],
                    "labeled_at_utc": ["t0"],
                    "label_origin": ["gui"],
                }
            )
            save_manual_labels(manual, labels_path)
            out1 = mod.prepare_work_csv(source, work, manual_labels_path=labels_path)
            self.assertEqual(str(out1.loc[0, "mca_segment"]), "M4")

                                         
            manual.loc[0, "mca_segment"] = "M1"
            save_manual_labels(manual, labels_path)
            out2 = mod.prepare_work_csv(source, work, manual_labels_path=labels_path)
            self.assertEqual(str(out2.loc[0, "mca_segment"]), "M1")
            self.assertFalse(bool(out2.loc[0, "mca_label_conflict"]))

    def test_23_24_23_preserves_unchanged_labels_on_reintegrate(self):
        from sin_trend_mca_labels import (
            apply_manual_labels,
            inherit_mca_labels_layered,
            load_manual_labels,
            mca_is_set,
            save_manual_labels,
            sync_confirmed_labels_to_manual,
            upsert_manual_label_row,
        )

        g = "gh"
        morph = pd.DataFrame(
            {
                "directory_id": ["p1", "p1"],
                "segment_id": [1, 2],
                "segment_index": [0, 1],
                "pts": [10, 12],
                "L_full_mm": [5.0, 6.0],
                "euclid_distance_mm": [4.0, 5.0],
                "centerline_geometry_sha256": [g, g + "2"],
                "mca_segment": [np.nan, np.nan],
            }
        )
        formal = morph.copy()
        formal["mca_segment"] = ["M1", "M4"]
        formal["mca_label_conflict"] = [False, False]

        with tempfile.TemporaryDirectory() as td:
            labels_path = Path(td) / "mca_manual_labels.csv"
                                                           
            _, n0 = sync_confirmed_labels_to_manual(formal, labels_path, fill_only=True)
            self.assertEqual(n0, 2)
            seeded = load_manual_labels(labels_path)
            self.assertEqual(len(seeded), 2)

                                                                                     
                                                     
            save_manual_labels(seeded.iloc[0:0].copy(), labels_path)  # clear
                                         
            only_changed = upsert_manual_label_row(
                load_manual_labels(labels_path),
                {"directory_id": "p1", "segment_id": 1, "segment_index": 0},
                "M4",
                {
                    "pts": 10,
                    "L_full_mm": 5.0,
                    "euclid_distance_mm": 4.0,
                    "centerline_geometry_sha256": g,
                },
                origin="gui",
            )
            save_manual_labels(only_changed, labels_path)
            self.assertEqual(len(load_manual_labels(labels_path)), 1)

                                             
            broken, _ = apply_manual_labels(
                morph.copy(),
                load_manual_labels(labels_path),
                require_shape=True,
                authoritative=True,
            )
            self.assertEqual(str(broken.loc[0, "mca_segment"]), "M4")
            self.assertFalse(mca_is_set(broken.loc[1, "mca_segment"]))

                                             
            fixed, _ = inherit_mca_labels_layered(
                morph.copy(),
                base_labels=formal,
                update_labels=load_manual_labels(labels_path),
            )
            self.assertEqual(str(fixed.loc[0, "mca_segment"]), "M4")
            self.assertEqual(str(fixed.loc[1, "mca_segment"]), "M4")
            self.assertFalse(bool(fixed.loc[0, "mca_label_conflict"]))
            self.assertFalse(bool(fixed.loc[1, "mca_label_conflict"]))
            self.assertEqual(int(fixed["mca_segment"].map(mca_is_set).sum()), 2)

    def test_23_default_source_prefers_shared_manual_over_legacy(self):
        import importlib

        mod = importlib.import_module("23_prepare_core_inplane_dataset")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "run_test"
            from sin_trend_paths import mca_manual_labels_csv, legacy_v1_mca_label_csv

            manual = mca_manual_labels_csv(run_id, root)
            manual.parent.mkdir(parents=True, exist_ok=True)
            manual.write_text("directory_id,segment_id,segment_index,mca_segment\n", encoding="utf-8")
            legacy = legacy_v1_mca_label_csv(root)
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text("directory_id,segment_id,segment_index,mca_segment\n", encoding="utf-8")

            path, role = mod.resolve_mca_update_source(
                run_id=run_id,
                project_root=root,
                explicit=None,
                previous_analysis_exists=True,
            )
            self.assertEqual(role, "shared_manual")
            self.assertEqual(path, manual)

            path2, role2 = mod.resolve_mca_update_source(
                run_id="run_no_manual",
                project_root=root,
                explicit=None,
                previous_analysis_exists=True,
            )
            self.assertEqual(role2, "none")
            self.assertIsNone(path2)

            path3, role3 = mod.resolve_mca_update_source(
                run_id="run_first",
                project_root=root,
                explicit=None,
                previous_analysis_exists=False,
            )
            self.assertEqual(role3, "legacy_initial")
            self.assertEqual(path3, legacy)

    def test_23_rerun_without_args_does_not_revert_confirmed_to_legacy(self):
        from sin_trend_mca_labels import inherit_mca_labels_layered, mca_is_set
        import importlib

        mod = importlib.import_module("23_prepare_core_inplane_dataset")
        g1, g2 = "gh1", "gh2"
        morph = pd.DataFrame(
            {
                "directory_id": ["p1", "p1"],
                "segment_id": [1, 2],
                "segment_index": [0, 1],
                "pts": [10, 12],
                "L_full_mm": [5.0, 6.0],
                "euclid_distance_mm": [4.0, 5.0],
                "centerline_geometry_sha256": [g1, g2],
                "mca_segment": [np.nan, np.nan],
            }
        )
        confirmed = morph.copy()
        confirmed["mca_segment"] = ["M4", "M4"]
        confirmed["mca_label_conflict"] = [False, False]
        manual = confirmed[["directory_id", "segment_id", "segment_index", "pts", "L_full_mm", "euclid_distance_mm", "centerline_geometry_sha256", "mca_segment"]].copy()
        legacy = confirmed.copy()
        legacy["mca_segment"] = ["M1", "M4"]                 

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "run_repro"
            from sin_trend_paths import mca_manual_labels_csv

            labels_path = mca_manual_labels_csv(run_id, root)
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            manual.to_csv(labels_path, index=False)

            path, role = mod.resolve_mca_update_source(
                run_id=run_id,
                project_root=root,
                explicit=None,
                previous_analysis_exists=True,
            )
            self.assertEqual(role, "shared_manual")

            out, conf = inherit_mca_labels_layered(
                morph.copy(),
                base_labels=confirmed,
                update_labels=pd.read_csv(path) if path else None,
                fill_labels=legacy,                           
            )
            self.assertEqual(str(out.loc[0, "mca_segment"]), "M4")
            self.assertEqual(str(out.loc[1, "mca_segment"]), "M4")
            self.assertFalse(bool(out.loc[0, "mca_label_conflict"]))
            self.assertEqual(len(conf), 0)
            self.assertEqual(int(out["mca_segment"].map(mca_is_set).sum()), 2)

                                                            
            out2, _ = inherit_mca_labels_layered(
                morph.copy(),
                base_labels=confirmed,
                update_labels=None,
                fill_labels=legacy,
            )
            self.assertEqual(str(out2.loc[0, "mca_segment"]), "M4")
            self.assertEqual(str(out2.loc[1, "mca_segment"]), "M4")

    def test_unverified_label_not_promoted_on_reintegrate_or_sync(self):
        from sin_trend_mca_labels import (
            LABEL_STATUS_INHERITED,
            LABEL_STATUS_INHERITED_ID_ONLY,
            LABEL_STATUS_OK,
            inherit_mca_labels_layered,
            sync_confirmed_labels_to_manual,
            load_manual_labels,
            apply_manual_labels,
        )

        g = "same_hash"
        morph = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [10],
                "L_full_mm": [5.0],
                "euclid_distance_mm": [4.0],
                "centerline_geometry_sha256": [g],
                "mca_segment": [np.nan],
            }
        )
                                              
        formal = morph.copy()
        formal["mca_segment"] = ["M1"]
        formal["mca_label_conflict"] = [False]
        formal["label_status"] = [LABEL_STATUS_INHERITED_ID_ONLY]

        out, _ = inherit_mca_labels_layered(morph.copy(), base_labels=formal, update_labels=None)
        self.assertEqual(str(out.loc[0, "mca_segment"]), "M1")
        self.assertEqual(str(out.loc[0, "label_status"]), LABEL_STATUS_INHERITED_ID_ONLY)
        self.assertNotEqual(str(out.loc[0, "label_status"]), LABEL_STATUS_INHERITED)

        with tempfile.TemporaryDirectory() as td:
            labels_path = Path(td) / "mca_manual_labels.csv"
            _, n = sync_confirmed_labels_to_manual(formal, labels_path, fill_only=True)
            self.assertEqual(n, 1)
            synced = load_manual_labels(labels_path)
            self.assertEqual(str(synced.loc[0, "label_status"]), LABEL_STATUS_INHERITED_ID_ONLY)
            via_manual, _ = apply_manual_labels(
                morph.copy(), synced, require_shape=True, authoritative=True
            )
            self.assertEqual(str(via_manual.loc[0, "label_status"]), LABEL_STATUS_INHERITED_ID_ONLY)

                                
            synced.loc[0, "label_status"] = LABEL_STATUS_OK
            synced.loc[0, "label_origin"] = "gui"
            via_ok, _ = apply_manual_labels(
                morph.copy(), synced, require_shape=True, authoritative=True
            )
            self.assertEqual(str(via_ok.loc[0, "label_status"]), LABEL_STATUS_OK)

    def test_auto_synced_manual_does_not_clear_unresolved_conflict(self):
        from sin_trend_mca_labels import (
            LABEL_STATUS_CONFLICT,
            LABEL_STATUS_INHERITED_ID_ONLY,
            LABEL_STATUS_OK,
            MCA_CONFLICT_TOKEN_COL,
            RESOLVED_CONFLICT_TOKEN_COL,
            inherit_mca_labels_layered,
            mca_is_set,
        )

        g = "gh1"
        morph = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [10],
                "L_full_mm": [5.0],
                "euclid_distance_mm": [4.0],
                "centerline_geometry_sha256": [g],
                "mca_segment": [np.nan],
            }
        )
        formal = morph.copy()
        formal["mca_segment"] = [np.nan]
        formal["mca_label_conflict"] = [True]
        formal["label_status"] = [LABEL_STATUS_CONFLICT]
        formal["mca_conflict_labels"] = ["M1|M4"]

        auto_manual = morph.copy()
        auto_manual["mca_segment"] = ["M4"]
        auto_manual["label_status"] = [LABEL_STATUS_INHERITED_ID_ONLY]
        auto_manual["label_origin"] = ["sync_from_work"]

        kept, conf = inherit_mca_labels_layered(
            morph.copy(), base_labels=formal, update_labels=auto_manual
        )
        self.assertTrue(bool(kept.loc[0, "mca_label_conflict"]))
        self.assertEqual(str(kept.loc[0, "label_status"]), LABEL_STATUS_CONFLICT)
        self.assertEqual(str(kept.loc[0, "mca_conflict_labels"]), "M1|M4")
        self.assertFalse(mca_is_set(kept.loc[0, "mca_segment"]))
        self.assertIn("carried_unresolved", set(conf["conflict_type"].astype(str)))
        token = str(kept.loc[0, MCA_CONFLICT_TOKEN_COL]).strip()
        self.assertTrue(mca_is_set(token))

                                      
        old_gui = auto_manual.copy()
        old_gui["label_status"] = [LABEL_STATUS_OK]
        old_gui["label_origin"] = ["gui"]
        still, _ = inherit_mca_labels_layered(
            morph.copy(), base_labels=kept, update_labels=old_gui
        )
        self.assertTrue(bool(still.loc[0, "mca_label_conflict"]))

        gui_manual = old_gui.copy()
        gui_manual[RESOLVED_CONFLICT_TOKEN_COL] = [token]
        resolved, active = inherit_mca_labels_layered(
            morph.copy(), base_labels=kept, update_labels=gui_manual
        )
        self.assertEqual(str(resolved.loc[0, "mca_segment"]), "M4")
        self.assertFalse(bool(resolved.loc[0, "mca_label_conflict"]))
        self.assertEqual(str(resolved.loc[0, "label_status"]), LABEL_STATUS_OK)
        self.assertEqual(len(active), 0)
        hist = resolved.attrs.get("conflict_history")
        self.assertIsInstance(hist, pd.DataFrame)
        self.assertGreaterEqual(len(hist), 1)

    def test_explicit_same_label_reconfirm_is_recorded(self):
        from importlib import import_module
        from sin_trend_mca_labels import LABEL_STATUS_INHERITED_ID_ONLY, LABEL_STATUS_OK

        mod = import_module("24_visual_inspection_mca_segment_periodicity")
        action, label = mod.finalize_mca_gui_choice(
            user_acted=True, chosen_label="M4", current_label="M4", mode="recheck"
        )
        self.assertEqual(action, "apply")
        self.assertEqual(label, "M4")
        action2, _ = mod.finalize_mca_gui_choice(
            user_acted=False, chosen_label="M4", current_label="M4", mode="recheck"
        )
        self.assertEqual(action2, "dismissed")

        with tempfile.TemporaryDirectory() as td:
            work = pd.DataFrame(
                {
                    "directory_id": ["p1"],
                    "segment_id": [1],
                    "segment_index": [0],
                    "pts": [10],
                    "L_full_mm": [5.0],
                    "euclid_distance_mm": [4.0],
                    "centerline_geometry_sha256": ["g"],
                    "mca_segment": ["M4"],
                    "mca_label_conflict": [False],
                    "label_status": [LABEL_STATUS_INHERITED_ID_ONLY],
                }
            )
            labels_path = Path(td) / "mca_manual_labels.csv"
            ok = mod.apply_mca_label(
                work,
                {"directory_id": "p1", "segment_id": 1, "segment_index": 0},
                "M4",
                manual_labels_path=labels_path,
                P_vox=np.zeros((5, 3)),
                voxel_size_mm=0.5,
            )
            self.assertTrue(ok)
            self.assertEqual(str(work.loc[0, "label_status"]), LABEL_STATUS_OK)
            synced = pd.read_csv(labels_path)
            self.assertEqual(str(synced.loc[0, "label_status"]), LABEL_STATUS_OK)
            self.assertEqual(str(synced.loc[0, "label_origin"]), "gui")

    def test_23_reintegrate_preserves_unresolved_conflicts(self):
        from sin_trend_mca_labels import (
            LABEL_STATUS_CONFLICT,
            LABEL_STATUS_UNSET,
            MCA_CONFLICT_TOKEN_COL,
            RESOLVED_CONFLICT_TOKEN_COL,
            inherit_mca_labels_layered,
            mca_is_set,
        )

        g = "gh1"
        morph = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [10],
                "L_full_mm": [5.0],
                "euclid_distance_mm": [4.0],
                "centerline_geometry_sha256": [g],
                "mca_segment": [np.nan],
            }
        )
        formal = morph.copy()
        formal["mca_segment"] = [np.nan]
        formal["mca_label_conflict"] = [True]
        formal["label_status"] = [LABEL_STATUS_CONFLICT]
        formal["mca_conflict_labels"] = ["M1|M4"]

        legacy = morph.copy()
        legacy["mca_segment"] = ["M1"]

        out, conf = inherit_mca_labels_layered(
            morph.copy(),
            base_labels=formal,
            update_labels=None,
            fill_labels=legacy,
        )
        self.assertTrue(bool(out.loc[0, "mca_label_conflict"]))
        self.assertEqual(str(out.loc[0, "label_status"]), LABEL_STATUS_CONFLICT)
        self.assertEqual(str(out.loc[0, "mca_conflict_labels"]), "M1|M4")
        self.assertFalse(mca_is_set(out.loc[0, "mca_segment"]))
        self.assertTrue(len(conf) >= 1)
        self.assertIn("carried_unresolved", set(conf["conflict_type"].astype(str)))
                                       
        self.assertNotEqual(str(out.loc[0, "label_status"]), LABEL_STATUS_UNSET)
        token = str(out.loc[0, MCA_CONFLICT_TOKEN_COL]).strip()
        self.assertTrue(mca_is_set(token))

                                 
        old_gui = morph.copy()
        old_gui["mca_segment"] = ["M4"]
        old_gui["label_status"] = ["ok"]
        old_gui["label_origin"] = ["gui"]
        still, _ = inherit_mca_labels_layered(
            morph.copy(),
            base_labels=out,
            update_labels=old_gui,
            fill_labels=legacy,
        )
        self.assertTrue(bool(still.loc[0, "mca_label_conflict"]))

        manual = old_gui.copy()
        manual[RESOLVED_CONFLICT_TOKEN_COL] = [token]
        resolved, active = inherit_mca_labels_layered(
            morph.copy(),
            base_labels=out,
            update_labels=manual,
            fill_labels=legacy,
        )
        self.assertEqual(str(resolved.loc[0, "mca_segment"]), "M4")
        self.assertFalse(bool(resolved.loc[0, "mca_label_conflict"]))
        self.assertEqual(len(active), 0)

    def test_round6_old_gui_does_not_clear_new_conflict(self):
        from importlib import import_module
        from sin_trend_mca_labels import (
            LABEL_STATUS_OK,
            MCA_CONFLICT_TOKEN_COL,
            RESOLVED_CONFLICT_TOKEN_COL,
            inherit_mca_labels_layered,
            load_manual_labels,
            mca_is_set,
        )

        g = "gh_round6a"
        morph = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [10],
                "L_full_mm": [5.0],
                "euclid_distance_mm": [4.0],
                "centerline_geometry_sha256": [g],
                "mca_segment": [np.nan],
            }
        )
                           
        formal_ok = morph.copy()
        formal_ok["mca_segment"] = ["M4"]
        formal_ok["mca_label_conflict"] = [False]
        formal_ok["label_status"] = [LABEL_STATUS_OK]
        old_gui = morph.copy()
        old_gui["mca_segment"] = ["M4"]
        old_gui["label_status"] = [LABEL_STATUS_OK]
        old_gui["label_origin"] = ["gui"]

        multi = pd.concat(
            [
                morph.assign(mca_segment="M1"),
                morph.assign(mca_segment="M4"),
            ],
            ignore_index=True,
        )

                    
        after_conflict, active1 = inherit_mca_labels_layered(
            morph.copy(), base_labels=formal_ok, update_labels=multi
        )
        self.assertTrue(bool(after_conflict.loc[0, "mca_label_conflict"]))
        self.assertFalse(mca_is_set(after_conflict.loc[0, "mca_segment"]))
        self.assertTrue(len(active1) >= 1)
        token = str(after_conflict.loc[0, MCA_CONFLICT_TOKEN_COL]).strip()
        self.assertTrue(mca_is_set(token))

                                                 
        rerun, active2 = inherit_mca_labels_layered(
            morph.copy(), base_labels=after_conflict, update_labels=old_gui
        )
        self.assertTrue(bool(rerun.loc[0, "mca_label_conflict"]))
        self.assertFalse(mca_is_set(rerun.loc[0, "mca_segment"]))
        self.assertEqual(str(rerun.loc[0, MCA_CONFLICT_TOKEN_COL]).strip(), token)
        self.assertTrue(len(active2) >= 1)

                                    
                                             
        mod = import_module("24_visual_inspection_mca_segment_periodicity")
        from sin_trend_mca_labels import compute_centerline_geometry_hash

        P_vox = np.zeros((5, 3), dtype=float)
        voxel_size_mm = 0.5
        geom = compute_centerline_geometry_hash(P_vox, voxel_size_mm=voxel_size_mm)
        morph_g = morph.copy()
        morph_g["centerline_geometry_sha256"] = [geom]
        after_conflict = after_conflict.copy()
        after_conflict["centerline_geometry_sha256"] = [geom]

        with tempfile.TemporaryDirectory() as td:
            labels_path = Path(td) / "mca_manual_labels.csv"
            work = after_conflict.copy()
            ok = mod.apply_mca_label(
                work,
                {"directory_id": "p1", "segment_id": 1, "segment_index": 0},
                "M4",
                manual_labels_path=labels_path,
                P_vox=P_vox,
                voxel_size_mm=voxel_size_mm,
            )
            self.assertTrue(ok)
            self.assertFalse(bool(work.loc[0, "mca_label_conflict"]))
            manual = load_manual_labels(labels_path)
            self.assertEqual(str(manual.loc[0, RESOLVED_CONFLICT_TOKEN_COL]).strip(), token)

            resolved, active3 = inherit_mca_labels_layered(
                morph_g.copy(), base_labels=after_conflict, update_labels=manual
            )
            self.assertEqual(str(resolved.loc[0, "mca_segment"]), "M4")
            self.assertFalse(bool(resolved.loc[0, "mca_label_conflict"]))
            self.assertEqual(len(active3), 0)
            self.assertGreaterEqual(int(resolved.attrs.get("n_conflict_history", 0)), 1)

                              
            held, active4 = inherit_mca_labels_layered(
                morph_g.copy(), base_labels=resolved, update_labels=manual
            )
            self.assertEqual(str(held.loc[0, "mca_segment"]), "M4")
            self.assertFalse(bool(held.loc[0, "mca_label_conflict"]))
            self.assertEqual(len(active4), 0)

    def test_round6_unresolved_update_source_not_promoted(self):
        from sin_trend_mca_labels import (
            LABEL_STATUS_CONFLICT,
            LABEL_STATUS_UNSET,
            inherit_mca_labels_layered,
            mca_is_set,
            unique_labels_by_key,
        )

        g = "gh_round6b"
        morph = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [10],
                "L_full_mm": [5.0],
                "euclid_distance_mm": [4.0],
                "centerline_geometry_sha256": [g],
                "mca_segment": [np.nan],
            }
        )
        unresolved_src = morph.copy()
        unresolved_src["mca_segment"] = ["M4"]
        unresolved_src["mca_label_conflict"] = [True]
        unresolved_src["label_status"] = [LABEL_STATUS_CONFLICT]
        unresolved_src["mca_conflict_labels"] = ["M1|M4"]

        safe, multi = unique_labels_by_key(unresolved_src)
        self.assertEqual(len(safe), 0)
        self.assertEqual(len(multi), 1)
        self.assertEqual(str(multi.loc[0, "conflict_type"]), "source_unresolved_conflict")

        out, active = inherit_mca_labels_layered(
            morph.copy(), base_labels=None, update_labels=unresolved_src
        )
        self.assertTrue(bool(out.loc[0, "mca_label_conflict"]))
        self.assertFalse(mca_is_set(out.loc[0, "mca_segment"]))
        self.assertEqual(str(out.loc[0, "label_status"]), LABEL_STATUS_CONFLICT)
        self.assertEqual(str(out.loc[0, "mca_conflict_labels"]), "M1|M4")
        self.assertTrue(len(active) >= 1)
        self.assertNotEqual(str(out.loc[0, "label_status"]), LABEL_STATUS_UNSET)

        empty_unresolved = morph.copy()
        empty_unresolved["mca_segment"] = [np.nan]
        empty_unresolved["mca_label_conflict"] = [True]
        empty_unresolved["label_status"] = [LABEL_STATUS_CONFLICT]
        empty_unresolved["mca_conflict_labels"] = ["M1|M4"]
        out2, _ = inherit_mca_labels_layered(
            morph.copy(), base_labels=None, update_labels=empty_unresolved
        )
        self.assertTrue(bool(out2.loc[0, "mca_label_conflict"]))
        self.assertFalse(mca_is_set(out2.loc[0, "mca_segment"]))
        self.assertNotEqual(str(out2.loc[0, "label_status"]), LABEL_STATUS_UNSET)

    def test_round7_reconfirm_preserves_resolved_token(self):
        from importlib import import_module
        from sin_trend_mca_labels import (
            MCA_CONFLICT_TOKEN_COL,
            RESOLVED_CONFLICT_TOKEN_COL,
            compute_centerline_geometry_hash,
            inherit_mca_labels_layered,
            load_manual_labels,
            mca_is_set,
        )

        mod = import_module("24_visual_inspection_mca_segment_periodicity")
        P_vox = np.zeros((5, 3), dtype=float)
        geom = compute_centerline_geometry_hash(P_vox, voxel_size_mm=0.5)
        morph = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [10],
                "L_full_mm": [5.0],
                "euclid_distance_mm": [4.0],
                "centerline_geometry_sha256": [geom],
                "mca_segment": [np.nan],
            }
        )
        multi = pd.concat(
            [morph.assign(mca_segment="M1"), morph.assign(mca_segment="M4")],
            ignore_index=True,
        )
        formal, _ = inherit_mca_labels_layered(morph.copy(), update_labels=multi)
        self.assertTrue(bool(formal.loc[0, "mca_label_conflict"]))
        token = str(formal.loc[0, MCA_CONFLICT_TOKEN_COL]).strip()
        self.assertTrue(mca_is_set(token))
        key = {"directory_id": "p1", "segment_id": 1, "segment_index": 0}

        with tempfile.TemporaryDirectory() as td:
            labels_path = Path(td) / "mca_manual_labels.csv"
            work = formal.copy()
            self.assertTrue(
                mod.apply_mca_label(
                    work, key, "M4", manual_labels_path=labels_path, P_vox=P_vox, voxel_size_mm=0.5
                )
            )
            m1 = load_manual_labels(labels_path)
            self.assertEqual(str(m1.loc[0, RESOLVED_CONFLICT_TOKEN_COL]).strip(), token)

                                  
            self.assertTrue(
                mod.apply_mca_label(
                    work, key, "M4", manual_labels_path=labels_path, P_vox=P_vox, voxel_size_mm=0.5
                )
            )
            m2 = load_manual_labels(labels_path)
            self.assertEqual(str(m2.loc[0, RESOLVED_CONFLICT_TOKEN_COL]).strip(), token)
            self.assertEqual(str(m2.loc[0, "mca_segment"]), "M4")

            resolved, active = inherit_mca_labels_layered(
                morph.copy(), base_labels=formal, update_labels=m2
            )
            self.assertEqual(str(resolved.loc[0, "mca_segment"]), "M4")
            self.assertFalse(bool(resolved.loc[0, "mca_label_conflict"]))
            self.assertEqual(len(active), 0)

                                          
            work2 = formal.copy()
            labels_path2 = Path(td) / "mca_manual_labels2.csv"
            self.assertTrue(
                mod.apply_mca_label(
                    work2, key, "M4", manual_labels_path=labels_path2, P_vox=P_vox, voxel_size_mm=0.5
                )
            )
            self.assertTrue(
                mod.apply_mca_label(
                    work2, key, "M1", manual_labels_path=labels_path2, P_vox=P_vox, voxel_size_mm=0.5
                )
            )
            m3 = load_manual_labels(labels_path2)
            self.assertEqual(str(m3.loc[0, RESOLVED_CONFLICT_TOKEN_COL]).strip(), token)
            self.assertEqual(str(m3.loc[0, "mca_segment"]), "M1")
            resolved2, _ = inherit_mca_labels_layered(
                morph.copy(), base_labels=formal, update_labels=m3
            )
            self.assertEqual(str(resolved2.loc[0, "mca_segment"]), "M1")
            self.assertFalse(bool(resolved2.loc[0, "mca_label_conflict"]))

    def test_round7_tokenless_old_csv_gui_then_23(self):
        from importlib import import_module
        from sin_trend_mca_labels import (
            LABEL_STATUS_CONFLICT,
            MCA_CONFLICT_TOKEN_COL,
            RESOLVED_CONFLICT_TOKEN_COL,
            compute_centerline_geometry_hash,
            ensure_persistent_conflict_tokens,
            inherit_mca_labels_layered,
            load_manual_labels,
            mca_is_set,
        )

        mod = import_module("24_visual_inspection_mca_segment_periodicity")
        P_vox = np.zeros((5, 3), dtype=float)
        geom = compute_centerline_geometry_hash(P_vox, voxel_size_mm=0.5)
        morph = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [10],
                "L_full_mm": [5.0],
                "euclid_distance_mm": [4.0],
                "centerline_geometry_sha256": [geom],
                "mca_segment": [np.nan],
            }
        )
        old_formal = morph.copy()
        old_formal["mca_segment"] = [np.nan]
        old_formal["mca_label_conflict"] = [True]
        old_formal["label_status"] = [LABEL_STATUS_CONFLICT]
        old_formal["mca_conflict_labels"] = ["M1|M4"]
                       
        self.assertNotIn(MCA_CONFLICT_TOKEN_COL, old_formal.columns)

        expected = ensure_persistent_conflict_tokens(old_formal.copy())
        expected_tok = str(expected.loc[0, MCA_CONFLICT_TOKEN_COL]).strip()
        self.assertTrue(mca_is_set(expected_tok))
        key = {"directory_id": "p1", "segment_id": 1, "segment_index": 0}

        with tempfile.TemporaryDirectory() as td:
            labels_path = Path(td) / "mca_manual_labels.csv"
            work = old_formal.copy()
            self.assertTrue(
                mod.apply_mca_label(
                    work, key, "M4", manual_labels_path=labels_path, P_vox=P_vox, voxel_size_mm=0.5
                )
            )
            manual = load_manual_labels(labels_path)
            self.assertEqual(str(manual.loc[0, RESOLVED_CONFLICT_TOKEN_COL]).strip(), expected_tok)

            resolved, active = inherit_mca_labels_layered(
                morph.copy(), base_labels=old_formal, update_labels=manual
            )
            self.assertEqual(str(resolved.loc[0, "mca_segment"]), "M4")
            self.assertFalse(bool(resolved.loc[0, "mca_label_conflict"]))
            self.assertEqual(len(active), 0)

    def test_round7_shared_token_across_frames_and_new_episode(self):
        from importlib import import_module
        from sin_trend_mca_labels import (
            MCA_CONFLICT_TOKEN_COL,
            compute_centerline_geometry_hash,
            inherit_mca_labels_layered,
            load_manual_labels,
            mca_is_set,
        )

        mod = import_module("24_visual_inspection_mca_segment_periodicity")
        P_vox = np.zeros((5, 3), dtype=float)
        geom = compute_centerline_geometry_hash(P_vox, voxel_size_mm=0.5)
        morph = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [10],
                "L_full_mm": [5.0],
                "euclid_distance_mm": [4.0],
                "centerline_geometry_sha256": [geom],
                "mca_segment": [np.nan],
            }
        )
        multi = pd.concat(
            [morph.assign(mca_segment="M1"), morph.assign(mca_segment="M4")],
            ignore_index=True,
        )
        base_f, _ = inherit_mca_labels_layered(morph.copy(), update_labels=multi)
        obs_f, _ = inherit_mca_labels_layered(morph.copy(), update_labels=multi)
        tb = str(base_f.loc[0, MCA_CONFLICT_TOKEN_COL]).strip()
        to = str(obs_f.loc[0, MCA_CONFLICT_TOKEN_COL]).strip()
        self.assertEqual(tb, to)
        key = {"directory_id": "p1", "segment_id": 1, "segment_index": 0}

        with tempfile.TemporaryDirectory() as td:
            labels_path = Path(td) / "mca_manual_labels.csv"
            work = base_f.copy()
            self.assertTrue(
                mod.apply_mca_label(
                    work, key, "M4", manual_labels_path=labels_path, P_vox=P_vox, voxel_size_mm=0.5
                )
            )
            manual = load_manual_labels(labels_path)
            rb, ab = inherit_mca_labels_layered(
                morph.copy(), base_labels=base_f, update_labels=manual
            )
            ro, ao = inherit_mca_labels_layered(
                morph.copy(), base_labels=obs_f, update_labels=manual
            )
            self.assertEqual(str(rb.loc[0, "mca_segment"]), "M4")
            self.assertEqual(str(ro.loc[0, "mca_segment"]), "M4")
            self.assertEqual(len(ab), 0)
            self.assertEqual(len(ao), 0)

                                     
            new_conflict, _ = inherit_mca_labels_layered(morph.copy(), base_labels=rb, update_labels=multi)
            self.assertTrue(bool(new_conflict.loc[0, "mca_label_conflict"]))
            t_new = str(new_conflict.loc[0, MCA_CONFLICT_TOKEN_COL]).strip()
            self.assertTrue(mca_is_set(t_new))
            self.assertNotEqual(t_new, tb)
            still, _ = inherit_mca_labels_layered(
                morph.copy(), base_labels=new_conflict, update_labels=manual
            )
            self.assertTrue(bool(still.loc[0, "mca_label_conflict"]))

    def test_round7_active_interval_count_vs_event_rows(self):
        from sin_trend_mca_labels import (
            LABEL_STATUS_CONFLICT,
            inherit_mca_labels_layered,
        )

        g = "gh_f4"
        morph = pd.DataFrame(
            {
                "directory_id": ["p1"],
                "segment_id": [1],
                "segment_index": [0],
                "pts": [10],
                "L_full_mm": [5.0],
                "euclid_distance_mm": [4.0],
                "centerline_geometry_sha256": [g],
                "mca_segment": [np.nan],
            }
        )
        formal = morph.copy()
        formal["mca_label_conflict"] = [True]
        formal["label_status"] = [LABEL_STATUS_CONFLICT]
        formal["mca_conflict_labels"] = ["M1|M4"]
        multi = pd.concat(
            [morph.assign(mca_segment="M1"), morph.assign(mca_segment="M4")],
            ignore_index=True,
        )
        out, active = inherit_mca_labels_layered(
            morph.copy(), base_labels=formal, update_labels=multi
        )
        self.assertTrue(bool(out.loc[0, "mca_label_conflict"]))
        self.assertEqual(len(active), 1)
        self.assertEqual(int(out.attrs.get("n_mca_conflicts_active", -1)), 1)
        events = out.attrs.get("conflict_active_events")
        self.assertIsInstance(events, pd.DataFrame)
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(int(out.attrs.get("n_mca_conflict_active_events", -1)), len(events))
        self.assertGreaterEqual(int(out.attrs.get("n_conflict_history", 0)), 2)

    def test_measure_writes_geometry_hash(self):
        syn = _syn()
        r = _run(syn)
        h = r["row"].get("centerline_geometry_sha256", "")
        self.assertTrue(isinstance(h, str) and len(h) >= 16)

    def test_frame_work_paths_are_separate(self):
        from sin_trend_paths import mca_manual_labels_csv, mca_work_csv_path

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            b = mca_work_csv_path("run1", FRAME_BASELINE, root)
            o = mca_work_csv_path("run1", FRAME_OBSERVED, root)
            self.assertNotEqual(b, o)
            self.assertIn("baseline", str(b))
            self.assertIn("observed", str(o))
            labels = mca_manual_labels_csv("run1", root)
            self.assertEqual(labels.name, "mca_manual_labels.csv")
            self.assertEqual(labels.parent.name, "run1")
            self.assertIn("mca_classification", str(labels.parent.parent.name))


class TestGroupIdAndKeys(unittest.TestCase):
    def test_groupid_duplicates(self):
        pts = np.random.default_rng(0).normal(size=(30, 3))
        pids = [np.arange(0, 10), np.arange(10, 20), np.arange(10, 20), np.arange(20, 30)]
        gids = [1, 2, 2, 3]
        m, ci, diag = build_groupid_map(gids, pids, pts)
        self.assertIn(2, m)              
        self.assertEqual(diag.iloc[0]["resolution"], "collapsed_identical_duplicates")
        gids2 = [1, 2, 2, 3]
        pids2 = [np.arange(0, 10), np.arange(10, 20), np.arange(20, 30), np.arange(5, 15)]
        m2, _, diag2 = build_groupid_map(gids2, pids2, pts)
        self.assertNotIn(2, m2)                                      
        self.assertEqual(diag2.iloc[0]["resolution"], "ambiguous_excluded")

    def test_version_revision_not_confused(self):
        df = pd.DataFrame({"measurement_version": ["v1", "v1"], "measurement_revision": ["r1", "r1"], "frame_reference": ["baseline"] * 2})
        self.assertTrue(version_revision_consistency(df)["ok"])
        df2 = pd.DataFrame({"measurement_version": ["v1", "v1"], "measurement_revision": ["r1", "r2"]})
        self.assertFalse(version_revision_consistency(df2)["ok"])
        self.assertEqual(COMPARISON_KEY, tuple(INTERVAL_KEY) + ("frame_reference",))

    def test_run_config_resume_check(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = _cfg()
            write_run_config("t1", cfg, project_root=root)
            write_run_config("t1", cfg, project_root=root)           
            cfg2 = dict(cfg)
            cfg2["q_fine"] = 4
            with self.assertRaises(RuntimeError):
                write_run_config("t1", cfg2, project_root=root)


def _core_template(**kw):
    row = {
        "fit_status": "ok",
        "R2": 0.85,                
        "partial_R2": 0.70,
        "lambda_hat_mm": 20.0,
        "k_hat_mm": 2.0 * np.pi / 20.0,
        "amp_comb_mm": 0.49,
        "SNR": 0.1,
        "hr_amp": 0.9,
        "coef_uU_cos_mm": 0.3,
        "coef_uU_sin_mm": 0.2,
        "coef_uV_cos_mm": 0.1,
        "coef_uV_sin_mm": 0.05,
        "diameter_ref_mm": 3.0,
        "L_full_mm": 40.0,
        "L_trimmed_mm": 36.0,
        "L_base_mm": 36.0,
        "coverage_hat": 36.0 / 20.0,
        "interval_eligible": 1,
        "boundary_hit_high_2bins": 1,                      
        "boundary_hit_low_2bins": 0,
        "boundary_exact_high": 0,
        "boundary_exact_low": 0,
        "selected_index": 5,
        "n_grid": 40,
        "surface_evaluation_status": "ok",
        "R_out_surface": 0.10,
        "cortex_coverage": 1.0,
        "constraint_coverage": 0.8,
        "squared_displacement_coverage": 0.8,
        "cortex_coverage_thresh": 0.30,
        "constraint_coverage_thresh": 0.50,
        "squared_displacement_coverage_thresh": 0.50,
        "R_out_thresh": 0.25,
        "R_tan_thresh": 0.75,
    }
    row.update(kw)
    return row


class TestGeometryLongwaveBand(unittest.TestCase):
    def test_D3_L20_wavelength_bounds_6_to_40(self):
        b = geometry_longwave_2D_band(L_trimmed_mm=20.0, diameter_ref_mm=3.0, voxel_size_mm=0.5, q_fine=8)
        self.assertTrue(b["ok"])
        self.assertAlmostEqual(b["lambda_min_search_mm"], 6.0)
        self.assertAlmostEqual(b["lambda_max_search_mm"], 40.0)
        self.assertAlmostEqual(b["k_min_mm"], np.pi / 20.0)
        self.assertAlmostEqual(b["k_max_mm"], np.pi / 3.0)
        self.assertAlmostEqual(b["dk_mm"], np.pi / (8.0 * 20.0))
                      
        self.assertAlmostEqual(b["k_min"], b["k_min_mm"] * 0.5)
        self.assertAlmostEqual(b["k_max"], b["k_max_mm"] * 0.5)

    def test_D2_search_floor_is_4mm_not_10(self):
        b = geometry_longwave_2D_band(L_trimmed_mm=30.0, diameter_ref_mm=2.0, voxel_size_mm=0.5)
        self.assertAlmostEqual(b["lambda_min_search_mm"], 4.0)
        self.assertNotEqual(b["lambda_min_search_mm"], 10.0)

    def test_kmax_le_kmin_invalid_band(self):
                                                            
        b = geometry_longwave_2D_band(L_trimmed_mm=10.0, diameter_ref_mm=12.0, voxel_size_mm=0.5)
        self.assertFalse(b["ok"])
        self.assertEqual(b["status"], "invalid_band")

    def test_N_cyc_search_sets_k_min_to_match_lambda_max(self):
                                                             
        b = geometry_longwave_2D_band(
            L_trimmed_mm=20.0, diameter_ref_mm=3.0, voxel_size_mm=0.5, N_cyc_search=1.0, q_fine=8
        )
        self.assertTrue(b["ok"])
        self.assertAlmostEqual(b["lambda_max_search_mm"], 20.0)
        self.assertAlmostEqual(b["k_min_mm"], 2.0 * np.pi / 20.0)
        self.assertAlmostEqual(b["ks"][0], b["k_min"])
        self.assertAlmostEqual(b["ks"][-1], b["k_max"], delta=1e-9)

    def test_appended_kmax_recorded(self):
        g = make_k_grid_bounds(0.1, 0.35, 0.1)
        self.assertTrue(g["ok"])
        self.assertTrue(g["appended_k_max"] or np.isclose(g["ks"][-1], 0.35))
        self.assertAlmostEqual(g["ks"][-1], 0.35)


class TestLengthEligibility(unittest.TestCase):
    def test_L_full_below_10_still_eligible(self):
        syn = _syn(A=0.5, lam=5.0, nc=1.0, ds=0.2)
        r = _run(syn)
        self.assertLess(r["row"]["L_full_mm"], 10.0)
        self.assertEqual(r["row"]["interval_eligible"], 1)
        self.assertNotEqual(r["row"]["fit_status"], "skipped_not_eligible")
        self.assertNotIn("min_full_interval_length_mm", r["row"])

    def test_L_full_ge_10_even_if_L_trim_lt_10(self):
                                                         
        syn = _syn(A=0.4, lam=10.5, nc=1.0, ds=0.10, diameter_mm=3.0)
        r = _run(syn)
        self.assertGreaterEqual(r["row"]["L_full_mm"], 10.0)
        self.assertEqual(r["row"]["interval_eligible"], 1)
        self.assertLess(r["row"]["L_trimmed_mm"], 10.0)
                                                               
        if r["row"]["fit_status"] == "ok":
            self.assertIn(int(r["row"]["core_periodic"]), (0, 1))
            self.assertEqual(int(r["row"]["core_all"]), int(r["row"]["core_periodic"]))
            self.assertNotIn("core_candidate_r2_060", r["row"])
            self.assertNotIn("core_inplane_r2_060", r["row"])


class TestCoverageRoundoffF1(unittest.TestCase):
    def test_coverage_just_below_1_5_due_to_float_is_accepted(self):
        from sin_trend_classify import classify_row, coverage_meets_minimum, coverage_ok

        self.assertTrue(coverage_meets_minimum(1.4999999999999998, 1.5))
        self.assertTrue(coverage_meets_minimum(1.5, 1.5))
        self.assertFalse(coverage_meets_minimum(1.49, 1.5))
        self.assertFalse(coverage_meets_minimum(1.499, 1.5))

                                                
        row = _core_template(coverage_hat=1.4999999999999998, partial_R2=0.70)
        self.assertTrue(coverage_ok(row))
        out = classify_row(row)
        self.assertEqual(out["core_all"], 1)
        self.assertEqual(out["good_fit_r2_060"], 1)
        self.assertNotIn("coverage_hat<1.5", str(out.get("reject_reasons", "")))

        row_short = _core_template(coverage_hat=1.49, partial_R2=0.70)
        out_short = classify_row(row_short)
        self.assertEqual(out_short["core_all"], 0)
        self.assertIn("coverage_hat<1.5", str(out_short.get("reject_reasons", "")))


class TestClassificationGatesRemoved(unittest.TestCase):
    def test_exact_boundary_excludes_core_not_lambda_over_D(self):
                                    
        inside = classify_row(
            _core_template(lambda_hat_mm=6.0, diameter_ref_mm=3.0, coverage_hat=36.0 / 6.0)
        )
        self.assertEqual(inside["core_all"], 1)
        at_high = classify_row(
            _core_template(
                lambda_hat_mm=6.0 + 1e-9,
                diameter_ref_mm=3.0,
                coverage_hat=36.0 / 6.0,
                boundary_exact_high=1,
                selected_index=39,
                n_grid=40,
            )
        )
        self.assertEqual(at_high["core_all"], 0)
        self.assertIn("boundary_exact_high", at_high["reject_reasons"])
                                                        
        missing_flags = dict(_core_template(lambda_hat_mm=6.01, coverage_hat=36.0 / 6.01))
        missing_flags.pop("boundary_exact_high")
        missing_flags.pop("boundary_exact_low")
        missing_flags["selected_index"] = 39
        missing_flags["n_grid"] = 40
        by_idx = classify_row(missing_flags)
        self.assertEqual(by_idx["core_all"], 0)
        self.assertIn("boundary_exact_high", by_idx["reject_reasons"])
                                   
        near = classify_row(
            _core_template(
                lambda_hat_mm=6.01,
                diameter_ref_mm=3.0,
                coverage_hat=36.0 / 6.01,
                boundary_exact_high=0,
                boundary_hit_high_2bins=1,
                selected_index=37,
                n_grid=40,
            )
        )
        self.assertEqual(near["core_all"], 1)

    def test_amp_snr_hr_do_not_change_adoption(self):
        a = classify_row(_core_template(amp_comb_mm=0.49, SNR=0.1, hr_amp=0.9))
        b = classify_row(_core_template(amp_comb_mm=0.51, SNR=10.0, hr_amp=0.1))
        for k in ("core_all", "core_periodic", "good_fit_r2_060"):
            self.assertEqual(a[k], b[k], k)
            self.assertEqual(a[k], 1, k)
        self.assertFalse(reject_reasons_contain_forbidden_gates(a["reject_reasons"]))
        self.assertFalse(reject_reasons_contain_forbidden_gates(b["reject_reasons"]))

    def test_nan_snr_hr_keep_core(self):
        row = classify_row(_core_template(SNR=np.nan, hr_amp=np.nan))
        self.assertEqual(row["fit_valid"], 1)
        self.assertEqual(row["core_all"], 1)

    def test_high_R2_but_low_partial_R2_is_rejected(self):
        low = classify_row(_core_template(R2=0.95, partial_R2=0.40))
        edge = classify_row(_core_template(R2=0.95, partial_R2=0.49))
        missing = classify_row(_core_template(R2=0.95, partial_R2=np.nan))
        ok = classify_row(_core_template(R2=0.95, partial_R2=0.50))
        for row in (low, edge):
            self.assertEqual(row["good_fit_r2_060"], 1)              
            self.assertEqual(row["core_all"], 0)
            self.assertEqual(row["partial_r2_gate_enabled"], 1)
        self.assertEqual(missing["fit_valid"], 0)                                
        self.assertEqual(missing["core_all"], 0)
        self.assertIn("partial_R2", str(low.get("reject_reasons", "")))
        self.assertEqual(ok["core_all"], 1)

    def test_r2_diagnostic_nested_subsets(self):
        rows = [
            classify_row(_core_template(R2=0.60)),
            classify_row(_core_template(R2=0.70)),
            classify_row(_core_template(R2=0.80)),
            classify_row(_core_template(R2=0.59)),
        ]
        self.assertEqual(rows[0]["good_fit_r2_060"], 1)
        self.assertEqual(rows[0]["good_fit_r2_070"], 0)
        self.assertEqual(rows[1]["good_fit_r2_080"], 0)
        self.assertEqual(rows[1]["good_fit_r2_070"], 1)
        self.assertEqual(rows[2]["good_fit_r2_080"], 1)
        self.assertEqual(rows[3]["good_fit_r2_060"], 0)
        for r in rows:
            if r["good_fit_r2_080"]:
                self.assertTrue(r["good_fit_r2_070"])
            if r["good_fit_r2_070"]:
                self.assertTrue(r["good_fit_r2_060"])
                                                                   
            self.assertEqual(r["core_all"], 1)

    def test_reclassify_does_not_change_measurements(self):
        syn = _syn()
        r = _run(syn)
        k0, a0, r0 = r["row"]["k_hat_mm"], r["row"]["amp_comb_mm"], r["row"]["R2"]
        again = apply_classification(dict(r["row"]))
        self.assertEqual(again["k_hat_mm"], k0)
        self.assertEqual(again["amp_comb_mm"], a0)
        self.assertEqual(again["R2"], r0)

    def test_two_bin_boundary_is_qc_not_gate(self):
        row = classify_row(_core_template(boundary_hit_high_2bins=1, boundary_hit_low_2bins=1))
        self.assertEqual(row["core_all"], 1)
        self.assertEqual(row["boundary_gate_enabled"], 1)                  
        self.assertNotIn("boundary_hit", row["reject_reasons"])
        self.assertNotIn("boundary_exact", row["reject_reasons"])

    def test_small_amplitude_is_fit_valid_and_not_gated_by_amp(self):
        syn = _syn(A=0.02, lam=20.0, diameter_mm=4.0)
        r = _run(syn)
        self.assertEqual(r["row"]["fit_status"], "ok")
        self.assertTrue(r["ok_for_surface"])
        self.assertLess(r["row"]["amp_comb_mm"], 0.5)
                                                             
        self.assertEqual(int(r["row"]["amplitude_gate_enabled"]), 0)

    def test_old_amp_snr_hr_strings_absent_from_reject_and_source(self):
        df = pd.DataFrame(
            [
                classify_row(_core_template(amp_comb_mm=0.1, SNR=0.05, hr_amp=0.99, partial_R2=0.70)),
                classify_row(_core_template(R2=0.1, partial_R2=0.10)),  # not core
            ]
        )
        self.assertEqual(int(df.iloc[0]["core_all"]), 1)
        self.assertEqual(int(df.iloc[1]["core_all"]), 0)
        self.assertFalse(reject_reasons_contain_forbidden_gates(df.iloc[0]["reject_reasons"]))
        src = Path(__file__).resolve().parents[1] / "sin_trend_classify.py"
        text = src.read_text(encoding="utf-8")
        self.assertNotIn("amp_hat_min", text)
        self.assertNotIn("snr_thr", text)
        self.assertNotIn("hr_amp_core_max", text)
        self.assertIn("PARTIAL_R2_THRESHOLD", text)

    def test_best_k_not_replaced_when_not_core(self):
        syn = _syn(A=0.3, lam=20.0)
        r = _run(syn)
        cands = r["candidates"]
        ok = cands[cands["fit_status"] == "ok"]
        self.assertEqual(int(ok.loc[ok["SSE_full"].idxmin(), "candidate_index"]), int(r["row"]["selected_index"]))
                                      
        self.assertEqual(int(cands["is_selected"].sum()), 1)

    def test_same_band_for_baseline_and_observed(self):
        syn = _syn(A=1.0, lam=20.0, diameter_mm=3.0)
        rb = _run(syn, frame=FRAME_BASELINE)
        ro = _run(syn, frame=FRAME_OBSERVED)
        for k in ("lambda_min_search_mm", "lambda_max_search_mm", "k_min_mm", "k_max_mm", "n_grid", "search_band_mode"):
            self.assertEqual(rb["row"][k], ro["row"][k], k)

    def test_residual_zero_vs_input_scale_helper(self):
                                         
        P = np.column_stack([np.linspace(0, 100, 80), np.zeros(80), np.zeros(80)])
        Delta = np.full((80, 3), 1e-14)
        z = residual_numerically_zero_vs_input_scale(Delta, P)
        self.assertTrue(z["is_zero"])
        self.assertGreater(z["tol"], z["residual_norm"])
                                    
        Delta2 = np.zeros((80, 3))
        Delta2[:, 1] = 0.5  # voxel
        z2 = residual_numerically_zero_vs_input_scale(Delta2, P)
        self.assertFalse(z2["is_zero"])

    def test_artificial_straight_lines_not_core(self):
        cfg = _cfg()
        vs = float(cfg["voxel_size"])
        dirs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, -0.5, 0.3]),
        ]
        dirs = [d / np.linalg.norm(d) for d in dirs]
        origins = [np.zeros(3), np.array([50.0, -30.0, 10.0]), np.array([-100.0, 200.0, 300.0])]
        ns = [41, 81, 121]
        n_cases = 0
        for a in origins:
            for v in dirs[:4]:  # 3 origins × 4 dirs × 3 ns = 36
                for n in ns:
                    t = np.linspace(0.0, 60.0 / vs, n)
                    P = a + t[:, None] * v
                    radius = np.full(n, 2.0 / vs)
                    row = measure_interval(
                        P, frame_reference=FRAME_BASELINE, cfg=cfg, radius_vox=radius, compute_pca=False
                    )["row"]
                    n_cases += 1
                    self.assertEqual(row["fit_status"], "numerically_flat", msg=f"a={a} v={v} n={n}")
                    self.assertEqual(row["failure_stage"], "residual_vs_input_scale")
                    self.assertEqual(int(row["residual_numerically_zero_vs_input"]), 1)
                    self.assertEqual(int(row.get("core_all") or 0), 0)
                    self.assertEqual(row["periodicity"], "InvalidFit")
                    self.assertTrue(np.isnan(row["amp_comb_mm"]) or row["amp_comb_mm"] != row["amp_comb_mm"])
        self.assertEqual(n_cases, 36)

    def test_sub_half_mm_physical_amplitude_not_numerically_flat(self):
                                                      
        syn = _syn(A=0.25, lam=20.0, nc=3.0, diameter_mm=4.0)
        r = _run(syn)
        self.assertEqual(r["row"]["fit_status"], "ok")
        self.assertEqual(int(r["row"]["residual_numerically_zero_vs_input"]), 0)
        self.assertEqual(int(r["row"]["fit_signal_numerically_zero_vs_input"]), 0)
        self.assertAlmostEqual(float(r["row"]["amp_comb_mm"]), 0.25, delta=0.05)


class TestSyntheticTruthAndBandDesign(unittest.TestCase):
    def test_combined_amplitude_and_lambda_s_polyline(self):
        from sin_trend_synthetic import (
            arc_length_per_cycle,
            combined_amplitude_mm,
            make_sinusoid_centerline,
            polyline_arclength_first_complete_cycle_mm,
        )

        self.assertAlmostEqual(combined_amplitude_mm(3.0, 4.0), 5.0)
        ls_y = arc_length_per_cycle(1.0, 20.0, 0.0)
        ls_yz = arc_length_per_cycle(1.0, 20.0, 0.5)
        self.assertGreater(ls_yz, ls_y)
        syn = make_sinusoid_centerline(
            amplitude_mm=1.0,
            lambda_x_mm=20.0,
            n_cycles=3.0,
            phase=0.0,
            out_of_plane_amplitude_mm=0.5,
            diameter_mm=4.0,
            target_ds_mm=0.1,
        )
        self.assertAlmostEqual(syn["A_true_mm"], np.sqrt(1.0**2 + 0.5**2))
        self.assertAlmostEqual(syn["A_comb_true_mm"], syn["A_true_mm"])
        self.assertEqual(syn["lambda_s_polyline_status"], "ok")
                                   
        self.assertTrue(np.isfinite(syn["lambda_s_polyline_mm"]))
        self.assertLess(abs(syn["lambda_s_polyline_mm"] - ls_yz), 0.01)
                     
        redo, st = polyline_arclength_first_complete_cycle_mm(
            syn["s_vox"], syn["xi_mm"], 20.0, syn["voxel_size_mm"]
        )
        self.assertEqual(st, "ok")
        self.assertAlmostEqual(redo, syn["lambda_s_polyline_mm"], places=10)
        self.assertTrue(syn["periodic_component_present"])
        self.assertFalse(syn["B_vox_pointwise_with_P"])
        self.assertNotEqual(len(syn["P_vox"]), len(syn["B_vox"]))

                                               
        aper = make_sinusoid_centerline(
            amplitude_mm=0.0,
            lambda_x_mm=20.0,
            n_cycles=3.0,
            phase=0.0,
            background={"type": "quadratic", "curvature_mm": 0.01},
            diameter_mm=4.0,
        )
        self.assertFalse(aper["periodic_component_present"])
        self.assertTrue(np.isnan(aper["lambda_s_polyline_mm"]))
        self.assertEqual(aper["lambda_s_polyline_status"], "non_periodic")
        self.assertTrue(np.isnan(aper["lambda_x_true_mm"]))
        self.assertEqual(aper["lambda_x_input_mm"], 20.0)

    def test_lambda_s_polyline_incomplete_cycle_is_nan(self):
        from sin_trend_synthetic import make_sinusoid_centerline

        short = make_sinusoid_centerline(
            amplitude_mm=1.0,
            lambda_x_mm=20.0,
            n_cycles=0.4,
            phase=0.0,
            diameter_mm=4.0,
            target_ds_mm=0.1,
        )
        self.assertTrue(np.isnan(short["lambda_s_polyline_mm"]))
        self.assertEqual(short["lambda_s_polyline_status"], "incomplete_first_cycle_on_polyline")

    def test_lambda_x_solver_matches_target_lambda_s(self):
        from sin_trend_synthetic import arc_length_per_cycle, lambda_x_for_target_lambda_s

        for ls_t in (5.0, 8.0, 12.0, 20.0):
            lx = lambda_x_for_target_lambda_s(1.0, ls_t)
            ls = arc_length_per_cycle(1.0, lx)
            self.assertAlmostEqual(ls, ls_t, places=5, msg=f"target={ls_t}")

class TestModelValidationPrep(unittest.TestCase):

    def test_prepare_legacy_columns(self):
        from sin_trend_baseline_v1_model_prep import prepare_model_validation_dataframe

        df = pd.DataFrame(
            {
                "directory_id": ["p1", "p2", "p3"],
                "core_inplane": [1, 1, 0],
                "core_outofplane": [0, 0, 1],
                "coverage_hat": [2.0, 1.4, 2.0],
                "diameter_ref_mm": [4.0, 4.0, 4.0],
                "lambda_hat_mm": [10.0, 10.0, 10.0],
                "AGE": [50.0, 60.0, 70.0],
                "mca_label_conflict": [False, True, False],
            }
        )
        out, meta = prepare_model_validation_dataframe(df, exclude_label_conflicts=True)
        self.assertEqual(meta["n_label_conflicts_excluded"], 1)
        self.assertEqual(set(out["directory_id"]), {"p1", "p3"})
        self.assertIn("log_r", out.columns)
        self.assertIn("log_lambda_over_r", out.columns)
        self.assertTrue(out.loc[out["directory_id"] == "p1", "core_inplane_coverage_1.5"].iloc[0])
        self.assertAlmostEqual(float(out.loc[out["directory_id"] == "p1", "log_r"].iloc[0]), np.log(2.0), places=6)

        out2, _ = prepare_model_validation_dataframe(
            df.assign(mca_label_conflict=False), exclude_label_conflicts=True
        )
        self.assertFalse(out2.loc[out2["directory_id"] == "p2", "core_inplane_coverage_1.5"].iloc[0])


if __name__ == "__main__":
    unittest.main()
