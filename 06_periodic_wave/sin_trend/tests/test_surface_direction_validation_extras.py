#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: test surface direction validation extras."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_classify import (  # noqa: E402
    SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL,
    SURFACE_LABEL_SUBSTANTIAL_NORMAL,
    surface_eligible_from_row,
)
from sin_trend_surface import evaluate_displacement  # noqa: E402
from sin_trend_surface_direction_validation import (  # noqa: E402
    TOL_GEOM_ABS,
    classify_direction,
    geometric_isotropic_R_tan_mean,
    theoretical_R,
)


class TestDirectionClassificationBoundary(unittest.TestCase):
    def test_boundary_strict_inequality(self):
        for r, expect in (
            (0.249999, SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL),
            (0.25, SURFACE_LABEL_SUBSTANTIAL_NORMAL),
            (0.250001, SURFACE_LABEL_SUBSTANTIAL_NORMAL),
        ):
            cls = classify_direction(
                R_out=r,
                constraint_coverage=0.9,
                squared_displacement_coverage=0.9,
                status="ok",
            )
            self.assertTrue(cls["evaluable"])
            self.assertEqual(cls["label"], expect)
            self.assertAlmostEqual(cls["R_tan"], 1.0 - r)

    def test_expected_labels_away_from_boundary(self):
                                               
        for th, expect in (
            (0.0, SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL),
            (29.0, SURFACE_LABEL_PREDOMINANTLY_TANGENTIAL),
            (31.0, SURFACE_LABEL_SUBSTANTIAL_NORMAL),
            (45.0, SURFACE_LABEL_SUBSTANTIAL_NORMAL),
            (90.0, SURFACE_LABEL_SUBSTANTIAL_NORMAL),
        ):
            ro, rt = theoretical_R(th)
            cls = classify_direction(
                R_out=ro,
                constraint_coverage=1.0,
                squared_displacement_coverage=1.0,
                status="ok",
            )
            self.assertEqual(cls["label"], expect, msg=f"theta={th}")
            self.assertAlmostEqual(cls["R_tan"], rt, places=12)

    def test_not_evaluable_when_coverage_low_or_undefined(self):
        cls = classify_direction(
            R_out=0.1,
            constraint_coverage=0.4,
            squared_displacement_coverage=0.9,
            status="ok",
        )
        self.assertFalse(cls["evaluable"])
        self.assertIsNone(cls["label"])
        cls2 = classify_direction(
            R_out=float("nan"),
            constraint_coverage=0.9,
            squared_displacement_coverage=0.9,
            status="zero_constrained_energy",
        )
        self.assertFalse(cls2["evaluable"])
        self.assertTrue(np.isnan(cls2["R_tan"]))


class TestTheoryAndGeometry(unittest.TestCase):
    def test_theoretical_R(self):
        for th, ro, rt in ((0, 0.0, 1.0), (30, 0.25, 0.75), (45, 0.5, 0.5), (90, 1.0, 0.0)):
            a, b = theoretical_R(th)
            self.assertAlmostEqual(a, ro, places=10)
            self.assertAlmostEqual(b, rt, places=10)

    def test_geometric_T_means_analytic_and_numeric(self):
        n = np.array([0.0, 0.0, 1.0])
        cases = (
            (np.array([1.0, 0.0, 0.0]), 0.5),
            (np.array([1.0, 0.0, 1.0]) / np.sqrt(2), 0.75),
            (np.array([0.0, 0.0, 1.0]), 1.0),
        )
        for T, expect in cases:
            g = geometric_isotropic_R_tan_mean(T, n)
            self.assertAlmostEqual(g["mean_R_tan_analytic"], expect, delta=TOL_GEOM_ABS)
            self.assertAlmostEqual(g["mean_R_tan_numeric"], expect, delta=1e-6)


class TestEtfIdentity(unittest.TestCase):
    def test_etf_from_evaluate_displacement(self):
                                                         
        n = 10
                                                                       
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
        d = np.tile(np.array([0.0, 0.6, 0.8]), (n, 1))
        mask = np.zeros(n, dtype=bool)
        mask[:6] = True  # sq_cov = 0.6
        est = dict(
            normals_full=normals,
            mask_eval=mask,
            eval_normal_source=np.zeros(n, dtype=int),  # all GOOD=0 in helpers; surface uses EVAL_SOURCE_*
            cortex_coverage=1.0,
            constraint_coverage=float(mask.mean()),
        )
                                                                   
        from sin_trend_surface import EVAL_SOURCE_GOOD

        est["eval_normal_source"] = np.full(n, EVAL_SOURCE_GOOD, dtype=int)
        ev = evaluate_displacement(d, est)
        self.assertEqual(ev["status"], "ok")
        self.assertAlmostEqual(ev["R_out"], 0.64, places=10)
        self.assertAlmostEqual(ev["R_tan"], 0.36, places=10)
        self.assertAlmostEqual(ev["squared_displacement_coverage"], 0.6, places=10)
        self.assertAlmostEqual(
            ev["evaluated_tangential_fraction"],
            ev["squared_displacement_coverage"] * ev["R_tan"],
            places=12,
        )


class TestThreshPairConsistency(unittest.TestCase):
    def test_inconsistent_tan_thresh_does_not_flip_R_out_gate(self):
                                                                    
        row = dict(
            surface_evaluation_status="ok",
            R_out_latent=0.275,
            R_tan_latent=0.725,
            constraint_coverage=0.9,
            squared_displacement_coverage=0.9,
            R_out_thresh=0.30,
            R_tan_thresh=0.75,                    
        )
        out = surface_eligible_from_row(row)
        self.assertTrue(out["thresh_pair_inconsistent"])
        self.assertAlmostEqual(out["R_tan_thresh_used"], 0.70)
        self.assertTrue(out["surface_eligible"])
        self.assertTrue(out["R_tan_ok"])  # 0.725 > 0.70


class TestAggregateOverallStatus(unittest.TestCase):
    def test_ok_false_or_empty_is_not_pass(self):
                                                    
        import importlib.util
        from pathlib import Path

        import pandas as pd

        path = Path(__file__).resolve().parents[1] / "38_surface_direction_validation.py"
        spec = importlib.util.spec_from_file_location("surf_dir_val38", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        plane_ok = pd.DataFrame(
            [
                dict(theta_plane_deg=0.0, ok=True, pass_all=True),
                dict(theta_plane_deg=45.0, ok=True, pass_all=True),
                dict(theta_plane_deg=90.0, ok=True, pass_all=True),
                dict(theta_plane_deg=29.0, ok=True, pass_all=True),
                dict(theta_plane_deg=30.0, ok=True, pass_all=True),
                dict(theta_plane_deg=31.0, ok=True, pass_all=True),
                dict(theta_plane_deg=45.0, ok=True, pass_all=True),
            ]
        )
        plane_fit_mixed = pd.DataFrame(
            [
                dict(case="ok_case", ok=True, pass_R=True),
                dict(case="fit_fail", ok=False, reason="no_pointwise"),
            ]
        )
        geom = pd.DataFrame(
            [
                dict(case="a", ok=True, pass_abs=True),
                dict(case="b", ok=True, pass_abs=True),
                dict(case="c", ok=True, pass_abs=True),
            ]
        )
        valley = pd.DataFrame(
            [
                dict(
                    path="air_ridge_short_trench",
                    tag="bridge_cfg_5mm_45deg",
                    ok=True,
                    pass_intent=True,
                    n_eval_bridged=5,
                    end_eval_source_none=False,
                ),
                dict(
                    path="air_ridge_short_trench",
                    tag="bridge_cfg_5mm_30deg",
                    ok=True,
                    pass_intent=True,
                    n_eval_bridged=0,
                ),
                dict(
                    path="air_ridge_long_trench",
                    tag="bridge_cfg_5mm_45deg",
                    ok=True,
                    pass_intent=True,
                    n_eval_bridged=0,
                    good_bridge_candidate_axis_deg_max=5.0,
                ),
                dict(
                    path="one_sided_into_trench",
                    tag="bridge_cfg_5mm_45deg",
                    ok=True,
                    pass_intent=True,
                    n_eval_bridged=0,
                    end_eval_source_none=True,
                ),
            ]
        )
        overall = mod.aggregate_overall_status(
            plane_ok,
            plane_fit_mixed,
            geom,
            {},
            valley=valley,
            clinical=dict(status="skipped_placeholder_targets"),
            expect_plane_fit=True,
        )
        self.assertEqual(overall["status"], "fail")
        self.assertTrue(any("plane_fit:ok=False" in f for f in overall["failures"]))

        overall_empty = mod.aggregate_overall_status(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            valley=pd.DataFrame(),
            expect_plane_fit=True,
        )
        self.assertEqual(overall_empty["status"], "fail")
        self.assertTrue(any("plane_direct:empty" in f for f in overall_empty["failures"]))


class TestBaselineTangentDiagnostics(unittest.TestCase):
    def test_evaluate_records_parallelism_and_upper_bound_with_T0(self):
        from sin_trend_surface import EVAL_SOURCE_GOOD, evaluate_displacement

        n = 8
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
                                              
        d = np.tile(np.array([0.0, np.sqrt(0.75), 0.5]), (n, 1))
        T0 = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
        mask = np.ones(n, dtype=bool)
        est = dict(
            normals_full=normals,
            mask_eval=mask,
            eval_normal_source=np.full(n, EVAL_SOURCE_GOOD, dtype=int),
            cortex_coverage=1.0,
            constraint_coverage=1.0,
        )
        ev = evaluate_displacement(d, est, T_frame=T0, T0=T0)
        self.assertEqual(ev["status"], "ok")
        self.assertAlmostEqual(ev["R_out"], 0.25, places=10)
        self.assertAlmostEqual(ev["R_tan"], 0.75, places=10)
        self.assertAlmostEqual(ev["abs_T0_dot_n_median"], 0.0, places=12)
        self.assertAlmostEqual(ev["R_out_upper_bound"], 1.0, places=12)  # 1-(T·n)²
        self.assertEqual(float(ev["R_out_within_upper_bound"]), 1.0)

    def test_T_at_45_to_plane_tightens_upper_bound(self):
        from sin_trend_surface import EVAL_SOURCE_GOOD, evaluate_displacement

        n = 8
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
        T0 = np.tile(np.array([1.0, 0.0, 1.0]) / np.sqrt(2), (n, 1))
                                                
                                                                   
        d = np.tile(np.array([-1.0, 0.0, 1.0]) / np.sqrt(2), (n, 1))
        est = dict(
            normals_full=normals,
            mask_eval=np.ones(n, dtype=bool),
            eval_normal_source=np.full(n, EVAL_SOURCE_GOOD, dtype=int),
            cortex_coverage=1.0,
            constraint_coverage=1.0,
        )
        ev = evaluate_displacement(d, est, T_frame=T0, T0=T0)
        self.assertAlmostEqual(ev["abs_T0_dot_n_median"], 1.0 / np.sqrt(2), places=10)
        self.assertAlmostEqual(ev["R_out_upper_bound"], 0.5, places=10)  # 1-0.5
        self.assertLessEqual(ev["R_out"], ev["R_out_upper_bound"] + 1e-9)


class TestClinicalAggregateAndPerpT(unittest.TestCase):
    def _load38(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "38_surface_direction_validation.py"
        spec = importlib.util.spec_from_file_location("surf_dir_val38_clin", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    def test_clinical_failures_fail_overall(self):
        import pandas as pd

        mod = self._load38()
        plane = pd.DataFrame([dict(theta_plane_deg=float(i), ok=True, pass_all=True) for i in range(7)])
        geom = pd.DataFrame([dict(case=c, ok=True, pass_abs=True) for c in ("a", "b", "c")])
        valley = pd.DataFrame(
            [
                dict(
                    path="air_ridge_short_trench",
                    tag="bridge_cfg_5mm_45deg",
                    ok=True,
                    pass_intent=True,
                    n_eval_bridged=5,
                    end_eval_source_none=False,
                ),
                dict(
                    path="air_ridge_short_trench",
                    tag="bridge_cfg_5mm_30deg",
                    ok=True,
                    pass_intent=True,
                    n_eval_bridged=0,
                ),
                dict(
                    path="air_ridge_long_trench",
                    tag="bridge_cfg_5mm_45deg",
                    ok=True,
                    pass_intent=True,
                    n_eval_bridged=0,
                    good_bridge_candidate_axis_deg_max=5.0,
                ),
                dict(
                    path="one_sided_into_trench",
                    tag="bridge_cfg_5mm_45deg",
                    ok=True,
                    pass_intent=True,
                    n_eval_bridged=0,
                    end_eval_source_none=True,
                ),
            ]
        )
                          
        fail_sens = mod.aggregate_overall_status(
            plane,
            pd.DataFrame([dict(case="ok", ok=True, pass_R=True)] * 3),
            geom,
            {},
            valley=valley,
            clinical=dict(
                status="ran_with_failures",
                n_ok=0,
                n_sensitivity_incomplete=1,
                n_target_exception=0,
                n_real_attempted=1,
            ),
            expect_plane_fit=True,
        )
        self.assertEqual(fail_sens["status"], "fail")
        self.assertTrue(any("clinical:" in f for f in fail_sens["failures"]))

                                         
        fail_all = mod.aggregate_overall_status(
            plane,
            pd.DataFrame([dict(case="ok", ok=True, pass_R=True)] * 3),
            geom,
            {},
            valley=valley,
            clinical=dict(
                status="ran_no_successful_targets",
                n_ok=0,
                n_sensitivity_incomplete=0,
                n_target_exception=0,
                n_real_attempted=2,
            ),
            expect_plane_fit=True,
        )
        self.assertEqual(fail_all["status"], "fail")

                          
        ok_ph = mod.aggregate_overall_status(
            plane,
            pd.DataFrame([dict(case="ok", ok=True, pass_R=True)] * 3),
            geom,
            {},
            valley=valley,
            clinical=dict(status="skipped_placeholder_targets", n_ok=0, n_real_attempted=0),
            expect_plane_fit=True,
        )
        self.assertEqual(ok_ph["status"], "pass")

    def test_sens_metrics_reclassify_no_duplicate_kwargs(self):
        m = dict(
            R_out=0.2,
            R_tan=0.8,
            constraint_coverage=0.9,
            squared_displacement_coverage=0.9,
            evaluable=True,
            direction_label="Predominantly tangential",
            surface_status="ok",
        )
        R_saved = 0.22
        m["R_out"] = R_saved
        m["R_tan"] = 1.0 - R_saved
        m["note"] = "reclassify_from_saved_R_out_only"
        row = dict(directory_id="X", segment_id=0, param="R_out_thresh", value=0.25, **m)
        self.assertAlmostEqual(row["R_out"], 0.22)
        self.assertEqual(row["note"], "reclassify_from_saved_R_out_only")

    def test_known_periodic_displacement_perp_T(self):
        from sin_trend_surface_direction_validation import (
            known_periodic_displacement_perp_T,
            unit_tangents_from_polyline,
        )

                              
        x = np.linspace(-20.0, 20.0, 41)
        z = 2.0 * np.exp(-(x**2) / (2 * 4.0**2))
        P = np.column_stack([x, np.zeros_like(x), z]) / 0.5  # voxel
        T = unit_tangents_from_polyline(P)
        d = known_periodic_displacement_perp_T(
            x,
            T,
            amplitude_mm=0.4,
            lambda_mm=15.0,
            phase=0.0,
            theta_plane_deg=30.0,
            voxel_size_mm=0.5,
        )
                        
        amp = np.linalg.norm(d, axis=1)
        mask = amp > 1e-8
        dots = np.sum(d[mask] * T[mask], axis=1)
        self.assertLess(float(np.max(np.abs(dots))), 1e-9)


if __name__ == "__main__":
    unittest.main()
