"""Workflow regressions using synthetic inputs only (no participant data)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
spec = importlib.util.spec_from_file_location("surface_validation_workflow", _HERE / "38_surface_direction_validation.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
import sin_trend_paths as paths
import sin_trend_surface as surface


class _PlotObject:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class ClinicalFixture:
    def __init__(self, root, *, missing_radius=False, fail_surface_call=None, invalid_ds=False, missing_target=False):
        self.root = Path(root)
        self.calls = []
        self.measure_calls = []
        self.plot_calls = []
        self.fail_surface_call = fail_surface_call
        self.invalid_ds = invalid_ds
        self.n = 41
        self.P = np.column_stack((np.arange(self.n) * .3, np.zeros(self.n), np.ones(self.n)))
        self.T = np.tile([1., 0., 0.], (self.n, 1))
        self.d = np.tile([0., np.sqrt(.9), np.sqrt(.1)], (self.n, 1))
        self.vtp = self.root / "synthetic.vtp"
        self.mesh = self.root / "synthetic.ply"
        self.vtp.write_text("synthetic marker: IO replaced by test adapter")
        self.mesh.write_text("synthetic marker: surface IO replaced by test adapter")
        self.targets = self.root / "targets.csv"
        rows = [dict(directory_id="SYNTHETIC_A", segment_id=7)]
        if missing_target:
            rows.append(dict(directory_id="SYNTHETIC_MISSING", segment_id=7))
        pd.DataFrame(rows).to_csv(self.targets, index=False)
        # A conflicting Radius field ensures the production radius key is selected.
        self.pointdata = {"Radius": np.full(self.n, 99.)}
        if not missing_radius:
            self.pointdata["MaximumInscribedSphereRadius"] = np.full(self.n, 1.4)
        self.est = dict(
            normals_full=np.tile([0., 0., 1.], (self.n, 1)),
            mask_eval=np.ones(self.n, bool),
            eval_normal_source=np.full(self.n, surface.EVAL_SOURCE_GOOD, int),
            coherence_full=np.ones(self.n), cortex_coverage=1., constraint_coverage=1.,
        )
        ce = types.ModuleType("centerline_extraction")
        ce.load_vtp = lambda path: None
        ce.vmtk_convert2numpy = lambda mesh: (
            {"GroupIds": np.array([7])}, [np.arange(self.n)], self.P, self.pointdata)
        vm = types.ModuleType("vessel_mesh_features")
        vm.classify_centerline_left_right_by_kmeans = lambda *a, **kw: (np.array([0]), np.array([]))
        pv = types.ModuleType("pyvista")
        pv.read = lambda path: types.SimpleNamespace(point_data={"Normals": np.array([[0., 0., 1.]])})
        plt = types.ModuleType("matplotlib.pyplot")
        plt.subplots = lambda *a, **kw: (_PlotObject(), [_PlotObject(), _PlotObject()])
        plt.close = lambda *a, **kw: None
        mpl = types.ModuleType("matplotlib")
        mpl.pyplot = plt
        self.modules = {"centerline_extraction": ce, "vessel_mesh_features": vm,
                        "pyvista": pv, "matplotlib": mpl, "matplotlib.pyplot": plt}

    def measure(self, *args, **kwargs):
        self.measure_calls.append(kwargs)
        radius_present = kwargs["radius_vox"] is not None
        return dict(
            row=dict(fit_status="ok", radius_status="ok" if radius_present else "missing",
                     radius_median_mm=.7, ds_median_vox=np.nan if self.invalid_ds else .3),
            pointwise=dict(P=self.P, P_full=self.P, d_latent=self.d, T0=self.T, T=self.T),
            patch=dict(ok=radius_present, r_patch_vox=5.6, r_patch_mm=2.8, reason="missing radius"),
        )

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == self.fail_surface_call:
            raise RuntimeError("synthetic sensitivity failure")
        ev = surface.evaluate_displacement(kwargs["d_sin"], self.est, T0=kwargs["T0"], T_frame=kwargs["T_frame"])
        return ev, self.est, "uncached_fixture"

    def plot(self, **kwargs):
        self.plot_calls.append(kwargs)
        return ["synthetic_plot.png"]

    def run(self):
        with patch.dict(sys.modules, self.modules), \
             patch.object(paths, "centerline_vtp_path", side_effect=lambda did, root: self.vtp if did == "SYNTHETIC_A" else self.root / "missing.vtp"), \
             patch.object(paths, "brain_mesh_path", return_value=self.mesh), \
             patch.object(validation, "measure_interval", side_effect=self.measure), \
             patch.object(surface, "run_surface_evaluation", side_effect=self.evaluate), \
             patch.object(validation, "write_clinical_pyvista_plots", side_effect=self.plot):
            guide = validation.run_clinical_sensitivity(self.root, self.targets, validation.validation_config())
        primary = pd.read_csv(self.root / "clinical_primary_results.csv")
        sensitivity_path = self.root / "clinical_sensitivity_results.csv"
        sensitivity = pd.read_csv(sensitivity_path) if sensitivity_path.exists() else pd.DataFrame()
        return guide, primary, sensitivity


class TestClinicalWorkflow(unittest.TestCase):
    def test_complete_target_reuses_production_inputs_and_writes_nine_sensitivities(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ClinicalFixture(tmp)
            guide, primary, sensitivity = fixture.run()
            self.assertEqual(guide["status"], "ran")
            self.assertEqual(guide["n_ok"], 1)
            self.assertEqual(len(primary), 1)
            self.assertEqual(primary.iloc[0]["status"], "ok")
            self.assertEqual(len(sensitivity), 9)
            self.assertEqual(len(fixture.calls), 7)  # primary + six normal-estimation settings
            self.assertEqual(len(fixture.plot_calls), 1)
            np.testing.assert_allclose(fixture.measure_calls[0]["radius_vox"], 1.4)
            for call in fixture.calls:
                self.assertEqual(call["ds_vox"], .3)  # measured value, not target .2 vox
                self.assertEqual(call["r_patch_vox"], 5.6)
                self.assertIsNone(call["cache_path"])
                np.testing.assert_array_equal(call["P_full_vox"], fixture.P)
                np.testing.assert_array_equal(call["T0"], fixture.T)
            self.assertEqual(sensitivity["param"].value_counts().to_dict(),
                             {"bridge_max_gap_mm": 3, "bridge_axis_max_deg": 3, "R_out_thresh": 3})
            thresholds = sensitivity[sensitivity.param == "R_out_thresh"]
            np.testing.assert_allclose(thresholds.R_out, primary.iloc[0].R_out)
            for col in ("squared_displacement_coverage", "squared_displacement_coverage_good",
                        "squared_displacement_coverage_bridged", "abs_T0_dot_n_median", "R_out_upper_bound"):
                self.assertTrue(sensitivity[col].notna().all(), col)
            self.assertEqual(len(primary.iloc[0].input_vtp_sha256), 64)
            self.assertEqual(len(primary.iloc[0].brain_mesh_sha256), 64)

    def test_sensitivity_failure_keeps_single_incomplete_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ClinicalFixture(tmp, fail_surface_call=3)
            guide, primary, sensitivity = fixture.run()
            self.assertEqual(guide["status"], "ran_with_failures")
            self.assertEqual(guide["n_ok"], 0)
            self.assertEqual(guide["n_sensitivity_incomplete"], 1)
            self.assertEqual(len(primary), 1)
            self.assertEqual(primary.iloc[0]["status"], "sensitivity_incomplete")
            self.assertIn("synthetic sensitivity failure", primary.iloc[0]["reason"])
            self.assertEqual(len(sensitivity), 1)

    def test_radius_alias_is_not_used_to_fill_missing_production_radius(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ClinicalFixture(tmp, missing_radius=True)
            guide, primary, sensitivity = fixture.run()
            self.assertIsNone(fixture.measure_calls[0]["radius_vox"])
            self.assertEqual(primary.iloc[0]["status"], "invalid_radius")
            self.assertEqual(guide["n_ok"], 0)
            self.assertFalse(fixture.calls)

    def test_invalid_measured_spacing_is_reported_instead_of_imputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ClinicalFixture(tmp, invalid_ds=True)
            guide, primary, sensitivity = fixture.run()
            self.assertEqual(primary.iloc[0]["status"], "target_exception")
            self.assertIn("ds_median_vox", primary.iloc[0]["reason"])
            self.assertEqual(guide["n_ok"], 0)
            self.assertFalse(fixture.calls)

    def test_missing_requested_target_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ClinicalFixture(tmp, missing_target=True)
            guide, primary, sensitivity = fixture.run()
            self.assertEqual(guide["status"], "ran_partial")
            self.assertEqual(guide["n_other_fail"], 1)
            self.assertEqual(guide["n_real_attempted"], 2)
            self.assertEqual(primary["status"].tolist(), ["ok", "missing_vtp"])


class TestMissingValidationEvidence(unittest.TestCase):
    def test_missing_pass_or_ok_is_failure(self):
        for table in (
            pd.DataFrame([dict(ok=True, pass_R=np.nan)]),
            pd.DataFrame([dict(ok=True, pass_R=pd.NA)]),
            pd.DataFrame([dict(ok=np.nan, pass_R=True)]),
        ):
            result = validation._table_required_failures(table, "fixture", pass_cols=("pass_R",), min_ok_rows=1)
            self.assertTrue(result, table)

    def test_partial_clinical_is_not_total_pass(self):
        result = validation.aggregate_overall_status(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {},
            clinical=dict(status="ran_partial", n_ok=1, n_real_attempted=2, n_other_fail=1))
        self.assertIn("clinical:ran_partial", result["failures"])

    def test_missing_valley_count_does_not_crash_or_pass(self):
        rows = [dict(ok=True, path="air_ridge_short_trench", tag="bridge_cfg_5mm_45deg",
                     intent="expect_possible_short_bridge_between_plateaus", n_eval_bridged=np.nan,
                     true_top_n_axis_err_deg_median=np.nan, R_out=np.nan, R_out_theory=.25)]
        annotated = validation._annotate_valley_intent_pass(pd.DataFrame(rows))
        self.assertFalse(annotated.iloc[0].pass_intent)
        self.assertTrue(validation._valley_required_failures(annotated))

    def test_long_gap_requires_observed_candidate_endpoints(self):
        row = dict(ok=True, intent="expect_gap_too_long_for_5mm_bridge", tag="bridge_cfg_5mm_45deg",
                   n_eval_bridged=0, n_bridge_candidate_pairs=0,
                   good_bridge_candidate_gap_mm_min=np.nan, good_bridge_candidate_axis_deg_max=np.nan)
        annotated = validation._annotate_valley_intent_pass(pd.DataFrame([row]))
        self.assertFalse(annotated.iloc[0].pass_intent)


if __name__ == "__main__":
    unittest.main()
