"""Immutable measurement lineage and isolated surface-tag configuration contracts."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sin_trend_paths as paths


class TestStageConfigContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.code = {}
        self.hash_patch = patch.object(paths, "source_code_hashes", side_effect=lambda files=paths.BASELINE_SOURCE_FILES, **kw: {
            name: self.code.get(name, "original") for name in files
        })
        self.hash_patch.start()
        self.versions_patch = patch.object(paths, "package_versions", return_value={})
        self.versions_patch.start()
        paths.write_run_config("run", {"voxel_size": 0.6, "q_fine": 3, "coh_thr": 0.75}, self.root)

    def tearDown(self):
        self.versions_patch.stop()
        self.hash_patch.stop()
        self.tmp.cleanup()

    def test_surface_keys_do_not_change_measurement_contract(self):
        p = paths.run_config_path("run", self.root)
        original = p.read_bytes()
        rec = paths.write_run_config("run", {"voxel_size": 0.6, "q_fine": 3, "coh_thr": 0.9}, self.root)
        self.assertNotIn("coh_thr", rec["config"])
        self.assertEqual(original, p.read_bytes())
        with self.assertRaises(RuntimeError):
            paths.write_run_config("run", {"voxel_size": 0.5, "q_fine": 3}, self.root)

    def test_surface_only_source_change_leaves_measurement_hash_unchanged(self):
        self.code["sin_trend_surface.py"] = "new_surface"
        self.code["sin_trend_surface_stage.py"] = "new_stage"
        paths.write_run_config("run", {"voxel_size": 0.6, "q_fine": 3}, self.root)
        self.code["sin_trend_forward_fit.py"] = "new_fit"
        with self.assertRaises(RuntimeError):
            paths.write_run_config("run", {"voxel_size": 0.6, "q_fine": 3}, self.root)

    def test_same_tag_is_immutable_even_with_overwrite(self):
        original_measurement = paths.run_config_path("run", self.root).read_bytes()
        cfg = {"coh_thr": 0.75, "R_out_thresh": 0.25}
        first = paths.write_surface_config("run", cfg, project_root=self.root)
        p = paths.surface_config_path("run", project_root=self.root)
        original_surface = p.read_bytes()
        self.assertEqual(first, paths.write_surface_config("run", cfg, project_root=self.root, overwrite=True))
        self.assertEqual(original_surface, p.read_bytes())
        with self.assertRaises(RuntimeError):
            paths.write_surface_config("run", {**cfg, "coh_thr": 0.8}, project_root=self.root, overwrite=True)
        paths.write_surface_config("run", {**cfg, "coh_thr": 0.8}, surface_tag="coh080", project_root=self.root)
        self.assertEqual(original_measurement, paths.run_config_path("run", self.root).read_bytes())
        self.assertEqual(first["config"]["voxel_size"], 0.6)

    def test_changed_surface_code_requires_new_tag_not_new_measurement_run(self):
        cfg = {"coh_thr": 0.75}
        paths.write_surface_config("run", cfg, project_root=self.root)
        self.code["sin_trend_surface_stage.py"] = "updated_stage"
        with self.assertRaises(RuntimeError):
            paths.write_surface_config("run", cfg, project_root=self.root)
        paths.write_surface_config("run", cfg, surface_tag="v2", project_root=self.root)

    def test_shared_voxel_is_inherited_and_cannot_be_overridden(self):
        self.assertEqual(paths.resolve_surface_config("run", {}, self.root)["voxel_size"], 0.6)
        for bad in (0.5, 0, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paths.resolve_surface_config("run", {"voxel_size": bad}, self.root)
        with self.assertRaises(ValueError):
            paths.resolve_surface_config("run", {"frame_reference": "observed"}, self.root)
        with self.assertRaises(ValueError):
            paths.resolve_surface_config("run", {"q_fine": 9}, self.root)

    def test_surface_only_builder_uses_saved_voxel_and_derives_tan_threshold(self):
        cfg = paths.surface_config_for_run("run", {"R_out_thresh": 0.2}, project_root=self.root)
        self.assertEqual(cfg["voxel_size"], 0.6)
        self.assertEqual(cfg["R_tan_thresh"], 0.8)
        paths.write_surface_config("run", cfg, project_root=self.root)
        self.assertEqual(paths.surface_config_for_run("run", project_root=self.root), cfg)

    def test_tan_only_override_derives_out_threshold_and_pair_mismatch_fails(self):
        cfg = paths.surface_config_for_run("run", {"R_tan_thresh": 0.8}, project_root=self.root)
        self.assertAlmostEqual(cfg["R_out_thresh"], 0.2)
        self.assertEqual(cfg["R_tan_thresh"], 0.8)
        partial = paths.resolve_surface_config("run", {"R_tan_thresh": 0.8}, self.root)
        self.assertAlmostEqual(partial["R_out_thresh"], 0.2)
        with self.assertRaises(ValueError):
            paths.surface_config_for_run("run", {"R_tan_thresh": 0.8, "R_out_thresh": 0.3}, project_root=self.root)

    def test_tan_only_decimal_boundary_retains_strict_out_comparison(self):
        from sin_trend_classify import surface_eligible_from_row
        for tan, out in ((0.70, 0.30), (0.75, 0.25), (0.80, 0.20)):
            with self.subTest(tan=tan):
                cfg = paths.surface_config_for_run("run", {"R_tan_thresh": tan}, project_root=self.root)
                self.assertEqual(cfg["R_out_thresh"], out)
                row = dict(cfg, surface_evaluation_status="ok", constraint_coverage=1.0, squared_displacement_coverage=1.0, R_out_latent=out)
                self.assertFalse(surface_eligible_from_row(row)["surface_eligible"])
                row["R_out_latent"] = np.nextafter(out, 0.0)
                self.assertTrue(surface_eligible_from_row(row)["surface_eligible"])

    def test_invalid_surface_values_fail_before_a_tag_is_written(self):
        invalid = [
            {"R_out_thresh": 2}, {"R_out_thresh": float("inf")},
            {"R_tan_thresh": -0.1}, {"coh_thr": float("nan")}, {"coh_thr": 1.1},
            {"constraint_coverage_thresh": -1}, {"squared_displacement_coverage_thresh": float("nan")},
            {"bridge_axis_max_deg": 91}, {"bridge_max_gap_mm": float("inf")},
            {"bridge_max_gap_mm": -1}, {"patch_r_ratio": -2},
            {"stride": 1.5}, {"stride": 0}, {"min_neighbors": 0},
            {"gap_fill_size": -1}, {"gap_fill_size": 0.5},
            {"r_min_mm": 0}, {"r_max_mm": -1}, {"r_min_mm": 6, "r_max_mm": 5},
        ]
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                paths.surface_config_for_run("run", override, project_root=self.root)
        self.assertFalse(paths.surface_config_path("run", project_root=self.root).exists())
        zero = paths.surface_config_for_run("run", {"gap_fill_size": 0, "bridge_max_gap_mm": 0, "bridge_axis_max_deg": 0}, project_root=self.root)
        self.assertEqual(zero["gap_fill_size"], 0)

    def test_zero_threshold_is_used_and_missing_value_still_uses_default(self):
        from sin_trend_classify import surface_eligible_from_row
        base = dict(surface_evaluation_status="ok", R_out_latent=0.1, constraint_coverage=0.25, squared_displacement_coverage=0.25)
        default = surface_eligible_from_row(base)
        self.assertFalse(default["surface_eligible"])
        zero_coverage = surface_eligible_from_row({**base, "constraint_coverage_thresh": 0, "squared_displacement_coverage_thresh": 0})
        self.assertTrue(zero_coverage["surface_eligible"])
        zero_out = surface_eligible_from_row({**base, "constraint_coverage": 1, "squared_displacement_coverage": 1, "R_out_thresh": 0})
        self.assertEqual(zero_out["R_out_thresh_used"], 0)
        self.assertEqual(zero_out["R_tan_thresh_used"], 1)
        self.assertFalse(zero_out["surface_eligible"])
        for missing in (None, np.nan, ""):
            restored = surface_eligible_from_row({**base, "R_out_thresh": missing, "constraint_coverage_thresh": missing})
            self.assertEqual(restored["R_out_thresh_used"], 0.25)
            self.assertFalse(restored["surface_eligible"])

    def test_tampered_measurement_and_surface_configs_are_rejected(self):
        cfg = {"coh_thr": 0.75}
        paths.write_surface_config("run", cfg, project_root=self.root)
        p = paths.surface_config_path("run", project_root=self.root)
        obj = json.loads(p.read_text()); obj["config"]["coh_thr"] = 0.9
        p.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):
            paths.read_surface_config("run", project_root=self.root)
        with self.assertRaises(ValueError):
            paths.write_surface_config("run", cfg, project_root=self.root)
        p = paths.run_config_path("run", self.root)
        obj = json.loads(p.read_text()); obj["config"]["voxel_size"] = 0.7
        p.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):
            paths.resolve_surface_config("run", {}, self.root)

    def test_surface_paths_are_distinct_from_measurement_and_each_other(self):
        measured = paths.per_case_csv_path("run", "baseline", "subject", self.root)
        primary = paths.surface_per_case_csv_path("run", "baseline", "subject", "primary", self.root)
        alternate = paths.surface_per_case_csv_path("run", "baseline", "subject", "alt", self.root)
        self.assertEqual(len({measured, primary, alternate}), 3)
        self.assertNotEqual(paths.long_csv_path("run", "baseline", self.root), paths.surface_long_csv_path("run", "baseline", "primary", self.root))
        self.assertNotEqual(paths.participant_status_path("run", "baseline", "subject", self.root), paths.surface_participant_status_path("run", "baseline", "subject", "primary", self.root))
        for tag in ("", ".", "..", "../primary", "a/b"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                paths.surface_per_case_dir("run", "baseline", tag, self.root)

    def test_invalid_or_missing_saved_pointwise_is_incomplete(self):
        for status in ("missing_pointwise", "invalid_pointwise", "not_run"):
            with self.subTest(status=status):
                result, reason = paths.participant_outcome_from_counts(
                    surface_enabled=True, surface_status_counts={status: 1}
                )
                self.assertEqual(result, "partial")
                self.assertIn(status, reason)


if __name__ == "__main__":
    unittest.main()
