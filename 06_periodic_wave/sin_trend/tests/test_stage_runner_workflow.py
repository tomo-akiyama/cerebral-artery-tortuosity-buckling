"""Runner/batch contracts: persisted measurements precede independent surface runs."""
from pathlib import Path
from contextlib import ExitStack
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import sin_trend_paths as paths
from sin_trend_surface_direction_validation import make_plane_tri_mesh


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestStageRunnerWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_script("runner_stage_workflow", "01_run_sin_trend_fitting.py")
        cls.batch = load_script("batch_stage_workflow", "02_batch_sin_trend_fitting.py")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.did, self.run, self.frame = "SYN001", "workflow", "baseline"
        self.ctx = ExitStack()
        self.ctx.enter_context(patch.object(paths, "package_versions", return_value={}))
        self.ctx.enter_context(patch.object(paths, "source_code_hashes", side_effect=lambda files=paths.BASELINE_SOURCE_FILES, **kw: {n: "stable_test_hash" for n in files}))
        self.ctx.enter_context(patch.object(self.runner, "_setup_import_paths"))
        self.ctx.enter_context(patch.object(self.runner, "project_root_from_here", return_value=self.root))
        self.ctx.enter_context(patch.object(self.batch, "project_root_from_here", return_value=self.root))

    def tearDown(self):
        self.ctx.close()
        self.tmp.cleanup()

    def make_measurement(self, do_surface=True):
        x = np.arange(401) * 0.1
        P = np.column_stack([x, 0.4 * np.sin(2 * np.pi * x / 20), np.full(len(x), 0.5)]) / 0.5
        pids = [np.arange(len(P))]
        io = types.ModuleType("centerline_extraction")
        io.load_vtp = lambda _: object()
        io.vmtk_convert2numpy = lambda _: ({"GroupIds": np.array([0])}, pids, P, {"MaximumInscribedSphereRadius": np.full(len(P), 1.4)})
        features = types.ModuleType("vessel_mesh_features")
        features.classify_centerline_left_right_by_kmeans = lambda *a, **kw: (np.array([], dtype=int), np.array([0]))
        vtp = paths.centerline_vtp_path(self.did, self.root)
        vtp.parent.mkdir(parents=True, exist_ok=True)
        vtp.write_bytes(b"controlled synthetic VTP loader fixture")
        mesh = make_plane_tri_mesh(x_min_mm=-5, x_max_mm=45, y_min_mm=-6, y_max_mm=6, edge_mm=0.5, voxel_size_mm=0.5)
        mp = paths.brain_mesh_path(self.did, "Lt", self.root)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mesh.save(str(mp))
        with patch.dict(sys.modules, {"centerline_extraction": io, "vessel_mesh_features": features}), patch.object(self.runner, "measure_interval", wraps=self.runner.measure_interval) as measure:
            df, status = self.runner.run_directory(self.did, self.run, project_root=self.root, cfg={"enable_plots": False, "verbose": False}, do_surface=do_surface)
        self.assertEqual(measure.call_count, 1)
        self.assertEqual(status["status"], "complete", status)
        measured = paths.per_case_csv_path(self.run, self.frame, self.did, self.root)
        row = pd.read_csv(measured).iloc[0]
        npz = Path(row["pointwise_npz_path"])
        self.assertEqual(row["fit_status"], "ok")
        self.assertTrue(npz.is_file())
        return df, status, measured, npz

    def measurement_snapshot(self, csv, npz):
        files = [csv, npz, paths.run_config_path(self.run, self.root), paths.participant_status_path(self.run, self.frame, self.did, self.root)]
        return {p: p.read_bytes() for p in files}

    def test_full_run_persists_measurement_then_writes_separate_surface(self):
        df, status, csv, npz = self.make_measurement()
        self.assertEqual(pd.read_csv(csv).iloc[0]["surface_evaluation_status"], "not_run")
        measured_status = json.loads(paths.participant_status_path(self.run, self.frame, self.did, self.root).read_text())
        self.assertFalse(measured_status["surface_enabled"])
        self.assertEqual(measured_status["per_case_csv_sha256"], paths.sha256_file(csv))
        self.assertEqual(status["surface_tag"], "primary")
        surfaced = paths.surface_per_case_csv_path(self.run, self.frame, self.did, "primary", self.root)
        self.assertTrue(surfaced.is_file())
        self.assertEqual(pd.read_csv(surfaced).iloc[0]["surface_evaluation_status"], "ok")
        with np.load(npz, allow_pickle=False) as arrays:
            self.assertNotIn("n_eff", arrays.files)

    def test_surface_only_new_patch_tag_never_calls_measurement_or_changes_artifacts(self):
        _, _, csv, npz = self.make_measurement()
        before = self.measurement_snapshot(csv, npz)
        override = self.root / "surface_override.json"
        override.write_text(json.dumps({"patch_r_ratio": 4.0}))
        # Surface-only must remain usable after the original centerline is unavailable.
        paths.centerline_vtp_path(self.did, self.root).unlink()
        with patch.object(self.runner, "measure_interval", side_effect=AssertionError("surface-only remeasured")), patch.object(self.runner, "run_directory", side_effect=AssertionError("surface-only entered measurement stage")):
            code = self.runner.main(["--directory-id", self.did, "--run-id", self.run, "--surface-only", "--surface-tag", "patch4", "--surface-config-json", str(override), "--no-plots", "--quiet"])
        self.assertEqual(code, 0)
        self.assertEqual(before, self.measurement_snapshot(csv, npz))
        out = pd.read_csv(paths.surface_per_case_csv_path(self.run, self.frame, self.did, "patch4", self.root))
        self.assertEqual(out.iloc[0]["patch_r_ratio"], 4.0)
        self.assertEqual(out.iloc[0]["surface_evaluation_status"], "ok")

    def test_rejected_same_tag_change_preserves_completed_surface_record(self):
        _, _, csv, npz = self.make_measurement()
        status_path = paths.surface_participant_status_path(self.run, self.frame, self.did, "primary", self.root)
        surface_csv = paths.surface_per_case_csv_path(self.run, self.frame, self.did, "primary", self.root)
        before = self.measurement_snapshot(csv, npz)
        before[status_path] = status_path.read_bytes()
        before[surface_csv] = surface_csv.read_bytes()
        changed = paths.surface_config_for_run(self.run, {"r_min_mm": 3.0}, project_root=self.root)
        with self.assertRaisesRegex(RuntimeError, "new surface_tag"):
            self.runner.run_saved_surface(self.did, self.run, frame_reference=self.frame,
                surface_cfg=changed, project_root=self.root, surface_tag="primary")
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content, str(path))

    def test_batch_surface_dry_run_discovers_saved_csv_without_vtp_and_preserves_configs(self):
        _, _, csv, npz = self.make_measurement(do_surface=False)
        paths.centerline_vtp_path(self.did, self.root).unlink()
        before = self.measurement_snapshot(csv, npz)
        config_path = paths.surface_config_path(self.run, "primary", self.root)
        self.assertFalse(config_path.exists())
        with patch.object(self.batch, "participants_csv_path", side_effect=AssertionError("surface-only accessed participant roster")), patch.object(self.batch, "write_run_config", side_effect=AssertionError("dry run wrote measurement config")), patch.object(self.batch, "write_surface_config", side_effect=AssertionError("dry run wrote surface config")):
            rc = self.batch.main(["--run-id", self.run, "--surface-only", "--all", "--dry-run", "--no-plots"])
        self.assertEqual(rc, 0)
        plan = json.loads((paths.run_root(self.run, self.root) / "surface_batches" / "primary" / "batch_dry_run_plan.json").read_text())
        self.assertEqual(plan["directory_ids"], [self.did])
        self.assertEqual(before, self.measurement_snapshot(csv, npz))
        self.assertFalse(config_path.exists())

    def test_full_batch_resume_skips_when_measure_and_surface_complete(self):
        args = self.batch.parse_args(["--run-id", self.run, "--directory-id", self.did, "--resume", "--no-plots"])
        logs = self.root / "logs"; logs.mkdir()
        called = []
        def child(cmd, **kwargs):
            called.append(cmd)
            return types.SimpleNamespace(returncode=0)
        with patch.object(self.batch, "read_participant_status", return_value={"surface_enabled": True}), patch.object(self.batch, "verify_participant_complete_for_resume", return_value=(True, "ok")) as verify, patch.object(self.batch.subprocess, "run", side_effect=child):
            rec = self.batch.run_one_participant(runner=HERE / "01_run_sin_trend_fitting.py", project_root=self.root, directory_id=self.did, run_id=self.run, frames=[self.frame], args=args, log_dir=logs)
        self.assertTrue(verify.call_args.kwargs["expect_surface"])
        self.assertEqual(len(called), 0)
        self.assertEqual(rec[0]["outcome"], "skipped_complete")

    def test_full_batch_resume_uses_surface_only_child_when_measurement_is_complete(self):
        args = self.batch.parse_args(["--run-id", self.run, "--directory-id", self.did, "--resume", "--no-plots"])
        logs = self.root / "logs"; logs.mkdir()
        called = []
        def child(cmd, **kwargs):
            called.append(cmd)
            return types.SimpleNamespace(returncode=0)
        def verify_side_effect(*_a, **kwargs):
            if kwargs.get("expect_surface"):
                return False, "surface_incomplete:surface_eval_failed"
            return True, "ok"
        with patch.object(self.batch, "read_participant_status", return_value={"surface_enabled": True}), patch.object(self.batch, "verify_participant_complete_for_resume", side_effect=verify_side_effect) as verify, patch.object(self.batch.subprocess, "run", side_effect=child):
            rec = self.batch.run_one_participant(runner=HERE / "01_run_sin_trend_fitting.py", project_root=self.root, directory_id=self.did, run_id=self.run, frames=[self.frame], args=args, log_dir=logs)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(len(called), 1)
        self.assertIn("--surface-only", called[0])
        self.assertEqual(rec[0]["outcome"], "complete")

    def test_surface_only_resume_never_queries_measurement_completion(self):
        args = self.batch.parse_args(["--run-id", self.run, "--directory-id", self.did, "--surface-only", "--resume", "--no-plots"])
        logs = self.root / "logs"; logs.mkdir()
        with patch.object(self.batch, "verify_participant_complete_for_resume", side_effect=AssertionError("surface-only queried measurement")), patch.object(self.batch.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as child:
            self.batch.run_one_participant(runner=HERE / "01_run_sin_trend_fitting.py", project_root=self.root, directory_id=self.did, run_id=self.run, frames=[self.frame], args=args, log_dir=logs)
        self.assertIn("--surface-only", child.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
