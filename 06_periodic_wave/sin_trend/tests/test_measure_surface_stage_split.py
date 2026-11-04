#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: test measure surface stage split."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_io_utils import compute_patch_radius_vox  # noqa: E402
from sin_trend_measure import default_measurement_config  # noqa: E402
from sin_trend_paths import (  # noqa: E402
    read_surface_config,
    split_measure_surface_config,
    validate_surface_tag,
    write_surface_config,
)
from sin_trend_surface_stage import (  # noqa: E402
    radius_median_vox_from_row,
    run_surface_stage_on_dataframe,
    apply_surface_eval_to_row,
    pointwise_path_for_row,
    run_surface_stage_for_directory,
)


class TestConfigSplit(unittest.TestCase):
    def test_split_puts_patch_in_surface_not_measure(self):
        cfg = default_measurement_config()
        measure, surface = split_measure_surface_config(cfg)
        self.assertIn("removal", measure)
        self.assertIn("search_band_mode", measure)
        self.assertNotIn("patch_r_ratio", measure)
        self.assertNotIn("bridge_max_gap_mm", measure)
        self.assertIn("patch_r_ratio", surface)
        self.assertIn("coh_thr", surface)
        self.assertIn("bridge_max_gap_mm", surface)
        self.assertIn("R_out_thresh", surface)
        self.assertEqual(measure["voxel_size"], surface["voxel_size"])

    def test_surface_config_same_tag_is_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
                                                                
                                                                              
            run_id = "run_test_surface_cfg_001"
            cfg = {"voxel_size": 0.5, "patch_r_ratio": 3.0, "r_min_mm": 2.5, "r_max_mm": 5.0, "coh_thr": 0.75}
            # monkey via project_root: results/morphometrics/... structure from run_root
            from sin_trend_paths import run_root, write_run_config
            write_run_config(run_id, {"voxel_size": 0.5}, project_root=root)

            rr = run_root(run_id, root)
            rr.mkdir(parents=True, exist_ok=True)
            rec = write_surface_config(run_id, cfg, surface_tag="primary", project_root=root, overwrite=True)
            self.assertEqual(rec["surface_tag"], "primary")
            loaded = read_surface_config(run_id, "primary", project_root=root)
            self.assertEqual(loaded["config"]["patch_r_ratio"], 3.0)
            cfg2 = dict(cfg)
            cfg2["patch_r_ratio"] = 4.0
            with self.assertRaises(RuntimeError):
                write_surface_config(run_id, cfg2, surface_tag="primary", project_root=root, overwrite=True)
            write_surface_config(run_id, cfg2, surface_tag="patch_r4", project_root=root)
            loaded2 = read_surface_config(run_id, "patch_r4", project_root=root)
            self.assertEqual(loaded2["config"]["patch_r_ratio"], 4.0)
            with self.assertRaises(RuntimeError):
                write_surface_config(
                    run_id, cfg2, surface_tag="primary", project_root=root, overwrite=False
                )

    def test_validate_surface_tag(self):
        self.assertEqual(validate_surface_tag("patch_r4"), "patch_r4")
        with self.assertRaises(ValueError):
            validate_surface_tag("../x")


class TestPatchRecomputeFromSavedRadius(unittest.TestCase):
    def test_different_patch_ratio_changes_r_patch_same_radius(self):
        row = dict(radius_median_mm=1.0)  # 2 vox at 0.5 mm
        vs = 0.5
        r_vox = radius_median_vox_from_row(row, vs)
        self.assertAlmostEqual(r_vox, 2.0)
        p3 = compute_patch_radius_vox(r_vox, patch_r_ratio=3.0, r_min_mm=2.5, r_max_mm=5.0, voxel_size=vs)
        p4 = compute_patch_radius_vox(r_vox, patch_r_ratio=4.0, r_min_mm=2.5, r_max_mm=5.0, voxel_size=vs)
        self.assertTrue(p3["ok"] and p4["ok"])
                                                   
        self.assertNotEqual(p3["r_patch_mm_raw"], p4["r_patch_mm_raw"])
                                                           
        self.assertAlmostEqual(p3["r_patch_mm"], 4.0)
        self.assertAlmostEqual(p4["r_patch_mm"], 5.0)


class TestSurfaceStageLeavesFitColumns(unittest.TestCase):
    def test_missing_pointwise_keeps_fit_fields(self):
        df = pd.DataFrame(
            [
                dict(
                    directory_id="SYN",
                    segment_id=1,
                    segment_index=0,
                    side="Lt",
                    fit_status="ok",
                    lambda_hat_mm=12.3,
                    partial_R2=0.7,
                    radius_median_mm=1.0,
                    surface_evaluation_status="not_run",
                )
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "run_test_surf_stage_001"
            from sin_trend_paths import pointwise_dir, run_root, write_run_config
            write_run_config(run_id, {"voxel_size": 0.5}, project_root=root)

            run_root(run_id, root).mkdir(parents=True, exist_ok=True)
            pointwise_dir(run_id, "baseline", root).mkdir(parents=True, exist_ok=True)
            scfg = split_measure_surface_config(default_measurement_config())[1]
            out, meta = run_surface_stage_on_dataframe(
                df,
                run_id=run_id,
                frame_reference="baseline",
                surface_cfg=scfg,
                project_root=root,
                surface_tag="primary",
                verbose=False,
            )
            self.assertEqual(int(meta["n_missing_pointwise"]), 1)
            self.assertEqual(out.iloc[0]["fit_status"], "ok")
            self.assertAlmostEqual(float(out.iloc[0]["lambda_hat_mm"]), 12.3)
            self.assertEqual(out.iloc[0]["surface_evaluation_status"], "missing_pointwise")


class TestSurfaceOnlyPatchRerun(unittest.TestCase):
    def test_surface_only_changes_patch_keeps_lambda(self):
        try:
            import pyvista  # noqa: F401
        except Exception:
            self.skipTest("pyvista required")
        from sin_trend_paths import (
            atomic_to_csv,
            brain_mesh_path,
            per_case_csv_path,
            pointwise_dir,
            run_root,
            write_run_config,
        )
        from sin_trend_surface_direction_validation import (
            centerline_along_x,
            known_periodic_displacement,
            make_plane_tri_mesh,
        )
        from sin_trend_surface_stage import run_surface_stage_for_directory

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "run_test_surf_e2e_001"
            fr = "baseline"
            did = "SYN001"
            run_root(run_id, root).mkdir(parents=True, exist_ok=True)
            measure, surface = split_measure_surface_config(
                {k: v for k, v in default_measurement_config().items() if k not in {"verbose", "enable_plots"}}
            )
            write_run_config(run_id, measure, project_root=root)
            vs = 0.5
            cl = centerline_along_x(L_mm=40.0, ds_mm=0.5, y_mm=0.0, z_mm=0.5, voxel_size_mm=vs)
            d = known_periodic_displacement(
                cl["x_mm"], amplitude_mm=0.4, lambda_mm=20.0, phase=0.0, theta_plane_deg=0.0, voxel_size_mm=vs
            )
            P = cl["P_vox"]
            T0 = np.tile(np.array([1.0, 0.0, 0.0]), (len(P), 1))
            pw = dict(P=P, P_full=P, d_latent=d, d_sin=d, T0=T0, T=T0)
            pw_dir = pointwise_dir(run_id, fr, root)
            pw_dir.mkdir(parents=True, exist_ok=True)
            npz_path = pw_dir / f"{did}_seg0_idx0_pointwise.npz"
            np.savez_compressed(
                npz_path,
                scalars_json=np.array(json.dumps({"fit_status": "ok"})),
                **{k: np.asarray(v) for k, v in pw.items()},
            )
            mesh = make_plane_tri_mesh(
                x_min_mm=-2, x_max_mm=42, y_min_mm=-8, y_max_mm=8, edge_mm=0.5, z_mm=0.0, voxel_size_mm=vs
            )
            mp = brain_mesh_path(did, "Lt", root)
            mp.parent.mkdir(parents=True, exist_ok=True)
            mesh.save(str(mp))
            row = dict(
                directory_id=did,
                segment_id=0,
                segment_index=0,
                side="Lt",
                fit_status="ok",
                radius_median_mm=1.0,
                ds_median_vox=1.0,
                voxel_size_mm=vs,
                pointwise_npz_path=str(npz_path),
                lambda_hat_mm=20.0,
                partial_R2=0.8,
                R_dir_evaluable=0,
                core_periodic=1, core_all=1, periodicity="Core",
                reject_reasons="saved_policy", classification_revision="saved_revision",
            )
            csv_path = per_case_csv_path(run_id, fr, did, root)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_to_csv(pd.DataFrame([row]), csv_path, index=False)

            measurement_bytes = csv_path.read_bytes()
            pointwise_bytes = npz_path.read_bytes()
            s3 = dict(surface)
            s3["patch_r_ratio"] = 3.0
            df3, _ = run_surface_stage_for_directory(
                did, run_id, frame_reference=fr, surface_cfg=s3, project_root=root, surface_tag="primary"
            )
            self.assertEqual(csv_path.read_bytes(), measurement_bytes)
            self.assertEqual(npz_path.read_bytes(), pointwise_bytes)
            self.assertEqual(df3.iloc[0]["surface_evaluation_status"], "ok")
            self.assertEqual(df3.iloc[0]["core_periodic"], 1)
            self.assertEqual(df3.iloc[0]["reject_reasons"], "saved_policy")
            self.assertEqual(df3.iloc[0]["classification_revision"], "saved_revision")
            self.assertEqual(df3.iloc[0]["core_inplane"], 1)
            self.assertAlmostEqual(float(df3.iloc[0]["r_patch_mm"]), 4.0)
            self.assertAlmostEqual(float(df3.iloc[0]["lambda_hat_mm"]), 20.0)

            s4 = dict(surface)
            s4["patch_r_ratio"] = 4.0
            df4, meta4 = run_surface_stage_for_directory(
                did, run_id, frame_reference=fr, surface_cfg=s4, project_root=root, surface_tag="patch_r4"
            )
            self.assertEqual(df4.iloc[0]["surface_evaluation_status"], "ok")
            self.assertAlmostEqual(float(df4.iloc[0]["r_patch_mm"]), 5.0)
            self.assertAlmostEqual(float(df4.iloc[0]["lambda_hat_mm"]), 20.0)
            self.assertTrue(Path(meta4["per_case_csv_surface"]).is_file())
            self.assertEqual(csv_path.read_bytes(), measurement_bytes)
            self.assertEqual(npz_path.read_bytes(), pointwise_bytes)


class TestSurfaceStageRegressionContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run_id = "run_stage_regressions"
        self.fr = "baseline"
        self.did = "SYN"
        from sin_trend_paths import write_run_config
        write_run_config(self.run_id, {"voxel_size": .5}, project_root=self.root)
        self.cfg = split_measure_surface_config(default_measurement_config())[1]
        self.rows = []
        for sid in (0, 1):
            from sin_trend_paths import pointwise_dir
            p = pointwise_dir(self.run_id, self.fr, self.root) / f"SYN_seg{sid}_idx{sid}_pointwise.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            P = np.column_stack([np.arange(12) * .37, np.zeros(12), np.ones(12)])
            d = np.tile([0., 1., 0.], (len(P), 1))
            np.savez_compressed(p, P=P, P_full=P, d_latent=d,
                                scalars_json=np.array(json.dumps({"directory_id": self.did, "segment_id": sid, "segment_index": sid, "frame_reference": self.fr, "voxel_size_mm": .5})))
            self.rows.append(dict(directory_id=self.did, segment_id=sid, segment_index=sid,
                frame_reference=self.fr, fit_status="ok", side="Lt", voxel_size_mm=.5,
                pointwise_npz_path=str(p), radius_median_mm=1., ds_median_vox=999.,
                lambda_hat_mm=20., partial_R2=.2, core_periodic=1, core_all=1,
                periodicity="Core", fit_valid=1, interval_eligible=1,
                classification_revision="saved_revision", reject_reasons="saved_policy",
                core_inplane=1, R_out_latent=.1, R_out_response=.2,
                squared_displacement_coverage_latent=.9, surface_evaluation_status="ok"))
        from sin_trend_paths import brain_mesh_path, per_case_csv_path
        self.mesh = brain_mesh_path(self.did, "Lt", self.root)
        self.mesh.parent.mkdir(parents=True, exist_ok=True)
        self.mesh.write_bytes(b"mock surface mesh")
        self.csv = per_case_csv_path(self.run_id, self.fr, self.did, self.root)
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.csv, index=False)

    def fake_eval(self, **kwargs):
        self.last_eval = kwargs
        ev = dict(status="ok", reason="", R_out=.1, R_tan=.9,
                  constraint_coverage=1., squared_displacement_coverage=1.)
        return ev, {"mask_eval": np.ones(len(kwargs["P_trim"]), bool)}, "test"

    def stage(self, df=None, tag="primary", segments=None):
        return run_surface_stage_on_dataframe(
            pd.DataFrame(self.rows) if df is None else df, run_id=self.run_id,
            frame_reference=self.fr, surface_cfg=self.cfg, project_root=self.root,
            surface_tag=tag, segment_ids=segments)

    def test_success_preserves_all_measurement_columns_and_uses_actual_ds(self):
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self.fake_eval):
            out, _ = self.stage()
        for i, original in enumerate(self.rows):
            for key in ("lambda_hat_mm", "partial_R2", "ds_median_vox", "core_periodic", "core_all", "periodicity", "classification_revision", "reject_reasons", "fit_valid", "interval_eligible"):
                self.assertEqual(out.iloc[i][key], original[key], key)
        self.assertAlmostEqual(self.last_eval["ds_vox"], .37)
        self.assertTrue(np.isnan(out.iloc[0]["R_out_response"]))
        self.assertEqual(out.iloc[0]["core_inplane"], 1)

    def test_missing_pointwise_clears_prior_surface_and_preserves_core(self):
        Path(self.rows[0]["pointwise_npz_path"]).unlink()
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self.fake_eval):
            out, _ = self.stage()
        self.assertEqual(out.iloc[0]["surface_evaluation_status"], "missing_pointwise")
        self.assertTrue(np.isnan(out.iloc[0]["R_out_latent"]))
        self.assertTrue(np.isnan(out.iloc[0]["R_out_response"]))
        self.assertEqual(out.iloc[0]["core_periodic"], 1)
        self.assertEqual(out.iloc[0]["core_inplane"], 0)

    def test_bad_npz_does_not_abort_other_rows(self):
        Path(self.rows[0]["pointwise_npz_path"]).write_bytes(b"broken")
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self.fake_eval):
            out, _ = self.stage()
        self.assertEqual(out.iloc[0]["surface_evaluation_status"], "invalid_pointwise")
        self.assertEqual(out.iloc[1]["surface_evaluation_status"], "ok")
        self.assertTrue(np.isnan(out.iloc[0]["R_out_latent"]))

    def test_bad_metadata_and_shape_are_rejected(self):
        p = Path(self.rows[0]["pointwise_npz_path"])
        np.savez(p, P=np.zeros((5,3)), d_latent=np.ones((4,3)), scalars_json=np.array('{"segment_id":99}'))
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self.fake_eval):
            out, _ = self.stage()
        self.assertEqual(out.iloc[0]["surface_evaluation_status"], "invalid_pointwise")
        self.assertIn("segment_id mismatch", out.iloc[0]["surface_evaluation_reason"])
        np.savez(p, P=np.column_stack([np.arange(5),np.zeros(5),np.zeros(5)]), d_latent=np.ones((4,3)), scalars_json=np.array('{}'))
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self.fake_eval):
            out, _ = self.stage()
        self.assertIn("d_latent shape", out.iloc[0]["surface_evaluation_reason"])

    def test_other_run_npz_pointer_is_rejected(self):
        self.rows[0]["pointwise_npz_path"] = self.rows[1]["pointwise_npz_path"]
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self.fake_eval):
            out, _ = self.stage()
        self.assertEqual(out.iloc[0]["surface_evaluation_status"], "invalid_pointwise")
        self.assertIn("requested run/frame/segment", out.iloc[0]["surface_evaluation_reason"])

    def test_row_voxel_mismatch_is_rejected(self):
        self.rows[0]["voxel_size_mm"] = .7
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self.fake_eval):
            out, _ = self.stage()
        self.assertEqual(out.iloc[0]["surface_evaluation_status"], "invalid_pointwise")
        self.assertIn("voxel_size", out.iloc[0]["surface_evaluation_reason"])

    def test_surface_failure_does_not_retain_prior_metrics(self):
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=RuntimeError("mesh error")):
            out, _ = self.stage()
        self.assertEqual(out.iloc[0]["surface_evaluation_status"], "surface_eval_failed")
        self.assertTrue(np.isnan(out.iloc[0]["R_out_latent"]))
        self.assertEqual(out.iloc[0]["core_periodic"], 1)

    def test_new_tag_partial_rerun_does_not_mix_old_surface_rows(self):
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self.fake_eval):
            out, _ = self.stage(tag="new", segments=[0])
        self.assertEqual(out.iloc[1]["surface_evaluation_status"], "not_run")
        self.assertEqual(out.iloc[1]["surface_tag"], "new")
        self.assertTrue(np.isnan(out.iloc[1]["R_out_latent"]))
        self.assertEqual(out.iloc[1]["core_periodic"], 1)

    def test_same_tag_partial_rerun_preserves_same_input_results(self):
        args = dict(directory_id=self.did, run_id=self.run_id, frame_reference=self.fr,
                    surface_cfg=self.cfg, project_root=self.root, surface_tag="primary")
        original_bytes = self.csv.read_bytes()
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self.fake_eval):
            full, meta = run_surface_stage_for_directory(**args)
            second, _ = run_surface_stage_for_directory(**args, segment_ids=[0])
        self.assertEqual(second.iloc[1]["surface_evaluation_status"], "ok")
        self.assertAlmostEqual(second.iloc[1]["R_out_latent"], .1)
        self.assertEqual(self.csv.read_bytes(), original_bytes)
        self.assertNotEqual(Path(meta["per_case_csv_surface"]), self.csv)
        self.assertTrue(Path(meta["per_case_csv_surface"]).is_file())

    def test_partial_rerun_invalidates_unselected_changed_npz(self):
        args = dict(directory_id=self.did, run_id=self.run_id, frame_reference=self.fr,
                    surface_cfg=self.cfg, project_root=self.root, surface_tag="primary")
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self.fake_eval):
            run_surface_stage_for_directory(**args)
            Path(self.rows[1]["pointwise_npz_path"]).write_bytes(b"changed")
            second, _ = run_surface_stage_for_directory(**args, segment_ids=[0])
        self.assertEqual(second.iloc[1]["surface_evaluation_status"], "not_run")
        self.assertTrue(np.isnan(second.iloc[1]["R_out_latent"]))


    def _prepare_plot_arrays(self):
        for row in self.rows:
            p = Path(row["pointwise_npz_path"])
            with np.load(p, allow_pickle=False) as saved:
                arrays = {key: np.array(saved[key], copy=True) for key in saved.files}
            arrays["P0_full"] = arrays["P"].copy()
            arrays["P0"] = arrays["P"].copy()
            np.savez_compressed(p, **arrays)

    def _fake_eval_for_plot(self, **kwargs):
        ev, est, cache = self.fake_eval(**kwargs)
        est["normals_full"] = np.tile([0., 0., 1.], (len(kwargs["P_trim"]), 1))
        est["anchor_indices"] = np.array([0, len(kwargs["P_trim"])-1])
        return ev, est, cache

    def test_surface_plot_saved_under_tag_without_mutating_measurements(self):
        self._prepare_plot_arrays()
        before_csv = self.csv.read_bytes()
        before_npz = {row["pointwise_npz_path"]: Path(row["pointwise_npz_path"]).read_bytes() for row in self.rows}
        seen = []
        def exporter(**kwargs):
            seen.append(kwargs)
            kwargs["out_path"].parent.mkdir(parents=True, exist_ok=True)
            kwargs["out_path"].write_text("<html>surface</html>", encoding="utf-8")
            # A plotting consumer may modify its arrays; source artifacts must remain untouched.
            kwargs["P_trim"][0] = -123
            return "ok"
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self._fake_eval_for_plot), patch("sin_trend_visualize.save_surface_constraint_html", side_effect=exporter):
            out, meta = run_surface_stage_for_directory(self.did, self.run_id,
                frame_reference=self.fr, surface_cfg=self.cfg, project_root=self.root,
                surface_tag="with_plots", enable_plots=True)
        self.assertEqual(self.csv.read_bytes(), before_csv)
        for path, content in before_npz.items():
            self.assertEqual(Path(path).read_bytes(), content)
        self.assertEqual(out.iloc[0]["viz_surface_html_status"], "ok")
        result = Path(out.iloc[0]["viz_surface_html_path"])
        self.assertTrue(result.is_file())
        self.assertIn("surface/with_plots/figures/SYN", str(result))
        self.assertEqual(meta["surface_plot_status_counts"], {"ok": 2})
        self.assertEqual(len(seen), 2)
        self.assertTrue(np.all(seen[0]["n_eff"][:, 2] == 1))
        self.assertEqual(out.iloc[0]["core_periodic"], 1)

    def test_plot_failure_is_explicit_but_keeps_numerical_surface_result(self):
        self._prepare_plot_arrays()
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self._fake_eval_for_plot), patch("sin_trend_visualize.save_surface_constraint_html", return_value="failed:renderer unavailable"):
            out, _ = run_surface_stage_for_directory(self.did, self.run_id,
                frame_reference=self.fr, surface_cfg=self.cfg, project_root=self.root,
                surface_tag="plot_failure", enable_plots=True)
        self.assertEqual(out.iloc[0]["surface_evaluation_status"], "ok")
        self.assertAlmostEqual(out.iloc[0]["R_out_latent"], .1)
        self.assertEqual(out.iloc[0]["viz_surface_html_status"], "failed:renderer unavailable")
        self.assertEqual(out.iloc[0]["viz_surface_html_path"], "")

    def test_plots_are_disabled_by_default(self):
        with patch("sin_trend_surface_stage.run_surface_evaluation", side_effect=self._fake_eval_for_plot), patch("sin_trend_visualize.save_surface_constraint_html") as exporter:
            out, _ = self.stage()
        exporter.assert_not_called()
        self.assertEqual(out.iloc[0]["viz_surface_html_status"], "not_requested")



if __name__ == "__main__":
    unittest.main()
