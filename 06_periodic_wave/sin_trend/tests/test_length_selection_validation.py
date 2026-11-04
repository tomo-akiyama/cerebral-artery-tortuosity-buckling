#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: test length selection validation."""

from __future__ import annotations

import sys
import unittest
import copy
import importlib
import json
import tempfile
import os
import signal
import subprocess
import time
from unittest.mock import patch

import pandas as pd
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_classify import classify_row, core_periodic_from_row
from sin_trend_length_selection_validation import (
    N_true_from_xi,
    cut_prefix_window_for_n_true,
    enrichment_from_measure,
    make_parent_centerline,
    measure_and_classify_centerline,
    measurement_config_from_validation_cfg,
    load_config,
    apply_smoke_overrides,
)
from sin_trend_measure import default_measurement_config
from sin_trend_synthetic import make_sinusoid_centerline, n_cycles_input_for_target_remaining_after_trim


class TestLengthSelectionContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.vcfg = apply_smoke_overrides(load_config())
        self.meas_cfg = measurement_config_from_validation_cfg(self.vcfg)

    def test_N_cyc_search_not_rewritten_to_1p5(self) -> None:
        self.assertAlmostEqual(float(self.meas_cfg["N_cyc_search"]), 0.5)

    def test_long_clean_sinusoid_recovers_wavelength(self) -> None:
        n_in = n_cycles_input_for_target_remaining_after_trim(
            target_remaining=3.0,
            amplitude_mm=1.0,
            lambda_x_mm=20.0,
            phase=0.0,
            target_ds_mm=0.1,
            removal_points=10,
            diameter_mm=1.4,
        )
        syn = make_sinusoid_centerline(
            amplitude_mm=1.0,
            lambda_x_mm=20.0,
            n_cycles=n_in,
            phase=0.0,
            target_ds_mm=0.1,
            diameter_mm=1.4,
        )
        meas = measure_and_classify_centerline(
            syn["P_vox"], cfg=self.meas_cfg, radius_vox=syn["radius_vox"]
        )
        self.assertEqual(meas["fit_status"], "ok")
        self.assertTrue(np.isfinite(meas["lambda_hat_mm"]))
                                           
        self.assertEqual(int(meas["core_all"]), int(meas["core_periodic"]))
        self.assertEqual(int(meas["core_all"]), int(meas["core_periodic_recheck"]))

    def test_window_N_true_around_1p5_and_phase_variants(self) -> None:
        parent = make_parent_centerline(
            amplitude_mm=1.0,
            lambda_true_mm=20.0,
            phase_rad=0.0,
            diameter_mm=1.4,
            target_ds_mm=0.1,
            removal=10,
            max_n_true=3.0,
            parent_extra_cycles=0.75,
            background=None,
            noise_sd_mm=0.0,
            seed=1,
            voxel_size_mm=0.5,
        )
        for n_t, phase in ((1.45, 0.0), (1.5, 0.0), (1.55, np.pi / 2)):
            if phase != 0.0:
                parent = make_parent_centerline(
                    amplitude_mm=1.0,
                    lambda_true_mm=20.0,
                    phase_rad=float(phase),
                    diameter_mm=1.4,
                    target_ds_mm=0.1,
                    removal=10,
                    max_n_true=3.0,
                    parent_extra_cycles=0.75,
                    background=None,
                    noise_sd_mm=0.0,
                    seed=2,
                    voxel_size_mm=0.5,
                )
            win = cut_prefix_window_for_n_true(
                parent, n_true_target=n_t, removal=10, atol=0.02
            )
            self.assertTrue(win["ok"], msg=win.get("reason"))
                                      
            n_true = N_true_from_xi(win["xi_mm"], 20.0, 10)
            self.assertAlmostEqual(n_true, win["n_true_achieved"], places=10)
            meas = measure_and_classify_centerline(
                win["P_vox"], cfg=self.meas_cfg, radius_vox=win["radius_vox"]
            )
                                                                      
            self.assertTrue(np.isfinite(win["L0_true_trimmed_mm"]))
            self.assertAlmostEqual(
                float(n_true),
                float(win["L0_true_trimmed_mm"]) / 20.0,
                places=10,
            )
            if meas.get("fit_status") == "ok":
                self.assertIn("L_base_mm", meas)
                self.assertIn("L_trimmed_mm", meas)
                                                
                self.assertEqual(int(meas["core_all"]), int(meas["core_periodic_recheck"]))

    def test_accepted_N_true_lt_1p5_enrichment_fields_exist(self) -> None:
        en = enrichment_from_measure(
            lambda_true_mm=20.0,
            L0_true_trimmed_mm=25.0,
            N_true=1.2,
            meas={
                "lambda_hat_mm": 10.0,
                "L_base_mm": 20.0,
                "coverage_hat": 2.0,
                "core_all": 1,
                "amp_comb_mm": 1.0,
                "lambda_min_search_mm": 2.8,
                "lambda_max_search_mm": 40.0,
                "boundary_exact_low": False,
                "boundary_exact_high": False,
            },
        )
        self.assertEqual(en["accepted_N_true_lt_1p5"], 1)
        self.assertNotIn("false_accept_N_true_lt_1p5", en)
        self.assertTrue(np.isfinite(en["N_hat_inflation_from_lambda_underest"]))

    def test_classify_row_agreement(self) -> None:
        syn = make_sinusoid_centerline(
            amplitude_mm=1.0,
            lambda_x_mm=20.0,
            n_cycles=4.0,
            phase=0.0,
            target_ds_mm=0.1,
            diameter_mm=1.4,
        )
        from sin_trend_measure import measure_interval
        from sin_trend_paths import FRAME_BASELINE

        res = measure_interval(
            syn["P_vox"],
            frame_reference=FRAME_BASELINE,
            cfg=self.meas_cfg,
            radius_vox=syn["radius_vox"],
            compute_pca=False,
            verbose=False,
        )
        row = classify_row(res["row"])
        self.assertEqual(int(row["core_all"]), int(core_periodic_from_row(row)))


class TestLengthSelectionSafeguards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = importlib.import_module("sin_trend_length_selection_validation")
        cls.cli = importlib.import_module("40_length_selection_validation")

    def setUp(self):
        self.cfg = apply_smoke_overrides(load_config())
        self.meas = measurement_config_from_validation_cfg(self.cfg)
        self.tmp = tempfile.TemporaryDirectory(prefix="lengthsel_test_")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def _fp(self, cfg=None, count=12, suite="all"):
        return self.helper.resume_fingerprint(
            cfg or self.cfg, seed=1, clinical_sha="test",
            run_spec={"n_per_replicate": count, "suite": suite})

    def test_fingerprint_uses_effective_settings_and_plan(self):
        self.assertNotEqual(self._fp(), self._fp(load_config()))
        self.assertNotEqual(self._fp(), self._fp(count=13))
        self.assertNotEqual(self._fp(), self._fp(suite="A"))
        files = self._fp()["code_hashes"]
        for suffix in ("sin_trend_forward_fit.py", "sin_trend_frames.py", "sin_trend_fitting_helpers.py"):
            self.assertTrue(any(str(name).endswith(suffix) for name in files), suffix)
        (self.path / "run_fingerprint.json").write_text(json.dumps(self._fp()))
        self.helper.check_resume_consistency(self.path, self._fp())
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            self.helper.check_resume_consistency(self.path, self._fp(count=13))

    def test_missing_or_legacy_fingerprint_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "without"):
            self.helper.check_resume_consistency(self.path, self._fp())
        (self.path / "run_fingerprint.json").write_text('{"schema_version":1}')
        with self.assertRaisesRegex(RuntimeError, "Legacy"):
            self.helper.check_resume_consistency(self.path, self._fp())

    def test_journal_saves_nan_as_null_and_rejects_duplicates(self):
        p = self.path / "cases.jsonl"
        j = self.cli.CaseJournal(p, {"a", "b"})
        j.append({"case_id": "a", "number": np.nan})
        with self.assertRaises(RuntimeError):
            j.append({"case_id": "a"})
        with self.assertRaises(RuntimeError):
            j.append({"case_id": "outside"})
        j.close()
        self.assertIsNone(json.loads(p.read_text())["number"])
        j = self.cli.CaseJournal(p, {"a", "b"})
        self.assertEqual(set(j.records), {"a"})
        j.append({"case_id": "b", "number": np.float64(2)})
        j.close()
        self.assertEqual(len(p.read_text().splitlines()), 2)

    def test_journal_recovers_only_incomplete_tail(self):
        p = self.path / "cases.jsonl"
        p.write_bytes(b'{"case_id":"a"}\n{"case_id":')
        j = self.cli.CaseJournal(p, {"a", "b"})
        self.assertGreater(j.recovered_tail_bytes, 0)
        self.assertEqual(set(j.records), {"a"})
        j.append({"case_id": "b"})
        j.close()
        self.assertEqual(len(p.read_text().splitlines()), 2)
        p.write_bytes(b'{"case_id":"a"}\nINVALID\n')
        with self.assertRaises(RuntimeError):
            self.cli.CaseJournal(p, {"a", "b"})

    def test_all_generation_failures_are_summarized_without_fabricating_truth(self):
        helper = self.helper
        jobs = helper.iter_suite_A_jobs(self.cfg, seed=1)
        with patch.object(helper, "make_parent_centerline", side_effect=ValueError("forced failure")):
            a = helper.run_suite_A_job(jobs[0], self.meas, self.cfg)
        self.assertEqual(a["generation_status"], "failed")
        self.assertEqual(a["execution_status"], "not_run")
        self.assertTrue(np.isnan(a["N_true"]))
        sa = helper.aggregate_suite_A(pd.DataFrame([a]))
        self.assertEqual(int(sa.iloc[0].n_N_true_unknown), 1)
        self.assertEqual(int(sa.iloc[0].n_generation_failed), 1)
        pool = helper.ClinicalPool(np.array([1., 1.4]), np.array([10., 20.]), "test", "test", 2, 2, {})
        bj = helper.build_suite_B_replicate_jobs(self.cfg, seed=1, replicate=0, n_per_replicate=2, pool=pool)
        with patch.object(helper, "generate_population_centerline", return_value={"ok":False, "reason":"forced"}):
            rows = [helper.run_suite_B_job(j, self.meas, self.cfg) for j in bj]
        b = pd.DataFrame(rows)
        self.assertTrue(b.N_true.isna().all())
        self.assertTrue(b.n_true_target.notna().all())
        sb = helper.aggregate_suite_B_replicate(b)
        self.assertTrue((sb.n_accepted == 0).all())
        self.assertTrue((sb.n_generation_failed == 2).all())
        self.assertEqual(len(helper.aggregate_suite_B_across_replicates(sb)), 10)

    def test_length_accuracy_is_measured_in_mm(self):
        for lam in (10., 20.):
            gen = self.helper.generate_population_centerline(
                D_mm=1.4, L0_true_trimmed_target_mm=1.5860133459985342,
                lambda_true_mm=lam, amplitude_mm=1., phase_rad=.3,
                target_ds_mm=.1, removal=10, voxel_size_mm=.5,
                background=None, noise_sd_mm=0., seed=1,
                length_atol_mm=1e-4, length_rtol=1e-6)
            self.assertTrue(gen["ok"], gen.get("reason"))
            self.assertLessEqual(abs(gen["length_error_mm"]), gen["length_tolerance_mm"])
            self.assertAlmostEqual(gen["N_true"] * lam, gen["L0_true_trimmed_mm"])


    def test_unattainable_length_is_reported_instead_of_relaxing_tolerance(self):
        gen = self.helper.generate_population_centerline(
            D_mm=1.4, L0_true_trimmed_target_mm=1.5860133459985342,
            lambda_true_mm=40., amplitude_mm=1., phase_rad=.3,
            target_ds_mm=.1, removal=10, voxel_size_mm=.5,
            background=None, noise_sd_mm=0., seed=1,
            length_atol_mm=1e-4, length_rtol=1e-6)
        self.assertFalse(gen["ok"])
        self.assertIn("n_cycles_solve_failed", gen["reason"])
        self.assertNotIn("N_true", gen)

    def test_execution_exception_is_not_a_generation_failure(self):
        job = self.helper.iter_suite_A_jobs(self.cfg, seed=1)[0]
        with patch.object(self.helper, "measure_and_classify_centerline", side_effect=RuntimeError("forced")):
            row = self.helper.run_suite_A_job(job, self.meas, self.cfg)
        self.assertEqual(row["generation_status"], "ok")
        self.assertEqual(row["execution_status"], "exception")
        self.assertTrue(np.isfinite(row["N_true"]))

    def test_cli_dry_run_and_existing_outputs_are_protected(self):
        out = self.path / "existing"
        out.mkdir()
        p = out / "run_fingerprint.json"
        p.write_text('{"preserve":"this"}')
        argv = ["--suite", "A", "--output-dir", str(out), "--no-plots", "--quiet"]
        original = {f.name: f.read_bytes() for f in out.iterdir()}
        self.assertEqual(self.cli.main(argv + ["--dry-run", "--seed", "2"]), 0)
        self.assertEqual(original, {f.name: f.read_bytes() for f in out.iterdir()})
        with self.assertRaises(FileExistsError):
            self.cli.main(argv)
        missing = self.path / "missing"
        self.cli.main(["--suite", "A", "--dry-run", "--output-dir", str(missing)])
        self.assertFalse(missing.exists())

    def test_cli_partial_exit_and_unknown_truth_on_generation_failures(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["suite_A"]["phase_rad"] = [0.]
        cfg["suite_A"]["n_true_targets"] = [1.45]
        cp = self.path / "config.json"
        cp.write_text(json.dumps(cfg))
        out = self.path / "partial"
        with patch.object(self.helper, "make_parent_centerline", side_effect=ValueError("forced")):
            result = self.cli.main(["--suite", "A", "--config", str(cp), "--output-dir", str(out),
                                    "--no-plots", "--quiet"])
        self.assertEqual(result, 2)
        self.assertEqual(json.loads((out / "run_state.json").read_text())["status"], "partial")
        self.assertTrue(pd.read_csv(out / "all_cases.csv").N_true.isna().all())

    def test_crosscheck_failure_produces_partial_exit(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["suite_A"]["phase_rad"] = [0.]
        cfg["suite_A"]["n_true_targets"] = [1.5]
        cp = self.path / "config.json"
        cp.write_text(json.dumps(cfg))
        out = self.path / "cross_failure"
        with patch.object(self.cli, "crosscheck_with_36_short_coverage", return_value={"ok": False}):
            result = self.cli.main(["--suite", "A", "--config", str(cp), "--output-dir", str(out),
                                    "--no-plots", "--quiet"])
        self.assertEqual(result, 2)
        self.assertEqual(json.loads((out / "metadata.json").read_text())["status"], "partial")



    def test_amplitude_additions_are_preserved(self):
        row = self.helper.enrichment_from_measure(
            lambda_true_mm=20., L0_true_trimmed_mm=30., N_true=1.5,
            A_true_mm=1., meas={"lambda_hat_mm":20., "L_base_mm":30.,
                               "coverage_hat":1.5, "amp_comb_mm":.8, "core_all":1})
        self.assertAlmostEqual(row["A_hat_over_true"], .8)
        self.assertAlmostEqual(row["A_abs_rel_error"], .2)
        self.assertAlmostEqual(row["A_error_signed_mm"], -.2)
        legacy = self.helper.ensure_amplitude_ratio_columns(
            pd.DataFrame({"A_true_mm":[1.], "amp_comb_mm":[.8]}))
        self.assertAlmostEqual(legacy.A_hat_over_true.iloc[0], .8)
        failed = self.helper.ensure_amplitude_ratio_columns(pd.DataFrame({"A_true_mm":[1.]}))
        self.assertTrue(failed.A_hat_over_true.isna().all())
        job = self.helper.iter_suite_A_jobs(self.cfg, seed=1)[0]
        old_row = self.helper.run_suite_A_job(job, self.meas, self.cfg)
        reference_ratio = old_row["A_hat_over_true"]
        for key in ("A_hat_mm", "A_hat_over_true", "A_error_signed_mm", "A_abs_rel_error"):
            old_row.pop(key, None)
        summary = self.helper.aggregate_suite_A(pd.DataFrame([old_row]))
        self.assertAlmostEqual(summary.A_hat_over_true_median_measured.iloc[0], reference_ratio)

    def test_real_interrupt_resume_and_parallel_results_match(self):
        clinical = self.path / "inputs.csv"
        pd.DataFrame({"radius_median_mm":[.4,.6,.8,1.0],
                      "L_base_mm":[10.,20.,30.,40.]}).to_csv(clinical, index=False)
        script = _HERE / "40_length_selection_validation.py"
        argv = [sys.executable, "-B", str(script), "--smoke", "--seed", "20261027",
                "--n-replicates", "1", "--n-per-replicate", "4",
                "--clinical-csv", str(clinical), "--no-plots", "--quiet"]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        serial, resumed = self.path / "serial", self.path / "resumed"
        log = self.path / "execution.log"

        def execute(extra):
            with log.open("a") as f:
                result = subprocess.run(argv + extra, stdout=f, stderr=subprocess.STDOUT,
                                        env=env, timeout=60)
            self.assertIn(result.returncode, (0, 2), log.read_text())

        execute(["--output-dir", str(serial)])
        with log.open("a") as f:
            proc = subprocess.Popen(argv + ["--output-dir", str(resumed)],
                                    stdout=f, stderr=subprocess.STDOUT, env=env)
            try:
                journal = resumed / "case_results.jsonl"
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if journal.exists() and len(journal.read_bytes().splitlines()) >= 2:
                        proc.send_signal(signal.SIGINT)
                        break
                    self.assertIsNone(proc.poll(), log.read_text())
                    time.sleep(.01)
                else:
                    self.fail("No saved cases before timeout")
                self.assertEqual(proc.wait(timeout=30), 130, log.read_text())
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        prefix = journal.read_bytes()
        self.assertGreaterEqual(len(prefix.splitlines()), 2)
        execute(["--output-dir", str(resumed), "--resume", "--n-workers", "2"])
        self.assertTrue(journal.read_bytes().startswith(prefix))
        a = pd.read_csv(serial / "all_cases.csv").sort_values("case_id").reset_index(drop=True)
        b = pd.read_csv(resumed / "all_cases.csv").sort_values("case_id").reset_index(drop=True)
        self.assertEqual(len(b), 16)
        self.assertTrue(b.case_id.is_unique)
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12, rtol=1e-12)

    def test_smoke_explicit_replication_counts_are_not_overridden(self):
        args = self.cli.parse_args(["--smoke", "--n-replicates", "20", "--n-per-replicate", "200"])
        self.assertEqual((args.n_replicates, args.n_per_replicate), (20, 200))
        self.assertIsNone(self.cli.parse_args(["--smoke"]).n_per_replicate)


if __name__ == "__main__":
    unittest.main()
