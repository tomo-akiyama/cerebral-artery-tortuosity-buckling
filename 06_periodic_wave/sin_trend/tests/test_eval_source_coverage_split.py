#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: test eval source coverage split."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_classify import classify_row, surface_eligible_from_row  # noqa: E402
from sin_trend_io_utils import atomic_to_csv  # noqa: E402
from sin_trend_measure import measurement_row_schema  # noqa: E402
from sin_trend_surface import (  # noqa: E402
    EVAL_SOURCE_BRIDGED,
    EVAL_SOURCE_GOOD,
    EVAL_SOURCE_NONE,
    evaluate_displacement,
)


def _est(normals, mask, eval_src, *, cortex=None):
    mask = np.asarray(mask, bool)
    return {
        "normals_full": np.asarray(normals, float),
        "mask_eval": mask,
        "eval_normal_source": np.asarray(eval_src, int),
        "cortex_coverage": float(np.mean(mask)) if cortex is None else float(cortex),
        "constraint_coverage": float(np.mean(mask)),
    }


class TestEvalSourceCoverageSplit(unittest.TestCase):
    def test_artificial_good_bridged_none_magnitudes_1_2_3(self):
                                                                   
        d = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], float)
        n = np.tile([0.0, 0.0, 1.0], (3, 1))
        mask = np.array([True, True, False], bool)
        src = np.array([EVAL_SOURCE_GOOD, EVAL_SOURCE_BRIDGED, EVAL_SOURCE_NONE], int)
        ev = evaluate_displacement(d, _est(n, mask, src))

        self.assertEqual(ev["status"], "ok")
        self.assertEqual(ev["n_points_total"], 3)
        self.assertEqual(ev["n_eval"], 2)
        self.assertEqual(ev["n_eval_good"], 1)
        self.assertEqual(ev["n_eval_bridged"], 1)
        self.assertEqual(ev["n_eval_good"] + ev["n_eval_bridged"], ev["n_eval"])

        self.assertAlmostEqual(ev["good_coverage"], 1.0 / 3.0)
        self.assertAlmostEqual(ev["bridged_coverage"], 1.0 / 3.0)
        self.assertAlmostEqual(ev["good_coverage"] + ev["bridged_coverage"], ev["constraint_coverage"])

        self.assertAlmostEqual(ev["squared_displacement_coverage_good"], 1.0 / 14.0)
        self.assertAlmostEqual(ev["squared_displacement_coverage_bridged"], 4.0 / 14.0)
        self.assertAlmostEqual(
            ev["squared_displacement_coverage_good"] + ev["squared_displacement_coverage_bridged"], ev["squared_displacement_coverage"]
        )
        self.assertAlmostEqual(ev["squared_displacement_coverage"], 5.0 / 14.0)
                          
        self.assertAlmostEqual(ev["R_out"], 0.0)

    def test_identities_on_mixed_mask(self):
        rng = np.random.default_rng(0)
        n = 40
        d = rng.normal(size=(n, 3))
        normals = rng.normal(size=(n, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        src = np.full(n, EVAL_SOURCE_NONE, int)
        src[:10] = EVAL_SOURCE_GOOD
        src[10:18] = EVAL_SOURCE_BRIDGED
        mask = (src == EVAL_SOURCE_GOOD) | (src == EVAL_SOURCE_BRIDGED)
                                               
        mask[0] = False
        ev = evaluate_displacement(d, _est(normals, mask, src))
        self.assertEqual(ev["n_eval_good"] + ev["n_eval_bridged"], ev["n_eval"])
        self.assertAlmostEqual(ev["good_coverage"] + ev["bridged_coverage"], ev["constraint_coverage"])
        self.assertAlmostEqual(
            ev["squared_displacement_coverage_good"] + ev["squared_displacement_coverage_bridged"], ev["squared_displacement_coverage"]
        )

    def test_zero_total_energy_leaves_energy_splits_nan(self):
        n = 5
        d = np.zeros((n, 3))
        mask = np.ones(n, bool)
        src = np.full(n, EVAL_SOURCE_GOOD, int)
        ev = evaluate_displacement(d, _est(np.tile([0, 0, 1.0], (n, 1)), mask, src))
        self.assertEqual(ev["status"], "zero_displacement_energy")
        self.assertTrue(np.isnan(ev["R_out"]))
        self.assertTrue(np.isnan(ev["squared_displacement_coverage"]))
        self.assertTrue(np.isnan(ev["squared_displacement_coverage_good"]))
        self.assertTrue(np.isnan(ev["squared_displacement_coverage_bridged"]))
                             
        self.assertEqual(ev["n_eval_good"], n)
        self.assertAlmostEqual(ev["good_coverage"], 1.0)
        self.assertAlmostEqual(ev["bridged_coverage"], 0.0)

    def test_empty_points_coverages_nan(self):
        d = np.zeros((0, 3))
        ev = evaluate_displacement(
            d,
            {
                "normals_full": np.zeros((0, 3)),
                "mask_eval": np.zeros(0, bool),
                "eval_normal_source": np.zeros(0, int),
                "cortex_coverage": float("nan"),
                "constraint_coverage": float("nan"),
            },
        )
        self.assertTrue(np.isnan(ev["good_coverage"]))
        self.assertTrue(np.isnan(ev["bridged_coverage"]))
        self.assertTrue(np.isnan(ev["squared_displacement_coverage_good"]))
        self.assertTrue(np.isnan(ev["squared_displacement_coverage_bridged"]))

    def test_missing_eval_source_leaves_splits_nan_keeps_r_out(self):
        n = 10
        d = np.zeros((n, 3))
        d[:, 1] = 1.0
        mask = np.ones(n, bool)
        est = {
            "normals_full": np.tile([0.0, 0.0, 1.0], (n, 1)),
            "mask_eval": mask,
            "cortex_coverage": 1.0,
            "constraint_coverage": 1.0,
        }
        ev = evaluate_displacement(d, est)
        self.assertEqual(ev["status"], "ok")
        self.assertAlmostEqual(ev["R_out"], 0.0)
        self.assertAlmostEqual(ev["squared_displacement_coverage"], 1.0)
        self.assertTrue(np.isnan(ev["n_eval_good"]))
        self.assertTrue(np.isnan(ev["squared_displacement_coverage_good"]))

    def test_display_normal_source_not_used_for_split(self):
                                                                          
        d = np.array([[1.0, 0, 0], [1.0, 0, 0]], float)
        n = np.tile([0.0, 0.0, 1.0], (2, 1))
        mask = np.array([True, True], bool)
        src = np.array([EVAL_SOURCE_BRIDGED, EVAL_SOURCE_BRIDGED], int)
        est = _est(n, mask, src)
        est["normal_source_full"] = np.array([1, 1], int)                  
        ev = evaluate_displacement(d, est)
        self.assertEqual(ev["n_eval_good"], 0)
        self.assertEqual(ev["n_eval_bridged"], 2)
        self.assertAlmostEqual(ev["squared_displacement_coverage_bridged"], 1.0)
        self.assertAlmostEqual(ev["squared_displacement_coverage_good"], 0.0)

    def test_schema_and_csv_roundtrip(self):
        schema = measurement_row_schema()
        for k in (
            "n_eval_good",
            "n_eval_bridged",
            "good_coverage",
            "bridged_coverage",
            "squared_displacement_coverage_good",
            "squared_displacement_coverage_bridged",
        ):
            self.assertIn(k, schema)

        d = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]], float)
        n = np.tile([0.0, 0.0, 1.0], (3, 1))
        mask = np.array([True, True, False], bool)
        src = np.array([EVAL_SOURCE_GOOD, EVAL_SOURCE_BRIDGED, EVAL_SOURCE_NONE], int)
        ev = evaluate_displacement(d, _est(n, mask, src))

        row = measurement_row_schema()
        for k in (
            "n_eval",
            "n_eval_good",
            "n_eval_bridged",
            "good_coverage",
            "bridged_coverage",
            "squared_displacement_coverage",
            "squared_displacement_coverage_good",
            "squared_displacement_coverage_bridged",
            "constraint_coverage",
            "R_out",
        ):
            dst = "R_out_surface" if k == "R_out" else k
            row[dst] = ev.get("R_out" if k == "R_out" else k, np.nan)
        row["R_out_latent"] = row["R_out_surface"]
        row["squared_displacement_coverage_latent"] = row["squared_displacement_coverage"]
        row["surface_evaluation_status"] = "ok"

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "row.csv"
            atomic_to_csv(pd.DataFrame([row]), path, index=False)
            loaded = pd.read_csv(path).iloc[0]
            self.assertEqual(int(loaded["n_eval_good"]), 1)
            self.assertEqual(int(loaded["n_eval_bridged"]), 1)
            self.assertAlmostEqual(float(loaded["good_coverage"]), 1.0 / 3.0)
            self.assertAlmostEqual(float(loaded["bridged_coverage"]), 1.0 / 3.0)
            self.assertAlmostEqual(float(loaded["squared_displacement_coverage_good"]), 1.0 / 14.0)
            self.assertAlmostEqual(float(loaded["squared_displacement_coverage_bridged"]), 4.0 / 14.0)

    def test_existing_gates_unchanged_by_new_columns(self):
                                               
        base = {
            "fit_status": "ok",
            "R2": 0.9,
            "partial_R2": 0.6,
            "lambda_hat_mm": 20.0,
            "k_hat_mm": 2.0 * np.pi / 20.0,
            "amp_comb_mm": 0.8,
            "coef_uU_cos_mm": 0.3,
            "coef_uU_sin_mm": 0.2,
            "coef_uV_cos_mm": 0.1,
            "coef_uV_sin_mm": 0.05,
            "coverage_hat": 2.0,
            "interval_eligible": 1,
            "boundary_exact_high": 0,
            "boundary_exact_low": 0,
            "surface_evaluation_status": "ok",
            "R_out_latent": 0.1,
            "R_out_surface": 0.1,
            "R_tan_latent": 0.9,
            "R_tan_surface": 0.9,
            "constraint_coverage": 0.8,
            "squared_displacement_coverage": 0.8,
            "squared_displacement_coverage_latent": 0.8,
            "constraint_coverage_thresh": 0.50,
            "squared_displacement_coverage_thresh": 0.50,
            "R_out_thresh": 0.25,
            "R_tan_thresh": 0.75,
            "good_coverage": 0.01,
            "bridged_coverage": 0.01,
            "squared_displacement_coverage_good": 0.01,
            "squared_displacement_coverage_bridged": 0.01,
            "n_eval_good": 1,
            "n_eval_bridged": 1,
        }
        surf = surface_eligible_from_row(base)
        self.assertTrue(surf["surface_coverage_ok"])
        out = classify_row(base)
        self.assertEqual(out["surface_constraint"], "Predominantly tangential")

                                   
        low_split = dict(base, good_coverage=0.0, bridged_coverage=0.8, squared_displacement_coverage_good=0.0)
        self.assertTrue(surface_eligible_from_row(low_split)["surface_coverage_ok"])

    def test_r_out_unchanged_vs_without_source_metadata(self):
        rng = np.random.default_rng(1)
        n = 25
        d = rng.normal(size=(n, 3))
        normals = rng.normal(size=(n, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        mask = rng.random(n) > 0.4
        src = np.where(mask, EVAL_SOURCE_GOOD, EVAL_SOURCE_NONE).astype(int)
        src[np.flatnonzero(mask)[::2]] = EVAL_SOURCE_BRIDGED
        est_full = _est(normals, mask, src)
        est_bare = {
            "normals_full": normals,
            "mask_eval": mask,
            "cortex_coverage": float(np.mean(mask)),
            "constraint_coverage": float(np.mean(mask)),
        }
        ev_full = evaluate_displacement(d, est_full)
        ev_bare = evaluate_displacement(d, est_bare)
        self.assertAlmostEqual(ev_full["R_out"], ev_bare["R_out"])
        self.assertAlmostEqual(ev_full["squared_displacement_coverage"], ev_bare["squared_displacement_coverage"])
        self.assertAlmostEqual(ev_full["constraint_coverage"], ev_bare["constraint_coverage"])
        self.assertEqual(ev_full["status"], ev_bare["status"])


if __name__ == "__main__":
    unittest.main()
