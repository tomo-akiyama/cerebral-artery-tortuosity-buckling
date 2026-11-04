#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: test surface eligibility no cortex gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_classify import classify_row, surface_eligible_from_row  # noqa: E402
from sin_trend_paths import CLASSIFICATION_REVISION  # noqa: E402


def _core_surface(**kw):
    row = {
        "fit_status": "ok",
        "R2": 0.85,
        "partial_R2": 0.60,
        "lambda_hat_mm": 20.0,
        "k_hat_mm": 2.0 * np.pi / 20.0,
        "amp_comb_mm": 0.8,
        "SNR": 2.0,
        "hr_amp": 0.1,
        "coef_uU_cos_mm": 0.3,
        "coef_uU_sin_mm": 0.2,
        "coef_uV_cos_mm": 0.1,
        "coef_uV_sin_mm": 0.05,
        "diameter_ref_mm": 3.0,
        "L_full_mm": 40.0,
        "L_trimmed_mm": 36.0,
        "coverage_hat": 36.0 / 20.0,
        "interval_eligible": 1,
        "boundary_exact_high": 0,
        "boundary_exact_low": 0,
        "boundary_hit_high_2bins": 0,
        "boundary_hit_low_2bins": 0,
        "selected_index": 5,
        "n_grid": 40,
        "surface_evaluation_status": "ok",
        "R_out_surface": 0.10,
        "R_out_latent": 0.10,
        "R_tan_surface": 0.90,
        "R_tan_latent": 0.90,
        "cortex_coverage": 1.0,
        "constraint_coverage": 0.80,
        "squared_displacement_coverage": 0.80,
        "squared_displacement_coverage_latent": 0.80,
        "cortex_coverage_thresh": 0.30,
        "constraint_coverage_thresh": 0.50,
        "squared_displacement_coverage_thresh": 0.50,
        "R_out_thresh": 0.25,
        "R_tan_thresh": 0.75,
    }
    row.update(kw)
    return row


class TestSurfaceEligibilityNoCortexGate(unittest.TestCase):
    def test_classification_revision_is_c4(self):
        self.assertEqual(CLASSIFICATION_REVISION, "v2_partialR2_050_c4")
        out = classify_row(_core_surface())
        self.assertEqual(out["classification_revision"], "v2_partialR2_050_c4")

    def test_low_cortex_coverage_alone_does_not_exclude(self):
        row = _core_surface(cortex_coverage=0.10, cortex_coverage_thresh=0.30)
        surf = surface_eligible_from_row(row)
        self.assertTrue(surf["surface_coverage_ok"])
        out = classify_row(row)
        self.assertEqual(out["core_all"], 1)
        self.assertEqual(out["surface_constraint"], "Predominantly tangential")
        self.assertAlmostEqual(float(out["cortex_coverage"]), 0.10)

    def test_missing_cortex_coverage_alone_does_not_block(self):
        row = _core_surface(cortex_coverage=np.nan)
        surf = surface_eligible_from_row(row)
        self.assertTrue(surf["surface_coverage_ok"])
        out = classify_row(row)
        self.assertEqual(out["surface_constraint"], "Predominantly tangential")

    def test_constraint_below_50_excluded(self):
        row = _core_surface(constraint_coverage=0.49)
        surf = surface_eligible_from_row(row)
        self.assertFalse(surf["surface_coverage_ok"])
        out = classify_row(row)
        self.assertEqual(out["core_all"], 1)
        self.assertIsNone(out["surface_constraint"])

    def test_energy_below_50_excluded(self):
        row = _core_surface(squared_displacement_coverage=0.49, squared_displacement_coverage_latent=0.49)
        surf = surface_eligible_from_row(row)
        self.assertFalse(surf["surface_coverage_ok"])
        out = classify_row(row)
        self.assertIsNone(out["surface_constraint"])

    def test_boundary_coverages_exactly_50_pass(self):
        row = _core_surface(constraint_coverage=0.50, squared_displacement_coverage=0.50, squared_displacement_coverage_latent=0.50)
        surf = surface_eligible_from_row(row)
        self.assertTrue(surf["surface_coverage_ok"])
        out = classify_row(row)
        self.assertEqual(out["surface_constraint"], "Predominantly tangential")

    def test_r_out_exactly_025_is_substantial_normal(self):
        # R_out=0.25 ⇔ R_tan=0.75 → Substantial（≤0.75）
        row = _core_surface(
            R_out_surface=0.25, R_out_latent=0.25, R_tan_surface=0.75, R_tan_latent=0.75
        )
        surf = surface_eligible_from_row(row)
        self.assertTrue(surf["surface_coverage_ok"])
        self.assertFalse(surf["surface_eligible"])
        self.assertTrue(surf["surface_out_of_plane_possible"])
        out = classify_row(row)
        self.assertEqual(out["surface_constraint"], "Substantial normal component")

    def test_r_tan_just_above_075_is_predominantly_tangential(self):
        row = _core_surface(
            R_out_surface=0.249999,
            R_out_latent=0.249999,
            R_tan_surface=0.750001,
            R_tan_latent=0.750001,
        )
        out = classify_row(row)
        self.assertEqual(out["surface_constraint"], "Predominantly tangential")

    def test_non_core_gets_no_plane_class(self):
        row = _core_surface(partial_R2=0.10)
        out = classify_row(row)
        self.assertEqual(out["core_all"], 0)
        self.assertIsNone(out["surface_constraint"])
        self.assertIsNone(out["plane_class"])

    def test_cortex_coverage_still_recorded_on_row(self):
        out = classify_row(_core_surface(cortex_coverage=0.42))
        self.assertIn("cortex_coverage", out)
        self.assertAlmostEqual(float(out["cortex_coverage"]), 0.42)

    def test_unused_cortex_thresh_does_not_gate(self):
                                           
        row = _core_surface(cortex_coverage=0.0, cortex_coverage_thresh=0.30)
        self.assertTrue(surface_eligible_from_row(row)["surface_coverage_ok"])


if __name__ == "__main__":
    unittest.main()
