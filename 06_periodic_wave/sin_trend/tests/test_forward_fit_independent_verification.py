#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: test forward fit independent verification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from sin_trend_forward_fit import (  # noqa: E402
    build_processed_design,
    fit_joint_sin_trend,
    search_k_forward_3d,
)
from sin_trend_measure import default_measurement_config, measure_interval  # noqa: E402
from sin_trend_synthetic import make_sinusoid_centerline, random_rotation  # noqa: E402


def _independent_reference_W(U, V, s, q, k, J):
    A = np.column_stack([np.ones(len(q)), q, q * q])
    cols = []
    for axis, wave in (
        (U, np.cos(k * s)),
        (U, np.sin(k * s)),
        (V, np.cos(k * s)),
        (V, np.sin(k * s)),
    ):
        F = axis * wave[:, None]
        F = F - A @ np.linalg.lstsq(A, F, rcond=None)[0]
        cols.append(np.r_[np.einsum("ij,ij->i", F, U)[J], np.einsum("ij,ij->i", F, V)[J]])
    return np.column_stack(cols)


def _trend_matrix(s):
    t = s - s.mean()
    n = len(s)
    Z = np.zeros((2 * n, 4))
    Z[:n, :2] = np.c_[np.ones(n), t]
    Z[n:, 2:] = np.c_[np.ones(n), t]
    return t, Z


class TestIndependentProcessedDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Analytic frame of B(q)=(30q, 0, 20(q-0.5)^2); nonuniform q.
        q = np.linspace(0, 1, 181) ** 1.13
        J = np.arange(9, len(q) - 11)
        der = np.c_[np.full(len(q), 30.0), np.zeros(len(q)), 40 * (q - 0.5)]
        T = der / np.linalg.norm(der, axis=1)[:, None]
        U = np.tile([0.0, 1.0, 0.0], (len(q), 1))
        V = np.cross(T, U)
        # baseline arclength via trapezoid of speed
        speed = np.hypot(30.0, 40.0 * (q - 0.5))
        s = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(q))])
        s = s - s[J[0]]
        cls.q, cls.J, cls.U, cls.V, cls.s = q, J, U, V, s
        cls.coef = np.array([0.8, -0.3, 0.4, 0.2, 0.02, 0.001, -0.03, -0.002])

    def test_W_coef_SSE_and_true_k_selection_for_cycle_grid(self):
        q, J, U, V, s = self.q, self.J, self.U, self.V, self.s
        t, Z = _trend_matrix(s[J])
        n = len(J)
        L = float(s[J[-1]] - s[J[0]])
        for cycles in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            with self.subTest(cycles=cycles):
                k = 2.0 * np.pi * cycles / L
                Wref = _independent_reference_W(U, V, s, q, k, J)
                W, info = build_processed_design(
                    U_full=U, V_full=V, s_phase_full=s, q_full=q, k=k, J=J
                )
                self.assertTrue(info["ok"])
                self.assertLess(float(np.max(np.abs(W - Wref))), 1e-11)
                y = np.c_[Wref, Z] @ self.coef
                fit = fit_joint_sin_trend(y[:n], y[n:], t, W)
                self.assertTrue(fit["ok"])
                self.assertLess(float(np.max(np.abs(fit["coef"] - self.coef))), 1e-8)
                self.assertLess(float(fit["SSE_full"]), 1e-20)
                ks = k * np.array([0.7, 0.85, 1.0, 1.15, 1.3])
                result = search_k_forward_3d(
                    u1=y[:n],
                    u2=y[n:],
                    s_phase_full=s,
                    s_phase_trim=s[J],
                    s_trend_trim=s[J],
                    q_full=q,
                    U_full=U,
                    V_full=V,
                    J=J,
                    ks=ks,
                    k_min=float(ks[0]),
                    k_max=float(ks[-1]),
                    dk=float(ks[1] - ks[0]),
                )
                self.assertEqual(int(result["selected_index"]), 2)


class TestInvarianceOfFullMeasure(unittest.TestCase):
    def test_rotation_reverse_voxel_scaling(self):
        syn = make_sinusoid_centerline(
            amplitude_mm=1.0,
            lambda_x_mm=20.0,
            n_cycles=2.5,
            phase=0.0,
            target_ds_mm=0.1,
            diameter_mm=1.4,
        )
        cfg = default_measurement_config()
        cfg["verbose"] = False
        cfg["enable_plots"] = False
        base = measure_interval(
            syn["P_vox"],
            frame_reference="baseline",
            cfg=cfg,
            radius_vox=syn["radius_vox"],
            compute_pca=False,
        )
        row0 = base["row"]
        P = np.asarray(syn["P_vox"], float)
        radius = np.asarray(syn["radius_vox"], float)
        vs = float(syn["voxel_size_mm"])
        R = random_rotation(seed=73)
        cases = [
            ("rotation_translation", P @ R.T + np.array([120.0, -87.0, 33.0]), radius, dict(cfg)),
            ("reversed_order", P[::-1], radius[::-1], dict(cfg)),
            ("voxel_size_1mm_same_physical", P * vs, radius * vs, dict(cfg, voxel_size=1.0)),
        ]
        for name, Ptest, rtest, cf in cases:
            with self.subTest(name=name):
                rr = measure_interval(
                    Ptest,
                    frame_reference="baseline",
                    cfg=cf,
                    radius_vox=rtest,
                    compute_pca=False,
                )["row"]
                self.assertAlmostEqual(float(rr["lambda_hat_mm"]), float(row0["lambda_hat_mm"]), places=8)
                self.assertAlmostEqual(float(rr["amp_comb_mm"]), float(row0["amp_comb_mm"]), places=8)
                self.assertAlmostEqual(float(rr["partial_R2"]), float(row0["partial_R2"]), places=8)
                self.assertEqual(int(rr["core_all"]), int(row0["core_all"]))


class TestPolylineLambdaS(unittest.TestCase):
    def test_finer_ds_converges_toward_continuous_reference(self):
        from sin_trend_synthetic import arc_length_per_cycle, make_sinusoid_centerline

        cont = arc_length_per_cycle(1.0, 20.0, 0.5)
        coarse = make_sinusoid_centerline(
            amplitude_mm=1.0,
            lambda_x_mm=20.0,
            n_cycles=3.0,
            phase=0.0,
            out_of_plane_amplitude_mm=0.5,
            target_ds_mm=0.5,
            diameter_mm=4.0,
        )
        fine = make_sinusoid_centerline(
            amplitude_mm=1.0,
            lambda_x_mm=20.0,
            n_cycles=3.0,
            phase=0.0,
            out_of_plane_amplitude_mm=0.5,
            target_ds_mm=0.1,
            diameter_mm=4.0,
        )
        self.assertEqual(coarse["lambda_s_polyline_status"], "ok")
        self.assertEqual(fine["lambda_s_polyline_status"], "ok")
        err_c = abs(coarse["lambda_s_polyline_mm"] - cont)
        err_f = abs(fine["lambda_s_polyline_mm"] - cont)
        self.assertLess(err_f, err_c)
        self.assertLess(err_f, 5e-4)


if __name__ == "__main__":
    unittest.main()
