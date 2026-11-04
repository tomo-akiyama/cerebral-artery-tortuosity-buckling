#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: test safety and io contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))


def test_source_code_hashes_local_only():
    from sin_trend_paths import source_code_hashes, OUTPUT_TAG

    h = source_code_hashes()
    assert "sin_trend_paths.py" in h
    assert h["sin_trend_paths.py"] != "missing"
    assert h.get("31_run_sin_trend_fitting.py") != "missing"
    assert OUTPUT_TAG == "sin_trend"
    assert isinstance(h, dict)


def test_pure_trend_partial_r2_undefined():
    from sin_trend_forward_fit import search_k_forward_3d
    from sin_trend_classify import core_periodic_from_row

    n = 100
    s = np.linspace(0, 60.0, n)
    t = s - s.mean()
    u1 = 0.5 + 0.2 * t
    u2 = 0.3 - 0.1 * t
    U = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    V = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    q = (s - s[0]) / (s[-1] - s[0])
    J = np.arange(n)
    L = float(s[-1] - s[0])
    D = 2.0
    k_min = np.pi / L
    k_max = np.pi / D
    dk = np.pi / (8 * L)
    ks = k_min + dk * np.arange(int(np.floor((k_max - k_min) / dk)) + 1)
    if ks[-1] < k_max - 1e-12:
        ks = np.append(ks, k_max)
    out = search_k_forward_3d(
        u1=u1, u2=u2,
        s_phase_full=s, s_phase_trim=s, s_trend_trim=s,
        q_full=q, U_full=U, V_full=V, J=J,
        ks=ks, k_min=float(k_min), k_max=float(k_max), dk=float(dk),
    )
    assert out["status"] == "ok"
    assert not np.isfinite(out["partial_R2"]), f"expected NaN partial_R2, got {out['partial_R2']}"
    row = {
        "fit_status": "ok",
        "interval_eligible": 1,
        "R2": out["R2"],
        "partial_R2": out["partial_R2"],
        "lambda_hat_mm": out["lambda_hat"],
        "k_hat_mm": out["k_hat"],
        "amp_comb_mm": out["A_comb"],
        "coef_uU_cos_mm": out["uU_cos"],
        "coef_uU_sin_mm": out["uU_sin"],
        "coef_uV_cos_mm": out["uV_cos"],
        "coef_uV_sin_mm": out["uV_sin"],
        "coverage_hat": L / out["lambda_hat"] if out["lambda_hat"] > 0 else np.nan,
        "boundary_exact_low": 0,
        "boundary_exact_high": 0,
        "L_trimmed_mm": L,
    }
    assert not core_periodic_from_row(row)


def test_boundary_nan_not_interior():
    from sin_trend_classify import boundary_excluded, core_periodic_from_row, parse_tri_bool

    assert parse_tri_bool(np.nan) is None
    assert parse_tri_bool(pd.NA) is None
    assert parse_tri_bool(2) is None
    assert parse_tri_bool("maybe") is None
    assert parse_tri_bool(True) is True
    assert parse_tri_bool("False") is False

    base = {
        "fit_status": "ok",
        "interval_eligible": 1,
        "R2": 0.9,
        "partial_R2": 0.7,
        "lambda_hat_mm": 20.0,
        "k_hat_mm": 0.3,
        "amp_comb_mm": 0.5,
        "coef_uU_cos_mm": 0.1,
        "coef_uU_sin_mm": 0.1,
        "coef_uV_cos_mm": 0.1,
        "coef_uV_sin_mm": 0.1,
        "coverage_hat": 2.0,
        "boundary_exact_low": np.nan,
        "boundary_exact_high": np.nan,
        "L_trimmed_mm": 40.0,
    }
    assert boundary_excluded(base)
    assert not core_periodic_from_row(base)


def test_version_rejects_foreign_method():
    from sin_trend_paths import version_revision_consistency

    df = pd.DataFrame(
        {
            "directory_id": ["A"],
            "segment_id": [0],
            "segment_index": [0],
            "output_tag": ["sin_trend"],
            "measurement_version": ["OLD_METHOD"],
            "measurement_revision": ["r0"],
            "frame_reference": ["baseline"],
            "fit_method": ["ordinary_unprocessed_sin"],
            "coordinate_unit_input": ["voxel"],
            "voxel_size_mm": [0.5],
        }
    )
    cons = version_revision_consistency(df)
    assert cons["ok"] is False


def test_ixi_id_float_int_compatible():
    import importlib.util

    path = _HERE / "33_prepare_core_inplane_dataset.py"
    spec = importlib.util.spec_from_file_location("prep33", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod._normalize_id_value(901.0) == mod._normalize_id_value(901)
    assert mod._values_compatible(901.0, 901, col="IXI_ID")


def test_mca_confirmed_required():
    from sin_trend_dataset import filter_mca_m2m4_core_periodic, mca_label_confirmed

    assert mca_label_confirmed("ok")
    assert mca_label_confirmed("inherited_shape_verified")
    assert not mca_label_confirmed("inherited_id_only_shape_unverified")
    df = pd.DataFrame(
        {
            "core_periodic": [1, 1],
            "mca_segment": ["M2-3", "M2-3"],
            "mca_label_conflict": [False, False],
            "label_status": ["ok", "inherited_id_only_shape_unverified"],
        }
    )
    out = filter_mca_m2m4_core_periodic(df)
    assert len(out) == 1
    assert out.iloc[0]["label_status"] == "ok"


def test_search_returns_ks():
    from sin_trend_forward_fit import search_k_forward_3d

    n = 60
    s = np.linspace(0, 30.0, n)
    k_true = 2 * np.pi / 15.0
    u1 = 0.5 * np.cos(k_true * s)
    u2 = 0.2 * np.sin(k_true * s)
    U = np.tile([1.0, 0.0, 0.0], (n, 1))
    V = np.tile([0.0, 1.0, 0.0], (n, 1))
    q = (s - s[0]) / (s[-1] - s[0])
    ks = np.array([k_true * 0.8, k_true, k_true * 1.2])
    out = search_k_forward_3d(
        u1=u1, u2=u2,
        s_phase_full=s, s_phase_trim=s, s_trend_trim=s,
        q_full=q, U_full=U, V_full=V, J=np.arange(n),
        ks=ks, k_min=float(ks[0]), k_max=float(ks[-1]), dk=float(ks[1] - ks[0]),
    )
    assert "ks" in out and len(out["ks"]) == 3


def test_config_gates_match_classifier():
    from sin_trend_measure import default_measurement_config
    from sin_trend_paths import PARTIAL_R2_GATE_ENABLED, BOUNDARY_GATE_ENABLED

    cfg = default_measurement_config()
    assert bool(cfg["partial_r2_gate_enabled"]) == bool(PARTIAL_R2_GATE_ENABLED)
    assert bool(cfg["boundary_gate_enabled"]) == bool(BOUNDARY_GATE_ENABLED)


def test_base_plane_defaults():
    from sin_trend_synthetic import (
        BASE_PLANE_CONDITIONS,
        VALIDATION_DIAMETERS_MM,
        expand_grid_conditions,
        stress_validation_conditions,
    )

    assert BASE_PLANE_CONDITIONS["lambda_x_mm"] == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    assert BASE_PLANE_CONDITIONS["target_ds_mm"] == [0.10]
    assert BASE_PLANE_CONDITIONS["n_cycles_target_after_trim"] == [3.0]
    assert "n_cycles" not in BASE_PLANE_CONDITIONS
    assert VALIDATION_DIAMETERS_MM == [1.4]
    assert list(BASE_PLANE_CONDITIONS["diameter_mm"]) == list(VALIDATION_DIAMETERS_MM)
    n_per_d = (
        len(BASE_PLANE_CONDITIONS["A_mm"])
        * len(BASE_PLANE_CONDITIONS["lambda_x_mm"])
        * len(BASE_PLANE_CONDITIONS["phase"])
    )
    assert n_per_d == 72
    plane = expand_grid_conditions(BASE_PLANE_CONDITIONS)
    assert len(plane) == 72 * len(VALIDATION_DIAMETERS_MM)
    assert {c["diameter_mm"] for c in plane} == set(VALIDATION_DIAMETERS_MM)
    assert all(c["n_cycles_target_after_trim"] == 3.0 for c in plane)
    assert all(c["n_cycles"] > c["n_cycles_target_after_trim"] for c in plane)
    stress = stress_validation_conditions()
    assert len(stress) == (54 + 24 + 6) * len(VALIDATION_DIAMETERS_MM)
    curved = [c for c in stress if c["variant"] == "curved_baseline_sinusoid"]
    assert all(c["n_cycles_target_after_trim"] == 3.0 for c in curved)
    assert all(c["n_cycles"] > 3.0 for c in curved)
    assert np.isclose(BASE_PLANE_CONDITIONS["phase"][1], np.pi / 4)


def test_post_trim_remaining_cycles_match_target_via_xi():
    from sin_trend_synthetic import (
        expand_grid_conditions,
        make_sinusoid_centerline,
        remaining_cycles_true_from_xi,
        SHORT_COVERAGE_STRESS,
        BASE_PLANE_CONDITIONS,
    )

                                     
    specs = [
        dict(expand_grid_conditions(BASE_PLANE_CONDITIONS)[0]),
        dict(
            next(
                c
                for c in expand_grid_conditions(SHORT_COVERAGE_STRESS)
                if c["n_cycles_target_after_trim"] == 1.5 and c["lambda_x_mm"] == 10.0
            )
        ),
        dict(
            next(
                c
                for c in expand_grid_conditions(SHORT_COVERAGE_STRESS)
                if c["n_cycles_target_after_trim"] == 2.0 and c["lambda_x_mm"] == 40.0
            )
        ),
    ]
    for cond in specs:
        syn = make_sinusoid_centerline(
            amplitude_mm=cond["amplitude_mm"],
            lambda_x_mm=cond["lambda_x_mm"],
            n_cycles=cond["n_cycles"],
            phase=cond["phase"],
            target_ds_mm=cond["target_ds_mm"],
            diameter_mm=cond["diameter_mm"],
            background=cond.get("background"),
        )
        rem = remaining_cycles_true_from_xi(
            syn["xi_mm"],
            syn["lambda_baseline_true_mm"],
            removal_points=10,
        )
        assert abs(rem - float(cond["n_cycles_target_after_trim"])) < 1e-8, (cond, rem)


def test_post_trim_target_unmet_raises_for_coarse_ds():
    from sin_trend_synthetic import n_cycles_input_for_target_remaining_after_trim

    try:
        n_cycles_input_for_target_remaining_after_trim(
            target_remaining=3.0,
            amplitude_mm=0.25,
            lambda_x_mm=10.0,
            phase=0.0,
            target_ds_mm=0.2,
            removal_points=10,
            diameter_mm=1.4,
        )
    except ValueError as exc:
        assert "post-trim remaining cycles target not met" in str(exc)
    else:
        raise AssertionError("expected ValueError for unmet post-trim target at ds=0.2")


def test_stress_only_conditions_include_curved_and_short():
    import importlib.util

    from sin_trend_synthetic import stress_validation_conditions

    path = _HERE / "36_measurement_validation.py"
    spec = importlib.util.spec_from_file_location("val36_stress", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    args = mod.parse_args(["--stress-only"])
    conds = mod._conditions(args)
    variants = {c["variant"] for c in conds}
    assert variants == {"curved_baseline_sinusoid", "short_coverage", "curved_short_coverage"}
    assert len(conds) == len(stress_validation_conditions())
    assert any(c.get("background") for c in conds if c["variant"] == "curved_baseline_sinusoid")
    curved = [c for c in conds if c["variant"] == "curved_baseline_sinusoid"]
    assert {c["n_cycles_target_after_trim"] for c in curved} == {3.0}
    assert all(c["n_cycles"] > c["n_cycles_target_after_trim"] for c in curved)
    short = [c for c in conds if c["variant"] == "short_coverage"]
    assert {c["n_cycles_target_after_trim"] for c in short} == {1.5, 2.0}
    assert all(c["n_cycles"] > c["n_cycles_target_after_trim"] for c in short)
                
    assert "pure_sinusoid" not in variants


def test_target_ds_mm_cli_removed():
    import importlib.util

    path = _HERE / "36_measurement_validation.py"
    spec = importlib.util.spec_from_file_location("val36_ds", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    try:
        mod.parse_args(["--stress", "--target-ds-mm", "0.2"])
    except SystemExit as exc:
        assert int(exc.code) != 0
    else:
        raise AssertionError("expected --target-ds-mm to be rejected")
    conds = mod._conditions(mod.parse_args(["--stress"]))
    assert {float(c["target_ds_mm"]) for c in conds} == {0.10}


def test_metadata_diameters_from_actual_conditions():
    import importlib.util

    path = _HERE / "36_measurement_validation.py"
    spec = importlib.util.spec_from_file_location("val36_meta_D", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    def _diameters(args_list):
        conds = mod._conditions(mod.parse_args(args_list))
        return sorted(
            {
                float(c["diameter_mm"])
                for c in conds
                if c.get("diameter_mm") is not None and np.isfinite(float(c["diameter_mm"]))
            }
        )

    assert _diameters(["--stress"]) == [1.4]
    assert _diameters(["--stress", "--diameter-mm", "2.5"]) == [2.5]
    assert len(mod._conditions(mod.parse_args(["--stress"]))) == 156


def test_summarize_reports_core_abs_errors_separately():
    import importlib.util

    import pandas as pd

    path = _HERE / "36_measurement_validation.py"
    spec = importlib.util.spec_from_file_location("val36_sum", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    df = pd.DataFrame(
        [
            {
                "variant": "v",
                "fit_status": "ok",
                "core_all": 1,
                "A_error_pct": 1.0,
                "lambda_error_vs_baseline_pct": -2.0,
                "R2": 0.99,
                "coverage_hat": 3.0,
                "n_cycles_remaining_true": 3.0,
            },
            {
                "variant": "v",
                "fit_status": "ok",
                "core_all": 0,
                "A_error_pct": 1000.0,
                "lambda_error_vs_baseline_pct": 300.0,
                "R2": 0.5,
                "coverage_hat": 0.5,
                "n_cycles_remaining_true": 3.0,
            },
        ]
    )
    summ = mod.summarize(df)
    assert int(summ.loc[0, "n_conditions"]) == 2
    assert int(summ.loc[0, "n_fit_ok"]) == 2
    assert int(summ.loc[0, "n_core_all"]) == 1
    assert int(summ.loc[0, "n_core_reject"]) == 1
    assert abs(float(summ.loc[0, "A_error_pct_abs_median_core"]) - 1.0) < 1e-12
    assert abs(float(summ.loc[0, "A_error_pct_abs_max_core"]) - 1.0) < 1e-12
    assert abs(float(summ.loc[0, "lambda_error_vs_baseline_pct_abs_median_core"]) - 2.0) < 1e-12
    assert abs(float(summ.loc[0, "lambda_error_vs_baseline_pct_abs_max_core"]) - 2.0) < 1e-12


def test_known_baseline_arclength_phase_is_uniform_per_cycle():
    from sin_trend_synthetic import make_sinusoid_centerline, quadratic_baseline_arclength_mm

    syn = make_sinusoid_centerline(
        amplitude_mm=0.5,
        lambda_x_mm=60.0,
        n_cycles=3.0,
        phase=0.0,
        target_ds_mm=0.1,
        diameter_mm=2.0,
        background={"type": "quadratic", "curvature_mm": 0.005},
        voxel_size_mm=0.5,
    )
    assert syn["lambda_baseline_true_basis"] == "known_baseline_arclength_xi"
    assert abs(float(syn["lambda_baseline_true_mm"]) - 60.0) < 1e-12
    assert abs(float(syn["L0_known_baseline_mm"]) - 180.0) < 1e-6
                              
    assert float(syn["x_max_mm"]) < 180.0
    xi = syn["xi_mm"]
                            
    assert abs(float(xi[0])) < 0.2
    assert abs(float(xi[-1]) - 180.0) < 0.5
                                                 
    x_max = float(syn["x_max_mm"])
    c = 0.005
    per = []
    for i in range(3):
                                           
        per.append(60.0)
    assert per == [60.0, 60.0, 60.0]
                                            
    xi_x0 = quadratic_baseline_arclength_mm(c, x_max / 3.0)
    assert abs(xi_x0 - 60.0) > 1.0


def test_flow_separates_cumulative():
    from sin_trend_selection_diagnostics import flow_counts

    df = pd.DataFrame(
        {
            "directory_id": ["A", "A", "B"],
            "interval_eligible": [True, True, True],
            "fit_valid": [True, True, False],
            "core_periodic": [True, False, False],
            "surface_evaluable": [True, False, False],
            "core_inplane": [True, False, False],
            "fit_status": ["ok", "ok", "exception"],
        }
    )
    f = flow_counts(df)
    assert f["n_fit_valid"] == 2
    assert f["cumulative_fit_valid"] == 2
    assert f["cumulative_core_periodic"] == 1
    assert f["n_fit_exception_or_ambiguous"] == 1


def test_grid_png_positive_k_limits():
    import tempfile
    from sin_trend_visualize import save_grid_search_png

    ks = np.linspace(0.0269128126, 0.7853981634, 40)
    cand = pd.DataFrame(
        {
            "k_rad_per_voxel": ks,
            "R2": 0.5 + 0.4 * np.exp(-((ks - 0.3) ** 2) / 0.01),
            "partial_R2": 0.4 + 0.3 * np.exp(-((ks - 0.3) ** 2) / 0.01),
        }
    )
    search = {"ks": ks, "k_min": float(ks[0]), "k_max": float(ks[-1])}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "grid.png"
        st = save_grid_search_png(
            out_path=out,
            search=search,
            candidates=cand,
            L_trimmed_vox=60.0,
            k_hat_vox=0.3,
            voxel_size_mm=0.5,
        )
        assert st == "ok", st
        assert out.is_file() and out.stat().st_size > 0


def test_selected_rank_copied_to_measurement_row():
    from sin_trend_measure import measure_interval, default_measurement_config, candidate_table

    n = 120
    x = np.linspace(0, 40.0, n)
    k = 2 * np.pi / 20.0
    P = np.column_stack([x, 0.4 * np.sin(k * x), 0.2 * np.cos(k * x)])
    cfg = default_measurement_config()
    cfg["verbose"] = False
    res = measure_interval(P, frame_reference="baseline", cfg=cfg, radius_vox=np.full(n, 1.0))
    row = res["row"]
    assert np.isfinite(row["rank_X"]) and row["rank_X"] > 0
    assert np.isfinite(row["rank_Z"]) and row["rank_Z"] > 0
    assert np.isfinite(row["cond_X"])
    assert np.isfinite(row["rcond"])
    ct = candidate_table(res["search"], float(cfg["voxel_size"]), {"directory_id": "T", "segment_id": 0, "segment_index": 0}, "baseline")
    sel = ct[ct["is_selected"] == 1]
    assert len(sel) == 1
    assert abs(float(sel.iloc[0]["rank_X"]) - float(row["rank_X"])) < 1e-9
    assert abs(float(sel.iloc[0]["cond_X"]) - float(row["cond_X"])) < 1e-6


def test_gui_cli_include_surface_unevaluable():
    import importlib.util

    path = _HERE / "34_visual_inspection_mca_segment_periodicity.py"
    spec = importlib.util.spec_from_file_location("gui34", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    args = mod.parse_args(["--run-id", "x", "--include-surface-unevaluable", "--dry-run"])
    assert args.include_surface_unevaluable is True
    df = pd.DataFrame(
        {
            "directory_id": ["A", "B"],
            "segment_id": [1, 2],
            "segment_index": [0, 1],
            "core_periodic": [1, 1],
            "periodicity": ["Core", "Core"],
            "surface_constraint": ["NotEvaluable", "Predominantly tangential"],
            "mca_segment": [None, None],
            "label_status": ["unset", "unset"],
            "mca_label_conflict": [False, False],
        }
    )
    t_req = mod.list_targets(df, "longwave_r2", mode="add", require_surface=True)
    t_all = mod.list_targets(df, "longwave_r2", mode="add", require_surface=False)
    assert len(t_req) == 1
    assert len(t_all) == 2


def test_unit_mismatch_rejected():
    from sin_trend_paths import version_revision_consistency, EXPECTED_MEASUREMENT_VERSION, EXPECTED_MEASUREMENT_REVISION, EXPECTED_FIT_METHOD, EXPECTED_OUTPUT_TAG

    df = pd.DataFrame(
        {
            "directory_id": ["A"],
            "segment_id": [0],
            "segment_index": [0],
            "output_tag": [EXPECTED_OUTPUT_TAG],
            "measurement_version": [EXPECTED_MEASUREMENT_VERSION],
            "measurement_revision": [EXPECTED_MEASUREMENT_REVISION],
            "frame_reference": ["baseline"],
            "fit_method": [EXPECTED_FIT_METHOD],
            "coordinate_unit_input": ["meters"],
            "voxel_size_mm": [0.7],
        }
    )
    cons = version_revision_consistency(df, expected_frame="baseline")
    assert cons["ok"] is False
    assert any("coordinate_unit" in r or "voxel_size" in r for r in cons["reasons"])


def test_resume_requires_identity_fields():
    from sin_trend_paths import verify_participant_complete_for_resume

    status = {
        "status": "complete",
        # identity intentionally missing
        "n_exceptions": 0,
        "n_save_failures": 0,
        "ambiguous_group_ids": [],
        "surface_enabled": False,
        "surface_status_counts": {},
        "n_intervals": 1,
        "input_vtp_sha256": "abc",
    }
    ok, reason = verify_participant_complete_for_resume(
        status, run_id="run1", frame_reference="baseline", directory_id="IXI001", expect_surface=False
    )
    assert ok is False
    assert "missing" in reason


def test_bool_parser_int2_unknown():
    from sin_trend_classify import parse_tri_bool
    from sin_trend_dataset import parse_bool_series

    assert parse_tri_bool(2) is None
    s = parse_bool_series(pd.Series([0, 1, 2]))
    assert bool(s.iloc[0]) is False
    assert bool(s.iloc[1]) is True
    assert pd.isna(s.iloc[2])


def test_circular_anisotropy_is_zero_not_nan():
    from sin_trend_forward_fit import amplitudes_from_coef

    amp2 = amplitudes_from_coef(0.5, 0.5, 0.5, -0.5)
    # When sigma1≈sigma2, anisotropy value is ~0 and principal_axis_unique is 0
    if np.isclose(amp2["sigma1"], amp2["sigma2"], rtol=1e-6):
        assert abs(amp2["anisotropy_1_minus_s2_over_s1"]) < 1e-9
        assert amp2["principal_axis_unique"] == 0


def test_unreadable_status_does_not_crash_manifest():
    import json
    import tempfile
    from sin_trend_paths import build_run_manifest, read_participant_status, participant_status_path

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # minimal project layout expected by paths
        (root / "results" / "morphometrics" / "sin_trend" / "run_p01" / "baseline" / "per_case").mkdir(parents=True)
        sp = participant_status_path("run_p01", "baseline", "IXI001", root)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text('{"status":', encoding="utf-8")
        st = read_participant_status("run_p01", "baseline", "IXI001", root)
        assert st is not None
        assert st["status"] == "status_unreadable"
        # list/null
        sp.write_text("[]", encoding="utf-8")
        st2 = read_participant_status("run_p01", "baseline", "IXI001", root)
        assert st2["status"] == "status_unreadable"
        sp.write_text(json.dumps({"status": "partial"}), encoding="utf-8")
        man = build_run_manifest("run_p01", ["IXI001", "IXI002"], ["baseline"], root)
        assert man["frames"]["baseline"]["participants"]["IXI001"]["status"] == "partial"
        assert man["frames"]["baseline"]["participants"]["IXI002"]["status"] == "missing"
        # broken again must not raise
        sp.write_text("{bad", encoding="utf-8")
        man2 = build_run_manifest("run_p01", ["IXI001"], ["baseline"], root)
        assert man2["frames"]["baseline"]["counts"]["status_unreadable"] == 1


def test_empty_table_schema_ok_partial_unit_missing_rejected():
    from sin_trend_measure import measurement_row_schema
    from sin_trend_paths import (
        EXPECTED_FIT_METHOD,
        EXPECTED_MEASUREMENT_REVISION,
        EXPECTED_MEASUREMENT_VERSION,
        EXPECTED_OUTPUT_TAG,
        version_revision_consistency,
    )

    cols = list(measurement_row_schema().keys())
    empty = pd.DataFrame(columns=cols)
    cons0 = version_revision_consistency(empty, expected_frame="baseline")
    assert cons0["ok"] is True
    assert cons0["n_rows"] == 0

    good = {
        "directory_id": ["A", "B"],
        "segment_id": [0, 1],
        "segment_index": [0, 1],
        "output_tag": [EXPECTED_OUTPUT_TAG, EXPECTED_OUTPUT_TAG],
        "measurement_version": [EXPECTED_MEASUREMENT_VERSION, EXPECTED_MEASUREMENT_VERSION],
        "measurement_revision": [EXPECTED_MEASUREMENT_REVISION, EXPECTED_MEASUREMENT_REVISION],
        "frame_reference": ["baseline", "baseline"],
        "fit_method": [EXPECTED_FIT_METHOD, EXPECTED_FIT_METHOD],
        "coordinate_unit_input": ["voxel", "voxel"],
        "voxel_size_mm": [0.5, 0.5],
    }
    assert version_revision_consistency(pd.DataFrame(good), expected_frame="baseline")["ok"]

    partial = dict(good)
    partial["voxel_size_mm"] = [0.5, np.nan]
    assert version_revision_consistency(pd.DataFrame(partial), expected_frame="baseline")["ok"] is False

    no_unit_col = {k: v for k, v in good.items() if k != "coordinate_unit_input"}
    assert version_revision_consistency(pd.DataFrame(no_unit_col), expected_frame="baseline")["ok"] is False


def test_resume_rejects_duplicate_and_missing_keys():
    import tempfile
    from sin_trend_paths import (
        EXPECTED_FIT_METHOD,
        EXPECTED_MEASUREMENT_REVISION,
        EXPECTED_MEASUREMENT_VERSION,
        centerline_vtp_path,
        interval_key_set_hash,
        sha256_file,
        verify_participant_complete_for_resume,
    )

    def _base_df():
        return pd.DataFrame(
            {
                "directory_id": ["IXI001", "IXI001"],
                "segment_id": [1, 2],
                "segment_index": [0, 1],
                "frame_reference": ["baseline", "baseline"],
                "fit_method": [EXPECTED_FIT_METHOD, EXPECTED_FIT_METHOD],
                "measurement_version": [EXPECTED_MEASUREMENT_VERSION, EXPECTED_MEASUREMENT_VERSION],
                "measurement_revision": [EXPECTED_MEASUREMENT_REVISION, EXPECTED_MEASUREMENT_REVISION],
            }
        )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        df = _base_df()
        vtp = centerline_vtp_path("IXI001", root)
        vtp.parent.mkdir(parents=True, exist_ok=True)
        vtp.write_bytes(b"dummy")
        vtp_sha = sha256_file(vtp)

        dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        dup_path = root / "dup.csv"
        dup.to_csv(dup_path, index=False)
        status = {
            "status": "complete",
            "run_id": "r",
            "directory_id": "IXI001",
            "frame_reference": "baseline",
            "measurement_version": EXPECTED_MEASUREMENT_VERSION,
            "measurement_revision": EXPECTED_MEASUREMENT_REVISION,
            "fit_method": EXPECTED_FIT_METHOD,
            "n_exceptions": 0,
            "n_save_failures": 0,
            "ambiguous_group_ids": [],
            "surface_enabled": False,
            "surface_status_counts": {},
            "n_intervals": 3,
            "input_vtp_sha256": vtp_sha,
            "per_case_csv": str(dup_path),
            "interval_key_set_sha256": interval_key_set_hash(df),
            "per_case_csv_sha256": sha256_file(dup_path),
        }
        ok, reason = verify_participant_complete_for_resume(
            status, run_id="r", frame_reference="baseline", directory_id="IXI001", project_root=root, expect_surface=False
        )
        assert ok is False
        assert "duplicate" in reason or "row_mismatch" in reason

        miss = _base_df()
        miss.loc[0, "segment_index"] = np.nan
        miss_path = root / "miss.csv"
        miss.to_csv(miss_path, index=False)
        status2 = dict(status)
        status2["per_case_csv"] = str(miss_path)
        status2["n_intervals"] = 2
        status2["per_case_csv_sha256"] = sha256_file(miss_path)
        status2.pop("interval_key_set_sha256", None)
        ok2, reason2 = verify_participant_complete_for_resume(
            status2, run_id="r", frame_reference="baseline", directory_id="IXI001", project_root=root, expect_surface=False
        )
        assert ok2 is False
        assert "segment_index" in reason2 or "missing" in reason2


def test_parse_tri_bool_empty_string_is_missing():
    from sin_trend_classify import parse_tri_bool

    assert parse_tri_bool("") is None
    assert parse_tri_bool("  ") is None


def test_run_config_rejects_nan_and_zero_voxel_independent_of_rows():
    from sin_trend_measure import measurement_row_schema
    from sin_trend_paths import (
        EXPECTED_FIT_METHOD,
        EXPECTED_MEASUREMENT_REVISION,
        EXPECTED_MEASUREMENT_VERSION,
        EXPECTED_OUTPUT_TAG,
        validate_run_config_contents,
        version_revision_consistency,
    )

    cols = list(measurement_row_schema().keys())
    empty = pd.DataFrame(columns=cols)
    good_cfg = {
        "output_tag": EXPECTED_OUTPUT_TAG,
        "measurement_version": EXPECTED_MEASUREMENT_VERSION,
        "measurement_revision": EXPECTED_MEASUREMENT_REVISION,
        "config": {"voxel_size": 0.5},
    }
    assert validate_run_config_contents(good_cfg)["ok"]
    assert version_revision_consistency(empty, expected_frame="baseline", run_config=good_cfg, require_run_config=True)["ok"]

    for bad_vs in (float("nan"), "NaN", float("inf"), 0.0, -0.5, "abc", None):
        bad = dict(good_cfg)
        bad["config"] = {"voxel_size": bad_vs}
        assert validate_run_config_contents(bad)["ok"] is False
        cons = version_revision_consistency(empty, expected_frame="baseline", run_config=bad, require_run_config=True)
        assert cons["ok"] is False, bad_vs

    assert validate_run_config_contents(None, require=True)["ok"] is False
    assert validate_run_config_contents({}, require=True)["ok"] is False
    wrong_rev = dict(good_cfg)
    wrong_rev["measurement_revision"] = "not_the_expected"
    assert version_revision_consistency(empty, run_config=wrong_rev, require_run_config=True)["ok"] is False

                                
    row = {
        "directory_id": ["A"],
        "segment_id": [0],
        "segment_index": [0],
        "output_tag": [EXPECTED_OUTPUT_TAG],
        "measurement_version": [EXPECTED_MEASUREMENT_VERSION],
        "measurement_revision": [EXPECTED_MEASUREMENT_REVISION],
        "frame_reference": ["baseline"],
        "fit_method": [EXPECTED_FIT_METHOD],
        "coordinate_unit_input": ["voxel"],
        "voxel_size_mm": [0.5],
    }
    nan_cfg = dict(good_cfg)
    nan_cfg["config"] = {"voxel_size": float("nan")}
    assert version_revision_consistency(pd.DataFrame(row), expected_frame="baseline", run_config=nan_cfg, require_run_config=True)["ok"] is False


def test_utf8_corrupt_status_is_unreadable():
    import tempfile
    from sin_trend_paths import build_run_manifest, read_participant_status, participant_status_path

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "results" / "morphometrics" / "sin_trend" / "run_u" / "baseline" / "per_case").mkdir(parents=True)
        sp = participant_status_path("run_u", "baseline", "IXI001", root)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_bytes(b"\xff")
        st = read_participant_status("run_u", "baseline", "IXI001", root)
        assert st is not None
        assert st["status"] == "status_unreadable"
        assert "UnicodeDecodeError" in str(st.get("status_read_error", ""))
        man = build_run_manifest("run_u", ["IXI001"], ["baseline"], root)
        assert man["frames"]["baseline"]["counts"]["status_unreadable"] == 1


def test_resume_requires_key_and_csv_hashes():
    import tempfile
    from sin_trend_paths import (
        EXPECTED_FIT_METHOD,
        EXPECTED_MEASUREMENT_REVISION,
        EXPECTED_MEASUREMENT_VERSION,
        centerline_vtp_path,
        interval_key_set_hash,
        sha256_file,
        verify_participant_complete_for_resume,
    )

    def _status(path, n, key_hash=None, csv_sha=None):
        st = {
            "status": "complete",
            "run_id": "r",
            "directory_id": "IXI001",
            "frame_reference": "baseline",
            "measurement_version": EXPECTED_MEASUREMENT_VERSION,
            "measurement_revision": EXPECTED_MEASUREMENT_REVISION,
            "fit_method": EXPECTED_FIT_METHOD,
            "n_exceptions": 0,
            "n_save_failures": 0,
            "ambiguous_group_ids": [],
            "surface_enabled": False,
            "surface_status_counts": {},
            "n_intervals": n,
            "input_vtp_sha256": None,  # filled below
            "per_case_csv": str(path),
        }
        if key_hash is not None:
            st["interval_key_set_sha256"] = key_hash
        if csv_sha is not None:
            st["per_case_csv_sha256"] = csv_sha
        return st

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vtp = centerline_vtp_path("IXI001", root)
        vtp.parent.mkdir(parents=True, exist_ok=True)
        vtp.write_bytes(b"dummy")
        vtp_sha = sha256_file(vtp)

        df_a = pd.DataFrame(
            {
                "directory_id": ["IXI001", "IXI001"],
                "segment_id": [1, 2],
                "segment_index": [0, 1],
                "frame_reference": ["baseline", "baseline"],
                "fit_method": [EXPECTED_FIT_METHOD, EXPECTED_FIT_METHOD],
                "measurement_version": [EXPECTED_MEASUREMENT_VERSION, EXPECTED_MEASUREMENT_VERSION],
                "measurement_revision": [EXPECTED_MEASUREMENT_REVISION, EXPECTED_MEASUREMENT_REVISION],
            }
        )
        df_b = df_a.copy()
        df_b["segment_id"] = [3, 4]  # same row count, different key set
        path_b = root / "alt.csv"
        df_b.to_csv(path_b, index=False)
        st = _status(path_b, 2)
        st["input_vtp_sha256"] = vtp_sha
        # both hashes missing → must reject (not silently ok)
        ok, reason = verify_participant_complete_for_resume(
            st, run_id="r", frame_reference="baseline", directory_id="IXI001", project_root=root, expect_surface=False
        )
        assert ok is False
        assert "missing_interval_key_set_sha256" in reason or "missing_per_case_csv_sha256" in reason

        # wrong key hash stored
        st2 = _status(path_b, 2, key_hash=interval_key_set_hash(df_a), csv_sha=sha256_file(path_b))
        st2["input_vtp_sha256"] = vtp_sha
        ok2, reason2 = verify_participant_complete_for_resume(
            st2, run_id="r", frame_reference="baseline", directory_id="IXI001", project_root=root, expect_surface=False
        )
        assert ok2 is False
        assert "interval_key_set" in reason2

                                                                        
        empty = pd.DataFrame(
            columns=[
                "directory_id",
                "segment_id",
                "segment_index",
                "frame_reference",
                "fit_method",
                "measurement_version",
                "measurement_revision",
            ]
        )
        empty_path = root / "empty.csv"
        empty.to_csv(empty_path, index=False)
        st0 = _status(
            empty_path,
            0,
            key_hash=interval_key_set_hash(empty),
            csv_sha=sha256_file(empty_path),
        )
        st0["input_vtp_sha256"] = vtp_sha
        st0["n_segments_in_vtp"] = 0
        ok0, reason0 = verify_participant_complete_for_resume(
            st0, run_id="r", frame_reference="baseline", directory_id="IXI001", project_root=root, expect_surface=False
        )
        assert ok0 is True, reason0


def test_resume_full_request_rejects_zero_of_nonzero_vtp():
    import tempfile
    from sin_trend_paths import (
        EXPECTED_FIT_METHOD,
        EXPECTED_MEASUREMENT_REVISION,
        EXPECTED_MEASUREMENT_VERSION,
        centerline_vtp_path,
        interval_key_set_hash,
        sha256_file,
        verify_participant_complete_for_resume,
    )

    cols = [
        "directory_id",
        "segment_id",
        "segment_index",
        "frame_reference",
        "fit_method",
        "measurement_version",
        "measurement_revision",
    ]

    def _base_status(path, n, *, key_hash, csv_sha, vtp_sha, n_vtp, segment_filter=None):
        return {
            "status": "complete",
            "run_id": "r",
            "directory_id": "IXI001",
            "frame_reference": "baseline",
            "measurement_version": EXPECTED_MEASUREMENT_VERSION,
            "measurement_revision": EXPECTED_MEASUREMENT_REVISION,
            "fit_method": EXPECTED_FIT_METHOD,
            "n_exceptions": 0,
            "n_save_failures": 0,
            "ambiguous_group_ids": [],
            "surface_enabled": False,
            "surface_status_counts": {},
            "n_intervals": n,
            "n_segments_in_vtp": n_vtp,
            "input_vtp_sha256": vtp_sha,
            "per_case_csv": str(path),
            "interval_key_set_sha256": key_hash,
            "per_case_csv_sha256": csv_sha,
            "segment_filter": segment_filter,
        }

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vtp = centerline_vtp_path("IXI001", root)
        vtp.parent.mkdir(parents=True, exist_ok=True)
        vtp.write_bytes(b"dummy")
        vtp_sha = sha256_file(vtp)

        empty = pd.DataFrame(columns=cols)
        empty_path = root / "empty.csv"
        empty.to_csv(empty_path, index=False)
        empty_hash = interval_key_set_hash(empty)
        empty_sha = sha256_file(empty_path)

                                                             
        st_bad = _base_status(
            empty_path, 0, key_hash=empty_hash, csv_sha=empty_sha, vtp_sha=vtp_sha, n_vtp=2, segment_filter=[]
        )
        ok_bad, reason_bad = verify_participant_complete_for_resume(
            st_bad,
            run_id="r",
            frame_reference="baseline",
            directory_id="IXI001",
            project_root=root,
            expect_surface=False,
            expect_segment_ids=None,
        )
        assert ok_bad is False
        assert "incomplete_interval_coverage" in reason_bad

                
        df2 = pd.DataFrame(
            {
                "directory_id": ["IXI001", "IXI001"],
                "segment_id": [1, 2],
                "segment_index": [0, 1],
                "frame_reference": ["baseline", "baseline"],
                "fit_method": [EXPECTED_FIT_METHOD, EXPECTED_FIT_METHOD],
                "measurement_version": [EXPECTED_MEASUREMENT_VERSION, EXPECTED_MEASUREMENT_VERSION],
                "measurement_revision": [EXPECTED_MEASUREMENT_REVISION, EXPECTED_MEASUREMENT_REVISION],
            }
        )
        path2 = root / "full.csv"
        df2.to_csv(path2, index=False)
        st2 = _base_status(
            path2,
            2,
            key_hash=interval_key_set_hash(df2),
            csv_sha=sha256_file(path2),
            vtp_sha=vtp_sha,
            n_vtp=2,
            segment_filter=None,
        )
        ok2, reason2 = verify_participant_complete_for_resume(
            st2,
            run_id="r",
            frame_reference="baseline",
            directory_id="IXI001",
            project_root=root,
            expect_surface=False,
            expect_segment_ids=None,
        )
        assert ok2 is True, reason2

                                                   
        st_sub = _base_status(
            empty_path,
            0,
            key_hash=empty_hash,
            csv_sha=empty_sha,
            vtp_sha=vtp_sha,
            n_vtp=2,
            segment_filter=[999],
        )
        ok_sub, reason_sub = verify_participant_complete_for_resume(
            st_sub,
            run_id="r",
            frame_reference="baseline",
            directory_id="IXI001",
            project_root=root,
            expect_surface=False,
            expect_segment_ids=[999],
        )
        assert ok_sub is True, reason_sub

                  
        st00 = _base_status(
            empty_path, 0, key_hash=empty_hash, csv_sha=empty_sha, vtp_sha=vtp_sha, n_vtp=0, segment_filter=None
        )
        ok00, reason00 = verify_participant_complete_for_resume(
            st00,
            run_id="r",
            frame_reference="baseline",
            directory_id="IXI001",
            project_root=root,
            expect_surface=False,
            expect_segment_ids=None,
        )
        assert ok00 is True, reason00


def test_grid_png_lambda_ticks_use_mm_coords():
    import tempfile
    from unittest.mock import patch

    from sin_trend_visualize import save_grid_search_png

                                                                    
    ks = np.linspace(0.0135, 0.80, 50)
    cand = pd.DataFrame(
        {
            "k_rad_per_voxel": ks,
            "R2": 0.5 + 0.3 * np.exp(-((ks - 0.2) ** 2) / 0.005),
            "partial_R2": 0.4 + 0.2 * np.exp(-((ks - 0.2) ** 2) / 0.005),
        }
    )
    search = {"ks": ks, "k_min": float(ks[0]), "k_max": float(ks[-1])}
    recorded = []

    def _capture(self, ticks, minor=False, **kwargs):
        recorded.append(np.asarray(ticks, float).copy())
        return orig_set_xticks(self, ticks, minor=minor, **kwargs)

    from matplotlib.axes import _secondary_axes as sec_mod

    orig_set_xticks = sec_mod.SecondaryAxis.set_xticks
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "grid_lam.png"
        with patch.object(sec_mod.SecondaryAxis, "set_xticks", _capture):
            st = save_grid_search_png(
                out_path=out,
                search=search,
                candidates=cand,
                L_trimmed_vox=60.0,
                k_hat_vox=0.2,
                voxel_size_mm=0.5,
            )
        assert st == "ok", st
        assert out.is_file() and out.stat().st_size > 0
        assert recorded, "set_xticks was not called on secondary axis"
        ticks = recorded[-1]
        assert len(ticks) >= 1
                                  
        assert float(np.min(ticks)) > 2.0, ticks
        assert float(np.max(ticks)) < 300.0, ticks


def test_radius_trimmed_vs_full_median():
    from sin_trend_measure import default_measurement_config, measure_interval

    n = 40
    removal = int(default_measurement_config()["removal"])
    x = np.linspace(0, 30.0, n)
    P = np.column_stack([x, np.zeros(n), np.zeros(n)])
                                               
    r = np.concatenate(
        [
            np.full(removal, 0.2),
            np.full(n - 2 * removal, 2.0),
            np.full(removal, 0.2),
        ]
    )
    cfg = default_measurement_config()
    cfg["verbose"] = False
    res = measure_interval(P, frame_reference="baseline", cfg=cfg, radius_vox=r, compute_pca=False)
    row = res["row"]
    assert row["radius_status_full"] == "ok"
    assert row["radius_status"] == "ok"
    r_full = np.asarray(r, float)
    r_trim = r_full[removal : n - removal]
    assert abs(float(row["radius_median_full_mm"]) - float(np.median(r_full)) * 0.5) < 1e-12
    assert abs(float(row["radius_median_mm"]) - float(np.median(r_trim)) * 0.5) < 1e-12
    assert float(row["radius_median_full_mm"]) != float(row["radius_median_mm"])
    assert abs(float(row["diameter_ref_mm"]) - 2.0 * float(row["radius_median_mm"])) < 1e-12


def test_radius_nan_outside_trim_keeps_trimmed_ok():
    from sin_trend_measure import default_measurement_config, measure_interval

    n = 120
    removal = int(default_measurement_config()["removal"])
    x = np.linspace(0, 40.0, n)
    k = 2 * np.pi / 20.0
    P = np.column_stack([x, 0.4 * np.sin(k * x), 0.2 * np.cos(k * x)])
    cfg = default_measurement_config()
    cfg["verbose"] = False
    r_ok = np.full(n, 2.2)  # voxel; median_mm = 1.1 at voxel_size=0.5

                                         
    r_edge = r_ok.copy()
    r_edge[0] = np.nan
    row_edge = measure_interval(
        P, frame_reference="baseline", cfg=cfg, radius_vox=r_edge, compute_pca=False
    )["row"]
    assert row_edge["radius_status_full"] == "invalid"
    assert row_edge["radius_status"] == "ok"
    assert abs(float(row_edge["radius_median_mm"]) - 1.1) < 1e-12
    assert abs(float(row_edge["diameter_ref_mm"]) - 2.2) < 1e-12
    assert row_edge.get("fit_status") != "missing_diameter"
    assert np.isfinite(float(row_edge.get("lambda_min_search_mm", np.nan)))
    assert np.isfinite(float(row_edge.get("r_patch_mm", np.nan)))
                                               
    assert row_edge.get("fit_status") == "ok", row_edge.get("fit_status")
    assert np.isfinite(float(row_edge["lambda_hat_mm"]))
    assert np.isfinite(float(row_edge["L_base_mm"]))
    assert abs(float(row_edge["lambda_baseline_eff_mm"]) - float(row_edge["lambda_hat_mm"])) < 1e-12
    expect_sl = float(row_edge["lambda_hat_mm"]) / (2.0 * float(row_edge["radius_median_mm"]))
    assert np.isfinite(float(row_edge["slenderness"]))
    assert abs(float(row_edge["slenderness"]) - expect_sl) < 1e-12
                                              
    assert abs(
        float(row_edge["coverage_hat"])
        - float(row_edge["L_base_mm"]) / float(row_edge["lambda_hat_mm"])
    ) < 1e-9

                         
    r_in = r_ok.copy()
    r_in[removal] = np.nan
    row_in = measure_interval(
        P, frame_reference="baseline", cfg=cfg, radius_vox=r_in, compute_pca=False
    )["row"]
    assert row_in["radius_status_full"] == "invalid"
    assert row_in["radius_status"] == "invalid"
    assert not np.isfinite(float(row_in.get("diameter_ref_mm", np.nan)))

                    
    row_len = measure_interval(
        P, frame_reference="baseline", cfg=cfg, radius_vox=r_ok[: n - 1], compute_pca=False
    )["row"]
    assert row_len["radius_status"] == "invalid"
    assert row_len["radius_status_full"] == "invalid"

          
    row_miss = measure_interval(
        P, frame_reference="baseline", cfg=cfg, radius_vox=None, compute_pca=False
    )["row"]
    assert row_miss["radius_status"] == "missing"
    assert row_miss["radius_status_full"] == "missing"


def test_parse_json_bool_strict():
    from sin_trend_paths import parse_json_bool

    assert parse_json_bool(True) is True
    assert parse_json_bool(False) is False
    assert parse_json_bool(None) is None
    assert parse_json_bool("false") is None
    assert parse_json_bool("true") is None
    assert parse_json_bool(0) is None
    assert parse_json_bool(1) is None


if __name__ == "__main__":
    failed = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed.append((name, exc))
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failed else 0)
