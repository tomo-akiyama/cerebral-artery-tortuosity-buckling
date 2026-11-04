#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: test good anchor bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))

from sin_trend_fitting_helpers import (  # noqa: E402
    EVAL_NORMAL_SOURCE_BRIDGED,
    EVAL_NORMAL_SOURCE_GOOD,
    EVAL_NORMAL_SOURCE_NONE,
    interpolate_good_anchor_normals_with_bridge,
    unsigned_axis_angle_deg,
)


def test_unsigned_axis_angle_ignores_sign():
    n0 = np.array([1.0, 0.0, 0.0])
    n1 = np.array([-1.0, 0.0, 0.0])
    assert abs(unsigned_axis_angle_deg(n0, n1) - 0.0) < 1e-9


def test_adjacent_good_anchors_are_good_source_not_bridged():
    s_full = np.linspace(0.0, 2.0, 21)  # voxel units
    s_anchors = np.array([0.0, 1.0, 2.0])
    n = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    good = np.array([True, True, True])
    out = interpolate_good_anchor_normals_with_bridge(
        s_full, s_anchors, n, good, voxel_size_mm=0.5, bridge_max_gap_mm=5.0
    )
    mid = (s_full > 0.0) & (s_full < 2.0)
    assert np.all(out["eval_normal_source"][mid] == EVAL_NORMAL_SOURCE_GOOD)
    assert not np.any(out["eval_normal_source"] == EVAL_NORMAL_SOURCE_BRIDGED)


def test_short_low_coh_gap_is_bridged_without_using_middle_normal():
    # anchors at 0, 2, 4 mm (s voxel with vs=0.5 → s=0,4,8); end-to-end gap=4 mm ≤ 5
    vs = 0.5
    s_anchors = np.array([0.0, 4.0, 8.0])
    s_full = np.linspace(0.0, 8.0, 9)
    # ends aligned; middle would contaminate if used (orthogonal)
    n = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    good = np.array([True, False, True])
    out = interpolate_good_anchor_normals_with_bridge(
        s_full,
        s_anchors,
        n,
        good,
        voxel_size_mm=vs,
        bridge_max_gap_mm=5.0,
        bridge_axis_max_deg=45.0,
    )
    between = (s_full > 0.0) & (s_full < 8.0)
    assert np.any(out["eval_normal_source"][between] == EVAL_NORMAL_SOURCE_BRIDGED)
    mid_n = out["normals_eval"][np.isclose(s_full, 4.0)][0]
    assert abs(mid_n[1]) < 0.2
    assert abs(mid_n[0]) > 0.9


def test_gap_longer_than_max_is_not_bridged():
    vs = 0.5
    s_anchors = np.array([0.0, 6.0, 14.0])  # ends span 14*0.5=7 mm > 5
    s_full = np.linspace(0.0, 14.0, 15)
    n = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    good = np.array([True, False, True])
    out = interpolate_good_anchor_normals_with_bridge(
        s_full, s_anchors, n, good, voxel_size_mm=vs, bridge_max_gap_mm=5.0
    )
    between = (s_full > 0.0) & (s_full < 14.0)
    assert np.all(out["eval_normal_source"][between] == EVAL_NORMAL_SOURCE_NONE)
    assert not np.any(out["has_constraint_full"][between])


def test_one_sided_end_is_not_filled():
    s_full = np.linspace(0.0, 10.0, 11)
    s_anchors = np.array([0.0, 5.0, 10.0])
    n = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    good = np.array([True, False, False])
    out = interpolate_good_anchor_normals_with_bridge(
        s_full, s_anchors, n, good, voxel_size_mm=0.5, bridge_max_gap_mm=5.0
    )
    after = s_full > 0.0
    assert np.all(out["eval_normal_source"][after] == EVAL_NORMAL_SOURCE_NONE)
    assert out["eval_normal_source"][0] == EVAL_NORMAL_SOURCE_GOOD


def test_large_axis_angle_blocks_bridge():
    vs = 0.5
    s_anchors = np.array([0.0, 4.0, 8.0])  # 4 mm span
    s_full = np.linspace(0.0, 8.0, 9)
    # 90° unsigned axis difference
    n = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    good = np.array([True, False, True])
    out = interpolate_good_anchor_normals_with_bridge(
        s_full,
        s_anchors,
        n,
        good,
        voxel_size_mm=vs,
        bridge_max_gap_mm=5.0,
        bridge_axis_max_deg=45.0,
    )
    between = (s_full > 0.0) & (s_full < 8.0)
    assert np.all(out["eval_normal_source"][between] == EVAL_NORMAL_SOURCE_NONE)


def test_cumulative_arclength_vs_index_times_ds_for_bridge_gap():
    vs = 0.5
    P = np.array(
        [[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [4.0, 0.0, 1.0], [8.0, 0.0, 1.0], [16.0, 0.0, 1.0]],
        float,
    )
    s_full = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    assert abs(s_full[-1] * vs - 8.0) < 1e-9
    n = np.tile(np.array([0.0, 0.0, 1.0]), (5, 1))
    good = np.array([True, False, False, False, True])
    out = interpolate_good_anchor_normals_with_bridge(
        s_full, s_full, n, good, voxel_size_mm=vs, bridge_max_gap_mm=5.0, bridge_axis_max_deg=45.0
    )
    mid = (s_full > 0) & (s_full < s_full[-1])
    assert np.all(out["eval_normal_source"][mid] == EVAL_NORMAL_SOURCE_NONE)

    s_wrong = np.arange(5, dtype=float) * 1.0  # gap=4 vox → 2 mm ≤ 5
    out_wrong = interpolate_good_anchor_normals_with_bridge(
        s_wrong, s_wrong, n, good, voxel_size_mm=vs, bridge_max_gap_mm=5.0, bridge_axis_max_deg=45.0
    )
    assert np.any(out_wrong["eval_normal_source"][mid] == EVAL_NORMAL_SOURCE_BRIDGED)


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
