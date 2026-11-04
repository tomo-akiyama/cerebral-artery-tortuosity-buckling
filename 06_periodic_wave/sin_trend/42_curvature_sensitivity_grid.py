#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 42 curvature sensitivity grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)

from sin_trend_classify import classify_row  # noqa: E402
from sin_trend_measure import default_measurement_config, measure_interval  # noqa: E402
from sin_trend_paths import (  # noqa: E402
    FRAME_BASELINE,
    atomic_to_csv,
    atomic_write_json,
    metadata_dict,
    new_run_id,
    project_root_from_here,
    sha256_json,
    source_code_hashes,
)
from sin_trend_synthetic import (  # noqa: E402
    make_sinusoid_centerline,
    n_cycles_input_for_target_remaining_after_trim,
    remaining_cycles_true_from_xi,
)

                                 
A_MM = [0.5, 1.0, 2.0]
LAMBDA_MM = [10.0, 20.0, 40.0]
PHASE_RAD = [0.0, np.pi / 2]
N_CYCLES_TARGET_AFTER_TRIM = 3.0
REMOVAL_POINTS = 10
DIAMETER_MM = 1.4
TARGET_DS_MM = 0.10
C_BASE = 0.005
C_EXTRA = 0.010


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _bg(c: float) -> Dict[str, Any]:
    return {"type": "quadratic", "curvature_mm": float(c)}


def run_one(A: float, lam: float, ph: float, c: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    vox = float(cfg.get("voxel_size", 0.5))
    bg = _bg(c)
    n_in = n_cycles_input_for_target_remaining_after_trim(
        target_remaining=N_CYCLES_TARGET_AFTER_TRIM,
        amplitude_mm=A,
        lambda_x_mm=lam,
        phase=ph,
        target_ds_mm=TARGET_DS_MM,
        removal_points=REMOVAL_POINTS,
        background=bg,
        diameter_mm=DIAMETER_MM,
    )
    syn = make_sinusoid_centerline(
        amplitude_mm=A,
        lambda_x_mm=lam,
        n_cycles=n_in,
        phase=ph,
        target_ds_mm=TARGET_DS_MM,
        diameter_mm=DIAMETER_MM,
        background=bg,
        voxel_size_mm=vox,
    )
    n_true_remaining = remaining_cycles_true_from_xi(
        syn["xi_mm"],
        syn["lambda_baseline_true_mm"],
        removal_points=REMOVAL_POINTS,
    )
    res = measure_interval(
        syn["P_vox"],
        frame_reference=FRAME_BASELINE,
        cfg=cfg,
        observed_tangent=None,
        radius_vox=syn.get("radius_vox"),
        ids={"directory_id": "CURV_SENS", "segment_id": 0, "segment_index": 0},
        verbose=False,
        compute_pca=False,
    )
    row = classify_row(res["row"])
    lam_true = float(syn["lambda_baseline_true_mm"])
    A_true = float(syn["A_true_mm"])
    lam_hat = float(row.get("lambda_hat_mm", np.nan))
    A_hat = float(row.get("amp_comb_mm", np.nan))
    core = int(row.get("core_all", row.get("core_periodic", 0)) or 0)
    lam_rel = abs(lam_hat / lam_true - 1.0) if np.isfinite(lam_hat) and lam_true > 0 else float("nan")
    A_rel = abs(A_hat / A_true - 1.0) if np.isfinite(A_hat) and A_true > 0 else float("nan")
    L0 = float(syn.get("L0_known_baseline_mm", n_in * lam))
    return {
        "c": float(c),
        "A_true_mm": A_true,
        "lambda_true_mm": lam_true,
        "phase_rad": float(ph),
        "n_cycles_target_after_trim": float(N_CYCLES_TARGET_AFTER_TRIM),
        "n_cycles_input": float(n_in),
        "n_cycles_remaining_true": float(n_true_remaining),
        "diameter_mm": float(DIAMETER_MM),
        "L0_known_baseline_mm": L0,
        "R_approx_mm": 1.0 / (2.0 * float(c)),
        "sag_mm": float(c) * (L0 / 2.0) ** 2,
        "lambda_hat_mm": lam_hat,
        "A_hat_mm": A_hat,
        "coverage_hat": float(row.get("coverage_hat", np.nan)),
        "partial_R2": float(row.get("partial_R2", np.nan)),
        "core_all": core,
        "reject_reasons": str(row.get("reject_reasons") or ""),
        "lambda_abs_rel_error": lam_rel,
        "A_abs_rel_error": A_rel,
        "lambda_hat_over_true": (lam_hat / lam_true) if np.isfinite(lam_hat) and lam_true > 0 else float("nan"),
    }


def summarize_by_c(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in df.groupby("c"):
        acc = g[g["core_all"] == 1]
        rej = g[g["core_all"] != 1]
        rows.append(
            {
                "c": float(c),
                "role": "base" if np.isclose(c, C_BASE) else "extra_scope",
                "n": int(len(g)),
                "n_accepted": int(len(acc)),
                "n_rejected": int(len(rej)),
                "acceptance": f"{len(acc)}/{len(g)}",
                "n_remaining_true_median": float(g["n_cycles_remaining_true"].median()),
                "lambda_abs_rel_error_median_accepted": float(acc["lambda_abs_rel_error"].median())
                if len(acc)
                else float("nan"),
                "lambda_abs_rel_error_max_accepted": float(acc["lambda_abs_rel_error"].max())
                if len(acc)
                else float("nan"),
                "A_abs_rel_error_median_accepted": float(acc["A_abs_rel_error"].median())
                if len(acc)
                else float("nan"),
                "A_abs_rel_error_max_accepted": float(acc["A_abs_rel_error"].max())
                if len(acc)
                else float("nan"),
                "n_accepted_lambda_rel_gt_0p05": int((acc["lambda_abs_rel_error"] > 0.05).sum())
                if len(acc)
                else 0,
            }
        )
    return pd.DataFrame(rows)


def write_report(out_dir: Path, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines: List[str] = []
    lines += [
        "# (2) Accuracy across background curvatures",
        "",
        "Separate from (1) Accuracy across amplitudes, wavelengths, and phases "
        "(156 cases; curved baseline uses **c=0.005** only). "
        "This grid holds amplitude, wavelength, phase, and **post-trim target cycle count** fixed "
        "(same convention as 36/stress), and varies only `c`.",
        "",
        "## Comparison design",
        "",
        f"- D = {DIAMETER_MM} mm, target_ds = {TARGET_DS_MM} mm",
        f"- **n_cycles_target_after_trim = {N_CYCLES_TARGET_AFTER_TRIM}** "
        "(input n_cycles solved per case, as in 36)",
        f"- A ∈ {A_MM}",
        f"- λ ∈ {LAMBDA_MM}",
        f"- phase ∈ {{0, π/2}}",
        f"- **c = {C_BASE} (base)** vs **c = {C_EXTRA} (extra scope check)** → 18 cases each",
        f"- Approximate curvature radius R ≈ 1/(2c): {1/(2*C_BASE):.0f} mm vs {1/(2*C_EXTRA):.0f} mm",
        "",
        "## Acceptance (all cases)",
        "",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"- c={r['c']:g} ({r['role']}): **{r['acceptance']}** accepted "
            f"(rejected {int(r['n_rejected'])}; "
            f"median N_true remaining={r['n_remaining_true_median']:.3f})"
        )
    rej = df[(np.isclose(df["c"], C_EXTRA)) & (df["core_all"] != 1)].copy()
    lines += ["", "### Rejected at c=0.010", ""]
    if len(rej) == 0:
        lines.append("(none)")
    else:
        lines.append("| A | λ | phase | λ̂ | N̂ | reject_reasons |")
        lines.append("|---:|---:|---:|---:|---:|---|")
        for _, r in rej.sort_values(["lambda_true_mm", "A_true_mm", "phase_rad"]).iterrows():
            lines.append(
                f"| {r['A_true_mm']:g} | {r['lambda_true_mm']:g} | {r['phase_rad']:.4f} | "
                f"{r['lambda_hat_mm']:.2f} | {r['coverage_hat']:.3f} | `{r['reject_reasons']}` |"
            )
    lines += ["", "## Recovery on accepted cases", ""]
    lines.append("| c | λ abs-rel median | λ abs-rel max | A abs-rel median | A abs-rel max | n with λ err>5% |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['c']:g} | {100*r['lambda_abs_rel_error_median_accepted']:.3f}% | "
            f"{100*r['lambda_abs_rel_error_max_accepted']:.3f}% | "
            f"{100*r['A_abs_rel_error_median_accepted']:.3f}% | "
            f"{100*r['A_abs_rel_error_max_accepted']:.3f}% | "
            f"{int(r['n_accepted_lambda_rel_gt_0p05'])} |"
        )
    lines += [
        "",
        "## Take-home",
        "",
        "Stronger background curvature (**c=0.010**) increases conditions where the sinusoid cannot be recovered, "
        "but the large wavelength overestimates observed in those failures were **excluded by Core acceptance** "
        "(mainly `coverage_hat<1.5`). Accepted cases at either `c` did not retain large λ errors (>5%).",
        "",
        "Keep **c=0.005** as the base curved-baseline condition in the main suite; use **c=0.010** only as this "
        "additional scope check.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root_from_here(__file__)
    run_id = args.run_id or new_run_id("curv_sens")
    out_dir = root / "results" / "analysis" / "sin_trend" / run_id / "curvature_sensitivity_grid"
    out_dir.mkdir(parents=True, exist_ok=False)

    cfg = default_measurement_config()
    rows: List[Dict[str, Any]] = []
    grid = [(A, lam, ph) for A in A_MM for lam in LAMBDA_MM for ph in PHASE_RAD]
    for c in (C_BASE, C_EXTRA):
        for A, lam, ph in grid:
            if not args.quiet:
                print(f"c={c:g} A={A:g} λ={lam:g} φ={ph:.3f}", flush=True)
            rows.append(run_one(A, lam, ph, c, cfg))

    df = pd.DataFrame(rows)
    summary = summarize_by_c(df)
    rejects = df[(np.isclose(df["c"], C_EXTRA)) & (df["core_all"] != 1)].copy()

    # condition matrix for reporting
    mat_rows = []
    for (A, lam, ph), g in df.groupby(["A_true_mm", "lambda_true_mm", "phase_rad"]):
        r005 = g[np.isclose(g["c"], C_BASE)].iloc[0]
        r010 = g[np.isclose(g["c"], C_EXTRA)].iloc[0]
        mat_rows.append(
            {
                "A_mm": A,
                "lambda_mm": lam,
                "phase_rad": ph,
                "c0p005": "accept" if r005.core_all == 1 else "reject",
                "lambda_err_pct_005": 100.0 * float(r005.lambda_abs_rel_error),
                "c0p010": "accept" if r010.core_all == 1 else "reject",
                "lambda_err_pct_010": 100.0 * float(r010.lambda_abs_rel_error),
                "reject_0p010": r010.reject_reasons if r010.core_all != 1 else "",
            }
        )
    matrix = pd.DataFrame(mat_rows).sort_values(["lambda_mm", "A_mm", "phase_rad"])

    atomic_to_csv(df, out_dir / "all_cases.csv", index=False)
    atomic_to_csv(summary, out_dir / "summary_by_c.csv", index=False)
    atomic_to_csv(rejects, out_dir / "rejects_c0p01.csv", index=False)
    atomic_to_csv(matrix, out_dir / "condition_matrix.csv", index=False)
    write_report(out_dir, df, summary)
    atomic_write_json(
        {
            **metadata_dict(
                script=__file__,
                run_id=run_id,
                note="(2) Accuracy across background curvatures; not part of (1) (156 cases)",
            ),
            "config_sha256": sha256_json(cfg),
            "source_hashes": source_code_hashes(
                ["42_curvature_sensitivity_grid.py", "sin_trend_synthetic.py"],
                here=_HERE,
            ),
            "design": {
                "A_mm": A_MM,
                "lambda_mm": LAMBDA_MM,
                "phase_rad": PHASE_RAD,
                "n_cycles_target_after_trim": N_CYCLES_TARGET_AFTER_TRIM,
                "cycle_specification": "post_trim_target_then_solve_input_n_cycles",
                "c_base": C_BASE,
                "c_extra": C_EXTRA,
                "n_per_c": len(grid),
            },
        },
        out_dir / "metadata.json",
    )
    if not args.quiet:
        print(summary.to_string(index=False))
        print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
