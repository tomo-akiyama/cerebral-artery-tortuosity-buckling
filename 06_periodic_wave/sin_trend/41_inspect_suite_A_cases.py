#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 41 inspect suite A cases."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sin_trend_classify import classify_row  # noqa: E402
from sin_trend_length_selection_validation import (  # noqa: E402
    cut_prefix_window_for_n_true,
    load_config,
    make_parent_centerline,
    measurement_config_from_validation_cfg,
    rng_for,
)
from sin_trend_measure import measure_interval  # noqa: E402
from sin_trend_paths import FRAME_BASELINE, project_root_from_here  # noqa: E402


                                                       
_OBS_COLOR = "C0"
_FIT_COLOR = "C3"
_RESP_COLOR = "C2"
_YLIM_FLOOR_MM = 1e-3


def load_validation_config_for_run(
    run_dir: Path, config_arg: Optional[Path] = None
) -> Dict[str, Any]:
    if config_arg is not None:
        return load_config(config_arg)
    used = Path(run_dir) / "config_used.json"
    if used.is_file():
        with used.open("r", encoding="utf-8") as f:
            return json.load(f)
    return load_config()


def measurement_config_for_inspect(vcfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(vcfg)
    sb = dict(cfg.get("suite_B") or {})
    sb.pop("n_true_match_atol", None)
    cfg["suite_B"] = sb
    return measurement_config_from_validation_cfg(cfg)


def _to_mm(arr: Any, voxel_size_mm: float) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return None
    return a * float(voxel_size_mm)


def _finite_abs_max(*arrays: Optional[np.ndarray]) -> float:
    m = 0.0
    any_finite = False
    for a in arrays:
        if a is None:
            continue
        x = np.asarray(a, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        any_finite = True
        m = max(m, float(np.max(np.abs(x))))
    if not any_finite:
        return float("nan")
    return m


def common_symmetric_ylim_mm(
    *arrays: Optional[np.ndarray], floor_mm: float = _YLIM_FLOOR_MM
) -> Tuple[float, float]:
    m = _finite_abs_max(*arrays)
    if not np.isfinite(m):
        m = float(floor_mm)
    m = max(float(m), float(floor_mm))
    pad = 1.08 * m
    return (-pad, pad)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive inspector for (3) Accuracy across cycle counts"
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "results/analysis/sin_trend/length_selection_validation/lengthsel_A_20261027"
        ),
        help="検証Aの出力ディレクトリ（all_cases.csv がある場所）",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--start", type=int, default=0, help="開始インデックス（フィルタ後）")
    p.add_argument("--case-id", default=None, help="特定 case_id から開始")
    p.add_argument("--only-underest", action="store_true", help="λ̂/λ_true < 0.9 のみ")
    p.add_argument("--only-accepted", action="store_true")
    p.add_argument("--only-rejected", action="store_true")
    p.add_argument(
        "--only-accepted-n-true-lt-1p5",
        action="store_true",
        help="真の周期数 N_true<1.5 かつ採択された例のみ",
    )
    p.add_argument("--only-exceptions", action="store_true")
    p.add_argument("--min-n-true-target", type=float, default=None)
    p.add_argument("--max-n-true-target", type=float, default=None)
    p.add_argument("--lambda-true", type=float, default=None, help="真波長フィルタ [mm]")
    p.add_argument("--background", default=None, choices=[None, "straight", "quadratic_c0p005"])
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="s キー保存先（省略時 run-dir/inspect_snapshots）",
    )
    return p.parse_args(argv)


def _finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def filter_cases(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    g = df.copy()
    if "suite" in g.columns:
        g = g[g["suite"].astype(str).eq("A")]
    if args.only_underest and "lambda_hat_over_true" in g.columns:
        r = pd.to_numeric(g["lambda_hat_over_true"], errors="coerce")
        g = g[r < 0.9]
    if args.only_accepted:
        g = g[pd.to_numeric(g.get("accepted_core_all"), errors="coerce").fillna(0).astype(int) == 1]
    if args.only_rejected:
        g = g[pd.to_numeric(g.get("accepted_core_all"), errors="coerce").fillna(0).astype(int) != 1]
    if args.only_accepted_n_true_lt_1p5:
        col = "accepted_N_true_lt_1p5"
        if col not in g.columns and "false_accept_N_true_lt_1p5" in g.columns:
            raise KeyError(
                "旧列 false_accept_N_true_lt_1p5 のみ存在します。"
                "accepted_N_true_lt_1p5 付きの結果を再生成してください。"
            )
        g = g[pd.to_numeric(g.get(col), errors="coerce").fillna(0).astype(int) == 1]
    if args.only_exceptions:
        g = g[g["execution_status"].astype(str).eq("exception")]
    if args.min_n_true_target is not None:
        g = g[pd.to_numeric(g["n_true_target"], errors="coerce") >= float(args.min_n_true_target)]
    if args.max_n_true_target is not None:
        g = g[pd.to_numeric(g["n_true_target"], errors="coerce") <= float(args.max_n_true_target)]
    if args.lambda_true is not None:
        g = g[np.isclose(pd.to_numeric(g["lambda_true_mm"], errors="coerce"), float(args.lambda_true))]
    if args.background:
        g = g[g["background_name"].astype(str).eq(str(args.background))]
    return g.reset_index(drop=True)


def rebuild_and_measure(row: pd.Series, vcfg: Dict[str, Any], meas_cfg: Dict[str, Any]) -> Dict[str, Any]:
    removal = int(vcfg.get("removal_points", 10))
    ds = float(vcfg.get("target_ds_mm", 0.1))
    vs = float(vcfg.get("voxel_size_mm", 0.5))
    a_cfg = vcfg.get("suite_A") or {}
    max_n = float(max(a_cfg.get("n_true_targets", [3.0])))
    extra = float(a_cfg.get("parent_extra_cycles", 0.75))
    pad = float(a_cfg.get("window_start_pad_cycles", 0.05))
    atol = float(a_cfg.get("n_true_match_atol", 0.02))

    bg_name = str(row.get("background_name", "straight"))
    curv = float(row.get("background_curvature_mm", 0.0) or 0.0)
    bg = None
    if bg_name not in ("", "straight", "nan", "None") and curv != 0.0:
        bg = {"type": "quadratic", "curvature_mm": curv}
    elif bg_name.startswith("quadratic"):
        bg = {"type": "quadratic", "curvature_mm": curv if curv else 0.005}

    parent_id = str(row["parent_curve_id"])
    seed = int(row.get("seed", 0))
    parent = make_parent_centerline(
        amplitude_mm=float(row["A_true_mm"]),
        lambda_true_mm=float(row["lambda_true_mm"]),
        phase_rad=float(row["phase_rad"]),
        diameter_mm=float(row["D_mm"]),
        target_ds_mm=ds,
        removal=removal,
        max_n_true=max_n,
        parent_extra_cycles=extra,
        background=bg,
        noise_sd_mm=float(row.get("noise_sd_mm", 0.0) or 0.0),
        seed=int(rng_for(seed, parent_id).integers(0, 2**31 - 1)),
        voxel_size_mm=vs,
        length_atol_mm=float(a_cfg.get("parent_length_atol_mm", 0.05)),
    )
    win = cut_prefix_window_for_n_true(
        parent,
        n_true_target=float(row["n_true_target"]),
        removal=removal,
        start_pad_cycles=pad,
        atol=atol,
    )
    out: Dict[str, Any] = {
        "ok": bool(win.get("ok")),
        "reason": win.get("reason", ""),
        "parent": parent,
        "win": win,
        "voxel_size": vs,
        "removal": removal,
        "res": None,
        "row_meas": None,
    }
    if not win.get("ok"):
        return out
    res = measure_interval(
        win["P_vox"],
        frame_reference=FRAME_BASELINE,
        cfg=meas_cfg,
        radius_vox=win.get("radius_vox"),
        ids={"directory_id": f"INSPECT_{row['case_id']}", "segment_id": 0, "segment_index": 0},
        verbose=False,
        compute_pca=False,
    )
    out["res"] = res
    out["row_meas"] = classify_row(res["row"])
    return out


def _decision_banner(csv_row: pd.Series, built: Dict[str, Any]) -> Tuple[str, str]:
    rm = built.get("row_meas") or {}
    if not rm:
        return ("NO FIT / FAILED", "0.35")
    acc = int(rm.get("core_all", csv_row.get("accepted_core_all", 0)) or 0)
    n_h = rm.get("coverage_hat", csv_row.get("N_hat"))
    reject = str(rm.get("reject_reasons", "") or "").strip()
    n_txt = f"{float(n_h):.4g}" if _finite(n_h) else "nan"
    if acc == 1:
        return (f"ACCEPTED  |  estimated cycle count N_hat={n_txt} ≥ 1.5", "#1b7f3a")
    if "coverage_hat" in reject:
        return (f"Not qualified: estimated cycle count {n_txt} < 1.5", "#b00020")
    if "partial_R2" in reject:
        p = rm.get("partial_R2", csv_row.get("partial_R2"))
        p_txt = f"{float(p):.3f}" if _finite(p) else "nan"
        return (f"Not qualified: partial R² {p_txt} < 0.50", "#b00020")
    if "boundary_exact" in reject:
        return (f"Not qualified: estimate on search boundary ({reject})", "#b00020")
    if "not_fit_valid" in reject or "not_interval_eligible" in reject:
        return (f"Not qualified: {reject}", "#b00020")
    reason = reject if reject else "not core_all"
    return (f"Not qualified: {reason}  (N_hat={n_txt})", "#b00020")


def _info_text(csv_row: pd.Series, built: Dict[str, Any]) -> str:
    rm = built.get("row_meas") or {}
    win = built.get("win") or {}
    lam_t = float(csv_row["lambda_true_mm"])
    n_t = float(win.get("n_true_achieved", csv_row.get("N_true", np.nan)))
    lam_h = rm.get("lambda_hat_mm", csv_row.get("lambda_hat_mm"))
    n_h = rm.get("coverage_hat", csv_row.get("N_hat"))
    lines = [
        f"case {csv_row.get('case_id')}",
        f"λ_true={lam_t:g} mm   A_true={float(csv_row['A_true_mm']):g} mm   φ={float(csv_row['phase_rad']):.3f}",
        f"bg={csv_row.get('background_name')}  D={float(csv_row['D_mm']):g} mm",
        f"target N_true={float(csv_row['n_true_target']):g}  achieved N_true={n_t:.4f}",
        f"L0_true_trim={float(win.get('L0_true_trimmed_mm', csv_row.get('L0_true_trimmed_mm', np.nan))):.3f} mm",
    ]
    if rm:
        L_base = float(rm.get("L_base_mm", np.nan))
        if _finite(lam_h) and lam_t > 0:
            rel = abs(float(lam_h) - lam_t) / lam_t
            lines.append(
                f"λ_hat={float(lam_h):.4g} mm   |λ_hat−λ_true|/λ_true={rel:.4f}   λ_hat/λ_true={float(lam_h)/lam_t:.4f}"
            )
        else:
            lines.append("λ_hat=nan")
        A_t = float(csv_row.get("A_true_mm", float("nan")))
        A_h = rm.get("amp_comb_mm", csv_row.get("amp_comb_mm"))
        if _finite(A_t) and _finite(A_h) and A_t > 0:
            lines.append(f"A_hat={float(A_h):.4g} mm   A_hat/A_true={float(A_h)/A_t:.4f}")
        if _finite(n_h):
            gate = "≥ 1.5  (gate OK)" if float(n_h) >= 1.5 - 1e-12 else "< 1.5  (fails gate)"
            lines.append(
                f"N_hat = L_base/λ_hat = {float(n_h):.4f}  {gate}   L_base={L_base:.3f} mm"
            )
        else:
            lines.append("N_hat=nan")
        lines += [
            f"partial_R2={float(rm.get('partial_R2', np.nan)):.4f}  (gate ≥0.50)   "
            f"R2={float(rm.get('R2', np.nan)):.4f}  (diagnostic only; not a gate)",
            f"reject_reasons={rm.get('reject_reasons') or '(none)'}",
            f"search λ=[{rm.get('lambda_min_search_mm')}, {rm.get('lambda_max_search_mm')}]",
            "Top-right: generated wave on true ξ + trim. Bottom: fit on estimated baseline s0.",
            "response = periodic part after quadratic residualization (not raw sinusoid).",
        ]
    else:
        lines.append(f"measure skipped/failed: {built.get('reason') or csv_row.get('execution_failure_reason')}")
    return "\n".join(lines)


def _add_cycle_ticks_from_lambda_hat(ax, s_mm: np.ndarray, lambda_hat_mm: float) -> None:
    s0 = float(s_mm[0])
    s1 = float(s_mm[-1])
    lam = float(lambda_hat_mm)
    if not (np.isfinite(lam) and lam > 0 and np.isfinite(s0) and np.isfinite(s1) and s1 > s0):
        return
    n_max = int(np.floor((s1 - s0) / lam + 1e-12))
    ticks = [s0 + i * lam for i in range(0, n_max + 1) if s0 + i * lam <= s1 + 1e-9]
    if len(ticks) < 2:
        ticks = [s0, s1]
        labels = ["0", f"{(s1 - s0) / lam:.2f}"]
    else:
        labels = [str(i) for i in range(len(ticks))]
    ax2 = ax.secondary_xaxis(
        "top",
        functions=(lambda s: (np.asarray(s, float) - s0) / lam, lambda c: s0 + np.asarray(c, float) * lam),
    )
    ax2.set_xticks([(t - s0) / lam for t in ticks])
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("cycles along estimated baseline: (s0−s0_start)/λ_hat", fontsize=8)


def _plot_uv_panel(
    ax,
    *,
    s_mm: Optional[np.ndarray],
    u_mm: Optional[np.ndarray],
    fit_mm: Optional[np.ndarray],
    response_mm: Optional[np.ndarray],
    ylim: Tuple[float, float],
    title: str,
    ylabel: str,
    show_legend: bool,
    lambda_hat_mm: Optional[float] = None,
) -> None:
    ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
    plotted = False
    if s_mm is not None and u_mm is not None and len(s_mm) == len(u_mm):
        ax.plot(s_mm, u_mm, color=_OBS_COLOR, lw=1.6, label="observed residual u (after quadratic)", zorder=2)
        plotted = True
    if s_mm is not None and fit_mm is not None and len(s_mm) == len(fit_mm):
        ax.plot(s_mm, fit_mm, color=_FIT_COLOR, lw=1.8, label="fit = trend + response", zorder=3)
        plotted = True
    if s_mm is not None and response_mm is not None and len(s_mm) == len(response_mm):
        ax.plot(
            s_mm,
            response_mm,
            color=_RESP_COLOR,
            ls="--",
            lw=1.7,
            label="response (periodic; after residualization)",
            zorder=4,
        )
        plotted = True
    if (
        s_mm is not None
        and len(s_mm) >= 2
        and lambda_hat_mm is not None
        and _finite(lambda_hat_mm)
        and float(lambda_hat_mm) > 0
    ):
        _add_cycle_ticks_from_lambda_hat(ax, np.asarray(s_mm, float), float(lambda_hat_mm))
        s0 = float(s_mm[0])
        s_one = s0 + float(lambda_hat_mm)
        if s_one <= float(s_mm[-1]) + 1e-9:
            ax.axvline(s_one, color="#ff7f0e", ls=":", lw=1.0, alpha=0.85, label="1×λ_hat on s0")
    if not plotted:
        ax.text(0.5, 0.5, "no U/V series", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylim(*ylim)
    ax.set_xlabel("estimated baseline arclength s0 [mm]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if show_legend and plotted:
        ax.legend(fontsize=6.5, loc="best")


def draw_case(fig, csv_row: pd.Series, built: Dict[str, Any], *, index: int, n_total: int) -> None:
    fig.clear()
    vs = float(built["voxel_size"])
    rem = int(built["removal"])
    parent = built["parent"]
    win = built["win"]
    res = built.get("res")

    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.2, 0.95],
        hspace=0.42,
        wspace=0.28,
    )
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_lat = fig.add_subplot(gs[0, 1])
    ax_U = fig.add_subplot(gs[1, 0])
    ax_V = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")

    # --- XY: parent faint + window ---
    Pp = np.asarray(parent["P_vox"], float) * vs
    Bp = np.asarray(parent["B_vox"], float) * vs if parent.get("B_vox") is not None else None
    ax_xy.plot(Pp[:, 0], Pp[:, 1], color="0.75", lw=1.0, label="parent P")
    if Bp is not None:
        ax_xy.plot(Bp[:, 0], Bp[:, 1], color="0.55", ls="--", lw=0.9, label="known B (parent)")
    if win.get("ok"):
        Pw = np.asarray(win["P_vox"], float) * vs
        ax_xy.plot(Pw[:, 0], Pw[:, 1], color="C0", lw=2.0, label="window P")
        if res is not None and res.get("baseline") and res["baseline"].get("P0") is not None:
            P0 = np.asarray(res["baseline"]["P0"], float) * vs
            ax_xy.plot(P0[:, 0], P0[:, 1], color="C3", lw=1.6, label="estimated P0")
            pw = res.get("pointwise") or {}
            if pw.get("P0") is not None and pw.get("d_response") is not None:
                P0t = np.asarray(pw["P0"], float) * vs
                d = np.asarray(pw["d_response"], float) * vs
                Pref = P0t + d
                ax_xy.plot(
                    Pref[:, 0],
                    Pref[:, 1],
                    color="C2",
                    ls=":",
                    lw=1.8,
                    label="P0 + response (not raw sinusoid)",
                )
                                                                     
                if len(P0t) >= 2:
                    tt = np.linspace(0.0, 1.0, len(P0t))
                    chord = (1.0 - tt)[:, None] * P0t[0][None, :] + tt[:, None] * P0t[-1][None, :]
                    Presp = chord + d
                    ax_xy.plot(
                        Presp[:, 0],
                        Presp[:, 1],
                        color="#2ca02c",
                        ls="-",
                        lw=2.2,
                        alpha=0.95,
                        zorder=6,
                        label="response shape on straight ref",
                    )
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("x [mm]")
    ax_xy.set_ylabel("y [mm]")
    ax_xy.set_title("XY centerlines")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=7, loc="best")

    # --- lateral vs true ξ: generated waveform + trim only (no λ_hat guide on ξ) ---
    rm = built.get("row_meas") or {}
    lam_t = float(csv_row["lambda_true_mm"])
    lam_h = rm.get("lambda_hat_mm", csv_row.get("lambda_hat_mm"))
    A = float(csv_row["A_true_mm"])
    n_h = rm.get("coverage_hat", csv_row.get("N_hat", csv_row.get("coverage_hat")))
    if win.get("ok"):
        xi = np.asarray(win["xi_mm"], float)
        yw = np.asarray(win["P_vox"], float)[:, 1] * vs
        ax_lat.plot(xi, yw, color="C0", lw=1.5, label="P_y (observed, generated)")
        if win.get("wave_phase_rad") is not None:
            ph = np.asarray(win["wave_phase_rad"], float)
            ax_lat.plot(
                xi,
                A * np.sin(ph),
                color="#d62728",
                ls="--",
                lw=2.2,
                alpha=0.95,
                zorder=5,
                label="true A·sin(phase on ξ)",
            )
        ax_lat.axvspan(xi[0], xi[min(rem, len(xi) - 1)], color="0.85", alpha=0.5, label="trim ends")
        ax_lat.axvspan(xi[max(0, len(xi) - rem - 1)], xi[-1], color="0.85", alpha=0.5)
                                       
        if _finite(lam_h) and lam_t > 0:
            rel = abs(float(lam_h) - lam_t) / lam_t
            n_line = (
                f"N_hat={float(n_h):.4g}  (gate 1.5)"
                if _finite(n_h)
                else "N_hat=nan"
            )
            box = (
                f"λ_true={lam_t:g} mm\n"
                f"λ_hat={float(lam_h):.4g} mm\n"
                f"|Δλ|/λ_true={rel:.4f}\n"
                f"{n_line}"
            )
        else:
            box = f"λ_true={lam_t:g} mm\nλ_hat=nan"
        ax_lat.text(
            0.02,
            0.98,
            box,
            transform=ax_lat.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.6", alpha=0.92),
            zorder=10,
        )
        ax_lat.set_xlabel("ξ true generation-baseline arclength [mm]")
        ax_lat.set_ylabel("y [mm]")
        ax_lat.set_title("Generated waveform + trim (ξ); λ compare as numbers")
        ax_lat.grid(True, alpha=0.3)
        ax_lat.legend(fontsize=6.5, loc="lower right")
    else:
        ax_lat.text(0.5, 0.5, "window failed", ha="center", va="center", transform=ax_lat.transAxes)

    # --- U / V: observed, fit, and response (same scale; no twin axis) ---
    pw = (res.get("pointwise") or {}) if res is not None else {}
    s_mm = _to_mm(pw.get("s_fit", pw.get("s_phase_trim")), vs)
    uU_mm = _to_mm(pw.get("uU"), vs)
    uV_mm = _to_mm(pw.get("uV"), vs)
    fitU_mm = _to_mm(pw.get("fit_U"), vs)
    fitV_mm = _to_mm(pw.get("fit_V"), vs)
    respU_mm = _to_mm(pw.get("response_U"), vs)
    respV_mm = _to_mm(pw.get("response_V"), vs)
                               
    n_s = len(s_mm) if s_mm is not None else 0

    def _match_len(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if a is None or n_s == 0 or len(a) != n_s:
            return None
        return a

    uU_mm = _match_len(uU_mm)
    uV_mm = _match_len(uV_mm)
    fitU_mm = _match_len(fitU_mm)
    fitV_mm = _match_len(fitV_mm)
    respU_mm = _match_len(respU_mm)
    respV_mm = _match_len(respV_mm)
    ylim = common_symmetric_ylim_mm(uU_mm, fitU_mm, respU_mm, uV_mm, fitV_mm, respV_mm)
    lam_h_f = float(lam_h) if _finite(lam_h) else None
    _plot_uv_panel(
        ax_U,
        s_mm=s_mm,
        u_mm=uU_mm,
        fit_mm=fitU_mm,
        response_mm=respU_mm,
        ylim=ylim,
        title="U on estimated baseline s0: residual / fit / response",
        ylabel="U displacement [mm]",
        show_legend=True,
        lambda_hat_mm=lam_h_f,
    )
    _plot_uv_panel(
        ax_V,
        s_mm=s_mm,
        u_mm=uV_mm,
        fit_mm=fitV_mm,
        response_mm=respV_mm,
        ylim=ylim,
        title="V on estimated baseline s0: residual / fit / response",
        ylabel="V displacement [mm]",
        show_legend=True,
        lambda_hat_mm=lam_h_f,
    )

    banner, banner_color = _decision_banner(csv_row, built)
    ax_txt.text(
        0.01,
        0.98,
        f"[{index + 1}/{n_total}]  {banner}",
        va="top",
        ha="left",
        family="monospace",
        fontsize=12,
        fontweight="bold",
        color=banner_color,
        transform=ax_txt.transAxes,
    )
    ax_txt.text(
        0.01,
        0.82,
        _info_text(csv_row, built),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.2,
        transform=ax_txt.transAxes,
    )
                               
    fig.text(
        0.07,
        0.008,
        "keys: n/→ next | p/← prev | s save | q quit",
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    fig.suptitle("(3) Accuracy across cycle counts — inspector (U/V)", fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.055)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root_from_here(__file__)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    csv_path = run_dir / "all_cases.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing {csv_path}")

    vcfg = load_validation_config_for_run(run_dir, args.config)
    meas_cfg = measurement_config_for_inspect(vcfg)
    df = filter_cases(pd.read_csv(csv_path), args)
    if len(df) == 0:
        print("no cases after filter")
        return 1

    idx = int(args.start)
    if args.case_id:
        hits = np.where(df["case_id"].astype(str).eq(str(args.case_id)))[0]
        if len(hits) == 0:
            raise SystemExit(f"case_id not found after filter: {args.case_id}")
        idx = int(hits[0])
    idx = max(0, min(idx, len(df) - 1))

    save_dir = Path(args.save_dir) if args.save_dir else (run_dir / "inspect_snapshots")
    save_dir.mkdir(parents=True, exist_ok=True)

                           
    import matplotlib

                                     
    try:
        matplotlib.use("MacOSX", force=False)
    except Exception:
        try:
            matplotlib.use("TkAgg", force=False)
        except Exception:
            pass
    import matplotlib.pyplot as plt

    state = {"idx": idx, "built": None, "quit": False}

    def load_current() -> None:
        row = df.iloc[state["idx"]]
        print(
            f"[{state['idx'] + 1}/{len(df)}] case_id={row['case_id']} "
            f"N_target={row['n_true_target']} λ={row['lambda_true_mm']} ..."
        )
        try:
            state["built"] = rebuild_and_measure(row, vcfg, meas_cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"  rebuild failed: {exc}")
            state["built"] = {
                "ok": False,
                "reason": str(exc),
                "parent": None,
                "win": {"ok": False},
                "voxel_size": float(vcfg.get("voxel_size_mm", 0.5)),
                "removal": int(vcfg.get("removal_points", 10)),
                "res": None,
                "row_meas": None,
            }

    def refresh() -> None:
        load_current()
        built = state["built"]
        row = df.iloc[state["idx"]]
        if built.get("parent") is None:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"[{state['idx'] + 1}/{len(df)}]\nrebuild failed\n{built.get('reason')}\n\n"
                f"csv execution_status={row.get('execution_status')}\n"
                f"{row.get('execution_failure_reason')}",
                ha="center",
                va="center",
                family="monospace",
                fontsize=10,
                wrap=True,
            )
        else:
            draw_case(fig, row, built, index=state["idx"], n_total=len(df))
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            state["quit"] = True
            plt.close(fig)
            return
        if key in ("n", "right", " ", "enter"):
            state["idx"] = min(state["idx"] + 1, len(df) - 1)
            refresh()
            return
        if key in ("p", "left", "backspace"):
            state["idx"] = max(state["idx"] - 1, 0)
            refresh()
            return
        if key == "s":
            row = df.iloc[state["idx"]]
            out = save_dir / f"{state['idx']:04d}_{row['case_id']}.png"
            fig.savefig(out, dpi=140)
            print(f"saved {out}")

    fig = plt.figure(figsize=(13.5, 11.6))
    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    print(
        f"Loaded {len(df)} cases from {csv_path}\n"
        "Focus the figure window, then use n/p/s/q."
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
