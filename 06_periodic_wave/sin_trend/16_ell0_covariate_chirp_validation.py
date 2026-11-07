#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: 16 ell0 covariate chirp validation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)

from sin_trend_paths import (  # noqa: E402
    atomic_to_csv,
    atomic_write_json,
    new_run_id,
    project_root_from_here,
)
from sin_trend_synthetic import make_sinusoid_centerline  # noqa: E402
from sin_trend_length_selection_validation import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    L0_true_trimmed_mm,
    N_true_from_xi,
    assign_lambda_true,
    background_dict,
    empty_result_row,
    enrichment_from_measure,
    load_clinical_DL_pool,
    load_config,
    measure_and_classify_centerline,
    measurement_config_from_validation_cfg,
    n_cycles_input_for_target_remaining_after_trim,
    rng_for,
    sample_DL,
    stable_case_id,
    _finite_pos,
)
from importlib import import_module  # noqa: E402

# reuse OLS / MC summary from (5)
_ell0 = import_module("15_ell0_covariate_scaling_validation")

K_LAMBDA_OVER_D = 17.4
CLINICAL_N_MED = 1.6875
CLINICAL_N_Q1 = 1.5625
CLINICAL_N_Q3 = 1.875
DEFAULT_SEED = 20261030


def generate_population_centerline_chirp(
    *,
    D_mm: float,
    L0_true_trimmed_target_mm: float,
    lambda_true_mm: float,
    amplitude_mm: float,
    phase_rad: float,
    target_ds_mm: float,
    removal: int,
    voxel_size_mm: float,
    background: Optional[Dict[str, Any]],
    noise_sd_mm: float,
    seed: int,
    chirp_alpha: float,
    phase_quad_cycles: float,
    length_atol_mm: float = 0.05,
    length_rtol: float = 0.0,
) -> Dict[str, Any]:
    if not (_finite_pos(D_mm) and _finite_pos(L0_true_trimmed_target_mm) and _finite_pos(lambda_true_mm)):
        return {"ok": False, "reason": "nonpositive_D_L_or_lambda"}
    n_target = float(L0_true_trimmed_target_mm) / float(lambda_true_mm)
    length_tol = float(length_atol_mm) + float(length_rtol) * float(L0_true_trimmed_target_mm)
    atol_cycles = length_tol / float(lambda_true_mm)
    try:
        # n_cycles solve uses un-chirped geometry (same L0 design); chirp is applied at synthesize.
        n_in = n_cycles_input_for_target_remaining_after_trim(
            target_remaining=n_target,
            amplitude_mm=float(amplitude_mm),
            lambda_x_mm=float(lambda_true_mm),
            phase=float(phase_rad),
            target_ds_mm=float(target_ds_mm),
            removal_points=int(removal),
            voxel_size_mm=float(voxel_size_mm),
            background=background,
            diameter_mm=float(D_mm),
            atol=float(atol_cycles),
        )
    except ValueError as exc:
        return {"ok": False, "reason": f"n_cycles_solve_failed:{exc}", "n_true_target": n_target}
    syn = make_sinusoid_centerline(
        amplitude_mm=float(amplitude_mm),
        lambda_x_mm=float(lambda_true_mm),
        n_cycles=float(n_in),
        phase=float(phase_rad),
        voxel_size_mm=float(voxel_size_mm),
        target_ds_mm=float(target_ds_mm),
        background=background,
        noise_sd_mm=float(noise_sd_mm),
        seed=int(seed),
        diameter_mm=float(D_mm),
        chirp_alpha=float(chirp_alpha),
        phase_quad_cycles=float(phase_quad_cycles),
    )
    n_true = N_true_from_xi(syn["xi_mm"], syn["lambda_baseline_true_mm"], removal)
    L0_trim = L0_true_trimmed_mm(syn["xi_mm"], removal)
    length_error = float(L0_trim - L0_true_trimmed_target_mm)
    ok = bool(np.isfinite(L0_trim) and abs(length_error) <= length_tol)
    return {
        "ok": ok,
        "reason": "" if ok else f"length_unmet:|err|={abs(length_error):.3e}",
        "syn": syn,
        "n_true_target": n_target,
        "N_true": float(n_true),
        "L0_true_trimmed_mm": float(L0_trim),
        "n_cycles_input": float(n_in),
        "length_error_mm": length_error,
        "length_abs_rel_error": abs(length_error) / float(L0_true_trimmed_target_mm),
        "length_tolerance_mm": length_tol,
        "chirp_alpha": float(chirp_alpha),
        "phase_quad_cycles": float(phase_quad_cycles),
    }


def run_chirp_job(job: Dict[str, Any], meas_cfg: Dict[str, Any], vcfg: Dict[str, Any]) -> Dict[str, Any]:
    removal = int(vcfg.get("removal_points", 10))
    ds = float(vcfg.get("target_ds_mm", 0.1))
    vs = float(vcfg.get("voxel_size_mm", 0.5))
    bg = background_dict(job["background"])
    base = {
        **empty_result_row(),
        "case_id": job["case_id"],
        "parent_curve_id": job["case_id"],
        "suite": "B",
        "scenario": job["scenario"],
        "lambda_kind": job.get("lambda_kind"),
        "replicate": job["replicate"],
        "seed": job["seed"],
        "unit_index": job["unit_index"],
        "dl_pairing": job["dl_pairing"],
        "D_mm": job["diameter_mm"],
        "A_true_mm": job["amplitude_mm"],
        "lambda_true_mm": job["lambda_true_mm"],
        "phase_rad": job["phase_rad"],
        "background_name": job["background"].get("name"),
        "background_curvature_mm": float(job["background"].get("curvature_mm", 0.0) or 0.0),
        "noise_sd_mm": job["noise_sd_mm"],
        "L0_proxy_from_clinical_L_base_mm": job["L0_proxy_mm"],
        "L0_target_mm": job["L0_proxy_mm"],
        "target_ds_mm": ds,
        "source_pool_index": job["source_pool_index"],
        "chirp_alpha_mag": job.get("chirp_alpha_mag"),
        "phase_quad_mag": job.get("phase_quad_mag"),
        "chirp_alpha": job.get("chirp_alpha"),
        "phase_quad_cycles": job.get("phase_quad_cycles"),
        "epsilon_sd": job.get("epsilon_sd"),
        "part": job.get("part", "5_chirp"),
    }
    try:
        gen = generate_population_centerline_chirp(
            D_mm=job["diameter_mm"],
            L0_true_trimmed_target_mm=job["L0_proxy_mm"],
            lambda_true_mm=job["lambda_true_mm"],
            amplitude_mm=job["amplitude_mm"],
            phase_rad=job["phase_rad"],
            target_ds_mm=ds,
            removal=removal,
            voxel_size_mm=vs,
            background=bg,
            noise_sd_mm=job["noise_sd_mm"],
            seed=int(rng_for(job["seed"], job["case_id"], "gen").integers(0, 2**31 - 1)),
            chirp_alpha=float(job["chirp_alpha"]),
            phase_quad_cycles=float(job["phase_quad_cycles"]),
            length_atol_mm=job["length_atol_mm"],
            length_rtol=job["length_rtol"],
        )
        if not gen["ok"]:
            base.update(
                {
                    "generation_status": "failed",
                    "generation_failure_reason": gen.get("reason", "gen_failed"),
                    "N_true": gen.get("N_true", float("nan")),
                    "accepted_core_all": 0,
                    "core_all": 0,
                }
            )
            return base
        syn = gen["syn"]
        base.update(
            {
                "generation_status": "ok",
                "N_true": float(gen["N_true"]),
                "n_true_target": gen["n_true_target"],
                "L0_true_trimmed_mm": float(gen["L0_true_trimmed_mm"]),
                "n_cycles_input": gen["n_cycles_input"],
                "chirp_alpha": gen["chirp_alpha"],
                "phase_quad_cycles": gen["phase_quad_cycles"],
            }
        )
        meas = measure_and_classify_centerline(
            syn["P_vox"],
            cfg=meas_cfg,
            radius_vox=syn.get("radius_vox"),
            ids={"directory_id": f"LS_chirp_{job['case_id']}", "segment_id": 0, "segment_index": 0},
        )
        base.update(meas)
        base["execution_status"] = "ok"
        base.update(
            enrichment_from_measure(
                lambda_true_mm=job["lambda_true_mm"],
                L0_true_trimmed_mm=float(gen["L0_true_trimmed_mm"]),
                N_true=float(gen["N_true"]),
                meas=meas,
                A_true_mm=float(job["amplitude_mm"]),
            )
        )
        return base
    except Exception as exc:  # noqa: BLE001
        base["generation_status"] = "failed"
        base["generation_failure_reason"] = f"{type(exc).__name__}: {exc}"
        base["accepted_core_all"] = 0
        base["core_all"] = 0
        return base


def _worker(payload):
    job, meas_cfg, vcfg = payload
    return run_chirp_job(job, meas_cfg, vcfg)


def run_jobs(jobs, *, meas_cfg, vcfg, n_workers, quiet) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not jobs:
        return results
    if n_workers <= 1:
        for i, job in enumerate(jobs, 1):
            results.append(run_chirp_job(job, meas_cfg, vcfg))
            if not quiet and (i % 50 == 0 or i == len(jobs)):
                print(f"  {i}/{len(jobs)}", flush=True)
        return results
    it = iter(jobs)
    ex = ProcessPoolExecutor(max_workers=n_workers)
    futures = {}

    def submit_next():
        job = next(it, None)
        if job is not None:
            futures[ex.submit(_worker, (job, meas_cfg, vcfg))] = job["case_id"]

    try:
        for _ in range(2 * n_workers):
            submit_next()
        count = 0
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                cid = futures.pop(fut)
                row = fut.result()
                if row.get("case_id") != cid:
                    raise RuntimeError("case_id mismatch")
                results.append(row)
                count += 1
                submit_next()
                if not quiet and (count % 50 == 0 or count == len(jobs)):
                    print(f"  {count}/{len(jobs)}", flush=True)
    finally:
        for fut in futures:
            fut.cancel()
        ex.shutdown(wait=True, cancel_futures=True)
    return results


def build_jobs(
    *,
    seed: int,
    n_replicates: int,
    n_per_replicate: int,
    pool,
    vcfg: Dict[str, Any],
    amplitude_mm: float,
    pairings: Sequence[str],
    epsilon_sd: float,
    chirp_alpha_mag: float,
    phase_quad_mag: float,
    part_tag: str,
) -> List[Dict[str, Any]]:
    b = vcfg["suite_B"]
    bg = {"name": "straight", "type": None, "curvature_mm": 0.0}
    sc = {
        "name": f"proportional_sigma{epsilon_sd:.4f}".replace(".", "p"),
        "kind": "proportional",
        "k_lambda_over_D": K_LAMBDA_OVER_D,
        "epsilon_sd": float(epsilon_sd),
    }
    jobs: List[Dict[str, Any]] = []
    for replicate in range(int(n_replicates)):
        for pairing in pairings:
            rng_dl = rng_for(seed, part_tag, "DL", replicate, pairing, amplitude_mm)
            D, L, src_idx = sample_DL(pool, n=n_per_replicate, pairing=str(pairing), rng=rng_dl)
            rng_phase = rng_for(seed, part_tag, "phase", replicate, pairing, amplitude_mm)
            phases = rng_phase.uniform(0.0, 2.0 * np.pi, size=n_per_replicate)
            rng_l = rng_for(seed, part_tag, "lam", replicate, pairing, amplitude_mm, sc["name"])
            lam, meta = assign_lambda_true(D, scenario=sc, rng=rng_l)
            rng_drift = rng_for(seed, part_tag, "drift", replicate, pairing, amplitude_mm)
            # random signs for slow drift
            signs_a = rng_drift.choice(np.array([-1.0, 1.0]), size=n_per_replicate)
            signs_b = rng_drift.choice(np.array([-1.0, 1.0]), size=n_per_replicate)
            for i in range(n_per_replicate):
                ca = float(chirp_alpha_mag) * float(signs_a[i])
                pb = float(phase_quad_mag) * float(signs_b[i])
                case_id = stable_case_id(
                    part_tag,
                    seed,
                    replicate,
                    pairing,
                    amplitude_mm,
                    bg["name"],
                    sc["name"],
                    i,
                    float(D[i]),
                    float(L[i]),
                    float(lam[i]),
                    float(phases[i]),
                    float(chirp_alpha_mag),
                    float(phase_quad_mag),
                )
                jobs.append(
                    {
                        "case_id": case_id,
                        "scenario": sc["name"],
                        "lambda_kind": "proportional",
                        "replicate": int(replicate),
                        "seed": int(seed),
                        "unit_index": int(i),
                        "dl_pairing": str(pairing),
                        "diameter_mm": float(D[i]),
                        "L0_proxy_mm": float(L[i]),
                        "lambda_true_mm": float(lam[i]),
                        "amplitude_mm": float(amplitude_mm),
                        "phase_rad": float(phases[i]),
                        "background": bg,
                        "noise_sd_mm": 0.0,
                        "source_pool_index": int(src_idx[i]),
                        "length_atol_mm": float(b.get("length_atol_mm", 0.05)),
                        "length_rtol": float(b.get("length_rtol", 0.0)),
                        "chirp_alpha_mag": float(chirp_alpha_mag),
                        "phase_quad_mag": float(phase_quad_mag),
                        "chirp_alpha": ca,
                        "phase_quad_cycles": pb,
                        "epsilon_sd": float(epsilon_sd),
                        "part": part_tag,
                        "gen_slope_loglog": meta.get("gen_slope_loglog"),
                        "gen_corr_loglog": meta.get("gen_corr_loglog"),
                        "gen_lambda_over_D_median": meta.get("gen_lambda_over_D_median"),
                    }
                )
    return jobs


def accepted_N_stats(df: pd.DataFrame) -> Dict[str, float]:
    work = df
    if "generation_status" in df.columns:
        work = df.loc[df["generation_status"].astype(str).eq("ok")].copy()
    n_all = max(len(df), 1)
    core = pd.to_numeric(work.get("core_all"), errors="coerce").fillna(0).astype(int) == 1
    acc = work.loc[core]
    N = pd.to_numeric(acc.get("coverage_hat"), errors="coerce").to_numpy(dtype=float)
    N = N[np.isfinite(N) & (N > 0)]
    if len(N) == 0:
        return {
            "n_accepted": 0,
            "acceptance_rate": float("nan"),
            "N_median": float("nan"),
            "N_q1": float("nan"),
            "N_q3": float("nan"),
            "score": float("inf"),
        }
    q1, med, q3 = np.percentile(N, [25, 50, 75])
    score = abs(med - CLINICAL_N_MED) + abs(q1 - CLINICAL_N_Q1) + abs(q3 - CLINICAL_N_Q3)
    return {
        "n_accepted": int(len(N)),
        "acceptance_rate": float(len(N) / n_all),
        "N_median": float(med),
        "N_q1": float(q1),
        "N_q3": float(q3),
        "score": float(score),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-replicates", type=int, default=20)
    p.add_argument("--n-per-replicate", type=int, default=500)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--amplitude-mm", type=float, default=1.0)
    p.add_argument("--tune-only", action="store_true")
    p.add_argument("--tune-n-per", type=int, default=300, help="curves per chirp grid cell (paired)")
    p.add_argument("--chirp-alpha", type=float, default=None, help="skip tune; fix |α|")
    p.add_argument("--phase-quad", type=float, default=None, help="skip tune; fix |β| cycles")
    p.add_argument("--epsilon-sd", type=float, default=None, help="default: clinical σ_real")
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root_from_here(__file__)
    if args.smoke:
        args.n_replicates = min(args.n_replicates, 2)
        args.n_per_replicate = min(args.n_per_replicate, 40)
        args.tune_n_per = min(args.tune_n_per, 80)

    run_id = args.run_id or f"ell0chirp_{new_run_id()}"
    out_dir = Path(args.output_dir) if args.output_dir else (
        root / "results" / "analysis" / "sin_trend" / "ell0_covariate_validation" / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    vcfg = load_config(DEFAULT_CONFIG_PATH)
    meas_cfg = measurement_config_from_validation_cfg(vcfg)
    clinical_csv = root / vcfg["suite_B"]["clinical_csv"]
    pool = load_clinical_DL_pool(clinical_csv)

    # σ_real from clinical 275
    sigma_real, _, clin_extras = _ell0.clinical_sigma_and_tables(root)
    eps = float(args.epsilon_sd) if args.epsilon_sd is not None else float(sigma_real)

    settings = {
        "run_id": run_id,
        "seed": int(args.seed),
        "epsilon_sd": eps,
        "sigma_real": float(sigma_real),
        "clinical_N_target": {"median": CLINICAL_N_MED, "q1": CLINICAL_N_Q1, "q3": CLINICAL_N_Q3},
        "amplitude_mm": float(args.amplitude_mm),
        "n_replicates": int(args.n_replicates),
        "n_per_replicate": int(args.n_per_replicate),
        "ell0_source": "pipeline_L_base_mm",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note": "slow wavelength chirp + quadratic phase drift; random signs per curve",
    }
    atomic_write_json(settings, out_dir / "run_settings.json")
    atomic_write_json(clin_extras, out_dir / "clinical_extras.json")

    t0 = time.monotonic()

    # ----- tune -----
    if args.chirp_alpha is not None and args.phase_quad is not None:
        best_a, best_b = float(args.chirp_alpha), float(args.phase_quad)
        tune_rows = [{
            "chirp_alpha_mag": best_a,
            "phase_quad_mag": best_b,
            "score": float("nan"),
            "note": "user_fixed",
        }]
    else:
        if not args.quiet:
            print("tuning chirp/phase to match clinical N_hat IQR ...", flush=True)
        alpha_grid = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55]
        beta_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
        tune_rows = []
        for a in alpha_grid:
            for b in beta_grid:
                jobs = build_jobs(
                    seed=int(args.seed) + 1,
                    n_replicates=1,
                    n_per_replicate=int(args.tune_n_per),
                    pool=pool,
                    vcfg=vcfg,
                    amplitude_mm=float(args.amplitude_mm),
                    pairings=("paired",),
                    epsilon_sd=eps,
                    chirp_alpha_mag=a,
                    phase_quad_mag=b,
                    part_tag="tune",
                )
                if not args.quiet:
                    print(f"  grid α={a:.2f} β={b:.2f} jobs={len(jobs)}", flush=True)
                rows = run_jobs(
                    jobs,
                    meas_cfg=meas_cfg,
                    vcfg=vcfg,
                    n_workers=int(args.n_workers),
                    quiet=True,
                )
                st = accepted_N_stats(pd.DataFrame(rows))
                tune_rows.append(
                    {
                        "chirp_alpha_mag": a,
                        "phase_quad_mag": b,
                        **st,
                    }
                )
                if not args.quiet:
                    print(
                        f"    N med/IQR={st['N_median']:.3f} [{st['N_q1']:.3f},{st['N_q3']:.3f}] "
                        f"acc={st['acceptance_rate']:.3f} score={st['score']:.3f}",
                        flush=True,
                    )
        # Prefer clinical N̂ match; break near-ties by higher acceptance.
        tune_df = pd.DataFrame(tune_rows).sort_values(
            ["score", "acceptance_rate"], ascending=[True, False]
        )
        atomic_to_csv(tune_df, out_dir / "tune_grid.csv")
        best = tune_df.iloc[0]
        best_a, best_b = float(best["chirp_alpha_mag"]), float(best["phase_quad_mag"])
        settings["tuned_chirp_alpha_mag"] = best_a
        settings["tuned_phase_quad_mag"] = best_b
        settings["tune_best"] = best.to_dict()
        atomic_write_json(settings, out_dir / "run_settings.json")
        if not args.quiet:
            print(f"selected α={best_a}, β={best_b} (score={best['score']:.4f})", flush=True)
        if args.tune_only:
            print(f"tune-only done → {out_dir}", flush=True)
            return 0

    settings["tuned_chirp_alpha_mag"] = best_a
    settings["tuned_phase_quad_mag"] = best_b
    atomic_write_json(settings, out_dir / "run_settings.json")

    # ----- full run: σ=0 and σ=eps, paired + shuffled -----
    if not args.quiet:
        print(
            f"full run α={best_a} β={best_b}; eps in {{0, {eps:.4f}}}; "
            f"{args.n_replicates}×{args.n_per_replicate}",
            flush=True,
        )
    all_rows: List[Dict[str, Any]] = []
    for eps_i in (0.0, eps):
        jobs = build_jobs(
            seed=int(args.seed),
            n_replicates=int(args.n_replicates),
            n_per_replicate=int(args.n_per_replicate),
            pool=pool,
            vcfg=vcfg,
            amplitude_mm=float(args.amplitude_mm),
            pairings=("paired", "shuffled_DL"),
            epsilon_sd=float(eps_i),
            chirp_alpha_mag=best_a,
            phase_quad_mag=best_b,
            part_tag="5_chirp",
        )
        if not args.quiet:
            print(f"  epsilon_sd={eps_i:.4f} jobs={len(jobs)}", flush=True)
        all_rows.extend(
            run_jobs(
                jobs,
                meas_cfg=meas_cfg,
                vcfg=vcfg,
                n_workers=int(args.n_workers),
                quiet=bool(args.quiet),
            )
        )

    df = pd.DataFrame(all_rows)
    atomic_to_csv(df, out_dir / "all_cases.csv")

    # N_hat match diagnostics by scenario/pairing
    diag_rows = []
    for keys, g in df.groupby(["scenario", "dl_pairing"], sort=True):
        st = accepted_N_stats(g)
        diag_rows.append({"scenario": keys[0], "dl_pairing": keys[1], **st})
    atomic_to_csv(pd.DataFrame(diag_rows), out_dir / "Nhat_match_diagnostics.csv")

    by_rep, mc = _ell0.reanalyze_suite4(df)
    # attach epsilon / chirp meta
    meta = (
        df.groupby(["scenario", "dl_pairing", "A_true_mm"], as_index=False)
        .agg(
            epsilon_sd=("epsilon_sd", "first"),
            chirp_alpha_mag=("chirp_alpha_mag", "first"),
            phase_quad_mag=("phase_quad_mag", "first"),
        )
        if len(df)
        else pd.DataFrame()
    )
    if len(meta):
        by_rep = by_rep.merge(meta, on=["scenario", "dl_pairing", "A_true_mm"], how="left")
        mc = mc.merge(meta, on=["scenario", "dl_pairing", "A_true_mm"], how="left")
    by_rep.insert(0, "part", "5_chirp")
    mc.insert(0, "part", "5_chirp")
    atomic_to_csv(by_rep, out_dir / "by_replicate.csv")
    atomic_to_csv(mc, out_dir / "monte_carlo.csv")
    diag_md = ["## N̂cyc match (accepted)", "", "| scenario | pairing | N med | N q1 | N q3 | acc | score |", "|---|---|---|---|---|---|---|"]
    for r in diag_rows:
        diag_md.append(
            f"| {r['scenario']} | {r['dl_pairing']} | {r['N_median']:.3f} | "
            f"{r['N_q1']:.3f} | {r['N_q3']:.3f} | {r['acceptance_rate']:.3f} | {r['score']:.3f} |"
        )
    (out_dir / "REPORT.md").write_text(
        _ell0.markdown_from_mc(mc, "Validation (5) with slow chirp/phase drift")
        + "\n"
        + "\n".join(diag_md)
        + f"\n\nTuned |α|={best_a}, |β|={best_b}; epsilon grid {{0, {eps:.4f}}}\n",
        encoding="utf-8",
    )
    _ell0.write_interpretation(out_dir / "INTERPRETATION.md", clin_extras, mc)

    if not args.quiet:
        acc = mc[mc["stage"] == "accepted_core_all"]
        print("\nAccepted β_D summary (median):", flush=True)
        cols = [
            c
            for c in [
                "scenario",
                "dl_pairing",
                "epsilon_sd",
                "beta_D_unadj_median",
                "beta_D_adj_median",
                "beta_ell_adj_median",
                "acceptance_rate_median",
                "N_hat_median_median",
            ]
            if c in acc.columns
        ]
        print(acc[cols].to_string(index=False), flush=True)
        print(f"\ndone in {time.monotonic()-t0:.1f}s → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
