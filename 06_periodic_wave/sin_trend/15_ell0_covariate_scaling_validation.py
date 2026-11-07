#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(5) Validation: adding estimated baseline length ℓ̂0 as a covariate.

Separate from (1)–(4). Does not modify existing accuracy-validation scripts or
their result trees. Reuses measurement / Suite-B job helpers and, when present,
reanalyzes Suite B (4) case-level outputs without re-running generation.

Parts
  5a  Reanalyze existing (4) all_cases.csv: log λ̂ ~ log D ± log ℓ̂0
  5b  New sims with scatter σ matched to clinical residual SD (and 0.3, 0.4)
  5c  Optional: length-determined world λ = ℓ0 / n, n ~ U(1.5, 3)
  clinical  Same regressions on predominantly tangential 275 intervals

ℓ̂0 is the pipeline estimate L_base_mm (same quantity used for N̂cyc).

Example:
  python 06_periodic_wave/sin_trend/15_ell0_covariate_scaling_validation.py \\
      --part 5a+clinical --suite4-dir results/analysis/sin_trend/length_selection_validation/lengthsel_B_n500_r20_20261027

  python 06_periodic_wave/sin_trend/15_ell0_covariate_scaling_validation.py \\
      --part 5b+5c --n-workers 4 --n-replicates 20 --n-per-replicate 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_AC = _HERE.parent
for _p in (str(_AC / "utils"), str(_AC / "mesh"), str(_AC / "morphometrics")):
    if _p not in sys.path:
        sys.path.append(_p)
_STATS = _HERE.parents[1] / "statistics"
if str(_STATS) not in sys.path:
    sys.path.insert(0, str(_STATS))

from sin_trend_paths import (  # noqa: E402
    FRAME_BASELINE,
    analysis_csv_path,
    atomic_to_csv,
    atomic_write_json,
    new_run_id,
    project_root_from_here,
)
from sin_trend_length_selection_validation import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    assign_lambda_true,
    load_clinical_DL_pool,
    load_config,
    measurement_config_from_validation_cfg,
    rng_for,
    run_suite_B_job,
    sample_DL,
    stable_case_id,
)

import importlib.util  # noqa: E402


K_LAMBDA_OVER_D = 17.4
DEFAULT_SUITE4_REL = Path(
    "results/analysis/sin_trend/length_selection_validation/lengthsel_B_n500_r20_20261027"
)
DEFAULT_SEED = 20261030


# ---------------------------------------------------------------------------
# OLS helpers
# ---------------------------------------------------------------------------
def _finite_pos_mask(*cols: np.ndarray) -> np.ndarray:
    m = np.ones(len(cols[0]), dtype=bool)
    for c in cols:
        x = np.asarray(c, float)
        m &= np.isfinite(x) & (x > 0)
    return m


def ols_log_unadj(D: np.ndarray, lam: np.ndarray) -> Dict[str, float]:
    """log λ ~ log D (np.polyfit)."""
    d = np.asarray(D, float)
    y = np.asarray(lam, float)
    m = _finite_pos_mask(d, y)
    out = {"n": int(m.sum()), "beta_D": float("nan"), "status": "ok"}
    if out["n"] < 3:
        out["status"] = "insufficient_n"
        return out
    ld = np.log(d[m])
    ly = np.log(y[m])
    if float(np.var(ld)) <= 1e-18:
        out["status"] = "insufficient_D_variance"
        return out
    out["beta_D"] = float(np.polyfit(ld, ly, 1)[0])
    return out


def ols_log_adj(D: np.ndarray, L: np.ndarray, lam: np.ndarray) -> Dict[str, float]:
    """log λ ~ log D + log ℓ̂0 via least squares."""
    d = np.asarray(D, float)
    ell = np.asarray(L, float)
    y = np.asarray(lam, float)
    m = _finite_pos_mask(d, ell, y)
    out = {
        "n": int(m.sum()),
        "beta_D": float("nan"),
        "beta_ell": float("nan"),
        "status": "ok",
    }
    if out["n"] < 4:
        out["status"] = "insufficient_n"
        return out
    ld = np.log(d[m])
    le = np.log(ell[m])
    ly = np.log(y[m])
    X = np.column_stack([np.ones(out["n"]), ld, le])
    # rank check
    if np.linalg.matrix_rank(X) < 3:
        out["status"] = "rank_deficient"
        return out
    coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
    out["beta_D"] = float(coef[1])
    out["beta_ell"] = float(coef[2])
    return out


def corr_log(a: np.ndarray, b: np.ndarray) -> float:
    m = _finite_pos_mask(a, b)
    if int(m.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(np.log(a[m]), np.log(b[m]))[0, 1])


def _median_iqr(x: np.ndarray) -> Tuple[float, float, float]:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return float(med), float(q1), float(q3)


def summarize_block(
    df: pd.DataFrame,
    *,
    stage: str,
    D_col: str = "D_mm",
    L_col: str = "L_base_mm",
    lam_hat_col: str = "lambda_hat_mm",
    lam_true_col: str = "lambda_true_mm",
    N_hat_col: str = "coverage_hat",
) -> Dict[str, Any]:
    D = pd.to_numeric(df[D_col], errors="coerce").to_numpy(dtype=float)
    L = pd.to_numeric(df[L_col], errors="coerce").to_numpy(dtype=float)
    lam_hat = pd.to_numeric(df[lam_hat_col], errors="coerce").to_numpy(dtype=float)
    if lam_true_col in df.columns:
        lam_true = pd.to_numeric(df[lam_true_col], errors="coerce").to_numpy(dtype=float)
    else:
        lam_true = np.full_like(lam_hat, np.nan)
    if N_hat_col in df.columns:
        N_hat = pd.to_numeric(df[N_hat_col], errors="coerce").to_numpy(dtype=float)
    else:
        N_hat = np.where(_finite_pos_mask(L, lam_hat), L / lam_hat, np.nan)

    unadj = ols_log_unadj(D, lam_hat)
    adj = ols_log_adj(D, L, lam_hat)
    ratio_hat = lam_hat / D
    ratio_true = lam_true / D
    med_r_hat, _, _ = _median_iqr(ratio_hat[_finite_pos_mask(D, lam_hat)])
    med_r_true, _, _ = _median_iqr(ratio_true[_finite_pos_mask(D, lam_true)])
    med_N, q1_N, q3_N = _median_iqr(N_hat)

    return {
        "stage": stage,
        "n": int(len(df)),
        "beta_D_unadj": unadj["beta_D"],
        "beta_D_unadj_status": unadj["status"],
        "beta_D_adj": adj["beta_D"],
        "beta_ell_adj": adj["beta_ell"],
        "beta_adj_status": adj["status"],
        "lambda_hat_over_D_median": med_r_hat,
        "lambda_true_over_D_median": med_r_true,
        "N_hat_median": med_N,
        "N_hat_q1": q1_N,
        "N_hat_q3": q3_N,
        "corr_logD_logL": corr_log(D, L),
    }


def mc_percentiles(series: pd.Series) -> Dict[str, float]:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"median": float("nan"), "p05": float("nan"), "p95": float("nan"), "n": 0}
    return {
        "median": float(np.median(v)),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "n": int(len(v)),
    }


# ---------------------------------------------------------------------------
# 5a: reanalyze Suite B (4)
# ---------------------------------------------------------------------------
def load_suite4_cases(path: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        csv_path = p / "all_cases.csv"
    else:
        csv_path = p
    if not csv_path.is_file():
        raise FileNotFoundError(f"Suite (4) all_cases.csv not found: {csv_path}")
    want = usecols or [
        "replicate",
        "scenario",
        "dl_pairing",
        "A_true_mm",
        "background_name",
        "D_mm",
        "L_base_mm",
        "lambda_hat_mm",
        "lambda_true_mm",
        "coverage_hat",
        "fit_valid",
        "core_all",
        "accepted_core_all",
        "generation_status",
        "execution_status",
    ]
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    cols = [c for c in want if c in header]
    return pd.read_csv(csv_path, usecols=cols, low_memory=False)


def reanalyze_suite4(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-replicate blocks + Monte Carlo summary for 5a."""
    work = df.copy()
    work["fit_valid"] = pd.to_numeric(work["fit_valid"], errors="coerce").fillna(0).astype(int) if "fit_valid" in work.columns else 0
    work["core_all"] = pd.to_numeric(work["core_all"], errors="coerce").fillna(0).astype(int) if "core_all" in work.columns else 0
    if "generation_status" in work.columns:
        work["generation_ok"] = work["generation_status"].astype(str).eq("ok")
    else:
        work["generation_ok"] = True
    if "execution_status" in work.columns:
        work["execution_ok"] = work["execution_status"].astype(str).eq("ok")
    else:
        work["execution_ok"] = True

    group_cols = ["scenario", "dl_pairing", "A_true_mm"]
    if "background_name" in work.columns:
        group_cols.append("background_name")
    group_cols.append("replicate")

    rows: List[Dict[str, Any]] = []
    for keys, g in work.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        n_planned = len(g)
        pre = g.loc[g["generation_ok"] & g["execution_ok"] & (g["fit_valid"] == 1)].copy()
        post = g.loc[g["generation_ok"] & g["execution_ok"] & (g["core_all"] == 1)].copy()
        acc_rate = float((g["core_all"] == 1).mean()) if n_planned else float("nan")

        for stage, sub in (("preaccept_fit_valid", pre), ("accepted_core_all", post)):
            s = summarize_block(sub, stage=stage)
            # also corr on all generated (pre-gate population for pairing structure)
            all_ok = g.loc[g["generation_ok"] & g["execution_ok"]]
            corr_all = corr_log(
                pd.to_numeric(all_ok["D_mm"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(all_ok["L_base_mm"], errors="coerce").to_numpy(dtype=float),
            )
            rows.append(
                {
                    **meta,
                    **s,
                    "acceptance_rate": acc_rate,
                    "corr_logD_logL_all_generated": corr_all,
                    "ell0_source": "pipeline_L_base_mm",
                }
            )

    by_rep = pd.DataFrame(rows)
    # MC aggregate
    value_cols = [
        "beta_D_unadj",
        "beta_D_adj",
        "beta_ell_adj",
        "acceptance_rate",
        "lambda_hat_over_D_median",
        "lambda_true_over_D_median",
        "N_hat_median",
        "corr_logD_logL",
        "corr_logD_logL_all_generated",
    ]
    meta_cols = [c for c in group_cols if c != "replicate"] + ["stage"]
    mc_rows: List[Dict[str, Any]] = []
    for keys, g in by_rep.groupby(meta_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(meta_cols, keys))
        row: Dict[str, Any] = {**meta, "n_replicates": int(g["replicate"].nunique())}
        for c in value_cols:
            pct = mc_percentiles(g[c])
            row[f"{c}_median"] = pct["median"]
            row[f"{c}_p05"] = pct["p05"]
            row[f"{c}_p95"] = pct["p95"]
        mc_rows.append(row)
    return by_rep, pd.DataFrame(mc_rows)


# ---------------------------------------------------------------------------
# Clinical comparison
# ---------------------------------------------------------------------------
def _load_model_validation_module():
    path = _STATS / "11_model_validation.py"
    spec = importlib.util.spec_from_file_location("model_validation_11", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clinical_sigma_and_tables(
    project_root: Path,
    *,
    run_id: str = "run_154_baseline_v2_r0",
) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
    """
    275 predominantly tangential + full 11808 pool correlations / N_cyc.
    Returns sigma_real (residual SD of log λ̂ ~ log D), regression table, extras.
    """
    mod = _load_model_validation_module()
    csv_path = analysis_csv_path(run_id, FRAME_BASELINE, project_root)
    raw = pd.read_csv(csv_path, low_memory=False)
    prepared = mod.prepare_analysis_frame(raw)
    prepared["log_L_base"] = np.log(
        np.where(
            pd.to_numeric(prepared["L_base_mm"], errors="coerce").to_numpy(dtype=float) > 0,
            pd.to_numeric(prepared["L_base_mm"], errors="coerce").to_numpy(dtype=float),
            np.nan,
        )
    )
    if "coverage_hat" in prepared.columns:
        prepared["N_hat"] = pd.to_numeric(prepared["coverage_hat"], errors="coerce")
    else:
        prepared["N_hat"] = (
            pd.to_numeric(prepared["L_base_mm"], errors="coerce")
            / pd.to_numeric(prepared["lambda_mm"], errors="coerce")
        )

    tang = mod.select_predominantly_tangential(prepared)
    # residual SD for unadj log-log on 275
    d = tang["inner_diameter_mm"].to_numpy(dtype=float)
    lam = tang["lambda_mm"].to_numpy(dtype=float)
    m = _finite_pos_mask(d, lam)
    ld, ly = np.log(d[m]), np.log(lam[m])
    slope, intercept = np.polyfit(ld, ly, 1)
    resid = ly - (slope * ld + intercept)
    sigma_real = float(np.std(resid, ddof=1)) if len(resid) > 2 else float("nan")

    cov = mod.covariate_formula(tang)
    cluster = tang["IXI_ID"].astype(str)

    def _fit(formula: str) -> Any:
        return smf.ols(formula, data=tang).fit(
            cov_type="cluster", cov_kwds={"groups": cluster}
        )

    specs = [
        ("unadj_no_ell", "log_lambda ~ log_inner_diameter"),
        ("unadj_with_ell", "log_lambda ~ log_inner_diameter + log_L_base"),
        ("multi_no_ell", f"log_lambda ~ log_inner_diameter + AGE + {cov}"),
        ("multi_with_ell", f"log_lambda ~ log_inner_diameter + log_L_base + AGE + {cov}"),
    ]
    rows = []
    for name, formula in specs:
        res = _fit(formula)
        for term in ("log_inner_diameter", "log_L_base"):
            if term not in res.params.index:
                continue
            ci = res.conf_int().loc[term]
            rows.append(
                {
                    "model": name,
                    "formula": formula,
                    "term": term,
                    "beta": float(res.params[term]),
                    "ci_low": float(ci[0]),
                    "ci_high": float(ci[1]),
                    "p": float(res.pvalues[term]),
                    "n_intervals": int(res.nobs),
                    "n_participants": int(cluster.nunique()),
                }
            )

    N = tang["N_hat"].to_numpy(dtype=float)
    med, q1, q3 = _median_iqr(N)
    extras = {
        "n_tangential": int(len(tang)),
        "n_all_intervals": int(len(prepared)),
        "sigma_real_log_residual": sigma_real,
        "unadj_beta_D_275": float(slope),
        "N_hat_median_275": med,
        "N_hat_q1_275": q1,
        "N_hat_q3_275": q3,
        "corr_logD_logL_275": corr_log(
            tang["inner_diameter_mm"].to_numpy(dtype=float),
            pd.to_numeric(tang["L_base_mm"], errors="coerce").to_numpy(dtype=float),
        ),
        "corr_logD_logL_11808": corr_log(
            prepared["inner_diameter_mm"].to_numpy(dtype=float),
            pd.to_numeric(prepared["L_base_mm"], errors="coerce").to_numpy(dtype=float),
        ),
        "analysis_csv": str(csv_path),
    }
    # histogram counts for N_hat (coarse bins)
    bins = [0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, np.inf]
    labels = ["<1", "1–1.5", "1.5–2", "2–2.5", "2.5–3", "3–4", "4–6", "6–10", "≥10"]
    cats = pd.cut(N[np.isfinite(N)], bins=bins, labels=labels, right=False)
    extras["N_hat_hist_275"] = cats.value_counts().sort_index().to_dict()
    return sigma_real, pd.DataFrame(rows), extras


# ---------------------------------------------------------------------------
# 5b / 5c new simulations (reuse run_suite_B_job)
# ---------------------------------------------------------------------------
def _straight_bg() -> Dict[str, Any]:
    return {"name": "straight", "type": None, "curvature_mm": 0.0}


def build_part5_jobs(
    *,
    seed: int,
    n_replicates: int,
    n_per_replicate: int,
    pool,
    vcfg: Dict[str, Any],
    amplitude_mm: float,
    pairings: Sequence[str],
    scenarios: Sequence[Dict[str, Any]],
    part_tag: str,
) -> List[Dict[str, Any]]:
    """Build Suite-B-compatible jobs for custom lambda scenarios."""
    b = vcfg["suite_B"]
    bg = _straight_bg()
    jobs: List[Dict[str, Any]] = []
    for replicate in range(int(n_replicates)):
        for pairing in pairings:
            rng_dl = rng_for(seed, part_tag, "DL", replicate, pairing, amplitude_mm)
            D, L, src_idx = sample_DL(
                pool, n=n_per_replicate, pairing=str(pairing), rng=rng_dl
            )
            rng_phase = rng_for(seed, part_tag, "phase", replicate, pairing, amplitude_mm)
            phases = rng_phase.uniform(0.0, 2.0 * np.pi, size=n_per_replicate)

            # proportional / length_determined first; permute uses a source pool
            lambda_by_name: Dict[str, np.ndarray] = {}
            meta_by_name: Dict[str, Dict[str, float]] = {}
            for sc in scenarios:
                kind = str(sc["kind"])
                name = str(sc["name"])
                rng_l = rng_for(seed, part_tag, "lam", replicate, pairing, amplitude_mm, name)
                if kind == "proportional":
                    lam, meta = assign_lambda_true(D, scenario=sc, rng=rng_l)
                elif kind == "length_determined":
                    n_cyc = rng_l.uniform(
                        float(sc.get("n_min", 1.5)),
                        float(sc.get("n_max", 3.0)),
                        size=len(L),
                    )
                    lam = np.asarray(L, float) / n_cyc
                    meta = {
                        "gen_slope_loglog": float("nan"),
                        "gen_corr_loglog": float("nan"),
                        "gen_lambda_over_D_median": float(np.nanmedian(lam / D)),
                    }
                    # fill gen slope for bookkeeping
                    from sin_trend_length_selection_validation import _relation_meta

                    meta.update(_relation_meta(D, lam))
                elif kind == "independent_permute":
                    continue
                else:
                    raise ValueError(f"unknown kind {kind}")
                lambda_by_name[name] = lam
                meta_by_name[name] = meta

            for sc in scenarios:
                if str(sc["kind"]) != "independent_permute":
                    continue
                src = sc.get("source_scenario")
                if src not in lambda_by_name:
                    raise ValueError(f"permute source {src!r} missing")
                rng_l = rng_for(
                    seed, part_tag, "lam", replicate, pairing, amplitude_mm, sc["name"]
                )
                lam, meta = assign_lambda_true(
                    D, scenario=sc, rng=rng_l, lambda_pool=lambda_by_name[src]
                )
                lambda_by_name[sc["name"]] = lam
                meta_by_name[sc["name"]] = meta

            for sc in scenarios:
                name = sc["name"]
                lam_arr = lambda_by_name[name]
                gen_meta = meta_by_name[name]
                for i in range(n_per_replicate):
                    case_id = stable_case_id(
                        part_tag,
                        seed,
                        replicate,
                        pairing,
                        amplitude_mm,
                        bg["name"],
                        name,
                        i,
                        float(D[i]),
                        float(L[i]),
                        float(lam_arr[i]),
                        float(phases[i]),
                    )
                    jobs.append(
                        {
                            "suite": "B",
                            "case_id": case_id,
                            "parent_curve_id": case_id,
                            "scenario": name,
                            "lambda_kind": sc["kind"],
                            "replicate": int(replicate),
                            "seed": int(seed),
                            "unit_index": int(i),
                            "dl_pairing": str(pairing),
                            "diameter_mm": float(D[i]),
                            "L0_proxy_mm": float(L[i]),
                            "lambda_true_mm": float(lam_arr[i]),
                            "amplitude_mm": float(amplitude_mm),
                            "phase_rad": float(phases[i]),
                            "background": bg,
                            "noise_sd_mm": 0.0,
                            "source_pool_index": int(src_idx[i]),
                            "gen_slope_loglog": gen_meta.get("gen_slope_loglog"),
                            "gen_corr_loglog": gen_meta.get("gen_corr_loglog"),
                            "gen_lambda_over_D_median": gen_meta.get(
                                "gen_lambda_over_D_median"
                            ),
                            "length_atol_mm": float(b.get("length_atol_mm", 0.05)),
                            "length_rtol": float(b.get("length_rtol", 0.0)),
                            "part5_tag": part_tag,
                        }
                    )
    return jobs


def _worker_B(payload):
    job, meas_cfg, vcfg = payload
    return run_suite_B_job(job, meas_cfg, vcfg)


def run_jobs_collect(
    jobs: List[Dict[str, Any]],
    *,
    meas_cfg: Dict[str, Any],
    vcfg: Dict[str, Any],
    n_workers: int,
    quiet: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not jobs:
        return results
    if n_workers <= 1:
        for i, job in enumerate(jobs, 1):
            results.append(run_suite_B_job(job, meas_cfg, vcfg))
            if not quiet and (i % 50 == 0 or i == len(jobs)):
                print(f"  {i}/{len(jobs)}", flush=True)
        return results

    iterator = iter(jobs)
    ex = ProcessPoolExecutor(max_workers=n_workers)
    futures: Dict[Any, str] = {}

    def submit_next():
        job = next(iterator, None)
        if job is not None:
            futures[ex.submit(_worker_B, (job, meas_cfg, vcfg))] = job["case_id"]

    try:
        for _ in range(2 * n_workers):
            submit_next()
        count = 0
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in finished:
                cid = futures.pop(fut)
                row = fut.result()
                if row.get("case_id") != cid:
                    raise RuntimeError("worker case_id mismatch")
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


def cases_to_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_mc(med, lo, hi, digits=3) -> str:
    if not np.isfinite(med):
        return "NA"
    return f"{med:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def markdown_from_mc(mc: pd.DataFrame, title: str) -> str:
    lines = [f"# {title}", ""]
    if mc.empty:
        lines.append("(empty)")
        return "\n".join(lines) + "\n"
    # prefer accepted stage in a compact table
    prefer = mc
    if "stage" in mc.columns:
        acc = mc[mc["stage"] == "accepted_core_all"]
        if len(acc):
            lines.append("## Accepted (`core_all`)")
            lines.append("")
            prefer = acc
    cols_show = [
        c
        for c in [
            "part",
            "scenario",
            "dl_pairing",
            "A_true_mm",
            "epsilon_sd",
            "stage",
            "acceptance_rate_median",
            "beta_D_unadj_median",
            "beta_D_adj_median",
            "beta_ell_adj_median",
        ]
        if c in prefer.columns
    ]
    # build human rows
    lines.append(
        "| condition | stage | acc. rate | β_D unadj | β_D adj | β_ℓ adj | λ̂/D med | λ/D true med | N̂ med | corr(logD,logℓ̂0) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in prefer.iterrows():
        cond = " / ".join(
            str(r[c])
            for c in ["part", "scenario", "dl_pairing", "A_true_mm", "epsilon_sd"]
            if c in r.index and pd.notna(r[c])
        )
        stage = r.get("stage", "")
        lines.append(
            "| {cond} | {stage} | {acc} | {bu} | {ba} | {be} | {rh} | {rt} | {nh} | {corr} |".format(
                cond=cond,
                stage=stage,
                acc=_fmt_mc(
                    r.get("acceptance_rate_median"),
                    r.get("acceptance_rate_p05"),
                    r.get("acceptance_rate_p95"),
                ),
                bu=_fmt_mc(
                    r.get("beta_D_unadj_median"),
                    r.get("beta_D_unadj_p05"),
                    r.get("beta_D_unadj_p95"),
                ),
                ba=_fmt_mc(
                    r.get("beta_D_adj_median"),
                    r.get("beta_D_adj_p05"),
                    r.get("beta_D_adj_p95"),
                ),
                be=_fmt_mc(
                    r.get("beta_ell_adj_median"),
                    r.get("beta_ell_adj_p05"),
                    r.get("beta_ell_adj_p95"),
                ),
                rh=_fmt_mc(
                    r.get("lambda_hat_over_D_median_median"),
                    r.get("lambda_hat_over_D_median_p05"),
                    r.get("lambda_hat_over_D_median_p95"),
                ),
                rt=_fmt_mc(
                    r.get("lambda_true_over_D_median_median"),
                    r.get("lambda_true_over_D_median_p05"),
                    r.get("lambda_true_over_D_median_p95"),
                ),
                nh=_fmt_mc(
                    r.get("N_hat_median_median"),
                    r.get("N_hat_median_p05"),
                    r.get("N_hat_median_p95"),
                ),
                corr=_fmt_mc(
                    r.get("corr_logD_logL_median"),
                    r.get("corr_logD_logL_p05"),
                    r.get("corr_logD_logL_p95"),
                ),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_interpretation(path: Path, extras: Dict[str, Any], mc: pd.DataFrame) -> None:
    lines = [
        "# Interpretation notes (validation 5)",
        "",
        f"- ℓ̂0 source: pipeline `L_base_mm` (same as N̂cyc).",
        f"- Clinical σ_real (residual SD of log λ̂ ~ log D on 275): "
        f"{extras.get('sigma_real_log_residual', float('nan')):.4f}",
        f"- Clinical unadj β_D on 275: {extras.get('unadj_beta_D_275', float('nan')):.4f}",
        f"- corr(log D, log ℓ̂0) 275 / 11808: "
        f"{extras.get('corr_logD_logL_275', float('nan')):.4f} / "
        f"{extras.get('corr_logD_logL_11808', float('nan')):.4f}",
        "",
        "## Reading guide",
        "- If under true λ ∝ D, adjusted β_D also falls near 0 → ℓ̂0 adjustment erases a true effect; unsuitable as primary model.",
        "- If adjusted β_D stays near 1 → procedure alone does not erase D scaling; clinical vanishing needs another explanation (e.g. length-linked λ).",
        "- If 5c unadj β_D ≈ 0.9 → length-determined λ is compatible with the clinical unadj slope.",
        "",
    ]
    # highlight proportional_sigma0 / σ_real accepted medians if present
    if not mc.empty and "scenario" in mc.columns:
        sub = mc[mc.get("stage", pd.Series(dtype=str)).eq("accepted_core_all")]
        for _, r in sub.iterrows():
            lines.append(
                f"- {r.get('part','')}/{r.get('scenario')}/{r.get('dl_pairing')}: "
                f"β_D unadj={r.get('beta_D_unadj_median', float('nan')):.3f}, "
                f"β_D adj={r.get('beta_D_adj_median', float('nan')):.3f}, "
                f"β_ℓ={r.get('beta_ell_adj_median', float('nan')):.3f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--part",
        default="5a+clinical",
        help="5a|5b|5c|clinical|5a+clinical|5b+5c|all (default: 5a+clinical)",
    )
    p.add_argument("--suite4-dir", type=Path, default=None, help="(4) run dir with all_cases.csv")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-replicates", type=int, default=20)
    p.add_argument("--n-per-replicate", type=int, default=500)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--amplitude-mm", type=float, default=1.0, help="A for 5b/5c (default 1.0)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--clinical-csv", type=Path, default=None)
    p.add_argument("--smoke", action="store_true", help="2 replicates × 20 curves")
    p.add_argument("--skip-5c", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _parse_parts(spec: str) -> List[str]:
    s = spec.strip().lower()
    if s == "all":
        return ["5a", "clinical", "5b", "5c"]
    out: List[str] = []
    for tok in s.replace(",", "+").split("+"):
        tok = tok.strip()
        if not tok:
            continue
        if tok in {"5a", "5b", "5c", "clinical"}:
            out.append(tok)
        else:
            raise ValueError(f"unknown part token {tok!r}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = project_root_from_here(__file__)
    parts = _parse_parts(args.part)
    if args.skip_5c and "5c" in parts:
        parts = [p for p in parts if p != "5c"]
    if args.smoke:
        args.n_replicates = min(args.n_replicates, 2)
        args.n_per_replicate = min(args.n_per_replicate, 20)

    run_id = args.run_id or f"ell0cov_{new_run_id()}"
    out_dir = args.output_dir or (
        root
        / "results"
        / "analysis"
        / "sin_trend"
        / "ell0_covariate_validation"
        / run_id
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vcfg = load_config(args.config or DEFAULT_CONFIG_PATH)
    meas_cfg = measurement_config_from_validation_cfg(vcfg)
    clinical_csv = args.clinical_csv
    if clinical_csv is None:
        clinical_csv = root / vcfg["suite_B"]["clinical_csv"]

    settings = {
        "run_id": run_id,
        "parts": parts,
        "seed": int(args.seed),
        "n_replicates": int(args.n_replicates),
        "n_per_replicate": int(args.n_per_replicate),
        "n_workers": int(args.n_workers),
        "amplitude_mm_5bc": float(args.amplitude_mm),
        "ell0_source": "pipeline_L_base_mm",
        "k_lambda_over_D": K_LAMBDA_OVER_D,
        "smoke": bool(args.smoke),
        "suite4_dir": str(args.suite4_dir or (root / DEFAULT_SUITE4_REL)),
        "clinical_csv": str(clinical_csv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(settings, out_dir / "run_settings.json")

    t0 = time.monotonic()
    mc_frames: List[pd.DataFrame] = []
    clinical_extras: Dict[str, Any] = {}

    # --- clinical (also provides σ_real for 5b) ---
    sigma_real = 0.2
    if "clinical" in parts or "5b" in parts:
        if not args.quiet:
            print("clinical regressions / σ_real ...", flush=True)
        sigma_real, clin_tab, clinical_extras = clinical_sigma_and_tables(root)
        atomic_to_csv(clin_tab, out_dir / "clinical_regressions.csv")
        atomic_write_json(clinical_extras, out_dir / "clinical_extras.json")
        settings["sigma_real"] = sigma_real
        settings["sigma_grid_5b"] = [sigma_real, 0.3, 0.4]
        atomic_write_json(settings, out_dir / "run_settings.json")
        # markdown snippet
        md = ["# Clinical comparison (275 predominantly tangential)", ""]
        md.append(f"- n = {clinical_extras['n_tangential']}")
        md.append(f"- σ_real (log residual of λ̂~D) = {sigma_real:.6f}")
        md.append(f"- unadj β_D = {clinical_extras['unadj_beta_D_275']:.6f}")
        md.append(
            f"- N̂cyc median [IQR] = {clinical_extras['N_hat_median_275']:.3f} "
            f"[{clinical_extras['N_hat_q1_275']:.3f}, {clinical_extras['N_hat_q3_275']:.3f}]"
        )
        md.append(
            f"- corr(log D, log ℓ̂0): 275={clinical_extras['corr_logD_logL_275']:.4f}, "
            f"11808={clinical_extras['corr_logD_logL_11808']:.4f}"
        )
        md.append("")
        md.append("| model | term | β | 95% CI | p |")
        md.append("|---|---|---|---|---|")
        for _, r in clin_tab.iterrows():
            md.append(
                f"| {r['model']} | {r['term']} | {r['beta']:.6f} | "
                f"[{r['ci_low']:.6f}, {r['ci_high']:.6f}] | {r['p']:.5g} |"
            )
        md.append("")
        md.append("N̂cyc histogram (275):")
        for k, v in clinical_extras.get("N_hat_hist_275", {}).items():
            md.append(f"- {k}: {v}")
        (out_dir / "REPORT_clinical.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"  σ_real={sigma_real:.4f}; wrote clinical tables", flush=True)

    # --- 5a ---
    if "5a" in parts:
        if not args.quiet:
            print("5a reanalyze Suite (4) ...", flush=True)
        suite4 = Path(args.suite4_dir) if args.suite4_dir else (root / DEFAULT_SUITE4_REL)
        cases = load_suite4_cases(suite4)
        by_rep, mc = reanalyze_suite4(cases)
        by_rep.insert(0, "part", "5a")
        mc.insert(0, "part", "5a")
        atomic_to_csv(by_rep, out_dir / "5a_by_replicate.csv")
        atomic_to_csv(mc, out_dir / "5a_monte_carlo.csv")
        (out_dir / "REPORT_5a.md").write_text(
            markdown_from_mc(mc, "5a: reanalysis of validation (4)"), encoding="utf-8"
        )
        mc_frames.append(mc)
        if not args.quiet:
            print(f"  rows={len(cases)}; replicates summarized={len(by_rep)}", flush=True)

    # --- 5b / 5c simulations ---
    need_sim = [p for p in parts if p in {"5b", "5c"}]
    if need_sim:
        if not args.quiet:
            print(f"load clinical D/L pool from {clinical_csv} ...", flush=True)
        pool = load_clinical_DL_pool(Path(clinical_csv))
        pool_info = {
            "source_csv": pool.source_csv,
            "source_sha256": pool.source_sha256,
            "n_total": pool.n_total,
            "n_usable": pool.n_usable,
            "drop_reasons": pool.drop_reasons,
        }
        atomic_write_json(pool_info, out_dir / "clinical_pool_info.json")

        all_new_rows: List[Dict[str, Any]] = []
        if "5b" in parts:
            if not args.quiet:
                print("5b simulations (scatter matched to clinical) ...", flush=True)
            sigmas = [float(sigma_real), 0.3, 0.4]
            # unique names
            scenarios_5b = []
            for sd in sigmas:
                tag = f"proportional_sigma{sd:.4f}".replace(".", "p")
                scenarios_5b.append(
                    {
                        "name": tag,
                        "kind": "proportional",
                        "k_lambda_over_D": K_LAMBDA_OVER_D,
                        "epsilon_sd": float(sd),
                        "epsilon_sd_value": float(sd),
                    }
                )
            jobs = build_part5_jobs(
                seed=int(args.seed),
                n_replicates=int(args.n_replicates),
                n_per_replicate=int(args.n_per_replicate),
                pool=pool,
                vcfg=vcfg,
                amplitude_mm=float(args.amplitude_mm),
                pairings=("paired", "shuffled_DL"),
                scenarios=scenarios_5b,
                part_tag="5b",
            )
            if not args.quiet:
                print(f"  5b jobs={len(jobs)}", flush=True)
            rows = run_jobs_collect(
                jobs,
                meas_cfg=meas_cfg,
                vcfg=vcfg,
                n_workers=int(args.n_workers),
                quiet=bool(args.quiet),
            )
            for r in rows:
                r["part"] = "5b"
                # attach epsilon_sd from scenario name lookup
                for sc in scenarios_5b:
                    if r.get("scenario") == sc["name"]:
                        r["epsilon_sd"] = sc["epsilon_sd"]
            all_new_rows.extend(rows)

        if "5c" in parts:
            if not args.quiet:
                print("5c simulations (length-determined λ) ...", flush=True)
            scenarios_5c = [
                {
                    "name": "length_determined_n1p5to3",
                    "kind": "length_determined",
                    "n_min": 1.5,
                    "n_max": 3.0,
                }
            ]
            jobs = build_part5_jobs(
                seed=int(args.seed) + 17,
                n_replicates=int(args.n_replicates),
                n_per_replicate=int(args.n_per_replicate),
                pool=pool,
                vcfg=vcfg,
                amplitude_mm=float(args.amplitude_mm),
                pairings=("paired", "shuffled_DL"),
                scenarios=scenarios_5c,
                part_tag="5c",
            )
            if not args.quiet:
                print(f"  5c jobs={len(jobs)}", flush=True)
            rows = run_jobs_collect(
                jobs,
                meas_cfg=meas_cfg,
                vcfg=vcfg,
                n_workers=int(args.n_workers),
                quiet=bool(args.quiet),
            )
            for r in rows:
                r["part"] = "5c"
                r["epsilon_sd"] = float("nan")
            all_new_rows.extend(rows)

        new_df = cases_to_frame(all_new_rows)
        atomic_to_csv(new_df, out_dir / "5bc_all_cases.csv")
        # analyze like 5a
        # ensure columns
        if "epsilon_sd" not in new_df.columns:
            new_df["epsilon_sd"] = np.nan
        by_rep, mc = reanalyze_suite4(new_df)
        # merge epsilon_sd / part onto mc via by_rep
        if "part" in new_df.columns:
            # reanalyze drops part — add back from new_df group
            pass
        # attach part and epsilon from by joining keys
        key_cols = [c for c in ["scenario", "dl_pairing", "A_true_mm"] if c in new_df.columns]
        meta_map = (
            new_df.groupby(key_cols, as_index=False)
            .agg(part=("part", "first"), epsilon_sd=("epsilon_sd", "first"))
            if key_cols and len(new_df)
            else pd.DataFrame()
        )
        if len(meta_map) and len(by_rep):
            by_rep = by_rep.merge(meta_map, on=key_cols, how="left")
            mc = mc.merge(meta_map, on=key_cols, how="left")
        atomic_to_csv(by_rep, out_dir / "5bc_by_replicate.csv")
        atomic_to_csv(mc, out_dir / "5bc_monte_carlo.csv")
        (out_dir / "REPORT_5bc.md").write_text(
            markdown_from_mc(mc, "5b/5c: new simulations with ℓ̂0-adjusted OLS"),
            encoding="utf-8",
        )
        mc_frames.append(mc)

    # combined report
    if mc_frames:
        combined = pd.concat(mc_frames, ignore_index=True, sort=False)
        atomic_to_csv(combined, out_dir / "summary_monte_carlo_all.csv")
        (out_dir / "REPORT.md").write_text(
            markdown_from_mc(combined, "Validation (5) Monte Carlo summary")
            + "\nSee also REPORT_clinical.md / REPORT_5a.md / REPORT_5bc.md\n",
            encoding="utf-8",
        )
        write_interpretation(out_dir / "INTERPRETATION.md", clinical_extras or settings, combined)

    elapsed = time.monotonic() - t0
    if not args.quiet:
        print(f"done in {elapsed:.1f}s → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
