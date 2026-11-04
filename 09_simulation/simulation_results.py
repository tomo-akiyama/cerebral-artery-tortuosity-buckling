"""Save/load buckling simulation results (.npz)."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

RESULT_FILENAME = "P_coupling.npz"
                                  
_LEGACY_RESULT_FILENAMES = ("P_coupling_step_noise_false.npz",)


def path_float_tag(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def buckling_P_coupling_run_subdir(
    *,
    mu: float,
    natural_elongation_rate: float,
    age_offset: float,
    t_max: float,
) -> str:
    tmin = float(age_offset)
    tmx = float(age_offset + t_max)
    param_tag = (
        f"mu{path_float_tag(mu)}"
        f"_g{path_float_tag(natural_elongation_rate)}"
    )
    return (
        "buckling_simulation_P_coupling_"
        f"age_{int(round(tmin))}to{int(round(tmx))}years_{param_tag}"
    )


def buckling_P_coupling_run_dir(
    *,
    mu: float,
    natural_elongation_rate: float,
    age_offset: float,
    t_max: float,
) -> str:
    from simulation.simulation_output import result_subdir

    return result_subdir(
        buckling_P_coupling_run_subdir(
            mu=mu,
            natural_elongation_rate=natural_elongation_rate,
            age_offset=age_offset,
            t_max=t_max,
        )
    )


def result_filename() -> str:
    return RESULT_FILENAME


def result_path_in_run(run_dir: str) -> str:
    primary = os.path.join(run_dir, RESULT_FILENAME)
    if os.path.isfile(primary):
        return primary
    for legacy in _LEGACY_RESULT_FILENAMES:
        legacy_path = os.path.join(run_dir, legacy)
        if os.path.isfile(legacy_path):
            return legacy_path
    return primary


def result_path() -> str:
    from simulation.simulation_figure_config import npz_path_for_active_run

    return npz_path_for_active_run()


def default_run_params() -> dict[str, Any]:
    from simulation.buckling_simulation_P_coupling import (
        _DEFAULT_AGE_OFFSET,
        _DEFAULT_DT,
        _DEFAULT_ES,
        _DEFAULT_EI,
        _DEFAULT_NATURAL_ELONGATION_RATE,
        _DEFAULT_K_S,
        _DEFAULT_L0,
        _DEFAULT_MU,
        _DEFAULT_N,
        _DEFAULT_NOISE_STD,
        _DEFAULT_SAVE_EVERY,
        _DEFAULT_SEED,
        _DEFAULT_T_MIN,
        _DEFAULT_T_MAX,
    )

    return {
        "mu": _DEFAULT_MU,
        "EI": _DEFAULT_EI,
        "k_s": _DEFAULT_K_S,
        "ES": _DEFAULT_ES,
        "L0": _DEFAULT_L0,
        "N": _DEFAULT_N,
        "t_min": _DEFAULT_T_MIN,
        "t_max": _DEFAULT_T_MAX,
        "age_offset": _DEFAULT_AGE_OFFSET,
        "dt": _DEFAULT_DT,
        "natural_elongation_rate": _DEFAULT_NATURAL_ELONGATION_RATE,
        "noise_std": _DEFAULT_NOISE_STD,
        "seed": _DEFAULT_SEED,
        "save_every": _DEFAULT_SAVE_EVERY,
    }


def save_simulation_run(
    path: str,
    x: np.ndarray,
    t_saved: np.ndarray,
    h_saved,
    P_saved: np.ndarray,
    *,
    epsilon_saved: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    h_stack = np.stack([np.asarray(h, dtype=float) for h in h_saved], axis=0)
    meta = dict(params or {})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = dict(
        x=np.asarray(x, dtype=float),
        t_saved=np.asarray(t_saved, dtype=float),
        h_saved=h_stack,
        P_saved=np.asarray(P_saved, dtype=float),
        metadata=np.array(json.dumps(meta)),
    )
    if epsilon_saved is not None:
        payload["epsilon_saved"] = np.asarray(epsilon_saved, dtype=float)
    np.savez_compressed(path, **payload)
    print(f"保存: {os.path.abspath(path)}")


def load_simulation_run(
    path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if path is None:
        path = result_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"シミュレーション結果が見つからない: {path}\n"
            "先に buckling_simulation_P_coupling.py を実行するか、\n"
            "simulation/simulation_figure_config.py の ACTIVE_NPZ_PATH を確認すること。"
        )
    data = np.load(path, allow_pickle=False)
    meta: dict[str, Any] = {}
    if "metadata" in data:
        meta = json.loads(str(data["metadata"]))
    return (
        data["x"],
        data["t_saved"],
        data["h_saved"],
        data["P_saved"],
        meta,
    )


def _simulation_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    meta_only = {"age_offset"}
    return {k: v for k, v in params.items() if k not in meta_only}


def run_and_save_default() -> str:
    from simulation.buckling_simulation_P_coupling import (
        run_buckling_simulation_with_P_coupling,
    )

    params = default_run_params()
    print("Running simulation (initial perturbation at t=0)...")
    x, t_saved, h_saved, P_saved, epsilon_saved = run_buckling_simulation_with_P_coupling(
        **_simulation_kwargs(params),
    )
    run_dir = buckling_P_coupling_run_dir(
        mu=params["mu"],
        natural_elongation_rate=params["natural_elongation_rate"],
        age_offset=params["age_offset"],
        t_max=params["t_max"],
    )
    out = os.path.join(run_dir, RESULT_FILENAME)
    save_simulation_run(
        out,
        x,
        t_saved,
        h_saved,
        P_saved,
        epsilon_saved=epsilon_saved,
        params=params,
    )
    return out
