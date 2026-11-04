"""Public pipeline module: simulation output."""
from __future__ import annotations

import os

FIGURES_DIR_NAME = "figures"
RESULTS_DIR_NAME = "results"


def get_simulation_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_figures_dir() -> str:
    d = os.path.join(get_simulation_root(), FIGURES_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def figures_subdir(name: str) -> str:
    d = os.path.join(get_figures_dir(), name)
    os.makedirs(d, exist_ok=True)
    return d


def get_results_dir() -> str:
    d = os.path.join(get_simulation_root(), RESULTS_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def result_subdir(name: str) -> str:
    d = os.path.join(get_results_dir(), name)
    os.makedirs(d, exist_ok=True)
    return d


def figure_path(filename: str, subdir: str | None = None) -> str:
    base = figures_subdir(subdir) if subdir else get_figures_dir()
    return os.path.join(base, filename)
