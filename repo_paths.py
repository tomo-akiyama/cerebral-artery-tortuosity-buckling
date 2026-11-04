"""Repository root and common data/code locations (relative layout)."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DIR_PREPROCESSING = REPO_ROOT / "01_preprocessing"
DIR_SEGMENTATION = REPO_ROOT / "02_segmentation"
DIR_MESH = REPO_ROOT / "03_mesh"
DIR_CENTERLINE = REPO_ROOT / "04_centerline"
DIR_M1 = REPO_ROOT / "05_m1_analysis"
DIR_PERIODIC_WAVE = REPO_ROOT / "06_periodic_wave"
DIR_VALIDATION = REPO_ROOT / "07_validation"
DIR_STATISTICS = REPO_ROOT / "08_statistics"
DIR_SIMULATION = REPO_ROOT / "09_simulation"
DIR_FIGURES = REPO_ROOT / "10_figures"
DIR_LIB = REPO_ROOT / "lib"

DATA_ROOT = REPO_ROOT / "data"
DATA_PARTICIPANTS = DATA_ROOT / "participants"
DATA_M1 = DATA_ROOT / "m1"
DATA_PERIODIC_WAVE = DATA_ROOT / "periodic_wave"
DATA_STATISTICS = DATA_ROOT / "statistics"
DATA_SIMULATION = DATA_ROOT / "simulation"
DATA_VALIDATION = DATA_ROOT / "validation"

SIN_TREND_PACKAGE_PARENT = DIR_PERIODIC_WAVE


def ensure_import_paths() -> None:
    """Insert package parents on sys.path for script-style execution."""
    import sys

    for p in (
        REPO_ROOT,
        DIR_LIB,
        DIR_PREPROCESSING,
        DIR_MESH,
        DIR_CENTERLINE,
        DIR_M1,
        DIR_PERIODIC_WAVE,
        DIR_STATISTICS,
        DIR_SIMULATION,
    ):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
