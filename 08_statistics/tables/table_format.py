#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public pipeline module: table format."""

from __future__ import annotations

import numpy as np

MISSING = "---"

MAGNITUDE_DECIMALS = 6
P_VALUE_DECIMALS = 5


def format_magnitude_rounded(x: float) -> str:
    if not np.isfinite(x):
        return MISSING
    return f"{float(x):.{MAGNITUDE_DECIMALS}f}"


def format_magnitude_latex(x: float) -> str:
    return format_magnitude_rounded(x)


def format_p_value(p: float) -> str:
    if not np.isfinite(p):
        return MISSING
    return f"\\Pvalue{{{float(p):.{P_VALUE_DECIMALS}f}}}"


def format_exploratory_p(p: float) -> str:
    return format_p_value(p)


def format_beta_cell(beta: float) -> str:
    return format_magnitude_rounded(beta)


def format_ci_cell(ci_low: float, ci_high: float) -> str:
    lo = format_magnitude_rounded(ci_low)
    hi = format_magnitude_rounded(ci_high)
    if lo == MISSING or hi == MISSING:
        return MISSING
    return f"[\\num{{{lo}}} & \\num{{{hi}}}]"


def format_beta_ci_cell(beta: float, ci_low: float, ci_high: float) -> str:
    b = format_magnitude_rounded(beta)
    ci = format_ci_cell(ci_low, ci_high)
    if b == MISSING or ci == MISSING:
        return MISSING
                                                    
    return f"\\makecell{{{b}\\\\ {{{ci}}}}}"


def format_est_ci_cell(est: float, ci_low: float, ci_high: float) -> str:
    e = format_magnitude_rounded(est)
    lo = format_magnitude_rounded(ci_low)
    hi = format_magnitude_rounded(ci_high)
    if e == MISSING or lo == MISSING or hi == MISSING:
        return MISSING
    return f"{e} [\\num{{{lo}}},\\,\\num{{{hi}}}]"