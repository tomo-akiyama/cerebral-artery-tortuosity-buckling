#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public pipeline module: normality diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import probplot


def show_shapiro_diagnostic_plots(
    values,
    label: str,
    *,
    mean: float | None = None,
    median: float | None = None,
) -> None:
    s = (
        pd.to_numeric(pd.Series(values), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if s.size < 3:
        print(f"Warning: 正規性プロットを描けない（n < 3）: {label}")
        return

    x = s.to_numpy(dtype=float)
    if mean is None:
        mean = float(np.mean(x))
    if median is None:
        median = float(np.median(x))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax_hist = axes[0]
    ax_hist.hist(x, bins="auto", color="0.7", edgecolor="black")
    ax_hist.axvline(
        mean, color="C3", linestyle="--", linewidth=1.5, label=f"mean = {mean:.4g}"
    )
    ax_hist.axvline(
        median, color="C0", linestyle="-", linewidth=1.5, label=f"median = {median:.4g}"
    )
    ax_hist.set_xlabel(label)
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Histogram: {label}")
    ax_hist.legend(frameon=False)

    ax_qq = axes[1]
    probplot(x, dist="norm", plot=ax_qq)
    ax_qq.set_title(f"Q-Q plot: {label}")
    ax_qq.set_xlabel("Theoretical quantiles")
    ax_qq.set_ylabel("Sample quantiles")
    ax_qq.grid(True, alpha=0.3)

    fig.suptitle(f"Normality diagnostics (Shapiro–Wilk): {label}", y=1.02)
    fig.tight_layout()
    plt.show()
