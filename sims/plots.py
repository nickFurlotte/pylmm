"""Small matplotlib helpers for study reports (matplotlib is in the ``sims``
dependency group).  Each function writes one PNG and returns its path.

Styling: fixed categorical order (series keep their colour across figures),
thin lines, recessive grid, a legend whenever there are >= 2 series, and
reference lines (y = x, nominal alpha) in neutral grey.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

# validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REF = "#8a8984"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )
    return plt


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def qq(series: dict[str, np.ndarray], path: str, title: str = "", max_points: int = 2000) -> str:
    """QQ plot of -log10 p for one or more labelled p-value vectors, with a
    pointwise 95% band under the uniform null."""
    from scipy import stats

    from sims.metrics import qq_data

    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    n_max = max((np.sum(~np.isnan(np.asarray(p, float))) for p in series.values()), default=0)
    if n_max:
        k = np.arange(1, n_max + 1)
        e = -np.log10((k - 0.5) / n_max)
        lo = -np.log10(stats.beta.ppf(0.975, k, n_max - k + 1))
        hi = -np.log10(stats.beta.ppf(0.025, k, n_max - k + 1))
        ax.fill_between(e, lo, hi, color=GRID, alpha=0.7, lw=0, label="95% null band")
        ax.plot([0, e.max()], [0, e.max()], color=REF, lw=1)
    for i, (label, p) in enumerate(series.items()):
        x, y = qq_data(p, max_points=max_points)
        ax.plot(x, y, "o", ms=2.5, color=SERIES[i % len(SERIES)], label=label)
    ax.set_xlabel("expected -log10 p")
    ax.set_ylabel("observed -log10 p")
    ax.set_title(title)
    if len(series) > 1 or n_max:
        ax.legend(loc="upper left")
    return _save(fig, path)


def lines(
    df,
    x: str,
    y: str,
    path: str,
    hue: str | None = None,
    yerr: str | None = None,
    title: str = "",
    ref: float | None = None,
    identity: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    logy: bool = False,
) -> str:
    """Metric ``y`` against param ``x`` (one line per ``hue`` level), with
    optional error bars (column ``yerr``), horizontal ``ref`` line (e.g. the
    nominal alpha) or the y = x ``identity`` line (e.g. h2 recovery)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    groups = [(None, df)] if hue is None else list(df.groupby(hue, sort=True))
    for i, (lev, d) in enumerate(groups):
        d = d.sort_values(x)
        ax.errorbar(
            d[x],
            d[y],
            yerr=None if yerr is None else d[yerr],
            marker="o",
            ms=4,
            capsize=2,
            color=SERIES[i % len(SERIES)],
            label=None if hue is None else "%s=%s" % (hue, lev),
        )
    if ref is not None:
        ax.axhline(ref, color=REF, lw=1, ls="--")
    if identity:
        lo, hi = np.nanmin(df[x]), np.nanmax(df[x])
        ax.plot([lo, hi], [lo, hi], color=REF, lw=1, ls="--")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    if hue is not None and len(groups) > 1:
        ax.legend()
    return _save(fig, path)


def scatter(
    x: Sequence[float],
    y: Sequence[float],
    path: str,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "",
    identity: bool = True,
) -> str:
    """Scatter of y vs x (e.g. pylmm vs oracle estimates) with y = x."""
    plt = _plt()
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.plot(x, y, "o", ms=3, color=SERIES[0], alpha=0.8)
    if identity and len(x):
        lo = np.nanmin([np.nanmin(x), np.nanmin(y)])
        hi = np.nanmax([np.nanmax(x), np.nanmax(y)])
        ax.plot([lo, hi], [lo, hi], color=REF, lw=1, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return _save(fig, path)
