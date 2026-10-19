"""Metric helpers for simulation studies.  All functions ignore nan inputs
(nan p-values, e.g. from monomorphic SNPs, are dropped) unless noted."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy import stats

#: median of the chi^2_1 distribution (0.45494...)
CHI2_MEDIAN = float(stats.chi2.ppf(0.5, 1))
DEFAULT_ALPHAS = (0.05, 0.01, 1e-3, 1e-4)


def _clean(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return p[~np.isnan(p)]


def alpha_label(alpha: float) -> str:
    """0.05 -> '0.05', 1e-3 -> '1e-03' (stable metric-name suffixes)."""
    return ("%g" % alpha) if alpha >= 1e-2 else ("%.0e" % alpha)


def chi2_from_p(p) -> np.ndarray:
    """1-df chi^2 statistics with the same two-sided p-values."""
    return stats.chi2.isf(_clean(p), 1)


def lambda_gc(p=None, chi2=None) -> float:
    """Genomic control inflation: median(chi^2) / 0.4549.

    Give p-values (converted with chi2.isf(p, 1)) or chi^2 statistics
    directly (e.g. t^2).  t^2 and chi2.isf(p_t, 1) differ slightly at small
    df because pylmm's p-values come from a t distribution.
    """
    if chi2 is None:
        chi2 = chi2_from_p(p)
    chi2 = _clean(chi2)
    if len(chi2) == 0:
        return np.nan
    return float(np.median(chi2) / CHI2_MEDIAN)


def type1_error(p, alphas: Sequence[float] = DEFAULT_ALPHAS) -> dict[str, float]:
    """{'t1e_<alpha>': fraction of p <= alpha} for null p-values."""
    p = _clean(p)
    return {"t1e_" + alpha_label(a): (float((p <= a).mean()) if len(p) else np.nan) for a in alphas}


def power(p_causal, alphas: Sequence[float] = (5e-8, 1e-5, 1e-3, 0.05)) -> dict[str, float]:
    """{'power_<alpha>': fraction of causal p <= alpha}."""
    p = _clean(p_causal)
    return {
        "power_" + alpha_label(a): (float((p <= a).mean()) if len(p) else np.nan) for a in alphas
    }


def rate_ci(k: float, n: float, level: float = 0.95):
    """Wilson score interval for a binomial proportion k/n -> (lo, hi)."""
    if n <= 0:
        return (np.nan, np.nan)
    z = stats.norm.isf((1 - level) / 2)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    w = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (float(c - w), float(c + w))


def qq_data(p, max_points: int = 500, keep_top: int = 100):
    """Thinned QQ data -> (expected, observed) arrays of -log10 p, sorted
    from most to least significant.

    Keeps the ``keep_top`` smallest p-values exactly and thins the rest to
    roughly ``max_points`` points evenly spaced on the expected -log10 scale,
    so the arrays stay small enough to store per replicate.
    """
    p = np.sort(_clean(p))
    n = len(p)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    exp = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    obs = -np.log10(np.clip(p, 1e-300, 1.0))
    if n <= max_points:
        return exp, obs
    top = np.arange(min(keep_top, n))
    grid = np.linspace(exp[-1], exp[min(keep_top, n - 1)], max(2, max_points - keep_top))
    rest = np.unique(np.searchsorted(-exp, -grid).clip(min(keep_top, n - 1), n - 1))
    idx = np.unique(np.r_[top, rest])
    return exp[idx], obs[idx]


def bias(est, truth) -> float:
    est = np.asarray(est, dtype=np.float64)
    return float(np.nanmean(est - truth))


def rmse(est, truth) -> float:
    est = np.asarray(est, dtype=np.float64)
    return float(np.sqrt(np.nanmean((est - truth) ** 2)))


def coverage(est, se, truth, level: float = 0.95) -> float:
    """Fraction of Wald intervals est +- z*se that contain truth."""
    est = np.asarray(est, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    z = stats.norm.isf((1 - level) / 2)
    ok = ~(np.isnan(est) | np.isnan(se))
    if not ok.any():
        return np.nan
    return float((np.abs(est[ok] - truth) <= z * se[ok]).mean())


def estimate_summary(est, truth, prefix: str = "") -> dict[str, float]:
    """{'<prefix>mean', 'bias', 'rmse', 'sd'} of estimates against truth."""
    est = np.asarray(est, dtype=np.float64)
    return {
        prefix + "mean": float(np.nanmean(est)),
        prefix + "bias": bias(est, truth),
        prefix + "rmse": rmse(est, truth),
        prefix + "sd": float(np.nanstd(est, ddof=1)) if np.sum(~np.isnan(est)) > 1 else np.nan,
    }


def compare(a, b, prefix: str = "") -> dict[str, float]:
    """Agreement between two estimate vectors (e.g. pylmm vs oracle t-stats):
    max/mean absolute difference, max relative difference, correlation."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = ~(np.isnan(a) | np.isnan(b))
    if not ok.any():
        return {prefix + k: np.nan for k in ("max_abs", "mean_abs", "max_rel", "corr")}
    d = np.abs(a[ok] - b[ok])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = d / np.maximum(np.abs(b[ok]), 1e-12)
    corr = (
        float(np.corrcoef(a[ok], b[ok])[0, 1])
        if ok.sum() > 1 and a[ok].std() > 0 and b[ok].std() > 0
        else np.nan
    )
    return {
        prefix + "max_abs": float(d.max()),
        prefix + "mean_abs": float(d.mean()),
        prefix + "max_rel": float(rel.max()),
        prefix + "corr": corr,
    }


def pool_pvalues(arrays: Iterable[Sequence[float]]) -> np.ndarray:
    """Concatenate per-replicate p-value arrays (e.g. stored QQ/p lists)."""
    parts = [_clean(a) for a in arrays]
    return np.concatenate(parts) if parts else np.zeros(0)
