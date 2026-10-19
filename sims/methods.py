"""Thin, faithful wrappers around pylmm calls, plus an OLS baseline.

The wrappers do *not* fix pylmm's behaviour (Phase 4 measures it); they only
handle plumbing: passing a copy of the cached eigenvalues (pylmm clamps Kva
in place, bug 12), silencing pylmm's stderr chatter, and returning numpy
arrays / plain floats.
"""

from __future__ import annotations

import contextlib
import io
from typing import Optional

import numpy as np
from scipy import stats


def _quiet():
    return contextlib.redirect_stderr(io.StringIO())


def pylmm_null(y: np.ndarray, kin, X0: Optional[np.ndarray] = None, REML: bool = True) -> dict:
    """Null-model fit with ``pylmm.lmm.LMM(...).fit(REML=REML)``.

    Returns {'h', 'sigma', 'll', 'beta'} (beta as an array).  Missing y are
    dropped by pylmm itself (which then recomputes the eigh).
    """
    from pylmm.lmm import LMM

    y = np.asarray(y, dtype=np.float64)
    Kva, Kve = kin.eig_for_pylmm()
    with _quiet():
        L = LMM(y, np.asarray(kin.K), Kva, Kve, X0=X0)
        h, beta, sigma, ll = L.fit(REML=REML)
    return {
        "h": float(np.sum(h)),
        "sigma": float(np.sum(sigma)),
        "ll": float(np.sum(ll)),
        "beta": np.asarray(beta).reshape(-1),
        "lmm": L,
    }


def pylmm_gwas(
    y: np.ndarray,
    G: np.ndarray,
    kin,
    X0: Optional[np.ndarray] = None,
    REML: bool = True,
    refit: bool = False,
):
    """``pylmm.lmm.GWAS`` on the columns of G -> (t, p) float arrays.

    Note what GWAS does: since Phase 5.2 ``REML`` reaches the null fit,
    every refit and the test (before, it only reached ``association``, and
    REML=False gave the REML Wald test: bugs 1 and 9).  SNPs with missing
    genotypes are refit on the non-missing subset with a fresh eigh each.
    Non-estimable SNPs (monomorphic, collinear with X0) give nan.
    """
    from pylmm.lmm import GWAS

    Kva, Kve = kin.eig_for_pylmm()
    G = np.array(G, dtype=np.float64)  # GWAS does not mutate, but be safe
    with _quiet():
        ts, ps = GWAS(
            np.asarray(y, dtype=np.float64),
            G,
            np.asarray(kin.K),
            Kva,
            Kve,
            X0=X0,
            REML=REML,
            refit=refit,
        )
    return np.asarray(ts, dtype=np.float64), np.asarray(ps, dtype=np.float64)


def pylmm_association(L, x: np.ndarray, REML: bool = True, h: Optional[float] = None):
    """``L.association`` for one SNP column x on a fitted pylmm LMM -> (t, p)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    ts, ps = L.association(x, h=h, REML=REML)
    return float(ts), float(ps)


def ols_gwas(y: np.ndarray, G: np.ndarray, X0: Optional[np.ndarray] = None):
    """Per-SNP OLS t-test of y ~ X0 + g_j -> (t, p) arrays, df = n - q - 1.

    Rows with missing y are dropped; for SNPs with missing calls those rows
    are dropped per SNP (as pylmm's GWAS does).  Monomorphic SNPs give nan.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    G = np.asarray(G, dtype=np.float64)
    n = len(y)
    X0 = np.ones((n, 1)) if X0 is None else np.asarray(X0, dtype=np.float64).reshape(n, -1)
    keep = ~np.isnan(y)
    y, G, X0 = y[keep], G[keep], X0[keep]
    m = G.shape[1]
    t = np.full(m, np.nan)
    df = np.full(m, np.nan)

    complete = ~np.isnan(G).any(axis=0)
    if complete.any():
        t[complete], df[complete] = _ols_block(y, G[:, complete], X0)
    for j in np.where(~complete)[0]:
        ok = ~np.isnan(G[:, j])
        tj, dfj = _ols_block(y[ok], G[ok, j : j + 1], X0[ok])
        t[j], df[j] = tj[0], dfj[0]
    p = np.full(m, np.nan)
    good = ~np.isnan(t)
    p[good] = 2 * stats.t.sf(np.abs(t[good]), df[good])
    return t, p


def _ols_block(y, G, X0):
    """Vectorised OLS t for each column of G given covariates X0."""
    n, q = X0.shape
    Qx, _ = np.linalg.qr(X0)
    ry = y - Qx @ (Qx.T @ y)
    RG = G - Qx @ (Qx.T @ G)
    ss_g = (RG**2).sum(axis=0)
    df = n - q - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (RG.T @ ry) / ss_g
        rss = (ry @ ry) - b**2 * ss_g
        se = np.sqrt(rss / df / ss_g)
        t = b / se
    # monomorphic (after covariates) -> nan, mirroring pylmm
    t[ss_g <= 1e-12 * max(1.0, n)] = np.nan
    return t, np.full(G.shape[1], float(df))
