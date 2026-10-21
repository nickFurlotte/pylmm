# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""Linear mixed models (LMM) for genome-wide association and heritability.

This module provides the numerical core of pylmm: a kinship matrix estimator
(`calculate_kinship`), the model class `LMM` (variance-component fit, Wald
test of one SNP or, batched, of a block of SNPs), the scan `gwas` that
applies it to many SNPs, and the input checks they share
(`check_covariates`, `check_ids`, `snp_estimable`, `snp_estimable_columns`).

The model, for ``n`` individuals, is::

    y = X0 b + x a + u + e
    u ~ N(0, sigma2 * h * K)
    e ~ N(0, sigma2 * (1 - h) * I)

where ``y`` is the phenotype (n,), ``X0`` the covariates (n, q; the default is
an intercept), ``x`` one SNP ((n,) or (n, 1)), ``K`` the kinship matrix (n, n), ``sigma2``
the total phenotypic variance and ``h`` in [0, 1) the heritability (the share
of ``sigma2`` due to ``K``).  ``K`` is eigendecomposed once, ``K = V diag(kva)
V'``; every likelihood is then evaluated on ``V' y`` and ``V' X`` with a
diagonal covariance, which makes a likelihood evaluation O(n) after the
rotation.

Conventions (see `LMM` and ``docs/conventions.md`` for details):

* REML is the default of every function and method that takes ``reml``
  (``reml=True``); ``reml=False`` uses ML.
* Scalars are returned as floats (``sigma``, ``Q``, ``beta_var``, the
  log-likelihood), vectors and matrices as arrays.
* ``sigma2`` is ``Q / (n - p)`` under REML and ``Q / n`` under ML (``p`` is
  the number of columns of the design, ``Q`` the weighted residual sum of
  squares), in `LMM.fit` and in the Wald tests.  `LMM.ml_solution` and
  `LMM.log_likelihood` always return ``Q / (n - p)``.  (With
  ``legacy_fit=True``, `LMM.fit` reports ``Q / (n - p)`` under ML too, as
  pylmm 1.0.0 did.)
* ``X0`` is used exactly as given; the default (``X0=None``) is a single
  column of ones.  If you pass covariates (for example from
  ``PlinkReader.read_covariates``, which returns the covariate columns only,
  no intercept) you must add the intercept yourself, or the model has none:
  ``X0 = np.column_stack([np.ones(n), cov])``.
* ``h`` is searched on the closed interval [0, 1]: a grid (for the default
  ``ngrids=100``, 0, 0.01, ..., 1) whose every local maximum, the two ends
  included, is refined by a bounded search; the best is kept (`LMM.fit`).
  ``legacy_fit=True`` (`LMM`, `gwas`, ``pylmm gwas --legacy-fit``) restores
  the search of pylmm 1.0.0: grid 0, ..., 0.99, the first local maximum.
* Eigenvalues of ``K`` below 1e-6 are raised to 1e-6 (the caller's arrays are
  not modified).
* Names that are a single capital letter (``Y``, ``X``, ``X0``, ``K``) are
  the matrices of the model; everything else is lower_snake_case.

Quick start
-----------
>>> import numpy as np
>>> from pylmm.lmm import calculate_kinship, gwas
>>> rng = np.random.default_rng(0)
>>> W = rng.binomial(2, 0.3, size=(60, 200)).astype(float)   # genotypes 0/1/2
>>> K = calculate_kinship(W)
>>> y = W[:, 0] * 0.8 + rng.normal(size=60)
>>> t_stats, p_values = gwas(y, W[:, :5], K)
>>> len(t_stats), len(p_values)
(5, 5)
"""

from __future__ import annotations

import sys
import time
import warnings
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple, TypeAlias, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, optimize, special, stats

from pylmm.genotypes import DEFAULT_BLOCK_SIZE, check_block_size

# calculate_kinship lives in pylmm.kinship; re-exported here, where pylmm 1.0.0 had it.
from pylmm.kinship import calculate_kinship as calculate_kinship

FloatArray: TypeAlias = NDArray[np.float64]
# A float64 numpy array of any shape (the shape is given in each docstring).

# np.seterr('raise')


def _matrix_mult(A: ArrayLike, B: ArrayLike) -> Any:
    """Matrix product of ``A`` and ``B`` (a thin wrapper over `numpy.dot`)."""
    # The original code tried scipy.linalg.fblas.dgemm first.  scipy has had
    # no linalg.fblas since 0.12, so that path was dead and every call fell
    # back to np.dot -- which is what the golden outputs (tests/golden/) were computed with.
    return np.dot(A, B)


# ---------------------------------------------------------------------------
# Input checks shared by GWAS, LMM and the CLI
# ---------------------------------------------------------------------------

# Relative tolerance of the estimability check: a column whose part outside
# the span of the other columns has squared norm <= ESTIMABLE_RTOL times its
# own squared norm is treated as collinear with them.
ESTIMABLE_RTOL = 1e-10
# Floor of the same check, relative to the uncentred squared norm (rounding level).
_ESTIMABLE_FLOOR = 1e-20


def _residual_on_span(x: FloatArray, X0: FloatArray) -> FloatArray:
    """Return ``x`` minus its least-squares projection on the columns of ``X0`` (via QR)."""
    if X0.shape[1] == 0:
        return x
    Q, _ = linalg.qr(X0, mode="economic")
    r: FloatArray = x - _matrix_mult(Q, _matrix_mult(Q.T, x))
    return r


def snp_estimable(
    x: ArrayLike, X0: ArrayLike, rtol: float = ESTIMABLE_RTOL, Q: FloatArray | None = None
) -> bool:
    """Whether the effect of a SNP can be estimated next to the covariates.

    True if ``x`` is finite, the design ``[X0, x]`` leaves at least one
    residual degree of freedom (``n - q - 1 >= 1``), and ``x`` is not, up to
    rounding, in the column span of ``X0``.  That rejects a constant SNP (when
    ``X0`` has an intercept), a SNP equal to a covariate, and SNPs with too
    few non-missing individuals.

    Parameters
    ----------
    x : (n,) or (n, 1) array
        The SNP column, without missing values (drop those individuals
        first).
    X0 : (n, q) array
        Covariates, assumed of full column rank (see `check_covariates`).
    rtol : float, default `ESTIMABLE_RTOL` (1e-10)
        ``x`` is collinear if the squared norm of its part outside span(X0)
        is at most ``rtol`` times its own squared norm.
    Q : (n, q) array, optional
        Orthonormal basis of the columns of ``X0``, i.e.
        ``scipy.linalg.qr(X0, mode='economic')[0]``; pass it to save the QR
        in a scan.

    Returns
    -------
    bool

    Notes
    -----
    If ``X0`` spans the intercept the tolerance is relative to the *centred*
    ``x``, so that a large constant offset does not make a low-variance SNP
    look collinear.  A second, absolute floor (1e-20 of the uncentred squared
    norm) guards against rounding noise.  This one check replaces the old
    exact-zero variance filters, which let columns that are constant up to
    rounding through with ``t ~ 1e-8`` (issue 19, bug 13).  `gwas` and both
    ``pylmm gwas`` branches call it after dropping individuals with a missing
    genotype.

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.lmm import snp_estimable
    >>> X0 = np.ones((10, 1))
    >>> snp_estimable(np.arange(10.0), X0)
    True
    >>> snp_estimable(np.full(10, 2.0), X0)      # constant: not estimable
    False
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    X0 = np.asarray(X0, dtype=np.float64).reshape(len(x), -1)
    n, q = X0.shape
    if n - q - 1 < 1 or not np.isfinite(x).all():
        return False
    if np.dot(x, x) == 0:
        return False
    if Q is None:
        Q = linalg.qr(X0, mode="economic")[0] if q else np.zeros((n, 0))
    r = x - _matrix_mult(Q, _matrix_mult(Q.T, x))
    # The tolerance is relative to the squared norm of x.  When X0 spans the
    # intercept, that norm is taken after projecting the intercept out (the
    # centred x), so that a big constant offset does not make a low-variance
    # SNP look collinear.
    ref = np.dot(x, x)
    u = np.full(n, 1.0 / np.sqrt(n))
    ru = u - _matrix_mult(Q, _matrix_mult(Q.T, u))
    if np.dot(ru, ru) <= ESTIMABLE_RTOL:
        xc = x - x.mean()
        ref = np.dot(xc, xc)
        if ref == 0:
            return False
    # ... and not below the rounding noise of the (uncentred) column
    return bool(np.dot(r, r) > max(rtol * ref, _ESTIMABLE_FLOOR * np.dot(x, x)))


# A column whose residual on span(X0), computed for the whole block, has squared
# norm above this share of the column's own squared norm is estimable for sure
# (see snp_estimable_columns); anything else is decided by snp_estimable itself.
_ESTIMABLE_CLEAR = 1e-6


def snp_estimable_columns(
    X: ArrayLike, X0: ArrayLike, Q: FloatArray | None = None
) -> NDArray[np.bool_]:
    """`snp_estimable` of every column of a block of SNPs, computed for the block at once.

    Parameters
    ----------
    X : (n, m) array
        SNPs, one per column (a column with ``nan`` or ``inf`` is not
        estimable).
    X0 : (n, q) array
        Covariates, assumed of full column rank (see `check_covariates`).
    Q : (n, q) array, optional
        Orthonormal basis of the columns of ``X0``, as for `snp_estimable`.

    Returns
    -------
    (m,) bool array
        Exactly the decisions of ``snp_estimable(X[:, j], X0)`` (default
        ``rtol``).

    Notes
    -----
    A column is accepted here only if it is finite and its residual ``r`` on
    span(X0), computed for the whole block, has ``|r|^2 > 1e-6 |x|^2``.
    `snp_estimable` computes the same quantities in a different summation
    order; the difference is a rounding error of order ``n * eps * |x|^2``
    (below 1e-9 |x|^2 for any realistic n), far below the margin between
    1e-6 and the thresholds `snp_estimable` applies (``ESTIMABLE_RTOL *
    ref`` with ``ref <= |x|^2``, and 1e-20 |x|^2), so it accepts these
    columns too.  Every other column (non-finite, constant, collinear,
    borderline, or with an overflowing norm) is passed to `snp_estimable`
    itself.  `gwas` and ``pylmm gwas`` use it to select the SNPs of a block
    for `LMM.association_batch`.

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.lmm import snp_estimable_columns
    >>> X0 = np.ones((10, 1))
    >>> X = np.column_stack([np.arange(10.0), np.full(10, 2.0), np.r_[np.nan, np.ones(9)]])
    >>> snp_estimable_columns(X, X0)
    array([ True, False, False])
    """
    X = np.asarray(X, dtype=np.float64)
    X0 = np.asarray(X0, dtype=np.float64)
    n, q = X0.shape
    m = X.shape[1]
    if n - q - 1 < 1:
        return np.zeros(m, dtype=bool)
    if Q is None:
        Q = linalg.qr(X0, mode="economic")[0] if q else np.zeros((n, 0))
    finite = np.isfinite(X).all(axis=0)
    Z = np.where(finite, X, 0.0)
    R = Z - _matrix_mult(Q, _matrix_mult(Q.T, Z))
    rr = np.einsum("ij,ij->j", R, R)
    xx = np.einsum("ij,ij->j", Z, Z)
    clear: NDArray[np.bool_] = finite & (rr > max(_ESTIMABLE_CLEAR, 1e4 * ESTIMABLE_RTOL) * xx)
    for j in np.where(~clear)[0]:
        clear[j] = snp_estimable(X[:, j], X0, Q=Q)
    return clear


def _column_list(cols: Sequence[object] | NDArray[Any]) -> str:
    """Comma-separated string of the items of ``cols`` (for error messages)."""
    return ", ".join(str(c) for c in cols)


def check_covariates(
    X0: ArrayLike, n: int | None = None, what: str = "covariate matrix X0"
) -> FloatArray:
    """Validate a covariate matrix and return it as a float64 array.

    Parameters
    ----------
    X0 : (n, q) array
        Covariates, one column each (include a column of ones for an
        intercept).  ``q = 0`` is allowed.
    n : int, optional
        Expected number of rows (individuals); not checked if None.
    what : str, default "covariate matrix X0"
        How the matrix is named in error messages.

    Returns
    -------
    X0 : (n, q) float64 array
        ``X0`` converted with ``np.asarray`` (a copy only if a conversion was
        needed).

    Raises
    ------
    ValueError
        If ``X0`` is not 2-d, does not have ``n`` rows, contains nan/inf, has
        an all-zero column, two identical columns, or a column that is a
        linear combination of the previous ones (rank deficiency, tolerance
        `ESTIMABLE_RTOL`).  The message names the offending columns (0-based).

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.lmm import check_covariates
    >>> X0 = np.column_stack([np.ones(5), np.arange(5.0)])
    >>> check_covariates(X0, n=5).shape
    (5, 2)
    >>> check_covariates(np.column_stack([X0, X0[:, 1]]))
    Traceback (most recent call last):
    ...
    ValueError: covariate matrix X0: columns 1 and 2 are identical
    """
    X0 = np.asarray(X0, dtype=np.float64)
    if X0.ndim != 2:
        raise ValueError("%s must be a 2-d array (n x q), got shape %s" % (what, X0.shape))
    if n is not None and X0.shape[0] != n:
        raise ValueError("%s has %d rows but there are %d individuals" % (what, X0.shape[0], n))
    bad = ~np.isfinite(X0)
    if bad.any():
        cols = np.where(bad.any(axis=0))[0]
        raise ValueError(
            "%s has %d nan/inf values, in column(s) %s; remove those individuals "
            "or impute the values" % (what, bad.sum(), _column_list(cols))
        )
    q = X0.shape[1]
    if q == 0:
        return X0
    zero = [j for j in range(q) if not X0[:, j].any()]
    if zero:
        raise ValueError("%s: column(s) %s are all zero" % (what, _column_list(zero)))
    for j in range(q):
        for k in range(j):
            if np.array_equal(X0[:, j], X0[:, k]):
                raise ValueError("%s: columns %d and %d are identical" % (what, k, j))
    for j in range(1, q):
        r = _residual_on_span(X0[:, j], X0[:, :j])
        if np.dot(r, r) <= ESTIMABLE_RTOL * np.dot(X0[:, j], X0[:, j]):
            raise ValueError(
                "%s is rank deficient: column %d is a linear combination of columns "
                "%s; remove it" % (what, j, _column_list(range(j)))
            )
    return X0


def check_ids(ids: Sequence[Any] | None, kinship_ids: Sequence[Any] | None) -> None:
    """Check that K and Y describe the same individuals in the same order.

    Parameters
    ----------
    ids : sequence, optional
        Identifiers of the individuals of ``Y`` (any hashable, comparable
        items, e.g. ``(FID, IID)`` tuples or strings).
    kinship_ids : sequence, optional
        Identifiers of the rows/columns of ``K``.  If either argument is None
        nothing is checked.

    Raises
    ------
    ValueError
        If some ``ids`` are not in ``kinship_ids``, if the lengths differ, or
        if the order differs (the message gives the first mismatch and how to
        reorder ``K``: ``K[o][:, o]``).

    Examples
    --------
    >>> from pylmm.lmm import check_ids
    >>> check_ids(["a", "b"], ["a", "b"])       # same: returns None
    >>> check_ids(["a", "b"], ["b", "a"])
    Traceback (most recent call last):
    ...
    ValueError: the kinship matrix is in a different order than the phenotype: position 0 is b in K but a in Y; reorder K (K[o][:, o]) to match
    """
    if ids is None or kinship_ids is None:
        return
    ids, kinship_ids = list(ids), list(kinship_ids)
    if ids == kinship_ids:
        return
    known = set(kinship_ids)
    missing = [i for i in ids if i not in known]
    if missing:
        raise ValueError(
            "%d individuals are not in the kinship matrix, e.g. %s"
            % (len(missing), _column_list(missing[:5]))
        )
    if len(ids) != len(kinship_ids):
        raise ValueError(
            "the kinship matrix has %d individuals, the phenotype %d" % (len(kinship_ids), len(ids))
        )
    i = next(i for i in range(len(ids)) if ids[i] != kinship_ids[i])
    raise ValueError(
        "the kinship matrix is in a different order than the phenotype: position %d "
        "is %s in K but %s in Y; reorder K (K[o][:, o]) to match" % (i, kinship_ids[i], ids[i])
    )


def _check_phenotype(Y: ArrayLike) -> FloatArray:
    """Return ``Y`` as a flat float64 vector (nan = missing).

    Raises ValueError on an infinite value or on a constant phenotype (zero
    variance over the non-missing values); issues 22 and 24.  An all-nan
    ``Y`` is not an error here.
    """
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    obs = Y[~np.isnan(Y)]
    if np.isinf(obs).any():
        raise ValueError("phenotype has %d infinite values" % np.isinf(obs).sum())
    if len(obs) and np.ptp(obs) == 0:
        raise ValueError(
            "phenotype has zero variance: all %d non-missing values equal %g" % (len(obs), obs[0])
        )
    return Y


def _check_design_size(n: int, q: int, what: str = "model") -> None:
    """Raise ValueError unless ``n > q + 1``: a Wald test of one SNP next to
    ``q`` covariates needs a residual degree of freedom (issue 20)."""
    if n <= q + 1:
        raise ValueError(
            "n=%d observations, but the %s has %d covariate column(s) plus the SNP, "
            "so no residual degree of freedom is left (need n > q + 1)" % (n, what, q)
        )


def _check_eigen_args(kva: ArrayLike | None, kve: ArrayLike | None) -> None:
    """Raise unless ``kva`` and ``kve`` are both None or both non-empty.

    Before 1.0.0 an empty sequence (the old mutable default ``[]``) also
    meant "compute the eigendecomposition"; None is now the only way to say so.
    """
    for name, v in (("kva", kva), ("kve", kve)):
        if v is not None and np.size(v) == 0:
            raise ValueError(
                "%s is empty; pass None (the default) to compute the eigendecomposition of K" % name
            )
    if (kva is None) != (kve is None):
        raise ValueError(
            "pass both kva and kve (eigenvalues and eigenvectors of K) or neither; got only %s"
            % ("kva" if kve is None else "kve")
        )


def _snp_column(X: ArrayLike, n: int) -> FloatArray:
    """One SNP, given as (n,) or (n, 1), as an (n, 1) float64 column."""
    x = np.ascontiguousarray(X, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape != (n, 1):
        raise ValueError(
            "the SNP must have shape (%d,) or (%d, 1) (one value per individual with a "
            "phenotype), got %s" % (n, n, np.shape(X))
        )
    return x


class LogLikResult(NamedTuple):
    """The result of `LMM.log_likelihood` (a tuple: ``ll, beta, sigma, xx_inv = ...``).

    Attributes
    ----------
    ll : float
        The log-likelihood (REML or ML).
    beta : (p, 1) array
        Coefficient estimates.
    sigma : float
        ``Q / (N - p)``, see `LMM.ml_solution`.
    xx_inv : (p, p) array
        ``(X'SX)^-1``.
    """

    ll: float
    beta: FloatArray
    sigma: float
    xx_inv: FloatArray


class AssociationResult(NamedTuple):
    """The result of `LMM.association_batch` (a tuple: ``t, p, beta, beta_var = ...``).

    Each field is an ``(m,)`` float64 array, one entry per tested SNP, with
    the same meaning as the scalars `LMM.association` returns.

    Attributes
    ----------
    t : (m,) array
        Wald t statistics.
    p : (m,) array
        Two-sided p-values.
    beta : (m,) array
        SNP effect estimates.
    beta_var : (m,) array
        Their variances (``beta_var = (beta / t) ** 2``).
    """

    t: FloatArray
    p: FloatArray
    beta: FloatArray
    beta_var: FloatArray


def _log_beta_tail(t: FloatArray, df: FloatArray) -> FloatArray:
    """``log I_x(df/2, 1/2)`` with ``x = df / (df + t^2)``: the log two-sided t p-value.

    The two-sided p-value of ``t`` with ``df`` degrees of freedom is the
    regularised incomplete beta function ``I_x(a, b)``, ``a = df/2``,
    ``b = 1/2``, and ``I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * F``,
    where ``F`` is the continued fraction of Numerical Recipes (``betacf``,
    modified Lentz), evaluated here in log space.  It converges quickly
    for ``x < (a + 1) / (a + b + 2)``, i.e. ``t^2 > 3a / (a + 1)``, which
    holds wherever it is used (``p < 1e-280``).  ``log x`` is formed
    without squaring ``t``, so any finite ``t`` is fine.
    """
    a = df / 2.0
    b = 0.5
    log_ratio = np.log1p((df / t) / t)  # log((df + t^2) / t^2)
    log_x = np.log(df) - 2.0 * np.log(t) - log_ratio  # log(df / (df + t^2))
    log1m_x = -log_ratio  # log(t^2 / (df + t^2))
    x = np.exp(log_x)
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones_like(t)
    d = 1.0 - qab * x / qap
    d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
    h = d.copy()
    for m in range(1, 1001):
        m2 = 2.0 * m
        for aa in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / np.where(np.abs(d) < tiny, tiny, d)
            c = 1.0 + aa / c
            c = np.where(np.abs(c) < tiny, tiny, c)
            step = d * c
            h = h * step
        if np.all(np.abs(step - 1.0) < 1e-16):
            break
    out: FloatArray = a * log_x + b * log1m_x - np.log(a) - special.betaln(a, b) + np.log(h)
    return out


def log_pvalue(t: ArrayLike, df: ArrayLike) -> Any:
    """Natural log of the two-sided p-value of a t statistic, without underflow.

    ``log(2 * P(T > |t|))`` for ``T ~ t(df)``.  Where the p-value is
    representable (above ~1e-280) this is ``log(2) + log(sf)``; further out,
    where ``2 * stats.t.sf`` underflows to 0, it is computed from the
    incomplete beta function in log space, so it stays finite and accurate
    for any finite ``t``.

    Parameters
    ----------
    t : float or array
        t statistics (the sign is ignored; ``nan`` gives ``nan``, ``inf``
        gives ``-inf``).
    df : float or array
        Degrees of freedom (``N - p``, see `LMM.association`), broadcast
        against ``t``.

    Returns
    -------
    float or array
        ``log p`` (a numpy float64 for scalar inputs).

    See Also
    --------
    neg_log10_pvalue : the same as ``-log10 p``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from pylmm.lmm import log_pvalue
    >>> bool(np.isclose(log_pvalue(2.5, 30), np.log(2 * stats.t.sf(2.5, 30))))
    True
    >>> float(2 * stats.t.sf(1e5, 100))             # underflows
    0.0
    >>> round(float(log_pvalue(1e5, 100)), 2)       # does not
    -923.56
    """
    t_arr, df_arr = np.broadcast_arrays(
        np.abs(np.asarray(t, dtype=np.float64)), np.asarray(df, dtype=np.float64)
    )
    out = np.full(t_arr.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sf = stats.t.sf(t_arr, df_arr)
        direct = sf > 1e-280
        out[direct] = np.log(2.0) + np.log(sf[direct])
        out[np.isinf(t_arr) & (df_arr > 0)] = -np.inf
        tail = ~direct & np.isfinite(t_arr) & (t_arr > 0) & (df_arr > 0)
        if tail.any():
            out[tail] = _log_beta_tail(t_arr[tail], df_arr[tail])
    if out.ndim == 0:
        return np.float64(out)
    return out


def neg_log10_pvalue(t: ArrayLike, df: ArrayLike) -> Any:
    """``-log10`` of the two-sided p-value of a t statistic, without underflow.

    ``-log_pvalue(t, df) / log(10)``: finite for any finite ``t``, where
    ``p`` itself underflows to 0 (``|t|`` above ~40 for ``df`` = 100).

    Parameters
    ----------
    t : float or array
        t statistics.
    df : float or array
        Degrees of freedom, broadcast against ``t``.

    Returns
    -------
    float or array
        ``-log10 p`` (a numpy float64 for scalar inputs).

    Examples
    --------
    >>> from pylmm.lmm import neg_log10_pvalue
    >>> round(float(neg_log10_pvalue(1.984, 100)), 3)   # p = 0.05
    1.301
    >>> round(float(neg_log10_pvalue(1e5, 100)), 1)
    401.1
    """
    return -log_pvalue(t, df) / np.log(10.0)


def gwas(
    Y: ArrayLike,
    X: ArrayLike,
    K: FloatArray,
    kva: ArrayLike | None = None,
    kve: ArrayLike | None = None,
    X0: ArrayLike | None = None,
    reml: bool = True,
    refit: bool = False,
    ids: Sequence[Any] | None = None,
    kinship_ids: Sequence[Any] | None = None,
    batch: bool = True,
    block_size: int = DEFAULT_BLOCK_SIZE,
    legacy_fit: bool = False,
) -> tuple[list[float], list[float]]:
    """Scan SNPs with the mixed model (a Wald test per SNP, in blocks of SNPs).

    For each column of ``X`` the model ``y = X0 b + x a + u + e`` (see the
    module docstring) is tested for ``a = 0``.  By default the variance
    components ``(h, sigma2)`` are fitted once without a SNP (the null fit)
    and reused for every SNP (the EMMA/"P3D" approximation); with
    ``refit=True`` they are refitted for each SNP.  Individuals with a missing
    phenotype are dropped; for a SNP with missing genotypes, a model on the
    called individuals only is built (the eigendecomposition of that subset of
    ``K`` is recomputed).

    Parameters
    ----------
    Y : (n,) or (n, 1) array
        Phenotype; ``nan`` = missing.  Not modified.
    X : (n, m) array, or (n,) for a single SNP
        Genotypes (any numeric coding), one column per SNP; ``nan`` = missing.
    K : (n, n) ndarray
        Kinship matrix in the same individual order as ``Y``.  It must be a
        numpy ndarray (not a list of lists).
    kva : (n,) array, optional
        Eigenvalues of ``K``, as from ``kva, kve = scipy.linalg.eigh(K)``.
        None (the default) computes them; pass both ``kva`` and ``kve`` or
        neither.  Ignored (recomputed) if ``Y`` has missing values.
    kve : (n, n) array, optional
        Matching eigenvectors (columns), see ``kva``.
    X0 : (n, q) array, optional
        Covariates, full column rank.  Default: one column of ones (an
        intercept).  If you pass ``X0`` it is used exactly as given, with NO
        intercept added: covariates from ``PlinkReader.read_covariates`` have
        no intercept column, so passing them alone silently fits a model
        without one (the test statistics can change a lot).  Add the
        intercept yourself: ``X0 = np.column_stack([np.ones(n), cov])``.
    reml : bool, default True
        REML for the null fit, every refit and the Wald test; False uses ML
        throughout (``sigma2 = Q / n`` in the test).
    refit : bool, default False
        Refit ``h`` and ``sigma2`` with each SNP in the model.
    ids, kinship_ids : sequence, optional
        Individual IDs of ``Y`` and of the rows of ``K``; if both are given
        they must match, in order.
    batch : bool, default True
        Test the complete SNPs (no ``nan``) in blocks with
        `LMM.association_batch`: one matrix product rotates a whole block,
        then closed-form per-SNP formulas.  Much faster.  For
        well-conditioned covariates the statistics equal those of the
        per-SNP path up to rounding (within about 1e-9 relative in the
        tests), also with badly scaled covariates (with ``legacy_fit=True``
        the per-SNP path inverts ``X'SX`` as 1.0.0 did, and with a raw date
        or year column the two can then differ by a few 1e-6 relative, the
        batched values being the more accurate; see `LMM.association_batch`).
        False tests every SNP with `LMM.association`, the code path of
        pylmm 1.0.0.  Ignored with ``refit=True``; SNPs with missing calls
        always take the per-SNP path.
    block_size : int, default `pylmm.genotypes.DEFAULT_BLOCK_SIZE` (1024)
        Number of SNPs per block with ``batch=True``.  A block takes about
        6-8 arrays of ``n * block_size`` floats of memory.
    legacy_fit : bool, default False
        Fit ``h`` as pylmm 1.0.0 did (see `LMM`): every null fit and refit,
        and so every statistic, as in 1.0.0.  With ``batch=False`` the
        results are those of pylmm 1.0.0.

    Returns
    -------
    t_stats : list of length m
        Wald t statistic of each SNP (``t = beta / se``; numpy float64 items;
        ``nan`` for a SNP that
        cannot be tested: all missing, constant, collinear with ``X0``, or
        with too few called individuals).
    p_values : list of length m
        Two-sided p-value of each SNP from ``t(n_s - q - 1)``, where ``n_s``
        is the number of individuals used for that SNP; ``nan`` where
        ``t_stats`` is.  If no individual has a phenotype, both lists are
        all ``nan``.

    The effect scale is that of ``X``: ``beta`` is the change in ``Y`` per
    one-unit increase of the column.  For genotypes read with
    ``PlinkReader(..., norm_genotype=True)`` (the default) that is per SD of
    the standardised genotype.  With ``norm_genotype=False`` it is per 0 -> 1
    step of the A2-allele fraction, i.e. two copies of A2, so the sign and
    scale differ from PLINK's per-A1-copy convention.  (`gwas` returns only
    ``t`` and ``p``; use `LMM.association` with ``return_beta=True``, or
    `LMM.association_batch`, for ``beta``.)

    Raises
    ------
    ValueError
        For an infinite or constant phenotype, ``X`` or ``X0`` with the wrong
        number of rows, nan/inf or rank-deficient ``X0``, ``n <= q + 1``
        observations, mismatching ``ids``/``kinship_ids``, a ``block_size``
        that is not a positive integer, and everything `LMM` raises for ``K`` (wrong shape, nan/inf, bad
        ``kva``/``kve``).

    Warns
    -----
    UserWarning
        If some SNPs are ``nan`` because, on the individuals with a called
        genotype, the covariates are no longer of full rank; and the warnings
        of `LMM` (asymmetric ``K``, clearly negative eigenvalues of ``K``).

    Notes
    -----
    The null ``h`` is searched on [0, 1] (see `LMM.fit`).  The p-value is
    exactly 0.0 when it underflows, for very large ``|t|``; `neg_log10_pvalue`
    gives ``-log10 p`` from ``t`` without underflow.

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.lmm import calculate_kinship, gwas
    >>> rng = np.random.default_rng(2)
    >>> W = rng.binomial(2, 0.3, size=(80, 300)).astype(float)
    >>> K = calculate_kinship(W)
    >>> y = 1.0 * W[:, 0] + rng.normal(size=80)
    >>> X = W[:, :4].copy()
    >>> X[5, 3] = np.nan                       # a missing call is handled
    >>> t_stats, p_values = gwas(y, X, K)
    >>> len(t_stats), len(p_values)
    (4, 4)
    >>> bool(p_values[0] < 1e-3)               # the causal SNP
    True
    >>> t_old, p_old = gwas(y, X, K, batch=False)  # one SNP at a time, as in 1.0.0
    >>> bool(np.allclose(t_stats, t_old, rtol=1e-9))
    True
    """
    _check_eigen_args(kva, kve)
    block_size = check_block_size(block_size)
    Y = _check_phenotype(Y)
    X = np.asarray(X, dtype=np.float64)
    n = len(Y)
    if X.ndim == 1:
        X = X.reshape((n, 1))
    if X.shape[0] != n:
        raise ValueError("SNP matrix X has %d rows but Y has %d values" % (X.shape[0], n))
    m = X.shape[1]
    check_ids(ids, kinship_ids)

    if X0 is None:
        X0 = np.ones((n, 1))
    X0 = np.asarray(X0, dtype=np.float64)
    nobs = (~np.isnan(Y)).sum()
    if nobs and X0.ndim == 2:
        _check_design_size(nobs, X0.shape[1])
    # Shape now; the values (nan, rank) are checked below, on the individuals
    # that have a phenotype (a covariate may be nan where Y is nan).
    if X0.ndim != 2 or X0.shape[0] != n:
        check_covariates(X0, n)

    # Remove missing values in Y and adjust associated parameters
    v = np.isnan(Y)
    if v.sum():
        keep: NDArray[np.bool_] = ~v
        keep = keep.reshape((-1,))
        Y = Y[keep]
        X = X[keep, :]
        X0 = X0[keep, :]
        K = K[keep, :][:, keep]
        kva = None
        kve = None

    if len(Y) == 0:
        return [np.nan] * m, [np.nan] * m

    X0 = check_covariates(X0, len(Y))
    _check_design_size(len(Y), X0.shape[1])
    L = LMM(Y, K, kva, kve, X0, legacy_fit=legacy_fit)
    if not refit:
        L.fit(reml=reml)
    Q0 = linalg.qr(X0, mode="economic")[0] if X0.shape[1] else np.zeros((X0.shape[0], 0))

    t_stats: list[float] = [np.nan] * m
    p_values: list[float] = [np.nan] * m
    counts: dict[str, int] = {}

    n = X.shape[0]
    m = X.shape[1]
    use_batch = batch and not refit

    for start in range(0, m, block_size):
        stop = min(start + block_size, m)
        if use_batch:
            complete = ~np.isnan(X[:, start:stop]).any(axis=0)
        else:
            complete = np.zeros(stop - start, dtype=bool)
        for j in start + np.where(~complete)[0]:  # one SNP at a time
            res = _gwas_one_snp(X[:, [j]], Y, K, X0, Q0, L, reml, refit, counts)
            if res is not None:
                t_stats[j], p_values[j] = res
        # the complete, estimable SNPs of this block, tested together
        cand = start + np.where(complete)[0]
        batched = [int(i) for i in cand[snp_estimable_columns(X[:, cand], X0, Q0)]]
        if batched:
            r = L.association_batch(X[:, batched], reml=reml)
            for k, i in enumerate(batched):
                if np.isfinite(r.t[k]):
                    t_stats[i], p_values[i] = r.t[k], r.p[k]
                else:  # not expected for an estimable SNP; keep the per-SNP answer
                    t_stats[i], p_values[i] = L.association(X[:, i], reml=reml)

    if counts.get("collinear"):
        warnings.warn(
            "%d of %d SNPs are nan because, on the individuals with a called genotype, "
            "the covariates are no longer of full rank" % (counts["collinear"], m),
            stacklevel=2,
        )
    return t_stats, p_values


def _gwas_one_snp(
    x: FloatArray,
    Y: FloatArray,
    K: FloatArray,
    X0: FloatArray,
    Q0: FloatArray,
    L: LMM,
    reml: bool,
    refit: bool,
    counts: dict[str, int],
) -> tuple[np.float64, np.float64] | None:
    """Test one SNP column ``x`` (n, 1) of `gwas` on its own; None if it cannot be tested.

    A SNP with missing calls is tested on the called individuals (a new
    `LMM` on that subset); a complete one with ``L`` (refitted if ``refit``).
    """
    v = np.isnan(x).reshape((-1,))
    if v.sum():
        keep = ~v
        xs = x[keep, :]
        ys = Y[keep]
        x0s = X0[keep, :]
        if not subset_testable(xs, ys, x0s, counts):
            return None
        ks = K[keep, :][:, keep]
        ls = LMM(ys, ks, X0=x0s, validate=False, legacy_fit=L.legacy_fit)
        if refit:
            ls.fit(X=xs, reml=reml)
        else:
            ls.fit(reml=reml)
        return ls.association(xs, reml=reml)
    if not snp_estimable(x, X0, Q=Q0):
        return None
    if refit:
        L.fit(X=x, reml=reml)
    return L.association(x, reml=reml)


def subset_testable(
    xs: FloatArray, ys: FloatArray, x0s: FloatArray, counts: dict[str, int] | None = None
) -> bool:
    """Whether a SNP with missing calls can be tested on the individuals that have a call.

    Parameters
    ----------
    xs : (n_s, 1) or (n_s,) array
        The SNP restricted to the called individuals.
    ys : (n_s,) array
        The phenotype on the same individuals.
    x0s : (n_s, q) array
        The covariates on the same individuals.
    counts : dict, optional
        Updated in place: ``counts['collinear']`` is incremented for each SNP
        rejected because the covariates lost rank on the subset, so that the
        caller can report them once.

    Returns
    -------
    bool
        True if there are at least 2 individuals, ``ys`` is not constant,
        ``x0s`` is still of full column rank and ``xs`` is estimable
        (`snp_estimable`).
    """
    if len(ys) < 2 or np.ptp(ys) == 0:
        return False
    try:
        check_covariates(x0s)
    except ValueError:
        if counts is not None:
            counts["collinear"] = counts.get("collinear", 0) + 1
        return False
    return snp_estimable(xs, x0s)


class LMM:
    """Linear mixed model ``y = X0 b + u + e`` with a kinship-structured random effect.

    ``u ~ N(0, sigma2 * h * K)`` and ``e ~ N(0, sigma2 * (1 - h) * I)``.  The
    three parameters are the heritability ``h`` (the share of the total
    variance ``sigma2`` due to ``K``), ``sigma2``, and the coefficients
    ``b``.  The constructor eigendecomposes ``K`` (unless ``kva``/``kve`` are
    given) and rotates ``Y`` and ``X0`` by the eigenvectors; `fit` then finds
    ``h`` and `association` tests one SNP given that ``h``.

    Parameters
    ----------
    Y : (n,) or (n, 1) array
        Phenotype; ``nan`` marks a missing value and the individual is
        dropped (``K`` and ``X0`` are subset to match, ``kva``/``kve`` are
        recomputed).  Not modified.
    K : (n, n) array
        Kinship (relationship) matrix, in the same individual order as ``Y``.
        Not modified.
    kva : (n,) array, optional
        Eigenvalues of ``K``, e.g. from ``kva, kve = scipy.linalg.eigh(K)``.
        None (the default): computed here.  Pass both ``kva`` and ``kve`` or
        neither; an empty sequence is an error (use None).
    kve : (n, n) array, optional
        Eigenvectors of ``K`` (columns), matching ``kva``.
    X0 : (n, q) array, optional
        Covariates, full column rank.  Default: a single column of ones, the
        mean.  The SNP is *not* part of ``X0``.  A given ``X0`` is used
        exactly as passed, with NO intercept added: covariates from
        ``PlinkReader.read_covariates`` have no intercept column, so passing
        them alone silently fits a model without one.  Use
        ``X0 = np.column_stack([np.ones(n), cov])``.
    verbose : bool, default False
        Write progress (missing values removed, eigendecomposition, number of
        eigenvalues cleaned, multiple optima) to stderr.
    ids, kinship_ids : sequence, optional
        Individual IDs of ``Y`` and of the rows of ``K``; if both are given
        they must match, in order (`check_ids`).
    validate : bool, default True
        Run the input checks below.  `gwas` passes False for its per-SNP
        subsets, which it has already checked.
    legacy_fit : bool, default False
        Fit as pylmm 1.0.0 did, to reproduce its results exactly: `fit`
        searches the grid 0, 0.01, ..., 0.99 only and keeps the first local
        maximum (so ``h`` never exceeds 0.99, and a peak at 0.98 is
        missed), reports ``sigma2 = Q / (N - p)`` under ML too and does not
        warn about an ill-posed ML fit; `log_likelihood` takes the REML
        determinants with ``det`` instead of ``slogdet``.  The Wald tests
        are the same in both modes; they differ only through the fitted
        ``h``.

    Attributes
    ----------
    K : (N, N) float64 array
        The kinship matrix of the individuals with a phenotype.
    N : int
        Number of individuals with a phenotype (rows of ``K``).
    Y : (N, 1) float64 array
        The non-missing phenotype, as a column.
    X0 : (N, q) float64 array
        The covariates of those individuals.
    q : int
        Number of covariate columns (set by `transform`).
    kva : (N,) float64 array
        Eigenvalues of ``K`` with values below 1e-6 raised to 1e-6.
    kve : (N, N) float64 array
        Eigenvectors of ``K`` (as given or computed).
    nonmissing : (n,) bool array
        True where ``Y`` was observed (use it to subset other per-individual
        arrays).
    yt : (N, 1) array
        ``kve.T @ Y``.
    x0t : (N, q) array
        ``kve.T @ X0``.
    x0t_stack : (N, q + 1) array
        ``x0t`` with a last column that `fit`, `log_likelihood` and
        `association` overwrite with the rotated SNP.  It is shared state:
        a design returned from one call is invalidated by the next call that
        stacks a SNP.
    ll_grid, h_grid : (ngrids + 1,) arrays
        After `fit`: the log-likelihood on the grid and the grid of ``h``
        (``ngrids`` points, 0 to 0.99, with ``legacy_fit``).
    opt_h : float
        After `fit`: the estimated heritability ``h``.
    opt_ll : float
        After `fit`: the (RE)ML log-likelihood at ``opt_h``.
    opt_beta : (p, 1) array
        After `fit`: the coefficients at ``opt_h`` (``p = q``, or ``q + 1``
        when `fit` was given a SNP).
    opt_sigma : float
        After `fit`: ``sigma2`` at ``opt_h``.
    legacy_fit : bool
        The ``legacy_fit`` argument.

    Raises
    ------
    ValueError
        (with ``validate=True``) for a nan/inf in ``K`` or ``X0``; an infinite
        or constant phenotype; no non-missing phenotype; a ``K`` that is not
        ``(n, n)``; ``kva``/``kve`` of the wrong shape or with nan/inf, empty,
        or only one of them given;
        rank-deficient, all-zero or duplicated covariate columns; ``X0`` with
        the wrong number of rows; ``n <= q + 1``; ``ids`` that do not match
        ``kinship_ids``.

    Warns
    -----
    UserWarning
        If ``K`` is not symmetric (``eigh`` uses only the lower triangle), or
        has eigenvalues clearly below zero (below ``-1e-8`` of the largest in
        magnitude); round-off negatives are silent.

    Notes
    -----
    * Eigenvalues below 1e-6 are set to 1e-6 (on a copy), so a singular or
      slightly indefinite ``K`` still gives a finite likelihood.
    * Every method that takes ``reml`` defaults to REML (``reml=True``):
      `fit`, `association` and `log_likelihood`.
    * ``sigma2 = Q / (N - p)`` under REML and ``Q / N`` under ML, in `fit`
      and in the Wald tests (`association`, `association_batch`, `gwas`).
    * `fit` searches ``h`` on the closed interval [0, 1] and keeps the best
      local maximum.  At ``h = 1`` the residual variance is zero and the
      model rests on ``K`` alone; the eigenvalue floor (above) keeps the
      likelihood finite there, and for a singular ``K`` it is then very
      low unless the data lie in the span of ``K``.

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.lmm import LMM, calculate_kinship
    >>> rng = np.random.default_rng(3)
    >>> W = rng.binomial(2, 0.3, size=(100, 400)).astype(float)
    >>> K = calculate_kinship(W)
    >>> u = rng.multivariate_normal(np.zeros(100), 0.6 * K)
    >>> y = u + rng.normal(scale=np.sqrt(0.4), size=100)
    >>> model = LMM(y, K)
    >>> model.N, model.q
    (100, 1)
    >>> h, beta, sigma, ll = model.fit()          # null model, REML
    >>> bool(0.0 <= h <= 1.0)
    True
    >>> t, p = model.association(W[:, 0])          # Wald test of one SNP
    >>> bool(0.0 <= p <= 1.0)
    True
    """

    K: FloatArray
    N: int
    Y: FloatArray
    X0: FloatArray
    q: int
    kva: FloatArray
    kve: FloatArray
    nonmissing: NDArray[np.bool_]
    verbose: bool
    yt: FloatArray
    x0t: FloatArray
    x0t_stack: FloatArray
    ll_grid: FloatArray
    h_grid: FloatArray
    opt_h: float
    opt_ll: float
    opt_beta: FloatArray
    opt_sigma: float
    legacy_fit: bool

    def __init__(
        self,
        Y: ArrayLike,
        K: FloatArray,
        kva: ArrayLike | None = None,
        kve: ArrayLike | None = None,
        X0: ArrayLike | None = None,
        verbose: bool = False,
        ids: Sequence[Any] | None = None,
        kinship_ids: Sequence[Any] | None = None,
        validate: bool = True,
        legacy_fit: bool = False,
    ) -> None:
        """Build the model: validate inputs, drop missing phenotypes,
        eigendecompose ``K`` if needed, and rotate ``Y`` and ``X0``.

        See the class docstring for the parameters.
        """

        _check_eigen_args(kva, kve)
        self.legacy_fit = bool(legacy_fit)
        self._ml_ill_posed: bool | None = None  # computed by the first ML fit
        if X0 is None:
            n_y = len(np.asarray(Y))
            X0 = np.ones(n_y).reshape(n_y, 1)
        self.verbose = verbose

        if validate:
            Y = _check_phenotype(Y)
            K = self._check_kinship(K, len(Y))
            X0 = np.asarray(X0, dtype=np.float64)
            nobs = (~np.isnan(Y)).sum()
            if nobs and X0.ndim == 2:
                _check_design_size(nobs, X0.shape[1])
            # shape only here; nan/rank are checked below, after dropping the missing-Y rows
            if X0.ndim != 2 or X0.shape[0] != len(Y):
                check_covariates(X0, len(Y))
            check_ids(ids, kinship_ids)
        else:
            Y = np.asarray(Y, dtype=np.float64).reshape(-1)
            X0 = np.asarray(X0, dtype=np.float64)  # no copy for the callers' float arrays

        x: NDArray[np.bool_] = ~np.isnan(Y)
        x = x.reshape(
            -1,
        )
        if x.sum() != len(Y):
            if self.verbose:
                sys.stderr.write("Removing %d missing values from Y\n" % ((~x).sum()))
            Y = Y[x]
            K = K[x, :][:, x]
            X0 = X0[x, :]
            kva = None
            kve = None
        self.nonmissing = x
        if validate:
            if len(Y) == 0:
                raise ValueError("phenotype has no non-missing values")
            _check_design_size(len(Y), X0.shape[1])
            X0 = check_covariates(
                X0,
                what="covariate matrix X0"
                + (" (individuals with a phenotype)" if x.sum() < len(x) else ""),
            )

        # float64 arrays (no copy if they already are), or None to compute them
        evals: FloatArray | None = None if kva is None else np.asarray(kva, dtype=np.float64)
        evecs: FloatArray | None = None if kve is None else np.asarray(kve, dtype=np.float64)
        if evals is None or evecs is None:
            if self.verbose:
                sys.stderr.write(
                    "Obtaining eigendecomposition for %dx%d matrix\n" % (K.shape[0], K.shape[1])
                )
            begin = time.time()
            evals, evecs = linalg.eigh(K)
            end = time.time()
            if self.verbose:
                sys.stderr.write("Total time: %0.3f\n" % (end - begin))
        elif validate:
            if evals.shape != (K.shape[0],) or evecs.shape != K.shape:
                raise ValueError(
                    "eigendecomposition has shapes %s and %s, expected (%d,) and %s"
                    % (evals.shape, evecs.shape, K.shape[0], K.shape)
                )
            if not (np.isfinite(evals).all() and np.isfinite(evecs).all()):
                raise ValueError("eigendecomposition (kva, kve) has nan/inf values")

        if validate:
            self._warn_negative_eigenvalues(evals)

        self.K = K
        self.kve = evecs
        self.N = self.K.shape[0]
        self.Y = Y.reshape((self.N, 1))
        self.X0 = X0

        # A new array: the caller's kva is not clamped in place (bug 12).
        self.kva = np.array(evals, dtype=np.float64)
        if sum(self.kva < 1e-6):
            if self.verbose:
                sys.stderr.write("Cleaning %d eigen values\n" % (sum(self.kva < 0)))
            self.kva[self.kva < 1e-6] = 1e-6

        self.transform()

    @staticmethod
    def _check_kinship(K: ArrayLike, n: int) -> FloatArray:
        """Return ``K`` as float64 after checking shape ``(n, n)`` and finiteness;
        warn if it is not symmetric.  Raises ValueError otherwise."""
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (n, n):
            raise ValueError(
                "kinship matrix has shape %s but there are %d phenotype values (expected %d x %d)"
                % (K.shape, n, n, n)
            )
        bad = ~np.isfinite(K)
        if bad.any():
            i, j = np.argwhere(bad)[0]
            raise ValueError(
                "kinship matrix has %d nan/inf values, the first at row %d, column %d (0-based)"
                % (bad.sum(), i, j)
            )
        if n:
            asym = np.abs(K - K.T).max()
            if asym > 1e-6 * max(1.0, np.abs(K).max()):
                warnings.warn(
                    "kinship matrix is not symmetric (max |K - K'| = %g); only its lower triangle "
                    "is used" % asym,
                    stacklevel=3,
                )
        return K

    @staticmethod
    def _warn_negative_eigenvalues(kva: FloatArray) -> None:
        """Warn if K is clearly not positive semi-definite.  Round-off
        eigenvalues (|e| ~ 1e-13 relative) are normal for a rank-deficient
        K and are not reported."""
        if not len(kva):
            return
        tol = 1e-8 * max(np.abs(kva).max(), 1e-300)
        neg = kva < -tol
        if neg.any():
            warnings.warn(
                "kinship matrix is not positive semi-definite: %d eigenvalue(s) below -%.1e, "
                "the most negative is %g; all %d eigenvalues below 1e-6 are set to 1e-6"
                % (neg.sum(), tol, kva.min(), (kva < 1e-6).sum()),
                stacklevel=3,
            )

    def transform(self) -> None:
        """Rotate the phenotype and covariates into the eigenbasis of ``K``.

        Sets ``yt = kve.T @ Y``, ``x0t = kve.T @ X0``, ``x0t_stack`` (``x0t``
        plus a column of ones that is later overwritten by the rotated SNP)
        and ``q`` (the number of covariate columns).  Called by the
        constructor; call it again only after changing ``Y``, ``X0`` or
        ``kve``.  In the rotated basis the covariance of ``yt`` is diagonal,
        ``sigma2 * diag(h * kva + 1 - h)``.
        """

        self.yt = _matrix_mult(self.kve.T, self.Y)
        self.x0t = _matrix_mult(self.kve.T, self.X0)
        self.x0t_stack = np.hstack([self.x0t, np.ones((self.N, 1))])
        self.q = self.x0t.shape[1]

    def ml_solution(
        self, h: float, X: FloatArray
    ) -> tuple[FloatArray, float, float, FloatArray, FloatArray]:
        """Generalized least squares at a fixed heritability.

        With ``S = 1 / (h * kva + 1 - h)`` (the weights in the rotated basis)
        this solves ``beta = (X'SX)^-1 X'S yt`` and computes the weighted
        residual sum of squares ``Q = r'Sr`` with ``r = yt - X beta``.  The
        solve is a QR factorisation of ``sqrt(S) X``, accurate also when the
        weights span many orders of magnitude (``h`` near 1 with a singular
        ``K``) or a covariate is badly scaled; with ``legacy_fit=True`` it
        inverts ``X'SX``, as pylmm 1.0.0 did.

        Parameters
        ----------
        h : float
            Heritability in [0, 1].
        X : (N, p) array
            Design matrix **already rotated** into the eigenbasis (e.g.
            ``self.x0t`` or ``self.x0t_stack``), not the raw covariates.

        Returns
        -------
        beta : (p, 1) array
            Coefficient estimates.
        sigma : float
            ``Q / (N - p)``: the estimate of the total variance ``sigma2``
            (REML-style denominator).
        Q : float
            Weighted residual sum of squares.
        xx_inv : (p, p) array
            ``(X'SX)^-1``; ``xx_inv[j, j] * sigma`` is the variance of
            ``beta[j]``.
        xx : (p, p) array
            ``X'SX``.

        Raises
        ------
        scipy.linalg.LinAlgError
            If ``X'SX`` is singular.
        """

        S = 1.0 / (h * self.kva + (1.0 - h))
        p = X.shape[1]
        if self.legacy_fit or not p:
            # pylmm 1.0.0: the normal equations, inverted
            xt = X.T * S
            xx = _matrix_mult(xt, X)
            xx_inv = linalg.inv(xx)
            beta = _matrix_mult(_matrix_mult(xx_inv, xt), self.yt)
            yt = self.yt - _matrix_mult(X, beta)
            Q = float(np.dot(yt.T * S, yt)[0, 0])
        else:
            # Weighted least squares by QR of sqrt(S) X: the normal equations
            # square the condition number, which loses digits when the weights
            # span many orders of magnitude (h near 1 with a singular K: up to 1e6).
            sw = np.sqrt(S)[:, None]
            xw = X * sw
            yw = self.yt * sw
            qx, R = np.linalg.qr(xw)  # numpy's: less call overhead than scipy's for small p
            r_inv = np.linalg.inv(R)  # (p, p) upper triangular, well conditioned as R is
            beta = _matrix_mult(r_inv, _matrix_mult(qx.T, yw))
            resid = yw - _matrix_mult(xw, beta)
            Q = float(np.dot(resid[:, 0], resid[:, 0]))
            xx_inv = _matrix_mult(r_inv, r_inv.T)
            xx = _matrix_mult(R.T, R)
        sigma = Q * 1.0 / (float(self.N) - float(p))
        return beta, sigma, Q, xx_inv, xx

    def _gls_normal(
        self, h: float, X: FloatArray
    ) -> tuple[FloatArray, float, float, FloatArray, float]:
        """`ml_solution` by the normal equations and a Cholesky factor, for the likelihood.

        Cheaper than `ml_solution`'s QR (the likelihood is evaluated ~130
        times per fit) and accurate enough for it; returns ``log|X'SX|`` (from
        the Cholesky factor) in place of ``X'SX``.  Not used with ``legacy_fit``.
        """
        S = 1.0 / (h * self.kva + (1.0 - h))
        p = X.shape[1]
        xt = X.T * S
        if p:
            xx = _matrix_mult(xt, X)
            try:
                chol = np.linalg.cholesky(xx)
            except np.linalg.LinAlgError:
                # not positive definite: only for h outside [0, 1], where the
                # likelihood is nan anyway (or for a singular X'SX: inv raises)
                xx_inv = linalg.inv(xx)
                logdet_xx = float(np.linalg.slogdet(xx)[1])
            else:
                c_inv = np.linalg.inv(chol)
                xx_inv = _matrix_mult(c_inv.T, c_inv)
                logdet_xx = 2.0 * float(np.log(np.diag(chol)).sum())
            beta = _matrix_mult(xx_inv, _matrix_mult(xt, self.yt))
        else:
            xx_inv = np.zeros((0, 0))
            beta = np.zeros((0, 1))
            logdet_xx = 0.0
        r = self.yt - _matrix_mult(X, beta)
        Q = float(np.dot(r[:, 0] * S, r[:, 0]))
        return beta, Q / (float(self.N) - float(p)), Q, xx_inv, logdet_xx

    def _neg_log_likelihood(
        self, h: float, X: FloatArray | None = None, reml: bool = True
    ) -> float:
        """Objective ``-log_likelihood(h)`` for the bounded searches of `_get_best`
        and `_get_max`; 1e6 outside [0, 1].  ``X`` is already rotated and not stacked."""
        # The searches never leave their grid interval inside [0, 1], so the
        # guard is purely defensive; it is kept so LL is never evaluated where
        # it is nan: for h > 1, h*kva + (1-h) can be negative (bug 4).
        if h < 0 or h > 1:
            return 1e6
        return -self.log_likelihood(h, X, stack=False, reml=reml).ll

    def log_likelihood(
        self, h: float, X: ArrayLike | None = None, stack: bool = True, reml: bool = True
    ) -> LogLikResult:
        """Log-likelihood of the data at heritability ``h``, profiled over ``beta`` and ``sigma2``.

        Parameters
        ----------
        h : float
            Heritability in [0, 1].  Above 1, ``h * kva + 1 - h`` can be
            negative and the result is nan.
        X : array, optional
            If None, the covariates ``x0t`` are used.  If given and
            ``stack=True``, ``X`` is a **raw** SNP, shape (N,) or (N, 1), in
            the original individual order; it is rotated and appended to
            ``x0t`` (this overwrites ``x0t_stack``).  If given and
            ``stack=False``, ``X`` is a full **already rotated** design
            matrix (N, p) used in place of ``x0t``.
        stack : bool, default True
            See ``X``.
        reml : bool, default True
            REML (the default; adds the REML terms, an adjustment for the
            estimated ``beta``) or, with False, the ML log-likelihood.  (It
            defaulted to ML before 1.0.0.)

        Returns
        -------
        LogLikResult
            A named tuple ``(ll, beta, sigma, xx_inv)``: ``ll`` (float) is the
            full Gaussian log-likelihood, including the ``n log(2 pi)`` term;
            ``beta`` the (p, 1) coefficients; ``sigma`` (float) is
            ``Q / (N - p)``, see `ml_solution`; ``xx_inv`` the (p, p)
            ``(X'SX)^-1``.

        Notes
        -----
        The likelihood uses ``Q / N`` for the variance, but the returned
        ``sigma`` is ``Q / (N - p)``.

        Examples
        --------
        >>> import numpy as np
        >>> from pylmm.lmm import LMM
        >>> rng = np.random.default_rng(6)
        >>> model = LMM(rng.normal(size=30), np.eye(30))
        >>> res = model.log_likelihood(0.5)
        >>> ll, beta, sigma, xx_inv = res              # still a tuple
        >>> type(res.sigma).__name__, beta.shape
        ('float', (1, 1))
        >>> bool(res.ll == ll)
        True
        """

        if X is None:
            X = self.x0t
        elif stack:
            self.x0t_stack[:, (self.q)] = _matrix_mult(self.kve.T, _snp_column(X, self.N))[:, 0]
            X = self.x0t_stack
        else:
            X = np.asarray(X, dtype=np.float64)

        n = float(self.N)
        q = float(X.shape[1])
        if self.legacy_fit:
            beta, sigma, Q, xx_inv, xx = self.ml_solution(h, X)
        else:
            beta, sigma, Q, xx_inv, logdet_xx = self._gls_normal(h, X)
        ll = (
            n * np.log(2 * np.pi)
            + np.log(h * self.kva + (1.0 - h)).sum()
            + n
            + n * np.log(1.0 / n * Q)
        )
        ll = -0.5 * ll

        if reml:
            if self.legacy_fit:  # pylmm 1.0.0: det can overflow or underflow
                ll_reml_part = (
                    q * np.log(2.0 * np.pi * sigma)
                    + np.log(linalg.det(_matrix_mult(X.T, X)))
                    - np.log(linalg.det(xx))
                )
            else:
                ll_reml_part = (
                    q * np.log(2.0 * np.pi * sigma)
                    + np.linalg.slogdet(_matrix_mult(X.T, X))[1]
                    - logdet_xx
                )
            ll = ll + 0.5 * ll_reml_part

        return LogLikResult(float(ll), beta, sigma, xx_inv)

    def _get_best(self, H: FloatArray, X: FloatArray | None = None, reml: bool = True) -> Any:
        """The maximiser of the likelihood on [0, 1], helper of `fit` (the default search).

        Every local maximum of ``self.ll_grid`` is refined with a bounded
        scalar search (``xatol=1e-10``) on its bracketing interval: an
        interior peak ``i`` on ``[H[i-1], H[i+1]]``, a grid end that is
        higher than its neighbour on ``[H[0], H[1]]`` or ``[H[-2], H[-1]]``.
        The highest of the refined points, the grid peaks and the two grid
        ends is returned; ties go to the smallest ``h``, so a flat profile
        (``h`` not identifiable, e.g. ``K = I``) gives 0.

        Parameters
        ----------
        H : (ngrids + 1,) array
            The grid of ``h`` values that ``ll_grid`` was computed on, from
            0 to 1.
        X : array, optional
            Rotated design passed to the objective (not stacked).
        reml : bool, default True
            REML or ML objective.

        Returns
        -------
        float
            The estimated ``h`` in [0, 1] (numpy float64).
        """
        ll = self.ll_grid
        n = len(ll)
        cands: list[tuple[float, float]] = [(float(ll[0]), float(H[0]))]
        if n > 1:
            cands.append((float(ll[-1]), float(H[-1])))
        finite = ll[np.isfinite(ll)]
        if not len(finite) or np.ptp(finite) <= 1e-12 * max(1.0, np.abs(finite).max()):
            return np.float64(H[0])  # a profile flat to rounding: h is not identifiable
        if n > 1:
            brackets = []
            if ll[0] > ll[1]:
                brackets.append((0, 1))
            for i in range(1, n - 1):
                if ll[i] >= ll[i - 1] and ll[i] >= ll[i + 1] and ll[i] > min(ll[i - 1], ll[i + 1]):
                    brackets.append((i - 1, i + 1))
                    cands.append((float(ll[i]), float(H[i])))
            if ll[-1] > ll[-2]:
                brackets.append((n - 2, n - 1))
            for lo, hi in brackets:
                res = optimize.minimize_scalar(
                    self._neg_log_likelihood,
                    args=(X, reml),
                    bounds=(H[lo], H[hi]),
                    method="bounded",
                    options={"xatol": 1e-10},
                )
                cands.append((-float(res.fun), float(res.x)))
            if len(brackets) > 1 and self.verbose:
                sys.stderr.write("NOTE: Found multiple optima.  Returning the best...\n")
        best = max(c[0] for c in cands)
        return np.float64(min(c[1] for c in cands if c[0] >= best))

    def _check_ml_ill_posed(self) -> bool:
        """Whether the ML likelihood is unbounded as ``h -> 1`` (bug 17), cached.

        It is when ``K`` is singular and its null space (the eigenvectors
        whose eigenvalue was raised to the 1e-6 floor) lies in the span of
        the covariates ``X0``: the covariates then absorb ``y`` in those
        directions, whose variance ``sigma2 * (1 - h)`` goes to 0 as
        ``h -> 1``.  The common case is a centred kinship (``K 1 = 0``, as
        `calculate_kinship` builds) from more SNPs than individuals, whose
        null space is exactly the intercept.  REML is not affected.
        """
        if self._ml_ill_posed is None:
            null = self.kva <= 1e-6
            ill = False
            if null.any() and self.q and null.sum() <= self.q:
                basis = linalg.qr(self.x0t, mode="economic")[0]
                inside = (basis[null, :] ** 2).sum(axis=1)  # 1 = in the span of X0
                ill = bool((inside > 1.0 - 1e-6).all())
            self._ml_ill_posed = ill
        return self._ml_ill_posed

    def _get_max(self, H: FloatArray, X: FloatArray | None = None, reml: bool = True) -> Any:
        """The search of pylmm 1.0.0 (``legacy_fit=True``), helper of `fit`.

        Scans ``self.ll_grid`` for interior local maxima (indices 1 to n-3, so
        the last two grid points are never candidate peaks: bug 2), refines
        each with a bounded scalar search on ``[H[i-1], H[i+1]]``
        (``xatol=1e-10``) and returns the first.  With no interior peak it
        returns the grid end with the larger likelihood (``H[0]`` or
        ``H[-1]``).

        Parameters
        ----------
        H : (ngrids,) array
            The grid of ``h`` values that ``ll_grid`` was computed on.
        X : array, optional
            Rotated design passed to the objective (not stacked).
        reml : bool, default True
            REML or ML objective.

        Returns
        -------
        float
            The estimated ``h`` (numpy float64).
        """
        n = len(self.ll_grid)
        h_opt = []
        for i in range(1, n - 2):
            if self.ll_grid[i - 1] < self.ll_grid[i] and self.ll_grid[i] > self.ll_grid[i + 1]:
                # Bounded to the bracketing grid interval [H[i-1], H[i+1]].  The
                # unbounded optimize.brent used before could step above h = 1,
                # where LL is nan, and scipy >= 1.11 then raised BracketError
                # (bug 4, fixed in 1.0.0).  With xatol this small the
                # stopping rule is ~sqrt(eps)*|h| relative, like optimize.brent.
                # Strictly, this is "the local optimum within the first grid
                # peak's bracket".  The old unbounded Brent could in principle
                # escape into a neighbouring mode (seen in 8.8% of adversarial
                # synthetic two-bump profiles, but in 0/2,100 simulated fits
                # and 0/183 real bimodal S2 fits; max |dh| = 7e-8 and
                # |dt| = 2.8e-7 on non-crashing fits).
                res = optimize.minimize_scalar(
                    self._neg_log_likelihood,
                    args=(X, reml),
                    bounds=(H[i - 1], H[i + 1]),
                    method="bounded",
                    options={"xatol": 1e-10},
                )
                h_opt.append(np.float64(res.x))

        if len(h_opt) > 1:
            if self.verbose:
                sys.stderr.write("NOTE: Found multiple optima.  Returning first...\n")
            return h_opt[0]
        elif len(h_opt) == 1:
            return h_opt[0]
        elif self.ll_grid[0] > self.ll_grid[n - 1]:
            return H[0]
        else:
            return H[n - 1]

    def fit(
        self, X: ArrayLike | None = None, ngrids: int = 100, reml: bool = True
    ) -> tuple[np.float64, FloatArray, float, float]:
        """Estimate the heritability ``h`` (and ``beta``, ``sigma2``) of the model.

        The log-likelihood is evaluated on the grid ``h = 0, 1/ngrids, ...,
        1`` (both ends included); every local maximum of the grid, an end
        included, is refined with a bounded search inside its bracketing
        grid interval, and the highest is kept (`_get_best`).  The result is
        stored on the instance (``opt_h``, ``opt_ll``, ``opt_beta``,
        ``opt_sigma``, ``ll_grid``, ``h_grid``) and returned.

        With ``legacy_fit=True`` (see `LMM`) the search is pylmm 1.0.0's:
        the grid ``0, 1/ngrids, ..., (ngrids - 1)/ngrids`` (0 to 0.99),
        only interior peaks below the last two grid points are refined, and
        the first is kept, else the higher grid end.

        Parameters
        ----------
        X : (N,) or (N, 1) array, optional
            A SNP, in the original individual order (of the individuals with
            a phenotype).  If given it is added to the covariates for this
            fit (the per-SNP refit of `gwas`); otherwise the null model with
            ``X0`` only is fitted.  Overwrites ``x0t_stack``.
        ngrids : int, default 100
            Number of grid intervals on [0, 1] (``ngrids + 1`` points, step
            ``1 / ngrids``); with 100 the grid is 0, 0.01, ..., 1.  (With
            ``legacy_fit``, ``ngrids`` points 0, ..., 0.99.)
        reml : bool, default True
            REML (default) or ML.

        Returns
        -------
        h : float
            Estimated heritability in [0, 1] (numpy float64).
        beta : (p, 1) array
            Coefficients at ``h`` (``p = q``, or ``q + 1`` with ``X``).
        sigma : float
            ``sigma2``: ``Q / (N - p)`` under REML, ``Q / N`` under ML (with
            ``legacy_fit``, ``Q / (N - p)`` in both).
        log_likelihood : float
            The REML or ML log-likelihood at ``h``.

        Raises
        ------
        ValueError
            If ``ngrids`` is not a positive integer.
        scipy.linalg.LinAlgError
            If the weighted normal equations are singular (not expected after
            the constructor's checks).

        Warns
        -----
        UserWarning
            Under ML (``reml=False``) when the ML likelihood has no maximum:
            ``K`` is singular and its null space lies in the span of the
            covariates, as for a centred kinship built from more SNPs than
            individuals with an intercept (bug 17).  The likelihood then grows
            without bound as ``h -> 1``, held back only by the 1e-6
            eigenvalue floor, so the ML ``h`` (an interior local maximum or a
            spike at 1) is unreliable; use REML.  Not with ``legacy_fit``.

        Notes
        -----
        A grid of step 0.01 finds every peak of the profile that is at
        least one grid step away from the others; two peaks closer than
        that are refined as one.  ``h = 1`` is returned when the profile
        increases up to 1 (the residual variance is then 0).

        Examples
        --------
        >>> import numpy as np
        >>> from pylmm.lmm import LMM, calculate_kinship
        >>> rng = np.random.default_rng(4)
        >>> W = rng.binomial(2, 0.3, size=(80, 300)).astype(float)
        >>> K = calculate_kinship(W)
        >>> y = rng.multivariate_normal(np.zeros(80), 0.5 * K + 0.5 * np.eye(80))
        >>> model = LMM(y, K)
        >>> h, beta, sigma, ll = model.fit()
        >>> beta.shape, type(sigma).__name__
        ((1, 1), 'float')
        >>> model.h_grid[[0, -1]]
        array([0., 1.])
        >>> bool(h == model.opt_h)
        True
        >>> old = LMM(y, K, legacy_fit=True)         # pylmm 1.0.0's search
        >>> _ = old.fit()
        >>> old.h_grid[[0, -1]]
        array([0.  , 0.99])
        """
        if isinstance(ngrids, bool) or not isinstance(ngrids, (int, np.integer)) or ngrids < 1:
            raise ValueError("ngrids must be a positive integer, got %r" % (ngrids,))
        if X is None:
            design = self.x0t
        else:
            self.x0t_stack[:, (self.q)] = _matrix_mult(self.kve.T, _snp_column(X, self.N))[:, 0]
            design = self.x0t_stack

        if self.legacy_fit:
            H = np.array(range(ngrids)) / float(ngrids)
        else:
            H = np.arange(ngrids + 1) / float(ngrids)
        grid_ll = np.array([self.log_likelihood(h, design, stack=False, reml=reml).ll for h in H])
        self.ll_grid = grid_ll

        if self.legacy_fit:
            hmax = self._get_max(H, design, reml)
        else:
            hmax = self._get_best(H, design, reml)
        L, beta, sigma, beta_stderr = self.log_likelihood(hmax, design, stack=False, reml=reml)
        if not reml and not self.legacy_fit:
            p = float(design.shape[1])
            sigma = sigma * (float(self.N) - p) / float(self.N)  # the ML Q / N (bug 9)
            if self._check_ml_ill_posed():
                warnings.warn(
                    "the ML likelihood has no maximum: the kinship matrix is singular and "
                    "its null space lies in the span of the covariates (e.g. a centred "
                    "kinship from more SNPs than individuals, with an intercept), so it "
                    "grows without bound as h -> 1 except for the 1e-6 eigenvalue floor, "
                    "and the ML estimate of h is unreliable; use REML",
                    stacklevel=2,
                )

        self.h_grid = H
        self.opt_h = hmax.sum()
        self.opt_ll = L
        self.opt_beta = beta
        self.opt_sigma = sigma

        return hmax, beta, sigma, L

    @overload
    def association(
        self,
        X: ArrayLike,
        h: float | None = ...,
        stack: bool = ...,
        reml: bool = ...,
        return_beta: Literal[False] = ...,
    ) -> tuple[np.float64, np.float64]: ...

    @overload
    def association(
        self,
        X: ArrayLike,
        h: float | None = ...,
        stack: bool = ...,
        reml: bool = ...,
        *,
        return_beta: Literal[True],
    ) -> tuple[np.float64, np.float64, np.float64, float]: ...

    def association(
        self,
        X: ArrayLike,
        h: float | None = None,
        stack: bool = True,
        reml: bool = True,
        return_beta: bool = False,
    ) -> tuple[np.float64, np.float64] | tuple[np.float64, np.float64, np.float64, float]:
        """Wald test of one SNP, at a fixed heritability.

        The SNP is added to the covariates, ``beta`` is re-estimated by
        generalized least squares at ``h``, and ``t = beta_snp / se`` is
        compared with a ``t(N - p)`` distribution (``p = q + 1`` columns).
        Call `fit` first unless ``h`` is given.

        Parameters
        ----------
        X : (N,) or (N, 1) array
            The SNP (raw, original individual order, over the individuals
            with a phenotype).  It must have NO missing values (see Raises); `gwas` handles
            missing calls.  With ``stack=False``, a full
            already-rotated design matrix (N, q + 1) whose last column is the
            SNP.
        h : float, optional
            Heritability to use.  Default: ``self.opt_h`` from the last
            `fit` (an ``AttributeError`` if there was none).
        stack : bool, default True
            Rotate ``X`` and append it to ``x0t``; False: see ``X``.
        reml : bool, default True
            True: REML Wald test with ``sigma2 = Q / (N - p)``.  False: ML
            Wald test with ``sigma2 = Q / N``.  Fit the null model with the
            same choice (``fit(reml=...)``).
        return_beta : bool, default False
            Also return the SNP effect and its variance.

        Returns
        -------
        t_stat : float
            Wald t statistic (numpy float64).
        p_value : float
            Two-sided p-value (numpy float64).
        beta : float
            Only if ``return_beta``: SNP effect estimate (numpy float64), per
            unit of the ``X`` column.  For genotypes from ``PlinkReader``:
            with ``norm_genotype=True`` (default) per SD of the standardised
            genotype; with ``norm_genotype=False`` per 0 -> 1 step of the
            A2-allele fraction (two copies of A2), so the sign and scale
            differ from PLINK's per-A1-copy convention.
        beta_var : float
            Only if ``return_beta``: its variance, ``xx_inv[-1, -1] * sigma``
            (with the sigma used for the test).

        Raises
        ------
        ValueError
            If ``X`` is not (N,) or (N, 1) (``stack=True``).  A SNP
            containing ``nan`` fails inside scipy ("array must not
            contain infs or NaNs"); `association` does not handle missing
            calls.  Use `gwas`, which tests the called individuals only, or
            impute first.

        Notes
        -----
        ``X`` is not checked for being constant or collinear with ``X0``;
        use `snp_estimable` (as `gwas` does), or the statistic is meaningless
        or the solve fails.  With ``stack=True`` the shared ``x0t_stack`` is
        overwritten.

        Examples
        --------
        >>> import numpy as np
        >>> from pylmm.lmm import LMM, calculate_kinship
        >>> rng = np.random.default_rng(5)
        >>> W = rng.binomial(2, 0.3, size=(80, 300)).astype(float)
        >>> K = calculate_kinship(W)
        >>> y = 1.0 * W[:, 0] + rng.normal(size=80)
        >>> model = LMM(y, K)
        >>> _ = model.fit()
        >>> t, p, b, var = model.association(W[:, 0], return_beta=True)
        >>> bool(p < 1e-3) and round(float(b)) == 1
        True
        >>> type(var).__name__
        'float'
        >>> t2, p2 = model.association(W[:, [1]])             # an (n, 1) column works too
        >>> bool(0 <= p2 <= 1)
        True
        """
        if stack:
            self.x0t_stack[:, (self.q)] = _matrix_mult(self.kve.T, _snp_column(X, self.N))[:, 0]
            design = self.x0t_stack
        else:
            design = np.asarray(X, dtype=np.float64)

        if h is None:
            h = self.opt_h

        if self.legacy_fit:  # 1.0.0: through log_likelihood (the same numbers, more work)
            L, beta, sigma, beta_var = self.log_likelihood(h, design, stack=False, reml=reml)
        else:
            beta, sigma, _, beta_var, _ = self.ml_solution(h, design)
        q = len(beta)
        # ML test: sigma^2 = Q/n, the ML estimate (bug 9, fixed in 1.0.0); REML
        # keeps Q/(n-q).  The reference distribution is t(n-q) either way.
        if not reml:
            sigma = sigma * (float(self.N) - q) / float(self.N)
        ts, ps = self.tstat(beta[q - 1], beta_var[q - 1, q - 1], sigma, q)

        if return_beta:
            return ts, ps, beta[q - 1].sum(), float(beta_var[q - 1, q - 1] * sigma)
        return ts, ps

    def association_batch(
        self, X: ArrayLike, h: float | None = None, reml: bool = True
    ) -> AssociationResult:
        """Wald tests of many SNPs at a fixed heritability, one SNP at a time in the model.

        The same tests as `LMM.association` (``return_beta=True``) applied to
        each column of ``X``, computed together: the block is rotated into
        the eigenbasis of ``K`` with one matrix product, and the per-SNP
        generalized least squares fit is written in closed form against the
        null model (below), so no ``(q + 1) x (q + 1)`` system is solved per
        SNP.  Call `fit` first unless ``h`` is given.

        Parameters
        ----------
        X : (N, m) array, or (N,) for one SNP
            SNPs (raw, original individual order, over the individuals with a
            phenotype), one per column, with NO missing values (a ``nan`` or
            ``inf`` column gives ``nan`` results).
        h : float, optional
            Heritability to use.  Default: ``self.opt_h`` from the last
            `fit`.
        reml : bool, default True
            True: ``sigma2 = Q / (N - p)``; False: ``Q / N`` (as
            `association`), with ``p = q + 1``.

        Returns
        -------
        AssociationResult
            Named tuple ``(t, p, beta, beta_var)`` of ``(m,)`` float64 arrays:
            the Wald t statistic, its two-sided p-value from ``t(N - p)``,
            the SNP effect (per unit of the ``X`` column) and its variance.

        Notes
        -----
        With weights ``S = 1 / (h * kva + 1 - h)``, rotated covariates
        ``x0t`` and phenotype ``yt``, let ``r0`` be the null-model GLS
        residual of ``yt`` and ``xr`` the GLS residual of a rotated SNP
        ``xt`` on ``x0t``.  Then (Frisch-Waugh-Lovell)::

            s    = xr' S xr            # 1 / (X'SX)^-1[-1, -1]
            beta = xr' S r0 / s
            Q    = (r0 - beta xr)' S (r0 - beta xr)
            beta_var = sigma2 / s,  t = beta / sqrt(beta_var)

        For well-conditioned covariates these equal `association`'s numbers
        up to rounding: within about 1e-9 relative in the tests and
        benchmarks (the largest relative differences are for values near 0)
        and a few 1e-12 of the largest value.  With badly scaled or nearly
        collinear covariates (e.g. a raw date ~2e7 or a year column next to
        the intercept) both stay accurate (`association` solves the
        weighted least squares by QR; about 3e-13 of the largest ``t`` off
        an accurate reference for a date ~2e7).  With ``legacy_fit=True``
        `association` inverts ``X'SX`` as pylmm 1.0.0 did, loses precision
        there, and the two can differ by up to a few 1e-6 relative; the
        batched values are then the more accurate ones (against a 40-digit
        reference, about 3e-11 vs 3e-7 off).  Centring or scaling such
        covariates makes both paths accurate.  Columns are not checked for being constant or collinear with ``X0`` (use
        `snp_estimable`, as `gwas` does): for such a column ``s`` is zero or
        rounding noise, and the result is ``nan`` (``s <= 0``) or
        meaningless.  The p-value is exactly 0.0 when it underflows.

        Examples
        --------
        >>> import numpy as np
        >>> from pylmm.lmm import LMM, calculate_kinship
        >>> rng = np.random.default_rng(5)
        >>> W = rng.binomial(2, 0.3, size=(80, 300)).astype(float)
        >>> K = calculate_kinship(W)
        >>> y = 1.0 * W[:, 0] + rng.normal(size=80)
        >>> model = LMM(y, K)
        >>> _ = model.fit()
        >>> res = model.association_batch(W[:, :50])
        >>> res.t.shape
        (50,)
        >>> t, p, b, var = model.association(W[:, 0], return_beta=True)
        >>> bool(np.isclose(res.t[0], t, rtol=1e-10) and np.isclose(res.beta_var[0], var))
        True
        """
        if h is None:
            h = self.opt_h
        G = np.asarray(X, dtype=np.float64)
        if G.ndim == 1:
            G = G.reshape(-1, 1)
        if G.ndim != 2 or G.shape[0] != self.N:
            raise ValueError(
                "X must have shape (%d, m) or (%d,) (one row per individual with a "
                "phenotype), got %s" % (self.N, self.N, np.shape(X))
            )
        S = 1.0 / (h * self.kva + (1.0 - h))
        xt = _matrix_mult(self.kve.T, G)  # (N, m): the one O(N^2 m) step
        yt = self.yt[:, 0]
        if self.q:
            # GLS of yt and of every SNP on the covariates, with weights S
            sx0 = self.x0t * S[:, None]
            cho = linalg.cho_factor(_matrix_mult(self.x0t.T, sx0))
            r0 = yt - _matrix_mult(
                self.x0t, linalg.cho_solve(cho, _matrix_mult(sx0.T, yt), check_finite=False)
            )
            xr = xt - _matrix_mult(
                self.x0t, linalg.cho_solve(cho, _matrix_mult(sx0.T, xt), check_finite=False)
            )
        else:
            r0 = yt
            xr = xt
        sxr = xr * S[:, None]
        s = np.einsum("ij,ij->j", sxr, xr)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = _matrix_mult(sxr.T, r0) / s
            resid = r0[:, None] - xr * beta
            Q = np.einsum("ij,ij->j", resid * S[:, None], resid)
            p = self.q + 1
            sigma = Q / (float(self.N) - float(p))
            if not reml:  # as in association: Q / N (bug 9)
                sigma = sigma * (float(self.N) - p) / float(self.N)
            beta_var = sigma / s
            t = beta / np.sqrt(beta_var)
            bad = ~(s > 0) | ~np.isfinite(t)
            t[bad] = np.nan
            beta[bad] = np.nan
            beta_var[bad] = np.nan
            pv = 2.0 * stats.t.sf(np.abs(t), self.N - p)
        return AssociationResult(t, np.asarray(pv, dtype=np.float64), beta, beta_var)

    def tstat(
        self,
        beta: FloatArray,
        var: float | FloatArray,
        sigma: float | FloatArray,
        q: int,
        log: bool = False,
    ) -> tuple[np.float64, np.float64]:
        """t statistic and p-value of one coefficient.

        ``t = beta / sqrt(var * sigma)``, referred to ``t(N - q)``.  (It is
        an F-test with one numerator degree of freedom, so it equals a
        t-test.)

        Parameters
        ----------
        beta : (1,) array
            Coefficient estimate.
        var : float
            Diagonal entry of ``xx_inv`` for the coefficient.
        sigma : float
            ``sigma2`` to use.
        q : int
            Number of columns of the design (the degrees of freedom are
            ``N - q``).
        log : bool, default False
            Return the natural log of the two-sided p-value instead of the
            p-value (`log_pvalue`: accurate where the p-value underflows to
            0).  Before this release it returned ``2 + logsf``, which was
            wrong by ``2 - log 2`` (bug 5); no caller passed it.

        Returns
        -------
        t_stat, p_value : float
            numpy float64 each (the p-value is two-sided).

        Raises
        ------
        Exception
            If the inputs are not scalar-like (more than one test).
        """

        ts = beta / np.sqrt(var * sigma)
        # sf (the survival function) is accurate in the tail, unlike 1 - cdf
        if log:  # bug 5, fixed: it returned 2 + logsf
            ps = np.asarray(log_pvalue(ts, self.N - q))
        else:
            ps = 2.0 * (stats.t.sf(np.abs(ts), self.N - q))
        if not len(ts) == 1 or not len(ps) == 1:
            raise Exception("Something bad happened :(")
        return ts.sum(), ps.sum()

    def plot_fit(self, color: str = "b-", title: str = "") -> None:
        """Plot the normalised likelihood over the ``h`` grid (needs matplotlib).

        The grid log-likelihoods of the last `fit` are shifted by their maximum,
        exponentiated and normalised to sum to one, then drawn against
        heritability on the current matplotlib axes.  This approximates the
        posterior of ``h`` under a flat prior and shows whether there is one
        distinct maximum or several, and how wide it is.

        Parameters
        ----------
        color : str, default "b-"
            A matplotlib format string.
        title : str, default ""
            Plot title.

        Raises
        ------
        AttributeError
            If `fit` has not been called.
        ImportError
            If matplotlib is not installed (it is imported lazily).
        """
        import matplotlib.pyplot as pl

        mx = self.ll_grid.max()
        p = np.exp(self.ll_grid - mx)
        p = p / p.sum()

        pl.plot(self.h_grid, p, color)
        pl.xlabel("Heritability")
        pl.ylabel("Probability of data")
        pl.title(title)

    def mean_and_var(self) -> tuple[float, float]:
        """Mean and variance of ``h`` under the normalised grid likelihood.

        Uses the same weights as `plot_fit` (a flat prior on the grid), so it
        summarises how well ``h`` is determined.  Needs a prior call to `fit`.

        Returns
        -------
        mean, variance : float
            numpy float64 each.
        """

        mx = self.ll_grid.max()
        p = np.exp(self.ll_grid - mx)
        p = p / p.sum()

        mn = (self.h_grid * p).sum()
        vx = ((self.h_grid - mn) ** 2 * p).sum()

        return mn, vx
