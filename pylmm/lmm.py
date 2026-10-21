# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""Linear mixed models (LMM) for genome-wide association and heritability.

This module provides the numerical core of pylmm: a kinship matrix estimator
(`calculate_kinship`), the model class `LMM` (variance-component fit, Wald
test of one SNP or, batched, of a block of SNPs; the exact per-SNP refit
`LMM.fit_snps`; the likelihood-ratio and score tests), the scan `gwas` that
applies it to many SNPs, and the input checks they share
(`check_covariates`, `check_ids`, `snp_estimable`, `snp_estimable_columns`,
`check_test`).

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


def _orthonormal_columns(X: FloatArray) -> FloatArray:
    """An orthonormal basis (numpy's reduced QR) of the columns of ``X``.

    Raises `numpy.linalg.LinAlgError` (scipy's is the same class) if a column
    of ``X`` is, to rounding, a combination of the earlier ones (|R_jj| at most
    1e-12 of the column's norm): the design is rank-deficient.
    """
    if not X.shape[1]:
        return X
    q, r = np.linalg.qr(X)
    norms = np.sqrt((X * X).sum(axis=0))
    if (np.abs(np.diag(r)) <= 1e-12 * norms).any():
        raise np.linalg.LinAlgError("the design matrix (covariates and SNP) is rank-deficient")
    out: FloatArray = q
    return out


def _weighted_grams(U: FloatArray, W: FloatArray) -> FloatArray:
    """``(k, q, q)`` Gram matrices ``U' diag(W[:, j]) U`` of ``U`` (N, q), one per column of ``W``.

    One matrix product of the ``q^2`` column products of ``U`` with ``W`` (N, k).
    """
    n, q = U.shape
    pairs = (U[:, :, None] * U[:, None, :]).reshape(n, q * q)
    out: FloatArray = _matrix_mult(pairs.T, W).T.reshape(W.shape[1], q, q)
    return out


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


class SnpFitResult(NamedTuple):
    """The result of `LMM.fit_snps` (a tuple: ``h, ll, t, p, beta, beta_var = ...``).

    Each field is an ``(m,)`` float64 array, one entry per SNP.

    Attributes
    ----------
    h : (m,) array
        The heritability fitted with the SNP in the model.
    ll : (m,) array
        The REML or ML log-likelihood there (what ``fit(X=x)`` returns).
    t : (m,) array
        Wald t statistics at that ``h``.
    p : (m,) array
        Their two-sided p-values.
    beta : (m,) array
        SNP effect estimates.
    beta_var : (m,) array
        Their variances.
    """

    h: FloatArray
    ll: FloatArray
    t: FloatArray
    p: FloatArray
    beta: FloatArray
    beta_var: FloatArray


TESTS = ("wald", "lrt")
"""The association tests of `gwas`, `LMM.association_tests` and ``pylmm gwas --test``.

The score test (`LMM.score_batch`) is a library function only, for comparison
with GEMMA's ``-lmm 3``: it is a monotone function of the null-fit Wald
statistic, so a scan of it adds nothing to the default one."""

# _reduced_ml_model: a singular value of the covariates' rows outside K's null
# block below this is 0 (those columns are orthonormal, so the values are in [0, 1];
# a covariate direction exactly inside the block, like the intercept of a centred K,
# gives ~1e-15).
_NULL_SPAN_TOL = 1e-8

# Safeguarded Newton-Raphson of LMM.fit_snps: iteration cap (bisection alone needs ~34
# halvings of a 0.01 grid interval to reach the tolerance) and the step/bracket
# tolerance in h (Brent's xatol in fit is 1e-10).
_NEWTON_MAXITER = 100
_NEWTON_XTOL = 1e-12
# LMM.fit_snps evaluates its per-SNP terms in chunks of this many columns (memory:
# about 15 n _SNP_CHUNK floats at a time; see LMM._snp_terms_at).
_SNP_CHUNK = 256


def check_test(test: str, refit: bool = False, legacy_fit: bool = False) -> None:
    """Raise ValueError unless ``test`` is one of `TESTS` and fits the other options.

    With ``legacy_fit`` (pylmm 1.0.0) only the Wald test exists.  ``refit`` is
    accepted with the LRT, which always fits each SNP's model (by ML).

    Examples
    --------
    >>> from pylmm.lmm import check_test
    >>> check_test("lrt")
    >>> check_test("score")
    Traceback (most recent call last):
    ...
    ValueError: test must be one of wald, lrt, got 'score' (the score test is LMM.score_batch)
    """
    if test not in TESTS:
        raise ValueError(
            "test must be one of %s, got %r%s"
            % (
                ", ".join(TESTS),
                test,
                " (the score test is LMM.score_batch)" if test == "score" else "",
            )
        )
    if test != "wald" and legacy_fit:
        raise ValueError(
            "legacy_fit reproduces pylmm 1.0.0, which had only the Wald test; test=%r needs "
            "the default fit" % test
        )


def _lrt_result(
    ll1: FloatArray, ll0: float, beta: FloatArray, beta_var: FloatArray
) -> AssociationResult:
    """The LRT of alternative log-likelihoods ``ll1`` against ``ll0``: ``(t, p, beta, beta_var)``.

    ``stat = max(2 (ll1 - ll0), 0)`` (negative only by rounding), ``t =
    sign(beta) sqrt(stat)``, ``p`` from ``chi2(1)``; nan where ``ll1`` or
    ``beta`` is not finite.
    """
    with np.errstate(invalid="ignore"):
        stat = np.maximum(2.0 * (np.asarray(ll1, dtype=np.float64) - ll0), 0.0)
        t = np.sign(beta) * np.sqrt(stat)
        bad = ~np.isfinite(ll1) | ~np.isfinite(beta)
        t[bad] = np.nan
        stat[bad] = np.nan
    pv = stats.chi2.sf(stat, 1)
    return AssociationResult(t, np.asarray(pv, dtype=np.float64), beta, beta_var)


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
    test: str = "wald",
) -> tuple[list[float], list[float]]:
    """Scan SNPs with the mixed model (a Wald or likelihood-ratio test per SNP).

    For each column of ``X`` the model ``y = X0 b + x a + u + e`` (see the
    module docstring) is tested for ``a = 0``.  By default the variance
    components ``(h, sigma2)`` are fitted once without a SNP (the null fit)
    and reused for every SNP (the EMMA/"P3D" approximation); with
    ``refit=True`` they are refitted for each SNP (exactly, see
    `LMM.fit_snps`).  ``test`` chooses the test (`LMM.association_tests`):
    the Wald test (default) or the likelihood-ratio test (ML fits with and
    without each SNP, `LMM.lrt_batch`).  Individuals with a missing
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
        REML for the null fit, every refit and the Wald test; False uses
        ML throughout (``sigma2 = Q / n`` in the test).  Ignored by the
        likelihood-ratio test, which is always ML.
    refit : bool, default False
        Refit ``h`` and ``sigma2`` with each SNP in the model, for the Wald
        test (the likelihood-ratio test always does; ``refit`` changes
        nothing there).  With ``batch=True`` (the default) the exact per-SNP fit of
        `LMM.fit_snps`, a block of SNPs at a time (Newton-Raphson on the
        analytic derivatives; the maximum ``fit(X=x)`` finds, 8-46
        times faster); with ``batch=False`` or ``legacy_fit=True``,
        ``LMM.fit(X=x)`` for each SNP (with ``legacy_fit``, pylmm 1.0.0's
        search).
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
        pylmm 1.0.0 (and refits with ``LMM.fit(X=x)``).  SNPs with missing
        calls are always tested one at a time.
    block_size : int, default `pylmm.genotypes.DEFAULT_BLOCK_SIZE` (1024)
        Number of SNPs per block with ``batch=True``.  A block takes about
        6-8 arrays of ``n * block_size`` floats of memory, also with
        ``refit=True`` and ``test="lrt"`` (`LMM.fit_snps` works on at most 256
        SNPs' per-SNP terms at a time; measured 4.6 arrays at n = 2000).
    legacy_fit : bool, default False
        Fit ``h`` as pylmm 1.0.0 did (see `LMM`): every null fit and refit,
        and so every statistic, as in 1.0.0.  With ``batch=False`` the
        results are those of pylmm 1.0.0.  Only with ``test="wald"``.
    test : {"wald", "lrt"}, default "wald"
        The test of each SNP: ``"wald"``, the Wald t test (pylmm's test;
        with ``refit``, GEMMA's ``-lmm 1``); ``"lrt"``, the likelihood-ratio
        test of ML fits with and without the SNP, ``chi2(1)``
        (`LMM.lrt_batch`: with K's null directions that the covariates span
        dropped, so that ML has a maximum).  The score test is
        `LMM.score_batch` (library only).

    Returns
    -------
    t_stats : list of length m
        The test statistic of each SNP (numpy float64 items): the Wald t
        statistic (``t = beta / se``), or for the LRT the signed square
        root of the statistic (the sign of ``beta``).  ``nan``
        for a SNP that cannot be tested: all missing, constant, collinear with
        ``X0``, or with too few called individuals.
    p_values : list of length m
        Two-sided p-value of each SNP: from ``t(n_s - q - 1)`` (Wald;
        ``n_s`` is the number of individuals used for that SNP) or
        ``chi2(1)`` (LRT); ``nan`` where ``t_stats`` is.  If no individual has
        a phenotype, both lists are all ``nan``.

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
        For an unknown ``test`` or ``test="lrt"`` with ``legacy_fit``; an
        infinite or constant phenotype, ``X`` or ``X0`` with the wrong
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
    check_test(test, refit, legacy_fit)
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
    _fit_null(L, test, reml, refit)
    Q0 = linalg.qr(X0, mode="economic")[0] if X0.shape[1] else np.zeros((X0.shape[0], 0))

    t_stats: list[float] = [np.nan] * m
    p_values: list[float] = [np.nan] * m
    counts: dict[str, int] = {}

    n = X.shape[0]
    m = X.shape[1]
    # pylmm 1.0.0's refit (legacy_fit) is one SNP at a time
    use_batch = batch and not (refit and legacy_fit)

    for start in range(0, m, block_size):
        stop = min(start + block_size, m)
        if use_batch:
            complete = ~np.isnan(X[:, start:stop]).any(axis=0)
        else:
            complete = np.zeros(stop - start, dtype=bool)
        for j in start + np.where(~complete)[0]:  # one SNP at a time
            res = _gwas_one_snp(X[:, [j]], Y, K, X0, Q0, L, reml, refit, counts, test, batch)
            if res is not None:
                t_stats[j], p_values[j] = res
        # the complete, estimable SNPs of this block, tested together
        cand = start + np.where(complete)[0]
        batched = [int(i) for i in cand[snp_estimable_columns(X[:, cand], X0, Q0)]]
        if batched:
            if test == "wald" and not refit:
                r = L.association_batch(X[:, batched], reml=reml)
            else:
                r = L.association_tests(X[:, batched], test, reml, refit)
            for k, i in enumerate(batched):
                if np.isfinite(r.t[k]):
                    t_stats[i], p_values[i] = r.t[k], r.p[k]
                elif test == "wald" and not refit:  # not expected for an estimable SNP
                    t_stats[i], p_values[i] = L.association(X[:, i], reml=reml)
                else:
                    r1 = L.association_tests(X[:, [i]], test, reml, refit, batch=False)
                    t_stats[i], p_values[i] = r1.t[0], r1.p[0]

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
    test: str = "wald",
    batch: bool = True,
) -> tuple[np.float64, np.float64] | None:
    """Test one SNP column ``x`` (n, 1) of `gwas` on its own; None if it cannot be tested.

    A SNP with missing calls is tested on the called individuals (a new
    `LMM` on that subset); a complete one with ``L`` (refitted if ``refit``).
    The Wald test without ``refit`` or with the per-SNP refit (``batch=False``
    or ``legacy_fit``) is pylmm 1.0.0's code; everything else goes through
    `LMM.association_tests`.
    """
    old = test == "wald" and (not refit or not batch or L.legacy_fit)
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
        if not old:
            _fit_null(ls, test, reml, refit)
            r = ls.association_tests(xs, test, reml, refit, batch)
            return r.t[0], r.p[0]
        if refit:
            ls.fit(X=xs, reml=reml)
        else:
            ls.fit(reml=reml)
        return ls.association(xs, reml=reml)
    if not snp_estimable(x, X0, Q=Q0):
        return None
    if not old:
        r = L.association_tests(x, test, reml, refit, batch)
        return r.t[0], r.p[0]
    if refit:
        L.fit(X=x, reml=reml)
    return L.association(x, reml=reml)


def _fit_null(L: LMM, test: str, reml: bool, refit: bool) -> None:
    """The null fit a scan with ``test`` needs: the LRT's (reduced) ML fit, none for the refit."""
    if test == "lrt":
        L.lrt_null_fit()
    elif not refit:
        L.fit(reml=reml)


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
        self._snp_fallbacks = 0  # fit_snps: brackets that fell back to the scalar search
        self._lrt_cache: LMM | None = None  # _lrt_model
        self._lrt_null: tuple[float, float] | None = None  # lrt_null_fit
        self._dropped = 0  # directions _reduced_ml_model dropped (on the reduced model)
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
        # An orthonormal basis of the rotated covariates: the likelihood is
        # invariant to X0 -> X0 A, and on this basis its weighted normal
        # equations stay well conditioned whatever the covariates' scale.
        self._x0t_ortho = _orthonormal_columns(self.x0t)

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
        if not p and not self.legacy_fit:
            # no covariates (e.g. the LRT's reduced model of an intercept-only fit):
            # nothing to solve (scipy < 1.14 cannot invert a 0 x 0 matrix)
            beta = np.zeros((0, 1))
            xx_inv = np.zeros((0, 0))
            xx = np.zeros((0, 0))
            Q = float(np.dot(self.yt[:, 0] * S, self.yt[:, 0]))
        elif self.legacy_fit or not p:
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
            If None, the covariates ``x0t`` are used (the likelihood is
            computed on an orthonormal basis of them, which leaves it
            unchanged and keeps it accurate for badly scaled covariates;
            ``beta`` and ``xx_inv`` are for ``x0t``).  If given and
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

        basis: FloatArray | None = None  # the default mode's orthonormal basis of X
        if X is None:
            X = self.x0t
            if not self.legacy_fit:
                basis = self._x0t_ortho
        elif stack:
            self.x0t_stack[:, (self.q)] = _matrix_mult(
                self.kve.T, _snp_column(X, self.kve.shape[0])
            )[:, 0]
            X = self.x0t_stack
        else:
            X = np.asarray(X, dtype=np.float64)
        given = X  # the design in the caller's terms
        if not self.legacy_fit and X.shape[1]:
            if basis is None:
                xtx = _matrix_mult(X.T, X)
                if np.abs(xtx - np.eye(X.shape[1])).max() <= 1e-10:
                    basis = X  # already orthonormal (fit passes one)
                else:
                    basis = _orthonormal_columns(X)
            X = basis

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

        if X is not given:  # report beta and (X'SX)^-1 for the design as given
            beta, sigma, _, xx_inv, _ = self.ml_solution(h, given)
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

    def _warn_boundary(self, h: float, p: int) -> None:
        """Warn when the residual variance is negligible and ``K`` is rank-deficient beyond the design.

        With no residual variance the data lie in the span of ``K``: only
        about ``rank(K) - p`` directions carry information, but ``sigma2`` and
        the Wald test use ``N - p`` degrees of freedom, so the test is
        anti-conservative (a known model boundary, not a search error).
        "Negligible" is scale-free: the residual variance ``(1 - h)`` is at
        most 1e-3 of the genetic variance along a typical direction of ``K``,
        ``h * median(non-null eigenvalues)`` (for a K of the usual scale,
        h >= 0.999).
        """
        null = self.kva <= 1e-6
        if int(null.sum()) <= p or null.all():
            return
        typical = float(np.median(self.kva[~null]))
        if (1.0 - h) <= 1e-3 * h * typical:
            warnings.warn(
                "h is at its upper boundary (the residual variance is negligible) and the "
                "kinship matrix is rank-deficient: sigma^2 and the Wald test's N - p degrees "
                "of freedom overstate the information (p-values too small); see the "
                "estimate-heritability and interpret-gwas-results skills",
                stacklevel=3,
            )

    def _check_ml_ill_posed(self) -> bool:
        """Whether the ML likelihood is unbounded as ``h -> 1`` (bug 17), cached.

        In the rotated basis the null directions of ``K`` (the eigenvalues
        raised to the 1e-6 floor) have variance ``sigma2 * (1 - h)``.  If the
        covariates can fit ``y`` exactly in all of them -- the rows of the
        rotated ``X0`` for those directions have full row rank -- the
        residual there is zero for every ``h``, and the ML log-likelihood
        gains ``-d/2 log(1 - h)`` (``d`` null directions) without bound; the
        exact condition, checked here.  The common case is a centred kinship
        (``K 1 = 0``, as `calculate_kinship` builds) from more SNPs than
        individuals, whose null space is exactly the intercept.  Otherwise the
        residual in a null direction makes the likelihood fall to -inf at
        ``h = 1`` (bounded).  REML is not affected.
        """
        if self._ml_ill_posed is None:
            null = self.kva <= 1e-6
            d = int(null.sum())
            ill = False
            if d and d <= self.q:
                # rows of an orthonormal basis of the rotated X0 (well scaled)
                rows = self._x0t_ortho[null, :]
                ill = bool(np.linalg.matrix_rank(rows, tol=1e-6) == d)
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
            ``K`` is singular and the covariates can fit ``y`` exactly along
            its null space (the rotated covariates' rows for the null
            directions have full row rank), as for a centred kinship built
            from more SNPs than individuals with an intercept (bug 17).  The
            likelihood then grows without bound as ``h -> 1``, held back only
            by the 1e-6 eigenvalue floor, so the ML ``h`` (an interior local
            maximum or a spike at 1) is unreliable; use REML.
            When the residual variance is negligible (``1 - h`` at most 1e-3
            of ``h`` times the median non-null eigenvalue of ``K``: ``h >=
            0.999`` for a K of the usual scale) and ``K`` has more floored
            (null) eigenvalues than the design has columns: with no residual variance the data
            lie in the span of ``K``, and ``sigma2`` and the Wald test's
            ``N - p`` degrees of freedom overstate the information (e.g.
            strain means expanded to replicate animals).  Neither warning
            with ``legacy_fit``; neither changes a number.

        Notes
        -----
        The likelihood is evaluated on an orthonormal basis of the (rotated)
        design: it is invariant to ``X -> X A``, and so the fitted ``h`` does
        not depend on how a covariate is scaled or centred, also near
        ``h = 1`` where the weights span many orders of magnitude.  ``beta``
        and ``sigma`` are then computed in the given parametrisation
        (`ml_solution`).  A grid of step 0.01 finds every peak of the profile
        that is at least one grid step away from the others; two peaks
        closer than that are refined as one.  ``h = 1`` is returned when the profile
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
            self.x0t_stack[:, (self.q)] = _matrix_mult(
                self.kve.T, _snp_column(X, self.kve.shape[0])
            )[:, 0]
            design = self.x0t_stack

        if self.legacy_fit:
            H = np.array(range(ngrids)) / float(ngrids)
        else:
            H = np.arange(ngrids + 1) / float(ngrids)
        # The default search evaluates the likelihood on an orthonormal basis of
        # the design (the same likelihood: it is invariant to X -> X A), so that
        # a badly scaled covariate cannot spoil it where the weights span many
        # orders of magnitude (h near 1 with a singular K).
        search = design if self.legacy_fit else _orthonormal_columns(design)
        grid_ll = np.array([self.log_likelihood(h, search, stack=False, reml=reml).ll for h in H])
        self.ll_grid = grid_ll

        if self.legacy_fit:
            hmax = self._get_max(H, design, reml)
            L, beta, sigma, _ = self.log_likelihood(hmax, design, stack=False, reml=reml)
        else:
            hmax = self._get_best(H, search, reml)
            L = self.log_likelihood(hmax, search, stack=False, reml=reml).ll
            beta, sigma, _, _, _ = self.ml_solution(hmax, design)  # by QR, in X's terms
            self._warn_boundary(float(hmax), design.shape[1])
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
            self.x0t_stack[:, (self.q)] = _matrix_mult(
                self.kve.T, _snp_column(X, self.kve.shape[0])
            )[:, 0]
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

    def _snp_block(self, X: ArrayLike) -> FloatArray:
        """``X`` (N, m) or (N,) as an (N, m) float64 array; ValueError on another shape."""
        G = np.asarray(X, dtype=np.float64)
        if G.ndim == 1:
            G = G.reshape(-1, 1)
        n = self.kve.shape[0]  # = N, except on the LRT's reduced model
        if G.ndim != 2 or G.shape[0] != n:
            raise ValueError(
                "X must have shape (%d, m) or (%d,) (one row per individual with a "
                "phenotype), got %s" % (n, n, np.shape(X))
            )
        return G

    def _residualize(self, h: float, X: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(S, r0, xr)`` at ``h``: the weights, the phenotype's and each SNP's GLS residual.

        ``S = 1 / (h * kva + 1 - h)``; ``r0`` (N,) is the weighted
        least-squares residual of ``yt`` on the rotated covariates and ``xr``
        (N, m) that of every rotated column of ``X`` (raw SNPs, as for
        `association_batch`).  Shared by `association_batch` and `score_batch`.
        """
        G = self._snp_block(X)
        S = 1.0 / (h * self.kva + (1.0 - h))
        xt = _matrix_mult(self.kve.T, G)  # (N, m): the one O(N^2 m) step
        yt = self.yt[:, 0]
        if self.q:
            # GLS of yt and of every SNP on the covariates, with weights S.  The
            # residuals depend only on span(X0), so the default mode uses its
            # orthonormal basis: x0t'Sx0t of a badly scaled covariate is far too
            # ill-conditioned at h near 1 (weights up to 1e6 on K's null space).
            x0 = self.x0t if self.legacy_fit else self._x0t_ortho
            sx0 = x0 * S[:, None]
            cho = linalg.cho_factor(_matrix_mult(x0.T, sx0))
            r0 = yt - _matrix_mult(
                x0, linalg.cho_solve(cho, _matrix_mult(sx0.T, yt), check_finite=False)
            )
            xr = xt - _matrix_mult(
                x0, linalg.cho_solve(cho, _matrix_mult(sx0.T, xt), check_finite=False)
            )
        else:
            r0 = yt
            xr = xt
        return S, r0, xr

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
        the intercept) both stay accurate, also at ``h = 1`` with a centred
        ``K`` (weights up to 1e6): `association` solves the weighted least
        squares by QR, and the residuals here are taken on an orthonormal
        basis of the covariates (about 3e-13 of the largest ``t`` off an
        accurate reference for a date ~2e7; with a centred kinship, batched
        vs per SNP <= 1.3e-15 of max ``|t|`` over 360 scans, 228 of them at
        ``h = 1``).  With a kinship that has many null directions (e.g.
        strain means expanded to replicate animals) at ``h >= 1 - 1e-7``,
        where the weights reach 1e6 on all of them, the per-SNP QR drifts by
        up to ~3e-10 of max ``|t|``; the batched values are then the
        accurate ones (about 2e-12 off a 50-digit reference).  With
        ``legacy_fit=True`` `association` inverts ``X'SX`` as pylmm 1.0.0
        did, loses precision there, and the two can differ by up to a few
        1e-6 relative; the batched values are then the more accurate ones
        (against a 40-digit reference, about 3e-11 vs 3e-7 off).  Centring
        or scaling such covariates makes both paths accurate.  Columns are
        not checked for being constant or collinear with ``X0`` (use
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
        S, r0, xr = self._residualize(h, X)
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

    def score_batch(
        self, X: ArrayLike, h: float | None = None, reml: bool = True
    ) -> AssociationResult:
        """Score tests of many SNPs at the null fit, for comparison with GEMMA's ``-lmm 3``.

        A library function only, not a test of `gwas` or ``pylmm gwas --test``:
        the score statistic is a monotone function of the null-fit Wald
        statistic of `association_batch` (Notes), so it ranks the SNPs as the
        default scan does, adds no information to it and costs as much, and
        its p-values have a floor at small ``N`` (Notes).  Use it to reproduce
        a GEMMA ``-lmm 3`` analysis (``reml=False``).

        For each column ``x`` of ``X`` the score statistic of ``a = 0`` in
        ``y = X0 b + x a + u + e`` is evaluated at the null model's ``h`` and
        ``sigma2``; nothing is fitted with the SNP.  With ``P0`` the null
        model's projection at ``h`` (``P0 v`` is the weighted GLS residual of
        ``v`` on the covariates)::

            sigma2_0 = y'P0y / (N - q)   (REML; y'P0y / N with reml=False)
            stat     = (x'P0y)^2 / (sigma2_0 x'P0x)

        ``t = sign(x'P0y) * sqrt(stat)`` is referred to ``t(N - q - 1)``, i.e.
        ``stat`` to ``F(1, N - q - 1)``, as GEMMA does.  Call `fit` first
        (the null model, REML or ML) unless ``h`` is given.

        Parameters
        ----------
        X : (N, m) array, or (N,) for one SNP
            SNPs, as for `association_batch` (no missing values).
        h : float, optional
            Heritability of the null model.  Default: ``self.opt_h``.
        reml : bool, default True
            The ``sigma2_0`` denominator: ``N - q`` (REML) or ``N`` (ML,
            GEMMA's score test).

        Returns
        -------
        AssociationResult
            ``(t, p, beta, beta_var)`` of ``(m,)`` arrays: the signed root of
            the score statistic, its two-sided p-value from ``t(N - q - 1)``,
            the one-step estimate ``beta = x'P0y / x'P0x`` and its variance
            under the null, ``sigma2_0 / x'P0x`` (so ``t = beta /
            sqrt(beta_var)``).

        Notes
        -----
        At the same ``h`` the score statistic is a monotone function of the
        Wald statistic of `association_batch`: with ``u = (x'P0y)^2 / (x'P0x
        y'P0y)``, REML gives ``t_score^2 = (N - q) u`` and ``t_wald^2 = (N - q -
        1) u / (1 - u)``.  So the score scan ranks the SNPs exactly as the
        default Wald scan does; its p-values are larger than the Wald test's
        where ``|t_score| > 1`` (much larger for strong signals: ``t_score^2 <=
        N - q``) and slightly smaller below.  The cost is that of
        `association_batch`.  (ML: ``t_score^2 = N u``, ``t_wald^2 = N u / (1 -
        u)``.)

        Since ``u < 1``, ``t_score^2 < N - q`` (``N`` under ML): the p-value
        cannot fall below ``2 P(T > sqrt(N - q))``, ``T ~ t(N - q - 1)``.  With
        an intercept alone that floor is about 9.7e-6 at ``N = 30``, 4.4e-8 at
        45, 7.4e-9 at 50 and 1.3e-15 at 94: below ``N`` ~ 45 (e.g. strain means
        of a small inbred panel) no SNP can reach 5e-8.  In S9
        (``sims/studies/s9_tests.md``) the test is conservative in the tail at
        ``N = 100``.

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
        >>> score = model.score_batch(W[:, :50])
        >>> wald = model.association_batch(W[:, :50])
        >>> bool(score.p[0] < 1e-3 and score.p[0] >= wald.p[0])
        True
        >>> bool((np.argsort(score.p) == np.argsort(wald.p)).all())   # the same ranking
        True
        """
        if h is None:
            h = self.opt_h
        S, r0, xr = self._residualize(h, X)
        sxr = xr * S[:, None]
        s = np.einsum("ij,ij->j", sxr, xr)  # x'P0x
        xy = _matrix_mult(sxr.T, r0)  # x'P0y
        Q0 = float(np.dot(r0 * S, r0))  # y'P0y
        sigma0 = Q0 / ((float(self.N) - float(self.q)) if reml else float(self.N))
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = xy / s
            beta_var = sigma0 / s
            t = beta / np.sqrt(beta_var)
            bad = ~(s > 0) | ~np.isfinite(t)
            t[bad] = np.nan
            beta[bad] = np.nan
            beta_var[bad] = np.nan
            pv = 2.0 * stats.t.sf(np.abs(t), self.N - self.q - 1)
        return AssociationResult(t, np.asarray(pv, dtype=np.float64), beta, beta_var)

    # -- the exact per-SNP fit (fit_snps) ---------------------------------------

    def _snp_rotate(self, X: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(xp, yp, nrm2)``: rotated SNPs and phenotype with their part in span(X0) removed.

        ``xp = (I - U U') kve' X`` and ``yp = (I - U U') yt`` for the
        orthonormal basis ``U`` of the rotated covariates (unweighted).  Every
        GLS quantity of the model ``[X0, x]`` depends on ``x`` and ``y`` only
        modulo span(X0), so `fit_snps` works with these: they have no
        component along a heavily weighted direction that the covariates
        span (e.g. the intercept at ``h = 1`` with a centred ``K``), which
        keeps the normal-equation forms of `_snp_grid_ll` accurate.
        ``nrm2 = |xp|^2`` enters the REML determinant of an orthonormal design;
        it is set to 0 where it is at the rounding level of the column
        (``_ESTIMABLE_FLOOR`` of ``|kve' x|^2``, as in `snp_estimable`).
        """
        G = self._snp_block(X)
        xt = _matrix_mult(self.kve.T, G)
        yt = self.yt[:, 0]
        U = self._x0t_ortho
        raw = np.einsum("ij,ij->j", xt, xt)
        if self.q:
            xt = xt - _matrix_mult(U, _matrix_mult(U.T, xt))
            yt = yt - _matrix_mult(U, _matrix_mult(U.T, yt))
        nrm2 = np.einsum("ij,ij->j", xt, xt)
        nrm2[~(nrm2 > _ESTIMABLE_FLOOR * raw)] = 0.0
        return xt, yt, nrm2

    def _snp_ll(
        self,
        Q1: FloatArray,
        logdet_v: FloatArray,
        logdet_g: FloatArray,
        xx0: FloatArray,
        nrm2: FloatArray,
        reml: bool,
    ) -> FloatArray:
        """The (RE)ML log-likelihood of ``[X0, x]``, as `log_likelihood` gives it for `fit`.

        ``Q1`` is the weighted residual sum of squares, ``logdet_v =
        sum(log(h kva + 1 - h))``, ``logdet_g = log|U'SU|`` for the
        covariates' orthonormal basis ``U`` and ``xx0 = x'P0x``; then
        ``log|B'SB| = logdet_g + log(xx0) - log(nrm2)`` for the orthonormal
        basis ``B`` of ``[U, x]``, the design `fit` evaluates.
        """
        n = float(self.N)
        p = float(self.q + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = -0.5 * (n * np.log(2 * np.pi) + logdet_v + n + n * np.log(Q1 / n))
            if reml:
                ll = ll + 0.5 * (
                    p * np.log(2.0 * np.pi * Q1 / (n - p)) - (logdet_g + np.log(xx0) - np.log(nrm2))
                )
        out: FloatArray = ll
        return out

    def _snp_grid_ll(
        self, H: FloatArray, xp: FloatArray, yp: FloatArray, nrm2: FloatArray, reml: bool
    ) -> FloatArray:
        """``(len(H), m)`` log-likelihoods of every SNP's model on the grid ``H``.

        The weighted inner products of every SNP at every grid point are a
        few matrix products ``S_grid @ (N, m)`` (BLAS 3); the GLS on the
        covariates is then the normal equations of a ``q x q`` system per grid
        point (accurate here because ``xp`` and ``yp`` have no part in
        span(X0), see `_snp_rotate`).  Used only to bracket the maxima; the
        values that are compared and returned come from `_snp_terms`.
        """
        U = self._x0t_ortho
        D = H[:, None] * self.kva[None, :] + (1.0 - H)[:, None]  # (nh, N)
        S = 1.0 / D
        logdet_v = np.log(D).sum(axis=1)
        xx0 = _matrix_mult(S, xp * xp)  # (nh, m)
        xy0 = _matrix_mult(S, xp * yp[:, None])
        yy0 = _matrix_mult(S, yp * yp)  # (nh,)
        logdet_g = np.zeros(len(H))
        if self.q:
            G = _weighted_grams(U, S.T)
            chol = np.linalg.cholesky(G)
            logdet_g = 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
            g_inv = np.linalg.inv(G)
            cx = np.stack([_matrix_mult(S * U[:, k], xp) for k in range(self.q)], axis=1)
            cy = _matrix_mult(S, U * yp[:, None])  # (nh, q)
            zx_g = np.einsum("hkl,hlm->hkm", g_inv, cx)
            zy = np.einsum("hkl,hl->hk", g_inv, cy)
            xx0 = xx0 - np.einsum("hkm,hkm->hm", cx, zx_g)
            xy0 = xy0 - np.einsum("hkm,hk->hm", cx, zy)
            yy0 = yy0 - np.einsum("hk,hk->h", cy, zy)
        with np.errstate(divide="ignore", invalid="ignore"):
            Q1 = yy0[:, None] - xy0 * xy0 / xx0
        return self._snp_ll(Q1, logdet_v[:, None], logdet_g[:, None], xx0, nrm2[None, :], reml)

    def _snp_terms(
        self,
        h: FloatArray,
        xp: FloatArray,
        yp: FloatArray,
        nrm2: FloatArray,
        reml: bool,
        derivatives: bool = False,
    ) -> dict[str, FloatArray]:
        """Log-likelihood (and its first two derivatives in ``h``) of ``[X0, x_k]`` at ``h_k``.

        ``h`` is (k,) and ``xp`` (N, k): each column at its own ``h``.  The
        GLS residuals are formed directly (as in `association_batch`), not
        from normal equations.  With ``V = diag(h kva + 1 - h)``, ``D =
        dV/dh = diag(kva - 1)``, ``P`` the projection of the model with the
        SNP (``P y`` is the weighted residual of ``y``) and ``Q = y'Py``, the
        derivatives (Zhou & Stephens 2012, written in ``h``) are::

            dQ/dh = -y'PDPy                  d2Q/dh2 = 2 y'PDPDPy
            ML:   l'  = -1/2 [tr(V^-1 D) + N (log Q)']
                  l'' = -1/2 [-tr(V^-1 D V^-1 D) + N (log Q)'']
            REML: l'  = -1/2 [tr(PD) + (N - p) (log Q)']
                  l'' = -1/2 [-tr(PDPD) + (N - p) (log Q)'']

        The SNP's terms come from the null model's by the one-column update
        ``P1 = P0 - P0 x x'P0 / x'P0x``.  Returns a dict of (k,) arrays:
        ``ll``, ``beta``, ``xx0`` (``x'P0x``), ``Q1`` and, with
        ``derivatives``, ``d1`` and ``d2``.
        """
        U = self._x0t_ortho
        q = self.q
        kva = self.kva[:, None]
        D = h[None, :] * kva + (1.0 - h)[None, :]  # (N, k)
        S = 1.0 / D
        a = kva - 1.0  # dV/dh
        G = np.zeros((len(h), 0, 0))
        if q:
            G = _weighted_grams(U, S)  # (k, q, q)
            chol = np.linalg.cholesky(G)
            logdet_g = 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
        else:
            logdet_g = np.zeros(len(h))

        def resid(V: FloatArray) -> FloatArray:
            """``V`` (N, k) minus its weighted projection on span(U), column by column."""
            if not q:
                return V
            rhs = _matrix_mult(U.T, S * V).T[:, :, None]
            b = np.linalg.solve(G, rhs)[:, :, 0]  # (k, q)
            out: FloatArray = V - _matrix_mult(U, b.T)
            return out

        ry = resid(np.repeat(yp[:, None], xp.shape[1], axis=1))
        rx = resid(xp)
        srx = S * rx
        xx0 = (rx * srx).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = (srx * ry).sum(axis=0) / xx0
            rw = ry - rx * beta  # the residual of y - beta x on X0
            py1 = S * rw  # P1 y
            Q1 = (rw * py1).sum(axis=0)
            logdet_v = np.log(D).sum(axis=0)
            out = {
                "ll": self._snp_ll(Q1, logdet_v, logdet_g, xx0, nrm2, reml),
                "beta": beta,
                "xx0": xx0,
                "Q1": Q1,
            }
            if not derivatives:
                return out
            n = float(self.N)
            p = float(q + 1)
            a_s = a * S
            ypdpy = (a * py1 * py1).sum(axis=0)
            g = (a * srx * py1).sum(axis=0)  # x'P0 D P1 y
            xx1 = (a * srx * srx).sum(axis=0)  # x'P0 D P0 x
            z = a * py1  # D P1 y
            zx = a * srx  # D P0 x
            rz, rzx = resid(z), resid(zx)
            ww2 = (S * rz * rz).sum(axis=0)  # (D P1 y)' P0 (D P1 y)
            xx2 = (S * rzx * rzx).sum(axis=0)  # (D P0 x)' P0 (D P0 x)
            dq = -ypdpy
            d2q = 2.0 * (ww2 - g * g / xx0)
            dlogq = dq / Q1
            d2logq = d2q / Q1 - dlogq * dlogq
            if reml:
                tr_pd0 = a_s.sum(axis=0)
                # tr(P0 D P0 D) = sum (aS)^2 - 2 tr(G^-1 F2) + tr((G^-1 F1)^2): at h
                # near 1 a floored direction that the covariates span contributes
                # ~1e12 to each term and cancels, leaving l'' with a relative error
                # of ~1e-5 there (measured: tests/helpers.py DERIV_H1_RTOL).  Only
                # Newton's step uses l''; l' and the log-likelihood, which pick the
                # maximum, have no such term.
                tr_pdpd0 = (a_s * a_s).sum(axis=0)
                if q:
                    F1 = _weighted_grams(U, a_s * S)
                    F2 = _weighted_grams(U, a_s * a_s * S)
                    g_f1 = np.linalg.solve(G, F1)
                    tr_pd0 = tr_pd0 - np.einsum("jkk->j", g_f1)
                    tr_pdpd0 = (
                        tr_pdpd0
                        - 2.0 * np.einsum("jkk->j", np.linalg.solve(G, F2))
                        + np.einsum("jkl,jlk->j", g_f1, g_f1)
                    )
                tr_pd1 = tr_pd0 - xx1 / xx0
                tr_pdpd1 = tr_pdpd0 - 2.0 * xx2 / xx0 + (xx1 / xx0) ** 2
                out["d1"] = -0.5 * (tr_pd1 + (n - p) * dlogq)
                out["d2"] = -0.5 * (-tr_pdpd1 + (n - p) * d2logq)
            else:
                out["d1"] = -0.5 * (a_s.sum(axis=0) + n * dlogq)
                out["d2"] = -0.5 * (-(a_s * a_s).sum(axis=0) + n * d2logq)
        return out

    def _snp_terms_at(
        self,
        h: FloatArray,
        cols: NDArray[np.intp],
        xp: FloatArray,
        yp: FloatArray,
        nrm2: FloatArray,
        reml: bool,
        derivatives: bool = False,
    ) -> dict[str, FloatArray]:
        """`_snp_terms` of the columns ``cols`` of ``xp`` at ``h``, `_SNP_CHUNK` at a time.

        `_snp_terms` keeps about 15 ``(N, k)`` arrays alive at once; in chunks of
        `_SNP_CHUNK` columns the peak stays near ``15 N _SNP_CHUNK`` floats
        whatever the number of brackets or candidates (several per SNP).
        """
        parts = []
        for a in range(0, len(cols), _SNP_CHUNK):
            c = cols[a : a + _SNP_CHUNK]
            parts.append(
                self._snp_terms(h[a : a + _SNP_CHUNK], xp[:, c], yp, nrm2[c], reml, derivatives)
            )
        if not parts:
            return self._snp_terms(h, xp[:, cols], yp, nrm2[cols], reml, derivatives)
        return {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}

    def _newton(
        self,
        lo: FloatArray,
        hi: FloatArray,
        cols: NDArray[np.intp],
        xp: FloatArray,
        yp: FloatArray,
        nrm2: FloatArray,
        reml: bool,
        x0: FloatArray | None = None,
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        """Safeguarded Newton-Raphson for a root of ``l'(h)`` in each bracket ``(lo, hi)``.

        Each bracket has ``l'(lo) > 0 > l'(hi)`` (SNP ``cols[k]``); the
        search starts at ``x0`` (default: the midpoint).  Every
        iteration takes the Newton step where ``l'' < 0``, the step stays
        inside the current bracket and it shrinks fast enough, and a
        bisection otherwise; the bracket is narrowed by the sign of ``l'``
        (Numerical Recipes' ``rtsafe``), all brackets at once.  Returns the
        roots and whether each converged (a step or bracket below
        `_NEWTON_XTOL`).
        """
        x = 0.5 * (lo + hi) if x0 is None else x0.copy()
        dx_old = hi - lo
        dx = dx_old.copy()
        done = np.zeros(len(x), dtype=bool)
        conv_all = np.zeros(len(x), dtype=bool)
        for _ in range(_NEWTON_MAXITER):
            act = np.where(~done)[0]
            if not len(act):
                break
            t = self._snp_terms_at(x[act], cols[act], xp, yp, nrm2, reml, True)
            g, g2 = t["d1"], t["d2"]
            xa = x[act]
            la = np.where(g > 0, xa, lo[act])
            ha = np.where(g < 0, xa, hi[act])
            with np.errstate(divide="ignore", invalid="ignore"):
                step = g / g2
            newton = xa - step
            ok = (
                np.isfinite(newton)
                & (g2 < 0)
                & (newton > la)
                & (newton < ha)
                & (np.abs(2.0 * step) <= np.abs(dx_old[act]))
            )
            xn = np.where(ok, newton, 0.5 * (la + ha))
            xn = np.where(g == 0, xa, xn)
            dx_old[act] = dx[act]
            dx[act] = np.abs(xn - xa)
            conv = (dx[act] <= _NEWTON_XTOL) | (ha - la <= _NEWTON_XTOL) | (g == 0)
            bad = ~np.isfinite(g)
            x[act], lo[act], hi[act] = xn, la, ha
            done[act] = conv | bad
            conv_all[act] = conv & ~bad
        return x, conv_all

    def fit_snps(self, X: ArrayLike, reml: bool = True, ngrids: int = 100) -> SnpFitResult:
        """Fit ``h`` with each SNP in the model, for a block of SNPs: the exact per-SNP fit.

        For every column ``x`` of ``X`` the model ``y = X0 b + x a + u + e``
        is fitted by REML or ML, as ``fit(X=x)`` does, and ``a`` is tested
        with a Wald test at that SNP's own ``h`` (as `association` after
        ``fit(X=x)``).  The search reaches the same maximum as `fit`'s (every
        local maximum of the grid ``0, 1/ngrids, ..., 1``, the ends
        included, refined, the best kept, ties to the smallest ``h``), but
        for the whole block at once:

        1. the log-likelihood of every SNP on the grid, as a few matrix
           products (`_snp_grid_ll`);
        2. the bracket of each grid peak is narrowed to an interval where the
           derivative ``l'(h)`` changes sign from + to -, and the root is
           found by a safeguarded Newton-Raphson on the analytic first and
           second derivatives of the rotated likelihood (Zhou & Stephens
           2012, GEMMA; written in ``h`` rather than ``lambda = h / (1 - h)``,
           so that ``h = 1`` is inside the domain), all brackets together;
        3. a bracket without such a sign change, or a Newton run that does
           not converge, falls back to `fit`'s bounded scalar search on that
           SNP alone.

        Parameters
        ----------
        X : (N, m) array, or (N,) for one SNP
            SNPs (raw, original individual order, over the individuals with a
            phenotype), one per column, with NO missing values.  A column in
            span(X0) up to rounding (e.g. a constant next to the intercept)
            gives ``nan``; a nearly collinear one a meaningless value (select
            the columns with `snp_estimable_columns` first, as `gwas` does).
        reml : bool, default True
            REML (the default) or ML, for the fit and the Wald test (``sigma2
            = Q / (N - p)`` or ``Q / N``, ``p = q + 1``).
        ngrids : int, default 100
            Grid intervals on [0, 1], as for `fit`.

        Returns
        -------
        SnpFitResult
            Named tuple ``(h, ll, t, p, beta, beta_var)`` of ``(m,)`` arrays:
            each SNP's fitted ``h`` and log-likelihood (the value `fit` with
            the SNP returns), and the Wald test at that ``h`` (``t(N - p)``).

        Raises
        ------
        ValueError
            If ``X`` has the wrong shape, ``ngrids`` is not a positive
            integer, or with ``legacy_fit=True`` (pylmm 1.0.0's per-SNP refit
            is ``fit(X=x)`` for each SNP).

        Warns
        -----
        UserWarning
            Once, if some SNP's fit is at the upper boundary with a
            rank-deficient ``K`` (see `fit`).

        Notes
        -----
        The cost is about that of 110 likelihood evaluations for the whole
        block (most of them matrix products), instead of about 130 scalar
        evaluations per SNP; ``sims/bench/refit.py`` measures the speedup
        and ``sims/studies/s9_tests.py`` the agreement with ``fit(X=x)`` and
        with the dense oracle.

        Examples
        --------
        >>> import numpy as np
        >>> from pylmm.lmm import LMM, calculate_kinship
        >>> rng = np.random.default_rng(5)
        >>> W = rng.binomial(2, 0.3, size=(80, 300)).astype(float)
        >>> K = calculate_kinship(W)
        >>> y = 1.0 * W[:, 0] + rng.normal(size=80)
        >>> model = LMM(y, K)
        >>> res = model.fit_snps(W[:, :20])
        >>> h0, _, _, ll0 = model.fit(X=W[:, 0])            # the same, one SNP at a time
        >>> bool(abs(res.h[0] - h0) < 1e-6 and abs(res.ll[0] - ll0) < 1e-9)
        True
        >>> t, p = model.association(W[:, 0])               # the Wald test at that h
        >>> bool(np.isclose(res.t[0], t, rtol=1e-6))
        True
        """
        if self.legacy_fit:
            raise ValueError(
                "fit_snps is not available with legacy_fit=True; pylmm 1.0.0's per-SNP refit "
                "is fit(X=x) for each SNP"
            )
        if isinstance(ngrids, bool) or not isinstance(ngrids, (int, np.integer)) or ngrids < 1:
            raise ValueError("ngrids must be a positive integer, got %r" % (ngrids,))
        xp, yp, nrm2 = self._snp_rotate(X)
        m = xp.shape[1]
        n = self.N
        p = self.q + 1
        self._snp_fallbacks = 0
        nan = np.full(m, np.nan)
        good = np.isfinite(xp).all(axis=0) & (nrm2 > 0)
        if not good.any():
            return SnpFitResult(nan, nan.copy(), nan.copy(), nan.copy(), nan.copy(), nan.copy())
        gj = np.where(good)[0]
        H = np.arange(ngrids + 1) / float(ngrids)
        ll = self._snp_grid_ll(H, xp[:, gj], yp, nrm2[gj], reml)  # (ngrids + 1, m_good)

        # candidates (column, h): the grid ends, the grid peaks, the refined points
        cand_j: list[NDArray[np.intp]] = [gj]
        cand_h: list[FloatArray] = [np.zeros(len(gj))]
        with np.errstate(invalid="ignore"):
            span = np.nanmax(ll, axis=0) - np.nanmin(ll, axis=0)
            scale = np.maximum(1.0, np.nanmax(np.abs(ll), axis=0))
        # a profile flat to rounding: h is not identifiable, h = 0 (as fit)
        live = span > 1e-12 * scale
        L = ll[:, live]
        fj = gj[live]
        cand_j.append(fj)
        cand_h.append(np.ones(len(fj)))
        # fit's brackets (_get_best): an end higher than its neighbour, every interior peak
        k0 = np.where(L[0] > L[1])[0]
        mid = L[1:-1]
        with np.errstate(invalid="ignore"):
            peak = (mid >= L[:-2]) & (mid >= L[2:]) & (mid > np.minimum(L[:-2], L[2:]))
        pi, pk = np.where(peak)
        pi = pi + 1
        k1 = np.where(L[-1] > L[-2])[0]
        cand_j.append(fj[pk])
        cand_h.append(H[pi])
        bj = np.concatenate([fj[k0], fj[pk], fj[k1]])
        blo = np.concatenate([np.zeros(len(k0)), H[pi - 1], np.full(len(k1), H[-2])])
        bhi = np.concatenate([np.full(len(k0), H[1]), H[pi + 1], np.ones(len(k1))])
        bmid = np.concatenate([np.full(len(k0), np.nan), H[pi], np.full(len(k1), np.nan)])

        if len(bj):
            # l' at the bracket ends and at the peak: a (+, -) sub-interval holds the maximum
            def d1(hh: FloatArray, cols: NDArray[np.intp]) -> FloatArray:
                return self._snp_terms_at(hh, cols, xp, yp, nrm2, reml, True)["d1"]

            g_lo, g_hi = d1(blo, bj), d1(bhi, bj)
            has_mid = np.isfinite(bmid)
            g_mid = np.full(len(bj), np.nan)
            if has_mid.any():
                g_mid[has_mid] = d1(bmid[has_mid], bj[has_mid])
            left = has_mid & (g_lo > 0) & (g_mid < 0)
            right = has_mid & ~left & (g_mid > 0) & (g_hi < 0)
            whole = ~has_mid & (g_lo > 0) & (g_hi < 0)
            lo = np.where(right, bmid, blo)
            hi = np.where(left, bmid, bhi)
            # start at the secant root of l' between the sub-interval's ends
            g_a = np.where(right, g_mid, g_lo)
            g_b = np.where(left, g_mid, g_hi)
            with np.errstate(divide="ignore", invalid="ignore"):
                x0 = lo + (hi - lo) * g_a / (g_a - g_b)
            inside = np.isfinite(x0) & (x0 > lo) & (x0 < hi)
            x0 = np.where(inside, x0, 0.5 * (lo + hi))
            # Done already: l' = 0 at the peak, or an end bracket without a (+, -)
            # change whose slope at the end of the domain points out of it, so that
            # its maximum is that end (a candidate already).  E.g. the ML profile of
            # a centred K with an intercept, which dips below 1 and rises into its
            # unbounded spike at h = 1 (l'(0.99) < 0 < l'(1)).
            known = (has_mid & (g_mid == 0)) | (
                ~has_mid & ~whole & (((blo == 0) & (g_lo <= 0)) | ((bhi == 1) & (g_hi >= 0)))
            )
            use = left | right | whole
            sel = np.where(use)[0]
            fallback = np.where(~use & ~known)[0]
            if len(sel):
                root, conv = self._newton(lo[sel], hi[sel], bj[sel], xp, yp, nrm2, reml, x0[sel])
                cand_j.append(bj[sel[conv]])
                cand_h.append(root[conv])
                fallback = np.concatenate([fallback, sel[~conv]])
            self._snp_fallbacks = len(fallback)
            for b in fallback:  # fit's bounded scalar search, on this SNP alone
                j = int(bj[b])
                design = _orthonormal_columns(np.column_stack([self._x0t_ortho, xp[:, j]]))
                res = optimize.minimize_scalar(
                    self._neg_log_likelihood,
                    args=(design, reml),
                    bounds=(blo[b], bhi[b]),
                    method="bounded",
                    options={"xatol": 1e-10},
                )
                cand_j.append(np.array([j]))
                cand_h.append(np.array([float(res.x)]))

        cj = np.concatenate(cand_j)
        ch = np.concatenate(cand_h)
        r = self._snp_terms_at(ch, cj, xp, yp, nrm2, reml)
        cl = r["ll"]
        # the best candidate per column; ties go to the smallest h
        order = np.lexsort((ch, -np.where(np.isfinite(cl), cl, -np.inf), cj))
        best = order[np.r_[True, cj[order][1:] != cj[order][:-1]]]
        h_out, ll_out, beta, xx0, Q1 = (nan.copy() for _ in range(5))
        bjj = cj[best]
        h_out[bjj], ll_out[bjj] = ch[best], cl[best]
        beta[bjj], xx0[bjj], Q1[bjj] = r["beta"][best], r["xx0"][best], r["Q1"][best]
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = Q1 / ((float(n) - float(p)) if reml else float(n))
            beta_var = sigma / xx0
            t = beta / np.sqrt(beta_var)
            bad = ~(xx0 > 0) | ~np.isfinite(t)
            t[bad] = np.nan
            beta[bad] = np.nan
            beta_var[bad] = np.nan
            pv = 2.0 * stats.t.sf(np.abs(t), n - p)
        self._warn_boundary(float(np.nanmax(h_out)), p)
        return SnpFitResult(h_out, ll_out, t, np.asarray(pv, dtype=np.float64), beta, beta_var)

    def _lrt_model(self) -> LMM:
        """The LRT's ML model: ``self`` without the null directions of K that the covariates span.

        In the rotated basis the floored (null) coordinates of ``K`` all have
        variance ``sigma2 * (1 - h + h * 1e-6)``: the covariance is a multiple
        of the identity on that block, so any orthonormal change of basis
        inside it keeps the covariance diagonal.  Let ``Z`` be the part of the
        covariates' span that lies in that block (``Z = U null(U_R)``, ``U`` the
        orthonormal rotated covariates, ``U_R`` its rows outside the block;
        ``r = dim Z``).  The ML likelihood is unbounded as ``h -> 1`` exactly
        when ``r > 0`` (bug 17): the covariates fit ``y`` along ``Z`` whatever
        ``h`` is, so the residual there is 0 while its variance goes to 0.  The
        reduced model drops ``Z``: it is the exact Gaussian likelihood of the
        ``N - r`` contrasts ``Q'y`` with ``Q`` an orthonormal basis of the
        complement of ``Z`` (a partial REML that removes only ``Z``), with the
        ``q - r`` covariate directions that remain.  Because ``Z`` lies in the
        span of the covariates, it is the same for the model with a SNP, so
        the null and alternative ML fits stay nested and their LRT valid.
        With no such direction (``r = 0``, e.g. a positive definite or
        uncentred ``K``) this is ``self``.  Computed once and cached.
        """
        if self._lrt_cache is not None:
            return self._lrt_cache
        reduced = self._reduced_ml_model()
        self._lrt_cache = self if reduced is None else reduced
        return self._lrt_cache

    def _reduced_ml_model(self) -> LMM | None:
        """`_lrt_model`'s reduced model, or None if no direction is dropped."""
        null = self.kva <= 1e-6
        d = int(null.sum())
        if not d or not self.q or self.legacy_fit:
            return None
        U = self._x0t_ortho
        u_rest = U[~null]
        if u_rest.shape[0]:
            _, s, vt = np.linalg.svd(u_rest, full_matrices=True)
            rank = int((s > _NULL_SPAN_TOL).sum())
        else:  # every coordinate is null
            vt, rank = np.eye(self.q), 0
        C = vt[rank:].T  # (q, r): coefficient directions with no weight outside the block
        r = C.shape[1]
        if not r:
            return None
        zn = U[null] @ C  # (d, r), orthonormal: Z restricted to the null block
        comp = np.linalg.qr(zn, mode="complete")[0][:, r:]  # (d, d - r)

        def reduce(A: FloatArray) -> FloatArray:
            """Rotated rows -> the N - r rows of the reduced model."""
            out: FloatArray = np.concatenate([A[~null], comp.T @ A[null]], axis=0)
            return out

        R: LMM = object.__new__(LMM)
        R.__dict__.update(self.__dict__)
        R.N = self.N - r
        R.kva = np.concatenate([self.kva[~null], np.full(d - r, 1e-6)])
        R.kve = np.concatenate([self.kve[:, ~null], self.kve[:, null] @ comp], axis=1)
        R.yt = reduce(self.yt)
        # the covariate span left after dropping Z (q - r directions)
        ub, sb, _ = np.linalg.svd(reduce(U), full_matrices=False)
        R.x0t = ub[:, : self.q - r]
        R._x0t_ortho = R.x0t
        R.q = R.x0t.shape[1]
        R.x0t_stack = np.hstack([R.x0t, np.ones((R.N, 1))])
        R._ml_ill_posed = None
        R._lrt_cache = R
        R._lrt_null = None
        R._dropped = r
        return R

    def lrt_null_fit(self, ngrids: int = 100) -> tuple[float, float]:
        """The null model's ML fit for the LRT: ``(h, log_likelihood)`` of `_lrt_model`.

        By maximum likelihood, on the model with ``K``'s covariate-spanned
        null directions dropped (see `lrt_batch`), so it has an interior
        maximum where the plain ML fit of `fit` sits at its ``h = 1`` spike.
        Computed once and cached; does not change this model's ``opt_*``.

        Examples
        --------
        >>> import numpy as np
        >>> from pylmm.lmm import LMM, calculate_kinship
        >>> rng = np.random.default_rng(1)
        >>> W = rng.binomial(2, 0.3, size=(100, 1000)).astype(float)
        >>> K = calculate_kinship(W)                  # centred: K 1 = 0
        >>> y = rng.normal(size=100)                  # h2 = 0
        >>> model = LMM(y, K)
        >>> h_lrt, ll_lrt = model.lrt_null_fit()
        >>> bool(h_lrt < 0.5)                         # the plain ML fit is at h = 1
        True
        """
        if self._lrt_null is None:
            M = self._lrt_model()
            if M._lrt_null is None:
                with warnings.catch_warnings():
                    # the reduced model is never ill-posed; self may be, and is not used
                    warnings.filterwarnings("ignore", message="the ML likelihood has no maximum")
                    h, _, _, ll = M._fit_copy(None, False, ngrids)
                M._lrt_null = (float(h), float(ll))
            self._lrt_null = M._lrt_null
        return self._lrt_null

    def _fit_copy(
        self, X: FloatArray | None, reml: bool, ngrids: int = 100
    ) -> tuple[np.float64, FloatArray, float, float]:
        """`fit` that leaves this model's ``opt_*``, ``ll_grid`` and ``h_grid`` as they were."""
        keys = ("opt_h", "opt_ll", "opt_beta", "opt_sigma", "ll_grid", "h_grid")
        saved = {k: self.__dict__[k] for k in keys if k in self.__dict__}
        stack = self.x0t_stack[:, self.q].copy()
        try:
            return self.fit(X=X, ngrids=ngrids, reml=reml)
        finally:
            for k in keys:
                self.__dict__.pop(k, None)
            self.__dict__.update(saved)
            self.x0t_stack[:, self.q] = stack

    def lrt_batch(
        self, X: ArrayLike, ll0: float | None = None, ngrids: int = 100
    ) -> AssociationResult:
        """Likelihood-ratio tests of many SNPs, by (reduced) maximum likelihood.

        ``stat = 2 (l1 - l0)``, with ``l1`` each SNP's ML fit (`fit_snps`,
        ``reml=False``) and ``l0`` the null model's (`lrt_null_fit`), is
        referred to ``chi2(1)``.  The likelihoods are ML, not REML: REML
        likelihoods of models with different fixed effects are not
        comparable.

        Both fits use the reduced model of `_lrt_model`: the null directions
        of ``K`` that the covariates span are dropped.  With pylmm's centred
        kinship (``K 1 = 0``) and an intercept that is one direction, along
        which the plain ML likelihood grows without bound as ``h -> 1``
        (bug 17): both plain fits would then sit at ``h = 1`` (in most
        replicates at n <= ~300, whatever the true h2) and the LRT would
        compare two models without residual variance, anti-conservatively.
        Dropping it leaves the exact likelihood of the remaining ``N - r``
        contrasts, the same for both models, so the comparison stays nested.
        Where nothing is dropped (``K`` positive definite, or its null space
        not in the covariates' span) this is the plain ML LRT, GEMMA's ``-lmm
        2``; GEMMA centres every kinship and keeps the unbounded likelihood,
        stopping at its bound ``lambda = 1e5``, so with a centred K the two
        differ.

        Parameters
        ----------
        X : (N, m) array, or (N,) for one SNP
            SNPs, as for `fit_snps`.
        ll0 : float, optional
            The null log-likelihood on the same (reduced) scale.  Default:
            `lrt_null_fit`.
        ngrids : int, default 100
            As for `fit_snps`.

        Returns
        -------
        AssociationResult
            ``(t, p, beta, beta_var)`` of ``(m,)`` arrays: ``t =
            sign(beta) * sqrt(stat)``, the ``chi2(1)`` p-value of ``stat``,
            and the SNP effect and its variance from the ML fit with the SNP
            (the Wald quantities at that SNP's ML ``h``, ``sigma2 = Q / N``,
            ``N`` the reduced sample size).

        Notes
        -----
        ``l1 >= l0`` in exact arithmetic (the null model is nested); a
        negative ``stat`` from rounding is set to 0.  ``sims/studies/s9_tests.md``
        measures the calibration: about that of the Wald test from n = 300,
        mildly anti-conservative at n = 100 (chi2(1) has no small-sample
        correction).

        Examples
        --------
        >>> import numpy as np
        >>> from pylmm.lmm import LMM, calculate_kinship
        >>> rng = np.random.default_rng(5)
        >>> W = rng.binomial(2, 0.3, size=(80, 300)).astype(float)
        >>> K = calculate_kinship(W)
        >>> y = 1.0 * W[:, 0] + rng.normal(size=80)
        >>> model = LMM(y, K)
        >>> res = model.lrt_batch(W[:, :20])
        >>> bool(res.p[0] < 1e-3), res.t.shape
        (True, (20,))
        """
        M = self._lrt_model()
        if ll0 is None:
            ll0 = self.lrt_null_fit(ngrids)[1]
        f = M.fit_snps(X, reml=False, ngrids=ngrids)
        return _lrt_result(f.ll, ll0, f.beta, f.beta_var)

    def association_tests(
        self,
        X: ArrayLike,
        test: str = "wald",
        reml: bool = True,
        refit: bool = False,
        batch: bool = True,
    ) -> AssociationResult:
        """Test every column of ``X`` with the Wald or the likelihood-ratio test.

        The dispatcher `gwas` and ``pylmm gwas`` use.  For the Wald test
        without ``refit`` the null model must be fitted first,
        ``fit(reml=reml)``; the LRT fits its own null model (`lrt_null_fit`).

        ===========  =============  ================================================
        test         refit          what
        ===========  =============  ================================================
        ``"wald"``   False          `association_batch` at the null ``h`` (EMMAX/P3D)
        ``"wald"``   True           `fit_snps`: the Wald test at each SNP's own ``h``
        ``"lrt"``    (always)       `lrt_batch`: reduced-ML fits with and without the SNP
        ===========  =============  ================================================

        With ``batch=False`` (or ``legacy_fit=True``) the SNPs are tested one
        at a time instead: the Wald test with `association`, the refit and
        the LRT's per-SNP fit with ``fit(X=x)`` (the grid search of `fit`,
        pylmm 1.0.0's with ``legacy_fit``).  The model's null fit
        (``opt_h``, ``opt_sigma``, ``opt_ll``, ...) is left as it was.  The
        score test is `score_batch` (library only).

        Parameters
        ----------
        X : (N, m) array, or (N,) for one SNP
            SNPs with no missing values, estimable (`snp_estimable_columns`).
        test : {"wald", "lrt"}, default "wald"
            The test, see the table.
        reml : bool, default True
            REML or ML for the Wald test; the LRT is always ML (``reml`` is
            ignored).
        refit : bool, default False
            Refit ``h`` with each SNP (the Wald test; the LRT always does).
        batch : bool, default True
            Test the block at once (see above).

        Returns
        -------
        AssociationResult
            ``(t, p, beta, beta_var)`` of ``(m,)`` arrays; for the LRT ``t`` is
            the signed root of the statistic (see `lrt_batch`).

        Raises
        ------
        ValueError
            For an unknown ``test``, or the LRT with ``legacy_fit`` (pylmm
            1.0.0 had only the Wald test).

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
        >>> wald = model.association_tests(W[:, :10])
        >>> refit = model.association_tests(W[:, :10], refit=True)
        >>> lrt = model.association_tests(W[:, :10], test="lrt")
        >>> [bool(r.p[0] < 1e-3) for r in (wald, refit, lrt)]
        [True, True, True]
        """
        check_test(test, refit, self.legacy_fit)
        G = self._snp_block(X)
        if self.legacy_fit or not batch:
            return self._association_tests_per_snp(G, test, reml, refit)
        if test == "lrt":
            return self.lrt_batch(G)
        if refit:
            f = self.fit_snps(G, reml=reml)
            return AssociationResult(f.t, f.p, f.beta, f.beta_var)
        return self.association_batch(G, reml=reml)

    def _association_tests_per_snp(
        self, G: FloatArray, test: str, reml: bool, refit: bool
    ) -> AssociationResult:
        """`association_tests` one SNP at a time (``batch=False``, or ``legacy_fit``).

        Each SNP's fit runs through `_fit_copy`, so the model's null fit is
        unchanged afterwards.
        """
        m = G.shape[1]
        out = [np.full(m, np.nan) for _ in range(4)]
        if test == "lrt":
            M = self._lrt_model()
            ll0 = self.lrt_null_fit()[1]
            for j in range(m):
                x = G[:, [j]]
                hj, _, _, ll1 = M._fit_copy(x, False)
                saved = M.x0t_stack[:, M.q].copy()
                _, _, beta, beta_var = M.association(x, float(hj), reml=False, return_beta=True)
                M.x0t_stack[:, M.q] = saved
                r = _lrt_result(np.array([ll1]), ll0, np.array([beta]), np.array([beta_var]))
                for k in range(4):
                    out[k][j] = r[k][0]
            return AssociationResult(*out)
        h0 = getattr(self, "opt_h", None)
        for j in range(m):
            x = G[:, [j]]
            if refit:
                hj = self._fit_copy(x, reml)[0]
                ts, ps, beta, beta_var = self.association(x, float(hj), reml=reml, return_beta=True)
            else:
                ts, ps, beta, beta_var = self.association(x, h0, reml=reml, return_beta=True)
            out[0][j], out[1][j], out[2][j], out[3][j] = ts, ps, beta, beta_var
        return AssociationResult(*out)

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
