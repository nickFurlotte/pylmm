# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""Linear mixed models (LMM) for genome-wide association and heritability.

This module provides the numerical core of pylmm: a kinship matrix estimator
(`calculate_kinship`), the model class `LMM` (variance-component fit, Wald
test of one SNP), the scan `gwas` that applies it to many SNPs, and the input
checks they share (`check_covariates`, `check_ids`, `snp_estimable`).

The model, for ``n`` individuals, is::

    y = X0 b + x a + u + e
    u ~ N(0, sigma2 * h * K)
    e ~ N(0, sigma2 * (1 - h) * I)

where ``y`` is the phenotype (n,), ``X0`` the covariates (n, q; the default is
an intercept), ``x`` one SNP (n,), ``K`` the kinship matrix (n, n), ``sigma2``
the total phenotypic variance and ``h`` in [0, 1) the heritability (the share
of ``sigma2`` due to ``K``).  ``K`` is eigendecomposed once, ``K = V diag(kva)
V'``; every likelihood is then evaluated on ``V' y`` and ``V' X`` with a
diagonal covariance, which makes a likelihood evaluation O(n) after the
rotation.

Conventions (see `LMM` and ``docs/conventions.md`` for details):

* REML is the default everywhere (``reml=True``); ``reml=False`` uses ML.
* ``sigma2`` is ``Q / (n - q)`` under REML and in `LMM.ml_solution`; the ML
  Wald test (``association(reml=False)``) rescales it to ``Q / n``.  ``Q`` is
  the weighted residual sum of squares.
* The h search runs on a grid on [0, 0.99] followed by a bounded refinement,
  so ``h = 1`` is never returned (known limitation, deferred to Phase 6.3).
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
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, optimize, stats

FloatArray: TypeAlias = NDArray[np.float64]
# A float64 numpy array of any shape (the shape is given in each docstring).

# np.seterr('raise')


def _matrix_mult(A: ArrayLike, B: ArrayLike) -> Any:
    """Matrix product of ``A`` and ``B`` (a thin wrapper over `numpy.dot`)."""
    # The original code tried scipy.linalg.fblas.dgemm first.  scipy has had
    # no linalg.fblas since 0.12, so that path was dead and every call fell
    # back to np.dot -- which is what the Phase 1 goldens were computed with.
    return np.dot(A, B)


def calculate_kinship(W: ArrayLike, center: bool = False) -> FloatArray:
    """Realized relationship matrix (genomic kinship) from a SNP matrix.

    Each SNP column is mean-imputed where missing (that is, with its minor
    allele frequency times two, for 0/1/2 coding), standardised to mean 0 and
    variance 1, and ``K = Z Z' / m`` is returned, where ``Z`` is the matrix of
    the kept standardised columns.

    Parameters
    ----------
    W : (n, m) array
        Genotypes of ``n`` individuals at ``m`` SNPs, coded as the number of
        minor alleles (0, 1, 2) or any numeric dosage; ``nan`` marks a missing
        call.  Not modified.
    center : bool, default False
        If True, return ``(n - 1) * K / trace(P K P)`` with
        ``P = I - 11'/n``, i.e. a K centred on the individuals and scaled so
        that its centred trace is ``n - 1``.

    Returns
    -------
    K : (n, n) float64 array
        Symmetric, positive semi-definite kinship matrix.

    Notes
    -----
    SNPs that are all missing or have zero variance are left out of ``Z`` but
    still count in the denominator ``m`` (the number of columns of ``W``), so
    K shrinks when many SNPs are monomorphic (bug 8).  Rescaling K changes the
    estimate of ``h`` but not the test statistics, up to the h grid.

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.lmm import calculate_kinship
    >>> rng = np.random.default_rng(1)
    >>> W = rng.binomial(2, 0.4, size=(20, 100)).astype(float)
    >>> W[0, 3] = np.nan                      # missing calls are imputed
    >>> K = calculate_kinship(W)
    >>> K.shape
    (20, 20)
    >>> bool(np.allclose(K, K.T))
    True
    >>> round(float(np.trace(K) / 20), 1)     # about 1 when no SNP is dropped
    1.0
    """
    # Work on a copy: the caller's W is not imputed or standardised in place
    # (bug 12, fixed in Phase 5.2).
    W = np.array(W, dtype=np.float64)
    n = W.shape[0]
    m = W.shape[1]
    keep = []
    for i in range(m):
        obs = ~np.isnan(W[:, i])
        if not obs.any():
            continue  # all-missing SNP: skip it (bug 11)
        mn = W[obs, i].mean()
        W[np.isnan(W[:, i]), i] = mn
        vr = W[:, i].var()
        if vr == 0:
            continue

        keep.append(i)
        W[:, i] = (W[:, i] - mn) / np.sqrt(vr)

    W = W[:, keep]
    K = _matrix_mult(W, W.T) * 1.0 / float(m)
    if center:
        P = np.diag(np.repeat(1, n)) - 1 / float(n) * np.ones((n, n))
        S = np.trace(_matrix_mult(_matrix_mult(P, K), P))
        k_centered = (n - 1) * K / S
        return k_centered
    return K


# ---------------------------------------------------------------------------
# Input checks shared by GWAS, LMM and the CLI (Phase 5.2)
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
    return x - _matrix_mult(Q, _matrix_mult(Q.T, x))


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
    ``pylmmGWAS`` branches call it after dropping individuals with a missing
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
            "phenotype has zero variance: all %d non-missing values equal %r" % (len(obs), obs[0])
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
) -> tuple[list[Any], list[Any]]:
    """Scan SNPs one at a time with the mixed model (Wald test per SNP).

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
    K : (n, n) array
        Kinship matrix in the same individual order as ``Y``.
    kva : (n,) array, optional
        Eigenvalues of ``K``, as from ``kva, kve = scipy.linalg.eigh(K)``.
        None (or empty) computes them.  Ignored (recomputed) if ``Y`` has
        missing values.
    kve : (n, n) array, optional
        Matching eigenvectors (columns), see ``kva``.
    X0 : (n, q) array, optional
        Covariates, full column rank.  Default: one column of ones (an
        intercept).
    reml : bool, default True
        REML for the null fit, every refit and the Wald test; False uses ML
        throughout (``sigma2 = Q / n`` in the test).
    refit : bool, default False
        Refit ``h`` and ``sigma2`` with each SNP in the model.
    ids, kinship_ids : sequence, optional
        Individual IDs of ``Y`` and of the rows of ``K``; if both are given
        they must match, in order.

    Returns
    -------
    t_stats : list of length m
        Wald t statistic of each SNP (numpy float64; ``nan`` for a SNP that
        cannot be tested: all missing, constant, collinear with ``X0``, or
        with too few called individuals).
    p_values : list of length m
        Two-sided p-value of each SNP from ``t(n_s - q - 1)``, where ``n_s``
        is the number of individuals used for that SNP; ``nan`` where
        ``t_stats`` is.  If no individual has a phenotype, both lists are
        all ``nan``.

    Raises
    ------
    ValueError
        For an infinite or constant phenotype, ``X`` or ``X0`` with the wrong
        number of rows, nan/inf or rank-deficient ``X0``, ``n <= q + 1``
        observations, mismatching ``ids``/``kinship_ids``, and everything
        `LMM` raises for ``K`` (wrong shape, nan/inf, bad ``kva``/``kve``).

    Warns
    -----
    UserWarning
        If some SNPs are ``nan`` because, on the individuals with a called
        genotype, the covariates are no longer of full rank; and the warnings
        of `LMM` (asymmetric ``K``, clearly negative eigenvalues of ``K``).

    Notes
    -----
    The null ``h`` is searched on [0, 0.99] (see `LMM.fit`).  The p-value is
    exactly 0.0 when it underflows, for very large ``|t|``.

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
    """
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
        keep = ~v
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
    L = LMM(Y, K, kva, kve, X0)
    if not refit:
        L.fit(reml=reml)
    Q0 = linalg.qr(X0, mode="economic")[0] if X0.shape[1] else np.zeros((X0.shape[0], 0))

    p_values = []
    t_stats = []
    counts: dict[str, int] = {}

    n = X.shape[0]
    m = X.shape[1]

    for i in range(m):
        x = X[:, i].reshape((n, 1))
        v = np.isnan(x).reshape((-1,))
        if v.sum():
            keep = ~v
            xs = x[keep, :]
            ys = Y[keep]
            x0s = X0[keep, :]
            if not subset_testable(xs, ys, x0s, counts):
                p_values.append(np.nan)
                t_stats.append(np.nan)
                continue

            ks = K[keep, :][:, keep]
            ls = LMM(ys, ks, X0=x0s, validate=False)
            if refit:
                ls.fit(X=xs, reml=reml)
            else:
                ls.fit(reml=reml)
            ts, ps = ls.association(xs, reml=reml)
        else:
            if not snp_estimable(x, X0, Q=Q0):
                p_values.append(np.nan)
                t_stats.append(np.nan)
                continue

            if refit:
                L.fit(X=x, reml=reml)
            ts, ps = L.association(x, reml=reml)

        p_values.append(ps)
        t_stats.append(ts)

    if counts.get("collinear"):
        warnings.warn(
            "%d of %d SNPs are nan because, on the individuals with a called genotype, "
            "the covariates are no longer of full rank" % (counts["collinear"], m),
            stacklevel=2,
        )
    return t_stats, p_values


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
        None (the default) or empty: computed here.
    kve : (n, n) array, optional
        Eigenvectors of ``K`` (columns), matching ``kva``.
    X0 : (n, q) array, optional
        Covariates, full column rank.  Default: a single column of ones, the
        mean.  The SNP is *not* part of ``X0``.
    verbose : bool, default False
        Write progress (missing values removed, eigendecomposition, number of
        eigenvalues cleaned, multiple optima) to stderr.
    ids, kinship_ids : sequence, optional
        Individual IDs of ``Y`` and of the rows of ``K``; if both are given
        they must match, in order (`check_ids`).
    validate : bool, default True
        Run the input checks below.  `gwas` passes False for its per-SNP
        subsets, which it has already checked.

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
    ll_grid, h_grid : (ngrids,) arrays
        After `fit`: the log-likelihood on the grid and the grid of ``h``.
    opt_h : float
        After `fit`: the estimated heritability ``h``.
    opt_ll : float
        After `fit`: the (RE)ML log-likelihood at ``opt_h``.
    opt_beta : (p, 1) array
        After `fit`: the coefficients at ``opt_h`` (``p = q``, or ``q + 1``
        when `fit` was given a SNP).
    opt_sigma : float
        After `fit`: ``sigma2`` at ``opt_h``.

    Raises
    ------
    ValueError
        (with ``validate=True``) for a nan/inf in ``K`` or ``X0``; an infinite
        or constant phenotype; no non-missing phenotype; a ``K`` that is not
        ``(n, n)``; ``kva``/``kve`` of the wrong shape or with nan/inf;
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
    * `fit` and `association` default to REML.  `log_likelihood` and
      `ml_solution` are lower-level; `log_likelihood` defaults to ML
      (``reml=False``).
    * ``sigma2 = Q / (N - p)`` in `ml_solution`, `fit` and the REML test, and
      ``Q / N`` in the ML Wald test of `association`.
    * `fit` searches ``h`` on [0, 0.99] (bugs 2 and 3, deferred to Phase 6.3):
      a true ``h`` of 0.99 or more is reported as 0.99, and the grid point
      0.98 is not refined.

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
    >>> bool(0.0 <= h < 1.0)
    True
    >>> t, p = model.association(W[:, [0]])        # Wald test of one SNP (a column)
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
    ) -> None:
        """Build the model: validate inputs, drop missing phenotypes,
        eigendecompose ``K`` if needed, and rotate ``Y`` and ``X0``.

        See the class docstring for the parameters.
        """

        if X0 is None:
            X0 = np.ones(len(Y)).reshape(len(Y), 1)
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

        x = ~np.isnan(Y)
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

        if kva is None or kve is None or len(kva) == 0 or len(kve) == 0:
            if self.verbose:
                sys.stderr.write(
                    "Obtaining eigendecomposition for %dx%d matrix\n" % (K.shape[0], K.shape[1])
                )
            begin = time.time()
            kva, kve = linalg.eigh(K)
            end = time.time()
            if self.verbose:
                sys.stderr.write("Total time: %0.3f\n" % (end - begin))
        elif validate:
            kva, kve = np.asarray(kva, dtype=np.float64), np.asarray(kve, dtype=np.float64)
            if kva.shape != (K.shape[0],) or kve.shape != K.shape:
                raise ValueError(
                    "eigendecomposition has shapes %s and %s, expected (%d,) and %s"
                    % (kva.shape, kve.shape, K.shape[0], K.shape)
                )
            if not (np.isfinite(kva).all() and np.isfinite(kve).all()):
                raise ValueError("eigendecomposition (kva, kve) has nan/inf values")

        if validate:
            self._warn_negative_eigenvalues(kva)

        self.K = K
        self.kve = kve
        self.N = self.K.shape[0]
        self.Y = Y.reshape((self.N, 1))
        self.X0 = X0

        # A new array: the caller's kva is not clamped in place (bug 12).
        self.kva = np.array(kva, dtype=np.float64)
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
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Generalized least squares at a fixed heritability.

        With ``S = 1 / (h * kva + 1 - h)`` (the weights in the rotated basis)
        this solves ``beta = (X'SX)^-1 X'S yt`` and computes the weighted
        residual sum of squares ``Q = r'Sr`` with ``r = yt - X beta``.

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
        sigma : (1, 1) array
            ``Q / (N - p)``: the estimate of the total variance ``sigma2``
            (REML-style denominator).
        Q : (1, 1) array
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
        xt = X.T * S
        xx = _matrix_mult(xt, X)
        xx_inv = linalg.inv(xx)
        beta = _matrix_mult(_matrix_mult(xx_inv, xt), self.yt)
        yt = self.yt - _matrix_mult(X, beta)
        Q = np.dot(yt.T * S, yt)
        # TODO(5b): sigma and Q are 1x1 arrays, not scalars; callers call .sum() on them.
        sigma = Q * 1.0 / (float(self.N) - float(X.shape[1]))
        return beta, sigma, Q, xx_inv, xx

    def _neg_log_likelihood(
        self, h: float, X: FloatArray | None = None, reml: bool = False
    ) -> float:
        """Objective ``-log_likelihood(h)`` for the bounded search in `_get_max`;
        1e6 outside [0, 1].  ``X`` is already rotated and not stacked."""
        # Objective (-LL) for the bounded search in _get_max.  That search never
        # leaves its grid interval inside [0, 0.99], so the h > 1 guard is purely
        # defensive; it is kept so LL is never evaluated where it is nan:
        # for h > 1, h*kva + (1-h) can be negative and LL is nan (bug 4).
        if h < 0 or h > 1:
            return 1e6
        return -self.log_likelihood(h, X, stack=False, reml=reml)[0]

    def log_likelihood(
        self, h: float, X: FloatArray | None = None, stack: bool = True, reml: bool = False
    ) -> tuple[Any, FloatArray, FloatArray, FloatArray]:
        """Log-likelihood of the data at heritability ``h``, profiled over ``beta`` and ``sigma2``.

        Parameters
        ----------
        h : float
            Heritability in [0, 1].  Above 1, ``h * kva + 1 - h`` can be
            negative and the result is nan.
        X : array, optional
            If None, the covariates ``x0t`` are used.  If given and
            ``stack=True``, ``X`` is a **raw** SNP column (N, 1) (a 1-d
            array raises IndexError) in the original individual order; it is rotated and appended to
            ``x0t`` (this overwrites ``x0t_stack``).  If given and
            ``stack=False``, ``X`` is a full **already rotated** design
            matrix (N, p) used in place of ``x0t``.
        stack : bool, default True
            See ``X``.
        reml : bool, default False
            If True, add the REML terms (an adjustment for the estimated
            ``beta``); the default is the ML log-likelihood.  Note that this
            differs from `fit` and `association`, which default to REML.

        Returns
        -------
        log_likelihood : float
            numpy float64; the full Gaussian log-likelihood, including the
            ``n log(2 pi)`` term.
        beta : (p, 1) array
            Coefficient estimates.
        sigma : (1, 1) array
            ``Q / (N - p)``, see `ml_solution`.
        xx_inv : (p, p) array
            ``(X'SX)^-1``.

        Notes
        -----
        The likelihood uses ``Q / N`` for the variance, but the returned
        ``sigma`` is ``Q / (N - p)``.
        """

        if X is None:
            X = self.x0t
        elif stack:
            self.x0t_stack[:, (self.q)] = _matrix_mult(self.kve.T, X)[:, 0]
            X = self.x0t_stack

        n = float(self.N)
        q = float(X.shape[1])
        beta, sigma, Q, xx_inv, xx = self.ml_solution(h, X)
        ll = (
            n * np.log(2 * np.pi)
            + np.log(h * self.kva + (1.0 - h)).sum()
            + n
            + n * np.log(1.0 / n * Q)
        )
        ll = -0.5 * ll

        if reml:
            ll_reml_part = (
                q * np.log(2.0 * np.pi * sigma)
                + np.log(linalg.det(_matrix_mult(X.T, X)))
                - np.log(linalg.det(xx))
            )
            ll = ll + 0.5 * ll_reml_part

        ll = ll.sum()
        return ll, beta, sigma, xx_inv

    def _get_max(self, H: FloatArray, X: FloatArray | None = None, reml: bool = False) -> Any:
        """Locate the maximiser of the likelihood from the grid, helper of `fit`.

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
        reml : bool, default False
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
                # (bug 4, fixed in Phase 5.1).  With xatol this small the
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
    ) -> tuple[Any, FloatArray, FloatArray, Any]:
        """Estimate the heritability ``h`` (and ``beta``, ``sigma2``) of the model.

        The log-likelihood is evaluated on the grid ``h = 0, 1/ngrids, ...,
        (ngrids - 1)/ngrids`` and the first interior peak is refined with a
        bounded search (`_get_max`).  The result is stored on the instance
        (``opt_h``, ``opt_ll``, ``opt_beta``, ``opt_sigma``, ``ll_grid``,
        ``h_grid``) and returned.

        Parameters
        ----------
        X : (N, 1) array, optional
            A SNP as a column (a 1-d array raises IndexError), in the original individual order (of the individuals with
            a phenotype).  If given it is added to the covariates for this
            fit (the per-SNP refit of `gwas`); otherwise the null model with
            ``X0`` only is fitted.  Overwrites ``x0t_stack``.
        ngrids : int, default 100
            Number of grid points on [0, (ngrids - 1)/ngrids]; with 100 the
            grid is 0, 0.01, ..., 0.99.
        reml : bool, default True
            REML (default) or ML.

        Returns
        -------
        h : float
            Estimated heritability in [0, 0.99] (numpy float64).
        beta : (p, 1) array
            Coefficients at ``h`` (``p = q``, or ``q + 1`` with ``X``).
        sigma : (1, 1) array
            ``sigma2 = Q / (N - p)``.
        log_likelihood : float
            The REML or ML log-likelihood at ``h``.

        Raises
        ------
        scipy.linalg.LinAlgError
            If the weighted normal equations are singular (not expected after
            the constructor's checks).

        Notes
        -----
        ``h`` never reaches 1: the grid ends at 0.99 (bug 3, deferred to
        Phase 6.3), so a very heritable trait is reported as 0.99.  A peak at
        grid index 98 (h = 0.98) is not detected as an interior maximum (bug
        2), so ``h`` snaps to 0.99 there.  ML is unbounded above for a
        centred GRM with more SNPs than individuals and can sit at 0.99; REML
        is preferred.  With several grid peaks the first is returned (a note
        goes to stderr if ``verbose``).

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
        >>> beta.shape, sigma.shape
        ((1, 1), (1, 1))
        >>> model.h_grid[[0, -1]]
        array([0.  , 0.99])
        >>> bool(h == model.opt_h)
        True
        """

        if X is None:
            X = self.x0t
        else:
            self.x0t_stack[:, (self.q)] = _matrix_mult(self.kve.T, X)[:, 0]
            X = self.x0t_stack

        H = np.array(range(ngrids)) / float(ngrids)
        L = np.array([self.log_likelihood(h, X, stack=False, reml=reml)[0] for h in H])
        self.ll_grid = L

        hmax = self._get_max(H, X, reml)
        L, beta, sigma, beta_stderr = self.log_likelihood(hmax, X, stack=False, reml=reml)

        self.h_grid = H
        self.opt_h = hmax.sum()
        self.opt_ll = L
        self.opt_beta = beta
        self.opt_sigma = sigma.sum()

        return hmax, beta, sigma, L

    def association(
        self,
        X: FloatArray,
        h: float | None = None,
        stack: bool = True,
        reml: bool = True,
        return_beta: bool = False,
    ) -> tuple[Any, ...]:
        """Wald test of one SNP, at a fixed heritability.

        The SNP is added to the covariates, ``beta`` is re-estimated by
        generalized least squares at ``h``, and ``t = beta_snp / se`` is
        compared with a ``t(N - p)`` distribution (``p = q + 1`` columns).
        Call `fit` first unless ``h`` is given.

        Parameters
        ----------
        X : (N, 1) array
            The SNP as a column (raw, original individual order, no missing
            values, over the individuals with a phenotype); a 1-d array raises
            IndexError.  With ``stack=False``, a full
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
            Only if ``return_beta``: SNP effect estimate.
        beta_var : float
            Only if ``return_beta``: its variance, ``xx_inv[-1, -1] * sigma``
            (with the sigma used for the test).

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
        >>> t, p, b, var = model.association(W[:, [0]], return_beta=True)
        >>> bool(p < 1e-3) and round(float(b)) == 1
        True
        >>> t2, p2 = model.association(W[:, [1]])
        >>> bool(0 <= p2 <= 1)
        True
        """
        if stack:
            # TODO(5b): accept 1-D x (X must be an (N, 1) column today).
            self.x0t_stack[:, (self.q)] = _matrix_mult(self.kve.T, X)[:, 0]
            X = self.x0t_stack

        if h is None:
            h = self.opt_h

        L, beta, sigma, beta_var = self.log_likelihood(h, X, stack=False, reml=reml)
        q = len(beta)
        # ML test: sigma^2 = Q/n, the ML estimate (bug 9, Phase 5.2); REML
        # keeps Q/(n-q).  The reference distribution is t(n-q) either way.
        if not reml:
            sigma = sigma * (float(self.N) - q) / float(self.N)
        ts, ps = self.tstat(beta[q - 1], beta_var[q - 1, q - 1], sigma, q)

        if return_beta:
            return ts, ps, beta[q - 1].sum(), beta_var[q - 1, q - 1].sum() * sigma
        return ts, ps

    def tstat(
        self, beta: ArrayLike, var: ArrayLike, sigma: ArrayLike, q: int, log: bool = False
    ) -> tuple[Any, Any]:
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
        sigma : (1, 1) array or float
            ``sigma2`` to use.
        q : int
            Number of columns of the design (the degrees of freedom are
            ``N - q``).
        log : bool, default False
            Return ``2 + logsf`` (the natural log of the one-sided tail, plus
            2) instead of the p-value.  Known bug 5 (it should add ``log 2``);
            unreachable, no caller passes it.

        Returns
        -------
        t_stat, p_value : float
            numpy float64 each (the p-value is two-sided).

        Raises
        ------
        Exception
            If the inputs are not scalar-like (more than one test).
        """

        # TODO(5b): the log option is wrong (adds 2, not log 2; bug 5) and unreachable.
        ts = beta / np.sqrt(var * sigma)
        # ps = 2.0*(1.0 - stats.t.cdf(np.abs(ts), self.N-q))
        # sf == survival function - this is more accurate -- could also use logsf if the precision is not good enough
        if log:
            ps = 2.0 + (stats.t.logsf(np.abs(ts), self.N - q))
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

    def mean_and_var(self) -> tuple[Any, Any]:
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
