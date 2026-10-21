# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.


"""Two-variance-component fit: the heritability split between two kinships.

:func:`fit_two` extends the single-kinship model of :class:`pylmm.lmm.LMM` to
two kinship matrices (for example, two sets of markers) by a grid search over
their relative weight.
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pylmm.lmm import LMM


def fit_two(
    Y: ArrayLike,
    K1: NDArray[np.float64],
    K2: NDArray[np.float64],
    X0: ArrayLike | None = None,
    ngrids: int = 100,
    legacy_fit: bool = False,
) -> tuple[float, float, float, NDArray[np.float64], float, float]:
    """Fit a mixed model with two variance components over a grid of weights.

    The model is the usual LMM, ``Y = X0 b + g + e``, with ``g ~ N(0, h2_total
    * s2 * K)`` and ``e ~ N(0, (1 - h2_total) * s2 * I)``, where the kinship is
    a convex combination of two::

        K = w * K1 + (1 - w) * K2

    ``w`` runs over the grid ``0, 1/ngrids, ..., 1`` (both ends included, so
    ``K1`` alone and ``K2`` alone are candidates). For each ``w`` the
    single-kinship model is fit with :meth:`pylmm.lmm.LMM.fit` (REML; ``h``
    searched on [0, 1] with its own grid of step 0.01, whatever ``ngrids``
    is), and the ``w`` with the highest log-likelihood is kept. Ties go to
    the first (smallest) ``w``, with a warning on stderr; progress is also
    written to stderr.

    Parameters
    ----------
    Y : array_like, shape (n,) or (n, 1)
        Phenotype.
    K1, K2 : ndarray, shape (n, n)
        The two kinship matrices.
    X0 : array_like, shape (n, q), optional
        Covariates. Default: an intercept only.
    ngrids : int, default 100
        Number of grid intervals for ``w`` only (``ngrids + 1`` points).  The
        heritability grid inside each fit is NOT affected.
    legacy_fit : bool, default False
        As pylmm 1.0.0: ``w`` on ``0, 1/ngrids, ..., (ngrids - 1)/ngrids``
        (``ngrids`` points, never 1) and every fit with
        ``LMM(..., legacy_fit=True)`` (``h`` on 0 to 0.99).

    Returns
    -------
    h1, h2 : float
        Heritability attributed to ``K1`` and to ``K2``: ``w * h`` and
        ``(1 - w) * h``, where ``h`` is the total heritability at the best ``w``.
    e : float
        The residual fraction of the variance, ``1 - h``.
    beta : ndarray, shape (q, 1)
        Covariate coefficients at the optimum.
    sigma : float
        The total variance estimate at the optimum.
    log_likelihood : float
        The (REML) log-likelihood at the optimum.

    Notes
    -----
    Cost: one eigendecomposition of an ``n x n`` matrix per ``w`` grid point,
    so ``ngrids + 1`` of them (O(``ngrids`` * n**3)), plus about 100 likelihood
    evaluations (the fixed inner heritability grid) and a bounded refinement
    for each. This is slow for large ``n``; lower ``ngrids`` for a coarse
    answer (it cuts the number of eigendecompositions, not the inner 100).

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.misc import fit_two
    >>> rng = np.random.default_rng(0)
    >>> n = 40
    >>> g1, g2 = rng.standard_normal((n, 100)), rng.standard_normal((n, 100))
    >>> K1, K2 = g1 @ g1.T / 100, g2 @ g2.T / 100
    >>> Y = rng.standard_normal(n)
    >>> h1, h2, e, beta, sigma, ll = fit_two(Y, K1, K2, ngrids=5)
    >>> bool(0 <= h1 <= 1 and 0 <= h2 <= 1)
    True
    >>> bool(np.isclose(h1 + h2 + e, 1.0))
    True
    >>> beta.shape
    (1, 1)
    """

    # A uniform grid of w; it stopped below 1 before (bug 3)
    if legacy_fit:
        W = np.array(range(ngrids)) / float(ngrids)
    else:
        W = np.arange(ngrids + 1) / float(ngrids)
    results = []
    ll_grid = []

    for w in W:
        # heritability will be estimated for linear combo of kinships
        K = w * K1 + (1.0 - w) * K2
        sys.stderr.write("Fitting weight %0.2f\n" % (w))
        L = LMM(Y, K, X0=X0, legacy_fit=legacy_fit)
        R = L.fit()
        results.append(R)
        ll_grid.append(R[-1])

        del K

    ll = np.array(ll_grid)
    i = np.where(ll.max() == ll)[0]
    if len(i) > 1:
        sys.stderr.write("WARNING: Found multiple maxes using first one\n")

    i = i[0]
    hmax, beta, sigma, log_likelihood = results[i]
    w = float(W[i])

    h1 = w * hmax
    h2 = (1.0 - w) * hmax
    e = 1.0 - hmax

    return h1, h2, e, beta, sigma, log_likelihood
