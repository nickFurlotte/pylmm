# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

import sys
import time
import warnings

import numpy as np
from scipy import linalg, optimize, stats

# np.seterr('raise')


def _matrix_mult(A, B):
    # The original code tried scipy.linalg.fblas.dgemm first.  scipy has had
    # no linalg.fblas since 0.12, so that path was dead and every call fell
    # back to np.dot -- which is what the Phase 1 goldens were computed with.
    return np.dot(A, B)


def calculate_kinship(W, center=False):
    """
    W is an n x m matrix encoding SNP minor alleles.

    This function takes a matrix oF SNPs, imputes missing values with the maf,
    normalizes the resulting vectors and returns the RRM matrix.
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


def _residual_on_span(x, X0):
    """x minus its least-squares projection on the columns of X0 (QR)."""
    if X0.shape[1] == 0:
        return x
    Q, _ = linalg.qr(X0, mode="economic")
    return x - _matrix_mult(Q, _matrix_mult(Q.T, x))


def snp_estimable(x, X0, rtol=ESTIMABLE_RTOL, Q=None):
    """True if the effect of the SNP column x can be estimated next to the
    covariates X0 (n x q, full rank): x is finite, the design [X0, x] leaves
    at least one residual degree of freedom, and x is not (up to rounding)
    in the span of X0.  Q, if given, is an orthonormal basis of X0's columns
    (linalg.qr(X0, mode='economic')[0]), to save the QR in a scan.  That covers a constant SNP (with an intercept), a SNP
    equal to a covariate, and SNPs with too few non-missing individuals.
    If X0 spans the intercept, the tolerance is relative to the centred x.

    GWAS and both pylmmGWAS branches use this one check, after dropping the
    individuals with a missing genotype.  It replaces the old exact-zero
    variance filters, which let columns that are constant up to rounding
    through with t ~ 1e-8 (issue 19, bug 13).
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


def _column_list(cols):
    return ", ".join(str(c) for c in cols)


def check_covariates(X0, n=None, what="covariate matrix X0"):
    """Raise ValueError if X0 has the wrong number of rows, a nan/inf, or is
    not of full column rank (zero, duplicate or collinear columns).  The
    message names the offending columns (0-based)."""
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


def check_ids(ids, kinship_ids):
    """Raise ValueError unless the individuals of K (K_ids) are those of Y
    (ids), in the same order.  Either may be None, then nothing is checked."""
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


def _check_phenotype(Y):
    """Y as a float vector; raise ValueError on inf, on no observed values or
    on a constant phenotype (issues 22 and 24)."""
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    obs = Y[~np.isnan(Y)]
    if np.isinf(obs).any():
        raise ValueError("phenotype has %d infinite values" % np.isinf(obs).sum())
    if len(obs) and np.ptp(obs) == 0:
        raise ValueError(
            "phenotype has zero variance: all %d non-missing values equal %r" % (len(obs), obs[0])
        )
    return Y


def _check_design_size(n, q, what="model"):
    """n observations vs q covariates: a Wald test of one SNP needs n > q+1
    (issue 20)."""
    if n <= q + 1:
        raise ValueError(
            "n=%d observations, but the %s has %d covariate column(s) plus the SNP, "
            "so no residual degree of freedom is left (need n > q + 1)" % (n, what, q)
        )


def gwas(Y, X, K, kva=[], kve=[], X0=None, reml=True, refit=False, ids=None, kinship_ids=None):
    """

    Performs a basic GWAS scan using the LMM.  This function
    uses the LMM module to assess association at each SNP and
    does some simple cleanup, such as removing missing individuals
    per SNP and re-computing the eigen-decomp

    Y - n x 1 phenotype vector
    X - n x m SNP matrix
    K - n x n kinship matrix
    Kva,Kve = linalg.eigh(K) - or the eigen vectors and values for K
    X0 - n x q covariate matrix
    REML - use restricted maximum likelihood (the default) for the null
           fit, every refit and the test; REML=False uses ML throughout
    refit - refit the variance component for each SNP
    ids, K_ids - optional individual IDs of Y and of K; if both are given
           they must match, in order

    Returns two lists (t-statistics, p-values), with nan for SNPs that
    cannot be tested (all missing, constant, or collinear with X0).

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
        kva = []
        kve = []

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
    counts = {}

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


def subset_testable(xs, ys, x0s, counts=None):
    """For a SNP with missing calls, on the individuals that have one: the
    SNP is estimable, X0 is still of full rank and Y is not constant.  If
    counts (a dict) is given, the SNPs rejected because the covariates lost
    rank on the subset are counted in counts['collinear'], so that the caller
    can report them once."""
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
    """
    This is a simple version of EMMA/fastLMM.
    The main purpose of this module is to take a phenotype vector (Y), a set of covariates (X) and a kinship matrix (K)
    and to optimize this model by finding the maximum-likelihood estimates for the model parameters.
    There are three model parameters: heritability (h), covariate coefficients (beta) and the total
    phenotypic variance (sigma).
    Heritability as defined here is the proportion of the total variance (sigma) that is attributed to
    the kinship matrix.

    For simplicity, we assume that everything being input is a numpy array.
    If this is not the case, the module may throw an error as conversion from list to numpy array
    is not done consistently.

    """

    def __init__(
        self,
        Y,
        K,
        kva=[],
        kve=[],
        X0=None,
        verbose=False,
        ids=None,
        kinship_ids=None,
        validate=True,
    ):
        """
        The constructor takes a phenotype vector or array of size n.
        It takes a kinship matrix of size n x n.  Kva and Kve can be computed as Kva,Kve = linalg.eigh(K) and cached.
        If they are not provided, the constructor will calculate them.
        X0 is an optional covariate matrix of size n x q, where there are q covariates.
        When this parameter is not provided, the constructor will set X0 to an n x 1 matrix of all ones to represent a mean effect.
        ids and K_ids are optional individual IDs of Y and of K; if both are given they must match, in order.

        The inputs are checked (Phase 5.2): a ValueError for a nan/inf in K or X0, an infinite or constant
        phenotype, rank-deficient covariates, too few observations, or a K of the wrong shape; a warning for an
        asymmetric K or one with clearly negative eigenvalues.  validate=False skips these checks (GWAS uses it
        for the per-SNP subsets, which it checks itself).  The caller's arrays are not modified.
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
            kva = []
            kve = []
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

        if len(kva) == 0 or len(kve) == 0:
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
                raise ValueError("eigendecomposition (Kva, Kve) has nan/inf values")

        if validate:
            self._warn_negative_eigenvalues(kva)

        self.K = K
        self.kve = kve
        self.N = self.K.shape[0]
        self.Y = Y.reshape((self.N, 1))
        self.X0 = X0

        # A new array: the caller's Kva is not clamped in place (bug 12).
        self.kva = np.array(kva, dtype=np.float64)
        if sum(self.kva < 1e-6):
            if self.verbose:
                sys.stderr.write("Cleaning %d eigen values\n" % (sum(self.kva < 0)))
            self.kva[self.kva < 1e-6] = 1e-6

        self.transform()

    @staticmethod
    def _check_kinship(K, n):
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
    def _warn_negative_eigenvalues(kva):
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

    def transform(self):
        """
        Computes a transformation on the phenotype vector and the covariate matrix.
        The transformation is obtained by left multiplying each parameter by the transpose of the
        eigenvector matrix of K (the kinship).
        """

        self.yt = _matrix_mult(self.kve.T, self.Y)
        self.x0t = _matrix_mult(self.kve.T, self.X0)
        self.x0t_stack = np.hstack([self.x0t, np.ones((self.N, 1))])
        self.q = self.x0t.shape[1]

    def ml_solution(self, h, X):
        """
        Obtains the maximum-likelihood estimates for the covariate coefficients (beta),
        the total variance of the trait (sigma) and also passes intermediates that can
        be utilized in other functions. The input parameter h is a value between 0 and 1 and represents
        the heritability or the proportion of the total variance attributed to genetics.  The X is the
        covariate matrix.
        """

        S = 1.0 / (h * self.kva + (1.0 - h))
        xt = X.T * S
        xx = _matrix_mult(xt, X)
        xx_inv = linalg.inv(xx)
        beta = _matrix_mult(_matrix_mult(xx_inv, xt), self.yt)
        yt = self.yt - _matrix_mult(X, beta)
        Q = np.dot(yt.T * S, yt)
        sigma = Q * 1.0 / (float(self.N) - float(X.shape[1]))
        return beta, sigma, Q, xx_inv, xx

    def _neg_log_likelihood(self, h, X=None, reml=False):
        # Objective (-LL) for the bounded search in getMax.  That search never
        # leaves its grid interval inside [0, 0.99], so the h > 1 guard is purely
        # defensive; it is kept so LL is never evaluated where it is nan:
        # for h > 1, h*Kva + (1-h) can be negative and LL is nan (bug 4).
        if h < 0 or h > 1:
            return 1e6
        return -self.log_likelihood(h, X, stack=False, reml=reml)[0]

    def log_likelihood(self, h, X=None, stack=True, reml=False):
        """
        Computes the log-likelihood for a given heritability (h).  If X==None, then the
        default X0t will be used.  If X is set and stack=True, then X0t will be matrix concatenated with
        the input X.  If stack is false, then X is used in place of X0t in the LL calculation.
        REML is computed by adding additional terms to the standard LL and can be computed by setting REML=True.
        """

        if X is None:
            X = self.x0t
        elif stack:
            self.x0t_stack[:, (self.q)] = _matrix_mult(self.kve.T, X)[:, 0]
            X = self.x0t_stack

        n = float(self.N)
        q = float(X.shape[1])
        beta, sigma, Q, xx_inv, xx = self.ml_solution(h, X)
        log_likelihood = (
            n * np.log(2 * np.pi)
            + np.log(h * self.kva + (1.0 - h)).sum()
            + n
            + n * np.log(1.0 / n * Q)
        )
        log_likelihood = -0.5 * log_likelihood

        if reml:
            ll_reml_part = (
                q * np.log(2.0 * np.pi * sigma)
                + np.log(linalg.det(_matrix_mult(X.T, X)))
                - np.log(linalg.det(xx))
            )
            log_likelihood = log_likelihood + 0.5 * ll_reml_part

        log_likelihood = log_likelihood.sum()
        return log_likelihood, beta, sigma, xx_inv

    def _get_max(self, H, X=None, reml=False):
        """
        Helper functions for .fit(...).
        This function takes a set of LLs computed over a grid and finds possible regions
        containing a maximum.  Within these regions, a bounded Brent search is performed
        to find the optimum.

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
                # if np.isnan(HOpt[-1]): HOpt[-1] = self.LLs[i-1]
                # if np.isnan(HOpt[-1][0]): HOpt[-1][0] = [self.LLs[i-1]]

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

    def fit(self, X=None, ngrids=100, reml=True):
        """
        Finds the maximum-likelihood solution for the heritability (h) given the current parameters.
        X can be passed and will transformed and concatenated to X0t.  Otherwise, X0t is used as
        the covariate matrix.

        This function calculates the LLs over a grid and then uses .getMax(...) to find the optimum.
        Given this optimum, the function computes the LL and associated ML solutions.
        """

        if X is None:
            X = self.x0t
        else:
            # X = np.hstack([self.X0t,matrixMult(self.Kve.T, X)])
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

    def association(self, X, h=None, stack=True, reml=True, return_beta=False):
        """
        Calculates association statitics for the SNPs encoded in the vector X of size n.
        If h == None, the optimal h stored in optH is used.
        REML=True (the default) is the REML Wald test, sigma^2 = Q/(n-q); REML=False the ML one,
        sigma^2 = Q/n.  Fit the null model with the same choice (fit(REML=...)).

        """
        if stack:
            # X = np.hstack([self.X0t,matrixMult(self.Kve.T, X)])
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

    def tstat(self, beta, var, sigma, q, log=False):
        """
        Calculates a t-statistic and associated p-value given the estimate of beta and its standard error.
        This is actually an F-test, but when only one hypothesis is being performed, it reduces to a t-test.
        """

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

    def plot_fit(self, color="b-", title=""):
        """
        Simple function to visualize the likelihood space.  It takes the LLs
        calcualted over a grid and normalizes them by subtracting off the mean and exponentiating.
        The resulting "probabilities" are normalized to one and plotted against heritability.
        This can be seen as an approximation to the posterior distribuiton of heritability.

        For diagnostic purposes this lets you see if there is one distinct maximum or multiple
        and what the variance of the parameter looks like.
        """
        import matplotlib.pyplot as pl

        mx = self.ll_grid.max()
        p = np.exp(self.ll_grid - mx)
        p = p / p.sum()

        pl.plot(self.h_grid, p, color)
        pl.xlabel("Heritability")
        pl.ylabel("Probability of data")
        pl.title(title)

    def mean_and_var(self):

        mx = self.ll_grid.max()
        p = np.exp(self.ll_grid - mx)
        p = p / p.sum()

        mn = (self.h_grid * p).sum()
        vx = ((self.h_grid - mn) ** 2 * p).sum()

        return mn, vx
