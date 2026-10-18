"""Independent reference ("oracle") REML/ML fit of the pylmm model, for small n.

Model
-----
    y = X beta + u + e,   u ~ N(0, sigma^2 h K),   e ~ N(0, sigma^2 (1 - h) I)
    V(h, sigma^2) = sigma^2 H(h),   H(h) = h K + (1 - h) I,   h in [0, 1]

This is pylmm's parameterisation (sigma^2 is the *total* variance, h the
heritability).  The code here deliberately shares nothing with pylmm: it never
rotates by eigenvectors, it works on the dense n x n matrix H with a Cholesky
factorisation per h, and every formula is the textbook one.  It is O(n^3) per
likelihood evaluation, so keep n <= ~500 (n = 200: ~0.1 s per fit).

Profile likelihoods (sigma^2 and beta profiled out), with p = rank(X) = ncol(X),
r = y - X beta_hat(h), Q(h) = r' H^-1 r:

    ML:    beta_hat = (X' H^-1 X)^-1 X' H^-1 y,   sigma2_hat = Q / n
           l_ML(h) = -1/2 [ n log(2 pi sigma2_hat) + log|H| + n ]

    REML:  sigma2_hat = Q / (n - p)
           l_R(h) = -1/2 [ (n-p) log(2 pi sigma2_hat) + log|H| + log|X' H^-1 X|
                           - log|X' X| + (n - p) ]

l_R is exactly the Gaussian log-likelihood of the n - p error contrasts A'y
(A'A = I, A'X = 0); the ``- log|X'X|`` term makes it independent of the
choice of A (tests/sims/test_oracle.py checks this numerically).

Optimisation: l(h) is evaluated on a uniform grid over [0, 1] (both endpoints
included); the maximiser is refined with bounded Brent (scipy
``minimize_scalar(method='bounded')``) inside the bracket of every interior
grid local maximum, and the best of the refined points and the two endpoints
wins.  With a fine grid this finds the global maximum of a 1-D profile
reliably, including boundary maxima at h = 0 and h = 1.

Assumptions / limits (documented, by design)
--------------------------------------------
* y and X must be complete (no nan); X must have full column rank.
* K must be symmetric PSD.  At h = 1, H = K; if K is singular the likelihood
  is -inf there (reported as such), so h = 1 is only selected when K is
  positive definite.  Near-singular K is fine for h < 1 (H is PD).
* ML with a singular K whose null space lies inside col(X) has an UNBOUNDED
  profile.  The common case: pylmm's ``calculateKinship`` GRM is centred
  (K 1 = 0); built from more SNPs than individuals (outbred data) its null
  space is exactly span(1), which the intercept absorbs.  As h -> 1 the
  variance along 1 vanishes, the GLS fit takes up that direction, and
  l_ML(h) ~ -1/2 log(1 - h) -> +inf.  (With a larger null space, as for the
  rank ~110 BED kinship, y has residual mass there and l_ML -> -inf instead,
  so the problem does not arise.)  The MLE
  does not exist; ``fit`` then returns h just below 1 (or ``h_max``) and sets
  ``ml_unbounded=True``.  REML is unaffected (the log|X'H^-1X| term cancels
  the divergence).  pylmm never sees this because it clamps eigenvalues
  < 1e-6 to 1e-6 and its grid stops at h = 0.99.
* When K = I (or any K with H(h) constant in h) h is unidentifiable: the
  profile is flat, ``fit`` returns ``identifiable=False`` and h = 0 (the first
  maximiser on the grid).  beta, sigma2 and the Wald test then reduce to OLS.
* Wald test: t = beta_x / se(beta_x), se^2 = sigma2_wald [(X'H^-1X)^-1]_xx,
  with sigma2_wald = Q / (n - p) (the unbiased/REML-type scale) by default and
  df = n - p for the t reference distribution, which is the convention pylmm
  uses.  Pass ``sigma2='ml'`` for Q / n.
* Log-likelihood constants: oracle ML equals pylmm's ML LL exactly.  pylmm's
  REML LL differs from l_R by a constant that does not depend on h,
  ``pylmm_reml_offset(n, p)``; compare ``oracle.ll + offset`` with pylmm.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from scipy import linalg, optimize, stats


@dataclasses.dataclass
class OracleFit:
    h: float
    ll: float
    sigma2: float           # profiled total variance at h (Q/n for ML, Q/(n-p) for REML)
    beta: np.ndarray
    reml: bool
    identifiable: bool      # False if the profile is flat in h
    at_boundary: bool       # h within 1e-6 of 0 or 1
    h_grid: np.ndarray
    ll_grid: np.ndarray
    ml_unbounded: bool = False  # ML profile -> +inf as h -> 1 (see module docstring)


@dataclasses.dataclass
class OracleWald:
    t: float
    p: float
    beta: float
    se: float
    df: int
    h: float
    fit: OracleFit


def _as_X(X, n):
    if X is None:
        return np.ones((n, 1))
    X = np.asarray(X, dtype=np.float64)
    return X.reshape(n, -1)


def _check(y, K, X):
    if np.isnan(y).any() or np.isnan(X).any():
        raise ValueError("oracle needs complete y and X (drop missing rows first)")
    if K.shape != (len(y), len(y)):
        raise ValueError("K must be n x n")


def profile_loglik(h: float, y: np.ndarray, K: np.ndarray,
                   X: Optional[np.ndarray] = None, reml: bool = True):
    """(ll, beta_hat, sigma2_hat) at a fixed h; ll = -inf if H(h) is singular."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    X = _as_X(X, n)
    p = X.shape[1]
    H = h * np.asarray(K, dtype=np.float64) + (1.0 - h) * np.eye(n)
    singular = (-np.inf, np.full(p, np.nan), np.nan)
    try:
        C = linalg.cho_factor(H, lower=True)
    except linalg.LinAlgError:
        return singular
    d2 = np.diag(C[0]) ** 2
    # LAPACK may "succeed" on an exactly singular K with a round-off pivot
    if not np.all(np.isfinite(d2)) or d2.min() <= 1e-12 * d2.max():
        return singular
    logdetH = 2.0 * np.log(np.diag(C[0])).sum()
    HiX = linalg.cho_solve(C, X)
    Hiy = linalg.cho_solve(C, y)
    XtHiX = X.T @ HiX
    try:
        beta = linalg.solve(XtHiX, X.T @ Hiy, assume_a="pos")
    except linalg.LinAlgError:
        return singular
    r = y - X @ beta
    Q = float(r @ linalg.cho_solve(C, r))
    if reml:
        sigma2 = Q / (n - p)
        _, logdet_XtHiX = np.linalg.slogdet(XtHiX)
        _, logdet_XtX = np.linalg.slogdet(X.T @ X)
        ll = -0.5 * ((n - p) * np.log(2 * np.pi * sigma2) + logdetH
                     + logdet_XtHiX - logdet_XtX + (n - p))
    else:
        sigma2 = Q / n
        ll = -0.5 * (n * np.log(2 * np.pi * sigma2) + logdetH + n)
    return float(ll), beta, sigma2


def fit(y: np.ndarray, K: np.ndarray, X: Optional[np.ndarray] = None,
        reml: bool = True, n_grid: int = 201, flat_tol: float = 1e-9,
        h_max: float = 1.0) -> OracleFit:
    """Maximise the profile (RE)ML over h in [0, h_max]; see module docstring."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    X = _as_X(X, len(y))
    _check(y, K, X)

    def ll(h):
        return profile_loglik(h, y, K, X, reml)[0]

    H = np.linspace(0.0, h_max, n_grid)
    L = np.array([ll(h) for h in H])
    finite = np.isfinite(L)
    span = L[finite].max() - L[finite].min() if finite.any() else 0.0
    identifiable = bool(span > flat_tol * max(1.0, abs(L[finite]).max()))

    candidates = [(L[0], H[0]), (L[-1], H[-1])]
    if identifiable:
        for i in range(1, n_grid - 1):
            if L[i] >= L[i - 1] and L[i] >= L[i + 1] and np.isfinite(L[i]):
                res = optimize.minimize_scalar(lambda h: -ll(h), bounds=(H[i - 1], H[i + 1]),
                                               method="bounded",
                                               options={"xatol": 1e-10})
                candidates.append((-res.fun, float(res.x)))
                candidates.append((L[i], H[i]))
    # best LL; ties -> smallest h (so a flat profile gives h = 0)
    best_ll = max(c[0] for c in candidates)
    h_best = min(c[1] for c in candidates if c[0] >= best_ll)
    llv, beta, sigma2 = profile_loglik(h_best, y, K, X, reml)
    # -1/2 log(1-h) grows by ~3.45 from 1-1e-6 to 1-1e-9; a bounded profile is flat there
    ml_unbounded = bool(not reml and h_max >= 1.0
                        and ll(1 - 1e-9) > ll(1 - 1e-6) + 1.0)
    return OracleFit(h=float(h_best), ll=llv, sigma2=sigma2, beta=beta, reml=reml,
                     identifiable=identifiable,
                     at_boundary=bool(h_best <= 1e-6 or h_best >= h_max - 1e-6),
                     h_grid=H, ll_grid=L, ml_unbounded=ml_unbounded)


def wald(y: np.ndarray, K: np.ndarray, x: np.ndarray,
         X0: Optional[np.ndarray] = None, reml: bool = True,
         h: Optional[float] = None, refit: bool = False,
         sigma2: str = "reml", n_grid: int = 201) -> OracleWald:
    """Wald t-test of the last column x in [X0, x].

    h is, in order of precedence: the given ``h``; a fit of the full model
    [X0, x] if ``refit``; otherwise a fit of the null model X0 (pylmm's
    default "no refit" behaviour).  ``reml`` picks the likelihood used for
    fitting h.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    X0 = _as_X(X0, n)
    X = np.hstack([X0, np.asarray(x, dtype=np.float64).reshape(n, 1)])
    K = np.asarray(K, dtype=np.float64)
    if h is None:
        f = fit(y, K, X if refit else X0, reml=reml, n_grid=n_grid)
        h = f.h
    else:
        llv, b, s2 = profile_loglik(h, y, K, X0, reml)
        f = OracleFit(h=float(h), ll=llv, sigma2=s2, beta=b, reml=reml,
                      identifiable=True, at_boundary=h in (0.0, 1.0),
                      h_grid=np.array([h]), ll_grid=np.array([llv]))
    p = X.shape[1]
    Hm = h * K + (1.0 - h) * np.eye(n)
    C = linalg.cho_factor(Hm, lower=True)
    XtHiX = X.T @ linalg.cho_solve(C, X)
    cov = linalg.inv(XtHiX)
    beta = cov @ (X.T @ linalg.cho_solve(C, y))
    r = y - X @ beta
    Q = float(r @ linalg.cho_solve(C, r))
    s2 = Q / (n - p) if sigma2 == "reml" else Q / n
    se = float(np.sqrt(s2 * cov[-1, -1]))
    t = float(beta[-1] / se)
    df = n - p
    return OracleWald(t=t, p=float(2 * stats.t.sf(abs(t), df)), beta=float(beta[-1]),
                      se=se, df=df, h=float(h), fit=f)


def pylmm_reml_offset(n: int, p: int) -> float:
    """pylmm REML LL minus oracle REML LL (a constant in h).

    pylmm (lmm.py ``LL``) writes the REML LL as the ML LL (with sigma^2 = Q/n)
    plus 1/2 [p log(2 pi Q/(n-p)) + log|X'X| - log|X'H^-1X|].  Subtracting l_R
    leaves n/2 log(n/(n-p)) - p/2, independent of h and of the data.
    """
    return 0.5 * n * np.log(n / (n - p)) - 0.5 * p
