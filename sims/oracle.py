"""Independent reference ("oracle") REML/ML fit of the pylmm model, for small n.

Model
-----
    y = X beta + u + e,   u ~ N(0, sigma^2 h K),   e ~ N(0, sigma^2 (1 - h) I)
    V(h, sigma^2) = sigma^2 H(h),   H(h) = h K + (1 - h) I,   h in [0, 1]

This is pylmm's parameterisation (sigma^2 is the *total* variance, h the
heritability).  The code here deliberately shares nothing with pylmm: it never
rotates by eigenvectors, it works on the dense n x n matrix H with a Cholesky
factorisation per h, and every formula is the textbook one.  It is O(n^3) per
likelihood evaluation, so keep n <= ~500 (n = 200: ~0.1 s per fit, n = 500:
~0.6 s).

Profile likelihoods (sigma^2 and beta profiled out), with p = rank(X) = ncol(X),
r = y - X beta_hat(h), Q(h) = r' H^-1 r:

    ML:    beta_hat = (X' H^-1 X)^-1 X' H^-1 y,   sigma2_hat = Q / n
           l_ML(h) = -1/2 [ n log(2 pi sigma2_hat) + log|H| + n ]

    REML:  sigma2_hat = Q / (n - p)
           l_R(h) = -1/2 [ (n-p) log(2 pi sigma2_hat) + log|H| + log|X' H^-1 X|
                           - log|X' X| + (n - p) ]

l_R is exactly the Gaussian log-likelihood of the n - p error contrasts A'y
(A'A = I, A'X = 0); the ``- log|X'X|`` term makes it independent of the
choice of A (tests/sims/test_sims_oracle.py checks this numerically).

Optimisation (``h_grid='fine'``, the default): l(h) is evaluated on a uniform
201-point grid over [0, 1] (both endpoints included); every interior grid
local maximum is refined with bounded Brent (scipy
``minimize_scalar(method='bounded')``) inside its bracket, and the best of the
refined points and the two endpoints wins.  This finds the global maximum of
a 1-D profile reliably, including boundary maxima at h = 0 and h = 1.

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
  rank ~110 BED kinship, y has residual mass there and l_ML -> -inf instead.)
  The MLE does not exist; ``fit`` then returns h just below 1 and sets
  ``ml_unbounded=True``.  REML is unaffected (the log|X'H^-1X| term cancels
  the divergence).  pylmm never sees this because it clamps eigenvalues
  < 1e-6 to 1e-6 and its grid stops at h = 0.99.
* When K = I (or any K with H(h) constant in h) h is unidentifiable: the
  profile is flat, ``fit`` returns ``identifiable=False`` and h = 0 (the first
  maximiser on the grid).  beta, sigma2 and the Wald test then reduce to OLS.
* Log-likelihood constants: pylmm's ML LL *is* the standard profile ML
  (sigma^2 = Q/n inside the LL), so raw ML values match ``ll``.  pylmm's REML
  LL is the standard l_R PLUS a constant: both depend on Q only through
  -(n-p)/2 log Q, but the constants differ by ``pylmm_reml_offset(n, p)`` =
  n/2 log(n/(n-p)) - p/2.  The argmax h is identical, but raw REML LLs must
  not be compared directly: compare h, beta, t, LL differences
  (ll(h_hat) - ll(0)), or ``OracleFit.ll_pylmm`` (= ll + offset for REML).
  tests/sims/test_sims_oracle.py verifies this at many h.

pylmm-convention switches
-------------------------
pylmm deviates from the textbook fit in several places, most of them
suspected bugs (docs/phase1-baseline.md).  Each deviation has its own switch,
defaulting to the textbook behaviour.  With every switch at its pylmm setting
the oracle reproduces pylmm (tests: ~1e-8 on h and t, on synthetic and real
BED data, including cases where pylmm's search fails); with the switches
flipped one at a time each effect is measured in isolation (study S4).

==================================  =====================  ===================
pylmm behaviour (bug #)             switch (textbook)      pylmm setting
==================================  =====================  ===================
eigenvalues of K < 1e-6 raised to   fit/wald               ``eig_clamp=1e-6``
1e-6 (matters as h -> 1; the BED K  ``eig_clamp=None``
has 1109/1219 below 1e-6)
grid 0..0.99 step 0.01 (#3),        fit/wald               ``h_grid='pylmm'``
``range(1, n-2)`` scan (#2), first  ``h_grid='fine'``
(not best) local optimum, Brent
bounded to the bracketing grid
interval (#4, fixed in Phase 5.1),
endpoint fallback
null fit always REML in ``GWAS``    wald ``null_reml``     ``null_reml=True,
while the test uses REML=... (#1)   = ``test_reml``        test_reml=False``
sigma reported/used as Q/(n-q)      fit reports both       ``sigma2_reml``;
even under ML (#9)                  ``sigma2_ml`` and      wald
                                    ``sigma2_reml``;       ``sigma_convention
                                    wald                   ='pylmm'``
                                    ``sigma_convention
                                    ='textbook'``
REML LL constant                    ``ll`` (standard)      ``ll_pylmm``
==================================  =====================  ===================

All Wald p-values use the t distribution with df = n - p (p counts the SNP),
as pylmm does.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from scipy import linalg, optimize, stats


@dataclasses.dataclass
class OracleFit:
    h: float
    ll: float  # standard profile (RE)ML log-likelihood at h
    sigma2: float  # = sigma2_reml if reml else sigma2_ml (see sigma2_kind)
    sigma2_ml: float  # Q/n at h
    sigma2_reml: float  # Q/(n-p) at h (what pylmm reports, also under ML)
    beta: np.ndarray
    reml: bool
    identifiable: bool  # False if the profile is flat in h
    at_boundary: bool  # h within 1e-6 of 0 or of the grid's upper end
    h_grid: np.ndarray  # grid the profile was evaluated on
    ll_grid: np.ndarray
    ml_unbounded: bool = False  # ML profile -> +inf as h -> 1 (see module docstring)
    n: int = 0
    p: int = 0
    eig_clamp: Optional[float] = None
    search: str = "fine"

    @property
    def sigma2_kind(self) -> str:
        return "reml (Q/(n-p))" if self.reml else "ml (Q/n)"

    @property
    def ll_pylmm(self) -> float:
        """``ll`` on pylmm's scale (adds the REML constant; ML unchanged)."""
        return self.ll + (pylmm_reml_offset(self.n, self.p) if self.reml else 0.0)


@dataclasses.dataclass
class OracleWald:
    t: float
    p: float
    beta: float
    se: float
    df: int
    h: float
    sigma2: float  # variance scale used for se
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


def clamp_kinship(K: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """K with eigenvalues below ``floor`` raised to ``floor`` (pylmm's
    ``Kva[Kva < 1e-6] = 1e-6``), rebuilt densely as V diag(d) V'."""
    K = np.asarray(K, dtype=np.float64)
    d, V = linalg.eigh(0.5 * (K + K.T))
    d = np.where(d < floor, floor, d)
    return (V * d) @ V.T


def _prep_K(K, eig_clamp):
    K = np.asarray(K, dtype=np.float64)
    return K if eig_clamp is None else clamp_kinship(K, eig_clamp)


def _profile(h, y, K, X, reml):
    """(ll, beta, Q) at h; ll = -inf if H(h) is not positive definite."""
    n, p = X.shape
    H = h * K + (1.0 - h) * np.eye(n)
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
    XtHiX = X.T @ linalg.cho_solve(C, X)
    try:
        beta = linalg.solve(XtHiX, X.T @ linalg.cho_solve(C, y), assume_a="pos")
    except linalg.LinAlgError:
        return singular
    r = y - X @ beta
    Q = float(r @ linalg.cho_solve(C, r))
    if reml:
        _, logdet_XtHiX = np.linalg.slogdet(XtHiX)
        _, logdet_XtX = np.linalg.slogdet(X.T @ X)
        ll = -0.5 * (
            (n - p) * np.log(2 * np.pi * Q / (n - p))
            + logdetH
            + logdet_XtHiX
            - logdet_XtX
            + (n - p)
        )
    else:
        ll = -0.5 * (n * np.log(2 * np.pi * Q / n) + logdetH + n)
    return float(ll), beta, Q


def profile_loglik(
    h: float,
    y: np.ndarray,
    K: np.ndarray,
    X: Optional[np.ndarray] = None,
    reml: bool = True,
    eig_clamp: Optional[float] = None,
):
    """(ll, beta_hat, sigma2_hat) at a fixed h; ll = -inf if H(h) is singular.
    sigma2_hat is Q/(n-p) for REML and Q/n for ML.  ``eig_clamp`` applies
    pylmm's eigenvalue floor to K first (an eigh per call; in loops clamp once
    with :func:`clamp_kinship`)."""
    K = _prep_K(K, eig_clamp)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    X = _as_X(X, len(y))
    n, p = X.shape
    ll, beta, Q = _profile(h, y, K, X, reml)
    return ll, beta, Q / ((n - p) if reml else n)


def _search_fine(ll, n_grid):
    H = np.linspace(0.0, 1.0, n_grid)
    L = np.array([ll(h) for h in H])
    finite = np.isfinite(L)
    span = L[finite].max() - L[finite].min() if finite.any() else 0.0
    identifiable = bool(span > 1e-9 * max(1.0, abs(L[finite]).max()))
    candidates = [(L[0], H[0]), (L[-1], H[-1])]
    if identifiable:
        for i in range(1, n_grid - 1):
            if L[i] >= L[i - 1] and L[i] >= L[i + 1] and np.isfinite(L[i]):
                res = optimize.minimize_scalar(
                    lambda h: -ll(h),
                    bounds=(H[i - 1], H[i + 1]),
                    method="bounded",
                    options={"xatol": 1e-10},
                )
                candidates.append((-res.fun, float(res.x)))
                candidates.append((L[i], H[i]))
    # best LL; ties -> smallest h (so a flat profile gives h = 0)
    best_ll = max(c[0] for c in candidates)
    h_best = min(c[1] for c in candidates if c[0] >= best_ll)
    return float(h_best), H, L, identifiable


def _search_pylmm(ll):
    """pylmm's LMM.fit/getMax search, step for step (lmm.py), on the oracle's
    likelihood.  The REML constant offset cannot change any comparison or
    Brent step, so the path is the same as pylmm's."""
    ngrids = 100
    H = np.array(range(ngrids)) / float(ngrids)  # 0, 0.01, ..., 0.99
    L = np.array([ll(h) for h in H])

    def LL_brent(h):
        if h < 0 or h > 1:
            return 1e6  # pylmm's guard
        v = ll(h)
        # pylmm takes log(h*Kva + 1 - h): any non-positive term (h > 1) gives nan
        return np.nan if not np.isfinite(v) else -v

    n = len(L)
    HOpt = []
    for i in range(1, n - 2):  # off-by-one kept (#2)
        if L[i - 1] < L[i] and L[i] > L[i + 1]:
            # bounded to the bracketing grid interval, like pylmm since Phase 5.1 (#4)
            res = optimize.minimize_scalar(
                LL_brent, bounds=(H[i - 1], H[i + 1]), method="bounded", options={"xatol": 1e-10}
            )
            HOpt.append(float(res.x))
    if len(HOpt) >= 1:
        h = HOpt[0]  # first, not best
    elif L[0] > L[n - 1]:
        h = H[0]
    else:
        h = H[n - 1]
    finite = np.isfinite(L)
    identifiable = bool(finite.any() and np.ptp(L[finite]) > 1e-9 * max(1.0, abs(L[finite]).max()))
    return float(h), H, L, identifiable


def fit(
    y: np.ndarray,
    K: np.ndarray,
    X: Optional[np.ndarray] = None,
    reml: bool = True,
    n_grid: int = 201,
    eig_clamp: Optional[float] = None,
    h_grid: str = "fine",
) -> OracleFit:
    """Maximise the profile (RE)ML over h; see module docstring.

    pylmm-convention switches: ``eig_clamp=1e-6`` (eigenvalue floor),
    ``h_grid='pylmm'`` (pylmm's exact search instead of the global one).
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    K = _prep_K(K, eig_clamp)
    X = _as_X(X, len(y))
    _check(y, K, X)
    n, p = X.shape

    def ll(h):
        return _profile(h, y, K, X, reml)[0]

    if h_grid == "fine":
        h_best, H, L, identifiable = _search_fine(ll, n_grid)
    elif h_grid == "pylmm":
        h_best, H, L, identifiable = _search_pylmm(ll)
    else:
        raise ValueError("h_grid must be 'fine' or 'pylmm'")
    llv, beta, Q = _profile(h_best, y, K, X, reml)
    # -1/2 log(1-h) grows by ~3.45 from 1-1e-6 to 1-1e-9; a bounded profile is flat there
    ml_unbounded = bool(not reml and ll(1 - 1e-9) > ll(1 - 1e-6) + 1.0)
    return OracleFit(
        h=h_best,
        ll=llv,
        sigma2=Q / ((n - p) if reml else n),
        sigma2_ml=Q / n,
        sigma2_reml=Q / (n - p),
        beta=beta,
        reml=reml,
        identifiable=identifiable,
        at_boundary=bool(h_best <= 1e-6 or h_best >= H[-1] - 1e-6),
        h_grid=H,
        ll_grid=L,
        ml_unbounded=ml_unbounded,
        n=n,
        p=p,
        eig_clamp=eig_clamp,
        search=h_grid,
    )


def wald(
    y: np.ndarray,
    K: np.ndarray,
    x: np.ndarray,
    X0: Optional[np.ndarray] = None,
    null_reml: bool = True,
    test_reml: Optional[bool] = None,
    h: Optional[float] = None,
    refit: bool = False,
    sigma_convention: str = "textbook",
    eig_clamp: Optional[float] = None,
    h_grid: str = "fine",
    n_grid: int = 201,
) -> OracleWald:
    """Wald t-test of the last column x in [X0, x].

    h: the given ``h``; else a fit of [X0, x] if ``refit``; else a fit of the
    null model X0 (pylmm's default, no refit).  ``null_reml`` is the
    likelihood used to fit h; ``test_reml`` (default = null_reml) the one for
    the test, which only matters through the variance scale:

    * ``sigma_convention='textbook'``: sigma^2 = Q/(n-p) if test_reml else Q/n
    * ``sigma_convention='pylmm'``: sigma^2 = Q/(n-p) always (bug 9)

    se^2 = sigma^2 [(X'H^-1X)^-1]_xx, p from t with df = n - p (p includes x).
    Before Phase 5.2, pylmm's ``GWAS(REML=False)`` was ``null_reml=True,
    test_reml=False, sigma_convention='pylmm'`` (bugs 1 and 9); since 5.2 it
    is ``null_reml=False`` with the textbook convention.
    """
    if sigma_convention not in ("textbook", "pylmm"):
        raise ValueError("sigma_convention must be 'textbook' or 'pylmm'")
    test_reml = null_reml if test_reml is None else test_reml
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(y)
    X0 = _as_X(X0, n)
    X = np.hstack([X0, np.asarray(x, dtype=np.float64).reshape(n, 1)])
    K = _prep_K(K, eig_clamp)
    Xf = X if refit else X0
    if h is None:
        f = fit(y, K, Xf, reml=null_reml, n_grid=n_grid, h_grid=h_grid)
        f.eig_clamp = eig_clamp
        h = f.h
    else:
        llv, b, Qf = _profile(h, y, K, Xf, null_reml)
        nf, pf = Xf.shape
        f = OracleFit(
            h=float(h),
            ll=llv,
            sigma2=Qf / ((nf - pf) if null_reml else nf),
            sigma2_ml=Qf / nf,
            sigma2_reml=Qf / (nf - pf),
            beta=b,
            reml=null_reml,
            identifiable=True,
            at_boundary=h in (0.0, 1.0),
            h_grid=np.array([h]),
            ll_grid=np.array([llv]),
            n=nf,
            p=pf,
            eig_clamp=eig_clamp,
            search="given",
        )
    p = X.shape[1]
    C = linalg.cho_factor(h * K + (1.0 - h) * np.eye(n), lower=True)
    cov = linalg.inv(X.T @ linalg.cho_solve(C, X))
    beta = cov @ (X.T @ linalg.cho_solve(C, y))
    r = y - X @ beta
    Q = float(r @ linalg.cho_solve(C, r))
    s2 = Q / (n - p) if (sigma_convention == "pylmm" or test_reml) else Q / n
    se = float(np.sqrt(s2 * cov[-1, -1]))
    t = float(beta[-1] / se)
    df = n - p
    return OracleWald(
        t=t,
        p=float(2 * stats.t.sf(abs(t), df)),
        beta=float(beta[-1]),
        se=se,
        df=df,
        h=float(h),
        sigma2=s2,
        fit=f,
    )


def pylmm_reml_offset(n: int, p: int) -> float:
    """pylmm REML LL minus oracle REML LL (a constant in h).

    pylmm (lmm.py ``LL``) writes the REML LL as the ML LL (with sigma^2 = Q/n)
    plus 1/2 [p log(2 pi Q/(n-p)) + log|X'X| - log|X'H^-1X|].  Subtracting l_R
    leaves n/2 log(n/(n-p)) - p/2, independent of h and of the data.
    """
    return 0.5 * n * np.log(n / (n - p)) - 0.5 * p
