"""Phase 6.5: the exact per-SNP refit (`LMM.fit_snps`), the (reduced-ML) LRT and the score test.

* `LMM._snp_terms`: the likelihood is the one `fit` maximises, and its analytic
  first and second derivatives in h match finite differences;
* `LMM.fit_snps` reaches the maximum of ``fit(X=x)`` (the per-SNP grid search)
  and of the dense oracle, at the boundaries too, with badly scaled covariates
  and with a singular kinship; it falls back to the scalar search when Newton
  fails;
* the score and LRT statistics equal GEMMA's formulas (Zhou & Stephens 2012),
  implemented here independently on the dense n x n matrices, without the
  eigen-rotation (GEMMA 0.98.5's own outputs matched these formulas to its 6
  printed digits, see the PR);
* `gwas(test=...)`, `LMM.association_tests` and ``pylmm gwas --test``: the
  batched and per-SNP paths agree, missing calls and phenotypes, option errors.
"""

import os
import warnings

import helpers as h
import numpy as np
import pytest
from scipy import linalg, optimize, stats

from pylmm import lmm
from pylmm.lmm import LMM, calculate_kinship, gwas


def _data(n=120, m=30, seed=0, h2=0.5, n_cov=0, kind="grm", effect=0.4):
    rng = np.random.default_rng(seed)
    W = rng.binomial(2, rng.uniform(0.1, 0.5, 600 + m), size=(n, 600 + m)).astype(float)
    X = W[:, 600:]
    if kind == "strains":  # 4 replicate animals per strain: K singular, h = 1 reachable
        s = n // 4
        k_strains = calculate_kinship(W[:s, :600])
        idx = np.repeat(np.arange(s), 4)
        K = k_strains[np.ix_(idx, idx)]
        X = X[:s][idx]
    elif kind == "pd":
        K = calculate_kinship(W[:, :600]) + 0.05 * np.eye(n)
    else:
        K = calculate_kinship(W[:, :600])
    kva, kve = linalg.eigh(K)
    u = kve @ (np.sqrt(np.clip(kva, 0, None)) * rng.normal(size=n))
    y = np.sqrt(h2) * u + np.sqrt(1 - h2) * rng.normal(size=n) + effect * X[:, 0]
    X0 = None
    if n_cov:
        cov = [rng.normal(size=n) * 10.0**k for k in range(n_cov)]
        X0 = np.column_stack([np.ones(n), *cov])
    X = X[:, X.std(axis=0) > 0]
    return y, X, K, X0


def _grid_refit(L, X, reml):
    """``fit(X=x)`` then `association` for every column: the per-SNP grid search."""
    out = []
    for j in range(X.shape[1]):
        hh, _, _, ll = L.fit(X=X[:, j], reml=reml)
        t, p, b, v = L.association(X[:, j], reml=reml, return_beta=True)
        out.append((float(hh), ll, t, p, b, v))
    return np.array(out).T


# ---------------------------------------------------------------------------
# the likelihood and its derivatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
@pytest.mark.parametrize("n_cov", [0, 2])
def test_snp_terms_is_fits_likelihood_and_derivatives_match(reml, n_cov):
    y, X, K, X0 = _data(n_cov=n_cov)
    L = LMM(y, K, X0=X0)
    xp, yp, nrm2 = L._snp_rotate(X[:, :6])
    for hh in (0.0, 0.2, 0.55, 0.9, 0.99, 0.999, 1.0):
        hv = np.full(6, hh)
        t = L._snp_terms(hv, xp, yp, nrm2, reml, derivatives=True)
        ref = []
        for j in range(6):
            L.x0t_stack[:, L.q] = L.kve.T @ X[:, j]
            design = lmm._orthonormal_columns(L.x0t_stack)
            ref.append(L.log_likelihood(hh, design, stack=False, reml=reml).ll)
        h.assert_close(t["ll"], ref, rtol=0, atol=h.REFIT_LL_ATOL, what="ll at h=%g" % hh)
        grid = L._snp_grid_ll(np.array([hh]), xp, yp, nrm2, reml)[0]
        h.assert_close(grid, ref, rtol=0, atol=h.REFIT_LL_ATOL, what="grid ll at h=%g" % hh)
        if hh > 0.99:
            continue  # the derivatives are checked where a difference formula is accurate

        def f(v, key):
            return L._snp_terms(np.full(6, v), xp, yp, nrm2, reml, True)[key]

        for key, dkey in (("ll", "d1"), ("d1", "d2")):
            fd = _difference(lambda v, key=key: f(v, key), hh, 1e-5)
            err = np.abs(t[dkey] - fd).max()
            assert err <= h.DERIV_RTOL * (np.abs(fd).max() + 1.0), (dkey, hh, err)


@pytest.mark.parametrize("n_cov", [0, 2])
def test_snp_terms_reml_derivatives_at_h1(n_cov):
    """At exactly h = 1 (REML, centred K): l' as everywhere, l'' to DERIV_H1_RTOL."""
    y, X, K, X0 = _data(n_cov=n_cov)
    L = LMM(y, K, X0=X0)
    xp, yp, nrm2 = L._snp_rotate(X[:, :6])
    t = L._snp_terms(np.ones(6), xp, yp, nrm2, True, True)

    def f(v, key):
        return L._snp_terms(np.full(6, v), xp, yp, nrm2, True, True)[key]

    e = 1e-5
    for key, dkey, tol in (("ll", "d1", h.DERIV_RTOL), ("d1", "d2", h.DERIV_H1_RTOL)):
        fd = (3 * f(1.0, key) - 4 * f(1.0 - e, key) + f(1.0 - 2 * e, key)) / (2 * e)
        err = np.abs(t[dkey] - fd).max()
        assert err <= tol * (np.abs(fd).max() + 1.0), (dkey, err)


def _difference(f, x, e):
    """Fourth-order central difference of ``f`` at ``x`` (second-order forward near 0)."""
    if x - 2 * e < 0:
        return (-3 * f(x) + 4 * f(x + e) - f(x + 2 * e)) / (2 * e)
    return (f(x - 2 * e) - 8 * f(x - e) + 8 * f(x + e) - f(x + 2 * e)) / (12 * e)


# ---------------------------------------------------------------------------
# fit_snps: the same maximum as fit(X=x)
# ---------------------------------------------------------------------------


def _check_refit(L, X, reml, what):
    new = L.fit_snps(X, reml=reml)
    old = _grid_refit(L, X, reml)
    hh, ll, t, p, b, v = old
    assert np.all(new.ll >= ll - h.REFIT_LL_ATOL), "%s: exact refit below the grid search" % what
    assert np.all(ll >= new.ll - h.REFIT_LL_ATOL), "%s: grid search above the exact refit" % what
    h.assert_close(new.h, hh, rtol=0, atol=h.REFIT_H_ATOL, what=what + " h")
    for name, a, e in (("t", new.t, t), ("beta", new.beta, b), ("beta_var", new.beta_var, v)):
        h.assert_refit_close(a, e, what="%s %s" % (what, name))
    h.assert_refit_close(new.p, p, what=what + " p")
    return new


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
@pytest.mark.parametrize(
    ("kind", "h2", "n_cov"),
    [("grm", 0.0, 0), ("grm", 0.5, 2), ("grm", 0.98, 1), ("strains", 1.0, 0), ("pd", 0.3, 1)],
)
def test_fit_snps_matches_the_grid_search(reml, kind, h2, n_cov):
    y, X, K, X0 = _data(kind=kind, h2=h2, n_cov=n_cov, seed=int(10 * h2) + n_cov)
    L = LMM(y, K, X0=X0)
    new = _check_refit(L, X[:, :15], reml, "%s h2=%g" % (kind, h2))
    assert L._snp_fallbacks == 0
    if kind == "strains" and reml:
        assert (new.h == 1.0).any()  # the boundary is reached, as fit(X=x) reaches it


def test_fit_snps_ml_spike_at_1_needs_no_fallback():
    """ML, centred GRM with an intercept: the profile dips below 1 and rises into its
    unbounded spike at h = 1, so the top bracket has no (+, -) change of l'.  Its
    maximum is the end h = 1, not a reason for the scalar search (it was, for every
    SNP, in a first version of fit_snps: the LRT then cost ~3 ms per SNP)."""
    y, X, K, _ = _data(n=300, h2=0.0, m=20)
    L = LMM(y, K)
    _check_refit(L, X, False, "ml spike")
    assert L._snp_fallbacks == 0


def test_fit_snps_matches_the_grid_search_on_hmdp():
    """94 inbred strains (singular K) with the HMDP SNPs mean-imputed."""
    Y = np.genfromtxt(h.HMDP_Y)
    K = np.loadtxt(h.HMDP_K_TXT)
    X = h.load_hmdp_snps(60)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    X = X[:, X.std(axis=0) > 0]
    L = LMM(Y, K)
    for reml in (True, False):
        _check_refit(L, X[L.nonmissing], reml, "hmdp reml=%s" % reml)


def test_fit_snps_with_a_raw_date_covariate():
    """Badly scaled covariate: the same h as with the centred one, and as fit(X=x)."""
    y, X, K, _ = _data(h2=0.98, seed=5)
    n = len(y)
    date = 2e7 + 1e3 * np.random.default_rng(1).normal(size=n)
    raw = LMM(y, K, X0=np.column_stack([np.ones(n), date]))
    cen = LMM(y, K, X0=np.column_stack([np.ones(n), date - date.mean()]))
    a = _check_refit(raw, X[:, :10], True, "raw date")
    b = cen.fit_snps(X[:, :10])
    h.assert_close(a.h, b.h, rtol=0, atol=h.REFIT_H_ATOL, what="raw vs centred h")
    h.assert_refit_close(a.t, b.t, what="raw vs centred t")


def test_fit_snps_matches_the_oracle():
    from sims import oracle

    y, X, K, X0 = _data(n=80, n_cov=1, h2=0.6, seed=3)
    L = LMM(y, K, X0=X0)
    for reml in (True, False):
        new = L.fit_snps(X[:, :3], reml=reml)
        for j in range(3):
            f = oracle.fit(y, K, X=np.column_stack([X0, X[:, j]]), reml=reml, eig_clamp=1e-6)
            assert f.ll_pylmm - new.ll[j] <= h.ORACLE_LL_ATOL
            assert abs(f.h - new.h[j]) <= h.REFIT_H_ATOL or f.h == new.h[j]


def test_fit_snps_flat_profile_gives_h0():
    """K = I: h is not identifiable, h = 0 (as fit), OLS statistics."""
    y, X, _, _ = _data()
    L = LMM(y, np.eye(len(y)))
    new = L.fit_snps(X[:, :5])
    assert np.all(new.h == 0.0)
    old = _grid_refit(L, X[:, :5], True)
    h.assert_refit_close(new.t, old[2], what="flat t")


def test_fit_snps_falls_back_to_the_scalar_search(monkeypatch):
    y, X, K, X0 = _data(n_cov=1)
    L = LMM(y, K, X0=X0)
    ref = L.fit_snps(X[:, :8])
    monkeypatch.setattr(lmm, "_NEWTON_MAXITER", 1)  # Newton cannot converge
    new = L.fit_snps(X[:, :8])
    assert L._snp_fallbacks > 0
    h.assert_close(new.h, ref.h, rtol=0, atol=h.REFIT_H_ATOL, what="fallback h")
    assert np.all(np.abs(new.ll - ref.ll) <= h.REFIT_LL_ATOL)


def test_fit_snps_nan_columns_and_errors():
    y, X, K, _ = _data(m=4)
    L = LMM(y, K)
    Z = np.column_stack([X[:, 0], np.ones(len(y)), X[:, 1]])  # a constant: not estimable
    r = L.fit_snps(Z)
    assert np.isnan(r.t[1]) and np.isnan(r.h[1]) and np.isfinite(r.t[[0, 2]]).all()
    with pytest.raises(ValueError, match="ngrids"):
        L.fit_snps(X, ngrids=0)
    with pytest.raises(ValueError, match="shape"):
        L.fit_snps(X[:-1])
    with pytest.raises(ValueError, match="legacy_fit"):
        LMM(y, K, legacy_fit=True).fit_snps(X)
    assert L.fit_snps(X[:, :0]).h.shape == (0,)


# ---------------------------------------------------------------------------
# GEMMA's formulas, dense and in lambda = h / (1 - h)
# ---------------------------------------------------------------------------


def _gemma_p(lam, K, W, y):
    """GEMMA's P = H^-1 - H^-1 W (W'H^-1W)^-1 W'H^-1, H = lam K + I (dense)."""
    n = len(y)
    h_inv = linalg.inv(lam * K + np.eye(n))
    hi_w = h_inv @ W
    return h_inv - hi_w @ linalg.solve(W.T @ hi_w, hi_w.T)


def _gemma_logl(lam, K, W, y, reml):
    """GEMMA's profile log-likelihoods (lmm.cpp LogL_f / LogRL_f), dense."""
    n, c = W.shape
    H = lam * K + np.eye(n)
    P = _gemma_p(lam, K, W, y)
    ypy = y @ P @ y
    logdet_h = np.linalg.slogdet(H)[1]
    if not reml:
        return (
            0.5 * n * (np.log(n) - np.log(2 * np.pi) - 1.0) - 0.5 * logdet_h - 0.5 * n * np.log(ypy)
        )
    df = n - c
    logdet_whw = np.linalg.slogdet(W.T @ linalg.solve(H, W))[1]
    return (
        0.5 * df * (np.log(df) - np.log(2 * np.pi) - 1.0)
        - 0.5 * logdet_h
        - 0.5 * logdet_whw
        - 0.5 * df * np.log(ypy)
    )  # GEMMA leaves out +1/2 log|W'W|, a constant


def _gemma_max(K, W, y, reml):
    """lambda maximising GEMMA's likelihood, by a dense bounded search over h."""
    hs = np.linspace(0.0, 0.99, 100)
    vals = [_gemma_logl(hh / (1 - hh), K, W, y, reml) for hh in hs]
    i = int(np.argmax(vals))
    res = optimize.minimize_scalar(
        lambda hh: -_gemma_logl(hh / (1 - hh), K, W, y, reml),
        bounds=(hs[max(i - 1, 0)], hs[min(i + 1, 99)]),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return res.x / (1 - res.x), -res.fun


def test_score_lrt_and_refit_wald_equal_gemmas_formulas():
    """On a positive definite K (no eigenvalue floor, bounded ML, nothing for the LRT to
    drop), with an intercept and a covariate: GEMMA's Wald (-lmm 1: REML refit), LRT
    (-lmm 2) and score (-lmm 3: the null ML lambda, F with n in the numerator)."""
    y, X, K, X0 = _data(n=100, kind="pd", h2=0.5, n_cov=1, seed=9, effect=0.6)
    X = X[:, :5]
    n, c = X0.shape
    df = n - c - 1
    L = LMM(y, K, X0=X0)
    assert L._lrt_model() is L  # positive definite K: the plain ML LRT
    L.fit(reml=False)
    lam0, l0 = _gemma_max(K, X0, y, reml=False)
    assert 0.01 < lam0 < 50  # interior: bounded ML
    lrt = L.lrt_batch(X)
    score = L.score_batch(X, reml=False)
    refit = L.fit_snps(X, reml=True)
    for j in range(X.shape[1]):
        x = X[:, j]
        W1 = np.column_stack([X0, x])
        # score at the null ML lambda
        P0 = _gemma_p(lam0, K, X0, y)
        pyy, pxx, pxy = y @ P0 @ y, x @ P0 @ x, x @ P0 @ y
        p_score = stats.f.sf(n * pxy * pxy / (pyy * pxx), 1, df)
        # LRT of the ML fits
        _, l1 = _gemma_max(K, W1, y, reml=False)
        p_lrt = stats.chi2.sf(2.0 * (l1 - l0), 1)
        # Wald at the per-SNP REML lambda
        lam1, _ = _gemma_max(K, W1, y, reml=True)
        p0r = _gemma_p(lam1, K, X0, y)
        p1r = _gemma_p(lam1, K, W1, y)
        pyy, px_yy = y @ p0r @ y, y @ p1r @ y
        p_wald = stats.f.sf((pyy - px_yy) * df / px_yy, 1, df)
        np.testing.assert_allclose(score.p[j], p_score, rtol=h.GEMMA_RTOL, err_msg="score")
        np.testing.assert_allclose(lrt.p[j], p_lrt, rtol=h.GEMMA_RTOL, err_msg="lrt")
        np.testing.assert_allclose(refit.p[j], p_wald, rtol=h.GEMMA_RTOL, err_msg="wald")
        np.testing.assert_allclose(
            refit.h[j] / (1 - refit.h[j]), lam1, rtol=h.GEMMA_LAMBDA_RTOL, err_msg="lambda"
        )


def test_score_is_a_monotone_function_of_the_wald_test():
    y, X, K, _ = _data(n_cov=0)
    L = LMM(y, K)
    L.fit()
    s = L.score_batch(X)
    w = L.association_batch(X)
    nq = L.N - L.q
    u = s.t**2 / nq  # REML: t_score^2 = (N - q) u, t_wald^2 = (N - q - 1) u / (1 - u)
    h.assert_batch_close(w.t**2, (nq - 1) * u / (1 - u), what="wald from score")
    h.assert_batch_close(s.beta, w.beta, what="beta")
    strong = u > 1.0 / nq  # |t_score| > 1: there the score statistic is the smaller one
    assert strong.any() and np.all(s.p[strong] >= w.p[strong])
    assert np.all(s.p[~strong] <= w.p[~strong])
    h.assert_p_matches_t(s.t, s.p, L.N - L.q - 1, what="score p")
    # the floor: t_score^2 <= N - q, so p >= P(|t(N - q - 1)| > sqrt(N - q))
    assert np.all(s.t**2 <= nq * (1 + 1e-12))


# ---------------------------------------------------------------------------
# the LRT's reduced ML, against a dense reference
# ---------------------------------------------------------------------------


def _dense_reduced_ml(y, K, X, floor=1e-6):
    """The reduced-ML maximum, densely: (h, ll, r).

    Z = the vectors that are both null directions of the floored K and in span(X)
    (computed as null((I - P_X) V_N), a different route from the code's);
    the likelihood is that of Q'y, Q an orthonormal basis of the complement of Z,
    with the covariance Q'(h K + (1 - h) I)Q and the design Q'X (rank q - r),
    maximised over a fine grid of h with bounded Brent on every grid peak.
    """
    n = len(y)
    kva, kve = linalg.eigh(K)
    kva = np.where(kva < floor, floor, kva)
    k_floor = (kve * kva) @ kve.T
    v_null = kve[:, kva <= floor]
    q_x = linalg.qr(X, mode="economic")[0]
    resid = v_null - q_x @ (q_x.T @ v_null)
    Z = np.zeros((n, 0))
    if v_null.shape[1]:  # v_null is orthonormal: an absolute cutoff on resid's singular values
        _, sv, vt = linalg.svd(resid, full_matrices=True)
        sv = np.r_[sv, np.zeros(vt.shape[0] - len(sv))]
        Z = v_null @ vt[sv < 1e-8].T
    r = Z.shape[1]
    Q = linalg.null_space(Z.T) if r else np.eye(n)
    yr = Q.T @ y
    # the design's span after dropping Z (absolute cutoff: X's columns can be dropped whole)
    ux, sx, _ = linalg.svd(Q.T @ q_x, full_matrices=False)
    x_red = ux[:, sx > 1e-8]
    m = n - r

    def ll(hh):
        H = Q.T @ (hh * k_floor + (1 - hh) * np.eye(n)) @ Q
        C = linalg.cho_factor(H)
        hi_y = linalg.cho_solve(C, yr)
        if x_red.shape[1]:
            hi_x = linalg.cho_solve(C, x_red)
            b = linalg.solve(x_red.T @ hi_x, x_red.T @ hi_y)
            res = yr - x_red @ b
        else:
            res = yr
        q_res = res @ linalg.cho_solve(C, res)
        logdet = 2 * np.log(np.diag(C[0])).sum()
        return -0.5 * (m * np.log(2 * np.pi * q_res / m) + logdet + m)

    hs = np.linspace(0.0, 1.0, 401)
    vals = np.array([ll(v) for v in hs])
    cands = [(vals[0], 0.0), (vals[-1], 1.0)]
    for i in range(1, len(hs) - 1):
        if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            res = optimize.minimize_scalar(
                lambda v: -ll(v),
                bounds=(hs[i - 1], hs[i + 1]),
                method="bounded",
                options={"xatol": 1e-12},
            )
            cands.append((-res.fun, res.x))
    best = max(cands)
    return best[1], best[0], r


@pytest.mark.parametrize(
    ("kind", "h2", "n_cov"),
    [("grm", 0.0, 0), ("grm", 0.3, 1), ("grm", 0.8, 2), ("centred-strains", 0.5, 0)],
)
def test_lrt_reduced_ml_matches_a_dense_reference(kind, h2, n_cov):
    """pylmm's centred kinship (K 1 = 0) with an intercept: the plain ML is unbounded
    as h -> 1, the reduced ML drops that one direction; a centred strain-expanded K
    (many null directions, one of them the intercept's) drops one and keeps the rest."""
    y, X, K, X0 = _data(
        n=80,
        kind="strains" if kind == "centred-strains" else "grm",
        h2=h2,
        n_cov=n_cov,
        seed=31 + n_cov,
        effect=0.5,
    )
    n = len(y)
    if kind == "centred-strains":
        P = np.eye(n) - 1.0 / n
        K = P @ K @ P
    x0_dense = np.ones((n, 1)) if X0 is None else X0
    L = LMM(y, K, X0=X0)
    M = L._lrt_model()
    n_reduced = M.N
    assert M is not L and M._dropped == 1 and n_reduced == n - 1
    h0, l0 = L.lrt_null_fit()
    hd, ld, r = _dense_reduced_ml(y, K, x0_dense)
    assert r == 1
    assert abs(l0 - ld) <= h.REDUCED_ML_LL_ATOL and abs(h0 - hd) <= h.REFIT_H_ATOL
    assert h0 < 1.0 or h2 >= 0.8  # an interior maximum where the plain ML sits at 1
    res = L.lrt_batch(X[:, :4])
    for j in range(4):
        _, l1, r1 = _dense_reduced_ml(y, K, np.column_stack([x0_dense, X[:, j]]))
        assert r1 == 1  # the same direction is dropped with the SNP: nested
        stat = max(2 * (l1 - ld), 0.0)
        assert abs(res.t[j] ** 2 - stat) <= h.REDUCED_ML_LL_ATOL * 4, (j, res.t[j] ** 2, stat)
    if kind == "grm" and h2 < 0.8:  # the plain ML fit of the same data sits at the spike
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert LMM(y, K, X0=X0).fit(reml=False)[0] > 0.999


def test_lrt_null_fit_leaves_the_model_alone():
    y, X, K, _ = _data()
    L = LMM(y, K)
    h_reml = L.fit()[0]
    before = (L.opt_h, L.opt_sigma, L.opt_ll, L.ll_grid.copy())
    L.lrt_batch(X)
    L.association_tests(X, "lrt", batch=False)
    L.association_tests(X, "wald", refit=True, batch=False)
    assert (L.opt_h, L.opt_sigma, L.opt_ll) == before[:3] and L.opt_h == h_reml
    np.testing.assert_array_equal(L.ll_grid, before[3])


def test_lrt_null_fit_and_ll0():
    y, X, K, _ = _data()
    L = LMM(y, K)
    h0, ll0 = L.lrt_null_fit()
    assert L.lrt_null_fit() == (h0, ll0)  # cached
    a = L.lrt_batch(X)
    b = LMM(y, K).lrt_batch(X, ll0=ll0)
    h.assert_close(a.p, b.p, what="ll0 given")
    assert np.all(a.p >= 0) and np.all(np.sign(a.t) == np.sign(a.beta))


def test_lrt_has_no_ml_warning():
    """The reduced ML has a maximum: no "the ML likelihood has no maximum" warning."""
    y, X, K, _ = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        gwas(y, X, K, test="lrt")


# ---------------------------------------------------------------------------
# gwas(test=...) and LMM.association_tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("test", "refit"), [("wald", True), ("lrt", False)], ids=str)
def test_gwas_tests_batched_vs_per_snp_with_missing_values(test, refit):
    y, X, K, X0 = _data(n=100, m=25, n_cov=1)
    y = y.copy()
    y[[4, 40]] = np.nan
    X = X.copy()
    X[[3, 7], 2] = np.nan  # missing calls: tested on a subset
    X[:, 5] = 1.0  # constant
    a = gwas(y, X, K, X0=X0, test=test, refit=refit)
    b = gwas(y, X, K, X0=X0, test=test, refit=refit, batch=False)
    assert np.isnan(a[0][5]) and np.isnan(b[0][5])
    h.assert_refit_close(a[0], b[0], what="t")
    h.assert_refit_close(a[1], b[1], what="p")


def test_gwas_default_test_is_the_wald_test_unchanged():
    y, X, K, _ = _data()
    assert gwas(y, X, K) == gwas(y, X, K, test="wald")


def test_gwas_lrt_matches_lrt_batch():
    y, X, K, _ = _data(effect=1.0)
    t, p = gwas(y, X, K, test="lrt")
    assert gwas(y, X, K, test="lrt", refit=True, reml=False) == (t, p)  # neither applies
    r = LMM(y, K).lrt_batch(X)
    h.assert_close(t, r.t, what="t")
    h.assert_close(p, r.p, what="p")
    assert p[0] < 1e-3  # the causal SNP


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"test": "f"}, "test must be one of"),
        ({"test": "score"}, "the score test is LMM.score_batch"),
        ({"test": "lrt", "legacy_fit": True}, "only the Wald test"),
    ],
)
def test_gwas_test_option_errors(kw, match):
    y, X, K, _ = _data(m=2)
    with pytest.raises(ValueError, match=match):
        gwas(y, X, K, **kw)


def test_association_tests_per_snp_path():
    y, X, K, _ = _data(m=8)
    L = LMM(y, K)
    L.fit()
    for refit in (False, True):
        a = L.association_tests(X, "wald", refit=refit)
        b = L.association_tests(X, "wald", refit=refit, batch=False)
        h.assert_refit_close(a.t, b.t, what="wald %s" % refit)
    a = L.association_tests(X, "lrt")
    b = L.association_tests(X, "lrt", batch=False)
    h.assert_refit_close(a.t, b.t, what="lrt")
    h.assert_refit_close(a.beta, b.beta, what="lrt beta")


# ---------------------------------------------------------------------------
# pylmm gwas --test
# ---------------------------------------------------------------------------


def _emma(tmp_path, nsnps=40):
    emma = h.write_emma_subset(tmp_path, nsnps)
    return [
        "gwas",
        "--emma-geno",
        emma,
        "--emma-pheno",
        h.HMDP_Y,
        "--kinship",
        h.write_headerless_kinship(tmp_path),
    ]


def test_cli_lrt_matches_the_library(tmp_path):
    out = str(tmp_path / "out_lrt")
    log = h.run_cli([*_emma(tmp_path), "-v", "--test", "lrt", "-o", out], cwd=tmp_path)
    assert "--test lrt" in log and "ML" in log
    ids, got = h.load_gwas_out(out)
    Y = np.genfromtxt(h.HMDP_Y)
    K = np.loadtxt(h.HMDP_K_TXT)
    X = h.load_hmdp_snps(40)
    t, p = gwas(Y, X, K, test="lrt")
    # the CLI imputes missing calls by the mean (the reader); compare complete SNPs
    full = ~np.isnan(X).any(axis=0)
    h.assert_close(got[full, 2], np.asarray(t)[full], rtol=1e-6, atol=1e-9, what="F_STAT")
    h.assert_close(got[full, 3], np.asarray(p)[full], rtol=1e-6, atol=0, what="P_VALUE")


def test_cli_refit_batched_vs_no_batch(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    h.run_cli([*_emma(tmp_path, 20), "--refit", "-o", a], cwd=tmp_path)
    h.run_cli([*_emma(tmp_path, 20), "--refit", "--no-batch", "-o", b], cwd=tmp_path)
    ids_a, got_a = h.load_gwas_out(a)
    ids_b, got_b = h.load_gwas_out(b)
    assert ids_a == ids_b
    for j in range(4):
        h.assert_refit_close(got_a[:, j], got_b[:, j], what="column %d" % j)


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        (["--test", "score"], "invalid choice"),
        (["--test", "lrt", "--legacy-fit"], "only the Wald test"),
        (["--test", "anova"], "invalid choice"),
    ],
)
def test_cli_test_option_errors(tmp_path, flags, match):
    out = h.run_cli([*_emma(tmp_path, 5), *flags, "-o", "o"], cwd=tmp_path, expect_rc=2)
    assert match in out
    assert not os.path.exists(tmp_path / "o")
