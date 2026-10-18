"""Validation of the independent oracle (sims/oracle.py), and a quick check
of pylmm against it."""

import numpy as np
import pytest
from scipy import linalg, stats

from sims import genotypes, kinship, oracle


def _blocks(sizes):
    return linalg.block_diag(*[np.ones((s, s)) for s in sizes])


def _grouped_y(sizes, sg=1.0, se=1.0, mu=2.0, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(len(sizes)) * sg
    return np.concatenate([mu + a[i] + rng.standard_normal(s) * se
                           for i, s in enumerate(sizes)])


def _ols(y, X):
    beta, rss, *_ = np.linalg.lstsq(X, y, rcond=None)
    n, p = X.shape
    s2 = float(rss[0]) / (n - p)
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, s2, beta / np.sqrt(np.diag(cov))


# ---------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reml", [True, False])
def test_identity_K_flat_profile_reduces_to_ols(reml):
    rng = np.random.default_rng(0)
    n = 40
    X = np.c_[np.ones(n), rng.standard_normal(n)]
    y = X @ [1.0, 0.5] + rng.standard_normal(n)
    f = oracle.fit(y, np.eye(n), X, reml=reml)
    assert not f.identifiable
    assert f.h == 0.0
    assert np.ptp(f.ll_grid) < 1e-9
    beta, s2, t = _ols(y, X)
    np.testing.assert_allclose(f.beta, beta, rtol=1e-10)
    np.testing.assert_allclose(f.sigma2, s2 if reml else s2 * (n - 2) / n, rtol=1e-10)
    # Wald at any h equals the OLS t-test
    for h in (0.0, 0.3, 0.9):
        w = oracle.wald(y, np.eye(n), X[:, 1], X0=X[:, :1], h=h)
        np.testing.assert_allclose(w.t, t[1], rtol=1e-10)
        np.testing.assert_allclose(w.p, 2 * stats.t.sf(abs(t[1]), n - 2), rtol=1e-8)


def test_balanced_one_way_anova_closed_form():
    """K = blocks of ones (a groups of m): REML = ANOVA estimators,
    ML sigma_g^2 = ((1 - 1/a) MSB - MSW) / m (Searle et al. 1992)."""
    a, m = 12, 6
    y = _grouped_y([m] * a, sg=1.2, se=1.0, seed=3)
    Y = y.reshape(a, m)
    msb = m * ((Y.mean(1) - Y.mean()) ** 2).sum() / (a - 1)
    msw = ((Y - Y.mean(1, keepdims=True)) ** 2).sum() / (a * (m - 1))
    K = _blocks([m] * a)

    sg_reml = (msb - msw) / m
    f = oracle.fit(y, K, reml=True)
    assert sg_reml > 0 and f.identifiable and not f.at_boundary
    np.testing.assert_allclose(f.h, sg_reml / (sg_reml + msw), atol=1e-7)
    np.testing.assert_allclose(f.sigma2, sg_reml + msw, rtol=1e-6)

    sg_ml = ((1 - 1 / a) * msb - msw) / m
    f = oracle.fit(y, K, reml=False)
    np.testing.assert_allclose(f.h, sg_ml / (sg_ml + msw), atol=1e-7)
    np.testing.assert_allclose(f.sigma2, sg_ml + msw, rtol=1e-6)


def test_boundary_h0_when_msb_below_msw():
    a, m = 10, 5
    y = _grouped_y([m] * a, sg=0.0, se=1.0, seed=1)
    Y = y.reshape(a, m)
    msb = m * ((Y.mean(1) - Y.mean()) ** 2).sum() / (a - 1)
    msw = ((Y - Y.mean(1, keepdims=True)) ** 2).sum() / (a * (m - 1))
    if msb >= msw:
        pytest.skip("seed gave MSB >= MSW")
    f = oracle.fit(y, _blocks([m] * a), reml=True)
    assert f.h == 0.0 and f.at_boundary


def test_boundary_h1_with_pd_K():
    """y drawn from N(0, K) exactly (no noise), K positive definite: h -> 1."""
    rng = np.random.default_rng(0)
    n = 60
    A = rng.standard_normal((n, 3 * n))
    K = A @ A.T / (3 * n)
    y = linalg.cholesky(K, lower=True) @ rng.standard_normal(n)
    f = oracle.fit(y, K, reml=True)
    assert f.h > 0.97
    # singular K: likelihood at h = 1 is -inf, never selected
    g = genotypes.unrelated(n, 20, seed=0)
    Ks = kinship.kinship(g.G)
    assert oracle.profile_loglik(1.0, y, Ks)[0] == -np.inf
    assert oracle.fit(y, Ks).h < 1.0


# ---------------------------------------------------------------------------
# likelihood identities (no closed-form h needed)
# ---------------------------------------------------------------------------

def _random_problem(n=50, seed=0):
    rng = np.random.default_rng(seed)
    g = genotypes.unrelated(n, 300, seed=seed)
    K = kinship.kinship(g.G)
    X = np.c_[np.ones(n), rng.standard_normal(n)]
    y = X @ [0.5, 1.0] + rng.standard_normal(n)
    return y, K, X


def test_ml_loglik_equals_mvn_logpdf():
    y, K, X = _random_problem()
    h = 0.37
    ll, beta, s2 = oracle.profile_loglik(h, y, K, X, reml=False)
    V = s2 * (h * K + (1 - h) * np.eye(len(y)))
    np.testing.assert_allclose(ll, stats.multivariate_normal(X @ beta, V).logpdf(y), rtol=1e-10)


def test_reml_loglik_equals_error_contrast_likelihood():
    y, K, X = _random_problem(seed=1)
    n, p = X.shape
    h = 0.42
    ll, beta, s2 = oracle.profile_loglik(h, y, K, X, reml=True)
    # orthonormal basis of the complement of col(X)
    Qf, _ = np.linalg.qr(X, mode="complete")
    A = Qf[:, p:]
    V = s2 * (h * K + (1 - h) * np.eye(n))
    ref = stats.multivariate_normal(np.zeros(n - p), A.T @ V @ A).logpdf(A.T @ y)
    np.testing.assert_allclose(ll, ref, rtol=1e-10)
    # and s2 maximises it (profile): perturbing sigma^2 lowers the likelihood
    for f in (0.9, 1.1):
        alt = stats.multivariate_normal(np.zeros(n - p), A.T @ (f * V) @ A).logpdf(A.T @ y)
        assert alt < ref


def test_fit_is_global_max_of_fine_grid():
    y, K, X = _random_problem(seed=2)
    f = oracle.fit(y, K, X, reml=True)
    grid = np.linspace(0, 1, 2001)
    best = max(oracle.profile_loglik(h, y, K, X, True)[0] for h in grid)
    assert f.ll >= best - 1e-9


def test_wald_refit_vs_null_h():
    y, K, X = _random_problem(seed=3)
    x = np.random.default_rng(5).standard_normal(len(y))
    w0 = oracle.wald(y, K, x, X0=X)
    w1 = oracle.wald(y, K, x, X0=X, refit=True)
    assert w0.h == oracle.fit(y, K, X).h
    assert w1.h == oracle.fit(y, K, np.c_[X, x]).h
    assert w0.df == len(y) - 3


# ---------------------------------------------------------------------------
# statsmodels MixedLM (optional; in the `sims` dependency group)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reml", [True, False])
def test_against_statsmodels_mixedlm_unbalanced(reml):
    sm = pytest.importorskip("statsmodels.api")
    sizes = [3, 5, 8, 4, 6, 7, 2, 9, 5, 6, 4, 8, 3, 7, 6]
    y = _grouped_y(sizes, sg=1.0, se=1.0, seed=7)
    groups = np.repeat(np.arange(len(sizes)), sizes)
    X = np.c_[np.ones(len(y)), np.random.default_rng(1).standard_normal(len(y))]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sm.MixedLM(y, X, groups=groups).fit(reml=reml, method=["bfgs"], gtol=1e-10)
    sg2, se2 = float(r.cov_re.iloc[0, 0]) if hasattr(r.cov_re, "iloc") \
        else float(np.asarray(r.cov_re)[0, 0]), float(r.scale)
    f = oracle.fit(y, _blocks(sizes), X, reml=reml)
    np.testing.assert_allclose(f.h, sg2 / (sg2 + se2), atol=1e-5)
    np.testing.assert_allclose(f.sigma2, sg2 + se2, rtol=1e-5)
    np.testing.assert_allclose(f.beta, np.asarray(r.fe_params), rtol=1e-5, atol=1e-6)
    # statsmodels' REML llf omits the constant +1/2 log|X'X| term
    adj = 0.5 * np.linalg.slogdet(X.T @ X)[1] if reml else 0.0
    np.testing.assert_allclose(f.ll - adj, r.llf, atol=1e-6)


# ---------------------------------------------------------------------------
# pylmm vs the oracle (quick check; the studies quantify this properly)
# ---------------------------------------------------------------------------

def test_pylmm_matches_oracle_on_easy_problem():
    from pylmm.lmm import LMM
    from sims import phenotypes
    n = 120
    g = genotypes.unrelated(n, 600, seed=4)
    kin = kinship.from_matrix(kinship.kinship(g.G))
    # ML on the raw (centred, singular) GRM is unbounded (see next test), so
    # the ML comparison uses a positive definite kinship
    Ks = {True: np.asarray(kin.K), False: 0.8 * np.asarray(kin.K) + 0.2 * np.eye(n)}
    # numpy's Generator streams may change between numpy versions, so pick
    # the first seed whose fits are interior (pylmm's grid stops at 0.99, bug 3)
    for seed in range(50):
        rng = np.random.default_rng(seed)
        y = phenotypes.simulate(kin, 0.5, rng).y
        fits = {r: oracle.fit(y, Ks[r], reml=r) for r in (True, False)}
        if all(0.02 < f.h < 0.95 for f in fits.values()):
            break
    x = rng.binomial(2, 0.3, n).astype(float)
    for reml in (True, False):
        K, f = Ks[reml], fits[reml]
        L = LMM(y, K, *linalg.eigh(K))
        h, beta, sigma, ll = L.fit(REML=reml)
        np.testing.assert_allclose(float(h), f.h, atol=1e-6)
        off = oracle.pylmm_reml_offset(n, 1) if reml else 0.0
        # at a common h the log-likelihoods agree up to the known constant
        ll_p = L.LL(f.h, REML=reml)[0]
        np.testing.assert_allclose(ll_p, f.ll + off, rtol=1e-9)
        t_p, _ = L.association(x.reshape(-1, 1), h=f.h, REML=reml)
        np.testing.assert_allclose(t_p, oracle.wald(y, K, x, h=f.h).t, rtol=1e-8)


def test_ml_unbounded_with_centred_grm_and_intercept():
    """calculateKinship's GRM has K 1 = 0, so with an intercept the ML profile
    diverges as h -> 1 (the MLE does not exist); REML stays bounded."""
    n = 120
    g = genotypes.unrelated(n, 600, seed=4)
    kin = kinship.from_matrix(kinship.kinship(g.G))
    assert np.abs(kin.K @ np.ones(n)).max() < 1e-10
    from sims import phenotypes
    y = phenotypes.simulate(kin, 0.6, np.random.default_rng(1)).y
    ml = oracle.fit(y, kin.K, reml=False)
    assert ml.ml_unbounded and ml.at_boundary
    assert oracle.profile_loglik(1 - 1e-10, y, kin.K, reml=False)[0] > ml.ll_grid[-2] + 5
    reml = oracle.fit(y, kin.K, reml=True)
    assert not reml.ml_unbounded
    # a positive definite K removes the degeneracy
    assert not oracle.fit(y, 0.8 * kin.K + 0.2 * np.eye(n), reml=False).ml_unbounded
