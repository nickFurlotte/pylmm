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


@pytest.mark.parametrize("h", [0.0, 0.3, 0.85])
def test_wald_matches_direct_gls_at_fixed_h(h):
    """The Wald test with K != I against a GLS computed a different way:
    whiten with the eigendecomposition of V = hK + (1-h)I (no Cholesky),
    then ordinary least squares (lstsq) on the whitened data, and the
    classical OLS variance / t / p of the whitened regression."""
    y, K, X = _random_problem(seed=4)
    n = len(y)
    x = np.random.default_rng(9).binomial(2, 0.3, n).astype(float)
    d, U = np.linalg.eigh(h * K + (1 - h) * np.eye(n))
    W = U / np.sqrt(d)                                   # V^-1/2 = U diag(d^-1/2) U'
    Xa = np.c_[X, x]
    yw, Xw = W.T @ y, W.T @ Xa
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    p = Xa.shape[1]
    resid = yw - Xw @ beta
    for test_reml, s2 in ((True, resid @ resid / (n - p)), (False, resid @ resid / n)):
        se = np.sqrt(s2 * np.linalg.inv(Xw.T @ Xw)[-1, -1])
        t = beta[-1] / se
        w = oracle.wald(y, K, x, X0=X, h=h, null_reml=test_reml)
        np.testing.assert_allclose(w.beta, beta[-1], rtol=1e-9)
        np.testing.assert_allclose(w.se, se, rtol=1e-9)
        np.testing.assert_allclose(w.t, t, rtol=1e-9)
        np.testing.assert_allclose(w.p, 2 * stats.t.sf(abs(t), n - p), rtol=1e-8)
        assert w.df == n - p
    # statsmodels GLS, when installed (the sims group), gives the same REML-convention t
    try:
        import statsmodels.api as sm
    except ImportError:         # pragma: no cover
        return
    r = sm.GLS(y, Xa, sigma=h * K + (1 - h) * np.eye(n)).fit()
    np.testing.assert_allclose(oracle.wald(y, K, x, X0=X, h=h).t, r.tvalues[-1], rtol=1e-8)


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
        np.testing.assert_allclose(t_p, oracle.wald(y, K, x, h=f.h, sigma_convention="pylmm").t,
                                   rtol=1e-8)


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




# ---------------------------------------------------------------------------
# pylmm-convention switches: with all of them on, the oracle reproduces pylmm
# ---------------------------------------------------------------------------

PYLMM = dict(eig_clamp=1e-6, h_grid="pylmm")


def _rank_deficient_problem(seed=0, n=150, m=60, h2=0.5):
    """K from fewer SNPs than individuals: rank <= m - 1, so pylmm clamps
    most eigenvalues to 1e-6 (as for the BED kinship, 1109/1219)."""
    from sims import phenotypes
    g = genotypes.unrelated(n, m, seed=seed)
    kin = kinship.from_matrix(kinship.kinship(g.G))
    rng = np.random.default_rng(seed)
    X0 = np.c_[np.ones(n), rng.standard_normal(n)]
    y = phenotypes.simulate(kin, h2, rng, X0=X0, alpha=[1.0, 0.3]).y
    x = rng.binomial(2, 0.3, n).astype(float)
    return y, kin, X0, x


def test_ll_conventions_ml_exact_reml_constant_offset():
    """pylmm ML LL == oracle ML LL; pylmm REML LL == oracle REML LL + the
    analytic constant, at every h (exactly, once the clamp is matched)."""
    from pylmm.lmm import LMM
    y, kin, X0, _ = _rank_deficient_problem()
    n, p = X0.shape
    assert (kin.Kva < 1e-6).sum() > 80
    L = LMM(y, np.asarray(kin.K), *kin.eig_for_pylmm(), X0=X0)
    Kc = oracle.clamp_kinship(kin.K)
    hs = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
    for reml in (True, False):
        d = [L.LL(h, REML=reml)[0] - oracle.profile_loglik(h, y, Kc, X0, reml)[0] for h in hs]
        np.testing.assert_allclose(d, oracle.pylmm_reml_offset(n, p) if reml else 0.0,
                                   atol=1e-8)
    # LL differences agree without any offset
    for reml in (True, False):
        dp = L.LL(0.7, REML=reml)[0] - L.LL(0.0, REML=reml)[0]
        do = (oracle.profile_loglik(0.7, y, Kc, X0, reml)[0]
              - oracle.profile_loglik(0.0, y, Kc, X0, reml)[0])
        np.testing.assert_allclose(dp, do, rtol=1e-10)
    # the clamp itself is a measurable effect, growing towards h = 1
    gap = [abs(L.LL(h, REML=True)[0] - oracle.pylmm_reml_offset(n, p)
               - oracle.profile_loglik(h, y, kin.K, X0)[0]) for h in (0.3, 0.99)]
    assert gap[1] > 10 * gap[0]


def _check_matches_pylmm(y, kin, X0, xs, tol_h=1e-6, tol_t=1e-6):
    """Oracle with all pylmm switches vs pylmm: null fit (REML and ML) and
    GWAS t (REML, and REML=False, which pylmm fits with a REML null)."""
    from pylmm.lmm import LMM
    from sims import methods
    K = np.asarray(kin.K)
    for reml in (True, False):
        L = LMM(y, K, *kin.eig_for_pylmm(), X0=X0)
        h, beta, sigma, ll = L.fit(REML=reml)
        f = oracle.fit(y, K, X0, reml=reml, **PYLMM)
        np.testing.assert_allclose(float(np.sum(h)), f.h, atol=tol_h)
        np.testing.assert_allclose(float(np.sum(ll)), f.ll_pylmm, atol=1e-6)
        np.testing.assert_allclose(float(np.sum(sigma)), f.sigma2_reml, rtol=1e-5)
        np.testing.assert_allclose(np.ravel(beta), f.beta, rtol=1e-5, atol=1e-7)
    for reml in (True, False):
        t_p, p_p = methods.pylmm_gwas(y, xs, kin, X0=X0, REML=reml)
        w = [oracle.wald(y, K, xs[:, j], X0=X0, null_reml=True, test_reml=reml,
                         sigma_convention="pylmm", **PYLMM) for j in range(xs.shape[1])]
        np.testing.assert_allclose(t_p, [v.t for v in w], atol=tol_t, rtol=tol_t)
        np.testing.assert_allclose(p_p, [v.p for v in w], rtol=1e-4)


def test_pylmm_convention_matches_pylmm_synthetic():
    for seed in range(3):
        y, kin, X0, x = _rank_deficient_problem(seed)
        xs = np.c_[x, np.random.default_rng(seed).binomial(2, 0.4, len(y))]
        _check_matches_pylmm(y, kin, X0, xs.astype(float))


def test_pylmm_convention_matches_pylmm_bed_subset():
    """Real genotypes: 300 random BED mice, K from 2500 SNPs (rank ~99)."""
    from sims import phenotypes
    spec = {"source": "bed", "n_snps": 2510, "select": "random", "n_indivs": 300, "seed": 1}
    g = genotypes.get(spec)
    kin = kinship.get(spec, snps=slice(0, 2500))
    T = g.G[:, 2500:]
    T = np.where(np.isnan(T), np.nanmean(T, axis=0), T)
    T = T[:, T.std(axis=0) > 0][:, :5]
    for h2, seed in ((0.3, 0), (0.8, 1)):
        y = phenotypes.simulate(kin, h2, np.random.default_rng(seed)).y
        _check_matches_pylmm(y, kin, None, T)


def test_pylmm_convention_reproduces_example_null_failures():
    """The example_null replicates where pylmm stopped at h = 0.99 while the
    textbook oracle found h > 0.99 (n=200, h2=0.5, reps 1 and 15).  With
    h_grid='pylmm' the oracle reproduces pylmm's 0.99 (the optimizer, not the
    eigenvalue clamp, is responsible: the clamped fine search still goes
    above 0.99).

    Those two replicates have optima within FP noise of 0.99, so whether the
    fine fit lands above 0.99 is platform dependent; for them only the
    implication is checked.  The constructed case (h2 = 1, optimum at the
    h = 1 boundary) is unambiguous on every platform."""
    from sims import phenotypes, runner
    from sims.studies import example_null as ex
    params = {"n": 200, "h2": 0.5, "m_kin": 2000, "m_test": 2000, "geno_seed": 1}
    spec = {"source": "unrelated", "n": 200, "m": 4000, "seed": 1}
    kin = kinship.get(spec, snps=slice(0, 2000))
    G = genotypes.get(spec).G[:, 2000:2003]
    for rep in (1, 15):
        rng = np.random.default_rng(runner.task_seed(ex.NAME, params, rep))
        y = phenotypes.simulate(kin, 0.5, rng).y
        _check_matches_pylmm(y, kin, None, G)
        fine = oracle.fit(y, kin.K, eig_clamp=1e-6)
        if fine.h > 0.99:
            assert oracle.fit(y, kin.K, **PYLMM).h == pytest.approx(0.99)
    # optimum clearly above 0.99: simulate at h2 = 1 and pick the first seed
    # (of a few) whose clamped fine optimum is >= 0.995
    for seed in range(20):
        y = phenotypes.simulate(kin, 1.0, np.random.default_rng(seed)).y
        fine = oracle.fit(y, kin.K, eig_clamp=1e-6)
        if fine.h >= 0.995:
            break
    else:
        pytest.fail("no seed gave a clamped fine optimum >= 0.995")
    _check_matches_pylmm(y, kin, None, G)
    assert oracle.fit(y, kin.K, **PYLMM).h == pytest.approx(0.99)


def test_sigma_conventions():
    y, kin, X0, x = _rank_deficient_problem(1)
    n, p = X0.shape
    f = oracle.fit(y, kin.K, X0, reml=False)
    assert f.sigma2 == f.sigma2_ml and f.sigma2_kind.startswith("ml")
    np.testing.assert_allclose(f.sigma2_reml / f.sigma2_ml, n / (n - p))
    fr = oracle.fit(y, kin.K, X0, reml=True)
    assert fr.sigma2 == fr.sigma2_reml
    wt = oracle.wald(y, kin.K, x, X0=X0, null_reml=False, h=f.h)
    wp = oracle.wald(y, kin.K, x, X0=X0, null_reml=False, h=f.h, sigma_convention="pylmm")
    # textbook ML uses Q/n, pylmm Q/(n-q): t differs by sqrt(n/(n-q))
    np.testing.assert_allclose(wt.t / wp.t, np.sqrt(n / (n - p - 1)), rtol=1e-10)
    assert wt.df == wp.df == n - p - 1
    with pytest.raises(ValueError):
        oracle.wald(y, kin.K, x, sigma_convention="other")
    with pytest.raises(ValueError):
        oracle.fit(y, kin.K, h_grid="coarse")
