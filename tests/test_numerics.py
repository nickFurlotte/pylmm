"""Phase 6.3 numerics: the h search on [0, 1], h = 1 with a singular K, log p-values."""

import contextlib
import io
import warnings

import helpers as h
import numpy as np
import pytest
from scipy import linalg, stats

from pylmm import misc
from pylmm.lmm import LMM, calculate_kinship, gwas, log_pvalue, neg_log10_pvalue

# ---------------------------------------------------------------------------
# B1 (review of #27): h must not depend on how a covariate is parametrised
# ---------------------------------------------------------------------------


def _seed62():
    """The review's reproducer: before the fix h was 1.0000 with the raw date
    and 0.9607 with it centred (the normal equations at h = 1, weight 1e6 on
    the null direction 1 of a centred K, next to a covariate ~2e7)."""
    rng = np.random.default_rng(62)
    W = rng.binomial(2, 0.3, size=(100, 500)).astype(float)
    K = calculate_kinship(W)
    y = rng.multivariate_normal(np.zeros(100), 0.95 * K + 0.05 * np.eye(100))
    date = 2e7 + 1e3 * rng.normal(size=100)
    return y, K, date, W


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_h_invariant_to_affine_rescaling_of_a_covariate(reml):
    """Null fit, per-SNP refit and fit_two give the same h for a raw, centred
    or scaled covariate (the likelihood is invariant to X0 -> X0 A)."""
    y, K, date, W = _seed62()
    n = len(y)
    one = np.ones(n)
    c = 1e4 + np.random.default_rng(1).normal(size=n)
    designs = {
        "date raw": np.column_stack([one, date]),
        "date centred": np.column_stack([one, date - 2e7]),
        "date scaled": np.column_stack([one, (date - date.mean()) / date.std()]),
        "1e4 +- 1": np.column_stack([one, c]),
        "1e4 centred": np.column_stack([one, c - 1e4]),
    }
    x = W[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the bug-17 and boundary warnings
        null = {k: float(LMM(y, K, X0=X0).fit(reml=reml)[0]) for k, X0 in designs.items()}
        refit = {k: float(LMM(y, K, X0=X0).fit(X=x, reml=reml)[0]) for k, X0 in designs.items()}
    for group in (("date raw", "date centred", "date scaled"), ("1e4 +- 1", "1e4 centred")):
        for fits in (null, refit):
            vals = [fits[k] for k in group]
            assert max(vals) - min(vals) <= h.H_REPARAM_ATOL, (group, vals)
    if reml:
        assert abs(null["date raw"] - 0.9607) < 1e-3  # the reviewer's centred value
        K2 = calculate_kinship(W[:, 250:])
        with contextlib.redirect_stderr(io.StringIO()):
            raw = misc.fit_two(y, K, K2, X0=designs["date raw"], ngrids=4)
            cen = misc.fit_two(y, K, K2, X0=designs["date centred"], ngrids=4)
        np.testing.assert_allclose(raw[:3], cen[:3], rtol=0, atol=h.H_REPARAM_ATOL)


def _grm(n=80, m=300, seed=0):
    rng = np.random.default_rng(seed)
    K = calculate_kinship(rng.binomial(2, 0.3, size=(n, m)).astype(float))
    return K, rng


# ---------------------------------------------------------------------------
# log p-values
# ---------------------------------------------------------------------------


def test_log_pvalue_equals_log_of_p_where_p_is_representable():
    t = np.array([0.0, 0.5, 1.96, 3.0, 8.0, 20.0, 35.0])
    for df in (1, 3, 10, 58, 300, 5000):
        exp = np.log(2 * stats.t.sf(t, df))
        np.testing.assert_allclose(log_pvalue(t, df), exp, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(log_pvalue(-t, df), exp, rtol=1e-12, atol=1e-15)
    assert log_pvalue(0.0, 10) == 0.0


def test_log_pvalue_tail_series_agrees_with_scipy_on_the_overlap():
    """The incomplete-beta continued fraction used past p ~ 1e-280 matches
    log(2 sf) where both are accurate (p from 1e-30 to 1e-250)."""
    from pylmm.lmm import _log_beta_tail

    for df in (1.0, 2.0, 5.0, 30.0, 100.0, 1000.0):
        for logp in (-70.0, -200.0, -575.0):
            t = stats.t.isf(np.exp(logp) / 2, df)
            if not np.isfinite(t) or stats.t.sf(t, df) == 0:
                continue
            got = _log_beta_tail(np.array([t]), np.array([df]))[0]
            assert got == pytest.approx(np.log(2 * stats.t.sf(t, df)), rel=1e-10), (df, logp)


def test_log_pvalue_beyond_underflow():
    t = np.array([1e8, 1e13, 1e100, 1e300])
    assert (2 * stats.t.sf(t, 50) == 0).all()
    lp = log_pvalue(t, 50)
    assert np.isfinite(lp).all() and (np.diff(lp) < 0).all()
    # asymptotically p ~ C t^-df: log p decreases by df * log(10) per decade of t
    assert lp[1] - lp[0] == pytest.approx(-50 * 5 * np.log(10), rel=1e-6)
    assert neg_log10_pvalue(1e13, 50) == pytest.approx(-lp[1] / np.log(10))
    # inf -> -inf, nan -> nan; vector df
    np.testing.assert_array_equal(log_pvalue([np.inf, np.nan], 10), [-np.inf, np.nan])
    assert log_pvalue([3.0, 3.0], [5, 50]).shape == (2,)
    assert type(log_pvalue(3.0, 5)) is np.float64


def test_tstat_log_is_log_p():
    K, rng = _grm()
    y = rng.standard_normal(80)
    L = LMM(y, K)
    L.fit()
    beta, var, sigma = np.array([[0.5]]), np.array([[0.01]]), np.array([[1.0]])
    t, p = L.tstat(beta, var, sigma, 2)
    t2, logp = L.tstat(beta, var, sigma, 2, log=True)
    assert t == t2 and logp == pytest.approx(np.log(p), rel=1e-12)


# ---------------------------------------------------------------------------
# the h search
# ---------------------------------------------------------------------------


def _simulate(K, h2, rng):
    d, V = linalg.eigh(K)
    u = V @ (np.sqrt(np.clip(d, 0, None)) * rng.standard_normal(len(K)))
    return np.sqrt(h2) * u + np.sqrt(1 - h2) * rng.standard_normal(len(K))


def test_h_equal_one_with_a_singular_kinship_is_safe():
    """h2 = 1 with a centred GRM (singular: K 1 = 0): the default search reaches
    h = 1 (or within the refinement tolerance), with no floating-point warning,
    and the Wald test at h = 1 is finite."""
    K, rng = _grm(n=100, m=400, seed=3)
    assert np.linalg.eigvalsh(K)[0] < 1e-10
    y = _simulate(K, 1.0, rng)
    L = LMM(y, K)
    with np.errstate(all="raise"):
        h, beta, sigma, ll = L.fit()
        t, p = L.association(rng.binomial(2, 0.3, 100).astype(float))
        tb = L.association_batch(rng.binomial(2, 0.3, (100, 5)).astype(float))
    assert float(h) > 0.999 and np.isfinite([sigma, ll, t, p]).all()
    assert np.isfinite(tb.t).all()
    with np.errstate(all="raise"):
        assert np.isfinite(L.log_likelihood(1.0).ll)


def test_boundary_warning_with_rank_deficient_kinship():
    """h at 1 with K of rank 30 < n - 1: a warning (the Wald df overstate the
    information), once per call site; no numbers change.  Not with K's null
    space inside the covariates (the centred GRM above), not for an interior h,
    not with legacy_fit."""
    K, rng = _grm(n=80, m=30, seed=4)
    assert (np.linalg.eigvalsh(K) < 1e-6).sum() > 1
    y = _simulate(K, 1.0, rng)
    with pytest.warns(UserWarning, match="upper boundary"):
        h1 = float(LMM(y, K).fit()[0])
    assert h1 >= 0.999
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        old = LMM(y, K, legacy_fit=True)
        old.fit()  # 1.0.0: no warning
        LMM(_simulate(K, 0.3, rng), K).fit()  # interior h
        K2, rng2 = _grm(n=100, m=400, seed=3)
        LMM(_simulate(K2, 1.0, rng2), K2).fit()  # null space = the intercept


def _date_at_h1(seed):
    """Review B2 data: a centred K, true h2 = 0.98 and an intercept + date
    2e7 +- 1e3; for seeds 5, 17 and 42 the default fit is at h = 1."""
    rng = np.random.default_rng(seed)
    n = 100
    W = rng.binomial(2, 0.3, size=(n, 500)).astype(float)
    K = calculate_kinship(W)
    y = rng.multivariate_normal(np.zeros(n), 0.98 * K + 0.02 * np.eye(n))
    d = 1e3 * rng.normal(size=n)
    return (
        y,
        K,
        W[:, 1:301],
        np.column_stack([np.ones(n), 2e7 + d]),
        np.column_stack([np.ones(n), d]),
    )


@pytest.mark.parametrize("seed", [5, 17, 42])
def test_batched_scan_at_h1_with_a_raw_date_covariate(seed):
    """Review B2: at h = 1 the batched path residualised on the raw covariates
    by the normal equations (condition ~1e20); its t was off by up to 0.51 of
    max |t|.  Now batched == per-SNP within BATCH_RTOL, and both are invariant
    to centring the covariate."""
    y, K, G, x0_raw, x0_cen = _date_at_h1(seed)
    assert float(LMM(y, K, X0=x0_raw).fit()[0]) == 1.0
    t_raw = gwas(y, G, K, X0=x0_raw)[0]
    t_raw_per_snp = gwas(y, G, K, X0=x0_raw, batch=False)[0]
    t_centred = gwas(y, G, K, X0=x0_cen)[0]
    h.assert_batch_close(t_raw, t_raw_per_snp, what="batched vs per SNP, raw date")
    h.assert_batch_close(t_raw, t_centred, what="raw vs centred date, batched")
    L = LMM(y, K, X0=x0_raw)
    L.fit()
    res = L.association_batch(G[:, :20])
    ref = np.array([L.association(G[:, j], return_beta=True) for j in range(20)])
    h.assert_batch_close(res.beta, ref[:, 2], what="beta")  # the caller's parametrisation
    h.assert_batch_close(res.beta_var, ref[:, 3], what="beta_var")


def test_log_likelihood_with_a_snp_is_invariant_to_covariate_centring():
    """Review S5: log_likelihood(h, X=snp) (a stacked SNP) also works on an
    orthonormal basis; beta is reported for the design as given."""
    y, K, G, x0_raw, x0_cen = _date_at_h1(5)
    raw, cen = LMM(y, K, X0=x0_raw), LMM(y, K, X0=x0_cen)
    for hh in (0.5, 0.99, 1.0):
        a, b = raw.log_likelihood(hh, G[:, 0]), cen.log_likelihood(hh, G[:, 0])
        assert a.ll == pytest.approx(b.ll, abs=1e-8)
        assert a.beta[-1, 0] == pytest.approx(b.beta[-1, 0], rel=1e-8)  # the SNP's effect
        assert a.beta[0, 0] == pytest.approx(b.beta[0, 0] - 2e7 * b.beta[1, 0], rel=1e-6)


def test_fit_with_a_collinear_snp_raises():
    """fit(X=x) with x collinear with X0 raises, as its docstring says."""
    K, rng = _grm()
    L = LMM(rng.standard_normal(80), K)
    with pytest.raises(np.linalg.LinAlgError, match="rank-deficient"):
        L.fit(X=np.ones(80) * 3.0)


def test_boundary_warning_is_scale_free(bed_null):
    """K x 1e-3 at true h2 = 0.5 gives h near 0.999 but a residual variance far
    from 0: no warning.  The README null fit (h = 0.004) is silent too."""
    K, rng = _grm(n=80, m=30, seed=4)
    y = _simulate(K, 0.5, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h_small = float(LMM(y, K * 1e-3).fit()[0])
        Y, K2, kva, kve = bed_null
        LMM(Y, K2, kva.copy(), kve).fit()
    assert h_small > 0.99


def test_boundary_warning_in_the_cli_log(tmp_path):
    """pylmm gwas logs the boundary warning once: 15 inbred strains x 4 animals,
    a phenotype that is the strain value (h2 = 1), 3 SNPs (EMMA input)."""
    rng = np.random.default_rng(5)
    strains = rng.binomial(1, 0.5, size=(15, 200)).astype(float)
    G = np.repeat(strains, 4, axis=0)  # 60 animals
    K = calculate_kinship(G[:, 3:])
    y = np.repeat(rng.normal(size=15), 4)
    np.savetxt(tmp_path / "geno", G[:, :3].T, fmt="%d")
    np.savetxt(tmp_path / "pheno", y[None, :])
    np.savetxt(tmp_path / "K", K)
    log = h.run_cli(
        ["gwas", "--emma-geno", "geno", "--emma-pheno", "pheno", "--kinship", "K", "-o", "out"],
        cwd=tmp_path,
    )
    assert log.count("upper boundary") == 1, log
    assert "pylmm: warning:" in log


def test_default_grid_and_ngrids():
    K, rng = _grm()
    y = rng.standard_normal(80)
    L = LMM(y, K)
    L.fit(ngrids=10)
    np.testing.assert_array_equal(L.h_grid, np.arange(11) / 10.0)
    L.fit()
    assert len(L.h_grid) == 101 and L.h_grid[-1] == 1.0
    old = LMM(y, K, legacy_fit=True)
    old.fit()
    assert len(old.h_grid) == 100 and old.h_grid[-1] == 0.99
    np.testing.assert_array_equal(old.h_grid, L.h_grid[:-1])  # the same points
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="ngrids"):
            L.fit(ngrids=bad)


def test_default_fit_never_below_legacy_likelihood():
    """On every replicate the default search's likelihood is >= 1.0.0's (it
    keeps the best of a superset of candidates)."""
    K, rng = _grm(n=60, m=40, seed=1)  # rank-deficient K: profiles with features
    for _ in range(30):
        y = _simulate(K, rng.uniform(0, 1), rng)
        for reml in (True, False):
            new = LMM(y, K).fit(reml=reml)[3]
            old = LMM(y, K, legacy_fit=True).fit(reml=reml)[3]
            assert new >= old - 1e-9


def test_gwas_legacy_fit_reaches_the_per_snp_subset_models():
    """legacy_fit reaches the models gwas builds for SNPs with missing calls."""
    K, rng = _grm(n=60, m=40, seed=1)
    y = _simulate(K, 0.97, rng)
    X = rng.binomial(2, 0.3, (60, 3)).astype(float)
    X[:5, 1] = np.nan
    t_new = gwas(y, X, K)[0]
    t_old = gwas(y, X, K, legacy_fit=True)[0]
    t_old_ref = gwas(y, X, K, legacy_fit=True, batch=False)[0]
    np.testing.assert_allclose(t_old, t_old_ref, rtol=1e-9)
    assert np.isfinite(t_new).all()
