"""The Phase 5b API cleanups of pylmm.lmm: scalar returns, 1-D SNPs, LogLikResult,
REML defaults and the kva/kve arguments.  None of them changes a number."""

import contextlib
import inspect
import io

import numpy as np
import pytest
from scipy import linalg

from pylmm.lmm import LMM, LogLikResult, calculate_kinship, gwas


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    W = rng.binomial(2, 0.3, size=(50, 120)).astype(float)
    K = calculate_kinship(W)
    y = 0.7 * W[:, 0] + rng.normal(size=50)
    return W, K, y


def test_every_reml_parameter_defaults_to_true():
    for name in ("fit", "association", "log_likelihood", "_neg_log_likelihood", "_get_max"):
        assert inspect.signature(getattr(LMM, name)).parameters["reml"].default is True, name
    assert inspect.signature(gwas).parameters["reml"].default is True


def test_log_likelihood_result(data):
    _, K, y = data
    L = LMM(y, K)
    h, beta, sigma, ll = L.fit()
    res = L.log_likelihood(h)  # REML by default, like fit
    assert isinstance(res, LogLikResult) and isinstance(res, tuple)
    assert res.ll == ll and res.sigma == sigma
    np.testing.assert_array_equal(res.beta, beta)
    ll2, beta2, sigma2, xx_inv = res
    assert ll2 == ll and xx_inv.shape == (1, 1)
    assert L.log_likelihood(h, reml=False).ll != ll


def test_scalar_types(data):
    W, K, y = data
    L = LMM(y, K)
    h, beta, sigma, ll = L.fit()
    assert type(sigma) is float and type(ll) is float and type(L.opt_sigma) is float
    _, s, Q, _, _ = L.ml_solution(h, L.x0t)
    assert type(s) is float and type(Q) is float
    for reml in (True, False):
        t, p, b, v = L.association(W[:, 1], reml=reml, return_beta=True)
        assert type(v) is float
        assert t == pytest.approx(b / np.sqrt(v), rel=1e-12)


def test_one_dimensional_snp_equals_a_column(data):
    W, K, y = data
    L = LMM(y, K)
    L.fit()
    # _snp_column makes the SNP a C-contiguous float64 column, so a strided view
    # W[:, j] gives exactly (bit-identically) the same numbers as W[:, [j]].
    col = [np.ascontiguousarray(W[:, j]) for j in range(4)]
    for j in range(4):
        assert L.association(col[j], return_beta=True) == L.association(W[:, [j]], return_beta=True)
        assert L.association(W[:, j], return_beta=True) == L.association(
            W[:, [j]], return_beta=True
        )
    a = L.fit(X=col[2])
    b = L.fit(X=W[:, [2]])
    assert a[0] == b[0] and a[2] == b[2] and a[3] == b[3]
    r1, r2 = L.log_likelihood(0.3, col[2]), L.log_likelihood(0.3, W[:, [2]])
    assert r1.ll == r2.ll and r1.sigma == r2.sigma
    np.testing.assert_array_equal(r1.beta, r2.beta)
    with pytest.raises(ValueError, match=r"the SNP must have shape \(50,\) or \(50, 1\)"):
        L.association(W[:, :2])
    with pytest.raises(ValueError, match="the SNP must have shape"):
        L.association(W[:10, 0])


def test_kva_kve_none_only(data):
    _, K, y = data
    kva, kve = linalg.eigh(K)
    with pytest.raises(ValueError, match="kva is empty; pass None"):
        LMM(y, K, [], [])
    with pytest.raises(ValueError, match="kve is empty; pass None"):
        gwas(y, np.ones((50, 1)), K, kva=kva, kve=np.zeros((0, 0)))
    with pytest.raises(ValueError, match=r"pass both kva and kve .* got only kva"):
        LMM(y, K, kva=kva)
    a = LMM(y, K, kva, kve).fit()
    b = LMM(y, K).fit()
    assert a[0] == pytest.approx(b[0], abs=1e-9)


def test_gwas_numbers_unchanged_by_the_cleanups(data):
    """gwas() and a manual loop over association() agree exactly."""
    W, K, y = data
    with contextlib.redirect_stderr(io.StringIO()):
        t, p = gwas(y, W[:, :6], K)
    L = LMM(y, K)
    L.fit()
    for j in range(6):
        assert (t[j], p[j]) == L.association(W[:, j])
