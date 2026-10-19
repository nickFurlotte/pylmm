"""The local h refinement in LMM.getMax (bug 4, fixed in Phase 5.1): a
bounded search inside the bracketing grid interval, never outside [0, 1]."""

import contextlib
import io

import numpy as np
import pytest
from scipy import linalg

from pylmm.lmm import LMM, calculateKinship


class SpyLMM(LMM):
    """Records every h the local search evaluates."""

    def LL_brent(self, h, X=None, REML=False):
        self.evals.append(float(h))
        return LMM.LL_brent(self, h, X, REML)


def make(y, K):
    L = SpyLMM.__new__(SpyLMM)
    L.evals = []
    with contextlib.redirect_stderr(io.StringIO()):
        SpyLMM.__init__(L, y, K)
    return L


def problem(seed, h2, n=60, m=30):
    rng = np.random.default_rng(seed)
    G = rng.binomial(2, 0.3, size=(n, m)).astype(float)
    K = calculateKinship(G.copy())
    d, V = linalg.eigh(K)
    u = V @ (np.sqrt(np.clip(d, 0, None)) * rng.standard_normal(n))
    y = np.sqrt(h2) * u + np.sqrt(1 - h2) * rng.standard_normal(n)
    x = rng.binomial(2, 0.3, size=(n, 1)).astype(float)
    return y, K, x


def check_evals_inside_bracket(L):
    H = L.H
    assert L.evals, "the search never ran"
    assert min(L.evals) >= 0.0 and max(L.evals) <= 1.0
    peaks = [i for i in range(1, len(H) - 2) if L.LLs[i - 1] < L.LLs[i] and L.LLs[i] > L.LLs[i + 1]]
    for h in L.evals:
        assert any(H[i - 1] <= h <= H[i + 1] for i in peaks), h


@pytest.mark.parametrize("h2", [0.05, 0.5, 0.9, 0.97])
@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_bounded_search_never_leaves_its_bracket(h2, reml):
    """Null fit and per-SNP refit, over seeds whose grid has an interior peak."""
    ran = 0
    for seed in range(12):
        y, K, x = problem(seed, h2)
        for X in (None, x):
            L = make(y, K)
            with np.errstate(
                divide="raise", invalid="raise", over="raise"
            ):  # no nan / log of a negative
                h = float(np.sum(L.fit(X=X, REML=reml)[0]))
            assert 0.0 <= h <= 0.99
            if L.evals:
                check_evals_inside_bracket(L)
                ran += 1
    assert ran > 0


def test_s5_seed1177_small_n_null_fit_does_not_crash():
    """S5 P1-7 reproducer (n = 10): the grid maximum is at 0.96 with a nearly
    flat profile to 0.99; on scipy >= 1.11 this raised BracketError."""
    rng = np.random.default_rng(1177)
    W = rng.binomial(2, rng.uniform(0.05, 0.5, 100), (10, 100)).astype(float)
    K = calculateKinship(W)
    y = rng.standard_normal(10)
    L = make(y, K)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        hmax, beta, sigma, ll = L.fit()
    h = float(np.sum(hmax))
    assert 0.95 <= h <= 0.99 and np.isfinite(np.sum(ll))
    check_evals_inside_bracket(L)


def test_ll_brent_guards_outside_unit_interval():
    y, K, _ = problem(0, 0.5)
    L = make(y, K)
    assert L.LL_brent(-1e-9) == 1e6
    assert L.LL_brent(1 + 1e-9) == 1e6
    assert np.isfinite(L.LL_brent(0.5))
