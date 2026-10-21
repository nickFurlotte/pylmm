"""The local h refinement (bug 4, fixed in Phase 5.1): a bounded search inside
the bracketing grid interval, never outside [0, 1] -- in the default search
(LMM._get_best, Phase 6.3) and in 1.0.0's (LMM._get_max, legacy_fit=True)."""

import contextlib
import io

import numpy as np
import pytest
from scipy import linalg

from pylmm.lmm import LMM, calculate_kinship


class SpyLMM(LMM):
    """Records every h the local search evaluates."""

    def _neg_log_likelihood(self, h, X=None, reml=True):
        self.evals.append(float(h))
        return LMM._neg_log_likelihood(self, h, X, reml)


def make(y, K, legacy_fit=False):
    L = SpyLMM.__new__(SpyLMM)
    L.evals = []
    with contextlib.redirect_stderr(io.StringIO()):
        SpyLMM.__init__(L, y, K, legacy_fit=legacy_fit)
    return L


def problem(seed, h2, n=60, m=30):
    rng = np.random.default_rng(seed)
    G = rng.binomial(2, 0.3, size=(n, m)).astype(float)
    K = calculate_kinship(G.copy())
    d, V = linalg.eigh(K)
    u = V @ (np.sqrt(np.clip(d, 0, None)) * rng.standard_normal(n))
    y = np.sqrt(h2) * u + np.sqrt(1 - h2) * rng.standard_normal(n)
    x = rng.binomial(2, 0.3, size=(n, 1)).astype(float)
    return y, K, x


def check_evals_inside_bracket(L):
    H, ll = L.h_grid, L.ll_grid
    assert L.evals, "the search never ran"
    assert min(L.evals) >= 0.0 and max(L.evals) <= 1.0
    if L.legacy_fit:
        brackets = [
            (H[i - 1], H[i + 1])
            for i in range(1, len(H) - 2)
            if ll[i - 1] < ll[i] and ll[i] > ll[i + 1]
        ]
    else:
        brackets = [
            (H[i - 1], H[i + 1])
            for i in range(1, len(H) - 1)
            if ll[i] >= ll[i - 1] and ll[i] >= ll[i + 1]
        ]
        if ll[0] > ll[1]:
            brackets.append((H[0], H[1]))
        if ll[-1] > ll[-2]:
            brackets.append((H[-2], H[-1]))
    for h in L.evals:
        assert any(lo <= h <= hi for lo, hi in brackets), h


@pytest.mark.parametrize("legacy_fit", [False, True], ids=["default", "legacy"])
@pytest.mark.parametrize("h2", [0.05, 0.5, 0.9, 0.97])
@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_bounded_search_never_leaves_its_bracket(h2, reml, legacy_fit):
    """Null fit and per-SNP refit, over seeds whose grid has a peak."""
    ran = 0
    for seed in range(12):
        y, K, x = problem(seed, h2)
        for X in (None, x):
            L = make(y, K, legacy_fit=legacy_fit)
            with np.errstate(
                divide="raise", invalid="raise", over="raise"
            ):  # no nan / log of a negative
                h = float(np.sum(L.fit(X=X, reml=reml)[0]))
            assert 0.0 <= h <= (0.99 if legacy_fit else 1.0)
            if L.evals:
                check_evals_inside_bracket(L)
                ran += 1
    assert ran > 0


def test_s5_seed1177_small_n_null_fit_does_not_crash():
    """S5 P1-7 reproducer (n = 10): the grid maximum is at 0.96 with a nearly
    flat profile to 0.99; on scipy >= 1.11 this raised BracketError."""
    rng = np.random.default_rng(1177)
    W = rng.binomial(2, rng.uniform(0.05, 0.5, 100), (10, 100)).astype(float)
    K = calculate_kinship(W)
    y = rng.standard_normal(10)
    for legacy_fit in (True, False):
        L = make(y, K, legacy_fit=legacy_fit)
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            hmax, beta, sigma, ll = L.fit()
        h = float(np.sum(hmax))
        assert 0.95 <= h <= (0.99 if legacy_fit else 1.0) and np.isfinite(np.sum(ll))
        check_evals_inside_bracket(L)


def test_ll_brent_guards_outside_unit_interval():
    y, K, _ = problem(0, 0.5)
    L = make(y, K)
    assert L._neg_log_likelihood(-1e-9) == 1e6
    assert L._neg_log_likelihood(1 + 1e-9) == 1e6
    assert np.isfinite(L._neg_log_likelihood(0.5))
