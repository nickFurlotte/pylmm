"""Null-model fits: LMM.fit() under REML/ML, missing Y, covariates.

The goldens are pylmm 1.0.0's (the 2013 code's) fits, so they are checked with
``legacy_fit=True``; the default search (Phase 6.3) is checked against them
where 1.0.0 found the maximum, and against the oracle where it did not.
"""

import helpers as h
import numpy as np
import pytest

from pylmm import lmm

FITS = h.load_kv("null_fits.kv.txt")


def check_fit(L, prefix):
    """Compare an LMM after fit() against null_fits.kv.txt[prefix.*]."""
    beta = np.asarray(L.opt_beta).ravel()
    mn, vx = L.mean_and_var()
    got = {
        "h": L.opt_h,
        "sigma": L.opt_sigma,
        "LL": L.opt_ll,
        "h_posterior_mean": mn,
        "h_posterior_var": vx,
    }
    for i, b in enumerate(beta):
        got["beta_%d" % i] = b
    expected_keys = sorted(k[len(prefix) + 1 :] for k in FITS if k.startswith(prefix + "."))
    assert sorted([*list(got), "N"]) == expected_keys
    assert FITS[prefix + ".N"] == L.N
    for k in sorted(got):
        h.assert_close(
            got[k], FITS[prefix + "." + k], rtol=h.RTOL_FIT, atol=h.ATOL_FIT, what=prefix + "." + k
        )
    assert 0.0 <= L.opt_h <= 1.0, "%s: h=%r outside [0, 1]" % (prefix, L.opt_h)


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_hmdp_null_fit(hmdp, reml):
    Y, K, _ = hmdp
    L = lmm.LMM(Y, K, legacy_fit=True)
    L.fit(reml=reml)
    check_fit(L, "hmdp_%s" % ("reml" if reml else "ml"))


def test_hmdp_ll_grid(hmdp):
    """LL over the fixed h grid: no optimiser involved, so tight tolerance."""
    Y, K, _ = hmdp
    grid = h.load_golden_txt("null_fit_hmdp_LLgrid.txt")
    for col, reml in ((1, True), (2, False)):
        L = lmm.LMM(Y, K, legacy_fit=True)
        L.fit(reml=reml)
        h.assert_close(L.h_grid, grid[:, 0], what="H grid")
        h.assert_close(L.ll_grid, grid[:, col], what="LL grid REML=%s" % reml)


def test_hmdp_missing_y(hmdp):
    Y, K, _ = hmdp
    y_missing = Y.copy()
    y_missing[h.HMDP_MISSING_Y] = np.nan
    L = lmm.LMM(y_missing, K, legacy_fit=True)
    L.fit(reml=True)
    check_fit(L, "hmdp_missingY_reml")


def test_hmdp_covariate(hmdp):
    Y, K, snps = hmdp
    c = snps[:, h.HMDP_COV_SNP].copy()
    c[np.isnan(c)] = c[~np.isnan(c)].mean()
    L = lmm.LMM(Y, K, X0=np.column_stack([np.ones(len(Y)), c]), legacy_fit=True)
    L.fit(reml=True)
    check_fit(L, "hmdp_cov_reml")


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_bed_null_fit(bed_null, reml):
    """README data; fits at the h=0 boundary (_get_max endpoint branch)."""
    Y, K, kva, kve = bed_null
    L = lmm.LMM(Y, K, kva.copy(), kve, legacy_fit=True)
    L.fit(reml=reml)
    check_fit(L, "bed_%s" % ("reml" if reml else "ml"))


# ---------------------------------------------------------------------------
# the default search (Phase 6.3) on the same data
# ---------------------------------------------------------------------------


def _check_default_fit_equals_golden(L, prefix, reml):
    """h, LL and beta equal the golden; sigma too under REML (ML: Q/n, bug 9)."""
    for k, got in (("h", L.opt_h), ("LL", L.opt_ll)):
        h.assert_close(got, FITS[prefix + "." + k], rtol=h.RTOL_FIT, atol=h.ATOL_FIT, what=k)
    for i, b in enumerate(np.asarray(L.opt_beta).ravel()):
        exp = FITS["%s.beta_%d" % (prefix, i)]
        h.assert_close(b, exp, rtol=h.RTOL_FIT, atol=h.ATOL_FIT, what="beta")
    n, p = L.N, L.q
    scale = 1.0 if reml else (n - p) / n
    h.assert_close(
        L.opt_sigma, FITS[prefix + ".sigma"] * scale, rtol=h.RTOL_FIT, atol=h.ATOL_FIT, what="s"
    )


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_hmdp_default_fit_equals_golden(hmdp, reml):
    """An interior, single-peaked profile: the default search finds 1.0.0's optimum."""
    Y, K, _ = hmdp
    L = lmm.LMM(Y, K)
    L.fit(reml=reml)
    _check_default_fit_equals_golden(L, "hmdp_%s" % ("reml" if reml else "ml"), reml)
    assert len(L.h_grid) == 101 and L.h_grid[-1] == 1.0


def test_hmdp_default_ll_grid_extends_golden(hmdp):
    """The default grid is 1.0.0's plus h = 1; the shared points agree (slogdet vs det)."""
    Y, K, _ = hmdp
    grid = h.load_golden_txt("null_fit_hmdp_LLgrid.txt")
    for col, reml in ((1, True), (2, False)):
        L = lmm.LMM(Y, K)
        L.fit(reml=reml)
        h.assert_close(L.h_grid[:-1], grid[:, 0], what="H grid")
        h.assert_close(L.ll_grid[:-1], grid[:, col], what="LL grid REML=%s" % reml)


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_bed_default_fit_refines_the_h0_end(bed_null, reml):
    """README data: 1.0.0 returned the grid end h = 0 unrefined, but the maximum
    is inside [0, 0.01]; the default search finds it, as the oracle does."""
    oracle = pytest.importorskip("sims.oracle")
    Y, K, kva, kve = bed_null
    L = lmm.LMM(Y, K, kva.copy(), kve)
    L.fit(reml=reml)
    prefix = "bed_%s" % ("reml" if reml else "ml")
    assert 0.003 < L.opt_h < 0.005
    assert L.opt_ll > FITS[prefix + ".LL"] + 0.05  # higher than 1.0.0's h = 0
    # the oracle's profile (dense, with pylmm's clamp) on a fine bracket around it
    hs = np.linspace(0.0, 0.01, 201)
    k_clamped = oracle.clamp_kinship(K)
    prof = [oracle.profile_loglik(x, Y, k_clamped, reml=reml)[0] for x in hs]
    assert abs(hs[int(np.argmax(prof))] - L.opt_h) <= 0.0001
