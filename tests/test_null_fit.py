"""Null-model fits: LMM.fit() under REML/ML, missing Y, covariates."""

import helpers as h
import numpy as np
import pytest

from pylmm import lmm

FITS = h.load_kv("null_fits.kv.txt")


def check_fit(L, prefix):
    """Compare an LMM after fit() against null_fits.kv.txt[prefix.*]."""
    beta = np.asarray(L.optBeta).ravel()
    mn, vx = L.meanAndVar()
    got = {
        "h": L.optH,
        "sigma": L.optSigma,
        "LL": L.optLL,
        "h_posterior_mean": mn,
        "h_posterior_var": vx,
    }
    for i, b in enumerate(beta):
        got["beta_%d" % i] = b
    expected_keys = sorted(k[len(prefix) + 1 :] for k in FITS if k.startswith(prefix + "."))
    assert sorted(list(got) + ["N"]) == expected_keys
    assert FITS[prefix + ".N"] == L.N
    for k in sorted(got):
        h.assert_close(
            got[k], FITS[prefix + "." + k], rtol=h.RTOL_FIT, atol=h.ATOL_FIT, what=prefix + "." + k
        )
    assert 0.0 <= L.optH <= 1.0, "%s: h=%r outside [0, 1]" % (prefix, L.optH)


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_hmdp_null_fit(hmdp, reml):
    Y, K, _ = hmdp
    L = lmm.LMM(Y, K)
    L.fit(REML=reml)
    check_fit(L, "hmdp_%s" % ("reml" if reml else "ml"))


def test_hmdp_LL_grid(hmdp):
    """LL over the fixed h grid: no optimiser involved, so tight tolerance."""
    Y, K, _ = hmdp
    grid = h.load_golden_txt("null_fit_hmdp_LLgrid.txt")
    for col, reml in ((1, True), (2, False)):
        L = lmm.LMM(Y, K)
        L.fit(REML=reml)
        h.assert_close(L.H, grid[:, 0], what="H grid")
        h.assert_close(L.LLs, grid[:, col], what="LL grid REML=%s" % reml)


def test_hmdp_missing_y(hmdp):
    Y, K, _ = hmdp
    Ym = Y.copy()
    Ym[h.HMDP_MISSING_Y] = np.nan
    L = lmm.LMM(Ym, K)
    L.fit(REML=True)
    check_fit(L, "hmdp_missingY_reml")


def test_hmdp_covariate(hmdp):
    Y, K, snps = hmdp
    c = snps[:, h.HMDP_COV_SNP].copy()
    c[np.isnan(c)] = c[~np.isnan(c)].mean()
    L = lmm.LMM(Y, K, X0=np.column_stack([np.ones(len(Y)), c]))
    L.fit(REML=True)
    check_fit(L, "hmdp_cov_reml")


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_bed_null_fit(bed_null, reml):
    """README data; fits at the h=0 boundary (getMax endpoint branch)."""
    Y, K, Kva, Kve = bed_null
    L = lmm.LMM(Y, K, Kva.copy(), Kve)
    L.fit(REML=reml)
    check_fit(L, "bed_%s" % ("reml" if reml else "ml"))
