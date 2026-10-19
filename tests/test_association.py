"""Per-SNP association through lmm.GWAS and LMM.association."""

import helpers as h
import numpy as np
import pytest

from pylmm import lmm

GWAS_COLS = [
    "snp_index",
    "n_missing",
    "ts_reml_norefit",
    "ps_reml_norefit",
    "ts_reml_refit",
    "ps_reml_refit",
    "ts_ml_norefit",
    "ps_ml_norefit",
]
REFIT_NSNPS = 200  # refit is ~10 ms/SNP emulated; first 200 SNPs is enough
# (scripts/example.py covers all 1000 with refit)


@pytest.fixture(scope="module")
def gwas_golden():
    g = h.load_golden_txt("assoc_hmdp_GWAS.txt")
    return {c: g[:, i] for i, c in enumerate(GWAS_COLS)}


# Note: the golden ts/ps_ml_norefit columns equal the REML columns exactly:
# in the original code REML never reached L.fit() in lmm.GWAS and the t-stat
# used N-q in both modes (bugs 1 and 9, docs/phase1-baseline.md).  Since
# Phase 5.2 GWAS(REML=False) is a real ML scan, so it is checked against the
# oracle in test_gwas_hmdp_ml below, not against those columns.
@pytest.mark.parametrize(
    ("reml", "refit", "tag", "nsnps"),
    [
        (True, False, "reml_norefit", None),
        (True, True, "reml_refit", REFIT_NSNPS),
    ],
)
def test_gwas_hmdp(hmdp, gwas_golden, reml, refit, tag, nsnps):
    """Missing genotypes left in place: GWAS drops them per SNP and refits."""
    Y, K, snps = hmdp
    X = snps[:, :nsnps].copy()
    t_stats, p_values = lmm.gwas(Y, X, K, reml=reml, refit=refit)
    m = X.shape[1]
    h.assert_fit_close(t_stats, gwas_golden["ts_" + tag][:m], what="ts_" + tag)
    h.assert_pvalues_close(p_values, gwas_golden["ps_" + tag][:m], what="ps_" + tag)
    h.assert_pvalues_valid(p_values, tag)
    # Each SNP uses only its non-missing individuals; q = intercept + SNP.
    df = len(Y) - gwas_golden["n_missing"][:m] - 2
    h.assert_p_matches_t(t_stats, p_values, df, tag)


def test_gwas_hmdp_ml_golden_columns_were_reml(gwas_golden):
    """Why ml_norefit is not compared with its golden: pre-5.2 it was REML."""
    np.testing.assert_array_equal(gwas_golden["ts_ml_norefit"], gwas_golden["ts_reml_norefit"])


def test_gwas_hmdp_ml(hmdp, gwas_golden):
    """GWAS(REML=False) = ML null fit + ML Wald test (sigma^2 = Q/n), on the
    first 60 hmdp SNPs, against the dense oracle with pylmm's grid/clamp, on
    each SNP's non-missing individuals.  It differs from the REML scan."""
    oracle = pytest.importorskip("sims.oracle")
    Y, K, snps = hmdp
    X = snps[:, :60].copy()
    t_stats, p_values = lmm.gwas(Y, X, K, reml=False)
    t_stats = np.asarray(t_stats)
    assert np.nanmax(np.abs(t_stats - gwas_golden["ts_reml_norefit"][:60])) > 1e-3
    for j in range(X.shape[1]):
        ok = ~np.isnan(X[:, j])
        x = X[ok, j]
        if np.ptp(x) == 0:
            assert np.isnan(t_stats[j])
            continue
        w = oracle.wald(
            Y[ok],
            K[np.ix_(ok, ok)],
            x,
            null_reml=False,
            sigma_convention="textbook",
            eig_clamp=1e-6,
            h_grid="pylmm",
        )
        assert abs(t_stats[j] - w.t) <= 1e-5 * max(1.0, abs(w.t)), (j, t_stats[j], w.t)
    df = len(Y) - gwas_golden["n_missing"][:60] - 2
    h.assert_p_matches_t(t_stats, p_values, df, "ml_norefit")


def test_association_imputed_with_beta(hmdp):
    """Fixed REML null fit, mean-imputed SNPs, returnBeta=True."""
    Y, K, snps = hmdp
    golden = h.load_golden_txt("assoc_hmdp_imputed.txt")
    L = lmm.LMM(Y, K)
    L.fit()
    rows = []
    for i in range(snps.shape[1]):
        x = snps[:, i].copy()
        x[np.isnan(x)] = x[~np.isnan(x)].mean()
        x = x.reshape(len(x), 1)
        if x.var() == 0:
            rows.append([i, np.nan, np.nan, np.nan, np.nan])
            continue
        ts, ps, beta, bvar = L.association(x, reml=True, return_beta=True)
        # bvar comes back as a 1x1 array (betaVAR * sigma); flatten to scalars
        rows.append([i, ts, ps, beta, np.asarray(bvar, dtype=np.float64).sum()])
    rows = np.array(rows, dtype=np.float64)
    for j, name in enumerate(["snp_index", "ts", "ps", "beta", "beta_var"]):
        if name == "ps":
            h.assert_pvalues_close(rows[:, j], golden[:, j], what="imputed ps")
        else:
            h.assert_fit_close(rows[:, j], golden[:, j], what="imputed " + name)
    h.assert_pvalues_valid(rows[:, 2], "imputed")
    h.assert_p_matches_t(rows[:, 1], rows[:, 2], len(Y) - 2, "imputed")
    # t = beta / sqrt(beta_var)
    ok = ~np.isnan(rows[:, 1])
    np.testing.assert_allclose(rows[ok, 1], rows[ok, 3] / np.sqrt(rows[ok, 4]), rtol=h.PVAL_T_RTOL)
