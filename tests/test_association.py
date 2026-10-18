"""Per-SNP association through lmm.GWAS and LMM.association."""
from __future__ import division, print_function

import numpy as np
import pytest

import helpers as h
from pylmm import lmm

GWAS_COLS = ('snp_index n_missing ts_reml_norefit ps_reml_norefit '
             'ts_reml_refit ps_reml_refit ts_ml_norefit ps_ml_norefit').split()
REFIT_NSNPS = 200   # refit is ~10 ms/SNP emulated; first 200 SNPs is enough
                    # (scripts/example.py covers all 1000 with refit)


@pytest.fixture(scope='module')
def gwas_golden():
    g = h.load_golden_txt('assoc_hmdp_GWAS.txt')
    return dict((c, g[:, i]) for i, c in enumerate(GWAS_COLS))


# Note: ts/ps_ml_norefit equal the REML columns exactly, because REML never
# reaches L.fit() in lmm.GWAS and the t-stat uses N-q in both modes (known
# bugs, docs/phase1-baseline.md #1, #9).  ML fitting itself is covered by
# test_null_fit.py::test_hmdp_null_fit[ml].
@pytest.mark.parametrize('reml,refit,tag,nsnps', [
    (True, False, 'reml_norefit', None),
    (True, True, 'reml_refit', REFIT_NSNPS),
    (False, False, 'ml_norefit', None),
])
def test_gwas_hmdp(hmdp, gwas_golden, reml, refit, tag, nsnps):
    """Missing genotypes left in place: GWAS drops them per SNP and refits."""
    Y, K, snps = hmdp
    X = snps[:, :nsnps].copy()
    TS, PS = lmm.GWAS(Y, X, K, REML=reml, refit=refit)
    m = X.shape[1]
    h.assert_close(TS, gwas_golden['ts_' + tag][:m], rtol=h.RTOL_FIT,
                   atol=h.ATOL_FIT, what='ts_' + tag)
    h.assert_close(PS, gwas_golden['ps_' + tag][:m], rtol=h.RTOL_FIT,
                   atol=h.ATOL_FIT, what='ps_' + tag)
    h.assert_pvalues_valid(PS, tag)
    # Each SNP uses only its non-missing individuals; q = intercept + SNP.
    df = len(Y) - gwas_golden['n_missing'][:m] - 2
    h.assert_p_matches_t(TS, PS, df, tag)


def test_gwas_golden_nan_pattern(gwas_golden):
    """nan p-values are exactly the 68 monomorphic-after-missing SNPs."""
    assert np.isnan(gwas_golden['ps_reml_norefit']).sum() == 68


def test_association_imputed_with_beta(hmdp):
    """Fixed REML null fit, mean-imputed SNPs, returnBeta=True."""
    Y, K, snps = hmdp
    golden = h.load_golden_txt('assoc_hmdp_imputed.txt')
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
        ts, ps, beta, bvar = L.association(x, REML=True, returnBeta=True)
        # bvar comes back as a 1x1 array (betaVAR * sigma); flatten to scalars
        rows.append([i, ts, ps, beta, np.asarray(bvar, dtype=np.float64).sum()])
    rows = np.array(rows, dtype=np.float64)
    for j, name in enumerate(['snp_index', 'ts', 'ps', 'beta', 'beta_var']):
        h.assert_close(rows[:, j], golden[:, j], rtol=h.RTOL_FIT, atol=h.ATOL_FIT,
                       what='imputed ' + name)
    h.assert_pvalues_valid(rows[:, 2], 'imputed')
    h.assert_p_matches_t(rows[:, 1], rows[:, 2], len(Y) - 2, 'imputed')
    # t = beta / sqrt(beta_var)
    ok = ~np.isnan(rows[:, 1])
    np.testing.assert_allclose(rows[ok, 1], rows[ok, 3] / np.sqrt(rows[ok, 4]),
                               rtol=h.PVAL_T_RTOL)
