"""scripts/example.py runs end to end and produces the golden values."""

import os
import runpy

import helpers as h
import numpy as np

from pylmm import lmm


def test_example_py(monkeypatch):
    monkeypatch.chdir(h.REPO)  # example.py uses relative data/ paths

    # example.py overwrites L, ts and ps, so record the intermediate results
    # (the null fit and the first, non-refit association) as they happen.
    fits, assocs = [], []
    orig_fit, orig_assoc = lmm.LMM.fit, lmm.LMM.association

    def fit(self, *a, **kw):
        res = orig_fit(self, *a, **kw)
        fits.append((self.opt_h, self.opt_sigma, self.opt_ll))
        return res

    def association(self, *a, **kw):
        res = orig_assoc(self, *a, **kw)
        assocs.append(res)
        return res

    monkeypatch.setattr(lmm.LMM, "fit", fit)
    monkeypatch.setattr(lmm.LMM, "association", association)

    g = runpy.run_path(os.path.join(h.SCRIPTS, "example.py"), run_name="__main__")
    exp = h.load_kv("example_py.kv.txt")

    def close(v, k):
        if k.endswith("ps") or "_ps_" in k:
            h.assert_pvalues_close(v, exp[k], what=k)
        else:
            h.assert_fit_close(v, exp[k], what=k)

    null_h, null_sigma, null_ll = fits[0]  # L.fit()
    close(null_h, "null_h")
    close(null_sigma, "null_sigma")
    close(null_ll, "null_LL")
    snp0_ts, snp0_ps = assocs[0]  # L.association(X), no refit
    close(snp0_ts, "snp0_ts")
    close(snp0_ps, "snp0_ps")

    L = g["L"]  # last fit is the refit on SNP 0
    close(L.opt_h, "snp0_refit_h")
    close(g["ts"], "snp0_refit_ts")
    close(g["ps"], "snp0_refit_ps")

    t_stats, p_values = np.asarray(g["t_stats"]), np.asarray(g["p_values"])
    assert len(p_values) == exp["gwas_n"]
    assert np.isnan(p_values).sum() == exp["gwas_n_nan"]
    close(t_stats[0], "gwas_ts_0")
    close(p_values[0], "gwas_ps_0")
    close(np.nanmin(p_values), "gwas_ps_min")
    assert np.nanargmin(p_values) == exp["gwas_ps_argmin"]
    # a sum of ~930 log10(p) terms (about -286): relative tolerance only
    h.assert_close(
        np.nansum(np.log10(p_values)),
        exp["gwas_sum_log10_ps"],
        rtol=h.RTOL_FIT,
        atol=0,
        what="gwas_sum_log10_ps",
    )
    h.assert_pvalues_valid(p_values, "example.py")

    # SNPs 1..999 equal the lmm.gwas REML+refit golden (SNP 0 was imputed
    # in place by example.py, see docs/phase1-baseline.md).
    gw = h.load_golden_txt("assoc_hmdp_GWAS.txt")
    h.assert_fit_close(t_stats[1:], gw[1:, 4], what="example.py TS[1:]")
    h.assert_pvalues_close(p_values[1:], gw[1:, 5], what="example.py PS[1:]")
