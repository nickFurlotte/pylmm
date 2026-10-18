"""scripts/example.py runs end to end and produces the golden values."""
from __future__ import division, print_function

import os
import runpy

import numpy as np

import helpers as h


def test_example_py(monkeypatch):
    monkeypatch.chdir(h.REPO)   # example.py uses relative data/ paths
    g = runpy.run_path(os.path.join(h.SCRIPTS, 'example.py'), run_name='__main__')
    exp = h.load_kv('example_py.kv.txt')

    def close(v, k):
        h.assert_close(v, exp[k], rtol=h.RTOL_FIT, atol=h.ATOL_FIT, what=k)

    L = g['L']   # last fit is the refit on SNP 0
    close(L.optH, 'snp0_refit_h')
    close(g['ts'], 'snp0_refit_ts')
    close(g['ps'], 'snp0_refit_ps')

    TS, PS = np.asarray(g['TS']), np.asarray(g['PS'])
    assert len(PS) == exp['gwas_n']
    assert np.isnan(PS).sum() == exp['gwas_n_nan']
    close(TS[0], 'gwas_ts_0')
    close(PS[0], 'gwas_ps_0')
    close(np.nanmin(PS), 'gwas_ps_min')
    assert np.nanargmin(PS) == exp['gwas_ps_argmin']
    close(np.nansum(np.log10(PS)), 'gwas_sum_log10_ps')
    h.assert_pvalues_valid(PS, 'example.py')

    # SNPs 1..999 equal the lmm.GWAS REML+refit golden (SNP 0 was imputed
    # in place by example.py, see docs/phase1-baseline.md).
    gw = h.load_golden_txt('assoc_hmdp_GWAS.txt')
    h.assert_close(TS[1:], gw[1:, 4], rtol=h.RTOL_FIT, atol=h.ATOL_FIT,
                   what='example.py TS[1:]')
    h.assert_close(PS[1:], gw[1:, 5], rtol=h.RTOL_FIT, atol=h.ATOL_FIT,
                   what='example.py PS[1:]')
