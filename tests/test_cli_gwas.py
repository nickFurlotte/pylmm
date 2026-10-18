"""scripts/pylmmGWAS.py end to end, on subsets, outputs parsed numerically."""
from __future__ import division, print_function

import os

import numpy as np
import pytest

import helpers as h

BED_N = 1219


def check_out(out, golden_name, df=None):
    ids, got = h.load_gwas_out(out)
    gids, exp = h.load_gwas_out(h.golden_path(golden_name))
    assert ids == gids
    for j, col in enumerate(['BETA', 'BETA_SD', 'F_STAT', 'P_VALUE']):
        h.assert_close(got[:, j], exp[:, j], rtol=h.RTOL_FIT, atol=h.ATOL_FIT,
                       what='%s %s' % (golden_name, col))
    h.assert_pvalues_valid(got[:, 3], golden_name)
    if df is not None:
        h.assert_p_matches_t(got[:, 2], got[:, 3], df, golden_name)
        ok = ~np.isnan(got[:, 2])
        np.testing.assert_allclose(got[ok, 2], got[ok, 0] / got[ok, 1],
                                   rtol=h.PVAL_T_RTOL)


def bed_args(base):
    return ['-v', '--bfile', base, '--kfile', h.README_KIN,
            '--phenofile', h.FAKE_PHENOS]


def test_cli_bed_readme_command(tmp_path):
    """README command on the first 1000 SNPs (missing genotypes imputed)."""
    base = h.write_bed_subset(tmp_path, h.BED_GWAS_NSNPS)
    out = str(tmp_path / 'out')
    h.run_script('pylmmGWAS.py', bed_args(base) + [out])
    check_out(out, 'gwas_cli_bed_snps1000.out', df=BED_N - 2)


@pytest.fixture(scope='module')
def bed50(tmp_path_factory):
    return h.write_bed_subset(tmp_path_factory.mktemp('bed50'),
                              h.BED_GWAS_SMALL_NSNPS)


@pytest.mark.parametrize('flags,tag,df', [
    (['--REML'], 'reml', BED_N - 2),
    (['--refit'], 'refit', BED_N - 2),
    (['--REML', '--refit'], 'reml_refit', BED_N - 2),
    # per-SNP N varies with missingness, so no df check here
    (['--removeMissingGenotypes'], 'removeMissing', None),
])
def test_cli_bed_variants(tmp_path, bed50, flags, tag, df):
    out = str(tmp_path / 'out')
    h.run_script('pylmmGWAS.py', bed_args(bed50) + flags + [out])
    check_out(out, 'gwas_cli_bed_snps50_%s.out' % tag, df=df)


@pytest.mark.parametrize('cov', [False, True], ids=['nocov', 'cov'])
def test_cli_emma(tmp_path, cov):
    emma = h.write_emma_subset(tmp_path)
    args = ['-v', '--emmaSNP', emma, '--emmaPHENO', h.HMDP_Y,
            '--kfile', h.HMDP_K_TXT]
    if cov:
        args += ['--emmaCOV', h.write_emma_cov(tmp_path)]
    out = str(tmp_path / 'out')
    h.run_script('pylmmGWAS.py', args + [out])
    name = 'gwas_cli_emma_hmdp_snps1000%s.out' % ('_cov' if cov else '')
    check_out(out, name, df=94 - (3 if cov else 2))


@pytest.mark.slow
def test_cli_bed_full_readme_command(tmp_path):
    """Full README GWAS (98,740 SNPs, ~2 min emulated); first 1000 rows must
    match the 1000-SNP golden."""
    out = str(tmp_path / 'out')
    h.run_script('pylmmGWAS.py', bed_args(h.BED_BASE) + [out])
    ids, got = h.load_gwas_out(out)
    assert len(ids) == 98740
    gids, exp = h.load_gwas_out(h.golden_path('gwas_cli_bed_snps1000.out'))
    assert ids[:1000] == gids
    h.assert_close(got[:1000], exp, rtol=h.RTOL_FIT, atol=h.ATOL_FIT,
                   what='full README GWAS, first 1000 rows')
    h.assert_pvalues_valid(got[:, 3], 'full README GWAS')
