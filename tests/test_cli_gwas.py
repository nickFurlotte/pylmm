"""scripts/pylmmGWAS.py end to end, on subsets, outputs parsed numerically.

Every run uses -v, and the null-model fit it prints ("heritability=%0.3f,
sigma=%0.3f") is checked against gwas_cli_nullfit.kv.txt and, where the same
model is in null_fits.kv.txt, against the Python API fit.  (The null fit is
REML by default; --REML is a no-op since Phase 5.2, which also made --refit
and --removeMissingGenotypes REML: bug 1, and their goldens were regenerated,
see tests/golden/README.md.)  That
check is what catches a kinship that was read wrongly, e.g. silently
zero-filled.
"""

import helpers as h
import numpy as np
import pytest

BED_N = 1219
HMDP_N = 94
NULLFIT = h.load_kv("gwas_cli_nullfit.kv.txt")
FITS = h.load_kv("null_fits.kv.txt")


def check_out(out, golden_name, df=None, nrows=None):
    ids, got = h.load_gwas_out(out)
    gids, exp = h.load_gwas_out(h.golden_path(golden_name), nrows=nrows)
    assert ids == gids
    for j, col in enumerate(["BETA", "BETA_SD", "F_STAT"]):
        h.assert_fit_close(got[:, j], exp[:, j], what="%s %s" % (golden_name, col))
    h.assert_pvalues_close(got[:, 3], exp[:, 3], what="%s P_VALUE" % golden_name)
    h.assert_pvalues_valid(got[:, 3], golden_name)
    if df is not None:
        h.assert_p_matches_t(got[:, 2], got[:, 3], df, golden_name)
        ok = ~np.isnan(got[:, 2])
        np.testing.assert_allclose(got[ok, 2], got[ok, 0] / got[ok, 1], rtol=h.PVAL_T_RTOL)
    return got


def check_null_fit(output, key, api=None):
    """The -v null fit equals the golden (to the printed precision) and, if
    given, the API REML fit null_fits.kv.txt[api] (to rounding)."""
    hh, ss = h.null_fit_from_log(output)
    for name, got in (("h", hh), ("sigma", ss)):
        exp = NULLFIT["%s.%s" % (key, name)]
        assert abs(got - exp) <= h.CLI_PRINT_ATOL, "%s: -v %s=%r, golden %r" % (key, name, got, exp)
    if api is not None:
        for name, got in (("h", hh), ("sigma", ss)):
            exp = FITS["%s.%s" % (api, name)]
            assert abs(got - exp) <= 0.5e-3 + 1e-9, "%s: -v %s=%r but the API fit (%s) gives %r" % (
                key,
                name,
                got,
                api,
                exp,
            )


def bed_args(base, phenofile=h.FAKE_PHENOS):
    return ["-v", "--bfile", base, "--kfile", h.README_KIN, "--phenofile", phenofile]


def test_cli_bed_readme_command(tmp_path):
    """README command on the first 1000 SNPs (missing genotypes imputed)."""
    base = h.write_bed_subset(tmp_path, h.BED_GWAS_NSNPS)
    out = str(tmp_path / "out")
    log = h.run_script("pylmmGWAS.py", [*bed_args(base), out], cwd=tmp_path)
    check_out(out, "gwas_cli_bed_snps1000.out", df=BED_N - 2)
    check_null_fit(log, "bed_snps1000", api="bed_reml")


@pytest.fixture(scope="module")
def bed50(tmp_path_factory):
    return h.write_bed_subset(tmp_path_factory.mktemp("bed50"), h.BED_GWAS_SMALL_NSNPS)


@pytest.mark.parametrize(
    ("flags", "tag", "df"),
    [
        (["--REML"], "reml", BED_N - 2),
        (["--refit"], "refit", BED_N - 2),
        (["--REML", "--refit"], "reml_refit", BED_N - 2),
        # per-SNP N varies with missingness, so no df check here
        (["--removeMissingGenotypes"], "removeMissing", None),
    ],
)
def test_cli_bed_variants(tmp_path, bed50, flags, tag, df):
    out = str(tmp_path / "out")
    log = h.run_script("pylmmGWAS.py", bed_args(bed50) + flags + [out], cwd=tmp_path)
    check_out(out, "gwas_cli_bed_snps50_%s.out" % tag, df=df)
    if "--refit" not in flags:  # no null fit (and no -v line) with --refit
        check_null_fit(log, "bed_snps50_" + tag, api="bed_reml")


def test_cli_bed_covfile(tmp_path, bed50):
    """--covfile (input.getCovariates), rows in reverse order: X0 = [1, c1, c2]."""
    out = str(tmp_path / "out")
    args = [*bed_args(bed50), "--covfile", h.write_covfile(tmp_path), out]
    log = h.run_script("pylmmGWAS.py", args, cwd=tmp_path)
    check_out(out, "gwas_cli_bed_snps50_covfile.out", df=BED_N - 4)
    check_null_fit(log, "bed_snps50_covfile")


def test_cli_bed_missing_phenotypes(tmp_path, bed50):
    """'NA' and '-9' phenotypes (input.getPhenos), rows in reverse order;
    the CLI drops those individuals from Y, X0 and K."""
    out = str(tmp_path / "out")
    args = [*bed_args(bed50, phenofile=h.write_missing_phenofile(tmp_path)), out]
    log = h.run_script("pylmmGWAS.py", args, cwd=tmp_path)
    nmiss = len(h.BED_MISSING_PHENO_NA) + len(h.BED_MISSING_PHENO_M9)
    assert "warning: %d of " % nmiss in log and "have a missing phenotype and are dropped" in log
    check_out(out, "gwas_cli_bed_snps50_missingpheno.out", df=BED_N - nmiss - 2)
    check_null_fit(log, "bed_snps50_missingpheno")


def test_cli_tped(tmp_path):
    """--tfile on the same 50 SNPs as transposed PLINK: must equal its golden
    and the first 50 rows of the --bfile README run."""
    base = h.write_tped_subset(tmp_path, h.BED_GWAS_SMALL_NSNPS)
    out = str(tmp_path / "out")
    log = h.run_script(
        "pylmmGWAS.py",
        ["-v", "--tfile", base, "--kfile", h.README_KIN, "--phenofile", h.FAKE_PHENOS, out],
        cwd=tmp_path,
    )
    check_out(out, "gwas_cli_tped_snps50.out", df=BED_N - 2)
    check_out(out, "gwas_cli_bed_snps1000.out", nrows=h.BED_GWAS_SMALL_NSNPS)
    check_null_fit(log, "tped_snps50", api="bed_reml")


def emma_args(emma, kfile, pheno=h.HMDP_Y):
    return ["-v", "--emmaSNP", emma, "--emmaPHENO", pheno, "--kfile", kfile]


@pytest.mark.parametrize("cov", [False, True], ids=["nocov", "cov"])
def test_cli_emma(tmp_path, cov):
    emma = h.write_emma_subset(tmp_path)
    args = emma_args(emma, h.write_headerless_kinship(tmp_path))
    if cov:
        args += ["--emmaCOV", h.write_emma_cov(tmp_path)]
    out = str(tmp_path / "out")
    log = h.run_script("pylmmGWAS.py", [*args, out], cwd=tmp_path)
    name = "gwas_cli_emma_hmdp_snps1000%s" % ("_cov" if cov else "")
    got = check_out(out, name + ".out", df=HMDP_N - (3 if cov else 2))
    check_null_fit(log, name[len("gwas_cli_") :], api=None if cov else "hmdp_reml")
    if not cov:
        # Independent of the CLI goldens: with the same K, Y and REML null
        # fit, the CLI t-stats (mean-imputed, standardised SNPs) equal the
        # API's L.association() on mean-imputed SNPs.
        imp = h.load_golden_txt("assoc_hmdp_imputed.txt")
        h.assert_fit_close(got[:, 2], imp[:, 1], what="EMMA CLI t vs API")


def test_cli_emma_gzip_kfile(tmp_path):
    """The .gz branch of the --kfile reader; same K as test_cli_emma."""
    emma = h.write_emma_subset(tmp_path, h.HMDP_GZ_NSNPS)
    out = str(tmp_path / "out")
    args = [*emma_args(emma, h.write_headerless_kinship(tmp_path, gz=True)), out]
    log = h.run_script("pylmmGWAS.py", args, cwd=tmp_path)
    check_out(out, "gwas_cli_emma_hmdp_snps1000.out", df=HMDP_N - 2, nrows=h.HMDP_GZ_NSNPS)
    check_null_fit(log, "emma_hmdp_snps1000", api="hmdp_reml")


def test_cli_emma_missing_phenotypes(tmp_path):
    """'NA' entries in --emmaPHENO."""
    emma = h.write_emma_subset(tmp_path, h.HMDP_GZ_NSNPS)
    out = str(tmp_path / "out")
    args = [
        *emma_args(
            emma, h.write_headerless_kinship(tmp_path), pheno=h.write_emma_missing_pheno(tmp_path)
        ),
        out,
    ]
    log = h.run_script("pylmmGWAS.py", args, cwd=tmp_path)
    nmiss = len(h.HMDP_MISSING_Y)
    assert "warning: %d of " % nmiss in log and "have a missing phenotype and are dropped" in log
    check_out(out, "gwas_cli_emma_hmdp_snps100_missingpheno.out", df=HMDP_N - nmiss - 2)
    check_null_fit(log, "emma_hmdp_snps100_missingpheno", api="hmdp_missingY_reml")


@pytest.mark.slow
def test_cli_bed_full_readme_command(tmp_path):
    """Full README GWAS (98,740 SNPs, ~2 min emulated); first 1000 rows must
    match the 1000-SNP golden."""
    out = str(tmp_path / "out")
    log = h.run_script("pylmmGWAS.py", [*bed_args(h.BED_BASE), out], cwd=tmp_path)
    ids, got = h.load_gwas_out(out)
    assert len(ids) == 98740
    gids, exp = h.load_gwas_out(h.golden_path("gwas_cli_bed_snps1000.out"))
    assert ids[:1000] == gids
    for j, col in enumerate(["BETA", "BETA_SD", "F_STAT"]):
        h.assert_fit_close(
            got[:1000, j], exp[:, j], what="full README GWAS, first 1000 rows, " + col
        )
    h.assert_pvalues_close(
        got[:1000, 3], exp[:, 3], what="full README GWAS, first 1000 rows, P_VALUE"
    )
    h.assert_pvalues_valid(got[:, 3], "full README GWAS")
    check_null_fit(log, "bed_snps1000", api="bed_reml")
