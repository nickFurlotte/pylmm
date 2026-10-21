"""
Shared helpers for the pylmm regression suite: paths, tolerances, golden-file
readers, input loaders and CLI runners.

Everything here must work unchanged on Python 2.7 (today) and Python 3
(after the Phase 3 port), so: no f-strings, no py2-only builtins.
"""

import gzip
import os
import re
import shutil
import subprocess
import sys
import threading

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data")
GOLDEN = os.path.join(REPO, "tests", "golden")
SCRIPTS = os.path.join(REPO, "scripts")  # scripts/example.py

BED_BASE = os.path.join(DATA, "snps.132k.clean.noX")
README_KIN = BED_BASE + ".pylmm.kin"
FAKE_PHENOS = BED_BASE + ".fake.phenos"
HMDP_Y = os.path.join(DATA, "hmdp.liver.exprs.1")
HMDP_SNPS = os.path.join(DATA, "hmdp.liver.snps")
HMDP_K_TXT = os.path.join(GOLDEN, "hmdp_K_npdump.txt")

# Inputs used by tests/golden/generate_golden.py (see tests/golden/README.md).
HMDP_NSNPS = 1000
HMDP_MISSING_Y = [5, 23, 60]
HMDP_COV_SNP = 10
BED_KIN_NSNPS = 2000
BED_GWAS_NSNPS = 1000
BED_GWAS_SMALL_NSNPS = 50
BED_BLOCK = 100
HMDP_GZ_NSNPS = 100
BED_MISSING_PHENO_NA = [3, 100, 500]
BED_MISSING_PHENO_M9 = [7, 800]

# Wall-clock limit for one CLI subprocess (the slowest, the full README GWAS,
# takes ~2 min emulated).  A port bug such as a broken EOF check in an input
# iterator would otherwise hang the suite forever.
CLI_TIMEOUT = 900

# ---------------------------------------------------------------------------
# Tolerances -- the ONLY place they are defined.  See tests/README.md.
#
# Today (py2.7, same image as the goldens) everything matches bit for bit.
# The tolerances are sized for a different numpy/BLAS/LAPACK (Phase 3), not
# for today's run.
# ---------------------------------------------------------------------------

# Deterministic linear algebra with no optimiser in the loop: kinship
# matrices, the LL grid at fixed h, eigenvalues of well-separated spectra.
RTOL = 1e-10
ATOL = 1e-12

# Anything downstream of scipy.optimize.brent (fitted h, sigma, beta, LL and
# every t-stat that uses a fitted h).  Brent stops at xtol ~1.5e-8 relative,
# so ulp-level changes in the inputs move the optimum by ~1e-8, and every
# per-SNP refit (refit=True, or a SNP with missing genotypes) adds its own
# jitter.
#
# Measurement (Phase 2 review, py2.7 image): K -> K*(1 + eps*N), N a
# symmetric standard-normal matrix, 200 seeds at eps=1e-15 and 100 seeds at
# eps=1e-14, then the REML null fit, lmm.gwas (REML, no refit, all 1000
# SNPs; refit, first 200 SNPs) and the mean-imputed association were
# compared with the goldens.  Worst case over all 300 runs:
#   fitted h        5.7e-8 abs      (1.3e-7 rel)
#   sigma / LL      1.1e-9 / 8e-13 abs
#   t-stats         5.0e-7 abs (refit), 1.3e-7 (norefit), 1.0e-7 (imputed);
#                   up to 1.6e-5 rel for |t| ~ 1e-3; the part not covered
#                   by rtol=1e-5 was at most 2.2e-8 abs
#   beta            1.1e-8 abs (1.6e-5 rel at |beta| ~ 1e-3)
#   p-values        1.2e-6 rel (5.4e-7 on the log10 scale)
# The old ATOL_FIT=1e-8 was below that (4.7e-8 abs was measured at
# |t|=0.003).  ATOL_FIT is 10x the worst absolute t-stat change; RTOL_P is
# ~8x the worst relative p-value change.  Both are still orders of magnitude
# below any real behavioural change.
RTOL_FIT = 1e-5
ATOL_FIT = 5e-6
RTOL_P = 1e-5

# `pylmm gwas -v` logs the null fit with %0.3f; allow one unit in the last
# printed digit (a value sitting on a rounding boundary may flip).
CLI_PRINT_ATOL = 1.0e-3 + 1e-9

# Eigenvalue files: most entries are LAPACK round-off around zero
# (|e| < 1e-10; 1109 of 1219 for the README kinship, 50 of 94 for the EMMA
# CLI kinship).  Their sign and relative value are meaningless, so compare
# with an absolute tolerance scaled by the largest eigenvalue.
EIG_ATOL_SCALE = 1e-10

# Batched vs one-SNP-at-a-time Wald tests (LMM.association_batch vs
# LMM.association; gwas() and `pylmm gwas` with and without --no-batch): the
# same formulas at the same h, different floating-point order (one GEMM over a
# block instead of a GEMV per SNP, a GLS residual instead of a (q+1)x(q+1)
# inverse).  |a - b| <= BATCH_RTOL * |b| + BATCH_ATOL_SCALE * max|b|; the
# second term covers values near 0 (t ~ 1e-3), whose error is absolute.
#
# Measurement (Phase 6.1, macOS arm64, numpy 2.5 / Accelerate): every
# comparison of tests/test_batch.py (143 arrays: synthetic data with 0-4
# covariates of scales 1 to 100 and 1e-3, REML and ML, the HMDP and README BED
# data, the EMMA CLI with and without --emma-covar), recorded by running it
# with PYLMM_RECORD_BATCH=<file>:
#   max |a - b| / max|b|                        3.5e-12  (p-values, EMMA CLI)
#   max |a - b| / |b| for |b| >= 1% of max|b|   9.5e-11  (BETA, EMMA CLI with
#       --emma-covar; against a whitened-QR least-squares reference the
#       per-SNP path was off by 9.5e-11 there, the batched one by 2.1e-12)
# and sims/bench/gwas_batch.py (n = 500 and 2000, up to 20,000 SNPs) stayed
# below both.  BATCH_RTOL is 10x the worst relative difference, BATCH_ATOL_SCALE
# ~6x the worst absolute one; both are far below any change of the model.
#
# These hold for WELL-CONDITIONED designs only (the data above).  With a badly
# scaled covariate next to the intercept (a raw date ~2e7, 1e4 +- 1, a year
# column) the 1.0.0 per-SNP path, which inverts X'SX, loses up to ~1e-6
# relative and the two paths differ by that much; the batched path stays
# accurate (BATCH_REF_RTOL below).  A failure of assert_batch_close on such
# data is the old path's error: do not loosen these to make it pass, compare
# with a reference instead (test_batch.py::test_batch_accurate_on_badly_scaled_covariates).
BATCH_RTOL = 1e-9
BATCH_ATOL_SCALE = 2e-11

# Batched Wald test vs an accurate reference on a badly scaled covariate: GLS by
# whitened QR (numpy) on the same model with the covariate centred and scaled
# (same column span, so the same SNP statistics in exact arithmetic, and a
# well-conditioned problem).  Measurement (macOS arm64, numpy 2.5 / Accelerate,
# n = 300, 200 SNPs, intercept + one covariate, REML, at the fitted h), max
# relative error in t over all SNPs (beta alike; beta_var <= 5e-13):
#   covariate             batched   1.0.0 per-SNP path
#   19000 +- 100          2.3e-12   1.4e-10
#   year 2000-2020        4.8e-12   1.8e-8
#   2e7 +- 1e3            1.1e-10   3.2e-6
#   1e4 +- 1              5.6e-11   2.6e-7
# (the review's 40-digit mpmath reference agreed: batched 3e-11, per-SNP 3e-7 off
# for 1e4 + N(0, 1)).  Relative errors of t near 0 are absolute in nature, so the
# test allows |a - b| <= BATCH_REF_RTOL |b| + BATCH_REF_ATOL_SCALE max|b|.  On the
# test's own data (year, 2e7 +- 1e3, 1e4 +- 1; REML and ML; t, beta, beta_var and
# gwas() t) the batched path's worst errors were 5.2e-13 of max|b| and 3.5e-11
# relative (entries >= 1% of max), while the 1.0.0 path fails the same check on
# the 2e7 and 1e4 covariates (6e-10 to 3.5e-9 of max|t|).  BATCH_REF_RTOL is ~300x
# and BATCH_REF_ATOL_SCALE ~200x the batched path's worst errors.
BATCH_REF_RTOL = 1e-8
BATCH_REF_ATOL_SCALE = 1e-10

# Phase 6.3 review (B1): the fitted h of the default search must not depend on
# an affine reparametrisation of a covariate (raw vs centred or scaled), since
# the likelihood is invariant to X0 -> X0 A; only Brent's stopping (xatol 1e-10)
# and rounding on a nearly flat profile separate the fits.  Measured: 150
# datasets (n = 100, m = 500, h2 in {0.5, 0.9, 0.98}, an intercept and a date
# 2e7 +- 1e3 raw vs centred, REML and ML) max |dh| = 1.5e-7; the seed-62
# reproducer 1.4e-8.  (1.0.0's normal-equation fit, legacy_fit=True, differs by
# up to 4e-4 on the same data; before the fix the default differed by 0.14.)
H_REPARAM_ATOL = 1e-6

# Structural invariants.
SYM_ATOL = 1e-12  # |K - K.T|
PSD_RTOL = 1e-10  # min eigenvalue >= -PSD_RTOL * max eigenvalue
PVAL_T_RTOL = 1e-10  # p == 2*t.sf(|ts|, df), recomputed from ts


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def assert_close(actual, expected, rtol=RTOL, atol=ATOL, what=""):
    """assert_allclose that also requires nan in exactly the same places."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, "%s: shape %s != expected %s" % (
        what,
        actual.shape,
        expected.shape,
    )
    an, en = np.isnan(actual), np.isnan(expected)
    if not np.array_equal(an, en):
        bad = np.where(an != en)
        raise AssertionError(
            "%s: nan positions differ (%d mismatches, first at %s)"
            % (what, len(bad[0]), tuple(int(b[0]) for b in bad))
        )
    np.testing.assert_allclose(actual[~an], expected[~en], rtol=rtol, atol=atol, err_msg=what)


def assert_fit_close(actual, expected, what=""):
    """Optimiser-dependent values (h, sigma, beta, t-stats, LL, ...)."""
    assert_close(actual, expected, rtol=RTOL_FIT, atol=ATOL_FIT, what=what)


def assert_pvalues_close(actual, expected, what=""):
    """p-values: purely relative (atol=0), so p=1e-12 vs 2e-12 fails too."""
    assert_close(actual, expected, rtol=RTOL_P, atol=0.0, what=what)


def assert_batch_close(actual, expected, what=""):
    """Batched vs per-SNP test statistics (t, p, beta, se): see BATCH_RTOL.

    nan in exactly the same places.  For p-values, compare the p-values
    themselves (a p that underflows to 0.0 in one path must in the other).
    """
    expected_arr = np.asarray(expected, dtype=np.float64)
    ok = ~np.isnan(expected_arr)
    scale = np.abs(expected_arr[ok]).max() if ok.any() else 0.0
    _record_batch_diff(actual, expected_arr, scale, what)
    assert_close(actual, expected_arr, rtol=BATCH_RTOL, atol=BATCH_ATOL_SCALE * scale, what=what)


def _record_batch_diff(actual, expected, scale, what):
    fn = os.environ.get("PYLMM_RECORD_BATCH")
    if not fn:
        return
    a = np.asarray(actual, dtype=np.float64)
    ok = ~np.isnan(a) & ~np.isnan(expected)
    if not ok.any() or scale == 0:
        return
    d = np.abs(a[ok] - expected[ok])
    big = np.abs(expected[ok]) >= 1e-2 * scale
    with open(fn, "a") as f:
        f.write(
            "%s\t%.3e\t%.3e\n" % (what, d.max() / scale, (d[big] / np.abs(expected[ok][big])).max())
        )


def assert_ref_close(actual, reference, what=""):
    """Batched statistics vs an accurate reference: see BATCH_REF_RTOL."""
    reference = np.asarray(reference, dtype=np.float64)
    scale = np.abs(reference).max()
    if os.environ.get("PYLMM_RECORD_BATCH"):
        _record_batch_diff(actual, reference, scale, "ref " + what)
    assert_close(
        actual, reference, rtol=BATCH_REF_RTOL, atol=BATCH_REF_ATOL_SCALE * scale, what=what
    )


def assert_pvalues_valid(ps, what=""):
    ps = np.asarray(ps, dtype=np.float64)
    ok = ~np.isnan(ps)
    assert np.all((ps[ok] >= 0.0) & (ps[ok] <= 1.0)), "%s: p-values outside [0, 1]" % what


def assert_p_matches_t(ts, ps, df, what=""):
    """Two-sided t-test p-value must equal 2*sf(|t|, df); nan together."""
    from scipy import stats

    ts = np.asarray(ts, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    assert np.array_equal(np.isnan(ts), np.isnan(ps)), (
        "%s: nan positions differ between t-stats and p-values" % what
    )
    ok = ~np.isnan(ts)
    df = np.broadcast_to(np.asarray(df, dtype=np.float64), ts.shape)
    expected = 2.0 * stats.t.sf(np.abs(ts[ok]), df[ok])
    np.testing.assert_allclose(ps[ok], expected, rtol=PVAL_T_RTOL, atol=1e-300, err_msg=what)


# ---------------------------------------------------------------------------
# Golden-file readers (text only; no pickles)
# ---------------------------------------------------------------------------


def golden_path(name):
    return os.path.join(GOLDEN, name)


def load_golden_txt(name):
    return np.loadtxt(golden_path(name))


def load_kv(name):
    """Read a *.kv.txt golden file into {key: float}."""
    out = {}
    f = open(golden_path(name))
    try:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, v = line.split()
            out[k] = float(v)
    finally:
        f.close()
    return out


def load_gwas_out(fn, nrows=None):
    """Parse a `pylmm gwas` output file numerically.

    Returns (snp_ids, values) where values is an (nsnps, 4) float array with
    columns BETA, BETA_SD, F_STAT, P_VALUE ('nan' parses to nan).  nrows
    limits the read to the first nrows SNPs.
    """
    f = open(fn)
    try:
        header = f.readline().rstrip("\n").split("\t")
        assert header == ["SNP_ID", "BETA", "BETA_SD", "F_STAT", "P_VALUE"], header
        ids, rows = [], []
        for line in f:
            if nrows is not None and len(ids) == nrows:
                break
            parts = line.rstrip("\n").split("\t")
            assert len(parts) == 5, line
            ids.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
    finally:
        f.close()
    return ids, np.array(rows, dtype=np.float64).reshape(-1, 4)


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------


def load_npdump(fn):
    """Load one of the data/*.npdump files (Python 2 ndarray.dump pickles).

    The only place tests touch the pickles.  encoding='latin1' is what
    Python 3 needs to read py2 pickles of numpy arrays; py2 ignores it.
    """
    return np.load(fn, allow_pickle=True, encoding="latin1")


def load_hmdp_snps(nsnps=HMDP_NSNPS):
    """First nsnps SNPs of data/hmdp.liver.snps as individuals x SNPs."""
    return np.genfromtxt(HMDP_SNPS, max_rows=nsnps).T.copy()


def load_readme_kinship(n):
    K = np.fromfile(README_KIN, sep=" ")
    return K.reshape((n, n))


def read_bed_snps(nsnps, norm):
    from pylmm import plink_io

    reader = plink_io.PlinkReader(BED_BASE, type="b", norm_genotype=norm)
    reader.iter_snps()
    cols = []
    for _ in range(nsnps):
        g, _sid = next(reader)
        cols.append(g)
    return np.array(cols).T.copy()


# ---------------------------------------------------------------------------
# Subset inputs + CLI runner
# ---------------------------------------------------------------------------


def write_bed_subset(workdir, nsnps):
    """PLINK BED/BIM/FAM with the first nsnps SNPs (byte-identical to what
    generate_golden.py builds)."""
    workdir = str(workdir)
    base = os.path.join(workdir, "bed%d" % nsnps)
    with open(BED_BASE + ".fam") as f:
        nind = sum(1 for _ in f)
    nbytes = nind // 4 + (1 if nind % 4 else 0)
    with open(BED_BASE + ".bed", "rb") as src:
        data = src.read(3 + nsnps * nbytes)
    with open(base + ".bed", "wb") as dst:
        dst.write(data)
    with open(BED_BASE + ".bim", "rb") as src:
        lines = [src.readline() for _ in range(nsnps)]
    with open(base + ".bim", "wb") as dst:
        dst.write(b"".join(lines))
    shutil.copy(BED_BASE + ".fam", base + ".fam")
    return base


def write_emma_subset(workdir, nsnps=HMDP_NSNPS):
    fn = os.path.join(str(workdir), "hmdp.snps%d.emma" % nsnps)
    with open(HMDP_SNPS, "rb") as src:
        lines = [src.readline() for _ in range(nsnps)]
    with open(fn, "wb") as dst:
        dst.write(b"".join(lines))
    return fn


def write_emma_cov(workdir):
    """Row 2 (0-based 1) of data/hmdp.liver.exprs, the --emmaCOV input."""
    fn = os.path.join(str(workdir), "hmdp.cov.emma")
    with open(os.path.join(DATA, "hmdp.liver.exprs"), "rb") as src:
        src.readline()
        line = src.readline()
    with open(fn, "wb") as dst:
        dst.write(line)
    return fn


def write_headerless_kinship(workdir, gz=False):
    """The hmdp kinship as a plain '%.17g' text file with NO '#' header,
    byte-identical to what generate_golden.py fed to the old --kfile.

    tests/golden/hmdp_K_npdump.txt cannot be passed to --kinship directly:
    before Phase 5.1 pylmmGWAS.py read K with np.fromfile(sep=' '), which
    stops at the first '#', and K.resize() then zero-filled the rest (an
    all-zero kinship); it now rejects the '#' lines (bug 16)."""
    fn = os.path.join(str(workdir), "hmdp.K.txt")
    np.savetxt(fn, np.loadtxt(HMDP_K_TXT), fmt="%.17g")
    if not gz:
        return fn
    with open(fn, "rb") as src:
        data = src.read()
    f = gzip.open(fn + ".gz", "wb")
    try:
        f.write(data)
    finally:
        f.close()
    return fn + ".gz"


_TPED_CODES = {0: "1 1", 1: "0 0", 2: "1 2", 3: "2 2"}  # PLINK 2-bit codes


def write_tped_subset(workdir, nsnps):
    """.tped/.tfam/.map with the first nsnps BED SNPs, decoded straight from
    the .bed bytes (independently of pylmm's BED reader).  Byte-identical to
    what generate_golden.py builds."""
    base = os.path.join(str(workdir), "tped%d" % nsnps)
    with open(BED_BASE + ".fam") as f:
        nind = sum(1 for _ in f)
    nbytes = nind // 4 + (1 if nind % 4 else 0)
    with (
        open(BED_BASE + ".bed", "rb") as src,
        open(BED_BASE + ".bim") as bim,
        open(base + ".tped", "w") as tped,
        open(base + ".map", "w") as mapf,
    ):
        src.read(3)
        for _ in range(nsnps):
            raw = bytearray(src.read(nbytes))
            v = bim.readline().split()
            codes = [(raw[i // 4] >> (2 * (i % 4))) & 3 for i in range(nind)]
            tped.write(" ".join(v[:4] + [_TPED_CODES[c] for c in codes]) + "\n")
            mapf.write(" ".join(v[:4]) + "\n")
    shutil.copy(BED_BASE + ".fam", base + ".tfam")
    return base


def write_covfile(workdir):
    """PLINK covariate file (FID IID cos(i) i%3), rows in reverse fam order."""
    fn = os.path.join(str(workdir), "bed.cov")
    with open(BED_BASE + ".fam") as f:
        fam = [line.split()[:2] for line in f]
    lines = [
        "%s %s %s %d\n" % (fid, iid, "%.17g" % np.cos(i), i % 3) for i, (fid, iid) in enumerate(fam)
    ]
    with open(fn, "w") as f:
        f.write("".join(lines[::-1]))
    return fn


def write_missing_phenofile(workdir):
    """fake.phenos with BED_MISSING_PHENO_NA -> 'NA', BED_MISSING_PHENO_M9 ->
    '-9', rows in reverse order."""
    fn = os.path.join(str(workdir), "bed.missing.phenos")
    with open(FAKE_PHENOS) as f:
        lines = f.read().splitlines()
    out = []
    for i, line in enumerate(lines):
        v = line.split()
        if i in BED_MISSING_PHENO_NA:
            v[2] = "NA"
        elif i in BED_MISSING_PHENO_M9:
            v[2] = "-9"
        out.append(" ".join(v) + "\n")
    with open(fn, "w") as f:
        f.write("".join(out[::-1]))
    return fn


def write_emma_missing_pheno(workdir):
    """data/hmdp.liver.exprs.1 with entries HMDP_MISSING_Y written as 'NA'."""
    fn = os.path.join(str(workdir), "hmdp.exprs.1.NA")
    with open(HMDP_Y) as f:
        v = f.read().split()
    for i in HMDP_MISSING_Y:
        v[i] = "NA"
    with open(fn, "w") as f:
        f.write(" ".join(v) + "\n")
    return fn


def check_no_comment_header(args):
    """Refuse to feed a '#'-headed text file to a CLI (see
    write_headerless_kinship for why that broke the old --kfile)."""
    for a in args:
        if os.path.isfile(a) and not a.endswith((".bed", ".gz")):
            with open(a, "rb") as f:
                first = f.read(1)
            assert first != b"#", (
                "%s starts with a # header; the CLI readers "
                'do not accept "#" lines -- write a header-less copy' % a
            )


def run_cli(args, cwd, timeout=CLI_TIMEOUT, expect_rc=0):
    """Run `python -m pylmm <args>` with the current interpreter in cwd
    (normally the test's tmp_path); fail if the exit code is not
    `expect_rc` or after `timeout` seconds.

    `args` starts with the subcommand (`gwas`, `kinship`).  Returns the
    merged stdout+stderr (the -v log).  The timeout is a threading.Timer
    watchdog."""
    args = [str(a) for a in args]
    check_no_comment_header(args)
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "pylmm", *args]
    p = subprocess.Popen(
        cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    killed = []

    def kill():
        killed.append(True)
        p.kill()

    timer = threading.Timer(timeout, kill)
    timer.start()
    try:
        output = p.communicate()[0]
    finally:
        timer.cancel()
    if not isinstance(output, str):
        output = output.decode("utf-8", "replace")
    assert not killed, "command timed out after %ss: %s\n%s" % (timeout, " ".join(cmd), output)
    assert p.returncode == expect_rc, "command exited %d, expected %d: %s\n%s" % (
        p.returncode,
        expect_rc,
        " ".join(cmd),
        output,
    )
    return output


_HERIT_RE = re.compile(r"heritability=(\S+), sigma=(\S+)")


def null_fit_from_log(output):
    """(h, sigma) of the null fit as logged (%0.3f) by `pylmm gwas -v`."""
    m = _HERIT_RE.findall(output)
    assert len(m) == 1, "expected one heritability line in the -v log, got %r\n%s" % (m, output)
    return float(m[0][0]), float(m[0][1])
