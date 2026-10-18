"""
Shared helpers for the pylmm regression suite: paths, tolerances, golden-file
readers, input loaders and CLI runners.

Everything here must work unchanged on Python 2.7 (today) and Python 3
(after the Phase 3 port), so: no f-strings, no py2-only builtins.
"""
from __future__ import division, print_function

import os
import shutil
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(REPO, 'data')
GOLDEN = os.path.join(REPO, 'tests', 'golden')
SCRIPTS = os.path.join(REPO, 'scripts')

BED_BASE = os.path.join(DATA, 'snps.132k.clean.noX')
README_KIN = BED_BASE + '.pylmm.kin'
FAKE_PHENOS = BED_BASE + '.fake.phenos'
HMDP_Y = os.path.join(DATA, 'hmdp.liver.exprs.1')
HMDP_SNPS = os.path.join(DATA, 'hmdp.liver.snps')
HMDP_K_TXT = os.path.join(GOLDEN, 'hmdp_K_npdump.txt')

# Inputs used by tests/golden/generate_golden.py (see tests/golden/README.md).
HMDP_NSNPS = 1000
HMDP_MISSING_Y = [5, 23, 60]
HMDP_COV_SNP = 10
BED_KIN_NSNPS = 2000
BED_GWAS_NSNPS = 1000
BED_GWAS_SMALL_NSNPS = 50
BED_BLOCK = 100

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
# every t-stat / p-value that uses a fitted h).  Brent stops at
# xtol ~1.5e-8 relative, so ulp-level changes in the inputs move the optimum
# by ~1e-8 and propagate further.  Measured on the hmdp data: perturbing K by
# 1e-14 relative changed h by 5e-8, per-SNP t-stats by up to 5.6e-6 relative
# (for |t| ~ 3e-3, i.e. ~2e-8 absolute) and p-values by up to 8e-7 relative.
# rtol=1e-5 / atol=1e-8 covers that with a little headroom and is still
# orders of magnitude below any real behavioural change.
RTOL_FIT = 1e-5
ATOL_FIT = 1e-8

# Eigenvalue files: most entries are LAPACK round-off around zero
# (|e| < 1e-10; 1109 of 1219 for the README kinship, 50 of 94 for the EMMA
# CLI kinship).  Their sign and relative value are meaningless, so compare
# with an absolute tolerance scaled by the largest eigenvalue.
EIG_ATOL_SCALE = 1e-10

# Structural invariants.
SYM_ATOL = 1e-12          # |K - K.T|
PSD_RTOL = 1e-10          # min eigenvalue >= -PSD_RTOL * max eigenvalue
PVAL_T_RTOL = 1e-10       # p == 2*t.sf(|ts|, df), recomputed from ts


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def assert_close(actual, expected, rtol=RTOL, atol=ATOL, what=''):
    """assert_allclose that also requires nan in exactly the same places."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, \
        '%s: shape %s != expected %s' % (what, actual.shape, expected.shape)
    an, en = np.isnan(actual), np.isnan(expected)
    if not np.array_equal(an, en):
        bad = np.where(an != en)
        raise AssertionError('%s: nan positions differ (%d mismatches, first at %s)'
                             % (what, len(bad[0]),
                                tuple(int(b[0]) for b in bad)))
    np.testing.assert_allclose(actual[~an], expected[~en], rtol=rtol, atol=atol,
                               err_msg=what)


def assert_pvalues_valid(ps, what=''):
    ps = np.asarray(ps, dtype=np.float64)
    ok = ~np.isnan(ps)
    assert np.all((ps[ok] >= 0.0) & (ps[ok] <= 1.0)), \
        '%s: p-values outside [0, 1]' % what


def assert_p_matches_t(ts, ps, df, what=''):
    """Two-sided t-test p-value must equal 2*sf(|t|, df); nan together."""
    from scipy import stats
    ts = np.asarray(ts, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    assert np.array_equal(np.isnan(ts), np.isnan(ps)), \
        '%s: nan positions differ between t-stats and p-values' % what
    ok = ~np.isnan(ts)
    df = np.broadcast_to(np.asarray(df, dtype=np.float64), ts.shape)
    expected = 2.0 * stats.t.sf(np.abs(ts[ok]), df[ok])
    np.testing.assert_allclose(ps[ok], expected, rtol=PVAL_T_RTOL, atol=1e-300,
                               err_msg=what)


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
            if not line or line.startswith('#'):
                continue
            k, v = line.split()
            out[k] = float(v)
    finally:
        f.close()
    return out


def load_gwas_out(fn):
    """Parse a scripts/pylmmGWAS.py output file numerically.

    Returns (snp_ids, values) where values is an (nsnps, 4) float array with
    columns BETA, BETA_SD, F_STAT, P_VALUE ('nan' parses to nan).
    """
    f = open(fn)
    try:
        header = f.readline().rstrip('\n').split('\t')
        assert header == ['SNP_ID', 'BETA', 'BETA_SD', 'F_STAT', 'P_VALUE'], header
        ids, rows = [], []
        for line in f:
            parts = line.rstrip('\n').split('\t')
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
    return np.load(fn, allow_pickle=True, encoding='latin1')


def load_hmdp_snps(nsnps=HMDP_NSNPS):
    """First nsnps SNPs of data/hmdp.liver.snps as individuals x SNPs."""
    return np.genfromtxt(HMDP_SNPS, max_rows=nsnps).T.copy()


def load_readme_kinship(n):
    K = np.fromfile(README_KIN, sep=' ')
    return K.reshape((n, n))


def read_bed_snps(nsnps, norm):
    from pylmm import input as pinput
    IN = pinput.plink(BED_BASE, type='b', normGenotype=norm)
    IN.getSNPIterator()
    cols = []
    for _ in range(nsnps):
        g, _sid = next(IN)
        cols.append(g)
    return np.array(cols).T.copy()


# ---------------------------------------------------------------------------
# Subset inputs + CLI runner
# ---------------------------------------------------------------------------

def write_bed_subset(workdir, nsnps):
    """PLINK BED/BIM/FAM with the first nsnps SNPs (byte-identical to what
    generate_golden.py builds)."""
    workdir = str(workdir)
    base = os.path.join(workdir, 'bed%d' % nsnps)
    with open(BED_BASE + '.fam') as f:
        nind = sum(1 for _ in f)
    nbytes = nind // 4 + (1 if nind % 4 else 0)
    with open(BED_BASE + '.bed', 'rb') as src:
        data = src.read(3 + nsnps * nbytes)
    with open(base + '.bed', 'wb') as dst:
        dst.write(data)
    with open(BED_BASE + '.bim', 'rb') as src:
        lines = [src.readline() for _ in range(nsnps)]
    with open(base + '.bim', 'wb') as dst:
        dst.write(b''.join(lines))
    shutil.copy(BED_BASE + '.fam', base + '.fam')
    return base


def write_emma_subset(workdir, nsnps=HMDP_NSNPS):
    fn = os.path.join(str(workdir), 'hmdp.snps%d.emma' % nsnps)
    with open(HMDP_SNPS, 'rb') as src:
        lines = [src.readline() for _ in range(nsnps)]
    with open(fn, 'wb') as dst:
        dst.write(b''.join(lines))
    return fn


def write_emma_cov(workdir):
    """Row 2 (0-based 1) of data/hmdp.liver.exprs, the --emmaCOV input."""
    fn = os.path.join(str(workdir), 'hmdp.cov.emma')
    with open(os.path.join(DATA, 'hmdp.liver.exprs'), 'rb') as src:
        src.readline()
        line = src.readline()
    with open(fn, 'wb') as dst:
        dst.write(line)
    return fn


def run_script(script, args, cwd=REPO):
    """Run scripts/<script> with the current interpreter; fail on non-zero exit."""
    env = dict(os.environ)
    env['PYTHONPATH'] = REPO + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [sys.executable, os.path.join(SCRIPTS, script)] + [str(a) for a in args]
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    output = p.communicate()[0]
    if not isinstance(output, str):
        output = output.decode('utf-8', 'replace')
    assert p.returncode == 0, 'command failed (%d): %s\n%s' % (
        p.returncode, ' '.join(cmd), output)
    return output
