#!/usr/bin/env python
"""
Generate the Phase 1 "golden" reference outputs for pylmm.

Runs the ORIGINAL (Python 2.7) pylmm code on the example data in data/ and
writes small, pickle-free reference files that later phases (Python 3
conversion) are regression-tested against.  See README.md in this directory
for a description of every file.

Run from the repository root inside the py27 baseline container:

    docker/py27/run.sh python tests/golden/generate_golden.py [OUTDIR]

OUTDIR defaults to tests/golden.  The script is deterministic: no random
numbers are used, and BLAS is pinned to one thread by the Docker image.

Output formats (all loadable from Python 3 without pickle):
  *.txt      numpy.savetxt, '%.17g' (exact float64 round trip); '#' header
             lines name the columns.  Load with numpy.loadtxt.
  *.kv.txt   one "key value" pair per line ('%.17g' values), '#' comments.
  *.out      raw output files of scripts/pylmmGWAS.py (tab separated; values
             written with str(); for numpy>=1.14 float64 that is the shortest
             round-trip repr, 'nan' for missing).
"""

import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

from pylmm import lmm  # noqa: E402
from pylmm import input as pinput  # noqa: E402
from pylmm import misc  # noqa: E402

DATA = os.path.join(REPO, 'data')
BED_BASE = os.path.join(DATA, 'snps.132k.clean.noX')
README_KIN = BED_BASE + '.pylmm.kin'
FAKE_PHENOS = BED_BASE + '.fake.phenos'

FMT = '%.17g'

# Subset sizes (documented in README.md)
HMDP_NSNPS = 1000          # first 1000 SNPs of data/hmdp.liver.snps
BED_KIN_NSNPS = 2000       # first 2000 SNPs of the BED file for kinship
BED_GWAS_NSNPS = 1000      # first 1000 SNPs of the BED file for CLI GWAS
BED_GWAS_SMALL_NSNPS = 50  # refit / REML / removeMissingGenotypes CLI variants
BED_BLOCK = 100            # leading/trailing 100x100 blocks stored for 1219x1219 K
HMDP_MISSING_Y = [5, 23, 60]  # phenotype entries set to NaN for missing-Y fit
HMDP_COV_SNP = 10          # SNP (0-based) used as covariate in covariate fit
HMDP_GZ_NSNPS = 100        # EMMA CLI with a gzipped --kfile / NA phenotypes
BED_MISSING_PHENO_NA = [3, 100, 500]   # fake.phenos entries written as 'NA'
BED_MISSING_PHENO_M9 = [7, 800]        # ... and as '-9'

# The goldens are only valid for the pinned baseline environment
# (docker/py27/Dockerfile).  Fail loudly if a stale or different image is used;
# set PYLMM_GOLDEN_ANY_VERSION=1 to override for cross-checks.
EXPECTED_VERSIONS = {'numpy': '1.16.6', 'scipy': '1.2.3'}

OUT = None


def check_versions():
    # The goldens come from the Phase 1 code under py2.7.  The current code
    # (Phase 5.2: REML refits, ML sigma^2=Q/n) would write different refit/ML
    # columns, so refuse to run under py3 even with PYLMM_GOLDEN_ANY_VERSION.
    if sys.version_info[0] != 2:
        raise SystemExit('generate_golden.py: this reproduces the Phase 1 goldens and needs the Phase 1 '
                         'code under Python 2.7 (docker/py27); the current code gives different '
                         'refit/ML columns. See tests/golden/README.md.')
    found = {'numpy': np.__version__, 'scipy': scipy.__version__}
    if found == EXPECTED_VERSIONS:
        return
    msg = ('expected numpy %(numpy)s / scipy %(scipy)s' % EXPECTED_VERSIONS +
           ', found numpy %(numpy)s / scipy %(scipy)s' % found)
    if os.environ.get('PYLMM_GOLDEN_ANY_VERSION') == '1':
        log('WARNING: %s (PYLMM_GOLDEN_ANY_VERSION=1)' % msg)
        return
    raise SystemExit('generate_golden.py: %s. Rebuild the image with '
                     '`docker/py27/run.sh --build`, or set '
                     'PYLMM_GOLDEN_ANY_VERSION=1 for a cross-check.' % msg)


def log(msg):
    sys.stderr.write('[golden] %s\n' % msg)


def path(name):
    return os.path.join(OUT, name)


def savetxt(name, arr, header):
    arr = np.asarray(arr, dtype=np.float64)
    np.savetxt(path(name), arr, fmt=FMT, header=header)
    log('wrote %s %s' % (name, arr.shape))


def savekv(name, items, header):
    f = open(path(name), 'w')
    for line in header.splitlines():
        f.write('# %s\n' % line)
    for k, v in items:
        f.write('%s %s\n' % (k, FMT % float(v)))
    f.close()
    log('wrote %s (%d keys)' % (name, len(items)))


def check_no_comment_header(args):
    """pylmmGWAS.py reads --kfile with np.fromfile(sep=' '), which stops at the
    first '#' and then zero-fills K.resize(); every text input the CLIs read
    must therefore be header-less.  Refuse to run if one is not."""
    for a in args:
        if os.path.isfile(a) and not a.endswith(('.bed', '.gz', '.py')):
            f = open(a, 'rb')
            first = f.read(1)
            f.close()
            if first == b'#':
                raise RuntimeError('CLI input %s starts with a # header' % a)


def run_cli(args, outfile, cwd):
    """Run one of the scripts/ CLIs with the current interpreter.

    Returns the merged stdout+stderr text (the -v log)."""
    check_no_comment_header(args)
    env = dict(os.environ)
    env['PYTHONPATH'] = REPO + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [sys.executable] + args + [outfile]
    log('running: %s' % ' '.join(cmd))
    begin = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    output = p.communicate()[0]
    if p.returncode != 0:
        sys.stderr.write(output)
        raise RuntimeError('command failed: %s' % ' '.join(cmd))
    log('  done in %.1fs' % (time.time() - begin))
    return output


HERIT_RE = re.compile(r'heritability=(\S+), sigma=(\S+)')


def null_fit_from_log(output):
    """(h, sigma) as printed (%0.3f) by pylmmGWAS.py -v for the null fit."""
    m = HERIT_RE.findall(output)
    assert len(m) == 1, 'expected one heritability line, got %r' % (m,)
    return float(m[0][0]), float(m[0][1])


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load_hmdp():
    # data/*.npdump are Python 2 pickles (ndarray.dump); fine to load here.
    K = np.load(os.path.join(DATA, 'hmdp.liver.K.npdump'), allow_pickle=True)
    Y = np.genfromtxt(os.path.join(DATA, 'hmdp.liver.exprs.1'))
    # The text SNP file is the source of truth for Python 3 (no pickle).
    # Rows are SNPs, columns individuals; 'NA' -> nan.
    snps = np.genfromtxt(os.path.join(DATA, 'hmdp.liver.snps'),
                         max_rows=HMDP_NSNPS).T.copy()  # n x m
    snps_pickle = np.load(os.path.join(DATA, 'hmdp.liver.snps.1000.npdump'),
                          allow_pickle=True).T
    assert snps.shape == snps_pickle.shape
    assert np.array_equal(np.isnan(snps), np.isnan(snps_pickle))
    assert np.array_equal(np.nan_to_num(snps), np.nan_to_num(snps_pickle))
    return Y, K, snps


def read_bed_snps(nsnps, norm):
    IN = pinput.plink(BED_BASE, type='b', normGenotype=norm)
    IN.getSNPIterator()
    cols = []
    ids = []
    for _ in range(nsnps):
        g, sid = IN.next()
        cols.append(g)
        ids.append(sid)
    return np.array(cols).T.copy(), ids


def write_bed_subset(workdir, nsnps):
    """Write a PLINK BED/BIM/FAM with the first nsnps SNPs (all individuals)."""
    base = os.path.join(workdir, 'bed%d' % nsnps)
    nind = sum(1 for _ in open(BED_BASE + '.fam'))
    nbytes = nind // 4 + (1 if nind % 4 else 0)
    src = open(BED_BASE + '.bed', 'rb')
    data = src.read(3 + nsnps * nbytes)
    src.close()
    dst = open(base + '.bed', 'wb')
    dst.write(data)
    dst.close()
    src = open(BED_BASE + '.bim')
    dst = open(base + '.bim', 'w')
    for _ in range(nsnps):
        dst.write(src.readline())
    src.close()
    dst.close()
    shutil.copy(BED_BASE + '.fam', base + '.fam')
    return base


def write_emma_subset(workdir, nsnps):
    fn = os.path.join(workdir, 'hmdp.snps%d.emma' % nsnps)
    src = open(os.path.join(DATA, 'hmdp.liver.snps'))
    dst = open(fn, 'w')
    for _ in range(nsnps):
        dst.write(src.readline())
    src.close()
    dst.close()
    return fn


def write_headerless_kinship(fn, K):
    """K as plain whitespace text with no '#' header (what --kfile needs)."""
    np.savetxt(fn, K, fmt=FMT)
    back = np.fromfile(open(fn, 'r'), sep=' ')
    assert back.size == K.size and np.array_equal(back.reshape(K.shape), K)
    return fn


TPED_CODES = {0: '1 1', 1: '0 0', 2: '1 2', 3: '2 2'}  # PLINK 2-bit codes


def write_tped_subset(workdir, nsnps):
    """Transposed PLINK (.tped/.tfam/.map) with the first nsnps BED SNPs.

    Genotypes are decoded straight from the .bed bytes (00 hom A1 -> '1 1',
    01 missing -> '0 0', 10 het -> '1 2', 11 hom A2 -> '2 2'), which is the
    allele coding input.getGenos_tped understands."""
    base = os.path.join(workdir, 'tped%d' % nsnps)
    nind = sum(1 for _ in open(BED_BASE + '.fam'))
    nbytes = nind // 4 + (1 if nind % 4 else 0)
    src = open(BED_BASE + '.bed', 'rb')
    src.read(3)
    bim = open(BED_BASE + '.bim')
    tped = open(base + '.tped', 'w')
    mapf = open(base + '.map', 'w')
    for _ in range(nsnps):
        raw = bytearray(src.read(nbytes))
        v = bim.readline().split()
        codes = [(raw[i // 4] >> (2 * (i % 4))) & 3 for i in range(nind)]
        tped.write(' '.join(v[:4] + [TPED_CODES[c] for c in codes]) + '\n')
        mapf.write(' '.join(v[:4]) + '\n')
    for f in (src, bim, tped, mapf):
        f.close()
    shutil.copy(BED_BASE + '.fam', base + '.tfam')
    return base


def write_covfile(workdir):
    """PLINK covariate file (FID IID c1 c2) for the BED individuals, rows in
    REVERSE fam order so getCovariates() has to re-order them by key."""
    fn = os.path.join(workdir, 'bed.cov')
    fam = [l.split()[:2] for l in open(BED_BASE + '.fam')]
    lines = []
    for i, (fid, iid) in enumerate(fam):
        lines.append('%s %s %s %d\n' % (fid, iid, FMT % np.cos(i), i % 3))
    f = open(fn, 'w')
    f.write(''.join(lines[::-1]))
    f.close()
    return fn


def write_missing_phenofile(workdir):
    """fake.phenos with some entries set to 'NA' / '-9', rows in reverse
    order so getPhenos() has to re-order them by key."""
    fn = os.path.join(workdir, 'bed.missing.phenos')
    lines = open(FAKE_PHENOS).read().splitlines()
    out = []
    for i, line in enumerate(lines):
        v = line.split()
        if i in BED_MISSING_PHENO_NA:
            v[2] = 'NA'
        elif i in BED_MISSING_PHENO_M9:
            v[2] = '-9'
        out.append(' '.join(v) + '\n')
    f = open(fn, 'w')
    f.write(''.join(out[::-1]))
    f.close()
    return fn


def write_emma_missing_pheno(workdir):
    """data/hmdp.liver.exprs.1 with entries HMDP_MISSING_Y written as 'NA'."""
    fn = os.path.join(workdir, 'hmdp.exprs.1.NA')
    v = open(os.path.join(DATA, 'hmdp.liver.exprs.1')).read().split()
    for i in HMDP_MISSING_Y:
        v[i] = 'NA'
    f = open(fn, 'w')
    f.write(' '.join(v) + '\n')
    f.close()
    return fn


# --------------------------------------------------------------------------
# Golden sections
# --------------------------------------------------------------------------

def big_kinship_summary(name, K, header):
    """Store a 1219x1219 kinship compactly: leading and trailing blocks, the
    diagonal and K.dot(v) for a fixed non-constant v.  (Row sums would be
    useless: the genotype columns are centred, so K.sum(1) is ~0 round-off.)"""
    savetxt(name + '.block%d.txt' % BED_BLOCK, K[:BED_BLOCK, :BED_BLOCK],
            header + '\nleading K[:%d,:%d] block' % (BED_BLOCK, BED_BLOCK))
    savetxt(name + '.tail%d.txt' % BED_BLOCK, K[-BED_BLOCK:, -BED_BLOCK:],
            header + '\ntrailing K[-%d:,-%d:] block' % (BED_BLOCK, BED_BLOCK))
    v = np.cos(np.arange(K.shape[0]))
    savetxt(name + '.diag_Kv.txt', np.column_stack([np.diag(K), K.dot(v)]),
            header + '\ncolumns: diag(K), K.dot(v) with v = cos(arange(n))   '
            '(one row per individual)')


def golden_kinship(Y, K, snps):
    savetxt('hmdp_K_npdump.txt', K,
            'data/hmdp.liver.K.npdump exported as text (94x94).\n'
            'Equals lmm.calculateKinship(all 132285 hmdp SNPs) to ~1e-14.\n'
            'Used as input (--kfile) for the EMMA CLI GWAS golden run.')
    savetxt('kinship_hmdp_snps1000.txt', lmm.calculateKinship(snps.copy()),
            'lmm.calculateKinship(W) with W = first %d SNPs of data/hmdp.liver.snps '
            '(individuals x SNPs, NA->nan), center=False' % HMDP_NSNPS)
    savetxt('kinship_hmdp_snps1000_centered.txt',
            lmm.calculateKinship(snps.copy(), center=True),
            'same as kinship_hmdp_snps1000.txt but center=True')

    W, _ = read_bed_snps(BED_KIN_NSNPS, norm=False)
    log('BED kinship input: %s, %d missing genotypes' % (W.shape, np.isnan(W).sum()))
    Kb = lmm.calculateKinship(W)
    big_kinship_summary(
        'kinship_bed_snps%d_calc' % BED_KIN_NSNPS, Kb,
        'lmm.calculateKinship(W), W = first %d SNPs of data/snps.132k.clean.noX.bed '
        'read with input.plink(type="b", normGenotype=False) (0/0.5/1, nan missing)'
        % BED_KIN_NSNPS)


def fit_items(L, res, prefix=''):
    hmax, beta, sigma, LL = res
    items = [('h', L.optH), ('sigma', L.optSigma), ('LL', L.optLL)]
    for i, b in enumerate(np.asarray(beta).ravel()):
        items.append(('beta_%d' % i, b))
    mn, vx = L.meanAndVar()
    items += [('h_posterior_mean', mn), ('h_posterior_var', vx), ('N', L.N)]
    return [(prefix + k, v) for k, v in items]


def golden_null_fits(Y, K, snps):
    items = []
    grid = []
    for reml in (True, False):
        tag = 'reml' if reml else 'ml'
        L = lmm.LMM(Y, K)
        res = L.fit(REML=reml)
        items += fit_items(L, res, 'hmdp_%s.' % tag)
        if not grid:
            grid.append(L.H)
        grid.append(L.LLs)

    # Missing phenotype values (removed inside LMM.__init__)
    Ym = Y.copy()
    Ym[HMDP_MISSING_Y] = np.nan
    L = lmm.LMM(Ym, K)
    items += fit_items(L, L.fit(REML=True), 'hmdp_missingY_reml.')

    # Extra covariate: intercept + mean-imputed SNP HMDP_COV_SNP
    c = snps[:, HMDP_COV_SNP].copy()
    c[np.isnan(c)] = c[~np.isnan(c)].mean()
    X0 = np.column_stack([np.ones(len(Y)), c])
    L = lmm.LMM(Y, K, X0=X0)
    items += fit_items(L, L.fit(REML=True), 'hmdp_cov_reml.')

    # BED dataset: fake phenotypes + README kinship (1219 individuals)
    Kr = np.fromfile(open(README_KIN, 'r'), sep=' ')
    IN = pinput.plink(BED_BASE, type='b', phenoFile=FAKE_PHENOS)
    n = len(IN.indivs)
    Kr.resize((n, n))
    Yb = IN.phenos[:, 0]
    Kva, Kve = lmm.linalg.eigh(Kr)
    savetxt('bed_readme_kin_eigenvalues.txt', Kva,
            'scipy.linalg.eigh(K) eigenvalues (ascending, before LMM clamps <1e-6) '
            'of data/snps.132k.clean.noX.pylmm.kin (1219x1219)')
    for reml in (True, False):
        tag = 'reml' if reml else 'ml'
        L = lmm.LMM(Yb, Kr, Kva.copy(), Kve)
        items += fit_items(L, L.fit(REML=reml), 'bed_%s.' % tag)

    savekv('null_fits.kv.txt', items,
           'LMM.fit() null-model results (ngrids=100). Keys are <config>.<field>.\n'
           'Fields: h=optH (heritability), sigma=optSigma, LL=optLL, beta_i=optBeta,\n'
           'h_posterior_mean/var=LMM.meanAndVar(), N=individuals used.\n'
           'hmdp_*: Y=data/hmdp.liver.exprs.1, K=data/hmdp.liver.K.npdump.\n'
           'hmdp_missingY_reml: Y[%s]=nan.\n'
           'hmdp_cov_reml: X0=[1, mean-imputed hmdp SNP %d (0-based)].\n'
           'bed_*: Y=column 0 of data/snps.132k.clean.noX.fake.phenos, '
           'K=data/snps.132k.clean.noX.pylmm.kin.'
           % (HMDP_MISSING_Y, HMDP_COV_SNP))
    savetxt('null_fit_hmdp_LLgrid.txt', np.column_stack(grid),
            'hmdp null model log-likelihood grid (LMM.LLs after fit())\n'
            'columns: h, LL_REML, LL_ML')


def golden_association(Y, K, snps):
    m = snps.shape[1]
    nmiss = np.isnan(snps).sum(axis=0)
    cols = [np.arange(m), nmiss]
    names = ['snp_index', 'n_missing']
    for reml, refit in ((True, False), (True, True), (False, False)):
        tag = '%s_%s' % ('reml' if reml else 'ml', 'refit' if refit else 'norefit')
        begin = time.time()
        TS, PS = lmm.GWAS(Y, snps.copy(), K, REML=reml, refit=refit)
        log('lmm.GWAS %s on %d SNPs: %.1fs' % (tag, m, time.time() - begin))
        cols += [TS, PS]
        names += ['ts_' + tag, 'ps_' + tag]
    savetxt('assoc_hmdp_GWAS.txt', np.column_stack(cols),
            'lmm.GWAS(Y, X, K, REML=..., refit=...) on hmdp data.\n'
            'Y=data/hmdp.liver.exprs.1, K=data/hmdp.liver.K.npdump, X=first %d SNPs '
            'of data/hmdp.liver.snps (missing genotypes NOT imputed: GWAS() drops '
            'those individuals per SNP).\ncolumns: %s' % (HMDP_NSNPS, ' '.join(names)))

    # Mean-imputed SNPs (as in scripts/example.py), fixed REML null fit, with beta
    L = lmm.LMM(Y, K)
    L.fit()
    rows = []
    for i in range(m):
        x = snps[:, i].copy()
        x[np.isnan(x)] = x[~np.isnan(x)].mean()
        x = x.reshape(len(x), 1)
        if x.var() == 0:
            rows.append([i, np.nan, np.nan, np.nan, np.nan])
            continue
        ts, ps, beta, bvar = L.association(x, REML=True, returnBeta=True)
        rows.append([i, ts, ps, beta, bvar])
    savetxt('assoc_hmdp_imputed.txt', rows,
            'LMM(Y,K).fit() (REML) then L.association(x, REML=True, returnBeta=True) '
            'for each of the first %d hmdp SNPs, missing genotypes mean-imputed.\n'
            'columns: snp_index ts ps beta beta_var(=betaVAR*sigma)' % HMDP_NSNPS)


def golden_example_py():
    """Re-trace scripts/example.py step by step and record its results."""
    Y = np.genfromtxt(os.path.join(DATA, 'hmdp.liver.exprs.1'))
    K = np.load(os.path.join(DATA, 'hmdp.liver.K.npdump'), allow_pickle=True)
    snps = np.load(os.path.join(DATA, 'hmdp.liver.snps.1000.npdump'),
                   allow_pickle=True).T
    L = lmm.LMM(Y, K)
    L.fit()
    items = [('null_h', L.optH), ('null_sigma', L.optSigma), ('null_LL', L.optLL)]
    X = snps[:, 0]
    X[np.isnan(X)] = X[~np.isnan(X)].mean()  # NB: mutates snps[:,0] in place
    X = X.reshape(len(X), 1)
    ts, ps = L.association(X)
    items += [('snp0_ts', ts), ('snp0_ps', ps)]
    L.fit(X=X)
    items += [('snp0_refit_h', L.optH)]
    ts, ps = L.association(X)
    items += [('snp0_refit_ts', ts), ('snp0_refit_ps', ps)]
    TS, PS = lmm.GWAS(Y, snps, K, REML=True, refit=True)
    TS = np.asarray(TS)
    PS = np.asarray(PS)
    items += [('gwas_n', len(PS)), ('gwas_n_nan', np.isnan(PS).sum()),
              ('gwas_ts_0', TS[0]), ('gwas_ps_0', PS[0]),
              ('gwas_ps_min', np.nanmin(PS)), ('gwas_ps_argmin', np.nanargmin(PS)),
              ('gwas_sum_log10_ps', np.nansum(np.log10(PS)))]
    savekv('example_py.kv.txt', items,
           'Values computed by the exact sequence of scripts/example.py.\n'
           'Note example.py mean-imputes snps[:,0] IN PLACE before calling\n'
           'lmm.GWAS(..., REML=True, refit=True), so SNP 0 differs from\n'
           'assoc_hmdp_GWAS.txt column ts_reml_refit; all other SNPs match it.')


def golden_misc(Y, K, snps):
    K2 = lmm.calculateKinship(snps.copy())
    begin = time.time()
    h1, h2, e, beta, sigma, LL = misc.fitTwo(Y, K, K2)
    log('misc.fitTwo: %.1fs' % (time.time() - begin))
    items = [('h1', h1), ('h2', h2), ('e', e), ('sigma', np.asarray(sigma).sum()),
             ('LL', LL)]
    for i, b in enumerate(np.asarray(beta).ravel()):
        items.append(('beta_%d' % i, b))
    savekv('misc_fitTwo.kv.txt', items,
           'misc.fitTwo(Y, K1, K2, wgrids=100): Y=data/hmdp.liver.exprs.1,\n'
           'K1=data/hmdp.liver.K.npdump, K2=kinship_hmdp_snps1000.txt '
           '(calculateKinship of first %d hmdp SNPs).' % HMDP_NSNPS)


def golden_cli(work, K):
    kinship_py = os.path.join(REPO, 'scripts', 'pylmmKinship.py')
    gwas_py = os.path.join(REPO, 'scripts', 'pylmmGWAS.py')
    exprs1 = os.path.join(DATA, 'hmdp.liver.exprs.1')
    null_items = []

    def gwas(args, name):
        """Run pylmmGWAS.py into the golden file `name`; record the -v null fit."""
        output = run_cli([gwas_py] + args, path(name), cwd=work)
        if '--refit' not in args:
            hh, ss = null_fit_from_log(output)
            key = name[len('gwas_cli_'):-len('.out')]
            null_items.extend([(key + '.h', hh), (key + '.sigma', ss)])
        return output

    # ---- pylmmKinship.py ------------------------------------------------
    bed_kin = write_bed_subset(work, BED_KIN_NSNPS)
    kout = os.path.join(work, 'kin_bed.txt')
    run_cli([kinship_py, '-v', '--bfile', bed_kin], kout, cwd=work)
    big_kinship_summary(
        'kinship_cli_bed_snps%d' % BED_KIN_NSNPS, np.loadtxt(kout),
        'scripts/pylmmKinship.py -v --bfile <first %d SNPs of '
        'data/snps.132k.clean.noX> OUT  (default -n 1000 => 2 chunks)'
        % BED_KIN_NSNPS)

    emma = write_emma_subset(work, HMDP_NSNPS)
    kout = os.path.join(work, 'kin_emma.txt')
    run_cli([kinship_py, '-v', '--emmaSNP', emma,
             '--emmaNumSNPs', str(HMDP_NSNPS), '-n', '300', '-e', 'x'], kout,
            cwd=work)
    savetxt('kinship_cli_emma_hmdp_snps1000.txt', np.loadtxt(kout),
            'scripts/pylmmKinship.py -v --emmaSNP <first %d lines of '
            'data/hmdp.liver.snps> --emmaNumSNPs %d -n 300 -e x OUT (94x94)'
            % (HMDP_NSNPS, HMDP_NSNPS))
    savetxt('kinship_cli_emma_hmdp_snps1000.kva.txt', np.loadtxt(kout + '.kva'),
            'eigenvalues written by the same run (-e => OUT.kva)')

    # ---- pylmmGWAS.py ---------------------------------------------------
    bed_gwas = write_bed_subset(work, BED_GWAS_NSNPS)
    common = ['-v', '--bfile', bed_gwas,
              '--kfile', README_KIN, '--phenofile', FAKE_PHENOS]
    gwas(common, 'gwas_cli_bed_snps%d.out' % BED_GWAS_NSNPS)

    bed_small = write_bed_subset(work, BED_GWAS_SMALL_NSNPS)
    common = ['-v', '--bfile', bed_small,
              '--kfile', README_KIN, '--phenofile', FAKE_PHENOS]
    for flags, tag in ((['--REML'], 'reml'),
                       (['--refit'], 'refit'),
                       (['--REML', '--refit'], 'reml_refit'),
                       (['--removeMissingGenotypes'], 'removeMissing')):
        gwas(common + flags,
             'gwas_cli_bed_snps%d_%s.out' % (BED_GWAS_SMALL_NSNPS, tag))

    # --covfile (plink covariate file, rows in reverse order)
    gwas(common + ['--covfile', write_covfile(work)],
         'gwas_cli_bed_snps%d_covfile.out' % BED_GWAS_SMALL_NSNPS)

    # --phenofile with NA and -9 entries (CLI missing-phenotype path)
    gwas(['-v', '--bfile', bed_small, '--kfile', README_KIN,
          '--phenofile', write_missing_phenofile(work)],
         'gwas_cli_bed_snps%d_missingpheno.out' % BED_GWAS_SMALL_NSNPS)

    # --tfile: the same 50 SNPs as transposed PLINK.  Must reproduce the
    # first 50 rows of the --bfile README run exactly.
    tped = write_tped_subset(work, BED_GWAS_SMALL_NSNPS)
    W_bed, _ = read_bed_snps(BED_GWAS_SMALL_NSNPS, norm=False)
    IN = pinput.plink(tped, type='t', normGenotype=False)
    W_tped = np.array([g for g, _ in IN]).T
    assert np.array_equal(np.isnan(W_bed), np.isnan(W_tped))
    assert np.array_equal(np.nan_to_num(W_bed), np.nan_to_num(W_tped))
    tname = 'gwas_cli_tped_snps%d.out' % BED_GWAS_SMALL_NSNPS
    gwas(['-v', '--tfile', tped, '--kfile', README_KIN,
          '--phenofile', FAKE_PHENOS], tname)
    full = open(path('gwas_cli_bed_snps%d.out' % BED_GWAS_NSNPS)).readlines()
    assert open(path(tname)).readlines() == full[:BED_GWAS_SMALL_NSNPS + 1]

    # EMMA input path.  --kfile must be header-less: hmdp_K_npdump.txt has
    # '#' lines, which pylmmGWAS.py's np.fromfile reader would stop at and
    # then zero-fill K (an all-zero kinship, i.e. plain OLS).
    kfile = write_headerless_kinship(os.path.join(work, 'hmdp.K.txt'), K)
    common = ['-v', '--emmaSNP', emma, '--emmaPHENO', exprs1, '--kfile', kfile]
    gwas(common, 'gwas_cli_emma_hmdp_snps%d.out' % HMDP_NSNPS)
    cov = os.path.join(work, 'hmdp.cov.emma')
    src = open(os.path.join(DATA, 'hmdp.liver.exprs'))
    src.readline()
    line2 = src.readline()  # 2nd expression trait used as a covariate
    src.close()
    open(cov, 'w').write(line2)
    gwas(common + ['--emmaCOV', cov],
         'gwas_cli_emma_hmdp_snps%d_cov.out' % HMDP_NSNPS)

    # gzipped --kfile (the .gz branch of the reader) on the first 100 SNPs:
    # must reproduce the first 100 rows of the plain-text run exactly.
    emma_small = write_emma_subset(work, HMDP_GZ_NSNPS)
    kgz = kfile + '.gz'
    f = gzip.open(kgz, 'wb')
    f.write(open(kfile, 'rb').read())
    f.close()
    gzout = os.path.join(work, 'gz.out')
    run_cli([gwas_py, '-v', '--emmaSNP', emma_small, '--emmaPHENO', exprs1,
             '--kfile', kgz], gzout, cwd=work)
    full = open(path('gwas_cli_emma_hmdp_snps%d.out' % HMDP_NSNPS)).readlines()
    assert open(gzout).readlines() == full[:HMDP_GZ_NSNPS + 1]

    # --emmaPHENO with 'NA' entries
    gwas(['-v', '--emmaSNP', emma_small, '--emmaPHENO',
          write_emma_missing_pheno(work), '--kfile', kfile],
         'gwas_cli_emma_hmdp_snps%d_missingpheno.out' % HMDP_GZ_NSNPS)

    savekv('gwas_cli_nullfit.kv.txt', null_items,
           'Null-model fit printed by pylmmGWAS.py -v ("heritability=%0.3f,\n'
           'sigma=%0.3f") for every gwas_cli_*.out run without --refit.\n'
           'Keys are <run>.h / <run>.sigma, <run> = file name minus gwas_cli_ / .out.\n'
           'Values are rounded to 3 decimals by the CLI.')


def write_env():
    savekv('env.kv.txt',
           [('python_major', sys.version_info[0]), ('python_minor', sys.version_info[1]),
            ('python_micro', sys.version_info[2])],
           'Environment used to generate these files\n'
           'python %s\nnumpy %s\nscipy %s'
           % (sys.version.split()[0], np.__version__, scipy.__version__))


def main():
    global OUT
    check_versions()
    OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1
                          else os.path.join(REPO, 'tests', 'golden'))
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    os.chdir(REPO)
    t0 = time.time()
    write_env()
    Y, K, snps = load_hmdp()
    golden_kinship(Y, K, snps)
    golden_null_fits(Y, K, snps)
    golden_association(Y, K, snps)
    golden_example_py()
    golden_misc(Y, K, snps)
    work = tempfile.mkdtemp(prefix='pylmm_golden_')
    try:
        golden_cli(work, K)
    finally:
        shutil.rmtree(work)
    log('all done in %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
