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
from __future__ import print_function

import os
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
BED_BLOCK = 100            # leading K[:100,:100] block stored for 1219x1219 K
HMDP_MISSING_Y = [5, 23, 60]  # phenotype entries set to NaN for missing-Y fit
HMDP_COV_SNP = 10          # SNP (0-based) used as covariate in covariate fit

OUT = None


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


def run_cli(args, outfile):
    """Run one of the scripts/ CLIs with the current interpreter."""
    env = dict(os.environ)
    env['PYTHONPATH'] = REPO + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [sys.executable] + args + [outfile]
    log('running: %s' % ' '.join(cmd))
    begin = time.time()
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    output = p.communicate()[0]
    if p.returncode != 0:
        sys.stderr.write(output)
        raise RuntimeError('command failed: %s' % ' '.join(cmd))
    log('  done in %.1fs' % (time.time() - begin))
    return output


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


# --------------------------------------------------------------------------
# Golden sections
# --------------------------------------------------------------------------

def big_kinship_summary(name, K, header):
    """Store a 1219x1219 kinship compactly: leading block + diag + row sums."""
    savetxt(name + '.block%d.txt' % BED_BLOCK, K[:BED_BLOCK, :BED_BLOCK],
            header + '\nleading K[:%d,:%d] block' % (BED_BLOCK, BED_BLOCK))
    savetxt(name + '.diag_rowsums.txt',
            np.column_stack([np.diag(K), K.sum(axis=1)]),
            header + '\ncolumns: diag(K), K.sum(axis=1)   (one row per individual)')


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


def golden_cli(work):
    # ---- pylmmKinship.py ------------------------------------------------
    bed_kin = write_bed_subset(work, BED_KIN_NSNPS)
    kout = os.path.join(work, 'kin_bed.txt')
    run_cli(['scripts/pylmmKinship.py', '-v', '--bfile', bed_kin], kout)
    big_kinship_summary(
        'kinship_cli_bed_snps%d' % BED_KIN_NSNPS, np.loadtxt(kout),
        'scripts/pylmmKinship.py -v --bfile <first %d SNPs of '
        'data/snps.132k.clean.noX> OUT  (default -n 1000 => 2 chunks)'
        % BED_KIN_NSNPS)

    emma = write_emma_subset(work, HMDP_NSNPS)
    kout = os.path.join(work, 'kin_emma.txt')
    run_cli(['scripts/pylmmKinship.py', '-v', '--emmaSNP', emma,
             '--emmaNumSNPs', str(HMDP_NSNPS), '-n', '300', '-e', 'x'], kout)
    savetxt('kinship_cli_emma_hmdp_snps1000.txt', np.loadtxt(kout),
            'scripts/pylmmKinship.py -v --emmaSNP <first %d lines of '
            'data/hmdp.liver.snps> --emmaNumSNPs %d -n 300 -e x OUT (94x94)'
            % (HMDP_NSNPS, HMDP_NSNPS))
    savetxt('kinship_cli_emma_hmdp_snps1000.kva.txt', np.loadtxt(kout + '.kva'),
            'eigenvalues written by the same run (-e => OUT.kva)')

    # ---- pylmmGWAS.py ---------------------------------------------------
    bed_gwas = write_bed_subset(work, BED_GWAS_NSNPS)
    common = ['scripts/pylmmGWAS.py', '-v', '--bfile', bed_gwas,
              '--kfile', README_KIN, '--phenofile', FAKE_PHENOS]
    gout = path('gwas_cli_bed_snps%d.out' % BED_GWAS_NSNPS)
    run_cli(common, gout)

    bed_small = write_bed_subset(work, BED_GWAS_SMALL_NSNPS)
    common = ['scripts/pylmmGWAS.py', '-v', '--bfile', bed_small,
              '--kfile', README_KIN, '--phenofile', FAKE_PHENOS]
    for flags, tag in ((['--REML'], 'reml'),
                       (['--refit'], 'refit'),
                       (['--REML', '--refit'], 'reml_refit'),
                       (['--removeMissingGenotypes'], 'removeMissing')):
        run_cli(common + flags,
                path('gwas_cli_bed_snps%d_%s.out' % (BED_GWAS_SMALL_NSNPS, tag)))

    # EMMA input path
    common = ['scripts/pylmmGWAS.py', '-v', '--emmaSNP', emma,
              '--emmaPHENO', os.path.join(DATA, 'hmdp.liver.exprs.1'),
              '--kfile', path('hmdp_K_npdump.txt')]
    run_cli(common, path('gwas_cli_emma_hmdp_snps%d.out' % HMDP_NSNPS))
    cov = os.path.join(work, 'hmdp.cov.emma')
    src = open(os.path.join(DATA, 'hmdp.liver.exprs'))
    src.readline()
    line2 = src.readline()  # 2nd expression trait used as a covariate
    src.close()
    open(cov, 'w').write(line2)
    run_cli(common + ['--emmaCOV', cov],
            path('gwas_cli_emma_hmdp_snps%d_cov.out' % HMDP_NSNPS))


def write_env():
    savekv('env.kv.txt',
           [('python_major', sys.version_info[0]), ('python_minor', sys.version_info[1]),
            ('python_micro', sys.version_info[2])],
           'Environment used to generate these files\n'
           'python %s\nnumpy %s\nscipy %s'
           % (sys.version.split()[0], np.__version__, scipy.__version__))


def main():
    global OUT
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
        golden_cli(work)
    finally:
        shutil.rmtree(work)
    log('all done in %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
