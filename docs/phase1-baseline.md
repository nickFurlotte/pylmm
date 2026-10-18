# Phase 1: Python 2.7 baseline

This phase sets up a reproducible Python 2.7 environment that runs the original
2015 pylmm code. It also captures golden outputs (in `tests/golden/`) for later
regression tests.

## How to run

You need Docker. The image is linux/amd64 because numpy 1.16.6 and scipy 1.2.3
have no aarch64 cp27 wheels, so it runs under emulation on Apple Silicon.

```
docker/py27/run.sh --build                  # build pylmm-py27:baseline, print versions
docker/py27/run.sh python scripts/example.py
docker/py27/run.sh python scripts/pylmmKinship.py -v --bfile data/snps.132k.clean.noX out.kin
docker/py27/run.sh python scripts/pylmmGWAS.py -v --bfile data/snps.132k.clean.noX \
    --kfile data/snps.132k.clean.noX.pylmm.kin --phenofile data/snps.132k.clean.noX.fake.phenos out.foo
docker/py27/run.sh python tests/golden/generate_golden.py
```

`run.sh` mounts the repo at `/repo` and uses it as the working directory, with
`PYTHONPATH=/repo`. The image is `python:2.7.18-slim-buster`, pinned by digest,
with numpy 1.16.6 and scipy 1.2.3 installed from binary wheels only. BLAS runs
single-threaded for reproducibility.

The same Dockerfile can build a legacy image for cross-checking:

```
docker build --platform linux/amd64 --build-arg NUMPY_VERSION=1.12.1 \
    --build-arg SCIPY_VERSION=0.19.1 -t pylmm-py27:np1.12.1 docker/py27
```

## What ran, and timings

Timings are wall clock, with linux/amd64 emulated on an arm64 Mac.

| run | time | result |
|---|---|---|
| `scripts/example.py` | 15 s (13.8 s for the 1000-SNP refit GWAS) | Runs OK. Unmodified on numpy 1.12.1: 11 s. |
| `pylmmKinship.py -v --bfile data/snps.132k.clean.noX` (98,740 SNPs × 1219) | 49 s | Reproduces the tracked `data/snps.132k.clean.noX.pylmm.kin` to within 1.2e-15. |
| README `pylmmGWAS.py` command (98,740 SNPs) | 1 min 43 s | Null fit gives heritability=0.000, sigma=0.986. 9.6 MB output, too large to store; its first 1000 rows equal `gwas_cli_bed_snps1000.out` byte for byte. |
| `pylmmKinship.py --emmaSNP` / `pylmmGWAS.py --emmaSNP --emmaPHENO [--emmaCOV]` | about 1 s each (1000 SNPs) | Runs OK. |
| `pylmmGWAS.py --REML`, `--refit`, `--removeMissingGenotypes` (50 SNPs) | 2–10 s | Runs OK. `--removeMissingGenotypes` runs one eigh per SNP with missing genotypes, about 0.5 s each. |
| `tests/golden/generate_golden.py` | about 65 s | Deterministic: two runs are byte-identical. |

The full runs are fast enough that reduced runs are not needed for timing. The
golden files still use subsets to keep their size down (1.8 MB total).

## Code changes (minimal, all forced by numpy ≥1.13 / 1.16.3)

The original code **does not run** on numpy 1.16.6. Three numpy behaviours
changed:

1. `True - bool_array` raises `TypeError` (numpy boolean subtract).
2. `ndarray == None` is now elementwise, so `if X0 == None:` raises "truth
   value … ambiguous" whenever an array is passed.
3. Since numpy 1.16.3, `np.load` defaults to `allow_pickle=False`, so loading
   the `.npdump` pickles fails.

There are 16 one-line edits. Each one keeps the semantics.

| file:line | change | reason |
|---|---|---|
| pylmm/lmm.py:54, 97, 122, 183, 186 | `True - x` → `~x` | (1) |
| pylmm/lmm.py:91, 179, 262, 317 | `X0 == None` / `X == None` → `is None` | (2) |
| pylmm/input.py:180 | `True - np.isnan(G)` → `~np.isnan(G)` | (1) |
| scripts/pylmmGWAS.py:197, 245 | `True - v` → `~v` | (1) |
| scripts/pylmmKinship.py:92 | `K == None` → `K is None` | (2): fails on the 2nd SNP chunk |
| scripts/example.py:16, 17 | `np.load(..., allow_pickle=True)` | (3) |
| scripts/example.py:31 | `True - np.isnan(X)` → `~np.isnan(X)` | (1) |

**Equivalence check.** I ran the unmodified `main` code on numpy 1.12.1 /
scipy 0.19.1, where it runs and emits only FutureWarnings. Every numeric golden
file matches the patched 1.16.6 output **bit for bit**. The CLI `.out` files
differ by ≤5e-12 relative, only because numpy 1.12 `str()` prints 12 digits.

## Warnings and oddities observed

- With numpy 1.12, there are FutureWarnings for `== None` at lmm.py:179, 262,
  317. With 1.16.6 and the patches, the runs are warning-free.
- scipy ≥0.12 has no `scipy.linalg.fblas`, so the dgemm paths in `matrixMult`
  and pylmmKinship always fall back to `np.dot`.
- README dataset:
  - The kinship is rank deficient (replicate inbred individuals): 1109 of
    1219 eigenvalues are below 1e-6 and get clamped to 1e-6. All of them are
    LAPACK round-off (|e| < 1e-12); the next eigenvalue is about 0.29, so the
    effective rank is about 110. 565 of the 1109 are negative.
  - The fake phenotype fits at the **h=0 boundary** under both REML and ML.
    Golden values for that dataset therefore exercise the boundary branch of
    `getMax`.
  - The verbose message "Cleaning %d eigen values" prints the count of
    eigenvalues below 0 (565 here), but the code clamps those below 1e-6
    (1109 here; lmm.py:209).
- 68 of the first 1000 hmdp SNPs give nan p-values. Each is monomorphic once
  missing individuals are removed.
- `pylmmGWAS.py` writes values with `str()`, so the output precision depends on
  the numpy version (12 digits on ≤1.13, shortest repr on ≥1.14). Tests must
  compare these numerically.
- `example.py:30-31` mean-imputes `snps[:,0]` **in place**, because
  `X = snps[:,0]` is a view. That changes the later `lmm.GWAS` result for
  SNP 0.
- `pylmmKinship.py -e` takes a value that is ignored. Output always goes to
  `OUT.kva` / `OUT.kve`.
- The `Dockerfile` uses `FROM --platform=linux/amd64` deliberately. Docker
  prints a lint warning about it, which is harmless.

## Python 3 conversion hotspots

- **Tabs mixed with spaces.** Python 3 raises `TabError` on input.py, lmm.py,
  misc.py, pylmmGWAS.py and pylmmKinship.py. Lines with tab indentation:
  input.py 106, lmm.py 133, misc.py 13, pylmmGWAS.py 16, pylmmKinship.py 5.
  `__init__.py` and example.py parse cleanly.
- **`dict.has_key`:** input.py:212, 275, 315.
- **Integer division:** input.py:96, `self.N / 4` becomes a float, and then
  `'c'*float` raises `TypeError`. Use `//`. No other int/int divisions were
  found; the rest use `float()`.
- **Iterator protocol:** input.py:112 `def next(self)` needs `__next__`. The
  explicit `IN.next()` call at pylmmKinship.py:81 also needs updating.
- **Bytes vs str (BED reader):**
  - input.py:104 compares `order == '\x01'` against bytes. In Python 3 this is
    always False, so every BED file gets rejected.
  - input.py:118 `struct.unpack('c'…)` + `ord` + `bin` string parsing is fragile
    and slow. Use `np.frombuffer` with bit ops.
  - input.py:106 raises `StopIteration` as an error signal outside a generator.
- **Pickles:** `data/*.npdump` are Python 2 `ndarray.dump` pickles, loaded at
  example.py:16-17 (and the commented line 22). Prefer the text inputs, or use
  `np.load(allow_pickle=True, encoding='latin1')`.
- **`import pdb`:** input.py:12, pylmmKinship.py:10, pylmmGWAS.py:9 (unused).
  There is also a commented `pdb.set_trace()` at pylmmGWAS.py:266.
- **distutils:** setup.py:2 uses `distutils`, which was removed in Python 3.12.
  The scripts have `#!/usr/bin/python` shebangs; misc.py:8 has a stray shebang
  after the header.
- **`scipy.linalg.fblas`:** lmm.py:21, 41 and pylmmKinship.py:94, 98. It is
  dead code on modern scipy; replace with `@` / `np.dot`.
- **gzip kinship:** pylmmGWAS.py:165-167, 184-186. `gzip.open(…,'r')` returns
  bytes, which then go to `np.fromstring` (deprecated). pylmmGWAS.py:170 and 188
  pass a text-mode file object to `np.fromfile`.
- **`h == None`:** remains at lmm.py:350 (harmless for scalars; use `is None`).
  input.py:245 `self.indivs == None` compares a list (OK).
- **Bare `except:`:** input.py:138, pylmmGWAS.py:131.
- **Mutable default arguments:** `Kva=[], Kve=[]` at lmm.py:71, 169.
- **`range()` objects** used as data: input.py:23 (`self.indivs = range(...)`)
  and pylmmKinship.py:89 (`W[:,range(0,j)]`). These work, but list semantics
  change.
- **`x == 'NA' and np.nan or float(x)` idiom:** input.py:202, 290, 306. It
  works, but it is fragile.

## Possible genuine bugs (logged, NOT fixed)

1. **REML/ML mismatch in the null fit.**
   - `pylmmGWAS.py:225` calls `L.fit()`, which defaults to REML=True, even
     though `--REML` defaults to False ("default is maximum-likelihood"). The
     association at line 276 then uses the ML likelihood with a REML-fitted h.
   - In `lmm.GWAS`, the `REML` argument also never reaches `L.fit()` at
     lmm.py:110, or the `Ls.fit()` / `L.fit(X=…)` calls at lmm.py:134, 136, 145.
   - The missing-genotype branch at pylmmGWAS.py:265 does pass `REML`, so the
     two branches are inconsistent.
2. **`getMax` off by one** (lmm.py:292): `range(1, n-2)` never tests grid index
   n-2, so a local maximum at h=0.98 is missed and the code falls back to the
   endpoints.
3. **Grids exclude 1.0.**
   - lmm.py:323 uses H = [0, 0.99]; misc.py:23 uses w = [0, 0.99].
   - `fitTwo` can therefore never pick K1 alone. In the golden run the best w is
     0.99, with LL 23.433 below the K1-only REML LL of 23.501.
   - In `getMax`, if the maximum is at the upper end it returns h=0.99, not 1.
4. **Brent search is unbounded above.** lmm.py:250 guards `h < 0` only. For
   h > 1, `h*Kva + (1-h)` can go negative, giving a log of a negative number or
   nan. lmm.py:295 then quietly replaces nan with `H[i-1]`.
5. **`tstat(log=True)`** (lmm.py:370) computes `2.0 + logsf`; it should be
   `log(2) + logsf`. This path is unused.
6. **`--kfile2` is broken:** `LMM_withK2` and `optW` are undefined anywhere
   (pylmmGWAS.py:220, 227, 260), so the option raises NameError.
7. **`--eigen` cannot read what pylmmKinship writes.** pylmmGWAS.py:211-212 runs
   `np.load(<base>.Kva/.Kve)` (binary .npy, capitalised). pylmmKinship.py:110-111
   writes text `<out>.kva/.kve` with `np.savetxt`.
8. **Kinship normalisation differs between the two implementations.**
   - `lmm.calculateKinship` (lmm.py:54-60) mean-imputes first, then
     standardises by the variance of the imputed column.
   - `pylmmKinship.py`, via `input.normalizeGenotype` (input.py:182-187), uses
     the variance of the non-missing values only.
   - With missing data the two kinships differ: max |ΔK| is 0.41 on the hmdp
     1000-SNP subset and 0.02 on the BED 2000-SNP subset.
   - Both divide by the total SNP count, including monomorphic or dropped SNPs
     (lmm.py:63, pylmmKinship.py:102), which shrinks the scale.
9. **`sigma` is not the ML estimate under ML.** lmm.py:244 always uses
   Q/(N-q). The LL uses Q/N internally, but the reported sigma and the t-stat
   use N-q in both modes.
10. **Phenotype/fam misalignment.** `input.getPhenos` (input.py:211-214)
    silently skips fam individuals missing from the pheno file. The phenotype
    vector then no longer lines up with K or the genotypes, and nothing raises
    an error.
11. **All-missing SNP.** `input.normalizeGenotype` (input.py:181) returns an
    empty array for a SNP with every genotype missing, which causes a shape
    error downstream.
12. **Caller's array changed in place.** `LMM.__init__` (lmm.py:208-210) clamps
    the caller's `Kva` array in place.
13. **Inconsistent zero-variance filters in pylmmGWAS.** The missing-genotype
    branch uses `xs.var() <= 1e-6` (line 247); the complete branch uses
    `x.var() == 0` (line 269).
14. **Crash instead of a usage error** (pylmmGWAS.py:117). If there is no
    `.phenos` file and no `--emmaPHENO`, `os.path.isfile(None)` raises
    TypeError rather than printing the intended `parser.error`.
15. **Inconsistent return types.** `lmm.GWAS` returns lists normally but numpy
    arrays when `len(Y)==0` (lmm.py:107).
