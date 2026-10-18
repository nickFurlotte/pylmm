# Golden outputs (Phase 1 baseline)

Reference outputs from the original Python 2.7 pylmm code. Later phases (the
Python 3 port) are regression-tested against these files.

Regenerate them from the repo root. Generation is deterministic: two runs
produce byte-identical files.

```
docker/py27/run.sh python tests/golden/generate_golden.py            # writes here
docker/py27/run.sh python tests/golden/generate_golden.py /repo/tmp  # elsewhere
```

## Environment

| | version |
|---|---|
| Python | 2.7.18 (`python:2.7.18-slim-buster`, linux/amd64) |
| numpy | 1.16.6 |
| scipy | 1.2.3 (has no `scipy.linalg.fblas`, so pylmm falls back to `np.dot`) |
| BLAS | OpenBLAS bundled with the numpy/scipy wheels, `OPENBLAS_NUM_THREADS=1` |

The code is the original code plus the minimal numpy-1.16 compatibility edits
listed in `docs/phase1-baseline.md`. Cross-check: running this generator with
the **unmodified** `main` code on numpy 1.12.1 / scipy 0.19.1 gives
**bit-identical** values for every `.txt` and `.kv.txt` file. The `.out` files
match to within 5e-12 relative; they differ only because numpy 1.12 `str()`
prints 12 significant digits.

The generator refuses to run unless numpy is 1.16.6 and scipy is 1.2.3, so a
stale Docker image cannot silently produce different goldens. For a cross-check
on another version, set `PYLMM_GOLDEN_ANY_VERSION=1` and write to a scratch
directory.

## Formats (no pickles, all readable from Python 3)

- `*.txt` is `numpy.savetxt(fmt='%.17g')`, which round-trips float64
  exactly. `#` header lines describe the contents. Load with `np.loadtxt`.
- `*.kv.txt` holds one `key value` pair per line, with `#` comment lines.
- `*.out` is the raw output of `scripts/pylmmGWAS.py`: a tab-separated header
  `SNP_ID BETA BETA_SD F_STAT P_VALUE`, with values written by `str()` (shortest
  repr under numpy ≥1.14, `nan` when missing). Compare these numerically, not
  as text.

## Comparison rules

- **Eigenvalue files** (`bed_readme_kin_eigenvalues.txt`,
  `kinship_cli_emma_hmdp_snps1000.kva.txt`) are mostly LAPACK round-off:
  1109 of 1219 and 50 of 94 entries are below 1e-6, with magnitudes around
  1e-13 to 1e-16 and arbitrary sign. The raw values are kept as written, but
  those entries are not reproducible across BLAS/LAPACK builds. Compare these
  files with `atol = 1e-10 * max(eig)` (or only on entries above 1e-6), and
  never check the sign of the near-zero entries.
- **`assoc_hmdp_GWAS.txt`**: `ts_ml_norefit`/`ps_ml_norefit` are exactly equal
  to `ts_reml_norefit`/`ps_reml_norefit`. `lmm.GWAS` never passes `REML` to
  `fit()`, and the t-statistic uses N-q in both modes (known bugs 1 and 9 in
  `docs/phase1-baseline.md`), so these columns do not exercise ML. A port that
  fixes those bugs is expected to change them.

## Inputs (all from `data/`)

- **hmdp**: 94 individuals.
  - `Y` = `hmdp.liver.exprs.1`, which is row 1 of `hmdp.liver.exprs`.
  - `K` = `hmdp.liver.K.npdump`. This equals `lmm.calculateKinship` over all
    132,285 SNPs to within 8e-15.
  - SNPs = the first 1000 rows of the EMMA-format text file `hmdp.liver.snps`
    (SNPs on rows, `NA` becomes nan), transposed to individuals × SNPs. These
    are identical to `hmdp.liver.snps.1000.npdump`.
  - 465 of the 1000 SNPs have missing genotypes. 68 of them are monomorphic
    once the missing individuals are dropped, so their p-values are nan.
- **bed**: 1219 individuals, 98,740 SNPs.
  - Genotypes from `snps.132k.clean.noX.{bed,bim,fam}`.
  - `Y` = column 0 of `snps.132k.clean.noX.fake.phenos` (no missing values).
  - `K` = `snps.132k.clean.noX.pylmm.kin`. Rerunning `pylmmKinship.py` on the
    full BED reproduces this file to within 1.2e-15.
  - The kinship is rank deficient: 1109 of 1219 eigenvalues are below 1e-6
    and get clamped to 1e-6. All of them are round-off (|e| < 1e-12); the
    next eigenvalue is about 0.29, so the effective rank is about 110. Only
    565 are negative, which is the count the misleading "Cleaning %d eigen
    values" message prints.
  - The fake phenotype fits at the h=0 boundary.
  - 1550 of the first 3000 SNPs have missing genotypes.
- Synthetic CLI inputs, all written by the generator (and byte-identically by
  `tests/helpers.py`) into a temp dir:
  - TPED: the BED's first 50 SNPs as `.tped/.tfam/.map`, decoded straight
    from the `.bed` bytes (`00`→`1 1`, `01`→`0 0`, `10`→`1 2`, `11`→`2 2`).
  - Covariate file: `FID IID cos(i) i%3` for every fam individual, rows in
    reverse fam order (so `getCovariates` has to re-order by key).
  - Missing phenotypes: `fake.phenos` with rows 3, 100, 500 set to `NA` and
    rows 7, 800 set to `-9`, rows in reverse order.
  - EMMA missing phenotypes: `hmdp.liver.exprs.1` with entries 5, 23, 60 set
    to `NA`.
  - Header-less kinship: `hmdp_K_npdump.txt`'s values re-written with
    `%.17g` and **no `#` lines**, plus a gzipped copy. See the warning under
    `hmdp_K_npdump.txt` below.
- "First N SNPs of the BED" is built by copying the 3-byte header plus N × 305
  bytes of the `.bed`, the first N lines of the `.bim`, and the whole `.fam`.
  The generator does this in a temp dir.
- Note for Phase 3: `data/*.npdump` are **Python 2 pickles** (`ndarray.dump`).
  Python 3 tests should read the text equivalents instead: `hmdp_K_npdump.txt`
  here and `data/hmdp.liver.snps`. Loading the pickles in Python 3 would need
  `np.load(..., allow_pickle=True, encoding='latin1')`.

## Files

### Kinship

| file | contents |
|---|---|
| `hmdp_K_npdump.txt` | `hmdp.liver.K.npdump` exported as text (94×94). **Do not pass it to `--kfile`**: `pylmmGWAS.py` reads K with `np.fromfile(sep=' ')`, which stops at the first `#`, and `K.resize()` then zero-fills, silently giving an all-zero kinship (plain OLS). The EMMA CLI runs use a header-less copy of the same values. |
| `kinship_hmdp_snps1000.txt` | `lmm.calculateKinship(W)`, W = hmdp first 1000 SNPs (94×94). |
| `kinship_hmdp_snps1000_centered.txt` | Same, with `center=True`. |
| `kinship_bed_snps2000_calc.block100.txt` | `lmm.calculateKinship` on the BED's first 2000 SNPs, read with `input.plink(normGenotype=False)`: the leading `K[:100,:100]` block. |
| `kinship_bed_snps2000_calc.tail100.txt` | Same K: the trailing `K[-100:,-100:]` block. |
| `kinship_bed_snps2000_calc.diag_Kv.txt` | Same K: columns `diag(K)` and `K.dot(v)` with `v = cos(arange(1219))` (1219 rows). Row sums are not stored: the genotype columns are centred, so `K.sum(1)` is pure round-off. |
| `kinship_cli_bed_snps2000.{block100,tail100,diag_Kv}.txt` | `pylmmKinship.py -v --bfile <first 2000 SNPs>` (default `-n 1000`, so 2 chunks), same summaries. |
| `kinship_cli_emma_hmdp_snps1000.txt` | `pylmmKinship.py -v --emmaSNP <hmdp first 1000 SNPs> --emmaNumSNPs 1000 -n 300 -e x` (94×94, 4 chunks). |
| `kinship_cli_emma_hmdp_snps1000.kva.txt` | Eigenvalues written by `-e` in that same run. |
| `bed_readme_kin_eigenvalues.txt` | `scipy.linalg.eigh` eigenvalues of the README kinship (1219), before clamping. |

The CLI kinships do **not** equal `calculateKinship` when genotypes are
missing; see the baseline report.

### Null model

| file | contents |
|---|---|
| `null_fits.kv.txt` | `LMM.fit()` results (`h`, `sigma`, `LL`, `beta_i`, `meanAndVar()`, `N`) for these configs: `hmdp_reml`, `hmdp_ml`, `hmdp_missingY_reml` (Y[5,23,60]=nan), `hmdp_cov_reml` (X0 = [1, mean-imputed SNP 10]), `bed_reml`, `bed_ml`. |
| `null_fit_hmdp_LLgrid.txt` | The hmdp `LMM.LLs` grid. Columns: `h`, `LL_REML`, `LL_ML` (100 rows, h = 0, 0.01, …, 0.99). |

### Association

| file | contents |
|---|---|
| `assoc_hmdp_GWAS.txt` | `lmm.GWAS` over the hmdp first 1000 SNPs, with missing genotypes left in place (GWAS drops those individuals per SNP and refits). Columns: `snp_index n_missing ts_reml_norefit ps_reml_norefit ts_reml_refit ps_reml_refit ts_ml_norefit ps_ml_norefit`. |
| `assoc_hmdp_imputed.txt` | REML null fit, then `L.association(x, REML=True, returnBeta=True)` per SNP, with mean-imputed genotypes. Columns: `snp_index ts ps beta beta_var`. |
| `example_py.kv.txt` | Values from the exact sequence in `scripts/example.py`. It mean-imputes `snps[:,0]` in place, so only SNP 0 differs from `ts/ps_reml_refit` above. |
| `misc_fitTwo.kv.txt` | `misc.fitTwo(Y, K1=hmdp K, K2=kinship_hmdp_snps1000)` gives `h1 h2 e sigma LL beta_0`. |

### `pylmmGWAS.py` CLI outputs

| file | command |
|---|---|
| `gwas_cli_bed_snps1000.out` | The README command (`-v --bfile … --kfile snps.132k.clean.noX.pylmm.kin --phenofile snps.132k.clean.noX.fake.phenos`) on the BED's first 1000 SNPs. Identical to the first 1000 rows of the full 98,740-SNP README run. |
| `gwas_cli_bed_snps50_{reml,refit,reml_refit,removeMissing}.out` | The same command on the first 50 SNPs, adding `--REML`, `--refit`, `--REML --refit`, or `--removeMissingGenotypes`. |
| `gwas_cli_bed_snps50_covfile.out` | The 50-SNP command plus `--covfile <synthetic covariate file>` (X0 = [1, cos(i), i%3]). |
| `gwas_cli_bed_snps50_missingpheno.out` | The 50-SNP command with `--phenofile <fake.phenos with 3 NA and 2 -9>`; the CLI drops those 5 individuals. |
| `gwas_cli_tped_snps50.out` | `-v --tfile <TPED of the first 50 SNPs> --kfile … --phenofile …`. The generator asserts it is identical to the first 50 rows of `gwas_cli_bed_snps1000.out`. |
| `gwas_cli_emma_hmdp_snps1000.out` | `-v --emmaSNP <hmdp first 1000 SNPs> --emmaPHENO data/hmdp.liver.exprs.1 --kfile <header-less hmdp K>`. The generator also runs a gzipped `--kfile` on the first 100 SNPs and asserts it reproduces the first 100 rows exactly. |
| `gwas_cli_emma_hmdp_snps1000_cov.out` | The same, plus `--emmaCOV <row 2 of data/hmdp.liver.exprs>`. |
| `gwas_cli_emma_hmdp_snps100_missingpheno.out` | First 100 hmdp SNPs, `--emmaPHENO <exprs.1 with 3 NA>`. |
| `gwas_cli_nullfit.kv.txt` | The null fit `pylmmGWAS.py -v` prints (`heritability=%0.3f, sigma=%0.3f`) for every run above without `--refit`. Keys `<run>.h`, `<run>.sigma`, where `<run>` is the `.out` name minus `gwas_cli_` and `.out`. |

The two `gwas_cli_emma_*1000*.out` files were regenerated in the Phase 2
review. The Phase 1 versions had been produced with `--kfile
hmdp_K_npdump.txt`, so K was all zeros (the CLI printed h=0.068 instead of
the REML 0.454) and they were effectively OLS results.

The generator refuses to pass any text file that starts with `#` to a CLI.

### Other

`env.kv.txt` records the Python, numpy and scipy versions used for generation.
