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

## Formats (no pickles, all readable from Python 3)

- `*.txt` is `numpy.savetxt(fmt='%.17g')`, which round-trips float64
  exactly. `#` header lines describe the contents. Load with `np.loadtxt`.
- `*.kv.txt` holds one `key value` pair per line, with `#` comment lines.
- `*.out` is the raw output of `scripts/pylmmGWAS.py`: a tab-separated header
  `SNP_ID BETA BETA_SD F_STAT P_VALUE`, with values written by `str()` (shortest
  repr under numpy ≥1.14, `nan` when missing). Compare these numerically, not
  as text.

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
  - The kinship is rank deficient: 565 eigenvalues are below 1e-6 and get
    clamped.
  - The fake phenotype fits at the h=0 boundary.
  - 1550 of the first 3000 SNPs have missing genotypes.
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
| `hmdp_K_npdump.txt` | `hmdp.liver.K.npdump` exported as text (94×94). Also used as `--kfile` for the EMMA CLI runs. |
| `kinship_hmdp_snps1000.txt` | `lmm.calculateKinship(W)`, W = hmdp first 1000 SNPs (94×94). |
| `kinship_hmdp_snps1000_centered.txt` | Same, with `center=True`. |
| `kinship_bed_snps2000_calc.block100.txt` | `lmm.calculateKinship` on the BED's first 2000 SNPs, read with `input.plink(normGenotype=False)`: the leading `K[:100,:100]` block. |
| `kinship_bed_snps2000_calc.diag_rowsums.txt` | Same K: columns `diag(K)` and `K.sum(1)` (1219 rows). |
| `kinship_cli_bed_snps2000.{block100,diag_rowsums}.txt` | `pylmmKinship.py -v --bfile <first 2000 SNPs>` (default `-n 1000`, so 2 chunks), same summaries. |
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
| `gwas_cli_emma_hmdp_snps1000.out` | `-v --emmaSNP <hmdp first 1000 SNPs> --emmaPHENO data/hmdp.liver.exprs.1 --kfile hmdp_K_npdump.txt` |
| `gwas_cli_emma_hmdp_snps1000_cov.out` | The same, plus `--emmaCOV <row 2 of data/hmdp.liver.exprs>`. |

### Other

`env.kv.txt` records the Python, numpy and scipy versions used for generation.
