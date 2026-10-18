# Phase 3: Python 3 port

This phase ports the 2015 Python 2 code (the Phase 1 baseline in
`docs/phase1-baseline.md`) to Python 3.10+. Packaging moves to uv and
hatchling. The goal was minimal *behavioural* change: none of the known
algorithmic bugs from Phase 1 are fixed, and the outputs are checked against
the unmodified py2 goldens in `tests/golden/`.

## Summary

- All library code, the two CLIs and `scripts/example.py` run on Python
  3.10–3.14.
- No tolerance in `tests/helpers.py` was changed, and no golden file was
  edited.
- Fast suite (slow tests skipped) at this commit, before the PR #5 rebase:
  - macOS arm64: 25 passed, 3 failed.
  - Linux amd64: 26 passed, 2 failed.
  - All three failures are already being fixed on `phase2-tests` (see
    [Test results](#test-results)).
- Each fitted quantity is within 3e-8 relative of py2: h, sigma, LL and beta
  from the null fits, and `misc.fitTwo`.
- Per-SNP t and beta reach 5.6e-6 relative only where |t| is close to 0. In
  absolute terms they stay at or below 1.5e-7. This is the Brent-jitter
  regime that `RTOL_FIT`/`ATOL_FIT` was sized for.
- No nan position, SNP order or array shape differs from py2 anywhere.
- The ported code also passes the full suite in the py2.7 baseline image.
  So the port's edits do not change the numbers, and every py3 difference
  comes from numpy, scipy and BLAS.

## (a) Edit ledger

Tags: [syntax] [stdlib] [numpy-api] [bytes/str] [packaging] [numeric-affecting].
"Same numbers" means the change cannot alter any computed value.

### Commit "reindent" (whole-file whitespace)

| file | change | tag |
|---|---|---|
| all `pylmm/*.py`, `scripts/*.py` | Mixed tabs/spaces (py3 `TabError`) → 4 spaces per level. Tabs were expanded with py2 semantics (tab stop 8), then re-indented with autopep8's `Reindenter`, which leaves string contents alone. For each file I checked that the AST was unchanged apart from whitespace inside string literals. One side effect: the `pylmmGWAS` usage text ends in 12 spaces where it used to end in a tab plus 4 spaces. | [syntax] |

### Library (`pylmm/`) and `scripts/example.py`

| file:line (py2) | change | tag |
|---|---|---|
| input.py:96 | `self.N / 4` → `self.N // 4`. py2 did integer division here; this is the only int/int `/` in the code base (see audit below). | [syntax] |
| input.py:97, 118, 157-177 | The `struct.unpack('c'*n)` + `bin(ord(x))` string decoder is replaced by `np.frombuffer(X, uint8)`, `(B[:,None] >> [0,2,4,6]) & 3` and a lookup into `[0.0, nan, 0.5, 1.0]`. The code-to-value mapping and the order are the same: lowest bit pair first, then truncated to N. I checked the result is bit-identical to the old decoder for all 256 byte values and for N mod 4 = 0..3. `formatBinaryGenotypes(X, norm)` now takes the raw bytes. | [bytes/str] |
| input.py:104 | `order == '\x01'` → `order == b'\x01'`. In py3 the str/bytes comparison was always False, so every BED file was rejected. The magic number is still not checked, as in py2 (see (d)). | [bytes/str] |
| input.py:106 | `raise StopIteration` as an error signal: **kept**. It is raised from `__iter__`, not from a generator, so it behaves as in py2 (PEP 479 does not apply). | — |
| input.py:112 | `def next` → `def __next__`, plus a `next = __next__` alias so callers that use `IN.next()` (including `tests/golden/generate_golden.py`) keep working. | [syntax] |
| input.py:117 (new) | A short read of a truncated `.bed` raises `ValueError`. Before, `struct.unpack` raised `struct.error`. | [bytes/str] |
| input.py:212, 275, 315 | `D.has_key(k)` → `k in D` | [syntax] |
| input.py:245 | `self.indivs == None` → `is None` | [syntax] |
| input.py:23 | `self.indivs = range(...)` → `list(range(...))`, which keeps py2 list semantics. | [syntax] |
| input.py:11-12 | Removed the unused `import struct, pdb`. | [stdlib] |
| lmm.py:17-41 | `matrixMult` drops the `scipy.linalg.fblas` probe and the `dgemm` branch, and is now `np.dot(A, B)`. scipy has had no `linalg.fblas` since 0.12, so the py2 baseline (scipy 1.2.3) always went through this same `np.dot` path. Same numbers. | [numpy-api] |
| lmm.py:350 | `h == None` → `h is None` | [syntax] |
| misc.py:8 | Removed the stray `#!/usr/bin/python` after the header. | [syntax] |
| example.py:16-17, 22 | `np.load(..., allow_pickle=True)` → `+ encoding='latin1'`. This is needed to unpickle py2 `ndarray.dump` files. The commented line 22 is updated the same way. | [numpy-api] |
| example.py:1 | Shebang → `#!/usr/bin/env python3`. | [syntax] |

**Integer-division audit.** I checked every `/` in the py2 sources:

- input.py:96 `self.N / 4` is int/int and becomes `//`.
- input.py:187 `(G - m) / s` divides float arrays.
- lmm.py:60 `/ np.sqrt(vr)` divides float arrays.
- lmm.py:63 `1.0/float(m)` and :65 `1/float(n)` are float.
- lmm.py:67 `(n-1)*K / S` divides by a float.
- lmm.py:237, 244, 270, 323 have an explicit `1.0`/`float()`.
- lmm.py:366 `beta / np.sqrt(...)` and :392, 403 `p/p.sum()` divide float arrays.
- misc.py:23 `/ float(wgrids)` is float.
- pylmmGWAS.py:255 `/ np.sqrt(xs.var())` divides a float array.
- pylmmKinship.py:102 `K / float(IN.numSNPs)` is float.

Every case other than input.py:96 is float division, and py2 and py3 give the
same result for those.

**Not needed:** the code had no `print` statements (all output uses
`sys.stderr.write`). It also had no `xrange`, `iteritems`, `unicode` or
`basestring`. Dict iteration order never affects a result: the dicts are
used only for lookups.

**Kept on purpose:**

- bare `except:` (input.py:138, gwas)
- mutable default arguments `Kva=[]`
- the `x == 'NA' and np.nan or float(x)` idiom
- every bug in the Phase 1 list (several now have `NOTE (known bug #n, kept)` comments in `pylmm/cli/gwas.py`)

### CLIs (`scripts/pylmm*.py` → `pylmm/cli/`)

| change | tag |
|---|---|
| The module-level script code moved into `pylmm/cli/gwas.py` and `pylmm/cli/kinship.py`. Each has `buildParser()` and `main(argv=None)`. Option definitions, usage and help strings are byte-identical; I checked that every string constant in the old script appears in the new module. | [packaging] |
| `scripts/pylmmGWAS.py` and `scripts/pylmmKinship.py` are now thin shims that call `main()`. If `pylmm` cannot be imported, they fall back to adding the repo root to `sys.path`. | [packaging] |
| gwas: the `.gz` kinship is read with `gzip.open(fn, 'rt')` and parsed with `np.fromstring(str, sep=' ')`. py2 read bytes via mode `'r'`. The parse is the same; cross-checked against py2, see (b). | [stdlib] [bytes/str] |
| gwas: `np.fromfile(open(fn,'r'), sep=" ")` → `np.fromfile(fn, sep=" ")`, so numpy is given the path instead of a py3 text-mode file object. A plain whitespace kinship parses exactly as before. | [numpy-api] |
| gwas: `K.resize((n,n))` → `K.resize((n,n), refcheck=False)`. It is the same in-place zero-fill/truncate; this only avoids refcount false positives on newer CPython. | [numpy-api] |
| gwas: the output file is opened with `with` and closed when `main` returns. py2 left it to interpreter exit. Content is the same. `printOutHead` / `outputResult` take the file as an argument instead of a global. | [stdlib] |
| gwas: the duplicated kinship/kinship2 reading code is factored into `readKinshipFile()`. It does the same thing. | [syntax] |
| kinship: `IN.next()` → `next(IN)`. `W[:,range(0,j)]` → `W[:,list(range(0,j))]`. | [syntax] |
| kinship: the `linalg.fblas.dgemm` try/except is replaced by `np.dot(W, W.T)`, the fallback py2 always took. Same numbers. | [numpy-api] |
| Removed the unused `import pdb` (gwas, kinship) and the commented `pdb.set_trace()`. | [stdlib] |
| With the entry points, `%prog` in the usage line shows `pylmmGWAS` / `pylmmKinship`. | [packaging] |

### Packaging, tests, tooling

| change | tag |
|---|---|
| `pyproject.toml` (hatchling, `requires-python >=3.10`, `numpy>=1.22`, `scipy>=1.8`, dev group `pytest>=7`), `.python-version` = 3.13, `uv.lock`. Console scripts `pylmmGWAS`, `pylmmKinship`. | [packaging] |
| Removed `setup.py` (distutils is gone in Python 3.12+) and `requirements.txt`. `.gitignore` adds `.venv/`. | [packaging] |
| tests: removed `from __future__ import division, print_function` from every test module and updated the `helpers.py` docstring. Nothing else changed: same assertions, same tolerances. `helpers.load_npdump` already had `encoding='latin1'`. | [syntax] |
| New `tests/golden/diff_vs_golden.py`, which produces the tables in (b). It is not a test. | — |
| `justfile` (sync, test, test-slow, test-py27, example, golden), `.github/workflows/tests.yml` (3.10–3.14, ubuntu-latest), README install section. | [packaging] |

**[numeric-affecting]: none.** No edit changes what is computed. Every
difference in (b) comes from the numerical stack: numpy 1.16.6 / scipy 1.2.3
/ OpenBLAS on py2, versus numpy 2.x / scipy 1.15–1.18 / Accelerate or
OpenBLAS now. Those differences are then amplified by Brent's xtol.

## (b) py3 vs py2 golden diffs

Produced with `uv run python tests/golden/diff_vs_golden.py`. The script
re-runs every `generate_golden.py` section on the current code and compares
the results numerically with the stored goldens.

- "rel" is `max |a-e|/|e|` over entries with e ≠ 0.
- For the eigenvalue and row-sum columns, rel is meaningless: those entries
  are round-off around 0. That is why the eigenvalue rows also give the
  `entries > 1e-6` subset.

### Worst cases (macOS arm64 / Accelerate, py3.13, numpy 2.5.4, scipy 1.18.1)

| quantity | max abs | max rel | note |
|---|---|---|---|
| null fits h/sigma/LL/beta (all 6 configs) | 1.4e-8 | 2.9e-8 | `hmdp_missingY_reml` is the worst |
| `misc.fitTwo` | 6.8e-9 | 1.5e-8 | |
| LL grid at fixed h | 3.0e-13 | 5.9e-14 | deterministic tier |
| kinships (hmdp, BED, both CLIs) | 1.0e-14 | 2.4e-13 | deterministic tier |
| eigenvalues > 1e-6 | 1.6e-13 | 7.7e-15 | |
| BED kinship **row sums** | 1.3e-12 | (37–137) | sums of about 0 (cancellation); fails `ATOL=1e-12` on macOS, see test results |
| per-SNP t (refit, hmdp `lmm.GWAS`) | 8.3e-8 | 5.6e-6 | rel is large only where \|t\| is about 1e-2 |
| per-SNP p | 1.4e-7 | 1.9e-7 | |
| per-SNP beta (imputed) | 1.3e-9 | 1.9e-6 | beta about 0 |
| BED CLI, fixed h (default, `--REML`) | 5.6e-14 | 4.7e-13 | the BED fit sits at the h=0 boundary, so no Brent |
| BED CLI `--refit` / `--removeMissingGenotypes` | 1.9e-7 | 6.4e-7 | F_STAT |
| EMMA CLI (stored goldens) | — | — | FAILED by design, see (d)1 |

Nothing exceeds 1e-6 relative on a fitted quantity (h, sigma, LL, beta of a
fit). The two items above 1e-6 relative are per-SNP t-stats and betas with
values near zero, where the absolute differences are 1e-9 to 1e-7.

### Header-less EMMA CLI cross-check (py2 Docker vs py3)

The stored EMMA CLI goldens were produced with an all-zero K; see (d)1. To
check the EMMA path anyway, I ran the original py2 code (`origin/phase2-tests`
in `pylmm-py27:baseline`) and the py3 port on the same header-less copy of
`hmdp_K_npdump.txt`. Both printed `heritability=0.454, sigma=0.040`.

| run | column | max abs | max rel |
|---|---|---|---|
| no cov | BETA / BETA_SD / F_STAT / P_VALUE | 2.7e-10 / 1.7e-10 / 1.2e-8 / 8.0e-9 | 1.9e-6 / 5.0e-9 / 1.9e-6 / 2.8e-8 |
| `--emmaCOV` | BETA / BETA_SD / F_STAT / P_VALUE | 1.7e-9 / 1.1e-9 / 7.4e-8 / 5.7e-8 | 2.5e-5 / 3.3e-8 / 2.5e-5 / 1.9e-7 |
| `.gz` `--kfile` (no cov) | same as the "no cov" row | | |

In every run the nan positions (68 of 1000) and SNP IDs match. The 2.5e-5
relative belongs to the smallest-|beta| SNP: its absolute difference is
1.7e-9, inside `ATOL_FIT`.

### Full table, macOS arm64, py3.13

python 3.13.9, numpy 2.5.4, scipy 1.18.1; regenerated in 33.9s

| file | column | n | nan | nan mismatches | max abs diff | max rel diff |
|---|---|---|---|---|---|---|
| assoc_hmdp_GWAS.txt | snp_index | 1000 | 0 | 0 | 0 (bit-identical) | 0 |
| assoc_hmdp_GWAS.txt | n_missing | 1000 | 0 | 0 | 0 (bit-identical) | 0 |
| assoc_hmdp_GWAS.txt | ts_reml_norefit | 1000 | 68 | 0 | 4.35e-08 | 2.36e-06 |
| assoc_hmdp_GWAS.txt | ps_reml_norefit | 1000 | 68 | 0 | 3.14e-08 | 7.93e-08 |
| assoc_hmdp_GWAS.txt | ts_reml_refit | 1000 | 68 | 0 | 8.28e-08 | 5.61e-06 |
| assoc_hmdp_GWAS.txt | ps_reml_refit | 1000 | 68 | 0 | 4.78e-08 | 1.7e-07 |
| assoc_hmdp_GWAS.txt | ts_ml_norefit | 1000 | 68 | 0 | 4.35e-08 | 2.36e-06 |
| assoc_hmdp_GWAS.txt | ps_ml_norefit | 1000 | 68 | 0 | 3.14e-08 | 7.93e-08 |
| assoc_hmdp_imputed.txt | snp_index | 1000 | 0 | 0 | 0 (bit-identical) | 0 |
| assoc_hmdp_imputed.txt | ts | 1000 | 68 | 0 | 1.16e-08 | 1.86e-06 |
| assoc_hmdp_imputed.txt | ps | 1000 | 68 | 0 | 8.04e-09 | 2.8e-08 |
| assoc_hmdp_imputed.txt | beta | 1000 | 68 | 0 | 1.28e-09 | 1.86e-06 |
| assoc_hmdp_imputed.txt | beta_var(=betaVAR*sigma) | 1000 | 68 | 0 | 3.67e-10 | 1e-08 |
| bed_readme_kin_eigenvalues.txt | all | 1219 | 0 | 0 | 1.56e-13 | 81.1 |
| bed_readme_kin_eigenvalues.txt | entries > 1e-6 | 110 | 0 | 0 | 1.56e-13 | 7.74e-15 |
| example_py.kv.txt | (all) | 15 | 0 | 0 | 5.07e-07 | 5.39e-08 |
| gwas_cli_bed_snps1000.out | BETA | 1000 | 0 | 0 | 1.83e-15 | 4.71e-13 |
| gwas_cli_bed_snps1000.out | BETA_SD | 1000 | 0 | 0 | 2.57e-16 | 8.95e-15 |
| gwas_cli_bed_snps1000.out | F_STAT | 1000 | 0 | 0 | 5.62e-14 | 4.73e-13 |
| gwas_cli_bed_snps1000.out | P_VALUE | 1000 | 0 | 0 | 2.43e-14 | 1.29e-13 |
| gwas_cli_bed_snps50_refit.out | BETA | 50 | 0 | 0 | 6.73e-09 | 4.25e-07 |
| gwas_cli_bed_snps50_refit.out | BETA_SD | 50 | 0 | 0 | 4.03e-09 | 1.24e-07 |
| gwas_cli_bed_snps50_refit.out | F_STAT | 50 | 0 | 0 | 1.47e-07 | 3e-07 |
| gwas_cli_bed_snps50_refit.out | P_VALUE | 50 | 0 | 0 | 1.04e-07 | 1.67e-07 |
| gwas_cli_bed_snps50_reml.out | BETA | 50 | 0 | 0 | 1.01e-15 | 3.18e-14 |
| gwas_cli_bed_snps50_reml.out | BETA_SD | 50 | 0 | 0 | 3.47e-17 | 1.22e-15 |
| gwas_cli_bed_snps50_reml.out | F_STAT | 50 | 0 | 0 | 3.4e-14 | 3.17e-14 |
| gwas_cli_bed_snps50_reml.out | P_VALUE | 50 | 0 | 0 | 1.1e-14 | 7.31e-14 |
| gwas_cli_bed_snps50_reml_refit.out | BETA | 50 | 0 | 0 | 5.95e-09 | 3.27e-07 |
| gwas_cli_bed_snps50_reml_refit.out | BETA_SD | 50 | 0 | 0 | 2.8e-09 | 8.68e-08 |
| gwas_cli_bed_snps50_reml_refit.out | F_STAT | 50 | 0 | 0 | 8.04e-08 | 2.4e-07 |
| gwas_cli_bed_snps50_reml_refit.out | P_VALUE | 50 | 0 | 0 | 6.07e-08 | 1.51e-07 |
| gwas_cli_bed_snps50_removeMissing.out | BETA | 50 | 0 | 0 | 5.12e-09 | 5.45e-07 |
| gwas_cli_bed_snps50_removeMissing.out | BETA_SD | 50 | 0 | 0 | 3.16e-09 | 9.86e-08 |
| gwas_cli_bed_snps50_removeMissing.out | F_STAT | 50 | 0 | 0 | 1.88e-07 | 6.44e-07 |
| gwas_cli_bed_snps50_removeMissing.out | P_VALUE | 50 | 0 | 0 | 1.44e-07 | 1.87e-07 |
| gwas_cli_emma_hmdp_snps1000.out | | | | | FAILED: ValueError: string or file could not be read to its end due to unmatched data | |
| gwas_cli_emma_hmdp_snps1000_cov.out | | | | | FAILED: ValueError: string or file could not be read to its end due to unmatched data | |
| hmdp_K_npdump.txt | all | 8836 | 0 | 0 | 0 (bit-identical) | 0 |
| kinship_bed_snps2000_calc.block100.txt | all | 10000 | 0 | 0 | 5.77e-15 | 5.63e-14 |
| kinship_bed_snps2000_calc.diag_rowsums.txt | diag | 1219 | 0 | 0 | 5.77e-15 | 3.64e-15 |
| kinship_bed_snps2000_calc.diag_rowsums.txt | rowsums | 1219 | 0 | 0 | 1.33e-12 | 137 |
| kinship_cli_bed_snps2000.block100.txt | all | 10000 | 0 | 0 | 2e-15 | 3.36e-14 |
| kinship_cli_bed_snps2000.diag_rowsums.txt | diag | 1219 | 0 | 0 | 1.11e-15 | 1.19e-15 |
| kinship_cli_bed_snps2000.diag_rowsums.txt | rowsums | 1219 | 0 | 0 | 5.04e-13 | 93 |
| kinship_cli_emma_hmdp_snps1000.kva.txt | all | 94 | 0 | 0 | 2.49e-14 | 5.52 |
| kinship_cli_emma_hmdp_snps1000.kva.txt | entries > 1e-6 | 44 | 0 | 0 | 2.49e-14 | 6.65e-13 |
| kinship_cli_emma_hmdp_snps1000.txt | all | 8836 | 0 | 0 | 5.33e-15 | 2.2e-13 |
| kinship_hmdp_snps1000.txt | all | 8836 | 0 | 0 | 7.77e-15 | 2.44e-13 |
| kinship_hmdp_snps1000_centered.txt | all | 8836 | 0 | 0 | 1.02e-14 | 2.45e-13 |
| misc_fitTwo.kv.txt | (all) | 6 | 0 | 0 | 6.83e-09 | 1.52e-08 |
| null_fit_hmdp_LLgrid.txt | h | 100 | 0 | 0 | 0 (bit-identical) | 0 |
| null_fit_hmdp_LLgrid.txt | LL_REML | 100 | 0 | 0 | 2.95e-13 | 3.96e-14 |
| null_fit_hmdp_LLgrid.txt | LL_ML | 100 | 0 | 0 | 2.98e-13 | 5.92e-14 |
| null_fits.kv.txt | bed_ml | 7 | 0 | 0 | 2.27e-13 | 6.63e-14 |
| null_fits.kv.txt | bed_reml | 7 | 0 | 0 | 2.27e-13 | 8.42e-14 |
| null_fits.kv.txt | hmdp_cov_reml | 8 | 0 | 0 | 3.37e-10 | 7.73e-10 |
| null_fits.kv.txt | hmdp_missingY_reml | 7 | 0 | 0 | 1.39e-08 | 2.93e-08 |
| null_fits.kv.txt | hmdp_ml | 7 | 0 | 0 | 6.29e-10 | 1.31e-09 |
| null_fits.kv.txt | hmdp_reml | 7 | 0 | 0 | 6.65e-09 | 1.46e-08 |

### Other Python versions / platforms (worst cases)

Worst abs / rel per file. For eigenvalue files, rel is over entries > 1e-6; for row-sum columns, rel is left out.

| file | macOS py3.10 abs / rel | macOS py3.13 abs / rel | macOS py3.14 abs / rel | linux/amd64 py3.13 abs / rel |
|---|---|---|---|---|
| assoc_hmdp_GWAS.txt | 1.4e-07 / 1e-05 | 8.3e-08 / 5.6e-06 | 8.3e-08 / 5.6e-06 | 2.4e-07 / 9.4e-06 |
| assoc_hmdp_imputed.txt | 1.2e-08 / 1.9e-06 | 1.2e-08 / 1.9e-06 | 1.2e-08 / 1.9e-06 | 4.1e-08 / 6.5e-06 |
| bed_readme_kin_eigenvalues.txt | 1.6e-13 / 7.7e-15 | 1.6e-13 / 7.7e-15 | 1.6e-13 / 7.7e-15 | 1.1e-13 / 2.3e-14 |
| example_py.kv.txt | 2.5e-07 / 5.4e-08 | 5.1e-07 / 5.4e-08 | 5.1e-07 / 5.4e-08 | 1.3e-06 / 2e-07 |
| gwas_cli_bed_snps1000.out | 5.6e-14 / 4.7e-13 | 5.6e-14 / 4.7e-13 | 5.6e-14 / 4.7e-13 | 8.4e-14 / 3.5e-13 |
| gwas_cli_bed_snps50_refit.out | 1.5e-07 / 4.3e-07 | 1.5e-07 / 4.3e-07 | 1.5e-07 / 4.3e-07 | 1.4e-07 / 4.2e-07 |
| gwas_cli_bed_snps50_reml.out | 3.4e-14 / 7.3e-14 | 3.4e-14 / 7.3e-14 | 3.4e-14 / 7.3e-14 | 5e-14 / 1.1e-13 |
| gwas_cli_bed_snps50_reml_refit.out | 8e-08 / 3.3e-07 | 8e-08 / 3.3e-07 | 8e-08 / 3.3e-07 | 1.5e-07 / 1.8e-06 |
| gwas_cli_bed_snps50_removeMissing.out | 1.9e-07 / 6.4e-07 | 1.9e-07 / 6.4e-07 | 1.9e-07 / 6.4e-07 | 5.7e-08 / 2.4e-07 |
| gwas_cli_emma_hmdp_snps1000.out | FAILED (headered K) | FAILED (headered K) | FAILED (headered K) | FAILED (headered K) |
| gwas_cli_emma_hmdp_snps1000_cov.out | FAILED (headered K) | FAILED (headered K) | FAILED (headered K) | FAILED (headered K) |
| hmdp_K_npdump.txt | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| kinship_bed_snps2000_calc.block100.txt | 5.8e-15 / 5.6e-14 | 5.8e-15 / 5.6e-14 | 5.8e-15 / 5.6e-14 | 0 / 0 |
| kinship_bed_snps2000_calc.diag_rowsums.txt | 1.3e-12 / 3.6e-15 | 1.3e-12 / 3.6e-15 | 1.3e-12 / 3.6e-15 | 0 / 0 |
| kinship_cli_bed_snps2000.block100.txt | 2e-15 / 3.4e-14 | 2e-15 / 3.4e-14 | 2e-15 / 3.4e-14 | 0 / 0 |
| kinship_cli_bed_snps2000.diag_rowsums.txt | 5e-13 / 1.2e-15 | 5e-13 / 1.2e-15 | 5e-13 / 1.2e-15 | 0 / 0 |
| kinship_cli_emma_hmdp_snps1000.kva.txt | 2.5e-14 / 6.6e-13 | 2.5e-14 / 6.6e-13 | 2.5e-14 / 6.6e-13 | 1.1e-14 / 7.1e-13 |
| kinship_cli_emma_hmdp_snps1000.txt | 5.3e-15 / 2.2e-13 | 5.3e-15 / 2.2e-13 | 5.3e-15 / 2.2e-13 | 0 / 0 |
| kinship_hmdp_snps1000.txt | 7.8e-15 / 2.4e-13 | 7.8e-15 / 2.4e-13 | 7.8e-15 / 2.4e-13 | 0 / 0 |
| kinship_hmdp_snps1000_centered.txt | 1e-14 / 2.5e-13 | 1e-14 / 2.5e-13 | 1e-14 / 2.5e-13 | 0 / 0 |
| misc_fitTwo.kv.txt | 6.8e-09 / 1.5e-08 | 6.8e-09 / 1.5e-08 | 6.8e-09 / 1.5e-08 | 1.3e-08 / 3e-08 |
| null_fit_hmdp_LLgrid.txt | 3e-13 / 5.9e-14 | 3e-13 / 5.9e-14 | 3e-13 / 5.9e-14 | 9.2e-13 / 1.8e-13 |
| null_fits.kv.txt | 1.4e-08 / 2.9e-08 | 1.4e-08 / 2.9e-08 | 1.4e-08 / 2.9e-08 | 2.3e-08 / 5.1e-08 |

- macOS py3.10: python 3.10.19, numpy 2.2.6, scipy 1.15.3; regenerated in 40.1s
- macOS py3.13: python 3.13.9, numpy 2.5.4, scipy 1.18.1; regenerated in 33.9s
- macOS py3.14: python 3.14.4, numpy 2.5.4, scipy 1.18.1; regenerated in 34.3s
- linux/amd64 py3.13: python 3.13.11, numpy 2.5.4, scipy 1.18.1; regenerated in 66.4s

## (c) Environment

| | py2 baseline (goldens) | py3 dev machine | CI-equivalent check |
|---|---|---|---|
| OS / arch | linux/amd64 (Docker, emulated) | macOS 15.7.2 arm64 | linux/amd64 Docker (`ghcr.io/astral-sh/uv:python3.13-bookworm-slim`, emulated) |
| Python | 2.7.18 | 3.13.9 (also 3.10.19, 3.14.4) | 3.13 |
| numpy | 1.16.6 | 2.5.4 (2.2.6 on 3.10) | 2.5.4 |
| scipy | 1.2.3 | 1.18.1 (1.15.3 on 3.10) | 1.18.1 |
| BLAS/LAPACK | OpenBLAS (wheel), 1 thread | Apple Accelerate (numpy and scipy wheels) | OpenBLAS (wheel), `OPENBLAS_NUM_THREADS=1` |
| tools | — | uv 0.12.19, pytest 9.1.1, hatchling | same lock |

## Test results

`uv run pytest` (slow tests skipped), on this branch before the rebase onto
the PR #5 fixes:

| env | result | wall time |
|---|---|---|
| macOS, py3.13 | 25 passed, 3 failed, 2 skipped | 44 s |
| macOS, py3.13 `--runslow` | 27 passed, 3 failed | 87 s |
| macOS, py3.10 | 25 passed, 3 failed, 2 skipped | 49 s |
| macOS, py3.14 | 25 passed, 3 failed, 2 skipped | 42 s |
| linux/amd64 Docker, py3.13 (emulated) | 26 passed, 2 failed, 2 skipped | 4 min 9 s |
| linux/amd64 Docker, py3.10, **lower bounds** numpy 1.22.0 / scipy 1.8.0 | 28 passed, 2 skipped | 57 s |
| **py2.7 baseline image, ported code** (`docker/py27/test.sh`) | 28 passed, 2 skipped | 62 s |

The last two rows matter:

- **Lower bounds.** On numpy 1.x, `np.fromfile` still silently zero-fills
  the headered K, so the EMMA CLI tests pass against the old (wrong)
  goldens. This confirms (d)1.
- **py2.7 baseline.** The ported code still runs on py2.7 with numpy 1.16.6.
  It passes every test there, including the row sums at `ATOL=1e-12`. So the
  port's edits do not change the numbers, and every py3 difference in (b)
  comes from the numerical stack. The code is not *meant* to stay
  py2-compatible; it just happens to be today.

The failures:

1. `test_cli_emma[nocov]` and `test_cli_emma[cov]` fail everywhere.
   - `pylmmGWAS` exits with `ValueError: string or file could not be read to
     its end due to unmatched data`, because `--kfile` points at
     `tests/golden/hmdp_K_npdump.txt`, which has `#` header lines. See (d)1.
   - PR #5 is regenerating these goldens with a header-less K.
   - The cross-check in (b) shows the port matches py2 on that input.
2. `test_calculate_kinship_bed`, rowsums only, macOS only.
   - 78 of 1219 row sums differ by up to 1.33e-12 absolute, against
     `ATOL=1e-12`.
   - The expected values are about 1e-13. They are sums of 1219 entries of
     order 1 that cancel to about 0, so the result is pure summation-order
     round-off (Accelerate vs OpenBLAS).
   - The block, the diagonal, and the CLI kinship row sums all pass.
   - It passes on linux/OpenBLAS (CI).
   - I did not loosen the tolerance: PR #5 is replacing the row-sum goldens
     with `K@v`.

The slow tests (full README kinship and full README GWAS) pass.

## (d) Behaviour changes and newly found bugs

### Behaviour changes

1. **A kinship file with non-numeric content is now an error.** In py2
   (numpy 1.16), `np.fromfile(sep=" ")` stopped at the first non-number with
   only a DeprecationWarning. `K.resize` then zero-filled the matrix, so a K
   file with a `#` header was silently read as an **all-zero kinship**. That
   is how the stored EMMA CLI goldens were made; PR #5 found this in review.
   numpy 2.x raises `ValueError`. I kept the numpy behaviour, because
   emulating the zero-K bug would mean matching wrong numbers. A plain
   whitespace kinship, with or without `.gz`, parses exactly as before. With
   numpy 1.x (the `>=1.22` lower bound) I expect the old warn-and-zero-fill
   behaviour, but I did not verify it.
2. A truncated `.bed` raises `ValueError` instead of `struct.error`.
3. `plink.formatBinaryGenotypes(X, norm)` takes the raw bytes of one SNP. It
   used to take a list of `bin()` strings. It is internal and only called
   from `__next__`.
4. The usage text of `pylmmGWAS` ends in different trailing whitespace (from
   the tab expansion), and `%prog` shows the entry-point name.
5. The `pylmmGWAS` output file is flushed and closed when `main()` returns,
   not at interpreter exit.

### Newly found bugs (not fixed)

1. `LMM.association(..., returnBeta=True)` returns `beta_var` as a **(1,1)
   ndarray**, because `sigma` is 1×1, while `ts`, `ps` and `beta` are
   scalars. `LMM.fit()` also returns `sigma` as 1×1.
   `generate_golden.py`'s `savetxt` of those rows only worked because numpy
   1.16 coerced ragged lists. numpy ≥1.24 refuses, so `diff_vs_golden.py`
   flattens them.
2. The BED magic number (`0x6c 0x1b`) is read but never checked. Only the
   SNP-major byte is checked, so a non-BED file with `\x01` as its third byte
   is decoded as genotypes. This is unchanged from py2.
3. The BED kinship row-sum goldens (`*.diag_rowsums.txt`, column 2) cannot
   be checked against `ATOL=1e-12` across BLAS builds (see Test results).
   This is a test-design issue that PR #5 is already addressing.
4. `scipy==1.8.0` (the declared lower bound) **segfaults in `linalg.eigh`**
   on macOS arm64 with Python 3.10 (`uv run --resolution lowest-direct`).
   This is a known problem with early arm64 scipy wheels, not a pylmm bug.
   On linux/amd64, numpy 1.22.0 with scipy 1.8.0 passes the suite (see Test
   results). I left the bounds alone. If macOS arm64 users on old pins
   matter, consider raising them.
5. `tests/README.md` still says the tests fail on Python 3 "only because
   pylmm itself is Python 2 code". I left it alone to keep this diff out of
   the PR #5 files.

## Notes for the next phases

- `docker/py27/` stays in the repo for reference and is not run in CI.
  `just golden` regenerates the goldens from whatever code is in the working
  tree. To reproduce the Phase 1 baseline, check out the baseline code first.
- Ruff, type annotations and the snake_case rename are left to the planned
  hygiene phase.
