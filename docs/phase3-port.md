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
- The branch is rebased onto the PR #5 test suite. The fast suite (slow
  tests skipped) gives **32 passed, 2 skipped** everywhere it was run: macOS
  arm64 py3.10/3.13/3.14, linux/amd64 py3.13, both platforms at the
  numpy/scipy lower bounds, and the py2.7 baseline image. With `--runslow`
  (macOS py3.13): 34 passed. See [Test results](#test-results).
- Each fitted quantity (h, sigma, LL and beta from the null fits, and
  `misc.fitTwo`) is within 2.9e-8 relative of py2 on macOS/Accelerate and
  5.1e-8 on linux/OpenBLAS.
- Per-SNP statistics (t/F, p, beta) differ from py2 by at most 1.9e-7
  absolute on macOS and 4.2e-7 on linux. Relative differences reach 2.5e-5
  (macOS) and 1.4e-4 (linux), but only for values close to 0 (both maxima
  are in the EMMA `--emmaCOV` run). This is the Brent-jitter regime that
  `RTOL_FIT`/`ATOL_FIT` (5e-6 absolute) was sized for.
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
| `scripts/pylmmGWAS.py` and `scripts/pylmmKinship.py` are now thin shims that call `main()`. They always put the repo root first on `sys.path` (and drop an already-imported `pylmm` from elsewhere), so they run the checkout's code even when another pylmm is installed. | [packaging] |
| gwas: the `.gz` kinship is read with `gzip.open(fn, 'rt')` and parsed with `np.fromstring(str, sep=' ')`. py2 read bytes via mode `'r'`. The parse is the same: the regenerated EMMA CLI goldens match py2, see (b), and the generator asserts the `.gz` run reproduces the plain-text run row for row. | [stdlib] [bytes/str] |
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
| `pyproject.toml` (`hatchling>=1.27`, the first release that accepts the PEP 639 `license-files` list; `requires-python >=3.10`; `numpy>=1.22`, `scipy>=1.9`; dev group `pytest>=7`), `.python-version` = 3.13, `uv.lock`. Console scripts `pylmmGWAS`, `pylmmKinship`. The sdist excludes `tests/` and `scripts/` (the tests need `tests/golden` and `data/`), so it holds only the package. | [packaging] |
| Lower bounds: scipy 1.8.0 and 1.8.1 segfault in the suite on macOS arm64 / py3.10, so scipy starts at 1.9. numpy 1.22.0 + scipy 1.9.0 (`--resolution lowest-direct` on py3.10) pass on macOS arm64 and linux/amd64; a CI job checks this on ubuntu. The rationale is in a comment in `pyproject.toml`. | [packaging] |
| Removed `setup.py` (distutils is gone in Python 3.12+) and `requirements.txt`. `.gitignore` adds `.venv/`. | [packaging] |
| tests: removed `from __future__ import division, print_function` from every test module and updated the `helpers.py` docstring. Nothing else changed: same assertions, same tolerances. `helpers.load_npdump` already had `encoding='latin1'`. | [syntax] |
| New `tests/golden/diff_vs_golden.py`, which produces the tables in (b). It is not a test. | — |
| `justfile` (sync, test, test-slow, test-py27, example, golden; `set positional-arguments` so quoted pytest args pass through), `.github/workflows/tests.yml` (3.10–3.14 on ubuntu-latest, plus a py3.10 `--resolution lowest-direct` job and a `--help` smoke test of both console entry points), README install section (`pip install git+...@main`). | [packaging] |

**[numeric-affecting]: none.** No edit changes what is computed. Every
difference in (b) comes from the numerical stack: numpy 1.16.6 / scipy 1.2.3
/ OpenBLAS on py2, versus numpy 2.x / scipy 1.15–1.18 / Accelerate or
OpenBLAS now. Those differences are then amplified by Brent's xtol.

## (b) py3 vs py2 golden diffs

Produced with `uv run python tests/golden/diff_vs_golden.py`, after the
rebase onto the PR #5 goldens. The script re-runs every `generate_golden.py`
section on the current code and compares the results numerically with the
stored goldens. Every golden file is regenerated; nothing fails.

- "rel" is `max |a-e|/|e|` over entries with e ≠ 0.
- For the eigenvalue files, rel over all entries is meaningless: most
  entries are round-off around 0. That is why the eigenvalue rows also give
  the `entries > 1e-6` subset.
- The EMMA CLI goldens now come from a header-less K (PR #5), so they are
  compared directly. Before the rebase they had been made with an all-zero K
  and could not be reproduced on numpy 2; see (d) behaviour change 1.

### Worst cases (macOS arm64 / Accelerate, py3.13, numpy 2.5.4, scipy 1.18.1)

| quantity | max abs | max rel | note |
|---|---|---|---|
| null fits h/sigma/LL/beta (all 6 configs) | 1.4e-8 | 2.9e-8 | `hmdp_missingY_reml` is the worst |
| `misc.fitTwo` | 6.8e-9 | 1.5e-8 | |
| `example.py` values | 5.1e-7 | 5.4e-8 | null fit, first SNP, GWAS summary |
| LL grid at fixed h | 3.0e-13 | 5.9e-14 | deterministic tier |
| kinships (hmdp, BED, both CLIs; blocks, diagonal, `K.dot(cos)`) | 3.5e-14 | 2.5e-13 | deterministic tier |
| eigenvalues > 1e-6 | 1.6e-13 | 6.7e-13 | |
| per-SNP t (hmdp `lmm.GWAS`, refit) | 8.3e-8 | 5.6e-6 | rel is large only where \|t\| is about 1e-2 |
| per-SNP p (all API and CLI runs) | 1.4e-7 | 1.9e-7 | |
| per-SNP beta (imputed) | 1.3e-9 | 1.9e-6 | beta about 0 |
| BED CLI, fixed h (default, `--REML`, `--covfile`, NA/-9 phenotypes, `--tfile`) | 5.6e-14 | 4.7e-13 | the BED fit sits at the h=0 boundary, so no Brent |
| BED CLI `--refit` / `--removeMissingGenotypes` | 1.9e-7 | 6.4e-7 | F_STAT |
| EMMA CLI (no cov, `--emmaCOV`, NA phenotypes) | 7.4e-8 | 2.5e-5 | the 2.5e-5 is BETA/F_STAT of a near-zero beta in the `--emmaCOV` run (BETA max abs 1.7e-9) |
| null fit printed by `pylmmGWAS -v` (all 9 runs) | 0 | 0 | identical at the printed 3 decimals |

Nothing exceeds 1e-6 relative on a fitted quantity (h, sigma, LL, beta of a
fit). The items above 1e-6 relative are per-SNP t-stats and betas with
values near zero, where the absolute differences are 1e-9 to 1e-7.

### Full table, macOS arm64, py3.13

python 3.13.9, numpy 2.5.4, scipy 1.18.1; regenerated in 41.0s

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
| gwas_cli_bed_snps50_covfile.out | BETA | 50 | 0 | 0 | 9.92e-16 | 3.15e-14 |
| gwas_cli_bed_snps50_covfile.out | BETA_SD | 50 | 0 | 0 | 4.16e-17 | 1.46e-15 |
| gwas_cli_bed_snps50_covfile.out | F_STAT | 50 | 0 | 0 | 3.44e-14 | 3.15e-14 |
| gwas_cli_bed_snps50_covfile.out | P_VALUE | 50 | 0 | 0 | 1.12e-14 | 7.46e-14 |
| gwas_cli_bed_snps50_missingpheno.out | BETA | 50 | 0 | 0 | 8.67e-16 | 7.72e-14 |
| gwas_cli_bed_snps50_missingpheno.out | BETA_SD | 50 | 0 | 0 | 1.08e-16 | 3.69e-15 |
| gwas_cli_bed_snps50_missingpheno.out | F_STAT | 50 | 0 | 0 | 2.84e-14 | 7.35e-14 |
| gwas_cli_bed_snps50_missingpheno.out | P_VALUE | 50 | 0 | 0 | 2.09e-14 | 4.57e-14 |
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
| gwas_cli_emma_hmdp_snps1000.out | BETA | 1000 | 68 | 0 | 2.65e-10 | 1.86e-06 |
| gwas_cli_emma_hmdp_snps1000.out | BETA_SD | 1000 | 68 | 0 | 1.69e-10 | 5.01e-09 |
| gwas_cli_emma_hmdp_snps1000.out | F_STAT | 1000 | 68 | 0 | 1.16e-08 | 1.86e-06 |
| gwas_cli_emma_hmdp_snps1000.out | P_VALUE | 1000 | 68 | 0 | 8.04e-09 | 2.8e-08 |
| gwas_cli_emma_hmdp_snps1000_cov.out | BETA | 1000 | 68 | 0 | 1.67e-09 | 2.54e-05 |
| gwas_cli_emma_hmdp_snps1000_cov.out | BETA_SD | 1000 | 68 | 0 | 1.09e-09 | 3.25e-08 |
| gwas_cli_emma_hmdp_snps1000_cov.out | F_STAT | 1000 | 68 | 0 | 7.43e-08 | 2.54e-05 |
| gwas_cli_emma_hmdp_snps1000_cov.out | P_VALUE | 1000 | 68 | 0 | 5.66e-08 | 1.89e-07 |
| gwas_cli_emma_hmdp_snps100_missingpheno.out | BETA | 100 | 12 | 0 | 5.35e-10 | 1.21e-06 |
| gwas_cli_emma_hmdp_snps100_missingpheno.out | BETA_SD | 100 | 12 | 0 | 2.48e-10 | 6.5e-09 |
| gwas_cli_emma_hmdp_snps100_missingpheno.out | F_STAT | 100 | 12 | 0 | 2.44e-08 | 1.21e-06 |
| gwas_cli_emma_hmdp_snps100_missingpheno.out | P_VALUE | 100 | 12 | 0 | 1.66e-08 | 4.92e-08 |
| gwas_cli_nullfit.kv.txt | bed_snps1000 | 2 | 0 | 0 | 0 (bit-identical) | 0 |
| gwas_cli_nullfit.kv.txt | bed_snps50_covfile | 2 | 0 | 0 | 0 (bit-identical) | 0 |
| gwas_cli_nullfit.kv.txt | bed_snps50_missingpheno | 2 | 0 | 0 | 0 (bit-identical) | 0 |
| gwas_cli_nullfit.kv.txt | bed_snps50_reml | 2 | 0 | 0 | 0 (bit-identical) | 0 |
| gwas_cli_nullfit.kv.txt | bed_snps50_removeMissing | 2 | 0 | 0 | 0 (bit-identical) | 0 |
| gwas_cli_nullfit.kv.txt | emma_hmdp_snps1000 | 2 | 0 | 0 | 0 (bit-identical) | 0 |
| gwas_cli_nullfit.kv.txt | emma_hmdp_snps1000_cov | 2 | 0 | 0 | 0 (bit-identical) | 0 |
| gwas_cli_nullfit.kv.txt | emma_hmdp_snps100_missingpheno | 2 | 0 | 0 | 0 (bit-identical) | 0 |
| gwas_cli_nullfit.kv.txt | tped_snps50 | 2 | 0 | 0 | 0 (bit-identical) | 0 |
| gwas_cli_tped_snps50.out | BETA | 50 | 0 | 0 | 1.01e-15 | 3.18e-14 |
| gwas_cli_tped_snps50.out | BETA_SD | 50 | 0 | 0 | 3.47e-17 | 1.22e-15 |
| gwas_cli_tped_snps50.out | F_STAT | 50 | 0 | 0 | 3.4e-14 | 3.17e-14 |
| gwas_cli_tped_snps50.out | P_VALUE | 50 | 0 | 0 | 1.1e-14 | 7.31e-14 |
| hmdp_K_npdump.txt | all | 8836 | 0 | 0 | 0 (bit-identical) | 0 |
| kinship_bed_snps2000_calc.block100.txt | all | 10000 | 0 | 0 | 5.77e-15 | 5.63e-14 |
| kinship_bed_snps2000_calc.diag_Kv.txt | diag | 1219 | 0 | 0 | 5.77e-15 | 3.64e-15 |
| kinship_bed_snps2000_calc.diag_Kv.txt | K.dot(cos) | 1219 | 0 | 0 | 3.46e-14 | 1.79e-13 |
| kinship_bed_snps2000_calc.tail100.txt | all | 10000 | 0 | 0 | 6.88e-15 | 8.64e-14 |
| kinship_cli_bed_snps2000.block100.txt | all | 10000 | 0 | 0 | 2e-15 | 3.36e-14 |
| kinship_cli_bed_snps2000.diag_Kv.txt | diag | 1219 | 0 | 0 | 1.11e-15 | 1.19e-15 |
| kinship_cli_bed_snps2000.diag_Kv.txt | K.dot(cos) | 1219 | 0 | 0 | 2.46e-14 | 2.04e-13 |
| kinship_cli_bed_snps2000.tail100.txt | all | 10000 | 0 | 0 | 2.44e-15 | 4.37e-14 |
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

Worst abs / rel per file. For eigenvalue files, rel is over entries > 1e-6.
All four columns were regenerated with the fixed script on the current branch.

| file | macOS py3.10 abs / rel | macOS py3.13 abs / rel | macOS py3.14 abs / rel | linux/amd64 py3.13 abs / rel |
|---|---|---|---|---|
| assoc_hmdp_GWAS.txt | 1.4e-07 / 1e-05 | 8.3e-08 / 5.6e-06 | 8.3e-08 / 5.6e-06 | 2.4e-07 / 9.4e-06 |
| assoc_hmdp_imputed.txt | 1.2e-08 / 1.9e-06 | 1.2e-08 / 1.9e-06 | 1.2e-08 / 1.9e-06 | 4.1e-08 / 6.5e-06 |
| bed_readme_kin_eigenvalues.txt | 1.6e-13 / 7.7e-15 | 1.6e-13 / 7.7e-15 | 1.6e-13 / 7.7e-15 | 1.1e-13 / 2.3e-14 |
| example_py.kv.txt | 2.5e-07 / 5.4e-08 | 5.1e-07 / 5.4e-08 | 5.1e-07 / 5.4e-08 | 1.3e-06 / 2e-07 |
| gwas_cli_bed_snps1000.out | 5.6e-14 / 4.7e-13 | 5.6e-14 / 4.7e-13 | 5.6e-14 / 4.7e-13 | 8.4e-14 / 3.5e-13 |
| gwas_cli_bed_snps50_covfile.out | 3.5e-14 / 7.4e-14 | 3.4e-14 / 7.5e-14 | 3.4e-14 / 7.5e-14 | 5e-14 / 1.1e-13 |
| gwas_cli_bed_snps50_missingpheno.out | 2.9e-14 / 7.8e-14 | 2.8e-14 / 7.7e-14 | 2.8e-14 / 7.7e-14 | 2.9e-14 / 4.8e-14 |
| gwas_cli_bed_snps50_refit.out | 1.5e-07 / 4.3e-07 | 1.5e-07 / 4.3e-07 | 1.5e-07 / 4.3e-07 | 1.4e-07 / 4.2e-07 |
| gwas_cli_bed_snps50_reml.out | 3.4e-14 / 7.3e-14 | 3.4e-14 / 7.3e-14 | 3.4e-14 / 7.3e-14 | 5e-14 / 1.1e-13 |
| gwas_cli_bed_snps50_reml_refit.out | 8e-08 / 3.3e-07 | 8e-08 / 3.3e-07 | 8e-08 / 3.3e-07 | 1.5e-07 / 1.8e-06 |
| gwas_cli_bed_snps50_removeMissing.out | 1.9e-07 / 6.4e-07 | 1.9e-07 / 6.4e-07 | 1.9e-07 / 6.4e-07 | 5.7e-08 / 2.4e-07 |
| gwas_cli_emma_hmdp_snps1000.out | 1.2e-08 / 1.9e-06 | 1.2e-08 / 1.9e-06 | 1.2e-08 / 1.9e-06 | 4.1e-08 / 6.5e-06 |
| gwas_cli_emma_hmdp_snps1000_cov.out | 7.1e-08 / 2.4e-05 | 7.4e-08 / 2.5e-05 | 7.4e-08 / 2.5e-05 | 4.2e-07 / 0.00014 |
| gwas_cli_emma_hmdp_snps100_missingpheno.out | 2.4e-08 / 1.2e-06 | 2.4e-08 / 1.2e-06 | 2.4e-08 / 1.2e-06 | 1.2e-08 / 6e-07 |
| gwas_cli_nullfit.kv.txt | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| gwas_cli_tped_snps50.out | 3.4e-14 / 7.3e-14 | 3.4e-14 / 7.3e-14 | 3.4e-14 / 7.3e-14 | 5e-14 / 1.1e-13 |
| hmdp_K_npdump.txt | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| kinship_bed_snps2000_calc.block100.txt | 5.8e-15 / 5.6e-14 | 5.8e-15 / 5.6e-14 | 5.8e-15 / 5.6e-14 | 0 / 0 |
| kinship_bed_snps2000_calc.diag_Kv.txt | 3.5e-14 / 1.8e-13 | 3.5e-14 / 1.8e-13 | 3.5e-14 / 1.8e-13 | 1.3e-14 / 5.1e-14 |
| kinship_bed_snps2000_calc.tail100.txt | 6.9e-15 / 8.6e-14 | 6.9e-15 / 8.6e-14 | 6.9e-15 / 8.6e-14 | 0 / 0 |
| kinship_cli_bed_snps2000.block100.txt | 2e-15 / 3.4e-14 | 2e-15 / 3.4e-14 | 2e-15 / 3.4e-14 | 0 / 0 |
| kinship_cli_bed_snps2000.diag_Kv.txt | 2.5e-14 / 2e-13 | 2.5e-14 / 2e-13 | 2.5e-14 / 2e-13 | 1.1e-14 / 6.4e-14 |
| kinship_cli_bed_snps2000.tail100.txt | 2.4e-15 / 4.4e-14 | 2.4e-15 / 4.4e-14 | 2.4e-15 / 4.4e-14 | 0 / 0 |
| kinship_cli_emma_hmdp_snps1000.kva.txt | 2.5e-14 / 6.6e-13 | 2.5e-14 / 6.6e-13 | 2.5e-14 / 6.6e-13 | 1.1e-14 / 7.1e-13 |
| kinship_cli_emma_hmdp_snps1000.txt | 5.3e-15 / 2.2e-13 | 5.3e-15 / 2.2e-13 | 5.3e-15 / 2.2e-13 | 0 / 0 |
| kinship_hmdp_snps1000.txt | 7.8e-15 / 2.4e-13 | 7.8e-15 / 2.4e-13 | 7.8e-15 / 2.4e-13 | 0 / 0 |
| kinship_hmdp_snps1000_centered.txt | 1e-14 / 2.5e-13 | 1e-14 / 2.5e-13 | 1e-14 / 2.5e-13 | 0 / 0 |
| misc_fitTwo.kv.txt | 6.8e-09 / 1.5e-08 | 6.8e-09 / 1.5e-08 | 6.8e-09 / 1.5e-08 | 1.3e-08 / 3e-08 |
| null_fit_hmdp_LLgrid.txt | 3e-13 / 5.9e-14 | 3e-13 / 5.9e-14 | 3e-13 / 5.9e-14 | 9.2e-13 / 1.8e-13 |
| null_fits.kv.txt | 1.4e-08 / 2.9e-08 | 1.4e-08 / 2.9e-08 | 1.4e-08 / 2.9e-08 | 2.3e-08 / 5.1e-08 |

- macOS py3.10: python 3.10.19, numpy 2.2.6, scipy 1.15.3; regenerated in 40.4s
- macOS py3.13: python 3.13.9, numpy 2.5.4, scipy 1.18.1; regenerated in 41.0s
- macOS py3.14: python 3.14.4, numpy 2.5.4, scipy 1.18.1; regenerated in 38.7s
- linux/amd64 py3.13: python 3.13.11, numpy 2.5.4, scipy 1.18.1; regenerated in 76.6s

The linux column was run in `ghcr.io/astral-sh/uv:python3.13-bookworm-slim`
under emulated linux/amd64 (OpenBLAS, `OPENBLAS_NUM_THREADS=1`), as CI would.
Its worst per-SNP case is the EMMA `--emmaCOV` run: 4.2e-7 absolute on
F_STAT, and 1.4e-4 relative on BETA/F_STAT of a near-zero beta (BETA max abs
9.4e-9). Both are well inside `ATOL_FIT` (5e-6), and the test passes there.

## (c) Environment

| | py2 baseline (goldens) | py3 dev machine | CI-equivalent check |
|---|---|---|---|
| OS / arch | linux/amd64 (Docker, emulated) | macOS 15.7.2 arm64 | linux/amd64 Docker (`ghcr.io/astral-sh/uv:python3.13-bookworm-slim` / `python3.10-bookworm-slim`, emulated) |
| Python | 2.7.18 | 3.13.9 (also 3.10.19, 3.14.4) | 3.13.11, 3.10 (lower bounds) |
| numpy | 1.16.6 | 2.5.4 (2.2.6 on 3.10; 1.22.0 at the lower bound) | 2.5.4; 1.22.0 |
| scipy | 1.2.3 | 1.18.1 (1.15.3 on 3.10; 1.9.0 at the lower bound) | 1.18.1; 1.9.0 |
| BLAS/LAPACK | OpenBLAS (wheel), 1 thread | Apple Accelerate (numpy 2.x wheels); OpenBLAS at the lower bound | OpenBLAS (wheel), `OPENBLAS_NUM_THREADS=1` |
| tools | — | uv 0.12.19, pytest 9.1.1 (7.0.0 at the lower bound), hatchling | same lock |

## Test results

`uv run pytest` (slow tests skipped unless noted), on this branch after the
rebase onto PR #5:

| env | result | wall time |
|---|---|---|
| macOS, py3.13 (`--locked`) | 32 passed, 2 skipped | 40 s |
| macOS, py3.13 `--runslow` | 34 passed | 93 s |
| macOS, py3.10 | 32 passed, 2 skipped | 37 s |
| macOS, py3.14 | 32 passed, 2 skipped | 34 s |
| macOS, py3.10, **lower bounds** (`--resolution lowest-direct`: numpy 1.22.0, scipy 1.9.0, pytest 7.0.0) | 32 passed, 2 skipped | 44 s |
| macOS, py3.10, numpy 1.22.0 / 1.23.5 with scipy 1.8.0 or 1.8.1 | segfault (exit 139) | — |
| linux/amd64 Docker, py3.13 (emulated, `--locked`) | 32 passed, 2 skipped | 69 s |
| linux/amd64 Docker, py3.10, **lower bounds** (`lowest-direct`: numpy 1.22.0, scipy 1.9.0) | 32 passed, 2 skipped | 60 s |
| **py2.7 baseline image, ported code** (`docker/py27/test.sh`) | 32 passed, 2 skipped | 61 s |

- **Lower bounds.** `numpy>=1.22`, `scipy>=1.9` are the oldest versions
  tested. scipy 1.8.x has py3.10 macOS arm64 wheels but crashes the suite
  there, which is why the scipy bound was raised from 1.8. CI runs the
  lowest-direct resolution on ubuntu py3.10.
- **py2.7 baseline.** The ported code still runs on py2.7 with numpy 1.16.6
  and passes every test there. So the port's edits do not change the
  numbers, and every py3 difference in (b) comes from the numerical stack.
  The code is not *meant* to stay py2-compatible; it just happens to be
  today.
- Before the rebase, three tests failed on this branch (both EMMA CLI tests
  against the old zero-K goldens, and the BED kinship row sums at
  `ATOL=1e-12` on macOS/Accelerate). PR #5 fixed both at the source: the
  EMMA goldens were regenerated from a header-less K, and the row-sum
  goldens were replaced with `K.dot(cos(arange(n)))`. No tolerance was
  loosened.

## (d) Behaviour changes and newly found bugs

### Behaviour changes

1. **A kinship file with non-numeric content is now an error.** In py2
   (numpy 1.16), `np.fromfile(sep=" ")` stopped at the first non-number with
   only a DeprecationWarning. `K.resize` then zero-filled the matrix, so a K
   file with a `#` header was silently read as an **all-zero kinship**. That
   is how the original EMMA CLI goldens were made; PR #5 found this in review
   and regenerated them from a header-less K. numpy 2.x raises `ValueError`.
   I kept the numpy behaviour, because emulating the zero-K bug would mean
   matching wrong numbers. A plain whitespace kinship, with or without `.gz`,
   parses exactly as before. numpy 1.x (down to the 1.22 lower bound) still
   warns and zero-fills: before the rebase, the linux numpy 1.22.0 run passed
   the EMMA CLI tests against the old zero-K goldens.
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
3. (Resolved by PR #5.) The old BED kinship row-sum goldens
   (`*.diag_rowsums.txt`) could not be checked against `ATOL=1e-12` across
   BLAS builds: they were sums that cancel to about 0. PR #5 replaced them
   with `*.diag_Kv.txt` (diagonal and `K.dot(cos(arange(n)))`).
4. (Bound raised.) scipy 1.8.0 and 1.8.1, the old `scipy>=1.8` lower
   bound, **segfault** in the suite on macOS arm64 with Python 3.10 (with
   numpy 1.22.0 and 1.23.5). This is a problem with early arm64 scipy
   wheels, not a pylmm bug; on linux/amd64 scipy 1.8.0 passed. The bound is
   now `scipy>=1.9`; numpy 1.22.0 + scipy 1.9.0 pass on both platforms.

## Notes for the next phases

- `docker/py27/` stays in the repo for reference and is not run in CI.
  `just golden` regenerates the goldens from whatever code is in the working
  tree. To reproduce the Phase 1 baseline, check out the baseline code first.
- Ruff, type annotations and the snake_case rename are left to the planned
  hygiene phase.
