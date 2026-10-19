# S5: edge cases and robustness: Phase 5 target list

Code uses the post-5.4 API; other numbers are as of Phase 4.

Study code: `sims/studies/s5_edge_cases.py`. Full table (161 cases, one row each, worst
outcome over 6 replicates): `sims/results/s5_edge_cases/summary.md` / `summary.csv`.
Reproduce: `just sim s5_edge_cases --jobs 6` then `just sim-report s5_edge_cases`
(`--quick`: 22 cells, about 5 s).

pylmm as on `phase4-sims-framework` was run unchanged. Nothing here modifies `pylmm/`.

## What was run

Each case builds a small problem (n = 100 individuals in 20 sibships of 5, so that h is
identifiable; K from 400 SNPs, 20 test SNPs, h2 = 0.5, no causal SNPs) and runs it through
four paths:

| path | what |
|---|---|
| `lib` | `lmm.LMM(...).fit()` plus `lmm.gwas(...)` on raw genotypes (missing calls are removed per SNP) |
| `lib_impute` | the caller mean-imputes first, then `lmm.gwas` |
| `cli_impute` | `python -m pylmm.cli.gwas` through a subprocess on BED/fam/phenos/kin files written by the study. This is the CLI default (the BED reader mean-imputes and standardizes) |
| `cli_remove` | the same with `--removeMissingGenotypes` |

For each case it records: exception type and message, nan/inf outputs, warnings, runtime,
and accuracy against the independent dense oracle (`sims/oracle.py`): `|h - h_oracle|` for
the null REML fit and the relative error of t for up to 6 SNPs (the oracle refits h on the
same rows pylmm uses, REML). SNPs where the oracle's own h is at the boundary (>= 0.98) are
skipped, because pylmm's 0.99 clamp (bugs 2 and 3) would otherwise swamp everything.

Each case declares its *ideal* behaviour (`result`, `result_or_error`, `warn_or_error`,
`error_or_nan`) and the outcome is judged against that:

| outcome | meaning |
|---|---|
| OK | right answer, or nan/error where that is right |
| silent | garbage-in accepted with no warning |
| nan | unexpected nan/inf on valid input |
| crash | exception (type and message recorded). Medium if the input was invalid, high if valid |
| wrong | finite numbers that disagree with the oracle, or finite numbers where the statistic is not estimable |

Severity: critical = believable wrong numbers, high = crash or wrong on valid input or a
documented-bug mismatch, medium = unclear error / spurious `t ~ 0` / silent on garbage
input, low = warnings only.

### Overall outcome counts (161 cases)

| outcome / severity | cases |
|---|---|
| OK, no issue | 101 |
| OK with warnings only | 2 |
| wrong, critical | 11 |
| wrong, high | 11 |
| wrong, medium (spurious `t ~ 0` where nan is right) | 14 |
| crash, high (valid input) | 4 |
| crash, medium (invalid input, unclear message) | 15 |
| silent, medium | 3 |

### What is fine

These all agree with the oracle (h to <= 0.01, t to 1e-6 or better) with no crash:

- Genotype missing rates 0, 1, 5, 20, 50% and phenotype missing rates 0, 5, 20%, on `lib`,
  `lib_impute` and `cli_impute` (the CLI default and the library agree with the oracle).
  The one exception is `cli_remove`, item 4 below.
- K from 2, 5 or 20 SNPs (rank 2 to 20), K = I (h is unidentifiable but t is right), K with
  duplicated individuals (pairs; and 5 clone groups of 20), K scaled by 1e-4 and 1e4.
- Up to 80 covariates at n = 100; covariates scaled by 1e-8 / 1e8 (right answers, but
  `LinAlgWarning: ill-conditioned`).
- Phenotype scaled by 1e-6 / 1e6, offset by 1e9, binary 0/1, t(2) heavy tails, lognormal,
  only 3 observed phenotypes, a y of `inf` (all-nan output, no crash).
- Monomorphic SNPs, SNPs monomorphic after dropping missing genotypes, a single minor allele,
  and a SNP equal to the intercept all give nan on every path.

## Phase 5 robustness targets (prioritized)

Reproducer prelude (library items):

```python
import numpy as np
from pylmm.lmm import LMM, calculate_kinship, gwas
rng = np.random.default_rng(0)
n = 40
K = calculate_kinship(rng.binomial(2, .3, (n, 200)).astype(float))
y = rng.standard_normal(n)
X = rng.binomial(2, .4, (n, 5)).astype(float)       # 5 test SNPs
```

The exact cell for each item is in `summary.csv` (`family`, `variant`, `path`); the problem
builders are `_b_<family>` in `s5_edge_cases.py`.

### P0: silent wrong or meaningless numbers

**1. `--kfile` of the wrong size is accepted silently** (bug 16). Severity critical.
(`io/kfile_big`, `io/kfile_small`, 6/6 replicates.)
- Reproducer: `pylmmGWAS.py --bfile d --phenofile d.phenos --kfile K.txt out` where d has 100
  individuals and K.txt is 95 x 95 (or 105 x 105).
- Observed: exit code 0 and a full result table. `np.fromfile` reads whatever is there and
  `K.resize((n, n))` truncates or zero-pads it. A kinship file with a `#` header line instead
  raises an unrelated `ValueError: string or file could not be read to its end due to unmatched
  data`.
- Expected: a clear error.
- Fix: raise if the number of values read is not n*n, naming the file, the count read and n.
  Also reject header lines with a specific message.

**2. SNPs that are not estimable get a finite t (about 0, p about 1) instead of nan.** Severity
medium to critical (measured |t| <= 1e-6 here, but the sign of the failure is "believable
number"). (`degenerate/mono_dosage`, `mono_ydrop`, `single_minor_ydrop`, `eq_cov`,
`eq_cov_x2`; frequency 17% to 100% depending on rounding.)
- Reproducers:
  - `x[:] = 0.37` (a constant dosage): `gwas(y, X_with_const_col, K)` gives `t = 6e-9`, not
    nan (n = 30, 50 and 100 all give finite values; n = 200 gives nan).
  - A SNP that equals a covariate (`X0 = [1, c]`, `x = c`): `t = 2e-8`, with a
    `LinAlgWarning`.
  - CLI: a SNP that varies only among individuals whose phenotype is `NA`, or has one
    minor allele in an individual whose phenotype is `NA`. The BED reader standardizes using
    all individuals, the CLI then drops the `NA` rows, and `x.var() == 0` is false for a
    column that is constant up to rounding.
- Observed: exact-zero variance tests (`x.var() == 0` in `lmm.gwas` and in the CLI complete-SNP
  branch) pass on columns that are constant up to rounding. Whether you get nan or `t ~ 1e-8`
  depends on the constant, n and BLAS rounding. The missing-genotype branch of the CLI uses
  `<= 1e-6` instead (bug 13), so the two CLI branches also disagree.
- Expected: nan.
- Fix: one shared helper `snp_estimable(x, X0)` used by `gwas` and both CLI branches: after
  dropping missing rows, project x on X0 (QR) and return nan if the residual variance is below
  `1e-10 * var(x)` or the remaining df is below 1. This covers constants, SNP = covariate and
  df <= 0 in one place.

**3. Saturated or over-parameterized designs (n <= number of columns + 1) return finite t.**
Severity critical. (`smalln` with k + 2 >= n at n = 10, 20, 50; every replicate.)
- Reproducer: `gwas(y, X, K, X0=np.column_stack([np.ones(n), rng.standard_normal((n, n - 1))]))`
  for n = 40 gives `[nan nan nan ...]`, but with `n - 3` covariate columns it gives
  `[-5.04, 0.56, -0.24, 0.38, -2.19]` with 1 residual df, and with `n - 2` columns it raises
  `BracketError` (item 7).
- Observed: `RuntimeWarning: divide by zero` / `LinAlgWarning`, and 8 to 20 of 20 SNPs get a
  finite t with 0 or negative df.
- Expected: `ValueError("n=%d observations, but the model has %d columns")`.
- Fix: check `n_obs > q + 1` (q = covariates including the intercept) in `LMM.__init__` and
  `gwas`, after dropping missing y.

**4. CLI `--removeMissingGenotypes` fits the per-SNP null by ML, the complete-SNP branch by
REML** (bug 1). Severity high. (`missing/mcar/cli_remove`, all 7 cells with
genotype-missing >= 1% and replicates mostly 6/6; relative t error 2% to 10%, h itself is right to 5e-4.)
- Reproducer: any BED with a few missing genotypes; run with and without
  `--removeMissingGenotypes` and compare to the oracle. The t of SNPs with missing calls match
  an ML-refit oracle, not the REML one.
- Observed: `Ls.fit(REML=options.REML)` (default False) in the missing branch, `L.fit()`
  (REML) in the complete branch. `lib` `gwas` always uses REML for both. The library and the
  CLI therefore disagree on the same data.
- Expected: one convention (REML for the null fit) in both branches.
- Fix: pass the same `REML` to every `fit`. Needs a decision on the default; the goldens in
  `tests/golden` encode the current behaviour for the complete-SNP branch.

**5. A wrongly ordered, asymmetric or non-PSD K is accepted with no warning.** Severity high
for ordering, medium for the rest. (`kin/wrong_order`, `upper_only`, `nonpsd`, `asym`.)
- Reproducer: `gwas(y, X, K[perm][:, perm])`. With sibship structure h changes by up to 0.78
  and the correlation of t with the right-order result drops below 0.9. No exception, no
  warning. pylmm has no way to detect this; the CLI kfile has no individual IDs at all.
- Also: `K = np.triu(K)` is read as the lower triangle only (so K becomes diagonal); a
  non-PSD K (noise 0.3 gives min eigenvalue about -0.2) is silently clamped to 1e-6
  (`LMM.__init__`), which shifts h by more than 0.1 in 1 of 3 replicates; a K with a nan raises
  `ValueError: array must not contain infs or NaNs` from scipy with no context.
- Expected: for ordering, an ID check where IDs exist; for the rest a warning or error.
- Fix: add optional `indiv_ids` to the library API and compare against K's IDs; in `LMM.__init__`
  validate finite, symmetric (`allclose(K, K.T)`), and warn with the most negative eigenvalue
  and the number clamped (the existing "Cleaning %d eigen values" message only prints under
  `verbose` and counts `< 0`, not the `< 1e-6` that are really clamped).

**6. Constant phenotype.** Severity critical when it returns numbers. (`pheno/constant`, lib
and CLI.)
- Reproducer: `LMM(np.full(10, 3.0), np.eye(10)).fit()` returns `h = 0.99`, `sigma = 2e-31`,
  `LL = 304.9`. With a realistic K it either raises `BracketError` or returns finite, very
  large or tiny t depending on rounding in `Yt - X beta`.
- Expected: `ValueError("phenotype has zero variance")`.
- Fix: check `np.var(y[~isnan(y)]) > 0` (and finite, so `inf` is caught too) in `LMM.__init__`.

### P1: crashes on valid input

**7. `BracketError` in the null fit (Brent leaves [0, 1]).** Severity high. (`smalln`
n = 20, 3 columns: 1/6 replicates; also n = 10 with 9 columns; fuzz with unrelated individuals:
2/300 at n = 10, 0/300 at n >= 20.)
- Reproducer: `rng = np.random.default_rng(1177); W = rng.binomial(2, rng.uniform(.05, .5, 100),
  (10, 100)).astype(float); K = calculate_kinship(W); y = rng.standard_normal(10)`, then
  `LMM(y, K).fit()` raises `scipy.optimize._optimize.BracketError`. In that case the grid
  maximum is at h = 0.96 with a nearly flat profile up to 0.99.
- Observed: `RuntimeWarning: invalid value encountered in log`, then the exception. A
  maximum near the top of the grid makes `brent` step to h > 1, where
  `h*Kva + (1-h)` is negative and the log is nan (bugs 2 to 4).
- Expected: a result (h at the boundary).
- Fix: replace `optimize.brent` by `minimize_scalar(method="bounded")` on [0, 1], make `getMax`
  scan the whole grid (bug 2), and return nan-safe LL. This also fixes the 0.99 vs 1.0
  disagreement with the oracle that shows up as `max_abs_dh` of 0.01 at small n.

**8. An all-missing SNP kills the whole scan** (bug 11). Severity high. (`degenerate/allmiss`:
`lib`, `lib_impute`, `cli_impute`, 6/6.)
- Reproducer: `X[:, 1] = np.nan; gwas(y, X, K)` raises `LinAlgError: A singular matrix
  detected` (after `RuntimeWarning: Degrees of freedom <= 0`). CLI: `IndexError: boolean index
  did not match indexed array along axis 0; size of axis is 0 ...` (`normalizeGenotype`
  returns an empty array).
- Expected: nan for that SNP, scan continues.
- Fix: use the shared estimability helper of item 2 before building the per-SNP LMM, and make
  `normalizeGenotype` return an all-nan (or all-zero) column of length n instead of `G[x]`.

**9. Collinear, duplicate or zero covariates give a bare `LinAlgError`.** Severity medium.
(`cov/collinear_dup`, `collinear_comb`, `intercept_dup`, `zero_cov`, lib and CLI, 6/6.)
- Reproducer: `gwas(y, X, K, X0=np.column_stack([np.ones(n), c, c]))`, or a covariate file
  with a column of ones (the CLI adds its own intercept unless `--noMean`, so this is easy to
  do by accident).
- Observed: `numpy.linalg.LinAlgError: A singular matrix detected: slice(s) [0] are singular`
  from `linalg.inv(XX)` in `getMLSoln`; no mention of covariates.
- Expected: either an error that names the problem or the redundant column dropped. SNP
  results with a full-rank reference design match the oracle, so dropping is feasible.
- Fix: check `np.linalg.matrix_rank(X0) == X0.shape[1]` in `LMM.__init__`; raise
  `ValueError("covariates are collinear (rank r of q); remove column(s) ...")` and, for the
  CLI, point at `--noMean` when an all-ones column is present.

**10. CLI input-file handling.** Severity medium (bugs 10 and 14 and one new message).
- Pheno file missing one fam individual: `IndexError: boolean index did not match indexed
  array along axis 0; size of axis is 100 but size of corresponding boolean axis is 99`
  (`getPhenos` silently skips the individual, bug 10). With a stranger added in its place the
  lengths match and the phenotypes would be silently shifted; this study only observed the
  length-mismatch crash.
- No `.phenos` and no `--phenofile`: `TypeError: stat: path should be string, bytes, os.PathLike
  or integer, not NoneType` (bug 14). A usage error was intended.
- A pheno file in a different order than the fam file, extra individuals in the pheno file, and
  `-9` coding all work.
- Fix: `getPhenos` should raise `ValueError` listing fam IDs absent from the pheno file; fix
  the `os.path.isfile(None)` guard.

**11. Unvalidated nan/inf input in the library.** Severity medium. nan in K or in a covariate:
`ValueError: array must not contain infs or NaNs` (scipy, no context; the CLI handles the
covariate case with a proper `parser.error`). All-nan y: `LinAlgError` from `LMM`; `gwas`
itself returns all-nan arrays, but as a numpy array instead of the list it normally returns
(bug 15). `y = inf` yields all-nan output with no warning.
- Fix: validate finite y (apart from nan), finite K and X0 at entry; one return type.

### P2: smaller or already known

**12. Per-SNP refit at h near 1 (bugs 2 and 3).** Severity low. Where the oracle's h is
>= 0.98 pylmm returns 0.99 (those SNPs are excluded from the t comparison above). One
replicate in six at 20% genotype missingness and 20% phenotype missingness still shows a
relative t error of 2e-3 against the oracle, and `max_abs_dh = 0.01` at n = 10. Covered by
item 7's fix.

**13. Ill-conditioned covariates.** Severity low. Covariates scaled 1e-8 or 1e8 give correct
answers (matches the oracle to 1e-7) but emit `LinAlgWarning: An ill-conditioned matrix
detected`. Standardizing X0 columns internally would remove the warning.

**14. In-place mutation** (bug 12 and `calculateKinship`). Severity low, but it silently
corrupts callers that reuse arrays. Verified: `calculateKinship(W)` overwrites W with the
imputed, standardized values, and `LMM(y, K, Kva, Kve)` clamps the caller's `Kva`.
Fix: copy inside both functions.

**15. `inf` and other non-finite results give no warning.** See item 11.

## Notes on interpretation

- Frequencies in `frac` are over 6 replicates and are only a guide. Several failures depend on
  floating-point rounding (items 2 and 6) and on the draw (item 7).
- `wrong` with `n_finite_unexp > 0` counts SNPs for which the oracle design is rank deficient
  or has no residual df. The returned `t` there is almost always about 0 (|t| <= 1e-6), which
  is why those cases are rated medium; items 1, 3 (partly), 4, 5 and 6 produce numbers that
  are plausible-looking and not close to 0.
- The harness runs `gwas` and a standalone `LMM.fit` separately, so a crash can be reported
  from the null fit even though `gwas` itself returns arrays (all-nan y).
- CLI cases pass the individual SNP t from the F_STAT column; h is parsed from the `-v`
  line, which prints three decimals (tolerances account for this).
- Runtime: every library cell takes under 0.2 s and a CLI cell about 0.7 s (mostly Python and
  scipy start-up). No case showed a runtime problem at n = 100.
- Cost of the study: quick (22 cells) about 5 s. Full grid, 161 cells x 6 replicates =
  978 tasks: sum of task time 510 to 640 s (measured at `--jobs 2` on a busy laptop, wall
  255 to 325 s), so about 2 to 3 min at `--jobs 6`.
