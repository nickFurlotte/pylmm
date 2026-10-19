# Changelog

All notable user-visible changes to pylmm. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Entries read "old -> new". "Default CLI output" (the GWAS output file of `pylmmGWAS` without `--refit`,
`--removeMissingGenotypes` or `--ML`) is unchanged by every entry below.

## [Unreleased]

### Changed

#### pylmmGWAS (CLI)
- REML is the default everywhere. `--refit` and `--removeMissingGenotypes` used to refit each SNP by ML
  while the null fit was REML; they now refit by REML, so their outputs change (`--refit`: max |dt| 0.24 on
  the 50-SNP golden).
- `--REML` is accepted and does nothing. New `--ML` uses ML for the null fit, every per-SNP fit and the
  test, with sigma^2 = Q/n. Passing `--REML` and `--ML` together is a usage error (exit code 2).
- `--kfile2` removed (it raised a NameError).
- `--eigen` reads a `.npz` file (written by the new `pylmmKinship --eigenNpz`) or text `B.kva`/`B.kve`
  (written by `pylmmKinship -e`; pass the base `B`). The old binary `B.Kva`/`B.Kve` files can no longer be
  read. The decomposition is checked against `--kfile`.
- Phenotype and covariate files (`--phenofile`, `--covfile`, `.phenos`) are matched to the genotype
  individuals by (FID, IID) instead of by row position. Individuals absent from the file become missing.
  File order no longer matters, extra individuals in the file are ignored. EMMA files still match by
  position and are length-checked. A first line whose first two fields are `FID IID` (any case) is a header
  and is skipped.
- Individuals with a missing covariate are dropped (the same as a missing phenotype); this used to be a
  fatal error.
- Messages about dropped individuals are plain stderr lines, `pylmmGWAS: warning: ...`, not Python
  `UserWarning`s with a source line. They are printed whenever the count is greater than 0, not only
  under `-v`: individuals not in the phenotype/covariate file, individuals with a missing phenotype,
  individuals with a phenotype but a missing covariate, and `--eigen` ignored because individuals were
  dropped. With `-v`, the number of SNPs that are nan because the covariates lost rank on the SNP's
  non-missing individuals is printed once at the end.
- SNPs that cannot be estimated give nan (all four value columns) instead of t ~ 1e-8 or a crash: constant
  SNPs (also constant up to rounding), SNPs collinear with a covariate, SNPs with no residual degree of
  freedom, and all-missing SNPs. One estimability check serves both the imputing and the
  `--removeMissingGenotypes` branches. `pylmmKinship` also survives an all-missing SNP.
- New usage errors (exit code 2): no phenotype file; a `--covfile` that does not exist; `-p` out of range;
  a constant phenotype; rank-deficient, duplicate or all-zero covariates (the message names the columns and
  hints at `--noMean`, since the CLI adds its own intercept); n <= q + 1; a nan/inf or wrong-shaped kinship
  matrix; duplicate IDs, no matching ID, malformed rows or non-numeric values in the phenotype/covariate files.
  Each used to give a traceback or a finite but meaningless result.

#### pylmmGWAS: kinship file (`--kfile`, PR #8)
- `readKinshipFile` reads plain text or `.gz`, line by line, with any whitespace. It raises an error naming
  the file for any non-numeric token (including a `#` header), for anything other than exactly n rows of n
  values, and for nan/inf. The CLI reports it as a usage error with exit code 2. It warns, without failing,
  for an asymmetric K. Before, depending on the numpy version, a headered file was read as an all-zero K and a
  wrong-size file was silently zero-filled or truncated.

#### Heritability search (PR #8)
- `getMax` refines each grid peak with `scipy.optimize.minimize_scalar(method="bounded")` inside the
  bracketing grid interval, instead of an unbounded Brent search. On scipy >= 1.11 the old search stepped
  past h = 1 and raised `BracketError`, aborting the whole GWAS (high-h^2 traits). The null fit, the
  per-SNP refit and `misc.fitTwo` all use it. `LL_brent` also guards h > 1 (as well as h < 0). Results agree
  with the previous ones to < 3e-7. Unchanged on purpose: the 0.99 grid cap, the `getMax` off-by-one and
  first-not-best (deferred to Phase 6.3).

#### Library
- `GWAS(REML=False)` is a real ML scan; it used to return the REML t. `association(REML=False)` uses
  sigma^2 = Q/n (it was Q/(n-q)), so its t and betaVar change. `LMM.fit()`'s reported sigma under ML is
  still Q/(n-q) (kept for the `-v` output).
- `GWAS` always returns lists (the type used to vary).
- New kwargs: `ids`/`K_ids` on `GWAS` and `LMM` (the individuals of Y and of K must match, in order), and
  `validate` on `LMM`.
- `LMM` and `GWAS` raise `ValueError` for inf or constant Y, nan/inf or wrong-shaped K or Kva/Kve,
  rank-deficient or nan/inf X0 (the message names the columns), and n <= q + 1 (these used to give a bare
  `LinAlgError` or a finite t). They warn for an asymmetric or non-PSD K. Covariate rows are checked after
  dropping the individuals with a missing Y, so X0 may be nan where Y is nan.
- `GWAS`, for SNPs with missing calls whose covariates lose rank on the non-missing individuals: nan, and
  one summary `UserWarning` with the count instead of nothing.
- New public helpers: `snp_estimable`, `subset_testable` (was the private `_subsetTestable`),
  `check_covariates`, `check_ids`, `ESTIMABLE_RTOL`. `snp_estimable` compares against the centred norm when
  X0 spans the intercept, so a SNP with a big constant offset is not flagged.
- `calculateKinship` no longer changes the caller's W in place, and `LMM` no longer clamps the caller's Kva.
- `plink.normalizeGenotype`: an all-missing SNP is returned as a length-n nan vector (it used to be empty);
  a SNP that is constant up to rounding is returned as zeros.
- `plink.getPhenos`/`getCovariates`: rows aligned by (FID, IID), nan for absent individuals, with a
  `UserWarning` giving the count; a missing covariate file raises `ValueError`; new `getPhenosEMMA`; EMMA files
  are length-checked.

### Removed
- `pylmmGWAS --kfile2`.

### Internal
- Goldens `gwas_cli_bed_snps50_refit.out` and `gwas_cli_bed_snps50_removeMissing.out` were regenerated for the
  REML refit (py2 originals in `tests/golden/py2_original/`); `generate_golden.py` now refuses to run under
  Python 3.
