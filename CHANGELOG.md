# Changelog

All notable user-visible changes to pylmm. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Entries read "old -> new". "Default CLI output" (the GWAS output file of `pylmm gwas`, formerly `pylmmGWAS`,
without `--refit`, `--remove-missing-genotypes` or `--ml`) is unchanged by every entry below. Entries before
Phase 5b.2 use the command and flag names of their time ([`docs/migration/cli-map.csv`](docs/migration/cli-map.csv)
maps them).

## [Unreleased]

### Changed (breaking)

#### Command line: one `pylmm` command (Phase 5b.2)
A clean break, with no aliases for the old commands or flags; no number changes: for an equivalent invocation
the output files (GWAS results, kinship, `.id`, `.kva`/`.kve`) are byte-identical to before.
[`docs/migration/cli-map.csv`](docs/migration/cli-map.csv) maps every old command and flag to the new one (or
says why it is gone).
- `pylmmGWAS` -> `pylmm gwas`, `pylmmKinship` -> `pylmm kinship`, one console script `pylmm` (argparse
  subcommands), also run as `python -m pylmm`; new `pylmm --version`. The `pylmmGWAS` / `pylmmKinship`
  console scripts and the `scripts/pylmmGWAS.py` / `scripts/pylmmKinship.py` shims are removed
  (`scripts/example.py` stays).
- The output file is the required option `-o/--out FILE`, not a positional argument.
- kebab-case flags, named after PLINK where it has one: `--kfile` -> `--kinship` (required), `--phenofile`
  -> `--pheno`, `--covfile` -> `--covar`, `--noMean` -> `--no-intercept`,
  `--removeMissingGenotypes` -> `--remove-missing-genotypes`, `--ML` -> `--ml`, `--REML` -> `--reml`
  (still the default), `--emmaSNP` / `--emmaPHENO` / `--emmaCOV` -> `--emma-geno` / `--emma-pheno` /
  `--emma-covar`; kinship: `--emmaNumSNPs` -> `--emma-num-snps`, `-n` -> `--block-size`, `-e/--efile VALUE`
  (value ignored) -> the flag `--eigen-text`, `--eigenNpz` -> `--eigen-npz`. `--bfile`, `--tfile`,
  `--eigen`, `--refit` are unchanged.
- **The phenotype column number is 1-based now**: `-p N` (0-based) -> `--pheno-col N+1`, like plink2
  `--pheno-col-nums`, GCTA `--mpheno` and GEMMA `-n` (`-p 0`, the default, is `--pheno-col 1`; `-p 1` is
  `--pheno-col 2`). `--pheno-col 0` is a usage error, and so is a number above the number of phenotype
  columns. A script that only renames the flag tests the next column instead of the intended one.
- `pylmm kinship --emma-num-snps` is optional: the SNPs of `--emma-geno` are counted (non-blank lines),
  and a given value that differs from the count is a usage error (exit 2). A wrong `--emmaNumSNPs` used to
  compute K from the first M SNPs (too small) or crash with StopIteration (too large).
- `python -m pylmm.cli.gwas` / `python -m pylmm.cli.kinship` (the old invocation) is a tombstone, not an
  alias: it prints `pylmm: error: python -m pylmm.cli.gwas was removed; use pylmm gwas ...` and exits 2.
  (Run as before, the new modules did nothing and exited 0.)
- argparse checks the combinations: exactly one of `--bfile`/`--tfile`/`--emma-geno`; `--reml` or `--ml`;
  `--pheno` or `--emma-pheno` (`--emma-pheno` used to override); `--covar` or `--emma-covar` (`--covfile`
  used to win); `--pheno-name` or `--pheno-col`; `-v` or `-q`; `--emma-num-snps` only with
  `--emma-geno`; `--log-file` not the `-o` file. Integer options are type-checked.
- Exit codes: 0 success; 2 for usage and input errors (missing or malformed files, unknown columns,
  unwritable output), reported as one `pylmm: error: ...` line, also for an input error found while reading
  the SNPs (a bad TPED allele code, a truncated BED or TPED file, a ragged EMMA row); 1 for an unexpected
  error, reported as `pylmm: error: unexpected error: ...`. No traceback unless `--debug`. Input errors no longer print the
  usage text (argparse errors still do). `pylmm gwas` without arguments is a usage error (exit 2); it used
  to print the help and exit 0. The `--kinship` file and the writability of `-o` / `--eigen-npz` are
  checked before any work; a missing genotype file names its option (`--bfile X: [Errno 2] ...`).
- Messages go through the `logging` module (logger `pylmm`) to stderr as `pylmm: <level>: <message>`:
  `pylmmGWAS: warning: ...` -> `pylmm: warning: ...`, and the `-v` progress lines are `pylmm: info: ...`
  (the null fit line reads `null model: heritability=%0.3f, sigma=%0.3f`). Default level WARNING; `-v`
  INFO; new `-q` ERROR; new `--log-file FILE` (timestamped, at least INFO); new `--debug` (DEBUG messages,
  tracebacks, and the library's own `LMM(verbose=True)` diagnostics, which `-v` used to print).
  `warnings.warn` calls of the library are logged as warnings. `pylmm kinship`'s "skipped N SNPs with no
  called genotype" is a logged warning, not a `UserWarning`.
- `pylmm.plink_io` logs "Read N individuals from ..." and "Reading kinship matrix from ..." at INFO (logger
  `pylmm.plink_io`) instead of writing them to stderr, so library users no longer see them by default.
- `pylmm.cli`: `main(argv)` (the `pylmm` entry point) returns the exit code instead of exiting, so it can run
  in-process; the per-command `pylmm.cli.gwas.main` / `pylmm.cli.kinship.main` are gone (`add_parser`,
  `run` and the typed steps `load_inputs`, `fit_null_model`, `scan` replace them). `read_kinship_file` and
  `read_eigen_file` are unchanged; `_read_aligned_kinship` -> `read_aligned_kinship`. `main` sets
  process-global logging for the duration of each call, so it is not thread-safe.
- `pylmm.samples.InputError` (also `pylmm.plink_io.InputError`), a `ValueError` subclass, is raised by the
  file readers of `pylmm.samples` and `pylmm.plink_io` for a malformed or inconsistent input file (as
  `ValueError` was), so `except ValueError` still works. Argument errors stay plain `ValueError`. New
  checks: a TPED line must have 4 + 2n fields, an EMMA genotype row n values, and a `.fam`/`.tfam` line
  FID and IID; blank lines of an EMMA genotype file are skipped.
- `pylmm kinship -v` logs "Processing first N SNPs" every 10000 SNPs (each block at `--debug`), and
  `--block-size` is listed under "performance" in `--help`. The constant-phenotype error prints the value
  with `%g` (`equal 1`, not `equal np.float64(1.0)`).

#### Python API cleanups (Phase 5b.1)
A clean break, with no aliases or shims; no number changes (the goldens and the README `pylmmGWAS` output
are byte-identical). [`docs/migration/name-map.csv`](docs/migration/name-map.csv) has a row for each (now
with `phase` and `notes` columns; `kind=signature` rows are changes of types or defaults, not names).
- Return types: `LMM.ml_solution` returns `sigma` and `Q` as floats, `LMM.fit` returns `sigma` (and the
  log-likelihood) as a float, and `LMM.association(return_beta=True)` returns `beta_var` as a float; they
  were (1, 1) arrays. `misc.fit_two`'s `sigma` is a float too. `beta` stays a (q, 1) array.
- `LMM.log_likelihood` returns `LogLikResult(ll, beta, sigma, xx_inv)`, a `NamedTuple` (so
  `ll, beta, sigma, xx_inv = L.log_likelihood(h)` still works; `ll` and `sigma` are floats).
- One `reml` rule: every function and method that takes `reml` defaults to `reml=True`.
  `LMM.log_likelihood` (and the private `_neg_log_likelihood`, `_get_max`) defaulted to ML (`reml=False`);
  pass `reml=False` for the ML log-likelihood.
- `LMM()` and `gwas()`: `kva`/`kve` are `None` (compute the eigendecomposition) or both given. An empty
  sequence (`[]`), which used to mean "compute", raises `ValueError` pointing to `None`, and so does giving
  only one of them.
- `PlinkReader` verbs: `read_*` reads a file, `iter_*` iterates. `get_phenos` -> `read_phenos`,
  `get_covariates` -> `read_covariates`, `get_phenos_emma` -> `read_phenos_emma`, `get_covariates_emma` ->
  `read_covariates_emma`, `get_snp_iterator` -> `iter_snps` (the private `_get_snp_iterator_*` ->
  `_iter_snps_*`). `pylmm.plink_io.IndivKey` -> `pylmm.samples.SampleID`.
- `PlinkReader` genotype vectors have `len(indivs)` values, in the order of `indivs`, also when a kinship
  with IDs (FaSTLMM labels, or the new `.id` sidecar) or `subset()` dropped individuals; they used to keep
  all `N` values (see Fixed). `N` is still the number of individuals in the genotype file.
- `PlinkReader.read_kinship` also stores `self.K`; a plain kinship without IDs must be `len(indivs)` square
  (`ValueError`; it was not checked), and a `k_file + ".id"` sidecar, if present, aligns it by ID.
- `PlinkReader` checks the BED header at construction: a non-SNP-major BED (or a file that is not a BED)
  raises `ValueError` there (see Fixed). Duplicate IDs in the genotype file now raise `ValueError` when
  phenotypes, covariates or a kinship with IDs are matched to it.
- A header line `FID IID name ...` of a phenotype or covariate file must name as many columns as the rows
  have (`ValueError`; it used to be skipped unread).
- `next(reader)` before `iter_snps()` raises `ValueError` (it was an `AttributeError` or a failed assert);
  after the last SNP the files are closed and `next` keeps raising `StopIteration`.

#### Python API: PEP 8 names (Phase 5.4)
- Every Python name of the pylmm API follows PEP 8; a clean break, with no aliases or deprecation shims.
  [`docs/migration/name-map.csv`](docs/migration/name-map.csv) lists each changed name (`old_name`,
  `new_name`, `kind`, `module` = where it lives now): 1 module, 1 class, 15 functions, 25 methods,
  19 attributes and 18 parameters (31 rows counting each function that takes a renamed parameter). The
  rule is in [`docs/conventions.md`](docs/conventions.md).
- `misc.fit_two(y, ..., wgrids=)` -> `fit_two(Y, ..., ngrids=)`; `PlinkReader.getIndivs` -> `_read_fam` (private);
  `fastLMM_kinship_style` -> `fastlmm_kinship`. The py2 `reader.next()` is gone: use `next(reader)`.
- Module `pylmm.input` -> `pylmm.plink_io` (`input` shadowed the builtin), class `plink` -> `PlinkReader`.
- Examples: `lmm.calculateKinship` -> `calculate_kinship`, `lmm.GWAS` -> `gwas`, `LMM.LL` ->
  `log_likelihood`, `LMM.getMLSoln` -> `ml_solution`, `misc.fitTwo` -> `fit_two`,
  `PlinkReader.normalizeGenotype` -> `normalize_genotype`, `getPhenos` -> `get_phenos`,
  `cli.gwas.readKinshipFile` -> `read_kinship_file`; attributes `optH`/`optSigma`/`optLL`/`optBeta` ->
  `opt_h`/..., `Kva`/`Kve` -> `kva`/`kve`, `X0t` -> `x0t`, `LLs` -> `ll_grid`, `H` -> `h_grid`; parameters
  `REML=` -> `reml=`, `K_ids=` -> `kinship_ids=`, `returnBeta=` -> `return_beta=`, `phenoFile=` ->
  `pheno_file=`, `normGenotype=` -> `norm_genotype=`.
- Helpers that are not public API are now private: `matrixMult` -> `_matrix_mult`, `LL_brent` ->
  `_neg_log_likelihood`, `getMax` -> `_get_max`, the per-format SNP iterators and genotype decoders.
- Math symbols keep their capitals: `K`, `X`, `Y`, `X0`, `K1`, `K2` as parameters, and `LMM.K`, `.Y`, `.X0`,
  `.N`.
- `gwas()` and `LMM()` take `kva=None, kve=None` (they were mutable `[]` defaults); `None` or an empty
  sequence means "compute the eigendecomposition".
- Unchanged: CLI commands, flags and output files (byte-identical), and the `.npz` array names `Kva`/`Kve`
  written by `pylmmKinship --eigenNpz` and read by `pylmmGWAS --eigen`. The entries below use the names of
  the time.

### Added

#### Sample IDs and alignment (Phase 5b.1)
- New module `pylmm.samples`: sample IDs are `(FID, IID)` pairs (`SampleID`); `align_samples()` aligns any
  number of per-sample sources (genotypes, phenotypes, covariates, kinship) by ID and returns an `Alignment`
  (the shared samples in the reference order, the rows of each source, and a report of the samples
  dropped from each source and why); `match_index()`, `read_sample_table()` (header-aware
  `FID IID v1 ...` tables), `read_id_file()` / `write_id_file()` / `read_kinship_ids()`. Matching is on
  (FID, IID), or on the IID alone with `match="iid"`. Duplicate IDs are an error.
- `PlinkReader(id_match="iid")` matches phenotype, covariate and kinship rows on the IID alone.
- `PlinkReader.read_phenos(..., columns=)` / `read_covariates(..., columns=)` select columns by name (from a
  `FID IID name ...` header) or by 0-based index; the names are in `pheno_names` / `covariate_names`
  (None for a headerless file, which still works as before).
- `PlinkReader.subset(ids)` keeps (and orders) individuals in `indivs`, `phenos`, `K` and the genotype
  vectors together; `indivs_removed` is always set.
- `PlinkReader` is a context manager (`with PlinkReader(...) as r:`) and has `close()`; `iter_snps()` closes
  the files of the previous iteration, and the end of an iteration closes them too.
- `LMM.association`, `LMM.fit(X=...)` and `LMM.log_likelihood(X=...)` accept a 1-D (n,) SNP as well as an
  (n, 1) column (same numbers); a wrong shape is a `ValueError` naming the expected shape.
- `pylmmKinship OUT` also writes `OUT.id`, the (FID, IID) of each row of the kinship, tab-separated (GCTA
  `.grm.id` style; not for EMMA input, which has no IDs). The kinship file itself is unchanged.

#### Command line (Phase 5b.2)
- `pylmm gwas --id-match {fid-iid,iid}`: match phenotype, covariate and kinship rows on FID and IID (the
  default) or on the IID alone (PLINK input only).
- `pylmm gwas --pheno-name NAME`: test the phenotype column of that name (from a `FID IID name ...` header);
  `--covar-name NAME ...` (space- or comma-separated): use only those covariate columns.
- `pylmm gwas --help` and `pylmm kinship --help` describe every option and end with examples.
- The command line is type-checked by mypy too (`just typecheck`, Python 3.10 and 3.14).

#### Types and docstrings (Phase 5.5/5.6)
- Type hints on the library modules (`pylmm.lmm`, `pylmm.plink_io`, `pylmm.misc`; the CLI modules followed in
  Phase 5b.2), checked by mypy in CI (`just typecheck`), and a `py.typed` marker so type checkers use them.
- numpydoc docstrings with runnable examples for the public API; pytest runs them as doctests
  (`--doctest-modules`).
- `pylmm.__version__`, the installed package version.
- `PlinkReader.N` is also set for EMMA input (the number of genotype columns); `indivs` is documented as
  `(FID, IID)` pairs, or column positions `0..n-1` for EMMA, which has no IDs.

### Changed

#### pylmmGWAS: kinship IDs (Phase 5b.1)
- `--kfile K` reads the ID file `K.id` if it exists and aligns the kinship to the genotype file by
  (FID, IID): genotype individuals that are not in K are dropped (from the phenotype, the covariates and
  every SNP vector) and K's extra individuals are ignored, each with a `pylmmGWAS: warning:` line listing
  them. Without `K.id` it warns once and assumes, as before, that K's rows are the genotype individuals in
  file order (not for EMMA input, which has no IDs). The shipped example kinship has a `.id` file, so the
  README example prints no such warning; its output file is unchanged.
- When K drops individuals, genotypes are imputed and standardised over the kept individuals, but
  individuals dropped for a missing phenotype or covariate still count in the mean and SD (max |dt| 0.024
  between the two rules; making them consistent is a Phase 6 item).
- Individuals are matched by FID and IID; matching by IID alone is available in the library
  (`PlinkReader(id_match="iid")`), and on the command line since 5b.2 (`--id-match iid`).
- `PlinkReader.read_kinship` ignores, with a warning, a `.id` next to a kinship for EMMA genotypes, as
  pylmmGWAS does. `pylmmKinship` on EMMA input removes a stale `OUT.id` (with a note on stderr).
- `--eigen` is ignored (with a warning) when K was reordered or subset by its IDs, as when individuals are
  dropped for a missing phenotype.
- The kinship is read before the covariates (only the order of the `-v` messages changes).

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

### Fixed

#### `PlinkReader` (Phase 5b.1)
- Silent misalignment: a FaSTLMM kinship (`fastlmm_kinship=True`) that lacked some individuals dropped them
  from `indivs`, `K` and `phenos`, but every SNP vector kept all `N` values, so genotype row `i` no longer
  belonged to individual `i` -> the SNP vectors are subset and reordered the same way (and standardised
  over the kept individuals). Regression test
  `tests/test_plink_io.py::test_fastlmm_kinship_that_drops_individuals_subsets_the_snp_vectors`.
- TPED: a half-missing allele pair (`0 1`, `2 0`) was read as heterozygous (0.5) -> missing (nan), as in
  PLINK. Output changes only for such malformed files.
- A BED file that is not SNP-major made `get_snp_iterator()` write to stderr and raise a bare
  `StopIteration`, which escaped a `for` loop as an exception -> `ValueError` at construction, with the
  PLINK command to rewrite it. A file without the BED magic number is a `ValueError` too.
- File handles: calling `get_snp_iterator()` again left the previous handles open until the reader was
  garbage-collected -> `iter_snps()` closes them first.

#### `PlinkReader` (Phase 5.5, found while adding type hints)
- TPED: an allele code other than `0`/`1`/`2` (e.g. an un-recoded nucleotide letter) in a homozygous pair
  silently reused the previous individual's genotype, or raised `UnboundLocalError` for the first individual
  -> `ValueError` naming the SNP and the code (recode with `plink --recode12`). Valid 1/2-coded files read as
  before.
- `normalize_genotype` mean-imputed the caller's genotype array in place -> the imputation is done on a copy.
- `get_covariates(None)` raised `TypeError` -> `ValueError`; `cov_file` is now a required argument.
- An unknown genotype `type` wrote a message to stderr and `get_snp_iterator()` returned `None` -> the
  constructor raises `ValueError`.

### Internal
- Goldens `gwas_cli_bed_snps50_refit.out` and `gwas_cli_bed_snps50_removeMissing.out` were regenerated for the
  REML refit (py2 originals in `tests/golden/py2_original/`); `generate_golden.py` now refuses to run under
  Python 3.
