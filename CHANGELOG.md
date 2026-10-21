# Changelog

All notable user-visible changes to pylmm. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Entries read "old -> new". "Default CLI output" (the GWAS output file of `pylmm gwas`, formerly `pylmmGWAS`,
without `--refit`, `--remove-missing-genotypes` or `--ml`) is unchanged by every entry below, except by the batched tests of
[Unreleased]: last-digit differences for well-conditioned covariates, up to a few 1e-6 relative with a badly
scaled covariate (where the new numbers are the more accurate); see there. Entries written
before the single `pylmm` command use the command and flag names of their time ([`docs/migration/cli-map.csv`](docs/migration/cli-map.csv)
maps them).

## [Unreleased]

### Added
- Variant metadata: `PlinkReader.variants` -> the `.bim` columns of every SNP (`chrom`, `snp`, `cm`, `bp`,
  `a1`, `a2`) as a `pylmm.genotypes.Variants`: read-only numpy columns (text as Python `str` objects, `cm`
  float64, `bp` int64); `variants[k]` is a `Variant` named tuple. Before, the reader kept only the SNP ID.
  For a TPED file the columns come from its `.bim`, or from its `.map` (4 columns, or 3 without cM; no
  alleles); EMMA input has SNP names (`SNP_<k>`) only. The file is read on first use, leniently: fields after
  the 6th (`.bim`) or 4th (`.map`) are ignored, a cM that is not a number (`NA`) is nan, a position written
  as an integral float (`100.0`, `1.5e2`) is that integer, and any other non-integer position is -1 with one
  logged warning (file, first line, count). A `.bim` line with fewer than 6 fields, a `.map` line whose width
  differs from the first line's, or a blank line raises `InputError` naming the file and line, when the
  metadata (or `iter_blocks`) is first needed; `iter_snps` does not read these columns and is unchanged.
- Blocks of SNPs: `PlinkReader.iter_blocks(block_size=1024)` yields `(G, variants)`, `G` float64
  `(len(indivs), b)`, bit for bit the vectors of `iter_snps` (standardised or not), for the individuals and
  `norm_genotype` of the time of the call. A BED file is decoded a block at a time: with the `.bim` read in
  both, 2.3-6.0x the throughput of `iter_snps` unstandardised and 1.0-1.6x standardised (block sizes 256 to
  4096, full and `--quick` runs of `sims/bench/bed_blocks.py`). `iter_snps` and the CLI output are
  unchanged.
- `pylmm.genotypes`: the `GenotypeSource` protocol (`sample_ids`, `num_samples`, `variants`,
  `norm_genotype`, `iter_blocks`), implemented by `PlinkReader` and by the new `ArraySource`, which wraps an in-memory
  `(n, m)` genotype matrix (e.g. read from VCF, BGEN or VCF Zarr with another library) with optional IDs and
  metadata. `normalize_genotype` is now also a function there (`PlinkReader.normalize_genotype` calls it).
- `sims/bench/`: Phase 6 benchmarks; `python -m sims.bench.bed_blocks [--quick]`.
- `LMM.association_batch(X, h=None, reml=True)`: the Wald test of `association(..., return_beta=True)` for
  every column of `X` at once; returns `AssociationResult(t, p, beta, beta_var)`, arrays.
- `gwas(..., batch=True, block_size=1024)`: `batch=False` -> one SNP at a time, the 1.0.0 code path;
  a block takes about 6-8 arrays of `n * block_size` floats. The default is `pylmm.genotypes.DEFAULT_BLOCK_SIZE`
  (the block size of `iter_blocks`), and `block_size` is checked with `check_block_size` (a non-integer such
  as 1.5 or True is a `ValueError`).
- `pylmm gwas --block-size N` (SNPs read and tested together, default 1024) and `--no-batch` (the 1.0.0
  one-SNP-at-a-time path).
- `pylmm.lmm.snp_estimable_columns(X, X0)`: `snp_estimable` for every column of a block at once, with exactly
  its decisions.
- `sims/bench/gwas_batch.py` (`just bench gwas_batch [--quick]`): batched vs per-SNP wall time and
  differences, synthetic and on the bundled BED file; `--cli-baseline DIR` also times the `pylmm gwas` of
  another checkout (e.g. an earlier revision) and checks that its output is byte-identical (every run
  imports `pylmm` from its own tree, which is checked).
- `pylmm.kinship`, the new home of `calculate_kinship` (`from pylmm.lmm import calculate_kinship` still
  works), with named kinships: `calculate_kinship(W, kind=...)` and `pylmm kinship --kinship-type ...`,
  `standardized` (the default; 1.0.0's formula, with the convention change below), `centered` (centred,
  not scaled: GEMMA `-gk 1`'s formula, on 0/1/2 counts in the CLI, i.e. twice the reader's 0/0.5/1 values;
  an `--emma-geno` file must be coded 0/0.5/1, and values above 1 are warned about) and `ibs`, EMMA's
  identity-by-state kinship (`emma.kinship(snps, method="additive", use="all")`, Kang et al. 2008:
  genotypes 0 / 0.5 / 1, heterozygotes split into two 0/1 codings, missing calls at the SNP mean, diagonal
  1), checked against a transliteration of EMMA's R code and against EMMA's output from R on a fixture.
  `pylmm kinship` writes bit for bit the `calculate_kinship` of the same raw genotypes (for `centered`, of
  twice them) at the same block size (`--block-size` defaults to the library's `KINSHIP_BLOCK_SIZE`, 1000).
  `standardized` is GEMMA `-gk 2`'s formula only on complete genotypes: GEMMA standardises after mean
  imputation (1.0.0's library convention) and by default drops SNPs with MAF < 0.01 or missingness > 0.05
  before both `-gk 1` and `-gk 2`.
- `kinship_scale(K)` -> `KinshipScale(mean_diag, centered_mean_diag)` (`trace(K)/n`, `trace(PKP)/(n-1)`), the
  scale of the h fitted with K (PVE = `h c / (h c + 1 - h)`); `pylmm kinship -v` logs both, and the number
  of SNPs used. `KinshipAccumulator`: build any kind one SNP at a time; it sums the blocks in place (two
  n x n arrays plus an n x block buffer; `tracemalloc` peak of `calculate_kinship` at n = 3000, 2500 SNPs:
  172 MB standardized / centered, 206 MB IBS).
- `calculate_kinship(..., legacy=True)` and `pylmm kinship --legacy-kinship`: the 1.0.0 kinships, byte for
  byte (checked on the bundled BED and HMDP data and a synthetic BED with missing calls and constant SNPs,
  against v1.0.0 and the phase6 base, at block sizes 1000 and 300).
- `sims/studies/s6_kinship_kinds.py` (`just sim s6_kinship_kinds [--quick]`): h2 recovery and null
  calibration of the three kinds on a synthetic RI-like inbred panel (`sims.genotypes.inbred_panel`, a new
  genotype source) and on the bundled HMDP strains; findings in `s6_kinship_kinds.md`.
- GCTA binary GRMs, read and written natively (`pylmm.kinship_io`): `read_grm(prefix)` -> `(K, ids)` from
  `prefix.grm.bin` (float32 lower triangle, row-major, with the diagonal) and `prefix.grm.id`;
  `write_grm(prefix, K, ids, n_snps)` writes those and `prefix.grm.N.bin`; `read_grm_counts`. File sizes
  are checked against the `.grm.id` (an `InputError` naming the expected size, with a hint for a float64
  triangle or a square matrix). `pylmm gwas --grm PREFIX` (GCTA's `--grm`) reads one in place of
  `--kinship`, aligned by ID like a kinship with a `.id` file; `pylmm kinship --out-format grm -o PREFIX`
  writes one (`.grm.N.bin`: the `m` K was divided by, for every pair; not for EMMA input, which has no IDs).
  A GRM is float32: K is rounded to about 7 significant digits, so a scan on it differs from the scan on
  the text kinship by rounding (synthetic data: max |Δt| 1.3e-7, relative Δp 3.2e-7). Checked against
  files written by plink2 v2.0.0-a.7.13 `--make-grm-bin` (`tests/data/plink2/`): pylmm reads them and
  rewrites them byte for byte. `PlinkReader.read_kinship` also reads a `.grm.bin`.
- plink2 `--make-rel` matrices, read only: `kinship_io.read_rel` reads `X.rel` / `X.rel.gz` (text, lower
  triangle or square), `X.rel.bin` (float64 or `bin4` float32, square or triangle) with `X.rel.id` (a
  `#FID IID` or `#IID` header; no FID -> `0`, as plink2 writes to a `.fam`). `pylmm gwas --kinship X.rel`
  (or `.rel.gz` / `.rel.bin`) and `PlinkReader.read_kinship` use it. A plink2 `.rel` used to fail (the
  `#FID IID` header was read as a sample, a triangle as a ragged matrix). plink2's text `.rel` holds 6
  significant digits (`%g`), less than a float32 GRM (scans: |Δt| up to 8e-6 vs 1.2e-7 against its float64
  `.rel.bin`): prefer `--make-rel bin`. A kinship whose FIDs are all `0` (plink2, no FID column) that
  matches nothing under `--id-match fid-iid` gets an error suggesting `--id-match iid`.
- `kinship_io.read_eigen` / `write_eigen`: eigendecomposition `.npz` files with sample IDs.

### Changed
- `pylmm kinship --eigen-npz`: arrays `Kva`, `Kve` -> also `fid`, `iid` (the samples of Kve's rows; not for
  EMMA input) and `kinship_type` (plain arrays, no pickle; 1.0.0 readers still find `Kva` and `Kve`).
- `pylmm gwas --eigen`: the decomposition was used only if no individual was dropped or reordered, its rows
  assumed to be the kinship's -> with sample IDs (the new `.npz`, or `PATH.id` / `PATH.grm.id` next to
  `--eigen-text`'s `PATH.kva`/`.kve`) they are matched to the tested individuals: one missing is a usage
  error (it is another kinship's decomposition), a permutation is reordered and used (a kinship realigned by
  its IDs no longer forces a recomputation), and extra individuals (dropped since) make it ignored and
  recomputed, as before. A file without IDs (1.0.0's) is used as before, with a warning. The `--eigen` file
  is now read (and must exist) even when it is then ignored.
- `pylmm gwas`: `--kinship` was required -> one of `--kinship` and `--grm` is (`one of the arguments
  --kinship --grm is required`). With `--kinship` the output is unchanged.
- `normalize_genotype` (per-SNP standardisation) extracts the called values once: ~40% faster, results
  bit for bit unchanged for one SNP (1-d input, the documented use; 2-d input also gives the 1.0.0 result,
  tested on a few arrays).
- Kinship normalisation (bug 8; `calculate_kinship` and `pylmm kinship`, default `standardized`): two
  conventions for missing calls -> one, the reader's (each SNP centred and scaled with the mean and SD of its
  called genotypes, missing calls then set to the mean), so the library and the CLI agree; and `m` = all
  SNPs, including skipped constant and all-missing ones -> `m` = the SNPs used. Effect: with missing calls
  the library's K moves to the CLI's (mean diagonal about 1 - the missing rate: bundled BED data 1.000 ->
  0.987, HMDP strains 0.925 -> 0.979); with constant SNPs both grow by 1 / (the share used) (HMDP 7.5%
  constant: CLI mean diagonal 0.905 -> 0.979). `pylmm kinship` on the bundled BED file (no constant SNPs) is
  byte-identical to 1.0.0. h moves with the scale (HMDP, 20 transcripts, library: median h 0.440 -> 0.425,
  PVE 0.423 both); t and p do not move for a pure rescaling (the CLI's change), and move slightly with the
  library's missing-call convention (the two 1.0.0 conventions differed by max |Δt| 0.03 on HMDP, study S4). 1.0.0's kinships: `legacy=True`, `--legacy-kinship`.

#### Batched association tests (faster scans)
- `gwas()` and `pylmm gwas` (without `--refit`) test the SNPs without missing calls in blocks: one matrix
  product rotates a block of SNPs into the eigenbasis of K, and each SNP's GLS fit is computed in closed form
  against the null model, instead of a matrix-vector product and a small matrix inverse per SNP. One SNP at
  a time -> blocks of 1024 SNPs. Speedup (single-threaded BLAS, `sims/bench/gwas_batch.py`, an 8-core
  arm64 laptop): `gwas()` 7.7-8.6x at n = 500 (20,000 SNPs) and 17-19x at n = 2000 (10,000 SNPs);
  `pylmm gwas` on 20,000 SNPs of the README data 9.1 s -> 2.6 s (3.5x, the whole command, kinship reading
  included); one BLAS thread per core gave the same times.
- Same model, same h, same statistics, computed in a different floating-point order. For well-conditioned
  covariates the results agree with 1.0.0 to rounding: in the benchmark (single-threaded run) t and beta at most 4.7e-13
  absolute (4.6e-10 relative, for values near 0), p 3.3e-13 absolute (1.9e-12 relative), se 1.8e-14
  relative; in the tests at most 9.5e-11 relative for values >= 1% of the largest; nan in the same places. On the README
  data (first 20,000 SNPs) the output files differ only in the last printed digits (17 of 20,000 lines byte-identical; max relative difference 1.6e-11 in BETA and
  F_STAT, 2.9e-14 in P_VALUE); the goldens
  pass at their existing tolerances.
- With a badly scaled covariate next to the intercept (a raw date such as 20240131, a value like 1e4 +- 1, a
  year column) 1.0.0's per-SNP path, which inverts `X'SX`, loses precision: old and new results can differ by
  up to a few 1e-6 relative (a year column 2000-2020: 3e-10 to 2e-8 in our checks). The new values are the
  more accurate ones (against an accurate reference: 1e-10 vs 3e-6 off for a date ~2e7; a 40-digit reference gave 3e-11 vs 3e-7 for
  1e4 + N(0, 1)). Centre or scale such covariates (`(c - c.mean()) / c.std()`): the model and its tests are
  the same and both paths are then accurate.
- Which SNPs are tested (nan for constant, collinear or untestable SNPs) is unchanged: the block check makes
  exactly the decisions of `snp_estimable`.
- SNPs with missing calls (tested on their called individuals) and `refit=True` / `--refit` still test one SNP
  at a time.
- `pylmm gwas` writes each result line, in file order, as soon as it is known. If a SNP's test fails, every
  SNP before it has its line and no later one does, as in 1.0.0 (byte for byte with `--no-batch`); if the
  reader fails mid-file, the SNPs read before the failure are tested and written first. No SNP is tested
  twice and no line is written twice; on Ctrl-C or a failed write the lines already written stay, and the
  SNPs of the current block read but not yet written get none.
- `pylmm gwas` and `pylmm kinship` read the genotypes a block at a time with `PlinkReader.iter_blocks`
  (a BED file is decoded block-wise) instead of one SNP at a time; SNP IDs come from the block's
  `variants.snp` (`SNP_<k>` for EMMA). Output files are byte-identical to before at the same
  `--block-size` (`pylmm gwas`, and with `--no-batch` still 1.0.0's; `pylmm kinship`, which still sums
  `W W'` over blocks of `--block-size` kept SNPs). `pylmm gwas` on 20,000 SNPs of the README data
  (single-threaded BLAS, median of 7, `sims/bench/gwas_batch.py --cli-baseline`): 2.51 s -> 2.25 s
  (1.1x; reading the genotypes 0.61 s -> 0.35 s, most of it now the per-SNP standardisation);
  `pylmm kinship` on the same SNPs 2.09 s -> 2.01 s.
- A block is read whole, so after a data or I/O error of the reader inside a block (`InputError`, e.g.
  a truncated BED file or a bad TPED or EMMA line, or `OSError`) `pylmm gwas` reads the rest of the file
  one SNP at a time from that block's first SNP (the SNPs already written are skipped unstandardised):
  the SNPs before the error are written, then the error is reported, as before. An error that does not
  recur (transient) is logged as a warning and the scan completes. Any other exception of the block
  reader is raised as is. A `.bim` that `variants` rejects (a line with fewer than 6 fields) is read one
  SNP at a time, with a warning at the start, by both commands, as before.

## [1.0.0] - 2026-10-19

The Python 3 port of pylmm, with a new PEP 8 API and a single `pylmm` command line, validated against the
original Python 2 outputs, simulation studies and two published mouse GWAS.

### Changed (breaking)

#### Python and dependency floors
- Python >= 3.10 -> >= 3.11 (3.10 reaches end of life in October 2026); tested on 3.11-3.14.
- numpy >= 1.22 -> >= 1.23.2 and scipy >= 1.9 -> >= 1.9.2, the oldest releases with Python 3.11 wheels; CI
  runs the suite at exactly these versions on Python 3.11 (lowest-direct).

#### Command line: one `pylmm` command
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

#### Python API cleanups
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

#### Python API: PEP 8 names
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

#### Replication follow-ups: data provenance, inbred panels, comparison with published GWAS
- `data/README.md`: what each bundled file is, its origin (the HMDP EMMA rows are, in order, the rows of the
  Broad merged mouse HapMap/WT file, build 37 / mm9; verified: same 132,285 rows, 122,242 informative rows
  with 0 discordances across 27 shared strains), the strain list (`AXB-19/PgnJ` three times, identical
  columns = the MPD sublines AXB19a/19/19b), the fake phenotypes of the BED demo set, and SHA-256
  checksums of every file (checked by `tests/test_data_provenance.py`).
- `data/hmdp.liver.snps.map.tsv` (4.4 MB): a validated marker map for `data/hmdp.liver.snps` (row,
  chromosome, build-37 position, the bases coded 0 and 1, validation status), and
  `scripts/build_hmdp_marker_map.py`, which downloads the Broad file, checks its SHA-256 and the row count,
  and refuses to write a map with a discordant row.
- Skill `inbred-strain-panels`: strain aliases and a genotype-identity audit of duplicate strain labels,
  joining phenotype sources to strain genotypes, strain → animal expansion, imputing over strains vs
  animals, PLINK-safe IDs, placeholder alleles, unequal replicates, the EMMA strain-mean path, and a sketch
  of the Farber 2011 / Bennett 2010 reanalyses.
- `skills/interpret-gwas-results/comparison-worksheet.md`: a template for comparing a pylmm scan with a
  published GWAS (kinship formula and scale, unit, transform, null-only vs per-SNP refit, build, named
  allele, SNP set, thresholds, a pre-registered rule for "replicated"), with an IBS-style kinship recipe.
- `plink_tools.sanitize_id` / `sanitize_ids`: PLINK-safe tokens for names such as `BTBR T+ tf/J`, checked
  for collisions, with the map back to the original names.
- `plink_tools.write_bed`: `None` (or `""`) in `a1`/`a2` for a marker without allele letters -> the
  placeholders `1` (A1) / `2` (A2, the counted allele).

#### Agent-facing docs
- `AGENTS.md` (and a `CLAUDE.md` that imports it): repo map, `just` commands, test and golden rules,
  conventions, commit norms, the skills and the known caveats.
- `docs/guide.md`: a user guide for ad-hoc analysis with the library (loading data and ID alignment,
  kinship, the null fit and h² scaling, single-SNP tests, scans with covariates, missing data, the CLI,
  reading the output and per-allele effects, performance, pitfalls, validated behaviour).
- `skills/` (Agent Skills format; `.claude/skills` is a relative symlink to it): `run-gwas` (with
  `reference.md`: phenotype QC and transforms, SNP QC, loci),
  `estimate-heritability`, `custom-association-analysis`, `simulate-lmm-data`,
  `convert-genotype-formats` (with `reference.md` and `scripts/plink_tools.py`: read `.bim`, write and
  subset BED files, write phenotype and kinship + `.id` files; `write_bed` refuses non-genotype values
  instead of rounding, `coding="fraction"` for 0/0.5/1), `migrate-from-old-pylmm` (with
  `scripts/migrate.py`, which rewrites old Python code and `pylmmGWAS`/`pylmmKinship` command lines from
  the migration maps) and `interpret-gwas-results` (with `scripts/summarize_gwas.py`: λ_GC, top hits,
  per-allele effects, window clumping into loci, Manhattan/QQ plots). `tests/test_skill_scripts.py`
  tests the scripts.
- `tests/test_docs.py` runs every `python`/`bash` code block of those files (blocks of a file share a
  namespace and a scratch directory; `skip-ci: <reason>` opts a block out, `needs=<module>` skips it when
  a module is missing; a block in an unknown language is an error; the pytest summary lists the counts).
  `just docs-test` runs it alone.

#### Sample IDs and alignment
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

#### Command line
- `pylmm gwas --id-match {fid-iid,iid}`: match phenotype, covariate and kinship rows on FID and IID (the
  default) or on the IID alone (PLINK input only).
- `pylmm gwas --pheno-name NAME`: test the phenotype column of that name (from a `FID IID name ...` header);
  `--covar-name NAME ...` (space- or comma-separated): use only those covariate columns.
- `pylmm gwas --help` and `pylmm kinship --help` describe every option and end with examples.
- The command line is type-checked by mypy too (`just typecheck`, Python 3.11 and 3.14).

#### Types and docstrings
- Type hints on the library modules (`pylmm.lmm`, `pylmm.plink_io`, `pylmm.misc`; the CLI modules followed
  with the single `pylmm` command), checked by mypy in CI (`just typecheck`), and a `py.typed` marker so type checkers use them.
- numpydoc docstrings with runnable examples for the public API; pytest runs them as doctests
  (`--doctest-modules`).
- `pylmm.__version__`, the installed package version.
- `PlinkReader.N` is also set for EMMA input (the number of genotype columns); `indivs` is documented as
  `(FID, IID)` pairs, or column positions `0..n-1` for EMMA, which has no IDs.

### Changed

#### License
- GNU AGPL v3 -> MIT.

#### pylmmGWAS: kinship IDs
- `--kfile K` reads the ID file `K.id` if it exists and aligns the kinship to the genotype file by
  (FID, IID): genotype individuals that are not in K are dropped (from the phenotype, the covariates and
  every SNP vector) and K's extra individuals are ignored, each with a `pylmmGWAS: warning:` line listing
  them. Without `K.id` it warns once and assumes, as before, that K's rows are the genotype individuals in
  file order (not for EMMA input, which has no IDs). The shipped example kinship has a `.id` file, so the
  README example prints no such warning; its output file is unchanged.
- When K drops individuals, genotypes are imputed and standardised over the kept individuals, but
  individuals dropped for a missing phenotype or covariate still count in the mean and SD (max |dt| 0.024
  between the two rules; making them consistent is planned).
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
  first-not-best (known, not yet fixed).

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

#### Replication follow-ups
- `pylmm.__version__` was `"unknown"` when pylmm was imported from a source checkout on `PYTHONPATH`
  without being installed -> the `[project] version` of the checkout's `pyproject.toml`.
- `plink_tools.write_bed` / `write_pheno` / `write_kinship` wrote IDs with whitespace (e.g. a strain name
  as FID), which corrupts the whitespace-separated PLINK files -> `ValueError` naming the ID and pointing
  at `sanitize_ids`; empty IDs and equal `a1`/`a2` for a marker are errors too.

#### `PlinkReader`
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

#### `PlinkReader` (found while adding type hints)
- TPED: an allele code other than `0`/`1`/`2` (e.g. an un-recoded nucleotide letter) in a homozygous pair
  silently reused the previous individual's genotype, or raised `UnboundLocalError` for the first individual
  -> `ValueError` naming the SNP and the code (recode with `plink --recode12`). Valid 1/2-coded files read as
  before.
- `normalize_genotype` mean-imputed the caller's genotype array in place -> the imputation is done on a copy.
- `get_covariates(None)` raised `TypeError` -> `ValueError`; `cov_file` is now a required argument.
- An unknown genotype `type` wrote a message to stderr and `get_snp_iterator()` returned `None` -> the
  constructor raises `ValueError`.

### Docs
- README rewritten for 1.0 (what pylmm does, how it is validated, install, quick start, agents, migration);
  new `CITATION.cff` for the software.
- `CONTRIBUTING.md`, `SECURITY.md` (private vulnerability reports; 1.0.x supported) and a bug report
  issue template.
- `data/README.md`: a licence and citation section. The example data is third-party research data, not
  covered by the MIT licence of the code; it says which sources to cite and that the 1219-animal BED's
  origin is undocumented. The README's license section points to it.
- Package metadata: description, Production/Stable and Python 3.11-3.14 classifiers, keywords, project
  URLs. The sdist ships the package, `sims/` (without result tables), `skills/`, the notebook and the docs
  they link to (including `data/README.md`, the example data's provenance and licence), not the example
  data or the tests; the wheel holds `pylmm/` only.
- `docs/replication.md`: the reanalysis of two published HMDP mouse GWAS by an agent working from
  the docs alone (Farber 2011 BMD: partially replicated, all 5 published loci have a nearby signal (median −log10 p
  0.47 below published), 1/5 pass Bonferroni and 3/5 the paper's threshold; Bennett 2010 HDL:
  replicated, same lead SNP, −log10 p 11.5 vs 9.4), method choices and caveats; linked from the guide
  (section 13) and the README.
- Kinship without a `.id` file, stated exactly (AGENTS.md, guide, `run-gwas`, `migrate-from-old-pylmm`,
  `pylmm gwas --help`): `pylmm gwas --bfile/--tfile` warns and assumes genotype order; `--emma-geno` does
  not warn (EMMA input has no IDs; an existing `.id` is ignored with a warning); the library's
  `PlinkReader.read_kinship` does not warn.
- AGENTS.md: `uv sync` / `uv run` create and use `.venv/` at the repository root; how to use another
  environment (`UV_PROJECT_ENVIRONMENT`, or an activated venv with `pip install -e .`).
- A per-animal BED of an inbred panel is mean-imputed over animals by `pylmm gwas`, not over strains
  (Δ−log10 p up to 0.28 in the replication); the library recipe to impute over strains is in
  `inbred-strain-panels`. A strain-aware option is planned.
- `convert-genotype-formats/reference.md`: a handoff template for downloads a site blocks to scripts
  (Dryad's landing page loads but `file_stream` returns 403: download in a browser, verify checksums).
- `estimate-heritability` / `custom-association-analysis`: expand strain genotypes to animals by row
  indexing, since `Z @ W` spreads a missing call to every animal.

### Internal
- `sims` runner: the `--jobs` cap 6 -> the CPU count minus 2 (at least 1); `--no-job-cap` still lifts it.
- Goldens `gwas_cli_bed_snps50_refit.out` and `gwas_cli_bed_snps50_removeMissing.out` were regenerated for the
  REML refit (py2 originals in `tests/golden/py2_original/`); `generate_golden.py` now refuses to run under
  Python 3.

[Unreleased]: https://github.com/nickFurlotte/pylmm/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/nickFurlotte/pylmm/releases/tag/v1.0.0
