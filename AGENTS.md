# AGENTS.md

**PyLMM: a modern, readable, validated mixed-model toolkit for statistical genetics, built to be used by
humans and agents.** A linear mixed model (LMM) for GWAS and heritability: `Y = X0 b + x a + u + e`,
`u ~ N(0, h s2 K)`, `e ~ N(0, (1 - h) s2 I)`, with a Wald t test per SNP. Python ≥ 3.11, numpy + scipy.
This file is for coding agents working in this repo or with pylmm; humans should start at
[README.md](README.md) and [docs/guide.md](docs/guide.md).

## Repo map

| path | what |
|---|---|
| `pylmm/lmm.py` | the model: `LMM` (`fit`, `association`, `association_batch`, `fit_snps`, `lrt_batch`, `score_batch`), `gwas` (`test=wald\|lrt`), input checks (re-exports `calculate_kinship`) |
| `pylmm/kinship.py` | `calculate_kinship` (`kind=` standardized / centered / ibs; `legacy=True` = 1.0.0), `kinship_scale`, `KinshipAccumulator` |
| `pylmm/kinship_io.py` | GCTA binary GRM `read_grm` / `write_grm` (float32), plink2 `.rel` `read_rel`, eigendecomposition `.npz` with IDs (`read_eigen`, `write_eigen`) |
| `pylmm/plink_io.py` | `PlinkReader`: PLINK BED, TPED and EMMA genotypes (per SNP or in blocks, with `.bim` metadata); phenotype, covariate, kinship files |
| `pylmm/pgen.py` | `PgenSource`, `read_psam`, `read_pvar`, `write_pgen`: PLINK 2 PGEN via the optional `pgenlib` (`pip install pgenlib` or the `pgen` extra, LGPL-3.0-only; imported lazily); `PlinkReader(type="pgen")` and `--pfile` use it |
| `pylmm/genotypes.py` | `Variants` (`.bim` columns), `GenotypeSource` protocol (`iter_blocks`), `ArraySource` (in-memory matrix), `normalize_genotype` |
| `pylmm/samples.py` | sample IDs `(FID, IID)`, `align_samples`, `.id` files, `InputError` |
| `pylmm/loco.py` | leave-one-chromosome-out: `LocoKinship` (every K_{-c} in one pass), `gwas_loco` |
| `pylmm/misc.py` | `fit_two`, two-kinship variance components |
| `pylmm/sumstats.py` | the summary-statistics table `pylmm gwas` writes: `SumStats`, `write_sumstats`, `read_sumstats`, per-allele conversion |
| `pylmm/cli/` | the `pylmm` command: `pylmm gwas`, `pylmm kinship`; `run_record.py`, the JSON run record of `pylmm gwas` |
| `tests/` | pytest suite; `tests/golden/` = reference outputs of the original Python 2 code; `tests/helpers.py` = all tolerances |
| `sims/` | simulation framework (genotype/phenotype generators, oracle, studies); `sims/bench/`: Phase 6 speed benchmarks (`just bench <name> [--quick]`, e.g. `gwas_batch`, `bed_blocks`, `refit`) |
| `docs/` | [guide.md](docs/guide.md) (library user guide), [replication.md](docs/replication.md) (two published mouse GWAS reanalysed: one partially replicated, one replicated), conventions, reports on the Python 2 baseline, the port and the simulation studies, `migration/` maps |
| `skills/` | agent skills (below); `.claude/skills` links here |
| `data/` | example data (~215 MB): 1219 mice BED (fake phenotypes), 94-strain HMDP EMMA text; provenance, marker map and checksums in [data/README.md](data/README.md) |
| `scripts/example.py` | a short library example |
| `scripts/build_hmdp_marker_map.py` | rebuilds `data/hmdp.liver.snps.map.tsv` (build-37 positions, alleles) from the Broad mouse HapMap file |
| `notebooks/` | `pylmm_tour.ipynb`: walkthrough for human readers (run it to see outputs) (simulate, fit, scan, interpret) |

## Commands

From a clone, `uv sync --group sims` once (or `just sync`), then prefix commands with `uv run` (or use the
`just` recipes). `uv sync` and `uv run` create and use the virtual environment **`.venv/` at the repository
root** (ignored by version control); the first `uv run` creates it if `uv sync` has not, so a `.venv/` you
did not make yourself is expected. To use a different environment, either set
`UV_PROJECT_ENVIRONMENT=/path/to/venv` for every `uv` command, or activate your own venv and run
`pip install -e .` there (plus `pytest`, and `matplotlib pandas statsmodels` for the `sims` group), then
call `pytest` and `pylmm` without `uv run`.

```bash skip-ci: runs the whole suite or the linters
just test             # uv run --group sims pytest (fast suite; doctests + docs snippets included)
just test-slow        # + the full-dataset tests (--runslow)
just docs-test        # only the docs snippets (tests/test_docs.py)
just lint             # ruff check + ruff format --check (CI runs the same)
just fmt              # apply ruff fixes and formatting
just typecheck        # mypy on pylmm/, Python 3.11 and 3.14
just sim example_null --quick   # a simulation study
just bench gwas_batch --quick   # a benchmark (sims/bench/)
```

`uv run pylmm ...` runs the command line from the checkout:

```bash
uv run pylmm --version
uv run pylmm gwas --help > /dev/null   # every option, with examples
```

## Tests: what they mean

* **Goldens** (`tests/golden/`) are the outputs of the original 2013 Python 2 code. The golden
  tests are a *regression* contract: a refactor must not change a number beyond the tolerances. **Never
  edit, regenerate or loosen a golden file or a tolerance to make a test pass.** `just golden` exists for a
  deliberate, reviewed re-baseline only (see `tests/golden/README.md`).
* **Tolerances live only in `tests/helpers.py`** (`RTOL`, `RTOL_FIT`, `ATOL_FIT`, `RTOL_P`, ...), each with the
  measurement behind it. Compare numbers through `helpers.assert_close` / `assert_fit_close` /
  `assert_pvalues_close`, never as text.
* `tests/sims/` checks the simulation framework and keeps the simulation-study bug reproducers (`xfail(strict=True)`
  for unfixed bugs: fixing one makes it XPASS and fail; then flip the marker).
* **Docs are executable.** `tests/test_docs.py` runs every `python` and `bash` code block of this file,
  README.md, `docs/guide.md` and `skills/**/*.md`. Blocks of one file share a namespace and a scratch directory; mark a
  block that needs external tools or a network with `skip-ci: <reason>` after the language. If you change
  behaviour, fix the docs in the same change.
* Docstring examples in `pylmm/` run as doctests.

## Conventions ([docs/conventions.md](docs/conventions.md))

* PEP 8 names, except **math symbols**: a single capital letter, optionally with digits, names a model
  matrix (`Y`, `X`, `X0`, `K`, `K2`) and may be a parameter, local or attribute. Everything else is
  snake_case (`kva`, `opt_h`, `reml`).
* Verbs: `read_*` reads a file, `iter_*` iterates, `write_*` writes; no `get_*`. Scalars are returned as
  floats; "not given" is `None`; every `reml` parameter defaults to `True`.
* Samples are matched by ID `(FID, IID)`, never by row position (except EMMA input, which has no IDs).
* Type hints on all library code (`py.typed`), checked by mypy; numpydoc docstrings with runnable examples.
* ruff (line length 100) for lint and format; CI runs `ruff check`, `ruff format --check`, mypy, pytest on
  Python 3.11–3.14 and a lowest-direct (numpy 1.23.2, scipy 1.9.2) job.
* CLI: kebab-case flags named after PLINK's, 1-based column numbers, exit codes 0 / 2 (usage or input
  error) / 1 (unexpected), messages via the `pylmm` logger.

## Commits and PRs

* Never `--no-verify`, never force-push, never push or merge without being asked.
* One logical change per commit, message body says what and why (see `git log`).
* Refactors are **behaviour-neutral**: CLI outputs byte-identical, goldens green. A change that moves
  numbers is a separate, flagged change with evidence (a simulation or a golden diff), never folded into a
  refactor.
* User-visible changes go in `CHANGELOG.md` (Keep a Changelog, "old -> new").

## Skills (`skills/<name>/SKILL.md`)

`.claude/skills` is a relative symlink to `skills/`, so Claude Code finds them. A symlink may not
survive a zip download or a Windows checkout without symlink support (it becomes a small text file); then
use `skills/` directly, or copy it to `.claude/skills`.

| skill | use it to |
|---|---|
| `run-gwas` | run a GWAS end to end (CLI or library), check it, fix common errors; `reference.md`: phenotype QC/transforms, SNP QC, loci |
| `estimate-heritability` | fit the null model, interpret h², SE / profile-likelihood intervals, K scale vs other tools, inbred panels (strain-mean h²) |
| `custom-association-analysis` | write ad-hoc loops: subsets, conditional analysis, LOCO (`pylmm.loco`), many phenotypes |
| `inbred-strain-panels` | strains with replicate animals: strain aliases and duplicate-genotype audits, strain → animal expansion, PLINK-safe IDs, imputing over strains, unequal replicates |
| `simulate-lmm-data` | simulate genotypes/phenotypes with known h² and causal SNPs; sanity-check a pipeline |
| `convert-genotype-formats` | VCF/BCF, BGEN, PGEN, VCF Zarr, Hail → PLINK BED; phenotype and `.id` files |
| `migrate-from-old-pylmm` | port camelCase API / `pylmmGWAS` / `pylmmKinship` code and commands |
| `interpret-gwas-results` | read the output, thresholds, λ_GC, effect allele and scale, when to distrust a scan; `comparison-worksheet.md`: compare with a published GWAS |

## Known caveats (read before trusting a number)

* **Genotype coding and effect allele**: the reader's genotype is the *fraction of copies of A2*
  (`.bim` column 6): 0, 0.5, 1, nan. `pylmm gwas` output (default: the summary-statistics table,
  `pylmm.sumstats`) reports `beta`/`standard_error` **per copy of A1** (`effect_allele`, `.bim` column 5;
  ALT for plink2-written files, as plink2 `--glm omit-ref`), `t_stat` signed for A1, `beta_std` per SD.
  The library is different: `gwas`/`LMM.association` give the effect per unit of the column you pass (per
  0 -> 1 of the A2 fraction for raw reader genotypes: opposite sign, 2x scale), and
  `--output-format legacy` keeps 1.0.0's `BETA` per SD of A2. Check the allele before the sign. See
  `interpret-gwas-results`.
* **Add the intercept to library `X0`.** `LMM`/`gwas` use `X0` exactly as given (default: an intercept
  alone); `read_covariates` returns no intercept column: `X0 = np.column_stack([np.ones(n), cov])`.
  `pylmm gwas --covar` adds it (unless `--no-intercept`).
* **REML is the default** everywhere (`reml=True`, `--reml`); ML (`--ml`) is ill-posed for a centred K built
  from more SNPs than individuals (bug 17; `fit` warns).
* **The h search** covers [0, 1] and keeps the best local maximum (Phase 6.3, bugs 2, 3, 9 fixed).
  `legacy_fit=True` (`LMM`, `gwas`, `misc.fit_two`) / `pylmm gwas --legacy-fit` restores pylmm 1.0.0's fit
  (grid to 0.99, first optimum, ML sigma Q/(n-p)); the golden tests run with it, and
  `--legacy-fit --no-batch` is 1.0.0's output byte for byte.
* **Known, not yet fixed** (bug numbers: the table in
  [sims/studies/s4_bug_confirmation.md](sims/studies/s4_bug_confirmation.md)): genotypes are
  standardised over individuals later dropped for a missing phenotype (|Δt| ≤ 0.024).
* **Kinship scale = h scale.** `calculate_kinship` and `pylmm kinship` give the same K (one missing-call
  convention; constant and all-missing SNPs not counted in `m`; bug 8 fixed after 1.0.0, `legacy=True` /
  `--legacy-kinship` reproduce 1.0.0). Its mean diagonal is about 1 − the missing rate; `kind="ibs"` (EMMA)
  has diagonal 1 but a much smaller centred scale, so a larger h for the same data. Compare h across
  kinships only as PVE: `kinship_scale(K)`, see `estimate-heritability`. Pass raw genotypes
  (`norm_genotype=False`) to `calculate_kinship`, not reader-standardised ones.
* `LMM.association` fails on a `nan` genotype; `gwas` handles missing calls.
* A kinship without a `K.id` sidecar is assumed to be in genotype order. `pylmm gwas --bfile` / `--tfile`
  warns about that. `pylmm gwas --emma-geno` does **not**: EMMA input has no IDs, so K's rows are always
  taken in genotype-column order (a `K.id` that exists is ignored, with a warning). The library's
  `PlinkReader.read_kinship` does not warn either; it only checks that K is n × n. `pylmm kinship` writes
  the sidecar for PLINK input.
* Inbred panels from a per-animal BED: `pylmm gwas` mean-imputes missing calls over **animals**, not
  strains (Δ−log10 p up to 0.28 in the [replication](docs/replication.md)). See `inbred-strain-panels`.

A 30-second sanity check that the install works:

```python
import numpy as np
from pylmm.lmm import LMM, calculate_kinship, gwas

rng = np.random.default_rng(0)
W = rng.binomial(2, 0.3, size=(200, 500)).astype(float)
K = calculate_kinship(W)
Y = 1.0 * W[:, 0] + rng.multivariate_normal(np.zeros(200), 0.5 * K + 0.5 * np.eye(200))
h = LMM(Y, K).fit()[0]
ts, ps = gwas(Y, W[:, :20], K)
assert 0 <= h < 1 and ps[0] < 1e-4
```
