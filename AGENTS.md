# AGENTS.md

**PyLMM: a modern, readable, validated mixed-model toolkit for statistical genetics, built to be used by
humans and agents.** A linear mixed model (LMM) for GWAS and heritability: `Y = X0 b + x a + u + e`,
`u ~ N(0, h s2 K)`, `e ~ N(0, (1 - h) s2 I)`, with a Wald t test per SNP. Python ≥ 3.11, numpy + scipy.
This file is for coding agents working in this repo or with pylmm; humans should start at
[README.md](README.md) and [docs/guide.md](docs/guide.md).

## Repo map

| path | what |
|---|---|
| `pylmm/lmm.py` | the model: `calculate_kinship`, `LMM` (`fit`, `association`), `gwas`, input checks |
| `pylmm/plink_io.py` | `PlinkReader`: PLINK BED, TPED and EMMA genotypes; phenotype, covariate, kinship files |
| `pylmm/samples.py` | sample IDs `(FID, IID)`, `align_samples`, `.id` files, `InputError` |
| `pylmm/misc.py` | `fit_two`, two-kinship variance components |
| `pylmm/cli/` | the `pylmm` command: `pylmm gwas`, `pylmm kinship` |
| `tests/` | pytest suite; `tests/golden/` = Phase 1 reference outputs; `tests/helpers.py` = all tolerances |
| `sims/` | Phase 4 simulation framework (genotype/phenotype generators, oracle, studies) |
| `docs/` | [guide.md](docs/guide.md) (library user guide), [replication.md](docs/replication.md) (two published mouse GWAS reanalysed: one partially replicated, one replicated), conventions, Phase 1/3/4 reports, `migration/` maps, `research/` |
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
```

`uv run pylmm ...` runs the command line from the checkout:

```bash
uv run pylmm --version
uv run pylmm gwas --help > /dev/null   # every option, with examples
```

## Tests: what they mean

* **Goldens** (`tests/golden/`) are the outputs of the original 2013 Python 2 code (Phase 1). The golden
  tests are a *regression* contract: a refactor must not change a number beyond the tolerances. **Never
  edit, regenerate or loosen a golden file or a tolerance to make a test pass.** `just golden` exists for a
  deliberate, reviewed re-baseline only (see `tests/golden/README.md`).
* **Tolerances live only in `tests/helpers.py`** (`RTOL`, `RTOL_FIT`, `ATOL_FIT`, `RTOL_P`, ...), each with the
  measurement behind it. Compare numbers through `helpers.assert_close` / `assert_fit_close` /
  `assert_pvalues_close`, never as text.
* `tests/sims/` checks the simulation framework and keeps the Phase 4 bug reproducers (`xfail(strict=True)`
  for unfixed bugs: fixing one makes it XPASS and fail; then flip the marker).
* **Docs are executable.** `tests/test_docs.py` runs every `python` and `bash` code block of this file,
  `docs/guide.md` and `skills/**/*.md`. Blocks of one file share a namespace and a scratch directory; mark a
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
| `custom-association-analysis` | write ad-hoc loops: subsets, conditional analysis, LOCO kinship, many phenotypes |
| `inbred-strain-panels` | strains with replicate animals: strain aliases and duplicate-genotype audits, strain → animal expansion, PLINK-safe IDs, imputing over strains, unequal replicates |
| `simulate-lmm-data` | simulate genotypes/phenotypes with known h² and causal SNPs; sanity-check a pipeline |
| `convert-genotype-formats` | VCF/BCF, BGEN, PGEN, VCF Zarr, Hail → PLINK BED; phenotype and `.id` files |
| `migrate-from-old-pylmm` | port camelCase API / `pylmmGWAS` / `pylmmKinship` code and commands |
| `interpret-gwas-results` | read the output, thresholds, λ_GC, effect allele and scale, when to distrust a scan; `comparison-worksheet.md`: compare with a published GWAS |

## Known caveats (read before trusting a number)

* **Genotype coding**: a BED genotype is the *fraction of copies of A2* (`.bim` column 6): 0, 0.5, 1, nan.
  plink2-written `.bim` files have A2 = REF, so pylmm's effect allele is REF, opposite to plink2 `--glm`.
  Output `BETA` is per SD of the standardised genotype, `BETA_SD` is its standard error, `F_STAT` is the
  signed t. See `interpret-gwas-results`.
* **Add the intercept to library `X0`.** `LMM`/`gwas` use `X0` exactly as given (default: an intercept
  alone); `read_covariates` returns no intercept column: `X0 = np.column_stack([np.ones(n), cov])`.
  `pylmm gwas --covar` adds it (unless `--no-intercept`).
* **REML is the default** everywhere (`reml=True`, `--reml`); ML (`--ml`) is ill-posed for a centred K built
  from more SNPs than individuals (bug 17).
* **Phase 6.3 items (known, not yet fixed)**: the h search grid stops at 0.99, so h = 1 is never returned
  and h² ≥ 0.98 reads 0.99 (bugs 2, 3); the two kinship normalisations differ with missing calls and
  monomorphic SNPs still count in `m` (bug 8); the first grid optimum is kept, not the best; `fit()`'s ML
  sigma is Q/(n-q) (bug 9); `tstat(log=True)` is wrong and unused (bug 5); genotypes are standardised over
  individuals later dropped for a missing phenotype (6.3b, |Δt| ≤ 0.024).
* `LMM.association` fails on a `nan` genotype; `gwas` handles missing calls.
* A kinship without a `K.id` sidecar is assumed to be in genotype order. `pylmm gwas --bfile` / `--tfile`
  warns about that. `pylmm gwas --emma-geno` does **not**: EMMA input has no IDs, so K's rows are always
  taken in genotype-column order (a `K.id` that exists is ignored, with a warning). The library's
  `PlinkReader.read_kinship` does not warn either; it only checks that K is n × n. `pylmm kinship` writes
  the sidecar for PLINK input.
* Inbred panels from a per-animal BED: `pylmm gwas` mean-imputes missing calls over **animals**, not
  strains (Δ−log10 p up to 0.28 in the Phase R replication). See `inbred-strain-panels`.

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
