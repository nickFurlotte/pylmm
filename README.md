# PyLMM

**PyLMM: a modern, readable, validated mixed-model toolkit for statistical genetics — built to be used by
humans and agents.**

PyLMM fits linear mixed models (LMMs) for genome-wide association studies (GWAS) and heritability. In
genetic association studies, individuals may be related or share population structure, which produces
misleading associations if it is ignored. A mixed model accounts for it through a kinship matrix of genetic
similarity between individuals:

```text
Y = X0 b + x a + u + e,    u ~ N(0, h s2 K),    e ~ N(0, (1 - h) s2 I)
```

with a Wald t test of each SNP `x`. PyLMM provides:

- **GWAS:** test variants for association with a phenotype, accounting for relatedness and population
  structure (REML by default, optional per-SNP variance refit, covariates, missing genotypes).
- **Heritability:** the null model gives the SNP heritability `h` and the variance components.
- **Kinship:** genetic similarity matrices from genotype data.
- **Command line:** `pylmm gwas` and `pylmm kinship` on PLINK BED/TPED or EMMA files, with samples
  matched by ID.
- **Python library:** `LMM`, `gwas`, `calculate_kinship` and `PlinkReader` for custom analyses.

The code favours being easy to read and check over speed: numpy and scipy only, type-annotated, with
documentation whose examples run in CI.

## Validated

- **Against the original code.** The test suite checks the outputs against reference outputs of the
  original 2013 Python 2 pylmm ([tests/golden/](tests/golden/)), with fixed tolerances. The Python 3 port
  matched them to within 5.1e-8 relative for every fitted quantity and 4.2e-7 absolute for per-SNP
  statistics ([docs/phase3-port.md](docs/phase3-port.md)). Deliberate changes since, such as bug fixes, are
  listed in the [CHANGELOG](CHANGELOG.md).
- **Simulation studies** with known heritability and causal SNPs, on five genotype structures
  ([docs/phase4-simulations.md](docs/phase4-simulations.md)). Under a correctly specified model, with the
  REML null fit and the Wald test, null p-values are calibrated when h² > 0 (type-I error 0.0495–0.0500
  at α = 0.05, mean χ² 0.99–1.01 over 1,575 replicates), and the Wald t matches an independent
  implementation to 2e-7. Under population structure pylmm has much more power than ordinary least
  squares, which is badly inflated there; the design partly favours the LMM, since the simulated
  polygenic effect is drawn from the same kinship that pylmm fits. The bugs the studies found are fixed
  or listed under *Known caveats* in [AGENTS.md](AGENTS.md).
- **Published results** ([docs/replication.md](docs/replication.md)). An agent working only from the docs
  reanalysed two published mouse GWAS of the Hybrid Mouse Diversity Panel:
  - *Farber et al. 2011*, bone mineral density: **partially replicated**. All 5 published trait–locus
    pairs have a signal in the published ±1.3 Mb interval, 3 at the published position. An interval
    always has a best SNP, so position alone is not a replication, and significance is systematically
    lower (by 0.47 in −log10 p, median): 1/5 pass Bonferroni and 3/5 the paper's threshold.
  - *Bennett et al. 2010*, HDL cholesterol: **replicated**, with the same lead SNP (the *Nos1ap* peak on
    chromosome 1); −log10 p 11.5 with a per-SNP refit (7.9–9.0 without the refit) vs 9.4 published.

## Install

PyLMM needs Python 3.11 or newer, numpy and scipy. It is not on PyPI yet; install it from GitHub:

```bash skip-ci: network
pip install "git+https://github.com/nickFurlotte/pylmm@main"
```

This installs the Python library and the `pylmm` command. A git install clones the whole repository,
including ~215 MB of example data ([data/README.md](data/README.md)), so it takes a while. To work from a
source checkout:

```bash skip-ci: network
git clone -b main https://github.com/nickFurlotte/pylmm
cd pylmm
uv sync           # then prefix commands with `uv run`
```

The `main` branch holds the current code; the older `master` branch keeps the original Python 2 code.

## Quick start

### Command line

From a checkout, with the bundled example data (1,219 mice, 98,740 SNPs, a fake phenotype); `pylmm` is
on the PATH after `pip install`, or prefix it with `uv run` after `uv sync`. It takes about a minute:

```bash skip-ci: full dataset, see just readme-example
pylmm kinship --bfile data/snps.132k.clean.noX -o snps.kin
pylmm gwas --bfile data/snps.132k.clean.noX --kinship snps.kin \
  --pheno data/snps.132k.clean.noX.fake.phenos -o results.tsv
```

`pylmm kinship` writes the kinship matrix `snps.kin` and its sample IDs `snps.kin.id`, which `pylmm gwas`
uses to match samples. `results.tsv` has one row per SNP: `SNP_ID`, `BETA`, `BETA_SD`, `F_STAT` (the
signed t) and `P_VALUE`. `pylmm gwas --help` lists every option, with examples.

### Python

```python
import numpy as np
from pylmm.lmm import LMM, calculate_kinship, gwas

rng = np.random.default_rng(0)
W = rng.binomial(2, 0.3, size=(200, 500)).astype(float)  # 200 individuals x 500 SNPs
K = calculate_kinship(W)
Y = 1.0 * W[:, 0] + rng.multivariate_normal(np.zeros(200), 0.5 * K + 0.5 * np.eye(200))

h, beta, sigma2, ll = LMM(Y, K).fit()  # null model: heritability h, REML
ts, ps = gwas(Y, W[:, :20], K)         # Wald t statistics and p-values, one per SNP
print(f"h = {h:.2f}, p(SNP 0) = {ps[0]:.1e}")
```

The [user guide](docs/guide.md) covers loading PLINK files, covariates (add the intercept to `X0`
yourself), missing data, reading the results and the pitfalls.

## For agents

[AGENTS.md](AGENTS.md) is the entry point for coding agents: the repository map, commands, what the tests
mean, conventions and the known caveats to read before trusting a number. [skills/](skills/) holds agent
skills for running a GWAS, estimating heritability, custom analyses, inbred strain panels, simulation,
converting genotype formats, interpreting results and migrating old code. Claude Code finds them through
`.claude/skills`.

## Migrating from the old pylmm

The camelCase API (`lmm.GWAS`, `calculateKinship`, ...) and the `pylmmGWAS` / `pylmmKinship` commands are
replaced by PEP 8 names and one `pylmm` command. The `migrate-from-old-pylmm` skill, the name and flag maps
in [docs/migration/](docs/migration/) and the [CHANGELOG](CHANGELOG.md) cover the port.

## History

PyLMM was written by Nick Furlotte during his PhD and released in 2013 as a lightweight Python
implementation of linear mixed models for genetic association. Others have since studied, adapted and
ported it: the GeneNetwork port [pylmm_gn2](https://github.com/genenetwork/pylmm_gn2); Karl Broman's
[R/lmmlite](https://kbroman.org/lmmlite/) (2015), an R and C++ implementation of parts of the algorithm,
with [numerical comparisons](https://kbroman.org/lmmlite/assets/compare2pylmm.html) to PyLMM; and The
Jackson Laboratory's [pylmm3](https://github.com/TheJacksonLaboratory/pylmm3) (2026), an independent
Python 3 modernisation. Version 1.0 is a rewrite by the original author: Python 3, a new API and command
line, and validation against the original code and published results.

## Links

- [User guide](docs/guide.md) · [Conventions](docs/conventions.md) · [CHANGELOG](CHANGELOG.md)
- [Replication of published GWAS](docs/replication.md) · [Example data](data/README.md)
- [Issues](https://github.com/nickFurlotte/pylmm/issues) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

Development uses `uv`, `pytest`, `ruff` and `mypy`: `just test`, `just lint` and `just typecheck` (see
the [justfile](justfile) and [AGENTS.md](AGENTS.md)).

## How to cite

If you use PyLMM, please cite the software; [CITATION.cff](CITATION.cff) has the details (GitHub's "Cite
this repository" button reads it).

## License

MIT. See [LICENSE](LICENSE).

Code: MIT. Example data in `data/`: see [data/README.md](data/README.md).
