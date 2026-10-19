# PyLMM

**A lightweight linear mixed-model toolkit for statistical genetics.**

PyLMM is an open-source Python package for fitting linear mixed models (LMMs), particularly for genome-wide association studies (GWAS).

In genetic association studies, individuals may be related or share population structure. These relationships can produce misleading associations if they are not accounted for. Linear mixed models address this by modeling genetic similarity between individuals.

PyLMM provides an accessible implementation of these methods, including:

- **GWAS:** Test variants for association with a phenotype while accounting for relatedness and population structure.
- **Kinship estimation:** Compute genetic similarity matrices from genotype data.
- **Python API:** Build custom analyses and experiment with mixed-model methods.
- **Command-line tools:** Run analyses with PLINK or EMMA-formatted data.

The goal is to keep the methods accessible, the implementation understandable, and the results reproducible.

## A brief history

PyLMM was originally developed by **Nick Furlotte** during his PhD research and released in 2013 as a lightweight Python implementation of linear mixed models for genetic association analysis.

Since then, the original software has been studied, adapted, and ported by other researchers:

- **Original PyLMM (2013):** The initial Python implementation and associated research software.
- **GeneNetwork port [pylmm_gn2](https://github.com/genenetwork/pylmm_gn2):** An effort to adapt PyLMM for newer Python environments and continued use in genetic analysis.
- **Karl Broman's [R/lmmlite](https://kbroman.org/lmmlite/) (2015):** An R and C++ implementation of parts of the PyLMM algorithm for GWAS and QTL mapping. Broman developed it while studying how LMMs work, documenting the process in [a detailed blog post](https://kbroman.org/blog/2015/11/24/fitting-linear-mixed-models-for-qtl-mapping/) and publishing reproducible [numerical comparisons](https://kbroman.org/lmmlite/assets/compare2pylmm.html) and [performance comparisons](https://kbroman.org/lmmlite/assets/performance.html) with PyLMM.
- **[The Jackson Laboratory's pylmm3](https://github.com/TheJacksonLaboratory/pylmm3) (2026):** An independent Python 3 modernization of the original package.
- **PyLMM rewrite (2026):** A new modernization by the original author, focused on clean Python, reproducibility, testing, and making the package easier for researchers and AI coding agents to work with.

The latest rewrite preserves the core statistical functionality while updating the implementation for modern Python development. A central goal is to validate the rewritten software against the original implementation and published results, providing a reliable foundation for future methodological improvements.

In this rewrite, we aimed to be agent friendly and optimized for code that is easier to read and understand rather than optimizing for speed. 

## Installation

PyLMM requires Python 3.10 or newer.

Install directly from GitHub:

```bash
pip install git+https://github.com/nickFurlotte/pylmm@main
```

This installs both the Python library and the `pylmm` command-line program.

To work from a source checkout:

```bash
git clone -b main https://github.com/nickFurlotte/pylmm
cd pylmm
uv sync
```

The `main` branch contains the modern Python implementation. The repository's older `master` branch retains the original Python 2 code.

## Quick start

PyLMM provides two primary commands: `kinship` and `gwas`.

### 1. Compute a kinship matrix

A kinship matrix captures genetic similarity between individuals and is used by the mixed model to account for relatedness.

Using the example PLINK dataset included in the repository:

```bash
pylmm kinship \
  --bfile data/snps.132k.clean.noX \
  -o snps.kin
```

This generates a kinship matrix (`snps.kin`) and a corresponding sample ID file (`snps.kin.id`).

### 2. Run a GWAS

Once a kinship matrix is available, test genetic variants for association with a phenotype:

```bash
pylmm gwas \
  --bfile data/snps.132k.clean.noX \
  --kinship data/snps.132k.clean.noX.pylmm.kin \
  --pheno data/snps.132k.clean.noX.fake.phenos \
  -o results.tsv
```

This example uses the original precomputed kinship matrix included with the dataset to support reproducibility testing. To use the matrix generated in step 1 instead, set `--kinship snps.kin`.

The resulting tab-separated file contains one row per variant, including its estimated effect size, standard error, F-statistic, and association p-value.

For additional options:

```bash
pylmm --help
pylmm gwas --help
pylmm kinship --help
```

For custom analyses using the Python API, see [scripts/example.py](scripts/example.py) and the [user guide](docs/guide.md).

## Built for humans and agents

The 2026 rewrite was developed with a particular goal: **make scientific software easier to understand, verify, and extend, whether the contributor is a human researcher or an AI coding agent.**

Beyond modern Python conventions, type annotations, and automated testing, the repository includes documentation and reusable agent skills for common statistical genetics workflows.

- [AGENTS.md](AGENTS.md) — Repository structure, development conventions, testing requirements, and guidance for coding agents.
- [Agent Skills](skills/) — Reusable workflows for running GWAS, estimating heritability, simulating data, interpreting results, and building custom association analyses.
- [User guide](docs/guide.md) — More detailed examples and documentation for the Python library.
- [Replication](docs/replication.md) — Two published mouse GWAS reanalysed by an agent from the docs alone.
- [Example data](data/README.md) — What the bundled files are, where they come from, and their checksums.

Documentation examples are executed in continuous integration to help keep instructions synchronized with working code.

The longer-term goal is to make PyLMM a useful environment for exploring and validating statistical genetics methods, including AI-assisted implementations of new research.

## Development

PyLMM uses modern Python development tools, including `uv`, `pytest`, `ruff`, and `mypy`.

```bash
uv run pytest           # Run the test suite
uv run pytest --runslow # Include full-dataset tests
just lint               # Check linting
just typecheck          # Check types
```

See the [justfile](justfile) for additional development commands and [AGENTS.md](AGENTS.md) for contributor guidance.

## License

PyLMM is available under the MIT License. See [LICENSE](LICENSE).
