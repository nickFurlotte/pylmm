
## pylmm - A lightweight linear mixed-model solver

pylmm is a fast and lightweight linear mixed-model (LMM) solver for use in genome-wide association studies (GWAS).

pylmm has a standalone program for running association studies called pylmmGWAS.  It can also be used as a python module to build your own custom programs.  If you want to create your own code, look at example.py for some basic usage patterns.  If you want to run basic GWAS analysis, the command below, which uses example data might be a helpful guide.

### An Example Command:

```
pylmmGWAS -v --bfile data/snps.132k.clean.noX --kfile data/snps.132k.clean.noX.pylmm.kin --phenofile data/snps.132k.clean.noX.fake.phenos out.foo
```

From a source checkout you can also run `uv run pylmmGWAS ...` or `python scripts/pylmmGWAS.py ...`. The same works for `pylmmKinship`.

The GWAS program pylmmGWAS.py reads PLINK formated input files (BED or TPED only).  There is also an option to use "EMMA" formatted files.  We included this in order to make it easier for people using EMMA currently to try pylmm.  The kinship matrix file can be calculated using pylmmKinship.py which also takes PLINK or EMMA files as input.  The kinship matrix output is just a plain text file and follows the same format as that used by EMMA, so that you can use pre-computed kinship matrices from EMMA as well, or any other program for that matter.

Individuals are matched by ID (FID, IID), not by row position. pylmmKinship writes, next to the kinship file `K`, an ID file `K.id` (one `FID IID` per line). If `K.id` exists, pylmmGWAS uses it to align the kinship to the genotype file (individuals missing from the kinship are dropped, with a warning); otherwise it warns and assumes that the kinship rows are in the order of the genotype file. You can write `K.id` yourself for a kinship computed elsewhere. The command line matches individuals by FID and IID; matching by IID alone is available in the library (`PlinkReader(id_match="iid")`), and a command-line option for it comes with the next CLI rewrite. When K drops individuals, genotypes are imputed and standardised over the kept individuals, but individuals dropped for a missing phenotype or covariate still count in the mean and SD (a difference of at most 0.024 in the test statistic in our check; making them consistent is planned).

## Installation
pylmm needs Python 3.10 or newer. pip installs numpy and scipy for you.
Install pylmm with pip:

```
   pip install git+https://github.com/nickFurlotte/pylmm@main
```
`@main` selects the Python 3 branch; the repository's default branch still holds the old Python 2 code.
A git install clones the whole repository, including about 215MB of example data in `data/`, so it can take a while.
This installs the `pylmm` module and the two command-line programs `pylmmGWAS` and `pylmmKinship`.

To work from a clone, use [uv](https://docs.astral.sh/uv/) (or run `pip install .`):
```
   git clone -b main https://github.com/nickFurlotte/pylmm
   cd pylmm
   uv sync
```

### Development
Run the tests with `uv run pytest`. Add `--runslow` to include the full-dataset tests.
The [justfile](justfile) holds the common commands. Run `just` to list them, for example
`just test`, `just test-slow`, `just example` and `just test-py27`.

Lint and formatting use [ruff](https://docs.astral.sh/ruff/): `just lint` checks and `just fmt`
applies fixes. CI runs the same checks. An optional [pre-commit](https://pre-commit.com) config
(`.pre-commit-config.yaml`) runs ruff on commit; it is not required.

The library modules carry type hints (the package ships `py.typed`). `just typecheck` runs
[mypy](https://mypy-lang.org) on them as CI does (config in `pyproject.toml`; `pylmm/cli/` is not
checked yet). It runs under Python 3.10, the oldest supported version. `uv run pytest` also runs the
docstring examples in `pylmm/` as doctests.

pylmm is offered under the MIT License. See [LICENSE](LICENSE).




