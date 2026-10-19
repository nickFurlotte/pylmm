
## pylmm - A lightweight linear mixed-model solver

pylmm is a fast and lightweight linear mixed-model (LMM) solver for use in genome-wide association studies (GWAS).

pylmm has a command-line program, `pylmm`, for running association studies (`pylmm gwas`) and computing kinship matrices (`pylmm kinship`). It can also be used as a python module to build your own custom programs.  If you want to create your own code, look at [scripts/example.py](scripts/example.py) for some basic usage patterns.  If you want to run a basic GWAS analysis, the commands below, which use the example data, might be a helpful guide.

### Example commands

```
# kinship matrix from the genotypes (also writes snps.kin.id, the sample IDs of its rows)
pylmm kinship -v --bfile data/snps.132k.clean.noX -o snps.kin

# association scan, using the kinship shipped with the example data
pylmm gwas -v --bfile data/snps.132k.clean.noX \
    --kinship data/snps.132k.clean.noX.pylmm.kin \
    --pheno data/snps.132k.clean.noX.fake.phenos -o results.tsv
```

The GWAS example uses the kinship shipped with the data (computed by the original 2013 code) rather than the `snps.kin` just computed, so that its output can be compared byte for byte with the reference results. The two kinships are not bit-identical: they agree to round-off (largest difference 1.8e-15), but the values themselves differ in their last bits, not only in how they are printed, so a scan with `snps.kin` gives p-values that differ slightly (relative difference up to about 2e-9).

`pylmm --help`, `pylmm gwas --help` and `pylmm kinship --help` list every option, with more examples. From a source checkout run `uv run pylmm ...` (or `python -m pylmm ...`); `just readme-example` runs both commands above.

The output of `pylmm gwas` (`-o/--out`) is a tab-separated file with one line per SNP: `SNP_ID`, `BETA`, `BETA_SD`, `F_STAT` and `P_VALUE`.

Common options of `pylmm gwas`:

| option | meaning |
|---|---|
| `--bfile PREFIX` / `--tfile PREFIX` / `--emma-geno FILE` | the genotypes: PLINK BED, PLINK TPED, or an EMMA matrix (exactly one) |
| `--pheno FILE` | PLINK-style phenotype file (`FID IID value ...`, `NA`/`-9` missing); default `PREFIX.phenos` |
| `--pheno-name NAME` / `--pheno-col N` | the phenotype column to test, by header name or by number, 1-based like plink2 (default: 1, the first) |
| `--covar FILE`, `--covar-name NAME ...` | covariates, optionally only the named columns; an intercept is added unless `--no-intercept` |
| `--kinship FILE` | the kinship matrix (required), e.g. from `pylmm kinship` |
| `--id-match {fid-iid,iid}` | match samples on FID and IID (default) or on the IID alone |
| `--ml` | maximum likelihood instead of REML (the default, `--reml`) |
| `--refit` | refit the variance components for each SNP |
| `--remove-missing-genotypes` | test each SNP on the individuals with a call, instead of imputing the mean |
| `-v` / `-q` / `--log-file FILE` / `--debug` | more or fewer messages, a log file, tracebacks |

Exit codes: 0 on success, 2 for a usage or input error (a one-line `pylmm: error:` message), 1 for an unexpected error (rerun with `--debug` for the traceback). The command line was redesigned in Phase 5b.2: [docs/migration/cli-map.csv](docs/migration/cli-map.csv) maps the old `pylmmGWAS` / `pylmmKinship` flags to the new ones.

`pylmm gwas` reads PLINK formatted input files (BED or TPED only).  There is also an option to use "EMMA" formatted files (`--emma-geno`, `--emma-pheno`, `--emma-covar`).  We included this in order to make it easier for people using EMMA currently to try pylmm.  The kinship matrix file can be calculated using `pylmm kinship`, which also takes PLINK or EMMA files as input.  The kinship matrix output is just a plain text file and follows the same format as that used by EMMA, so that you can use pre-computed kinship matrices from EMMA as well, or any other program for that matter.

Individuals are matched by ID (FID, IID), not by row position. `pylmm kinship` writes, next to the kinship file `K`, an ID file `K.id` (one `FID IID` per line). If `K.id` exists, `pylmm gwas` uses it to align the kinship to the genotype file (individuals missing from the kinship are dropped, with a warning); otherwise it warns and assumes that the kinship rows are in the order of the genotype file. You can write `K.id` yourself for a kinship computed elsewhere. `--id-match iid` matches by IID alone (in the library: `PlinkReader(id_match="iid")`). When K drops individuals, genotypes are imputed and standardised over the kept individuals, but individuals dropped for a missing phenotype or covariate still count in the mean and SD (a difference of at most 0.024 in the test statistic in our check; making them consistent is planned).

## Installation
pylmm needs Python 3.10 or newer. pip installs numpy and scipy for you.
Install pylmm with pip:

```
   pip install git+https://github.com/nickFurlotte/pylmm@main
```
`@main` selects the Python 3 branch; the repository's default branch still holds the old Python 2 code.
A git install clones the whole repository, including about 215MB of example data in `data/`, so it can take a while.
This installs the `pylmm` module and the `pylmm` command-line program.

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
[mypy](https://mypy-lang.org) on the whole package, the command line included, as CI does (config in
`pyproject.toml`), under Python 3.10, the oldest supported version, and 3.14. `uv run pytest` also runs the
docstring examples in `pylmm/` as doctests.

pylmm is offered under the MIT License. See [LICENSE](LICENSE).




