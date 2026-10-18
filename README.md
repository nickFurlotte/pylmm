
## pylmm - A lightweight linear mixed-model solver

pylmm is a fast and lightweight linear mixed-model (LMM) solver for use in genome-wide association studies (GWAS).

pylmm has a standalone program for running association studies called pylmmGWAS.  It can also be used as a python module to build your own custom programs.  If you want to create your own code, look at example.py for some basic usage patterns.  If you want to run basic GWAS analysis, the command below, which uses example data might be a helpful guide.

### An Example Command:

```
pylmmGWAS -v --bfile data/snps.132k.clean.noX --kfile data/snps.132k.clean.noX.pylmm.kin --phenofile data/snps.132k.clean.noX.fake.phenos out.foo
```

From a source checkout you can also run `uv run pylmmGWAS ...` or `python scripts/pylmmGWAS.py ...`. The same works for `pylmmKinship`.

The GWAS program pylmmGWAS.py reads PLINK formated input files (BED or TPED only).  There is also an option to use "EMMA" formatted files.  We included this in order to make it easier for people using EMMA currently to try pylmm.  The kinship matrix file can be calculated using pylmmKinship.py which also takes PLINK or EMMA files as input.  The kinship matrix output is just a plain text file and follows the same format as that used by EMMA, so that you can use pre-computed kinship matrices from EMMA as well, or any other program for that matter.

## Installation
pylmm needs Python 3.10 or newer. pip installs numpy and scipy for you.
Install pylmm with pip:

```
   pip install git+https://github.com/nickFurlotte/pylmm
```
This installs the `pylmm` module and the two command-line programs `pylmmGWAS` and `pylmmKinship`.

To work from a clone, use [uv](https://docs.astral.sh/uv/) (or run `pip install .`):
```
   git clone https://github.com/nickFurlotte/pylmm
   cd pylmm
   uv sync
```

### Development
Run the tests with `uv run pytest`. Add `--runslow` to include the full-dataset tests.
The [justfile](justfile) holds the common commands. Run `just` to list them, for example
`just test`, `just test-slow`, `just example` and `just test-py27`.

pylmm is offered under the MIT License. See [LICENSE](LICENSE).




