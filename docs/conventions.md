# Naming conventions

pylmm follows [PEP 8](https://peps.python.org/pep-0008/#naming-conventions), with one
exception for math symbols. `ruff` enforces both (the `N` rules, configured in
`pyproject.toml`).

## The rule

| what | style | examples |
|---|---|---|
| modules | `lower_snake` | `lmm`, `plink_io`, `misc` |
| classes | `CapWords` | `LMM`, `PlinkReader` |
| functions, methods | `lower_snake`; a leading `_` if not public API | `calculate_kinship`, `gwas`, `log_likelihood`, `_get_max` |
| parameters, locals, attributes | `lower_snake`, or a math symbol (below) | `reml`, `kva`, `opt_h`, `x0t`, `K`, `X0` |
| module constants | `UPPER_SNAKE` | `ESTIMABLE_RTOL` |

**Math symbols.** A name that is a single capital letter, optionally followed by digits,
names a matrix or vector of the model and keeps the notation of the papers: `Y`
(phenotype), `X` (SNPs or design), `X0` (covariates), `K` (kinship), `K1`, `K2`, `G`
(genotypes), `W`, `H`, `Q`, `N`. It may be used as a parameter, a local variable or an
attribute (`LMM.K`, `LMM.N`). Anything longer is ordinary snake_case: derived names are
lower-cased (`Kva` -> `kva`, `X0t` -> `x0t`, `XX_i` -> `xx_inv`), and acronyms are not
kept in capitals (`REML` -> `reml`, `GWAS` -> `gwas`, `LL` -> `log_likelihood`).

ruff allows the symbols through `[tool.ruff.lint.pep8-naming] ignore-names`
(`[A-Z]`, `[A-Z][0-9]`, `[A-Z][0-9][0-9]`), so they need no `noqa`. ruff does not check
attribute names; the rule applies to them all the same.

## Verbs, return values and defaults

- `read_*` reads a file and returns (or stores) its data: `PlinkReader.read_phenos`,
  `read_covariates`, `read_kinship`, `pylmm.samples.read_id_file`. `iter_*` returns an iterator:
  `PlinkReader.iter_snps`. `write_*` writes a file. There are no `get_*` methods.
- A scalar is returned as a float (`sigma`, `Q`, `beta_var`, a log-likelihood), never as a 1x1 array.
  A result with more than two or three parts is a `NamedTuple` (`LogLikResult`), so tuple
  unpacking keeps working.
- Every function or method that takes `reml` defaults to `reml=True`.
- "Not given" is `None` (`kva=None`), never an empty sequence.
- One SNP may be passed as an `(n,)` vector or an `(n, 1)` column.
- Individuals are matched by ID (`pylmm.samples`), never by row position, except for EMMA input,
  which has no IDs.

## Command line

The earlier CLI exception to the PEP 8 rename (camelCase flags such as `pylmmGWAS --removeMissingGenotypes`
and optparse `dest` names such as `options.phenoFile`) is resolved: there is one command,
`pylmm`, with subcommands (`pylmm gwas`, `pylmm kinship`), and

- long options are kebab-case (`--remove-missing-genotypes`, `--pheno-name`), named after the
  PLINK option where there is one (`--bfile`, `--pheno`, `--covar`, `--covar-name`,
  `--pheno-name`, `--out`); EMMA inputs share an `--emma-` prefix (`--emma-geno`, `--emma-pheno`);
- the argparse `dest` is the option name in snake_case (`args.remove_missing_genotypes`), except
  where one dest serves two options (`--reml`/`--ml` -> `args.reml`, `-v`/`-q` ->
  `args.log_level`);
- choice values are kebab-case too (`--id-match fid-iid`), mapped to the library's values
  (`id_match="fid_iid"`);
- column numbers are 1-based, as in plink2 (`--pheno-col N`, like `--pheno-col-nums`; GCTA
  `--mpheno` and GEMMA `-n` count the same way); the library's `columns=` indices stay 0-based;
- every output file is an option (`-o/--out`), never a positional argument; the commands check
  that the outputs can be written before they start the work;
- exit codes: 0 success, 2 usage or input error, 1 unexpected error; messages go to the `pylmm`
  logger as `pylmm: <level>: <message>`. The file readers of `pylmm.samples` and
  `pylmm.plink_io` raise `InputError` (a `ValueError` subclass) for a malformed or inconsistent
  input file, also one found while reading the SNPs, and the CLI reports it as an input error
  (exit 2); a plain `ValueError` from the library means a bad argument;
- removed invocations are tombstones, not aliases: `python -m pylmm.cli.gwas` exits 2 with one
  line naming `pylmm gwas`.

`docs/migration/cli-map.csv` maps every old command and flag to the new one. Its columns:
`old_command`, `old_flag`, `old_takes_value`, `new_command`, `new_flag`, `new_takes_value`
(`yes`/`no`), `kind` and `notes`. The flag columns hold only flags (a positional argument has
an empty flag and is described in `notes`), and `kind` is one of `renamed`, `value→flag`
(an option that took a value is now a flag), `removed`, `removed-earlier` (gone before the single
`pylmm` command), `new`, `behaviour-changed`, `semantics-changed` (same option, different meaning of its
value) or `unchanged`. `tests/test_cli.py` checks the schema and that every option of the
parser has a row.

## Exceptions that are not Python names

These are data or user-facing interface and keep their spelling:

- File formats: the `.npz` arrays written by `pylmm kinship --eigen-npz` are still named
  `Kva` and `Kve`; text-file extensions `.kva` / `.kve`.
- Keys of the golden result files under `tests/golden/`.

## Renames

The PEP 8 rename of 1.0.0 is a clean break (no aliases). `docs/migration/name-map.csv` maps
every changed pylmm name, old to new (`module` is where the name lives after the rename: the
module, `module.Class`, or the function that takes the parameter);
`docs/migration/rename_5_4.py` is the script that applied it and wrote the CSV (`--csv`).
The row `next,__next__` records that the py2 `reader.next()` method is gone: callers should use
`next(reader)`.

A second round of renames and signature changes is in the same CSV, which has a `phase`
column (`5.4` for the first round, `5b.1` for the second; both are in 1.0.0) and a `notes` column. Rows whose `old_name` equals `new_name`
(`kind=signature`, or `parameter`) record a change of type or default, not of name. A name renamed
twice has a row per phase (`getSNPIterator` -> `get_snp_iterator` in 5.4, -> `iter_snps` in 5b.1).
`rename_5_4.py --csv` rewrites the 5.4 rows and keeps the later ones.
