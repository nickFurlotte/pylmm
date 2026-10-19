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

## Exceptions that are not Python names

These are data or user-facing interface and keep their spelling:

- CLI flags and command names (`pylmmGWAS --removeMissingGenotypes`), and the optparse
  `dest` names in `pylmm/cli` (`options.phenoFile`), which set the `--help` metavars.
  The CLI is redesigned separately (Phase 5b).
- File formats: the `.npz` arrays written by `pylmmKinship --eigenNpz` are still named
  `Kva` and `Kve`; text-file extensions `.kva` / `.kve`.
- Keys of the golden result files under `tests/golden/`.

## Renames

The Phase 5.4 rename is a clean break (no aliases). `docs/migration/name-map.csv` maps
every changed pylmm name, old to new; `docs/migration/rename_5_4.py` is the script that
applied it.
