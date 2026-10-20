# Contributing to pylmm

Bug reports, fixes and documentation improvements are welcome. For anything larger than a small fix,
open an issue first so the change can be discussed.

## Development setup

pylmm uses [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/):

```bash
just sync        # uv sync --group sims: creates .venv/ at the repository root
just test        # the fast suite: unit, golden, doctest and docs-snippet tests
just test-slow   # also the full-dataset tests
just lint        # ruff check + ruff format --check (CI runs the same)
just fmt         # apply ruff fixes and formatting
just typecheck   # mypy on pylmm/
```

Without `just`, prefix commands with `uv run` (e.g. `uv run --group sims pytest`). CI runs lint, mypy and
the tests on Python 3.11-3.14 and at the lowest supported numpy and scipy.

## Rules for changes

- **Golden files and tolerances are a contract.** `tests/golden/` holds the outputs of the original 2013
  Python 2 code, and all numerical tolerances live in `tests/helpers.py`. Never edit, regenerate or loosen
  a golden file or a tolerance to make a test pass; a deliberate re-baseline is a separate, reviewed change
  (see `tests/golden/README.md`).
- **Refactors are behaviour-neutral**: command-line outputs stay byte-identical and the golden tests stay
  green. A change that moves numbers is its own change, with evidence (a simulation in `sims/` or a
  golden diff).
- **Docs are executable.** `tests/test_docs.py` runs the code blocks of `README.md`, `AGENTS.md`,
  `docs/guide.md` and `skills/`; if you change behaviour, update the docs in the same change.
- **User-visible changes go in [CHANGELOG.md](CHANGELOG.md)** (Keep a Changelog, entries read
  "old -> new").
- One logical change per commit, with a message that says what and why.

Naming, typing and command-line conventions are in [docs/conventions.md](docs/conventions.md).
[AGENTS.md](AGENTS.md) has the repository map, the commands and the known caveats; it is written for
coding agents but is a good overview for anyone.

## Reporting bugs

Use the [bug report template](https://github.com/nickFurlotte/pylmm/issues/new/choose). For security
issues, see [SECURITY.md](SECURITY.md) instead.
