# Common pylmm development commands.  Run `just` to list them.

# Pass recipe arguments through as "$@" so quoted pytest args such as
# `just test -k "a or b"` survive intact.
set positional-arguments

# List the available recipes
default:
    @just --list

# Create/update the project virtualenv from uv.lock
sync:
    uv sync --group sims

# Check lint and formatting (ruff check + ruff format --check)
lint:
    uv run --locked ruff check
    uv run --locked ruff format --check

# mypy config is in pyproject.toml.  Each run uses a throwaway environment with the
# numpy (and numpy stubs) that the lock resolves for that Python.
# Type-check the package, CLI included, under Python 3.11 and 3.14 (as CI does)
typecheck:
    uv run --locked --isolated --python 3.11 mypy
    uv run --locked --isolated --python 3.14 mypy --python-version 3.14

# Apply formatting and auto-fixable lint fixes
fmt:
    uv run ruff format
    uv run ruff check --fix

# Run the test suite (slow tests skipped); extra args go to pytest
test *args:
    uv run --group sims pytest "$@"

# Execute notebooks/pylmm_tour.ipynb top to bottom (needs the notebooks group)
notebook-test:
    uv run --locked --group sims --group notebooks pytest -q tests/test_notebook.py

# Run only the executable docs: every code block of AGENTS.md, docs/guide.md, skills/
docs-test *args:
    uv run --group sims pytest tests/test_docs.py "$@"

# Run the test suite including the slow full-dataset tests
test-slow *args:
    uv run --group sims pytest --runslow "$@"

# Run the test suite in the Python 2.7 baseline Docker image
test-py27 *args:
    docker/py27/test.sh "$@"

# Run a Phase 4 simulation study, e.g. `just sim example_null --quick --jobs 4`
sim STUDY *args:
    uv run --group sims python -m sims run "$@"

# Summarise a study (summary.csv/md + PNGs); add --quick for the quick run
sim-report STUDY *args:
    uv run --group sims python -m sims report "$@"

# Run scripts/example.py
example:
    uv run python scripts/example.py

# Run the pylmm command line, e.g. `just pylmm gwas --help`
pylmm *args:
    uv run pylmm "$@"

# The README example: kinship, then GWAS, of the bundled data into OUTDIR
readme-example OUTDIR="readme-example":
    mkdir -p "{{OUTDIR}}"
    uv run pylmm kinship -v --bfile data/snps.132k.clean.noX -o "{{OUTDIR}}/snps.kin"
    uv run pylmm gwas -v --bfile data/snps.132k.clean.noX --kinship "{{OUTDIR}}/snps.kin" --pheno data/snps.132k.clean.noX.fake.phenos -o "{{OUTDIR}}/results.tsv"

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# DANGER: OVERWRITES tests/golden/ -- ONLY for a deliberate, reviewed
# regeneration of the Phase 1 reference outputs.  Never run this to "fix" a
# failing test.  It runs whatever pylmm code is in this working tree under
# the py2.7 baseline image; the goldens are meant to come from the Phase 1
# baseline code, so check that code out first.  See tests/golden/README.md.
# To measure the current code against the goldens instead, use:
#     uv run python tests/golden/diff_vs_golden.py
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Regenerate tests/golden/ in the py2.7 Docker image (deliberate use only!)
golden:
    docker/py27/run.sh python tests/golden/generate_golden.py
