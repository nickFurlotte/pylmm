# Common pylmm development commands.  Run `just` to list them.

# List the available recipes
default:
    @just --list

# Create/update the project virtualenv from uv.lock
sync:
    uv sync

# Run the test suite (slow tests skipped); extra args go to pytest
test *args:
    uv run pytest {{args}}

# Run the test suite including the slow full-dataset tests
test-slow *args:
    uv run pytest --runslow {{args}}

# Run the test suite in the Python 2.7 baseline Docker image
test-py27 *args:
    docker/py27/test.sh {{args}}

# Run scripts/example.py
example:
    uv run python scripts/example.py

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
