#!/usr/bin/env bash
# Run the pytest regression suite in the Python 2.7 baseline container.
#   docker/py27/test.sh                # fast suite (~1 min emulated)
#   docker/py27/test.sh --runslow      # also the full-dataset tests (~+3 min)
#   docker/py27/test.sh -k kinship     # any pytest args pass through
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/run.sh" pytest -q "$@"
