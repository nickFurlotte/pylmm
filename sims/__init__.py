"""Phase 4 simulation framework for pylmm.

Not part of the installed package; run from a source checkout with
``uv run --group sims python -m sims ...``.  See sims/README.md.

Importing ``sims`` pins the BLAS/OpenMP thread pools to one thread *if numpy
has not been imported yet* (the env vars only take effect before numpy loads).
Worker processes always start this way, because the runner spawns them with
the variables already set.  Set ``SIMS_BLAS_THREADS`` to override.
"""

import os
import sys

BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pin_blas_threads(n=None):
    """Set the BLAS thread env vars (to $SIMS_BLAS_THREADS or 1).

    Returns True if numpy was not yet imported, i.e. the setting is
    effective in this process.  It is always effective in child processes
    started afterwards.
    """
    n = str(n if n is not None else os.environ.get("SIMS_BLAS_THREADS", "1"))
    for var in BLAS_ENV_VARS:
        os.environ[var] = n
    return "numpy" not in sys.modules


pin_blas_threads()
