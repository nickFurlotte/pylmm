#!/usr/bin/env python3

# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

# Thin shim kept so `python scripts/pylmmGWAS.py ...` keeps working.  The
# program lives in pylmm/cli/gwas.py and is installed as `pylmmGWAS`.

import os
import sys

# Always put this checkout's repo root first, so the shim runs the code next
# to it even if another (installed or already-imported) pylmm is around.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
for _m in [m for m in sys.modules if m == "pylmm" or m.startswith("pylmm.")]:
    del sys.modules[_m]

from pylmm.cli.gwas import main  # noqa: E402

if __name__ == "__main__":
    main()
