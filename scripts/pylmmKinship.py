#!/usr/bin/env python3

# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

# Thin shim kept so `python scripts/pylmmKinship.py ...` keeps working.  The
# program lives in pylmm/cli/kinship.py and is installed as `pylmmKinship`.

import os
import sys

try:
    from pylmm.cli.kinship import main
except ImportError:
    # Running from a source checkout without installing the package.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from pylmm.cli.kinship import main

if __name__ == '__main__':
    main()
