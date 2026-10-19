# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""``python -m pylmm``: the ``pylmm`` command line (see `pylmm.cli`)."""

from pylmm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
