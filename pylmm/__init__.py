# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""PyLMM: a modern, readable, validated mixed-model toolkit for statistical genetics.

Built to be used by humans and agents.

The model
---------
For n individuals, a phenotype ``Y``, covariates ``X0`` and a kinship matrix
``K``::

    Y = X0 b + g + e,   g ~ N(0, h * s2 * K),   e ~ N(0, (1 - h) * s2 * I)

``h`` is the heritability (the share of the variance ``s2`` carried by ``K``).
The SNP effect is tested by adding the SNP as a covariate (a Wald test).

Module map
----------
``pylmm.lmm``
    The model and GWAS: :func:`~pylmm.lmm.calculate_kinship`,
    :class:`~pylmm.lmm.LMM` (``fit``, ``association``) and
    :func:`~pylmm.lmm.gwas`.
``pylmm.plink_io``
    Readers for PLINK and EMMA files (:class:`~pylmm.plink_io.PlinkReader`).
``pylmm.misc``
    :func:`~pylmm.misc.fit_two`, a two-kinship variance-component fit.
``pylmm.cli``
    The command-line tools (``pylmmGWAS``, ``pylmmKinship``).

Quick start
-----------
>>> import numpy as np
>>> from pylmm.lmm import LMM, calculate_kinship, gwas
>>> rng = np.random.default_rng(0)
>>> W = rng.integers(0, 3, size=(50, 200)).astype(float)   # 50 people, 200 SNPs
>>> K = calculate_kinship(W)                                # kinship
>>> Y = W[:, :5].sum(axis=1) + rng.standard_normal(50)      # phenotype
>>> model = LMM(Y, K)                                       # the LMM
>>> h, beta, sigma, ll = model.fit()                        # null-model fit
>>> bool(0 <= h < 1)
True
>>> ts, ps = model.association(W[:, [0]])                   # one SNP (n x 1)
>>> 0 <= ps <= 1
True
>>> ts, ps = gwas(Y, W[:, :10], K)                          # a scan
>>> len(ps)
10

Where to read more
------------------
* ``docs/conventions.md``: naming (PEP 8, except that a single capital letter,
  optionally followed by digits, is a math symbol: K, X, Y, X0, K2).
* ``CHANGELOG.md``: what changed, release by release.
* ``docs/migration/name-map.csv``: old name to new name for every renamed
  API name (Phase 5.4).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("pylmm")
except PackageNotFoundError:  # not installed (running from a source tree)
    __version__ = "unknown"

__all__ = ["__version__"]
