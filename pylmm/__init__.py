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
``pylmm.genotypes``
    Variant metadata (:class:`~pylmm.genotypes.Variants`), the
    :class:`~pylmm.genotypes.GenotypeSource` interface (blocks of SNPs) and
    :class:`~pylmm.genotypes.ArraySource` for a genotype matrix in memory.
``pylmm.samples``
    Sample IDs ``(FID, IID)`` and the alignment of genotype, phenotype,
    covariate and kinship data by ID (:func:`~pylmm.samples.align_samples`).
``pylmm.misc``
    :func:`~pylmm.misc.fit_two`, a two-kinship variance-component fit.
``pylmm.cli``
    The ``pylmm`` command line (``pylmm gwas``, ``pylmm kinship``).

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
>>> ts, ps = model.association(W[:, 0])                     # one SNP
>>> bool(0 <= ps <= 1)
True
>>> ts, ps = gwas(Y, W[:, :10], K)                          # a scan
>>> len(ps)
10

Reading PLINK files
-------------------
A BED -> genotype matrix ``W`` (n x m) recipe, for the first ``m`` SNPs
(``reader.phenos`` is ``(n, k)``, so take ``[:, 0]`` for one phenotype):

>>> import itertools                                          # doctest: +SKIP
>>> from pylmm.plink_io import PlinkReader                    # doctest: +SKIP
>>> with PlinkReader("data/snps.132k.clean.noX", type="b",       # doctest: +SKIP
...                  pheno_file="data/snps.132k.clean.noX.fake.phenos") as reader:
...     W = np.column_stack([g for g, _ in itertools.islice(reader, m)])
...     Y = reader.phenos[:, 0]

Phenotype, covariate and kinship rows are matched to the genotype file by
``(FID, IID)``, never by position (see ``pylmm.plink_io`` and
``pylmm.samples``); a kinship with an ID file ``<kinship>.id`` (written by
``pylmm kinship``) is reordered to the genotypes, and individuals missing
from it are dropped from the phenotypes and the genotype vectors alike.

``gwas`` handles missing genotype calls (``nan``); ``LMM.association`` does
not, and fails on a SNP that contains one.

Effect scale and covariates
---------------------------
* ``beta`` is per unit of the genotype column you pass.  With
  ``PlinkReader``'s default ``norm_genotype=True`` it is per SD of the
  standardised genotype.  With ``norm_genotype=False`` it is per 0 -> 1 step
  of the A2-allele fraction, i.e. two copies of A2, so its sign and scale
  differ from PLINK's per-A1-copy convention (see ``pylmm.plink_io``).
* ``PlinkReader.read_covariates`` returns the covariate columns WITHOUT an
  intercept, and ``X0`` is used as given.  Passing covariates alone fits a
  model with no intercept (test statistics can change a lot); add one:
  ``X0 = np.column_stack([np.ones(n), cov])``.

Where to read more
------------------
* ``docs/conventions.md``: naming (PEP 8, except that a single capital letter,
  optionally followed by digits, is a math symbol: K, X, Y, X0, K2).
* ``CHANGELOG.md``: what changed, release by release.
* ``docs/migration/name-map.csv``: old name to new name for every renamed
  API name (the pre-1.0 names).
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _source_tree_version(pyproject: Path) -> str:
    """The ``[project] version`` of a pylmm ``pyproject.toml``, or "unknown".

    Used when pylmm is imported from a source checkout that is not installed
    (e.g. ``PYTHONPATH=/path/to/pylmm``), so there is no distribution
    metadata.  A small line parser, not a TOML parser, keeps the import cheap;
    the version is a plain ``version = "X"`` line.  A
    single-quoted version (``version = 'X'``) or a dynamic one (``dynamic =
    ["version"]``) is not recognised and returns "unknown".
    """
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    table = ""
    name = got = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            table = line
            continue
        if table != "[project]":
            continue
        m = re.match(r'(name|version)\s*=\s*"([^"]+)"', line)
        if m and m.group(1) == "name":
            name = m.group(2)
        elif m:
            got = m.group(2)
    return got if name == "pylmm" and got else "unknown"


try:
    __version__: str = version("pylmm")
except PackageNotFoundError:  # not installed: a source checkout on PYTHONPATH
    __version__ = _source_tree_version(Path(__file__).resolve().parent.parent / "pyproject.toml")

__all__ = ["__version__"]
