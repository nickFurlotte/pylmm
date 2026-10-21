# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""The JSON run record ``pylmm gwas`` writes next to its results.

The record (``<out>.json``, `sidecar_path`) is one JSON object, written with
sorted keys and an indent of 2.  Its layout is versioned by its ``schema``
key (`RUN_RECORD_SCHEMA`): keys are only added within a version, never
renamed or removed.  The top-level keys, all always present (a value that
does not apply is ``null``):

``schema``, ``pylmm_version``, ``status`` (``"ok"`` or ``"error"``),
``error`` (the message, or null), ``command`` (the arguments of ``pylmm``,
as a list), ``started_at`` / ``finished_at`` (UTC, ISO 8601),
``environment`` (`EnvironmentInfo`), ``inputs`` (one `FileInfo` per input
file, by role), ``samples`` (`SampleCounts`), ``model`` (`ModelInfo`),
``kinship`` (`KinshipInfo`), ``null_fit`` (`NullFit`, or null with
``--refit`` and no LRT; with ``--loco`` the fit of each chromosome is in
``loco``), ``loco`` (``{chromosome: LocoChromosome}``, or null), ``results``
(`ResultsInfo`), ``timings_s`` (seconds by step) and ``warnings`` (the
warning messages logged during the run, in order).

The ``TypedDict`` classes below are the schema of the nested objects; the
keys of each are fixed.  ``tests/test_sumstats_cli.py`` checks a written
record against them.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import sys
from typing import Any, TypedDict

import numpy as np
import scipy

from pylmm import __version__

RUN_RECORD_SCHEMA = "pylmm.gwas.run/1"
"""The ``schema`` value of the run record; its version changes only if a key changes meaning."""

HASH_MAX_BYTES = 1 << 30
"""Input files up to this size (1 GiB) get a SHA-256 in the record; larger ones
only their size and modification time (hashing reads the whole file, ~0.5 s per GiB)."""


class FileInfo(TypedDict):
    """An input file: ``path`` as given, ``size`` (bytes), ``mtime`` (UTC, ISO 8601), ``sha256``
    (hex, or null above `HASH_MAX_BYTES` or if the file does not exist)."""

    path: str
    exists: bool
    size: int | None
    mtime: str | None
    sha256: str | None


class EnvironmentInfo(TypedDict):
    python: str
    numpy: str
    scipy: str
    platform: str


class SampleCounts(TypedDict):
    """Individuals at each step.  ``n_genotyped``: in the genotype file;
    ``n_analysed``: tested (``n`` of the null model); ``dropped``: by reason
    (``not_in_kinship``, ``missing_phenotype``, ``missing_covariate``)."""

    id_match: str | None
    n_genotyped: int
    n_analysed: int
    dropped: dict[str, int]
    phenotype: str | None


class ModelInfo(TypedDict):
    test: str
    reml: bool
    refit: bool
    legacy_fit: bool
    batch: bool
    block_size: int
    loco: bool
    missing_genotypes: str
    emma_coding: str | None
    intercept: bool
    q: int
    covariates: list[str] | None


class KinshipInfo(TypedDict):
    """``source``: ``file`` or ``loco``; ``format``: text, grm or rel (file);
    ``kind``: the ``--kinship-type`` (loco); ``mean_diag`` / ``centered_mean_diag``:
    `pylmm.kinship.kinship_scale` of the tested individuals' K (file)."""

    source: str
    format: str | None
    has_ids: bool | None
    kind: str | None
    mean_diag: float | None
    centered_mean_diag: float | None


class NullFit(TypedDict):
    """The null model: ``h``, ``sigma2`` (total), ``sigma_g2 = h sigma2``,
    ``sigma_e2 = (1 - h) sigma2``, ``log_likelihood`` and whether it is REML."""

    h: float
    sigma2: float
    sigma_g2: float
    sigma_e2: float
    log_likelihood: float
    reml: bool


class LocoChromosome(NullFit):
    n_snps_kinship: int
    mean_diag: float
    centered_mean_diag: float


class ResultsInfo(TypedDict):
    """The table: ``path``, ``format`` (``pylmm.sumstats/1`` or ``legacy``),
    ``columns``, ``n_variants`` (rows), ``n_tested``, ``not_tested`` by reason
    (``no_called_genotype``, ``monomorphic``, ``not_estimable``: collinear
    with the covariates or too few called individuals), ``lambda_gc`` (from
    the tested p-values; under ``--loco`` not a calibration check),
    ``max_neg_log_10_p_value``."""

    path: str
    format: str
    columns: list[str]
    effect_allele: str
    n_variants: int
    n_tested: int
    not_tested: dict[str, int]
    lambda_gc: float | None
    max_neg_log_10_p_value: float | None


def sidecar_path(out: str) -> str:
    """The run record's path for the results file ``out``.

    A trailing ``.gz``, then ``.tsv`` or ``.txt``, is removed and ``.json``
    added: ``results.tsv.gz`` -> ``results.json``, ``results.tsv`` ->
    ``results.json``, ``results`` -> ``results.json``.

    >>> sidecar_path("out/bmi.tsv.gz")
    'out/bmi.json'
    >>> sidecar_path("bmi.txt")
    'bmi.json'
    """
    base = out[:-3] if out.endswith(".gz") else out
    for ext in (".tsv", ".txt"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return base + ".json"


def utc_now() -> str:
    """The current time, UTC, ISO 8601 with seconds."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def file_info(path: str) -> FileInfo:
    """Size, modification time and (up to `HASH_MAX_BYTES`) SHA-256 of ``path``."""
    try:
        st = os.stat(path)
    except OSError:
        return FileInfo(path=path, exists=False, size=None, mtime=None, sha256=None)
    digest: str | None = None
    if st.st_size <= HASH_MAX_BYTES:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
    mtime = datetime.datetime.fromtimestamp(st.st_mtime, datetime.UTC)
    return FileInfo(
        path=path,
        exists=True,
        size=int(st.st_size),
        mtime=mtime.isoformat(timespec="seconds"),
        sha256=digest,
    )


def environment() -> EnvironmentInfo:
    """Python, numpy and scipy versions and the platform."""
    return EnvironmentInfo(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
    )


def null_fit(h: float, sigma2: float, ll: float, reml: bool) -> NullFit:
    """A `NullFit` from the fitted ``h``, ``sigma2`` and log-likelihood."""
    h, sigma2 = float(h), float(sigma2)
    return NullFit(
        h=h,
        sigma2=sigma2,
        sigma_g2=h * sigma2,
        sigma_e2=(1.0 - h) * sigma2,
        log_likelihood=float(ll),
        reml=bool(reml),
    )


def finite_or_none(x: float) -> float | None:
    """``x`` as a float, or None if it is nan or infinite (JSON has no nan)."""
    x = float(x)
    return x if np.isfinite(x) else None


class WarningCollector(logging.Handler):
    """Collect the messages of the WARNING (and above) records of a logger."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.ERROR:
            self.messages.append(record.getMessage())


def new_record(command: list[str]) -> dict[str, Any]:
    """A run record with every top-level key, filled as far as known at the start."""
    return {
        "schema": RUN_RECORD_SCHEMA,
        "pylmm_version": __version__,
        "status": "error",
        "error": None,
        "command": ["pylmm", *command],
        "started_at": utc_now(),
        "finished_at": None,
        "environment": environment(),
        "inputs": {},
        "samples": None,
        "model": None,
        "kinship": None,
        "null_fit": None,
        "loco": None,
        "results": None,
        "timings_s": {},
        "warnings": [],
    }


def write_record(path: str, record: dict[str, Any]) -> None:
    """Write ``record`` as JSON (sorted keys, indent 2; nan/inf are an error)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def command_line() -> list[str]:
    """``sys.argv[1:]``: the arguments of the running ``pylmm``."""
    return list(sys.argv[1:])
