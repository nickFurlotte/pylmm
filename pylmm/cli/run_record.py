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
keys of each are fixed.  ``tests/test_sumstats.py`` checks a written record
against them.  A number that is not finite (nan, inf: e.g. a log-likelihood
that overflowed) is written as ``null``, so the file is always valid JSON.

**What the record reveals.**  By default it holds the *base names* of the
input files (with their sizes, modification times and SHA-256) and the
command line with every path argument cut to its base name; the platform
string (OS, release, architecture) and the Python, numpy and scipy versions;
no host name, user name or directory.  It is meant to be shared with the
summary statistics.  ``--run-record-full-paths`` keeps the paths as typed.

**Where it goes.**  ``-o x.tsv.gz``, ``-o x.tsv``, ``-o x.txt`` and ``-o x``
all write ``x.json`` (`sidecar_path`), overwriting an existing one with a
warning; ``--run-record PATH`` names it, ``--no-run-record`` skips it, and
there is none by default when ``-o`` is not a regular file (``/dev/stdout``,
a pipe).  It is written to a temporary file and renamed, so a failed write
leaves no partial file, and a failure to write it never fails the run.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import logging
import os
import platform
import sys
import tempfile
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
    """An input file: ``path`` (its base name; as typed with ``--run-record-full-paths``),
    ``size`` (bytes), ``mtime`` (UTC, ISO 8601), ``sha256`` (hex, or null above
    `HASH_MAX_BYTES` or if the file does not exist)."""

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
    ``sigma_e2 = (1 - h) sigma2``, ``log_likelihood``, whether it is REML, and
    ``kind``: ``reml``, ``ml`` or ``lrt-reduced-ml`` (``--test lrt``: the ML fit
    of the model with ``dropped_null_directions`` null directions of the kinship,
    which the covariates span, left out, `pylmm.lmm.LMM.lrt_null_fit`).
    ``per_snp_null_fits``: under ``--remove-missing-genotypes``, each SNP with
    missing calls was tested at its own null fit on its called individuals, not
    at this one."""

    h: float | None
    sigma2: float | None
    sigma_g2: float | None
    sigma_e2: float | None
    log_likelihood: float | None
    reml: bool
    kind: str
    dropped_null_directions: int
    per_snp_null_fits: bool


class LocoChromosome(NullFit):
    n_snps_kinship: int
    mean_diag: float
    centered_mean_diag: float


class ResultsInfo(TypedDict):
    """The table: ``path``, ``format`` (``pylmm.sumstats/1`` or ``legacy``),
    ``columns``, ``effect_allele`` (the rule, or ``unknown: ...`` when the input
    names no alleles and the table's allele columns are NA), ``n_variants``
    (rows), ``n_tested``, ``not_tested`` by reason (``no_called_genotype``,
    ``monomorphic``, ``collinear_on_called``: the covariates lose rank on the
    called individuals, ``lrt_unbounded``: ``--test lrt`` without an ML maximum,
    ``not_estimable``: any other, e.g. collinear with the covariates or too few
    called individuals), ``lambda_gc`` (from
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


def basenames(argv: list[str]) -> list[str]:
    """``argv`` with every argument that names a path cut to its base name.

    >>> basenames(["--bfile", "/home/me/study/geno", "--pheno-col", "2", "-o", "out/r.tsv.gz"])
    ['--bfile', 'geno', '--pheno-col', '2', '-o', 'r.tsv.gz']
    """
    return [os.path.basename(a) if os.sep in a and not a.startswith("-") else a for a in argv]


def file_info(path: str, full_path: bool = False) -> FileInfo:
    """Size, modification time and (up to `HASH_MAX_BYTES`) SHA-256 of ``path``.

    The record keeps ``path``'s base name, or ``path`` as given with ``full_path``.
    """
    name = path if full_path else os.path.basename(path)
    try:
        st = os.stat(path)
    except OSError:
        return FileInfo(path=name, exists=False, size=None, mtime=None, sha256=None)
    digest: str | None = None
    if st.st_size <= HASH_MAX_BYTES:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
    mtime = datetime.datetime.fromtimestamp(st.st_mtime, datetime.UTC)
    return FileInfo(
        path=name,
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


def null_fit(
    h: float,
    sigma2: float,
    ll: float,
    reml: bool,
    kind: str = "reml",
    dropped_null_directions: int = 0,
    per_snp_null_fits: bool = False,
) -> NullFit:
    """A `NullFit` from the fitted ``h``, ``sigma2`` and log-likelihood (non-finite -> None)."""
    h, sigma2 = float(h), float(sigma2)
    return NullFit(
        h=finite_or_none(h),
        sigma2=finite_or_none(sigma2),
        sigma_g2=finite_or_none(h * sigma2),
        sigma_e2=finite_or_none((1.0 - h) * sigma2),
        log_likelihood=finite_or_none(float(ll)),
        reml=bool(reml),
        kind=kind,
        dropped_null_directions=int(dropped_null_directions),
        per_snp_null_fits=bool(per_snp_null_fits),
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


def json_safe(obj: Any) -> Any:
    """``obj`` with every non-finite float (nan, inf) replaced by None, recursively.

    numpy scalars become Python numbers, tuples lists.

    >>> json_safe({"a": float("nan"), "b": [1.5, float("inf")], "c": (2,)})
    {'a': None, 'b': [1.5, None], 'c': [2]}
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return finite_or_none(float(obj))
    return obj


def write_record(path: str, record: dict[str, Any]) -> None:
    """Write ``record`` as JSON (sorted keys, indent 2; non-finite numbers as null).

    The text is written to a temporary file next to ``path`` and renamed onto
    it, so ``path`` is either the old file or the complete new one.
    """
    text = json.dumps(json_safe(record), indent=2, sort_keys=True, allow_nan=False) + "\n"
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".%s." % os.path.basename(path), dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def command_line() -> list[str]:
    """``sys.argv[1:]``: the arguments of the running ``pylmm``."""
    return list(sys.argv[1:])
