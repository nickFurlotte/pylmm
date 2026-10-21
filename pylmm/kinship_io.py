# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""Kinship files of other tools, and eigendecomposition files with sample IDs.

pylmm's own kinship file is a whitespace-separated ``n x n`` text matrix with
an ID sidecar ``<kinship>.id`` (``pylmm kinship``; read by ``pylmm gwas
--kinship`` and `pylmm.plink_io.PlinkReader.read_kinship`).  This module adds:

**GCTA binary GRMs** (`read_grm`, `write_grm`, `read_grm_counts`): the
format of ``gcta --make-grm`` and ``plink2 --make-grm-bin``, three files
named by a prefix,

* ``<prefix>.grm.bin``: the lower triangle of K, diagonal included, row by
  row (``K[0,0], K[1,0], K[1,1], K[2,0], ...``), as little-endian float32:
  ``n (n + 1) / 2 x 4`` bytes;
* ``<prefix>.grm.N.bin``: the same layout, the number of SNPs behind each
  entry (float32);
* ``<prefix>.grm.id``: one ``FID IID`` per line (tab-separated, no header),
  the samples of K's rows.

**float32 precision.**  A GRM stores K in single precision: each entry is
rounded to about 7 significant digits (a relative error of at most
``2**-24``, about 6e-8).  A K written with `write_grm` and read back
is therefore not bit for bit the K computed in float64, and a scan run on it
differs from one run on the text kinship by rounding: see
``tests/helpers.py`` (``GRM_FLOAT32_RTOL_P``) for the measured size.  The text
format keeps all the digits.

**plink2 relationship matrices** (`read_rel`, read only): ``plink2
--make-rel`` writes ``<prefix>.rel`` (text; the lower triangle by default,
or the square matrix with ``square``), ``<prefix>.rel.bin`` (binary; float64,
or float32 with ``bin4``; square by default, or the triangle) and
``<prefix>.rel.id`` (a ``#FID IID`` or ``#IID`` header, then one sample per
line; without an FID column, the FID is ``0``, as plink2's ``.fam`` export
writes it).  The shape and precision are told from the number of values (or
bytes) and the number of samples.

**Eigendecompositions** (`read_eigen`, `write_eigen`): ``pylmm kinship
--eigen-npz`` writes a ``.npz`` with arrays ``Kva`` (eigenvalues), ``Kve``
(eigenvectors, one per column) and, from this version on, ``fid`` and
``iid`` (the samples of Kve's rows) and ``kinship_type``.  ``pylmm gwas
--eigen`` matches those IDs to the tested samples.  A ``.npz`` without IDs
(pylmm 1.0.0 to 1.x) and the text pair ``<path>.kva`` / ``<path>.kve``
(``--eigen-text``) still read; the text pair takes its IDs from ``<path>.id``
or ``<path>.grm.id`` (the ID file ``pylmm kinship -o <path>`` writes next to
them) if one exists.

`kinship_format` tells which reader a kinship path needs.

Examples
--------
>>> import os, tempfile
>>> import numpy as np
>>> from pylmm.kinship_io import read_grm, write_grm
>>> K = np.array([[1.0, 0.25, 0.1], [0.25, 1.0, -0.05], [0.1, -0.05, 1.0]])
>>> ids = [("F1", "I1"), ("F1", "I2"), ("F2", "I3")]
>>> prefix = os.path.join(tempfile.mkdtemp(), "study")
>>> write_grm(prefix, K, ids, n_snps=1000)
>>> sorted(os.listdir(os.path.dirname(prefix)))
['study.grm.N.bin', 'study.grm.bin', 'study.grm.id']
>>> K2, ids2 = read_grm(prefix)
>>> ids2 == ids, bool(np.allclose(K2, K, rtol=1e-7, atol=0)), bool((K2 == K).all())
(True, True, False)
"""

from __future__ import annotations

import gzip
import os
from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pylmm.samples import InputError, SampleID, read_id_file, write_id_file

FloatArray = NDArray[np.float64]

KinshipFormat = Literal["text", "grm", "rel"]

GRM_SUFFIXES = (".grm.bin", ".grm.N.bin", ".grm.id")
"""The files of a GCTA GRM, after its prefix."""

REL_SUFFIXES = (".rel", ".rel.gz", ".rel.bin")
"""The plink2 ``--make-rel`` matrix files `read_rel` reads (their IDs: ``.rel.id``)."""

_GRM_DTYPE = np.dtype("<f4")
_ASYMMETRY_RTOL = 1e-6
"""`write_grm` refuses a K with ``max |K - K'| > _ASYMMETRY_RTOL * max |K|``."""


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


def grm_prefix(path: str) -> str:
    """The prefix of a GCTA GRM: `path` without a ``.grm.bin`` / ``.grm.N.bin`` / ``.grm.id`` suffix.

    >>> grm_prefix("data/study.grm.bin"), grm_prefix("data/study")
    ('data/study', 'data/study')
    """
    for suffix in GRM_SUFFIXES:
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def kinship_format(path: str) -> KinshipFormat:
    """Which reader a kinship file needs, from its name.

    ``"grm"`` for a GCTA ``.grm.bin`` (or ``.grm.N.bin`` / ``.grm.id``) file,
    ``"rel"`` for a plink2 ``.rel``, ``.rel.gz`` or ``.rel.bin`` file, and
    ``"text"`` (pylmm's own text matrix) for anything else.

    >>> kinship_format("a.grm.bin"), kinship_format("a.rel.bin"), kinship_format("a.kin")
    ('grm', 'rel', 'text')
    """
    if path.endswith(GRM_SUFFIXES):
        return "grm"
    if path.endswith(REL_SUFFIXES):
        return "rel"
    return "text"


def read_kinship_ids(path: str) -> list[SampleID]:
    """Read a kinship's sample IDs: a GCTA ``.grm.id``, a plink2 ``.rel.id`` or a pylmm ``.id``.

    One sample per line, ``FID IID`` (further columns are ignored).  A first
    line starting with ``#`` is a plink2 header naming the columns
    (``#FID IID``, ``#IID``, ``#FID IID SID``, ...): the FID and IID are taken
    from the named columns, and without an FID column the FID is ``0`` (as
    plink2 writes it to a ``.fam`` or ``.grm.id``).

    Raises
    ------
    InputError
        If the file lists no sample, a line has too few fields, or a header
        has no IID column.
    """
    with open(path) as f:
        first = f.readline()
    if not first.startswith("#"):
        return read_id_file(path)
    names = first[1:].split()
    if "IID" not in names:
        raise InputError("ID file %s: the header %r has no IID column" % (path, first.strip()))
    i_iid = names.index("IID")
    i_fid = names.index("FID") if "FID" in names else None
    ids: list[SampleID] = []
    with open(path) as f:
        next(f)
        for lineno, line in enumerate(f, 2):
            v = line.split()
            if not v:
                continue
            if len(v) < len(names):
                raise InputError(
                    "ID file %s, line %d: expected %d fields (%s)"
                    % (path, lineno, len(names), " ".join(names))
                )
            ids.append((v[i_fid] if i_fid is not None else "0", v[i_iid]))
    if not ids:
        raise InputError("ID file %s lists no samples" % path)
    return ids


def _tri_size(n: int) -> int:
    return n * (n + 1) // 2


def _from_lower(tri: NDArray[np.floating], n: int) -> FloatArray:
    """The symmetric ``(n, n)`` float64 matrix of a row-major lower triangle (with diagonal)."""
    K = np.zeros((n, n))
    rows, cols = np.tril_indices(n)
    K[rows, cols] = tri
    K[cols, rows] = tri
    return K


# ---------------------------------------------------------------------------
# GCTA GRM
# ---------------------------------------------------------------------------


def _read_tri_bin(fn: str, n: int, what: str, id_file: str) -> FloatArray:
    """The float32 lower triangle of a GRM binary file, as an ``(n, n)`` float64 matrix."""
    want = _tri_size(n) * _GRM_DTYPE.itemsize
    try:
        size = os.path.getsize(fn)
    except OSError as e:
        raise InputError("%s %s: cannot read (%s)" % (what, fn, e)) from e
    if size != want:
        hint = ""
        if size == 2 * want:
            hint = " (twice that: a float64 triangle? GCTA writes float32)"
        elif size == n * n * 4 or size == n * n * 8:
            hint = " (that of a square %d x %d matrix: a plink2 .rel.bin? read it as such)" % (n, n)
        raise InputError(
            "%s %s has %d bytes, but %s lists %d samples, so it should hold their lower "
            "triangle as float32: %d x %d / 2 x 4 = %d bytes%s"
            % (what, fn, size, id_file, n, n, n + 1, want, hint)
        )
    tri = np.fromfile(fn, dtype=_GRM_DTYPE)
    bad = ~np.isfinite(tri)
    if bad.any():
        raise InputError("%s %s: %d non-finite values (nan/inf)" % (what, fn, int(bad.sum())))
    return _from_lower(tri, n)


def read_grm(prefix: str) -> tuple[FloatArray, list[SampleID]]:
    """Read a GCTA binary GRM: ``prefix.grm.bin`` and ``prefix.grm.id``.

    Parameters
    ----------
    prefix : str
        The GRM's prefix, as given to ``gcta --grm`` (a path ending in
        ``.grm.bin`` is accepted too).

    Returns
    -------
    K : numpy.ndarray
        The ``(n, n)`` kinship, float64 (the float32 values of the file).
    ids : list of SampleID
        ``(FID, IID)`` of K's rows, from ``prefix.grm.id``.

    Raises
    ------
    InputError
        If a file is missing, the ``.grm.bin`` size is not that of the
        lower triangle of the ``.grm.id`` samples in float32, or it holds
        nan/inf values.

    Examples
    --------
    See the module docstring.
    """
    prefix = grm_prefix(prefix)
    id_file, bin_file = prefix + ".grm.id", prefix + ".grm.bin"
    for fn in (bin_file, id_file):
        if not os.path.isfile(fn):
            raise InputError("GRM %s: no file %s" % (prefix, fn))
    ids = read_kinship_ids(id_file)
    K = _read_tri_bin(bin_file, len(ids), "GRM file", id_file)
    return K, ids


def read_grm_counts(prefix: str) -> FloatArray:
    """Read ``prefix.grm.N.bin``: the number of SNPs behind each GRM entry, ``(n, n)``.

    Raises
    ------
    InputError
        If the file or ``prefix.grm.id`` is missing, or the size is wrong.
    """
    prefix = grm_prefix(prefix)
    id_file, n_file = prefix + ".grm.id", prefix + ".grm.N.bin"
    for fn in (n_file, id_file):
        if not os.path.isfile(fn):
            raise InputError("GRM %s: no file %s" % (prefix, fn))
    return _read_tri_bin(n_file, len(read_kinship_ids(id_file)), "GRM count file", id_file)


def write_grm(
    prefix: str, K: ArrayLike, ids: Sequence[SampleID], n_snps: float | ArrayLike
) -> None:
    """Write K as a GCTA binary GRM: ``prefix.grm.bin``, ``prefix.grm.N.bin``, ``prefix.grm.id``.

    The lower triangle of K is stored in float32 (about 7 significant
    digits; see the module docstring), which GCTA, plink2 and other tools
    read.

    Parameters
    ----------
    prefix : str
        The output prefix (a ``.grm.bin`` suffix is dropped); existing files
        are overwritten.
    K : array_like
        The ``(n, n)`` symmetric kinship.
    ids : sequence of SampleID
        ``(FID, IID)`` of K's rows; no whitespace inside a field.
    n_snps : float or array_like
        The number of SNPs behind K, for ``prefix.grm.N.bin``: one number
        for every entry (pylmm's kinships divide every entry by the same
        ``m``), or an ``(n, n)`` matrix.

    Raises
    ------
    ValueError
        If K is not square, not symmetric (up to a relative 1e-6), has
        nan/inf values, or its size or that of `n_snps` does not match `ids`,
        or an ID field is empty or has whitespace.
    """
    prefix = grm_prefix(prefix)
    K = np.asarray(K, dtype=np.float64)
    n = len(ids)
    if K.ndim != 2 or K.shape != (n, n):
        raise ValueError("K has shape %s but %d IDs are given" % (K.shape, n))
    if not np.isfinite(K).all():
        raise ValueError("K has nan/inf values")
    scale = float(np.abs(K).max()) if n else 0.0
    asym = float(np.abs(K - K.T).max()) if n else 0.0
    if asym > _ASYMMETRY_RTOL * max(scale, 1e-300):
        raise ValueError("K is not symmetric: max |K - K'| = %g" % asym)
    for sample in ids:
        if len(sample) != 2 or any((not str(x)) or len(str(x).split()) != 1 for x in sample):
            raise ValueError("sample ID %r: expected (FID, IID) without whitespace" % (sample,))
    N = np.asarray(n_snps, dtype=np.float64)
    if N.ndim == 0:
        N = np.full((n, n), float(N))
    elif N.shape != (n, n):
        raise ValueError("n_snps has shape %s; expected a number or (%d, %d)" % (N.shape, n, n))
    rows, cols = np.tril_indices(n)
    K[rows, cols].astype(_GRM_DTYPE).tofile(prefix + ".grm.bin")
    N[rows, cols].astype(_GRM_DTYPE).tofile(prefix + ".grm.N.bin")
    write_id_file(prefix + ".grm.id", [(str(f), str(i)) for f, i in ids])


def round_like_grm(K: ArrayLike) -> FloatArray:
    """K as `write_grm` stores it: each entry rounded to float32, back in float64.

    ``read_grm`` of a GRM written from K returns exactly this (``pylmm
    kinship --out-format grm`` decomposes it for ``--eigen-npz``, so the
    eigendecomposition is that of the K ``pylmm gwas --grm`` reads).

    >>> import numpy as np
    >>> float(round_like_grm(np.array([[0.1]]))[0, 0]) == float(np.float32(0.1))
    True
    """
    return np.asarray(K, dtype=np.float64).astype(_GRM_DTYPE).astype(np.float64)


# ---------------------------------------------------------------------------
# plink2 .rel
# ---------------------------------------------------------------------------


def rel_id_path(path: str) -> str:
    """The ID file of a plink2 relationship matrix: ``<prefix>.rel.id``.

    >>> rel_id_path("out.rel.bin"), rel_id_path("out.rel")
    ('out.rel.id', 'out.rel.id')
    """
    for suffix in REL_SUFFIXES:
        if path.endswith(suffix):
            return path[: -len(suffix)] + ".rel.id"
    return path + ".id"


def _square_or_triangle(values: FloatArray, n: int, fn: str, id_file: str) -> FloatArray:
    if values.size == n * n:
        K = values.reshape(n, n)
        upper = K[np.triu_indices(n, 1)]
        if n > 1 and not upper.any() and K[np.tril_indices(n, -1)].any():
            K = _from_lower(K[np.tril_indices(n)], n)  # plink2 "square0": upper half zero
        return K
    if values.size == _tri_size(n):
        return _from_lower(values, n)
    raise InputError(
        "relationship matrix %s has %d values, but %s lists %d samples: expected %d "
        "(square) or %d (lower triangle)" % (fn, values.size, id_file, n, n * n, _tri_size(n))
    )


def read_rel(path: str) -> tuple[FloatArray, list[SampleID]]:
    """Read a plink2 ``--make-rel`` relationship matrix and its ``.rel.id``.

    Parameters
    ----------
    path : str
        ``<prefix>.rel`` or ``<prefix>.rel.gz`` (text: square or lower
        triangle, whitespace-separated) or ``<prefix>.rel.bin`` (float64, or
        float32 from ``bin4``; square or lower triangle).  The samples are read
        from ``<prefix>.rel.id``.

    Returns
    -------
    K : numpy.ndarray
        The ``(n, n)`` kinship, float64.
    ids : list of SampleID
        ``(FID, IID)`` of K's rows (FID ``0`` when the file has none).

    Raises
    ------
    InputError
        If a file is missing, the number of values (or bytes) fits neither
        shape for the samples of the ``.rel.id``, or a value is not a finite
        number.

    Examples
    --------
    >>> import os, tempfile
    >>> d = tempfile.mkdtemp()
    >>> with open(os.path.join(d, "x.rel"), "w") as f:
    ...     _ = f.write("1\\n0.25\\t1\\n")
    >>> with open(os.path.join(d, "x.rel.id"), "w") as f:
    ...     _ = f.write("#IID\\nI1\\nI2\\n")
    >>> K, ids = read_rel(os.path.join(d, "x.rel"))
    >>> K.tolist(), ids
    ([[1.0, 0.25], [0.25, 1.0]], [('0', 'I1'), ('0', 'I2')])
    """
    id_file = rel_id_path(path)
    for fn in (path, id_file):
        if not os.path.isfile(fn):
            raise InputError("plink2 relationship matrix %s: no file %s" % (path, fn))
    ids = read_kinship_ids(id_file)
    n = len(ids)
    if path.endswith(".rel.bin"):
        size = os.path.getsize(path)
        sizes = {
            n * n * 8: ("<f8", "square"),
            n * n * 4: ("<f4", "square"),
            _tri_size(n) * 8: ("<f8", "triangle"),
            _tri_size(n) * 4: ("<f4", "triangle"),
        }
        if size not in sizes:
            raise InputError(
                "relationship matrix %s has %d bytes, but %s lists %d samples: expected %s"
                % (
                    path,
                    size,
                    id_file,
                    n,
                    " or ".join(
                        "%d (%s, %s)" % (s, sh, dt) for s, (dt, sh) in sorted(sizes.items())
                    ),
                )
            )
        values = np.fromfile(path, dtype=sizes[size][0]).astype(np.float64)
    else:
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt") as f:
                values = np.array([float(x) for line in f for x in line.split()])
        except (ValueError, OSError, EOFError) as e:
            raise InputError("relationship matrix %s: cannot read (%s)" % (path, e)) from e
    if not np.isfinite(values).all():
        raise InputError("relationship matrix %s has nan/inf values" % path)
    return _square_or_triangle(values, n, path, id_file), ids


def read_kinship_with_ids(path: str) -> tuple[FloatArray, list[SampleID]]:
    """Read a GCTA GRM (``.grm.bin``) or a plink2 ``.rel`` file, by `kinship_format`.

    Raises
    ------
    ValueError
        For a pylmm text kinship (``kinship_format(path) == "text"``): read
        that with its own reader.
    InputError
        As `read_grm` / `read_rel`.
    """
    fmt = kinship_format(path)
    if fmt == "grm":
        return read_grm(path)
    if fmt == "rel":
        return read_rel(path)
    raise ValueError("%s is not a GCTA GRM or plink2 .rel file" % path)


# ---------------------------------------------------------------------------
# eigendecompositions
# ---------------------------------------------------------------------------


class Eigen(NamedTuple):
    """An eigendecomposition read by `read_eigen`.

    Attributes
    ----------
    kva : numpy.ndarray
        Eigenvalues, ``(n,)``.
    kve : numpy.ndarray
        Eigenvectors, one per column, ``(n, n)``.
    ids : list of SampleID or None
        ``(FID, IID)`` of Kve's rows, or None if the file has none.
    kinship_type : str or None
        The ``--kinship-type`` of the kinship, if recorded.
    """

    kva: FloatArray
    kve: FloatArray
    ids: list[SampleID] | None
    kinship_type: str | None


def write_eigen(
    path: str,
    kva: ArrayLike,
    kve: ArrayLike,
    ids: Sequence[SampleID] | None = None,
    kinship_type: str | None = None,
) -> None:
    """Write an eigendecomposition to the ``.npz`` file `path` (``np.savez`` adds ``.npz``).

    Arrays ``Kva`` and ``Kve`` (as pylmm 1.0.0 wrote them), plus ``fid`` and
    ``iid`` (the samples of Kve's rows) when `ids` is given and
    ``kinship_type`` when given.  Everything is plain arrays (no pickle).

    >>> import os, tempfile
    >>> import numpy as np
    >>> fn = os.path.join(tempfile.mkdtemp(), "e.npz")
    >>> write_eigen(fn, [1.0, 2.0], np.eye(2), ids=[("F", "a"), ("F", "b")])
    >>> e = read_eigen(fn)
    >>> e.ids, e.kva.tolist()
    ([('F', 'a'), ('F', 'b')], [1.0, 2.0])
    """
    arrays: dict[str, NDArray[np.generic]] = {
        "Kva": np.asarray(kva, dtype=np.float64),
        "Kve": np.asarray(kve, dtype=np.float64),
    }
    if ids is not None:
        arrays["fid"] = np.array([str(f) for f, _ in ids], dtype=np.str_)
        arrays["iid"] = np.array([str(i) for _, i in ids], dtype=np.str_)
    if kinship_type is not None:
        arrays["kinship_type"] = np.array(kinship_type, dtype=np.str_)
    np.savez(path, **arrays)  # type: ignore[arg-type]


def read_eigen(path: str) -> Eigen:
    """Read an eigendecomposition: a ``.npz`` file, or the text files ``path.kva`` / ``path.kve``.

    A ``.npz`` holds arrays ``Kva`` and ``Kve`` and, if written by this
    version of ``pylmm kinship --eigen-npz``, ``fid`` / ``iid`` and
    ``kinship_type``.  The text pair has no IDs of its own; they are read
    from ``path.id`` or ``path.grm.id`` (the ID file ``pylmm kinship -o
    path`` writes, as text or ``--out-format grm``) if one exists.  The
    shapes are not checked against each other, except that the IDs must
    name the rows of Kve.

    Raises
    ------
    InputError
        If the files cannot be read, a ``.npz`` lacks ``Kva`` or ``Kve``,
        its ``fid`` / ``iid`` arrays are not two equal-length 1-d arrays,
        or the number of IDs is not the number of rows of Kve.
    """
    ids: list[SampleID] | None = None
    kinship_type: str | None = None
    try:
        if path.endswith(".npz"):
            with np.load(path) as z:
                if "Kva" not in z or "Kve" not in z:
                    raise InputError(
                        "eigen file %s must hold arrays Kva and Kve, has %s"
                        % (path, ", ".join(z.files))
                    )
                kva, kve = z["Kva"], z["Kve"]
                if ("fid" in z) != ("iid" in z):
                    raise InputError("eigen file %s has only one of the arrays fid, iid" % path)
                if "iid" in z:
                    fid, iid = z["fid"], z["iid"]
                    if fid.ndim != 1 or fid.shape != iid.shape:
                        raise InputError(
                            "eigen file %s: arrays fid and iid of shapes %s and %s, expected "
                            "two vectors of one length" % (path, fid.shape, iid.shape)
                        )
                    ids = [(str(f), str(i)) for f, i in zip(fid.tolist(), iid.tolist())]
                if "kinship_type" in z:
                    kinship_type = str(z["kinship_type"])
        else:
            kva, kve = np.loadtxt(path + ".kva", ndmin=1), np.loadtxt(path + ".kve", ndmin=2)
            for id_file in (path + ".id", path + ".grm.id"):
                if os.path.isfile(id_file):
                    ids = read_kinship_ids(id_file)
                    break
    except OSError as e:
        raise InputError(
            "cannot read the --eigen decomposition %s: %s (expected a .npz file or %s.kva and "
            "%s.kve)" % (path, e, path, path)
        ) from e
    kva, kve = np.asarray(kva, dtype=np.float64), np.asarray(kve, dtype=np.float64)
    if ids is not None and (kve.ndim != 2 or kve.shape[0] != len(ids)):
        raise InputError(
            "eigen file %s: %d sample IDs but eigenvectors of shape %s"
            % (path, len(ids), kve.shape)
        )
    return Eigen(kva, kve, ids, kinship_type)
