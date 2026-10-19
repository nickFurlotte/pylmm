# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""Sample IDs, and the alignment of genotype, phenotype, covariate and kinship data by ID.

Every per-individual input of a mixed-model scan (the genotype file, the
phenotype and covariate tables, the kinship matrix) lists its samples in its
own order, and the lists need not agree: a table may skip individuals or add
strangers, and a kinship may have been computed on another set.  This module
matches them by ID, never by row position, so that a reordered or partial
file cannot silently pair one individual's phenotype with another's genotype.

A sample ID is the pair ``(FID, IID)`` of PLINK's ``.fam`` file (`SampleID`).
Matching is on both fields by default (``match="fid_iid"``); ``match="iid"``
matches on the IID alone, for files whose FIDs disagree (e.g. one source uses
``0`` or repeats the IID as the FID).  Duplicate IDs in a source (duplicate
IIDs, with ``match="iid"``) are an error, because the match would be
ambiguous.

The pieces:

* `align_samples` aligns several sources at once and returns an `Alignment`:
  the samples present in all of them (in the order of the reference source,
  the genotype file by default), the rows of each source to take, and a
  report of the samples dropped from each source and why (`DroppedSamples`).
* `match_index` is the one-to-one lookup underneath: the row of each target
  ID in a source, or -1.
* `read_sample_table` reads a PLINK-style ``FID IID v1 v2 ...`` phenotype or
  covariate file, with or without a ``FID IID name1 name2 ...`` header, and
  can select columns by name (with a header) or by 0-based index.
* `read_id_file` / `write_id_file` read and write a list of IDs, one
  ``FID IID`` per line (PLINK ``--keep`` style; GCTA's ``.grm.id``).
  ``pylmmKinship`` writes one next to each kinship matrix it computes,
  ``<kinship>.id`` (`kinship_id_path`), so that ``pylmmGWAS`` can align the
  matrix by ID instead of assuming that its rows are in genotype order.

Examples
--------
>>> from pylmm.samples import align_samples
>>> geno = [("F1", "I1"), ("F1", "I2"), ("F2", "I3"), ("F3", "I4")]
>>> kin = [("F3", "I4"), ("F1", "I1"), ("F2", "I3"), ("F9", "I9")]
>>> al = align_samples({"genotypes": geno, "kinship": kin})
>>> al.ids
[('F1', 'I1'), ('F2', 'I3'), ('F3', 'I4')]
>>> al.rows["genotypes"].tolist(), al.rows["kinship"].tolist()
([0, 2, 3], [1, 2, 0])
>>> for line in al.report():
...     print(line)
1 of 4 samples of the genotypes are not in the kinship and are dropped: F1 I2
1 of 4 samples of the kinship are not in the genotypes and are ignored: F9 I9

Match on the IID alone when the FIDs disagree:

>>> al = align_samples({"genotypes": geno, "phenotypes": [("0", "I2"), ("0", "I1")]}, match="iid")
>>> al.ids, al.rows["phenotypes"].tolist()
([('F1', 'I1'), ('F1', 'I2')], [1, 0])
"""

from __future__ import annotations

import os
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

# An (FID, IID) pair.
SampleID = tuple[str, str]

# How IDs are matched: on (FID, IID), or on the IID alone.
MATCH_MODES = ("fid_iid", "iid")

# How many example IDs a message lists.
_N_EXAMPLES = 5


def _check_match(match: str) -> None:
    if match not in MATCH_MODES:
        raise ValueError(
            "match=%r: use one of %s" % (match, ", ".join(repr(m) for m in MATCH_MODES))
        )


def _key(sample: Any, match: str) -> Hashable:
    """The matching key of one ID: itself, or its IID (second field) for ``match="iid"``."""
    if match == "iid":
        if not isinstance(sample, tuple) or len(sample) != 2:
            raise ValueError("match='iid' needs (FID, IID) pairs, got %r" % (sample,))
        return cast("Hashable", sample[1])
    return cast("Hashable", sample)


def format_id(sample: Any) -> str:
    """``"FID IID"`` for an ``(FID, IID)`` pair, ``str(sample)`` otherwise.

    >>> format_id(("F1", "I1"))
    'F1 I1'
    """
    if isinstance(sample, tuple):
        return " ".join(str(x) for x in sample)
    return str(sample)


def _examples(ids: Sequence[Any]) -> str:
    s = ", ".join(format_id(i) for i in ids[:_N_EXAMPLES])
    return s + (", ..." if len(ids) > _N_EXAMPLES else "")


def _lookup(ids: Sequence[Any], match: str, source: str) -> dict[Hashable, int]:
    """{key: row} of a source; raises ValueError for a duplicate key."""
    out: dict[Hashable, int] = {}
    for i, sample in enumerate(ids):
        k = _key(sample, match)
        if k in out:
            what = "IID %s" % k if match == "iid" else "sample %s" % format_id(sample)
            raise ValueError(
                "%s lists %s twice (rows %d and %d)%s"
                % (
                    source,
                    what,
                    out[k] + 1,
                    i + 1,
                    "; match on FID and IID instead" if match == "iid" else "",
                )
            )
        out[k] = i
    return out


def match_index(
    target: Sequence[Any],
    source: Sequence[Any],
    *,
    match: str = "fid_iid",
    target_name: str = "target",
    source_name: str = "source",
) -> NDArray[np.intp]:
    """Row of each ``target`` ID in ``source``, or -1 where it is absent.

    Parameters
    ----------
    target, source : sequence of SampleID
        The IDs to look up, and the IDs of the source's rows.  Any hashable
        IDs work with ``match="fid_iid"`` (EMMA positions, for example).
    match : {"fid_iid", "iid"}, default "fid_iid"
        Match on the whole ID, or on the IID alone.
    target_name, source_name : str
        How the two lists are named in error messages.

    Returns
    -------
    numpy.ndarray of intp, shape (len(target),)

    Raises
    ------
    ValueError
        If either list has a duplicate ID (a duplicate IID with
        ``match="iid"``), or `match` is unknown.

    Examples
    --------
    >>> match_index([("F", "a"), ("F", "b")], [("F", "b"), ("F", "c")]).tolist()
    [-1, 0]
    """
    _check_match(match)
    _lookup(target, match, target_name)  # duplicates in the target are an error too
    rows = _lookup(source, match, source_name)
    return np.array([rows.get(_key(s, match), -1) for s in target], dtype=np.intp)


@dataclass(frozen=True)
class DroppedSamples:
    """Samples of one source that are not used, and why.

    Attributes
    ----------
    source : str
        The source the samples are listed in.
    reason : str
        Why they are not used, e.g. ``"not in the kinship"``.
    ids : tuple of SampleID
        The samples, in the source's order.
    n_source : int
        Number of samples in the source.
    action : str
        ``"dropped"`` (a reference sample, removed from the analysis) or
        ``"ignored"`` (an extra sample of another source).
    """

    source: str
    reason: str
    ids: tuple[Any, ...]
    n_source: int
    action: str = "dropped"

    def __str__(self) -> str:
        return "%d of %d samples of the %s are %s and are %s: %s" % (
            len(self.ids),
            self.n_source,
            self.source,
            self.reason,
            self.action,
            _examples(self.ids),
        )


@dataclass(frozen=True)
class Alignment:
    """The samples shared by several sources, and how to subset each source to them.

    Made by `align_samples`.

    Attributes
    ----------
    ids : list of SampleID
        The kept samples (present in every source), in the order of the
        reference source.
    rows : dict of str -> numpy.ndarray of intp
        For each source, the row of each kept sample in that source:
        ``data[rows[source]]`` puts the source's data in the order of `ids`.
    dropped : list of DroppedSamples
        The samples that are not used, per source and reason (empty entries
        are left out).
    match : str
        The match mode used (``"fid_iid"`` or ``"iid"``).
    reference : str
        The source whose order `ids` follows.
    """

    ids: list[Any]
    rows: dict[str, NDArray[np.intp]]
    dropped: list[DroppedSamples] = field(default_factory=list)
    match: str = "fid_iid"
    reference: str = ""

    @property
    def n(self) -> int:
        """Number of kept samples."""
        return len(self.ids)

    def is_identity(self, source: str) -> bool:
        """Whether `source` is already exactly the kept samples, in order (no subsetting needed)."""
        r = self.rows[source]
        return bool(np.array_equal(r, np.arange(len(r)))) and not any(
            d.source == source for d in self.dropped
        )

    def take(self, source: str, data: Any) -> Any:
        """The rows of `data` (one row per sample of `source`) for the kept samples, in order."""
        return np.asarray(data)[self.rows[source]]

    def take_square(self, source: str, K: Any) -> NDArray[Any]:
        """A square matrix over the samples of `source` (a kinship), restricted and reordered."""
        r = self.rows[source]
        out: NDArray[Any] = np.asarray(K)[np.ix_(r, r)]
        return out

    def report(self) -> list[str]:
        """One line per `DroppedSamples` entry (for warnings and logs)."""
        return [str(d) for d in self.dropped]


def align_samples(
    sources: Mapping[str, Sequence[Any]],
    *,
    reference: str | None = None,
    match: str = "fid_iid",
) -> Alignment:
    """Align several per-sample sources by ID.

    The kept samples are those of the `reference` source that appear in
    every other source, in the reference's order.  Each reference sample
    missing from another source is reported as dropped (reason ``"not in the
    <source>"``); each sample of another source that is not in the reference
    is reported as ignored.

    Parameters
    ----------
    sources : mapping of str -> sequence of SampleID
        The IDs of each source's rows, by source name (the name is used in
        the report and as the key of `Alignment.rows`), e.g.
        ``{"genotype file": reader.indivs, "kinship": read_id_file(...)}``.
    reference : str, optional
        The source whose order is kept.  Default: the first one.
    match : {"fid_iid", "iid"}, default "fid_iid"
        Match on (FID, IID), or on the IID alone.

    Returns
    -------
    Alignment

    Raises
    ------
    ValueError
        If a source has a duplicate ID (duplicate IID with ``match="iid"``),
        `reference` is not a source, `match` is unknown, or no sample is in
        every source (the message gives an example ID of each source).

    Examples
    --------
    See the module docstring.
    """
    _check_match(match)
    if not sources:
        raise ValueError("align_samples needs at least one source")
    names = list(sources)
    ref = names[0] if reference is None else reference
    if ref not in sources:
        raise ValueError("reference %r is not one of the sources %s" % (ref, ", ".join(names)))
    ref_ids = list(sources[ref])
    lookups = {name: _lookup(sources[name], match, "the %s" % name) for name in names}
    ref_keys = [_key(s, match) for s in ref_ids]

    present = np.ones(len(ref_ids), dtype=bool)
    dropped: list[DroppedSamples] = []
    for name in names:
        if name == ref:
            continue
        miss = np.array([k not in lookups[name] for k in ref_keys], dtype=bool)
        if miss.any():
            dropped.append(
                DroppedSamples(
                    ref,
                    "not in the %s" % name,
                    tuple(s for s, m in zip(ref_ids, miss) if m),
                    len(ref_ids),
                )
            )
        present &= ~miss
    ref_set = set(ref_keys)
    for name in names:
        if name == ref:
            continue
        extra = tuple(s for s in sources[name] if _key(s, match) not in ref_set)
        if extra:
            dropped.append(
                DroppedSamples(
                    name, "not in the %s" % ref, extra, len(sources[name]), action="ignored"
                )
            )

    if ref_ids and not present.any():
        raise ValueError(
            "no sample is in all of %s (matched on %s), e.g. %s"
            % (
                ", ".join(names),
                "the IID" if match == "iid" else "FID and IID",
                "; ".join(
                    "%s: %s" % (name, format_id(sources[name][0]) if len(sources[name]) else "none")
                    for name in names
                ),
            )
        )
    ids = [s for s, p in zip(ref_ids, present) if p]
    keys = [k for k, p in zip(ref_keys, present) if p]
    rows = {name: np.array([lookups[name][k] for k in keys], dtype=np.intp) for name in names}
    return Alignment(ids=ids, rows=rows, dropped=dropped, match=match, reference=ref)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def kinship_id_path(k_file: str) -> str:
    """The ID sidecar of a kinship file: ``k_file + ".id"``.

    >>> kinship_id_path("out.kin")
    'out.kin.id'
    """
    return k_file + ".id"


def read_id_file(path: str) -> list[SampleID]:
    """Read sample IDs, one ``FID IID`` per line (further columns are ignored).

    A ``.fam`` file can be read this way too.  Blank lines are skipped.

    Raises
    ------
    ValueError
        If a line has fewer than two fields, or the file has no IDs.
    """
    ids: list[SampleID] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            v = line.split()
            if not v:
                continue
            if len(v) < 2:
                raise ValueError("ID file %s, line %d: expected FID and IID" % (path, lineno))
            ids.append((v[0], v[1]))
    if not ids:
        raise ValueError("ID file %s lists no samples" % path)
    return ids


def write_id_file(path: str, ids: Sequence[SampleID]) -> None:
    """Write sample IDs, one ``FID IID`` per line (tab-separated), as `read_id_file` reads them."""
    with open(path, "w") as f:
        for fid, iid in ids:
            f.write("%s\t%s\n" % (fid, iid))


def read_kinship_ids(k_file: str) -> list[SampleID] | None:
    """The IDs of a kinship's rows from its sidecar ``k_file + ".id"``, or None if there is none."""
    path = kinship_id_path(k_file)
    if not os.path.isfile(path):
        return None
    return read_id_file(path)


class SampleTable(NamedTuple):
    """A PLINK-style per-sample table, as read by `read_sample_table`.

    Attributes
    ----------
    ids : list of SampleID
        ``(FID, IID)`` of each row, in file order.
    names : list of str or None
        Column names from the header line (selected columns only), or None
        if the file has no header.
    values : numpy.ndarray, shape (n_rows, n_columns)
        float64; nan where missing.
    """

    ids: list[SampleID]
    names: list[str] | None
    values: NDArray[np.float64]


def _is_header(v: list[str]) -> bool:
    return len(v) >= 2 and v[0].upper() == "FID" and v[1].upper() == "IID"


def read_sample_table(
    path: str,
    *,
    what: str = "phenotype",
    is_missing: Callable[[str], bool] | None = None,
    columns: Sequence[str | int] | None = None,
) -> SampleTable:
    """Read a ``FID IID v1 v2 ...`` table, with or without a header.

    A first line whose first two fields are ``FID IID`` (any case) is a
    header; its other fields name the value columns.  Without one the file
    is a plain PLINK ``--pheno`` / ``--covar`` table.

    Parameters
    ----------
    path : str
        The file (whitespace-separated).
    what : str, default "phenotype"
        How the file is named in error messages (``"phenotype"`` or
        ``"covariate"``).
    is_missing : callable, optional
        Whether a token is a missing value.  Default: ``NA`` only.
    columns : sequence of str or int, optional
        The value columns to return, in this order: names from the header,
        or 0-based indices of the value columns (after FID and IID).
        Default: all of them.

    Returns
    -------
    SampleTable

    Raises
    ------
    ValueError
        For a row without a value, a non-numeric value, rows of different
        lengths, a file with no data rows, a column name that is not in the
        header (or a name given for a headerless file), an index out of
        range, or a column selected twice.
    """
    missing = is_missing if is_missing is not None else (lambda x: x == "NA")
    ids: list[SampleID] = []
    rows: list[list[float]] = []
    header: list[str] | None = None
    with open(path) as f:
        first = True
        for lineno, line in enumerate(f, 1):
            v = line.split()
            if not v:
                continue
            if first and _is_header(v):
                header = v[2:]
                first = False
                continue
            first = False
            if len(v) < 3:
                raise ValueError(
                    "%s file %s, line %d: expected FID IID and at least one value"
                    % (what, path, lineno)
                )
            try:
                rows.append([np.nan if missing(x) else float(x) for x in v[2:]])
            except ValueError:
                raise ValueError(
                    "%s file %s, line %d: a value is not a number (missing values are "
                    "NA%s)" % (what, path, lineno, " or -9" if what == "phenotype" else "")
                ) from None
            ids.append((v[0], v[1]))
    widths = {len(r) for r in rows}
    if len(widths) > 1:
        raise ValueError(
            "%s file %s: rows have different numbers of values (%s)"
            % (what, path, ", ".join(str(w) for w in sorted(widths)))
        )
    if not rows:
        raise ValueError("%s file %s has no data rows" % (what, path))
    ncol = len(rows[0])
    if header is not None and len(header) != ncol:
        raise ValueError(
            "%s file %s: the header names %d columns but the rows have %d values"
            % (what, path, len(header), ncol)
        )
    values = np.array(rows, dtype=np.float64)
    names = header
    if columns is not None:
        idx = _column_indices(columns, header, ncol, what, path)
        values = values[:, idx]
        names = None if header is None else [header[i] for i in idx]
    return SampleTable(ids, names, values)


def _column_indices(
    columns: Sequence[str | int], header: list[str] | None, ncol: int, what: str, path: str
) -> list[int]:
    idx: list[int] = []
    for c in columns:
        if isinstance(c, str):
            if header is None:
                raise ValueError(
                    "%s file %s has no FID IID header line, so column %r cannot be selected by "
                    "name; use a 0-based index" % (what, path, c)
                )
            if c not in header:
                raise ValueError(
                    "%s file %s has no column %r; its columns are %s"
                    % (what, path, c, ", ".join(header))
                )
            if header.count(c) > 1:
                raise ValueError("%s file %s has two columns named %r" % (what, path, c))
            idx.append(header.index(c))
        else:
            i = int(c)
            if not 0 <= i < ncol:
                raise ValueError(
                    "%s file %s: column index %d is out of range (%d value columns, 0-based)"
                    % (what, path, i, ncol)
                )
            idx.append(i)
    if len(set(idx)) != len(idx):
        raise ValueError("%s file %s: a column is selected twice (%s)" % (what, path, columns))
    return idx
