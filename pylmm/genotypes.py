# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""Genotype sources: blocks of SNPs with their variant metadata.

A *genotype source* is anything that can say who its samples are, describe its
variants and hand out its genotypes in blocks of SNPs.  Code that scans a
genome (a GWAS, a kinship, leave-one-chromosome-out) is written once against
the :class:`GenotypeSource` protocol, and works with any file format a source
exists for:

* :class:`pylmm.plink_io.PlinkReader` (PLINK BED, TPED and EMMA files);
* :class:`ArraySource`, an ``(n, m)`` numpy array already in memory, e.g.
  genotypes loaded from a VCF, BGEN or VCF Zarr store with another library.

The pieces:

``Variants``
    The variant metadata of a source (or of one block), one entry per SNP,
    as columns: ``snp`` (the ID) and, when the format has them, ``chrom``,
    ``cm``, ``bp``, ``a1`` and ``a2``, PLINK's ``.bim`` columns.  A column the
    format does not have is None (EMMA input has SNP names only).
``Variant``
    One row of `Variants` (``variants[k]``), a ``NamedTuple``.
``GenotypeSource``
    The protocol: ``sample_ids``, ``num_samples``, ``variants`` and
    ``iter_blocks(block_size)``, which yields ``(G, variants)``: a float64
    ``(num_samples, b)`` genotype block and the `Variants` of its ``b`` SNPs.
``normalize_genotype``
    The per-SNP standardisation every source applies when asked to (mean
    imputation of missing calls, centring, unit variance).

Examples
--------
Wrap a genotype matrix, then read it in blocks of two SNPs:

>>> import numpy as np
>>> from pylmm.genotypes import ArraySource, Variants
>>> G = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 0.0], [2.0, 0.0, np.nan], [1.0, 2.0, 1.0]])
>>> v = Variants(snp=["rs1", "rs2", "rs3"], chrom=["1", "1", "2"], bp=[100, 200, 50])
>>> src = ArraySource(G, variants=v, norm_genotype=False)
>>> src.num_samples, len(src.variants)
(4, 3)
>>> for block, bv in src.iter_blocks(2):
...     print(block.shape, bv.snp.tolist(), bv.chrom.tolist())
(4, 2) ['rs1', 'rs2'] ['1', '1']
(4, 1) ['rs3'] ['2']
>>> src.variants[2]
Variant(chrom='2', snp='rs3', cm=None, bp=50, a1=None, a2=None)

The SNPs of one chromosome, as a boolean mask over the variants:

>>> on_1 = src.variants.chrom == "1"
>>> src.variants[on_1].snp.tolist()
['rs1', 'rs2']
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pylmm.samples import SampleID

log = logging.getLogger(__name__)

# Default number of SNPs per block of `iter_blocks`.  A block is 8 * n * block_size bytes
# of float64 (10 MB for n = 1219), plus about as much again while it is decoded.
DEFAULT_BLOCK_SIZE = 1024


class Variant(NamedTuple):
    """One variant (SNP): the six columns of a PLINK ``.bim`` line.

    A field the source does not have is None (e.g. ``a1``/``a2`` for a TPED
    whose SNPs come from a ``.map`` file, all but ``snp`` for EMMA).
    """

    chrom: str | None
    snp: str
    cm: float | None
    bp: int | None
    a1: str | None
    a2: str | None


def _text_column(values: ArrayLike) -> NDArray[np.object_]:
    """`values` as an array of Python ``str`` objects (dtype object).

    An object array costs one pointer per entry plus each string's own size,
    whereas a fixed-width ``<U`` array is as wide as its longest string: one
    300-bp indel allele in a million-SNP ``.bim`` would make a 1.2 GB column.
    An object array is taken as it is (its entries are assumed to be str).
    """
    if isinstance(values, np.ndarray) and values.dtype == object:
        return values
    if isinstance(values, np.ndarray) and values.dtype.kind == "U":
        return values.astype(object)  # entries become str
    seq = list(cast(Iterable[Any], values))
    a = np.empty(len(seq), dtype=object)
    a[:] = [str(x) for x in seq]
    return a


def _bp_column(values: ArrayLike) -> NDArray[np.int64]:
    """`values` as int64 positions; ValueError for a non-integral or non-numeric value."""
    a = np.asarray(values)
    if a.dtype.kind in "iub":
        return a.astype(np.int64, copy=False)
    try:
        f = a.astype(np.float64)
    except (TypeError, ValueError):
        raise ValueError("Variants.bp: positions must be integers") from None
    bad = ~np.isfinite(f) | (f != np.floor(f))
    if bad.any():
        raise ValueError(
            "Variants.bp: positions must be integers, got %r" % (a.reshape(-1)[np.argmax(bad)],)
        )
    return f.astype(np.int64)


def _readonly(a: NDArray[Any], name: str, m: int) -> NDArray[Any]:
    """A read-only view of the 1-d array `a` of length `m` (the caller's array stays writeable)."""
    if a.ndim != 1 or len(a) != m:
        raise ValueError("Variants.%s has shape %s, expected (%d,) like snp" % (name, a.shape, m))
    a = a.view()
    a.flags.writeable = False
    return a


@dataclass(frozen=True, eq=False, kw_only=True)
class Variants:
    """Variant metadata, one entry per SNP, stored as columns (numpy arrays).

    The columns are those of a PLINK ``.bim`` file.  Only ``snp`` is required;
    any other column may be None when the source does not have it.  Columns
    are converted to read-only numpy arrays of one length: the text columns
    (``snp``, ``chrom``, ``a1``, ``a2``) to arrays of Python ``str`` (dtype
    object, so one long allele or ID does not widen the whole column),
    ``cm`` to float64 and ``bp`` to int64.

    Parameters
    ----------
    snp : array_like of str
        SNP IDs (``.bim`` column 2).
    chrom : array_like of str, optional
        Chromosome codes as written in the file (``"1"``, ``"X"``, ...).
    cm : array_like of float, optional
        Genetic position (centimorgans).
    bp : array_like of int, optional
        Base-pair position.  Floats are accepted if integral (``100.0``);
        anything else raises ValueError.
    a1, a2 : array_like of str, optional
        Alleles.  pylmm's genotype is the fraction of copies of ``a2``.

    Notes
    -----
    ``len(variants)`` is the number of SNPs.  Indexing with an int gives a
    `Variant`; indexing with a slice, a boolean mask or an index array gives
    a new `Variants` (``variants[variants.chrom == "2"]``).  A slice shares
    memory with the original, like any numpy slice.  The columns are
    read-only views, so the metadata cached by a reader cannot be changed by
    accident; copy a column (``variants.chrom.copy()``) to edit it.

    Examples
    --------
    >>> from pylmm.genotypes import Variants
    >>> v = Variants(snp=["rs1", "rs2"], chrom=["1", "2"], cm=[0.0, 1.5], bp=[100, 200],
    ...              a1=["A", "C"], a2=["G", "T"])
    >>> len(v), v[1]
    (2, Variant(chrom='2', snp='rs2', cm=1.5, bp=200, a1='C', a2='T'))
    >>> v[v.bp > 150].snp.tolist()
    ['rs2']
    >>> Variants.from_names(["SNP_1"])[0]
    Variant(chrom=None, snp='SNP_1', cm=None, bp=None, a1=None, a2=None)
    """

    snp: NDArray[np.object_]
    chrom: NDArray[np.object_] | None = None
    cm: NDArray[np.float64] | None = None
    bp: NDArray[np.int64] | None = None
    a1: NDArray[np.object_] | None = None
    a2: NDArray[np.object_] | None = None

    def __post_init__(self) -> None:
        if np.ndim(self.snp) != 1:
            raise ValueError("Variants.snp must be 1-d, got shape %s" % (np.shape(self.snp),))
        snp = _text_column(self.snp)
        m = len(snp)
        object.__setattr__(self, "snp", _readonly(snp, "snp", m))
        for name in ("chrom", "cm", "bp", "a1", "a2"):
            values = getattr(self, name)
            if values is None:
                continue
            a: NDArray[Any]
            if name == "cm":
                a = np.asarray(values, dtype=np.float64)
            elif name == "bp":
                a = _bp_column(values)
            else:
                if np.ndim(values) != 1:
                    raise ValueError(
                        "Variants.%s has shape %s, expected (%d,) like snp"
                        % (name, np.shape(values), m)
                    )
                a = _text_column(values)
            object.__setattr__(self, name, _readonly(a, name, m))

    @classmethod
    def from_names(cls, names: Sequence[str]) -> Variants:
        """Variants with SNP IDs only (every other column None)."""
        return cls(snp=names)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.snp)

    @overload
    def __getitem__(self, key: int) -> Variant: ...

    @overload
    def __getitem__(self, key: slice | Sequence[int] | NDArray[Any]) -> Variants: ...

    def __getitem__(self, key: int | slice | Sequence[int] | NDArray[Any]) -> Variant | Variants:
        if isinstance(key, (int, np.integer)):
            k = int(key)
            return Variant(
                chrom=None if self.chrom is None else str(self.chrom[k]),
                snp=str(self.snp[k]),
                cm=None if self.cm is None else float(self.cm[k]),
                bp=None if self.bp is None else int(self.bp[k]),
                a1=None if self.a1 is None else str(self.a1[k]),
                a2=None if self.a2 is None else str(self.a2[k]),
            )
        return Variants(
            snp=self.snp[key],
            chrom=None if self.chrom is None else self.chrom[key],
            cm=None if self.cm is None else self.cm[key],
            bp=None if self.bp is None else self.bp[key],
            a1=None if self.a1 is None else self.a1[key],
            a2=None if self.a2 is None else self.a2[key],
        )

    def __iter__(self) -> Iterator[Variant]:
        for k in range(len(self)):
            yield self[k]

    def __repr__(self) -> str:
        cols = [c for c in ("chrom", "snp", "cm", "bp", "a1", "a2") if getattr(self, c) is not None]
        return "Variants(%d SNPs; columns %s)" % (len(self), ", ".join(cols))


class GenotypeSource(Protocol):
    """What a genome scan needs from a genotype file or array.

    Implemented by :class:`pylmm.plink_io.PlinkReader` and `ArraySource`.

    Attributes
    ----------
    sample_ids : sequence
        The samples, in the row order of every genotype block: ``(FID, IID)``
        pairs, or positions ``0 .. n-1`` when the source has no IDs (EMMA).
    num_samples : int
        ``len(sample_ids)``, the rows of a block.
    variants : Variants
        Metadata of every SNP, in the order the blocks return them;
        ``len(variants)`` is the number of SNPs.
    norm_genotype : bool
        Whether the blocks are standardised: each SNP (column) mean-imputed,
        centred and scaled to unit variance by `normalize_genotype` (an
        all-missing SNP stays all nan, a constant one is all zeros).  If
        false, blocks hold the source's raw values, nan where missing (for
        PLINK input the fraction of A2 alleles, 0 / 0.5 / 1).
    """

    @property
    def norm_genotype(self) -> bool: ...

    @property
    def sample_ids(self) -> Sequence[SampleID] | Sequence[int]: ...

    @property
    def num_samples(self) -> int: ...

    @property
    def variants(self) -> Variants: ...

    def iter_blocks(
        self, block_size: int = DEFAULT_BLOCK_SIZE
    ) -> Iterator[tuple[NDArray[np.float64], Variants]]:
        """Yield ``(G, variants)`` for consecutive blocks of up to `block_size` SNPs.

        ``G`` is float64, shape ``(num_samples, b)`` with ``b <= block_size``
        (only the last block is shorter); ``variants`` is the `Variants` of
        its ``b`` SNPs.
        """
        ...


def check_block_size(block_size: int) -> int:
    """Return `block_size` if it is a positive int, else raise ValueError."""
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise ValueError("block_size must be a positive integer, got %r" % (block_size,))
    if block_size < 1:
        raise ValueError("block_size must be a positive integer, got %d" % block_size)
    return int(block_size)


def normalize_genotype(G: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standardise one SNP: mean-impute missing values, centre, scale to unit variance.

    This is the standardisation of ``PlinkReader(norm_genotype=True)`` and
    ``ArraySource(norm_genotype=True)`` (``PlinkReader.normalize_genotype``
    calls it).

    Parameters
    ----------
    G : numpy.ndarray
        Raw genotypes of one SNP, shape ``(n,)``, nan where missing.  It is
        not modified: the mean imputation is done on a copy.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``: ``(G - mean) / std`` using the non-missing values
        (population std).  Special cases, which callers skip: an all-missing
        SNP is returned unchanged (all nan), and a SNP that is constant up to
        rounding is returned as zeros.

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.genotypes import normalize_genotype
    >>> normalize_genotype(np.array([0.0, 1.0, np.nan, 2.0])).tolist()
    [-1.224744871391589, 0.0, 0.0, 1.224744871391589]
    """
    # The called values are extracted once (they used to be extracted four
    # times, and the nan test done twice); every result is computed with the
    # same operations as before, so it is bit for bit the same.
    x = ~np.isnan(G)
    called = G[x]
    # All missing: return the all-nan SNP unchanged (length n), which the
    # CLIs then skip.  It used to return an empty array (bug 11).
    if not called.size:
        return G
    m = called.mean()
    # A SNP that is constant up to rounding (e.g. an EMMA dosage) is
    # returned as zeros, which the CLIs skip: standardising it would blow
    # its rounding noise up to unit variance (issue 19).
    if np.ptp(called) <= 1e-12 * max(1.0, abs(m)):
        G = np.zeros_like(G)
        return G
    var = called.var()
    if var == 0:
        s = 1.0
    else:
        s = np.sqrt(var)
    if called.size < G.size:  # mean-impute the missing calls (on a copy)
        G = np.where(x, G, m)
    if s == 0:
        G = G - m
    else:
        G = (G - m) / s

    return G


# -- the same standardisation for a whole block of SNPs at once -------------------
#
# `_normalize_rows` gives every SNP of a block exactly (bit for bit) the values
# `normalize_genotype` gives it, without a Python call per SNP.  `normalize_genotype`
# is a handful of elementwise operations (correctly rounded, so the same whether done
# on one SNP or on many) and three reductions of the called values: their sum (the
# mean), their max - min (the constant test) and the sum of their squared deviations
# (the variance).  max and min are exact.  A float64 sum is not: numpy adds a 1-d array
# by pairwise summation, whose rounding depends on the length and the order of the
# terms.  The block code therefore gives each sum the same terms, in the same order,
# laid out as `normalize_genotype` lays them out (the called values of one SNP,
# contiguous in memory), and reduces them with the same ufunc along the last axis of a
# ``(k, c)`` array whose rows are contiguous: numpy then adds each row as it adds a 1-d
# array of length ``c`` (one inner-loop call per row, pairwise summation over the row).
# SNPs without a missing call are the rows of the block itself; SNPs with missing calls
# are compacted to their called values and reduced in groups of equal call count.
# (Layouts that do NOT give the same sums, measured: zeros in place of the missing
# calls; a reduction across rows, as ``G.sum(axis=0)`` of a C-contiguous ``(n, b)``
# block, which adds the rows one after the other; ``np.add.reduceat`` over the
# concatenated called values, which starts each sum from its first value.)
#
# That numpy reduces a row of a 2-d array exactly as it reduces the same values in a
# 1-d array is an implementation property, not a documented guarantee, so it is also
# checked: once per process for each row length n, `_rows_match_per_snp(n)` runs a
# small block of awkward SNPs through both paths and compares the bits; if any differs
# (a numpy or platform that sums otherwise), `_normalize_rows` standardises the SNPs
# one at a time with `normalize_genotype` instead.  tests/test_normalize_rows.py fuzzes
# the two paths against each other on every CI job (numpy 1.23.2 to current, x86-64).

# Rows per chunk of the elementwise passes of `_normalize_rows`: about 1 MiB of
# float64, so that a chunk and its scratch array stay in cache.
_CHUNK_BYTES = 1 << 20

# Row length n -> whether `_normalize_rows` matches `normalize_genotype` (the probe).
_ROWS_MATCH: dict[int, bool] = {}


def _normalize_rows(R: NDArray[np.float64]) -> None:
    """Standardise each row (one SNP) of the ``(b, n)`` float64 array `R` in place.

    Row ``j`` becomes ``normalize_genotype(R[j])``, bit for bit (see the
    comment above).  `R` may have any layout.  It is worked on where it is
    if each row is contiguous (a C-contiguous array, or rows of one
    separated by padding, as a decoded BED block); otherwise a row-major
    copy is made, and written back.
    """
    b, n = R.shape
    if b == 0 or n == 0:  # n == 0: every row is all missing, kept
        return
    ok = _ROWS_MATCH.get(n)
    if ok is None:
        ok = _ROWS_MATCH[n] = _rows_match_per_snp(n)
    if not ok:
        for j in range(b):
            R[j] = normalize_genotype(R[j])
        return
    rows_contiguous = R.strides[1] == R.itemsize and R.strides[0] >= R.itemsize * n
    X = R if rows_contiguous else np.ascontiguousarray(R)
    _normalize_c_rows(X)
    if X is not R:
        R[...] = X


def _normalize_c_rows(X: NDArray[np.float64]) -> None:
    """`_normalize_rows` for a ``(b, n)`` array whose rows are contiguous, without the probe.

    Every reduction is along the rows: of `X` itself (whose rows are
    contiguous), or of a C-contiguous array of compacted called values.
    """
    b, n = X.shape
    miss = np.isnan(X)
    cnt = n - np.count_nonzero(miss, axis=1)  # called values per row
    ncall = np.maximum(cnt, 1)  # (no division by zero for all-missing rows, unused)
    # The sums of called.mean() and called.var().  A row without missing calls is
    # reduced where it is (contiguous); its variance is done below.  The rows with
    # some, not all, calls missing are compacted: their called values, row after
    # row with the rows sorted by call count, so that each group of rows with
    # equal count c is a C-contiguous (rows, c) array, reduced along its rows.
    sums = np.add.reduce(X, axis=1)
    ssd = np.zeros(b)  # sums of squared deviations (all-missing rows: 0, unused)
    part = np.flatnonzero((cnt > 0) & (cnt < n))
    if len(part):
        order = part[np.argsort(cnt[part], kind="stable")]
        counts = cnt[order]
        bounds = np.r_[0, np.flatnonzero(np.diff(counts)) + 1, len(order)]
        called = X[order][~miss[order]]
        off = 0
        for a, z in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
            c = int(counts[a])
            C = called[off : off + (z - a) * c].reshape(z - a, c)
            off += (z - a) * c
            rows = order[a:z]
            sums[rows] = s1 = np.add.reduce(C, axis=1)
            D = C - (s1 / c)[:, None]  # called - called.mean(), as in called.var()
            ssd[rows] = np.add.reduce(np.multiply(D, D, out=D), axis=1)
        del called
    m = sums / ncall  # called.mean()
    # np.ptp(called): fmax and fmin skip the missing calls (max and min are exact).
    ptp = np.fmax.reduce(X, axis=1) - np.fmin.reduce(X, axis=1)
    am = np.abs(m)
    const = ptp <= 1e-12 * np.where(am > 1.0, am, 1.0)  # 1e-12 * max(1.0, abs(m))
    dead = np.flatnonzero((cnt == 0) | const)
    kept = X[dead]  # all-missing rows (kept) and constant ones (zeros), restored below
    kept[const[dead]] = 0.0
    # The elementwise passes, a chunk of rows at a time.
    full = cnt == n
    step = min(b, max(1, _CHUNK_BYTES // (8 * n)))
    scratch = np.empty((step, n))
    for r0 in range(0, b, step):
        r1 = min(r0 + step, b)
        X1 = X[r0:r1]
        mc = m[r0:r1, None]
        X1 -= mc  # G - m
        Q = np.multiply(X1, X1, out=scratch[: r1 - r0])  # (called - mean) ** 2, no missing call
        ssd1 = ssd[r0:r1]
        np.copyto(ssd1, np.add.reduce(Q, axis=1), where=full[r0:r1])
        if len(part):
            np.copyto(X1, mc - mc, where=miss[r0:r1])  # np.where(x, G, m) - m
        var = ssd1 / ncall[r0:r1]  # called.var()
        s = np.sqrt(var)
        s[var == 0] = 1.0
        X1 /= s[:, None]
    if len(dead):
        X[dead] = kept


def _rows_match_per_snp(n: int) -> bool:
    """Whether `_normalize_c_rows` gives `normalize_genotype`'s bits for SNPs of length `n`.

    The probe of `_normalize_rows`: a fixed (seeded) block of 16 SNPs of
    length `n`, each with values of many magnitudes (so that a sum done in
    another order would round differently), with no missing call, one, a few,
    all but one, all, and repeated call counts, plus constant and
    near-constant SNPs.  Compared as raw bits.
    """
    rng = np.random.default_rng(n)
    R = rng.uniform(0.0, 2.0, (16, n)) * 10.0 ** rng.integers(-3, 4, (16, n))
    R[1] = rng.integers(0, 3, n) / 2.0  # BED values
    R[2] = 0.3 + 1e-16 * rng.standard_normal(n)  # constant up to rounding
    R[3] = 1.0
    for j, k in ((4, 1), (5, n // 3), (6, n // 3), (7, n - 1), (8, n), (9, 2), (10, n // 2)):
        R[j, rng.choice(n, min(k, n), replace=False)] = np.nan
    R[11:13, : n // 2] = np.nan  # two rows with the same count, same positions
    X = R.copy()
    _normalize_c_rows(X)
    for j in range(len(R)):
        want = normalize_genotype(R[j].copy())
        if not np.array_equal(want.view(np.int64), X[j].view(np.int64)):
            log.debug("block standardisation differs from normalize_genotype at n=%d", n)
            return False
    return True


class ArraySource:
    """A `GenotypeSource` over an ``(n, m)`` genotype array in memory.

    Use it to feed genotypes read with any library (VCF, BGEN, VCF Zarr,
    Hail, ...) to code written for `GenotypeSource`.

    Parameters
    ----------
    G : array_like
        Genotypes, shape ``(n samples, m SNPs)``, any numeric coding (0/1/2
        counts, fractions, dosages), nan for a missing call.  Converted to
        float64 (copied only if it is not float64 already; not modified).
    sample_ids : sequence, optional
        One ID per row, e.g. ``(FID, IID)`` pairs.  Default: the positions
        ``0 .. n-1``, as for EMMA input.
    variants : Variants, optional
        One entry per column.  Default: IDs ``SNP_1 .. SNP_m`` (as EMMA input
        is named) and no other columns.
    norm_genotype : bool, default True
        Standardise each SNP of a block with `normalize_genotype`, as
        ``PlinkReader(norm_genotype=True)`` does.  If false, blocks are
        copies of the columns of `G`, unchanged.

    Raises
    ------
    ValueError
        If `G` is not 2-d, or `sample_ids` / `variants` do not match its shape.

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.genotypes import ArraySource
    >>> src = ArraySource(np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]]))
    >>> src.sample_ids, src.variants.snp.tolist()
    ([0, 1, 2], ['SNP_1', 'SNP_2'])
    >>> G, v = next(src.iter_blocks())
    >>> G[:, 0].round(3).tolist(), G[:, 1].tolist()    # SNP_2 is constant: zeros
    ([-1.225, 0.0, 1.225], [0.0, 0.0, 0.0])
    """

    def __init__(
        self,
        G: ArrayLike,
        sample_ids: Sequence[SampleID] | Sequence[int] | None = None,
        variants: Variants | None = None,
        norm_genotype: bool = True,
    ) -> None:
        A = np.asarray(G, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError("G must be (n samples, m SNPs), got shape %s" % (A.shape,))
        n, m = A.shape
        if sample_ids is None:
            sample_ids = list(range(n))
        elif len(sample_ids) != n:
            raise ValueError("%d sample_ids for %d rows of G" % (len(sample_ids), n))
        if variants is None:
            variants = Variants.from_names(["SNP_%d" % (k + 1) for k in range(m)])
        elif len(variants) != m:
            raise ValueError("%d variants for %d columns of G" % (len(variants), m))
        self._G = A
        self._sample_ids = sample_ids
        self._variants = variants
        self.norm_genotype = norm_genotype

    @property
    def sample_ids(self) -> Sequence[SampleID] | Sequence[int]:
        """The sample IDs, one per row of `G`."""
        return self._sample_ids

    @property
    def num_samples(self) -> int:
        """The number of samples (rows of `G`)."""
        return int(self._G.shape[0])

    @property
    def variants(self) -> Variants:
        """The variant metadata, one entry per column of `G`."""
        return self._variants

    def iter_blocks(
        self, block_size: int = DEFAULT_BLOCK_SIZE
    ) -> Iterator[tuple[NDArray[np.float64], Variants]]:
        """Yield ``(G, variants)`` for consecutive blocks of up to `block_size` columns.

        ``G`` is a new float64 array of shape ``(num_samples, b)``: the
        columns of the block, each standardised exactly (bit for bit) as
        `normalize_genotype` standardises it, if ``norm_genotype``.

        Raises
        ------
        ValueError
            If `block_size` is not a positive integer.
        """
        block_size = check_block_size(block_size)
        m = self._G.shape[1]
        for start in range(0, m, block_size):
            stop = min(start + block_size, m)
            block = np.array(self._G[:, start:stop], dtype=np.float64, order="F")
            if self.norm_genotype:
                _normalize_rows(block.T)  # each column as normalize_genotype(column)
            yield block, self._variants[start:stop]


if TYPE_CHECKING:
    # ArraySource implements the GenotypeSource protocol (checked by mypy).
    def _is_genotype_source(s: ArraySource) -> GenotypeSource:
        return s
