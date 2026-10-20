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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pylmm.samples import SampleID

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


def _column(values: ArrayLike | None, dtype: Any, name: str, m: int) -> NDArray[Any] | None:
    """`values` as a 1-d array of `dtype` and length `m`, or None."""
    if values is None:
        return None
    a = np.asarray(values, dtype=dtype)
    if a.ndim != 1 or len(a) != m:
        raise ValueError("Variants.%s has shape %s, expected (%d,) like snp" % (name, a.shape, m))
    return a


@dataclass(frozen=True, eq=False, kw_only=True)
class Variants:
    """Variant metadata, one entry per SNP, stored as columns (numpy arrays).

    The columns are those of a PLINK ``.bim`` file.  Only ``snp`` is required;
    any other column may be None when the source does not have it.  Columns
    are converted to numpy arrays (``str`` for the text columns, float64 for
    ``cm``, int64 for ``bp``) of one length.

    Parameters
    ----------
    snp : array_like of str
        SNP IDs (``.bim`` column 2).
    chrom : array_like of str, optional
        Chromosome codes as written in the file (``"1"``, ``"X"``, ...).
    cm : array_like of float, optional
        Genetic position (centimorgans).
    bp : array_like of int, optional
        Base-pair position.
    a1, a2 : array_like of str, optional
        Alleles.  pylmm's genotype is the fraction of copies of ``a2``.

    Notes
    -----
    ``len(variants)`` is the number of SNPs.  Indexing with an int gives a
    `Variant`; indexing with a slice, a boolean mask or an index array gives
    a new `Variants` (``variants[variants.chrom == "2"]``).  A slice shares
    memory with the original, like any numpy slice.

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

    snp: NDArray[np.str_]
    chrom: NDArray[np.str_] | None = None
    cm: NDArray[np.float64] | None = None
    bp: NDArray[np.int64] | None = None
    a1: NDArray[np.str_] | None = None
    a2: NDArray[np.str_] | None = None

    def __post_init__(self) -> None:
        snp = np.asarray(self.snp, dtype=str)
        if snp.ndim != 1:
            raise ValueError("Variants.snp must be 1-d, got shape %s" % (snp.shape,))
        m = len(snp)
        object.__setattr__(self, "snp", snp)
        for name, dtype in (
            ("chrom", str),
            ("cm", np.float64),
            ("bp", np.int64),
            ("a1", str),
            ("a2", str),
        ):
            object.__setattr__(self, name, _column(getattr(self, name), dtype, name, m))

    @classmethod
    def from_names(cls, names: Sequence[str]) -> Variants:
        """Variants with SNP IDs only (every other column None)."""
        return cls(snp=np.asarray(names, dtype=str))

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
        Metadata of every SNP, in the order the blocks return them.
    """

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
    x = ~np.isnan(G)
    # All missing: return the all-nan SNP unchanged (length n), which the
    # CLIs then skip.  It used to return an empty array (bug 11).
    if not x.any():
        return G
    m = G[x].mean()
    # A SNP that is constant up to rounding (e.g. an EMMA dosage) is
    # returned as zeros, which the CLIs skip: standardising it would blow
    # its rounding noise up to unit variance (issue 19).
    if np.ptp(G[x]) <= 1e-12 * max(1.0, abs(m)):
        G = np.zeros_like(G)
        return G
    if G[x].var() == 0:
        s = 1.0
    else:
        s = np.sqrt(G[x].var())
    G = np.where(np.isnan(G), m, G)
    if s == 0:
        G = G - m
    else:
        G = (G - m) / s

    return G


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
        columns of the block, standardised one by one with
        `normalize_genotype` if ``norm_genotype``.

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
                for j in range(stop - start):
                    block[:, j] = normalize_genotype(block[:, j])
            yield block, self._variants[start:stop]


if TYPE_CHECKING:
    # ArraySource implements the GenotypeSource protocol (checked by mypy).
    def _is_genotype_source(s: ArraySource) -> GenotypeSource:
        return s
