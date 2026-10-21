# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""Leave-one-chromosome-out (LOCO) kinships and scans.

A kinship built from every SNP contains the tested SNP and the SNPs near it,
so the polygenic term absorbs part of their effect ("proximal
contamination"): the test loses power and, when the phenotype is polygenic,
the statistics are deflated (λ_GC below 1, most of all in inbred panels with
few distinct genotypes; ``sims/studies/s8_loco.md``).  LOCO tests the SNPs of
chromosome ``c`` with ``K_{-c}``, the kinship of the SNPs on the *other*
chromosomes (GCTA ``--mlma-loco``, BOLT-LMM, REGENIE, GEMMA-LOCO).

`LocoKinship` builds every ``K_{-c}`` from one pass over the SNPs.  Each
kind of kinship of `pylmm.kinship` is a sum over SNPs of terms that depend on
that SNP alone (`pylmm.kinship.snp_columns`): the standardised or centred
SNP's outer product, and for EMMA's IBS kinship also the SNP's two codings
(their sum, ``a``), each divided by the number ``m`` of SNPs used.  So
``LocoKinship`` keeps, per chromosome, ``S_c = W_c W_c'`` (and ``a_c`` for
IBS) and the number of SNPs used ``m_c``, and ``K_{-c}`` is
`pylmm.kinship.finish_kinship` of ``sum_{d != c} S_d`` and ``sum_{d != c}
m_d``: the same K as `pylmm.kinship.calculate_kinship` of the other
chromosomes' SNPs, up to the order of the floating-point sums (the tests
check it within ``LOCO_KINSHIP_ATOL_SCALE`` of ``max|K|``).  The rules for which SNPs count in
``m`` (no called genotype: never; constant: not for ``"standardized"`` and
``"centered"``, yes for ``"ibs"``) are per SNP, so they carry over exactly.
``legacy`` (pylmm 1.0.0's K) is not supported.

Memory: one ``n x n`` float64 array per chromosome (plus a buffer of
``n x block_size`` SNPs and, while a ``K_{-c}`` is built, one more ``n x
n``): about 230 MB for 1219 individuals and 19 chromosomes, 4 GB for 5000
individuals and 20 chromosomes (`loco_memory_bytes`).

Chromosome labels are normalised (`normalize_chrom`): a ``chr`` prefix and
leading zeros are dropped and ``x``, ``y``, ``mt`` / ``m`` are written ``X``,
``Y``, ``MT``, so ``"chr1"``, ``"01"`` and ``1`` are one chromosome.  Every
other label is a chromosome of its own, ``"0"`` (unplaced) and ``"MT"``
included.

`gwas_loco` scans SNP matrices with these kinships, one chromosome at a
time: a null fit on ``K_{-c}`` (its own ``h``) and the SNPs of ``c`` tested
by `pylmm.lmm.gwas`.  ``pylmm gwas --loco`` does the same from files.  Its
cost is one eigendecomposition (``O(n^3)``) and null fit per chromosome:
about the cost of the all-SNP scan at ``n ~ 1200`` (19 chromosomes), about
6 times it at ``n = 3000`` (10 chromosomes).

**Reading lambda_GC under LOCO.**  With a polygenic trait, the SNPs linked to
causal variants on the tested chromosome are no longer explained by K, so
they are associated: lambda_GC of a LOCO scan rises above 1 without any
confounding (``sims/studies/s8_loco.md``: about 1.5 on a mouse panel at h2 =
0.5 with causal SNPs on every chromosome).  It is not a calibration check
there, and genomic control must not be applied to a LOCO scan (it would
remove that signal).  Check calibration on a null phenotype (permuted or
simulated), or on chromosomes without causal variants.

Examples
--------
>>> import numpy as np
>>> from pylmm.kinship import calculate_kinship
>>> from pylmm.loco import LocoKinship, gwas_loco
>>> rng = np.random.default_rng(0)
>>> W = rng.binomial(2, 0.3, size=(100, 300)).astype(float)
>>> chrom = np.repeat(["1", "2", "3"], 100)
>>> loco = LocoKinship.from_genotypes(W, chrom)
>>> loco.chromosomes
('1', '2', '3')
>>> K_not2 = loco.kinship(leave_out="2")          # the K of chromosomes 1 and 3
>>> bool(np.allclose(K_not2, calculate_kinship(W[:, chrom != "2"]), rtol=1e-12, atol=1e-14))
True
>>> y = W[:, 5] + rng.normal(size=100)
>>> res = gwas_loco(y, W, chrom, kinships=loco)
>>> len(res.p_values), sorted(res.h)
(300, ['1', '2', '3'])
>>> bool(res.p_values[5] < 1e-4)
True
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pylmm.genotypes import DEFAULT_BLOCK_SIZE, check_block_size
from pylmm.kinship import (
    KINSHIP_BLOCK_SIZE,
    FloatArray,
    KinshipKind,
    _check_kind,
    finish_kinship,
    snp_columns,
)
from pylmm.lmm import LMM, _check_phenotype, gwas

__all__ = ["LocoKinship", "LocoResult", "gwas_loco", "loco_memory_bytes", "normalize_chrom"]

_CHROM_ALIASES = {"X": "X", "Y": "Y", "XY": "XY", "MT": "MT", "M": "MT"}


def normalize_chrom(label: object) -> str:
    """A chromosome label in one spelling: ``"chr01"``, ``"01"`` and ``1`` are all ``"1"``.

    Drops a ``chr`` prefix (any case) and the leading zeros of a number, and
    writes ``x``, ``y``, ``xy``, ``mt`` and ``m`` as ``X``, ``Y``, ``XY``,
    ``MT``, ``MT``.  Anything else is kept as written (``"0"``, contig names).
    Numbers are not mapped to letters (``"23"`` stays ``"23"``: its meaning
    depends on the species).

    Examples
    --------
    >>> from pylmm.loco import normalize_chrom
    >>> [normalize_chrom(c) for c in ["chr1", "01", 1, "chrX", "mt", "0", "Un_gl000220"]]
    ['1', '1', '1', 'X', 'MT', '0', 'Un_gl000220']
    """
    s = str(label).strip()
    if len(s) > 3 and s[:3].lower() == "chr":
        s = s[3:]
    if s.isdigit():
        return str(int(s))
    return _CHROM_ALIASES.get(s.upper(), s)


def loco_memory_bytes(n: int, n_chrom: int) -> int:
    """About the peak memory of LOCO kinships for ``n`` individuals and ``n_chrom`` chromosomes.

    ``(n_chrom + 6) * n**2 * 8`` bytes: one float64 ``n x n`` sum per
    chromosome, plus about six for the ``K_{-c}`` being built, its
    eigendecomposition and the null model (measured: 591 MB at n = 1219, 19
    chromosomes; 1.16 GB at n = 3000, 10 chromosomes, peak RSS of ``pylmm
    gwas --loco`` before the all-SNP K was dropped from it).

    Examples
    --------
    >>> from pylmm.loco import loco_memory_bytes
    >>> round(loco_memory_bytes(1219, 19) / 1e6)
    297
    """
    return int((n_chrom + 6) * n * n * 8)


class LocoKinship:
    """Leave-one-chromosome-out kinships, from one pass over the SNPs.

    Feed every SNP's *raw* genotypes and its chromosome to `add` (or a block
    of them to `add_block`); then `kinship` returns ``K_{-c}`` for any
    chromosome ``c`` (or the K of all SNPs), as many times as needed.  More
    SNPs may be added after a `kinship` call.  The kinships are those of
    `pylmm.kinship.calculate_kinship` (and ``pylmm kinship``) of the same
    raw genotypes, to rounding; see `pylmm.loco` for why that holds for every
    kind, and for the memory (one ``n x n`` array per chromosome).

    Parameters
    ----------
    n : int
        The number of individuals.
    kind : {"standardized", "centered", "ibs"}, default "standardized"
        The kinship (`pylmm.kinship`).
    block_size : int, default `pylmm.kinship.KINSHIP_BLOCK_SIZE`
        SNPs buffered before their products are summed.
    scale : float, default 1.0
        ``kind="centered"`` only: multiply each centred SNP by ``scale``
        (``pylmm gwas --loco`` passes 2, as ``pylmm kinship`` does).

    Attributes
    ----------
    n : int
        The number of individuals.
    kind : str
        The kinship kind.
    n_snps : int
        SNPs added.
    n_used : int
        SNPs used (counted in ``m``), over all chromosomes.
    n_all_missing : int
        SNPs skipped because no genotype was called.

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.loco import LocoKinship
    >>> G = np.random.default_rng(1).integers(0, 2, size=(20, 60)).astype(float)
    >>> loco = LocoKinship(20, kind="ibs")
    >>> for j in range(60):
    ...     _ = loco.add(G[:, j], chrom="1" if j < 40 else "2")
    >>> loco.chromosomes, loco.snps_used(), loco.snps_used(leave_out="1")
    (('1', '2'), 60, 20)
    >>> loco.kinship(leave_out="1").shape
    (20, 20)
    """

    def __init__(
        self,
        n: int,
        kind: KinshipKind = "standardized",
        block_size: int = KINSHIP_BLOCK_SIZE,
        scale: float = 1.0,
    ) -> None:
        self.kind = _check_kind(kind)
        if scale != 1.0 and kind != "centered":
            raise ValueError("scale applies to kind='centered' only")
        self.scale = float(scale)
        self.n = int(n)
        block_size = check_block_size(block_size)
        self._width = 2 * block_size if kind == "ibs" else block_size
        self._W = np.empty((self.n, self._width))
        self._buf_chrom: list[str] = []  # the chromosome of each buffered column
        self._S: dict[str, FloatArray] = {}  # chromosome -> sum of W_c W_c'
        self._rowsum: dict[str, FloatArray] = {}  # IBS: chromosome -> sum of its columns
        self._used: dict[str, int] = {}  # chromosome -> SNPs used (insertion order kept)
        self.n_snps = 0
        self.n_used = 0
        self.n_all_missing = 0

    @classmethod
    def from_genotypes(
        cls,
        W: ArrayLike,
        chroms: Sequence[object] | NDArray[np.object_],
        kind: KinshipKind = "standardized",
        block_size: int = KINSHIP_BLOCK_SIZE,
    ) -> LocoKinship:
        """The LOCO kinships of a genotype matrix.

        Parameters
        ----------
        W : (n, m) array_like
            *Raw* genotypes, as for `pylmm.kinship.calculate_kinship`.  Not
            modified.
        chroms : sequence of length m
            The chromosome of each column (any labels; normalised by `normalize_chrom`).
        kind, block_size
            As for `LocoKinship`.

        Raises
        ------
        ValueError
            If ``W`` is not 2-d, ``chroms`` has the wrong length, or (``"ibs"``)
            a genotype is not 0, 0.5, 1 or nan (the message names the column).
        """
        A = np.asarray(W, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError("W must be 2-d (n individuals x m SNPs), got shape %s" % (A.shape,))
        labels = _labels(chroms, A.shape[1])
        loco = cls(A.shape[0], kind=kind, block_size=block_size)
        for j in range(A.shape[1]):
            try:
                loco.add(A[:, j], labels[j])
            except ValueError as e:
                raise ValueError("SNP column %d of W: %s" % (j, e)) from e
        return loco

    @property
    def chromosomes(self) -> tuple[str, ...]:
        """The chromosomes of the SNPs added, in the order first seen."""
        return tuple(self._used)

    def add(self, g: ArrayLike, chrom: object) -> bool:
        """Add one SNP: its raw genotypes (length ``n``, nan = missing) and its chromosome.

        Returns
        -------
        bool
            Whether the SNP is used (counted in ``m``).

        Raises
        ------
        ValueError
            If ``g`` has the wrong length, or for ``"ibs"`` a value other than
            0, 0.5, 1 or nan.
        """
        x = np.asarray(g, dtype=np.float64)
        if x.shape != (self.n,):
            raise ValueError(
                "expected a genotype vector of length %d, got shape %s" % (self.n, x.shape)
            )
        c = normalize_chrom(chrom)
        cols = snp_columns(x, self.kind, self.scale)
        self.n_snps += 1
        self._used.setdefault(c, 0)
        if cols is None:
            if np.isnan(x).all():
                self.n_all_missing += 1
            return False
        if self.kind == "ibs":
            r = self._rowsum.get(c)
            self._rowsum[c] = cols[0] + cols[1] if r is None else r + (cols[0] + cols[1])
        for col in cols:
            self._W[:, len(self._buf_chrom)] = col
            self._buf_chrom.append(c)
            if len(self._buf_chrom) == self._width:
                self._flush()
        self._used[c] += 1
        self.n_used += 1
        return True

    def add_block(
        self, G: ArrayLike, chroms: Sequence[object] | NDArray[np.object_]
    ) -> NDArray[np.bool_]:
        """Add a block of SNPs: ``G`` is ``(n, b)`` raw genotypes, ``chroms`` their chromosomes.

        Returns
        -------
        numpy.ndarray of bool
            Whether each SNP is used.
        """
        A = np.asarray(G, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != self.n:
            raise ValueError("expected an (%d, b) genotype block, got shape %s" % (self.n, A.shape))
        labels = _labels(chroms, A.shape[1])
        return np.array([self.add(A[:, j], labels[j]) for j in range(A.shape[1])], dtype=bool)

    def _flush(self) -> None:
        """Sum the buffered columns' products into their chromosomes' ``S_c``."""
        k = len(self._buf_chrom)
        a = 0
        while a < k:  # runs of one chromosome (usually the whole buffer)
            c = self._buf_chrom[a]
            b = a + 1
            while b < k and self._buf_chrom[b] == c:
                b += 1
            W = self._W[:, a:b]
            S = self._S.get(c)
            if S is None:
                self._S[c] = np.dot(W, W.T)
            else:
                S += np.dot(W, W.T)
            a = b
        self._buf_chrom = []

    def snps_used(self, leave_out: object = None) -> int:
        """The number of SNPs ``m`` in ``K_{-leave_out}`` (all chromosomes if None).

        Raises
        ------
        KeyError
            If ``leave_out`` is not one of `chromosomes`.
        """
        if leave_out is None:
            return self.n_used
        return self.n_used - self._used[self._chrom_key(leave_out)]

    def _chrom_key(self, chrom: object) -> str:
        c = normalize_chrom(chrom)
        if c not in self._used:
            raise KeyError(
                "no SNP on chromosome %r was added (chromosomes: %s)"
                % (c, ", ".join(self._used) or "none")
            )
        return c

    def kinship(self, leave_out: object = None) -> FloatArray:
        """``K_{-leave_out}``: the kinship of the SNPs on every other chromosome.

        Parameters
        ----------
        leave_out : chromosome label, optional
            The chromosome to leave out (normalised by `normalize_chrom`); None gives the K
            of all SNPs.

        Returns
        -------
        (n, n) float64 array
            A new array.

        Raises
        ------
        KeyError
            If ``leave_out`` is not one of `chromosomes`.
        ValueError
            If no SNP is used outside ``leave_out`` (LOCO needs SNPs on at
            least two chromosomes).
        """
        skip = None if leave_out is None else self._chrom_key(leave_out)
        if self._buf_chrom:
            self._flush()
        m = self.snps_used(leave_out)
        if m == 0:
            raise ValueError(
                "no SNP with a %s genotype outside chromosome %s; cannot compute a kinship "
                "(LOCO needs SNPs on at least two chromosomes)"
                % ("called" if self.kind == "ibs" else "non-constant", skip)
                if skip is not None
                else "no SNP was used; cannot compute a kinship"
            )
        K: FloatArray | None = None
        rowsum: FloatArray | None = None
        for c, S in self._S.items():
            if c == skip:
                continue
            if K is None:
                K = S.copy()
            else:
                K += S
            if self.kind == "ibs":
                r = self._rowsum[c]
                rowsum = r.copy() if rowsum is None else rowsum + r
        assert K is not None
        return finish_kinship(K, self.kind, m, rowsum)


def _labels(chroms: Sequence[object] | NDArray[np.object_], m: int) -> list[str]:
    """The chromosome labels, normalised (`normalize_chrom`), checked to be ``m`` of them.

    Warns when two spellings in ``chroms`` become one chromosome.
    """
    a = np.asarray(chroms, dtype=object)
    if a.ndim != 1 or len(a) != m:
        raise ValueError(
            "chroms must give one chromosome per SNP: expected %d, got shape %s" % (m, a.shape)
        )
    raw = [str(c) for c in a]
    merged = merged_chrom_labels(raw)
    if merged:
        warnings.warn(
            "chromosome labels written differently are treated as one chromosome: %s"
            % describe_merged(merged),
            stacklevel=3,
        )
    return [normalize_chrom(c) for c in raw]


def merged_chrom_labels(labels: Sequence[str]) -> dict[str, list[str]]:
    """The normalised labels that more than one spelling in ``labels`` maps to.

    Examples
    --------
    >>> from pylmm.loco import merged_chrom_labels
    >>> merged_chrom_labels(["1", "chr1", "2", "01"])
    {'1': ['01', '1', 'chr1']}
    """
    spellings: dict[str, set[str]] = {}
    for c in set(labels):
        spellings.setdefault(normalize_chrom(c), set()).add(c)
    return {k: sorted(v) for k, v in sorted(spellings.items()) if len(v) > 1}


def describe_merged(merged: dict[str, list[str]], limit: int = 5) -> str:
    """``"'1', 'chr1' -> 1; ..."`` for a warning (the first ``limit``)."""
    items = list(merged.items())
    text = "; ".join("%s -> %s" % (", ".join(repr(x) for x in v), k) for k, v in items[:limit])
    return text + ("; ..." if len(items) > limit else "")


class LocoResult(NamedTuple):
    """The result of `gwas_loco`."""

    t_stats: list[float]
    """Wald t statistic of each SNP, as `pylmm.lmm.gwas` (nan if not testable)."""
    p_values: list[float]
    """Two-sided p-value of each SNP, as `pylmm.lmm.gwas`."""
    h: dict[str, float]
    """The null-model ``h`` fitted with ``K_{-c}``, per tested chromosome ``c``."""


def gwas_loco(
    Y: ArrayLike,
    X: ArrayLike,
    chroms: Sequence[object] | NDArray[np.object_],
    kinships: LocoKinship | None = None,
    X0: ArrayLike | None = None,
    reml: bool = True,
    refit: bool = False,
    batch: bool = True,
    block_size: int = DEFAULT_BLOCK_SIZE,
    kind: KinshipKind = "standardized",
    legacy_fit: bool = False,
) -> LocoResult:
    """Scan SNPs with leave-one-chromosome-out kinships.

    For each chromosome ``c`` of ``chroms`` (in the order first seen), the
    null model is fitted with ``K_{-c}`` and the SNPs of ``c`` are tested
    with `pylmm.lmm.gwas` (batched, missing calls, ``refit``: as there).  The
    numbers are those of ``gwas(Y, X[:, on_c], K_{-c}, X0=X0, ...)`` for
    each ``c``.  Every chromosome of ``chroms`` must be one of
    ``kinships.chromosomes`` (after `normalize_chrom`): a missing one is an
    error, not a silent all-SNP K.  Under LOCO, lambda_GC of a polygenic
    trait is above 1 by design; see `pylmm.loco`.

    Parameters
    ----------
    Y : (n,) array
        Phenotype; ``nan`` = missing (those individuals are dropped).
    X : (n, m) array
        Genotypes, one column per SNP.  Without ``kinships`` they are also
        the SNPs of the kinships, so pass *raw* genotypes (as for
        `pylmm.kinship.calculate_kinship`; ``nan`` = missing call).
    chroms : sequence of length m
        The chromosome of each column of ``X``.
    kinships : LocoKinship, optional
        Prebuilt LOCO kinships of the ``n`` individuals (e.g. from other or
        more SNPs, or reused for several phenotypes).  Default: built from
        ``X`` and ``chroms`` (`LocoKinship.from_genotypes` with ``kind``).
    X0 : (n, q) array, optional
        Covariates, used as given (default: an intercept; add it yourself
        with covariates, as for `pylmm.lmm.gwas`).
    reml, refit, batch, block_size
        As for `pylmm.lmm.gwas`.
    kind : {"standardized", "centered", "ibs"}, default "standardized"
        The kinship built from ``X`` (ignored with ``kinships``).
    legacy_fit : bool, default False
        pylmm 1.0.0's variance-component fit, as for `pylmm.lmm.gwas`.

    Returns
    -------
    LocoResult
        ``(t_stats, p_values, h)``: lists of length ``m`` in the column order
        of ``X``, and the null ``h`` of each chromosome.

    Raises
    ------
    ValueError
        As `pylmm.lmm.gwas`; if ``chroms`` does not have one label per
        column; if ``kinships`` is for another number of individuals; or if
        a ``K_{-c}`` has no SNP (all SNPs on one chromosome); if a chromosome
        of ``chroms`` has no SNP in ``kinships`` (e.g. labels from another
        file: ``"chrUn"`` vs ``"Un"``).

    Examples
    --------
    See `pylmm.loco`.  The scan equals a hand-built loop:

    >>> import numpy as np
    >>> from pylmm.kinship import calculate_kinship
    >>> from pylmm.lmm import gwas
    >>> from pylmm.loco import gwas_loco
    >>> rng = np.random.default_rng(3)
    >>> W = rng.binomial(2, 0.4, size=(60, 200)).astype(float)
    >>> chrom = np.repeat(["1", "2"], 100)
    >>> y = rng.normal(size=60)
    >>> t, p, h = gwas_loco(y, W, chrom)
    >>> t1, p1 = gwas(y, W[:, :100], calculate_kinship(W[:, 100:]))
    >>> bool(np.allclose(p[:100], p1, rtol=1e-9))
    True
    """
    y_all = _check_phenotype(Y)
    n = len(y_all)
    A = np.asarray(X, dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape((n, 1))
    if A.ndim != 2 or A.shape[0] != n:
        raise ValueError("SNP matrix X has shape %s but Y has %d values" % (A.shape, n))
    m = A.shape[1]
    labels = _labels(chroms, m)
    if kinships is None:
        kinships = LocoKinship.from_genotypes(A, labels, kind=kind)
    elif kinships.n != n:
        raise ValueError("kinships are for %d individuals but Y has %d values" % (kinships.n, n))
    else:
        missing = [c for c in dict.fromkeys(labels) if c not in kinships.chromosomes]
        if missing:
            raise ValueError(
                "chromosome(s) %s of chroms have no SNP in kinships (whose chromosomes are %s): "
                "K_{-c} would be the K of all SNPs, not a LOCO kinship; check the labels"
                % (
                    ", ".join(repr(c) for c in missing[:5]) + (", ..." if len(missing) > 5 else ""),
                    ", ".join(repr(c) for c in kinships.chromosomes[:10])
                    + (", ..." if len(kinships.chromosomes) > 10 else ""),
                )
            )
    x0_all = np.ones((n, 1)) if X0 is None else np.asarray(X0, dtype=np.float64)

    t_stats: list[float] = [np.nan] * m
    p_values: list[float] = [np.nan] * m
    h: dict[str, float] = {}
    keep = ~np.isnan(y_all)
    if not keep.any():
        return LocoResult(t_stats, p_values, h)
    all_kept = bool(keep.all())
    y_kept = y_all if all_kept else y_all[keep]
    g_kept = A if all_kept else A[keep]
    x0_kept = x0_all if all_kept or x0_all.ndim != 2 or x0_all.shape[0] != n else x0_all[keep]
    lab = np.array(labels, dtype=object)
    for c in dict.fromkeys(labels):
        cols = np.flatnonzero(lab == c)
        K = kinships.kinship(c)
        if not all_kept:
            K = K[keep][:, keep]
        L = LMM(y_kept, K, X0=x0_kept, legacy_fit=legacy_fit)
        L.fit(reml=reml)
        h[c] = float(L.opt_h)
        tc, pc = gwas(
            y_kept,
            g_kept[:, cols],
            K,
            L.kva,
            L.kve,
            X0=x0_kept,
            reml=reml,
            refit=refit,
            batch=batch,
            block_size=block_size,
            legacy_fit=legacy_fit,
        )
        for k, j in enumerate(cols):
            t_stats[j], p_values[j] = tc[k], pc[k]
    return LocoResult(t_stats, p_values, h)
