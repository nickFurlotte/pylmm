# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""Kinship matrices from genotypes, and their scale.

Three kinds of kinship (``kind=`` of `calculate_kinship`, ``--kinship-type``
of ``pylmm kinship``):

``"standardized"`` (the default)
    The standardized relatedness matrix (GCTA-style GRM, GEMMA ``-gk 2``):
    every SNP is centred and scaled to unit variance, ``K = Z Z' / m``.
``"centered"``
    The centred relatedness matrix (GEMMA ``-gk 1``): every SNP is centred,
    not scaled, ``K = C C' / m``, so common SNPs weigh more than rare ones and
    the scale of K depends on the genotype coding.
``"ibs"``
    EMMA's identity-by-state kinship (Kang et al. 2008), the kinship of the
    EMMA-era inbred mouse panel papers: the share of alleles two individuals
    have in common, averaged over SNPs, with a diagonal of 1.

**Missing calls** (one convention for the library and the command line):

* ``"standardized"``: each SNP is standardised with the mean and the
  (population) standard deviation of its *called* genotypes, and a missing
  call is then set to 0, the mean: `pylmm.genotypes.normalize_genotype`, the
  standardisation of the genotype readers, so K is the average of the outer
  products of the same SNP vectors ``pylmm gwas`` tests.  A missing call
  contributes nothing to its individual's diagonal, so the mean diagonal of
  K is about ``1 - (the missing rate)``.
* ``"centered"``: centred on the mean of the called genotypes; a missing call
  is set to that mean (0 after centring).
* ``"ibs"``: as EMMA's ``emma.kinship(snps, method="additive", use="all")``:
  missing calls are replaced by the SNP's mean.

**Which SNPs count in m**: a SNP with no called genotype is skipped, and so is,
for ``"standardized"`` and ``"centered"``, a SNP that is constant (up to
rounding) over the individuals; ``m`` is the number of SNPs used.  (pylmm 1.0.0
skipped them too but still counted them in ``m``, which shrank K: bug 8;
``legacy=True`` / ``pylmm kinship --legacy-kinship`` reproduce it.)  For
``"ibs"`` a constant SNP counts, as in EMMA: it is identical by state for
every pair, which is part of that kinship's definition.

The **scale** of K sets the scale of the heritability ``h`` fitted with it
(the share of the variance carried by K): `kinship_scale` reports it.
Nothing is rescaled silently.

Examples
--------
>>> import numpy as np
>>> from pylmm.kinship import calculate_kinship, kinship_scale
>>> rng = np.random.default_rng(0)
>>> G = rng.integers(0, 2, size=(30, 400)).astype(float)   # 30 inbred strains, 0/1
>>> K = calculate_kinship(G)                                # standardized
>>> K_ibs = calculate_kinship(G, kind="ibs")                # EMMA's IBS
>>> round(kinship_scale(K).mean_diag, 6), round(kinship_scale(K_ibs).mean_diag, 6)
(1.0, 1.0)
>>> s = kinship_scale(K_ibs)    # IBS: off-diagonals ~0.5, so its centred scale is ~0.5
>>> bool(0.4 < s.centered_mean_diag < 0.6)
True
"""

from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pylmm.genotypes import check_block_size, normalize_genotype

FloatArray: TypeAlias = NDArray[np.float64]

KinshipKind: TypeAlias = Literal["standardized", "centered", "ibs"]
"""The kinds of kinship `calculate_kinship` computes."""

KINSHIP_KINDS: tuple[str, ...] = ("standardized", "centered", "ibs")
"""The values of ``kind`` (and of ``pylmm kinship --kinship-type``)."""

KINSHIP_BLOCK_SIZE = 1000
"""The number of SNPs `calculate_kinship` sums into K at once.

The default ``--block-size`` of ``pylmm kinship`` too, so that the library and
the command line add the same products in the same order and agree to the
last bit on the same SNPs."""


class KinshipScale(NamedTuple):
    """The scale of a kinship matrix (`kinship_scale`)."""

    mean_diag: float
    """``trace(K) / n``: the mean variance of the polygenic effect, in units
    of the genetic variance ``h * s2``."""
    centered_mean_diag: float
    """``trace(P K P) / (n - 1)`` with ``P = I - 11'/n``: the expected sample
    variance of the polygenic effect, in the same units, once the mean is
    removed (what a model with an intercept can see)."""


def kinship_scale(K: ArrayLike) -> KinshipScale:
    """The scale of a kinship matrix: its mean diagonal and its centred mean diagonal.

    With ``Y = X0 b + g + e``, ``g ~ N(0, h s2 K)``, ``e ~ N(0, (1 - h) s2 I)``,
    ``h`` is the share of the variance carried by ``K`` *on K's scale*.  The
    proportion of the phenotypic variance explained by the kinship (PVE) in
    the sample, with an intercept in ``X0``, is::

        c = centered_mean_diag
        pve = h * c / (h * c + 1 - h)

    which is ``h`` when ``c = 1``.  Kinships with the same correlation
    structure but another scale give another ``h`` and the same PVE and test
    statistics, so compare heritabilities only on one scale (the
    ``estimate-heritability`` skill).

    Parameters
    ----------
    K : (n, n) array_like
        A kinship matrix, ``n >= 2``.

    Returns
    -------
    KinshipScale
        ``(mean_diag, centered_mean_diag)``, floats.

    Raises
    ------
    ValueError
        If ``K`` is not square with at least 2 rows.

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.kinship import kinship_scale
    >>> kinship_scale(np.eye(4))
    KinshipScale(mean_diag=1.0, centered_mean_diag=1.0)
    >>> kinship_scale(np.eye(4) + 1.0)     # a constant added: only the mean diagonal moves
    KinshipScale(mean_diag=2.0, centered_mean_diag=1.0)
    """
    A = np.asarray(K, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 2:
        raise ValueError(
            "K must be a square matrix with at least 2 rows, got shape %s" % (A.shape,)
        )
    n = A.shape[0]
    mean_diag = float(np.trace(A) / n)
    # trace(P K P) = trace(K) - sum(K) / n
    centered = float((np.trace(A) - A.sum() / n) / (n - 1))
    return KinshipScale(mean_diag, centered)


def _check_kind(kind: str) -> str:
    if kind not in KINSHIP_KINDS:
        raise ValueError("kind must be one of %s, got %r" % (", ".join(KINSHIP_KINDS), kind))
    return kind


def _check_ibs_coding(g: FloatArray) -> None:
    called = g[~np.isnan(g)]
    if not np.isin(called, (0.0, 0.5, 1.0)).all():
        bad = called[~np.isin(called, (0.0, 0.5, 1.0))][0]
        raise ValueError(
            "the IBS kinship needs genotypes coded 0, 0.5 and 1 (the fraction of one allele, "
            "as EMMA and PLINK input are read), or nan; got %r (divide 0/1/2 allele counts by "
            "2; dosages are not supported)" % (float(bad),)
        )


def _emma_split(g: FloatArray) -> FloatArray | None:
    """EMMA's two 0/1 codings of one SNP, mean-imputed, as an ``(n, 2)`` array.

    ``emma.kinship(method="additive")`` replaces a heterozygote (0.5) by 1 in
    the first copy if the SNP's mean is above 0.5 (else 0), and by 1 in the
    second copy if it is below 0.5 (else 0); a missing call is then replaced
    by the mean of its copy.  None for a SNP with no called genotype.
    """
    called = ~np.isnan(g)
    if not called.any():
        return None
    mu = g[called].mean()
    het = g == 0.5
    d = np.where(het, float(mu > 0.5), g)
    r = np.where(het, float(mu < 0.5), g)
    out = np.empty((g.shape[0], 2))
    for k, x in enumerate((d, r)):
        out[:, k] = np.where(called, x, x[called].mean())
    return out


class KinshipAccumulator:
    """Build a kinship one SNP at a time (the engine of `calculate_kinship` and ``pylmm kinship``).

    Feed the *raw* genotype vector of every SNP to `add`, then call
    `result`.  The used SNPs are packed into blocks of ``block_size``
    columns, and ``W W'`` is summed over blocks, so memory is a few ``n x
    block_size`` arrays plus K, and the result depends on the block size
    only through rounding.

    Parameters
    ----------
    n : int
        The number of individuals.
    kind : {"standardized", "centered", "ibs"}, default "standardized"
        The kinship; see `pylmm.kinship`.
    block_size : int, default KINSHIP_BLOCK_SIZE
        Used SNPs per block.
    legacy : bool, default False
        ``kind="standardized"`` only: divide by the number of SNPs *added*
        rather than used, as ``pylmm kinship`` 1.0.0 did (bug 8).

    Attributes
    ----------
    n_snps : int
        SNPs added.
    n_used : int
        SNPs in K (the ``m`` of K).
    n_all_missing : int
        SNPs skipped because no genotype was called.

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.kinship import KinshipAccumulator, calculate_kinship
    >>> G = np.random.default_rng(2).integers(0, 3, size=(10, 50)) / 2.0
    >>> acc = KinshipAccumulator(10, block_size=16)
    >>> used = [acc.add(g) for g in G.T]     # True for each SNP used
    >>> bool(np.allclose(acc.result(), calculate_kinship(G)))
    True
    """

    def __init__(
        self,
        n: int,
        kind: KinshipKind = "standardized",
        block_size: int = KINSHIP_BLOCK_SIZE,
        legacy: bool = False,
    ) -> None:
        self.kind = _check_kind(kind)
        if legacy and kind != "standardized":
            raise ValueError("legacy applies to kind='standardized' only")
        self.legacy = legacy
        self.n = int(n)
        block_size = check_block_size(block_size)
        # IBS adds EMMA's two codings of each SNP: two columns per SNP.
        self._width = 2 * block_size if kind == "ibs" else block_size
        # As pylmm kinship 1.0.0 did (same buffer, same products, same order),
        # so that legacy=True reproduces its K to the last bit.
        self._W = np.ones((self.n, self._width)) * np.nan
        self._j = 0  # columns in the buffer
        self._flushed = 0  # SNPs added when the buffer was last summed into K
        self._S: FloatArray | None = None
        self._rowsum = np.zeros(self.n)  # IBS: sum over columns of the codings
        self.n_snps = 0
        self.n_used = 0
        self.n_all_missing = 0

    def _flush(self, W: FloatArray) -> None:
        ww: FloatArray = np.dot(W, W.T)
        self._S = ww if self._S is None else self._S + ww
        self._j = 0
        self._flushed = self.n_snps

    def _push(self, col: FloatArray) -> None:
        self._W[:, self._j] = col
        self._j += 1
        if self._j == self._width:
            self._flush(self._W)

    def add(self, g: ArrayLike) -> bool:
        """Add one SNP: its raw genotypes, length ``n``, nan for a missing call.

        ``"standardized"``: any additive coding.  ``"centered"``: the coding
        sets the scale of K.  ``"ibs"``: 0 / 0.5 / 1 (the fraction of one
        allele).  Not modified.

        Returns
        -------
        bool
            Whether the SNP is used (counted in ``m``).

        Raises
        ------
        ValueError
            If ``g`` has the wrong length, or for ``"ibs"`` a value other
            than 0, 0.5, 1 or nan.
        """
        x = np.asarray(g, dtype=np.float64)
        if x.shape != (self.n,):
            raise ValueError(
                "expected a genotype vector of length %d, got shape %s" % (self.n, x.shape)
            )
        self.n_snps += 1
        if self.kind == "ibs":
            _check_ibs_coding(x)
            pair = _emma_split(x)
            if pair is None:
                self.n_all_missing += 1
                return False
            self._rowsum += pair[:, 0] + pair[:, 1]
            self._push(pair[:, 0])
            self._push(pair[:, 1])
            self.n_used += 1
            return True
        if self.kind == "standardized":
            z = normalize_genotype(x)
            # As pylmm kinship 1.0.0: all-missing -> nan, constant -> zeros.
            if not np.isfinite(z).all() or z.var() == 0:
                if np.isnan(x).all():
                    self.n_all_missing += 1
                return False
            self._push(z)
            self.n_used += 1
            return True
        # centered
        called = ~np.isnan(x)
        if not called.any():
            self.n_all_missing += 1
            return False
        obs = x[called]
        mu = obs.mean()
        if np.ptp(obs) <= 1e-12 * max(1.0, abs(mu)):  # constant, as normalize_genotype
            return False
        self._push(np.where(called, x - mu, 0.0))
        self.n_used += 1
        return True

    def result(self) -> FloatArray:
        """The ``(n, n)`` kinship of the SNPs added so far.

        Raises
        ------
        ValueError
            If no SNP was added, or (except with ``legacy``) none was used.
        """
        if self.n_snps > self._flushed:  # the SNPs after the last full block
            self._flush(self._W[:, list(range(0, self._j))])
        if self.n_snps == 0:
            raise ValueError("no SNP was given; cannot compute a kinship")
        if self.legacy:
            assert self._S is not None
            legacy_k: FloatArray = self._S / float(self.n_snps)
            return legacy_k
        if self.n_used == 0:
            raise ValueError(
                "no SNP with a non-constant genotype was given; cannot compute a kinship"
                if self.kind != "ibs"
                else "no SNP with a called genotype was given; cannot compute a kinship"
            )
        assert self._S is not None
        if self.kind != "ibs":
            K: FloatArray = self._S / float(self.n_used)
            return K
        # EMMA: mean over the 0/1 codings of x_i x_j + (1 - x_i)(1 - x_j)
        #     = 1 - x_i - x_j + 2 x_i x_j, with a diagonal of 1.
        m = 2.0 * self.n_used
        a = self._rowsum
        K = (m - a[:, None] - a[None, :] + 2.0 * self._S) / m
        np.fill_diagonal(K, 1.0)
        return K


def _calculate_kinship_1_0(W: FloatArray) -> FloatArray:
    """``calculate_kinship`` of pylmm 1.0.0, unchanged (``legacy=True``).

    Mean-imputes each SNP, standardises it with the variance *after*
    imputation, and divides by all ``m`` columns, used or not (bug 8).
    """
    n, m = W.shape
    keep = []
    for i in range(m):
        obs = ~np.isnan(W[:, i])
        if not obs.any():
            continue  # all-missing SNP: skip it (bug 11)
        mn = W[obs, i].mean()
        W[np.isnan(W[:, i]), i] = mn
        vr = W[:, i].var()
        if vr == 0:
            continue

        keep.append(i)
        W[:, i] = (W[:, i] - mn) / np.sqrt(vr)

    W = W[:, keep]
    K: FloatArray = np.dot(W, W.T) * 1.0 / float(m)
    return K


def calculate_kinship(
    W: ArrayLike,
    center: bool = False,
    kind: KinshipKind = "standardized",
    legacy: bool = False,
) -> FloatArray:
    """Kinship matrix (genomic relatedness) from a SNP matrix.

    ``kind="standardized"`` (the default): each SNP column is standardised
    with the mean and standard deviation of its called genotypes, a missing
    call is set to the mean (0), and ``K = Z Z' / m`` over the ``m`` SNPs
    used.  This is the K ``pylmm kinship`` writes for the same SNPs.  The
    other kinds and the conventions are described in `pylmm.kinship`.

    Parameters
    ----------
    W : (n, m) array
        Genotypes of ``n`` individuals at ``m`` SNPs; ``nan`` marks a missing
        call.  ``"standardized"``: any additive coding (0/1/2 counts,
        dosages, the 0/0.5/1 fractions of `pylmm.plink_io.PlinkReader`); K
        does not depend on it, nor on which allele is counted.
        ``"centered"``: K is on the scale of the coding (0/1/2 counts give
        GEMMA's ``-gk 1`` matrix; 0/0.5/1 fractions a quarter of it).
        ``"ibs"``: 0 / 0.5 / 1 (EMMA's coding; inbred strains are 0 or 1).
        Not modified.
    center : bool, default False
        If True, return ``(n - 1) * K / trace(P K P)`` with
        ``P = I - 11'/n``, i.e. K rescaled so that its centred trace is
        ``n - 1`` (Gower's centring scale; `kinship_scale` then reports a
        centred mean diagonal of 1).
    kind : {"standardized", "centered", "ibs"}, default "standardized"
        ``"standardized"``: the GRM (GCTA-style; GEMMA ``-gk 2``).
        ``"centered"``: centred, not scaled (GEMMA ``-gk 1``).
        ``"ibs"``: EMMA's identity-by-state kinship, ``emma.kinship(snps,
        method="additive", use="all")`` of the EMMA R package (Kang et al.
        2008, *Genetics* 178:1709): for each SNP, coded ``x`` in {0, 0.5, 1},
        the similarity of individuals i and j is ``x_i x_j + (1 - x_i)(1 -
        x_j)``, averaged over SNPs, with the diagonal set to 1.  A
        heterozygote is split into two 0/1 codings first (as EMMA does), so
        two heterozygotes share 1, a heterozygote and a homozygote 0.5, two
        opposite homozygotes 0: the allele-sharing IBS ``1 - |x_i - x_j|``
        (except on a SNP whose mean is exactly 0.5, where EMMA codes a
        heterozygote 0 in both codings).  A missing call is replaced by the SNP's mean, an all-missing SNP is
        left out, and constant SNPs count.
    legacy : bool, default False
        Reproduce pylmm 1.0.0's ``calculate_kinship`` exactly (only with
        ``kind="standardized"``): mean-impute first, then standardise with
        the variance after imputation, and divide by all ``m`` SNPs,
        including the skipped all-missing and monomorphic ones (bug 8).
        It differs from the default only with missing calls or skipped SNPs.

    Returns
    -------
    K : (n, n) float64 array
        Symmetric, positive semi-definite kinship matrix.

    Raises
    ------
    ValueError
        If ``W`` is not 2-d, ``kind`` is unknown, ``legacy`` is used with
        another kind, no SNP can be used, or (``"ibs"``) a genotype is not
        0, 0.5, 1 or nan.

    Notes
    -----
    Rescaling K changes the estimate of ``h`` but not the test statistics,
    up to the h grid; `kinship_scale` reports the scale.  ``pylmm 1.0.0``
    imputed before standardising here but not in ``pylmm kinship``, so the
    two disagreed with missing calls (bug 8; mean diagonal 1.000 vs 0.988 on
    the bundled BED data).

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.kinship import calculate_kinship
    >>> rng = np.random.default_rng(1)
    >>> W = rng.binomial(2, 0.4, size=(20, 100)).astype(float)
    >>> W[0, 3] = np.nan                      # missing calls are mean-imputed
    >>> K = calculate_kinship(W)
    >>> K.shape
    (20, 20)
    >>> bool(np.allclose(K, K.T))
    True
    >>> round(float(np.trace(K) / 20), 2)     # about 1 - (missing rate)
    1.0
    >>> mono = np.hstack([W, np.zeros((20, 50))])           # monomorphic SNPs ...
    >>> bool(np.allclose(calculate_kinship(mono), K))       # ... are not counted
    True
    >>> G = rng.integers(0, 2, size=(20, 300)).astype(float)   # inbred: 0/1
    >>> K_ibs = calculate_kinship(G, kind="ibs")
    >>> ibs01 = 1 - np.abs(G[0] - G[1]).mean()                 # share of SNPs alike
    >>> bool(np.isclose(K_ibs[0, 1], ibs01)), float(K_ibs[0, 0])
    (True, 1.0)
    """
    _check_kind(kind)
    if legacy and kind != "standardized":
        raise ValueError("legacy=True applies to kind='standardized' only")
    # A copy: the caller's W is not imputed or standardised in place (bug 12).
    A = np.array(W, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("W must be 2-d (n individuals x m SNPs), got shape %s" % (A.shape,))
    n = A.shape[0]
    if legacy:
        K = _calculate_kinship_1_0(A)
    else:
        acc = KinshipAccumulator(n, kind=kind)
        for j in range(A.shape[1]):
            acc.add(A[:, j])
        K = acc.result()
    if center:
        P = np.diag(np.repeat(1, n)) - 1 / float(n) * np.ones((n, n))
        S = np.trace(np.dot(np.dot(P, K), P))
        k_centered: FloatArray = (n - 1) * K / S
        return k_centered
    return K
