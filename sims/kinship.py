"""Kinship matrices the way pylmm builds them, plus cached eigendecompositions.

``kinship(G)`` is ``pylmm.lmm.calculateKinship`` on a *copy* of G (the pylmm
function mean-imputes and standardises its argument in place).  Note its
known quirks, which simulations inherit on purpose because they are what
pylmm users get (docs/phase1-baseline.md, bug 8): monomorphic SNPs are
dropped but still counted in the 1/m scale factor, and missing calls are
mean-imputed before the variance is taken.

``get(geno_spec, snps=...)`` caches (K, Kva, Kve) per process, keyed by the
genotype spec and the SNP subset, so replicates reuse one O(n^3) eigh.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
from scipy import linalg

from sims import genotypes as _geno


@dataclasses.dataclass
class Kinship:
    K: np.ndarray  # n x n, read-only
    Kva: np.ndarray  # eigenvalues (ascending), read-only, NOT clamped
    Kve: np.ndarray  # eigenvectors (columns), read-only
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.K.shape[0]

    def eig_for_pylmm(self):
        """(Kva, Kve) safe to hand to ``pylmm.lmm.LMM``.

        LMM clamps eigenvalues < 1e-6 *in place* in the array it is given
        (docs/phase1-baseline.md bug 12), which would corrupt the cache (and
        fails on the read-only cached array), so Kva is copied.  Kve is only
        read by pylmm and is passed as is.
        """
        return self.Kva.copy(), self.Kve

    def subset(self, keep: np.ndarray) -> Kinship:
        """Kinship for a subset of individuals (recomputes the eigh)."""
        return from_matrix(self.K[np.ix_(keep, keep)], **self.meta)


def kinship(G: np.ndarray, center: bool = False) -> np.ndarray:
    """``pylmm.lmm.calculateKinship`` on a copy of G (G is left untouched)."""
    from pylmm.lmm import calculateKinship

    return calculateKinship(np.array(G, dtype=np.float64), center=center)


def from_matrix(K: np.ndarray, **meta) -> Kinship:
    """Wrap an existing kinship matrix (symmetrised) with its eigh."""
    K = np.array(K, dtype=np.float64)
    K = 0.5 * (K + K.T)
    Kva, Kve = linalg.eigh(K)
    for a in (K, Kva, Kve):
        a.setflags(write=False)
    return Kinship(K, Kva, Kve, dict(meta))


def _snps_key(snps) -> str:
    if snps is None:
        return "all"
    if isinstance(snps, slice):
        return "slice:%r:%r:%r" % (snps.start, snps.stop, snps.step)
    a = np.ascontiguousarray(np.asarray(snps))
    return "idx:%s:%s" % (a.dtype, hashlib.sha1(a.tobytes()).hexdigest())


_CACHE: dict = {}
_CACHE_MAX = 4


def get(geno_spec: dict, snps=None, center: bool = False) -> Kinship:
    """Cached kinship + eigh for ``genotypes.get(geno_spec).G[:, snps]``.

    ``snps`` (slice, index array, boolean mask or None = all SNPs) selects
    the kinship SNP set, so a study can use disjoint SNP sets for K and for
    the tested SNPs.  Up to 4 kinships are kept per process (an n=1219
    kinship plus its eigenvectors is ~24 MB).
    """
    key = (_geno.spec_key(geno_spec), _snps_key(snps), bool(center))
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        G = _geno.get(geno_spec).G
        if snps is not None:
            G = G[:, snps]
        _CACHE[key] = from_matrix(kinship(G, center=center), geno_spec=geno_spec, n_snps=G.shape[1])
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()


def split_snps(
    m: int,
    n_kinship: int | None,
    n_test: int | None,
    rng: np.random.Generator | None = None,
    overlap: bool = False,
):
    """Split range(m) into (kinship_idx, test_idx).

    ``overlap=False`` makes the sets disjoint (kinship SNPs first, then test
    SNPs, in random order if ``rng`` is given, else contiguous); with
    ``overlap=True`` the test SNPs are drawn from the kinship SNPs (the usual
    "K from all SNPs" setting, with proximal contamination).  ``None`` for a
    count means "all remaining".
    """
    order = rng.permutation(m) if rng is not None else np.arange(m)
    n_kinship = m if n_kinship is None else n_kinship
    kin = np.sort(order[:n_kinship])
    if overlap:
        pool = order[:n_kinship]
    else:
        pool = order[n_kinship:]
    n_test = len(pool) if n_test is None else n_test
    if n_test > len(pool):
        raise ValueError("not enough SNPs: want %d test SNPs, have %d" % (n_test, len(pool)))
    return kin, np.sort(pool[:n_test])
