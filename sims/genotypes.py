"""Genotype sources: the bundled real data and seeded synthetic generators.

Every source returns a :class:`Genotypes` (an n x m float64 matrix with nan
for missing calls, plus ids).  Encodings differ by source and do not matter
to pylmm (``calculateKinship`` standardises each SNP, and t-statistics are
scale invariant):

=================  ==========================================================
``bed``            pylmm's BED decoding: 0, 0.5, 1 (allele dosage / 2), nan
``hmdp``           0 / 1 (inbred strains), nan for NA
synthetic          0, 1, 2 copies of the alternative allele (no missing)
=================  ==========================================================

Studies normally go through :func:`get` with a JSON-able *spec* dict, e.g.
``{"source": "unrelated", "n": 500, "m": 3000, "seed": 1}``.  ``get`` caches
the result per process, so every replicate of a cell (and every cell sharing
the genotype spec) reuses the same matrix inside a worker.  The cached arrays
are read-only: copy before mutating (``Genotypes.copy``).

Synthetic sources draw from ``numpy.random.default_rng(seed)``; the same spec
yields the same matrix for a given numpy version (numpy does not promise
identical Generator streams across versions, e.g. 1.22 vs 2.x ``binomial``,
so pin via uv.lock when exact reproduction matters).
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BED_BASE = os.path.join(REPO, "data", "snps.132k.clean.noX")
HMDP_SNPS = os.path.join(REPO, "data", "hmdp.liver.snps")


@dataclasses.dataclass
class Genotypes:
    G: np.ndarray  # n x m float64, nan = missing
    snp_ids: list
    indiv_ids: list
    pops: np.ndarray | None = None  # population / family label per individual
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.G.shape[0]

    @property
    def m(self) -> int:
        return self.G.shape[1]

    def copy(self) -> Genotypes:
        return dataclasses.replace(
            self,
            G=self.G.copy(),
            pops=None if self.pops is None else self.pops.copy(),
            meta=dict(self.meta),
        )

    def subset(self, snps=None, indivs=None) -> Genotypes:
        """Column/row subset (index arrays, slices or boolean masks); copies."""
        G = self.G
        snp_ids, indiv_ids, pops = self.snp_ids, self.indiv_ids, self.pops
        if indivs is not None:
            ii = np.arange(self.n)[indivs]
            G = G[ii, :]
            indiv_ids = [indiv_ids[i] for i in ii]
            pops = None if pops is None else pops[ii]
        if snps is not None:
            jj = np.arange(self.m)[snps]
            G = G[:, jj]
            snp_ids = [snp_ids[j] for j in jj]
        return Genotypes(
            np.array(G, dtype=np.float64),
            list(snp_ids),
            list(indiv_ids),
            None if pops is None else pops.copy(),
            dict(self.meta),
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _choose(
    total: int,
    count: int | None,
    idx: Sequence[int] | None,
    select: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sorted indices of a subset of range(total)."""
    if idx is not None:
        out = np.asarray(idx, dtype=np.int64)
        if out.size and (out.min() < 0 or out.max() >= total):
            raise IndexError("index out of range for %d items" % total)
        return out
    if count is None or count >= total:
        return np.arange(total)
    if select == "first":
        return np.arange(count)
    if select == "even":
        return np.unique(np.linspace(0, total - 1, count).round().astype(np.int64))
    if select == "random":
        return np.sort(rng.choice(total, size=count, replace=False))
    raise ValueError("select must be 'first', 'even' or 'random', got %r" % select)


def sample_maf(maf: Any, m: int, rng: np.random.Generator) -> np.ndarray:
    """Draw m minor-allele frequencies.

    ``maf`` is a number (all SNPs share it), ``["uniform", lo, hi]`` or
    ``["beta", a, b]`` (folded to <= 0.5).  Lists and tuples both work so that
    specs survive a JSON round trip.
    """
    if isinstance(maf, (int, float)):
        if not 0 < maf <= 0.5:
            raise ValueError("maf must be in (0, 0.5]")
        return np.full(m, float(maf))
    kind, a, b = maf
    if kind == "uniform":
        if not 0 <= a < b <= 0.5:
            raise ValueError("uniform maf bounds must satisfy 0 <= lo < hi <= 0.5")
        return rng.uniform(a, b, size=m)
    if kind == "beta":
        p = rng.beta(a, b, size=m)
        return np.minimum(p, 1.0 - p)
    raise ValueError("unknown maf distribution %r" % (kind,))


def with_missing(G: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Copy of G with each entry independently set to nan with prob ``rate``."""
    G = np.array(G, dtype=np.float64)
    if rate > 0:
        G[rng.random(G.shape) < rate] = np.nan
    return G


# ---------------------------------------------------------------------------
# real genotypes
# ---------------------------------------------------------------------------


def bed(
    n_snps: int | None = None,
    snps: Sequence[int] | None = None,
    n_indivs: int | None = None,
    indivs: Sequence[int] | None = None,
    select: str = "first",
    seed: int = 0,
    base: str = BED_BASE,
) -> Genotypes:
    """The bundled PLINK BED data (1219 mice x 98,740 SNPs), read with
    ``pylmm.input.plink`` (unnormalised: 0/0.5/1/nan).

    Subset SNPs with ``n_snps`` (+ ``select`` = first/even/random) or explicit
    ``snps`` indices, and individuals likewise with ``n_indivs``/``indivs``.
    ``seed`` only matters for ``select='random'``.  All 98,740 SNPs for all
    individuals is ~960 MB as float64, so subset in studies.

    Note: the full kinship of this data has effective rank ~110 (replicate
    inbred animals), see docs/phase1-baseline.md.
    """
    import contextlib
    import io

    from pylmm import plink_io

    rng = np.random.default_rng(seed)
    with contextlib.redirect_stderr(io.StringIO()):  # "Read N individuals"
        reader = plink_io.PlinkReader(base, type="b", norm_genotype=False)
    n_total = len(reader.indivs)
    with open(base + ".bim") as f:
        m_total = sum(1 for _ in f)
    jj = _choose(m_total, n_snps, snps, select, rng)
    ii = _choose(n_total, n_indivs, indivs, select, rng)
    want = np.zeros(m_total, dtype=bool)
    want[jj] = True
    cols, ids = {}, {}
    reader.get_snp_iterator()
    for j in range(m_total):
        g, sid = next(reader)
        if want[j]:
            cols[j] = g[ii]
            ids[j] = sid
        if j >= jj.max():
            break
    order = list(jj)
    G = np.column_stack([cols[j] for j in order]) if order else np.zeros((len(ii), 0))
    indiv_ids = ["%s:%s" % reader.indivs[i] for i in ii]
    return Genotypes(
        G.astype(np.float64),
        [ids[j] for j in order],
        indiv_ids,
        meta={"source": "bed", "snp_index": jj, "indiv_index": ii},
    )


def hmdp(
    n_snps: int | None = None,
    snps: Sequence[int] | None = None,
    n_indivs: int | None = None,
    indivs: Sequence[int] | None = None,
    select: str = "first",
    seed: int = 0,
    path: str = HMDP_SNPS,
) -> Genotypes:
    """The HMDP liver SNP text file (EMMA layout: one SNP per row, 0/1/NA).

    ~132k SNPs x 94 inbred strains, ~2% missing calls.  Subsetting as in
    :func:`bed`.
    """
    rng = np.random.default_rng(seed)
    with open(path) as f:
        lines = f.readlines()
    m_total = len(lines)
    n_total = len(lines[0].split())
    jj = _choose(m_total, n_snps, snps, select, rng)
    ii = _choose(n_total, n_indivs, indivs, select, rng)
    rows = []
    for j in jj:
        rows.append([np.nan if x == "NA" else float(x) for x in lines[j].split()])
    G = np.array(rows, dtype=np.float64).reshape(len(jj), n_total).T[ii, :].copy()
    return Genotypes(
        G,
        ["hmdp_%d" % j for j in jj],
        ["strain_%d" % i for i in ii],
        meta={"source": "hmdp", "snp_index": jj, "indiv_index": ii},
    )


# ---------------------------------------------------------------------------
# synthetic genotypes
# ---------------------------------------------------------------------------


def unrelated(n: int, m: int, maf: Any = ("uniform", 0.05, 0.5), seed: int = 0) -> Genotypes:
    """n unrelated individuals, m independent SNPs in Hardy-Weinberg
    equilibrium: g_ij ~ Binomial(2, p_j), p_j drawn from ``maf``."""
    rng = np.random.default_rng(seed)
    p = sample_maf(maf, m, rng)
    G = rng.binomial(2, p, size=(n, m)).astype(np.float64)
    return Genotypes(
        G,
        ["snp_%d" % j for j in range(m)],
        ["ind_%d" % i for i in range(n)],
        meta={"source": "unrelated", "maf": p},
    )


def balding_nichols(
    n: int,
    m: int,
    k: int = 2,
    fst: float = 0.1,
    maf: Any = ("uniform", 0.05, 0.5),
    admixture: float | None = None,
    seed: int = 0,
) -> Genotypes:
    """Structured population under the Balding-Nichols model.

    Ancestral frequencies p_j come from ``maf``; population frequencies are
    p_kj ~ Beta(p_j (1-F)/F, (1-p_j)(1-F)/F) with F = ``fst``.

    * ``admixture=None``: individuals split evenly into the k populations
      (``pops`` holds the label).
    * ``admixture=a``: each individual has ancestry proportions
      q_i ~ Dirichlet(a * 1_k) and genotype frequency sum_k q_ik p_kj
      (``pops`` holds the majority ancestry; ``meta['Q']`` the proportions).
    """
    if not 0 < fst < 1:
        raise ValueError("fst must be in (0, 1)")
    rng = np.random.default_rng(seed)
    p = sample_maf(maf, m, rng)
    c = (1.0 - fst) / fst
    P = rng.beta(p * c, (1.0 - p) * c, size=(k, m))  # k x m
    if admixture is None:
        pops = np.repeat(np.arange(k), int(np.ceil(n / k)))[:n]
        F = P[pops, :]
        Q = None
    else:
        Q = rng.dirichlet(np.full(k, float(admixture)), size=n)  # n x k
        F = Q @ P
        pops = Q.argmax(axis=1)
    G = rng.binomial(2, F).astype(np.float64)
    meta = {"source": "balding_nichols", "maf": p, "pop_freqs": P, "fst": fst}
    if Q is not None:
        meta["Q"] = Q
    return Genotypes(
        G, ["snp_%d" % j for j in range(m)], ["ind_%d" % i for i in range(n)], pops=pops, meta=meta
    )


def family(
    n_families: int,
    n_sibs: int,
    m: int,
    include_parents: bool = False,
    maf: Any = ("uniform", 0.05, 0.5),
    seed: int = 0,
) -> Genotypes:
    """Nuclear families: two unrelated HWE founders per family, ``n_sibs``
    full sibs each receiving one random allele from each parent (unlinked
    SNPs).  Expected relatedness is 0.5 between sibs (and parent-child),
    0 between families.  ``include_parents`` adds the founders as rows.
    ``pops`` holds the family index; ``meta['role']`` is 'parent'/'child'.
    """
    rng = np.random.default_rng(seed)
    p = sample_maf(maf, m, rng)
    # founder haplotypes: n_families x 2 parents x 2 haplotypes x m
    H = (rng.random((n_families, 2, 2, m)) < p).astype(np.int8)
    # each child picks one haplotype of each parent per SNP
    pick = rng.integers(0, 2, size=(n_families, n_sibs, 2, m))
    fam_idx = np.arange(n_families)[:, None, None]
    snp_idx = np.arange(m)[None, None, :]
    from_mother = H[fam_idx, 0, pick[:, :, 0, :], snp_idx]
    from_father = H[fam_idx, 1, pick[:, :, 1, :], snp_idx]
    kids = (from_mother + from_father).reshape(n_families * n_sibs, m)
    rows = [kids]
    fam = [np.repeat(np.arange(n_families), n_sibs)]
    role = ["child"] * (n_families * n_sibs)
    if include_parents:
        rows.insert(0, H.sum(axis=2).reshape(n_families * 2, m))
        fam.insert(0, np.repeat(np.arange(n_families), 2))
        role = ["parent"] * (n_families * 2) + role
    G = np.vstack(rows).astype(np.float64)
    pops = np.concatenate(fam)
    return Genotypes(
        G,
        ["snp_%d" % j for j in range(m)],
        ["ind_%d" % i for i in range(G.shape[0])],
        pops=pops,
        meta={"source": "family", "maf": p, "role": role},
    )


SOURCES = {
    "bed": bed,
    "hmdp": hmdp,
    "unrelated": unrelated,
    "balding_nichols": balding_nichols,
    "family": family,
}


def make(spec: dict) -> Genotypes:
    """Build a Genotypes from a spec dict: ``{"source": name, **kwargs}``."""
    spec = dict(spec)
    source = spec.pop("source")
    if source not in SOURCES:
        raise ValueError("unknown genotype source %r (have %s)" % (source, sorted(SOURCES)))
    return SOURCES[source](**spec)


def spec_key(spec: Any) -> str:
    """Canonical JSON for a spec (sorted keys; tuples become lists)."""
    return json.dumps(spec, sort_keys=True, default=_json_default)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("not JSON serialisable: %r" % (o,))


_CACHE: dict = {}
_CACHE_MAX = 8


def get(spec: dict) -> Genotypes:
    """Cached :func:`make`.  Arrays in the result are read-only."""
    key = spec_key(spec)
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        g = make(spec)
        g.G.setflags(write=False)
        _CACHE[key] = g
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()
