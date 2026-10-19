"""Small numpy-only helpers for PLINK 1 files that pylmm itself does not provide.

pylmm reads PLINK BED/BIM/FAM (`pylmm.plink_io.PlinkReader`) but does not
write them, and its reader drops the ``.bim`` columns other than the SNP ID.
These helpers fill the gap for ad-hoc work:

* `read_bim` -- the ``.bim`` columns (chromosome, ID, cM, position, A1, A2),
  in the order `PlinkReader` iterates the SNPs;
* `write_bed` -- a genotype matrix (copies of A2, 0/1/2, nan = missing; or
  fractions 0/0.5/1 with ``coding="fraction"``) to ``.bed/.bim/.fam``, e.g.
  simulated or converted data;
* `subset_bed` -- copy some SNPs (e.g. one chromosome) of a BED file set to a
  new one, byte for byte, without decoding;
* `write_pheno` -- a ``FID IID name ...`` phenotype or covariate file with
  ``NA`` for missing values;
* `write_kinship` -- a kinship matrix plus its ``.id`` sidecar, the way
  ``pylmm kinship`` writes them.

Coding: pylmm reads a BED genotype as the FRACTION of copies of A2 (the 6th
``.bim`` column): BED code 00 -> 0.0, 10 -> 0.5, 11 -> 1.0, 01 -> nan.
`write_bed` takes COUNTS of A2 (0, 1, 2) by default, so ``PlinkReader(...,
norm_genotype=False)`` reads back ``G / 2``; pass ``coding="fraction"`` to
write what that reader returns.  Any other value raises ValueError (nothing
is rounded).

Use from the repository root::

    import sys
    sys.path.insert(0, "skills/convert-genotype-formats/scripts")
    import plink_tools
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

BED_MAGIC = bytes([0x6C, 0x1B, 0x01])  # PLINK 1 BED, SNP-major


def read_bim(prefix: str) -> dict[str, np.ndarray]:
    """The columns of ``prefix.bim`` as arrays.

    Returns a dict with ``chrom`` (str), ``snp`` (str), ``cm`` (float),
    ``pos`` (int), ``a1`` and ``a2`` (str), one entry per SNP in file order,
    which is the order of ``PlinkReader.iter_snps()``.
    """
    rows = []
    with open(prefix + ".bim") as f:
        for line in f:
            v = line.split()
            if v:
                rows.append(v[:6])
    a = np.array(rows, dtype=str).reshape(-1, 6)
    return {
        "chrom": a[:, 0],
        "snp": a[:, 1],
        "cm": a[:, 2].astype(float),
        "pos": a[:, 3].astype(np.int64),
        "a1": a[:, 4],
        "a2": a[:, 5],
    }


def _bytes_per_snp(n: int) -> int:
    return (n + 3) // 4


def write_bed(
    prefix: str,
    G: np.ndarray,
    iids: Sequence[str] | None = None,
    fids: Sequence[str] | None = None,
    snp_ids: Sequence[str] | None = None,
    chrom: Sequence[str] | None = None,
    pos: Sequence[int] | None = None,
    a1: Sequence[str] | None = None,
    a2: Sequence[str] | None = None,
    coding: str = "count",
) -> None:
    """Write ``prefix.bed/.bim/.fam`` from an (n, m) genotype matrix.

    Parameters
    ----------
    prefix : str
        Output prefix.
    G : (n, m) array
        Copies of allele A2 per individual and SNP: 0, 1 or 2, nan for a
        missing call (``coding="count"``, the default); or the fraction of
        A2, 0, 0.5 or 1, as ``PlinkReader(..., norm_genotype=False)`` and the
        ``bed`` source of ``sims.genotypes`` return it (``coding="fraction"``).
        Any other value raises ValueError; nothing is rounded.
    iids, fids : sequence of str, optional
        Sample IDs.  Default IIDs ``ind1 .. indn``; default FID = IID (as
        ``plink2 --double-id``).
    snp_ids, chrom, pos, a1, a2 : sequence, optional
        ``.bim`` columns.  Defaults: ``snp1 ..``, chromosome ``1``, positions
        ``1 ..``, alleles ``A``/``G`` (A1/A2).
    coding : {"count", "fraction"}, default "count"
        How ``G`` is coded (see ``G``).
    """
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError("G must be (n individuals, m SNPs), got shape %s" % (G.shape,))
    n, m = G.shape
    if coding == "fraction":
        allowed, G = (0.0, 0.5, 1.0), 2.0 * G
    elif coding == "count":
        allowed = (0.0, 1.0, 2.0)
    else:
        raise ValueError("coding must be 'count' or 'fraction', got %r" % (coding,))
    called = ~np.isnan(G)
    bad = called & ~np.isin(G, (0.0, 1.0, 2.0))
    if bad.any():
        i, j = np.argwhere(bad)[0]
        value = G[i, j] / 2 if coding == "fraction" else G[i, j]
        if coding == "count":
            hint = (
                "pylmm's PlinkReader returns fractions of A2 (0/0.5/1); pass 2*G, "
                "or coding='fraction'"
            )
        else:
            hint = "fractions must be 0, 0.5 or 1"
        raise ValueError(
            "G[%d, %d] = %r: with coding=%r, G must hold %s or nan (%d such values; nothing is "
            "rounded). %s" % (i, j, float(value), coding, allowed, int(bad.sum()), hint)
        )
    iids = ["ind%d" % (i + 1) for i in range(n)] if iids is None else [str(x) for x in iids]
    fids = list(iids) if fids is None else [str(x) for x in fids]
    snp_ids = ["snp%d" % (j + 1) for j in range(m)] if snp_ids is None else list(snp_ids)
    chrom = ["1"] * m if chrom is None else [str(c) for c in chrom]
    pos = list(range(1, m + 1)) if pos is None else [int(p) for p in pos]
    a1 = ["A"] * m if a1 is None else list(a1)
    a2 = ["G"] * m if a2 is None else list(a2)
    for name, v, size in (
        ("iids", iids, n),
        ("fids", fids, n),
        ("snp_ids", snp_ids, m),
        ("chrom", chrom, m),
        ("pos", pos, m),
        ("a1", a1, m),
        ("a2", a2, m),
    ):
        if len(v) != size:
            raise ValueError("%s has %d entries, expected %d" % (name, len(v), size))

    # 2-bit codes: 0 copies of A2 -> 00, 1 -> 10, 2 -> 11, missing -> 01
    codes = np.full((n, m), 1, dtype=np.uint8)
    lut = np.array([0, 2, 3], dtype=np.uint8)
    codes[called] = lut[np.rint(G[called]).astype(int)]
    nb = _bytes_per_snp(n)
    padded = np.zeros((nb * 4, m), dtype=np.uint8)
    padded[:n] = codes
    quads = padded.reshape(nb, 4, m)
    packed = quads[:, 0] | (quads[:, 1] << 2) | (quads[:, 2] << 4) | (quads[:, 3] << 6)
    with open(prefix + ".bed", "wb") as f:
        f.write(BED_MAGIC)
        f.write(np.ascontiguousarray(packed.T).tobytes())
    with open(prefix + ".bim", "w") as f:
        for j in range(m):
            f.write("%s\t%s\t0\t%d\t%s\t%s\n" % (chrom[j], snp_ids[j], pos[j], a1[j], a2[j]))
    with open(prefix + ".fam", "w") as f:
        for fid, iid in zip(fids, iids):
            f.write("%s %s 0 0 0 -9\n" % (fid, iid))


def subset_bed(src: str, dst: str, snps: Sequence[int] | np.ndarray) -> None:
    """Copy the SNPs ``snps`` (0-based indices or a boolean mask) of ``src`` to ``dst``.

    The ``.bed`` bytes and ``.bim`` lines are copied unchanged (no decoding),
    and the ``.fam`` file is copied whole.  The SNPs keep their file order.
    """
    with open(src + ".fam") as f:
        fam = f.read()
    n = sum(1 for line in fam.splitlines() if line.strip())
    with open(src + ".bim") as f:
        bim = [line for line in f if line.strip()]
    idx = np.arange(len(bim))[np.asarray(snps)]
    idx = np.unique(idx)
    nb = _bytes_per_snp(n)
    with open(src + ".bed", "rb") as f:
        head = f.read(3)
        if head != BED_MAGIC:
            raise ValueError("%s.bed is not a SNP-major PLINK 1 BED file" % src)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != nb * len(bim):
        raise ValueError(
            "%s.bed has %d bytes of genotypes, expected %d" % (src, data.size, nb * len(bim))
        )
    blocks = data.reshape(len(bim), nb)[idx]
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst + ".bed", "wb") as f:
        f.write(BED_MAGIC)
        f.write(blocks.tobytes())
    with open(dst + ".bim", "w") as f:
        f.writelines(bim[i] for i in idx)
    with open(dst + ".fam", "w") as f:
        f.write(fam)


def write_pheno(
    path: str,
    ids: Sequence[tuple[str, str]],
    values: np.ndarray,
    names: Sequence[str],
) -> None:
    """Write a ``FID IID name1 name2 ...`` table (tab-separated, ``NA`` = missing).

    ``ids`` are ``(FID, IID)`` pairs (e.g. ``PlinkReader.indivs``), ``values``
    is (n,) or (n, k).  pylmm reads it with ``--pheno``/``--covar`` (and
    ``--pheno-name``/``--covar-name``) or ``PlinkReader.read_phenos``.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != (len(ids), len(names)):
        raise ValueError(
            "values has shape %s, expected (%d, %d)" % (values.shape, len(ids), len(names))
        )
    with open(path, "w") as f:
        f.write("\t".join(["FID", "IID", *names]) + "\n")
        for (fid, iid), row in zip(ids, values):
            cells = ["NA" if np.isnan(x) else repr(float(x)) for x in row]
            f.write("\t".join([str(fid), str(iid), *cells]) + "\n")


def write_kinship(path: str, K: np.ndarray, ids: Sequence[tuple[str, str]]) -> None:
    """Write ``K`` as text (as ``pylmm kinship`` does) and ``path + ".id"`` with its row IDs."""
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (len(ids), len(ids)):
        raise ValueError("K is %s but there are %d IDs" % (K.shape, len(ids)))
    np.savetxt(path, K)
    with open(path + ".id", "w") as f:
        for fid, iid in ids:
            f.write("%s\t%s\n" % (fid, iid))
