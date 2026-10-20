"""Phase 6.2: variant metadata, blockwise genotype reading and the GenotypeSource interface.

The block readers must return exactly (bit for bit) what the per-SNP iterator returns:
every comparison below is ``np.array_equal(..., equal_nan=True)``, never a tolerance.
"""

import itertools
import os
import warnings

import numpy as np
import pytest

from pylmm.genotypes import (
    DEFAULT_BLOCK_SIZE,
    ArraySource,
    Variant,
    Variants,
    normalize_genotype,
)
from pylmm.plink_io import PlinkReader
from pylmm.samples import InputError
from sims.bench import bed_blocks

BED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BED = os.path.join(BED, "snps.132k.clean.noX")


def _per_snp(reader, m=None):
    """(W, names) of the first `m` SNPs from the per-SNP iterator."""
    snps = list(itertools.islice(reader.iter_snps(), m))
    return np.column_stack([g for g, _ in snps]), [name for _, name in snps]


def _blocks(reader, block_size, m=None):
    """(W, Variants) of the first `m` SNPs from iter_blocks, and the block widths."""
    blocks, vs, widths = [], [], []
    k = 0
    for G, v in reader.iter_blocks(block_size):
        assert G.dtype == np.float64 and G.shape == (len(reader.indivs), len(v))
        blocks.append(G)
        vs.append(v)
        widths.append(G.shape[1])
        k += G.shape[1]
        if m is not None and k >= m:
            break
    W = np.hstack(blocks)[:, :m]
    snp = np.concatenate([v.snp for v in vs])[:m]
    return W, snp, widths


def _write_tped(base, n=3, m=3, map_cols=4, bim=False):
    """A .tped/.tfam set with a .map of `map_cols` columns (and a .bim if `bim`)."""
    rng = np.random.default_rng(5)
    with open(base + ".tfam", "w") as f:
        for i in range(n):
            f.write("F%d I%d 0 0 1 -9\n" % (i, i))
    with open(base + ".map", "w") as f:
        for k in range(m):
            cols = ["%d" % (k % 2 + 1), "rs%d" % (k + 1), "%.1f" % (k / 2), "%d" % (100 * (k + 1))]
            if map_cols == 3:
                del cols[2]
            f.write("\t".join(cols) + "\n")
    if bim:
        with open(base + ".bim", "w") as f:
            for k in range(m):
                f.write("%d\trs%d\t0\t%d\tA\tC\n" % (k % 2 + 1, k + 1, 100 * (k + 1)))
    with open(base + ".tped", "w") as f:
        for k in range(m):
            pairs = rng.choice(["1 1", "1 2", "2 2", "0 0"], size=n, p=[0.3, 0.3, 0.3, 0.1])
            f.write("%d rs%d 0 %d %s\n" % (k % 2 + 1, k + 1, 100 * (k + 1), " ".join(pairs)))


# -- Variants ---------------------------------------------------------------


def test_variants_columns_and_indexing():
    v = Variants(
        snp=["a", "b", "c"],
        chrom=["1", "1", "X"],
        cm=[0, 0.5, 1],
        bp=[10, 20, 30],
        a1=["A", "C", "G"],
        a2=["G", "T", "A"],
    )
    assert len(v) == 3
    assert v.cm.dtype == np.float64 and v.bp.dtype == np.int64 and v.snp.dtype.kind == "U"
    assert v[2] == Variant("X", "c", 1.0, 30, "G", "A")
    assert v[-1] == v[2]
    assert isinstance(v[np.int64(0)], Variant)
    assert list(v[1:].snp) == ["b", "c"]
    assert list(v[v.chrom == "1"].bp) == [10, 20]
    assert list(v[[2, 0]].a1) == ["G", "A"]
    assert [x.snp for x in v] == ["a", "b", "c"]
    assert "3 SNPs" in repr(v)


def test_variants_missing_columns_and_validation():
    v = Variants.from_names(["SNP_1", "SNP_2"])
    assert v.chrom is None and v.cm is None and v.bp is None and v.a1 is None and v.a2 is None
    assert v[0] == Variant(None, "SNP_1", None, None, None, None)
    assert v[0:1].chrom is None
    assert len(Variants.from_names([])) == 0
    with pytest.raises(ValueError, match=r"Variants.bp has shape \(1,\), expected \(2,\)"):
        Variants(snp=["a", "b"], bp=[1])
    with pytest.raises(ValueError, match="1-d"):
        Variants(snp=[["a"]])


# -- PlinkReader.variants -------------------------------------------------------


def test_bed_variants_are_the_bim_lines():
    r = PlinkReader(BED)
    v = r.variants
    assert len(v) == 98740
    with open(BED + ".bim") as f:
        first = [line.split() for line in itertools.islice(f, 3)]
    for k, cols in enumerate(first):
        assert v[k] == Variant(cols[0], cols[1], float(cols[2]), int(cols[3]), cols[4], cols[5])
    _, names = _per_snp(PlinkReader(BED, norm_genotype=False), 300)
    assert list(v.snp[:300]) == names
    assert r.variants is v  # read once


@pytest.mark.parametrize("map_cols", [4, 3])
def test_tped_variants_from_map(tmp_path, map_cols):
    base = str(tmp_path / "t")
    _write_tped(base, map_cols=map_cols)
    v = PlinkReader(base, type="t").variants
    assert list(v.snp) == ["rs1", "rs2", "rs3"]
    assert list(v.chrom) == ["1", "2", "1"] and list(v.bp) == [100, 200, 300]
    assert v.a1 is None and v.a2 is None
    if map_cols == 4:
        assert list(v.cm) == [0.0, 0.5, 1.0]
    else:
        assert v.cm is None


def test_tped_variants_prefer_bim(tmp_path):
    base = str(tmp_path / "t")
    _write_tped(base, bim=True)
    v = PlinkReader(base, type="t").variants
    assert list(v.a1) == ["A"] * 3 and list(v.a2) == ["C"] * 3 and list(v.cm) == [0.0] * 3


def test_emma_variants_are_names_only(tmp_path):
    fn = tmp_path / "g.emma"
    fn.write_text("0 1 2\n\n1 1 0\n2 NA 1\n")
    r = PlinkReader(str(fn), type="emma", norm_genotype=False)
    v = r.variants
    assert list(v.snp) == ["SNP_1", "SNP_2", "SNP_3"]
    assert v.chrom is None and v.bp is None
    assert list(v.snp) == [name for _, name in r.iter_snps()]


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("1 rs2 0 200 C", r"line 2: expected 6 fields .* found 5"),
        ("1 rs2 0 2e2x C T", "SNP file .*bim"),
        ("", r"line 2: expected 6 fields"),
    ],
)
def test_bad_bim_raises_input_error(tmp_path, line, match):
    base = str(tmp_path / "b")
    bed_blocks.write_synthetic_bed(base, 3, 2, seed=1)
    with open(base + ".bim") as f:
        first = f.readline()
    with open(base + ".bim", "w") as f:
        f.write(first + line + "\n")
    r = PlinkReader(base)
    with pytest.raises(InputError, match=match):
        r.variants  # noqa: B018
    with pytest.raises(InputError):
        r.iter_blocks()


def test_map_with_mixed_widths_raises(tmp_path):
    base = str(tmp_path / "t")
    _write_tped(base)
    with open(base + ".map", "a") as f:
        f.write("1 rs9 900\n")
    with pytest.raises(InputError, match=r"line 4: expected 4 fields"):
        PlinkReader(base, type="t").variants  # noqa: B018


# -- iter_blocks: bit-identical to iter_snps -------------------------------------


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("block_size", [1, 97, 1024])
def test_bed_blocks_identical_on_bundled_data(norm, block_size):
    m = 1500
    W, names = _per_snp(PlinkReader(BED, norm_genotype=norm), m)
    B, snp, widths = _blocks(PlinkReader(BED, norm_genotype=norm), block_size, m)
    assert np.array_equal(W, B, equal_nan=True)
    assert list(snp) == names
    assert all(w == block_size for w in widths[:-1])


def test_bed_blocks_identical_on_bundled_subset():
    """Individuals dropped and reordered (as by a kinship with IDs): same rows, same values."""
    keep = PlinkReader(BED).indivs[::-3]
    for norm in (False, True):
        a = PlinkReader(BED, norm_genotype=norm)
        a.subset(keep)
        b = PlinkReader(BED, norm_genotype=norm)
        b.subset(keep)
        W, _ = _per_snp(a, 700)
        B, _, _ = _blocks(b, 256, 700)
        assert B.shape == (len(keep), 700)
        assert np.array_equal(W, B, equal_nan=True)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 9, 18])
@pytest.mark.parametrize("norm", [False, True])
def test_bed_blocks_identical_edge_shapes(tmp_path, n, norm):
    """N not a multiple of 4, a last partial block, block sizes 1 and > num_snps, missing calls."""
    base = str(tmp_path / "s")
    m = 13
    raw = bed_blocks.write_synthetic_bed(base, n, m, missing=0.15, seed=n)
    W, names = _per_snp(PlinkReader(base, norm_genotype=norm))
    if not norm:
        assert np.array_equal(W, raw, equal_nan=True)  # the writer and the reader agree
    for block_size in (1, 2, 4, 5, 13, 14, 1000):
        B, snp, widths = _blocks(PlinkReader(base, norm_genotype=norm), block_size)
        assert np.array_equal(W, B, equal_nan=True), block_size
        assert list(snp) == names
        assert widths == [block_size] * (m // block_size) + (
            [m % block_size] if m % block_size else []
        )


def test_bed_blocks_with_kinship_dropping_individuals(tmp_path):
    base = str(tmp_path / "s")
    bed_blocks.write_synthetic_bed(base, 11, 9, missing=0.1, seed=3)
    with open(base + ".fam") as f:
        ids = [tuple(line.split()[:2]) for line in f]
    kept = [ids[i] for i in (7, 0, 3, 10, 5)]
    np.savetxt(str(tmp_path / "K"), np.eye(len(kept)))
    with open(str(tmp_path / "K.id"), "w") as f:
        f.writelines("%s %s\n" % k for k in kept)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = PlinkReader(base, k_file=str(tmp_path / "K"))
        b = PlinkReader(base, k_file=str(tmp_path / "K"))
    assert a.sample_ids == sorted(kept, key=ids.index) and a.num_samples == 5  # .fam order
    W, _ = _per_snp(a)
    B, _, _ = _blocks(b, 4)
    assert B.shape == (5, 9) and np.array_equal(W, B, equal_nan=True)


def test_bed_all_missing_and_constant_snps(tmp_path):
    """normalize_genotype's special cases (all nan kept, constant -> zeros) survive blocks."""
    base = str(tmp_path / "s")
    with open(base + ".fam", "w") as f:
        f.writelines("F I%d 0 0 0 -9\n" % i for i in range(5))
    with open(base + ".bim", "w") as f:
        f.writelines("1 rs%d 0 %d A G\n" % (k, k) for k in range(3))
    with open(base + ".bed", "wb") as f:
        # SNP 1 all missing (code 01), SNP 2 all homozygous A2 (11), SNP 3 mixed
        f.write(bytes([0x6C, 0x1B, 0x01, 0x55, 0x01, 0xFF, 0x03, 0b11100100, 0x00]))
    W, _ = _per_snp(PlinkReader(base))
    B, _, _ = _blocks(PlinkReader(base), 3)
    assert np.isnan(B[:, 0]).all() and (B[:, 1] == 0).all()
    assert np.array_equal(W, B, equal_nan=True)


def test_truncated_bed_raises_like_iter_snps(tmp_path):
    base = str(tmp_path / "s")
    bed_blocks.write_synthetic_bed(base, 6, 5, seed=2)  # 2 bytes per SNP
    with open(base + ".bed", "rb") as f:
        data = f.read()
    with open(base + ".bed", "wb") as f:
        f.write(data[:-3])  # SNP 4 has 1 of its 2 bytes, SNP 5 none
    with pytest.raises(InputError) as per_snp:
        _per_snp(PlinkReader(base))
    with pytest.raises(InputError) as blocks:
        _blocks(PlinkReader(base), 2)
    assert str(blocks.value) == str(per_snp.value)
    assert "expected 2 bytes for SNP 4, got 1" in str(blocks.value)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "8"])
def test_bad_block_size_raises_at_call(bad):
    r = PlinkReader(BED)
    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        r.iter_blocks(bad)
    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        next(ArraySource(np.zeros((2, 2))).iter_blocks(bad))


@pytest.mark.parametrize("norm", [False, True])
def test_tped_and_emma_blocks_identical(tmp_path, norm):
    base = str(tmp_path / "t")
    _write_tped(base, n=6, m=7)
    emma = tmp_path / "g.emma"
    emma.write_text("0 1 2 NA\n1 1 0 2\n\n2 2 2 2\n0.5 1 1.5 2\n")
    for r_args in ((base, "t"), (str(emma), "emma")):
        W, names = _per_snp(PlinkReader(r_args[0], type=r_args[1], norm_genotype=norm))
        for block_size in (1, 3, 100):
            B, snp, _ = _blocks(
                PlinkReader(r_args[0], type=r_args[1], norm_genotype=norm), block_size
            )
            assert np.array_equal(W, B, equal_nan=True)
            assert list(snp) == names


def test_iter_blocks_is_independent_of_iter_snps(tmp_path):
    """Interleaving the two iterations on one reader leaves both intact."""
    base = str(tmp_path / "t")
    _write_tped(base, n=4, m=5)
    for r in (PlinkReader(base, type="t"), PlinkReader(BED)):
        snps = r.iter_snps()
        g1, n1 = next(snps)
        G, v = next(r.iter_blocks(2))
        g2, n2 = next(snps)
        assert np.array_equal(G[:, 0], g1, equal_nan=True) and v.snp[0] == n1
        assert np.array_equal(G[:, 1], g2, equal_nan=True) and v.snp[1] == n2
        r.close()


def test_default_block_size():
    G, v = next(PlinkReader(BED).iter_blocks())
    assert G.shape == (1219, DEFAULT_BLOCK_SIZE) and len(v) == DEFAULT_BLOCK_SIZE


# -- ArraySource ------------------------------------------------------------------


def test_array_source_matches_plink_reader_standardisation():
    """ArraySource(raw genotypes) standardises exactly as PlinkReader does."""
    raw, _ = _per_snp(PlinkReader(BED, norm_genotype=False), 600)
    W, _ = _per_snp(PlinkReader(BED), 600)
    before = raw.copy()
    src = ArraySource(raw)
    B = np.hstack([G for G, _ in src.iter_blocks(64)])
    assert np.array_equal(W, B, equal_nan=True)
    assert np.array_equal(raw, before, equal_nan=True)  # input not modified


def test_array_source_defaults_and_metadata():
    G = np.arange(12, dtype=float).reshape(4, 3)
    src = ArraySource(G, norm_genotype=False)
    assert src.sample_ids == [0, 1, 2, 3] and src.num_samples == 4
    assert list(src.variants.snp) == ["SNP_1", "SNP_2", "SNP_3"]
    blocks = list(src.iter_blocks(2))
    assert [b.shape for b, _ in blocks] == [(4, 2), (4, 1)]
    assert np.array_equal(np.hstack([b for b, _ in blocks]), G)
    blocks[0][0][:] = -1  # a block is a copy
    assert G[0, 0] == 0
    ids = [("F", "I%d" % i) for i in range(4)]
    v = Variants(snp=["x", "y", "z"], chrom=["2", "2", "3"])
    src = ArraySource(G.astype(int), sample_ids=ids, variants=v)
    assert src.sample_ids == ids and src.variants is v
    _, bv = next(src.iter_blocks(5))
    assert list(bv.chrom) == ["2", "2", "3"]


def test_array_source_validation():
    with pytest.raises(ValueError, match="G must be"):
        ArraySource(np.zeros(3))
    with pytest.raises(ValueError, match="2 sample_ids for 3 rows"):
        ArraySource(np.zeros((3, 2)), sample_ids=[0, 1])
    with pytest.raises(ValueError, match="1 variants for 2 columns"):
        ArraySource(np.zeros((3, 2)), variants=Variants.from_names(["a"]))


def test_genotype_sources_are_interchangeable():
    """Code written against GenotypeSource runs on both implementations."""

    def column_means(source, block_size=100):
        return np.concatenate([G.mean(axis=0) for G, _ in source.iter_blocks(block_size)])

    raw, _ = _per_snp(PlinkReader(BED, norm_genotype=False), 300)
    reader = PlinkReader(BED, norm_genotype=False)
    a = column_means(reader)[:300]
    b = column_means(ArraySource(raw, sample_ids=reader.sample_ids, norm_genotype=False))
    assert np.array_equal(a, b, equal_nan=True)


def test_plink_reader_normalize_genotype_delegates():
    g = np.array([0.0, 0.5, np.nan, 1.0, 1.0])
    assert np.array_equal(PlinkReader.normalize_genotype(None, g), normalize_genotype(g))  # type: ignore[arg-type]


# -- the benchmark ----------------------------------------------------------------


def test_bench_bed_blocks_small(tmp_path):
    base = str(tmp_path / "s")
    bed_blocks.write_synthetic_bed(base, 9, 300, seed=4)
    rows = bed_blocks.bench_file("tiny", base, 300, repeats=1)
    assert len(rows) == 2 * (1 + len(bed_blocks.BLOCK_SIZES))
    assert all(r["identical"] for r in rows)
    table = bed_blocks.format_table(rows)
    assert table.startswith("| data |") and "iter_blocks(1024)" in table


@pytest.mark.slow
def test_bench_bed_blocks_quick(capsys):
    assert bed_blocks.main(["--quick", "--repeats", "1"]) == 0
    assert "bit-identical to iter_snps: yes" in capsys.readouterr().out
