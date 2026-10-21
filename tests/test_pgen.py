"""PLINK 2 PGEN input (`pylmm.pgen`, ``PlinkReader(type="pgen")``, ``--pfile``).

The fixtures in tests/data/pgen/ were written by plink2 (README.md there): BED -> PGEN,
PGEN with flipped REF -> BED, and an imputed VCF (DS) -> PGEN -> BED.  The tests check
that a PGEN reads exactly as the BED plink2 converts it to or from (genotypes bit for bit,
missing calls, allele orientation, sample IDs), that dosages are those of the VCF, and
that ``pylmm gwas`` / ``pylmm kinship`` give byte-identical output from either.

The genotype tests need the optional pgenlib (``pip install 'pylmm[pgen]'``) and are
skipped without it (the lowest-direct CI job); the .psam / .pvar parsers and the
missing-pgenlib errors are tested everywhere.
"""

import contextlib
import filecmp
import importlib.util
import io
import os
import shutil
import sys

import numpy as np
import pytest

from pylmm import cli
from pylmm.genotypes import ArraySource, Variants
from pylmm.pgen import PgenSource, pgen_paths, read_psam, read_pvar, write_pgen
from pylmm.plink_io import PlinkReader
from pylmm.samples import InputError

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pgen")
HAVE_PGENLIB = importlib.util.find_spec("pgenlib") is not None
needs_pgenlib = pytest.mark.skipif(not HAVE_PGENLIB, reason="pgenlib not installed (pylmm[pgen])")

SWAPPED = [0, 3, 4, 8, 13, 21, 30, 39]  # swap.pgen: REF is hc.bed's A1 (make_inputs.SWAPPED)
SNP12 = 11  # hc.bim's A1 is PLINK 1's missing allele "0"; plink2 writes "."


def p(name):
    return os.path.join(DATA, name)


def bits(a):
    """The raw float64 bits (nan payloads and signed zeros count)."""
    return np.ascontiguousarray(a).view(np.int64)


def blocks(src, block_size=7):
    return np.hstack([G for G, _ in src.iter_blocks(block_size)])


def vcf_dosages():
    """ALT dosages (samples x variants) of dos.vcf, nan where DS is '.'."""
    rows = []
    with open(p("dos.vcf")) as f:
        for line in f:
            if not line.startswith("#"):
                ds = [x.split(":")[1] for x in line.rstrip("\n").split("\t")[9:]]
                rows.append([np.nan if d == "." else float(d) for d in ds])
    return np.array(rows).T


# -- .psam and .pvar (no pgenlib needed) -----------------------------------------


def test_read_psam_headers(tmp_path):
    fn = str(tmp_path / "x.psam")
    cases = {
        "#FID\tIID\tSEX\nf1\ti1\t1\nf1\ti2\tNA\n": [("f1", "i1"), ("f1", "i2")],
        "#IID\tSEX\ni1\t1\ni2\t2\n": [("0", "i1"), ("0", "i2")],
        "##comment\n#IID\tFID\ni1\tf9\n": [("f9", "i1")],
        "f1 i1 0 0 1 -9\nf2 i2 0 0 2 -9\n": [("f1", "i1"), ("f2", "i2")],  # .fam layout
    }
    for text, want in cases.items():
        with open(fn, "w") as f:
            f.write(text)
        assert read_psam(fn) == want
    assert read_psam(p("hc.psam")) == PlinkReader(p("hc")).indivs
    assert read_psam(p("dos.psam")) == [("0", "s%02d" % (i + 1)) for i in range(21)]


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("#FID\tSEX\nf1\t1\n", "no IID column"),
        ("#FID\tIID\nf1\n", "line 2: expected at least 2 fields"),
        ("#IID\n", "lists no samples"),
        ("", "lists no samples"),
    ],
)
def test_read_psam_errors(tmp_path, text, match):
    fn = str(tmp_path / "x.psam")
    with open(fn, "w") as f:
        f.write(text)
    with pytest.raises(InputError, match=match):
        read_psam(fn)


def test_read_pvar_matches_the_bim_plink2_writes():
    """REF is a2, ALT is a1: the .bim of `plink2 --pfile X --make-bed` (alleles, IDs, positions)."""
    for pgen, bed in (("hc", "hc"), ("swap", "swap"), ("dos", "dos")):
        v, b = read_pvar(p(pgen + ".pvar")), PlinkReader(p(bed)).variants
        for col in ("snp", "chrom", "bp", "a2"):
            assert np.array_equal(getattr(v, col), getattr(b, col)), (pgen, col)
        a1_differs = np.flatnonzero(v.a1 != b.a1).tolist()
        # hc.bed was written by hand with PLINK 1's "0" for snp12's missing allele.
        assert a1_differs == ([SNP12] if bed == "hc" else []), pgen
        assert v.a1[SNP12] == "." if pgen == "hc" else True
        if pgen == "dos":
            assert v.cm is None  # no CM column
        else:
            assert np.allclose(v.cm, b.cm)


def test_read_pvar_without_header_has_the_bim_columns(tmp_path):
    fn = str(tmp_path / "x.pvar")
    shutil.copy(p("hc.bim"), fn)
    assert np.array_equal(read_pvar(fn).a1, PlinkReader(p("hc")).variants.a1)
    assert np.array_equal(read_pvar(fn).bp, read_pvar(p("hc.pvar")).bp)


def test_read_pvar_errors(tmp_path):
    with pytest.raises(InputError, match=r"1 multiallelic variant\(s\), e\.g\. m2 \(ALT C,T\)"):
        read_pvar(p("multi.pvar"))
    fn = str(tmp_path / "x.pvar")
    for text, match in [
        ("#CHROM\tPOS\tID\tREF\n1\t1\ta\tA\n", "no ALT column"),
        ("#CHROM\tPOS\tID\tREF\tALT\n1\t1.5\ta\tA\tG\n", "position '1.5' is not an integer"),
        ("#CHROM\tPOS\tID\tREF\tALT\n1\t1\ta\tA\n", "line 2: expected at least 5 fields"),
    ]:
        with open(fn, "w") as f:
            f.write(text)
        with pytest.raises(InputError, match=match):
            read_pvar(fn)


def test_pgen_paths(tmp_path):
    assert pgen_paths(p("hc")) == (p("hc.pgen"), p("hc.pvar"), p("hc.psam"))
    assert pgen_paths(p("hczst"))[1] == p("hczst.pvar.zst")
    with pytest.raises(FileNotFoundError, match=r"nope\.pgen"):
        pgen_paths(str(tmp_path / "nope"))


def test_missing_pgenlib_says_how_to_install(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pgenlib", None)  # import pgenlib -> ImportError
    with pytest.raises(ImportError, match=r"pip install 'pylmm\[pgen\]'"):
        PgenSource(p("hc"))
    with pytest.raises(ImportError, match=r"pip install 'pylmm\[pgen\]'"):
        PlinkReader(p("hc"), type="pgen")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = cli.main(["kinship", "--pfile", p("hc"), "-o", str(tmp_path / "k")])
    assert rc == 2
    assert "pylmm: error: --pfile %s: reading PLINK 2 PGEN files needs" % p("hc") in err.getvalue()


def test_dosage_false_needs_pgen():
    with pytest.raises(ValueError, match="dosage=False applies to type='pgen' only"):
        PlinkReader(p("hc"), dosage=False)


def test_hardcalls_needs_pfile(tmp_path):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = cli.main(["kinship", "--bfile", p("hc"), "--hardcalls", "-o", str(tmp_path / "k")])
    assert rc == 2
    assert "--hardcalls applies only to --pfile" in err.getvalue()


# -- genotypes (pgenlib) ------------------------------------------------------------


@needs_pgenlib
@pytest.mark.parametrize(("pgen", "bed"), [("hc", "hc"), ("hczst", "hc"), ("swap", "swap")])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("block_size", [1, 7, 1024])
def test_hardcall_pgen_reads_as_its_bed(pgen, bed, norm, block_size):
    """Bit for bit the BED's genotypes (missing calls, all-missing and monomorphic SNPs)."""
    for dosage in (True, False):  # no dosages stored: the same values either way
        src = PgenSource(p(pgen), norm_genotype=norm, dosage=dosage)
        ref = PlinkReader(p(bed), norm_genotype=norm)
        assert src.sample_ids == ref.indivs
        assert np.array_equal(bits(blocks(src, block_size)), bits(blocks(ref, block_size)))
        assert np.array_equal(src.variants.snp, ref.variants.snp)


@needs_pgenlib
def test_allele_orientation():
    """The counted allele is REF: flipping REF (plink2 --ref-allele) flips the genotype."""
    hc = blocks(PgenSource(p("hc"), norm_genotype=False))
    sw = blocks(PgenSource(p("swap"), norm_genotype=False))
    other = np.setdiff1d(np.arange(hc.shape[1]), SWAPPED)
    assert np.array_equal(sw[:, SWAPPED], 1.0 - hc[:, SWAPPED], equal_nan=True)
    assert np.array_equal(sw[:, other], hc[:, other], equal_nan=True)
    v_hc, v_sw = read_pvar(p("hc.pvar")), read_pvar(p("swap.pvar"))
    assert np.array_equal(v_sw.a2[SWAPPED], v_hc.a1[SWAPPED])
    # pylmm's 0 / 0.5 / 1 is the fraction of copies of REF = A2: hc.bed's own coding
    raw = blocks(PlinkReader(p("hc"), norm_genotype=False))
    assert np.array_equal(hc, raw, equal_nan=True)
    assert np.isnan(hc[:, 4]).all() and (hc[:, 8] == 1.0).all()  # snp5, snp9


@needs_pgenlib
def test_dosages_are_the_vcf_ds():
    D = blocks(PgenSource(p("dos"), norm_genotype=False))
    ds = vcf_dosages()
    assert np.array_equal(np.isnan(D), np.isnan(ds))
    ok = ~np.isnan(ds)
    # stored at 1/16384 precision: at most half a step from 1 - DS/2, on the 0..1 scale
    assert np.abs(D[ok] - (1 - ds[ok] / 2)).max() <= 2.0**-16
    exact = ok & np.isin(ds, (0.0, 0.5, 1.0, 1.5, 2.0))
    assert exact.sum() >= 50
    assert np.array_equal(D[exact], 1 - ds[exact] / 2)
    assert len(np.unique(D[ok])) > 100  # real dosages, not hard calls


@needs_pgenlib
@pytest.mark.parametrize("norm", [False, True])
def test_dosage_pgen_hardcalls_read_as_plink2s_bed(norm):
    src = PgenSource(p("dos"), norm_genotype=norm, dosage=False)
    ref = PlinkReader(p("dos"), norm_genotype=norm)
    assert src.sample_ids == ref.indivs
    assert np.array_equal(bits(blocks(src)), bits(blocks(ref)))
    # plink2's hard calls: the dosages within 0.1 of 0, 1 or 2; the rest are missing
    H = blocks(PgenSource(p("dos"), norm_genotype=False, dosage=False))
    D = blocks(PgenSource(p("dos"), norm_genotype=False))
    called = ~np.isnan(H)
    assert np.abs(H[called] - D[called]).max() <= 0.05  # 0.1 on the 0..2 scale
    assert np.isnan(H[~np.isnan(D)]).sum() > 0


@needs_pgenlib
def test_subset_reorders_and_drops_samples():
    full = blocks(PgenSource(p("hc"), norm_genotype=False))
    src = PgenSource(p("hc"), norm_genotype=False)
    ids = src.sample_ids
    keep = [40, 3, 7, 0, 52, 21]
    src.subset([ids[i] for i in keep])
    assert src.sample_ids == [ids[i] for i in keep] and src.num_samples == 6
    assert np.array_equal(blocks(src), full[keep], equal_nan=True)
    src.subset([ids[0], ids[40]])  # a subset of the subset
    assert np.array_equal(blocks(src, 3), full[[0, 40]], equal_nan=True)
    everyone = PgenSource(p("hc"), norm_genotype=False)
    everyone.subset(ids[::-1])  # all samples, reversed
    assert np.array_equal(blocks(everyone), full[::-1], equal_nan=True)
    by_iid = PgenSource(p("hc"), norm_genotype=False, id_match="iid")
    by_iid.subset([("x", ids[5][1])])
    assert np.array_equal(blocks(by_iid), full[[5]], equal_nan=True)
    with pytest.raises(ValueError, match="1 of the requested samples are not in"):
        src.subset([("nope", "nope")])


@needs_pgenlib
@pytest.mark.parametrize("norm", [False, True])
def test_plink_reader_pgen_matches_bed_reader(norm):
    """PlinkReader(type="pgen"): blocks, the per-SNP iterator and a kinship-.id subset."""
    for dosage in (True, False):
        r = PlinkReader(p("swap"), type="pgen", norm_genotype=norm, dosage=dosage)
        b = PlinkReader(p("swap"), norm_genotype=norm)
        assert r.indivs == b.indivs and r.N == b.N == 53
        assert np.array_equal(bits(blocks(r)), bits(blocks(b)))
        per_snp = [(g, s) for g, s in r.iter_snps()]
        assert [s for _, s in per_snp] == b.variants.snp.tolist()
        assert np.array_equal(bits(np.column_stack([g for g, _ in per_snp])), bits(blocks(b)))
        keep = [b.indivs[i] for i in (9, 2, 30)]
        r.subset(keep)
        b.subset(keep)
        assert np.array_equal(bits(blocks(r, 5)), bits(blocks(b, 5)))
        assert np.array_equal(
            bits(np.column_stack([g for g, _ in r])), bits(np.column_stack([g for g, _ in b]))
        )
        r.close()


@needs_pgenlib
def test_pgen_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"nope\.pgen"):
        PgenSource(str(tmp_path / "nope"))
    # a .psam with one sample too few for the .pgen
    bad = str(tmp_path / "bad")
    shutil.copy(p("hc.pgen"), bad + ".pgen")
    shutil.copy(p("hc.pvar"), bad + ".pvar")
    with open(p("hc.psam")) as f, open(bad + ".psam", "w") as g:
        g.writelines(f.readlines()[:-1])
    with pytest.raises(InputError, match=r"does it have the 52 samples of the \.psam"):
        PgenSource(bad)
    # multiallelic variants: an error as soon as the variants are needed
    src = PgenSource(p("multi"))
    with pytest.raises(InputError, match="multiallelic"):
        src.iter_blocks()
    # a .pvar with fewer variants than the .pgen
    short = str(tmp_path / "short")
    for ext in (".pgen", ".psam"):
        shutil.copy(p("hc" + ext), short + ext)
    with open(p("hc.pvar")) as f, open(short + ".pvar", "w") as g:
        g.writelines(f.readlines()[:-3])
    with pytest.raises(InputError, match=r"and the 37 variants of the \.pvar"):
        list(PgenSource(short).iter_blocks())


@needs_pgenlib
@pytest.mark.parametrize("dosage", [False, True])
def test_write_pgen_round_trip(tmp_path, dosage):
    rng = np.random.default_rng(3)
    n, m = 11, 9
    if dosage:
        G = rng.integers(0, 32769, size=(n, m)) / 32768.0  # multiples of 2^-15: exact
    else:
        G = rng.integers(0, 3, size=(n, m)) / 2.0
    G[rng.random((n, m)) < 0.1] = np.nan
    ids = [("f%d" % (i // 2), "i%d" % i) for i in range(n)]
    v = Variants(
        snp=["v%d" % k for k in range(m)],
        chrom=["1"] * 5 + ["X"] * 4,
        bp=list(range(10, 10 * m + 1, 10)),
        cm=np.linspace(0, 1, m),
        a1=["A"] * m,
        a2=["C"] * m,
    )
    prefix = str(tmp_path / "w")
    write_pgen(prefix, G, ids, v, dosage=dosage)
    src = PgenSource(prefix, norm_genotype=False)
    assert src.sample_ids == ids
    assert np.array_equal(blocks(src, 4), G, equal_nan=True)
    for col in ("snp", "chrom", "bp", "a1", "a2", "cm"):
        assert np.array_equal(getattr(src.variants, col), getattr(v, col)), col
    norm = PgenSource(prefix)
    assert np.array_equal(bits(blocks(norm)), bits(blocks(ArraySource(G))))
    with pytest.raises(ValueError, match=r"hard calls must be 0, 0\.5, 1 or nan"):
        write_pgen(prefix, G + 0.1, ids, v)


@needs_pgenlib
def test_pvar_zst_parses_as_the_text_pvar():
    a, b = read_pvar(p("hczst.pvar.zst")), read_pvar(p("hc.pvar"))
    for col in ("snp", "chrom", "bp", "a1", "a2"):
        assert np.array_equal(getattr(a, col), getattr(b, col)), col
    assert a.cm is None


# -- the command line: --pfile gives the --bfile output, byte for byte ------------


def _run(argv):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    assert rc == 0, err.getvalue()
    return err.getvalue()


@pytest.fixture
def inputs(tmp_path):
    """A phenotype (some NA), a covariate, and a kinship over 50 of the 53 samples, with .id."""
    rng = np.random.default_rng(11)
    ids = PlinkReader(p("hc")).indivs
    G = blocks(PlinkReader(p("hc"), norm_genotype=False))
    y = 0.8 * np.nan_to_num(G[:, 2], nan=0.5) + rng.standard_normal(len(ids))
    pheno, covar = str(tmp_path / "y.txt"), str(tmp_path / "c.txt")
    with open(pheno, "w") as f:
        f.write("FID IID y\n")
        for i, (fid, iid) in enumerate(ids):
            f.write("%s %s %s\n" % (fid, iid, "NA" if i % 13 == 5 else "%.6f" % y[i]))
    with open(covar, "w") as f:
        for fid, iid in ids:
            f.write("%s %s %.4f\n" % (fid, iid, rng.standard_normal()))
    kin = str(tmp_path / "sub.kin")
    _run(["kinship", "--bfile", p("hc"), "-o", kin])
    K = np.loadtxt(kin)
    keep = [i for i in range(len(ids)) if i not in (1, 17, 44)]
    np.savetxt(kin, K[np.ix_(keep, keep)])
    with open(kin + ".id", "w") as f:
        f.writelines("%s %s\n" % ids[i] for i in keep)
    return {"pheno": pheno, "covar": covar, "kin": kin, "dir": tmp_path}


def _gwas(inputs, geno, name, extra=()):
    out = str(inputs["dir"] / name)
    err = _run(
        ["gwas", *geno, "--kinship", inputs["kin"], "--pheno", inputs["pheno"], "-o", out, *extra]
    )
    return out, err


@needs_pgenlib
@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("--covar", "COVAR"),
        ("--remove-missing-genotypes",),
        ("--no-batch", "--block-size", "3"),
        ("--ml", "--refit"),
    ],
)
@pytest.mark.parametrize(("pgen", "bed"), [("hc", "hc"), ("swap", "swap")])
def test_gwas_pfile_is_byte_identical_to_bfile(inputs, extra, pgen, bed):
    extra = tuple(inputs["covar"] if x == "COVAR" else x for x in extra)
    a, err_a = _gwas(inputs, ["--bfile", p(bed)], "bed.tsv", extra)
    b, err_b = _gwas(inputs, ["--pfile", p(pgen)], "pgen.tsv", extra)
    assert filecmp.cmp(a, b, shallow=False)
    assert err_a.replace(p(bed), "X") == err_b.replace(p(pgen), "X")
    with open(a) as f:
        assert len(f.readlines()) == 41


def _results(fn):
    rows = [line.rstrip("\n").split("\t") for line in open(fn)][1:]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:]] for r in rows])


@needs_pgenlib
def test_gwas_dosage_pgen(tmp_path):
    """--hardcalls gives the output of plink2's BED; the dosage scan finds the dosage effect."""
    rng = np.random.default_rng(5)
    ids = read_psam(p("dos.psam"))
    D = blocks(PgenSource(p("dos"), norm_genotype=False))
    y = 4.0 * np.nan_to_num(D[:, 2], nan=np.nanmean(D[:, 2])) + 0.3 * rng.standard_normal(21)
    pheno, kin = str(tmp_path / "y.txt"), str(tmp_path / "k")
    with open(pheno, "w") as f:
        f.writelines("%s %s %.6f\n" % (fid, iid, v) for (fid, iid), v in zip(ids, y))
    _run(["kinship", "--bfile", p("dos"), "-o", kin])
    common = ["--kinship", kin, "--pheno", pheno, "-o"]
    out = {k: str(tmp_path / k) for k in ("bed", "hc", "dos")}
    _run(["gwas", "--bfile", p("dos"), *common, out["bed"]])
    _run(["gwas", "--pfile", p("dos"), "--hardcalls", *common, out["hc"]])
    _run(["gwas", "--pfile", p("dos"), *common, out["dos"]])
    assert filecmp.cmp(out["bed"], out["hc"], shallow=False)
    snps, res_dos = _results(out["dos"])
    _, res_hc = _results(out["hc"])
    assert snps == ["v%d" % (k + 1) for k in range(12)]
    p_dos, p_hc = res_dos[:, 3], res_hc[:, 3]
    assert np.all((p_dos > 0) & (p_dos <= 1))
    assert np.argmin(p_dos) == 2 and p_dos[2] < 1e-6
    assert p_dos[2] < p_hc[2]  # the phenotype follows the dosage, not the hard call
    # BETA > 0: y rises with the REF dosage fraction pylmm counts
    assert res_dos[2, 0] > 0


@needs_pgenlib
@pytest.mark.parametrize("extra", [(), ("--kinship-type", "ibs"), ("--out-format", "grm")])
@pytest.mark.parametrize(("pgen", "bed"), [("hc", "hc"), ("hczst", "hc"), ("swap", "swap")])
def test_kinship_pfile_is_byte_identical_to_bfile(tmp_path, extra, pgen, bed):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run(["kinship", "--bfile", p(bed), "-o", a, *extra])
    _run(["kinship", "--pfile", p(pgen), "-o", b, *extra])
    suffixes = (".grm.bin", ".grm.N.bin", ".grm.id") if "grm" in extra else ("", ".id")
    for s in suffixes:
        assert filecmp.cmp(a + s, b + s, shallow=False), s


@needs_pgenlib
def test_kinship_dosage_pgen(tmp_path):
    hc, dos = str(tmp_path / "hc"), str(tmp_path / "dos")
    bed = str(tmp_path / "bed")
    _run(["kinship", "--bfile", p("dos"), "-o", bed])
    _run(["kinship", "--pfile", p("dos"), "--hardcalls", "-o", hc])
    _run(["kinship", "--pfile", p("dos"), "-o", dos])
    assert filecmp.cmp(bed, hc, shallow=False) and filecmp.cmp(bed + ".id", dos + ".id")
    k_hc, k_dos = np.loadtxt(hc), np.loadtxt(dos)
    assert not np.array_equal(k_hc, k_dos)
    # 12 variants of uniform dosages, about a third of them missing as hard calls:
    # related, not equal (0.64 measured)
    assert np.corrcoef(k_hc.ravel(), k_dos.ravel())[0, 1] > 0.5
    # IBS needs 0 / 0.5 / 1 genotypes: dosages are a usage error, hard calls work
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = cli.main(["kinship", "--pfile", p("dos"), "--kinship-type", "ibs", "-o", dos])
    assert rc == 2 and "--kinship-type ibs" in err.getvalue()
