"""Leave-one-chromosome-out: pylmm.loco (LocoKinship, gwas_loco) and `pylmm gwas --loco`."""

import contextlib
import io
import os
import re

import helpers as h
import numpy as np
import pytest
from synthetic_bed import write_synthetic_bed

from pylmm import cli
from pylmm.kinship import calculate_kinship
from pylmm.lmm import gwas
from pylmm.loco import LocoKinship, gwas_loco
from pylmm.plink_io import PlinkReader

KINDS = ("standardized", "centered", "ibs")


def _err(argv, rc=0):
    """Run `pylmm <argv>` in-process; check the exit code; return its stderr."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        got = cli.main(argv)
    assert got == rc, err.getvalue()
    return err.getvalue()


def _out(fn):
    rows = [line.rstrip("\n").split("\t") for line in open(fn)][1:]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:]] for r in rows])


def _assert_k_close(K, k_ref, what):
    np.testing.assert_allclose(
        K, k_ref, rtol=0, atol=h.LOCO_KINSHIP_ATOL_SCALE * np.abs(k_ref).max(), err_msg=what
    )


def _genotypes(seed, n=60, m=400, inbred=False, missing=0.03):
    rng = np.random.default_rng(seed)
    if inbred:
        G = rng.integers(0, 2, size=(n, m)).astype(float)
    else:
        G = rng.binomial(2, rng.uniform(0.05, 0.5, m), size=(n, m)) / 2.0
    G[rng.random((n, m)) < missing] = np.nan
    G[:, 7] = 0.5  # constant: not in m, except for IBS
    G[:, 11] = np.nan  # no called genotype: never in m
    return G


# ---------------------------------------------------------------------------
# LocoKinship
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("interleaved", [False, True])
def test_loco_kinship_equals_direct_kinship(kind, interleaved):
    """K_{-c} == calculate_kinship of the other chromosomes' SNPs, for every c and kind."""
    G = _genotypes(1, inbred=kind == "ibs")
    m = G.shape[1]
    chrom = np.repeat(["1", "2", "X", "MT"], [150, 120, 100, 30])
    if interleaved:
        chrom = np.random.default_rng(2).permutation(chrom)
    loco = LocoKinship.from_genotypes(G, chrom, kind=kind, block_size=37)
    assert set(loco.chromosomes) == {"1", "2", "X", "MT"}
    assert loco.n_snps == m and loco.n_all_missing == 1
    assert loco.n_used == (m - 1 if kind == "ibs" else m - 2)
    for c in [None, *loco.chromosomes]:
        sel = np.ones(m, bool) if c is None else chrom != c
        k_ref = calculate_kinship(G[:, sel], kind=kind)
        _assert_k_close(loco.kinship(c), k_ref, "%s, leave out %s" % (kind, c))
        expected_m = int(sel.sum()) - int(sel[11]) - (0 if kind == "ibs" else int(sel[7]))
        assert loco.snps_used(c) == expected_m


def test_loco_kinship_add_block_and_streaming_equal():
    """add / add_block in any block size give the same K (up to rounding); kinship() mid-stream."""
    G = _genotypes(3)
    chrom = np.repeat([1, 2, 3], [100, 150, 150])  # int labels are compared as str
    a = LocoKinship(G.shape[0], block_size=1000)
    for start in range(0, 400, 64):
        used = a.add_block(G[:, start : start + 64], chrom[start : start + 64])
        assert used.dtype == bool and len(used) == min(64, 400 - start)
        if start == 192:
            a.kinship("1")  # a mid-stream kinship does not disturb the sums
    b = LocoKinship(G.shape[0], block_size=16)
    for j in range(400):
        b.add(G[:, j], chrom[j])
    for c in ("1", "2", "3"):
        _assert_k_close(a.kinship(c), b.kinship(c), "leave out %s" % c)
        _assert_k_close(a.kinship(int(c)), b.kinship(c), "int label %s" % c)
    assert a.chromosomes == ("1", "2", "3")


def test_loco_kinship_errors():
    G = _genotypes(4)
    one = LocoKinship.from_genotypes(G, ["1"] * G.shape[1])
    with pytest.raises(ValueError, match="at least two chromosomes"):
        one.kinship("1")
    one.kinship()  # all SNPs: fine
    with pytest.raises(KeyError, match="no SNP on chromosome '2'"):
        one.kinship("2")
    with pytest.raises(ValueError, match="one chromosome per SNP"):
        LocoKinship.from_genotypes(G, ["1"] * 3)
    with pytest.raises(ValueError, match=r"SNP column 0 of W.*0, 0\.5 and 1"):
        LocoKinship.from_genotypes(G * 2, ["1"] * G.shape[1], kind="ibs")
    with pytest.raises(ValueError, match="kind must be one of"):
        LocoKinship(5, kind="gcta")
    with pytest.raises(ValueError, match="scale applies"):
        LocoKinship(5, scale=2.0)
    with pytest.raises(ValueError, match="length 5"):
        LocoKinship(5).add(np.zeros(4), "1")


# ---------------------------------------------------------------------------
# gwas_loco
# ---------------------------------------------------------------------------


def _hand_loco(Y, X, chrom, loco, X0=None, **kw):
    """The by-hand LOCO loop: gwas on each chromosome with K_{-c}."""
    keep = ~np.isnan(Y)
    t = np.full(X.shape[1], np.nan)
    p = np.full(X.shape[1], np.nan)
    for c in dict.fromkeys(chrom):
        on = chrom == c
        K = loco.kinship(c)[keep][:, keep]
        x0 = None if X0 is None else X0[keep]
        tc, pc = gwas(Y[keep], X[keep][:, on], K, X0=x0, **kw)
        t[on], p[on] = tc, pc
    return t, p


@pytest.mark.parametrize("kw", [{}, {"batch": False}, {"refit": True}, {"reml": False}], ids=str)
def test_gwas_loco_equals_hand_loop(kw):
    """gwas_loco == gwas per chromosome with K_{-c}, bit for bit (missing Y and calls, X0)."""
    rng = np.random.default_rng(5)
    G = _genotypes(5, n=80, m=240)
    chrom = np.repeat(["1", "2", "3"], 80)
    Y = np.nan_to_num(G[:, 20]) + rng.normal(size=80)
    Y[[3, 40]] = np.nan
    X0 = np.column_stack([np.ones(80), rng.normal(size=80)])
    loco = LocoKinship.from_genotypes(G, chrom)
    res = gwas_loco(Y, G, chrom, X0=X0, **kw)
    t, p = _hand_loco(Y, G, chrom, loco, X0=X0, **kw)
    np.testing.assert_array_equal(res.t_stats, t)
    np.testing.assert_array_equal(res.p_values, p)
    assert sorted(res.h) == ["1", "2", "3"] and all(0 <= v <= 1 for v in res.h.values())
    # prebuilt kinships give the same
    res2 = gwas_loco(Y, G, chrom, kinships=loco, X0=X0, **kw)
    np.testing.assert_array_equal(res2.p_values, p)


def test_gwas_loco_vs_calculate_kinship_loop():
    """The same scan with K_{-c} from calculate_kinship of the complement (rounding only)."""
    rng = np.random.default_rng(6)
    G = _genotypes(6, n=70, m=300)
    chrom = np.repeat(["1", "2", "3"], 100)
    Y = rng.normal(size=70)
    t, p, _ = gwas_loco(Y, G, chrom)
    for c in ("1", "2", "3"):
        on = chrom == c
        tc, pc = gwas(Y, G[:, on], calculate_kinship(G[:, ~on]))
        h.assert_fit_close(np.asarray(t)[on], tc, what="t, chromosome %s" % c)
        h.assert_pvalues_close(np.asarray(p)[on], pc, what="p, chromosome %s" % c)


def test_gwas_loco_other_chromosome_and_errors():
    G = _genotypes(7, n=50, m=200)
    chrom = np.repeat(["1", "2"], 100)
    loco = LocoKinship.from_genotypes(G, chrom)
    Y = np.random.default_rng(7).normal(size=50)
    # SNPs on a chromosome the kinships do not have: tested with the all-SNP K
    t, p, hh = gwas_loco(Y, G[:, :10], ["X"] * 10, kinships=loco)
    t_all, _ = gwas(Y, G[:, :10], loco.kinship())
    np.testing.assert_array_equal(t, t_all)
    assert list(hh) == ["X"]
    with pytest.raises(ValueError, match="one chromosome per SNP"):
        gwas_loco(Y, G, chrom[:5])
    with pytest.raises(ValueError, match="kinships are for 50 individuals but Y has 49"):
        gwas_loco(Y[:49], G[:49], chrom, kinships=loco)
    with pytest.raises(ValueError, match="at least two chromosomes"):
        gwas_loco(Y, G, ["1"] * 200)
    all_nan = gwas_loco(np.full(50, np.nan), G, chrom)
    assert np.isnan(all_nan.p_values).all() and all_nan.h == {}


# ---------------------------------------------------------------------------
# pylmm gwas --loco
# ---------------------------------------------------------------------------


def _bed_with_pheno(tmp_path, n=120, m=450, seed=11, missing=0.02):
    base = str(tmp_path / "geno")
    raw = write_synthetic_bed(base, n, m, missing=missing, seed=seed)
    rng = np.random.default_rng(seed)
    y = 0.8 * np.nan_to_num(raw[:, 30] - np.nanmean(raw[:, 30])) / 0.3
    y = y + 0.5 * (np.nan_to_num(raw - np.nanmean(raw, 0)) @ rng.normal(0, 0.05, m))
    y = y + rng.normal(size=n)
    y[[5, 17]] = np.nan
    with open(base + ".phen", "w") as f:
        for i, v in enumerate(y):
            f.write("F%d I%d %s\n" % (i + 1, i + 1, "NA" if np.isnan(v) else repr(float(v))))
    return base, raw, y


def _library_loco(base, y, kind="standardized", norm=True, **kw):
    """What `pylmm gwas --loco` computes, with the library."""
    with PlinkReader(base, norm_genotype=False) as r:
        raw = np.hstack([B for B, _ in r.iter_blocks(100)])
        chrom = np.asarray(r.variants.chrom)
    with PlinkReader(base, norm_genotype=norm) as r:
        Z = np.hstack([B for B, _ in r.iter_blocks(100)])
    loco = LocoKinship.from_genotypes(raw, chrom, kind=kind)
    if kind == "centered":
        loco = LocoKinship.from_genotypes(raw * 2, chrom, kind=kind)
    return gwas_loco(y, Z, chrom, kinships=loco, **kw), loco


@pytest.mark.parametrize("kind", KINDS)
def test_cli_loco_equals_library(tmp_path, kind):
    base, raw, y = _bed_with_pheno(tmp_path)
    out = str(tmp_path / "loco.tsv")
    log = _err(
        ["gwas", "-v", "--bfile", base, "--pheno", base + ".phen", "--loco",
         "--kinship-type", kind, "--block-size", "64", "-o", out]
    )  # fmt: skip
    ids, vals = _out(out)
    assert ids == ["snp%d" % (j + 1) for j in range(450)]
    res, loco = _library_loco(base, y, kind=kind)
    h.assert_batch_close(vals[:, 2], res.t_stats, what="t")
    h.assert_batch_close(vals[:, 3], res.p_values, what="p")
    hs = [float(x) for x in re.findall(r"heritability=(\S+),", log)]
    assert len(hs) == 3
    np.testing.assert_allclose(hs, [res.h[c] for c in ("1", "2", "3")], atol=5.1e-4)
    for c in ("1", "2", "3"):
        assert "chromosome %s: K from the %d SNPs" % (c, loco.snps_used(c)) in log
    # The signal is found
    assert vals[30, 3] < 1e-4  # snp31, the causal SNP


def test_cli_loco_equals_kinship_on_complement(tmp_path):
    """--loco on chromosome 1 == `pylmm kinship` of the other chromosomes + `pylmm gwas --kinship`."""
    base, raw, y = _bed_with_pheno(tmp_path, n=90, m=300, seed=12)
    out = str(tmp_path / "loco.tsv")
    _err(["gwas", "--bfile", base, "--pheno", base + ".phen", "--loco", "-o", out])
    _, vals = _out(out)
    # PLINK files of chromosome 1 and of the rest, from the same bytes
    with open(base + ".bim") as f:
        bim = list(f)
    chrom = np.array([line.split()[0] for line in bim])
    nb = (90 + 3) // 4
    with open(base + ".bed", "rb") as f:
        body = f.read()[3:]
    for name, sel in (("c1", chrom == "1"), ("rest", chrom != "1")):
        p = str(tmp_path / name)
        with open(p + ".bed", "wb") as f:
            f.write(
                b"\x6c\x1b\x01" + b"".join(body[j * nb : (j + 1) * nb] for j in np.flatnonzero(sel))
            )
        with open(p + ".bim", "w") as f:
            f.writelines(bim[j] for j in np.flatnonzero(sel))
        with open(base + ".fam") as src, open(p + ".fam", "w") as dst:
            dst.write(src.read())
    _err(["kinship", "--bfile", str(tmp_path / "rest"), "-o", str(tmp_path / "rest.kin")])
    out1 = str(tmp_path / "c1.tsv")
    _err(
        ["gwas", "--bfile", str(tmp_path / "c1"), "--kinship", str(tmp_path / "rest.kin"),
         "--pheno", base + ".phen", "-o", out1]
    )  # fmt: skip
    _, ref = _out(out1)
    on = chrom == "1"
    h.assert_fit_close(vals[on, 2], ref[:, 2], what="t, chromosome 1")
    h.assert_pvalues_close(vals[on, 3], ref[:, 3], what="p, chromosome 1")


@pytest.mark.parametrize(
    "flags", [["--no-batch"], ["--refit"], ["--remove-missing-genotypes"], ["--ml"]], ids=str
)
def test_cli_loco_options(tmp_path, flags):
    """--loco with --no-batch, --refit, --remove-missing-genotypes, --ml: the library's numbers."""
    base, raw, y = _bed_with_pheno(tmp_path, n=80, m=150, seed=13, missing=0.05)
    out = str(tmp_path / "o.tsv")
    _err(["gwas", "--bfile", base, "--pheno", base + ".phen", "--loco", *flags, "-o", out])
    _, vals = _out(out)
    kw = {
        "--no-batch": {"batch": False},
        "--refit": {"refit": True},
        "--remove-missing-genotypes": {},
        "--ml": {"reml": False},
    }[flags[0]]
    norm = flags[0] != "--remove-missing-genotypes"
    res, _ = _library_loco(base, y, norm=norm, **kw)
    # (--remove-missing-genotypes: the CLI standardises a SNP over its called
    # individuals, gwas tests the raw values: t and p are the same, BETA is not)
    h.assert_batch_close(vals[:, 2], res.t_stats, what="t")
    h.assert_batch_close(vals[:, 3], res.p_values, what="p")


def test_cli_loco_emma_with_map(tmp_path):
    """EMMA genotypes + --emma-map (the bundled HMDP marker map format) == the library."""
    n_snps = 400
    emma = h.write_emma_subset(tmp_path, n_snps)
    rows = [line.split("\t") for line in open(os.path.join(h.DATA, "hmdp.liver.snps.map.tsv"))]
    # the first 400 rows are all chromosome 1: relabel into 3 chromosomes for the test
    chrom = np.repeat(["1", "2", "3"], [150, 150, 100])
    mp = str(tmp_path / "map.tsv")
    with open(mp, "w") as f:
        f.write("\t".join(rows[0]))
        for k in range(n_snps):
            r = rows[k + 1]
            f.write("\t".join([r[0], chrom[k], *r[2:]]))
    out = str(tmp_path / "o.tsv")
    _err(
        ["gwas", "--emma-geno", emma, "--emma-pheno", h.HMDP_Y, "--loco", "--emma-map", mp,
         "--kinship-type", "ibs", "-o", out]
    )  # fmt: skip
    ids, vals = _out(out)
    assert ids[0] == "SNP_1" and len(ids) == n_snps
    G = h.load_hmdp_snps(n_snps)
    Y = np.genfromtxt(h.HMDP_Y)
    with PlinkReader(emma, type="emma") as r:
        Z = np.hstack([B for B, _ in r.iter_blocks(100)])
    res = gwas_loco(Y, Z, chrom, kinships=LocoKinship.from_genotypes(G, chrom, kind="ibs"))
    h.assert_batch_close(vals[:, 3], res.p_values, what="p")
    # a PLINK .map without a header works too
    mp2 = str(tmp_path / "plink.map")
    with open(mp2, "w") as f:
        for k in range(n_snps):
            f.write("%s SNP_%d 0 %d\n" % (chrom[k], k + 1, k + 1))
    out2 = str(tmp_path / "o2.tsv")
    _err(
        ["gwas", "--emma-geno", emma, "--emma-pheno", h.HMDP_Y, "--loco", "--emma-map", mp2,
         "--kinship-type", "ibs", "-o", out2]
    )  # fmt: skip
    np.testing.assert_array_equal(_out(out2)[1], vals)


def test_cli_loco_usage_errors(tmp_path):
    base, raw, y = _bed_with_pheno(tmp_path, n=40, m=60, seed=14)
    ph = base + ".phen"
    out = str(tmp_path / "o.tsv")
    common = ["gwas", "--bfile", base, "--pheno", ph, "-o", out]
    kin = str(tmp_path / "k.kin")
    _err(["kinship", "--bfile", base, "-o", kin])
    cases = [
        (
            ["--loco", "--kinship", kin],
            "--loco builds the kinships from the genotypes: drop --kinship",
        ),
        (["--loco", "--grm", base], "drop --grm"),
        (["--loco", "--eigen", kin], "--eigen does not apply to --loco"),
        (["--kinship", kin, "--kinship-type", "ibs"], "--kinship-type applies only to --loco"),
        ([], "one of the arguments --kinship --grm is required (or --loco)"),
        (["--loco", "--emma-map", ph], "--emma-map applies only to --emma-geno"),
    ]
    for flags, msg in cases:
        err = _err(common + flags, rc=2)
        assert msg in err, (flags, err)
    emma = h.write_emma_subset(tmp_path, 50)
    err = _err(["gwas", "--emma-geno", emma, "--emma-pheno", h.HMDP_Y, "--loco", "-o", out], rc=2)
    assert "--loco needs the chromosome of each SNP" in err and "--emma-map" in err
    short = str(tmp_path / "short.map")
    with open(short, "w") as f:
        f.write("chr\n1\n2\n")
    err = _err(
        ["gwas", "--emma-geno", emma, "--emma-pheno", h.HMDP_Y, "--loco", "--emma-map", short,
         "-o", out], rc=2,
    )  # fmt: skip
    assert "has 2 SNP lines but the EMMA genotype file has 50 rows" in err


def _rewrite_bim_chroms(base, chroms):
    lines = open(base + ".bim").read().splitlines()
    with open(base + ".bim", "w") as f:
        for line, c in zip(lines, chroms):
            f.write("%s\t%s\n" % (c, line.split("\t", 1)[1]))


def test_cli_loco_chromosome_errors(tmp_path):
    base, raw, y = _bed_with_pheno(tmp_path, n=40, m=60, seed=15)
    out = str(tmp_path / "o.tsv")
    argv = ["gwas", "--bfile", base, "--pheno", base + ".phen", "--loco", "-o", out]
    _rewrite_bim_chroms(base, ["1"] * 60)
    assert "at least two chromosomes; the genotype input has chromosome 1 only" in _err(argv, rc=2)
    _rewrite_bim_chroms(base, ["1"] * 20 + ["2"] * 20 + ["1"] * 20)
    err = _err(argv, rc=2)
    assert "the SNPs of chromosome 1 are not together" in err and "--sort-vars" in err
    # chromosome 2 has one SNP, constant: no usable SNP is left for K_{-1}
    _rewrite_bim_chroms(base, ["1"] * 59 + ["2"])
    with open(base + ".bed", "rb") as f:
        data = bytearray(f.read())
    nb = (40 + 3) // 4
    data[3 + 59 * nb : 3 + 60 * nb] = bytes(nb)  # SNP 60: all homozygous -> constant
    with open(base + ".bed", "wb") as f:
        f.write(bytes(data))
    assert "no usable SNP outside chromosome 1, so no kinship without it" in _err(argv, rc=2)


@pytest.mark.slow
def test_cli_loco_readme_data(tmp_path):
    """The README BED: --loco runs on the full file; on chromosome 19 it equals the library."""
    out = str(tmp_path / "loco.tsv")
    _err(["gwas", "--bfile", h.BED_BASE, "--pheno", h.FAKE_PHENOS, "--loco", "-o", out])
    ids, vals = _out(out)
    assert len(ids) == 98740 and np.isfinite(vals[:, 3]).mean() > 0.99
    with PlinkReader(h.BED_BASE, norm_genotype=False) as r:
        chrom = np.asarray(r.variants.chrom)
        raw = np.hstack([B for B, _ in r.iter_blocks(4096)])
    with PlinkReader(h.BED_BASE, pheno_file=h.FAKE_PHENOS) as r:
        on = np.flatnonzero(chrom == "19")
        Z = np.hstack([B for B, _ in r.iter_blocks(4096)])[:, on]
        y = r.phenos[:, 0]
    loco = LocoKinship.from_genotypes(raw, chrom)
    del raw
    t, p = gwas(y, Z, loco.kinship("19"))
    h.assert_batch_close(vals[on, 3], p, what="p, chromosome 19")
