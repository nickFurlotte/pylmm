"""Kinship: lmm.calculate_kinship (Python API) and `pylmm kinship`."""

import os

import helpers as h
import numpy as np
import pytest

from pylmm import lmm


def check_kinship_invariants(K, what):
    assert K.shape[0] == K.shape[1]
    np.testing.assert_allclose(K, K.T, rtol=0, atol=h.SYM_ATOL, err_msg=what + ": not symmetric")
    eig = np.linalg.eigvalsh((K + K.T) / 2.0)
    assert eig.min() >= -h.PSD_RTOL * eig.max(), "%s: not PSD (min eig %g, max %g)" % (
        what,
        eig.min(),
        eig.max(),
    )


def check_big_kinship(K, name):
    """Compare a 1219x1219 kinship against its stored leading and trailing
    blocks, diagonal and K.dot(cos(arange(n)))."""
    b = h.BED_BLOCK
    h.assert_close(
        K[:b, :b], h.load_golden_txt(name + ".block%d.txt" % b), what=name + " leading block"
    )
    h.assert_close(
        K[-b:, -b:], h.load_golden_txt(name + ".tail%d.txt" % b), what=name + " trailing block"
    )
    dk = h.load_golden_txt(name + ".diag_Kv.txt")
    h.assert_close(np.diag(K), dk[:, 0], what=name + " diag")
    # Unlike the row sums (~0, pure round-off, since the genotype columns are
    # centred), |K.dot(v)| is 0.09..5.9 here, so the default tolerance works.
    h.assert_close(K.dot(np.cos(np.arange(K.shape[0]))), dk[:, 1], what=name + " K.dot(v)")


def assert_eigenvalues_close(actual, expected, what):
    atol = h.EIG_ATOL_SCALE * np.abs(expected).max()
    h.assert_close(np.sort(actual), expected, rtol=h.RTOL, atol=atol, what=what)


@pytest.mark.parametrize(
    ("center", "name"),
    [
        (False, "kinship_hmdp_snps1000.txt"),
        (True, "kinship_hmdp_snps1000_centered.txt"),
    ],
)
def test_calculate_kinship_hmdp(hmdp, center, name):
    _, _, snps = hmdp
    K = lmm.calculate_kinship(snps.copy(), center=center, legacy=True)
    h.assert_close(K, h.load_golden_txt(name), what=name)
    check_kinship_invariants(K, name)


def test_calculate_kinship_bed():
    W = h.read_bed_snps(h.BED_KIN_NSNPS, norm=False)
    K = lmm.calculate_kinship(W, legacy=True)
    check_big_kinship(K, "kinship_bed_snps%d_calc" % h.BED_KIN_NSNPS)
    check_kinship_invariants(K, "calculate_kinship(bed)")


def test_cli_kinship_bed(tmp_path):
    base = h.write_bed_subset(tmp_path, h.BED_KIN_NSNPS)
    out = str(tmp_path / "kin.txt")
    h.run_cli(["kinship", "-v", "--bfile", base, "-o", out], cwd=tmp_path)
    K = np.loadtxt(out)
    check_big_kinship(K, "kinship_cli_bed_snps%d" % h.BED_KIN_NSNPS)
    check_kinship_invariants(K, "pylmm kinship --bfile")


def test_cli_kinship_emma(tmp_path):
    emma = h.write_emma_subset(tmp_path)
    out = str(tmp_path / "kin.txt")
    h.run_cli(
        [
            "kinship",
            "-v",
            "--emma-geno",
            emma,
            "--emma-num-snps",
            h.HMDP_NSNPS,
            "--block-size",
            300,
            "--eigen-text",
            "--legacy-kinship",  # the golden is 1.0.0's K (7% of these SNPs are monomorphic)
            "-o",
            out,
        ],
        cwd=tmp_path,
    )
    K = np.loadtxt(out)
    h.assert_close(
        K, h.load_golden_txt("kinship_cli_emma_hmdp_snps1000.txt"), what="pylmm kinship --emma-geno"
    )
    check_kinship_invariants(K, "pylmm kinship --emma-geno")
    assert_eigenvalues_close(
        np.loadtxt(out + ".kva"),
        h.load_golden_txt("kinship_cli_emma_hmdp_snps1000.kva.txt"),
        "pylmm kinship --eigen-text (.kva)",
    )


def test_cli_kinship_emma_removes_stale_id_file(tmp_path):
    """(Also: without --emma-num-snps the SNPs are counted.)"""
    emma = h.write_emma_subset(tmp_path)
    out = str(tmp_path / "kin.txt")
    stale = out + ".id"
    with open(stale, "w") as f:
        f.write("1\t1\n")
    log = h.run_cli(
        [
            "kinship",
            "--emma-geno",
            emma,
            "--block-size",
            300,
            "-o",
            out,
        ],
        cwd=tmp_path,
    )
    assert not os.path.exists(stale)
    assert "stale ID file" in log


def test_readme_kinship_eigenvalues(bed_null):
    _, K, kva, _ = bed_null
    assert_eigenvalues_close(
        kva, h.load_golden_txt("bed_readme_kin_eigenvalues.txt"), "eigh(README kinship)"
    )
    check_kinship_invariants(K, "README kinship")


def test_npdump_pickles_match_text_equivalents(hmdp):
    """The pickled inputs used by example.py equal the text inputs the suite
    uses.  Kept so Phase 3 notices if pickle loading changes."""
    _, K, snps = hmdp
    h.assert_close(
        h.load_npdump(os.path.join(h.DATA, "hmdp.liver.K.npdump")),
        K,
        rtol=0,
        atol=0,
        what="hmdp.liver.K.npdump",
    )
    h.assert_close(
        h.load_npdump(os.path.join(h.DATA, "hmdp.liver.snps.1000.npdump")).T,
        snps,
        rtol=0,
        atol=0,
        what="hmdp.liver.snps.1000.npdump",
    )


@pytest.mark.slow
def test_cli_kinship_full_bed_reproduces_readme_kin(tmp_path):
    """README kinship command on all 98,740 SNPs (~1 min emulated)."""
    out = str(tmp_path / "kin.txt")
    h.run_cli(["kinship", "--bfile", h.BED_BASE, "-o", out], cwd=tmp_path)
    K = np.loadtxt(out)
    h.assert_close(
        K, h.load_readme_kinship(K.shape[0]), what="pylmm kinship full BED vs data/*.pylmm.kin"
    )


def test_cli_kinship_bim_the_block_reader_rejects(tmp_path):
    """A 5-column .bim (no A2): read one SNP at a time, with a warning; the same K bytes."""
    base = h.write_bed_subset(tmp_path, 300)
    h.run_cli(["kinship", "--bfile", base, "-o", "six.kin", "--block-size", "64"], cwd=tmp_path)
    with open(base + ".bim") as f:
        rows = [line.split()[:5] for line in f]
    with open(base + ".bim", "w") as f:
        f.writelines(" ".join(r) + "\n" for r in rows)
    log = h.run_cli(
        ["kinship", "--bfile", base, "-o", "five.kin", "--block-size", "64"], cwd=tmp_path
    )
    assert "reading the SNPs one at a time" in log and "line 1" in log
    with open(tmp_path / "six.kin", "rb") as a, open(tmp_path / "five.kin", "rb") as b:
        assert a.read() == b.read()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("block_size", ["1", "7", "1000"])
def test_cli_kinship_is_w_wt_over_m_at_any_block_size(tmp_path, block_size, legacy):
    """Missing calls, an all-missing and a constant SNP: at each block size, K is W W' / m over
    the kept SNPs, to rounding; m counts the kept SNPs, or all SNPs with --legacy-kinship
    (1.0.0).  (Byte identity of --legacy-kinship with 1.0.0 was checked outside the suite.)"""
    base = h.write_bed_subset(tmp_path, 40)
    with open(base + ".fam") as f:
        n = sum(1 for _ in f)
    nb = (n + 3) // 4
    raw = bytearray(open(base + ".bed", "rb").read())
    for i in range(n):  # SNP 3 all missing (code 01), SNP 4 constant (code 00)
        sh = 2 * (i % 4)
        raw[3 + 2 * nb + i // 4] = (raw[3 + 2 * nb + i // 4] & ~(3 << sh)) | (1 << sh)
        raw[3 + 3 * nb + i // 4] &= ~(3 << sh)
    for i in range(0, n, 7):  # missing calls in SNP 6
        sh = 2 * (i % 4)
        raw[3 + 5 * nb + i // 4] = (raw[3 + 5 * nb + i // 4] & ~(3 << sh)) | (1 << sh)
    with open(base + ".bed", "wb") as f:
        f.write(bytes(raw))
    out = str(tmp_path / "k.txt")
    flags = ["--legacy-kinship"] if legacy else []
    log = h.run_cli(
        ["kinship", "--bfile", base, "-o", out, "--block-size", block_size, *flags], cwd=tmp_path
    )
    assert "skipped 1 SNPs with no called genotype" in log
    K = np.loadtxt(out)
    from pylmm.plink_io import PlinkReader

    W = np.column_stack([g for g, _ in PlinkReader(base).iter_snps()])
    keep = np.isfinite(W).all(axis=0) & (W.var(axis=0) > 0)
    m = W.shape[1] if legacy else keep.sum()
    h.assert_close(K, W[:, keep] @ W[:, keep].T / m, what="K")
    if not legacy:  # the library computes the same K
        G = np.column_stack([g for g, _ in PlinkReader(base, norm_genotype=False).iter_snps()])
        h.assert_close(K, lmm.calculate_kinship(G), what="K vs calculate_kinship")
