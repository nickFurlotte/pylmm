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
    K = lmm.calculate_kinship(snps.copy(), center=center)
    h.assert_close(K, h.load_golden_txt(name), what=name)
    check_kinship_invariants(K, name)


def test_calculate_kinship_bed():
    W = h.read_bed_snps(h.BED_KIN_NSNPS, norm=False)
    K = lmm.calculate_kinship(W)
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
            "--emma-num-snps",
            h.HMDP_NSNPS,
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
