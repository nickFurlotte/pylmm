"""pylmm.kinship: the kinds of kinship, one missing-call convention for the library and
`pylmm kinship`, the kinship scale, and the legacy (1.0.0) switches."""

import warnings

import helpers as h
import numpy as np
import pytest
from synthetic_bed import write_synthetic_bed

from pylmm import kinship, lmm
from pylmm.kinship import KinshipAccumulator, calculate_kinship, kinship_scale


def emma_kinship_reference(snps):
    """A line-by-line transliteration of ``emma.kinship(snps, method="additive", use="all")``
    from the EMMA R package (emma.R, Kang et al. 2008): ``snps`` is SNPs x individuals,
    coded 0 / 0.5 / 1, nan for NA.  Slow on purpose (the R double loop)."""
    snps = np.array(snps, dtype=float)
    called = snps[~np.isnan(snps)]
    assert np.isin(called, (0, 0.5, 1)).all()  # stopifnot(n0+nh+n1+nNA == length(snps))
    nh = (called == 0.5).sum()
    if nh > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row_means = np.nanmean(snps, axis=1)  # rowMeans(snps, na.rm=TRUE)
        het = ~np.isnan(snps) & (snps == 0.5)
        dsnps, rsnps = snps.copy(), snps.copy()
        flags = np.repeat((row_means > 0.5).astype(float)[:, None], snps.shape[1], axis=1)
        dsnps[het] = flags[het]
        flags = np.repeat((row_means < 0.5).astype(float)[:, None], snps.shape[1], axis=1)
        rsnps[het] = flags[het]
        snps = np.vstack([dsnps, rsnps])
    with warnings.catch_warnings():  # the mean of an all-NA row is nan, as in R
        warnings.simplefilter("ignore", RuntimeWarning)
        mafs = np.repeat(np.nanmean(snps, axis=1)[:, None], snps.shape[1], axis=1)
    na = np.isnan(snps)
    snps[na] = mafs[na]
    n = snps.shape[1]
    K = np.full((n, n), np.nan)
    np.fill_diagonal(K, 1.0)
    for i in range(1, n):
        for j in range(i):
            x = snps[:, i] * snps[:, j] + (1 - snps[:, i]) * (1 - snps[:, j])
            K[i, j] = np.nansum(x) / (~np.isnan(x)).sum()
            K[j, i] = K[i, j]
    return K


def _panel(n=25, m=120, het=0.0, missing=0.05, seed=0):
    """0 / 0.5 / 1 genotypes with missing calls, an all-missing and two constant SNPs."""
    rng = np.random.default_rng(seed)
    G = rng.integers(0, 2, size=(n, m)).astype(float)
    G[rng.random(G.shape) < het] = 0.5
    G[rng.random(G.shape) < missing] = np.nan
    G[:, 3] = np.nan  # all missing
    G[:, 5] = 1.0  # constant
    G[:, 7] = 0.0
    G[0, 7] = np.nan  # constant where called
    return G


def test_lmm_reexports_calculate_kinship():
    assert lmm.calculate_kinship is kinship.calculate_kinship


@pytest.mark.parametrize("het", [0.0, 0.2])
def test_ibs_is_emma_kinship(het):
    G = _panel(het=het, seed=int(het * 10))
    np.testing.assert_allclose(
        calculate_kinship(G, kind="ibs"), emma_kinship_reference(G.T), rtol=0, atol=1e-13
    )


def test_ibs_is_allele_sharing_on_complete_calls():
    """Without missing calls, EMMA's IBS is 1 - |x_i - x_j| averaged over SNPs (two
    heterozygotes share 1, a heterozygote and a homozygote 0.5), except on a SNP whose mean
    is exactly 0.5, where EMMA codes a heterozygote as 0 in both of its 0/1 codings."""
    rng = np.random.default_rng(3)
    G = rng.choice([0.0, 0.5, 1.0], size=(12, 200))
    G = G[:, G.mean(axis=0) != 0.5]
    expected = 1 - np.abs(G[:, None, :] - G[None, :, :]).mean(axis=2)
    np.testing.assert_allclose(calculate_kinship(G, kind="ibs"), expected, rtol=0, atol=1e-13)


def test_ibs_rejects_counts():
    with pytest.raises(ValueError, match=r"0, 0\.5 and 1"):
        calculate_kinship(np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]), kind="ibs")


def test_centered_is_gemma_gk1():
    """GEMMA -gk 1: centred 0/1/2 counts, missing calls at the mean, K = C C' / m."""
    G = 2 * _panel(het=0.2, seed=4)
    used = [j for j in range(G.shape[1]) if np.nanstd(G[:, j]) > 0]
    C = G[:, used] - np.nanmean(G[:, used], axis=0)
    C[np.isnan(C)] = 0
    np.testing.assert_allclose(
        calculate_kinship(G, kind="centered"), C @ C.T / len(used), rtol=1e-12, atol=1e-14
    )


def test_standardized_convention():
    """Standardised with the called genotypes' mean and SD, missing -> 0, m = SNPs used."""
    G = _panel(het=0.2, seed=5)
    used = [j for j in range(G.shape[1]) if np.nanstd(G[:, j]) > 0]
    Z = (G[:, used] - np.nanmean(G[:, used], axis=0)) / np.nanstd(G[:, used], axis=0)
    Z[np.isnan(Z)] = 0
    K = calculate_kinship(G)
    np.testing.assert_allclose(K, Z @ Z.T / len(used), rtol=1e-12, atol=1e-14)
    # the mean diagonal is the share of called genotypes among the SNPs used
    called = (~np.isnan(G[:, used])).mean()
    np.testing.assert_allclose(kinship_scale(K).mean_diag, called, rtol=1e-12)


def test_legacy_is_the_1_0_0_function():
    """legacy=True: impute, then standardise with the variance after imputation; m = all SNPs."""
    G = _panel(het=0.2, seed=6)
    m = G.shape[1]
    A = G.copy()
    used = []
    for j in range(m):
        obs = ~np.isnan(A[:, j])
        if not obs.any():
            continue
        A[~obs, j] = A[obs, j].mean()
        if A[:, j].var() > 0:
            used.append(j)
    Z = (A[:, used] - A[:, used].mean(axis=0)) / A[:, used].std(axis=0)
    np.testing.assert_allclose(
        calculate_kinship(G, legacy=True), Z @ Z.T / m, rtol=1e-12, atol=1e-14
    )
    with pytest.raises(ValueError, match="legacy"):
        calculate_kinship(G, kind="ibs", legacy=True)


def test_monomorphic_and_all_missing_snps_do_not_change_k():
    G = _panel(seed=7)
    keep = [j for j in range(G.shape[1]) if j not in (3, 5, 7)]
    for kind in ("standardized", "centered"):
        np.testing.assert_allclose(
            calculate_kinship(G, kind=kind), calculate_kinship(G[:, keep], kind=kind), rtol=1e-12
        )


def test_errors():
    with pytest.raises(ValueError, match="kind must be"):
        calculate_kinship(np.ones((3, 2)), kind="grm")
    with pytest.raises(ValueError, match="non-constant"):
        calculate_kinship(np.ones((3, 2)))
    with pytest.raises(ValueError, match="2-d"):
        calculate_kinship(np.ones(3))
    with pytest.raises(ValueError, match="square"):
        kinship_scale(np.ones((2, 3)))
    with pytest.raises(ValueError, match="length 3"):
        KinshipAccumulator(3).add(np.ones(4))


def test_kinship_scale_is_invariant_to_a_constant():
    K = calculate_kinship(_panel(seed=8))
    s, s1 = kinship_scale(K), kinship_scale(K + 0.3)
    assert s.mean_diag + 0.3 == pytest.approx(s1.mean_diag)
    assert s.centered_mean_diag == pytest.approx(s1.centered_mean_diag)
    k_gower = calculate_kinship(_panel(seed=8), center=True)
    assert kinship_scale(k_gower).centered_mean_diag == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["standardized", "centered", "ibs"])
def test_cli_equals_library_bit_for_bit(tmp_path, kind):
    """Missing calls (15%), an all-missing and a constant SNP: `pylmm kinship` writes the K
    of calculate_kinship, to the last bit (the same products in the same order at the
    default block size of both); the scale is logged."""
    base = str(tmp_path / "s")
    raw = write_synthetic_bed(base, 40, 2500, missing=0.15, seed=1)
    raw = raw.copy()
    with open(base + ".bed", "r+b") as f:  # SNP 10 all missing (code 01), SNP 11 constant (00)
        nb = (40 + 3) // 4
        f.seek(3 + 10 * nb)
        f.write(bytes([0x55]) * nb)
        f.write(bytes([0x00]) * nb)
    raw[:, 10] = np.nan
    raw[:, 11] = 0.0
    out = str(tmp_path / "k")
    log = h.run_cli(["kinship", "-v", "--bfile", base, "--kinship-type", kind, "-o", out], tmp_path)
    W = 2 * raw if kind == "centered" else raw
    expected = calculate_kinship(W, kind=kind)
    np.testing.assert_array_equal(np.loadtxt(out), expected)
    s = kinship_scale(expected)
    assert "kinship scale: mean diagonal %.4f" % s.mean_diag in log
    assert "centred mean diagonal trace(PKP)/(n-1) %.4f" % s.centered_mean_diag in log


def test_cli_legacy_flag_checks(tmp_path):
    base = str(tmp_path / "s")
    write_synthetic_bed(base, 10, 20, seed=2)
    log = h.run_cli(
        ["kinship", "--bfile", base, "--kinship-type", "ibs", "--legacy-kinship", "-o", "k"],
        tmp_path,
        expect_rc=2,
    )
    assert "--legacy-kinship applies only to --kinship-type standardized" in log


def test_cli_ibs_rejects_dosages(tmp_path):
    emma = tmp_path / "g.txt"
    emma.write_text("0 1 0.25 1\n1 0 1 0\n")
    log = h.run_cli(
        ["kinship", "--emma-geno", str(emma), "--kinship-type", "ibs", "-o", "k"],
        tmp_path,
        expect_rc=2,
    )
    assert "--kinship-type ibs, SNP 1" in log and "0, 0.5 and 1" in log


def test_cli_emma_ibs_matches_library(tmp_path, hmdp):
    _, _, snps = hmdp  # 94 strains x 1000 SNPs, 0/1, nan
    emma = h.write_emma_subset(tmp_path)
    h.run_cli(["kinship", "--emma-geno", emma, "--kinship-type", "ibs", "-o", "k"], tmp_path)
    np.testing.assert_array_equal(
        np.loadtxt(str(tmp_path / "k")), calculate_kinship(snps, kind="ibs")
    )
