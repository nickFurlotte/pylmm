"""Regression tests for pylmm.plink_io bugs found while adding type hints (Phase 5.5)."""

import numpy as np
import pytest

from pylmm.plink_io import PlinkReader


def _write_tped(tmp_path, rows):
    """A 3-individual .tped/.tfam/.map set; ``rows`` are the allele fields of each SNP."""
    base = str(tmp_path / "toy")
    with open(base + ".tfam", "w") as f:
        f.write("F1 I1 0 0 1 -9\nF1 I2 0 0 1 -9\nF2 I3 0 0 2 -9\n")
    with open(base + ".map", "w") as f:
        for k in range(len(rows)):
            f.write("1 rs%d 0 %d\n" % (k + 1, 100 * (k + 1)))
    with open(base + ".tped", "w") as f:
        for k, alleles in enumerate(rows):
            f.write("1 rs%d 0 %d %s\n" % (k + 1, 100 * (k + 1), alleles))
    return base


def test_tped_valid_codes_unchanged(tmp_path):
    base = _write_tped(tmp_path, ["1 1 1 2 2 2", "0 0 2 1 1 1"])
    r = PlinkReader(base, type="t", norm_genotype=False)
    out = [(g.tolist(), name) for g, name in r.get_snp_iterator()]
    assert out[0] == ([0.0, 0.5, 1.0], "rs1")
    assert np.isnan(out[1][0][0]) and out[1][0][1:] == [0.5, 0.0]


@pytest.mark.parametrize("alleles", ["1 1 A A 2 2", "1 1 3 3 2 2", "A A 1 1 2 2"])
def test_tped_unknown_allele_code_raises(tmp_path, alleles):
    """Bug: a homozygous code other than 0/1/2 left `g` unassigned, so the
    previous individual's genotype was silently reused (or, for the first
    individual, UnboundLocalError)."""
    base = _write_tped(tmp_path, ["1 1 1 2 2 2", alleles])
    r = PlinkReader(base, type="t", norm_genotype=False)
    it = r.get_snp_iterator()
    next(it)
    with pytest.raises(ValueError, match=r"SNP rs2: allele code '(A|3)' is not 0, 1 or 2"):
        next(it)


def test_normalize_genotype_does_not_modify_input():
    """Bug 12 class: mean imputation used to fill the caller's array in place."""
    G = np.array([0.0, np.nan, 1.0, 0.5])
    before = G.copy()
    out = PlinkReader.normalize_genotype(None, G)  # type: ignore[arg-type]
    np.testing.assert_array_equal(G, before)  # nan still there
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(out.std(), np.std([0.0, 0.5, 1.0, 0.5]) / np.std([0, 1, 0.5]))


def test_get_covariates_none_raises_value_error(tmp_path):
    """Bug: get_covariates(None) raised TypeError from os.path.isfile."""
    base = _write_tped(tmp_path, ["1 1 1 2 2 2"])
    r = PlinkReader(base, type="t", norm_genotype=False)
    with pytest.raises(ValueError, match="cov_file is None"):
        r.get_covariates(None)  # type: ignore[arg-type]


def test_unknown_type_raises_at_construction(tmp_path):
    """get_snp_iterator used to write to stderr and return None for an unknown type."""
    base = _write_tped(tmp_path, ["1 1 1 2 2 2"])
    with pytest.raises(ValueError, match="unknown genotype file type 'x'"):
        PlinkReader(base, type="x")


def test_emma_sets_n_and_integer_indivs(tmp_path):
    fn = tmp_path / "geno.emma"
    fn.write_text("0 1 2 NA\n1 1 0 2\n")
    r = PlinkReader(str(fn), type="emma", norm_genotype=False)
    assert r.N == 4
    assert r.indivs == [0, 1, 2, 3]
    g, name = next(r.get_snp_iterator())
    assert name == "SNP_1" and np.isnan(g[3]) and g[:3].tolist() == [0.0, 1.0, 2.0]
