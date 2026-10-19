"""Tests of pylmm.plink_io: bugs found while adding type hints (Phase 5.5), and the
Phase 5b ID alignment, file handles and format checks."""

import warnings

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
    out = [(g.tolist(), name) for g, name in r.iter_snps()]
    assert out[0] == ([0.0, 0.5, 1.0], "rs1")
    assert np.isnan(out[1][0][0]) and out[1][0][1:] == [0.5, 0.0]


@pytest.mark.parametrize("alleles", ["1 1 A A 2 2", "1 1 3 3 2 2", "A A 1 1 2 2"])
def test_tped_unknown_allele_code_raises(tmp_path, alleles):
    """Bug: a homozygous code other than 0/1/2 left `g` unassigned, so the
    previous individual's genotype was silently reused (or, for the first
    individual, UnboundLocalError)."""
    base = _write_tped(tmp_path, ["1 1 1 2 2 2", alleles])
    r = PlinkReader(base, type="t", norm_genotype=False)
    it = r.iter_snps()
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


def test_read_covariates_none_raises_value_error(tmp_path):
    """Bug: read_covariates(None) raised TypeError from os.path.isfile."""
    base = _write_tped(tmp_path, ["1 1 1 2 2 2"])
    r = PlinkReader(base, type="t", norm_genotype=False)
    with pytest.raises(ValueError, match="cov_file is None"):
        r.read_covariates(None)  # type: ignore[arg-type]


def test_unknown_type_raises_at_construction(tmp_path):
    """iter_snps used to write to stderr and return None for an unknown type."""
    base = _write_tped(tmp_path, ["1 1 1 2 2 2"])
    with pytest.raises(ValueError, match="unknown genotype file type 'x'"):
        PlinkReader(base, type="x")


def test_emma_sets_n_and_integer_indivs(tmp_path):
    fn = tmp_path / "geno.emma"
    fn.write_text("0 1 2 NA\n1 1 0 2\n")
    r = PlinkReader(str(fn), type="emma", norm_genotype=False)
    assert r.N == 4
    assert r.indivs == [0, 1, 2, 3]
    g, name = next(r.iter_snps())
    assert name == "SNP_1" and np.isnan(g[3]) and g[:3].tolist() == [0.0, 1.0, 2.0]


def test_auto_found_kin_file_not_read_gives_k_none(tmp_path):
    """Bug: with fbase + '.kin' present and read_k_file=False, self.K was never
    assigned, so reader.K raised AttributeError."""
    fn = tmp_path / "geno.emma"
    fn.write_text("0 1 2\n1 1 0\n")
    (tmp_path / "geno.emma.kin").write_text("1 0 0\n0 1 0\n0 0 1\n")
    r = PlinkReader(str(fn), type="emma")
    assert r.K is None
    assert r.k_file == str(fn) + ".kin"
    r2 = PlinkReader(str(fn), type="emma", read_k_file=True)
    assert r2.K is not None and r2.K.shape == (3, 3)


# ---------------------------------------------------------------------------
# Phase 5b
# ---------------------------------------------------------------------------

# 4 individuals x 3 SNPs, coded as pylmm decodes BED (fraction of A2 alleles).
G4 = np.array(
    [
        [0.0, 1.0, 0.5],
        [0.5, 0.0, 0.5],
        [1.0, np.nan, 0.0],
        [0.0, 0.5, 1.0],
    ]
)
IDS4 = [("F1", "I1"), ("F1", "I2"), ("F2", "I3"), ("F3", "I4")]


def _write_bed(tmp_path, G=G4, ids=IDS4, header=bytes([0x6C, 0x1B, 0x01])):
    """BED/BIM/FAM for G, written independently of pylmm's reader."""
    base = str(tmp_path / "geno")
    n, m = G.shape
    code = np.where(np.isnan(G), 1, np.where(G == 0, 0, np.where(G == 0.5, 2, 3))).astype(np.uint8)
    nb = (n + 3) // 4
    with open(base + ".bed", "wb") as f:
        f.write(header)
        for j in range(m):
            c = np.zeros(nb * 4, dtype=np.uint8)
            c[:n] = code[:, j]
            c = c.reshape(nb, 4)
            f.write((c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).tobytes())
    with open(base + ".bim", "w") as f:
        for j in range(m):
            f.write("1\trs%d\t0\t%d\tA\tG\n" % (j + 1, 100 * (j + 1)))
    with open(base + ".fam", "w") as f:
        for fid, iid in ids:
            f.write("%s %s 0 0 0 -9\n" % (fid, iid))
    return base


def _write_fastlmm_kinship(fn, ids, K):
    """FaSTLMM style: a tab-separated header of 'FID IID' labels, then labelled rows."""
    with open(fn, "w") as f:
        f.write("var\t" + "\t".join("%s %s" % k for k in ids) + "\n")
        for k, row in zip(ids, K):
            f.write("%s %s\t" % k + "\t".join("%r" % float(x) for x in row) + "\n")


def _snps(reader):
    return np.array([g for g, _ in reader]).T  # individuals x SNPs


def test_fastlmm_kinship_that_drops_individuals_subsets_the_snp_vectors(tmp_path):
    """Regression (the TODO(5b) misalignment): a FaSTLMM kinship that lists only
    some individuals shrank indivs, K and phenos, but every SNP vector kept
    all N values, so genotype row i no longer belonged to individual i.  The
    vectors are now subset (and reordered) the same way."""
    base = _write_bed(tmp_path)
    kin_ids = [IDS4[3], IDS4[0], IDS4[2]]  # I2 is missing, the order differs
    K = np.array([[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]])
    kfn = str(tmp_path / "k.fastlmm")
    _write_fastlmm_kinship(kfn, kin_ids, K)
    with open(base + ".phenos", "w") as f:
        for i, (fid, iid) in enumerate(IDS4):
            f.write("%s %s %d\n" % (fid, iid, 10 + i))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = PlinkReader(base, k_file=kfn, fastlmm_kinship=True, norm_genotype=False)
    assert r.indivs == [IDS4[0], IDS4[2], IDS4[3]]
    assert r.indivs_removed == [IDS4[1]]
    assert r.N == 4
    W = _snps(r)
    assert W.shape == (3, 3)  # it used to be (4, 3)
    np.testing.assert_array_equal(W, G4[[0, 2, 3]])
    assert "1 of 4 samples of the genotype file are not in the kinship and are dropped: F1 I2" in [
        str(w.message) for w in rec
    ]
    np.testing.assert_array_equal(r.phenos[:, 0], [10, 12, 13])
    # K reordered from the file's order (I4, I1, I3) to the reader's (I1, I3, I4)
    o = [1, 2, 0]
    np.testing.assert_array_equal(r.K, K[np.ix_(o, o)])


def test_kinship_id_sidecar_aligns_k_and_snp_vectors(tmp_path):
    base = _write_bed(tmp_path)
    kin_ids = [IDS4[2], IDS4[1], IDS4[0], ("FX", "IX")]  # reversed, no I4, a stranger
    K = np.diag([1.0, 2.0, 3.0, 4.0])
    kfn = str(tmp_path / "geno.kin")
    np.savetxt(kfn, K)
    with open(kfn + ".id", "w") as f:
        f.write("".join("%s %s\n" % k for k in kin_ids))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = PlinkReader(base, k_file=kfn)
    msgs = [str(w.message) for w in rec]
    assert (
        "1 of 4 samples of the genotype file are not in the kinship and are dropped: F3 I4" in msgs
    )
    assert (
        "1 of 4 samples of the kinship are not in the genotype file and are ignored: FX IX" in msgs
    )
    assert r.indivs == IDS4[:3]
    np.testing.assert_array_equal(np.diag(r.K), [3.0, 2.0, 1.0])
    W = _snps(r)
    # standardised over the 3 kept individuals
    raw = G4[:3].copy()
    for j in range(3):
        col = raw[:, j]
        ok = ~np.isnan(col)
        m, s = col[ok].mean(), col[ok].std()
        exp = (np.where(ok, col, m) - m) / s
        np.testing.assert_allclose(W[:, j], exp, atol=1e-12)


def test_plain_kinship_without_ids_must_match_the_genotypes(tmp_path):
    base = _write_bed(tmp_path)
    kfn = str(tmp_path / "geno.kin")
    np.savetxt(kfn, np.eye(3))
    with pytest.raises(ValueError, match="is a 3x3 matrix but there are 4 individuals; without"):
        PlinkReader(base, k_file=kfn)
    np.savetxt(kfn, np.eye(4))
    np.testing.assert_array_equal(PlinkReader(base, k_file=kfn).K, np.eye(4))


def test_subset_reorders_everything(tmp_path):
    base = _write_bed(tmp_path)
    with open(base + ".phenos", "w") as f:
        f.write("".join("%s %s %d\n" % (k[0], k[1], i) for i, k in enumerate(IDS4)))
    r = PlinkReader(base, norm_genotype=False)
    r.subset([IDS4[3], IDS4[1]])
    assert r.indivs == [IDS4[3], IDS4[1]] and r.indivs_removed == [IDS4[0], IDS4[2]]
    np.testing.assert_array_equal(r.phenos[:, 0], [3, 1])
    np.testing.assert_array_equal(_snps(r), G4[[3, 1]])
    r.subset([IDS4[1]])  # subsetting again composes
    np.testing.assert_array_equal(_snps(r), G4[[1]])
    with pytest.raises(ValueError, match="not in the genotype file"):
        r.subset([IDS4[0]])


def test_iid_only_matching_of_phenotypes(tmp_path):
    base = _write_bed(tmp_path)
    fn = str(tmp_path / "p.txt")
    with open(fn, "w") as f:  # FIDs differ from the .fam file
        f.write("0 I3 3.0\n0 I1 1.0\n")
    with pytest.warns(UserWarning, match="2 of 4 individuals are not in the phenotype file"):
        r = PlinkReader(base, pheno_file=fn, id_match="iid")
    np.testing.assert_array_equal(r.phenos[[0, 2], 0], [1.0, 3.0])
    with pytest.raises(ValueError, match="none of the 4 individuals"):
        PlinkReader(base, pheno_file=fn)
    with pytest.raises(ValueError, match="id_match='x'"):
        PlinkReader(base, id_match="x")


def test_phenotype_columns_by_name_with_a_header(tmp_path):
    base = _write_bed(tmp_path)
    fn = str(tmp_path / "p.txt")
    with open(fn, "w") as f:
        f.write("FID IID height bmi\n")
        f.write("".join("%s %s %d %d\n" % (k[0], k[1], i, 20 + i) for i, k in enumerate(IDS4)))
    r = PlinkReader(base, pheno_file=fn)
    assert r.pheno_names == ["height", "bmi"] and r.phenos.shape == (4, 2)
    P = r.read_phenos(fn, columns=["bmi"])
    assert r.pheno_names == ["bmi"]
    np.testing.assert_array_equal(P[:, 0], [20, 21, 22, 23])
    C = r.read_covariates(fn, columns=["height"])
    assert r.covariate_names == ["height"]
    np.testing.assert_array_equal(C[:, 0], [0, 1, 2, 3])


def test_duplicate_ids_in_a_phenotype_file_raise(tmp_path):
    base = _write_bed(tmp_path)
    fn = str(tmp_path / "p.txt")
    with open(fn, "w") as f:
        f.write("F1 I1 1\nF1 I2 2\nF1 I1 3\n")
    with pytest.raises(ValueError, match=r"phenotype file .* lists sample F1 I1 twice"):
        PlinkReader(base, pheno_file=fn)


def test_bed_not_snp_major_raises_value_error_at_construction(tmp_path):
    """It used to write to stderr and raise a bare StopIteration from
    get_snp_iterator(), which escaped a for loop as an exception."""
    base = _write_bed(tmp_path, header=bytes([0x6C, 0x1B, 0x00]))
    with pytest.raises(ValueError, match="not in SNP-major order"):
        PlinkReader(base)
    base = _write_bed(tmp_path, header=b"abc")
    with pytest.raises(ValueError, match="is not a PLINK BED file"):
        PlinkReader(base)


def test_iteration_restarts_and_closes_its_files(tmp_path):
    base = _write_bed(tmp_path)
    r = PlinkReader(base, norm_genotype=False)
    with pytest.raises(ValueError, match="call iter_snps"):
        next(r)
    it = r.iter_snps()
    next(it)
    first = (r.fhandle, r.snp_file_handle)
    r.iter_snps()  # restart: the previous handles are closed, not leaked
    assert first[0].closed and first[1].closed
    assert len(list(r)) == 3
    assert r.fhandle is None and r.snp_file_handle is None  # closed at the end
    with pytest.raises(StopIteration):
        next(r)  # and it stays exhausted
    with PlinkReader(base) as r2:
        next(r2.iter_snps())
        fh = r2.fhandle
    assert fh.closed and r2.fhandle is None
    r2.close()  # idempotent


def test_tped_half_missing_pair_is_missing(tmp_path):
    """PLINK treats a half-missing call (one allele 0) as missing; pylmm read it as 0.5."""
    base = _write_tped(tmp_path, ["0 1 2 0 1 2"])
    r = PlinkReader(base, type="t", norm_genotype=False)
    ((g, _),) = list(r)
    assert np.isnan(g[0]) and np.isnan(g[1]) and g[2] == 0.5
