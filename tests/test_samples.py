"""pylmm.samples: ID matching, alignment reports, sample tables and ID files (Phase 5b)."""

import numpy as np
import pytest

from pylmm import samples

GENO = [("F1", "I1"), ("F1", "I2"), ("F2", "I3"), ("F3", "I4")]


def test_match_index_by_fid_iid_and_by_iid():
    src = [("0", "I3"), ("F1", "I1"), ("FX", "IX")]
    assert samples.match_index(GENO, src).tolist() == [1, -1, -1, -1]
    assert samples.match_index(GENO, src, match="iid").tolist() == [1, -1, 0, -1]


@pytest.mark.parametrize("match", ["fid_iid", "iid"])
def test_duplicates_are_an_error(match):
    dup = [("F1", "I1"), ("F1", "I2"), ("F1", "I1")]
    with pytest.raises(ValueError, match=r"the kinship lists .*I1 twice \(rows 1 and 3\)"):
        samples.align_samples({"genotypes": GENO, "kinship": dup}, match=match)
    # in the reference source too
    with pytest.raises(ValueError, match="twice"):
        samples.align_samples({"genotypes": dup, "kinship": GENO}, match=match)


def test_same_iid_in_two_families_is_only_a_duplicate_when_matching_on_iid():
    ids = [("F1", "A"), ("F2", "A")]
    al = samples.align_samples({"g": ids, "k": ids[::-1]})
    assert al.rows["k"].tolist() == [1, 0]
    with pytest.raises(ValueError, match=r"IID A twice.*match on FID and IID instead"):
        samples.align_samples({"g": ids, "k": ids}, match="iid")


def test_alignment_keeps_reference_order_and_reports_each_source():
    kin = [("F3", "I4"), ("F1", "I1"), ("F2", "I3"), ("F9", "I9")]
    phe = [("F2", "I3"), ("F1", "I1"), ("F3", "I4"), ("F1", "I2")]
    al = samples.align_samples({"genotypes": GENO, "kinship": kin, "phenotypes": phe})
    assert al.ids == [("F1", "I1"), ("F2", "I3"), ("F3", "I4")]
    assert al.n == 3 and al.reference == "genotypes"
    for name, src in (("genotypes", GENO), ("kinship", kin), ("phenotypes", phe)):
        assert [src[i] for i in al.rows[name]] == al.ids
    assert [(d.source, d.reason, d.action, d.ids) for d in al.dropped] == [
        ("genotypes", "not in the kinship", "dropped", (("F1", "I2"),)),
        ("kinship", "not in the genotypes", "ignored", (("F9", "I9"),)),
    ]
    assert al.report() == [
        "1 of 4 samples of the genotypes are not in the kinship and are dropped: F1 I2",
        "1 of 4 samples of the kinship are not in the genotypes and are ignored: F9 I9",
    ]
    K = np.arange(16.0).reshape(4, 4)
    np.testing.assert_array_equal(al.take_square("kinship", K), K[np.ix_([1, 2, 0], [1, 2, 0])])
    np.testing.assert_array_equal(al.take("phenotypes", np.arange(4)), [1, 0, 2])
    assert not al.is_identity("genotypes") and not al.is_identity("kinship")


def test_identity_alignment():
    al = samples.align_samples({"genotypes": GENO, "kinship": list(GENO)})
    assert al.ids == GENO and al.dropped == [] and al.report() == []
    assert al.is_identity("genotypes") and al.is_identity("kinship")


def test_other_reference_and_errors():
    al = samples.align_samples({"g": GENO, "k": GENO[::-1]}, reference="k")
    assert al.ids == GENO[::-1]
    with pytest.raises(ValueError, match="reference 'x'"):
        samples.align_samples({"g": GENO}, reference="x")
    with pytest.raises(ValueError, match="match='fid'"):
        samples.align_samples({"g": GENO}, match="fid")
    with pytest.raises(ValueError, match=r"no sample is in all of g, k .*g: F1 I1; k: X Y"):
        samples.align_samples({"g": GENO, "k": [("X", "Y")]})


def test_long_drop_lists_are_abbreviated():
    many = [("F", "I%d" % i) for i in range(10)]
    al = samples.align_samples({"g": many, "k": many[:2]})
    (line,) = al.report()
    assert line.startswith("8 of 10 samples of the g are not in the k and are dropped: F I2, ")
    assert line.endswith("F I6, ...")


def test_id_files_round_trip(tmp_path):
    fn = str(tmp_path / "k.kin.id")
    samples.write_id_file(fn, GENO)
    assert open(fn).read().splitlines()[0] == "F1\tI1"
    assert samples.read_id_file(fn) == GENO
    assert samples.read_kinship_ids(str(tmp_path / "k.kin")) == GENO
    assert samples.read_kinship_ids(str(tmp_path / "other.kin")) is None
    (tmp_path / "bad.id").write_text("F1 I1\nF2\n")
    with pytest.raises(ValueError, match="line 2: expected FID and IID"):
        samples.read_id_file(str(tmp_path / "bad.id"))
    (tmp_path / "empty.id").write_text("\n")
    with pytest.raises(ValueError, match="lists no samples"):
        samples.read_id_file(str(tmp_path / "empty.id"))


def test_read_sample_table_with_header_selects_by_name_or_index(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("FID IID height bmi\nF1 I1 1.5 NA\nF1 I2 1.7 22\n")
    t = samples.read_sample_table(str(fn))
    assert t.ids == [("F1", "I1"), ("F1", "I2")] and t.names == ["height", "bmi"]
    assert t.values.shape == (2, 2) and np.isnan(t.values[0, 1])
    t = samples.read_sample_table(str(fn), columns=["bmi", "height"])
    assert t.names == ["bmi", "height"]
    np.testing.assert_array_equal(t.values[:, 1], [1.5, 1.7])
    t = samples.read_sample_table(str(fn), columns=[1])
    assert t.names == ["bmi"]
    with pytest.raises(ValueError, match="has no column 'weight'; its columns are height, bmi"):
        samples.read_sample_table(str(fn), columns=["weight"])
    with pytest.raises(ValueError, match="column index 2 is out of range"):
        samples.read_sample_table(str(fn), columns=[2])
    with pytest.raises(ValueError, match="selected twice"):
        samples.read_sample_table(str(fn), columns=["bmi", 1])


def test_read_sample_table_without_header(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("F1 I1 1.5 -9\nF1 I2 1.7 22\n")
    t = samples.read_sample_table(str(fn), is_missing=lambda x: x in ("NA", "-9"), columns=[1, 0])
    assert t.names is None and np.isnan(t.values[0, 0]) and t.values[1, 1] == 1.7
    with pytest.raises(ValueError, match="no FID IID header line, so column 'bmi' cannot"):
        samples.read_sample_table(str(fn), columns=["bmi"])


def test_read_sample_table_header_width_must_match(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("FID IID a b\nF1 I1 1.5\n")
    with pytest.raises(ValueError, match="the header names 2 columns but the rows have 1"):
        samples.read_sample_table(str(fn))
