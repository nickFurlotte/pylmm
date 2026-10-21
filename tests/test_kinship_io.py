"""GCTA binary GRMs, plink2 .rel files and eigendecomposition files with IDs (Phase 6.2c).

The plink2-written files in tests/data/plink2/ pin the layouts to an external
tool (see its README.md); the rest are round trips and CLI checks.
"""

import contextlib
import io
import os
import shutil

import helpers as h
import numpy as np
import pytest
from scipy import linalg
from synthetic_bed import write_synthetic_bed

from pylmm import cli, kinship_io, samples
from pylmm.kinship_io import read_grm, read_grm_counts, read_rel, write_grm
from pylmm.plink_io import PlinkReader
from pylmm.samples import InputError

P2 = os.path.join(os.path.dirname(__file__), "data", "plink2")
N_P2 = 50


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


# ---------------------------------------------------------------------------
# plink2-written files
# ---------------------------------------------------------------------------


def test_plink2_grm_layout():
    """plink2 --make-grm-bin == plink2 --make-rel triangle bin (float64), up to float32."""
    K, ids = read_grm(os.path.join(P2, "geno"))
    assert ids == [("F%d" % i, "I%d" % i) for i in range(1, N_P2 + 1)]
    ref, ids_ref = read_rel(os.path.join(P2, "geno_iid.rel.bin"))  # float64 triangle
    assert ids_ref == [("0", "I%d" % i) for i in range(1, N_P2 + 1)]  # '#IID' header: FID 0
    np.testing.assert_array_equal(K, K.T)
    np.testing.assert_allclose(K, ref, rtol=2.0**-24, atol=0)
    assert np.abs(K - ref).max() > 0  # float32: not exact
    # The row-major lower triangle, not the column-major one (= the upper triangle).
    tri = np.fromfile(os.path.join(P2, "geno.grm.bin"), dtype="<f4")
    wrong = np.zeros((N_P2, N_P2))
    wrong[np.triu_indices(N_P2)] = tri
    wrong = wrong + wrong.T - np.diag(wrong.diagonal())
    assert np.abs(wrong - ref).max() > 0.05


def test_plink2_rel_shapes_agree():
    """Text triangle, float32 square (bin4) and float64 triangle .rel files."""
    tri_text, ids = read_rel(os.path.join(P2, "geno_tri.rel"))
    sq4, ids4 = read_rel(os.path.join(P2, "geno_sq4.rel.bin"))
    ref, _ = read_rel(os.path.join(P2, "geno_iid.rel.bin"))
    assert ids == ids4 == [("F%d" % i, "I%d" % i) for i in range(1, N_P2 + 1)]
    np.testing.assert_allclose(tri_text, ref, rtol=1e-5, atol=1e-6)  # 6 significant digits
    np.testing.assert_array_equal(sq4, read_grm(os.path.join(P2, "geno"))[0])  # both float32


def test_plink2_grm_rewritten_byte_identical(tmp_path):
    """write_grm of what read_grm / read_grm_counts read gives plink2's bytes back."""
    K, ids = read_grm(os.path.join(P2, "geno"))
    N = read_grm_counts(os.path.join(P2, "geno"))
    assert N.max() <= 300 and N.min() > 250  # SNPs with both calls, per pair
    write_grm(str(tmp_path / "rt"), K, ids, N)
    for suffix in kinship_io.GRM_SUFFIXES:
        with (
            open(os.path.join(P2, "geno" + suffix), "rb") as a,
            open(str(tmp_path / ("rt" + suffix)), "rb") as b,
        ):
            assert a.read() == b.read(), suffix


# ---------------------------------------------------------------------------
# round trips and validation
# ---------------------------------------------------------------------------


def _random_k(n, seed=0):
    G = np.random.default_rng(seed).binomial(2, 0.3, size=(n, 3 * n)) / 2.0
    from pylmm.kinship import calculate_kinship

    return calculate_kinship(G)


def test_grm_round_trip_is_float32_rounding(tmp_path):
    K = _random_k(40)
    ids = [("F%d" % i, "I%d" % i) for i in range(40)]
    prefix = str(tmp_path / "k")
    write_grm(prefix + ".grm.bin", K, ids, n_snps=120)  # a .grm.bin suffix is dropped
    K2, ids2 = read_grm(prefix)
    assert ids2 == ids
    np.testing.assert_array_equal(K2, kinship_io.round_like_grm(K))
    rel = np.abs(K2 - K) / np.abs(K)
    assert 0 < rel.max() <= 2.0**-24
    np.testing.assert_array_equal(read_grm_counts(prefix), np.full((40, 40), 120.0))
    assert os.path.getsize(prefix + ".grm.bin") == 40 * 41 // 2 * 4


def test_grm_size_errors(tmp_path):
    K = _random_k(5)
    ids = [("F", "I%d" % i) for i in range(5)]
    prefix = str(tmp_path / "k")
    write_grm(prefix, K, ids, n_snps=15)
    data = open(prefix + ".grm.bin", "rb").read()
    open(prefix + ".grm.bin", "wb").write(data[:-4])
    with pytest.raises(InputError, match=r"has 56 bytes, but .*\.grm\.id lists 5 samples.* = 60"):
        read_grm(prefix)
    np.tril(K)[np.tril_indices(5)].astype("<f8").tofile(prefix + ".grm.bin")
    with pytest.raises(InputError, match="twice that: a float64 triangle"):
        read_grm(prefix)
    K.astype("<f4").tofile(prefix + ".grm.bin")
    with pytest.raises(InputError, match="square 5 x 5 matrix"):
        read_grm(prefix)
    bad = K[np.tril_indices(5)].astype("<f4")
    bad[3] = np.nan
    bad.tofile(prefix + ".grm.bin")
    with pytest.raises(InputError, match="1 non-finite"):
        read_grm(prefix)
    os.remove(prefix + ".grm.id")
    with pytest.raises(InputError, match=r"no file .*k\.grm\.id"):
        read_grm(prefix)


def test_write_grm_argument_errors(tmp_path):
    K = _random_k(4)
    ids = [("F", "I%d" % i) for i in range(4)]
    prefix = str(tmp_path / "k")
    with pytest.raises(ValueError, match="shape"):
        write_grm(prefix, K, ids[:3], 1)
    A = K.copy()
    A[0, 1] += 0.1
    with pytest.raises(ValueError, match="not symmetric"):
        write_grm(prefix, A, ids, 1)
    with pytest.raises(ValueError, match="without whitespace"):
        write_grm(prefix, K, [("F", "I 0"), *ids[1:]], 1)
    with pytest.raises(ValueError, match="n_snps has shape"):
        write_grm(prefix, K, ids, np.ones(4))
    assert not os.listdir(tmp_path)


def test_rel_errors(tmp_path):
    fn = str(tmp_path / "x.rel")
    with open(fn, "w") as f:
        f.write("1\n0.5 1\n0.1 0.2\n")  # 5 values: neither 3 x 3 nor its triangle
    with open(str(tmp_path / "x.rel.id"), "w") as f:
        f.write("#FID\tIID\nA a\nA b\nA c\n")
    with pytest.raises(InputError, match=r"has 5 values.*expected 9 \(square\) or 6"):
        read_rel(fn)
    np.ones(7).tofile(str(tmp_path / "x.rel.bin"))
    with pytest.raises(InputError, match="has 56 bytes"):
        read_rel(str(tmp_path / "x.rel.bin"))
    with open(str(tmp_path / "x.rel.id"), "w") as f:
        f.write("#FID\tSID\nA a\n")
    with pytest.raises(InputError, match="no IID column"):
        read_rel(fn)


def test_rel_square0_is_symmetrised(tmp_path):
    """plink2 'square0': the upper triangle is written as zeros."""
    K = np.array([[1.0, 0.0, 0.0], [0.25, 1.0, 0.0], [0.1, -0.05, 1.0]])
    np.savetxt(str(tmp_path / "x.rel"), K, delimiter="\t")
    with open(str(tmp_path / "x.rel.id"), "w") as f:
        f.write("#IID\na\nb\nc\n")
    got, _ = read_rel(str(tmp_path / "x.rel"))
    np.testing.assert_array_equal(got, np.tril(K) + np.tril(K, -1).T)


def test_kinship_format():
    assert kinship_io.kinship_format("x.grm.bin") == "grm"
    assert kinship_io.kinship_format("x.rel.gz") == "rel"
    assert kinship_io.kinship_format("x.kin") == "text"
    assert kinship_io.kinship_format("x.kin.gz") == "text"
    assert kinship_io.grm_prefix("a/b.grm.id") == "a/b"


def test_plink_reader_reads_a_grm_and_aligns_it(tmp_path):
    """PlinkReader.read_kinship: a .grm.bin is read with its IDs and aligned."""
    base = os.path.join(P2, "geno")
    K, ids = read_grm(base)
    o = np.random.default_rng(1).permutation(N_P2)[:45]
    write_grm(str(tmp_path / "perm"), K[np.ix_(o, o)], [ids[i] for i in o], 300)
    with pytest.warns(UserWarning, match="5 of 50 samples of the genotype file are not in"):
        r = PlinkReader(base, k_file=str(tmp_path / "perm.grm.bin"))
    keep = sorted(o)
    assert r.indivs == [ids[i] for i in keep]
    np.testing.assert_array_equal(r.K, K[np.ix_(keep, keep)])


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


@pytest.fixture
def study(tmp_path):
    """A synthetic BED with a heritable phenotype, its text kinship and its GRM."""
    n, m = 200, 1000
    base = str(tmp_path / "g")
    G = write_synthetic_bed(base, n, m, seed=1)
    rng = np.random.default_rng(101)
    G = np.where(np.isnan(G), np.nanmean(G, 0), G)
    Z = (G - G.mean(0)) / G.std(0)
    y = Z @ (rng.standard_normal(m) * np.sqrt(0.8 / m)) + rng.standard_normal(n) * np.sqrt(0.2)
    with open(base + ".phenos", "w") as f:
        for i in range(n):
            f.write("F%d I%d %.12f\n" % (i + 1, i + 1, y[i]))
    _err(["kinship", "--bfile", base, "-o", base + ".kin", "--eigen-npz", base + ".eig.npz"])
    _err(["kinship", "--bfile", base, "--out-format", "grm", "-o", base])
    return {"base": base, "n": n, "m": m, "dir": tmp_path}


def _gwas(study, kin_args, name, extra=(), rc=0):
    out = str(study["dir"] / name)
    err = _err(["gwas", "--bfile", study["base"], *kin_args, "-o", out, *extra], rc=rc)
    return (_out(out) if rc == 0 else None), err


def test_cli_kinship_writes_the_grm(study):
    base = study["base"]
    k_text = np.loadtxt(base + ".kin")
    K, ids = read_grm(base)
    assert ids == samples.read_id_file(base + ".kin.id")
    np.testing.assert_array_equal(K, kinship_io.round_like_grm(k_text))
    # m: SNPs used (none constant or all-missing here)
    np.testing.assert_array_equal(read_grm_counts(base), np.full(K.shape, float(study["m"])))
    assert not os.path.exists(base + ".id")  # the text sidecar is not written


def test_cli_gwas_grm_matches_text_kinship_within_float32(study):
    (snps_t, a), _ = _gwas(study, ["--kinship", study["base"] + ".kin"], "text.out")
    (snps_g, b), err = _gwas(study, ["--grm", study["base"]], "grm.out")
    assert "warning" not in err and snps_g == snps_t
    np.testing.assert_allclose(b[:, 2], a[:, 2], rtol=0, atol=h.GRM_FLOAT32_ATOL_T)
    np.testing.assert_allclose(b[:, 3], a[:, 3], rtol=h.GRM_FLOAT32_RTOL_P, atol=0)
    assert np.abs(b - a).max() > 0  # float32 K: not the same numbers
    # The text kinship of the float32-rounded K gives exactly the --grm scan.
    rounded = str(study["dir"] / "rounded.kin")
    np.savetxt(rounded, read_grm(study["base"])[0])
    shutil.copy(study["base"] + ".kin.id", rounded + ".id")
    (_, c), _ = _gwas(study, ["--kinship", rounded], "rounded.out")
    np.testing.assert_array_equal(c, b)


def test_cli_gwas_grm_reordered_and_suffix(study):
    """A permuted GRM is aligned by ID; --grm PREFIX.grm.bin works too."""
    K, ids = read_grm(study["base"])
    o = np.random.default_rng(0).permutation(study["n"])
    perm = str(study["dir"] / "perm")
    write_grm(perm, K[np.ix_(o, o)], [ids[i] for i in o], study["m"])
    (_, a), _ = _gwas(study, ["--grm", study["base"]], "a.out")
    (_, b), err = _gwas(study, ["--grm", perm + ".grm.bin"], "b.out")
    assert "warning" not in err
    np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-12)


def test_cli_gwas_kinship_reads_plink2_rel(tmp_path):
    """--kinship FILE.rel.bin / FILE.rel and --grm on plink2's files."""
    for f in os.listdir(P2):
        shutil.copy(os.path.join(P2, f), str(tmp_path / f))
    base = str(tmp_path / "geno")
    y = np.random.default_rng(3).standard_normal(N_P2)
    with open(base + ".phenos", "w") as f:
        for i in range(N_P2):
            f.write("F%d I%d %.10f\n" % (i + 1, i + 1, y[i]))
    outs = {}
    for name, kin in [
        ("grm", ["--grm", base]),
        ("sq4", ["--kinship", base + "_sq4.rel.bin"]),
        ("tri", ["--kinship", base + "_tri.rel"]),
    ]:
        out = str(tmp_path / (name + ".out"))
        err = _err(["gwas", "--bfile", base, *kin, "-o", out])
        # plink2's missingness-corrected GRM is not PSD here (one eigenvalue -0.02): that
        # warning is expected; no ID warning is.
        assert err.count("warning") == err.count("not positive semi-definite") == 1, err
        outs[name] = _out(out)[1]
    np.testing.assert_array_equal(outs["sq4"], outs["grm"])  # the same float32 values
    np.testing.assert_allclose(outs["tri"][:, 3], outs["grm"][:, 3], rtol=1e-3)  # 6 digits


def test_cli_grm_errors(study, tmp_path):
    _, err = _gwas(study, ["--grm", str(tmp_path / "nope")], "x.out", rc=2)
    nope = str(tmp_path / "nope")
    assert "pylmm: error: --grm %s: no such file %s.grm.bin" % (nope, nope) in err
    _, err = _gwas(study, ["--grm", study["base"], "--kinship", study["base"] + ".kin"], "x", rc=2)
    assert "not allowed with argument" in err
    emma = h.write_emma_subset(tmp_path)
    err = _err(["kinship", "--emma-geno", emma, "--out-format", "grm", "-o", "x"], rc=2)
    assert "--out-format grm needs sample IDs" in err


# ---------------------------------------------------------------------------
# eigendecompositions with IDs
# ---------------------------------------------------------------------------


def test_eigen_npz_has_ids_and_kind(study):
    e = kinship_io.read_eigen(study["base"] + ".eig.npz")
    assert e.ids == samples.read_id_file(study["base"] + ".kin.id")
    assert e.kinship_type == "standardized"
    with np.load(study["base"] + ".eig.npz") as z:  # no pickle needed
        assert sorted(z.files) == ["Kva", "Kve", "fid", "iid", "kinship_type"]


def test_eigen_used_after_the_kinship_is_reordered(study):
    """A permuted kinship with IDs: the ID-carrying eigen file is reordered, not ignored."""
    K = np.loadtxt(study["base"] + ".kin")
    ids = samples.read_id_file(study["base"] + ".kin.id")
    o = np.random.default_rng(5).permutation(study["n"])
    perm = str(study["dir"] / "perm.kin")
    np.savetxt(perm, K[np.ix_(o, o)])
    samples.write_id_file(perm + ".id", [ids[i] for i in o])
    kva, kve = linalg.eigh(K[np.ix_(o, o)])
    kinship_io.write_eigen(str(study["dir"] / "perm.npz"), kva, kve, ids=[ids[i] for i in o])
    (_, ref), _ = _gwas(study, ["--kinship", study["base"] + ".kin"], "ref.out")
    (_, a), err = _gwas(
        study, ["--kinship", perm, "--eigen", str(study["dir"] / "perm.npz"), "-v"], "a.out"
    )
    assert "ignored" not in err and "Reordering the --eigen eigenvectors" in err
    np.testing.assert_allclose(a, ref, rtol=1e-8, atol=1e-10)


def test_eigen_ids_of_other_samples_rejected(study):
    e = kinship_io.read_eigen(study["base"] + ".eig.npz")
    other = [("X", iid) for _, iid in e.ids]
    bad = str(study["dir"] / "other.npz")
    kinship_io.write_eigen(bad, e.kva, e.kve, ids=other)
    _, err = _gwas(study, ["--kinship", study["base"] + ".kin", "--eigen", bad], "x", rc=2)
    assert "is not the eigendecomposition of this kinship: 200 of the 200 tested" in err
    # matched on the IID alone, the same file is fine
    (_, a), err = _gwas(
        study,
        ["--kinship", study["base"] + ".kin", "--eigen", bad, "--id-match", "iid"],
        "iid.out",
    )
    assert "warning" not in err


def test_eigen_with_ids_ignored_when_samples_dropped(study):
    """An individual without a phenotype: the decomposition is recomputed (as before)."""
    pheno = str(study["dir"] / "na.ph")
    lines = open(study["base"] + ".phenos").read().splitlines()
    lines[7] = " ".join([*lines[7].split()[:2], "NA"])
    open(pheno, "w").write("\n".join(lines) + "\n")
    _, err = _gwas(
        study,
        ["--kinship", study["base"] + ".kin", "--eigen", study["base"] + ".eig.npz"],
        "x.out",
        extra=["--pheno", pheno],
    )
    assert "--eigen is ignored: 1 of its 200 individuals are not tested" in err


def test_old_eigen_without_ids_still_reads_with_a_warning(study):
    e = kinship_io.read_eigen(study["base"] + ".eig.npz")
    old = str(study["dir"] / "old.npz")
    np.savez(old, Kva=e.kva, Kve=e.kve)  # pylmm 1.0.0's file
    (_, ref), _ = _gwas(study, ["--kinship", study["base"] + ".kin"], "ref.out")
    (_, a), err = _gwas(study, ["--kinship", study["base"] + ".kin", "--eigen", old], "a.out")
    assert err.count("--eigen %s has no sample IDs; assuming" % old) == 1
    np.testing.assert_allclose(a, ref, rtol=1e-8, atol=1e-10)


def test_eigen_text_takes_ids_from_the_kinship_id_file(study):
    base = study["base"]
    _err(["kinship", "--bfile", base, "-o", base + ".kin", "--eigen-text"])
    e = kinship_io.read_eigen(base + ".kin")
    assert e.ids == samples.read_id_file(base + ".kin.id")
    _, err = _gwas(study, ["--kinship", base + ".kin", "--eigen", base + ".kin"], "x.out")
    assert "warning" not in err


def test_grm_eigen_is_that_of_the_float32_k(study):
    """`pylmm kinship --out-format grm --eigen-npz` decomposes the K the GRM holds."""
    base = study["base"]
    npz = str(study["dir"] / "grm.eig.npz")
    _err(["kinship", "--bfile", base, "--out-format", "grm", "-o", base, "--eigen-npz", npz])
    K, _ = read_grm(base)
    kva, _ = linalg.eigh(K)
    np.testing.assert_array_equal(kinship_io.read_eigen(npz).kva, kva)
    (_, a), _ = _gwas(study, ["--grm", base], "a.out")
    (_, b), err = _gwas(study, ["--grm", base, "--eigen", npz], "b.out")
    assert "warning" not in err
    np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-10)
