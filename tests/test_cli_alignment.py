"""`pylmm kinship`'s kinship ID file and `pylmm gwas`'s alignment by ID (Phase 5b)."""

import contextlib
import io

import numpy as np
import pytest

from pylmm import cli, samples

N, M = 30, 12


def _write_bed(base, G):
    """BED/BIM/FAM for G (fraction of A2 alleles, nan missing), independent of pylmm."""
    n, m = G.shape
    code = np.where(np.isnan(G), 1, np.where(G == 0, 0, np.where(G == 0.5, 2, 3))).astype(np.uint8)
    nb = (n + 3) // 4
    with open(base + ".bed", "wb") as f:
        f.write(bytes([0x6C, 0x1B, 0x01]))
        for j in range(m):
            c = np.zeros(nb * 4, dtype=np.uint8)
            c[:n] = code[:, j]
            c = c.reshape(nb, 4)
            f.write((c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).tobytes())
    with open(base + ".bim", "w") as f:
        for j in range(m):
            f.write("1\trs%d\t0\t%d\tA\tG\n" % (j, 1000 * (j + 1)))
    with open(base + ".fam", "w") as f:
        for i in range(n):
            f.write("F%d I%d 0 0 0 -9\n" % (i, i))


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


@pytest.fixture
def toy(tmp_path):
    rng = np.random.default_rng(7)
    G = rng.binomial(2, 0.4, size=(N, M)) / 2.0  # no missing calls
    base = str(tmp_path / "toy")
    _write_bed(base, G)
    y = rng.standard_normal(N)
    with open(base + ".phenos", "w") as f:
        for i in range(N):
            f.write("F%d I%d %.10f\n" % (i, i, y[i]))
    kin = str(tmp_path / "toy.kin")
    _err(["kinship", "--bfile", base, "-o", kin])
    return {"base": base, "kin": kin, "dir": tmp_path}


def _gwas(toy, kin, name, extra=()):
    out = str(toy["dir"] / name)
    err = _err(["gwas", "--bfile", toy["base"], "--kinship", kin, "-o", out, *extra])
    return _out(out), err


def test_pylmm_kinship_writes_the_id_file(toy):
    ids = samples.read_id_file(toy["kin"] + ".id")
    assert ids == [("F%d" % i, "I%d" % i) for i in range(N)]


def test_no_id_file_warns_and_assumes_genotype_order(toy):
    (ids, a), err = _gwas(toy, toy["kin"], "a.out")
    assert "warning" not in err
    plain = str(toy["dir"] / "plain.kin")
    np.savetxt(plain, np.loadtxt(toy["kin"]))
    (_, b), err = _gwas(toy, plain, "b.out")
    assert err.count("pylmm: warning: no kinship ID file %s.id; assuming" % plain) == 1
    np.testing.assert_array_equal(a, b)


def test_a_reordered_kinship_with_ids_gives_the_same_result(toy):
    """The row-order trap: a permuted K is realigned by its IDs."""
    (ids, a), _ = _gwas(toy, toy["kin"], "a.out")
    K = np.loadtxt(toy["kin"])
    o = np.random.default_rng(0).permutation(N)
    perm = str(toy["dir"] / "perm.kin")
    np.savetxt(perm, K[np.ix_(o, o)])
    samples.write_id_file(perm + ".id", [("F%d" % i, "I%d" % i) for i in o])
    (ids_b, b), err = _gwas(toy, perm, "b.out")
    assert "warning" not in err and ids_b == ids
    np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-12)


def test_kinship_missing_an_individual_drops_it_everywhere(toy):
    """The individual absent from K is dropped from the phenotype AND the SNP
    vectors: the result equals a dataset without that individual."""
    K = np.loadtxt(toy["kin"])
    keep = [i for i in range(N) if i != 4]
    sub = str(toy["dir"] / "sub.kin")
    k_sub = np.eye(N)  # the 29 kept individuals, plus a stranger in the last row
    k_sub[: N - 1, : N - 1] = K[np.ix_(keep, keep)]
    np.savetxt(sub, k_sub)
    samples.write_id_file(sub + ".id", [("F%d" % i, "I%d" % i) for i in keep] + [("FX", "IX")])
    (_, got), err = _gwas(toy, sub, "sub.out")
    assert (
        "pylmm: warning: 1 of 30 samples of the genotype file are not in the kinship and are "
        "dropped: F4 I4" in err
    )
    assert (
        "1 of 30 samples of the kinship are not in the genotype file and are ignored: FX IX" in err
    )

    # the same data written without individual 4
    from pylmm.plink_io import PlinkReader

    with (
        contextlib.redirect_stderr(io.StringIO()),
        PlinkReader(toy["base"], norm_genotype=False) as r,
    ):
        G = np.array([g for g, _ in r]).T
    base2 = str(toy["dir"] / "toy2")
    _write_bed(base2, G[keep])
    with open(base2 + ".fam", "w") as f:
        f.write("".join("F%d I%d 0 0 0 -9\n" % (i, i) for i in keep))
    kin2 = str(toy["dir"] / "toy2.kin")
    np.savetxt(kin2, K[np.ix_(keep, keep)])
    samples.write_id_file(kin2 + ".id", [("F%d" % i, "I%d" % i) for i in keep])
    out2 = str(toy["dir"] / "toy2.out")
    _err(
        [
            "gwas",
            "--bfile",
            base2,
            "--kinship",
            kin2,
            "--pheno",
            toy["base"] + ".phenos",
            "-o",
            out2,
        ]
    )
    _, exp = _out(out2)
    np.testing.assert_allclose(got, exp, rtol=1e-9, atol=1e-12)


def test_kinship_ids_must_match_the_matrix(toy):
    bad = str(toy["dir"] / "bad.kin")
    np.savetxt(bad, np.loadtxt(toy["kin"]))
    samples.write_id_file(bad + ".id", [("F0", "I0")])
    argv = ["gwas", "--bfile", toy["base"], "--kinship", bad, "-o", str(toy["dir"] / "x.out")]
    _err(argv, rc=2)
    samples.write_id_file(bad + ".id", [("F0", "I0")] * N)
    assert "lists sample F0 I0 twice" in _err(argv, rc=2)


# ---------------------------------------------------------------------------
# Phase 5b.2: --id-match, --pheno-name / --pheno-col, --covar-name
# ---------------------------------------------------------------------------


def test_id_match_iid_matches_the_kinship_on_the_iid(toy):
    """A kinship whose FIDs differ from the genotype file's: no sample in
    common on (FID, IID), the same result as the default run on the IID."""
    (ids, a), _ = _gwas(toy, toy["kin"], "a.out")
    other = str(toy["dir"] / "other.kin")
    np.savetxt(other, np.loadtxt(toy["kin"]))
    samples.write_id_file(other + ".id", [("G%d" % i, "I%d" % i) for i in range(N)])
    argv = ["gwas", "--bfile", toy["base"], "--kinship", other, "-o", str(toy["dir"] / "x")]
    _err(argv, rc=2)
    (ids_b, b), _ = _gwas(toy, other, "b.out", extra=["--id-match", "iid"])
    assert ids_b == ids
    np.testing.assert_array_equal(b, a)


def test_id_match_iid_needs_plink_genotypes(toy):
    err = _err(["gwas", "--emma-geno", "x", "--kinship", "k", "--id-match", "iid", "-o", "o"], rc=2)
    assert "--id-match iid needs PLINK genotypes" in err


def _write_table(fn, header, cols):
    with open(fn, "w") as f:
        if header:
            f.write("FID IID " + " ".join(header) + "\n")
        for i in range(N):
            f.write("F%d I%d %s\n" % (i, i, " ".join("%.10f" % c[i] for c in cols)))
    return fn


def test_pheno_name_and_col_select_the_column(toy):
    (_, a), _ = _gwas(toy, toy["kin"], "a.out")
    y = np.loadtxt(toy["base"] + ".phenos", usecols=2)
    noise = np.random.default_rng(1).standard_normal(N)
    named = _write_table(str(toy["dir"] / "named.ph"), ["noise", "y"], [noise, y])
    # Equal up to round-off: a column of a 2-column table is a strided view,
    # which can change the BLAS summation order (seen with numpy 1.22).
    (_, b), _ = _gwas(toy, toy["kin"], "b.out", extra=["--pheno", named, "--pheno-name", "y"])
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-14)
    (_, c), _ = _gwas(toy, toy["kin"], "c.out", extra=["--pheno", named, "--pheno-col", "2"])
    np.testing.assert_array_equal(c, b)
    (_, d), _ = _gwas(toy, toy["kin"], "d.out", extra=["--pheno", named])  # first column
    assert not np.array_equal(d, a)
    (_, e), _ = _gwas(toy, toy["kin"], "e.out", extra=["--pheno", named, "--pheno-col", "1"])
    np.testing.assert_array_equal(e, d)  # 1-based: column 1 is the first, the default

    base = ["gwas", "--bfile", toy["base"], "--kinship", toy["kin"], "-o", str(toy["dir"] / "x")]
    err = _err([*base, "--pheno", named, "--pheno-name", "bmi"], rc=2)
    assert "--pheno-name bmi: no such column" in err and "noise, y" in err
    plain = _write_table(str(toy["dir"] / "plain.ph"), None, [y])
    assert "has no FID IID name" in _err([*base, "--pheno", plain, "--pheno-name", "y"], rc=2)
    assert "--pheno-col 2: the phenotype file has 1 phenotype column(s) (1-based" in _err(
        [*base, "--pheno", plain, "--pheno-col", "2"], rc=2
    )
    assert "--pheno-col 3: the phenotype file has 2" in _err(
        [*base, "--pheno", named, "--pheno-col", "3"], rc=2
    )
    # one of --pheno-name and --pheno-col (also when the column is the default 1)
    _err([*base, "--pheno", named, "--pheno-name", "y", "--pheno-col", "1"], rc=2)


def test_covar_name_selects_columns(toy):
    rng = np.random.default_rng(2)
    c1, c2 = rng.standard_normal(N), rng.standard_normal(N)
    both = _write_table(str(toy["dir"] / "both.cov"), ["c1", "c2"], [c1, c2])
    only2 = _write_table(str(toy["dir"] / "only2.cov"), ["c2"], [c2])
    (_, a), _ = _gwas(toy, toy["kin"], "a.out", extra=["--covar", only2])
    (_, b), _ = _gwas(toy, toy["kin"], "b.out", extra=["--covar", both, "--covar-name", "c2"])
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-14)
    (_, c), _ = _gwas(toy, toy["kin"], "c.out", extra=["--covar", both])
    for names in (["c1", "c2"], ["c1,c2"]):
        (_, d), _ = _gwas(toy, toy["kin"], "d.out", extra=["--covar", both, "--covar-name", *names])
        np.testing.assert_array_equal(d, c)
    base = ["gwas", "--bfile", toy["base"], "--kinship", toy["kin"], "-o", str(toy["dir"] / "x")]
    _err([*base, "--covar", both, "--covar-name", "c3"], rc=2)
    assert "--covar-name needs --covar" in _err([*base, "--covar-name", "c1"], rc=2)
