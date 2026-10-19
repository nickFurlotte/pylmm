"""--kfile validation (bug 16, fixed in Phase 5.1): pylmmGWAS's
readKinshipFile and the CLI's usage error."""

import contextlib
import gzip
import io
import warnings

import helpers as h
import numpy as np
import pytest

from pylmm.cli import gwas as cli_gwas


def K3():
    return np.array([[1.0, 0.25, 0.1], [0.25, 1.0, -0.05], [0.1, -0.05, 1.0]])


def write(tmp_path, text, name="k.txt"):
    fn = str(tmp_path / name)
    with open(fn, "w") as f:
        f.write(text)
    return fn


def test_valid_plain_and_gz(tmp_path):
    fn = str(tmp_path / "k.txt")
    np.savetxt(fn, K3(), fmt="%.17g")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        K = cli_gwas.readKinshipFile(fn, 3)
    np.testing.assert_array_equal(K, K3())
    with open(fn, "rb") as src, gzip.open(fn + ".gz", "wb") as dst:
        dst.write(src.read())
    np.testing.assert_array_equal(cli_gwas.readKinshipFile(fn + ".gz", 3), K3())


def test_valid_whitespace_variants(tmp_path):
    """Tabs, repeated spaces, blank lines and a missing final newline are fine."""
    fn = write(tmp_path, "\n1\t0.25  0.1\n\n 0.25 1 -0.05\n0.1   -0.05\t1")
    np.testing.assert_array_equal(cli_gwas.readKinshipFile(fn, 3), K3())


def test_hmdp_kinship_reads_identically_to_np_loadtxt(tmp_path):
    fn = h.write_headerless_kinship(tmp_path)
    exp = np.loadtxt(h.HMDP_K_TXT)
    np.testing.assert_array_equal(cli_gwas.readKinshipFile(fn, exp.shape[0]), exp)


@pytest.mark.parametrize(
    "text", ["# a header line\n1 0\n0 1\n", "id1 id2\n1 0\n0 1\n"], ids=["hash", "names"]
)
def test_header_rejected(tmp_path, text):
    fn = write(tmp_path, text)
    with pytest.raises(ValueError) as e:
        cli_gwas.readKinshipFile(fn, 2)
    msg = str(e.value)
    assert fn in msg and "line 1" in msg and "not a number" in msg and "2 rows of 2" in msg


@pytest.mark.parametrize("m,n", [(3, 4), (4, 3), (95, 100), (105, 100)])
def test_wrong_size_rejected(tmp_path, m, n):
    fn = str(tmp_path / "k.txt")
    np.savetxt(fn, np.eye(m))
    with pytest.raises(ValueError) as e:
        cli_gwas.readKinshipFile(fn, n)
    msg = str(e.value)
    assert fn in msg
    assert "expected a %d x %d matrix (%d values" % (n, n, n * n) in msg
    assert "found %d values in %d rows (row lengths: %d)" % (m * m, m, m) in msg


def test_non_square_rejected(tmp_path):
    """The right number of values (n * n) but not n rows of n."""
    fn = write(tmp_path, "1 0 0 1\n0 0 0 0\n1 0 0 1\n0 0 0 0\n")  # 16 values, 4 x 4 for n = 4
    assert cli_gwas.readKinshipFile(fn, 4).shape == (4, 4)
    fn = write(tmp_path, "1 0 0 0 0 1 0 0\n0 0 1 0 0 0 0 1\n")  # 16 values in 2 x 8
    with pytest.raises(ValueError, match=r"found 16 values in 2 rows \(row lengths: 8\)"):
        cli_gwas.readKinshipFile(fn, 4)
    fn = write(tmp_path, "1 0\n0 1 0\n0 0 1\n")  # ragged
    with pytest.raises(ValueError, match=r"found 8 values in 3 rows \(row lengths: 2, 3\)"):
        cli_gwas.readKinshipFile(fn, 3)


@pytest.mark.parametrize("tok", ["nan", "inf", "-inf", "NaN"])
def test_non_finite_rejected(tmp_path, tok):
    fn = write(tmp_path, "1 0 0\n0 1 %s\n0 0 1\n" % tok)
    with pytest.raises(
        ValueError, match=r"1 non-finite values \(nan/inf\), the first at row 2, column 3"
    ):
        cli_gwas.readKinshipFile(fn, 3)


def test_empty_rejected(tmp_path):
    fn = write(tmp_path, "")
    with pytest.raises(ValueError, match="found 0 values in 0 rows"):
        cli_gwas.readKinshipFile(fn, 2)


def test_asymmetric_warns_but_loads(tmp_path):
    K = K3()
    K[0, 2] += 0.01
    fn = str(tmp_path / "k.txt")
    np.savetxt(fn, K, fmt="%.17g")
    with pytest.warns(UserWarning, match="not symmetric"):
        got = cli_gwas.readKinshipFile(fn, 3)
    np.testing.assert_array_equal(got, K)


def test_cli_wrong_size_kfile_is_a_usage_error(tmp_path):
    """The CLI turns the ValueError into optparse's usage error (exit 2)
    naming the file and the expected size."""
    emma = h.write_emma_subset(tmp_path, 5)
    n = len(np.loadtxt(h.HMDP_Y))
    fn = str(tmp_path / "k.txt")
    np.savetxt(fn, np.eye(n - 1))
    err = io.StringIO()
    with pytest.raises(SystemExit) as e, contextlib.redirect_stderr(err):
        cli_gwas.main(
            ["--emmaSNP", emma, "--emmaPHENO", h.HMDP_Y, "--kfile", fn, str(tmp_path / "out")]
        )
    assert e.value.code == 2
    out = err.getvalue()
    assert "usage:" in out and fn in out and "expected a %d x %d matrix" % (n, n) in out


def test_missing_file_rejected(tmp_path):
    fn = str(tmp_path / "nope.txt")
    with pytest.raises(ValueError, match="nope.txt"):
        cli_gwas.readKinshipFile(fn, 3)


def test_binary_file_rejected(tmp_path):
    fn = str(tmp_path / "k.bin")
    with open(fn, "wb") as f:
        f.write(bytes(range(128, 256)) * 4)
    with pytest.raises(ValueError, match="k.bin"):
        cli_gwas.readKinshipFile(fn, 3)


def test_plain_file_named_gz_rejected(tmp_path):
    fn = write(tmp_path, "1 0 0\n0 1 0\n0 0 1\n", name="k.txt.gz")
    with pytest.raises(ValueError, match="k.txt.gz"):
        cli_gwas.readKinshipFile(fn, 3)
