"""Input validation (Phase 5.2): the S5 robustness targets.

Each test names its target in sims/studies/s5_edge_cases.md ("P0-2" etc.)
and docs/phase4-simulations.md section 7.  Before Phase 5.2 these inputs
gave a finite t where none exists, a bare scipy/numpy exception, or no
warning at all.
"""

import contextlib
import io
import warnings

import numpy as np
import pytest

from pylmm.cli import gwas as cli_gwas
from pylmm.lmm import GWAS, LMM, calculateKinship, check_covariates, snp_estimable


def quiet():
    return contextlib.redirect_stderr(io.StringIO())


@pytest.fixture
def prob():
    """The S5 reproducer prelude: n = 40, K from 200 SNPs, 5 test SNPs."""
    rng = np.random.default_rng(0)
    n = 40
    K = calculateKinship(rng.binomial(2, 0.3, (n, 200)).astype(float))
    y = rng.standard_normal(n)
    X = rng.binomial(2, 0.4, (n, 5)).astype(float)
    return rng, n, K, y, X


# ---------------------------------------------------------------------------
# P0-2 / issue 19 / bug 13: SNPs that cannot be estimated give nan
# ---------------------------------------------------------------------------


def test_constant_dosage_snp_is_nan(prob):
    rng, n, K, y, X = prob
    X[:, 2] = 0.37  # var() is ~1e-33, not exactly 0
    t, p = GWAS(y, X, K)
    assert np.isnan(t[2]) and np.isnan(p[2])
    assert np.isfinite(np.delete(t, 2)).all()


def test_snp_equal_to_covariate_is_nan(prob):
    rng, n, K, y, X = prob
    c = rng.standard_normal(n)
    X[:, 1] = c
    X[:, 3] = 2 * c + 1  # in the span of [1, c]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no LinAlgWarning any more
        t, p = GWAS(y, X, K, X0=np.column_stack([np.ones(n), c]))
    assert np.isnan(t[1]) and np.isnan(t[3])
    assert np.isfinite([t[0], t[2], t[4]]).all()


def test_snp_constant_after_dropping_missing_is_nan(prob):
    rng, n, K, y, X = prob
    X[:, 0] = 1.0
    X[:3, 0] = np.nan
    X[3, 0] = 1.0 + 1e-14  # one rounding-level difference
    t, p = GWAS(y, X, K)
    assert np.isnan(t[0])


def test_snp_estimable_helper():
    n = 30
    one = np.ones((n, 1))
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n)
    assert snp_estimable(x, one)
    assert not snp_estimable(np.full(n, 0.37), one)
    assert not snp_estimable(np.zeros(n), one)
    assert not snp_estimable(x, np.column_stack([np.ones(n), 3 * x]))
    assert not snp_estimable(x[:2], one[:2])  # no residual df
    assert not snp_estimable(np.r_[np.nan, x[1:]], one)
    # without an intercept a constant SNP is estimable
    assert snp_estimable(np.full(n, 0.37), rng.standard_normal((n, 1)))
    assert snp_estimable(np.full(n, 0.37), np.zeros((n, 0)))


def _write_emma(tmp_path, rows, y, K):
    snps = str(tmp_path / "s.txt")
    with open(snps, "w") as f:
        for row in rows:
            f.write(" ".join("NA" if np.isnan(v) else "%.17g" % v for v in row) + "\n")
    ph = str(tmp_path / "p.txt")
    with open(ph, "w") as f:
        f.write(" ".join("NA" if np.isnan(v) else "%.10f" % v for v in y) + "\n")
    kf = str(tmp_path / "k")
    np.savetxt(kf, K)
    return snps, ph, kf


def _run(argv):
    with quiet():
        cli_gwas.main(argv)


def _read(fn):
    rows = [line.rstrip("\n").split("\t") for line in open(fn)][1:]
    return np.array([[float(x) for x in r[1:]] for r in rows])


@pytest.mark.parametrize("remove", [False, True], ids=["impute", "removeMissing"])
def test_cli_non_estimable_snps_are_nan_in_both_branches(tmp_path, prob, remove):
    """CLI: a SNP that varies only among individuals whose phenotype is NA,
    a constant dosage, and the same with one missing call, give nan in both
    CLI branches (they used to give t ~ 1e-8 in one and nan in the other)."""
    rng, n, K, y, X = prob
    y = y.copy()
    y[:4] = np.nan
    only_na = np.full(n, 1.0)
    only_na[:4] = [0.0, 2.0, 0.0, 2.0]  # varies only where y is NA
    const = np.full(n, 0.37)
    const_miss = const.copy()
    const_miss[10] = np.nan
    ok = X[:, 0]
    snps, ph, kf = _write_emma(tmp_path, [ok, only_na, const, const_miss], y, K)
    out = str(tmp_path / "o")
    _run(
        ["--emmaSNP", snps, "--emmaPHENO", ph, "--kfile", kf, out]
        + (["--removeMissingGenotypes"] if remove else [])
    )
    r = _read(out)
    assert np.isfinite(r[0]).all()
    assert np.isnan(r[1:]).all()


# ---------------------------------------------------------------------------
# P0-3 / issue 20: saturated designs raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_minus_n", [-1, 0, 1])
def test_saturated_design_raises(prob, q_minus_n):
    rng, n, K, y, X = prob
    X0 = np.column_stack([np.ones(n), rng.standard_normal((n, n + q_minus_n - 1))])
    assert n <= X0.shape[1] + 1
    with pytest.raises(ValueError, match="observations"):
        GWAS(y, X, K, X0=X0)
    with pytest.raises(ValueError, match="observations"):
        LMM(y, K, X0=X0)


def test_design_with_one_residual_df_is_fine(prob):
    rng, n, K, y, X = prob
    X0 = np.column_stack([np.ones(n), rng.standard_normal((n, n - 3))])  # q + 1 = n - 1
    with quiet(), np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWAS(y, X[:, :1], K, X0=X0)


def test_saturated_after_dropping_missing_y_raises(prob):
    rng, n, K, y, X = prob
    y = y.copy()
    y[5:] = np.nan  # 5 observations
    X0 = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    with pytest.raises(ValueError, match="n=5 observations"):
        GWAS(y, X, K, X0=X0)


# ---------------------------------------------------------------------------
# P0-5 / issue 21: K checks
# ---------------------------------------------------------------------------


def test_kinship_nan_raises(prob):
    rng, n, K, y, X = prob
    K = K.copy()
    K[3, 7] = K[7, 3] = np.nan
    with pytest.raises(
        ValueError, match="kinship matrix has 2 nan/inf values, the first at row 3, column 7"
    ):
        LMM(y, K)
    with pytest.raises(ValueError, match="kinship"):
        GWAS(y, X, K)


def test_kinship_wrong_shape_raises(prob):
    rng, n, K, y, X = prob
    with pytest.raises(ValueError, match="kinship matrix has shape"):
        LMM(y, K[:-1, :-1])


def test_kinship_asymmetric_warns(prob):
    rng, n, K, y, X = prob
    with quiet(), pytest.warns(UserWarning, match="not symmetric"):
        LMM(y, np.triu(K))


def test_kinship_non_psd_warns_with_most_negative_eigenvalue(prob):
    rng, n, K, y, X = prob
    E = rng.standard_normal((n, n)) * 0.3
    Kn = K + (E + E.T) / 2
    emin = np.linalg.eigvalsh(Kn)[0]
    assert emin < -0.1
    with quiet(), pytest.warns(UserWarning, match="not positive semi-definite") as rec:
        LMM(y, Kn)
    assert ("%g" % emin) in str(rec[0].message)


def test_valid_rank_deficient_kinship_does_not_warn(prob):
    """Round-off eigenvalues of a rank-deficient GRM are not reported."""
    rng, n, K, y, X = prob
    Klow = calculateKinship(rng.binomial(2, 0.3, (n, 5)).astype(float))  # rank <= 5
    with quiet(), warnings.catch_warnings():
        warnings.simplefilter("error")
        LMM(y, Klow).fit()


def test_individual_ids_checked(prob):
    rng, n, K, y, X = prob
    ids = ["i%d" % i for i in range(n)]
    perm = list(reversed(ids))
    GWAS(y, X[:, :1], K, ids=ids, K_ids=ids)
    with pytest.raises(ValueError, match="different order"):
        GWAS(y, X[:, :1], K, ids=ids, K_ids=perm)
    with pytest.raises(ValueError, match="not in the kinship"):
        LMM(y, K, ids=ids, K_ids=ids[:-1] + ["other"])


# ---------------------------------------------------------------------------
# P0-6 / issues 22, 24: phenotype checks
# ---------------------------------------------------------------------------


def test_constant_phenotype_raises(prob):
    rng, n, K, y, X = prob
    with pytest.raises(ValueError, match="phenotype has zero variance"):
        LMM(np.full(10, 3.0), np.eye(10))
    yc = np.full(n, 2.5)
    yc[:3] = np.nan
    with pytest.raises(ValueError, match="zero variance"):
        GWAS(yc, X, K)


def test_infinite_phenotype_raises(prob):
    rng, n, K, y, X = prob
    y = y.copy()
    y[4] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        GWAS(y, X, K)


def test_all_missing_phenotype(prob):
    rng, n, K, y, X = prob
    with pytest.raises(ValueError, match="no non-missing"):
        LMM(np.full(n, np.nan), K)
    t, p = GWAS(np.full(n, np.nan), X, K)
    assert np.isnan(t).all() and isinstance(t, list)


def test_cli_constant_phenotype_is_a_usage_error(tmp_path, prob):
    rng, n, K, y, X = prob
    snps, ph, kf = _write_emma(tmp_path, [X[:, 0]], np.full(n, 1.0), K)
    with pytest.raises(SystemExit):
        _run(["--emmaSNP", snps, "--emmaPHENO", ph, "--kfile", kf, str(tmp_path / "o")])


# ---------------------------------------------------------------------------
# P1-9 / issue 23, P1-11: covariate checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case,match",
    [
        ("dup", "columns 1 and 2 are identical"),
        ("comb", "column 3 is a linear combination of columns 0, 1, 2"),
        ("intercept", "columns 0 and 1 are identical"),
        ("zero", r"column\(s\) 1 are all zero"),
        ("nan", r"nan/inf values, in column\(s\) 1"),
    ],
)
def test_bad_covariates_raise_naming_the_column(prob, case, match):
    rng, n, K, y, X = prob
    c, d = rng.standard_normal(n), rng.standard_normal(n)
    X0 = {
        "dup": [np.ones(n), c, c],
        "comb": [np.ones(n), c, d, c - 2 * d + 1],
        "intercept": [np.ones(n), np.ones(n), c],
        "zero": [np.ones(n), np.zeros(n)],
        "nan": [np.ones(n), np.r_[np.nan, c[1:]]],
    }[case]
    X0 = np.column_stack(X0)
    with pytest.raises(ValueError, match=match):
        check_covariates(X0)
    with pytest.raises(ValueError, match=match):
        GWAS(y, X, K, X0=X0)
    if case != "nan":
        with pytest.raises(ValueError, match=match):
            LMM(y, K, X0=X0)


def test_covariate_collinear_only_after_dropping_missing_y(prob):
    rng, n, K, y, X = prob
    y = y.copy()
    y[:10] = np.nan
    b = np.r_[np.ones(10), np.zeros(n - 10)]  # 0 on every observed individual
    with pytest.raises(ValueError, match="individuals with a phenotype"):
        LMM(y, K, X0=np.column_stack([np.ones(n), b]))


def test_cli_covariate_file_with_intercept_points_at_nomean(tmp_path):
    """A covariate column of ones next to the intercept the CLI adds: the
    usage error names the columns and --noMean."""
    rng = np.random.default_rng(3)
    n = 30
    G = rng.binomial(2, 0.4, (n, 3)).astype(float)
    snps, ph, kf = _write_emma(tmp_path, G.T, rng.standard_normal(n), calculateKinship(G))
    cov = str(tmp_path / "c.txt")
    with open(cov, "w") as f:
        f.write(" ".join(["1"] * n) + "\n")
        f.write(" ".join("%.6f" % v for v in rng.standard_normal(n)) + "\n")
    err = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stderr(err):
        cli_gwas.main(
            [
                "--emmaSNP",
                snps,
                "--emmaPHENO",
                ph,
                "--kfile",
                kf,
                "--emmaCOV",
                cov,
                str(tmp_path / "o"),
            ]
        )
    assert "columns 0 and 1 are identical" in err.getvalue() and "--noMean" in err.getvalue()
    out = str(tmp_path / "o")
    _run(["--emmaSNP", snps, "--emmaPHENO", ph, "--kfile", kf, "--emmaCOV", cov, "--noMean", out])
    assert np.isfinite(_read(out)).all()


# ---------------------------------------------------------------------------
# bug 12: inputs are not modified
# ---------------------------------------------------------------------------


def test_gwas_and_lmm_do_not_modify_inputs(prob):
    rng, n, K, y, X = prob
    y = y.copy()
    y[2] = np.nan
    X[5, 1] = np.nan
    Kva, Kve = np.linalg.eigh(K)
    Kva[0] = -1e-3  # gets clamped inside LMM
    X0 = np.column_stack([np.ones(n), rng.standard_normal(n)])
    before = [a.copy() for a in (y, X, K, Kva, Kve, X0)]
    with quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GWAS(y, X, K, X0=X0, refit=True)
        LMM(y[~np.isnan(y)], K[np.ix_(~np.isnan(y), ~np.isnan(y))], X0=X0[~np.isnan(y)])
        LMM(np.nan_to_num(y), K, Kva, Kve, X0=X0)
    for a, b in zip((y, X, K, Kva, Kve, X0), before):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Phase 5.2 review fixes
# ---------------------------------------------------------------------------


def test_covariate_nan_only_where_y_is_nan_is_accepted(prob):
    """A covariate that is nan only for individuals without a phenotype is
    fine (it was on main): those rows are dropped before X0 is checked."""
    rng, n, K, y, X = prob
    y = y.copy()
    y[[3, 9]] = np.nan
    c = rng.standard_normal(n)
    X0 = np.column_stack([np.ones(n), c])
    X0[[3, 9], 1] = np.nan
    t, p = GWAS(y, X, K, X0=X0)
    keep = ~np.isnan(y)
    t2, p2 = GWAS(y[keep], X[keep], K[keep][:, keep], X0=X0[keep])
    np.testing.assert_allclose(t, t2, rtol=1e-10)
    L = LMM(y, K, X0=X0)
    assert keep.sum() == L.N
    # a nan on an individual that has a phenotype is still an error
    X0[5, 1] = np.nan
    with pytest.raises(ValueError, match="nan/inf"):
        GWAS(y, X, K, X0=X0)
    with pytest.raises(ValueError, match="nan/inf"):
        LMM(y, K, X0=X0)


def test_snp_estimable_with_big_offset():
    """The tolerance is relative to the centred x when X0 has an intercept:
    a rare SNP (MAF 0.002, n = 1000) plus a big offset is estimable."""
    n = 1000
    one = np.ones((n, 1))
    x = np.zeros(n)
    x[:4] = 1.0
    assert snp_estimable(x, one)
    for off in (1e2, 1e4, 1e6):
        assert snp_estimable(x + off, one)
        assert not snp_estimable(np.full(n, off), one)
        assert not snp_estimable(np.full(n, off) + 1e-12 * np.arange(n), one)
    # without an intercept in X0 the offset is part of the signal
    assert snp_estimable(x + 1e4, np.zeros((n, 0)))


def test_collinear_on_subset_gives_nan_and_one_summary_warning(prob):
    rng, n, K, y, X = prob
    b = np.zeros(n)
    b[:3] = 1.0  # an indicator of three individuals
    X0 = np.column_stack([np.ones(n), b])
    X[:3, 0] = np.nan  # ... all missing: b is 0 on the subset
    X[:3, 1] = np.nan
    X[:3, 2] = np.nan
    X[0, 3] = np.nan  # b stays non-constant
    with pytest.warns(UserWarning, match=r"3 of 5 SNPs are nan") as rec:
        t, p = GWAS(y, X, K, X0=X0)
    assert len([w for w in rec if "SNPs are nan" in str(w.message)]) == 1
    assert np.isnan(t[:3]).all() and np.isfinite(t[3:]).all()
