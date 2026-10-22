"""Batched Wald tests (Phase 6.1): `LMM.association_batch`, `gwas(batch=...)` and
``pylmm gwas --block-size / --no-batch``.

On well-conditioned designs the batched path must give the numbers of the
one-SNP-at-a-time path (the 1.0.0 code, ``batch=False`` / ``--no-batch``) up
to rounding (``helpers.BATCH_RTOL``), nan in the same places, in the same
order; on a badly scaled covariate, where the 1.0.0 path loses precision, it
must match an accurate reference (``helpers.BATCH_REF_RTOL``).  The
goldens (test_association.py, test_cli_gwas.py) run on the default, batched,
path and are not touched.
"""

import contextlib
import io
import os
import warnings

import helpers as h
import numpy as np
import pytest
import scipy
from scipy import linalg

from pylmm.lmm import (
    LMM,
    AssociationResult,
    calculate_kinship,
    gwas,
    snp_estimable,
    snp_estimable_columns,
)


def _data(n=150, m=300, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.binomial(2, 0.3, size=(n, 1000 + m)).astype(float)
    K = calculate_kinship(W[:, :1000])
    X = W[:, 1000:]
    kva, kve = linalg.eigh(K)
    u = kve @ (np.sqrt(np.clip(kva, 0, None)) * rng.normal(size=n))
    y = 0.5 * X[:, 0] + 0.8 * u + 0.6 * rng.normal(size=n)
    return y, X, K, kva, kve, rng


def _covariates(kind, n, rng):
    if kind == "intercept":
        return None
    cov = rng.normal(size=(n, 3)) * [1.0, 100.0, 1e-3]  # very different scales
    if kind == "covariates":
        return np.column_stack([np.ones(n), cov])
    if kind == "no-intercept":
        return cov
    assert kind == "empty"
    return np.zeros((n, 0))


# An X0 with no column (q = 0): the null fit (LMM.fit, scipy.linalg.inv of a 0 x 0 matrix)
# fails on it before scipy 1.14 (checked: 1.13.1 fails, 1.14.0 works), in 1.0.0 as on
# either path here, so the comparison needs scipy >= 1.14.
OLD_SCIPY = tuple(int(v) for v in scipy.__version__.split(".")[:2]) < (1, 14)
EMPTY_X0 = pytest.param(
    "empty",
    marks=pytest.mark.skipif(
        OLD_SCIPY, reason="LMM.fit with q = 0 needs scipy >= 1.14 (fails in 1.0.0 too)"
    ),
)
COVARIATE_KINDS = ["intercept", "covariates", "no-intercept", EMPTY_X0]


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
@pytest.mark.parametrize("kind", COVARIATE_KINDS)
def test_association_batch_matches_association(kind, reml):
    y, X, K, kva, kve, rng = _data()
    X0 = _covariates(kind, len(y), rng)
    L = LMM(y, K, kva, kve, X0)
    L.fit(reml=reml)
    cols = [j for j in range(X.shape[1]) if snp_estimable(X[:, j], L.X0)]
    assert len(cols) == X.shape[1] or kind in ("no-intercept", "empty")
    ref = np.array([L.association(X[:, j], reml=reml, return_beta=True) for j in cols])
    res = L.association_batch(X[:, cols], reml=reml)
    assert isinstance(res, AssociationResult)
    for k, name in enumerate(["t", "p", "beta", "beta_var"]):
        assert res[k].shape == (len(cols),) and res[k].dtype == np.float64
        h.assert_batch_close(res[k], ref[:, k], what="%s %s %s" % (kind, reml, name))
    # and at a given h
    ref_h = np.array([L.association(X[:, j], h=0.3, reml=reml) for j in cols[:20]])
    res_h = L.association_batch(X[:, cols[:20]], h=0.3, reml=reml)
    h.assert_batch_close(res_h.t, ref_h[:, 0], what="h=0.3 t")
    h.assert_batch_close(res_h.p, ref_h[:, 1], what="h=0.3 p")


def test_association_batch_shapes_and_degenerate_columns():
    y, X, K, kva, kve, _ = _data(m=5)
    L = LMM(y, K, kva, kve)
    L.fit()
    one = L.association_batch(X[:, 0])  # (n,) is one SNP
    t, p, b, v = L.association(X[:, 0], return_beta=True)
    assert one.t.shape == (1,)
    h.assert_batch_close(one.t, [t], what="1-d t")
    with pytest.raises(ValueError, match="must have shape"):
        L.association_batch(X[:-1])
    Z = X.copy()
    Z[:, 1] = 0.0  # s = 0 exactly
    Z[3, 2] = np.nan  # a missing call (association_batch does not handle them)
    res = L.association_batch(Z)
    for field in res:
        assert np.isnan(field[[1, 2]]).all() and np.isfinite(field[[0, 3, 4]]).all()


def _hard_columns(X0, rng):
    """Columns on and around every boundary of snp_estimable."""
    n = X0.shape[0]
    cols = [rng.binomial(2, 0.3, size=n).astype(float), rng.normal(size=n)]
    for offset in [0.0, 1.0, 1e8]:
        cols.append(np.full(n, offset))  # constant
        one = np.full(n, offset)
        one[0] += 1e-9 * max(offset, 1.0)  # constant but for one tiny step
        cols.append(one)
        cols.append(offset + 1e-6 * rng.normal(size=n))  # tiny variance, big offset
    span = X0 @ rng.normal(size=X0.shape[1]) if X0.shape[1] else np.zeros(n)
    noise = rng.normal(size=n)
    for d in 10.0 ** -np.arange(1, 13, 0.5):  # in span(X0) up to a relative d
        cols.append(span + d * np.linalg.norm(span) / np.sqrt(n) * noise)
    for v in [np.nan, np.inf]:
        bad = rng.normal(size=n)
        bad[3] = v
        cols.append(bad)
    return np.column_stack(cols)


@pytest.mark.parametrize("kind", ["intercept", "covariates", "no-intercept", "empty"])
def test_estimable_columns_equals_snp_estimable(kind):
    """The vectorised estimability check makes exactly the decisions of snp_estimable."""
    rng = np.random.default_rng(3)
    n = 120
    X0 = np.zeros((n, 0)) if kind == "empty" else _covariates(kind, n, rng)
    X0 = np.ones((n, 1)) if X0 is None else X0
    X = _hard_columns(X0, rng)
    expected = [snp_estimable(X[:, j], X0) for j in range(X.shape[1])]
    assert snp_estimable_columns(X, X0).tolist() == expected
    assert 0 < sum(expected) < X.shape[1]
    # too few individuals for q covariates plus the SNP
    small = X0[: X0.shape[1] + 1]
    got = snp_estimable_columns(X[: len(small)], small)
    assert got.tolist() == [snp_estimable(c, small) for c in X[: len(small)].T]


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
@pytest.mark.parametrize("kind", COVARIATE_KINDS)
def test_gwas_batch_matches_per_snp(kind, reml):
    """Mixed missing and complete, constant and collinear SNPs, any covariates."""
    y, X, K, kva, kve, rng = _data(m=120)
    X0 = _covariates(kind, len(y), rng)
    X = X.copy()
    X[rng.random(X.shape) < 0.01] = np.nan  # ~95% of the SNPs have a missing call...
    X[:, ::3] = np.nan_to_num(X[:, ::3], nan=1.0)  # ... except every third one
    X[:, 5] = 2.0  # constant
    X[:, 6] = np.nan  # all missing
    if X0 is not None and X0.shape[1] >= 2:
        X[:, 9] = 3.0 * X0[:, 1] + (1.0 if kind == "covariates" else 0.0)  # collinear with X0
    t_new, p_new = gwas(y, X, K, kva, kve, X0, reml=reml)
    t_old, p_old = gwas(y, X, K, kva, kve, X0, reml=reml, batch=False)
    assert len(t_new) == len(p_new) == X.shape[1]
    h.assert_batch_close(t_new, t_old, what="t")
    h.assert_batch_close(p_new, p_old, what="p")
    nan = np.isnan(np.asarray(t_new, dtype=float))
    if kind in ("intercept", "covariates"):
        assert nan[[5, 6]].all()
    if kind == "covariates":
        assert nan[9]
    # SNPs with a missing call take the per-SNP path in both: identical
    missing = np.isnan(X).any(axis=0)
    assert [t_new[j] for j in np.where(missing)[0]] == [t_old[j] for j in np.where(missing)[0]]


def test_gwas_block_boundaries_and_order():
    y, X, K, kva, kve, rng = _data(m=23)
    X = X.copy()
    X[4, 7] = np.nan  # one SNP on the per-SNP path, inside a block
    t_ref, p_ref = gwas(y, X, K, kva, kve, batch=False)
    for block_size in [1, 2, 5, 7, 22, 23, 24, 1000]:
        t, p = gwas(y, X, K, kva, kve, block_size=block_size)
        h.assert_batch_close(t, t_ref, what="t, block_size=%d" % block_size)
        h.assert_batch_close(p, p_ref, what="p, block_size=%d" % block_size)
        assert t[7] == t_ref[7]
    perm = rng.permutation(X.shape[1])
    t_perm, _ = gwas(y, X[:, perm], K, kva, kve, block_size=5)
    h.assert_batch_close(t_perm, np.asarray(t_ref)[perm], what="permuted columns")
    t1, _ = gwas(y, X[:, 0], K, kva, kve)  # one SNP as an (n,) vector
    h.assert_batch_close(t1, t_ref[:1], what="one SNP")
    for bad in [0, -3, 1.5, True, "8"]:
        with pytest.raises(ValueError, match="block_size"):
            gwas(y, X, K, block_size=bad)


def test_gwas_missing_phenotype_batched():
    y, X, K, _, _, _ = _data(m=40)
    y = y.copy()
    y[[3, 50, 77]] = np.nan
    t_new, p_new = gwas(y, X, K)
    t_old, p_old = gwas(y, X, K, batch=False)
    h.assert_batch_close(t_new, t_old, what="t")
    h.assert_batch_close(p_new, p_old, what="p")


def test_gwas_legacy_refit_ignores_batch():
    """pylmm 1.0.0's refit (legacy_fit) is one SNP at a time; the default refit is
    batched (LMM.fit_snps, tests/test_association_tests.py)."""
    y, X, K, kva, kve, _ = _data(n=60, m=4)
    a = gwas(y, X, K, kva, kve, refit=True, legacy_fit=True)
    assert a == gwas(y, X, K, kva, kve, refit=True, batch=False, legacy_fit=True)


def test_gwas_falls_back_to_per_snp_on_a_nan_batch_result(monkeypatch):
    """Not expected for an estimable SNP, but then the per-SNP answer is kept."""
    y, X, K, kva, kve, _ = _data(m=6)

    def all_nan(self, X, h=None, reml=True):
        nan = np.full(np.shape(X)[1], np.nan)
        return AssociationResult(nan, nan, nan, nan)

    expected = gwas(y, X, K, kva, kve, batch=False)
    monkeypatch.setattr(LMM, "association_batch", all_nan)
    assert gwas(y, X, K, kva, kve) == expected


@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_gwas_batch_hmdp(hmdp, reml):
    """The HMDP data: 94 strains, the first 300 SNPs, many with missing calls."""
    Y, K, snps = hmdp
    snps = snps[:, :300]
    with contextlib.redirect_stderr(io.StringIO()):
        t_new, p_new = gwas(Y, snps, K, reml=reml)
        t_old, p_old = gwas(Y, snps, K, reml=reml, batch=False)
    h.assert_batch_close(t_new, t_old, what="hmdp t")
    h.assert_batch_close(p_new, p_old, what="hmdp p")


def test_gwas_batch_bed(bed_null):
    """The README BED data: 1219 mice, first 1000 SNPs (standardised, imputed)."""
    Y, K, kva, kve = bed_null
    X = h.read_bed_snps(h.BED_GWAS_NSNPS, norm=True)
    t_new, p_new = gwas(Y, X, K, kva, kve)
    t_old, p_old = gwas(Y, X, K, kva, kve, batch=False)
    h.assert_batch_close(t_new, t_old, what="bed t")
    h.assert_batch_close(p_new, p_old, what="bed p")


def _gls_reference(L, X, h, x0_scaled, reml=True):
    """t, beta, beta_var per SNP by whitened QR on an equivalent, well-scaled X0."""
    w = np.sqrt(1.0 / (h * L.kva + 1 - h))
    x0t = L.kve.T @ x0_scaled
    yw = L.yt[:, 0] * w
    out = []
    for j in range(X.shape[1]):
        D = np.column_stack([x0t, L.kve.T @ X[:, j]]) * w[:, None]
        Q, R = np.linalg.qr(D)
        b = linalg.solve_triangular(R, Q.T @ yw)
        r = yw - D @ b
        p = D.shape[1]
        s2 = (r @ r) / ((L.N - p) if reml else L.N)
        rinv = linalg.solve_triangular(R, np.eye(p))[-1]
        v = s2 * (rinv @ rinv)
        out.append((b[-1] / np.sqrt(v), b[-1], v))
    return np.array(out)


@pytest.mark.parametrize(
    "covariate",
    ["year", "yyyymmdd", "1e4"],
    ids=["year 2000-2020", "date 2e7 +- 1e3", "1e4 +- 1"],
)
@pytest.mark.parametrize("reml", [True, False], ids=["reml", "ml"])
def test_batch_accurate_on_badly_scaled_covariates(covariate, reml):
    """Where the 1.0.0 path loses ~1e-6 (review S2), the batched path matches an accurate reference."""
    rng = np.random.default_rng(1)
    n = 300
    W = rng.binomial(2, 0.3, size=(n, 1200)).astype(float)
    K = calculate_kinship(W[:, :1000])
    X = W[:, 1000:]
    y = rng.normal(size=n) + 0.3 * X[:, 0]
    c = {
        "year": rng.integers(2000, 2021, n).astype(float),
        "yyyymmdd": 2.0e7 + 1e3 * rng.normal(size=n),
        "1e4": 1e4 + rng.normal(size=n),
    }[covariate]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy's LinAlgWarning from the null fit's inverse
        L = LMM(y, K, X0=np.column_stack([np.ones(n), c]))
        h0 = L.fit(reml=reml)[0]
        res = L.association_batch(X, h=h0, reml=reml)
        t_gwas, _ = gwas(y, X, K, X0=L.X0, reml=reml)
    ref = _gls_reference(L, X, h0, np.column_stack([np.ones(n), (c - c.mean()) / c.std()]), reml)
    for got, k, name in [(res.t, 0, "t"), (res.beta, 1, "beta"), (res.beta_var, 2, "beta_var")]:
        h.assert_ref_close(got, ref[:, k], what=name)
    h.assert_ref_close(t_gwas, ref[:, 0], what="gwas t")


# ---------------------------------------------------------------------------
# pylmm gwas
# ---------------------------------------------------------------------------


def _emma_run(tmp_path, tag, flags, nsnps=h.HMDP_NSNPS, cov=False):
    d = tmp_path / tag
    d.mkdir()
    emma = h.write_emma_subset(d, nsnps)
    args = [
        "gwas",
        "--emma-geno",
        emma,
        "--emma-pheno",
        h.HMDP_Y,
        "--kinship",
        h.write_headerless_kinship(d),
    ]
    if cov:
        args += ["--emma-covar", h.write_emma_cov(d)]
    out = str(d / "out")
    h.run_cli([*args, *flags, "-o", out], cwd=d)
    return h.load_gwas_out(out)


def _check_same(got, ref, what):
    ids, vals = got
    ref_ids, ref_vals = ref
    assert ids == ref_ids, what
    for j, col in enumerate(["BETA", "BETA_SD", "F_STAT", "P_VALUE"]):
        h.assert_batch_close(vals[:, j], ref_vals[:, j], what="%s %s" % (what, col))


@pytest.mark.parametrize(
    "flags",
    [[], ["--ml"], ["--remove-missing-genotypes"]],
    ids=["reml", "ml", "remove-missing"],
)
@pytest.mark.parametrize("cov", [False, True], ids=["nocov", "cov"])
def test_cli_batch_matches_no_batch(tmp_path, flags, cov):
    """1000 HMDP SNPs (100 with --remove-missing-genotypes, one eigendecomposition per SNP
    with missing calls)."""
    nsnps = h.HMDP_GZ_NSNPS if "--remove-missing-genotypes" in flags else h.HMDP_NSNPS
    ref = _emma_run(tmp_path, "ref", [*flags, "--no-batch"], nsnps=nsnps, cov=cov)
    _check_same(_emma_run(tmp_path, "new", flags, nsnps=nsnps, cov=cov), ref, "default")
    if not cov:
        new = _emma_run(tmp_path, "b7", [*flags, "--block-size", "7"], nsnps=nsnps)
        _check_same(new, ref, "block size 7")


def test_cli_block_size_one_and_refit(tmp_path):
    ref = _emma_run(tmp_path, "ref", ["--no-batch"], nsnps=h.HMDP_GZ_NSNPS)
    _check_same(_emma_run(tmp_path, "b1", ["--block-size", "1"], nsnps=h.HMDP_GZ_NSNPS), ref, "bs1")
    # --refit --legacy-fit (1.0.0's refit) is always one SNP at a time: --no-batch changes
    # nothing.  The default --refit is batched (tests/test_association_tests.py).
    refit = _emma_run(tmp_path, "refit", ["--refit", "--legacy-fit"], nsnps=20)
    refit_old = _emma_run(
        tmp_path, "refit_old", ["--refit", "--legacy-fit", "--no-batch"], nsnps=20
    )
    assert refit[0] == refit_old[0]
    np.testing.assert_array_equal(refit[1], refit_old[1])


def test_cli_block_size_must_be_positive(tmp_path):
    out = h.run_cli(
        ["gwas", "--bfile", "b", "--kinship", "k", "--block-size", "0", "-o", "o"],
        cwd=tmp_path,
        expect_rc=2,
    )
    assert "not positive" in out or "--block-size" in out


# ---------------------------------------------------------------------------
# pylmm gwas: what is written when something fails mid-scan (review finding S1)
#
# The rule is 1.0.0's: every SNP before the failing one has its line, the
# failing one and later ones have none, no SNP is tested twice and no line is
# written twice.  1.0.0 tested and wrote one SNP at a time, so with
# --no-batch the partial output is a prefix of the clean --no-batch output,
# which is byte for byte 1.0.0's.
# ---------------------------------------------------------------------------

FAIL_NSNPS = 20  # the first HMDP SNPs, with the missing calls of all but these filled in:
FAIL_MISSING = [2, 5, 7, 10, 12, 15, 17, 19]  # (tested one at a time, the others batched)


def _write_mixed_emma(d):
    """The first FAIL_NSNPS HMDP SNPs; NA set to 0 except in the rows FAIL_MISSING."""
    fn = h.write_emma_subset(d, FAIL_NSNPS)
    with open(fn) as f:
        rows = f.read().splitlines()
    rows = [r if j in FAIL_MISSING else r.replace("NA", "0") for j, r in enumerate(rows)]
    with open(fn, "w") as f:
        f.write("\n".join(rows) + "\n")
    return fn


def _inprocess(tmp_path, tag, flags, patch=None):
    """Run pylmm gwas in this process on FAIL_NSNPS HMDP SNPs; (rc, output lines).

    ``patch`` is (monkeypatch, object, attribute, k, exception): the k-th
    call of ``object.attribute`` raises ``exception`` instead.  ``k`` can
    also be a function of the call's arguments: the calls for which it is
    true raise.
    """
    from pylmm.cli import main

    d = tmp_path / tag
    d.mkdir()
    out = str(d / "out")
    argv = ["gwas", "-q", "--emma-geno", _write_mixed_emma(d)]
    argv += ["--emma-pheno", h.HMDP_Y, "--kinship", h.write_headerless_kinship(d)]
    argv += [*flags, "-o", out]
    calls = [0]
    with contextlib.ExitStack() as stack:
        if patch is not None:
            monkeypatch, obj, name, k, exc = patch
            mp = stack.enter_context(monkeypatch.context())
            orig = getattr(obj, name)

            def fail_on_kth_call(*a, **kw):
                calls[0] += 1
                if k(*a, **kw) if callable(k) else calls[0] == k:
                    raise exc
                return orig(*a, **kw)

            mp.setattr(obj, name, fail_on_kth_call)
        rc = main(h.legacy_output(argv))
    with open(out) as f:
        return rc, f.read().splitlines(), calls[0]


@pytest.mark.parametrize(
    "flags",
    [
        ["--no-batch"],
        ["--no-batch", "--block-size", "3"],
        ["--remove-missing-genotypes"],
        ["--remove-missing-genotypes", "--block-size", "3"],
    ],
    ids=["no-batch", "no-batch-bs3", "default", "default-bs3"],
)
def test_cli_error_while_testing_a_snp(tmp_path, monkeypatch, flags):
    """An unexpected error in the 5th per-SNP test: the lines of every SNP before it, only."""
    from pylmm.cli import gwas as cli_gwas

    rc, clean, _ = _inprocess(tmp_path, "clean", flags)
    assert rc == 0 and len(clean) == 1 + FAIL_NSNPS
    patch = (monkeypatch, cli_gwas, "_test_snp", 5, FloatingPointError("boom"))
    rc, lines, calls = _inprocess(tmp_path, "fail", flags, patch)
    if "--no-batch" in flags:
        failing = 4  # every SNP is tested on its own: the 5th SNP
    else:  # only the SNPs with a missing call are; the others are batched
        failing = FAIL_MISSING[4]
    assert rc == 1 and calls == 5  # no SNP tested twice
    assert lines == clean[: 1 + failing]


def test_cli_interrupt_and_write_failure_do_not_duplicate_lines(tmp_path, monkeypatch):
    from pylmm.cli import gwas as cli_gwas

    flags = ["--remove-missing-genotypes", "--block-size", "3"]
    rc, clean, _ = _inprocess(tmp_path, "clean", flags)
    failing = FAIL_MISSING[4]
    patch = (monkeypatch, cli_gwas, "_test_snp", 5, KeyboardInterrupt())
    rc, lines, calls = _inprocess(tmp_path, "interrupt", flags, patch)
    assert rc == 130 and calls == 5 and lines == clean[: 1 + failing]
    # the 5th result line cannot be written (e.g. disk full): the 4 before it stay, once
    patch = (monkeypatch, cli_gwas, "write_result", 5, OSError("No space left on device"))
    rc, lines, calls = _inprocess(tmp_path, "write", flags, patch)
    assert rc == 2 and calls == 5 and lines == clean[:5]


def _reading_snp(k):
    """For a `PlinkReader.__next__` patch: true when the call would read the k-th SNP."""
    return lambda reader: reader.have_read == k - 1


@pytest.mark.parametrize(
    "flags", [[], ["--no-batch"], ["--block-size", "3"]], ids=["default", "no-batch", "bs3"]
)
def test_cli_reader_error_mid_file(tmp_path, monkeypatch, flags):
    """The reader fails on SNP 7: SNPs 1-6 are tested and written, then the error is reported.

    The failure is tied to SNP 7, however often it is read: a block is read whole, so after
    a failed block `scan` reads it again one SNP at a time.  (A failure on the 7th read only
    is the transient case of `test_cli_transient_reader_error_is_recovered`.)  The error is
    an OSError, which the CLI reports with exit code 2: only data and I/O errors of the block
    reader are read again one SNP at a time (`test_cli_block_reader_bug_is_not_hidden`).
    """
    from pylmm import plink_io

    rc, clean, _ = _inprocess(tmp_path, "clean", flags)
    patch = (monkeypatch, plink_io.PlinkReader, "__next__", _reading_snp(7), OSError("disk gone"))
    rc, lines, _ = _inprocess(tmp_path, "fail", flags, patch)
    assert rc == 2 and len(lines) == 7
    if flags:  # the same blocks (or none) as the clean run: the same bytes
        assert lines == clean[:7]
    else:  # a block of 6 SNPs, not 20: a different GEMM, the same numbers to rounding
        got = [line.split("\t") for line in lines[1:]]
        ref = [line.split("\t") for line in clean[1:7]]
        assert [g[0] for g in got] == [r[0] for r in ref]
        for j in range(1, 5):
            h.assert_batch_close(
                [float(g[j]) for g in got], [float(r[j]) for r in ref], what="column %d" % j
            )


@pytest.mark.parametrize("flags", [[], ["--block-size", "3"]], ids=["default", "bs3"])
def test_cli_transient_reader_error_is_recovered(tmp_path, monkeypatch, flags):
    """The 7th read fails once (inside a block): the block is read again one SNP at a time,
    every SNP gets its line, once, the output is the clean run's, and a warning says so."""
    from pylmm import plink_io

    rc, clean, _ = _inprocess(tmp_path, "clean", flags)
    patch = (monkeypatch, plink_io.PlinkReader, "__next__", 7, OSError("disk gone"))
    log_file = str(tmp_path / "fail.log")
    rc, lines, _ = _inprocess(tmp_path, "fail", [*flags, "--log-file", log_file], patch)
    # Read again in blocks of the same size (one block of 20 by default): the same bytes.
    assert rc == 0 and lines == clean
    first = 1 if not flags else 7  # the first SNP of the failed block
    with open(log_file) as f:
        assert "failed from SNP %d (disk gone)" % first in f.read()


@pytest.mark.parametrize("flags", [[], ["--no-batch"]], ids=["default", "no-batch"])
def test_cli_input_error_mid_block_writes_the_snps_before_it(tmp_path, flags):
    """A ragged EMMA row: the SNPs before it are written, as before batching."""
    emma = h.write_emma_subset(tmp_path, 10)
    with open(emma) as f:
        rows = f.readlines()
    rows[6] = " ".join(rows[6].split()[:-1]) + "\n"  # SNP 7 has one value too few
    with open(emma, "w") as f:
        f.writelines(rows)
    out = str(tmp_path / "out")
    args = ["gwas", "--emma-geno", emma, "--emma-pheno", h.HMDP_Y]
    args += ["--kinship", h.write_headerless_kinship(tmp_path), "-o", out, *flags]
    log = h.run_cli(args, cwd=tmp_path, expect_rc=2)
    assert "line 7" in log
    ids, _ = h.load_gwas_out(out)
    assert ids == ["SNP_%d" % k for k in range(1, 7)]
    assert os.path.getsize(out) > 0


def _bed_run(tmp_path, tag, base, flags, expect_rc=0):
    """`pylmm gwas` on the BED file set ``base`` (README phenotypes, kinship); (log, lines)."""
    out = str(tmp_path / ("%s.tsv" % tag))
    args = ["gwas", "--bfile", base, "--kinship", h.README_KIN, "--pheno", h.FAKE_PHENOS]
    log = h.run_cli([*args, "-o", out, *flags], cwd=tmp_path, expect_rc=expect_rc)
    with open(out) as f:
        return log, f.read().splitlines()


@pytest.mark.parametrize(
    "flags", [[], ["--no-batch"], ["--block-size", "4"]], ids=["default", "no-batch", "bs4"]
)
def test_cli_truncated_bed_writes_the_snps_before_it(tmp_path, flags):
    """A BED file that ends inside SNP 7: SNPs 1-6 are written, then the error (exit 2).

    The block reader fails on the whole block; `scan` reads it again one SNP at a time."""
    base = h.write_bed_subset(tmp_path, h.BED_GWAS_SMALL_NSNPS)
    _, clean = _bed_run(tmp_path, "clean", base, flags)
    with open(base + ".fam") as f:
        nbytes = (sum(1 for _ in f) + 3) // 4
    with open(base + ".bed", "r+b") as f:
        f.truncate(3 + 6 * nbytes + 10)
    log, lines = _bed_run(tmp_path, "fail", base, flags, expect_rc=2)
    assert "Truncated BED file" in log and "SNP 7" in log
    assert len(lines) == 7
    if flags == ["--no-batch"]:
        assert lines == clean[:7]
    else:  # a shorter last block than the clean run's: the same numbers to rounding
        got = [line.split("\t") for line in lines[1:]]
        ref = [line.split("\t") for line in clean[1:7]]
        assert [g[0] for g in got] == [r[0] for r in ref]
        for j in range(1, 5):
            h.assert_batch_close(
                [float(g[j]) for g in got], [float(r[j]) for r in ref], what="column %d" % j
            )


def test_cli_bim_the_block_reader_rejects_is_read_one_snp_at_a_time(tmp_path):
    """A 5-column .bim (no A2): `iter_snps` reads it, `variants` (so `iter_blocks`) does not;
    the scan reads it one SNP at a time, with a warning, and writes the same bytes."""
    base = h.write_bed_subset(tmp_path, h.BED_GWAS_SMALL_NSNPS)
    _, clean = _bed_run(tmp_path, "clean", base, [])
    with open(base + ".bim") as f:
        rows = [line.split()[:5] for line in f]
    with open(base + ".bim", "w") as f:
        f.writelines(" ".join(r) + "\n" for r in rows)
    log, lines = _bed_run(tmp_path, "five", base, ["-v"])
    assert lines == clean
    warning = log.index("reading the SNPs one at a time (slower)")
    assert "line 1" in log[:warning]
    assert warning < log.index("Wrote the results")  # said at the start of the scan
    assert "blocks failed" not in log


def _main(argv, out):
    """Run ``pylmm <argv> -o out`` in this process; (rc, output lines or None)."""
    from pylmm.cli import main

    rc = main(h.legacy_output([*argv, "-o", out]))
    if not os.path.exists(out):
        return rc, None
    with open(out) as f:
        return rc, f.read().splitlines()


def _clean_inputs(tmp_path):
    """BED, TPED and EMMA `pylmm gwas` arguments (without -o) on small subsets."""
    kin = ["--kinship", h.README_KIN, "--pheno", h.FAKE_PHENOS]
    bed = ["gwas", "-q", "--bfile", h.write_bed_subset(tmp_path, h.BED_GWAS_SMALL_NSNPS), *kin]
    tped = ["gwas", "-q", "--tfile", h.write_tped_subset(tmp_path, 30), *kin]
    emma = ["gwas", "-q", "--emma-geno", h.write_emma_subset(tmp_path, 30)]
    emma += ["--emma-pheno", h.HMDP_Y, "--kinship", h.write_headerless_kinship(tmp_path)]
    return {"bed": bed, "tped": tped, "emma": emma}


@pytest.mark.parametrize("kind", ["bed", "tped", "emma"])
@pytest.mark.parametrize("flags", [[], ["--block-size", "7"]], ids=["default", "bs7"])
def test_cli_clean_run_takes_the_block_path(tmp_path, monkeypatch, kind, flags):
    """A clean BED, TPED or EMMA run never falls back to reading one SNP at a time."""
    from pylmm.cli import gwas as cli_gwas

    def no_fallback(*a, **kw):
        raise AssertionError("fell back to reading one SNP at a time")

    monkeypatch.setattr(cli_gwas, "_scan_snps", no_fallback)
    rc, lines = _main([*_clean_inputs(tmp_path)[kind], *flags], str(tmp_path / "out"))
    assert rc == 0 and len(lines) > 1


@pytest.mark.parametrize("output", ["legacy", "sumstats"])
@pytest.mark.parametrize("flags", [["--block-size", "7"], ["--block-size", "20"]])
def test_cli_transient_bed_decode_error_is_recovered(tmp_path, monkeypatch, flags, output):
    """An OSError while decoding the 2nd BED block, once: exit 0, a warning, the clean bytes.

    (The fallback stacks per-SNP vectors, C-ordered, where blocks are F-ordered views:
    the same numbers and the same bytes; in the summary-statistics table the SNPs read
    one at a time keep their positions and alleles.)"""
    from pylmm.plink_io import PlinkReader

    argv = [*_clean_inputs(tmp_path)["bed"], *flags, "--output-format", output]
    rc, clean = _main(argv, str(tmp_path / "clean"))
    assert rc == 0
    orig = PlinkReader._decode_bed_block.__func__
    calls = [0]

    def fail_once(cls, *a):
        calls[0] += 1
        if calls[0] == 2:
            raise OSError(5, "Input/output error")
        return orig(cls, *a)

    monkeypatch.setattr(PlinkReader, "_decode_bed_block", classmethod(fail_once))
    log_file = str(tmp_path / "fail.log")
    rc, lines = _main([*argv, "--log-file", log_file], str(tmp_path / "fail"))
    assert rc == 0 and lines == clean
    first = int(flags[1]) + 1  # the first SNP of the 2nd block
    with open(log_file) as f:
        assert "failed from SNP %d" % first in f.read()


@pytest.mark.parametrize("where", ["call", "next"])
def test_cli_block_reader_bug_is_not_hidden(tmp_path, monkeypatch, where):
    """An exception from the block reader that is not a data or I/O error (a bug) is raised:
    no fallback, exit 1."""
    from pylmm.plink_io import PlinkReader

    orig = PlinkReader.iter_blocks

    def broken(self, block_size):
        if where == "call":
            raise TypeError("simulated bug in the block reader")
        it = orig(self, block_size)

        def gen():
            yield next(it)
            raise TypeError("simulated bug in the block reader")

        return gen()

    monkeypatch.setattr(PlinkReader, "iter_blocks", broken)
    argv = [*_clean_inputs(tmp_path)["bed"], "--block-size", "7"]
    rc, lines = _main(argv, str(tmp_path / "out"))
    assert rc == 1
    assert lines is None or len(lines) <= 1 + 7  # the header and, at most, the first block
