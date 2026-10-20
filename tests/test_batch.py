"""Batched Wald tests (Phase 6.1): `LMM.association_batch`, `gwas(batch=...)` and
``pylmm gwas --block-size / --no-batch``.

The batched path must give the numbers of the one-SNP-at-a-time path (the
1.0.0 code, ``batch=False`` / ``--no-batch``) up to rounding
(``helpers.BATCH_RTOL``), nan in the same places, in the same order.  The
goldens (test_association.py, test_cli_gwas.py) run on the default, batched,
path and are not touched.
"""

import contextlib
import io
import os

import helpers as h
import numpy as np
import pytest
from scipy import linalg

from pylmm.lmm import (
    LMM,
    AssociationResult,
    _estimable_columns,
    calculate_kinship,
    gwas,
    snp_estimable,
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


COVARIATE_KINDS = ["intercept", "covariates", "no-intercept", "empty"]


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


@pytest.mark.parametrize("kind", COVARIATE_KINDS)
def test_estimable_columns_equals_snp_estimable(kind):
    """The vectorised estimability check makes exactly the decisions of snp_estimable."""
    rng = np.random.default_rng(3)
    n = 120
    X0 = _covariates(kind, n, rng)
    X0 = np.ones((n, 1)) if X0 is None else X0
    X = _hard_columns(X0, rng)
    expected = [snp_estimable(X[:, j], X0) for j in range(X.shape[1])]
    assert _estimable_columns(X, X0).tolist() == expected
    assert 0 < sum(expected) < X.shape[1]
    # too few individuals for q covariates plus the SNP
    small = X0[: X0.shape[1] + 1]
    got = _estimable_columns(X[: len(small)], small)
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
    with pytest.raises(ValueError, match="block_size"):
        gwas(y, X, K, block_size=0)


def test_gwas_missing_phenotype_batched():
    y, X, K, _, _, _ = _data(m=40)
    y = y.copy()
    y[[3, 50, 77]] = np.nan
    t_new, p_new = gwas(y, X, K)
    t_old, p_old = gwas(y, X, K, batch=False)
    h.assert_batch_close(t_new, t_old, what="t")
    h.assert_batch_close(p_new, p_old, what="p")


def test_gwas_refit_ignores_batch():
    y, X, K, kva, kve, _ = _data(n=60, m=4)
    assert gwas(y, X, K, kva, kve, refit=True) == gwas(y, X, K, kva, kve, refit=True, batch=False)


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
    # --refit is always one SNP at a time: the flags change nothing
    refit = _emma_run(tmp_path, "refit", ["--refit"], nsnps=20)
    refit_old = _emma_run(tmp_path, "refit_old", ["--refit", "--no-batch"], nsnps=20)
    assert refit[0] == refit_old[0]
    np.testing.assert_array_equal(refit[1], refit_old[1])


def test_cli_block_size_must_be_positive(tmp_path):
    out = h.run_cli(
        ["gwas", "--bfile", "b", "--kinship", "k", "--block-size", "0", "-o", "o"],
        cwd=tmp_path,
        expect_rc=2,
    )
    assert "not positive" in out or "--block-size" in out


def test_cli_input_error_mid_block_writes_the_snps_before_it(tmp_path):
    """A ragged EMMA row: the SNPs before it are written, as before batching."""
    emma = h.write_emma_subset(tmp_path, 10)
    with open(emma) as f:
        rows = f.readlines()
    rows[6] = " ".join(rows[6].split()[:-1]) + "\n"  # SNP 7 has one value too few
    with open(emma, "w") as f:
        f.writelines(rows)
    out = str(tmp_path / "out")
    args = ["gwas", "--emma-geno", emma, "--emma-pheno", h.HMDP_Y]
    args += ["--kinship", h.write_headerless_kinship(tmp_path), "-o", out]
    log = h.run_cli(args, cwd=tmp_path, expect_rc=2)
    assert "line 7" in log
    ids, _ = h.load_gwas_out(out)
    assert ids == ["SNP_%d" % k for k in range(1, 7)]
    assert os.path.getsize(out) > 0
