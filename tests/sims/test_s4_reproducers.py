"""S4 reproducers: minimal, deterministic demonstrations of pylmm's suspected bugs.

Bug numbers follow docs/phase1-baseline.md (17 is new; see
sims/studies/s4_bug_confirmation.md for the ledger and measured impact).

Two styles:

* ``@pytest.mark.xfail(strict=True, reason="bug N: ...")`` tests assert the
  CORRECT behaviour.  They fail today; when Phase 5/6 fixes the bug they
  pass, strict xfail turns that into an XPASS failure, and the marker must
  be removed.  That is the intended flip.
* ``test_bugN_current_...`` tests assert the current (buggy or conventional)
  behaviour, so the bug stays documented and visible.  A fix makes them fail;
  delete or invert them in the same change.

Everything is small (n <= 120) and in-process; the whole file runs in a few
seconds.  pylmm is never modified here.
"""

import contextlib
import inspect
import io
import os

import numpy as np
import pytest
from scipy import linalg

from pylmm import cli, misc, plink_io
from pylmm import lmm as pylmm_lmm
from pylmm.cli import gwas as cli_gwas
from pylmm.lmm import LMM, LogLikResult, calculate_kinship, gwas
from sims import oracle

try:  # scipy >= 1.11 (not exported publicly)
    from scipy.optimize._optimize import BracketError

    SCIPY_HAS_BRACKETERROR = True
except ImportError:  # pragma: no cover
    BracketError = RuntimeError
    SCIPY_HAS_BRACKETERROR = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def quiet():
    return contextlib.redirect_stderr(io.StringIO())


def toy(n=60, m=30, seed=0, maf=0.3):
    """Unrelated 0/1/2 genotypes, a GRM from m SNPs and a phenotype."""
    rng = np.random.default_rng(seed)
    G = rng.binomial(2, maf, size=(n, m)).astype(float)
    K = calculate_kinship(G.copy())
    y = rng.standard_normal(n) + 0.8 * (K @ rng.standard_normal(n)) / np.sqrt(m)
    return G, K, y, rng


def write_bed(base, G):
    """PLINK .bed/.bim/.fam for G coded the way pylmm decodes BED (0, 0.5, 1,
    nan).  Independent of pylmm's reader."""
    n, m = G.shape
    code = np.where(np.isnan(G), 1, np.where(G == 0, 0, np.where(G == 0.5, 2, 3))).astype(np.uint8)
    nb = (n + 3) // 4
    with open(base + ".bed", "wb") as f:
        f.write(bytes([0x6C, 0x1B, 0x01]))
        for j in range(m):
            c = np.zeros(nb * 4, dtype=np.uint8)
            c[:n] = code[:, j]
            c = c.reshape(nb, 4)
            f.write(
                (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6))
                .astype(np.uint8)
                .tobytes()
            )
    with open(base + ".bim", "w") as f:
        for j in range(m):
            f.write("1\trs%d\t0\t%d\tA\tG\n" % (j, 1000 * (j + 1)))
    with open(base + ".fam", "w") as f:
        for i in range(n):
            f.write("F%d I%d 0 0 0 -9\n" % (i, i))


def write_phenos(fn, y, skip=(), na=()):
    with open(fn, "w") as f:
        for i, v in enumerate(y):
            if i in skip:
                continue
            f.write("F%d I%d %s\n" % (i, i, "NA" if i in na else "%.10f" % v))


@pytest.fixture
def bed_toy(tmp_path):
    """A 40 x 25 BED (one missing call), its `pylmm kinship` kinship and phenos."""
    rng = np.random.default_rng(1)
    n, m = 40, 25
    G = rng.binomial(2, 0.4, size=(n, m)) / 2.0
    G[3, 2] = np.nan
    base = str(tmp_path / "toy")
    write_bed(base, G)
    y = rng.standard_normal(n)
    write_phenos(base + ".phenos", y)
    kin = str(tmp_path / "toy.kin")
    run_kinship(["--bfile", base, "-o", kin])
    return {"base": base, "kin": kin, "G": G, "y": y, "n": n, "m": m, "dir": tmp_path}


def run_gwas(argv, rc=0):
    """`pylmm gwas <argv>` in-process, stderr silenced; check the exit code."""
    with quiet():
        got = cli.main(["gwas", *argv])
    assert got == rc


def run_gwas_err(argv, rc=0):
    """run_gwas, returning what the CLI wrote to stderr."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        got = cli.main(["gwas", *argv])
    assert got == rc, err.getvalue()
    return err.getvalue()


def run_kinship(argv):
    """`pylmm kinship <argv>` in-process, stderr silenced; must succeed."""
    with quiet():
        assert cli.main(["kinship", *argv]) == 0


def read_out(fn):
    rows = [line.rstrip("\n").split("\t") for line in open(fn)][1:]
    return np.array([[float(x) for x in r[1:]] for r in rows])


def fake_lmm(ll):
    """An LMM whose likelihood is the given function of h (for _get_max logic)."""
    G, K, y, _ = toy(n=20, m=10)

    class Fake(LMM):
        def log_likelihood(self, h, X=None, stack=True, reml=True):
            return LogLikResult(float(ll(h)), np.zeros((1, 1)), 1.0, np.eye(1))

    with quiet():
        return Fake(y, K)


# ---------------------------------------------------------------------------
# bug 1: the null fit ignores REML
# ---------------------------------------------------------------------------


def _ml_bounded_case():
    # m < n - 1: K's null space is larger than span(1), so the ML profile is bounded
    G, K, y, rng = toy(n=80, m=25, seed=3)
    g_test = rng.binomial(2, 0.3, size=(80, 6)).astype(float)
    return g_test, K, y


def test_bug1_gwas_reml_flag_changes_t():
    """Regression (bug 1, fixed in Phase 5.2): reml=False used to give the
    REML t and p exactly; it is now an ML scan."""
    g_test, K, y = _ml_bounded_case()
    with quiet():
        t1, p1 = gwas(y, g_test, K, reml=True)
        t0, p0 = gwas(y, g_test, K, reml=False)
    assert np.max(np.abs(np.subtract(t0, t1))) > 1e-3


def _imputed(G):
    G = G.copy()
    for j in range(G.shape[1]):
        G[np.isnan(G[:, j]), j] = np.nanmean(G[:, j])
    return G


def test_bug1_cli_refit_follows_reml_choice(bed_toy):
    """Regression (bug 1, fixed in Phase 5.2): CLI --refit used to fit h by
    ML per SNP while the library fitted REML.  Now --refit is REML, --refit
    --ML is ML, and both match the oracle refit (CLI SNPs are mean-imputed)."""
    d = bed_toy["dir"]
    K = np.loadtxt(bed_toy["kin"])
    G = _imputed(bed_toy["G"])
    for flags, reml in (([], True), (["--ml"], False)):
        out = str(d / ("refit%d.out" % reml))
        run_gwas(
            ["--bfile", bed_toy["base"], "--kinship", bed_toy["kin"], "--refit", *flags, "-o", out]
        )
        t = read_out(out)[:, 2]
        for j in range(6):
            w = oracle.wald(
                bed_toy["y"], K, G[:, j], null_reml=reml, refit=True, eig_clamp=1e-6, h_grid="pylmm"
            )
            assert t[j] == pytest.approx(w.t, rel=1e-5, abs=1e-6), (flags, j)


def test_bug1_cli_ml_flag(bed_toy):
    """--ML fits the null model and runs the test by ML (sigma^2 = Q/n);
    --REML is a no-op; both together are a usage error."""
    d = bed_toy["dir"]
    K = np.loadtxt(bed_toy["kin"])
    G = _imputed(bed_toy["G"])
    outs = {}
    for tag, flags in (("default", []), ("reml", ["--reml"]), ("ml", ["--ml"])):
        outs[tag] = str(d / (tag + ".out"))
        run_gwas(["--bfile", bed_toy["base"], "--kinship", bed_toy["kin"], *flags, "-o", outs[tag]])
    np.testing.assert_array_equal(read_out(outs["default"]), read_out(outs["reml"]))
    t_ml = read_out(outs["ml"])[:, 2]
    for j in range(bed_toy["m"]):
        w = oracle.wald(bed_toy["y"], K, G[:, j], null_reml=False, eig_clamp=1e-6, h_grid="pylmm")
        assert t_ml[j] == pytest.approx(w.t, rel=1e-5, abs=1e-6)
    run_gwas(
        [
            "--bfile",
            bed_toy["base"],
            "--kinship",
            bed_toy["kin"],
            "--reml",
            "--ml",
            "-o",
            str(d / "x"),
        ],
        rc=2,
    )


def test_bug1_remove_missing_follows_reml_choice(bed_toy):
    """Regression (bug 1 / S5 P0-4, fixed in Phase 5.2): with
    --removeMissingGenotypes the SNP with a missing call (SNP 2) is refitted
    on its non-missing individuals by REML, like the library, or by ML with
    --ML; complete SNPs give the same t as without the flag."""
    d = bed_toy["dir"]
    G = bed_toy["G"]
    with quiet():
        lib_reml = gwas(bed_toy["y"], G, np.loadtxt(bed_toy["kin"]), reml=True)[0]
        lib_ml = gwas(bed_toy["y"], G, np.loadtxt(bed_toy["kin"]), reml=False)[0]
    for flags, lib in (([], lib_reml), (["--ml"], lib_ml)):
        a, b = str(d / "rm.out"), str(d / "imp.out")
        run_gwas(
            [
                "--bfile",
                bed_toy["base"],
                "--kinship",
                bed_toy["kin"],
                "--remove-missing-genotypes",
                *flags,
                "-o",
                a,
            ]
        )
        run_gwas(["--bfile", bed_toy["base"], "--kinship", bed_toy["kin"], *flags, "-o", b])
        ta, tb = read_out(a)[:, 2], read_out(b)[:, 2]
        complete = np.delete(np.arange(bed_toy["m"]), 2)
        np.testing.assert_allclose(ta[complete], tb[complete], rtol=1e-8)
        np.testing.assert_allclose(ta, lib, rtol=1e-6, atol=1e-8)


def test_bug1_gwas_ml_equals_textbook_ml_wald():
    """Regression (bug 1 and, under ML only, bug 9; fixed in Phase 5.2):
    gwas(reml=False) is an ML null fit plus the textbook ML Wald test,
    sigma^2 = Q/n."""
    g_test, K, y = _ml_bounded_case()
    with quiet():
        t0, _ = gwas(y, g_test, K, reml=False)
    w = [
        oracle.wald(y, K, g_test[:, j], null_reml=False, sigma_convention="textbook").t
        for j in range(g_test.shape[1])
    ]
    np.testing.assert_allclose(t0, w, atol=1e-4)


# ---------------------------------------------------------------------------
# bugs 2, 3, 4: the h search
# ---------------------------------------------------------------------------


def test_bug2_current_getmax_skips_grid_index_98():
    L = fake_lmm(lambda h: -((h - 0.981) ** 2))  # single peak, nearest grid point 0.98
    h = L.fit()[0]
    assert h == 0.99  # endpoint fallback


@pytest.mark.xfail(strict=True, reason="bug 2: range(1, n-2) never tests grid index n-2 (h = 0.98)")
def test_bug2_getmax_finds_peak_at_098():
    L = fake_lmm(lambda h: -((h - 0.981) ** 2))
    assert abs(L.fit()[0] - 0.981) < 1e-4


def test_bug3_current_h_grid_stops_at_099():
    L = fake_lmm(lambda h: h)  # increasing: maximum at h = 1
    assert L.fit()[0] == 0.99
    assert L.h_grid.max() == 0.99


@pytest.mark.xfail(strict=True, reason="bug 3: the h grid excludes 1.0")
def test_bug3_h_search_reaches_one():
    L = fake_lmm(lambda h: h)
    assert L.fit()[0] == 1.0


@pytest.mark.xfail(strict=True, reason="bug 3: misc.fit_two's w grid excludes 1.0 (K1 alone)")
def test_bug3_fittwo_can_select_k1_alone():
    rng = np.random.default_rng(5)
    n = 50
    G1 = rng.binomial(2, 0.4, size=(n, 200)).astype(float)
    G2 = rng.binomial(2, 0.4, size=(n, 200)).astype(float)
    K1, K2 = calculate_kinship(G1), calculate_kinship(G2)
    d, V = linalg.eigh(K1)
    y = V @ (np.sqrt(np.clip(d, 0, None)) * rng.standard_normal(n)) * 3 + 0.1 * rng.standard_normal(
        n
    )
    with quiet():
        h1, h2, e, beta, sigma, log_likelihood = misc.fit_two(y, K1, K2, ngrids=20)
    assert h2 == 0.0


def test_bug4_ll_is_nan_above_one_but_ll_brent_guards_both_ends():
    G, K, y, _ = toy()
    with quiet():
        L = LMM(y, K)
    with np.errstate(invalid="ignore"):
        assert np.isnan(L.log_likelihood(1.02, reml=False).ll)  # log of a negative h*kva + 1 - h
    assert L._neg_log_likelihood(-0.1) == 1e6
    assert L._neg_log_likelihood(1.02) == 1e6  # fixed in Phase 5.1: h > 1 guarded too


def _peak_0972(h):
    if h > 1:
        return np.nan  # what pylmm's LL gives above 1
    return -((h - 0.972) ** 2)


@pytest.mark.skipif(not SCIPY_HAS_BRACKETERROR, reason="scipy < 1.11 returns instead of raising")
def test_bug4_old_unbounded_brent_crashed_with_bracketerror():
    """Documents scipy's behaviour on the stored profile (not pylmm code, which
    no longer calls unbounded brent): peak at grid index 97 with
    LL(0.98) > LL(0.96), so the unbounded optimize.brent pylmm used before
    Phase 5.1 stepped to 0.98 + 1.618 * 0.02 = 1.012, got nan, and on
    scipy >= 1.11 raised BracketError."""
    from scipy import optimize

    with pytest.raises(BracketError):
        optimize.brent(lambda h: 1e6 if h < 0 else -_peak_0972(h), brack=(0.96, 0.98))


def test_bug4_fit_with_peak_near_one():
    """Regression (bug 4, fixed in Phase 5.1): the bounded search returns the
    interior optimum on every scipy (it used to raise BracketError on
    scipy >= 1.11)."""
    L = fake_lmm(_peak_0972)
    assert abs(L.fit()[0] - 0.972) < 1e-6


def _simulate_high_h2(seed):
    rng = np.random.default_rng(seed)
    n, m = 60, 30
    G = rng.binomial(2, 0.3, size=(n, m)).astype(float)
    K = calculate_kinship(G.copy())
    d, V = linalg.eigh(K)
    u = V @ (np.sqrt(np.clip(d, 0, None)) * rng.standard_normal(n))
    y = np.sqrt(0.97) * u + np.sqrt(0.03) * rng.standard_normal(n)
    return y, K


def _reml_grid(y, K):
    with quiet():
        L = LMM(y, K)
    return L, np.array([L.log_likelihood(h, reml=True)[0] for h in np.arange(100) / 100.0])


def _high_h2_case():
    """A real GRM (n = 60, m = 30) and y simulated at h2 = 0.97 whose REML
    grid peak is at index 97 with LL(0.98) > LL(0.96) -- the shape on which
    the old unbounded Brent stepped past h = 1.  The profile near h = 1 of a
    rank-deficient K is sensitive to platform round-off, so the seed is
    searched for here rather than hard-coded."""
    for seed in range(500):
        y, K = _simulate_high_h2(seed)
        _, ll_grid = _reml_grid(y, K)
        if ll_grid[96] < ll_grid[97] > ll_grid[98] and ll_grid[98] > ll_grid[96]:
            return y, K
    pytest.fail("no seed in range(500) gives a REML peak at grid index 97")


def test_bug4_high_h2_case_matches_oracle():
    """Regression (bug 4, fixed in Phase 5.1) on real likelihoods, null fit
    and per-SNP refit: pylmm matches the oracle with pylmm's conventions
    (eig_clamp=1e-6, the pylmm grid and search) to ~1e-6.  On scipy >= 1.11
    the old unbounded Brent raised BracketError on this case.  The null fit has
    a high-h2 peak; the refit part uses a random x, so it is only a no-crash,
    matches-oracle check, not a near-1 peak."""
    from scipy import optimize

    y, K = _high_h2_case()
    L, _ = _reml_grid(y, K)
    if SCIPY_HAS_BRACKETERROR:
        old = lambda h: 1e6 if h < 0 else -L.log_likelihood(h, stack=False, reml=True)[0]  # noqa: E731
        with pytest.raises(BracketError), np.errstate(all="ignore"):
            optimize.brent(old, brack=(0.96, 0.98))
    f = oracle.fit(y, K, reml=True, eig_clamp=1e-6, h_grid="pylmm")
    assert 0.96 < f.h < 0.98
    h = float(np.sum(L.fit(reml=True)[0]))
    assert abs(h - f.h) < 1e-6
    # the 0.99 cap does not bind: the clamped global optimum is the same h
    assert abs(h - oracle.fit(y, K, reml=True, eig_clamp=1e-6).h) < 1e-6
    # per-SNP refit path (fit(X=x)) on the same data: no crash, h in [0, 1]
    x = np.random.default_rng(3).binomial(2, 0.3, size=(len(y), 1)).astype(float)
    with quiet():
        hx = float(np.sum(L.fit(X=x, reml=True)[0]))
    fx = oracle.fit(y, K, X=np.c_[np.ones(len(y)), x], reml=True, eig_clamp=1e-6, h_grid="pylmm")
    assert 0.0 <= hx <= 1.0 and abs(hx - fx.h) < 1e-6


# ---------------------------------------------------------------------------
# bug 5: tstat(log=True)
# ---------------------------------------------------------------------------


def test_bug5_current_log_p_is_off_by_2_minus_log2_and_unreachable():
    G, K, y, _ = toy()
    with quiet():
        L = LMM(y, K)
    L.fit()
    beta, var, sigma = np.array([[0.3]]), np.array([[0.01]]), np.array([[1.0]])
    _, p = L.tstat(beta, var, sigma, 2)
    _, logp = L.tstat(beta, var, sigma, 2, log=True)
    assert logp - np.log(p) == pytest.approx(2.0 - np.log(2.0), abs=1e-12)
    # unreachable: nothing in the package calls tstat(..., log=True)
    for mod in (pylmm_lmm, cli_gwas, misc):
        src = inspect.getsource(mod)
        assert "log=True" not in src.replace("def tstat(self,beta,var,sigma,q,log=False)", "")


@pytest.mark.xfail(strict=True, reason="bug 5: tstat(log=True) adds 2.0 instead of log(2)")
def test_bug5_log_p():
    G, K, y, _ = toy()
    with quiet():
        L = LMM(y, K)
    L.fit()
    beta, var, sigma = np.array([[0.3]]), np.array([[0.01]]), np.array([[1.0]])
    _, p = L.tstat(beta, var, sigma, 2)
    _, logp = L.tstat(beta, var, sigma, 2, log=True)
    assert logp == pytest.approx(np.log(p), abs=1e-12)


# ---------------------------------------------------------------------------
# bugs 6, 7, 14: CLI options that crash
# ---------------------------------------------------------------------------


def test_cli_toy_default_run_works(bed_toy):
    out = str(bed_toy["dir"] / "ok.out")
    run_gwas(["--bfile", bed_toy["base"], "--kinship", bed_toy["kin"], "-o", out])
    assert read_out(out).shape == (bed_toy["m"], 4)


def test_bug6_kfile2_removed(bed_toy):
    """Regression (bug 6, fixed in Phase 5.2): --kfile2 called the undefined
    LMM_withK2 (NameError); the option is removed, so it is a usage error."""
    out = str(bed_toy["dir"] / "k2.out")
    run_gwas(
        [
            "--bfile",
            bed_toy["base"],
            "--kinship",
            bed_toy["kin"],
            "--kfile2",
            bed_toy["kin"],
            "-o",
            out,
        ],
        rc=2,
    )


@pytest.mark.parametrize("fmt", ["text", "npz"])
def test_bug7_eigen_reads_pylmmkinship_output(bed_toy, fmt):
    """Regression (bug 7, fixed in Phase 5.2): --eigen reads what
    `pylmm kinship` writes, as text (--eigen-text: <kin>.kva/.kve) or .npz
    (--eigen-npz), and gives the same results as without it."""
    d = bed_toy["dir"]
    npz = str(d / "eig.npz")
    run_kinship(
        ["--bfile", bed_toy["base"], "--eigen-text", "--eigen-npz", npz, "-o", bed_toy["kin"]]
    )
    assert os.path.exists(bed_toy["kin"] + ".kva") and os.path.exists(npz)
    np.testing.assert_array_equal(np.loadtxt(bed_toy["kin"] + ".kva"), np.load(npz)["Kva"])
    ref, out = str(d / "ref.out"), str(d / "eig.out")
    run_gwas(["--bfile", bed_toy["base"], "--kinship", bed_toy["kin"], "-o", ref])
    run_gwas(
        [
            "--bfile",
            bed_toy["base"],
            "--kinship",
            bed_toy["kin"],
            "--eigen",
            npz if fmt == "npz" else bed_toy["kin"],
            "-o",
            out,
        ]
    )
    np.testing.assert_allclose(read_out(out), read_out(ref), rtol=1e-8, atol=1e-10)


def test_bug7_eigen_of_another_kinship_rejected(bed_toy):
    """A decomposition of another K, of the wrong size, or a missing file is
    a usage error."""
    d = bed_toy["dir"]
    np.savez(str(d / "bad.npz"), Kva=np.ones(bed_toy["n"]), Kve=np.eye(bed_toy["n"]))
    np.savez(str(d / "short.npz"), Kva=np.ones(3), Kve=np.eye(3))
    for f in ("bad.npz", "short.npz", "missing.npz"):
        run_gwas(
            [
                "--bfile",
                bed_toy["base"],
                "--kinship",
                bed_toy["kin"],
                "--eigen",
                str(d / f),
                "-o",
                str(d / "x"),
            ],
            rc=2,
        )


def test_bug14_no_pheno_file_is_a_usage_error(bed_toy):
    """Regression (bug 14, fixed in Phase 5.2): used to be a TypeError."""
    os.remove(bed_toy["base"] + ".phenos")
    run_gwas(
        ["--bfile", bed_toy["base"], "--kinship", bed_toy["kin"], "-o", str(bed_toy["dir"] / "x")],
        rc=2,
    )


# ---------------------------------------------------------------------------
# bug 8: kinship normalisation
# ---------------------------------------------------------------------------


def test_bug8_current_monomorphic_snps_shrink_k():
    G, _, _, _ = toy(n=30, m=40)
    K = calculate_kinship(G.copy())
    k_mono = calculate_kinship(np.hstack([G, np.zeros((30, 10))]))
    np.testing.assert_allclose(k_mono, K * 40 / 50, rtol=1e-12)


@pytest.mark.xfail(
    strict=True, reason="bug 8: calculate_kinship divides by m including dropped monomorphic SNPs"
)
def test_bug8_monomorphic_snps_do_not_change_k():
    G, _, _, _ = toy(n=30, m=40)
    np.testing.assert_allclose(
        calculate_kinship(np.hstack([G, np.zeros((30, 10))])),
        calculate_kinship(G.copy()),
        rtol=1e-12,
    )


def test_bug8_current_kinship_cli_counts_monomorphic_snps(tmp_path):
    rng = np.random.default_rng(2)
    G = rng.binomial(2, 0.4, size=(24, 12)) / 2.0
    base = str(tmp_path / "mono")
    write_bed(base, np.hstack([G, np.zeros((24, 4))]))  # 4 monomorphic SNPs
    run_kinship(["--bfile", base, "-o", str(tmp_path / "k")])
    k_cli = np.loadtxt(str(tmp_path / "k"))
    np.testing.assert_allclose(k_cli, calculate_kinship(G.copy()) * 12 / 16, rtol=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason="bug 8: lmm.calculate_kinship and `pylmm kinship` normalise missing data differently",
)
def test_bug8_two_kinships_agree_with_missing_data(tmp_path):
    rng = np.random.default_rng(3)
    G = rng.binomial(2, 0.4, size=(24, 12)) / 2.0
    G[rng.random(G.shape) < 0.1] = np.nan
    base = str(tmp_path / "miss")
    write_bed(base, G)
    run_kinship(["--bfile", base, "-o", str(tmp_path / "k")])
    np.testing.assert_allclose(
        np.loadtxt(str(tmp_path / "k")), calculate_kinship(G.copy()), atol=1e-8
    )


# ---------------------------------------------------------------------------
# bug 9: ML sigma
# ---------------------------------------------------------------------------


def test_bug9_ml_wald_uses_q_over_n():
    """Phase 5.2: association(reml=False) uses the ML sigma^2 = Q/n (it used
    Q/(n-q) in both modes); REML keeps Q/(n-q).  fit()'s reported sigma is
    unchanged (Phase 6.3)."""
    g_test, K, y = _ml_bounded_case()
    with quiet():
        L = LMM(y, K)
    L.fit(reml=False)
    x = g_test[:, :1]
    t_ml, _ = L.association(x, reml=False)
    t_r, _ = L.association(x, reml=True)
    wtb = oracle.wald(y, K, x, h=L.opt_h, null_reml=False, sigma_convention="textbook")
    assert t_ml == pytest.approx(wtb.t, rel=1e-5)
    n, p = len(y), 2
    assert t_ml / t_r == pytest.approx(np.sqrt(n / (n - p)), rel=1e-10)


# ---------------------------------------------------------------------------
# bug 10: phenotype file missing individuals
# ---------------------------------------------------------------------------


def test_bug10_getphenos_aligns_by_id(bed_toy):
    """Regression (bug 10, fixed in Phase 5.2): absent individuals get nan
    in their own row (they used to shift everyone after them), with a
    warning that counts them."""
    fn = str(bed_toy["dir"] / "miss.ph")
    write_phenos(fn, bed_toy["y"], skip=(5, 17))
    with (
        quiet(),
        pytest.warns(UserWarning, match="2 of 40 individuals are not in the phenotype file"),
    ):
        reader = plink_io.PlinkReader(bed_toy["base"], type="b", pheno_file=fn)
    assert reader.phenos.shape == (bed_toy["n"], 1)
    assert np.isnan(reader.phenos[[5, 17], 0]).all()
    keep = np.setdiff1d(np.arange(bed_toy["n"]), [5, 17])
    np.testing.assert_allclose(reader.phenos[keep, 0], bed_toy["y"][keep], atol=1e-10)  # %.10f


def test_bug10_pheno_file_reordered_or_with_strangers(bed_toy):
    """Rows in another order, plus an unknown individual: same as the plain file."""
    d = bed_toy["dir"]
    y = bed_toy["y"]
    with open(str(d / "shuf.ph"), "w") as f:
        f.write("FX IX 1.0\n")
        for i in np.random.default_rng(0).permutation(len(y)):
            f.write("F%d I%d %.10f\n" % (i, i, y[i]))
    run_gwas(["--bfile", bed_toy["base"], "--kinship", bed_toy["kin"], "-o", str(d / "a.out")])
    run_gwas(
        [
            "--bfile",
            bed_toy["base"],
            "--kinship",
            bed_toy["kin"],
            "--pheno",
            str(d / "shuf.ph"),
            "-o",
            str(d / "b.out"),
        ]
    )
    np.testing.assert_allclose(read_out(str(d / "b.out")), read_out(str(d / "a.out")), rtol=1e-12)


def test_bug10_unalignable_pheno_file_is_a_usage_error(bed_toy):
    d = bed_toy["dir"]
    with open(str(d / "none.ph"), "w") as f:
        f.write("A B 1.0\nC D 2.0\n")
    with open(str(d / "dup.ph"), "w") as f:
        f.write("F0 I0 1.0\nF0 I0 2.0\n")
    for fn in ("none.ph", "dup.ph"):
        run_gwas(
            [
                "--bfile",
                bed_toy["base"],
                "--kinship",
                bed_toy["kin"],
                "--pheno",
                str(d / fn),
                "-o",
                str(d / "x"),
            ],
            rc=2,
        )


def test_bug10_absent_individuals_treated_as_missing(bed_toy):
    """Regression (bug 10, fixed in Phase 5.2)."""
    d = bed_toy["dir"]
    write_phenos(str(d / "na.ph"), bed_toy["y"], na=(5, 17))
    write_phenos(str(d / "skip.ph"), bed_toy["y"], skip=(5, 17))
    run_gwas(
        [
            "--bfile",
            bed_toy["base"],
            "--kinship",
            bed_toy["kin"],
            "--pheno",
            str(d / "na.ph"),
            "-o",
            str(d / "na.out"),
        ]
    )
    err = run_gwas_err(
        [
            "--bfile",
            bed_toy["base"],
            "--kinship",
            bed_toy["kin"],
            "--pheno",
            str(d / "skip.ph"),
            "-o",
            str(d / "skip.out"),
        ]
    )
    # a plain stderr line, not a UserWarning with a source line
    assert "pylmm: warning: 2 of 40 individuals are not in the phenotype file" in err
    assert "UserWarning" not in err
    assert "2 of 40 individuals have a missing phenotype and are dropped" in err
    np.testing.assert_allclose(read_out(str(d / "skip.out")), read_out(str(d / "na.out")))


def test_bug10_covariate_file_absent_individuals_dropped(bed_toy):
    """A covariate file missing an individual, or with NA: those individuals
    are dropped with a warning, the same as setting their phenotype to NA
    (a missing covariate used to be a fatal error)."""
    d = bed_toy["dir"]
    c = np.cos(np.arange(bed_toy["n"]))
    with open(str(d / "cov.txt"), "w") as f:
        for i in range(bed_toy["n"]):
            if i == 5:
                continue
            f.write("F%d I%d %s\n" % (i, i, "NA" if i == 17 else "%.10f" % c[i]))
    with open(str(d / "cov_full.txt"), "w") as f:
        for i in range(bed_toy["n"]):
            f.write("F%d I%d %.10f\n" % (i, i, c[i]))
    write_phenos(str(d / "na.ph"), bed_toy["y"], na=(5, 17))
    msgs = run_gwas_err(
        [
            "--bfile",
            bed_toy["base"],
            "--kinship",
            bed_toy["kin"],
            "--covar",
            str(d / "cov.txt"),
            "-o",
            str(d / "a.out"),
        ]
    )
    assert "pylmm: warning: 1 of 40 individuals are not in the covariate file" in msgs
    assert "2 individuals with a phenotype have a missing covariate value and are dropped" in msgs
    assert "UserWarning" not in msgs
    run_gwas(
        [
            "--bfile",
            bed_toy["base"],
            "--kinship",
            bed_toy["kin"],
            "--covar",
            str(d / "cov_full.txt"),
            "--pheno",
            str(d / "na.ph"),
            "-o",
            str(d / "b.out"),
        ]
    )
    np.testing.assert_allclose(read_out(str(d / "a.out")), read_out(str(d / "b.out")))


# ---------------------------------------------------------------------------
# bug 11: an all-missing SNP
# ---------------------------------------------------------------------------


def test_bug11_normalize_all_missing_keeps_length():
    """Regression (bug 11, fixed in Phase 5.2): used to return an empty array."""
    g = plink_io.PlinkReader.normalize_genotype(None, np.full(5, np.nan))
    assert g.shape == (5,) and np.isnan(g).all()


@pytest.mark.parametrize("flags", [[], ["--remove-missing-genotypes"]])
def test_bug11_cli_gwas_all_missing_snp(tmp_path, flags):
    """Regression (bug 11, fixed in Phase 5.2): nan for that SNP, and the
    scan continues."""
    rng = np.random.default_rng(4)
    G = rng.binomial(2, 0.4, size=(24, 6)) / 2.0
    G[:, 3] = np.nan
    base = str(tmp_path / "allmiss")
    write_bed(base, G)
    write_phenos(base + ".phenos", rng.standard_normal(24))
    np.savetxt(str(tmp_path / "k"), calculate_kinship(G[:, [0, 1, 2, 4, 5]].copy()))
    run_gwas(["--bfile", base, "--kinship", str(tmp_path / "k"), *flags, "-o", str(tmp_path / "o")])
    r = read_out(str(tmp_path / "o"))
    assert r.shape == (6, 4)
    assert np.isnan(r[3]).all() and np.isfinite(np.delete(r, 3, axis=0)).all()


def test_bug11_library_all_missing_snp():
    """GWAS: nan for the all-missing SNP (it used to raise LinAlgError)."""
    G, K, y, _ = toy()
    X = G[:, :4].copy()
    X[:, 1] = np.nan
    with quiet():
        t, p = gwas(y, X, K)
    assert np.isnan(t[1]) and np.isnan(p[1]) and np.isfinite(np.delete(t, 1)).all()


def test_bug11_cli_kinship_all_missing_snp(tmp_path):
    """Regression (bug 11, fixed in Phase 5.2): `pylmm kinship` skips the SNP
    with a warning (it used to raise).  The denominator still counts it
    (bug 8, Phase 6.3)."""
    rng = np.random.default_rng(4)
    G = rng.binomial(2, 0.4, size=(24, 6)) / 2.0
    G[:, 3] = np.nan
    base = str(tmp_path / "allmiss")
    write_bed(base, G)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert cli.main(["kinship", "--bfile", base, "-o", str(tmp_path / "k")]) == 0
    assert "pylmm: warning: skipped 1 SNPs with no called genotype" in err.getvalue()
    K = np.loadtxt(str(tmp_path / "k"))
    np.testing.assert_allclose(K, calculate_kinship(G[:, [0, 1, 2, 4, 5]]) * 5 / 6, rtol=1e-10)


def test_bug11_calculatekinship_skips_all_missing_snp():
    G, _, _, _ = toy(n=30, m=10)
    K = calculate_kinship(np.hstack([G, np.full((30, 1), np.nan)]))
    assert np.isfinite(K).all()
    np.testing.assert_allclose(K, calculate_kinship(G) * 10 / 11, rtol=1e-12)


# ---------------------------------------------------------------------------
# bug 12: callers' arrays mutated
# ---------------------------------------------------------------------------


def test_bug12_lmm_does_not_mutate_kva():
    """Regression (bug 12, fixed in Phase 5.2)."""
    G, K, y, _ = toy()
    kva, kve = linalg.eigh(K)
    before = kva.copy()
    with quiet():
        LMM(y, K, kva, kve)
    np.testing.assert_array_equal(kva, before)


def test_bug12_calculatekinship_does_not_mutate_w():
    """Regression (bug 12, fixed in Phase 5.2)."""
    G, _, _, _ = toy()
    W = G.copy()
    W[0, 0] = np.nan
    before = W.copy()
    calculate_kinship(W)
    np.testing.assert_array_equal(W, before)


# ---------------------------------------------------------------------------
# bug 13: zero-variance filters
# ---------------------------------------------------------------------------


def test_bug13_one_filter_for_both_branches(tmp_path):
    """Regression (bug 13, fixed in Phase 5.2), --removeMissingGenotypes,
    EMMA dosages: two SNPs with the same tiny variance, one with a missing
    call.  Both branches use the same estimability check, so both are tested
    (the one with the missing call used to be dropped by var <= 1e-6)."""
    rng = np.random.default_rng(6)
    n = 30
    tiny = np.zeros(n)
    tiny[:3] = 1e-3  # var ~ 9e-8
    a = tiny.copy()
    b = tiny.copy()
    b[10] = np.nan
    other = rng.binomial(2, 0.4, size=n).astype(float)
    snps = str(tmp_path / "s.txt")
    with open(snps, "w") as f:
        for row in (other, a, b):
            f.write(" ".join("NA" if np.isnan(v) else "%g" % v for v in row) + "\n")
    with open(str(tmp_path / "p.txt"), "w") as f:
        f.write(" ".join("%.6f" % v for v in rng.standard_normal(n)) + "\n")
    g_kin = rng.binomial(2, 0.4, size=(n, 60)).astype(float)
    np.savetxt(str(tmp_path / "k"), calculate_kinship(g_kin))
    out = str(tmp_path / "o")
    run_gwas(
        [
            "--emma-geno",
            snps,
            "--emma-pheno",
            str(tmp_path / "p.txt"),
            "--kinship",
            str(tmp_path / "k"),
            "--remove-missing-genotypes",
            "-o",
            out,
        ]
    )
    r = read_out(out)
    assert np.isfinite(r[1]).all()  # complete tiny-variance SNP: tested
    assert np.isfinite(r[2]).all()  # same SNP with one missing call: tested too
    assert "var() <= 1e-6" not in inspect.getsource(cli_gwas)


# ---------------------------------------------------------------------------
# bug 15: return types
# ---------------------------------------------------------------------------


def test_bug15_fit_and_association_return_scalars():
    """Regression (the rest of bug 15, fixed in Phase 5b): sigma, Q and
    beta_var are floats, not 1x1 arrays; beta stays a (q, 1) column."""
    G, K, y, _ = toy()
    with quiet():
        L = LMM(y, K)
    h, beta, sigma, ll = L.fit()
    assert type(sigma) is float and type(ll) is float and np.shape(beta) == (1, 1)
    _, s2, Q, _, _ = L.ml_solution(h, L.x0t)
    assert type(s2) is float and type(Q) is float and s2 == sigma
    res = L.log_likelihood(h)
    assert type(res.ll) is float and type(res.sigma) is float and res.ll == ll
    _, _, b, bvar = L.association(G[:, :1], return_beta=True)
    assert type(bvar) is float and np.ndim(b) == 0
    # a 1-D SNP gives the same numbers as the (n, 1) column
    assert L.association(G[:, 0], return_beta=True) == (*L.association(G[:, :1])[:2], b, bvar)


def test_bug15_gwas_return_type_consistent():
    """Regression (bug 15, the GWAS part, fixed in Phase 5.2): always lists."""
    G, K, y, _ = toy()
    with quiet():
        a = gwas(np.full(len(y), np.nan), G[:, :3], K)
        b = gwas(y, G[:, :3], K)
    assert type(a[0]) is type(b[0]) is list
    assert len(a[0]) == 3 and np.isnan(a[0]).all()


# ---------------------------------------------------------------------------
# bug 16: --kinship (formerly --kfile) read unchecked
# ---------------------------------------------------------------------------


def test_bug16_headered_kinship_rejected(tmp_path):
    """Regression (bug 16, fixed in Phase 5.1): on numpy < 2.3 this used to
    give a silent all-zero K."""
    fn = str(tmp_path / "k.txt")
    with open(fn, "w") as f:
        f.write("# a header line\n1 0\n0 1\n")
    with pytest.raises(ValueError, match="not a number"):
        cli_gwas.read_kinship_file(fn, 2)


def test_bug16_wrong_size_kinship_rejected(tmp_path):
    """Regression (bug 16, fixed in Phase 5.1): 9 numbers for a 4 x 4 K used
    to be zero-filled into a scrambled matrix."""
    fn = str(tmp_path / "k.txt")
    np.savetxt(fn, np.eye(3))
    with pytest.raises((ValueError, SystemExit)):
        cli_gwas.read_kinship_file(fn, 4)


# ---------------------------------------------------------------------------
# bug 17: ML with the centred GRM (model property, not a code bug)
# ---------------------------------------------------------------------------


def test_bug17_ml_profile_unbounded_with_centred_grm_m_gt_n():
    G, K, y, _ = toy(n=40, m=200, seed=7)
    assert abs(K.sum(axis=1)).max() < 1e-10  # K 1 = 0 (centred SNPs)
    fml = oracle.fit(y, K, reml=False, n_grid=51)
    fre = oracle.fit(y, K, reml=True, n_grid=51)
    assert fml.ml_unbounded and not fre.ml_unbounded
    # pylmm's clamp turns the divergence into a bounded spike beyond its 0.99 cap
    with quiet():
        L = LMM(y, K)
    ll = [L.log_likelihood(h, reml=False)[0] for h in (0.99, 0.9999, 1 - 1e-9)]
    assert ll[0] < ll[1] < ll[2] < np.inf
    # a larger null space (m < n - 1): bounded
    G2, K2, y2, _ = toy(n=40, m=15, seed=7)
    assert not oracle.fit(y2, K2, reml=False, n_grid=51).ml_unbounded


# ---------------------------------------------------------------------------
# Phase 5.2 review fixes
# ---------------------------------------------------------------------------


def test_pheno_and_covariate_files_accept_fid_iid_header(bed_toy):
    """A first line starting FID IID (any case) is skipped."""
    d = bed_toy["dir"]
    n, y = bed_toy["n"], bed_toy["y"]
    c = np.cos(np.arange(n))
    write_phenos(str(d / "plain.ph"), y)
    with open(str(d / "head.ph"), "w") as f:
        f.write("fid IID pheno\n" + open(str(d / "plain.ph")).read())
    with open(str(d / "head.cov"), "w") as f:
        f.write("FID\tIID\tc1\n")
        for i in range(n):
            f.write("F%d I%d %.10f\n" % (i, i, c[i]))
    with open(str(d / "plain.cov"), "w") as f:
        for i in range(n):
            f.write("F%d I%d %.10f\n" % (i, i, c[i]))
    base, kin = bed_toy["base"], bed_toy["kin"]
    run_gwas(
        [
            "--bfile",
            base,
            "--kinship",
            kin,
            "--pheno",
            str(d / "plain.ph"),
            "--covar",
            str(d / "plain.cov"),
            "-o",
            str(d / "a.out"),
        ]
    )
    err = run_gwas_err(
        [
            "--bfile",
            base,
            "--kinship",
            kin,
            "--pheno",
            str(d / "head.ph"),
            "--covar",
            str(d / "head.cov"),
            "-o",
            str(d / "b.out"),
        ]
    )
    assert err.count("warning") == 0
    np.testing.assert_allclose(read_out(str(d / "b.out")), read_out(str(d / "a.out")), rtol=1e-12)
    # a non-numeric first line that is not a FID IID header is still an error
    with open(str(d / "bad.ph"), "w") as f:
        f.write("ID1 ID2 pheno\n" + open(str(d / "plain.ph")).read())
    run_gwas(
        ["--bfile", base, "--kinship", kin, "--pheno", str(d / "bad.ph"), "-o", str(d / "x")], rc=2
    )


def test_cli_eigen_ignored_is_reported_unconditionally(bed_toy):
    d = bed_toy["dir"]
    write_phenos(str(d / "na.ph"), bed_toy["y"], na=(5,))
    K = np.loadtxt(bed_toy["kin"])
    kva, kve = linalg.eigh(K)
    np.savez(str(d / "e.npz"), Kva=kva, Kve=kve)
    err = run_gwas_err(
        [
            "--bfile",
            bed_toy["base"],
            "--kinship",
            bed_toy["kin"],
            "--eigen",
            str(d / "e.npz"),
            "--pheno",
            str(d / "na.ph"),
            "-o",
            str(d / "o"),
        ]
    )
    assert "pylmm: warning: --eigen is ignored: 1 individuals were dropped" in err
