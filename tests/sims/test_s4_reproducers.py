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

from pylmm import input as pinput
from pylmm import lmm as pylmm_lmm
from pylmm import misc
from pylmm.cli import gwas as cli_gwas
from pylmm.cli import kinship as cli_kinship
from pylmm.lmm import GWAS, LMM, calculateKinship
from sims import oracle

try:                                    # scipy >= 1.11 (not exported publicly)
    from scipy.optimize._optimize import BracketError
    SCIPY_HAS_BRACKETERROR = True
except ImportError:                     # pragma: no cover
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
    K = calculateKinship(G.copy())
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
            f.write((c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).astype(np.uint8).tobytes())
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
    """A 40 x 25 BED (one missing call), its pylmmKinship kinship and phenos."""
    rng = np.random.default_rng(1)
    n, m = 40, 25
    G = rng.binomial(2, 0.4, size=(n, m)) / 2.0
    G[3, 2] = np.nan
    base = str(tmp_path / "toy")
    write_bed(base, G)
    y = rng.standard_normal(n)
    write_phenos(base + ".phenos", y)
    kin = str(tmp_path / "toy.kin")
    with quiet():
        cli_kinship.main(["--bfile", base, kin])
    return {"base": base, "kin": kin, "G": G, "y": y, "n": n, "m": m, "dir": tmp_path}


def run_gwas(argv):
    with quiet():
        cli_gwas.main(argv)


def read_out(fn):
    rows = [line.rstrip("\n").split("\t") for line in open(fn)][1:]
    return np.array([[float(x) for x in r[1:]] for r in rows])


def fake_lmm(ll):
    """An LMM whose likelihood is the given function of h (for getMax logic)."""
    G, K, y, _ = toy(n=20, m=10)

    class Fake(LMM):
        def LL(self, h, X=None, stack=True, REML=False):
            return (np.float64(ll(h)), np.zeros((1, 1)), np.ones((1, 1)), np.eye(1))
    with quiet():
        return Fake(y, K)


# ---------------------------------------------------------------------------
# bug 1: the null fit ignores REML
# ---------------------------------------------------------------------------

def _ml_bounded_case():
    # m < n - 1: K's null space is larger than span(1), so the ML profile is bounded
    G, K, y, rng = toy(n=80, m=25, seed=3)
    Gt = rng.binomial(2, 0.3, size=(80, 6)).astype(float)
    return Gt, K, y


def test_bug1_current_gwas_reml_flag_does_not_change_t_or_p():
    Gt, K, y = _ml_bounded_case()
    with quiet():
        t1, p1 = GWAS(y, Gt, K, REML=True)
        t0, p0 = GWAS(y, Gt, K, REML=False)
    np.testing.assert_array_equal(t0, t1)
    np.testing.assert_array_equal(p0, p1)


def test_bug1_current_cli_refit_fits_ml_but_library_refit_fits_reml():
    """CLI --refit calls L.fit(X=x, REML=options.REML) (ML by default);
    lmm.GWAS(refit=True, REML=False) calls L.fit(X=x) (REML)."""
    Gt, K, y = _ml_bounded_case()
    src = inspect.getsource(pylmm_lmm.GWAS)
    assert "L.fit(X=x)" in src and "Ls.fit(X=xs)" in src
    assert "L.fit(X=x,REML=options.REML)" in inspect.getsource(cli_gwas.main)
    with quiet():
        L = LMM(y, K)
    x = Gt[:, :1]
    L.fit(X=x, REML=False)
    h_ml = L.optH
    L.fit(X=x)
    h_reml = L.optH
    assert abs(h_ml - h_reml) > 1e-3


@pytest.mark.xfail(strict=True, reason="bug 1: GWAS(REML=False) uses a REML null h, not the ML Wald")
def test_bug1_gwas_ml_equals_textbook_ml_wald():
    """An ML null fit + ML Wald, isolated from bug 9: the reference keeps
    pylmm's sigma^2 = Q/(n-q) (``sigma_convention='pylmm'``), so fixing bug 1
    alone flips this test (an ML-fitted pylmm matches it to ~5e-8 here, while
    the textbook Q/n would still differ by sqrt(n/(n-q)) = 1.0127; that is
    bug 9's own reproducer below)."""
    Gt, K, y = _ml_bounded_case()
    with quiet():
        t0, _ = GWAS(y, Gt, K, REML=False)
    w = [oracle.wald(y, K, Gt[:, j], null_reml=False, sigma_convention="pylmm").t
         for j in range(Gt.shape[1])]
    np.testing.assert_allclose(t0, w, atol=1e-4)


# ---------------------------------------------------------------------------
# bugs 2, 3, 4: the h search
# ---------------------------------------------------------------------------

def test_bug2_current_getmax_skips_grid_index_98():
    L = fake_lmm(lambda h: -(h - 0.981) ** 2)        # single peak, nearest grid point 0.98
    h = L.fit()[0]
    assert h == 0.99                                  # endpoint fallback


@pytest.mark.xfail(strict=True, reason="bug 2: range(1, n-2) never tests grid index n-2 (h = 0.98)")
def test_bug2_getmax_finds_peak_at_098():
    L = fake_lmm(lambda h: -(h - 0.981) ** 2)
    assert abs(L.fit()[0] - 0.981) < 1e-4


def test_bug3_current_h_grid_stops_at_099():
    L = fake_lmm(lambda h: h)                         # increasing: maximum at h = 1
    assert L.fit()[0] == 0.99
    assert L.H.max() == 0.99


@pytest.mark.xfail(strict=True, reason="bug 3: the h grid excludes 1.0")
def test_bug3_h_search_reaches_one():
    L = fake_lmm(lambda h: h)
    assert L.fit()[0] == 1.0


@pytest.mark.xfail(strict=True, reason="bug 3: misc.fitTwo's w grid excludes 1.0 (K1 alone)")
def test_bug3_fittwo_can_select_k1_alone():
    rng = np.random.default_rng(5)
    n = 50
    G1 = rng.binomial(2, 0.4, size=(n, 200)).astype(float)
    G2 = rng.binomial(2, 0.4, size=(n, 200)).astype(float)
    K1, K2 = calculateKinship(G1), calculateKinship(G2)
    d, V = linalg.eigh(K1)
    y = V @ (np.sqrt(np.clip(d, 0, None)) * rng.standard_normal(n)) * 3 + 0.1 * rng.standard_normal(n)
    with quiet():
        h1, h2, e, beta, sigma, LL = misc.fitTwo(y, K1, K2, wgrids=20)
    assert h2 == 0.0


def test_bug4_ll_is_nan_above_one_but_ll_brent_guards_both_ends():
    G, K, y, _ = toy()
    with quiet():
        L = LMM(y, K)
    with np.errstate(invalid="ignore"):
        assert np.isnan(L.LL(1.02)[0])            # log(h*Kva + 1 - h) of a negative number
    assert L.LL_brent(-0.1) == 1e6
    assert L.LL_brent(1.02) == 1e6                 # fixed in Phase 5.1: h > 1 guarded too


def _peak_0972(h):
    if h > 1:
        return np.nan                              # what pylmm's LL gives above 1
    return -(h - 0.972) ** 2


@pytest.mark.skipif(not SCIPY_HAS_BRACKETERROR, reason="scipy < 1.11 returns instead of raising")
def test_bug4_old_unbounded_brent_crashed_with_bracketerror():
    """Why this case is the regression test: peak at grid index 97 with
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


def _high_h2_case():
    """A real GRM (n = 60, m = 30) and y simulated at h2 = 0.97 whose REML
    grid peak is at index 97 with LL(0.98) > LL(0.96) (seed found by search)."""
    rng = np.random.default_rng(2)
    n, m = 60, 30
    G = rng.binomial(2, 0.3, size=(n, m)).astype(float)
    K = calculateKinship(G.copy())
    d, V = linalg.eigh(K)
    u = V @ (np.sqrt(np.clip(d, 0, None)) * rng.standard_normal(n))
    y = np.sqrt(0.97) * u + np.sqrt(0.03) * rng.standard_normal(n)
    return y, K


def test_bug4_high_h2_case_matches_oracle():
    """Regression (bug 4, fixed in Phase 5.1) on real likelihoods, null fit
    and per-SNP refit: pylmm matches the oracle with pylmm's conventions
    (eig_clamp=1e-6, the pylmm grid and search) to ~1e-6.  On scipy >= 1.11
    the old unbounded Brent raised BracketError on this case."""
    from scipy import optimize
    y, K = _high_h2_case()
    with quiet():
        L = LMM(y, K)
    LLs = np.array([L.LL(h, REML=True)[0] for h in np.arange(100) / 100.0])
    assert LLs[96] < LLs[97] > LLs[98] and LLs[98] > LLs[96]
    if SCIPY_HAS_BRACKETERROR:
        old = lambda h: 1e6 if h < 0 else -L.LL(h, stack=False, REML=True)[0]  # noqa: E731
        with pytest.raises(BracketError), np.errstate(all="ignore"):
            optimize.brent(old, brack=(0.96, 0.98))
    f = oracle.fit(y, K, reml=True, eig_clamp=1e-6, h_grid="pylmm")
    assert 0.96 < f.h < 0.98
    h = float(np.sum(L.fit(REML=True)[0]))
    assert abs(h - f.h) < 1e-6
    # the 0.99 cap does not bind: the clamped global optimum is the same h
    assert abs(h - oracle.fit(y, K, reml=True, eig_clamp=1e-6).h) < 1e-6
    # per-SNP refit path (fit(X=x)) on the same data: no crash, h in [0, 1]
    x = np.random.default_rng(3).binomial(2, 0.3, size=(len(y), 1)).astype(float)
    with quiet():
        hx = float(np.sum(L.fit(X=x, REML=True)[0]))
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
    run_gwas(["--bfile", bed_toy["base"], "--kfile", bed_toy["kin"], out])
    assert read_out(out).shape == (bed_toy["m"], 4)


@pytest.mark.xfail(strict=True, raises=NameError, reason="bug 6: --kfile2 uses undefined LMM_withK2")
def test_bug6_kfile2(bed_toy):
    out = str(bed_toy["dir"] / "k2.out")
    run_gwas(["--bfile", bed_toy["base"], "--kfile", bed_toy["kin"], "--kfile2", bed_toy["kin"], out])


@pytest.mark.xfail(strict=True, raises=(ValueError, OSError),
                   reason="bug 7: --eigen np.load()s <base>.Kva; pylmmKinship -e writes text <out>.kva")
def test_bug7_eigen_reads_pylmmkinship_output(bed_toy):
    with quiet():
        cli_kinship.main(["--bfile", bed_toy["base"], "-e", "x", bed_toy["kin"]])
    assert os.path.exists(bed_toy["kin"] + ".kva")
    out = str(bed_toy["dir"] / "eig.out")
    run_gwas(["--bfile", bed_toy["base"], "--kfile", bed_toy["kin"], "--eigen", bed_toy["kin"], out])


def test_bug14_current_no_pheno_file_raises_typeerror(bed_toy):
    os.remove(bed_toy["base"] + ".phenos")
    with pytest.raises(TypeError):
        run_gwas(["--bfile", bed_toy["base"], "--kfile", bed_toy["kin"], str(bed_toy["dir"] / "x")])


@pytest.mark.xfail(strict=True, raises=TypeError, reason="bug 14: os.path.isfile(None) instead of parser.error")
def test_bug14_no_pheno_file_is_a_usage_error(bed_toy):
    os.remove(bed_toy["base"] + ".phenos")
    with pytest.raises(SystemExit):
        run_gwas(["--bfile", bed_toy["base"], "--kfile", bed_toy["kin"], str(bed_toy["dir"] / "x")])


# ---------------------------------------------------------------------------
# bug 8: kinship normalisation
# ---------------------------------------------------------------------------

def test_bug8_current_monomorphic_snps_shrink_k():
    G, _, _, _ = toy(n=30, m=40)
    K = calculateKinship(G.copy())
    K_mono = calculateKinship(np.hstack([G, np.zeros((30, 10))]))
    np.testing.assert_allclose(K_mono, K * 40 / 50, rtol=1e-12)


@pytest.mark.xfail(strict=True, reason="bug 8: calculateKinship divides by m including dropped monomorphic SNPs")
def test_bug8_monomorphic_snps_do_not_change_k():
    G, _, _, _ = toy(n=30, m=40)
    np.testing.assert_allclose(calculateKinship(np.hstack([G, np.zeros((30, 10))])),
                               calculateKinship(G.copy()), rtol=1e-12)


def test_bug8_current_kinship_cli_counts_monomorphic_snps(tmp_path):
    rng = np.random.default_rng(2)
    G = rng.binomial(2, 0.4, size=(24, 12)) / 2.0
    base = str(tmp_path / "mono")
    write_bed(base, np.hstack([G, np.zeros((24, 4))]))           # 4 monomorphic SNPs
    with quiet():
        cli_kinship.main(["--bfile", base, str(tmp_path / "k")])
    K_cli = np.loadtxt(str(tmp_path / "k"))
    np.testing.assert_allclose(K_cli, calculateKinship(G.copy()) * 12 / 16, rtol=1e-10)


@pytest.mark.xfail(strict=True, reason="bug 8: lmm.calculateKinship and pylmmKinship normalise missing data differently")
def test_bug8_two_kinships_agree_with_missing_data(tmp_path):
    rng = np.random.default_rng(3)
    G = rng.binomial(2, 0.4, size=(24, 12)) / 2.0
    G[rng.random(G.shape) < 0.1] = np.nan
    base = str(tmp_path / "miss")
    write_bed(base, G)
    with quiet():
        cli_kinship.main(["--bfile", base, str(tmp_path / "k")])
    np.testing.assert_allclose(np.loadtxt(str(tmp_path / "k")), calculateKinship(G.copy()), atol=1e-8)


# ---------------------------------------------------------------------------
# bug 9: ML sigma
# ---------------------------------------------------------------------------

def test_bug9_current_convention_ml_wald_uses_q_over_n_minus_q():
    Gt, K, y = _ml_bounded_case()
    with quiet():
        L = LMM(y, K)
    L.fit(REML=False)
    x = Gt[:, :1]
    t_ml, _ = L.association(x, REML=False)
    w9 = oracle.wald(y, K, x, h=L.optH, null_reml=False, sigma_convention="pylmm")
    wtb = oracle.wald(y, K, x, h=L.optH, null_reml=False, sigma_convention="textbook")
    assert t_ml == pytest.approx(w9.t, rel=1e-5)
    n, p = len(y), 2
    assert wtb.t / w9.t == pytest.approx(np.sqrt(n / (n - p)), rel=1e-10)


# ---------------------------------------------------------------------------
# bug 10: phenotype file missing individuals
# ---------------------------------------------------------------------------

def test_bug10_current_getphenos_shifts_rows(bed_toy):
    fn = str(bed_toy["dir"] / "miss.ph")
    write_phenos(fn, bed_toy["y"], skip=(5, 17))
    with quiet():
        IN = pinput.plink(bed_toy["base"], type="b", phenoFile=fn)
    assert IN.phenos.shape[0] == bed_toy["n"] - 2 and len(IN.indivs) == bed_toy["n"]
    # by position, individual I5 is now paired with I6's phenotype
    assert IN.indivs[5] == ("F5", "I5")
    assert IN.phenos[5, 0] == pytest.approx(bed_toy["y"][6])


def test_bug10_current_cli_crashes_on_pheno_missing_individuals(bed_toy):
    fn = str(bed_toy["dir"] / "miss.ph")
    write_phenos(fn, bed_toy["y"], skip=(5, 17))
    with pytest.raises(ValueError, match="reshape"):
        run_gwas(["--bfile", bed_toy["base"], "--kfile", bed_toy["kin"], "--phenofile", fn,
                  str(bed_toy["dir"] / "x")])
    write_phenos(fn, bed_toy["y"], skip=(5, 17), na=(9,))
    with pytest.raises(IndexError):
        run_gwas(["--bfile", bed_toy["base"], "--kfile", bed_toy["kin"], "--phenofile", fn,
                  str(bed_toy["dir"] / "x")])


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="bug 10: individuals absent from the pheno file are skipped, not set to NA")
def test_bug10_absent_individuals_treated_as_missing(bed_toy):
    d = bed_toy["dir"]
    write_phenos(str(d / "na.ph"), bed_toy["y"], na=(5, 17))
    write_phenos(str(d / "skip.ph"), bed_toy["y"], skip=(5, 17))
    run_gwas(["--bfile", bed_toy["base"], "--kfile", bed_toy["kin"], "--phenofile", str(d / "na.ph"),
              str(d / "na.out")])
    run_gwas(["--bfile", bed_toy["base"], "--kfile", bed_toy["kin"], "--phenofile", str(d / "skip.ph"),
              str(d / "skip.out")])
    np.testing.assert_allclose(read_out(str(d / "skip.out")), read_out(str(d / "na.out")))


# ---------------------------------------------------------------------------
# bug 11: an all-missing SNP
# ---------------------------------------------------------------------------

def test_bug11_current_normalize_all_missing_returns_empty():
    g = pinput.plink.normalizeGenotype(None, np.full(5, np.nan))
    assert g.shape == (0,)


@pytest.mark.xfail(strict=True, raises=IndexError, reason="bug 11: all-missing SNP -> empty array -> crash")
def test_bug11_cli_gwas_all_missing_snp(tmp_path):
    rng = np.random.default_rng(4)
    G = rng.binomial(2, 0.4, size=(24, 6)) / 2.0
    G[:, 3] = np.nan
    base = str(tmp_path / "allmiss")
    write_bed(base, G)
    write_phenos(base + ".phenos", rng.standard_normal(24))
    np.savetxt(str(tmp_path / "k"), calculateKinship(G[:, [0, 1, 2, 4, 5]].copy()))
    run_gwas(["--bfile", base, "--kfile", str(tmp_path / "k"), str(tmp_path / "o")])
    assert np.isnan(read_out(str(tmp_path / "o"))[3]).all()


@pytest.mark.xfail(strict=True, raises=ValueError, reason="bug 11: pylmmKinship crashes on an all-missing SNP")
def test_bug11_cli_kinship_all_missing_snp(tmp_path):
    rng = np.random.default_rng(4)
    G = rng.binomial(2, 0.4, size=(24, 6)) / 2.0
    G[:, 3] = np.nan
    base = str(tmp_path / "allmiss")
    write_bed(base, G)
    with quiet(), np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        cli_kinship.main(["--bfile", base, str(tmp_path / "k")])


# ---------------------------------------------------------------------------
# bug 12: callers' arrays mutated
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="bug 12: LMM.__init__ clamps the caller's Kva in place")
def test_bug12_lmm_does_not_mutate_kva():
    G, K, y, _ = toy()
    Kva, Kve = linalg.eigh(K)
    before = Kva.copy()
    with quiet():
        LMM(y, K, Kva, Kve)
    np.testing.assert_array_equal(Kva, before)


@pytest.mark.xfail(strict=True, reason="bug 12: calculateKinship imputes/standardises its argument in place")
def test_bug12_calculatekinship_does_not_mutate_w():
    G, _, _, _ = toy()
    W = G.copy()
    W[0, 0] = np.nan
    before = W.copy()
    calculateKinship(W)
    np.testing.assert_array_equal(W, before)


# ---------------------------------------------------------------------------
# bug 13: zero-variance filters
# ---------------------------------------------------------------------------

def test_bug13_current_filters_differ_between_branches(tmp_path):
    """--removeMissingGenotypes, EMMA dosages: two SNPs with the same tiny
    variance; the one with a missing call is dropped (var <= 1e-6), the
    complete one is tested (var == 0 check)."""
    rng = np.random.default_rng(6)
    n = 30
    tiny = np.zeros(n)
    tiny[:3] = 1e-3                                   # var ~ 9e-8
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
    Gk = rng.binomial(2, 0.4, size=(n, 60)).astype(float)
    np.savetxt(str(tmp_path / "k"), calculateKinship(Gk))
    out = str(tmp_path / "o")
    run_gwas(["--emmaSNP", snps, "--emmaPHENO", str(tmp_path / "p.txt"), "--kfile",
              str(tmp_path / "k"), "--removeMissingGenotypes", out])
    r = read_out(out)
    assert np.isfinite(r[1]).all()          # complete tiny-variance SNP: tested
    assert np.isnan(r[2]).all()             # same SNP with one missing call: skipped
    assert "xs.var() <= 1e-6" in inspect.getsource(cli_gwas.main)
    assert "x.var() == 0" in inspect.getsource(cli_gwas.main)


# ---------------------------------------------------------------------------
# bug 15: return types
# ---------------------------------------------------------------------------

def test_bug15_current_return_types():
    G, K, y, _ = toy()
    with quiet():
        r_all_missing = GWAS(np.full(len(y), np.nan), G[:, :3], K)
        r_normal = GWAS(y, G[:, :3], K)
        L = LMM(y, K)
    assert isinstance(r_all_missing[0], np.ndarray) and isinstance(r_normal[0], list)
    h, beta, sigma, ll = L.fit()
    assert np.shape(sigma) == (1, 1) and np.shape(beta) == (1, 1)
    _, _, b, bvar = L.association(G[:, :1], returnBeta=True)
    assert np.shape(bvar) == (1, 1) and np.ndim(b) == 0


@pytest.mark.xfail(strict=True, reason="bug 15: GWAS returns lists normally but arrays when all Y are missing")
def test_bug15_gwas_return_type_consistent():
    G, K, y, _ = toy()
    with quiet():
        a = GWAS(np.full(len(y), np.nan), G[:, :3], K)
        b = GWAS(y, G[:, :3], K)
    assert type(a[0]) is type(b[0])


# ---------------------------------------------------------------------------
# bug 16: --kfile read unchecked
# ---------------------------------------------------------------------------

def test_bug16_headered_kinship_rejected(tmp_path):
    """Regression (bug 16, fixed in Phase 5.1): on numpy < 2.3 this used to
    give a silent all-zero K."""
    fn = str(tmp_path / "k.txt")
    with open(fn, "w") as f:
        f.write("# a header line\n1 0\n0 1\n")
    with pytest.raises(ValueError, match="not a number"):
        cli_gwas.readKinshipFile(fn, 2)


def test_bug16_wrong_size_kinship_rejected(tmp_path):
    """Regression (bug 16, fixed in Phase 5.1): 9 numbers for a 4 x 4 K used
    to be zero-filled into a scrambled matrix."""
    fn = str(tmp_path / "k.txt")
    np.savetxt(fn, np.eye(3))
    with pytest.raises((ValueError, SystemExit)):
        cli_gwas.readKinshipFile(fn, 4)


# ---------------------------------------------------------------------------
# bug 17: ML with the centred GRM (model property, not a code bug)
# ---------------------------------------------------------------------------

def test_bug17_ml_profile_unbounded_with_centred_grm_m_gt_n():
    G, K, y, _ = toy(n=40, m=200, seed=7)
    assert abs(K.sum(axis=1)).max() < 1e-10          # K 1 = 0 (centred SNPs)
    fml = oracle.fit(y, K, reml=False, n_grid=51)
    fre = oracle.fit(y, K, reml=True, n_grid=51)
    assert fml.ml_unbounded and not fre.ml_unbounded
    # pylmm's clamp turns the divergence into a bounded spike beyond its 0.99 cap
    with quiet():
        L = LMM(y, K)
    ll = [L.LL(h, REML=False)[0] for h in (0.99, 0.9999, 1 - 1e-9)]
    assert ll[0] < ll[1] < ll[2] < np.inf
    # a larger null space (m < n - 1): bounded
    G2, K2, y2, _ = toy(n=40, m=15, seed=7)
    assert not oracle.fit(y2, K2, reml=False, n_grid=51).ml_unbounded
