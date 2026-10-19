"""S4: bug confirmation -- practical impact of pylmm's suspected bugs.

The bug ledger (verdicts, reproducers, severity, fix sketches) is
``sims/studies/s4_bug_confirmation.md``; the deterministic reproducers are
``tests/sims/test_s4_reproducers.py``.  This module measures the bugs whose
impact depends on the data regime.  Bug numbers follow
docs/phase1-baseline.md (17 and 18 are new, see the .md).

Experiments (the ``exp`` param; one cell = one experiment x regime):

``remlml``   bugs 1, 9, 17.  The CLI default (``GWAS(REML=False)``) against
             pylmm REML, a textbook ML Wald (oracle ML h, sigma^2 = Q/n), the
             same with pylmm's Q/(n-q) sigma (isolates bug 9) and a textbook ML
             LRT; plus the ``--refit`` path, where the CLI fits h by ML per SNP
             but the library's ``GWAS(refit=True)`` fits REML.
``hgrid``    bugs 2, 3, 4.  High h2.  pylmm's REML search (instrumented: hits
             of the 0.99 cap, a skipped local max at grid index 98, Brent
             evaluations above h = 1 and scipy BracketError crashes) against
             the dense oracle (fine grid over [0, 1], with/without the
             eigenvalue clamp); t / -log10 p at pylmm's h vs the oracle's.
``kinship``  bug 8.  K from ``lmm.calculateKinship`` vs the pylmmKinship CLI
             normalisation (``input.normalizeGenotype``) vs the denominator
             fixed (divide by the SNPs actually used) vs the complete-data
             truth (synthetic); effect on h-hat and t.
``mlcent``   bug 17.  ML on the centred GRM: how often the textbook ML
             profile is unbounded (m > n), and what pylmm's ML h does.
``bednull``  item 18.  Null calibration on the real BED genotypes over many
             replicates: pylmm vs the textbook oracle vs the *exact* test at
             the true h (exactly t-distributed by construction), replicate
             lambda_GC spread vs the effective number of tests; a causal-SNP
             cell (S3-like) to show LD tagging, with a conditional test.

Run:  just sim s4_bug_confirmation --quick
      just sim s4_bug_confirmation --jobs 6           # full grid (~10 min)
      S4_ONLY=bednull just sim s4_bug_confirmation --jobs 2
          # only the item-18 cells of the full grid (same cell ids, so a
          # later full run skips them)
      just sim-report s4_bug_confirmation
"""

from __future__ import annotations

import contextlib
import io
import json
import os

import numpy as np
from scipy import linalg, stats

from sims import genotypes, kinship, methods, metrics, oracle, phenotypes

NAME = "s4_bug_confirmation"
REPS = 20
QUICK_REPS = 30
ALPHAS = (0.05, 1e-2, 1e-3, 1e-4)
MIN_MAF = 0.05


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------


def _full_grid():
    g = []
    # bugs 1 / 9 / 17: n = 300; 'unrel_m200' has m_kin < n (ML bounded)
    for pop, mk in (("unrelated", 200), ("unrelated", 2000), ("family", 2000), ("bed", 2000)):
        for h2 in (0.2, 0.5, 0.8):
            for blk in (0,):
                g.append(
                    {
                        "exp": "remlml",
                        "pop": pop,
                        "n": 300,
                        "m_kin": mk,
                        "m_test": 200,
                        "h2": h2,
                        "block": blk,
                    }
                )
    # bugs 2 / 3 / 4: high h2
    for pop, n in (("family", 200), ("unrelated", 300), ("bed", 300)):
        for h2 in (0.8, 0.9, 0.95, 0.99):
            for blk in (0, 1):
                g.append(
                    {
                        "exp": "hgrid",
                        "pop": pop,
                        "n": n,
                        "m_kin": 2000,
                        "m_test": 200,
                        "h2": h2,
                        "block": blk,
                    }
                )
    # bug 8
    for pop, n, miss, mono in (
        ("hmdp", 94, 0.0, 0.0),
        ("bed", 300, 0.0, 0.0),
        ("unrelated", 300, 0.02, 0.25),
        ("unrelated", 300, 0.10, 0.25),
    ):
        for h2 in (0.5, 0.8):
            g.append(
                {
                    "exp": "kinship",
                    "pop": pop,
                    "n": n,
                    "m_kin": 2000,
                    "m_test": 200,
                    "h2": h2,
                    "miss": miss,
                    "mono": mono,
                    "block": 0,
                }
            )
    # bug 17
    for pop, mk in (
        ("unrelated", 100),
        ("unrelated", 150),
        ("unrelated", 400),
        ("unrelated", 2000),
        ("bed", 2000),
    ):
        for h2 in (0.2, 0.5, 0.8):
            g.append(
                {
                    "exp": "mlcent",
                    "pop": pop,
                    "n": 200,
                    "m_kin": mk,
                    "m_test": 0,
                    "h2": h2,
                    "block": 0,
                }
            )
    g += _bednull_grid()
    return g


def _bednull_grid():
    g = []
    # (pop, n, h2, n_causal, blocks): blocks x REPS replicates per cell
    for pop, n, h2, nc, nb in (
        ("bed", 300, 0.0, 0, 4),
        ("bed", 300, 0.5, 0, 4),
        ("bed", 300, 0.95, 0, 4),
        ("bed", 600, 0.0, 0, 4),
        ("bed", 600, 0.5, 0, 2),
        ("bed", 600, 0.95, 0, 2),
        ("bed", 600, 0.5, 5, 2),
        ("unrelated", 600, 0.5, 0, 2),
    ):
        for blk in range(nb):
            g.append(
                {
                    "exp": "bednull",
                    "pop": pop,
                    "n": n,
                    "m_kin": 6000,
                    "m_test": 5000,
                    "h2": h2,
                    "n_causal": nc,
                    "block": blk,
                }
            )
    return g


def _quick_grid():
    """Reduced but informative grid (~1.5-2 min at --jobs 2): one or two
    regimes per experiment at full per-cell size, few replicates.  The S4
    ledger quotes these numbers where the full grid has not been run."""
    q = []
    for pop, mk in (("unrelated", 200), ("unrelated", 2000), ("bed", 2000)):
        q.append(
            {
                "exp": "remlml",
                "pop": pop,
                "n": 300,
                "m_kin": mk,
                "m_test": 200,
                "h2": 0.5,
                "block": 0,
            }
        )
    for pop, n in (("family", 200), ("bed", 300)):
        q.append(
            {
                "exp": "hgrid",
                "pop": pop,
                "n": n,
                "m_kin": 2000,
                "m_test": 200,
                "h2": 0.95,
                "block": 0,
            }
        )
    q.append(
        {
            "exp": "kinship",
            "pop": "hmdp",
            "n": 94,
            "m_kin": 2000,
            "m_test": 200,
            "h2": 0.5,
            "miss": 0.0,
            "mono": 0.0,
            "block": 0,
        }
    )
    q.append(
        {
            "exp": "kinship",
            "pop": "unrelated",
            "n": 300,
            "m_kin": 2000,
            "m_test": 200,
            "h2": 0.5,
            "miss": 0.10,
            "mono": 0.25,
            "block": 0,
        }
    )
    for pop, mk in (("unrelated", 100), ("unrelated", 2000), ("bed", 2000)):
        q.append(
            {"exp": "mlcent", "pop": pop, "n": 200, "m_kin": mk, "m_test": 0, "h2": 0.5, "block": 0}
        )
    q.append(
        {
            "exp": "bednull",
            "pop": "bed",
            "n": 200,
            "m_kin": 1000,
            "m_test": 300,
            "h2": 0.5,
            "n_causal": 0,
            "block": 0,
        }
    )
    return q


def grid(quick):
    if quick:
        return _quick_grid()
    only = os.environ.get("S4_ONLY")
    g = _full_grid()
    if only:
        g = [c for c in g if c["exp"] in only.split(",")]
    return g


def reps(quick):
    return QUICK_REPS if quick else REPS


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _quiet():
    return contextlib.redirect_stderr(io.StringIO())


def _geno_spec(pop, n, m, seed=1):
    if pop == "unrelated":
        return {"source": "unrelated", "n": n, "m": m, "seed": seed}
    if pop == "family":
        return {"source": "family", "n_families": n // 4, "n_sibs": 4, "m": m, "seed": seed}
    if pop == "bed":
        return {"source": "bed", "n_snps": m, "n_indivs": n, "select": "random", "seed": seed}
    if pop == "hmdp":
        return {"source": "hmdp", "n_snps": m, "select": "random", "seed": seed}
    raise ValueError(pop)


def impute_filter(G, m_test=None, min_maf=MIN_MAF):
    """MAF >= min_maf columns of G (first m_test), missing calls mean-imputed
    (the CLI default).  Works for 0/0.5/1 and 0/1/2 codings."""
    G = np.array(G, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(G, axis=0)
        top = np.where(np.nanmax(G, axis=0) > 1.0, 2.0, 1.0)
    f = mean / top
    ok = np.minimum(f, 1 - f) >= min_maf
    G = G[:, ok]
    if m_test is not None:
        G = G[:, :m_test]
    bad = np.isnan(G)
    if bad.any():
        G[bad] = np.broadcast_to(np.nanmean(G, axis=0), G.shape)[bad]
    return G


def gls_t(y, K, G, h, X0=None, sigma="reml"):
    """Textbook GLS Wald t (and p, df = n - p) for every column of G at a
    fixed h, V = h K + (1-h) I, via one dense Cholesky.  Independent of pylmm
    (same formulas as ``oracle.wald`` with ``h`` given, vectorised).
    sigma: 'reml' -> Q/(n-p), 'ml' -> Q/n."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    X0 = np.ones((n, 1)) if X0 is None else np.asarray(X0, dtype=np.float64).reshape(n, -1)
    C = linalg.cholesky(h * np.asarray(K) + (1.0 - h) * np.eye(n), lower=True)
    yt = linalg.solve_triangular(C, y, lower=True)
    xt = linalg.solve_triangular(C, X0, lower=True)
    gt = linalg.solve_triangular(C, np.asarray(G, dtype=np.float64), lower=True)
    qx, _ = np.linalg.qr(xt)
    ry = yt - qx @ (qx.T @ yt)
    rg = gt - qx @ (qx.T @ gt)
    ss = (rg**2).sum(axis=0)
    p = X0.shape[1] + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (rg.T @ ry) / ss
        rss = ry @ ry - b**2 * ss
        s2 = rss / ((n - p) if sigma == "reml" else n)
        t = b / np.sqrt(s2 / ss)
    t[ss <= 1e-12] = np.nan
    pv = 2 * stats.t.sf(np.abs(t), n - p)
    return t, pv


def mlogp(p):
    with np.errstate(divide="ignore"):
        return -np.log10(np.asarray(p, dtype=np.float64))


def _counts(p, prefix):
    p = np.asarray(p, dtype=np.float64)
    p = p[~np.isnan(p)]
    out = {prefix + "n": len(p)}
    for a in ALPHAS:
        out[prefix + "k_" + metrics.alpha_label(a)] = int((p <= a).sum())
    return out


def _null_summary(p, prefix):
    out = {
        prefix + "lambda": metrics.lambda_gc(p),
        prefix + "meanchi2": float(np.mean(metrics.chi2_from_p(p))),
    }
    out.update(_counts(p, prefix))
    return out


class SpyLMM:
    """Factory for an instrumented pylmm LMM (subclass; pylmm unchanged).

    Records every Brent evaluation so the study can tell when pylmm's
    unbounded Brent (bug 4) evaluates h > 1."""

    @staticmethod
    def make(y, K, kva, kve, X0=None):
        from pylmm.lmm import LMM

        class _Spy(LMM):
            def _neg_log_likelihood(self, h, X=None, reml=False):
                v = LMM._neg_log_likelihood(self, h, X, reml)
                self.evals.append((float(h), float(v)))
                return v

        L = _Spy.__new__(_Spy)
        L.evals = []
        with _quiet():
            _Spy.__init__(L, y, K, kva, kve, X0=X0)
        return L


def pylmm_fit_instrumented(y, kin, reml=True, X=None):
    """pylmm ``LMM.fit`` with diagnostics.  Returns a dict with h (nan on a
    crash), crashed, crash_type, cap (h == 0.99), skip98 (a grid local max
    at index 98 that range(1, n-2) never tests), gt1 (Brent evaluated h > 1),
    nan_fallback (Brent returned nan -> H[i-1]) and the LMM object."""
    kva, kve = kin.eig_for_pylmm()
    L = SpyLMM.make(np.asarray(y, dtype=np.float64), np.asarray(kin.K), kva, kve)
    out = {"crashed": 0, "crash_type": "", "lmm": L}
    try:
        with _quiet(), np.errstate(all="ignore"):
            h = L.fit(X=X, reml=reml)[0]
        out["h"] = float(np.sum(h))
    except Exception as e:  # scipy >= 1.11 raises BracketError (bug 4)
        out.update(h=np.nan, crashed=1, crash_type=type(e).__name__)
    ll_grid = getattr(L, "ll_grid", None)
    out["skip98"] = int(ll_grid is not None and bool(ll_grid[97] < ll_grid[98] > ll_grid[99]))
    out["cap"] = int(out["h"] == 0.99)
    out["gt1"] = int(any(h > 1 for h, _ in L.evals))
    out["nan_fallback"] = int(
        not out["crashed"]
        and any(np.isnan(v) for _, v in L.evals)
        and out["h"] in set(np.round(np.arange(100) / 100.0, 2))
    )
    return out


_SETUP: dict = {}


def _cached(key, build):
    if key not in _SETUP:
        if len(_SETUP) >= 3:
            _SETUP.pop(next(iter(_SETUP)))
        _SETUP[key] = build()
    return _SETUP[key]


def _std_setup(params):
    """(kin, G_test, geno) for exp remlml / hgrid / mlcent / bednull."""
    key = json.dumps({k: params[k] for k in ("pop", "n", "m_kin", "m_test")}, sort_keys=True)

    def build():
        mk, mt = params["m_kin"], params["m_test"]
        spec = _geno_spec(params["pop"], params["n"], mk + mt + mt // 2 + (1 if mt == 0 else 0))
        geno = genotypes.get(spec)
        kin = kinship.get(spec, snps=slice(0, mk))
        g_test = impute_filter(geno.G[:, mk:], mt) if mt else np.zeros((params["n"], 0))
        return kin, g_test, geno

    return _cached(key, build)


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------


def run_remlml(params, rng):
    from pylmm.lmm import LMM

    kin, G, _ = _std_setup(params)
    K = np.asarray(kin.K)
    n = params["n"]
    cidx = phenotypes.choose_causal(G.shape[1], 2, rng, G=G)
    ph = phenotypes.simulate(
        kin, h2=params["h2"], rng=rng, G=G, causal_idx=cidx, causal_var=0.08
    )  # 4% each
    y = ph.y
    out = {}
    null_reml = pylmm_fit_instrumented(y, kin, reml=True)
    null_ml = pylmm_fit_instrumented(y, kin, reml=False)
    out["reml_pylmm_crashed"] = null_reml["crashed"]
    if null_reml["crashed"]:  # bug 4: GWAS would raise too
        return out
    t_reml, p_reml = methods.pylmm_gwas(y, G, kin, reml=True)
    # the pre-Phase-5.2 CLI default (REML null, Q/(n-q) Wald) == REML=True;
    # since 5.2, REML=False would be a real ML scan (bug 1 fixed)
    t_cli, p_cli = methods.pylmm_gwas(y, G, kin, reml=True)  # CLI default (pre-5.2)
    out["cli_vs_reml_max_abs_dt"] = float(np.nanmax(np.abs(t_cli - t_reml)))
    out.update(
        h_reml_pylmm=null_reml["h"],
        h_ml_pylmm=null_ml["h"],
        ml_pylmm_cap=null_ml["cap"],
        ml_pylmm_crashed=null_ml["crashed"],
    )
    fml = oracle.fit(y, K, reml=False, n_grid=101)
    fre = oracle.fit(y, K, reml=True, n_grid=101)
    out.update(h_ml_oracle=fml.h, h_reml_oracle=fre.h, ml_unbounded=int(fml.ml_unbounded))
    null = np.setdiff1d(np.arange(G.shape[1]), cidx)
    if not fml.ml_unbounded:
        t_mlw, p_mlw = gls_t(y, K, G, fml.h, sigma="ml")  # textbook ML Wald
        t_mlw9, p_mlw9 = gls_t(y, K, G, fml.h, sigma="reml")  # ML h, pylmm sigma
        d = mlogp(p_cli) - mlogp(p_mlw)
        out.update(
            cli_vs_mlwald_mean_abs_dt=float(np.nanmean(np.abs(t_cli - t_mlw))),
            cli_vs_mlwald_max_abs_dt=float(np.nanmax(np.abs(t_cli - t_mlw))),
            cli_vs_mlwald_dlogp_causal=float(np.mean(d[cidx])),
            cli_vs_mlwald_max_abs_dlogp=float(np.nanmax(np.abs(d))),
            bug1_only_max_abs_dt=float(np.nanmax(np.abs(t_cli - t_mlw9))),  # h REML vs ML
            bug9_only_max_abs_dt=float(np.nanmax(np.abs(t_mlw9 - t_mlw))),  # sigma convention
            bug9_dlogp_causal=float(np.mean((mlogp(p_mlw9) - mlogp(p_mlw))[cidx])),
            bug1_dlogp_causal=float(np.mean((mlogp(p_cli) - mlogp(p_mlw9))[cidx])),
        )
        out.update(_counts(p_mlw[null], "mlw_"))
        # ML LRT (per-SNP ML refit of h) on the causal SNPs + 4 null SNPs
        sel = np.r_[cidx, rng.choice(null, size=4, replace=False)]
        ll0 = fml.ll
        lrt_p = []
        for j in sel:
            fa = oracle.fit(y, K, np.column_stack([np.ones(n), G[:, j]]), reml=False, n_grid=101)
            lrt_p.append(stats.chi2.sf(max(0.0, 2 * (fa.ll - ll0)), 1))
        lrt_p = np.array(lrt_p)
        dl = mlogp(p_cli[sel]) - mlogp(lrt_p)
        out.update(
            cli_vs_lrt_dlogp_causal=float(np.mean(dl[:2])),
            cli_vs_lrt_max_abs_dlogp=float(np.max(np.abs(dl))),
        )
    out.update(_counts(p_cli[null], "cli_"))
    out["dlogp_causal_cli"] = float(np.mean(mlogp(p_cli[cidx])))
    # --refit: CLI fits h by ML per SNP (REML=options.REML), lib GWAS(refit) by REML
    sel = np.r_[cidx, rng.choice(null, size=4, replace=False)]
    h_r_cli, h_r_lib, t_r_cli, t_r_lib, crash = [], [], [], [], 0
    kva, kve = kin.eig_for_pylmm()
    with _quiet():
        L = LMM(y, K, kva, kve)
    for j in sel:
        x = G[:, j].reshape(-1, 1)
        try:
            with np.errstate(all="ignore"):
                # pre-Phase-5.2 CLI --refit: ML refit, Q/(n-q) Wald, which is
                # association(REML=True) since 5.2 (REML=False now uses Q/n)
                L.fit(X=x, reml=False)
                h_r_cli.append(L.opt_h)
                t_r_cli.append(L.association(x, reml=True)[0])
                L.fit(X=x)  # lmm.GWAS(refit=True)
                h_r_lib.append(L.opt_h)
                t_r_lib.append(L.association(x, reml=True)[0])
        except Exception:
            crash += 1
    out["refit_crashes"] = crash
    if t_r_cli:
        out.update(
            refit_h_cli_mean=float(np.mean(h_r_cli)),
            refit_h_lib_mean=float(np.mean(h_r_lib)),
            refit_cli_cap_frac=float(np.mean(np.array(h_r_cli) == 0.99)),
            refit_cli_vs_lib_max_abs_dt=float(
                np.max(np.abs(np.array(t_r_cli) - np.array(t_r_lib)))
            ),
        )
    return out


def run_hgrid(params, rng):
    kin, G, _ = _std_setup(params)
    K = np.asarray(kin.K)
    cidx = phenotypes.choose_causal(G.shape[1], 2, rng, G=G)
    ph = phenotypes.simulate(kin, h2=params["h2"], rng=rng, G=G, causal_idx=cidx, causal_var=0.06)
    y = ph.y
    fp = pylmm_fit_instrumented(y, kin, reml=True)
    fo = oracle.fit(y, K, reml=True, n_grid=201)
    fc = oracle.fit(y, K, reml=True, n_grid=201, eig_clamp=1e-6)
    out = {k: fp[k] for k in ("h", "crashed", "cap", "skip98", "gt1", "nan_fallback")}
    out.update(h_oracle=fo.h, h_oracle_clamp=fc.h, oracle_above_099=int(fo.h > 0.99))
    if fp["crashed"]:
        return out
    out["dh"] = fp["h"] - fo.h
    out["abs_dh"] = abs(out["dh"])
    out["mismatch"] = int(abs(out["dh"]) > 1e-3)
    out["clamp_dh"] = fc.h - fo.h
    t_p, p_p = methods.pylmm_gwas(y, G, kin, reml=True)
    t_o, p_o = gls_t(y, K, G, fo.h, sigma="reml")
    d = mlogp(p_p) - mlogp(p_o)
    out.update(
        max_abs_dt=float(np.nanmax(np.abs(t_p - t_o))),
        max_abs_dlogp=float(np.nanmax(np.abs(d))),
        dlogp_causal=float(np.mean(d[cidx])),
        mlogp_causal_oracle=float(np.mean(mlogp(p_o[cidx]))),
    )
    # refit-mode crash exposure: per-SNP REML refits on 20 SNPs
    from pylmm.lmm import LMM

    kva, kve = kin.eig_for_pylmm()
    with _quiet():
        L = LMM(y, K, kva, kve)
    crashes = 0
    for j in rng.choice(G.shape[1], size=20, replace=False):
        try:
            with np.errstate(all="ignore"):
                L.fit(X=G[:, j].reshape(-1, 1))
        except Exception:  # noqa: PERF203 - count per-SNP crashes
            crashes += 1
    out["refit_crash_frac"] = crashes / 20.0
    return out


def _kinship_setup(params):
    key = json.dumps(
        {k: params[k] for k in ("pop", "n", "m_kin", "m_test", "miss", "mono")}, sort_keys=True
    )

    def build():
        from pylmm import plink_io
        from pylmm.lmm import calculate_kinship

        mk, mt = params["m_kin"], params["m_test"]
        spec = _geno_spec(params["pop"], params["n"], mk + mt + mt // 2)
        geno = genotypes.get(spec)
        g_kin = np.array(geno.G[:, :mk], dtype=np.float64)
        g_test = impute_filter(geno.G[:, mk:], mt)
        rng = np.random.default_rng(12345)
        k_true = None
        if params["pop"] == "unrelated":
            n_mono = round(mk * params["mono"] / (1 - params["mono"]))
            g_kin = np.hstack([g_kin, np.zeros((g_kin.shape[0], n_mono))])
            k_true = calculate_kinship(g_kin[:, :mk].copy()) * 1.0  # complete, all polymorphic
            g_kin = genotypes.with_missing(g_kin, params["miss"], rng)
        m = g_kin.shape[1]
        poly = np.array([np.nanvar(g_kin[:, j]) > 0 for j in range(m)])
        k_lib = calculate_kinship(g_kin.copy())
        k_fix = k_lib * m / poly.sum()
        W = np.zeros((g_kin.shape[0], 0))
        cols = []
        for j in range(m):  # pylmmKinship CLI path
            g = plink_io.PlinkReader.normalize_genotype(None, g_kin[:, j].copy())
            if g.var() == 0:
                continue
            cols.append(g)
        W = np.column_stack(cols)
        k_cli = W @ W.T / float(m)
        ks = {"lib": k_lib, "cli": k_cli, "fix": k_fix}
        if k_true is not None:
            ks["true"] = k_true
        kins = {k: kinship.from_matrix(v) for k, v in ks.items()}
        info = {
            "mono_frac": float(1 - poly.mean()),
            "miss_frac": float(np.isnan(g_kin).mean()),
            "scale_c": float(poly.sum() / m),
            "max_abs_dK_lib_cli": float(np.max(np.abs(k_lib - k_cli))),
            "meandiag_lib": float(np.mean(np.diag(k_lib))),
            "meandiag_cli": float(np.mean(np.diag(k_cli))),
            "meandiag_fix": float(np.mean(np.diag(k_fix))),
        }
        return kins, g_test, info

    return _cached(key, build)


def run_kinship(params, rng):
    kins, G, info = _kinship_setup(params)
    ref = "true" if "true" in kins else "fix"
    cidx = phenotypes.choose_causal(G.shape[1], 2, rng, G=G)
    ph = phenotypes.simulate(
        kins[ref], h2=params["h2"], rng=rng, G=G, causal_idx=cidx, causal_var=0.08
    )
    y = ph.y
    out = dict(info)
    res = {}
    for k, kin in kins.items():
        f = pylmm_fit_instrumented(y, kin, reml=True)
        if f["crashed"]:
            out["crashed_" + k] = 1
            continue
        t, p = methods.pylmm_gwas(y, G, kin, reml=True)
        res[k] = (f["h"], t, p)
        out["h_" + k] = f["h"]
        out["cap_" + k] = f["cap"]
    if ref in res:
        h0, t0, p0 = res[ref]
        for k in res:
            if k == ref:
                continue
            h, t, p = res[k]
            d = mlogp(p) - mlogp(p0)
            out["dh_%s" % k] = h - h0
            out["max_abs_dt_%s" % k] = float(np.nanmax(np.abs(t - t0)))
            out["dlogp_causal_%s" % k] = float(np.mean(d[cidx]))
            out["max_abs_dlogp_%s" % k] = float(np.nanmax(np.abs(d)))
    return out


def run_mlcent(params, rng):
    kin, _, _ = _std_setup(params)
    K = np.asarray(kin.K)
    ph = phenotypes.simulate(kin, h2=params["h2"], rng=rng)
    y = ph.y
    fml = oracle.fit(y, K, reml=False, n_grid=101)
    fre = oracle.fit(y, K, reml=True, n_grid=101)
    pml = pylmm_fit_instrumented(y, kin, reml=False)
    pre = pylmm_fit_instrumented(y, kin, reml=True)
    # the -1/2 log(1-h) term of the null direction: LL gain from 0.9 to 0.99
    ll = lambda h: oracle.profile_loglik(h, y, K, reml=False)[0]  # noqa: E731
    out = {
        "ml_unbounded": int(fml.ml_unbounded),
        "h_ml_oracle": fml.h,
        "h_reml_oracle": fre.h,
        "h_ml_pylmm": pml["h"],
        "h_reml_pylmm": pre["h"],
        "ml_pylmm_cap": pml["cap"],
        "ml_pylmm_crashed": pml["crashed"],
        "reml_pylmm_cap": pre["cap"],
        "rank_K": int((np.asarray(kin.kva) > 1e-8).sum()),
        "ml_ll_gain_0999": float(ll(0.999) - ll(0.99)),
    }
    if not pml["crashed"]:
        out["dh_ml_minus_reml_pylmm"] = pml["h"] - pre["h"]
    return out


def _bednull_setup(params):
    key = json.dumps({k: params[k] for k in ("pop", "n", "m_kin", "m_test")}, sort_keys=True)

    def build():
        kin, G, geno = _std_setup(params)
        g_all = np.where(np.isnan(geno.G), -1.0, geno.G)
        Z = (G - G.mean(0)) / G.std(0)
        ev = np.clip(np.linalg.eigvalsh(Z @ Z.T / G.shape[1]), 0, None)
        info = {
            "rank_K": int((np.asarray(kin.kva) > 1e-6).sum()),
            "unique_indivs": len(np.unique(g_all, axis=0)),
            "unique_test_patterns": len(np.unique(np.round(G.T, 6), axis=0)),
            # effective number of independent tests (participation ratio of
            # the SNP correlation spectrum) -- n x n form of the same spectrum
            "meff": float(ev.sum() ** 2 / (ev**2).sum()),
            "m_test_used": int(G.shape[1]),
        }
        return kin, G, info

    return _cached("bn" + key, build)


def run_bednull(params, rng):
    kin, G, info = _bednull_setup(params)
    K = np.asarray(kin.K)
    nc = params.get("n_causal", 0)
    cidx = np.zeros(0, dtype=int)
    if nc:
        cidx = phenotypes.choose_causal(G.shape[1], nc, rng, G=G)
    ph = phenotypes.simulate(
        kin, h2=params["h2"], rng=rng, G=G if nc else None, causal_idx=cidx, causal_var=0.02 * nc
    )
    y = ph.y
    out = dict(info)
    null = np.setdiff1d(np.arange(G.shape[1]), cidx)
    nf = pylmm_fit_instrumented(y, kin, reml=True)
    out["pylmm_crashed"] = nf["crashed"]  # bug 4 (BracketError) at high h2
    fo = oracle.fit(y, K, reml=True, n_grid=101)
    if nf["crashed"]:
        p_l = np.full(G.shape[1], np.nan)
    else:
        _, p_l = methods.pylmm_gwas(y, G, kin, reml=True)
    _, p_o = gls_t(y, K, G, fo.h)
    h_true = ph.sigma_g2 / (ph.sigma_g2 + ph.sigma_e2)
    _, p_t = gls_t(y, K, G, h_true) if h_true < 1 else (None, p_o)
    _, p_ols = methods.ols_gwas(y, G)
    out.update(
        h_pylmm=nf["h"], h_oracle=fo.h, dh_oracle=nf["h"] - fo.h, pylmm_at_h0=int(nf["h"] == 0.0)
    )
    sets = (("orc_", p_o), ("true_", p_t), ("ols_", p_ols))
    if not nf["crashed"]:
        sets = (("lmm_", p_l), *sets)
        out["max_abs_dlogp_lmm_orc"] = float(np.nanmax(np.abs(mlogp(p_l) - mlogp(p_o))))
    for tag, p in sets:
        out.update(_null_summary(p[null], tag))
    if nc:
        X0 = np.column_stack([np.ones(len(y)), G[:, cidx]])
        _, p_c = gls_t(y, K, G[:, null], h_true, X0=X0)
        out.update(_null_summary(p_c, "cond_"))
    return out


EXPERIMENTS = {
    "remlml": run_remlml,
    "hgrid": run_hgrid,
    "kinship": run_kinship,
    "mlcent": run_mlcent,
    "bednull": run_bednull,
}


def run_cell(params, seed):
    rng = np.random.default_rng(seed)
    return EXPERIMENTS[params["exp"]](params, rng)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _pooled_t1e(d, prefix):
    rows = {}
    n = d[prefix + "n"].sum()
    for a in ALPHAS:
        k = d[prefix + "k_" + metrics.alpha_label(a)].sum()
        lo, hi = metrics.rate_ci(k, n)
        rows["t1e_%s_%s" % (prefix.rstrip("_"), metrics.alpha_label(a))] = k / n if n else np.nan
        rows["t1e_%s_%s_ci" % (prefix.rstrip("_"), metrics.alpha_label(a))] = (
            "[%.3g, %.3g]" % (lo, hi) if n else ""
        )
    return rows


def _q(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(np.quantile(x, q)) if len(x) else np.nan


def report(df, outdir):
    import pandas as pd

    from sims import plots
    from sims import report as rpt

    tables = []
    allrows = []

    def mean(d, c):
        return float(d[c].mean()) if c in d and d[c].notna().any() else np.nan

    def mx(d, c):
        return float(d[c].max()) if c in d and d[c].notna().any() else np.nan

    def frac(d, c):
        return float(d[c].mean()) if c in d and d[c].notna().any() else np.nan

    # remlml
    d0 = df[df.exp == "remlml"]
    if len(d0):
        rows = []
        for (pop, mk, h2), d in d0.groupby(["pop", "m_kin", "h2"]):
            r = {
                "pop": pop,
                "m_kin": mk,
                "h2": h2,
                "reps": len(d),
                "reml_crash_rate": frac(d, "reml_pylmm_crashed"),
                "cli_vs_reml_max_dt": mx(d, "cli_vs_reml_max_abs_dt"),
                "h_reml": mean(d, "h_reml_pylmm"),
                "h_ml_oracle": mean(d, "h_ml_oracle"),
                "h_ml_pylmm": mean(d, "h_ml_pylmm"),
                "ml_unbounded": frac(d, "ml_unbounded"),
                "ml_pylmm_cap": frac(d, "ml_pylmm_cap"),
                "cli_vs_mlwald_mean_dt": mean(d, "cli_vs_mlwald_mean_abs_dt"),
                "cli_vs_mlwald_max_dt": mx(d, "cli_vs_mlwald_max_abs_dt"),
                "dlogp_causal_cli_vs_mlwald": mean(d, "cli_vs_mlwald_dlogp_causal"),
                "bug1_dlogp_causal": mean(d, "bug1_dlogp_causal"),
                "bug9_dlogp_causal": mean(d, "bug9_dlogp_causal"),
                "dlogp_causal_cli_vs_lrt": mean(d, "cli_vs_lrt_dlogp_causal"),
                "max_abs_dlogp_cli_vs_lrt": mx(d, "cli_vs_lrt_max_abs_dlogp"),
                "mlogp_causal_cli": mean(d, "dlogp_causal_cli"),
                "refit_h_cli(ML)": mean(d, "refit_h_cli_mean"),
                "refit_h_lib(REML)": mean(d, "refit_h_lib_mean"),
                "refit_cli_cap_frac": mean(d, "refit_cli_cap_frac"),
                "refit_cli_vs_lib_max_dt": mx(d, "refit_cli_vs_lib_max_abs_dt"),
                "refit_crashes": float(d["refit_crashes"].sum())
                if "refit_crashes" in d
                else np.nan,
            }
            r.update({k: v for k, v in _pooled_t1e(d, "cli_").items() if "0.05" in k})
            if "mlw_n" in d and d["mlw_n"].notna().any():
                r.update(
                    {
                        k: v
                        for k, v in _pooled_t1e(d.dropna(subset=["mlw_n"]), "mlw_").items()
                        if "0.05" in k
                    }
                )
            rows.append(r)
        t = pd.DataFrame(rows)
        tables.append(("Bugs 1 / 9 / 17: CLI default vs REML vs textbook ML (n = 300)", t))

    # hgrid
    d0 = df[df.exp == "hgrid"]
    if len(d0):
        rows = []
        for (pop, n, h2), d in d0.groupby(["pop", "n", "h2"]):
            ok = d[d.crashed == 0]
            rows.append(
                {
                    "pop": pop,
                    "n": n,
                    "h2": h2,
                    "reps": len(d),
                    "crash_rate": frac(d, "crashed"),
                    "cap_rate": frac(ok, "cap"),
                    "skip98_rate": frac(ok, "skip98"),
                    "oracle_h>0.99": frac(d, "oracle_above_099"),
                    "mismatch_rate(|dh|>1e-3)": frac(ok, "mismatch"),
                    "mean_abs_dh": mean(ok, "abs_dh"),
                    "max_abs_dh": mx(ok, "abs_dh"),
                    "max_abs_clamp_dh": float(ok["clamp_dh"].abs().max()) if len(ok) else np.nan,
                    "max_abs_dt": mx(ok, "max_abs_dt"),
                    "max_abs_dlogp": mx(ok, "max_abs_dlogp"),
                    "mean_dlogp_causal": mean(ok, "dlogp_causal"),
                    "q05_dlogp_causal": _q(ok.get("dlogp_causal", []), 0.05),
                    "mlogp_causal_oracle": mean(ok, "mlogp_causal_oracle"),
                    "refit_crash_frac": mean(d, "refit_crash_frac"),
                }
            )
        t = pd.DataFrame(rows)
        tables.append(("Bugs 2 / 3 / 4: pylmm REML search vs dense oracle at high h2", t))
        _plot_hgrid(d0, outdir, plots)

    # kinship
    d0 = df[df.exp == "kinship"]
    if len(d0):
        rows = []
        for (pop, miss, mono, h2), d in d0.groupby(["pop", "miss", "mono", "h2"]):
            r = {
                "pop": pop,
                "miss_added": miss,
                "mono_added": mono,
                "h2": h2,
                "reps": len(d),
                "mono_frac": mean(d, "mono_frac"),
                "miss_frac": mean(d, "miss_frac"),
                "scale_c": mean(d, "scale_c"),
                "max|K_lib-K_cli|": mean(d, "max_abs_dK_lib_cli"),
                "diag_lib": mean(d, "meandiag_lib"),
                "diag_cli": mean(d, "meandiag_cli"),
                "diag_fix": mean(d, "meandiag_fix"),
            }
            for k in ("lib", "cli", "fix", "true"):
                if "h_" + k in d and d["h_" + k].notna().any():
                    r["h_" + k] = mean(d, "h_" + k)
            for k in ("lib", "cli", "fix"):
                for c in (
                    "dh_%s" % k,
                    "max_abs_dt_%s" % k,
                    "max_abs_dlogp_%s" % k,
                    "dlogp_causal_%s" % k,
                ):
                    if c in d and d[c].notna().any():
                        r[c] = mx(d, c) if c.startswith("max") else mean(d, c)
            rows.append(r)
        t = pd.DataFrame(rows)
        tables.append(
            (
                "Bug 8: kinship normalisation (reference K: 'true' for synthetic, "
                "'fix' = lib with corrected denominator for real data)",
                t,
            )
        )

    # mlcent
    d0 = df[df.exp == "mlcent"]
    if len(d0):
        rows = []
        for (pop, mk, h2), d in d0.groupby(["pop", "m_kin", "h2"]):
            rows.append(
                {
                    "pop": pop,
                    "n": int(d.n.iloc[0]),
                    "m_kin": mk,
                    "h2": h2,
                    "reps": len(d),
                    "rank_K": mean(d, "rank_K"),
                    "ml_unbounded": frac(d, "ml_unbounded"),
                    "h_reml_oracle": mean(d, "h_reml_oracle"),
                    "h_ml_pylmm": mean(d, "h_ml_pylmm"),
                    "h_reml_pylmm": mean(d, "h_reml_pylmm"),
                    "ml_pylmm_cap": frac(d, "ml_pylmm_cap"),
                    "ml_minus_reml_h": mean(d, "dh_ml_minus_reml_pylmm"),
                    "ml_ll_gain_0.99->0.999": mean(d, "ml_ll_gain_0999"),
                }
            )
        t = pd.DataFrame(rows)
        tables.append(("Bug 17: ML with the centred GRM (n = 200)", t))

    # bednull
    d0 = df[df.exp == "bednull"]
    if len(d0):
        rows = []
        for (pop, n, h2, nc), d in d0.groupby(["pop", "n", "h2", "n_causal"]):
            r = {
                "pop": pop,
                "n": n,
                "h2": h2,
                "n_causal": nc,
                "reps": len(d),
                "rank_K": mean(d, "rank_K"),
                "unique_indivs": mean(d, "unique_indivs"),
                "unique_test_patterns": mean(d, "unique_test_patterns"),
                "meff": mean(d, "meff"),
                "h_pylmm": mean(d, "h_pylmm"),
                "pylmm_at_h0": frac(d, "pylmm_at_h0"),
                "pylmm_crash_rate": float(d["pylmm_crashed"].fillna(0).mean())
                if "pylmm_crashed" in d
                else 0.0,
                "max_abs_dh_oracle": float(d["dh_oracle"].abs().max()),
                "max_abs_dlogp_lmm_orc": mx(d, "max_abs_dlogp_lmm_orc"),
            }
            tags = ("lmm_", "orc_", "true_", "ols_") + (("cond_",) if nc else ())
            for tag in tags:
                lam = d[tag + "lambda"]
                r[tag + "lambda_mean"] = float(lam.mean())
                r[tag + "lambda_q05"] = _q(lam, 0.05)
                r[tag + "lambda_q95"] = _q(lam, 0.95)
                r[tag + "lambda_min"] = float(lam.min())
                r[tag + "lambda_max"] = float(lam.max())
                r[tag + "P(lambda>1.3)"] = float((lam > 1.3).mean())
                r[tag + "meanchi2"] = float(d[tag + "meanchi2"].mean())
                r[tag + "meanchi2_se"] = float(d[tag + "meanchi2"].sem())
                r.update(_pooled_t1e(d, tag))
            rows.append(r)
        t = pd.DataFrame(rows)
        tables.append(("Item 18: null calibration on the BED genotypes", t))
        _plot_bednull(d0, outdir, plots)

    for title, t in tables:
        t = t.copy()
        t.insert(0, "table", title.split(":")[0])
        allrows.append(t)
    summary = pd.concat(allrows, ignore_index=True, sort=False) if allrows else pd.DataFrame()
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    for title, t in tables:
        slug = title.split(":")[0].lower().replace(" ", "").replace("/", "_").replace(".", "")
        t.to_csv(os.path.join(outdir, "table_%s.csv" % slug), index=False)
    _write_md(tables, outdir, rpt)
    _write_ledger_csv(outdir)


def _write_md(tables, outdir, rpt):
    runs = os.path.join(
        outdir if os.path.basename(outdir) != "quick" else os.path.dirname(outdir),
        "raw",
        "runs.jsonl",
    )
    info = ""
    if os.path.exists(runs):
        mode = "quick" if os.path.basename(outdir) == "quick" else "full"
        last = [json.loads(x) for x in open(runs) if x.strip()]
        last = [r for r in last if r.get("mode") == mode]
        if last:
            info = (
                "_Last run: %s, git %s, jobs=%d, wall %.0fs; total task time over all "
                "runs %.0fs._\n\n"
                % (
                    last[-1]["time"],
                    last[-1]["git"],
                    last[-1]["jobs"],
                    last[-1]["wall_s"],
                    sum(x["task_cpu_s"] for x in last),
                )
            )
    parts = [
        "# S4: bug confirmation (measured impact)\n\n",
        info,
        "The verdicts and the ledger are in sims/studies/s4_bug_confirmation.md. "
        "Every number below has its column in summary.csv / table_*.csv.\n",
    ]
    for title, t in tables:
        if "Item 18" in title:
            keep = [
                c
                for c in t.columns
                if not c.endswith("_ci")
                and "q05" not in c
                and not c.endswith("_se")
                and "1e-0" not in c
                and "0.01" not in c
                and not c.endswith("_min")
                and not c.endswith("_max")
            ]
            t = t[keep]
        parts.append("\n## %s\n\n" % title)
        parts.append(rpt.markdown_table(t, floatfmt="%.3g"))
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("".join(parts))


def _plot_hgrid(d0, outdir, plots):
    ok = d0[d0.crashed == 0]
    if not len(ok):
        return
    plots.scatter(
        ok["h_oracle"].values,
        ok["h"].values,
        os.path.join(outdir, "hgrid_h_pylmm_vs_oracle.png"),
        xlabel="oracle REML h (fine grid over [0, 1])",
        ylabel="pylmm REML h",
        title="bugs 2/3/4: pylmm vs oracle h at high h2 (crashes omitted)",
        identity=True,
    )


def _plot_bednull(d0, outdir, plots):
    plt = plots._plt()
    cells = list(d0.groupby(["pop", "n", "h2", "n_causal"]))
    fig, axes = plt.subplots(1, len(cells), figsize=(2.6 * len(cells), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ((pop, n, h2, nc), d) in zip(axes, cells):
        data = [d["lmm_lambda"].dropna().values, d["true_lambda"].values]
        labels = ["pylmm", "exact\n(true h)"]
        if nc:
            data.append(d["cond_lambda"].dropna().values)
            labels.append("exact,\ncond.")
        ax.boxplot(data, widths=0.6)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
        ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
        ax.set_title("%s n=%d h2=%g%s" % (pop, n, h2, " +%d causal" % nc if nc else ""), fontsize=7)
    axes[0].set_ylabel("replicate lambda_GC")
    fig.suptitle("item 18: replicate lambda_GC, pylmm vs the exact test", fontsize=9)
    plots._save(fig, os.path.join(outdir, "bednull_lambda.png"))


# ---------------------------------------------------------------------------
# the ledger (machine-readable copy of the table in s4_bug_confirmation.md)
# ---------------------------------------------------------------------------

LEDGER = [
    # bug, short name, verdict, severity, affected path, recommended phase
    (
        1,
        "null fit always REML; --REML no-op on default path",
        "CONFIRMED",
        "medium",
        "CLI default + lmm.GWAS(REML=False); CLI --refit/--removeMissingGenotypes use ML",
        "5",
    ),
    (2, "getMax range(1, n-2) skips grid index 98", "CONFIRMED", "low", "every null fit", "6.3"),
    (
        3,
        "h grid capped at 0.99 (and fitTwo w grid)",
        "CONFIRMED",
        "medium",
        "every null fit at high h2; misc.fitTwo",
        "6.3",
    ),
    (
        4,
        "unbounded Brent above h = 1",
        "CONFIRMED",
        "high",
        "every null fit and refit at high h2: crash on scipy >= 1.11; on scipy < 1.11 correct, "
        "except h > 1 (LL nan) when the ML profile is unbounded",
        "5",
    ),
    (5, "tstat(log=True) adds 2.0 not log 2", "UNREACHABLE", "cosmetic", "no caller", "6.3"),
    (6, "--kfile2 NameError", "CONFIRMED", "low", "CLI --kfile2 only", "5"),
    (7, "--eigen cannot read pylmmKinship output", "CONFIRMED", "low", "CLI --eigen only", "5"),
    (
        8,
        "kinship normalisation / denominator",
        "CONFIRMED",
        "low",
        "lmm.calculateKinship, pylmmKinship",
        "6.3",
    ),
    (9, "ML sigma uses Q/(n-q)", "CONVENTION", "cosmetic", "association(REML=False)", "6.3"),
    (
        10,
        "getPhenos drops individuals missing from the pheno file",
        "CONFIRMED",
        "medium",
        "CLI whenever the pheno/cov file lacks a fam individual (crash)",
        "5",
    ),
    (
        11,
        "all-missing SNP returns an empty array",
        "CONFIRMED",
        "medium",
        "CLI GWAS + pylmmKinship (crash)",
        "5",
    ),
    (
        12,
        "LMM mutates caller's Kva; calculateKinship mutates W",
        "CONFIRMED",
        "low",
        "library API",
        "5",
    ),
    (13, "inconsistent zero-variance filters", "CONFIRMED", "cosmetic", "CLI", "5"),
    (14, "crash without a pheno file", "CONFIRMED", "low", "CLI usage error", "5"),
    (15, "inconsistent return types; 1x1 arrays", "CONFIRMED", "cosmetic", "library API", "5"),
    (
        16,
        "headered / wrong-size --kfile read silently",
        "CONFIRMED",
        "high",
        "CLI --kfile (numpy < 2.3; size mismatch on all)",
        "5",
    ),
    (
        17,
        "ML unbounded as h -> 1 with centred GRM",
        "NOT A BUG",
        "medium",
        "ML paths: fit(REML=False), CLI --refit / --removeMissingGenotypes",
        "6.3",
    ),
    (
        18,
        "lambda_GC 1.3-4.9 on BED genotypes",
        "NOT A BUG",
        "none",
        "study design (LD, few effective tests)",
        "-",
    ),
]


def _write_ledger_csv(outdir):
    import csv

    with open(os.path.join(outdir, "ledger.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bug", "name", "verdict", "severity", "affected", "phase"])
        for row in LEDGER:
            w.writerow(row)
