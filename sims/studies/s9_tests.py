"""S9: the association tests (Phase 6.5) -- Wald, Wald with the exact refit, LRT and score.

Questions: are the tests calibrated under the null (lambda_GC, type-I
error down to 1e-5), how much power does each have, does the exact per-SNP
refit (`LMM.fit_snps`) reach the maximum the per-SNP grid search
(``LMM.fit(X=x)``) and the dense oracle find, and what does each test cost?

Per replicate (one simulated phenotype on the cell's fixed cohort):

1. genotypes and K (cached per worker): K from ``M_KIN`` SNPs (pylmm's
   centred, standardised kinship), ``m_test`` disjoint test SNPs (complete,
   polymorphic);
2. ``y = u + e`` with ``u ~ N(0, h2 K)``; in the ``power`` scenario
   ``N_CAUSAL`` of the test SNPs are causal, each explaining ``vpe`` of the
   phenotypic variance;
3. on every test SNP, each timed: ``wald`` (REML null fit, the default scan),
   ``wald_refit`` (REML, `LMM.fit_snps`), ``lrt`` (`gwas(test="lrt")`: reduced
   ML null and per-SNP fits, chi2(1)), ``lrt_plain`` (the same with the plain
   ML likelihood, whose fits sit at the h = 1 spike of a centred K: the LRT
   before the PR #31 review, and GEMMA's) and ``score`` (`LMM.score_batch`,
   REML null fit, F(1, n - q - 1));
4. metrics: lambda_GC and the counts of null p-values below 0.05, 1e-3,
   1e-4 and 1e-5 (pooled over replicates in the report); in ``power``, the
   counts of causal p-values below 1e-3 and 1e-5 and their mean -log10 p;
5. (not in the ``tail`` scenario) the exact refit vs the per-SNP grid search
   on the first ``N_CHECK`` test SNPs, REML and ML: max |dh| and the largest
   log-likelihood shortfall of each against the other; and vs the dense
   oracle (`sims.oracle.fit` with pylmm's eigenvalue floor) on ``N_ORACLE``
   of them.

Cohorts: ``unrelated300`` (n = 300) and ``unrelated100`` (n = 100, where the
asymptotic chi2 of the LRT and score test is least accurate), ``family300``
(100 families of 3 sibs: strong relatedness, where refitting matters most).
h2 in {0, 0.3, 0.8}.  The ``tail`` cell (unrelated100, h2 = 0.3, 50000 null
SNPs) pools 2 million null tests per test for the 1e-5 tail.

Run:   just sim s9_tests --quick           (seconds)
       just sim s9_tests --jobs 6 --report (about 15 min)
"""

from __future__ import annotations

import contextlib
import io
import time
import warnings

import numpy as np

from sims import genotypes, kinship, metrics, oracle, phenotypes

NAME = "s9_tests"
STUDY_VERSION = 2

TESTS = ("wald", "wald_refit", "lrt", "lrt_plain", "score")
# lrt: the reduced-ML LRT of gwas(test="lrt"); lrt_plain: the plain ML LRT (both fits
# on the full model, as GEMMA and the first version of this PR), study only; score:
# LMM.score_batch, library only (not a gwas test).
H2S = [0.0, 0.3, 0.8]
SOURCES = {
    "unrelated300": {"source": "unrelated", "n": 300},
    "unrelated100": {"source": "unrelated", "n": 100},
    "family300": {"source": "family", "n_families": 100, "n_sibs": 3},
}
M_KIN = 1000
N_CAUSAL = 5
# per-causal-SNP variance explained: power around 0.5 at 1e-3 for n = 300
VPE = {"unrelated300": 0.03, "unrelated100": 0.09, "family300": 0.03}
ALPHAS = (0.05, 1e-3, 1e-4, 1e-5)
POWER_ALPHAS = (1e-3, 1e-5)
N_CHECK = 20
N_ORACLE = 2


def grid(quick: bool) -> list[dict]:
    if quick:
        return [
            {"cohort": "unrelated100", "h2": 0.3, "scenario": s, "m_test": 200, "geno_seed": 1}
            for s in ("null", "power")
        ]
    cells = [
        {"cohort": c, "h2": h2, "scenario": s, "m_test": 5000 if s == "null" else 1000}
        for c in SOURCES
        for h2 in H2S
        for s in ("null", "power")
    ]
    cells.append({"cohort": "unrelated100", "h2": 0.3, "scenario": "tail", "m_test": 50000})
    return [{**c, "geno_seed": 1} for c in cells]


def reps(quick: bool) -> int:
    return 1 if quick else 40


def _cohort(params):
    """(kinship, complete polymorphic test SNPs) of the cell's cohort."""
    base = dict(SOURCES[params["cohort"]])
    m = M_KIN + int(params["m_test"] * 1.05) + 10  # spare columns: monomorphic ones are dropped
    spec = {**base, "m": m, "seed": params["geno_seed"]}
    g = genotypes.get(spec)
    kin = kinship.get(spec, snps=slice(0, M_KIN))
    T = g.G[:, M_KIN:]
    T = T[:, T.std(axis=0) > 0][:, : params["m_test"]]
    return kin, T


def _quiet(fn):
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        return fn()


def _scan(L, T, test, out):
    """One test of all SNPs (in blocks of 1024, as gwas does); (p, seconds).

    Records the null fit's h of the two LRTs in ``out``.
    """
    from pylmm.lmm import _lrt_result

    t0 = time.perf_counter()
    ll0 = 0.0
    if test == "lrt":
        out["h0_lrt"] = L.lrt_null_fit()[0]
    elif test == "lrt_plain":
        h0, _, _, ll0 = L.fit(reml=False)
        out["h0_lrt_plain"] = float(h0)
    else:
        L.fit(reml=True)
    ps = []
    for b in range(0, T.shape[1], 1024):
        X = T[:, b : b + 1024]
        if test == "wald_refit":
            r = L.association_tests(X, "wald", refit=True)
        elif test == "score":
            r = L.score_batch(X)
        elif test == "lrt_plain":
            f = L.fit_snps(X, reml=False)
            r = _lrt_result(f.ll, ll0, f.beta, f.beta_var)
        else:
            r = L.association_tests(X, test)
        ps.append(r.p)
    return np.concatenate(ps), time.perf_counter() - t0


def _refit_check(L, T, y, K, reml, out, tag):
    """The exact refit vs fit(X=x) and the dense oracle on the first SNPs."""
    X = T[:, :N_CHECK]
    new = L.fit_snps(X, reml=reml)
    h_old, ll_old = [], []
    for j in range(X.shape[1]):
        h, _, _, ll = L.fit(X=X[:, j], reml=reml)
        h_old.append(float(h))
        ll_old.append(ll)
    h_old_a, ll_old_a = np.asarray(h_old), np.asarray(ll_old)
    out["dh_" + tag] = float(np.max(np.abs(new.h - h_old_a)))
    out["short_new_" + tag] = float(np.max(ll_old_a - new.ll))
    out["short_old_" + tag] = float(np.max(new.ll - ll_old_a))
    out["fallbacks_" + tag] = float(L._snp_fallbacks)
    out["at1_" + tag] = float(np.mean(new.h >= 1.0))
    short, dh = [], []
    n = len(y)
    for j in range(N_ORACLE):
        f = oracle.fit(y, K, X=np.column_stack([np.ones(n), X[:, j]]), reml=reml, eig_clamp=1e-6)
        short.append(f.ll_pylmm - new.ll[j])
        dh.append(abs(f.h - new.h[j]))
    out["oracle_short_" + tag] = float(np.max(short))
    out["oracle_dh_" + tag] = float(np.max(dh))


def run_cell(params, seed):
    from pylmm.lmm import LMM

    rng = np.random.default_rng(seed)
    kin, T = _cohort(params)
    K = np.asarray(kin.K)
    causal = np.zeros(0, dtype=np.int64)
    if params["scenario"] == "power":
        causal = rng.choice(T.shape[1], N_CAUSAL, replace=False)
        ph = phenotypes.simulate(
            kin,
            params["h2"],
            rng,
            G=T,
            causal_idx=causal,
            causal_var=VPE[params["cohort"]] * N_CAUSAL,
        )
    else:
        ph = phenotypes.simulate(kin, params["h2"], rng)
    y = ph.y
    kva, kve = kin.eig_for_pylmm()
    L = _quiet(lambda: LMM(y, K, kva, kve))
    null = np.ones(T.shape[1], dtype=bool)
    null[causal] = False
    out: dict = {"n": len(y), "m_null": int(null.sum())}
    for test in TESTS:
        p, dt = _quiet(lambda test=test: _scan(L, T, test, out))
        out["time_" + test] = dt
        out["us_per_snp_" + test] = 1e6 * dt / T.shape[1]
        out["lambda_" + test] = metrics.lambda_gc(p[null])
        for a in ALPHAS:
            out["n_%s_%s" % (metrics.alpha_label(a), test)] = int((p[null] <= a).sum())
        if len(causal):
            pc = p[causal]
            for a in POWER_ALPHAS:
                out["pow_%s_%s" % (metrics.alpha_label(a), test)] = int((pc <= a).sum())
            out["mlp_" + test] = float(np.mean(-np.log10(pc)))
    if params["scenario"] != "tail":
        L.fit()
        for reml, tag in ((True, "reml"), (False, "ml")):
            _quiet(lambda reml=reml, tag=tag: _refit_check(L, T, y, K, reml, out, tag))
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _table(rows, cols):
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r) for _, fmt in cols) + " |")
    return "\n".join(lines)


def _g(v, f="%.3g"):
    return "nan" if v is None or not np.isfinite(v) else f % v


def summarize(df):
    """One dict per (cohort, h2, scenario) cell with the numbers of the tables."""
    out = []
    for (cohort, h2, scen), d in df.groupby(["cohort", "h2", "scenario"], sort=False):
        r = {"cohort": cohort, "h2": h2, "scenario": scen, "reps": len(d), "n": int(d["n"].iloc[0])}
        n_null = float(d["m_null"].sum())
        r["n_null_tests"] = n_null
        for test in TESTS:
            r["lambda_" + test] = d["lambda_" + test].mean()
            r["us_per_snp_" + test] = d["us_per_snp_" + test].median()
            for a in ALPHAS:
                k = float(d["n_%s_%s" % (metrics.alpha_label(a), test)].sum())
                lo, hi = metrics.rate_ci(k, n_null)
                r["t1e_%s_%s" % (metrics.alpha_label(a), test)] = k / n_null
                r["t1e_%s_%s_count" % (metrics.alpha_label(a), test)] = k
                r["t1e_%s_%s_lo" % (metrics.alpha_label(a), test)] = lo
                r["t1e_%s_%s_hi" % (metrics.alpha_label(a), test)] = hi
            if scen == "power":
                for a in POWER_ALPHAS:
                    key = "pow_%s_%s" % (metrics.alpha_label(a), test)
                    r[key] = d[key].sum() / (N_CAUSAL * len(d))
                r["mlp_" + test] = d["mlp_" + test].mean()
        if "dh_reml" in d:
            for tag in ("reml", "ml"):
                for k in ("dh", "short_new", "short_old", "oracle_short", "oracle_dh"):
                    r["%s_%s" % (k, tag)] = d["%s_%s" % (k, tag)].max()
                r["fallbacks_" + tag] = d["fallbacks_" + tag].sum()
                r["at1_" + tag] = d["at1_" + tag].mean()
        for t in ("lrt", "lrt_plain"):
            r["h0_at1_" + t] = float((d["h0_" + t] >= 1.0 - 1e-9).mean())
            r["h0_mean_" + t] = float(d["h0_" + t].mean())
        # replicate overdispersion of the 0.05 counts: var over replicates / binomial var
        m_rep = d["m_null"].to_numpy(dtype=float)
        for test in TESTS:
            k = d["n_0.05_%s" % test].to_numpy(dtype=float)
            r["disp_0.05_" + test] = (
                float(np.var(k, ddof=1) / np.mean(m_rep * 0.05 * 0.95)) if len(k) > 1 else np.nan
            )
        out.append(r)
    return out


def report(df, outdir):
    import os

    import pandas as pd

    rows = summarize(df)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "summary.csv"), index=False)
    parts = ["# S9: Wald, Wald with the exact refit, LRT and score tests\n"]
    parts.append(_null_table([r for r in rows if r["scenario"] == "null"]))
    parts.append(_lrt_table([r for r in rows if r["scenario"] in ("null", "tail")]))
    parts.append(_tail_table([r for r in rows if r["scenario"] in ("null", "tail")]))
    parts.append(_power_table([r for r in rows if r["scenario"] == "power"]))
    parts.append(_refit_table([r for r in rows if "dh_reml" in r]))
    parts.append(_time_table([r for r in rows if r["scenario"] == "null"]))
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("".join(parts))


def _null_table(rows):
    parts = [
        "\n## Null calibration\n\nlambda_GC (mean over replicates) and type-I error at 0.05 and "
        "1e-3 (pooled), per test.\n\n"
    ]
    cols = [("cohort", lambda r: r["cohort"]), ("h2", lambda r: "%g" % r["h2"])]
    for t in TESTS:
        cols.append(("lambda %s" % t, lambda r, t=t: _g(r["lambda_" + t], "%.3f")))
    for a in ("0.05", "1e-03"):
        for t in TESTS:
            cols.append(("%s %s" % (a, t), lambda r, a=a, t=t: _g(r["t1e_%s_%s" % (a, t)], "%.4f")))
    parts.append(_table(rows, cols) + "\n")
    return "".join(parts)


def _lrt_table(rows):
    parts = [
        "\n## The LRT's null fit and the replicate dispersion\n\n"
        "Share of replicates whose null ML fit is at h = 1 (and the mean null h): reduced ML "
        "(lrt) vs plain ML (lrt_plain).  disp: the variance over replicates of the count of "
        "null p-values below 0.05, over its binomial variance (1 = independent tests; the "
        "Wilson intervals of the tail table assume 1).\n\n"
    ]
    cols = [
        ("cohort", lambda r: r["cohort"]),
        ("h2", lambda r: "%g" % r["h2"]),
        ("scenario", lambda r: r["scenario"]),
        ("at 1 lrt", lambda r: _g(r["h0_at1_lrt"], "%.2f")),
        ("at 1 lrt_plain", lambda r: _g(r["h0_at1_lrt_plain"], "%.2f")),
        ("mean h lrt", lambda r: _g(r["h0_mean_lrt"], "%.3f")),
        ("mean h lrt_plain", lambda r: _g(r["h0_mean_lrt_plain"], "%.3f")),
    ]
    for t in TESTS:
        cols.append(("disp %s" % t, lambda r, t=t: _g(r["disp_0.05_" + t], "%.1f")))
    parts.append(_table(rows, cols) + "\n")
    return "".join(parts)


def _tail_table(rows):
    parts = [
        "\n## The tail: type-I error at 1e-4 and 1e-5\n\nCount of null p-values below the "
        "threshold / number of null tests (pooled), and the rate over the threshold with its "
        "95% Wilson interval.\n\n"
    ]
    cols = [
        ("cohort", lambda r: r["cohort"]),
        ("h2", lambda r: "%g" % r["h2"]),
        ("scenario", lambda r: r["scenario"]),
        ("null tests", lambda r: "%d" % r["n_null_tests"]),
    ]
    for a in ("1e-04", "1e-05"):
        for t in TESTS:
            cols.append(
                (
                    "%s %s" % (a, t),
                    lambda r, a=a, t=t: (
                        "%d (%.2f, %.2f-%.2f)"
                        % (
                            r["t1e_%s_%s_count" % (a, t)],
                            r["t1e_%s_%s" % (a, t)] / float(a),
                            r["t1e_%s_%s_lo" % (a, t)] / float(a),
                            r["t1e_%s_%s_hi" % (a, t)] / float(a),
                        )
                    ),
                )
            )
    parts.append(_table(rows, cols) + "\n")
    return "".join(parts)


def _power_table(rows):
    parts = [
        "\n## Power\n\n%d causal SNPs per replicate, each explaining %s of the variance; share "
        "of causal SNPs with p below 1e-3 and 1e-5, and their mean -log10 p.\n\n"
        % (N_CAUSAL, ", ".join("%g (%s)" % (v, k) for k, v in VPE.items()))
    ]
    cols = [("cohort", lambda r: r["cohort"]), ("h2", lambda r: "%g" % r["h2"])]
    for a in ("1e-03", "1e-05"):
        for t in TESTS:
            cols.append(("%s %s" % (a, t), lambda r, a=a, t=t: _g(r["pow_%s_%s" % (a, t)], "%.3f")))
    for t in TESTS:
        cols.append(("-log10p %s" % t, lambda r, t=t: _g(r["mlp_" + t], "%.2f")))
    parts.append(_table(rows, cols) + "\n")
    return "".join(parts)


def _refit_table(rows):
    parts = [
        "\n## The exact refit vs the per-SNP grid search and the oracle\n\n"
        "First %d test SNPs per replicate: max |dh| between `fit_snps` and `fit(X=x)`; "
        "short new / short old: the largest log-likelihood by which the exact fit is below "
        "the grid search's / the reverse (> 1e-6 would be a missed maximum); oracle: the "
        "largest shortfall of the exact fit below the dense oracle's maximum (%d SNPs); "
        "at 1: share of exact fits at h = 1; fallbacks: brackets that fell back to the "
        "scalar search.\n\n" % (N_CHECK, N_ORACLE)
    ]
    cols = [
        ("cohort", lambda r: r["cohort"]),
        ("h2", lambda r: "%g" % r["h2"]),
        ("scenario", lambda r: r["scenario"]),
    ]
    for tag in ("reml", "ml"):
        cols += [
            ("|dh| %s" % tag, lambda r, t=tag: _g(r["dh_" + t], "%.1e")),
            ("short new %s" % tag, lambda r, t=tag: _g(r["short_new_" + t], "%.1e")),
            ("short old %s" % tag, lambda r, t=tag: _g(r["short_old_" + t], "%.1e")),
            ("oracle %s" % tag, lambda r, t=tag: _g(r["oracle_short_" + t], "%.1e")),
            ("at 1 %s" % tag, lambda r, t=tag: _g(r["at1_" + t], "%.2f")),
            ("fallbacks %s" % tag, lambda r, t=tag: "%d" % r["fallbacks_" + t]),
        ]
    parts.append(_table(rows, cols) + "\n")
    return "".join(parts)


def _time_table(rows):
    parts = [
        "\n## Cost\n\nMedian microseconds per SNP of one scan (null fit included, blocks of 1024, "
        "one process per job).\n\n"
    ]
    cols = [("cohort", lambda r: r["cohort"]), ("h2", lambda r: "%g" % r["h2"])]
    for t in TESTS:
        cols.append((t, lambda r, t=t: _g(r["us_per_snp_" + t], "%.1f")))
    parts.append(_table(rows, cols) + "\n")
    return "".join(parts)
