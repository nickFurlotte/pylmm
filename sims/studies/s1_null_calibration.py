"""S1: null calibration of pylmm p-values vs OLS.

Question: are pylmm's p-values uniform under the null (no causal SNPs) across
population structures and heritabilities, and how does that compare with OLS?
See s1_null_calibration.md for the design, hypotheses and interpretation.

Per replicate (one phenotype y = mu + u + e, u ~ N(0, h2 K), K the kinship of
the "K SNPs"; every tested SNP is null):

* test set B: SNPs disjoint from the K SNPs (no proximal contamination);
* test set A: a subset of the K SNPs themselves (tested SNPs are in K);
* methods on each test set:
    ols      per-SNP OLS t-test (no kinship)
    reml     pylmm GWAS(REML=True):  null h by REML, SNP test by REML Wald
    cli      what `pylmmGWAS` did by default before Phase 5.2 (no --REML, no
             --refit): null h by REML (bug 1), SNP test labelled ML but with
             sigma^2 = Q/(n-q) (bug 9), i.e. exactly the REML Wald test.
             Since Phase 5.2 GWAS(REML=False) is a real ML scan, so this is
             computed as GWAS(REML=True), which gives the same numbers as
             the pre-5.2 GWAS(REML=False) that produced the results.
  Suffix _in marks test set A (tested SNPs in K).
* metrics per replicate: lambda_GC, and *counts* k (p <= alpha) and n (SNPs
  tested) so that report() can pool exact rates over replicates and give
  binomial CIs down to alpha = 1e-4; plus the smallest and a random sample of
  p-values for a pooled QQ plot.

Missing genotypes (real BED data) are mean-imputed per SNP before testing, as
the pylmm CLI does by default; otherwise GWAS refits an eigendecomposition
for every SNP with a missing call (about 1.4% of calls are missing, so nearly
every SNP).  Tested SNPs must have MAF >= 0.05.

Run:  just sim s1_null_calibration --quick
      just sim s1_null_calibration --jobs 6          # full grid (the manager runs this)
"""

import json
import os

import numpy as np

from sims import genotypes, kinship, methods, metrics, phenotypes

NAME = "s1_null_calibration"

POPS = ["unrelated", "bn_fst0.05", "bn_fst0.2", "family", "bed"]
H2S = [0.0, 0.2, 0.5, 0.8, 0.95]

GRID = {
    "pop": POPS,
    "h2": H2S,
    "n": [600],
    "m_kin": [6000],
    "m_test": [5000],
    "geno_seed": [1],
}
REPS = 80  # 25 cells x 80 reps x 5000 SNPs = 4e5 tests per cell/method

QUICK_GRID = {
    "pop": ["unrelated", "family", "bed"],
    "h2": [0.5],
    "n": [200],
    "m_kin": [1000],
    "m_test": [300],
    "geno_seed": [1],
}
QUICK_REPS = 2

ALPHAS = (0.05, 1e-2, 1e-3, 1e-4)
METHODS = ("ols", "reml", "cli", "ols_in", "reml_in", "cli_in")
MIN_MAF = 0.05
N_TAIL = 20  # smallest p-values kept per method and replicate
N_RAND = 200  # random p-values kept per method and replicate

_SETUP = {}  # per-worker cache: geometry of one cell (independent of h2)


def _geno_spec(params):
    m = params["m_kin"] + params["m_test"] + params["m_test"] // 2  # margin for the MAF filter
    n, seed, pop = params["n"], params["geno_seed"], params["pop"]
    if pop == "unrelated":
        return {"source": "unrelated", "n": n, "m": m, "seed": seed}
    if pop.startswith("bn_fst"):
        return {
            "source": "balding_nichols",
            "n": n,
            "m": m,
            "k": 3,
            "fst": float(pop[len("bn_fst") :]),
            "seed": seed,
        }
    if pop == "family":
        return {"source": "family", "n_families": n // 4, "n_sibs": 4, "m": m, "seed": seed}
    if pop == "bed":
        return {"source": "bed", "n_snps": m, "n_indivs": n, "select": "random", "seed": seed}
    raise ValueError(pop)


def _impute_filter(G, idx, m_test):
    """Columns of G[:, idx] with MAF >= MIN_MAF (first m_test of them), missing
    calls replaced by the SNP mean (the pylmm CLI default)."""
    G = np.array(G[:, idx], dtype=np.float64)
    mean = np.nanmean(G, axis=0)
    # allele frequency from the observed calls; the BED encoding is 0/0.5/1,
    # synthetic 0/1/2, so rescale by the max code
    scale = np.where(np.nanmax(G, axis=0) > 1.0, 2.0, 1.0)
    f = mean / scale
    ok = np.minimum(f, 1 - f) >= MIN_MAF
    G = G[:, ok][:, :m_test]
    bad = np.isnan(G)
    if bad.any():
        G[bad] = np.broadcast_to(np.nanmean(G, axis=0), G.shape)[bad]
    return G


def _setup(params):
    key = json.dumps({k: v for k, v in params.items() if k != "h2"}, sort_keys=True)
    if key not in _SETUP:
        if len(_SETUP) >= 2:
            _SETUP.pop(next(iter(_SETUP)))
        spec = _geno_spec(params)
        geno = genotypes.get(spec)
        mk, mt = params["m_kin"], params["m_test"]
        kin = kinship.get(spec, snps=slice(0, mk))  # K from SNPs [0, mk)
        G_dis = _impute_filter(geno.G, np.arange(mk, geno.G.shape[1]), mt)  # disjoint from K
        G_in = _impute_filter(geno.G, np.arange(0, mk), mt)  # inside K
        _SETUP[key] = (kin, G_dis, G_in)
    return _SETUP[key]


def _tail_and_random(p, rng):
    p = p[~np.isnan(p)]
    tail = np.sort(p)[:N_TAIL]
    rand = rng.choice(p, size=min(N_RAND, len(p)), replace=False)
    return tail.tolist(), rand.tolist()


def run_cell(params, seed):
    rng = np.random.default_rng(seed)
    kin, G_dis, G_in = _setup(params)
    ph = phenotypes.simulate(kin, h2=params["h2"], rng=rng)
    y = ph.y

    out = {
        "h_hat": methods.pylmm_null(y, kin, REML=True)["h"],
        "m_dis": G_dis.shape[1],
        "m_in": G_in.shape[1],
    }
    pv = {}
    for suffix, G in (("", G_dis), ("_in", G_in)):
        pv["ols" + suffix] = methods.ols_gwas(y, G)[1]
        pv["reml" + suffix] = methods.pylmm_gwas(y, G, kin, REML=True)[1]
        # pre-5.2 CLI default == REML Wald (bugs 1 and 9); see the docstring
        pv["cli" + suffix] = methods.pylmm_gwas(y, G, kin, REML=True)[1]

    for meth in METHODS:
        p = np.asarray(pv[meth], dtype=np.float64)
        ok = ~np.isnan(p)
        out["lam_" + meth] = metrics.lambda_gc(p)
        out["n_" + meth] = int(ok.sum())
        for a in ALPHAS:
            out["k_%s_%s" % (meth, metrics.alpha_label(a))] = int((p[ok] <= a).sum())
        if meth in ("ols", "reml", "cli", "reml_in"):
            out["qqtail_" + meth], out["qqrand_" + meth] = _tail_and_random(p, rng)
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _lambda_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)


def build_summary(df):
    import pandas as pd

    rows = []
    for (pop, h2), d in df.groupby(["pop", "h2"], sort=False):
        for meth in METHODS:
            lam, lo, hi = _lambda_ci(d["lam_" + meth])
            nt = d["n_" + meth].sum()
            row = {
                "pop": pop,
                "h2": h2,
                "method": meth,
                "n_reps": len(d),
                "n_tests": int(nt),
                "lambda_gc": lam,
                "lambda_lo": lo,
                "lambda_hi": hi,
                "h_hat_mean": d["h_hat"].mean(),
            }
            for a in ALPHAS:
                lab = metrics.alpha_label(a)
                k = d["k_%s_%s" % (meth, lab)].sum()
                lo_, hi_ = metrics.rate_ci(k, nt)
                row["t1e_" + lab] = k / nt if nt else np.nan
                row["t1e_%s_lo" % lab], row["t1e_%s_hi" % lab] = lo_, hi_
                row["hits_" + lab] = int(k)
                row["ratio_" + lab] = (k / nt) / a if nt else np.nan
                row["outside_" + lab] = bool(lo_ > a or hi_ < a)
            rows.append(row)
    return pd.DataFrame(rows)


def _pooled_qq(d, meth):
    """(expected, observed) -log10 p from the stored tail + random samples."""
    tails = np.sort(np.concatenate([np.asarray(x, float) for x in d["qqtail_" + meth]]))
    rand = np.sort(np.concatenate([np.asarray(x, float) for x in d["qqrand_" + meth]]))
    N = int(d["n_" + meth].sum())
    S = len(rand)
    j = np.arange(1, S + 1)
    keep = j >= 30  # bulk from the random sample; the extreme tail is exact below
    ex_b, ob_b = -np.log10((j[keep] - 0.5) / S), -np.log10(np.clip(rand[keep], 1e-300, 1))
    # the pooled i-th smallest p (i <= N_TAIL) lies in the union of per-replicate tails
    i = np.arange(1, min(N_TAIL, len(tails)) + 1)
    ex_t, ob_t = -np.log10((i - 0.5) / N), -np.log10(np.clip(tails[: len(i)], 1e-300, 1))
    if len(ex_b) > 150:
        sel = np.unique(np.linspace(0, len(ex_b) - 1, 150).astype(int))
        ex_b, ob_b = ex_b[sel], ob_b[sel]
    return np.r_[ex_b, ex_t], np.r_[ob_b, ob_t]


def _plot_qq(df, outdir):
    from sims import plots

    plt = plots._plt()
    pops = [p for p in POPS if p in set(df["pop"])]
    h2s = sorted(df["h2"].unique())
    show = [h for h in (0.0, 0.5, 0.95) if h in h2s] or h2s[:3]
    series = [
        ("ols", "OLS", plots.SERIES[1]),
        ("reml", "pylmm REML", plots.SERIES[0]),
        ("cli", "pylmm CLI (ML test)", plots.SERIES[2]),
        ("reml_in", "pylmm REML, SNPs in K", plots.SERIES[6]),
    ]
    fig, axes = plt.subplots(
        len(pops), len(show), figsize=(3.1 * len(show), 2.9 * len(pops)), squeeze=False
    )
    for r, pop in enumerate(pops):
        for c, h2 in enumerate(show):
            ax = axes[r][c]
            d = df[(df["pop"] == pop) & (df["h2"] == h2)]
            if d.empty:
                continue
            mx = 0
            for meth, lab, col in series:
                ex, ob = _pooled_qq(d, meth)
                ax.plot(ex, ob, "o", ms=1.8, color=col, label=lab)
                mx = max(mx, ex.max())
            ax.plot([0, mx], [0, mx], color=plots.REF, lw=1)
            ax.set_title("%s, h2=%g" % (pop, h2), fontsize=8)
            if r == len(pops) - 1:
                ax.set_xlabel("expected -log10 p")
            if c == 0:
                ax.set_ylabel("observed -log10 p")
            if r == 0 and c == 0:
                ax.legend(fontsize=6, loc="upper left", markerscale=3)
    return plots._save(fig, os.path.join(outdir, "qq_panel.png"))


def _plot_lambda(summary, outdir):
    from sims import plots

    plt = plots._plt()
    pops = [p for p in POPS if p in set(summary["pop"])]
    fig, axes = plt.subplots(
        1, len(pops), figsize=(2.9 * len(pops), 3.0), squeeze=False, sharey=True
    )
    show = [
        ("ols", "OLS", "o", "-", plots.SERIES[1]),
        ("reml", "pylmm REML", "o", "-", plots.SERIES[0]),
        ("cli", "pylmm CLI (ML test)", "s", "--", plots.SERIES[2]),
        ("reml_in", "pylmm REML, SNPs in K", "^", ":", plots.SERIES[6]),
    ]
    for ax, pop in zip(axes[0], pops):
        for meth, lab, mk, ls, col in show:
            d = summary[(summary["pop"] == pop) & (summary["method"] == meth)].sort_values("h2")
            ax.errorbar(
                d["h2"],
                d["lambda_gc"],
                yerr=[d["lambda_gc"] - d["lambda_lo"], d["lambda_hi"] - d["lambda_gc"]],
                marker=mk,
                ms=3.5,
                ls=ls,
                capsize=2,
                lw=1.2,
                color=col,
                label=lab,
            )
        ax.axhline(1.0, color=plots.REF, lw=1, ls="--")
        ax.set_title(pop, fontsize=9)
        ax.set_xlabel("h2")
    axes[0][0].set_ylabel("lambda_GC (mean, 95% CI over reps)")
    axes[0][0].legend(fontsize=6)
    if summary["lambda_gc"].max() > 3:
        axes[0][0].set_yscale("log")
    return plots._save(fig, os.path.join(outdir, "lambda_vs_h2.png"))


def report(df, outdir):
    from sims import report as rpt

    summary = build_summary(df)
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    md = summary[["pop", "h2", "method", "n_reps", "n_tests"]].copy()
    md["lambda_GC [95% CI]"] = [
        "%.3f [%.3f, %.3f]" % (r.lambda_gc, r.lambda_lo, r.lambda_hi) for r in summary.itertuples()
    ]
    for a in ALPHAS:
        lab = metrics.alpha_label(a)
        md["obs/alpha @%g" % a] = [
            "%.2g%s" % (r["ratio_" + lab], "*" if r["outside_" + lab] else "")
            for _, r in summary.iterrows()
        ]
    md["hits@1e-4"] = summary["hits_1e-04"]
    notes = (
        "Null SNPs only: lambda_GC should be 1 and each obs/alpha ratio 1 (observed type-I error "
        "divided by nominal alpha, pooled over replicates and SNPs; `*` = the 95% Wilson interval "
        "excludes 1; the interval ignores correlation between SNPs and is optimistic for real LD "
        "data).  Methods: ols; reml = pylmm GWAS(REML=True); cli = GWAS(REML=False), i.e. the "
        "pylmmGWAS default (REML null fit, ML test); suffix _in = tested SNPs are part of K, "
        "otherwise tested SNPs are disjoint from K.  Full numeric columns (CIs, hit counts) are "
        "in summary.csv.  See s1_null_calibration.md for interpretation."
    )
    rpt.write_summary(md, outdir, "S1: null calibration", notes=notes)
    _plot_qq(df, outdir)
    _plot_lambda(summary, outdir)
