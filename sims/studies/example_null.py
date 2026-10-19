"""Example study: null calibration of pylmm vs OLS on synthetic unrelated data.

This is the template the Phase 4 studies follow (see sims/README.md).  It is
deliberately tiny.  Per replicate:

1. genotypes: one fixed synthetic cohort per cell (cached per worker), SNPs
   split into a kinship set and a disjoint test set;
2. phenotype: y = mu + u + e with u ~ N(0, h2 K) -- no causal SNPs, so every
   tested SNP is null;
3. pylmm GWAS (REML, no refit) and an OLS scan over the test SNPs;
4. metrics: lambda_GC, type-I error at a few alphas, the fitted h, and a
   pylmm-vs-oracle check of h, the log-likelihood and a few t-statistics;
5. a small random sample of null p-values for a pooled QQ plot.

Run:   just sim example_null --quick      (or: uv run --group sims python -m sims run example_null --quick)
       just sim-report example_null --quick
"""

import numpy as np

from sims import genotypes, kinship, methods, metrics, oracle, phenotypes

NAME = "example_null"

# Full grid: cartesian product of these lists.
GRID = {
    "n": [200, 500],
    "h2": [0.0, 0.5],
    "m_kin": [2000],
    "m_test": [2000],
    "geno_seed": [1],
}
REPS = 20

# --quick: must finish in well under a minute single-core.
QUICK_GRID = {
    "n": [200],
    "h2": [0.0, 0.5],
    "m_kin": [1000],
    "m_test": [300],
    "geno_seed": [1],
}
QUICK_REPS = 2

ALPHAS = (0.05, 0.01, 1e-3)
N_ORACLE_SNPS = 10  # t-stats compared with the oracle per replicate
N_QQ_KEEP = 200  # null p-values kept per replicate for the pooled QQ plot


def run_cell(params, seed):
    rng = np.random.default_rng(seed)

    # 1. genotypes (cached per worker: same spec -> same matrix, no recompute)
    spec = {
        "source": "unrelated",
        "n": params["n"],
        "m": params["m_kin"] + params["m_test"],
        "seed": params["geno_seed"],
    }
    geno = genotypes.get(spec)
    kin = kinship.get(spec, snps=slice(0, params["m_kin"]))  # cached K + eigh
    g_test = geno.G[:, params["m_kin"] :]

    # 2. phenotype under the null for every tested SNP
    ph = phenotypes.simulate(kin, h2=params["h2"], rng=rng)

    # 3. methods
    null = methods.pylmm_null(ph.y, kin, reml=True)
    t_lmm, p_lmm = methods.pylmm_gwas(ph.y, g_test, kin, reml=True, refit=False)
    t_ols, p_ols = methods.ols_gwas(ph.y, g_test)

    # 4. metrics
    out = {
        "h_hat": null["h"],
        "sigma_hat": null["sigma"],
        "lambda_lmm": metrics.lambda_gc(p_lmm),
        "lambda_ols": metrics.lambda_gc(p_ols),
    }
    out.update({k + "_lmm": v for k, v in metrics.type1_error(p_lmm, ALPHAS).items()})
    out.update({k + "_ols": v for k, v in metrics.type1_error(p_ols, ALPHAS).items()})

    # independent oracle (dense REML), for small n only
    if params["n"] <= 500:
        ofit = oracle.fit(ph.y, np.asarray(kin.K), reml=True)
        out["h_oracle"] = ofit.h
        out["dh_oracle"] = null["h"] - ofit.h
        # pylmm's REML LL = oracle REML LL + a known h-independent constant
        out["dll_oracle"] = null["ll"] - (ofit.ll + oracle.pylmm_reml_offset(params["n"], 1))
        idx = rng.choice(g_test.shape[1], size=N_ORACLE_SNPS, replace=False)
        t_or = [oracle.wald(ph.y, np.asarray(kin.K), g_test[:, j], h=ofit.h).t for j in idx]
        out.update(metrics.compare(t_lmm[idx], t_or, prefix="t_vs_oracle_"))

    # 5. a small array: random null p-values for the pooled QQ plot
    keep = rng.choice(len(p_lmm), size=min(N_QQ_KEEP, len(p_lmm)), replace=False)
    out["qq_p_lmm"] = p_lmm[keep].tolist()
    out["qq_p_ols"] = p_ols[keep].tolist()
    return out


def report(df, outdir):
    import os

    from sims import plots
    from sims import report as rpt

    by = ["n", "h2", "m_kin", "m_test"]
    scalar = [c for c in rpt.scalar_metric_columns(df) if c not in [*by, "geno_seed"]]
    summary = rpt.summarize(df, by, scalar)
    rpt.write_summary(
        summary,
        outdir,
        "Example: null calibration (synthetic unrelated)",
        notes=(
            "Null SNPs only, so lambda_GC should be ~1 and t1e_<alpha> ~ alpha for "
            "both methods.  dh_oracle / dll_oracle / t_vs_oracle_* compare pylmm "
            "with the independent dense REML oracle (sims/oracle.py).  All columns "
            "(with standard errors over replicates) are in summary.csv."
        ),
        md_columns=[
            c
            for c in summary.columns
            if not c.endswith("_se")
            and c
            not in (
                "m_kin",
                "sigma_hat",
                "t_vs_oracle_mean_abs",
                "t_vs_oracle_max_rel",
                "t_vs_oracle_corr",
            )
        ],
    )
    for (n, h2), d in df.groupby(["n", "h2"]):
        plots.qq(
            {
                "pylmm LMM": metrics.pool_pvalues(d["qq_p_lmm"]),
                "OLS": metrics.pool_pvalues(d["qq_p_ols"]),
            },
            os.path.join(outdir, "qq_n%d_h%g.png" % (n, h2)),
            title="null QQ, n=%d, h2=%g (pooled sample)" % (n, h2),
        )
    plots.lines(
        summary,
        "h2",
        "h_hat",
        os.path.join(outdir, "h_hat.png"),
        hue="n",
        yerr="h_hat_se",
        identity=True,
        title="fitted h (pylmm REML)",
        ylabel="mean h_hat",
    )
