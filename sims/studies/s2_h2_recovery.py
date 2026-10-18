"""S2: heritability (and sigma^2, beta) recovery of pylmm under REML and ML.

See s2_h2_recovery.md for the design, hypotheses and interpretation.

Per replicate (one phenotype y = 1*mu + u + e, truth h2, sigma^2 = 1, mu = 1):

* pylmm null fit, REML and ML (``methods.pylmm_null``): h, sigma, beta, and
  pylmm's *own* log-likelihood on a 201-point grid over [0, 1] (cheap, uses the
  fitted LMM object) -> number of profile modes and the LL deficit of pylmm's
  answer relative to the best grid point ("non-global optimum").
* oracle (``sims.oracle``) REML and ML, in four variants:
    T  textbook (fine global search, no clamp)       = what any correct method does
    G  h_grid="pylmm" only                           = optimizer effect
    C  eig_clamp=1e-6 only                           = clamp effect
    B  both                                          = reproduces pylmm
  For n = 600 the oracle is run on a subset of replicates (seed % ORACLE_MOD_600
  == 0) because it costs O(n^3) per likelihood evaluation.

``kind == "probe"`` cells are the small multiple-optima probe (n = 100,
deliberately misspecified phenotypes); they have no oracle.
"""

import numpy as np

from sims import genotypes, kinship, methods, oracle, phenotypes

NAME = "s2_h2_recovery"

H2S = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 1.0]
SOURCES = ["unrelated", "balding_nichols", "family", "bed"]
NS = [200, 600]
MKS = ["lo", "hi"]          # kinship SNPs: n/2 ("lo", m < n) or 5n ("hi", m > n)
GENO_SEEDS = [1, 2]

MU = 1.0                    # true intercept (beta)
REPS = 12
ORACLE_MOD_600 = 3          # at n = 600 the oracle runs on 1 replicate in 3
CLAMP = 1e-6
LL_GRID = np.linspace(0.0, 1.0, 201)
PROBES = ["correct", "mix_family", "mix_bn", "mix_rank"]

GRID = ([{"kind": "main", "src": s, "n": n, "mk": mk, "h2": h, "geno_seed": g}
         for s in SOURCES for n in NS for mk in MKS for h in H2S for g in GENO_SEEDS]
        + [{"kind": "probe", "src": p, "n": 100, "mk": "hi", "h2": h, "geno_seed": 1}
           for p in PROBES for h in (0.2, 0.5, 0.8)])

QUICK_GRID = ([{"kind": "main", "src": s, "n": 200, "mk": mk, "h2": h, "geno_seed": 1}
               for s in ("unrelated", "family") for mk in MKS for h in (0.5, 1.0)]
              + [{"kind": "main", "src": "bed", "n": 200, "mk": "lo", "h2": 0.5, "geno_seed": 1},
                 {"kind": "probe", "src": "mix_family", "n": 100, "mk": "hi", "h2": 0.5,
                  "geno_seed": 1}])
QUICK_REPS = 2


def reps(quick):
    return QUICK_REPS if quick else REPS


# ---------------------------------------------------------------------------

def m_kin(n, mk):
    return n // 2 if mk == "lo" else 5 * n


def geno_spec(src, n, m, seed):
    if src == "unrelated":
        return {"source": "unrelated", "n": n, "m": m, "seed": seed}
    if src == "balding_nichols":
        return {"source": "balding_nichols", "n": n, "m": m, "k": 3, "fst": 0.1, "seed": seed}
    if src == "family":
        return {"source": "family", "n_families": n // 4, "n_sibs": 4, "m": m, "seed": seed}
    if src == "bed":
        return {"source": "bed", "n_snps": m, "n_indivs": n, "select": "random", "seed": seed}
    raise ValueError(src)


def n_modes(L, prom=0.1):
    """Number of local maxima of an LL profile with prominence >= ``prom`` LL
    units (endpoints count as maxima if they are higher than their neighbour)."""
    from scipy.signal import find_peaks
    L = np.asarray(L, float)
    fin = np.isfinite(L)
    if not fin.any():
        return 0
    L = np.where(fin, L, L[fin].min() - 1.0)
    pad = L.min() - 1e3
    pk, _ = find_peaks(np.r_[pad, L, pad], prominence=prom)
    return int(len(pk))


def pylmm_profile(lmm, reml):
    """pylmm's own LL (clamped eigenvalues) on LL_GRID; REML includes pylmm's
    constant, irrelevant for deficits."""
    return np.array([float(np.sum(lmm.LL(h, lmm.X0t, stack=False, REML=reml)[0]))
                     for h in LL_GRID])


def pylmm_block(prefix, y, kin, reml, n):
    r = methods.pylmm_null(y, kin, REML=reml)
    L = pylmm_profile(r["lmm"], reml)
    ll_hat = float(r["ll"])
    h = r["h"]
    out = {
        prefix + "h_pylmm": h,
        prefix + "sigma_pylmm": r["sigma"],
        prefix + "sigma_ml_pylmm": r["sigma"] * (n - 1) / n,    # Q/n implied by pylmm's Q/(n-q)
        prefix + "beta_pylmm": float(r["beta"][0]),
        prefix + "ll_deficit": float(max(0.0, np.nanmax(L) - ll_hat)),
        prefix + "h_gridmax": float(LL_GRID[int(np.nanargmax(L))]),
        prefix + "n_modes": n_modes(L),
        prefix + "hit0": int(h <= 1e-6),
        prefix + "hit99": int(h >= 0.99 - 1e-6),
    }
    # does pylmm's first-optimum rule lose to the best *other* mode?
    out[prefix + "nonglobal"] = int(out[prefix + "ll_deficit"] > 1e-3)
    # ... specifically because a *different mode* is higher (not the 0.99 grid cap)
    out[prefix + "lostmode"] = int(out[prefix + "nonglobal"] and out[prefix + "n_modes"] > 1)
    return out


def oracle_block(prefix, y, K, reml):
    """Four oracle variants for one likelihood."""
    fT = oracle.fit(y, K, reml=reml)
    fG = oracle.fit(y, K, reml=reml, h_grid="pylmm")
    fC = oracle.fit(y, K, reml=reml, eig_clamp=CLAMP)
    fB = oracle.fit(y, K, reml=reml, eig_clamp=CLAMP, h_grid="pylmm")
    out = {prefix + "h_T": fT.h, prefix + "h_G": fG.h, prefix + "h_C": fC.h,
           prefix + "h_B": fB.h,
           prefix + "beta_T": float(fT.beta[0]),
           prefix + "sigma2_T": fT.sigma2,
           prefix + "hit1_T": int(fT.h >= 1 - 1e-6),
           prefix + "n_modes_T": n_modes(fT.ll_grid),
           # LL (textbook, unclamped) of pylmm-optimizer answer vs the global answer
           prefix + "ll_deficit_G": float(max(0.0, fT.ll - oracle.profile_loglik(
               fG.h, y, K, None, reml)[0])),
           prefix + "ll_deficit_B": float(max(0.0, fC.ll - fB.ll)),
           }
    if not reml:
        out["ml_unbounded"] = int(fT.ml_unbounded)
        out["sigma2_ml_T"] = fT.sigma2_ml
        out["sigma2_reml_T"] = fT.sigma2_reml
    return out


# ---------------------------------------------------------------------------

def run_main(params, seed):
    rng = np.random.default_rng(seed)
    n, mk = params["n"], params["mk"]
    spec = geno_spec(params["src"], n, m_kin(n, mk), params["geno_seed"])
    kin = kinship.get(spec)
    ph = phenotypes.simulate(kin, h2=params["h2"], rng=rng, mu=MU)
    y = ph.y
    out = {"rank_K": int((kin.Kva > 1e-6).sum()), "frac_clamped": float((kin.Kva < CLAMP).mean())}
    out.update(pylmm_block("r_", y, kin, True, n))
    out.update(pylmm_block("m_", y, kin, False, n))
    do_oracle = n <= 200 or seed % ORACLE_MOD_600 == 0
    out["has_oracle"] = int(do_oracle)
    if do_oracle:
        K = np.asarray(kin.K)
        out.update(oracle_block("r_", y, K, True))
        out.update(oracle_block("m_", y, K, False))
        out["r_dh_B_vs_pylmm"] = out["r_h_B"] - out["r_h_pylmm"]
        out["m_dh_B_vs_pylmm"] = out["m_h_B"] - out["m_h_pylmm"]
    return out


def run_probe(params, seed):
    """Misspecified phenotypes on n = 100, fitted with one K: does the profile
    become multimodal, and does pylmm's first-optimum rule pick a worse mode?"""
    rng = np.random.default_rng(seed)
    n = params["n"]
    kind = params["src"]
    kin = kinship.get(geno_spec("balding_nichols", n, 5 * n, params["geno_seed"]))
    h2 = params["h2"]
    ph = phenotypes.simulate(kin, h2=h2, rng=rng, mu=MU)
    y = ph.y.copy()
    if kind == "mix_family":      # extra family component not in K (K is balding_nichols)
        k2 = kinship.get(geno_spec("family", n, 5 * n, params["geno_seed"]))
        y = y + np.sqrt(1.0) * phenotypes.sample_polygenic(k2.Kva, k2.Kve, 1.0, rng)
    elif kind == "mix_bn":        # extra component from a different family draw (seed+1)
        k2 = kinship.get(geno_spec("family", n, 5 * n, params["geno_seed"] + 1))
        y = y + phenotypes.sample_polygenic(k2.Kva, k2.Kve, 1.0, rng)
    elif kind == "mix_rank":      # signal on a few top eigenvectors only (low-rank, spiky)
        top = np.argsort(kin.Kva)[-3:]
        y = y + 3.0 * np.asarray(kin.Kve)[:, top] @ rng.standard_normal(3)
    out = {}
    for pre, reml in (("r_", True), ("m_", False)):
        out.update(pylmm_block(pre, y, kin, reml, n))
    return out


def run_cell(params, seed):
    if params.get("kind") == "probe":
        return run_probe(params, seed)
    return run_main(params, seed)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _stat(x, truth):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, bias=np.nan, rmse=np.nan, sd=np.nan)
    return dict(n=len(x), mean=x.mean(), bias=x.mean() - truth,
                rmse=float(np.sqrt(np.mean((x - truth) ** 2))), sd=x.std(ddof=1) if len(x) > 1 else np.nan)


METHODS = [("pylmm REML", "r_h_pylmm"), ("pylmm ML", "m_h_pylmm"),
           ("oracle REML", "r_h_T"), ("oracle ML", "m_h_T")]
KEYS = ["src", "n", "mk", "h2"]


def bias_rmse_table(d):
    import pandas as pd
    rows = []
    for key, g in d.groupby(KEYS, sort=False):
        row = dict(zip(KEYS, key))
        row["reps"] = len(g)
        for name, col in METHODS:
            if col not in g:
                continue
            s = _stat(g[col], key[3])
            tag = name.replace("pylmm ", "p").replace("oracle ", "o")
            row["bias " + tag] = s["bias"]
            row["rmse " + tag] = s["rmse"]
            row["sd " + tag] = s["sd"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["src", "n", "mk", "h2"]).reset_index(drop=True)


def boundary_table(d):
    import pandas as pd
    rows = []
    for key, g in d.groupby(["src", "n", "mk", "h2"], sort=False):
        row = dict(zip(["src", "n", "mk", "h2"], key))
        row["reps"] = len(g)
        for pre, nm in (("r_", "REML"), ("m_", "ML")):
            row["%s pylmm h=0" % nm] = g[pre + "hit0"].mean()
            row["%s pylmm h=.99" % nm] = g[pre + "hit99"].mean()
            row["%s non-global" % nm] = g[pre + "nonglobal"].mean()
            row["%s lost-mode" % nm] = g[pre + "lostmode"].mean()
            row["%s multi-modal" % nm] = (g[pre + "n_modes"] > 1).mean()
        o = g[g["has_oracle"] == 1]
        row["oracle REML h=1"] = o["r_hit1_T"].mean() if len(o) else np.nan
        row["oracle ML h=1"] = o["m_hit1_T"].mean() if len(o) else np.nan
        row["ml_unbounded"] = o["ml_unbounded"].mean() if len(o) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["src", "n", "mk", "h2"]).reset_index(drop=True)


def attribution_table(d, by=("src", "n", "mk", "h2")):
    """Decompose bias(pylmm) = est + opt + clamp + interaction on oracle reps:
    est = T - truth (any correct method), opt = G - T, clamp = C - T,
    inter = B - G - C + T, total = pylmm - truth.  ``rms_*`` are root-mean-square
    sizes of the per-replicate effects (they do not cancel)."""
    import pandas as pd
    o = d[d["has_oracle"] == 1]
    rows = []
    for key, g in o.groupby(list(by), sort=False):
        for pre, nm in (("r_", "REML"), ("m_", "ML")):
            T, G, C, B = (g[pre + "h_" + v].to_numpy() for v in "TGCB")
            P = g[pre + "h_pylmm"].to_numpy()
            truth = g["h2"].to_numpy()
            row = dict(zip(by, key))
            row.update(method=nm, reps=len(g),
                       total=np.mean(P - truth), est=np.mean(T - truth),
                       opt=np.mean(G - T), clamp=np.mean(C - T),
                       inter=np.mean(B - G - C + T),
                       rms_opt=np.sqrt(np.mean((G - T) ** 2)),
                       rms_clamp=np.sqrt(np.mean((C - T) ** 2)),
                       rms_pylmm_vs_T=np.sqrt(np.mean((P - T) ** 2)),
                       rmse_T=np.sqrt(np.mean((T - truth) ** 2)),
                       rmse_pylmm=np.sqrt(np.mean((P - truth) ** 2)),
                       max_abs_B_vs_pylmm=np.max(np.abs(B - P)))
            rows.append(row)
    return pd.DataFrame(rows)


def sigma_beta_table(d):
    import pandas as pd
    rows = []
    for key, g in d.groupby(["src", "n", "mk", "h2"], sort=False):
        row = dict(zip(["src", "n", "mk", "h2"], key))
        row["reps"] = len(g)
        row["sigma pylmm REML"] = g["r_sigma_pylmm"].mean()
        row["sigma pylmm ML(reported)"] = g["m_sigma_pylmm"].mean()
        row["sigma2_ml implied by pylmm ML"] = g["m_sigma_ml_pylmm"].mean()
        o = g[g["has_oracle"] == 1]
        row["oracle ML sigma2_ml"] = o["sigma2_ml_T"].mean() if "sigma2_ml_T" in o and len(o) else np.nan
        row["oracle ML sigma2_reml"] = o["sigma2_reml_T"].mean() if "sigma2_reml_T" in o and len(o) else np.nan
        row["oracle REML sigma2"] = o["r_sigma2_T"].mean() if len(o) else np.nan
        row["beta bias pylmm REML"] = g["r_beta_pylmm"].mean() - MU
        row["beta bias pylmm ML"] = g["m_beta_pylmm"].mean() - MU
        row["beta sd pylmm REML"] = g["r_beta_pylmm"].std(ddof=1)
        row["beta sd pylmm ML"] = g["m_beta_pylmm"].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["src", "n", "mk", "h2"]).reset_index(drop=True)


def ml_cap_table(d):
    """ML vs REML where the ML profile is unbounded (m > n, outbred K)."""
    import pandas as pd
    rows = []
    for key, g in d.groupby(["src", "n", "mk"], sort=False):
        o = g[g["has_oracle"] == 1]
        row = dict(zip(["src", "n", "mk"], key))
        row["reps"] = len(g)
        row["rank K"] = g["rank_K"].mean()
        row["ml_unbounded"] = o["ml_unbounded"].mean() if len(o) else np.nan
        row["pylmm ML h=.99"] = g["m_hit99"].mean()
        row["pylmm REML h=.99"] = g["r_hit99"].mean()
        row["bias pylmm ML"] = (g["m_h_pylmm"] - g["h2"]).mean()
        row["bias pylmm REML"] = (g["r_h_pylmm"] - g["h2"]).mean()
        row["rmse pylmm ML"] = float(np.sqrt(((g["m_h_pylmm"] - g["h2"]) ** 2).mean()))
        row["rmse pylmm REML"] = float(np.sqrt(((g["r_h_pylmm"] - g["h2"]) ** 2).mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def probe_table(p):
    import pandas as pd
    rows = []
    for key, g in p.groupby(["src", "h2"], sort=False):
        row = dict(zip(["probe", "h2"], key))
        row["reps"] = len(g)
        for pre, nm in (("r_", "REML"), ("m_", "ML")):
            row[nm + " multi-modal"] = (g[pre + "n_modes"] > 1).mean()
            row[nm + " non-global"] = g[pre + "nonglobal"].mean()
            row[nm + " lost-mode"] = g[pre + "lostmode"].mean()
            bad = g[g[pre + "nonglobal"] == 1]
            row[nm + " mean LL loss | non-global"] = bad[pre + "ll_deficit"].mean() if len(bad) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_recovery(d, outdir):
    import os

    from sims import plots
    plt = plots._plt()
    paths = []
    styles = {"pylmm REML": ("#1f77b4", "-", "o"), "pylmm ML": ("#d62728", "-", "s"),
              "oracle REML": ("#1f77b4", "--", "^"), "oracle ML": ("#d62728", "--", "v")}
    for src, g in d.groupby("src"):
        ns = sorted(g["n"].unique())
        mks = sorted(g["mk"].unique(), key=lambda s: ["lo", "hi"].index(s))
        fig, axes = plt.subplots(len(ns), len(mks), figsize=(4.6 * len(mks), 3.7 * len(ns)),
                                 squeeze=False, sharex=True, sharey=True)
        for i, n in enumerate(ns):
            for j, mk in enumerate(mks):
                ax = axes[i][j]
                c = g[(g["n"] == n) & (g["mk"] == mk)]
                ax.plot([0, 1], [0, 1], color="grey", lw=1, ls=":")
                ax.axhline(0.99, color="grey", lw=0.6, ls=":")
                for name, col in METHODS:
                    if col not in c:
                        continue
                    s = c.groupby("h2")[col].agg(["mean", "std"]).dropna(subset=["mean"])
                    col_, ls, mk_ = styles[name]
                    ax.errorbar(s.index, s["mean"], yerr=s["std"], color=col_, ls=ls,
                                marker=mk_, ms=3, capsize=1.5, lw=1, label=name, alpha=0.9)
                ax.set_title("%s, n=%d, m_K=%s" % (src, n, "n/2" if mk == "lo" else "5n"), fontsize=9)
                if i == len(ns) - 1:
                    ax.set_xlabel("true h2")
                if j == 0:
                    ax.set_ylabel("h-hat (mean +- sd)")
        axes[0][0].legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        path = os.path.join(outdir, "h_recovery_%s.png" % src)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


def report(df, outdir):
    from sims import report as rpt

    main = df[df["kind"] == "main"].copy()
    probe = df[df["kind"] == "probe"].copy()
    br = bias_rmse_table(main)
    bd = boundary_table(main)
    at = attribution_table(main)
    at_pool = attribution_table(main, by=("src", "mk", "h2"))
    sb = sigma_beta_table(main)
    cap = ml_cap_table(main)
    for name, t in (("bias_rmse", br), ("boundary", bd), ("attribution", at),
                    ("attribution_pooled_n", at_pool), ("sigma_beta", sb), ("ml_cap", cap)):
        t.to_csv("%s/%s.csv" % (outdir, name), index=False)
    pr = probe_table(probe) if len(probe) else None
    if pr is not None:
        pr.to_csv("%s/probe.csv" % outdir, index=False)
    _plot_recovery(main, outdir)

    extra = [("Attribution (pooled over n and geno_seed; bias of pylmm h-hat = est + opt + clamp + inter)",
              at_pool.round(4)),
             ("Bias / RMSE / SD of h-hat", br),
             ("Boundary hits, non-global optima, multimodality", bd),
             ("ML vs REML where the ML likelihood can be unbounded (pooled over h2)", cap),
             ("sigma^2 and beta", sb)]
    if pr is not None:
        extra.append(("Multiple-optima probe (n = 100, misspecified phenotypes)", pr))
    summary = br
    rpt.write_summary(
        summary, outdir, "S2: heritability recovery",
        notes=("Tables are also written as separate CSVs (bias_rmse, boundary, attribution, "
               "attribution_pooled_n, sigma_beta, ml_cap, probe).  Figures: h_recovery_<structure>.png "
               "(mean +- sd of h-hat vs true h2; solid = pylmm, dashed = textbook oracle; "
               "blue REML, red ML).  Interpretation in sims/studies/s2_h2_recovery.md."),
        extra_tables=extra)
