"""S3: power to detect causal SNPs -- pylmm (fixed null h and per-SNP refit) vs OLS.

See s3_power.md for the design, hypotheses and interpretation.

Per replicate (one phenotype draw on a fixed per-cell genotype design):

1. design (cached per worker): a kinship SNP set (m_kin) and a disjoint test
   SNP set (m_test), both filtered to complete calls with MAF >= 0.05;
2. choose ``n_causal`` causal SNPs inside the test set, simulate
   y = u + e + sum_j beta_j g_j with each causal SNP explaining ``vpe`` of the
   variance and the rest split h2 : 1 - h2 between K and noise;
3. fit pylmm's null model once (REML) and scan all test SNPs with
   ``L.association`` (the same call ``lmm.GWAS(refit=False)`` makes, but
   keeping beta-hat); OLS on all test SNPs; "refit": ``L.fit(X=x)`` then
   ``association`` for the causal SNPs plus a random sample of N_NULL_SAMPLE
   null SNPs (a Brent/grid fit per SNP is the expensive part); optionally the
   independent oracle Wald (REML h) on the causal SNPs;
4. store counts (causal hits and null rejections at three thresholds), sums
   for beta-hat bias/RMSE, and short p-value lists (causal p-values, and a
   fixed sample of null p-values per method) for the pooled, empirically
   calibrated power computed in report().

Run:  just sim s3_power --quick ; just sim-report s3_power --quick
"""

import numpy as np

from sims import genotypes, kinship, methods, metrics, oracle, phenotypes

NAME = "s3_power"

M_KIN, M_TEST = 1500, 2000
M_KIN_CONTAM = 500            # small K makes proximal contamination visible
N_NULL_SAMPLE = 100           # null SNPs refit (and stored as p-value sample)
MIN_MAF = 0.05
ALPHA_FIXED = (5e-8, 1e-5)    # plus Bonferroni 0.05 / m_test
ALPHA_LABELS = ["5e-08", "1e-05", "bonf"]
METHODS = ("lmm", "lmmr", "ols", "orc")   # pylmm fixed h, pylmm refit, OLS, oracle
N_BY_SOURCE = {"unrelated": 500, "balding_nichols": 500, "bed": 600}
VPE = [0.005, 0.01, 0.02, 0.05]
REPS = 80


def _grid(quick):
    cells = []
    if not quick:
        for src in N_BY_SOURCE:
            for h2 in (0.2, 0.5, 0.8):
                for nc in (1, 5):
                    for v in VPE:
                        cells.append(dict(source=src, n=N_BY_SOURCE[src], h2=h2, n_causal=nc,
                                          vpe=v, contam=0, m_kin=M_KIN, m_test=M_TEST,
                                          oracle=int(src != "bed" and h2 == 0.5), geno_seed=1))
        # proximal contamination: K from 500 SNPs, with vs without the causal SNPs in K
        for src in ("balding_nichols", "bed"):
            for contam in (0, 1):
                for v in VPE:
                    cells.append(dict(source=src, n=N_BY_SOURCE[src], h2=0.5, n_causal=5,
                                      vpe=v, contam=contam, m_kin=M_KIN_CONTAM,
                                      m_test=M_TEST, oracle=0, geno_seed=1))
    else:
        base = dict(h2=0.5, vpe=0.02, contam=0, m_kin=400, m_test=400, geno_seed=1)
        cells = [
            dict(base, source="unrelated", n=300, n_causal=1, oracle=1),
            dict(base, source="balding_nichols", n=300, n_causal=5, oracle=0),
            dict(base, source="bed", n=300, n_causal=5, oracle=0),
            dict(base, source="balding_nichols", n=300, n_causal=5, oracle=0, contam=1,
                 m_kin=200),
        ]
    return cells


def grid(quick):
    return _grid(quick)


def reps(quick):
    return 3 if quick else REPS


# ---------------------------------------------------------------------------
# design cache
# ---------------------------------------------------------------------------

_DESIGN = {}


def _usable(G):
    """Columns with complete calls, polymorphic, MAF >= MIN_MAF."""
    ok = ~np.isnan(G).any(axis=0)
    top = np.nanmax(G)
    f = np.nanmean(G, axis=0) / top
    return ok & (np.minimum(f, 1 - f) >= MIN_MAF)


def _design(p):
    key = (p["source"], p["n"], p["m_kin"], p["m_test"], p["geno_seed"])
    if key in _DESIGN:
        return _DESIGN[key]
    need = p["m_kin"] + p["m_test"]
    s, n, gs = p["source"], p["n"], p["geno_seed"]
    if s == "unrelated":
        spec = {"source": "unrelated", "n": n, "m": int(need * 1.3), "seed": gs}
    elif s == "balding_nichols":
        spec = {"source": "balding_nichols", "n": n, "m": int(need * 1.3), "k": 3,
                "fst": 0.1, "seed": gs}
    else:   # real mice: random individuals and SNPs; ~half the SNPs have a missing call
        spec = {"source": "bed", "n_indivs": n, "n_snps": 12000, "select": "random",
                "seed": gs}
    G = genotypes.get(spec).G
    cols = np.flatnonzero(_usable(G))
    if len(cols) < need:
        raise RuntimeError("only %d usable SNPs, need %d" % (len(cols), need))
    Gk, Gt = G[:, cols[:p["m_kin"]]], G[:, cols[p["m_kin"]:need]]
    kin = kinship.from_matrix(kinship.kinship(Gk), n_snps=Gk.shape[1])
    if len(_DESIGN) >= 3:
        _DESIGN.pop(next(iter(_DESIGN)))
    _DESIGN[key] = (Gk, Gt, kin)
    return _DESIGN[key]


# ---------------------------------------------------------------------------
# helpers implemented locally
# ---------------------------------------------------------------------------

def ols_beta(y, G):
    """OLS slope of y on each column of G (intercept only)."""
    gc = G - G.mean(axis=0)
    return gc.T @ (y - y.mean()) / (gc ** 2).sum(axis=0)


def _lmm_scan(L, G, idx, refit):
    """pylmm per-SNP (t, p, beta) for columns idx; refit=True refits h per SNP
    exactly as lmm.GWAS(refit=True) does (L.fit(X=x) then L.association)."""
    t = np.empty(len(idx))
    p = np.empty(len(idx))
    b = np.empty(len(idx))
    for i, j in enumerate(idx):
        x = G[:, j:j + 1]
        if refit:
            L.fit(X=x)
        t[i], p[i], b[i], _ = L.association(x, REML=True, returnBeta=True)
    return t, p, b


def _record(out, name, p_c, b_c, beta, p_null, thresholds):
    """Counts and sums for one method.  p_c/b_c: causal p and beta-hat."""
    rel = b_c / beta - 1.0                       # signed relative error
    for lab, thr in zip(ALPHA_LABELS, thresholds):
        out["hit_%s_%s" % (name, lab)] = int((p_c <= thr).sum())
        if p_null is not None:
            out["t1_%s_%s" % (name, lab)] = int((p_null <= thr).sum())
    if p_null is not None:
        out["nn_" + name] = int(len(p_null))
    out["rel_" + name] = float(rel.sum())
    out["sq_" + name] = float((rel ** 2).sum())
    sig = p_c <= 1e-5
    out["nsig_" + name] = int(sig.sum())
    out["relsig_" + name] = float(rel[sig].sum())
    out["p_c_" + name] = p_c.tolist()


# ---------------------------------------------------------------------------

def run_cell(params, seed):
    rng = np.random.default_rng(seed)
    p = params
    Gk, Gt, kin = _design(p)
    m = Gt.shape[1]
    k = p["n_causal"]
    thresholds = list(ALPHA_FIXED) + [0.05 / m]

    cidx = phenotypes.choose_causal(m, k, rng, G=Gt, min_maf=MIN_MAF)
    ph = phenotypes.simulate(kin, p["h2"], rng, G=Gt, causal_idx=cidx,
                             causal_var=[p["vpe"]] * k)
    y = ph.y
    beta = ph.beta

    if p["contam"]:   # K rebuilt with the causal SNPs included (proximal contamination)
        kin = kinship.from_matrix(kinship.kinship(np.hstack([Gk, Gt[:, cidx]])))

    is_null = np.ones(m, dtype=bool)
    is_null[cidx] = False
    nulls = np.flatnonzero(is_null)
    samp = rng.choice(nulls, size=N_NULL_SAMPLE, replace=False)

    null = methods.pylmm_null(y, kin, REML=True)
    L = null["lmm"]
    all_idx = np.arange(m)
    out = {"h_hat": null["h"], "n_causal_obs": int(k)}

    # pylmm, fixed null h (identical to lmm.GWAS(refit=False))
    t, pl, bl = _lmm_scan(L, Gt, all_idx, refit=False)
    _record(out, "lmm", pl[cidx], bl[cidx], beta, pl[nulls], thresholds)
    out["lam_lmm"] = metrics.lambda_gc(pl[nulls])
    out["p_n_lmm"] = pl[samp].tolist()

    # OLS
    to, po = methods.ols_gwas(y, Gt)
    bo = ols_beta(y, Gt)
    _record(out, "ols", po[cidx], bo[cidx], beta, po[nulls], thresholds)
    out["lam_ols"] = metrics.lambda_gc(po[nulls])
    out["p_n_ols"] = po[samp].tolist()

    # pylmm refit: causal SNPs + the null sample only
    ridx = np.r_[cidx, samp]
    _, pr, br = _lmm_scan(L, Gt, ridx, refit=True)
    _record(out, "lmmr", pr[:k], br[:k], beta, pr[k:], thresholds)
    out["p_n_lmmr"] = pr[k:].tolist()
    # paired: how much do refit and fixed-h p-values/betas differ on the same SNPs
    out["dlogp_refit"] = float(np.mean(np.log10(pr) - np.log10(pl[ridx])))
    out["dbeta_refit"] = float(np.mean(br - bl[ridx]))

    if p["oracle"]:
        K = np.asarray(kin.K)
        f = oracle.fit(y, K, reml=True)
        ws = [oracle.wald(y, K, Gt[:, j], h=f.h) for j in cidx]
        _record(out, "orc", np.array([w.p for w in ws]), np.array([w.beta for w in ws]),
                beta, None, thresholds)
        out["dh_oracle"] = null["h"] - f.h
        out["dt_oracle"] = float(np.max(np.abs(t[cidx] - [w.t for w in ws])))
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

LABEL = {"lmm": "pylmm (fixed h)", "lmmr": "pylmm (refit)", "ols": "OLS", "orc": "oracle Wald"}
# every grid key that varies: m_kin separates the contamination arm's control
# (contam=0, K from 500 SNPs) from the main cell with the same other params
KEYS = ["source", "h2", "n_causal", "vpe", "contam", "m_kin"]


def power_table(df):
    """One row per (design cell, method) with CIs; uses all replicates of a cell."""
    import pandas as pd
    rows = []
    for key, d in df.groupby(KEYS, sort=True):
        base = dict(zip(KEYS, key))
        nc = d["n_causal_obs"].sum()
        for name in METHODS:
            if "p_c_" + name not in d or d["p_c_" + name].isna().all():
                continue
            dd = d[d["p_c_" + name].notna()]
            ncm = dd["n_causal_obs"].sum()
            r = dict(base, method=name, n_reps=len(dd), n_causal_total=int(ncm))
            for lab in ALPHA_LABELS:
                k_ = dd["hit_%s_%s" % (name, lab)].sum()
                lo, hi = metrics.rate_ci(k_, ncm)
                r.update({"power_" + lab: k_ / ncm, "power_%s_lo" % lab: lo,
                          "power_%s_hi" % lab: hi})
                if "t1_%s_%s" % (name, lab) in dd and dd["nn_" + name].sum() > 0:
                    k0, n0 = dd["t1_%s_%s" % (name, lab)].sum(), dd["nn_" + name].sum()
                    r.update({"t1e_" + lab: k0 / n0, "t1e_%s_lo" % lab: metrics.rate_ci(k0, n0)[0],
                              "t1e_%s_hi" % lab: metrics.rate_ci(k0, n0)[1], "n_null_" + lab: int(n0)})
            # empirically calibrated power: threshold = pooled null quantile
            pc = metrics.pool_pvalues(dd["p_c_" + name])
            pn = metrics.pool_pvalues(dd["p_n_" + name]) if "p_n_" + name in dd else np.zeros(0)
            for a in (0.01, 0.001):
                if len(pn) >= 1 / a:
                    thr = np.quantile(pn, a)
                    kk = int((pc <= thr).sum())
                    lo, hi = metrics.rate_ci(kk, len(pc))
                    r.update({"powcal_%g" % a: kk / len(pc), "powcal_%g_lo" % a: lo,
                              "powcal_%g_hi" % a: hi})
            # beta-hat accuracy (relative to the true beta)
            r["rel_bias"] = dd["rel_" + name].sum() / ncm
            r["rel_rmse"] = float(np.sqrt(dd["sq_" + name].sum() / ncm))
            ns = dd["nsig_" + name].sum()
            r["n_sig_1e-5"] = int(ns)
            r["rel_bias_sig"] = dd["relsig_" + name].sum() / ns if ns else np.nan
            if name in ("lmm", "ols"):
                r["lambda_gc"] = dd["lam_" + name].mean()
            rows.append(r)
    return pd.DataFrame(rows)


def _plot_power(tab, source, outdir):
    import os

    from sims import plots
    plt = plots._plt()
    t = tab[(tab.source == source) & (tab.contam == 0) & (tab.m_kin == M_KIN)
            & (tab.method != "orc")]
    h2s = sorted(t.h2.unique())
    if not len(t):
        return
    rows = [("power_bonf", "power at Bonferroni (0.05/m), raw"),
            ("powcal_0.001", "power at empirically calibrated alpha = 1e-3")]
    fig, axes = plt.subplots(len(rows), len(h2s), figsize=(3.6 * len(h2s), 6.2),
                             squeeze=False, sharey=True)
    colors = {"lmm": plots.SERIES[0], "lmmr": plots.SERIES[2], "ols": plots.SERIES[1]}
    for r_, (col, lab) in enumerate(rows):
        for c_, h2 in enumerate(h2s):
            ax = axes[r_][c_]
            for name in ("lmm", "lmmr", "ols"):
                for nc, ls in ((1, "--"), (5, "-")):
                    d = t[(t.h2 == h2) & (t.method == name) & (t.n_causal == nc)].sort_values("vpe")
                    if not len(d) or col not in d or d[col].isna().all():
                        continue
                    ax.errorbar(100 * d.vpe, d[col],
                                yerr=[(d[col] - d[col + "_lo"]).clip(lower=0),
                                      (d[col + "_hi"] - d[col]).clip(lower=0)],
                                color=colors[name], ls=ls, marker="o", ms=3, capsize=1.5,
                                label="%s, %d causal" % (LABEL[name], nc))
            ax.set_title("h2 = %g" % h2)
            ax.set_xscale("log")
            ax.set_xticks([0.5, 1, 2, 5])
            ax.set_xticklabels(["0.5", "1", "2", "5"])
            ax.set_ylim(-0.02, 1.02)
            if r_ == len(rows) - 1:
                ax.set_xlabel("variance explained per causal SNP (%)")
            if c_ == 0:
                ax.set_ylabel(lab)
    axes[0][0].legend(fontsize=6.5)
    fig.suptitle("S3 power, %s" % source)
    plots._save(fig, os.path.join(outdir, "power_%s.png" % source))


def _plot_type1(tab, outdir):
    """OLS vs pylmm null rejection at Bonferroni, by source and h2 (one point per cell mean)."""
    import os

    from sims import plots
    plt = plots._plt()
    t = tab[(tab.contam == 0) & (tab.m_kin == M_KIN) & tab.method.isin(["lmm", "ols"])]
    g = t.groupby(["source", "h2", "method"]).agg(k=("t1e_1e-05", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    srcs = sorted(g.source.unique())
    for i, name in enumerate(("lmm", "ols")):
        for j, h2 in enumerate(sorted(g.h2.unique())):
            d = g[(g.method == name) & (g.h2 == h2)].set_index("source").reindex(srcs)
            ax.bar(np.arange(len(srcs)) + 0.13 * (3 * i + j) - 0.4, d.k.values, width=0.12,
                   color=plots.SERIES[0] if name == "lmm" else plots.SERIES[1],
                   alpha=0.45 + 0.25 * j, label="%s h2=%g" % (LABEL[name], h2))
    ax.axhline(1e-5, color=plots.REF, ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(srcs)) - 0.05)
    ax.set_xticklabels(srcs)
    ax.set_ylabel("null rejection rate at 1e-5")
    ax.legend(fontsize=6, ncol=2)
    plots._save(fig, os.path.join(outdir, "type1_1e-05.png"))


def report(df, outdir):
    import os

    from sims import report as rpt
    tab = power_table(df)
    tab.to_csv(os.path.join(outdir, "power_table.csv"), index=False)
    for src in tab.source.unique():
        _plot_power(tab, src, outdir)
    if "t1e_1e-05" in tab:
        _plot_type1(tab, outdir)

    main = tab[(tab.contam == 0) & (tab.m_kin == M_KIN)]
    cols = ["source", "h2", "n_causal", "vpe", "method", "power_bonf", "power_1e-05",
            "power_5e-08", "powcal_0.001", "t1e_1e-05", "t1e_bonf", "lambda_gc",
            "rel_bias", "rel_rmse", "rel_bias_sig"]
    cols = [c for c in cols if c in main]
    short = main[cols].rename(columns={"power_bonf": "pow_bonf", "power_1e-05": "pow_1e-5",
                                       "power_5e-08": "pow_5e-8", "powcal_0.001": "pow_cal1e-3",
                                       "t1e_1e-05": "t1e_1e-5", "rel_bias_sig": "rel_bias|sig"})
    contam = tab[tab.contam.isin([0, 1]) & (tab.m_kin == M_KIN_CONTAM) & (tab.method == "lmm")
                 & (tab.source.isin(["balding_nichols", "bed"]))
                 & (tab.n_causal == 5) & (tab.h2 == 0.5)]
    extra = []
    if len(contam):
        piv = contam.pivot_table(index=["source", "vpe"], columns="contam",
                                 values=[c for c in ("power_bonf", "powcal_0.001", "rel_bias")
                                         if c in contam])
        piv.columns = ["%s_contam%d" % (a, b) for a, b in piv.columns]
        extra.append(("Proximal contamination (pylmm fixed h, K from 500 SNPs; contam=1 adds "
                      "the 5 causal SNPs to K)", piv.reset_index()))
    notes = ("Main grid (K from %d SNPs); the contamination arm (K from %d SNPs) is in "
             "its own table below and in power_table.csv (column m_kin).  "
             "Pooled over replicates within each cell.  pow_* = fraction of causal SNPs with "
             "p <= threshold (Wilson CIs in power_table.csv); pow_cal1e-3 = power at the "
             "threshold giving 1e-3 type-I error on the pooled non-causal SNPs of that "
             "method and cell; t1e = empirical null rejection rate; rel_bias = mean of "
             "beta_hat/beta - 1; rel_bias|sig = same restricted to p <= 1e-5 (winner's "
             "curse); lambda_gc on non-causal SNPs.  See s3_power.md." % (M_KIN, M_KIN_CONTAM))
    rpt.write_summary(short, outdir, "S3: power of pylmm vs OLS", notes=notes,
                      extra_tables=extra)
