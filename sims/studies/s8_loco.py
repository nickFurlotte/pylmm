"""S8: leave-one-chromosome-out (LOCO) vs the all-SNP kinship (Phase 6.4).

Question: does LOCO (`pylmm.loco.gwas_loco`, ``pylmm gwas --loco``) fix the
conservative null calibration of the all-SNP kinship (S6: λ_GC ~0.7 on
inbred panels once h2 > 0), and what does it do to power, on outbred and
inbred genotype sources with a polygenic background spread over many
chromosomes?

Per cell, one genotype panel (cached per worker):

* ``family``: 300 outbred individuals in 100 sib trios (`sims.genotypes.family`),
  8000 unlinked SNPs labelled with 10 chromosomes of 800 SNPs, no missing calls;
* ``inbred_panel``: 200 synthetic RI-like strains from 8 founders
  (`sims.genotypes.inbred_panel`), 10 chromosomes, 10,000 SNPs (linked within
  a chromosome), 2% missing calls;
* ``hmdp``: the bundled 94 HMDP strains, every 6th of the 132,285 SNPs (2.1%
  missing), chromosomes 1-19 and X from the validated marker map.

The kinships are the standardized kinship of all SNPs (missing calls
included) and its LOCO versions (`pylmm.loco.LocoKinship`, built once per
panel from the same SNPs).  The phenotype is::

    y = sqrt(h2) g + sqrt(q2) x_qtl + sqrt(1 - h2 - q2) e

with ``g`` the polygenic background of ``n_causal`` per-SD effects on the
*odd* autosomes, ``x_qtl`` one more SNP on an odd autosome (``q2 = 0.05``),
each standardised in the sample, and ``e`` N(0, 1).  Tested SNPs
(mean-imputed, MAF >= 0.05): up to ``n_null`` null SNPs on the *even*
autosomes (no causal SNP on their chromosome), the QTL, and up to 100 of the
background's causal SNPs.

Per replicate, for ``all`` (one kinship, `pylmm.lmm.gwas`) and ``loco``
(`pylmm.loco.gwas_loco`): the REML null h (``loco``: the mean over the
tested chromosomes), λ_GC and type-I error over the null SNPs, -log10 p of
the QTL and its power at 1e-3 and 1e-5, and the mean χ² (t²) of the tested
background causal SNPs.

Run:   just sim s8_loco --jobs 6       (full; a few minutes)
       just sim s8_loco --quick        (well under a minute)
       just sim-report s8_loco
"""

from __future__ import annotations

import os

import numpy as np

from sims import genotypes, kinship, methods, metrics

NAME = "s8_loco"
STUDY_VERSION = 1

ALPHAS = (0.05, 0.01, 1e-3)
POWER_ALPHAS = (1e-3, 1e-5)
MIN_MAF = 0.05
Q2 = 0.05

GRID = {
    "panel": ["family", "inbred_panel", "hmdp"],
    "h2": [0.0, 0.3, 0.6, 0.9],
    "n_causal": [300],
}
REPS = 100

QUICK_GRID = {
    "panel": ["inbred_panel_quick"],
    "h2": [0.0, 0.6],
    "n_causal": [100],
}
QUICK_REPS = 2

_PANELS = {
    "family": {
        "spec": {"source": "family", "n_families": 100, "n_sibs": 3, "m": 8000, "seed": 1},
        "n_chrom": 10,
        "n_null": 1500,
    },
    "inbred_panel": {
        "spec": {"source": "inbred_panel", "n": 200, "m": 10000, "missing": 0.02, "seed": 1},
        "n_null": 1500,
    },
    "inbred_panel_quick": {
        "spec": {"source": "inbred_panel", "n": 100, "m": 3000, "missing": 0.02, "seed": 1},
        "n_null": 300,
    },
    "hmdp": {"spec": {"source": "hmdp", "n_snps": 22048, "select": "even"}, "n_null": 1500},
}

_CACHE: dict = {}


def _hmdp_chrom(snp_index):
    path = os.path.join(genotypes.REPO, "data", "hmdp.liver.snps.map.tsv")
    with open(path) as f:
        next(f)
        chrom = [line.split("\t")[1] for line in f]
    return np.array(chrom, dtype=object)[np.asarray(snp_index)]


def _setup(panel):
    """The panel's genotypes, LOCO kinships, all-SNP kinship and SNP pools (cached)."""
    if panel in _CACHE:
        return _CACHE[panel]
    from pylmm.loco import LocoKinship

    cfg = _PANELS[panel]
    geno = genotypes.get(cfg["spec"])
    G = np.asarray(geno.G)
    if cfg["spec"]["source"] == "hmdp":
        chrom = _hmdp_chrom(geno.meta["snp_index"])
    elif "chrom" in geno.meta:
        chrom = np.asarray(geno.meta["chrom"]).astype(str).astype(object)
    else:  # unlinked SNPs: label them with contiguous chromosomes
        n_chrom = cfg["n_chrom"]
        m = G.shape[1]
        chrom = np.repeat(np.arange(1, n_chrom + 1), int(np.ceil(m / n_chrom)))[:m]
        chrom = chrom.astype(str).astype(object)
    loco = LocoKinship.from_genotypes(G, chrom)
    kin_all = kinship.from_matrix(loco.kinship())
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(G, axis=0)
    # MAF on the genotype's own scale: 0/1/2 counts (family) or 0/1 (inbred)
    top = 2.0 if np.nanmax(G) > 1 else 1.0
    freq = mean / top
    auto = np.array([c.isdigit() for c in chrom])
    num = np.array([int(c) if c.isdigit() else 0 for c in chrom])
    ok = (np.minimum(freq, 1 - freq) >= MIN_MAF) & auto
    g_imp = np.where(np.isnan(G), mean, G)
    odd = np.where(ok & (num % 2 == 1))[0]
    even = np.where(ok & (num % 2 == 0))[0]
    rng = np.random.default_rng(0)
    null = np.sort(rng.choice(even, size=min(cfg["n_null"], len(even)), replace=False))
    out = {
        "g_imp": g_imp,
        "chrom": chrom,
        "loco": loco,
        "kin_all": kin_all,
        "odd": odd,
        "null": null,
        "n": G.shape[0],
        "m": G.shape[1],
        "n_chrom": len(loco.chromosomes),
    }
    _CACHE[panel] = out
    return out


def _std(x):
    x = x - x.mean()
    s = x.std()
    return x / s if s > 0 else x


def run_cell(params, seed):
    from pylmm.loco import gwas_loco

    rng = np.random.default_rng(seed)
    s = _setup(params["panel"])
    g_imp, n = s["g_imp"], s["n"]
    h2 = params["h2"]
    pick = rng.choice(s["odd"], size=params["n_causal"] + 1, replace=False)
    qtl, causal = pick[0], pick[1:]
    y = np.sqrt(Q2) * _std(g_imp[:, qtl]) + np.sqrt(1 - h2 - Q2) * _std(rng.standard_normal(n))
    if h2 > 0:
        gc = g_imp[:, causal]
        gc = (gc - gc.mean(axis=0)) / gc.std(axis=0)
        y = y + np.sqrt(h2) * _std(gc @ rng.standard_normal(len(causal)))
    tested_causal = causal[:100]
    cols = np.concatenate([s["null"], [qtl], tested_causal])
    X = g_imp[:, cols]
    n_null = len(s["null"])
    out = {"n_indiv": n, "m_snps": s["m"], "n_chrom": s["n_chrom"], "n_null": n_null}

    res = {}
    null = methods.pylmm_null(y, s["kin_all"], reml=True)
    t_all, p_all = methods.pylmm_gwas(y, X, s["kin_all"], reml=True)
    res["all"] = (null["h"], np.asarray(t_all), np.asarray(p_all))
    with methods._quiet():
        r = gwas_loco(y, X, s["chrom"][cols], kinships=s["loco"])
    res["loco"] = (
        float(np.mean(list(r.h.values()))),
        np.asarray(r.t_stats),
        np.asarray(r.p_values),
    )
    for name, (h, t, p) in res.items():
        p_null = p[:n_null]
        out["h_" + name] = h
        out["lam_" + name] = metrics.lambda_gc(p_null)
        for k, v in metrics.type1_error(p_null, ALPHAS).items():
            out[k + "_" + name] = v
        p_qtl = p[n_null]
        out["qtl_nlp_" + name] = float(-np.log10(max(p_qtl, 1e-300)))
        for a in POWER_ALPHAS:
            out["power_%s_%s" % (metrics.alpha_label(a), name)] = float(p_qtl <= a)
        tc = t[n_null + 1 :]
        out["causal_chi2_" + name] = float(np.nanmean(tc**2))
    return out


def report(df, outdir):
    from sims import report as rpt

    by = ["panel", "h2"]
    cols = []
    for name in ("all", "loco"):
        cols += ["h_" + name, "lam_" + name]
        cols += ["t1e_%s_%s" % (metrics.alpha_label(a), name) for a in ALPHAS]
        cols += ["qtl_nlp_" + name]
        cols += ["power_%s_%s" % (metrics.alpha_label(a), name) for a in POWER_ALPHAS]
        cols += ["causal_chi2_" + name]
    summary = rpt.summarize(df, by, [c for c in cols if c in df.columns])
    rpt.write_summary(
        summary,
        outdir,
        "S8: leave-one-chromosome-out (LOCO) vs the all-SNP kinship",
        notes=(
            "all: one standardized kinship of all SNPs (pylmm.lmm.gwas); loco: K_{-c} per "
            "chromosome (pylmm.loco.gwas_loco). h: REML null h (loco: mean over chromosomes). "
            "lam / t1e: over null SNPs on even autosomes (no causal SNP on their chromosome). "
            "qtl_nlp: -log10 p of the QTL (q2 = 0.05, odd autosome); power_<a>: share of "
            "replicates with QTL p <= a. causal_chi2: mean t^2 of up to 100 background causal "
            "SNPs (h2 / n_causal each)."
        ),
        md_columns=[c for c in summary.columns if not c.endswith("_se")],
    )
