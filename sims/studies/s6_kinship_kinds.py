"""S6: the standardized, centred and IBS kinships on inbred-strain panels (Phase 6.6).

Question: on an inbred panel (strains related through a few founders, 0/1
genotypes, missing calls), how do the three kinships of ``pylmm.kinship``
compare for heritability recovery and null calibration?  And does the
kinship scale (`pylmm.kinship.kinship_scale`) put their h on a common PVE
scale?

Per cell, one genotype panel (cached per worker):

* ``inbred_panel``: synthetic RI-like strains (`sims.genotypes.inbred_panel`),
  ``n`` strains from 8 founders, 10 chromosomes, 2% missing calls;
* ``hmdp``: the bundled 94 HMDP strains (every 6th of the 132,285 SNPs,
  2.1% missing), chromosomes from the validated marker map.

The kinships are built from all SNPs, missing calls included (each kind's
own convention).  The phenotype is polygenic from ``n_causal`` SNPs on the
*causal* chromosomes (the first half), with effects either per SD of the
genotype (``effect = "standardized"``, the GRM's model) or per allele
(``"per_allele"``, the unscaled model that the centred and IBS kinships
assume), scaled so that the genetic share of the sample variance is ``h2``,
plus N(0, 1 - h2) noise.  No kinship is the generating covariance, so this
compares approximations, as with real data.

Per replicate and kinship (suffix ``_std``, ``_cen``, ``_ibs``):

* ``h``: pylmm's REML h (on that kinship's scale);
* ``pve``: ``h c / (h c + 1 - h)`` with ``c`` the centred mean diagonal of K,
  the PVE implied by the fit (comparable across kinships);
* ``lam``, ``t1e_*``: lambda_GC and type-I error over the null test SNPs, which
  lie on the other chromosomes (unlinked to every causal SNP; mean-imputed).

``_std_loco`` and ``_ibs_loco`` are the standardized and IBS kinships built
without the tested chromosomes (as a leave-one-chromosome-out scan would):
with K from all SNPs, the tested SNPs' own chromosomes in K make the null-fit
Wald test conservative once h2 > 0 (lambda_GC < 1), whatever the kind.

``pve_true`` is the realised genetic share of the sample variance.

Run:   just sim s6_kinship_kinds --jobs 6     (full)
       just sim s6_kinship_kinds --quick      (well under a minute)
       just sim-report s6_kinship_kinds
"""

from __future__ import annotations

import os

import numpy as np

from sims import genotypes, kinship, methods, metrics

NAME = "s6_kinship_kinds"
STUDY_VERSION = 1

KINDS = {"std": "standardized", "cen": "centered", "ibs": "ibs"}
# The same kinships without the tested chromosomes (LOCO-style: the tested SNPs are not in K)
LOCO = {"std_loco": "standardized", "ibs_loco": "ibs"}
ALPHAS = (0.05, 0.01, 1e-3)
MIN_MAF = 0.05

GRID = {
    "panel": ["inbred_panel", "hmdp"],
    "h2": [0.0, 0.3, 0.6],
    "effect": ["standardized", "per_allele"],
    "n_causal": [200],
}
REPS = 100

QUICK_GRID = {
    "panel": ["inbred_panel_quick"],
    "h2": [0.0, 0.5],
    "effect": ["standardized"],
    "n_causal": [100],
}
QUICK_REPS = 2

_PANELS = {
    # n strains, m SNPs; the quick run uses a smaller synthetic panel and fewer test SNPs
    "inbred_panel": {"source": "inbred_panel", "n": 200, "m": 12000, "missing": 0.02, "seed": 1},
    "inbred_panel_quick": {
        "source": "inbred_panel",
        "n": 120,
        "m": 4000,
        "missing": 0.02,
        "seed": 1,
    },
    "hmdp": {"source": "hmdp", "n_snps": 22048, "select": "even"},
}

_CACHE: dict = {}


def _hmdp_chrom(snp_index):
    path = os.path.join(genotypes.REPO, "data", "hmdp.liver.snps.map.tsv")
    with open(path) as f:
        next(f)
        chrom = [line.split("\t")[1] for line in f]
    c = np.array(chrom)[np.asarray(snp_index)]
    return np.array([20 if x == "X" else int(x) for x in c])


def _setup(panel):
    if panel in _CACHE:
        return _CACHE[panel]
    from pylmm.kinship import calculate_kinship, kinship_scale

    spec = _PANELS[panel]
    geno = genotypes.get(spec)
    G = np.asarray(geno.G)
    if spec["source"] == "hmdp":
        chrom = _hmdp_chrom(geno.meta["snp_index"])
        chrom[chrom == 20] = 0  # X: neither causal nor tested
        n_auto = 19
    else:
        chrom = np.asarray(geno.meta["chrom"])
        n_auto = int(chrom.max())
    kins, scale = {}, {}
    not_tested = (chrom > 0) & (chrom <= n_auto // 2)
    for short, kind in [*KINDS.items(), *LOCO.items()]:
        K = calculate_kinship(G[:, not_tested] if short in LOCO else G, kind=kind)
        kins[short] = kinship.from_matrix(K)
        scale[short] = kinship_scale(K)
    # mean-imputed, MAF-filtered genotypes; causal on the first half of the
    # chromosomes, tested on the second half
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(G, axis=0)
    ok = (np.minimum(mean, 1 - mean) >= MIN_MAF) & (chrom > 0)
    g_imp = np.where(np.isnan(G), mean, G)
    causal_pool = np.where(ok & (chrom <= n_auto // 2))[0]
    test = np.where(ok & (chrom > n_auto // 2))[0]
    if panel.endswith("_quick"):
        test = test[:: max(1, len(test) // 600)]
    out = {
        "g_imp": g_imp,
        "kins": kins,
        "scale": scale,
        "causal_pool": causal_pool,
        "test": test,
        "n": G.shape[0],
        "m": G.shape[1],
        "miss": float(np.isnan(G).mean()),
    }
    _CACHE[panel] = out
    return out


def run_cell(params, seed):
    rng = np.random.default_rng(seed)
    s = _setup(params["panel"])
    g_imp = s["g_imp"]
    n = s["n"]
    cidx = rng.choice(s["causal_pool"], size=params["n_causal"], replace=False)
    g_causal = g_imp[:, cidx]
    if params["effect"] == "standardized":
        g_causal = (g_causal - g_causal.mean(axis=0)) / g_causal.std(axis=0)
    g = g_causal @ rng.standard_normal(g_causal.shape[1])
    g = g - g.mean()
    h2 = params["h2"]
    e = rng.standard_normal(n)
    e = (e - e.mean()) / e.std()
    if h2 > 0:
        g = g / g.std() * np.sqrt(h2)
        y = g + e * np.sqrt(1 - h2)
    else:
        y = e
    pve_true = float(np.var(g) / np.var(y)) if h2 > 0 else 0.0
    out = {"pve_true": pve_true, "n_strains": n, "m_snps": s["m"], "miss_frac": s["miss"]}
    g_test = g_imp[:, s["test"]]
    out["n_test"] = len(s["test"])
    for short, kin in s["kins"].items():
        c = s["scale"][short]
        null = methods.pylmm_null(y, kin, reml=True)
        h = null["h"]
        cm = c.centered_mean_diag
        _, p = methods.pylmm_gwas(y, g_test, kin, reml=True)
        out["h_" + short] = h
        out["pve_" + short] = float(h * cm / (h * cm + 1 - h))
        out["meandiag_" + short] = c.mean_diag
        out["cmeandiag_" + short] = cm
        out["lam_" + short] = metrics.lambda_gc(p)
        for k, v in metrics.type1_error(p, ALPHAS).items():
            out[k + "_" + short] = v
    return out


def report(df, outdir):
    from sims import report as rpt

    by = ["panel", "h2", "effect"]
    cols = ["pve_true"]
    for short in [*KINDS, *LOCO]:
        cols += ["h_" + short, "pve_" + short, "lam_" + short]
        cols += [c for c in df.columns if c.startswith("t1e_") and c.endswith("_" + short)]
        cols += ["cmeandiag_" + short]
    summary = rpt.summarize(df, by, [c for c in cols if c in df.columns])
    rpt.write_summary(
        summary,
        outdir,
        "S6: standardized vs centred vs IBS kinship on inbred panels",
        notes=(
            "h_<kind>: REML h on that kinship's scale; pve_<kind> = h c / (h c + 1 - h) with c "
            "the centred mean diagonal (cmeandiag_<kind>); pve_true: the realised genetic share. "
            "lam / t1e: over null SNPs on chromosomes without causal SNPs."
        ),
        md_columns=[
            c
            for c in summary.columns
            if not c.endswith("_se") and not c.startswith("cmeandiag") and "1e-03" not in c
        ],
    )
