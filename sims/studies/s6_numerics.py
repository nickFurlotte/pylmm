"""S6: the Phase 6.3 h search (default) vs pylmm 1.0.0's (``legacy_fit=True``).

Question: does the new variance-component fit -- ``h`` searched on the closed
interval [0, 1], every grid peak (the ends included) refined, the best kept,
the ML ``sigma2 = Q / n`` -- recover ``h`` better near the boundaries, agree
with the independent oracle's maximiser and log-likelihood, and leave the null
calibration of the scan unchanged?  And what does it cost?

Per replicate (one simulated phenotype, no causal SNP):

1. genotypes and K: one cohort per cell (cached per worker); K from ``m_kin``
   SNPs, ``m_test`` disjoint test SNPs (mean-imputed, polymorphic);
2. ``y = u + e`` with ``u ~ N(0, h2 K)`` drawn exactly from K (h2 = 1: no e);
3. REML and ML null fits, new and legacy, on the same eigendecomposition; the
   oracle (`sims.oracle.fit`, dense Cholesky, its own fine global search,
   with pylmm's 1e-6 eigenvalue floor so that both maximise the same
   function);
4. metrics: the fitted h of each, |h - h_oracle|, the log-likelihood
   shortfall ``ll_oracle - ll(h)`` (>= 0; > 1e-6 means the fit missed the
   maximum), how many local maxima the profile has and whether the legacy
   search kept a worse one (first != best), fit wall time, the ML-warning
   flag (bug 17), and for the REML scan of the test SNPs: lambda_GC, type-I
   error at 0.05 and 1e-3, and max |d -log10 p| new vs legacy.

Sources: ``unrelated`` (n = 300, K from 1000 SNPs: a centred GRM of rank
n - 1, so ML has no maximum, bug 17), ``family`` (100 families of 3 sibs),
``bed`` (random mice of the bundled inbred panel, K from 2500 SNPs, rank
about 100: most eigenvalues are floored) at n = 300 and n = 60.

Run:   just sim s6_numerics --quick           (seconds)
       just sim s6_numerics --jobs 6 --report (about 10 min)
"""

from __future__ import annotations

import contextlib
import io
import time
import warnings

import numpy as np

from sims import genotypes, kinship, metrics, oracle, phenotypes

NAME = "s6_numerics"
STUDY_VERSION = 1

H2S = [0.0, 0.05, 0.5, 0.95, 0.98, 1.0]
SOURCES = {
    "unrelated300": {"source": "unrelated", "n": 300},
    "family300": {"source": "family", "n_families": 100, "n_sibs": 3},
    "bed300": {"source": "bed", "n_indivs": 300, "select": "random"},
    "bed60": {"source": "bed", "n_indivs": 60, "select": "random"},
}
M_KIN = {"unrelated": 1000, "family": 1000, "bed": 2500}


def grid(quick: bool) -> list[dict]:
    if quick:
        return [
            {"cohort": c, "h2": h2, "m_test": 100, "geno_seed": 1}
            for c in ("unrelated300", "bed60")
            for h2 in (0.5, 0.98)
        ]
    return [{"cohort": c, "h2": h2, "m_test": 1000, "geno_seed": 1} for c in SOURCES for h2 in H2S]


def reps(quick: bool) -> int:
    return 1 if quick else 40


ALPHAS = (0.05, 1e-3)


def _cohort(params):
    """(kinship, complete test SNPs) of the cell's cohort."""
    base = dict(SOURCES[params["cohort"]])
    src = base["source"]
    m_kin = M_KIN[src]
    m = m_kin + params["m_test"] * 2  # spare columns: monomorphic ones are dropped
    if src == "bed":
        spec = {**base, "n_snps": m, "seed": params["geno_seed"]}
    else:
        spec = {**base, "m": m, "seed": params["geno_seed"]}
    g = genotypes.get(spec)
    kin = kinship.get(spec, snps=slice(0, m_kin))
    T = g.G[:, m_kin:]
    T = np.where(np.isnan(T), np.nanmean(T, axis=0), T)
    T = T[:, T.std(axis=0) > 0][:, : params["m_test"]]
    return kin, T


def _n_peaks(ll: np.ndarray) -> int:
    """Local maxima of a grid profile, the ends included."""
    up = np.r_[True, ll[1:] > ll[:-1]]
    down = np.r_[ll[:-1] >= ll[1:], True]
    return int((up & down).sum())


def _fit(y, kin, reml, legacy_fit):
    from pylmm.lmm import LMM

    kva, kve = kin.eig_for_pylmm()
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        L = LMM(y, np.asarray(kin.K), kva, kve, legacy_fit=legacy_fit)
        t0 = time.perf_counter()
        h, _, sigma, ll = L.fit(reml=reml)
        dt = time.perf_counter() - t0
    warned = any("no maximum" in str(x.message) for x in w)
    return float(h), float(sigma), float(ll), dt, L, warned


def run_cell(params, seed):
    rng = np.random.default_rng(seed)
    kin, T = _cohort(params)
    K = np.asarray(kin.K)
    y = phenotypes.simulate(kin, params["h2"], rng).y
    out: dict = {"n": len(y)}
    for reml, tag in ((True, "reml"), (False, "ml")):
        f = oracle.fit(y, K, reml=reml, eig_clamp=1e-6)
        out["h_oracle_" + tag] = f.h
        for legacy, m in ((False, "new"), (True, "old")):
            h, sigma, ll, dt, L, warned = _fit(y, kin, reml, legacy)
            k = "%s_%s" % (m, tag)
            out["h_" + k] = h
            out["dh_" + k] = abs(h - f.h)
            # both on pylmm's LL scale; the oracle's maximum minus pylmm's
            out["llshort_" + k] = f.ll_pylmm - ll
            out["sigma_rel_" + k] = sigma / (f.sigma2_ml if not reml else f.sigma2_reml) - 1.0
            out["time_ms_" + k] = 1e3 * dt
            out["warned_" + k] = float(warned)
            if not legacy:
                out["npeaks_" + tag] = _n_peaks(L.ll_grid)
        out["first_ne_best_" + tag] = float(
            out["llshort_old_" + tag] > 1e-6 + out["llshort_new_" + tag]
        )
        out["llgain_" + tag] = out["llshort_old_" + tag] - out["llshort_new_" + tag]

    # the REML scan of null SNPs, both fits, batched path
    from pylmm.lmm import gwas, neg_log10_pvalue

    kva, kve = kin.eig_for_pylmm()
    with contextlib.redirect_stderr(io.StringIO()):
        t_new, p_new = (np.asarray(v) for v in gwas(y, T, K, kva, kve))
        t_old, p_old = (np.asarray(v) for v in gwas(y, T, K, kva, kve, legacy_fit=True))
    df = len(y) - 2
    out["dlog10p_max"] = float(
        np.nanmax(np.abs(neg_log10_pvalue(t_new, df) - neg_log10_pvalue(t_old, df)))
    )
    for m, p in (("new", p_new), ("old", p_old)):
        out["lambda_" + m] = metrics.lambda_gc(p)
        for a, v in metrics.type1_error(p, ALPHAS).items():
            out["%s_%s" % (a, m)] = v
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
    """One dict per (cohort, h2) cell with the numbers of the PR tables."""
    out = []
    for (cohort, h2), d in df.groupby(["cohort", "h2"], sort=False):
        r = {"cohort": cohort, "h2": h2, "reps": len(d), "n": int(d["n"].iloc[0])}
        for tag in ("reml", "ml"):
            for m in ("new", "old"):
                k = "%s_%s" % (m, tag)
                r["h_" + k] = d["h_" + k].mean()
                r["dhmax_" + k] = d["dh_" + k].max()
                r["llshort_mean_" + k] = d["llshort_" + k].clip(lower=0).mean()
                r["llshort_max_" + k] = d["llshort_" + k].max()
                r["missed_" + k] = (d["llshort_" + k] > 1e-6).mean()
                r["at1_" + k] = (d["h_" + k] >= 0.99 - 1e-12).mean()
                r["time_" + k] = d["time_ms_" + k].median()
                r["sigma_rel_max_" + k] = d["sigma_rel_" + k].abs().max()
                r["warned_" + k] = d["warned_" + k].mean()
            r["h_oracle_" + tag] = d["h_oracle_" + tag].mean()
            r["multipeak_" + tag] = (d["npeaks_" + tag] > 1).mean()
            r["first_ne_best_" + tag] = d["first_ne_best_" + tag].mean()
            r["llgain_max_" + tag] = d["llgain_" + tag].max()
        for m in ("new", "old"):
            r["lambda_" + m] = d["lambda_" + m].mean()
            for a in ALPHAS:
                key = "t1e_%s_%s" % (metrics.alpha_label(a), m)
                r[key] = d[key].mean()
        r["dlog10p_max"] = d["dlog10p_max"].max()
        out.append(r)
    return out


def report(df, outdir):
    import os

    import pandas as pd

    rows = summarize(df)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "summary.csv"), index=False)
    parts = ["# S6: h search, default (new) vs pylmm 1.0.0 (old, legacy_fit=True)\n"]
    for tag in ("reml", "ml"):
        parts.append(_fit_table(rows, tag))
    parts.append(_calibration_table(rows))
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("".join(parts))


def _fit_table(rows, tag):
    """The markdown table of the REML or ML null fits."""
    parts = []
    parts.append("\n## %s null fit\n" % tag.upper())
    parts.append(
        "h: mean fitted h (oracle: the dense oracle's global maximiser with pylmm's "
        "eigenvalue floor).  |dh|max: largest |h - h_oracle|.  missed: share of "
        "fits whose LL is more than 1e-6 below the oracle's maximum; short max: the "
        "largest shortfall.  at >=0.99: share of fits at 0.99 or above.  "
        "multi-peak: share of profiles with more than one grid maximum; "
        "new > old: share where the default search found a likelihood more than 1e-6 "
        "higher than 1.0.0's (a maximum 1.0.0 missed: not the first of several, at "
        "0.98, above 0.99 or inside an unrefined end interval).  time: median fit (ms)."
        "\n\n"
    )
    cols = [
        ("cohort", lambda r: r["cohort"]),
        ("h2", lambda r: "%g" % r["h2"]),
        ("h oracle", lambda r: _g(r["h_oracle_" + tag], "%.4f")),
        ("h new", lambda r: _g(r["h_new_" + tag], "%.4f")),
        ("h old", lambda r: _g(r["h_old_" + tag], "%.4f")),
        ("|dh|max new", lambda r: _g(r["dhmax_new_" + tag], "%.1e")),
        ("|dh|max old", lambda r: _g(r["dhmax_old_" + tag], "%.1e")),
        ("missed new", lambda r: _g(r["missed_new_" + tag], "%.2f")),
        ("missed old", lambda r: _g(r["missed_old_" + tag], "%.2f")),
        ("short max new", lambda r: _g(r["llshort_max_new_" + tag], "%.1e")),
        ("short max old", lambda r: _g(r["llshort_max_old_" + tag], "%.1e")),
        ("at >=0.99 new", lambda r: _g(r["at1_new_" + tag], "%.2f")),
        ("at >=0.99 old", lambda r: _g(r["at1_old_" + tag], "%.2f")),
        ("multi-peak", lambda r: _g(r["multipeak_" + tag], "%.2f")),
        ("new > old", lambda r: _g(r["first_ne_best_" + tag], "%.2f")),
        ("time new", lambda r: _g(r["time_new_" + tag], "%.1f")),
        ("time old", lambda r: _g(r["time_old_" + tag], "%.1f")),
    ]
    if tag == "ml":
        cols.append(("ML warned", lambda r: _g(r["warned_new_ml"], "%.2f")))
        cols.append(("sigma rel err new", lambda r: _g(r["sigma_rel_max_new_ml"], "%.0e")))
        cols.append(("sigma rel err old", lambda r: _g(r["sigma_rel_max_old_ml"], "%.0e")))
    parts.append(_table(rows, cols) + "\n")
    return "".join(parts)


def _calibration_table(rows):
    """The markdown table of the null scan's calibration."""
    t1e = ["t1e_" + metrics.alpha_label(a) for a in ALPHAS]
    parts = ["\n## Null calibration of the REML scan (test SNPs, batched)\n\n"]
    cols = [
        ("cohort", lambda r: r["cohort"]),
        ("h2", lambda r: "%g" % r["h2"]),
        ("lambda new", lambda r: _g(r["lambda_new"], "%.3f")),
        ("lambda old", lambda r: _g(r["lambda_old"], "%.3f")),
    ]
    for a in t1e:
        cols.append(("%s new" % a, lambda r, a=a: _g(r["%s_new" % a], "%.4f")))
        cols.append(("%s old" % a, lambda r, a=a: _g(r["%s_old" % a], "%.4f")))
    cols.append(("max abs d-log10p", lambda r: _g(r["dlog10p_max"], "%.2g")))
    parts.append(_table(rows, cols) + "\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# real data: the per-SNP subset fits of the HMDP scan (multi-modal profiles)
# ---------------------------------------------------------------------------

HMDP_Y = "data/hmdp.liver.exprs.1"
HMDP_K = "data/hmdp.liver.K.npdump"


def hmdp_subset_fits(n_snps: int = 5000) -> tuple[int, list[dict]]:
    """REML fits, default vs legacy, of the models ``gwas`` builds for the HMDP
    SNPs with missing calls (the strains with a call), among the first
    ``n_snps`` SNPs.  Returns the number of distinct subsets and one dict per
    subset whose default fit has a likelihood more than 1e-6 higher.

    The 94-strain HMDP liver phenotype and kinship of ``scripts/example.py``;
    each SNP with missing calls is tested on its called strains with that
    subset's own null fit, so these are many small real fits.
    """
    import os

    from pylmm.lmm import LMM

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    y = np.genfromtxt(os.path.join(root, HMDP_Y))
    K = np.load(os.path.join(root, HMDP_K), allow_pickle=True, encoding="latin1")
    G = genotypes.hmdp(n_snps=n_snps).G
    seen = set()
    rows = []
    for j in range(G.shape[1]):
        ok = ~np.isnan(G[:, j])
        if ok.all() or ok.sum() < 10 or ok.tobytes() in seen:
            continue
        seen.add(ok.tobytes())
        K2 = K[np.ix_(ok, ok)]
        with contextlib.redirect_stderr(io.StringIO()):
            new = LMM(y[ok], K2)
            old = LMM(y[ok], K2, legacy_fit=True)
            h_new, h_old = float(new.fit()[0]), float(old.fit()[0])
        if new.opt_ll - old.opt_ll > 1e-6:
            f = oracle.fit(y[ok], K2, reml=True, eig_clamp=1e-6)
            rows.append(
                {
                    "snp": j,
                    "n": int(ok.sum()),
                    "peaks": _n_peaks(new.ll_grid),
                    "h_old": h_old,
                    "h_new": h_new,
                    "h_oracle": f.h,
                    "ll_gain": new.opt_ll - old.opt_ll,
                }
            )
    return len(seen), rows


def main(argv=None) -> None:
    """``python -m sims.studies.s6_numerics [n_snps]``: the HMDP subset table."""
    import sys

    argv = sys.argv[1:] if argv is None else argv
    n, res = hmdp_subset_fits(int(argv[0]) if argv else 5000)
    print("HMDP per-SNP subset REML fits: %d distinct subsets, %d differ\n" % (n, len(res)))
    keys = ("snp", "n", "peaks", "h_old", "h_new", "h_oracle", "ll_gain")
    print(_table(res, [(k, lambda r, k=k: "%.4g" % r[k]) for k in keys]))


if __name__ == "__main__":
    main()
