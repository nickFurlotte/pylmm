"""Phase 4 synthesis: the numbers in docs/phase4-simulations.md that are not
already in a committed study summary.

Run:  uv run --group sims python sims/synthesis.py            (all parts, ~3 min)
      uv run --group sims python sims/synthesis.py crash s1   (selected parts)

Parts (outputs in sims/results/synthesis/):

crash   crash_accounting.csv, crash_by_h2.csv
        Expected tasks (study grid x reps) vs rows in raw/full.jsonl and
        raw/full.errors.jsonl for S1, S2, S3: which cells lost replicates to
        pylmm crashes (bug 4: scipy BracketError), with Wilson CIs.
crashbias  crash_bias_s1.csv, crash_bias_s2.csv, crash_bias_s3.csv
        Re-simulates each crashed replicate from its stored seed (run_cell is
        a pure function of (params, seed)) and fits what a pylmm with a
        bounded h search would see: the textbook oracle REML h, capped at
        0.99 (bug 3 not fixed), with pylmm's own association step.  Compares
        each affected cell's summary with and without those replicates.
s1      s1_calibration.csv: pooled type-I error with replicate-cluster CIs,
        mean chi^2 (from the stored random p-value sample) and the spread of
        single-replicate lambda_GC, per (structure, h2, method).
s2      s2_attribution_compact.csv: pylmm h-hat bias split into estimator /
        optimizer / clamp by h2 band and kinship type; boundary rates.
s3      s3_calibrated_power.csv, power_calibrated.png: power at an empirically
        calibrated 1e-3 threshold, with the "null" SNPs restricted to those
        not in LD (r^2 < R2_MAX with every causal SNP of the replicate), so the
        threshold is not set by SNPs that tag the causal ones (S4 item 18).
ledger  bug_ledger.csv: the consolidated ledger (S4 + new S5 issues).

Nothing here modifies pylmm; the crashed-replicate refits call pylmm's
LMM.association with an externally supplied h.
"""

import contextlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from sims import metrics, oracle, phenotypes, runner  # noqa: E402

OUT = os.path.join(runner.results_root(), "synthesis")
R2_MAX = 0.1          # LD pruning of S3 "null" SNPs against the causal SNPs
H_CAP = 0.99          # pylmm's grid cap (bug 3) kept in the "bounded search" proxy


def _log(msg):
    sys.stderr.write("[synthesis %s] %s\n" % (time.strftime("%H:%M:%S"), msg))


def _write(df, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False, float_format="%.6g")
    _log("wrote %s (%d rows)" % (os.path.relpath(path), len(df)))
    return path


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stderr(io.StringIO()):
        yield


def _clustered_rate(k, n):
    """Pooled rate sum(k)/sum(n) with a replicate-cluster (ratio-estimator) 95% CI."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    R = k.sum() / n.sum()
    m = len(k)
    if m < 2:
        return R, np.nan, np.nan
    resid = k - R * n
    se = np.sqrt(m / (m - 1) * np.sum(resid ** 2)) / n.sum()
    return R, max(0.0, R - 1.96 * se), R + 1.96 * se


def _mean_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)


# ---------------------------------------------------------------------------
# crash accounting
# ---------------------------------------------------------------------------

STUDIES = {"s1_null_calibration": ["pop", "h2", "n"],
           "s2_h2_recovery": ["kind", "src", "n", "mk", "h2", "geno_seed"],
           "s3_power": ["source", "n", "h2", "n_causal", "vpe", "contam"]}


def _errors(name):
    return runner.read_rows(os.path.join(runner.study_dir(name), "raw", "full.errors.jsonl"))


def _stage(err_text):
    """Where in the study the crash happened (null fit vs per-SNP refit)."""
    lines = [ln.strip() for ln in err_text.splitlines()]
    if any("_lmm_scan" in ln for ln in lines):
        return "per-SNP refit (L.fit(X=x))"
    return "null fit (LMM.fit)"


def _s2_which_fit(err_text):
    # pylmm_block("r_", ...) is REML, ("m_", ...) ML; the traceback shows the call line
    for ln in err_text.splitlines():
        if 'out.update(pylmm_block("r_"' in ln:
            return "REML"
        if 'out.update(pylmm_block("m_"' in ln:
            return "ML"
        if "out.update(pylmm_block(pre" in ln:
            return "probe"
    return "?"


def crash_accounting():
    import pandas as pd
    rows, byh = [], []
    for name, keys in STUDIES.items():
        tasks = runner.plan(name, quick=False)
        done = {(r["cell_id"], r["rep"]) for r in runner.read_rows(runner.raw_path(name, False))}
        errs = {(e["cell_id"], e["rep"]): e for e in _errors(name)}
        per = {}
        for cid, params, rep, seed in tasks:
            d = per.setdefault(cid, {"params": params, "expected": 0, "done": 0, "crashed": 0,
                                     "other_missing": 0, "stages": set(), "excs": set()})
            d["expected"] += 1
            if (cid, rep) in done:
                d["done"] += 1
            elif (cid, rep) in errs:
                txt = errs[(cid, rep)]["error"]
                last = txt.strip().splitlines()[-1]
                if "BracketError" in last:
                    d["crashed"] += 1
                    st = _stage(txt)
                    if name == "s2_h2_recovery":
                        st += " " + _s2_which_fit(txt)
                    d["stages"].add(st)
                else:
                    d["other_missing"] += 1
                    d["excs"].add(last[:80])
            else:
                d["other_missing"] += 1
        tot_e = sum(d["expected"] for d in per.values())
        tot_c = sum(d["crashed"] for d in per.values())
        tot_o = sum(d["other_missing"] for d in per.values())
        _log("%s: %d tasks, %d crashed (BracketError), %d other missing" % (name, tot_e, tot_c, tot_o))
        for cid, d in per.items():
            p = d["params"]
            lo, hi = metrics.rate_ci(d["crashed"], d["expected"])
            row = {"study": name, "cell_id": cid}
            row.update({k: p.get(k) for k in keys})
            row.update({"expected": d["expected"], "done": d["done"], "crashed": d["crashed"],
                        "crash_rate": d["crashed"] / d["expected"], "crash_lo": lo,
                        "crash_hi": hi, "other_missing": d["other_missing"],
                        "stage": "; ".join(sorted(d["stages"])),
                        "other_exc": "; ".join(sorted(d["excs"]))})
            rows.append(row)
        # by h2 (and n for the studies where n varies)
        df = pd.DataFrame([dict(h2=d["params"]["h2"], n=d["params"].get("n"),
                                e=d["expected"], c=d["crashed"]) for d in per.values()])
        for (h2, n), g in df.groupby(["h2", "n"]):
            lo, hi = metrics.rate_ci(g.c.sum(), g.e.sum())
            byh.append({"study": name, "h2": h2, "n": n, "cells": len(g),
                        "cells_with_crash": int((g.c > 0).sum()), "expected": int(g.e.sum()),
                        "crashed": int(g.c.sum()), "crash_rate": g.c.sum() / g.e.sum(),
                        "crash_lo": lo, "crash_hi": hi})
    out = pd.DataFrame(rows)
    _write(out, "crash_accounting.csv")
    _write(pd.DataFrame(byh), "crash_by_h2.csv")
    return out


# ---------------------------------------------------------------------------
# crash bias: refit the crashed replicates with a bounded-search proxy
# ---------------------------------------------------------------------------

def _crashed_tasks(name):
    tasks = {(cid, rep): (params, seed) for cid, params, rep, seed in runner.plan(name, quick=False)}
    out = []
    for e in _errors(name):
        if "BracketError" in e["error"].strip().splitlines()[-1]:
            params, seed = tasks[(e["cell_id"], e["rep"])]
            out.append((e["cell_id"], e["rep"], params, seed, e["error"]))
    return out


def _pylmm_fit_or_none(y, kin, reml):
    from sims import methods
    try:
        return methods.pylmm_null(y, kin, REML=reml)
    except Exception as ex:   # noqa: BLE001 - we only want to know whether it crashed
        if "BracketError" in type(ex).__name__:
            return None
        raise


def _lmm_at_h(y, kin):
    from pylmm.lmm import LMM
    Kva, Kve = kin.eig_for_pylmm()
    with _quiet():
        return LMM(np.asarray(y, float), np.asarray(kin.K), Kva, Kve)


def _assoc_scan(L, G, h):
    p = np.full(G.shape[1], np.nan)
    with _quiet():
        for j in range(G.shape[1]):
            x = G[:, j:j + 1]
            if x.var() == 0:
                continue
            _, p[j] = L.association(x, h=h, REML=True)
    return p


def crash_bias_s1():
    import pandas as pd
    from sims.studies import s1_null_calibration as s1
    crashed = _crashed_tasks("s1_null_calibration")
    rows = []
    for k, (cid, rep, params, seed, _) in enumerate(crashed):
        rng = np.random.default_rng(seed)
        kin, G_dis, G_in = s1._setup(params)
        y = phenotypes.simulate(kin, h2=params["h2"], rng=rng).y
        fT = oracle.fit(y, np.asarray(kin.K), reml=True)
        h = min(fT.h, H_CAP)
        p = _assoc_scan(_lmm_at_h(y, kin), G_dis, h)
        ok = ~np.isnan(p)
        r = {"cell_id": cid, "rep": rep, "pop": params["pop"], "h2": params["h2"],
             "h_oracle": fT.h, "h_used": h, "n_reml": int(ok.sum()),
             "lam_reml": metrics.lambda_gc(p), "meanchi2_reml": float(np.mean(metrics.chi2_from_p(p[ok])))}
        for a in s1.ALPHAS:
            r["k_reml_%s" % metrics.alpha_label(a)] = int((p[ok] <= a).sum())
        rows.append(r)
        _log("s1 crashed rep %d/%d: %s h2=%g oracle h=%.4f" % (k + 1, len(crashed), params["pop"],
                                                              params["h2"], fT.h))
    cr = pd.DataFrame(rows)
    # per affected cell: pooled t1e excluding (reported) vs including the refit replicates
    done = pd.DataFrame([dict(cell_id=r["cell_id"], **{c: r["metrics"][c] for c in
                                                       ["h_hat", "n_reml", "lam_reml"] +
                                                       ["k_reml_%s" % metrics.alpha_label(a) for a in s1.ALPHAS]})
                         for r in runner.read_rows(runner.raw_path("s1_null_calibration", False))])
    out = []
    for cid, g in cr.groupby("cell_id"):
        d = done[done.cell_id == cid]
        row = {"pop": g["pop"].iloc[0], "h2": g["h2"].iloc[0], "reps_done": len(d),
               "reps_crashed": len(g), "h_oracle_crashed_mean": g.h_oracle.mean(),
               "h_oracle_crashed_min": g.h_oracle.min(), "h_pylmm_done_mean": d.h_hat.mean(),
               "lam_crashed_mean": g.lam_reml.mean(), "lam_done_mean": d.lam_reml.mean()}
        for a in s1.ALPHAS:
            lab = metrics.alpha_label(a)
            kd, nd = d["k_reml_" + lab].sum(), d["n_reml"].sum()
            kc, nc = g["k_reml_" + lab].sum(), g["n_reml"].sum()
            row["t1e_%s_excl" % lab] = kd / nd
            row["t1e_%s_incl" % lab] = (kd + kc) / (nd + nc)
            row["t1e_%s_crashed_only" % lab] = kc / nc
        out.append(row)
    _write(cr, "crash_bias_s1_reps.csv")
    _write(pd.DataFrame(out), "crash_bias_s1.csv")


def crash_bias_s2():
    import pandas as pd
    from sims import kinship
    from sims.studies import s2_h2_recovery as s2
    crashed = _crashed_tasks("s2_h2_recovery")
    rows = []
    for k, (cid, rep, params, seed, txt) in enumerate(crashed):
        if params.get("kind") == "probe" and params["src"] != "correct":
            continue
        rng = np.random.default_rng(seed)
        n = params["n"]
        if params.get("kind") == "probe":
            kin = kinship.get(s2.geno_spec("balding_nichols", n, 5 * n, params["geno_seed"]))
        else:
            kin = kinship.get(s2.geno_spec(params["src"], n, s2.m_kin(n, params["mk"]),
                                           params["geno_seed"]))
        y = phenotypes.simulate(kin, h2=params["h2"], rng=rng, mu=s2.MU).y
        r = {"cell_id": cid, "rep": rep, **{k_: params[k_] for k_ in ("kind", "src", "n", "mk", "h2", "geno_seed")},
             "crashed_fit": _s2_which_fit(txt)}
        K = np.asarray(kin.K)
        for pre, reml in (("r_", True), ("m_", False)):
            f = _pylmm_fit_or_none(y, kin, reml)
            r[pre + "pylmm_crash"] = int(f is None)
            r[pre + "h_pylmm"] = np.nan if f is None else f["h"]
            fT = oracle.fit(y, K, reml=reml)
            r[pre + "h_T"] = fT.h
        rows.append(r)
        _log("s2 crashed rep %d/%d: %s n=%d %s h2=%g REML crash=%d oracle REML h=%.4f" % (
            k + 1, len(crashed), params["src"], n, params["mk"], params["h2"],
            r["r_pylmm_crash"], r["r_h_T"]))
    cr = pd.DataFrame(rows)
    _write(cr, "crash_bias_s2_reps.csv")
    # effect on the reported pylmm REML bias per (src, n, mk, h2) (geno seeds pooled, as in bias_rmse)
    done = []
    for r in runner.read_rows(runner.raw_path("s2_h2_recovery", False)):
        p = r["params"]
        if p.get("kind") != "main":
            continue
        done.append(dict(src=p["src"], n=p["n"], mk=p["mk"], h2=p["h2"],
                         h=r["metrics"]["r_h_pylmm"], hT=r["metrics"].get("r_h_T", np.nan)))
    done = pd.DataFrame(done)
    out = []
    main = cr[cr.kind == "main"]
    for key, g in main.groupby(["src", "n", "mk", "h2"]):
        d = done[(done.src == key[0]) & (done.n == key[1]) & (done.mk == key[2]) & (done.h2 == key[3])]
        h2 = key[3]
        # a bounded-search pylmm: pylmm's own h where it did not crash, else min(oracle, 0.99)
        imp = np.where(g.r_pylmm_crash == 1, np.minimum(g.r_h_T, H_CAP), g.r_h_pylmm)
        out.append({"src": key[0], "n": key[1], "mk": key[2], "h2": h2, "reps_done": len(d),
                    "reps_crashed": len(g), "crash_in_REML_fit": int(g.r_pylmm_crash.sum()),
                    "bias_pREML_excl": d.h.mean() - h2,
                    "bias_pREML_incl": np.r_[d.h.values, imp].mean() - h2,
                    "oracle_hT_done_mean": d.hT.mean(), "oracle_hT_crashed_mean": g.r_h_T.mean(),
                    "oracle_hT_crashed_min": g.r_h_T.min()})
    _write(pd.DataFrame(out), "crash_bias_s2.csv")


def crash_bias_s3():
    import pandas as pd
    from sims.studies import s3_power as s3
    crashed = _crashed_tasks("s3_power")
    rows = []
    for k, (cid, rep, p, seed, txt) in enumerate(crashed):
        rng = np.random.default_rng(seed)
        Gk, Gt, kin = s3._design(p)
        m = Gt.shape[1]
        kc = p["n_causal"]
        cidx = phenotypes.choose_causal(m, kc, rng, G=Gt, min_maf=s3.MIN_MAF)
        ph = phenotypes.simulate(kin, p["h2"], rng, G=Gt, causal_idx=cidx, causal_var=[p["vpe"]] * kc)
        f = _pylmm_fit_or_none(ph.y, kin, True)
        if f is None:
            fT = oracle.fit(ph.y, np.asarray(kin.K), reml=True)
            h, src_h = min(fT.h, H_CAP), "oracle(capped)"
        else:
            h, src_h = f["h"], "pylmm"
        pc = _assoc_scan(_lmm_at_h(ph.y, kin), Gt[:, cidx], h)
        rows.append({"cell_id": cid, "rep": rep, "source": p["source"], "h2": p["h2"],
                     "n_causal": kc, "vpe": p["vpe"], "stage": _stage(txt), "h_used": h,
                     "h_from": src_h, "hits_bonf": int((pc <= 0.05 / m).sum())})
        _log("s3 crashed rep %d/%d (%s)" % (k + 1, len(crashed), _stage(txt)))
    cr = pd.DataFrame(rows)
    out = []
    pt = pd.read_csv(os.path.join(runner.study_dir("s3_power"), "power_table.csv"))
    for key, g in cr.groupby(["source", "h2", "n_causal", "vpe"]):
        r = pt[(pt.source == key[0]) & (pt.h2 == key[1]) & (pt.n_causal == key[2]) &
               (pt.vpe == key[3]) & (pt.contam == 0) & (pt.method == "lmm")].iloc[0]
        nc = r.n_causal_total
        k_done = r.power_bonf * nc
        out.append({"source": key[0], "h2": key[1], "n_causal": key[2], "vpe": key[3],
                    "reps_done": int(r.n_reps), "reps_crashed": len(g),
                    "stage": "; ".join(sorted(set(g.stage))),
                    "pow_bonf_lmm_excl": r.power_bonf,
                    "pow_bonf_lmm_incl": (k_done + g.hits_bonf.sum()) / (nc + len(g) * key[2])})
    _write(cr, "crash_bias_s3_reps.csv")
    _write(pd.DataFrame(out), "crash_bias_s3.csv")


# ---------------------------------------------------------------------------
# S1 calibration (pooled type-I error, mean chi^2, lambda spread)
# ---------------------------------------------------------------------------

def s1_calibration():
    import pandas as pd
    rows = runner.read_rows(runner.raw_path("s1_null_calibration", False))
    cells = {}
    for r in rows:
        p = r["params"]
        cells.setdefault((p["pop"], p["h2"]), []).append(r["metrics"])
    out = []
    for (pop, h2), ms in cells.items():
        for meth in ("ols", "reml", "reml_in"):
            r = {"pop": pop, "h2": h2, "method": meth, "reps": len(ms)}
            n = np.array([m["n_" + meth] for m in ms], float)
            for lab in ("0.05", "1e-03", "1e-04"):
                k = np.array([m["k_%s_%s" % (meth, lab)] for m in ms], float)
                R, lo, hi = _clustered_rate(k, n)
                r["t1e_" + lab], r["t1e_%s_lo" % lab], r["t1e_%s_hi" % lab] = R, lo, hi
            chi = [np.mean(metrics.chi2_from_p(np.asarray(m["qqrand_" + meth], float)))
                   for m in ms if ("qqrand_" + meth) in m]
            r["meanchi2"], r["meanchi2_lo"], r["meanchi2_hi"] = _mean_ci(chi)
            lam = np.array([m["lam_" + meth] for m in ms], float)
            r["lambda_mean"] = lam.mean()
            r["lambda_q05"], r["lambda_q95"] = np.quantile(lam, [0.05, 0.95])
            r["lambda_min"], r["lambda_max"] = lam.min(), lam.max()
            out.append(r)
    df = pd.DataFrame(out)
    order = {p: i for i, p in enumerate(["unrelated", "bn_fst0.05", "bn_fst0.2", "family", "bed"])}
    df = df.sort_values(["method", "pop", "h2"], key=lambda s: s.map(order) if s.name == "pop" else s)
    _write(df, "s1_calibration.csv")

    # compact: h2 = 0 vs pooled over h2 in {0.2, 0.5, 0.8, 0.95}; the replicate is the cluster
    pooled = []
    for pop in order:
        for meth in ("ols", "reml", "reml_in"):
            for band, keep in (("h2=0", lambda h: h == 0), ("h2>0", lambda h: h > 0)):
                ms = [m for (p_, h), mm in cells.items() if p_ == pop and keep(h) for m in mm]
                r = {"pop": pop, "method": meth, "band": band, "reps": len(ms)}
                n = np.array([m["n_" + meth] for m in ms], float)
                for lab in ("0.05", "1e-03", "1e-04"):
                    k = np.array([m["k_%s_%s" % (meth, lab)] for m in ms], float)
                    R, lo, hi = _clustered_rate(k, n)
                    r["t1e_" + lab], r["t1e_%s_lo" % lab], r["t1e_%s_hi" % lab] = R, lo, hi
                chi = [np.mean(metrics.chi2_from_p(np.asarray(m["qqrand_" + meth], float)))
                       for m in ms if ("qqrand_" + meth) in m]
                r["meanchi2"], r["meanchi2_lo"], r["meanchi2_hi"] = _mean_ci(chi)
                lam = np.array([m["lam_" + meth] for m in ms], float)
                r["lambda_q05"], r["lambda_q95"] = np.quantile(lam, [0.05, 0.95])
                pooled.append(r)
    _write(pd.DataFrame(pooled), "s1_calibration_pooled.csv")
    return df


# ---------------------------------------------------------------------------
# S2 compact attribution
# ---------------------------------------------------------------------------

def s2_attribution():
    import pandas as pd
    rows = runner.read_rows(runner.raw_path("s2_h2_recovery", False))
    recs = []
    for r in rows:
        p, m = r["params"], r["metrics"]
        if p.get("kind") != "main":
            continue
        ktype = "bed" if p["src"] == "bed" else ("synthetic m>n" if p["mk"] == "hi" else "synthetic m<n")
        h2 = p["h2"]
        band = "h2<=0.95" if h2 <= 0.95 else "h2=%g" % h2
        for pre, meth in (("r_", "REML"), ("m_", "ML")):
            rec = {"ktype": ktype, "band": band, "method": meth, "h2": h2,
                   "h_pylmm": m[pre + "h_pylmm"], "cap": m[pre + "hit99"],
                   "lost": m[pre + "lostmode"], "nonglobal": m[pre + "nonglobal"],
                   "has_oracle": m["has_oracle"]}
            if m["has_oracle"]:
                rec.update({"T": m[pre + "h_T"], "G": m[pre + "h_G"], "C": m[pre + "h_C"],
                            "B": m[pre + "h_B"]})
            recs.append(rec)
    d = pd.DataFrame(recs)
    out = []
    for (kt, band, meth), g in d.groupby(["ktype", "band", "method"]):
        o = g[g.has_oracle == 1]
        est, opt, clamp = (o["T"] - o.h2), (o["G"] - o["T"]), (o["C"] - o["T"])
        r = {"ktype": kt, "band": band, "method": meth, "reps": len(g), "oracle_reps": len(o),
             "bias_pylmm": (g.h_pylmm - g.h2).mean(),
             "rmse_pylmm": np.sqrt(((g.h_pylmm - g.h2) ** 2).mean()),
             "frac_at_099": g.cap.mean(), "frac_nonglobal": g.nonglobal.mean(),
             "lostmode_count": int(g.lost.sum()),
             "est_mean": est.mean(), "opt_mean": opt.mean(), "clamp_mean": clamp.mean(),
             "inter_mean": (o["B"] - o["G"] - o["C"] + o["T"]).mean(),
             "rms_opt": np.sqrt((opt ** 2).mean()), "rms_clamp": np.sqrt((clamp ** 2).mean()),
             "rmse_oracle": np.sqrt((est ** 2).mean()),
             "rmse_pylmm_oracle_reps": np.sqrt(((o["B"] - o.h2) ** 2).mean()),
             "max_abs_B_vs_pylmm": (o["B"] - o.h_pylmm).abs().max()}
        se = est.std(ddof=1) / np.sqrt(len(o)) if len(o) > 1 else np.nan
        r["est_lo"], r["est_hi"] = r["est_mean"] - 1.96 * se, r["est_mean"] + 1.96 * se
        out.append(r)
    df = pd.DataFrame(out)
    _write(df, "s2_attribution_compact.csv")
    # REML lost-mode over every REML fit (main + probe)
    lost = sum(r["metrics"].get("r_lostmode", 0) for r in rows)
    _log("S2 REML lost-mode fits: %d of %d; max |B - pylmm| over oracle reps: %.3g" % (
        lost, len(rows), d[d.has_oracle == 1].pipe(lambda o: (o["B"] - o.h_pylmm).abs().max())))
    return df


# ---------------------------------------------------------------------------
# S3 calibrated power with LD-pruned nulls
# ---------------------------------------------------------------------------

def s3_calibrated(check_reps=1):
    import pandas as pd
    from sims.studies import s3_power as s3
    rows = runner.read_rows(runner.raw_path("s3_power", False))
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cell_id"], []).append(r)
    out, checked = [], 0
    for cid, rs in by_cell.items():
        p = rs[0]["params"]
        Gk, Gt, kin = s3._design(p)
        m = Gt.shape[1]
        Gs = (Gt - Gt.mean(0)) / Gt.std(0)
        kc = p["n_causal"]
        pooled = {meth: {"null": [], "null_pruned": [], "causal": []} for meth in ("lmm", "lmmr", "ols")}
        n_kept = []
        for i, r in enumerate(sorted(rs, key=lambda r: r["rep"])):
            rng = np.random.default_rng(r["seed"])
            cidx = phenotypes.choose_causal(m, kc, rng, G=Gt, min_maf=s3.MIN_MAF)
            ph = phenotypes.simulate(kin, p["h2"], rng, G=Gt, causal_idx=cidx,
                                     causal_var=[p["vpe"]] * kc)
            nulls = np.setdiff1d(np.arange(m), cidx)
            samp = rng.choice(nulls, size=s3.N_NULL_SAMPLE, replace=False)
            if i < check_reps and not p["contam"]:   # check that the replay matches the stored run
                L = _lmm_at_h(ph.y, kin)
                with _quiet():
                    L.fit()
                pc = _assoc_scan(L, Gt[:, cidx], L.optH)
                stored = np.asarray(r["metrics"]["p_c_lmm"], float)
                if not np.allclose(pc, stored, rtol=1e-6, atol=1e-300):
                    raise RuntimeError("S3 replay mismatch in cell %s rep %d" % (cid, r["rep"]))
                checked += 1
            r2 = ((Gs[:, samp].T @ Gs[:, cidx]) / Gt.shape[0]) ** 2
            keep = r2.max(axis=1) < R2_MAX
            n_kept.append(int(keep.sum()))
            for meth in pooled:
                pn = np.asarray(r["metrics"]["p_n_" + meth], float)
                pooled[meth]["null"].append(pn)
                pooled[meth]["null_pruned"].append(pn[keep])
                pooled[meth]["causal"].append(np.asarray(r["metrics"]["p_c_" + meth], float))
        for meth, d in pooled.items():
            pc = np.concatenate(d["causal"])
            row = {"source": p["source"], "h2": p["h2"], "n_causal": kc, "vpe": p["vpe"],
                   "contam": p["contam"], "m_kin": p["m_kin"], "method": meth, "reps": len(rs), "n_causal_total": len(pc),
                   "null_kept_frac": float(np.sum(n_kept) / (len(rs) * s3.N_NULL_SAMPLE))}
            for tag in ("null", "null_pruned"):
                pn = np.concatenate(d[tag])
                pn = pn[~np.isnan(pn)]
                for a, lab in ((1e-3, "1e-3"), (1e-2, "1e-2")):
                    col = "powcal_%s_%s" % (lab, "pruned" if tag == "null_pruned" else "all")
                    if len(pn) >= 1 / a:
                        thr = np.quantile(pn, a)
                        k = int((pc <= thr).sum())
                        lo, hi = metrics.rate_ci(k, len(pc))
                        row[col], row[col + "_lo"], row[col + "_hi"] = k / len(pc), lo, hi
                        row["thr_%s_%s" % (lab, tag)] = thr
                    row["n_null_" + tag] = len(pn)
                if tag == "null_pruned":
                    for lab, thr in (("1e-3", 1e-3), ("1e-2", 1e-2), ("0.05", 0.05)):
                        row["t1e_pruned_" + lab] = float(np.mean(pn <= thr))
            out.append(row)
        _log("s3 cell %s %s h2=%g k=%d vpe=%g contam=%d: kept %.0f%% of nulls" % (
            cid, p["source"], p["h2"], kc, p["vpe"], p["contam"],
            100 * np.sum(n_kept) / (len(rs) * s3.N_NULL_SAMPLE)))
    _log("S3 replay verified on %d replicates" % checked)
    chk = []
    for cid, rs in by_cell.items():
        p = rs[0]["params"]
        ms = [r["metrics"] for r in rs]
        c = {"source": p["source"], "h2": p["h2"], "n_causal": p["n_causal"], "vpe": p["vpe"],
             "contam": p["contam"], "m_kin": p["m_kin"], "reps": len(ms),
             "mean_dlogp_refit_minus_fixed": float(np.mean([m["dlogp_refit"] for m in ms]))}
        if p["oracle"]:
            c["max_abs_dt_oracle"] = float(np.max([abs(m["dt_oracle"]) for m in ms]))
            c["max_abs_dh_oracle"] = float(np.max([abs(m["dh_oracle"]) for m in ms]))
        chk.append(c)
    _write(pd.DataFrame(chk), "s3_checks.csv")
    df = pd.DataFrame(out)
    _write(df, "s3_calibrated_power.csv")
    _plot_s3(df)
    return df


def _plot_s3(df):
    from sims import plots
    plt = plots._plt()
    d = df[(df.contam == 0) & (df.h2 == 0.5) & (df.m_kin == 1500)]   # main grid, not the contamination arm
    srcs = ["unrelated", "balding_nichols", "bed"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    style = {"lmm": ("pylmm (fixed h)", plots.SERIES[0]), "lmmr": ("pylmm (refit)", plots.SERIES[2]),
             "ols": ("OLS", plots.SERIES[1])}
    for ax, s in zip(axes, srcs):
        for meth, (lab, col) in style.items():
            for kc, ls in ((1, "--"), (5, "-")):
                g = d[(d.source == s) & (d.method == meth) & (d.n_causal == kc)].sort_values("vpe")
                if g.empty or "powcal_1e-3_pruned" not in g:
                    continue
                ax.plot(g.vpe * 100, g["powcal_1e-3_pruned"], ls, color=col, marker="o", ms=3,
                        label="%s, %d causal" % (lab, kc))
        ax.set_title("%s, h2 = 0.5" % s, fontsize=9)
        ax.set_xlabel("variance explained per causal SNP (%)")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5])
        ax.set_xticklabels(["0.5", "1", "2", "5"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    axes[0].set_ylabel("power at calibrated 1e-3\n(LD-pruned null threshold)")
    axes[0].legend(fontsize=6.5, loc="upper left")
    plots._save(fig, os.path.join(OUT, "power_calibrated.png"))
    _log("wrote power_calibrated.png")


# ---------------------------------------------------------------------------
# consolidated ledger
# ---------------------------------------------------------------------------

# id, title, verdict, severity, impact_metric, impact_value, path, phase, source
LEDGER = [
    ("1", "Null fit always REML; REML= only reaches the Wald step", "CONFIRMED", "medium",
     "max |dt| CLI default vs textbook ML Wald", "0.039 (S4 unrelated m<n h2=0.8); S1: CLI == REML exactly in 25/25 cells",
     "CLI default; lmm.GWAS(REML=False); CLI --refit/--removeMissingGenotypes", "5", "S4, S1, S5"),
    ("2", "getMax range(1, n-2) skips grid index 98 (h = 0.98)", "CONFIRMED", "low",
     "REML fits returning 0.99 at true h2 = 0.98 (well-identified K)", "99% (S2 m<n and BED; mean optimizer error +0.009); 5.1% skip-98 rate in S4 h2=0.95",
     "every null fit and refit", "6.3", "S4, S2"),
    ("3", "h grid stops at 0.99 (also misc.fitTwo w grid)", "CONFIRMED", "medium",
     "fraction of REML fits at the 0.99 cap; max |d -log10 p|", "38-46% at h2 0.95-0.99 (S4 synthetic); max |d -log10 p| 0.81 (S4 BED h2=0.99)",
     "every null fit at high h2; misc.fitTwo", "6.3", "S4, S2"),
    ("4", "Brent search unbounded above h = 1 (BracketError on scipy >= 1.11)", "CONFIRMED", "high",
     "fraction of replicates that crash", "S1 25/2000, S2 68/3984, S3 14/7040; up to 12.5% per cell (S4 BED n=300 h2=0.95)",
     "every null fit and per-SNP refit; CLI default", "5", "S4, S1, S2, S3, S5"),
    ("5", "tstat(log=True) adds 2.0 instead of log 2", "UNREACHABLE", "cosmetic",
     "log p error", "1.307 (no caller)", "none today", "6.3", "S4"),
    ("6", "--kfile2 uses undefined LMM_withK2 / optW", "CONFIRMED", "low",
     "crash rate when used", "100%", "CLI --kfile2", "5", "S4"),
    ("7", "--eigen cannot read what pylmmKinship -e writes", "CONFIRMED", "low",
     "crash rate when used", "100%", "CLI --eigen", "5", "S4"),
    ("8", "Two kinship normalisations; denominator counts dropped monomorphic SNPs", "CONFIRMED", "low",
     "dh (t and p unchanged under pure rescaling)", "+0.017 hmdp h2=0.5; +0.06 at 25% monomorphic; max |dt| 0.03 between the two K with missing data",
     "lmm.calculateKinship, pylmmKinship", "6.3", "S4"),
    ("9", "Under ML the reported sigma^2 is Q/(n-q)", "CONVENTION", "cosmetic",
     "t ratio", "sqrt(n/(n-p)) = 1.0034 at n=300", "association(REML=False), reported sigma", "6.3", "S4, S2"),
    ("10", "getPhenos skips fam individuals missing from the pheno file", "CONFIRMED", "medium",
     "outcome on CLI", "crash (reshape/IndexError); silent shift only for library callers", "CLI pheno/cov files; input.plink", "5", "S4, S5"),
    ("11", "All-missing SNP: normalizeGenotype returns an empty array", "CONFIRMED", "medium",
     "outcome", "whole scan aborts (LinAlgError / IndexError / ValueError), 6/6", "lmm.GWAS; CLI GWAS; pylmmKinship", "5", "S4, S5"),
    ("12", "LMM mutates caller's Kva; calculateKinship mutates W", "CONFIRMED", "low",
     "outcome", "caller arrays silently changed", "library API", "5", "S4, S5"),
    ("13", "Zero-variance filters differ between CLI branches", "CONFIRMED", "cosmetic",
     "outcome", "same SNP tested or skipped depending on one missing call", "CLI --removeMissingGenotypes", "5", "S4, S5"),
    ("14", "No pheno file: os.path.isfile(None) TypeError", "CONFIRMED", "low",
     "outcome", "traceback instead of usage error", "CLI", "5", "S4, S5"),
    ("15", "Inconsistent return types; 1x1 arrays", "CONFIRMED", "cosmetic",
     "outcome", "list vs ndarray; numbers unaffected", "library API", "5", "S4, S5"),
    ("16", "--kfile read unchecked (fromfile + resize)", "CONFIRMED", "high",
     "outcome", "wrong-size K accepted, exit 0, 6/6 (S5); headered K all zeros on numpy < 2.3", "CLI --kfile", "5", "S4, S5"),
    ("17", "ML unbounded as h -> 1 with centred GRM from m > n SNPs", "NOT A BUG (property)", "medium",
     "pylmm ML h - REML h; ML fits at 0.99", "+0.14 to +0.16, 50-75% at cap (S4 n=200 m=2000 h2>=0.5); ML at cap 33-52% vs REML 21-26% (S2 m>n)",
     "ML paths: fit(REML=False), CLI --refit/--removeMissingGenotypes", "6.3", "S4, S2"),
    ("18", "lambda_GC 1.3-4.9 on BED genotypes", "NOT A BUG", "none",
     "pooled t1e at 0.05 (pylmm / exact test)", "0.038-0.055 / 0.049-0.058", "study design (M_eff ~ 18)", "-", "S4"),
    ("19", "Non-estimable SNP (constant up to rounding, = covariate) gets finite t ~ 0 instead of nan", "CONFIRMED (new, S5)", "medium",
     "|t| returned", "<= 1e-6 (p ~ 1); 17-100% of replicates", "lmm.GWAS; CLI both branches", "5", "S5"),
    ("20", "Saturated design (n <= q + 1) returns finite t with 0 or negative df", "CONFIRMED (new, S5)", "high",
     "SNPs with finite t where not estimable", "8-20 of 20 SNPs, every replicate (n = 10, 20, 50)", "LMM / lmm.GWAS", "5", "S5"),
    ("21", "K not validated: wrong order, asymmetric, non-PSD, nan accepted", "CONFIRMED (new, S5)", "high",
     "dh with permuted K", "up to 0.78, corr(t) < 0.9, no warning; non-PSD: dh > 0.1 in 1/3", "LMM.__init__; CLI --kfile", "5", "S5"),
    ("22", "Constant phenotype returns h = 0.99, sigma ~ 0 (or BracketError)", "CONFIRMED (new, S5)", "medium",
     "outcome", "finite numbers or crash, 50-67% of replicates", "LMM; CLI", "5", "S5"),
    ("23", "Collinear / duplicate / zero covariates give a bare LinAlgError", "CONFIRMED (new, S5)", "medium",
     "outcome", "crash with no mention of covariates, 6/6", "LMM.getMLSoln; CLI --covfile", "5", "S5"),
    ("24", "Non-finite inputs not validated in the library (nan in K/X0, inf in y)", "CONFIRMED (new, S5)", "low",
     "outcome", "scipy ValueError without context; inf y -> all-nan with no warning", "library API", "5", "S5"),
    ("25", "Badly scaled covariates emit LinAlgWarning (answers correct)", "CONFIRMED (new, S5)", "low",
     "relative error vs oracle", "<= 1e-7", "LMM.getMLSoln", "6.3", "S5"),
]


def ledger():
    import pandas as pd
    df = pd.DataFrame(LEDGER, columns=["id", "title", "verdict", "severity", "impact_metric",
                                       "impact_value", "path", "phase", "evidence"])
    _write(df, "bug_ledger.csv")


PARTS = {"crash": crash_accounting, "crashbias": lambda: (crash_bias_s1(), crash_bias_s2(), crash_bias_s3()),
         "s1": s1_calibration, "s2": s2_attribution, "s3": s3_calibrated, "ledger": ledger}


def main(argv):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    parts = argv or list(PARTS)
    for p in parts:
        t0 = time.time()
        PARTS[p]()
        _log("part %s done in %.0f s" % (p, time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1:])
