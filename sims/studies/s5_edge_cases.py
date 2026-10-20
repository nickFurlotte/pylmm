"""S5: edge cases and robustness -- a structured fuzz of pylmm.

For each *case* (a params dict: family, variant, path, plus knobs) we build a
small problem (n ~ 100 related individuals from nuclear families, 20 test
SNPs, K from 400 other SNPs), run pylmm on it, and record what happens:

* crash        exception type + message (caught, never fails the study)
* nan / inf    unexpected nan h/sigma/t, inf anywhere, finite t where the
               statistic is not estimable (a *silent* wrong answer)
* warnings     python warnings raised during the run
* accuracy     |h - h_oracle| and max relative |t - t_oracle| against the
               independent dense REML oracle (sims/oracle.py) where defined
* runtime

Paths:  lib         pylmm.lmm.gwas on raw genotypes (missing = per-SNP removal)
        lib_impute  user mean-imputes, then pylmm.lmm.gwas
        cli_impute  `python -m pylmm gwas` through subprocess (BED/fam/pheno/kin files
                    written here), default behaviour: mean-impute + standardise
        cli_remove  same with --removeMissingGenotypes

Each problem declares its *ideal* behaviour (``ideal``):
  result           pylmm should return correct numbers (any crash/nan is a bug)
  result_or_error  correct numbers, or a clear error (e.g. rank-deficient X0)
  warn_or_error    garbage-in: acceptable if it warns/raises, "silent" if not
  error_or_nan     statistic not estimable: nan or a clear error is right,
                   finite numbers are a silent wrong answer

Outcome labels (per replicate; the report takes the worst across replicates):
OK, silent (no warning on dubious input), nan, crash, wrong.  See
sims/studies/s5_edge_cases.md for the interpretation and the Phase 5 target list.

Run:  just sim s5_edge_cases --quick ; just sim-report s5_edge_cases --quick
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

from sims import genotypes, kinship, oracle, phenotypes

NAME = "s5_edge_cases"

REPS = 6
QUICK_REPS = 1

N_BASE = 100
M_KIN = 400
M_TEST = 20
N_ORACLE_SNPS = 6
CLI_TIMEOUT = 120

# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------


def _cell(family, variant, path="lib", **kw):
    d = {"family": family, "variant": variant, "path": path}
    d.update(kw)
    return d


def _build_grid(quick):
    g = []
    # 1. missing data: genotype rate x phenotype rate x (lib vs CLI, remove vs impute)
    if quick:
        for path in ("lib", "lib_impute", "cli_impute", "cli_remove"):
            g.append(_cell("missing", "mcar", path, gm=0.2, pm=0.05))
    else:
        for gm in (0.0, 0.01, 0.05, 0.2, 0.5):
            for pm in (0.0, 0.05, 0.2):
                for path in ("lib", "lib_impute"):
                    g.append(_cell("missing", "mcar", path, gm=gm, pm=pm))
            for pm in (0.0, 0.2):
                for path in ("cli_impute", "cli_remove"):
                    g.append(_cell("missing", "mcar", path, gm=gm, pm=pm))
    # 2. degenerate SNPs (planted in column 0 of the test set)
    deg = [
        "mono",
        "mono_dosage",
        "mono_ydrop",
        "mono_xdrop",
        "allmiss",
        "single_minor",
        "single_minor_ydrop",
        "eq_cov",
        "eq_cov_x2",
        "eq_intercept",
    ]
    for v in ["mono", "allmiss", "eq_cov"] if quick else deg:
        for path in (
            ("lib", "cli_impute") if quick else ("lib", "lib_impute", "cli_impute", "cli_remove")
        ):
            g.append(_cell("degenerate", v, path))
    # 3. small n, including n < covariates + 2
    for n in (10,) if quick else (10, 20, 50):
        for k in (0, 8, 10) if quick else (0, 2, n // 2, n - 3, n - 2, n - 1, n + 2):
            g.append(_cell("smalln", "n_k", "lib", n=n, k=k))
    # 4. kinship pathologies
    kin = [
        ("lowrank", {"m_kin": 2}),
        ("lowrank", {"m_kin": 5}),
        ("lowrank", {"m_kin": 20}),
        ("identity", {}),
        ("dup_pairs", {}),
        ("dup_all", {}),
        ("nonpsd", {"noise": 0.1}),
        ("nonpsd", {"noise": 0.3}),
        ("wrong_order", {}),
        ("scale", {"scale": 1e-4}),
        ("scale", {"scale": 1e4}),
        ("upper_only", {}),
        ("nan_K", {}),
        ("asym", {}),
    ]
    for v, kw in kin[::4] if quick else kin:
        g.append(_cell("kin", v, "lib", **kw))
    # 5. covariates
    cov = [
        ("collinear_dup", "lib"),
        ("collinear_dup", "cli_impute"),
        ("collinear_comb", "lib"),
        ("intercept_dup", "lib"),
        ("intercept_dup", "cli_impute"),
        ("zero_cov", "lib"),
        ("zero_cov", "cli_impute"),
        ("cov_nan", "lib"),
        ("cov_nan", "cli_impute"),
    ]
    cov = [_cell("cov", v, p) for v, p in cov]
    for k in (5, 20, 40, 80):
        cov.append(_cell("cov", "many", "lib", k=k))
    for s in (1e-8, 1e8):
        cov.append(_cell("cov", "cov_scale", "lib", scale=s))
    g += cov[::5] if quick else cov
    # 6. phenotype scaling / distribution
    ph = [
        ("scale", {"scale": 1e-6}),
        ("scale", {"scale": 1e6}),
        ("offset", {"offset": 1e9}),
        ("constant", {}),
        ("const_noise", {"noise": 1e-12}),
        ("binary", {}),
        ("heavy_t2", {}),
        ("lognormal", {}),
        ("inf_y", {}),
        ("all_nan", {}),
        ("few_obs", {"n_obs": 3}),
    ]
    for v, kw in ph[3:4] if quick else ph:
        g.append(_cell("pheno", v, "lib", **kw))
    if not quick:
        for v in ("constant", "binary", "all_nan", "inf_y"):
            g.append(_cell("pheno", v, "cli_impute"))
    # 7. CLI file handling
    io_cases = [
        "nophenofile",
        "pheno_shuffled",
        "pheno_missing_indiv",
        "pheno_extra_indiv",
        "pheno_neg9",
        "kfile_header",
        "kfile_big",
        "kfile_small",
    ]
    for v in ["pheno_missing_indiv"] if quick else io_cases:
        g.append(_cell("io", v, "cli_impute"))
    return g


GRID = _build_grid(False)
QUICK_GRID = _build_grid(True)


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------


class Problem:
    def __init__(self, y, g_test, K, cov, ideal="result"):
        self.y = np.asarray(y, dtype=np.float64)
        self.g_test = np.asarray(g_test, dtype=np.float64)  # n x m, may contain nan
        self.K = np.asarray(K, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64).reshape(len(y), -1)  # extra covariates
        self.ideal = ideal
        self.h_ident = True  # False: h not identifiable (skip h check)
        self.cov_ref = None  # full-rank covariates for the oracle
        self.k_ref = None  # the "right" K (to measure the shift)
        self.io = {}  # CLI file-handling mutations
        self.cli_ok = True
        self.skip_oracle = False
        self.expect_all_finite = False

    @property
    def X0(self):
        return np.hstack([np.ones((len(self.y), 1)), self.cov])


def _geno_split(rng, n, m_kin, m_test):
    sib = 5 if n % 5 == 0 else (2 if n % 2 == 0 else 1)
    G = genotypes.family(n // sib, sib, m_kin + m_test, seed=int(rng.integers(2**31))).G
    return G[:, :m_kin], G[:, m_kin:]


def _base(rng, n=N_BASE, m_kin=M_KIN, m_test=M_TEST, h2=0.5, k=0, pm=0.0):
    g_kin, g_test = _geno_split(rng, n, m_kin, m_test)
    K = kinship.kinship(g_kin)
    kin = kinship.from_matrix(K)
    ph = phenotypes.simulate(kin, h2=h2, rng=rng, n_covariates=k, missing_rate=pm)
    return Problem(ph.y, g_test, K, ph.X0[:, 1:])


def _binary_cov(rng, n):
    return (rng.random(n) < 0.5).astype(np.float64)


def build(params, rng):
    fam, var = params["family"], params["variant"]
    return BUILDERS[fam](params, var, rng)


def _b_missing(p, var, rng):
    P = _base(rng, pm=p["pm"])
    P.g_test = genotypes.with_missing(P.g_test, p["gm"], rng)
    return P


def _b_degenerate(p, var, rng):
    n = N_BASE
    P = _base(rng, k=1 if var.startswith("eq_cov") else 0)
    P.cli_ok = var != "mono_dosage"
    x = P.g_test[:, 0].copy()
    y = P.y.copy()
    if var == "mono":
        x[:] = 0.0
    elif var == "mono_dosage":
        x[:] = 0.37
    elif var == "mono_ydrop":  # x varies only among individuals whose y is missing
        x[:] = 1.0
        x[:10] = 2.0
        y[:10] = np.nan
    elif var == "mono_xdrop":  # x constant among non-missing, missing elsewhere
        x[:] = 1.0
        x[: n // 2] = np.nan
    elif var == "allmiss":
        x[:] = np.nan
    elif var == "single_minor":
        x[:] = 0.0
        x[7] = 2.0
    elif var == "single_minor_ydrop":
        x[:] = 0.0
        x[7] = 2.0
        y[7] = np.nan
    elif var in ("eq_cov", "eq_cov_x2"):
        c = _binary_cov(rng, n)
        P.cov = c.reshape(-1, 1)
        x = c * (2.0 if var == "eq_cov_x2" else 1.0)
    elif var == "eq_intercept":
        x[:] = 1.0
    else:
        raise ValueError(var)
    P.g_test[:, 0] = x
    P.y = y
    return P


def _b_smalln(p, var, rng):
    n, k = p["n"], p["k"]
    P = _base(rng, n=n, m_kin=200, m_test=M_TEST, k=0)
    P.cov = rng.standard_normal((n, k)) if k else np.zeros((n, 0))
    # y with covariate signal so the fit is not degenerate
    P.y = P.y + (P.cov @ np.full(k, 0.3) if k else 0.0)
    if k + 2 >= n:  # SNP column makes the design saturated (df <= 0)
        P.ideal = "error_or_nan"
    elif k + 3 >= n:  # df = 1: estimable but hardly meaningful
        P.ideal = "result_or_error"
    return P


def _b_kin(p, var, rng):
    if var == "lowrank":
        g_kin, g_test = _geno_split(rng, N_BASE, p["m_kin"], M_TEST)
        K = kinship.kinship(g_kin)
        kin = kinship.from_matrix(K)
        ph = phenotypes.simulate(kin, h2=0.5, rng=rng)
        P = Problem(ph.y, g_test, K, np.zeros((N_BASE, 0)))
        P.skip_oracle = False
        return P
    P = _base(rng)
    K0 = P.K.copy()
    n = N_BASE
    if var == "identity":
        P.K = np.eye(n)
        P.h_ident = False
    elif var in ("dup_pairs", "dup_all"):
        n_unique, reps = (50, 2) if var == "dup_pairs" else (5, 20)
        g_kin, g_test = _geno_split(rng, n_unique, M_KIN, M_TEST)
        g_kin, g_test = np.repeat(g_kin, reps, axis=0), np.repeat(g_test, reps, axis=0)
        K = kinship.kinship(g_kin)
        ph = phenotypes.simulate(kinship.from_matrix(K), h2=0.5, rng=rng)
        P = Problem(ph.y, g_test, K, np.zeros((n, 0)))
    elif var == "nonpsd":
        E = rng.standard_normal((n, n)) * p["noise"] / np.sqrt(n) * np.sqrt(2)
        P.K = K0 + 0.5 * (E + E.T)
        P.k_ref = K0
        P.ideal = "warn_or_error"
        P.skip_oracle = True
    elif var == "wrong_order":
        perm = rng.permutation(n)
        P.K = K0[np.ix_(perm, perm)]
        P.k_ref = K0
        P.ideal = "warn_or_error"
    elif var == "scale":
        P.K = K0 * p["scale"]
    elif var == "upper_only":
        P.K = np.triu(K0)
        P.k_ref = K0
        P.ideal = "warn_or_error"
        P.skip_oracle = True
    elif var == "nan_K":
        P.K = K0.copy()
        P.K[3, 5] = P.K[5, 3] = np.nan
        P.ideal = "error_or_nan"
    elif var == "asym":
        P.K = K0 + np.triu(rng.standard_normal((n, n)) * 0.05, 1)
        P.k_ref = K0
        P.ideal = "warn_or_error"
        P.skip_oracle = True
    else:
        raise ValueError(var)
    return P


def _b_cov(p, var, rng):
    n = N_BASE
    if var == "many":
        return _base(rng, k=p["k"])
    P = _base(rng, k=1)
    c1 = P.cov[:, 0]
    P.ideal = "result_or_error"
    if var == "collinear_dup":
        P.cov_ref = P.cov.copy()
        P.cov = np.column_stack([c1, c1])
    elif var == "collinear_comb":
        c2 = rng.standard_normal(n)
        P.cov_ref = np.column_stack([c1, c2])
        P.cov = np.column_stack([c1, c2, c1 + 2.0 * c2])
    elif var == "intercept_dup":
        P.cov_ref = np.zeros((n, 0))
        P.cov = np.ones((n, 1))
    elif var == "zero_cov":
        P.cov_ref = P.cov.copy()
        P.cov = np.column_stack([c1, np.zeros(n)])
    elif var == "cov_nan":
        P.cov = c1.copy().reshape(-1, 1)
        P.cov[4, 0] = np.nan
        P.ideal = "error_or_nan"
    elif var == "cov_scale":
        P.cov = P.cov * p["scale"]
        P.ideal = "result"
    else:
        raise ValueError(var)
    return P


def _b_pheno(p, var, rng):
    P = _base(rng)
    y = P.y.copy()
    n = len(y)
    if var == "scale":
        y = y * p["scale"]
    elif var == "offset":
        y = y + p["offset"]
    elif var == "constant":
        y = np.full(n, 3.0)
        P.ideal = "error_or_nan"
        P.skip_oracle = True
    elif var == "const_noise":
        y = 3.0 + p["noise"] * rng.standard_normal(n)
        P.skip_oracle = True  # cancellation: even the oracle is unreliable here
    elif var == "binary":
        y = (y > np.median(y)).astype(np.float64)
    elif var == "heavy_t2":
        y = rng.standard_t(2, size=n)
    elif var == "lognormal":
        y = np.exp(2.0 * y)
    elif var == "inf_y":
        y[3] = np.inf
        P.ideal = "error_or_nan"
        P.skip_oracle = True
    elif var == "all_nan":
        y[:] = np.nan
        P.ideal = "error_or_nan"
        P.skip_oracle = True
    elif var == "few_obs":
        keep = rng.choice(n, size=p["n_obs"], replace=False)
        yy = np.full(n, np.nan)
        yy[keep] = y[keep]
        y = yy
    else:
        raise ValueError(var)
    P.y = y
    if var == "const_noise":
        P.ideal = "warn_or_error"
    return P


def _b_io(p, var, rng):
    P = _base(rng, pm=0.05)
    P.io = {"variant": var}
    if var in ("nophenofile", "pheno_missing_indiv", "kfile_header", "kfile_big", "kfile_small"):
        P.ideal = "error_or_nan"
    if var == "pheno_extra_indiv":
        P.ideal = "result"
    return P


BUILDERS = {
    "missing": _b_missing,
    "degenerate": _b_degenerate,
    "smalln": _b_smalln,
    "kin": _b_kin,
    "cov": _b_cov,
    "pheno": _b_pheno,
    "io": _b_io,
}


# ---------------------------------------------------------------------------
# running pylmm
# ---------------------------------------------------------------------------


# An absolute path up to a repo package directory, e.g. "/opt/clone/pylmm/lmm.py:220"
# -> "pylmm/lmm.py:220", so the summary never records the machine-specific checkout path.
_ABS_TO_REPO = re.compile(
    r"(?:[A-Za-z]:)?[\\/]\S*[\\/]"  # the absolute prefix (greedy: up to the last match)
    r"(?=(?:pylmm|sims|tests)[\\/](?:[\w.-]+[\\/])*[\w.-]+\.py)"  # a repo .py path follows
)


def _relpaths(s):
    return _ABS_TO_REPO.sub("", s)


def _err(e):
    return _relpaths("%s: %s" % (type(e).__name__, " ".join(str(e).split())))[:160]


def _impute(G):
    G = np.array(G, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = np.nanmean(G, axis=0)
    idx = np.where(np.isnan(G))
    G[idx] = mu[idx[1]]
    return G


def _warn_summary(ws):
    if not ws:
        return 0, ""
    w = ws[0]
    msg = "%s: %s" % (w.category.__name__, " ".join(str(w.message).split()))
    return len(ws), _relpaths(msg)[:120]


def run_lib(P, impute):
    from pylmm.lmm import LMM, gwas

    out = {"crash": 0, "crash_err": "", "clean_error": 0}
    G = _impute(P.g_test) if impute else P.g_test.copy()
    m = G.shape[1]
    t = np.full(m, np.nan)
    h = np.nan
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as ws, contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("always")
        try:
            L = LMM(P.y.copy(), P.K.copy(), X0=P.X0.copy())
            hh, beta, sigma, ll = L.fit(reml=True)
            h = float(np.sum(hh))
            out["sigma_hat"] = float(np.sum(sigma))
            out["ll_hat"] = float(np.sum(ll))
        except Exception as e:
            out["null_err"] = _err(e)
            out["crash"] = 1
            out["crash_err"] = "null fit: " + _err(e)
        try:
            ts, ps = gwas(P.y.copy(), G, P.K.copy(), X0=P.X0.copy(), reml=True, refit=False)
            out["ret_type"] = type(ts).__name__
            t = np.asarray(ts, dtype=np.float64).reshape(-1)
            if len(t) != m:
                out["crash"] = 1
                out["crash_err"] = "GWAS returned %d stats for %d SNPs" % (len(t), m)
                t = np.full(m, np.nan)
        except Exception as e:
            out["crash"] = 1
            out["crash_err"] = (
                (out["crash_err"] + " | " if out["crash_err"] else "") + "GWAS: " + _err(e)
            )
    out["runtime"] = time.perf_counter() - t0
    out["n_warn"], out["warn_first"] = _warn_summary(ws)
    out["h_hat"] = h
    return out, t, G


def _write_bed(prefix, G01, n):
    """Write a PLINK BED (SNP-major).  G01 values in {0, .5, 1, nan} (n x m)."""
    code = np.full(G01.shape, 0, dtype=np.uint8)
    code[np.isnan(G01)] = 1
    code[G01 == 0.5] = 2
    code[G01 == 1.0] = 3
    pad = (-n) % 4
    with open(prefix + ".bed", "wb") as f:
        f.write(bytes([0x6C, 0x1B, 0x01]))
        for j in range(G01.shape[1]):
            c = np.concatenate([code[:, j], np.zeros(pad, dtype=np.uint8)]).reshape(-1, 4)
            b = c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)
            f.write(b.astype(np.uint8).tobytes())
    with open(prefix + ".bim", "w") as f:
        for j in range(G01.shape[1]):
            f.write("1 snp%d 0 %d A G\n" % (j, j + 1))
    with open(prefix + ".fam", "w") as f:
        for i in range(n):
            f.write("f%d i%d 0 0 0 -9\n" % (i, i))


def _fmt(v, na="NA"):
    return na if np.isnan(v) else repr(float(v))


def run_cli(P, remove):
    n, m = len(P.y), P.g_test.shape[1]
    var = P.io.get("variant", "")
    out = {"crash": 0, "crash_err": "", "clean_error": 0}
    t = np.full(m, np.nan)
    h = np.nan
    with tempfile.TemporaryDirectory() as d:
        pre = os.path.join(d, "d")
        _write_bed(pre, P.g_test / 2.0, n)
        # phenotype file
        rows = [
            "f%d i%d %s" % (i, i, _fmt(P.y[i], "-9" if var == "pheno_neg9" else "NA"))
            for i in range(n)
        ]
        if var == "pheno_shuffled":
            rows = [rows[i] for i in np.random.default_rng(1).permutation(n)]
        elif var == "pheno_missing_indiv":
            rows = rows[:-1]
        elif var == "pheno_extra_indiv":
            rows = [*rows, "fX iX 1.0"]
        with open(pre + ".phenos", "w") as f:
            f.write("\n".join(rows) + "\n")
        # kinship file
        K = P.K
        if var == "kfile_big":
            K = np.pad(K, ((0, 5), (0, 5)))
        elif var == "kfile_small":
            K = K[:-5, :-5]
        np.savetxt(pre + ".kin", K, fmt="%.17g")
        if var == "kfile_header":
            txt = open(pre + ".kin").read()
            open(pre + ".kin", "w").write("# kinship\n" + txt)
        args = [
            sys.executable,
            "-m",
            "pylmm",
            "gwas",
            "-v",
            "--bfile",
            pre,
            "--kinship",
            pre + ".kin",
        ]
        if var != "nophenofile":
            args += ["--pheno", pre + ".phenos"]
        else:
            os.rename(pre + ".phenos", pre + ".other")  # no <base>.phenos either
        if P.cov.shape[1]:
            with open(pre + ".cov", "w") as f:
                for i in range(n):
                    f.write("f%d i%d %s\n" % (i, i, " ".join(_fmt(v) for v in P.cov[i])))
            args += ["--covar", pre + ".cov"]
        if remove:
            args += ["--remove-missing-genotypes"]
        args += ["-o", os.path.join(d, "out.txt")]
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONWARNINGS="default",
        )
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        t0 = time.perf_counter()
        try:
            r = subprocess.run(
                args, capture_output=True, text=True, timeout=CLI_TIMEOUT, env=env, cwd=d
            )
            rc, err = r.returncode, r.stderr
        except subprocess.TimeoutExpired:
            rc, err = -9, "TimeoutExpired"
        out["runtime"] = time.perf_counter() - t0
        lines = [ln for ln in err.strip().splitlines() if ln.strip()]
        warns = [ln for ln in lines if "Warning" in ln]
        out["n_warn"] = len(warns)
        out["warn_first"] = _relpaths(warns[0].strip())[:120] if warns else ""
        for ln in lines:
            if "heritability=" in ln:
                with contextlib.suppress(ValueError):
                    h = float(ln.split("heritability=")[1].split(",")[0])
        if rc != 0:
            out["crash"] = 1
            out["clean_error"] = int(rc == 2 and "error:" in err)
            out["crash_err"] = _relpaths(lines[-1].strip() if lines else "rc=%d" % rc)[:160]
        else:
            fn = os.path.join(d, "out.txt")
            try:
                with open(fn) as f:
                    next(f)
                    vals = [ln.split("\t") for ln in f if ln.strip()]
                if len(vals) != m:
                    out["crash"] = 1
                    out["crash_err"] = "CLI wrote %d rows for %d SNPs" % (len(vals), m)
                else:
                    t = np.array([float(v[3]) for v in vals])
            except Exception as e:
                out["crash"] = 1
                out["crash_err"] = "reading CLI output: " + _err(e)
    out["h_hat"] = h
    return out, t


# ---------------------------------------------------------------------------
# oracle comparison and classification
# ---------------------------------------------------------------------------


def _expected_finite(P, G, i, mode):
    ok = ~np.isnan(P.y) & np.isfinite(P.y)
    x = G[:, i]
    ok &= ~np.isnan(x)
    cov = P.cov_ref if P.cov_ref is not None else P.cov
    X = np.column_stack([np.ones(len(ok)), cov, x])[ok]
    if len(X) <= X.shape[1]:
        return False
    return np.linalg.matrix_rank(X) == X.shape[1]


def oracle_compare(P, G, t, h_hat, remove_mode, cli_ml=False):
    """Oracle h for the y-complete null fit and oracle t for a few SNPs."""
    res = {"or_ok": 0}
    if P.skip_oracle:
        return res
    try:
        ok_y = ~np.isnan(P.y)
        cov = P.cov_ref if P.cov_ref is not None else P.cov
        X0 = np.column_stack([np.ones(len(P.y)), cov])
        K = P.K
        f = oracle.fit(P.y[ok_y], K[np.ix_(ok_y, ok_y)], X0[ok_y], reml=True)
        res["or_h"] = f.h
        res["or_ok"] = 1
        if np.isfinite(h_hat) and P.h_ident:
            res["dh"] = h_hat - f.h
        cand = [i for i in range(G.shape[1]) if _expected_finite(P, G, i, remove_mode)]
        dts, dts_ml = [], []
        g_raw = P.g_test
        skipped = 0
        for i in cand[:N_ORACLE_SNPS]:
            ok = ok_y & ~np.isnan(G[:, i])
            k_ok = K[np.ix_(ok, ok)]
            w = oracle.wald(P.y[ok], k_ok, G[ok, i], X0[ok], reml=True)
            if w.h >= 0.98:  # h at the upper boundary: pylmm's 0.99 clamp (bugs 2/3)
                skipped += 1
                continue
            if np.isfinite(t[i]):
                dts.append(abs(t[i] - w.t) / max(1.0, abs(w.t)))
            else:
                dts.append(np.inf)
            if cli_ml and np.isnan(g_raw[:, i]).any() and np.isfinite(t[i]):
                # CLI --removeMissingGenotypes refits the per-SNP null by ML (bug 1)
                wm = oracle.wald(P.y[ok], k_ok, G[ok, i], X0[ok], reml=False)
                if wm.h < 0.98:
                    dts_ml.append(abs(t[i] - wm.t) / max(1.0, abs(wm.t)))
        res["n_boundary_skipped"] = skipped
        if dts and abs(res.get("dh", 0.0)) < 0.02:
            res["max_dt"] = float(np.max(dts))
            if dts_ml:
                res["max_dt_ml"] = float(np.max(dts_ml))
    except Exception as e:
        res["or_err"] = _err(e)
    return res


def classify(out, ideal):
    """-> (outcome, severity)."""
    if out.get("crash"):
        if out.get("clean_error") and ideal != "result":
            return "OK", "none"
        return "crash", ("high" if ideal == "result" else "medium")
    nan_h = not np.isfinite(out.get("h_hat", np.nan)) or not np.isfinite(out.get("sigma_hat", 0.0))
    n_fin = out.get("n_finite_t", 0)
    if ideal == "error_or_nan":
        if n_fin + out.get("n_inf", 0) > 0:  # numbers where an error/nan is right
            # |t| ~ 0 (p ~ 1) is a spurious null result; larger |t| is believable garbage
            return "wrong", ("medium" if out.get("max_abs_t", 1.0) < 1e-6 else "critical")
        return "OK", ("low" if out.get("n_warn") else "none")
    if out.get("n_inf", 0) > 0:
        return "nan", "high"
    if out.get("n_finite_unexp", 0) > 0:
        return "wrong", ("medium" if out.get("max_abs_t_unexp", 1.0) < 1e-6 else "critical")
    if out.get("bug1_ml"):
        return "wrong", "high"
    if out.get("dh_bad") or out.get("dt_bad"):
        return "wrong", "critical"
    if out.get("dh_ref_bad"):
        return "wrong", "high"
    if nan_h or out.get("n_nan_unexp", 0) > 0:
        return "nan", ("high" if ideal == "result" else "medium")
    if ideal == "warn_or_error":
        return ("OK", "low") if out.get("n_warn") else ("silent", "medium")
    return "OK", ("low" if out.get("n_warn") else "none")


def run_cell(params, seed):
    rng = np.random.default_rng(seed)
    path = params["path"]
    out = {}
    try:
        P = build(params, rng)
    except Exception as e:
        raise RuntimeError("builder failed for %r: %s" % (params, _err(e))) from e
    out["ideal"] = P.ideal
    if path.startswith("cli") and not P.cli_ok:
        out.update(skipped=1, outcome="n/a", severity="none")
        return out
    out["skipped"] = 0
    remove = path in ("lib", "cli_remove")
    if path.startswith("cli"):
        r, t = run_cli(P, remove=(path == "cli_remove"))
        G = P.g_test if remove else _impute(P.g_test)
        out["cli"] = 1
    else:
        r, t, G = run_lib(P, impute=(path == "lib_impute"))
        out["cli"] = 0
    out.update(r)
    m = len(t)
    finite = np.isfinite(t)
    out["n_snps"] = m
    out["n_finite_t"] = int(finite.sum())
    out["n_inf"] = int(np.isinf(t).sum())
    out["max_abs_t"] = float(np.abs(t[finite]).max()) if finite.any() else 0.0
    out["t0"] = float(t[0]) if m and np.isfinite(t[0]) else float("nan")
    if not out["crash"]:
        exp = np.array([_expected_finite(P, G, i, remove) for i in range(m)])
        out["n_expected_nan"] = int((~exp).sum())
        out["n_nan_unexp"] = int((exp & ~finite).sum())
        out["n_finite_unexp"] = int((~exp & finite).sum())
        out["max_abs_t_unexp"] = (
            float(np.abs(t[~exp & finite]).max()) if (~exp & finite).any() else 0.0
        )
    # distance to the "right" K (wrong order / non-PSD / ...)
    if P.k_ref is not None and not out["crash"] and not path.startswith("cli"):
        P2 = Problem(P.y, P.g_test, P.k_ref, P.cov)
        r2, t2, _ = run_lib(P2, impute=False)
        out["h_ref"] = r2["h_hat"]
        both = np.isfinite(t) & np.isfinite(t2)
        out["t_corr_ref"] = (
            float(np.corrcoef(t[both], t2[both])[0, 1]) if both.sum() > 2 else float("nan")
        )
        out["dh_ref"] = out["h_hat"] - r2["h_hat"]
        out["dh_ref_bad"] = int(
            abs(out["dh_ref"]) > 0.1 or (np.isfinite(out["t_corr_ref"]) and out["t_corr_ref"] < 0.9)
        )
    if P.K.shape == (len(P.y), len(P.y)) and np.all(np.isfinite(P.K)):
        with contextlib.suppress(np.linalg.LinAlgError):
            out["min_eig_K"] = float(np.linalg.eigvalsh(0.5 * (P.K + P.K.T))[0])
    if not out["crash"] and P.ideal != "error_or_nan":
        oc = oracle_compare(
            P, G, t, out.get("h_hat", np.nan), remove, cli_ml=(path == "cli_remove")
        )
        out.update(oc)
        out["dh_bad"] = int(abs(oc.get("dh", 0.0)) > 0.02 and oc.get("or_h", 0) < 0.98)
        out["dt_bad"] = int(oc.get("max_dt", 0.0) > 1e-3 + 5 * abs(oc.get("dh", 0.0)))
        if path.startswith("cli") and abs(oc.get("dh", 0.0)) <= 0.02:
            out["dt_bad"] = int(
                oc.get("max_dt", 0.0) > 5e-3 + 5 * abs(oc.get("dh", 0.0))
            )  # h printed to 3 decimals
        # t matches an ML-refit oracle but not REML: the CLI missing-SNP branch (bug 1)
        out["bug1_ml"] = int(out["dt_bad"] and oc.get("max_dt_ml", 1.0) < 5e-3)
    out["outcome"], out["severity"] = classify(out, P.ideal)
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

_ORDER = {"OK": 0, "silent": 1, "nan": 2, "crash": 3, "wrong": 4, "n/a": -1}
_SEV = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _label(row, keys):
    parts = []
    for k in keys:
        v = row[k]
        if k in ("family", "variant", "path"):
            continue
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        parts.append(
            "%s=%g" % (k, v) if isinstance(v, (int, float, np.number)) else "%s=%s" % (k, v)
        )
    return ", ".join(parts)


def case_table(df):
    import pandas as pd

    keys = [
        c
        for c in df.columns
        if c in {"gm", "pm", "n", "k", "m_kin", "noise", "scale", "offset", "n_obs"}
    ]
    d = df.copy()
    d["knobs"] = d.apply(lambda r: _label(r, keys), axis=1)
    rows = []
    for (fam, var, path, knobs), g in d.groupby(["family", "variant", "path", "knobs"], sort=False):
        g = g[g["outcome"] != "n/a"]
        if g.empty:
            continue
        worst = max(g["outcome"], key=lambda o: _ORDER.get(o, 0))
        sev = max(g["severity"], key=lambda s: _SEV.get(s, 0))
        frac = (g["outcome"] == worst).mean()
        errs = g["crash_err"].dropna() if "crash_err" in g else []
        errs = [e for e in errs if e]
        warns = [w for w in (g["warn_first"].dropna() if "warn_first" in g else []) if w]
        rows.append(
            {
                "family": fam,
                "variant": var,
                "path": path,
                "knobs": knobs,
                "ideal": g["ideal"].iloc[0],
                "outcome": worst,
                "frac": round(float(frac), 2),
                "severity": sev,
                "crash_err": errs[0] if errs else "",
                "warn": warns[0] if warns else "",
                "n_nan_unexp": g["n_nan_unexp"].max() if "n_nan_unexp" in g else np.nan,
                "n_finite_unexp": g["n_finite_unexp"].max() if "n_finite_unexp" in g else np.nan,
                "max_abs_dh": g["dh"].abs().max() if "dh" in g else np.nan,
                "max_dt": g["max_dt"].max() if "max_dt" in g else np.nan,
                "dh_ref": g["dh_ref"].abs().max() if "dh_ref" in g else np.nan,
                "runtime_s": round(float(g["runtime"].median()), 3) if "runtime" in g else np.nan,
                "reps": len(g),
            }
        )
    out = pd.DataFrame(rows)
    out["_o"] = out["outcome"].map(_ORDER)
    out["_s"] = out["severity"].map(_SEV)
    return out.sort_values(["_s", "_o"], ascending=False, kind="stable").drop(columns=["_o", "_s"])


def report(df, outdir):
    from sims import report as rpt

    tab = case_table(df)
    counts = tab.groupby(["family", "outcome"]).size().unstack(fill_value=0).reset_index()
    cols = [
        "family",
        "variant",
        "path",
        "knobs",
        "outcome",
        "severity",
        "frac",
        "crash_err",
        "n_nan_unexp",
        "n_finite_unexp",
        "max_abs_dh",
        "max_dt",
        "runtime_s",
    ]
    rpt.write_summary(
        tab,
        outdir,
        "S5: edge cases and robustness",
        notes=(
            "One row per case (params), worst outcome over replicates.  outcome: OK, "
            "silent (dubious input, no warning), nan (unexpected nan/inf), crash (exception; "
            "message in crash_err), wrong (finite but incorrect vs the oracle, or finite where "
            "the statistic is not estimable).  severity: critical = silent wrong answer, "
            "high = crash/nan on valid input, medium = unclear error/silent on garbage input, "
            "low = warnings only.  n_nan_unexp counts SNPs with nan where the statistic is "
            "estimable; n_finite_unexp counts SNPs with a finite t where it is not.  "
            "max_abs_dh / max_dt compare with the dense oracle (h, and relative t).  Sorted by "
            "severity.  See s5_edge_cases.md for the Phase 5 target list."
        ),
        md_columns=cols,
        extra_tables=[("Outcome counts by family", counts)],
    )
