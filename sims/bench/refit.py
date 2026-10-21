"""Benchmark: the exact per-SNP refit (Phase 6.5) vs the per-SNP grid search, and the tests' cost.

``gwas(refit=True)`` / ``pylmm gwas --refit`` fit ``h`` with each SNP in the
model.  Up to Phase 6.3 that was ``LMM.fit(X=x)`` for every SNP: a 101-point
grid and a bounded Brent search per peak, about 130 scalar likelihood
evaluations per SNP (``batch=False`` / ``--no-batch`` still does that, and
``--legacy-fit`` with pylmm 1.0.0's grid).  Now `LMM.fit_snps` does it a block
at a time: the grid for the whole block as matrix products, then a vectorised
safeguarded Newton-Raphson on the analytic derivatives.  This script measures,
on seeded synthetic data (unrelated individuals, K from separate SNPs, h2 =
0.5, an intercept and ``q - 1`` covariates):

* wall time of ``gwas(refit=True)`` vs ``gwas(refit=True, batch=False)``
  (median of repeats; the per-SNP search once; eigendecomposition excluded), and the speedup;
* whether both reach the same maximum: max |dh| and the largest
  log-likelihood shortfall of each against the other (``fit_snps`` vs
  ``fit(X=x)`` per SNP), max |dt| / max |t| and max |d -log10 p| of the scans;
* the wall time of one ``gwas`` scan per test: Wald (fixed ``h``), Wald with
  the exact refit and the LRT.

Run::

    uv run --group sims python -m sims.bench.refit [--quick]
    just bench refit [--quick]

Single-threaded BLAS (``python -m`` imports `sims` first, which pins it).
Tables are printed as markdown.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import platform
import time
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import linalg

from pylmm.lmm import LMM, calculate_kinship, gwas, neg_log10_pvalue
from sims import genotypes

FULL_SIZES = [(300, 2000), (1000, 2000), (2000, 1000)]
QUICK_SIZES = [(200, 300)]
M_KIN = 2000
FULL_COVARIATES = [0, 2]
QUICK_COVARIATES = [2]
FULL_REPEATS = 3
QUICK_REPEATS = 1
TESTS = (("wald", False), ("wald", True), ("lrt", False))


def simulate(n: int, m: int, n_cov: int, seed: int = 0) -> dict[str, Any]:
    """Seeded synthetic data: 0/1/2 genotypes, K from separate SNPs, h2 = 0.5, two causal SNPs."""
    g = genotypes.unrelated(n, m + M_KIN, seed=seed).G
    W, X = g[:, :M_KIN], g[:, M_KIN:]
    X = X[:, X.std(axis=0) > 0]
    K = calculate_kinship(W)
    kva, kve = linalg.eigh(K)
    rng = np.random.default_rng(seed + 1)
    X0 = np.column_stack([np.ones(n), rng.normal(size=(n, n_cov))])
    u = kve @ (np.sqrt(np.clip(kva, 0, None)) * rng.normal(size=n))
    y = X0 @ rng.normal(size=X0.shape[1]) + np.sqrt(0.5) * u + np.sqrt(0.5) * rng.normal(size=n)
    y += 0.3 * X[:, 0] + 0.2 * X[:, 1]
    return {"y": y, "X": X, "K": K, "kva": kva, "kve": kve, "X0": X0}


def median_time(fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    """Median wall time of ``fn()`` over ``repeats`` calls, and the last result."""
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def _quiet(fn: Callable[[], Any]) -> Any:
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        return fn()


def agreement(d: dict[str, Any], reml: bool, m_check: int) -> dict[str, float]:
    """``fit_snps`` vs ``fit(X=x)`` per SNP on the first ``m_check`` SNPs."""
    L = LMM(d["y"], d["K"], d["kva"], d["kve"], d["X0"])
    X = d["X"][:, :m_check]
    new = _quiet(lambda: L.fit_snps(X, reml=reml))
    h_old, ll_old = [], []
    for j in range(X.shape[1]):
        h, _, _, ll = _quiet(lambda j=j: L.fit(X=X[:, j], reml=reml))
        h_old.append(float(h))
        ll_old.append(ll)
    h_old_a, ll_old_a = np.asarray(h_old), np.asarray(ll_old)
    return {
        "dh_max": float(np.max(np.abs(new.h - h_old_a))),
        "short_new": float(np.max(ll_old_a - new.ll)),  # > 0: the new fit is below the old
        "short_old": float(np.max(new.ll - ll_old_a)),
        "fallbacks": float(L._snp_fallbacks),
    }


def run_synthetic(n: int, m: int, n_cov: int, reml: bool, repeats: int) -> dict[str, Any]:
    """Time the refit both ways on one synthetic data set and compare the numbers."""
    d = simulate(n, m, n_cov)
    args = (d["y"], d["X"], d["K"], d["kva"], d["kve"], d["X0"], reml)
    # the per-SNP search takes seconds to minutes: timed once
    t_old, (ts_old, _) = median_time(
        lambda: _quiet(lambda: gwas(*args, refit=True, batch=False)), 1
    )
    t_new, (ts_new, _) = median_time(lambda: _quiet(lambda: gwas(*args, refit=True)), repeats)
    ts_old, ts_new = np.asarray(ts_old, dtype=float), np.asarray(ts_new, dtype=float)
    df = n - d["X0"].shape[1] - 1
    dlp = np.abs(neg_log10_pvalue(ts_new, df) - neg_log10_pvalue(ts_old, df))
    out = {
        "n": n,
        "m": d["X"].shape[1],
        "q": d["X0"].shape[1],
        "method": "REML" if reml else "ML",
        "per_snp_s": t_old,
        "exact_s": t_new,
        "dt_rel": float(np.nanmax(np.abs(ts_new - ts_old)) / np.nanmax(np.abs(ts_old))),
        "dlog10p": float(np.nanmax(dlp)),
    }
    out.update(agreement(d, reml, min(200, d["X"].shape[1])))
    return out


def run_tests(n: int, m: int, repeats: int) -> dict[str, Any]:
    """Wall time of one ``gwas`` scan per test (REML; the LRT is ML)."""
    d = simulate(n, m, 2)
    args = (d["y"], d["X"], d["K"], d["kva"], d["kve"], d["X0"])
    out: dict[str, Any] = {"n": n, "m": d["X"].shape[1]}
    for test, refit in TESTS:
        key = test + ("_refit" if refit else "")
        out[key] = median_time(
            lambda test=test, refit=refit: _quiet(lambda: gwas(*args, test=test, refit=refit)),
            repeats,
        )[0]
    return out


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, Callable[[dict], str]]]) -> str:
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f(r) for _, f in cols) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m sims.bench.refit", description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small sizes (seconds)")
    args = ap.parse_args(argv)
    sizes = QUICK_SIZES if args.quick else FULL_SIZES
    covs = QUICK_COVARIATES if args.quick else FULL_COVARIATES
    repeats = QUICK_REPEATS if args.quick else FULL_REPEATS
    print("# refit benchmark\n")
    print(
        "python %s, numpy %s, %s %s, single-threaded BLAS\n"
        % (platform.python_version(), np.__version__, platform.system(), platform.machine())
    )
    rows = []
    for n, m in sizes:
        for n_cov in covs:
            for reml in (True, False):
                rows.append(run_synthetic(n, m, n_cov, reml, repeats))
    print("### gwas(refit=True): exact (fit_snps) vs per-SNP grid search (batch=False)\n")
    print(
        "dh, shortfalls: fit_snps vs fit(X=x) on the first 200 SNPs; short new: the largest "
        "log-likelihood by which the exact fit is below the grid search (> 0: it missed); "
        "short old: the reverse.  dt: max |dt| / max |t| of the scans.\n"
    )
    print(
        _table(
            rows,
            [
                ("n", lambda r: str(r["n"])),
                ("m", lambda r: str(r["m"])),
                ("q", lambda r: str(r["q"])),
                ("method", lambda r: r["method"]),
                ("per-SNP s", lambda r: "%.2f" % r["per_snp_s"]),
                ("exact s", lambda r: "%.3f" % r["exact_s"]),
                ("speedup", lambda r: "%.0fx" % (r["per_snp_s"] / r["exact_s"])),
                ("max abs dh", lambda r: "%.1e" % r["dh_max"]),
                ("short new", lambda r: "%.1e" % r["short_new"]),
                ("short old", lambda r: "%.1e" % r["short_old"]),
                ("dt", lambda r: "%.1e" % r["dt_rel"]),
                ("max abs d-log10p", lambda r: "%.1e" % r["dlog10p"]),
                ("fallbacks", lambda r: "%d" % r["fallbacks"]),
            ],
        )
    )
    print()
    trows = [run_tests(n, m, repeats) for n, m in sizes]
    print("### one gwas() scan per test (q = 3; seconds)\n")
    print(
        _table(
            trows,
            [("n", lambda r: str(r["n"])), ("m", lambda r: str(r["m"]))]
            + [
                (
                    t + (" + refit" if rf else ""),
                    lambda r, k=t + ("_refit" if rf else ""): "%.3f" % r[k],
                )
                for t, rf in TESTS
            ],
        )
    )


if __name__ == "__main__":
    main()
