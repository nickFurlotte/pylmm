"""Benchmark: batched vs one-SNP-at-a-time Wald tests (Phase 6.1).

`pylmm.lmm.gwas` and ``pylmm gwas`` test the complete SNPs of a block together
(`LMM.association_batch`: one matrix product rotates the block, closed-form
per-SNP GLS) unless ``batch=False`` / ``--no-batch``, which is the 1.0.0 code
path (`LMM.association` per SNP).  This script measures, on seeded synthetic
data and on the bundled 1219-mouse BED file:

* wall time of ``gwas()`` batched vs per SNP (median of repeats, null fit and
  estimability checks included, eigendecomposition excluded), and the speedup;
* the largest differences between the two paths in t, p, beta and its
  standard error (beta and se from `LMM.association` vs
  `LMM.association_batch` on the estimable SNPs);
* end-to-end ``pylmm gwas`` on a SNP slice of ``data/snps.132k.clean.noX``,
  default vs ``--no-batch``, and a diff of the two output files.  With
  ``--cli-baseline DIR`` (a checkout of another revision, e.g. ``git worktree
  add /tmp/base origin/phase6``) the default command of DIR's ``pylmm`` is
  timed too (``PYTHONPATH=DIR``) and its output compared byte for byte.

Run::

    uv run --group sims python -m sims.bench.gwas_batch [--quick] [--cli-baseline DIR] [--cli-only]
    just bench gwas_batch [--quick]

Timings are single-threaded BLAS by default (``python -m`` imports the
`sims` package first, which pins the thread pools to 1); the full run also repeats everything with one BLAS thread
per CPU in a child process.  ``--threads N`` runs only N threads.  Tables are
printed as markdown.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import linalg

from pylmm.lmm import LMM, calculate_kinship, gwas, snp_estimable
from sims import genotypes

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BED_BASE = os.path.join(REPO, "data", "snps.132k.clean.noX")

# (n individuals, m tested SNPs); kinship from a separate set of M_KIN SNPs
FULL_SIZES = [(500, 20000), (2000, 10000)]
QUICK_SIZES = [(300, 2000)]
M_KIN = 2000
COVARIATES = [0, 3]  # covariates besides the intercept
FULL_REPEATS = 3
QUICK_REPEATS = 3
FULL_CLI_SNPS = 20000
QUICK_CLI_SNPS = 1000
FULL_CLI_REPEATS = 3
QUICK_CLI_REPEATS = 1


def simulate(n: int, m: int, n_cov: int, seed: int = 0) -> dict[str, Any]:
    """Seeded synthetic data: 0/1/2 genotypes, K from separate SNPs, y with h2 = 0.5.

    The covariates (besides the intercept) have very different scales
    (1, 100, 0.001, ...), to stress the closed-form GLS.
    """
    g = genotypes.unrelated(n, m + M_KIN, seed=seed).G
    W, X = g[:, :M_KIN], g[:, M_KIN:]
    K = calculate_kinship(W)
    kva, kve = linalg.eigh(K)
    rng = np.random.default_rng(seed + 1)
    scales = np.array([1.0, 100.0, 1e-3] * n_cov)[:n_cov]
    X0 = np.column_stack([np.ones(n), rng.normal(size=(n, n_cov)) * scales])
    u = kve @ (np.sqrt(np.clip(kva, 0, None)) * rng.normal(size=n))
    beta0 = rng.normal(size=X0.shape[1]) / np.maximum(np.abs(X0).mean(axis=0), 1e-12)
    y = X0 @ beta0 + np.sqrt(0.5) * u + np.sqrt(0.5) * rng.normal(size=n)
    y += 0.3 * X[:, 0] + 0.2 * X[:, 1]  # two causal SNPs, so some p are small
    return {"y": y, "X": X, "K": K, "kva": kva, "kve": kve, "X0": X0}


def median_time(fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    """Median wall time of ``fn()`` over ``repeats`` calls, and the last result."""
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def max_diffs(a: Any, b: Any, p_floor: float = 0.0) -> tuple[float, float, bool]:
    """Max |a - b|, max |a - b| / |b| (over b != 0) and whether the nan positions agree.

    With ``p_floor``, only entries with both values above it are compared
    (p-values: the underflow region is excluded).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    ok = ~np.isnan(a) & ~np.isnan(b) & (np.abs(a) > p_floor) & (np.abs(b) > p_floor)
    if not ok.any():
        return 0.0, 0.0, same_nan
    d = np.abs(a[ok] - b[ok])
    nz = b[ok] != 0
    rel = float((d[nz] / np.abs(b[ok][nz])).max()) if nz.any() else 0.0
    return float(d.max()), rel, same_nan


def run_synthetic(n: int, m: int, n_cov: int, reml: bool, repeats: int) -> dict[str, Any]:
    """Time gwas() both ways on one synthetic data set and compare the numbers."""
    d = simulate(n, m, n_cov)
    args = (d["y"], d["X"], d["K"], d["kva"], d["kve"], d["X0"], reml)
    t_old, (ts_old, ps_old) = median_time(lambda: gwas(*args, batch=False), repeats)
    t_new, (ts_new, ps_new) = median_time(lambda: gwas(*args), repeats)

    # beta and its standard error, on the SNPs gwas tests
    L = LMM(d["y"], d["K"], d["kva"], d["kve"], d["X0"])
    L.fit(reml=reml)
    Q0 = linalg.qr(d["X0"], mode="economic")[0]
    est = [j for j in range(m) if snp_estimable(d["X"][:, j], d["X0"], Q=Q0)]
    ref = np.array([L.association(d["X"][:, j], reml=reml, return_beta=True) for j in est])
    res = L.association_batch(d["X"][:, est], reml=reml)
    p_old, p_new = np.asarray(ps_old, dtype=float), np.asarray(ps_new, dtype=float)
    return {
        "n": n,
        "m": m,
        "q": d["X0"].shape[1],
        "method": "REML" if reml else "ML",
        "h": float(L.opt_h),
        "per_snp_s": t_old,
        "batched_s": t_new,
        "t": max_diffs(ts_new, ts_old),
        "p": max_diffs(ps_new, ps_old, p_floor=1e-300),
        "p_zero": int(((p_old == 0) | (p_new == 0)).sum()),
        "beta": max_diffs(res.beta, ref[:, 2]),
        "se": max_diffs(np.sqrt(res.beta_var), np.sqrt(ref[:, 3])),
        "n_nan": int(np.isnan(np.asarray(ts_new, dtype=float)).sum()),
    }


# ---------------------------------------------------------------------------
# pylmm gwas on the bundled BED file
# ---------------------------------------------------------------------------


def write_bed_slice(out_base: str, m: int) -> int:
    """Write the first ``m`` SNPs of the bundled BED file set to ``out_base``.*; returns m."""
    with open(BED_BASE + ".fam") as f:
        n = sum(1 for line in f if line.strip())
    with open(BED_BASE + ".bim") as f:
        bim = f.readlines()
    m = min(m, len(bim))
    nbytes = (n + 3) // 4
    with open(BED_BASE + ".bed", "rb") as f:
        data = f.read(3 + m * nbytes)
    with open(out_base + ".bed", "wb") as f:
        f.write(data)
    with open(out_base + ".bim", "w") as f:
        f.writelines(bim[:m])
    shutil.copyfile(BED_BASE + ".fam", out_base + ".fam")
    return m


def read_output(fn: str) -> tuple[list[str], list[str], np.ndarray]:
    """The lines (without header), SNP IDs and numeric columns of a ``pylmm gwas`` output."""
    with open(fn) as f:
        lines = f.read().splitlines()[1:]
    rows = [line.split("\t") for line in lines]
    return lines, [r[0] for r in rows], np.array([[float(v) for v in r[1:]] for r in rows])


def run_cli(
    m: int, repeats: int, workdir: str, baseline: str | None = None
) -> dict[str, Any] | None:
    """Time ``pylmm gwas`` on the first ``m`` SNPs, default vs ``--no-batch``; diff the outputs.

    With ``baseline`` (the root of another pylmm checkout), also time its default
    command (``PYTHONPATH=baseline``) and compare its output with this one's.
    """
    if not all(
        os.path.exists(BED_BASE + ext)
        for ext in (".bed", ".bim", ".fam", ".pylmm.kin", ".fake.phenos")
    ):
        return None
    base = os.path.join(workdir, "slice")
    m = write_bed_slice(base, m)
    timings = {}
    # Each run imports pylmm from its own tree: this checkout (REPO), or the baseline.
    runs: list[tuple[str, list[str], str]] = [
        ("batched", [], REPO),
        ("per_snp", ["--no-batch"], REPO),
    ]
    if baseline:
        runs.append(("baseline", [], os.path.abspath(baseline)))
    trees = {}
    for tag, extra, path in runs:
        out = os.path.join(workdir, "%s.tsv" % tag)
        env = dict(os.environ)
        env["PYTHONPATH"] = path + os.pathsep + env.get("PYTHONPATH", "")
        # Run in workdir (`python -m` puts the working directory first on sys.path),
        # and check where pylmm is imported from.
        where = subprocess.run(
            [sys.executable, "-c", "import pylmm; print(pylmm.__file__)"],
            env=env,
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if os.path.commonpath([os.path.realpath(where), os.path.realpath(path)]) != (
            os.path.realpath(path)
        ):
            raise RuntimeError(
                "%s run: pylmm is imported from %s, not from %s" % (tag, where, path)
            )
        trees[tag] = path
        cmd = [
            sys.executable,
            "-m",
            "pylmm",
            "gwas",
            "-q",
            "--bfile",
            base,
            "--kinship",
            BED_BASE + ".pylmm.kin",
            "--pheno",
            BED_BASE + ".fake.phenos",
            "-o",
            out,
            *extra,
        ]
        timings[tag] = median_time(
            lambda cmd=cmd, env=env: subprocess.run(
                cmd, check=True, capture_output=True, env=env, cwd=workdir
            ),
            repeats,
        )[0]
    lines_b, ids_b, vals_b = read_output(os.path.join(workdir, "batched.tsv"))
    lines_o, ids_o, vals_o = read_output(os.path.join(workdir, "per_snp.tsv"))
    assert ids_b == ids_o, "SNP order differs"
    baseline_result = None
    if baseline:
        with (
            open(os.path.join(workdir, "baseline.tsv"), "rb") as f1,
            open(os.path.join(workdir, "batched.tsv"), "rb") as f2,
        ):
            baseline_result = {"s": timings["baseline"], "identical": f1.read() == f2.read()}
    return {
        "baseline": baseline_result,
        "trees": trees,
        "m": m,
        "n": sum(1 for _ in open(BED_BASE + ".fam")),
        "batched_s": timings["batched"],
        "per_snp_s": timings["per_snp"],
        "identical_lines": sum(a == b for a, b in zip(lines_b, lines_o)),
        "cols": {
            name: max_diffs(vals_b[:, j], vals_o[:, j], 1e-300 if name == "P_VALUE" else 0.0)
            for j, name in enumerate(["BETA", "BETA_SD", "F_STAT", "P_VALUE"])
        },
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _g(x: float) -> str:
    return "%.1e" % x


def print_synthetic(rows: Sequence[dict[str, Any]]) -> None:
    print("### gwas(): batched vs per SNP (synthetic, seeded)\n")
    print("| n | m | q | method | h | per SNP (s) | batched (s) | speedup | nan SNPs |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            "| %d | %d | %d | %s | %.2f | %.2f | %.3f | %.1fx | %d |"
            % (
                r["n"],
                r["m"],
                r["q"],
                r["method"],
                r["h"],
                r["per_snp_s"],
                r["batched_s"],
                r["per_snp_s"] / r["batched_s"],
                r["n_nan"],
            )
        )
    print("\n### Largest differences, batched vs per SNP (max abs / max rel)\n")
    print("| n | m | q | method | t | p (p > 1e-300) | beta | se | nan positions equal |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        same = all(r[k][2] for k in ("t", "p", "beta", "se"))
        print(
            "| %d | %d | %d | %s | %s | %s | %s | %s | %s |"
            % (
                r["n"],
                r["m"],
                r["q"],
                r["method"],
                *("%s / %s" % (_g(r[k][0]), _g(r[k][1])) for k in ("t", "p", "beta", "se")),
                "yes" if same else "NO",
            )
        )
    print()


def print_cli(r: dict[str, Any] | None) -> None:
    print("### pylmm gwas on data/snps.132k.clean.noX (first SNPs), default vs --no-batch\n")
    if r is None:
        print("(skipped: the bundled data files are not present)\n")
        return
    print("| n | SNPs | --no-batch (s) | default (s) | speedup | identical lines |")
    print("|---|---|---|---|---|---|")
    print(
        "| %d | %d | %.2f | %.2f | %.1fx | %d of %d |"
        % (
            r["n"],
            r["m"],
            r["per_snp_s"],
            r["batched_s"],
            r["per_snp_s"] / r["batched_s"],
            r["identical_lines"],
            r["m"],
        )
    )
    if r["baseline"] is not None:
        b = r["baseline"]
        print("\n| --cli-baseline default (s) | default (s) | speedup | output byte-identical |")
        print("|---|---|---|---|")
        print(
            "| %.2f | %.2f | %.2fx | %s |"
            % (b["s"], r["batched_s"], b["s"] / r["batched_s"], "yes" if b["identical"] else "NO")
        )
    print("\npylmm imported from: %s\n" % "; ".join("%s: %s" % kv for kv in r["trees"].items()))
    print("| column | max abs diff | max rel diff | nan positions equal |")
    print("|---|---|---|---|")
    for name, (a, rel, same) in r["cols"].items():
        print("| %s | %s | %s | %s |" % (name, _g(a), _g(rel), "yes" if same else "NO"))
    print(
        "\n(Wall time of the whole command, including reading the 1219 x 1219 text kinship "
        "and its eigendecomposition.)\n"
    )


def run_all(args: argparse.Namespace, threads: int) -> list[dict[str, Any]]:
    """Run and print every benchmark at the current BLAS thread setting."""
    sizes = QUICK_SIZES if args.quick else FULL_SIZES
    repeats = QUICK_REPEATS if args.quick else FULL_REPEATS
    print("## BLAS threads: %d\n" % threads)
    rows = []
    if not args.cli_only:
        rows = [run_synthetic(n, m, c, True, repeats) for n, m in sizes for c in COVARIATES]
        n, m = sizes[0]
        rows.append(run_synthetic(n, m, COVARIATES[-1], False, repeats))  # ML
        print_synthetic(rows)
    if not args.no_cli:
        with tempfile.TemporaryDirectory() as workdir:
            print_cli(
                run_cli(
                    QUICK_CLI_SNPS if args.quick else FULL_CLI_SNPS,
                    args.cli_repeats or (QUICK_CLI_REPEATS if args.quick else FULL_CLI_REPEATS),
                    workdir,
                    args.cli_baseline,
                )
            )
    sys.stdout.flush()
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m sims.bench.gwas_batch", description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small sizes (well under a minute)")
    ap.add_argument("--no-cli", action="store_true", help="skip the pylmm gwas timing")
    ap.add_argument(
        "--cli-only", action="store_true", help="only the pylmm gwas timing (no synthetic gwas())"
    )
    ap.add_argument(
        "--cli-baseline",
        metavar="DIR",
        help="also time the default pylmm gwas of the pylmm checkout DIR (another revision)",
    )
    ap.add_argument("--cli-repeats", type=int, help="repeats of each pylmm gwas timing")
    ap.add_argument(
        "--threads",
        type=int,
        help="BLAS threads (default: 1; the full run then repeats with one per CPU)",
    )
    args = ap.parse_args(argv)
    if args.threads:
        counts = [args.threads]
    else:
        counts = [1] if args.quick else [1, os.cpu_count() or 1]
    child = os.environ.get("PYLMM_BENCH_CHILD") == "1"
    if not child:
        print("# gwas batch benchmark\n")
        print(
            "python %s, numpy %s, %s %s, %d CPUs\n"
            % (
                platform.python_version(),
                np.__version__,
                platform.system(),
                platform.machine(),
                os.cpu_count() or 1,
            )
        )
        sys.stdout.flush()
    for k in counts:
        if os.environ.get("SIMS_BLAS_THREADS", "1") == str(k):
            run_all(args, k)
        else:  # the thread count is fixed when numpy loads: run it in a child process
            flags = ["--quick"] * args.quick + ["--no-cli"] * args.no_cli
            flags += ["--cli-only"] * args.cli_only
            if args.cli_baseline:
                flags += ["--cli-baseline", args.cli_baseline]
            if args.cli_repeats:
                flags += ["--cli-repeats", str(args.cli_repeats)]
            env = dict(os.environ, SIMS_BLAS_THREADS=str(k), PYLMM_BENCH_CHILD="1")
            subprocess.run(
                [sys.executable, "-m", "sims.bench.gwas_batch", *flags, "--threads", str(k)],
                env=env,
                check=True,
            )


if __name__ == "__main__":
    main()
