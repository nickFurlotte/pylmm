"""Benchmark: ``pylmm gwas --loco`` vs the standard kinship + scan (Phase 6.4).

Times, end to end on the command line (``python -m pylmm``, this checkout):

* **standard**: ``pylmm kinship`` (one K of all SNPs) then ``pylmm gwas
  --kinship``: two passes over the genotypes, one eigendecomposition;
* **loco**: ``pylmm gwas --loco``: one pass to build every K_{-c}
  (`pylmm.loco.LocoKinship`), one scan pass, one eigendecomposition and null
  fit per chromosome;

and compares the two outputs: λ_GC, the number of SNPs below 1e-3 / 1e-5, and
the top hits of each.  The full run uses the bundled README data (1219 mice x
98,740 SNPs on 19 autosomes, fake phenotype), and then also scans a seeded
polygenic trait simulated on those genotypes (h2 = 0.5 from 300 SNPs on the
odd autosomes) with both kinships, for λ_GC on the null even autosomes (the
fake phenotype has h ~ 0, so the two kinships barely differ on it); ``--quick`` a seeded synthetic
BED (300 x 3000, 3 chromosomes).  Wall times include Python start-up and file
I/O.  BLAS threads: ``--threads N``, else the environment's (importing `sims`
sets it to ``$SIMS_BLAS_THREADS``, default 1, and the pylmm runs inherit it).

Run::

    just bench loco [--quick] [--threads N]
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy import stats

from sims.bench.bed_blocks import write_synthetic_bed

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BED_BASE = os.path.join(REPO, "data", "snps.132k.clean.noX")
PHENOS = BED_BASE + ".fake.phenos"
QUICK_N, QUICK_M = 300, 3000
TOP = 10


def _pylmm(args: list[str], workdir: str, threads: int | None) -> tuple[float, str]:
    """Run ``python -m pylmm <args>`` from this checkout; (wall seconds, merged log)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    if threads is not None:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):  # fmt: skip
            env[var] = str(threads)
    begin = time.perf_counter()
    p = subprocess.run(
        [sys.executable, "-m", "pylmm", *args, "-v"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - begin
    if p.returncode != 0:
        raise RuntimeError("pylmm %s failed:\n%s" % (" ".join(args), p.stderr))
    return wall, p.stderr


def _read(fn: str) -> tuple[list[str], np.ndarray]:
    ids, vals = [], []
    with open(fn) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[0])
            vals.append([float(x) for x in parts[1:]])
    return ids, np.array(vals)


def _lambda(p: np.ndarray) -> float:
    p = p[np.isfinite(p)]
    return float(np.median(stats.chi2.isf(p, 1)) / stats.chi2.ppf(0.5, 1))


def _synthetic(workdir: str) -> tuple[str, str]:
    """A seeded BED (3 chromosomes) and a heritable phenotype; (prefix, pheno file)."""
    base = os.path.join(workdir, "synthetic")
    G = write_synthetic_bed(base, QUICK_N, QUICK_M, missing=0.02, seed=4)
    rng = np.random.default_rng(4)
    Z = np.nan_to_num((G - np.nanmean(G, 0)) / np.nanstd(G, 0))
    y = Z @ rng.normal(0, np.sqrt(0.6 / QUICK_M), QUICK_M) + rng.normal(0, np.sqrt(0.4), QUICK_N)
    y = y + 0.4 * Z[:, 100]
    pheno = base + ".phen"
    with open(pheno, "w") as f:
        for i, v in enumerate(y):
            f.write("F%d I%d %r\n" % (i + 1, i + 1, float(v)))
    return base, pheno


SIM_H2 = 0.5
SIM_CAUSAL = 300


def _simulated_readme(workdir: str, seed: int = 0) -> tuple[str, np.ndarray]:
    """A polygenic phenotype on the README genotypes; (pheno file, chromosome of each SNP).

    ``SIM_CAUSAL`` causal SNPs (per-SD effects, mean-imputed) on the odd
    autosomes, ``h2 = SIM_H2``: the SNPs of the even autosomes are null.
    """
    from pylmm.plink_io import PlinkReader

    with PlinkReader(BED_BASE, norm_genotype=False) as r:
        chrom = np.asarray(r.variants.chrom)
        ids = r.indivs
        rng = np.random.default_rng(seed)
        odd = np.flatnonzero(np.array([int(c) % 2 == 1 for c in chrom]))
        causal = np.sort(rng.choice(odd, SIM_CAUSAL, replace=False))
        cols, start = [], 0
        for G, _ in r.iter_blocks(4096):
            sel = causal[(causal >= start) & (causal < start + G.shape[1])] - start
            cols.append(G[:, sel])
            start += G.shape[1]
    C = np.hstack(cols)
    C = np.where(np.isnan(C), np.nanmean(C, 0), C)
    C = (C - C.mean(0)) / C.std(0)
    g = C @ rng.standard_normal(SIM_CAUSAL)
    e = rng.standard_normal(len(g))
    y = np.sqrt(SIM_H2) * (g - g.mean()) / g.std() + np.sqrt(1 - SIM_H2) * (e - e.mean()) / e.std()
    fn = os.path.join(workdir, "simulated.phen")
    with open(fn, "w") as f:
        for (fid, iid), v in zip(ids, y):
            f.write("%s %s %r\n" % (fid, iid, float(v)))
    return fn, chrom


def run_simulated(threads: int | None, workdir: str, kin: str) -> dict:
    """README genotypes, a simulated polygenic trait: lambda_GC on the null (even) autosomes."""
    pheno, chrom = _simulated_readme(workdir)
    even = np.array([int(c) % 2 == 0 for c in chrom])
    out = {}
    for name, flags in (("standard", ["--kinship", kin]), ("loco", ["--loco"])):
        fn = os.path.join(workdir, "sim_%s.tsv" % name)
        _, log = _pylmm(
            ["gwas", "--bfile", BED_BASE, *flags, "--pheno", pheno, "-o", fn], workdir, threads
        )
        hs = [float(x) for x in re.findall(r"heritability=(\S+),", log)]
        p = _read(fn)[1][:, 3]
        out[name] = {
            "h": (min(hs), max(hs)),
            "lambda_even": _lambda(p[even]),
            "t1e_0.05_even": float(np.mean(p[even][np.isfinite(p[even])] <= 0.05)),
            "lambda_odd": _lambda(p[~even]),
        }
    return out


def run(quick: bool, threads: int | None, workdir: str) -> dict:
    """Time both pipelines and compare their outputs."""
    if quick:
        base, pheno = _synthetic(workdir)
        label = "synthetic %d x %d, 3 chromosomes" % (QUICK_N, QUICK_M)
    else:
        if not os.path.exists(BED_BASE + ".bed"):
            raise SystemExit("the bundled README data is missing (%s.bed)" % BED_BASE)
        base, pheno = BED_BASE, PHENOS
        label = "README BED, 1219 x 98,740, 19 autosomes, fake phenotype"
    kin = os.path.join(workdir, "all.kin")
    std_out = os.path.join(workdir, "standard.tsv")
    loco_out = os.path.join(workdir, "loco.tsv")
    t_kin, _ = _pylmm(["kinship", "--bfile", base, "-o", kin], workdir, threads)
    t_gwas, log_std = _pylmm(
        ["gwas", "--bfile", base, "--kinship", kin, "--pheno", pheno, "-o", std_out],
        workdir,
        threads,
    )
    t_loco, log_loco = _pylmm(
        ["gwas", "--bfile", base, "--loco", "--pheno", pheno, "-o", loco_out], workdir, threads
    )
    built = re.search(r"Built the kinships of .* in (\S+)s", log_loco)
    h_std = [float(x) for x in re.findall(r"heritability=(\S+),", log_std)]
    h_loco = [float(x) for x in re.findall(r"heritability=(\S+),", log_loco)]
    ids, std = _read(std_out)
    ids2, loco = _read(loco_out)
    assert ids == ids2
    out = {
        "label": label,
        "time": {
            "standard: pylmm kinship": t_kin,
            "standard: pylmm gwas --kinship": t_gwas,
            "standard: total": t_kin + t_gwas,
            "pylmm gwas --loco": t_loco,
            "  of which building the LOCO kinships": float(built.group(1)) if built else np.nan,
        },
        "h": {"standard": h_std[0], "loco min": min(h_loco), "loco max": max(h_loco)},
        "n_chrom": len(h_loco),
        "compare": {},
        "top": {},
    }
    for name, v in (("standard", std), ("loco", loco)):
        p = v[:, 3]
        out["compare"][name] = {
            "lambda_gc": _lambda(p),
            "p<1e-3": int((p < 1e-3).sum()),
            "p<1e-5": int((p < 1e-5).sum()),
            "min p": float(np.nanmin(p)),
        }
        order = np.argsort(np.where(np.isfinite(p), p, 2.0))[:TOP]
        out["top"][name] = [(ids[j], float(p[j])) for j in order]
    d = -np.log10(loco[:, 3]) + np.log10(std[:, 3])
    out["max |d -log10 p|"] = float(np.nanmax(np.abs(d)))
    out["median d -log10 p"] = float(np.nanmedian(d))
    if not quick:
        out["simulated"] = run_simulated(threads, workdir, kin)
    return out


def print_report(r: dict, threads: int | None) -> None:
    print("## pylmm gwas --loco vs pylmm kinship + pylmm gwas\n")
    print("%s; %s, Python %s, BLAS threads: %s\n" % (
        r["label"], platform.platform(), platform.python_version(),
        os.environ.get("OMP_NUM_THREADS", "unset") if threads is None else threads))  # fmt: skip
    print("| step | wall time (s) |\n|---|---|")
    for k, v in r["time"].items():
        print("| %s | %.1f |" % (k, v))
    print("\nnull h: standard %.4f; LOCO %.4f-%.4f over %d chromosomes\n" % (
        r["h"]["standard"], r["h"]["loco min"], r["h"]["loco max"], r["n_chrom"]))  # fmt: skip
    print("| | lambda_GC | SNPs p < 1e-3 | SNPs p < 1e-5 | min p |\n|---|---|---|---|---|")
    for k, v in r["compare"].items():
        print("| %s | %.4f | %d | %d | %.3g |" % (k, v["lambda_gc"], v["p<1e-3"], v["p<1e-5"],
                                                  v["min p"]))  # fmt: skip
    print("\nmax |Δ-log10 p| (LOCO - standard) %.3f, median %.4f\n" % (
        r["max |d -log10 p|"], r["median d -log10 p"]))  # fmt: skip
    print("| rank | standard: SNP | p | LOCO: SNP | p |\n|---|---|---|---|---|")
    for k, ((a, pa), (b, pb)) in enumerate(zip(r["top"]["standard"], r["top"]["loco"]), 1):
        print("| %d | %s | %.3g | %s | %.3g |" % (k, a, pa, b, pb))
    sim = r.get("simulated")
    if sim:
        print("\n### A simulated polygenic trait on the same genotypes\n")
        print("h2 = %.1f from %d causal SNPs (per-SD effects) on the odd autosomes, seed 0; the "
              "even autosomes are null.\n" % (SIM_H2, SIM_CAUSAL))  # fmt: skip
        print("| | null h | lambda_GC, even (null) | type-I 0.05, even | lambda_GC, odd |")
        print("|---|---|---|---|---|")
        for k, v in sim.items():
            print("| %s | %.3f-%.3f | %.3f | %.4f | %.3f |" % (
                k, v["h"][0], v["h"][1], v["lambda_even"], v["t1e_0.05_even"], v["lambda_odd"]))  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="a small synthetic BED instead")
    ap.add_argument("--threads", type=int, default=None, help="BLAS threads for the pylmm runs")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as workdir:
        r = run(args.quick, args.threads, workdir)
    print_report(r, args.threads)


if __name__ == "__main__":
    main()
