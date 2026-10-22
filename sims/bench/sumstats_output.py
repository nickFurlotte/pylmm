"""Benchmark: the cost of the summary-statistics output of ``pylmm gwas`` (Phase 6.2b).

Runs ``pylmm gwas`` (in this process, ``pylmm.cli.main``) on the bundled 1219-mouse
BED file and its kinship, once with ``--output-format legacy`` (pylmm 1.0.0's five
columns) and once with the default table, as ``.tsv`` and ``.tsv.gz``, and reports
the wall time, the scan time of the run record (reading, testing and writing the
SNPs), the file size, and checks that the two formats agree: ``t_stat = -F_STAT``
and ``p_value = P_VALUE`` exactly, and ``beta = -BETA / (2 sd)`` with ``sd`` the SD of
the A2 fraction over the called genotypes (relative difference reported).  With
``--quick`` the first 8,192 SNPs (a copy in a temporary directory).  Times are the
median of ``--repeats`` runs; BLAS is pinned to one thread (``sims`` sets it).

Run::

    uv run --group sims python -m sims.bench.sumstats_output [--quick] [--repeats R]

(or ``just bench sumstats_output [--quick]``).  Prints a markdown table.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import statistics
import tempfile
import time
from collections.abc import Sequence

import numpy as np

from pylmm import cli
from pylmm.cli import run_record
from pylmm.plink_io import PlinkReader
from pylmm.sumstats import read_sumstats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLED = os.path.join(REPO, "data", "snps.132k.clean.noX")
QUICK_SNPS = 8192


def _slice(base: str, out: str, m: int) -> str:
    """The first ``m`` SNPs of the BED file set ``base`` as ``out``."""
    with open(base + ".fam") as f:
        n = sum(1 for _ in f)
    nb = (n + 3) // 4
    with open(base + ".bed", "rb") as src, open(out + ".bed", "wb") as dst:
        dst.write(src.read(3 + m * nb))
    with open(base + ".bim") as src, open(out + ".bim", "w") as dst:
        dst.writelines(src.readline() for _ in range(m))
    shutil.copy(base + ".fam", out + ".fam")
    return out


def _run(argv: list[str]) -> float:
    begin = time.perf_counter()
    with contextlib.redirect_stderr(io.StringIO()):
        rc = cli.main(["gwas", "-q", *argv])
    if rc:
        raise RuntimeError("pylmm gwas %s failed (exit %d)" % (" ".join(argv), rc))
    return time.perf_counter() - begin


def _legacy(fn: str) -> tuple[list[str], np.ndarray]:
    with open(fn) as f:
        next(f)
        rows = [line.rstrip("\n").split("\t") for line in f]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:]] for r in rows])


def run(quick: bool, repeats: int, workdir: str) -> list[dict[str, object]]:
    base = BUNDLED
    if quick:
        base = _slice(BUNDLED, os.path.join(workdir, "slice"), QUICK_SNPS)
    common = ["--bfile", base, "--kinship", BUNDLED + ".pylmm.kin", "--pheno"]
    common += [BUNDLED + ".fake.phenos"]
    outs = {
        "legacy": (os.path.join(workdir, "legacy.tsv"), ["--output-format", "legacy"]),
        "sumstats .tsv": (os.path.join(workdir, "new.tsv"), []),
        "sumstats .tsv.gz": (os.path.join(workdir, "new.tsv.gz"), []),
    }
    rows: list[dict[str, object]] = []
    for name, (fn, flags) in outs.items():
        walls, scans = [], []
        for _ in range(repeats):
            walls.append(_run([*common, *flags, "-o", fn]))
            with open(run_record.sidecar_path(fn)) as f:
                scans.append(json.load(f)["timings_s"]["scan"])
        rows.append(
            {
                "output": name,
                "wall_s": statistics.median(walls),
                "scan_s": statistics.median(scans),
                "MB": os.path.getsize(fn) / 1e6,
            }
        )
    ids, leg = _legacy(outs["legacy"][0])
    new = read_sumstats(outs["sumstats .tsv.gz"][0])
    with PlinkReader(base, norm_genotype=False) as r:
        G = np.hstack([g for g, _ in r.iter_blocks(4096)])
    sd = np.nanstd(G, axis=0)
    rel = np.abs(new["beta"] - (-leg[:, 0] / (2 * sd))) / np.abs(new["beta"])
    check = {
        "snps": len(ids),
        "ids": bool(list(new["variant_id"]) == ids),
        "t": bool(np.array_equal(new["t_stat"], -leg[:, 2], equal_nan=True)),
        "p": bool(np.array_equal(new["p_value"], leg[:, 3], equal_nan=True)),
        "beta_rel": float(np.nanmax(rel)),
    }
    rows.append({"check": check})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="the first %d SNPs" % QUICK_SNPS)
    ap.add_argument(
        "--repeats", type=int, default=None, help="runs per output (default 3, quick 1)"
    )
    args = ap.parse_args(argv)
    repeats = args.repeats or (1 if args.quick else 3)
    with tempfile.TemporaryDirectory() as d:
        rows = run(args.quick, repeats, d)
    check = rows.pop()["check"]
    assert isinstance(check, dict)
    print("| output | wall s | scan s | file MB |")
    print("|---|---|---|---|")
    for r in rows:
        print("| %s | %.2f | %.2f | %.1f |" % (r["output"], r["wall_s"], r["scan_s"], r["MB"]))
    print()
    print(
        "%d SNPs; variant_id == SNP_ID: %s; t_stat == -F_STAT: %s; p_value == P_VALUE: %s; "
        "max |beta - (-BETA / 2 sd)| / |beta|: %.1e"
        % (check["snps"], check["ids"], check["t"], check["p"], check["beta_rel"])
    )
    ok = check["ids"] and check["t"] and check["p"] and check["beta_rel"] < 1e-12
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
