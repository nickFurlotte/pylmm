"""Benchmark: BED decoding SNP by SNP (``PlinkReader.iter_snps``) vs in blocks (``iter_blocks``).

Phase 6.2.  Measures the decode throughput (SNPs per second) of the per-SNP
iterator and of ``iter_blocks`` at a few block sizes, with and without
standardisation (``norm_genotype``), on

* the bundled 1219-mouse BED file ``data/snps.132k.clean.noX`` (98,740 SNPs;
  the first 8,192 with ``--quick``), and
* a seeded synthetic BED file (written to a temporary directory) with an
  individual count that is not a multiple of 4 and 2% missing calls.

It also checks, outside the timed runs, that every block column is bit for bit
the per-SNP vector (``numpy.array_equal(..., equal_nan=True)``), and that the
block variants are the ``.bim`` lines.  Times are the median of ``--repeats``
runs, after the ``.bim`` file has been read once (``PlinkReader.variants``,
timed separately).  BLAS is pinned to one thread: ``python -m
sims.bench.bed_blocks`` imports the ``sims`` package, which sets the thread
variables, before numpy (decoding uses no BLAS anyway).

Run::

    uv run --group sims python -m sims.bench.bed_blocks [--quick] [--repeats R] [--seed S]

(or ``just bench bed_blocks [--quick]``).  Prints a markdown table.
"""

from __future__ import annotations

import argparse
import itertools
import os
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterator

import numpy as np

import pylmm
from pylmm.plink_io import PlinkReader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLED = os.path.join(REPO, "data", "snps.132k.clean.noX")
BLOCK_SIZES = (256, 1024, 4096)


def write_synthetic_bed(
    prefix: str, n: int, m: int, missing: float = 0.02, seed: int = 0
) -> np.ndarray:
    """Write a random SNP-major BED/BIM/FAM set; return its raw genotypes (fraction of A2).

    Allele frequencies are uniform on (0.05, 0.5), genotypes Hardy-Weinberg,
    each call missing with probability `missing`.  The ``.bim`` spreads the
    SNPs over three chromosomes with increasing positions.  The returned
    ``(n, m)`` array holds 0, 0.5, 1 or nan, as ``PlinkReader(...,
    norm_genotype=False)`` reads them.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.5, m)
    counts = rng.binomial(2, p, size=(n, m))  # copies of A2
    miss = rng.random((n, m)) < missing
    # 2-bit codes: 0 copies of A2 -> 00, 1 -> 10, 2 -> 11, missing -> 01
    codes = np.array([0, 2, 3], dtype=np.uint8)[counts]
    codes[miss] = 1
    nb = (n + 3) // 4
    padded = np.zeros((nb * 4, m), dtype=np.uint8)
    padded[:n] = codes
    q = padded.reshape(nb, 4, m)
    packed = q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)
    with open(prefix + ".bed", "wb") as f:
        f.write(b"\x6c\x1b\x01")
        f.write(np.ascontiguousarray(packed.T).tobytes())
    chrom = np.repeat(["1", "2", "3"], [m - 2 * (m // 3), m // 3, m // 3])
    with open(prefix + ".bim", "w") as f:
        for j in range(m):
            f.write("%s\tsnp%d\t%.3f\t%d\tA\tG\n" % (chrom[j], j + 1, j / 1000, 1000 * (j + 1)))
    with open(prefix + ".fam", "w") as f:
        for i in range(n):
            f.write("F%d I%d 0 0 0 -9\n" % (i + 1, i + 1))
    G = counts / 2.0
    G[miss] = np.nan
    return G


def _consume(it: Iterator[object], limit: int | None) -> None:
    for _ in itertools.islice(it, limit):
        pass


def _median_time(run: Callable[[], None], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def check_identity(prefix: str, m: int, norm: bool, block_size: int) -> bool:
    """Every block column of the first `m` SNPs equals the per-SNP vector bit for bit."""
    snps = PlinkReader(prefix, norm_genotype=norm).iter_snps()
    blocks = PlinkReader(prefix, norm_genotype=norm)
    names = list(blocks.variants.snp[:m])
    k = 0
    for G, v in blocks.iter_blocks(block_size):
        for j in range(G.shape[1]):
            if k == m:
                return True
            g, name = next(snps)
            if not (np.array_equal(g, G[:, j], equal_nan=True) and name == v.snp[j] == names[k]):
                return False
            k += 1
    return k == m


def time_variants(prefix: str) -> tuple[float, int]:
    """Seconds to read the whole ``.bim`` file (``PlinkReader.variants``), and its SNP count."""
    reader = PlinkReader(prefix)
    t0 = time.perf_counter()
    m = len(reader.variants)
    return time.perf_counter() - t0, m


def bench_file(label: str, prefix: str, m: int, repeats: int) -> list[dict[str, object]]:
    """Time per-SNP vs block decoding of the first `m` SNPs.

    `m` should be a multiple of every block size (or all the SNPs), so that
    every method decodes the same SNPs.  Each row also says whether the block
    columns were bit-identical to the per-SNP vectors (checked for these and
    two more block sizes, 1 and 777, outside the timing).
    """
    rows: list[dict[str, object]] = []
    n = PlinkReader(prefix).num_samples
    for norm in (False, True):
        reader = PlinkReader(prefix, norm_genotype=norm)
        _ = reader.variants  # read the .bim once, outside the timing
        same = all(check_identity(prefix, m, norm, b) for b in (*BLOCK_SIZES, 1, 777))
        common = {"data": label, "n": n, "snps": m, "norm": norm, "identical": same}
        base = _median_time(lambda r=reader: _consume(r.iter_snps(), m), repeats)
        rows.append({**common, "method": "iter_snps", "sec": base, "speedup": 1.0})
        for b in BLOCK_SIZES:
            nblocks = -(-m // b)
            sec = _median_time(
                lambda r=reader, b=b, k=nblocks: _consume(r.iter_blocks(b), k), repeats
            )
            rows.append(
                {**common, "method": "iter_blocks(%d)" % b, "sec": sec, "speedup": base / sec}
            )
    return rows


def format_table(rows: list[dict[str, object]]) -> str:
    out = [
        "| data | n | SNPs | norm | method | median s | SNPs/s | speedup | bit-identical |",
        "|---|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        sec = float(r["sec"])  # type: ignore[arg-type]
        out.append(
            "| %s | %d | %d | %s | %s | %.3f | %s | %.1fx | %s |"
            % (
                r["data"],
                r["n"],
                r["snps"],
                "on" if r["norm"] else "off",
                r["method"],
                sec,
                format(round(int(r["snps"]) / sec), ","),  # type: ignore[call-overload]
                r["speedup"],
                "yes" if r["identical"] else "NO",
            )
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sims.bench.bed_blocks", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--quick", action="store_true", help="fewer SNPs (well under a minute)")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per cell (median reported)")
    ap.add_argument("--seed", type=int, default=20261020, help="seed of the synthetic BED")
    args = ap.parse_args(argv)

    if args.quick:
        bundled_m, syn_n, syn_m = 8192, 1001, 8192
    else:
        bundled_m, syn_n, syn_m = None, 5003, 40960

    rows: list[dict[str, object]] = []
    notes = []
    with tempfile.TemporaryDirectory() as tmp:
        syn = os.path.join(tmp, "synthetic")
        write_synthetic_bed(syn, syn_n, syn_m, missing=0.02, seed=args.seed)
        rows += bench_file("synthetic (2% missing)", syn, syn_m, args.repeats)
        sec, m_all = time_variants(syn)
        notes.append("synthetic .bim (%d SNPs) read in %.3f s" % (m_all, sec))
    if os.path.isfile(BUNDLED + ".bed"):
        sec, m_all = time_variants(BUNDLED)
        notes.append("bundled .bim (%d SNPs) read in %.3f s" % (m_all, sec))
        rows += bench_file("bundled 1219 mice", BUNDLED, bundled_m or m_all, args.repeats)
    else:
        print("(bundled data %s not found; skipped)" % BUNDLED, file=sys.stderr)

    print(
        "pylmm %s, numpy %s, Python %s, %s %s, %d CPUs, OPENBLAS_NUM_THREADS=%s, "
        "seed %d, median of %d runs%s\n"
        % (
            pylmm.__version__,
            np.__version__,
            platform.python_version(),
            platform.system(),
            platform.machine(),
            os.cpu_count() or 0,
            os.environ.get("OPENBLAS_NUM_THREADS"),
            args.seed,
            args.repeats,
            ", --quick" if args.quick else "",
        )
    )
    print(format_table(rows))
    print(
        "\n" + "; ".join(notes) + " (PlinkReader.variants, once per reader; not in the times above)"
    )
    ok = all(r["identical"] for r in rows)
    print("\nblock columns bit-identical to iter_snps: %s" % ("yes" if ok else "NO"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
