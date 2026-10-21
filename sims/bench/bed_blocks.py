"""Benchmark: BED decoding SNP by SNP (``PlinkReader.iter_snps``) vs in blocks (``iter_blocks``).

Phase 6.2 (6.1c: standardisation of whole blocks).  Measures the decode
throughput (SNPs per second) of the per-SNP iterator and of ``iter_blocks`` at
a few block sizes, with and without standardisation (``norm_genotype``), on

* the bundled 1219-mouse BED file ``data/snps.132k.clean.noX`` (98,740 SNPs;
  with ``--quick``, a copy of its first 8,192 SNPs written to a temporary
  directory), and
* a seeded synthetic BED file (written to a temporary directory) with an
  individual count that is not a multiple of 4 and 2% missing calls.

With standardisation, ``iter_blocks`` is timed twice: as it is (a block
standardised at once, ``pylmm.genotypes._normalize_rows``) and with the
per-SNP standardisation it used before 6.1c (``normalize_genotype`` called on
each column in a Python loop, the fallback of ``_normalize_rows``); the
"vs per-SNP std" column is the speedup of the first over the second.

It also checks, outside the timed runs, that every block column is bit for bit
the per-SNP vector (the raw float64 bits compared, so nan payloads and the sign
of zero count too), and that the block variants are the ``.bim`` lines.  Times are the median of ``--repeats``
runs (default 5 with ``--quick``, 3 otherwise).  Each timed run opens a new
``PlinkReader`` and reads the whole file, so both methods pay for reading the
``.fam`` and ``.bim`` (``iter_snps`` a ``.bim`` line per SNP, ``iter_blocks``
the whole ``.bim`` up front, for ``PlinkReader.variants``).  BLAS is pinned
to one thread: ``python -m
sims.bench.bed_blocks`` imports the ``sims`` package, which sets the thread
variables, before numpy (decoding uses no BLAS anyway).

Run::

    uv run --group sims python -m sims.bench.bed_blocks [--quick] [--repeats R] [--seed S]

(or ``just bench bed_blocks [--quick]``).  Prints a markdown table.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterator

import numpy as np

import pylmm
from pylmm import genotypes
from pylmm.plink_io import PlinkReader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLED = os.path.join(REPO, "data", "snps.132k.clean.noX")
BLOCK_SIZES = (256, 1024, 4096)


def write_synthetic_bed(
    prefix: str, n: int, m: int, missing: float = 0.02, seed: int = 0
) -> np.ndarray:
    """Write a random SNP-major BED/BIM/FAM set; return its raw genotypes (fraction of A2).

    The same as ``tests/synthetic_bed.py`` (tests/sims/test_bench_bed_blocks.py
    checks that they write identical files).

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


def _consume(it: Iterator[object]) -> None:
    for _ in it:
        pass


def _median_time(run: Callable[[], None], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def copy_first_snps(src: str, dst: str, m: int) -> None:
    """Write the first `m` SNPs of the BED file set `src` as `dst` (``.bed/.bim/.fam``)."""
    n = PlinkReader(src).N
    with open(src + ".bed", "rb") as f:
        data = f.read(3 + m * ((n + 3) // 4))
    with open(dst + ".bed", "wb") as f:
        f.write(data)
    with open(src + ".bim") as f, open(dst + ".bim", "w") as g:
        g.writelines(itertools.islice(f, m))
    shutil.copyfile(src + ".fam", dst + ".fam")


class _AlwaysPerSnp(dict):  # type: ignore[type-arg]
    def get(self, key: object, default: object = None) -> bool:
        return False


@contextlib.contextmanager
def per_snp_standardisation() -> Iterator[None]:
    """Within the block, ``iter_blocks`` standardises SNP by SNP (the code before 6.1c).

    ``_normalize_rows`` then takes its fallback, ``R[j] = normalize_genotype(R[j])``
    for each SNP ``j`` of the block, as ``iter_blocks`` did before.
    """
    saved = genotypes._ROWS_MATCH
    genotypes._ROWS_MATCH = _AlwaysPerSnp()
    try:
        yield
    finally:
        genotypes._ROWS_MATCH = saved


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(
        np.ascontiguousarray(a).view(np.int64), np.ascontiguousarray(b).view(np.int64)
    )


def check_identity(prefix: str, norm: bool, block_size: int) -> bool:
    """Every block column equals the per-SNP vector bit for bit, with the same SNP name."""
    snps = PlinkReader(prefix, norm_genotype=norm).iter_snps()
    blocks = PlinkReader(prefix, norm_genotype=norm)
    names = blocks.variants.snp.tolist()
    k = 0
    for G, v in blocks.iter_blocks(block_size):
        for j in range(G.shape[1]):
            g, name = next(snps)
            if not (_same_bits(g, G[:, j]) and name == v.snp[j] == names[k]):
                return False
            k += 1
    return k == len(names) and next(snps, None) is None


def bench_file(label: str, prefix: str, repeats: int) -> list[dict[str, object]]:
    """Time per-SNP vs block reading of all the SNPs of `prefix`, norm off and on.

    Each timed run opens a new reader (so the ``.bim`` is read in both).  Each
    row also says whether the block columns were bit-identical to the per-SNP
    vectors (checked outside the timing for these and two more block sizes, 1
    and 777).
    """
    rows: list[dict[str, object]] = []
    n = PlinkReader(prefix).num_samples
    m = len(PlinkReader(prefix).variants)
    for norm in (False, True):
        same = all(check_identity(prefix, norm, b) for b in (*BLOCK_SIZES, 1, 777))
        common = {"data": label, "n": n, "snps": m, "norm": norm, "identical": same}
        base = _median_time(
            lambda norm=norm: _consume(PlinkReader(prefix, norm_genotype=norm).iter_snps()),
            repeats,
        )
        rows.append({**common, "method": "iter_snps", "sec": base, "speedup": 1.0})
        for b in BLOCK_SIZES:
            sec = _median_time(
                lambda b=b, norm=norm: _consume(
                    PlinkReader(prefix, norm_genotype=norm).iter_blocks(b)
                ),
                repeats,
            )
            row = {**common, "method": "iter_blocks(%d)" % b, "sec": sec, "speedup": base / sec}
            if norm:
                with per_snp_standardisation():
                    before = _median_time(
                        lambda b=b: _consume(PlinkReader(prefix).iter_blocks(b)), repeats
                    )
                rows.append(
                    {
                        **common,
                        "method": "iter_blocks(%d), per-SNP std" % b,
                        "sec": before,
                        "speedup": base / before,
                    }
                )
                row["std_speedup"] = before / sec
            rows.append(row)
    return rows


def format_table(rows: list[dict[str, object]]) -> str:
    out = [
        "| data | n | SNPs | norm | method | median s | SNPs/s | speedup | vs per-SNP std "
        "| bit-identical |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        sec = float(r["sec"])  # type: ignore[arg-type]
        out.append(
            "| %s | %d | %d | %s | %s | %.3f | %s | %.1fx | %s | %s |"
            % (
                r["data"],
                r["n"],
                r["snps"],
                "on" if r["norm"] else "off",
                r["method"],
                sec,
                format(round(int(r["snps"]) / sec), ","),  # type: ignore[call-overload]
                r["speedup"],
                "%.1fx" % r["std_speedup"] if "std_speedup" in r else "",  # type: ignore[str-format]
                "yes" if r["identical"] else "NO",
            )
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sims.bench.bed_blocks", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--quick", action="store_true", help="fewer SNPs (well under a minute)")
    ap.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="timed runs per cell, median reported (default 5 with --quick, else 3)",
    )
    ap.add_argument("--seed", type=int, default=20261020, help="seed of the synthetic BED")
    args = ap.parse_args(argv)
    if args.repeats is None:
        args.repeats = 5 if args.quick else 3

    if args.quick:
        bundled_m, syn_n, syn_m = 8192, 1001, 8192
    else:
        bundled_m, syn_n, syn_m = None, 5003, 40960

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        syn = os.path.join(tmp, "synthetic")
        write_synthetic_bed(syn, syn_n, syn_m, missing=0.02, seed=args.seed)
        rows += bench_file("synthetic (2% missing)", syn, args.repeats)
        if os.path.isfile(BUNDLED + ".bed"):
            bundled = BUNDLED
            if bundled_m is not None:
                bundled = os.path.join(tmp, "bundled_first")
                copy_first_snps(BUNDLED, bundled, bundled_m)
            rows += bench_file("bundled 1219 mice", bundled, args.repeats)
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
    ok = all(r["identical"] for r in rows)
    print("\nblock columns bit-identical to iter_snps: %s" % ("yes" if ok else "NO"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
