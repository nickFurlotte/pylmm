"""Benchmark: reading PGEN genotypes (``pylmm.pgen``, pgenlib) vs the same genotypes as BED.

Phase 6.2d.  Measures the read throughput (SNPs per second) of
``iter_blocks(1024)``, with and without standardisation (``norm_genotype``), of

* ``PlinkReader`` on a BED file (pylmm's own decoder, the reference),
* ``PgenSource`` on the PGEN file of the same genotypes, hard calls only
  (``dosage=False``, pgenlib's ``read_range``) and ``dosage=True`` (pgenlib's
  ``read_dosages_range``; the file stores no dosages, so the values are the same),
* ``PgenSource`` on a PGEN file that stores dosages for every call (the same
  genotypes plus seeded noise of up to +-0.2 copies), ``dosage=True``,

for the bundled 1219-mouse BED file ``data/snps.132k.clean.noX`` (98,740 SNPs; with
``--quick`` its first 8,192 SNPs) and a seeded synthetic BED file (an individual
count that is not a multiple of 4, 2% missing calls).  The PGEN files are written
with ``pylmm.pgen.write_pgen`` to a temporary directory.

Outside the timed runs it checks that the hard-call PGEN blocks are bit for bit
the BED blocks (raw and standardised, the float64 bits compared), with both
``dosage`` settings, and that the dosage file reads back the dosages written
(within 2^-15, half the 1/16384-copy precision, on pylmm's 0..1 scale).  Times
are the median of ``--repeats`` runs (default 5 with ``--quick``, 3 otherwise);
each run opens a new reader and reads the whole file, its ``.bim`` / ``.pvar``
included.  BLAS is pinned to one thread by the ``sims`` package (decoding uses no
BLAS); pgenlib may use its own threads.

Needs the optional pgenlib (``uv sync --extra pgen``).  Run::

    uv run --group sims --extra pgen python -m sims.bench.pgen_read [--quick] [--repeats R]

(or ``just bench pgen_read [--quick]`` in an environment with pgenlib).  Prints a
markdown table.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from functools import partial

import numpy as np

import pylmm
from pylmm.genotypes import GenotypeSource
from pylmm.pgen import PgenSource, write_pgen
from pylmm.plink_io import PlinkReader
from sims.bench.bed_blocks import BUNDLED, copy_first_snps, write_synthetic_bed

BLOCK_SIZE = 1024


def _median_time(run: Callable[[], None], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _read_all(make: Callable[[], GenotypeSource]) -> Callable[[], None]:
    def run() -> None:
        for _ in make().iter_blocks(BLOCK_SIZE):
            pass

    return run


def _blocks(src: GenotypeSource) -> np.ndarray:
    return np.hstack([G for G, _ in src.iter_blocks(BLOCK_SIZE)])


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.int64), b.view(np.int64))


def write_pgen_copies(bed: str, tmp: str, seed: int) -> tuple[str, str, np.ndarray]:
    """Write `bed` as a hard-call PGEN and a dosage PGEN in `tmp`; return their prefixes and the dosages."""
    r = PlinkReader(bed, norm_genotype=False)
    G = _blocks(r)
    hc = os.path.join(tmp, os.path.basename(bed) + ".hc")
    write_pgen(hc, G, r.indivs, r.variants)
    rng = np.random.default_rng(seed)
    D = np.clip(G + rng.uniform(-0.1, 0.1, G.shape), 0.0, 1.0)  # +-0.2 copies
    dos = os.path.join(tmp, os.path.basename(bed) + ".dos")
    write_pgen(dos, D, r.indivs, r.variants, dosage=True)
    return hc, dos, D


def bench_file(label: str, bed: str, tmp: str, repeats: int, seed: int) -> list[dict[str, object]]:
    hc, dos, D = write_pgen_copies(bed, tmp, seed)
    n, m = PlinkReader(bed).N, D.shape[1]
    rows: list[dict[str, object]] = []
    for norm in (False, True):
        ref = _blocks(PlinkReader(bed, norm_genotype=norm))
        readers: list[tuple[str, Callable[[], GenotypeSource], bool]] = [
            ("BED (PlinkReader)", partial(PlinkReader, bed, norm_genotype=norm), True),
            (
                "PGEN hard calls, dosage=False",
                partial(PgenSource, hc, norm_genotype=norm, dosage=False),
                _same_bits(_blocks(PgenSource(hc, norm_genotype=norm, dosage=False)), ref),
            ),
            (
                "PGEN hard calls, dosage=True",
                partial(PgenSource, hc, norm_genotype=norm),
                _same_bits(_blocks(PgenSource(hc, norm_genotype=norm)), ref),
            ),
        ]
        got = _blocks(PgenSource(dos, norm_genotype=False))
        dos_ok = bool(
            np.array_equal(np.isnan(got), np.isnan(D)) and np.nanmax(np.abs(got - D)) <= 2.0**-15
        )
        readers.append(
            ("PGEN dosages, dosage=True", partial(PgenSource, dos, norm_genotype=norm), dos_ok)
        )
        base = None
        for name, make, ok in readers:
            sec = _median_time(_read_all(make), repeats)
            base = sec if base is None else base
            rows.append(
                {
                    "data": label,
                    "n": n,
                    "snps": m,
                    "norm": norm,
                    "method": name,
                    "sec": sec,
                    "vs_bed": base / sec,
                    "ok": ok,
                }
            )
    return rows


def file_sizes(bed: str, tmp: str) -> str:
    stem = os.path.join(tmp, os.path.basename(bed))
    sizes = [
        ("BED", os.path.getsize(bed + ".bed")),
        ("PGEN hard calls", os.path.getsize(stem + ".hc.pgen")),
        ("PGEN dosages", os.path.getsize(stem + ".dos.pgen")),
    ]
    return ", ".join("%s %s bytes" % (k, format(v, ",")) for k, v in sizes)


def format_table(rows: list[dict[str, object]]) -> str:
    out = [
        "| data | n | SNPs | norm | reader | median s | SNPs/s | vs BED | check |",
        "|---|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        sec = float(r["sec"])  # type: ignore[arg-type]
        out.append(
            "| %s | %d | %d | %s | %s | %.3f | %s | %.2fx | %s |"
            % (
                r["data"],
                r["n"],
                r["snps"],
                "on" if r["norm"] else "off",
                r["method"],
                sec,
                format(round(int(r["snps"]) / sec), ","),  # type: ignore[call-overload]
                r["vs_bed"],
                "ok" if r["ok"] else "FAILED",
            )
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sims.bench.pgen_read", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--quick", action="store_true", help="fewer SNPs (well under a minute)")
    ap.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="timed runs per cell, median reported (default 5 with --quick, else 3)",
    )
    ap.add_argument("--seed", type=int, default=20261021, help="seed of the synthetic data")
    args = ap.parse_args(argv)
    if args.repeats is None:
        args.repeats = 5 if args.quick else 3
    if args.quick:
        bundled_m, syn_n, syn_m = 8192, 1001, 8192
    else:
        bundled_m, syn_n, syn_m = None, 5003, 40960

    rows: list[dict[str, object]] = []
    sizes = []
    with tempfile.TemporaryDirectory() as tmp:
        syn = os.path.join(tmp, "synthetic")
        write_synthetic_bed(syn, syn_n, syn_m, missing=0.02, seed=args.seed)
        rows += bench_file("synthetic (2% missing)", syn, tmp, args.repeats, args.seed)
        sizes.append("synthetic: " + file_sizes(syn, tmp))
        if os.path.isfile(BUNDLED + ".bed"):
            bundled = BUNDLED
            if bundled_m is not None:
                bundled = os.path.join(tmp, "bundled_first")
                copy_first_snps(BUNDLED, bundled, bundled_m)
            rows += bench_file("bundled 1219 mice", bundled, tmp, args.repeats, args.seed)
            sizes.append("bundled: " + file_sizes(bundled, tmp))
        else:
            print("(bundled data %s not found; skipped)" % BUNDLED, file=sys.stderr)

    import pgenlib

    print(
        "pylmm %s, pgenlib %s, numpy %s, Python %s, %s %s, %d CPUs, block size %d, "
        "seed %d, median of %d runs%s\n"
        % (
            pylmm.__version__,
            getattr(pgenlib, "__version__", "?"),
            np.__version__,
            platform.python_version(),
            platform.system(),
            platform.machine(),
            os.cpu_count() or 0,
            BLOCK_SIZE,
            args.seed,
            args.repeats,
            ", --quick" if args.quick else "",
        )
    )
    print(format_table(rows))
    print("\nfile sizes: " + "; ".join(sizes))
    ok = all(r["ok"] for r in rows)
    print(
        "\nhard-call PGEN blocks bit-identical to BED, dosages as written: %s"
        % ("yes" if ok else "NO")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
