#!/usr/bin/env python3
"""Build data/hmdp.liver.snps.map.tsv: a validated marker map for the bundled HMDP EMMA matrix.

``data/hmdp.liver.snps`` (132,285 rows x 94 strains, coded 0/1, ``NA`` missing) has no SNP
names, positions or alleles.  Its rows are in the order of the Broad Institute's merged mouse
HapMap / WT genotype file (NCBI build 37 / mm9)::

    https://data.broadinstitute.org/mouse/hapmap/Merged_Hapmap_WT_Genotypes.txt.gz

This script downloads that file (or uses ``--broad FILE``), checks its SHA-256, and walks both
files row by row.  For each row it compares the EMMA 0/1 codes with the Broad base calls of the
strains whose names appear in both files (27 classic inbred strains), and writes

    row      1-based row of data/hmdp.liver.snps
    chr      chromosome (1-19, X), from Broad's ``build37_loc`` (``mm37-<chr>-<bp>``)
    bp_b37   base-pair position, NCBI build 37 (mm9)
    allele0  the base coded 0 in the EMMA matrix, "." if no shared strain carries it
    allele1  the base coded 1 (the allele whose effect pylmm reports for EMMA input), or "."
    status   concordant: the shared strains carry both bases and every 0 maps to one base and
                 every 1 to the other (the row identity is checked);
             uninformative: the shared strains show one base only (consistent, not a check);
             discordant: the codes and bases disagree (the row join would be wrong)

The script exits 1 if the row counts differ or any row is discordant.  Published result
(2026-10-19): 132,285 rows, 122,242 concordant, 10,043 uninformative, 0 discordant.

Usage (from the repository root; downloads ~20 MB unless --broad is given)::

    uv run python scripts/build_hmdp_marker_map.py
    uv run python scripts/build_hmdp_marker_map.py --broad Merged_Hapmap_WT_Genotypes.txt.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import sys
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator

BROAD_URL = "https://data.broadinstitute.org/mouse/hapmap/Merged_Hapmap_WT_Genotypes.txt.gz"
# Last-Modified 2009-02-20, 20,446,478 bytes; checked 2026-10-19.
BROAD_SHA256 = "dd4f80e9e737635c02a9f2f2e843ee3666a91d9b4ed4903ce453128bb126796f"
COLUMNS = ("row", "chr", "bp_b37", "allele0", "allele1", "status")
BASES = frozenset("ACGT")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: str) -> None:
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)


def read_strains(path: str) -> list[str]:
    """The strain names of ``hmdp.liver.strains`` (one per line, double quotes stripped)."""
    with open(path) as f:
        return [line.strip().strip('"') for line in f if line.strip()]


def classify(codes: Iterable[float | None], calls: Iterable[str]) -> tuple[str, str, str]:
    """``(allele0, allele1, status)`` of one row from paired EMMA codes and Broad calls.

    ``codes`` are the EMMA values (0, 1, or None for NA) and ``calls`` the Broad base calls
    (``A``/``C``/``G``/``T``; anything else is no call) of the same strains, in the same order.

    >>> classify([0, 0, 1, None], ["A", "A", "T", "T"])
    ('A', 'T', 'concordant')
    >>> classify([0, 0, None], ["C", "C", "G"])
    ('C', '.', 'uninformative')
    >>> classify([0, 1, 1], ["A", "A", "T"])[2]
    'discordant'
    """
    zeros: set[str] = set()
    ones: set[str] = set()
    for g, c in zip(codes, calls):
        if g is None or c not in BASES:
            continue
        if g == 0:
            zeros.add(c)
        elif g == 1:
            ones.add(c)
        else:
            raise ValueError("EMMA code %r is not 0, 1 or NA" % (g,))
    consistent = len(zeros) <= 1 and len(ones) <= 1 and not (zeros & ones)
    if not consistent:
        status = "discordant"
    elif zeros and ones:
        status = "concordant"
    else:
        status = "uninformative"
    a0 = next(iter(zeros)) if len(zeros) == 1 else "."
    a1 = next(iter(ones)) if len(ones) == 1 else "."
    return a0, a1, status


def iter_emma(lines: Iterable[str]) -> Iterator[list[float | None]]:
    """EMMA genotype rows: one list of codes (None for NA) per non-blank line."""
    for line in lines:
        v = line.split()
        if v:
            yield [None if x == "NA" else float(x) for x in v]


def build_map(
    emma_lines: Iterable[str], strains: list[str], broad_lines: Iterable[str]
) -> tuple[list[tuple[int, str, int, str, str, str]], Counter[str], list[str]]:
    """Join the EMMA rows to the Broad rows in order and classify each one.

    Returns the map rows, the status counts and the shared strain names.  Raises ValueError if
    the files have different numbers of rows.
    """
    broad = iter(broad_lines)
    names = next(broad).rstrip("\n").split("\t")[2:]
    next(broad)  # the second header line: build37_loc, oxford_name, CI/WD strain classes
    shared = sorted(set(names) & set(strains))
    if not shared:
        raise ValueError("no strain names in common between the EMMA strain list and Broad")
    e_idx = [strains.index(s) for s in shared]  # first column of a repeated name (identical)
    b_idx = [names.index(s) + 2 for s in shared]
    rows = []
    status: Counter[str] = Counter()
    emma = iter_emma(emma_lines)
    for j, (codes, line) in enumerate(zip(emma, broad), start=1):
        cells = line.rstrip("\n").split("\t")
        if len(codes) != len(strains):
            raise ValueError(
                "EMMA row %d has %d values, expected %d" % (j, len(codes), len(strains))
            )
        _, chrom, bp = cells[0].split("-", 2)
        calls = [cells[i].split(":", 1)[0].upper() for i in b_idx]
        a0, a1, st = classify([codes[i] for i in e_idx], calls)
        status[st] += 1
        rows.append((j, chrom, int(bp), a0, a1, st))
    extra_emma = next(emma, None) is not None
    extra_broad = next(broad, None) is not None
    if extra_emma or extra_broad:
        raise ValueError(
            "row counts differ after %d rows: the %s file has more"
            % (len(rows), "EMMA" if extra_emma else "Broad")
        )
    return rows, status, shared


def write_map(path: str, rows: Iterable[tuple[int, str, int, str, str, str]]) -> None:
    with open(path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("%d\t%s\t%d\t%s\t%s\t%s\n" % r)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--broad", help="a local copy of Merged_Hapmap_WT_Genotypes.txt.gz")
    p.add_argument("--emma", default="data/hmdp.liver.snps")
    p.add_argument("--strains", default="data/hmdp.liver.strains")
    p.add_argument("--out", default="data/hmdp.liver.snps.map.tsv")
    p.add_argument(
        "--no-checksum", action="store_true", help="skip the SHA-256 check of the Broad file"
    )
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        broad = args.broad
        if broad is None:
            broad = os.path.join(tmp, os.path.basename(BROAD_URL))
            print("downloading %s" % BROAD_URL, file=sys.stderr)
            download(BROAD_URL, broad)
        if not args.no_checksum:
            got = sha256(broad)
            if got != BROAD_SHA256:
                print(
                    "error: %s has SHA-256 %s, expected %s (a different release?); "
                    "rerun with --no-checksum to build a map from it anyway"
                    % (broad, got, BROAD_SHA256),
                    file=sys.stderr,
                )
                return 1
        strains = read_strains(args.strains)
        with open(args.emma) as e, gzip.open(broad, "rt") as b:
            rows, status, shared = build_map(e, strains, b)

    print("rows: %d; shared strains: %d" % (len(rows), len(shared)), file=sys.stderr)
    print("status: %s" % dict(sorted(status.items())), file=sys.stderr)
    if status["discordant"]:
        print("error: %d discordant rows; not writing %s" % (status["discordant"], args.out))
        return 1
    write_map(args.out, rows)
    print("wrote %s" % args.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
