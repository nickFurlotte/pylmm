# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""``pylmm kinship``: compute a realized-relationship kinship matrix.

``K = W W' / m`` over the ``m`` SNPs of the genotype input, with ``W`` the
standardised genotypes (mean-imputed); monomorphic SNPs and SNPs with no
called genotype are skipped, but still counted in ``m``.  The SNPs are read
in blocks of ``--block-size`` (`pylmm.plink_io.PlinkReader.iter_blocks`), and
``W W'`` is summed over blocks of ``--block-size`` kept SNPs.

Next to the kinship ``FILE`` (``-o``) it writes ``FILE.id``, the (FID, IID) of each row
(one ``FID IID`` per line), which ``pylmm gwas --kinship FILE`` uses to align
the matrix to its genotype file by ID.  EMMA input has no IDs, so no ID file
is written for it (and a stale one is removed).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from pylmm import plink_io, samples
from pylmm.cli._common import (
    UsageError,
    add_logging_options,
    check_log_file,
    check_writable,
    genotype_source,
    log,
    positive_int,
)
from pylmm.samples import InputError

FloatArray = NDArray[np.float64]

_REPORT_EVERY = 10000
"""Log the progress of `compute_kinship` at INFO every this many SNPs."""

_DESCRIPTION = """\
Compute a realized-relationship kinship matrix K = W W'/m from genotypes,
for `pylmm gwas --kinship`.  Writes K as whitespace-separated text to FILE
(-o) and the sample IDs of its rows to FILE.id (not for EMMA input, which
has no IDs)."""

_EPILOG = """\
examples:
  pylmm kinship --bfile data/snps.132k.clean.noX -o snps.kin
  pylmm kinship --bfile study -o study.kin --eigen-npz study.eig.npz
  pylmm kinship --emma-geno snps.emma -o K.txt"""


def add_parser(commands: Any) -> argparse.ArgumentParser:
    """Add the ``kinship`` subcommand to the ``pylmm`` parser.

    Parameters
    ----------
    commands : argparse._SubParsersAction
        The ``pylmm`` subparsers (``parser.add_subparsers()``).

    Returns
    -------
    argparse.ArgumentParser
        The ``kinship`` parser.
    """
    p: argparse.ArgumentParser = commands.add_parser(
        "kinship",
        help="compute a kinship matrix from genotypes",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(run=run, check=check_args, parser=p)

    geno = p.add_argument_group("genotypes (exactly one)")
    source = geno.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--bfile", metavar="PREFIX", help="PLINK binary genotypes PREFIX.bed/.bim/.fam"
    )
    source.add_argument(
        "--tfile", metavar="PREFIX", help="PLINK transposed text genotypes PREFIX.tped/.tfam"
    )
    source.add_argument(
        "--emma-geno",
        metavar="FILE",
        help="EMMA genotype matrix: one SNP per row, one individual per column, no IDs",
    )
    geno.add_argument(
        "--emma-num-snps",
        metavar="M",
        type=positive_int,
        help="optional check: the number of SNPs (non-blank rows) of the --emma-geno file; "
        "it is counted from the file, and a different M is an error",
    )

    output = p.add_argument_group("output")
    output.add_argument(
        "-o",
        "--out",
        metavar="FILE",
        required=True,
        help="(required) write the kinship to FILE and its sample IDs to FILE.id (both "
        "overwritten)",
    )
    output.add_argument(
        "--eigen-text",
        action="store_true",
        help="also write the eigendecomposition as text, FILE.kva and FILE.kve (for "
        "`pylmm gwas --eigen FILE`)",
    )
    output.add_argument(
        "--eigen-npz",
        metavar="NPZ",
        help="also write the eigendecomposition to NPZ, arrays Kva and Kve (for "
        "`pylmm gwas --eigen NPZ`; binary, exact and faster to read than --eigen-text)",
    )
    performance = p.add_argument_group("performance")
    performance.add_argument(
        "--block-size",
        metavar="N",
        type=positive_int,
        default=1000,
        help="the number of SNPs read, and summed into K, at once: a few arrays of n x N floats in "
        "memory (default: 1000)",
    )
    add_logging_options(p)
    return p


def check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Check the option combinations argparse cannot express (exits 2 on an error).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ``kinship`` parser, for `argparse.ArgumentParser.error`.
    args : argparse.Namespace
        The parsed options.
    """
    if args.emma_num_snps is not None and not args.emma_geno:
        parser.error("--emma-num-snps applies only to --emma-geno")
    check_log_file(parser, args)


def open_genotypes(args: argparse.Namespace) -> plink_io.PlinkReader:
    """Open the genotype input (``--bfile``, ``--tfile`` or ``--emma-geno``).

    Raises
    ------
    UsageError
        On an invalid genotype file.
    """
    flag, base, kind = genotype_source(args)
    try:
        return plink_io.PlinkReader(base, type=kind)
    except OSError as e:  # e.g. a missing .fam: name the option
        raise UsageError("%s %s: %s" % (flag, base, e)) from e
    except ValueError as e:  # the message names the file
        raise UsageError(str(e)) from e


def count_emma_snps(fn: str) -> int:
    """The number of SNPs of an EMMA genotype file: its non-blank lines."""
    with open(fn) as f:
        return sum(1 for line in f if line.strip())


def iter_snp_vectors(reader: plink_io.PlinkReader, block_size: int) -> Iterator[FloatArray]:
    """The genotype vectors of every SNP of ``reader``, read ``block_size`` SNPs at a time.

    Read with `pylmm.plink_io.PlinkReader.iter_blocks` (a BED file is decoded
    a block at a time); each vector is bit for bit the one `iter_snps`
    returns.  If the SNP metadata `iter_blocks` needs cannot be parsed (e.g. a
    ``.bim`` line with fewer than 6 fields, which `iter_snps` reads), the
    SNPs are read one at a time instead, with a warning.

    Each vector is a copy, and a block is released before the next one is
    read, so about one block (and its decoding) is held at a time.
    """
    try:
        blocks = reader.iter_blocks(block_size)
    except InputError as e:
        log.warning("%s; reading the SNPs one at a time", e)
        for snp, _ in reader.iter_snps():
            yield snp
        return
    while True:
        item = next(blocks, None)
        if item is None:
            return
        G = item[0]
        del item
        for j in range(G.shape[1]):
            yield G[:, j].copy()
        del G  # before the next block is decoded


def compute_kinship(reader: plink_io.PlinkReader, block_size: int) -> FloatArray:
    """``K = W W' / m`` over all ``m`` SNPs of ``reader``, ``block_size`` SNPs at a time.

    SNPs that are monomorphic or have a missing value (after the reader's
    imputation: no called genotype at all) are skipped but still counted in
    ``m`` (bug 8 in docs/phase4-simulations.md; unchanged here).  ``W W'`` is
    accumulated over blocks of ``block_size`` kept SNPs (`iter_snp_vectors`
    reads blocks of ``block_size`` SNPs), so K does not depend on how the SNPs
    are read.

    Parameters
    ----------
    reader : pylmm.plink_io.PlinkReader
        The genotype reader, with standardised SNPs (the default).
    block_size : int
        The number of SNPs in memory at once.

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` kinship.

    Raises
    ------
    UsageError
        If the genotype input has no SNP.
    """
    n = len(reader.indivs)
    W = np.ones((n, block_size)) * np.nan
    i = 0  # SNPs read
    j = 0  # SNPs in W
    flushed = 0  # SNPs read when W was last added to K
    K: FloatArray | None = None
    n_all_missing = 0
    next_report = _REPORT_EVERY

    def add(K: FloatArray | None, W: FloatArray, last: bool) -> FloatArray:
        nonlocal next_report
        # Progress at INFO every 10000 SNPs (and at the end), else at DEBUG.
        if i >= next_report or last:
            log.info("Processing first %d SNPs", i)
            next_report = (i // _REPORT_EVERY + 1) * _REPORT_EVERY
        else:
            log.debug("Processing first %d SNPs", i)
        ww: FloatArray = np.dot(W, W.T)
        return ww if K is None else K + ww

    snps = iter_snp_vectors(reader, block_size)
    snp = next(snps, None)
    while snp is not None:
        i += 1
        following = next(snps, None)  # None after the last SNP
        # Skip monomorphic SNPs and SNPs with no called genotype (bug 11:
        # an all-missing SNP used to crash, or to make K all nan).
        if not np.isfinite(snp).all() or snp.var() == 0:
            if np.isnan(snp).all():
                n_all_missing += 1
        else:
            W[:, j] = snp
            j += 1
            if j == block_size:
                K = add(K, W, following is None)
                j = 0
                flushed = i
        snp = following
    if i > flushed:  # the SNPs after the last full block (perhaps all skipped)
        K = add(K, W[:, list(range(0, j))], True)
    if n_all_missing:
        log.warning("skipped %d SNPs with no called genotype", n_all_missing)
    if K is None:
        raise UsageError("No SNP with a non-constant genotype was read; cannot compute a kinship.")
    result: FloatArray = K / float(i)
    return result


def write_kinship(K: FloatArray, reader: plink_io.PlinkReader, out_file: str) -> None:
    """Write K to ``out_file`` and its row IDs to ``out_file.id``.

    The ID file (``FID IID`` per line, GCTA ``.grm.id`` style) is what
    ``pylmm gwas --kinship`` aligns K by.  EMMA input has no IDs: a stale ID
    file from an earlier run is removed instead, since it would be taken as
    the IDs of this K.
    """
    log.info("Saving Kinship file to %s", out_file)
    np.savetxt(out_file, K)
    id_file = samples.kinship_id_path(out_file)
    if reader.type != "emma":
        samples.write_id_file(id_file, cast(list[samples.SampleID], reader.indivs))
        log.info("Saving kinship IDs to %s", id_file)
    elif os.path.isfile(id_file):
        os.remove(id_file)
        log.warning("removed the stale ID file %s (EMMA input has no IDs)", id_file)


def write_eigen(K: FloatArray, out_file: str, text: bool, npz: str | None) -> None:
    """Write the eigendecomposition of K as text (``out_file.kva/.kve``) and/or ``npz``."""
    if not (text or npz):
        return
    log.info("Obtaining Eigendecomposition")
    kva, kve = linalg.eigh(K)
    if text:
        log.info("Saving eigendecomposition to %s.[kva | kve]", out_file)
        np.savetxt(out_file + ".kva", kva)
        np.savetxt(out_file + ".kve", kve)
    if npz:
        log.info("Saving eigendecomposition to %s", npz)
        np.savez(npz, Kva=kva, Kve=kve)


def run(args: argparse.Namespace) -> None:
    """Run ``pylmm kinship`` with parsed options.

    Parameters
    ----------
    args : argparse.Namespace
        From the ``pylmm`` parser (`pylmm.cli.build_parser`).

    Raises
    ------
    UsageError
        On a usage or input error.
    """
    check_writable("--out", args.out)
    if args.eigen_npz:
        # np.savez adds .npz to a name without it
        npz = args.eigen_npz
        check_writable("--eigen-npz", npz if npz.endswith(".npz") else npz + ".npz")
    log.info("Reading PLINK input...")
    reader = open_genotypes(args)
    if args.emma_num_snps is not None:
        # The SNPs are counted from the file (one cheap pass).  A wrong
        # --emma-num-snps used to compute K from the first M SNPs, or to crash
        # with StopIteration.
        m = count_emma_snps(args.emma_geno)
        if args.emma_num_snps != m:
            raise UsageError(
                "--emma-num-snps %d: the --emma-geno file %s has %d SNPs (non-blank lines); "
                "fix the number or leave the option out" % (args.emma_num_snps, args.emma_geno, m)
            )
    K = compute_kinship(reader, args.block_size)
    try:
        write_kinship(K, reader, args.out)
        write_eigen(K, args.out, args.eigen_text, args.eigen_npz)
    except OSError as e:
        raise UsageError("cannot write the output: %s" % e) from e


if __name__ == "__main__":  # the removed pre-5b.2 invocation; see removed_module_main
    from pylmm.cli._common import removed_module_main

    raise SystemExit(removed_module_main("kinship"))
