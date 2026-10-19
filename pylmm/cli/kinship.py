# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""``pylmm kinship``: compute a realized-relationship kinship matrix.

``K = W W' / m`` over the ``m`` SNPs of the genotype input, with ``W`` the
standardised genotypes (mean-imputed); monomorphic SNPs and SNPs with no
called genotype are skipped, but still counted in ``m``.  The SNPs are read
in blocks of ``--block-size``.

Next to the kinship ``OUT`` it writes ``OUT.id``, the (FID, IID) of each row
(one ``FID IID`` per line), which ``pylmm gwas --kinship OUT`` uses to align
the matrix to its genotype file by ID.  EMMA input has no IDs, so no ID file
is written for it (and a stale one is removed).
"""

from __future__ import annotations

import argparse
import os
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from pylmm import plink_io, samples
from pylmm.cli._common import UsageError, add_logging_options, log

FloatArray = NDArray[np.float64]

_DESCRIPTION = """\
Compute a realized-relationship kinship matrix K = W W'/m from genotypes,
for `pylmm gwas --kinship`.  Writes K as whitespace-separated text to OUT
and the sample IDs of its rows to OUT.id (not for EMMA input, which has no
IDs)."""

_EPILOG = """\
examples:
  pylmm kinship --bfile data/snps.132k.clean.noX -o snps.kin
  pylmm kinship --bfile study -o study.kin --eigen-npz study.eig.npz
  pylmm kinship --emma-geno snps.emma --emma-num-snps 1000 -o K.txt"""


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
        help="EMMA genotype matrix: one SNP per row, one individual per column, no IDs "
        "(needs --emma-num-snps)",
    )
    geno.add_argument(
        "--emma-num-snps",
        metavar="M",
        type=_positive_int,
        help="the number of SNPs (rows) of the --emma-geno file",
    )
    geno.add_argument(
        "--block-size",
        metavar="N",
        type=_positive_int,
        default=1000,
        help="the number of SNPs held in memory at once (default: 1000)",
    )

    output = p.add_argument_group("output")
    output.add_argument(
        "-o",
        "--out",
        metavar="FILE",
        required=True,
        help="write the kinship to FILE and its sample IDs to FILE.id (both overwritten)",
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
    add_logging_options(p)
    return p


def _positive_int(text: str) -> int:
    """argparse type: an integer >= 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("%r is not an integer" % text) from None
    if value < 1:
        raise argparse.ArgumentTypeError("%d is not positive" % value)
    return value


def check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Check the option combinations argparse cannot express (exits 2 on an error).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ``kinship`` parser, for `argparse.ArgumentParser.error`.
    args : argparse.Namespace
        The parsed options.
    """
    if args.emma_geno and args.emma_num_snps is None:
        parser.error("--emma-geno needs --emma-num-snps (the number of SNPs in the file)")
    if args.emma_num_snps is not None and not args.emma_geno:
        parser.error("--emma-num-snps applies only to --emma-geno")


def open_genotypes(args: argparse.Namespace) -> plink_io.PlinkReader:
    """Open the genotype input (``--bfile``, ``--tfile`` or ``--emma-geno``).

    Raises
    ------
    UsageError
        On an invalid genotype file.
    """
    if args.bfile:
        base, kind = args.bfile, "b"
    elif args.tfile:
        base, kind = args.tfile, "t"
    else:
        base, kind = args.emma_geno, "emma"
    try:
        return plink_io.PlinkReader(base, type=kind)
    except ValueError as e:
        raise UsageError(str(e)) from e


def compute_kinship(reader: plink_io.PlinkReader, block_size: int) -> FloatArray:
    """``K = W W' / m`` over all SNPs of ``reader``, ``block_size`` SNPs at a time.

    SNPs that are monomorphic or have a missing value (after the reader's
    imputation: no called genotype at all) are skipped but still counted in
    ``m`` (bug 8 in docs/phase4-simulations.md; unchanged here).

    Parameters
    ----------
    reader : pylmm.plink_io.PlinkReader
        The genotype reader, with standardised SNPs (the default).  For EMMA
        input, ``reader.num_snps`` must be set after `iter_snps`.
    block_size : int
        The number of SNPs in memory at once.

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` kinship.

    Raises
    ------
    UsageError
        If no SNP with a non-constant genotype was read.
    """
    n = len(reader.indivs)
    W = np.ones((n, block_size)) * np.nan
    i = 0
    K: FloatArray | None = None
    n_all_missing = 0
    while i < reader.num_snps:
        j = 0
        while j < block_size and i < reader.num_snps:
            snp, _ = next(reader)
            # Skip monomorphic SNPs and SNPs with no called genotype (bug 11:
            # an all-missing SNP used to crash, or to make K all nan).
            if not np.isfinite(snp).all() or snp.var() == 0:
                if np.isnan(snp).all():
                    n_all_missing += 1
                i += 1
                continue
            W[:, j] = snp
            i += 1
            j += 1
        if j < block_size:
            W = W[:, list(range(0, j))]
        log.info("Processing first %d SNPs", i)
        K = np.dot(W, W.T) if K is None else K + np.dot(W, W.T)
    if n_all_missing:
        log.warning("skipped %d SNPs with no called genotype", n_all_missing)
    if K is None:
        raise UsageError("No SNP with a non-constant genotype was read; cannot compute a kinship.")
    result: FloatArray = K / float(reader.num_snps)
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
    log.info("Reading PLINK input...")
    reader = open_genotypes(args)
    reader.iter_snps()
    if reader.type == "emma":
        # Counting the SNPs of an EMMA file is expensive, so it is given.
        reader.num_snps = args.emma_num_snps
    K = compute_kinship(reader, args.block_size)
    try:
        write_kinship(K, reader, args.out)
        write_eigen(K, args.out, args.eigen_text, args.eigen_npz)
    except OSError as e:
        raise UsageError("cannot write the output: %s" % e) from e
