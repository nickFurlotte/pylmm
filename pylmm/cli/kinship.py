# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""``pylmm kinship``: compute a kinship matrix from genotypes.

``--kinship-type`` picks the kinship (`pylmm.kinship`): ``standardized`` (the
default, ``K = Z Z' / m`` over the standardised SNPs), ``centered`` (centred
0/1/2 allele counts, not scaled: GEMMA ``-gk 1``'s formula) or ``ibs``
(EMMA's identity-by-state kinship).  K is bit for bit
`pylmm.kinship.calculate_kinship` of the same raw genotypes (for
``centered``, of twice them: the readers' 0 / 0.5 / 1 fractions as 0/1/2
counts) at the same block size (the default ``--block-size`` is the library's
`pylmm.kinship.KINSHIP_BLOCK_SIZE`): missing calls are mean-imputed, and SNPs
with no called genotype
(and, except for ``ibs``, constant SNPs) are skipped and not counted in ``m``.
``--legacy-kinship`` reproduces the K of ``pylmm kinship`` 1.0.0, which still
counted them (bug 8).  The scale of K (mean diagonal, centred mean diagonal;
`pylmm.kinship.kinship_scale`) is logged.  The SNPs are read in blocks of
``--block-size`` (`pylmm.plink_io.PlinkReader.iter_blocks`).

Next to the kinship ``FILE`` (``-o``) it writes ``FILE.id``, the (FID, IID) of each row
(one ``FID IID`` per line), which ``pylmm gwas --kinship FILE`` uses to align
the matrix to its genotype file by ID.  EMMA input has no IDs, so no ID file
is written for it (and a stale one is removed).

``--out-format grm`` writes a GCTA binary GRM instead
(`pylmm.kinship_io.write_grm`): ``FILE.grm.bin`` (K's lower triangle in
float32), ``FILE.grm.N.bin`` (the ``m`` K was divided by, for every entry)
and ``FILE.grm.id``, for ``pylmm gwas --grm FILE``, GCTA and plink2.  The
eigendecomposition (``--eigen-text``, ``--eigen-npz``) is then that of the
float32-rounded K the GRM holds.  The ``--eigen-npz`` file carries the
sample IDs (arrays ``fid``, ``iid``) and the ``kinship_type``.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from pylmm import kinship_io, plink_io, samples
from pylmm.cli._common import (
    UsageError,
    add_logging_options,
    check_log_file,
    check_writable,
    genotype_source,
    log,
    positive_int,
)
from pylmm.kinship import (
    KINSHIP_BLOCK_SIZE,
    KINSHIP_KINDS,
    KinshipAccumulator,
    KinshipKind,
    kinship_scale,
)
from pylmm.samples import InputError

FloatArray = NDArray[np.float64]

_REPORT_EVERY = 10000
"""Log the progress of `compute_kinship` at INFO every this many SNPs."""

_DESCRIPTION = """\
Compute a kinship matrix from genotypes, for `pylmm gwas --kinship`: by
default the standardized relatedness matrix K = Z Z'/m (Z: the standardised
SNPs, missing calls mean-imputed; m: the SNPs used).  Writes K as
whitespace-separated text to FILE (-o) and the sample IDs of its rows to
FILE.id (not for EMMA input, which has no IDs), or with --out-format grm as a
GCTA binary GRM (FILE.grm.bin, FILE.grm.N.bin, FILE.grm.id; float32), and logs
the scale of K (its mean diagonal), the scale of the heritability fitted with
it."""

_EPILOG = """\
examples:
  pylmm kinship --bfile data/snps.132k.clean.noX -o snps.kin
  pylmm kinship --bfile study -o study.kin --eigen-npz study.eig.npz
  pylmm kinship --bfile study --out-format grm -o study
  pylmm kinship --emma-geno snps.emma -o K.txt
  pylmm kinship --emma-geno snps.emma --kinship-type ibs -o K.ibs.txt"""


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

    method = p.add_argument_group("kinship")
    method.add_argument(
        "--kinship-type",
        choices=KINSHIP_KINDS,
        default="standardized",
        help="standardized: the standardized relatedness matrix (GRM; GEMMA -gk 2's formula on "
        "complete genotypes); centered: centred allele counts 0/1/2, not scaled (GEMMA -gk 1's "
        "formula; it counts twice the reader's 0/0.5/1 values, so an --emma-geno file must be "
        "coded 0/0.5/1, EMMA's coding); ibs: EMMA's identity-by-state kinship, the share of "
        "alleles in common (inbred panels; genotypes must be 0, 0.5 or 1, as BED, TPED and "
        "EMMA 0/1 files are) (default: standardized)",
    )
    method.add_argument(
        "--legacy-kinship",
        action="store_true",
        help="reproduce the K of pylmm kinship 1.0.0 (standardized only): divide by all SNPs "
        "read, including the skipped constant and all-missing ones (bug 8)",
    )

    output = p.add_argument_group("output")
    output.add_argument(
        "-o",
        "--out",
        metavar="FILE",
        required=True,
        help="(required) write the kinship to FILE and its sample IDs to FILE.id (both "
        "overwritten); with --out-format grm, FILE is the prefix of the GRM files",
    )
    output.add_argument(
        "--out-format",
        choices=("text", "grm"),
        default="text",
        help="text: an n x n text matrix FILE plus FILE.id (the default; all digits); grm: a "
        "GCTA binary GRM, FILE.grm.bin (the lower triangle in float32, about 7 significant "
        "digits), FILE.grm.N.bin (the number of SNPs) and FILE.grm.id, as gcta --make-grm "
        "and plink2 --make-grm-bin write it (for `pylmm gwas --grm FILE`, GCTA, plink2; "
        "needs --bfile or --tfile: EMMA genotypes have no IDs) (default: text)",
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
        help="also write the eigendecomposition to NPZ, arrays Kva and Kve, with the sample "
        "IDs of Kve's rows (arrays fid, iid; not for EMMA input) and the kinship_type (for "
        "`pylmm gwas --eigen NPZ`, which checks the IDs; binary, exact and faster to read "
        "than --eigen-text)",
    )
    performance = p.add_argument_group("performance")
    performance.add_argument(
        "--block-size",
        metavar="N",
        type=positive_int,
        default=KINSHIP_BLOCK_SIZE,
        help="the number of SNPs read, and summed into K, at once: a few arrays of n x N floats in "
        "memory (default: %d, the library's, so that K is bit for bit calculate_kinship's)"
        % KINSHIP_BLOCK_SIZE,
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
    if args.legacy_kinship and args.kinship_type != "standardized":
        parser.error("--legacy-kinship applies only to --kinship-type standardized")
    if args.out_format == "grm" and args.emma_geno:
        parser.error(
            "--out-format grm needs sample IDs (FILE.grm.id), and EMMA genotypes have none: "
            "use --bfile or --tfile, or --out-format text"
        )
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
        # Raw genotypes: the kinship standardises (or not) each SNP itself.
        return plink_io.PlinkReader(base, type=kind, norm_genotype=False)
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


def compute_kinship(
    reader: plink_io.PlinkReader,
    block_size: int,
    kind: str = "standardized",
    legacy: bool = False,
) -> tuple[FloatArray, int]:
    """The kinship of all the SNPs of ``reader``, ``block_size`` SNPs at a time.

    `pylmm.kinship.KinshipAccumulator`, fed the raw genotypes: the same K as
    `pylmm.kinship.calculate_kinship` (bit for bit at the same block size).
    For ``kind="centered"`` the genotypes are counted as 0/1/2 copies of A2
    (twice the reader's fractions), the coding of GEMMA's ``-gk 1``: an EMMA
    file must be coded 0 / 0.5 / 1 for that (a value above 1 is warned
    about: a 0/1/2-coded file gives 4 times that K).

    Parameters
    ----------
    reader : pylmm.plink_io.PlinkReader
        The genotype reader, with ``norm_genotype=False`` (raw genotypes).
    block_size : int
        The number of SNPs in memory at once.
    kind : {"standardized", "centered", "ibs"}
        The kinship (`pylmm.kinship`).
    legacy : bool
        ``kind="standardized"`` only: pylmm 1.0.0's K, which divides by all
        the SNPs read (bug 8).

    Returns
    -------
    K : numpy.ndarray
        The ``(n, n)`` kinship.
    m : int
        The number of SNPs K was divided by (with ``legacy``, all SNPs
        read; otherwise the SNPs used).

    Raises
    ------
    UsageError
        If the genotype input has no usable SNP, or (``ibs``) a genotype that
        is not 0, 0.5 or 1.
    """
    acc = KinshipAccumulator(
        len(reader.indivs),
        kind=cast(KinshipKind, kind),
        block_size=block_size,
        legacy=legacy,
        scale=2.0 if kind == "centered" else 1.0,  # 0/0.5/1 fractions -> 0/1/2 counts
    )
    next_report = _REPORT_EVERY
    check_coding = kind == "centered" and reader.type == "emma"
    above_one = 0  # centered, EMMA input: SNPs with a value above 1 (not 0/0.5/1 coded)
    for snp in iter_snp_vectors(reader, block_size):
        if check_coding and np.nanmax(snp, initial=0.0) > 1.0:
            above_one += 1
        try:
            acc.add(snp)
        except ValueError as e:  # an IBS genotype that is not 0, 0.5 or 1
            raise UsageError("--kinship-type %s, SNP %d: %s" % (kind, acc.n_snps, e)) from e
        # Progress at INFO every 10000 SNPs, else at DEBUG every block.
        if acc.n_snps >= next_report:
            log.info("Processing first %d SNPs", acc.n_snps)
            next_report += _REPORT_EVERY
        elif acc.n_snps % block_size == 0:
            log.debug("Processing first %d SNPs", acc.n_snps)
    try:
        K = acc.result()
    except ValueError as e:  # no usable SNP
        raise UsageError(str(e)) from e
    log.info("Processed %d SNPs", acc.n_snps)
    if above_one:
        log.warning(
            "--kinship-type centered counts twice the --emma-geno values (0/0.5/1 -> 0/1/2), "
            "but %d SNPs have values above 1: if the file is coded 0/1/2, K is 4 times "
            "GEMMA -gk 1's (h is on that scale)",
            above_one,
        )
    if acc.n_all_missing:
        log.warning("skipped %d SNPs with no called genotype", acc.n_all_missing)
    skipped = acc.n_snps - acc.n_used - acc.n_all_missing
    if skipped:
        log.info("skipped %d constant SNPs", skipped)
    m = acc.n_snps if legacy else acc.n_used
    log.info(
        "kinship: %s, m = %d SNPs%s",
        kind,
        m,
        " (--legacy-kinship: all SNPs read; %d used)" % acc.n_used if legacy else "",
    )
    return K, m


def log_kinship_scale(K: FloatArray) -> None:
    """Log the scale of K (`pylmm.kinship.kinship_scale`), the scale of the h fitted with it."""
    if K.shape[0] < 2:
        return
    s = kinship_scale(K)
    log.info(
        "kinship scale: mean diagonal %.4f, centred mean diagonal trace(PKP)/(n-1) %.4f; "
        "h is the variance share on this scale (PVE = h c / (h c + 1 - h), c = the centred "
        "mean diagonal)",
        s.mean_diag,
        s.centered_mean_diag,
    )


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


def write_grm(K: FloatArray, m: int, reader: plink_io.PlinkReader, prefix: str) -> None:
    """Write K as a GCTA binary GRM (`pylmm.kinship_io.write_grm`): ``prefix.grm.*``."""
    log.info("Saving the kinship as a GCTA GRM to %s.grm.bin / .grm.N.bin / .grm.id", prefix)
    kinship_io.write_grm(prefix, K, cast(list[samples.SampleID], reader.indivs), n_snps=m)


def write_eigen(
    K: FloatArray,
    out_file: str,
    text: bool,
    npz: str | None,
    ids: list[samples.SampleID] | None = None,
    kinship_type: str | None = None,
) -> None:
    """Write the eigendecomposition of K as text (``out_file.kva/.kve``) and/or ``npz``.

    The ``.npz`` also holds ``ids`` (the samples of K's rows; None for EMMA
    input) and ``kinship_type`` (`pylmm.kinship_io.write_eigen`).
    """
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
        kinship_io.write_eigen(npz, kva, kve, ids=ids, kinship_type=kinship_type)


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
    grm = args.out_format == "grm"
    out = kinship_io.grm_prefix(args.out) if grm else args.out
    if grm:
        for suffix in kinship_io.GRM_SUFFIXES:
            check_writable("--out", out + suffix)
    else:
        check_writable("--out", out)
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
    K, m = compute_kinship(reader, args.block_size, args.kinship_type, args.legacy_kinship)
    log_kinship_scale(K)
    ids = None if reader.type == "emma" else cast(list[samples.SampleID], reader.indivs)
    try:
        if grm:
            write_grm(K, m, reader, out)
            K = kinship_io.round_like_grm(K)  # decompose the K the GRM holds
        else:
            write_kinship(K, reader, out)
        write_eigen(K, out, args.eigen_text, args.eigen_npz, ids, args.kinship_type)
    except OSError as e:
        raise UsageError("cannot write the output: %s" % e) from e


if __name__ == "__main__":  # the removed pre-5b.2 invocation; see removed_module_main
    from pylmm.cli._common import removed_module_main

    raise SystemExit(removed_module_main("kinship"))
