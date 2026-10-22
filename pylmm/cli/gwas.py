# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""``pylmm gwas``: genome-wide association with a linear mixed model.

Reads genotypes (PLINK BED or TPED, or an EMMA matrix), a phenotype, optional
covariates and a precomputed kinship (``pylmm kinship``), fits the null model
once (or per SNP with ``--refit``) and writes one tab-separated line per SNP:
``SNP_ID BETA BETA_SD F_STAT P_VALUE``.  With ``--loco`` the kinships are
built from the genotypes instead, one per chromosome, leaving that
chromosome out (`pylmm.loco`; `scan_loco`).

The steps are separate functions: `load_inputs` (genotypes, phenotype,
covariates, kinship and its eigendecomposition, aligned by sample ID),
`fit_null_model` and `scan` (the per-SNP tests, on blocks of ``--block-size``
SNPs read with `pylmm.plink_io.PlinkReader.iter_blocks`, written in file order).
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import os
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from pylmm import kinship_io, plink_io, samples
from pylmm.cli._common import (
    UsageError,
    add_logging_options,
    add_pfile_options,
    check_log_file,
    check_pfile_args,
    check_writable,
    genotype_source,
    log,
    positive_int,
)
from pylmm.genotypes import DEFAULT_BLOCK_SIZE, Variants, check_block_size
from pylmm.kinship import KINSHIP_KINDS, KinshipKind, kinship_scale
from pylmm.lmm import (
    LMM,
    AssociationResult,
    check_covariates,
    snp_estimable,
    snp_estimable_columns,
    subset_testable,
)
from pylmm.loco import (
    LocoKinship,
    describe_merged,
    loco_memory_bytes,
    merged_chrom_labels,
    normalize_chrom,
)

FloatArray = NDArray[np.float64]
SnpResult = tuple[Any, Any, Any, Any]
"""One SNP's ``(beta, beta_sd, t, p)``."""

OUTPUT_COLUMNS = ("SNP_ID", "BETA", "BETA_SD", "F_STAT", "P_VALUE")
"""The header of the output file (tab-separated)."""

ID_MATCH_CHOICES = {"fid-iid": "fid_iid", "iid": "iid"}
"""``--id-match`` values and the `pylmm.plink_io.PlinkReader` ``id_match`` they select."""

_DESCRIPTION = """\
Test each SNP for association with a phenotype under a linear mixed model.

The kinship (--kinship, computed with `pylmm kinship`, or a GCTA GRM with
--grm) accounts for relatedness.  Phenotype, covariate and kinship rows are
matched to the genotype file by sample ID (FID and IID, or IID alone with
--id-match iid), never by position; the kinship IDs come from its
`<kinship>.id` file (`<prefix>.grm.id` for --grm).
The output has one tab-separated line per SNP:
SNP_ID, BETA, BETA_SD, F_STAT and P_VALUE."""

_EPILOG = """\
examples:
  # BED genotypes, the README example data
  pylmm gwas --bfile data/snps.132k.clean.noX \\
      --kinship data/snps.132k.clean.noX.pylmm.kin \\
      --pheno data/snps.132k.clean.noX.fake.phenos -o results.tsv

  # a named phenotype column and two named covariates, IDs matched on IID
  pylmm gwas --bfile study --kinship study.kin --pheno study.phen \\
      --pheno-name bmi --covar study.cov --covar-name age sex \\
      --id-match iid -o bmi.tsv

  # EMMA-format inputs (no sample IDs: everything is in genotype order)
  pylmm gwas --emma-geno snps.emma --emma-pheno pheno.emma \\
      --kinship K.txt -o results.tsv

  # smaller blocks of SNPs tested together (less memory), or one SNP at a
  # time as in pylmm 1.0.0 (slower; see --no-batch for how the numbers compare)
  pylmm gwas --bfile study --kinship study.kin --pheno study.phen \\
      --block-size 200 -o results.tsv
  pylmm gwas --bfile study --kinship study.kin --pheno study.phen \\
      --no-batch -o results.tsv

  # the exact output of pylmm 1.0.0 (its h search and its one-SNP-at-a-time path)
  pylmm gwas --bfile study --kinship study.kin --pheno study.phen \\
      --legacy-fit --no-batch -o results.tsv

  # leave-one-chromosome-out: no --kinship, K_{-c} built from the genotypes
  pylmm gwas --bfile study --pheno study.phen --loco -o results.tsv
  pylmm gwas --emma-geno snps.emma --emma-pheno pheno.emma \\
      --loco --emma-map snps.map.tsv --kinship-type ibs -o results.tsv"""


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def add_parser(commands: Any) -> argparse.ArgumentParser:
    """Add the ``gwas`` subcommand to the ``pylmm`` parser.

    Parameters
    ----------
    commands : argparse._SubParsersAction
        The ``pylmm`` subparsers (``parser.add_subparsers()``).

    Returns
    -------
    argparse.ArgumentParser
        The ``gwas`` parser.
    """
    p: argparse.ArgumentParser = commands.add_parser(
        "gwas",
        help="genome-wide association scan with a linear mixed model",
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
    add_pfile_options(source, geno)

    pheno = p.add_argument_group("phenotype")
    pheno_file = pheno.add_mutually_exclusive_group()
    pheno_file.add_argument(
        "--pheno",
        metavar="FILE",
        help="PLINK-style phenotype file: FID IID value ..., with an optional FID IID name ... "
        "header; NA and -9 are missing (default: PREFIX.phenos of the genotype input)",
    )
    pheno_file.add_argument(
        "--emma-pheno",
        metavar="FILE",
        help="EMMA phenotype file: one phenotype per row, individuals in genotype order",
    )
    pheno_col = pheno.add_mutually_exclusive_group()
    pheno_col.add_argument(
        "--pheno-name",
        metavar="NAME",
        help="the phenotype column to test, by its name in the --pheno header",
    )
    pheno_col.add_argument(
        "--pheno-col",
        metavar="N",
        type=positive_int,
        help="the phenotype column to test, by its number among the value columns after "
        "FID IID; 1-based, like plink2 --pheno-col-nums (default: 1, the first)",
    )

    covar = p.add_argument_group("covariates")
    covar_file = covar.add_mutually_exclusive_group()
    covar_file.add_argument(
        "--covar",
        metavar="FILE",
        help="PLINK-style covariate file: FID IID value ..., with an optional header; NA is "
        "missing.  An intercept is added unless --no-intercept",
    )
    covar_file.add_argument(
        "--emma-covar",
        metavar="FILE",
        help="EMMA covariate file: one covariate per row, individuals in genotype order",
    )
    covar.add_argument(
        "--covar-name",
        metavar="NAME",
        nargs="+",
        help="use only these --covar columns, by their header names (space- or comma-separated)",
    )
    covar.add_argument(
        "--no-intercept",
        action="store_true",
        help="do not add an intercept column to the covariates (use when the covariate file "
        "has its own)",
    )

    kin = p.add_argument_group("kinship (--kinship, --grm or --loco)")
    # Required unless --loco (check_args).
    kin_source = kin.add_mutually_exclusive_group()
    kin_source.add_argument(
        "--kinship",
        metavar="FILE",
        help="n x n kinship matrix, whitespace-separated text (or .gz), from "
        "`pylmm kinship`.  "
        "With --bfile/--tfile/--pfile its rows are matched to the genotypes by the IDs in FILE.id if "
        "that exists; otherwise they are assumed to be in genotype order (with a warning).  "
        "With --emma-geno, which has no IDs, the rows are always in genotype order: no "
        "warning without FILE.id, and a FILE.id is ignored (with a warning).  A plink2 "
        "--make-rel file (FILE.rel, .rel.gz or .rel.bin; square or triangle) is read with "
        "its IDs from PREFIX.rel.id",
    )
    kin_source.add_argument(
        "--grm",
        metavar="PREFIX",
        help="a GCTA binary GRM instead of --kinship: PREFIX.grm.bin (float32) and "
        "PREFIX.grm.id, as gcta --make-grm, plink2 --make-grm-bin and `pylmm kinship "
        "--out-format grm` write it (the name of GCTA's --grm).  Its rows are matched to "
        "the genotypes by ID",
    )
    kin.add_argument(
        "--eigen",
        metavar="PATH",
        help="precomputed eigendecomposition of the kinship: a .npz file (`pylmm kinship "
        "--eigen-npz`) or the PATH.kva/PATH.kve text files (`pylmm kinship --eigen-text`).  "
        "Its sample IDs (in the .npz, or PATH.id / PATH.grm.id for text) must include every "
        "tested individual (an error otherwise); its rows are reordered to them, and with "
        "extra individuals (dropped since) it is ignored and recomputed.  Without IDs its "
        "rows are assumed to be the kinship's, and it is used only when no individual is "
        "dropped or reordered",
    )
    kin.add_argument(
        "--id-match",
        choices=sorted(ID_MATCH_CHOICES),
        default="fid-iid",
        help="match sample IDs on FID and IID (default) or on the IID alone (PLINK input only)",
    )

    loco = p.add_argument_group("leave one chromosome out (instead of --kinship)")
    loco.add_argument(
        "--loco",
        action="store_true",
        help="test the SNPs of each chromosome with a kinship built from the SNPs of the other "
        "chromosomes (like GCTA --mlma-loco), computed from the genotypes in one pass (no "
        "--kinship): one eigendecomposition and null fit per chromosome.  Needs each SNP's "
        "chromosome (the .bim, .map or .pvar file; --emma-map for --emma-geno; labels such as chr1, "
        "01 and 1 are one chromosome) and the SNPs of a chromosome together in the file.  "
        "Memory: about (chromosomes + 6) x n^2 x 8 bytes (logged); time: about the standard "
        "run at n ~ 1200, several times it for larger n.  The genotypes are read in full "
        "for the kinships first, so a genotype read error stops the run before any output.  "
        "Under --loco a polygenic trait gives lambda_GC above 1 without confounding: do not "
        "apply genomic control to it",
    )
    loco.add_argument(
        "--loco-max-chroms",
        metavar="N",
        type=positive_int,
        default=None,
        help="with --loco: refuse more than N chromosomes (default: %d; more usually means "
        "contigs, or a chromosome column read from the wrong place)" % LOCO_MAX_CHROMS,
    )
    loco.add_argument(
        "--kinship-type",
        choices=KINSHIP_KINDS,
        default=None,
        help="with --loco: the kinship to build, as `pylmm kinship --kinship-type` "
        "(standardized, centered or ibs; default: standardized)",
    )
    loco.add_argument(
        "--emma-map",
        metavar="FILE",
        help="with --loco and --emma-geno: the chromosome of each EMMA row, one line per row "
        "in order: a header line naming a chr (or chrom, CHR, #CHROM) column, as "
        "data/hmdp.liver.snps.map.tsv, or no header and the chromosome first, as a PLINK .map",
    )

    model = p.add_argument_group("model")
    method = model.add_mutually_exclusive_group()
    method.add_argument(
        "--reml",
        dest="reml",
        action="store_true",
        default=True,
        help="restricted maximum likelihood, for the null fit, the per-SNP fits and the test "
        "(the default)",
    )
    method.add_argument(
        "--ml",
        dest="reml",
        action="store_false",
        help="maximum likelihood instead of REML (sigma^2 = Q/n in the test).  With a centred "
        "kinship built from more SNPs than individuals the ML estimate of h can be biased upward",
    )
    model.add_argument(
        "--refit",
        action="store_true",
        help="refit the variance components at each SNP (default: fit them once, under the "
        "null model)",
    )
    model.add_argument(
        "--legacy-fit",
        action="store_true",
        help="fit the variance components as pylmm 1.0.0 did: h searched on 0..0.99 only "
        "(never 1; a peak at 0.98 missed), the first local maximum kept, and sigma^2 = "
        "Q/(n-p) under --ml.  With --no-batch the output is that of pylmm 1.0.0",
    )
    model.add_argument(
        "--remove-missing-genotypes",
        action="store_true",
        help="test each SNP on the individuals with a called genotype (refitting the model on "
        "them) instead of replacing missing calls by the mean.  Slower: one eigendecomposition "
        "per SNP with missing calls",
    )

    performance = p.add_argument_group("performance")
    performance.add_argument(
        "--block-size",
        metavar="N",
        type=positive_int,
        default=DEFAULT_BLOCK_SIZE,
        help="the number of SNPs read and tested together: a block is read at once (a BED "
        "file is decoded a block at a time), and its complete SNPs (no missing call) are "
        "rotated with one matrix product and tested in closed form; memory "
        "is about 6-8 x n x N floats (default: %d)" % DEFAULT_BLOCK_SIZE,
    )
    performance.add_argument(
        "--no-batch",
        action="store_true",
        help="test one SNP at a time (LMM.association), the code path of pylmm 1.0.0: slower.  "
        "The numbers are the same up to rounding (within about 1e-9 relative); with "
        "--legacy-fit and a badly scaled covariate (e.g. a raw date or year column) they can "
        "differ by a few 1e-6 relative, the batched default being the more accurate: centre "
        "or scale such covariates.  --refit and SNPs with missing calls always take this path",
    )

    output = p.add_argument_group("output")
    output.add_argument(
        "-o",
        "--out",
        metavar="FILE",
        required=True,
        help="(required) write the results to FILE (overwritten)",
    )
    add_logging_options(p)
    return p


def check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Check the option combinations argparse cannot express (exits 2 on an error).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ``gwas`` parser, for `argparse.ArgumentParser.error`.
    args : argparse.Namespace
        The parsed options; ``covar_name`` is flattened (comma-separated
        values are split).
    """
    if args.covar_name is not None:
        args.covar_name = [n for arg in args.covar_name for n in arg.split(",") if n]
        if not args.covar_name:
            parser.error("--covar-name: no column name given")
        if not args.covar:
            parser.error("--covar-name needs --covar (EMMA covariate files have no names)")
    if args.pheno_name is not None and args.emma_pheno:
        parser.error("--pheno-name needs --pheno (EMMA phenotype files have no names)")
    if args.emma_geno and args.id_match != "fid-iid":
        parser.error("--id-match iid needs PLINK genotypes; EMMA genotypes have no IDs")
    check_pfile_args(parser, args)
    if args.loco and args.loco_max_chroms is None:
        args.loco_max_chroms = LOCO_MAX_CHROMS
    if args.loco:
        given = [f for f, v in (("--kinship", args.kinship), ("--grm", args.grm)) if v]
        if given:
            parser.error(
                "--loco builds the kinships from the genotypes: drop %s (to use your own "
                "per-chromosome kinships, see the custom-association-analysis skill)" % given[0]
            )
        if args.eigen:
            parser.error("--eigen does not apply to --loco (one kinship per chromosome)")
        if args.emma_geno and not args.emma_map:
            parser.error(
                "--loco needs the chromosome of each SNP, and EMMA genotypes have none: give a "
                "marker map with --emma-map (e.g. data/hmdp.liver.snps.map.tsv)"
            )
    elif args.kinship is None and args.grm is None:
        parser.error(
            "one of the arguments --kinship --grm is required (or --loco): give --kinship FILE "
            "(from `pylmm kinship`) or --grm PREFIX, or --loco to build a leave-one-chromosome-out "
            "kinship for each chromosome from the genotypes"
        )
    else:
        for flag, value in (
            ("--kinship-type", args.kinship_type),
            ("--emma-map", args.emma_map),
            ("--loco-max-chroms", args.loco_max_chroms),
        ):
            if value is not None:
                parser.error("%s applies only to --loco" % flag)
    if args.emma_map and not args.emma_geno:
        parser.error(
            "--emma-map applies only to --emma-geno (PLINK input has a .bim, .map or .pvar)"
        )
    if args.grm is not None:
        # The GRM is read as a kinship file named PREFIX.grm.bin (kinship_io.kinship_format).
        args.kinship = kinship_io.grm_prefix(args.grm) + ".grm.bin"
    try:
        check_block_size(args.block_size)
    except ValueError as e:
        parser.error("--block-size: %s" % e)
    check_log_file(parser, args)


# ---------------------------------------------------------------------------
# kinship and eigendecomposition files
# ---------------------------------------------------------------------------


def read_kinship_file(fn: str, n: int) -> FloatArray:
    """Read an ``n x n`` kinship matrix from a text file.

    Parameters
    ----------
    fn : str
        Whitespace-separated text (gzip if ``fn`` ends in ``.gz``): ``n``
        lines of ``n`` numbers, with no header or comment lines.
    n : int
        The expected number of rows and columns.

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` kinship.  A noticeably asymmetric matrix is accepted
        with a warning.

    Raises
    ------
    ValueError
        Naming the file, if it cannot be read, a token is not a number, the
        file is not exactly ``n`` rows of ``n`` values, or the matrix has a
        nan/inf.  (Fixed in 1.0.0: the old ``np.fromfile`` +
        ``resize`` read silently zero-filled or truncated K.)
    """
    try:
        f: IO[str] = gzip.open(fn, "rt") if fn[-3:] == ".gz" else open(fn)
    except OSError as e:
        raise ValueError("kinship file %s: cannot read (%s)" % (fn, e)) from e
    K = np.zeros((n, n))
    nrows = 0
    nvalues = 0
    lengths: set[int] = set()
    try:
        with f:
            for lineno, line in enumerate(f, 1):
                v = line.split()
                if not v:
                    continue
                try:
                    row = np.array(v, dtype=np.float64)
                except ValueError:
                    bad = next((x for x in v if not _is_number(x)), v[0])
                    raise ValueError(
                        "kinship file %s, line %d: %r is not a number (expected %d rows of %d "
                        "whitespace-separated numbers, with no header or comment lines)"
                        % (fn, lineno, bad[:40], n, n)
                    ) from None
                if nrows < n and len(row) == n:
                    K[nrows] = row
                nrows += 1
                nvalues += len(row)
                lengths.add(len(row))
    except (OSError, UnicodeDecodeError) as e:
        # missing file, bad gzip, binary content: a usage-style error, not a traceback
        raise ValueError("kinship file %s: cannot read (%s: %s)" % (fn, type(e).__name__, e)) from e
    if nrows != n or lengths != {n}:
        found = sorted(lengths)
        raise ValueError(
            "kinship file %s: expected a %d x %d matrix (%d values, one row and column per "
            "individual in the genotype input), found %d values in %d rows (row lengths: %s)"
            % (
                fn,
                n,
                n,
                n * n,
                nvalues,
                nrows,
                ", ".join(str(x) for x in found[:5]) + (", ..." if len(found) > 5 else ""),
            )
        )
    bad_values = ~np.isfinite(K)
    if bad_values.any():
        i, j = np.argwhere(bad_values)[0]
        raise ValueError(
            "kinship file %s: %d non-finite values (nan/inf), the first at row %d, column %d"
            % (fn, bad_values.sum(), i + 1, j + 1)
        )
    asym = np.abs(K - K.T).max() if n else 0.0
    if asym > 1e-6 * max(1.0, np.abs(K).max()):
        warnings.warn(
            "kinship file %s is not symmetric: max |K - K'| = %g" % (fn, asym), stacklevel=2
        )
    return K


def _is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def read_eigen_file(fn: str, K: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Read the eigendecomposition given to ``--eigen`` and check it against ``K``.

    Parameters
    ----------
    fn : str
        A ``.npz`` file with arrays ``Kva`` and ``Kve`` (``pylmm kinship
        --eigen-npz``), or the base name of the text files ``fn.kva`` and
        ``fn.kve`` (``pylmm kinship --eigen-text``).
    K : numpy.ndarray
        The ``(n, n)`` kinship the decomposition must belong to.

    Returns
    -------
    kva, kve : numpy.ndarray
        Eigenvalues ``(n,)`` and eigenvectors ``(n, n)``.

    Raises
    ------
    ValueError
        If the files are missing, have the wrong shape or nan/inf values, or
        are not the eigendecomposition of ``K`` (checked on a random
        vector).  (Fixed in 1.0.0: this used to read files that
        the kinship program never wrote.)
    """
    e = kinship_io.read_eigen(fn)
    return check_eigen(fn, e.kva, e.kve, K)


def check_eigen(
    fn: str, kva: FloatArray, kve: FloatArray, K: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Check that ``kva``, ``kve`` (read from ``fn``) are the eigendecomposition of ``K``.

    Raises
    ------
    ValueError
        As `read_eigen_file`.
    """
    n = K.shape[0]
    if kva.shape != (n,) or kve.shape != (n, n):
        raise ValueError(
            "--eigen %s: eigenvalues of shape %s and eigenvectors of shape %s, expected (%d,) and (%d, %d)"
            % (fn, kva.shape, kve.shape, n, n, n)
        )
    if not (np.isfinite(kva).all() and np.isfinite(kve).all()):
        raise ValueError("--eigen %s has nan/inf values" % fn)
    u = np.random.default_rng(0).standard_normal(n)
    err = np.abs(kve.dot(kva * kve.T.dot(u)) - K.dot(u)).max()
    if err > 1e-6 * max(1.0, np.abs(K.dot(u)).max()):
        raise ValueError(
            "--eigen %s is not the eigendecomposition of the --kinship matrix (max error %g)"
            % (fn, err)
        )
    return kva, kve


def read_aligned_kinship(k_file: str, reader: plink_io.PlinkReader) -> tuple[FloatArray, bool]:
    """Read the ``--kinship`` matrix in the order of ``reader.indivs``.

    With an ID file ``k_file + ".id"`` (``pylmm kinship`` writes one), K is
    matched to the genotype individuals by ID (as ``reader.id_match`` says):
    genotype individuals that are not in K are dropped from the reader
    (indivs, phenotypes and SNP vectors) and K's extra individuals are
    ignored, each with a warning.  Without one, K's rows are assumed to be
    the genotype individuals in file order, with a warning (not for EMMA
    input, which has no IDs).  A GCTA GRM (``k_file`` ending in ``.grm.bin``)
    or a plink2 ``.rel`` file brings its own IDs (`pylmm.kinship_io`).

    Note: when K drops individuals, genotypes are imputed and standardised
    over the KEPT individuals, but individuals dropped for a missing
    phenotype or covariate still count in the mean and SD (max |dt| 0.024
    between the two rules; making them consistent is planned).

    Parameters
    ----------
    k_file : str
        The kinship file.
    reader : pylmm.plink_io.PlinkReader
        The genotype reader; individuals not in K are dropped from it.

    Returns
    -------
    K : numpy.ndarray
        The kinship, ``(len(reader.indivs), len(reader.indivs))``.
    aligned : bool
        Whether K or the reader was reordered or subset (a precomputed
        eigendecomposition no longer applies).

    Raises
    ------
    ValueError
        If the kinship or its ID file is malformed or they disagree.
    """
    if kinship_io.kinship_format(k_file) != "text":
        K, file_ids = kinship_io.read_kinship_with_ids(k_file)
        if reader.type == "emma":
            if K.shape[0] != len(reader.indivs):
                raise samples.InputError(
                    "kinship %s has %d samples but the EMMA genotypes have %d individuals"
                    % (k_file, K.shape[0], len(reader.indivs))
                )
            log.warning(
                "the sample IDs of %s are ignored: EMMA genotypes have no IDs, so K is in "
                "genotype order",
                k_file,
            )
            return K, False
        return _align_kinship(K, file_ids, reader)
    id_file = samples.kinship_id_path(k_file)
    if reader.type == "emma":
        if os.path.isfile(id_file):
            log.warning(
                "%s is ignored: EMMA genotypes have no IDs, so K is in genotype order", id_file
            )
        return read_kinship_file(k_file, len(reader.indivs)), False
    kin_ids = samples.read_kinship_ids(k_file)
    if kin_ids is None:
        log.warning(
            "no kinship ID file %s; assuming the rows of %s are the individuals of the "
            "genotype file, in its order (pylmm kinship writes the ID file)",
            id_file,
            k_file,
        )
        return read_kinship_file(k_file, len(reader.indivs)), False
    try:
        K = read_kinship_file(k_file, len(kin_ids))
    except ValueError as e:
        raise ValueError(
            "%s (its ID file %s lists %d individuals)" % (e, id_file, len(kin_ids))
        ) from e
    return _align_kinship(K, kin_ids, reader)


def _align_kinship(
    K: FloatArray, kin_ids: list[samples.SampleID], reader: plink_io.PlinkReader
) -> tuple[FloatArray, bool]:
    """K (rows ``kin_ids``) and ``reader`` aligned by ID; see `read_aligned_kinship`."""
    try:
        al = samples.align_samples(
            {"genotype file": reader.indivs, "kinship": kin_ids}, match=reader.id_match
        )
    except samples.InputError as e:
        if reader.id_match == "fid_iid" and all(fid == "0" for fid, _ in kin_ids):
            raise samples.InputError(
                "%s.  Every kinship FID is 0 (plink2 writes that for samples without an FID): "
                "try --id-match iid" % e
            ) from e
        raise
    for line in al.report():
        log.warning("%s", line)
    aligned = False
    if not al.is_identity("genotype file"):
        reader.subset(al.ids)
        aligned = True
    if not al.is_identity("kinship"):
        K = al.take_square("kinship", K)
        aligned = True
    return K, aligned


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


@dataclass
class GwasInputs:
    """Everything the scan needs, aligned to the tested individuals.

    Attributes
    ----------
    reader : pylmm.plink_io.PlinkReader
        The genotype reader (its SNP vectors have ``len(reader.indivs)``
        values; `keep` selects the tested individuals among them).
    Y : numpy.ndarray
        The phenotype of the tested individuals, ``(n,)``.
    X0 : numpy.ndarray
        Covariates (with the intercept, unless ``--no-intercept``), ``(n, q)``.
    K : numpy.ndarray
        The kinship of the tested individuals, ``(n, n)``; with ``--loco`` an
        empty ``(0, 0)`` placeholder (each chromosome gets its own, `scan_loco`).
    keep : numpy.ndarray
        Boolean mask over ``reader.indivs``: individuals with a phenotype and
        all covariates.
    kva, kve : numpy.ndarray or None
        The ``--eigen`` decomposition of K, or None to compute it.
    loco : pylmm.loco.LocoKinship or None
        With ``--loco``: the leave-one-chromosome-out kinships, over all of
        ``reader.indivs``.
    chroms : numpy.ndarray or None
        With ``--loco``: the chromosome of each SNP, in file order.
    """

    reader: plink_io.PlinkReader
    Y: FloatArray
    X0: FloatArray
    K: FloatArray
    keep: NDArray[np.bool_]
    kva: FloatArray | None
    kve: FloatArray | None
    loco: LocoKinship | None = None
    chroms: NDArray[np.object_] | None = None


def _usage(fn: Any, *a: Any, **kw: Any) -> Any:
    """Call ``fn``, turning a ValueError (invalid input) into a `UsageError`."""
    try:
        return fn(*a, **kw)
    except ValueError as e:
        raise UsageError(str(e)) from e


def open_genotypes(args: argparse.Namespace) -> plink_io.PlinkReader:
    """Open the genotype input (``--bfile``, ``--tfile``, ``--pfile`` or ``--emma-geno``).

    The reader also reads the ``--pheno`` file (or ``PREFIX.phenos``).

    Raises
    ------
    UsageError
        On an invalid genotype or phenotype file.
    """
    flag, base, kind = genotype_source(args)
    try:
        return plink_io.PlinkReader(
            base,
            type=kind,
            pheno_file=args.pheno,
            norm_genotype=not args.remove_missing_genotypes,
            id_match=ID_MATCH_CHOICES[args.id_match],
            dosage=not args.hardcalls,
        )
    except ImportError as e:  # --pfile without pgenlib: the message says how to install it
        raise UsageError("%s %s: %s" % (flag, base, e)) from e
    except OSError as e:  # e.g. a missing .fam: name the option
        raise UsageError("%s %s: %s" % (flag, base, e)) from e
    except ValueError as e:  # the message names the file (genotype or phenotype)
        raise UsageError(str(e)) from e


def _check_input_files(args: argparse.Namespace, reader: plink_io.PlinkReader) -> None:
    """Fail early, before the kinship is read, if it or a phenotype or covariate file is missing."""
    if args.loco:
        pass  # no kinship file
    elif args.grm is not None:
        for fn in (args.kinship, kinship_io.grm_prefix(args.kinship) + ".grm.id"):
            if not os.path.isfile(fn):
                raise UsageError("--grm %s: no such file %s" % (args.grm, fn))
    elif not os.path.isfile(args.kinship):
        raise UsageError("--kinship %s: no such file" % args.kinship)
    elif kinship_io.kinship_format(args.kinship) == "rel":
        id_file = kinship_io.rel_id_path(args.kinship)
        if not os.path.isfile(id_file):
            raise UsageError(
                "--kinship %s: no such file %s (the IDs of a plink2 .rel)" % (args.kinship, id_file)
            )
    # No phenotype file is a usage error (bug 14, fixed in 1.0.0: this
    # used to be os.path.isfile(None), a TypeError).
    pheno_file = args.pheno or reader.fbase + ".phenos"
    if args.emma_pheno:
        if not os.path.isfile(args.emma_pheno):
            raise UsageError("--emma-pheno: the file %s does not exist" % args.emma_pheno)
    elif not os.path.isfile(pheno_file):
        raise UsageError("no phenotype file %s; give one with --pheno or --emma-pheno" % pheno_file)
    if args.covar and not os.path.isfile(args.covar):
        raise UsageError("--covar: the file %s does not exist" % args.covar)


def _read_covariates(args: argparse.Namespace, reader: plink_io.PlinkReader) -> FloatArray | None:
    """The ``--covar`` / ``--emma-covar`` columns (no intercept), or None."""
    if args.covar:
        log.info("Reading covariate file...")
        cov: FloatArray = _usage(reader.read_covariates, args.covar, columns=args.covar_name)
        return cov
    if args.emma_covar:
        log.info("Reading covariate file...")
        cov = _usage(reader.read_covariates_emma, args.emma_covar)
        return cov
    return None


def _pheno_column(args: argparse.Namespace, reader: plink_io.PlinkReader) -> int:
    """The index of the tested phenotype among ``reader.phenos``' columns."""
    assert reader.phenos is not None
    n_cols = reader.phenos.shape[1]
    if args.pheno_name is not None:
        names = reader.pheno_names
        if names is None:
            raise UsageError(
                "--pheno-name %s: the phenotype file has no FID IID name ... header; "
                "use --pheno-col" % args.pheno_name
            )
        if args.pheno_name not in names:
            raise UsageError(
                "--pheno-name %s: no such column in the phenotype file (it has %s)"
                % (args.pheno_name, ", ".join(names))
            )
        return names.index(args.pheno_name)
    # The default (None) is the first column.  (argparse's default of a
    # mutually exclusive option must be None, or an explicit `--pheno-col 1`
    # would not conflict with --pheno-name.)  --pheno-col is 1-based.
    col: int = 1 if args.pheno_col is None else args.pheno_col
    if col > n_cols:
        raise UsageError(
            "--pheno-col %d: the phenotype file has %d phenotype column(s) (1-based, like "
            "plink2 --pheno-col-nums)" % (col, n_cols)
        )
    return col - 1


def _design_matrix(n: int, P: FloatArray | None, no_intercept: bool) -> FloatArray:
    """The covariate matrix X0: an intercept (unless ``no_intercept``), then the covariates ``P``."""
    if P is None:
        return np.ones((n, 1))
    if no_intercept:
        return P
    return np.hstack([np.ones((n, 1)), P])


def _drop_incomplete(
    Y: FloatArray, X0: FloatArray, K: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
    """Drop the individuals with a missing phenotype or covariate, with a warning.

    (Before 1.0.0 a missing covariate was a fatal error.)

    Returns
    -------
    Y, X0, K : numpy.ndarray
        Restricted to the kept individuals (an empty K, the ``--loco``
        placeholder, is returned as is).
    keep : numpy.ndarray
        Boolean mask of the kept individuals (over the reader's ``indivs``).

    Raises
    ------
    UsageError
        If no individual is left.
    """
    missing_y = np.isnan(Y)
    missing_cov = np.isnan(X0).any(axis=1)
    dropped = missing_y | missing_cov
    keep = ~dropped
    if missing_y.sum():
        log.warning(
            "%d of %d individuals have a missing phenotype and are dropped", missing_y.sum(), len(Y)
        )
    if (missing_cov & ~missing_y).sum():
        log.warning(
            "%d individuals with a phenotype have a missing covariate value and are dropped",
            (missing_cov & ~missing_y).sum(),
        )
    if dropped.sum():
        Y = Y[keep]
        X0 = X0[keep, :]
        if K.size:
            K = K[keep, :][:, keep]
    if not len(Y):
        raise UsageError("No individual has both a phenotype and all covariates.")
    return Y, X0, K, keep


def load_inputs(args: argparse.Namespace) -> GwasInputs:
    """Read and align the genotypes, phenotype, covariates and kinship.

    Individuals with a missing phenotype or covariate are dropped (with a
    warning), and so are genotype individuals that are not in a kinship with
    IDs.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed ``pylmm gwas`` options.

    Returns
    -------
    GwasInputs

    Raises
    ------
    UsageError
        On a missing or invalid input, or when no individual is left.
    """
    log.info("Reading SNP input...")
    reader = open_genotypes(args)
    _check_input_files(args, reader)

    # The kinship is read before the covariates: when it has IDs, the
    # genotype individuals that are not in it are dropped from the reader
    # (indivs, phenos, SNP vectors).
    loco: LocoKinship | None = None
    chroms: NDArray[np.object_] | None = None
    if args.loco:
        # The kinships are built after the phenotype and covariates are read
        # (below), so that an error in those comes first; no all-SNP K is
        # built or kept (K is an empty placeholder: each chromosome's K_{-c}
        # is made when its scan starts).
        chroms = loco_chromosomes(args, reader)
        K, kinship_aligned = np.empty((0, 0)), False
    else:
        log.info("Reading kinship...")
        begin = time.time()
        K, kinship_aligned = _usage(read_aligned_kinship, args.kinship, reader)
        log.info(
            "Read the %d x %d kinship matrix in %0.3fs", K.shape[0], K.shape[1], time.time() - begin
        )

    if args.emma_pheno:
        # rows are phenotypes, columns are individuals
        reader.phenos = _usage(reader.read_phenos_emma, args.emma_pheno)
    P = _read_covariates(args, reader)
    column = _pheno_column(args, reader)
    assert reader.phenos is not None
    n_all = reader.phenos.shape[0]
    X0 = _design_matrix(n_all, P, args.no_intercept)

    Y, X0, K, keep = _drop_incomplete(reader.phenos[:, column], X0, K)
    name = "the covariates" + (
        ""
        if args.no_intercept or (P is None)
        else " (column 0 is the intercept pylmm gwas adds; use --no-intercept if your file has its own)"
    )
    _usage(check_covariates, X0, what=name)
    if chroms is not None:
        loco = compute_loco_kinship(args, reader, chroms)

    kva, kve = _eigen(args, K, reader, keep, kinship_aligned)
    return GwasInputs(
        reader=reader, Y=Y, X0=X0, K=K, keep=keep, kva=kva, kve=kve, loco=loco, chroms=chroms
    )


def _eigen(
    args: argparse.Namespace,
    K: FloatArray,
    reader: plink_io.PlinkReader,
    keep: NDArray[np.bool_],
    kinship_aligned: bool,
) -> tuple[FloatArray | None, FloatArray | None]:
    """The ``--eigen`` decomposition, if given and still valid for K; else (None, None).

    With sample IDs (PLINK genotypes), its IDs must include every tested
    individual (``reader.indivs[keep]``, K's rows), else it is an error; if
    they are exactly the tested individuals, its rows are reordered to them,
    and if it has more (individuals dropped since) it is ignored, with a
    warning, and recomputed.  Without IDs it is valid only if no individual
    was dropped or reordered, as before.  Either way it must then be the
    eigendecomposition of K (`check_eigen`).
    """
    if not args.eigen:
        return None, None
    e = _usage(kinship_io.read_eigen, args.eigen)
    if e.ids is not None and reader.type != "emma":
        return _eigen_by_id(args.eigen, e, K, reader, keep)
    if e.ids is not None:
        log.warning(
            "the sample IDs of --eigen %s are ignored: EMMA genotypes have no IDs", args.eigen
        )
    n_dropped = int((~keep).sum())
    if n_dropped:
        log.warning(
            "--eigen is ignored: %d individuals were dropped, so the decomposition is recomputed",
            n_dropped,
        )
        return None, None
    if kinship_aligned:
        log.warning(
            "--eigen is ignored: the kinship was reordered to the genotype individuals by "
            "its IDs, so the decomposition is recomputed"
        )
        return None, None
    if reader.type != "emma":
        log.warning(
            "--eigen %s has no sample IDs; assuming its rows are the kinship's, in genotype "
            "order (pylmm kinship --eigen-npz writes the IDs)",
            args.eigen,
        )
    log.info("Loading pre-computed eigendecomposition...")
    kva, kve = _usage(check_eigen, args.eigen, e.kva, e.kve, K)
    return kva, kve


def _eigen_by_id(
    fn: str,
    e: kinship_io.Eigen,
    K: FloatArray,
    reader: plink_io.PlinkReader,
    keep: NDArray[np.bool_],
) -> tuple[FloatArray | None, FloatArray | None]:
    """`_eigen` for a decomposition with sample IDs: match them to the tested individuals."""
    assert e.ids is not None
    tested = [reader.indivs[i] for i in np.flatnonzero(keep)]
    rows = _usage(
        samples.match_index,
        tested,
        e.ids,
        match=reader.id_match,
        target_name="the tested individuals",
        source_name="--eigen %s" % fn,
    )
    missing = np.flatnonzero(rows < 0)
    if len(missing):
        raise UsageError(
            "--eigen %s is not the eigendecomposition of this kinship: %d of the %d tested "
            "individuals are not among its %d sample IDs, e.g. %s"
            % (
                fn,
                len(missing),
                len(tested),
                len(e.ids),
                ", ".join(samples.format_id(tested[i]) for i in missing[:3]),
            )
        )
    if len(e.ids) != len(tested):
        log.warning(
            "--eigen is ignored: %d of its %d individuals are not tested (not in the genotypes "
            "or the kinship, or missing a phenotype or covariate), so the decomposition is "
            "recomputed",
            len(e.ids) - len(tested),
            len(e.ids),
        )
        return None, None
    kve = e.kve
    if not np.array_equal(rows, np.arange(len(rows))):
        log.info("Reordering the --eigen eigenvectors to the tested individuals by sample ID")
        kve = kve[rows]
    log.info("Loading pre-computed eigendecomposition...")
    kva, kve = _usage(check_eigen, fn, e.kva, kve, K)
    return kva, kve


# ---------------------------------------------------------------------------
# model fitting and the scan
# ---------------------------------------------------------------------------


def fit_null_model(inputs: GwasInputs, args: argparse.Namespace) -> LMM:
    """Build the LMM of the tested individuals and fit its null model.

    With ``--refit`` the variance components are fitted per SNP instead,
    so the null model is not fitted here.  REML by default, ML with
    ``--ml``, here and in every per-SNP fit (before 1.0.0 the null fit was
    always REML).

    Raises
    ------
    UsageError
        If the model inputs are invalid.
    """
    # LMM(verbose=True) prints its own diagnostic lines (eigendecomposition
    # timing, clamped eigenvalues) straight to stderr; show them with --debug.
    verbose: bool = args.debug
    L: LMM = _usage(
        LMM,
        inputs.Y,
        inputs.K,
        inputs.kva,
        inputs.kve,
        inputs.X0,
        verbose=verbose,
        legacy_fit=args.legacy_fit,
    )
    if not args.refit:
        log.info("Computing fit for null model")
        L.fit(reml=args.reml)
        log.info("null model: heritability=%0.3f, sigma=%0.3f", L.opt_h, L.opt_sigma)
    return L


def write_header(out: IO[str]) -> None:
    """Write the output header line (`OUTPUT_COLUMNS`, tab-separated)."""
    out.write("\t".join(OUTPUT_COLUMNS) + "\n")


def write_result(out: IO[str], snp_id: str, beta: Any, beta_sd: Any, ts: Any, ps: Any) -> None:
    """Write one SNP's result line; the values are formatted with `str`."""
    out.write("\t".join([str(x) for x in [snp_id, beta, beta_sd, ts, ps]]) + "\n")


def _test_snp(
    snp: FloatArray,
    inputs: GwasInputs,
    L: LMM,
    Q0: FloatArray,
    args: argparse.Namespace,
    counts: dict[str, int],
) -> tuple[Any, Any, Any, Any] | None:
    """Test one SNP; ``(beta, beta_sd, t, p)``, or None if it is not estimable."""
    n = inputs.K.shape[0]
    x = snp[inputs.keep].reshape((n, 1))
    missing = np.isnan(x).reshape((-1,))
    if missing.sum():
        called = ~missing
        xs = x[called, :]
        ys = inputs.Y[called]
        x0s = inputs.X0[called, :]
        # One estimability check for both branches (bug 13, issue 19)
        if not subset_testable(xs, ys, x0s, counts):
            return None
        # Only reached with --remove-missing-genotypes (otherwise missing
        # calls are imputed by the reader): standardise over the called individuals.
        if args.remove_missing_genotypes:
            xs = (xs - xs.mean()) / np.sqrt(xs.var())
        ks = inputs.K[called, :][:, called]
        ls = LMM(ys, ks, X0=x0s, verbose=L.verbose, validate=False, legacy_fit=L.legacy_fit)
        if args.refit:
            ls.fit(X=xs, reml=args.reml)
        else:
            ls.fit(reml=args.reml)
        ts, ps, beta, beta_var = ls.association(xs, reml=args.reml, return_beta=True)
    else:
        if not snp_estimable(x, inputs.X0, Q=Q0):
            return None
        if args.refit:
            L.fit(X=x, reml=args.reml)
        ts, ps, beta, beta_var = L.association(x, reml=args.reml, return_beta=True)
    return beta, np.sqrt(beta_var), ts, ps


def iter_block_results(
    G: FloatArray,
    inputs: GwasInputs,
    L: LMM,
    Q0: FloatArray,
    args: argparse.Namespace,
    counts: dict[str, int],
) -> Iterator[SnpResult | None]:
    """Test a block of SNPs; yield one result per SNP, in column order, as soon as it is known.

    Parameters
    ----------
    G : (len(inputs.reader.indivs), b) array
        The SNPs of the block as columns, over all the reader's individuals,
        as `pylmm.plink_io.PlinkReader.iter_blocks` returns them (or the
        vectors of its SNP iterator stacked as columns); `inputs.keep`
        selects the tested individuals.
    inputs, L, Q0, args, counts
        As for `_test_snp`: the inputs, the null model, an orthonormal
        basis of the covariates, the options and the collinearity counter.

    Yields
    ------
    tuple or None
        ``(beta, beta_sd, t, p)`` per SNP, or None for a SNP that cannot
        be tested.

    Notes
    -----
    Unless ``--no-batch`` or ``--refit``, the complete SNPs of the block
    (no missing call) that are estimable (`pylmm.lmm.snp_estimable_columns`)
    are tested together with `LMM.association_batch`, the first time one of
    them is reached; every other SNP is tested on its own (`_test_snp`) when
    it is reached.  Nothing is computed twice.  If testing a SNP raises,
    every SNP before it has been yielded and no later one is.
    """
    use_batch = not (args.no_batch or args.refit)
    X = G[inputs.keep, :]
    complete: NDArray[np.bool_] = np.zeros(X.shape[1], dtype=bool)
    if use_batch:
        complete = np.asarray(~np.isnan(X).any(axis=0), dtype=bool)
    batched: dict[int, int] = {}  # column -> its index in the batch result
    result: AssociationResult | None = None
    for j in range(X.shape[1]):
        if not complete[j]:
            yield _test_snp(G[:, j], inputs, L, Q0, args, counts)
            continue
        if result is None:  # the first complete SNP: test them all together
            cols = np.where(complete)[0]
            cols = cols[snp_estimable_columns(X[:, cols], inputs.X0, Q0)]
            batched = {int(c): k for k, c in enumerate(cols)}
            result = L.association_batch(X[:, cols], reml=args.reml)
        k = batched.get(j)
        if k is None:  # constant or collinear with the covariates
            yield None
        elif np.isfinite(result.t[k]):
            yield result.beta[k], np.sqrt(result.beta_var[k]), result.t[k], result.p[k]
        else:  # not expected for an estimable SNP; keep the per-SNP answer
            yield _test_snp(G[:, j], inputs, L, Q0, args, counts)


def _log_progress(count: int, b: int) -> None:
    """Log ``At SNP k`` for every multiple k of 1000 among SNPs ``count + 1 .. count + b``."""
    for k in range((count // 1000 + 1) * 1000, count + b + 1, 1000):
        log.info("At SNP %d", k)


def scan(inputs: GwasInputs, L: LMM, args: argparse.Namespace, out: IO[str]) -> int:
    """Test every SNP of ``inputs.reader`` and write a result line for each.

    The SNPs are read in blocks of ``--block-size``
    (`pylmm.plink_io.PlinkReader.iter_blocks`; a BED file is decoded a block
    at a time) and tested together (`iter_block_results`); each line is
    written, in file order, as soon as its result is known.  The SNP IDs are
    the block's ``variants.snp`` (``SNP_<k>`` for EMMA input).  A SNP that
    cannot be tested (constant, collinear with the covariates, or too few
    called individuals) gets a line of ``nan``.

    On an error, the lines written are those of 1.0.0's one-SNP-at-a-time
    scan: if the reader fails (e.g. a truncated file), the SNPs read before
    the failure are tested and written, then the error is raised; if testing
    a SNP fails, every SNP before it has its line and no later one does
    (with ``--no-batch``, byte for byte the 1.0.0 output).  No SNP is tested
    twice and no line is written twice.  On an interrupt (Ctrl-C) or a failed
    write, the lines already written stay; the SNPs of the current block that
    were read but not yet written get none.

    A block is read whole, so a reader error inside a block (an `InputError`
    such as a truncated BED file or a bad TPED or EMMA line, or an `OSError`)
    loses the block's earlier SNPs.  The scan then reads the rest of the
    file one SNP at a time (`_scan_snps`, buffered into blocks of
    ``--block-size`` as before), from the first SNP of that block: the SNPs
    before the error are tested and written and the error is raised, as
    above (an error that does not recur there, e.g. a transient one, is
    logged as a warning and the scan continues).  If the SNP metadata
    `iter_blocks` needs cannot be parsed (a ``.bim`` line with fewer than 6
    fields, which `iter_snps` reads), the whole scan reads one SNP at a
    time, with a warning at the start.  Any other exception from the block
    reader (a bug, not a data or I/O error) is raised as is.

    Returns
    -------
    int
        The number of SNPs.
    """
    X0 = inputs.X0
    Q0 = linalg.qr(X0, mode="economic")[0] if X0.shape[1] else np.zeros((X0.shape[0], 0))
    counts: dict[str, int] = {}
    count = 0

    write_header(out)
    error: Exception | None = None
    blocks: Iterator[tuple[FloatArray, Variants]] | None = None
    try:
        blocks = inputs.reader.iter_blocks(args.block_size)
    except _READ_ERRORS as e:  # e.g. a .bim the variant metadata cannot parse
        log.warning("%s; reading the SNPs one at a time (slower)", e)
        return _finish(_scan_snps(inputs, L, args, out, Q0, counts, 0), counts)
    try:
        while blocks is not None:
            # Only the reader is guarded: an error in a test or a write is raised as is.
            try:
                G, variants = next(blocks)
            except StopIteration:
                break
            except _READ_ERRORS as e:
                error = e
                break
            _log_progress(count, G.shape[1])
            count += G.shape[1]
            _write_block(out, G, variants.snp, inputs, L, Q0, args, counts)
            del G, variants  # before the next block is decoded
    finally:
        close = getattr(blocks, "close", None)
        if close is not None:
            close()
    if error is not None:
        # Read the SNPs of the failed block one at a time, so that every SNP
        # before the error is written (and the error raised) as before blocks.
        restart = count + 1
        log.debug(
            "reading the block of SNP %d failed (%r): reading one SNP at a time", restart, error
        )
        count = _scan_snps(inputs, L, args, out, Q0, counts, count)
        log.warning(
            "reading SNPs in blocks failed from SNP %d (%s), but reading them one at a time "
            "worked; the results are complete",
            restart,
            error,
        )
    return _finish(count, counts)


_READ_ERRORS = (samples.InputError, OSError)
"""The block reader's data and I/O errors, after which `scan` reads one SNP at a time."""


def _finish(count: int, counts: dict[str, int]) -> int:
    """Log the count of SNPs that are nan for lack of rank; return ``count``."""
    if counts.get("collinear"):
        log.info(
            "%d of %d SNPs are nan because, on the individuals with a called genotype, "
            "the covariates are no longer of full rank",
            counts["collinear"],
            count,
        )
    return count


def _write_block(
    out: IO[str],
    G: FloatArray,
    snp_ids: Any,
    inputs: GwasInputs,
    L: LMM,
    Q0: FloatArray,
    args: argparse.Namespace,
    counts: dict[str, int],
) -> None:
    """Test the SNPs of a block (`iter_block_results`) and write their lines in order."""
    results = iter_block_results(G, inputs, L, Q0, args, counts)
    for snp_id, result in zip(snp_ids, results):
        if result is None:
            write_result(out, snp_id, np.nan, np.nan, np.nan, np.nan)
        else:
            write_result(out, snp_id, *result)


def _scan_snps(
    inputs: GwasInputs,
    L: LMM,
    args: argparse.Namespace,
    out: IO[str],
    Q0: FloatArray,
    counts: dict[str, int],
    start: int,
) -> int:
    """`scan` from SNP ``start + 1``, reading one SNP at a time (`iter_snps`); the SNP count.

    The fallback of `scan` when a block cannot be read: the SNPs are
    buffered into blocks of ``--block-size`` and tested as in `scan`; if the
    reader fails, the SNPs read before the failure are tested and written,
    then the error is raised.
    """
    snps: Iterator[tuple[FloatArray, str]] = iter(inputs.reader)
    # Skip the SNPs already written from blocks; unstandardised, which is
    # most of the cost of reading a SNP.
    reader = inputs.reader
    norm = reader.norm_genotype
    reader.norm_genotype = False
    try:
        for _ in range(start):
            if next(snps, None) is None:
                return start
    finally:
        reader.norm_genotype = norm
    count = start

    def test_and_write(block: list[tuple[FloatArray, str]]) -> None:
        if block:
            G = np.column_stack([snp for snp, _ in block])
            _write_block(out, G, [snp_id for _, snp_id in block], inputs, L, Q0, args, counts)

    while True:
        block: list[tuple[FloatArray, str]] = []
        try:
            while len(block) < args.block_size:
                try:
                    snp, snp_id = next(snps)
                except StopIteration:
                    break
                count += 1
                if count % 1000 == 0:
                    log.info("At SNP %d", count)
                block.append((snp, snp_id))
        except Exception:
            # The reader failed (e.g. a truncated file): write the SNPs read
            # before it, as the one-SNP-at-a-time scan did, then report it.
            test_and_write(block)
            raise
        test_and_write(block)
        if len(block) < args.block_size:
            break
    return count


# ---------------------------------------------------------------------------
# leave one chromosome out (--loco)
# ---------------------------------------------------------------------------

_MAP_CHROM_COLUMNS = ("chr", "chrom", "CHR", "#CHROM", "chromosome", "CHROM")
"""Header names of the chromosome column of an ``--emma-map`` file."""


LOCO_MAX_CHROMS = 100
"""Default of ``--loco-max-chroms``: more "chromosomes" than any karyotype that is
commonly analysed (human 22 + X, Y, XY, MT; mouse 19 + X, Y, MT; rat, dog,
horse, cattle, chicken and the major crops all have fewer than 100 codes),
so a larger count almost always means contigs or scaffolds, or a map read
from the wrong column.  Each chromosome costs an ``n x n`` matrix and an
eigendecomposition, so the count is also the cost."""

_LARGE_LOCO_BYTES = 2e9
"""``--loco`` logs its memory estimate as a warning above this many bytes."""


def _format_bytes(b: float) -> str:
    return "%.1f GB" % (b / 1e9) if b >= 1e9 else "%.0f MB" % (b / 1e6)


def read_emma_map(fn: str, m: int) -> tuple[NDArray[np.object_], bool]:
    """The chromosome of each of the ``m`` rows of an EMMA genotype file, from ``--emma-map``.

    The file has one line per EMMA row, in order: either a header line that
    names the chromosome column (`_MAP_CHROM_COLUMNS`; e.g.
    ``data/hmdp.liver.snps.map.tsv``, whose ``row`` column, if present, must
    count 1, 2, ...), or no header and the chromosome in the first column (a
    PLINK ``.map`` or ``.bim``).

    Returns
    -------
    chroms : numpy.ndarray
        The labels as written (`loco_chromosomes` normalises them).
    has_header : bool
        Whether a header named the column.

    Raises
    ------
    pylmm.samples.InputError
        Naming the file, if it cannot be read, has another number of lines
        than ``m``, or its ``row`` column is out of order.
    """
    try:
        with open(fn) as f:
            lines = [line.split() for line in f if line.strip()]
    except (OSError, UnicodeDecodeError) as e:
        raise samples.InputError("--emma-map %s: cannot read it (%s)" % (fn, e)) from e
    if not lines:
        raise samples.InputError("--emma-map %s is empty" % fn)
    header = lines[0]
    col = next((header.index(c) for c in _MAP_CHROM_COLUMNS if c in header), None)
    rows = lines[1:] if col is not None else lines
    col = 0 if col is None else col
    if len(rows) != m:
        raise samples.InputError(
            "--emma-map %s has %d SNP lines but the EMMA genotype file has %d rows (one line "
            "per row, in order%s)"
            % (fn, len(rows), m, ", after the header" if rows is not lines else "")
        )
    short = next((k for k, r in enumerate(rows) if len(r) <= col), None)
    if short is not None:
        raise samples.InputError(
            "--emma-map %s, SNP line %d: no chromosome column (%d fields)"
            % (fn, short + 1, len(rows[short]))
        )
    if rows is not lines and "row" in header:
        rc = header.index("row")
        bad = next((k for k, r in enumerate(rows) if len(r) <= rc or r[rc] != str(k + 1)), None)
        if bad is not None:
            raise samples.InputError(
                "--emma-map %s: SNP line %d has row %r, expected %d (one line per EMMA row, in "
                "order)" % (fn, bad + 1, rows[bad][rc] if len(rows[bad]) > rc else "", bad + 1)
            )
    return np.array([r[col] for r in rows], dtype=object), rows is not lines


def loco_chromosomes(args: argparse.Namespace, reader: plink_io.PlinkReader) -> NDArray[np.object_]:
    """The chromosome of each SNP for ``--loco``, normalised and checked.

    From the ``.bim`` / ``.map`` / ``.pvar`` file (`pylmm.plink_io.PlinkReader.variants`),
    or for EMMA input from ``--emma-map``.  Labels are normalised
    (`pylmm.loco.normalize_chrom`: ``chr1``, ``01`` and ``1`` are one
    chromosome, with a warning when a file mixes spellings).  Logs the
    chromosomes and the memory estimate (`pylmm.loco.loco_memory_bytes`; a
    warning above 2 GB).

    Raises
    ------
    UsageError
        If there is no chromosome information, a chromosome's SNPs are not
        together in the file, there are fewer than two chromosomes, more
        than ``--loco-max-chroms``, or every SNP is on a chromosome of its
        own (a map read from the wrong column).
    """
    has_header = True
    if reader.type == "emma":
        raw, has_header = read_emma_map(args.emma_map, len(reader.variants))
    else:
        variants = _usage(lambda: reader.variants)
        if variants.chrom is None:
            raise UsageError("--loco needs the chromosome of each SNP; the genotype input has none")
        raw = np.asarray(variants.chrom, dtype=object)
    labels = [str(c) for c in raw]
    merged = merged_chrom_labels(labels)
    if merged:
        log.warning(
            "--loco: chromosome labels written differently are treated as one chromosome: %s",
            describe_merged(merged),
        )
    chroms = np.array([normalize_chrom(c) for c in labels], dtype=object)
    order: list[str] = []
    if len(chroms):
        starts = np.flatnonzero(np.r_[True, chroms[1:] != chroms[:-1]])
        order = [str(chroms[i]) for i in starts]
    if len(set(order)) != len(order):
        again = next(c for k, c in enumerate(order) if c in order[:k])
        raise UsageError(
            "--loco: the SNPs of chromosome %s are not together in the genotype file (each "
            "chromosome's SNPs must be one run; sort them, e.g. plink2 --sort-vars)" % again
        )
    if len(order) < 2:
        raise UsageError(
            "--loco needs SNPs on at least two chromosomes; the genotype input has %s"
            % ("chromosome %s only" % order[0] if order else "no SNP")
        )
    where = (
        "; --emma-map %s has no header, so its first column is read as the chromosome: if it "
        "holds SNP IDs, add a header line naming the chromosome column (chr)" % args.emma_map
        if not has_header
        else ""
    )
    if len(order) == len(chroms) and len(chroms) > 2:
        raise UsageError(
            "--loco: every one of the %d SNPs is on a chromosome of its own (%s, ...): the "
            "chromosome column is probably something else%s"
            % (len(chroms), ", ".join(order[:3]), where)
        )
    n = len(reader.indivs)
    need = loco_memory_bytes(n, len(order))
    if len(order) > args.loco_max_chroms:
        raise UsageError(
            "--loco: %d chromosomes (%s, ...), more than --loco-max-chroms %d.  Each needs an "
            "n x n matrix (about %s in all for n = %d) and its own eigendecomposition.  Check "
            "the chromosome column%s, give unplaced contigs one label, or raise "
            "--loco-max-chroms"
            % (
                len(order),
                ", ".join(order[:5]),
                args.loco_max_chroms,
                _format_bytes(need),
                n,
                where,
            )
        )
    sizes = np.diff(np.r_[starts, len(chroms)])
    tiny = int((sizes < 2).sum())
    if tiny > len(order) // 2:
        log.warning(
            "--loco: %d of the %d chromosomes have fewer than 2 SNPs: are the chromosome "
            "labels right%s?",
            tiny,
            len(order),
            where,
        )
    shown = ", ".join(order[:30]) + (", ..." if len(order) > 30 else "")
    log.info("--loco: %d SNPs on %d chromosomes (%s)", len(chroms), len(order), shown)
    (log.warning if need > _LARGE_LOCO_BYTES else log.info)(
        "--loco: the kinships of %d chromosomes for %d individuals need about %s of memory "
        "(one n x n matrix per chromosome) and %d eigendecompositions",
        len(order),
        n,
        _format_bytes(need),
        len(order),
    )
    return chroms


def compute_loco_kinship(
    args: argparse.Namespace, reader: plink_io.PlinkReader, chroms: NDArray[np.object_]
) -> LocoKinship:
    """Read the genotypes once and build the leave-one-chromosome-out kinships.

    The raw genotypes of every individual of ``reader`` (as ``pylmm
    kinship`` reads them: the kinship of ``--kinship-type`` of the SNPs
    of the other chromosomes, the K ``pylmm kinship`` would write for them,
    to rounding; ``centered`` counts 0/1/2 copies), read in blocks of
    ``--block-size``.

    Raises
    ------
    UsageError
        For an IBS genotype that is not 0, 0.5 or 1, or a chromosome
        without a usable SNP outside it.
    """
    kind: KinshipKind = args.kinship_type or "standardized"
    loco = LocoKinship(
        len(reader.indivs),
        kind=kind,
        scale=2.0 if kind == "centered" else 1.0,  # 0/0.5/1 fractions -> 0/1/2 counts
    )
    log.info("Computing the leave-one-chromosome-out kinships (%s)...", kind)
    begin = time.time()
    norm = reader.norm_genotype
    reader.norm_genotype = False  # raw genotypes; iter_blocks captures it at the call
    try:
        blocks = reader.iter_blocks(args.block_size)
    finally:
        reader.norm_genotype = norm
    start = 0
    for G, _ in blocks:
        b = G.shape[1]
        try:
            loco.add_block(G, chroms[start : start + b])
        except ValueError as e:  # an IBS genotype that is not 0, 0.5 or 1
            raise UsageError(
                "--kinship-type %s, SNPs %d-%d: %s" % (kind, start + 1, start + b, e)
            ) from e
        start += b
        del G
    if start != len(chroms):  # the reader and the variant metadata disagree
        raise samples.InputError(
            "read %d SNPs from the genotype file but its SNP list has %d" % (start, len(chroms))
        )
    if loco.n_all_missing:
        log.warning("skipped %d SNPs with no called genotype", loco.n_all_missing)
    skipped = loco.n_snps - loco.n_used - loco.n_all_missing
    if skipped:
        log.info("skipped %d constant SNPs", skipped)
    for c in loco.chromosomes:
        if loco.snps_used(leave_out=c) == 0:
            raise UsageError(
                "--loco: no usable SNP outside chromosome %s, so no kinship without it" % c
            )
    log.info(
        "Built the kinships of %d SNPs (%d used) in %0.3fs",
        loco.n_snps,
        loco.n_used,
        time.time() - begin,
    )
    return loco


def _loco_model(inputs: GwasInputs, args: argparse.Namespace, chrom: str) -> tuple[GwasInputs, LMM]:
    """The inputs and the null model for chromosome ``chrom``: K = K_{-chrom}.

    The caller drops the previous chromosome's inputs and model first, so
    that one ``K_{-c}`` (and its eigendecomposition) is held at a time.
    """
    assert inputs.loco is not None
    K = inputs.loco.kinship(leave_out=chrom)
    if not inputs.keep.all():
        K = K[inputs.keep][:, inputs.keep]
    scale = kinship_scale(K)
    log.info(
        "chromosome %s: K from the %d SNPs of the other chromosomes (mean diagonal %0.4f, "
        "centred %0.4f)",
        chrom,
        inputs.loco.snps_used(leave_out=chrom),
        scale.mean_diag,
        scale.centered_mean_diag,
    )
    inputs_c = dataclasses.replace(inputs, K=K, kva=None, kve=None)
    return inputs_c, fit_null_model(inputs_c, args)


def scan_loco(inputs: GwasInputs, args: argparse.Namespace, out: IO[str]) -> int:
    """`scan` with ``--loco``: each chromosome's SNPs tested with its own K and null fit.

    The SNPs are read once more, in blocks of ``--block-size``, and written
    in file order, with the columns of `scan`.  When a chromosome starts, its
    kinship ``K_{-c}`` (`pylmm.loco.LocoKinship.kinship`, restricted to the
    tested individuals) is eigendecomposed and the null model fitted
    (`fit_null_model`, so ``--ml``, ``--refit`` and the fit options apply
    per chromosome); the SNPs are then tested as by `scan`
    (`iter_block_results`).  Unlike `scan`, a reader error is raised as is,
    without a one-SNP-at-a-time retry; in practice the kinship pass
    (`compute_loco_kinship`) reads every SNP first, so a truncated or
    malformed genotype file aborts there, before any output is written
    (and a ``.bim`` line `iter_blocks` cannot parse is an error, where
    `scan` falls back to reading one SNP at a time).

    Returns
    -------
    int
        The number of SNPs.
    """
    assert inputs.chroms is not None
    X0 = inputs.X0
    Q0 = linalg.qr(X0, mode="economic")[0] if X0.shape[1] else np.zeros((X0.shape[0], 0))
    counts: dict[str, int] = {}
    count = 0
    current: str | None = None
    inputs_c: GwasInputs = inputs
    L: LMM | None = None
    write_header(out)
    blocks = inputs.reader.iter_blocks(args.block_size)
    try:
        for G, variants in blocks:
            b = G.shape[1]
            cs = inputs.chroms[count : count + b]
            a = 0
            while a < b:  # runs of one chromosome within the block
                e = a + 1
                while e < b and cs[e] == cs[a]:
                    e += 1
                if cs[a] != current:
                    current = str(cs[a])
                    # release the previous chromosome's K and model first
                    inputs_c, L = inputs, None
                    inputs_c, L = _loco_model(inputs, args, current)
                assert L is not None
                _write_block(out, G[:, a:e], variants.snp[a:e], inputs_c, L, Q0, args, counts)
                a = e
            _log_progress(count, b)  # after the block's chromosome lines, not before
            count += b
            del G, variants
    finally:
        close = getattr(blocks, "close", None)
        if close is not None:
            close()
    return _finish(count, counts)


def run(args: argparse.Namespace) -> None:
    """Run ``pylmm gwas`` with parsed options.

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
    inputs = load_inputs(args)
    # With --loco the null model is fitted per chromosome (scan_loco).
    L = fit_null_model(inputs, args) if inputs.loco is None else None
    try:
        out = open(args.out, "w")
    except OSError as e:
        raise UsageError("--out %s: cannot write it (%s)" % (args.out, e)) from e
    with out:
        if L is None:
            n = scan_loco(inputs, args, out)
        else:
            n = scan(inputs, L, args, out)
    log.info("Wrote the results for %d SNPs to %s", n, args.out)


if __name__ == "__main__":  # the removed pre-5b.2 invocation; see removed_module_main
    from pylmm.cli._common import removed_module_main

    raise SystemExit(removed_module_main("gwas"))
