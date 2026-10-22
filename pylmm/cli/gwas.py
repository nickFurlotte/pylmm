# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""``pylmm gwas``: genome-wide association with a linear mixed model.

Reads genotypes (PLINK BED or TPED, or an EMMA matrix), a phenotype, optional
covariates and a precomputed kinship (``pylmm kinship``), fits the null model
once (or per SNP with ``--refit``) and writes one line per SNP: by default a
summary-statistics table of per-allele effects (`pylmm.sumstats`; gzip if the
``-o`` file ends in ``.gz``), with ``--output-format legacy`` the 1.0.0 columns
``SNP_ID BETA BETA_SD F_STAT P_VALUE``; and a JSON run record next to it
(`pylmm.cli.run_record`).  With ``--loco`` the kinships are
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
import stat
import time
import warnings
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from pylmm import kinship_io, plink_io, samples, sumstats
from pylmm.cli import run_record
from pylmm.cli._common import (
    UsageError,
    add_logging_options,
    check_log_file,
    check_writable,
    genotype_source,
    log,
    positive_int,
)
from pylmm.genotypes import DEFAULT_BLOCK_SIZE, Variants, check_block_size, normalize_block
from pylmm.kinship import KINSHIP_KINDS, KinshipKind, kinship_scale
from pylmm.lmm import (
    LMM,
    TESTS,
    AssociationResult,
    check_covariates,
    check_test,
    neg_log10_p_for,
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
"""The header of the ``--output-format legacy`` output file (tab-separated)."""

OUTPUT_FORMATS = ("sumstats", "legacy")
"""``--output-format`` values: the summary-statistics table (`pylmm.sumstats`), the default,
or pylmm 1.0.0's five columns."""

ID_MATCH_CHOICES = {"fid-iid": "fid_iid", "iid": "iid"}
"""``--id-match`` values and the `pylmm.plink_io.PlinkReader` ``id_match`` they select."""

_DESCRIPTION = """\
Test each SNP for association with a phenotype under a linear mixed model.

The kinship (--kinship, computed with `pylmm kinship`, or a GCTA GRM with
--grm) accounts for relatedness.  Phenotype, covariate and kinship rows are
matched to the genotype file by sample ID (FID and IID, or IID alone with
--id-match iid), never by position; the kinship IDs come from its
`<kinship>.id` file (`<prefix>.grm.id` for --grm).

The output (-o FILE; gzip-compressed if FILE ends in .gz) is a tab-separated
summary-statistics table, one line per SNP, in the column order of the GWAS
Catalog format (GWAS-SSF): chromosome, base_pair_location, effect_allele,
other_allele, beta, standard_error, effect_allele_frequency, p_value, then
variant_id, rsid, n, t_stat, neg_log_10_p_value, missing_rate and beta_std.
beta is the effect of one copy of the effect allele, the .bim A1 allele (ALT
for plink2 files; for EMMA input the allele coded 1).  Missing values are NA.
A JSON run record (inputs and their checksums, samples, null fit, lambda_GC,
warnings, timings) is written next to it: FILE without .gz/.tsv, plus .json.
--output-format legacy writes pylmm 1.0.0's columns instead: SNP_ID, BETA
(per SD of the A2 allele's dosage), BETA_SD (its standard error), F_STAT (the
t statistic) and P_VALUE."""

_EPILOG = """\
examples:
  # BED genotypes, the README example data: results.tsv.gz and results.json
  pylmm gwas --bfile data/snps.132k.clean.noX \\
      --kinship data/snps.132k.clean.noX.pylmm.kin \\
      --pheno data/snps.132k.clean.noX.fake.phenos -o results.tsv.gz

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

  # the Wald test at each SNP's own h, or a likelihood-ratio test (GEMMA's -lmm 1, 2)
  pylmm gwas --bfile study --kinship study.kin --pheno study.phen --refit -o refit.tsv.gz
  pylmm gwas --bfile study --kinship study.kin --pheno study.phen --test lrt -o lrt.tsv.gz

  # the exact output of pylmm 1.0.0 (its columns, its h search and its
  # one-SNP-at-a-time path)
  pylmm gwas --bfile study --kinship study.kin --pheno study.phen \\
      --output-format legacy --legacy-fit --no-batch -o results.tsv

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

    geno.add_argument(
        "--emma-coding",
        choices=("fraction", "count"),
        default="fraction",
        help="what an --emma-geno value counts, for the per-allele output columns: the "
        "fraction of copies of the allele coded 1, 0 / 0.5 / 1 as in EMMA and pylmm's "
        "example data (the default), or the number of copies, 0 / 1 / 2.  The test does "
        "not depend on it",
    )

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
        "With --bfile/--tfile its rows are matched to the genotypes by the IDs in FILE.id if "
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
        "chromosome (the .bim or .map file; --emma-map for --emma-geno; labels such as chr1, "
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
        help="with --emma-geno: the marker map of the EMMA rows, one line per row in order: "
        "a header line naming a chr (or chrom, CHR, #CHROM) column and optionally a position "
        "(bp, bp_b37, pos, POS, base_pair_location) and the alleles coded 0 and 1 (allele0, "
        "allele1), as data/hmdp.liver.snps.map.tsv; or no header, as a PLINK .map (chromosome, "
        "ID, cM, position).  Needed by --loco (the chromosomes); otherwise it fills the "
        "chromosome, position and allele columns of the output",
    )

    model = p.add_argument_group("model")
    method = model.add_mutually_exclusive_group()
    method.add_argument(
        "--reml",
        dest="reml",
        action="store_true",
        default=True,
        help="restricted maximum likelihood, for the null fit, the per-SNP fits and the test "
        "(the default; --test lrt is always ML)",
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
        help="refit the variance components at each SNP, exactly (Newton-Raphson on the "
        "likelihood's derivatives, a block of SNPs at a time; with --no-batch or --legacy-fit, "
        "the grid search of the null fit per SNP).  Default: fit them once, under the null "
        "model.  --test lrt always fits each SNP's model",
    )
    model.add_argument(
        "--test",
        choices=TESTS,
        default="wald",
        help="the test of each SNP: wald (the Wald t test, the default; with --refit, GEMMA's "
        "-lmm 1) or lrt (likelihood ratio of ML fits with and without the SNP, chi2(1), with "
        "the null directions of the kinship that the covariates span left out, so that ML "
        "has a maximum; it always fits each SNP's model by ML, so --reml, --ml and --refit do "
        "not apply).  With lrt, t_stat (legacy F_STAT) is the signed square root of the "
        "statistic (the sign of beta) and beta, standard_error (BETA, BETA_SD) are the ML fit's",
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
        "is about 6-8 x n x N floats, also with --refit and --test lrt (default: %d)"
        % DEFAULT_BLOCK_SIZE,
    )
    performance.add_argument(
        "--no-batch",
        action="store_true",
        help="test one SNP at a time (LMM.association), the code path of pylmm 1.0.0: slower.  "
        "The numbers are the same up to rounding (within about 1e-9 relative); with "
        "--legacy-fit and a badly scaled covariate (e.g. a raw date or year column) they can "
        "differ by a few 1e-6 relative, the batched default being the more accurate: centre "
        "or scale such covariates.  SNPs with missing calls, and --refit with --legacy-fit, "
        "always take this path; --refit and --test lrt then fit each SNP with the grid search "
        "of the null fit",
    )

    output = p.add_argument_group("output")
    output.add_argument(
        "-o",
        "--out",
        metavar="FILE",
        required=True,
        help="(required) write the results to FILE (overwritten; gzip-compressed if FILE ends "
        "in .gz, e.g. results.tsv.gz), and the JSON run record to FILE without .gz and "
        ".tsv/.txt, plus .json (results.json)",
    )
    output.add_argument(
        "--output-format",
        choices=OUTPUT_FORMATS,
        default="sumstats",
        help="sumstats (the default): the GWAS-SSF-ordered table of per-allele effects of the "
        "A1 allele described above; legacy: pylmm 1.0.0's SNP_ID BETA BETA_SD F_STAT P_VALUE, "
        "with BETA per SD of the A2 allele's dosage (as before; with "
        "--remove-missing-genotypes per 0-1 step of the A2 fraction for SNPs without a missing "
        "call)",
    )
    record = output.add_mutually_exclusive_group()
    record.add_argument(
        "--run-record",
        metavar="PATH",
        help="write the JSON run record to PATH (default: FILE without .gz and .tsv/.txt, "
        "plus .json, when FILE is a regular file; none when it is not, e.g. /dev/stdout or a "
        "pipe).  -o x.tsv.gz, -o x.tsv and -o x all have the record x.json; an existing "
        "record is overwritten, with a warning",
    )
    record.add_argument(
        "--no-run-record",
        action="store_true",
        help="do not write the JSON run record",
    )
    output.add_argument(
        "--run-record-full-paths",
        action="store_true",
        help="record the input file names and the command line as typed (full paths); by "
        "default the record keeps only their base names, so it can be shared without "
        "revealing your directory layout (it has no host or user name either)",
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
            ("--loco-max-chroms", args.loco_max_chroms),
        ):
            if value is not None:
                parser.error("%s applies only to --loco" % flag)
    if args.emma_coding != "fraction" and not args.emma_geno:
        parser.error("--emma-coding applies only to --emma-geno")
    if args.emma_map and not args.emma_geno:
        parser.error("--emma-map applies only to --emma-geno (PLINK input has a .bim or .map)")
    if args.grm is not None:
        # The GRM is read as a kinship file named PREFIX.grm.bin (kinship_io.kinship_format).
        args.kinship = kinship_io.grm_prefix(args.grm) + ".grm.bin"
    try:
        check_block_size(args.block_size)
    except ValueError as e:
        parser.error("--block-size: %s" % e)
    try:
        check_test(args.test, args.refit, args.legacy_fit)
    except ValueError:
        parser.error("--test %s: --legacy-fit (pylmm 1.0.0) has only the Wald test" % args.test)
    check_log_file(parser, args)
    record = record_path(args)
    if record is not None and os.path.realpath(record) == os.path.realpath(args.out):
        parser.error("--run-record %s is also the -o/--out file" % record)
    same_as_log = args.log_file and os.path.realpath(args.log_file) == os.path.realpath(
        record or ""
    )
    if record is not None and same_as_log:
        parser.error("--log-file %s is also the run record" % args.log_file)


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
    emma_variants : pylmm.genotypes.Variants or None
        For ``--emma-geno`` with ``--emma-map``: the map's SNP metadata (``a2``
        is the allele coded 1, ``a1`` the allele coded 0).
    sample_counts : dict
        The ``samples`` part of the run record (`pylmm.cli.run_record.SampleCounts`).
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
    emma_variants: Variants | None = None
    sample_counts: dict[str, Any] = dataclasses.field(default_factory=dict)
    covariate_names: list[str] | None = None


def _usage(fn: Any, *a: Any, **kw: Any) -> Any:
    """Call ``fn``, turning a ValueError (invalid input) into a `UsageError`."""
    try:
        return fn(*a, **kw)
    except ValueError as e:
        raise UsageError(str(e)) from e


def open_genotypes(args: argparse.Namespace) -> plink_io.PlinkReader:
    """Open the genotype input (``--bfile``, ``--tfile`` or ``--emma-geno``).

    The reader also reads the ``--pheno`` file (or ``PREFIX.phenos``).  It
    returns raw genotypes (``norm_genotype=False``): the scan standardises
    each block itself (`pylmm.genotypes.normalize_block`, the same values),
    after reading the allele frequencies from it.

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
            norm_genotype=False,
            id_match=ID_MATCH_CHOICES[args.id_match],
        )
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


def _covariate_names(
    args: argparse.Namespace, reader: plink_io.PlinkReader, P: FloatArray | None
) -> list[str] | None:
    """The covariates used (without the intercept), for the run record; None without any."""
    if P is None:
        return None
    if args.covar_name:
        return list(args.covar_name)
    names = reader.covariate_names if args.covar else None
    if names is not None and len(names) == P.shape[1]:
        return list(names)
    return ["column %d" % (k + 1) for k in range(P.shape[1])]


def _design_matrix(n: int, P: FloatArray | None, no_intercept: bool) -> FloatArray:
    """The covariate matrix X0: an intercept (unless ``no_intercept``), then the covariates ``P``."""
    if P is None:
        return np.ones((n, 1))
    if no_intercept:
        return P
    return np.hstack([np.ones((n, 1)), P])


def _drop_incomplete(
    Y: FloatArray, X0: FloatArray, K: FloatArray, dropped_counts: dict[str, int] | None = None
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
    """Drop the individuals with a missing phenotype or covariate, with a warning.

    (Before 1.0.0 a missing covariate was a fatal error.)  The counts are
    added to ``dropped_counts`` (``missing_phenotype``, ``missing_covariate``).

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
    if dropped_counts is not None:
        dropped_counts["missing_phenotype"] = int(missing_y.sum())
        dropped_counts["missing_covariate"] = int((missing_cov & ~missing_y).sum())
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
    n_genotyped = len(reader.indivs)
    emma_variants = None
    if args.emma_map:
        emma_variants = _usage(read_emma_map_variants, args.emma_map, len(reader.variants))

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

    dropped: dict[str, int] = {}
    Y, X0, K, keep = _drop_incomplete(reader.phenos[:, column], X0, K, dropped)
    name = "the covariates" + (
        ""
        if args.no_intercept or (P is None)
        else " (column 0 is the intercept pylmm gwas adds; use --no-intercept if your file has its own)"
    )
    _usage(check_covariates, X0, what=name)
    if chroms is not None:
        loco = compute_loco_kinship(args, reader, chroms)

    kva, kve = _eigen(args, K, reader, keep, kinship_aligned)
    names = reader.pheno_names
    sample_counts = {
        "id_match": None if reader.type == "emma" else ID_MATCH_CHOICES[args.id_match],
        "n_genotyped": n_genotyped,
        "n_analysed": len(Y),
        "dropped": {"not_in_kinship": n_genotyped - n_all, **dropped},
        "phenotype": names[column] if names is not None and not args.emma_pheno else None,
    }
    return GwasInputs(
        reader=reader,
        Y=Y,
        X0=X0,
        K=K,
        keep=keep,
        kva=kva,
        kve=kve,
        loco=loco,
        chroms=chroms,
        emma_variants=emma_variants,
        sample_counts=sample_counts,
        covariate_names=_covariate_names(args, reader, P),
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
    always REML).  With ``--test lrt``, the LRT's ML null fit
    (`pylmm.lmm.LMM.lrt_null_fit`).

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
    if args.test == "lrt":
        log.info(
            "--test lrt: maximum-likelihood fits with and without each SNP (--reml/--ml and "
            "--refit do not apply)"
        )
        h, _ = L.lrt_null_fit()
        dropped = L._lrt_model()._dropped
        log.info(
            "null model (ML%s): heritability=%0.3f",
            ", %d null direction(s) of the kinship in the covariates' span left out" % dropped
            if dropped
            else "",
            h,
        )
    elif not args.refit:
        log.info("Computing fit for null model")
        L.fit(reml=args.reml)
        log.info("null model: heritability=%0.3f, sigma=%0.3f", L.opt_h, L.opt_sigma)
    return L


def write_header(out: IO[str]) -> None:
    """Write the output header line (`OUTPUT_COLUMNS`, tab-separated)."""
    out.write("\t".join(OUTPUT_COLUMNS) + "\n")


def write_result(out: IO[str], snp_id: str, beta: Any, beta_sd: Any, ts: Any, ps: Any) -> None:
    """Write one SNP's legacy result line; the values are formatted with `str`."""
    out.write("\t".join([str(x) for x in [snp_id, beta, beta_sd, ts, ps]]) + "\n")


@dataclass
class BlockStats:
    """What the output needs from a block's raw genotypes, per SNP (`ScanOutput.describe`).

    Attributes
    ----------
    scale : (b,) array
        ``s`` of the tested column ``x = (g - m) / s`` (``g`` the raw
        genotype): the SD over the reader's individuals for a standardised
        SNP; with ``--remove-missing-genotypes``, 1 for a SNP called in every
        tested individual (tested raw) and the SD over the called tested
        individuals otherwise (as `_test_snp` standardises it).
    n_used : (b,) array
        The individuals the SNP is tested on: all tested individuals (missing
        calls mean-imputed), or the called ones with ``--remove-missing-genotypes``.
    n_called : (b,) array
        Tested individuals with a called genotype.
    effect_freq, effect_sd : (b,) array
        Mean / 2 and SD of the effect allele count over the called tested
        individuals (nan without any).
    missing_rate : (b,) array
        Share of the tested individuals without a call.
    """

    scale: FloatArray
    n_used: FloatArray
    n_called: FloatArray
    effect_freq: FloatArray
    effect_sd: FloatArray
    missing_rate: FloatArray


class ScanOutput:
    """The results file of a scan, and the per-SNP tallies the run record needs.

    ``--output-format sumstats`` (`pylmm.sumstats`): per-allele effects of the
    effect allele.  pylmm's genotype ``g`` is the fraction of copies of the
    counted allele (A2; for EMMA input the allele coded 1), tested as
    ``x = (g - m) / s`` (`BlockStats.scale`).  The effect allele is A1, the
    other allele, for PLINK input, and the counted allele for EMMA input, so
    ``dx/dc = -1 / (2 s)`` (PLINK), ``+1 / (2 s)`` (EMMA, ``--emma-coding
    fraction``) or ``+1 / s`` (``--emma-coding count``) per copy ``c`` of the
    effect allele (`pylmm.sumstats.per_allele_effects`).
    ``--output-format legacy``: 1.0.0's lines (`write_result`), unchanged.

    Parameters
    ----------
    out : str
        The results file (`pylmm.sumstats.SumstatsWriter` picks the compression).
    inputs : GwasInputs
    args : argparse.Namespace
    """

    def __init__(self, out: str, inputs: GwasInputs, args: argparse.Namespace) -> None:
        self.legacy = args.output_format == "legacy"
        self.remove_missing = bool(args.remove_missing_genotypes)
        self.lrt = getattr(args, "test", "wald") == "lrt"
        self.q = int(inputs.X0.shape[1])
        self.emma = inputs.reader.type == "emma"
        self.emma_coding = getattr(args, "emma_coding", "fraction")
        # dc/dg: copies of the effect allele per unit of the raw genotype g.  PLINK: g is
        # the fraction of A2, c = 2 - 2 g copies of A1.  EMMA: g counts the allele coded 1,
        # as a fraction (c = 2 g) or as a count (c = g).
        if not self.emma:
            self.copies = -2.0
        else:
            self.copies = 2.0 if self.emma_coding == "fraction" else 1.0
        # The metadata of every SNP of the file, in file order, when known: the --emma-map, or
        # the .bim/.map (so that SNPs read one at a time after a failed block, which carry
        # only their IDs, still get positions and alleles).  None: each block's own.
        self.meta: Variants | None = inputs.emma_variants
        if self.meta is None and not self.emma and not self.legacy:
            try:
                self.meta = inputs.reader.variants
            except _READ_ERRORS:  # a .bim the metadata reader rejects: IDs only
                self.meta = None
        self.index = 0
        self.p_values: list[FloatArray] = []
        self.neg_log10: list[FloatArray] = []
        self.not_tested: Counter[str] = Counter()
        self.counts: dict[str, int] = {}  # the scan's counters (collinear, lrt_unbounded)
        self.alleles_known = False
        self.loco_fits: dict[str, dict[str, Any]] = {}
        self._warned_range = False
        self._legacy_out: IO[str] | None = None
        self._writer: sumstats.SumstatsWriter | None = None
        if self.legacy:
            self._legacy_out = open(out, "w")
            write_header(self._legacy_out)
        else:
            self._writer = sumstats.SumstatsWriter(out)

    def close(self) -> None:
        """Close the results file."""
        if self._legacy_out is not None:
            self._legacy_out.close()
        if self._writer is not None:
            self._writer.close()

    def describe(self, G: FloatArray, keep: NDArray[np.bool_]) -> BlockStats:
        """The `BlockStats` of a raw block ``G`` (all the reader's individuals; ``keep``: tested)."""
        T = G[keep]
        n_t = T.shape[0]
        # g, the fraction of the counted allele, passed as if it were a count:
        # g_half = mean(g) / 2, g_sd = sd(g)
        g_half, g_sd, missing = sumstats.allele_count_stats(T)
        with np.errstate(invalid="ignore", divide="ignore"):
            called = np.rint(n_t * (1.0 - missing)) if n_t else np.zeros(G.shape[1])
            if self.remove_missing:
                scale = np.where(called == n_t, 1.0, g_sd)
                n_used = called
            else:
                # over the reader's individuals, as the standardisation; layout-independent sums
                _, sd_all, _ = sumstats.allele_count_stats(G)
                scale = np.where(sd_all > 0, sd_all, 1.0)
                n_used = np.full(G.shape[1], float(n_t))
        mean_g = 2.0 * g_half  # the mean raw value
        top = 2.0 / abs(self.copies)  # the largest raw value: 1 (fraction) or 2 (count)
        if self.emma and not self.legacy:
            v = T[~np.isnan(T)]
            if v.size and (v.min() < 0.0 or v.max() > top):
                hint = (
                    "; pass --emma-coding count for 0 / 1 / 2 allele counts"
                    if self.emma_coding == "fraction" and v.min() >= 0.0 and v.max() <= 2.0
                    else ""
                )
                raise samples.InputError(
                    "EMMA genotypes outside 0..%g (SNPs %d-%d): --emma-coding %s reads them "
                    "as %s copies of the allele coded 1%s (or --output-format legacy, which "
                    "has no per-allele columns)"
                    % (
                        top,
                        self.index + 1,
                        self.index + T.shape[1],
                        self.emma_coding,
                        "the fraction 0 / 0.5 / 1 of"
                        if self.emma_coding == "fraction"
                        else "0 / 1 / 2",
                        hint,
                    )
                )
        # the effect allele count c = copies * g (+ 2 for PLINK's A1 = 2 - 2 g)
        freq = mean_g * self.copies / 2.0 + (0.0 if self.emma else 1.0)
        return BlockStats(
            scale=scale,
            n_used=n_used,
            n_called=called,
            effect_freq=freq,
            effect_sd=abs(self.copies) * g_sd,
            missing_rate=missing,
        )

    def write(self, variants: Variants, results: list[SnpResult | None], desc: BlockStats) -> None:
        """Write the lines of the first ``len(results)`` SNPs of a block."""
        b = len(results)
        nan = np.nan
        vals = np.array(
            [(nan, nan, nan, nan) if r is None else tuple(float(x) for x in r) for r in results],
            dtype=np.float64,
        ).reshape(b, 4)
        beta_x, se_x, t_x, p = vals.T
        for k in range(b):
            if not np.isfinite(p[k]):
                if desc.n_called[k] == 0:
                    reason = "no_called_genotype"
                elif desc.effect_sd[k] == 0:
                    reason = "monomorphic"
                else:
                    reason = "not_estimable"
                self.not_tested[reason] += 1
        nl10 = neg_log10_p_for(t_x, p, desc.n_used[:b] - self.q - 1, "lrt" if self.lrt else "wald")
        self.p_values.append(p.copy())
        self.neg_log10.append(nl10)
        if self._legacy_out is not None:
            for snp_id, r in zip(variants.snp[:b], results):
                if r is None:
                    write_result(self._legacy_out, snp_id, nan, nan, nan, nan)
                else:
                    write_result(self._legacy_out, snp_id, *r)
        else:
            assert self._writer is not None
            meta = variants[:b] if self.meta is None else self.meta[self.index : self.index + b]
            slope = 1.0 / (self.copies * desc.scale[:b])  # dx/dc for x = (g - m) / s
            beta, se, t = sumstats.per_allele_effects(beta_x, se_x, t_x, slope)
            effect = meta.a2 if self.emma else meta.a1
            if effect is not None and any(
                a is not None and a not in sumstats.MISSING_ALLELES for a in effect
            ):
                self.alleles_known = True
            self._writer.write(
                sumstats.SumStats.from_variants(
                    meta,
                    beta=beta,
                    standard_error=se,
                    t_stat=t,
                    p_value=p,
                    neg_log_10_p_value=nl10,
                    effect_allele_frequency=desc.effect_freq[:b],
                    n=np.where(np.isfinite(p), desc.n_used[:b], nan),
                    missing_rate=desc.missing_rate[:b],
                    beta_std=beta * desc.effect_sd[:b],
                    effect_allele=meta.a2 if self.emma else meta.a1,
                    other_allele=meta.a1 if self.emma else meta.a2,
                )
            )
        self.index += b

    def _not_tested(self) -> dict[str, int]:
        """SNPs without a test, by reason (`pylmm.cli.run_record.ResultsInfo`)."""
        collinear = int(self.counts.get("collinear", 0))
        unbounded = int(self.counts.get("lrt_unbounded", 0))
        other = int(self.not_tested.get("not_estimable", 0))
        return {
            "no_called_genotype": int(self.not_tested.get("no_called_genotype", 0)),
            "monomorphic": int(self.not_tested.get("monomorphic", 0)),
            "collinear_on_called": collinear,
            "lrt_unbounded": unbounded,
            "not_estimable": max(other - collinear - unbounded, 0),
        }

    def results_info(self) -> dict[str, Any]:
        """The ``results`` part of the run record (`pylmm.cli.run_record.ResultsInfo`)."""
        p = np.concatenate(self.p_values) if self.p_values else np.zeros(0)
        nl = np.concatenate(self.neg_log10) if self.neg_log10 else np.zeros(0)
        tested = np.isfinite(p)
        if self.legacy:
            effect = "A2 (BETA per SD of the A2 dosage; legacy)"
        elif self.emma:
            effect = "allele coded 1 (EMMA)"
        else:
            effect = "A1"
        if not self.legacy and not self.alleles_known:
            effect = "unknown: the input names no alleles (beta is per copy of the %s)" % (
                "allele coded 1 of the EMMA file" if self.emma else "allele coded 1 of the TPED, A1"
            )
        return {
            "path": None,
            "format": "legacy" if self.legacy else sumstats.SUMSTATS_FORMAT,
            "columns": list(OUTPUT_COLUMNS if self.legacy else sumstats.SUMSTATS_COLUMNS),
            "effect_allele": effect,
            "n_variants": int(self.index),
            "n_tested": int(tested.sum()),
            "not_tested": self._not_tested(),
            "lambda_gc": (
                run_record.finite_or_none(sumstats.lambda_gc(p[tested])) if tested.any() else None
            ),
            "max_neg_log_10_p_value": (
                run_record.finite_or_none(float(np.max(nl[tested]))) if tested.any() else None
            ),
        }


def _test_snp(
    snp: FloatArray,
    inputs: GwasInputs,
    L: LMM,
    Q0: FloatArray,
    args: argparse.Namespace,
    counts: dict[str, int],
) -> tuple[Any, Any, Any, Any] | None:
    """Test one SNP; ``(beta, beta_sd, t, p)``, or None if it is not estimable."""
    # pylmm 1.0.0's code: the Wald test, without refit or with the per-SNP grid refit
    old = args.test == "wald" and (not args.refit or args.no_batch or args.legacy_fit)
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
        if not old:
            if args.test != "lrt" and not args.refit:
                ls.fit(reml=args.reml)
            return _first(_tests(ls, xs, args, counts))
        if args.refit:
            ls.fit(X=xs, reml=args.reml)
        else:
            ls.fit(reml=args.reml)
        ts, ps, beta, beta_var = ls.association(xs, reml=args.reml, return_beta=True)
    else:
        if not snp_estimable(x, inputs.X0, Q=Q0):
            return None
        if not old:
            return _first(_tests(L, x, args, counts))
        if args.refit:
            L.fit(X=x, reml=args.reml)
        ts, ps, beta, beta_var = L.association(x, reml=args.reml, return_beta=True)
    return beta, np.sqrt(beta_var), ts, ps


def _tests(
    L: LMM, X: FloatArray, args: argparse.Namespace, counts: dict[str, int]
) -> AssociationResult:
    """The ``--test`` of the SNP columns of ``X`` (`pylmm.lmm.LMM.association_tests`)."""
    if args.test == "wald" and not args.refit:
        return L.association_batch(X, reml=args.reml)
    return L.association_tests(
        X, args.test, args.reml, args.refit, batch=not args.no_batch, counts=counts
    )


def _first(r: AssociationResult) -> SnpResult:
    """The ``(beta, beta_sd, t, p)`` of the first SNP of ``r``."""
    return r.beta[0], np.sqrt(r.beta_var[0]), r.t[0], r.p[0]


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
    Unless ``--no-batch`` (or ``--refit --legacy-fit``), the complete SNPs
    of the block (no missing call) that are estimable
    (`pylmm.lmm.snp_estimable_columns`) are tested together with
    `LMM.association_batch` (or, for ``--refit`` and ``--test lrt``,
    `LMM.association_tests`), the first time one of them is reached; every
    other SNP is tested on its own (`_test_snp`) when it is reached.
    Nothing is computed twice.  If testing a SNP raises,
    every SNP before it has been yielded and no later one is.
    """
    use_batch = not (args.no_batch or (args.refit and args.legacy_fit))
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
            result = _tests(L, X[:, cols], args, counts)
        k = batched.get(j)
        if k is None:  # constant or collinear with the covariates
            yield None
        elif np.isfinite(result.t[k]):
            yield result.beta[k], np.sqrt(result.beta_var[k]), result.t[k], result.p[k]
        elif args.test == "lrt":  # no ML maximum with the SNP: nan by design (counted)
            yield None
        else:  # not expected for an estimable SNP; keep the per-SNP answer
            yield _test_snp(G[:, j], inputs, L, Q0, args, counts)


def _log_progress(count: int, b: int) -> None:
    """Log ``At SNP k`` for every multiple k of 1000 among SNPs ``count + 1 .. count + b``."""
    for k in range((count // 1000 + 1) * 1000, count + b + 1, 1000):
        log.info("At SNP %d", k)


def scan(inputs: GwasInputs, L: LMM, args: argparse.Namespace, out: ScanOutput) -> int:
    """Test every SNP of ``inputs.reader`` and write a result line for each.

    The SNPs are read raw in blocks of ``--block-size``
    (`pylmm.plink_io.PlinkReader.iter_blocks`; a BED file is decoded a block
    at a time), described for the output (`ScanOutput.describe`: allele
    frequencies, missing calls), standardised (unless
    ``--remove-missing-genotypes``) and tested together
    (`iter_block_results`); the lines are written in file order
    (`ScanOutput.write`).  The SNP IDs are the block's ``variants.snp``
    (``SNP_<k>`` for EMMA input).  A SNP that cannot be tested (constant,
    collinear with the covariates, or too few called individuals) gets a
    line of ``nan`` (``NA`` in the summary-statistics format).

    On an error, the lines written are those of 1.0.0's one-SNP-at-a-time
    scan: if the reader fails (e.g. a truncated file), the SNPs read before
    the failure are tested and written, then the error is raised; if testing
    a SNP fails, every SNP before it has its line and no later one does
    (with ``--no-batch``, byte for byte the 1.0.0 output).  No SNP is tested
    twice and no line is written twice.  On an interrupt (Ctrl-C) or a failed
    write, the lines already written stay; so do those of the SNPs of the
    current block already tested, and the others get none.

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
    counts = out.counts
    count = 0

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
            _write_block(out, G, variants, inputs, L, Q0, args, counts)
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
    """Log the counts of SNPs that are nan for lack of rank or of an ML maximum; return ``count``."""
    if counts.get("lrt_unbounded"):
        log.warning(
            "%d of %d SNPs are nan for --test lrt: with the SNP in the model the ML likelihood "
            "has no maximum (the SNP and the covariates fit the phenotype exactly in every null "
            "direction of the kinship, e.g. a SNP whose calls differ between animals of one "
            "strain); use the Wald test for them",
            counts["lrt_unbounded"],
            count,
        )
    if counts.get("collinear"):
        log.info(
            "%d of %d SNPs are nan because, on the individuals with a called genotype, "
            "the covariates are no longer of full rank",
            counts["collinear"],
            count,
        )
    return count


def _write_block(
    out: ScanOutput,
    G: FloatArray,
    variants: Variants,
    inputs: GwasInputs,
    L: LMM,
    Q0: FloatArray,
    args: argparse.Namespace,
    counts: dict[str, int],
) -> None:
    """Test the SNPs of a raw block (`iter_block_results`) and write their lines in order.

    ``G`` (raw genotypes, all the reader's individuals) is described for the
    output first (`ScanOutput.describe`), then standardised in place, unless
    ``--remove-missing-genotypes`` (`pylmm.genotypes.normalize_block`: bit
    for bit what the reader's ``norm_genotype=True`` gives).  If testing a
    SNP raises, the lines of the SNPs before it are written first.
    """
    desc = out.describe(G, inputs.keep)
    if not args.remove_missing_genotypes:
        normalize_block(G)
    results: list[SnpResult | None] = []
    try:
        for result in iter_block_results(G, inputs, L, Q0, args, counts):
            results.append(result)  # noqa: PERF402 (kept if a later SNP raises)
    finally:
        if results:
            out.write(variants, results, desc)


def _scan_snps(
    inputs: GwasInputs,
    L: LMM,
    args: argparse.Namespace,
    out: ScanOutput,
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
            names = Variants.from_names([snp_id for _, snp_id in block])
            _write_block(out, G, names, inputs, L, Q0, args, counts)

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


_MAP_BP_COLUMNS = ("bp", "BP", "pos", "POS", "position", "base_pair_location")
"""Header names of the position column of an ``--emma-map`` file (also ``bp_<build>``)."""

_MAP_ID_COLUMNS = ("snp", "SNP", "id", "ID", "rsid", "variant_id")
"""Header names of the SNP ID column of an ``--emma-map`` file."""


def read_emma_map_variants(fn: str, m: int) -> Variants:
    """The SNP metadata of the ``m`` rows of an EMMA genotype file, from ``--emma-map``.

    The file is read as by `read_emma_map`.  With a header line, the
    chromosome column (`_MAP_CHROM_COLUMNS`), a position column
    (`_MAP_BP_COLUMNS`, or a name starting ``bp_`` such as ``bp_b37``), an ID
    column (`_MAP_ID_COLUMNS`) and the alleles coded 0 and 1 (``allele0``,
    ``allele1``) are used when present.  Without one it is a PLINK ``.map``
    (or ``.bim``): chromosome, ID, then the position in the 4th column (3rd
    of a 3-column ``.map``); no alleles.  The SNP IDs default to EMMA's
    ``SNP_<row>``.

    Returns
    -------
    pylmm.genotypes.Variants
        ``a2`` is the allele coded 1 (the EMMA value counts it, as PLINK's
        genotype counts A2) and ``a1`` the allele coded 0.

    Raises
    ------
    pylmm.samples.InputError
        As `read_emma_map`, and for a position that is not an integer.
    """
    chroms, has_header = read_emma_map(fn, m)
    with open(fn) as f:
        lines = [line.split() for line in f if line.strip()]
    rows = lines[1:] if has_header else lines
    names = ["SNP_%d" % (k + 1) for k in range(m)]

    def column(c: int | None) -> list[str] | None:
        if c is None:
            return None
        bad = next((k for k, r in enumerate(rows) if len(r) <= c), None)
        if bad is not None:
            raise samples.InputError(
                "--emma-map %s, SNP line %d: %d fields, expected at least %d"
                % (fn, bad + 1, len(rows[bad]), c + 1)
            )
        return [r[c] for r in rows]

    id_col = bp_col = a0_col = a1_col = None
    if has_header:
        header = lines[0]
        id_col = next((header.index(c) for c in _MAP_ID_COLUMNS if c in header), None)
        bp_col = next((header.index(c) for c in _MAP_BP_COLUMNS if c in header), None)
        if bp_col is None:
            bp_col = next((k for k, c in enumerate(header) if c.startswith("bp_")), None)
        if "allele0" in header and "allele1" in header:
            a0_col, a1_col = header.index("allele0"), header.index("allele1")
    else:
        width = min(len(r) for r in rows) if rows else 0
        id_col = 1 if width >= 2 else None
        bp_col = 3 if width >= 4 else (2 if width == 3 else None)
    ids = column(id_col) or names
    bp_text = column(bp_col)
    bp = None
    if bp_text is not None:
        try:
            bp = [int(float(x)) for x in bp_text]
        except ValueError as e:
            raise samples.InputError(
                "--emma-map %s: the position column has a value that is not a number (%s)" % (fn, e)
            ) from e
    return Variants(
        snp=ids,  # type: ignore[arg-type]
        chrom=list(chroms),  # type: ignore[arg-type]
        bp=bp,  # type: ignore[arg-type]
        a1=column(a0_col),  # type: ignore[arg-type]
        a2=column(a1_col),  # type: ignore[arg-type]
    )


def loco_chromosomes(args: argparse.Namespace, reader: plink_io.PlinkReader) -> NDArray[np.object_]:
    """The chromosome of each SNP for ``--loco``, normalised and checked.

    From the ``.bim`` / ``.map`` file (`pylmm.plink_io.PlinkReader.variants`),
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


def _null_fit_record(L: LMM, args: argparse.Namespace) -> dict[str, Any] | None:
    """The run record's ``null_fit`` of ``L`` (`pylmm.cli.run_record.NullFit`).

    None with ``--refit`` and the Wald test (every SNP has its own fit).
    With ``--test lrt``, the LRT's null fit: ML on the reduced model
    (`pylmm.lmm.LMM.lrt_null_fit`), with the number of null directions of
    the kinship dropped from it.
    """
    per_snp = bool(args.remove_missing_genotypes)
    if args.test == "lrt":
        M = L._lrt_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # lrt_null_fit has warned already
            h, _, sigma2, ll = M._fit_copy(None, False)
        return dict(
            run_record.null_fit(h, sigma2, ll, False, "lrt-reduced-ml", int(M._dropped), per_snp)
        )
    if args.refit or getattr(L, "opt_h", None) is None:
        return None
    return dict(
        run_record.null_fit(
            L.opt_h,
            L.opt_sigma,
            L.opt_ll,
            bool(args.reml),
            "reml" if args.reml else "ml",
            0,
            per_snp,
        )
    )


def _loco_record(
    inputs: GwasInputs, inputs_c: GwasInputs, L: LMM, args: argparse.Namespace, chrom: str
) -> dict[str, Any]:
    """The run record's entry of chromosome ``chrom`` under ``--loco``."""
    assert inputs.loco is not None
    scale = kinship_scale(inputs_c.K)
    fit = _null_fit_record(L, args) or {}
    return {
        **fit,
        "n_snps_kinship": int(inputs.loco.snps_used(leave_out=chrom)),
        "mean_diag": scale.mean_diag,
        "centered_mean_diag": scale.centered_mean_diag,
    }


def scan_loco(inputs: GwasInputs, args: argparse.Namespace, out: ScanOutput) -> int:
    """`scan` with ``--loco``: each chromosome's SNPs tested with its own K and null fit.

    The SNPs are read once more, in blocks of ``--block-size``, and written
    in file order, with the columns of `scan`.  When a chromosome starts, its
    kinship ``K_{-c}`` (`pylmm.loco.LocoKinship.kinship`, restricted to the
    tested individuals) is eigendecomposed and the null model fitted
    (`fit_null_model`, so ``--ml``, ``--refit``, ``--test`` and the fit options apply
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
    counts = out.counts
    count = 0
    current: str | None = None
    inputs_c: GwasInputs = inputs
    L: LMM | None = None
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
                    out.loco_fits[current] = _loco_record(inputs, inputs_c, L, args, current)
                assert L is not None
                _write_block(out, G[:, a:e], variants[a:e], inputs_c, L, Q0, args, counts)
                a = e
            _log_progress(count, b)  # after the block's chromosome lines, not before
            count += b
            del G, variants
    finally:
        close = getattr(blocks, "close", None)
        if close is not None:
            close()
    return _finish(count, counts)


def _regular_output(path: str) -> bool:
    """Whether ``path`` is (or will be created as) a regular file, not a device or a pipe."""
    try:
        st = os.stat(path)
    except OSError:
        return True  # it does not exist yet: a regular file will be created
    return stat.S_ISREG(st.st_mode)


def record_path(args: argparse.Namespace) -> str | None:
    """Where the run record goes: ``--run-record``, none with ``--no-run-record``, else the default.

    The default is `pylmm.cli.run_record.sidecar_path` of ``-o`` when ``-o``
    is a regular file, and none when it is not (``/dev/stdout``, a FIFO, a
    process substitution ``>(...)``).
    """
    if getattr(args, "no_run_record", False):
        return None
    if getattr(args, "run_record", None):
        return str(args.run_record)
    if not _regular_output(args.out):
        return None
    return run_record.sidecar_path(args.out)


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
    rec_path = record_path(args)
    check_writable("--out", args.out)
    if rec_path is None:
        if not args.no_run_record:
            log.info("-o %s is not a regular file: no run record (--run-record PATH)", args.out)
    else:
        if os.path.exists(rec_path):
            log.warning("overwriting the existing run record %s", rec_path)
        check_writable("--run-record" if args.run_record else "--out (its run record)", rec_path)
    argv = getattr(args, "argv", None) or run_record.command_line()
    if not args.run_record_full_paths:
        argv = run_record.basenames(argv, run_record.path_values(args.parser, args))
    record = run_record.new_record(argv)
    collector = run_record.WarningCollector()
    log.addHandler(collector)
    timings: dict[str, float] = {}
    begin = time.time()
    try:
        _run(args, record, timings)
        record["status"] = "ok"
    except BaseException as e:
        record["error"] = "%s: %s" % (type(e).__name__, e) if str(e) else type(e).__name__
        raise
    finally:
        log.removeHandler(collector)
        timings["total"] = time.time() - begin
        record["timings_s"] = {k: round(v, 3) for k, v in timings.items()}
        record["warnings"] = collector.messages
        record["finished_at"] = run_record.utc_now()
        if rec_path is not None:
            # Never fails the run: the results are written by now.
            try:
                run_record.write_record(rec_path, record)
            except Exception as e:
                log.warning(
                    "cannot write the run record %s (%s: %s)", rec_path, type(e).__name__, e
                )
            else:
                log.info("Wrote the run record to %s", rec_path)


def _run(args: argparse.Namespace, record: dict[str, Any], timings: dict[str, float]) -> None:
    """`run`, filling the run ``record`` and ``timings`` as it goes."""
    record["inputs"] = {
        role: run_record.file_info(path, full_path=args.run_record_full_paths)
        for role, path in _input_files(args)
    }
    record["model"] = _model_record(args)
    begin = time.time()
    inputs = load_inputs(args)
    timings["read_inputs"] = time.time() - begin
    record["samples"] = inputs.sample_counts
    record["model"]["q"] = int(inputs.X0.shape[1])
    record["model"]["covariates"] = inputs.covariate_names
    record["kinship"] = _kinship_record(args, inputs)
    # With --loco the null model is fitted per chromosome (scan_loco).
    begin = time.time()
    L = fit_null_model(inputs, args) if inputs.loco is None else None
    timings["null_fit"] = time.time() - begin
    if L is not None:
        record["null_fit"] = _null_fit_record(L, args)
    try:
        out = ScanOutput(args.out, inputs, args)
    except OSError as e:
        raise UsageError("--out %s: cannot write it (%s)" % (args.out, e)) from e
    begin = time.time()
    try:
        if L is None:
            n = scan_loco(inputs, args, out)
        else:
            n = scan(inputs, L, args, out)
    finally:
        out.close()
        timings["scan"] = time.time() - begin
        out_name = args.out if args.run_record_full_paths else os.path.basename(args.out)
        record["results"] = {**out.results_info(), "path": out_name}
        if inputs.loco is not None:
            record["loco"] = out.loco_fits
    log.info("Wrote the results for %d SNPs to %s", n, args.out)


def _input_files(args: argparse.Namespace) -> list[tuple[str, str]]:
    """The input files of the run, ``(role, path)``, for the run record."""
    files: list[tuple[str, str]] = []
    if args.bfile:
        files += [("bed", args.bfile + ".bed"), ("bim", args.bfile + ".bim")]
        files += [("fam", args.bfile + ".fam")]
    elif args.tfile:
        files += [("tped", args.tfile + ".tped"), ("tfam", args.tfile + ".tfam")]
        meta = args.tfile + (".bim" if os.path.isfile(args.tfile + ".bim") else ".map")
        files.append(("map" if meta.endswith(".map") else "bim", meta))
    else:
        files.append(("emma_geno", args.emma_geno))
    if args.emma_pheno:
        files.append(("pheno", args.emma_pheno))
    else:
        base = args.bfile or args.tfile or args.emma_geno
        files.append(("pheno", args.pheno or base + ".phenos"))
    if args.covar or args.emma_covar:
        files.append(("covar", args.covar or args.emma_covar))
    if args.kinship and not args.loco:
        files.append(("kinship", args.kinship))
        fmt = kinship_io.kinship_format(args.kinship)
        if fmt == "text":
            id_file = samples.kinship_id_path(args.kinship)
            if os.path.isfile(id_file):
                files.append(("kinship_ids", id_file))
        elif fmt == "rel":
            files.append(("kinship_ids", kinship_io.rel_id_path(args.kinship)))
        else:
            files.append(("kinship_ids", kinship_io.grm_prefix(args.kinship) + ".grm.id"))
    if args.eigen:
        if os.path.isfile(args.eigen):
            files.append(("eigen", args.eigen))
        else:
            files += [("eigen_values", args.eigen + ".kva"), ("eigen_vectors", args.eigen + ".kve")]
    if args.emma_map:
        files.append(("emma_map", args.emma_map))
    return files


def _model_record(args: argparse.Namespace) -> dict[str, Any]:
    """The run record's ``model`` (`pylmm.cli.run_record.ModelInfo`)."""
    test = getattr(args, "test", "wald")
    return {
        "test": test,
        "reml": bool(args.reml) and test != "lrt",
        "refit": bool(args.refit),
        "legacy_fit": bool(args.legacy_fit),
        "batch": not args.no_batch,
        "block_size": int(args.block_size),
        "loco": bool(args.loco),
        "missing_genotypes": "remove" if args.remove_missing_genotypes else "mean-impute",
        "emma_coding": args.emma_coding if args.emma_geno else None,
        "intercept": not args.no_intercept,
        "covariates": list(args.covar_name) if args.covar_name else None,
        "q": None,
    }


def _kinship_record(args: argparse.Namespace, inputs: GwasInputs) -> dict[str, Any]:
    """The run record's ``kinship`` (`pylmm.cli.run_record.KinshipInfo`)."""
    if inputs.loco is not None:
        return {
            "source": "loco",
            "format": None,
            "has_ids": None,
            "kind": args.kinship_type or "standardized",
            "mean_diag": None,
            "centered_mean_diag": None,
        }
    fmt = kinship_io.kinship_format(args.kinship)
    if inputs.reader.type == "emma":
        has_ids = False
    elif fmt == "text":
        has_ids = os.path.isfile(samples.kinship_id_path(args.kinship))
    else:
        has_ids = True
    scale = kinship_scale(inputs.K) if inputs.K.shape[0] >= 2 else None
    return {
        "source": "file",
        "format": "grm" if args.grm else fmt,
        "has_ids": has_ids,
        "kind": None,
        "mean_diag": None if scale is None else scale.mean_diag,
        "centered_mean_diag": None if scale is None else scale.centered_mean_diag,
    }


if __name__ == "__main__":  # the removed pre-5b.2 invocation; see removed_module_main
    from pylmm.cli._common import removed_module_main

    raise SystemExit(removed_module_main("gwas"))
