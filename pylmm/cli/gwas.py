# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""``pylmm gwas``: genome-wide association with a linear mixed model.

Reads genotypes (PLINK BED or TPED, or an EMMA matrix), a phenotype, optional
covariates and a precomputed kinship (``pylmm kinship``), fits the null model
once (or per SNP with ``--refit``) and writes one tab-separated line per SNP:
``SNP_ID BETA BETA_SD F_STAT P_VALUE``.

The steps are separate functions: `load_inputs` (genotypes, phenotype,
covariates, kinship and its eigendecomposition, aligned by sample ID),
`fit_null_model` and `scan` (the per-SNP tests, on blocks of ``--block-size``
SNPs read with `pylmm.plink_io.PlinkReader.iter_blocks`, written in file order).
"""

from __future__ import annotations

import argparse
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
from pylmm.genotypes import DEFAULT_BLOCK_SIZE, Variants, check_block_size
from pylmm.lmm import (
    LMM,
    AssociationResult,
    check_covariates,
    snp_estimable,
    snp_estimable_columns,
    subset_testable,
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

The kinship (--kinship, computed with `pylmm kinship`) accounts for
relatedness.  Phenotype, covariate and kinship rows are matched to the
genotype file by sample ID (FID and IID, or IID alone with --id-match iid),
never by position; the kinship IDs come from its `<kinship>.id` file.
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
      --no-batch -o results.tsv"""


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

    kin = p.add_argument_group("kinship")
    kin.add_argument(
        "--kinship",
        metavar="FILE",
        required=True,
        help="(required) n x n kinship matrix, whitespace-separated text (or .gz), from "
        "`pylmm kinship`.  "
        "With --bfile/--tfile its rows are matched to the genotypes by the IDs in FILE.id if "
        "that exists; otherwise they are assumed to be in genotype order (with a warning).  "
        "With --emma-geno, which has no IDs, the rows are always in genotype order: no "
        "warning without FILE.id, and a FILE.id is ignored (with a warning)",
    )
    kin.add_argument(
        "--eigen",
        metavar="PATH",
        help="precomputed eigendecomposition of the kinship: a .npz file (`pylmm kinship "
        "--eigen-npz`) or the PATH.kva/PATH.kve text files (`pylmm kinship --eigen-text`).  "
        "Used only when no individual is dropped or reordered",
    )
    kin.add_argument(
        "--id-match",
        choices=sorted(ID_MATCH_CHOICES),
        default="fid-iid",
        help="match sample IDs on FID and IID (default) or on the IID alone (PLINK input only)",
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
        "With well-conditioned covariates the numbers are the same up to rounding (within "
        "about 1e-9 relative); with badly scaled ones (e.g. a raw date or year column) they "
        "can differ by up to a few 1e-6 relative, the batched default being the more accurate: "
        "centre or scale such covariates.  --refit and SNPs with missing calls always take "
        "this path",
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
    n = K.shape[0]
    try:
        if fn.endswith(".npz"):
            with np.load(fn) as z:
                if "Kva" not in z or "Kve" not in z:
                    raise ValueError(
                        "eigen file %s must hold arrays Kva and Kve, has %s"
                        % (fn, ", ".join(z.files))
                    )
                kva, kve = z["Kva"], z["Kve"]
        else:
            kva, kve = np.loadtxt(fn + ".kva", ndmin=1), np.loadtxt(fn + ".kve", ndmin=2)
    except OSError as e:
        raise ValueError(
            "cannot read the --eigen decomposition %s: %s (expected a .npz file or %s.kva and %s.kve)"
            % (fn, e, fn, fn)
        ) from e
    kva, kve = np.asarray(kva, dtype=np.float64), np.asarray(kve, dtype=np.float64)
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
    input, which has no IDs).

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
    al = samples.align_samples(
        {"genotype file": reader.indivs, "kinship": kin_ids}, match=reader.id_match
    )
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
        The kinship of the tested individuals, ``(n, n)``.
    keep : numpy.ndarray
        Boolean mask over ``reader.indivs``: individuals with a phenotype and
        all covariates.
    kva, kve : numpy.ndarray or None
        The ``--eigen`` decomposition of K, or None to compute it.
    """

    reader: plink_io.PlinkReader
    Y: FloatArray
    X0: FloatArray
    K: FloatArray
    keep: NDArray[np.bool_]
    kva: FloatArray | None
    kve: FloatArray | None


def _usage(fn: Any, *a: Any, **kw: Any) -> Any:
    """Call ``fn``, turning a ValueError (invalid input) into a `UsageError`."""
    try:
        return fn(*a, **kw)
    except ValueError as e:
        raise UsageError(str(e)) from e


def open_genotypes(args: argparse.Namespace) -> plink_io.PlinkReader:
    """Open the genotype input (``--bfile``, ``--tfile`` or ``--emma-geno``).

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
        )
    except OSError as e:  # e.g. a missing .fam: name the option
        raise UsageError("%s %s: %s" % (flag, base, e)) from e
    except ValueError as e:  # the message names the file (genotype or phenotype)
        raise UsageError(str(e)) from e


def _check_input_files(args: argparse.Namespace, reader: plink_io.PlinkReader) -> None:
    """Fail early, before the kinship is read, if it or a phenotype or covariate file is missing."""
    if not os.path.isfile(args.kinship):
        raise UsageError("--kinship %s: no such file" % args.kinship)
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
        Restricted to the kept individuals.
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

    kva, kve = _eigen(args, K, int((~keep).sum()), kinship_aligned)
    return GwasInputs(reader=reader, Y=Y, X0=X0, K=K, keep=keep, kva=kva, kve=kve)


def _eigen(
    args: argparse.Namespace, K: FloatArray, n_dropped: int, kinship_aligned: bool
) -> tuple[FloatArray | None, FloatArray | None]:
    """The ``--eigen`` decomposition, if given and still valid for K; else (None, None).

    It is valid only if no individual was dropped or reordered; otherwise it
    is ignored, with a warning, and recomputed.
    """
    if not args.eigen:
        return None, None
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
    log.info("Loading pre-computed eigendecomposition...")
    kva, kve = _usage(read_eigen_file, args.eigen, K)
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
    L: LMM = _usage(LMM, inputs.Y, inputs.K, inputs.kva, inputs.kve, inputs.X0, verbose=verbose)
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
        ls = LMM(ys, ks, X0=x0s, verbose=L.verbose, validate=False)
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

    A block is read whole, so a reader error inside a block (a truncated BED
    file, a bad TPED or EMMA line, or SNP metadata `iter_blocks` cannot
    parse) loses the block's earlier SNPs.  The scan then reads the rest of
    the file one SNP at a time (`_scan_snps`, buffered into blocks of
    ``--block-size`` as before), from the first SNP of that block: the SNPs
    before the error are tested and written and the error is raised, as
    above (an error that does not recur there, e.g. a transient one, is
    logged as a warning and the scan continues).

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
    except Exception as e:  # e.g. a .bim the variant metadata cannot parse
        error = e
    try:
        while blocks is not None:
            # Only the reader is guarded: an error in a test or a write is raised as is.
            try:
                G, variants = next(blocks)
            except StopIteration:
                break
            except Exception as e:
                error = e
                break
            _log_progress(count, G.shape[1])
            count += G.shape[1]
            _write_block(out, G, variants.snp, inputs, L, Q0, args, counts)
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
    for _ in range(start):  # already written from blocks
        if next(snps, None) is None:
            return start
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
    L = fit_null_model(inputs, args)
    try:
        out = open(args.out, "w")
    except OSError as e:
        raise UsageError("--out %s: cannot write it (%s)" % (args.out, e)) from e
    with out:
        n = scan(inputs, L, args, out)
    log.info("Wrote the results for %d SNPs to %s", n, args.out)


if __name__ == "__main__":  # the removed pre-5b.2 invocation; see removed_module_main
    from pylmm.cli._common import removed_module_main

    raise SystemExit(removed_module_main("gwas"))
