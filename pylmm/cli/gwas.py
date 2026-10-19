# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""pylmmGWAS: genome-wide association with a linear mixed model.

This is the former scripts/pylmmGWAS.py, moved into the package so it can be
installed as a console script.  scripts/pylmmGWAS.py is now a thin shim that
calls main().
"""

import gzip
import os
import sys
import time
import warnings
from optparse import OptionGroup, OptionParser

import numpy as np
from scipy import linalg

from pylmm import input
from pylmm.lmm import LMM, check_covariates, snp_estimable, subset_testable

usage = """usage: %prog [options] --kfile kinshipFile --[tfile | bfile] plinkFileBase outfileBase

This program provides basic genome-wide association (GWAS) functionality.  You provide a phenotype and genotype file as well as a pre-computed (use pylmmKinship.py) kinship matrix and the program outputs a result file with information about each SNP, including the association p-value.  
The input file are all standard plink formatted with the first two columns specifiying the individual and family ID.  For the phenotype file, we accept either NA or -9 to denote missing values.  

Basic usage:

      python pylmmGWAS.py -v --bfile plinkFile --kfile preComputedKinship.kin --phenofile plinkFormattedPhenotypeFile resultFile

            """


def printOutHead(out):
    out.write("\t".join(["SNP_ID", "BETA", "BETA_SD", "F_STAT", "P_VALUE"]) + "\n")


def outputResult(out, id, beta, betaSD, ts, ps):
    out.write("\t".join([str(x) for x in [id, beta, betaSD, ts, ps]]) + "\n")


def readKinshipFile(fn, n):
    """Read a whitespace-separated n x n kinship (plain text, or gzip if fn
    ends in .gz): n lines of n numbers, no header or comment lines.

    Raises ValueError, naming the file, if a token is not a number, the file
    is not exactly n rows of n values, or K has a nan/inf.  Warns (does not
    fail) if K is noticeably asymmetric.  (Bug 16, fixed in Phase 5.1: the
    old np.fromfile + resize read silently zero-filled or truncated K.)
    """
    try:
        if fn[-3:] == ".gz":
            f = gzip.open(fn, "rt")
        else:
            f = open(fn)
    except OSError as e:
        raise ValueError("kinship file %s: cannot read (%s)" % (fn, e)) from e
    K = np.zeros((n, n))
    nrows = 0
    nvalues = 0
    lengths = set()
    try:
        with f:
            for lineno, line in enumerate(f, 1):
                v = line.split()
                if not v:
                    continue
                try:
                    row = np.array(v, dtype=np.float64)
                except ValueError:
                    bad = next((x for x in v if not _isNumber(x)), v[0])
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
        lengths = sorted(lengths)
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
                ", ".join(str(x) for x in lengths[:5]) + (", ..." if len(lengths) > 5 else ""),
            )
        )
    bad = ~np.isfinite(K)
    if bad.any():
        i, j = np.argwhere(bad)[0]
        raise ValueError(
            "kinship file %s: %d non-finite values (nan/inf), the first at row %d, column %d"
            % (fn, bad.sum(), i + 1, j + 1)
        )
    asym = np.abs(K - K.T).max() if n else 0.0
    if asym > 1e-6 * max(1.0, np.abs(K).max()):
        warnings.warn(
            "kinship file %s is not symmetric: max |K - K'| = %g" % (fn, asym), stacklevel=2
        )
    return K


def _isNumber(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def buildParser():
    parser = OptionParser(usage=usage)

    basicGroup = OptionGroup(parser, "Basic Options")
    advancedGroup = OptionGroup(parser, "Advanced Options")

    # basicGroup.add_option("--pfile", dest="pfile",
    #                  help="The base for a PLINK ped file")
    basicGroup.add_option("--tfile", dest="tfile", help="The base for a PLINK tped file")
    basicGroup.add_option("--bfile", dest="bfile", help="The base for a PLINK binary bed file")
    basicGroup.add_option(
        "--phenofile",
        dest="phenoFile",
        default=None,
        help="Without this argument the program will look for a file with .pheno that has the plinkFileBase root.  If you want to specify an alternative phenotype file, then use this argument.  This file should be in plink format. ",
    )

    # EMMA Options
    basicGroup.add_option(
        "--emmaSNP",
        dest="emmaFile",
        default=None,
        help='For backwards compatibility with emma, we allow for "EMMA" file formats.  This is just a text file with individuals on the columns and snps on the rows.',
    )
    basicGroup.add_option(
        "--emmaPHENO",
        dest="emmaPheno",
        default=None,
        help='For backwards compatibility with emma, we allow for "EMMA" file formats.  This is just a text file with each phenotype as one row.',
    )
    basicGroup.add_option(
        "--emmaCOV",
        dest="emmaCov",
        default=None,
        help='For backwards compatibility with emma, we allow for "EMMA" file formats.  This is just a text file with each covariate as one row.',
    )

    basicGroup.add_option(
        "--kfile",
        dest="kfile",
        help="The location of a kinship file.  This is an nxn plain text file and can be computed with the pylmmKinship program.",
    )
    basicGroup.add_option(
        "--covfile",
        dest="covfile",
        help="The location of a covariate file file.  This is a plink formatted covariate file.",
    )
    basicGroup.add_option(
        "-p",
        type="int",
        dest="pheno",
        help="The phenotype index to be used in association.",
        default=0,
    )

    advancedGroup.add_option(
        "--removeMissingGenotypes",
        action="store_false",
        dest="normalizeGenotype",
        default=True,
        help="By default the program replaces missing genotypes with the minor allele frequency.  This option overrides that behavior making the program remove missing individuals.  NOTE: This can increase running time due to the need to recompute the eigendecomposition for each SNP with missing values.",
    )
    advancedGroup.add_option(
        "--refit",
        action="store_true",
        dest="refit",
        default=False,
        help="Refit the variance components at each SNP (default is to lock in the variance components under the null).",
    )

    advancedGroup.add_option(
        "--REML",
        action="store_true",
        dest="REML",
        default=False,
        help="Use restricted maximum-likelihood (REML).  This is the default; the flag is accepted for backward compatibility and does nothing.",
    )
    advancedGroup.add_option(
        "--ML",
        action="store_true",
        dest="ML",
        default=False,
        help="Use maximum likelihood instead of REML, for the null fit, every per-SNP fit (--refit, --removeMissingGenotypes) and the test (sigma^2 = Q/n).  Note: with a centred kinship built from more SNPs than individuals the ML estimate of h can be biased upward.",
    )
    advancedGroup.add_option(
        "--eigen",
        dest="eigenfile",
        help="A precomputed eigendecomposition of the kinship: either a .npz file with arrays Kva and Kve (pylmmKinship --eigenNpz) or the base name B of the text files B.kva and B.kve (pylmmKinship -e, where B is the kinship output file).  Used only when no individuals are dropped.",
    )
    advancedGroup.add_option(
        "--noMean",
        dest="noMean",
        default=False,
        action="store_true",
        help="This option only applies when --cofile is used.  When covfile is provided, the program will automatically add a global mean covariate to the model unless this option is specified.",
    )

    advancedGroup.add_option(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Print extra info",
    )

    parser.add_option_group(basicGroup)
    parser.add_option_group(advancedGroup)
    return parser


def _plainWarning(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write("pylmmGWAS: warning: %s\n" % message)


def main(argv=None):
    # Warnings of the library (input, lmm) are printed as one plain line, not
    # as a UserWarning with a source line.
    with warnings.catch_warnings():
        warnings.showwarning = _plainWarning
        _main(argv)


def _warn(msg):
    sys.stderr.write("pylmmGWAS: warning: %s\n" % msg)


def _main(argv=None):
    parser = buildParser()
    (options, args) = parser.parse_args(argv)

    if len(args) != 1:
        parser.print_help()
        sys.exit()

    outFile = args[0]

    if not options.tfile and not options.bfile and not options.emmaFile:
        # if not options.pfile and not options.tfile and not options.bfile:
        parser.error(
            "You must provide at least one PLINK input file base (--tfile or --bfile) or an EMMA formatted file (--emmaSNP)."
        )
    if not options.kfile:
        parser.error("Please provide a pre-computed kinship file")

    # READING PLINK input
    if options.verbose:
        sys.stderr.write("Reading SNP input...\n")
    try:
        if options.bfile:
            IN = input.plink(
                options.bfile,
                type="b",
                phenoFile=options.phenoFile,
                normGenotype=options.normalizeGenotype,
            )
        elif options.tfile:
            IN = input.plink(
                options.tfile,
                type="t",
                phenoFile=options.phenoFile,
                normGenotype=options.normalizeGenotype,
            )
        # elif options.pfile: IN = input.plink(options.pfile,type='p', phenoFile=options.phenoFile,normGenotype=options.normalizeGenotype)
        elif options.emmaFile:
            IN = input.plink(
                options.emmaFile,
                type="emma",
                phenoFile=options.phenoFile,
                normGenotype=options.normalizeGenotype,
            )
        else:
            parser.error("You must provide at least one PLINK input file base")
    except ValueError as e:
        parser.error(str(e))

    if options.REML and options.ML:
        parser.error("--REML and --ML are mutually exclusive (REML is the default)")
    REML = not options.ML

    # No phenotype file is a usage error (bug 14, fixed in Phase 5.2: this
    # used to be os.path.isfile(None), a TypeError).
    phenoFile = options.phenoFile or IN.fbase + ".phenos"
    if options.emmaPheno:
        if not os.path.isfile(options.emmaPheno):
            parser.error("The --emmaPHENO file %s does not exist." % options.emmaPheno)
    elif not os.path.isfile(phenoFile):
        parser.error(
            "No .pheno file exist for %s.  Please provide a phenotype file using the --phenofile or --emmaPHENO argument."
            % phenoFile
        )
    if options.covfile and not os.path.isfile(options.covfile):
        parser.error("The --covfile file %s does not exist." % options.covfile)

    try:
        # Read the emma phenotype file if provided.
        # Format should be rows are phenotypes and columns are individuals.
        if options.emmaPheno:
            IN.phenos = IN.getPhenosEMMA(options.emmaPheno)

        # READING Covariate File
        P = None
        if options.covfile:
            if options.verbose:
                sys.stderr.write("Reading covariate file...\n")
            P = IN.getCovariates(options.covfile)
        elif options.emmaCov:
            if options.verbose:
                sys.stderr.write("Reading covariate file...\n")
            P = IN.getCovariatesEMMA(options.emmaCov)
    except ValueError as e:
        parser.error(str(e))

    if not 0 <= options.pheno < IN.phenos.shape[1]:
        parser.error(
            "-p %d: the phenotype file has %d phenotype column(s) (0-based index)"
            % (options.pheno, IN.phenos.shape[1])
        )

    if P is None or options.noMean:
        X0 = np.ones((IN.phenos.shape[0], 1)) if P is None else P
    else:
        X0 = np.hstack([np.ones((IN.phenos.shape[0], 1)), P])

    # READING Kinship - major bottleneck for large datasets
    if options.verbose:
        sys.stderr.write("Reading kinship...\n")
    begin = time.time()
    try:
        K = readKinshipFile(options.kfile, len(IN.indivs))
    except ValueError as e:
        parser.error(str(e))
    end = time.time()
    if options.verbose:
        sys.stderr.write(
            "Read the %d x %d kinship matrix in %0.3fs \n" % (K.shape[0], K.shape[1], end - begin)
        )

    # PROCESS the phenotype data -- Remove individuals with a missing
    # phenotype or a missing covariate (Phase 5.2: individuals with missing
    # covariates used to be a fatal error).
    # Keep will now index into the "full" data to select what we keep (either everything or a subset of non missing data
    Y = IN.phenos[:, options.pheno]
    missingCov = np.isnan(X0).any(axis=1)
    v = np.isnan(Y) | missingCov
    keep = ~v
    if np.isnan(Y).sum():
        _warn(
            "%d of %d individuals have a missing phenotype and are dropped"
            % (np.isnan(Y).sum(), len(Y))
        )
    if (missingCov & ~np.isnan(Y)).sum():
        _warn(
            "%d individuals with a phenotype have a missing covariate value and are dropped"
            % (missingCov & ~np.isnan(Y)).sum()
        )
    if v.sum():
        Y = Y[keep]
        X0 = X0[keep, :]
        K = K[keep, :][:, keep]
    if not len(Y):
        parser.error("No individual has both a phenotype and all covariates.")
    try:
        name = "the covariates" + (
            ""
            if options.noMean or (P is None)
            else " (column 0 is the intercept pylmmGWAS adds; use --noMean if your file has its own)"
        )
        check_covariates(X0, what=name)
    except ValueError as e:
        parser.error(str(e))

    # Only load the decomposition if we did not remove individuals.
    # Otherwise it would not be correct and we would have to compute it again.
    if not v.sum() and options.eigenfile:
        if options.verbose:
            sys.stderr.write("Loading pre-computed eigendecomposition...\n")
        try:
            Kva, Kve = readEigenFile(options.eigenfile, K)
        except ValueError as e:
            parser.error(str(e))
    else:
        if options.eigenfile:
            _warn(
                "--eigen is ignored: %d individuals were dropped, so the decomposition is recomputed"
                % v.sum()
            )
        Kva = []
        Kve = []

    # CREATE LMM object for association
    n = K.shape[0]
    try:
        L = LMM(Y, K, Kva, Kve, X0, verbose=options.verbose)
    except ValueError as e:
        parser.error(str(e))

    # Fit the null model -- if refit is true we will refit for each SNP, so no reason to run here
    # REML by default, ML with --ML, here and in every per-SNP fit (bug 1,
    # fixed in Phase 5.2: --refit and --removeMissingGenotypes used to fit ML).
    if not options.refit:
        if options.verbose:
            sys.stderr.write("Computing fit for null model\n")
        L.fit(REML=REML)
        if options.verbose:
            sys.stderr.write("\t heritability=%0.3f, sigma=%0.3f\n" % (L.optH, L.optSigma))
    Q0 = linalg.qr(X0, mode="economic")[0] if X0.shape[1] else np.zeros((X0.shape[0], 0))

    # Buffers for pvalues and t-stats
    PS = []
    TS = []
    count = 0
    counts = {}
    with open(outFile, "w") as out:
        printOutHead(out)

        for snp, id in IN:
            count += 1
            if options.verbose and count % 1000 == 0:
                sys.stderr.write("At SNP %d\n" % count)

            x = snp[keep].reshape((n, 1))
            v = np.isnan(x).reshape((-1,))
            # Check SNPs for missing values
            if v.sum():
                keeps = ~v
                xs = x[keeps, :]
                Ys = Y[keeps]
                X0s = X0[keeps, :]
                # One estimability check for both branches (bug 13, issue 19)
                if not subset_testable(xs, Ys, X0s, counts):
                    PS.append(np.nan)
                    TS.append(np.nan)
                    outputResult(out, id, np.nan, np.nan, np.nan, np.nan)
                    continue

                # Its ok to center the genotype -  I used options.normalizeGenotype to
                # force the removal of missing genotypes as opposed to replacing them with MAF.
                if not options.normalizeGenotype:
                    xs = (xs - xs.mean()) / np.sqrt(xs.var())
                Ks = K[keeps, :][:, keeps]
                Ls = LMM(Ys, Ks, X0=X0s, verbose=options.verbose, validate=False)
                if options.refit:
                    Ls.fit(X=xs, REML=REML)
                else:
                    Ls.fit(REML=REML)
                ts, ps, beta, betaVar = Ls.association(xs, REML=REML, returnBeta=True)
            else:
                if not snp_estimable(x, X0, Q=Q0):
                    PS.append(np.nan)
                    TS.append(np.nan)
                    outputResult(out, id, np.nan, np.nan, np.nan, np.nan)
                    continue

                if options.refit:
                    L.fit(X=x, REML=REML)
                ts, ps, beta, betaVar = L.association(x, REML=REML, returnBeta=True)

            outputResult(out, id, beta, np.sqrt(betaVar).sum(), ts, ps)
            PS.append(ps)
            TS.append(ts)

    if options.verbose and counts.get("collinear"):
        sys.stderr.write(
            "%d of %d SNPs are nan because, on the individuals with a called genotype, "
            "the covariates are no longer of full rank\n" % (counts["collinear"], count)
        )


def readEigenFile(fn, K):
    """The eigendecomposition for --eigen: a .npz with arrays Kva and Kve
    (pylmmKinship --eigenNpz), or text files fn.kva / fn.kve (pylmmKinship
    -e).  Raises ValueError if it is missing, has the wrong shape or nan/inf,
    or is not the eigendecomposition of K (checked on a random vector).
    (Bug 7, fixed in Phase 5.2: this used to np.load fn.Kva / fn.Kve, which
    pylmmKinship never wrote.)"""
    n = K.shape[0]
    try:
        if fn.endswith(".npz"):
            with np.load(fn) as z:
                if "Kva" not in z or "Kve" not in z:
                    raise ValueError(
                        "eigen file %s must hold arrays Kva and Kve, has %s"
                        % (fn, ", ".join(z.files))
                    )
                Kva, Kve = z["Kva"], z["Kve"]
        else:
            Kva, Kve = np.loadtxt(fn + ".kva", ndmin=1), np.loadtxt(fn + ".kve", ndmin=2)
    except OSError as e:
        raise ValueError(
            "cannot read the --eigen decomposition %s: %s (expected a .npz file or %s.kva and %s.kve)"
            % (fn, e, fn, fn)
        ) from e
    Kva, Kve = np.asarray(Kva, dtype=np.float64), np.asarray(Kve, dtype=np.float64)
    if Kva.shape != (n,) or Kve.shape != (n, n):
        raise ValueError(
            "--eigen %s: eigenvalues of shape %s and eigenvectors of shape %s, expected (%d,) and (%d, %d)"
            % (fn, Kva.shape, Kve.shape, n, n, n)
        )
    if not (np.isfinite(Kva).all() and np.isfinite(Kve).all()):
        raise ValueError("--eigen %s has nan/inf values" % fn)
    u = np.random.default_rng(0).standard_normal(n)
    err = np.abs(Kve.dot(Kva * Kve.T.dot(u)) - K.dot(u)).max()
    if err > 1e-6 * max(1.0, np.abs(K.dot(u)).max()):
        raise ValueError(
            "--eigen %s is not the eigendecomposition of the --kfile kinship (max error %g)"
            % (fn, err)
        )
    return Kva, Kve


if __name__ == "__main__":
    main()
