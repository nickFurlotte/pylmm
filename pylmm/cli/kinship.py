# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""pylmmKinship: compute a realized-relationship kinship matrix.

This is the former scripts/pylmmKinship.py, moved into the package so it can
be installed as a console script.  scripts/pylmmKinship.py is now a thin shim
that calls main().
"""

import sys
import warnings
from optparse import OptionGroup, OptionParser

import numpy as np
from scipy import linalg

from pylmm import input

usage = """usage: %prog [options] --[tfile | bfile] plinkFileBase outfile

"""


def buildParser():
    parser = OptionParser(usage=usage)

    basicGroup = OptionGroup(parser, "Basic Options")
    # advancedGroup = OptionGroup(parser, "Advanced Options")

    # basicGroup.add_option("--pfile", dest="pfile",
    #                  help="The base for a PLINK ped file")
    basicGroup.add_option("--tfile", dest="tfile", help="The base for a PLINK tped file")
    basicGroup.add_option("--bfile", dest="bfile", help="The base for a PLINK binary ped file")
    basicGroup.add_option(
        "--emmaSNP",
        dest="emmaFile",
        default=None,
        help='For backwards compatibility with emma, we allow for "EMMA" file formats.  This is just a text file with individuals on the rows and snps on the columns.',
    )
    basicGroup.add_option(
        "--emmaNumSNPs",
        dest="numSNPs",
        type="int",
        default=0,
        help="When providing the emmaSNP file you need to specify how many snps are in the file",
    )

    # NOTE (kept): the value given to -e is ignored; output always goes to
    # <outfile>.kva / <outfile>.kve.
    basicGroup.add_option(
        "-e",
        "--efile",
        dest="saveEig",
        help="Save the eigendecomposition as text files <outfile>.kva and <outfile>.kve (the value of this option is ignored).  pylmmGWAS --eigen <outfile> reads them.",
    )
    basicGroup.add_option(
        "--eigenNpz",
        dest="eigenNpz",
        default=None,
        help="Save the eigendecomposition to this .npz file (arrays Kva and Kve), which pylmmGWAS --eigen reads.  Binary and exact, and faster to read than -e.",
    )
    basicGroup.add_option(
        "-n",
        default=1000,
        dest="computeSize",
        type="int",
        help="The maximum number of SNPs to read into memory at once (default 1000).  This is important when there is a large number of SNPs, because memory could be an issue.",
    )

    basicGroup.add_option(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Print extra info",
    )

    parser.add_option_group(basicGroup)
    # parser.add_option_group(advancedGroup)
    return parser


def main(argv=None):
    parser = buildParser()
    (options, args) = parser.parse_args(argv)
    if len(args) != 1:
        parser.print_help()
        sys.exit()

    outFile = args[0]

    if not options.tfile and not options.bfile and not options.emmaFile:
        parser.error(
            "You must provide at least one PLINK input file base (--tfile or --bfile) or an emma formatted file (--emmaSNP)."
        )

    if options.verbose:
        sys.stderr.write("Reading PLINK input...\n")
    if options.bfile:
        IN = input.plink(options.bfile, type="b")
    elif options.tfile:
        IN = input.plink(options.tfile, type="t")
    # elif options.pfile: IN = input.plink(options.pfile,type='p')
    elif options.emmaFile:
        if not options.numSNPs:
            parser.error(
                "You must provide the number of SNPs when specifying an emma formatted file."
            )
        IN = input.plink(options.emmaFile, type="emma")
    else:
        parser.error(
            "You must provide at least one PLINK input file base (--tfile or --bfile) or an emma formatted file (--emmaSNP)."
        )

    n = len(IN.indivs)
    m = options.computeSize
    W = np.ones((n, m)) * np.nan

    IN.getSNPIterator()
    # Annoying hack to get around the fact that it is expensive to determine the number of SNPs in an emma file
    if options.emmaFile:
        IN.numSNPs = options.numSNPs
    i = 0
    K = None
    nAllMissing = 0
    while i < IN.numSNPs:
        j = 0
        while j < options.computeSize and i < IN.numSNPs:
            snp, id = next(IN)
            # Skip monomorphic SNPs and SNPs with no called genotype (bug 11:
            # an all-missing SNP used to crash, or to make K all nan).
            if not np.isfinite(snp).all() or snp.var() == 0:
                if np.isnan(snp).all():
                    nAllMissing += 1
                i += 1
                continue
            W[:, j] = snp

            i += 1
            j += 1
        if j < options.computeSize:
            W = W[:, list(range(0, j))]

        if options.verbose:
            sys.stderr.write("Processing first %d SNPs\n" % i)
        # The original tried scipy.linalg.fblas.dgemm first; scipy has had no
        # linalg.fblas since 0.12, so it always fell back to np.dot.
        if K is None:
            K = np.dot(W, W.T)
        else:
            K_j = np.dot(W, W.T)
            K = K + K_j

    if nAllMissing:
        warnings.warn("skipped %d SNPs with no called genotype" % nAllMissing)
    if K is None:
        parser.error("No SNP with a non-constant genotype was read; cannot compute a kinship.")
    K = K / float(IN.numSNPs)
    if options.verbose:
        sys.stderr.write("Saving Kinship file to %s\n" % outFile)
    np.savetxt(outFile, K)

    if options.saveEig or options.eigenNpz:
        if options.verbose:
            sys.stderr.write("Obtaining Eigendecomposition\n")
        Kva, Kve = linalg.eigh(K)
    if options.saveEig:
        if options.verbose:
            sys.stderr.write("Saving eigendecomposition to %s.[kva | kve]\n" % outFile)
        np.savetxt(outFile + ".kva", Kva)
        np.savetxt(outFile + ".kve", Kve)
    if options.eigenNpz:
        if options.verbose:
            sys.stderr.write("Saving eigendecomposition to %s\n" % options.eigenNpz)
        np.savez(options.eigenNpz, Kva=Kva, Kve=Kve)


if __name__ == "__main__":
    main()
