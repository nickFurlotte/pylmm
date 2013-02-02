#!/usr/bin/python

# pyLMM is a python-based linear mixed-model solver with applications to GWAS

# Copyright (C) 2013  Nicholas A. Furlotte (nick.furlotte@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from optparse import OptionParser,OptionGroup
usage = """usage: %prog [options] --[t | b | p]file plinkFileBase outfile

NOTE: The current running version only supports binary PED files (PLINK).  It is simple to convert between ped or tped and bed using PLINK.  Sorry for the inconvinience.
"""

parser = OptionParser(usage=usage)

basicGroup = OptionGroup(parser, "Basic Options")
#advancedGroup = OptionGroup(parser, "Advanced Options")

basicGroup.add_option("--pfile", dest="pfile",
                  help="The base for a PLINK ped file")
basicGroup.add_option("--tfile", dest="tfile",
                  help="The base for a PLINK tped file")
basicGroup.add_option("--bfile", dest="bfile",
                  help="The base for a PLINK binary ped file")

basicGroup.add_option("-e", "--efile", dest="saveEig", help="Save eigendecomposition to this file.")
basicGroup.add_option("-n", default=1000,dest="computeSize", type="int", help="The maximum number of SNPs to read into memory at once (default 1000).  This is important when there is a large number of SNPs, because memory could be an issue.")

basicGroup.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print extra info")

parser.add_option_group(basicGroup)
#parser.add_option_group(advancedGroup)

(options, args) = parser.parse_args()
if len(args) != 1: parser.error("Incorrect number of arguments")
outFile = args[0]

import sys
import os
import numpy as np
from scipy import linalg
from pylmm.lmm import calculateKinship
from pylmm import input

if not options.pfile and not options.tfile and not options.bfile: 
   parser.error("You must provide at least one PLINK input file base")

if options.verbose: sys.stderr.write("Reading PLINK input...\n")
if options.bfile: IN = input.plink(options.bfile,type='b')
elif options.tfile: IN = input.plink(options.tfile,type='t')
elif options.pfile: IN = input.plink(options.pfile,type='p')
else: parser.error("You must provide at least one PLINK input file base")

n = len(IN.indivs)
m = options.computeSize
W = np.ones((n,m)) * np.nan

IN.getSNPIterator()
i = 0
K = None
while i < IN.numSNPs:
   j = 0
   while j < options.computeSize and i < IN.numSNPs:
      snp,id = IN.next()
      if snp.var() == 0:
	 i += 1
	 continue
      W[:,j] = snp

      i += 1
      j += 1
   if j < options.computeSize: W = W[:,range(0,j)] 

   if options.verbose: sys.stderr.write("Processing first %d SNPs\n" % i)
   if K == None: K = linalg.fblas.dgemm(alpha=1.,a=W.T,b=W.T,trans_a=True,trans_b=False) # calculateKinship(W) * j
   #if K == None: K = np.dot(W,W.T) # calculateKinship(W) * j
   else:
      K_j = linalg.fblas.dgemm(alpha=1.,a=W.T,b=W.T,trans_a=True,trans_b=False) # calculateKinship(W) * j
      K = K + K_j

K = K / float(IN.numSNPs)
if options.verbose: sys.stderr.write("Saving Kinship file to %s\n" % outFile)
np.savetxt(outFile,K)

if options.saveEig:
   if options.verbose: sys.stderr.write("Obtaining Eigendecomposition\n")
   Kva,Kve = linalg.eigh(K)
   if options.verbose: sys.stderr.write("Saving eigendecomposition to %s.[kva | kve]\n" % outFile)
   np.savetxt(outFile+".kva",Kva)
   np.savetxt(outFile+".kve",Kve)
      


