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

import pdb
import time

def printOutHead(): out.write("\t".join(["SNP_ID","BETA","BETA_SD","F_STAT","P_VALUE"]) + "\n")
def outputResult(id,beta,betaSD,ts,ps):
   out.write("\t".join([str(x) for x in [id,beta,betaSD,ts,ps]]) + "\n")

from optparse import OptionParser,OptionGroup
usage = """usage: %prog [options] --kfile kinshipFile --[t | b | p]file plinkFileBase outfileBase

This program provides basic genome-wide association (GWAS) functionality.  You provide a phenotype and genotype file as well as a pre-computed (use pylmmKinship.py) kinship matrix and the program outputs a result file with information about each SNP, including the association p-value.  
The input file are all standard plink formatted with the first two columns specifiying the individual and family ID.  For the phenotype file, we accept either NA or -9 to denote missing values.  

Basic usage:

      python pylmmGWAS.py -v --bfile plinkFile --kfile preComputedKinship.kin --phenofile plinkFormattedPhenotypeFile resultFile

NOTE: The current running version only supports binary PED files (PLINK).  It is simple to convert between ped or tped and bed using PLINK.  Sorry for the inconvinience.

	    """
parser = OptionParser(usage=usage)

basicGroup = OptionGroup(parser, "Basic Options")
advancedGroup = OptionGroup(parser, "Advanced Options")

basicGroup.add_option("--pfile", dest="pfile",
                  help="The base for a PLINK ped file")
basicGroup.add_option("--tfile", dest="tfile",
                  help="The base for a PLINK tped file")
basicGroup.add_option("--bfile", dest="bfile",
                  help="The base for a PLINK binary bed file")
basicGroup.add_option("--phenofile", dest="phenoFile", default=None,
                  help="Without this argument the program will look for a file with .pheno that has the plinkFileBase root.  If you want to specify an alternative phenotype file, then use this argument.  The order does not matter for this file. ")
basicGroup.add_option("--kfile", dest="kfile",
                  help="The location of a kinship file.  This is an nxn plain text file and can be computed with the pylmmKinship program.")
basicGroup.add_option("--covfile", dest="covfile",
                  help="The location of a covariate file file.  This is a plink formatted covariate file.")
basicGroup.add_option("-p", type="int", dest="pheno", help="The phenotype index to be used in association.", default=0)

advancedGroup.add_option("--removeMissingGenotypes",
                  action="store_false", dest="normalizeGenotype", default=True,
                  help="By default the program replaces missing genotypes with the minor allele frequency.  This option overrides that behavior making the program remove missing individuals.  NOTE: This can increase running time due to the need to recompute the eigendecomposition for each SNP with missing values.")
advancedGroup.add_option("--refit",
                  action="store_true", dest="refit", default=False,
                  help="Refit the variance components at each SNP (default is to lock in the variance components under the null).")

advancedGroup.add_option("--REML",
                  action="store_true", dest="REML", default=False,
                  help="Use restricted maximum-likelihood (REML) (default is maximum-likelihood).")
#advancedGroup.add_option("-e", "--efile", dest="saveEig", help="Save eigendecomposition to this file.")
advancedGroup.add_option("--eigen", dest="eigenfile",
                  help="The location of the precomputed eigendecomposition for the kinship file")
advancedGroup.add_option("--noMean", dest="noMean", default=False,action="store_true",
                  help="This option only applies when --cofile is used.  When covfile is provided, the program will automatically add a global mean covariate to the model unless this option is specified.")

advancedGroup.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print extra info")

parser.add_option_group(basicGroup)
parser.add_option_group(advancedGroup)

(options, args) = parser.parse_args()

import sys
import os
import numpy as np
from scipy import linalg
from pylmm.lmm import LMM
from pylmm import input

if len(args) != 1: parser.error("Incorrect number of arguments")
outFile = args[0]

if not options.pfile and not options.tfile and not options.bfile: 
   parser.error("You must provide at least one PLINK input file base")
if not options.kfile:
   parser.error("Please provide a pre-computed kinship file")

# READING PLINK input
if options.verbose: sys.stderr.write("Reading PLINK input...\n")
if options.bfile: IN = input.plink(options.bfile,type='b', phenoFile=options.phenoFile,normGenotype=options.normalizeGenotype)
elif options.tfile: IN = input.plink(options.tfile,type='t', phenoFile=options.phenoFile,normGenotype=options.normalizeGenotype)
elif options.pfile: IN = input.plink(options.pfile,type='p', phenoFile=options.phenoFile,normGenotype=options.normalizeGenotype)
else: parser.error("You must provide at least one PLINK input file base")

if not os.path.isfile(options.phenoFile or IN.fbase + '.phenos'):
   parser.error("No .pheno file exist for %s" % (options.phenoFile or IN.fbase + '.phenos'))

# READING Covariate File
if options.covfile: 
   if options.verbose: sys.stderr.write("Reading covariate file...\n")
   # Read the covariate file -- write this into input.plink
   P = IN.getCovariates(options.covfile) 

   if options.noMean: X0 = P
   else: X0 = np.hstack([np.ones((IN.phenos.shape[0],1)),P])

   if np.isnan(X0).sum(): 
      parser.error("The covariate file %s contains missing values. At this time we are not dealing with this case.  Either remove those individuals with missing values or replace them in some way.")
else: X0 = np.ones((IN.phenos.shape[0],1))

# READING Kinship - major bottleneck for large datasets
if options.verbose: sys.stderr.write("Reading kinship...\n")
begin = time.time()
# This method seems to be the fastest and works if you already know the size of the matrix
if options.kfile[-3:] == '.gz':
   import gzip
   f = gzip.open(options.kfile,'r')
   F = f.read() # might exhaust mem if the file is huge
   K = np.fromstring(F,sep=' ') # Assume that space separated
   f.close()
else: K = np.fromfile(open(options.kfile,'r'),sep=" ")
K.resize((len(IN.indivs),len(IN.indivs)))
end = time.time()
# Other slower ways
#K = np.loadtxt(options.kfile)
#K = np.genfromtxt(options.kfile)
if options.verbose: sys.stderr.write("Read the %d x %d kinship matrix in %0.3fs \n" % (K.shape[0],K.shape[1],end-begin))


# PROCESS the phenotype data -- Remove missing phenotype values
# Keep will now index into the "full" data to select what we keep (either everything or a subset of non missing data
Y = IN.phenos[:,options.pheno]
v = np.isnan(Y)
keep = True - v
if v.sum():
   if options.verbose: sys.stderr.write("Cleaning the phenotype vector by removing %d individuals...\n" % (v.sum()))
   Y = Y[keep]
   X0 = X0[keep,:]
   K = K[keep,:][:,keep]
   Kva = []
   Kve = []

# Only load the decomposition if we did not remove individuals.
# Otherwise it would not be correct and we would have to compute it again.
if not v.sum() and options.eigenfile:
   if options.verbose: sys.stderr.write("Loading pre-computed eigendecomposition...\n")
   Kva = np.load(options.eigenfile + ".Kva")
   Kve = np.load(options.eigenfile + ".Kve")
else: 
   Kva = []
   Kve = []

# CREATE LMM object for association
n = K.shape[0]
L = LMM(Y,K,Kva,Kve,X0)
# Fit the null model -- if refit is true we will refit for each SNP, so no reason to run here
if not options.refit: 
   if options.verbose: sys.stderr.write("Computing fit for null model\n")
   L.fit()
   if options.verbose: sys.stderr.write("\t heritability=%0.3f, sigma=%0.3f\n" % (L.optH,L.optSigma))

# Buffers for pvalues and t-stats
PS = []
TS = []
count = 0
out = open(outFile,'w')
printOutHead()

for snp,id in IN:
   count += 1
   if options.verbose and count % 1000 == 0: 
      sys.stderr.write("At SNP %d\n" % count)
      
   x = snp[keep].reshape((n,1))
   #x[[1,50,100,200,3000],:] = np.nan
   v = np.isnan(x).reshape((-1,))
   # Check SNPs for missing values
   if v.sum():
      keeps = True - v
      xs = x[keeps,:]
      if keeps.sum() <= 1 or xs.var() == 0: 
	 PS.append(np.nan)
	 TS.append(np.nan)
	 outputResult(id,np.nan,np.nan,np.nan,np.nan)
	 continue

      # Its ok to center the genotype -  I used options.normalizeGenotype to 
      # force the removal of missing genotypes as opposed to replacing them with MAF.
      if not options.normalizeGenotype: xs = (xs - xs.mean()) / np.sqrt(xs.var())
      Ys = Y[keeps]
      X0s = X0[keeps,:]
      Ks = K[keeps,:][:,keeps]
      Ls = LMM(Ys,Ks,X0=X0s)
      if options.refit: Ls.fit(X=xs,REML=options.REML)
      else: Ls.fit(REML=options.REML)
      ts,ps,beta,betaVar = Ls.association(xs,REML=options.REML,returnBeta=True)
   else: 
      if x.var() == 0: 
	 PS.append(np.nan)
	 TS.append(np.nan)
	 outputResult(id,np.nan,np.nan,np.nan,np.nan)
	 continue

      if options.refit: L.fit(X=x,REML=options.REML)
      ts,ps,beta,betaVar = L.association(x,REML=options.REML,returnBeta=True)
	    
   outputResult(id,beta,np.sqrt(betaVar).sum(),ts,ps)
   PS.append(ps)
   TS.append(ts)

