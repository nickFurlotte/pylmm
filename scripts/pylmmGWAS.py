#!/usr/bin/python

# pylmm is a python-based linear mixed-model solver with applications to GWAS

# Copyright (C) 2013  Nicholas A. Furlotte (nick.furlotte@gmail.com)

#The program is free for academic use. Please contact Nick Furlotte
#<nick.furlotte@gmail.com> if you are interested in using the software for
#commercial purposes.

#The software must not be modified and distributed without prior
#permission of the author.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pdb
import time
import sys

def printOutHead(): out.write("\t".join(["SNP_ID","BETA","BETA_SD","F_STAT","P_VALUE"]) + "\n")
def outputResult(id,beta,betaSD,ts,ps):
   out.write("\t".join([str(x) for x in [id,beta,betaSD,ts,ps]]) + "\n")


from optparse import OptionParser,OptionGroup
usage = """usage: %prog [options] --kfile kinshipFile --[tfile | bfile] plinkFileBase outfileBase

This program provides basic genome-wide association (GWAS) functionality.  You provide a phenotype and genotype file as well as a pre-computed (use pylmmKinship.py) kinship matrix and the program outputs a result file with information about each SNP, including the association p-value.  
The input file are all standard plink formatted with the first two columns specifiying the individual and family ID.  For the phenotype file, we accept either NA or -9 to denote missing values.  

Basic usage:

      python pylmmGWAS.py -v --bfile plinkFile --kfile preComputedKinship.kin --phenofile plinkFormattedPhenotypeFile resultFile

	    """
parser = OptionParser(usage=usage)

basicGroup = OptionGroup(parser, "Basic Options")
advancedGroup = OptionGroup(parser, "Advanced Options")
experimentalGroup = OptionGroup(parser, "Experimental Options")

#basicGroup.add_option("--pfile", dest="pfile",
#                  help="The base for a PLINK ped file")
basicGroup.add_option("--tfile", dest="tfile",
                  help="The base for a PLINK tped file")
basicGroup.add_option("--bfile", dest="bfile",
                  help="The base for a PLINK binary bed file")
basicGroup.add_option("--phenofile", dest="phenoFile", default=None,
                  help="Without this argument the program will look for a file with .pheno that has the plinkFileBase root.  If you want to specify an alternative phenotype file, then use this argument.  This file should be in plink format. ")

# EMMA Options
basicGroup.add_option("--emmaSNP", dest="emmaFile", default=None,
                  help="For backwards compatibility with emma, we allow for \"EMMA\" file formats.  This is just a text file with individuals on the columns and snps on the rows.")
basicGroup.add_option("--emmaPHENO", dest="emmaPheno", default=None,
                  help="For backwards compatibility with emma, we allow for \"EMMA\" file formats.  This is just a text file with each phenotype as one row.")
basicGroup.add_option("--emmaCOV", dest="emmaCov", default=None,
                  help="For backwards compatibility with emma, we allow for \"EMMA\" file formats.  This is just a text file with each covariate as one row.")

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
                  help="The location of the precomputed eigendecomposition for the kinship file.  These can be computed with pylmmKinship.py.")
advancedGroup.add_option("--noMean", dest="noMean", default=False,action="store_true",
                  help="This option only applies when --cofile is used.  When covfile is provided, the program will automatically add a global mean covariate to the model unless this option is specified.")

advancedGroup.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print extra info")

# Experimental Group Options
experimentalGroup.add_option("--kfile2", dest="kfile2", 
                  help="The location of a second kinship file.  This file has the same format as the first kinship.  This might be used if you want to correct for another form of confounding.")

parser.add_option_group(basicGroup)
parser.add_option_group(advancedGroup)
parser.add_option_group(experimentalGroup)

(options, args) = parser.parse_args()

import sys
import os
import numpy as np
from scipy import linalg
from pylmm.lmm import LMM
from pylmm import input

if len(args) != 1:  
   parser.print_help()
   sys.exit()

outFile = args[0]

if not options.tfile and not options.bfile and not options.emmaFile: 
#if not options.pfile and not options.tfile and not options.bfile: 
   parser.error("You must provide at least one PLINK input file base (--tfile or --bfile) or an EMMA formatted file (--emmaSNP).")
if not options.kfile:
   parser.error("Please provide a pre-computed kinship file")

# READING PLINK input
if options.verbose: sys.stderr.write("Reading SNP input...\n")
if options.bfile: IN = input.plink(options.bfile,type='b', phenoFile=options.phenoFile,normGenotype=options.normalizeGenotype)
elif options.tfile: IN = input.plink(options.tfile,type='t', phenoFile=options.phenoFile,normGenotype=options.normalizeGenotype)
#elif options.pfile: IN = input.plink(options.pfile,type='p', phenoFile=options.phenoFile,normGenotype=options.normalizeGenotype)
elif options.emmaFile: IN = input.plink(options.emmaFile,type='emma', phenoFile=options.phenoFile,normGenotype=options.normalizeGenotype)
else: parser.error("You must provide at least one PLINK input file base")

if not os.path.isfile(options.phenoFile or IN.fbase + '.phenos') and not os.path.isfile(options.emmaPheno):
   parser.error("No .pheno file exist for %s.  Please provide a phenotype file using the --phenofile or --emmaPHENO argument." % (options.phenoFile or IN.fbase + '.phenos'))

# Read the emma phenotype file if provided.
# Format should be rows are phenotypes and columns are individuals.
if options.emmaPheno:
   f = open(options.emmaPheno,'r')
   P = []
   for line in f:
      v = line.strip().split()
      p = []
      for x in v:
	 try:
	    p.append(float(x))
	 except: p.append(np.nan)
      P.append(p)
   f.close()
   IN.phenos = np.array(P).T

# READING Covariate File
if options.covfile: 
   if options.verbose: sys.stderr.write("Reading covariate file...\n")
   # Read the covariate file -- write this into input.plink
   P = IN.getCovariates(options.covfile) 

   if options.noMean: 
      X0 = P
   else: 
      X0 = np.hstack([np.ones((IN.phenos.shape[0],1)),P])
elif options.emmaCov:
   if options.verbose: 
      sys.stderr.write("Reading covariate file...\n")
   P = IN.getCovariatesEMMA(options.emmaCov) 
   if options.noMean: 
      X0 = P
   else: 
      X0 = np.hstack([np.ones((IN.phenos.shape[0],1)),P])

   # TODO: Replace this with a mechanism ot remove the individuals with missing covariates and realign all other parameters.
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
else: 
   K = np.fromfile(open(options.kfile,'r'),sep=" ")
K.resize((len(IN.indivs),len(IN.indivs)))
end = time.time()
# Other slower ways
#K = np.loadtxt(options.kfile)
#K = np.genfromtxt(options.kfile)
if options.verbose: sys.stderr.write("Read the %d x %d kinship matrix in %0.3fs \n" % (K.shape[0],K.shape[1],end-begin))

if options.kfile2:
   if options.verbose: sys.stderr.write("Reading second kinship...\n")
   begin = time.time()
   # This method seems to be the fastest and works if you already know the size of the matrix
   if options.kfile2[-3:] == '.gz':
      import gzip
      f = gzip.open(options.kfile2,'r')
      F = f.read() # might exhaust mem if the file is huge
      K2 = np.fromstring(F,sep=' ') # Assume that space separated
      f.close()
   else: K2 = np.fromfile(open(options.kfile2,'r'),sep=" ")
   K2.resize((len(IN.indivs),len(IN.indivs)))
   end = time.time()
   if options.verbose: sys.stderr.write("Read the %d x %d second kinship matrix in %0.3fs \n" % (K2.shape[0],K2.shape[1],end-begin))

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
   if options.kfile2: K2 = K2[keep,:][:,keep]
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
if not options.kfile2:  L = LMM(Y,K,Kva,Kve,X0,verbose=options.verbose)
else:  L = LMM_withK2(Y,K,Kva,Kve,X0,verbose=options.verbose,K2=K2)

# Fit the null model -- if refit is true we will refit for each SNP, so no reason to run here
if not options.refit: 
   if options.verbose: sys.stderr.write("Computing fit for null model\n")
   L.fit()
   if options.verbose and not options.kfile2: sys.stderr.write("\t heritability=%0.3f, sigma=%0.3f\n" % (L.optH,L.optSigma))
   if options.verbose and options.kfile2: sys.stderr.write("\t heritability=%0.3f, sigma=%0.3f, w=%0.3f\n" % (L.optH,L.optSigma,L.optW))

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
   v = np.isnan(x).reshape((-1,))
   # Check SNPs for missing values
   if v.sum():
      keeps = True - v
      xs = x[keeps,:]
      if keeps.sum() <= 1 or xs.var() <= 1e-6: 
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
      if options.kfile2: K2s = K2[keeps,:][:,keeps]
      if options.kfile2: Ls = LMM_withK2(Ys,Ks,X0=X0s,verbose=options.verbose,K2=K2s)
      else: Ls = LMM(Ys,Ks,X0=X0s,verbose=options.verbose)
      if options.refit: Ls.fit(X=xs,REML=options.REML)
      else: 
	 #try:
	 Ls.fit(REML=options.REML)
	 #except: pdb.set_trace()
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

