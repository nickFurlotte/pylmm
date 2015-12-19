#!/usr/bin/python


# This script illustrates how the pylmm module can be used to compute association 
# statistics.  

import sys
import time

import numpy as np
from pylmm import lmm

Y = np.genfromtxt('data/hmdp.liver.exprs.1')

# Loading npdump and first 1000 snps for speed
K = np.load('data/hmdp.liver.K.npdump', encoding='latin1')
snps = np.load('data/hmdp.liver.snps.1000.npdump', encoding='latin1').T

# These three lines will load all SNPs (from npdump or from txt) and 
# calculate the kinship
#snps = np.genfromtxt('data/hmdp.liver.snps').T
#snps = np.load('data/hmdp.liver.snps.npdump').T
#K = lmm.calculateKinship(snps)

# Instantiate a LMM object for the phentoype Y and fit the null model
L = lmm.LMM(Y,K)
L.fit()

# Manually calculate the association at one SNP
X = snps[:,0]
X[np.isnan(X)] = X[True - np.isnan(X)].mean() # Fill missing with MAF
X = X.reshape(len(X),1)
if X.var() == 0:
    ts,ps = (np.nan,np.nan)
else:
    ts,ps = L.association(X)

# If I want to refit the variance component
L.fit(X=X)
ts,ps = L.association(X)

# If I want to do a genome-wide scan over the 1000 SNPs.
# This call will use REML (REML = False means use ML).
# It will also refit the variance components for each SNP.
# Setting refit = False will cause the program to fit the model once
# and hold those variance component estimates for each SNP.
begin = time.time()
TS,PS = lmm.GWAS(Y,snps,K,REML=True,refit=True)
end = time.time()
sys.stderr.write("Total time for 1000 SNPs: %0.3f\n" % (end- begin))



