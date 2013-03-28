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


# This script illustrates how the pylmm module can be used to compute association 
# statistics.  

import sys
import time

import numpy as np
from pylmm import lmm

Y = np.genfromtxt('data/hmdp.liver.exprs.1')

# Loading npdump and first 1000 snps for speed
K = np.load('data/hmdp.liver.K.npdump')
snps = np.load('data/hmdp.liver.snps.1000.npdump').T

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
if X.var() == 0: ts,ps = (np.nan,np.nan)
else: ts,ps = L.association(X)

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



