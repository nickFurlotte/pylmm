#!/usr/bin/python

import sys
import time

import numpy as np
from pyLMM import lmm

Y = np.genfromtxt('data/hmdp.liver.exprs.1')
#snps = np.genfromtxt('data/hmdp.liver.snps')
snps = np.load('data/hmdp.liver.snps.npdump')
K = lmm.calculateKinship(snps.T)

L = lmm.LMM(Y,K)
PS = []

begin = time.time()
for i in range(1000,2000):
   if i % 100 == 0: sys.stderr.write("SNP %d\n" % i)
   X = snps[i,:]
   X[np.isnan(X)] = X[True - np.isnan(X)]
   X = X.reshape(len(X),1)

   if X.var() == 0: PS.append(np.nan)
   else:
      L.fit(X=X)
      ps = L.association(X)
      PS.append(ps)

end = time.time()
sys.stderr.write("Total Time %0.3f\n" % (end - begin))


