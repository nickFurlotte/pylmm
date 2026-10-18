
# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.


import sys
import numpy as np
from pylmm.lmm import LMM

def fitTwo(y,K1,K2,X0=None,wgrids=100):
    '''
       Simple function to fit a model with two variance components.
       It works by running the standard pylmm algorithm in a loop
       where at each iteration of the loop a new kinship is generated
       as a linear combination of the original two.
    '''

    # Create a uniform grid
    W = np.array(range(wgrids)) / float(wgrids)
    Res = []
    LLs = []

    for w in W:
        # heritability will be estimated for linear combo of kinships
        K = w*K1 + (1.0 - w)*K2
        sys.stderr.write("Fitting weight %0.2f\n" % (w))
        L = LMM(y,K,X0=X0)
        R = L.fit()
        Res.append(R)
        LLs.append(R[-1])

        del K

    L = np.array(LLs)
    i = np.where(L == L.max())[0]
    if len(i) > 1:
        sys.stderr.write("WARNING: Found multiple maxes using first one\n")

    i = i[0]
    hmax,beta,sigma,LL = Res[i]
    w = W[i]

    h1 = w * hmax
    h2 = (1.0 - w) * hmax
    e = (1.0 - hmax)

    return h1,h2,e,beta,sigma,LL
