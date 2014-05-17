
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


#!/usr/bin/python

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
 
