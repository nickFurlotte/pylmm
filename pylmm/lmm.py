# pyLMM software Copyright 2012, Nicholas A. Furlotte
# Version 0.1

#License Details
#---------------

# The program is free for academic use. Please contact Nick Furlotte
# <nick.furlotte@gmail.com> if you are interested in using the software for
# commercial purposes.

# The software must not be modified and distributed without prior
# permission of the author.
# Any instance of this software must retain the above copyright notice.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import time
import numpy as np
import numpy.linalg as linalg
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as pl
import pdb


class LMM:

   """
	 This is a simple version of EMMA/fastLMM.  
	 The main purpose of this module is to take a phenotype vector (Y), a set of covariates (X) and a kinship matrix (K)
	 and to optimize this model by finding the maximum-likelihood estimates for the model parameters.
	 There are three model parameters: heritability (h), covariate coefficients (beta) and the total
	 phenotypic variance (sigma).
	 Heritability as defined here is the proportion of the total variance (sigma) that is attributed to 
	 the kinship matrix.

	 For simplicity, we assume that everything being input is a numpy array.
	 If this is not the case, the module may throw an error as conversion from list to numpy array
	 is not done consistently.

   """
   def __init__(self,Y,K,Kva=[],Kve=[],X0=None):

      """
      The constructor takes a phenotype vector or array of size n.
      It takes a kinship matrix of size n x n.  Kva and Kve can be computed as Kva,Kve = linalg.eigh(K) and cached.
      If they are not provided, the constructor will calculate them.
      X0 is an optional covariate matrix of size n x q, where there are q covariates.
      When this parameter is not provided, the constructor will set X0 to an n x 1 matrix of all ones to represent a mean effect.
      """

      if not X0 == None: X0 = np.ones(len(Y)).reshape(len(Y),1)

      x = Y != -9
      if not x.sum() == len(Y):
	 sys.stderr.write("Removing %d missing values from Y\n" % ((True - x).sum()))
	 Y = Y[x]
	 K = K[x,:][:,x]
	 X0 = X0[x,:]
	 Kva = []
	 Kve = []
      self.nonmissing = x

      if len(Kva) == 0 or len(Kve) == 0:
	 sys.stderr.write("Obtaining eigendecomposition for %dx%d matrix\n" % (K.shape[0],K.shape[1]) )
	 begin = time.time()
	 Kva,Kve = linalg.eigh(K)
	 end = time.time()
	 sys.stderr.write("Total time: %0.3f\n" % (end - begin))
      self.K = K
      self.Kva = Kva
      self.Kve = Kve
      self.Y = Y
      self.X0 = X0
      self.N = self.K.shape[0]

      self.transform()

   def transform(self):

      """
	 Computes a transformation on the phenotype vector and the covariate matrix.
	 The transformation is obtained by left multiplying each parameter by the transpose of the 
	 eigenvector matrix of K (the kinship).
      """

      self.Yt = np.dot(self.Kve.T, self.Y)
      self.X0t = np.dot(self.Kve.T, self.X0)

   def getMLSoln(self,h,X):

      """
	 Obtains the maximum-likelihood estimates for the covariate coefficients (beta),
	 the total variance of the trait (sigma) and also passes intermediates that can 
	 be utilized in other functions. The input parameter h is a value between 0 and 1 and represents
	 the heritability or the proportion of the total variance attributed to genetics.  The X is the 
	 covariate matrix.
      """
   
      S = 1.0/(h*self.Kva + (1.0 - h))
      Xt = X.T*S
      XX = np.dot(Xt,X)
      XX_i = linalg.inv(XX)
      beta =  np.dot(np.dot(XX_i,Xt),self.Yt)
      Yt = self.Yt - np.dot(X,beta)
      Q = np.dot(Yt.T*S,Yt)
      sigma = Q * 1.0 / (float(len(self.Yt)) - float(X.shape[1]))
      return beta,sigma,Q,XX_i,XX

   def LL_brent(self,h,X=None,REML=False): return -self.LL(h,X,stack=False,REML=REML)[0]
   def LL(self,h,X=None,stack=True,REML=False):

      """
	 Computes the log-likelihood for a given heritability (h).  If X==None, then the 
	 default X0t will be used.  If X is set and stack=True, then X0t will be matrix concatenated with
	 the input X.  If stack is false, then X is used in place of X0t in the LL calculation.
	 REML is computed by adding additional terms to the standard LL and can be computed by setting REML=True.
      """

      if X == None: X = self.X0t
      elif stack: X = np.hstack([self.X0t,np.dot(self.Kve.T, X)])

      n = float(self.N)
      q = float(X.shape[1])
      beta,sigma,Q,XX_i,XX = self.getMLSoln(h,X)
      LL = n*np.log(2*np.pi) + np.log(h*self.Kva + (1.0-h)).sum() + n + n*np.log(1.0/n * Q)
      LL = -0.5 * LL

      if REML:
	 LL_REML_part = q*np.log(2.0*np.pi*sigma) + np.log(linalg.det(np.dot(X.T,X))) - np.log(linalg.det(XX))
	 LL = LL + 0.5*LL_REML_part

      return LL,beta,sigma,XX_i

   def getMax(self,H, X=None,REML=False):

      """
	 Helper functions for .fit(...).  
	 This function takes a set of LLs computed over a grid and finds possible regions 
	 containing a maximum.  Within these regions, a Brent search is performed to find the 
	 optimum.

      """
      n = len(self.LLs)
      HOpt = []
      for i in range(1,n-2):
	 if self.LLs[i-1] < self.LLs[i] and self.LLs[i] > self.LLs[i+1]: HOpt.append(optimize.brent(self.LL_brent,args=(X,REML),brack=(H[i-1],H[i+1])))

      if len(HOpt) > 1: 
	 sys.stderr.write("ERR: Found multiple maximum.  Returning first...\n")
	 return HOpt[0]
      elif len(HOpt) == 1: return HOpt[0]
      elif self.LLs[0] > self.LLs[n-1]: return H[0]
      else: return H[n-1]

   def fit(self,X=None,ngrids=100,REML=True):

      """
	 Finds the maximum-likelihood solution for the heritability (h) given the current parameters.
	 X can be passed and will transformed and concatenated to X0t.  Otherwise, X0t is used as 
	 the covariate matrix.

	 This function calculates the LLs over a grid and then uses .getMax(...) to find the optimum.
	 Given this optimum, the function computes the LL and associated ML solutions.
      """
   
      if X == None: X = self.X0t
      else: X = np.hstack([self.X0t,np.dot(self.Kve.T, X)])
      H = np.array(range(ngrids)) / float(ngrids)
      L = np.array([self.LL(h,X,stack=False,REML=REML)[0] for h in H])
      self.LLs = L

      hmax = self.getMax(H,X,REML)
      L,beta,sigma,betaSTDERR = self.LL(hmax,X,stack=False,REML=REML)
      
      self.H = H
      self.optH = hmax
      self.optLL = L
      self.optBeta = beta
      self.optSigma = sigma

      return hmax,beta,sigma,L

   def association(self,X, h = None, stack=True,REML=True):

      """
	Calculates association statitics for the SNPs encoded in the vector X of size n.
	If h == None, the optimal h stored in optH is used.

      """

      if stack: X = np.hstack([self.X0t,np.dot(self.Kve.T, X)])
      if h == None: h = self.optH

      L,beta,sigma,betaSTDERR = self.LL(h,X,stack=False,REML=REML)
      q  = len(beta)
      ts,ps = self.tstat(beta[q-1],betaSTDERR[q-1,q-1],sigma,q)
      return ts,ps

   def tstat(self,beta,stderr,sigma,q): 

	 """
	    Calculates a t-statistic and associated p-value given the estimate of beta and its standard error.
	    This is actually an F-test, but when only one hypothesis is being performed, it reduces to a t-test.
	 """

	 ts = beta / np.sqrt(stderr * sigma)	 
	 ps = 2.0*(1.0 - stats.t.cdf(np.abs(ts), self.N-q))
	 return ts,ps

   def plotFit(self,color='b-',title=''):

      """
	 Simple function to visualize the likelihood space.  It takes the LLs 
	 calcualted over a grid and normalizes them by subtracting off the mean and exponentiating.
	 The resulting "probabilities" are normalized to one and plotted against heritability.
	 This can be seen as an approximation to the posterior distribuiton of heritability.

	 For diagnostic purposes this lets you see if there is one distinct maximum or multiple 
	 and what the variance of the parameter looks like.
      """
      mx = self.LLs.max()
      p = np.exp(self.LLs - mx)
      p = p/p.sum()

      pl.plot(self.H,p,color)
      pl.xlabel("Heritability")
      pl.ylabel("Probability of data")
      pl.title(title)

