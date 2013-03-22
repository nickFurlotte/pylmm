
# pylmm is a python-based linear mixed-model solver with applications to GWAS

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


import os
import sys
import numpy as np
import struct
import pdb

class plink:
   def __init__(self,fbase,kFile=None,phenoFile=None,type='b',normGenotype=True,readKFile=False):

      self.fbase = fbase
      self.type = type
      self.indivs = self.getIndivs(self.fbase,type)
      self.kFile = kFile
      self.phenos = None
      self.normGenotype = normGenotype
      self.phenoFile = phenoFile
      # Originally I was using the fastLMM style that has indiv IDs embedded.
      # NOW I want to use this module to just read SNPs so I'm allowing 
      # the programmer to turn off the kinship reading.
      self.readKFile = readKFile

      if self.kFile: self.K = self.readKinship(self.kFile)
      elif os.path.isfile("%s.kin" % fbase): 
	 self.kFile = "%s.kin" %fbase
	 if self.readKFile: self.K = self.readKinship(self.kFile)
      else: 
	 self.kFile = None
	 self.K = None

      self.getPhenos(self.phenoFile)

      self.fhandle = None
      self.snpFileHandle = None

   def __del__(self): 
      if self.fhandle: self.fhandle.close()
      if self.snpFileHandle: self.snpFileHandle.close()

   def getSNPIterator(self):
      if not self.type == 'b': 
	 sys.stderr.write("Have only implemented this for binary plink files (bed)\n")
	 return

      # get the number of snps
      file = self.fbase + '.bim'
      i = 0
      f = open(file,'r')
      for line in f: i += 1
      f.close()
      self.numSNPs = i
      self.have_read = 0
      self.snpFileHandle = open(file,'r')

      self.BytestoRead = self.N / 4 + (self.N % 4 and 1 or 0)
      self._formatStr = 'c'*self.BytestoRead

      file = self.fbase + '.bed'
      self.fhandle = open(file,'rb')

      magicNumber = self.fhandle.read(2)
      order = self.fhandle.read(1)
      if not order == '\x01': 
	 sys.stderr.write("This is not in SNP major order - you did not handle this case\n")
	 raise StopIteration

      return self

   def __iter__(self): return self.getSNPIterator()

   def next(self):
      if self.have_read == self.numSNPs: raise StopIteration
      X = self.fhandle.read(self.BytestoRead)
      XX = [bin(ord(x)) for x in struct.unpack(self._formatStr,X)]
      self.have_read += 1
      return self.formatBinaryGenotypes(XX,self.normGenotype),self.snpFileHandle.readline().strip().split()[1]

   def formatBinaryGenotypes(self,X,norm=True):
	 D = { \
	       '00': 0.0, \
	       '10': 0.5, \
	       '11': 1.0, \
	       '01': np.nan \
	    }

	 D_tped = { \
	       '00': '1 1', \
	       '10': '1 2', \
	       '11': '2 2', \
	       '01': '0 0' \
	    }

	 #D = D_tped
	       
	 G = []
	 for x in X:
	    if not len(x) == 10:
	       xx = x[2:]
	       x = '0b' + '0'*(8 - len(xx)) + xx
	    a,b,c,d = (x[8:],x[6:8],x[4:6],x[2:4]) 
	    L = [D[y] for y in [a,b,c,d]]
	    G += L
	 # only take the leading values because whatever is left should be null
	 G = G[:self.N]
	 G = np.array(G)
	 if norm: G = self.normalizeGenotype(G)
	 return G

   def normalizeGenotype(self,G):
      x = True - np.isnan(G)
      m = G[x].mean()
      s = np.sqrt(G[x].var())
      G[np.isnan(G)] = m
      G = (G - m) / s
      return G

   def getPhenos(self,phenoFile=None):
      if not phenoFile: self.phenoFile = phenoFile = self.fbase+".phenos"
      if not os.path.isfile(phenoFile): 
	 sys.stderr.write("Could not find phenotype file: %s\n" % (phenoFile))
	 return
      f = open(phenoFile,'r')
      keys = []
      P = []
      for line in f:
	 v = line.strip().split()
	 keys.append((v[0],v[1]))
	 P.append([(x == 'NA' or x == '-9') and np.nan or float(x) for x in v[2:]])
      f.close()
      P = np.array(P)

      # reorder to match self.indivs
      D = {}
      L = []
      for i in range(len(keys)): D[keys[i]] = i
      for i in range(len(self.indivs)):
	 if not D.has_key(self.indivs[i]): continue 
	 L.append(D[self.indivs[i]])
      P = P[L,:]

      self.phenos = P
      return P

   def getIndivs(self,base,type='b'):
      if type == 't': famFile = "%s.tfam" % base
      else: famFile = "%s.fam" % base

      keys = []
      i = 0
      f = open(famFile,'r')
      for line in f:
	 v = line.strip().split()
	 famId = v[0]
	 indivId = v[1]
	 k = (famId.strip(),indivId.strip())
	 keys.append(k)
	 i += 1
      f.close()

      self.N = len(keys)
      sys.stderr.write("Read %d individuals from %s\n" % (self.N, famFile))

      return keys

   def readKinship(self,kFile):
      # Assume the fastLMM style
      # This will read in the kinship matrix and then reorder it
      # according to self.indivs - additionally throwing out individuals 
      # that are not in both sets
      if self.indivs == None or len(self.indivs) == 0:
	 sys.stderr.write("Did not read any individuals so can't load kinship\n")
	 return 

      sys.stderr.write("Reading kinship matrix from %s\n" % (kFile) )

      f = open(kFile,'r')
      # read indivs 
      v = f.readline().strip().split("\t")[1:]
      keys = [tuple(y.split()) for y in v]
      D = {}
      for i in range(len(keys)): D[keys[i]] = i

      # read matrix
      K = []
      for line in f: K.append([float(x) for x in line.strip().split("\t")[1:]])
      f.close()
      K  = np.array(K)

      # reorder to match self.indivs
      L = []
      KK = []
      X = []
      for i in range(len(self.indivs)):
	 if not D.has_key(self.indivs[i]): X.append(self.indivs[i])
	 else: 
	    KK.append(self.indivs[i])
	    L.append(D[self.indivs[i]])
      K = K[L,:][:,L]
      self.indivs = KK
      self.indivs_removed = X
      if len(self.indivs_removed): sys.stderr.write("Removed %d individuals that did not appear in Kinship\n" % (len(self.indivs_removed)))
      return K 

   def getCovariates(self,covFile=None):
      if not os.path.isfile(covFile): 
	 sys.stderr.write("Could not find covariate file: %s\n" % (phenoFile))
	 return
      f = open(covFile,'r')
      keys = []
      P = []
      for line in f:
	 v = line.strip().split()
	 keys.append((v[0],v[1]))
	 P.append([x == 'NA' and np.nan or float(x) for x in v[2:]])
      f.close()
      P = np.array(P)

      # reorder to match self.indivs
      D = {}
      L = []
      for i in range(len(keys)): D[keys[i]] = i
      for i in range(len(self.indivs)):
	 if not D.has_key(self.indivs[i]): continue 
	 L.append(D[self.indivs[i]])
      P = P[L,:]

      return P


      


      

      
	    
