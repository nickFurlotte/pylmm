# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

import os
import sys
import warnings

import numpy as np


class plink:
    def __init__(
        self,
        fbase,
        kFile=None,
        phenoFile=None,
        type="b",
        normGenotype=True,
        readKFile=False,
        fastLMM_kinship=False,
    ):

        self.fhandle = None
        self.snpFileHandle = None
        self.fbase = fbase
        self.type = type
        if not type == "emma":
            self.indivs = self.getIndivs(self.fbase, type)
        else:
            # Just read a line from the SNP file and see how many individuals we have
            f = open(fbase)
            self.indivs = list(range(len(f.readline().strip().split())))
            f.close()
        self.kFile = kFile
        self.phenos = None
        self.normGenotype = normGenotype
        self.phenoFile = phenoFile
        # Originally I was using the fastLMM style that has indiv IDs embedded.
        # NOW I want to use this module to just read SNPs so I'm allowing
        # the programmer to turn off the kinship reading.
        self.readKFile = readKFile
        self.fastLMM_kinship_style = fastLMM_kinship

        if self.kFile:
            self.K = self.readKinship(self.kFile)
        elif os.path.isfile("%s.kin" % fbase):
            self.kFile = "%s.kin" % fbase
            if self.readKFile:
                self.K = self.readKinship(self.kFile)
        else:
            self.kFile = None
            self.K = None

        self.getPhenos(self.phenoFile)

        self.fhandle = None
        self.snpFileHandle = None

    def __del__(self):
        if self.fhandle:
            self.fhandle.close()
        if self.snpFileHandle:
            self.snpFileHandle.close()

    def getSNPIterator(self):
        if self.type == "b":
            return self.getSNPIterator_bed()
        elif self.type == "t":
            return self.getSNPIterator_tped()
        elif self.type == "emma":
            return self.getSNPIterator_emma()
        else:
            sys.stderr.write("Please set type to either b or t\n")
            return

    def getSNPIterator_emma(self):
        self.have_read = 0
        self.numSNPs = -1
        file = self.fbase
        self.fhandle = open(file)

        return self

    def getSNPIterator_tped(self):
        # get the number of snps
        file = self.fbase + ".bim"
        if not os.path.isfile(file):
            file = self.fbase + ".map"
        i = 0
        f = open(file)
        for line in f:
            i += 1
        f.close()
        self.numSNPs = i
        self.have_read = 0
        self.snpFileHandle = open(file)

        file = self.fbase + ".tped"
        self.fhandle = open(file)

        return self

    def getSNPIterator_bed(self):
        # get the number of snps
        file = self.fbase + ".bim"
        i = 0
        f = open(file)
        for line in f:
            i += 1
        f.close()
        self.numSNPs = i
        self.have_read = 0
        self.snpFileHandle = open(file)

        self.BytestoRead = self.N // 4 + ((self.N % 4 and 1) or 0)

        file = self.fbase + ".bed"
        self.fhandle = open(file, "rb")

        magicNumber = self.fhandle.read(2)
        order = self.fhandle.read(1)
        if not order == b"\x01":
            sys.stderr.write("This is not in SNP major order - you did not handle this case\n")
            raise StopIteration

        return self

    def __iter__(self):
        return self.getSNPIterator()

    def __next__(self):
        if self.have_read == self.numSNPs:
            raise StopIteration
        self.have_read += 1

        if self.type == "b":
            X = self.fhandle.read(self.BytestoRead)
            if len(X) != self.BytestoRead:
                raise ValueError(
                    "Truncated BED file %s.bed: expected %d bytes for SNP %d, got %d"
                    % (self.fbase, self.BytestoRead, self.have_read, len(X))
                )
            return self.formatBinaryGenotypes(
                X, self.normGenotype
            ), self.snpFileHandle.readline().strip().split()[1]

        elif self.type == "t":
            X = self.fhandle.readline()
            XX = X.strip().split()
            chrm, rsid, pos1, pos2 = tuple(XX[:4])
            XX = XX[4:]
            G = self.getGenos_tped(XX)
            if self.normGenotype:
                G = self.normalizeGenotype(G)
            return G, self.snpFileHandle.readline().strip().split()[1]

        elif self.type == "emma":
            X = self.fhandle.readline()
            if X == "":
                raise StopIteration
            XX = X.strip().split()
            G = []
            for x in XX:
                try:
                    G.append(float(x))
                except:
                    G.append(np.nan)
            G = np.array(G)
            if self.normGenotype:
                G = self.normalizeGenotype(G)
            return G, "SNP_%d" % self.have_read

        else:
            sys.stderr.write("Do not understand type %s\n" % (self.type))

    # Python 2 spelling, kept so existing callers of IN.next() keep working.
    next = __next__

    def getGenos_tped(self, X):
        G = []
        for i in range(0, len(X) - 1, 2):
            a = X[i]
            b = X[i + 1]
            if a == b == "0":
                g = np.nan
            if a == b == "1":
                g = 0
            if a == b == "2":
                g = 1
            if a != b:
                g = 0.5
            G.append(g)
        return np.array(G)

    # 2-bit PLINK code -> genotype value.  The four codes of each byte are
    # read from the low bits up (individual 4k+j is bits 2j..2j+1).  Same
    # mapping as the original string-based decoder: low bit 0/high bit 0 ->
    # 0.0, 01 (missing) -> nan, 10 -> 0.5, 11 -> 1.0, where the pair is
    # written high bit first.
    _BED_CODE_VALUES = np.array([0.0, np.nan, 0.5, 1.0])
    _BED_SHIFTS = np.array([0, 2, 4, 6], dtype=np.uint8)

    def formatBinaryGenotypes(self, X, norm=True):
        # X is the raw bytes for one SNP (self.BytestoRead bytes).
        B = np.frombuffer(X, dtype=np.uint8)
        codes = (B[:, None] >> self._BED_SHIFTS) & 3
        # only take the leading values because whatever is left should be null
        G = self._BED_CODE_VALUES[codes.reshape(-1)[: self.N]]
        if norm:
            G = self.normalizeGenotype(G)
        return G

    def normalizeGenotype(self, G):
        x = ~np.isnan(G)
        # All missing: return the all-nan SNP unchanged (length n), which the
        # CLIs then skip.  It used to return an empty array (bug 11).
        if not x.any():
            return G
        m = G[x].mean()
        # A SNP that is constant up to rounding (e.g. an EMMA dosage) is
        # returned as zeros, which the CLIs skip: standardising it would blow
        # its rounding noise up to unit variance (issue 19).
        if np.ptp(G[x]) <= 1e-12 * max(1.0, abs(m)):
            G = np.zeros_like(G)
            return G
        if G[x].var() == 0:
            s = 1.0
        else:
            s = np.sqrt(G[x].var())
        G[np.isnan(G)] = m
        if s == 0:
            G = G - m
        else:
            G = (G - m) / s

        return G

    def getPhenos(self, phenoFile=None):
        """Read a PLINK-style phenotype file (FID IID value...; NA or -9 is
        missing) into self.phenos, one row per individual of self.indivs, in
        that order, matched by (FID, IID).  Individuals absent from the file
        get nan (and a warning with the count); the file's extra individuals
        are ignored.  (Bug 10, fixed in Phase 5.2: absent individuals used to
        be skipped, which shifted the rows of everyone after them.)  For EMMA
        input, which has no IDs, rows are taken in order and their number
        must equal the number of individuals.
        """
        if not phenoFile:
            self.phenoFile = phenoFile = self.fbase + ".phenos"
        if not os.path.isfile(phenoFile):
            # sys.stderr.write("Could not find phenotype file: %s\n" % (phenoFile))
            return
        keys, P = self._readTable(phenoFile, lambda x: x == "NA" or x == "-9", "phenotype")
        self.phenos = self._alignRows(keys, P, phenoFile, "phenotype")
        return self.phenos

    def getPhenosEMMA(self, emmaFile):
        """EMMA phenotypes: one phenotype per row, one column per individual
        (in genotype order); anything that is not a number is missing."""
        P = []
        with open(emmaFile) as f:
            for line in f:
                v = line.strip().split()
                if not v:
                    continue
                p = []
                for x in v:
                    try:
                        p.append(float(x))
                    except ValueError:
                        p.append(np.nan)
                P.append(p)
        return self._emmaColumns(P, emmaFile, "phenotype")

    def _emmaColumns(self, rows, fn, what):
        n = len(self.indivs)
        bad = [i for i, r in enumerate(rows) if len(r) != n]
        if bad or not rows:
            raise ValueError(
                "EMMA %s file %s: expected rows of %d values (one per individual), %s"
                % (
                    what,
                    fn,
                    n,
                    "found no rows"
                    if not rows
                    else "row %d has %d" % (bad[0] + 1, len(rows[bad[0]])),
                )
            )
        return np.array(rows, dtype=np.float64).T

    @staticmethod
    def _readTable(fn, isMissing, what):
        """(keys, values) of a FID IID v1 v2 ... file.  A first line whose first two
        fields are FID IID (case-insensitive) is a header and is skipped."""
        keys = []
        P = []
        with open(fn) as f:
            first = True
            for lineno, line in enumerate(f, 1):
                v = line.strip().split()
                if not v:
                    continue
                # a header line starting FID IID (any case) is skipped
                if first and len(v) >= 2 and v[0].upper() == "FID" and v[1].upper() == "IID":
                    first = False
                    continue
                first = False
                if len(v) < 3:
                    raise ValueError(
                        "%s file %s, line %d: expected FID IID and at least one value"
                        % (what, fn, lineno)
                    )
                try:
                    P.append([np.nan if isMissing(x.strip()) else float(x) for x in v[2:]])
                except ValueError:
                    raise ValueError(
                        "%s file %s, line %d: a value is not a number (missing values are "
                        "NA%s)" % (what, fn, lineno, " or -9" if what == "phenotype" else "")
                    )
                keys.append((v[0], v[1]))
        widths = set(len(p) for p in P)
        if len(widths) > 1:
            raise ValueError(
                "%s file %s: rows have different numbers of values (%s)"
                % (what, fn, ", ".join(str(w) for w in sorted(widths)))
            )
        return keys, P

    def _alignRows(self, keys, P, fn, what):
        """Rows of P (keyed by (FID, IID)) in self.indivs order; nan rows for
        absent individuals.  Raises ValueError for duplicate IDs or when no
        individual matches."""
        if not P:
            raise ValueError("%s file %s has no data rows" % (what, fn))
        if self.type == "emma":
            if len(P) != len(self.indivs):
                raise ValueError(
                    "%s file %s has %d rows but the EMMA genotypes have %d individuals; "
                    "with EMMA input the rows are matched by position"
                    % (what, fn, len(P), len(self.indivs))
                )
            return np.array(P, dtype=np.float64)
        D = {}
        for i, k in enumerate(keys):
            if k in D:
                raise ValueError("%s file %s lists individual %s %s twice" % (what, fn, k[0], k[1]))
            D[k] = i
        idx = [D.get(k, -1) for k in self.indivs]
        absent = [k for k, i in zip(self.indivs, idx) if i < 0]
        if len(absent) == len(self.indivs):
            raise ValueError(
                "none of the %d individuals of the genotype file appear in the %s file %s "
                "(matched on the first two columns, FID and IID; e.g. %s %s)"
                % (len(self.indivs), what, fn, self.indivs[0][0], self.indivs[0][1])
            )
        ncol = len(P[0])
        out = np.full((len(self.indivs), ncol), np.nan)
        for r, i in enumerate(idx):
            if i >= 0:
                out[r] = P[i]
        if absent:
            warnings.warn(
                "%d of %d individuals are not in the %s file %s and are treated as missing "
                "(e.g. %s)"
                % (
                    len(absent),
                    len(self.indivs),
                    what,
                    fn,
                    ", ".join("%s %s" % k for k in absent[:5]),
                ),
                stacklevel=3,
            )
        return out

    def getIndivs(self, base, type="b"):
        if type == "t":
            famFile = "%s.tfam" % base
        else:
            famFile = "%s.fam" % base

        keys = []
        i = 0
        f = open(famFile)
        for line in f:
            v = line.strip().split()
            famId = v[0]
            indivId = v[1]
            k = (famId.strip(), indivId.strip())
            keys.append(k)
            i += 1
        f.close()

        self.N = len(keys)
        sys.stderr.write("Read %d individuals from %s\n" % (self.N, famFile))

        return keys

    def readKinship(self, kFile):
        # Assume the fastLMM style
        # This will read in the kinship matrix and then reorder it
        # according to self.indivs - additionally throwing out individuals
        # that are not in both sets
        if self.indivs is None or len(self.indivs) == 0:
            sys.stderr.write("Did not read any individuals so can't load kinship\n")
            return

        sys.stderr.write("Reading kinship matrix from %s\n" % (kFile))

        f = open(kFile)
        # read indivs
        if self.fastLMM_kinship_style:
            v = f.readline().strip().split("\t")[1:]
            keys = [tuple(y.split()) for y in v]
            D = {}
            for i in range(len(keys)):
                D[keys[i]] = i

        # read matrix
        K = []
        if self.fastLMM_kinship_style:
            for line in f:
                K.append([float(x) for x in line.strip().split("\t")[1:]])
        else:
            for line in f:
                K.append([float(x) for x in line.strip().split()])
        f.close()
        K = np.array(K)

        if self.fastLMM_kinship_style:
            # reorder to match self.indivs
            L = []
            KK = []
            X = []
            for i in range(len(self.indivs)):
                if self.indivs[i] not in D:
                    X.append(self.indivs[i])
                else:
                    KK.append(self.indivs[i])
                    L.append(D[self.indivs[i]])
            K = K[L, :][:, L]
            self.indivs = KK
            self.indivs_removed = X
            if len(self.indivs_removed):
                sys.stderr.write(
                    "Removed %d individuals that did not appear in Kinship\n"
                    % (len(self.indivs_removed))
                )

        return K

    def getCovariatesEMMA(self, emmaFile):
        P = []
        with open(emmaFile) as f:
            for line in f:
                v = line.strip().split()
                if not v:
                    continue
                try:
                    P.append([np.nan if x == "NA" else float(x) for x in v])
                except ValueError:
                    raise ValueError(
                        "EMMA covariate file %s: a value is not a number (missing is NA)" % emmaFile
                    )
        return self._emmaColumns(P, emmaFile, "covariate")

    def getCovariates(self, covFile=None):
        """A PLINK-style covariate file (FID IID c1 c2 ...; NA is missing),
        matched to self.indivs by (FID, IID) like getPhenos; absent
        individuals get nan rows."""
        if not os.path.isfile(covFile):
            raise ValueError("Could not find covariate file: %s" % (covFile))
        keys, P = self._readTable(covFile, lambda x: x == "NA", "covariate")
        return self._alignRows(keys, P, covFile, "covariate")
