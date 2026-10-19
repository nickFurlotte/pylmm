# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

r"""Readers for PLINK (BED, TPED) and EMMA genotype, phenotype and covariate files.

The single entry point is :class:`PlinkReader`.  (This module was ``pylmm.input``
before Phase 5.4.)

Supported formats
-----------------
``type="b"``: PLINK binary (BED/BIM/FAM)
    ``<fbase>.bed`` holds the genotypes, ``<fbase>.bim`` the SNPs (six
    columns: chromosome, SNP id, genetic distance, position, A1, A2) and
    ``<fbase>.fam`` the individuals (FID, IID, ...).  Only SNP-major BED files
    are supported (third header byte ``0x01``).  Each SNP is two bits per
    individual, packed four to a byte.  The SNP name returned by the iterator
    is the second column of the ``.bim`` line.

``type="t"``: PLINK transposed text (TPED/TFAM)
    ``<fbase>.tped`` has one SNP per line: ``chr snp cM pos`` followed by two
    allele fields per individual.  ``<fbase>.tfam`` lists the individuals
    (FID, IID, ...).  SNP names are read from ``<fbase>.bim`` if it exists,
    otherwise from ``<fbase>.map``, which must list the same SNPs in the same
    order.  Alleles must already be recoded as ``1``/``2`` (``0`` is missing);
    nucleotide letters are not supported.

``type="emma"``: EMMA genotype matrix
    ``fbase`` is the path of a whitespace-separated numeric text file with
    one SNP per row and one column per individual (no header, no IDs).
    Values are used as given (typically 0/1/2 or dosages); anything that does
    not parse as a number is missing.  SNPs are named ``SNP_1``, ``SNP_2``, ...
    and individuals are identified by position only.

Phenotype and covariate files are PLINK-style text tables ``FID IID v1 [v2
...]`` with an optional ``FID IID ...`` header line (any case); EMMA ones are
plain numeric matrices (see :meth:`PlinkReader.get_phenos_emma` and
:meth:`PlinkReader.get_covariates_emma`).  A kinship matrix is a plain
whitespace-separated square matrix in genotype order, or in the FaSTLMM style
(tab-separated, with a header row and a first column of ``FID IID`` labels).

Genotype coding
---------------
A genotype is the **fraction of copies of the second allele** (the ``A2``
allele of the ``.bim`` file; the second allele of each TPED pair after
``1``/``2`` recoding): ``0.0`` (homozygous A1), ``0.5`` (heterozygous) or
``1.0`` (homozygous A2).  It is **not** a 0/1/2 allele count; EMMA input is
passed through unchanged.  With ``norm_genotype=True`` (the default) each SNP
is then standardised (see :meth:`PlinkReader.normalize_genotype`): missing
values are mean-imputed, then the SNP is centred and scaled to unit
variance.  Pass ``norm_genotype=False`` to get the raw 0 / 0.5 / 1 / nan
values.

Missing values
--------------
Missing genotypes are ``numpy.nan`` in raw output (BED code ``01``, TPED
``0 0``, an unparsable EMMA value).  After normalisation there are no nans,
except that an all-missing SNP is returned unchanged (all nan) and a
constant SNP is returned as all zeros; callers skip both.  Missing phenotypes
and covariates (``NA``, and ``-9`` for phenotypes) are ``nan``.

ID alignment
------------
Individuals are identified by the pair ``(FID, IID)`` from the ``.fam`` /
``.tfam`` file, kept in file order in ``PlinkReader.indivs``.  Phenotype and
covariate rows are matched to that order by ``(FID, IID)``, whatever order
the file lists them in: extra individuals in the file are ignored, individuals
missing from the file get ``nan`` rows (with a warning giving the count), and
duplicate IDs or a file with no matching individual raise ``ValueError``.
EMMA input has no IDs, so rows are matched by position and the row count must
equal the number of individuals.  A FaSTLMM-style kinship matrix is reordered
to ``indivs``, and individuals absent from it are dropped from ``indivs``
(listed in ``indivs_removed``).

Examples
--------
Write a tiny BED/BIM/FAM set (3 individuals, 2 SNPs) and read it back:

>>> import os, tempfile
>>> from pylmm.plink_io import PlinkReader
>>> base = os.path.join(tempfile.mkdtemp(), "toy")
>>> with open(base + ".fam", "w") as f:
...     _ = f.write("F1 I1 0 0 1 -9\nF1 I2 0 0 1 -9\nF2 I3 0 0 2 -9\n")
>>> with open(base + ".bim", "w") as f:
...     _ = f.write("1 rs1 0 100 A G\n1 rs2 0 200 C T\n")
>>> with open(base + ".bed", "wb") as f:
...     _ = f.write(bytes([0x6C, 0x1B, 0x01, 0b00111000, 0b00000111]))
>>> reader = PlinkReader(base, type="b", norm_genotype=False)
>>> reader.indivs
[('F1', 'I1'), ('F1', 'I2'), ('F2', 'I3')]
>>> for g, name in reader.get_snp_iterator():
...     print(name, g)
rs1 [0.  0.5 1. ]
rs2 [ 1. nan  0.]

The bundled 132k-SNP example is read the same way; the path is relative to
the repository root, hence the skips:

>>> r = PlinkReader("data/snps.132k.clean.noX", type="b")  # doctest: +SKIP
>>> g, name = next(r.get_snp_iterator())  # doctest: +SKIP
>>> g.shape == (r.N,)  # doctest: +SKIP
True
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Callable
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray

# An (FID, IID) pair.
IndivKey = tuple[str, str]


class PlinkReader:
    """Read genotypes (and optionally phenotypes and a kinship matrix) of one dataset.

    The object is its own iterator: ``for g, name in reader.get_snp_iterator()``
    (or ``for g, name in reader``) yields one SNP at a time as
    ``(genotypes, snp_name)``.  See the module docstring for the file formats,
    the genotype coding, missing values and ID alignment.

    Parameters
    ----------
    fbase : str
        File prefix for ``type="b"`` (``fbase.bed/.bim/.fam``) and ``type="t"``
        (``fbase.tped/.tfam`` plus ``.bim`` or ``.map``); the full path of the
        genotype matrix for ``type="emma"``.
    k_file : str, optional
        Kinship matrix file.  If omitted and ``fbase + ".kin"`` exists, that
        file is used (and read only if ``read_k_file`` is true).
    pheno_file : str, optional
        PLINK-style phenotype file.  Defaults to ``fbase + ".phenos"``, which
        is silently skipped if it does not exist.
    type : {"b", "t", "emma"}, default "b"
        Genotype format: PLINK binary, PLINK transposed text, or EMMA.
    norm_genotype : bool, default True
        If true, each SNP is standardised (mean-imputed, centred, unit
        variance; see `normalize_genotype`).  If false the raw values
        (0, 0.5, 1 or nan, the fraction of A2 alleles) are returned.
    read_k_file : bool, default False
        Whether to load the automatically found ``fbase + ".kin"``.  An
        explicit ``k_file`` is always read.
    fastlmm_kinship : bool, default False
        Whether the kinship file is FaSTLMM style (tab separated, header row
        and first column of ``FID IID`` labels), in which case it is reordered
        to match the individuals.

    Attributes
    ----------
    indivs : list of (str, str)
        ``(FID, IID)`` of each individual in genotype-column order (for EMMA
        input, the integers ``0 .. n-1``).
    N : int
        Number of individuals (not set for EMMA input).
    K : numpy.ndarray or None
        Kinship matrix, shape ``(n, n)``, or None if none was read.
    phenos : numpy.ndarray or None
        Phenotypes, shape ``(n, n_phenotypes)``, aligned with `indivs`; None
        if there is no phenotype file.
    num_snps : int
        Number of SNPs (set when iteration starts; -1 for EMMA, unknown).
    have_read : int
        Number of SNPs returned so far by the current iteration.

    Examples
    --------
    See the module docstring.
    """

    fbase: str
    type: str
    indivs: list[Any]  # TODO(5b): list[IndivKey] for "b"/"t" but list[int] for "emma"
    N: int  # TODO(5b): not assigned for type="emma"
    K: NDArray[np.float64] | None
    phenos: NDArray[np.float64] | None
    k_file: str | None
    pheno_file: str | None
    norm_genotype: bool
    read_k_file: bool
    fastlmm_kinship: bool
    fhandle: IO[Any] | None
    snp_file_handle: IO[str] | None
    num_snps: int
    have_read: int
    _bytes_to_read: int
    indivs_removed: list[IndivKey]

    def __init__(
        self,
        fbase: str,
        k_file: str | None = None,
        pheno_file: str | None = None,
        type: str = "b",
        norm_genotype: bool = True,
        read_k_file: bool = False,
        fastlmm_kinship: bool = False,
    ) -> None:

        self.fhandle = None
        self.snp_file_handle = None
        self.fbase = fbase
        self.type = type
        if type != "emma":
            self.indivs = self._read_fam(self.fbase, type)
        else:
            # Just read a line from the SNP file and see how many individuals we have
            f = open(fbase)
            self.indivs = list(range(len(f.readline().strip().split())))
            f.close()
        self.k_file = k_file
        self.phenos = None
        self.norm_genotype = norm_genotype
        self.pheno_file = pheno_file
        # Originally I was using the fastLMM style that has indiv IDs embedded.
        # NOW I want to use this module to just read SNPs so I'm allowing
        # the programmer to turn off the kinship reading.
        self.read_k_file = read_k_file
        self.fastlmm_kinship = fastlmm_kinship

        if self.k_file:
            self.K = self.read_kinship(self.k_file)
        elif os.path.isfile("%s.kin" % fbase):
            self.k_file = "%s.kin" % fbase
            if self.read_k_file:
                self.K = self.read_kinship(self.k_file)
        else:
            self.k_file = None
            self.K = None

        self.get_phenos(self.pheno_file)

        self.fhandle = None
        self.snp_file_handle = None

    def __del__(self) -> None:
        """Close any open file handles."""
        if self.fhandle:
            self.fhandle.close()
        if self.snp_file_handle:
            self.snp_file_handle.close()

    def get_snp_iterator(self) -> PlinkReader:
        """Open the genotype files and return the iterator (this reader).

        Resets the iteration, so it can be called again to re-read from the
        first SNP.  Each ``next()`` returns ``(genotypes, snp_name)``: a
        float64 array of length ``len(indivs)`` and the SNP's name.

        Returns
        -------
        PlinkReader
            ``self``, positioned before the first SNP.  (If `type` is not a
            known format an error is written to stderr and None is returned.)
        """
        if self.type == "b":
            return self._get_snp_iterator_bed()
        elif self.type == "t":
            return self._get_snp_iterator_tped()
        elif self.type == "emma":
            return self._get_snp_iterator_emma()
        else:
            sys.stderr.write("Please set type to either b or t\n")
            return  # type: ignore[return-value]  # TODO(5b): returns None here

    def _get_snp_iterator_emma(self) -> PlinkReader:
        """Open the EMMA matrix; the number of SNPs is unknown (-1)."""
        self.have_read = 0
        self.num_snps = -1
        file = self.fbase
        self.fhandle = open(file)

        return self

    def _get_snp_iterator_tped(self) -> PlinkReader:
        """Open ``.tped`` and the ``.bim`` (or ``.map``) used for SNP names."""
        # get the number of snps
        file = self.fbase + ".bim"
        if not os.path.isfile(file):
            file = self.fbase + ".map"
        i = 0
        f = open(file)
        for _ in f:
            i += 1
        f.close()
        self.num_snps = i
        self.have_read = 0
        self.snp_file_handle = open(file)

        file = self.fbase + ".tped"
        self.fhandle = open(file)

        return self

    def _get_snp_iterator_bed(self) -> PlinkReader:
        """Open ``.bed`` (checking SNP-major order) and ``.bim``."""
        # get the number of snps
        file = self.fbase + ".bim"
        i = 0
        f = open(file)
        for _ in f:
            i += 1
        f.close()
        self.num_snps = i
        self.have_read = 0
        self.snp_file_handle = open(file)

        self._bytes_to_read = self.N // 4 + ((self.N % 4 and 1) or 0)

        file = self.fbase + ".bed"
        self.fhandle = open(file, "rb")

        self.fhandle.read(2)  # magic number
        order = self.fhandle.read(1)
        if order != b"\x01":
            sys.stderr.write("This is not in SNP major order - you did not handle this case\n")
            raise StopIteration

        return self

    def __iter__(self) -> PlinkReader:
        """Start (or restart) iteration over the SNPs; see `get_snp_iterator`."""
        return self.get_snp_iterator()

    def __next__(self) -> tuple[NDArray[np.float64], str]:
        """Return the next SNP as ``(genotypes, snp_name)``.

        ``genotypes`` is a float64 array of length ``len(indivs)``: the
        fraction of A2 alleles (0, 0.5, 1; nan if missing), standardised if
        ``norm_genotype`` is true.  ``snp_name`` is the SNP id from the
        ``.bim``/``.map`` file (``SNP_<k>`` for EMMA).

        Raises
        ------
        StopIteration
            After the last SNP.
        ValueError
            If the BED file is truncated.
        """
        if self.have_read == self.num_snps:
            raise StopIteration
        self.have_read += 1

        if self.type == "b":
            X = self.fhandle.read(self._bytes_to_read)
            if len(X) != self._bytes_to_read:
                raise ValueError(
                    "Truncated BED file %s.bed: expected %d bytes for SNP %d, got %d"
                    % (self.fbase, self._bytes_to_read, self.have_read, len(X))
                )
            return self._format_binary_genotypes(
                X, self.norm_genotype
            ), self.snp_file_handle.readline().strip().split()[1]

        elif self.type == "t":
            X = self.fhandle.readline()
            xx = X.strip().split()
            chrm, rsid, pos1, pos2 = tuple(xx[:4])
            xx = xx[4:]
            G = self._get_genos_tped(xx)
            if self.norm_genotype:
                G = self.normalize_genotype(G)
            return G, self.snp_file_handle.readline().strip().split()[1]

        elif self.type == "emma":
            X = self.fhandle.readline()
            if X == "":
                raise StopIteration
            xx = X.strip().split()
            G = []
            for x in xx:
                try:
                    G.append(float(x))
                except Exception:  # noqa: PERF203 - per-value fallback to NaN
                    G.append(np.nan)
            G = np.array(G)
            if self.norm_genotype:
                G = self.normalize_genotype(G)
            return G, "SNP_%d" % self.have_read

        else:
            sys.stderr.write("Do not understand type %s\n" % (self.type))
            # TODO(5b): falls through and returns None (unreachable: type is validated earlier)

    def _get_genos_tped(self, X: list[str]) -> NDArray[np.float64]:
        """Convert one TPED row's allele pairs to 0 / 0.5 / 1 / nan.

        ``0 0`` is missing, ``1 1`` is 0, ``2 2`` is 1, mixed is 0.5.
        """
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

    def _format_binary_genotypes(self, X: bytes, norm: bool = True) -> NDArray[np.float64]:
        """Decode the raw BED bytes of one SNP (``_bytes_to_read`` of them).

        Returns the fraction of A2 alleles (0, 0.5, 1, nan) for the first
        ``N`` individuals, standardised if `norm`.
        """
        B = np.frombuffer(X, dtype=np.uint8)
        codes = (B[:, None] >> self._BED_SHIFTS) & 3
        # only take the leading values because whatever is left should be null
        G = self._BED_CODE_VALUES[codes.reshape(-1)[: self.N]]
        if norm:
            G = self.normalize_genotype(G)
        return G

    def normalize_genotype(self, G: NDArray[np.float64]) -> NDArray[np.float64]:
        """Standardise one SNP: mean-impute missing values, centre, scale to unit variance.

        Parameters
        ----------
        G : numpy.ndarray
            Raw genotypes of one SNP, shape ``(n,)``, nan where missing.  The
            array is modified in place when imputing (missing entries are
            filled with the mean); the standardised result is a new array.

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)``: ``(G - mean) / std`` using the non-missing
            values (population std).  Special cases, which callers skip: an
            all-missing SNP is returned unchanged (all nan), and a SNP that
            is constant up to rounding is returned as zeros.
        """
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

    def get_phenos(self, pheno_file: str | None = None) -> NDArray[np.float64] | None:
        """Read a PLINK-style phenotype file into ``self.phenos``.

        The file has one individual per line, ``FID IID value [value ...]``,
        with an optional ``FID IID ...`` header; ``NA`` and ``-9`` are
        missing.  Rows are matched by ``(FID, IID)`` and returned in the
        order of ``self.indivs``.  Individuals absent from the file get nan
        (with a warning giving the count); the file's extra individuals are
        ignored.  (Bug 10, fixed in Phase 5.2: absent individuals used to be
        skipped, which shifted the rows of everyone after them.)  For EMMA
        input, which has no IDs, rows are taken in order and their number
        must equal the number of individuals.

        Parameters
        ----------
        pheno_file : str, optional
            Path of the phenotype file.  If omitted, ``fbase + ".phenos"`` is
            used (and recorded in ``self.pheno_file``).

        Returns
        -------
        numpy.ndarray or None
            Shape ``(len(indivs), n_phenotypes)``, float64, also stored in
            ``self.phenos``.  None (and ``self.phenos`` unchanged) if the file
            does not exist.

        Raises
        ------
        ValueError
            On a malformed file, duplicate IDs, or when no individual matches.
        """
        if not pheno_file:
            self.pheno_file = pheno_file = self.fbase + ".phenos"
        if not os.path.isfile(pheno_file):
            # sys.stderr.write("Could not find phenotype file: %s\n" % (pheno_file))
            return
        keys, P = self._read_table(pheno_file, lambda x: x == "NA" or x == "-9", "phenotype")
        self.phenos = self._align_rows(keys, P, pheno_file, "phenotype")
        return self.phenos

    def get_phenos_emma(self, emma_file: str) -> NDArray[np.float64]:
        """Read EMMA phenotypes: one phenotype per row, one column per individual.

        Columns are in genotype order (no IDs); anything that is not a number
        is missing (nan).  Blank lines are skipped.

        Parameters
        ----------
        emma_file : str
            Path of the whitespace-separated matrix.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_individuals, n_phenotypes)`` (the file transposed).

        Raises
        ------
        ValueError
            If the file is empty or a row does not have one value per
            individual.
        """
        P = []
        with open(emma_file) as f:
            for line in f:
                v = line.strip().split()
                if not v:
                    continue
                p = []
                for x in v:
                    try:
                        p.append(float(x))
                    except ValueError:  # noqa: PERF203 - per-value fallback to NaN
                        p.append(np.nan)
                P.append(p)
        return self._emma_columns(P, emma_file, "phenotype")

    def _emma_columns(self, rows: list[list[float]], fn: str, what: str) -> NDArray[np.float64]:
        """Check that every row has one value per individual; return the transpose."""
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
    def _read_table(
        fn: str, is_missing: Callable[[str], bool], what: str
    ) -> tuple[list[IndivKey], list[list[float]]]:
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
                    P.append([np.nan if is_missing(x.strip()) else float(x) for x in v[2:]])
                except ValueError:
                    raise ValueError(
                        "%s file %s, line %d: a value is not a number (missing values are "
                        "NA%s)" % (what, fn, lineno, " or -9" if what == "phenotype" else "")
                    ) from None
                keys.append((v[0], v[1]))
        widths = {len(p) for p in P}
        if len(widths) > 1:
            raise ValueError(
                "%s file %s: rows have different numbers of values (%s)"
                % (what, fn, ", ".join(str(w) for w in sorted(widths)))
            )
        return keys, P

    def _align_rows(
        self, keys: list[IndivKey], P: list[list[float]], fn: str, what: str
    ) -> NDArray[np.float64]:
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

    def _read_fam(self, base: str, type: str = "b") -> list[IndivKey]:
        """Read ``(FID, IID)`` of each individual from ``.tfam`` ("t") or ``.fam``; sets ``N``."""
        if type == "t":
            fam_file = "%s.tfam" % base
        else:
            fam_file = "%s.fam" % base

        keys = []
        f = open(fam_file)
        for line in f:
            v = line.strip().split()
            fam_id = v[0]
            indiv_id = v[1]
            k = (fam_id.strip(), indiv_id.strip())
            keys.append(k)
        f.close()

        self.N = len(keys)
        sys.stderr.write("Read %d individuals from %s\n" % (self.N, fam_file))

        return keys

    def read_kinship(self, k_file: str) -> NDArray[np.float64] | None:
        """Read a kinship matrix from a text file.

        Plain format: ``n`` lines of ``n`` whitespace-separated numbers in
        genotype order.  FaSTLMM format (``fastlmm_kinship=True``): the first
        line is a tab-separated header whose first field is ignored and
        whose others are ``FID IID`` labels, and each following line starts
        with a label field that is ignored.  The matrix is then reordered to
        ``self.indivs`` and individuals absent from it are dropped: ``indivs``
        is replaced by the kept individuals and the dropped ones are stored
        in ``indivs_removed``.

        Parameters
        ----------
        k_file : str
            Path of the kinship file.

        Returns
        -------
        numpy.ndarray or None
            The kinship matrix, shape ``(n, n)``; None (with a message on
            stderr) if no individuals were read.
        """
        if self.indivs is None or len(self.indivs) == 0:
            sys.stderr.write("Did not read any individuals so can't load kinship\n")
            return

        sys.stderr.write("Reading kinship matrix from %s\n" % (k_file))

        f = open(k_file)
        # read indivs
        if self.fastlmm_kinship:
            v = f.readline().strip().split("\t")[1:]
            keys = [tuple(y.split()) for y in v]
            D = {}
            for i in range(len(keys)):
                D[keys[i]] = i

        # read matrix
        K = []
        if self.fastlmm_kinship:
            for line in f:
                K.append([float(x) for x in line.strip().split("\t")[1:]])
        else:
            for line in f:
                K.append([float(x) for x in line.strip().split()])
        f.close()
        K = np.array(K)

        if self.fastlmm_kinship:
            # reorder to match self.indivs
            L = []
            kept = []
            X = []
            for i in range(len(self.indivs)):
                if self.indivs[i] not in D:
                    X.append(self.indivs[i])
                else:
                    kept.append(self.indivs[i])
                    L.append(D[self.indivs[i]])
            K = K[L, :][:, L]
            self.indivs = kept
            self.indivs_removed = X
            if len(self.indivs_removed):
                sys.stderr.write(
                    "Removed %d individuals that did not appear in Kinship\n"
                    % (len(self.indivs_removed))
                )

        return K

    def get_covariates_emma(self, emma_file: str) -> NDArray[np.float64]:
        """Read EMMA covariates: one covariate per row, one column per individual.

        Like `get_phenos_emma` but only ``NA`` is a missing value; any other
        non-number raises.

        Parameters
        ----------
        emma_file : str
            Path of the whitespace-separated matrix.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_individuals, n_covariates)`` (the file transposed).

        Raises
        ------
        ValueError
            On a non-numeric value, an empty file, or a row that does not have
            one value per individual.
        """
        P = []
        with open(emma_file) as f:
            for line in f:
                v = line.strip().split()
                if not v:
                    continue
                try:
                    P.append([np.nan if x == "NA" else float(x) for x in v])
                except ValueError:
                    raise ValueError(
                        "EMMA covariate file %s: a value is not a number (missing is NA)"
                        % emma_file
                    ) from None
        return self._emma_columns(P, emma_file, "covariate")

    def get_covariates(self, cov_file: str | None = None) -> NDArray[np.float64]:
        """Read a PLINK-style covariate file, aligned to ``self.indivs``.

        Format ``FID IID c1 c2 ...`` with an optional header; ``NA`` is
        missing (``-9`` is *not*).  Matched to ``self.indivs`` by
        ``(FID, IID)`` like `get_phenos`; absent individuals get nan rows.

        Parameters
        ----------
        cov_file : str
            Path of the covariate file.  (Typed optional for historical
            reasons; None raises TypeError.)

        Returns
        -------
        numpy.ndarray
            Shape ``(len(indivs), n_covariates)``, float64.

        Raises
        ------
        ValueError
            If the file is missing or malformed, has duplicate IDs, or no
            individual matches.
        """
        if not os.path.isfile(cov_file):
            raise ValueError("Could not find covariate file: %s" % (cov_file))
        keys, P = self._read_table(cov_file, lambda x: x == "NA", "covariate")
        return self._align_rows(keys, P, cov_file, "covariate")
