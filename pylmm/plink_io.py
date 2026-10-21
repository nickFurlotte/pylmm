# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

r"""Readers for PLINK (BED, TPED) and EMMA genotype, phenotype and covariate files.

The single entry point is :class:`PlinkReader`.  (This module was ``pylmm.input``
before 1.0.0.)

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
    order.  Alleles must already be recoded as ``1``/``2`` (``0`` is missing,
    and so is a half-missing pair such as ``0 1``, as in PLINK); nucleotide
    letters are not supported.  With PLINK 1.9, make such a file with
    ``plink --bfile X --recode 12 transpose --out Y``.

``type="emma"``: EMMA genotype matrix
    ``fbase`` is the path of a whitespace-separated numeric text file with
    one SNP per row and one column per individual (no header, no IDs).
    Values are used as given (typically 0/1/2 or dosages); anything that does
    not parse as a number is missing.  SNPs are named ``SNP_1``, ``SNP_2``, ...
    and individuals are identified by position only.

Phenotype and covariate files are PLINK-style text tables ``FID IID v1 [v2
...]`` with an optional ``FID IID name1 [name2 ...]`` header line (any case),
whose names can select columns (``read_phenos(columns=["bmi"])``); EMMA ones
are plain numeric matrices (see :meth:`PlinkReader.read_phenos_emma` and
:meth:`PlinkReader.read_covariates_emma`).  A kinship matrix is a plain
whitespace-separated square matrix, or in the FaSTLMM style (tab-separated,
with a header row and a first column of ``FID IID`` labels).  A plain matrix
is in genotype order unless an ID sidecar ``<k_file>.id`` (one ``FID IID``
per line, written by ``pylmm kinship``) names its rows.

Variant metadata and blocks of SNPs
-----------------------------------
`PlinkReader.variants` holds the metadata of every SNP (chromosome, SNP id,
cM, position, A1, A2 from the ``.bim``, or the ``.map`` columns for a TPED
without a ``.bim``; SNP names only for EMMA), as a
:class:`pylmm.genotypes.Variants`.  `PlinkReader.iter_blocks` returns the
genotypes a block of SNPs at a time, ``(G, variants)`` with ``G`` of shape
``(len(indivs), b)``, the same values as the per-SNP iterator; a BED file is
decoded a block at once.  Together with ``sample_ids`` and ``num_samples``
they make `PlinkReader` a :class:`pylmm.genotypes.GenotypeSource`.

Genotype coding
---------------
A genotype is the **fraction of copies of the second allele** (the ``A2``
allele of the ``.bim`` file; for a TPED, the allele coded ``2``): ``0.0``
(homozygous A1), ``0.5`` (heterozygous) or ``1.0`` (homozygous A2).  It is
**not** a 0/1/2 allele count; EMMA input is passed through unchanged.  With ``norm_genotype=True`` (the default) each SNP
is then standardised (see :meth:`PlinkReader.normalize_genotype`): missing
values are mean-imputed, then the SNP is centred and scaled to unit
variance.  Pass ``norm_genotype=False`` to get the raw 0 / 0.5 / 1 / nan
values.

Effect scale and allele
-----------------------
The scale of an association effect (``beta`` of ``LMM.association``) is the
scale of the genotype column it is fitted on.  With ``norm_genotype=True``
(the default) ``beta`` is per SD of the standardised genotype.  With
``norm_genotype=False`` it is per 0 -> 1 step, which is two copies of the A2
allele (fraction coding), so it is 2x the per-A2-copy effect and has the
opposite sign to PLINK's default per-A1-copy ``--linear`` / ``--assoc``
convention (the t statistic changes sign, its size does not).  The
t statistic and p-value do not depend on the scale.

Missing values
--------------
Missing genotypes are ``numpy.nan`` in raw output (BED code ``01``, TPED
``0 0`` or a half-missing pair, an unparsable EMMA value).  After
normalisation there are no nans, except that an all-missing SNP is returned
unchanged (all nan) and a constant SNP is returned as all zeros; callers skip
both.  Missing phenotypes and covariates (``NA``, and ``-9`` for phenotypes)
are ``nan``.

ID alignment
------------
Individuals are identified by the pair ``(FID, IID)`` from the ``.fam`` /
``.tfam`` file, kept in file order in ``PlinkReader.indivs``; the matching
itself is done by :mod:`pylmm.samples`.  ``id_match="iid"`` matches on the
IID alone.  Phenotype and covariate rows are matched to ``indivs`` by ID,
whatever order the file lists them in: extra individuals in the file are
ignored, individuals missing from the file get ``nan`` rows (with a warning
giving the count), and duplicate IDs or a file with no matching individual
raise `InputError` (a ``ValueError``).  EMMA input has no IDs, so rows are matched by position
and the row count must equal the number of individuals.

A kinship matrix with IDs (FaSTLMM labels or a ``.id`` sidecar) is reordered
to ``indivs``, and the individuals absent from it are dropped from the reader
(listed in ``indivs_removed``, with a warning): from ``indivs``, ``phenos``
**and the genotype vectors**, which always have ``len(indivs)`` values in the
order of ``indivs``.  `PlinkReader.subset` drops individuals the same way.

Examples
--------
Write a tiny BED/BIM/FAM set (3 individuals, 2 SNPs) and read it back:

>>> import os, tempfile
>>> from pylmm.plink_io import PlinkReader
>>> tmp = tempfile.TemporaryDirectory()
>>> base = os.path.join(tmp.name, "toy")
>>> with open(base + ".fam", "w") as f:
...     _ = f.write("F1 I1 0 0 1 -9\nF1 I2 0 0 1 -9\nF2 I3 0 0 2 -9\n")
>>> with open(base + ".bim", "w") as f:
...     _ = f.write("1 rs1 0 100 A G\n1 rs2 0 200 C T\n")
>>> with open(base + ".bed", "wb") as f:
...     _ = f.write(bytes([0x6C, 0x1B, 0x01, 0b00111000, 0b00000111]))
>>> with PlinkReader(base, type="b", norm_genotype=False) as reader:
...     print(reader.indivs)
...     for g, name in reader.iter_snps():
...         print(name, g)
[('F1', 'I1'), ('F1', 'I2'), ('F2', 'I3')]
rs1 [0.  0.5 1. ]
rs2 [ 1. nan  0.]

Keep two of the individuals, in another order; the genotype vectors follow:

>>> with PlinkReader(base, type="b", norm_genotype=False) as reader:
...     reader.subset([("F2", "I3"), ("F1", "I1")])
...     print([g.tolist() for g, _ in reader])
[[1.0, 0.0], [0.0, 1.0]]
>>> tmp.cleanup()                     # remove the temporary files

The bundled 132k-SNP example is read the same way; the path is relative to
the repository root, hence the skips:

>>> r = PlinkReader("data/snps.132k.clean.noX", type="b")  # doctest: +SKIP
>>> g, name = next(r.iter_snps())  # doctest: +SKIP
>>> g.shape == (len(r.indivs),)  # doctest: +SKIP
True
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import logging
import os
import sys
import warnings
from collections.abc import Iterator, Sequence
from typing import IO, TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from pylmm import kinship_io, samples
from pylmm.genotypes import (
    DEFAULT_BLOCK_SIZE,
    GenotypeSource,
    Variants,
    check_block_size,
    normalize_genotype,
)
from pylmm.samples import InputError, SampleID

log = logging.getLogger(__name__)

# The genotype formats PlinkReader reads (its ``type`` argument).
GENOTYPE_TYPES = ("b", "t", "emma")

# The first three bytes of a SNP-major PLINK 1 BED file.
_BED_MAGIC = b"\x6c\x1b"
_BED_SNP_MAJOR = b"\x01"


@contextlib.contextmanager
def _number_errors(what: str) -> Iterator[None]:
    """Turn the plain ValueError of a ``float()`` of a non-number into an `InputError`."""
    try:
        yield
    except InputError:
        raise
    except ValueError as e:
        raise InputError("%s: %s" % (what, e)) from e


class PlinkReader:
    """Read genotypes (and optionally phenotypes and a kinship matrix) of one dataset.

    The object is its own iterator: ``for g, name in reader.iter_snps()``
    (or ``for g, name in reader``) yields one SNP at a time as
    ``(genotypes, snp_name)``; `iter_blocks` yields blocks of SNPs with
    their metadata (`variants`), which makes the reader a
    :class:`pylmm.genotypes.GenotypeSource`.  It is also a context manager
    that closes its files (``with PlinkReader(...) as reader:``); `close`
    does the same.  See the module docstring for the file formats, the
    genotype coding, missing values and ID alignment.

    Parameters
    ----------
    fbase : str
        File prefix for ``type="b"`` (``fbase.bed/.bim/.fam``) and ``type="t"``
        (``fbase.tped/.tfam`` plus ``.bim`` or ``.map``); the full path of the
        genotype matrix for ``type="emma"``.
    k_file : str, optional
        Kinship matrix file.  If omitted and ``fbase + ".kin"`` exists, that
        file is used (and read only if ``read_k_file`` is true).  If
        ``k_file + ".id"`` exists, it names the rows (see `read_kinship`).
    pheno_file : str, optional
        PLINK-style phenotype file.  Defaults to ``fbase + ".phenos"``, which
        is silently skipped if it does not exist.
    type : {"b", "t", "emma"}, default "b"
        Genotype format: PLINK binary, PLINK transposed text, or EMMA.
        Anything else raises ValueError.
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
    id_match : {"fid_iid", "iid"}, default "fid_iid"
        Match phenotype, covariate and kinship rows to the genotype file on
        (FID, IID), or on the IID alone (PLINK input only).

    Attributes
    ----------
    indivs : list of (str, str), or list of int for EMMA
        ``(FID, IID)`` of each individual in genotype-column order.  EMMA
        input has no IDs, so it is the integers ``0 .. n-1`` (column
        positions) instead.  Individuals dropped by a kinship with IDs or by
        `subset` are not in it.
    N : int
        Number of individuals in the genotype file (genotype columns),
        whatever was dropped since.  The genotype vectors have
        ``len(indivs)`` values.
    K : numpy.ndarray or None
        Kinship matrix, shape ``(len(indivs), len(indivs))``, or None if none
        was read (also None when an automatically found ``fbase + ".kin"``
        exists but ``read_k_file`` is false).
    phenos : numpy.ndarray or None
        Phenotypes, shape ``(len(indivs), n_phenotypes)``, aligned with
        `indivs`; None if there is no phenotype file.
    pheno_names, covariate_names : list of str or None
        Column names of the last phenotype / covariate file read, from its
        ``FID IID ...`` header; None for a file without one.
    indivs_removed : list
        Individuals of the genotype file that were dropped (see `indivs`).
    sample_ids, num_samples
        `indivs` and its length (the `GenotypeSource` names).
    variants : pylmm.genotypes.Variants
        Metadata of every SNP (read from the ``.bim`` / ``.map`` file on
        first use).
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
    # (FID, IID) pairs for "b"/"t"; column positions 0..n-1 for "emma" (no IDs).
    indivs: list[SampleID] | list[int]
    N: int
    K: NDArray[np.float64] | None
    phenos: NDArray[np.float64] | None
    pheno_names: list[str] | None
    covariate_names: list[str] | None
    k_file: str | None
    pheno_file: str | None
    norm_genotype: bool
    read_k_file: bool
    fastlmm_kinship: bool
    id_match: str
    fhandle: IO[Any] | None
    snp_file_handle: IO[str] | None
    num_snps: int
    have_read: int
    _bytes_to_read: int
    indivs_removed: list[Any]
    # Positions (in the genotype file) of `indivs`; None while that is all of them, in order.
    _geno_index: NDArray[np.intp] | None
    _exhausted: bool
    _variants: Variants | None

    def __init__(
        self,
        fbase: str,
        k_file: str | None = None,
        pheno_file: str | None = None,
        type: str = "b",
        norm_genotype: bool = True,
        read_k_file: bool = False,
        fastlmm_kinship: bool = False,
        id_match: str = "fid_iid",
    ) -> None:

        self.fhandle = None
        self.snp_file_handle = None
        self._exhausted = False
        self._variants = None
        self.num_snps = 0
        self.have_read = 0
        if type not in GENOTYPE_TYPES:
            raise ValueError(
                "unknown genotype file type %r; use one of %s"
                % (type, ", ".join(repr(t) for t in GENOTYPE_TYPES))
            )
        if id_match not in samples.MATCH_MODES:
            raise ValueError(
                "id_match=%r: use one of %s"
                % (id_match, ", ".join(repr(m) for m in samples.MATCH_MODES))
            )
        if type == "emma" and id_match != "fid_iid":
            raise ValueError("id_match='iid' needs PLINK input; EMMA genotypes have no IDs")
        self.fbase = fbase
        self.type = type
        self.id_match = id_match
        self._geno_index = None
        self.indivs_removed = []
        self.K = None
        self.phenos = None
        self.pheno_names = None
        self.covariate_names = None
        if type != "emma":
            self.indivs = self._read_fam(self.fbase, type)
        else:
            # Just read a line from the SNP file and see how many individuals we have
            with open(fbase) as f:
                self.indivs = list(range(len(f.readline().strip().split())))
            self.N = len(self.indivs)
        if type == "b":
            self._check_bed_header()
        self.k_file = k_file
        self.norm_genotype = norm_genotype
        self.pheno_file = pheno_file
        # Originally I was using the fastLMM style that has indiv IDs embedded.
        # NOW I want to use this module to just read SNPs so I'm allowing
        # the programmer to turn off the kinship reading.
        self.read_k_file = read_k_file
        self.fastlmm_kinship = fastlmm_kinship

        if self.k_file:
            self.read_kinship(self.k_file)
        elif os.path.isfile("%s.kin" % fbase):
            self.k_file = "%s.kin" % fbase
            if self.read_k_file:
                self.read_kinship(self.k_file)
        else:
            self.k_file = None

        self.read_phenos(self.pheno_file)

    # -- file handles -------------------------------------------------------

    def close(self) -> None:
        """Close the genotype files opened by `iter_snps` (safe to call twice)."""
        for attr in ("fhandle", "snp_file_handle"):
            fh = getattr(self, attr, None)
            if fh is not None:
                fh.close()
                setattr(self, attr, None)

    def __enter__(self) -> PlinkReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        """Close any open file handles."""
        self.close()

    def _check_bed_header(self) -> None:
        """Raise ValueError unless ``fbase.bed`` starts with the SNP-major BED magic bytes."""
        fn = self.fbase + ".bed"
        with open(fn, "rb") as f:
            head = f.read(3)
        if head[:2] != _BED_MAGIC:
            raise InputError(
                "%s is not a PLINK BED file (its first bytes are %s, expected 6c 1b)"
                % (fn, head[:2].hex() or "empty")
            )
        if head[2:3] != _BED_SNP_MAJOR:
            raise InputError(
                "%s is not in SNP-major order (third byte %s, expected 01); pylmm reads "
                "SNP-major BED files only (rewrite it with PLINK: plink --bfile X --make-bed "
                "--out Y)" % (fn, head[2:3].hex() or "missing")
            )

    # -- SNP iteration ------------------------------------------------------

    def iter_snps(self) -> PlinkReader:
        """Open the genotype files and return the iterator (this reader).

        Resets the iteration, so it can be called again to re-read from the
        first SNP; the files of the previous iteration are closed first.
        Each ``next()`` returns ``(genotypes, snp_name)``: a float64 array of
        length ``len(indivs)``, in the order of `indivs`, and the SNP's name.
        The files are closed when the iteration ends (and by `close`).

        Returns
        -------
        PlinkReader
            ``self``, positioned before the first SNP.

        Raises
        ------
        ValueError
            If `type` is not a known format (the constructor already rejects
            one; this catches a later reassignment), or the BED file is not
            SNP-major (also checked by the constructor).
        """
        self.close()
        self._exhausted = False
        if self.type == "b":
            return self._iter_snps_bed()
        elif self.type == "t":
            return self._iter_snps_tped()
        elif self.type == "emma":
            return self._iter_snps_emma()
        else:
            raise ValueError("unknown genotype file type %r" % (self.type,))

    def _iter_snps_emma(self) -> PlinkReader:
        """Open the EMMA matrix; the number of SNPs is unknown (-1)."""
        self.have_read = 0
        self.num_snps = -1
        self.fhandle = open(self.fbase)

        return self

    @staticmethod
    def _count_lines(fn: str) -> int:
        with open(fn) as f:
            return sum(1 for _ in f)

    def _iter_snps_tped(self) -> PlinkReader:
        """Open ``.tped`` and the ``.bim`` (or ``.map``) used for SNP names."""
        file = self.fbase + ".bim"
        if not os.path.isfile(file):
            file = self.fbase + ".map"
        self.num_snps = self._count_lines(file)
        self.have_read = 0
        self.snp_file_handle = open(file)
        self.fhandle = open(self.fbase + ".tped")

        return self

    def _iter_snps_bed(self) -> PlinkReader:
        """Open ``.bed`` (checking SNP-major order) and ``.bim``."""
        self._check_bed_header()
        file = self.fbase + ".bim"
        self.num_snps = self._count_lines(file)
        self.have_read = 0
        self._bytes_to_read = self.N // 4 + ((self.N % 4 and 1) or 0)
        self.snp_file_handle = open(file)
        self.fhandle = open(self.fbase + ".bed", "rb")
        self.fhandle.read(3)  # magic number and SNP-major flag, checked above

        return self

    def __iter__(self) -> PlinkReader:
        """Start (or restart) iteration over the SNPs; see `iter_snps`."""
        return self.iter_snps()

    def _stop(self) -> StopIteration:
        """End the iteration: close the files and return the StopIteration to raise."""
        self.close()
        self._exhausted = True
        return StopIteration()

    def __next__(self) -> tuple[NDArray[np.float64], str]:
        """Return the next SNP as ``(genotypes, snp_name)``.

        ``genotypes`` is a float64 array of length ``len(indivs)``, in the
        order of `indivs` (individuals dropped by a kinship with IDs or by
        `subset` are left out): the fraction of A2 alleles (0, 0.5, 1; nan
        if missing), standardised over those individuals if
        ``norm_genotype`` is true.  ``snp_name`` is the SNP id from the
        ``.bim``/``.map`` file (``SNP_<k>`` for EMMA).

        Raises
        ------
        StopIteration
            After the last SNP (the files are then closed).
        InputError
            (A ``ValueError``.)  If the BED file is truncated, a TPED line
            does not have 4 + 2n fields or a TPED allele code is not 0, 1
            or 2 (the message names the SNP and gives the PLINK recode
            command), or an EMMA row does not have n values.
        ValueError
            If `iter_snps` was not called first.
        """
        if self._exhausted:
            raise StopIteration
        fh = self.fhandle
        if fh is None:
            raise ValueError("no genotype file is open: call iter_snps() (or iterate) first")
        if self.have_read == self.num_snps:
            raise self._stop()
        self.have_read += 1

        if self.type == "b":
            snp_names = self.snp_file_handle
            assert snp_names is not None
            X = fh.read(self._bytes_to_read)
            if len(X) != self._bytes_to_read:
                raise InputError(
                    "Truncated BED file %s.bed: expected %d bytes for SNP %d, got %d"
                    % (self.fbase, self._bytes_to_read, self.have_read, len(X))
                )
            return self._format_binary_genotypes(
                X, self.norm_genotype
            ), snp_names.readline().strip().split()[1]

        elif self.type == "t":
            snp_names = self.snp_file_handle
            assert snp_names is not None
            X = fh.readline()
            xx = X.strip().split()
            if len(xx) != 4 + 2 * self.N:
                raise InputError(
                    "TPED file %s.tped, line %d: expected 4 + 2 x %d fields (CHR SNP CM BP, "
                    "then two alleles per individual of the .tfam), found %d%s"
                    % (
                        self.fbase,
                        self.have_read,
                        self.N,
                        len(xx),
                        " (the file has fewer lines than the .map/.bim)" if not X else "",
                    )
                )
            chrm, rsid, pos1, pos2 = tuple(xx[:4])
            xx = xx[4:]
            G = self._subset_genotypes(self._get_genos_tped(xx, rsid))
            if self.norm_genotype:
                G = self.normalize_genotype(G)
            return G, snp_names.readline().strip().split()[1]

        elif self.type == "emma":
            X = fh.readline()
            while X and not X.strip():  # skip blank lines (an empty SNP used to crash)
                X = fh.readline()
            if X == "":
                raise self._stop()
            xx = X.strip().split()
            vals: list[float] = []
            for x in xx:
                try:
                    vals.append(float(x))
                except Exception:  # per-value fallback to NaN
                    vals.append(np.nan)
            if len(vals) != self.N:
                raise InputError(
                    "EMMA genotype file %s, line %d: %d values, expected %d (one per "
                    "individual, as on its first line)"
                    % (self.fbase, self.have_read, len(vals), self.N)
                )
            G = self._subset_genotypes(np.array(vals))
            if self.norm_genotype:
                G = self.normalize_genotype(G)
            return G, "SNP_%d" % self.have_read

        else:
            # Unreachable unless `type` was reassigned after iter_snps.
            raise ValueError("unknown genotype file type %r" % (self.type,))

    def _subset_genotypes(self, G: NDArray[np.float64]) -> NDArray[np.float64]:
        """The values of `indivs`, in their order, from a genotype-file-order vector."""
        if self._geno_index is None:
            return G
        out: NDArray[np.float64] = G[self._geno_index]
        return out

    # Allele codes of a 1/2-recoded TPED (PLINK 1.9 --recode 12 transpose):
    # 1 and 2 are the alleles, 0 is PLINK's default missing-genotype code.
    _TPED_ALLELE_CODES = frozenset(("0", "1", "2"))

    def _get_genos_tped(self, X: list[str], snp: str = "?") -> NDArray[np.float64]:
        """Convert one TPED row's allele pairs to 0 / 0.5 / 1 / nan.

        ``1 1`` is 0, ``2 2`` is 1, ``1 2`` / ``2 1`` is 0.5, and a pair with
        a ``0`` (``0 0``, or half-missing ``0 1`` / ``2 0``) is missing, as
        in PLINK.  (Before 1.0.0 a half-missing pair was read as 0.5.)

        Raises
        ------
        InputError
            If an allele code is not ``0``, ``1`` or ``2`` (e.g. a nucleotide
            letter: the alleles must be recoded to 1/2 first, with PLINK
            1.9 ``plink --bfile X --recode 12 transpose --out Y``), naming the SNP (`snp`) and the code.  Before this
            check such a code silently reused the previous individual's
            genotype or raised UnboundLocalError.
        """
        bad = set(X) - self._TPED_ALLELE_CODES
        if bad:
            raise InputError(
                "TPED file %s.tped, SNP %s: allele code %r is not 0, 1 or 2; pylmm reads "
                "1/2-recoded TPED files only (0 = missing; recode with PLINK 1.9: "
                "plink --bfile X --recode 12 transpose --out Y)" % (self.fbase, snp, sorted(bad)[0])
            )
        G: list[float] = []
        for i in range(0, len(X) - 1, 2):
            a = X[i]
            b = X[i + 1]
            if a == "0" or b == "0":
                g = np.nan
            elif a != b:
                g = 0.5
            elif a == "1":
                g = 0.0
            else:
                g = 1.0
            G.append(g)
        return np.array(G)

    # 2-bit PLINK code -> genotype value.  The four codes of each byte are
    # read from the low bits up (individual 4k+j is bits 2j..2j+1).  Same
    # mapping as the original string-based decoder: low bit 0/high bit 0 ->
    # 0.0, 01 (missing) -> nan, 10 -> 0.5, 11 -> 1.0, where the pair is
    # written high bit first.
    _BED_CODE_VALUES = np.array([0.0, np.nan, 0.5, 1.0])
    _BED_SHIFTS = np.array([0, 2, 4, 6], dtype=np.uint8)
    # The four genotype values of each of the 256 bytes (row = byte), built from the
    # two tables above, for decoding a block of SNPs at once (`_decode_bed_block`).
    _BED_BYTE_VALUES = _BED_CODE_VALUES[
        (np.arange(256, dtype=np.uint8)[:, None] >> _BED_SHIFTS) & 3
    ]

    def _format_binary_genotypes(self, X: bytes, norm: bool = True) -> NDArray[np.float64]:
        """Decode the raw BED bytes of one SNP (``_bytes_to_read`` of them).

        Returns the fraction of A2 alleles (0, 0.5, 1, nan) of `indivs`,
        standardised if `norm`.
        """
        B = np.frombuffer(X, dtype=np.uint8)
        codes = (B[:, None] >> self._BED_SHIFTS) & 3
        # only take the leading values because whatever is left should be null
        G: NDArray[np.float64] = self._BED_CODE_VALUES[codes.reshape(-1)[: self.N]]
        G = self._subset_genotypes(G)
        if norm:
            G = self.normalize_genotype(G)
        return G

    def normalize_genotype(self, G: NDArray[np.float64]) -> NDArray[np.float64]:
        """Standardise one SNP: mean-impute missing values, centre, scale to unit variance.

        The same as :func:`pylmm.genotypes.normalize_genotype`, which does the
        work (it does not depend on the reader).

        Parameters
        ----------
        G : numpy.ndarray
            Raw genotypes of one SNP, shape ``(n,)``, nan where missing.  It
            is not modified: the mean imputation is done on a copy (it used
            to fill the caller's array in place; fixed in 1.0.0).

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)``: ``(G - mean) / std`` using the non-missing
            values (population std).  Special cases, which callers skip: an
            all-missing SNP is returned unchanged (all nan), and a SNP that
            is constant up to rounding is returned as zeros.
        """
        return normalize_genotype(G)

    # -- variant metadata and SNP blocks (the GenotypeSource interface) -------

    @property
    def sample_ids(self) -> list[SampleID] | list[int]:
        """The individuals of the genotype vectors and blocks: `indivs`."""
        return self.indivs

    @property
    def num_samples(self) -> int:
        """``len(indivs)``: the length of a genotype vector, the rows of a block."""
        return len(self.indivs)

    @property
    def variants(self) -> Variants:
        """Metadata of every SNP of the file, in iteration order (read on first use).

        For a BED file, the six columns of the ``.bim`` file (``chrom``,
        ``snp``, ``cm``, ``bp``, ``a1``, ``a2``).  For a TPED file, those of
        the ``.bim`` file if it exists, otherwise the ``.map`` file's
        (``chrom``, ``snp``, ``cm``, ``bp``; no alleles, and no ``cm`` for a
        three-column ``.map``).  For EMMA input, the names ``SNP_1``,
        ``SNP_2``, ... of its non-blank rows and no other column (this
        property reads the whole EMMA file once to count them; `iter_blocks`
        does not need it).  The ``snp`` column holds the names `iter_snps`
        returns.  ``len(variants)`` is the number of SNPs (`num_snps` is set
        only by an iteration, and is -1 for EMMA).

        The file is read leniently, so that every file `iter_snps` reads
        has metadata: extra fields after the sixth (``.bim``) or fourth
        (``.map``) are ignored; a ``cM`` that is not a number (``NA``) is
        nan; a position written as an integral float (``100.0``,
        ``1.5e2``) is that integer, and any other non-integer position is
        -1, with one logged warning that names the first such line and the
        count.

        Raises
        ------
        InputError
            (A ``ValueError``.)  If a ``.bim`` line has fewer than 6 fields
            (or a blank line), or a ``.map`` line fewer than 3 or not the same
            number as the first line (3, or 4 or more); the message names
            the file and the line.

        Examples
        --------
        >>> import os, tempfile
        >>> from pylmm.plink_io import PlinkReader
        >>> tmp = tempfile.TemporaryDirectory()
        >>> base = os.path.join(tmp.name, "toy")
        >>> with open(base + ".fam", "w") as f:
        ...     _ = f.write("F1 I1 0 0 1 -9\\nF1 I2 0 0 1 -9\\nF2 I3 0 0 2 -9\\n")
        >>> with open(base + ".bim", "w") as f:
        ...     _ = f.write("1 rs1 0 100 A G\\n2 rs2 0.5 200 C T\\n")
        >>> with open(base + ".bed", "wb") as f:
        ...     _ = f.write(bytes([0x6C, 0x1B, 0x01, 0b00111000, 0b00000111]))
        >>> r = PlinkReader(base)
        >>> r.variants[1]
        Variant(chrom='2', snp='rs2', cm=0.5, bp=200, a1='C', a2='T')
        >>> r.variants.chrom.tolist(), r.variants.bp.tolist()
        (['1', '2'], [100, 200])
        >>> tmp.cleanup()
        """
        if self._variants is None:
            if self.type == "emma":
                with open(self.fbase) as f:
                    m = sum(1 for line in f if line.strip())
                self._variants = Variants.from_names(["SNP_%d" % (k + 1) for k in range(m)])
            elif self.type == "t" and not os.path.isfile(self.fbase + ".bim"):
                self._variants = _read_variant_file(self.fbase + ".map")
            else:
                self._variants = _read_variant_file(self.fbase + ".bim")
        return self._variants

    def iter_blocks(
        self, block_size: int = DEFAULT_BLOCK_SIZE
    ) -> Iterator[tuple[NDArray[np.float64], Variants]]:
        """Iterate over the SNPs in blocks: ``(G, variants)`` per block.

        ``G`` is float64, shape ``(len(indivs), b)``, ``b <= block_size``
        (only the last block is shorter): column ``j`` is exactly (bit for
        bit) the genotype vector `iter_snps` returns for that SNP, in the
        order of `indivs` and standardised if ``norm_genotype``.  ``variants``
        is the `Variants` of the block's SNPs (a slice of `variants`).

        A BED file is decoded a block at a time (one vectorised pass over the
        block's bytes), which is several times faster than `iter_snps` when
        the genotypes are not standardised; TPED and EMMA blocks are
        stacked from the per-SNP reader.  The iteration is independent of
        `iter_snps` (it opens its own file handle, closed when the iteration
        ends or the iterator is discarded).  A block holds
        ``8 * len(indivs) * block_size`` bytes, and the decoding about as
        much again.

        The individuals, ``norm_genotype`` and the SNP metadata are those of
        the time of the call (`variants` is read then, if it was not yet,
        except for EMMA input, whose block variants are built from the rows
        as they are read).  A later `subset`, kinship read or change of
        ``norm_genotype`` does not affect an iterator already created: its
        blocks keep the earlier rows, while `indivs` and `num_samples` then
        describe the new state, so call `iter_blocks` again after a
        `subset`.

        Parameters
        ----------
        block_size : int, default 1024
            The number of SNPs per block.

        Yields
        ------
        G : numpy.ndarray
            ``(len(indivs), b)`` float64 genotypes.
        variants : pylmm.genotypes.Variants
            The metadata of the ``b`` SNPs.

        Raises
        ------
        ValueError
            If `block_size` is not a positive integer.
        InputError
            (A ``ValueError``.)  As `variants` and `__next__`, e.g. for a
            truncated BED file.

        Examples
        --------
        >>> import os, tempfile
        >>> import numpy as np
        >>> from pylmm.plink_io import PlinkReader
        >>> tmp = tempfile.TemporaryDirectory()
        >>> base = os.path.join(tmp.name, "toy")
        >>> with open(base + ".fam", "w") as f:
        ...     _ = f.write("F1 I1 0 0 1 -9\\nF1 I2 0 0 1 -9\\nF2 I3 0 0 2 -9\\n")
        >>> with open(base + ".bim", "w") as f:
        ...     _ = f.write("1 rs1 0 100 A G\\n1 rs2 0 200 C T\\n2 rs3 0 50 A C\\n")
        >>> with open(base + ".bed", "wb") as f:
        ...     _ = f.write(bytes([0x6C, 0x1B, 0x01, 0b00111000, 0b00000111, 0b00101100]))
        >>> r = PlinkReader(base, norm_genotype=False)
        >>> for G, v in r.iter_blocks(2):
        ...     print(v.snp.tolist(), G.tolist())
        ['rs1', 'rs2'] [[0.0, 1.0], [0.5, nan], [1.0, 0.0]]
        ['rs3'] [[0.0], [1.0], [0.5]]
        >>> W = np.column_stack([g for g, _ in r.iter_snps()])
        >>> blocks = np.hstack([G for G, _ in r.iter_blocks(2)])
        >>> np.array_equal(W, blocks, equal_nan=True)
        True
        >>> tmp.cleanup()
        """
        block_size = check_block_size(block_size)
        # Everything the iteration depends on is captured now, at the call (the
        # generators below would otherwise read it at the first next()).
        if self.type == "b":
            self._check_bed_header()
            index = None if self._geno_index is None else self._geno_index.copy()
            return self._iter_blocks_bed(
                block_size, self.variants, self.N, index, self.norm_genotype
            )
        reader = copy.copy(self)  # same individuals and options, its own file handles
        reader.fhandle = reader.snp_file_handle = None
        # EMMA names are SNP_<k>: built from the rows read, without a counting pass.
        variants = None if self.type == "emma" else self.variants
        return self._iter_blocks_stacked(block_size, reader, variants)

    def _iter_blocks_bed(
        self,
        block_size: int,
        variants: Variants,
        n_file: int,
        index: NDArray[np.intp] | None,
        norm: bool,
    ) -> Iterator[tuple[NDArray[np.float64], Variants]]:
        """`iter_blocks` for a BED file: decode `block_size` SNPs at a time.

        `n_file` individuals in the file, of which `index` are kept (None: all,
        in order); `norm`: standardise.  Captured by `iter_blocks` at the call.
        """
        bytes_per_snp = (n_file + 3) // 4
        m = len(variants)
        with open(self.fbase + ".bed", "rb") as fh:
            fh.read(3)  # magic number and SNP-major flag, checked by iter_blocks
            for start in range(0, m, block_size):
                b = min(block_size, m - start)
                buf = fh.read(b * bytes_per_snp)
                if len(buf) != b * bytes_per_snp:
                    raise InputError(
                        "Truncated BED file %s.bed: expected %d bytes for SNP %d, got %d"
                        % (
                            self.fbase,
                            bytes_per_snp,
                            start + len(buf) // bytes_per_snp + 1,
                            len(buf) % bytes_per_snp,
                        )
                    )
                R = self._decode_bed_block(buf, b, n_file)  # (b, N), one SNP per row
                if index is not None:
                    R = R[:, index]
                if norm:
                    for j in range(b):
                        R[j] = normalize_genotype(R[j])
                yield R.T, variants[start : start + b]
                del R  # before the next block is decoded

    @classmethod
    def _decode_bed_block(cls, buf: bytes, b: int, n: int) -> NDArray[np.float64]:
        """Decode the BED bytes of `b` consecutive SNPs of `n` individuals: shape ``(b, n)``.

        The values of `_format_binary_genotypes`, for a block of SNPs at once:
        each byte is looked up in `_BED_BYTE_VALUES` (its four genotypes).
        """
        B = np.frombuffer(buf, dtype=np.uint8).reshape(b, -1)
        R: NDArray[np.float64] = np.take(cls._BED_BYTE_VALUES, B, axis=0).reshape(b, -1)[:, :n]
        return R

    @staticmethod
    def _iter_blocks_stacked(
        block_size: int, reader: PlinkReader, variants: Variants | None
    ) -> Iterator[tuple[NDArray[np.float64], Variants]]:
        """`iter_blocks` for TPED and EMMA input: stack the vectors of `reader`'s per-SNP path.

        `reader` is a shallow copy made by `iter_blocks`; `variants` is None for
        EMMA, whose block variants are made from the SNP names read.
        """
        reader.iter_snps()
        # Call __next__ directly: iter(reader) (as in islice) would restart the iteration.
        try:
            start = 0
            done = False
            while not done:
                cols: list[NDArray[np.float64]] = []
                names: list[str] = []
                while len(cols) < block_size:
                    try:
                        g, name = next(reader)
                    except StopIteration:
                        done = True
                        break
                    cols.append(g)
                    names.append(name)
                if cols:
                    v = (
                        Variants.from_names(names)
                        if variants is None
                        else variants[start : start + len(cols)]
                    )
                    yield np.column_stack(cols), v
                    start += len(cols)
        finally:
            reader.close()

    # -- individuals --------------------------------------------------------

    def subset(self, ids: Sequence[Any]) -> None:
        """Keep only these individuals, in this order.

        `indivs`, `phenos`, `K` and the genotype vectors of later SNPs are
        all restricted and reordered together; the others are added to
        `indivs_removed`.  Genotypes are standardised over the kept
        individuals.

        Parameters
        ----------
        ids : sequence
            IDs from `indivs` (``(FID, IID)`` pairs, or positions for EMMA),
            matched as ``id_match`` says.

        Raises
        ------
        ValueError
            If an ID is not in `indivs`, is listed twice, or `ids` is empty.
        """
        ids = list(ids)
        if not ids:
            raise ValueError("subset needs at least one individual")
        idx = samples.match_index(
            ids,
            self.indivs,
            match=self.id_match,
            target_name="the requested individuals",
            source_name="the genotype file",
        )
        absent = [s for s, i in zip(ids, idx) if i < 0]
        if absent:
            raise ValueError(
                "%d of the requested individuals are not in the genotype file %s: %s"
                % (len(absent), self.fbase, ", ".join(samples.format_id(s) for s in absent[:5]))
            )
        self._restrict(idx)

    def _restrict(self, pos: NDArray[np.intp]) -> None:
        """Keep the individuals at positions `pos` of `indivs` (in that order)."""
        kept = set(pos.tolist())
        self.indivs_removed = self.indivs_removed + [
            s for i, s in enumerate(self.indivs) if i not in kept
        ]
        base = np.arange(self.N) if self._geno_index is None else self._geno_index
        index = base[pos]
        self._geno_index = (
            None if len(index) == self.N and np.array_equal(index, np.arange(self.N)) else index
        )
        self.indivs = cast("list[Any]", [self.indivs[i] for i in pos])
        if self.phenos is not None:
            self.phenos = self.phenos[pos]
        if self.K is not None:
            self.K = self.K[np.ix_(pos, pos)]

    # -- phenotypes and covariates -------------------------------------------

    def read_phenos(
        self, pheno_file: str | None = None, columns: Sequence[str | int] | None = None
    ) -> NDArray[np.float64] | None:
        """Read a PLINK-style phenotype file into ``self.phenos``.

        The file has one individual per line, ``FID IID value [value ...]``,
        with an optional ``FID IID name [name ...]`` header; ``NA`` and
        ``-9`` are missing.  Rows are matched by ID (see ``id_match``) and
        returned in the order of ``self.indivs``.  Individuals absent from
        the file get nan (with a warning giving the count); the file's extra
        individuals are ignored.  For EMMA input, which has no IDs, rows are
        taken in order and their number must equal the number of
        individuals.

        Parameters
        ----------
        pheno_file : str, optional
            Path of the phenotype file.  If omitted, ``fbase + ".phenos"`` is
            used (and recorded in ``self.pheno_file``).
        columns : sequence of str or int, optional
            The phenotypes to read: names from the header, or 0-based indices
            of the value columns.  Default: all.

        Returns
        -------
        numpy.ndarray or None
            Shape ``(len(indivs), n_phenotypes)``, float64, also stored in
            ``self.phenos`` (and the names in ``self.pheno_names``).  None
            (and ``self.phenos`` unchanged) if the file does not exist.

        Raises
        ------
        ValueError
            On a malformed file, duplicate IDs, no matching individual, or
            an unknown column.
        """
        if not pheno_file:
            self.pheno_file = pheno_file = self.fbase + ".phenos"
        if not os.path.isfile(pheno_file):
            return None
        table = samples.read_sample_table(
            pheno_file,
            what="phenotype",
            is_missing=lambda x: x == "NA" or x == "-9",
            columns=columns,
        )
        self.phenos = self._align_rows(table.ids, table.values, pheno_file, "phenotype")
        self.pheno_names = table.names
        return self.phenos

    def read_covariates(
        self, cov_file: str, columns: Sequence[str | int] | None = None
    ) -> NDArray[np.float64]:
        """Read a PLINK-style covariate file, aligned to ``self.indivs``.

        Format ``FID IID c1 c2 ...`` with an optional ``FID IID name ...``
        header; ``NA`` is missing (``-9`` is *not*).  Matched to
        ``self.indivs`` by ID like `read_phenos`; absent individuals get nan
        rows.

        Parameters
        ----------
        cov_file : str
            Path of the covariate file (required; there is no default file,
            unlike `read_phenos`).
        columns : sequence of str or int, optional
            The covariates to read: names from the header, or 0-based
            indices of the value columns.  Default: all.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(indivs), n_covariates)``, float64: the covariate
            columns of the file only, WITHOUT an intercept column.  Passing
            this alone as ``X0`` to `pylmm.lmm.gwas` or `pylmm.lmm.LMM`
            silently fits a model with no intercept (test statistics can
            change a lot).  Add one: ``X0 = np.column_stack([np.ones(n),
            cov])``.  The names are stored in ``self.covariate_names``.

        Raises
        ------
        ValueError
            If `cov_file` is None, or the file is missing or malformed, has
            duplicate IDs, no individual matches, or a column is unknown.
        """
        if cov_file is None:
            raise ValueError("read_covariates needs the path of a covariate file; cov_file is None")
        if not os.path.isfile(cov_file):
            raise InputError("Could not find covariate file: %s" % (cov_file))
        table = samples.read_sample_table(
            cov_file, what="covariate", is_missing=lambda x: x == "NA", columns=columns
        )
        self.covariate_names = table.names
        return self._align_rows(table.ids, table.values, cov_file, "covariate")

    def read_phenos_emma(self, emma_file: str) -> NDArray[np.float64]:
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
                    except ValueError:  # per-value fallback to NaN
                        p.append(np.nan)
                P.append(p)
        return self._emma_columns(P, emma_file, "phenotype")

    def read_covariates_emma(self, emma_file: str) -> NDArray[np.float64]:
        """Read EMMA covariates: one covariate per row, one column per individual.

        Like `read_phenos_emma` but only ``NA`` is a missing value; any other
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
                    raise InputError(
                        "EMMA covariate file %s: a value is not a number (missing is NA)"
                        % emma_file
                    ) from None
        return self._emma_columns(P, emma_file, "covariate")

    def _emma_columns(self, rows: list[list[float]], fn: str, what: str) -> NDArray[np.float64]:
        """Check that every row has one value per individual; return the transpose."""
        n = len(self.indivs)
        bad = [i for i, r in enumerate(rows) if len(r) != n]
        if bad or not rows:
            raise InputError(
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

    def _align_rows(
        self, keys: list[SampleID], P: NDArray[np.float64], fn: str, what: str
    ) -> NDArray[np.float64]:
        """Rows of P (keyed by ID) in self.indivs order; nan rows for absent
        individuals.  Raises ValueError for duplicate IDs or when no
        individual matches."""
        if self.type == "emma":
            if len(P) != len(self.indivs):
                raise InputError(
                    "%s file %s has %d rows but the EMMA genotypes have %d individuals; "
                    "with EMMA input the rows are matched by position"
                    % (what, fn, len(P), len(self.indivs))
                )
            return np.array(P, dtype=np.float64)
        indivs = cast("list[SampleID]", self.indivs)
        idx = samples.match_index(
            indivs,
            keys,
            match=self.id_match,
            target_name="the genotype file %s" % self.fbase,
            source_name="%s file %s" % (what, fn),
        )
        found = idx >= 0
        if not found.any():
            raise InputError(
                "none of the %d individuals of the genotype file appear in the %s file %s "
                "(matched on %s; e.g. %s %s)"
                % (
                    len(indivs),
                    what,
                    fn,
                    "the second column, IID"
                    if self.id_match == "iid"
                    else "the first two columns, FID and IID",
                    indivs[0][0],
                    indivs[0][1],
                )
            )
        out = np.full((len(indivs), P.shape[1]), np.nan)
        out[found] = P[idx[found]]
        absent = [k for k, f in zip(indivs, found) if not f]
        if absent:
            warnings.warn(
                "%d of %d individuals are not in the %s file %s and are treated as missing "
                "(e.g. %s)"
                % (
                    len(absent),
                    len(indivs),
                    what,
                    fn,
                    ", ".join("%s %s" % k for k in absent[:5]),
                ),
                stacklevel=3,
            )
        return out

    def _read_fam(self, base: str, type: str = "b") -> list[SampleID]:
        """Read ``(FID, IID)`` of each individual from ``.tfam`` ("t") or ``.fam``; sets ``N``."""
        if type == "t":
            fam_file = "%s.tfam" % base
        else:
            fam_file = "%s.fam" % base

        keys = []
        with open(fam_file) as f:
            for lineno, line in enumerate(f, 1):
                v = line.strip().split()
                if len(v) < 2:
                    raise InputError(
                        "%s, line %d: expected FID IID ... (found %d fields)"
                        % (fam_file, lineno, len(v))
                    )
                keys.append((v[0].strip(), v[1].strip()))

        self.N = len(keys)
        log.info("Read %d individuals from %s", self.N, fam_file)

        return keys

    # -- kinship ------------------------------------------------------------

    def read_kinship(self, k_file: str) -> NDArray[np.float64] | None:
        """Read a kinship matrix from a text file into ``self.K``.

        Plain format: ``n`` lines of ``n`` whitespace-separated numbers.  The
        rows are named by the ID sidecar ``k_file + ".id"`` (one ``FID IID``
        per line, as ``pylmm kinship`` writes it) if it exists; otherwise they
        must be the individuals of `indivs`, in that order.  For EMMA
        genotypes, which have no IDs, a sidecar is ignored with a warning
        (as in ``pylmm gwas``) and the rows are in genotype order.  FaSTLMM format
        (``fastlmm_kinship=True``): the first line is a tab-separated header
        whose first field is ignored and whose others are ``FID IID`` labels,
        and each following line starts with a label field that is ignored.
        A GCTA binary GRM (``k_file`` ending in ``.grm.bin``) or a plink2
        ``.rel`` / ``.rel.gz`` / ``.rel.bin`` file is read with its own IDs
        (`pylmm.kinship_io.read_grm`, `pylmm.kinship_io.read_rel`).

        With IDs (a sidecar or FaSTLMM labels) the matrix is reordered to
        `indivs`, and individuals absent from it are dropped from the reader
        as by `subset`: from `indivs`, `phenos` and the genotype vectors, and
        listed in `indivs_removed`, with a warning.  Its extra individuals
        are ignored, with a warning.

        Parameters
        ----------
        k_file : str
            Path of the kinship file.

        Returns
        -------
        numpy.ndarray or None
            The kinship matrix, shape ``(len(indivs), len(indivs))``, also
            stored in ``self.K``; None (with a logged warning) if no
            individuals were read.

        Raises
        ------
        ValueError
            If a plain matrix without IDs is not ``len(indivs)`` square, the
            IDs do not match the size of the matrix, an ID is listed twice,
            no individual is in both, or FaSTLMM labels are given for EMMA
            genotypes (which have none).
        """
        if self.indivs is None or len(self.indivs) == 0:
            log.warning("Did not read any individuals so can't load kinship")
            return None

        log.info("Reading kinship matrix from %s", k_file)

        kin_ids: list[SampleID] | None = None
        rows: list[list[float]] = []
        if kinship_io.kinship_format(k_file) != "text":
            K, kin_ids = kinship_io.read_kinship_with_ids(k_file)
            if self.type == "emma":  # as for an ID sidecar: K is in genotype order
                warnings.warn(
                    "the sample IDs of %s are ignored: EMMA genotypes have no IDs, so K is in "
                    "genotype order" % k_file,
                    stacklevel=2,
                )
                kin_ids = None
            return self._set_kinship(K, kin_ids, k_file)
        with open(k_file) as f, _number_errors("kinship file %s" % k_file):
            if self.fastlmm_kinship:
                header = f.readline().strip().split("\t")[1:]
                kin_ids = []
                for label in header:
                    v = label.split()
                    if len(v) != 2:
                        raise InputError(
                            "FaSTLMM kinship %s: header label %r is not 'FID IID'" % (k_file, label)
                        )
                    kin_ids.append((v[0], v[1]))
                for line in f:
                    rows.append([float(x) for x in line.strip().split("\t")[1:]])
            else:
                for line in f:
                    rows.append([float(x) for x in line.strip().split()])
        K = np.array(rows)
        if not self.fastlmm_kinship:
            if self.type == "emma" and os.path.isfile(samples.kinship_id_path(k_file)):
                # As in `pylmm gwas`: EMMA genotypes have no IDs, so K is in genotype order.
                warnings.warn(
                    "%s is ignored: EMMA genotypes have no IDs, so K is in genotype order"
                    % samples.kinship_id_path(k_file),
                    stacklevel=2,
                )
            else:
                kin_ids = samples.read_kinship_ids(k_file)
        return self._set_kinship(K, kin_ids, k_file)

    def _set_kinship(
        self, K: NDArray[np.float64], kin_ids: list[SampleID] | None, k_file: str
    ) -> NDArray[np.float64]:
        """Store K, aligned to `indivs` by ``kin_ids`` (or in genotype order without IDs)."""
        assert self.indivs is not None
        if kin_ids is None:
            n = len(self.indivs)
            if K.shape != (n, n):
                raise InputError(
                    "kinship file %s is a %s matrix but there are %d individuals; without an ID "
                    "file %s its rows must be the individuals of the genotype file, in order"
                    % (
                        k_file,
                        "x".join(str(d) for d in K.shape),
                        n,
                        samples.kinship_id_path(k_file),
                    )
                )
            self.K = K
            return K

        if self.type == "emma":
            raise InputError(
                "kinship %s has sample IDs but EMMA genotypes have none, so they cannot be "
                "matched" % k_file
            )
        if K.ndim != 2 or K.shape != (len(kin_ids), len(kin_ids)):
            raise InputError(
                "kinship file %s is a %s matrix but names %d individuals"
                % (k_file, "x".join(str(d) for d in K.shape), len(kin_ids))
            )
        al = samples.align_samples(
            {"genotype file": self.indivs, "kinship": kin_ids}, match=self.id_match
        )
        for line in al.report():
            warnings.warn(line, stacklevel=3)
        self.K = None  # replaced below; do not subset the previous matrix
        self._restrict(al.rows["genotype file"])
        self.K = al.take_square("kinship", K)
        return self.K


# Lines of a .bim / .map file parsed at a time by `_read_variant_file` (bounds the
# memory of the Python lists of a file with millions of SNPs).
_VARIANT_CHUNK = 65536


def _text(values: Sequence[str]) -> NDArray[np.object_]:
    """A 1-d object array of the strings `values` (see `pylmm.genotypes.Variants`)."""
    a = np.empty(len(values), dtype=object)
    a[:] = values
    return a


def _parse_cm(values: Sequence[str]) -> NDArray[np.float64]:
    """Genetic positions; a value that is not a number (e.g. ``NA``) is nan."""
    try:
        return np.array([float(x) for x in values], dtype=np.float64)
    except ValueError:
        pass
    out = np.empty(len(values), dtype=np.float64)
    for i, x in enumerate(values):
        try:
            out[i] = float(x)
        except ValueError:
            out[i] = np.nan
    return out


_INT64_MAX = np.iinfo(np.int64).max


def _parse_bp(values: Sequence[str]) -> tuple[NDArray[np.int64], list[int]]:
    """Base-pair positions, and the indices of the values that are not integers (set to -1).

    An integral float (``100.0``, ``1.5e2``) is that integer.
    """
    try:
        return np.array([int(x) for x in values], dtype=np.int64), []
    except (ValueError, OverflowError):
        pass
    out = np.empty(len(values), dtype=np.int64)
    bad = []
    for i, x in enumerate(values):
        v: int | None
        try:
            v = int(x)
        except ValueError:
            try:
                f = float(x)
            except ValueError:
                f = np.nan
            v = int(f) if np.isfinite(f) and f == np.floor(f) else None
        if v is None or abs(v) > _INT64_MAX:
            bad.append(i)
            out[i] = -1
        else:
            out[i] = v
    return out, bad


def _variant_fields(is_bim: bool, width: int | None) -> tuple[str, str]:
    """The field count and names a ``.bim`` / ``.map`` line needs, for error messages."""
    if is_bim:
        return "6 or more", "chromosome, SNP id, cM, position, A1, A2"
    if width == 4:
        return "4 or more", "chromosome, SNP id, cM, position"
    if width == 3:
        return "3", "chromosome, SNP id, position"
    return "3, or 4 or more", "chromosome, SNP id, [cM,] position"


def _read_variant_file(fn: str) -> Variants:
    """Read a ``.bim`` or ``.map`` file into `Variants`, leniently (see `PlinkReader.variants`).

    ``.bim``: ``chrom snp cm bp a1 a2`` (6 or more fields; the first 6 are
    used).  ``.map``: ``chrom snp cm bp`` (4 or more fields), or ``chrom snp
    bp`` (exactly 3, no ``cm``), as on the first line, for every line; no
    alleles.  The lines are converted chunk by chunk, so only one chunk is held
    as Python lists at a time.

    Raises
    ------
    InputError
        If a line has too few fields, naming the file and line.
    """
    is_bim = fn.endswith(".bim")
    width: int | None = 6 if is_bim else None
    chunks: list[list[NDArray[Any]]] = []
    bad_bp: list[tuple[int, str]] = []  # (line, value) of the first non-integer position
    n_bad_bp = 0
    lineno = 0
    with open(fn) as f:
        while True:
            lines = list(itertools.islice(f, _VARIANT_CHUNK))
            if not lines:
                break
            first = lineno + 1
            rows = []
            for line in lines:
                lineno += 1
                r = line.split()
                if width is None:  # a .map: its first line says 3 or 4 columns
                    width = 4 if len(r) >= 4 else 3 if len(r) == 3 else None
                if width is None or len(r) < width or (width == 3 and len(r) != 3):
                    need, fields = _variant_fields(is_bim, width)
                    raise InputError(
                        "%s, line %d: expected %s fields (%s), found %d"
                        % (fn, lineno, need, fields, len(r))
                    )
                rows.append(r)
            cols = list(zip(*(r[:width] for r in rows)))
            bp_col = cols[2] if width == 3 else cols[3]
            bp, bad = _parse_bp(bp_col)
            if bad:
                n_bad_bp += len(bad)
                if not bad_bp:
                    bad_bp.append((first + bad[0], bp_col[bad[0]]))
            # Chromosome codes repeat: one str object per distinct code.
            chunk = [_text([sys.intern(x) for x in cols[0]]), _text(cols[1]), bp]
            if width != 3:
                chunk.append(_parse_cm(cols[2]))
            if width == 6:
                chunk += [_text(cols[4]), _text(cols[5])]
            chunks.append(chunk)
    if n_bad_bp:
        log.warning(
            "%s: %d position(s) are not integers and are -1 in the variant metadata (first: line "
            "%d, %r)",
            fn,
            n_bad_bp,
            bad_bp[0][0],
            bad_bp[0][1],
        )
    if not chunks:
        return Variants.from_names([])
    c = [np.concatenate(col) for col in zip(*chunks)]
    if width == 3:
        return Variants(chrom=c[0], snp=c[1], bp=c[2])
    if width == 4:
        return Variants(chrom=c[0], snp=c[1], cm=c[3], bp=c[2])
    return Variants(chrom=c[0], snp=c[1], cm=c[3], bp=c[2], a1=c[4], a2=c[5])


if TYPE_CHECKING:
    # PlinkReader implements the GenotypeSource protocol (checked by mypy).
    def _is_genotype_source(r: PlinkReader) -> GenotypeSource:
        return r
