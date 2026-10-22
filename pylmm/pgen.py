# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""PLINK 2 PGEN genotypes (``.pgen`` / ``.pvar`` or ``.pvar.zst`` / ``.psam``).

Reading the ``.pgen`` file needs the optional package ``pgenlib``, plink2's own
reader (LGPL-3.0-only), installed with ``pip install pgenlib``
(from a clone of pylmm: ``uv sync --extra pgen``).  It is
imported only when genotypes are read, so the rest of pylmm, and the ``.psam``
and plain-text ``.pvar`` parsers of this module, work without it.

Two ways in:

:class:`PgenSource`
    A :class:`pylmm.genotypes.GenotypeSource` over a PGEN file set, for
    library code (``PgenSource("geno").iter_blocks()``).
``PlinkReader(prefix, type="pgen")``
    The same genotypes through :class:`pylmm.plink_io.PlinkReader`, with its
    phenotype, covariate and kinship reading and ID alignment; this is what
    ``pylmm gwas --pfile`` and ``pylmm kinship --pfile`` use.

Genotype coding and alleles
---------------------------
pylmm's genotype is the **fraction of copies of A2** (0, 0.5, 1; nan when
missing), and plink2 writes REF as A2 (and ALT as A1) when it converts a PGEN
to BED.  So for a PGEN the counted allele is **REF**: a genotype is
``REF dosage / 2 = 1 - ALT dosage / 2``, and `Variants` has ``a2 = REF`` and
``a1 = ALT``.  The values (and the variant metadata) are then exactly those
`PlinkReader` reads from ``plink2 --pfile X --make-bed`` of the same file, and
a BED converted to PGEN by plink2 (which takes A2 as REF) reads back as the
original BED.  ``pylmm gwas`` reports its effects per copy of A1 (its
``effect_allele`` column), which for a PGEN is **ALT**, as ``plink2 --glm``,
REGENIE and GWAS-SSF report them; ``effect_allele_frequency`` is the mean ALT
dosage / 2 (dosages included).  Only ``--output-format legacy`` reports BETA
for the counted allele, REF.

Hard calls and dosages
----------------------
By default (``dosage=True``) the genotypes are the **dosages** where the file
has them (imputed data: ``plink2 --vcf ... dosage=DS`` or ``--bgen``), stored
at a precision of 1/16384, and the hard calls elsewhere; for a PGEN without
dosages (converted from BED or a hard-call VCF) both are the same.  The
values are exact: ``1 - d / 2`` of a multiple ``d`` of 2^-14 is a float64
without rounding.  ``dosage=False`` reads the hard calls only, which plink2
derives from the dosages when it imports them (``--hard-call-threshold``,
0.1 by default: a dosage more than 0.1 from 0, 1 or 2 is a missing hard call)
and which is what ``plink2 --make-bed`` writes.

Multiallelic variants
---------------------
A variant with more than one ALT allele (a comma in the ALT column) is an
`InputError` when the variants are read, naming the first one: pylmm tests one
allele per variant and does not choose one for you.  Drop them with
``plink2 --pfile X --max-alleles 2 --make-pgen --out Y``, or split them into
biallelic records first (``bcftools norm -m -any`` on the VCF).

Sample IDs
----------
From the ``.psam`` header (``#FID IID ...`` or ``#IID ...``; a file without a
header line has the ``.fam`` columns ``FID IID ...``).  Without an FID column
the FID is ``0``, as plink2 writes it to a ``.fam``, so PGEN and BED files of
the same samples match by ``(FID, IID)``; a phenotype file that repeats the IID
as FID (plink 1.9's ``--double-id``) then matches only with ``id_match="iid"``
(``--id-match iid``).  The SID column is ignored, and a sample listed twice
(the same FID and IID) is an `InputError`.

Variant IDs
-----------
The ``.pvar`` ID column is used as it is, as the ``.bim``'s is.  plink2 keeps a
VCF's missing ID ``.``, so every such variant is named ``.`` in the output: give
them IDs when converting (``plink2 ... --set-all-var-ids '@:#:$r:$a'``).

Examples
--------
Write a small PGEN (`write_pgen`) and read it back:

>>> import os, tempfile
>>> import numpy as np
>>> from pylmm.genotypes import Variants
>>> from pylmm.pgen import PgenSource, write_pgen
>>> tmp = tempfile.TemporaryDirectory()
>>> prefix = os.path.join(tmp.name, "toy")
>>> G = np.array([[0.0, 1.0], [0.5, np.nan], [1.0, 0.0]])  # fraction of REF copies
>>> v = Variants(snp=["rs1", "rs2"], chrom=["1", "2"], bp=[100, 200],
...              a1=["A", "C"], a2=["G", "T"])
>>> write_pgen(prefix, G, [("F1", "I1"), ("F1", "I2"), ("F2", "I3")], v)  # doctest: +SKIP
>>> src = PgenSource(prefix, norm_genotype=False)  # doctest: +SKIP
>>> src.sample_ids  # doctest: +SKIP
[('F1', 'I1'), ('F1', 'I2'), ('F2', 'I3')]
>>> block, bv = next(src.iter_blocks())  # doctest: +SKIP
>>> block.tolist(), bv.a2.tolist()  # doctest: +SKIP
([[0.0, 1.0], [0.5, nan], [1.0, 0.0]], ['G', 'T'])
>>> tmp.cleanup()
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pylmm import samples
from pylmm.genotypes import (
    DEFAULT_BLOCK_SIZE,
    GenotypeSource,
    Variants,
    _normalize_rows,
    check_block_size,
    normalize_genotype,
)
from pylmm.samples import InputError, SampleID

INSTALL_HINT = "pip install pgenlib (from a clone of pylmm: uv sync --extra pgen)"
"""How to install the optional reader (pgenlib)."""

# pgenlib's code for a missing hard call or dosage.
_MISSING = -9

# Hard call (copies of ALT, pgenlib's int8; -9 missing) + 9 -> fraction of REF copies.
_HARDCALL_VALUES = np.full(12, np.nan)
_HARDCALL_VALUES[9:12] = [1.0, 0.5, 0.0]


def _pgenlib() -> Any:
    """Import pgenlib, or raise ImportError saying how to install it."""
    try:
        import pgenlib
    except ImportError as e:
        raise ImportError(
            "reading PLINK 2 PGEN files needs the optional package pgenlib: %s "
            "(or convert the files to BED: plink2 --pfile PREFIX --make-bed --out NEW)"
            % INSTALL_HINT
        ) from e
    return pgenlib


def pgen_paths(prefix: str) -> tuple[str, str, str]:
    """The ``(.pgen, .pvar, .psam)`` paths of a PGEN file set.

    The variant file is ``prefix.pvar``, or ``prefix.pvar.zst`` if there is
    no ``prefix.pvar`` (``plink2 --make-pgen vzs``).

    Raises
    ------
    FileNotFoundError
        If the ``.pgen``, the ``.psam`` or both variant files are missing.
    """
    pgen, psam = prefix + ".pgen", prefix + ".psam"
    pvar = prefix + ".pvar"
    if not os.path.isfile(pvar) and os.path.isfile(pvar + ".zst"):
        pvar += ".zst"
    for fn in (pgen, pvar, psam):
        if not os.path.isfile(fn):
            raise FileNotFoundError("no such file %s (of the PGEN file set %s)" % (fn, prefix))
    return pgen, pvar, psam


def read_psam(path: str) -> list[SampleID]:
    """The ``(FID, IID)`` of each sample of a plink2 ``.psam`` (or ``.fam``) file.

    Lines starting with ``##`` are skipped.  A header line ``#FID IID ...``
    or ``#IID ...`` names the columns; without an FID column the FID is
    ``"0"``, as plink2 writes it to a ``.fam``.  A file without a header has
    the ``.fam`` columns ``FID IID ...``.  Other columns (SID, parents, sex,
    phenotypes) are ignored.

    Raises
    ------
    InputError
        If the header has no IID column, a line has too few fields, the
        file lists no sample, or a sample (FID, IID) twice.

    Examples
    --------
    >>> import os, tempfile
    >>> from pylmm.pgen import read_psam
    >>> tmp = tempfile.TemporaryDirectory()
    >>> fn = os.path.join(tmp.name, "x.psam")
    >>> with open(fn, "w") as f:
    ...     _ = f.write("#IID\\tSEX\\ns1\\t1\\ns2\\tNA\\n")
    >>> read_psam(fn)
    [('0', 's1'), ('0', 's2')]
    >>> tmp.cleanup()
    """
    ids: list[SampleID] = []
    i_fid: int | None = 0
    i_iid = 1
    width = 2
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                if ids:
                    raise InputError(
                        "%s, line %d: a header line after the samples" % (path, lineno)
                    )
                names = line[1:].split()
                if "IID" not in names:
                    raise InputError(
                        "%s: the header %r has no IID column (expected #FID IID ... or #IID ...)"
                        % (path, line.strip())
                    )
                i_iid = names.index("IID")
                i_fid = names.index("FID") if "FID" in names else None
                width = max(i_iid, -1 if i_fid is None else i_fid) + 1
                continue
            v = line.split()
            if not v:
                continue
            if len(v) < width:
                raise InputError(
                    "%s, line %d: expected at least %d fields, found %d"
                    % (path, lineno, width, len(v))
                )
            ids.append(("0" if i_fid is None else v[i_fid], v[i_iid]))
    if not ids:
        raise InputError("%s lists no samples" % path)
    seen: set[SampleID] = set()
    for dup in ids:
        if dup in seen:
            break
        seen.add(dup)
    else:
        return ids
    raise InputError(
        "%s: the sample %s is listed twice (pylmm identifies samples by FID and IID; the SID "
        "column is ignored)" % (path, samples.format_id(dup))
    )


def _check_biallelic(fn: str, ids: Sequence[str], alt: Sequence[str]) -> None:
    """InputError if a variant has more than one ALT allele."""
    multi = [k for k, a in enumerate(alt) if "," in a]
    if multi:
        k = multi[0]
        raise InputError(
            "%s: %d multiallelic variant(s), e.g. %s (ALT %s); pylmm reads biallelic variants "
            "only: drop them (plink2 --pfile PREFIX --max-alleles 2 --make-pgen --out NEW) or "
            "split them first (bcftools norm -m -any)" % (fn, len(multi), ids[k], alt[k])
        )


def read_pvar(path: str) -> Variants:
    """The variants of a plink2 ``.pvar`` (or ``.pvar.zst``) file, as `Variants`.

    ``snp`` is the ID column, ``chrom`` and ``bp`` the CHROM and POS
    columns, ``cm`` the CM column if there is one (else None), and the
    alleles follow pylmm's convention for PGEN (module docstring): ``a2`` is
    REF, the counted allele, and ``a1`` is ALT (``.`` when there is none), as
    in the ``.bim`` plink2 writes for the file.  A ``.pvar`` without a
    ``#CHROM`` header line has the ``.bim`` columns (CHROM ID CM POS ALT REF).
    A ``.pvar.zst`` is read with pgenlib (so it needs the ``pgen`` extra)
    and has no ``cm``.

    Raises
    ------
    InputError
        If a line has too few fields, a position is not an integer, a column
        is missing from the header, or a variant is multiallelic (see the
        module docstring).

    Examples
    --------
    >>> import os, tempfile
    >>> from pylmm.pgen import read_pvar
    >>> tmp = tempfile.TemporaryDirectory()
    >>> fn = os.path.join(tmp.name, "x.pvar")
    >>> with open(fn, "w") as f:
    ...     _ = f.write("##fileformat=PVARv1.0\\n#CHROM\\tPOS\\tID\\tREF\\tALT\\n"
    ...                 "1\\t100\\trs1\\tG\\tA\\n2\\t200\\trs2\\tT\\tC\\n")
    >>> v = read_pvar(fn)
    >>> v[0]
    Variant(chrom='1', snp='rs1', cm=None, bp=100, a1='A', a2='G')
    >>> tmp.cleanup()
    """
    if path.endswith(".zst"):
        return _read_pvar_pgenlib(path)
    cols: dict[str, int] | None = None
    chrom: list[str] = []
    ids: list[str] = []
    pos: list[str] = []
    cm: list[str] = []
    ref: list[str] = []
    alt: list[str] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                names = line[1:].split()
                missing = [c for c in ("CHROM", "POS", "ID", "REF", "ALT") if c not in names]
                if missing:
                    raise InputError(
                        "%s: the header line has no %s column" % (path, ", ".join(missing))
                    )
                cols = {
                    c: names.index(c)
                    for c in ("CHROM", "POS", "ID", "REF", "ALT", "CM")
                    if c in names
                }
                continue
            v = line.split()
            if not v:
                continue
            if cols is None:  # no header: the .bim columns
                cols = {"CHROM": 0, "ID": 1, "CM": 2, "POS": 3, "ALT": 4, "REF": 5}
            width = max(cols.values()) + 1
            if len(v) < width:
                raise InputError(
                    "%s, line %d: expected at least %d fields, found %d"
                    % (path, lineno, width, len(v))
                )
            chrom.append(v[cols["CHROM"]])
            ids.append(v[cols["ID"]])
            pos.append(v[cols["POS"]])
            ref.append(v[cols["REF"]])
            alt.append(v[cols["ALT"]])
            if "CM" in cols:
                cm.append(v[cols["CM"]])
    _check_biallelic(path, ids, alt)
    try:
        bp = np.array(pos, dtype=np.int64)
    except ValueError:
        bad = next(p for p in pos if not p.lstrip("-").isdigit())
        raise InputError("%s: the position %r is not an integer" % (path, bad)) from None
    cm_col = None
    if cols is not None and "CM" in cols:
        try:
            cm_col = np.array(cm, dtype=np.float64)
        except ValueError:
            raise InputError("%s: a CM value is not a number" % path) from None
    return Variants(snp=ids, chrom=chrom, cm=cm_col, bp=bp, a1=alt, a2=ref)  # type: ignore[arg-type]


def _read_pvar_pgenlib(path: str) -> Variants:
    """`read_pvar` for a ``.pvar.zst``: pgenlib's parser (no CM column)."""
    pgenlib = _pgenlib()
    try:
        pr = pgenlib.PvarReader(path.encode())
    except RuntimeError as e:
        raise InputError("%s: %s" % (path, _message(e))) from e
    try:
        m = pr.get_variant_ct()
        ids = [pr.get_variant_id(k).decode() for k in range(m)]
        if pr.get_max_allele_ct() > 2:
            alt = [
                ",".join(pr.get_allele_code(k, a).decode() for a in range(1, pr.get_allele_ct(k)))
                for k in range(m)
            ]
            _check_biallelic(path, ids, alt)
        chrom = [pr.get_variant_chrom(k).decode() for k in range(m)]
        bp = [pr.get_variant_pos(k) for k in range(m)]
        ref = [pr.get_allele_code(k, 0).decode() for k in range(m)]
        alt = [pr.get_allele_code(k, 1).decode() for k in range(m)]
    finally:
        pr.close()
    return Variants(snp=ids, chrom=chrom, bp=bp, a1=alt, a2=ref)  # type: ignore[arg-type]


def _message(e: BaseException) -> str:
    """The text of a pgenlib error (its argument may be ``bytes``), stripped."""
    arg = e.args[0] if e.args else ""
    if isinstance(arg, bytes):
        arg = arg.decode("utf-8", "replace")
    return str(arg).strip()


def _open_reader(pgen: str, n_file: int, m: int | None, subset: NDArray[np.uint32] | None) -> Any:
    """A ``pgenlib.PgenReader`` of `pgen` (`n_file` samples, `m` variants or None), on `subset`."""
    pgenlib = _pgenlib()
    try:
        return pgenlib.PgenReader(
            pgen.encode(), raw_sample_ct=n_file, variant_ct=m, sample_subset=subset
        )
    except RuntimeError as e:
        text = _message(e)
        hint = ""
        if "raw_sample_ct" in text:
            hint = (
                " (the .psam lists %d samples: are the .pgen and .psam of one file set?)" % n_file
            )
        elif "raw_variant_ct" in text and m is not None:
            hint = " (the .pvar lists %d variants: are the .pgen and .pvar of one file set?)" % m
        raise InputError("%s: %s%s" % (pgen, text, hint)) from e


def check_pgen(pgen: str, n_file: int) -> None:
    """Open the ``.pgen`` once: ImportError without pgenlib, InputError unless it has `n_file` samples."""
    _open_reader(pgen, n_file, None, None).close()


def iter_pgen_blocks(
    pgen: str,
    n_file: int,
    variants: Variants,
    index: NDArray[np.intp] | None,
    norm: bool,
    dosage: bool,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Iterator[tuple[NDArray[np.float64], Variants]]:
    """Yield ``(G, variants)`` blocks of the PGEN file `pgen` (the `PgenSource.iter_blocks` engine).

    Parameters
    ----------
    pgen : str
        The ``.pgen`` file.
    n_file : int
        The number of samples of the file (its ``.psam``).
    variants : Variants
        All its variants (its ``.pvar``).
    index : numpy.ndarray or None
        The file positions of the samples to return, in that order (None:
        all, in file order).
    norm : bool
        Standardise each SNP, bit for bit as `pylmm.genotypes.normalize_genotype`.
    dosage : bool
        Read dosages (else hard calls).
    block_size : int
        SNPs per block.

    Yields
    ------
    G : numpy.ndarray
        ``(len(index) or n_file, b)`` float64: the fraction of REF copies
        (nan if missing), standardised if `norm`.
    variants : Variants
        The ``b`` variants of the block.
    """
    block_size = check_block_size(block_size)
    m = len(variants)
    subset: NDArray[np.uint32] | None = None
    perm: NDArray[np.intp] | None = None
    if index is not None:
        order = np.sort(index)
        if len(order) != n_file:  # pgenlib reads a sorted subset of the samples
            subset = order.astype(np.uint32)
        if not np.array_equal(order, index):
            perm = np.searchsorted(order, index)
    n_read = n_file if subset is None else len(subset)
    if m == 0:
        return
    reader = _open_reader(pgen, n_file, m, subset)
    try:
        for start in range(0, m, block_size):
            b = min(block_size, m - start)
            try:
                if dosage:
                    R = np.empty((b, n_read), dtype=np.float64)
                    reader.read_dosages_range(start, start + b, R)
                    missing = R == _MISSING
                    R *= -0.5
                    R += 1.0  # 1 - d/2, exact for d a multiple of 2^-14
                    R[missing] = np.nan
                else:
                    codes = np.empty((b, n_read), dtype=np.int8)
                    reader.read_range(start, start + b, codes)
                    R = _HARDCALL_VALUES[codes.astype(np.intp) + 9]
            except RuntimeError as e:
                raise InputError(
                    "%s: reading variants %d-%d failed (%s): is the file truncated or corrupt?"
                    % (pgen, start + 1, start + b, _message(e))
                ) from e
            if perm is not None:
                R = R[:, perm]
            if norm:
                _normalize_rows(R)  # each row (SNP) as normalize_genotype(R[j])
            yield R.T, variants[start : start + b]
            del R
    finally:
        reader.close()


def iter_pgen_snps(
    pgen: str,
    n_file: int,
    variants: Variants,
    index: NDArray[np.intp] | None,
    norm: bool,
    dosage: bool,
) -> Iterator[tuple[NDArray[np.float64], str]]:
    """Yield ``(genotypes, snp_id)`` one SNP at a time (`PlinkReader` ``type="pgen"``'s iterator).

    The same values as `iter_pgen_blocks`, standardised per SNP with
    `pylmm.genotypes.normalize_genotype` if `norm`.
    """
    for G, v in iter_pgen_blocks(pgen, n_file, variants, index, False, dosage, 256):
        for j in range(G.shape[1]):
            g = G[:, j].copy()
            yield (normalize_genotype(g) if norm else g), str(v.snp[j])


class PgenSource:
    """A `pylmm.genotypes.GenotypeSource` over a PLINK 2 PGEN file set.

    Reads ``prefix.pgen``, ``prefix.pvar`` (or ``prefix.pvar.zst``) and
    ``prefix.psam``.  Needs the optional package pgenlib
    (``pip install pgenlib``).  See the module docstring for the
    genotype coding (fraction of REF copies; ``a2`` is REF), dosages, sample
    IDs and multiallelic variants.

    Parameters
    ----------
    prefix : str
        The file prefix (plink2 ``--pfile``).
    norm_genotype : bool, default True
        Standardise each SNP of a block, bit for bit as
        `pylmm.genotypes.normalize_genotype` (missing calls mean-imputed); if
        false, blocks hold the raw fractions, nan where missing.
    dosage : bool, default True
        Read the dosages where the file has them; false: the hard calls.
    id_match : {"fid_iid", "iid"}, default "fid_iid"
        How `subset` matches sample IDs.

    Attributes
    ----------
    prefix, pgen_file, pvar_file, psam_file : str
        The file set.
    N : int
        The number of samples in the file.
    norm_genotype, dosage : bool
        As given; may be changed between `iter_blocks` calls.

    Raises
    ------
    ImportError
        If pgenlib is not installed.
    FileNotFoundError
        If a file of the set is missing.
    InputError
        If the ``.psam`` is malformed or the ``.pgen`` does not match it.

    Examples
    --------
    See the module docstring.
    """

    def __init__(
        self,
        prefix: str,
        norm_genotype: bool = True,
        dosage: bool = True,
        id_match: str = "fid_iid",
    ) -> None:
        if id_match not in samples.MATCH_MODES:
            raise ValueError(
                "id_match=%r: use one of %s"
                % (id_match, ", ".join(repr(m) for m in samples.MATCH_MODES))
            )
        _pgenlib()  # fail early, with the install hint
        self.prefix = prefix
        self.pgen_file, self.pvar_file, self.psam_file = pgen_paths(prefix)
        self.norm_genotype = norm_genotype
        self.dosage = dosage
        self.id_match = id_match
        self._file_ids = read_psam(self.psam_file)
        self.N = len(self._file_ids)
        self._ids: list[SampleID] = list(self._file_ids)
        self._index: NDArray[np.intp] | None = None
        self._variants: Variants | None = None
        check_pgen(self.pgen_file, self.N)

    @property
    def sample_ids(self) -> list[SampleID]:
        """The ``(FID, IID)`` of the samples of a block, in row order."""
        return self._ids

    @property
    def num_samples(self) -> int:
        """The rows of a block."""
        return len(self._ids)

    @property
    def variants(self) -> Variants:
        """The variants of the ``.pvar`` (`read_pvar`; read on first use)."""
        if self._variants is None:
            self._variants = read_pvar(self.pvar_file)
        return self._variants

    def subset(self, ids: Sequence[SampleID]) -> None:
        """Keep only these samples, in this order (matched as ``id_match`` says).

        Raises
        ------
        ValueError
            If `ids` is empty, or an ID is absent or listed twice.
        """
        ids = list(ids)
        if not ids:
            raise ValueError("subset needs at least one sample")
        idx = samples.match_index(
            ids,
            self._ids,
            match=self.id_match,
            target_name="the requested samples",
            source_name="the PGEN file set %s" % self.prefix,
        )
        absent = [s for s, i in zip(ids, idx) if i < 0]
        if absent:
            raise ValueError(
                "%d of the requested samples are not in %s: %s"
                % (len(absent), self.psam_file, ", ".join(samples.format_id(s) for s in absent[:5]))
            )
        base = np.arange(self.N) if self._index is None else self._index
        index = base[idx]
        self._index = None if np.array_equal(index, np.arange(self.N)) else index
        self._ids = [self._ids[i] for i in idx]

    def iter_blocks(
        self, block_size: int = DEFAULT_BLOCK_SIZE
    ) -> Iterator[tuple[NDArray[np.float64], Variants]]:
        """Yield ``(G, variants)`` for consecutive blocks of up to `block_size` SNPs.

        ``G`` is float64, ``(num_samples, b)``, rows in the order of
        `sample_ids`: the fraction of REF copies (dosage or hard call; nan if
        missing), standardised if ``norm_genotype``.  The samples, options and
        variants are those of the time of the call.

        Raises
        ------
        ValueError
            If `block_size` is not a positive integer.
        InputError
            (A ``ValueError``.)  As `variants`, or if pgenlib cannot read the file.
        """
        block_size = check_block_size(block_size)
        index = None if self._index is None else self._index.copy()
        return iter_pgen_blocks(
            self.pgen_file,
            self.N,
            self.variants,
            index,
            self.norm_genotype,
            self.dosage,
            block_size,
        )


def write_pgen(
    prefix: str,
    G: ArrayLike,
    sample_ids: Sequence[SampleID],
    variants: Variants,
    dosage: bool = False,
) -> None:
    """Write genotypes as a PLINK 2 PGEN file set (``prefix.pgen/.pvar/.psam``), with pgenlib.

    The inverse of `PgenSource`: ``G`` holds pylmm's values, the fraction of
    copies of ``variants.a2`` (written as REF; ``a1`` is ALT), nan where
    missing.  Meant for tests, benchmarks and small conversions (e.g. an
    `pylmm.genotypes.ArraySource` to plink2); it holds `G` in memory.

    Parameters
    ----------
    prefix : str
        Output prefix.
    G : array_like
        ``(n, m)`` genotypes.  Hard calls (``dosage=False``) must be 0, 0.5,
        1 or nan; dosages must be in [0, 1] or nan.
    sample_ids : sequence of (FID, IID)
        One per row; written as ``#FID IID``.
    variants : Variants
        One per column; needs ``chrom``, ``bp``, ``a1`` and ``a2`` (``cm`` is
        written if present).
    dosage : bool, default False
        Write dosages (precision 1/16384) instead of hard calls.

    Raises
    ------
    ValueError
        If the shapes do not match, a column is missing from `variants`, a
        hard call is not 0, 0.5, 1 or nan, or a dosage is not in [0, 1] or
        nan (+-inf included).
    ImportError
        If pgenlib is not installed.
    """
    pgenlib = _pgenlib()
    A = np.asarray(G, dtype=np.float64)
    if A.ndim != 2 or A.shape != (len(sample_ids), len(variants)):
        raise ValueError(
            "G has shape %s, expected (%d samples, %d variants)"
            % (A.shape, len(sample_ids), len(variants))
        )
    if variants.chrom is None or variants.bp is None or variants.a1 is None or variants.a2 is None:
        raise ValueError("write_pgen needs the chrom, bp, a1 and a2 columns of variants")
    n, m = A.shape
    called = ~np.isnan(A)
    if dosage:
        bad = called & ~((A >= 0.0) & (A <= 1.0))  # inf is called and out of range
        if bad.any():
            i, k = np.argwhere(bad)[0]
            raise ValueError(
                "dosages must be in [0, 1] (fraction of REF copies) or nan, got %r at row %d, "
                "column %d (%d such values)" % (float(A[i, k]), i, k, int(bad.sum()))
            )
    elif not np.isin(A[called], (0.0, 0.5, 1.0)).all():
        raise ValueError("hard calls must be 0, 0.5, 1 or nan (fraction of REF copies)")
    alt = 2.0 - 2.0 * A  # copies of ALT
    with open(prefix + ".psam", "w") as f:
        f.write("#FID\tIID\n")
        for fid, iid in sample_ids:
            f.write("%s\t%s\n" % (fid, iid))
    with open(prefix + ".pvar", "w") as f:
        f.write("#CHROM\tPOS\tID\tREF\tALT%s\n" % ("\tCM" if variants.cm is not None else ""))
        for k in range(m):
            f.write(
                "%s\t%d\t%s\t%s\t%s%s\n"
                % (
                    variants.chrom[k],
                    variants.bp[k],
                    variants.snp[k],
                    variants.a2[k],
                    variants.a1[k],
                    "" if variants.cm is None else "\t%r" % float(variants.cm[k]),
                )
            )
    writer = pgenlib.PgenWriter(
        (prefix + ".pgen").encode(), n, variant_ct=m, nonref_flags=False, dosage_present=dosage
    )
    try:
        for start in range(0, m, DEFAULT_BLOCK_SIZE):
            X = np.ascontiguousarray(alt[:, start : start + DEFAULT_BLOCK_SIZE].T)
            if dosage:
                X[np.isnan(X)] = _MISSING
                writer.append_dosages_batch(X)
            else:
                codes = np.where(np.isnan(X), _MISSING, X).astype(np.int8)
                writer.append_biallelic_batch(codes)
    finally:
        writer.close()


if TYPE_CHECKING:
    # PgenSource implements the GenotypeSource protocol (checked by mypy).
    def _is_genotype_source(s: PgenSource) -> GenotypeSource:
        return s
