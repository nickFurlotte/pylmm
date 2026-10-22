# pylmm is a python-based linear mixed-model solver with applications to GWAS
# Copyright (C) 2013-2026 Nicholas A. Furlotte (nick.furlotte@gmail.com)

# SPDX-License-Identifier: MIT
# See the LICENSE file in the project root for full license text.

"""Summary statistics: per-allele effects in a GWAS-SSF-style table.

``pylmm gwas`` writes its results as a tab-separated table (gzip-compressed
when the file name ends in ``.gz``) whose first eight columns are the
mandatory columns of the GWAS Catalog summary-statistics format, GWAS-SSF
v1.1.0, in its order, followed by optional and pylmm-specific columns
(`SUMSTATS_COLUMNS`):

=========================  ====================================================
column                     meaning
=========================  ====================================================
chromosome                 the chromosome, as written in the genotype input
base_pair_location         the position (``.bim`` column 4)
effect_allele              the allele whose copies ``beta`` is the effect of:
                           A1 (``.bim`` column 5; ALT for plink2-written files)
other_allele               the other allele (A2, ``.bim`` column 6)
beta                       the effect of one copy of the effect allele
standard_error             its standard error
effect_allele_frequency    the effect allele's frequency among the individuals
                           used, over their called genotypes
p_value                    the p-value of the test, never written as 0 when it
                           underflows (see `format_p_values`)
variant_id                 the SNP ID of the genotype input, verbatim
rsid                       the SNP ID if it is an rsID (``rs`` + digits), else NA
n                          the number of individuals the SNP was tested on
t_stat                     the test statistic, signed for the effect allele
neg_log_10_p_value         ``-log10 p``, accurate where ``p`` underflows
missing_rate               the share of the individuals with a phenotype
                           (and covariates) whose genotype is missing
beta_std                   ``beta`` times the standard deviation of the effect
                           allele count over the called genotypes used: the
                           effect of one SD of the genotype
=========================  ====================================================

Missing values are written ``NA``.  Floats are written with Python's
``repr`` (the shortest string that reads back as the same double), integers
as integers.

Deviations from GWAS-SSF v1.1.0, all deliberate: missing values are ``NA``
rather than the recommended ``#NA`` (R's ``read.table`` treats ``#`` as a
comment and its readers do not take ``#NA`` as missing; pandas reads both);
``chromosome`` is written as in the genotype input (``X``, mouse 20, ...)
rather than recoded to 1-25; ``variant_id`` is the input's SNP ID, not a
``chr_pos_ref_alt`` string (pylmm does not know which allele is the
reference); the alleles are written as in the input (PLINK's ``0`` for a
missing allele becomes ``NA``).  The GWAS-SSF metadata YAML is not written;
``pylmm gwas`` writes a JSON run record instead.

The genotype coding
-------------------
pylmm reads a genotype as the fraction of copies of A2 (0, 0.5, 1), and
``pylmm gwas`` tests it standardised.  The table reports the effect of the
other allele, A1, per copy: the convention of plink2 ``--glm``, REGENIE,
GEMMA, GCTA and GWAS-SSF (effect allele = ALT).  If the model column is
``x = (g - m) / s`` for the A2 fraction ``g``, the A1 count is ``c = 2 - 2 g``
and ``dx/dc = -1 / (2 s)``, so::

    beta = beta_x * dx/dc        standard_error = se_x * |dx/dc|
    t_stat = t_x * sign(dx/dc)

whatever ``s`` is: `per_allele_effects` does this.  ``t_stat`` and the
p-value do not depend on the coding; only the sign of ``t_stat`` follows
the effect allele.

Examples
--------
A small table, written and read back:

>>> import gzip, os, tempfile
>>> import numpy as np
>>> from pylmm.genotypes import Variants
>>> from pylmm.sumstats import SumStats, write_sumstats
>>> v = Variants(snp=["rs1", "1:200"], chrom=["1", "1"], bp=[100, 200], a1=["A", "C"], a2=["G", "T"])
>>> table = SumStats.from_variants(
...     v, beta=[0.5, np.nan], standard_error=[0.1, np.nan], t_stat=[5.0, np.nan],
...     p_value=[1e-6, np.nan], neg_log_10_p_value=[6.0, np.nan],
...     effect_allele_frequency=[0.25, 0.0], n=[100, np.nan], missing_rate=[0.0, 0.0],
...     beta_std=[0.2, np.nan])
>>> tmp = tempfile.TemporaryDirectory()
>>> path = os.path.join(tmp.name, "out.tsv.gz")
>>> write_sumstats(path, table)
>>> with gzip.open(path, "rt") as f:
...     lines = f.read().splitlines()
>>> lines[0].split("\\t")[:8]
['chromosome', 'base_pair_location', 'effect_allele', 'other_allele', 'beta', 'standard_error', 'effect_allele_frequency', 'p_value']
>>> lines[1].split("\\t")
['1', '100', 'A', 'G', '0.5', '0.1', '0.25', '1e-06', 'rs1', 'rs1', '100', '5.0', '6.0', '0.0', '0.2']
>>> lines[2].split("\\t")[4:11]
['NA', 'NA', '0.0', 'NA', '1:200', 'NA', 'NA']
>>> tmp.cleanup()
"""

from __future__ import annotations

import gzip
import io
import math
import re
from dataclasses import dataclass, fields
from typing import IO, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from pylmm.genotypes import Variants

FloatArray = NDArray[np.float64]

SUMSTATS_COLUMNS = (
    "chromosome",
    "base_pair_location",
    "effect_allele",
    "other_allele",
    "beta",
    "standard_error",
    "effect_allele_frequency",
    "p_value",
    "variant_id",
    "rsid",
    "n",
    "t_stat",
    "neg_log_10_p_value",
    "missing_rate",
    "beta_std",
)
"""The columns of the summary-statistics table, in order: the eight mandatory
GWAS-SSF v1.1.0 columns, then ``variant_id``, ``rsid`` and ``n`` (optional in
GWAS-SSF), then pylmm's own."""

SUMSTATS_FORMAT = "pylmm.sumstats/1"
"""The version of the table's layout, recorded in the JSON run record."""

NA = "NA"
"""How a missing value is written."""

P_VALUE_DIGITS = 6
"""Significant digits of a p-value written from ``-log10 p`` (`format_p_values`)."""

P_VALUE_DIRECT_MIN = 1e-300
"""p-values at or above this are written as computed; below it (or 0) they are
written from ``-log10 p``."""

_RSID = re.compile(r"^rs[0-9]+$")
_MISSING_ALLELES = frozenset(("", "0", ".", "-9", "NA"))

Coding = Literal["a2_fraction", "a1_count"]
"""Genotype codings `SumStats.from_genotypes` accepts: ``a2_fraction`` (pylmm's raw
coding, 0 / 0.5 / 1 copies of A2 halved, as ``PlinkReader(norm_genotype=False)``
reads it) or ``a1_count`` (0 / 1 / 2 copies of A1, e.g. an ALT dosage)."""


def _float_column(values: ArrayLike, m: int, name: str) -> FloatArray:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if a.shape != (m,):
        raise ValueError("SumStats.%s has %d values, expected %d" % (name, a.size, m))
    return a


def _text_column(values: ArrayLike | None, m: int, name: str) -> NDArray[np.object_]:
    if values is None:
        return np.full(m, None, dtype=object)
    a = np.empty(m, dtype=object)
    vals = list(np.asarray(values, dtype=object).reshape(-1))
    if len(vals) != m:
        raise ValueError("SumStats.%s has %d values, expected %d" % (name, len(vals), m))
    a[:] = [None if v is None else str(v) for v in vals]
    return a


def _allele_column(values: ArrayLike | None, m: int, name: str) -> NDArray[np.object_]:
    """Alleles as text; PLINK's missing-allele codes (``0``, ``.``) become None."""
    a = _text_column(values, m, name)
    for k, v in enumerate(a):
        if v is not None and v in _MISSING_ALLELES:
            a[k] = None
    return a


@dataclass(frozen=True, eq=False)
class SumStats:
    """A summary-statistics table: one entry per SNP in each column.

    Build one with `SumStats.from_variants` (from effects already on the
    per-copy scale of the effect allele) or `SumStats.from_genotypes` (from
    the results of `pylmm.lmm.gwas` and the genotypes), and write it with
    `write_sumstats`.  The attributes are the columns of `SUMSTATS_COLUMNS`
    (``rsid`` is derived from ``variant_id`` when written): text columns
    are object arrays with None for missing, numeric columns float64 with
    nan for missing (``n`` too).
    """

    chromosome: NDArray[np.object_]
    base_pair_location: FloatArray
    effect_allele: NDArray[np.object_]
    other_allele: NDArray[np.object_]
    beta: FloatArray
    standard_error: FloatArray
    effect_allele_frequency: FloatArray
    p_value: FloatArray
    variant_id: NDArray[np.object_]
    n: FloatArray
    t_stat: FloatArray
    neg_log_10_p_value: FloatArray
    missing_rate: FloatArray
    beta_std: FloatArray

    def __post_init__(self) -> None:
        m = len(self.variant_id)
        for f in fields(self):
            v = getattr(self, f.name)
            if len(v) != m:
                raise ValueError("SumStats.%s has %d values, expected %d" % (f.name, len(v), m))

    def __len__(self) -> int:
        return len(self.variant_id)

    @classmethod
    def from_variants(
        cls,
        variants: Variants,
        *,
        beta: ArrayLike,
        standard_error: ArrayLike,
        t_stat: ArrayLike,
        p_value: ArrayLike,
        neg_log_10_p_value: ArrayLike | None = None,
        effect_allele_frequency: ArrayLike,
        n: ArrayLike,
        missing_rate: ArrayLike,
        beta_std: ArrayLike,
        effect_allele: ArrayLike | None = None,
        other_allele: ArrayLike | None = None,
    ) -> SumStats:
        """A table for ``variants`` from effects already per copy of the effect allele.

        Parameters
        ----------
        variants : pylmm.genotypes.Variants
            The SNPs, in order: ``snp`` is the ``variant_id``, ``chrom`` and
            ``bp`` the position (NA where None).
        beta, standard_error, t_stat, p_value, effect_allele_frequency, n, missing_rate, beta_std : array_like
            One value per SNP, as the columns of `SUMSTATS_COLUMNS`; nan
            for missing.
        neg_log_10_p_value : array_like, optional
            Default: ``-log10(p_value)``.  Give it from the test statistic
            (e.g. `pylmm.lmm.neg_log10_pvalue`) when p-values can underflow.
        effect_allele, other_allele : array_like of str, optional
            Default: ``variants.a1`` and ``variants.a2``.
        """
        m = len(variants)
        p = _float_column(p_value, m, "p_value")
        if neg_log_10_p_value is None:
            with np.errstate(divide="ignore"):
                nl10 = -np.log10(p)
        else:
            nl10 = _float_column(neg_log_10_p_value, m, "neg_log_10_p_value")
        bp = (
            np.full(m, np.nan)
            if variants.bp is None
            else np.where(variants.bp > 0, variants.bp, np.nan).astype(np.float64)
        )
        return cls(
            chromosome=_text_column(variants.chrom, m, "chromosome"),
            base_pair_location=bp,
            effect_allele=_allele_column(
                variants.a1 if effect_allele is None else effect_allele, m, "effect_allele"
            ),
            other_allele=_allele_column(
                variants.a2 if other_allele is None else other_allele, m, "other_allele"
            ),
            beta=_float_column(beta, m, "beta"),
            standard_error=_float_column(standard_error, m, "standard_error"),
            effect_allele_frequency=_float_column(
                effect_allele_frequency, m, "effect_allele_frequency"
            ),
            p_value=p,
            variant_id=_text_column(variants.snp, m, "variant_id"),
            n=_float_column(n, m, "n"),
            t_stat=_float_column(t_stat, m, "t_stat"),
            neg_log_10_p_value=nl10,
            missing_rate=_float_column(missing_rate, m, "missing_rate"),
            beta_std=_float_column(beta_std, m, "beta_std"),
        )

    @classmethod
    def from_genotypes(
        cls,
        result: Any,
        X: ArrayLike,
        variants: Variants | None = None,
        *,
        coding: Coding = "a2_fraction",
        Y: ArrayLike | None = None,
    ) -> SumStats:
        """The table of a `pylmm.lmm.gwas` scan, per copy of the effect allele A1.

        Parameters
        ----------
        result : pylmm.lmm.GwasResult
            ``gwas(Y, X, K, ..., return_details=True)``: effects per unit
            of the columns of ``X``.
        X : (n, m) array
            The genotypes the scan tested, raw: ``coding`` says how.  Not
            standardised (the per-copy scale is lost then).
        variants : pylmm.genotypes.Variants, optional
            The SNPs' metadata; default: IDs ``SNP_1``, ``SNP_2``, ...
            and no positions or alleles.
        coding : {"a2_fraction", "a1_count"}, default "a2_fraction"
            ``a2_fraction``: the fraction of copies of A2 (``variants.a2``)
            in 0..1, as ``PlinkReader(norm_genotype=False)`` and
            ``ArraySource`` with such data give it; the effect allele is A1.
            ``a1_count``: copies of A1 (``variants.a1``) in 0..2, e.g. an
            ALT dosage; the effect allele is A1.
        Y : (n,) array, optional
            The phenotype the scan used: individuals with a missing (nan)
            phenotype are left out of ``effect_allele_frequency`` and
            ``missing_rate``.  Default: all individuals.

        Returns
        -------
        SumStats

        Raises
        ------
        ValueError
            For an unknown ``coding``, genotypes outside its range (e.g.
            standardised ones), or sizes that do not match.

        Examples
        --------
        >>> import numpy as np
        >>> from pylmm.lmm import calculate_kinship, gwas
        >>> from pylmm.sumstats import SumStats
        >>> rng = np.random.default_rng(1)
        >>> C = rng.binomial(2, 0.3, size=(100, 50)).astype(float)   # copies of A1
        >>> y = 0.5 * C[:, 0] + rng.normal(size=100)
        >>> res = gwas(y, C, calculate_kinship(C), return_details=True)
        >>> table = SumStats.from_genotypes(res, C, coding="a1_count")
        >>> bool(abs(table.beta[0] - 0.5) < 0.3), float(table.n[0])
        (True, 100.0)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_all, m = X.shape
        if variants is None:
            variants = Variants.from_names(["SNP_%d" % (k + 1) for k in range(m)])
        if len(variants) != m:
            raise ValueError("%d variants for %d SNP columns" % (len(variants), m))
        used = np.ones(n_all, dtype=bool)
        if Y is not None:
            y_used = np.asarray(Y, dtype=np.float64).reshape(-1)
            if y_used.shape != (n_all,):
                raise ValueError("Y has %d values but X has %d rows" % (y_used.size, n_all))
            used = ~np.isnan(y_used)
        x_used = X[used]
        if coding == "a2_fraction":
            counts, hi = 2.0 - 2.0 * x_used, 1.0
            slope = -0.5  # d(fraction of A2) / d(copies of A1)
        elif coding == "a1_count":
            counts, hi = x_used, 2.0
            slope = 1.0
        else:
            raise ValueError("coding must be 'a2_fraction' or 'a1_count', not %r" % (coding,))
        called = ~np.isnan(x_used)
        if np.any((x_used[called] < 0) | (x_used[called] > hi)):
            raise ValueError(
                "genotypes outside 0..%g: not %s coded (pass the raw genotypes, not "
                "standardised ones)" % (hi, coding)
            )
        freq, sd, missing = allele_count_stats(counts)
        beta, se, t = per_allele_effects(result.beta, result.beta_se, result.t, slope)
        return cls.from_variants(
            variants,
            beta=beta,
            standard_error=se,
            t_stat=t,
            p_value=result.p,
            neg_log_10_p_value=result.neg_log10_p,
            effect_allele_frequency=freq,
            n=result.n,
            missing_rate=missing,
            beta_std=beta * sd,
        )


def allele_count_stats(C: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per-SNP effect allele frequency, count SD and missing rate.

    Parameters
    ----------
    C : (n, m) array
        Copies of the effect allele (0..2; nan = missing) of the
        individuals to describe, one column per SNP.

    Returns
    -------
    freq : (m,) array
        ``mean(C) / 2`` over the called values (nan if none).
    sd : (m,) array
        The population SD of the called values (nan if none).
    missing_rate : (m,) array
        The share of missing values (nan if ``n == 0``).

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.sumstats import allele_count_stats
    >>> f, sd, miss = allele_count_stats(np.array([[0.0, 2.0], [2.0, np.nan], [1.0, 2.0]]))
    >>> f.tolist(), sd.round(4).tolist(), miss.round(4).tolist()
    ([0.5, 1.0], [0.8165, 0.0], [0.0, 0.3333])
    """
    A = np.asarray(C, dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    n, m = A.shape
    # One SNP per contiguous row, summed along it: the same sums (bit for bit) whatever the
    # memory layout of C.
    R = np.ascontiguousarray(A.T)
    called = ~np.isnan(R)
    k = called.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(called, R, 0.0).sum(axis=1) / k
        var = np.where(called, (R - mean[:, None]) ** 2, 0.0).sum(axis=1) / k
        missing = 1.0 - k / n if n else np.full(m, np.nan)
    mean = np.where(k > 0, mean, np.nan)
    sd = np.where(k > 0, np.sqrt(var), np.nan)
    return mean / 2.0, sd, np.asarray(missing, dtype=np.float64)


def per_allele_effects(
    beta_x: ArrayLike, se_x: ArrayLike, t_x: ArrayLike, slope: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Effects per unit of a model column ``x`` -> per copy ``c`` of the effect allele.

    Parameters
    ----------
    beta_x, se_x, t_x : array_like
        The effect, its standard error and the test statistic, per unit of
        the tested column ``x``.
    slope : array_like
        ``dx/dc``: the change of ``x`` per copy of the effect allele, e.g.
        ``-1 / (2 s)`` for ``x = (g - m) / s`` and ``g`` the fraction of A2.

    Returns
    -------
    beta, standard_error, t_stat : numpy.ndarray
        ``beta_x * slope``, ``se_x * |slope|``, ``t_x * sign(slope)``.

    Examples
    --------
    >>> from pylmm.sumstats import per_allele_effects
    >>> b, se, t = per_allele_effects([0.4], [0.1], [4.0], -0.5)
    >>> b.tolist(), se.tolist(), t.tolist()
    ([-0.2], [0.05], [-4.0])
    """
    s = np.asarray(slope, dtype=np.float64)
    beta = np.asarray(beta_x, dtype=np.float64) * s
    se = np.asarray(se_x, dtype=np.float64) * np.abs(s)
    t = np.asarray(t_x, dtype=np.float64) * np.sign(s)
    return beta, se, t


def lambda_gc(p_values: ArrayLike) -> float:
    """Genomic control: ``median(chi2) / 0.4549`` of the 1-df statistics of the p-values.

    nan p-values are ignored; nan if there is none.  A p-value of 0 counts
    as an infinite statistic (it can only move the median up by one place).

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.sumstats import lambda_gc
    >>> round(lambda_gc(np.random.default_rng(0).uniform(size=100000)), 2)
    1.0
    """
    p = np.asarray(p_values, dtype=np.float64).reshape(-1)
    p = p[~np.isnan(p)]
    if not p.size:
        return float("nan")
    return float(np.median(stats.chi2.isf(p, 1)) / stats.chi2.ppf(0.5, 1))


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _format_floats(a: FloatArray) -> list[str]:
    """``repr`` of each value (shortest round-trip), ``NA`` for nan."""
    return [NA if x != x else repr(x) for x in a.tolist()]


def _format_ints(a: FloatArray) -> list[str]:
    return [NA if x != x else str(int(x)) for x in a.tolist()]


def _format_text(a: NDArray[np.object_]) -> list[str]:
    return [NA if x is None else str(x) for x in a]


def format_p_values(p: ArrayLike, neg_log10_p: ArrayLike) -> list[str]:
    """p-values as text, never ``0`` while ``-log10 p`` is finite.

    A p-value at or above `P_VALUE_DIRECT_MIN` (1e-300) is written with
    ``repr``; a smaller one (subnormal, or 0 after underflow) is written as
    ``10 ** -neg_log10_p`` in scientific notation with `P_VALUE_DIGITS`
    significant digits (e.g. ``1.23457e-400``), beyond the range of a
    double.  ``NA`` where ``p`` is nan.

    Examples
    --------
    >>> import numpy as np
    >>> from pylmm.sumstats import format_p_values
    >>> format_p_values([0.05, 0.0, np.nan], [1.30103, 400.5, np.nan])
    ['0.05', '3.16228e-401', 'NA']
    """
    out: list[str] = []
    pv = np.asarray(p, dtype=np.float64).reshape(-1).tolist()
    nl = np.asarray(neg_log10_p, dtype=np.float64).reshape(-1).tolist()
    for x, L in zip(pv, nl):
        if x != x:
            out.append(NA)
        elif x >= P_VALUE_DIRECT_MIN or not math.isfinite(L) or L <= 0:
            out.append(repr(x))
        else:
            e = math.ceil(L)
            mant = 10.0 ** (e - L)  # in (1, 10]
            text = "%.*g" % (P_VALUE_DIGITS, mant)
            if float(text) >= 10.0:  # rounded up to 10
                text, e = "%.*g" % (P_VALUE_DIGITS, mant / 10.0), e - 1
            out.append("%se-%d" % (text, e))
    return out


def _rsids(ids: NDArray[np.object_]) -> list[str]:
    return [x if x is not None and _RSID.match(x) else NA for x in ids]


def sumstats_lines(table: SumStats) -> list[str]:
    """The data lines of ``table`` (without the header), each ending in a newline."""
    cols = [
        _format_text(table.chromosome),
        _format_ints(table.base_pair_location),
        _format_text(table.effect_allele),
        _format_text(table.other_allele),
        _format_floats(table.beta),
        _format_floats(table.standard_error),
        _format_floats(table.effect_allele_frequency),
        format_p_values(table.p_value, table.neg_log_10_p_value),
        _format_text(table.variant_id),
        _rsids(table.variant_id),
        _format_ints(table.n),
        _format_floats(table.t_stat),
        _format_floats(table.neg_log_10_p_value),
        _format_floats(table.missing_rate),
        _format_floats(table.beta_std),
    ]
    return ["\t".join(row) + "\n" for row in zip(*cols)]


GZIP_LEVEL = 6
"""gzip compression level of a ``.gz`` table (zlib's default trade-off)."""


class SumstatsWriter:
    """Write a summary-statistics table a block of SNPs at a time.

    The file is gzip-compressed if its name ends in ``.gz`` (level
    `GZIP_LEVEL`, with no time stamp or name in the gzip header, so that the
    same table gives the same bytes), plain text otherwise.  The header is
    written when the writer is created.  Use it as a context manager, or
    call `close`.

    Parameters
    ----------
    path : str
        The file (overwritten).
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.n_rows = 0
        raw = open(path, "wb")
        try:
            if path.endswith(".gz"):
                gz = gzip.GzipFile(
                    filename="", mode="wb", compresslevel=GZIP_LEVEL, fileobj=raw, mtime=0
                )
                self._raw: IO[bytes] | None = raw
                self._out: IO[str] = io.TextIOWrapper(gz, encoding="utf-8", newline="\n")
            else:
                self._raw = None
                self._out = io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
            self._out.write("\t".join(SUMSTATS_COLUMNS) + "\n")
        except BaseException:
            raw.close()
            raise

    def write(self, table: SumStats) -> None:
        """Append the rows of ``table``."""
        self._out.writelines(sumstats_lines(table))
        self.n_rows += len(table)

    def close(self) -> None:
        """Flush and close the file."""
        self._out.close()  # closes the gzip stream (writing its trailer)
        if self._raw is not None:
            self._raw.close()

    def __enter__(self) -> SumstatsWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_sumstats(path: str, table: SumStats) -> None:
    """Write ``table`` to ``path``: tab-separated, gzip-compressed if ``path`` ends in ``.gz``.

    See the module docstring for the columns, and `SumstatsWriter`.
    ``pandas.read_csv(path, sep="\\t")`` reads it back (``NA`` as missing).
    """
    with SumstatsWriter(path) as w:
        w.write(table)


TEXT_COLUMNS = frozenset(("chromosome", "effect_allele", "other_allele", "variant_id", "rsid"))
"""The columns `read_sumstats` returns as text; every other column is numeric."""


def read_sumstats(path: str) -> dict[str, NDArray[Any]]:
    """Read a summary-statistics table (`write_sumstats`, ``pylmm gwas``) into numpy columns.

    Parameters
    ----------
    path : str
        The table, gzip-compressed if the name ends in ``.gz``.

    Returns
    -------
    dict
        One array per column, in file order: `TEXT_COLUMNS` as ``str``
        arrays (``"NA"`` kept as is), every other column float64 with nan for
        ``NA`` (``p_value`` below the double range reads as 0.0: use
        ``neg_log_10_p_value``).  Columns pylmm does not know are read as text.
        ``pandas.read_csv(path, sep="\t")`` reads the same file.

    Raises
    ------
    ValueError
        If a line has another number of fields than the header, or a numeric
        column holds something that is not a number.

    Examples
    --------
    >>> import os, tempfile
    >>> from pylmm.genotypes import Variants
    >>> from pylmm.sumstats import SumStats, read_sumstats, write_sumstats
    >>> t = SumStats.from_variants(
    ...     Variants(snp=["rs1"], chrom=["1"], bp=[5], a1=["A"], a2=["G"]), beta=[0.5],
    ...     standard_error=[0.1], t_stat=[5.0], p_value=[0.0], neg_log_10_p_value=[400.0],
    ...     effect_allele_frequency=[0.2], n=[90], missing_rate=[0.0], beta_std=[0.3])
    >>> path = os.path.join(tempfile.mkdtemp(), "t.tsv.gz")
    >>> write_sumstats(path, t)
    >>> r = read_sumstats(path)
    >>> str(r["variant_id"][0]), float(r["beta"][0]), float(r["neg_log_10_p_value"][0]), str(r["rsid"][0])
    ('rs1', 0.5, 400.0, 'rs1')
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    for k, r in enumerate(rows):
        if len(r) != len(header):
            raise ValueError(
                "%s, line %d: %d fields, the header has %d" % (path, k + 2, len(r), len(header))
            )
    out: dict[str, NDArray[Any]] = {}
    known = set(SUMSTATS_COLUMNS)
    for j, name in enumerate(header):
        values = [r[j] for r in rows]
        if name in TEXT_COLUMNS or name not in known:
            out[name] = np.array(values, dtype=str)
            continue
        try:
            out[name] = np.array([np.nan if v == NA else float(v) for v in values])
        except ValueError as e:
            raise ValueError("%s, column %s: %s" % (path, name, e)) from e
    return out
