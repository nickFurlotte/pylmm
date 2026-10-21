"""Phase 6.1c: a block of SNPs standardised at once, bit for bit as SNP by SNP.

`pylmm.genotypes._normalize_rows` must give every SNP (row) exactly the raw
float64 bits of `normalize_genotype` (nan payloads and the sign of zero
included), on every numpy and platform CI runs (numpy 1.23.2 to current,
x86-64 Linux; arm64 macOS locally).  Its identity rests on numpy reducing a
row of a C-contiguous 2-d array exactly as it reduces the same values as a
1-d array; the fuzz below checks that on adversarial input, calling the block
code directly (`_normalize_c_rows`), so that the per-SNP fallback cannot hide
a difference.  See the comment above `_normalize_rows`.
"""

import warnings

import numpy as np
import pytest
from synthetic_bed import write_synthetic_bed

from pylmm import genotypes
from pylmm.genotypes import ArraySource, _normalize_c_rows, _normalize_rows, normalize_genotype
from pylmm.plink_io import PlinkReader

# Row lengths around the boundaries of numpy's pairwise summation (a plain loop below 8
# values, 8 accumulators up to 128, recursive halving above) and of its 8192-element
# reduction buffer.
NS = [*range(1, 10), 15, 16, 17, 63, 64, 65, 127, 128, 129, 136, 137, 255, 257]
NS += [1000, 1219, 4095, 4097]
NS_LARGE = [8191, 8193, 16385]


def _bits(a):
    return np.ascontiguousarray(a).view(np.int64)


def _per_snp(R):
    """The reference: normalize_genotype on each row (a copy of it)."""
    return np.array([normalize_genotype(R[j].copy()) for j in range(R.shape[0])]).reshape(R.shape)


def _adversarial_block(rng, n, b, kind, missing):
    """A (b, n) block of one kind of values with one missing-call pattern."""
    if kind == "bed":
        R = rng.binomial(2, rng.uniform(0.01, 0.99, b)[:, None], (b, n)) / 2.0
    elif kind == "dosage":
        R = rng.uniform(0, 2, (b, n))
    elif kind == "huge":
        R = rng.standard_normal((b, n)) * 10.0 ** rng.integers(100, 308, (b, 1))
    elif kind == "tiny":  # down to subnormals
        R = rng.standard_normal((b, n)) * 10.0 ** -rng.integers(100, 320, (b, 1))
    elif kind == "near_constant":
        base = rng.uniform(-5, 5, (b, 1))
        R = base + base * rng.choice([0.0, 1e-16, -1e-16, 1e-13, 1e-11], (b, n))
    elif kind == "mixed_scales":  # every element its own magnitude: order-sensitive sums
        R = rng.standard_normal((b, n)) * 10.0 ** rng.integers(-30, 30, (b, n))
    elif kind == "offset_counts":
        R = rng.integers(0, 3, (b, n)) + rng.choice([0.0, 1e8, 1e15, -1e15], (b, 1))
    else:  # "specials"
        R = rng.standard_normal((b, n))
        R[rng.random((b, n)) < 0.02] = -0.0
        R[rng.random((b, n)) < 0.01] = rng.choice([np.inf, -np.inf, 1e308, -1e308, 5e-324])
    R = np.ascontiguousarray(R, dtype=np.float64)
    if missing == "sparse":
        R[rng.random((b, n)) < 0.02] = np.nan
    elif missing == "heavy":
        R[rng.random((b, n)) < 0.6] = np.nan
    elif missing == "rows":  # all missing, a single call, and partly missing rows
        u = rng.random(b)
        R[u < 0.15] = np.nan
        for i in np.flatnonzero((u >= 0.15) & (u < 0.3)):
            keep = R[i, rng.integers(n)]
            R[i] = np.nan
            R[i, rng.integers(n)] = keep
        R[(u >= 0.3)[:, None] & (rng.random((b, n)) < 0.05)] = np.nan
    elif missing == "equal_counts":  # one group: same count, different positions
        k = int(rng.integers(0, n))
        for i in range(b):
            R[i, rng.choice(n, k, replace=False)] = np.nan
    elif missing == "constant_rows":
        R[rng.random(b) < 0.3] = rng.choice([0.0, 0.5, 1.0, 1e10])
        R[rng.random((b, n)) < 0.02] = np.nan
    return R


KINDS = [
    "bed",
    "dosage",
    "huge",
    "tiny",
    "near_constant",
    "mixed_scales",
    "offset_counts",
    "specials",
]
MISSING = ["none", "sparse", "heavy", "rows", "equal_counts", "constant_rows"]


# Missing values with other nan bits than np.nan: R's NA_real_ (a signalling nan, what
# survives when an R matrix reaches numpy without arithmetic) and a negative quiet nan
# with a payload.
R_NA = np.array([0x7FF00000000007A2], dtype=np.uint64).view(np.float64)[0]
NEG_PAYLOAD_NAN = np.array([0xFFF8000000000ABC], dtype=np.uint64).view(np.float64)[0]


def _with_nan(R, bits):
    """R with every nan replaced by the nan `bits` (np.nan: R itself)."""
    R = R.copy()
    R[np.isnan(R)] = bits
    return R


def _check(R):
    """The block path on R (C-contiguous) and on R in padded rows (a BED block's layout),
    with the missing values as np.nan, as R's NA (signalling) and as a payload nan."""
    b, n = R.shape
    for bits in (np.nan, R_NA, NEG_PAYLOAD_NAN):
        R1 = _with_nan(R, bits)
        want = _per_snp(R1)
        padded = np.full((b, n + 3), 7.0)[:, :n]
        padded[...] = R1
        for got in (R1.copy(), padded):
            _normalize_c_rows(got)
            bad = [j for j in range(b) if not np.array_equal(_bits(want[j]), _bits(got[j]))]
            assert not bad, "rows %s differ (n=%d, missing %r)" % (bad[:5], n, bits)


@pytest.mark.parametrize("n", NS)
def test_block_standardisation_bit_identical_fuzz(n):
    rng = np.random.default_rng(1000 + n)
    with np.errstate(all="ignore"):  # overflow etc. on the extreme kinds, in both paths
        for kind in KINDS:
            for missing in MISSING:
                _check(_adversarial_block(rng, n, int(rng.integers(1, 25)), kind, missing))


@pytest.mark.parametrize("n", NS_LARGE)
def test_block_standardisation_bit_identical_long_rows(n):
    """Rows longer than numpy's 8192-element reduction buffer."""
    rng = np.random.default_rng(n)
    with np.errstate(all="ignore"):
        for kind in ("bed", "mixed_scales"):
            for missing in ("none", "sparse", "equal_counts"):
                _check(_adversarial_block(rng, n, 3, kind, missing))


@pytest.mark.parametrize("n", [1, 2, 9, 129, 1219, 8193])
def test_probe_accepts_the_block_path(n):
    """The block path is used (not the fallback) on every numpy and platform CI runs."""
    assert genotypes._rows_match_per_snp(n)


def test_probe_rejects_a_different_summation(monkeypatch):
    """A naive vectorisation (nan-aware sums, another summation order) fails the probe,
    and _normalize_rows then falls back to normalize_genotype per SNP."""

    def naive(X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = np.nanmean(X, axis=1, keepdims=True)
            s = np.nanstd(X, axis=1, keepdims=True)
        X[...] = np.where(np.isnan(X), 0.0, (X - m) / np.where(s == 0, 1.0, s))

    monkeypatch.setattr(genotypes, "_normalize_c_rows", naive)
    monkeypatch.setattr(genotypes, "_ROWS_MATCH", {})
    assert not genotypes._rows_match_per_snp(1219)
    rng = np.random.default_rng(3)
    R = _adversarial_block(rng, 1219, 20, "dosage", "sparse")
    got = R.copy()
    _normalize_rows(got)
    assert genotypes._ROWS_MATCH == {1219: False}
    assert np.array_equal(_bits(got), _bits(_per_snp(R)))


def test_normalize_rows_any_layout_and_special_rows():
    """F-ordered and strided input; all-missing rows keep their bits, constant rows are +0."""
    rng = np.random.default_rng(11)
    R = _adversarial_block(rng, 37, 12, "bed", "rows")
    R[0] = np.nan
    R[1] = 0.5
    R[2, :] = -0.0
    want = _per_snp(R)
    fortran = np.asfortranarray(R)
    _normalize_rows(fortran)
    assert fortran.flags.f_contiguous and np.array_equal(_bits(fortran), _bits(want))
    strided = np.zeros((12, 74))[:, ::2]
    strided[...] = R
    _normalize_rows(strided)
    assert np.array_equal(_bits(strided), _bits(want))
    assert np.array_equal(_bits(want[0]), _bits(R[0]))  # all missing: unchanged
    assert np.array_equal(_bits(want[1]), _bits(np.zeros(37)))
    _normalize_rows(np.empty((0, 5)))
    _normalize_rows(np.empty((3, 0)))


def test_no_warnings_on_genotype_data(tmp_path):
    """Missing calls, all-missing and constant SNPs: no RuntimeWarning (as SNP by SNP)."""
    base = str(tmp_path / "s")
    write_synthetic_bed(base, 50, 40, missing=0.05, seed=2)
    raw = np.column_stack([g for g, _ in PlinkReader(base, norm_genotype=False).iter_snps()])
    raw[:, 3] = np.nan
    raw[:, 5] = 1.0
    raw[:-1, 7] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        B = np.hstack([G for G, _ in ArraySource(raw).iter_blocks(16)])
    assert np.array_equal(_bits(B.T), _bits(_per_snp(np.ascontiguousarray(raw.T))))


@pytest.mark.parametrize("fmt", ["b", "b-subset", "t", "emma", "array"])
def test_iter_blocks_bits_and_layout(tmp_path, fmt):
    """Standardised blocks have the bits of iter_snps / normalize_genotype, and the
    memory layout (strides) of the unstandardised blocks, which 6.1c does not touch and
    which is the layout standardised blocks had before (later matrix products might
    round differently on another layout)."""
    rng = np.random.default_rng(5)
    n, m = 23, 41
    raw = rng.integers(0, 3, (n, m)) / 2.0
    raw[rng.random((n, m)) < 0.08] = np.nan
    raw[:, 4] = np.nan
    raw[:, 6] = 0.5
    if fmt == "array":
        blocks = list(ArraySource(raw).iter_blocks(16))
        raw_blocks = list(ArraySource(raw, norm_genotype=False).iter_blocks(16))
        want = _per_snp(np.ascontiguousarray(raw.T)).T
    else:
        if fmt == "emma":
            base = str(tmp_path / "g.emma")
            with open(base, "w") as f:
                for j in range(m):
                    f.write(" ".join("NA" if np.isnan(x) else repr(2 * x) for x in raw[:, j]))
                    f.write("\n")
            kind = "emma"
        elif fmt == "t":
            base = str(tmp_path / "t")
            with open(base + ".tfam", "w") as f:
                f.writelines("F%d I%d 0 0 1 -9\n" % (i, i) for i in range(n))
            allele = {0.0: "1 1", 0.5: "1 2", 1.0: "2 2"}
            with open(base + ".tped", "w") as f, open(base + ".map", "w") as g:
                for j in range(m):
                    calls = ["0 0" if np.isnan(x) else allele[x] for x in raw[:, j]]
                    f.write("1 rs%d 0 %d %s\n" % (j, j + 1, " ".join(calls)))
                    g.write("1 rs%d 0 %d\n" % (j, j + 1))
            kind = "t"
        else:
            base = str(tmp_path / "s")
            write_synthetic_bed(base, n, m, missing=0.1, seed=7)  # n % 4 != 0: padded rows
            kind = "b"

        def reader(norm=True):
            r = PlinkReader(base, type=kind, norm_genotype=norm)
            if fmt == "b-subset":
                r.subset(r.indivs[::-2])
            return r

        want = np.column_stack([g for g, _ in reader().iter_snps()])
        blocks = list(reader().iter_blocks(16))
        raw_blocks = list(reader(norm=False).iter_blocks(16))
    B = np.hstack([G for G, _ in blocks])
    assert np.array_equal(_bits(B), _bits(want))
    for (G, _), (G0, _) in zip(blocks, raw_blocks, strict=True):
        assert G.strides == G0.strides and G.shape == G0.shape


def test_normalize_rows_in_parts_and_chunks(monkeypatch):
    """A block larger than one part (and one chunk) is done part by part, same bits."""
    monkeypatch.setattr(genotypes, "_PART_BYTES", 8 * 37 * 5)  # 5 rows per part
    monkeypatch.setattr(genotypes, "_CHUNK_BYTES", 8 * 37 * 2)  # 2 rows per chunk
    rng = np.random.default_rng(8)
    R = _adversarial_block(rng, 37, 23, "bed", "rows")
    got = R.copy()
    _normalize_rows(got)
    assert np.array_equal(_bits(got), _bits(_per_snp(R)))


@pytest.mark.parametrize("missing", ["none", "sparse", "rows"])
def test_normalize_rows_long_rows_with_missing_calls_one_by_one(monkeypatch, missing):
    """Above _GROUP_MAX_N, SNPs with missing calls go through normalize_genotype and the
    others through the block code: same bits."""
    monkeypatch.setattr(genotypes, "_GROUP_MAX_N", 16)
    rng = np.random.default_rng(9)
    R = _adversarial_block(rng, 41, 30, "dosage", missing)
    got = R.copy()
    _normalize_rows(got)
    assert np.array_equal(_bits(got), _bits(_per_snp(R)))


def test_signalling_nan_missing_values():
    """R's NA_real_ as the missing value (review of #28, S1): [NA, 0, 1, 0.5] was zeroed,
    because np.fmax.reduce does not skip a signalling nan."""
    x = np.array([R_NA, 0.0, 1.0, 0.5])
    X = x[None].copy()
    _normalize_rows(X)
    assert np.array_equal(_bits(X[0]), _bits(normalize_genotype(x.copy())))
    assert X[0, 1] < 0 < X[0, 2]
    rng = np.random.default_rng(1)
    G = rng.integers(0, 3, (200, 400)).astype(float)
    G[rng.random(G.shape) < 0.05] = R_NA
    G[:, 7] = R_NA  # all missing: kept, bits included
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no FP warning from a signalling nan either
        B = np.hstack([g for g, _ in ArraySource(G).iter_blocks(100)])
    want = np.column_stack([normalize_genotype(G[:, j].copy()) for j in range(G.shape[1])])
    assert np.array_equal(_bits(B), _bits(want))
    assert np.array_equal(_bits(B[:, 7]), _bits(G[:, 7]))


def test_probe_covers_small_call_counts(monkeypatch):
    """A numpy whose row reduction differed only for 2-7 values (review of #28, S2): the
    probe rejects it, and _normalize_rows then falls back, with one warning."""

    class _Add:
        def __getattr__(self, name):
            return getattr(np.add, name)

        @staticmethod
        def reduce(a, axis=None, **kw):
            a = np.asarray(a)
            if a.ndim == 2 and axis == 1 and 2 <= a.shape[1] < 8:
                return np.add.reduce(np.ascontiguousarray(a[:, ::-1]), axis=1, **kw)
            return np.add.reduce(a, axis=axis, **kw)

    class _Numpy:
        add = _Add()

        def __getattr__(self, name):
            return getattr(np, name)

    rng = np.random.default_rng(4)
    R = rng.uniform(0, 2, (50, 1219)) * 10.0 ** rng.integers(-3, 4, (50, 1219))
    for j in range(50):
        R[j, rng.choice(1219, 1219 - (2 + j % 6), replace=False)] = np.nan  # 2-7 calls
    monkeypatch.setattr(genotypes, "np", _Numpy())
    monkeypatch.setattr(genotypes, "_ROWS_MATCH", {})
    monkeypatch.setattr(genotypes, "_PROBE_WARNED", False)
    X = R.copy()
    _normalize_c_rows(X)  # the simulated numpy does change the block result ...
    assert not np.array_equal(_bits(X), _bits(_per_snp(R)))
    assert not genotypes._rows_match_per_snp(1219)  # ... and the probe sees it
    calls = []
    monkeypatch.setattr(genotypes.log, "warning", lambda *a: calls.append(a))
    got = R.copy()
    _normalize_rows(got)
    _normalize_rows(got[:0])
    _normalize_rows(R[:, :1000].copy())  # another failing n: still one warning
    assert genotypes._ROWS_MATCH[1219] is False and len(calls) == 1
    assert "SNP by SNP" in calls[0][0]
    assert np.array_equal(_bits(got), _bits(_per_snp(R)))


@pytest.mark.parametrize("n", [1, 7, 130, 1219, 2048, 2049, 5000])
def test_probe_accepts_with_signalling_nans_and_padding(n):
    """The probe (signalling and payload nans, padded rows, the count sweep) passes."""
    assert genotypes._rows_match_per_snp(n)
