"""The summary-statistics output (`pylmm.sumstats`) and the JSON run record of `pylmm gwas`.

Every column is checked on fixtures against something computed independently:
plink2 --glm (an external tool) with K = identity, where the mixed model is ordinary
least squares; a generalised least squares fit on 0/1/2 counts of A1 written here with
numpy; and the legacy columns of the same run.
"""

import contextlib
import gzip
import hashlib
import io
import json
import math
import os
import typing

import helpers as h
import numpy as np
import pytest
from scipy import stats
from synthetic_bed import write_synthetic_bed

from pylmm import cli, sumstats
from pylmm.cli import run_record
from pylmm.genotypes import Variants
from pylmm.kinship import calculate_kinship
from pylmm.lmm import neg_log10_pvalue

PLINK2 = os.path.join(h.REPO, "tests", "data", "plink2")


def _run(argv, rc=0):
    """`pylmm gwas <argv>` in-process; check the exit code; return the log."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        got = cli.main(["gwas", *argv])
    assert got == rc, err.getvalue()
    return err.getvalue()


def _read_table(fn):
    """The table as {column: list of str}, and its header."""
    opener = gzip.open if fn.endswith(".gz") else open
    with opener(fn, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in f]
    assert all(len(r) == len(header) for r in rows)
    return {c: [r[k] for r in rows] for k, c in enumerate(header)}, header


def _num(col):
    return np.array([np.nan if x == "NA" else float(x) for x in col])


def _write_kinship(path, K, ids):
    np.savetxt(path, K, fmt="%.17g")
    with open(path + ".id", "w") as f:
        f.writelines("%s %s\n" % tuple(i) for i in ids)


def _fam_ids(base):
    with open(base + ".fam") as f:
        return [tuple(line.split()[:2]) for line in f]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """80 individuals x 300 SNPs (3% missing calls), a kinship and a phenotype with two causal SNPs."""
    d = tmp_path_factory.mktemp("synth")
    base = str(d / "g")
    G = write_synthetic_bed(base, 80, 300, missing=0.03, seed=5)
    C = 2.0 - 2.0 * G  # copies of A1
    c_imp = np.where(np.isnan(C), np.nanmean(C, axis=0), C)
    K = calculate_kinship(c_imp)
    rng = np.random.default_rng(8)  # null h 0.62 (seed 6: h = 1)
    sd = c_imp.std(axis=0)
    j1, j2 = int(np.argmax(sd[:100])), 150 + int(np.argmax(sd[150:250]))  # common SNPs
    noise = rng.multivariate_normal(np.zeros(80), 0.4 * K + 0.6 * np.eye(80))
    y = 0.9 * c_imp[:, j1] - 0.6 * c_imp[:, j2] + noise
    ids = _fam_ids(base)
    with open(base + ".pheno", "w") as f:
        f.writelines("%s %s %.10f\n" % (a, b, v) for (a, b), v in zip(ids, y))
    _write_kinship(base + ".kin", K, ids)
    out = str(d / "res.tsv.gz")
    _run(["--bfile", base, "--pheno", base + ".pheno", "--kinship", base + ".kin", "-o", out])
    leg = str(d / "res_legacy.tsv")
    _run(
        [
            "--bfile",
            base,
            "--pheno",
            base + ".pheno",
            "--kinship",
            base + ".kin",
            "-o",
            leg,
            "--output-format",
            "legacy",
        ]
    )
    return {
        "base": base,
        "G": G,
        "C": C,
        "K": K,
        "y": y,
        "out": out,
        "legacy": leg,
        "dir": d,
        "causal": (j1, j2),
    }


# ---------------------------------------------------------------------------
# the library pieces
# ---------------------------------------------------------------------------


def test_columns_follow_gwas_ssf():
    # GWAS-SSF v1.1.0: eight mandatory columns, in this order; rsid (not rs_id) and n optional
    assert sumstats.SUMSTATS_COLUMNS[:8] == (
        "chromosome",
        "base_pair_location",
        "effect_allele",
        "other_allele",
        "beta",
        "standard_error",
        "effect_allele_frequency",
        "p_value",
    )
    assert "rsid" in sumstats.SUMSTATS_COLUMNS and "n" in sumstats.SUMSTATS_COLUMNS
    assert len(set(sumstats.SUMSTATS_COLUMNS)) == len(sumstats.SUMSTATS_COLUMNS)


def test_per_allele_effects_sign_and_scale():
    b, se, t = sumstats.per_allele_effects([0.3, -0.2], [0.1, 0.05], [3.0, -4.0], [-0.25, 2.0])
    np.testing.assert_array_equal(b, [-0.075, -0.4])
    np.testing.assert_array_equal(se, [0.025, 0.1])
    np.testing.assert_array_equal(t, [-3.0, -4.0])


def test_allele_count_stats():
    C = np.array(
        [[0.0, 2.0, np.nan], [1.0, np.nan, np.nan], [2.0, 2.0, np.nan], [1.0, 0.0, np.nan]]
    )
    f, sd, miss = sumstats.allele_count_stats(C)
    np.testing.assert_allclose(f[:2], [0.5, 4.0 / 6.0])
    np.testing.assert_allclose(sd[:2], [np.std([0, 1, 2, 1.0]), np.std([2, 2, 0.0])])
    np.testing.assert_allclose(miss, [0.0, 0.25, 1.0])
    assert np.isnan(f[2]) and np.isnan(sd[2])


def test_format_p_values_never_zero():
    nl = np.array([1.0, 299.0, 400.123456789, 1000.0, np.nan, 0.0])
    p = np.array([0.1, 1e-299, 0.0, 0.0, np.nan, 1.0])
    out = sumstats.format_p_values(p, nl)
    assert out[0] == "0.1" and out[1] == "1e-299" and out[4] == "NA" and out[5] == "1.0"
    mant, exp = out[2].split("e")
    assert int(exp) == -401
    np.testing.assert_allclose(-math.log10(float(mant)) - int(exp), 400.123456789, atol=1e-5)
    assert out[3] == "1e-1000"
    # the mantissa never rounds to 10
    assert sumstats.format_p_values([0.0], [400.0000000001]) == ["1e-400"]
    assert sumstats.format_p_values([0.0], [400.9999999999]) == ["1e-401"]
    assert sumstats.format_p_values([0.0], [math.inf]) == ["0.0"]  # an infinite statistic only
    with pytest.raises(ValueError, match="inconsistent"):
        sumstats.format_p_values([1e-310], [np.nan])


def test_lambda_gc():
    p = stats.chi2.sf(stats.chi2.ppf(np.linspace(0.0005, 0.9995, 2000), 1) * 1.2, 1)
    assert sumstats.lambda_gc(np.r_[p, np.nan]) == pytest.approx(1.2, rel=1e-3)
    assert math.isnan(sumstats.lambda_gc([np.nan]))


def _table(m=3):
    v = Variants(
        snp=["rs12", "chr1:5", "SNP_3"][:m],
        chrom=["1", "X", "2"][:m],
        bp=[100, 200, 0][:m],
        a1=["A", "0", "T"][:m],
        a2=["G", "C", "."][:m],
    )
    return sumstats.SumStats.from_variants(
        v,
        beta=[0.5, np.nan, -1e-20][:m],
        standard_error=[0.25, np.nan, 3.0][:m],
        t_stat=[2.0, np.nan, -1.0][:m],
        p_value=[0.05, np.nan, 0.0][:m],
        neg_log_10_p_value=[1.3, np.nan, 500.0][:m],
        effect_allele_frequency=[0.1, 0.0, 1.0][:m],
        n=[10, np.nan, 9][:m],
        missing_rate=[0.0, 0.5, 0.1][:m],
        beta_std=[0.2, np.nan, 0.0][:m],
    )


def test_writer_text_na_and_rsid(tmp_path):
    fn = str(tmp_path / "t.tsv")
    sumstats.write_sumstats(fn, _table())
    cols, header = _read_table(fn)
    assert tuple(header) == sumstats.SUMSTATS_COLUMNS
    assert cols["rsid"] == ["rs12", "NA", "NA"]
    assert cols["variant_id"] == ["rs12", "chr1:5", "SNP_3"]
    assert cols["chromosome"] == ["1", "X", "2"]
    assert cols["base_pair_location"] == ["100", "200", "NA"]  # 0 is not a position
    assert cols["effect_allele"] == ["A", "NA", "T"]  # PLINK's 0 = no allele
    assert cols["other_allele"] == ["G", "C", "NA"]
    assert cols["n"] == ["10", "NA", "9"]
    assert cols["beta"] == ["0.5", "NA", "-1e-20"]
    assert cols["p_value"] == ["0.05", "NA", "1e-500"]


def test_writer_floats_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=500) * 10.0 ** rng.integers(-30, 30, size=500)
    v = Variants.from_names(["s%d" % k for k in range(500)])
    t = sumstats.SumStats.from_variants(
        v,
        beta=x,
        standard_error=np.abs(x),
        t_stat=x,
        p_value=rng.uniform(size=500),
        effect_allele_frequency=rng.uniform(size=500),
        n=np.full(500, 7),
        missing_rate=x * 0,
        beta_std=x,
    )
    fn = str(tmp_path / "t.tsv.gz")
    sumstats.write_sumstats(fn, t)
    cols, _ = _read_table(fn)
    np.testing.assert_array_equal(_num(cols["beta"]), x)  # repr: lossless
    np.testing.assert_array_equal(_num(cols["p_value"]), t.p_value)


def test_gzip_output_is_reproducible(tmp_path):
    a, b = str(tmp_path / "a.tsv.gz"), str(tmp_path / "b.tsv.gz")
    sumstats.write_sumstats(a, _table())
    sumstats.write_sumstats(b, _table())
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read() == fb.read()
    with open(a, "rb") as f:
        assert f.read(2) == b"\x1f\x8b"


def test_pandas_reads_the_table(tmp_path):
    pd = pytest.importorskip("pandas")
    fn = str(tmp_path / "t.tsv.gz")
    sumstats.write_sumstats(fn, _table())
    df = pd.read_csv(fn, sep="\t")
    assert list(df.columns) == list(sumstats.SUMSTATS_COLUMNS)
    assert df["beta"].dtype == np.float64 and np.isnan(df["beta"][1])
    assert df["rsid"].isna().tolist() == [False, True, True]
    assert df["p_value"][2] == 0.0  # 1e-500 underflows in a double ...
    assert df["neg_log_10_p_value"][2] == 500.0  # ... its -log10 does not


def test_sidecar_path():
    assert run_record.sidecar_path("a/b.tsv.gz") == "a/b.json"
    assert run_record.sidecar_path("b.txt") == "b.json"
    assert run_record.sidecar_path("b.tsv") == "b.json"
    assert run_record.sidecar_path("b") == "b.json"
    assert run_record.sidecar_path("b.gz") == "b.json"


# ---------------------------------------------------------------------------
# pylmm gwas against plink2 --glm (K = identity: the LMM is OLS)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plink2_glm():
    rows = {}
    with open(os.path.join(PLINK2, "geno_alt.y.glm.linear")) as f:
        hdr = f.readline().lstrip("#").split()
        for line in f:
            v = dict(zip(hdr, line.split()))
            rows[v["ID"]] = v
    return rows


@pytest.mark.parametrize("remove_missing", [False, True])
def test_matches_plink2_glm_with_identity_kinship(tmp_path, plink2_glm, remove_missing):
    base = os.path.join(PLINK2, "geno")
    ids = _fam_ids(base)
    kin = str(tmp_path / "I.kin")
    _write_kinship(kin, np.eye(len(ids)), ids)
    out = str(tmp_path / "r.tsv.gz")
    argv = ["--bfile", base, "--pheno", base + ".pheno", "--kinship", kin, "-o", out]
    _run(argv + (["--remove-missing-genotypes"] if remove_missing else []))
    cols, _ = _read_table(out)
    q = [plink2_glm[s] for s in cols["variant_id"]]
    assert len(q) == 300
    # plink2 --glm omit-ref: A1 = ALT = the .bim's column 5, pylmm's effect allele
    assert cols["effect_allele"] == [r["A1"] for r in q]
    assert cols["other_allele"] == [r["REF"] for r in q]
    assert cols["chromosome"] == [r["CHROM"] for r in q]
    assert cols["base_pair_location"] == [r["POS"] for r in q]
    complete = np.array([r["OBS_CT"] == str(len(ids)) for r in q])
    assert 100 < complete.sum() < 300  # both kinds of SNP are checked
    # plink2 drops the missing calls of each SNP; pylmm does with --remove-missing-genotypes.
    # With mean imputation the slope is the same (the imputed values sit at the mean), but
    # n, the SE, t and p are not, for SNPs with missing calls.
    both = np.ones_like(complete) if remove_missing else complete
    for ours, theirs, sel in (
        ("beta", "BETA", np.ones_like(complete)),
        ("standard_error", "SE", both),
        ("t_stat", "T_STAT", both),
        ("p_value", "P", both),
        ("effect_allele_frequency", "A1_FREQ", np.ones_like(complete)),
        ("n", "OBS_CT", both),
    ):
        np.testing.assert_allclose(
            _num(cols[ours])[sel],
            np.array([float(r[theirs]) for r in q])[sel],
            rtol=h.PLINK2_GLM_RTOL,
            atol=0,
            err_msg=ours,
        )
    if not remove_missing:
        np.testing.assert_array_equal(_num(cols["n"]), len(ids))
    miss = 1.0 - np.array([float(r["OBS_CT"]) for r in q]) / len(ids)
    np.testing.assert_allclose(_num(cols["missing_rate"]), miss, rtol=0, atol=1e-15)


# ---------------------------------------------------------------------------
# pylmm gwas against a GLS fit on A1 counts, and against its legacy columns
# ---------------------------------------------------------------------------


def _record(out):
    with open(run_record.sidecar_path(out)) as f:
        return json.load(f)


def test_beta_is_the_gls_effect_per_copy_of_a1(synth):
    cols, _ = _read_table(synth["out"])
    rec = _record(synth["out"])
    hh = rec["null_fit"]["h"]
    assert 0.05 < hh < 0.95  # inside: V is well conditioned
    K, y, C = synth["K"], synth["y"], synth["C"]
    n = len(y)
    v_inv = np.linalg.inv(hh * K + (1 - hh) * np.eye(n))
    beta, se, t = [], [], []
    for j in range(C.shape[1]):
        c = np.where(np.isnan(C[:, j]), np.nanmean(C[:, j]), C[:, j])  # mean imputation
        if np.ptp(c) == 0:
            beta.append(np.nan), se.append(np.nan), t.append(np.nan)
            continue
        X = np.column_stack([np.ones(n), c])
        A = np.linalg.inv(X.T @ v_inv @ X)
        b = A @ X.T @ v_inv @ y
        r = y - X @ b
        s2 = r @ v_inv @ r / (n - 2)  # REML sigma2: Q / (n - p)
        (
            beta.append(b[1]),
            se.append(math.sqrt(s2 * A[1, 1])),
            t.append(b[1] / math.sqrt(s2 * A[1, 1])),
        )
    for ours, ref in (("beta", beta), ("standard_error", se), ("t_stat", t)):
        h.assert_close(
            _num(cols[ours]),
            np.array(ref),
            rtol=h.SUMSTATS_GLS_RTOL,
            atol=h.SUMSTATS_GLS_RTOL * np.nanmax(np.abs(ref)),
            what=ours,
        )
    # beta_std: beta times the SD of the A1 count over the called genotypes
    np.testing.assert_allclose(
        _num(cols["beta_std"]),
        np.array(beta) * np.nanstd(C, axis=0),
        rtol=h.SUMSTATS_GLS_RTOL,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _num(cols["effect_allele_frequency"]), np.nanmean(C, axis=0) / 2, rtol=1e-14
    )
    np.testing.assert_allclose(_num(cols["missing_rate"]), np.isnan(C).mean(axis=0), rtol=1e-14)
    # the causal SNPs: right sign, near the simulated effect per copy of A1
    b = _num(cols["beta"])
    j1, j2 = synth["causal"]
    assert 0.5 < b[j1] < 1.3 and -1.0 < b[j2] < -0.2


def test_new_columns_against_legacy_columns(synth):
    cols, _ = _read_table(synth["out"])
    ids, leg = h.load_gwas_out(synth["legacy"])
    assert cols["variant_id"] == ids
    # same test: t flips sign with the allele (A2 -> A1), p is unchanged (bit for bit)
    np.testing.assert_array_equal(_num(cols["t_stat"]), -leg[:, 2])
    np.testing.assert_array_equal(_num(cols["p_value"]), leg[:, 3])
    # BETA is per SD of the A2 fraction over the genotyped individuals: beta = -BETA / (2 sd)
    sd = np.nanstd(synth["G"], axis=0)
    np.testing.assert_allclose(_num(cols["beta"]), -leg[:, 0] / (2 * sd), rtol=1e-13)
    np.testing.assert_allclose(_num(cols["standard_error"]), leg[:, 1] / (2 * sd), rtol=1e-13)
    # -log10 p: from p where it is representable
    np.testing.assert_allclose(_num(cols["neg_log_10_p_value"]), -np.log10(leg[:, 3]), rtol=1e-15)


def test_remove_missing_genotypes_one_scale_for_all_snps(synth, tmp_path):
    # 6.3d: the legacy BETA is per 0-1 step of the A2 fraction for complete SNPs and per SD
    # for SNPs with missing calls; the new beta is per copy of A1 for both
    base = synth["base"]
    out = str(tmp_path / "rmg.tsv.gz")
    leg = str(tmp_path / "rmg.tsv")
    argv = [
        "--bfile",
        base,
        "--pheno",
        base + ".pheno",
        "--kinship",
        base + ".kin",
        "--remove-missing-genotypes",
    ]
    _run([*argv, "-o", out])
    _run([*argv, "-o", leg, "--output-format", "legacy"])
    cols, _ = _read_table(out)
    _, L = h.load_gwas_out(leg)
    G = synth["G"]
    complete = ~np.isnan(G).any(axis=0)
    scale = np.where(complete, 1.0, np.nanstd(G, axis=0))
    np.testing.assert_allclose(_num(cols["beta"]), -L[:, 0] / (2 * scale), rtol=1e-13)
    np.testing.assert_array_equal(_num(cols["n"]), (~np.isnan(G)).sum(axis=0))
    # and per copy of A1 it is the GLS effect on the called individuals (null h of each subset
    # differs a little, so only compare against the imputed run loosely: same sign, close)
    b0 = _num(_read_table(synth["out"])[0]["beta"])
    ok = np.isfinite(b0)
    assert np.corrcoef(b0[ok], _num(cols["beta"])[ok])[0, 1] > 0.99


def test_p_value_below_double_range(tmp_path):
    # a SNP that explains the phenotype almost exactly: p underflows to 0
    base = str(tmp_path / "g")
    G = write_synthetic_bed(base, 60, 20, missing=0.0, seed=2)
    C = 2.0 - 2.0 * G
    ids = _fam_ids(base)
    y = C[:, 3] + 1e-9 * np.random.default_rng(0).normal(size=60)
    with open(base + ".pheno", "w") as f:
        f.writelines("%s %s %.17g\n" % (a, b, v) for (a, b), v in zip(ids, y))
    _write_kinship(base + ".kin", calculate_kinship(C), ids)
    out = str(tmp_path / "r.tsv")
    leg = str(tmp_path / "r_legacy.tsv")
    argv = ["--bfile", base, "--pheno", base + ".pheno", "--kinship", base + ".kin"]
    _run([*argv, "-o", out])
    _run([*argv, "-o", leg, "--output-format", "legacy"])
    cols, _ = _read_table(out)
    _, L = h.load_gwas_out(leg)
    # the legacy P_VALUE underflows: to 0, or to a subnormal (scipy 1.9's t.sf)
    assert L[3, 3] < 1e-300
    text = cols["p_value"][3]
    assert text != "0.0" and "e-" in text
    nl = _num(cols["neg_log_10_p_value"])[3]
    assert nl > 320
    t = _num(cols["t_stat"])[3]
    assert nl == pytest.approx(float(neg_log10_pvalue(t, 60 - 2)), rel=1e-12)
    mant, exp = text.split("e")
    assert -math.log10(float(mant)) - int(exp) == pytest.approx(nl, abs=1e-5)


# ---------------------------------------------------------------------------
# EMMA input
# ---------------------------------------------------------------------------


def test_emma_with_and_without_map(tmp_path):
    emma = h.write_emma_subset(tmp_path, 300)
    K = h.write_headerless_kinship(tmp_path)
    mapfile = os.path.join(h.DATA, "hmdp.liver.snps.map.tsv")
    with open(mapfile) as f:
        f.readline()
        mp = [f.readline().split() for _ in range(300)]
    head = str(tmp_path / "map300.tsv")
    with open(mapfile) as src, open(head, "w") as dst:
        dst.writelines(src.readline() for _ in range(301))
    out_map, out_plain, leg = (str(tmp_path / x) for x in ("m.tsv.gz", "p.tsv.gz", "l.tsv"))
    argv = ["--emma-geno", emma, "--emma-pheno", h.HMDP_Y, "--kinship", K]
    _run([*argv, "--emma-map", head, "-o", out_map])
    _run([*argv, "-o", out_plain])
    _run([*argv, "-o", leg, "--output-format", "legacy"])
    cm, _ = _read_table(out_map)
    cp, _ = _read_table(out_plain)
    _, L = h.load_gwas_out(leg)
    # the map fills the position and alleles; the effect allele is the allele coded 1
    assert cm["chromosome"] == [r[1] for r in mp]
    assert cm["base_pair_location"] == [r[2] for r in mp]
    assert cm["effect_allele"] == [("NA" if r[4] == "." else r[4]) for r in mp]
    assert cm["other_allele"] == [("NA" if r[3] == "." else r[3]) for r in mp]
    assert cm["variant_id"] == ["SNP_%d" % (k + 1) for k in range(300)]
    for c in ("chromosome", "base_pair_location", "effect_allele", "other_allele"):
        assert set(cp[c]) == {"NA"}
    for c in ("beta", "t_stat", "p_value", "effect_allele_frequency", "n"):
        assert cm[c] == cp[c]
    # the EMMA value counts the effect allele: t keeps its sign; beta per copy = BETA / (2 sd)
    np.testing.assert_array_equal(_num(cm["t_stat"]), L[:, 2])
    X = h.load_hmdp_snps(300)  # individuals x SNPs, 0/1 (nan missing)
    np.testing.assert_allclose(_num(cm["beta"]), L[:, 0] / (2 * np.nanstd(X, axis=0)), rtol=1e-12)
    np.testing.assert_allclose(
        _num(cm["effect_allele_frequency"]), np.nanmean(X, axis=0), rtol=1e-14
    )


# ---------------------------------------------------------------------------
# the run record
# ---------------------------------------------------------------------------


def _check_typed(obj, cls):
    keys = typing.get_type_hints(cls)
    assert set(obj) == set(keys), (cls.__name__, set(obj) ^ set(keys))


def test_run_record(synth):
    rec = _record(synth["out"])
    assert rec["schema"] == run_record.RUN_RECORD_SCHEMA and rec["status"] == "ok"
    assert rec["error"] is None
    assert set(rec) == {
        "schema",
        "pylmm_version",
        "status",
        "error",
        "command",
        "started_at",
        "finished_at",
        "environment",
        "inputs",
        "samples",
        "model",
        "kinship",
        "null_fit",
        "loco",
        "results",
        "timings_s",
        "warnings",
    }
    _check_typed(rec["environment"], run_record.EnvironmentInfo)
    _check_typed(rec["samples"], run_record.SampleCounts)
    _check_typed(rec["model"], run_record.ModelInfo)
    _check_typed(rec["kinship"], run_record.KinshipInfo)
    _check_typed(rec["null_fit"], run_record.NullFit)
    _check_typed(rec["results"], run_record.ResultsInfo)
    for info in rec["inputs"].values():
        _check_typed(info, run_record.FileInfo)
    assert set(rec["inputs"]) == {"bed", "bim", "fam", "pheno", "kinship", "kinship_ids"}
    with open(synth["base"] + ".bed", "rb") as f:
        assert rec["inputs"]["bed"]["sha256"] == hashlib.sha256(f.read()).hexdigest()
    assert rec["command"][:2] == ["pylmm", "gwas"]
    assert rec["samples"]["n_analysed"] == 80 and rec["samples"]["n_genotyped"] == 80
    assert rec["model"]["test"] == "wald" and rec["model"]["reml"] is True
    assert rec["model"]["q"] == 1
    nf = rec["null_fit"]
    assert 0 <= nf["h"] <= 1
    assert nf["sigma_g2"] + nf["sigma_e2"] == pytest.approx(nf["sigma2"])
    res = rec["results"]
    cols, _ = _read_table(synth["out"])
    p = _num(cols["p_value"])
    assert res["n_variants"] == 300 and res["n_tested"] == int(np.isfinite(p).sum())
    assert res["n_tested"] + sum(res["not_tested"].values()) == 300
    assert res["lambda_gc"] == pytest.approx(sumstats.lambda_gc(p), rel=1e-12)
    assert res["format"] == sumstats.SUMSTATS_FORMAT
    assert res["columns"] == list(sumstats.SUMSTATS_COLUMNS)
    assert rec["kinship"]["has_ids"] is True
    assert rec["kinship"]["mean_diag"] == pytest.approx(np.trace(synth["K"]) / 80)
    assert rec["timings_s"]["total"] >= rec["timings_s"]["scan"] >= 0
    assert rec["loco"] is None


def test_run_record_counts_and_warnings(tmp_path, synth):
    base = synth["base"]
    ids = _fam_ids(base)
    # 3 missing phenotypes, 2 individuals not in the kinship, a covariate missing for 1
    with open(base + ".pheno") as f:
        lines = f.read().splitlines()
    for k in (0, 5, 9):
        v = lines[k].split()
        lines[k] = " ".join([*v[:2], "NA"])
    ph = str(tmp_path / "ph")
    with open(ph, "w") as f:
        f.write("\n".join(lines) + "\n")
    cov = str(tmp_path / "cov")
    with open(cov, "w") as f:
        for k, (a, b) in enumerate(ids):
            f.write("%s %s %s\n" % (a, b, "NA" if k == 20 else str(k % 7)))
    kin = str(tmp_path / "k.kin")
    _write_kinship(kin, synth["K"][2:, 2:], ids[2:])
    out = str(tmp_path / "r.tsv.gz")
    _run(["--bfile", base, "--pheno", ph, "--covar", cov, "--kinship", kin, "-o", out])
    rec = _record(out)
    s = rec["samples"]
    assert s["n_genotyped"] == 80
    assert s["dropped"] == {"not_in_kinship": 2, "missing_phenotype": 2, "missing_covariate": 1}
    assert s["n_analysed"] == 75
    assert rec["model"]["q"] == 2
    assert any("not in" in w or "dropped" in w for w in rec["warnings"]), rec["warnings"]
    assert rec["inputs"]["covar"]["path"] == "cov"  # the base name only (privacy)
    assert rec["model"]["covariates"] == ["column 1"]  # no header, no --covar-name


def test_run_record_on_error(tmp_path, synth):
    base = synth["base"]
    out = str(tmp_path / "r.tsv.gz")
    _run(
        [
            "--bfile",
            base,
            "--pheno",
            base + ".pheno",
            "--kinship",
            str(tmp_path / "nope"),
            "-o",
            out,
        ],
        rc=2,
    )
    rec = _record(out)
    assert rec["status"] == "error" and "nope" in rec["error"]
    assert not os.path.exists(out)


def test_cli_output_is_reproducible(tmp_path, synth):
    base = synth["base"]
    argv = ["--bfile", base, "--pheno", base + ".pheno", "--kinship", base + ".kin"]
    a, b = str(tmp_path / "a.tsv.gz"), str(tmp_path / "b.tsv.gz")
    _run([*argv, "-o", a])
    _run([*argv, "-o", b])
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read() == fb.read()


def test_loco_writes_the_new_format(tmp_path, synth):
    base = synth["base"]
    out = str(tmp_path / "loco.tsv.gz")
    leg = str(tmp_path / "loco.tsv")
    _run(["--bfile", base, "--pheno", base + ".pheno", "--loco", "-o", out])
    _run(
        [
            "--bfile",
            base,
            "--pheno",
            base + ".pheno",
            "--loco",
            "-o",
            leg,
            "--output-format",
            "legacy",
        ]
    )
    cols, header = _read_table(out)
    assert tuple(header) == sumstats.SUMSTATS_COLUMNS
    _, L = h.load_gwas_out(leg)
    np.testing.assert_array_equal(_num(cols["t_stat"]), -L[:, 2])
    assert cols["chromosome"][0] == "1" and cols["chromosome"][-1] == "3"
    rec = _record(out)
    assert set(rec["loco"]) == {"1", "2", "3"}
    for fit in rec["loco"].values():
        _check_typed(fit, run_record.LocoChromosome)
    assert rec["null_fit"] is None and rec["kinship"]["source"] == "loco"


@pytest.mark.parametrize("coding", ["fraction", "count"])
def test_emma_coding(tmp_path, coding):
    # EMMA values as 0/1/2 counts (count) or the same halved (fraction); K = identity, so
    # beta per copy of the allele coded 1 is the OLS slope on the count
    rng = np.random.default_rng(3)
    n = 40
    C = rng.binomial(2, 0.4, size=(n, 6)).astype(float)
    y = 0.7 * C[:, 2] + rng.normal(size=n)
    geno = str(tmp_path / "g.emma")
    np.savetxt(geno, (C if coding == "count" else C / 2).T)
    ph = str(tmp_path / "p.emma")
    np.savetxt(ph, y[None, :])
    kin = str(tmp_path / "I.txt")
    np.savetxt(kin, np.eye(n))
    out = str(tmp_path / "r.tsv")
    argv = ["--emma-geno", geno, "--emma-pheno", ph, "--kinship", kin, "-o", out]
    log = _run([*argv, "--emma-coding", coding])
    assert "outside" not in log
    cols, _ = _read_table(out)
    slopes = [np.polyfit(C[:, j], y, 1)[0] for j in range(6)]
    np.testing.assert_allclose(_num(cols["beta"]), slopes, rtol=1e-10)
    np.testing.assert_allclose(_num(cols["effect_allele_frequency"]), C.mean(axis=0) / 2)
    if coding == "count":  # the default (fraction) on counts: an input error, exit 2
        log = _run(argv, rc=2)
        assert "outside 0..1" in log and "--emma-coding count" in log
        _run([*argv, "--output-format", "legacy"])  # no per-allele columns: fine


def test_emma_coding_needs_emma_input(synth, tmp_path):
    base = synth["base"]
    _run(
        [
            "--bfile",
            base,
            "--pheno",
            base + ".pheno",
            "--kinship",
            base + ".kin",
            "--emma-coding",
            "count",
            "-o",
            str(tmp_path / "r.tsv"),
        ],
        rc=2,
    )


def test_write_failure_is_an_error_and_keeps_the_lines_before(synth, tmp_path, monkeypatch):
    calls = [0]
    orig = sumstats.SumstatsWriter.write

    def fail_second(self, table):
        calls[0] += 1
        if calls[0] == 2:
            raise OSError("No space left on device")
        orig(self, table)

    monkeypatch.setattr(sumstats.SumstatsWriter, "write", fail_second)
    base = synth["base"]
    out = str(tmp_path / "r.tsv.gz")
    log = _run(
        [
            "--bfile",
            base,
            "--pheno",
            base + ".pheno",
            "--kinship",
            base + ".kin",
            "--block-size",
            "100",
            "-o",
            out,
        ],
        rc=2,
    )
    assert "No space left" in log
    cols, _ = _read_table(out)
    assert len(cols["variant_id"]) == 100  # the first block, complete; the gzip stream closed
    assert _record(out)["status"] == "error"


def test_library_table_matches_the_cli(synth, tmp_path):
    # gwas(return_details=True) on the raw A2 fractions -> SumStats.from_genotypes: the table of
    # `pylmm gwas --remove-missing-genotypes` (the library tests SNPs with missing calls on
    # their called individuals, as that option does)
    from pylmm.lmm import gwas
    from pylmm.plink_io import PlinkReader

    base = synth["base"]
    with PlinkReader(base, norm_genotype=False, pheno_file=base + ".pheno") as r:
        G = np.hstack([g for g, _ in r.iter_blocks(1000)])
        variants = r.variants
        y = r.phenos[:, 0]  # as the CLI reads it (written with 10 decimals)
    res = gwas(y, G, synth["K"], return_details=True)
    table = sumstats.SumStats.from_genotypes(res, G, variants, coding="a2_fraction")
    out = str(tmp_path / "lib.tsv.gz")
    sumstats.write_sumstats(out, table)
    cli = str(tmp_path / "cli.tsv.gz")
    _run(
        [
            "--bfile",
            base,
            "--pheno",
            base + ".pheno",
            "--kinship",
            base + ".kin",
            "--remove-missing-genotypes",
            "-o",
            cli,
        ]
    )
    a, b = sumstats.read_sumstats(out), sumstats.read_sumstats(cli)
    for c in (
        "chromosome",
        "base_pair_location",
        "effect_allele",
        "other_allele",
        "variant_id",
        "n",
    ):
        np.testing.assert_array_equal(a[c], b[c], err_msg=c)
    for c in ("effect_allele_frequency", "missing_rate"):  # the same mean, other roundings
        np.testing.assert_allclose(a[c], b[c], rtol=1e-14, err_msg=c)
    for c in ("beta", "standard_error", "t_stat", "p_value", "beta_std", "neg_log_10_p_value"):
        h.assert_close(a[c], b[c], rtol=h.SUMSTATS_LIB_CLI_RTOL, atol=0, what=c)
    # and with 0/1/2 counts of A1 (e.g. an ALT dosage)
    C = 2.0 - 2.0 * G
    res_c = gwas(y, C, synth["K"], return_details=True)
    t_c = sumstats.SumStats.from_genotypes(res_c, C, variants, coding="a1_count")
    h.assert_close(t_c.beta, a["beta"], rtol=h.SUMSTATS_LIB_CLI_RTOL, atol=0, what="a1_count")
    with pytest.raises(ValueError, match=r"outside 0\.\.1"):
        sumstats.SumStats.from_genotypes(res, (G - 0.5) * 3, variants, coding="a2_fraction")


def test_from_genotypes_refuses_a_scan_of_other_genotypes(synth):
    # the reviewer's trap: a scan of standardised genotypes (PlinkReader's default) converted
    # with the raw ones gave beta 0.103 instead of 0.346, silently
    from pylmm.lmm import gwas
    from pylmm.plink_io import PlinkReader

    base = synth["base"]
    with PlinkReader(base, norm_genotype=False, pheno_file=base + ".pheno") as r:
        G = np.hstack([g for g, _ in r.iter_blocks(1000)])[:, :20]
        y = r.phenos[:, 0]
    with PlinkReader(base, pheno_file=base + ".pheno") as r:
        Z = np.hstack([g for g, _ in r.iter_blocks(1000)])[:, :20]  # standardised
    res_z = gwas(y, Z, synth["K"], return_details=True)
    with pytest.raises(ValueError, match="not the genotype matrix the scan tested"):
        sumstats.SumStats.from_genotypes(res_z, G, coding="a2_fraction")
    res_g = gwas(y, G, synth["K"], return_details=True)
    ok = sumstats.SumStats.from_genotypes(res_g, G, coding="a2_fraction")
    assert np.isfinite(ok.beta).sum() > 15
    # a scan that dropped individuals (missing phenotype) needs Y
    y2 = y.copy()
    y2[:5] = np.nan
    res2 = gwas(y2, G, synth["K"], return_details=True)
    with pytest.raises(ValueError, match="and Y if"):
        sumstats.SumStats.from_genotypes(res2, G, coding="a2_fraction")
    t2 = sumstats.SumStats.from_genotypes(res2, G, coding="a2_fraction", Y=y2)
    np.testing.assert_array_equal(t2.n[np.isfinite(t2.p_value)] <= 75, True)
    with pytest.raises(TypeError):
        sumstats.SumStats.from_genotypes(res_g, G)  # coding is required


# ---------------------------------------------------------------------------
# review round 1 (PR #32): the run record's robustness, location and privacy; LRT
# ---------------------------------------------------------------------------


def test_record_with_non_finite_numbers_is_valid_json(synth, tmp_path):
    # the reviewer's case: a phenotype scaled by 1e300 makes the null log-likelihood nan
    base = synth["base"]
    ids = _fam_ids(base)
    ph = str(tmp_path / "huge.pheno")
    with open(ph, "w") as f:
        f.writelines("%s %s %.17g\n" % (a, b, v * 1e300) for (a, b), v in zip(ids, synth["y"]))
    out = str(tmp_path / "huge.tsv.gz")
    _run(["--bfile", base, "--pheno", ph, "--kinship", base + ".kin", "-o", out])  # exit 0
    with open(run_record.sidecar_path(out)) as f:
        text = f.read()
    rec = json.loads(text)  # valid JSON: no NaN or Infinity tokens
    assert "NaN" not in text and "Infinity" not in text
    assert rec["status"] == "ok"
    assert all(
        v is None or math.isfinite(v)
        for v in rec["null_fit"].values()
        if not isinstance(v, (bool, str))
    )


def test_record_write_failure_does_not_fail_the_run(synth, tmp_path, monkeypatch):
    def broken(path, record):
        raise ValueError("boom")

    monkeypatch.setattr(run_record, "write_record", broken)
    base = synth["base"]
    out = str(tmp_path / "r.tsv.gz")
    log = _run(["--bfile", base, "--pheno", base + ".pheno", "--kinship", base + ".kin", "-o", out])
    assert "cannot write the run record" in log and os.path.exists(out)
    assert not os.path.exists(run_record.sidecar_path(out))


def test_record_is_written_atomically(tmp_path):
    path = str(tmp_path / "r.json")
    run_record.write_record(path, {"a": 1})
    with pytest.raises(TypeError):
        run_record.write_record(path, {"a": object()})  # not serialisable
    with open(path) as f:
        assert json.load(f) == {"a": 1}  # the old file, intact
    assert os.listdir(str(tmp_path)) == ["r.json"]  # no temporary file left


def test_record_location_options(synth, tmp_path):
    base = synth["base"]
    argv = ["--bfile", base, "--pheno", base + ".pheno", "--kinship", base + ".kin"]
    out = str(tmp_path / "a.tsv.gz")
    other = str(tmp_path / "elsewhere" / "run.json")
    os.makedirs(os.path.dirname(other))
    _run([*argv, "-o", out, "--run-record", other])
    assert os.path.exists(other) and not os.path.exists(str(tmp_path / "a.json"))
    _run([*argv, "-o", out, "--no-run-record"])
    assert not os.path.exists(str(tmp_path / "a.json"))
    _run([*argv, "-o", out])
    log = _run([*argv, "-o", str(tmp_path / "a.tsv")])  # the same record name: a.json
    assert "overwriting the existing run record" in log
    _run([*argv, "-o", out, "--run-record", "x.json", "--no-run-record"], rc=2)


def test_stdout_output_writes_no_record(synth, tmp_path):
    import subprocess
    import sys

    base = synth["base"]
    argv = ["--bfile", base, "--pheno", base + ".pheno", "--kinship", base + ".kin"]
    for fmt, first in (("legacy", "SNP_ID"), ("sumstats", "chromosome")):
        p = subprocess.run(
            [
                sys.executable,
                "-m",
                "pylmm",
                "gwas",
                *argv,
                "--output-format",
                fmt,
                "-o",
                "/dev/stdout",
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
            env={**os.environ, "PYTHONPATH": h.REPO},
        )
        assert p.returncode == 0, p.stderr
        assert p.stdout.startswith(first) and len(p.stdout.splitlines()) == 301
    assert os.listdir(str(tmp_path)) == []  # no record anywhere


def test_record_paths_are_base_names_unless_asked(synth, tmp_path):
    base = synth["base"]
    argv = ["--bfile", base, "--pheno", base + ".pheno", "--kinship", base + ".kin"]
    out = str(tmp_path / "r.tsv.gz")
    _run([*argv, "-o", out])
    text = open(run_record.sidecar_path(out)).read()
    assert str(tmp_path) not in text and os.path.dirname(base) not in text
    rec = json.loads(text)
    assert rec["inputs"]["bed"]["path"] == "g.bed" and rec["results"]["path"] == "r.tsv.gz"
    assert rec["command"][rec["command"].index("--bfile") + 1] == "g"
    _run([*argv, "-o", out, "--run-record-full-paths"])
    rec = _record(out)
    assert rec["inputs"]["bed"]["path"] == base + ".bed"


def test_record_says_when_the_effect_allele_is_unknown(tmp_path):
    tped = h.write_tped_subset(tmp_path, 30)  # .map only: no allele names
    out = str(tmp_path / "t.tsv.gz")
    _run(["--tfile", tped, "--kinship", h.README_KIN, "--pheno", h.FAKE_PHENOS, "-o", out])
    cols, _ = _read_table(out)
    assert set(cols["effect_allele"]) == {"NA"}
    assert _record(out)["results"]["effect_allele"].startswith("unknown")


def test_lrt_table_record_and_library(synth, tmp_path):
    from pylmm.lmm import gwas, neg_log10_p_for

    base = synth["base"]
    out = str(tmp_path / "lrt.tsv.gz")
    _run(
        [
            "--bfile",
            base,
            "--pheno",
            base + ".pheno",
            "--kinship",
            base + ".kin",
            "--test",
            "lrt",
            "-o",
            out,
        ]
    )
    rec = _record(out)
    nf = rec["null_fit"]
    assert nf["kind"] == "lrt-reduced-ml" and nf["reml"] is False and rec["model"]["reml"] is False
    assert nf["h"] is not None and nf["log_likelihood"] is not None
    assert isinstance(nf["dropped_null_directions"], int)
    assert "lrt_unbounded" in rec["results"]["not_tested"]
    res = sumstats.read_sumstats(out)
    ok = np.isfinite(res["p_value"])
    # t_stat is the signed root of the chi2(1) statistic, the sign of beta
    chi2 = stats.chi2.isf(res["p_value"][ok], 1)
    np.testing.assert_allclose(res["t_stat"][ok] ** 2, chi2, rtol=1e-6)
    assert np.all(np.sign(res["t_stat"][ok]) == np.sign(res["beta"][ok]))
    # library: -log10 p of the LRT from the chi2(1) tail where p underflows
    nl = neg_log10_p_for([45.0], [0.0], [100.0], test="lrt")[0]
    assert nl == pytest.approx(-stats.norm.logsf(45.0) / math.log(10) - math.log10(2), rel=1e-12)
    r = gwas(synth["y"], synth["C"], synth["K"], test="lrt", return_details=True)
    assert r.test == "lrt"
    fin = np.isfinite(r.p)
    np.testing.assert_allclose(r.neg_log10_p[fin], -np.log10(r.p[fin]), rtol=1e-12)


def test_neg_log10_p_is_never_negative_zero():
    from pylmm.lmm import neg_log10_p_for

    v = neg_log10_p_for([0.0], [1.0], [10.0])
    assert v[0] == 0.0 and math.copysign(1.0, v[0]) == 1.0
