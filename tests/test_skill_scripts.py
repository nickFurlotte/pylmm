"""Tests of the helper scripts shipped with the agent skills (skills/*/scripts/).

The docs test (tests/test_docs.py) runs them on the happy path; these tests
pin their error handling and edge cases.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from pylmm.plink_io import PlinkReader

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "skills", "convert-genotype-formats", "scripts"))
import plink_tools  # noqa: E402


def _read_raw(prefix: str) -> np.ndarray:
    with PlinkReader(prefix, norm_genotype=False) as r:
        return np.column_stack([g for g, _ in r])


# -- plink_tools.write_bed ------------------------------------------------------


def test_write_bed_counts_round_trip(tmp_path):
    G = np.array([[0, 1, 2], [2, np.nan, 0], [1, 1, 1], [0, 2, np.nan], [2, 0, 1]], dtype=float)
    prefix = str(tmp_path / "c")
    plink_tools.write_bed(prefix, G)
    np.testing.assert_array_equal(_read_raw(prefix), G / 2)


def test_write_bed_rejects_fractions_as_counts(tmp_path):
    """0/0.5/1 input (what PlinkReader returns) must not be rounded into wrong counts."""
    G = np.array([[0.0, 0.5], [1.0, 0.5], [0.5, np.nan]])
    with pytest.raises(
        ValueError, match=r"pylmm's PlinkReader returns fractions of A2 \(0/0\.5/1\); pass 2\*G"
    ):
        plink_tools.write_bed(str(tmp_path / "x"), G)
    assert not os.path.exists(str(tmp_path / "x.bed"))


@pytest.mark.parametrize("bad", [0.4, 1.7, -1.0, 3.0, np.inf])
def test_write_bed_rejects_non_genotype_counts(tmp_path, bad):
    G = np.array([[0.0, 1.0], [2.0, bad]])
    with pytest.raises(ValueError, match="nothing is rounded"):
        plink_tools.write_bed(str(tmp_path / "x"), G)


def test_write_bed_fraction_coding_round_trip(tmp_path):
    F = np.array([[0.0, 0.5, 1.0], [1.0, np.nan, 0.0], [0.5, 0.5, 0.5], [0.0, 1.0, 0.0]])
    prefix = str(tmp_path / "f")
    plink_tools.write_bed(prefix, F, coding="fraction")
    np.testing.assert_array_equal(_read_raw(prefix), F)


def test_write_bed_fraction_coding_of_the_bundled_data(tmp_path):
    """Reading BED fractions and writing them back with coding='fraction' is lossless."""
    src = os.path.join(REPO, "data", "snps.132k.clean.noX")
    plink_tools.subset_bed(src, str(tmp_path / "s"), np.arange(40))
    F = _read_raw(str(tmp_path / "s"))
    assert np.isnan(F).any()
    plink_tools.write_bed(str(tmp_path / "w"), F, coding="fraction")
    with open(str(tmp_path / "s.bed"), "rb") as a, open(str(tmp_path / "w.bed"), "rb") as b:
        assert a.read() == b.read()


@pytest.mark.parametrize("bad", [0.25, 2.0])
def test_write_bed_fraction_coding_rejects_other_values(tmp_path, bad):
    with pytest.raises(ValueError, match=r"fractions must be 0, 0\.5 or 1"):
        plink_tools.write_bed(str(tmp_path / "x"), np.array([[0.0, bad]]), coding="fraction")


def test_write_bed_unknown_coding(tmp_path):
    with pytest.raises(ValueError, match="coding must be"):
        plink_tools.write_bed(str(tmp_path / "x"), np.zeros((2, 2)), coding="dosage")


# -- migrate.py -----------------------------------------------------------------

MIGRATE_REAL = os.path.join(REPO, "skills", "migrate-from-old-pylmm", "scripts", "migrate.py")
MIGRATE_LINK = os.path.join(
    REPO, ".claude", "skills", "migrate-from-old-pylmm", "scripts", "migrate.py"
)


def _migrate(script: str, *args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, script, *args], capture_output=True, text=True, check=True, cwd=cwd
    )


@pytest.mark.parametrize("script", [MIGRATE_REAL, MIGRATE_LINK], ids=["skills", ".claude-symlink"])
def test_migrate_runs_through_either_path(script, tmp_path):
    """The maps are found from the real folder and through the .claude/skills symlink, from any cwd."""
    old = tmp_path / "old.py"
    old.write_text("from pylmm import lmm\nK = lmm.calculateKinship(W)\n")
    p = _migrate(script, str(old), cwd=str(tmp_path))
    assert "lmm.calculate_kinship(W)" in p.stdout
    p = _migrate(
        script, "--command", "pylmmGWAS --bfile x --kfile x.kin out.txt", cwd=str(tmp_path)
    )
    assert p.stdout.strip() == "pylmm gwas --bfile x --kinship x.kin -o out.txt"


def test_migrate_notes(tmp_path):
    old = tmp_path / "old.py"
    old.write_text("import numpy as np\nh = L.getMax(H, X, REML=True)\nkeep = True - np.isnan(Y)\n")
    p = _migrate(MIGRATE_REAL, str(old))
    assert "L._get_max(H, X, reml=True)" in p.stdout
    assert "old.py:2: getMax is private now (_get_max)" in p.stderr
    assert "old.py:3: Python 2 idiom True - np.isnan(...)" in p.stderr
    assert "use ~np.isnan(...)" in p.stderr
    p = _migrate(MIGRATE_REAL, "--command", "pylmmGWAS --bfile x --kfile x.kin --refit out.txt")
    assert p.stdout.strip() == "pylmm gwas --bfile x --kinship x.kin --refit -o out.txt"
    assert "--refit: each SNP's fit is now by REML" in p.stderr


# -- summarize_gwas.py ----------------------------------------------------------

sys.path.insert(0, os.path.join(REPO, "skills", "interpret-gwas-results", "scripts"))
import summarize_gwas  # noqa: E402

SUMMARIZE = os.path.join(REPO, "skills", "interpret-gwas-results", "scripts", "summarize_gwas.py")


def _res(p, t):
    p, t = np.asarray(p, dtype=float), np.asarray(t, dtype=float)
    return {
        "SNP_ID": np.array(["s%d" % j for j in range(len(p))]),
        "BETA": t * 0.1,
        "BETA_SD": np.full(len(p), 0.1),
        "F_STAT": t,
        "P_VALUE": p,
    }


def test_rank_breaks_p_zero_ties_by_abs_t():
    res = _res([0.5, 0.0, np.nan, 0.0, 1e-300, 0.0], [0.7, 40.0, np.nan, -90.0, 37.0, 60.0])
    assert list(summarize_gwas.rank(res)) == [3, 5, 1, 4, 0]


def test_clump_by_snp_index_and_by_position():
    p = [1e-3, 1e-9, 1e-8, 0.5, 0.2, 1e-6, 1e-7, 0.9]
    res = _res(p, -np.log10(p))
    loci = summarize_gwas.clump(res, snps=2, p_max=1e-4)
    assert [lo["lead"] for lo in loci] == [1, 6]
    assert list(loci[0]["members"]) == [0, 1, 2, 3]
    assert list(loci[1]["members"]) == [4, 5, 6, 7]
    bim = {
        "chrom": np.array(["1", "1", "1", "2", "2", "2", "2", "2"]),
        "pos": np.array([100, 5000, 9000, 100, 200, 300, 900_000, 950_000]),
        "a1": np.array(["A"] * 8),
        "a2": np.array(["G"] * 8),
    }
    loci = summarize_gwas.clump(res, bim, kb=50)
    leads = [lo["lead"] for lo in loci]
    assert leads[:3] == [1, 6, 5]  # SNP 2 joins SNP 1's locus; chr 2 splits at 300 vs 900 kb
    assert list(loci[0]["members"]) == [0, 1, 2]
    s = summarize_gwas.summarize(res, bim, top=2, loci=loci)
    assert s["n_loci"] == len(loci)
    assert s["top"][0]["start"] == 100 and s["top"][0]["end"] == 9000
    with pytest.raises(ValueError, match=r"needs the \.bim positions"):
        summarize_gwas.clump(res, kb=50)


def test_dosage_sd_matches_pylmm_standardisation(tmp_path):
    src = os.path.join(REPO, "data", "snps.132k.clean.noX")
    plink_tools.subset_bed(src, str(tmp_path / "s"), np.arange(25))
    F = _read_raw(str(tmp_path / "s"))
    got = summarize_gwas.dosage_sd(str(tmp_path / "s"))
    np.testing.assert_allclose(got, np.nanstd(2 * F, axis=0))


def test_summarize_cli_per_allele_clump_plot(tmp_path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(5)
    G = rng.binomial(2, 0.3, size=(60, 30)).astype(float)
    prefix = str(tmp_path / "g")
    plink_tools.write_bed(prefix, G, chrom=["1"] * 15 + ["2"] * 15, pos=np.arange(1, 31) * 1000)
    p = np.full(30, 0.5)
    p[[4, 20]] = [0.0, 1e-12]
    t = np.zeros(30)
    t[[4, 20]] = [50.0, 9.0]
    res = tmp_path / "r.tsv"
    with open(res, "w") as f:
        f.write("\t".join(summarize_gwas.COLUMNS) + "\n")
        for j in range(30):
            f.write("snp%d\t%.17g\t0.1\t%.17g\t%.17g\n" % (j + 1, t[j] * 0.1, t[j], p[j]))
    png = str(tmp_path / "m.png")
    args = [str(res), "--bfile", prefix, "--per-allele", "--clump-kb", "3", "--clump-p", "1e-6"]
    out = subprocess.run(
        [sys.executable, SUMMARIZE, *args, "--plot", png],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "loci (+-3 kb windows, leads with p < 1e-06): 2" in out
    assert "snp5 chr1:5000 A1=A A2=G" in out
    assert "per A1 copy" in out
    assert os.path.getsize(png) > 0
    bad = subprocess.run(
        [sys.executable, SUMMARIZE, str(res), "--per-allele"], capture_output=True, text=True
    )
    assert bad.returncode == 2
    assert "--per-allele needs --bfile" in bad.stderr
    none = subprocess.run(
        [sys.executable, SUMMARIZE, str(res), "--clump-snps", "2", "--clump-p", "0"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "loci (+-2 SNPs windows, leads with p < 0): 0" in none
    assert "none: no SNP has p < 0 (--clump-p)" in none
