"""The Phase 6.2 benchmark sims/bench/bed_blocks.py (PlinkReader.iter_snps vs iter_blocks)."""

import filecmp
import os

import numpy as np
import pytest
from synthetic_bed import write_synthetic_bed

from sims.bench import bed_blocks


def test_bench_writer_matches_the_tests_writer(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    g_a = bed_blocks.write_synthetic_bed(a, 7, 20, missing=0.1, seed=9)
    g_b = write_synthetic_bed(b, 7, 20, missing=0.1, seed=9)
    assert np.array_equal(g_a, g_b, equal_nan=True)
    for ext in (".bed", ".bim", ".fam"):
        assert filecmp.cmp(a + ext, b + ext, shallow=False)


def test_bench_bed_blocks_small(tmp_path):
    base = str(tmp_path / "s")
    bed_blocks.write_synthetic_bed(base, 9, 300, seed=4)
    rows = bed_blocks.bench_file("tiny", base, repeats=1)
    # norm off: iter_snps + one row per block size; norm on: the same, plus the
    # per-SNP standardisation (before 6.1c) for each block size.
    assert len(rows) == 2 * (1 + len(bed_blocks.BLOCK_SIZES)) + len(bed_blocks.BLOCK_SIZES)
    assert all(r["identical"] for r in rows)
    assert sum("std_speedup" in r for r in rows) == len(bed_blocks.BLOCK_SIZES)
    table = bed_blocks.format_table(rows)
    assert table.startswith("| data |") and "iter_blocks(1024)" in table
    assert "iter_blocks(1024), per-SNP std" in table and "vs per-SNP std" in table


def test_bench_per_snp_standardisation_takes_the_fallback(monkeypatch):
    """Inside per_snp_standardisation(), _normalize_rows calls normalize_genotype per SNP."""
    from pylmm import genotypes

    calls = []
    real = genotypes.normalize_genotype
    monkeypatch.setattr(genotypes, "normalize_genotype", lambda g: calls.append(1) or real(g))
    R = np.random.default_rng(0).random((5, 9))
    with bed_blocks.per_snp_standardisation():
        genotypes._normalize_rows(R)
    assert len(calls) == 5
    assert type(genotypes._ROWS_MATCH) is dict  # restored


def test_bench_subset_copy_reads_the_first_snps(tmp_path):
    src, dst = str(tmp_path / "s"), str(tmp_path / "d")
    bed_blocks.write_synthetic_bed(src, 6, 50, seed=1)
    bed_blocks.copy_first_snps(src, dst, 17)
    from pylmm.plink_io import PlinkReader

    a = np.column_stack([g for g, _ in PlinkReader(src, norm_genotype=False).iter_snps()])
    b = np.column_stack([g for g, _ in PlinkReader(dst, norm_genotype=False).iter_snps()])
    assert np.array_equal(a[:, :17], b, equal_nan=True)
    assert os.path.getsize(dst + ".bed") == 3 + 17 * 2


@pytest.mark.slow
def test_bench_bed_blocks_quick(capsys):
    assert bed_blocks.main(["--quick", "--repeats", "1"]) == 0
    assert "bit-identical to iter_snps: yes" in capsys.readouterr().out


def test_bench_reports_the_probe(monkeypatch):
    from pylmm import genotypes

    monkeypatch.setattr(genotypes, "_ROWS_MATCH", {})
    assert bed_blocks.probe_report() == "block standardisation probe: not run"
    monkeypatch.setattr(genotypes, "_ROWS_MATCH", {1219: True, 7: False})
    assert bed_blocks.probe_report().endswith("n=7: NO, per-SNP fallback, n=1219: yes")
