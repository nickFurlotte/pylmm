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
    assert len(rows) == 2 * (1 + len(bed_blocks.BLOCK_SIZES))
    assert all(r["identical"] for r in rows)
    table = bed_blocks.format_table(rows)
    assert table.startswith("| data |") and "iter_blocks(1024)" in table


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
