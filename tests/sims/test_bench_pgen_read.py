"""The Phase 6.2d benchmark sims/bench/pgen_read.py (PGEN vs BED read throughput)."""

import pytest

pytest.importorskip("pgenlib")

from sims.bench import pgen_read
from sims.bench.bed_blocks import write_synthetic_bed


def test_bench_pgen_read_small(tmp_path):
    bed = str(tmp_path / "s")
    write_synthetic_bed(bed, 9, 300, missing=0.1, seed=4)
    rows = pgen_read.bench_file("tiny", bed, str(tmp_path), repeats=1, seed=1)
    assert len(rows) == 2 * 4  # norm off/on x BED, PGEN hard calls (x2), PGEN dosages
    assert all(r["ok"] for r in rows)
    table = pgen_read.format_table(rows)
    assert table.startswith("| data |") and "PGEN dosages, dosage=True" in table
    assert "BED 903 bytes" in pgen_read.file_sizes(bed, str(tmp_path))  # 3 + 300 x 3 bytes


def test_bench_pgen_read_quick(capsys):
    assert pgen_read.main(["--quick", "--repeats", "1"]) == 0
    out = capsys.readouterr().out
    assert "dosages as written: yes" in out
