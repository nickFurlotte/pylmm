"""S7 numerics study (Phase 6.3): the quick grid runs, is JSON-able, and reports."""

import json

import pytest

from sims import runner


def test_s7_quick_cells_and_report(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    from sims.studies import s7_numerics as s7

    rows = []
    for params in runner.study_grid(s7, quick=True):
        out = s7.run_cell(params, runner.task_seed(s7.NAME, params, 0))
        json.dumps(out)
        rows.append({**params, **out})
    for r in rows:
        # REML: the default search reaches the oracle's maximum
        assert r["llshort_new_reml"] < 1e-6 and r["dh_new_reml"] < 1e-5
        assert 0.0 <= r["h_new_reml"] <= 1.0
    s7.report(pd.DataFrame(rows), str(tmp_path))
    text = (tmp_path / "summary.md").read_text()
    assert "## REML null fit" in text and "## Null calibration" in text
    assert (tmp_path / "summary.csv").exists()


def test_s7_hmdp_subset_fits_find_snp_110(capsys):
    """The real multi-modal case: of the subsets of the first 120 HMDP SNPs only
    SNP 110's fit changes, to the oracle's h = 0."""
    from sims.studies import s7_numerics as s7

    n, rows = s7.hmdp_subset_fits(120)
    assert n > 10 and [r["snp"] for r in rows] == [110]
    r = rows[0]
    assert r["peaks"] == 2 and r["h_new"] == r["h_oracle"] == 0.0 and r["ll_gain"] > 0.02
    s7.main(["120"])
    assert "| 110 | 58 | 2 |" in capsys.readouterr().out
